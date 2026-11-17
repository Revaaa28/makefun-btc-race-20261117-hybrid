"""Adaptive Regime-Switching Hybrid Trading Engine for the make.fun cbBTC race.

Entry for btc-race-24h-20261117-0603 (agent-fun/bars-replay/v1).

The engine watches one minute tape and answers one question per bar:
which regime is this market in, and what — if anything — is worth doing
in it? Three indicator families vote:

* Trend: EMA 9 / 21 / 50 stacked with a rising slow EMA marks a bullish
  regime (the mirror image marks a bearish one). In a bullish regime the
  bot buys confirmed breakouts over the recent high and pullbacks that
  hold the mid EMA — but only when the trend's EMA separation is
  exceptional (see "What the data did to this design" in the README:
  at ordinary strengths these entries lose after costs on this tape,
  so the activation bar is set where the validation set stops
  bleeding).
* Mean reversion: RSI(14) plus Bollinger Bands (20, 2 sigma) frame
  oversold, but the entry gate is dislocation depth: a close at least
  220 bps under its trailing 60-bar high with RSI oversold, bought on
  the first uptick. It is armed only when the backdrop is sideways —
  and the backdrop is read on a slow axis (the 120-bar EMA50 slope),
  because a sharp dip inside a flat-or-rising tape is a dislocation
  in a sideways market, not a bear market. A decisively falling slow
  axis is a bear regime: flat, no entries, capital preserved.
* Volatility: ATR(14) sets the trend module's stops and targets in
  price terms and sizes every position, and exposure is cut (or
  entries skipped) when ATR runs extreme.

Risk is sized, never guessed: each trade risks a fixed fraction of
equity (0.5-1%), the notional follows from the ATR stop distance, size
only ever shrinks as equity falls, a drawdown brake halts new entries
well inside the 20% internal limit, and a hard cap on entries per
window keeps the bot out of churn. The engine is long-only (the race
has no shorts): a bearish regime means flat, and a position whose
regime turns bearish is closed.

Everything is a deterministic function of the bars seen: no clock, no
randomness, no network, standard library only.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))

import makefun_bars  # noqa: E402

SYMBOL = "CBBTC"
BPS = 10000

PARAMS = {
    # --- indicators ---
    "ema_fast": 9,          # trend stack, fast leg
    "ema_mid": 21,          # trend stack, mid leg / pullback rail
    "ema_slow": 50,         # trend stack, slow leg
    "rsi_len": 14,
    "bb_win": 20,           # Bollinger lookback
    "bb_k": 2.0,            # Bollinger width in sigmas
    "atr_len": 14,
    # --- regime ---
    "slope_win": 10,        # bars over which the fast slow-EMA slope reads
    "slow_win": 120,        # bars over which the backdrop slope reads
    "bear_slow_atr": 2.5,   # backdrop slope (x ATR) at/under which the
                            # market is a true downtrend: flat, no entries
    "regime_hold": 3,       # bars a new regime must persist to take over
    # --- trend entries ---
    "trend_on": 1,          # 0: trend module detects but never enters
    "brk_win": 20,          # breakout reference: highest high of N bars
    "brk_margin_atr": 0.75,  # breakout close must clear the high by this
    "trend_min_spread_atr": 3.2,  # EMA21-EMA50 separation a trend entry needs
    "pb_band_atr": 0.35,    # pullback counts within this x ATR of EMA21
    "pb_fast": 1,           # pullback must also close back over EMA9
    # --- mean-reversion entries (sideways regime only) ---
    "mr_win": 60,           # trailing-high window the dip is measured on
    "mr_dip_bps": 220,      # close at least this far under the trailing
                            # high: only deep dislocations are bought
    "mr_rsi": 35.0,         # RSI at or under this at the dip
    "mr_bb_req": 0,         # 1: also require the close at the lower band
    "mr_touch_atr": 0.15,   # close may sit this x ATR above the lower band
    # --- exits: trend in ATR multiples, mean reversion in bps ---
    "trend_stop_atr": 3.5,
    "trend_tp_atr": 7.0,
    "trail_arm_atr": 2.0,   # trailing stop arms this far into profit
    "trail_atr": 2.0,       # ... and trails this far behind the peak close
    "trend_hold": 300,      # time stop, bars
    "mr_sl_bps": 90,        # mean-reversion stop, bps under the entry
    "mr_tp_bps": 0,         # 0: no fixed target, the time stop exits
    "mr_hold": 480,         # time stop, bars
    # --- risk ---
    "risk_pct": 1.0,        # percent of equity risked per trade
    "size_cap_pct": 100,    # never stake more than this % of cash at once
    "vol_extreme_bps": 45,  # ATR/price above this halves size...
    "vol_scale": 0.5,
    "vol_skip_bps": 80,     # ...and above this entries are skipped
    "dd_halt_pct": 10.0,    # drawdown from peak equity that halts entries
    "max_entries": 10,      # entries per window, hard cap
    "cooldown": 10,         # bars to sit out after any exit
    "loss_cooldown": 25,    # bars to sit out after a losing exit
    "late_cutoff": 1330,    # no entries after this bar of the window
    "min_bars": 60,         # history required before the first entry
    "buy_slip_bps": 150,    # a fill far above the close means it ran away
    "sell_slip_bps": 2000,  # an exit must never be refused when it matters
}


class Bot:
    """One bot instance per hello; feed it batches, it answers orders."""

    def __init__(self, hello):
        p = PARAMS
        sym = hello["symbols"][0]
        self.qty_scale = sym["qty_scale"]
        self.price_scale = sym["price_scale"]
        self.quote_scale = hello["capital"]["quote_scale"]
        self.min_notional = int(hello["fees"]["min_notional_quote_units"])
        self.start_equity = int(hello["capital"]["quote_units"])
        # tape
        self.opens = []
        self.highs = []
        self.lows = []
        self.closes = []
        # EMA legs (floats, SMA-seeded)
        self._ema_seeds = {p["ema_fast"]: [], p["ema_mid"]: [], p["ema_slow"]: []}
        self.ema = {p["ema_fast"]: [], p["ema_mid"]: [], p["ema_slow"]: []}
        # RSI (Wilder)
        self.rsi = []
        self._rsi_sg = 0.0
        self._rsi_sl = 0.0
        self._rsi_ag = None
        self._rsi_al = None
        # ATR (Wilder, price units)
        self.atr = []
        self._atr_seed = []
        self._atr_val = None
        # regime, with persistence hysteresis
        self.regime = "none"        # none | bull | bear | range
        self._cand = "none"
        self._cand_n = 0
        # order / position state
        self.awaiting = None        # "buy" / "sell" while an order is open
        self.seq = 0                # client-id counter
        self.entries = 0            # filled buys this window
        self.cool_to = -1           # first bar a new entry is allowed
        self.peak_equity = self.start_equity
        self.halted = False         # drawdown brake
        # open-position trade plan (set at the buy decision)
        self.zone_sent = None       # mode of the unresolved buy
        self.mode_open = None       # mode of the open position
        self.plan_stop = 0          # absolute stop price of the open trade
        self.plan_tp = 0            # absolute target price of the open trade
        self.plan_hold = 0          # time-stop bar count of the open trade
        self.plan_peak = 0          # best close since entry (trailing)
        self.open_bar = -1          # closes-index of the entry decision
        self._pend_stop = 0
        self._pend_tp = 0
        self._pend_hold = 0
        self._last_exit_pnl = None  # sign of the last closed trade, if known

    # ------------------------------------------------------------------
    # indicators
    # ------------------------------------------------------------------
    def _feed(self, o, h, l, c):
        p = PARAMS
        self.opens.append(o)
        self.highs.append(h)
        self.lows.append(l)
        self.closes.append(c)
        i = len(self.closes) - 1
        for n in (p["ema_fast"], p["ema_mid"], p["ema_slow"]):
            k = 2.0 / (n + 1)
            seed = self._ema_seeds[n]
            if len(seed) < n:
                seed.append(float(c))
                self.ema[n].append(sum(seed) / n if len(seed) == n else None)
            else:
                prev = self.ema[n][i - 1]
                self.ema[n].append(prev + k * (c - prev))
        self._feed_rsi(c, i)
        self._feed_atr(h, l, c, i)
        self._feed_regime(i)

    def _feed_rsi(self, c, i):
        n = PARAMS["rsi_len"]
        if i == 0:
            self.rsi.append(None)
            return
        delta = self.closes[i] - self.closes[i - 1]
        gain = float(delta) if delta > 0 else 0.0
        loss = float(-delta) if delta < 0 else 0.0
        if i <= n:
            self._rsi_sg += gain
            self._rsi_sl += loss
            if i < n:
                self.rsi.append(None)
                return
            self._rsi_ag = self._rsi_sg / n
            self._rsi_al = self._rsi_sl / n
        else:
            self._rsi_ag = (self._rsi_ag * (n - 1) + gain) / n
            self._rsi_al = (self._rsi_al * (n - 1) + loss) / n
        if self._rsi_al == 0.0:
            self.rsi.append(100.0 if self._rsi_ag > 0 else 50.0)
        else:
            rs = self._rsi_ag / self._rsi_al
            self.rsi.append(100.0 - 100.0 / (1.0 + rs))

    def _feed_atr(self, h, l, c, i):
        n = PARAMS["atr_len"]
        if i == 0:
            self.atr.append(None)
            return
        pc = self.closes[i - 1]
        tr = float(max(h - l, abs(h - pc), abs(l - pc)))
        if i <= n:
            self._atr_seed.append(tr)
            if i < n:
                self.atr.append(None)
                return
            self._atr_val = sum(self._atr_seed) / n
        else:
            self._atr_val = (self._atr_val * (n - 1) + tr) / n
        self.atr.append(self._atr_val)

    def _bb(self, i):
        """(mid, lower) of the Bollinger band at bar i, or (None, None)."""
        w = PARAMS["bb_win"]
        if i + 1 < w:
            return None, None
        win = self.closes[i - w + 1:i + 1]
        mid = sum(win) / w
        var = sum((x - mid) ** 2 for x in win) / w
        lower = mid - PARAMS["bb_k"] * (var ** 0.5)
        return mid, lower

    def _regime_raw(self, i):
        """Classify the backdrop on the slow axis.

        The fast EMA stack alone calls every sharp dip a bear market —
        but a 200 bps drop inside a tape whose 120-bar drift is flat or
        rising is a dislocation in a sideways market, which is exactly
        what mean reversion exists for. So: a true bear regime needs
        the slow axis (EMA50 slope over slow_win bars) decisively down;
        a bull regime needs the fast stack up with a rising slow EMA;
        everything else is sideways.
        """
        p = PARAMS
        ef = self.ema[p["ema_fast"]][i]
        em = self.ema[p["ema_mid"]][i]
        es = self.ema[p["ema_slow"]][i]
        atr = self.atr[i]
        if ef is None or em is None or es is None or atr is None or atr <= 0:
            return "none"
        j = i - p["slow_win"]
        if j < 0 or self.ema[p["ema_slow"]][j] is None:
            return "none"
        slow_slope = es - self.ema[p["ema_slow"]][j]
        if slow_slope <= -p["bear_slow_atr"] * atr:
            return "bear"
        k = i - p["slope_win"]
        if k >= 0 and self.ema[p["ema_slow"]][k] is not None:
            slope = es - self.ema[p["ema_slow"]][k]
            if ef > em > es and slope > 0:
                return "bull"
        return "range"

    def _feed_regime(self, i):
        raw = self._regime_raw(i)
        if raw == self.regime:
            self._cand = raw
            self._cand_n = 0
            return
        if raw == self._cand:
            self._cand_n += 1
        else:
            self._cand = raw
            self._cand_n = 1
        if self._cand_n >= PARAMS["regime_hold"]:
            self.regime = raw
            self._cand_n = 0

    # ------------------------------------------------------------------
    # bookkeeping
    # ------------------------------------------------------------------
    def _basis(self, pos):
        """Price level at which the position's value equals its cost."""
        return (pos.cost * 10 ** (self.qty_scale + self.price_scale)
                // (pos.qty * 10 ** self.quote_scale))

    def _cid(self, tag):
        self.seq += 1
        return "%s%d" % (tag, self.seq)

    def _absorb_fills(self, fills, state):
        for f in fills:
            if self.awaiting == "buy" and f["side"] == "buy":
                if f.get("status") == "filled":
                    self.entries += 1
                    self.mode_open = self.zone_sent
                    self.plan_stop = self._pend_stop
                    self.plan_tp = self._pend_tp
                    self.plan_hold = self._pend_hold
                    self.plan_peak = self.closes[-1]
                    self.open_bar = len(self.closes) - 1
                self.awaiting = None
                self.zone_sent = None
            elif self.awaiting == "sell" and f["side"] == "sell":
                if f.get("status") == "filled":
                    cool = PARAMS["loss_cooldown"] if self._last_exit_pnl is not None \
                        and self._last_exit_pnl < 0 else PARAMS["cooldown"]
                    self.cool_to = len(self.closes) - 1 + cool
                    self.mode_open = None
                self.awaiting = None

    # ------------------------------------------------------------------
    # sizing
    # ------------------------------------------------------------------
    def _size(self, state, stop_dist_price, ref_price):
        """Notional for a trade whose stop sits stop_dist_price under ref.

        Risk is PARAMS['risk_pct'] of equity; the notional that loses
        exactly that at the stop follows. Capped by cash share and the
        weight room; halved in extreme volatility. None if unusable.
        """
        p = PARAMS
        if stop_dist_price <= 0 or ref_price <= 0:
            return None
        risk_units = state.equity * p["risk_pct"] / 100.0
        notional = risk_units * ref_price / stop_dist_price
        cap = state.cash * p["size_cap_pct"] / 100.0
        room = state.weight_room(SYMBOL)
        notional = min(notional, cap, float(room))
        atr_bps = self.atr[-1] * BPS / ref_price if self.atr[-1] else 0.0
        if atr_bps > p["vol_extreme_bps"]:
            notional *= p["vol_scale"]
        notional = int(notional)
        if notional < self.min_notional:
            return None
        return notional

    # ------------------------------------------------------------------
    # entries
    # ------------------------------------------------------------------
    def _buy(self, state, mode, stop, tp, hold):
        notional = self._size(state, self.closes[-1] - stop, self.closes[-1])
        if notional is None:
            return None
        self.awaiting = "buy"
        self.zone_sent = mode
        self._pend_stop = stop
        self._pend_tp = tp
        self._pend_hold = hold
        return [{"side": "buy", "client_id": self._cid("b"),
                 "symbol": SYMBOL, "notional_quote_units": notional,
                 "max_slippage_bps": PARAMS["buy_slip_bps"]}]

    def _trend_entry(self, state, i, close):
        p = PARAMS
        atr = self.atr[i]
        # The trend must have real separation, not a fresh tangle-cross.
        spread = self.ema[p["ema_mid"]][i] - self.ema[p["ema_slow"]][i]
        if spread < p["trend_min_spread_atr"] * atr:
            return None
        # Confirmed breakout: this close clears the highest high of the
        # prior brk_win bars by a margin (the decision is the
        # confirmation; the fill comes at the next open).
        if i >= p["brk_win"]:
            ref = max(self.highs[i - p["brk_win"]:i])
            if close > ref + p["brk_margin_atr"] * atr \
                    and close > self.ema[p["ema_fast"]][i]:
                stop = close - p["trend_stop_atr"] * atr
                tp = close + p["trend_tp_atr"] * atr
                return self._buy(state, "trend", int(stop), int(tp), p["trend_hold"])
        # Pullback: price comes back to the mid-EMA rail inside the
        # uptrend and this bar turns up off it — back over the fast EMA
        # with RSI recovering.
        em = self.ema[p["ema_mid"]][i]
        if (abs(close - em) <= p["pb_band_atr"] * atr
                and close > self.closes[i - 1]
                and (not p["pb_fast"] or close > self.ema[p["ema_fast"]][i])
                and self.rsi[i] is not None and self.rsi[i - 1] is not None
                and self.rsi[i] > self.rsi[i - 1]
                and close > self.ema[p["ema_slow"]][i]):
            stop = close - p["trend_stop_atr"] * atr
            tp = close + p["trend_tp_atr"] * atr
            return self._buy(state, "trend", int(stop), int(tp), p["trend_hold"])
        return None

    def _mr_entry(self, state, i, close):
        """Deep-dislocation reversion, sideways regime only.

        The tape evidence on this universe is blunt: shallow band tags
        bleed fees, but a close 200+ bps under its trailing high with
        RSI oversold, bought on the first uptick, is a real recovery
        trade. The Bollinger tag is available as a co-condition
        (mr_bb_req) but the dip depth is the primary gate.
        """
        p = PARAMS
        if i < p["mr_win"] or self.rsi[i] is None:
            return None
        ref = max(self.highs[i - p["mr_win"]:i])
        dip_bps = (ref - close) * BPS // ref
        if dip_bps < p["mr_dip_bps"]:
            return None
        if self.rsi[i] > p["mr_rsi"]:
            return None
        if close <= self.closes[i - 1]:
            return None                     # still falling: no catch
        if p["mr_bb_req"]:
            _mid, lower = self._bb(i)
            if lower is None or close > lower + p["mr_touch_atr"] * self.atr[i]:
                return None
        stop = close * (BPS - p["mr_sl_bps"]) // BPS
        if p["mr_tp_bps"] > 0:
            tp = close * (BPS + p["mr_tp_bps"]) // BPS
        else:
            tp = close * 4                  # no fixed target: time stop
        return self._buy(state, "mr", int(stop), int(tp), p["mr_hold"])

    # ------------------------------------------------------------------
    # exits
    # ------------------------------------------------------------------
    def _exit_order(self, state, pos, i, close):
        p = PARAMS
        basis = self._basis(pos)
        if close > self.plan_peak:
            self.plan_peak = close
        stop = self.plan_stop
        if self.mode_open == "trend":
            arm = basis + p["trail_arm_atr"] * self.atr[self.open_bar]
            if self.plan_peak >= arm:
                trail = self.plan_peak - p["trail_atr"] * self.atr[i]
                if trail > stop:
                    stop = trail
        hit = close <= stop or close >= self.plan_tp
        stale = (i - self.open_bar) >= self.plan_hold
        # A trend trade dies with its regime; a reversion trade has its
        # own invalidation (the bps stop) and is left to it.
        regime_dead = self.regime == "bear" and self.mode_open == "trend"
        if hit or stale or regime_dead:
            # The trade's result sign (unrealised at decision) drives the
            # longer post-loss cooldown in _absorb_fills.
            self._last_exit_pnl = (close - basis) * pos.qty
            self.awaiting = "sell"
            return [{"side": "sell", "client_id": self._cid("s"),
                     "symbol": SYMBOL, "qty": pos.qty,
                     "max_slippage_bps": p["sell_slip_bps"]}]
        return None

    # ------------------------------------------------------------------
    # per-batch decision
    # ------------------------------------------------------------------
    def on_batch(self, state, batch):
        p = PARAMS
        for bar in batch["bars"]:
            if bar["symbol"] == SYMBOL:
                self._feed(int(bar["o"]), int(bar["h"]), int(bar["l"]), int(bar["c"]))
        self._absorb_fills(batch["fills"], state)
        if state.equity > self.peak_equity:
            self.peak_equity = state.equity
        if self.peak_equity > 0 and \
                state.equity * BPS < self.peak_equity * (BPS - int(p["dd_halt_pct"] * 100)):
            self.halted = True
        if state.warming_up or not self.closes:
            return None
        i = len(self.closes) - 1
        close = self.closes[i]
        pos = state.positions.get(SYMBOL)
        if pos is not None and self.awaiting is None:
            return self._exit_order(state, pos, i, close)
        if pos is not None or self.awaiting is not None:
            return None
        # flat: consider an entry
        if self.halted or self.entries >= p["max_entries"]:
            return None
        if i < p["min_bars"] or i < self.cool_to or i > p["late_cutoff"]:
            return None
        if self.atr[i] is None or self.atr[i] <= 0:
            return None
        if self.atr[i] * BPS / close > p["vol_skip_bps"]:
            return None                     # extreme volatility: stand aside
        if self.regime == "bull" and p["trend_on"]:
            return self._trend_entry(state, i, close)
        if self.regime == "range":
            return self._mr_entry(state, i, close)
        return None


def main():
    holder = {}

    def on_batch(state, batch):
        bot = holder.get("bot")
        if bot is None:
            bot = holder["bot"] = Bot(state.hello)
        return bot.on_batch(state, batch)

    makefun_bars.run(on_batch)


if __name__ == "__main__":
    main()
