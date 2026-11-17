[![make.fun · btc-race-24h-20261117-0603](https://make.fun/badge/btc-race-24h-20261117-0603.svg)](https://make.fun/bounty/btc-race-24h-20261117-0603?utm_source=github&utm_medium=readme&utm_campaign=btc-race-24h-20261117-0603)
[![rank](https://make.fun/badge/btc-race-24h-20261117-0603/Revaaa28/makefun-btc-race-20261117-hybrid/rank.svg)](https://make.fun/bounty/btc-race-24h-20261117-0603/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=btc-race-24h-20261117-0603)
[![reward](https://make.fun/badge/btc-race-24h-20261117-0603/Revaaa28/makefun-btc-race-20261117-hybrid/reward.svg)](https://make.fun/bounty/btc-race-24h-20261117-0603?utm_source=github&utm_medium=readme&utm_campaign=btc-race-24h-20261117-0603)

# Adaptive Regime-Switching Hybrid — cbBTC 24h race

Entry for make.fun **btc-race-24h-20261117-0603** ("Race a BTC trading
bot live for 24 hours"), agent-fun/bars-replay/v1, Python 3.13,
standard library only.

## The design

One engine, three modules, switched by a regime classifier:

- **Regime.** Read on two axes. The fast axis is the EMA 9/21/50 stack
  with the EMA50 slope over 10 bars: stacked up with a rising slope is
  *bull*. The slow axis is the EMA50 slope over 120 bars in ATR units:
  at or under −2.5×ATR the market is a true *bear* downtrend. Anything
  neither bull nor bear is *sideways*. A new regime must persist 3 bars
  to take over (hysteresis).
- **Trend following** (bull only): confirmed breakouts — a close
  clearing the highest high of the prior 20 bars by 0.75×ATR — and
  pullbacks that hold the EMA21 rail and close back over EMA9 with RSI
  recovering. Entries additionally require the EMA21–EMA50 separation
  to exceed 3.2×ATR (see below). Stops and targets are ATR multiples
  (stop 3.5×ATR, target 7×ATR) with a trailing stop (arms at +2×ATR,
  trails 2×ATR). A trend position is closed if the backdrop turns bear.
- **Mean reversion** (sideways only): the gate is dislocation depth —
  a close at least **220 bps under its trailing 60-bar high** with
  RSI(14) ≤ 35, bought on the first uptick (never a falling bar). The
  Bollinger lower band (20, 2σ) is computed and available as a
  co-condition (`mr_bb_req`, off in the shipped configuration: the dip
  gate subsumes it on every validation window). Exit: −90 bps stop, no
  fixed target, 480-bar time stop.
- **Volatility intelligence.** ATR(14) sizes every trade and sets the
  trend exits; size is halved when ATR/price exceeds 45 bps and entries
  are skipped outright above 80 bps.

**Risk.** Each trade risks 1.0% of equity: the notional is the risk
budget divided by the stop distance, capped at the cash balance and
the weight room. Size therefore only ever shrinks as equity falls —
there is no recovery sizing anywhere. A drawdown brake halts new
entries at −10% from the peak equity (the internal limit is 20%;
the objective is maxDD < 25%). At most 10 entries per window, a
cooldown after every exit (longer after a loser), no entries after
bar 1330. Long-only, one position at a time: a bear regime is flat.

## What the data did to this design

The spec asked for a hybrid, and the hybrid was built and measured —
all numbers below are the kit's own fill model (4 bps fee, 2 bps
slippage, constant-product impact), run on the published dev window
and on eight real Binance BTCUSDT days (Nov 9–16) at the dev window's
median pool depth.

- **Trend entries at ordinary strength are a bloodbath.** With the
  EMA-separation bar at 0.8×ATR the trend module took 51 trades across
  the 9 windows and lost $188 (14% win rate); every window negative.
  Raising the bar monotonically reduced the damage; at 3.2×ATR the
  module stood down on all 9 validation windows. It is shipped armed
  at that level — on a synthetic +0.35%/bar trend tape it takes 10
  trades and makes +$198.56 with 0.15% maxDD, so the machinery is
  live for a genuinely trending race day; it simply refuses the
  chop that fills the validation set.
- **Shallow mean reversion bleeds; deep dislocations pay.** Band-tag
  entries lost on every window. The 220 bps dip gate fired exactly 3
  times in 9 windows and won all 3.
- **The slow-axis bear veto is what makes the dip gate safe.** The
  earlier ungated dip buyer lost −$12.9 and −$11.3 on Nov 13–14
  (sustained downtrend days). With the veto this bot sits both days
  out entirely ($0.00).

## Validation (shipped configuration)

| window | net PnL | trades | maxDD |
|---|---|---|---|
| dev (official) | **+$12.58** | 1 | 0.62% |
| Nov 9 | $0.00 | 0 | 0.00% |
| Nov 10 | +$5.75 | 1 | 1.03% |
| Nov 11 | $0.00 | 0 | 0.00% |
| Nov 12 | $0.00 | 0 | 0.00% |
| Nov 13 | $0.00 | 0 | 0.00% |
| Nov 14 | $0.00 | 0 | 0.00% |
| Nov 15 | $0.00 | 0 | 0.00% |
| Nov 16 | +$12.67 | 1 | 0.61% |
| **battery total** | **+$18.43** | | worst window $0.00 |

9-window total **+$31.01**. Determinism: two end-to-end stdio runs
produce byte-identical action logs, matching the in-process result to
the micro-USDC. Edge cases: flat tape, a −0.35%/bar crash grind and a
−3% cliff all produce zero trades and zero drawdown; warm-up batches
and batches missing the symbol are handled.

**Limitations, honestly.** The entire positive record rests on 3
trades. 3/3 wins is a small sample, not a law; the dev gate itself
rests on a single trade (+$12.58). The configuration sits on a
measured plateau (stop 75–105 bps, dip 200–220, bear veto 2.0–3.0 all
score within ~$2 of the shipped point), but a dip gate at 240 bps
fires nothing at all — the edge is real and narrow. On a race day
with no deep dip and no strong trend, this bot finishes at $0.00,
which loses to any bot that made a dollar. That is the chosen risk:
it cannot bleed, and it only plays the two situations it has
evidence for.

## Layout

- `main.py` — the bot (the `Bot` class; `PARAMS` at the top is the
  full configuration).
- `vendor/makefun_bars.py` — the kit's protocol shim, byte-identical
  to the kit SDK (sha256 80452b00…).

## License

MIT — see `LICENSE`.
