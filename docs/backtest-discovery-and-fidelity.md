# Backtest discovery, the verdict, and the fidelity failure (2026-06-05)

The arc of validating whether SignalBolt's signals have edge — what we found, and
the critical caveat that reframes it.

## 1. The discovery tool

`engine/historical_backtest.py` replays each detector's ENTRY logic over years of
real Alpaca SIP daily bars, feeds entries into `replay_backtest` with a cost
model + regime proxy, and reports per (detector × regime) expectancy, edge-vs-SPY,
and walk-forward (70/30). The point: **backtest the frozen rules over history
(minutes, thousands of trades, all regimes) instead of waiting months for live
data** (slow + confounded by engine changes).

## 2. The verdict (6 years incl. 2022 bear, regime-gated, managed, 4156 trades)

Split by **alpha (beats SPY) vs beta (positive but doesn't)**:

| Tier | Detectors |
|---|---|
| **TRUE ALPHA** (positive net AND beats SPY, OOS-robust) | **BREAKOUT** (+1.04, edge +0.33, OOS +1.31), **TREND_MOMENTUM** (+0.99, +0.18, +1.65), **SWING_BREAKOUT** (+0.98, +0.23, +0.90) |
| Positive but BETA (make money, underperform SPY) | PULLBACK, COMPRESSION (thin), BREAKOUT_FORMING, TURNAROUND |
| Dead / hedge-only | ACCUM_FORMING, PEAK, BREAKDOWN, BREAKDOWN_FORMING, DISTRIB_FORMING |

The short detectors that looked best in 6 weeks of **live** data are regime-noise
losers over 6 years. Each time we added data, false hopes died and the boring
long-trend edges held.

## 3. Exits are second-order; entries are first-order

Tested regime gate, BE+trail, and a **MACD profit-lock** (the `signal_advisor`'s
"lock on momentum reversal" as an actual exit). The MACD-lock moved every
detector **<0.05%** and rescued none of the losers. **No exit tweak fixes a
bad-entry detector** — it can only help a trade that reaches profit. The 3 alpha
edges hold regardless of exit. This closes the recurring "maybe a smarter exit
saves the losers" hypothesis.

## 4. ⚠️ THE FIDELITY FAILURE — the verdict is about ARCHETYPES, not the live code

Empirical overlap (does the backtest predicate fire on the SAME signals the live
engine produced?):

| Detector | live signals | predicate agrees | agreement |
|---|---|---|---|
| BREAKOUT | 54 | 4 | **7%** |
| SWING_BREAKOUT | 14 | 2 | **14%** |
| BREAKDOWN | 57 | 10 | **18%** |

**7% means the backtest's `_breakout` fired on only 4 of 54 live BREAKOUT signals.
They test different things.** So "BREAKOUT has +1% alpha" is true for *a confirmed
daily-close 20-day-high breakout ARCHETYPE* — **NOT** the live BREAKOUT detector.

**Root cause:** the live detectors fire **intraday, on the breakout *level*** (per
tick via the stream), often on breakouts that **fade by the close**, plus looser
conditions (proximity not confirmed-close; SWING needs a retest the predicate
ignores; live momentum is vol-adjusted 12-1 vs the predicate's raw 126-day).

**Implication:** the proven edge lives in the **confirmed-daily-close** version;
the live detectors run a **looser intraday** version → may capture less edge, or
none. The backtest has **not** validated the shipping code.

## 5. The resolution — BO_POC

Rather than switch the live BREAKOUT (risky), `engine/bo_poc.py` is an **isolated
POC detector** whose live entry condition **is** the backtest predicate
(`historical_backtest._breakout` — confirmed daily close above the 20-day high on
volume). Because live == predicate, **fidelity is 100% by construction.** It fires
small-size, tagged `detector_source='BO_POC'`, regardless of regime (regime
tagged for later segmentation), once/day after the close. This lets us:
1. **Backtest it** — it IS the +1% archetype.
2. **Re-run fidelity** — BO_POC live signals should match the predicate ~100%.
3. **Test it live** — real forward signals of the *proven* archetype.

If BO_POC's live results track the backtest, we have a validated live edge and can
graduate the approach (confirmed-daily-close breakout) into the product. If not,
the backtest's look-ahead/cost assumptions need scrutiny.

## Honest standing summary

- **Useful + true:** breakout + cross-sectional momentum *styles* have real,
  modest, bear-tested edge.
- **NOT yet true:** that the live detectors have it (fidelity 7–18%).
- **The path:** validate the *faithful* archetype live (BO_POC), then rebuild the
  survivor detectors to match it — confirmed daily-close, not looser intraday.
