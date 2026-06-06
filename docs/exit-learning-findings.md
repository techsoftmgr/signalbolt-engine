# Exit-learning stack & findings

How SignalBolt learns SL/TP + exit timing, and the findings so far. All of this
is **advisory / read-only** — nothing here changes live firing or exits until
explicitly enforced.

## The stack (built 2026-06-05)

| Layer | Module | What it does | Status |
|---|---|---|---|
| 1 | `engine/replay_backtest.py` | Replays a closed signal on its **actual Alpaca SIP forward bars** under a candidate exit policy (stop / target / trail / breakeven / time-stop). Pure, no-look-ahead, same-bar stop-wins-ties. `run_param_set()` A/Bs a policy vs as-traded. | ✅ built |
| 2 | `engine/exit_optimizer.py` | Per-detector walk-forward search over candidate exit policies; gated by sample floor + ≥0.10%/trade out-of-sample improvement vs as-traded. | ✅ built (advisory) |
| 3 | `engine/regime_exit.py` | "Lock profit / cut when the regime flips against the position" decision brain. `ENFORCE=False`. | ✅ brain built, **not wired** |
| — | `engine/detector_policy.py` | Recommended size multiplier per (detector × regime bucket) from realized expectancy + alpha. | ✅ built (advisory) |

Endpoints: `POST /admin/replay-backtest`, `GET /admin/exit-optimizer?detector=`,
`GET /admin/detector-policy`.

The old `engine/backtester.py` was switched off yfinance synthetic data onto
**Alpaca SIP** (it was too weak to tune real-money rules).

## Finding #1 — Momentum give-back (2026-06-05, n=6, NOISE — do not act)

Replayed the 6 closed `TREND_MOMENTUM` trades on real SIP daily bars under
several give-back caps (a % trailing stop as a proxy for the chandelier).

As-traded chandelier: **−3.12%/trade, −18.8% total, 1/6 win.**

| Exit policy (replayed) | exp/trade | total | win |
|---|---|---|---|
| Wide stop, no trail (ride) | −2.99% | −17.9% | 1/6 |
| Give-back cap 20% (~as-traded) | −0.92% | −5.5% | 1/6 |
| Give-back cap 15% | +0.75% | +4.5% | 1/6 |
| **Give-back cap 10% (tight)** | **+3.46%** | **+20.7%** | 2/6 |
| Breakeven@+20% + 12% trail | +2.13% | +12.8% | 1/6 |

Per-trade (give-back cap 15%): **MRVL** +25.9% → **+41.5%** (locked more of its
+56.5% peak); **MU** −12.5% → **−3.6%** (caught the round-trip from +11.5%);
TXN/CRWD/SMH unchanged (never went green → an **entry** problem, not exit).

### The conclusion that matters: exit tuning is REGIME-DEPENDENT

This result **contradicts** the earlier larger backtests (253–430 trades) where
tightening the trail **hurt** by clipping runners. That is **not** a
contradiction — it is regime-dependence:

- **Trending / bull regime** → ride wide; a tight cap clips the fat-tail runners.
- **Risk-off / topping regime** → snap the cap tight; everything peaks and gives
  back, so locking early wins.

The 6 momentum trades above were a risk-off/topping cluster, so the tight cap
shone. In a bull sample it would bleed. **A single global give-back setting is
therefore wrong; the correct form is regime-conditional** — exactly what
`regime_exit.py` (Layer 3) is built to switch. This is the Layer-3 thesis
validated in miniature.

**n=6 with one dominant winner (MRVL) is pure noise — do not change the
chandelier.** The real tuning is making the give-back cap regime-aware, which
needs the per-regime cells to fill (hundreds of trades). Re-run the fuller
experiment once `TREND_MOMENTUM` has ≥~30 closed per regime.

## Data gate (applies to all exit/SL-TP learning)

Trustworthy learned parameters need ~hundreds of closed signals **per
(detector × regime) cell**; today there are ~tens, mostly single-regime. The
system is built; the parameters are not yet trustworthy. Sequence: replay engine
(done) → run advisory → enforce when the cells mature **and** the user approves.
