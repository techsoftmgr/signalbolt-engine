# Trade Quality Analysis — Friday session

**Date written:** 2026-05-25
**Data window:** Friday 2026-05-22 (full session, 93 signals)
**Status:** Diagnosis only. No code fix shipped.

---

## TL;DR

Friday produced 93 signals at a **41.6% overall win rate**. Drilling in, the
scorer is not separating winners from losers — A-grade trades actually
underperformed B+ — and the stop-loss logic ignores ticker volatility,
producing uniform SL widths that get blown through on volatile names.

Until noon the system behaved fine. The damage was concentrated in (a) the
first 13 minutes of the session and (b) volatile names where the fixed-%
SL was too tight.

---

## Findings

### 1. Scorer saturates above 70 — grade is anti-predictive at the top

| Grade | Win rate |
|-------|----------|
| A     | 40.9%    |
| B+    | 45.0%    |
| B     | ~41%     |

A-grade signals are *worse* than B+. The score→grade mapping rewards
higher numeric scores, but the underlying signal quality flatlines once
the score crosses ~70. Either the inputs at the top end are noisy or the
weights compound wrong.

### 2. L8 gamma is a constant

Every signal Friday had `L8 = 65`. Whatever this dimension is supposed
to measure, it's contributing zero discrimination. It either needs to
actually vary, or be removed from the scorer so it stops inflating
totals uniformly.

### 3. `quant_bonus` is capped at 10 — everyone hits the cap

Same story as L8: a "bonus" that every signal receives isn't a bonus,
it's a constant offset. Cap needs to be raised (so only the best
signals reach it) or the bonus needs a steeper curve.

### 4. Stop-loss width is uniform regardless of ticker

SL widths Friday clustered at **0.79–0.84%** across the board. NVDA
(daily ATR ~3%) and KO (daily ATR ~0.8%) got the same SL distance.
Result: NVDA stops are inside its normal noise band → premature exits.

### 5. Stop-loss slippage on fills

6 Friday cases where the actual fill was **25–60% past** the intended
stop price. Root cause is some combo of:
- Market orders into thin pre-/post-open liquidity
- Gap-throughs on news catalysts
- No slippage guard or limit-stop on the exit

### 6. Opening-chop concentration

**11 of 93 signals** fired in the first **13 minutes** of the session.
This window had disproportionately bad outcomes — opening auction
imbalance + spread chaos doesn't suit a momentum entry.

---

## Proposed fixes (in order of impact ÷ effort)

### Tier 1 — quick, ships in a day

1. **10:00 AM ET entry cutoff** — block new signals during 09:30–10:00.
   Cuts the opening-chop losses outright.
2. **Pause scalping strategy** — until scorer is recalibrated, this
   strategy is the loudest source of bad fills. Keep day_trade and
   swing alive.

### Tier 2 — deeper, multi-day

3. **ATR-based SL widths** — replace fixed % with `k × ATR(14)` per
   ticker. Tune `k` per strategy (scalp ~0.5, day ~1.0, swing ~1.5).
4. **Score → grade recalibration** — refit the grade boundaries using
   actual realized win rates as the target. Either change the
   thresholds or shrink the score range that maps to A.
5. **Fix L8 gamma** — either make it actually vary across signals, or
   drop it from the scoring sum.
6. **Widen `quant_bonus` cap** — raise to 25 or replace flat cap with
   a curve so only the top decile reaches max.

### Tier 3 — infrastructure for future tuning

7. **Slippage guard on stop fills** — use stop-limit with a max
   slippage envelope, or cancel-and-replace if quote moves past a
   threshold before fill.
8. **Per-strategy backtest harness** — so future scorer tweaks are
   validated against historical signals before going live.

---

## What's NOT in this analysis

- Did not split by sector / market cap / IV regime. Sample was 93
  signals over one day; subsamples would be too small to trust.
- Did not check whether A-grade underperformance is a Friday artifact
  or a persistent pattern — needs a multi-week pull.
- Did not measure whether the half-day / Memorial Day gating fix
  (shipped separately) affects any of these numbers.

---

## Next decisions for the operator

- **Ship Tier 1 before Tuesday open?** (10 AM cutoff + pause scalping)
- **Schedule Tier 2 work?** ATR-based SLs is the single highest-impact
  item and probably 1–2 days of focused work + a validation pass.
- **Pull a 4-week sample** before recalibrating the scorer so the new
  boundaries aren't overfit to one bad Friday.
