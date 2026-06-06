# Entry-gate effectiveness — what the data says (2026-06-05)

The entry gate is a post-scorer, pre-SL/TP filter pipeline. The `gate_validator`
job replays every rejection forward to ask "would it have won?". Here is what
923 judged rejections actually show — and an important correction.

## Win-rate flatters it; P&L tells the real story

| Rejected signals | n | Avg P&L if taken |
|---|---|---|
| Would-have-WON | 317 | **+1.70%** |
| Would-have-LOST | 683 | **−0.71%** |
| **All rejects, taken** | 1,000 | **+0.05%/trade gross (≈ break-even)** |

By **win-rate** the gate looks great (68% of rejects would have lost). By
**P&L magnitude** it is roughly **break-even gross** — because the winners it
kills are *bigger* (+1.70%) than the losers it kills (−0.71%). That is the
fat-tail asymmetry: winners run further than losers.

**Conclusion: the entry gate is a COST / QUALITY filter, not a raw-alpha
engine.** Its real value is rejecting wide-spread, dead-tape, illiquid junk that
would bleed to spread + slippage — so net of realistic costs, rejecting is +EV.
But do not credit it with edge it doesn't have, and know that it *does* clip some
fat-tail runners.

## Per-gate net expectancy of its rejects (gross; lower = better gate)

| Gate | n | would-win % | net P&L of its rejects |
|---|---|---|---|
| regime alignment | 363 | 26% | **−0.21%** (best — kills −EV) |
| 5m trend | 12 | 25% | −0.20% |
| 15m trend | 26 | 31% | +0.07% |
| 5m MACD | 180 | 43% | +0.12% |
| dead tape | 252 | 42% | +0.22% |
| low volume | 230 | 43% | +0.25% |
| 1m reversal | 312 | 44% | +0.29% |
| **overextended** | 250 | **49%** | **+0.33%** (weakest — likely clips runners) |

The **trend + regime** gates carry the value (their rejects are genuinely −EV).
The **overextension** gate is the soft spot — near a coin-flip, and its rejects
are gross-positive (it's killing marginally-profitable, fat-tail-heavy signals;
the HOOD class).

## Can we loosen a gate to capture the winners? — No clean pattern

Loosening only helps if the would-have-won rejects share a *findable* feature.
They don't. Overextension by ATR distance is **flat**:

| ATR distance | n | would-win % | avg P&L |
|---|---|---|---|
| 2.5–3 | 91 | 51% | +0.33% |
| 3–4 | 92 | 50% | +0.16% |
| 4–5 | 35 | 49% | +0.67% |
| 5+ | 32 | 41% | +0.43% |

The winners are fat-tail outliers spread across all distances — there is no
threshold that cleanly separates them from the losers. **Recommendation: do not
loosen.** Loosening adds variance, not reliable wins. The catalyst/momentum tiers
already relax overextension where it's justified (news / strong trend).
