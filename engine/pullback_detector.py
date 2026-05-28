"""
Pullback Completion Detector
============================
Predictive setup detector that fires after a healthy pullback completes —
NOT during the pullback (knife-catch). Aimed at the trade structure:

  1. A "leg" — directional impulse of N+ bars
  2. A "pullback" — 1-3 counter-trend bars that DON'T break the leg origin
  3. A "reclaim" — price retakes the swing high (LONG) or swing low (SHORT)
     made just before the pullback started

Why this fires earlier than SMC at HEALTHY entries:
  - SMC pullback entries often fire mid-pullback (because the OB/FVG is mid-
    pullback). That's the knife-catch — if the pullback turns into reversal,
    stop hit.
  - This detector waits for confirmation that the pullback is OVER (price
    reclaims swing high) before firing. Worse R:R than catching the dead-cat-
    bounce low, but much higher WR.

Honest limits:
  - Reclaim sometimes fails on first attempt — fakeout, then real continuation.
    Downstream gates filter these.
  - Detector returns "setup ready" only at the bar that crosses the swing
    level — caller must run on every bar / tick to catch the moment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("signalbolt.pullback")

# ── Tunables ────────────────────────────────────────────────────────────────

LEG_BARS_MIN          = 2     # minimum strong bars in the initial impulse leg
LEG_RANGE_MULT        = 0.8   # each leg bar's range must be >= this × ATR(14)
PULLBACK_BARS_MIN     = 1
PULLBACK_BARS_MAX     = 3
MAX_PULLBACK_RETRACE  = 0.786 # if pullback retraces > 78.6% of leg → it's a reversal, skip
RECLAIM_BUFFER_PCT    = 0.05  # how far past swing level counts as a reclaim


@dataclass
class PullbackSetup:
    direction:           str       # 'LONG' or 'SHORT'
    entry:               float     # current price (reclaim level)
    swing_level:         float     # the swing high (LONG) / swing low (SHORT) being reclaimed
    leg_start:           float     # price at start of impulse leg
    leg_end:             float     # price at end of impulse leg (= swing level)
    pullback_low_or_high: float    # extreme of the pullback (for SL placement)
    atr:                 float
    leg_bars:            int
    pullback_bars:       int
    retracement_pct:     float
    setup_type:          str = "PULLBACK_CONTINUATION"


def _atr_hl(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) < period:
        return 0.0
    hl = (df["high"].values[-period:].astype(float) -
          df["low"].values[-period:].astype(float))
    return float(np.mean(hl))


def _is_green(o: float, c: float) -> bool:  return c > o
def _is_red  (o: float, c: float) -> bool:  return c < o


def detect(df: pd.DataFrame, current_price: float) -> Optional[PullbackSetup]:
    """
    Look at the last ~10 bars and find: impulse leg + pullback + reclaim.
    Returns a PullbackSetup if conditions met, else None.
    """
    if df is None or len(df) < 20:
        return None

    atr = _atr_hl(df, period=14)
    if atr <= 0:
        return None

    opens  = df["open"].values.astype(float)
    closes = df["close"].values.astype(float)
    highs  = df["high"].values.astype(float)
    lows   = df["low"].values.astype(float)

    # Search for the pattern across the last 10 bars (excluding the current
    # most-recent forming bar). For each possible pullback length, try to
    # match: [leg ≥ LEG_BARS_MIN] + [pullback BARS_MIN..BARS_MAX] + [reclaim now]
    n = len(df) - 1   # last completed bar index (we evaluate reclaim via current_price)

    for pullback_bars in range(PULLBACK_BARS_MIN, PULLBACK_BARS_MAX + 1):
        # Pullback occupies bars n - pullback_bars + 1 .. n
        pb_start = n - pullback_bars + 1
        if pb_start < LEG_BARS_MIN:
            continue
        pb_slice_o = opens[pb_start: n + 1]
        pb_slice_c = closes[pb_start: n + 1]

        # Try LONG: leg = green, pullback = red, reclaim above swing high
        all_red    = all(_is_red(o, c)   for o, c in zip(pb_slice_o, pb_slice_c))
        all_green  = all(_is_green(o, c) for o, c in zip(pb_slice_o, pb_slice_c))

        # ── LONG case: green leg, red pullback, current price reclaims swing high ──
        if all_red:
            # Find leg: bars before pullback that are green + each ≥ LEG_RANGE_MULT × ATR
            leg_end_idx = pb_start - 1   # last green bar before pullback
            leg_bars = 0
            for i in range(leg_end_idx, -1, -1):
                if _is_green(opens[i], closes[i]) and (highs[i] - lows[i]) >= LEG_RANGE_MULT * atr:
                    leg_bars += 1
                else:
                    break
            if leg_bars < LEG_BARS_MIN:
                continue

            leg_start_idx = leg_end_idx - leg_bars + 1
            leg_start     = float(opens[leg_start_idx])
            swing_high    = float(highs[leg_end_idx])    # the level to reclaim
            leg_size      = swing_high - leg_start
            if leg_size <= 0:
                continue

            pullback_low  = float(min(lows[pb_start: n + 1]))
            retrace_amt   = swing_high - pullback_low
            retrace_pct   = retrace_amt / leg_size
            if retrace_pct > MAX_PULLBACK_RETRACE:
                continue   # too deep — likely reversal not pullback

            # Reclaim check: current price > swing_high * (1 + buffer)
            reclaim_level = swing_high * (1 + RECLAIM_BUFFER_PCT / 100)
            if current_price >= reclaim_level:
                return PullbackSetup(
                    direction            = "LONG",
                    entry                = current_price,
                    swing_level          = swing_high,
                    leg_start            = leg_start,
                    leg_end              = swing_high,
                    pullback_low_or_high = pullback_low,
                    atr                  = atr,
                    leg_bars             = leg_bars,
                    pullback_bars        = pullback_bars,
                    retracement_pct      = round(retrace_pct, 3),
                )

        # ── SHORT case: red leg, green pullback, current price breaks swing low ──
        if all_green:
            leg_end_idx = pb_start - 1
            leg_bars = 0
            for i in range(leg_end_idx, -1, -1):
                if _is_red(opens[i], closes[i]) and (highs[i] - lows[i]) >= LEG_RANGE_MULT * atr:
                    leg_bars += 1
                else:
                    break
            if leg_bars < LEG_BARS_MIN:
                continue

            leg_start_idx = leg_end_idx - leg_bars + 1
            leg_start     = float(opens[leg_start_idx])
            swing_low     = float(lows[leg_end_idx])
            leg_size      = leg_start - swing_low
            if leg_size <= 0:
                continue

            pullback_high = float(max(highs[pb_start: n + 1]))
            retrace_amt   = pullback_high - swing_low
            retrace_pct   = retrace_amt / leg_size
            if retrace_pct > MAX_PULLBACK_RETRACE:
                continue

            break_level   = swing_low * (1 - RECLAIM_BUFFER_PCT / 100)
            if current_price <= break_level:
                return PullbackSetup(
                    direction            = "SHORT",
                    entry                = current_price,
                    swing_level          = swing_low,
                    leg_start            = leg_start,
                    leg_end              = swing_low,
                    pullback_low_or_high = pullback_high,
                    atr                  = atr,
                    leg_bars             = leg_bars,
                    pullback_bars        = pullback_bars,
                    retracement_pct      = round(retrace_pct, 3),
                )

    return None
