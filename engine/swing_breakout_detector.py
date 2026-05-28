"""
Swing-High Breakout Detector
============================
Fires when price breaks above the recent swing high (LONG) or below the recent
swing low (SHORT) — catching MOMENTUM breakouts that aren't preceded by tight
consolidation (which compression_detector handles).

This is the broader cousin of compression:
  - Compression: breakout from a TIGHT coil (low fakeout, but rare)
  - Swing breakout: breakout from any recent swing high (common, but noisier)

The CCL case (2026-05-28) motivated this: price drifted up, then a huge green
candle broke the recent swing high. Compression missed it (no tight coil), SMC
caught it late (after bar close). A swing-high breakout fires on the tick price
crosses the prior swing high — at the START of the move.

Fakeout control (swing breakouts fail more than compression):
  1. MIN_BREAKOUT_PCT buffer — price must CLEAR the level, not just touch it
  2. MAX_BREAKOUT_PCT cap — don't chase if it already ran too far
  3. Swing high must be a real local high over SWING_LOOKBACK bars
  4. Downstream entry gate (patterns/volume/tape) filters low-conviction breaks
  5. Only stage if price is still on the pre-breakout side (room to break)

Two-phase like compression/pullback:
  1. detect_zone() on bar close → stage swing high/low levels
  2. per-tick check in stream.on_trade → fire on the cross
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("signalbolt.swing_breakout")

# ── Tunables ────────────────────────────────────────────────────────────────

SWING_LOOKBACK   = 12     # bars to find the swing high / low
MIN_BREAKOUT_PCT = 0.05   # price must clear the level by this %
MAX_BREAKOUT_PCT = 0.80   # don't stage/fire if already this far past (anti-chase)
MIN_ROOM_PCT     = 0.10   # swing level must be at least this % away from price
                          # (so we're staging a real pending breakout, not noise)


@dataclass
class SwingBreakoutZone:
    swing_high: float
    swing_low:  float
    atr:        float
    avg_volume: float


def _atr_hl(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) < period:
        return 0.0
    hl = (df["high"].values[-period:].astype(float)
          - df["low"].values[-period:].astype(float))
    return float(np.mean(hl))


def detect_zone(df: pd.DataFrame, current_price: float) -> Optional[SwingBreakoutZone]:
    """
    Stage the recent swing high/low for per-tick breakout watching.

    Returns the zone if there's a meaningful swing level with room for a
    breakout (price still on the pre-breakout side, not already past).
    None if no clean level or price already broke out.
    """
    if df is None or len(df) < (SWING_LOOKBACK + 5):
        return None
    atr = _atr_hl(df, period=14)
    if atr <= 0:
        return None

    # Swing high/low over the lookback window (exclude the most recent forming
    # bar so we don't anchor on an in-progress move).
    window = df.iloc[-(SWING_LOOKBACK + 1):-1]
    swing_high = float(window["high"].max())
    swing_low  = float(window["low"].min())

    avg_volume = float(window["volume"].mean()) if "volume" in window else 0.0

    # Only stage if price has ROOM below the swing high (LONG) or above the
    # swing low (SHORT). If price already cleared a level, it's too late.
    room_high = (swing_high - current_price) / current_price * 100
    room_low  = (current_price - swing_low)  / current_price * 100

    # We need at least one side with room (the other may already be broken).
    # Keep the zone if EITHER level is a valid pending breakout target.
    long_ok  = MIN_ROOM_PCT <= room_high <= 3.0   # high is 0.1%–3% above price
    short_ok = MIN_ROOM_PCT <= room_low  <= 3.0   # low is 0.1%–3% below price

    if not (long_ok or short_ok):
        return None

    return SwingBreakoutZone(
        swing_high = swing_high if long_ok  else 0.0,   # 0 = don't watch this side
        swing_low  = swing_low  if short_ok else 0.0,
        atr        = atr,
        avg_volume = avg_volume,
    )
