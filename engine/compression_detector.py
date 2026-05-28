"""
Compression Breakout Detector
=============================
Predictive setup detector that fires at the START of breakouts, not after
SMC structure confirms (which is always 50-70% into the move).

Logic:
  1. Find a "compression" — N consecutive bars where the range high-low is
     tight relative to recent ATR (< COMPRESSION_RATIO × ATR).
  2. Define the consolidation envelope: max(high) and min(low) across those bars.
  3. The MOST RECENT bar must be inside the envelope (otherwise the breakout
     already happened on the last bar and we'd be chasing).
  4. The NEXT bar that breaks above range_high → LONG breakout (entry = breakout price).
     Or breaks below range_low → SHORT breakdown.

Why this fires earlier than SMC:
  - SMC needs a Break-of-Structure or Change-of-Character confirmation, which
    typically happens 2-3 bars AFTER the initial impulse.
  - Compression breakout fires on bar #1 of the impulse — at the exact moment
    price exits the consolidation.

Honest limits:
  - More fakeouts than SMC (compression breaks can fail).
  - The downstream entry gate (multi-tf trend, tape, spread, patterns) filters
    these the same way it filters SMC signals, so noise is bounded.
  - Returns the "setup ready" state — actual fire requires breakout, which the
    runner checks on every scan pass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("signalbolt.compression")

# ── Tunables ────────────────────────────────────────────────────────────────

COMPRESSION_BARS  = 4       # consecutive bars required in tight range
COMPRESSION_RATIO = 0.80    # bar range must be ≤ this × ATR(14) — loosened from
                            # 0.55 (was 0/12 on liquid names; real consolidations
                            # often run 0.6-0.8× ATR per bar, not ultra-tight)
MIN_BREAKOUT_PCT  = 0.10    # current price must be at least this much past
                            # the envelope edge (avoid 1-tick wicks)
MAX_BREAKOUT_PCT  = 1.20    # if it's already ≥this % beyond the envelope,
                            # the move is too far along — don't chase


@dataclass
class CompressionSetup:
    direction:    str       # 'LONG' or 'SHORT'
    entry:        float     # current price (breakout level)
    range_high:   float
    range_low:    float
    atr:          float
    breakout_pct: float     # how far past the envelope edge the price is
    setup_type:   str = "COMPRESSION_BREAKOUT"


@dataclass
class CompressionZone:
    """A staged compression envelope — detected on bar close, watched per-tick
    for a breakout. No direction yet; that's decided by which edge price breaks."""
    range_high: float
    range_low:  float
    atr:        float


def _atr_hl(df: pd.DataFrame, period: int = 14) -> float:
    """High-low ATR over `period` bars. Intraday-friendly (no overnight gap)."""
    if df is None or len(df) < period:
        return 0.0
    hl = (df["high"].values[-period:].astype(float) -
          df["low"].values[-period:].astype(float))
    return float(np.mean(hl))


def detect(df: pd.DataFrame, current_price: float) -> Optional[CompressionSetup]:
    """
    Return a CompressionSetup if a fresh breakout is happening RIGHT NOW.
    None otherwise.

    `df` should be the entry-timeframe candles. `current_price` is the live
    tick (or last close) used to decide if a breakout has just occurred.
    """
    if df is None or len(df) < (COMPRESSION_BARS + 15):
        return None

    atr = _atr_hl(df, period=14)
    if atr <= 0:
        return None

    # The last COMPRESSION_BARS completed bars should be the tight consolidation.
    # The current (still-forming or just-closed) breakout candle is captured by
    # `current_price` — passed in separately by the caller.
    if len(df) < COMPRESSION_BARS + 1:
        return None
    comp_window = df.iloc[-COMPRESSION_BARS:]
    bar_ranges  = (comp_window["high"].values.astype(float)
                   - comp_window["low"].values.astype(float))
    max_range   = float(bar_ranges.max())

    if max_range > COMPRESSION_RATIO * atr:
        return None   # not tight enough

    # Envelope = compression high/low
    range_high = float(comp_window["high"].max())
    range_low  = float(comp_window["low"].min())

    # The compression must be NARROW (range high-low not too wide)
    env_width = range_high - range_low
    if env_width > 1.0 * atr:
        return None

    # Breakout check: is the current price past the envelope edge?
    upper_buffer = range_high * (1 + MIN_BREAKOUT_PCT / 100)
    upper_cap    = range_high * (1 + MAX_BREAKOUT_PCT / 100)
    lower_buffer = range_low  * (1 - MIN_BREAKOUT_PCT / 100)
    lower_cap    = range_low  * (1 - MAX_BREAKOUT_PCT / 100)

    if upper_buffer <= current_price <= upper_cap:
        return CompressionSetup(
            direction    = "LONG",
            entry        = current_price,
            range_high   = range_high,
            range_low    = range_low,
            atr          = atr,
            breakout_pct = (current_price - range_high) / range_high * 100,
        )
    if lower_cap <= current_price <= lower_buffer:
        return CompressionSetup(
            direction    = "SHORT",
            entry        = current_price,
            range_high   = range_high,
            range_low    = range_low,
            atr          = atr,
            breakout_pct = (range_low - current_price) / range_low * 100,
        )
    return None


def detect_zone(df: pd.DataFrame) -> Optional[CompressionZone]:
    """
    Detect a compression envelope WITHOUT requiring a breakout — used to stage
    a ticker for per-tick breakout watching. Returns the {range_high, range_low,
    atr} if the last COMPRESSION_BARS are tight, else None.

    This is the staging half of the two-phase compression flow:
      1. detect_zone() on bar close → stage envelope in stream watch set
      2. per-tick check in stream.on_trade → fire when price crosses an edge
    """
    if df is None or len(df) < (COMPRESSION_BARS + 15):
        return None
    atr = _atr_hl(df, period=14)
    if atr <= 0:
        return None

    comp_window = df.iloc[-COMPRESSION_BARS:]
    bar_ranges  = (comp_window["high"].values.astype(float)
                   - comp_window["low"].values.astype(float))
    if float(bar_ranges.max()) > COMPRESSION_RATIO * atr:
        return None

    range_high = float(comp_window["high"].max())
    range_low  = float(comp_window["low"].min())
    if (range_high - range_low) > 1.5 * atr:   # loosened from 1.0× (see above)
        return None

    return CompressionZone(range_high=range_high, range_low=range_low, atr=atr)
