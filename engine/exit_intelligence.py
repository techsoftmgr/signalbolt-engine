"""
Exit Intelligence
=================
Decides — in real time, after T1 — whether to CLOSE a winning position now or
HOLD for more. Replaces "blindly ride to T2 / trailing stop" with a multi-factor
read of market conditions.

Philosophy:
  - The trailing stop is the HARD FLOOR (never give back more than X% from peak).
  - This layer closes EARLY when multiple signals converge on a genuine reversal,
    capturing profit before the trailing stop would (which always gives back the
    trail distance).
  - CONVERGENCE is required: a single MACD blip won't close (that was the VLO
    over-booking mistake). We need ≥2 independent factors agreeing.

Factors (each adds to an "exit pressure" score 0-100):
  • Reversal from peak     — price pulled back from the post-T1 high
  • Momentum cross         — MACD turned against the position
  • RSI extreme            — overbought (LONG) / oversold (SHORT) and rolling over
  • Tape distribution      — net block flow against the position (from trade_tape)
  • Rejection candle       — last bar shows a rejection wick against the position

Dampeners (reduce pressure):
  • Near T2                — within NEAR_T2_PCT of the full target → hold for it

Decision: CLOSE if pressure ≥ EXIT_THRESHOLD AND pnl ≥ MIN_PROFIT_PCT.
Otherwise HOLD (the trailing stop still protects the downside).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("signalbolt.exit_intel")

# ── Tunables ────────────────────────────────────────────────────────────────

MIN_PROFIT_PCT  = 1.0    # don't even consider an early close below +1%
EXIT_THRESHOLD  = 55     # exit pressure needed to close (requires ~2+ factors)
NEAR_T2_PCT     = 0.4    # within 0.4% of T2 → hold for the full target

# Factor weights
W_REVERSAL_BIG   = 35    # ≥1.0% reversal from peak
W_REVERSAL_SMALL = 20    # ≥0.5% reversal from peak
W_MACD_CROSS     = 30
W_RSI_EXTREME    = 20
W_TAPE_AGAINST   = 25
W_REJECTION      = 15
W_NEAR_T2        = -40   # strong dampener — don't bail right before T2


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1):])
    gains  = np.where(deltas > 0, deltas, 0).sum() / period
    losses = np.where(deltas < 0, -deltas, 0).sum() / period
    if losses == 0:
        return 100.0
    rs = gains / losses
    return float(100 - (100 / (1 + rs)))


def evaluate_exit(
    sig:          dict,
    price:        float,
    df:           pd.DataFrame,
    peak:         float,
    tape_summary: Optional[dict] = None,
) -> dict:
    """
    Returns {"action": "close"|"hold", "score": int, "pnl_pct": float,
             "reasons": [str]}.

    `df` = entry-timeframe candles. `peak` = best price since T1.
    `tape_summary` = engine.trade_tape.get_summary(ticker) or None.
    """
    is_long = sig["direction"] == "LONG"
    entry   = float(sig["entry_price"])
    t2      = float(sig["target_two"])
    pnl_pct = ((price - entry) / entry * 100) if is_long else ((entry - price) / entry * 100)

    if pnl_pct < MIN_PROFIT_PCT:
        return {"action": "hold", "score": 0, "pnl_pct": round(pnl_pct, 2),
                "reasons": [f"profit {pnl_pct:.1f}% < {MIN_PROFIT_PCT}% floor"]}

    pressure = 0
    reasons: list[str] = []

    # 1. Reversal from peak
    if peak > 0:
        rev = (peak - price) / peak * 100 if is_long else (price - peak) / peak * 100
        if rev >= 1.0:
            pressure += W_REVERSAL_BIG;   reasons.append(f"{rev:.1f}% reversal from peak ${peak:.2f}")
        elif rev >= 0.5:
            pressure += W_REVERSAL_SMALL; reasons.append(f"{rev:.1f}% pullback from peak ${peak:.2f}")

    # 2. MACD cross against the position
    try:
        if df is not None and len(df) >= 35:
            c = df["close"]
            macd = _ema(c, 12) - _ema(c, 26)
            sigl = _ema(macd, 9)
            hist = (macd - sigl)
            h_now, h_prev = float(hist.iloc[-1]), float(hist.iloc[-2])
            if is_long and h_now < 0 <= h_prev:
                pressure += W_MACD_CROSS; reasons.append("MACD bearish cross")
            elif (not is_long) and h_now > 0 >= h_prev:
                pressure += W_MACD_CROSS; reasons.append("MACD bullish cross")
    except Exception:
        pass

    # 3. RSI extreme + rolling over
    try:
        if df is not None and len(df) >= 20:
            closes = df["close"].astype(float).values
            rsi = _rsi(closes)
            if is_long and rsi >= 72:
                pressure += W_RSI_EXTREME; reasons.append(f"RSI overbought {rsi:.0f}")
            elif (not is_long) and rsi <= 28:
                pressure += W_RSI_EXTREME; reasons.append(f"RSI oversold {rsi:.0f}")
    except Exception:
        pass

    # 4. Tape distribution — net block flow against the position
    if tape_summary:
        net = tape_summary.get("block_net_flow", 0) or 0
        if is_long and net < 0:
            pressure += W_TAPE_AGAINST; reasons.append(f"tape net selling ({net:,} blk shares)")
        elif (not is_long) and net > 0:
            pressure += W_TAPE_AGAINST; reasons.append(f"tape net buying ({net:,} blk shares)")

    # 5. Rejection candle on the last completed bar
    try:
        if df is not None and len(df) >= 1:
            o, h, l, c = (float(df["open"].iloc[-1]), float(df["high"].iloc[-1]),
                          float(df["low"].iloc[-1]),  float(df["close"].iloc[-1]))
            rng = h - l
            if rng > 0:
                upper_wick = h - max(o, c)
                lower_wick = min(o, c) - l
                if is_long and c < o and upper_wick > 0.5 * rng:
                    pressure += W_REJECTION; reasons.append("upper-wick rejection")
                elif (not is_long) and c > o and lower_wick > 0.5 * rng:
                    pressure += W_REJECTION; reasons.append("lower-wick rejection")
    except Exception:
        pass

    # Dampener: near T2 → hold for the full target
    dist_to_t2 = abs(t2 - price) / price * 100
    if dist_to_t2 <= NEAR_T2_PCT:
        pressure += W_NEAR_T2; reasons.append(f"within {dist_to_t2:.1f}% of T2 — holding")

    pressure = max(0, pressure)
    action = "close" if pressure >= EXIT_THRESHOLD else "hold"
    return {
        "action":  action,
        "score":   pressure,
        "pnl_pct": round(pnl_pct, 2),
        "reasons": reasons or ["no exit pressure"],
    }
