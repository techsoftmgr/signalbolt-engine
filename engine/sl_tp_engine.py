"""
SL/TP Engine (Gamma-Aware)
==========================
Calculates intelligent Stop Loss and Take Profit levels that:

  1. Use ATR(14) as the base risk unit
  2. Widen for regime (high VIX) and session (first 15 min)
  3. Avoid round numbers (stop raid magnets)
  4. Place SL beyond the nearest gamma support/resistance level
  5. Cap TP before the nearest gamma wall (MMs dump there)
  6. Validate minimum 1:2 Risk/Reward

Replaces the fixed-percentage SL/TP in smc.py for all strategies.
"""

import logging
import math
from typing import Optional

logger = logging.getLogger("signalbolt.sl_tp")

MIN_RR      = 2.0    # minimum risk/reward ratio
TARGET2_RR  = 3.0    # extended target
MAX_SL_PCT  = 0.08   # max 8% stop
MIN_SL_PCT  = 0.002  # min 0.2% stop


def _round_number_adjustment(price: float, nudge: str = "down", buffer: float = 0.002) -> float:
    """
    Shift price away from round numbers (stop raid magnets).

    Args:
        nudge: 'down' for LONG SL (push SL lower, away from raid zone)
               'up' for SHORT SL (push SL higher, away from raid zone)
    """
    rounded = round(price)
    distance = abs(price - rounded) / max(price, 1)
    if distance < buffer:
        if nudge == "down":
            return round(rounded * (1 - buffer), 2)
        else:
            return round(rounded * (1 + buffer), 2)
    return price


def _compute_atr(df, period: int = 14) -> float:
    """Compute ATR(14) from OHLCV DataFrame."""
    try:
        highs  = df["high"].tolist()
        lows   = df["low"].tolist()
        closes = df["close"].tolist()

        trs = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            trs.append(tr)

        if len(trs) < period:
            # Fallback: use last bar range
            return highs[-1] - lows[-1] if highs and lows else closes[-1] * 0.01

        # Wilder smoothing
        atr = sum(trs[:period]) / period
        for tr in trs[period:]:
            atr = (atr * (period - 1) + tr) / period

        return atr
    except Exception as e:
        logger.debug(f"ATR compute error: {e}")
        return 0.0


def calculate(
    direction: str,
    entry: float,
    df,
    regime: dict,
    session: dict,
    gamma: dict,
    strategy_type: str = "day_trade",
) -> dict:
    """
    Calculate gamma-aware SL/TP levels.

    Args:
        direction:     'LONG' or 'SHORT'
        entry:         entry price
        df:            OHLCV DataFrame (for ATR)
        regime:        output of regime_detector.detect()
        session:       output of session_classifier.classify()
        gamma:         output of gamma_engine.fetch()
        strategy_type: for ATR multiplier selection

    Returns:
        {
          "stop_loss":         float,
          "target_one":        float,
          "target_two":        float,
          "risk_reward_1":     float,
          "risk_reward_2":     float,
          "atr":               float,
          "atr_multiple":      float,
          "adjustments":       list[str],
          "valid":             bool,    # False if R:R < 2
        }
    """
    adjustments = []

    # ── Step 1: Base ATR multiple by strategy ─────────────────
    atr_multiples = {
        "scalping":     1.0,
        "day_trade":    1.5,
        "swing_trade":  2.0,
        "options_flow": 1.5,
        "dark_pool":    1.5,
    }
    atr_mult = atr_multiples.get(strategy_type, 1.5)

    # ── Step 2: Compute ATR ───────────────────────────────────
    atr = _compute_atr(df) if df is not None and not df.empty else entry * 0.01
    if atr <= 0:
        atr = entry * 0.01

    # ── Step 3: Regime adjustment ─────────────────────────────
    from engine.regime_detector import get_sl_adjustment as regime_sl_adj
    reg_adj = regime_sl_adj(regime)
    if reg_adj != 1.0:
        atr_mult *= reg_adj
        adjustments.append(
            f"SL +{int((reg_adj - 1) * 100)}% ({regime.get('regime_type', '?')} regime)"
        )

    # ── Step 4: Session adjustment ─────────────────────────────
    sess_adj = session.get("sl_adjustment", 1.0)
    if sess_adj != 1.0:
        atr_mult *= sess_adj
        adjustments.append(
            f"SL +{int((sess_adj - 1) * 100)}% ({session.get('mode', '?')} session)"
        )

    # ── Step 5: Negative gamma zone → widen ──────────────────
    if gamma.get("is_negative_gamma"):
        atr_mult *= 1.15
        adjustments.append("SL +15% (negative gamma zone — amplified moves)")

    # ── Step 6: Calculate base SL ────────────────────────────
    stop_dist = atr * atr_mult
    if direction == "LONG":
        sl = entry - stop_dist
    else:
        sl = entry + stop_dist

    # ── Step 7: Avoid round numbers ──────────────────────────
    # LONG  SL is below entry → nudge DOWN (further from round-number raid zone)
    # SHORT SL is above entry → nudge UP  (further from round-number raid zone)
    before = sl
    if direction == "LONG":
        sl = _round_number_adjustment(sl, nudge="down")
    else:
        sl = _round_number_adjustment(sl, nudge="up")
    if sl != before:
        adjustments.append(f"SL shifted from {before:.2f} (round number avoidance)")

    # ── Step 8: Gamma SL adjustment ──────────────────────────
    if gamma.get("available"):
        from engine.gamma_engine import adjust_sl_for_gamma
        sl, gamma_sl_reason = adjust_sl_for_gamma(sl, direction, gamma, entry)
        if gamma_sl_reason:
            adjustments.append(gamma_sl_reason)

    # ── Step 9: Clamp SL within bounds ───────────────────────
    sl_dist_pct = abs(entry - sl) / entry
    if sl_dist_pct > MAX_SL_PCT:
        sl = (entry * (1 - MAX_SL_PCT)) if direction == "LONG" else (entry * (1 + MAX_SL_PCT))
        adjustments.append("SL capped at 8% max")
    if sl_dist_pct < MIN_SL_PCT:
        sl = (entry * (1 - MIN_SL_PCT)) if direction == "LONG" else (entry * (1 + MIN_SL_PCT))
        adjustments.append("SL floored at 0.2% min")

    sl = round(sl, 2)

    # ── Step 10: Calculate targets ────────────────────────────
    risk = abs(entry - sl)
    if direction == "LONG":
        t1 = round(entry + risk * MIN_RR, 2)
        t2 = round(entry + risk * TARGET2_RR, 2)
    else:
        t1 = round(entry - risk * MIN_RR, 2)
        t2 = round(entry - risk * TARGET2_RR, 2)

    # ── Step 11: Gamma TP adjustment ─────────────────────────
    if gamma.get("available"):
        from engine.gamma_engine import adjust_tp_for_gamma_wall
        t1, tp1_reason = adjust_tp_for_gamma_wall(t1, direction, gamma)
        t2, tp2_reason = adjust_tp_for_gamma_wall(t2, direction, gamma)
        if tp1_reason: adjustments.append(f"TP1: {tp1_reason}")
        if tp2_reason: adjustments.append(f"TP2: {tp2_reason}")

    # ── Step 12: Compute R:R ──────────────────────────────────
    rr1 = abs(t1 - entry) / risk if risk > 0 else 0
    rr2 = abs(t2 - entry) / risk if risk > 0 else 0

    valid = rr1 >= MIN_RR

    if not valid:
        logger.debug(
            f"[sl_tp] R:R={rr1:.2f} below minimum {MIN_RR} "
            f"entry={entry} sl={sl} t1={t1}"
        )

    return {
        "stop_loss":     sl,
        "target_one":    t1,
        "target_two":    t2,
        "risk_reward_1": round(rr1, 2),
        "risk_reward_2": round(rr2, 2),
        "atr":           round(atr, 4),
        "atr_multiple":  round(atr_mult, 2),
        "adjustments":   adjustments,
        "valid":         valid,
    }
