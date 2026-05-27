"""
SL/TP Engine (Gamma-Aware, Realistic Targets)
==============================================
Calculates Stop Loss and Take Profit levels that are:

  1. Based on INTRADAY ATR for scalping/day_trade (High-Low only — no overnight
     gap inflation).  Swing trade uses classic True Range ATR (gaps matter there).
  2. Capped per strategy with VOLATILITY-AWARE widening: the cap is at least
     the per-strategy floor (e.g. 0.8% for day_trade) but can extend up to a
     fraction of the ticker's ADR (e.g. 40% of ADR for day_trade), never
     exceeding the absolute ceiling (e.g. 2.5%). Fixes the "uniform 0.8% SL
     on every signal" pattern that swept volatile names (NVDA, TSLA) on
     Friday — those got stopped inside normal noise, then price recovered.
  3. Targeted at achievable moves: T1 ≤ 50% of the Average Daily Range for
     intraday strategies, T2 ≤ 80%.  Swing trade has no ADR cap.
  4. R:R validated: T1 must clear the per-strategy minimum AFTER any ADR cap.
     Signals that can't produce a realistic T1 are rejected.
  5. Avoid round numbers (stop raid magnets).
  6. Placed beyond nearest gamma support/resistance.

Per-strategy parameters:
  ┌──────────────┬──────────┬───────────┬─────────────┬─────────────┬───────┬───────┐
  │ Strategy     │ ATR type │ Floor cap │ ADR fraction│ Ceiling cap │ T1 RR │ T2 RR │
  ├──────────────┼──────────┼───────────┼─────────────┼─────────────┼───────┼───────┤
  │ scalping     │ H-L      │ 0.4%      │ 0% (off)    │ 0.6%        │ 1.5×  │ 2.5×  │
  │ day_trade    │ H-L      │ 0.8%      │ 40% of ADR  │ 2.5%        │ 1.5×  │ 2.5×  │
  │ swing_trade  │ True Rng │ 4.0%      │ 0% (off)    │ 6.0%        │ 2.0×  │ 3.5×  │
  │ options_flow │ H-L      │ 1.0%      │ 40% of ADR  │ 2.0%        │ 1.5×  │ 2.5×  │
  │ dark_pool    │ H-L      │ 1.0%      │ 40% of ADR  │ 2.0%        │ 1.5×  │ 2.5×  │
  └──────────────┴──────────┴───────────┴─────────────┴─────────────┴───────┴───────┘

  Effective SL cap = min( max(floor, ADR×fraction), ceiling )
  So KO (low vol) stays at floor; NVDA (high vol) widens via ADR; nothing
  ever blows past the ceiling.
"""

import logging
import math

logger = logging.getLogger("signalbolt.sl_tp")

# ── ATR multipliers (base, before regime/session adjustments) ─────────────────
# H-L ATR is ~40-60% of True Range ATR (no overnight gaps).
# Keeping the same multiplier as before (1.5 for day_trade) gives a stop
# that is roughly 0.6-0.8× the old stop — realistic and still outside 1-bar noise.
_ATR_MULT: dict[str, float] = {
    "scalping":     1.0,
    "day_trade":    1.5,   # same mult, but ATR is now H-L only → stop ~60% of old
    "swing_trade":  2.0,
    "options_flow": 1.5,
    "dark_pool":    1.5,
}

# ── Per-strategy stop-loss FLOOR cap ──────────────────────────────────────────
# This is the MINIMUM cap — a low-volatility stock (e.g. KO with ~0.8% ADR)
# still gets at least this much SL room so we don't choke it inside 1-bar noise.
_MAX_SL_PCT: dict[str, float] = {
    "scalping":     0.004,   # 0.4%
    "day_trade":    0.008,   # 0.8%  — ↓ from global 8%; keeps stops realistic
    "swing_trade":  0.040,   # 4.0%
    "options_flow": 0.010,   # 1.0%
    "dark_pool":    0.010,   # 1.0%
}

# ── Per-strategy ADR-fraction cap ─────────────────────────────────────────────
# Volatility-aware widening: cap may extend up to this fraction of the ticker's
# Average Daily Range. Friday's data showed every day_trade signal getting
# clamped to ~0.8% because the floor cap above ignored ticker volatility —
# NVDA (~3% ADR) got the same SL distance as KO (~0.8% ADR), so NVDA stops
# fell inside its normal noise band and got swept; price then recovered.
# A fraction of 0.4 means NVDA's SL can extend to ~1.2% of price (40% of 3%),
# which sits outside typical intra-bar wiggle for high-vol names.
#
# Scalping stays at 0.0 — its tight stops are by design (fast exits).
# Swing already runs wide (4% floor) so 0.0 too.
_ADR_SL_FRACTION: dict[str, float] = {
    "scalping":     0.00,
    "day_trade":    0.40,
    "swing_trade":  0.00,
    "options_flow": 0.40,
    "dark_pool":    0.40,
}

# ── Per-strategy absolute SL ceiling ──────────────────────────────────────────
# Hard upper bound — even on a frothy day with 5% ADR, the SL never goes
# beyond this. Protects against a single trade blowing the day's risk budget.
_ABSOLUTE_MAX_SL_PCT: dict[str, float] = {
    "scalping":     0.006,   # 0.6%
    "day_trade":    0.025,   # 2.5%
    "swing_trade":  0.060,   # 6.0%
    "options_flow": 0.020,   # 2.0%
    "dark_pool":    0.020,   # 2.0%
}

# ── Per-strategy minimum stop floor ──────────────────────────────────────────
_MIN_SL_PCT: dict[str, float] = {
    "scalping":     0.001,   # 0.1%
    "day_trade":    0.002,   # 0.2%
    "swing_trade":  0.005,   # 0.5%
    "options_flow": 0.002,
    "dark_pool":    0.002,
}

# ── Per-strategy R:R ratios ───────────────────────────────────────────────────
# RAISED 2026-05-27 after this week's data showed avg_loss (1.67%) running
# 26% bigger than avg_win (1.32%) on day_trade. Even at 50% WR that math
# loses money. Raising the minimum R:R required at fire time filters out
# marginal setups whose target is too close relative to the (now-wider,
# volatility-aware) stop. Trades that DO fire have to clear a higher
# theoretical bar — fewer signals, better ratio when they print.
_MIN_RR_T1: dict[str, float] = {
    "scalping":     1.8,    # ↑ from 1.5
    "day_trade":    2.0,    # ↑ from 1.5
    "swing_trade":  2.5,    # ↑ from 2.0
    "options_flow": 1.8,    # ↑ from 1.5
    "dark_pool":    1.8,    # ↑ from 1.5
}

_TARGET2_RR: dict[str, float] = {
    "scalping":     3.0,    # ↑ from 2.5
    "day_trade":    3.0,    # ↑ from 2.5
    "swing_trade":  4.0,    # ↑ from 3.5
    "options_flow": 3.0,    # ↑ from 2.5
    "dark_pool":    3.0,    # ↑ from 2.5
}

# ── Which strategies use H-L ATR (intraday, no overnight gap) ────────────────
_USE_HL_ATR = {"scalping", "day_trade", "options_flow", "dark_pool"}

# ── ADR caps for intraday strategies ─────────────────────────────────────────
# Prevents targets that price cannot realistically reach within one session.
# 0.0 = no cap (swing trade).
_ADR_T1_FRACTION: dict[str, float] = {
    "scalping":     0.30,   # T1 ≤ 30% of Average Daily Range from entry
    "day_trade":    0.50,   # T1 ≤ 50% of ADR
    "swing_trade":  0.00,   # no intraday cap
    "options_flow": 0.50,
    "dark_pool":    0.50,
}
_ADR_T2_FRACTION: dict[str, float] = {
    "scalping":     0.50,
    "day_trade":    0.80,   # T2 ≤ 80% of ADR
    "swing_trade":  0.00,
    "options_flow": 0.80,
    "dark_pool":    0.80,
}

# Approximate regular-session bars per day for ADR estimation
_BARS_PER_DAY: dict[str, int] = {
    "5m":  78,
    "15m": 26,
    "1h":  7,
    "4h":  2,
}


# ── ATR helpers ───────────────────────────────────────────────────────────────

def _compute_hl_atr(df, period: int = 14) -> float:
    """
    Intraday ATR: average of (High - Low) per bar only.
    Eliminates overnight close-to-open gaps that inflate True Range ATR
    and produce unrealistically wide stops for day trades.
    """
    try:
        highs  = df["high"].tolist()
        lows   = df["low"].tolist()
        ranges = [h - l for h, l in zip(highs, lows) if h > l]
        if not ranges:
            return 0.0
        n = min(period, len(ranges))
        return sum(ranges[-n:]) / n
    except Exception as e:
        logger.debug(f"H-L ATR error: {e}")
        return 0.0


def _compute_atr(df, period: int = 14) -> float:
    """Classic True Range ATR — appropriate for swing trade where overnight gaps matter."""
    try:
        highs  = df["high"].tolist()
        lows   = df["low"].tolist()
        closes = df["close"].tolist()

        trs = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i]  - closes[i - 1]),
            )
            trs.append(tr)

        if len(trs) < period:
            return highs[-1] - lows[-1] if highs and lows else closes[-1] * 0.01

        # Wilder smoothing
        atr = sum(trs[:period]) / period
        for tr in trs[period:]:
            atr = (atr * (period - 1) + tr) / period

        return atr
    except Exception as e:
        logger.debug(f"True-range ATR error: {e}")
        return 0.0


def _compute_adr(df, interval: str = "15m", days: int = 5) -> float:
    """
    Average Daily Range: mean of (session_high - session_low) over `days` trading days.
    Groups bars into day-sized chunks using approximate bars-per-day for the interval.
    Returns 0.0 if data is insufficient.
    """
    try:
        highs = df["high"].tolist()
        lows  = df["low"].tolist()
        n     = len(highs)

        bars_per_day = _BARS_PER_DAY.get(interval, 26)
        if n < bars_per_day:
            # Fewer bars than one day — use overall range as rough proxy
            return max(highs) - min(lows)

        daily_ranges = []
        for day_i in range(days):
            end   = n - day_i * bars_per_day
            start = end - bars_per_day
            if start < 0:
                break
            day_h = max(highs[start:end])
            day_l = min(lows[start:end])
            if day_h > day_l:
                daily_ranges.append(day_h - day_l)

        return sum(daily_ranges) / len(daily_ranges) if daily_ranges else 0.0
    except Exception as e:
        logger.debug(f"ADR compute error: {e}")
        return 0.0


def _round_number_adjustment(price: float, nudge: str = "down", buffer: float = 0.002) -> float:
    """Shift price away from round numbers (stop raid magnets)."""
    rounded  = round(price)
    distance = abs(price - rounded) / max(price, 1)
    if distance < buffer:
        if nudge == "down":
            return round(rounded * (1 - buffer), 2)
        else:
            return round(rounded * (1 + buffer), 2)
    return price


# ── Main entry point ─────────────────────────────────────────────────────────

def calculate(
    direction: str,
    entry: float,
    df,
    regime: dict,
    session: dict,
    gamma: dict,
    strategy_type: str = "day_trade",
    interval: str = "15m",
) -> dict:
    """
    Calculate realistic, gamma-aware SL/TP levels.

    Args:
        direction:     'LONG' or 'SHORT'
        entry:         entry price
        df:            OHLCV DataFrame
        regime:        output of regime_detector.detect()
        session:       output of session_classifier.classify()
        gamma:         output of gamma_engine.fetch()
        strategy_type: determines SL/TP sizing and caps
        interval:      bar interval ('5m', '15m', '1h') — used for ADR estimation

    Returns dict with:
        stop_loss, target_one, target_two,
        risk_reward_1, risk_reward_2,
        atr, atr_multiple, adr, adjustments, valid
    """
    adjustments: list[str] = []

    # ── Step 1: Choose ATR method by strategy ──────────────────────────────
    # Intraday strategies: H-L ATR (no overnight gaps → realistic bar-size stops)
    # Swing trade: True Range ATR (overnight gaps are genuine price moves)
    use_hl = strategy_type in _USE_HL_ATR

    if df is not None and not df.empty:
        atr = _compute_hl_atr(df) if use_hl else _compute_atr(df)
    else:
        atr = 0.0

    # Fallback if ATR is zero or suspiciously small
    if atr <= 0 or atr < entry * 0.0003:
        atr = entry * (0.004 if use_hl else 0.01)
        adjustments.append("ATR fallback used (insufficient data)")

    atr_mult = _ATR_MULT.get(strategy_type, 1.5)

    # ── Step 2: Regime adjustment ──────────────────────────────────────────
    from engine.regime_detector import get_sl_adjustment as regime_sl_adj
    reg_adj = regime_sl_adj(regime)
    if reg_adj != 1.0:
        atr_mult *= reg_adj
        adjustments.append(
            f"SL +{int((reg_adj - 1) * 100)}% ({regime.get('regime_type', '?')} regime)"
        )

    # ── Step 3: Session adjustment ─────────────────────────────────────────
    sess_adj = session.get("sl_adjustment", 1.0)
    if sess_adj != 1.0:
        atr_mult *= sess_adj
        adjustments.append(
            f"SL +{int((sess_adj - 1) * 100)}% ({session.get('mode', '?')} session)"
        )

    # ── Step 4: Negative gamma → widen stop ───────────────────────────────
    if gamma.get("is_negative_gamma"):
        atr_mult *= 1.15
        adjustments.append("SL +15% (negative gamma — amplified moves)")

    # ── Step 5: Compute base stop distance ────────────────────────────────
    stop_dist = atr * atr_mult

    # ── Step 5.5: Compute ADR early — needed for the volatility-aware cap ─
    # (Used again in Step 10 for target capping; cached here.)
    adr = 0.0
    if df is not None and not df.empty:
        adr = _compute_adr(df, interval=interval)

    # ── Step 6: Clamp stop to per-strategy bounds ─────────────────────────
    # The cap is volatility-aware: at least the per-strategy floor, can
    # widen up to a fraction of ADR for high-vol names, never above the
    # absolute ceiling. See _ADR_SL_FRACTION / _ABSOLUTE_MAX_SL_PCT for why.
    floor_pct    = _MAX_SL_PCT.get(strategy_type, 0.02)
    adr_fraction = _ADR_SL_FRACTION.get(strategy_type, 0.0)
    ceiling_pct  = _ABSOLUTE_MAX_SL_PCT.get(strategy_type, 0.025)
    min_sl_pct   = _MIN_SL_PCT.get(strategy_type, 0.002)

    floor_stop   = entry * floor_pct
    adr_stop     = adr * adr_fraction if (adr > 0 and adr_fraction > 0) else 0.0
    ceiling_stop = entry * ceiling_pct

    max_stop = min(max(floor_stop, adr_stop), ceiling_stop)
    min_stop = entry * min_sl_pct

    if stop_dist > max_stop:
        cap_pct = (max_stop / entry) * 100 if entry > 0 else 0.0
        # Be explicit about which lever actually set the cap so we can
        # debug later from signal logs without re-deriving the math.
        if adr_stop >= floor_stop and adr_stop < ceiling_stop:
            cap_source = f"ADR-aware ({adr_fraction:.0%} of ADR ${adr:.2f})"
        elif ceiling_stop <= max(floor_stop, adr_stop):
            cap_source = f"absolute ceiling {ceiling_pct * 100:.1f}%"
        else:
            cap_source = f"floor {floor_pct * 100:.1f}%"
        stop_dist = max_stop
        adjustments.append(
            f"SL capped at {cap_pct:.2f}% via {cap_source} "
            f"(raw ATR×mult was ${atr * atr_mult:.2f} = {(atr * atr_mult / entry) * 100:.2f}%)"
        )
    if stop_dist < min_stop:
        stop_dist = min_stop
        adjustments.append(f"SL floored at {min_sl_pct * 100:.1f}% min")

    # ── Step 7: Place SL ──────────────────────────────────────────────────
    if direction == "LONG":
        sl = entry - stop_dist
    else:
        sl = entry + stop_dist

    # ── Step 8: Avoid round numbers ───────────────────────────────────────
    before = sl
    sl = _round_number_adjustment(sl, nudge="down" if direction == "LONG" else "up")
    if sl != before:
        adjustments.append(f"SL shifted from {before:.2f} (round number avoidance)")

    # ── Step 9: Gamma SL adjustment ───────────────────────────────────────
    if gamma.get("available"):
        from engine.gamma_engine import adjust_sl_for_gamma
        sl_before = sl
        sl, gamma_sl_reason = adjust_sl_for_gamma(sl, direction, gamma, entry)
        if gamma_sl_reason:
            # Re-clamp after gamma adjustment — gamma should not breach the max
            new_dist = abs(entry - sl)
            if new_dist > max_stop:
                sl = (entry - max_stop) if direction == "LONG" else (entry + max_stop)
                gamma_sl_reason += f" (re-capped to {(max_stop / entry) * 100:.2f}%)"
            adjustments.append(gamma_sl_reason)

    sl = round(sl, 2)

    # Recalculate actual risk after all adjustments
    risk = abs(entry - sl)
    if risk <= 0:
        risk = entry * min_sl_pct

    # ── Step 10: (ADR already computed in Step 5.5 for the SL cap) ────────

    # ── Step 11: Calculate targets ────────────────────────────────────────
    min_rr_t1  = _MIN_RR_T1.get(strategy_type, 1.5)
    target2_rr = _TARGET2_RR.get(strategy_type, 2.5)

    if direction == "LONG":
        t1 = round(entry + risk * min_rr_t1,  2)
        t2 = round(entry + risk * target2_rr, 2)
    else:
        t1 = round(entry - risk * min_rr_t1,  2)
        t2 = round(entry - risk * target2_rr, 2)

    # ── Step 12: ADR-based target cap (intraday only) ─────────────────────
    adr_t1_frac = _ADR_T1_FRACTION.get(strategy_type, 0.0)
    adr_t2_frac = _ADR_T2_FRACTION.get(strategy_type, 0.0)

    if adr > 0 and adr_t1_frac > 0:
        max_t1_dist = adr * adr_t1_frac
        max_t2_dist = adr * adr_t2_frac

        t1_dist = abs(t1 - entry)
        t2_dist = abs(t2 - entry)

        if t1_dist > max_t1_dist:
            t1 = round(entry + max_t1_dist, 2) if direction == "LONG" else round(entry - max_t1_dist, 2)
            adjustments.append(
                f"T1 capped at {adr_t1_frac * 100:.0f}% ADR "
                f"({adr:.2f} ADR → max {max_t1_dist:.2f} move)"
            )

        if t2_dist > max_t2_dist:
            t2 = round(entry + max_t2_dist, 2) if direction == "LONG" else round(entry - max_t2_dist, 2)
            adjustments.append(
                f"T2 capped at {adr_t2_frac * 100:.0f}% ADR"
            )

    # ── Step 13: Gamma TP adjustment ──────────────────────────────────────
    if gamma.get("available"):
        from engine.gamma_engine import adjust_tp_for_gamma_wall
        t1, tp1_reason = adjust_tp_for_gamma_wall(t1, direction, gamma)
        t2, tp2_reason = adjust_tp_for_gamma_wall(t2, direction, gamma)
        if tp1_reason: adjustments.append(f"TP1: {tp1_reason}")
        if tp2_reason: adjustments.append(f"TP2: {tp2_reason}")

    # ── Step 14: Final R:R validation ─────────────────────────────────────
    t1_dist = abs(t1 - entry)
    t2_dist = abs(t2 - entry)
    rr1 = t1_dist / risk if risk > 0 else 0.0
    rr2 = t2_dist / risk if risk > 0 else 0.0

    # Signal is valid if T1 clears the minimum R:R after all caps
    valid = rr1 >= min_rr_t1 * 0.85   # allow 15% tolerance for gamma/ADR nudges

    if not valid:
        logger.debug(
            f"[sl_tp] INVALID — rr1={rr1:.2f} < min {min_rr_t1 * 0.85:.2f} "
            f"entry={entry} sl={sl} t1={t1} atr={atr:.3f} adr={adr:.2f} "
            f"strategy={strategy_type}"
        )

    return {
        "stop_loss":     sl,
        "target_one":    t1,
        "target_two":    t2,
        "risk_reward_1": round(rr1, 2),
        "risk_reward_2": round(rr2, 2),
        "atr":           round(atr, 4),
        "atr_multiple":  round(atr_mult, 2),
        "adr":           round(adr, 2),
        "adjustments":   adjustments,
        "valid":         valid,
    }
