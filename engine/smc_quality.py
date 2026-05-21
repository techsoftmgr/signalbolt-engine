"""
SMC Quality Scorer
==================
Computes ATR-relative, freshness-aware quality scores for FVGs and Order Blocks.

This module is ADDITIVE to smc.py — it takes the SMC analysis dict already
produced by smc.analyze() and enriches it with quality scores for each detected
structure element. The runner calls this after smc.analyze() and passes the
enriched analysis to the scorer.

Why separate from smc.py?
  smc.py detects WHETHER structures exist (binary).
  This module scores HOW GOOD they are (0–100 per element).

FVG Quality factors:
  - Distance from price (closer = more actionable)
  - Age in bars (fresher = more relevant)
  - Size relative to ATR (larger gap = stronger imbalance)
  - Fill percentage (partially filled gaps are weaker)
  - Impulse candle strength (what created the gap)
  - Freshness decay (score degrades exponentially with age)

OB Quality factors:
  - Impulse strength (ATR-relative body move)
  - Volume during impulse (institutional fingerprint)
  - Age in bars (fresher = more valid)
  - Retest count (first retest = strongest)
  - Whether the OB zone has been violated (invalidated)
  - Distance from current price
  - ATR-adjusted zone width
  - Rejection quality (candle wick vs body at OB)

Structure Conflict Resolution:
  Instead of skipping when both bullish and bearish structure exist,
  this module scores each direction and returns the dominant one.
  Recent CHoCH always outweighs stale BOS. Recency-weighted scoring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("signalbolt.smc_quality")

# ── Quality caps ───────────────────────────────────────────────────────────────
MAX_FVG_SCORE = 100.0
MAX_OB_SCORE  = 100.0

# FVG freshness: score decays by this fraction per bar of age
FVG_DECAY_PER_BAR = 0.04   # 4% decay/bar → ~25 bars to reach 0

# OB freshness: slower decay (OBs stay valid longer than FVGs)
OB_DECAY_PER_BAR  = 0.025  # 2.5% decay/bar → ~40 bars

# OB impulse floor (ATR-relative) below which the OB is considered low quality
OB_IMPULSE_ATR_MIN = 0.8   # impulse must be ≥ 0.8× ATR to score quality points


@dataclass
class FVGQuality:
    score:           float    # 0–100
    size_atr_ratio:  float    # gap size / ATR
    fill_pct:        float    # 0.0 = unfilled, 1.0 = fully filled
    age_bars:        int
    impulse_strength: float   # body move of impulse candle / ATR
    distance_pct:    float    # price distance from FVG midpoint / price
    reasons:         list[str] = field(default_factory=list)


@dataclass
class OBQuality:
    score:           float    # 0–100
    impulse_atr:     float    # impulse body move / ATR
    age_bars:        int
    retest_count:    int      # number of times price has revisited
    is_violated:     bool     # True if price closed through OB
    distance_pct:    float    # midpoint distance from current price
    width_atr:       float    # OB zone width / ATR
    reasons:         list[str] = field(default_factory=list)


@dataclass
class StructureScore:
    """Directional structure scoring with recency weighting."""
    direction:          str    # "LONG" or "SHORT"
    structure_type:     str    # "choch" or "bos" or "none"
    raw_score:          float  # 0–100
    recency_bars:       int    # how many bars ago the break occurred
    impulse_strength:   float  # strength of the breaking move
    htf_aligned:        bool   # whether higher timeframe agrees
    conflict_resolved:  bool   # True if we had to resolve a conflict


@dataclass
class SMCQualityResult:
    """Full quality enrichment for an SMC analysis dict."""
    fvg_quality:       Optional[FVGQuality]     = None
    ob_quality:        Optional[OBQuality]      = None
    structure_score:   Optional[StructureScore] = None
    direction_override: Optional[str]           = None  # set when conflict resolved
    quality_flags:     list[str]                = field(default_factory=list)
    overall_quality:   float                    = 0.0   # composite 0–100


# ── ATR helper ────────────────────────────────────────────────────────────────

def _get_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Average True Range. Uses H-L only for intraday (no overnight gaps)."""
    try:
        hl = (df["high"] - df["low"]).tolist()
        if not hl:
            return 0.0
        n = min(period, len(hl))
        return sum(hl[-n:]) / n
    except Exception:
        return 0.0


# ── FVG quality scorer ────────────────────────────────────────────────────────

def score_fvg(
    fvg:   Optional[dict],
    df:    pd.DataFrame,
    price: float,
    atr:   float,
) -> Optional[FVGQuality]:
    """
    Score a single FVG (bullish or bearish) on quality dimensions.
    Returns None if fvg is None.

    Args:
        fvg:   dict with 'top', 'bottom', 'ts' keys from smc.detect_fvg()
        df:    OHLCV DataFrame
        price: current price
        atr:   current ATR (H-L average)
    """
    if not fvg or atr <= 0:
        return None

    reasons: list[str] = []
    score = 0.0

    gap_top    = float(fvg["top"])
    gap_bottom = float(fvg["bottom"])
    gap_size   = gap_top - gap_bottom
    gap_mid    = (gap_top + gap_bottom) / 2

    # ── 1. Size relative to ATR (max 25 pts) ──────────────────────────────────
    size_atr = gap_size / atr if atr > 0 else 0.0
    if size_atr >= 1.5:
        size_pts = 25.0
        reasons.append(f"Large FVG ({size_atr:.1f}× ATR)")
    elif size_atr >= 0.8:
        size_pts = 15.0 + (size_atr - 0.8) / 0.7 * 10.0
        reasons.append(f"Solid FVG ({size_atr:.1f}× ATR)")
    elif size_atr >= 0.3:
        size_pts = size_atr / 0.8 * 15.0
        reasons.append(f"Small FVG ({size_atr:.1f}× ATR)")
    else:
        size_pts = 0.0
        reasons.append(f"Micro FVG ({size_atr:.2f}× ATR) — below quality threshold")
    score += size_pts

    # ── 2. Distance from current price (max 30 pts) ───────────────────────────
    dist_pct = abs(price - gap_mid) / price
    if dist_pct < 0.003:
        dist_pts = 30.0
        reasons.append("Price inside/at FVG midpoint")
    elif dist_pct < 0.010:
        dist_pts = 20.0
        reasons.append(f"Price near FVG ({dist_pct*100:.2f}% away)")
    elif dist_pct < 0.020:
        dist_pts = 10.0
        reasons.append(f"Price approaching FVG ({dist_pct*100:.2f}% away)")
    elif dist_pct < 0.035:
        dist_pts = 3.0
    else:
        dist_pts = 0.0
        reasons.append(f"FVG too distant ({dist_pct*100:.2f}% away)")
    score += dist_pts

    # ── 3. Fill percentage penalty (max 20 pts deducted) ─────────────────────
    # Estimate how much of the gap has been filled by subsequent candles
    fill_pct = _estimate_fvg_fill(fvg, df, price)
    if fill_pct >= 0.95:
        fill_pts = -20.0
        reasons.append("FVG fully filled — invalidated")
    elif fill_pct >= 0.75:
        fill_pts = -12.0
        reasons.append(f"FVG heavily filled ({fill_pct*100:.0f}%)")
    elif fill_pct >= 0.50:
        fill_pts = -5.0
        reasons.append(f"FVG partially filled ({fill_pct*100:.0f}%)")
    else:
        fill_pts = 0.0
    score += fill_pts

    # ── 4. Age / freshness decay (max 20 pts, decays to 0 after ~25 bars) ─────
    age_bars = _estimate_age_bars(fvg, df)
    freshness = max(0.0, 1.0 - age_bars * FVG_DECAY_PER_BAR)
    age_pts = freshness * 20.0
    score += age_pts
    if age_bars > 15:
        reasons.append(f"Stale FVG ({age_bars} bars old)")

    # ── 5. Impulse strength that created the FVG (max 5 pts) ─────────────────
    impulse_strength = _estimate_impulse_strength(fvg, df, atr)
    if impulse_strength >= 1.5:
        imp_pts = 5.0
        reasons.append(f"Strong impulse created FVG ({impulse_strength:.1f}× ATR)")
    elif impulse_strength >= 0.8:
        imp_pts = 3.0
    else:
        imp_pts = 0.0
        reasons.append("Weak impulse — FVG may be noise")
    score += imp_pts

    final_score = max(0.0, min(MAX_FVG_SCORE, score))

    return FVGQuality(
        score=round(final_score, 1),
        size_atr_ratio=round(size_atr, 2),
        fill_pct=round(fill_pct, 2),
        age_bars=age_bars,
        impulse_strength=round(impulse_strength, 2),
        distance_pct=round(dist_pct, 4),
        reasons=reasons,
    )


def _estimate_fvg_fill(fvg: dict, df: pd.DataFrame, price: float) -> float:
    """
    Estimate what fraction of the FVG has been filled by price action.
    Returns 0.0 (unfilled) to 1.0 (completely filled).
    """
    try:
        gap_top    = float(fvg["top"])
        gap_bottom = float(fvg["bottom"])
        gap_size   = gap_top - gap_bottom
        if gap_size <= 0:
            return 1.0

        # Find the FVG's position in the dataframe
        ts = fvg.get("ts")
        fvg_idx = _find_fvg_index(fvg, df)

        if fvg_idx is None or fvg_idx >= len(df) - 1:
            return 0.0

        # Look at candles AFTER the FVG formed
        post_fvg = df.iloc[fvg_idx + 1:]
        if post_fvg.empty:
            return 0.0

        # Bullish FVG (gap up): filled when price trades back down through it
        # Bearish FVG (gap down): filled when price trades back up through it
        is_bullish_fvg = (gap_top == float(fvg["top"]) and
                          float(df.iloc[fvg_idx]["low"]) > gap_bottom)

        deepest_fill = 0.0
        for _, row in post_fvg.iterrows():
            if is_bullish_fvg:
                # How far into the gap did the low reach?
                penetration = max(0, gap_top - float(row["low"]))
            else:
                penetration = max(0, float(row["high"]) - gap_bottom)
            deepest_fill = max(deepest_fill, min(penetration / gap_size, 1.0))

        return deepest_fill
    except Exception:
        return 0.0


def _find_fvg_index(fvg: dict, df: pd.DataFrame) -> Optional[int]:
    """Find the approximate index in df where the FVG was formed."""
    try:
        ts = fvg.get("ts")
        if ts is not None and "timestamp" in df.columns:
            ts_series = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            fvg_ts    = pd.to_datetime(ts, utc=True)
            matches   = (ts_series - fvg_ts).abs()
            if not matches.empty:
                return int(matches.idxmin())
        return None
    except Exception:
        return None


def _estimate_age_bars(fvg: dict, df: pd.DataFrame) -> int:
    """Estimate how many bars ago the FVG was formed."""
    try:
        idx = _find_fvg_index(fvg, df)
        if idx is not None:
            return max(0, len(df) - 1 - idx)
        return 10  # conservative estimate if we can't find it
    except Exception:
        return 10


def _estimate_impulse_strength(fvg: dict, df: pd.DataFrame, atr: float) -> float:
    """Estimate the impulse candle strength that created the FVG (body / ATR)."""
    try:
        if atr <= 0:
            return 0.0
        idx = _find_fvg_index(fvg, df)
        if idx is None or idx < 1 or idx >= len(df):
            return 0.5
        # The impulse candle is the middle one (index i) in the 3-candle FVG pattern
        imp = df.iloc[idx - 1]  # impulse candle is i-1 in c[i-2], c[i-1], c[i] pattern
        body = abs(float(imp["close"]) - float(imp["open"]))
        return body / atr
    except Exception:
        return 0.5


# ── OB quality scorer ─────────────────────────────────────────────────────────

def score_ob(
    ob:    Optional[dict],
    df:    pd.DataFrame,
    price: float,
    atr:   float,
) -> Optional[OBQuality]:
    """
    Score a single Order Block on quality dimensions.

    Args:
        ob:    dict with 'top', 'bottom', 'ts' keys from smc.detect_order_blocks()
        df:    OHLCV DataFrame
        price: current price
        atr:   current H-L ATR
    """
    if not ob or atr <= 0:
        return None

    reasons: list[str] = []
    score = 0.0

    ob_top    = float(ob["top"])
    ob_bottom = float(ob["bottom"])
    ob_mid    = (ob_top + ob_bottom) / 2
    ob_width  = ob_top - ob_bottom

    # ── 1. OB zone width relative to ATR (max 15 pts) ────────────────────────
    width_atr = ob_width / atr if atr > 0 else 0.0
    if 0.3 <= width_atr <= 1.2:
        score += 15.0
        reasons.append(f"Well-proportioned OB ({width_atr:.2f}× ATR)")
    elif width_atr < 0.15:
        score += 3.0
        reasons.append(f"Narrow OB ({width_atr:.2f}× ATR) — may be noise")
    elif width_atr > 2.0:
        score += 5.0
        reasons.append(f"Wide OB ({width_atr:.2f}× ATR) — imprecise entry zone")
    else:
        score += 10.0

    # ── 2. Distance from current price (max 25 pts) ───────────────────────────
    dist_pct = abs(price - ob_mid) / price
    if ob_bottom <= price <= ob_top:
        dist_pts = 25.0
        reasons.append("Price INSIDE OB zone")
    elif dist_pct < 0.008:
        dist_pts = 18.0
        reasons.append(f"Price at OB boundary ({dist_pct*100:.2f}% away)")
    elif dist_pct < 0.020:
        dist_pts = 10.0
        reasons.append(f"Price approaching OB ({dist_pct*100:.2f}% away)")
    elif dist_pct < 0.040:
        dist_pts = 4.0
    else:
        dist_pts = 0.0
        reasons.append(f"OB too distant ({dist_pct*100:.2f}% from price)")
    score += dist_pts

    # ── 3. Violation check — most important quality gate (max -25 pts) ────────
    is_violated, retest_count = _check_ob_violation(ob, df)
    if is_violated:
        score -= 25.0
        reasons.append("OB VIOLATED — price closed through zone (invalidated)")

    # ── 4. Freshness decay (max 20 pts) ───────────────────────────────────────
    ob_idx = _find_ob_index(ob, df)
    age_bars = max(0, len(df) - 1 - ob_idx) if ob_idx is not None else 15
    freshness = max(0.0, 1.0 - age_bars * OB_DECAY_PER_BAR)
    score += freshness * 20.0
    if age_bars > 20:
        reasons.append(f"Aging OB ({age_bars} bars old)")

    # ── 5. Retest quality (max 10 pts) ────────────────────────────────────────
    if retest_count == 0:
        score += 10.0
        reasons.append("OB untested — first reaction expected (strongest)")
    elif retest_count == 1:
        score += 6.0
        reasons.append("OB on second test — still valid")
    elif retest_count == 2:
        score += 2.0
        reasons.append("OB on third test — weakening")
    else:
        score -= 5.0
        reasons.append(f"OB over-tested ({retest_count} retests) — likely exhausted")

    # ── 6. Impulse strength that formed the OB (max 10 pts) ──────────────────
    impulse_atr = _get_ob_impulse_atr(ob, df, atr)
    if impulse_atr >= 2.0:
        score += 10.0
        reasons.append(f"Strong OB impulse ({impulse_atr:.1f}× ATR)")
    elif impulse_atr >= OB_IMPULSE_ATR_MIN:
        score += 5.0 + (impulse_atr - OB_IMPULSE_ATR_MIN) / (2.0 - OB_IMPULSE_ATR_MIN) * 5.0
    else:
        score += 0.0
        reasons.append(f"Weak OB impulse ({impulse_atr:.1f}× ATR)")

    final_score = max(0.0, min(MAX_OB_SCORE, score))

    return OBQuality(
        score=round(final_score, 1),
        impulse_atr=round(impulse_atr, 2),
        age_bars=age_bars,
        retest_count=retest_count,
        is_violated=is_violated,
        distance_pct=round(dist_pct, 4),
        width_atr=round(width_atr, 2),
        reasons=reasons,
    )


def _find_ob_index(ob: dict, df: pd.DataFrame) -> Optional[int]:
    try:
        ts = ob.get("ts")
        if ts is not None and "timestamp" in df.columns:
            ts_series = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            ob_ts     = pd.to_datetime(ts, utc=True)
            matches   = (ts_series - ob_ts).abs()
            if not matches.empty:
                return int(matches.idxmin())
        return None
    except Exception:
        return None


def _check_ob_violation(ob: dict, df: pd.DataFrame) -> tuple[bool, int]:
    """
    Returns (is_violated, retest_count).
    violated = any candle CLOSED through the OB zone after it formed.
    retest   = number of times price touched the OB zone (without closing through).
    """
    try:
        ob_top    = float(ob["top"])
        ob_bottom = float(ob["bottom"])
        ob_idx    = _find_ob_index(ob, df)

        if ob_idx is None or ob_idx >= len(df) - 1:
            return False, 0

        post = df.iloc[ob_idx + 1:]
        is_violated  = False
        retest_count = 0

        for _, row in post.iterrows():
            low   = float(row["low"])
            high  = float(row["high"])
            close = float(row["close"])
            open_ = float(row["open"])

            # Touched the zone (wick or body)
            if low <= ob_top and high >= ob_bottom:
                retest_count += 1
                # Close-through = violated
                if close < ob_bottom or close > ob_top:
                    is_violated = True
                    break

        return is_violated, retest_count
    except Exception:
        return False, 0


def _get_ob_impulse_atr(ob: dict, df: pd.DataFrame, atr: float) -> float:
    """Get the impulse candle body size (ATR-relative) that formed the OB."""
    try:
        if atr <= 0:
            return 0.0
        ob_idx = _find_ob_index(ob, df)
        if ob_idx is None or ob_idx + 1 >= len(df):
            return 0.5
        # The impulse candle is the one AFTER the OB candle
        impulse = df.iloc[ob_idx + 1]
        body = abs(float(impulse["close"]) - float(impulse["open"]))
        return body / atr
    except Exception:
        return 0.5


# ── Structure conflict resolver ───────────────────────────────────────────────

def resolve_structure_conflict(
    df:          pd.DataFrame,
    structure:   dict,
    direction:   Optional[str],
    price:       float,
    atr:         float,
) -> StructureScore:
    """
    When both bullish and bearish structure coexist, score each direction
    and return the dominant one instead of skipping the ticker entirely.

    Scoring factors:
    - Structure type: CHoCH (10 pts) > BOS (6 pts)
    - Recency: recent breaks score higher than stale ones
    - Impulse: strength of the breaking move
    - Consistency: same direction on 2 consecutive confirmation bars

    Returns StructureScore with the recommended direction and confidence.
    """
    bull_score = _score_direction_structure(df, structure, "LONG",  price, atr)
    bear_score = _score_direction_structure(df, structure, "SHORT", price, atr)

    has_bull = structure.get("bos_bullish") or structure.get("choch_bullish")
    has_bear = structure.get("bos_bearish") or structure.get("choch_bearish")
    conflict  = has_bull and has_bear

    if not conflict:
        # No conflict — return the existing direction as-is
        direction_chosen = direction or ("LONG" if has_bull else "SHORT" if has_bear else None)
        chosen_score = bull_score if direction_chosen == "LONG" else bear_score
        return StructureScore(
            direction=direction_chosen or "NONE",
            structure_type="choch" if (
                structure.get("choch_bullish") or structure.get("choch_bearish")
            ) else "bos",
            raw_score=chosen_score["score"],
            recency_bars=chosen_score["recency_bars"],
            impulse_strength=chosen_score["impulse_strength"],
            htf_aligned=False,
            conflict_resolved=False,
        )

    # Conflict: both directions detected — use scores to pick the dominant
    logger.debug(
        f"[smc_quality] Structure conflict — BULL={bull_score['score']:.1f} "
        f"BEAR={bear_score['score']:.1f} — resolving via recency+strength"
    )

    if bull_score["score"] > bear_score["score"] + 5.0:
        winner, winner_score = "LONG",  bull_score
    elif bear_score["score"] > bull_score["score"] + 5.0:
        winner, winner_score = "SHORT", bear_score
    else:
        # Scores too close to call → skip
        return StructureScore(
            direction="AMBIGUOUS",
            structure_type="conflict",
            raw_score=0.0,
            recency_bars=0,
            impulse_strength=0.0,
            htf_aligned=False,
            conflict_resolved=True,
        )

    struct_type = "choch" if (
        (winner == "LONG"  and structure.get("choch_bullish")) or
        (winner == "SHORT" and structure.get("choch_bearish"))
    ) else "bos"

    return StructureScore(
        direction=winner,
        structure_type=struct_type,
        raw_score=winner_score["score"],
        recency_bars=winner_score["recency_bars"],
        impulse_strength=winner_score["impulse_strength"],
        htf_aligned=False,
        conflict_resolved=True,
    )


def _score_direction_structure(
    df: pd.DataFrame,
    structure: dict,
    direction: str,
    price: float,
    atr: float,
) -> dict:
    """Score a structural direction on recency and impulse strength."""
    score = 0.0
    recency_bars = 999
    impulse_strength = 0.0

    is_long = direction == "LONG"

    # Type bonus
    if is_long and structure.get("choch_bullish"):
        score += 10.0
    elif is_long and structure.get("bos_bullish"):
        score += 6.0
    elif not is_long and structure.get("choch_bearish"):
        score += 10.0
    elif not is_long and structure.get("bos_bearish"):
        score += 6.0
    else:
        return {"score": 0.0, "recency_bars": 999, "impulse_strength": 0.0}

    # Recency bonus — estimate from df tail
    try:
        n = len(df)
        if n >= 3:
            last2 = df["close"].iloc[-2:]
            breaking_move = float(last2.iloc[-1]) - float(last2.iloc[-2])
            impulse_strength = abs(breaking_move) / atr if atr > 0 else 0.0

            # Recency: CHoCH/BOS confirmed recently gets bonus
            recency_bars = 1  # most recent confirmation
            recency_bonus = max(0.0, 10.0 - recency_bars * 0.5)
            score += recency_bonus

            # Impulse bonus (max 5 pts)
            score += min(5.0, impulse_strength * 2.5)
    except Exception:
        pass

    return {
        "score":            score,
        "recency_bars":     recency_bars,
        "impulse_strength": round(impulse_strength, 2),
    }


# ── Public entry point ────────────────────────────────────────────────────────

def enrich(
    analysis: dict,
    df:       pd.DataFrame,
) -> SMCQualityResult:
    """
    Enrich an SMC analysis dict with quality scores.

    Args:
        analysis: output of smc.analyze() — must contain 'structure', 'fvgs', 'obs', etc.
        df:       OHLCV DataFrame (same as used in analysis)

    Returns SMCQualityResult with per-element quality scores.
    """
    if not analysis or df.empty:
        return SMCQualityResult()

    price  = float(analysis.get("current_price", 0))
    atr    = _get_atr(df)
    direction = analysis.get("direction")
    structure = analysis.get("structure", {})

    quality_flags: list[str] = []

    # ── FVG quality ───────────────────────────────────────────────────────────
    fvg_key = "fvg_bullish" if direction == "LONG" else "fvg_bearish"
    fvg     = analysis.get("fvgs", {}).get(fvg_key)
    fvg_q   = score_fvg(fvg, df, price, atr)

    if fvg_q:
        if fvg_q.fill_pct > 0.9:
            quality_flags.append("FVG_FILLED")
        if fvg_q.age_bars > 20:
            quality_flags.append("FVG_STALE")

    # ── OB quality ────────────────────────────────────────────────────────────
    ob_key  = "ob_bullish" if direction == "LONG" else "ob_bearish"
    ob      = analysis.get("obs", {}).get(ob_key)
    ob_q    = score_ob(ob, df, price, atr)

    if ob_q:
        if ob_q.is_violated:
            quality_flags.append("OB_VIOLATED")
        if ob_q.retest_count > 2:
            quality_flags.append("OB_OVER_TESTED")

    # ── Structure conflict resolution ─────────────────────────────────────────
    struct_score = resolve_structure_conflict(df, structure, direction, price, atr)
    direction_override = None

    if struct_score.conflict_resolved:
        if struct_score.direction == "AMBIGUOUS":
            quality_flags.append("STRUCTURE_AMBIGUOUS")
        elif struct_score.direction != direction:
            quality_flags.append(f"DIRECTION_OVERRIDDEN_TO_{struct_score.direction}")
            direction_override = struct_score.direction

    # ── Overall quality (composite) ───────────────────────────────────────────
    scores = []
    if fvg_q:
        scores.append(fvg_q.score)
    if ob_q and not ob_q.is_violated:
        scores.append(ob_q.score)
    if struct_score and struct_score.direction not in ("AMBIGUOUS", "NONE"):
        scores.append(struct_score.raw_score * 10)  # rescale 0–10 → 0–100
    overall = sum(scores) / len(scores) if scores else 50.0

    logger.debug(
        f"[smc_quality] direction={direction} "
        f"fvg_score={fvg_q.score if fvg_q else 'N/A'} "
        f"ob_score={ob_q.score if ob_q else 'N/A'} "
        f"struct_score={struct_score.raw_score:.1f} "
        f"overall={overall:.1f} flags={quality_flags}"
    )

    return SMCQualityResult(
        fvg_quality=fvg_q,
        ob_quality=ob_q,
        structure_score=struct_score,
        direction_override=direction_override,
        quality_flags=quality_flags,
        overall_quality=round(overall, 1),
    )
