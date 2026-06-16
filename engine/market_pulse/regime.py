"""
Market Pulse — regime resolver. Rule-based tiers, top-down, first match wins.
NO numeric score is produced or shown. VIX is a SOFT confirmer only: it can nudge
a borderline read but can never, by itself, create a CORRECTION. If VIX is null,
its clauses are simply skipped (regime computes from pillars 1-4).
"""
from __future__ import annotations

from typing import Optional

from . import config as C


def resolve(
    *,
    dd_max: int,
    net_nhnl: int,
    pct_above_50: float,
    pct_above_200: float,
    ad_divergence: bool,
    vix_level: Optional[float] = None,
    vix_rising: Optional[bool] = None,
    follow_through: bool = False,
) -> str:
    """Return one of CONFIRMED_UPTREND | UNDER_PRESSURE | CORRECTION.

    `follow_through` (a fresh IBD follow-through day off a low) SOFT-UPGRADES the read one
    tier: it confirms the rally has resumed, so a strong rally off a low flips the regime
    without waiting for the distribution-day count to age out. It can't, by itself, fabricate
    a CONFIRMED read out of a deep CORRECTION (only lifts CORRECTION→UNDER_PRESSURE)."""
    vix_on = vix_level is not None and vix_rising is not None

    # ── CORRECTION (breadth/selling only — VIX cannot create it) ──
    if (
        dd_max >= C.DD_CORRECTION
        or pct_above_200 < C.PCT200_CORRECTION
        or (net_nhnl < 0 and pct_above_50 < C.PCT50_WEAK)
    ):
        regime = C.CORRECTION

    # ── UNDER PRESSURE ──
    elif (
        dd_max >= C.DD_PRESSURE
        or ad_divergence
        or pct_above_50 < C.PCT50_PRESSURE
        or net_nhnl < 0
        or (vix_on and vix_level > C.VIX_PRESSURE_LEVEL and vix_rising)   # soft confirmer
    ):
        regime = C.UNDER_PRESSURE

    # ── Otherwise CONFIRMED — with one boundary soft-downgrade ──
    elif vix_on and dd_max == C.DD_BOUNDARY and vix_level > C.VIX_BOUNDARY_LEVEL and vix_rising:
        regime = C.UNDER_PRESSURE
    else:
        regime = C.CONFIRMED_UPTREND

    # ── Follow-through-day soft-UPGRADE (IBD: a valid FTD off a low confirms the rally) ──
    if follow_through:
        if regime == C.UNDER_PRESSURE:
            return C.CONFIRMED_UPTREND
        if regime == C.CORRECTION:
            return C.UNDER_PRESSURE

    return regime
