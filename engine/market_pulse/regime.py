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
) -> str:
    """Return one of CONFIRMED_UPTREND | UNDER_PRESSURE | CORRECTION."""
    vix_on = vix_level is not None and vix_rising is not None

    # ── CORRECTION (breadth/selling only — VIX cannot create it) ──
    if (
        dd_max >= C.DD_CORRECTION
        or pct_above_200 < C.PCT200_CORRECTION
        or (net_nhnl < 0 and pct_above_50 < C.PCT50_WEAK)
    ):
        return C.CORRECTION

    # ── UNDER PRESSURE ──
    if (
        dd_max >= C.DD_PRESSURE
        or ad_divergence
        or pct_above_50 < C.PCT50_PRESSURE
        or net_nhnl < 0
        or (vix_on and vix_level > C.VIX_PRESSURE_LEVEL and vix_rising)   # soft confirmer
    ):
        return C.UNDER_PRESSURE

    # ── Otherwise CONFIRMED — with one boundary soft-downgrade ──
    if vix_on and dd_max == C.DD_BOUNDARY and vix_level > C.VIX_BOUNDARY_LEVEL and vix_rising:
        return C.UNDER_PRESSURE

    return C.CONFIRMED_UPTREND
