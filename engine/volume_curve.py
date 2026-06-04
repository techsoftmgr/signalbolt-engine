"""
Intraday volume curve — the fraction of a full RTH day's volume typically
completed by N minutes after the 9:30 ET open.

Derived empirically from ~340 ticker-days of 5-min bars (2026-06). Volume is
heavily FRONT-LOADED by the opening surge (~14% in the first 15 min), so
projecting today's volume-so-far to a full day with a naive `elapsed / 390`
massively OVER-states early-session relative volume — e.g. HOOD 2026-06-04 9:46am:
real ~0.8x opening volume was projected to a fake "2.3x" and fired a false
accumulation signal. Use this curve as the projection divisor instead → a valid
"relative volume at this time of day", correct at the open as well as midday.

SHARED single source of truth: used by heatmap_service (the movers/heatmap
DISPLAY) AND quant_score_service (the actual SIGNAL-firing volume_score that gates
accumulation / distribution / breakout / breakdown / turnaround / peak).
"""
from __future__ import annotations

# (minutes-since-open, cumulative fraction of the day's RTH volume)
_VOL_CURVE = [
    (0, 0.0), (5, 0.087), (10, 0.113), (15, 0.139), (20, 0.160), (30, 0.200),
    (45, 0.255), (60, 0.306), (90, 0.387), (120, 0.459), (180, 0.570),
    (240, 0.670), (300, 0.768), (360, 0.880), (390, 1.0),
]


def expected_volume_fraction(elapsed_min: float) -> float:
    """Fraction of a full RTH day's volume typically done by `elapsed_min` minutes
    after the open (linear-interpolated empirical curve). Floored at the 5-min mark
    (~8.7%) to avoid div-by-zero + tame first-bar noise; 1.0 at/after the close."""
    if elapsed_min >= 390:
        return 1.0
    if elapsed_min <= 5:
        return _VOL_CURVE[1][1]
    for (m0, f0), (m1, f1) in zip(_VOL_CURVE, _VOL_CURVE[1:]):
        if m0 <= elapsed_min <= m1:
            t = (elapsed_min - m0) / (m1 - m0) if m1 > m0 else 0.0
            return f0 + t * (f1 - f0)
    return 1.0
