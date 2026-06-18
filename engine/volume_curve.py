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

from datetime import datetime
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")
_OPEN_MIN, _CLOSE_MIN = 9 * 60 + 30, 16 * 60   # 9:30 / 16:00 ET

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


def _et_dates(intraday_df):
    """ET calendar date of each bar (handles a tz-aware UTC index or a naive one)."""
    idx = intraday_df.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(_ET)
    else:
        idx = idx.tz_localize("UTC").tz_convert(_ET)
    return idx.date


def latest_session_bars(intraday_df):
    """Slice intraday_df to the MOST RECENT ET trading session present — keyed to the ET
    session date, NOT the UTC calendar date. The UTC date rolls over at 8 PM ET, so a
    UTC-date filter silently loses the day's bars overnight (relvol/VWAP reset to nothing).
    Returns the input unchanged on any failure."""
    try:
        d = _et_dates(intraday_df)
        if len(d) == 0:
            return intraday_df
        return intraday_df[d == max(d)]
    except Exception:
        return intraday_df


def session_relvol(intraday_df, avg_vol, now=None) -> float:
    """Pace-adjusted relative volume keyed to the ET TRADING SESSION — fixes (a) the
    overnight reset-to-1.0 (UTC date rolled over at 8 PM ET → today's bars no longer
    matched → volume read 0 → fell back to 1.0x) and (b) the DST-hardcoded 13:30-UTC open.

    Projects today's volume-so-far via the front-loaded curve ONLY during live RTH of the
    current session; OUTSIDE RTH (after the close, overnight, pre-open, weekends) it returns
    the REALIZED session volume ÷ avg with NO projection — so the last session's reading
    carries instead of dropping to a flat 1.0x. Returns 1.0 only when there's genuinely no
    usable volume/avg. `now` (tz-aware) is injectable for tests."""
    try:
        if not avg_vol or avg_vol <= 0 or intraday_df is None or len(intraday_df) == 0:
            return 1.0
        now_et = (now or datetime.now(_ET)).astimezone(_ET)
        d = _et_dates(intraday_df)
        if len(d) == 0:
            return 1.0
        session_date = max(d)
        session_vol  = float(intraday_df[d == session_date]["volume"].sum())
        if session_vol <= 0:
            return 1.0
        cur_min  = now_et.hour * 60 + now_et.minute
        live_rth = (session_date == now_et.date() and now_et.weekday() < 5
                    and _OPEN_MIN <= cur_min < _CLOSE_MIN)
        frac = expected_volume_fraction(cur_min - _OPEN_MIN) if live_rth else 1.0
        return (session_vol / max(0.05, frac)) / avg_vol
    except Exception:
        return 1.0
