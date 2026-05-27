"""
Session Classifier
==================
Determines what type of trading session we are in and enforces
time-based signal rules.

Session modes:
  PRE_MARKET    — before 9:30 AM ET → no signals
  CATALYST_ONLY — 9:30-9:45 AM ET  → only if pre-market catalyst + volume
  ORB           — 9:45-10:00 AM ET → Opening Range Breakout mode
  STANDARD      — 10:00-3:30 PM ET → full signal engine
  CLOSE_ONLY    — 3:30-4:00 PM ET  → intraday only, no swing
  AFTER_HOURS   — after 4:00 PM ET → no signals
  BLOCKED       — FOMC/CPI/NFP active → no signals

Minimum confidence thresholds by session:
  CATALYST_ONLY → 85
  ORB           → 80
  STANDARD      → 70  (matches existing STRATEGY_THRESHOLDS)
  CLOSE_ONLY    → 80

Used by: runner.py (pre-scan gate), scorer.py (L7 bonus layer)
"""

import logging
from datetime import datetime, date, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo

import pandas_market_calendars as mcal

logger = logging.getLogger("signalbolt.session")

ET = ZoneInfo("America/New_York")

# NYSE calendar object — created once, queried per-day. The library handles
# all the regular + observed holidays + early closes + future schedule
# changes automatically, so we never need to update a hard-coded list again.
_NYSE = mcal.get_calendar("NYSE")

# ── ET minutes from midnight thresholds ──────────────────────
MARKET_OPEN   = 9 * 60 + 30   # 9:30  = 570
CATALYST_END  = 9 * 60 + 45   # 9:45  = 585
ORB_END       = 10 * 60        # 10:00 = 600
CLOSE_START   = 15 * 60 + 30  # 15:30 = 930
MARKET_CLOSE  = 16 * 60        # 16:00 = 960

# Minimum composite score per session
SESSION_THRESHOLDS = {
    "CATALYST_ONLY": 85,
    "ORB":           80,
    "STANDARD":      70,
    "CLOSE_ONLY":    80,
    "PRE_MARKET":    999,
    "AFTER_HOURS":   999,
    "BLOCKED":       999,
}

# ── FOMC / major macro calendar (rolling — update quarterly) ──
# Format: "YYYY-MM-DD"  (ET date of announcement)
FOMC_DATES = [
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
]

# NYSE holiday + early-close detection is now sourced from
# pandas_market_calendars (see _NYSE above). The library auto-tracks
# observed holidays, the rotating Good Friday date, Juneteenth,
# Black Friday early closes, and any future calendar changes — so we
# never need to hand-edit a yearly list again.

# Standard close fallback when the calendar lookup fails for any reason
EARLY_CLOSE_MINS = 13 * 60   # 1:00 PM ET (default early-close)


@lru_cache(maxsize=512)
def _session_for_date(iso_date: str) -> tuple[bool, int]:
    """
    Return (is_trading_day, close_mins_et) for a given YYYY-MM-DD string.

    close_mins_et is ET minutes from midnight when the market closes —
    960 (16:00) on normal days, 780 (13:00) on early-close days.
    Returns (False, 0) on weekends and holidays.

    Cached per-date — calendar lookups are cheap but happen many times
    per scan cycle. The cache key is the ET date string, so it auto-
    invalidates at midnight when a new day rolls over.
    """
    try:
        sched = _NYSE.schedule(start_date=iso_date, end_date=iso_date)
        if sched.empty:
            return (False, 0)
        # sched['market_close'] is a UTC timestamp — convert to ET minutes
        close_utc = sched.iloc[0]["market_close"]
        close_et  = close_utc.tz_convert(ET)
        close_mins = close_et.hour * 60 + close_et.minute
        return (True, close_mins)
    except Exception as e:
        logger.warning(f"[session] calendar lookup failed for {iso_date}: {e} — assuming trading day")
        # Fail open: if the calendar library breaks, don't silently halt trading.
        # The session classifier's own time-of-day gates still apply.
        return (True, 16 * 60)

# NFP always first Friday of month — we approximate
# CPI/PPI — 2nd or 3rd week — we use a simple heuristic


def _et_now() -> datetime:
    return datetime.now(ET)


def _et_minutes() -> int:
    now = _et_now()
    return now.hour * 60 + now.minute


def _is_opex_day(dt: datetime) -> bool:
    """3rd Friday of the month = monthly options expiration."""
    return dt.weekday() == 4 and 15 <= dt.day <= 21  # Friday + 3rd occurrence


def _is_opex_week(dt: datetime) -> bool:
    """Week containing 3rd Friday."""
    # Find what day of month the Friday of this week would be
    days_to_friday = (4 - dt.weekday()) % 7
    friday_date = dt.day + days_to_friday
    return 15 <= friday_date <= 21


def _is_quad_witching(dt: datetime) -> bool:
    """3rd Friday of March, June, September, December."""
    return _is_opex_day(dt) and dt.month in (3, 6, 9, 12)


def _is_fomc_day() -> bool:
    today = date.today().isoformat()
    return today in FOMC_DATES


def _today_et_iso() -> str:
    """ET-local 'YYYY-MM-DD' string. Holidays change at midnight ET, not UTC."""
    return _et_now().date().isoformat()


def _today_session() -> tuple[bool, int]:
    """(is_trading_day, close_mins_et) for today, ET."""
    return _session_for_date(_today_et_iso())


def _is_market_holiday() -> bool:
    """
    True if today is a NYSE holiday OR weekend (no trading session at all).
    Combines the old "holiday" and "weekend" checks since the library treats
    both as "no schedule for this date".
    """
    is_trading, _ = _today_session()
    return not is_trading


def _is_early_close() -> bool:
    """True on early-close days (default 1pm ET, day after Thanksgiving etc)."""
    is_trading, close_mins = _today_session()
    return is_trading and close_mins < 16 * 60


def _is_weekend() -> bool:
    """Saturday or Sunday."""
    return _et_now().weekday() >= 5


# ── Public helpers used by runner.py / signal_monitor.py ─────
def is_market_open_today() -> bool:
    """True if today has any trading session at all (any time of day)."""
    is_trading, _ = _today_session()
    return is_trading


def is_market_open_now() -> bool:
    """
    True if right now is within today's trading hours (9:30 AM ET to
    market close, accounting for early-close days). False on weekends,
    holidays, pre-market, and after-hours.
    """
    is_trading, close_mins = _today_session()
    if not is_trading:
        return False
    mins = _et_minutes()
    return MARKET_OPEN <= mins < close_mins


def today_close_mins_et() -> int:
    """Effective close time today in ET minutes-from-midnight (e.g. 960=16:00)."""
    _, close_mins = _today_session()
    return close_mins or (16 * 60)


def _fomc_active_now() -> bool:
    """FOMC announcement usually at 2:00 PM ET — block 1:30-2:30 PM window."""
    if not _is_fomc_day():
        return False
    mins = _et_minutes()
    return 13 * 60 + 30 <= mins <= 14 * 60 + 30  # 1:30 PM – 2:30 PM ET


# Low-win-rate hour windows (ET, in minutes from midnight).
# Determined from production data analysis 2026-05-26 over 172 closed signals:
#   10:00-11:00 ET = 35.0% WR (opening reversal chop, post-ORB fade)
#   14:00-15:00 ET = 30.0% WR (mid-afternoon Fed news / pre-EOD reversals)
# High-WR windows that we KEEP firing:
#   09:30-10:00 = 60.5% (ORB session)
#   11:00-12:00 = 50.0%
#   12:00-13:00 = 57.1%
#   13:00-14:00 = 75.0% (best window)
#   15:00-16:00 = 35.3% (already covered by CLOSE_ONLY at 15:30+)
LOW_WR_WINDOWS = [
    (10 * 60,      11 * 60),     # 10:00-11:00 ET
    (14 * 60,      15 * 60),     # 14:00-15:00 ET
]


def _in_low_wr_window(et_mins: int) -> bool:
    return any(start <= et_mins < end for start, end in LOW_WR_WINDOWS)


def _classify_mode(et_mins: int, has_catalyst: bool) -> str:
    """Map ET minutes to session mode."""
    # Weekend or NYSE holiday — market is closed all day. Must come first so
    # we don't return PRE_MARKET on a Saturday morning, etc.
    if _is_market_holiday():   # library covers weekends + holidays in one check
        return "BLOCKED"
    if _fomc_active_now():
        return "BLOCKED"
    # Today's effective close — 16:00 on normal days, 13:00 on Black Friday etc.
    effective_close = today_close_mins_et()
    if et_mins < MARKET_OPEN:
        return "PRE_MARKET"
    if et_mins >= effective_close:
        return "AFTER_HOURS"
    # Low-WR hour blocks — production data shows 30-35% WR vs 50-75% elsewhere.
    # Block both day_trade scheduler AND tick-event scans inside these windows.
    if _in_low_wr_window(et_mins):
        return "BLOCKED"
    if et_mins >= CLOSE_START:
        return "CLOSE_ONLY"
    if et_mins >= ORB_END:
        return "STANDARD"
    if et_mins >= CATALYST_END:
        return "ORB"
    # 9:30–9:45: only fire if pre-market catalyst exists
    return "CATALYST_ONLY" if has_catalyst else "BLOCKED"


def classify(has_premarket_catalyst: bool = False) -> dict:
    """
    Return full session snapshot.

    Returns:
        {
          "mode":              str,    # session mode
          "market_open":       bool,
          "minutes_since_open": int,
          "is_opex_day":       bool,
          "is_opex_week":      bool,
          "is_quad_witching":  bool,
          "is_fomc_day":       bool,
          "blocked":           bool,
          "block_reason":      str,
          "threshold":         int,   # min confidence score for this session
          "sl_adjustment":     float, # SL width multiplier
          "allows_swing":      bool,
        }
    """
    now     = _et_now()
    et_mins = now.hour * 60 + now.minute

    holiday      = _is_market_holiday()   # weekends + NYSE holidays
    weekend      = _is_weekend()
    early_close  = _is_early_close()
    mode         = _classify_mode(et_mins, has_premarket_catalyst)
    market_open  = is_market_open_now()
    mins_since   = max(0, et_mins - MARKET_OPEN)
    opex_day     = _is_opex_day(now)
    opex_week    = _is_opex_week(now)
    quad_witch   = _is_quad_witching(now)
    fomc_day     = _is_fomc_day()

    blocked      = mode in ("PRE_MARKET", "AFTER_HOURS", "BLOCKED")
    block_reason = ""
    # Order matters: holiday/weekend reason takes precedence over time-of-day.
    if weekend:
        block_reason = "Weekend — market closed"
    elif holiday:
        block_reason = f"NYSE holiday ({_today_et_iso()}) — market closed all day"
    elif mode == "PRE_MARKET":
        block_reason = "Pre-market: no signals before 9:30 AM ET"
    elif mode == "AFTER_HOURS":
        block_reason = (
            "After early close (1:00 PM ET)" if early_close
            else "After hours: no signals after 4:00 PM ET"
        )
    elif mode == "BLOCKED":
        # BLOCKED can be FOMC, low-WR hour, or a few other cases.
        # Choose the most specific reason.
        if fomc_day and _fomc_active_now():
            block_reason = "FOMC active — signals paused 1:30–2:30 PM ET"
        elif _in_low_wr_window(et_mins):
            block_reason = (
                f"Low win-rate hour ({now.strftime('%H:%M')} ET): historical WR <40% "
                f"in this window — see LOW_WR_WINDOWS"
            )
        else:
            block_reason = "Trading paused"
    elif mode == "CATALYST_ONLY" and not has_premarket_catalyst:
        block_reason = "9:30-9:45 AM: catalyst required — no pre-market sweep detected"

    # SL width adjustment
    sl_adj = 1.0
    if mode == "CATALYST_ONLY": sl_adj = 1.20
    elif mode == "ORB":         sl_adj = 1.10
    if opex_day:                sl_adj = max(sl_adj, 1.15)
    if quad_witch:              sl_adj = max(sl_adj, 1.20)

    allows_swing = mode == "STANDARD" and not opex_day

    result = {
        "mode":               mode,
        "market_open":        market_open,
        "minutes_since_open": mins_since,
        "is_opex_day":        opex_day,
        "is_opex_week":       opex_week,
        "is_quad_witching":   quad_witch,
        "is_fomc_day":        fomc_day,
        "blocked":            blocked,
        "block_reason":       block_reason,
        "threshold":          SESSION_THRESHOLDS.get(mode, 70),
        "sl_adjustment":      sl_adj,
        "allows_swing":       allows_swing,
    }

    logger.info(
        f"[session] {mode} | {'OPEN' if market_open else 'CLOSED'} | "
        f"{'OpEx ' if opex_day else ''}{'OpExWeek ' if opex_week else ''}"
        f"{'FOMC ' if fomc_day else ''}| threshold={result['threshold']}"
    )
    return result


def score_for_signal(session: dict, has_catalyst: bool, vol_multiple: float) -> float:
    """
    Return 0-100 score contribution for session quality.
    Used as L7 bonus in scorer.py.
    """
    mode = session.get("mode", "STANDARD")

    if mode in ("PRE_MARKET", "AFTER_HOURS", "BLOCKED"):
        return 0.0

    base = {
        "CATALYST_ONLY": 72.0 if has_catalyst else 20.0,
        "ORB":           74.0,
        "STANDARD":      87.0,
        "CLOSE_ONLY":    60.0,
    }.get(mode, 70.0)

    # Volume bonus (catalyst sessions)
    if mode == "CATALYST_ONLY":
        if vol_multiple >= 5: base += 15
        elif vol_multiple >= 3: base += 10
        elif vol_multiple >= 2: base += 5

    # OpEx penalties
    if session.get("is_opex_day"):   base -= 20
    elif session.get("is_opex_week"): base -= 10
    if session.get("is_quad_witching"): base -= 15

    return max(0.0, min(100.0, base))
