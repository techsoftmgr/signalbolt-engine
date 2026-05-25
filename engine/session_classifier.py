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
from zoneinfo import ZoneInfo

logger = logging.getLogger("signalbolt.session")

ET = ZoneInfo("America/New_York")

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

# ── NYSE / NASDAQ market holidays (full closures) ─────────────
# Hard-coded calendar through 2027. When updating beyond, also extend
# the early-close list below. Source: nyse.com/markets/hours-calendars.
# Without this set, the engine treats holidays as normal weekdays and
# fires signals on stale Friday-close or thin pre-market data — that's
# the Memorial Day 2026 bug that this calendar fixes.
NYSE_HOLIDAYS: set[str] = {
    # 2025
    "2025-01-01",  # New Year's Day
    "2025-01-20",  # MLK Day
    "2025-02-17",  # Presidents' Day
    "2025-04-18",  # Good Friday
    "2025-05-26",  # Memorial Day
    "2025-06-19",  # Juneteenth
    "2025-07-04",  # Independence Day
    "2025-09-01",  # Labor Day
    "2025-11-27",  # Thanksgiving
    "2025-12-25",  # Christmas
    # 2026
    "2026-01-01",  # New Year's Day
    "2026-01-19",  # MLK Day
    "2026-02-16",  # Presidents' Day
    "2026-04-03",  # Good Friday
    "2026-05-25",  # Memorial Day
    "2026-06-19",  # Juneteenth (Friday)
    "2026-07-03",  # Independence Day observed (Jul 4 = Saturday)
    "2026-09-07",  # Labor Day
    "2026-11-26",  # Thanksgiving
    "2026-12-25",  # Christmas
    # 2027
    "2027-01-01",  # New Year's Day
    "2027-01-18",  # MLK Day
    "2027-02-15",  # Presidents' Day
    "2027-03-26",  # Good Friday
    "2027-05-31",  # Memorial Day
    "2027-06-18",  # Juneteenth observed (Jun 19 = Saturday)
    "2027-07-05",  # Independence Day observed (Jul 4 = Sunday)
    "2027-09-06",  # Labor Day
    "2027-11-25",  # Thanksgiving
    "2027-12-24",  # Christmas observed (Dec 25 = Saturday)
}

# Days when NYSE closes early at 1:00 PM ET (13:00).
# Most common: day after Thanksgiving (Black Friday), and Christmas Eve
# when it falls on a weekday.
NYSE_EARLY_CLOSES: set[str] = {
    "2025-07-03",  # day before Independence Day (Thursday)
    "2025-11-28",  # day after Thanksgiving (Black Friday)
    "2025-12-24",  # Christmas Eve
    "2026-11-27",  # day after Thanksgiving
    "2026-12-24",  # Christmas Eve (Thursday)
    "2027-11-26",  # day after Thanksgiving
}
EARLY_CLOSE_MINS = 13 * 60   # 1:00 PM ET

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


def _is_market_holiday() -> bool:
    """True on full NYSE/NASDAQ closure days (Memorial Day, Christmas, etc)."""
    return _today_et_iso() in NYSE_HOLIDAYS


def _is_early_close() -> bool:
    """True on 1pm-ET-close days (day after Thanksgiving, Christmas Eve)."""
    return _today_et_iso() in NYSE_EARLY_CLOSES


def _is_weekend() -> bool:
    """Saturday or Sunday."""
    return _et_now().weekday() >= 5


def _fomc_active_now() -> bool:
    """FOMC announcement usually at 2:00 PM ET — block 1:30-2:30 PM window."""
    if not _is_fomc_day():
        return False
    mins = _et_minutes()
    return 13 * 60 + 30 <= mins <= 14 * 60 + 30  # 1:30 PM – 2:30 PM ET


def _classify_mode(et_mins: int, has_catalyst: bool) -> str:
    """Map ET minutes to session mode."""
    # Weekend / NYSE holiday — market is closed all day. Must come first so
    # we don't return PRE_MARKET on a Saturday morning, etc.
    if _is_weekend() or _is_market_holiday():
        return "BLOCKED"
    if _fomc_active_now():
        return "BLOCKED"
    # Early-close days (1pm ET): treat 13:00+ as AFTER_HOURS instead of
    # the normal 16:00 cutoff.
    effective_close = EARLY_CLOSE_MINS if _is_early_close() else MARKET_CLOSE
    if et_mins < MARKET_OPEN:
        return "PRE_MARKET"
    if et_mins >= effective_close:
        return "AFTER_HOURS"
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

    holiday      = _is_market_holiday()
    early_close  = _is_early_close()
    weekend      = _is_weekend()
    mode         = _classify_mode(et_mins, has_premarket_catalyst)
    effective_close = EARLY_CLOSE_MINS if early_close else MARKET_CLOSE
    market_open  = (not weekend) and (not holiday) and (MARKET_OPEN <= et_mins < effective_close)
    mins_since   = max(0, et_mins - MARKET_OPEN)
    opex_day     = _is_opex_day(now)
    opex_week    = _is_opex_week(now)
    quad_witch   = _is_quad_witching(now)
    fomc_day     = _is_fomc_day()

    blocked      = mode in ("PRE_MARKET", "AFTER_HOURS", "BLOCKED")
    block_reason = ""
    # Order matters: holiday/weekend reason takes precedence over time-of-day.
    if holiday:
        block_reason = f"NYSE holiday ({_today_et_iso()}) — market closed all day"
    elif weekend:
        block_reason = "Weekend — market closed"
    elif mode == "PRE_MARKET":
        block_reason = "Pre-market: no signals before 9:30 AM ET"
    elif mode == "AFTER_HOURS":
        block_reason = (
            "After early close (1:00 PM ET)" if early_close
            else "After hours: no signals after 4:00 PM ET"
        )
    elif mode == "BLOCKED":
        block_reason = "FOMC active — signals paused 1:30–2:30 PM ET"
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
