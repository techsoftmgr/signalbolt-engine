"""
earnings_guard.py — flatten a momentum position before its earnings print.

A daily-close trend system CANNOT protect against an overnight / earnings GAP —
the market is shut when the number drops, so no stop can fill. Holding a momentum
winner into its own earnings is the classic way to eat a −15% gap. This guard
closes the position on the last trading day BEFORE the earnings date, so we go into
the print flat.

- Kill-switched: EARNINGS_GUARD_ENABLED (default OFF).
- Lead configurable: EARNINGS_GUARD_DAYS = trading days of lead (default 1 → close
  the trading day before earnings; handles weekends, e.g. Mon earnings → close Fri).
- Fail-open: no earnings data / no Finnhub key → never closes anything.
- Earnings dates come from engine.earnings_service.get_next_earnings (Finnhub, cached).

Applied to the LIVE (smart-exit) arm only in momentum_monitor; the control arm holds
through, so the A/B measures whether skipping earnings actually helps net.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta, date as _date
from typing import Optional, Tuple

from engine import earnings_service


def enabled() -> bool:
    return os.environ.get("EARNINGS_GUARD_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")


def _lead_days() -> int:
    try:
        return max(1, int(os.environ.get("EARNINGS_GUARD_DAYS", "1")))
    except (TypeError, ValueError):
        return 1


def _add_weekdays(d: _date, n: int) -> _date:
    """d plus n business days (Mon-Fri); weekends skipped (holidays ignored — erring
    a day early near a holiday is safe for a gap guard)."""
    cur, added = d, 0
    while added < n:
        cur += timedelta(days=1)
        if cur.weekday() < 5:
            added += 1
    return cur


def should_close(ticker: str, today: Optional[_date] = None) -> Tuple[bool, Optional[dict]]:
    """(True, info) when `ticker` reports on/within `lead` trading days → flatten before
    the print. Fail-open: (False, info|None) when earnings are unknown or further out.
    `info` is the earnings_service payload {date, when, daysAway, eps_estimate}."""
    try:
        info = earnings_service.get_next_earnings(ticker)
    except Exception:
        return False, None
    if not info or not info.get("date"):
        return False, info
    try:
        ed = datetime.strptime(info["date"], "%Y-%m-%d").date()
    except Exception:
        return False, info
    today = today or datetime.now(timezone.utc).date()
    threshold = _add_weekdays(today, _lead_days())
    return (today <= ed <= threshold), info
