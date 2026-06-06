"""
Forward economic calendar for the Market Tape (CPI / jobs / FOMC / PCE / GDP …).

Tries Finnhub's economic-calendar endpoint (reuses the FINNHUB_API_KEY already
used by the earnings calendar). That endpoint may require a paid tier — so this
degrades gracefully: on any failure/empty it still surfaces the high-impact
session flags we already know (FOMC day, OpEx) via session_classifier. Cached;
never raises.

Env: FINNHUB_API_KEY (optional — without it / on free tier, only the flags show).
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx

from engine.cache import kv

logger = logging.getLogger("signalbolt.econ_calendar")

_FINNHUB = "https://finnhub.io/api/v1"
_TTL = 60 * 60 * 6              # 6h fresh
_FAIL_TTL = 60 * 30            # remember a failure 30m so we don't hammer


def _api_key() -> Optional[str]:
    return os.environ.get("FINNHUB_API_KEY", "").strip() or None


def _impact_word(v) -> str:
    s = str(v).lower()
    if s in ("3", "high"):
        return "high"
    if s in ("2", "medium"):
        return "medium"
    return "low"


def _finnhub_events(days: int) -> list[dict]:
    key = _api_key()
    if not key:
        return []
    today = datetime.now(timezone.utc).date()
    frm, to = today.isoformat(), (today + timedelta(days=days)).isoformat()
    ck = f"econ_cal:{frm}:{to}"
    cached = kv.get_json(ck)
    if cached is not None:
        return cached
    try:
        r = httpx.get(f"{_FINNHUB}/calendar/economic",
                      params={"from": frm, "to": to, "token": key}, timeout=8)
        if r.status_code != 200:
            kv.set_json(ck, [], _FAIL_TTL)
            return []
        rows = (r.json() or {}).get("economicCalendar") or []
        out = []
        for e in rows:
            if (e.get("country") or "").upper() not in ("US", "UNITED STATES", "USA"):
                continue
            imp = _impact_word(e.get("impact"))
            if imp == "low":
                continue
            out.append({"event": e.get("event"), "time": e.get("time"), "impact": imp,
                        "estimate": e.get("estimate"), "prev": e.get("prev"),
                        "actual": e.get("actual"), "source": "finnhub"})
        kv.set_json(ck, out, _TTL)
        return out
    except Exception as ex:
        logger.debug(f"[econ_calendar] finnhub failed: {ex}")
        kv.set_json(ck, [], _FAIL_TTL)
        return []


def _flag_events(now: datetime) -> list[dict]:
    """High-impact session flags we already track — always available, no feed."""
    out = []
    try:
        from engine import session_classifier as sc
        try:
            if sc._is_fomc_day():
                out.append({"event": "FOMC decision day", "impact": "high",
                            "time": None, "source": "flag"})
        except Exception:
            pass
        try:
            if sc._is_opex_day(now):
                out.append({"event": "Options expiration (OpEx)", "impact": "medium",
                            "time": None, "source": "flag"})
        except Exception:
            pass
    except Exception:
        pass
    return out


def today_and_upcoming(now: datetime | None = None, days: int = 7) -> dict:
    """{ today: [...], upcoming: [...], has_feed: bool }. Never raises."""
    try:
        now = now or datetime.now(timezone.utc)
        feed = _finnhub_events(days)
        flags = _flag_events(now)
        today_iso = now.date().isoformat()
        todays = [e for e in feed if (e.get("time") or "").startswith(today_iso)]
        return {"today": flags + todays, "upcoming": feed, "has_feed": bool(feed)}
    except Exception as e:
        logger.debug(f"[econ_calendar] today_and_upcoming failed: {e}")
        return {"today": [], "upcoming": [], "has_feed": False}
