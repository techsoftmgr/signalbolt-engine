"""
Weekly earnings calendar fetcher (Finnhub free tier).

Why Finnhub:
  - Free 60 req/min, no credit card
  - Single call returns the whole week's earnings across all US tickers,
    including BMO/AMC time-of-day, EPS estimate, revenue estimate
  - Closest free analogue to Earnings Whispers (whose API is paid-only)

Why no engine logic on top:
  - This module is display-only. Signal scoring intentionally ignores
    earnings for now; the user wanted to see the calendar but keep the
    engine independent of it.

Cache:
  - 1h TTL via engine.cache (Redis when available, in-mem otherwise)
  - Cache key encodes the from/to dates so Mon/Tue/Wed all share data
  - On Finnhub failure we serve the last-good cache for an extra 2h
    rather than show an empty calendar

Env:
  FINNHUB_API_KEY — required; signup at finnhub.io (free tier)
                    If unset, the module returns an empty list and the
                    /earnings/calendar endpoint reports source="unavailable"
                    so the app can render a "set up Finnhub" hint.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import httpx

from engine.cache import kv

logger = logging.getLogger("signalbolt.earnings")

_FINNHUB_BASE = "https://finnhub.io/api/v1"
_CACHE_TTL_SEC = 60 * 60          # 1h fresh
_STALE_FALLBACK_TTL_SEC = 60 * 60 * 3   # serve up to 3h-old data on API failure


def _api_key() -> Optional[str]:
    key = os.environ.get("FINNHUB_API_KEY", "").strip()
    return key or None


def _week_window(today: Optional[date] = None) -> tuple[str, str]:
    """
    Return (from_iso, to_iso) covering Monday→Friday of the CURRENT week.

    "Current week" is Monday-anchored. If today is Sunday, we look at
    tomorrow's week (Mon–Fri ahead) — Sunday's calendar should already
    show the upcoming week, not the just-finished one.
    """
    today = today or datetime.now(timezone.utc).date()
    weekday = today.weekday()   # Mon=0, Sun=6
    if weekday == 6:            # Sunday → roll to tomorrow's week
        monday = today + timedelta(days=1)
    else:
        monday = today - timedelta(days=weekday)
    friday = monday + timedelta(days=4)
    return monday.isoformat(), friday.isoformat()


def _normalize_entry(raw: dict) -> dict:
    """
    Map a Finnhub earnings row to the shape the app consumes.

    Finnhub returns:
      { symbol, date, hour ("bmo"|"amc"|"dmh"|""),
        epsEstimate, epsActual, revenueEstimate, revenueActual, year, quarter }

    We collapse `hour` to a human label ("Before Open" / "After Close" /
    "During Market") and round revenue to $M for compact display.
    """
    hour = (raw.get("hour") or "").lower()
    when = {
        "bmo": "Before Open",
        "amc": "After Close",
        "dmh": "During Market",
    }.get(hour, "Unscheduled")

    rev_est_m = None
    if raw.get("revenueEstimate") is not None:
        try:
            rev_est_m = round(float(raw["revenueEstimate"]) / 1_000_000, 1)
        except (TypeError, ValueError):
            rev_est_m = None

    return {
        "ticker":        raw.get("symbol"),
        "date":          raw.get("date"),
        "when":          when,
        "eps_estimate":  raw.get("epsEstimate"),
        "revenue_est_m": rev_est_m,
        "quarter":       raw.get("quarter"),
        "year":          raw.get("year"),
    }


def get_weekly_earnings(tickers: Optional[list[str]] = None) -> dict:
    """
    Fetch this week's earnings calendar.

    Args:
      tickers: optional whitelist. If given, results are filtered to
               just those symbols (case-insensitive). None = full
               US calendar.

    Returns:
      {
        "from":    "2026-05-25",
        "to":      "2026-05-29",
        "source":  "finnhub" | "cache" | "stale-cache" | "unavailable",
        "fetched_at": "2026-05-26T03:59:47+00:00",
        "earnings": [ {ticker, date, when, eps_estimate, revenue_est_m, quarter, year}, ... ]
      }
    """
    from_iso, to_iso = _week_window()
    cache_key = f"earnings:week:{from_iso}:{to_iso}"

    # Fresh cache?
    cached = kv.get_json(cache_key)
    if cached:
        out = dict(cached)
        out["source"] = "cache"
        return _filter(out, tickers)

    key = _api_key()
    if not key:
        logger.info("[earnings] FINNHUB_API_KEY not set — returning empty calendar")
        return {
            "from":       from_iso,
            "to":         to_iso,
            "source":     "unavailable",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "earnings":   [],
        }

    url = f"{_FINNHUB_BASE}/calendar/earnings"
    params = {"from": from_iso, "to": to_iso, "token": key}
    try:
        with httpx.Client(timeout=8.0) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            payload = resp.json() or {}
    except Exception as e:
        # Network/HTTP failure — try to serve last-good stale cache so the
        # UI doesn't go blank on a transient Finnhub blip.
        stale = kv.get_json(f"{cache_key}:stale")
        if stale:
            logger.warning(f"[earnings] Finnhub fetch failed ({e}) — serving stale cache")
            out = dict(stale)
            out["source"] = "stale-cache"
            return _filter(out, tickers)
        logger.warning(f"[earnings] Finnhub fetch failed and no stale cache: {e}")
        return {
            "from":       from_iso,
            "to":         to_iso,
            "source":     "unavailable",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "earnings":   [],
        }

    rows = payload.get("earningsCalendar", []) or []
    normalized = [_normalize_entry(r) for r in rows if r.get("symbol")]
    # Sort by date then ticker for stable UI ordering
    normalized.sort(key=lambda r: (r.get("date") or "", r.get("ticker") or ""))

    result = {
        "from":       from_iso,
        "to":         to_iso,
        "source":     "finnhub",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "earnings":   normalized,
    }

    # Write to both the fresh and stale-fallback caches.
    kv.set_json(cache_key, result, ttl_sec=_CACHE_TTL_SEC)
    kv.set_json(f"{cache_key}:stale", result, ttl_sec=_STALE_FALLBACK_TTL_SEC)

    return _filter(result, tickers)


def get_next_earnings(ticker: str, horizon_days: int = 120) -> Optional[dict]:
    """
    Next UPCOMING earnings date for one ticker (looks ahead up to horizon_days),
    for the Ticker Hub volatility heads-up. Unlike get_weekly_earnings this is
    per-symbol and can reach beyond the current week.

    Returns {date, when, daysAway, eps_estimate} or None when unknown /
    no key / nothing scheduled in the window. Cached 12h per ticker.
    """
    sym = (ticker or "").upper().strip()
    if not sym:
        return None

    cache_key = f"earnings:next:{sym}"
    cached = kv.get_json(cache_key)
    if cached is not None:
        # cached may be {} (sentinel for "checked, nothing scheduled")
        return cached or None

    key = _api_key()
    if not key:
        return None

    today    = datetime.now(timezone.utc).date()
    from_iso = today.isoformat()
    to_iso   = (today + timedelta(days=max(7, horizon_days))).isoformat()
    url      = f"{_FINNHUB_BASE}/calendar/earnings"
    params   = {"from": from_iso, "to": to_iso, "symbol": sym, "token": key}
    try:
        with httpx.Client(timeout=8.0) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            payload = resp.json() or {}
    except Exception as e:
        logger.debug(f"[earnings] next({sym}) fetch failed: {e}")
        return None

    rows = payload.get("earningsCalendar", []) or []
    upcoming = sorted(
        [r for r in rows if r.get("date") and r["date"] >= from_iso],
        key=lambda r: r["date"],
    )
    if not upcoming:
        kv.set_json(cache_key, {}, ttl_sec=_CACHE_TTL_SEC)   # remember "none" briefly
        return None

    nxt = _normalize_entry(upcoming[0])
    try:
        d = datetime.strptime(nxt["date"], "%Y-%m-%d").date()
        nxt["daysAway"] = (d - today).days
    except Exception:
        nxt["daysAway"] = None

    out = {
        "date":         nxt.get("date"),
        "when":         nxt.get("when"),
        "daysAway":     nxt.get("daysAway"),
        "eps_estimate": nxt.get("eps_estimate"),
    }
    kv.set_json(cache_key, out, ttl_sec=_CACHE_TTL_SEC * 12)   # 12h
    return out


def _filter(result: dict, tickers: Optional[list[str]]) -> dict:
    if not tickers:
        return result
    wanted = {t.upper() for t in tickers if t}
    filtered = [r for r in result.get("earnings", []) if (r.get("ticker") or "").upper() in wanted]
    out = dict(result)
    out["earnings"] = filtered
    return out
