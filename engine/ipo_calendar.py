"""
IPO calendar — in-progress (upcoming) IPOs + recently priced IPOs.

Source: Polygon `/vX/reference/ipos` (uses the engine's existing POLYGON_API_KEY).
Read-only, cached, never raises — returns source="unavailable" + empty lists if the
key is missing so the app renders a hint instead of erroring.

Two views the app shows:
  • upcoming — ipo_status pending / rumor / direct_listing: expected listing date
    + price RANGE (lowest_offer_price – highest_offer_price)
  • priced   — ipo_status new (recently listed): the FINALIZED issue price + date
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger("signalbolt.ipo_calendar")

_CACHE_KEY = "markets:ipo:v1"
_CACHE_TTL = 6 * 3600          # IPO data moves slowly — refresh every 6h
_BASE = "https://api.polygon.io/vX/reference/ipos"

_UPCOMING_STATUS = {"pending", "rumor", "direct_listing_process"}


def _f(v):
    try:
        return round(float(v), 2) if v is not None else None
    except (TypeError, ValueError):
        return None


def _row(r: dict) -> dict:
    return {
        "ticker":      r.get("ticker"),
        "name":        r.get("issuer_name"),
        "date":        r.get("listing_date"),          # 'YYYY-MM-DD' or None (TBD)
        "price_low":   _f(r.get("lowest_offer_price")),
        "price_high":  _f(r.get("highest_offer_price")),
        "final_price": _f(r.get("final_issue_price")),
        "exchange":    r.get("primary_exchange"),
        "currency":    r.get("currency_code") or "USD",
        "status":      r.get("ipo_status"),
        "shares":      r.get("shares_outstanding") or r.get("max_shares_offered"),
    }


def _fetch(params: dict) -> list[dict]:
    """One Polygon IPOs page. Fails open to []."""
    key = os.environ.get("POLYGON_API_KEY", "")
    if not key:
        return []
    try:
        import httpx
        with httpx.Client(timeout=20) as c:
            resp = c.get(_BASE, params={**params, "apiKey": key},
                         headers={"User-Agent": "signalbolt"})
            resp.raise_for_status()
            return (resp.json() or {}).get("results") or []
    except Exception as e:
        logger.debug(f"[ipo] fetch failed {params}: {e}")
        return []


def get_ipo_calendar(force: bool = False) -> dict:
    """Upcoming + recently-priced IPOs. Cached 6h. Never raises."""
    from engine import cache
    if not force:
        try:
            cached = cache.kv.get_json(_CACHE_KEY)
            if cached:
                return cached
        except Exception:
            pass

    has_key = bool(os.environ.get("POLYGON_API_KEY"))
    upcoming_raw = _fetch({"ipo_status": "pending", "limit": 80})
    priced_raw = _fetch({"ipo_status": "new", "order": "desc", "sort": "listing_date", "limit": 80})

    upcoming = [_row(r) for r in upcoming_raw]
    # Date ascending; undated (TBD) sink to the bottom.
    upcoming.sort(key=lambda x: (x["date"] is None, x["date"] or ""))
    # Recently priced: only rows that actually have a finalized issue price.
    priced = [_row(r) for r in priced_raw if _f(r.get("final_issue_price")) is not None]

    out = {
        "available": has_key,
        "source": "polygon" if has_key else "unavailable",
        "upcoming": upcoming[:60],
        "priced": priced[:60],
        "updated": datetime.now(timezone.utc).isoformat(),
        "note": ("Upcoming = expected listing date + price range. "
                 "Priced = finalized issue price. Source: Polygon."),
    }
    try:
        cache.kv.set_json(_CACHE_KEY, out, _CACHE_TTL)
    except Exception:
        pass
    return out
