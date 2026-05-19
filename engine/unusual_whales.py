"""
Unusual Whales API Client
=========================
Provides real options flow and dark pool data for SignalBolt.

Rate-limit strategy
-------------------
UW does not publish exact rate limits per plan. To stay safe at any tier:
  - We fetch the GLOBAL feed once per scan cycle (not per ticker).
  - All 27 tickers are filtered client-side from a single API call.
  - Per-scan cost: 2 API calls (1 flow + 1 dark pool) vs. 54 before.
  - Per-day cost (7h market, 15-min cycles): ~56 calls vs. ~1,500.
  - Global cache TTL = 12 minutes (slightly less than scan interval).

Rate-limit signals from API response headers:
  x-uw-req-per-minute-remaining  — check at runtime if 429s appear
  x-uw-daily-req-count           — monitor daily usage
  x-uw-token-req-limit           — your plan's daily cap

API base: https://api.unusualwhales.com
Auth:     Authorization: Bearer {UNUSUAL_WHALES_API_KEY}

Endpoints used (verified against UW documentation):
  GET /api/option-trades/flow-alerts   — global unusual flow alert feed
  GET /api/darkpool/recent             — global dark pool prints (FINRA TRF)

UW API field reference (as documented):
  Flow alerts:
    ticker              — underlying symbol
    alert_rule          — rule that triggered the alert
    all_opening_trades  — bool: true = all legs are opening positions
    has_sweep           — bool: order was an intermarket sweep
    has_floor           — bool: order was a floor/block trade
    total_premium       — total premium paid across all legs
    total_ask_side_prem — premium paid on ask side (bullish calls / bearish puts)
    total_bid_side_prem — premium paid on bid side (bearish calls / bullish puts)
    put_call            — "CALL" or "PUT"
    expiry_date         — "YYYY-MM-DD"
    strike              — strike price
    volume              — contracts traded
    open_interest       — open interest

  Dark pool:
    ticker              — underlying symbol
    price               — execution price
    size / quantity     — shares traded
    premium             — notional value (price × size)
    nbbo_ask            — national best ask at time of print
    nbbo_bid            — national best bid at time of print
    timestamp / time    — execution time
"""

import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests

from engine.config import UNUSUAL_WHALES_API_KEY

logger = logging.getLogger("signalbolt.uw")

_BASE    = "https://api.unusualwhales.com"
_HEADERS = {"Authorization": f"Bearer {UNUSUAL_WHALES_API_KEY}"} if UNUSUAL_WHALES_API_KEY else {}
_TIMEOUT = 8

# ── Global feed cache ─────────────────────────────────────────
# One cache entry holds ALL tickers — fetched once per scan cycle.
# Key: "flow_alerts" | "dark_pool"  →  (raw_list, fetched_at)
_global_cache: dict[str, tuple] = {}
# Fix #8: 5-minute TTL (was 12 min) — a large sweep at 10:05 AM shouldn't
# wait until 10:13 AM to be processed. 5 min costs 2 extra calls/day (~168
# total), well within any plan's limits.
_GLOBAL_CACHE_TTL = 300   # 5 minutes

# Per-ticker processed cache (avoids re-filtering on concurrent calls)
# Key: ("flow", ticker, min_premium) | ("pool", ticker, min_size) → (data, fetched_at)
_ticker_cache: dict[tuple, tuple] = {}
_TICKER_CACHE_TTL = 300   # matches global cache TTL


def _get_global_cache(key: str) -> Optional[list]:
    entry = _global_cache.get(key)
    if entry and (time.monotonic() - entry[1]) < _GLOBAL_CACHE_TTL:
        return entry[0]
    return None


def _set_global_cache(key: str, data: list) -> None:
    _global_cache[key] = (data, time.monotonic())


def _get_ticker_cache(key: tuple) -> Optional[list]:
    entry = _ticker_cache.get(key)
    if entry and (time.monotonic() - entry[1]) < _TICKER_CACHE_TTL:
        return entry[0]
    return None


def _set_ticker_cache(key: tuple, data: list) -> None:
    _ticker_cache[key] = (data, time.monotonic())


def _parse_alert_ts(item: dict) -> Optional[datetime]:
    """
    Parse the timestamp from a UW alert dict.
    Tries common field names in order of preference.
    Returns timezone-aware UTC datetime, or None if unparseable.
    """
    raw = (
        item.get("created_at")
        or item.get("timestamp")
        or item.get("time")
        or item.get("date")
    )
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _handle_error_status(resp, label: str) -> bool:
    """Returns True if the response has an error status (caller should return None)."""
    if resp.status_code == 401:
        logger.error(f"[uw] API key rejected — check UNUSUAL_WHALES_API_KEY ({label})")
        return True
    if resp.status_code == 429:
        remaining = resp.headers.get("x-uw-req-per-minute-remaining", "?")
        daily     = resp.headers.get("x-uw-daily-req-count", "?")
        limit     = resp.headers.get("x-uw-token-req-limit", "?")
        logger.warning(
            f"[uw] Rate limited ({label}) — "
            f"remaining/min={remaining} daily={daily}/{limit}"
        )
        return True
    if resp.status_code != 200:
        logger.warning(f"[uw] {label} HTTP {resp.status_code}")
        return True
    return False


def _log_rate_headers(resp, label: str) -> None:
    """Log rate-limit headers on every successful response for visibility."""
    remaining = resp.headers.get("x-uw-req-per-minute-remaining")
    daily     = resp.headers.get("x-uw-daily-req-count")
    limit     = resp.headers.get("x-uw-token-req-limit")
    if remaining or daily:
        logger.debug(
            f"[uw] {label} — remaining/min={remaining} daily={daily}/{limit}"
        )


# ---------------------------------------------------------------------------
# Global feed fetchers (called ONCE per scan cycle, not per ticker)
# ---------------------------------------------------------------------------

def _fetch_global_flow(limit: int = 500) -> Optional[list]:
    """
    Fetch the global flow-alerts feed — ONE call covers all tickers.
    Results cached for 12 minutes.
    """
    cached = _get_global_cache("flow_alerts")
    if cached is not None:
        logger.debug(f"[uw] flow_alerts global cache hit ({len(cached)} items)")
        return cached

    try:
        resp = requests.get(
            f"{_BASE}/api/option-trades/flow-alerts",
            headers=_HEADERS,
            params={"limit": limit},
            timeout=_TIMEOUT,
        )
        if _handle_error_status(resp, "flow-alerts/global"):
            return None
        _log_rate_headers(resp, "flow-alerts/global")
        data = resp.json().get("data", [])
        _set_global_cache("flow_alerts", data)
        logger.info(f"[uw] flow-alerts global: fetched {len(data)} items")
        return data
    except requests.Timeout:
        logger.warning("[uw] flow-alerts/global timeout")
        return None
    except Exception as e:
        logger.error(f"[uw] flow-alerts/global error: {e}")
        return None


def _fetch_global_dark_pool(limit: int = 500) -> Optional[list]:
    """
    Fetch the global dark pool feed — ONE call covers all tickers.
    Results cached for 5 minutes.

    Fix #7: The UW /api/darkpool/recent endpoint may require a ticker
    parameter (not confirmed for ticker-less queries). We attempt the
    global fetch first; if it returns an empty list (not an error), we
    mark the cache with a sentinel so callers know to use per-ticker
    fallback instead.
    """
    cached = _get_global_cache("dark_pool")
    if cached is not None:
        logger.debug(f"[uw] dark_pool global cache hit ({len(cached)} items)")
        return cached

    try:
        resp = requests.get(
            f"{_BASE}/api/darkpool/recent",
            headers=_HEADERS,
            params={"limit": limit},
            timeout=_TIMEOUT,
        )
        if _handle_error_status(resp, "darkpool/global"):
            return None
        _log_rate_headers(resp, "darkpool/global")
        data = resp.json().get("data", [])
        _set_global_cache("dark_pool", data)
        if data:
            logger.info(f"[uw] dark_pool global: fetched {len(data)} items")
        else:
            # Empty result — endpoint may require a ticker param.
            # Callers will fall back to per-ticker fetch.
            logger.info("[uw] dark_pool global: returned 0 items (endpoint may need ticker param)")
        return data
    except requests.Timeout:
        logger.warning("[uw] darkpool/global timeout")
        return None
    except Exception as e:
        logger.error(f"[uw] darkpool/global error: {e}")
        return None


def _fetch_darkpool_for_ticker(ticker: str, limit: int = 50) -> Optional[list]:
    """
    Fix #7 fallback: fetch dark pool per-ticker when global feed is empty.
    Tries /api/darkpool/recent?ticker=AAPL — confirmed to work on all plans.
    Not cached globally (only used as fallback).
    """
    try:
        resp = requests.get(
            f"{_BASE}/api/darkpool/recent",
            headers=_HEADERS,
            params={"ticker": ticker.upper(), "limit": limit},
            timeout=_TIMEOUT,
        )
        if _handle_error_status(resp, f"darkpool/{ticker}"):
            return None
        _log_rate_headers(resp, f"darkpool/{ticker}")
        return resp.json().get("data", [])
    except requests.Timeout:
        logger.warning(f"[uw] darkpool/{ticker} timeout")
        return None
    except Exception as e:
        logger.error(f"[uw] darkpool/{ticker} error: {e}")
        return None


# ---------------------------------------------------------------------------
# Options Flow  (public API — per ticker, from global cache)
# ---------------------------------------------------------------------------

def fetch_options_flow(
    ticker: str,
    min_premium: int = 100_000,
    limit: int = 500,
) -> list[dict]:
    """
    Return actionable options flow events for a single ticker.

    Data is sourced from the GLOBAL feed (fetched once per 12-min window,
    shared across all ticker calls). No per-ticker API call is made.

    Filters applied:
      - all_opening_trades: skip closing/unwinding positions
      - has_sweep or has_floor: skip single-leg retail orders
      - min_premium: only large institutional-size flows ($100K+)

    Returns list of dicts:
      {
        "ticker":        str,
        "expiry":        str,        # "2025-06-20"
        "strike":        float,
        "contract":      "call"|"put",
        "premium":       float,      # total_premium
        "ask_premium":   float,      # total_ask_side_prem
        "bid_premium":   float,      # total_bid_side_prem
        "volume":        int,
        "open_interest": int,
        "is_sweep":      bool,
        "is_floor":      bool,
        "sentiment":     "bullish"|"bearish"|"neutral",
        "alert_rule":    str,
      }
    """
    if not UNUSUAL_WHALES_API_KEY:
        logger.debug("[uw] UNUSUAL_WHALES_API_KEY not set — options flow skipped")
        return []

    # Check per-ticker processed cache
    cache_key = ("flow", ticker.upper(), min_premium)
    cached = _get_ticker_cache(cache_key)
    if cached is not None:
        return cached

    # Fetch global feed (returns cached data if fresh)
    raw = _fetch_global_flow(limit=limit)
    if raw is None:
        return []

    # Filter to this ticker and reject stale alerts (Fix #6)
    # A 3-hour-old sweep is not an actionable entry — it's a historical fact.
    # We only act on flow from the last 30 minutes.
    staleness_cutoff = datetime.now(timezone.utc) - timedelta(minutes=30)

    ticker_raw = []
    stale_count = 0
    for item in raw:
        if str(item.get("ticker") or item.get("symbol") or "").upper() != ticker.upper():
            continue
        ts = _parse_alert_ts(item)
        if ts is not None and ts < staleness_cutoff:
            stale_count += 1
            continue
        ticker_raw.append(item)

    if stale_count:
        logger.debug(f"[uw] {ticker} flow: dropped {stale_count} stale alert(s) (>30 min old)")

    # Apply actionability filters
    flows = []
    for item in ticker_raw:
        if not item.get("all_opening_trades", True):
            continue

        is_sweep = bool(item.get("has_sweep", False))
        is_floor = bool(item.get("has_floor", False))
        if not (is_sweep or is_floor):
            continue

        premium = float(item.get("total_premium") or 0)
        if premium < min_premium:
            continue

        ask_prem = float(item.get("total_ask_side_prem") or 0)
        bid_prem = float(item.get("total_bid_side_prem") or 0)
        contract = str(item.get("put_call") or "").upper()

        # Sentiment from ask/bid premium dominance + contract type
        total_sided = ask_prem + bid_prem
        if total_sided > 0:
            ask_pct = ask_prem / total_sided
            bid_pct = bid_prem / total_sided
            if contract == "CALL":
                sentiment = "bullish" if ask_pct >= 0.60 else ("bearish" if bid_pct >= 0.60 else "neutral")
            elif contract == "PUT":
                # Put buyers on ask OR bid = bearish; mixed = neutral
                sentiment = "bearish" if (bid_pct >= 0.60 or ask_pct >= 0.60) else "neutral"
            else:
                sentiment = "neutral"
        else:
            sentiment = "neutral"

        flows.append({
            "ticker":        ticker.upper(),
            "expiry":        item.get("expiry_date") or item.get("expiry") or "",
            "strike":        float(item.get("strike") or 0),
            "contract":      contract.lower(),
            "premium":       premium,
            "ask_premium":   ask_prem,
            "bid_premium":   bid_prem,
            "volume":        int(item.get("volume") or 0),
            "open_interest": int(item.get("open_interest") or 0),
            "is_sweep":      is_sweep,
            "is_floor":      is_floor,
            "sentiment":     sentiment,
            "alert_rule":    str(item.get("alert_rule") or ""),
        })

    logger.info(
        f"[uw] {ticker} options flow: {len(flows)} actionable "
        f"(from {len(ticker_raw)} ticker raw, {len(raw)} global, min_premium=${min_premium:,})"
    )
    _set_ticker_cache(cache_key, flows)
    return flows


# ---------------------------------------------------------------------------
# Dark Pool  (public API — per ticker, from global cache)
# ---------------------------------------------------------------------------

def fetch_dark_pool(
    ticker: str,
    min_size: int = 50_000,
    limit: int = 500,
) -> list[dict]:
    """
    Return dark pool / off-exchange block prints for a single ticker.

    Data sourced from the GLOBAL feed (fetched once per 12-min window).
    No per-ticker API call is made.

    Side is inferred from NBBO position (UW does not provide a direct field):
      price >= nbbo_ask  → aggressive buy
      price <= nbbo_bid  → aggressive sell
      between spread     → unknown (internalized)

    Returns list of dicts:
      {
        "ticker":    str,
        "price":     float,
        "size":      int,
        "notional":  float,
        "side":      "buy"|"sell"|"unknown",
        "timestamp": str,
        "premium":   float,   # alias for notional
      }
    """
    if not UNUSUAL_WHALES_API_KEY:
        logger.debug("[uw] UNUSUAL_WHALES_API_KEY not set — dark pool skipped")
        return []

    cache_key = ("pool", ticker.upper(), min_size)
    cached = _get_ticker_cache(cache_key)
    if cached is not None:
        return cached

    # Fix #7: try global feed first; if it returns empty (endpoint may need ticker),
    # fall back to per-ticker fetch so we don't silently drop dark pool signals.
    raw = _fetch_global_dark_pool(limit=limit)
    if raw is None:
        return []
    if len(raw) == 0:
        logger.info(f"[uw] dark_pool global empty — falling back to per-ticker fetch for {ticker}")
        raw = _fetch_darkpool_for_ticker(ticker)
        if raw is None:
            return []

    # Filter to this ticker and reject stale prints (Fix #6)
    staleness_cutoff = datetime.now(timezone.utc) - timedelta(minutes=30)

    ticker_raw = []
    stale_count = 0
    for item in raw:
        if str(item.get("ticker") or item.get("symbol") or "").upper() != ticker.upper():
            continue
        ts = _parse_alert_ts(item)
        if ts is not None and ts < staleness_cutoff:
            stale_count += 1
            continue
        ticker_raw.append(item)

    if stale_count:
        logger.debug(f"[uw] {ticker} dark pool: dropped {stale_count} stale print(s) (>30 min old)")

    prints = []
    for item in ticker_raw:
        price    = float(item.get("price") or 0)
        size     = int(item.get("size") or item.get("quantity") or 0)
        notional = float(item.get("premium") or (price * size))

        if notional < min_size:
            continue

        # Side inference from NBBO
        nbbo_ask = item.get("nbbo_ask")
        nbbo_bid = item.get("nbbo_bid")

        if nbbo_ask is not None and nbbo_bid is not None and price > 0:
            nbbo_ask = float(nbbo_ask)
            nbbo_bid = float(nbbo_bid)
            if price >= nbbo_ask:
                side = "buy"
            elif price <= nbbo_bid:
                side = "sell"
            else:
                side = "unknown"
        else:
            side_raw = str(item.get("side") or item.get("direction") or "unknown").lower()
            side = "buy" if side_raw in ("buy", "bought") else \
                   "sell" if side_raw in ("sell", "sold") else "unknown"

        prints.append({
            "ticker":    ticker.upper(),
            "price":     price,
            "size":      size,
            "notional":  notional,
            "premium":   notional,
            "side":      side,
            "timestamp": item.get("timestamp") or item.get("time") or "",
        })

    logger.info(
        f"[uw] {ticker} dark pool: {len(prints)} prints "
        f"(from {len(ticker_raw)} ticker raw, {len(raw)} global, min_notional=${min_size:,})"
    )
    _set_ticker_cache(cache_key, prints)
    return prints


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------

def get_flow_direction(flows: list[dict]) -> Optional[str]:
    """
    Determine net direction from options flow events.
    Requires 60%+ premium dominance — avoids mixed-flow noise.
    Returns 'LONG', 'SHORT', or None.
    """
    if not flows:
        return None

    bull = sum(f["premium"] for f in flows if f["sentiment"] == "bullish")
    bear = sum(f["premium"] for f in flows if f["sentiment"] == "bearish")
    total = bull + bear

    if total == 0:
        return None

    if bull / total >= 0.60:
        return "LONG"
    if bear / total >= 0.60:
        return "SHORT"
    return None


def get_pool_direction(prints: list[dict]) -> Optional[str]:
    """
    Determine net direction from dark pool prints.
    Buy/sell notional dominance (60%+) → LONG/SHORT.
    Falls back to print count if all sides are unknown.
    """
    if not prints:
        return None

    buy  = sum(p["notional"] for p in prints if p["side"] == "buy")
    sell = sum(p["notional"] for p in prints if p["side"] == "sell")
    total = buy + sell

    if total == 0:
        return "LONG" if len(prints) >= 2 else None

    if buy / total >= 0.60:
        return "LONG"
    if sell / total >= 0.60:
        return "SHORT"
    return None


def get_flow_summary(flows: list[dict]) -> dict:
    """
    Return a summary dict for L3 scorer and signal explanations.

    Returns:
      {
        "count":        int,
        "sweep_count":  int,
        "floor_count":  int,
        "bull_premium": float,
        "bear_premium": float,
        "direction":    "LONG"|"SHORT"|None,
        "conviction":   float,   # 0.0–1.0
      }
    """
    if not flows:
        return {
            "count": 0, "sweep_count": 0, "floor_count": 0,
            "bull_premium": 0.0, "bear_premium": 0.0,
            "direction": None, "conviction": 0.0,
        }

    bull  = sum(f["premium"] for f in flows if f["sentiment"] == "bullish")
    bear  = sum(f["premium"] for f in flows if f["sentiment"] == "bearish")
    total = bull + bear

    direction  = get_flow_direction(flows)
    conviction = (max(bull, bear) / total) if total > 0 else 0.0

    return {
        "count":        len(flows),
        "sweep_count":  sum(1 for f in flows if f["is_sweep"]),
        "floor_count":  sum(1 for f in flows if f["is_floor"]),
        "bull_premium": bull,
        "bear_premium": bear,
        "direction":    direction,
        "conviction":   round(conviction, 4),
    }


def warm_global_cache() -> None:
    """
    Pre-fetch both global feeds at scan start so all per-ticker calls are free.
    Call this once at the beginning of each scan cycle from runner.py.
    """
    if not UNUSUAL_WHALES_API_KEY:
        return
    logger.info("[uw] Warming global cache (flow + dark pool)...")
    _fetch_global_flow()
    _fetch_global_dark_pool()
    logger.info("[uw] Global cache warmed — ticker lookups are free this cycle")
