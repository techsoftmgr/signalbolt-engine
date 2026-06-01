"""
Social Sentiment Aggregator
===========================
Pulls trending-ticker data from public social-media aggregators and merges
them into a single ranked list for the Community tab.

Sources (free, no scraping, ToS-friendly):
  - Apewisdom    — aggregates r/wallstreetbets, r/stocks, r/options mentions
                   with sentiment scoring. Public JSON API, no auth.
  - StockTwits   — trending symbols + per-stream message volume. Public API.

Output is cached 10 minutes in Redis (engine.cache) — these APIs request
modest polling and the data doesn't change minute-to-minute.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests

from engine import cache

logger = logging.getLogger("signalbolt.social_sentiment")

CACHE_KEY = "social_sentiment:trending:v1"
CACHE_TTL = 600   # 10 minutes
HTTP_TIMEOUT = 8

APEWISDOM_URL  = "https://apewisdom.io/api/v1.0/filter/all-stocks/page/1"
STOCKTWITS_URL = "https://api.stocktwits.com/api/2/trending/symbols.json"

# StockTwits' public endpoint now sits behind Cloudflare bot-protection that
# blocks the requests library at the TLS-fingerprint level (returns a 403 HTML
# challenge page). A browser User-Agent does NOT get past it, and defeating the
# challenge would mean circumventing bot-detection — which we don't do. So this
# source is BEST-EFFORT: if it answers we use it; if it's blocked/throttled
# (403/429 or an HTML body) we soft-skip and serve Reddit-only for that cycle.
# The legit way to restore StockTwits is their official/authenticated API.
_STOCKTWITS_HEADERS  = {"User-Agent": "SignalBolt/1.0 (+https://signalbolt.app)"}
_STOCKTWITS_THROTTLE = {403, 429}


# ── Per-source fetchers (each returns ticker → partial record, or {} on error) ──

def _fetch_apewisdom() -> dict[str, dict]:
    """
    Apewisdom shape (truncated):
      {
        "results": [
          {
            "ticker": "NVDA", "name": "Nvidia",
            "mentions": 1234, "upvotes": 5678,
            "rank": 1, "rank_24h_ago": 3,
            "mentions_24h_ago": 800,
            "sentiment": "Bullish",
            "sentiment_score": "72"
          },
          ...
        ]
      }
    """
    try:
        r = requests.get(APEWISDOM_URL, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        results = r.json().get("results", []) or []
    except Exception as e:
        logger.warning(f"[social] apewisdom fetch failed: {e}")
        return {}

    out: dict[str, dict] = {}
    for row in results:
        t = (row.get("ticker") or "").upper()
        if not t:
            continue
        mentions  = int(row.get("mentions") or 0)
        prev      = int(row.get("mentions_24h_ago") or 0)
        change    = round(((mentions - prev) / prev * 100), 1) if prev > 0 else None
        rank      = int(row.get("rank") or 0)
        prev_rank = int(row.get("rank_24h_ago") or 0)
        rank_chg  = (prev_rank - rank) if prev_rank > 0 else None   # positive = moved up
        try:
            sent_score = float(row.get("sentiment_score") or 50) / 100.0   # → 0..1 bullish weight
        except Exception:
            sent_score = 0.5
        out[t] = {
            "ticker":            t,
            "name":              row.get("name", ""),
            "reddit_mentions":   mentions,
            "reddit_change_pct": change,
            "reddit_rank":       rank,
            "reddit_rank_change":rank_chg,
            "reddit_sentiment":  round(sent_score, 3),
            "reddit_sentiment_label": row.get("sentiment", ""),
            "sources":           {"reddit"},
        }
    return out


def _fetch_stocktwits() -> dict[str, dict]:
    """
    StockTwits trending shape (truncated):
      {
        "symbols": [
          {"symbol": "NVDA", "title": "Nvidia", "watchlist_count": 12345}, ...
        ]
      }
    """
    try:
        r = requests.get(STOCKTWITS_URL, timeout=HTTP_TIMEOUT, headers=_STOCKTWITS_HEADERS)
        if r.status_code in _STOCKTWITS_THROTTLE:
            logger.info(f"[social] stocktwits unavailable ({r.status_code}, Cloudflare-gated) — Reddit-only this cycle")
            return {}
        r.raise_for_status()
        symbols = r.json().get("symbols", []) or []   # HTML challenge → json() raises → soft-skip below
    except Exception as e:
        logger.info(f"[social] stocktwits fetch failed: {e} — Reddit-only this cycle")
        return {}

    out: dict[str, dict] = {}
    for i, s in enumerate(symbols):
        t = (s.get("symbol") or "").upper()
        if not t:
            continue
        out[t] = {
            "ticker":                t,
            "name":                  s.get("title", ""),
            "stocktwits_rank":       i + 1,
            "stocktwits_watchers":   int(s.get("watchlist_count") or 0),
            "sources":               {"stocktwits"},
        }
    return out


# ── Merge + score ──────────────────────────────────────────────────────────

def _merge(reddit: dict[str, dict], st: dict[str, dict]) -> list[dict]:
    """
    Merge per-ticker partials by symbol. Combined score:
      score = reddit_mentions (normalized) + stocktwits_rank_boost
    Symbols present in BOTH get a multiplier — they're trending more widely.
    """
    all_keys = set(reddit) | set(st)
    merged: list[dict] = []

    # Normalize mentions for scoring
    max_mentions = max((r.get("reddit_mentions", 0) for r in reddit.values()), default=1) or 1

    for t in all_keys:
        rec_r = reddit.get(t, {})
        rec_s = st.get(t, {})
        sources = (rec_r.get("sources", set()) | rec_s.get("sources", set()))
        rec: dict = {
            "ticker":            t,
            "name":              rec_r.get("name") or rec_s.get("name") or "",
            "sources":           sorted(sources),
        }
        # Reddit fields
        if rec_r:
            for k in ("reddit_mentions", "reddit_change_pct", "reddit_rank",
                      "reddit_rank_change", "reddit_sentiment", "reddit_sentiment_label"):
                rec[k] = rec_r.get(k)
        # StockTwits fields
        if rec_s:
            for k in ("stocktwits_rank", "stocktwits_watchers"):
                rec[k] = rec_s.get(k)
        # Combined score (drives sort order)
        m_norm = (rec_r.get("reddit_mentions", 0) / max_mentions) if max_mentions else 0
        st_boost = 0.0
        if rec_s:
            # StockTwits top-30 gets a 0..0.5 boost (higher = better rank)
            st_boost = max(0.0, 0.5 - (rec_s.get("stocktwits_rank", 999) / 60))
        cross_boost = 1.3 if len(sources) > 1 else 1.0
        rec["score"] = round((m_norm + st_boost) * cross_boost, 4)
        merged.append(rec)

    merged.sort(key=lambda r: r.get("score", 0), reverse=True)
    return merged


# ── Public API ─────────────────────────────────────────────────────────────

def get_trending(limit: int = 30, force: bool = False) -> dict:
    """
    Returns {"trending": [...], "last_updated": iso, "sources_used": [...]}.
    Cached 10 minutes in Redis; pass force=True to bypass.
    """
    if not force:
        try:
            cached = cache.kv.get_json(CACHE_KEY)
            if cached:
                return cached
        except Exception:
            pass

    t0 = time.time()
    reddit = _fetch_apewisdom()
    st     = _fetch_stocktwits()
    merged = _merge(reddit, st)
    sources_used = []
    if reddit: sources_used.append("apewisdom")
    if st:     sources_used.append("stocktwits")

    payload = {
        "trending":      merged[: max(1, min(limit, 100))],
        "last_updated":  __iso_now(),
        "sources_used":  sources_used,
        "fetch_ms":      int((time.time() - t0) * 1000),
    }
    try:
        cache.kv.set_json(CACHE_KEY, payload, ttl_sec=CACHE_TTL)
    except Exception:
        pass
    return payload


def __iso_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
