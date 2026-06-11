"""
Top Reddit post titles for a ticker — the literal "what the community is saying",
for the community tap-through detail screen.

Source: reddit.com public search JSON (r/wallstreetbets+stocks+investing). No auth.
BEST-EFFORT: Reddit increasingly rate-limits / 403s unauthenticated server IPs, so
this soft-skips (returns []) on any block — the detail screen still has news
headlines. Cached per-ticker so we don't hammer Reddit. Never raises.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("signalbolt.reddit_posts")

_SUBS = "wallstreetbets+stocks+investing+StockMarket"
_TTL = 900   # 15 min
_UA = "SignalBolt/1.0 (community buzz; +https://signalbolt.app)"


def top_titles(ticker: str, limit: int = 5) -> list[dict]:
    """Top recent post titles mentioning the ticker. Best-effort → [] on block/error."""
    tk = (ticker or "").upper().strip()
    if not tk:
        return []
    cache_key = f"reddit_posts:{tk}:v1"
    try:
        from engine import cache
        cached = cache.kv.get_json(cache_key)
        if cached is not None:
            return cached
    except Exception:
        cache = None  # type: ignore

    out: list[dict] = []
    try:
        import httpx
        url = f"https://www.reddit.com/r/{_SUBS}/search.json"
        params = {"q": tk, "restrict_sr": 1, "sort": "hot", "t": "week",
                  "limit": max(limit, 8), "include_over_18": "off"}
        with httpx.Client(timeout=12, headers={"User-Agent": _UA}) as c:
            r = c.get(url, params=params)
            if r.status_code in (401, 403, 429):
                logger.info(f"[reddit_posts] {tk} blocked ({r.status_code}) — soft-skip")
                return []
            r.raise_for_status()
            children = ((r.json() or {}).get("data") or {}).get("children") or []
        for ch in children:
            d = ch.get("data") or {}
            title = (d.get("title") or "").strip()
            if not title:
                continue
            out.append({
                "title":        title[:180],
                "subreddit":    d.get("subreddit"),
                "ups":          d.get("ups"),
                "num_comments": d.get("num_comments"),
                "url":          ("https://www.reddit.com" + d["permalink"]) if d.get("permalink") else None,
            })
        # most-discussed first
        out.sort(key=lambda x: (x.get("num_comments") or 0), reverse=True)
        out = out[:limit]
    except Exception as e:
        logger.debug(f"[reddit_posts] {tk} fetch failed: {e}")
        return []

    try:
        if cache is not None:
            cache.kv.set_json(cache_key, out, _TTL)
    except Exception:
        pass
    return out
