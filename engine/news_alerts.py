"""
Per-ticker news push alerts.

Notifies a user when a FRESH headline lands on a ticker THEY watch (e.g. ABAT's
DOE reinstatement). Watchlist-scoped + pref-gated ('news_alerts') via
push.send_news_alert. Anti-spam by construction:

  • One batched Alpaca news call for all watched tickers (get_multi_news).
  • Per-headline dedup + COLD-START seeding per ticker (a deploy / a newly-watched
    ticker seeds its current headlines silently — no burst of old news).
  • Freshness gate (only headlines within _FRESH_HOURS) + per-ticker/day cap +
    a global per-run cap.
  • ENV-gated by NEWS_ALERTS_ENABLED so it ships dark and is flipped on.

Runs ~every 5 min, ALL hours (catalysts drop premarket/overnight). Never raises.
"""
from __future__ import annotations

import hashlib
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone

logger = logging.getLogger("signalbolt.news_alerts")

_MAX_TICKERS = 80
_SEEN_TTL = 7 * 24 * 3600
_DEDUP_TTL = 36 * 3600
_FRESH_HOURS = 6
_MAX_PER_TICKER_DAY = 4
_MAX_PER_RUN = 12


def _enabled() -> bool:
    return (os.getenv("NEWS_ALERTS_ENABLED", "") or "").strip().lower() in ("1", "true", "yes", "on")


def _hkey(url: str | None, headline: str) -> str:
    return hashlib.md5(((url or headline) or "").encode("utf-8", "ignore")).hexdigest()[:16]


def _is_fresh(ts, now: datetime) -> bool:
    if not ts:
        return False
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return (now - dt).total_seconds() <= _FRESH_HOURS * 3600
    except Exception:
        return False


def run(sb, now: datetime | None = None) -> dict:
    """Push fresh headlines on watched tickers. Best-effort."""
    if not _enabled():
        return {"disabled": True}
    from engine import cache, push, alpaca_client

    stats = {"tickers": 0, "alerts": 0, "seeded": 0}
    now = now or datetime.now(timezone.utc)
    try:
        rows = sb.table("watchlist").select("ticker").execute().data or []
    except Exception as e:
        logger.error(f"[news_alerts] fetch watchlist failed: {e}")
        return stats

    watched = list({(r.get("ticker") or "").upper() for r in rows if r.get("ticker")})[:_MAX_TICKERS]
    stats["tickers"] = len(watched)
    if not watched:
        return stats
    watch_set = set(watched)

    try:
        news = alpaca_client.get_multi_news(watched, limit=50) or []
    except Exception as e:
        logger.debug(f"[news_alerts] get_multi_news failed: {e}")
        return stats

    by_tk: dict[str, list] = defaultdict(list)
    for item in news:                               # newest-first (sort=desc)
        for s in (item.get("symbols") or []):
            su = (s or "").upper()
            if su in watch_set:
                by_tk[su].append(item)

    today = now.strftime("%Y-%m-%d")
    sent_run = 0
    for tk, items in by_tk.items():
        try:
            cold = cache.kv.get_json(f"news_init:{tk}") is None
            cap_key = f"news_cap:{tk}:{today}"
            sent_today = int((cache.kv.get_json(cap_key) or {}).get("n", 0))
            for item in items:
                head = (item.get("headline") or "").strip()
                if not head:
                    continue
                key = f"news_seen:{tk}:{_hkey(item.get('url'), head)}"
                if cache.kv.get_json(key):
                    continue
                cache.kv.set_json(key, {"s": 1}, _SEEN_TTL)   # mark seen regardless
                if cold:
                    continue                                   # cold start: seed only
                if sent_run >= _MAX_PER_RUN or sent_today >= _MAX_PER_TICKER_DAY:
                    continue
                if not _is_fresh(item.get("created_at") or item.get("time"), now):
                    continue                                   # don't push stale headlines
                n = push.send_news_alert(tk, head, item.get("url"), sb=sb)
                sent_today += 1
                sent_run += 1
                if n:
                    stats["alerts"] += 1
                logger.info(f"[news_alerts] {tk} '{head[:40]}' -> pushed {n}")
            if cold:
                cache.kv.set_json(f"news_init:{tk}", {"v": 1}, _SEEN_TTL)
                stats["seeded"] += 1
            else:
                cache.kv.set_json(cap_key, {"n": sent_today}, _DEDUP_TTL)
        except Exception as e:
            logger.debug(f"[news_alerts] {tk} failed: {e}")

    logger.info(f"[news_alerts] {stats}")
    return stats
