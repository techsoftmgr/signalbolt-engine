"""
Market Tape push alerts (V2).

Broadcasts two things, with anti-spam by construction:
  • SOCIAL posts — market-moving posts (e.g. Trump via the social feed). Pushed
    ANY hour (they move futures overnight). Per-post dedup + COLD-START seeding so
    a deploy can't fire a burst of old posts.
  • MARKET events — major index-level tape events (gap / VWAP reclaim-lose / sharp
    move) during regular hours only. Watermark + per-day cap + per-event dedup.

Broadcast + pref-gated via push.send_social_alert / push.send_market_alert.
ENV-GATED by MARKET_ALERTS_ENABLED so it ships dark and is flipped on when ready.
Runs every ~3 min from runner.py. Best-effort; never raises.
"""
from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger("signalbolt.market_alerts")

_MKT_ALERT_TYPES = {"GAP", "VWAP", "MOVE", "MACD_CROSS"}
_MIN_SEV = 2
_SEEN_TTL = 16 * 3600
_DEDUP_TTL = 36 * 3600
_MAX_MARKET_PER_DAY = 6
_MAX_SOCIAL_PER_RUN = 5


def _enabled() -> bool:
    return (os.getenv("MARKET_ALERTS_ENABLED", "") or "").strip().lower() in ("1", "true", "yes", "on")


def _post_key(p: dict) -> str:
    raw = p.get("url") or ((p.get("author") or "") + "|" + (p.get("text") or "")[:60])
    return hashlib.md5(raw.encode("utf-8", "ignore")).hexdigest()[:16]


def _run_social(cache, push) -> dict:
    stats = {"social": 0, "seeded": 0}
    try:
        from engine import social_feed
        posts = social_feed.recent_posts(15)
    except Exception:
        return stats
    if not posts:
        return stats
    # Cold start: seed everything currently in the channel, push nothing.
    if cache.kv.get_json("mkt_social_init") is None:
        for p in posts:
            cache.kv.set_json(f"mkt_social:{_post_key(p)}", {"seen": True}, _DEDUP_TTL)
        cache.kv.set_json("mkt_social_init", {"v": 1}, 7 * 24 * 3600)
        stats["seeded"] = len(posts)
        return stats
    sent = 0
    for p in reversed(posts):                      # oldest → newest
        if sent >= _MAX_SOCIAL_PER_RUN:
            break
        key = f"mkt_social:{_post_key(p)}"
        if cache.kv.get_json(key):
            continue
        n = push.send_social_alert(p.get("author") or "Market-moving post",
                                   p.get("text") or "", p.get("url"))
        cache.kv.set_json(key, {"seen": True}, _DEDUP_TTL)
        sent += 1
        if n:
            stats["social"] += 1
        logger.info(f"[market_alerts] social '{(p.get('author') or '')[:20]}' -> pushed {n}")
    return stats


def _run_market(cache, push) -> dict:
    stats = {"market": 0, "seeded": 0}
    try:
        from engine import session_classifier
        if not session_classifier.is_market_open_now():
            return stats
    except Exception:
        return stats
    try:
        from engine import market_commentary
        res = market_commentary.build()
    except Exception:
        return stats
    events = [e for e in (res.get("events") or [])
              if e.get("type") in _MKT_ALERT_TYPES and int(e.get("severity") or 0) >= _MIN_SEV and e.get("time")]
    newest = max((e["time"] for e in events), default=None)
    seen = cache.kv.get_json("mkt_evt_seen")
    last = (seen or {}).get("t")
    if seen is None:
        if newest:
            cache.kv.set_json("mkt_evt_seen", {"t": newest}, _SEEN_TTL)
        stats["seeded"] = 1
        return stats
    fresh = sorted([e for e in events if last is None or e["time"] > last], key=lambda e: e["time"])
    if fresh:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        cap_key = f"mkt_evt_cap:{today}"
        sent_today = int((cache.kv.get_json(cap_key) or {}).get("n", 0))
        for e in fresh:
            if sent_today >= _MAX_MARKET_PER_DAY:
                break
            dk = f"mkt_evt:{e['type']}:{e['time']}"
            if cache.kv.get_json(dk):
                continue
            n = push.send_market_alert(f"📈 Market: {e.get('title')}", (e.get("detail") or "")[:200], e["type"])
            cache.kv.set_json(dk, {"seen": True}, _DEDUP_TTL)
            sent_today += 1
            if n:
                stats["market"] += 1
        cache.kv.set_json(cap_key, {"n": sent_today}, _DEDUP_TTL)
    if newest and newest != last:
        cache.kv.set_json("mkt_evt_seen", {"t": newest}, _SEEN_TTL)
    return stats


def run() -> dict:
    """Push new social posts (any hour) + major market events (RTH). Best-effort."""
    if not _enabled():
        return {"disabled": True}
    from engine import cache, push
    stats = {"social": 0, "market": 0, "seeded": 0}
    try:
        s = _run_social(cache, push)
        m = _run_market(cache, push)
        for k in stats:
            stats[k] = s.get(k, 0) + m.get(k, 0)
    except Exception as e:
        logger.debug(f"[market_alerts] run failed: {e}")
    logger.info(f"[market_alerts] {stats}")
    return stats
