"""
Social post feed for the Market Tape — market-moving posts (e.g. Trump's Truth
Social, mirrored to X and then to a Discord channel by TweetShift).

We do NOT scrape X/Truth ourselves. The user already runs X Premium → TweetShift
→ a Discord channel; this module simply READS that channel via the Discord API
(a bot token with read access). Clean, legal, and reuses the existing pipeline.

Env (all optional — without them this no-ops and the tape falls back to the
licensed-news POLICY stream):
  DISCORD_BOT_TOKEN       — bot token with View Channel + Read Message History
  SOCIAL_FEED_CHANNEL_ID  — the channel id TweetShift posts into
  SOCIAL_FEED_AUTHOR      — optional handle/name substring filter (e.g. realDonaldTrump)

Cached ~45s; best-effort; never raises.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import httpx

from engine.cache import kv

logger = logging.getLogger("signalbolt.social_feed")

_API = "https://discord.com/api/v10"
_TTL = 45
_FAIL_TTL = 120


def _config() -> Optional[tuple[str, str, str]]:
    token = os.environ.get("DISCORD_BOT_TOKEN", "").strip()
    channel = os.environ.get("SOCIAL_FEED_CHANNEL_ID", "").strip()
    author = os.environ.get("SOCIAL_FEED_AUTHOR", "").strip().lower()
    if not token or not channel:
        return None
    return token, channel, author


def is_configured() -> bool:
    return _config() is not None


def _parse_message(m: dict) -> Optional[dict]:
    """Normalize a Discord message (TweetShift embed, or plain content) → post."""
    try:
        text, author, url, ts = "", "", None, m.get("timestamp")
        embeds = m.get("embeds") or []
        if embeds:
            e = embeds[0]
            text = (e.get("description") or "").strip()
            url = e.get("url")
            au = e.get("author") or {}
            author = (au.get("name") or "").strip()
            ts = e.get("timestamp") or ts
            if not text:
                text = (e.get("title") or "").strip()
        if not text:
            text = (m.get("content") or "").strip()
        if not text:
            return None
        return {"id": m.get("id"), "text": text, "author": author or "post",
                "url": url, "created_at": ts}
    except Exception:
        return None


def recent_posts(limit: int = 15) -> list[dict]:
    """Most recent market-moving posts from the configured Discord channel.
    Returns [] when unconfigured or on any failure. Never raises."""
    cfg = _config()
    if not cfg:
        return []
    token, channel, author = cfg
    ck = f"social_feed:{channel}:{limit}"
    cached = kv.get_json(ck)
    if cached is not None:
        return cached
    try:
        r = httpx.get(f"{_API}/channels/{channel}/messages",
                      params={"limit": max(1, min(limit, 50))},
                      headers={"Authorization": f"Bot {token}"}, timeout=8)
        if r.status_code != 200:
            logger.debug(f"[social_feed] discord {r.status_code}")
            kv.set_json(ck, [], _FAIL_TTL)
            return []
        out = []
        for m in (r.json() or []):
            p = _parse_message(m)
            if not p:
                continue
            if author and author not in (p["author"] or "").lower():
                continue
            out.append(p)
        kv.set_json(ck, out, _TTL)
        return out
    except Exception as e:
        logger.debug(f"[social_feed] fetch failed: {e}")
        kv.set_json(ck, [], _FAIL_TTL)
        return []
