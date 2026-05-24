"""
Community Signal Service — social engagement layer around signals.

Tables (added via supabase-social-migration.sql):
  signal_votes    (signal_id, user_id, vote_type, created_at)
  signal_comments (id, signal_id, user_id, content, created_at, is_flagged)
  signal_follows  (signal_id, user_id, created_at)

Community score formula:
  communityScore = bullishVotesPct * 0.4
                 + followCountNormalized * 0.3
                 + commentActivityNormalized * 0.2
                 + recentActivityBoost * 0.1

Displayed as "Community Interest" — never used as a trade signal.
All user IDs are validated via Supabase JWT before writing.
"""

import logging
import os
import time
from typing import Optional

logger = logging.getLogger("signalbolt.community")

# ── Community feed cache ───────────────────────────────────────────────────────
_feed_cache: list[dict] = []
_feed_ts: float         = 0.0
_FEED_TTL: int          = int(os.environ.get("COMMUNITY_CACHE_TTL", "30"))


def get_social_summary(signal_id: str, sb) -> dict:
    """
    Return social summary for a single signal.
    sb = Supabase client (service role, passed in from endpoint).
    """
    try:
        votes    = _get_votes(signal_id, sb)
        follows  = _get_follow_count(signal_id, sb)
        comments = _get_comment_count(signal_id, sb)
        score    = _community_score(votes, follows, comments)

        return {
            "signalId":        signal_id,
            "votes":           votes,
            "followCount":     follows,
            "commentCount":    comments,
            "communityScore":  score,
            "communityLabel":  _score_label(score),
        }
    except Exception as e:
        logger.error(f"[community] get_social_summary({signal_id}): {e}")
        return {"signalId": signal_id, "votes": {}, "followCount": 0, "commentCount": 0, "communityScore": 0}


def get_community_feed(sb, limit: int = 20) -> list[dict]:
    """
    Return the community feed: most followed / most discussed / rising interest signals.
    Cached for COMMUNITY_CACHE_TTL seconds.
    """
    global _feed_cache, _feed_ts

    now = time.monotonic()
    if now - _feed_ts < _FEED_TTL and _feed_cache:
        return _feed_cache[:limit]

    try:
        # Get active signals
        sig_res = (
            sb.table("signals")
            .select("id, ticker, direction, confidence_score, strategy_type, created_at")
            .eq("status", "active")
            .order("created_at", desc=True)
            .limit(50)
            .execute()
        )
        signals = sig_res.data or []
        if not signals:
            return []

        signal_ids = [s["id"] for s in signals]

        # Batch fetch social data
        votes_res = (
            sb.table("signal_votes")
            .select("signal_id, vote_type")
            .in_("signal_id", signal_ids)
            .execute()
        )
        follows_res = (
            sb.table("signal_follows")
            .select("signal_id")
            .in_("signal_id", signal_ids)
            .execute()
        )
        comments_res = (
            sb.table("signal_comments")
            .select("signal_id, created_at")
            .in_("signal_id", signal_ids)
            .order("created_at", desc=True)
            .execute()
        )

        # Aggregate
        vote_map:    dict[str, list] = {}
        follow_map:  dict[str, int]  = {}
        comment_map: dict[str, int]  = {}
        recent_map:  dict[str, str]  = {}  # signal_id → most recent comment time

        for v in (votes_res.data or []):
            vote_map.setdefault(v["signal_id"], []).append(v["vote_type"])
        for f in (follows_res.data or []):
            follow_map[f["signal_id"]] = follow_map.get(f["signal_id"], 0) + 1
        for c in (comments_res.data or []):
            sid = c["signal_id"]
            comment_map[sid] = comment_map.get(sid, 0) + 1
            if sid not in recent_map:
                recent_map[sid] = c["created_at"]

        # Max values for normalization
        max_follows  = max(follow_map.values(),  default=1)
        max_comments = max(comment_map.values(), default=1)

        enriched: list[dict] = []
        for sig in signals:
            sid   = sig["id"]
            votes = vote_map.get(sid, [])
            follows  = follow_map.get(sid, 0)
            comments = comment_map.get(sid, 0)

            # Tally votes
            bull_ct = votes.count("bullish")
            bear_ct = votes.count("bearish")
            watch_ct= votes.count("watching")
            total   = len(votes) or 1
            bull_pct= round(bull_ct / total * 100)

            # Recent activity boost (comment in last 10 min)
            recent_boost = 0
            if sid in recent_map:
                from datetime import datetime, timezone
                try:
                    rc = datetime.fromisoformat(recent_map[sid].replace("Z", "+00:00"))
                    delta = (datetime.now(timezone.utc) - rc).total_seconds()
                    recent_boost = max(0, 1 - delta / 600)  # 0-1 within 10 min
                except Exception:
                    pass

            score = _community_score_raw(
                bull_pct, follows, max_follows,
                comments, max_comments, recent_boost,
            )

            enriched.append({
                **sig,
                "votes": {
                    "bullish":  bull_ct,
                    "bearish":  bear_ct,
                    "watching": watch_ct,
                    "total":    len(votes),
                    "bullishPct": bull_pct,
                },
                "followCount":    follows,
                "commentCount":   comments,
                "communityScore": round(score),
                "communityLabel": _score_label(score),
                "lastActivity":   recent_map.get(sid),
            })

        # Sort by community score desc
        enriched.sort(key=lambda x: x["communityScore"], reverse=True)
        _feed_cache = enriched
        _feed_ts    = now
        return enriched[:limit]

    except Exception as e:
        logger.error(f"[community] get_community_feed: {e}")
        return _feed_cache[:limit]


def add_vote(signal_id: str, user_id: str, vote_type: str, sb) -> dict:
    """
    Upsert vote — one vote per user per signal, update allowed.
    vote_type: "bullish" | "bearish" | "watching"
    """
    if vote_type not in ("bullish", "bearish", "watching"):
        raise ValueError("Invalid vote_type")

    sb.table("signal_votes").upsert(
        {
            "signal_id": signal_id,
            "user_id":   user_id,
            "vote_type": vote_type,
        },
        on_conflict="signal_id,user_id",
    ).execute()

    # Invalidate feed cache so next fetch is fresh
    global _feed_ts
    _feed_ts = 0.0

    return get_social_summary(signal_id, sb)


def add_comment(signal_id: str, user_id: str, content: str, sb) -> dict:
    """
    Add a comment. Basic spam/length checks. Returns the new comment row.
    """
    content = content.strip()
    if len(content) < 3:
        raise ValueError("Comment too short")
    if len(content) > 500:
        raise ValueError("Comment exceeds 500 characters")

    # Spam: max 5 comments per user per signal
    existing = (
        sb.table("signal_comments")
        .select("id", count="exact")
        .eq("signal_id", signal_id)
        .eq("user_id", user_id)
        .execute()
    )
    if (existing.count or 0) >= 5:
        raise ValueError("Comment limit reached for this signal")

    res = (
        sb.table("signal_comments")
        .insert({
            "signal_id":  signal_id,
            "user_id":    user_id,
            "content":    content,
            "is_flagged": False,
        })
        .execute()
    )
    global _feed_ts
    _feed_ts = 0.0
    return res.data[0] if res.data else {}


def get_comments(signal_id: str, sb, limit: int = 20) -> list[dict]:
    """Return public comments for a signal (no user email exposed)."""
    res = (
        sb.table("signal_comments")
        .select("id, content, created_at, is_flagged")
        .eq("signal_id", signal_id)
        .eq("is_flagged", False)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return res.data or []


def toggle_follow(signal_id: str, user_id: str, sb) -> dict:
    """Follow/unfollow toggle. Returns {following: bool, followCount: int}."""
    existing = (
        sb.table("signal_follows")
        .select("signal_id")
        .eq("signal_id", signal_id)
        .eq("user_id", user_id)
        .execute()
    )
    if existing.data:
        sb.table("signal_follows").delete().eq("signal_id", signal_id).eq("user_id", user_id).execute()
        following = False
    else:
        sb.table("signal_follows").insert({"signal_id": signal_id, "user_id": user_id}).execute()
        following = True

    count = _get_follow_count(signal_id, sb)
    global _feed_ts
    _feed_ts = 0.0
    return {"following": following, "followCount": count}


def report_comment(comment_id: str, sb) -> None:
    """Flag a comment for moderation."""
    sb.table("signal_comments").update({"is_flagged": True}).eq("id", comment_id).execute()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_votes(signal_id: str, sb) -> dict:
    res = sb.table("signal_votes").select("vote_type").eq("signal_id", signal_id).execute()
    votes = res.data or []
    bull  = sum(1 for v in votes if v["vote_type"] == "bullish")
    bear  = sum(1 for v in votes if v["vote_type"] == "bearish")
    watch = sum(1 for v in votes if v["vote_type"] == "watching")
    total = len(votes) or 1
    return {
        "bullish":    bull,
        "bearish":    bear,
        "watching":   watch,
        "total":      len(votes),
        "bullishPct": round(bull / total * 100),
    }


def _get_follow_count(signal_id: str, sb) -> int:
    res = sb.table("signal_follows").select("signal_id", count="exact").eq("signal_id", signal_id).execute()
    return res.count or 0


def _get_comment_count(signal_id: str, sb) -> int:
    res = sb.table("signal_comments").select("id", count="exact").eq("signal_id", signal_id).eq("is_flagged", False).execute()
    return res.count or 0


def _community_score(votes: dict, follows: int, comments: int) -> int:
    bull_pct     = votes.get("bullishPct", 50)
    follow_norm  = min(100, follows  * 10)
    comment_norm = min(100, comments * 15)
    score = bull_pct * 0.4 + follow_norm * 0.3 + comment_norm * 0.2
    return int(min(100, round(score)))


def _community_score_raw(
    bull_pct: float, follows: int, max_follows: int,
    comments: int, max_comments: int, recent_boost: float,
) -> float:
    follow_norm  = (follows  / max_follows)  * 100 if max_follows  > 0 else 0
    comment_norm = (comments / max_comments) * 100 if max_comments > 0 else 0
    return (
        bull_pct    * 0.4
      + follow_norm * 0.3
      + comment_norm* 0.2
      + recent_boost* 0.1 * 100
    )


def _score_label(score: float) -> str:
    if score >= 75: return "High Interest"
    if score >= 50: return "Rising Interest"
    if score >= 25: return "Some Interest"
    return "Low Activity"
