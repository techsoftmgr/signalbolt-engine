"""
Commentary push alerts — V2 of Ticker Commentary ("Today's Tape").

Scans WATCHED tickers on a schedule, finds the NEW high-severity intraday events
that appeared since the last scan (via ticker_commentary.build), and pushes the
watching users. Anti-spam by construction:

  • Watchlist-scoped + pref-gated ('commentary_alerts') via push.send_commentary_alert.
  • Only ALERT-WORTHY event types fire a push (MACD cross / ORB break / gap /
    sharp move / VWAP reclaim-lose) — the noisier RSI/HoD/LoD/EMA/volume events
    stay in the in-app feed only.
  • A per-ticker watermark (last event time seen) in cache.kv → only events newer
    than the last scan are considered. COLD START seeds the watermark and pushes
    nothing (a deploy / cache wipe can't fire a burst).
  • Per-ticker-per-day cap + per-event dedup so an oscillating tape can't spam.
  • ENV-GATED by COMMENTARY_ALERTS_ENABLED so it can ship dark and be flipped on.

Runs every ~10 min on trading days (RTH only) from runner.py.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger("signalbolt.commentary_alerts")

# High-signal event types worth a phone buzz (others stay in the in-app feed).
_ALERT_TYPES = {"MACD_CROSS", "ORB", "GAP", "VWAP", "MOVE"}
_MIN_SEVERITY = 2
_SEEN_TTL = 16 * 3600          # remember a ticker's watermark ~1 session
_DEDUP_TTL = 36 * 3600
_MAX_TICKERS = 80              # bound per-run cost
_MAX_PER_TICKER_DAY = 3        # conservative — at most 3 pushes per ticker per day

_TONE_EMOJI = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}


def _enabled() -> bool:
    return (os.getenv("COMMENTARY_ALERTS_ENABLED", "") or "").strip().lower() in ("1", "true", "yes", "on")


def _alertworthy(ev: dict) -> bool:
    # Never push counter-trend events — they're "watch only", not actionable.
    return (ev.get("type") in _ALERT_TYPES
            and int(ev.get("severity") or 0) >= _MIN_SEVERITY
            and not ev.get("against_trend"))


def _new_events(events: list, last_iso: str | None) -> list:
    """PURE — alert-worthy events strictly newer than the watermark, oldest→newest."""
    fresh = [e for e in (events or [])
             if _alertworthy(e) and e.get("time") and (last_iso is None or e["time"] > last_iso)]
    return sorted(fresh, key=lambda e: e["time"])


def _format(ticker: str, ev: dict) -> tuple[str, str]:
    """Push title/body from an event. Educational; appends the idea if present."""
    emoji = _TONE_EMOJI.get(ev.get("tone"), "•")
    # strip the "(5m)"/"(15m)" suffix from the feed title for a cleaner push headline
    headline = (ev.get("title") or "Event").split(" (")[0]
    title = f"{emoji} {ticker} — {headline}"
    body = ev.get("detail") or ""
    idea = ev.get("idea")
    if idea and idea.get("text"):
        body = f"{body}  {idea['text']}"
    return title, body[:240]


def run(sb) -> dict:
    """Scan watched tickers, push NEW high-severity intraday events. Best-effort."""
    stats = {"tickers": 0, "alerts": 0, "seeded": 0, "skipped": 0}
    if not _enabled():
        stats["disabled"] = True
        return stats

    from engine import cache, push, ticker_commentary

    try:
        rows = sb.table("watchlist").select("ticker").execute().data or []
    except Exception as e:
        logger.error(f"[commentary_alerts] fetch watchlist failed: {e}")
        return stats

    tickers = list({(r.get("ticker") or "").upper() for r in rows if r.get("ticker")})[:_MAX_TICKERS]
    stats["tickers"] = len(tickers)
    if not tickers:
        return stats
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for tk in tickers:
        try:
            res = ticker_commentary.build(tk)
            if not res or not res.get("available"):
                continue
            events = res.get("events") or []
            newest_iso = max((e.get("time") for e in events if e.get("time")), default=None)

            seen = cache.kv.get_json(f"cmt_seen:{tk}")
            last_iso = (seen or {}).get("t")

            # Cold start: seed the watermark, push nothing (no burst on deploy).
            if seen is None:
                if newest_iso:
                    cache.kv.set_json(f"cmt_seen:{tk}", {"t": newest_iso}, _SEEN_TTL)
                stats["seeded"] += 1
                continue

            fresh = _new_events(events, last_iso)
            if fresh:
                cap_key = f"cmt_cap:{tk}:{today}"
                sent_today = int((cache.kv.get_json(cap_key) or {}).get("n", 0))
                for ev in fresh:
                    if sent_today >= _MAX_PER_TICKER_DAY:
                        stats["skipped"] += 1
                        continue
                    dedup = f"cmt_alert:{tk}:{ev['type']}:{ev['time']}"
                    if cache.kv.get_json(dedup):
                        continue
                    title, body = _format(tk, ev)
                    n = push.send_commentary_alert(tk, title, body, ev.get("type"), sb=sb)
                    cache.kv.set_json(dedup, {"sent": True, "n": n}, _DEDUP_TTL)
                    sent_today += 1
                    if n:
                        stats["alerts"] += 1
                    logger.info(f"[commentary_alerts] {tk} {ev['type']} @ {ev['time']} -> pushed {n}")
                cache.kv.set_json(cap_key, {"n": sent_today}, _DEDUP_TTL)

            # Advance the watermark to the newest event so old ones aren't re-evaluated.
            if newest_iso and newest_iso != last_iso:
                cache.kv.set_json(f"cmt_seen:{tk}", {"t": newest_iso}, _SEEN_TTL)
        except Exception as e:
            logger.debug(f"[commentary_alerts] {tk} failed: {e}")

    logger.info(f"[commentary_alerts] {stats}")
    return stats
