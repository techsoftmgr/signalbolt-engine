"""
Breakout Watch lifecycle + history.

The Quant dashboard's "Breakout Watch" (quant_score_service, setupType=="breakout")
is a stateless live screen — it re-ranks every request and keeps no history. To
(a) measure whether the watch actually works and (b) manage when a ticker enters
and leaves, we track each WATCH EPISODE in `breakout_watch_history` — ONE row per
continuous stay, NOT one per refresh.

State machine (per ticker):
    WATCHING ── price breaks above the 20-day high ──▶ TRIGGERED ──exit──▶ (judged)
             ── leaves the breakout bucket, never broke ───────────────▶ FADED
             ── on watch > EXPIRE_DAYS with no breakout ───────────────▶ EXPIRED

`sync_watch()` is idempotent and called on a schedule (~every 5 min, RTH) with the
current breakout-bucket rows. It opens/updates/closes episodes by diffing the live
set against the open episodes in the table. Outcome (win|loss) is backfilled later
by breakout_validator (judges whether a TRIGGERED episode ran to target).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger("signalbolt.breakout_watch")

# Tunable lifecycle thresholds.
EXPIRE_DAYS = 5          # WATCHING expires after this many days with no breakout
_TABLE      = "breakout_watch_history"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sync_watch(sb, watch_rows: list[dict]) -> dict:
    """
    Reconcile the live breakout-watch set with the persisted episodes.

    watch_rows: [{"ticker", "price", "level", "score"}] currently in the breakout
                bucket (level = the 20-day high being tested).
    Returns small stats dict. Never raises (best-effort telemetry).
    """
    stats = {"entered": 0, "triggered": 0, "faded": 0, "expired": 0, "updated": 0}
    now = datetime.now(timezone.utc)
    try:
        open_eps = sb.table(_TABLE).select("*").is_("exited_at", "null").execute().data or []
    except Exception as e:
        logger.warning(f"[breakout_watch] fetch open episodes failed: {e}")
        return stats

    open_by = {r["ticker"]: r for r in open_eps}
    cur_by = {
        r["ticker"]: r for r in (watch_rows or [])
        if r.get("ticker") and r.get("level") and r.get("price")
    }

    # ── 1) Tickers currently on watch: enter new, or update existing ──────────
    for tk, row in cur_by.items():
        px = float(row["price"]); lvl = float(row["level"]); score = float(row.get("score") or 0)
        ep = open_by.get(tk)
        if ep is None:
            try:
                sb.table(_TABLE).insert({
                    "ticker": tk, "state": "WATCHING", "entered_at": _now(),
                    "enter_price": round(px, 4), "breakout_level": round(lvl, 4),
                    "enter_score": round(score, 1), "last_seen_at": _now(),
                    "peak_price": round(px, 4), "max_favorable_pct": 0.0,
                }).execute()
                stats["entered"] += 1
            except Exception as e:
                logger.debug(f"[breakout_watch] enter {tk} failed: {e}")
            continue

        enter_price = float(ep.get("enter_price") or px)
        peak = max(float(ep.get("peak_price") or px), px)
        upd = {
            "last_seen_at": _now(),
            "peak_price": round(peak, 4),
            "max_favorable_pct": round((peak - enter_price) / enter_price * 100, 2) if enter_price else 0.0,
            "updated_at": _now(),
        }
        # TRIGGER: price broke above the level it was watching.
        if ep.get("state") == "WATCHING" and px > lvl:
            upd.update({"state": "TRIGGERED", "triggered_at": _now(), "trigger_price": round(px, 4)})
            stats["triggered"] += 1
        else:
            stats["updated"] += 1
        try:
            sb.table(_TABLE).update(upd).eq("id", ep["id"]).execute()
        except Exception as e:
            logger.debug(f"[breakout_watch] update {tk} failed: {e}")

    # ── 2) Close open episodes that left the watch (or went stale) ────────────
    # Tickers still in the live bucket were already updated in step 1; here we
    # only close ones that dropped out, or WATCHING ones that have gone stale.
    for tk, ep in open_by.items():
        state = ep.get("state")
        in_cur = tk in cur_by
        age_days = 0
        try:
            age_days = (now - datetime.fromisoformat(ep["entered_at"].replace("Z", "+00:00"))).days
        except Exception:
            pass
        if state == "WATCHING" and age_days >= EXPIRE_DAYS:
            reason = "EXPIRED"
        elif not in_cur and state == "TRIGGERED":
            reason = "TRIGGERED"   # broke out then left the bucket — judge forward outcome later
        elif not in_cur and state == "WATCHING":
            reason = "FADED"       # left the breakout zone without ever breaking
        else:
            continue               # still live (in_cur) and not stale — leave open
        try:
            sb.table(_TABLE).update({
                "exited_at": _now(), "exit_reason": reason, "updated_at": _now(),
            }).eq("id", ep["id"]).execute()
            stats[reason.lower()] = stats.get(reason.lower(), 0) + 1
        except Exception as e:
            logger.debug(f"[breakout_watch] close {tk} failed: {e}")

    if any(stats.values()):
        logger.info(f"[breakout_watch] sync — {stats}")
    return stats
