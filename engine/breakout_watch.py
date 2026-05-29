"""
Setup-watch lifecycle + history (generalized across all Quant buckets).

The Quant dashboard's buckets (quant_score_service) are stateless live screens —
they re-rank every request and keep no history. To (a) measure whether each
bucket actually works and (b) manage when a ticker enters and leaves, we track
each WATCH EPISODE in `breakout_watch_history` — ONE row per continuous stay
per bucket, NOT one per refresh. The `bucket` column distinguishes sections
(breakouts, breakdowns, topMomentum, pullbacks, highVolume, vwapReclaim,
oversoldBounce).

State machine (per ticker, per bucket):
    WATCHING ── breaks the level (breakout/breakdown only) ──▶ TRIGGERED ──exit──▶ (judged)
             ── leaves the bucket, never broke ───────────────────────────────▶ FADED
             ── on watch > EXPIRE_DAYS with no break (trigger buckets) ────────▶ EXPIRED

`sync_watch()` is idempotent and called on a schedule (~every 5 min, RTH) with
the current rows of ONE bucket. Trigger-less buckets (momentum/pullback/…) have
no level break — their episodes simply track presence (entry → exit) and are
graded on the forward move from entry.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger("signalbolt.breakout_watch")

EXPIRE_DAYS = 5          # WATCHING expires after this many days with no breakout
_TABLE      = "breakout_watch_history"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _broke(px: float, lvl: float, direction: str) -> bool:
    """Did price break the level in the bucket's direction?"""
    if not lvl:
        return False
    return px > lvl if direction == "up" else px < lvl


def sync_watch(sb, watch_rows: list[dict], *, bucket: str = "breakouts",
               direction: str = "up", needs_trigger: bool = True) -> dict:
    """
    Reconcile ONE bucket's live set with its persisted episodes.

    watch_rows: [{"ticker","price","level","score"}] currently in the bucket
                (level = the level being tested; ignored for non-trigger buckets).
    Returns small stats dict. Never raises (best-effort telemetry).
    """
    stats = {"entered": 0, "triggered": 0, "faded": 0, "expired": 0, "updated": 0}
    now = datetime.now(timezone.utc)
    try:
        open_eps = (sb.table(_TABLE).select("*")
                      .eq("bucket", bucket).is_("exited_at", "null")
                      .execute().data) or []
    except Exception as e:
        logger.warning(f"[setup_watch:{bucket}] fetch open episodes failed: {e}")
        return stats

    open_by = {r["ticker"]: r for r in open_eps}
    cur_by = {
        r["ticker"]: r for r in (watch_rows or [])
        if r.get("ticker") and r.get("price") and (r.get("level") or not needs_trigger)
    }

    # ── 1) Tickers currently in the bucket: enter new, or update existing ─────
    for tk, row in cur_by.items():
        px = float(row["price"]); lvl = float(row.get("level") or 0); score = float(row.get("score") or 0)
        ep = open_by.get(tk)
        if ep is None:
            broke = needs_trigger and _broke(px, lvl, direction)
            rec = {
                "ticker": tk, "bucket": bucket,
                "state": "TRIGGERED" if broke else "WATCHING",
                "entered_at": _now(), "enter_price": round(px, 4),
                "breakout_level": round(lvl, 4) if lvl else None,
                "enter_score": round(score, 1),
                "last_seen_at": _now(), "peak_price": round(px, 4), "max_favorable_pct": 0.0,
            }
            if broke:
                rec["triggered_at"] = _now()
                rec["trigger_price"] = round(px, 4)
            try:
                sb.table(_TABLE).insert(rec).execute()
                stats["entered"] += 1
                if broke:
                    stats["triggered"] += 1
            except Exception as e:
                logger.debug(f"[setup_watch:{bucket}] enter {tk} failed: {e}")
            continue

        enter_price = float(ep.get("enter_price") or px)
        peak = max(float(ep.get("peak_price") or px), px)
        upd = {
            "last_seen_at": _now(),
            "peak_price": round(peak, 4),
            "max_favorable_pct": round((peak - enter_price) / enter_price * 100, 2) if enter_price else 0.0,
            "updated_at": _now(),
        }
        # TRIGGER: price broke the level it was watching (trigger buckets only).
        if needs_trigger and ep.get("state") == "WATCHING" and _broke(px, lvl, direction):
            upd.update({"state": "TRIGGERED", "triggered_at": _now(), "trigger_price": round(px, 4)})
            stats["triggered"] += 1
        else:
            stats["updated"] += 1
        try:
            sb.table(_TABLE).update(upd).eq("id", ep["id"]).execute()
        except Exception as e:
            logger.debug(f"[setup_watch:{bucket}] update {tk} failed: {e}")

    # ── 2) Close open episodes that left the bucket (or went stale) ───────────
    for tk, ep in open_by.items():
        state = ep.get("state")
        in_cur = tk in cur_by
        age_days = 0
        try:
            age_days = (now - datetime.fromisoformat(ep["entered_at"].replace("Z", "+00:00"))).days
        except Exception:
            pass
        if needs_trigger and state == "WATCHING" and age_days >= EXPIRE_DAYS:
            reason = "EXPIRED"
        elif not in_cur and state == "TRIGGERED":
            reason = "TRIGGERED"   # broke out then left — judge forward outcome later
        elif not in_cur:
            reason = "FADED"       # left the bucket (never broke, or trigger-less bucket)
        else:
            continue               # still live and not stale — leave open
        try:
            sb.table(_TABLE).update({
                "exited_at": _now(), "exit_reason": reason, "updated_at": _now(),
            }).eq("id", ep["id"]).execute()
            stats[reason.lower()] = stats.get(reason.lower(), 0) + 1
        except Exception as e:
            logger.debug(f"[setup_watch:{bucket}] close {tk} failed: {e}")

    if any(stats.values()):
        logger.info(f"[setup_watch:{bucket}] sync — {stats}")
    return stats
