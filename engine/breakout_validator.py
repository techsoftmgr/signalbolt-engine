"""
Breakout-watch outcome validator → Watch Accuracy.

Judges `breakout_watch_history` episodes: did a *watched* setup actually break
out AND follow through? Backfills outcome (win|loss) + realized_pct so the Quant
dashboard can show a credible "Watch Accuracy" number.

  WIN  = a TRIGGERED episode whose price ran ≥ WIN_PCT past the breakout level
         within HOLD_DAYS (the watch correctly called a breakout that paid).
  LOSS = triggered-but-failed-to-follow-through, OR FADED / EXPIRED (the watch
         flagged it but it never broke / never ran).

Only episodes whose judging window has elapsed (or that have already exited) are
scored. Runs post-close from the scheduler (alongside the gate/zone validators).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger("signalbolt.breakout_validator")

from engine.breakout_history import judge_path, HORIZON_DAYS  # single source of truth

HOLD_DAYS = HORIZON_DAYS   # alias kept for readability; resolution window = HORIZON_DAYS
_TABLE    = "breakout_watch_history"


def _judge_one(row: dict):
    """Net-result outcome over the holding window, bucket/direction-aware —
    shared with the track-record screen via breakout_history.judge_path."""
    from engine import alpaca_client
    from engine.breakout_history import bucket_cfg
    cfg = bucket_cfg(row.get("bucket") or "breakouts")
    state, exit_reason = row.get("state"), row.get("exit_reason")

    # Trigger-required buckets (breakout/breakdown): never broke = call didn't pay.
    if cfg["needsTrigger"] and (state in ("FADED", "EXPIRED") or exit_reason in ("FADED", "EXPIRED")):
        return {"outcome": "loss", "realized_pct": 0.0}

    # Anchor: trigger for trigger-buckets, else entry (when it joined the bucket).
    if cfg["needsTrigger"]:
        anchor_at = row.get("triggered_at")
        entry = float(row.get("trigger_price") or row.get("breakout_level") or 0)
    else:
        anchor_at = row.get("entered_at")
        entry = float(row.get("enter_price") or 0)
    if not anchor_at or entry <= 0:
        return None
    try:
        t = datetime.fromisoformat(anchor_at.replace("Z", "+00:00"))
    except Exception:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)

    # Window must have elapsed (unless the episode already exited) before judging.
    if datetime.now(timezone.utc) - t < timedelta(days=HORIZON_DAYS) and not row.get("exited_at"):
        return None

    bars = alpaca_client.get_bars(row.get("ticker"), timeframe="1Day", days=HORIZON_DAYS + 8)
    if bars is None or len(bars) < 2:
        return None

    jp = judge_path(bars, t, entry, direction=cfg["direction"])
    if jp.get("outcome") is None:
        return None   # not yet resolved (window not fully elapsed)
    return {"outcome": jp["outcome"], "realized_pct": jp["resultPct"]}


def judge_batch(sb, limit: int = 500) -> dict:
    """Backfill outcome/realized_pct on unjudged episodes whose window elapsed."""
    try:
        rows = (
            sb.table(_TABLE)
              .select("id,ticker,bucket,state,exit_reason,triggered_at,trigger_price,breakout_level,entered_at,enter_price,exited_at")
              .is_("outcome", "null")
              .order("entered_at", desc=True)
              .limit(max(1, min(limit, 2000)))
              .execute()
        ).data or []
    except Exception as e:
        logger.error(f"[breakout_validator] fetch failed: {e}")
        return {"error": str(e), "processed": 0}

    stats = {"processed": 0, "wins": 0, "losses": 0, "skipped": 0}
    for row in rows:
        try:
            res = _judge_one(row)
            if res is None:
                stats["skipped"] += 1
                continue
            sb.table(_TABLE).update(res).eq("id", row["id"]).execute()
            stats["processed"] += 1
            stats["wins" if res["outcome"] == "win" else "losses"] += 1
        except Exception as e:
            logger.debug(f"[breakout_validator] row {row.get('id')} error: {e}")
            stats["skipped"] += 1
    logger.info(f"[breakout_validator] batch — {stats}")
    return stats


def watch_accuracy(sb, days: int = 14) -> dict:
    """Aggregate Watch Accuracy over judged episodes in the last `days`."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    try:
        rows = (
            sb.table(_TABLE)
              .select("outcome,state,exit_reason,realized_pct")
              .eq("bucket", "breakouts")
              .gte("session_date", since)
              .not_.is_("outcome", "null")
              .limit(5000)
              .execute()
        ).data or []
    except Exception as e:
        logger.debug(f"[breakout_validator] accuracy fetch failed: {e}")
        return {"judged": 0, "accuracy_pct": None}

    judged = len(rows)
    if judged == 0:
        return {"judged": 0, "accuracy_pct": None}
    wins = sum(1 for r in rows if r.get("outcome") == "win")
    triggered = sum(1 for r in rows if r.get("state") == "TRIGGERED" or r.get("exit_reason") == "TRIGGERED")
    avg = round(sum((r.get("realized_pct") or 0) for r in rows) / judged, 2)
    return {
        "judged":           judged,
        "wins":             wins,
        "accuracy_pct":     round(100 * wins / judged),
        "trigger_rate_pct": round(100 * triggered / judged),
        "avg_realized_pct": avg,
        "window_days":      days,
    }
