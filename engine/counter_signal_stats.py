"""
Counter-signal exit scorer — measures the user's idea: when the engine's OWN
opposing detector forms on an open position, would BOOKING there have beaten
HOLDING to the real exit?

Joins the 'counter_signal' timeline events (signal_events) to their signals and,
for each:
  • LOCK P&L%  = P&L if you'd closed AT the opposing signal (from the stored lock
                 price vs entry, direction-aware) — the "P&L during the opposite signal".
  • HOLD P&L%  = the signal's actual final result_pct (held to the real exit).
  • edge       = lock − hold  (positive = locking at the counter-signal beat holding).

Aggregates avg lock vs hold + "% of times locking beat holding", segmented by the
opposing stage (forming vs confirmed). Open positions are reported separately
(lock recorded, hold still pending). Best-effort; never raises.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger("signalbolt.counter_signal_stats")


def _lock_pnl(direction: str, entry: float, lock: float) -> float | None:
    try:
        entry = float(entry); lock = float(lock)
        if entry <= 0:
            return None
        return round((entry - lock) / entry * 100 if direction == "SHORT"
                     else (lock - entry) / entry * 100, 2)
    except (TypeError, ValueError):
        return None


def _stage_of(note: str) -> str:
    n = (note or "").lower()
    if "confirmed" in n:
        return "confirmed"
    if "forming" in n or "early" in n:
        return "forming"
    return "unknown"


def _aggregate(rows: list[dict]) -> dict:
    """PURE: rows = [{lock_pnl, hold_pnl(None if open), stage}]. Returns the
    lock-vs-hold verdict over the CLOSED rows + per-stage + open count."""
    closed = [r for r in rows if r.get("hold_pnl") is not None and r.get("lock_pnl") is not None]
    open_n = sum(1 for r in rows if r.get("hold_pnl") is None)

    def block(rs: list[dict]) -> dict:
        n = len(rs)
        if not n:
            return {"n": 0}
        avg_lock = round(sum(r["lock_pnl"] for r in rs) / n, 2)
        avg_hold = round(sum(r["hold_pnl"] for r in rs) / n, 2)
        beat = sum(1 for r in rs if r["lock_pnl"] > r["hold_pnl"])
        return {"n": n, "avg_lock_pnl": avg_lock, "avg_hold_pnl": avg_hold,
                "edge": round(avg_lock - avg_hold, 2), "lock_beat_hold_pct": round(beat / n * 100)}

    by_stage = {}
    for stg in ("forming", "confirmed"):
        sub = [r for r in closed if r.get("stage") == stg]
        if sub:
            by_stage[stg] = block(sub)

    return {
        "total_events": len(rows),
        "scored": len(closed),
        "open_pending": open_n,
        "overall": block(closed),
        "by_stage": by_stage,
        "note": "LOCK = P&L if booked at the opposing signal; HOLD = actual final result. "
                "edge>0 ⇒ locking at the counter-signal beat holding. Descriptive, not advice.",
    }


def stats(sb, days: int = 90) -> dict:
    """Build the counter-signal lock-vs-hold scorecard. Best-effort."""
    if sb is None:
        return {"available": False, "note": "No data yet."}
    try:
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        events = (sb.table("signal_events").select("signal_id,price,note,created_at")
                  .eq("event_type", "counter_signal").gte("created_at", since)
                  .limit(3000).execute().data) or []
    except Exception as e:
        logger.debug(f"[counter_signal_stats] events fetch failed: {e}")
        return {"available": False, "note": "Track record not available yet (no counter-signal events)."}
    if not events:
        return {"available": False, "scored": 0,
                "note": "No counter-signal events yet. Enable REVERSAL_EXIT_ALERTS_ENABLED and let data accrue."}
    sig_ids = list({e["signal_id"] for e in events if e.get("signal_id")})
    try:
        sigs = (sb.table("signals").select("id,direction,entry_price,result_pct,status,ticker")
                .in_("id", sig_ids).limit(3000).execute().data) or []
    except Exception as e:
        logger.debug(f"[counter_signal_stats] signals fetch failed: {e}")
        return {"available": False, "note": "Track record not available yet."}
    smap = {s["id"]: s for s in sigs}

    rows = []
    for e in events:
        s = smap.get(e["signal_id"])
        if not s or s.get("entry_price") is None or e.get("price") is None:
            continue
        lp = _lock_pnl(s.get("direction"), s["entry_price"], e["price"])
        if lp is None:
            continue
        closed = s.get("status") in ("closed", "cancelled")
        rows.append({
            "lock_pnl": lp,
            "hold_pnl": s.get("result_pct") if closed else None,
            "stage": _stage_of(e.get("note")),
        })
    agg = _aggregate(rows)
    agg["available"] = agg["scored"] > 0 or agg["open_pending"] > 0
    return agg
