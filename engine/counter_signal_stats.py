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


# Opposing-reversal signal types (mirror the forward logic): a bullish bottom is
# the counter to a SHORT; a bearish top is the counter to a LONG.
_BULL_BOTTOM = {"turnaround", "turn_forming", "accum_forming"}
_BEAR_TOP = {"peak", "peak_forming", "distrib_forming"}


def _parse(ts):
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except Exception:
        return None


def backfill(sb, commit: bool = False, lookback: int = 6000) -> dict:
    """One-time: scan historical signals for cases where an OPPOSING cycle-reversal
    signal fired while a (filled, closed) position was open, and record a
    'counter_signal' event (lock price = the opposing signal's fire price) so the
    scorer has data NOW instead of only going forward. Idempotent (skips positions
    that already have a counter_signal event). DRY-RUN unless commit=True. Best-effort."""
    from collections import defaultdict
    if sb is None:
        return {"pairs": 0, "committed": False}
    try:
        rows = (sb.table("signals").select(
            "id,ticker,direction,strategy_type,entry_price,result_pct,status,created_at,closed_at")
            .order("created_at", desc=True).limit(lookback).execute().data) or []
        seen = (sb.table("signal_events").select("signal_id").eq("event_type", "counter_signal")
                .limit(10000).execute().data) or []
    except Exception as e:
        logger.debug(f"[counter_signal_stats] backfill fetch failed: {e}")
        return {"pairs": 0, "committed": False, "error": str(e)}
    existing = {r["signal_id"] for r in seen}
    by = defaultdict(list)
    for r in rows:
        by[r["ticker"]].append(r)

    inserts = []
    for tk, rs in by.items():
        for pos in rs:
            # only genuinely-filled, closed positions have a HOLD outcome to compare
            if pos["id"] in existing or pos.get("status") != "closed" or pos.get("result_pct") is None:
                continue
            if pos.get("entry_price") is None:
                continue
            opp = _BULL_BOTTOM if pos["direction"] == "SHORT" else _BEAR_TOP if pos["direction"] == "LONG" else None
            if not opp:
                continue
            pc = _parse(pos["created_at"]); px = _parse(pos.get("closed_at"))
            if not pc:
                continue
            cands = [o for o in rs
                     if o["id"] != pos["id"] and o.get("strategy_type") in opp
                     and _parse(o["created_at"]) and pc <= _parse(o["created_at"]) <= (px or _parse(o["created_at"]))]
            if not cands:
                continue
            o = min(cands, key=lambda x: _parse(x["created_at"]))
            lock = o.get("entry_price")
            if not lock:
                continue
            lp = _lock_pnl(pos["direction"], pos["entry_price"], lock)
            if lp is None:
                continue
            side = "short" if pos["direction"] == "SHORT" else "long"
            note = (f"Counter-signal (backfill): {o['strategy_type']} {o['direction']} fired vs this {side} "
                    f"— lock here = {lp:+.1f}%.")
            inserts.append({"signal_id": pos["id"], "event_type": "counter_signal",
                            "price": float(lock), "note": note, "created_at": o["created_at"]})
            existing.add(pos["id"])

    if commit and inserts:
        for i in range(0, len(inserts), 100):
            try:
                sb.table("signal_events").insert(inserts[i:i + 100]).execute()
            except Exception as e:
                logger.error(f"[counter_signal_stats] backfill insert batch failed: {e}")
    return {"pairs": len(inserts), "committed": bool(commit), "preview": inserts[:5]}


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
