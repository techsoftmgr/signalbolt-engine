"""
Module #3 — Signal Follow-Up Engine.

A post-signal MONITORING layer. The existing signal engine is UNTOUCHED — this
only READS an existing signal + its already-recorded signal_events (advisor
warnings) and adds a live "current status" read, assembling a human timeline so
users get continuous trade management, not just "BUY NOW". Never raises.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger("signalbolt.phase2.signal_followup")


def _current_status(direction, entry, stop, t1, price):
    """Pure: a one-line live status for an active signal."""
    if not price or not entry:
        return {"state": "UNKNOWN", "detail": "no live price"}
    is_long = (direction or "").upper() == "LONG"
    pnl = ((price - entry) if is_long else (entry - price)) / entry * 100
    state, detail = "OPEN", f"{pnl:+.1f}% unrealized"
    if t1:
        hit_t1 = (price >= t1) if is_long else (price <= t1)
        if hit_t1:
            state, detail = "TARGET1_REACHED", f"hit target 1 ({t1}), {pnl:+.1f}%"
    if stop:
        hit_stop = (price <= stop) if is_long else (price >= stop)
        if hit_stop:
            state, detail = "STOPPED", f"at stop ({stop}), {pnl:+.1f}%"
    return {"state": state, "detail": detail, "unrealized_pct": round(pnl, 1)}


def timeline(sb, signal_id: str) -> dict:
    """Assemble the follow-up timeline for a signal. Never raises."""
    try:
        srows = (sb.table("signals").select("*").eq("id", signal_id).limit(1).execute().data) or []
        if not srows:
            return {"enabled": True, "error": "signal not found"}
        s = srows[0]
        tk, direction = s.get("ticker"), s.get("direction")
        entry = s.get("entry_price"); stop = s.get("stop_loss"); t1 = s.get("target_one")

        tl = [{"time": s.get("created_at"), "event": "Signal Generated",
               "detail": f"{direction} {tk} @ {entry}  (stop {stop}, T1 {t1})"}]
        try:
            evs = (sb.table("signal_events").select("*").eq("signal_id", signal_id)
                   .order("created_at").limit(200).execute().data) or []
            for e in evs:
                tl.append({"time": e.get("created_at"),
                           "event": (e.get("event_type") or e.get("type") or "Update").replace("_", " ").title(),
                           "detail": e.get("message") or e.get("note") or e.get("detail") or ""})
        except Exception:
            pass

        price = None
        try:
            from engine.alpaca_client import get_latest_price
            price = get_latest_price(tk)
        except Exception:
            pass
        try:
            entry = float(entry) if entry is not None else None
        except (TypeError, ValueError):
            entry = None
        cur = _current_status(direction, entry, _f(stop), _f(t1), _f(price))
        tl.append({"time": datetime.now(timezone.utc).isoformat(), "event": "Current Status",
                   "detail": cur["detail"]})
        return {"enabled": True, "signal_id": signal_id, "ticker": tk, "direction": direction,
                "status": s.get("status"), "current": cur, "timeline": tl}
    except Exception as e:
        logger.error(f"[signal_followup] {signal_id} failed: {e}")
        return {"enabled": True, "error": str(e)}


def _f(x):
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None
