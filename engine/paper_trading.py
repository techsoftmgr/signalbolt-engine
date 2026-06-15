"""
Paper-trading service — turns SignalBolt signals into APPROVE-then-execute paper
orders on the Alpaca PAPER account (admin-only). The whole point: dry-run the
engine's own signals with zero real-money risk before any live/Robinhood step.

Lifecycle in the `paper_trades` table:
  proposed → (approved →) submitted → filled → closed
            ↘ rejected            ↘ canceled (entry never filled)   ↘ error

Approve-first by design: a job proposes paper trades from active signals; nothing
is placed until the admin approves it. INTEGRITY: execution goes through
paper_broker (paper=True) only — never the live account.
"""
from __future__ import annotations

import logging
import math
import os
from datetime import datetime, timedelta, timezone

from . import paper_broker

logger = logging.getLogger("signalbolt.paper_trading")

_ALLOC_USD = float(os.environ.get("PAPER_ALLOC_USD", "2000"))      # paper $ per trade
_LOOKBACK_H = float(os.environ.get("PAPER_PROPOSE_LOOKBACK_H", "12"))


# ── Pure helpers (unit-tested directly) ─────────────────────────────────────
def build_proposal(sig: dict, alloc: float = _ALLOC_USD) -> dict | None:
    """Signal row → a proposed paper trade (qty/side/levels). None if unusable."""
    try:
        entry = float(sig.get("entry_price") or 0)
        if entry <= 0:
            return None
        qty = int(max(1, math.floor(alloc / entry)))
        meta = sig.get("score_breakdown") if isinstance(sig.get("score_breakdown"), dict) else {}
        return {
            "signal_id": sig.get("id"),
            "ticker": sig.get("ticker"),
            "direction": (sig.get("direction") or "LONG").upper(),
            "qty": qty,
            "entry_price": round(entry, 2),
            "stop_loss": round(float(sig["stop_loss"]), 2) if sig.get("stop_loss") else None,
            "target_one": round(float(sig["target_one"]), 2) if sig.get("target_one") else None,
            "alloc_usd": round(float(alloc), 2),
            "strategy_type": sig.get("strategy_type"),
            "detector_source": (meta or {}).get("detector_source"),
            "status": "proposed",
        }
    except Exception as e:
        logger.debug(f"[paper] build_proposal failed: {e}")
        return None


def realized(direction: str, entry, exit_) -> tuple:
    """(pnl_per_share, pct) for a closed paper trade — direction-aware."""
    try:
        entry = float(entry); exit_ = float(exit_)
        if entry <= 0:
            return None, None
        diff = (exit_ - entry) if (direction or "LONG").upper() == "LONG" else (entry - exit_)
        return diff, diff / entry * 100
    except (TypeError, ValueError):
        return None, None


# ── DB-touching operations ──────────────────────────────────────────────────
def propose_from_active_signals(sb) -> dict:
    """Create 'proposed' paper trades for recent active signals that don't have one
    yet. Idempotent (skips signals already proposed)."""
    created = 0
    try:
        since = (datetime.now(timezone.utc) - timedelta(hours=_LOOKBACK_H)).isoformat()
        sigs = (sb.table("signals")
                .select("id,ticker,direction,entry_price,stop_loss,target_one,strategy_type,score_breakdown,created_at")
                .eq("status", "active").gte("created_at", since).execute().data) or []
        if not sigs:
            return {"created": 0}
        ids = [s["id"] for s in sigs]
        seen: set = set()
        for i in range(0, len(ids), 200):
            rows = (sb.table("paper_trades").select("signal_id")
                    .in_("signal_id", ids[i:i + 200]).execute().data) or []
            seen |= {r["signal_id"] for r in rows}
        for s in sigs:
            if s["id"] in seen:
                continue
            p = build_proposal(s)
            if not p:
                continue
            try:
                sb.table("paper_trades").insert(p).execute()
                created += 1
            except Exception as e:
                logger.debug(f"[paper] insert proposal {s.get('ticker')} failed: {e}")
    except Exception as e:
        logger.error(f"[paper] propose scan failed: {e}")
    return {"created": created}


def _get_row(sb, paper_id):
    rows = (sb.table("paper_trades").select("*").eq("id", paper_id).limit(1).execute().data) or []
    return rows[0] if rows else None


def approve(sb, paper_id: str) -> dict:
    """Admin-approve a proposal → place the bracket on the paper account."""
    row = _get_row(sb, paper_id)
    if not row:
        return {"error": "not found"}
    if row.get("status") != "proposed":
        return {"error": f"not proposable (status={row.get('status')})"}
    if not paper_broker.is_configured():
        return {"error": "paper broker not configured — set ALPACA_PAPER_API_KEY / ALPACA_PAPER_SECRET_KEY"}
    res = paper_broker.submit_bracket(row["ticker"], row["direction"], int(row["qty"]),
                                      row["entry_price"], row.get("stop_loss"), row.get("target_one"))
    now = datetime.now(timezone.utc).isoformat()
    if not res or res.get("error"):
        msg = (res or {}).get("error", "submit failed")
        sb.table("paper_trades").update({"status": "error", "note": msg, "decided_at": now}).eq("id", paper_id).execute()
        return {"error": msg}
    sb.table("paper_trades").update({
        "status": "submitted", "broker_order_id": res["order_id"], "decided_at": now,
    }).eq("id", paper_id).execute()
    return {"ok": True, "order_id": res["order_id"], "status": res["status"]}


def reject(sb, paper_id: str) -> dict:
    row = _get_row(sb, paper_id)
    if not row:
        return {"error": "not found"}
    if row.get("status") != "proposed":
        return {"error": f"not proposable (status={row.get('status')})"}
    sb.table("paper_trades").update({
        "status": "rejected", "decided_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", paper_id).execute()
    return {"ok": True}


def close_trade(sb, paper_id: str) -> dict:
    """Manually flatten an open paper position + cancel its working entry."""
    row = _get_row(sb, paper_id)
    if not row:
        return {"error": "not found"}
    if row.get("status") not in ("submitted", "filled"):
        return {"error": f"not open (status={row.get('status')})"}
    if row.get("broker_order_id"):
        paper_broker.cancel_order(row["broker_order_id"])
    paper_broker.close_position(row["ticker"])
    sb.table("paper_trades").update({"note": "manual close requested"}).eq("id", paper_id).execute()
    return {"ok": True, "note": "close requested — reconcile will record the fill"}


def reconcile(sb) -> dict:
    """Sync open paper trades with the broker: entry fills, cancels, and bracket-leg
    exits (→ realized P&L). Safe no-op if the broker is disabled."""
    if not paper_broker.is_configured():
        return {"reconciled": 0}
    updated = 0
    try:
        rows = (sb.table("paper_trades").select("*").in_("status", ["submitted", "filled"]).execute().data) or []
        for row in rows:
            oid = row.get("broker_order_id")
            if not oid:
                continue
            od = paper_broker.get_order_with_legs(oid)
            if not od:
                continue
            st = (od.get("status") or "").lower()
            upd: dict = {}
            if row["status"] == "submitted":
                if od.get("filled_avg_price") and "filled" in st:
                    upd = {"status": "filled", "fill_price": od["filled_avg_price"]}
                elif any(x in st for x in ("canceled", "expired", "rejected", "done_for_day")):
                    upd = {"status": "canceled", "note": f"entry {st}"}
            # bracket leg (TP/SL) filled → position closed
            if row["status"] == "filled" or upd.get("status") == "filled":
                entry_px = upd.get("fill_price") or row.get("fill_price")
                for leg in od.get("legs", []):
                    if "filled" in (leg.get("status") or "").lower() and leg.get("filled_avg_price"):
                        pnl, pct = realized(row["direction"], entry_px, leg["filled_avg_price"])
                        qty = float(row.get("qty") or 0)
                        upd = {
                            "status": "closed", "exit_price": leg["filled_avg_price"],
                            "realized_pnl": round(pnl * qty, 2) if pnl is not None else None,
                            "realized_pct": round(pct, 2) if pct is not None else None,
                            "closed_at": datetime.now(timezone.utc).isoformat(),
                        }
                        break
            if upd:
                sb.table("paper_trades").update(upd).eq("id", row["id"]).execute()
                updated += 1
    except Exception as e:
        logger.error(f"[paper] reconcile failed: {e}")
    return {"reconciled": updated}


def portfolio(sb) -> dict:
    """Paper account snapshot + open positions + lifecycle counts + realized stats."""
    acct = paper_broker.account()
    pos = paper_broker.positions()
    counts: dict = {}
    realized_total = 0.0
    wins = losses = 0
    try:
        rows = (sb.table("paper_trades").select("status,realized_pnl").execute().data) or []
        for r in rows:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
            if r["status"] == "closed":
                pnl = float(r.get("realized_pnl") or 0)
                realized_total += pnl
                wins += pnl > 0
                losses += pnl < 0
    except Exception as e:
        logger.debug(f"[paper] portfolio counts failed: {e}")
    return {
        "configured": paper_broker.is_configured(),
        "account": acct, "positions": pos, "counts": counts,
        "realized_pnl_total": round(realized_total, 2), "wins": wins, "losses": losses,
    }
