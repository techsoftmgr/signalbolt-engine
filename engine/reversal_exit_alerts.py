"""
Counter-signal (reversal-aware) exit alerts.

The idea (user-proposed): when the engine's OWN opposing detector starts forming
on a ticker you already have an open position in, that's a strong cue to BOOK /
de-risk. Example: you're SHORT GLD (breakdown) and in profit, then a TURNAROUND
("base forming") appears on GLD — the down-thesis is exhausting, so lock it in.

This is a heads-up ALERT only (the app is decision-support; it never closes a
broker position). It scans ACTIVE engine signals and, when the live quant shows
an OPPOSING reversal stage forming on the same ticker, pushes a "consider
booking" alert. Anti-spam by construction:

  • Pref-gated ('reversal_exit_alerts') via push.send_reversal_exit_alert.
  • Per-signal-id dedup in cache.kv → one alert per open position per reversal.
  • RTH-only + ENV-gated (REVERSAL_EXIT_ALERTS_ENABLED, default OFF) so it ships
    dark and flips on once measured.

Runs every ~5 min on trading days (RTH) from runner.py.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger("signalbolt.reversal_exit_alerts")

_DEDUP_TTL = 24 * 3600        # one alert per open position per day
_MAX_TICKERS = 80


def _enabled() -> bool:
    return (os.getenv("REVERSAL_EXIT_ALERTS_ENABLED", "") or "").strip().lower() in ("1", "true", "yes", "on")


def _stage_active(stage) -> bool:
    """A turnaround/peak stage that is anything other than 'none'/empty = forming."""
    return bool(stage) and str(stage).strip().lower() not in ("none", "", "null")


def opposing_reversal(direction: str, snap: dict | None) -> tuple[str, str] | None:
    """PURE: given an open position's direction and the ticker's quant snapshot,
    return (reversal_type, stage) when an OPPOSING reversal is forming, else None.
      • open SHORT  + turnaround forming (a bottom) → ('turnaround', stage)
      • open LONG   + peak forming (a top)          → ('peak', stage)
    """
    if not snap:
        return None
    if direction == "SHORT" and _stage_active(snap.get("turnaroundStage")):
        return ("turnaround", str(snap.get("turnaroundStage")))
    if direction == "LONG" and _stage_active(snap.get("peakStage")):
        return ("peak", str(snap.get("peakStage")))
    return None


def run(sb) -> dict:
    """Alert on open positions that now face an opposing reversal. Best-effort."""
    stats = {"checked": 0, "alerts": 0}
    if not _enabled() or sb is None:
        return stats
    try:
        from engine import session_classifier
        if not session_classifier.is_market_open_now():
            return stats
    except Exception:
        return stats
    try:
        active = (sb.table("signals").select("id,ticker,direction,strategy_type,entry_price")
                  .eq("status", "active").limit(200).execute().data) or []
    except Exception as e:
        logger.debug(f"[reversal_exit] active fetch failed: {e}")
        return stats
    if not active:
        return stats
    tickers = list({r["ticker"] for r in active if r.get("ticker")})[:_MAX_TICKERS]
    try:
        from engine import quant_score_service as q
        snaps = q.snapshot(tickers) or {}
    except Exception as e:
        logger.debug(f"[reversal_exit] snapshot failed: {e}")
        return stats
    try:
        from engine.alpaca_client import get_latest_prices
        px = get_latest_prices(tickers) or {}
    except Exception:
        px = {}
    from engine import cache, push
    for r in active:
        stats["checked"] += 1
        try:
            opp = opposing_reversal(r.get("direction"), snaps.get(r["ticker"]))
            if not opp:
                continue
            key = f"revexit:{r['id']}"
            if cache.kv.get_json(key):
                continue
            # Live unrealized P&L (direction-aware) for the alert body.
            p = px.get(r["ticker"]); entry = r.get("entry_price")
            pnl = None
            if p and entry:
                pnl = round((p - entry) / entry * 100 if r["direction"] == "LONG"
                            else (entry - p) / entry * 100, 1)
            push.send_reversal_exit_alert(r["ticker"], r["direction"], pnl, opp, sb)
            cache.kv.set_json(key, True, _DEDUP_TTL)
            stats["alerts"] += 1
        except Exception as e:
            logger.debug(f"[reversal_exit] {r.get('ticker')} failed: {e}")
    if stats["alerts"]:
        logger.info(f"[reversal_exit] {stats}")
    return stats
