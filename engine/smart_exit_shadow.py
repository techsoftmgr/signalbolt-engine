"""
Shadow backfill — the ALTERNATE (smart-exit) outcome for CLOSED shadow-tagged trades.

The in-loop shadow only observes while a trade is active, so it misses the case that
matters most: the old logic bails early and smart-exit would have HELD LONGER and made
more. This job closes that gap — after a shadow trade is closed, it replays smart-exit's
full lifecycle over daily price bars that extend PAST the real close, and records the
outcome in score_breakdown.smart_exit_shadow_final (with the delta vs the real result).

READ-ONLY on outcomes: it only adds one namespaced key to score_breakdown; it never
touches result / result_pct / status / closed_reason. Sibling of gate_validator.
"""
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import logging

from engine import exit_engine, alpaca_client

logger = logging.getLogger(__name__)

_MAX_HOLD = 25            # trading days the replay manages a position
_FULL_WINDOW_DAYS = 35    # calendar days after entry for a full replay window to exist


def _needs_backfill(bd) -> bool:
    return (isinstance(bd, dict) and bd.get("smart_exit_managed")
            and bd.get("smart_exit_mode") == "shadow"
            and not bd.get("smart_exit_shadow_final"))


def backfill_batch(sb, limit: int = 300, lookback_days: int = 60) -> dict:
    """Replay smart-exit over forward daily bars for each un-evaluated closed shadow
    trade; store the alternate outcome. Early-exiting shadows finalize immediately;
    long-holders finalize once a full ~25-trading-day window has elapsed."""
    since = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
    rows = (sb.table("signals")
              .select("id,ticker,direction,entry_price,stop_loss,created_at,result_pct,score_breakdown")
              .eq("status", "closed").gte("created_at", since).limit(3000).execute()).data or []
    todo = [r for r in rows if _needs_backfill(r.get("score_breakdown")) and r.get("entry_price") and r.get("stop_loss")][:limit]
    if not todo:
        return {"candidates": 0, "evaluated": 0}

    now = datetime.now(timezone.utc)
    try:
        spy = alpaca_client.get_bars("SPY", "1Day", days=180)
    except Exception:
        spy = None

    done = better = worse = pending = 0
    for r in todo:
        try:
            ed = datetime.fromisoformat(r["created_at"].replace("Z", "+00:00"))
            daily = alpaca_client.get_bars(r["ticker"], "1Day", days=180)
            if daily is None or len(daily) < 40:
                continue
            res = exit_engine.replay(r["direction"], float(r["entry_price"]), float(r["stop_loss"]),
                                     daily, entry_date=ed.date(), spy_df=spy, max_hold=_MAX_HOLD)
            if not res:
                continue
            # window_end before the full window has elapsed = premature → retry later
            if res["exit_reason"] == "window_end" and (now - ed) < timedelta(days=_FULL_WINDOW_DAYS):
                pending += 1
                continue
            actual = r.get("result_pct")
            delta = round(res["pnl_pct"] - float(actual), 3) if actual is not None else None
            bd = dict(r.get("score_breakdown") or {})
            bd["smart_exit_shadow_final"] = {**res, "actual_pct": actual, "delta": delta,
                                             "v": exit_engine.VERSION}
            sb.table("signals").update({"score_breakdown": bd}).eq("id", r["id"]).execute()
            done += 1
            if delta is not None:
                better += delta > 0.01
                worse += delta < -0.01
        except Exception as e:
            logger.debug(f"[shadow_backfill] {r.get('ticker')} failed: {e}")
    summary = {"candidates": len(todo), "evaluated": done, "pending": pending,
               "smart_better": better, "smart_worse": worse}
    logger.info(f"[shadow_backfill] {summary}")
    return summary


def _detector(bd: dict) -> str:
    return (bd.get("detector_source") if isinstance(bd, dict) else None) or "SMC"


def scorecard(sb, days: int = 45) -> dict:
    """Per-detector comparison of the ACTUAL exit vs the smart-exit SHADOW replay,
    over closed shadow-tagged trades that have been backfilled. Read-only. Shaped for
    the in-app screen: {days, overall{}, detectors[], closed_pulled, evaluated}."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    rows = []
    for off in range(0, 8000, 1000):
        chunk = (sb.table("signals")
                 .select("direction,result_pct,score_breakdown")
                 .eq("status", "closed").gte("created_at", since)
                 .order("created_at").range(off, off + 999).execute().data) or []
        rows += chunk
        if len(chunk) < 1000:
            break

    def _new():
        return {"n": 0, "actual": 0.0, "shadow": 0.0, "better": 0, "worse": 0,
                "awin": 0, "swin": 0, "reasons": defaultdict(int)}
    seg: dict = defaultdict(_new)
    overall = _new()
    evaluated = 0
    for r in rows:
        bd = r.get("score_breakdown")
        f = bd.get("smart_exit_shadow_final") if isinstance(bd, dict) else None
        actual = r.get("result_pct")
        if not (isinstance(f, dict) and f.get("pnl_pct") is not None and actual is not None):
            continue
        actual = float(actual); shadow = float(f["pnl_pct"]); evaluated += 1
        d = seg[_detector(bd)]
        for b in (d, overall):
            b["n"] += 1; b["actual"] += actual; b["shadow"] += shadow
            b["better"] += shadow > actual + 0.01
            b["worse"] += shadow < actual - 0.01
            b["awin"] += actual > 0; b["swin"] += shadow > 0
        d["reasons"][f.get("exit_reason") or "?"] += 1

    def _pack(b, name=None):
        n = b["n"] or 1
        o = {"n": b["n"], "actual_total": round(b["actual"], 1), "shadow_total": round(b["shadow"], 1),
             "delta_total": round(b["shadow"] - b["actual"], 1),
             "actual_avg": round(b["actual"] / n, 3), "shadow_avg": round(b["shadow"] / n, 3),
             "shadow_better": b["better"], "shadow_worse": b["worse"],
             "actual_win": round(100 * b["awin"] / n, 1), "shadow_win": round(100 * b["swin"] / n, 1)}
        if name is not None:
            o["detector"] = name
            o["exit_reasons"] = dict(b["reasons"])
        return o

    dets = sorted((_pack(v, k) for k, v in seg.items()), key=lambda x: x["delta_total"], reverse=True)
    return {"days": days, "overall": _pack(overall), "detectors": dets,
            "evaluated": evaluated, "closed_pulled": len(rows)}
