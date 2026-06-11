"""
Signal-combination scorecard — realized P&L (expectancy) for EVERY signal type,
plus the cross-detector combination studies we're evaluating.

Read-only. Never changes firing. Best-effort — never raises.

All studies run over CLOSED signals that have a real result_pct:
  • per_strategy  — P&L for every strategy_type (the "P&L for others")
  • volume        — does higher relativeVolume improve a directional break / reversal?
  • location      — do reversals fired NEAR the 20-day MA beat mid-air ones?
  • exit_stack    — when N independent exit-warnings fire on a position, how does it
                    end, and was the peak (MFE) already above the final exit?
  • divergence    — PENDING (needs a sector-ETF history join)

Each cell carries {n, win_pct, avg_pnl, avg_mfe, thin}. `thin` = sample below
MIN_CONFIDENT, so the UI can warn instead of letting anyone tune on noise.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone, timedelta

logger = logging.getLogger("signalbolt.combo_scorecard")

MIN_CONFIDENT = 30

# Directional break / reversal detectors that log relativeVolume.
_VOL_TAGGED = {"breakdown", "breakdown_forming", "distrib_forming", "peak_forming",
               "turn_forming", "accum_forming", "turnaround", "peak", "breakout_forming"}
_REVERSAL = {"turn_forming", "turnaround", "peak_forming", "distrib_forming"}
# Independent "consider booking" events fired on an open position.
_WARN_EVENTS = {"near_stop", "reversal", "counter_signal"}


def _sbd(s: dict) -> dict:
    v = s.get("score_breakdown")
    return v if isinstance(v, dict) else {}


def _agg(rows: list[dict]) -> dict:
    """PURE: {n, win_pct, avg_pnl, avg_mfe, thin} over rows with result_pct."""
    rows = [r for r in rows if r.get("result_pct") is not None]
    n = len(rows)
    if not n:
        return {"n": 0}
    pnls = [float(r["result_pct"]) for r in rows]
    avg = sum(pnls) / n
    win = sum(1 for p in pnls if p > 0) / n * 100
    mfes = [_sbd(r).get("mfe_pct") for r in rows if _sbd(r).get("mfe_pct") is not None]
    out = {"n": n, "avg_pnl": round(avg, 2), "win_pct": round(win), "thin": n < MIN_CONFIDENT}
    if mfes:
        out["avg_mfe"] = round(sum(mfes) / len(mfes), 2)
    return out


def _vol_bucket(v) -> str | None:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return "<1.0" if v < 1.0 else "1.0-1.5" if v < 1.5 else "1.5-2.0" if v < 2.0 else ">=2.0"


def scorecard(sb, days: int = 120) -> dict:
    """Build the combination scorecard. Best-effort — returns available:False on
    any data problem rather than raising."""
    if sb is None:
        return {"available": False, "note": "No data."}
    try:
        since = (datetime.now(timezone.utc) - timedelta(days=max(7, min(days, 365)))).isoformat()
        sigs = (sb.table("signals")
                .select("id,direction,strategy_type,entry_price,result_pct,score_breakdown,created_at")
                .eq("status", "closed").gte("created_at", since)
                .order("created_at", desc=True).limit(3000).execute().data) or []
    except Exception as e:
        logger.debug(f"[combo_scorecard] fetch failed: {e}")
        return {"available": False, "note": "Scorecard unavailable."}
    sigs = [s for s in sigs if s.get("result_pct") is not None]
    if not sigs:
        return {"available": False, "scored": 0, "note": "No closed signals in this window yet."}

    # 1) Per-strategy P&L — the "P&L for others", every signal type.
    by_strat: dict[str, list] = defaultdict(list)
    for s in sigs:
        by_strat[s.get("strategy_type") or "?"].append(s)
    per_strategy = sorted(
        [{"strategy": k, **_agg(v)} for k, v in by_strat.items()],
        key=lambda x: x.get("n", 0), reverse=True)

    # 2) Volume study — bucket vol-tagged directional types by relativeVolume.
    vbuckets: dict[str, list] = defaultdict(list)
    for s in sigs:
        if s.get("strategy_type") in _VOL_TAGGED:
            b = _vol_bucket(_sbd(s).get("relativeVolume"))
            if b:
                vbuckets[b].append(s)
    volume = [{"bucket": b, **_agg(vbuckets[b])}
              for b in ("<1.0", "1.0-1.5", "1.5-2.0", ">=2.0") if vbuckets[b]]

    # 3) Location study — reversal distance from the 20-day MA.
    lbuckets: dict[str, list] = defaultdict(list)
    for s in sigs:
        if s.get("strategy_type") in _REVERSAL and _sbd(s).get("ma20") and s.get("entry_price"):
            try:
                ma = float(_sbd(s)["ma20"]); ep = float(s["entry_price"])
                d = abs(ep - ma) / ma * 100 if ma else None
            except (TypeError, ValueError, ZeroDivisionError):
                d = None
            if d is not None:
                b = "near (<1%)" if d < 1 else "mid (1-3%)" if d < 3 else "far (>3%)"
                lbuckets[b].append(s)
    location = [{"bucket": b, **_agg(lbuckets[b])}
                for b in ("near (<1%)", "mid (1-3%)", "far (>3%)") if lbuckets[b]]

    # 4) Exit-conviction stack — # of independent warning events fired on a position.
    exit_stack: list = []
    peak_gap = None
    try:
        byid = {s["id"]: s for s in sigs}
        ids = list(byid.keys())
        warn_by: dict[str, set] = defaultdict(set)
        for i in range(0, len(ids), 50):
            evs = (sb.table("signal_events").select("signal_id,event_type")
                   .in_("signal_id", ids[i:i + 50]).execute().data) or []
            for e in evs:
                if e.get("event_type") in _WARN_EVENTS:
                    warn_by[e["signal_id"]].add(e["event_type"])
        cbuckets: dict[int, list] = defaultdict(list)
        for sid, s in byid.items():
            cbuckets[len(warn_by.get(sid, ()))].append(s)
        exit_stack = [{"warnings": c, **_agg(cbuckets[c])} for c in sorted(cbuckets)]
        warned = [s for sid, s in byid.items() if warn_by.get(sid)]
        gaps = [(_sbd(s)["mfe_pct"] - float(s["result_pct"]))
                for s in warned if _sbd(s).get("mfe_pct") is not None and s.get("result_pct") is not None]
        if gaps:
            peak_gap = round(sum(gaps) / len(gaps), 2)
    except Exception as e:
        logger.debug(f"[combo_scorecard] exit_stack failed: {e}")

    return {
        "available": True,
        "since": since,
        "scored": len(sigs),
        "min_confident": MIN_CONFIDENT,
        "per_strategy": per_strategy,
        "volume": volume,
        "location": location,
        "exit_stack": exit_stack,
        "exit_stack_peak_gap": peak_gap,   # avg pts peak(MFE) beat the final exit, on warned positions
        "divergence": {"available": False, "note": "Needs sector-ETF history join — scheduled."},
        "note": f"Realized P&L on CLOSED signals. Cells under n={MIN_CONFIDENT} are thin — don't tune on them.",
    }
