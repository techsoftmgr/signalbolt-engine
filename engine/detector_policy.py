"""
Detector policy recommender — the automated counterpart to the manual scorecard
keep/cut, for the PREDICTIVE detectors (which the weight-optimizer does NOT
cover). It reads realized net expectancy + alpha per (detector × regime bucket)
and recommends a SIZE MULTIPLIER + action.

⚠️ ADVISORY ONLY — this computes what an auto-tuner WOULD do; it does NOT change
firing or sizing. Enforcement (the fire path reading these multipliers) is a
separate, explicitly-approved step. Design lives in the module docstring below.

Guardrails baked into the recommendation (measure-first):
  • SAMPLE FLOOR — under `floor` judged trades → MEASURING (keep full size, keep
    collecting); never act on noise.
  • THROTTLE, NOT KILL — a confirmed loser is shrunk to 0.25×, never zeroed, so
    the cell keeps producing data to confirm/deny the call (the forming-guard
    lesson). Enforcement would add hysteresis + K-consecutive-reads before any
    further cut.
  • PER BUCKET (RISK_ON/NEUTRAL/RISK_OFF) so cells reach the floor — the
    automation of the regime→detector matrix.

ENFORCEMENT SKETCH (future, approved): persist these rows to a `detector_policy`
table; at fire time `forming_signals`/detector paths look up
(detector, bucket_of(current_regime)) → multiplier; apply to position_multiplier
(0 only if a SUPPRESS action is ever added after K consecutive confirmed-negative
reads). Log every applied change; a weekly job recomputes. Until then: read-only.

`recommend()` is PURE (unit-tested). `compute()` fetches + recommends.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone, timedelta

from engine.regime_buckets import bucket_of

logger = logging.getLogger("signalbolt.detector_policy")

_FLOOR   = 20    # min judged trades before we act at all
_CONFIRM = 40    # min judged trades before throttling a loser
_COST    = 0.10  # round-trip cost % per trade


def _decide(net_exp: float, n: int, avg_alpha, floor: int, confirm: int) -> tuple:
    """(action, multiplier, note) from a cell's stats. Throttle-not-kill."""
    if n < floor:
        return "MEASURING", 1.0, f"only {n} trades — keep full size, keep collecting"
    if net_exp >= 0.10:
        if avg_alpha is not None and avg_alpha <= 0:
            return "FULL", 1.0, "+EV but BETA-only (alpha≤0) — made money riding the tape"
        return "FULL", 1.0, "+EV with real alpha"
    if net_exp >= -0.05:
        return "HALF", 0.5, "marginal edge — size down, keep measuring"
    if n >= confirm:
        return "THROTTLE", 0.25, f"negative over {n} trades — shrink (not kill) to keep confirming"
    return "HALF", 0.5, "negative but thin sample — cautious half-size"


def recommend(rows: list, cost_pct: float = _COST,
              floor: int = _FLOOR, confirm: int = _CONFIRM) -> list:
    """Pure: per (detector × regime bucket) → recommended multiplier + action."""
    groups: dict = defaultdict(list)
    for r in rows or []:
        pct = r.get("result_pct")
        if pct is None:
            continue
        try:
            pct = float(pct)
        except (TypeError, ValueError):
            continue
        sbd = r.get("score_breakdown") if isinstance(r.get("score_breakdown"), dict) else {}
        det = sbd.get("detector_source") or r.get("strategy_type") or "NA"
        bkt = bucket_of(r.get("regime_type"))
        groups[(det, bkt)].append((pct, sbd.get("alpha_pct")))

    out = []
    for (det, bkt), pts in groups.items():
        n = len(pts)
        pcts = [p for p, _ in pts]
        net_exp = round(sum(pcts) / n - cost_pct, 3)
        wins = sum(1 for p in pcts if p > 0)
        alphas = [a for _, a in pts if a is not None]
        avg_alpha = round(sum(float(a) for a in alphas) / len(alphas), 2) if alphas else None
        action, mult, note = _decide(net_exp, n, avg_alpha, floor, confirm)
        out.append({
            "detector": det, "bucket": bkt, "n": n,
            "net_exp_pct": net_exp, "win_rate": round(100 * wins / n, 1),
            "avg_alpha": avg_alpha,
            "action": action, "recommended_multiplier": mult, "note": note,
        })
    # worst first (what to look at): throttles/halves on top, then by net_exp asc
    rank = {"THROTTLE": 0, "HALF": 1, "MEASURING": 2, "FULL": 3}
    out.sort(key=lambda x: (rank.get(x["action"], 9), x["net_exp_pct"]))
    return out


def compute(sb, days: int = 45) -> dict:
    """Fetch closed signals + build the advisory policy. Never raises."""
    try:
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = (sb.table("signals")
                .select("result_pct,score_breakdown,strategy_type,regime_type,status,created_at")
                .eq("status", "closed").gte("created_at", since).limit(5000).execute().data) or []
        policy = recommend(rows)
        return {
            "days": days, "enforced": False,
            "note": "ADVISORY — recommendations only; firing/sizing unchanged.",
            "floor": _FLOOR, "confirm": _CONFIRM, "cost_pct": _COST,
            "count": len(policy), "policy": policy,
        }
    except Exception as e:
        logger.error(f"[detector_policy] compute failed: {e}")
        return {"days": days, "enforced": False, "count": 0, "policy": [], "error": str(e)}
