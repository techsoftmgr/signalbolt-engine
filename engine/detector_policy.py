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
    """Per (detector × regime bucket): the FULL learned outcome profile + a
    recommended multiplier + action. Reuses the scorecard engine (win-rate,
    payoff, MFE/give-back, winner-MAE, timing, alpha, market-beat) regrouped by
    bucket, then layers the policy decision on top. Advisory only."""
    from engine import scorecard
    res = scorecard.compute(rows or [], group_by="detector_bucket",
                            cost_pct=cost_pct, min_n=floor)
    out = []
    for s in res.get("segments", []):
        n = s.get("n", 0)
        # net expectancy already net of cost in the scorecard
        net_exp = s.get("expectancy_net")
        net_exp = round(float(net_exp), 3) if net_exp is not None else -99.0
        action, mult, note = _decide(net_exp, n, s.get("avg_alpha"), floor, confirm)
        out.append({
            "detector": s.get("detector"), "bucket": s.get("bucket"), "n": n,
            # decision
            "action": action, "recommended_multiplier": mult, "note": note,
            # the learned outcome profile (what we know about this detector here)
            "net_exp_pct": net_exp, "win_rate": s.get("win_rate"),
            "avg_win": s.get("avg_win"), "avg_loss": s.get("avg_loss"),
            "payoff": s.get("payoff"), "best_win": s.get("best_win"),
            "worst_loss": s.get("worst_loss"), "total_return_pct": s.get("total_return_pct"),
            "avg_mfe": s.get("avg_mfe"), "avg_giveback": s.get("avg_giveback"),
            "winner_mae": s.get("winner_mae"), "avg_t_mfe_min": s.get("avg_t_mfe_min"),
            "mae_before_mfe_pct": s.get("mae_before_mfe_pct"),
            "avg_alpha": s.get("avg_alpha"), "market_beat_rate": s.get("market_beat_rate"),
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
                .select("result_pct,score_breakdown,strategy_type,regime_type,confidence_score,status,created_at")
                .eq("status", "closed").gte("created_at", since).limit(5000).execute().data) or []
        policy = recommend(rows)                       # per detector × regime bucket (+ action)
        # per-detector OVERALL learning profile (all buckets combined) — the full
        # outcome picture for every detector the engine has fired.
        from engine import scorecard
        by_detector = scorecard.compute(rows, group_by="detector", min_n=_FLOOR).get("segments", [])
        return {
            "days": days, "enforced": False,
            "note": "ADVISORY — recommendations only; firing/sizing unchanged.",
            "floor": _FLOOR, "confirm": _CONFIRM, "cost_pct": _COST,
            "count": len(policy), "policy": policy,
            "by_detector": by_detector,
        }
    except Exception as e:
        logger.error(f"[detector_policy] compute failed: {e}")
        return {"days": days, "enforced": False, "count": 0, "policy": [], "error": str(e)}
