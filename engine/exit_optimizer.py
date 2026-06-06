"""
Exit / SL-TP optimizer — learns the best stop / target / trail / breakeven
policy per detector by REPLAYING real SIP bars (engine/replay_backtest), with the
same discipline as weight_optimizer: walk-forward 70/30, beat the AS-TRADED
baseline on BOTH splits by ≥ MIN_IMPROVEMENT, and a hard sample floor.

⚠️ ADVISORY / DATA-GATED. Returns the learned policy + the improvement vs
as-traded; it does NOT change sl_tp_engine or any exit logic. Until a detector
has ≥ MIN_SAMPLES closed signals it returns status="insufficient_data". When the
data matures + the user approves, the chosen policy can be wired into
sl_tp_engine / the monitors. Bars are fetched ONCE per signal and reused across
every candidate (cheap replays, one I/O pass).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

from engine import replay_backtest as rb

logger = logging.getLogger("signalbolt.exit_optimizer")

MIN_SAMPLES = 30
MIN_IMPROVEMENT = 0.10      # candidate must beat as-traded expectancy by ≥0.10%/trade
TRAIN_FRACTION = 0.70

# Curated candidate exit policies (distances are % of entry). Kept small + sane;
# the search reports the best that ALSO survives out-of-sample.
CANDIDATES: list[dict] = [
    {"label": "2:1 fixed",          "stop_pct": 3.0, "target_pct": 6.0},
    {"label": "tight 2:1",          "stop_pct": 2.5, "target_pct": 5.0},
    {"label": "wide 2:1",           "stop_pct": 4.0, "target_pct": 8.0},
    {"label": "3% trail",           "stop_pct": 3.0, "trail_pct": 3.0},
    {"label": "wide trail",         "stop_pct": 4.0, "trail_pct": 5.0},
    {"label": "2:1 + breakeven",    "stop_pct": 3.0, "target_pct": 6.0, "breakeven_at_pct": 3.0},
    {"label": "trail + breakeven",  "stop_pct": 4.0, "trail_pct": 4.0, "breakeven_at_pct": 4.0},
    {"label": "trail + time-stop",  "stop_pct": 3.5, "trail_pct": 3.0, "time_stop_bars": 20},
]


def _expectancy(loaded: list, params: dict) -> float | None:
    """Mean realized-net % of replaying `loaded` [(signal, bars)] under params."""
    xs = []
    for sig, bars in loaded:
        out = rb.replay(sig.get("direction"), sig.get("entry_price"), bars, params)
        if out.get("outcome") is not None:
            xs.append(out["realized_net_pct"])
    return round(sum(xs) / len(xs), 3) if xs else None


def optimize(sb, detector: str, days: int = 90, horizon_days: int = 7,
             timeframe: str = "15Min", min_n: int = MIN_SAMPLES) -> dict:
    """Walk-forward best-policy search for one detector. Never raises."""
    try:
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = (sb.table("signals")
                .select("ticker,direction,entry_price,result_pct,created_at,score_breakdown,strategy_type")
                .eq("status", "closed").gte("created_at", since)
                .neq("strategy_type", "deep_value").order("created_at")
                .limit(2000).execute().data) or []
        rows = [r for r in rows
                if ((r.get("score_breakdown") or {}).get("detector_source") == detector
                    or r.get("strategy_type") == detector)]

        # one bar-fetch per signal, reused across all candidates
        loaded = []
        for r in rows:
            bars = rb.fetch_forward_bars(r.get("ticker"), r.get("created_at"),
                                         horizon_days, timeframe)
            if bars:
                loaded.append((r, bars))
        n = len(loaded)
        if n < min_n:
            return {"detector": detector, "status": "insufficient_data",
                    "n": n, "need": min_n,
                    "note": f"{n}/{min_n} replayable signals — keep collecting"}

        split = int(n * TRAIN_FRACTION)
        train, test = loaded[:split], loaded[split:]
        as_traded = [float(r["result_pct"]) for r, _ in loaded if r.get("result_pct") is not None]
        base_train = ([float(r["result_pct"]) for r, _ in train if r.get("result_pct") is not None])
        base_test  = ([float(r["result_pct"]) for r, _ in test  if r.get("result_pct") is not None])
        base_train_e = round(sum(base_train) / len(base_train), 3) if base_train else None
        base_test_e  = round(sum(base_test) / len(base_test), 3) if base_test else None

        scored = []
        for cand in CANDIDATES:
            params = {k: v for k, v in cand.items() if k != "label"}
            tr = _expectancy(train, params)
            scored.append((cand["label"], params, tr))
        scored = [s for s in scored if s[2] is not None]
        if not scored:
            return {"detector": detector, "status": "no_valid", "n": n}
        best_label, best_params, best_train_e = max(scored, key=lambda s: s[2])
        best_test_e = _expectancy(test, best_params)

        beats_train = base_train_e is not None and (best_train_e - base_train_e) >= MIN_IMPROVEMENT
        beats_test  = base_test_e is not None and best_test_e is not None and (best_test_e - base_test_e) >= MIN_IMPROVEMENT
        passes = bool(beats_train and beats_test)

        return {
            "detector": detector, "status": "ok", "n": n,
            "best_policy": best_label, "best_params": best_params,
            "best_train_exp": best_train_e, "best_test_exp": best_test_e,
            "as_traded_train_exp": base_train_e, "as_traded_test_exp": base_test_e,
            "beats_baseline_oos": passes,
            "recommendation": ("ADOPT (advisory)" if passes
                               else "KEEP as-traded — no validated improvement"),
            "all_candidates": [{"label": l, "train_exp": e} for l, _, e in
                               sorted(scored, key=lambda s: -(s[2] or -9))],
            "enforced": False,
        }
    except Exception as e:
        logger.error(f"[exit_optimizer] {detector} failed: {e}")
        return {"detector": detector, "status": "error", "error": str(e)}
