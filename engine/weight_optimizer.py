"""
Weight Optimizer
================
Finds the optimal L1-L5 scoring weights for each strategy using
Bayesian optimization (Optuna). Runs weekly via the scheduler.

Pipeline:
  1. Pull REAL outcomes from Supabase (closed signals with score_breakdown)
  2. Run BACKTESTER for synthetic training data (historical simulation)
  3. Combine both datasets
  4. For each (strategy, regime) combination:
     a. Run Optuna with 150 trials
     b. Objective: maximize win_rate × avg_rr × selectivity
     c. If improvement > 2% over current: save new weights
  5. Invalidate weight cache so scorer picks up new weights immediately

Minimum requirements before updating weights:
  - 30 combined data points (real + synthetic)
  - Optimizer must find strictly better objective than current weights

Budget note: Optuna is open-source. No API calls. Runs free on Railway.
"""

import logging
import os
from typing import Optional

import numpy as np

from engine import backtester
from engine import adaptive_weights

logger = logging.getLogger("signalbolt.optimizer")

# Minimum data points needed before we trust the optimization
MIN_SAMPLES = 30

# Optuna trial count (more = better weights, more CPU time)
N_TRIALS = 150

# Improvement threshold: only save new weights if objective improves by this %
MIN_IMPROVEMENT = 0.02   # 2%

# Layer raw maxima (used to normalize scores to 0-1)
_L_MAX = {
    'l1': 25.0,
    'l2': 25.0,
    'l3': 20.0,
    'l4': 15.0,
    'l5': 15.0,
}

# Strategy fire thresholds (from scorer.py)
_THRESHOLDS = {
    'scalping':     70,
    'day_trade':    65,
    'swing_trade':  68,
    'options_flow': 75,
    'dark_pool':    72,
}

# Tickers to run backtest on (representative subset — not all 30)
_BACKTEST_TICKERS: dict[str, list[str]] = {
    'scalping':     ['SPY', 'QQQ', 'AAPL', 'TSLA'],
    'day_trade':    ['SPY', 'QQQ', 'NVDA', 'AAPL', 'META', 'TSLA'],
    'swing_trade':  ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA'],
    'options_flow': ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA'],
    'dark_pool':    ['SPY', 'QQQ', 'JPM', 'AAPL', 'MSFT'],
}


# ---------------------------------------------------------------------------
# Pull real outcomes from Supabase
# ---------------------------------------------------------------------------

def _fetch_real_outcomes(strategy_type: str) -> list[dict]:
    """
    Pull closed signals that have score_breakdown stored.
    Returns list of {l1, l2, l3, l4, l5, outcome, risk_reward, regime_type}.
    """
    try:
        from supabase import create_client
        sb = create_client(
            os.environ["SUPABASE_URL"],
            os.environ.get("SUPABASE_KEY") or os.environ["SUPABASE_SECRET_KEY"],
        )
        rows = (
            sb.table("signals")
            .select("result, risk_reward, regime_type, score_breakdown")
            .eq("strategy_type", strategy_type)
            .eq("status", "closed")
            .not_.is_("score_breakdown", "null")
            .not_.is_("result", "null")
            .neq("result", "expired")
            .order("closed_at", desc=True)
            .limit(500)    # cap at 500 most recent
            .execute()
            .data
        ) or []

        points = []
        for row in rows:
            bd = row.get("score_breakdown") or {}
            if not bd:
                continue
            outcome = 1 if row.get("result") == "win" else 0
            points.append({
                'l1':          float(bd.get("l1_smc",       10)),
                'l2':          float(bd.get("l2_technical",  10)),
                'l3':          float(bd.get("l3_sentiment",  10)),
                'l4':          float(bd.get("l4_risk",        7)),
                'l5':          float(bd.get("l5_mtf",         7)),
                'outcome':     outcome,
                'risk_reward': float(row.get("risk_reward") or 0),
                'regime_type': row.get("regime_type") or "ANY",
            })
        logger.info(f"[optimizer] {strategy_type}: {len(points)} real outcome points from DB")
        return points
    except Exception as e:
        logger.warning(f"[optimizer] DB fetch failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Objective function (used by Optuna)
# ---------------------------------------------------------------------------

def _build_objective(points: list[dict], threshold: int):
    """
    Returns an Optuna objective function that maximizes win_rate × avg_rr.

    The trial proposes weights {smc, technical, sentiment, risk}.
    We normalize them so they sum to 100, compute weighted scores for
    all data points, classify as "would fire" (score >= threshold),
    then measure quality of those fired signals.
    """

    def objective(trial):
        # Sample weights (Dirichlet-like via uniform + normalize)
        w_smc  = trial.suggest_float("smc",       5.0,  55.0)
        w_tech = trial.suggest_float("technical",  5.0,  55.0)
        w_sent = trial.suggest_float("sentiment",  5.0,  55.0)
        w_risk = trial.suggest_float("risk",       5.0,  35.0)
        total  = w_smc + w_tech + w_sent + w_risk
        # Normalize so they sum to 100
        w_smc  = w_smc  / total * 100
        w_tech = w_tech / total * 100
        w_sent = w_sent / total * 100
        w_risk = w_risk / total * 100

        fired_outcomes = []
        fired_rrs      = []

        for p in points:
            # Compute weighted score (same formula as scorer.py)
            score = (
                (p['l1'] / _L_MAX['l1']) * w_smc  +
                (p['l2'] / _L_MAX['l2']) * w_tech +
                (p['l3'] / _L_MAX['l3']) * w_sent +
                (p['l4'] / _L_MAX['l4']) * w_risk
            )
            # L5 fixed bonus (not being optimized in this run)
            score += (p['l5'] / _L_MAX['l5']) * 5.0

            if score >= threshold:
                fired_outcomes.append(p['outcome'])
                rr = p['risk_reward'] if p['outcome'] == 1 else -1.0
                fired_rrs.append(rr)

        n_fired = len(fired_outcomes)
        if n_fired < 5:
            return 0.0   # too selective — penalize

        win_rate = sum(fired_outcomes) / n_fired
        avg_rr   = float(np.mean(fired_rrs)) if fired_rrs else 0.0

        # Selectivity bonus: we want to fire good signals, not everything
        # Ideal fire rate is 10-30% of candidates
        total_candidates = len(points)
        fire_rate = n_fired / total_candidates if total_candidates > 0 else 0
        selectivity = 1.0
        if fire_rate > 0.5:
            selectivity = 0.7     # penalize if firing too many
        elif fire_rate < 0.05:
            selectivity = 0.8     # penalize if too strict

        return max(0.0, win_rate * max(avg_rr, 0) * selectivity)

    return objective


# ---------------------------------------------------------------------------
# Single strategy + regime optimization
# ---------------------------------------------------------------------------

def _optimize_one(
    strategy_type: str,
    regime_type: str,
    points: list[dict],
) -> Optional[dict]:
    """
    Run Optuna on a single (strategy, regime) combination.
    Returns new weights dict if they improve on current, else None.
    """
    if len(points) < MIN_SAMPLES:
        logger.info(
            f"[optimizer] {strategy_type}/{regime_type}: "
            f"only {len(points)} points < {MIN_SAMPLES} min — skipping"
        )
        return None

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.error("[optimizer] optuna not installed — run: pip install optuna")
        return None

    threshold = _THRESHOLDS.get(strategy_type, 70)

    # ── Evaluate current weights ─────────────────────────────────────────────
    current_weights = adaptive_weights.get_weights(strategy_type, regime_type)
    current_obj     = _evaluate_weights(current_weights, points, threshold)

    # ── Run Optuna study ─────────────────────────────────────────────────────
    study = optuna.create_study(direction="maximize")
    study.optimize(
        _build_objective(points, threshold),
        n_trials=N_TRIALS,
        show_progress_bar=False,
    )

    best = study.best_trial
    if best.value is None or best.value <= 0:
        logger.info(f"[optimizer] {strategy_type}/{regime_type}: no valid solution found")
        return None

    # ── Check if new weights actually improve ───────────────────────────────
    improvement = (best.value - current_obj) / max(current_obj, 0.001)
    if improvement < MIN_IMPROVEMENT:
        logger.info(
            f"[optimizer] {strategy_type}/{regime_type}: "
            f"improvement {improvement:.1%} < {MIN_IMPROVEMENT:.0%} threshold — keeping current"
        )
        return None

    # ── Normalize and package new weights ───────────────────────────────────
    p     = best.params
    total = p["smc"] + p["technical"] + p["sentiment"] + p["risk"]
    new_weights = current_weights.copy()   # preserve L5-L9 bonus values
    new_weights["smc"]       = round(p["smc"]       / total * 100, 1)
    new_weights["technical"] = round(p["technical"] / total * 100, 1)
    new_weights["sentiment"] = round(p["sentiment"] / total * 100, 1)
    new_weights["risk"]      = round(p["risk"]      / total * 100, 1)

    # Compute metrics for storage
    wins     = [p2 for p2 in points if p2["outcome"] == 1]
    losses   = [p2 for p2 in points if p2["outcome"] == 0]
    win_rate = len(wins) / len(points)
    avg_rr   = float(np.mean([p2["risk_reward"] for p2 in wins])) if wins else 0.0

    metrics = {
        "win_rate":     round(win_rate, 4),
        "avg_rr":       round(avg_rr, 3),
        "objective":    round(best.value, 4),
        "sample_count": len(points),
        "prev_objective": round(current_obj, 4),
        "improvement":  round(improvement, 4),
    }

    logger.info(
        f"[optimizer] {strategy_type}/{regime_type}: "
        f"NEW WEIGHTS saved — objective {current_obj:.3f} → {best.value:.3f} "
        f"(+{improvement:.1%})  win_rate={win_rate:.1%}  avg_rr={avg_rr:.2f}  n={len(points)}"
    )
    return {"weights": new_weights, "metrics": metrics}


def _evaluate_weights(weights: dict, points: list[dict], threshold: int) -> float:
    """Evaluate an existing weight dict against training data. Returns objective value."""
    w_smc  = weights.get("smc",       25)
    w_tech = weights.get("technical", 25)
    w_sent = weights.get("sentiment", 25)
    w_risk = weights.get("risk",      25)

    fired_outcomes = []
    fired_rrs      = []

    for p in points:
        score = (
            (p['l1'] / _L_MAX['l1']) * w_smc  +
            (p['l2'] / _L_MAX['l2']) * w_tech +
            (p['l3'] / _L_MAX['l3']) * w_sent +
            (p['l4'] / _L_MAX['l4']) * w_risk +
            (p['l5'] / _L_MAX['l5']) * 5.0
        )
        if score >= threshold:
            fired_outcomes.append(p['outcome'])
            fired_rrs.append(p['risk_reward'] if p['outcome'] == 1 else -1.0)

    n = len(fired_outcomes)
    if n < 5:
        return 0.0
    win_rate = sum(fired_outcomes) / n
    avg_rr   = float(np.mean(fired_rrs)) if fired_rrs else 0.0
    fire_rate = n / len(points)
    selectivity = 0.7 if fire_rate > 0.5 else (0.8 if fire_rate < 0.05 else 1.0)
    return max(0.0, win_rate * max(avg_rr, 0) * selectivity)


# ---------------------------------------------------------------------------
# Main entry point — runs all strategies
# ---------------------------------------------------------------------------

def run_full_optimization() -> dict:
    """
    Run the complete optimization cycle. Called weekly by the scheduler.

    Returns a summary dict:
        {strategy: {'regime': {'updated': bool, 'objective': float}}}
    """
    logger.info("[optimizer] ═══ Weekly weight optimization started ═══")
    summary = {}

    strategies = ['scalping', 'day_trade', 'swing_trade', 'options_flow', 'dark_pool']

    for strategy in strategies:
        summary[strategy] = {}
        logger.info(f"[optimizer] ── {strategy.upper()} ──")

        # ── Step 1: Pull real outcomes from DB ──────────────────────────────
        real_points = _fetch_real_outcomes(strategy)

        # ── Step 2: Run backtester for synthetic data ────────────────────────
        tickers = _BACKTEST_TICKERS.get(strategy, ['SPY', 'QQQ', 'AAPL'])
        try:
            bt_points_raw = backtester.run(tickers, strategy, max_points_per_ticker=40)
            # Convert TrainingPoint dataclasses to dict format
            synthetic_points = [
                {
                    'l1': p.l1_raw, 'l2': p.l2_raw, 'l3': p.l3_raw,
                    'l4': p.l4_raw, 'l5': p.l5_raw,
                    'outcome': p.outcome,
                    'risk_reward': p.risk_reward,
                    'regime_type': 'ANY',
                }
                for p in bt_points_raw if p.outcome != -1
            ]
        except Exception as e:
            logger.warning(f"[optimizer] Backtester failed for {strategy}: {e}")
            synthetic_points = []

        all_points = real_points + synthetic_points
        logger.info(
            f"[optimizer] {strategy}: {len(real_points)} real + "
            f"{len(synthetic_points)} synthetic = {len(all_points)} total"
        )

        # ── Step 3: Optimize for ALL-REGIME combined ────────────────────────
        result = _optimize_one(strategy, "ANY", all_points)
        if result:
            adaptive_weights.save_weights(
                strategy, "ANY", result["weights"], result["metrics"]
            )
            summary[strategy]["ANY"] = {
                "updated":   True,
                "objective": result["metrics"]["objective"],
                "win_rate":  result["metrics"]["win_rate"],
                "n":         result["metrics"]["sample_count"],
            }
        else:
            summary[strategy]["ANY"] = {"updated": False}

        # ── Step 4: Optimize per regime (if enough data) ────────────────────
        regimes_found = set(p.get("regime_type", "ANY") for p in real_points)
        for regime in regimes_found:
            if regime in ("ANY", "", None):
                continue
            regime_points = [p for p in all_points if p.get("regime_type") == regime]
            if len(regime_points) < MIN_SAMPLES:
                continue
            result = _optimize_one(strategy, regime, regime_points)
            if result:
                adaptive_weights.save_weights(
                    strategy, regime, result["weights"], result["metrics"]
                )
                summary[strategy][regime] = {
                    "updated":   True,
                    "objective": result["metrics"]["objective"],
                    "win_rate":  result["metrics"]["win_rate"],
                    "n":         result["metrics"]["sample_count"],
                }

    # Invalidate cache so all scorer calls pick up new weights
    adaptive_weights.invalidate_cache()

    # ── Summary log ─────────────────────────────────────────────────────────
    updated_count = sum(
        1 for strat in summary.values()
        for combo in strat.values()
        if isinstance(combo, dict) and combo.get("updated")
    )
    logger.info(
        f"[optimizer] ═══ Optimization complete — "
        f"{updated_count} weight sets updated ═══"
    )
    return summary
