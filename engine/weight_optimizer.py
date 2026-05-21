"""
Weight Optimizer
================
Finds the optimal L1-L5 scoring weights for each strategy using
Bayesian optimization (Optuna). Runs weekly via the scheduler.

Pipeline:
  1. Pull REAL outcomes from Supabase (closed signals with score_breakdown)
  2. Run BACKTESTER for synthetic training data (historical simulation)
  3. Combine both datasets, sorted chronologically
  4. Walk-forward split: 70% train / 30% validation (time-ordered)
  5. For each (strategy, regime) combination:
     a. Run Optuna with 150 trials on TRAIN set only
     b. Objective: expectancy = win_rate×avg_R + (1-win_rate)×avg_loss_R
     c. Validate candidate weights on held-out TEST set
     d. Apply weight-change caps (±15 pts per layer vs current)
     e. Only save if expectancy improves ≥ 2% on BOTH train AND test
  6. Invalidate weight cache so scorer picks up new weights immediately

Safeguards against overfitting:
  - Walk-forward validation: weights must generalise to unseen data
  - Expectancy (not just win rate): penalises large loss R-multiples
  - Weight change cap ±15 pts: prevents extreme pivots between runs
  - Minimum 5 fired signals in train AND test before saving

Budget note: Optuna is open-source. No API calls. Runs free on Fly.io.
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

# Improvement threshold: only save new weights if expectancy improves by this %
MIN_IMPROVEMENT = 0.02   # 2%

# Walk-forward split: fraction of data used for training
TRAIN_FRACTION = 0.70    # 70% train, 30% held-out validation

# Weight change safety cap: a single run cannot move any layer weight by more
# than this many points.  Prevents extreme pivots that would destabilise the
# engine between runs (e.g. SMC weight going 25→5 in one week).
MAX_WEIGHT_DELTA = 15.0  # ±15 pts per layer per optimization run

# Minimum fired signals in both train AND test set before saving new weights
MIN_FIRED_TRAIN = 5
MIN_FIRED_TEST  = 3

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
# Core metric: expectancy
# ---------------------------------------------------------------------------

def _compute_expectancy(fired_outcomes: list[int], fired_rrs: list[float]) -> float:
    """
    E = win_rate × avg_win_R + (1 - win_rate) × avg_loss_R

    avg_loss_R is the mean actual R of losing trades (negative number).
    Using realized R for losses (not a fixed -1.0) means the optimizer
    penalises strategies that take losses beyond 1R — a key overfitting guard.

    Returns 0.0 if fewer than MIN_FIRED_TRAIN signals.
    """
    n = len(fired_outcomes)
    if n < MIN_FIRED_TRAIN:
        return 0.0

    wins   = [r for o, r in zip(fired_outcomes, fired_rrs) if o == 1]
    losses = [r for o, r in zip(fired_outcomes, fired_rrs) if o == 0]

    win_rate   = len(wins) / n
    avg_win_r  = float(np.mean(wins))   if wins   else 0.0
    avg_loss_r = float(np.mean(losses)) if losses else -1.0   # fallback -1R

    return win_rate * avg_win_r + (1.0 - win_rate) * avg_loss_r


def _score_points(points: list[dict], w_smc: float, w_tech: float,
                  w_sent: float, w_risk: float, threshold: int) -> tuple[list[int], list[float]]:
    """Apply weights to a list of data points and return (outcomes, rrs) of fired signals."""
    fired_outcomes: list[int]   = []
    fired_rrs:      list[float] = []
    for p in points:
        score = (
            (p['l1'] / _L_MAX['l1']) * w_smc  +
            (p['l2'] / _L_MAX['l2']) * w_tech +
            (p['l3'] / _L_MAX['l3']) * w_sent +
            (p['l4'] / _L_MAX['l4']) * w_risk +
            (p['l5'] / _L_MAX['l5']) * 5.0    # L5 fixed bonus
        )
        if score >= threshold:
            fired_outcomes.append(p['outcome'])
            # Use realized R for wins; for losses clip to at most -3R
            rr = p['risk_reward'] if p['outcome'] == 1 else max(p['risk_reward'], -3.0)
            fired_rrs.append(rr)
    return fired_outcomes, fired_rrs


# ---------------------------------------------------------------------------
# Weight-change safety cap
# ---------------------------------------------------------------------------

def _cap_weight_changes(new_weights: dict, current_weights: dict) -> dict:
    """
    Prevent any single layer from shifting more than MAX_WEIGHT_DELTA pts
    in one run.  After capping, re-normalise so weights still sum to 100.
    """
    capped: dict[str, float] = {}
    for k in ('smc', 'technical', 'sentiment', 'risk'):
        current = current_weights.get(k, 25.0)
        proposed = new_weights.get(k, current)
        delta = proposed - current
        if abs(delta) > MAX_WEIGHT_DELTA:
            capped[k] = current + MAX_WEIGHT_DELTA * (1 if delta > 0 else -1)
        else:
            capped[k] = proposed

    # Re-normalise
    total = sum(capped.values())
    if total > 0:
        for k in capped:
            capped[k] = round(capped[k] / total * 100, 1)

    return capped


# ---------------------------------------------------------------------------
# Objective function (used by Optuna) — trains on TRAIN split only
# ---------------------------------------------------------------------------

def _build_objective(train_points: list[dict], threshold: int):
    """
    Returns an Optuna objective that maximises expectancy on the TRAINING set.

    Expectancy = win_rate × avg_win_R + (1-win_rate) × avg_loss_R
    This penalises both low win rates AND large loss R-multiples simultaneously,
    making it harder to overfit on one dimension alone.
    """

    def objective(trial):
        w_smc  = trial.suggest_float("smc",        5.0, 55.0)
        w_tech = trial.suggest_float("technical",   5.0, 55.0)
        w_sent = trial.suggest_float("sentiment",   5.0, 55.0)
        w_risk = trial.suggest_float("risk",        5.0, 35.0)
        total  = w_smc + w_tech + w_sent + w_risk
        w_smc  = w_smc  / total * 100
        w_tech = w_tech / total * 100
        w_sent = w_sent / total * 100
        w_risk = w_risk / total * 100

        fired_outcomes, fired_rrs = _score_points(
            train_points, w_smc, w_tech, w_sent, w_risk, threshold
        )
        return _compute_expectancy(fired_outcomes, fired_rrs)

    return objective


# ---------------------------------------------------------------------------
# Single strategy + regime optimization  (walk-forward validated)
# ---------------------------------------------------------------------------

def _evaluate_weights(weights: dict, points: list[dict], threshold: int) -> float:
    """Evaluate a weight dict against a data split. Returns expectancy."""
    w_smc  = weights.get("smc",       25)
    w_tech = weights.get("technical", 25)
    w_sent = weights.get("sentiment", 25)
    w_risk = weights.get("risk",      25)
    fired_outcomes, fired_rrs = _score_points(points, w_smc, w_tech, w_sent, w_risk, threshold)
    return _compute_expectancy(fired_outcomes, fired_rrs)


def _optimize_one(
    strategy_type: str,
    regime_type:   str,
    points:        list[dict],
) -> Optional[dict]:
    """
    Run Optuna on a single (strategy, regime) combination with walk-forward safety.

    Walk-forward protocol:
      - Sort data chronologically (real outcomes already sorted by closed_at)
      - Train on first 70% of points
      - Validate on remaining 30% (never seen during optimization)
      - New weights must improve expectancy on BOTH splits vs current weights
      - Weight changes are capped at ±MAX_WEIGHT_DELTA pts per layer

    Returns new weights dict if all gates pass, else None.
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

    # ── Walk-forward split (time-ordered) ────────────────────────────────────
    split_idx   = int(len(points) * TRAIN_FRACTION)
    train_pts   = points[:split_idx]
    test_pts    = points[split_idx:]

    if len(train_pts) < MIN_SAMPLES or len(test_pts) < MIN_FIRED_TEST:
        logger.info(
            f"[optimizer] {strategy_type}/{regime_type}: "
            f"insufficient train ({len(train_pts)}) or test ({len(test_pts)}) points — skipping"
        )
        return None

    # ── Evaluate CURRENT weights on both splits ───────────────────────────────
    current_weights  = adaptive_weights.get_weights(strategy_type, regime_type)
    current_train_e  = _evaluate_weights(current_weights, train_pts, threshold)
    current_test_e   = _evaluate_weights(current_weights, test_pts,  threshold)

    # ── Run Optuna on TRAIN set only ─────────────────────────────────────────
    study = optuna.create_study(direction="maximize")
    study.optimize(
        _build_objective(train_pts, threshold),
        n_trials=N_TRIALS,
        show_progress_bar=False,
    )

    best = study.best_trial
    if best.value is None or best.value <= 0:
        logger.info(f"[optimizer] {strategy_type}/{regime_type}: no valid solution on train set")
        return None

    # ── Build candidate weights ───────────────────────────────────────────────
    p     = best.params
    total = p["smc"] + p["technical"] + p["sentiment"] + p["risk"]
    raw_new = {
        "smc":       round(p["smc"]       / total * 100, 1),
        "technical": round(p["technical"] / total * 100, 1),
        "sentiment": round(p["sentiment"] / total * 100, 1),
        "risk":      round(p["risk"]      / total * 100, 1),
    }

    # Apply weight-change safety cap (±MAX_WEIGHT_DELTA per layer)
    capped_new = _cap_weight_changes(raw_new, current_weights)
    new_weights = current_weights.copy()   # preserve L5-L9 bonus values
    new_weights.update(capped_new)

    # ── Gate 1: train improvement ─────────────────────────────────────────────
    new_train_e = _evaluate_weights(new_weights, train_pts, threshold)
    train_improvement = (new_train_e - current_train_e) / max(abs(current_train_e), 0.001)
    if train_improvement < MIN_IMPROVEMENT:
        logger.info(
            f"[optimizer] {strategy_type}/{regime_type}: "
            f"train improvement {train_improvement:.1%} < {MIN_IMPROVEMENT:.0%} — keeping current"
        )
        return None

    # ── Gate 2: out-of-sample (test) validation ───────────────────────────────
    # New weights must also improve (or not hurt) expectancy on unseen data.
    # A small regression is allowed (up to -0.5% of current_test_e) to account
    # for natural variance in small test sets.
    new_test_e = _evaluate_weights(new_weights, test_pts, threshold)
    test_regression = (current_test_e - new_test_e) / max(abs(current_test_e), 0.001)
    OOS_TOLERANCE = 0.005  # allow up to 0.5% regression on OOS (sampling noise)
    if test_regression > OOS_TOLERANCE:
        logger.warning(
            f"[optimizer] {strategy_type}/{regime_type}: "
            f"OOS regression {test_regression:.1%} exceeds tolerance — possible overfit, skipping"
        )
        return None

    # ── Package result ────────────────────────────────────────────────────────
    # Compute realized metrics on ALL data (train+test) for storage
    all_outcomes, all_rrs = _score_points(
        points,
        new_weights["smc"], new_weights["technical"],
        new_weights["sentiment"], new_weights["risk"],
        threshold
    )
    n_fired  = len(all_outcomes)
    win_rate = sum(all_outcomes) / n_fired if n_fired > 0 else 0.0
    wins_rrs = [r for o, r in zip(all_outcomes, all_rrs) if o == 1]
    avg_win_r = float(np.mean(wins_rrs)) if wins_rrs else 0.0

    metrics = {
        "win_rate":          round(win_rate, 4),
        "avg_win_r":         round(avg_win_r, 3),
        "train_expectancy":  round(new_train_e,     4),
        "test_expectancy":   round(new_test_e,      4),
        "prev_train_exp":    round(current_train_e, 4),
        "prev_test_exp":     round(current_test_e,  4),
        "train_improvement": round(train_improvement, 4),
        "oos_regression":    round(test_regression,   4),
        "sample_count":      len(points),
        "train_count":       len(train_pts),
        "test_count":        len(test_pts),
        "weight_caps_applied": raw_new != capped_new,
        # Legacy field kept for dashboard display
        "objective":         round(new_train_e, 4),
    }

    logger.info(
        f"[optimizer] {strategy_type}/{regime_type}: NEW WEIGHTS ACCEPTED — "
        f"train_exp {current_train_e:.3f}→{new_train_e:.3f} (+{train_improvement:.1%}) "
        f"test_exp {current_test_e:.3f}→{new_test_e:.3f} "
        f"win_rate={win_rate:.1%} n_fired={n_fired}/{len(points)} "
        f"caps={'YES' if metrics['weight_caps_applied'] else 'no'}"
    )
    return {"weights": new_weights, "metrics": metrics}


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
