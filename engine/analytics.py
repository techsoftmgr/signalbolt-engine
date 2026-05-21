"""
Analytics & Validation Reporter
================================
Generates win-rate, R-multiple, drawdown, and false-positive reports
from closed signals in Supabase.

Answers key business questions:
  - Which strategy has the best win rate?
  - Which setup type performs best in trending vs ranging markets?
  - What is the average R-multiple realized per signal?
  - Which tickers consistently lose?
  - Which time sessions perform best?
  - Is the WATCHLIST → CONFIRMED conversion rate improving?

All metrics are computed from actual closed signal data — NOT backtested.
Reports are stored in signal_analytics_cache for fast API access.

Run daily after market close (5:30 PM ET) via APScheduler.
"""

from __future__ import annotations

import logging
import statistics
from datetime import date, datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger("signalbolt.analytics")


# ── Core metric computation ───────────────────────────────────────────────────

def _compute_metrics(signals: list[dict]) -> dict:
    """
    Compute core performance metrics from a list of closed signal dicts.
    Requires: result ('win'/'loss'/'expired'), result_pct, risk_reward columns.
    """
    if not signals:
        return {
            "total_signals": 0, "win_count": 0, "loss_count": 0,
            "win_rate": None, "avg_r": None, "median_r": None,
            "profit_factor": None, "max_drawdown": None, "expectancy": None,
        }

    wins   = [s for s in signals if s.get("result") == "win"]
    losses = [s for s in signals if s.get("result") == "loss"]
    total  = len(wins) + len(losses)

    if total == 0:
        return {
            "total_signals": len(signals), "win_count": 0, "loss_count": 0,
            "win_rate": None, "avg_r": None, "median_r": None,
            "profit_factor": None, "max_drawdown": None, "expectancy": None,
        }

    win_rate = len(wins) / total

    # R-multiples: result_pct / (entry_price * risk_pct) — approximated by result_pct / risk_reward ratio
    # Use risk_reward stored at signal creation for the denominator
    r_multiples = []
    for s in wins + losses:
        pct = s.get("result_pct") or 0.0
        rr  = s.get("risk_reward") or 1.5
        r   = pct / (100 / (rr + 1)) if rr > 0 else pct / 1.0
        r_multiples.append(r if s.get("result") == "win" else -abs(r))

    avg_r    = statistics.mean(r_multiples) if r_multiples else 0.0
    median_r = statistics.median(r_multiples) if r_multiples else 0.0

    # Profit factor: gross wins / gross losses
    gross_wins   = sum(abs(r) for r in r_multiples if r > 0)
    gross_losses = sum(abs(r) for r in r_multiples if r < 0)
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else (gross_wins if gross_wins > 0 else 0)

    # Drawdown: maximum peak-to-trough in equity curve (simplified)
    equity = [0.0]
    for r in r_multiples:
        equity.append(equity[-1] + r)
    peak = equity[0]
    max_drawdown = 0.0
    for val in equity[1:]:
        if val > peak:
            peak = val
        dd = peak - val
        if dd > max_drawdown:
            max_drawdown = dd

    # Expectancy per trade in R
    expectancy = win_rate * avg_r + (1 - win_rate) * (-abs(avg_r)) if avg_r > 0 else avg_r

    return {
        "total_signals":   len(signals),
        "win_count":       len(wins),
        "loss_count":      len(losses),
        "expired_count":   len([s for s in signals if s.get("result") == "expired"]),
        "win_rate":        round(win_rate, 4),
        "avg_r":           round(avg_r, 3),
        "median_r":        round(median_r, 3),
        "profit_factor":   round(profit_factor, 3),
        "max_drawdown":    round(max_drawdown, 3),
        "expectancy":      round(expectancy, 4),
    }


def _group_by(signals: list[dict], field: str) -> dict[str, list[dict]]:
    """Group signal list by a field value."""
    groups: dict[str, list] = {}
    for s in signals:
        key = str(s.get(field) or "unknown")
        groups.setdefault(key, []).append(s)
    return groups


# ── Report generators ─────────────────────────────────────────────────────────

def generate_report(sb, days: int = 30) -> dict:
    """
    Generate a full performance report from Supabase closed signals.

    Args:
        sb:   Supabase client
        days: lookback window for the report

    Returns a comprehensive report dict.
    """
    logger.info(f"[analytics] Generating {days}-day performance report")

    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    # ── Fetch closed signals ───────────────────────────────────────────────────
    try:
        signals = (
            sb.table("signals")
            .select(
                "id, ticker, direction, strategy_type, setup_type, result, "
                "result_pct, result_pnl, risk_reward, entry_price, "
                "regime_type, session_mode, confidence_score, confidence_grade, "
                "hit_target, closed_reason, created_at, closed_at, "
                "setup_age_minutes, mae, mfe"
            )
            .eq("status", "closed")
            .gte("created_at", since)
            .neq("result", None)
            .execute()
            .data
        ) or []
    except Exception as e:
        logger.error(f"[analytics] Failed to fetch signals: {e}")
        signals = []

    if not signals:
        return {"error": "No closed signals in the period", "days": days, "total": 0}

    # ── Overall metrics ────────────────────────────────────────────────────────
    overall = _compute_metrics(signals)

    # ── By strategy ───────────────────────────────────────────────────────────
    by_strategy = {
        k: _compute_metrics(v)
        for k, v in _group_by(signals, "strategy_type").items()
    }

    # ── By setup type ─────────────────────────────────────────────────────────
    by_setup_type = {
        k: _compute_metrics(v)
        for k, v in _group_by(signals, "setup_type").items()
        if k and k != "unknown"
    }

    # ── By regime ─────────────────────────────────────────────────────────────
    by_regime = {
        k: _compute_metrics(v)
        for k, v in _group_by(signals, "regime_type").items()
        if k and k != "unknown"
    }

    # ── By session ────────────────────────────────────────────────────────────
    by_session = {
        k: _compute_metrics(v)
        for k, v in _group_by(signals, "session_mode").items()
        if k and k != "unknown"
    }

    # ── By ticker (top 10 / worst 10) ─────────────────────────────────────────
    by_ticker_raw = _group_by(signals, "ticker")
    by_ticker = {
        k: _compute_metrics(v)
        for k, v in by_ticker_raw.items()
        if len(v) >= 3  # minimum 3 signals for statistical meaning
    }

    # Rank tickers
    sorted_tickers = sorted(
        [(t, m) for t, m in by_ticker.items() if m.get("win_rate") is not None],
        key=lambda x: x[1].get("expectancy") or 0,
        reverse=True,
    )
    best_tickers  = sorted_tickers[:5]
    worst_tickers = sorted_tickers[-5:] if len(sorted_tickers) > 5 else []

    # ── False positive analysis ────────────────────────────────────────────────
    false_positives = _analyze_false_positives(signals)

    # ── Setup type predictiveness ──────────────────────────────────────────────
    feature_analysis = _analyze_predictive_features(signals)

    # ── Watchlist conversion ───────────────────────────────────────────────────
    lifecycle_stats = _watchlist_conversion_stats(sb, since)

    # ── Time-to-outcome ────────────────────────────────────────────────────────
    time_analysis = _time_analysis(signals)

    report = {
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "period_days":   days,
        "overall":       overall,
        "by_strategy":   by_strategy,
        "by_setup_type": by_setup_type,
        "by_regime":     by_regime,
        "by_session":    by_session,
        "best_tickers":  [{"ticker": t, **m} for t, m in best_tickers],
        "worst_tickers": [{"ticker": t, **m} for t, m in worst_tickers],
        "false_positive_analysis": false_positives,
        "predictive_features":     feature_analysis,
        "lifecycle_stats":         lifecycle_stats,
        "time_analysis":           time_analysis,
        "quality_flags":           _quality_flags(overall, by_strategy),
    }

    # ── Cache the report ───────────────────────────────────────────────────────
    _cache_report(sb, report, days)

    logger.info(
        f"[analytics] Report complete — {overall['total_signals']} signals, "
        f"win_rate={overall.get('win_rate', 0)*100:.1f}% "
        f"avg_R={overall.get('avg_r', 0):.3f} "
        f"expectancy={overall.get('expectancy', 0):.4f}"
    )

    return report


def _analyze_false_positives(signals: list[dict]) -> dict:
    """
    Identify patterns in losing signals (false positives).
    A false positive is a signal that lost without hitting a meaningful target first.
    """
    losses = [s for s in signals if s.get("result") == "loss"]
    if not losses:
        return {"count": 0, "common_regimes": [], "common_setups": [], "common_sessions": []}

    regime_dist  = _group_by(losses, "regime_type")
    setup_dist   = _group_by(losses, "setup_type")
    session_dist = _group_by(losses, "session_mode")

    return {
        "count":           len(losses),
        "loss_rate":       round(len(losses) / len(signals), 4) if signals else 0,
        "common_regimes":  [
            {"regime": k, "loss_count": len(v)}
            for k, v in sorted(regime_dist.items(), key=lambda x: len(x[1]), reverse=True)
            if k and k != "unknown"
        ][:5],
        "common_setups":   [
            {"setup": k, "loss_count": len(v)}
            for k, v in sorted(setup_dist.items(), key=lambda x: len(x[1]), reverse=True)
            if k and k != "unknown"
        ][:5],
        "common_sessions": [
            {"session": k, "loss_count": len(v)}
            for k, v in sorted(session_dist.items(), key=lambda x: len(x[1]), reverse=True)
            if k and k != "unknown"
        ][:3],
        "avg_loss_pct": round(
            statistics.mean([abs(s.get("result_pct") or 0) for s in losses]), 3
        ) if losses else 0,
    }


def _analyze_predictive_features(signals: list[dict]) -> dict:
    """
    Identify which score layers correlate with winning vs losing signals.
    Rough correlation between L1–L9 scores and outcome.
    """
    wins   = [s for s in signals if s.get("result") == "win"]
    losses = [s for s in signals if s.get("result") == "loss"]

    if not wins or not losses:
        return {}

    layer_names = {
        "l1_smc":          "SMC Structure",
        "l2_technical":    "Technical",
        "l3_sentiment":    "Sentiment",
        "l4_risk":         "Risk",
        "l5_mtf":          "Multi-Timeframe",
        "l6_regime":       "Market Regime",
        "l7_session":      "Session Quality",
        "l8_gamma":        "Gamma Exposure",
        "l9_manipulation": "Manipulation Check",
    }

    feature_analysis = {}
    for layer_key, layer_name in layer_names.items():
        win_vals  = [s.get("score_breakdown", {}).get(layer_key, 0) for s in wins if s.get("score_breakdown")]
        loss_vals = [s.get("score_breakdown", {}).get(layer_key, 0) for s in losses if s.get("score_breakdown")]

        if not win_vals or not loss_vals:
            continue

        win_avg  = statistics.mean(win_vals)
        loss_avg = statistics.mean(loss_vals)
        diff     = win_avg - loss_avg

        feature_analysis[layer_key] = {
            "name":             layer_name,
            "win_avg_score":    round(win_avg, 1),
            "loss_avg_score":   round(loss_avg, 1),
            "discriminability": round(diff, 2),  # positive = this layer predicts wins
        }

    # Sort by discriminability (most predictive first)
    ranked = sorted(feature_analysis.items(), key=lambda x: x[1]["discriminability"], reverse=True)

    return {
        "most_predictive":  [v["name"] for _, v in ranked[:3]],
        "least_predictive": [v["name"] for _, v in ranked[-3:]],
        "layer_details":    {k: v for k, v in ranked},
    }


def _watchlist_conversion_stats(sb, since: str) -> dict:
    """Compute WATCHLIST → DEVELOPING → CONFIRMED conversion funnel."""
    try:
        watchlist = (
            sb.table("setup_watchlist")
            .select("setup_state, promoted_to_signal_id, created_at")
            .gte("created_at", since)
            .execute()
            .data
        ) or []

        total_created  = len(watchlist)
        developed      = len([w for w in watchlist if w.get("setup_state") in
                               ("DEVELOPING", "CONFIRMED_SIGNAL", "EXPIRED", "INVALIDATED")])
        confirmed      = len([w for w in watchlist if w.get("promoted_to_signal_id")])
        expired        = len([w for w in watchlist if w.get("setup_state") == "EXPIRED"])
        invalidated    = len([w for w in watchlist if w.get("setup_state") == "INVALIDATED"])

        return {
            "total_watchlist":    total_created,
            "progressed":         developed,
            "confirmed_signals":  confirmed,
            "expired":            expired,
            "invalidated":        invalidated,
            "watchlist_to_confirmed_rate": round(confirmed / total_created, 4) if total_created > 0 else 0,
            "watchlist_to_expired_rate":   round(expired / total_created, 4) if total_created > 0 else 0,
        }
    except Exception as e:
        logger.debug(f"[analytics] Watchlist stats failed: {e}")
        return {}


def _time_analysis(signals: list[dict]) -> dict:
    """Analyze time-to-outcome patterns."""
    wins   = [s for s in signals if s.get("result") == "win"]
    losses = [s for s in signals if s.get("result") == "loss"]

    def avg_hold_mins(sigs):
        times = []
        for s in sigs:
            try:
                c = datetime.fromisoformat(s["created_at"].replace("Z", "+00:00"))
                e = datetime.fromisoformat(s["closed_at"].replace("Z", "+00:00"))
                times.append((e - c).total_seconds() / 60)
            except Exception:
                pass
        return round(statistics.mean(times), 0) if times else None

    def avg_setup_age(sigs):
        ages = [s.get("setup_age_minutes") or 0 for s in sigs if s.get("setup_age_minutes")]
        return round(statistics.mean(ages), 0) if ages else 0

    # MAE/MFE if available
    maes = [s.get("mae") for s in signals if s.get("mae") is not None]
    mfes = [s.get("mfe") for s in signals if s.get("mfe") is not None]

    return {
        "avg_win_hold_mins":    avg_hold_mins(wins),
        "avg_loss_hold_mins":   avg_hold_mins(losses),
        "avg_setup_age_mins":   avg_setup_age(signals),
        "avg_mae":              round(statistics.mean(maes), 3) if maes else None,
        "avg_mfe":              round(statistics.mean(mfes), 3) if mfes else None,
        "quick_wins":           len([s for s in wins  if _hold_mins(s) and _hold_mins(s) < 60]),
        "slow_losses":          len([s for s in losses if _hold_mins(s) and _hold_mins(s) > 180]),
    }


def _hold_mins(signal: dict) -> Optional[float]:
    try:
        c = datetime.fromisoformat(signal["created_at"].replace("Z", "+00:00"))
        e = datetime.fromisoformat(signal["closed_at"].replace("Z", "+00:00"))
        return (e - c).total_seconds() / 60
    except Exception:
        return None


def _quality_flags(overall: dict, by_strategy: dict) -> list[str]:
    """Generate actionable quality flags for the engine operators."""
    flags: list[str] = []

    win_rate = overall.get("win_rate") or 0
    if win_rate < 0.40:
        flags.append(f"CRITICAL: Win rate {win_rate*100:.1f}% is below 40% — review signal quality gates")
    elif win_rate < 0.50:
        flags.append(f"WARNING: Win rate {win_rate*100:.1f}% below 50% — review SMC filters")

    avg_r = overall.get("avg_r") or 0
    if avg_r < 0:
        flags.append("CRITICAL: Negative average R — signals are losing more than winning even on wins")

    pf = overall.get("profit_factor") or 0
    if pf < 1.0:
        flags.append(f"WARNING: Profit factor {pf:.2f} < 1.0 — system is net negative")
    elif pf > 1.5:
        flags.append(f"GOOD: Profit factor {pf:.2f} — system is net positive")

    for strategy, metrics in by_strategy.items():
        wr = metrics.get("win_rate")
        if wr is not None and wr < 0.35 and metrics.get("total_signals", 0) >= 5:
            flags.append(f"ALERT: {strategy} win rate {wr*100:.1f}% — consider raising threshold")

    return flags


def _cache_report(sb, report: dict, days: int) -> None:
    """Store report summary in signal_analytics_cache for fast API access."""
    try:
        sb.table("signal_analytics_cache").insert({
            "report_date":  date.today().isoformat(),
            "report_type":  f"{days}d",
            "total_signals": report.get("overall", {}).get("total_signals", 0),
            "win_count":    report.get("overall", {}).get("win_count", 0),
            "loss_count":   report.get("overall", {}).get("loss_count", 0),
            "win_rate":     report.get("overall", {}).get("win_rate"),
            "avg_r":        report.get("overall", {}).get("avg_r"),
            "profit_factor": report.get("overall", {}).get("profit_factor"),
            "report_data":  report,
        }).execute()
    except Exception as e:
        logger.debug(f"[analytics] Cache write failed: {e}")
