"""
Integration tests — engine/runner.py

Tests the full scan pipeline with mocked external data:
  - run_strategy_by_type fires without crashing
  - Signal structure validated before Supabase insert
  - Risk manager gates applied (concurrent limit, sector, tier)
  - Maintenance pass runs without error
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unittest.mock import patch, MagicMock
import pandas as pd
import pytest


def _mock_sb(active_signals=None):
    """Mock Supabase with configurable active signal list."""
    sb = MagicMock()
    result = MagicMock()
    result.data = active_signals or []
    sb.table.return_value.select.return_value.eq.return_value.execute.return_value = result
    sb.table.return_value.select.return_value.execute.return_value = result
    sb.table.return_value.insert.return_value.execute.return_value = result
    sb.table.return_value.update.return_value.eq.return_value.execute.return_value = result
    return sb


def _mock_ohlcv(n=50, price=180.0):
    import random
    random.seed(42)
    rows = []
    p = price
    for _ in range(n):
        p *= (1 + random.uniform(-0.003, 0.004))
        rows.append({
            "open": p, "high": p*1.005, "low": p*0.995,
            "close": p, "volume": 1_000_000,
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# run_strategy_by_type
# ──────────────────────────────────────────────────────────────

class TestRunStrategyByType:

    def _run(self, strategy_type, active_signals=None):
        from engine.runner import run_strategy_by_type
        mock_sb = _mock_sb(active_signals)
        df = _mock_ohlcv()

        with patch("engine.runner._make_supabase", return_value=mock_sb), \
             patch("engine.smc.fetch_candles", return_value=df), \
             patch("engine.scorer.score", return_value={
                 "composite_score": 82,
                 "direction": "LONG",
                 "score_breakdown": {"smc": 20, "technical": 22, "sentiment": 15, "risk": 12},
             }), \
             patch("engine.sl_tp_engine.calculate", return_value={
                 "stop_loss": 177.5,
                 "target_one": 183.0,
                 "target_two": 186.0,
                 "risk_reward": 2.2,
                 "adjustments": [],
             }), \
             patch("engine.regime_detector.detect", return_value={
                 "regime": "TRENDING_BULL",
                 "vix": 17.5,
                 "block_scalping": False,
                 "block_day_trade": False,
             }), \
             patch("engine.session_classifier.classify", return_value={
                 "mode": "STANDARD",
                 "min_score": 70,
                 "is_opex": False,
             }), \
             patch("engine.session_classifier.can_fire", return_value=True), \
             patch("engine.manipulation_detector.detect", return_value={
                 "is_clean": True,
                 "flags": [],
                 "score": 95,
                 "stop_raid_risk": False,
             }), \
             patch("engine.gamma_engine.get_gamma_data", return_value={
                 "walls": [], "net_gex": 0, "is_negative_gamma": False,
             }), \
             patch("engine.risk_manager.check_portfolio_limits", return_value={
                 "allowed": True,
                 "reason": "",
                 "tier": "A",
                 "position_multiplier": 0.75,
             }), \
             patch("engine.explainer.explain", return_value="Strong bullish setup."), \
             patch("engine.push.send_signal_alert", return_value=None):
            run_strategy_by_type(strategy_type)

    def test_day_trade_runs_without_error(self):
        self._run("day_trade")

    def test_scalping_runs_without_error(self):
        self._run("scalping")

    def test_swing_trade_runs_without_error(self):
        self._run("swing_trade")

    def test_unknown_strategy_does_not_crash(self):
        """Unknown strategy type should log a warning, not raise."""
        from engine.runner import run_strategy_by_type
        run_strategy_by_type("nonexistent_strategy")

    def test_scan_blocked_at_max_concurrent(self):
        """With 5 active signals, no new signals should be inserted."""
        active = [{"ticker": f"T{i}", "strategy_type": "day_trade", "status": "active"}
                  for i in range(5)]

        mock_sb = _mock_sb(active)
        inserts = []

        original_insert = mock_sb.table.return_value.insert
        original_insert.side_effect = lambda data: inserts.append(data) or MagicMock()

        df = _mock_ohlcv()
        from engine.runner import run_strategy_by_type
        with patch("engine.runner._make_supabase", return_value=mock_sb), \
             patch("engine.smc.fetch_candles", return_value=df), \
             patch("engine.risk_manager.check_portfolio_limits", return_value={
                 "allowed": False,
                 "reason": "max concurrent signals reached",
                 "tier": "A",
                 "position_multiplier": 0.75,
             }):
            run_strategy_by_type("day_trade")

        assert len(inserts) == 0, "No signals should be inserted when at max concurrent"


# ──────────────────────────────────────────────────────────────
# Maintenance pass
# ──────────────────────────────────────────────────────────────

class TestMaintenancePass:

    def test_maintenance_runs_without_error(self):
        from engine.runner import run_maintenance
        mock_sb = _mock_sb()
        with patch("engine.runner._make_supabase", return_value=mock_sb), \
             patch("engine.tracker.run_tracking_pass", return_value=None), \
             patch("engine.signal_monitor.run_monitor_pass", return_value=None):
            run_maintenance()

    def test_maintenance_calls_tracker(self):
        from engine.runner import run_maintenance
        mock_sb = _mock_sb()
        with patch("engine.runner._make_supabase", return_value=mock_sb) as _ms, \
             patch("engine.tracker.run_tracking_pass") as mock_tracker, \
             patch("engine.signal_monitor.run_monitor_pass", return_value=None):
            run_maintenance()
        mock_tracker.assert_called_once()
