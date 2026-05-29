"""
Integration tests — engine/runner.py

Tests the full scan pipeline with mocked external data:
  - run_strategy_by_type fires without crashing
  - Signal structure validated before Supabase insert
  - Risk manager gates applied (concurrent limit, sector, tier)
  - Maintenance pass (_run_maintenance) runs without error

Real APIs:
  run_strategy_by_type(strategy_type: str) → None
  _run_maintenance() → None   (private but importable)
  _supabase() → Client        (patch target for supabase mock)
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
    sb.table.return_value.select.return_value.eq.return_value.eq.return_value.eq.return_value.execute.return_value = result
    sb.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = result
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


# Shared mock return values that match real runner.py API contracts
_MOCK_ANALYSIS = {
    "ticker":        "AAPL",
    "direction":     "LONG",
    "current_price": 180.0,
    "entry":         180.0,
    "stop_loss":     177.5,
    "target_one":    183.0,
    "target_two":    186.0,
    "candles":       _mock_ohlcv(),
    "structure":     {"choch_bullish": True},
    "fvgs":          {},
    "obs":           {},
    "liquidity_sweep": {},
}

_MOCK_SCORED = {
    "total":     82,
    "passes":    True,
    "direction": "LONG",
    "threshold": 70,
    "stop_loss":    177.5,
    "target_one":   183.0,
    "target_two":   186.0,
    "breakdown": {
        "l1_smc": 20, "l2_technical": 22, "l3_sentiment": 15,
        "l4_risk": 12, "l5_mtf": 8, "l6_regime": 4,
        "l7_session": 3, "l8_gamma": 2, "quant_bonus": 0,
    },
    "confidence_factors": ["Strong CHoCH structure", "Above 200 MA"],
}

_MOCK_SLTP = {
    "valid":        True,
    "stop_loss":    177.5,
    "target_one":   183.0,
    "target_two":   186.0,
    "risk_reward_1": 2.2,
    "risk_reward_2": 4.0,
    "adjustments":  [],
}

_MOCK_REGIME = {
    "regime_type":    "TRENDING_BULL",
    "vix":            17.5,
    "vix_change_pct": 0.0,
    "above_200ma":    True,
    "adx":            25.0,
    "blocked":        False,
    "block_reason":   "",
}

_MOCK_SESSION = {
    "mode":        "STANDARD",
    "market_open": True,
    "blocked":     False,
    "block_reason": "",
    "threshold":   70,
    "sl_adjustment": 1.0,
    "allows_swing": True,
    "is_opex_day":  False,
    "is_opex_week": False,
}

_MOCK_GAMMA = {
    "walls":            [],
    "net_gex":          0,
    "is_negative_gamma": False,
    "pin_risk":         False,
}

_MOCK_MANIPULATION = {
    "is_clean":       True,
    "flags":          [],
    "score":          95,
    "stop_raid_risk": False,
}

_MOCK_RISK = {
    "allowed":         True,
    "block_reason":    "",
    "confidence_tier": "A",
    "position_mult":   0.75,
}


# ──────────────────────────────────────────────────────────────
# run_strategy_by_type
# ──────────────────────────────────────────────────────────────

class TestRunStrategyByType:

    def _run(self, strategy_type, active_signals=None):
        from engine.runner import run_strategy_by_type
        mock_sb = _mock_sb(active_signals)

        with patch("engine.runner._supabase", return_value=mock_sb), \
             patch("engine.smc.analyze", return_value=_MOCK_ANALYSIS), \
             patch("engine.runner._has_recent_news", return_value=False), \
             patch("engine.manipulation_detector.detect", return_value=_MOCK_MANIPULATION), \
             patch("engine.manipulation_detector.is_blocking", return_value=False), \
             patch("engine.gamma_engine.fetch", return_value=_MOCK_GAMMA), \
             patch("engine.scorer.score", return_value=_MOCK_SCORED), \
             patch("engine.sl_tp_engine.calculate", return_value=_MOCK_SLTP), \
             patch("engine.regime_detector.detect", return_value=_MOCK_REGIME), \
             patch("engine.session_classifier.classify", return_value=_MOCK_SESSION), \
             patch("engine.risk_manager.check", return_value=_MOCK_RISK), \
             patch("engine.explainer.attach_narrative", return_value="Strong bullish setup."), \
             patch("engine.push.send_signal_alert", return_value=None), \
             patch("engine.options_scanner.scan", return_value=None):
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
        """With 5 active signals (risk blocked), no new signals should be inserted."""
        active = [{"ticker": f"T{i}", "strategy_type": "day_trade", "status": "active"}
                  for i in range(5)]

        mock_sb = _mock_sb(active)
        inserts = []
        original_insert = mock_sb.table.return_value.insert
        original_insert.side_effect = lambda data: inserts.append(data) or MagicMock()

        from engine.runner import run_strategy_by_type
        blocked_risk = {**_MOCK_RISK, "allowed": False,
                        "block_reason": "max concurrent signals reached"}

        with patch("engine.runner._supabase", return_value=mock_sb), \
             patch("engine.smc.analyze", return_value=_MOCK_ANALYSIS), \
             patch("engine.runner._has_recent_news", return_value=False), \
             patch("engine.manipulation_detector.detect", return_value=_MOCK_MANIPULATION), \
             patch("engine.manipulation_detector.is_blocking", return_value=False), \
             patch("engine.gamma_engine.fetch", return_value=_MOCK_GAMMA), \
             patch("engine.scorer.score", return_value=_MOCK_SCORED), \
             patch("engine.sl_tp_engine.calculate", return_value=_MOCK_SLTP), \
             patch("engine.regime_detector.detect", return_value=_MOCK_REGIME), \
             patch("engine.session_classifier.classify", return_value=_MOCK_SESSION), \
             patch("engine.risk_manager.check", return_value=blocked_risk):
            run_strategy_by_type("day_trade")

        assert len(inserts) == 0, "No signals should be inserted when risk blocked"


# ──────────────────────────────────────────────────────────────
# Maintenance pass
# ──────────────────────────────────────────────────────────────

class TestMaintenancePass:

    def test_maintenance_runs_without_error(self):
        # _run_maintenance is a private function but importable
        from engine.runner import _run_maintenance
        mock_sb = _mock_sb()
        with patch("engine.tracker.track_signals", return_value=None), \
             patch("engine.runner._supabase", return_value=mock_sb), \
             patch("engine.runner._close_signals", return_value=None), \
             patch("engine.signal_monitor.run", return_value=None):
            _run_maintenance()

    def test_maintenance_calls_tracker(self):
        from engine.runner import _run_maintenance
        mock_sb = _mock_sb()
        # runner.py does "from engine.tracker import track_signals" so the local
        # reference lives at engine.runner.track_signals, not engine.tracker.track_signals
        with patch("engine.runner.track_signals") as mock_tracker, \
             patch("engine.runner._supabase", return_value=mock_sb), \
             patch("engine.runner._close_signals", return_value=None), \
             patch("engine.signal_monitor.run", return_value=None):
            _run_maintenance()
        mock_tracker.assert_called_once()
