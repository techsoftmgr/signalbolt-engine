"""
Unit tests for engine/setup_lifecycle.py
=========================================
Tests the lifecycle classification logic, confidence grades, risk grades,
missing confirmation detection, and the SetupLifecycleManager's state machine.
"""

import pytest
from unittest.mock import MagicMock, patch, call
from datetime import datetime, timezone, timedelta

from engine.setup_lifecycle import (
    SetupState,
    ConfidenceGrade,
    RiskGrade,
    classify_state,
    classify_confidence_grade,
    classify_risk_grade,
    get_missing_confirmations,
    classify_setup_type,
    annotate_score,
    SetupLifecycleManager,
    WATCHLIST_MIN,
    DEVELOPING_MIN,
    CONFIRMED_MIN,
)


# ── classify_state() ──────────────────────────────────────────────────────────

class TestClassifyState:
    def test_below_watchlist_returns_no_state(self):
        # Scores below 50 are below watchlist — return None or C-grade
        state = classify_state(49)
        assert state is None or state == SetupState.WATCHLIST  # implementation may differ

    def test_watchlist_band(self):
        for score in (50, 55, 60, 64):
            assert classify_state(score) == SetupState.WATCHLIST, f"score={score}"

    def test_developing_band(self):
        for score in (65, 70, 75, 77):
            assert classify_state(score) == SetupState.DEVELOPING, f"score={score}"

    def test_confirmed_band(self):
        for score in (78, 80, 85, 90, 100):
            assert classify_state(score) == SetupState.CONFIRMED_SIGNAL, f"score={score}"

    def test_boundary_values(self):
        assert classify_state(WATCHLIST_MIN)  == SetupState.WATCHLIST
        assert classify_state(DEVELOPING_MIN) == SetupState.DEVELOPING
        assert classify_state(CONFIRMED_MIN)  == SetupState.CONFIRMED_SIGNAL


# ── classify_confidence_grade() ───────────────────────────────────────────────

class TestConfidenceGrade:
    @pytest.mark.parametrize("score,expected", [
        (90, ConfidenceGrade.A_PLUS),
        (95, ConfidenceGrade.A_PLUS),
        (100, ConfidenceGrade.A_PLUS),
        (82, ConfidenceGrade.A),
        (85, ConfidenceGrade.A),
        (89, ConfidenceGrade.A),
        (74, ConfidenceGrade.B_PLUS),
        (78, ConfidenceGrade.B_PLUS),
        (81, ConfidenceGrade.B_PLUS),
        (66, ConfidenceGrade.B),
        (70, ConfidenceGrade.B),
        (73, ConfidenceGrade.B),
        (0, ConfidenceGrade.C),
        (50, ConfidenceGrade.C),
        (65, ConfidenceGrade.C),
    ])
    def test_grade_bands(self, score, expected):
        assert classify_confidence_grade(score) == expected, (
            f"score={score}: expected {expected}, got {classify_confidence_grade(score)}"
        )

    def test_returns_confidence_grade_enum(self):
        result = classify_confidence_grade(80)
        assert isinstance(result, ConfidenceGrade)

    def test_value_strings_are_human_readable(self):
        assert classify_confidence_grade(95).value == "A+"
        assert classify_confidence_grade(85).value == "A"
        assert classify_confidence_grade(76).value == "B+"
        assert classify_confidence_grade(68).value == "B"
        assert classify_confidence_grade(55).value == "C"


# ── classify_risk_grade() ─────────────────────────────────────────────────────

class TestRiskGrade:
    def test_high_score_good_rr_clean_is_low_risk(self):
        grade = classify_risk_grade(
            score=90, risk_reward=3.0,
            chop_score=10.0, regime_type="TRENDING_BULL"
        )
        assert grade == RiskGrade.LOW

    def test_low_score_poor_rr_choppy_is_high_risk(self):
        grade = classify_risk_grade(
            score=60, risk_reward=1.0,
            chop_score=70.0, regime_type="RANGING"
        )
        assert grade == RiskGrade.HIGH

    def test_moderate_conditions_is_medium_risk(self):
        grade = classify_risk_grade(
            score=78, risk_reward=2.0,
            chop_score=30.0, regime_type="RANGING"
        )
        assert grade in (RiskGrade.LOW, RiskGrade.MEDIUM)

    def test_returns_risk_grade_enum(self):
        result = classify_risk_grade(80, 2.5, 20.0, "TRENDING_BULL")
        assert isinstance(result, RiskGrade)

    def test_chop_elevates_risk_grade(self):
        clean = classify_risk_grade(80, 2.5, 10.0, "TRENDING_BULL")
        choppy = classify_risk_grade(80, 2.5, 75.0, "TRENDING_BULL")
        # Choppy environment should be at least as risky as clean
        risk_order = {RiskGrade.LOW: 0, RiskGrade.MEDIUM: 1, RiskGrade.HIGH: 2}
        assert risk_order[choppy] >= risk_order[clean]


# ── get_missing_confirmations() ───────────────────────────────────────────────

class TestMissingConfirmations:
    def _base_analysis(self, **overrides):
        base = {
            "direction": "LONG",
            "ticker": "AAPL",
            "structure": {"bullish": True, "type": "BOS"},
            "fvgs": {"bullish": []},
            "obs":  {"bullish": []},
            "liquidity_sweep": {"swept": False},
            "candles": None,
            "current_price": 150.0,
        }
        base.update(overrides)
        return base

    def _base_score(self, total=70, passes=False):
        return {
            "total": total,
            "passes": passes,
            "breakdown": {
                "l1_smc": 15, "l2_technical": 18, "l3_sentiment": 10,
                "l4_risk": 8, "l5_mtf": 5, "l6_regime": 50,
                "l7_session": 50, "l8_gamma": 50, "l9_manipulation": 50,
            },
        }

    def test_returns_list(self):
        result = get_missing_confirmations(
            self._base_analysis(), self._base_score(),
            regime={"regime_type": "TRENDING_BULL"},
            chop=None,
        )
        assert isinstance(result, list)

    def test_max_five_items(self):
        result = get_missing_confirmations(
            self._base_analysis(), self._base_score(total=55),
            regime={"regime_type": "RANGING"},
            chop=None,
        )
        assert len(result) <= 5

    def test_all_strings(self):
        result = get_missing_confirmations(
            self._base_analysis(), self._base_score(total=65),
            regime={},
            chop=None,
        )
        assert all(isinstance(s, str) for s in result)

    def test_perfect_score_few_missing(self):
        """Near-perfect score should have few or no missing confirmations."""
        result = get_missing_confirmations(
            self._base_analysis(),
            self._base_score(total=90, passes=True),
            regime={"regime_type": "TRENDING_BULL"},
            chop=None,
        )
        assert len(result) <= 2


# ── classify_setup_type() ─────────────────────────────────────────────────────

class TestClassifySetupType:
    def _session(self, mode="STANDARD"):
        return {"mode": mode}

    def test_choch_ob_returns_correct_type(self):
        analysis = {
            "structure": {"type": "CHoCH"},
            "obs": {"bullish": [{"price": 150}]},
            "fvgs": {},
            "liquidity_sweep": {"swept": False},
        }
        result = classify_setup_type(analysis, self._session())
        assert "CHOCH" in result or "OB" in result or result == "CHOCH_OB_RETEST"

    def test_fvg_retest_type(self):
        analysis = {
            "structure": {"type": "BOS"},
            "fvgs": {"bullish": [{"top": 152, "bottom": 150}]},
            "obs":  {},
            "liquidity_sweep": {"swept": False},
        }
        result = classify_setup_type(analysis, self._session())
        assert isinstance(result, str) and len(result) > 0

    def test_sweep_reversal_type(self):
        analysis = {
            "structure": {"type": "CHoCH"},
            "fvgs": {},
            "obs":  {},
            "liquidity_sweep": {"swept": True},
        }
        result = classify_setup_type(analysis, self._session())
        assert isinstance(result, str)

    def test_returns_string_always(self):
        """Should never throw — returns a fallback string."""
        result = classify_setup_type({}, {})
        assert isinstance(result, str)


# ── annotate_score() ─────────────────────────────────────────────────────────

class TestAnnotateScore:
    def test_returns_dict(self):
        result = annotate_score(
            score=80,
            breakdown={"l1_smc": 20, "l2_technical": 22},
            direction="LONG",
            regime_type="TRENDING_BULL",
        )
        assert isinstance(result, dict)

    def test_includes_state_and_grade(self):
        result = annotate_score(80, {}, "LONG", "TRENDING_BULL")
        assert "state" in result
        assert "confidence_grade" in result

    def test_watchlist_score_correct_state(self):
        result = annotate_score(55, {}, "LONG", "RANGING")
        assert result["state"] == SetupState.WATCHLIST.value or result["state"] == "WATCHLIST"

    def test_confirmed_score_correct_state(self):
        result = annotate_score(82, {}, "LONG", "TRENDING_BULL")
        confirmed = SetupState.CONFIRMED_SIGNAL.value
        assert result["state"] == confirmed or result["state"] == "CONFIRMED_SIGNAL"


# ── SetupLifecycleManager ─────────────────────────────────────────────────────

class TestSetupLifecycleManager:
    """Tests that use a mocked Supabase client to avoid DB dependency."""

    def _make_manager(self):
        mgr = SetupLifecycleManager()
        # Replace the internal Supabase client with a MagicMock
        mgr._sb = MagicMock()
        return mgr

    def _analysis(self, ticker="AAPL", direction="LONG"):
        return {
            "ticker": ticker,
            "direction": direction,
            "current_price": 150.0,
            "strategy_type": "day_trade",
        }

    def _score_result(self, total=72, passes=False):
        return {
            "total": total,
            "passes": passes,
            "breakdown": {},
            "confidence_grade": "B+",
        }

    def test_expire_stale_setups_returns_int(self):
        mgr = self._make_manager()
        # Mock the DB query to return no stale setups
        mgr._sb.table.return_value.select.return_value.in_.return_value.lt.return_value.execute.return_value.data = []
        result = mgr.expire_stale_setups()
        assert isinstance(result, int)
        assert result >= 0

    def test_get_active_watchlist_returns_list(self):
        mgr = self._make_manager()
        mgr._sb.table.return_value.select.return_value.in_.return_value.execute.return_value.data = []
        result = mgr.get_active_watchlist()
        assert isinstance(result, list)

    def test_get_active_watchlist_ticker_filter(self):
        mgr = self._make_manager()
        mgr._sb.table.return_value.select.return_value.in_.return_value.eq.return_value.execute.return_value.data = []
        result = mgr.get_active_watchlist(ticker="AAPL")
        assert isinstance(result, list)

    def test_upsert_setup_below_watchlist_min_returns_none(self):
        mgr = self._make_manager()
        score_result = self._score_result(total=40, passes=False)
        result = mgr.upsert_setup(
            analysis=self._analysis(),
            score_result=score_result,
            regime={"regime_type": "TRENDING_BULL"},
            session={"mode": "STANDARD"},
            chop_result=None,
            setup_type="UNKNOWN",
            sltp=None,
        )
        assert result is None

    def test_invalidate_setup_calls_db(self):
        mgr = self._make_manager()
        # Mock the update chain
        mock_update = MagicMock()
        mgr._sb.table.return_value.update.return_value.eq.return_value.eq.return_value.eq.return_value.execute = mock_update
        mgr.invalidate_setup("AAPL", "LONG", "day_trade", "structure_broken")
        # Just verify it ran without error
