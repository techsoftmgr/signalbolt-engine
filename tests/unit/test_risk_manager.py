"""
Unit tests — engine/risk_manager.py

Real API:
  check(sb: Client, ticker: str, score: int) -> dict
    Returns: allowed, block_reason, confidence_tier, position_mult,
             open_count, portfolio_heat, consecutive_losses, regime_mismatch

  get_confidence_tier(score: int) -> (tier_label, position_multiplier)

Covers:
  - Confidence tier assignment (A+/A/B+/B/C)
  - C tier (score < 60) blocks signal
  - Max concurrent signals (5) blocks new signal
  - Max sector signals (2) blocks same sector
  - Position size multiplier per tier
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unittest.mock import MagicMock
import pytest
from engine.risk_manager import (
    TIERS,
    MAX_CONCURRENT_SIGNALS,
    MAX_SECTOR_SIGNALS,
    MIN_CONFIDENCE_FIRE,
    SECTOR_MAP,
    get_confidence_tier,
    check,
)


# ──────────────────────────────────────────────────────────────
# get_confidence_tier
# ──────────────────────────────────────────────────────────────

class TestConfidenceTiers:

    def test_a_plus_tier(self):
        name, mult = get_confidence_tier(92)
        assert name == "A+"
        assert mult == 1.00

    def test_a_tier(self):
        name, mult = get_confidence_tier(83)
        assert name == "A"
        assert mult == 0.75

    def test_b_plus_tier(self):
        name, mult = get_confidence_tier(72)
        assert name == "B+"
        assert mult == 0.50

    def test_b_tier(self):
        name, mult = get_confidence_tier(62)
        assert name == "B"
        assert mult == 0.25

    def test_c_tier_blocked(self):
        name, mult = get_confidence_tier(55)
        assert name == "C"
        assert mult == 0.00

    def test_boundary_90_is_a_plus(self):
        name, _ = get_confidence_tier(90)
        assert name == "A+"

    def test_boundary_89_is_a(self):
        name, _ = get_confidence_tier(89)
        assert name == "A"

    def test_boundary_60_is_b(self):
        name, _ = get_confidence_tier(60)
        assert name == "B"

    def test_boundary_59_is_c(self):
        name, _ = get_confidence_tier(59)
        assert name == "C"

    def test_all_tiers_defined(self):
        tier_names = [t[0] for t in TIERS]
        for expected in ["A+", "A", "B+", "B", "C"]:
            assert expected in tier_names


# ──────────────────────────────────────────────────────────────
# check(sb, ticker, score)
# ──────────────────────────────────────────────────────────────

class TestPortfolioCheck:

    def _mock_sb(self, active_signals):
        """Mock Supabase returning active_signals from .eq('status','active')."""
        sb = MagicMock()
        result = MagicMock()
        result.data = active_signals
        # check() queries: sb.table("signals").select(...).eq("status","active").execute()
        sb.table.return_value.select.return_value.eq.return_value.execute.return_value = result
        # For the loss count subquery
        sb.table.return_value.select.return_value.eq.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = result
        return sb

    def _active(self, ticker, strategy="day_trade"):
        return {"ticker": ticker, "strategy_type": strategy, "result": "pending", "status": "active"}

    def test_allows_first_signal(self):
        sb = self._mock_sb([])
        result = check(sb, "AAPL", 80)
        assert result["allowed"] is True

    def test_blocks_c_tier_score(self):
        sb = self._mock_sb([])
        result = check(sb, "AAPL", 55)
        assert result["allowed"] is False
        assert "55" in result["block_reason"] or "60" in result["block_reason"]

    def test_blocks_when_max_concurrent_reached(self):
        sigs = [self._active(f"T{i}") for i in range(MAX_CONCURRENT_SIGNALS)]
        sb = self._mock_sb(sigs)
        result = check(sb, "TSLA", 80)
        assert result["allowed"] is False
        assert "concurrent" in result["block_reason"].lower() or str(MAX_CONCURRENT_SIGNALS) in result["block_reason"]

    def test_allows_below_max_concurrent(self):
        sigs = [self._active(f"T{i}") for i in range(MAX_CONCURRENT_SIGNALS - 1)]
        sb = self._mock_sb(sigs)
        result = check(sb, "TSLA", 80)
        assert result["allowed"] is True

    def test_blocks_sector_limit(self):
        """2 tech signals open → 3rd tech signal (NVDA) blocked."""
        sigs = [self._active("AAPL"), self._active("MSFT")]  # both Technology
        sb = self._mock_sb(sigs)
        result = check(sb, "NVDA", 80)  # NVDA = Technology
        assert result["allowed"] is False
        assert "sector" in result["block_reason"].lower()

    def test_allows_different_sector(self):
        """2 tech signals open → energy signal still passes."""
        sigs = [self._active("AAPL"), self._active("MSFT")]
        sb = self._mock_sb(sigs)
        result = check(sb, "XOM", 80)  # XOM = Energy
        assert result["allowed"] is True

    def test_result_has_required_keys(self):
        sb = self._mock_sb([])
        result = check(sb, "AAPL", 80)
        for key in ["allowed", "block_reason", "confidence_tier", "position_mult"]:
            assert key in result, f"Missing key: {key}"

    def test_position_mult_matches_tier(self):
        sb = self._mock_sb([])
        result = check(sb, "AAPL", 92)   # A+ tier
        assert result["position_mult"] == 1.00
        assert result["confidence_tier"] == "A+"


# ──────────────────────────────────────────────────────────────
# Sector map coverage
# ──────────────────────────────────────────────────────────────

class TestSectorMap:

    def test_all_watched_tickers_have_sector(self):
        WATCHED = [
            "AAPL", "MSFT", "NVDA", "GOOGL", "META", "TSLA", "AMD",
            "SPY", "QQQ", "IWM", "DIA", "COIN", "PLTR", "MSTR", "HOOD",
            "RBLX", "UBER", "ABNB", "JPM", "GS", "XOM", "CVX",
            "MARA", "RIOT", "CLSK", "MRNA", "BNTX",
        ]
        missing = [t for t in WATCHED if t not in SECTOR_MAP]
        assert missing == [], f"Tickers missing from SECTOR_MAP: {missing}"

    def test_sector_values_are_strings(self):
        for ticker, sector in SECTOR_MAP.items():
            assert isinstance(sector, str) and len(sector) > 0
