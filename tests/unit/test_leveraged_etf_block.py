"""
Unit tests — two-tier leveraged/inverse ETF firing policy.

  • ALWAYS blocked: single-stock 2x/3x, vol ETNs, commodity/metal/bond/EM leveraged.
  • Leveraged BROAD-INDEX / US-sector equity: allowed short-horizon, blocked on
    months-horizon strategies (deep_value/position_trade).

Covers engine/leveraged_etfs.{is_blocked_leveraged_etf,is_leveraged_index_etf,
should_block_signal} + the runner._is_untradeable firing gate.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine.leveraged_etfs import (
    is_blocked_leveraged_etf,
    is_leveraged_index_etf,
    should_block_signal,
)
from engine import runner


class TestAlwaysBlocked:
    def test_blocks_single_stock_leveraged(self):
        for t in ("TSLL", "NVDL", "MSTU", "CONL", "AMDL", "AAPU", "PLTU"):
            assert is_blocked_leveraged_etf(t), f"{t} (single-stock) should be blocked"

    def test_blocks_vol_etns(self):
        for t in ("UVXY", "SVXY", "VXX", "VIXY", "UVIX", "SVIX", "VIXM"):
            assert is_blocked_leveraged_etf(t), f"{t} (vol ETN) should be blocked"

    def test_blocks_commodity_metal_bond_em(self):
        for t in ("BOIL", "KOLD", "NUGT", "DUST", "JNUG", "AGQ",   # commodity/metal
                  "TMF", "TMV", "TBT",                              # bonds/rates
                  "YINN", "YANG", "EDC", "EDZ"):                    # foreign/EM
            assert is_blocked_leveraged_etf(t), f"{t} should be always-blocked"

    def test_leveraged_index_is_NOT_always_blocked(self):
        for t in ("TQQQ", "SQQQ", "SOXL", "SOXS", "SPXL", "SPXU", "TNA", "TZA",
                  "FAS", "FAZ", "LABU", "LABD", "TECL", "TECS"):
            assert not is_blocked_leveraged_etf(t), f"{t} (lev index) is not always-blocked"

    def test_case_insensitive_and_none_safe(self):
        assert is_blocked_leveraged_etf("tsll") and is_blocked_leveraged_etf("UvXy")
        assert not is_blocked_leveraged_etf(None) and not is_blocked_leveraged_etf("")


class TestLeveragedIndex:
    def test_recognizes_leveraged_index(self):
        for t in ("TQQQ", "SQQQ", "SOXL", "SOXS", "SPXL", "SPXU", "TNA", "TZA",
                  "FAS", "FAZ", "LABU", "LABD", "TECL", "TECS", "UPRO", "ERX"):
            assert is_leveraged_index_etf(t), f"{t} should be a leveraged index/sector ETF"

    def test_single_stock_is_not_index(self):
        for t in ("TSLL", "NVDL", "MSTU", "UVXY", "NUGT", "TMF"):
            assert not is_leveraged_index_etf(t), f"{t} is not a broad-index ETF"

    def test_plain_1x_is_not_index(self):
        for t in ("SPY", "QQQ", "IWM", "XLK", "SMH", "GLD", "AAPL"):
            assert not is_leveraged_index_etf(t)


class TestShouldBlockSignal:
    def test_always_blocked_block_on_every_strategy(self):
        for strat in (None, "day_trade", "swing_trade", "momentum",
                      "deep_value", "position_trade"):
            assert should_block_signal("TSLL", strat) is True
            assert should_block_signal("UVXY", strat) is True
            assert should_block_signal("NUGT", strat) is True

    def test_leveraged_index_allowed_short_horizon(self):
        for strat in (None, "day_trade", "swing_trade", "momentum",
                      "breakout", "breakdown", "cycle"):
            assert should_block_signal("TQQQ", strat) is False, \
                f"TQQQ should fire on short-horizon {strat}"
            assert should_block_signal("SOXL", strat) is False

    def test_leveraged_index_blocked_long_horizon(self):
        for strat in ("deep_value", "position_trade"):
            assert should_block_signal("TQQQ", strat) is True, \
                f"TQQQ must NOT fire on long-horizon {strat}"
            assert should_block_signal("SQQQ", strat) is True
            assert should_block_signal("SOXL", strat) is True

    def test_plain_1x_never_blocked(self):
        for strat in (None, "day_trade", "deep_value", "position_trade"):
            assert should_block_signal("SPY", strat) is False
            assert should_block_signal("AAPL", strat) is False


class TestUntradeableGate:
    def test_always_blocked_untradeable_any_strategy(self):
        assert runner._is_untradeable("TSLL", 20.0) is True
        assert runner._is_untradeable("UVXY", 12.0) is True
        assert runner._is_untradeable("NUGT", 40.0) is True
        assert runner._is_untradeable("TSLL", 20.0, "day_trade") is True

    def test_leveraged_index_tradeable_short_horizon(self):
        assert runner._is_untradeable("TQQQ", 80.0) is False
        assert runner._is_untradeable("SQQQ", 38.0, "day_trade") is False
        assert runner._is_untradeable("SOXL", 45.0, "swing_trade") is False

    def test_leveraged_index_blocked_long_horizon(self):
        assert runner._is_untradeable("TQQQ", 80.0, "deep_value") is True
        assert runner._is_untradeable("SOXL", 45.0, "position_trade") is True

    def test_normal_equity_tradeable(self):
        assert runner._is_untradeable("AAPL", 200.0) is False
        assert runner._is_untradeable("SMH", 610.0) is False     # 1x sector ETF — OK
        assert runner._is_untradeable("SPY", 600.0, "deep_value") is False

    def test_still_blocks_penny_and_warrant(self):
        assert runner._is_untradeable("ABCDW", 5.0) is True      # warrant
        assert runner._is_untradeable("AAPL", 0.5) is True       # penny
