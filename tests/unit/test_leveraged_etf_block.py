"""
Unit tests — leveraged/inverse ETFs must NEVER fire a signal.
engine/leveraged_etfs.is_leveraged_etf + runner._is_untradeable.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine.leveraged_etfs import is_leveraged_etf
from engine import runner


class TestIsLeveragedEtf:
    def test_blocks_known_leveraged_inverse(self):
        for t in ("TQQQ", "SQQQ", "SOXL", "SOXS", "UVXY", "SVXY", "VXX",
                  "SPXL", "SPXU", "TNA", "TZA", "FAS", "FAZ", "NUGT", "DUST",
                  "TMF", "TMV", "TSLL", "NVDL", "MSTU"):
            assert is_leveraged_etf(t), f"{t} should be blocked"

    def test_allows_plain_1x(self):
        for t in ("SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "SMH", "GLD",
                  "AAPL", "NVDA", "HOOD", "MRVL"):
            assert not is_leveraged_etf(t), f"{t} should be tradeable"

    def test_case_insensitive(self):
        assert is_leveraged_etf("tqqq") and is_leveraged_etf("Sqqq")

    def test_none_safe(self):
        assert not is_leveraged_etf(None) and not is_leveraged_etf("")


class TestUntradeableGate:
    def test_leveraged_etf_is_untradeable(self):
        assert runner._is_untradeable("TQQQ", 80.0) is True
        assert runner._is_untradeable("SQQQ", 38.0) is True
        assert runner._is_untradeable("SOXL", 45.0) is True

    def test_normal_equity_tradeable(self):
        assert runner._is_untradeable("AAPL", 200.0) is False
        assert runner._is_untradeable("SMH", 610.0) is False     # 1x sector ETF — OK
        assert runner._is_untradeable("SPY", 600.0) is False

    def test_still_blocks_penny_and_warrant(self):
        assert runner._is_untradeable("ABCDW", 5.0) is True      # warrant
        assert runner._is_untradeable("AAPL", 0.5) is True       # penny
