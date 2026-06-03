"""
Unit tests — engine/breakdown_signals.py

Focus: the asset-class / relativeVolume logging is PURE METADATA. It must
never change which signals fire or their entry/stop/targets — especially for
single-name STOCKS (the detector running 80%+ must not be disturbed).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unittest.mock import patch, MagicMock
import pytest

from engine import breakdown_signals as bd


# ── classify_asset (pure) ──────────────────────────────────────────────────
class TestClassifyAsset:
    def test_commodity(self):
        assert bd.classify_asset("GLD")  == {"asset_class": "commodity", "is_etf": True}
        assert bd.classify_asset("slv")["asset_class"] == "commodity"   # case-insensitive
        assert bd.classify_asset("USO")["is_etf"] is True

    def test_bond(self):
        assert bd.classify_asset("TLT")  == {"asset_class": "bond", "is_etf": True}

    def test_broad_etf(self):
        assert bd.classify_asset("SPY")["asset_class"] == "broad_etf"
        assert bd.classify_asset("QQQ")["is_etf"] is True

    def test_sector_etf(self):
        assert bd.classify_asset("XLE")["asset_class"] == "sector_etf"

    def test_equity_default(self):
        # Single names — including ones whose letters resemble an ETF — stay equity.
        for t in ("NVDA", "GILD", "AAPL", "BA", "T", "CMCSA"):
            r = bd.classify_asset(t)
            assert r == {"asset_class": "equity", "is_etf": False}, t

    def test_unknown_and_empty_default_equity(self):
        assert bd.classify_asset("ZZZZ")["asset_class"] == "equity"
        assert bd.classify_asset("")["asset_class"]     == "equity"
        assert bd.classify_asset(None)["asset_class"]   == "equity"


# ── generate(): logging is additive, firing unchanged ──────────────────────
def _quant_row(ticker, price=100.0, rvol=2.0):
    return {
        "ticker": ticker, "price": price, "ma20": 105.0,
        "breakdownLevel": 99.0, "atrPct": 2.0,
        "relativeVolume": rvol, "breakdownScore": 70.0,
    }


def _run_generate(ticker, rvol=2.0):
    """Run generate() with all DB/IO mocked; return the SHORT signal_row that
    would have been written."""
    captured = {}

    def fake_write(_sb, row):
        captured["row"] = row
        return "sig-id"

    with patch("engine.runner._write_signal", side_effect=fake_write), \
         patch("engine.runner._has_active_option_signal", return_value=True), \
         patch("engine.push.send_signal_alert"), \
         patch("engine.options_scanner.scan", return_value=None):
        bd.generate(MagicMock(), _quant_row(ticker, rvol=rvol))
    return captured["row"]


class TestGenerateMetadata:
    def test_stock_breakdown_logs_equity_metadata(self):
        row = _run_generate("NVDA", rvol=2.5)
        sbk = row["score_breakdown"]
        assert sbk["detector_source"] == "BREAKDOWN"
        assert sbk["asset_class"]     == "equity"
        assert sbk["is_etf"]          is False
        assert sbk["relativeVolume"]  == 2.5

    def test_commodity_etf_tagged_but_STILL_fires(self):
        row = _run_generate("GLD", rvol=3.0)
        sbk = row["score_breakdown"]
        assert sbk["asset_class"] == "commodity"
        assert sbk["is_etf"]      is True
        # CRITICAL: we LOG, we don't GATE — the signal still fires with levels.
        assert row["direction"]   == "SHORT"
        assert row["entry_price"] == 100.0
        assert row["stop_loss"]   > row["entry_price"]   # short stop above entry
        assert row["target_one"]  < row["entry_price"]

    def test_levels_identical_regardless_of_asset_class(self):
        """Same quant inputs → identical entry/stop/targets/confidence/direction
        whether the ticker is an equity or an ETF. Proves the metadata can't
        corrupt the detector's actual trade."""
        eq  = _run_generate("NVDA")
        etf = _run_generate("GLD")
        for k in ("entry_price", "stop_loss", "target_one", "target_two",
                  "confidence_score", "direction", "strategy_type"):
            assert eq[k] == etf[k], f"{k} differs — asset class must NOT affect firing"

    def test_missing_relative_volume_is_none_not_crash(self):
        row = _run_generate("NVDA", rvol=None)
        assert row["score_breakdown"]["relativeVolume"] is None
        assert row["direction"] == "SHORT"   # still fires
