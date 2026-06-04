"""
Unit tests — engine/deep_value_signal.py (crash/deep-value combine, backlog #10).

Verifies the gating: window-closed short-circuit, the deep-discount threshold, and
— the key one — the FALLING-KNIFE GATE (only fire when turnaround stage=='buyzone').
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unittest.mock import patch
import pandas as pd
import pytest

from engine import deep_value_signal as dv


def _df(last, high=600.0, n=220):
    closes = [high * 0.7] * (n - 1) + [last]
    return pd.DataFrame({"high": [high] * n, "low": [c * 0.9 for c in closes], "close": closes})


class TestStockDrawdown:
    def test_off_high(self):
        dd, hi = dv._stock_drawdown_and_high(_df(420.0))   # 420/600 - 1 = -30%
        assert dd == pytest.approx(-30.0, abs=0.5) and hi == 600.0

    def test_short_df_none(self):
        dd, hi = dv._stock_drawdown_and_high(pd.DataFrame({"high": [1, 2], "low": [1, 1], "close": [1, 2]}))
        assert dd is None and hi is None


def _run(regime, candidates, df, ta):
    captured = {"rows": []}
    def _write(_sb, row):
        captured["rows"].append(row)
        return "sig-id"
    with patch("engine.drawdown_regime.assess", return_value=regime), \
         patch("engine.fundamentals.get_ranked", return_value=candidates), \
         patch("engine.runner._has_active_signal", return_value=False), \
         patch("engine.runner._write_signal", side_effect=_write), \
         patch("engine.alpaca_client.get_bars", return_value=df), \
         patch("engine.turnaround_detector.score_turnaround", return_value=ta), \
         patch("engine.push._send_raw"), \
         patch("engine.push._record_alert"):
        res = dv.generate(sb=object())
    return res, captured["rows"]


_OPEN  = {"accumulation_window": True,  "regime": "bear",      "off_high_pct": -23}
_DEEP  = {"accumulation_window": True,  "regime": "deep_bear", "off_high_pct": -32, "deep": True}
_CLOSED = {"accumulation_window": False, "regime": "healthy", "off_high_pct": -1}
_QUAL  = [{"ticker": "MSFT", "quality_score": 5}]
_MANY  = [{"ticker": f"T{i:03d}", "quality_score": 5} for i in range(40)]


class TestGenerateGating:
    def test_window_closed_no_fire(self):
        res, rows = _run(_CLOSED, _QUAL, _df(420.0), {"stage": "buyzone"})
        assert res["count"] == 0 and res["reason"] == "accumulation_window_closed"
        assert rows == []

    def test_fires_when_quality_deep_and_turning(self):
        res, rows = _run(_OPEN, _QUAL, _df(420.0), {"stage": "buyzone", "score": 80})
        assert res["count"] == 1 and "MSFT" in res["fired"]
        row = rows[0]
        assert row["strategy_type"] == "deep_value"
        assert row["direction"] == "LONG"
        assert row["management_mode"] == "manual"     # engine hands-off
        assert row["score_breakdown"]["detector_source"] == "DEEP_VALUE"
        # Ordinary accumulation window → HALF size (scale in), not deep-bear.
        assert row["position_multiplier"] == 0.5
        assert row["score_breakdown"]["deep_regime"] is False

    def test_falling_knife_gate_blocks_non_buyzone(self):
        # Deeply discounted quality name, but NOT a confirmed turn (still 'watch') → skip
        res, rows = _run(_OPEN, _QUAL, _df(420.0), {"stage": "watch", "score": 50})
        assert res["count"] == 0 and rows == []

    def test_not_deep_enough_skipped(self):
        # Only -10% off high (> -25 threshold) → skip even with a buyzone turn
        res, rows = _run(_OPEN, _QUAL, _df(540.0), {"stage": "buyzone"})
        assert res["count"] == 0 and rows == []

    def test_no_quality_candidates(self):
        res, rows = _run(_OPEN, [], _df(420.0), {"stage": "buyzone"})
        assert res["count"] == 0 and rows == []


class TestDeepBearEscalation:
    def test_deep_regime_sizes_up_and_bumps_confidence(self):
        res, rows = _run(_DEEP, _QUAL, _df(420.0), {"stage": "buyzone", "score": 80})
        assert res["count"] == 1
        row = rows[0]
        assert row["position_multiplier"] == 1.0          # full size in a deep-bear washout
        assert row["score_breakdown"]["deep_regime"] is True
        assert row["confidence_score"] == 90              # min(90, 70 + 5*3 + 5)
        assert any("DEEP-BEAR" in f for f in row["confidence_factors"])
        assert "DEEP-BEAR" in row["ai_explanation"]

    def test_deep_regime_uses_bigger_basket(self):
        # 40 deeply-discounted quality names; deep cap is 30, normal cap 15.
        res, rows = _run(_DEEP, _MANY, _df(420.0), {"stage": "buyzone", "score": 80})
        assert res["count"] == 30 and len(rows) == 30

    def test_normal_regime_caps_at_fifteen(self):
        res, rows = _run(_OPEN, _MANY, _df(420.0), {"stage": "buyzone", "score": 80})
        assert res["count"] == 15 and len(rows) == 15
