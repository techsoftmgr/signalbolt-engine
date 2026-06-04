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
         patch("engine.runner._has_active_option_signal", return_value=False), \
         patch("engine.runner._write_option_signal", return_value="opt-id"), \
         patch("engine.options_scanner.scan_leap", return_value=None), \
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


# ── Deep-ITM LEAP companion ──────────────────────────────────────────────────
from engine import options_scanner as _opt


class TestPickLeap:
    """Pure selection logic for the deep-ITM LEAP: delta band + liquidity gates."""
    @staticmethod
    def _c(delta, oi=500, spread=0.05):
        return {"delta": delta, "oi": oi, "spread_pct": spread, "strike": 100.0}

    def test_picks_delta_closest_to_target(self):
        best = _opt._pick_leap([self._c(0.70), self._c(0.82), self._c(0.95)])
        assert best["delta"] == 0.82            # 0.80 target → 0.82 wins

    def test_rejects_below_deep_itm_band(self):
        assert _opt._pick_leap([self._c(0.50)]) is None     # < 0.65 floor

    def test_rejects_thin_open_interest(self):
        assert _opt._pick_leap([self._c(0.80, oi=10)]) is None   # < 100 OI

    def test_rejects_wide_spread(self):
        assert _opt._pick_leap([self._c(0.80, spread=0.30)]) is None  # > 12%

    def test_missing_spread_is_allowed(self):
        assert _opt._pick_leap([self._c(0.80, spread=None)]) is not None


_LEAP = {
    "ticker": "MSFT", "direction": "LONG", "contract_type": "CALL",
    "strike_price": 380.0, "expiry_date": "2027-01-15", "dte": 560,
    "underlying_price": 420.0, "entry_premium": 95.0, "target_premium": 140.0,
    "stop_premium": 47.5, "delta": 0.82, "theta": -0.02, "iv": 28.0,
    "open_interest": 1200, "volume": 50, "breakeven": 475.0,
    "max_loss": 9500.0, "max_gain": 4500.0,
}


class TestLeapCompanion:
    def test_writes_manual_leap_when_contract_found(self):
        captured = {"opts": []}
        def _wo(_sb, r):
            captured["opts"].append(r)
            return "opt-id"
        with patch("engine.drawdown_regime.assess", return_value=_OPEN), \
             patch("engine.fundamentals.get_ranked", return_value=_QUAL), \
             patch("engine.runner._has_active_signal", return_value=False), \
             patch("engine.runner._write_signal", return_value="sig-id"), \
             patch("engine.runner._has_active_option_signal", return_value=False), \
             patch("engine.runner._write_option_signal", side_effect=_wo), \
             patch("engine.options_scanner.scan_leap", return_value=dict(_LEAP)), \
             patch("engine.alpaca_client.get_bars", return_value=_df(420.0)), \
             patch("engine.turnaround_detector.score_turnaround", return_value={"stage": "buyzone", "score": 80}), \
             patch("engine.push._send_raw"), patch("engine.push._record_alert"):
            res = dv.generate(sb=object())
        assert res["leaps"] == ["MSFT"]
        assert len(captured["opts"]) == 1
        opt = captured["opts"][0]
        assert opt["strategy_type"] == "deep_value"
        assert opt["management_mode"] == "manual"     # engine hands-off (no 24h expiry)
        assert opt["contract_type"] == "CALL"
        assert opt["status"] == "active"
        assert "LEAP" in opt["ai_explanation"]

    def test_stock_fires_without_leap_when_none(self):
        # scan_leap → None: the stock signal still fires, no option written.
        res, rows = _run(_OPEN, _QUAL, _df(420.0), {"stage": "buyzone", "score": 80})
        assert res["count"] == 1 and res["leaps"] == []
