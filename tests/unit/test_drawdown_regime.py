"""
Unit tests — engine/drawdown_regime.py (Phase 0 of the crash/deep-value signal).

Verifies the SPY-anchored drawdown classification + accumulation-window trigger.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
import pytest
from engine import drawdown_regime as dd


def _df(high_52w: float, last: float, n: int = 260):
    """Daily-bar df whose 252-window high == high_52w and last close == `last`.
    Every bar's high == high_52w so the peak is always inside tail(252)."""
    closes = [high_52w * 0.96] * (n - 1) + [last]
    return pd.DataFrame({
        "high":  [high_52w] * n,
        "low":   [c * 0.99 for c in closes],
        "close": closes,
    })


def _spy(off_pct: float):
    """An SPY df currently `off_pct`% below its 52-week high."""
    hi = 600.0
    return {"SPY": _df(hi, hi * (1 + off_pct / 100))}


class TestRegimeGrades:
    def test_healthy_near_highs(self):
        r = dd.compute(_spy(-2))
        assert r["regime"] == "healthy"
        assert r["accumulation_window"] is False and r["watch"] is False

    def test_pullback(self):
        assert dd.compute(_spy(-7))["regime"] == "pullback"

    def test_correction_is_watch_not_buy(self):
        r = dd.compute(_spy(-14))
        assert r["regime"] == "correction"
        assert r["watch"] is True
        assert r["accumulation_window"] is False   # buy trigger only at -20%

    def test_bear_opens_accumulation_window(self):
        r = dd.compute(_spy(-23))
        assert r["regime"] == "bear"
        assert r["accumulation_window"] is True
        assert r["deep"] is False

    def test_deep_bear(self):
        r = dd.compute(_spy(-35))
        assert r["regime"] == "deep_bear"
        assert r["accumulation_window"] is True and r["deep"] is True

    def test_off_high_pct_reported(self):
        r = dd.compute(_spy(-23))
        assert r["off_high_pct"] == pytest.approx(-23.0, abs=0.6)


class TestEdgeCases:
    def test_no_data(self):
        r = dd.compute({})
        assert r["regime"] == "unknown" and r["accumulation_window"] is False

    def test_spy_missing_uses_worst_index(self):
        # No SPY; QQQ -8 (pullback), IWM -25 (bear) → worst = IWM → bear
        bars = {"QQQ": _df(500, 460), "IWM": _df(200, 150)}
        r = dd.compute(bars)
        assert r["accumulation_window"] is True   # worst index drives it

    def test_insufficient_bars_skipped(self):
        short = pd.DataFrame({"high": [100, 101], "low": [99, 100], "close": [100, 100]})
        r = dd.compute({"SPY": short})
        assert r["regime"] == "unknown"
