"""Unit tests — chart_read._fib auto-anchors Fibonacci to the most recent swing,
returning retracement levels + golden-pocket dip/bounce zone + extension target +
a plain-English read."""
import sys, os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from engine.chart_read import _fib


def _df(closes):
    return pd.DataFrame({"high": [c * 1.01 for c in closes],
                         "low":  [c * 0.99 for c in closes],
                         "close": closes})


def test_up_leg_retracement_and_golden_pocket():
    closes = list(np.linspace(100, 150, 45)) + list(np.linspace(150, 135, 15))  # up then pulling back
    f = _fib(_df(closes))
    assert f["direction"] == "up"
    assert f["swingHigh"] > f["swingLow"]
    # golden pocket sits between the 50% and 61.8% retracements, below the high
    gp = f["goldenPocket"]
    assert gp["low"] < gp["high"] < f["swingHigh"]
    # 1.618 extension (the projected target) is ABOVE the swing high for an up-leg
    assert f["target"] > f["swingHigh"]
    assert isinstance(f["explain"], str) and "golden pocket" in f["explain"]


def test_down_leg_direction():
    closes = list(np.linspace(150, 100, 45)) + list(np.linspace(100, 115, 15))  # down then bouncing
    f = _fib(_df(closes))
    assert f["direction"] == "down"
    # extension target projects BELOW the swing low for a down-leg
    assert f["target"] < f["swingLow"]


def test_levels_present_and_ordered():
    f = _fib(_df(list(np.linspace(50, 80, 60))))
    ratios = [l["ratio"] for l in f["levels"]]
    assert ratios == [0.236, 0.382, 0.5, 0.618, 0.786]
    assert all("price" in l and "label" in l for l in f["levels"])


def test_flat_returns_none():
    flat = pd.DataFrame({"high": [100.0] * 60, "low": [100.0] * 60, "close": [100.0] * 60})
    assert _fib(flat) is None   # zero range → no fib


# ── price-aware status (the COIN "buy area $185–193 while at $162" bug) ─────────
def _up_df():
    closes = list(np.linspace(150, 220, 45)) + list(np.linspace(220, 210, 15))
    return _df(closes)


def test_status_none_without_price_backcompat():
    f = _fib(_up_df())
    assert f["status"] is None          # no price passed → unchanged behavior


def test_price_inside_pocket_is_active():
    f0 = _fib(_up_df()); gp = f0["goldenPocket"]
    f = _fib(_up_df(), price=(gp["low"] + gp["high"]) / 2)
    assert f["status"]["position"] == "inside" and f["status"]["failed"] is False


def test_price_below_pocket_is_overhead_not_buy():
    f0 = _fib(_up_df()); gp, inval = f0["goldenPocket"], f0["invalidation"]
    f = _fib(_up_df(), price=(gp["low"] + inval) / 2)   # below pocket, above 78.6%
    assert f["status"]["position"] == "below" and f["status"]["failed"] is False
    assert "overhead" in f["explain"].lower()
    assert "dip-buy zone: a hold" not in f["explain"].lower()   # not framed as a buy


def test_price_under_invalidation_is_failed():
    inval = _fib(_up_df())["invalidation"]
    f = _fib(_up_df(), price=inval - 15)                 # below 78.6% → leg failed
    assert f["status"]["failed"] is True
    assert "failed" in f["explain"].lower() and "overhead resistance" in f["explain"].lower()
