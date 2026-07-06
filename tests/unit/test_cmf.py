"""
Chaikin Money Flow (CMF) — money-flow indicator in quant_score_service.
Pure OHLCV math; pins the bounds, the state classifier, and edge cases.
"""
import pandas as pd

from engine.quant_score_service import _cmf, _cmf_state


def _df(rows):
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])


def test_all_closes_at_high_is_strong_accumulation():
    # Every bar closes at its high → MFM = +1 → CMF ≈ +1.
    rows = [[10, 11, 9, 11, 1000]] * 25
    cmf, hist = _cmf(_df(rows))
    assert cmf is not None and cmf > 0.99
    assert _cmf_state(cmf) == "accumulation"


def test_all_closes_at_low_is_strong_distribution():
    rows = [[10, 11, 9, 9, 1000]] * 25
    cmf, _ = _cmf(_df(rows))
    assert cmf < -0.99 and _cmf_state(cmf) == "distribution"


def test_midrange_closes_are_neutral():
    rows = [[10, 11, 9, 10, 1000]] * 25   # close at midpoint → MFM 0 → CMF 0
    cmf, _ = _cmf(_df(rows))
    assert abs(cmf) < 1e-6 and _cmf_state(cmf) == "neutral"


def test_history_length_and_latest():
    rows = [[10, 11, 9, 10.5, 1000]] * 60
    cmf, hist = _cmf(_df(rows), period=20, hist=30)
    assert len(hist) == 30 and hist[-1] == cmf


def test_insufficient_data_returns_none():
    cmf, hist = _cmf(_df([[10, 11, 9, 10, 100]] * 5))
    assert cmf is None and hist == []


def test_zero_range_bar_does_not_crash():
    rows = [[10, 10, 10, 10, 1000]] * 12 + [[10, 11, 9, 11, 1000]] * 12
    cmf, _ = _cmf(_df(rows))
    assert cmf is not None            # flat bars contribute 0 flow, no div-by-zero


def test_state_thresholds():
    assert _cmf_state(0.12) == "accumulation"
    assert _cmf_state(0.06) == "mild_accumulation"
    assert _cmf_state(0.0) == "neutral"
    assert _cmf_state(-0.07) == "mild_distribution"
    assert _cmf_state(-0.20) == "distribution"
    assert _cmf_state(None) == "unknown"
