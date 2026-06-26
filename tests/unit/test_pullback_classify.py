"""
_classify_setup pullback branch — surfaces a healthy buy-the-dip pullback.

The old gate required rsi<50, which only caught DEEP pullbacks and missed the
good shallow ones (HOOD held its rising 20-EMA at ~93 with RSI 55 → was tagged
'none'/avoid). New gate: RSI 40–62 AND trend_score>=50 (uptrend), so a bear-flag
at a FALLING MA (MSFT) is NOT mislabeled a buyable pullback.
"""
from engine.quant_score_service import _classify_setup


def test_healthy_shallow_pullback_in_uptrend_is_pullback():
    # price at the 20-MA, RSI 55, uptrend (trend_score 70) — the HOOD case.
    st, _ = _classify_setup(93.47, 93.67, 55.0, 1.0, 30, 39, None, 0.0, 70.0)
    assert st == "pullback"


def test_bear_flag_at_falling_ma_is_not_pullback():
    # same price/RSI but trend_score 0 (downtrend) → must NOT be a pullback.
    st, _ = _classify_setup(93.47, 93.67, 55.0, 1.0, 30, 39, None, 0.0, 0.0)
    assert st != "pullback"


def test_deep_pullback_still_classified():
    st, _ = _classify_setup(93.0, 93.5, 45.0, 1.0, 30, 39, None, 0.0, 60.0)
    assert st == "pullback"


def test_overbought_at_ma_not_pullback():
    # RSI 70 at the MA is not a pullback (out of band).
    st, _ = _classify_setup(100.2, 100.0, 70.0, 0.9, 30, 10, None, 0.0, 80.0)
    assert st != "pullback"


def test_momentum_still_wins_over_pullback():
    # strong momentum (rsi>=55, price>ma20, relvol>=1.1) takes priority.
    st, _ = _classify_setup(102.0, 100.0, 60.0, 1.5, 40, 10, None, 0.0, 80.0)
    assert st == "momentum"
