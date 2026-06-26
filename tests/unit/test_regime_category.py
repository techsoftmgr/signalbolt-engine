"""
quant_score_service._regime_category — the watchlist "Game Plan" tap-filter
classifier. Buckets a name as rs_pullback (+EV long in a weak tape) / rs_leader
(strong-but-extended, wait) / knife (downtrend underperformer, -EV) / neutral.
"""
import numpy as np
import pandas as pd

from engine.quant_score_service import _regime_category


def _spy(ret20_pct):
    # 60 bars of SPY ending with a 20-day return of ret20_pct.
    base = 100.0
    end = base * (1 + ret20_pct / 100)
    closes = list(np.linspace(base, base, 40)) + list(np.linspace(base, end, 20))
    return pd.DataFrame({"close": closes})


def test_rs_leader_at_pullback():
    # Uptrend, above 50-SMA, rising 20-EMA, outperforming SPY, price ON the 20-MA.
    closes = np.linspace(60, 100, 60)            # steady uptrend → +stock 20d return
    ma20 = float(np.mean(closes[-20:]))
    cat, rs = _regime_category(closes, ma20, ma20, 55.0, _spy(2.0))
    assert cat == "rs_pullback" and rs > 0


def test_rs_leader_extended():
    # Same uptrend leader but price far ABOVE the 20-MA → extended, not a pullback.
    closes = np.linspace(60, 100, 60)
    ma20 = float(np.mean(closes[-20:]))
    cat, _ = _regime_category(closes, ma20 * 1.10, ma20, 60.0, _spy(2.0))
    assert cat == "rs_leader"


def test_knife_downtrend_underperformer():
    closes = np.linspace(100, 60, 60)            # falling → below 50-SMA, EMA not rising
    ma20 = float(np.mean(closes[-20:]))
    cat, rs = _regime_category(closes, float(closes[-1]), ma20, 38.0, _spy(-1.0))
    assert cat == "knife" and rs < 0


def test_neutral_when_no_spy():
    closes = np.linspace(60, 100, 60)
    cat, rs = _regime_category(closes, 100.0, 95.0, 55.0, None)
    assert cat == "neutral" and rs is None


def test_never_raises_on_short_data():
    cat, rs = _regime_category(np.array([1.0, 2.0]), 2.0, 1.5, 50.0, _spy(0.0))
    assert cat == "neutral"
