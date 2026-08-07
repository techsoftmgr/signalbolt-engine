"""Breakout SL floor (GLD shake-out fix): post-gamma, breakout-scoped, kill-switched min
stop so a breakout long isn't ejected by a normal pullback / gamma-tightened near-entry stop."""
import pandas as pd
from unittest.mock import patch

from engine import sl_tp_engine as slt

_regime  = {"regime_type": "TRENDING_BULL", "vix": 18.0, "vix_change_pct": 0.0, "blocked": False}
_session = {"mode": "STANDARD", "sl_adjustment": 1.0, "is_opex_day": False, "threshold": 70}
_gamma   = {"walls": [], "net_gex": 0, "is_negative_gamma": False, "pin_risk": False, "available": False}

# near-flat df → tiny ATR → the natural stop floors near the min, well under the breakout floor
_FLAT = pd.DataFrame({"high": [182.05] * 40, "low": [181.95] * 40, "close": [182.0] * 40})


def _calc(strategy):
    with patch("engine.regime_detector.get_sl_adjustment", return_value=1.0):
        return slt.calculate(direction="LONG", entry=182.0, df=_FLAT, regime=_regime,
                             session=_session, gamma=_gamma, strategy_type=strategy)


def test_breakout_floor_off_by_default(monkeypatch):
    monkeypatch.delenv("BREAKOUT_WIDE_STOP_ENABLED", raising=False)
    monkeypatch.setattr(slt, "_compute_adr", lambda *a, **k: 3.0)
    r = _calc("breakout_forming")
    assert not any("breakout SL floored" in a for a in r.get("adjustments", []))


def test_breakout_floor_widens_when_enabled(monkeypatch):
    monkeypatch.setenv("BREAKOUT_WIDE_STOP_ENABLED", "true")
    monkeypatch.setattr(slt, "_compute_adr", lambda *a, **k: 3.0)   # ADR=$3 → floor 1.2×=$3.6
    monkeypatch.setattr(slt, "_BREAKOUT_MIN_ADR_MULT", 1.2)
    bo   = _calc("breakout_forming")
    norm = _calc("day_trade")                                       # non-breakout → untouched
    assert any("breakout SL floored" in a for a in bo.get("adjustments", []))
    assert (182.0 - bo["stop_loss"]) > (182.0 - norm["stop_loss"])  # breakout stop is wider
    assert not any("breakout SL floored" in a for a in norm.get("adjustments", []))
