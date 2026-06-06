"""Unit tests — historical_backtest entry predicates (pure, no look-ahead) +
regime proxy. Synthetic bars so the logic is deterministic."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import pandas as pd
from engine import historical_backtest as hb


def _df(rows):
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])


def _flat(n, price=100.0, vol=1000.0):
    return [[price, price + 0.5, price - 0.5, price, vol] for _ in range(n)]


def test_breakout_fires_on_new_high_with_volume():
    rows = _flat(20, 100, 1000)
    rows.append([100, 106, 99, 105, 2000])      # new high + 2x volume
    assert hb._breakout(_df(rows)) == "LONG"


def test_breakout_no_volume_no_fire():
    rows = _flat(20, 100, 1000)
    rows.append([100, 106, 99, 105, 900])        # new high but LOW volume
    assert hb._breakout(_df(rows)) is None


def test_breakdown_fires_on_new_low_with_volume():
    rows = _flat(20, 100, 1000)
    rows.append([100, 101, 94, 95, 2000])        # new low + volume
    assert hb._breakdown(_df(rows)) == "SHORT"


def test_accum_forming_heavy_upvol_below_high():
    rows = _flat(20, 100, 1000)
    rows.append([100, 102, 99, 101.5, 2500])     # up bar, 2.5x vol, below 100-high? high is 100.5
    # prior 20-bar high ~100.5; close 101.5 is ABOVE → not accum. Adjust:
    rows[-1] = [98, 99, 97.5, 98.5, 2500]        # up close 98.5 < hi*0.985 (~99.0), heavy vol
    assert hb._accum_forming(_df(rows)) == "LONG"


def test_distrib_forming_heavy_downvol_above_ma():
    rows = _flat(20, 100, 1000)
    rows.append([101, 101.5, 100.2, 100.5, 2500])  # down bar, heavy vol, above MA(~100)
    assert hb._distrib_forming(_df(rows)) == "SHORT"


def test_predicates_handle_short_window():
    short = _df(_flat(5))
    for p in (hb._breakout, hb._breakdown, hb._accum_forming, hb._distrib_forming, hb._compression):
        assert p(short) is None


def test_spy_regime_labels():
    import numpy as np
    idx = pd.date_range("2020-01-01", periods=260, freq="D")
    # uptrend then crash
    close = list(range(100, 100 + 220)) + [320 - 4 * i for i in range(40)]
    df = pd.DataFrame({"close": close}, index=idx)
    reg = hb._spy_regime(df)
    assert reg  # produced labels
    # the crash tail (price below 200-SMA) should be RISK_OFF
    assert reg[str(idx[-1].date())[:10]] == "RISK_OFF"
    assert set(reg.values()) <= {"RISK_ON", "NEUTRAL", "RISK_OFF"}


def test_regime_gate_mirrors_live_filter():
    # LONG blocked in risk-off; SHORT blocked in risk-on; NEUTRAL allows both
    assert hb._regime_allows("LONG", "RISK_ON") is True
    assert hb._regime_allows("LONG", "RISK_OFF") is False
    assert hb._regime_allows("LONG", "NEUTRAL") is True
    assert hb._regime_allows("SHORT", "RISK_OFF") is True
    assert hb._regime_allows("SHORT", "RISK_ON") is False
    assert hb._regime_allows("SHORT", "NEUTRAL") is True
