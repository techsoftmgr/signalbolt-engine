"""Unit tests — peak_regime_study pure helpers (the measurement math, not the yfinance fetch)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
from engine import peak_regime_study as prs


def test_stats_short_return_sign_and_winrate():
    # short_ret > 0 = the short made money (price fell); win-rate = % > 0
    s = prs._stats([10.0, -5.0, 2.0, -1.0])
    assert s["n"] == 4
    assert s["short_win_pct"] == 50.0
    assert s["avg_short_ret_pct"] == 1.5
    assert s["best"] == 10.0 and s["worst"] == -5.0
    assert prs._stats([])["n"] == 0


def test_bull_flags_above_rising_50ma():
    # 80 strictly-rising closes → above a rising 50-MA on the recent bars = bull
    idx = pd.date_range("2026-01-01", periods=80, freq="D")
    spy = pd.DataFrame({"close": [100 + i for i in range(80)]}, index=idx)
    flags = prs._bull_flags(spy)
    assert bool(flags.iloc[-1]) is True            # rising tape → bull
    # a flat tape → 50-MA not rising → not bull
    spy_flat = pd.DataFrame({"close": [100.0] * 80}, index=idx)
    assert bool(prs._bull_flags(spy_flat).iloc[-1]) is False
