"""MFE / MAE per-signal excursion — favorable is positive for LONG and SHORT alike."""
import numpy as np
import pandas as pd

from engine import excursion as ex


def _df(highs, lows, start="2026-05-01"):
    idx = pd.date_range(start, periods=len(highs), freq="D")
    closes = (np.asarray(highs, float) + np.asarray(lows, float)) / 2
    return pd.DataFrame({"open": closes, "high": np.asarray(highs, float),
                         "low": np.asarray(lows, float), "close": closes,
                         "volume": [1e6] * len(highs)}, index=idx)


def test_long_mfe_mae():
    # entry 100; ran to a high of 130 (+30% MFE), dipped to a low of 94 (-6% MAE)
    df = _df(highs=[105, 130, 120], lows=[98, 110, 94])
    mfe, mae = ex.mfe_mae("LONG", 100.0, df)
    assert mfe == 30.0 and mae == -6.0


def test_short_mfe_mae():
    # SHORT entry 100; price fell to a low of 78 (+22% favorable), spiked to 108 (-8% adverse)
    df = _df(highs=[102, 108, 90], lows=[95, 78, 82])
    mfe, mae = ex.mfe_mae("SHORT", 100.0, df)
    assert mfe == 22.0 and mae == -8.0


def test_only_counts_bars_since_entry():
    # a big pre-entry high must NOT inflate MFE — entry is bar index 2 (2026-05-03)
    df = _df(highs=[200, 190, 110, 115], lows=[180, 170, 99, 108])
    mfe, mae = ex.mfe_mae("LONG", 100.0, df, entry_date=df.index[2].date())
    assert mfe == 15.0 and mae == -1.0     # from bars 3-4 only, not the 200 high


def test_merge_is_monotonic():
    # MFE only ratchets up, MAE only down — a shorter later window can't erase extremes
    assert ex.merge(30.0, -6.0, 12.0, -2.0) == (30.0, -6.0)
    assert ex.merge(30.0, -6.0, 35.0, -9.0) == (35.0, -9.0)
    assert ex.merge(None, None, 5.0, -1.0) == (5.0, -1.0)


def test_bad_data_is_none():
    assert ex.mfe_mae("LONG", 0.0, _df([1], [1])) is None
    assert ex.mfe_mae("LONG", 100.0, None) is None
