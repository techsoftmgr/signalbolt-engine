"""Unit tests for Sector Leaders — RS blend, ranking, momentum, tape character."""
import numpy as np
import pandas as pd

from engine.sector_leaders import compute, config as C


def _df(closes):
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="B")
    c = list(closes)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": [1e6] * len(c)}, index=idx)


def test_relative_strength():
    n = C.L_1M
    sector = _df(np.linspace(100, 110, n + 1))   # +10%
    spy = _df(np.linspace(100, 105, n + 1))       # +5%
    rs = compute.relative_strength(sector, spy, n)
    assert 4.0 < rs < 6.0                         # ~ +5% relative


def test_rank_map():
    r = compute.rank_map({"XLK": 5.0, "XLF": 3.0, "XLP": 1.0})
    assert r["XLK"] == 1 and r["XLF"] == 2 and r["XLP"] == 3


def test_tape_character():
    assert compute.tape_character(["XLK", "XLY", "XLP"]) == C.OFFENSE_LED    # 2 offense
    assert compute.tape_character(["XLP", "XLU", "XLK"]) == C.DEFENSE_LED    # 2 defense
    assert compute.tape_character(["XLK", "XLP", "XLE"]) == C.ROTATING       # 1 off / 1 def / 1 cyc


def test_compute_ranks_strong_above_weak():
    n = C.L_6M + C.RANK_MOM_LOOKBACK + 12
    bars = {
        "SPY": _df(np.linspace(100, 105, n)),
        "XLK": _df(np.linspace(100, 135, n)),     # strong outperformer
        "XLU": _df(np.linspace(100, 97, n)),      # underperformer
        "XLF": _df(np.linspace(100, 108, n)),
    }
    rows, summary = compute.compute(bars)
    ranks = {r["sector_etf"]: r["rs_rank"] for r in rows}
    assert ranks["XLK"] == 1
    assert ranks["XLK"] < ranks["XLF"] < ranks["XLU"]
    # every row carries a valid momentum label + tilt
    assert all(r["rank_momentum"] in ("IMPROVING", "DETERIORATING", "FLAT") for r in rows)
    assert summary["tape_character"] in (C.OFFENSE_LED, C.DEFENSE_LED, C.ROTATING)
    assert summary["top3"][0] == "XLK"
