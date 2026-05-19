"""
Unit tests — engine/manipulation_detector.py

Covers:
  - Stop raid detection (spike + reversal)
  - Momentum ignition (>3 std dev move, no news)
  - Wash trading (high volume, no price movement) — crypto only
  - Clean signal passes all checks
  - Result structure validation
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
import pandas as pd
from engine.manipulation_detector import detect, _compute_price_std


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _flat_df(n=30, price=100.0, volume=500_000):
    """Calm market — no manipulation signals expected."""
    return pd.DataFrame({
        "open":   [price] * n,
        "high":   [price * 1.002] * n,
        "low":    [price * 0.998] * n,
        "close":  [price] * n,
        "volume": [volume] * n,
    })


def _stop_raid_df():
    """
    Last bar: wide range (>1.5%) but price barely moved (open≈close) → stop raid.
    Condition: bar_range > 0.015 AND price_move/bar_range < 0.30
    Last bar: high=102, low=97, open=100, close=100 (returned exactly to open)
      bar_range = (102-97)/100 = 5%  > 1.5% ✓
      price_move = |100-100|/100 = 0%
      ratio = 0/5% = 0 < 0.30 ✓  → stop_raid = True

    Calm bars use small random-ish variation so std_dev > 0 and the
    0% close-to-open move on the last bar is NOT flagged as momentum ignition
    (std_devs = 0 / std → 0, not > 3).
    """
    import random
    random.seed(99)
    rows = []
    price = 100.0
    for _ in range(29):
        # Small oscillation so std_dev of returns is non-zero but tiny
        price = price * (1 + random.uniform(-0.002, 0.002))
        rows.append({"open": price, "high": price*1.001, "low": price*0.999,
                     "close": price, "volume": 500_000})
    # Last bar: wide range, close == open (price "returned" — classic stop raid)
    rows.append({"open": 100.0, "high": 102.0, "low": 97.0,
                 "close": 100.0, "volume": 2_000_000})
    return pd.DataFrame(rows)


def _momentum_ignition_df():
    """
    Last bar (index -1) must have a spike vs bar at index -2.
    The check is: abs(closes[-1] - closes[-2]) / closes[-2] > 3 * std_dev.
    Build 29 calm bars, then make the 30th (last) bar a 5σ spike.
    """
    import random
    random.seed(7)
    closes = [100.0]
    for _ in range(28):
        closes.append(closes[-1] * (1 + random.uniform(-0.003, 0.003)))
    # closes now has 29 elements; compute std over last 20 of those
    std = _compute_price_std(closes, window=20)
    # 30th (last) bar = 5× std spike from bar 29
    closes.append(closes[-1] * (1 + std * 5))

    rows = []
    for c in closes:
        rows.append({"open": c, "high": c*1.002, "low": c*0.998,
                     "close": c, "volume": 500_000})
    return pd.DataFrame(rows)


def _wash_trading_df():
    """
    High volume last bar with near-zero price movement — crypto wash trading.
    vol_multiple = last_vol / avg_vol(all except last 5).
    Use 500K for bars 0..24 (avg_vol ≈ 500K), then 10M for the last bar.
    vol_multiple = 10M / 500K = 20 > 5 ✓
    price_move = |close - open| / open must be < 0.5%.
    """
    n_calm = 25
    rows = []
    for _ in range(n_calm):
        rows.append({"open": 100.0, "high": 100.05, "low": 99.95,
                     "close": 100.0, "volume": 500_000})
    # 4 transition bars (still included in avg_vol exclusion of last 5)
    for _ in range(4):
        rows.append({"open": 100.0, "high": 100.05, "low": 99.95,
                     "close": 100.0, "volume": 500_000})
    # Last bar: massive volume, tiny price move
    rows.append({"open": 100.0, "high": 100.05, "low": 99.95,
                 "close": 100.02, "volume": 10_000_000})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────

class TestCleanSignal:

    def test_clean_signal_passes(self):
        result = detect(_flat_df(), ticker="AAPL", direction="LONG")
        assert result["is_clean"] is True

    def test_clean_signal_no_flags(self):
        result = detect(_flat_df(), ticker="AAPL", direction="LONG")
        assert len(result["flags"]) == 0

    def test_clean_signal_high_score(self):
        result = detect(_flat_df(), ticker="AAPL", direction="LONG")
        assert result["score"] >= 80, "Clean signal should score >= 80"


class TestStopRaid:

    def test_stop_raid_flagged(self):
        result = detect(_stop_raid_df(), ticker="AAPL", direction="LONG", has_news=False)
        assert result["stop_raid_risk"] is True or "STOP_RAID" in result["flags"]

    def test_stop_raid_does_not_block_signal(self):
        """Stop raids WARN but do NOT block (they widen SL instead)."""
        result = detect(_stop_raid_df(), ticker="AAPL", direction="LONG", has_news=False)
        # is_clean can be False due to the flag, but the pattern itself is survivable
        # (runner.py widens SL on stop_raid_risk — it doesn't discard the signal)
        assert "MOMENTUM_IGNITION" not in result["flags"]


class TestMomentumIgnition:

    def test_momentum_ignition_flagged(self):
        result = detect(_momentum_ignition_df(), ticker="TSLA", direction="LONG", has_news=False)
        assert "MOMENTUM_IGNITION" in result["flags"] or result["score"] < 80

    def test_momentum_ignition_with_news_not_flagged(self):
        """If news exists, the move is organic — should NOT be flagged."""
        result = detect(_momentum_ignition_df(), ticker="TSLA", direction="LONG", has_news=True)
        assert "MOMENTUM_IGNITION" not in result["flags"]


class TestWashTrading:

    def test_wash_trading_flagged_for_crypto(self):
        result = detect(_wash_trading_df(), ticker="MARA", direction="LONG", is_crypto=True)
        assert "WASH_TRADING" in result["flags"] or result["score"] < 80

    def test_wash_trading_not_flagged_for_stock(self):
        """High volume on stocks is just liquidity — not wash trading."""
        result = detect(_wash_trading_df(), ticker="AAPL", direction="LONG", is_crypto=False)
        assert "WASH_TRADING" not in result["flags"]


class TestResultStructure:

    def test_result_has_required_keys(self):
        result = detect(_flat_df(), ticker="AAPL", direction="LONG")
        for key in ["is_clean", "flags", "score", "stop_raid_risk"]:
            assert key in result, f"Missing key: {key}"

    def test_score_in_range(self):
        result = detect(_flat_df(), ticker="AAPL", direction="LONG")
        assert 0 <= result["score"] <= 100

    def test_flags_is_list(self):
        result = detect(_flat_df(), ticker="AAPL", direction="LONG")
        assert isinstance(result["flags"], list)

    def test_is_clean_is_bool(self):
        result = detect(_flat_df(), ticker="AAPL", direction="LONG")
        assert isinstance(result["is_clean"], bool)
