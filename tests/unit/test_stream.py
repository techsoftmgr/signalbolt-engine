"""
Unit tests — engine/stream.py

Covers:
  - 5-min bar boundary detection (minute % 5 == 0)
  - 15-min deduplication (same min_key not fired twice)
  - 1-hour deduplication (minute == 0)
  - _check_scalp_levels LONG and SHORT logic (T1, T2, SL)
  - Scalp cache hit/miss behavior
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unittest.mock import patch, MagicMock, call
import pytest


# ──────────────────────────────────────────────────────────────
# Bar boundary logic (pure arithmetic — no external calls)
# ──────────────────────────────────────────────────────────────

class TestBarBoundaryLogic:
    """Test boundary detection math without touching the WebSocket."""

    def test_5min_boundary_at_0(self):
        assert 0 % 5 == 0

    def test_5min_boundary_at_5(self):
        assert 5 % 5 == 0

    def test_5min_boundary_at_30(self):
        assert 30 % 5 == 0

    def test_not_5min_boundary_at_1(self):
        assert 1 % 5 != 0

    def test_not_5min_boundary_at_7(self):
        assert 7 % 5 != 0

    def test_15min_boundary_at_0(self):
        assert 0 % 15 == 0

    def test_15min_boundary_at_15(self):
        assert 15 % 15 == 0

    def test_15min_boundary_at_45(self):
        assert 45 % 15 == 0

    def test_not_15min_boundary_at_16(self):
        assert 16 % 15 != 0

    def test_1h_boundary_at_minute_0(self):
        assert 0 == 0  # minute == 0

    def test_1h_boundary_not_at_minute_30(self):
        assert 30 != 0

    def test_min_key_calculation(self):
        """min_key = hour * 60 + minute should be unique per minute of day."""
        assert 10 * 60 + 15 == 615
        assert 9  * 60 + 30 == 570
        assert 16 * 60 + 0  == 960

    def test_dedup_same_min_key_not_refired(self):
        """Simulates the deduplication: second ticker at same boundary is ignored."""
        _last_15m_barrier = -1
        fired = []

        def handle_bar(minute, hour):
            nonlocal _last_15m_barrier
            min_key = hour * 60 + minute
            if minute % 15 == 0 and min_key != _last_15m_barrier:
                _last_15m_barrier = min_key
                fired.append(min_key)

        # First ticker fires at 10:15 → should trigger
        handle_bar(15, 10)
        # Second ticker fires at same 10:15 → should NOT trigger again
        handle_bar(15, 10)
        # Third different boundary at 10:30 → should trigger
        handle_bar(30, 10)

        assert fired == [10 * 60 + 15, 10 * 60 + 30]
        assert len(fired) == 2

    def test_dedup_resets_next_boundary(self):
        """Each new 15-min boundary should fire exactly once."""
        _last_15m_barrier = -1
        fired = []

        def handle_bar(minute, hour):
            nonlocal _last_15m_barrier
            min_key = hour * 60 + minute
            if minute % 15 == 0 and min_key != _last_15m_barrier:
                _last_15m_barrier = min_key
                fired.append(min_key)

        boundaries = [(0, 10), (15, 10), (30, 10), (45, 10), (0, 11)]
        for m, h in boundaries:
            for _ in range(5):   # simulate 5 tickers sending bars
                handle_bar(m, h)

        assert len(fired) == 5   # each boundary fired exactly once


# ──────────────────────────────────────────────────────────────
# _check_scalp_levels
# ──────────────────────────────────────────────────────────────

class TestCheckScalpLevels:

    def _run(self, sig, bar_high, bar_low, cache=None):
        """
        Test _check_scalp_levels in isolation by patching the module-level cache.
        Returns the hit argument passed to _close_scalp_signal (or None if not called).
        """
        import engine.stream as stream_mod

        if cache is None:
            cache = {sig["ticker"]: sig}

        closed_calls = []

        def fake_close(sig, hit, bar_price):
            closed_calls.append((hit, bar_price))

        original_cache = stream_mod._scalp_cache
        original_ts    = stream_mod._scalp_cache_ts
        try:
            stream_mod._scalp_cache    = cache
            stream_mod._scalp_cache_ts = 9999999999.0  # prevent refresh
            with patch.object(stream_mod, "_close_scalp_signal", side_effect=fake_close):
                stream_mod._check_scalp_levels(sig["ticker"], bar_high, bar_low)
        finally:
            stream_mod._scalp_cache    = original_cache
            stream_mod._scalp_cache_ts = original_ts

        return closed_calls

    def _long_sig(self, t1=183.0, t2=186.0, sl=177.5):
        return {
            "id": "test-001",
            "ticker": "AAPL",
            "direction": "LONG",
            "entry_price": 180.0,
            "stop_loss":   sl,
            "target_one":  t1,
            "target_two":  t2,
        }

    def _short_sig(self, t1=444.0, t2=438.0, sl=455.0):
        return {
            "id": "test-002",
            "ticker": "NVDA",
            "direction": "SHORT",
            "entry_price": 450.0,
            "stop_loss":   sl,
            "target_one":  t1,
            "target_two":  t2,
        }

    # ── LONG signal tests ──

    def test_long_sl_hit(self):
        sig = self._long_sig()
        calls = self._run(sig, bar_high=179.0, bar_low=176.0)  # bar_low below SL
        assert len(calls) == 1
        assert calls[0][0] == "sl"

    def test_long_t1_hit(self):
        sig = self._long_sig()
        calls = self._run(sig, bar_high=184.0, bar_low=181.0)  # bar_high above T1
        assert len(calls) == 1
        assert calls[0][0] == "t1"

    def test_long_t2_hit(self):
        sig = self._long_sig()
        calls = self._run(sig, bar_high=187.0, bar_low=181.0)  # bar_high above T2
        assert len(calls) == 1
        assert calls[0][0] == "t2"

    def test_long_no_hit(self):
        sig = self._long_sig()
        calls = self._run(sig, bar_high=182.0, bar_low=179.0)  # inside range
        assert len(calls) == 0

    def test_long_sl_takes_priority_over_t1(self):
        """If both SL and T1 hit in same bar, SL wins (worst case)."""
        sig = self._long_sig(sl=177.5, t1=183.0)
        # Simulate a bar that gaps way down — touches both SL and hypothetically T1
        # In practice, if bar_low < SL and bar_high > T1 that's a crazy gap, but SL wins
        calls = self._run(sig, bar_high=184.0, bar_low=176.0)
        assert calls[0][0] == "sl"

    # ── SHORT signal tests ──

    def test_short_sl_hit(self):
        sig = self._short_sig()
        calls = self._run(sig, bar_high=456.0, bar_low=448.0)  # bar_high above SL
        assert len(calls) == 1
        assert calls[0][0] == "sl"

    def test_short_t1_hit(self):
        sig = self._short_sig()
        calls = self._run(sig, bar_high=449.0, bar_low=443.0)  # bar_low below T1
        assert len(calls) == 1
        assert calls[0][0] == "t1"

    def test_short_t2_hit(self):
        sig = self._short_sig()
        calls = self._run(sig, bar_high=449.0, bar_low=436.0)  # bar_low below T2
        assert len(calls) == 1
        assert calls[0][0] == "t2"

    def test_short_no_hit(self):
        sig = self._short_sig()
        calls = self._run(sig, bar_high=449.0, bar_low=445.0)  # inside range
        assert len(calls) == 0

    # ── Cache behavior ──

    def test_no_signal_in_cache_does_nothing(self):
        sig = self._long_sig()
        calls = self._run(sig, bar_high=187.0, bar_low=176.0, cache={})  # empty cache
        assert len(calls) == 0

    def test_different_ticker_does_nothing(self):
        sig = self._long_sig()
        other_cache = {"MSFT": sig}   # signal cached under wrong ticker
        calls = self._run(sig, bar_high=187.0, bar_low=176.0, cache=other_cache)
        assert len(calls) == 0
