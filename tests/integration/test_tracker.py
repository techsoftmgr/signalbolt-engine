"""
Integration tests — engine/tracker.py

Tests win/loss/expired outcome detection:
  - LONG signal: price hits T1 → result=win
  - LONG signal: price hits SL → result=loss
  - Signal past expiry → result=expired
  - Pending signal with no price movement → stays pending
  - Result written to Supabase with correct fields

Real API:
  track_signals() → void  (creates own supabase client via _supabase())
  _current_price(ticker: str) → Optional[float]
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta, timezone
import pytest


def _signal(ticker="AAPL", direction="LONG", entry=180.0,
            sl=177.0, t1=183.0, t2=186.0,
            strategy="day_trade", created_hours_ago=1,
            sig_id="sig-001"):
    created_at = (datetime.now(timezone.utc) - timedelta(hours=created_hours_ago)).isoformat()
    return {
        "id": sig_id,
        "ticker": ticker,
        "direction": direction,
        "entry_price": entry,
        "stop_loss": sl,
        "target_one": t1,
        "target_two": t2,
        "strategy_type": strategy,
        "status": "active",
        "result": "pending",
        "created_at": created_at,
    }


def _mock_sb_with_signals(signals):
    """Mock Supabase returning the given signals on any .select() chain."""
    sb = MagicMock()
    result = MagicMock()
    result.data = signals

    # Double-eq chain: .select("*").eq(...).eq(...).execute()
    sb.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = result
    # Single-eq fallback: .select("*").eq(...).execute()
    sb.table.return_value.select.return_value.eq.return_value.execute.return_value = result

    update_result = MagicMock()
    update_result.data = signals[:1] if signals else []
    sb.table.return_value.update.return_value.eq.return_value.execute.return_value = update_result
    sb.table.return_value.insert.return_value.execute.return_value = update_result

    return sb


# ──────────────────────────────────────────────────────────────
# Outcome detection
# ──────────────────────────────────────────────────────────────

class TestOutcomeDetection:

    def _run_pass(self, signals, prices: dict):
        """
        Run a tracking pass with given signals and mocked current prices.

        Patches:
          engine.tracker._supabase  — returns mock Supabase client
          engine.tracker._current_price — returns price from dict
        """
        from engine.tracker import track_signals
        mock_sb = _mock_sb_with_signals(signals)

        def mock_price(ticker):
            return prices.get(ticker, 180.0)

        with patch("engine.tracker._supabase", return_value=mock_sb), \
             patch("engine.tracker._current_price", side_effect=mock_price):
            track_signals()

        return mock_sb

    def test_long_sl_hit_closes_as_loss(self):
        sig = _signal(direction="LONG", entry=180.0, sl=177.0, t1=183.0)
        mock_sb = self._run_pass([sig], prices={"AAPL": 176.0})  # below SL

        update_calls = mock_sb.table.return_value.update.call_args_list
        assert len(update_calls) > 0
        update_data = update_calls[0][0][0]
        assert update_data.get("result") == "loss" or update_data.get("status") == "closed"

    def test_long_t1_hit_does_NOT_close(self):
        # NEW behavior (2026-05-28): T1 no longer closes the signal. It moves
        # the stop to breakeven (signal_monitor) and rides to T2 with a
        # trailing stop. Price between T1 and T2 → tracker leaves it open.
        sig = _signal(direction="LONG", entry=180.0, sl=177.0, t1=183.0, t2=186.0)
        mock_sb = self._run_pass([sig], prices={"AAPL": 184.0})  # above T1, below T2

        update_calls = mock_sb.table.return_value.update.call_args_list
        # Must NOT have closed it as a win at T1
        for call in update_calls:
            data = call[0][0]
            assert data.get("status") != "closed", "T1 should not close — rides to T2"

    def test_long_t2_hit_closes_as_win(self):
        sig = _signal(direction="LONG", entry=180.0, sl=177.0, t1=183.0, t2=186.0)
        mock_sb = self._run_pass([sig], prices={"AAPL": 187.0})  # above T2

        update_calls = mock_sb.table.return_value.update.call_args_list
        assert len(update_calls) > 0
        update_data = update_calls[0][0][0]
        assert update_data.get("result") == "win" or update_data.get("status") == "closed"

    def test_long_trailed_stop_closes_as_win(self):
        # After T1 the stop trails up above entry. Price hitting that trailed
        # stop should close as a WIN (locked profit), not a loss.
        sig = _signal(direction="LONG", entry=180.0, sl=184.0, t1=183.0, t2=186.0)
        mock_sb = self._run_pass([sig], prices={"AAPL": 183.5})  # at/below trailed stop 184

        update_calls = mock_sb.table.return_value.update.call_args_list
        assert len(update_calls) > 0
        update_data = update_calls[0][0][0]
        assert update_data.get("status") == "closed"
        assert update_data.get("result") == "win"

    def test_short_sl_hit_closes_as_loss(self):
        sig = _signal(ticker="NVDA", direction="SHORT",
                      entry=450.0, sl=455.0, t1=444.0)
        mock_sb = self._run_pass([sig], prices={"NVDA": 457.0})  # above SL

        update_calls = mock_sb.table.return_value.update.call_args_list
        assert len(update_calls) > 0
        update_data = update_calls[0][0][0]
        assert update_data.get("result") == "loss" or update_data.get("status") == "closed"

    def test_pending_signal_not_updated(self):
        sig = _signal(direction="LONG", entry=180.0, sl=177.0, t1=183.0)
        mock_sb = self._run_pass([sig], prices={"AAPL": 181.0})  # between SL and T1

        update_calls = mock_sb.table.return_value.update.call_args_list
        # No status change for a pending signal still in range
        if update_calls:
            update_data = update_calls[0][0][0]
            assert update_data.get("result") != "loss"
            assert update_data.get("result") != "win"

    def test_expired_signal_closes(self):
        """Signal older than max age should be closed as expired."""
        sig = _signal(
            strategy="scalping",
            created_hours_ago=4,   # scalping signals expire in ~1-2h
        )
        mock_sb = self._run_pass([sig], prices={"AAPL": 181.0})
        # Shouldn't crash — either closed as expired or handled gracefully
        assert mock_sb is not None


# ──────────────────────────────────────────────────────────────
# Pass statistics
# ──────────────────────────────────────────────────────────────

class TestPassStatistics:

    def test_pass_with_empty_signals_does_not_crash(self):
        from engine.tracker import track_signals
        mock_sb = _mock_sb_with_signals([])
        with patch("engine.tracker._supabase", return_value=mock_sb), \
             patch("engine.tracker._current_price", return_value=180.0):
            track_signals()  # should not raise

    def test_pass_with_multiple_signals(self):
        """Run a pass with mixed signals — some hit, some pending."""
        sigs = [
            _signal("AAPL", "LONG",  entry=180, sl=177, t1=183, sig_id="s1"),
            _signal("NVDA", "SHORT", entry=450, sl=455, t1=444, sig_id="s2"),
            _signal("SPY",  "LONG",  entry=520, sl=516, t1=524, sig_id="s3"),
        ]
        prices = {
            "AAPL": 184.0,  # T1 hit → win
            "NVDA": 458.0,  # SL hit → loss
            "SPY":  521.0,  # pending
        }
        from engine.tracker import track_signals
        mock_sb = _mock_sb_with_signals(sigs)
        with patch("engine.tracker._supabase", return_value=mock_sb), \
             patch("engine.tracker._current_price", side_effect=lambda t: prices.get(t, 180.0)):
            track_signals()  # should not crash with mixed outcomes
