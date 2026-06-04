"""
Unit tests — result classification by P&L SIGN (engine/tracker.result_from_pnl_pct).

Regression for the 'ON +1.47% loss' corruption (2026-06-04): a breakout whose
TRAILING stop was raised above entry closed with reason='stop_hit' but IN PROFIT,
and the runner tagged it 'loss' (from the reason) while storing a positive
result_pct. Result must always follow the realized P&L sign, never the close
reason.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine.tracker import result_from_pnl_pct


class TestResultFromPnlPct:
    def test_positive_is_win_even_if_stop_hit(self):
        # The ON case: trailing stop hit while +1.47% in profit → WIN.
        assert result_from_pnl_pct(1.47) == "win"

    def test_negative_is_loss(self):
        assert result_from_pnl_pct(-2.3) == "loss"

    def test_zero_is_loss(self):
        assert result_from_pnl_pct(0.0) == "loss"

    def test_none_is_loss(self):
        assert result_from_pnl_pct(None) == "loss"

    def test_numeric_string_coerced(self):
        assert result_from_pnl_pct("3.2") == "win"
        assert result_from_pnl_pct("-1.0") == "loss"

    def test_garbage_is_loss(self):
        assert result_from_pnl_pct("n/a") == "loss"


class TestRunnerCloseLogicParity:
    """Mirrors the runner._close_signals stock-close branch: result must come
    from the realized pct sign, not from whether target or stop was the reason."""
    @staticmethod
    def _close(entry, close_price, is_long, reason):
        raw_pct = ((close_price - entry) / entry) * 100 if is_long \
                  else ((entry - close_price) / entry) * 100
        return result_from_pnl_pct(round(raw_pct, 4)), round(raw_pct, 4)

    def test_trailing_stop_in_profit_long_is_win(self):
        # LONG entry 100, trailing stop raised to 101.47 → close 101.47, reason stop_hit
        res, pct = self._close(100.0, 101.47, True, "stop_hit")
        assert pct > 0 and res == "win"

    def test_real_stop_below_entry_long_is_loss(self):
        res, pct = self._close(100.0, 97.5, True, "stop_hit")
        assert pct < 0 and res == "loss"

    def test_short_trailing_stop_in_profit_is_win(self):
        # SHORT entry 100, price fell then trailing stop at 98.5 (still profit)
        res, pct = self._close(100.0, 98.5, False, "stop_hit")
        assert pct > 0 and res == "win"

    def test_target_hit_is_win(self):
        res, pct = self._close(100.0, 103.0, True, "target_hit")
        assert pct > 0 and res == "win"
