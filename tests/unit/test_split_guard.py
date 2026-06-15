"""
Unit tests — signal_monitor corporate-action (split) guard.

Regression cover for the KLAC 10:1 split (2026-06-12): Alpaca bars are split-
adjusted but a stored signal's levels are nominal-at-entry, so after a split the
price jumps scale (entry ~$2,000 vs price ~$240) and a naive monitor books a
phantom -90% stop. The guard detects a CONFIRMED split (clean ratio vs the split-
adjusted entry-date close) and rescales the levels instead of closing — while a
real price crash (no split in the adjusted history) is left untouched.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unittest.mock import MagicMock, patch
import pandas as pd

from engine import signal_monitor as sm


# ── detect_split_factor (pure) ──────────────────────────────────────────────
def test_detect_split_factor_forward():
    # KLAC 10:1 — entry $2,319.76, adjusted scale ~$232
    assert sm.detect_split_factor(2319.76, 232.0) == 10.0
    assert sm.detect_split_factor(400.0, 200.0) == 2.0      # 2:1
    assert sm.detect_split_factor(300.0, 100.0) == 3.0      # 3:1


def test_detect_split_factor_reverse():
    # reverse 1:10 — price rose 10×
    assert sm.detect_split_factor(5.0, 50.0) == 0.1


def test_detect_split_factor_rejects_normal_moves():
    assert sm.detect_split_factor(100.0, 130.0) is None     # +30% move, not a split
    assert sm.detect_split_factor(100.0, 100.0) is None     # flat
    assert sm.detect_split_factor(100.0, 72.0) is None      # -28% crash, not a clean ratio


def test_detect_split_factor_guards_bad_input():
    assert sm.detect_split_factor(0, 100) is None
    assert sm.detect_split_factor(100, 0) is None
    assert sm.detect_split_factor(None, 100) is None
    assert sm.detect_split_factor("x", 100) is None


# ── _confirm_split_factor (split vs crash) ──────────────────────────────────
def _daily(entry_date_close, later_close):
    idx = pd.DatetimeIndex([pd.Timestamp("2026-06-02", tz="UTC"),
                            pd.Timestamp("2026-06-10", tz="UTC")])
    return pd.DataFrame({"close": [entry_date_close, later_close]}, index=idx)


def test_confirm_split_fires_on_real_split():
    sig = {"ticker": "KLAC", "entry_price": 2000.0, "created_at": "2026-06-02T14:00:00+00:00"}
    # split-adjusted entry-date close is 1/10 of the nominal entry → factor 10
    with patch.object(sm._alpaca, "get_bars", return_value=_daily(200.0, 254.0)):
        assert sm._confirm_split_factor(sig, current_price=254.0) == 10.0


def test_confirm_split_ignores_a_real_crash():
    # entry 2000, price halved to 1000 (prefilter trips on ratio ~2), BUT the
    # adjusted entry-date close still ≈ entry (no split) → must NOT fire.
    sig = {"ticker": "ACME", "entry_price": 2000.0, "created_at": "2026-06-02T14:00:00+00:00"}
    with patch.object(sm._alpaca, "get_bars", return_value=_daily(2000.0, 1000.0)):
        assert sm._confirm_split_factor(sig, current_price=1000.0) is None


def test_confirm_split_noop_when_price_near_entry():
    # normal small move — cheap pre-filter rejects before any bar fetch
    sig = {"ticker": "ACME", "entry_price": 100.0, "created_at": "2026-06-02T14:00:00+00:00"}
    with patch.object(sm._alpaca, "get_bars", side_effect=AssertionError("should not fetch")):
        assert sm._confirm_split_factor(sig, current_price=103.0) is None


# ── _apply_split_adjustment (DB rescale) ────────────────────────────────────
def test_apply_split_adjustment_rescales_all_levels():
    sb = MagicMock()
    sig = {"id": "sig1", "ticker": "KLAC", "entry_price": 2000.0,
           "stop_loss": 2100.0, "target_one": 2400.0, "target_two": 2600.0}
    sm._apply_split_adjustment(sb, sig, 10.0)
    payload = sb.table.return_value.update.call_args[0][0]
    assert payload["entry_price"] == 200.0
    assert payload["stop_loss"] == 210.0
    assert payload["target_one"] == 240.0
    assert payload["target_two"] == 260.0


def test_apply_split_adjustment_reverse_multiplies():
    sb = MagicMock()
    sig = {"id": "s", "ticker": "X", "entry_price": 5.0, "stop_loss": 4.5, "target_one": 6.0}
    sm._apply_split_adjustment(sb, sig, 0.1)        # reverse 1:10 → ×10
    payload = sb.table.return_value.update.call_args[0][0]
    assert payload["entry_price"] == 50.0
    assert payload["stop_loss"] == 45.0
    assert payload["target_one"] == 60.0
