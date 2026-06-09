"""Unit tests — overnight (Blue Ocean) display-only data path. Additive, offline.

Covers the session gate (is_overnight_now), the price_store 'overnight' tag, and
the safe-empty behavior of the overnight price fetch. The overnight path is
DISPLAY-ONLY and must never feed the signal/stop engine — these tests pin the
gating, not any signal behavior."""
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine import session_classifier as sc
from engine import alpaca_client as ac
from engine import price_store as ps


def _et(y, mo, d, h, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=sc.ET)


# ── is_overnight_now ────────────────────────────────────────────────────────────
def test_overnight_true_monday_evening(monkeypatch):
    monkeypatch.setattr(sc, "_et_now", lambda: _et(2026, 6, 8, 21, 0))   # Mon 9pm
    assert sc.is_overnight_now() is True


def test_overnight_true_tuesday_early_am(monkeypatch):
    monkeypatch.setattr(sc, "_et_now", lambda: _et(2026, 6, 9, 2, 30))   # Tue 2:30am
    assert sc.is_overnight_now() is True


def test_overnight_false_midday(monkeypatch):
    monkeypatch.setattr(sc, "_et_now", lambda: _et(2026, 6, 10, 10, 0))  # Wed 10am
    assert sc.is_overnight_now() is False


def test_overnight_false_friday_night(monkeypatch):
    # Fri night has no overnight session (market reopens Sunday 8pm)
    monkeypatch.setattr(sc, "_et_now", lambda: _et(2026, 6, 12, 21, 0))  # Fri 9pm
    assert sc.is_overnight_now() is False


def test_overnight_true_sunday_night(monkeypatch):
    monkeypatch.setattr(sc, "_et_now", lambda: _et(2026, 6, 14, 21, 0))  # Sun 9pm
    assert sc.is_overnight_now() is True


def test_overnight_false_saturday_early_am(monkeypatch):
    monkeypatch.setattr(sc, "_et_now", lambda: _et(2026, 6, 13, 2, 0))   # Sat 2am
    assert sc.is_overnight_now() is False


# ── price_store session tag ─────────────────────────────────────────────────────
class _FakeDateTime:
    _now = None
    @classmethod
    def now(cls, tz=None):
        return cls._now


def test_price_store_tags_overnight(monkeypatch):
    fake = _FakeDateTime
    fake._now = _et(2026, 6, 8, 22, 0)     # Mon 10pm → overnight
    monkeypatch.setattr(ps, "datetime", fake)
    assert ps._market_session_now() == "overnight"


def test_price_store_tags_market_hours(monkeypatch):
    fake = _FakeDateTime
    fake._now = _et(2026, 6, 8, 11, 0)     # Mon 11am → market
    monkeypatch.setattr(ps, "datetime", fake)
    assert ps._market_session_now() == "market"


def test_price_store_weekend_overnight_window_is_closed(monkeypatch):
    fake = _FakeDateTime
    fake._now = _et(2026, 6, 12, 22, 0)    # Fri 10pm → no overnight → closed
    monkeypatch.setattr(ps, "datetime", fake)
    assert ps._market_session_now() == "closed"


# ── overnight price fetch — safe-empty ──────────────────────────────────────────
def test_get_overnight_prices_empty_input():
    assert ac.get_overnight_prices([]) == {}


def test_get_overnight_prices_no_client(monkeypatch):
    # Force the no-client path → returns {} (dormant), never raises.
    monkeypatch.setattr(ac, "_ok", False)
    monkeypatch.setattr(ac, "_client", None)
    monkeypatch.setattr(ac, "_init", lambda: None)
    assert ac.get_overnight_prices(["AAPL", "TSLA"]) == {}
