"""Earnings guard — flatten a momentum position before its earnings print."""
import datetime as dt

from engine import earnings_guard as eg


def _info(datestr, when="After Close"):
    return {"date": datestr, "when": when, "daysAway": 0, "eps_estimate": None}


def test_enabled_default_off(monkeypatch):
    monkeypatch.delenv("EARNINGS_GUARD_ENABLED", raising=False)
    assert eg.enabled() is False
    monkeypatch.setenv("EARNINGS_GUARD_ENABLED", "true")
    assert eg.enabled() is True


def test_add_weekdays_skips_weekend():
    assert eg._add_weekdays(dt.date(2026, 9, 25), 1) == dt.date(2026, 9, 28)   # Fri → Mon
    assert eg._add_weekdays(dt.date(2026, 9, 28), 1) == dt.date(2026, 9, 29)   # Mon → Tue


def test_closes_the_trading_day_before_earnings(monkeypatch):
    monkeypatch.setenv("EARNINGS_GUARD_DAYS", "1")
    monkeypatch.setattr(eg.earnings_service, "get_next_earnings", lambda t: _info("2026-09-30"))  # Wed
    assert eg.should_close("MU", today=dt.date(2026, 9, 29))[0] is True    # Tue → close
    assert eg.should_close("MU", today=dt.date(2026, 9, 28))[0] is False   # Mon → not yet


def test_weekend_earnings_closes_on_friday(monkeypatch):
    monkeypatch.setenv("EARNINGS_GUARD_DAYS", "1")
    monkeypatch.setattr(eg.earnings_service, "get_next_earnings", lambda t: _info("2026-09-28"))  # Mon
    assert eg.should_close("X", today=dt.date(2026, 9, 25))[0] is True     # Fri closes before Mon print


def test_does_not_close_when_earnings_far_out(monkeypatch):
    monkeypatch.setenv("EARNINGS_GUARD_DAYS", "1")
    monkeypatch.setattr(eg.earnings_service, "get_next_earnings", lambda t: _info("2026-11-03"))
    assert eg.should_close("AMD", today=dt.date(2026, 9, 28))[0] is False


def test_fail_open_on_missing_data(monkeypatch):
    monkeypatch.setattr(eg.earnings_service, "get_next_earnings", lambda t: None)
    assert eg.should_close("X")[0] is False
    monkeypatch.setattr(eg.earnings_service, "get_next_earnings", lambda t: (_ for _ in ()).throw(Exception("boom")))
    assert eg.should_close("X")[0] is False
