"""Unit tests for the Market Pulse INTRADAY provisional read (Part B)."""
from datetime import datetime
from zoneinfo import ZoneInfo

import inspect

from engine.market_pulse import intraday


def test_curve_projection_higher_than_linear_midday():
    """The U-curve says LESS volume is done by midday than the clock implies, so it
    projects a HIGHER full-day volume than naive linear — i.e. linear understates."""
    curve = intraday._FALLBACK_FULL          # 13 buckets
    bucket = 6                                # ~ midday
    vol = 100.0
    curve_frac = intraday.expected_fraction(curve, bucket)        # ~0.51
    linear_frac = (bucket + 1) / len(curve)                       # ~0.538 (clock)
    assert curve_frac < linear_frac
    assert intraday.project_volume(vol, curve_frac) > intraday.project_volume(vol, linear_frac)


def test_confidence_floor():
    assert intraday.confidence_for(10.0) == "TOO_EARLY"
    assert intraday.confidence_for(12.0) == "MEDIUM"
    assert intraday.confidence_for(15.0) == "HIGH"


def test_classify_status():
    assert intraday.classify_status(2.0e9, 1.0e9, -0.5, 0.3) == "ON_PACE_DISTRIBUTION"
    assert intraday.classify_status(2.0e9, 1.0e9, 0.1, 0.3) == "ON_PACE_STALLING"
    assert intraday.classify_status(1.0e9, 1.0e9, -0.5, 0.3) == "NEUTRAL"   # vol doesn't clear
    assert intraday.classify_status(2.0e9, 1.0e9, 1.5, 0.3) == "NEUTRAL"    # gain too big (healthy)
    assert intraday.classify_status(2.0e9, 1.0e9, 0.1, 0.9) == "NEUTRAL"    # closed strong


def test_too_early_status(monkeypatch):
    monkeypatch.setattr(intraday, "_session_et", lambda now: (
        datetime(now.year, now.month, now.day, 9, 30, tzinfo=ZoneInfo("America/New_York")),
        datetime(now.year, now.month, now.day, 16, 0, tzinfo=ZoneInfo("America/New_York")),
    ))
    mon_10am = datetime(2026, 6, 15, 10, 0, tzinfo=ZoneInfo("America/New_York"))
    r = intraday.intraday_read("SPY", now_et=mon_10am)
    assert r["status"] == "TOO_EARLY" and r["provisional"] is True


def test_market_closed(monkeypatch):
    monkeypatch.setattr(intraday, "_session_et", lambda now: None)   # not a trading day
    r = intraday.intraday_read("SPY", now_et=datetime(2026, 6, 13, 12, 0, tzinfo=ZoneInfo("America/New_York")))
    assert r["status"] == "MARKET_CLOSED"


def test_read_summary():
    # both too early
    assert "too early" in intraday._read_summary(
        {"SPY": {"status": "TOO_EARLY"}, "QQQ": {"status": "TOO_EARLY"}}).lower()
    # one on-pace distribution → building + not-confirmed
    line = intraday._read_summary({
        "SPY": {"status": "ON_PACE_DISTRIBUTION", "confidence": "MEDIUM"},
        "QQQ": {"status": "NEUTRAL", "confidence": "MEDIUM"},
    }).lower()
    assert "distribution day" in line and "not confirmed until the close" in line and "spy" in line
    # all closed
    assert "closed" in intraday._read_summary({"SPY": {"status": "MARKET_CLOSED"}}).lower()


def test_intraday_has_no_daily_write_path():
    """Integrity firewall: the intraday module has no DB-WRITE code path — it never
    imports the daily store, never upserts, never touches a Supabase table."""
    src = inspect.getsource(intraday)
    assert "upsert" not in src           # no writes of any kind
    assert ".table(" not in src          # no direct Supabase table access
    assert "import store" not in src and "from .store" not in src
