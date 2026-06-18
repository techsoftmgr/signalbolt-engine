"""Unit tests — trend_ride: confirmed-green swing ride gate, MA-trail, decisive-close exit."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine import trend_ride as tr


def _ctx(ma20, ma20_prev, last_close, swing_low=0.0, swing_high=0.0, atr=4.0):
    return {"ma20": ma20, "ma20_prev": ma20_prev, "last_close": last_close,
            "swing_low": swing_low, "swing_high": swing_high, "atr": atr}


def _sig(direction="LONG", entry=90.0, riding=False):
    return {"direction": direction, "entry_price": entry,
            "score_breakdown": {"trend_ride": True} if riding else {}}


# ── swing gate ───────────────────────────────────────────────────────────────
def test_is_swing_by_timeframe_and_strategy():
    assert tr.is_swing({"timeframe": "1Day", "strategy_type": "x"}) is True
    assert tr.is_swing({"timeframe": "15m", "strategy_type": "breakout"}) is True   # swing strategy
    assert tr.is_swing({"timeframe": "15m", "strategy_type": "day_trade"}) is False
    assert tr.is_swing({"timeframe": "5m", "strategy_type": "scalping"}) is False


# ── activation: green + above a RISING 20-MA ──────────────────────────────────
def test_active_long_when_green_above_rising_ma():
    d = tr.evaluate(_sig("LONG", 90), price=100, ctx=_ctx(ma20=95, ma20_prev=92, last_close=99, swing_low=93))
    assert d["active"] is True and d["break_exit"] is False
    # trail sits UNDER the recent swing low (93) by the ATR buffer (4*0.25=1.0) → 92.0
    assert d["trail_sl"] == 92.0


def test_not_active_when_ma_flat_or_falling():
    d = tr.evaluate(_sig("LONG", 90), price=100, ctx=_ctx(ma20=95, ma20_prev=96, last_close=99, swing_low=93))
    assert d["active"] is False          # MA not rising → chop, don't ride
    assert d["break_exit"] is False


def test_not_active_when_below_ma_and_not_yet_riding():
    # fresh green signal that hasn't cleared the MA → falls through to normal mgmt, NOT closed
    d = tr.evaluate(_sig("LONG", 90, riding=False), price=100, ctx=_ctx(ma20=95, ma20_prev=92, last_close=94))
    assert d["active"] is False and d["break_exit"] is False


def test_not_active_when_not_green():
    d = tr.evaluate(_sig("LONG", 90), price=88, ctx=_ctx(ma20=95, ma20_prev=92, last_close=99, swing_low=93))
    assert d["active"] is False          # underwater → not a ride


# ── exit: a RIDING trade whose daily close crosses back through the MA ────────
def test_break_exit_only_when_was_riding():
    ctx = _ctx(ma20=95, ma20_prev=92, last_close=94)   # close fell below MA
    riding = tr.evaluate(_sig("LONG", 90, riding=True), price=100, ctx=ctx)
    assert riding["break_exit"] is True and riding["active"] is False
    fresh = tr.evaluate(_sig("LONG", 90, riding=False), price=100, ctx=ctx)
    assert fresh["break_exit"] is False  # never trend-close a position that wasn't riding


# ── SHORT mirror ──────────────────────────────────────────────────────────────
def test_active_short_when_green_below_falling_ma():
    d = tr.evaluate(_sig("SHORT", 100), price=90, ctx=_ctx(ma20=95, ma20_prev=98, last_close=92, swing_high=96))
    assert d["active"] is True and d["break_exit"] is False
    assert d["trail_sl"] == 97.0         # swing high (96) + 1.0 buffer

def test_short_break_exit_on_close_above_ma():
    d = tr.evaluate(_sig("SHORT", 100, riding=True), price=90, ctx=_ctx(ma20=95, ma20_prev=98, last_close=96))
    assert d["break_exit"] is True


# ── kill switch ───────────────────────────────────────────────────────────────
def test_enabled_killswitch(monkeypatch):
    monkeypatch.setenv("TREND_RIDE_ENABLED", "false")
    assert tr.enabled() is False
    monkeypatch.setenv("TREND_RIDE_ENABLED", "true")
    assert tr.enabled() is True
    monkeypatch.delenv("TREND_RIDE_ENABLED", raising=False)
    assert tr.enabled() is True           # default on


# ── daily_context: drops today's forming bar, computes MA + swing low ─────────
def test_daily_context_uses_completed_bars(monkeypatch):
    import pandas as pd
    from datetime import datetime, timedelta
    # 25 daily bars ending TODAY (ET) — the last (today) bar must be dropped as forming
    today = datetime.now(tr._ET).date()
    N = 30
    idx = pd.to_datetime([today - timedelta(days=(N - 1 - i)) for i in range(N)])
    closes = [100 + i for i in range(N)]           # steadily rising → MA rising
    df = pd.DataFrame({"close": closes,
                       "high": [c + 2 for c in closes],
                       "low":  [c - 2 for c in closes]}, index=idx)
    monkeypatch.setattr(tr.smc, "fetch_candles", lambda *a, **k: df)
    tr.reset_cache()
    ctx = tr.daily_context("TEST")
    assert ctx is not None
    # last COMPLETED close is the 2nd-to-last row (today's forming bar dropped)
    assert ctx["last_close"] == closes[-2]
    assert ctx["ma20"] > ctx["ma20_prev"]          # rising MA
    assert ctx["swing_low"] > 0
