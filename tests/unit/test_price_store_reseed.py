"""
price_store.reseed_prev_close — keeps the live changePercent baseline anchored
to the current session's prior close.

Regression guard for the "live price correct but % wrong across all watchlist
tickers" bug: the long-lived worker seeded prev_close once at boot, so after a
daily session rollover every streamed % was computed against a multi-day-old
close. reseed_prev_close refreshes the baseline WITHOUT disturbing the live
price, and recomputes the stored %.
"""
from engine import price_store


def _reset():
    price_store._prices.clear()
    price_store._prev_close.clear()
    price_store._dirty.clear()


def test_reseed_recomputes_pct_without_touching_price():
    _reset()
    # Boot-time seed: stale prev_close from days ago when the name sat higher.
    # MSFT live 368.57; a stale baseline of 367.34 yields a misleading +0.33%.
    price_store.seed("MSFT", 368.57, 0.33, "market")
    assert price_store._prices["MSFT"]["price"] == 368.57

    # Today's true prior close is 352.83 → the move is really +4.46%.
    price_store.reseed_prev_close("MSFT", 352.83)

    entry = price_store._prices["MSFT"]
    # Live price is untouched (no visible blip)...
    assert entry["price"] == 368.57
    # ...but the % is now correct against the refreshed baseline.
    assert round(entry["changePercent"], 1) == 4.5
    assert price_store._prev_close["MSFT"] == 352.83


def test_reseed_marks_ticker_dirty_for_immediate_push():
    _reset()
    price_store.seed("AAPL", 279.32, -5.0, "market")  # stale negative %
    price_store._dirty.clear()
    price_store.reseed_prev_close("AAPL", 275.10)
    # Even without _loop wired (call_soon_threadsafe skipped), the entry is fixed.
    assert round(price_store._prices["AAPL"]["changePercent"], 2) == 1.53


def test_reseed_ignores_bad_prev_close():
    _reset()
    price_store.seed("NVDA", 192.56, -1.62, "market")
    before = dict(price_store._prices["NVDA"])
    price_store.reseed_prev_close("NVDA", 0.0)      # garbage → no-op
    price_store.reseed_prev_close("NVDA", -5.0)     # garbage → no-op
    assert price_store._prices["NVDA"] == before


def test_reseed_noop_when_no_live_price_yet():
    _reset()
    # prev_close known but no price entry yet → just stores baseline, no crash.
    price_store.reseed_prev_close("TSLA", 375.10)
    assert price_store._prev_close["TSLA"] == 375.10
    assert "TSLA" not in price_store._prices
