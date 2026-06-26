"""
rs_pullback_signals — turn the RS-leader-pullback edge into a live LONG signal.

Covers the pure level math + the kill switch + the scan's dedup/PANIC/cap gates.
"""
from engine import rs_pullback_signals as rp


def _row(tk="TSM", price=100.0, ma20=99.5, atrpct=2.0, rs=5.0, cat="rs_pullback"):
    return {"ticker": tk, "price": price, "ma20": ma20, "atrPct": atrpct,
            "rsVsSpy": rs, "regimeCategory": cat, "relativeVolume": 1.2}


def test_build_signal_row_levels():
    row = rp.build_signal_row(_row(price=100.0, ma20=99.5, atrpct=2.0))
    assert row["direction"] == "LONG" and row["timeframe"] == "1Day"
    assert row["strategy_type"] == "swing_trade"                 # → trend_ride rides it
    assert row["score_breakdown"]["detector_source"] == "RS_PULLBACK"
    # stop sits below BOTH entry and the 20-MA; targets above entry; R:R positive
    assert row["stop_loss"] < row["entry_price"]
    assert row["stop_loss"] <= 99.5 * 0.99 + 1e-9
    assert row["target_one"] > row["entry_price"] < row["target_two"]
    assert row["risk_reward"] > 0


def test_build_returns_none_on_bad_data():
    assert rp.build_signal_row({"ticker": "X", "price": 0, "ma20": 10}) is None
    assert rp.build_signal_row({"ticker": "X", "price": 10, "ma20": 0}) is None
    # stop would land above entry (price far below a high MA) → unusable
    assert rp.build_signal_row(_row(price=50.0, ma20=100.0, atrpct=0.1)) is None


def test_kill_switch(monkeypatch):
    monkeypatch.setenv("RS_PULLBACK_SIGNALS_ENABLED", "false")
    assert rp.enabled() is False
    assert rp.scan_and_fire(object()) == 0          # disabled → no work, no raise
    monkeypatch.setenv("RS_PULLBACK_SIGNALS_ENABLED", "true")
    assert rp.enabled() is True


class _SB:
    """Minimal Supabase stub: returns the configured active-signals list."""
    def __init__(self, active): self._active = active
    def table(self, *_): return self
    def select(self, *_): return self
    def eq(self, *_): return self
    def execute(self): return type("R", (), {"data": self._active})()


def test_scan_fires_top_rs_and_dedups(monkeypatch):
    rows = [_row("TSM", rs=8.0), _row("SNOW", rs=2.0),
            _row("MSFT", rs=-10.0, cat="knife")]      # knife filtered out
    monkeypatch.setattr(rp, "_SCORED_KEY", "k")
    from engine import cache
    monkeypatch.setattr(cache.kv, "get_json", lambda *a, **k: rows)
    monkeypatch.setattr(rp, "_has_active_long",
                        lambda sb, tk: tk == "SNOW")   # SNOW already long → skip
    fired = {}
    import engine.runner as runner
    monkeypatch.setattr(runner, "_write_signal",
                        lambda sb, row: fired.setdefault(row["ticker"], row) or "id1")
    from engine import signal_telemetry
    monkeypatch.setattr(signal_telemetry, "live_regime_type", lambda: "RISK_OFF")
    monkeypatch.setattr(signal_telemetry, "capture", lambda *a, **k: ("RISK_OFF", {}))
    import engine.push as push
    monkeypatch.setattr(push, "_send_raw", lambda *a, **k: None)

    n = rp.scan_and_fire(_SB([]))
    assert n == 1 and "TSM" in fired and "SNOW" not in fired and "MSFT" not in fired


def test_scan_skips_in_panic(monkeypatch):
    from engine import signal_telemetry
    monkeypatch.setattr(signal_telemetry, "live_regime_type", lambda: "PANIC")
    assert rp.scan_and_fire(_SB([])) == 0
