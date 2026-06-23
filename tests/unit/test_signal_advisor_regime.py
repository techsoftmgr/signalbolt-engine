"""Unit test — signal_advisor: market-wide regime-shift push fires ONCE, not per card."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine import signal_advisor as sa


def test_global_cooldown_fires_once_per_state(monkeypatch):
    sa._global_last.clear()
    # First position to hit the PANIC regime shift → pushes (True); the rest in the window → False
    assert sa._global_cooldown_ok("regime_shift", "PANIC") is True
    assert sa._global_cooldown_ok("regime_shift", "PANIC") is False   # 2nd open position → no push
    assert sa._global_cooldown_ok("regime_shift", "PANIC") is False   # 3rd → no push
    # A genuinely different regime state may notify again
    assert sa._global_cooldown_ok("regime_shift", "HIGH_VOL") is True


def test_send_advice_push_flag_logs_event_without_push(monkeypatch):
    # push=False must still log the per-card event but NOT call the push sender.
    calls = {"events": 0, "pushes": 0}

    class _FakeTbl:
        def insert(self, *a, **k): return self
        def execute(self, *a, **k): calls.__setitem__("events", calls["events"] + 1); return None
    class _FakeSB:
        def table(self, *a, **k): return _FakeTbl()
    import supabase
    monkeypatch.setattr(supabase, "create_client", lambda *a, **k: _FakeSB())
    monkeypatch.setenv("SUPABASE_URL", "http://x"); monkeypatch.setenv("SUPABASE_KEY", "k")
    from engine import push as _push
    monkeypatch.setattr(_push, "_send_raw", lambda *a, **k: calls.__setitem__("pushes", calls["pushes"] + 1))

    sig = {"id": "s1", "ticker": "HOOD", "direction": "SHORT"}
    sa._send_advice(sig, price=100.0, title="t", body="b", advice_type="regime_shift", push=False)
    assert calls["events"] == 1 and calls["pushes"] == 0     # event logged, no push
    sa._send_advice(sig, price=100.0, title="t", body="b", advice_type="adverse_move", push=True)
    assert calls["pushes"] == 1                              # push fires when push=True
