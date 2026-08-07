"""Entry guards (SNOW/GLD analysis): re-fire/dedup suppression + short-into-strength veto."""
import numpy as np
import pandas as pd

from engine import entry_guards as eg


class _FakeQ:
    def __init__(self, active, closed):
        self.active, self.closed, self._status = active, closed, None
    def select(self, *a, **k): return self
    def eq(self, col, val):
        if col == "status": self._status = val
        return self
    def gte(self, *a, **k): return self
    def order(self, *a, **k): return self
    def limit(self, *a, **k): return self
    def execute(self):
        return type("R", (), {"data": self.active if self._status == "active" else self.closed})()


class _FakeSB:
    def __init__(self, active=None, closed=None):
        self.active, self.closed = active or [], closed or []
    def table(self, _): return _FakeQ(self.active, self.closed)


def _df(closes, highs=None):
    closes = np.asarray(closes, float)
    return pd.DataFrame({"open": closes, "high": closes if highs is None else np.asarray(highs, float),
                         "low": closes - 1, "close": closes, "volume": [1e6] * len(closes)})


# ── suppression (re-fire cooldown + active dedup) ────────────────────────────
def test_suppress_active_dedup():
    sup, r = eg.should_suppress(_FakeSB(active=[{"id": "x"}]), "GLD", "LONG")
    assert sup and "dedup" in r


def test_suppress_recent_stopout():
    sb = _FakeSB(active=[], closed=[{"closed_reason": "stop_hit", "result_pct": -1.0}])
    sup, r = eg.should_suppress(sb, "SNOW", "SHORT")
    assert sup and "cooldown" in r


def test_suppress_none_when_clean():
    sb = _FakeSB(active=[], closed=[{"closed_reason": "target_hit", "result_pct": 2.0}])
    assert eg.should_suppress(sb, "AAPL", "LONG")[0] is False


# ── short-into-strength veto ─────────────────────────────────────────────────
def test_short_veto_blocks_strong_uptrend():
    up = list(np.linspace(200, 330, 40))            # strong uptrend, new highs, rising EMA
    spy = _df(np.linspace(400, 405, 40))            # stock (+65%) ≫ SPY (+1.25%)
    veto, _ = eg.short_into_strength("SNOW", _df(up), spy)
    assert veto is True


def test_short_veto_allows_rolled_over():
    dn = list(np.linspace(330, 300, 40))            # rolling over → below EMA, off the high
    veto, _ = eg.short_into_strength("SNOW", _df(dn), _df(np.linspace(400, 405, 40)))
    assert veto is False


def test_kill_switches_default_off(monkeypatch):
    for v in ("REFIRE_COOLDOWN_ENABLED", "SHORT_STRENGTH_VETO_ENABLED"):
        monkeypatch.delenv(v, raising=False)
    assert eg.cooldown_enabled() is False and eg.short_veto_enabled() is False
