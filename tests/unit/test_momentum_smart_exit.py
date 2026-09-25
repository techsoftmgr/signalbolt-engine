"""TREND_MOMENTUM A/B — arm A routed through exit_engine (giveback + confluence +
exhaustion TP) instead of the chandelier. Pins the smart-exit management decisions."""
import numpy as np
import pandas as pd

from engine import momentum_monitor as mm


class _Rec:
    def __init__(self):
        self.closed, self.sl, self.events = [], [], []


def _patch(monkeypatch, rec):
    monkeypatch.setattr(mm, "_close_momentum",
                        lambda sb, sid, tk, d, e, p, why: rec.closed.append((tk, why)))
    monkeypatch.setattr(mm, "_update_sl",
                        lambda sb, sid, eff, sig=None: rec.sl.append(eff))
    monkeypatch.setattr(mm, "_log_event",
                        lambda *a, **k: rec.events.append(k.get("note")))


def _daily(closes, highs=None):
    closes = np.asarray(closes, float)
    idx = pd.date_range("2026-05-01", periods=len(closes), freq="D")
    return pd.DataFrame({"open": closes, "high": closes + 1 if highs is None else np.asarray(highs, float),
                         "low": closes - 1, "close": closes, "volume": [1e6] * len(closes)}, index=idx)


def _sig(ticker="AMD"):
    return {"id": "x", "ticker": ticker, "created_at": "2026-05-01"}


# ── kill switch ──────────────────────────────────────────────────────────────
def test_smart_exit_kill_switch_default_off(monkeypatch):
    monkeypatch.delenv("MOMENTUM_SMART_EXIT_ENABLED", raising=False)
    assert mm._smart_exit_on() is False
    monkeypatch.setenv("MOMENTUM_SMART_EXIT_ENABLED", "true")
    assert mm._smart_exit_on() is True


# ── giveback cap ─────────────────────────────────────────────────────────────
def test_closes_on_giveback_breach(monkeypatch):
    rec = _Rec(); _patch(monkeypatch, rec)
    # ran to a ~+20% peak then gave back below the +10.5% giveback floor
    closes = list(np.linspace(100, 120, 24)) + [108]
    df = _daily(closes)
    mm._manage_smart_exit(None, _sig(), df, 108.0, 100.0, 95.0, True, None,
                          {"closed": 0, "trailed": 0})
    assert rec.closed and "smart-exit" in rec.closed[0][1]


def test_ratchets_stop_while_holding(monkeypatch):
    rec = _Rec(); _patch(monkeypatch, rec)
    closes = list(np.linspace(100, 118, 25))          # +18%, above floor, still trending up
    df = _daily(closes)
    mm._manage_smart_exit(None, _sig("MU"), df, 118.0, 100.0, 95.0, True, None,
                          {"closed": 0, "trailed": 0})
    assert not rec.closed                              # still in the trade
    assert rec.sl and rec.sl[0] > 95                   # stop ratcheted UP to the giveback floor


# ── exhaustion take-profit ───────────────────────────────────────────────────
def test_takes_exhaustion_profit_into_a_spike(monkeypatch):
    rec = _Rec(); _patch(monkeypatch, rec)
    monkeypatch.setenv("SMART_EXIT_EXT_K", "4")
    closes = list(np.linspace(99, 100, 24)) + [130]    # flat base → parabolic stretch, in profit
    df = _daily(closes)
    mm._manage_smart_exit(None, _sig("MRVL"), df, 130.0, 100.0, 95.0, True, None,
                          {"closed": 0, "trailed": 0})
    assert rec.closed and "exhaustion" in rec.closed[0][1]


# ── hard stop backstop ───────────────────────────────────────────────────────
def test_hard_stop_on_daily_close_below_stop(monkeypatch):
    rec = _Rec(); _patch(monkeypatch, rec)
    closes = list(np.linspace(100, 96, 25))            # drifted to a daily close under the stop
    df = _daily(closes)
    mm._manage_smart_exit(None, _sig("NKE"), df, 96.0, 100.0, 97.0, True, None,
                          {"closed": 0, "trailed": 0})
    assert rec.closed and "stop" in rec.closed[0][1]


def test_peak_since_entry_uses_bars_after_entry():
    df = _daily([90, 95, 100, 110, 105])              # entry date = first bar
    assert mm._peak_since_entry(df, "2026-05-01", True) == 111.0   # max high (110+1)
