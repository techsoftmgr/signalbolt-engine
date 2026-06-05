"""
Unit tests — regime_history.record_if_changed appends a row ONLY when the regime
flips (write-on-change), with an in-memory short-circuit + cross-process dedupe.
"""
import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine import regime_history


def _sb(last_row=None):
    sb = MagicMock()
    sel = sb.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value
    sel.data = ([last_row] if last_row else [])
    sb.table.return_value.insert.return_value.execute.return_value = MagicMock()
    return sb


def _reset():
    regime_history._last.update(regime_type=None, session=None)


def _inserted(sb):
    ins = sb.table.return_value.insert
    return ins.call_args[0][0] if ins.called else None


def test_first_record_writes(monkeypatch):
    _reset()
    monkeypatch.setattr(regime_history, "session_now", lambda: "rth")
    sb = _sb(last_row=None)
    wrote = regime_history.record_if_changed(sb, {"regime_type": "PANIC", "vix": 30.0, "blocked": True})
    assert wrote is True
    row = _inserted(sb)
    assert row["regime_type"] == "PANIC" and row["session"] == "rth" and row["blocked"] is True


def test_no_write_when_unchanged(monkeypatch):
    _reset()
    monkeypatch.setattr(regime_history, "session_now", lambda: "rth")
    sb = _sb(last_row=None)
    regime_history.record_if_changed(sb, {"regime_type": "RANGING"})       # first → writes
    sb.table.return_value.insert.reset_mock()
    regime_history.record_if_changed(sb, {"regime_type": "RANGING"})       # same → in-memory short-circuit
    assert not sb.table.return_value.insert.called


def test_writes_on_flip(monkeypatch):
    _reset()
    monkeypatch.setattr(regime_history, "session_now", lambda: "rth")
    sb = _sb(last_row=None)
    regime_history.record_if_changed(sb, {"regime_type": "TRENDING_BULL"})
    sb.table.return_value.insert.reset_mock()
    wrote = regime_history.record_if_changed(sb, {"regime_type": "PANIC"})  # flip
    assert wrote is True and _inserted(sb)["regime_type"] == "PANIC"


def test_cross_process_dedupe(monkeypatch):
    _reset()
    monkeypatch.setattr(regime_history, "session_now", lambda: "rth")
    # another process already logged PANIC/rth → DB last row matches → skip
    sb = _sb(last_row={"regime_type": "PANIC", "session": "rth"})
    wrote = regime_history.record_if_changed(sb, {"regime_type": "PANIC"})
    assert wrote is False and not sb.table.return_value.insert.called


def test_session_flip_writes(monkeypatch):
    _reset()
    sb = _sb(last_row=None)
    monkeypatch.setattr(regime_history, "session_now", lambda: "pre")
    regime_history.record_if_changed(sb, {"regime_type": "TRENDING_BULL"})  # pre/bull
    sb.table.return_value.insert.reset_mock()
    monkeypatch.setattr(regime_history, "session_now", lambda: "rth")
    wrote = regime_history.record_if_changed(sb, {"regime_type": "TRENDING_BULL"})  # same regime, new session
    assert wrote is True and _inserted(sb)["session"] == "rth"


def test_empty_regime_noop(monkeypatch):
    _reset()
    monkeypatch.setattr(regime_history, "session_now", lambda: "rth")
    sb = _sb(last_row=None)
    assert regime_history.record_if_changed(sb, {"regime_type": ""}) is False
    assert not sb.table.return_value.insert.called
