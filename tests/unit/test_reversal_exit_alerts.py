"""Unit tests — counter-signal (reversal-aware) exit alerts. Additive, offline."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine import reversal_exit_alerts as rx


def test_short_with_turnaround_forming_alerts():
    # open SHORT + a bottom (turnaround) forming → opposing reversal
    assert rx.opposing_reversal("SHORT", {"turnaroundStage": "forming"}) == ("turnaround", "forming")


def test_long_with_peak_forming_alerts():
    assert rx.opposing_reversal("LONG", {"peakStage": "confirmed"}) == ("peak", "confirmed")


def test_short_without_turnaround_is_quiet():
    assert rx.opposing_reversal("SHORT", {"turnaroundStage": "none", "peakStage": "forming"}) is None


def test_long_without_peak_is_quiet():
    # a turnaround forming is NOT opposing to a LONG (it's confirming) → no alert
    assert rx.opposing_reversal("LONG", {"turnaroundStage": "forming", "peakStage": "none"}) is None


def test_missing_snapshot_is_quiet():
    assert rx.opposing_reversal("SHORT", None) is None
    assert rx.opposing_reversal("SHORT", {}) is None


def test_stage_active_helper():
    assert rx._stage_active("forming") is True
    assert rx._stage_active("confirmed") is True
    assert rx._stage_active("none") is False
    assert rx._stage_active("") is False
    assert rx._stage_active(None) is False


def test_run_noop_without_flag(monkeypatch):
    monkeypatch.delenv("REVERSAL_EXIT_ALERTS_ENABLED", raising=False)
    assert rx.run(object())["alerts"] == 0


def test_run_noop_without_sb(monkeypatch):
    monkeypatch.setenv("REVERSAL_EXIT_ALERTS_ENABLED", "true")
    assert rx.run(None)["alerts"] == 0
