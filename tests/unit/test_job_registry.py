"""Unit tests — job_registry.merge (catalog × last-run ledger) for the Daily
Jobs report, and job_runs._summary_of."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from engine import job_registry as reg
from engine import job_runs


def test_every_job_has_required_fields():
    for j in reg.JOBS:
        for k in ("id", "label", "cadence", "schedule", "category", "what"):
            assert j.get(k), f"{j.get('id')} missing {k}"
    ids = [j["id"] for j in reg.JOBS]
    assert len(ids) == len(set(ids))                       # no dup ids
    assert "daily_performance" in ids and "phantom_audit" in ids


def test_merge_attaches_run_rows_and_sorts():
    runs = [
        {"job_id": "daily_performance", "last_finished": "2026-06-05T20:06:00Z",
         "last_status": "success", "last_duration_ms": 1200,
         "last_summary": "12 closed · win 58%", "run_count": 3, "error_count": 0},
        {"job_id": "phantom_audit", "last_finished": "2026-06-05T20:50:00Z",
         "last_status": "error", "last_error": "boom", "run_count": 5, "error_count": 1},
    ]
    merged = reg.merge(runs)
    by = {m["id"]: m for m in merged}
    assert by["daily_performance"]["status"] == "success"
    assert by["daily_performance"]["summary"] == "12 closed · win 58%"
    assert by["phantom_audit"]["status"] == "error" and by["phantom_audit"]["last_error"] == "boom"
    # jobs with no run row still appear, with null status
    assert by["maintenance"]["status"] is None
    assert len(merged) == len(reg.JOBS)
    # sorted: intraday cadence group comes before daily
    cadences = [reg.CADENCE_ORDER[m["cadence"]] for m in merged]
    assert cadences == sorted(cadences)


def test_merge_includes_unknown_ledger_id():
    merged = reg.merge([{"job_id": "brand_new_job", "last_status": "success"}])
    assert any(m["id"] == "brand_new_job" for m in merged)


def test_summary_of_shapes():
    assert job_runs._summary_of(None) is None
    assert job_runs._summary_of("12 closed") == "12 closed"
    assert "closed_n=2" in job_runs._summary_of({"closed_n": 2, "win": 50})
    assert job_runs._summary_of(123) is None
