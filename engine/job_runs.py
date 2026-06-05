"""
Per-job run ledger — records the last execution of EVERY scheduled job to the
`job_runs` table (one upserted row per job_id), so the Market-tab "Daily Jobs"
report (served from the WEB process) can show last-run / status / duration /
summary even though the scheduler lives on the WORKER. The DB row is the
cross-process bridge.

Wiring is a SINGLE APScheduler event listener (no per-job edits): attach() hooks
EVENT_JOB_SUBMITTED (start time) + EVENT_JOB_EXECUTED / EVENT_JOB_ERROR (finish).
Best-effort throughout — a ledger write must never disrupt a job.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger("signalbolt.job_runs")

# start timestamps keyed by job_id (monotonic-ish wall clock; only for duration)
_starts: dict[str, datetime] = {}


def _summary_of(retval) -> str | None:
    """A short human summary if the job returned one. Most jobs return None →
    no summary (the report still shows last-run + status)."""
    if retval is None:
        return None
    if isinstance(retval, str):
        return retval[:300] or None
    if isinstance(retval, dict):
        # common shapes: {'summary': ...} or small count dicts
        if isinstance(retval.get("summary"), str):
            return retval["summary"][:300]
        bits = [f"{k}={v}" for k, v in list(retval.items())[:6]
                if isinstance(v, (int, float, str, bool))]
        return ", ".join(bits)[:300] or None
    return None


def _record(sb, job_id: str, status: str, started: datetime | None,
            summary: str | None, error: str | None) -> None:
    if sb is None or not job_id:
        return
    now = datetime.now(timezone.utc)
    dur_ms = None
    if started is not None:
        dur_ms = int(max(0.0, (now - started).total_seconds()) * 1000)
    row = {
        "job_id":         job_id,
        "last_started":   started.isoformat() if started else None,
        "last_finished":  now.isoformat(),
        "last_status":    status,
        "last_duration_ms": dur_ms,
        "last_summary":   summary,
        "last_error":     (error or "")[:500] or None,
        "updated_at":     now.isoformat(),
    }
    try:
        # increment counters via read-modify-write (one row per job, low volume)
        prev = (sb.table("job_runs").select("run_count,error_count")
                  .eq("job_id", job_id).limit(1).execute().data or [])
        rc = (prev[0].get("run_count") if prev else 0) or 0
        ec = (prev[0].get("error_count") if prev else 0) or 0
        row["run_count"]   = rc + 1
        row["error_count"] = ec + (1 if status == "error" else 0)
        sb.table("job_runs").upsert(row, on_conflict="job_id").execute()
    except Exception as e:
        logger.debug(f"[job_runs] write failed for {job_id}: {e}")


def attach(scheduler, sb_factory) -> None:
    """Attach the single execution listener. sb_factory() → a supabase client."""
    try:
        from apscheduler.events import (
            EVENT_JOB_SUBMITTED, EVENT_JOB_EXECUTED, EVENT_JOB_ERROR, EVENT_JOB_MISSED,
        )
    except Exception as e:
        logger.warning(f"[job_runs] apscheduler events unavailable: {e}")
        return

    def _listener(event):
        try:
            jid = getattr(event, "job_id", None)
            if not jid:
                return
            code = event.code
            if code == EVENT_JOB_SUBMITTED:
                _starts[jid] = datetime.now(timezone.utc)
                return
            started = _starts.pop(jid, None)
            sb = None
            try:
                sb = sb_factory()
            except Exception:
                return
            if code == EVENT_JOB_EXECUTED:
                _record(sb, jid, "success", started,
                        _summary_of(getattr(event, "retval", None)), None)
            elif code == EVENT_JOB_ERROR:
                exc = getattr(event, "exception", None)
                _record(sb, jid, "error", started, None, repr(exc) if exc else "error")
            elif code == EVENT_JOB_MISSED:
                _record(sb, jid, "missed", started, None, "missed run (scheduler busy/down)")
        except Exception as e:
            logger.debug(f"[job_runs] listener error: {e}")

    mask = EVENT_JOB_SUBMITTED | EVENT_JOB_EXECUTED | EVENT_JOB_ERROR | EVENT_JOB_MISSED
    scheduler.add_listener(_listener, mask)
    logger.info("[job_runs] execution listener attached (job_runs ledger)")


def recent(sb) -> list[dict]:
    """All ledger rows (one per job). Best-effort → [] on any error."""
    try:
        return (sb.table("job_runs").select("*").limit(200).execute().data) or []
    except Exception as e:
        logger.debug(f"[job_runs] read failed: {e}")
        return []
