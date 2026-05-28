"""
Armed-zone lifecycle logging.

Records every predictive zone arming and its eventual outcome (fired /
expired) into the `armed_zone_history` table so we can analyse conversion
rate and win-rate per detector over time. Previously zones were trashed at
the overnight clear with no trace.

Every function here is BEST-EFFORT: all DB work is wrapped so a logging
failure can never break the trading path. Called only from the scan / fire /
scheduler paths — never from the per-tick hot loop.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")


def _session_date_iso() -> str:
    return datetime.now(_ET).date().isoformat()


def log_armed(
    sb,
    *,
    ticker: str,
    detector: str,
    direction: str | None,
    armed_level: float | None,
    range_high: float | None,
    range_low: float | None,
    atr: float | None,
    relaxed: bool = False,
    ext_atr: float | None = None,
) -> None:
    """Insert one 'armed' row — called once when a zone is FRESHLY armed
    (not on every re-stage)."""
    try:
        sb.table("armed_zone_history").insert({
            "ticker":       ticker,
            "detector":     detector,
            "direction":    direction,
            "armed_level":  _f(armed_level),
            "range_high":   _f(range_high),
            "range_low":    _f(range_low),
            "atr":          _f(atr),
            "relaxed":      bool(relaxed),
            "ext_atr":      _f(ext_atr),
            "session_date": _session_date_iso(),
            "outcome":      "armed",
        }).execute()
    except Exception as e:
        logger.debug(f"[zone_history] log_armed failed for {ticker}/{detector}: {e}")


def mark_fired(sb, *, ticker: str, detector: str, fired_signal_id) -> None:
    """Close out the most-recent open ('armed') row for this ticker+detector
    as 'fired', linking the signal it produced."""
    try:
        rows = (
            sb.table("armed_zone_history")
              .select("id")
              .eq("ticker", ticker)
              .eq("detector", detector)
              .eq("outcome", "armed")
              .order("armed_at", desc=True)
              .limit(1)
              .execute()
        ).data or []
        if not rows:
            return
        sb.table("armed_zone_history").update({
            "outcome":         "fired",
            "outcome_at":      datetime.now(timezone.utc).isoformat(),
            "fired_signal_id": str(fired_signal_id) if fired_signal_id else None,
        }).eq("id", rows[0]["id"]).execute()
    except Exception as e:
        logger.debug(f"[zone_history] mark_fired failed for {ticker}/{detector}: {e}")


def expire_stale(sb) -> int:
    """Mark every still-'armed' row from a PRIOR session as 'expired'. Called
    by the overnight clear (00:30 ET) so unfired zones close out cleanly.
    Returns the number of rows expired."""
    try:
        today = _session_date_iso()
        res = (
            sb.table("armed_zone_history")
              .update({"outcome": "expired",
                       "outcome_at": datetime.now(timezone.utc).isoformat()})
              .eq("outcome", "armed")
              .lt("session_date", today)
              .execute()
        )
        n = len(res.data or [])
        logger.info(f"[zone_history] expired {n} unfired armed zones from prior sessions")
        return n
    except Exception as e:
        logger.debug(f"[zone_history] expire_stale failed: {e}")
        return 0


def _f(v):
    """Coerce numpy floats / None to plain JSON-safe float."""
    if v is None:
        return None
    try:
        return float(round(float(v), 4))
    except Exception:
        return None
