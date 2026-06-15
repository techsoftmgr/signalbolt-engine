"""
Market Pulse — Supabase persistence. One row per `date` in market_pulse_daily.
We store the COMPUTED verdict (regime + guidance_key), not just raw inputs.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("signalbolt.market_pulse.store")

_TABLE = "market_pulse_daily"


def latest_cumulative_ad(sb) -> int:
    """Most recent cumulative A/D value (0 if the table is empty)."""
    try:
        res = (sb.table(_TABLE).select("ad_line_cumulative")
               .order("date", desc=True).limit(1).execute())
        if res.data:
            return int(res.data[0].get("ad_line_cumulative") or 0)
    except Exception as e:
        logger.debug(f"[market_pulse] latest_cumulative_ad failed: {e}")
    return 0


def cumulative_before(sb, date_iso: str) -> int:
    """Cumulative A/D as of the most recent row STRICTLY BEFORE `date_iso`. Used so
    re-running a given day is idempotent (today = prior-day cumulative + today's net,
    never today's existing value + net again)."""
    try:
        res = (sb.table(_TABLE).select("ad_line_cumulative")
               .lt("date", date_iso).order("date", desc=True).limit(1).execute())
        if res.data:
            return int(res.data[0].get("ad_line_cumulative") or 0)
    except Exception as e:
        logger.debug(f"[market_pulse] cumulative_before failed: {e}")
    return 0


def recent_ad_history(sb, n: int = 260) -> list[int]:
    """Last `n` cumulative A/D values, oldest→newest (for divergence detection)."""
    try:
        res = (sb.table(_TABLE).select("ad_line_cumulative")
               .order("date", desc=True).limit(n).execute())
        vals = [int(r.get("ad_line_cumulative") or 0) for r in (res.data or [])]
        return list(reversed(vals))
    except Exception as e:
        logger.debug(f"[market_pulse] recent_ad_history failed: {e}")
        return []


def upsert_daily(sb, row: dict) -> bool:
    """Idempotent write keyed on `date` (re-runs overwrite the same day)."""
    try:
        sb.table(_TABLE).upsert(row, on_conflict="date").execute()
        return True
    except Exception as e:
        logger.error(f"[market_pulse] upsert_daily failed: {e}")
        return False


def get_today(sb) -> Optional[dict]:
    try:
        res = sb.table(_TABLE).select("*").order("date", desc=True).limit(1).execute()
        return (res.data or [None])[0]
    except Exception as e:
        logger.error(f"[market_pulse] get_today failed: {e}")
        return None


def get_history(sb, days: int = 90) -> list[dict]:
    try:
        res = (sb.table(_TABLE).select("*")
               .order("date", desc=True).limit(max(1, min(days, 400))).execute())
        return list(reversed(res.data or []))
    except Exception as e:
        logger.error(f"[market_pulse] get_history failed: {e}")
        return []
