"""Sector Leaders — Supabase persistence (sector_leaders_daily + _summary)."""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("signalbolt.sector_leaders.store")

_DAILY = "sector_leaders_daily"
_SUMMARY = "sector_leaders_summary"


def upsert_day(sb, date_iso: str, rows: list[dict], summary: dict) -> bool:
    try:
        payload = [{**r, "date": date_iso} for r in rows]
        sb.table(_DAILY).upsert(payload, on_conflict="date,sector_etf").execute()
        sb.table(_SUMMARY).upsert({
            "date": date_iso,
            "tape_character": summary["tape_character"],
            "top3": summary["top3"],
            "guidance_key": summary["guidance_key"],
        }, on_conflict="date").execute()
        return True
    except Exception as e:
        logger.error(f"[sector_leaders] upsert_day failed: {e}")
        return False


def get_today(sb) -> dict:
    """{date, summary, sectors[]} for the latest computed day, or {} if empty."""
    try:
        srow = (sb.table(_SUMMARY).select("*").order("date", desc=True).limit(1).execute().data or [None])[0]
        if not srow:
            return {}
        d = srow["date"]
        sectors = (sb.table(_DAILY).select("*").eq("date", d).order("rs_rank").execute().data) or []
        return {"date": d, "summary": srow, "sectors": sectors}
    except Exception as e:
        logger.error(f"[sector_leaders] get_today failed: {e}")
        return {}


def get_history(sb, sector: str, days: int = 90) -> list[dict]:
    try:
        res = (sb.table(_DAILY).select("date, rs_blended, rs_rank")
               .eq("sector_etf", sector.upper())
               .order("date", desc=True).limit(max(1, min(days, 400))).execute())
        return list(reversed(res.data or []))
    except Exception as e:
        logger.error(f"[sector_leaders] get_history failed: {e}")
        return []
