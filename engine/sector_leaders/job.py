"""
Sector Leaders — daily orchestrator + backfill. Ranks the 11 SPDR sectors by
relative strength vs SPY for the last SETTLED session; idempotent upsert.
"""
from __future__ import annotations

import logging

import pandas as pd

from . import compute, config as C, data, store

logger = logging.getLogger("signalbolt.sector_leaders.job")


def _settled_cutoff(spy: pd.DataFrame):
    """Exclusive UTC cutoff for the last settled session (drop a forming same-day
    bar before 4pm ET); returns (cutoff_ts, date_iso)."""
    from datetime import datetime as _dt, timezone as _tz
    try:
        from zoneinfo import ZoneInfo
        now_et = _dt.now(ZoneInfo("America/New_York"))
    except Exception:
        now_et = _dt.now(_tz.utc)
    last = pd.Timestamp(spy.index[-1]).date()
    if last == now_et.date() and now_et.hour < 16:
        cutoff = pd.Timestamp(now_et.date()).tz_localize("UTC")
    else:
        cutoff = pd.Timestamp(last).tz_localize("UTC") + pd.Timedelta(days=1)
    return cutoff


def run_daily(sb) -> dict:
    bars = data.fetch_bars(days=400)
    spy = bars.get(C.BENCHMARK)
    if spy is None or len(spy) < C.L_6M + 2:
        logger.error("[sector_leaders] insufficient SPY history — aborting")
        return {}
    cutoff = _settled_cutoff(spy)
    sliced = {s: df[df.index < cutoff] for s, df in bars.items() if df is not None}
    spy_s = sliced.get(C.BENCHMARK)
    if spy_s is None or len(spy_s) < 2:
        return {}
    date_iso = pd.Timestamp(spy_s.index[-1]).date().isoformat()

    rows, summary = compute.compute(sliced)
    if not rows:
        logger.error("[sector_leaders] no rows computed — aborting")
        return {}
    store.upsert_day(sb, date_iso, rows, summary)
    logger.info(f"[sector_leaders] {date_iso} {summary['tape_character']} top3={summary['top3']}")
    return {"date": date_iso, **summary, "count": len(rows)}


def run_backfill(sb, days: int = 130) -> dict:
    """Replay the last `days` settled sessions from one bulk fetch."""
    bars = data.fetch_bars(days=900)
    spy = bars.get(C.BENCHMARK)
    if spy is None or len(spy) < C.L_6M + 2:
        logger.error("[sector_leaders] backfill: insufficient SPY history")
        return {"written": 0}
    dates = [pd.Timestamp(d).date() for d in spy.index]
    # Drop a still-forming final bar (today before the close) — settled sessions only.
    from datetime import datetime as _dt
    try:
        from zoneinfo import ZoneInfo as _ZI
        _now_et = _dt.now(_ZI("America/New_York"))
    except Exception:
        _now_et = None
    if dates and _now_et is not None and dates[-1] == _now_et.date() and _now_et.hour < 16:
        dates = dates[:-1]
    start_idx = max(C.L_6M + C.RANK_MOM_LOOKBACK, len(dates) - days)
    written = 0
    for i in range(start_idx, len(dates)):
        d = dates[i]
        cutoff = pd.Timestamp(d).tz_localize("UTC") + pd.Timedelta(days=1)
        sliced = {s: df[df.index < cutoff] for s, df in bars.items() if df is not None}
        rows, summary = compute.compute(sliced)
        if rows and store.upsert_day(sb, d.isoformat(), rows, summary):
            written += 1
    logger.info(f"[sector_leaders] backfill wrote {written} days")
    return {"written": written}
