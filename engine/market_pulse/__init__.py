"""
Market Pulse — standalone, market-wide, end-of-day regime read (IBD-style).
Completely separate from the per-signal confluence engine.

Public surface:
  job.run_daily(sb)        — compute + upsert today's row (scheduled ~45m after close)
  job.run_backfill(sb, n)  — seed the A/D line + history
  store.get_today(sb) / store.get_history(sb, days)
  guidance.build(regime, vix_band, vix_rising)
"""
from . import config, constituents, data, guidance, pillars, regime, store  # noqa: F401
from .job import run_backfill, run_daily  # noqa: F401

__all__ = ["run_daily", "run_backfill", "store", "guidance", "config"]
