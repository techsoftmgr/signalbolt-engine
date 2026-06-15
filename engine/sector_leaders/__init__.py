"""
Sector Leaders — daily relative-strength ranking of the 11 S&P 500 sector SPDR
ETFs vs SPY, with a one-line offense/defense "tape character" read. Standalone
(separate from market_pulse and the signal engine). Free tier.
"""
from . import compute, config, data, store  # noqa: F401
from .job import run_backfill, run_daily  # noqa: F401

__all__ = ["run_daily", "run_backfill", "store", "config", "compute"]
