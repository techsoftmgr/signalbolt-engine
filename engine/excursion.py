"""
excursion.py — Maximum Favorable / Adverse Excursion (MFE / MAE) per signal.

MFE = the best the trade got (peak favorable % from entry); MAE = the worst heat it
took (deepest adverse % from entry). Both in the trade's P&L frame — favorable is
POSITIVE for LONG and SHORT alike, adverse is NEGATIVE. Computed from daily HIGH/LOW
bars since entry so it captures intraday extremes, not just close-to-close. Pure +
deterministic.

Why track it: capture ratio (realized ÷ MFE) is the cleanest measure of exit
efficiency; MAE measures stop-placement quality; MFE isolates the ENTRY edge from the
exit. This is OUR data (entry-relative, per signal) — not available from a generic feed.
"""
from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd


def mfe_mae(direction: str, entry: float, daily_df: pd.DataFrame,
            entry_date=None) -> Optional[Tuple[float, float]]:
    """(mfe_pct, mae_pct) over the life of the trade from daily bars since entry.
    mfe_pct = best favorable % (≥ 0 once it moves the trade's way); mae_pct = worst
    adverse % (≤ 0). Returns None when there is no usable data."""
    if daily_df is None or len(daily_df) == 0 or not entry:
        return None
    cols = {c.lower(): c for c in daily_df.columns}
    if "high" not in cols or "low" not in cols:
        return None
    df = daily_df
    if entry_date is not None and hasattr(df.index, "date"):
        try:
            mask = [ix.date() >= entry_date for ix in df.index]
            if any(mask):
                df = df[mask]
        except Exception:
            df = daily_df
    hi = pd.to_numeric(df[cols["high"]], errors="coerce").max()
    lo = pd.to_numeric(df[cols["low"]], errors="coerce").min()
    if pd.isna(hi) or pd.isna(lo):
        return None
    hi, lo = float(hi), float(lo)
    if direction.upper() == "LONG":
        mfe = (hi - entry) / entry * 100.0
        mae = (lo - entry) / entry * 100.0
    else:  # SHORT — favorable is price DOWN
        mfe = (entry - lo) / entry * 100.0
        mae = (entry - hi) / entry * 100.0
    return round(mfe, 2), round(mae, 2)


def merge(prev_mfe: Optional[float], prev_mae: Optional[float],
          new_mfe: float, new_mae: float) -> Tuple[float, float]:
    """Monotonic running update — MFE only ratchets UP, MAE only DOWN, so a later
    fetch with a shorter bar window can never erase a prior extreme."""
    mfe = new_mfe if prev_mfe is None else max(prev_mfe, new_mfe)
    mae = new_mae if prev_mae is None else min(prev_mae, new_mae)
    return round(mfe, 2), round(mae, 2)
