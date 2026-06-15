"""
Sector Leaders — pure RS computation over already-fetched daily bars.

bars: {symbol: DataFrame(open,high,low,close,volume)} including SPY. All functions
are defensive (missing/short data → that sector is skipped) and side-effect-free.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from . import config as C


def _ret(df: Optional[pd.DataFrame], n: int) -> Optional[float]:
    """Trailing return over n trading days (fraction)."""
    if df is None or len(df) <= n:
        return None
    a = float(df["close"].iloc[-1]); b = float(df["close"].iloc[-1 - n])
    return (a / b - 1.0) if b > 0 else None


def relative_strength(sector: Optional[pd.DataFrame], spy: Optional[pd.DataFrame], n: int) -> Optional[float]:
    """Sector trailing return minus SPY's, over n days, in percent."""
    sr, br = _ret(sector, n), _ret(spy, n)
    if sr is None or br is None:
        return None
    return round((sr - br) * 100, 2)


def _blended_map(bars: dict[str, pd.DataFrame], offset: int = 0) -> dict[str, float]:
    """{etf: rs_blended} as of `offset` trading days ago (offset=0 → today)."""
    spy = bars.get(C.BENCHMARK)
    if spy is None:
        return {}
    spy_s = spy.iloc[: len(spy) - offset] if offset else spy
    out: dict[str, float] = {}
    for etf in C.ETFS:
        df = bars.get(etf)
        if df is None:
            continue
        s = df.iloc[: len(df) - offset] if offset else df
        r1 = relative_strength(s, spy_s, C.L_1M)
        r3 = relative_strength(s, spy_s, C.L_3M)
        r6 = relative_strength(s, spy_s, C.L_6M)
        if r1 is None or r3 is None or r6 is None:
            continue
        out[etf] = round(C.W_1M * r1 + C.W_3M * r3 + C.W_6M * r6, 3)
    return out


def rank_map(blended: dict[str, float]) -> dict[str, int]:
    """{etf: rank} with 1 = strongest blended RS."""
    order = sorted(blended.items(), key=lambda kv: -kv[1])
    return {etf: i + 1 for i, (etf, _v) in enumerate(order)}


def _above_50d(df: Optional[pd.DataFrame]) -> Optional[bool]:
    if df is None or len(df) < C.SMA_POSTURE:
        return None
    return bool(float(df["close"].iloc[-1]) > float(df["close"].tail(C.SMA_POSTURE).mean()))


def tape_character(top3: list[str]) -> str:
    off = sum(1 for e in top3 if e in C.OFFENSE)
    deff = sum(1 for e in top3 if e in C.DEFENSE)
    if off >= 2:
        return C.OFFENSE_LED
    if deff >= 2:
        return C.DEFENSE_LED
    return C.ROTATING


def compute(bars: dict[str, pd.DataFrame]) -> tuple[list[dict], dict]:
    """Return (per-sector rows, summary). Empty if SPY/data missing."""
    blended_now = _blended_map(bars, 0)
    if not blended_now:
        return [], {}
    blended_5d = _blended_map(bars, C.RANK_MOM_LOOKBACK)
    rank_now = rank_map(blended_now)
    rank_5d = rank_map(blended_5d) if blended_5d else {}

    rows: list[dict] = []
    for etf in C.ETFS:
        if etf not in blended_now:
            continue
        df = bars.get(etf)
        spy = bars.get(C.BENCHMARK)
        r_now = rank_now.get(etf)
        r_5d = rank_5d.get(etf)
        if r_5d is None:
            mom = "FLAT"
        elif r_now < r_5d:
            mom = "IMPROVING"
        elif r_now > r_5d:
            mom = "DETERIORATING"
        else:
            mom = "FLAT"
        rows.append({
            "sector_etf": etf,
            "rs_1m": relative_strength(df, spy, C.L_1M),
            "rs_3m": relative_strength(df, spy, C.L_3M),
            "rs_6m": relative_strength(df, spy, C.L_6M),
            "rs_blended": blended_now[etf],
            "rs_rank": r_now,
            "rs_rank_5d_ago": r_5d,
            "rank_momentum": mom,
            "above_50d": _above_50d(df),
            "tilt": C.tilt_of(etf),
        })

    rows.sort(key=lambda r: r["rs_rank"])
    top3 = [r["sector_etf"] for r in rows[:3]]
    tc = tape_character(top3)
    summary = {"tape_character": tc, "top3": top3, "guidance_key": tc}
    return rows, summary
