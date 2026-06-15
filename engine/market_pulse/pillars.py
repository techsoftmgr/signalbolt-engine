"""
Market Pulse — the five pillar calculators, as PURE functions over already-fetched
data (pandas DataFrames). No network here → fully unit-testable. All are defensive:
missing / short data yields a neutral/empty result, never an exception.

Bar DataFrames are date-indexed ascending with lowercase columns
(open, high, low, close, volume) — the shape engine.alpaca_client returns.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from . import config as C


# ── Pillar 1: distribution-day count (one index at a time) ──────────────────
def distribution_days(
    df: Optional[pd.DataFrame],
    window: int = C.DD_WINDOW,
    down_pct: float = C.DD_DOWN_PCT,
    expire_rise: float = C.DD_EXPIRE_RISE,
) -> int:
    """Count distribution days in the trailing `window` trading days.

    A distribution day = close down >= down_pct vs the prior close, on HIGHER
    volume than the prior day. It EXPIRES (stops counting) when EITHER it falls
    out of the window OR the index later closes `expire_rise` (5%) above that
    day's close. Both rules implemented.
    """
    if df is None or len(df) < 2:
        return 0
    sub = df.tail(window + 1)               # window days + 1 prior for the first comparison
    closes = sub["close"].to_numpy(dtype=float)
    vols = sub["volume"].to_numpy(dtype=float)
    n = len(sub)
    count = 0
    start = max(1, n - window)
    for i in range(start, n):
        is_dd = closes[i] <= closes[i - 1] * (1 - down_pct) and vols[i] > vols[i - 1]
        if not is_dd:
            continue
        # Expired if any LATER close (through today) is >= 5% above this day's close.
        ceiling = closes[i] * (1 + expire_rise)
        if any(closes[j] >= ceiling for j in range(i + 1, n)):
            continue
        count += 1
    return count


def stalling_days(
    df: Optional[pd.DataFrame],
    window: int = C.DD_WINDOW,
    max_gain: float = C.STALL_MAX_GAIN_PCT,
    close_frac: float = C.STALL_CLOSE_RANGE_FRAC,
    expire_rise: float = C.DD_EXPIRE_RISE,
) -> int:
    """Count STALLING days in the trailing `window` — a softer form of distribution:
    close UP vs prior but a tiny gain (<= max_gain), on HIGHER volume, and closing
    in the lower half of the day's range (institutions selling into strength).
    Same 25-day window + 5%-rise expiration as distribution days."""
    if df is None or len(df) < 2:
        return 0
    sub = df.tail(window + 1)
    c = sub["close"].to_numpy(dtype=float)
    v = sub["volume"].to_numpy(dtype=float)
    hi = sub["high"].to_numpy(dtype=float)
    lo = sub["low"].to_numpy(dtype=float)
    n = len(sub)
    count = 0
    start = max(1, n - window)
    for i in range(start, n):
        if c[i - 1] <= 0:
            continue
        gain = c[i] / c[i - 1] - 1.0
        rng = hi[i] - lo[i]
        close_pos = (c[i] - lo[i]) / rng if rng > 0 else 0.0
        is_stall = (0 < gain <= max_gain) and (v[i] > v[i - 1]) and (close_pos <= close_frac)
        if not is_stall:
            continue
        ceiling = c[i] * (1 + expire_rise)
        if any(c[j] >= ceiling for j in range(i + 1, n)):
            continue
        count += 1
    return count


# ── Pillar 2: net new highs vs new lows (S&P 500) ───────────────────────────
def net_new_highs_lows(bars: dict[str, pd.DataFrame], lookback: int = C.HL_LOOKBACK) -> tuple[int, int, int]:
    """(new_highs, new_lows, net). A name makes a 52-week high when today's HIGH
    is the max of the trailing `lookback` highs (low symmetric). Names without a
    full year of history are skipped (can't confirm a 52-week extreme)."""
    new_highs = new_lows = 0
    for df in bars.values():
        if df is None or len(df) < lookback:
            continue
        window = df.tail(lookback)
        hi = window["high"]; lo = window["low"]
        # A NEW high = today exceeds every PRIOR day in the window (strict, so a
        # flat series that merely ties its own level is not a new high).
        if float(hi.iloc[-1]) > float(hi.iloc[:-1].max()):
            new_highs += 1
        elif float(lo.iloc[-1]) < float(lo.iloc[:-1].min()):
            new_lows += 1
    return new_highs, new_lows, new_highs - new_lows


# ── Pillar 3: % above moving averages (S&P 500) ─────────────────────────────
def pct_above_mas(bars: dict[str, pd.DataFrame], fast: int = C.SMA_FAST, slow: int = C.SMA_SLOW) -> tuple[float, float]:
    """(% above `fast`-day SMA, % above `slow`-day SMA). Each pct is over the names
    with enough history for that SMA (denominators can differ slightly)."""
    above_fast = total_fast = 0
    above_slow = total_slow = 0
    for df in bars.values():
        if df is None or len(df) < fast:
            continue
        close = df["close"]
        last = float(close.iloc[-1])
        total_fast += 1
        if last > float(close.tail(fast).mean()):
            above_fast += 1
        if len(df) >= slow:
            total_slow += 1
            if last > float(close.tail(slow).mean()):
                above_slow += 1
    pct_fast = round(100.0 * above_fast / total_fast, 2) if total_fast else 0.0
    pct_slow = round(100.0 * above_slow / total_slow, 2) if total_slow else 0.0
    return pct_fast, pct_slow


# ── Pillar 4: advance / decline ─────────────────────────────────────────────
def advance_decline(bars: dict[str, pd.DataFrame]) -> tuple[int, int, int]:
    """(advancers, decliners, net) — names up vs down on the latest close vs the
    prior close. `net` is added to the running cumulative A/D line by the caller."""
    adv = dec = 0
    for df in bars.values():
        if df is None or len(df) < 2:
            continue
        c = df["close"]
        last = float(c.iloc[-1]); prev = float(c.iloc[-2])
        if last > prev:
            adv += 1
        elif last < prev:
            dec += 1
    return adv, dec, adv - dec


def ad_divergence(
    spy_df: Optional[pd.DataFrame],
    ad_cumulative_today: int,
    ad_cumulative_history: list[int],
    near_high_pct: float = C.AD_NEAR_HIGH_PCT,
) -> bool:
    """TRUE when SPY is within `near_high_pct` of its 52-week high while the
    cumulative A/D line is NOT making a new high over the same window — the classic
    breadth-divergence warning. Needs A/D history; returns False if undeterminable."""
    if spy_df is None or len(spy_df) < C.HL_LOOKBACK or not ad_cumulative_history:
        return False
    window = spy_df.tail(C.HL_LOOKBACK)
    spy_last = float(window["close"].iloc[-1])
    spy_hi = float(window["high"].max())
    near_high = spy_hi > 0 and (spy_hi - spy_last) / spy_hi <= near_high_pct
    if not near_high:
        return False
    ad_new_high = ad_cumulative_today >= max(ad_cumulative_history + [ad_cumulative_today])
    return not ad_new_high


# ── Pillar 5: VIX level + trend (from a secondary source) ───────────────────
def vix_band(level: float) -> str:
    if level < C.VIX_CALM_MAX:
        return "CALM"
    if level < C.VIX_NORMAL_MAX:
        return "NORMAL"
    if level <= C.VIX_ELEVATED_MAX:
        return "ELEVATED"
    return "HIGH"


def vix_read(vix_closes: Optional[pd.Series], sma: int = C.VIX_SMA) -> Optional[dict]:
    """{close, sma10, rising, band} from a chronological series of VIX closes, or
    None if no data (the regime then computes from pillars 1-4 only)."""
    if vix_closes is None or len(vix_closes) < 1:
        return None
    try:
        s = pd.Series(vix_closes).dropna().astype(float)
        if s.empty:
            return None
        close = float(s.iloc[-1])
        sma10 = float(s.tail(sma).mean()) if len(s) >= 1 else close
        return {
            "close": round(close, 2),
            "sma10": round(sma10, 2),
            "rising": bool(close > sma10),
            "band": vix_band(close),
        }
    except Exception:
        return None
