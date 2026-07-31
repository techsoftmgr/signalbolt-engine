"""
exit_engine.py — multi-factor CONFLUENCE exit for open swing / momentum positions.

The problem it solves
---------------------
Single-indicator exits (one bearish CMF cross, an RSI peak, a lone CHoCH) whipsaw
us out right before the trend resumes: we exit, the stock keeps going our way, and
realized return shrinks. Ground-truth MFE analysis (2026-07) showed we capture only
~10-15% of the favorable move — good runs round-tripping to scratch or a loss
(AAPL ran +10.6% → −3%; TTD +7.3% → flat). The fix has two independent legs:

  1. GIVEBACK CAP (profit protection — trend-agnostic).
     Once peak favorable excursion ≥ ARM% (default 5%), never let the position give
     back below a locked floor = peak_gain × KEEP (default 0.5). A trailing profit
     stop that only arms after a real run. Protects the AAPL/TTD round-trip case
     regardless of what the trend indicators say.

  2. CONFLUENCE TREND-EXIT ("are we still in the trade?").
     Scored on the last COMPLETED daily bar only (no intraday wicks — the trend_ride
     lesson). A PRICE ANCHOR is REQUIRED: a daily close through the 20-EMA, or a
     confirmed structure break (CHoCH) against the position. Momentum / money-flow
     signals (RSI rollover, MACD, CMF, RS-vs-SPY) only CORROBORATE — they can NEVER
     trigger an exit on their own. Exit only when: anchor present AND corroboration
     score ≥ THRESHOLD.

Default action is HOLD (let winners run). Pure + deterministic — no I/O, no DB, no
network — so it is unit-testable and can be backtested bar-by-bar over history. The
monitor owns the hard stop, the target, and all DB writes; this only answers
"hold, ratchet, or exit — and why."
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# Bump when the logic/defaults change so cohorts are distinguishable in the data.
VERSION = 1


def _flag(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


# ── kill switch (matches trend_ride.enabled pattern) ──────────────────────────
def enabled() -> bool:
    return _flag("SMART_EXIT_ENABLED")


def shadow() -> bool:
    """Shadow mode — evaluate + record what we WOULD do, but DON'T act (old logic
    keeps running). Captures forward data on the new exit with zero live risk."""
    return _flag("SMART_EXIT_SHADOW")


def manages(detector_source: Optional[str]) -> bool:
    """Per-detector allowlist so we can enable on the LOSERS first and leave the
    already-profitable detectors on their current (working) exit logic. Empty /
    unset = all swings. e.g. SMART_EXIT_DETECTORS=TREND_MOMENTUM,RS_PULLBACK."""
    allow = os.environ.get("SMART_EXIT_DETECTORS", "").strip()
    if not allow:
        return True
    want = {d.strip().upper() for d in allow.split(",") if d.strip()}
    return (detector_source or "SMC").upper() in want


@dataclass
class ExitConfig:
    giveback_arm_pct: float = 5.0     # arm the giveback cap once peak gain ≥ this %
    giveback_keep: float = 0.5        # lock this fraction of the peak gain
    confluence_threshold: float = 3.0 # corroboration points required (WITH an anchor)
    ema_fast: int = 10
    ema_slow: int = 20
    rsi_len: int = 14
    rsi_peak: float = 65.0            # RSI must have reached ≥ this (lookback) …
    rsi_rolled: float = 55.0          # … and now close < this to count a rollover
    rsi_lookback: int = 8
    cmf_bear: float = -0.05
    vol_len: int = 20


DEFAULT = ExitConfig()


# ── small, self-contained indicator helpers (work on any OHLCV daily frame) ───
def _ema(s: pd.Series, span: int) -> Optional[float]:
    if len(s) < span:
        return None
    return float(s.ewm(span=span, adjust=False).mean().iloc[-1])


def _rsi_series(close: pd.Series, length: int = 14) -> Optional[pd.Series]:
    if len(close) < length + 2:
        return None
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    rs = up.ewm(alpha=1 / length, adjust=False).mean() / dn.ewm(alpha=1 / length, adjust=False).mean().replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def _macd_hist(close: pd.Series) -> Optional[pd.Series]:
    if len(close) < 35:
        return None
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    return macd - macd.ewm(span=9, adjust=False).mean()


def _cmf(df: pd.DataFrame, period: int = 20) -> Optional[float]:
    if len(df) < period:
        return None
    hi, lo, cl, vol = df["high"], df["low"], df["close"], df["volume"]
    rng = (hi - lo).replace(0, np.nan)
    mfv = (((cl - lo) - (hi - cl)) / rng * vol).fillna(0.0)
    denom = vol.rolling(period).sum().iloc[-1]
    if not denom:
        return None
    return float(mfv.rolling(period).sum().iloc[-1] / denom)


def _swing_low(low: pd.Series, lookback: int = 10, ignore_last: int = 1) -> Optional[float]:
    seg = low.iloc[-(lookback + ignore_last):-ignore_last] if ignore_last else low.iloc[-lookback:]
    return float(seg.min()) if len(seg) else None


def _swing_high(high: pd.Series, lookback: int = 10, ignore_last: int = 1) -> Optional[float]:
    seg = high.iloc[-(lookback + ignore_last):-ignore_last] if ignore_last else high.iloc[-lookback:]
    return float(seg.max()) if len(seg) else None


def _norm(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Lower-case OHLCV columns; tolerate yfinance/Alpaca casing. Drop today's row
    is the caller's job (we read the frame as given — pass COMPLETED bars)."""
    if df is None or len(df) < 25:
        return None
    cols = {c.lower(): c for c in df.columns}
    need = ("open", "high", "low", "close", "volume")
    if not all(k in cols for k in need):
        return None
    return pd.DataFrame({k: pd.to_numeric(df[cols[k]], errors="coerce") for k in need}).dropna()


def giveback_floor(direction: str, entry: float, peak: float, cfg: ExitConfig = DEFAULT) -> Optional[float]:
    """Price level below/above which a run ≥ ARM% is considered 'given back'. None
    until the position has actually run ≥ ARM%. This is a trailing PROFIT stop."""
    is_long = direction.upper() == "LONG"
    peak_gain = ((peak - entry) if is_long else (entry - peak)) / entry * 100.0
    if peak_gain < cfg.giveback_arm_pct:
        return None
    lock = peak_gain * cfg.giveback_keep                       # % gain to protect
    return entry * (1 + lock / 100.0) if is_long else entry * (1 - lock / 100.0)


def evaluate(direction: str, entry: float, price: float, peak: float,
             daily_df: pd.DataFrame, *, spy_df: Optional[pd.DataFrame] = None,
             cfg: ExitConfig = DEFAULT) -> dict:
    """
    Decide HOLD / TRAIL / EXIT for one open position.

    direction  'LONG' | 'SHORT'
    entry      fill price
    price      current price (live)
    peak       best price seen since entry (MFE anchor for the giveback cap)
    daily_df   OHLCV daily bars, COMPLETED (caller drops today's forming bar)
    spy_df     optional SPY daily bars for relative-strength corroboration

    Returns dict: action, reason, score, anchor, signals{}, giveback_floor, exit.
    """
    is_long = direction.upper() == "LONG"
    out = {"action": "hold", "reason": "hold", "score": 0.0, "anchor": False,
           "signals": {}, "giveback_floor": None, "exit": False}

    gb = giveback_floor(direction, entry, peak, cfg)
    out["giveback_floor"] = gb

    # ── 1) GIVEBACK CAP — profit protection, independent of the trend read ────
    if gb is not None:
        breached = (price <= gb) if is_long else (price >= gb)
        if breached:
            peak_gain = ((peak - entry) if is_long else (entry - peak)) / entry * 100.0
            out.update(action="exit", exit=True, reason="giveback_cap",
                       signals={"giveback_cap": True, "peak_gain_pct": round(peak_gain, 2)})
            return out

    # ── 2) CONFLUENCE TREND-EXIT — price anchor REQUIRED, momentum corroborates ─
    d = _norm(daily_df)
    if d is None:
        return out  # not enough data → HOLD (fail-safe)
    close, high, low, vol = d["close"], d["high"], d["low"], d["volume"]
    c = float(close.iloc[-1])

    ema10 = _ema(close, cfg.ema_fast)
    ema20 = _ema(close, cfg.ema_slow)
    sig: dict = {}
    score = 0.0

    def against(a: float, b: float) -> bool:               # a is 'below b' bearish for LONG
        return (a < b) if is_long else (a > b)

    # --- ANCHOR (price): a daily CLOSE through the 20-EMA, or a structure break ---
    anchor = False
    if ema20 is not None and against(c, ema20):
        anchor = True
        sig["close_vs_ema20"] = True
        score += 1.5
    if ema10 is not None and against(c, ema10):
        sig["close_vs_ema10"] = True
        score += 1.0

    # structure break: close through the prior swing low (LONG) / high (SHORT)
    sw = _swing_low(low) if is_long else _swing_high(high)
    if sw is not None and against(c, sw):
        anchor = True
        sig["structure_break"] = True
        score += 1.5

    # --- CORROBORATION (momentum / flow) — cannot exit alone; only add to score ---
    rsi = _rsi_series(close, cfg.rsi_len)
    if rsi is not None:
        recent = rsi.iloc[-cfg.rsi_lookback:]
        peaked = (recent.max() >= cfg.rsi_peak) if is_long else (recent.min() <= (100 - cfg.rsi_peak))
        rolled = (float(rsi.iloc[-1]) < cfg.rsi_rolled) if is_long else (float(rsi.iloc[-1]) > (100 - cfg.rsi_rolled))
        if peaked and rolled:
            sig["rsi_rollover"] = round(float(rsi.iloc[-1]), 1)
            score += 1.0

    hist = _macd_hist(close)
    if hist is not None and len(hist) >= 2:
        h0, h1 = float(hist.iloc[-1]), float(hist.iloc[-2])
        bearish = (h0 < 0 and h0 < h1) if is_long else (h0 > 0 and h0 > h1)
        if bearish:
            sig["macd_falling"] = round(h0, 3)
            score += 1.0

    cmf = _cmf(d)
    if cmf is not None:
        bad = (cmf < cfg.cmf_bear) if is_long else (cmf > -cfg.cmf_bear)
        if bad:
            sig["cmf"] = round(cmf, 3)
            score += 1.0

    # distribution / accumulation day: last bar closes against us on rising volume
    if len(d) >= cfg.vol_len + 1:
        vavg = float(vol.iloc[-(cfg.vol_len + 1):-1].mean())
        down_bar = (close.iloc[-1] < close.iloc[-2]) if is_long else (close.iloc[-1] > close.iloc[-2])
        if down_bar and vavg and float(vol.iloc[-1]) > 1.2 * vavg:
            sig["distribution_day"] = True
            score += 0.5

    # relative strength vs SPY rolling over (5-day)
    if spy_df is not None:
        sd = _norm(spy_df)
        if sd is not None and len(sd) >= 6 and len(close) >= 6:
            stock_ret = c / float(close.iloc[-6]) - 1
            spy_ret = float(sd["close"].iloc[-1]) / float(sd["close"].iloc[-6]) - 1
            underperf = (stock_ret < spy_ret) if is_long else (stock_ret > spy_ret)
            if underperf:
                sig["rs_rollover"] = True
                score += 0.5

    out["signals"] = sig
    out["score"] = round(score, 2)
    out["anchor"] = anchor

    # EXIT only with an anchor AND enough corroboration. No anchor → HOLD, no matter
    # how bearish momentum/flow looks (this is what stops the single-metric whipsaw).
    if anchor and score >= cfg.confluence_threshold:
        out.update(action="exit", exit=True, reason="confluence_break")
        return out

    # Still in the trade → ratchet the trailing stop to the giveback floor if armed.
    if gb is not None:
        out["action"] = "trail"
    return out


def replay(direction: str, entry: float, stop: float, daily_df: pd.DataFrame, *,
           entry_date=None, spy_df: Optional[pd.DataFrame] = None,
           cfg: Optional[ExitConfig] = None, max_hold: int = 25) -> Optional[dict]:
    """Replay the FULL smart-exit lifecycle over daily bars from entry — the original
    hard stop kept as the backstop, the upside managed by giveback cap + confluence.
    Because it walks forward from entry over price history that extends PAST where the
    real trade closed, this measures 'would have held longer / capped the giveback'
    AFTER the actual trade is already closed (the case the in-loop shadow can't see).

    daily_df : OHLCV daily bars with a DatetimeIndex; must include ~40+ bars BEFORE
               entry (for the indicators) plus the forward bars to replay.
    Returns {pnl_pct, exit_reason, held_days, exit_price} or None (insufficient data).
    """
    cfg = cfg or config_from_env()
    if daily_df is None or len(daily_df) < 40:
        return None
    cols = {c.lower(): c for c in daily_df.columns}
    if not all(k in cols for k in ("high", "low", "close")):
        return None
    hi_c, lo_c, cl_c = cols["high"], cols["low"], cols["close"]
    is_long = direction.upper() == "LONG"

    # locate the entry bar (first bar on/after entry_date); else assume the last
    # ~max_hold bars are the trade window and history precedes it.
    start = None
    if entry_date is not None and hasattr(daily_df.index, "date"):
        for i in range(len(daily_df)):
            if daily_df.index[i].date() >= entry_date:
                start = i
                break
    if start is None:
        start = max(0, len(daily_df) - max_hold)
    if start < 30:                              # not enough history before entry for indicators
        return None

    peak = entry
    for i in range(start, min(start + max_hold, len(daily_df))):
        hi = float(daily_df[hi_c].iloc[i]); lo = float(daily_df[lo_c].iloc[i]); cl = float(daily_df[cl_c].iloc[i])
        peak = max(peak, hi) if is_long else min(peak, lo)
        # original hard stop first (intraday) — downside protection unchanged
        if (is_long and lo <= stop) or (not is_long and hi >= stop):
            px = stop
            return {"pnl_pct": round(((px - entry) if is_long else (entry - px)) / entry * 100, 3),
                    "exit_reason": "stop_hit", "held_days": i - start, "exit_price": round(px, 2)}
        hist = daily_df.iloc[:i + 1]
        sp = spy_df[spy_df.index <= daily_df.index[i]] if spy_df is not None else None
        r = evaluate(direction, entry, cl, peak, hist, spy_df=sp, cfg=cfg)
        if r["exit"]:
            px = r["giveback_floor"] if (r["reason"] == "giveback_cap" and r.get("giveback_floor")) else cl
            return {"pnl_pct": round(((px - entry) if is_long else (entry - px)) / entry * 100, 3),
                    "exit_reason": r["reason"], "held_days": i - start, "exit_price": round(px, 2)}
    # ran the whole window without an exit signal → mark the window-end close
    last = float(daily_df[cl_c].iloc[min(start + max_hold, len(daily_df)) - 1])
    return {"pnl_pct": round(((last - entry) if is_long else (entry - last)) / entry * 100, 3),
            "exit_reason": "window_end", "held_days": min(max_hold, len(daily_df) - start) - 1,
            "exit_price": round(last, 2)}


def config_from_env() -> ExitConfig:
    """Allow live tuning without a deploy (env overrides on the dataclass defaults)."""
    c = ExitConfig()
    g = os.environ.get
    try:
        c.giveback_arm_pct = float(g("SMART_EXIT_GIVEBACK_ARM", c.giveback_arm_pct))
        c.giveback_keep = float(g("SMART_EXIT_GIVEBACK_KEEP", c.giveback_keep))
        c.confluence_threshold = float(g("SMART_EXIT_THRESHOLD", c.confluence_threshold))
    except (TypeError, ValueError):
        pass
    return c
