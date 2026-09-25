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


# ── exhaustion / extension take-profit (OBSERVE-ONLY, shadow) ─────────────────
# Ground-truth (2026-09): momentum tops ARE parabolic stretches — the names that ran
# huge then reversed (MRVL, ARM) PEAKED 4.8-5.7×ATR above their 20-EMA. A close-based
# trend/trail exit can never bank that (it only confirms the top AFTER price comes off
# it → ~40% MFE capture ceiling). A profit-target into strength CAN: it is a LIMIT, not
# a stop, so it fills on the spike and cannot be wicked out — checking it intraday is
# safe. This is measured in shadow before it ever manages a live cent.
def extension_atr(direction: str, daily_df: pd.DataFrame,
                  ema_span: int = 20, atr_len: int = 14) -> Optional[float]:
    """How far the LAST close is stretched beyond its mean, in ATR units, signed so a
    large POSITIVE value = extended in the trade's favour (= exhaustion risk). None if
    insufficient data. Pure — for observation only."""
    d = _norm(daily_df)
    if d is None or len(d) < max(ema_span, atr_len) + 1:
        return None
    close, high, low = d["close"], d["high"], d["low"]
    ema = float(close.ewm(span=ema_span, adjust=False).mean().iloc[-1])
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(atr_len).mean().iloc[-1])
    if not atr or np.isnan(atr):
        return None
    ext = (float(close.iloc[-1]) - ema) / atr
    return round(ext if direction.upper() == "LONG" else -ext, 2)


def replay_ext_tp(direction: str, entry: float, stop: float, daily_df: pd.DataFrame, *,
                  entry_date=None, spy_df: Optional[pd.DataFrame] = None,
                  cfg: Optional[ExitConfig] = None, k: float = 5.0, scale: float = 0.5,
                  arm: Optional[float] = None, ema_span: int = 20, atr_len: int = 14,
                  max_hold: int = 25) -> Optional[dict]:
    """HYBRID exit replay: sell `scale` of the position with an INTRADAY limit into an
    exhaustion stretch (≥ k×ATR beyond the ema_span mean), let the remainder ride the
    normal smart-exit lifecycle (`replay`). Blended pnl = scale·(exhaustion fill) +
    (1-scale)·(trail exit). If the stretch is never reached, pnl == the trail.

    An exhaustion take-profit is a PROFIT tool: it only fires (1) after a real run-up
    (running peak gain ≥ `arm`, default the giveback arm) and (2) when the fill is in
    profit (level beyond entry). Without those guards a bounce toward a FALLING mean in a
    losing trade would 'take profit' at a big loss. Pure + deterministic; OBSERVE-ONLY."""
    cfg = cfg or config_from_env()
    arm = cfg.giveback_arm_pct if arm is None else arm
    base = replay(direction, entry, stop, daily_df, entry_date=entry_date,
                  spy_df=spy_df, cfg=cfg, max_hold=max_hold)
    if base is None:
        return None
    cols = {c.lower(): c for c in daily_df.columns}
    if not all(x in cols for x in ("high", "low", "close")):
        return None
    hi = pd.to_numeric(daily_df[cols["high"]], errors="coerce")
    lo = pd.to_numeric(daily_df[cols["low"]], errors="coerce")
    cl = pd.to_numeric(daily_df[cols["close"]], errors="coerce")
    is_long = direction.upper() == "LONG"
    ema = cl.ewm(span=ema_span, adjust=False).mean()
    tr = pd.concat([hi - lo, (hi - cl.shift()).abs(), (lo - cl.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(atr_len).mean()

    start = None
    if entry_date is not None and hasattr(daily_df.index, "date"):
        for i in range(len(daily_df)):
            if daily_df.index[i].date() >= entry_date:
                start = i
                break
    if start is None:
        start = max(0, len(daily_df) - max_hold)

    peak_ext = None
    peak_price = entry
    ext_pnl = None
    ext_day = None
    for i in range(start, min(start + max_hold, len(daily_df))):
        hi_i, lo_i = float(hi.iloc[i]), float(lo.iloc[i])
        peak_price = max(peak_price, hi_i) if is_long else min(peak_price, lo_i)
        run = ((peak_price - entry) if is_long else (entry - peak_price)) / entry * 100.0
        a = float(atr.iloc[i]) if not pd.isna(atr.iloc[i]) else None
        if not a or a <= 0:
            continue
        e_i = float(ema.iloc[i])
        stretch = (hi_i - e_i) / a if is_long else (e_i - lo_i) / a
        peak_ext = stretch if peak_ext is None else max(peak_ext, stretch)
        if ext_pnl is None and run >= arm:                 # only after a genuine run-up
            lvl = e_i + k * a if is_long else e_i - k * a
            hit = (hi_i >= lvl) if is_long else (lo_i <= lvl)
            if hit:
                g = ((lvl - entry) if is_long else (entry - lvl)) / entry * 100.0
                if g > 0:                                   # take PROFIT into strength only
                    ext_pnl = g
                    ext_day = i - start
    ext_hit = ext_pnl is not None
    blended = (scale * ext_pnl + (1 - scale) * base["pnl_pct"]) if ext_hit else base["pnl_pct"]
    return {"pnl_pct": round(blended, 3), "ext_hit": ext_hit,
            "ext_pnl": round(ext_pnl, 3) if ext_hit else None, "ext_day": ext_day,
            "peak_ext": round(peak_ext, 2) if peak_ext is not None else None,
            "trail_pnl": base["pnl_pct"], "trail_reason": base["exit_reason"],
            "k": k, "scale": scale, "v": VERSION}
