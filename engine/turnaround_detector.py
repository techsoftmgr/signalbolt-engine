"""
Turnaround detector — the BOTTOM half of the cycle/swing feature.

Catches swing-low reversals: a liquid name fell hard (20-50%+), and is now
turning back up. Buy near the turn, ride to the next peak.

Honest design (see memory: signalbolt-cycle-detector):
  • We do NOT try to catch the exact low. We score a high-probability turnaround
    ZONE, then only call it a BUY ZONE once a CONFIRMATION trigger prints
    (reclaim / CHoCH / bullish divergence) — so we don't catch a falling knife.
  • We explicitly skip mid-downtrend bear-flag shelves (the HOOD "$120 shelf"
    trap): a sideways shelf making lower-highs under a falling MA, with no
    capitulation and no upside confirmation, is CONTINUATION, not a reversal.

Five scored ingredients (→ 0-100):
  1. Regime / quality gate  (falling-knife filter; gates BUY ZONE)
  2. Oversold stretch        (RSI, distance below MA, drawdown band)
  3. Capitulation            (volume climax, down-streak, wide-range down bar)
  4. Confirmation trigger    (CHoCH up / reclaim prior high / bullish divergence
                              / reversal candle) — separates catch from knife
  5. Support confluence      (200-day / prior swing low / gamma wall)

Operates on DAILY bars. Pass ~1 year (>=200 rows) for a meaningful 200-day
trend gate + drawdown + structure; degrades gracefully with fewer. Reuses
engine.smc for market structure / CHoCH.

API:
  score_turnaround(df, *, regime_type=None, gamma_wall_below=None) -> dict | None
    stage ∈ {"none","watch","buyzone"}
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("signalbolt.turnaround")

# ── Tunables ─────────────────────────────────────────────────────────────────
_MIN_BARS          = 40
_RSI_OVERSOLD      = 35.0
_RSI_DEEP          = 25.0
_DD_MIN            = 15.0     # buyable-dip drawdown band (% off recent high)
_DD_MAX            = 45.0     # beyond this = knife territory (needs stronger proof)
_DD_LOOKBACK       = 60       # bars to measure the swing high for drawdown
_VOL_CLIMAX        = 1.75     # down-day volume vs 20d avg = capitulation
_SUPPORT_ATR       = 1.5      # within N ATR of a level = "at support"
_BEAR_REGIMES      = {"PANIC", "TRENDING_BEAR", "RISK_OFF"}

_WATCH_MIN_SCORE   = 45       # oversold zone, not yet confirmed
_BUYZONE_MIN_SCORE = 62       # confirmed reversal


# ── Small indicator helpers (no shared module exists in this codebase) ───────
def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    val = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean().iloc[-1]
    return float(val) if pd.notna(val) and val > 0 else 0.0


def _sma(s: pd.Series, n: int) -> Optional[float]:
    if len(s) < n:
        return None
    v = s.tail(n).mean()
    return float(v) if pd.notna(v) else None


# ── Main ─────────────────────────────────────────────────────────────────────
def score_turnaround(df: pd.DataFrame, *, regime_type: Optional[str] = None,
                     gamma_wall_below: Optional[float] = None) -> Optional[dict]:
    if df is None or len(df) < _MIN_BARS:
        return None
    try:
        df = df.sort_index()
        close = df["close"].astype(float)
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)
        vol   = df["volume"].astype(float)
        last  = float(close.iloc[-1])
        atr   = _atr(df)
        reasons: list[str] = []

        # ── 1. Regime / quality gate (falling-knife filter) ──────────────────
        ma200 = _sma(close, 200)
        ma50  = _sma(close, 50)
        hi52  = float(high.tail(min(len(df), 252)).max())
        above_200 = ma200 is not None and last > ma200
        near_high = hi52 > 0 and (hi52 - last) / hi52 <= 0.25
        regime_bear = (regime_type or "").upper() in _BEAR_REGIMES
        # falling 50-day MA = downtrend backdrop
        ma50_falling = (ma50 is not None and len(close) >= 60
                        and ma50 < (_sma(close.iloc[:-10], 50) or ma50))
        trend_ok = (above_200 or near_high) and not regime_bear
        if trend_ok:
            reasons.append("uptrend/quality intact" if above_200 else "near 52w high")
        elif regime_bear:
            reasons.append(f"counter-trend ({regime_type})")

        # ── 2. Oversold stretch (0-35) ───────────────────────────────────────
        rsi_series = _rsi(close)
        rsi = float(rsi_series.iloc[-1])
        ma20 = _sma(close, 20) or last
        dist_below_ma = (ma20 - last) / ma20 * 100 if ma20 else 0.0
        peak = float(high.tail(_DD_LOOKBACK).max())
        drawdown = (peak - last) / peak * 100 if peak > 0 else 0.0

        oversold_pts = 0.0
        if rsi <= _RSI_DEEP:
            oversold_pts += 18; reasons.append(f"RSI {rsi:.0f} (deeply oversold)")
        elif rsi <= _RSI_OVERSOLD:
            oversold_pts += 12; reasons.append(f"RSI {rsi:.0f} (oversold)")
        if dist_below_ma >= 3:
            oversold_pts += min(9, dist_below_ma)
        if _DD_MIN <= drawdown <= _DD_MAX:
            oversold_pts += 8; reasons.append(f"-{drawdown:.0f}% from recent high (buyable band)")
        elif drawdown > _DD_MAX:
            oversold_pts += 3; reasons.append(f"-{drawdown:.0f}% drawdown (deep — knife risk)")
        oversold_pts = min(35.0, oversold_pts)

        # ── 3. Capitulation (0-25) ───────────────────────────────────────────
        vol20 = float(vol.tail(20).mean()) if len(vol) >= 20 else float(vol.mean())
        down_day = close.iloc[-1] < close.iloc[-2]
        # The capitulation climax is often the 1-2 bars JUST BEFORE the reclaim,
        # not the last bar — scan the recent window.
        vol_climax = False
        for i in range(1, min(4, len(close))):
            if (close.iloc[-i] < close.iloc[-i - 1] and vol20 > 0
                    and float(vol.iloc[-i]) >= _VOL_CLIMAX * vol20):
                vol_climax = True
                break
        down_streak = 0
        for i in range(1, min(8, len(close))):
            if close.iloc[-i] < close.iloc[-i - 1]:
                down_streak += 1
            else:
                break
        rng = float(high.iloc[-1] - low.iloc[-1])
        wide_down = down_day and atr > 0 and rng >= 1.5 * atr

        capitulation_pts = 0.0
        if vol_climax:
            capitulation_pts += 13; reasons.append("volume climax (capitulation)")
        if down_streak >= 3:
            capitulation_pts += 7; reasons.append(f"{down_streak} down days")
        if wide_down:
            capitulation_pts += 5
        capitulation = capitulation_pts > 0
        capitulation_pts = min(25.0, capitulation_pts)

        # ── 4. Confirmation trigger (0-30) — catch vs knife ──────────────────
        confirm_pts = 0.0
        choch = False
        try:
            from engine import smc
            structure = smc.detect_structure(smc.detect_swings(df.copy()))
            choch = bool(structure.get("choch_bullish"))
        except Exception:
            choch = False
        if choch:
            confirm_pts += 14; reasons.append("CHoCH — structure turned up")

        # reclaim prior bar high after a down move
        reclaim = last > float(high.iloc[-2]) and down_streak == 0 and close.iloc[-1] > close.iloc[-2]
        if reclaim:
            confirm_pts += 8; reasons.append("reclaimed prior-day high")

        # bullish RSI divergence: price lower-low vs ~10-15 bars ago, RSI higher-low
        div = False
        if len(close) >= 16:
            p_now = float(low.iloc[-1]); p_prev = float(low.iloc[-15:-5].min())
            r_now = float(rsi_series.iloc[-1]); r_prev = float(rsi_series.iloc[-15:-5].min())
            if p_now < p_prev and r_now > r_prev and rsi <= 45:
                div = True; confirm_pts += 9; reasons.append("bullish RSI divergence")

        # bullish reversal candle (hammer / engulfing) at the low
        body = abs(close.iloc[-1] - df["open"].iloc[-1])
        lower_wick = min(close.iloc[-1], df["open"].iloc[-1]) - low.iloc[-1]
        hammer = atr > 0 and lower_wick >= 1.5 * body and close.iloc[-1] >= df["open"].iloc[-1]
        if hammer:
            confirm_pts += 6; reasons.append("hammer/rejection candle")
        confirm_pts = min(30.0, confirm_pts)
        confirmed = confirm_pts >= 8  # at least one real trigger

        # ── 5. Support confluence (0-15) ─────────────────────────────────────
        support_pts = 0.0
        if ma200 is not None and atr > 0 and abs(last - ma200) <= _SUPPORT_ATR * atr:
            support_pts += 6; reasons.append("at 200-day support")
        prior_low = float(low.iloc[:-3].tail(_DD_LOOKBACK).min()) if len(low) > _DD_LOOKBACK else float(low.min())
        if atr > 0 and abs(last - prior_low) <= _SUPPORT_ATR * atr:
            support_pts += 6; reasons.append("at prior swing low (double-bottom)")
        if gamma_wall_below and atr > 0 and abs(last - gamma_wall_below) <= _SUPPORT_ATR * atr:
            support_pts += 4; reasons.append("at gamma support wall")
        at_support = support_pts > 0
        support_pts = min(15.0, support_pts)

        score = round(min(100.0, oversold_pts + capitulation_pts + confirm_pts + support_pts))

        # ── Staircase / bear-flag guard ──────────────────────────────────────
        # A sideways shelf making lower-highs under a FALLING MA, with no
        # capitulation and no upside confirmation = mid-downtrend continuation
        # (the HOOD "$120 shelf"). Never a BUY ZONE.
        # In a confirmed downtrend, only a real CHoCH or a capitulation flush
        # lifts the block — a lone divergence/hammer is NOT enough to buy into a
        # staircase shelf (the HOOD "$120 shelf").
        downtrend = (ma50_falling and not above_200) or regime_bear
        staircase_blocked = downtrend and not choch and not capitulation
        if staircase_blocked:
            reasons.append("bear-flag shelf (continuation) — blocked")

        # ── Stage ────────────────────────────────────────────────────────────
        oversold_enough = oversold_pts >= 12
        if (confirmed and oversold_enough and score >= _BUYZONE_MIN_SCORE
                and (trend_ok or support_pts >= 6) and not staircase_blocked):
            stage = "buyzone"
        elif oversold_enough and (capitulation or at_support) and score >= _WATCH_MIN_SCORE:
            stage = "watch"
        else:
            stage = "none"

        return {
            "score":            score,
            "stage":            stage,
            "rsi":              round(rsi, 1),
            "drawdownPct":      round(drawdown, 1),
            "distBelowMaPct":   round(dist_below_ma, 1),
            "relVolLastDay":    round(float(vol.iloc[-1]) / vol20, 2) if vol20 else None,
            "downStreak":       down_streak,
            "trendOk":          bool(trend_ok),
            "capitulation":     bool(capitulation),
            "confirmed":        bool(confirmed),
            "choch":            bool(choch),
            "atSupport":        bool(at_support),
            "staircaseBlocked": bool(staircase_blocked),
            "components": {
                "oversold":      round(oversold_pts, 1),
                "capitulation":  round(capitulation_pts, 1),
                "confirmation":  round(confirm_pts, 1),
                "support":       round(support_pts, 1),
            },
            "reasons": reasons,
        }
    except Exception as e:
        logger.debug(f"[turnaround] score failed: {e}")
        return None
