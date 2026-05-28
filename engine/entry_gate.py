"""
Entry Gate v2 — multi-timeframe + pattern confirmation gate.

Runs AFTER scoring passes and BEFORE SL/TP calculation. Rejects signals
that look statistically good on the entry timeframe but lack confluence
from higher / lower timeframes or print an obviously-bad entry candle.

Four sub-gates (all must pass):
  1. 15m trend filter   — EMA9 vs EMA21 must agree with signal direction
  2. 5m MACD filter     — histogram trending in signal direction
  3. 1m reversal candle — last completed bar confirms direction
  4. Pattern rejectors  — 3-consecutive-opposite bars / overextended / volume drop

Each sub-gate failure is logged with a structured reason so a losing signal's
post-mortem can show which gates passed and which would have caught it.

If any required-timeframe fetch fails (data outage), the gate FAILS OPEN
(allows the signal) and logs a warning. The current engine without this gate
would have allowed it anyway, so failing-open is no worse than today.

Caveat: this module makes up to 3 Alpaca bar fetches per signal scored.
Cost is acceptable because it only runs AFTER scorer.passes, which is a
small fraction of tickers per cycle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from engine import alpaca_client

logger = logging.getLogger(__name__)


# ── Result type ──────────────────────────────────────────────────────────

@dataclass
class GateResult:
    allowed: bool
    reasons: list[str] = field(default_factory=list)   # human-readable failure reasons
    gate_log: dict     = field(default_factory=dict)   # per-gate {name: "pass"|"fail:reason"}


# ── Indicator helpers (kept tiny so the module has no extra deps) ───────

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd     = ema_fast - ema_slow
    sig      = _ema(macd, signal)
    return macd - sig


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    """ATR over last `period` bars. Returns 0.0 if insufficient data."""
    if df is None or len(df) < period + 1:
        return 0.0
    high  = df["high"].values[-(period + 1):].astype(float)
    low   = df["low"].values[-(period + 1):].astype(float)
    close = df["close"].values[-(period + 1):].astype(float)
    prev_close = close[:-1]
    tr = np.maximum.reduce([
        high[1:] - low[1:],
        np.abs(high[1:] - prev_close),
        np.abs(low[1:]  - prev_close),
    ])
    return float(np.mean(tr))


# ── Individual gates ─────────────────────────────────────────────────────

def _gate_15m_trend(ticker: str, direction: str, df_entry: pd.DataFrame, entry_tf: str) -> tuple[bool, str]:
    """Higher-timeframe trend must agree with signal direction (EMA9 vs EMA21 on 15m)."""
    # If the entry strategy is already on 15m or higher, reuse its df
    if entry_tf in ("15m", "15Min", "1h", "1Hour", "1d", "1Day") and df_entry is not None:
        df15 = df_entry
    else:
        df15 = alpaca_client.get_bars(ticker, timeframe="15Min", days=3)

    if df15 is None or len(df15) < 25:
        # fail open — don't punish missing data
        return True, "skipped (insufficient 15m bars)"

    close = df15["close"]
    ema9  = _ema(close, 9).iloc[-1]
    ema21 = _ema(close, 21).iloc[-1]

    if direction == "LONG" and ema9 > ema21:
        return True, f"15m trend up (ema9={ema9:.2f} > ema21={ema21:.2f})"
    if direction == "SHORT" and ema9 < ema21:
        return True, f"15m trend down (ema9={ema9:.2f} < ema21={ema21:.2f})"
    return False, f"15m trend against {direction} (ema9={ema9:.2f} vs ema21={ema21:.2f})"


def _gate_5m_macd(ticker: str, direction: str, df_entry: pd.DataFrame, entry_tf: str) -> tuple[bool, str]:
    """5m MACD histogram must lean in signal direction (last bar > 0 for LONG, < 0 for SHORT)."""
    if entry_tf in ("5m", "5Min") and df_entry is not None:
        df5 = df_entry
    else:
        df5 = alpaca_client.get_bars(ticker, timeframe="5Min", days=2)

    if df5 is None or len(df5) < 35:
        return True, "skipped (insufficient 5m bars)"

    hist = _macd_hist(df5["close"])
    h_now  = float(hist.iloc[-1])
    h_prev = float(hist.iloc[-2])

    if direction == "LONG":
        # Allow either: already positive, OR negative-but-rising (early reversal)
        if h_now > 0 or (h_now > h_prev and h_now > -0.05):
            return True, f"5m MACD hist={h_now:+.3f} (prev={h_prev:+.3f}) supports LONG"
        return False, f"5m MACD hist={h_now:+.3f} (prev={h_prev:+.3f}) against LONG"
    else:  # SHORT
        if h_now < 0 or (h_now < h_prev and h_now < 0.05):
            return True, f"5m MACD hist={h_now:+.3f} (prev={h_prev:+.3f}) supports SHORT"
        return False, f"5m MACD hist={h_now:+.3f} (prev={h_prev:+.3f}) against SHORT"


def _gate_1m_reversal(ticker: str, direction: str) -> tuple[bool, str]:
    """Last completed 1m bar must confirm direction (close > prev close for LONG)."""
    df1 = alpaca_client.get_bars(ticker, timeframe="1Min", days=1)
    if df1 is None or len(df1) < 3:
        return True, "skipped (insufficient 1m bars)"

    last_close = float(df1["close"].iloc[-1])
    prev_close = float(df1["close"].iloc[-2])

    if direction == "LONG" and last_close > prev_close:
        return True, f"1m reversal up ({prev_close:.2f} → {last_close:.2f})"
    if direction == "SHORT" and last_close < prev_close:
        return True, f"1m reversal down ({prev_close:.2f} → {last_close:.2f})"
    return False, f"1m no reversal for {direction} ({prev_close:.2f} → {last_close:.2f})"


def _gate_tape(ticker: str) -> tuple[bool, str, dict]:
    """
    Block signals on tickers with no real trade-tape activity. Catches:
      - Dead tape (no one is trading right now → setup is theoretical)
      - Illiquid window (cumulative volume too low to enter without slippage)

    Threshold rationale:
      - trades_per_sec < 0.3  → fewer than 1 trade per 3 seconds = dead
      - total_volume < 50k    → 5-min window thin enough to slip stops
    Stocks below these in real market hours are usually pre-market wakeups
    or low-float micros we shouldn't be firing on anyway.

    Returns (passed, reason, telemetry_dict) — telemetry stored in gate_log.
    """
    from engine import trade_tape
    summary = trade_tape.get_summary(ticker)
    if summary is None or summary.get("trades", 0) < 5:
        # No tape data → fail open (might be off-hours or engine cold-start)
        return True, "skipped (no tape data)", {}

    tps  = summary.get("trades_per_sec", 0.0)
    vol  = summary.get("total_volume", 0)
    tele = {
        "trades_per_sec": tps,
        "tape_volume":    vol,
        "block_count":    summary.get("block_count", 0),
    }

    if tps < 0.3:
        return False, f"dead tape ({tps:.2f}/sec — illiquid setup)", tele
    if vol < 50_000:
        return False, f"thin tape volume ({vol:,} shares in 5min)", tele
    return True, f"tape ok ({tps:.1f}/sec, {vol:,} shares)", tele


def _gate_spread(ticker: str) -> tuple[bool, str, Optional[float]]:
    """
    Block entries when bid-ask spread is too wide → slippage trap.
    Direct response to the Friday finding of 25-60% SL slippage.

    Threshold: 0.3% absolute. Tickers with naturally wide spreads (low-float
    biotech, illiquid micro-caps) should not be in the watchlist anyway.

    Returns (passed, reason, spread_pct_for_telemetry).
    """
    q = alpaca_client.get_latest_quote(ticker)
    if q is None:
        return True, "skipped (no quote available)", None

    spread_pct = q["spread_pct"]
    if spread_pct > 0.30:
        return False, f"wide spread {spread_pct:.2f}% (bid={q['bid']:.2f}/ask={q['ask']:.2f})", spread_pct
    return True, f"spread ok {spread_pct:.2f}%", spread_pct


def _gate_patterns(direction: str, df_entry: pd.DataFrame, price: float) -> tuple[bool, str]:
    """Reject obvious bad-entry patterns on the entry timeframe."""
    if df_entry is None or len(df_entry) < 22:
        return True, "skipped (insufficient entry bars)"

    # 1. Three consecutive opposite-direction bars (chasing into a pullback)
    last3 = df_entry.iloc[-3:]
    opens  = last3["open"].values.astype(float)
    closes = last3["close"].values.astype(float)
    if direction == "LONG" and all(closes < opens):
        return False, "3 consecutive red bars on entry tf (chasing pullback)"
    if direction == "SHORT" and all(closes > opens):
        return False, "3 consecutive green bars on entry tf (chasing pullback)"

    # 2. Overextended from EMA21 (mean-reversion risk too high)
    ema21 = float(_ema(df_entry["close"], 21).iloc[-1])
    atr   = _atr(df_entry, 14)
    if atr > 0:
        deviation = (price - ema21) / atr
        if direction == "LONG" and deviation > 2.5:
            return False, f"overextended above EMA21 ({deviation:+.1f} ATRs)"
        if direction == "SHORT" and deviation < -2.5:
            return False, f"overextended below EMA21 ({deviation:+.1f} ATRs)"

    # 3. Volume drop into entry (last bar < 50% of avg of prior 10 bars)
    if "volume" in df_entry.columns and len(df_entry) >= 11:
        recent_vol = float(df_entry["volume"].iloc[-1])
        avg_vol    = float(df_entry["volume"].iloc[-11:-1].mean())
        if avg_vol > 0 and recent_vol < 0.5 * avg_vol:
            return False, f"volume drop on entry bar ({recent_vol:.0f} vs avg {avg_vol:.0f})"

    return True, "no rejection patterns"


# ── Public entry point ──────────────────────────────────────────────────

# Gates that don't fit specific strategy timeframes — skipped (not failed) for
# those strategies. Swing trades hold for ~10 days, so 5-min MACD, 1-min
# reversal, and 5-min tape activity are noise at that timescale. We keep
# the structurally meaningful gates (15m trend, patterns, spread).
_SKIP_GATES_BY_STRATEGY: dict[str, set[str]] = {
    "swing_trade":   {"5m_macd", "1m_reversal", "tape"},
    "position_trade":{"5m_macd", "1m_reversal", "tape", "15m_trend"},  # even longer hold
}


def check(
    ticker:        str,
    direction:     str,
    strategy_type: str,
    df_entry:      Optional[pd.DataFrame],
    price:         float,
    entry_tf:      str = "15m",
) -> GateResult:
    """
    Run all six gates. Returns GateResult with allowed=False if any fails.

    df_entry is the entry-timeframe candles already loaded by the runner —
    we reuse it where possible to avoid extra Alpaca calls.

    Gates that don't fit the strategy timeframe are skipped (logged as
    'skipped (n/a for strategy)') instead of evaluated — see
    _SKIP_GATES_BY_STRATEGY.
    """
    result = GateResult(allowed=True)
    skip = _SKIP_GATES_BY_STRATEGY.get(strategy_type, set())

    def _maybe_skip(gate_name: str) -> bool:
        if gate_name in skip:
            result.gate_log[gate_name] = f"skipped (n/a for {strategy_type})"
            return True
        return False

    # Gate 1: 15m trend
    if not _maybe_skip("15m_trend"):
        ok, reason = _gate_15m_trend(ticker, direction, df_entry, entry_tf)
        result.gate_log["15m_trend"] = "pass" if ok else f"fail: {reason}"
        if not ok:
            result.allowed = False
            result.reasons.append(reason)

    # Gate 2: 5m MACD
    if not _maybe_skip("5m_macd"):
        ok, reason = _gate_5m_macd(ticker, direction, df_entry, entry_tf)
        result.gate_log["5m_macd"] = "pass" if ok else f"fail: {reason}"
        if not ok:
            result.allowed = False
            result.reasons.append(reason)

    # Gate 3: 1m reversal candle
    if not _maybe_skip("1m_reversal"):
        ok, reason = _gate_1m_reversal(ticker, direction)
        result.gate_log["1m_reversal"] = "pass" if ok else f"fail: {reason}"
        if not ok:
            result.allowed = False
            result.reasons.append(reason)

    # Gate 4: Pattern rejectors (uses entry-tf df, no extra fetch)
    if not _maybe_skip("patterns"):
        ok, reason = _gate_patterns(direction, df_entry, price)
        result.gate_log["patterns"] = "pass" if ok else f"fail: {reason}"
        if not ok:
            result.allowed = False
            result.reasons.append(reason)

    # Gate 5: Spread filter (1 extra Alpaca quote call) — universal
    if not _maybe_skip("spread"):
        ok, reason, spread_pct = _gate_spread(ticker)
        result.gate_log["spread"] = "pass" if ok else f"fail: {reason}"
        if spread_pct is not None:
            # Store actual spread for telemetry — see distribution after a week
            result.gate_log["spread_pct"] = round(spread_pct, 3)
        if not ok:
            result.allowed = False
            result.reasons.append(reason)

    # Gate 6: Trade tape health (no extra API call — uses in-memory tape state)
    if not _maybe_skip("tape"):
        ok, reason, tape_tele = _gate_tape(ticker)
        result.gate_log["tape"] = "pass" if ok else f"fail: {reason}"
        if tape_tele:
            result.gate_log["tape_telemetry"] = tape_tele
        if not ok:
            result.allowed = False
            result.reasons.append(reason)

    return result


# ── Telemetry ────────────────────────────────────────────────────────────

def log_rejection(
    sb,
    ticker:           str,
    direction:        str,
    strategy_type:    str,
    price:            float,
    confidence_score: float,
    gate:             GateResult,
) -> None:
    """
    Insert a rejection row into entry_gate_rejections so we can later
    measure whether the gate is correctly rejecting losers.

    Schema in supabase-entry-gate-rejections.sql. Failures here are
    swallowed — telemetry must never break the scan loop.
    """
    try:
        sb.table("entry_gate_rejections").insert({
            "ticker":           ticker,
            "direction":        direction,
            "strategy_type":    strategy_type,
            "price":            round(float(price), 4) if price else None,
            "confidence_score": round(float(confidence_score), 2) if confidence_score else None,
            "gate_log":         gate.gate_log,
            "reasons":          gate.reasons,
        }).execute()
    except Exception as e:
        logger.debug(f"[entry_gate] telemetry write failed for {ticker}: {e}")
