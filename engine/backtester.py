"""
Historical Backtester
=====================
Simulates signals on historical candles and records outcomes.
Provides training data for the weight optimizer.

How it works:
  1. Fetch N days of OHLCV candles for a ticker (yfinance, free, no API cost)
  2. Slide a rolling window across the candles (step = 5 bars)
  3. For each window:
     a. Run SMC structure detection (BOS/CHoCH/FVG/OB) on the window
     b. Determine signal direction from structure
     c. Compute L1-L5 raw scores (same functions as scorer.py)
     d. Calculate ATR-based T1 and SL
     e. Look forward N bars: did price hit T1 (win) or SL (loss)?
  4. Return list[TrainingPoint] for the optimizer

Notes:
  - L3 (sentiment) uses neutral 10.0 — news data is live-only
  - L5 (MTF) uses a simplified same-data approximation
  - L6-L9 quant layers are not included — backtester focuses on L1-L4 weights
  - Minimum 30 bars needed per window; minimum 20 forward bars to simulate outcome
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger("signalbolt.backtester")

# Strategy-specific backtest config
BACKTEST_CONFIG: dict[str, dict] = {
    'scalping': {
        'interval':     '5m',
        'fetch_period': '30d',    # yfinance max for 5m
        'window':       60,       # bars per analysis window
        'step':         5,        # slide step
        'forward':      12,       # bars to simulate outcome (~1h)
        'atr_tp':       1.0,      # T1 = entry ± ATR × atr_tp
        'atr_sl':       0.6,      # SL = entry ∓ ATR × atr_sl
    },
    'day_trade': {
        'interval':     '15m',
        'fetch_period': '60d',
        'window':       80,
        'step':         8,
        'forward':      20,       # bars (~5h)
        'atr_tp':       1.5,
        'atr_sl':       1.0,
    },
    'swing_trade': {
        'interval':     '1h',
        'fetch_period': '180d',
        'window':       100,
        'step':         10,
        'forward':      30,       # bars (~5 days)
        'atr_tp':       2.5,
        'atr_sl':       1.5,
    },
    'options_flow': {
        'interval':     '15m',
        'fetch_period': '60d',
        'window':       80,
        'step':         8,
        'forward':      20,
        'atr_tp':       1.5,
        'atr_sl':       1.0,
    },
    'dark_pool': {
        'interval':     '15m',
        'fetch_period': '60d',
        'window':       80,
        'step':         8,
        'forward':      20,
        'atr_tp':       1.5,
        'atr_sl':       1.0,
    },
}


@dataclass
class TrainingPoint:
    """A single simulated historical signal with its outcome."""
    ticker:       str
    strategy:     str
    direction:    str           # 'LONG' or 'SHORT'
    # Raw layer scores (0 to max for each layer)
    l1_raw:       float         # 0-25
    l2_raw:       float         # 0-25
    l3_raw:       float         # 0-20 (always 10.0 in backtest)
    l4_raw:       float         # 0-15
    l5_raw:       float         # 0-15 (simplified)
    # Outcome
    outcome:      int           # 1 = win (T1 hit first), 0 = loss (SL hit first), -1 = no resolution
    risk_reward:  float         # actual R:R if outcome ≠ -1, else 0
    entry_price:  float
    sl_price:     float
    tp_price:     float


# ---------------------------------------------------------------------------
# ATR helper
# ---------------------------------------------------------------------------

def _atr(df: pd.DataFrame, period: int = 14) -> float:
    try:
        high  = df["high"]
        low   = df["low"]
        prev  = df["close"].shift(1)
        tr = pd.concat([
            high - low,
            (high - prev).abs(),
            (low  - prev).abs(),
        ], axis=1).max(axis=1)
        val = float(tr.rolling(period).mean().iloc[-1])
        return val if not np.isnan(val) else float(df["close"].iloc[-1]) * 0.01
    except Exception:
        return float(df["close"].iloc[-1]) * 0.01


# ---------------------------------------------------------------------------
# Simplified L1 scorer (no live data needed — works on DataFrame slice)
# ---------------------------------------------------------------------------

def _score_l1(df: pd.DataFrame, direction: str) -> float:
    """Simplified L1 SMC score on a historical window. Returns 0-25."""
    try:
        from engine import smc
        df_sw  = smc.detect_swings(df)
        struct = smc.detect_structure(df_sw)
        fvgs   = smc.detect_fvg(df_sw)
        obs    = smc.detect_order_blocks(df_sw)
        sweep  = smc.detect_liquidity_sweep(df_sw, direction)

        from engine.scorer import _l1_smc
        price = float(df["close"].iloc[-1])
        return _l1_smc(struct, fvgs, obs, direction, price, sweep)
    except Exception as e:
        logger.debug(f"[backtest] L1 error: {e}")
        return 5.0   # neutral


# ---------------------------------------------------------------------------
# Simplified L2 scorer
# ---------------------------------------------------------------------------

def _score_l2(df: pd.DataFrame, direction: str, strategy_type: str) -> float:
    try:
        from engine.scorer import _l2_technical
        return _l2_technical(df, direction, strategy_type)
    except Exception as e:
        logger.debug(f"[backtest] L2 error: {e}")
        return 10.0


# ---------------------------------------------------------------------------
# Simplified L4 scorer (ATR only, skip live session timing)
# ---------------------------------------------------------------------------

def _score_l4(df: pd.DataFrame) -> float:
    try:
        atr_val = _atr(df)
        price   = float(df["close"].iloc[-1])
        atr_pct = atr_val / price if price > 0 else 0
        if 0.005 <= atr_pct <= 0.025:
            return 10.0    # good ATR range + neutral session timing (5 pts each)
        elif 0.002 <= atr_pct <= 0.04:
            return 7.0
        return 5.0
    except Exception:
        return 5.0


# ---------------------------------------------------------------------------
# Direction detection from SMC structure
# ---------------------------------------------------------------------------

def _detect_direction(df: pd.DataFrame) -> Optional[str]:
    """
    Return 'LONG', 'SHORT', or None if no clear structure.
    CHoCH takes priority over BOS (stronger signal).
    """
    try:
        from engine import smc
        df_sw  = smc.detect_swings(df)
        struct = smc.detect_structure(df_sw)

        if struct.get("choch_bullish"):
            return "LONG"
        if struct.get("choch_bearish"):
            return "SHORT"
        if struct.get("bos_bullish"):
            return "LONG"
        if struct.get("bos_bearish"):
            return "SHORT"
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Outcome simulation
# ---------------------------------------------------------------------------

def _simulate_outcome(
    forward_df: pd.DataFrame,
    direction: str,
    entry: float,
    tp: float,
    sl: float,
) -> tuple[int, float]:
    """
    Walk forward bars until T1 or SL is hit.
    Returns (outcome, realized_rr):
      outcome 1 = win, 0 = loss, -1 = no resolution within window
    """
    for _, row in forward_df.iterrows():
        high  = float(row["high"])
        low   = float(row["low"])

        if direction == "LONG":
            if low <= sl:
                return 0, -1.0
            if high >= tp:
                rr = (tp - entry) / (entry - sl) if (entry - sl) > 0 else 0
                return 1, round(rr, 2)
        else:
            if high >= sl:
                return 0, -1.0
            if low <= tp:
                rr = (entry - tp) / (sl - entry) if (sl - entry) > 0 else 0
                return 1, round(rr, 2)

    return -1, 0.0   # no resolution — discard this point in optimizer


# ---------------------------------------------------------------------------
# Fetch historical candles (Alpaca SIP — real tape, NOT yfinance synthetic)
# ---------------------------------------------------------------------------

# yfinance interval/period strings → Alpaca timeframe
_TF_MAP = {"5m": "5Min", "15m": "15Min", "30m": "30Min", "1h": "1Hour", "1d": "1Day"}


def _fetch_historical(ticker: str, interval: str, period: str) -> pd.DataFrame:
    """Real OHLCV from Alpaca SIP (replaces the old yfinance source — too weak to
    tune real-money rules). Returns lowercase-column df with a 'timestamp' column,
    or empty df on failure."""
    try:
        from engine.alpaca_client import get_bars
        tf = _TF_MAP.get(interval, "15Min")
        try:
            days = int("".join(c for c in str(period) if c.isdigit()) or "60")
        except Exception:
            days = 60
        df = get_bars(ticker, tf, max(2, days))
        if df is None or df.empty or len(df) < 50:
            return pd.DataFrame()
        df = df.reset_index()
        first = df.columns[0]                       # get_bars indexes by UTC timestamp
        df = df.rename(columns={first: "timestamp"})
        df.columns = [str(c).lower() for c in df.columns]
        return df.sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        logger.debug(f"[backtest] Alpaca fetch error for {ticker}: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    tickers: list[str],
    strategy_type: str,
    max_points_per_ticker: int = 50,
) -> list[TrainingPoint]:
    """
    Run backtester for a list of tickers and a strategy.

    Returns a list of TrainingPoint (only resolved outcomes — win or loss).
    """
    cfg     = BACKTEST_CONFIG.get(strategy_type, BACKTEST_CONFIG["day_trade"])
    results: list[TrainingPoint] = []

    for ticker in tickers:
        logger.info(f"[backtest] {ticker} [{strategy_type}] ...")
        df = _fetch_historical(ticker, cfg["interval"], cfg["fetch_period"])
        if df.empty or len(df) < cfg["window"] + cfg["forward"] + 10:
            logger.debug(f"[backtest] {ticker}: insufficient data")
            continue

        ticker_points = 0
        # Slide window with step
        for i in range(cfg["window"], len(df) - cfg["forward"], cfg["step"]):
            if ticker_points >= max_points_per_ticker:
                break

            window  = df.iloc[i - cfg["window"]: i].copy()
            forward = df.iloc[i: i + cfg["forward"]].copy()

            if len(window) < cfg["window"] or len(forward) < cfg["forward"]:
                continue

            direction = _detect_direction(window)
            if direction is None:
                continue

            entry = float(window["close"].iloc[-1])
            atr   = _atr(window)

            if direction == "LONG":
                tp = entry + atr * cfg["atr_tp"]
                sl = entry - atr * cfg["atr_sl"]
            else:
                tp = entry - atr * cfg["atr_tp"]
                sl = entry + atr * cfg["atr_sl"]

            # R:R check (only simulate worthwhile setups)
            potential_rr = (abs(tp - entry) / abs(entry - sl)) if abs(entry - sl) > 0 else 0
            if potential_rr < 1.5:
                continue

            outcome, realized_rr = _simulate_outcome(forward, direction, entry, tp, sl)
            if outcome == -1:
                continue   # no resolution — skip

            # Compute raw layer scores on the window
            l1 = _score_l1(window, direction)
            l2 = _score_l2(window, direction, strategy_type)
            l3 = 10.0       # neutral — live news only
            l4 = _score_l4(window)
            l5 = 7.5        # neutral — MTF approximated

            results.append(TrainingPoint(
                ticker=ticker,
                strategy=strategy_type,
                direction=direction,
                l1_raw=l1, l2_raw=l2, l3_raw=l3, l4_raw=l4, l5_raw=l5,
                outcome=outcome,
                risk_reward=realized_rr,
                entry_price=entry,
                sl_price=sl,
                tp_price=tp,
            ))
            ticker_points += 1

        logger.info(
            f"[backtest] {ticker} [{strategy_type}]: "
            f"{ticker_points} training points "
            f"(win_rate={sum(1 for p in results[-ticker_points:] if p.outcome == 1) / max(ticker_points, 1):.0%})"
        )

    total_wins = sum(1 for p in results if p.outcome == 1)
    logger.info(
        f"[backtest] {strategy_type} complete — "
        f"{len(results)} points | win_rate={total_wins / max(len(results), 1):.1%}"
    )
    return results
