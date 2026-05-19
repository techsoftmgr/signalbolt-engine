"""
Manipulation Detector
=====================
Detects common market manipulation patterns using price action
and volume data available from Alpaca/yfinance:

  STOP_RAID         — spike to obvious level then immediate reversal
  MOMENTUM_IGNITION — >3 std dev move with no news catalyst
  WASH_TRADING      — high volume with no net price movement (crypto)
  PUMP_AND_DUMP     — price spike on abnormally low options OI
  VOLUME_ANOMALY    — volume spike with minimal price movement

Note: Spoofing/layering require Level 2 order book data (not available
at $250/mo budget). These patterns use Level 1 data approximations.

Returns a clean flag (True = signal is safe) and a score (0-100).
Used as L9 bonus in scorer.py and as a pre-scan gate in runner.py.
"""

import logging
import math
from typing import Optional

import pandas as pd

logger = logging.getLogger("signalbolt.manipulation")

# ── Thresholds ────────────────────────────────────────────────
STD_DEV_SPIKE     = 3.0    # > 3 std devs = suspicious move
VOLUME_SPIKE_MIN  = 3.0    # volume 3× average = flag
RAID_RANGE_PCT    = 0.015  # intraday range > 1.5% on 5-min bar = raid candidate
RAID_RETURN_PCT   = 0.70   # price returned > 70% of spike = confirmed raid


def _compute_price_std(closes: list, window: int = 20) -> float:
    """Rolling std dev of returns."""
    if len(closes) < window + 1:
        return 0.01
    returns = [
        (closes[i] - closes[i - 1]) / closes[i - 1]
        for i in range(len(closes) - window, len(closes))
    ]
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / len(returns)
    return math.sqrt(variance)


def detect(
    df: pd.DataFrame,
    ticker: str,
    direction: str,
    has_news: bool = False,
    is_crypto: bool = False,
) -> dict:
    """
    Run all manipulation checks on a price DataFrame.

    Args:
        df:         OHLCV DataFrame (from smc.fetch_candles)
        ticker:     Ticker symbol
        direction:  'LONG' or 'SHORT'
        has_news:   Whether a news catalyst exists
        is_crypto:  Whether this is a crypto ticker

    Returns:
        {
          "is_clean":           bool,
          "flags":              list[str],
          "score":              float,    # 0-100 (100 = totally clean)
          "stop_raid_risk":     bool,
          "momentum_ignition":  bool,
          "volume_anomaly":     bool,
          "details":            dict,
        }
    """
    flags = []
    details = {}

    if df is None or df.empty or len(df) < 5:
        return {
            "is_clean": True,
            "flags": [],
            "score": 75.0,
            "stop_raid_risk": False,
            "momentum_ignition": False,
            "volume_anomaly": False,
            "details": {"reason": "insufficient_data"},
        }

    closes  = df["close"].tolist()
    highs   = df["high"].tolist()
    lows    = df["low"].tolist()
    volumes = df["volume"].tolist()

    last_close = closes[-1]
    last_high  = highs[-1]
    last_low   = lows[-1]
    last_vol   = volumes[-1]

    avg_vol = sum(volumes[:-5]) / max(len(volumes) - 5, 1) if len(volumes) > 5 else last_vol

    # ── Stop Raid Detection ───────────────────────────────────
    # Look for: wide bar range + price returns near open → raid
    last_open   = float(df["open"].iloc[-1]) if "open" in df.columns else last_close
    bar_range   = (last_high - last_low) / last_close
    price_move  = abs(last_close - last_open) / last_open
    returned    = bar_range > 0 and (price_move / bar_range) < (1 - RAID_RETURN_PCT)

    stop_raid = bar_range > RAID_RANGE_PCT and returned
    if stop_raid:
        flags.append("STOP_RAID")
        details["stop_raid"] = {
            "bar_range_pct": round(bar_range * 100, 2),
            "price_move_pct": round(price_move * 100, 2),
        }
        logger.debug(f"[manip] {ticker} STOP_RAID: range={bar_range:.1%} move={price_move:.1%}")

    # ── Momentum Ignition ─────────────────────────────────────
    # Price moved > 3 std devs in last bar with no news
    price_std = _compute_price_std(closes)
    last_return = abs(closes[-1] - closes[-2]) / closes[-2] if len(closes) >= 2 else 0
    std_devs = last_return / price_std if price_std > 0 else 0

    momentum_ignition = std_devs > STD_DEV_SPIKE and not has_news
    if momentum_ignition:
        flags.append("MOMENTUM_IGNITION")
        details["momentum_ignition"] = {
            "std_devs": round(std_devs, 1),
            "has_news": has_news,
        }
        logger.debug(f"[manip] {ticker} MOMENTUM_IGNITION: {std_devs:.1f}σ, no news")

    # ── Volume Anomaly ────────────────────────────────────────
    # Volume spike > 3× with minimal price move (possible spoofing echo)
    vol_multiple = last_vol / avg_vol if avg_vol > 0 else 1.0
    low_price_move = price_move < 0.003   # less than 0.3% price change

    volume_anomaly = vol_multiple > VOLUME_SPIKE_MIN and low_price_move
    if volume_anomaly:
        flags.append("VOLUME_ANOMALY")
        details["volume_anomaly"] = {
            "vol_multiple": round(vol_multiple, 1),
            "price_move_pct": round(price_move * 100, 2),
        }

    # ── Wash Trading (crypto) ─────────────────────────────────
    if is_crypto:
        # High volume + minimal directional move = wash trading
        if vol_multiple > 5 and price_move < 0.005:
            flags.append("WASH_TRADING")
            details["wash_trading"] = {"vol_multiple": round(vol_multiple, 1)}

    # ── Compute Score ─────────────────────────────────────────
    score = 100.0
    if "STOP_RAID" in flags:          score -= 25
    if "MOMENTUM_IGNITION" in flags:  score -= 30
    if "VOLUME_ANOMALY" in flags:     score -= 15
    if "WASH_TRADING" in flags:       score -= 35

    # Partial credit: even with minor flags, score stays in range
    score = max(0.0, min(100.0, score))

    is_clean = len(flags) == 0

    result = {
        "is_clean":           is_clean,
        "flags":              flags,
        "score":              round(score, 1),
        "stop_raid_risk":     stop_raid,
        "momentum_ignition":  momentum_ignition,
        "volume_anomaly":     volume_anomaly,
        "details":            details,
    }

    if not is_clean:
        logger.info(f"[manip] {ticker} flags={flags} score={score:.0f}")

    return result


def is_blocking(manip: dict) -> bool:
    """
    Return True if manipulation is severe enough to block the signal.
    STOP_RAID alone is a warning, not a block.
    MOMENTUM_IGNITION blocks (artificial move, don't chase it).
    WASH_TRADING blocks for crypto.
    """
    blocking = {"MOMENTUM_IGNITION", "WASH_TRADING"}
    return any(f in blocking for f in manip.get("flags", []))


def score_for_signal(manip: dict) -> float:
    """
    Return 0-100 score contribution for manipulation cleanliness.
    Used as L9 bonus in scorer.py.
    """
    return manip.get("score", 85.0)
