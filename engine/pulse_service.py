"""
Market Pulse — real-time sector-ETF bias.

For each tracked ETF we compute a directional bias score in [-100, +100]:
  +100  strong buy   (price > EMA9 > EMA21, with separation)
   ~0   neutral
  -100  strong sell  (price < EMA9 < EMA21)

The score combines four cheap inputs on 1H candles:
  1. Price position vs EMA9   (±40 points)
  2. EMA9 position vs EMA21   (±30 points)
  3. EMA9 slope               (±15 points)
  4. Recent momentum (5h ROC) (±15 points)

This is intentionally simple and complementary to the per-stock signal
engine — the pulse tab tells the user "where is sector money flowing
right now" so they understand the macro context of individual signals.

Cached 30s — pulse updates are minute-bar driven, no point hammering
upstream on every request.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from engine.cache import kv

logger = logging.getLogger("signalbolt.pulse")

# Default ETF set — broad market + key sectors. Matches the inspiration
# screenshot (QQQ/SPY/DIA/SMH/IGV/RSP) plus IWM small caps + XLK tech.
DEFAULT_ETFS = ["SPY", "QQQ", "DIA", "IWM", "SMH", "IGV", "XLK", "RSP"]

_CACHE_KEY = "pulse:dashboard"
_CACHE_TTL = 30   # seconds — minute bars don't change faster than this


# ── Bias derivation ──────────────────────────────────────────────────────────

def _ema(values: list[float], period: int) -> Optional[float]:
    """Simple EMA computed in pure Python (avoids a pandas import per call)."""
    if not values or len(values) < period:
        return None
    k = 2 / (period + 1)
    # Seed with SMA of the first `period` values
    ema = sum(values[:period]) / period
    for v in values[period:]:
        ema = v * k + ema * (1 - k)
    return ema


def _score_etf(closes: list[float]) -> dict:
    """
    Return {score, bias, confidence, reasons:[...]} from a price series.
    Caller passes ≥30 1H closes (more is fine).
    """
    if not closes or len(closes) < 22:
        return {
            "score": 0, "bias": "neutral", "confidence": "low",
            "reasons": ["Insufficient history"],
            "ema9": None, "ema21": None,
        }

    price = closes[-1]
    ema9  = _ema(closes, 9)
    ema21 = _ema(closes, 21)
    if ema9 is None or ema21 is None:
        return {
            "score": 0, "bias": "neutral", "confidence": "low",
            "reasons": ["EMA calc failed"],
            "ema9": None, "ema21": None,
        }

    reasons: list[str] = []
    score = 0.0

    # 1. Price vs EMA9 (±40)
    if price > ema9:
        gap = (price - ema9) / ema9 * 100
        contrib = min(40, gap * 20)   # 2% gap = +40
        score += contrib
        reasons.append(f"Price above EMA9 by {gap:.2f}%")
    else:
        gap = (ema9 - price) / ema9 * 100
        contrib = -min(40, gap * 20)
        score += contrib
        reasons.append(f"Price below EMA9 by {gap:.2f}%")

    # 2. EMA9 vs EMA21 (±30)
    if ema9 > ema21:
        gap = (ema9 - ema21) / ema21 * 100
        contrib = min(30, gap * 15)   # 2% spread = +30
        score += contrib
        reasons.append("EMA9 above EMA21")
    else:
        gap = (ema21 - ema9) / ema21 * 100
        contrib = -min(30, gap * 15)
        score += contrib
        reasons.append("EMA9 below EMA21")

    # 3. EMA9 slope — compare to EMA9 from 3 bars ago for trend direction (±15)
    ema9_prev = _ema(closes[:-3], 9)
    if ema9_prev is not None:
        slope_pct = (ema9 - ema9_prev) / ema9_prev * 100
        contrib = max(-15, min(15, slope_pct * 10))   # 1.5% / 3h slope = ±15
        score += contrib
        if slope_pct > 0.05:
            reasons.append(f"EMA9 rising ({slope_pct:+.2f}%/3h)")
        elif slope_pct < -0.05:
            reasons.append(f"EMA9 falling ({slope_pct:+.2f}%/3h)")

    # 4. 5h ROC (±15)
    if len(closes) >= 6:
        roc = (price - closes[-6]) / closes[-6] * 100
        contrib = max(-15, min(15, roc * 5))
        score += contrib
        if abs(roc) >= 0.3:
            reasons.append(f"5h ROC {roc:+.2f}%")

    score = round(max(-100, min(100, score)))

    if   score >=  60: bias, confidence = "buy",     "high"
    elif score >=  25: bias, confidence = "buy",     "medium"
    elif score <= -60: bias, confidence = "sell",    "high"
    elif score <= -25: bias, confidence = "sell",    "medium"
    else:              bias, confidence = "neutral", "low"

    return {
        "score":      score,
        "bias":       bias,
        "confidence": confidence,
        "reasons":    reasons[:3],   # keep card tidy
        "ema9":       round(ema9, 2),
        "ema21":      round(ema21, 2),
    }


# ── Public entry point ───────────────────────────────────────────────────────

def get_pulse(tickers: Optional[list[str]] = None) -> dict:
    """
    Build the market pulse payload:
      {
        "as_of":    ISO string,
        "etfs":     [{ticker, price, changePercent, volume, score, bias, ...}, ...],
        "strongest": {ticker, score, bias, ...}  # highest |score|
        "alerts":   {"buy": [tickers...], "sell": [tickers...]},
      }
    Cached 30s. Falls back to last-good cache on upstream failures.
    """
    tickers = tickers or DEFAULT_ETFS

    cached = kv.get_json(_CACHE_KEY)
    if cached:
        # Refresh-on-stale: serve cache if it covers the same ticker set.
        cached_set = {e["ticker"] for e in cached.get("etfs", [])}
        if cached_set >= set(tickers):
            return _filter(cached, tickers)

    try:
        from engine.alpaca_client import get_multi_bars, get_latest_prices
        from datetime import datetime, timezone
        bars  = get_multi_bars(tickers, timeframe="1Hour", days=10)
        prices = get_latest_prices(tickers)

        etfs = []
        for t in tickers:
            df = bars.get(t)
            px = prices.get(t)
            if df is None or len(df) < 22 or px is None:
                continue
            closes = df["close"].astype(float).tolist()
            yest_close = closes[-min(7, len(closes))]  # ~7h ago = approx prev close
            chg_pct = (px - yest_close) / yest_close * 100 if yest_close else 0.0
            volume = int(df["volume"].iloc[-1]) if "volume" in df else 0

            bias = _score_etf(closes)
            etfs.append({
                "ticker":        t,
                "price":         round(px, 2),
                "changePercent": round(chg_pct, 2),
                "volume":        volume,
                **bias,
            })

        strongest = max(etfs, key=lambda e: abs(e["score"]), default=None) if etfs else None
        alerts = {
            "buy":  [e["ticker"] for e in etfs if e["bias"] == "buy"],
            "sell": [e["ticker"] for e in etfs if e["bias"] == "sell"],
        }
        result = {
            "as_of":     datetime.now(timezone.utc).isoformat(),
            "etfs":      etfs,
            "strongest": strongest,
            "alerts":    alerts,
        }
        kv.set_json(_CACHE_KEY, result, ttl_sec=_CACHE_TTL)
        return _filter(result, tickers)
    except Exception as e:
        logger.error(f"[pulse] build failed: {e}")
        if cached:
            return _filter(cached, tickers)
        return {"as_of": None, "etfs": [], "strongest": None, "alerts": {"buy": [], "sell": []}}


def _filter(payload: dict, tickers: list[str]) -> dict:
    wanted = set(tickers)
    etfs = [e for e in payload.get("etfs", []) if e["ticker"] in wanted]
    return {
        "as_of":     payload.get("as_of"),
        "etfs":      etfs,
        "strongest": payload.get("strongest") if (payload.get("strongest") and payload["strongest"]["ticker"] in wanted) else (max(etfs, key=lambda e: abs(e["score"]), default=None) if etfs else None),
        "alerts": {
            "buy":  [t for t in payload.get("alerts", {}).get("buy",  []) if t in wanted],
            "sell": [t for t in payload.get("alerts", {}).get("sell", []) if t in wanted],
        },
    }
