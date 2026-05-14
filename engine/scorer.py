"""
Confluence scorer — five independent layers summing to a normalised 0-100 score.

  L1  SMC structure      25 pts  (BOS, CHoCH, FVG, Order Block)
  L2  Technical          25 pts  (RSI divergence, MACD, VWAP, EMA alignment)
  L3  Sentiment          20 pts  (yfinance news keyword sentiment)
  L4  Risk               15 pts  (ATR regime, session timing, earnings proximity)
  L5  Multi-timeframe    15 pts  (15m + 4h direction alignment)
  ──────────────────────────────
  Raw max               100 pts
  confidence_score  =  round(raw)   → stored 0-100
  Fire threshold    >=  78          → raised from 75 after adding L5
"""

import logging
from datetime import date, datetime, timezone, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

FIRE_THRESHOLD = 78
_RAW_MAX = 100.0
_L1_MINIMUM = 13


# ---------------------------------------------------------------------------
# L1 — SMC structure  (25 pts)
# ---------------------------------------------------------------------------

def _l1_smc(structure: dict, fvgs: dict, obs: dict, direction: str, price: float) -> float:
    score = 0.0

    if direction == "LONG":
        if structure.get("choch_bullish"):
            score += 10
        elif structure.get("bos_bullish"):
            score += 7

        fvg = fvgs.get("fvg_bullish")
        if fvg:
            mid = (fvg["top"] + fvg["bottom"]) / 2
            dist = abs(price - mid) / price
            score += 7 if dist < 0.005 else (5 if dist < 0.02 else 3)

        ob = obs.get("ob_bullish")
        if ob:
            if ob["bottom"] <= price <= ob["top"]:
                score += 8
            else:
                dist = abs(price - (ob["top"] + ob["bottom"]) / 2) / price
                score += 5 if dist < 0.01 else (3 if dist < 0.025 else 0)

    else:
        if structure.get("choch_bearish"):
            score += 10
        elif structure.get("bos_bearish"):
            score += 7

        fvg = fvgs.get("fvg_bearish")
        if fvg:
            mid = (fvg["top"] + fvg["bottom"]) / 2
            dist = abs(price - mid) / price
            score += 7 if dist < 0.005 else (5 if dist < 0.02 else 3)

        ob = obs.get("ob_bearish")
        if ob:
            if ob["bottom"] <= price <= ob["top"]:
                score += 8
            else:
                dist = abs(price - (ob["top"] + ob["bottom"]) / 2) / price
                score += 5 if dist < 0.01 else (3 if dist < 0.025 else 0)

    return min(score, 25.0)


# ---------------------------------------------------------------------------
# L2 — Technical indicators  (25 pts)
# ---------------------------------------------------------------------------

def _l2_technical(df: pd.DataFrame, direction: str) -> float:
    score = 0.0

    try:
        from ta.momentum import RSIIndicator
        from ta.trend import MACD, EMAIndicator
        from ta.volume import VolumeWeightedAveragePrice

        closes  = df["close"]
        highs   = df["high"]
        lows    = df["low"]
        volumes = df["volume"]

        # RSI: up to 8 pts
        rsi_vals = RSIIndicator(close=closes, window=14).rsi().dropna()
        if len(rsi_vals) >= 5:
            rsi = float(rsi_vals.iloc[-1])
            if direction == "LONG":
                if rsi < 30:   score += 8
                elif rsi < 40: score += 6
                elif rsi < 50: score += 3
                if (float(closes.iloc[-1]) - float(closes.iloc[-5])) < 0 and \
                   (float(rsi_vals.iloc[-1]) - float(rsi_vals.iloc[-5])) > 0:
                    score += 2
            else:
                if rsi > 70:   score += 8
                elif rsi > 60: score += 6
                elif rsi > 50: score += 3
                if (float(closes.iloc[-1]) - float(closes.iloc[-5])) > 0 and \
                   (float(rsi_vals.iloc[-1]) - float(rsi_vals.iloc[-5])) < 0:
                    score += 2

        # MACD: up to 7 pts
        hist = MACD(close=closes).macd_diff().dropna()
        if len(hist) >= 2:
            h_now, h_prev = float(hist.iloc[-1]), float(hist.iloc[-2])
            if direction == "LONG":
                if h_now > 0:      score += 4
                if h_now > h_prev: score += 3
            else:
                if h_now < 0:      score += 4
                if h_now < h_prev: score += 3

        # VWAP: up to 5 pts
        vwap_vals = VolumeWeightedAveragePrice(
            high=highs, low=lows, close=closes, volume=volumes
        ).volume_weighted_average_price().dropna()
        if len(vwap_vals):
            vwap = float(vwap_vals.iloc[-1])
            last = float(closes.iloc[-1])
            if (direction == "LONG" and last > vwap) or (direction == "SHORT" and last < vwap):
                score += 5

        # EMA alignment: up to 5 pts
        ema20 = EMAIndicator(close=closes, window=20).ema_indicator().dropna()
        ema50 = EMAIndicator(close=closes, window=50).ema_indicator().dropna()
        if len(ema20) and len(ema50):
            e20, e50, last = float(ema20.iloc[-1]), float(ema50.iloc[-1]), float(closes.iloc[-1])
            if direction == "LONG":
                if last > e20 > e50: score += 5
                elif last > e20:     score += 2
            else:
                if last < e20 < e50: score += 5
                elif last < e20:     score += 2

    except Exception as e:
        logger.debug(f"L2 technical error: {e}")

    return min(score, 25.0)


# ---------------------------------------------------------------------------
# L3 — News sentiment  (20 pts)
# ---------------------------------------------------------------------------

_POSITIVE = {"surge", "rally", "beat", "record", "growth", "gain", "up", "rise",
             "strong", "bullish", "buy", "upgrade", "soar", "jump", "breakout"}
_NEGATIVE = {"fall", "drop", "miss", "loss", "decline", "down", "weak", "bearish",
             "sell", "downgrade", "plunge", "crash", "concern", "risk", "warn"}


def _l3_sentiment(ticker: str, direction: str) -> float:
    try:
        news = yf.Ticker(ticker).news
        if not news:
            return 10.0

        pos = neg = 0
        for article in news[:6]:
            title = (article.get("title") or "").lower()
            pos += sum(1 for w in _POSITIVE if w in title)
            neg += sum(1 for w in _NEGATIVE if w in title)

        total = pos + neg
        if total == 0:
            return 10.0

        ratio = (pos - neg) / total
        pts = ((ratio + 1) / 2 * 20) if direction == "LONG" else ((-ratio + 1) / 2 * 20)
        return max(0.0, min(round(pts, 1), 20.0))

    except Exception as e:
        logger.debug(f"L3 sentiment error for {ticker}: {e}")
        return 10.0


# ---------------------------------------------------------------------------
# L4 — Risk: ATR + session timing + earnings proximity  (15 pts)
# ---------------------------------------------------------------------------

def _earnings_days_away(ticker: str) -> Optional[int]:
    """Return days until next earnings, or None if unknown."""
    try:
        cal = yf.Ticker(ticker).calendar
        if cal is None:
            return None
        # calendar can be a DataFrame or dict depending on yfinance version
        if isinstance(cal, pd.DataFrame):
            if "Earnings Date" in cal.index:
                val = cal.loc["Earnings Date"].iloc[0]
            elif not cal.empty:
                val = cal.iloc[0, 0]
            else:
                return None
        elif isinstance(cal, dict):
            val = cal.get("Earnings Date") or cal.get("earningsDate")
            if isinstance(val, list):
                val = val[0] if val else None
        else:
            return None

        if val is None:
            return None
        if hasattr(val, "date"):
            val = val.date()
        elif not isinstance(val, date):
            return None
        return abs((val - date.today()).days)
    except Exception:
        return None


def _l4_risk(df: pd.DataFrame, ticker: str) -> float:
    score = 0.0

    # ATR regime: 5 pts
    try:
        from ta.volatility import AverageTrueRange
        atr_vals = AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"], window=14
        ).average_true_range().dropna()
        if len(atr_vals):
            atr_pct = float(atr_vals.iloc[-1]) / float(df["close"].iloc[-1])
            if 0.005 <= atr_pct <= 0.025: score += 5
            elif 0.002 <= atr_pct <= 0.04: score += 3
            else: score += 1
    except Exception:
        score += 2

    # Session timing: 5 pts
    hour = datetime.now(timezone.utc).hour
    if 13 <= hour <= 16:   score += 5   # peak US liquidity
    elif 9 <= hour <= 20:  score += 3
    else:                  score += 1

    # Earnings proximity: up to 5 pts bonus, or hard penalty
    days = _earnings_days_away(ticker)
    if days is not None:
        if days <= 3:
            score -= 10   # within 3 days = dangerous, penalise hard
        elif days <= 7:
            score += 0    # within a week = neutral, no bonus
        elif days >= 14:
            score += 5    # comfortably away from earnings

    return max(0.0, min(score, 15.0))


# ---------------------------------------------------------------------------
# L5 — Multi-timeframe alignment  (15 pts)
# ---------------------------------------------------------------------------

def _l5_multiframe(ticker: str, direction: str) -> float:
    """
    Fetch 15m and 4h candles independently and check if SMC direction agrees.
    +7.5 pts per timeframe that agrees → max 15 pts.
    Returns 5 (neutral) if data is unavailable.
    """
    try:
        from engine import smc
        score = 0.0
        for tf, period in [("15m", "5d"), ("4h", "60d")]:
            try:
                df = smc.fetch_candles(ticker, period=period, interval=tf)
                if df.empty:
                    score += 3.0   # can't verify → partial credit
                    continue
                df   = smc.detect_swings(df)
                stru = smc.detect_structure(df)
                if direction == "LONG":
                    agrees = stru.get("choch_bullish") or stru.get("bos_bullish")
                else:
                    agrees = stru.get("choch_bearish") or stru.get("bos_bearish")
                score += 7.5 if agrees else 0.0
            except Exception:
                score += 3.0
        logger.debug(f"[scorer] {ticker} L5 multiframe={score:.1f}")
        return min(score, 15.0)
    except Exception as e:
        logger.debug(f"L5 multiframe error: {e}")
        return 5.0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def score(analysis: dict) -> dict:
    direction = analysis.get("direction")
    if not direction:
        return {"total": 0, "passes": False, "breakdown": {}}

    price  = analysis["current_price"]
    df     = analysis["candles"]
    ticker = analysis["ticker"]

    l1 = _l1_smc(analysis["structure"], analysis["fvgs"], analysis["obs"], direction, price)

    # Hard gate: no SMC structure → skip without burning API calls on other layers
    if l1 < _L1_MINIMUM:
        logger.debug(f"[scorer] {ticker} L1={l1:.0f} < {_L1_MINIMUM} — no SMC backing, skip")
        return {
            "total":     round(l1),
            "passes":    False,
            "breakdown": {"l1_smc": round(l1), "l2_technical": 0,
                          "l3_sentiment": 0, "l4_risk": 0, "l5_mtf": 0},
            "direction":  direction,
            "entry":      analysis["entry"],
            "stop_loss":  analysis["stop_loss"],
            "target_one": analysis["target_one"],
            "target_two": analysis["target_two"],
        }

    l2 = _l2_technical(df, direction)
    l3 = _l3_sentiment(ticker, direction)
    l4 = _l4_risk(df, ticker)
    l5 = _l5_multiframe(ticker, direction)

    raw        = l1 + l2 + l3 + l4 + l5
    normalised = round(raw / _RAW_MAX * 100)

    breakdown = {
        "l1_smc":       round(l1),
        "l2_technical": round(l2),
        "l3_sentiment": round(l3),
        "l4_risk":      round(l4),
        "l5_mtf":       round(l5),
    }

    logger.info(
        f"[scorer] {ticker} total={normalised} "
        f"(L1={round(l1)} L2={round(l2)} L3={round(l3)} L4={round(l4)} L5={round(l5)})"
    )

    return {
        "total":      normalised,
        "passes":     normalised >= FIRE_THRESHOLD,
        "breakdown":  breakdown,
        "direction":  direction,
        "entry":      analysis["entry"],
        "stop_loss":  analysis["stop_loss"],
        "target_one": analysis["target_one"],
        "target_two": analysis["target_two"],
    }
