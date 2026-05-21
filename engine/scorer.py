"""
Confluence scorer — nine independent layers → normalised 0-100 composite score.

  L1  SMC structure      25 pts  (BOS, CHoCH, FVG, Order Block)
  L2  Technical          25 pts  (RSI divergence, MACD, VWAP, EMA alignment)
  L3  Sentiment          20 pts  (Alpaca news sentiment, yfinance fallback)
  L4  Risk               15 pts  (ATR regime, session timing, earnings proximity)
  L5  Multi-timeframe    15 pts  (15m + 4h direction alignment)
  ── Quant layers (bonus — max ±10 pts total) ──────────────────────────────
  L6  Market Regime       bonus  (VIX, ADX, SPY/200MA classification)
  L7  Session Quality     bonus  (time-of-day, OpEx, FOMC)
  L8  Gamma Exposure      bonus  (SpotGamma walls, pin risk, vanna/charm)
  L9  Manipulation Check  bonus  (stop raid, momentum ignition detection)
  ──────────────────────────────────────────────────────────────────────────
  Score architecture (anti-inflation):
    base_score  = min(85, weighted_L1_L5 + l5_bonus)   ← hard cap
    quant_bonus = clamp(L6+L7+L8+L9, -10, +10)         ← hard cap
    chop_penalty= chop_detector.as_penalty()            ← 0 to -15
    final       = clamp(base + quant - chop, 0, 100)

  Confidence grades: A+(≥90) A(≥82) B+(≥74) B(≥66) C(<66)
  Fire threshold  >= session threshold (78 standard / 80 ORB / 85 catalyst)
"""

import logging
import time
from datetime import date, datetime, timezone, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf   # kept for L4 earnings calendar ONLY — Alpaca has no earnings data
                        # L3 news now uses Alpaca News API primary (yfinance fallback)

logger = logging.getLogger(__name__)

# ── L3 news sentiment cache ───────────────────────────────────
# Alpaca News API called once per scan. News headlines don't change
# minute-to-minute — 30-min cache is safe.
_l3_cache: dict[str, tuple[float, float]] = {}  # ticker → (score, fetched_at)
_L3_CACHE_TTL = 1800  # 30 minutes

# ── L4 earnings calendar cache ────────────────────────────────
# Earnings dates are fixed weeks ahead — 24h cache is fine.
_earnings_cache: dict[str, tuple[Optional[int], float]] = {}  # ticker → (days, fetched_at)
_EARNINGS_CACHE_TTL = 86400  # 24 hours

FIRE_THRESHOLD = 78  # legacy default — use STRATEGY_THRESHOLDS per strategy

STRATEGY_THRESHOLDS: dict[str, int] = {
    'scalping':     78,   # quality bar: stricter SMC filters do the heavy lifting
    'day_trade':    78,   # 78 with stricter filters >> 75 with loose filters
    'swing_trade':  76,   # swing needs room — structure filters are already tight
    'options_flow': 78,
    'dark_pool':    76,
}

# Weights for each scoring layer (must sum to 100 per strategy)
STRATEGY_WEIGHTS: dict[str, dict[str, float]] = {
    'scalping':     {'smc': 15, 'technical': 40, 'sentiment': 15, 'risk': 30},
    'day_trade':    {'smc': 25, 'technical': 35, 'sentiment': 25, 'risk': 15},
    'swing_trade':  {'smc': 40, 'technical': 30, 'sentiment': 20, 'risk': 10},
    'options_flow': {'smc': 10, 'technical': 20, 'sentiment': 50, 'risk': 20},
    'dark_pool':    {'smc': 10, 'technical': 20, 'sentiment': 60, 'risk': 10},
}

# Raw max per layer (used to normalise to 0-1 before applying strategy weights)
_L_MAX = {'smc': 25.0, 'technical': 25.0, 'sentiment': 20.0, 'risk': 15.0}

_L1_MINIMUM = 12   # requires BOS/CHoCH (7-10 pts) + at least one nearby FVG/OB


# ---------------------------------------------------------------------------
# L1 — SMC structure  (25 pts)
# ---------------------------------------------------------------------------

def _l1_smc(
    structure: dict,
    fvgs: dict,
    obs: dict,
    direction: str,
    price: float,
    sweep: Optional[dict] = None,
) -> float:
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
            # Only award points if price is near the FVG — distant FVGs are noise
            score += 7 if dist < 0.005 else (5 if dist < 0.015 else (2 if dist < 0.025 else 0))

        ob = obs.get("ob_bullish")
        if ob:
            if ob["bottom"] <= price <= ob["top"]:
                score += 8
            else:
                dist = abs(price - (ob["top"] + ob["bottom"]) / 2) / price
                # Only award points for nearby OBs — distant OBs are irrelevant
                score += 5 if dist < 0.01 else (2 if dist < 0.02 else 0)

    else:
        if structure.get("choch_bearish"):
            score += 10
        elif structure.get("bos_bearish"):
            score += 7

        fvg = fvgs.get("fvg_bearish")
        if fvg:
            mid = (fvg["top"] + fvg["bottom"]) / 2
            dist = abs(price - mid) / price
            score += 7 if dist < 0.005 else (5 if dist < 0.015 else (2 if dist < 0.025 else 0))

        ob = obs.get("ob_bearish")
        if ob:
            if ob["bottom"] <= price <= ob["top"]:
                score += 8
            else:
                dist = abs(price - (ob["top"] + ob["bottom"]) / 2) / price
                score += 5 if dist < 0.01 else (2 if dist < 0.02 else 0)

    # Liquidity sweep bonus — market maker raid confirmed before entry
    if sweep and sweep.get("swept"):
        candles_ago = sweep.get("candles_ago", 99)
        if candles_ago <= 2:
            score += 8   # very fresh raid (within 2 candles) — high-quality entry
        elif candles_ago <= 5:
            score += 5   # recent raid — still valid confirmation

    return min(score, 25.0)


# ---------------------------------------------------------------------------
# L2 — Technical indicators  (25 pts)
# ---------------------------------------------------------------------------

def _l2_technical(df: pd.DataFrame, direction: str, strategy_type: str = "day_trade") -> float:
    score = 0.0

    # EMA windows vary by strategy
    if strategy_type == "scalping":
        ema_fast, ema_slow, rsi_window = 9, 21, 7
    elif strategy_type == "swing_trade":
        ema_fast, ema_slow, rsi_window = 50, 200, 14
    else:
        ema_fast, ema_slow, rsi_window = 20, 50, 14

    try:
        from ta.momentum import RSIIndicator
        from ta.trend import MACD, EMAIndicator
        from ta.volume import VolumeWeightedAveragePrice

        closes  = df["close"]
        highs   = df["high"]
        lows    = df["low"]
        volumes = df["volume"]

        # RSI: up to 8 pts
        rsi_vals = RSIIndicator(close=closes, window=rsi_window).rsi().dropna()
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
        ema_f = EMAIndicator(close=closes, window=ema_fast).ema_indicator().dropna()
        ema_s = EMAIndicator(close=closes, window=ema_slow).ema_indicator().dropna()
        if len(ema_f) and len(ema_s):
            ef, es, last = float(ema_f.iloc[-1]), float(ema_s.iloc[-1]), float(closes.iloc[-1])
            if direction == "LONG":
                if last > ef > es: score += 5
                elif last > ef:    score += 2
            else:
                if last < ef < es: score += 5
                elif last < ef:    score += 2

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
    """
    News-keyword sentiment for standard strategies (scalping/day_trade/swing_trade).
    Cached 30 minutes — news headlines change slowly.

    Data source: Alpaca News API (real-time, included in SIP plan) with
    yfinance fallback. Both are keyword-scanned for bullish/bearish signals.
    """
    now = time.monotonic()
    cached = _l3_cache.get(ticker)
    if cached and (now - cached[1]) < _L3_CACHE_TTL:
        return cached[0] if direction == "LONG" else (20.0 - cached[0])

    headlines: list[str] = []

    # ── Alpaca News API (primary) ─────────────────────────────────────────────
    try:
        from engine.alpaca_client import get_news
        articles = get_news(ticker, limit=6)
        for a in articles:
            # Alpaca news fields: headline, summary, author, content
            title = (a.get("headline") or a.get("summary") or "").lower()
            if title:
                headlines.append(title)
    except Exception as e:
        logger.debug(f"L3 Alpaca news error for {ticker}: {e}")

    # ── yfinance fallback ─────────────────────────────────────────────────────
    if not headlines:
        try:
            news_yf = yf.Ticker(ticker).news
            for article in (news_yf or [])[:6]:
                title = (article.get("title") or "").lower()
                if title:
                    headlines.append(title)
        except Exception as e:
            logger.debug(f"L3 yfinance news fallback error for {ticker}: {e}")

    if not headlines:
        _l3_cache[ticker] = (10.0, now)
        return 10.0

    pos = neg = 0
    for title in headlines:
        pos += sum(1 for w in _POSITIVE if w in title)
        neg += sum(1 for w in _NEGATIVE if w in title)

    total = pos + neg
    if total == 0:
        _l3_cache[ticker] = (10.0, now)
        return 10.0

    ratio      = (pos - neg) / total
    long_score = round(((ratio + 1) / 2 * 20), 1)
    long_score = max(0.0, min(long_score, 20.0))
    _l3_cache[ticker] = (long_score, now)
    return long_score if direction == "LONG" else (20.0 - long_score)


def _l3_flow_sentiment(analysis: dict, direction: str) -> float:
    """
    Fix #1: For options_flow and dark_pool strategies, use actual UW data
    (premium-weighted bull/bear sentiment) instead of news keyword scanning.

    options_flow: bull_premium vs bear_premium ratio from UW flow alerts.
    dark_pool:    total_notional size as a conviction proxy.

    Both are far more reliable signals than headline keyword matching.
    """
    strategy = analysis.get("strategy_type", "")

    if strategy == "options_flow":
        bull_prem = float(analysis.get("bull_premium") or 0)
        bear_prem = float(analysis.get("bear_premium") or 0)
        total = bull_prem + bear_prem

        if total == 0:
            return 10.0  # no UW premium data — neutral

        dominant = bull_prem if direction == "LONG" else bear_prem
        ratio = dominant / total  # 0.0–1.0 (1.0 = fully aligned)

        # Scale to 0-20: 60% dominance → 12 pts, 80% → 16 pts, 95% → 19 pts
        score = ratio * 20.0
        return round(max(0.0, min(score, 20.0)), 1)

    elif strategy == "dark_pool":
        total_notional = float(analysis.get("total_notional") or 0)
        # Conviction based on block size — larger blocks = stronger signal
        if total_notional >= 5_000_000:   return 19.0   # $5M+  — institutional conviction
        elif total_notional >= 2_000_000: return 17.0   # $2M+
        elif total_notional >= 1_000_000: return 15.0   # $1M+
        elif total_notional >= 500_000:   return 13.0   # $500K+
        elif total_notional >= 200_000:   return 11.0   # $200K+
        else:                             return 9.0    # below threshold


# ---------------------------------------------------------------------------
# L4 — Risk: ATR + session timing + earnings proximity  (15 pts)
# ---------------------------------------------------------------------------

def _earnings_days_away(ticker: str) -> Optional[int]:
    """
    Return days until next earnings, or None if unknown.
    Cached 24h — earnings dates don't change intraday.
    """
    now = time.monotonic()
    cached = _earnings_cache.get(ticker)
    if cached and (now - cached[1]) < _EARNINGS_CACHE_TTL:
        return cached[0]

    result: Optional[int] = None
    try:
        cal = yf.Ticker(ticker).calendar
        if cal is None:
            pass
        elif isinstance(cal, pd.DataFrame):
            if "Earnings Date" in cal.index:
                val = cal.loc["Earnings Date"].iloc[0]
            elif not cal.empty:
                val = cal.iloc[0, 0]
            else:
                val = None
            if val is not None:
                if hasattr(val, "date"):
                    val = val.date()
                if isinstance(val, date):
                    result = abs((val - date.today()).days)
        elif isinstance(cal, dict):
            val = cal.get("Earnings Date") or cal.get("earningsDate")
            if isinstance(val, list):
                val = val[0] if val else None
            if val is not None:
                if hasattr(val, "date"):
                    val = val.date()
                if isinstance(val, date):
                    result = abs((val - date.today()).days)
    except Exception:
        pass

    _earnings_cache[ticker] = (result, now)
    return result


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
# Confidence factor labels  (human-readable breakdown for the app)
# ---------------------------------------------------------------------------

def _factors_from_breakdown(breakdown: dict, direction: str) -> list:
    factors = []
    l1     = breakdown.get("l1_smc", 0)
    l2     = breakdown.get("l2_technical", 0)
    l3     = breakdown.get("l3_sentiment", 0)
    l4     = breakdown.get("l4_risk", 0)
    l5     = breakdown.get("l5_mtf", 0)
    swept  = breakdown.get("sweep_confirmed", False)

    bull = direction == "LONG"

    # Liquidity sweep — show first if confirmed (most distinctive signal)
    if swept:
        factors.append("Stop Hunt Confirmed" if bull else "Liquidity Raid Confirmed")

    # L1 — SMC structure
    if l1 >= 20:
        factors.append("CHoCH + Order Block")
    elif l1 >= 15:
        factors.append("Break of Structure")
    elif l1 >= 13:
        factors.append("SMC Setup")

    # L2 — Technicals
    if l2 >= 20:
        factors.append("RSI + MACD Aligned")
    elif l2 >= 14:
        factors.append("Bullish Technicals" if bull else "Bearish Technicals")
    elif l2 >= 8:
        factors.append("VWAP + EMA Stack")

    # L3 — Sentiment
    if l3 >= 14:
        factors.append("Bullish News" if bull else "Bearish News")
    elif l3 >= 11:
        factors.append("Positive Sentiment" if bull else "Negative Sentiment")

    # L4 — Risk environment
    if l4 >= 10:
        factors.append("Optimal Risk Window")

    # L5 — Multi-timeframe
    if l5 >= 12:
        factors.append("15m + 4H Aligned")
    elif l5 >= 7:
        factors.append("Multi-TF Aligned")

    return factors[:4]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def score(
    analysis: dict,
    strategy_type: str = "day_trade",
    regime: Optional[dict] = None,
    session: Optional[dict] = None,
    gamma: Optional[dict] = None,
    manipulation: Optional[dict] = None,
    chop: Optional[object] = None,   # ChopResult from chop_detector.detect()
) -> dict:
    """
    Score a signal candidate across all 9 layers.

    Args:
        analysis:      SMC analysis dict (from smc.analyze)
        strategy_type: trading strategy
        regime:        output of regime_detector.detect()  (optional — adds L6)
        session:       output of session_classifier.classify()  (optional — adds L7)
        gamma:         output of gamma_engine.fetch()  (optional — adds L8)
        manipulation:  output of manipulation_detector.detect()  (optional — adds L9)
        chop:          ChopResult from chop_detector.detect()  (optional — subtracts penalty)
    """
    direction = analysis.get("direction")
    if not direction:
        return {"total": 0, "passes": False, "breakdown": {}}

    price  = analysis["current_price"]
    df     = analysis.get("candles") if analysis.get("candles") is not None else __import__("pandas").DataFrame()
    ticker = analysis["ticker"]

    flow_strategy = strategy_type in ("options_flow", "dark_pool")
    sweep = analysis.get("liquidity_sweep", {})

    # ── L1–L5 (existing layers) ───────────────────────────────
    l1 = _l1_smc(analysis.get("structure", {}), analysis.get("fvgs", {}), analysis.get("obs", {}), direction, price, sweep)

    # Hard gate: require SMC structure for standard strategies only
    if l1 < _L1_MINIMUM and not flow_strategy:
        logger.debug(f"[scorer] {ticker} L1={l1:.0f} < {_L1_MINIMUM} — no SMC backing, skip")
        return {
            "total":     round(l1),
            "passes":    False,
            "breakdown": {"l1_smc": round(l1), "l2_technical": 0,
                          "l3_sentiment": 0, "l4_risk": 0, "l5_mtf": 0},
            "direction":  direction,
            "entry":      analysis.get("entry"),
            "stop_loss":  analysis.get("stop_loss"),
            "target_one": analysis.get("target_one"),
            "target_two": analysis.get("target_two"),
        }

    l2 = _l2_technical(df, direction, strategy_type) if not df.empty else 10.0
    # Fix #1: flow strategies use UW premium-weighted sentiment, not news keywords
    l3 = _l3_flow_sentiment(analysis, direction) if flow_strategy else _l3_sentiment(ticker, direction)
    l4 = _l4_risk(df, ticker) if not df.empty else 5.0
    l5 = 0.0 if flow_strategy else _l5_multiframe(ticker, direction)

    # ── Load adaptive weights (learned by optimizer; defaults on first run) ──
    try:
        from engine.adaptive_weights import get_weights as _get_adaptive_weights
        regime_type = (regime or {}).get("regime_type", "ANY")
        w = _get_adaptive_weights(strategy_type, regime_type)
    except Exception:
        w = STRATEGY_WEIGHTS.get(strategy_type, STRATEGY_WEIGHTS["day_trade"])

    weighted = (
        (l1 / _L_MAX["smc"])       * w["smc"] +
        (l2 / _L_MAX["technical"]) * w["technical"] +
        (l3 / _L_MAX["sentiment"]) * w["sentiment"] +
        (l4 / _L_MAX["risk"])      * w["risk"]
    )
    l5_bonus_max = w.get("l5_bonus", 5.0)
    l5_bonus = (l5 / 15.0) * l5_bonus_max
    # ── Anti-inflation cap: base cannot exceed 85 ─────────────
    # Prevents the engine from auto-scoring near 100 just by stacking
    # L1-L5. Real A+ signals earn the last 5-15 pts via quant layers.
    base_score = min(85.0, weighted + l5_bonus)

    # ── L6–L9 (quant bonus layers, hard-capped at ±10 total) ──
    # Combined quant bonus kept tight: +10 → confirmed institutional,
    # -10 → hostile environment. No layer can dominate.
    quant_bonus = 0.0
    l6_regime    = 0.0
    l7_session   = 0.0
    l8_gamma     = 0.0
    l9_manip     = 0.0

    # Adaptive bonus magnitudes (optimizer tunes these over time)
    l6_pts = w.get("l6_bonus", 8.0)
    l7_pts = w.get("l7_bonus", 6.0)
    l8_pts = w.get("l8_bonus", 8.0)
    l9_pts = w.get("l9_bonus", 8.0)

    if regime is not None:
        from engine.regime_detector import score_for_signal as regime_score
        l6_regime = regime_score(regime, direction)
        quant_bonus += (l6_regime - 50) / 50 * l6_pts

    if session is not None:
        from engine.session_classifier import score_for_signal as session_score
        l7_session = session_score(session, analysis.get("has_catalyst", False), analysis.get("vol_multiple", 1.0))
        quant_bonus += (l7_session - 50) / 50 * l7_pts

    if gamma is not None:
        from engine.gamma_engine import score_for_signal as gamma_score
        l8_gamma = gamma_score(gamma, direction, session.get("is_opex_day", False) if session else False)
        quant_bonus += (l8_gamma - 50) / 50 * l8_pts

    if manipulation is not None:
        from engine.manipulation_detector import score_for_signal as manip_score
        l9_manip = manip_score(manipulation)
        quant_bonus += (l9_manip - 50) / 50 * l9_pts

    # Hard cap: quant bonus is ±10 max — prevents any single layer from dominating
    quant_bonus = max(-10.0, min(10.0, quant_bonus))

    # ── Chop penalty (0 to -15 pts) ───────────────────────────
    # Applied AFTER quant layers so chop can't be masked by bonus pts
    chop_penalty = chop.as_penalty() if chop is not None else 0.0
    chop_score_val = chop.chop_score if chop is not None else 0.0

    normalised = round(min(max(base_score + quant_bonus - chop_penalty, 0), 100))

    # ── Determine threshold ───────────────────────────────────
    if session is not None:
        threshold = session.get("threshold", STRATEGY_THRESHOLDS.get(strategy_type, FIRE_THRESHOLD))
    else:
        threshold = STRATEGY_THRESHOLDS.get(strategy_type, FIRE_THRESHOLD)

    breakdown = {
        "l1_smc":          round(l1),
        "l2_technical":    round(l2),
        "l3_sentiment":    round(l3),
        "l4_risk":         round(l4),
        "l5_mtf":          round(l5),
        "l6_regime":       round(l6_regime),
        "l7_session":      round(l7_session),
        "l8_gamma":        round(l8_gamma),
        "l9_manipulation": round(l9_manip),
        "quant_bonus":     round(quant_bonus, 1),
        "chop_penalty":    round(chop_penalty, 1),
        "base_score":      round(base_score, 1),
        "sweep_confirmed": bool(sweep and sweep.get("swept")),
        "regime_type":     regime.get("regime_type", "") if regime else "",
        "session_mode":    session.get("mode", "") if session else "",
    }

    logger.info(
        f"[scorer] {ticker} [{strategy_type}] total={normalised} threshold={threshold} "
        f"base={base_score:.1f} quant={quant_bonus:+.1f} chop_pen={chop_penalty:.1f} "
        f"(L1={round(l1)} L2={round(l2)} L3={round(l3)} L4={round(l4)} L5={round(l5)} "
        f"L6={round(l6_regime)} L7={round(l7_session)} L8={round(l8_gamma)} "
        f"L9={round(l9_manip)})"
    )

    factors = _factors_from_breakdown(breakdown, direction)
    # Add quant factors
    if regime and regime.get("regime_type") in ("TRENDING_BULL", "TRENDING_BEAR"):
        factors.append(f"Regime: {regime['regime_type'].replace('_', ' ').title()}")
    if gamma and gamma.get("available") and not gamma.get("is_negative_gamma"):
        factors.append("Gamma Positive Zone")
    if manipulation and manipulation.get("is_clean"):
        factors.append("Clean Price Action")
    factors = factors[:4]

    # ── Confidence and Risk grades ──────────────────────────────
    try:
        from engine.setup_lifecycle import (
            classify_confidence_grade,
            classify_risk_grade,
            get_missing_confirmations,
        )
        risk_reward = None
        try:
            entry_p  = analysis.get("entry") or price
            stop_p   = analysis.get("stop_loss") or entry_p
            target_p = analysis.get("target_one") or entry_p
            if entry_p != stop_p:
                risk_reward = abs(target_p - entry_p) / abs(entry_p - stop_p)
        except Exception:
            pass

        regime_type_str = (regime or {}).get("regime_type", "UNKNOWN")
        confidence_grade    = classify_confidence_grade(normalised).value
        risk_grade          = classify_risk_grade(
            normalised,
            risk_reward or 0.0,
            chop_score_val,
            regime_type_str,
        ).value

        # Build a partial score_result proxy for missing-confirmation check
        _score_proxy = {"total": normalised, "breakdown": breakdown, "passes": normalised >= threshold}
        missing_confirmations = get_missing_confirmations(
            analysis, _score_proxy, regime or {}, chop
        )
    except Exception as exc:
        logger.debug(f"[scorer] grade/missing step skipped: {exc}")
        confidence_grade      = "B"
        risk_grade            = "MEDIUM"
        missing_confirmations = []

    return {
        "total":                normalised,
        "passes":               normalised >= threshold,
        "breakdown":            breakdown,
        "confidence_factors":   factors,
        "confidence_grade":     confidence_grade,
        "risk_grade":           risk_grade,
        "missing_confirmations": missing_confirmations,
        "chop_score":           round(chop_score_val, 1),
        "direction":            direction,
        "entry":                analysis.get("entry"),
        "stop_loss":            analysis.get("stop_loss"),
        "target_one":           analysis.get("target_one"),
        "target_two":           analysis.get("target_two"),
        "threshold":            threshold,
    }
