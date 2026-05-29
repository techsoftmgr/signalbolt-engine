"""
Quant Score Service — retail-friendly quantitative analysis dashboard.

Computes 8 scores per ticker and groups them into insight buckets:
  • Top Momentum Stocks
  • Best Pullback Setups
  • Breakout Candidates
  • High Relative Volume
  • VWAP Reclaim candidates
  • Oversold Bounce candidates

Final score formula:
  finalQuantScore = 0.25*trend + 0.25*momentum + 0.20*volume
                  + 0.15*breakout + 0.15*meanReversion - 0.20*risk

Cache TTL: 60 seconds (QUANT_CACHE_TTL env var).
Language: never "buy/sell", always "setup detected / watch / avoid".
"""

import logging
import time
import os
from typing import Optional

import numpy as np

logger = logging.getLogger("signalbolt.quant")

# ── Cache ─────────────────────────────────────────────────────────────────────
_cache: dict       = {}
_cache_ts: float   = 0.0
_CACHE_TTL: int    = int(os.environ.get("QUANT_CACHE_TTL", "60"))

DEFAULT_TICKERS: list[str] = [
    "SPY", "QQQ", "IWM", "DIA",
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "TSLA", "AMD",
    "COIN", "PLTR", "MSTR", "HOOD", "RBLX", "UBER", "ABNB",
    "JPM", "GS", "XOM", "CVX",
    "MARA", "RIOT", "CLSK", "MRNA", "BNTX",
]


def _safe(val, default: float = 0.0) -> float:
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _normalize(value: float, lo: float, hi: float) -> float:
    """Map value from [lo, hi] to [0, 100]."""
    if hi == lo:
        return 50.0
    return float(np.clip((value - lo) / (hi - lo) * 100, 0, 100))


def get_quant_dashboard(symbols: Optional[list[str]] = None) -> dict:
    """
    Returns the full quant dashboard payload:
    {
      marketRegime: {...},
      topMomentum:  [...],
      pullbacks:    [...],
      breakouts:    [...],
      highVolume:   [...],
      vwapReclaim:  [...],
      oversoldBounce: [...],
    }
    Results are cached for QUANT_CACHE_TTL seconds.
    """
    global _cache, _cache_ts

    now = time.monotonic()
    if now - _cache_ts < _CACHE_TTL and _cache:
        return _cache

    result = _build_dashboard(symbols or DEFAULT_TICKERS)
    if result:
        try:
            _enrich_breakouts(result)
        except Exception as e:
            logger.debug(f"[quant] breakout enrich skipped: {e}")
        _cache    = result
        _cache_ts = now
    return result


def _enrich_breakouts(result: dict) -> None:
    """Add lifecycle state + catalyst + R:R to each breakout row, plus a top-level
    Watch Accuracy summary. Best-effort — never breaks the dashboard."""
    import os
    from datetime import datetime, timezone

    rows = result.get("breakouts") or []
    sb = None
    try:
        from supabase import create_client
        from engine.runner import _supabase_key
        sb = create_client(os.environ["SUPABASE_URL"], _supabase_key())
    except Exception:
        sb = None

    # Lifecycle state per ticker (open episodes).
    state_by: dict = {}
    if sb is not None and rows:
        try:
            eps = (sb.table("breakout_watch_history")
                     .select("ticker,state,entered_at")
                     .is_("exited_at", "null").execute().data) or []
            now = datetime.now(timezone.utc)
            for e in eps:
                age = None
                try:
                    age = int((now - datetime.fromisoformat(
                        e["entered_at"].replace("Z", "+00:00"))).total_seconds() // 60)
                except Exception:
                    pass
                state_by[e["ticker"]] = {"state": e.get("state"), "ageMin": age}
        except Exception:
            pass

    try:
        from engine.runner import _has_recent_news
    except Exception:
        _has_recent_news = lambda _t: False

    for r in rows:
        tk = r.get("ticker"); lvl = r.get("breakoutLevel"); px = r.get("price")
        st = state_by.get(tk) or {}
        r["watchState"] = st.get("state") or (
            "TRIGGERED" if (lvl and px and px > lvl) else "WATCHING")
        r["watchAgeMin"] = st.get("ageMin")
        try:
            r["catalyst"] = bool(_has_recent_news(tk))
        except Exception:
            r["catalyst"] = False
        # Rough breakout R:R — entry≈price, stop just below the level, target +3%.
        try:
            if lvl and px:
                stop = round(lvl * 0.99, 2); target = round(lvl * 1.03, 2)
                risk = px - stop
                r["stopRef"] = stop
                r["targetRef"] = target
                r["riskReward"] = round((target - px) / risk, 1) if risk > 0 else None
        except Exception:
            pass

    # Top-level Watch Accuracy (judged episodes, last 14d).
    if sb is not None:
        try:
            from engine import breakout_validator
            result["breakoutWatch"] = breakout_validator.watch_accuracy(sb, days=14)
        except Exception:
            result["breakoutWatch"] = {"judged": 0, "accuracy_pct": None}
    else:
        result["breakoutWatch"] = {"judged": 0, "accuracy_pct": None}


def _build_dashboard(tickers: list[str]) -> dict:
    from engine.alpaca_client import get_multi_bars, get_latest_prices
    from engine import regime_detector

    try:
        daily_bars    = get_multi_bars(tickers, timeframe="1Day", days=25)
        intraday_bars = get_multi_bars(tickers, timeframe="5Min", days=2)
        latest_prices = get_latest_prices(tickers)

        scored: list[dict] = []
        for ticker in tickers:
            try:
                row = _score_ticker(
                    ticker,
                    latest_prices.get(ticker),
                    daily_bars.get(ticker),
                    intraday_bars.get(ticker),
                )
                if row:
                    scored.append(row)
            except Exception as e:
                logger.debug(f"[quant] {ticker}: {e}")

        # ── Market regime (reuse existing detector) ───────────────────────────
        try:
            regime_raw = regime_detector.detect()
            regime_map = {
                "TRENDING_BULL":   "Bullish",
                "TRENDING_BEAR":   "Bearish",
                "CHOPPY":          "Choppy",
                "HIGH_VOLATILITY": "Volatile",
                "RANGING":         "Risk-Off",
                "NEUTRAL":         "Neutral",
            }
            regime_label = regime_map.get(regime_raw.get("regime", "NEUTRAL"), "Neutral")
            regime_desc  = _regime_description(regime_label)
        except Exception:
            regime_label = "Neutral"
            regime_desc  = "Market regime data unavailable."

        # ── Bucket the scored stocks ──────────────────────────────────────────
        sorted_by_quant = sorted(scored, key=lambda x: x["finalQuantScore"], reverse=True)

        # Bucket thresholds intentionally permissive so dashboard stays useful
        # outside market hours when realized volume drops the scores. During
        # live RTH the thresholds still cluster the strongest setups at the top.
        top_momentum = [
            x for x in sorted_by_quant
            if x["momentumScore"] >= 55 and x["finalQuantScore"] >= 45
        ][:6]

        pullbacks = [
            x for x in scored
            if x["setupType"] == "pullback"
        ][:6]

        breakouts = [
            x for x in scored
            if x["setupType"] == "breakout"
        ][:6]

        high_volume = sorted(
            [x for x in scored if x["volumeScore"] >= 50],
            key=lambda x: x["volumeScore"], reverse=True,
        )[:6]

        vwap_reclaim = [
            x for x in scored
            if x["setupType"] == "vwap_reclaim"
        ][:6]

        oversold_bounce = [
            x for x in scored
            if x["setupType"] == "oversold_bounce"
        ][:6]

        return {
            "marketRegime": {
                "label":       regime_label,
                "description": regime_desc,
                "color":       _regime_color(regime_label),
            },
            "topMomentum":    top_momentum,
            "pullbacks":      pullbacks,
            "breakouts":      breakouts,
            "highVolume":     high_volume,
            "vwapReclaim":    vwap_reclaim,
            "oversoldBounce": oversold_bounce,
            "allScored":      sorted_by_quant[:20],
        }

    except Exception as e:
        logger.error(f"[quant] _build_dashboard failed: {e}")
        return {}


def _score_ticker(
    ticker: str,
    latest_price: Optional[float],
    daily_df,
    intraday_df,
) -> Optional[dict]:
    """Compute all quant scores for one ticker. Returns None if data insufficient."""
    import pandas as pd
    from datetime import datetime, timezone

    if daily_df is None or len(daily_df) < 10:
        return None

    closes = daily_df["close"].values.astype(float)
    volumes = daily_df["volume"].values.astype(float)

    current = _safe(latest_price or closes[-1])
    if current <= 0:
        return None

    # ── Trend Score (0-100) ───────────────────────────────────────────────────
    # Based on: price vs 10-day MA, price vs 20-day MA, MA slope
    ma10 = float(np.mean(closes[-10:])) if len(closes) >= 10 else current
    ma20 = float(np.mean(closes[-20:])) if len(closes) >= 20 else current
    ma5  = float(np.mean(closes[-5:]))  if len(closes) >= 5  else current

    trend_raw  = 0.0
    trend_raw += 40 if current > ma20 else -40     # above/below 20MA
    trend_raw += 30 if current > ma10 else -30     # above/below 10MA
    trend_raw += 30 if ma5 > ma20    else -30      # short MA > long MA = uptrend
    trend_score = _normalize(trend_raw, -100, 100)

    # ── Momentum Score (0-100) ────────────────────────────────────────────────
    # RSI(14) + rate of change
    if len(closes) >= 15:
        deltas = np.diff(closes[-15:])
        gains  = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = float(np.mean(gains)) if gains.any() else 0
        avg_loss = float(np.mean(losses)) if losses.any() else 1e-10
        rs  = avg_gain / avg_loss
        rsi = 100 - 100 / (1 + rs)
    else:
        rsi = 50.0

    roc_5  = (current - _safe(closes[-5]))  / _safe(closes[-5],  1) * 100 if len(closes) >= 5  else 0
    roc_10 = (current - _safe(closes[-10])) / _safe(closes[-10], 1) * 100 if len(closes) >= 10 else 0

    momentum_raw = rsi * 0.5 + _normalize(roc_5,  -10, 10) * 0.3 + _normalize(roc_10, -15, 15) * 0.2
    momentum_score = float(np.clip(momentum_raw, 0, 100))

    # ── Volume Score (0-100) ──────────────────────────────────────────────────
    avg_vol    = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(np.mean(volumes))
    today_vol  = 0.0
    if intraday_df is not None and not intraday_df.empty:
        today = datetime.now(timezone.utc).date()
        try:
            today_mask = intraday_df.index.date == today
            today_vol  = _safe(intraday_df[today_mask]["volume"].sum())
        except Exception:
            today_vol  = _safe(intraday_df["volume"].tail(78).sum())

    if avg_vol > 0 and today_vol > 0:
        now_utc = datetime.now(timezone.utc)
        elapsed  = max(1, now_utc.hour * 60 + now_utc.minute - (13 * 60 + 30))
        proj_vol = today_vol / max(0.05, elapsed / 390)
        rel_vol  = proj_vol / avg_vol
    else:
        rel_vol = 1.0

    volume_score = float(np.clip(_normalize(rel_vol, 0.5, 3.0), 0, 100))

    # ── Breakout Score (0-100) ────────────────────────────────────────────────
    # How close is price to 20-day high? Above = breakout candidate
    high_20 = float(np.max(daily_df["high"].values[-20:])) if "high" in daily_df else float(np.max(closes[-20:]))
    dist_to_high_pct = (current - high_20) / high_20 * 100  # negative = below high
    breakout_score = _normalize(dist_to_high_pct, -5, 2)

    # ── Mean Reversion Score (0-100) ──────────────────────────────────────────
    # RSI < 35 + price well below MA = oversold bounce candidate
    oversold_score = _normalize(100 - rsi, 0, 100)  # higher when RSI low
    dist_below_ma  = (ma20 - current) / ma20 * 100    # positive = below MA
    mean_rev_raw   = oversold_score * 0.6 + _normalize(dist_below_ma, -3, 5) * 0.4
    mean_reversion_score = float(np.clip(mean_rev_raw, 0, 100))

    # ── Risk Score (0-100; higher = riskier) ─────────────────────────────────
    # Based on ATR% (average true range as % of price).
    # True Range for bar i needs prev_close (close[i-1]). For 14 bars of
    # TR we need 14 prev_closes — closes[-15:-1] is the correct slice
    # (length 14, aligned with highs[-14:]). Previous code used
    # closes[-14:-1] which is length 13 → numpy broadcast failure that
    # silently zeroed out the entire quant dashboard.
    if len(closes) >= 15 and "high" in daily_df and "low" in daily_df:
        highs       = daily_df["high"].values[-14:].astype(float)
        lows        = daily_df["low"].values[-14:].astype(float)
        prev_closes = closes[-15:-1]                # length 14, aligned with highs/lows
        tr = np.maximum.reduce([
            highs - lows,
            np.abs(highs - prev_closes),
            np.abs(lows  - prev_closes),
        ])
        atr     = float(np.mean(tr))
        atr_pct = (atr / current * 100) if current > 0 else 2.0
    else:
        atr_pct = 2.0  # default 2% if not enough history

    risk_score = _normalize(atr_pct, 0.5, 5.0)

    # ── Final Quant Score ─────────────────────────────────────────────────────
    final = (
        0.25 * trend_score
      + 0.25 * momentum_score
      + 0.20 * volume_score
      + 0.15 * breakout_score
      + 0.15 * mean_reversion_score
      - 0.20 * risk_score
    )
    final_score = float(np.clip(round(final), 0, 100))

    # ── VWAP for intraday context ─────────────────────────────────────────────
    vwap = None
    if intraday_df is not None and not intraday_df.empty:
        try:
            today = datetime.now(timezone.utc).date()
            t_df  = intraday_df[intraday_df.index.date == today]
            if not t_df.empty:
                vol_col = t_df["volume"]
                typ_col = (t_df["high"] + t_df["low"] + t_df["close"]) / 3
                total_v = _safe(vol_col.sum())
                if total_v > 0:
                    vwap = round(_safe((typ_col * vol_col).sum() / total_v), 2)
        except Exception:
            pass

    # ── Setup type classification ─────────────────────────────────────────────
    setup_type, setup_reason = _classify_setup(
        current, ma20, rsi, rel_vol, breakout_score, mean_reversion_score, vwap,
    )

    # ── Watch status ──────────────────────────────────────────────────────────
    if final_score >= 65 and risk_score < 60:
        watch_status = "actionable"
    elif final_score >= 45 or setup_type != "none":
        watch_status = "watch"
    else:
        watch_status = "avoid"

    # ── Risk label ────────────────────────────────────────────────────────────
    if risk_score < 35:
        risk_label = "Low"
    elif risk_score < 65:
        risk_label = "Medium"
    else:
        risk_label = "High"

    return {
        "ticker":              ticker,
        "price":               round(current, 2),
        "vwap":                vwap,
        "rsi":                 round(rsi, 1),
        "relativeVolume":      round(rel_vol, 2),
        # Component scores (0-100)
        "trendScore":          round(trend_score, 1),
        "momentumScore":       round(momentum_score, 1),
        "volumeScore":         round(volume_score, 1),
        "breakoutScore":       round(breakout_score, 1),
        "breakoutLevel":       round(high_20, 2),        # 20-day high being tested
        "distToBreakoutPct":   round(dist_to_high_pct, 2),  # negative = below the high
        "meanReversionScore":  round(mean_reversion_score, 1),
        "riskScore":           round(risk_score, 1),
        "finalQuantScore":     round(final_score, 1),
        # Qualitative outputs
        "setupType":           setup_type,
        "setupReason":         setup_reason,
        "watchStatus":         watch_status,
        "riskLevel":           risk_label,
        # Confidence: maps final score to "low/medium/high" label
        "confidence":          "high" if final_score >= 70 else ("medium" if final_score >= 50 else "low"),
    }


def _classify_setup(
    price: float, ma20: float, rsi: float, rel_vol: float,
    breakout_score: float, mean_rev_score: float, vwap: Optional[float],
) -> tuple[str, str]:
    """Return (setup_type, plain-English reason)."""

    # Momentum: strong RSI + above MA. Volume requirement loosened so
    # after-hours / pre-market still surfaces top movers from RTH session.
    if rsi >= 55 and price > ma20 and rel_vol >= 1.1:
        return "momentum", f"RSI {rsi:.0f} — momentum above 20-day average"

    # Breakout candidate: near 20-day high. Volume requirement loosened
    # for the same reason as momentum.
    if breakout_score >= 60 and rel_vol >= 1.0:
        return "breakout", "Price approaching 20-day high — breakout candidate"

    # VWAP reclaim: price just crossed back above VWAP
    if vwap and price > vwap and price < vwap * 1.005:
        return "vwap_reclaim", "Price reclaiming VWAP — intraday structure improving"

    # Pullback: healthy stock pulling back to MA
    if price > ma20 * 0.97 and price < ma20 * 1.01 and rsi < 50:
        return "pullback", f"Pulling back to 20-day average — possible support zone at {ma20:.2f}"

    # Oversold bounce: RSI < 35, well below MA
    if rsi <= 35 and price < ma20 * 0.97:
        return "oversold_bounce", f"RSI {rsi:.0f} — oversold setup detected, watching for stabilisation"

    return "none", "No specific setup detected — monitoring"


def _regime_description(label: str) -> str:
    descriptions = {
        "Bullish":  "Broad market trending higher. Momentum setups historically perform better.",
        "Bearish":  "Market under pressure. Risk management elevated. Setups may fail faster.",
        "Choppy":   "No clear direction. Breakout setups may whipsaw. Patience recommended.",
        "Volatile": "High volatility detected. Wider price swings expected. Sizing carefully.",
        "Risk-Off": "Markets in risk-off mode. Defensive posture. Setups carry higher uncertainty.",
        "Neutral":  "Mixed signals. No strong directional bias detected.",
    }
    return descriptions.get(label, "Market data being analysed.")


def _regime_color(label: str) -> str:
    colors = {
        "Bullish": "#22C55E", "Bearish": "#EF4444",
        "Choppy": "#F59E0B",  "Volatile": "#F59E0B",
        "Risk-Off": "#6366F1", "Neutral": "#9CA3AF",
    }
    return colors.get(label, "#9CA3AF")
