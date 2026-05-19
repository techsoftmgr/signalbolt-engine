"""
Market Regime Detector
======================
Classifies current market environment using:
  - VIX level and intraday change
  - SPY vs 200-day moving average
  - ADX (trend strength)
  - Fear & Greed (derived from VIX)
  - Put/Call ratio (from yfinance or UW)

Regime types:
  TRENDING_BULL   — ADX > 25, SPY above 200MA
  TRENDING_BEAR   — ADX > 25, SPY below 200MA
  RANGING         — ADX < 20
  HIGH_VOL        — VIX 25-30
  LOW_VOL         — VIX < 15
  PANIC           — VIX > 30 or spike > 10% intraday
  RISK_OFF        — bonds outperforming, defensive rotation

Used by: runner.py (pre-scan gate), scorer.py (L6 bonus layer)
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import yfinance as yf

logger = logging.getLogger("signalbolt.regime")

# ── Thresholds ────────────────────────────────────────────────
VIX_PANIC     = 30.0
VIX_HIGH      = 25.0
VIX_LOW       = 15.0
VIX_SPIKE     = 0.10   # 10% intraday spike → pause everything
ADX_TRENDING  = 25.0
ADX_RANGING   = 20.0


def _fetch_vix() -> dict:
    """Fetch VIX current value and prev close from yfinance."""
    try:
        ticker = yf.Ticker("^VIX")
        info   = ticker.fast_info
        hist   = ticker.history(period="2d", interval="1d")

        vix_now   = float(info.last_price or 18.0)
        prev_close = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else vix_now
        return {"vix": vix_now, "prev_close": prev_close}
    except Exception as e:
        logger.debug(f"VIX fetch error: {e}")
        return {"vix": 18.0, "prev_close": 18.0}


def _fetch_spy_vs_200ma() -> dict:
    """
    Check if SPY is above its 200-day MA.
    Also returns raw history for reuse in _fetch_risk_off_signal so we
    don't fetch SPY a second time in the same detect() call.
    """
    try:
        hist = yf.Ticker("SPY").history(period="210d", interval="1d")
        if len(hist) < 200:
            return {"above_200ma": True, "spy_price": 0, "ma200": 0, "adx": 0, "_hist": None}

        closes  = hist["Close"].tolist()
        highs   = hist["High"].tolist()
        lows    = hist["Low"].tolist()
        price   = closes[-1]
        ma200   = sum(closes[-200:]) / 200

        # ADX(14) from last 30 bars
        adx = _compute_adx(highs[-30:], lows[-30:], closes[-30:])

        return {
            "above_200ma": price > ma200,
            "spy_price":   round(price, 2),
            "ma200":       round(ma200, 2),
            "adx":         round(adx, 1),
            "_hist":       hist,   # reused by _fetch_risk_off_signal — no double fetch
        }
    except Exception as e:
        logger.debug(f"SPY/200MA fetch error: {e}")
        return {"above_200ma": True, "spy_price": 0, "ma200": 0, "adx": 0, "_hist": None}


def _compute_adx(highs: list, lows: list, closes: list, period: int = 14) -> float:
    """Compute ADX(14) from price lists."""
    if len(closes) < period + 1:
        return 20.0

    dx_values = []
    for i in range(1, len(closes)):
        up_move   = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        plus_dm   = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm  = down_move if (down_move > up_move and down_move > 0) else 0.0

        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        if tr > 0:
            plus_di  = (plus_dm / tr) * 100
            minus_di = (minus_dm / tr) * 100
            di_diff  = abs(plus_di - minus_di)
            di_sum   = plus_di + minus_di
            dx_values.append((di_diff / di_sum) * 100 if di_sum > 0 else 0)

    if len(dx_values) < period:
        return 20.0

    adx = sum(dx_values[-period:]) / period
    return adx


def _fetch_risk_off_signal(spy_hist=None) -> bool:
    """
    Check if market is in risk-off rotation:
    bonds (TLT) outperforming SPY over 5 days AND TLT 5-day return > +1%.

    Fix #6: accepts spy_hist from _fetch_spy_vs_200ma so SPY is not
    fetched twice per detect() call. Falls back to fresh fetch if not provided.
    """
    try:
        # Reuse SPY history passed from detect() — tail(5) gives the 5-day window
        if spy_hist is not None and len(spy_hist) >= 2:
            spy_5d = spy_hist.tail(5)
        else:
            spy_5d = yf.Ticker("SPY").history(period="5d", interval="1d")

        tlt_hist = yf.Ticker("TLT").history(period="5d", interval="1d")
        if len(spy_5d) < 2 or len(tlt_hist) < 2:
            return False
        spy_ret = (float(spy_5d["Close"].iloc[-1]) / float(spy_5d["Close"].iloc[0])) - 1
        tlt_ret = (float(tlt_hist["Close"].iloc[-1]) / float(tlt_hist["Close"].iloc[0])) - 1
        # True risk-off: bonds up meaningfully AND outpacing equities by >1%
        return tlt_ret > 0.01 and tlt_ret > spy_ret + 0.01
    except Exception as e:
        logger.debug(f"Risk-off fetch error: {e}")
        return False


def _classify(vix: float, vix_change_pct: float, above_200ma: bool, adx: float) -> str:
    """Return regime type string."""
    # Panic overrides everything
    if vix > VIX_PANIC or vix_change_pct > VIX_SPIKE:
        return "PANIC"
    if vix > VIX_HIGH:
        return "HIGH_VOL"
    if vix < VIX_LOW:
        return "LOW_VOL"
    if adx > ADX_TRENDING:
        return "TRENDING_BULL" if above_200ma else "TRENDING_BEAR"
    return "RANGING"


def detect() -> dict:
    """
    Fetch live data and return full regime snapshot.

    Returns:
        {
          "regime_type":     str,
          "vix":             float,
          "vix_change_pct":  float,
          "adx":             float,
          "above_200ma":     bool,
          "spy_price":       float,
          "ma200":           float,
          "fear_greed":      int,     # 0-100 (derived from VIX)
          "blocked":         bool,
          "block_reason":    str,
        }
    """
    vix_data = _fetch_vix()
    spy_data = _fetch_spy_vs_200ma()

    vix     = vix_data["vix"]
    prev    = vix_data["prev_close"]
    vix_chg = (vix - prev) / prev if prev > 0 else 0.0

    regime_type = _classify(vix, vix_chg, spy_data["above_200ma"], spy_data["adx"])

    # Upgrade to RISK_OFF — reuse SPY hist already fetched above (no double fetch)
    if regime_type in ("RANGING", "LOW_VOL", "TRENDING_BULL") and _fetch_risk_off_signal(spy_hist=spy_data.get("_hist")):
        regime_type = "RISK_OFF"
        logger.info("[regime] RISK_OFF detected — TLT outperforming SPY (bonds flight)")

    # Derive Fear & Greed from VIX (inverted and scaled)
    # VIX 10 → ~90 (greed), VIX 40 → ~10 (fear)
    fear_greed = max(0, min(100, int(100 - (vix - 10) * 3)))

    blocked = regime_type == "PANIC" or vix_chg > VIX_SPIKE
    block_reason = ""
    if regime_type == "PANIC":
        block_reason = f"PANIC regime: VIX={vix:.1f} — all long signals blocked"
    elif vix_chg > VIX_SPIKE:
        block_reason = f"VIX spiked {vix_chg * 100:.1f}% intraday — signals paused"

    result = {
        "regime_type":    regime_type,
        "vix":            round(vix, 2),
        "vix_change_pct": round(vix_chg, 4),
        "adx":            spy_data["adx"],
        "above_200ma":    spy_data["above_200ma"],
        "spy_price":      spy_data["spy_price"],
        "ma200":          spy_data["ma200"],
        "fear_greed":     fear_greed,
        "blocked":        blocked,
        "block_reason":   block_reason,
    }

    logger.info(
        f"[regime] {regime_type} | VIX={vix:.1f} ({vix_chg:+.1%}) | "
        f"SPY {'>' if spy_data['above_200ma'] else '<'} 200MA | ADX={spy_data['adx']:.0f}"
    )
    return result


def score_for_signal(regime: dict, direction: str) -> float:
    """
    Return a 0-100 score contribution based on regime alignment with signal direction.
    Used as L6 bonus in scorer.py.
    """
    rtype = regime.get("regime_type", "RANGING")
    vix   = regime.get("vix", 18.0)
    vix_chg = regime.get("vix_change_pct", 0.0)

    base = {
        "PANIC":         0   if direction == "LONG" else 30,
        "HIGH_VOL":      50  if direction == "LONG" else 65,
        "TRENDING_BULL": 90  if direction == "LONG" else 40,
        "TRENDING_BEAR": 40  if direction == "LONG" else 90,
        "RANGING":       62,
        "LOW_VOL":       58,
        "RISK_OFF":      40  if direction == "LONG" else 70,
    }.get(rtype, 65)

    # VIX adjustment
    if vix > 20:       base -= 8
    if vix_chg > 0.05: base -= 5   # rising fast
    if vix_chg < -0.05: base += 4  # falling = calm

    return max(0.0, min(100.0, float(base)))


def get_sl_adjustment(regime: dict) -> float:
    """Return SL width multiplier based on regime (1.0 = no change)."""
    rtype = regime.get("regime_type", "RANGING")
    vix   = regime.get("vix", 18.0)
    if rtype == "PANIC":    return 1.30
    if rtype == "HIGH_VOL": return 1.20
    if vix > 20:            return 1.15
    if rtype == "LOW_VOL":  return 0.90
    return 1.0
