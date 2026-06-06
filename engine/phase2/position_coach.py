"""
Module #2 — AI Position Coach.

User passes a ticker (+ optional entry/size/current/account/risk tolerance). We
return a position assessment: trend, risk/volatility level, earnings & news risk,
support/resistance zones, a plain-English status, and a Scenario Engine
(bull/base/bear with trigger levels). Reuses chart_read for levels/scenarios.
Read-only; never raises; educational wording only (no advice language).
"""
from __future__ import annotations

import logging

logger = logging.getLogger("signalbolt.phase2.position_coach")


def _risk_level(unreal_pct, stop_dist_pct):
    """Pure: position risk label from unrealized P&L + distance to a sensible stop."""
    if stop_dist_pct is None:
        return "UNKNOWN"
    if stop_dist_pct >= 10:
        return "HIGH"
    if stop_dist_pct >= 5:
        return "MODERATE"
    return "CONTAINED"


def _vol_level(atr_pct):
    if atr_pct is None:
        return "UNKNOWN"
    return "HIGH" if atr_pct >= 4 else "ELEVATED" if atr_pct >= 2.5 else "NORMAL"


def _status_text(trend, vol_level, earnings_soon, near_support, near_resistance):
    bits = [f"Trend is {trend}."]
    if near_support:
        bits.append("Price is holding near a key support zone.")
    elif near_resistance:
        bits.append("Price is pressing into resistance.")
    if vol_level in ("HIGH", "ELEVATED"):
        bits.append(f"Volatility is {vol_level.lower()}.")
    if earnings_soon:
        bits.append("Earnings are approaching — event risk is elevated.")
    return " ".join(bits)


def assess(ticker: str, entry=None, size=None, current=None,
           account=None, risk_tolerance: str = "moderate") -> dict:
    """Position assessment + scenario engine. Never raises."""
    try:
        tk = (ticker or "").upper()
        if not tk:
            return {"enabled": True, "error": "ticker required"}
        from engine.alpaca_client import get_bars
        df = get_bars(tk, "1Day", 80)
        if df is None or len(df) < 30:
            return {"enabled": True, "ticker": tk, "error": "insufficient data"}
        import numpy as np
        c = df["close"]; last = float(c.iloc[-1])
        cur = float(current) if current else last
        sma20, sma50 = float(c.rolling(20).mean().iloc[-1]), float(c.rolling(50).mean().iloc[-1])
        trend = "constructive (above 20 & 50-DMA)" if cur > sma20 > sma50 \
            else "weak (below key averages)" if cur < sma20 and cur < sma50 else "mixed"
        h, l, cl = df["high"].values, df["low"].values, df["close"].values
        tr = np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - cl[:-1]), abs(l[1:] - cl[:-1])))
        atr = float(tr[-14:].mean()); atr_pct = atr / cur * 100 if cur else None
        sup = round(float(df["low"].iloc[-21:-1].min()), 2)
        res = round(float(df["high"].iloc[-21:-1].max()), 2)
        near_sup = abs(cur - sup) / cur <= 0.03
        near_res = abs(cur - res) / cur <= 0.03

        earnings_soon = False
        try:
            from engine import earnings_service
            nx = earnings_service.get_next_earnings(tk, horizon_days=10)
            earnings_soon = bool(nx)
        except Exception:
            pass

        unreal = ((cur - float(entry)) / float(entry) * 100) if entry else None
        stop = round(cur - 1.5 * atr, 2)
        stop_dist = (cur - stop) / cur * 100 if cur else None

        # ── Scenario Engine: bull / base / bear (reuse chart_read levels) ──
        scenarios = {
            "bull": {"trigger": f"reclaim/hold above ${res}",
                     "opportunity": f"momentum extension toward ${round(res * 1.06, 2)}",
                     "risk": "false breakout / fade back below the level"},
            "base": {"trigger": f"range between ${sup} and ${res}",
                     "opportunity": "patience — let price pick a side",
                     "risk": "chop / whipsaw losses"},
            "bear": {"trigger": f"lose ${sup} on a daily close",
                     "opportunity": f"downside toward ${round(sup * 0.94, 2)}",
                     "risk": "support holds and squeezes shorts"},
        }
        vol_level = _vol_level(atr_pct)
        return {
            "enabled": True, "ticker": tk, "current": round(cur, 2),
            "trend": trend, "risk_level": _risk_level(unreal, stop_dist),
            "volatility_level": vol_level,
            "atr_pct": round(atr_pct, 1) if atr_pct else None,
            "earnings_risk": "ELEVATED" if earnings_soon else "LOW",
            "support": sup, "resistance": res,
            "suggested_stop_ref": stop,
            "unrealized_pct": round(unreal, 1) if unreal is not None else None,
            "status": _status_text(trend, vol_level, earnings_soon, near_sup, near_res),
            "scenarios": scenarios,
            "disclaimer": "Educational analysis only — not financial advice.",
        }
    except Exception as e:
        logger.error(f"[position_coach] {ticker} failed: {e}")
        return {"enabled": True, "ticker": ticker, "error": str(e)}
