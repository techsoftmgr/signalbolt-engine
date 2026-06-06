"""
Module #5 — Watchlist Intelligence.

Additive layer over a user's watchlist: per name, relative strength vs SPY, trend
state, and earnings proximity → a priority score ("what to watch today") with a
plain-English reason, plus an Opportunity Board (interesting situations, NOT
signals). Read-only; never raises. Pure scoring → unit-tested.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("signalbolt.phase2.watchlist_intel")


def _priority(rel_strength, trend_change, earnings_soon, near_level):
    """0-100 'watch me today' urgency. Pure."""
    s = 20.0
    if rel_strength is not None:
        s += min(25.0, abs(float(rel_strength)) * 2.5)        # big rel move (either way)
    if trend_change:
        s += 25.0                                             # flipped trend today
    if earnings_soon:
        s += 25.0                                             # event imminent
    if near_level:
        s += 20.0                                             # at a key level
    return int(round(min(100.0, s)))


def _why(rel_strength, trend, trend_change, earnings_soon, near_level):
    bits = []
    if earnings_soon:
        bits.append("earnings imminent")
    if trend_change:
        bits.append(f"trend just turned {trend}")
    if near_level:
        bits.append("at a key level")
    if rel_strength is not None and abs(rel_strength) >= 3:
        bits.append(f"{'out' if rel_strength > 0 else 'under'}performing SPY by {abs(round(rel_strength,1))}% (5d)")
    return "; ".join(bits) if bits else "no notable change"


def compute(sb, tickers: list, days: int = 40) -> dict:
    """Rank a watchlist by what's worth watching today. Never raises."""
    try:
        tickers = [t.upper() for t in (tickers or []) if t][:80]
        if not tickers:
            return {"enabled": True, "items": [], "note": "Empty watchlist."}
        from engine.alpaca_client import get_multi_bars
        bars = get_multi_bars(tickers + ["SPY"], "1Day", days) or {}
        spy = bars.get("SPY")
        spy_5d = None
        if spy is not None and len(spy) > 6:
            spy_5d = (float(spy["close"].iloc[-1]) / float(spy["close"].iloc[-6]) - 1) * 100

        earn_soon = set()
        try:
            from engine import earnings_service
            wk = earnings_service.get_weekly_earnings(tickers) or {}
            for e in (wk.get("events") or wk.get("items") or []):
                if e.get("ticker"):
                    earn_soon.add(e["ticker"].upper())
        except Exception:
            pass

        items = []
        for tk in tickers:
            df = bars.get(tk)
            if df is None or len(df) < 25:
                continue
            c = df["close"]
            ret5d = (float(c.iloc[-1]) / float(c.iloc[-6]) - 1) * 100 if len(c) > 6 else None
            rel = (ret5d - spy_5d) if (ret5d is not None and spy_5d is not None) else None
            sma20 = c.rolling(20).mean()
            last, prev = float(c.iloc[-1]), float(c.iloc[-2])
            above = last > float(sma20.iloc[-1])
            trend_change = above != (prev > float(sma20.iloc[-2]))
            trend = "up" if above else "down"
            hi20 = float(df["high"].iloc[-21:-1].max())
            lo20 = float(df["low"].iloc[-21:-1].min())
            near_level = (abs(last - hi20) / last <= 0.02) or (abs(last - lo20) / last <= 0.02)
            earnings = tk in earn_soon
            items.append({
                "ticker": tk, "priority": _priority(rel, trend_change, earnings, near_level),
                "rel_strength_5d": round(rel, 1) if rel is not None else None,
                "trend": trend, "trend_changed": trend_change,
                "earnings_soon": earnings, "near_key_level": near_level,
                "why": _why(rel, trend, trend_change, earnings, near_level),
            })
        items.sort(key=lambda x: -x["priority"])
        # opportunity board = the most interesting situations (priority >= 50)
        board = [{"ticker": i["ticker"], "why": i["why"], "priority": i["priority"]}
                 for i in items if i["priority"] >= 50][:8]
        return {"enabled": True, "count": len(items), "ranked": items, "opportunity_board": board}
    except Exception as e:
        logger.error(f"[watchlist_intel] failed: {e}")
        return {"enabled": True, "ranked": [], "error": str(e)}
