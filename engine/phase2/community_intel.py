"""
Module #6 — Community Intelligence (Hype vs Reality).

Additive layer over the existing community data: for each trending ticker, an
explicit Hype Score (0-100, social activity/velocity) and Reality Score (0-100,
price + volume + trend confirmation), plus a plain-English explanation of the
gap. Read-only; self-contained (reads social_snapshots + bars); never raises.
Pure scoring → unit-tested.
"""
from __future__ import annotations

import logging
from collections import defaultdict

logger = logging.getLogger("signalbolt.phase2.community_intel")


def _hype(velocity_pct, mentions):
    v = 0.0 if velocity_pct is None else max(0.0, min(60.0, float(velocity_pct) * 0.6))
    m = 0.0 if not mentions else min(40.0, float(mentions) ** 0.5 * 4)
    return int(round(min(100.0, v + m)))


def _reality(ret_5d, vol_ratio, above_ma):
    s = 50.0
    if ret_5d is not None:
        s += max(-35.0, min(35.0, float(ret_5d) * 3.0))      # price confirms?
    if vol_ratio is not None:
        s += max(-10.0, min(20.0, (float(vol_ratio) - 1.0) * 20.0))  # volume confirms?
    if above_ma is True:
        s += 10.0
    elif above_ma is False:
        s -= 10.0
    return int(round(max(0.0, min(100.0, s))))


def _explain(hype, reality):
    if reality >= 60 and hype >= 55:
        return "Real momentum — the buzz is confirmed by price and volume."
    if hype >= 55 and reality < 45:
        return "Social discussion elevated but price action has not confirmed — hype risk."
    if reality >= 60 and hype < 45:
        return "Strong, quiet move — price acting well with limited chatter (under the radar)."
    if hype >= 55 and reality < 30:
        return "Loud but unconfirmed — possible pump/crowd-trap; wait for the tape."
    return "Mixed — no clear confirmation either way."


def _verdict(hype, reality):
    if reality >= 60 and hype >= 55:
        return "REAL_MOMENTUM"
    if hype >= 55 and reality < 30:
        return "PUMP_RISK"
    if hype >= 55 and reality < 45:
        return "HYPE_UNCONFIRMED"
    if reality >= 60 and hype < 45:
        return "UNDER_RADAR"
    return "MIXED"


def compute(sb, limit: int = 15) -> dict:
    """Hype/Reality per trending ticker. Never raises."""
    try:
        snaps = (sb.table("social_snapshots")
                 .select("ticker,captured_at,reddit_mentions,reddit_sentiment")
                 .order("captured_at", desc=True).limit(3000).execute().data) or []
        by = defaultdict(list)
        for r in snaps:
            if r.get("ticker"):
                by[r["ticker"]].append(r)
        # rank by latest mentions, take top `limit`
        ranked = sorted(by.items(),
                        key=lambda kv: -(kv[1][0].get("reddit_mentions") or 0))[:limit]
        if not ranked:
            return {"enabled": True, "items": [], "note": "No social data yet."}

        tickers = [tk for tk, _ in ranked]
        bars = {}
        try:
            from engine.alpaca_client import get_multi_bars
            bars = get_multi_bars(tickers, "1Day", 40) or {}
        except Exception:
            pass

        items = []
        for tk, rows in ranked:
            latest = rows[0]
            mentions = latest.get("reddit_mentions")
            prior = rows[1] if len(rows) > 1 else None
            vel = None
            if prior and (prior.get("reddit_mentions") or 0) > 0:
                vel = (((mentions or 0) - prior["reddit_mentions"]) / prior["reddit_mentions"]) * 100
            ret5d = vol_ratio = above_ma = None
            df = bars.get(tk)
            if df is not None and len(df) >= 21:
                c = df["close"]
                ret5d = (float(c.iloc[-1]) / float(c.iloc[-6]) - 1) * 100 if len(c) > 6 else None
                v = df["volume"]
                avg = float(v.iloc[-21:-1].mean())
                vol_ratio = float(v.iloc[-1]) / avg if avg > 0 else None
                above_ma = float(c.iloc[-1]) > float(c.rolling(20).mean().iloc[-1])
            h = _hype(vel, mentions)
            r = _reality(ret5d, vol_ratio, above_ma)
            items.append({"ticker": tk, "hype_score": h, "reality_score": r,
                          "gap": h - r, "verdict": _verdict(h, r),
                          "explanation": _explain(h, r),
                          "mentions": mentions, "ret_5d_pct": round(ret5d, 1) if ret5d is not None else None})
        items.sort(key=lambda x: -x["hype_score"])
        return {"enabled": True, "count": len(items), "items": items}
    except Exception as e:
        logger.error(f"[community_intel] failed: {e}")
        return {"enabled": True, "items": [], "error": str(e)}
