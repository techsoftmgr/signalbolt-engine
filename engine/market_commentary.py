"""
Market Tape — a market-WIDE "play-by-play" from pre-market through after-hours.

The index-level sibling of ticker_commentary: one chronological feed for the whole
market with a risk-on/risk-off bias header. Phase-aware:
  • premarket  → overnight gap on SPY/QQQ + today's scheduled catalysts + policy news
  • open       → intraday technical events on SPY & QQQ (reuses ticker_commentary),
                 VIX/regime context, sector rotation, breadth, policy news
  • afterhours → where the day closed + notable movers + policy news
  • closed     → last session recap + next scheduled catalysts

Reuses regime_detector (VIX/regime), pulse_service (sector rotation), econ_calendar
(catalysts), alpaca news (policy/headline stream), and the ticker_commentary differ.
Read-only, best-effort, never raises. Educational framing — context, not alpha.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

logger = logging.getLogger("signalbolt.market_commentary")

_DISCLAIMER = "Live market context — educational awareness, not financial advice."
_MAX_EVENTS = 30
_INDEXES = [("SPY", "S&P 500"), ("QQQ", "Nasdaq")]

# Headlines that historically move the whole tape (policy / macro / geopolitics).
# This is the reliable, licensed-news version of a "market-moving posts" feed —
# it surfaces such statements as reported by the news provider (incl. Trump /
# Fed / tariffs), tagged POLICY. A literal real-time social feed would need a
# dedicated paid source.
_POLICY_KW = [
    "trump", "tariff", "tariffs", "powell", "fed", "federal reserve", "fomc",
    "rate cut", "rate hike", "interest rate", "interest rates", "inflation",
    "cpi", "jobs report", "payrolls", "nonfarm", "unemployment", "sanction",
    "sanctions", "stimulus", "shutdown", "debt ceiling", "white house",
    "treasury", "yellen", "bessent", "opec", "war", "executive order",
    "recession", "gdp", "pce", "rate decision",
]
# Word-boundary match so "war" doesn't hit "award", "tax" doesn't hit "syntax", etc.
_POLICY_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in _POLICY_KW) + r")\b", re.IGNORECASE)


def _round(v):
    try:
        return round(float(v), 2)
    except (TypeError, ValueError):
        return v


def _phase(now: datetime) -> str:
    try:
        from engine import session_classifier as sc
        if not sc.is_market_open_today():
            return "closed"
        if sc.is_market_open_now():
            return "open"
    except Exception:
        return "open"
    mins = now.hour * 60 + now.minute      # UTC; RTH ≈ 13:30–20:00 UTC
    return "premarket" if mins < 13 * 60 + 30 else "afterhours"


def _market_bias():
    """risk-on / risk-off / neutral from SPY vs VWAP + the 200-DMA + regime/VIX."""
    bias, vix, regime_type = "neutral", None, None
    above200 = None
    try:
        from engine import regime_detector
        reg = regime_detector.detect() or {}
        vix, regime_type, above200 = reg.get("vix"), reg.get("regime_type"), reg.get("above_200ma")
    except Exception:
        pass
    spy_above_vwap = None
    try:
        from engine.alpaca_client import get_bars
        from engine import ticker_commentary as tcm
        spy = tcm._session_slice(get_bars("SPY", "5Min", days=4))
        if spy is not None and len(spy) >= 2:
            spy_above_vwap = float(spy["close"].iloc[-1]) > tcm._vwap(spy)[-1]
    except Exception:
        pass
    if spy_above_vwap is True and above200 is not False:
        bias = "risk-on"
    elif spy_above_vwap is False and above200 is not True:
        bias = "risk-off"
    return {"bias": bias, "vix": vix, "regime_type": regime_type, "above_200ma": above200}


def _policy_headlines(limit: int = 6) -> list[dict]:
    try:
        from engine import alpaca_client
        news = alpaca_client.get_multi_news(["SPY", "QQQ", "DIA", "IWM"], limit=40) or []
    except Exception:
        return []
    out = []
    seen = set()
    for n in news:
        head = (n.get("headline") or "").strip()
        if not head or head in seen:
            continue
        if _POLICY_RE.search(head):
            seen.add(head)
            out.append({
                "time": n.get("created_at") or n.get("time"),
                "type": "POLICY", "tone": "neutral", "severity": 2,
                "title": head,
                "detail": (n.get("summary") or "")[:180],
                "url": n.get("url"), "source": n.get("source") or "news",
            })
        if len(out) >= limit:
            break
    return out


def _index_events(phase: str) -> list[dict]:
    if phase != "open":
        return []
    out = []
    try:
        from engine import ticker_commentary as tcm
        for sym, label in _INDEXES:
            try:
                tc = tcm.build(sym)
                if not tc.get("available"):
                    continue
                for e in (tc.get("events") or [])[:6]:
                    ev = dict(e)
                    ev["title"] = f"{label}: {ev.get('title')}"
                    ev["scope"] = "index"
                    ev.pop("idea", None)        # index-level: context, not a trade idea
                    out.append(ev)
            except Exception:
                continue
    except Exception:
        pass
    return out


def _gap_event(now: datetime) -> list[dict]:
    out = []
    try:
        from engine.alpaca_client import get_bars
        for sym, label in _INDEXES:
            df = get_bars(sym, "5Min", days=4)
            if df is None or df.empty:
                continue
            last_day = df.index[-1].date()
            prior = df[df.index.map(lambda t: t.date() != last_day)]
            if not len(prior):
                continue
            prior_close = float(prior["close"].iloc[-1])
            last = float(df["close"].iloc[-1])
            chg = (last - prior_close) / prior_close * 100 if prior_close else 0
            if abs(chg) >= 0.4:
                up = chg > 0
                out.append({"time": df.index[-1].isoformat(), "type": "GAP",
                            "tone": "bullish" if up else "bearish", "severity": 2,
                            "title": f"{label}: {'up' if up else 'down'} {abs(chg):.1f}% vs prior close",
                            "detail": f"{label} near ${_round(last)} ({'+' if up else ''}{chg:.1f}%) before/after the regular session.",
                            "scope": "index"})
    except Exception:
        pass
    return out


def _social_events(limit: int = 6) -> list[dict]:
    """Market-moving social posts (e.g. Trump via TweetShift→Discord). High
    severity — these move the tape. Empty when the feed isn't configured."""
    try:
        from engine import social_feed
        out = []
        for p in social_feed.recent_posts(limit):
            text = (p.get("text") or "").strip()
            if not text:
                continue
            out.append({"time": p.get("created_at"), "type": "SOCIAL", "tone": "neutral",
                        "severity": 3,
                        "title": f"{p.get('author') or 'Post'}: {text[:140]}",
                        "detail": text[:300], "url": p.get("url"),
                        "source": p.get("author") or "social"})
        return out
    except Exception:
        return []


def _sector_event() -> list[dict]:
    try:
        from engine import pulse_service
        p = pulse_service.compute()
        s = (p or {}).get("strongest")
        if s and s.get("ticker"):
            tone = "bullish" if s.get("bias") == "buy" else "bearish" if s.get("bias") == "sell" else "neutral"
            return [{"time": (p or {}).get("as_of"), "type": "SECTOR", "tone": tone, "severity": 1,
                     "title": f"Sector leadership: {s['ticker']} {str(s.get('bias','')).upper()}",
                     "detail": f"Strongest sector-ETF bias right now is {s['ticker']} "
                               f"({str(s.get('bias',''))}, score {s.get('score')}). Overall pulse: {(p or {}).get('bias')}.",
                     "scope": "sector"}]
    except Exception:
        pass
    return []


def build(now: datetime | None = None) -> dict:
    """Assemble the market tape. Never raises."""
    try:
        now = now or datetime.now(timezone.utc)
        phase = _phase(now)
        b = _market_bias()

        events: list[dict] = []
        if phase in ("premarket", "afterhours", "closed"):
            events += _gap_event(now)
        events += _index_events(phase)
        events += _sector_event()
        events += _social_events(6)        # market-moving posts (any phase)
        events += _policy_headlines(6)

        # sort newest-first where a timestamp exists; undated context floats to top
        def _key(e):
            return e.get("time") or "9999"
        events.sort(key=_key, reverse=True)
        events = events[:_MAX_EVENTS]

        try:
            from engine import econ_calendar
            cal = econ_calendar.today_and_upcoming(now)
        except Exception:
            cal = {"today": [], "upcoming": [], "has_feed": False}

        # headline summary
        bias_word = {"risk-on": "RISK-ON", "risk-off": "RISK-OFF"}.get(b["bias"], "NEUTRAL")
        vix_txt = f" · VIX {_round(b['vix'])}" if b.get("vix") else ""
        phase_txt = {"premarket": "Pre-market", "open": "Market open",
                     "afterhours": "After-hours", "closed": "Market closed"}.get(phase, phase)
        cat_txt = ""
        if cal["today"]:
            cat_txt = " · today: " + ", ".join(c.get("event", "") for c in cal["today"][:2])
        summary = f"{phase_txt} — tape is {bias_word}{vix_txt}{cat_txt}."

        return {
            "available": True, "phase": phase,
            "as_of": now.isoformat(),
            "bias": b["bias"], "vix": b.get("vix"), "regime_type": b.get("regime_type"),
            "summary": summary,
            "events": events,
            "catalysts": cal["today"],
            "upcoming_catalysts": cal["upcoming"][:6],
            "has_calendar_feed": cal["has_feed"],
            "disclaimer": _DISCLAIMER,
        }
    except Exception as e:
        logger.debug(f"[market_commentary] build failed: {e}")
        return {"available": False, "note": "Market tape unavailable.", "disclaimer": _DISCLAIMER}
