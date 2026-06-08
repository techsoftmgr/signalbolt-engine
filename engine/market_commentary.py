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

# The tape is market-WIDE (identical for everyone), so we compute it once and
# cache the RAW result; the endpoint scrubs per-viewer. A background warmer
# (runner.py, ~60s) keeps it fresh so requests return instantly.
_CACHE_KEY = "market_tape:v1"
_CACHE_TTL = 90

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


# For the MARKET tape, keep only STRUCTURAL index moves — routine MACD/EMA/RSI
# crosses ("momentum turned up/down") flip constantly intraday and are low-signal
# noise here (they belong on the per-ticker tape). Keep gaps, sharp moves,
# opening-range breaks, and VWAP regime flips.
_INDEX_KEEP = {"GAP", "MOVE", "ORB", "VWAP"}


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
                for e in (tc.get("events") or []):
                    if e.get("type") not in _INDEX_KEEP:
                        continue                # drop momentum/EMA/RSI/HoD-LoD chatter
                    ev = dict(e)
                    ev["title"] = f"{label}: {ev.get('title')}"
                    ev["scope"] = "index"
                    ev.pop("idea", None)        # index-level: context, not a trade idea
                    out.append(ev)
                    if len([x for x in out if x.get("scope") == "index"]) >= 4:
                        break
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


def _index_day_pct() -> dict:
    """{'SPY': pct, 'QQQ': pct} — each index's move vs its prior session close."""
    out = {}
    try:
        from engine.alpaca_client import get_bars
        for sym in ("SPY", "QQQ"):
            try:
                df = get_bars(sym, "5Min", days=4)
                if df is None or df.empty:
                    continue
                last_day = df.index[-1].date()
                prior = df[df.index.map(lambda t: t.date() != last_day)]
                if not len(prior):
                    continue
                pc = float(prior["close"].iloc[-1])
                last = float(df["close"].iloc[-1])
                if pc:
                    out[sym] = (last - pc) / pc * 100
            except Exception:
                continue
    except Exception:
        pass
    return out


def _internals() -> dict | None:
    """SPY vs QQQ leadership / divergence — a market-internals read. NOT a
    correlation coefficient (those are ~always high); what matters is who's
    leading and whether the two indices SPLIT (a narrow, lower-conviction tape)."""
    p = _index_day_pct()
    spy, qqq = p.get("SPY"), p.get("QQQ")
    if spy is None or qqq is None:
        return None
    spread = qqq - spy                         # + = Nasdaq leading, - = S&P leading
    split = (spy > 0.05 and qqq < -0.05) or (spy < -0.05 and qqq > 0.05)
    if split:
        state, leader = "divergent", ("tech" if qqq > spy else "broad")
    elif spread >= 0.4:
        state, leader = "growth_leading", "tech"
    elif spread <= -0.4:
        state, leader = "broad_leading", "broad"
    else:
        state, leader = "in_line", "none"
    return {"spy_pct": round(spy, 2), "qqq_pct": round(qqq, 2), "spread": round(spread, 2),
            "state": state, "leader": leader, "divergent": split}


def build(now: datetime | None = None) -> dict:
    """Assemble the market tape. Never raises. Result is cached ~90s (market-wide,
    so shared across viewers); pass an explicit `now` to bypass the cache."""
    use_cache = now is None
    if use_cache:
        try:
            from engine import cache
            cached = cache.kv.get_json(_CACHE_KEY)
            if cached:
                return cached
        except Exception:
            pass
    try:
        now = now or datetime.now(timezone.utc)
        phase = _phase(now)
        b = _market_bias()

        # SPY vs QQQ internals — leadership + divergence. A split = narrow tape →
        # downgrade a risk-on/off read to NEUTRAL (lower conviction).
        intern = _internals()
        bias = b["bias"]
        if intern and intern["divergent"] and bias in ("risk-on", "risk-off"):
            bias = "neutral"

        events: list[dict] = []
        if phase in ("premarket", "afterhours", "closed"):
            events += _gap_event(now)
        events += _index_events(phase)
        if intern and intern["divergent"]:
            spy, qqq = intern["spy_pct"], intern["qqq_pct"]
            events.append({
                "time": now.isoformat(), "type": "DIVERGENCE", "tone": "neutral", "severity": 2,
                "title": f"S&P {spy:+.1f}% vs Nasdaq {qqq:+.1f}% — index divergence",
                "detail": "The S&P and Nasdaq are splitting — a narrow tape. Treat the move as "
                          "lower-conviction until the two realign.", "scope": "internals"})
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
        bias_word = {"risk-on": "RISK-ON", "risk-off": "RISK-OFF"}.get(bias, "NEUTRAL")
        vix_txt = f" · VIX {_round(b['vix'])}" if b.get("vix") else ""
        phase_txt = {"premarket": "Pre-market", "open": "Market open",
                     "afterhours": "After-hours", "closed": "Market closed"}.get(phase, phase)
        cat_txt = ""
        if cal["today"]:
            cat_txt = " · today: " + ", ".join(c.get("event", "") for c in cal["today"][:2])
        intern_txt = ""
        if intern:
            intern_txt = {"divergent": " · indices diverging",
                          "growth_leading": " · tech leading",
                          "broad_leading": " · broad market leading"}.get(intern["state"], "")
        summary = f"{phase_txt} — tape is {bias_word}{vix_txt}{intern_txt}{cat_txt}."

        result = {
            "available": True, "phase": phase,
            "as_of": now.isoformat(),
            "bias": bias, "vix": b.get("vix"), "regime_type": b.get("regime_type"),
            "internals": intern,
            "summary": summary,
            "events": events,
            "catalysts": cal["today"],
            "upcoming_catalysts": cal["upcoming"][:6],
            "has_calendar_feed": cal["has_feed"],
            "disclaimer": _DISCLAIMER,
        }
        if use_cache:
            try:
                from engine import cache
                cache.kv.set_json(_CACHE_KEY, result, _CACHE_TTL)
            except Exception:
                pass
        return result
    except Exception as e:
        logger.debug(f"[market_commentary] build failed: {e}")
        return {"available": False, "note": "Market tape unavailable.", "disclaimer": _DISCLAIMER}


# ── V3: bias track record — was the day's risk-on/off call right next session? ──
_BIAS_HORIZON_DAYS = 1


def _bias_correct(bias: str | None, fwd_return_pct: float) -> bool | None:
    """Pure: did the bias match the forward SPY move? Neutral isn't scored."""
    if bias == "risk-on":
        return fwd_return_pct > 0.2
    if bias == "risk-off":
        return fwd_return_pct < -0.2
    return None


def log_bias_snapshot(sb, now: datetime | None = None) -> dict:
    """One row/day: today's market bias + SPY price (for a forward track record).
    Best-effort; no-ops if the table is missing or a row already exists today."""
    if sb is None:
        return {"logged": 0}
    try:
        now = now or datetime.now(timezone.utc)
        today = now.date().isoformat()
        snap = build(now)
        if not snap.get("available"):
            return {"logged": 0}
        exists = (sb.table("market_bias_log").select("id")
                  .gte("created_at", today + "T00:00:00Z").limit(1).execute().data)
        if exists:
            return {"logged": 0, "skipped": True}
        spy = None
        try:
            from engine.alpaca_client import get_latest_price
            spy = get_latest_price("SPY")
        except Exception:
            pass
        sb.table("market_bias_log").insert({
            "bias": snap.get("bias"), "vix": snap.get("vix"),
            "regime_type": snap.get("regime_type"),
            "spy_price": round(float(spy), 2) if spy else None,
        }).execute()
        return {"logged": 1}
    except Exception as e:
        logger.debug(f"[market_commentary] log_bias_snapshot failed: {e}")
        return {"logged": 0}


def score_bias_snapshots(sb) -> dict:
    """Fill the forward outcome for bias rows past the horizon. Best-effort."""
    if sb is None:
        return {"scored": 0}
    try:
        from datetime import timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(days=_BIAS_HORIZON_DAYS)).isoformat()
        rows = (sb.table("market_bias_log").select("*")
                .is_("forward_return_pct", "null").lte("created_at", cutoff)
                .limit(200).execute().data) or []
    except Exception as e:
        logger.debug(f"[market_commentary] score fetch failed: {e}")
        return {"scored": 0}
    from engine.alpaca_client import get_latest_price
    n = 0
    for r in rows:
        try:
            entry = float(r.get("spy_price") or 0)
            now_px = get_latest_price("SPY")
            if not entry or not now_px:
                continue
            ret = (now_px - entry) / entry * 100.0
            sb.table("market_bias_log").update({
                "horizon_days": _BIAS_HORIZON_DAYS, "forward_price": round(float(now_px), 2),
                "forward_return_pct": round(ret, 3), "correct": _bias_correct(r.get("bias"), ret),
                "scored_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", r["id"]).execute()
            n += 1
        except Exception:
            continue
    return {"scored": n}
