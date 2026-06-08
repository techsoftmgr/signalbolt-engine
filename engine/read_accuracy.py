"""
Read self-grade — measures whether the Expert Read's flagged FACTS held up, with
NO prediction involved. We log the levels the read flagged (support, resistance,
the bull/bear triggers) and later check, factually, whether they behaved as
defined: did the flagged support act as support (tested and not closed through)?
did resistance reject? That's "accuracy of the description," separate from any
directional call.

Mirrors the chart_read_log / market_bias_log pattern: a daily log job + a forward
scorer + an admin aggregate endpoint. Best-effort; never raises. Needs the
`read_accuracy_log` table (supabase-read-accuracy-log.sql); no-ops if absent.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger("signalbolt.read_accuracy")

_HORIZON_DAYS = 5
_MAX_TICKERS = 80
_TEST_BAND = 0.005    # within 0.5% counts as "tested"
_BREAK_BAND = 0.01    # a daily close >1% beyond the level = broken (didn't hold)


def _grade_levels(support, resistance, bars: list[dict]) -> dict:
    """PURE: given the flagged levels + forward daily bars (high/low/close), did
    each level behave as support/resistance? 'held' is only meaningful once
    'tested'. Never raises."""
    out = {"support_tested": None, "support_held": None,
           "resistance_tested": None, "resistance_held": None}
    try:
        if not bars:
            return out
        highs = [float(b["high"]) for b in bars]
        lows = [float(b["low"]) for b in bars]
        closes = [float(b["close"]) for b in bars]
        if support:
            s = float(support)
            tested = any(l <= s * (1 + _TEST_BAND) for l in lows)
            out["support_tested"] = tested
            if tested:
                out["support_held"] = not any(c < s * (1 - _BREAK_BAND) for c in closes)
        if resistance:
            r = float(resistance)
            tested = any(h >= r * (1 - _TEST_BAND) for h in highs)
            out["resistance_tested"] = tested
            if tested:
                out["resistance_held"] = not any(c > r * (1 + _BREAK_BAND) for c in closes)
    except Exception:
        pass
    return out


def log_levels(sb, tickers: list[str] | None = None) -> dict:
    """One row/day per ticker: the levels the daily read flagged. Best-effort."""
    stats = {"logged": 0}
    if sb is None:
        return stats
    try:
        if tickers is None:
            rows = sb.table("watchlist").select("ticker").execute().data or []
            tickers = list({(r.get("ticker") or "").upper() for r in rows if r.get("ticker")})[:_MAX_TICKERS]
    except Exception as e:
        logger.debug(f"[read_accuracy] watchlist fetch failed: {e}")
        return stats
    today = datetime.now(timezone.utc).date().isoformat()
    from engine import chart_read
    for tk in (tickers or []):
        try:
            exists = (sb.table("read_accuracy_log").select("id").eq("ticker", tk)
                      .gte("created_at", today + "T00:00:00Z").limit(1).execute().data)
            if exists:
                continue
            r = chart_read.analyze(tk)            # daily read (settled)
            if not r:
                continue
            lv = r.get("levels") or {}
            scen = r.get("scenarios") or {}
            sb.table("read_accuracy_log").insert({
                "ticker": tk, "bias": r.get("taBias"), "price": r.get("price"),
                "support": lv.get("support"), "resistance": lv.get("resistance"),
                "bull_trigger": (scen.get("bull") or {}).get("trigger"),
                "bear_trigger": (scen.get("bear") or {}).get("trigger"),
            }).execute()
            stats["logged"] += 1
        except Exception as e:
            logger.debug(f"[read_accuracy] log {tk} failed: {e}")
    logger.info(f"[read_accuracy] logged {stats}")
    return stats


def score_levels(sb) -> dict:
    """Fill the held/tested flags for rows past the horizon. Best-effort."""
    stats = {"scored": 0}
    if sb is None:
        return stats
    cutoff = (datetime.now(timezone.utc) - timedelta(days=_HORIZON_DAYS)).isoformat()
    try:
        rows = (sb.table("read_accuracy_log").select("*")
                .is_("scored_at", "null").lte("created_at", cutoff)
                .limit(200).execute().data) or []
    except Exception as e:
        logger.debug(f"[read_accuracy] fetch failed: {e}")
        return stats
    from engine.alpaca_client import get_bars
    for row in rows:
        try:
            tk = row.get("ticker")
            logged_date = str(row.get("created_at") or "")[:10]
            df = get_bars(tk, "1Day", days=_HORIZON_DAYS + 6)
            if df is None or df.empty:
                continue
            fwd = df[df.index.map(lambda t: t.date().isoformat() > logged_date)]
            bars = [{"high": float(r2.high), "low": float(r2.low), "close": float(r2.close)}
                    for r2 in fwd.itertuples()]
            g = _grade_levels(row.get("support"), row.get("resistance"), bars)
            sb.table("read_accuracy_log").update({
                **g, "horizon_days": _HORIZON_DAYS,
                "scored_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", row["id"]).execute()
            stats["scored"] += 1
        except Exception as e:
            logger.debug(f"[read_accuracy] score row failed: {e}")
    logger.info(f"[read_accuracy] scored {stats}")
    return stats


def stats(sb, days: int = 90) -> dict:
    """Aggregate: of the levels we flagged, how often did they act as support/
    resistance? Descriptive accuracy — not a win-rate. Best-effort."""
    if sb is None:
        return {"available": False}
    try:
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = (sb.table("read_accuracy_log").select("support_tested,support_held,resistance_tested,resistance_held")
                .gte("created_at", since).not_.is_("scored_at", "null")
                .limit(2000).execute().data) or []
    except Exception:
        return {"available": False, "note": "Track record not available yet (table empty/uncreated)."}
    s_tested = [r for r in rows if r.get("support_tested")]
    r_tested = [r for r in rows if r.get("resistance_tested")]
    s_held = sum(1 for r in s_tested if r.get("support_held"))
    r_held = sum(1 for r in r_tested if r.get("resistance_held"))
    return {
        "available": bool(rows), "scored": len(rows),
        "support": {"tested": len(s_tested), "held_pct": round(s_held / len(s_tested) * 100) if s_tested else None},
        "resistance": {"tested": len(r_tested), "held_pct": round(r_held / len(r_tested) * 100) if r_tested else None},
        "note": "How often flagged levels acted as support/resistance — descriptive accuracy, not a prediction.",
    }


_CACHE_KEY = "read_accuracy:stats:v1"
_CACHE_TTL = 3600   # app-wide aggregate moves slowly; 1h cache avoids a DB hit per read


def stats_cached(sb, days: int = 90) -> dict:
    """App-wide stats with a 1h cache so it can ride along on every chart-read
    without a per-request DB hit. Falls back to a direct read if cache is down."""
    try:
        from engine.cache import kv
        hit = kv.get_json(_CACHE_KEY)
        if hit is not None:
            return hit
    except Exception:
        kv = None
    out = stats(sb, days=days)
    try:
        if kv is not None and out.get("available"):
            kv.set_json(_CACHE_KEY, out, _CACHE_TTL)
    except Exception:
        pass
    return out
