"""
Churn / Absorption screen — names trading on HIGH (pace-adjusted) relative volume
but going almost NOWHERE in price. Heavy two-sided volume at a level = absorption:
big buyers and sellers exchanging size while price stalls. The Movers tab sorts by
price %, so it misses this entirely.

Each name is tagged by where price sits in its 20-day range:
  • near LOWS  → possible ACCUMULATION (buyers soaking up supply at the bottom)
  • near HIGHS → possible DISTRIBUTION (sellers feeding the top)
  • mid-range  → pure CHURN / indecision
…and flagged `event=True` when a recent large news gap on heavy volume suggests an
EVENT-driven pin (e.g. M&A / buyout), where the technical read does NOT apply (the
price is anchored to a deal, not supply/demand — ROKU 2026-06-15 is the archetype).

Relative volume is PACE-ADJUSTED via volume_curve (project the partial day to a
full day) so a mid-session read isn't understated. Same warmer→cache→poll design
as movers_service: the heavy ~250-name bars build runs off the request path.
"""
from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np

logger = logging.getLogger("signalbolt.churn")

_CACHE_KEY  = "markets:churn:v1"
_TTL        = 180   # safety-net: the warmer keeps it fresh; serve last-good if a cycle hiccups
_inflight   = threading.Lock()
_ET         = ZoneInfo("America/New_York")
_STALE_DAYS = 5     # carry the last session this many days before treating data as stale

_MIN_PRICE   = float(os.environ.get("CHURN_MIN_PRICE", "5"))
_MIN_RELVOL  = float(os.environ.get("CHURN_MIN_RELVOL", "1.0"))     # at/above its own avg pace
_MAX_MOVE    = float(os.environ.get("CHURN_MAX_MOVE_PCT", "3.0"))   # ceiling on the move; the churn SCORE (below) does the real ranking
_STRONG_RELVOL = float(os.environ.get("CHURN_STRONG_RELVOL", "1.5"))
_EVENT_GAP   = float(os.environ.get("CHURN_EVENT_GAP", "0.08"))     # ≥8% gap on heavy vol in last few days


def classify_zone(range_pos: float) -> str:
    """Where price sits in its 20-day range → the absorption interpretation."""
    if range_pos <= 0.30:
        return "accumulation"
    if range_pos >= 0.70:
        return "distribution"
    return "churn"


def session_fraction(now_et: datetime) -> float:
    """Fraction of a normal RTH day's volume expected done by now (pace divisor).
    1.0 outside RTH (premarket → no projection; after close → full day)."""
    elapsed = (now_et.hour * 60 + now_et.minute) - (9 * 60 + 30)
    if elapsed <= 0 or elapsed >= 390:
        return 1.0
    from engine.volume_curve import expected_volume_fraction
    return expected_volume_fraction(elapsed)


def _evaluate(symbol: str, df, frac: float, today, signal=None) -> dict | None:
    """Pure per-symbol churn read from daily bars (today's bar = forming/so-far).
    Returns the item dict if it qualifies as high-volume-low-move, else None."""
    try:
        if df is None or len(df) < 22:
            return None
        # Use the MOST RECENT session's daily bar. During RTH that IS today's forming bar
        # (Alpaca includes it) so session_fraction() pace-projects it; OUTSIDE RTH it's the
        # last COMPLETED session and session_fraction()==1.0 gives the realized read — so the
        # screen CARRIES overnight / pre-open instead of going blank once the ET date rolls
        # past the last bar (the old `!= today` guard rejected everything from midnight ET to
        # the next open), then updates live at the open when a new forming bar appears. Reject
        # only genuinely STALE data (halted / delisted).
        try:
            _last = df.index[-1]
            _last_et = _last.tz_convert(_ET).date() if getattr(df.index, "tz", None) is not None else _last.date()
        except Exception:
            _last_et = today
        if (today - _last_et).days > _STALE_DAYS:
            return None
        c = df["close"].values.astype(float)
        v = df["volume"].values.astype(float)
        h = df["high"].values.astype(float)
        l = df["low"].values.astype(float)
        price, prev, vol = c[-1], c[-2], v[-1]
        if price < _MIN_PRICE or prev <= 0:
            return None
        avg20 = float(np.mean(v[-21:-1]))
        if avg20 <= 0:
            return None
        relvol = (vol / frac) / avg20 if frac > 0 else vol / avg20
        chg = (price / prev - 1) * 100
        if relvol < _MIN_RELVOL or abs(chg) > _MAX_MOVE:
            return None
        # Churn score = volume per unit of price move. The whole point: ROKU on 8.7×
        # volume but only -1.6% should outrank a name at 1.1× and 0.0% — heavy volume
        # going almost nowhere is the strongest absorption. (floor |chg| so a literal
        # 0.00% move doesn't divide-by-zero / dominate on noise alone.)
        churn_score = relvol / max(abs(chg), 0.5)
        wh, wl = float(h[-20:].max()), float(l[-20:].min())
        pos = (price - wl) / (wh - wl) if wh > wl else 0.5
        # recent event gap (last ~3 completed sessions): big move on heavy volume
        event = False
        for j in range(max(1, len(c) - 4), len(c) - 1):
            if c[j - 1] > 0 and abs(c[j] / c[j - 1] - 1) >= _EVENT_GAP and v[j] >= 2.5 * avg20:
                event = True
                break
        return {
            "symbol": symbol,
            "price": round(price, 2),
            "changePct": round(chg, 2),
            "volume": int(vol),
            "avgVolume": int(avg20),
            "relVol": round(relvol, 2),
            "churnScore": round(churn_score, 2),
            "rangePos": round(pos, 2),
            "zone": classify_zone(pos),
            "strong": bool(relvol >= _STRONG_RELVOL),
            "event": bool(event),
            "signal": signal,
        }
    except Exception:
        return None


def peek_churn() -> dict | None:
    """Fast, non-blocking read of the cached churn list (None if not warmed yet)."""
    try:
        from engine import cache
        return cache.kv.get_json(_CACHE_KEY)
    except Exception:
        return None


def compute_churn(limit: int = 25, force: bool = False) -> dict:
    """{asOf, items[]} — high pace-adjusted volume + small price move, ranked by
    relVol desc. Cached; the heavy bars build is coalesced to one in-flight worker."""
    from engine import cache
    empty = {"asOf": datetime.now(timezone.utc).isoformat(), "items": []}
    if not force:
        cached = cache.kv.get_json(_CACHE_KEY)
        if cached:
            return cached
    if not _inflight.acquire(blocking=False):
        return cache.kv.get_json(_CACHE_KEY) or empty
    try:
        cached = cache.kv.get_json(_CACHE_KEY)
        if cached and not force:
            return cached

        from engine.movers_service import _candidate_symbols
        syms = _candidate_symbols()
        if not syms:
            return empty

        from engine.alpaca_client import get_multi_bars
        bars = get_multi_bars(syms, "1Day", 50) or {}
        if not bars:
            return empty

        # Active-signal overlay (flag a churning name we already have a trade on).
        sig: dict = {}
        try:
            from engine import runner
            sb = runner._supabase()
            present = list(bars.keys())
            for i in range(0, len(present), 200):
                rows = (sb.table("signals").select("ticker,direction")
                        .eq("status", "active").in_("ticker", present[i:i + 200]).execute().data) or []
                for r in rows:
                    sig[r["ticker"]] = r.get("direction")
        except Exception as e:
            logger.debug(f"[churn] active-signal overlay failed: {e}")

        now_et = datetime.now(_ET)
        frac = session_fraction(now_et)
        today = now_et.date()
        items = [it for s, df in bars.items()
                 if (it := _evaluate(s, df, frac, today, sig.get(s)))]
        items.sort(key=lambda x: -x["churnScore"])
        out = {"asOf": datetime.now(timezone.utc).isoformat(), "items": items[:limit]}
        try:
            cache.kv.set_json(_CACHE_KEY, out, _TTL)
        except Exception:
            pass
        return out
    finally:
        _inflight.release()
