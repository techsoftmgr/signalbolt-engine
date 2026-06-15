"""
Market Pulse — INTRADAY provisional read (Part B). A soft, clearly-labelled
preview of whether TODAY is on pace to become a distribution / stalling day,
BEFORE the close confirms it.

INTEGRITY FIREWALL: this module is physically incapable of writing to
market_pulse_daily — it does NOT import engine.market_pulse.store / .job and has
no DB-write path. The provisional read is ephemeral (live-return only). The EOD
table is confirmed-only.

Accuracy: intraday volume is U/J-shaped (heavy open, thin midday, surge into the
close), so we DO NOT project linearly off clock time — we divide volume-so-far by
the expected cumulative fraction from an empirical 30-min volume-profile curve
(hardcoded U-curve fallback until the empirical one is built). Confidence rises
through the session; before 11:00 ET we return TOO_EARLY rather than guess.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from . import config as C

logger = logging.getLogger("signalbolt.market_pulse.intraday")

# Hardcoded fallback cumulative-volume fraction by 30-min bucket (regular ~6.5h
# session = 13 buckets; half-day ~3.5h = 7). Far better than linear; replaced by
# the empirical curve once built. Each ends at 1.0.
_FALLBACK_FULL = [0.12, 0.20, 0.27, 0.33, 0.39, 0.45, 0.51, 0.57, 0.63, 0.70, 0.78, 0.88, 1.00]
_FALLBACK_HALF = [0.16, 0.28, 0.40, 0.52, 0.66, 0.82, 1.00]
_PROFILE_TTL = 24 * 3600


# ── Pure helpers (unit-tested directly; no I/O) ─────────────────────────────
def expected_fraction(curve: Optional[list], bucket_idx: int) -> Optional[float]:
    """Cumulative fraction of full-day volume expected complete by `bucket_idx`
    (0-based, the bucket currently in progress)."""
    if not curve:
        return None
    i = min(max(bucket_idx, 0), len(curve) - 1)
    return float(curve[i])


def project_volume(volume_so_far: float, frac: Optional[float]) -> Optional[float]:
    """Project full-day volume = volume_so_far / expected_fraction_complete_now."""
    if not frac or frac <= 0:
        return None
    return float(volume_so_far) / float(frac)


def confidence_for(hour_et: float) -> str:
    """TOO_EARLY before the floor, then MEDIUM, then HIGH late in the session."""
    if hour_et < C.INTRADAY_CONF_FLOOR_ET:
        return "TOO_EARLY"
    return "HIGH" if hour_et >= C.INTRADAY_HIGH_CONF_ET else "MEDIUM"


def classify_status(projected_vol: Optional[float], prior_vol: Optional[float],
                    price_chg_pct: float, close_pos: float,
                    margin: float = C.INTRADAY_MARGIN) -> str:
    """ON_PACE_DISTRIBUTION / ON_PACE_STALLING / NEUTRAL from the projection."""
    if projected_vol is None or not prior_vol or prior_vol <= 0:
        return "NEUTRAL"
    vol_clears = projected_vol > prior_vol * (1 + margin)
    if not vol_clears:
        return "NEUTRAL"
    if price_chg_pct <= -0.2:
        return "ON_PACE_DISTRIBUTION"
    if 0 < price_chg_pct <= C.STALL_MAX_GAIN_PCT * 100 and close_pos <= C.STALL_CLOSE_RANGE_FRAC:
        return "ON_PACE_STALLING"
    return "NEUTRAL"


def _label(status: str, confidence: str) -> str:
    if status == "MARKET_CLOSED":
        return "Market closed / no intraday data."
    if status == "TOO_EARLY":
        return "Provisional: too early in the session to call — check back midday."
    body = {
        "ON_PACE_DISTRIBUTION": "on pace for a distribution day",
        "ON_PACE_STALLING": "on pace for a stalling day",
        "NEUTRAL": "no distribution/stalling pace",
    }.get(status, "no read")
    return f"Provisional ({confidence} confidence): {body} — not confirmed until the close."


# ── Session + curve (I/O; still no DB-write path) ───────────────────────────
def _session_et(now_et):
    """(open_et, close_et) for today, or None if not a trading day. Half-days have
    an early close, handled via the real NYSE calendar."""
    try:
        import pandas_market_calendars as mcal
        sched = mcal.get_calendar("NYSE").schedule(start_date=now_et.date(), end_date=now_et.date())
        if sched.empty:
            return None
        o = sched.iloc[0]["market_open"].tz_convert("America/New_York")
        c = sched.iloc[0]["market_close"].tz_convert("America/New_York")
        return o, c
    except Exception as e:
        logger.debug(f"[intraday] session calendar failed: {e}")
        return None


def _volume_profile(symbol: str, half: bool) -> list:
    """Empirical cumulative-fraction curve from the trailing ~60 regular sessions
    of 30-min bars; falls back to the hardcoded U-curve. 24h-cached."""
    fallback = _FALLBACK_HALF if half else _FALLBACK_FULL
    ck = f"mp:volcurve:{symbol}:{'half' if half else 'full'}"
    try:
        from engine import cache
        c = cache.kv.get_json(ck)
        if c:
            return c
    except Exception:
        pass
    curve = fallback
    try:
        from engine.alpaca_client import get_bars
        import numpy as np
        df = get_bars(symbol, "30Min", days=C.INTRADAY_PROFILE_DAYS)
        if df is not None and not df.empty:
            et = df.index.tz_convert("America/New_York")
            d = df.copy()
            d["date"] = [t.date() for t in et]
            d["t"] = [t.hour * 60 + t.minute for t in et]
            d = d[(d["t"] >= 9 * 60 + 30) & (d["t"] < 16 * 60)]   # regular session only
            L = len(fallback)
            stacks = []
            for _, g in d.groupby("date"):
                g = g.sort_index()
                tot = float(g["volume"].sum())
                if tot <= 0 or len(g) < L:
                    continue
                stacks.append((g["volume"].cumsum() / tot).to_numpy()[:L])
            if stacks:
                curve = [float(x) for x in np.mean(np.array(stacks), axis=0)]
    except Exception as e:
        logger.debug(f"[intraday] curve build {symbol} failed: {e}")
    try:
        from engine import cache
        cache.kv.set_json(ck, curve, _PROFILE_TTL)
    except Exception:
        pass
    return curve


def intraday_read(symbol: str, now_et=None) -> dict:
    """Provisional per-index read. now_et injectable for tests. Never writes anything."""
    from datetime import datetime as _dt
    if now_et is None:
        try:
            from zoneinfo import ZoneInfo
            now_et = _dt.now(ZoneInfo("America/New_York"))
        except Exception:
            now_et = _dt.utcnow()

    base = {"symbol": symbol, "provisional": True, "confidence": None,
            "projected_full_volume": None, "prior_day_volume": None, "curve_fraction_used": None}

    sess = _session_et(now_et)
    if sess is None or now_et < sess[0] or now_et > sess[1]:
        return {**base, "status": "MARKET_CLOSED", "label": _label("MARKET_CLOSED", "")}

    open_et, close_et = sess
    hour = now_et.hour + now_et.minute / 60.0
    conf = confidence_for(hour)
    if conf == "TOO_EARLY":
        return {**base, "status": "TOO_EARLY", "confidence": "LOW", "label": _label("TOO_EARLY", "LOW")}

    half = close_et.hour < 16
    try:
        from engine.alpaca_client import get_bars
        intr = get_bars(symbol, "30Min", days=2)
        daily = get_bars(symbol, "1Day", 6)
        if intr is None or intr.empty or daily is None or len(daily) < 2:
            return {**base, "status": "MARKET_CLOSED", "label": _label("MARKET_CLOSED", "")}
        et = intr.index.tz_convert("America/New_York")
        today = intr[[t.date() == now_et.date() and (t.hour * 60 + t.minute) >= 9 * 60 + 30 for t in et]]
        if today.empty:
            return {**base, "status": "MARKET_CLOSED", "label": _label("MARKET_CLOSED", "")}
        vol_so_far = float(today["volume"].sum())
        price = float(today["close"].iloc[-1])
        hi = float(today["high"].max()); lo = float(today["low"].min())
        prior_close = float(daily["close"].iloc[-2])
        prior_vol = float(daily["volume"].iloc[-2])

        bucket_idx = len(today) - 1          # 0-based bucket currently in progress
        curve = _volume_profile(symbol, half)
        frac = expected_fraction(curve, bucket_idx)
        projected = project_volume(vol_so_far, frac)
        price_chg_pct = (price / prior_close - 1) * 100 if prior_close > 0 else 0.0
        close_pos = (price - lo) / (hi - lo) if hi > lo else 0.0
        status = classify_status(projected, prior_vol, price_chg_pct, close_pos)
        return {
            **base, "status": status, "confidence": conf,
            "projected_full_volume": round(projected) if projected else None,
            "prior_day_volume": round(prior_vol),
            "curve_fraction_used": round(frac, 3) if frac else None,
            "price_change_pct": round(price_chg_pct, 2),
            "half_day": half,
            "label": _label(status, conf),
        }
    except Exception as e:
        logger.debug(f"[intraday] read {symbol} failed: {e}")
        return {**base, "status": "NEUTRAL", "confidence": conf, "label": _label("NEUTRAL", conf)}


_PHRASE = {
    "ON_PACE_DISTRIBUTION": "on pace for a distribution day",
    "ON_PACE_STALLING": "on pace for a stalling day",
    "NEUTRAL": "tracking a normal session",
    "TOO_EARLY": "too early to call",
}


def _read_summary(indices: dict) -> str:
    """One plain-English sentence across SPY + QQQ — the intraday equivalent of the
    EOD 'quick read'. Always flagged provisional."""
    live = {s: v for s, v in indices.items() if v.get("status") and v["status"] != "MARKET_CLOSED"}
    if not live:
        return "Market closed — no intraday read."
    if all(v["status"] == "TOO_EARLY" for v in live.values()):
        return "Too early in the session to call — a provisional read appears midday."
    parts = [f"{s} {_PHRASE.get(v['status'], '')}".strip() for s, v in live.items()]
    conf = "high" if any(v.get("confidence") == "HIGH" for v in live.values()) else "medium"
    building = any(v["status"] in ("ON_PACE_DISTRIBUTION", "ON_PACE_STALLING") for v in live.values())
    tail = " — selling pressure building, not confirmed until the close." if building \
        else " — provisional, not confirmed until the close."
    return f"{'; '.join(parts)} ({conf} confidence){tail}"


_CACHE_KEY = "market_pulse:intraday:v1"
_CACHE_TTL = 60   # recompute once/min server-side, served to all viewers (NOT the daily table)


def read_all(now_et=None) -> dict:
    """{asOf, provisional, summary, indices:{SPY, QQQ}} — provisional, never persisted
    to the EOD table. The LIVE path (now_et=None) is cached 60s in the ephemeral kv
    so it's computed once on the server and served to every viewer (tests pass now_et
    to bypass the cache)."""
    from datetime import datetime as _dt, timezone as _tz
    if now_et is None:
        try:
            from engine import cache
            cached = cache.kv.get_json(_CACHE_KEY)
            if cached:
                return cached
        except Exception:
            pass
    idx = {sym: intraday_read(sym, now_et) for sym in ("SPY", "QQQ")}
    out = {
        "provisional": True,
        "asOf": _dt.now(_tz.utc).isoformat(),
        "summary": _read_summary(idx),
        "disclaimer": "Provisional intraday estimate — the official daily read is confirmed only after the close.",
        "indices": idx,
    }
    if now_et is None:
        try:
            from engine import cache
            cache.kv.set_json(_CACHE_KEY, out, _CACHE_TTL)
        except Exception:
            pass
    return out
