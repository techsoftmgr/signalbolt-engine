"""
Chart Read — programmatic "expert technical read" of a ticker, computed from
OHLCV (NOT chart-image vision). Deterministic, fast, backtestable.

Phase 1 (read-only — surfaces an Expert Read card on the hub; no signals/trading):
  • trend per timeframe (15m / 1h / daily) + multi-timeframe (MTF) confluence
  • swing pivots → diagonal support/resistance TRENDLINES (slope + test/break)
  • regression CHANNEL + position (top / mid / bottom)
  • key horizontal SUPPORT/RESISTANCE near price (pivots + MAs + prior-day H/L)
  • GAPS (most recent unfilled) + classification (breakaway / runaway / exhaustion)
  • VOLUME regime (accumulation / distribution / climax / dry-up)
  • PATTERNS (geometrically robust set): double top/bottom, triangle (asc/desc/sym),
    flag/pennant — each with confidence + a measured target. (H&S / cup-handle are
    deferred — higher false-positive rate; flagged 'planned'.)

Output: a structured dict (for the text card) + `overlays` (line/level geometry
the chart layer can later DRAW) + plain-English `narrative` + `bias`/`confidence`.

Best-effort throughout — never raises into the request path.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("signalbolt.chart_read")

_SWING_N = 3          # bars each side for a swing pivot
_GAP_MIN_PCT = 1.0    # min % gap to flag


# ── Swings ────────────────────────────────────────────────────────────────────
def _swings(df: pd.DataFrame, n: int = _SWING_N) -> tuple[list[int], list[int]]:
    """Return (swing_high_idx, swing_low_idx) — local extrema with n bars each side."""
    highs = df["high"].values.astype(float)
    lows = df["low"].values.astype(float)
    hi_idx, lo_idx = [], []
    for i in range(n, len(df) - n):
        if highs[i] == max(highs[i - n:i + n + 1]):
            hi_idx.append(i)
        if lows[i] == min(lows[i - n:i + n + 1]):
            lo_idx.append(i)
    return hi_idx, lo_idx


# ── Trend ─────────────────────────────────────────────────────────────────────
def _trend(df: pd.DataFrame) -> str:
    """'up' / 'down' / 'sideways' from EMA20 vs EMA50 + slope + price position."""
    if df is None or len(df) < 50:
        return "sideways"
    c = df["close"]
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    px = float(c.iloc[-1])
    e20, e50 = float(ema20.iloc[-1]), float(ema50.iloc[-1])
    slope20 = (e20 - float(ema20.iloc[-10])) / float(ema20.iloc[-10]) * 100 if len(ema20) >= 10 else 0.0
    if px > e20 > e50 and slope20 > 0.2:
        return "up"
    if px < e20 < e50 and slope20 < -0.2:
        return "down"
    return "sideways"


# ── Trendlines (least-squares on recent swing highs / lows) ───────────────────
def _fit_line(idxs: list[int], vals: np.ndarray, last_x: int) -> Optional[dict]:
    """Fit y = m·x + b to the last ≤4 pivots; return slope + value at last bar."""
    pts = idxs[-4:]
    if len(pts) < 2:
        return None
    x = np.array(pts, dtype=float)
    y = np.array([vals[i] for i in pts], dtype=float)
    m, b = np.polyfit(x, y, 1)
    return {"slope": float(m), "atLast": float(m * last_x + b),
            "x0": int(pts[0]), "y0": float(y[0]), "x1": int(last_x), "y1": float(m * last_x + b)}


def _trendlines(df: pd.DataFrame, hi_idx: list[int], lo_idx: list[int]) -> dict:
    highs = df["high"].values.astype(float)
    lows = df["low"].values.astype(float)
    last = len(df) - 1
    px = float(df["close"].iloc[-1])
    res = _fit_line(hi_idx, highs, last)   # resistance line (down/flat from highs)
    sup = _fit_line(lo_idx, lows, last)    # support line (up/flat from lows)
    out: dict = {"resistance": res, "support": sup}
    # Test/break flags (within ~0.4% of the line, or beyond it).
    def _rel(line):
        if not line:
            return None
        d = (px - line["atLast"]) / line["atLast"] * 100
        if abs(d) <= 0.4:
            return "testing"
        return "above" if d > 0 else "below"
    out["vsResistance"] = _rel(res)
    out["vsSupport"] = _rel(sup)
    return out


# ── Regression channel + position ─────────────────────────────────────────────
def _channel(df: pd.DataFrame, look: int = 60) -> Optional[dict]:
    if df is None or len(df) < 20:
        return None
    seg = df.tail(look)
    y = seg["close"].values.astype(float)
    x = np.arange(len(y), dtype=float)
    m, b = np.polyfit(x, y, 1)
    mid = m * x + b
    resid = y - mid
    sd = float(np.std(resid)) or (float(np.mean(y)) * 0.01)
    px = float(y[-1])
    mid_now = float(mid[-1])
    upper, lower = mid_now + 2 * sd, mid_now - 2 * sd
    pos = (px - lower) / (upper - lower) if upper > lower else 0.5
    where = "top" if pos >= 0.7 else "bottom" if pos <= 0.3 else "mid"
    return {"slope": float(m), "mid": round(mid_now, 2), "upper": round(upper, 2),
            "lower": round(lower, 2), "position": where, "posPct": round(pos * 100, 0),
            "dir": "rising" if m > 0 else "falling" if m < 0 else "flat"}


# ── Key horizontal S/R near price ─────────────────────────────────────────────
def _levels(df: pd.DataFrame, hi_idx: list[int], lo_idx: list[int]) -> dict:
    px = float(df["close"].iloc[-1])
    highs = df["high"].values.astype(float)
    lows = df["low"].values.astype(float)
    c = df["close"]
    cands: list[float] = []
    for i in hi_idx[-6:]:
        cands.append(float(highs[i]))
    for i in lo_idx[-6:]:
        cands.append(float(lows[i]))
    for span in (20, 50, 200):
        if len(c) >= span:
            cands.append(float(c.rolling(span).mean().iloc[-1]))
    if len(df) >= 2:
        cands.append(float(df["high"].iloc[-2]))   # prior bar H/L
        cands.append(float(df["low"].iloc[-2]))
    above = sorted([v for v in cands if v > px * 1.001])
    below = sorted([v for v in cands if v < px * 0.999], reverse=True)
    return {"resistance": round(above[0], 2) if above else None,
            "support": round(below[0], 2) if below else None}


# ── Gaps ──────────────────────────────────────────────────────────────────────
def _gaps(df: pd.DataFrame, trend: str) -> Optional[dict]:
    if df is None or len(df) < 6:
        return None
    o = df["open"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    px = float(c[-1])
    # Scan recent bars for the most recent significant gap.
    for i in range(len(df) - 1, max(len(df) - 15, 1), -1):
        prev_c = c[i - 1]
        gap_pct = (o[i] - prev_c) / prev_c * 100
        if abs(gap_pct) < _GAP_MIN_PCT:
            continue
        up = gap_pct > 0
        # Filled if later price traded back through the prior close.
        filled = bool(np.min(l[i:]) <= prev_c) if up else bool(np.max(h[i:]) >= prev_c)
        # Classify by position in trend (rough): breakaway (start), runaway (mid), exhaustion (extended).
        kind = "runaway"
        if (up and trend != "up") or (not up and trend != "down"):
            kind = "breakaway"
        elif i >= len(df) - 3:
            kind = "exhaustion"
        return {"pct": round(gap_pct, 1), "dir": "up" if up else "down",
                "level": round(float(prev_c), 2), "filled": filled, "kind": kind}
    return None


# ── Volume regime ─────────────────────────────────────────────────────────────
def _volume_regime(df: pd.DataFrame) -> str:
    if df is None or len(df) < 25:
        return "normal"
    v = df["volume"].values.astype(float)
    c = df["close"].values.astype(float)
    avg = float(np.mean(v[-20:-1])) or 1.0
    last = float(v[-1])
    if last >= 2.5 * avg:
        return "climax"
    recent = v[-5:]
    up_days = [v[i] for i in range(len(df) - 5, len(df)) if c[i] >= c[i - 1]]
    dn_days = [v[i] for i in range(len(df) - 5, len(df)) if c[i] < c[i - 1]]
    uavg = np.mean(up_days) if up_days else 0
    davg = np.mean(dn_days) if dn_days else 0
    if float(np.mean(recent)) < 0.6 * avg:
        return "dry-up"
    if uavg > davg * 1.3:
        return "accumulation"
    if davg > uavg * 1.3:
        return "distribution"
    return "normal"


# ── Patterns (robust set) ─────────────────────────────────────────────────────
def _patterns(df: pd.DataFrame, hi_idx: list[int], lo_idx: list[int], tl: dict) -> list[dict]:
    out: list[dict] = []
    highs = df["high"].values.astype(float)
    lows = df["low"].values.astype(float)
    px = float(df["close"].iloc[-1])

    # Double top / bottom — two comparable swings (~within 3%) with a valley/peak
    # between. GATED: the pattern must be RECENT (≤40 bars) AND price must still be
    # NEAR it (≤10%) — otherwise it's an old, already-resolved base far from the
    # current price (the SNOW bug: a ~155 double-bottom while price is 245).
    last = len(df) - 1
    if len(hi_idx) >= 2:
        a, b = hi_idx[-2], hi_idx[-1]
        if (abs(highs[a] - highs[b]) / highs[b] <= 0.03 and b - a >= 3
                and (last - b) <= 40 and abs(px - highs[b]) / px <= 0.10):
            neck = float(np.min(lows[a:b + 1]))
            out.append({"type": "Double Top", "tone": "bearish", "confidence": 0.6,
                        "level": round(float(highs[b]), 2), "neckline": round(neck, 2),
                        "target": round(neck - (float(highs[b]) - neck), 2)})
    if len(lo_idx) >= 2:
        a, b = lo_idx[-2], lo_idx[-1]
        if (abs(lows[a] - lows[b]) / lows[b] <= 0.03 and b - a >= 3
                and (last - b) <= 40 and abs(px - lows[b]) / px <= 0.10):
            neck = float(np.max(highs[a:b + 1]))
            out.append({"type": "Double Bottom", "tone": "bullish", "confidence": 0.6,
                        "level": round(float(lows[b]), 2), "neckline": round(neck, 2),
                        "target": round(neck + (neck - float(lows[b])), 2)})

    # Triangle — converging support & resistance trendlines, with price INSIDE/near
    # the apex (else the lines are stale relative to current price).
    res, sup = tl.get("resistance"), tl.get("support")
    if (res and sup and res["atLast"] > sup["atLast"]
            and sup["atLast"] * 0.95 <= px <= res["atLast"] * 1.05):
        rs, ss = res["slope"], sup["slope"]
        converging = (rs < -1e-6 and ss > 1e-6) or (abs(rs) < 1e-6 and ss > 1e-6) or (rs < -1e-6 and abs(ss) < 1e-6)
        if converging or (rs < 0 and ss > 0):
            if abs(rs) < 1e-6 and ss > 0:
                kind = "Ascending Triangle"; tone = "bullish"
            elif rs < 0 and abs(ss) < 1e-6:
                kind = "Descending Triangle"; tone = "bearish"
            else:
                kind = "Symmetrical Triangle"; tone = "neutral"
            height = res["atLast"] - sup["atLast"]
            out.append({"type": kind, "tone": tone, "confidence": 0.55,
                        "upper": round(res["atLast"], 2), "lower": round(sup["atLast"], 2),
                        "target": round(px + height, 2) if tone != "bearish" else round(px - height, 2)})

    # Flag/pennant — a sharp pole then a tight, lower-volume consolidation.
    if len(df) >= 18:
        pole = (px - float(df["close"].iloc[-13])) / float(df["close"].iloc[-13]) * 100
        rng = (df["high"].tail(6).max() - df["low"].tail(6).min()) / px * 100
        if abs(pole) >= 8 and rng <= 5:
            out.append({"type": "Bull Flag" if pole > 0 else "Bear Flag",
                        "tone": "bullish" if pole > 0 else "bearish", "confidence": 0.5,
                        "target": round(px * (1 + pole / 100 / 2), 2)})
    return out


# ── Per-timeframe + MTF ───────────────────────────────────────────────────────
def _tf_trend(symbol: str, timeframe: str, days: int) -> str:
    try:
        from engine.alpaca_client import get_bars
        df = get_bars(symbol, timeframe, days=days)
        return _trend(df) if df is not None and len(df) else "sideways"
    except Exception:
        return "sideways"


def analyze(symbol: str) -> Optional[dict]:
    """Full Phase-1 chart read for `symbol`. Returns the structured read or None."""
    sym = (symbol or "").upper().strip()
    if not sym:
        return None
    try:
        from engine.alpaca_client import get_bars
        daily = get_bars(sym, "1Day", days=180)
    except Exception as e:
        logger.debug(f"[chart_read] {sym} bars fetch failed: {e}")
        return None
    if daily is None or len(daily) < 50:
        return None

    px = float(daily["close"].iloc[-1])
    hi_idx, lo_idx = _swings(daily)
    trend_d = _trend(daily)
    tl = _trendlines(daily, hi_idx, lo_idx)
    ch = _channel(daily)
    lv = _levels(daily, hi_idx, lo_idx)
    gap = _gaps(daily, trend_d)
    vol = _volume_regime(daily)
    pats = _patterns(daily, hi_idx, lo_idx, tl)

    # MTF
    t15 = _tf_trend(sym, "15Min", 5)
    t1h = _tf_trend(sym, "1Hour", 20)
    trends = [t15, t1h, trend_d]
    ups = trends.count("up"); dns = trends.count("down")
    if ups == 3:   mtf, mtf_dir = "aligned", "up"
    elif dns == 3: mtf, mtf_dir = "aligned", "down"
    elif ups >= 2: mtf, mtf_dir = "leaning", "up"
    elif dns >= 2: mtf, mtf_dir = "leaning", "down"
    else:          mtf, mtf_dir = "mixed", "none"

    # ── Two INDEPENDENT verdicts, then COMPARE ───────────────────────────────
    # (1) TA verdict — purely the chart structure (daily trend + 1h + patterns +
    #     volume). NOT derived from the quant data — a genuine second opinion.
    # (2) QUANT verdict — the SAME cached row the hub Game Plan reads.
    # We surface BOTH + whether they AGREE: agreement = confirmation; DISAGREEMENT
    # is the interesting case to verify (and, later, to measure which side was right).
    # Headline TA verdict = DAILY/SWING horizon ONLY — matches the quant verdict's
    # horizon so AGREE/DISAGREE is apples-to-apples. 1h/15m are NOT in the headline;
    # they're reported separately as short-term CONFIRMATION (context).
    ta_score = {"up": 2, "down": -2}.get(trend_d, 0)
    for p in pats:
        ta_score += {"bullish": 1, "bearish": -1}.get(p.get("tone"), 0)
    ta_score += {"accumulation": 1, "distribution": -1}.get(vol, 0)
    ta_bias = "bullish" if ta_score >= 2 else "bearish" if ta_score <= -2 else "neutral"

    # Short-term (1h + 15m) confirmation of the DAILY verdict — context, not headline.
    _st = {"up": 1, "down": -1}.get(t1h, 0) + {"up": 1, "down": -1}.get(t15, 0)
    if ta_bias == "neutral" or _st == 0:
        short_term = "neutral"
    elif (ta_bias == "bullish" and _st > 0) or (ta_bias == "bearish" and _st < 0):
        short_term = "confirming"
    else:
        short_term = "diverging"

    quant_bias = None
    try:
        from engine import quant_score_service as _qs
        qrow, _as_of = _qs.cached_score(sym)
    except Exception:
        qrow = None
    if qrow:
        _ma = qrow.get("ma20"); _ts = float(qrow.get("trendScore") or 0); _setup = qrow.get("setupType")
        if qrow.get("peakStage") == "peak" or _setup == "breakdown":
            quant_bias = "bearish"
        elif qrow.get("turnaroundStage") == "buyzone" or _setup == "breakout":
            quant_bias = "bullish"
        # ±0.5% hysteresis band around the 20-day MA so the verdict doesn't flicker
        # when price hovers right at it.
        elif _ma and px > _ma * 1.005 and _ts >= 55:
            quant_bias = "bullish"
        elif _ma and px < _ma * 0.995:
            quant_bias = "bearish"
        else:
            quant_bias = "neutral"

    if quant_bias is None:
        agreement = "n/a"
    elif ta_bias == "neutral" or quant_bias == "neutral":
        agreement = "partial"
    elif ta_bias == quant_bias:
        agreement = "agree"
    else:
        agreement = "disagree"

    bias = ta_bias   # the Expert Read's OWN, independent technical call
    conf = int(min(90, 52 + min(16, abs(ta_score) * 4)
                   + (16 if agreement == "agree" else -14 if agreement == "disagree" else 0)))

    # Plain-English narrative — LEAD with the agreement vs the quant read.
    bullets: list[str] = []
    if agreement == "agree":
        bullets.append(f"✅ Technicals AGREE with the quant read — both {ta_bias}. Confirmation; higher confidence.")
    elif agreement == "disagree":
        bullets.append(f"⚠️ Technicals DISAGREE with the quant read — TA says {ta_bias}, quant says {quant_bias}. "
                       f"Conflicting — treat as low-confidence and watch which side resolves.")
    elif agreement == "partial":
        bullets.append(f"Technicals {ta_bias} vs quant {quant_bias} — only partial overlap (one is neutral).")
    bullets.append(f"Daily trend: {trend_d} (the verdict horizon). Short-term 1h/15m is "
                   f"{short_term} it (15m {t15} · 1h {t1h}).")
    if ch:
        bullets.append(f"{ch['dir'].capitalize()} regression channel — price in the {ch['position']} "
                       f"(~{int(ch['posPct'])}% up the channel; {ch['lower']}–{ch['upper']}).")
    if tl.get("vsSupport") == "testing":
        bullets.append("Testing its rising support trendline — a hold here is the lower-risk long; a break warns.")
    if tl.get("vsResistance") == "testing":
        bullets.append("Testing its resistance trendline — a clean break/close above opens upside.")
    for p in pats:
        tgt = f", target ~{p['target']}" if p.get("target") else ""
        bullets.append(f"Pattern: {p['type']} ({p['tone']}, {int(p['confidence']*100)}% conf){tgt}.")
    if lv.get("support") or lv.get("resistance"):
        bullets.append(f"Key levels — support {lv.get('support') or '—'} · resistance {lv.get('resistance') or '—'}.")
    if gap:
        f = "filled" if gap["filled"] else "UNFILLED"
        bullets.append(f"{gap['kind'].capitalize()} gap {gap['dir']} {abs(gap['pct'])}% at {gap['level']} ({f}).")
    bullets.append(f"Volume regime: {vol}.")

    return {
        "ticker": sym, "price": round(px, 2),
        "bias": bias, "confidence": int(conf),
        "taBias": ta_bias, "quantBias": quant_bias, "agreement": agreement,
        "shortTerm": short_term,
        "trend": {"d1": trend_d, "h1": t1h, "m15": t15},
        "mtf": {"state": mtf, "dir": mtf_dir},
        "channel": ch, "trendlines": tl, "levels": lv,
        "gap": gap, "volumeRegime": vol, "patterns": pats,
        "narrative": bullets,
        # Geometry for the chart layer to DRAW later (Phase 1b).
        "overlays": {
            "trendlines": [v for v in (tl.get("support"), tl.get("resistance")) if v],
            "channel": ch, "levels": lv, "patterns": pats, "gap": gap,
        },
    }
