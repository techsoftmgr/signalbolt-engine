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
    """'up' / 'down' / 'sideways' from EMA20 (vs EMA50 when available) + slope."""
    if df is None or len(df) < 20:
        return "sideways"
    c = df["close"]
    ema20 = c.ewm(span=20, adjust=False).mean()
    px = float(c.iloc[-1]); e20 = float(ema20.iloc[-1])
    slope20 = (e20 - float(ema20.iloc[-10])) / float(ema20.iloc[-10]) * 100 if len(ema20) >= 10 else 0.0
    if len(df) >= 50:
        e50 = float(c.ewm(span=50, adjust=False).mean().iloc[-1])
        if px > e20 > e50 and slope20 > 0.2:
            return "up"
        if px < e20 < e50 and slope20 < -0.2:
            return "down"
        return "sideways"
    # Newer ticker (<50 bars, e.g. DRAM): EMA20 + slope only.
    if px > e20 and slope20 > 0.2:
        return "up"
    if px < e20 and slope20 < -0.2:
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


def _pattern_explain(p: dict) -> str:
    """Beginner-friendly one-liner for a detected pattern: what it is + what to
    watch + the measured target. Keeps the jargon label but makes it readable to
    a non-technical user (so 'Bull Flag' isn't meaningless)."""
    t   = p.get("type", "")
    tgt = p.get("target")
    tgt_s = f"${tgt}" if tgt is not None else "the measured move"
    if t == "Bull Flag":
        return (f"A brief pause after a sharp run-up. Flags usually resolve in the "
                f"prior direction (up here), and a push out of the range targets ~{tgt_s}.")
    if t == "Bear Flag":
        return (f"A brief pause after a sharp drop. Flags usually resolve in the "
                f"prior direction (down here), and a break lower targets ~{tgt_s}.")
    if t == "Double Top":
        return (f"Price stalled twice near ${p.get('level')} and failed to break higher, "
                f"a possible top. A close below the ${p.get('neckline')} neckline targets ~{tgt_s}.")
    if t == "Double Bottom":
        return (f"Price held twice near ${p.get('level')} and bounced, a possible bottom. "
                f"A close above the ${p.get('neckline')} neckline targets ~{tgt_s}.")
    if t == "Ascending Triangle":
        return (f"Rising lows pressing into flat resistance at ${p.get('upper')}, with buyers "
                f"gaining control. A break above ${p.get('upper')} targets ~{tgt_s}.")
    if t == "Descending Triangle":
        return (f"Falling highs pressing into flat support at ${p.get('lower')}, with sellers "
                f"gaining control. A break below ${p.get('lower')} targets ~{tgt_s}.")
    if t == "Symmetrical Triangle":
        return (f"The range is tightening between ${p.get('lower')} and ${p.get('upper')}. "
                f"A breakout is brewing and its direction decides the move. Measured move ~{tgt_s}.")
    return f"{t} forming, measured target ~{tgt_s}."


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

    # Attach a plain-English read to every detected pattern (so "Bull Flag" means
    # something to a non-technical user).
    for p in out:
        p["explain"] = _pattern_explain(p)
    return out


# ── Fibonacci (auto-anchored to the most recent swing) ────────────────────────
def _fib_explain(direction: str, gp: dict, target, lo: float, hi: float, inval: float,
                 status: dict | None = None) -> str:
    """Plain-English read of the Fib overlay — what the levels mean + how to use
    them (dip/bounce zone, target, invalidation). 'Golden pocket' demystified.
    Price-aware: if price has slipped below (up) / above (down) the zone, the
    dip-buy/bounce framing is no longer valid — say so instead of implying a buy."""
    band = f"${gp['low']}–${gp['high']}"
    pos = (status or {}).get("position")
    failed = (status or {}).get("failed")
    if direction == "up":
        if failed:
            return (f"Ran from ${round(lo,2)} up to ${round(hi,2)}, then gave it all back. Price is now below "
                    f"the 78.6% level (~${round(inval,2)}), so the {band} golden pocket dip-buy zone has FAILED. "
                    f"It's overhead resistance now, not a buy area. A reclaim of {band} would be the first sign the run is repairing.")
        if pos == "below":
            return (f"Ran from ${round(lo,2)} up to ${round(hi,2)}. Price has slipped below the {band} golden pocket "
                    f"(50–61.8%) dip-buy zone, so that zone sits overhead. Price would need to reclaim {band} to put the "
                    f"dip-buy thesis back in play. A close below ~${round(inval,2)} (78.6%) says the run failed.")
        return (f"Ran from ${round(lo,2)} up to ${round(hi,2)}. The {band} band (the 50–61.8% retracement, the "
                f"golden pocket) is the most-watched dip-buy zone, where a hold or bounce is a lower-risk add. "
                f"A close below ~${round(inval,2)} (78.6%) says the run failed. "
                f"If it resumes, the 1.618 extension targets ~${target}.")
    if failed:
        return (f"Fell from ${round(hi,2)} down to ${round(lo,2)}, then reclaimed. Price is now above the 78.6% level "
                f"(~${round(inval,2)}), so the {band} bounce/sell zone has FAILED and the drop is reversing. If it resumes, the 1.618 extension targets ~${target}.")
    if pos == "above":
        return (f"Fell from ${round(hi,2)} down to ${round(lo,2)}. Price sits above the {band} (50–61.8%) zone. "
                f"A bounce into {band} is where sellers most often re-enter, so it acts as resistance, not a dip-buy. "
                f"A close above ~${round(inval,2)} (78.6%) says the drop is reversing.")
    return (f"Fell from ${round(hi,2)} down to ${round(lo,2)}. This is a downtrend, so there's no dip-buy zone yet. "
            f"The {band} band (50–61.8% retracement) is where a relief bounce most often stalls and sellers re-enter, "
            f"so treat it as overhead resistance, not a buy. A close above ~${round(inval,2)} (78.6%) would be the "
            f"first sign the drop is reversing. If the decline resumes, the 1.618 extension targets ~${target}.")


def _fib(df: pd.DataFrame, lookback: int = 60, price: float | None = None) -> Optional[dict]:
    """Auto-anchored Fibonacci over the most recent significant swing in the last
    `lookback` bars. Direction = the most recent leg (whichever of the swing
    high/low is more recent). Returns retracement levels + golden pocket (dip/
    bounce zone) + 1.272/1.618 extension targets + a plain-English read.

    `price` makes the zone price-aware: `status.position` (above/inside/below the
    golden pocket) + `status.failed` (price beyond the 78.6% level → the leg's
    dip-buy/bounce thesis is broken) so the UI never shows a "buy area" the price
    has already fallen through."""
    sub = df.tail(lookback)
    if len(sub) < 10:
        return None
    highs = sub["high"].values.astype(float)
    lows  = sub["low"].values.astype(float)
    hi = float(highs.max()); lo = float(lows.min())
    rng = hi - lo
    if rng <= 0:
        return None
    up_leg = int(highs.argmax()) > int(lows.argmin())   # high more recent → last leg was up
    direction = "up" if up_leg else "down"

    def at(r):   # retracement price at ratio r
        return (hi - rng * r) if up_leg else (lo + rng * r)
    levels = [{"ratio": r, "label": f"{r*100:.1f}%".rstrip("0").rstrip("."),
               "price": round(at(r), 2)} for r in (0.236, 0.382, 0.5, 0.618, 0.786)]

    def ext(r):   # extension/projection beyond the swing, in the leg direction
        return (lo + rng * r) if up_leg else (hi - rng * r)
    extensions = [{"ratio": r, "price": round(ext(r), 2)} for r in (1.272, 1.618)]

    gp_a, gp_b = at(0.5), at(0.618)
    golden = {"low": round(min(gp_a, gp_b), 2), "high": round(max(gp_a, gp_b), 2)}
    target = extensions[-1]["price"]          # 1.618 projection
    inval  = at(0.786)                         # break beyond 78.6% = leg failed

    # Price-aware status: where is price vs the golden pocket, and has the leg failed?
    status = None
    if price is not None:
        try:
            px = float(price)
            if golden["low"] <= px <= golden["high"]:
                pos = "inside"
            elif px > golden["high"]:
                pos = "above"
            else:
                pos = "below"
            failed = (px < inval) if up_leg else (px > inval)
            status = {"position": pos, "failed": bool(failed)}
        except (TypeError, ValueError):
            status = None

    return {
        "direction": direction,
        "swingHigh": round(hi, 2), "swingLow": round(lo, 2),
        "levels": levels, "extensions": extensions,
        "goldenPocket": golden, "target": target, "invalidation": round(inval, 2),
        "status": status,
        "explain": _fib_explain(direction, golden, target, lo, hi, inval, status),
    }


# ── "What to watch" — two-sided IF/THEN trigger plan ──────────────────────────
def _scenarios(px: float, lv: dict, fib: Optional[dict], pats: list) -> Optional[dict]:
    """Turn the read into a NEUTRAL if/then plan: the level to RECLAIM for the
    bullish case + the level to LOSE for the bearish case, each with a next
    upside/downside level to watch. Built only from already-computed geometry
    (nearest support/resistance, Fib levels/extensions, pattern necklines). It
    never says buy/sell — it hands the user the decision rule ("let price pick the
    side"), which is also why it's not advice."""
    res = lv.get("resistance")     # nearest level ABOVE price
    sup = lv.get("support")        # nearest level BELOW price
    bull_pat = next((p for p in (pats or []) if p.get("tone") == "bullish"), None)
    bear_pat = next((p for p in (pats or []) if p.get("tone") == "bearish"), None)

    def _round(v):
        return round(float(v), 2) if v is not None else None

    # ── Bull trigger: a level ABOVE price to reclaim ──
    bull_trig = res
    if bull_pat and (bull_pat.get("neckline") or 0) > px:
        bull_trig = bull_pat["neckline"]
    if bull_trig is None and fib:
        ups = sorted(l["price"] for l in fib.get("levels", []) if l["price"] > px)
        bull_trig = ups[0] if ups else None
    # Bull target: next level to watch ABOVE the trigger
    bull_tgt = None
    if bull_pat and (bull_pat.get("target") or 0) > px:
        bull_tgt = bull_pat["target"]
    elif fib:
        if fib.get("direction") == "up" and fib.get("extensions"):
            bull_tgt = fib["extensions"][0]["price"]                       # 1.272 up-extension
        elif (fib.get("goldenPocket", {}).get("high") or 0) > (bull_trig or px):
            bull_tgt = fib["goldenPocket"]["high"]                          # bounce destination (down-leg)

    # ── Bear trigger: a level BELOW price to lose ──
    bear_trig = sup
    if bear_pat and 0 < (bear_pat.get("neckline") or 0) < px:
        bear_trig = bear_pat["neckline"]
    if bear_trig is None and fib and (fib.get("swingLow") or 0) < px:
        bear_trig = fib["swingLow"]                                         # at the lows → the swing low
    # Bear target: next level to watch BELOW the trigger (NOT just below price —
    # a downside target must sit beyond the level you lose, else it's nonsense).
    bear_ref = bear_trig if bear_trig is not None else px
    bear_tgt = None
    if bear_pat and 0 < (bear_pat.get("target") or 0) < bear_ref:
        bear_tgt = bear_pat["target"]
    elif fib:
        if fib.get("direction") == "down" and fib.get("extensions"):
            bear_tgt = fib["extensions"][0]["price"]                       # 1.272 down-extension
        else:
            downs = sorted((l["price"] for l in fib.get("levels", []) if l["price"] < bear_ref), reverse=True)
            bear_tgt = downs[0] if downs else None
            if bear_tgt is None and (fib.get("swingLow") or 0) and fib["swingLow"] < bear_ref:
                bear_tgt = fib["swingLow"]                                  # fall back to the swing low

    # ── Sanity: targets must point the right way (above the bull trigger / below
    # the bear trigger). Drop a target that would contradict its own direction. ──
    if bull_tgt is not None and bull_trig is not None and bull_tgt <= bull_trig:
        bull_tgt = None
    if bear_tgt is not None and bear_trig is not None and bear_tgt >= bear_trig:
        bear_tgt = None

    bull = {"trigger": _round(bull_trig), "then": "bullish (upside opens up / a bottom is holding)",
            "target": _round(bull_tgt)} if bull_trig else None
    bear = {"trigger": _round(bear_trig), "then": "bearish (downside resumes)",
            "target": _round(bear_tgt)} if bear_trig else None
    if not bull and not bear:
        return None

    parts = []
    if bull:
        parts.append(f"reclaim ${bull['trigger']} = bullish" + (f" (→ ${bull['target']})" if bull.get("target") else ""))
    if bear:
        parts.append(f"lose ${bear['trigger']} = bearish" + (f" (→ ${bear['target']})" if bear.get("target") else ""))
    return {"bull": bull, "bear": bear,
            "summary": "Let price pick the side: " + "; ".join(parts) + ".",
            "note": "Until one of these triggers, it's undecided. React to the level, don't anticipate it."}


# ── Per-timeframe + MTF ───────────────────────────────────────────────────────
def _tf_trend(symbol: str, timeframe: str, days: int) -> str:
    try:
        from engine.alpaca_client import get_bars
        df = get_bars(symbol, timeframe, days=days)
        return _trend(df) if df is not None and len(df) else "sideways"
    except Exception:
        return "sideways"


# base-timeframe → (alpaca tf, lookback days, min bars). Daily = the swing read
# (default); 1Hour = an intraday-swing read that updates through the session
# (incl. pre/after-hours via SIP) without the noise of sub-hour candles.
_BASE_TF = {
    "1Day":  ("1Day", 180, 30),
    "1Hour": ("1Hour", 60, 60),
}


def _recent_catalyst(sym: str) -> Optional[dict]:
    """Most-recent material news headline within ~48h, if any. Used to FLAG that a
    catalyst is driving the tape (so technical levels are less reliable) — we do
    NOT change the technical math. Best-effort; never raises."""
    try:
        from datetime import datetime, timezone
        from engine.alpaca_client import get_news
        news = get_news(sym, limit=4) or []
        if not news:
            return None
        news = sorted(news, key=lambda n: (n.get("created_at") or n.get("time") or ""), reverse=True)
        top = news[0]
        ts = top.get("created_at") or top.get("time")
        if ts:
            try:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                if (datetime.now(timezone.utc) - dt).total_seconds() > 48 * 3600:
                    return None
            except Exception:
                pass
        head = (top.get("headline") or "").strip()
        if not head:
            return None
        return {"has_news": True, "headline": head[:160], "url": top.get("url"),
                "source": top.get("source") or "news", "time": ts}
    except Exception:
        return None


def _settled(df, tf: str, now=None):
    """Drop the still-FORMING last bar so the read is a STABLE plan within its
    period — the daily read excludes today's bar until the 4 PM ET close (so it
    doesn't drift intraday); the 1h read excludes the current forming hour. After
    the bar closes it's included → the read updates once, matching its label.
    Best-effort: returns df unchanged on any issue."""
    try:
        if df is None or len(df) < 2:
            return df
        from datetime import datetime, timezone
        from zoneinfo import ZoneInfo
        now = now or datetime.now(timezone.utc)
        et = now.astimezone(ZoneInfo("America/New_York"))
        last = df.index[-1]
        last_utc = last.tz_convert("UTC") if hasattr(last, "tz_convert") else last
        if tf == "1Day":
            forming = (last_utc.date() == now.date()) and (et.hour < 16)
        else:  # 1Hour (and finer): the bar for the current hour is still forming
            forming = (last_utc.date() == now.date() and last_utc.hour == now.hour)
        return df.iloc[:-1] if forming else df
    except Exception:
        return df


def analyze(symbol: str, timeframe: str = "1Day") -> Optional[dict]:
    """Full Phase-1 chart read for `symbol` on `timeframe` (1Day default, or
    1Hour for an intraday-swing read). Returns the structured read or None."""
    sym = (symbol or "").upper().strip()
    if not sym:
        return None
    tf, days, min_bars = _BASE_TF.get(timeframe, _BASE_TF["1Day"])
    try:
        from engine.alpaca_client import get_bars
        daily = get_bars(sym, tf, days=days)   # `daily` = the base df for this read
    except Exception as e:
        logger.debug(f"[chart_read] {sym} {tf} bars fetch failed: {e}")
        return None
    daily = _settled(daily, tf)                # anchor to the last CLOSED bar (stable plan)
    if daily is None or len(daily) < min_bars:
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
    fib = _fib(daily, price=px)
    scenarios = _scenarios(px, lv, fib, pats)

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
    qrow = None
    try:
        from engine import quant_score_service as _qs
        qrow, _as_of = _qs.cached_score(sym)
        if qrow is None:
            # Not in the cached universe scan (e.g. DRAM) — score it on-demand so
            # the AGREE/DISAGREE comparison still works (mirrors the hub's live
            # overview, which also live-scores non-universe tickers). Reuses the
            # daily bars we already fetched.
            try:
                from engine.alpaca_client import get_bars as _gb
                _idf = _gb(sym, "15Min", days=5)
                qrow = _qs._score_ticker(sym, px, daily, _idf, daily_long_df=daily)
            except Exception:
                qrow = None
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

    # ── Actionable idea (educational) from the TA structure — gated on a decisive,
    # non-conflicting read. This is why the hub can always offer SOMETHING even when
    # no tracked signal has fired (the #3 gap). Conflicting/neutral → "wait".
    idea = {"action": "WAIT",
            "text": "No clean idea right now: the read is neutral or the chart and model verdicts conflict. Wait for them to line up."}
    if bias != "neutral" and agreement != "disagree":
        if bias == "bullish":
            _sup = lv.get("support") or (ch and ch.get("lower"))
            _raw = (_sup * 0.99) if _sup else px * 0.95
            _stop = round(max(px * 0.92, min(_raw, px * 0.985)), 2)   # clamp risk to 1.5–8%
            _patt = next((p["target"] for p in pats if p.get("tone") == "bullish" and p.get("target")), None)
            _tgt = round(float(_patt or lv.get("resistance") or (ch and ch.get("upper")) or px * 1.06), 2)
            _rr = (_tgt - px) / (px - _stop) if px > _stop else 0
            if _rr >= 1.0:
                idea = {"action": "LONG", "option": "CALL", "entry": round(px, 2), "stop": _stop,
                        "target": _tgt, "rr": round(_rr, 1),
                        "text": f"Long plan (if/then). If it holds near {round(px,2)}, invalidation is {_stop} "
                                f"(below support) and the first level is {_tgt} (R:R {round(_rr,1)}). Educational, not a prediction."}
            else:
                idea = {"action": "WAIT", "text": f"Up-bias, but the risk/reward is poor at {round(px,2)} "
                        f"(little room to resistance, far from support). No clean trigger yet; a pullback toward {_stop} would set one up."}
        else:
            _res = lv.get("resistance") or (ch and ch.get("upper"))
            _raw = (_res * 1.01) if _res else px * 1.05
            _stop = round(min(px * 1.08, max(_raw, px * 1.015)), 2)   # clamp risk to 1.5–8%
            _patt = next((p["target"] for p in pats if p.get("tone") == "bearish" and p.get("target")), None)
            _tgt = round(float(_patt or lv.get("support") or (ch and ch.get("lower")) or px * 0.94), 2)
            _rr = (px - _tgt) / (_stop - px) if _stop > px else 0
            if _rr >= 1.0:
                idea = {"action": "SHORT", "option": "PUT", "entry": round(px, 2), "stop": _stop,
                        "target": _tgt, "rr": round(_rr, 1),
                        "text": f"Short plan (if/then). If it rejects near {round(px,2)}, invalidation is {_stop} "
                                f"(above resistance) and the first level is {_tgt} (R:R {round(_rr,1)}). Educational, not a prediction."}
            else:
                idea = {"action": "WAIT", "text": f"Down-bias, but the risk/reward is poor at {round(px,2)}. "
                        f"No clean trigger yet; a bounce toward {_stop} would set one up."}

    # Plain-English narrative — LEAD with the agreement vs the quant read.
    bullets: list[str] = []
    if agreement == "agree":
        bullets.append(f"✅ Technicals agree with the quant read: both {ta_bias}. That's confirmation and higher confidence.")
    elif agreement == "disagree":
        bullets.append(f"⚠️ Technicals disagree with the quant read: TA says {ta_bias}, quant says {quant_bias}. "
                       f"Conflicting, so treat it as low-confidence and watch which side resolves.")
    elif agreement == "partial":
        bullets.append(f"Technicals {ta_bias} vs quant {quant_bias}: only partial overlap (one is neutral).")
    bullets.append(f"Daily trend: {trend_d} (the verdict horizon). Short-term 1h/15m is "
                   f"{short_term} it (15m {t15}, 1h {t1h}).")
    if ch:
        bullets.append(f"{ch['dir'].capitalize()} regression channel, price in the {ch['position']} "
                       f"(~{int(ch['posPct'])}% up the channel; {ch['lower']}–{ch['upper']}).")
    if tl.get("vsSupport") == "testing":
        bullets.append("Testing its rising support trendline. A hold here is the lower-risk long; a break is a warning.")
    if tl.get("vsResistance") == "testing":
        bullets.append("Testing its resistance trendline. A clean break and close above opens upside.")
    for p in pats:
        tgt = f", target ~{p['target']}" if p.get("target") else ""
        bullets.append(f"Pattern: {p['type']} ({p['tone']}, {int(p['confidence']*100)}% conf){tgt}.")
    if lv.get("support") or lv.get("resistance"):
        bullets.append(f"Key levels: support {lv.get('support') or '—'}, resistance {lv.get('resistance') or '—'}.")
    if gap:
        f = "filled" if gap["filled"] else "UNFILLED"
        bullets.append(f"{gap['kind'].capitalize()} gap {gap['dir']} {abs(gap['pct'])}% at {gap['level']} ({f}).")
    bullets.append(f"Volume regime: {vol}.")

    try:
        as_of = daily.index[-1].isoformat()    # the closed bar this read is anchored to
    except Exception:
        as_of = None
    return {
        "ticker": sym, "price": round(px, 2), "timeframe": tf, "as_of": as_of,
        "catalyst": _recent_catalyst(sym),
        "bias": bias, "confidence": int(conf),
        "taBias": ta_bias, "quantBias": quant_bias, "agreement": agreement,
        "shortTerm": short_term, "idea": idea,
        "trend": {"d1": trend_d, "h1": t1h, "m15": t15},
        "mtf": {"state": mtf, "dir": mtf_dir},
        "channel": ch, "trendlines": tl, "levels": lv,
        "gap": gap, "volumeRegime": vol, "patterns": pats, "fib": fib,
        "scenarios": scenarios,
        "narrative": bullets,
        # Geometry for the chart layer to DRAW later (Phase 1b).
        "overlays": {
            "trendlines": [v for v in (tl.get("support"), tl.get("resistance")) if v],
            "channel": ch, "levels": lv, "patterns": pats, "gap": gap, "fib": fib,
        },
    }


_FULL_TTL = 300   # 5 min — the read is a settled/swing analysis; brief staleness is fine


def build_full(symbol: str, timeframe: str = "1Day", sb=None, force: bool = False) -> Optional[dict]:
    """The full Expert-Read payload the hub needs (analyze + decision_support +
    read track-record), CACHED per (ticker, timeframe) so the hub doesn't recompute
    the multi-fetch read on every 1H/1D toggle. The worker pre-warms this for the
    core tickers so the first open is instant too. Does NOT scrub wording — the
    endpoint applies plainspeak per-request (admin vs not). Best-effort; never raises."""
    sym = (symbol or "").upper().strip()
    ck = f"chartread:full:v1:{sym}:{timeframe}"
    if not force:
        try:
            from engine import cache
            cached = cache.kv.get_json(ck)
            if cached is not None:
                return cached
        except Exception:
            pass

    r = analyze(sym, timeframe=timeframe)
    if r and not r.get("unavailable"):
        try:
            from engine import decision_support
            hist = decision_support.historical_similar_setups(sb, sym, r.get("taBias"))
            r["decision_support"] = decision_support.derive(r, historical=hist)
        except Exception as e:
            logger.debug(f"[chart_read] decision_support derive failed for {sym}: {e}")
        try:
            from engine import read_accuracy
            tr = read_accuracy.stats_cached(sb)
            if tr and tr.get("available"):
                r["readTrackRecord"] = tr
        except Exception as e:
            logger.debug(f"[chart_read] read track record failed for {sym}: {e}")

    if r:
        try:
            from engine import cache
            cache.kv.set_json(ck, r, _FULL_TTL)
        except Exception:
            pass
    return r


# ── Agreement track record — daily snapshot logger + forward-outcome scorer ──
_HORIZON_DAYS = 5   # trading-ish days to judge who was right


def log_snapshot(sb, tickers: list[str]) -> dict:
    """Log each ticker's TA-vs-Quant read to chart_read_log — one row per ticker
    per day. Best-effort; no-ops gracefully if the table doesn't exist yet."""
    from datetime import datetime, timezone
    stats = {"logged": 0}
    if sb is None:
        return stats
    today = datetime.now(timezone.utc).date().isoformat()
    for tk in tickers or []:
        try:
            r = analyze(tk)
            if not r:
                continue
            exists = (sb.table("chart_read_log").select("id")
                      .eq("ticker", r["ticker"]).gte("created_at", today + "T00:00:00Z")
                      .limit(1).execute().data)
            if exists:
                continue
            sb.table("chart_read_log").insert({
                "ticker": r["ticker"], "ta_bias": r["taBias"], "quant_bias": r.get("quantBias"),
                "agreement": r["agreement"], "short_term": r.get("shortTerm"), "price": r["price"],
            }).execute()
            stats["logged"] += 1
        except Exception as e:
            logger.debug(f"[chart_read] log_snapshot {tk} failed: {e}")
    logger.info(f"[chart_read] snapshot logged {stats}")
    return stats


def score_snapshots(sb) -> dict:
    """Fill forward-outcome columns for chart_read_log rows past their horizon, and
    mark the winner (TA / Quant / both / neither) when the two disagreed. Best-effort."""
    from datetime import datetime, timezone, timedelta
    from engine.alpaca_client import get_latest_price
    stats = {"scored": 0}
    if sb is None:
        return stats
    cutoff = (datetime.now(timezone.utc) - timedelta(days=_HORIZON_DAYS)).isoformat()
    try:
        rows = (sb.table("chart_read_log").select("*")
                .is_("winner", "null").lte("created_at", cutoff)
                .limit(200).execute().data) or []
    except Exception as e:
        logger.debug(f"[chart_read] score_snapshots fetch failed: {e}")
        return stats
    for row in rows:
        try:
            entry = float(row.get("price") or 0)
            now_px = get_latest_price(row["ticker"])
            if not entry or not now_px:
                continue
            ret = (now_px - entry) / entry * 100.0
            up = ret > 1.0; down = ret < -1.0
            def _right(bias):
                return (bias == "bullish" and up) or (bias == "bearish" and down)
            ta_ok = _right(row.get("ta_bias")); q_ok = _right(row.get("quant_bias"))
            winner = ("both" if ta_ok and q_ok else "ta" if ta_ok else "quant" if q_ok else "neither")
            sb.table("chart_read_log").update({
                "horizon_days": _HORIZON_DAYS, "forward_price": round(float(now_px), 2),
                "forward_return_pct": round(ret, 3), "winner": winner,
                "scored_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", row["id"]).execute()
            stats["scored"] += 1
        except Exception as e:
            logger.debug(f"[chart_read] score row failed: {e}")
    logger.info(f"[chart_read] snapshots scored {stats}")
    return stats
