"""
Decision Support — an ADDITIVE "expert read enhancement" layer.

It does NOT compute any new market analysis. It READS the already-computed
`chart_read.analyze()` output (trend, TA vs Quant verdict + agreement, key
support/resistance, Fibonacci golden pocket / invalidation / target, volume
regime, gaps, channel, scenario triggers) and DERIVES decision-support fields
that answer: "what should I do right now — chase, wait, watch, avoid?",
"what are the probabilities?", "where is the best entry?", "what is the risk?",
"why can this fail?".

Design rules:
  • PURE + DEFENSIVE — `derive(read)` never raises and tolerates missing
    sub-objects (no fib / no levels / no quant). Missing inputs degrade to
    "Not enough data", never a crash.
  • NO advice language. Heuristic/educational framing only.
  • Probabilities ALWAYS total 100.
  • Backward-compatible: this is attached as an OPTIONAL `decision_support`
    object on the chart-read response; existing fields are untouched.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger("signalbolt.decision_support")

_NEAR = 0.025      # within 2.5% of a level = "near"
_FAR = 0.05        # >5% away = "far"
_HIST_MIN = 8      # min scored samples before historical stats are shown
_DISCLAIMER = "Educational decision-support, not financial advice."


# ── small safe helpers ────────────────────────────────────────────────────────
def _num(x):
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _factors(read: dict) -> dict:
    """Reduce the chart-read into the boolean/numeric factors every card uses.
    Everything is best-effort: a missing sub-object just leaves its factors False/None."""
    px = _num(read.get("price"))
    ta = (read.get("taBias") or read.get("bias") or "neutral")
    quant = read.get("quantBias")            # may be None
    agreement = read.get("agreement")        # agree/disagree/partial/n-a
    short_term = read.get("shortTerm")       # confirming/diverging/neutral
    trend = (read.get("trend") or {})
    trend_d = trend.get("d1") or "sideways"
    mtf = (read.get("mtf") or {})
    levels = (read.get("levels") or {})
    res = _num(levels.get("resistance"))     # nearest level ABOVE price
    sup = _num(levels.get("support"))        # nearest level BELOW price
    tl = (read.get("trendlines") or {})
    fib = read.get("fib") or None
    gp = (fib or {}).get("goldenPocket") or {}
    gp_lo, gp_hi = _num(gp.get("low")), _num(gp.get("high"))
    inval = _num((fib or {}).get("invalidation"))
    fib_tgt = _num((fib or {}).get("target"))
    fib_dir = (fib or {}).get("direction")
    vol = read.get("volumeRegime") or "normal"
    gap = read.get("gap") or None
    ch = read.get("channel") or None
    scen = read.get("scenarios") or {}
    bull_trig = _num((scen.get("bull") or {}).get("trigger"))
    bull_tgt = _num((scen.get("bull") or {}).get("target"))
    bear_trig = _num((scen.get("bear") or {}).get("trigger"))
    bear_tgt = _num((scen.get("bear") or {}).get("target"))
    idea = read.get("idea") or {}

    near_res = bool(px and res and 0 <= (res - px) / px <= _NEAR)
    near_sup = bool(px and sup and 0 <= (px - sup) / px <= _NEAR)
    far_from_sup = bool(px and sup and (px - sup) / px > _FAR)
    in_gp = bool(px and gp_lo and gp_hi and gp_lo <= px <= gp_hi)
    lost_support = bool(
        tl.get("vsSupport") == "below"
        or (px and fib_dir == "up" and inval and px < inval)
    )
    above_support = bool(sup is not None and not lost_support)
    breakout_confirmed = bool(tl.get("vsResistance") == "above" and trend_d == "up")
    extended = bool(ch and ch.get("position") == "top")
    gap_down_unfilled = bool(gap and gap.get("dir") == "down" and not gap.get("filled"))
    near_target = bool(px and bull_tgt and px >= bull_tgt * 0.97)

    return {
        "px": px, "ta": ta, "quant": quant, "agreement": agreement, "short_term": short_term,
        "trend_d": trend_d, "mtf_state": mtf.get("state"), "mtf_dir": mtf.get("dir"),
        "res": res, "sup": sup, "gp_lo": gp_lo, "gp_hi": gp_hi, "inval": inval,
        "fib_tgt": fib_tgt, "fib_dir": fib_dir, "vol": vol, "gap": gap, "channel": ch,
        "bull_trig": bull_trig, "bull_tgt": bull_tgt, "bear_trig": bear_trig, "bear_tgt": bear_tgt,
        "idea": idea, "confidence": _num(read.get("confidence")) or 50,
        "ta_bull": ta == "bullish", "ta_bear": ta == "bearish",
        "quant_bull": quant == "bullish", "quant_bear": quant == "bearish",
        "trend_up": trend_d == "up", "trend_down": trend_d == "down",
        "agree": agreement == "agree", "disagree": agreement == "disagree",
        "divergence": short_term == "diverging",
        "vol_strong": vol == "accumulation", "vol_weak": vol in ("dry-up", "distribution"),
        "near_res": near_res, "near_sup": near_sup, "far_from_sup": far_from_sup,
        "in_gp": in_gp, "lost_support": lost_support, "above_support": above_support,
        "breakout_confirmed": breakout_confirmed, "extended": extended,
        "gap_down_unfilled": gap_down_unfilled, "near_target": near_target,
    }


# ── Feature 1: Decision Summary (action + reason + confidence + quality + RR) ──
def _action(f: dict) -> tuple[str, str]:
    """Action label + one-sentence STATE description (what is true now) — not a
    forecast. Priority-ordered. No advice/prediction language."""
    if (f["trend_down"] or f["ta_bear"]) and f["lost_support"]:
        return "AVOID", "State: down-trend and a key support level is lost — conditions are not constructive for a long."
    if f["breakout_confirmed"]:
        return "BREAKOUT CONFIRMATION", "State: price has closed back above resistance in an up-trend — the breakout condition is met."
    if f["near_target"] and (f["trend_up"] or f["ta_bull"]):
        return "TAKE PROFIT / REDUCE RISK", "State: price is already at the measured target — most of the move is behind, risk rises from here."
    if f["quant_bull"] and f["ta_bull"] and f["in_gp"] and not f["trend_down"]:
        return "BUY ZONE", "State: chart and model agree up, and price is inside the pullback (golden-pocket) area."
    if f["quant_bull"] and f["ta_bull"] and f["near_res"]:
        return "WAIT", "State: up-trend, but price is at resistance — no trigger met. A daily close above resistance, or a pullback, would set one up."
    if f["disagree"]:
        return "WATCH", "State: chart and model disagree — no edge until one side confirms; watch which resolves."
    if f["ta_bull"] and f["quant_bull"]:
        return "WATCH", "State: chart and model are both up, but no clean trigger yet — watch for a close above resistance or a pullback."
    if (f["ta_bear"] and f["quant_bear"]) or (f["trend_down"] and f["ta_bear"]):
        return "AVOID", "State: chart and model are both down — conditions don't support a long here."
    return "WAIT", "State: no clear edge — conditions are mixed/neutral."


def _confidence_label(f: dict) -> str:
    c = f["confidence"]
    if f["disagree"]:
        return "Low"
    lab = "High" if c >= 72 else "Medium" if c >= 58 else "Low"
    if f["divergence"] and lab == "High":
        lab = "Medium"
    return lab


def _rr(f: dict):
    """Risk/reward estimate — prefer the chart-read idea's rr, else derive from geometry."""
    rr = _num((f.get("idea") or {}).get("rr"))
    if rr is not None:
        return rr
    px, sup, res = f["px"], f["sup"], f["res"]
    if f["ta_bull"] and px and sup and px > sup:
        tgt = f["fib_tgt"] if (f["fib_tgt"] and f["fib_tgt"] > px) else res
        if tgt and tgt > px:
            return round((tgt - px) / (px - sup), 2)
    if f["ta_bear"] and px and res and res > px:
        tgt = f["fib_tgt"] if (f["fib_tgt"] and f["fib_tgt"] < px) else sup
        if tgt and tgt < px:
            return round((px - tgt) / (res - px), 2)
    return None


def _rr_quality(rr) -> str:
    if rr is None:
        return "Fair"
    return "Excellent" if rr >= 3 else "Good" if rr >= 2 else "Fair" if rr >= 1 else "Poor"


def _trade_quality(f: dict, rr) -> str:
    q = 50
    q += 15 if f["agree"] else -20 if f["disagree"] else 0
    if (f["trend_up"] and f["ta_bull"]) or (f["trend_down"] and f["ta_bear"]):
        q += 10
    if f["mtf_state"] == "aligned":
        q += 8
    if rr is not None:
        q += 8 if rr >= 2 else 4 if rr >= 1.5 else 0 if rr >= 1 else -8
    q -= 10 if f["divergence"] else 0
    q += 6 if f["vol_strong"] else -6 if f["vol_weak"] else 0
    q -= 8 if f["near_res"] else 0
    q += 6 if f["in_gp"] else 0
    q = _clamp(q, 0, 100)
    return "A+" if q >= 85 else "A" if q >= 72 else "B" if q >= 58 else "C" if q >= 45 else "D"


# ── Feature 2: Probability View (always totals 100) ───────────────────────────
def _probabilities(f: dict) -> dict:
    bull = 50
    bull += 12 if f["quant_bull"] else 0
    bull += 12 if f["ta_bull"] else 0
    bull += 8 if f["trend_up"] else 0
    bull += 6 if f["above_support"] else 0
    bull += 5 if f["vol_strong"] else 0
    bull += 5 if (f["in_gp"] and not f["trend_down"]) else 0
    bull += 5 if (f["mtf_state"] == "aligned" and f["mtf_dir"] == "up") else 0
    bull -= 10 if f["near_res"] else 0
    bull -= 8 if f["divergence"] else 0
    bull -= 8 if f["lost_support"] else 0
    bull -= 6 if f["vol_weak"] else 0
    bull -= 8 if (f["gap_down_unfilled"] or f["trend_down"]) else 0
    bull -= 10 if f["disagree"] else 0
    bull = _clamp(bull, 5, 90)

    bear = 50
    bear += 12 if f["quant_bear"] else 0
    bear += 12 if f["ta_bear"] else 0
    bear += 8 if f["trend_down"] else 0
    bear += 10 if f["disagree"] else 0
    bear += 8 if f["divergence"] else 0
    bear += 8 if f["lost_support"] else 0
    bear += 10 if f["near_res"] else 0
    bear += 6 if f["vol_weak"] else 0
    bear += 8 if f["gap_down_unfilled"] else 0
    bear -= 12 if f["quant_bull"] else 0
    bear -= 12 if f["ta_bull"] else 0
    bear -= 8 if f["trend_up"] else 0
    bear = _clamp(bear, 5, 90)

    neutral = 20
    neutral += 15 if f["disagree"] else 0
    neutral += 12 if (f["agreement"] == "partial" or f["quant"] is None) else 0
    neutral += 10 if f["divergence"] else 0
    neutral += 10 if f["trend_d"] == "sideways" else 0
    neutral += 8 if (f["near_res"] and f["near_sup"]) else 0
    neutral += 8 if f["mtf_state"] == "mixed" else 0
    neutral = _clamp(neutral, 5, 60)

    total = bull + bear + neutral
    b = round(bull / total * 100)
    n = round(neutral / total * 100)
    r = 100 - b - n                       # guarantees exact total of 100
    if r < 0:                             # rounding spill — repair against the largest
        b += r
        r = 0
    spread = abs(b - r)
    if f["disagree"]:
        conf = "Low" if spread < 25 else "Medium"
    else:
        conf = ("High" if spread >= 45 and f["agree"]
                else "Medium-High" if spread >= 30
                else "Medium" if spread >= 15 else "Low")
    return {"bullish": b, "neutral": n, "bearish": r, "confidence_label": conf}


def _drivers(f: dict) -> list:
    d = []
    def add(label, supportive, cond):
        if cond:
            d.append({"label": label, "supportive": supportive})
    add("Daily trend bullish", True, f["trend_up"])
    add("Quant read bullish", True, f["quant_bull"])
    add("Technical read bullish", True, f["ta_bull"])
    add("TA & Quant agree", True, f["agree"])
    add("Holding above support", True, f["above_support"] and not f["trend_down"])
    add("Inside Fibonacci golden pocket", True, f["in_gp"])
    add("Accumulation volume", True, f["vol_strong"])
    add("Near resistance", False, f["near_res"])
    add("Short-term divergence", False, f["divergence"])
    add("Lost key support", False, f["lost_support"])
    add("Weak / declining volume", False, f["vol_weak"])
    add("Unfilled gap below", False, f["gap_down_unfilled"])
    add("TA & Quant disagree", False, f["disagree"])
    add("Daily trend bearish", False, f["trend_down"])
    return d[:8]


def _scorecard(f: dict) -> dict:
    """Conditions Scorecard — a factual tally of what's TRUE right now (no
    forecast %). The user weighs it; we don't predict a direction."""
    items = _drivers(f)
    bull = sum(1 for d in items if d["supportive"])
    bear = len(items) - bull
    lean = "constructive" if bull > bear else "cautious" if bear > bull else "balanced"
    return {"items": items, "bullish": bull, "bearish": bear, "total": len(items),
            "summary": f"{bull} of {len(items)} conditions constructive · {lean}"}


# ── Feature 3: Scenario Tree ──────────────────────────────────────────────────
def _scenario_tree(f: dict, probs: dict) -> dict:
    px = f["px"]
    bull_trig = f["bull_trig"] or f["res"]
    bull_tgt = f["bull_tgt"] or (f["fib_tgt"] if (f["fib_tgt"] and f["fib_tgt"] > (px or 0)) else None) or f["res"]
    bear_trig = f["bear_trig"] or f["sup"]
    bear_tgt = f["bear_tgt"] or f["inval"] or f["sup"]
    # Directional sanity (mirror of chart_read._scenarios): an upside target must
    # sit ABOVE the reclaim level and a downside target BELOW the lose level —
    # the fallbacks above (res == bull_trig, inval/sup at-or-above bear_trig) can
    # violate that, so drop a contradictory target rather than show it.
    if bull_tgt is not None and bull_trig is not None and bull_tgt <= bull_trig:
        bull_tgt = None
    if bear_tgt is not None and bear_trig is not None and bear_tgt >= bear_trig:
        bear_tgt = None
    rng = None
    if bear_trig and bull_trig:
        rng = f"Holds between ${bear_trig} and ${bull_trig}"
    return {
        "bullish": {
            "trigger": (f"Reclaim ${bull_trig}" if bull_trig else "No clear trigger"),
            "meaning": "Buyers regain control",
            "target": bull_tgt, "probability": probs["bullish"],
        },
        "neutral": {
            "trigger": rng or "Choppy / range-bound",
            "meaning": "No clean edge yet",
            "action": "Wait for confirmation", "probability": probs["neutral"],
        },
        "bearish": {
            "trigger": (f"Lose ${bear_trig}" if bear_trig else "No clear trigger"),
            "meaning": "Downside resumes",
            "target": bear_tgt, "probability": probs["bearish"],
        },
    }


# ── Feature 4: Entry Quality ──────────────────────────────────────────────────
def _entry_quality(f: dict, rr) -> dict:
    px = f["px"]
    pullback = (f"${f['gp_lo']}–${f['gp_hi']}" if (f["gp_lo"] and f["gp_hi"]) else None)
    trig = f["bull_trig"] or f["res"]
    breakout = (f"Above ${trig}" if trig else None)
    inval = f["inval"] or f["sup"] or _num((f.get("idea") or {}).get("stop"))
    first_tgt = f["bull_tgt"] or (f["fib_tgt"] if (f["fib_tgt"] and px and f["fib_tgt"] > px) else None) or f["res"]

    if f["breakout_confirmed"]:
        state, note = "BREAKOUT_CONFIRMATION", "Breakout is confirming — risk is defined below the breakout level."
    elif f["lost_support"]:
        state, note = "AVOID_UNTIL_RECLAIMED", "Support is lost — avoid until price reclaims the broken level."
    elif f["in_gp"] and not f["trend_down"]:
        state, note = "DIP_BUY_ZONE", "Price is in the Fibonacci pullback zone — a lower-risk area to watch."
    elif f["near_res"] and f["far_from_sup"]:
        state, note = "DO_NOT_CHASE", "Do not chase — wait for a pullback or breakout confirmation."
    else:
        state, note = "WAIT", "No clean entry trigger yet — wait for confirmation."

    return {
        "current": px, "ideal_pullback_zone": pullback, "breakout_trigger": breakout,
        "invalidation": (f"Below ${inval}" if inval else None),
        "first_target": first_tgt, "risk_reward": rr, "state": state, "note": note,
    }


# ── Feature 5: Reasons for / against ──────────────────────────────────────────
def _reasons(f: dict) -> tuple[list, list]:
    for_ = []
    if f["quant_bull"]: for_.append("Quant read is bullish")
    if f["ta_bull"]: for_.append("Technical read is bullish")
    if f["trend_up"]: for_.append("Daily trend is up")
    if f["above_support"] and not f["trend_down"]: for_.append("Price holding above support")
    if f["in_gp"]: for_.append(f"In Fibonacci pullback zone (${f['gp_lo']}–${f['gp_hi']})")
    if f["vol_strong"]: for_.append("Volume confirms (accumulation)")
    if f["mtf_state"] == "aligned" and f["mtf_dir"] == "up": for_.append("Multi-timeframe aligned up")
    if f["agree"]: for_.append("TA and Quant agree")

    against = []
    if f["near_res"]: against.append("Price near resistance")
    if f["divergence"]: against.append("Short-term momentum is diverging")
    if f["gap_down_unfilled"]: against.append("Unfilled gap below")
    if f["vol_weak"]: against.append("Weak / declining volume")
    if f["lost_support"]: against.append("Price has lost key support")
    if f["extended"]: against.append("Extended at the top of its channel")
    if f["disagree"]: against.append("TA and Quant disagree")
    if f["trend_down"]: against.append("Daily trend is down")
    if f["inval"]: against.append(f"A close below ${f['inval']} invalidates the setup")
    return for_, against


# ── Feature 6: Risk Meter ─────────────────────────────────────────────────────
def _risk_meter(f: dict) -> dict:
    r = 50
    r += 10 if f["near_res"] else 0
    r += 15 if f["disagree"] else 0
    r += 10 if f["divergence"] else 0
    r += 15 if f["lost_support"] else 0
    r += 8 if f["vol"] == "climax" else 0
    r += 8 if f["vol_weak"] else 0
    r += 10 if f["gap_down_unfilled"] else 0
    r += 10 if f["extended"] else 0
    r -= 10 if f["near_sup"] else 0
    r -= 8 if (f["in_gp"] and not f["trend_down"]) else 0
    r -= 12 if f["agree"] else 0
    r -= 8 if (f["trend_up"] and f["ta_bull"]) else 0
    r -= 8 if f["vol_strong"] else 0
    r = _clamp(r, 0, 100)
    level = "High" if r >= 62 else "Low" if r <= 38 else "Medium"
    expl = {"High": "Multiple risk factors are active — size and timing matter most here.",
            "Medium": "A balanced mix of supportive and cautionary factors.",
            "Low": "Conditions are relatively constructive, but risk is never zero."}[level]

    factors = []
    def add(label, increases, cond):
        if cond:
            factors.append({"label": label, "increases_risk": increases})
    add("Price near resistance", True, f["near_res"])
    add("TA / Quant disagreement", True, f["disagree"])
    add("Short-term divergence", True, f["divergence"])
    add("Lost key support", True, f["lost_support"])
    add("Unfilled gap below", True, f["gap_down_unfilled"])
    add("Weak / declining volume", True, f["vol_weak"])
    add("Extended at channel top", True, f["extended"])
    add("Daily trend still up", False, f["trend_up"])
    add("In Fibonacci pullback zone", False, f["in_gp"])
    add("Near support", False, f["near_sup"])
    add("TA and Quant agree", False, f["agree"])
    add("Accumulation volume", False, f["vol_strong"])
    return {"level": level, "score": r, "explanation": expl, "factors": factors[:6]}


# ── Feature 8: Plain-English Read (≤5 sentences, ends with an action phrase) ──
_ACTION_PHRASE = {
    "AVOID": "Avoid until reclaimed.",
    "BREAKOUT CONFIRMATION": "Breakout confirmation needed.",
    "TAKE PROFIT / REDUCE RISK": "Watch only.",
    "BUY ZONE": "Pullback entry preferred.",
    "WAIT": "Wait for confirmation.",
    "WATCH": "Watch only.",
}


def _plain_english(read: dict, f: dict, action: str) -> str:
    tk = read.get("ticker", "This name")
    s = []
    if f["trend_up"]:
        s.append(f"{tk} is still in a larger bullish trend")
    elif f["trend_down"]:
        s.append(f"{tk} is in a downtrend")
    else:
        s.append(f"{tk} is moving sideways with no clear trend")
    if f["divergence"]:
        s[0] += ", but the short-term trend is weakening."
    else:
        s[0] += "."
    if action == "BUY ZONE" or (f["in_gp"] and not f["trend_down"]):
        s.append("Price is sitting in a Fibonacci pullback zone, which is a lower-risk area to watch.")
    elif f["lost_support"]:
        s.append("Price has lost support, so the setup is not clean until that level is reclaimed.")
    elif f["near_res"]:
        s.append("Buying immediately is not ideal because resistance is nearby.")
    trig = f["bull_trig"] or f["res"]
    if f["gp_lo"] and f["gp_hi"] and trig:
        s.append(f"The cleaner setup is a pullback into ${f['gp_lo']}–${f['gp_hi']} or a confirmed reclaim above ${trig}.")
    elif trig:
        s.append(f"A confirmed reclaim above ${trig} would strengthen the bullish case.")
    s.append(_ACTION_PHRASE.get(action, "Watch only."))
    return " ".join(s[:5])


# ── Feature 9: Signal Freshness ───────────────────────────────────────────────
def _freshness(now: datetime, calculated_at: datetime | None) -> dict:
    ca = calculated_at or now
    age = max(0, int((now - ca).total_seconds()))
    status = "Fresh" if age < 300 else "Aging" if age < 900 else "Stale"
    return {
        "calculated_at": ca.isoformat(),
        "age_seconds": age,
        "status": status,
        "data_freshness": "Real-time — computed on request",
        "next_refresh": None,
    }


# ── Feature 10: Tags ──────────────────────────────────────────────────────────
def _tags(f: dict, action: str, conf_label: str) -> list:
    t = []
    if f["agree"] and f["confidence"] >= 72:
        t.append("High Conviction")
    if action == "AVOID":
        t.append("Avoid")
    if action in ("WAIT", "WATCH") or f["disagree"] or f["divergence"]:
        t.append("Needs Confirmation")
    if f["near_res"] and f["trend_up"] and not f["divergence"]:
        t.append("Ready to Breakout")
    if f["near_res"]:
        t.append("Near Resistance")
    if f["in_gp"] and not f["trend_down"]:
        t.append("Dip-Buy Zone")
    if f["above_support"] and not f["trend_down"]:
        t.append("Holding Support")
    if f["divergence"]:
        t.append("Momentum Weakening")
    if f["trend_up"]:
        t.append("Daily Trend Bullish")
    elif f["trend_down"]:
        t.append("Daily Trend Bearish")
    # de-dup, preserve order, cap at 6
    seen, out = set(), []
    for x in t:
        if x not in seen:
            seen.add(x); out.append(x)
    return out[:6]


# ── Feature 7: Historical similar setups (REAL data from chart_read_log) ──────
def _historical_unavailable() -> dict:
    return {
        "available": False,
        "note": "Not enough historical matches yet. This will improve as more completed setups are tracked.",
    }


def historical_similar_setups(sb, ticker: str, ta_bias: str | None) -> dict:
    """Real base-rate from chart_read_log: of past days this ticker carried the
    SAME technical read, how did the next ~5 trading days resolve? Returns the
    unavailable state when the table is missing or the sample is too small (never
    fabricates data). Best-effort; never raises."""
    if sb is None or not ticker or ta_bias not in ("bullish", "bearish"):
        return _historical_unavailable()
    try:
        rows = (sb.table("chart_read_log").select("forward_return_pct,horizon_days,created_at")
                .eq("ticker", ticker.upper()).eq("ta_bias", ta_bias)
                .not_.is_("forward_return_pct", "null")
                .order("created_at", desc=True).limit(400).execute().data) or []
    except Exception as e:
        logger.debug(f"[decision_support] historical query failed for {ticker}: {e}")
        return _historical_unavailable()

    samples = []
    for r in rows:
        ret = _num(r.get("forward_return_pct"))
        if ret is None:
            continue
        # Profit if you took the read's side: long → +ret, short → -ret.
        dir_ret = ret if ta_bias == "bullish" else -ret
        samples.append({"dir_ret": dir_ret, "date": r.get("created_at"),
                        "horizon": r.get("horizon_days")})
    if len(samples) < _HIST_MIN:
        return _historical_unavailable()

    wins = [s for s in samples if s["dir_ret"] > 1.0]
    gains = [s["dir_ret"] for s in samples if s["dir_ret"] > 0]
    losses = [s["dir_ret"] for s in samples if s["dir_ret"] < 0]
    best = max(samples, key=lambda s: s["dir_ret"])
    worst = min(samples, key=lambda s: s["dir_ret"])
    horizon = next((s["horizon"] for s in samples if s["horizon"]), 5)
    return {
        "available": True,
        "count": len(samples),
        "win_rate": round(len(wins) / len(samples) * 100),
        "avg_gain": round(sum(gains) / len(gains), 1) if gains else None,
        "avg_loss": round(sum(losses) / len(losses), 1) if losses else None,
        "avg_time_to_target_days": None,   # not tracked yet (only fixed-horizon return)
        "best_match_date": (best["date"] or "")[:10] or None,
        "worst_match_date": (worst["date"] or "")[:10] or None,
        "horizon_days": horizon,
        "basis": f"Past days this ticker's technical read was {ta_bias}, {horizon}-day forward return.",
    }


# ── Public entry point ────────────────────────────────────────────────────────
def derive(read: dict, now: datetime | None = None,
           calculated_at: datetime | None = None, historical: dict | None = None) -> dict:
    """Build the decision_support object from a chart_read.analyze() result.
    PURE + DEFENSIVE — never raises; tolerates missing sub-objects."""
    try:
        now = now or datetime.now(timezone.utc)
        f = _factors(read or {})
        if f["px"] is None:
            return {"available": False, "note": "Not enough data for a decision read.",
                    "disclaimer": _DISCLAIMER}

        action, reason = _action(f)
        conf_label = _confidence_label(f)
        rr = _rr(f)
        rr_quality = _rr_quality(rr)
        trade_quality = _trade_quality(f, rr)
        probs = _probabilities(f)
        for_, against = _reasons(f)

        return {
            "available": True,
            "action": action,
            "reason": reason,
            "confidence": conf_label,
            "trade_quality": trade_quality,
            "risk_reward_quality": rr_quality,
            "risk_reward": rr,
            "bullish_probability": probs["bullish"],
            "neutral_probability": probs["neutral"],
            "bearish_probability": probs["bearish"],
            "probability_confidence": probs["confidence_label"],
            "probability_drivers": _drivers(f),
            "scorecard": _scorecard(f),
            "scenario_tree": _scenario_tree(f, probs),
            "entry_quality": _entry_quality(f, rr),
            "reasons_for": for_,
            "reasons_against": against,
            "risk_meter": _risk_meter(f),
            "plain_english_read": _plain_english(read or {}, f, action),
            "historical_similar_setups": historical if historical is not None else _historical_unavailable(),
            "signal_freshness": _freshness(now, calculated_at),
            "tags": _tags(f, action, conf_label),
            "disclaimer": _DISCLAIMER,
        }
    except Exception as e:
        logger.debug(f"[decision_support] derive failed: {e}")
        return {"available": False, "note": "Decision read unavailable.", "disclaimer": _DISCLAIMER}
