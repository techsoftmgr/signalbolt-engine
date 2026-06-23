"""
Market Pulse — guidance copy. Phrased as what the regime HISTORICALLY IMPLIES,
never as a personal buy/sell instruction. Every verdict carries the disclaimer.
"""
from __future__ import annotations

from . import config as C

DISCLAIMER = (
    "Educational only — not financial advice. Market regime is a general context "
    "read, not a recommendation to buy or sell any security."
)

# A read of CURRENT conditions, not a price forecast. Surfaced on the card so a label like
# "Correction" isn't misread as "the market is about to fall" — it describes the tape RIGHT NOW.
NOT_FORECAST = "Describes current market conditions, not a prediction of where it goes next."

# guidance_key == regime label. Layers:
#   state    — plain "what's happening RIGHT NOW" (not a forecast)
#   posture  — the one-word stance the conditions historically favor
#   headline + bullets — the "what it historically implies" layer
#   toDo — concrete "what to do in this regime" (educational; how traders typically act)
GUIDANCE: dict[str, dict] = {
    C.CONFIRMED_UPTREND: {
        "emoji": "🟢",
        "title": "Confirmed Uptrend",
        "posture": "Offense",
        "state": "Right now the broad market is rising with broad participation and selling is contained — the wind is at your back.",
        "headline": "Market is in a confirmed uptrend — the most favorable environment for new long setups.",
        "bullets": [
            "Breadth is participating and institutional selling is contained.",
            "Historically the regime where breakouts follow through and leaders make their biggest moves.",
            "Favors offense; traders are typically most willing to take new long setups here.",
        ],
        "toDo": [
            "Offense: new long setups have their best odds here — normal position size.",
            "Lean into leaders and breakouts that follow through; let winners run.",
            "Still honor stops — even strong uptrends have sharp pullbacks.",
        ],
    },
    C.UNDER_PRESSURE: {
        "emoji": "🟡",
        "title": "Under Pressure",
        "posture": "Selective",
        "state": "Right now selling is picking up and breadth is softening — a transition phase, not yet a full correction.",
        "headline": "Uptrend under pressure — institutional selling is elevated.",
        "bullets": [
            "Distribution or weakening breadth signals more sellers stepping in.",
            "Historically, breakouts fail more often and follow-through is less reliable.",
            "A transition state — many traders turn more selective, tighten risk, and avoid aggressive new buys.",
        ],
        "toDo": [
            "Turn selective: smaller new positions, tighten stops, trim the weakest longs.",
            "Demand stronger setups — marginal breakouts fail more often here.",
            "Let it resolve — up (a follow-through day) or down (into correction) — before adding risk.",
        ],
    },
    C.CORRECTION: {
        "emoji": "🔴",
        "title": "Correction",
        "posture": "Defense",
        "state": "Right now the broad market is selling off and most stocks are trending down with it.",
        "headline": "Market in correction — historically a capital-preservation environment (a state, not a forecast).",
        "bullets": [
            "Most new buys fail when the broad market is in correction (~3 of 4 stocks follow the market's trend).",
            "Historically the regime where defense matters most and patience tends to be rewarded.",
            "Traders commonly wait for a new confirmed uptrend (follow-through day) before re-engaging long.",
        ],
        "toDo": [
            "Defense first: smaller size, fewer new longs, more cash — preserve capital.",
            "If you go long, only relative-strength leaders (holding up / higher lows while the market falls).",
            "Short strength, not weakness — fade bounces into resistance; don't short a multi-day flush into oversold.",
            "Don't dip-buy every red day — catching the falling knife is how most longs fail here.",
            "Wait for a follow-through day (a strong up day on rising volume) before re-engaging long.",
        ],
    },
}


def vix_line(band: str | None, rising: bool | None) -> str:
    """Dynamic fear-gauge line keyed off the VIX band + trend. Null-safe."""
    if not band:
        return "Volatility data unavailable."
    if band == "CALM":
        return "Volatility is low — markets are pricing little fear."
    if band == "NORMAL":
        return "Volatility is in a normal range."
    if band == "ELEVATED":
        return f"Volatility is elevated and {'rising' if rising else 'easing'} — fear is {'building' if rising else 'receding'}."
    if band == "HIGH":
        return "Volatility is high — markets are pricing significant stress."
    return "Volatility data unavailable."


def summary_line(row: dict) -> str:
    """One plain-English sentence synthesizing the pillars into a quick read, e.g.
    'Healthy breadth, calm volatility, but 5 distribution days = institutions
    selling → be selective and tighten risk.' Derived from the row, so it updates
    daily. Never raises."""
    try:
        dd = max(int(row.get("dd_count_spy") or 0), int(row.get("dd_count_qqq") or 0))
        stall = max(int(row.get("stall_count_spy") or 0), int(row.get("stall_count_qqq") or 0))
        p50 = float(row.get("pct_above_50") or 0)
        nh_nl = int(row.get("net_nhnl") or 0)
        vb = row.get("vix_band")
        vr = bool(row.get("vix_rising"))
        div = bool(row.get("ad_divergence"))
        thrust = bool(row.get("breadth_thrust"))
        regime = row.get("regime")

        parts = []
        # Breadth participation
        if p50 >= 60:
            parts.append("healthy breadth")
        elif p50 >= 50:
            parts.append("breadth holding up")
        elif p50 >= 40:
            parts.append("weakening breadth")
        else:
            parts.append("poor breadth")
        # Volatility
        vol = {"CALM": "calm volatility", "NORMAL": "normal volatility",
               "ELEVATED": "elevated volatility", "HIGH": "high volatility"}.get(vb)
        if vol:
            if vb in ("ELEVATED", "HIGH") and vr:
                vol += " and rising"
            parts.append(vol)
        # Selling pressure (the distribution story)
        if dd >= 5:
            parts.append(f"but {dd} distribution days = institutions selling")
        elif dd >= 3:
            parts.append(f"with {dd} distribution days (some selling)")
        if stall >= 2:
            parts.append(f"plus {stall} stalling days")
        if nh_nl < 0:
            parts.append("more new lows than highs")
        if div:
            parts.append("breadth diverging from price")
        if thrust:
            parts.append("a breadth thrust just fired")
        if bool(row.get("breadth_breakdown")):
            parts.append("a breadth breakdown just fired")

        stance = {
            "CONFIRMED_UPTREND": "favors offense — breakouts tend to follow through",
            "UNDER_PRESSURE": "be selective and tighten risk",
            "CORRECTION": "defense first — capital preservation",
        }.get(regime, "")
        sentence = ", ".join(parts)
        sentence = sentence[:1].upper() + sentence[1:]
        return f"{sentence} → {stance}." if stance else f"{sentence}."
    except Exception:
        return ""


def build(regime: str, vix_band: str | None, vix_rising: bool | None) -> dict:
    """Full guidance payload for an API response."""
    g = GUIDANCE.get(regime) or GUIDANCE[C.UNDER_PRESSURE]
    return {
        "regime": regime,
        "emoji": g["emoji"],
        "title": g["title"],
        "posture": g.get("posture"),
        "state": g.get("state"),
        "headline": g["headline"],
        "bullets": g["bullets"],
        "toDo": g.get("toDo", []),
        "notForecast": NOT_FORECAST,
        "vix_line": vix_line(vix_band, vix_rising),
        "disclaimer": DISCLAIMER,
    }
