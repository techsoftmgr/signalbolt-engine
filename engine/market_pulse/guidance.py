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

# guidance_key == regime label. headline + bullets are the "what it implies" layer.
GUIDANCE: dict[str, dict] = {
    C.CONFIRMED_UPTREND: {
        "emoji": "🟢",
        "title": "Confirmed Uptrend",
        "headline": "Market is in a confirmed uptrend — the most favorable environment for new long setups.",
        "bullets": [
            "Breadth is participating and institutional selling is contained.",
            "Historically the regime where breakouts follow through and leaders make their biggest moves.",
            "Favors offense; traders are typically most willing to take new long setups here.",
        ],
    },
    C.UNDER_PRESSURE: {
        "emoji": "🟡",
        "title": "Under Pressure",
        "headline": "Uptrend under pressure — institutional selling is elevated.",
        "bullets": [
            "Distribution or weakening breadth signals more sellers stepping in.",
            "Historically, breakouts fail more often and follow-through is less reliable.",
            "A transition state — many traders turn more selective, tighten risk, and avoid aggressive new buys.",
        ],
    },
    C.CORRECTION: {
        "emoji": "🔴",
        "title": "Correction",
        "headline": "Market in correction — historically a capital-preservation environment.",
        "bullets": [
            "Most new buys fail when the broad market is in correction (~3 of 4 stocks follow the market's trend).",
            "Historically the regime where defense matters most and patience tends to be rewarded.",
            "Traders commonly wait for a new confirmed uptrend (follow-through day) before re-engaging long.",
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


def build(regime: str, vix_band: str | None, vix_rising: bool | None) -> dict:
    """Full guidance payload for an API response."""
    g = GUIDANCE.get(regime) or GUIDANCE[C.UNDER_PRESSURE]
    return {
        "regime": regime,
        "emoji": g["emoji"],
        "title": g["title"],
        "headline": g["headline"],
        "bullets": g["bullets"],
        "vix_line": vix_line(vix_band, vix_rising),
        "disclaimer": DISCLAIMER,
    }
