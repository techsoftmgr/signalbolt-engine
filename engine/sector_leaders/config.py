"""Sector Leaders — config constants (the 11 SPDR sector ETFs + RS methodology)."""
from __future__ import annotations

# The 11 S&P 500 sector SPDR ETFs (all tradable on Alpaca).
ETFS = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"]
BENCHMARK = "SPY"

# RS blend: relative return (sector − SPY) over multiple lookbacks.
L_1M, L_3M, L_6M = 21, 63, 126          # trading days
W_1M, W_3M, W_6M = 0.40, 0.35, 0.25     # blend weights

SMA_POSTURE       = 50                   # above/below 50-day MA flag
RANK_MOM_LOOKBACK = 5                    # rank change vs ~5 trading days ago

# Offense / defense / cyclical classification → "tape character".
OFFENSE  = {"XLK", "XLY", "XLI", "XLF", "XLC"}
DEFENSE  = {"XLP", "XLU", "XLV", "XLRE"}
CYCLICAL = {"XLE", "XLB"}

OFFENSE_LED = "OFFENSE_LED"
DEFENSE_LED = "DEFENSE_LED"
ROTATING    = "ROTATING"

DISCLAIMER = "Educational only — not financial advice."

GUIDANCE = {
    OFFENSE_LED: "Leadership is risk-on — offensive sectors (tech, discretionary, industrials) are leading. Historically a healthy backdrop for growth.",
    DEFENSE_LED: "Leadership is defensive — staples, utilities, and healthcare are leading. Historically a more cautious backdrop.",
    ROTATING:    "Leadership is rotating — new sectors are gaining rank while prior leaders hold. Often a sign of a healthy, broadening tape.",
}

SECTOR_NAMES = {
    "XLC": "Communication Services", "XLY": "Consumer Discretionary", "XLP": "Consumer Staples",
    "XLE": "Energy", "XLF": "Financials", "XLV": "Health Care", "XLI": "Industrials",
    "XLB": "Materials", "XLRE": "Real Estate", "XLK": "Technology", "XLU": "Utilities",
}


def tilt_of(etf: str) -> str:
    if etf in OFFENSE:
        return "OFFENSE"
    if etf in DEFENSE:
        return "DEFENSE"
    return "CYCLICAL"
