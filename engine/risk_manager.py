"""
Risk Manager
============
Pre-fire gate that grades every signal and adds metadata before it fires.

  - Grades signal by confidence tier (A+ / A / B+ / B / C)
  - Warns when 3+ consecutive losses detected (possible regime mismatch)
  - Tags sector so the UI can surface "2nd Tech signal active" as a label
  - Hard caps (max 5 concurrent, max 2/sector) removed — the 9-layer scorer
    is the quality gate. A valid A+ setup should fire even if others are active.

Confidence tiers (composite score → position size guidance):
  A+  ≥90 → 1.00x (full)
  A   ≥80 → 0.75x
  B+  ≥70 → 0.50x
  B   ≥60 → 0.25x
  C   <60 → blocked

Used by: runner.py (pre-fire gate)
"""

import logging
import time
from typing import Optional
from supabase import Client

logger = logging.getLogger("signalbolt.risk")

# ── Consecutive-losses cache ──────────────────────────────────
# This query looks at the last 5 closed signals globally — it's the same
# result for every ticker in the same scan. Cache for 60s so 27 tickers
# share one DB query instead of 27.
_loss_cache: tuple[int, float] = (0, 0.0)  # (count, fetched_at)
_LOSS_CACHE_TTL = 60  # seconds

MIN_CONFIDENCE_FIRE    = 60     # anything below = blocked (tier C)

# Confidence tier thresholds
TIERS = [
    ("A+", 90, 1.00),
    ("A",  80, 0.75),
    ("B+", 70, 0.50),
    ("B",  60, 0.25),
    ("C",   0, 0.00),  # blocked
]

SECTOR_MAP = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "GOOGL": "Technology", "GOOG": "Technology", "META": "Technology",
    "AMD": "Technology", "INTC": "Technology", "QCOM": "Technology",
    "AVGO": "Technology", "ORCL": "Technology", "ADBE": "Technology",
    "CRM": "Technology", "NOW": "Technology", "SNOW": "Technology",
    "PLTR": "Technology", "RBLX": "Technology", "NET": "Technology",
    "PANW": "Technology", "CRWD": "Technology", "ZS": "Technology",
    # Automotive / EV
    "TSLA": "Automotive", "RIVN": "Automotive", "LCID": "Automotive",
    "F": "Automotive", "GM": "Automotive",
    # Consumer / Retail / Travel
    "AMZN": "Consumer", "UBER": "Consumer", "ABNB": "Consumer",
    "LYFT": "Consumer", "NFLX": "Consumer", "DIS": "Consumer",
    "SBUX": "Consumer", "MCD": "Consumer", "TGT": "Consumer",
    "WMT": "Consumer", "HD": "Consumer", "LOW": "Consumer",
    # ETFs
    "SPY": "ETF", "QQQ": "ETF", "IWM": "ETF", "DIA": "ETF",
    "XLK": "ETF", "XLF": "ETF", "XLE": "ETF", "XLU": "ETF",
    "XLP": "ETF", "XLV": "ETF", "XLI": "ETF", "XLB": "ETF",
    "GLD": "ETF", "SLV": "ETF", "TLT": "ETF", "HYG": "ETF",
    "VXX": "ETF", "SQQQ": "ETF", "TQQQ": "ETF",
    # Financials
    "JPM": "Financials", "GS": "Financials", "BAC": "Financials",
    "MS": "Financials", "C": "Financials", "WFC": "Financials",
    "BLK": "Financials", "V": "Financials", "MA": "Financials",
    "PYPL": "Fintech", "SQ": "Fintech", "HOOD": "Fintech",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "OXY": "Energy", "SLB": "Energy",
    # Crypto-linked
    "COIN": "Crypto", "MSTR": "Crypto", "MARA": "Crypto",
    "RIOT": "Crypto", "CLSK": "Crypto", "HUT": "Crypto",
    "BTBT": "Crypto", "CIFR": "Crypto",
    # Healthcare / Biotech
    "MRNA": "Healthcare", "BNTX": "Healthcare", "PFE": "Healthcare",
    "JNJ": "Healthcare", "ABBV": "Healthcare", "LLY": "Healthcare",
    "AMGN": "Healthcare", "GILD": "Healthcare", "BIIB": "Healthcare",
}


def get_sector(ticker: str) -> str:
    return SECTOR_MAP.get(ticker, "Other")


def get_confidence_tier(score: int) -> tuple[str, float]:
    """Return (tier_label, position_multiplier) for a composite score."""
    for label, threshold, multiplier in TIERS:
        if score >= threshold:
            return label, multiplier
    return "C", 0.0


def check(sb: Client, ticker: str, score: int) -> dict:
    """
    Grade and annotate a signal before it fires. No hard caps on concurrent
    signals or sector concentration — the 9-layer scorer is the quality gate.
    Sector info is returned as metadata so the UI can show a warning label
    ("2nd Tech signal active") without silently blocking valid setups.

    Returns:
        {
          "allowed":            bool,
          "block_reason":       str,
          "confidence_tier":    str,   # A+ / A / B+ / B / C
          "position_mult":      float, # position size guidance
          "open_count":         int,
          "sector":             str,
          "sector_count":       int,   # active signals in same sector (for UI warning)
          "consecutive_losses": int,
          "regime_mismatch":    bool,
        }
    """
    tier, pos_mult = get_confidence_tier(score)

    # Score too low → blocked (tier C, <60)
    if score < MIN_CONFIDENCE_FIRE:
        return {
            "allowed":            False,
            "block_reason":       f"Score {score} below minimum {MIN_CONFIDENCE_FIRE} (tier C)",
            "confidence_tier":    tier,
            "position_mult":      0.0,
            "open_count":         0,
            "sector":             get_sector(ticker),
            "sector_count":       0,
            "consecutive_losses": 0,
            "regime_mismatch":    False,
        }

    try:
        # Fetch open signals for sector metadata + consecutive loss check
        active = (
            sb.table("signals")
            .select("ticker, result")
            .eq("status", "active")
            .execute()
            .data or []
        )

        open_count    = len(active)
        ticker_sector = get_sector(ticker)
        sector_count  = sum(1 for s in active if get_sector(s["ticker"]) == ticker_sector)

        # Consecutive losses — cached 60s so all tickers in a scan share one DB query
        global _loss_cache
        cached_losses, loss_ts = _loss_cache
        if (time.monotonic() - loss_ts) > _LOSS_CACHE_TTL:
            recent_closed = (
                sb.table("signals")
                .select("result")
                .eq("status", "closed")
                .order("closed_at", desc=True)
                .limit(5)
                .execute()
                .data or []
            )
            consecutive_losses = 0
            for row in recent_closed:
                if row.get("result") == "loss":
                    consecutive_losses += 1
                else:
                    break
            _loss_cache = (consecutive_losses, time.monotonic())
        else:
            consecutive_losses = cached_losses

        regime_mismatch = consecutive_losses >= 3
        if regime_mismatch:
            logger.warning(
                f"[risk] {consecutive_losses} consecutive losses — possible regime mismatch"
            )

        if sector_count > 0:
            logger.info(
                f"[risk] {ticker} tier={tier} pos={pos_mult:.0%} "
                f"open={open_count} sector={ticker_sector} ({sector_count} active in sector)"
            )
        else:
            logger.info(f"[risk] {ticker} tier={tier} pos={pos_mult:.0%} open={open_count}")

        return {
            "allowed":            True,
            "block_reason":       "",
            "confidence_tier":    tier,
            "position_mult":      pos_mult,
            "open_count":         open_count,
            "sector":             ticker_sector,
            "sector_count":       sector_count,
            "consecutive_losses": consecutive_losses,
            "regime_mismatch":    regime_mismatch,
        }

    except Exception as e:
        logger.error(f"[risk] portfolio check error: {e} — allowing signal (no cap enforcement)")
        return {
            "allowed":            True,
            "block_reason":       "",
            "confidence_tier":    tier,
            "position_mult":      pos_mult,
            "open_count":         0,
            "sector":             get_sector(ticker),
            "sector_count":       0,
            "consecutive_losses": 0,
            "regime_mismatch":    False,
        }


