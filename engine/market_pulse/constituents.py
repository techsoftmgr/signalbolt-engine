"""
Market Pulse — S&P 500 constituent list for the breadth pillars (2, 3, 4).

Reuses engine.fundamentals.get_universe(), which already fetches the current
constituents from a maintained CSV (cached daily, with a curated fallback if the
source is unreachable) — so we don't hardcode a permanent list.

TODO (quarterly): fundamentals.get_universe()'s source is refreshed on its own
cache cadence; if Pulse coverage drifts, force a refresh there.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("signalbolt.market_pulse.constituents")


def sp500_tickers() -> list[str]:
    """Current S&P 500 tickers (uppercased), or [] if the source is unreachable."""
    try:
        from engine import fundamentals
        uni = fundamentals.get_universe() or []
        out, seen = [], set()
        for u in uni:
            t = (u.get("ticker") or "").upper().strip()
            if t and t not in seen:
                seen.add(t)
                out.append(t)
        return out
    except Exception as e:
        logger.warning(f"[market_pulse] constituent fetch failed: {e}")
        return []
