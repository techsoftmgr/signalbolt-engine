"""Alpaca broker adapter — read the user's positions via their Alpaca keys.

INVASIVE (uses the user's brokerage credentials) → only reachable when the
PHASE2_PORTFOLIO_DOCTOR flag is on. Keys are passed per-request and never stored
here. Returns (holdings, cash). Never raises.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("signalbolt.phase2.brokers.alpaca")


def fetch(api_key: str, secret: str, paper: bool = True):
    """Pull positions + cash from the user's Alpaca account."""
    holdings, cash = [], 0.0
    if not api_key or not secret:
        return holdings, cash
    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(api_key, secret, paper=bool(paper))
        for p in client.get_all_positions():
            try:
                holdings.append({
                    "ticker": p.symbol.upper(),
                    "qty": abs(float(p.qty)),
                    "avg_price": float(p.avg_entry_price),
                    "current_price": float(p.current_price) if p.current_price else float(p.avg_entry_price),
                })
            except Exception:
                continue
        try:
            acct = client.get_account()
            cash = float(acct.cash or 0)
        except Exception:
            pass
    except Exception as e:
        logger.debug(f"[alpaca_adapter] fetch failed: {e}")
    return holdings, cash
