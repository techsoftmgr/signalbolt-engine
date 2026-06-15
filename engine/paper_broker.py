"""
Alpaca PAPER trading broker wrapper — admin-only paper execution of SignalBolt
signals.

INTEGRITY: this uses a SEPARATE paper account + keys (ALPACA_PAPER_API_KEY /
ALPACA_PAPER_SECRET_KEY) and the paper endpoint (TradingClient(..., paper=True)).
It is physically incapable of touching real money or the live data keys. If the
paper keys aren't set the broker is simply DISABLED (is_configured() == False) and
every call returns None/[]/False — never raises into callers.

Entry is a LIMIT order at the signal's entry price with a BRACKET (stop-loss +
take-profit from the signal), so a paper trade faithfully mirrors the signal's
levels and never chases if price has already run past entry.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger("signalbolt.paper_broker")

_client = None
_init_done = False


def is_configured() -> bool:
    return bool(os.environ.get("ALPACA_PAPER_API_KEY") and os.environ.get("ALPACA_PAPER_SECRET_KEY"))


def _get():
    global _client, _init_done
    if _init_done:
        return _client
    _init_done = True
    try:
        if not is_configured():
            logger.warning("[paper] ALPACA_PAPER_API_KEY/SECRET not set — paper broker disabled")
            return None
        from alpaca.trading.client import TradingClient
        _client = TradingClient(os.environ["ALPACA_PAPER_API_KEY"],
                                os.environ["ALPACA_PAPER_SECRET_KEY"], paper=True)
        logger.info("[paper] Alpaca PAPER trading client initialised (paper=True)")
    except Exception as e:
        logger.error(f"[paper] client init failed: {e}")
        _client = None
    return _client


def account() -> dict | None:
    c = _get()
    if not c:
        return None
    try:
        a = c.get_account()
        return {
            "equity": float(a.equity), "last_equity": float(a.last_equity),
            "cash": float(a.cash), "buying_power": float(a.buying_power),
            "status": str(a.status),
        }
    except Exception as e:
        logger.error(f"[paper] account failed: {e}")
        return None


def positions() -> list:
    c = _get()
    if not c:
        return []
    try:
        out = []
        for p in c.get_all_positions():
            out.append({
                "symbol": p.symbol, "qty": float(p.qty), "side": str(p.side),
                "avg_entry": float(p.avg_entry_price),
                "current": float(p.current_price or 0),
                "market_value": float(p.market_value or 0),
                "unrealized_pl": float(p.unrealized_pl or 0),
                "unrealized_plpc": float(p.unrealized_plpc or 0) * 100,
            })
        return out
    except Exception as e:
        logger.error(f"[paper] positions failed: {e}")
        return []


def submit_bracket(symbol: str, direction: str, qty: int,
                   entry_limit: float, stop_price, take_profit) -> dict | None:
    """LIMIT entry at `entry_limit` with a bracket (stop_loss + take_profit). LONG→buy,
    SHORT→sell. Returns {order_id,status} on success, {error} on rejection, None if the
    broker is disabled. TIF=DAY so a stale limit doesn't fill at an irrelevant level."""
    c = _get()
    if not c:
        return None
    try:
        from alpaca.trading.requests import LimitOrderRequest, TakeProfitRequest, StopLossRequest
        from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
        side = OrderSide.BUY if (direction or "LONG").upper() == "LONG" else OrderSide.SELL
        kwargs = dict(
            symbol=symbol, qty=int(qty), side=side, time_in_force=TimeInForce.DAY,
            limit_price=round(float(entry_limit), 2), order_class=OrderClass.BRACKET,
        )
        if take_profit:
            kwargs["take_profit"] = TakeProfitRequest(limit_price=round(float(take_profit), 2))
        if stop_price:
            kwargs["stop_loss"] = StopLossRequest(stop_price=round(float(stop_price), 2))
        o = c.submit_order(LimitOrderRequest(**kwargs))
        return {"order_id": str(o.id), "status": str(o.status), "symbol": symbol, "qty": int(qty)}
    except Exception as e:
        logger.error(f"[paper] submit_bracket {symbol} failed: {e}")
        return {"error": str(e)}


def get_order_with_legs(order_id: str) -> dict | None:
    """Parent order status + fill, plus its bracket legs (TP/SL) so reconcile can
    detect entry fill and which leg closed the position + at what price."""
    c = _get()
    if not c:
        return None
    try:
        o = c.get_order_by_id(order_id)
        legs = []
        for leg in (getattr(o, "legs", None) or []):
            legs.append({
                "status": str(leg.status),
                "filled_avg_price": float(leg.filled_avg_price) if leg.filled_avg_price else None,
            })
        return {
            "status": str(o.status),
            "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else None,
            "filled_qty": float(o.filled_qty or 0),
            "legs": legs,
        }
    except Exception as e:
        logger.error(f"[paper] get_order_with_legs failed: {e}")
        return None


def close_position(symbol: str) -> bool:
    c = _get()
    if not c:
        return False
    try:
        c.close_position(symbol)
        return True
    except Exception as e:
        logger.error(f"[paper] close_position {symbol} failed: {e}")
        return False


def cancel_order(order_id: str) -> bool:
    c = _get()
    if not c:
        return False
    try:
        c.cancel_order_by_id(order_id)
        return True
    except Exception as e:
        logger.error(f"[paper] cancel_order failed: {e}")
        return False
