"""CSV broker adapter — parse a positions export into (holdings, cash).

Tolerant of common column names from broker exports:
  symbol/ticker · qty/quantity/shares · avg_price/cost/cost_basis/purchase_price
  · current_price/price/last/market_price · (optional) a CASH/cash row.
"""
from __future__ import annotations

import csv as _csv
import io
import logging

logger = logging.getLogger("signalbolt.phase2.brokers.csv")

_SYM = ("symbol", "ticker", "instrument")
_QTY = ("qty", "quantity", "shares", "units")
_AVG = ("avg_price", "average_price", "cost", "cost_basis", "purchase_price", "avg_cost", "average_cost")
_CUR = ("current_price", "price", "last", "last_price", "market_price", "mark")


def _norm(k: str) -> str:
    return k.lower().strip().replace(" ", "_")


def _pick(row: dict, keys) -> str | None:
    low = {_norm(k): v for k, v in row.items() if k}     # normalize "Last Price" → "last_price"
    for k in keys:
        if k in low and str(low[k]).strip() not in ("", "-", "—"):
            return str(low[k]).strip()
    return None


def _num(s):
    if s is None:
        return None
    try:
        return float(str(s).replace("$", "").replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def parse(text: str):
    """Return (holdings, cash). Never raises — best-effort over messy exports."""
    holdings, cash = [], 0.0
    if not text or not text.strip():
        return holdings, cash
    try:
        reader = _csv.DictReader(io.StringIO(text))
        for row in reader:
            if not row:
                continue
            sym = _pick(row, _SYM)
            if not sym:
                continue
            if sym.upper() in ("CASH", "USD", "$CASH"):
                cash += _num(_pick(row, _CUR) or _pick(row, _QTY)) or 0.0
                continue
            qty = _num(_pick(row, _QTY))
            avg = _num(_pick(row, _AVG))
            cur = _num(_pick(row, _CUR))
            if qty is None or qty <= 0:
                continue
            holdings.append({"ticker": sym.upper(), "qty": qty,
                             "avg_price": avg or cur or 0.0,
                             "current_price": cur or avg or 0.0})
    except Exception as e:
        logger.debug(f"[csv_adapter] parse error: {e}")
    return holdings, cash
