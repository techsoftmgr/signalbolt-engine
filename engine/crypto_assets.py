"""
Crypto symbol registry + helpers. Maps the base symbol a user types/stores
(e.g. "BTC") to the Alpaca crypto pair ("BTC/USD"), so crypto support stays
additive and fully isolated from the stock path.

Routing rule: a watchlist symbol is treated as crypto iff its base is in the
curated map below. The set is deliberately limited to well-known coins. A few
bases (SOL, LINK) also exist as obscure micro-cap *stock* tickers (ReneSola,
Interlink) — in a trading app the coin intent dominates, so we resolve those to
crypto. Anything not in the map flows through the normal stock pipeline.
"""
from __future__ import annotations

# base symbol -> Alpaca USD pair
_PAIRS: dict[str, str] = {
    "BTC": "BTC/USD",   "ETH": "ETH/USD",   "SOL": "SOL/USD",   "DOGE": "DOGE/USD",
    "XRP": "XRP/USD",   "ADA": "ADA/USD",   "AVAX": "AVAX/USD", "LINK": "LINK/USD",
    "DOT": "DOT/USD",   "LTC": "LTC/USD",   "BCH": "BCH/USD",   "SHIB": "SHIB/USD",
    "UNI": "UNI/USD",   "AAVE": "AAVE/USD", "MKR": "MKR/USD",   "XTZ": "XTZ/USD",
    "GRT": "GRT/USD",   "BAT": "BAT/USD",   "CRV": "CRV/USD",   "YFI": "YFI/USD",
    "SUSHI": "SUSHI/USD","USDT": "USDT/USD","USDC": "USDC/USD",
}


def _norm(sym: str) -> str:
    return (sym or "").upper().replace("/", "").replace("-", "").replace(" ", "").strip()


def base(sym: str) -> str:
    """Normalise to a base symbol: 'BTC/USD' | 'BTCUSD' | 'BTC' -> 'BTC'.
    Returns '' if the symbol is not a known crypto."""
    n = _norm(sym)
    if n in _PAIRS:
        return n
    if n.endswith("USD") and n[:-3] in _PAIRS:   # 'BTCUSD' -> 'BTC'
        return n[:-3]
    return ""


def is_crypto(sym: str) -> bool:
    return bool(base(sym))


def to_pair(sym: str) -> str | None:
    """Base/any form -> Alpaca pair ('BTC' -> 'BTC/USD'), or None if unknown."""
    b = base(sym)
    return _PAIRS.get(b) if b else None


def all_bases() -> list[str]:
    return list(_PAIRS.keys())
