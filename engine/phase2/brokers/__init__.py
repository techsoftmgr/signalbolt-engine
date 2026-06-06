"""
Pluggable broker adapters for Portfolio Doctor.

Each adapter returns (holdings, cash):
  holdings = [{ticker, qty, avg_price, current_price?}, ...]
  cash     = float

Adding a broker = drop a new module + register it here. CSV + Alpaca are live;
the rest are stubbed (no official/easy retail API) and raise a clear message
until implemented.
"""
from __future__ import annotations

# which brokers are actually wired (vs pluggable-but-not-yet)
SUPPORTED = {
    "csv": True, "alpaca": True,
    "robinhood": False, "fidelity": False, "schwab": False, "ibkr": False,
}


def load(broker: str, **kw):
    """Return (holdings, cash) from the named broker. Raises ValueError if a
    broker isn't wired yet (so the caller can show a friendly message)."""
    name = (broker or "").lower().strip()
    if name == "csv":
        from .csv_adapter import parse
        return parse(kw.get("csv") or kw.get("text") or "")
    if name == "alpaca":
        from .alpaca_adapter import fetch
        return fetch(kw.get("api_key"), kw.get("secret"), kw.get("paper", True))
    if name in SUPPORTED:
        raise ValueError(f"Broker '{name}' is on the roadmap but not yet connected. "
                         f"Use CSV upload or Alpaca for now.")
    raise ValueError(f"Unknown broker '{broker}'.")
