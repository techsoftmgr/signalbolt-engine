"""
Leveraged & inverse ETF / ETN signal-firing policy.

Daily-rebalanced products are path-dependent (volatility decay + compounding
drift). Two tiers:

  • BLOCKED_LEVERAGED_ETFS — NEVER fire a signal. The worst products:
      - SINGLE-STOCK leveraged (TSLL/NVDL/MSTU/CONL…): single-name gap risk +
        leverage + decay = uniquely dangerous.
      - VOLATILITY ETNs (UVXY/VXX/SVXY/UVIX…): VIX-futures, contango decay — bleed
        in any non-trending tape regardless of direction.
      - COMMODITY / PRECIOUS-METAL futures (BOIL/KOLD/UCO/NUGT/JNUG/AGQ…): futures-
        roll decay, not clean index tracking.
      - LEVERAGED BONDS/RATES (TMF/TMV/TBT…) + FOREIGN/EM (YINN/YANG…): niche +
        decay; not the broad US equity index the short-horizon signals assume.

  • LEVERAGED_INDEX_ETFS — leveraged BROAD-INDEX & US SECTOR EQUITY (TQQQ/SQQQ,
    SPXL/SPXU, SOXL/SOXS, TNA/TZA, FAS/FAZ, TECL/TECS, LABU/LABD, ERX/GUSH…).
    Liquid, diversified, commonly traded on technicals → ALLOWED for short-horizon
    signals (day/swing/momentum/breakout/breakdown/cycle), but BLOCKED on the
    months-horizon signals (deep_value/position_trade/LEAPS) where multi-month
    decay ruins a 3x hold.

Plain 1x ETFs (SPY/QQQ/IWM/XLK/SMH/GLD…) are not leveraged → always tradeable.
"""
from __future__ import annotations

# Months-horizon strategies — even an allowed leveraged INDEX ETF must not fire
# these (a 3x ETF held for months is a decay disaster).
_LONG_HORIZON_STRATEGIES: frozenset[str] = frozenset({"deep_value", "position_trade"})

# ── ALLOWED for short-horizon (blocked only on long-horizon): broad-index + US sector equity ──
LEVERAGED_INDEX_ETFS: frozenset[str] = frozenset({
    # Broad US index (2x/3x long + inverse)
    "TQQQ", "SQQQ", "QLD", "QID", "UPRO", "SPXU", "SPXL", "SPXS", "SSO", "SDS",
    "UDOW", "SDOW", "DDM", "DXD", "TNA", "TZA", "URTY", "SRTY", "UWM", "TWM",
    # US sector equity (2x/3x)
    "SOXL", "SOXS", "USD", "SSG", "TECL", "TECS", "ROM", "REW",
    "FAS", "FAZ", "DRN", "DRV", "LABU", "LABD", "CURE", "RXD", "DPST", "DFEN",
    "ERX", "ERY", "GUSH", "DRIP",
})

# ── ALWAYS blocked: single-stock, vol ETNs, commodity/metal futures, bonds, EM ──
BLOCKED_LEVERAGED_ETFS: frozenset[str] = frozenset({
    # Single-stock leveraged
    "TSLL", "TSLT", "TSLQ", "TSLS", "TSLR", "NVDL", "NVDU", "NVDD", "NVDS", "NVDX",
    "MSTU", "MSTX", "MSTZ", "CONL", "AMDL", "AMUU", "AMDD", "GGLL", "GGLS",
    "AAPU", "AAPD", "METU", "METD", "AMZU", "AMZD", "MSFU", "MSFD", "PLTU", "COIU",
    # Volatility ETNs/ETFs
    "UVXY", "SVXY", "VXX", "VIXY", "UVIX", "SVIX", "VIXM",
    # Commodity / precious-metal (futures/miners — decay)
    "BOIL", "KOLD", "UCO", "SCO", "UGA", "OILU", "OILD",
    "NUGT", "DUST", "JNUG", "JDST", "UGL", "GLL", "AGQ", "ZSL", "USLV", "DSLV", "PILL",
    # Leveraged bonds / rates
    "TMF", "TMV", "TBT", "UBT", "TYO", "TYD",
    # Foreign / EM leveraged
    "YINN", "YANG", "CWEB", "CHAU", "EDC", "EDZ",
})


def is_blocked_leveraged_etf(ticker: str) -> bool:
    """ALWAYS-blocked leveraged product (single-stock / vol / commodity / bond / EM)."""
    return (ticker or "").upper() in BLOCKED_LEVERAGED_ETFS


def is_leveraged_index_etf(ticker: str) -> bool:
    """Leveraged broad-index / US-sector equity ETF (OK short-horizon, not long)."""
    return (ticker or "").upper() in LEVERAGED_INDEX_ETFS


def should_block_signal(ticker: str, strategy_type: str | None = None) -> bool:
    """The firing-gate rule. Block if: (a) an always-blocked leveraged product, or
    (b) a leveraged INDEX/sector ETF on a months-horizon strategy. Short-horizon
    signals on leveraged index/sector ETFs are allowed."""
    sym = (ticker or "").upper()
    if sym in BLOCKED_LEVERAGED_ETFS:
        return True
    if sym in LEVERAGED_INDEX_ETFS and (strategy_type or "") in _LONG_HORIZON_STRATEGIES:
        return True
    return False
