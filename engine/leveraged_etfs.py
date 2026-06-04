"""
Leveraged & inverse ETF / ETN exclusion list.

These products (2x/3x long, -1x/-2x/-3x inverse, and vol ETNs) are DAILY-rebalanced
and path-dependent: volatility decay + compounding drift make them mean-revert and
bleed over multi-day holds. They are NOT suitable for the swing / momentum / breakout
/ breakdown / cycle signals (which assume a clean multi-day directional move), so the
engine must never FIRE a signal on them. Blocked centrally in runner._is_untradeable
(every stock fire path) + _write_option_signal (options on them are leverage-on-
leverage — worse).

Curated set of the common, liquid leveraged/inverse products + the major
single-stock leveraged ETFs. Not exhaustive — new single-stock leveraged products
launch often; add here as needed. (Plain 1x sector/index ETFs like SPY/QQQ/XLK/SMH
are NOT leveraged and stay tradeable.)
"""
from __future__ import annotations

LEVERAGED_INVERSE_ETFS: frozenset[str] = frozenset({
    # ── Broad index (2x/3x long + inverse) ──
    "TQQQ", "SQQQ", "QLD", "QID", "UPRO", "SPXU", "SPXL", "SPXS", "SSO", "SDS",
    "UDOW", "SDOW", "DDM", "DXD", "TNA", "TZA", "URTY", "SRTY", "UWM", "TWM",
    # ── Volatility (decaying ETNs/ETFs) ──
    "UVXY", "SVXY", "VXX", "VIXY", "UVIX", "SVIX", "VIXM",
    # ── Sector 2x/3x ──
    "SOXL", "SOXS", "USD", "SSG", "TECL", "TECS", "ROM", "REW",
    "FAS", "FAZ", "DRN", "DRV", "LABU", "LABD", "CURE", "RXD",
    "DPST", "WDRW", "PILL", "DFEN",
    # ── Energy / commodity 2x/3x ──
    "ERX", "ERY", "GUSH", "DRIP", "BOIL", "KOLD", "UCO", "SCO", "UGA", "OILU", "OILD",
    # ── Gold / silver / miners 2x/3x ──
    "NUGT", "DUST", "JNUG", "JDST", "UGL", "GLL", "AGQ", "ZSL", "USLV", "DSLV",
    # ── China / EM 2x/3x ──
    "YINN", "YANG", "CWEB", "CHAU", "EDC", "EDZ",
    # ── Treasury / rates 2x/3x ──
    "TMF", "TMV", "TBT", "UBT", "TYO", "TYD",
    # ── Single-stock leveraged (growing category) ──
    "TSLL", "TSLT", "TSLQ", "TSLS", "TSLR", "NVDL", "NVDU", "NVDD", "NVDS", "NVDX",
    "MSTU", "MSTX", "MSTZ", "CONL", "AMDL", "AMUU", "AMDD", "GGLL", "GGLS",
    "AAPU", "AAPD", "METU", "METD", "AMZU", "AMZD", "MSFU", "MSFD", "PLTU", "COIU",
})


def is_leveraged_etf(ticker: str) -> bool:
    """True if `ticker` is a known leveraged/inverse ETF/ETN that must NOT fire a
    signal (daily-rebalanced, decay-prone). Case-insensitive."""
    return (ticker or "").upper() in LEVERAGED_INVERSE_ETFS
