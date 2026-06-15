"""
Market Pulse — named, tunable threshold constants (no magic numbers in logic).

Pulse is a standalone, market-WIDE, end-of-day regime read (IBD-style). It is
completely separate from the per-signal confluence engine. Every threshold here
can be tuned without touching the pillar / regime code.
"""
from __future__ import annotations

# ── Pillar 1: distribution days ────────────────────────────────────────────
DD_WINDOW         = 25       # rolling trading-day window
DD_DOWN_PCT       = 0.002    # index closes down >= 0.2%
DD_EXPIRE_RISE    = 0.05     # a DD expires once the index closes 5%+ above its close

# ── Pillar 2/3: 52-week + moving averages ──────────────────────────────────
HL_LOOKBACK       = 252      # trading days for the 52-week high/low
SMA_FAST          = 50
SMA_SLOW          = 200

# ── Pillar 4: A/D divergence ───────────────────────────────────────────────
AD_NEAR_HIGH_PCT  = 0.005    # SPY within 0.5% of its 52-week high
AD_DIVERGENCE_LOOKBACK = HL_LOOKBACK   # A/D "new high" measured over the same window

# ── Pillar 5: VIX bands ────────────────────────────────────────────────────
VIX_CALM_MAX      = 15.0     # < 15 = calm
VIX_NORMAL_MAX    = 20.0     # 15-20 = normal
VIX_ELEVATED_MAX  = 30.0     # 20-30 = elevated; > 30 = high
VIX_SMA           = 10       # trend reference (close vs its own 10-day SMA)

# ── Regime tiers ───────────────────────────────────────────────────────────
# CORRECTION
DD_CORRECTION     = 6        # dd_max >= 6
PCT200_CORRECTION = 40.0     # % above 200d < 40
PCT50_WEAK        = 40.0     # paired with net_nhnl < 0
# UNDER PRESSURE
DD_PRESSURE       = 5        # dd_max >= 5
PCT50_PRESSURE    = 50.0     # % above 50d < 50
# VIX soft confirmers (never a sole trigger)
VIX_PRESSURE_LEVEL = 30.0    # vix > 30 AND rising -> contributes to UNDER_PRESSURE
DD_BOUNDARY        = 4        # dd_max == 4 ...
VIX_BOUNDARY_LEVEL = 25.0    # ... AND vix > 25 AND rising -> soft-downgrade to UNDER_PRESSURE

# ── Regime labels (match the Supabase CHECK constraint) ────────────────────
CONFIRMED_UPTREND = "CONFIRMED_UPTREND"
UNDER_PRESSURE    = "UNDER_PRESSURE"
CORRECTION        = "CORRECTION"
