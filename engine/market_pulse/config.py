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

# ── Pillar 1b: stalling days (softer distribution — selling into strength) ──
# Close UP but tiny gain, on higher volume, closing in the lower half of range.
STALL_MAX_GAIN_PCT     = 0.002   # gain <= 0.20% (barely advanced)
STALL_CLOSE_RANGE_FRAC = 0.5     # (close-low)/(high-low) <= 0.5 (closed weak)
STALL_WEIGHT           = 0.5     # 2 stalling days ≈ 1 distribution day in the pressure metric

# ── Pillar 2/3: 52-week + moving averages ──────────────────────────────────
HL_LOOKBACK       = 252      # trading days for the 52-week high/low
SMA_FAST          = 50
SMA_SLOW          = 200

# ── Pillar 4: A/D divergence ───────────────────────────────────────────────
AD_NEAR_HIGH_PCT  = 0.005    # SPY within 0.5% of its 52-week high
AD_DIVERGENCE_LOOKBACK = HL_LOOKBACK   # A/D "new high" measured over the same window

# ── Pillar 4b: breadth thrust (Zweig-style — a rare "launch" tell) ──────────
# 10-day EMA of advancers/(advancers+decliners) surges from oversold (<0.40) to
# >0.615 within 10 trading days — historically a powerful rally-launch signal.
BREADTH_THRUST_EMA    = 10
BREADTH_THRUST_LOW    = 0.40
BREADTH_THRUST_HIGH   = 0.615
BREADTH_THRUST_WINDOW = 10   # the low→high surge must happen within this many sessions

# ── Pillar 4c: follow-through day (IBD — a rally off a low CONFIRMS the turn) ─
# On day FTD_MIN_DAY+ of a rally attempt off a recent low, a major index closes up
# >= FTD_MIN_GAIN% on HIGHER volume than the prior day. A fresh FTD (within the last
# FTD_RECENT sessions, low still held) historically signals the uptrend has resumed —
# it soft-UPGRADES the regime so a strong rally off a low flips the read without waiting
# for the distribution-day count to age out.
FTD_MIN_GAIN  = 1.25     # index closes up >= 1.25% ...
FTD_MIN_DAY   = 4        # ... on day 4+ of the rally attempt off the low ...
FTD_RECENT    = 5        # ... and the FTD is within the last 5 sessions (a FRESH confirmation)
FTD_LOOKBACK  = 35       # window to locate the rally-attempt low

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

# ── Intraday provisional read (Part B) ─────────────────────────────────────
INTRADAY_MARGIN        = 0.03    # projected vol must clear prior-day vol by 3% to flip status
INTRADAY_BUCKET_MIN    = 30      # 30-min volume-profile buckets
INTRADAY_CONF_FLOOR_ET = 11.0    # before 11:00 AM ET → TOO_EARLY (too little elapsed to project)
INTRADAY_HIGH_CONF_ET  = 14.0    # from 2:00 PM ET → HIGH confidence (else MEDIUM)
INTRADAY_PROFILE_DAYS  = 120     # calendar days of 30-min bars to build the ~60-session curve


# ── Regime labels (match the Supabase CHECK constraint) ────────────────────
CONFIRMED_UPTREND = "CONFIRMED_UPTREND"
UNDER_PRESSURE    = "UNDER_PRESSURE"
CORRECTION        = "CORRECTION"
