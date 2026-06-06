"""
SignalBolt Phase 2 — Trader Intelligence Platform.

ADDITIVE ONLY. Everything in this package is new, isolated, and gated behind
feature flags (engine/phase2/flags.py). It does NOT modify, import-mutate, or
depend on the behavior of any existing signal-engine module — it only READS from
existing data sources (regime, drawdown, bars, earnings, community). Deleting
this package + its endpoints removes Phase 2 entirely, with zero impact on
existing functionality.

Modules:
  flags         — feature-flag reader (env PHASE2_<NAME>, per-flag defaults)
  threat_radar  — Module #4: market threat dashboard (GREEN/YELLOW/ORANGE/RED)
  (more added incrementally: watchlist_intel, community_intel, position_coach,
   signal_followup, portfolio_doctor, trader_home / ai_briefing)
"""
