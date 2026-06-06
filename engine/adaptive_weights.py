"""
Adaptive Weights Manager
========================
Loads and stores learned scoring weights from Supabase.
The weight optimizer writes here; the scorer reads from here.

Weight format (per strategy + regime):
  {
    'smc':       float,   # L1 contribution % (sum of smc+technical+sentiment+risk ≈ 100)
    'technical': float,   # L2
    'sentiment': float,   # L3
    'risk':      float,   # L4
    'l5_bonus':  float,   # max pts from L5 (default 5)
    'l6_bonus':  float,   # max pts from L6 regime (default 8)
    'l7_bonus':  float,   # max pts from L7 session (default 6)
    'l8_bonus':  float,   # max pts from L8 gamma (default 8)
    'l9_bonus':  float,   # max pts from L9 manipulation (default 8)
  }

regime_type 'ANY' = applies to all regimes (used when insufficient data per regime).
Cache TTL: 1 hour (weights change weekly at most).
"""

import logging
import os
import time
from typing import Optional

logger = logging.getLogger("signalbolt.adaptive_weights")

# ── Default weights (same structure as scorer.py STRATEGY_WEIGHTS) ──────────
# These are the starting point before any optimization has run.
# The optimizer improves on these over time as signal outcomes accumulate.
DEFAULT_WEIGHTS: dict[str, dict] = {
    'scalping': {
        'smc': 15, 'technical': 40, 'sentiment': 15, 'risk': 30,
        'l5_bonus': 0, 'l6_bonus': 8, 'l7_bonus': 6, 'l8_bonus': 8, 'l9_bonus': 8,
    },
    'day_trade': {
        'smc': 25, 'technical': 35, 'sentiment': 25, 'risk': 15,
        'l5_bonus': 5, 'l6_bonus': 8, 'l7_bonus': 6, 'l8_bonus': 8, 'l9_bonus': 8,
    },
    'swing_trade': {
        'smc': 40, 'technical': 30, 'sentiment': 20, 'risk': 10,
        'l5_bonus': 8, 'l6_bonus': 8, 'l7_bonus': 6, 'l8_bonus': 8, 'l9_bonus': 8,
    },
    'options_flow': {
        'smc': 10, 'technical': 20, 'sentiment': 50, 'risk': 20,
        'l5_bonus': 0, 'l6_bonus': 8, 'l7_bonus': 6, 'l8_bonus': 8, 'l9_bonus': 8,
    },
    'dark_pool': {
        'smc': 10, 'technical': 20, 'sentiment': 60, 'risk': 10,
        'l5_bonus': 0, 'l6_bonus': 8, 'l7_bonus': 6, 'l8_bonus': 8, 'l9_bonus': 8,
    },
}

# In-memory cache: (strategy, regime) → (weights_dict, loaded_at_timestamp)
_cache: dict[tuple, tuple] = {}
_CACHE_TTL = 3600  # 1 hour


def _supabase():
    from supabase import create_client
    return create_client(
        os.environ["SUPABASE_URL"],
        os.environ.get("SUPABASE_KEY") or os.environ["SUPABASE_SECRET_KEY"],
    )


def get_weights(strategy_type: str, regime_type: str = "ANY") -> dict:
    """
    Return scoring weights for a strategy + regime combination.
    Priority: regime-specific learned → ANY-regime learned → hardcoded default.
    """
    now = time.time()

    # Resolve the live regime → its bucket (RISK_ON/NEUTRAL/RISK_OFF) since the
    # optimizer now learns per-bucket. Priority: exact regime row (legacy) →
    # bucket row → ANY-regime learned → hardcoded default.
    try:
        from engine.regime_buckets import bucket_of
        bucket = bucket_of(regime_type)
    except Exception:
        bucket = "ANY"
    lookup, _seen = [], set()
    for k in (regime_type, bucket, "ANY"):
        if k and k not in _seen:
            _seen.add(k); lookup.append(k)
    for regime_key in lookup:
        cache_key = (strategy_type, regime_key)
        if cache_key in _cache:
            weights, ts = _cache[cache_key]
            if now - ts < _CACHE_TTL:
                return weights.copy()

        try:
            sb = _supabase()
            rows = (
                sb.table("signal_weights")
                .select("weights, metrics")
                .eq("strategy_type", strategy_type)
                .eq("regime_type", regime_key)
                .order("updated_at", desc=True)
                .limit(1)
                .execute()
                .data
            )
            if rows:
                weights = rows[0]["weights"]
                _cache[cache_key] = (weights, now)
                logger.debug(
                    f"[weights] Loaded {strategy_type}/{regime_key} "
                    f"(win_rate={rows[0].get('metrics', {}).get('win_rate', '?')})"
                )
                return weights.copy()
        except Exception as e:
            logger.debug(f"[weights] DB load failed for {strategy_type}/{regime_key}: {e}")

    # Fall back to hardcoded defaults
    defaults = DEFAULT_WEIGHTS.get(strategy_type, DEFAULT_WEIGHTS["day_trade"]).copy()
    _cache[(strategy_type, "ANY")] = (defaults, now)
    return defaults


def save_weights(
    strategy_type: str,
    regime_type: str,
    weights: dict,
    metrics: Optional[dict] = None,
) -> bool:
    """
    Persist optimized weights to Supabase.

    Args:
        weights:  The weight dict (same keys as DEFAULT_WEIGHTS)
        metrics:  Performance metrics from the optimizer:
                  {'win_rate': float, 'avg_rr': float,
                   'objective': float, 'sample_count': int}
    Returns True on success.
    """
    try:
        from datetime import datetime, timezone
        sb = _supabase()
        row = {
            "strategy_type": strategy_type,
            "regime_type":   regime_type,
            "weights":       weights,
            "metrics":       metrics or {},
            "updated_at":    datetime.now(timezone.utc).isoformat(),
        }
        existing = (
            sb.table("signal_weights")
            .select("id")
            .eq("strategy_type", strategy_type)
            .eq("regime_type",   regime_type)
            .execute()
            .data
        )
        if existing:
            sb.table("signal_weights").update(row).eq("id", existing[0]["id"]).execute()
        else:
            sb.table("signal_weights").insert(row).execute()

        # Invalidate cache so next scorer call picks up new weights
        _cache.pop((strategy_type, regime_type), None)
        _cache.pop((strategy_type, "ANY"), None)

        m = metrics or {}
        logger.info(
            f"[weights] Saved {strategy_type}/{regime_type} — "
            f"win_rate={m.get('win_rate', 0):.1%}  "
            f"avg_rr={m.get('avg_rr', 0):.2f}  "
            f"n={m.get('sample_count', 0)}"
        )
        return True
    except Exception as e:
        logger.error(f"[weights] Save failed for {strategy_type}/{regime_type}: {e}")
        return False


def invalidate_cache() -> None:
    """Clear in-memory cache. Call after optimization completes."""
    _cache.clear()
    logger.debug("[weights] Cache cleared")


def get_all_saved_metrics() -> list[dict]:
    """Return all saved weight rows (for reporting/dashboard)."""
    try:
        return (
            _supabase()
            .table("signal_weights")
            .select("strategy_type, regime_type, metrics, updated_at")
            .order("updated_at", desc=True)
            .execute()
            .data
        ) or []
    except Exception:
        return []
