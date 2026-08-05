"""
ops_diagnostics.py — production observability readable via curl (no CLI/logs access).

The gap behind the multi-deploy troubleshooting: engine WARNING/ERROR logs (e.g. the
per-batch Alpaca errors) only went to Fly logs, which need CLI auth I don't have — so
silent failures were invisible without a probe deploy each time. This keeps the last N
WARNING+ records in memory and exposes them (plus a live subsystem snapshot) so the
next incident is one `GET /ops/diagnostics` away.

PRE-LAUNCH: served without auth (matches the public scorecards). GATE before launch.
"""
from __future__ import annotations
import logging
import time
from collections import deque
from typing import Optional

_RING: deque = deque(maxlen=500)
_installed = False


class _RingHandler(logging.Handler):
    """Keeps the last WARNING+ log records in memory for the diagnostics endpoint."""
    def emit(self, record: logging.LogRecord) -> None:
        try:
            _RING.append({
                "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage()[:700],
            })
        except Exception:
            pass


def install() -> None:
    """Attach the ring handler to the root logger once (idempotent). Call at startup."""
    global _installed
    if _installed:
        return
    try:
        h = _RingHandler(level=logging.WARNING)
        logging.getLogger().addHandler(h)
        _installed = True
    except Exception:
        pass


def recent(limit: int = 100, contains: Optional[str] = None, level: Optional[str] = None) -> list:
    """Most-recent captured warnings/errors, newest last. Filter by substring / level."""
    items = list(_RING)
    if level:
        lv = level.strip().upper()
        items = [r for r in items if r["level"] == lv]
    if contains:
        c = contains.strip().lower()
        items = [r for r in items if c in r["msg"].lower() or c in r["logger"].lower()]
    return items[-max(1, min(limit, 500)):]


def health() -> dict:
    """Live subsystem snapshot — the things that silently break. No heavy compute beyond
    a small, bounded Alpaca probe at increasing sizes (pinpoints size/throughput limits)."""
    out: dict = {"as_of": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    # Alpaca bulk-fetch at increasing sizes — small OK / large empty = a throughput/rate
    # limit (the signature that sent us chasing the wrong root).
    try:
        from engine import quant_score_service as q
        from engine.alpaca_client import get_multi_bars
        uni = q._scan_universe()
        out["universe_size"] = len(uni)
        fetch: dict = {}
        for n in (5, 50, 120):
            sample = uni[:n] if len(uni) >= n else uni
            t0 = time.monotonic()
            try:
                r = get_multi_bars(sample, "1Day", 5) or {}
                fetch[f"n{len(sample)}"] = {"returned": len(r), "ms": int((time.monotonic() - t0) * 1000)}
            except Exception as e:
                fetch[f"n{len(sample)}"] = {"error": repr(e)[:200]}
        out["alpaca_bulk_fetch"] = fetch
    except Exception as e:
        out["alpaca_error"] = repr(e)[:300]
    # Scan / cache freshness (what feeds the dashboard + watchlist).
    try:
        from engine import quant_score_service as q, cache
        scored = cache.kv.get_json(q._SCORED_KEY) or []
        out["scan"] = {
            "scored": len(scored),
            "scored_as_of": cache.kv.get_json(q._SCORED_TS_KEY),
            "degraded": len(scored) < getattr(q, "_MIN_HEALTHY_SCORED", 40),
        }
    except Exception as e:
        out["scan_error"] = repr(e)[:300]
    return out
