"""
Trade Tape — rolling per-ticker stream analytics from Alpaca trade ticks.

Hooks into the existing on_trade handler in engine/stream.py (every trade tick).
Maintains a rolling window per ticker so the entry gate (or any other consumer)
can answer: "what does the tape look like right now for NVDA?"

Tracked per ticker:
  - Block prints (single trades > BLOCK_SIZE shares) in the rolling window
  - Trade rate (trades/sec) over last 60s — proxy for tape acceleration
  - Window VWAP (volume-weighted average price)
  - Total share volume in window
  - Latest trade price + timestamp

Design notes:
  - Pure in-memory, per-process. State is rebuilt on engine restart (acceptable
    because the window is short — 5 min default).
  - Thread-safe: uses a single lock per ticker. record_trade() is called from
    the asyncio event loop in stream.py at high frequency, get_summary() from
    scoring threads.
  - Bounded memory: each ticker's deque is capped by window duration + a hard
    MAX_EVENTS ceiling so a single ultra-liquid name (e.g. SPY) can't OOM.
  - No bid/ask context here. Aggressor-side classification requires quote
    subscriptions which we don't have yet — punted to a future iteration.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("signalbolt.trade_tape")

# ── Configuration ───────────────────────────────────────────────────────────

WINDOW_SECS  = 300         # 5-minute rolling window
MAX_EVENTS   = 5000        # hard cap per ticker (overrides time window if exceeded)
BLOCK_SIZE   = 10_000      # trades >= this size count as institutional block prints
RATE_WINDOW  = 60          # trades-per-second computed over last N seconds


# ── Per-ticker rolling state ────────────────────────────────────────────────

@dataclass
class _TapeWindow:
    # deque of (ts_epoch, price, size) tuples
    events: deque   = field(default_factory=lambda: deque(maxlen=MAX_EVENTS))
    lock:   threading.Lock = field(default_factory=threading.Lock)


_tapes: dict[str, _TapeWindow] = {}
_tapes_lock = threading.Lock()


def _get_window(ticker: str) -> _TapeWindow:
    """Get-or-create the per-ticker tape window."""
    w = _tapes.get(ticker)
    if w is not None:
        return w
    with _tapes_lock:
        w = _tapes.get(ticker)
        if w is None:
            w = _TapeWindow()
            _tapes[ticker] = w
        return w


# ── Public API ──────────────────────────────────────────────────────────────

def record_trade(ticker: str, price: float, size: float) -> None:
    """
    Called from on_trade() in stream.py for every trade tick.
    Must be very fast — runs at multi-kHz on liquid tickers.
    """
    if price <= 0 or size <= 0:
        return
    w = _get_window(ticker)
    now = time.time()
    cutoff = now - WINDOW_SECS
    with w.lock:
        w.events.append((now, float(price), float(size)))
        # Trim events older than window. deque maxlen also caps total size.
        while w.events and w.events[0][0] < cutoff:
            w.events.popleft()


def get_summary(ticker: str) -> Optional[dict]:
    """
    Return a snapshot summary of the tape for `ticker`, or None if we have
    no data. Callers (entry_gate, admin endpoints) read this.
    """
    w = _tapes.get(ticker)
    if w is None:
        return None
    now = time.time()
    cutoff_window = now - WINDOW_SECS
    cutoff_rate   = now - RATE_WINDOW

    with w.lock:
        # Trim stale (defensive — in case record_trade hasn't fired recently)
        while w.events and w.events[0][0] < cutoff_window:
            w.events.popleft()
        if not w.events:
            return None

        total_volume = 0.0
        vwap_num     = 0.0           # Σ price × size
        block_count  = 0
        block_volume = 0.0
        recent_count = 0             # trades within RATE_WINDOW seconds
        last_price   = w.events[-1][1]
        last_ts      = w.events[-1][0]
        n            = len(w.events)

        for ts, p, sz in w.events:
            total_volume += sz
            vwap_num     += p * sz
            if sz >= BLOCK_SIZE:
                block_count  += 1
                block_volume += sz
            if ts >= cutoff_rate:
                recent_count += 1

        vwap = (vwap_num / total_volume) if total_volume > 0 else last_price
        trades_per_sec = recent_count / RATE_WINDOW if RATE_WINDOW > 0 else 0.0

    return {
        "ticker":         ticker,
        "trades":         n,
        "total_volume":   round(total_volume),
        "vwap":           round(vwap, 4),
        "block_count":    block_count,
        "block_volume":   round(block_volume),
        "trades_per_sec": round(trades_per_sec, 2),
        "last_price":     round(last_price, 4),
        "last_age_sec":   round(now - last_ts, 1),
        "window_secs":    WINDOW_SECS,
    }


def get_all_summaries(limit: int = 30) -> list[dict]:
    """Return summaries for all tracked tickers, sorted by activity desc."""
    summaries = []
    for t in list(_tapes.keys()):
        s = get_summary(t)
        if s and s["trades"] > 0:
            summaries.append(s)
    summaries.sort(key=lambda x: x["total_volume"], reverse=True)
    return summaries[:limit]


def reset(ticker: Optional[str] = None) -> None:
    """Clear tape state (for tests / manual reset). Pass ticker=None for all."""
    with _tapes_lock:
        if ticker is None:
            _tapes.clear()
        else:
            _tapes.pop(ticker, None)
