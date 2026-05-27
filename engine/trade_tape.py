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

WINDOW_SECS     = 300        # 5-minute rolling window
MAX_EVENTS      = 5000       # hard cap per ticker (overrides time window if exceeded)
BLOCK_SIZE      = 10_000     # trades >= this size count as institutional block prints
RATE_WINDOW     = 60         # trades-per-second computed over last N seconds

# ── Push alert thresholds ──
PUSH_BLOCK_SIZE = 100_000    # only alert on truly large prints (~$5M+ at typical prices)
PUSH_THROTTLE_S = 1800       # max 1 push per ticker per 30 min (prevents flood on liquid names)
_last_push: dict[str, float] = {}
_last_push_lock = threading.Lock()


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

    # Block-print push alert (throttled per ticker — see PUSH_THROTTLE_S)
    # CRITICAL: this runs inside the async on_trade event loop. The push call
    # makes synchronous HTTP to Expo's API which would block the entire loop
    # and starve every other ticker's tick processing → scans stop firing.
    # Always dispatch in a daemon thread so we return immediately.
    if size >= PUSH_BLOCK_SIZE:
        with _last_push_lock:
            last = _last_push.get(ticker, 0)
            if now - last >= PUSH_THROTTLE_S:
                _last_push[ticker] = now
                send = True
            else:
                send = False
        if send:
            try:
                threading.Thread(
                    target=_dispatch_block_alert,
                    args=(ticker, int(size), float(price)),
                    daemon=True,
                ).start()
            except Exception:
                pass


def _dispatch_block_alert(ticker: str, size: int, price: float) -> None:
    """Fire push in a daemon thread so we don't block the stream event loop."""
    try:
        from engine import push
        push.send_block_print_alert(ticker, size, price)
    except Exception as e:
        logger.debug(f"[trade_tape] block alert dispatch failed for {ticker}: {e}")


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


def compute_signal_bonus(ticker: str) -> dict:
    """
    Calculate a confluence bonus for a fired signal based on current tape.
    Returns {"bonus": int, "reasons": [str]} — caller appends to score_breakdown.

    Bonuses:
      +5  block prints present (institutional flow agrees)
      +3  tape rate >= 3.0/sec (hot tape = real momentum)
      +2  tape volume >= 250k shares (deep liquidity)

    Penalties (negative bonus):
      -3  zero block prints AND tape rate < 1.0/sec (only retail interest)

    Capped at -3..+10. Stored in score_breakdown.tape_bonus so the validator
    can correlate "did tape bonus actually predict outcome."
    """
    summary = get_summary(ticker)
    if summary is None:
        return {"bonus": 0, "reasons": ["no tape data"]}

    bonus   = 0
    reasons: list[str] = []

    tps   = summary.get("trades_per_sec", 0.0)
    vol   = summary.get("total_volume", 0)
    bcount= summary.get("block_count", 0)
    bvol  = summary.get("block_volume", 0)

    if bcount > 0:
        bonus += 5
        reasons.append(f"+5 institutional block prints ({bcount} blocks, {bvol:,} shares)")
    if tps >= 3.0:
        bonus += 3
        reasons.append(f"+3 hot tape ({tps:.1f}/sec)")
    if vol >= 250_000:
        bonus += 2
        reasons.append(f"+2 deep liquidity ({vol:,} shares)")
    if bcount == 0 and tps < 1.0:
        bonus -= 3
        reasons.append(f"-3 retail-only tape ({tps:.1f}/sec, no blocks)")

    bonus = max(-3, min(10, bonus))
    return {"bonus": bonus, "reasons": reasons or ["neutral tape"]}


def reset(ticker: Optional[str] = None) -> None:
    """Clear tape state (for tests / manual reset). Pass ticker=None for all."""
    with _tapes_lock:
        if ticker is None:
            _tapes.clear()
        else:
            _tapes.pop(ticker, None)
