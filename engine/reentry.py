"""
Shakeout re-entry for TREND_MOMENTUM
====================================
Measured 2026-10: momentum entries are good (the book offered +1,611 pts of MFE), but
~⅓ of the loss is SHAKEOUT — we hit the stop in ~2 days, then the name runs +27%
WITHOUT us. The fix with the best measured edge (+43 pts / book, win-rate 23%→32%,
≈3× any wider-stop or trail-grace variant) is RE-ENTRY: after a losing stop/trend-break
exit, get back into the SAME validated setup if price RECLAIMS the original entry within
a few days. It's asymmetric — only winners reclaim, so it re-engages them without
re-arming the dead entries (that's why it beats a symmetric wider stop).

This does NOT change entry SELECTION — it only re-engages a name the detector already
fired. Confirmation to avoid re-entering a dead name that merely poked up:
  1) a daily CLOSE back above (LONG) / below (SHORT) the ORIGINAL entry,
  2) that close holds beyond the 20-EMA, and
  3) a fresh momentum re-score still agrees with the original direction (trend intact).

Default-OFF (RE_ENTRY_ENABLED). Fires through runner._fire_momentum with a
score_breakdown.reentry tag so the A/B scorecard can measure it. Best-effort; never
raises into the scan.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone

import pandas as pd

logger = logging.getLogger("signalbolt.reentry")

_EMA_CONFIRM = 20          # the reclaim close must hold beyond this EMA
_LOOKBACK_DAYS = 500       # daily bars for the momentum re-score (md.score needs ≥150)


def enabled() -> bool:
    return os.environ.get("RE_ENTRY_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")


def _reclaim_days() -> int:
    try:
        return max(1, int(os.environ.get("RE_ENTRY_RECLAIM_DAYS", "5")))
    except (TypeError, ValueError):
        return 5


def _max_reentries() -> int:
    try:
        return max(1, int(os.environ.get("RE_ENTRY_MAX", "1")))
    except (TypeError, ValueError):
        return 1


def _is_momentum(sig: dict) -> bool:
    return ((sig.get("score_breakdown") or {}).get("detector_source")) == "TREND_MOMENTUM"


def _naive(idx):
    idx = pd.to_datetime(idx)
    try:
        return idx.tz_localize(None) if getattr(idx, "tz", None) is not None else idx
    except (TypeError, AttributeError):
        return idx


def scan(sb) -> dict:
    """Daily (called from the momentum scan): re-enter shaken-out momentum names that
    have reclaimed their entry. Returns a small stats dict. Default-OFF; never raises."""
    stats = {"candidates": 0, "reclaimed": 0, "reentered": 0}
    if not enabled():
        return stats
    try:
        from engine import alpaca_client, momentum_detector as md, runner
    except Exception as e:
        logger.warning(f"[reentry] import failed: {e}")
        return stats

    # Recently-closed momentum signals (14 calendar days covers the ~5-trading-day window).
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()
        rows = (sb.table("signals").select("*")
                .eq("status", "closed").gte("closed_at", cutoff)
                .limit(500).execute().data) or []
    except Exception as e:
        logger.warning(f"[reentry] fetch failed: {e}")
        return stats

    for sig in rows:
        try:
            if not _is_momentum(sig) or sig.get("origin") == "ab_control":
                continue
            rp = sig.get("result_pct")
            if rp is None or float(rp) >= 0:            # only LOSING shakeouts
                continue
            direction = sig.get("direction")
            ticker = sig.get("ticker")
            if direction not in ("LONG", "SHORT") or not ticker:
                continue
            is_long = direction == "LONG"
            entry0 = float(sig.get("entry_price") or 0)
            if entry0 <= 0:
                continue
            sbrk = sig.get("score_breakdown") or {}
            attempt = int((sbrk.get("reentry") or {}).get("attempt", 0))
            if attempt >= _max_reentries():
                continue
            stats["candidates"] += 1

            # dedup: a signal already active on this ticker (incl. a prior re-entry)
            if runner._has_active_signal(sb, ticker, "swing_trade"):
                continue
            # one re-entry per original setup (guard against duplicates across runs)
            origin_id = (sbrk.get("reentry") or {}).get("origin") or sig.get("id")
            try:
                dup = (sb.table("signals").select("id")
                       .contains("score_breakdown", {"reentry": {"origin": origin_id}})
                       .limit(1).execute().data)
                if dup:
                    continue
            except Exception:
                pass

            # daily bars for the reclaim test + momentum re-score
            df = alpaca_client.get_bars(ticker, timeframe="1Day", days=_LOOKBACK_DAYS)
            if df is None or len(df) < _EMA_CONFIRM + 2:
                continue
            df = df.rename(columns={c: str(c).lower() for c in df.columns})
            df.index = _naive(df.index)
            closed_at = pd.to_datetime(sig.get("closed_at"), utc=True).tz_localize(None)
            after = df[df.index.normalize() > closed_at.normalize()]
            n_after = len(after)
            if n_after == 0 or n_after > _reclaim_days():   # no completed bar yet / past the window
                continue

            last_close = float(df["close"].iloc[-1])
            ema = float(df["close"].ewm(span=_EMA_CONFIRM, adjust=False).mean().iloc[-1])
            reclaimed = (last_close > entry0) if is_long else (last_close < entry0)
            holds = (last_close > ema) if is_long else (last_close < ema)
            if not (reclaimed and holds):
                continue
            stats["reclaimed"] += 1

            # re-validate the trend: a fresh momentum re-score must still agree
            ms = md.score(ticker, df)
            if ms is None or ms.bias != direction:
                logger.info(f"[reentry] {ticker} reclaimed ${entry0:.2f} but re-score bias "
                            f"{getattr(ms, 'bias', None)} != {direction} — trend gone, skip")
                continue

            tag = {"origin": origin_id, "attempt": attempt + 1,
                   "original_entry": round(entry0, 2),
                   "shaken_reason": sig.get("closed_reason")}
            runner._fire_momentum(sb, ms, direction, reentry=tag)
            stats["reentered"] += 1
            logger.info(f"[reentry] {ticker} {direction} RE-ENTERED — reclaimed ${entry0:.2f} "
                        f"(attempt {attempt + 1}, {n_after}d after shakeout)")
        except Exception as e:
            logger.debug(f"[reentry] {sig.get('ticker')} error: {e}")

    if stats["reentered"] or stats["reclaimed"]:
        logger.info(f"[reentry] done — {stats['reentered']} re-entered, "
                    f"{stats['reclaimed']} reclaimed / {stats['candidates']} candidates")
    return stats
