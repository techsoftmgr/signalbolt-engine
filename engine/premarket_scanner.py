"""
Pre-market Scanner — runs at 8:00 AM and 9:00 AM ET.

Scans the EXTENDED_UNIVERSE for overnight gappers, high PM volume,
and momentum consistency before regular market hours.  Results are:

  1. Stored in Supabase `premarket_watchlist` (for app display)
  2. Cached in-memory for 2 hours (runner.py reads this to prioritise tickers)

Watch score (0-100):
  gap_pts        0-40  — how big and clean the gap is
  volume_pts     0-25  — PM volume vs 20-day average daily volume
  momentum_pts   0-20  — directional consistency of 1-min bars
  news_pts       0-15  — news catalyst in last 24h

Tickers with watch_score ≥ 60 are flagged "Watch at Open" and are
moved to the front of the morning scanner list so the SMC engine
processes them first at 9:30 AM.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger("signalbolt.premarket")

# ── Watch score threshold ─────────────────────────────────────────────────────
WATCH_AT_OPEN_THRESHOLD = 60   # ≥ 60/100 → "Watch at Open"
CACHE_TTL_SECONDS       = 7_200  # 2 hours

# ── In-memory cache ───────────────────────────────────────────────────────────
@dataclass
class _CacheEntry:
    results:    list["PremarketResult"]
    fetched_at: float   # monotonic timestamp


_cache: Optional[_CacheEntry] = None


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class PremarketResult:
    ticker:           str
    pm_gap_pct:       float          # e.g. +3.2 or -1.8  (percentage, not decimal)
    pm_direction:     str            # "UP" | "DOWN" | "FLAT"
    pm_high:          float
    pm_low:           float
    pm_latest_price:  float
    pm_volume:        int            # total shares traded 4:00–9:29 AM
    pm_volume_ratio:  float          # pm_volume / avg_daily_volume (20d)
    pm_has_news:      bool
    pm_news_headline: str            # first headline or ""
    watch_score:      int            # 0-100
    watch_reasons:    list[str]      # e.g. ["Gap +3.2%", "Volume 4.1×", "Bullish 1m"]
    prior_close:      float
    scanned_at:       str            # ISO-8601 UTC


# ── Core scanner ─────────────────────────────────────────────────────────────

def scan(force: bool = False) -> list[PremarketResult]:
    """
    Run the pre-market scan for the full EXTENDED_UNIVERSE.

    Returns the list sorted by watch_score descending.
    Uses an in-memory cache (2h TTL) so back-to-back calls are free.
    Pass force=True to bypass the cache (used by the 9:00 AM re-scan).
    """
    global _cache

    if not force and _cache is not None:
        age = time.monotonic() - _cache.fetched_at
        if age < CACHE_TTL_SECONDS:
            logger.info(
                f"[premarket] Cache hit — {len(_cache.results)} results "
                f"(age {int(age)}s)"
            )
            return _cache.results

    logger.info("[premarket] Starting pre-market scan …")
    t0 = time.monotonic()

    results = _run_scan()

    _cache = _CacheEntry(results=results, fetched_at=time.monotonic())
    elapsed = time.monotonic() - t0
    logger.info(
        f"[premarket] Scan complete in {elapsed:.1f}s — "
        f"{len(results)} tickers analysed, "
        f"{sum(1 for r in results if r.watch_score >= WATCH_AT_OPEN_THRESHOLD)} "
        f"flagged Watch at Open (≥{WATCH_AT_OPEN_THRESHOLD})"
    )
    return results


def get_watch_list() -> list[PremarketResult]:
    """Return only the tickers flagged Watch at Open (score ≥ threshold)."""
    return [r for r in (_cache.results if _cache else []) if r.watch_score >= WATCH_AT_OPEN_THRESHOLD]


def get_priority_tickers(base_list: list[str]) -> list[str]:
    """
    Re-order `base_list` so high-watch-score tickers appear first.

    Used by runner.py to ensure the SMC engine processes the most
    interesting pre-market movers before the rest of the universe.
    """
    if _cache is None:
        return base_list

    # Build watch_score lookup
    score_map: dict[str, int] = {r.ticker: r.watch_score for r in _cache.results}

    # Partition into watched (score ≥ threshold) and normal
    watched  = [t for t in base_list if score_map.get(t, 0) >= WATCH_AT_OPEN_THRESHOLD]
    normal   = [t for t in base_list if score_map.get(t, 0) <  WATCH_AT_OPEN_THRESHOLD]

    # Sort watched by score desc so the hottest tickers go first
    watched.sort(key=lambda t: score_map.get(t, 0), reverse=True)

    combined = watched + normal
    if watched:
        logger.info(
            f"[premarket] Prioritised {len(watched)} Watch-at-Open tickers: "
            + ", ".join(watched[:10])
        )
    return combined


# ── Internal scan logic ───────────────────────────────────────────────────────

def _run_scan() -> list[PremarketResult]:
    """
    Fetch Alpaca data and score every ticker in the extended universe.
    Falls back gracefully if Alpaca is unavailable.
    """
    from engine.prescreener import EXTENDED_UNIVERSE, CORE_TICKERS

    try:
        daily_df_map  = _fetch_daily_bars_batch(EXTENDED_UNIVERSE)
        pm_bars_map   = _fetch_pm_bars_batch(EXTENDED_UNIVERSE)
    except Exception as e:
        logger.error(f"[premarket] Alpaca fetch failed: {e}")
        return []

    scanned_at = datetime.now(timezone.utc).isoformat()
    results: list[PremarketResult] = []

    for ticker in EXTENDED_UNIVERSE:
        try:
            result = _score_ticker(
                ticker       = ticker,
                daily_df     = daily_df_map.get(ticker),
                pm_bars      = pm_bars_map.get(ticker),
                scanned_at   = scanned_at,
            )
            if result is not None:
                results.append(result)
        except Exception as e:
            logger.debug(f"[premarket] {ticker} scoring failed: {e}")
            continue

    results.sort(key=lambda r: r.watch_score, reverse=True)
    return results


def _score_ticker(
    ticker:     str,
    daily_df,   # pd.DataFrame | None — recent daily bars
    pm_bars,    # pd.DataFrame | None — today's 1-min PM bars (4:00–9:29 AM ET)
    scanned_at: str,
) -> Optional[PremarketResult]:
    """
    Score a single ticker.  Returns None if we don't have enough data.
    """
    import pandas as pd

    if daily_df is None or daily_df.empty:
        return None

    # ── Prior close ──────────────────────────────────────────────────────────
    # Use the last regular-session close (most recent daily bar)
    # Daily bars from Alpaca cover regular hours close price.
    try:
        prior_close = float(daily_df["close"].iloc[-1])
        if prior_close <= 0:
            return None
    except Exception:
        return None

    # ── Average daily volume (20 days) ───────────────────────────────────────
    try:
        avg_daily_vol = float(daily_df["volume"].tail(20).mean())
    except Exception:
        avg_daily_vol = 0.0

    # ── Pre-market bars analysis ──────────────────────────────────────────────
    pm_high          = prior_close
    pm_low           = prior_close
    pm_latest_price  = prior_close
    pm_volume        = 0
    pm_volume_ratio  = 0.0
    momentum_score   = 0   # -10 to +10
    valid_pm         = False

    if pm_bars is not None and not pm_bars.empty:
        valid_pm         = True
        pm_high          = float(pm_bars["high"].max())
        pm_low           = float(pm_bars["low"].min())
        pm_latest_price  = float(pm_bars["close"].iloc[-1])
        pm_volume        = int(pm_bars["volume"].sum())
        pm_volume_ratio  = pm_volume / avg_daily_vol if avg_daily_vol > 0 else 0.0

        # Directional consistency: count bullish vs bearish 1-min closes
        closes = pm_bars["close"].values
        opens  = pm_bars["open"].values
        if len(closes) > 1:
            bullish = int(sum(c > o for c, o in zip(closes, opens)))
            bearish = int(sum(c < o for c, o in zip(closes, opens)))
            total   = bullish + bearish
            if total > 0:
                bull_ratio = bullish / total
                if bull_ratio >= 0.70:
                    momentum_score = 10    # strongly bullish
                elif bull_ratio >= 0.55:
                    momentum_score = 5
                elif bull_ratio <= 0.30:
                    momentum_score = -10   # strongly bearish (good for shorts)
                elif bull_ratio <= 0.45:
                    momentum_score = -5

    # ── Gap calculation ───────────────────────────────────────────────────────
    pm_gap_pct = ((pm_latest_price - prior_close) / prior_close) * 100.0

    if pm_gap_pct > 0.1:
        pm_direction = "UP"
    elif pm_gap_pct < -0.1:
        pm_direction = "DOWN"
    else:
        pm_direction = "FLAT"

    # ── News check ────────────────────────────────────────────────────────────
    pm_has_news      = False
    pm_news_headline = ""
    try:
        from engine import alpaca_client
        news = alpaca_client.get_news(ticker, limit=3)
        if news:
            # Check if any article is from the last 24 hours
            cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
            for article in news:
                created = article.get("created_at", "") or article.get("updated_at", "")
                if created:
                    try:
                        art_dt = datetime.fromisoformat(
                            created.replace("Z", "+00:00")
                        )
                        if art_dt >= cutoff:
                            pm_has_news      = True
                            pm_news_headline = article.get("headline", "")[:120]
                            break
                    except Exception:
                        continue
    except Exception:
        pass

    # ── Watch score calculation ───────────────────────────────────────────────
    gap_abs    = abs(pm_gap_pct)
    reasons: list[str] = []

    # Gap points (0-40): bigger gap = more points, capped at 5% = 40 pts
    gap_pts = 0
    if gap_abs >= 0.5:
        gap_pts = min(40, int((gap_abs / 5.0) * 40))
        reasons.append(f"Gap {'+' if pm_gap_pct > 0 else ''}{pm_gap_pct:.1f}%")

    # Volume points (0-25): 1× = 0pts, 2× = 10pts, 5× = 25pts
    vol_pts = 0
    if pm_volume_ratio >= 1.0:
        vol_pts = min(25, int(((pm_volume_ratio - 1.0) / 4.0) * 25))
        if vol_pts >= 5:
            reasons.append(f"Volume {pm_volume_ratio:.1f}×")

    # Momentum consistency (0-20): abs(momentum_score) → 0-20
    mom_pts = abs(momentum_score) * 2   # ±10 → 0-20
    if mom_pts >= 10:
        direction_word = "Bullish" if momentum_score > 0 else "Bearish"
        reasons.append(f"{direction_word} 1m momentum")

    # News catalyst (0-15)
    news_pts = 15 if pm_has_news else 0
    if pm_has_news:
        reasons.append("News catalyst")

    watch_score = int(gap_pts + vol_pts + mom_pts + news_pts)
    watch_score = max(0, min(100, watch_score))

    # Penalise FLAT direction — low gap kills the score regardless
    if pm_direction == "FLAT" and not pm_has_news:
        watch_score = max(0, watch_score - 20)

    if not reasons:
        reasons.append("Monitoring")

    return PremarketResult(
        ticker           = ticker,
        pm_gap_pct       = round(pm_gap_pct, 2),
        pm_direction     = pm_direction,
        pm_high          = round(pm_high, 4),
        pm_low           = round(pm_low, 4),
        pm_latest_price  = round(pm_latest_price, 4),
        pm_volume        = pm_volume,
        pm_volume_ratio  = round(pm_volume_ratio, 2),
        pm_has_news      = pm_has_news,
        pm_news_headline = pm_news_headline,
        watch_score      = watch_score,
        watch_reasons    = reasons,
        prior_close      = round(prior_close, 4),
        scanned_at       = scanned_at,
    )


# ── Alpaca data fetchers ──────────────────────────────────────────────────────

def _fetch_daily_bars_batch(tickers: list[str]) -> dict[str, "pd.DataFrame"]:
    """
    Fetch 25 days of daily bars for all tickers in ONE batch call.
    Returns {ticker: DataFrame} with columns [open, high, low, close, volume].
    """
    import pandas as pd
    from engine import alpaca_client as _ac

    _ac._init()
    if not _ac._ok or _ac._client is None:
        return {}

    try:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        start = datetime.now(timezone.utc) - timedelta(days=30)
        req   = StockBarsRequest(
            symbol_or_symbols = tickers,
            timeframe          = TimeFrame(1, TimeFrameUnit.Day),
            start              = start,
            feed               = "sip",
        )
        bars = _ac._client.get_stock_bars(req)
        df   = bars.df

        if df is None or df.empty:
            return {}

        df.columns = [c.lower() for c in df.columns]
        result: dict[str, pd.DataFrame] = {}

        if isinstance(df.index, pd.MultiIndex):
            for ticker in tickers:
                try:
                    sub = df.xs(ticker, level=0)
                    if not sub.empty:
                        result[ticker] = sub
                except KeyError:
                    continue
        else:
            result[tickers[0]] = df

        return result
    except Exception as e:
        logger.warning(f"[premarket] daily bars batch failed: {e}")
        return {}


def _fetch_pm_bars_batch(tickers: list[str]) -> dict[str, "pd.DataFrame"]:
    """
    Fetch today's pre-market 1-minute bars (4:00 AM – 9:29 AM ET) in one batch.

    Alpaca SIP returns extended-hours data when using the `sip` feed and
    requesting a start time before 9:30 AM.  We explicitly filter to only
    keep bars in the [4:00, 9:30) ET window.
    """
    import pandas as pd
    from zoneinfo import ZoneInfo
    from engine import alpaca_client as _ac

    _ac._init()
    if not _ac._ok or _ac._client is None:
        return {}

    et = ZoneInfo("America/New_York")
    now_et = datetime.now(et)

    # Build today's PM window in UTC
    today_date = now_et.date()
    pm_start_et = datetime(today_date.year, today_date.month, today_date.day,
                           4, 0, 0, tzinfo=et)
    pm_end_et   = datetime(today_date.year, today_date.month, today_date.day,
                           9, 29, 59, tzinfo=et)

    pm_start_utc = pm_start_et.astimezone(timezone.utc)
    pm_end_utc   = pm_end_et.astimezone(timezone.utc)

    # If we're running before 4 AM ET or on a weekend, no PM bars exist yet
    if now_et.hour < 4:
        logger.debug("[premarket] Before 4 AM ET — no PM bars available yet")
        return {}

    try:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        req = StockBarsRequest(
            symbol_or_symbols = tickers,
            timeframe          = TimeFrame(1, TimeFrameUnit.Minute),
            start              = pm_start_utc,
            end                = min(pm_end_utc, datetime.now(timezone.utc)),
            feed               = "sip",
        )
        bars = _ac._client.get_stock_bars(req)
        df   = bars.df

        if df is None or df.empty:
            return {}

        df.columns = [c.lower() for c in df.columns]
        result: dict[str, pd.DataFrame] = {}

        if isinstance(df.index, pd.MultiIndex):
            for ticker in tickers:
                try:
                    sub = df.xs(ticker, level=0)
                    # Extra safety: keep only 4:00–9:29 AM ET bars
                    sub.index = pd.to_datetime(sub.index, utc=True)
                    sub = sub[
                        (sub.index >= pm_start_utc) &
                        (sub.index <= pm_end_utc)
                    ]
                    if not sub.empty:
                        result[ticker] = sub
                except KeyError:
                    continue
        else:
            result[tickers[0]] = df

        return result
    except Exception as e:
        logger.warning(f"[premarket] PM bars batch failed: {e}")
        return {}


# ── Supabase persistence ──────────────────────────────────────────────────────

def persist_to_supabase(results: list[PremarketResult]) -> None:
    """
    Upsert today's pre-market watchlist into Supabase.

    Runs after each scan (8 AM and 9 AM ET).  The table is keyed on
    (ticker, scan_date) so the 9 AM run overwrites the 8 AM run cleanly.
    """
    if not results:
        return

    try:
        from supabase import create_client
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_KEY", "")
        if not url or not key:
            logger.warning("[premarket] Supabase keys not set — skipping persist")
            return

        sb = create_client(url, key)

        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        rows = []
        for r in results:
            rows.append({
                "ticker":           r.ticker,
                "scan_date":        today_str,
                "pm_gap_pct":       r.pm_gap_pct,
                "pm_direction":     r.pm_direction,
                "pm_high":          r.pm_high,
                "pm_low":           r.pm_low,
                "pm_latest_price":  r.pm_latest_price,
                "pm_volume":        r.pm_volume,
                "pm_volume_ratio":  r.pm_volume_ratio,
                "pm_has_news":      r.pm_has_news,
                "pm_news_headline": r.pm_news_headline,
                "watch_score":      r.watch_score,
                "watch_reasons":    r.watch_reasons,
                "prior_close":      r.prior_close,
                "scanned_at":       r.scanned_at,
            })

        # Upsert in batches of 50 to stay well under Supabase payload limits
        for i in range(0, len(rows), 50):
            batch = rows[i : i + 50]
            sb.table("premarket_watchlist").upsert(
                batch,
                on_conflict="ticker,scan_date"
            ).execute()

        logger.info(
            f"[premarket] Persisted {len(rows)} rows to premarket_watchlist "
            f"(scan_date={today_str})"
        )
    except Exception as e:
        logger.error(f"[premarket] Supabase persist failed: {e}")


# ── Public job entrypoints (called by runner scheduler) ───────────────────────

def run_8am_scan() -> None:
    """
    8:00 AM ET scan — first look at the pre-market landscape.
    Builds the initial Watch at Open list; forces a fresh scan.
    """
    logger.info("[premarket] ── 8:00 AM ET scan starting ──")
    results = scan(force=True)
    persist_to_supabase(results)

    watched = [r for r in results if r.watch_score >= WATCH_AT_OPEN_THRESHOLD]
    if watched:
        logger.info(
            f"[premarket] 8 AM Watch at Open ({len(watched)} tickers): "
            + ", ".join(
                f"{r.ticker}({r.watch_score})" for r in watched[:15]
            )
        )
    else:
        logger.info("[premarket] 8 AM — no tickers cleared Watch at Open threshold")


def run_9am_scan() -> None:
    """
    9:00 AM ET scan — final pre-market read, 30 min before open.
    Forces a fresh scan to capture last-hour PM moves; overwrites DB rows.
    """
    logger.info("[premarket] ── 9:00 AM ET scan starting ──")
    results = scan(force=True)
    persist_to_supabase(results)

    watched = [r for r in results if r.watch_score >= WATCH_AT_OPEN_THRESHOLD]
    if watched:
        top5 = watched[:5]
        logger.info(
            f"[premarket] 9 AM Watch at Open ({len(watched)} tickers): "
            + ", ".join(
                f"{r.ticker}({r.watch_score}, {r.pm_gap_pct:+.1f}%)" for r in top5
            )
        )
    else:
        logger.info("[premarket] 9 AM — no tickers cleared Watch at Open threshold")


def invalidate_cache() -> None:
    """Force a fresh scan on the next call."""
    global _cache
    _cache = None
