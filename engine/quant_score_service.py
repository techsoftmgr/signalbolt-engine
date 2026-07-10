"""
Quant Score Service — retail-friendly quantitative analysis dashboard.

Computes 8 scores per ticker and groups them into insight buckets:
  • Top Momentum Stocks
  • Best Pullback Setups
  • Breakout Candidates
  • High Relative Volume
  • VWAP Reclaim candidates
  • Oversold Bounce candidates

Final score formula:
  finalQuantScore = 0.25*trend + 0.25*momentum + 0.20*volume
                  + 0.15*breakout + 0.15*meanReversion - 0.20*risk

Cache TTL: 60 seconds (QUANT_CACHE_TTL env var).
Language: never "buy/sell", always "setup detected / watch / avoid".
"""

import logging
import time
import os
from typing import Optional

import numpy as np

from engine import cache

logger = logging.getLogger("signalbolt.quant")

# ── Cache ─────────────────────────────────────────────────────────────────────
_cache: dict       = {}
_cache_ts: float   = 0.0
_CACHE_TTL: int    = int(os.environ.get("QUANT_CACHE_TTL", "60"))
_REDIS_KEY         = "quant:dashboard:v1"   # cross-process precomputed result (worker → web)
_SCORED_KEY        = "quant:scored:v1"      # FULL scored universe (for universe-wide alerts)
_SCORED_TS_KEY     = "quant:scored:v1:asOf" # ISO timestamp of the last universe scan (hub "updated" label)
# The full scan refreshes every ~3 min when healthy. Keep the LAST good scan for
# hours so the watchlist's vol/RSI never blanks if a refresh stalls (e.g. a slow
# data source after hours) — slightly-stale chips beat empty ones. Healthy refreshes
# overwrite it well within this window.
_SCORED_TTL: int   = int(os.environ.get("QUANT_SCORED_TTL", "28800"))   # 8h
# Per-ticker cache for CUSTOM watchlist tickers scored ON DEMAND (names NOT in the universe scan).
# Without this, snapshot() re-fetched 60d daily + 2d 5-min bars and re-scored every such ticker on
# EVERY watchlist load — the "vol/RSI/signal takes a while to show" lag. Cache each freshly-scored
# row so the next load (any user, within the TTL) serves it warm. Short TTL keeps intraday relVol/
# RSI reasonably fresh; universe names stay on the 3-min worker refresh.
_SNAP_KEY          = "quant:snap:"   # + TICKER → one freshly-scored row
_SNAP_TTL: int     = int(os.environ.get("QUANT_SNAP_TTL", "180"))   # 3 min

# ~1y daily bars barely change intraday — cache them so we don't refetch
# ~150×250 bars on every dashboard build.
_long_bars_cache: dict = {}
_long_bars_ts: float   = 0.0
_LONG_BARS_TTL: int    = 2 * 3600

DEFAULT_TICKERS: list[str] = [
    "SPY", "QQQ", "IWM", "DIA",
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "TSLA", "AMD",
    "COIN", "PLTR", "MSTR", "HOOD", "RBLX", "UBER", "ABNB",
    "JPM", "GS", "XOM", "CVX",
    "MARA", "RIOT", "CLSK", "MRNA", "BNTX",
]

_BUCKET_LIMIT = 10   # max cards/episodes per bucket per cycle (was 6)

# ── Liquidity-filtered, movers-augmented scan universe ───────────────────────
# Rather than a hand-picked list, build the scan set dynamically (~every 3h)
# from a broad liquid candidate POOL (curated liquid base + momentum universe +
# today's Alpaca movers), keep only TRADABLE names (price ≥ $MIN, avg
# $-volume ≥ $THRESHOLD), then cap to the most-liquid _LIQ_MAX_NAMES for cost.
# Core + today's movers are always kept (they're liquid by definition and the
# whole point is to catch the in-play names). Falls back to the last good set
# / DEFAULT_TICKERS on any failure. NOT the whole market — illiquid names are
# untradable and would poison the track-record accuracy.
_LIQ_MIN_PRICE      = 5.0
_LIQ_MIN_DOLLAR_VOL = 10_000_000     # $10M/day average
_LIQ_MAX_NAMES      = 150            # per-cycle scan cap (cost on the web VM)
_LIQ_TTL            = 3 * 3600       # rebuild at most every ~3h
_liq_universe: list[str] = []
_liq_built_ts: float     = 0.0


def _candidate_pool() -> list[str]:
    """Broad liquid candidate set before the dollar-volume filter."""
    pool = set(DEFAULT_TICKERS)
    try:
        from engine.prescreener import EXTENDED_UNIVERSE, fetch_movers
        pool.update(EXTENDED_UNIVERSE)
        pool.update(fetch_movers(top=40) or [])
    except Exception:
        pass
    try:
        from engine.momentum_detector import UNIVERSE as _MOM
        pool.update(_MOM)
    except Exception:
        pass
    return sorted(pool)


def _scan_universe() -> list[str]:
    """Liquidity-filtered, movers-augmented universe, rebuilt ~every 3h."""
    global _liq_universe, _liq_built_ts
    if _liq_universe and (time.monotonic() - _liq_built_ts) < _LIQ_TTL:
        return _liq_universe

    pool = _candidate_pool()
    try:
        from engine.alpaca_client import get_multi_bars
        # Only the core is always-kept (mega-liquid). Today's movers are in the
        # pool and must pass the SAME liquidity filter — so we keep liquid movers
        # (real volume) and drop low-float % pumps. That's the whole point of a
        # liquidity base.
        must = set(DEFAULT_TICKERS)

        bars = get_multi_bars(pool, "1Day", 30) or {}
        keep_must: list[str] = []
        ranked:    list[tuple[str, float]] = []
        for tk, df in bars.items():
            if df is None or len(df) < 5:
                continue
            closes = df["close"].values.astype(float)
            vols   = df["volume"].values.astype(float)
            if float(closes[-1]) < _LIQ_MIN_PRICE:
                continue                               # drop sub-$5 junk
            dvol = float(np.mean(closes[-20:] * vols[-20:]))
            if tk in must:
                keep_must.append(tk)
            elif dvol >= _LIQ_MIN_DOLLAR_VOL:
                ranked.append((tk, dvol))
        ranked.sort(key=lambda x: x[1], reverse=True)
        liq = list(dict.fromkeys(keep_must + [t for t, _ in ranked]))[:_LIQ_MAX_NAMES]
        if liq:
            _liq_universe = liq
            _liq_built_ts = time.monotonic()
            logger.info(f"[quant] liquid universe rebuilt: {len(liq)} of {len(pool)} candidates")
            return liq
    except Exception as e:
        logger.warning(f"[quant] liquid universe build failed: {e}")

    return _liq_universe or list(DEFAULT_TICKERS)


def _get_long_bars(tickers: list[str]) -> dict:
    """
    ~1y daily bars, cached ~2h — they barely change intraday, so there's no need
    to refetch ~150×250 bars on every 60s dashboard build (this fetch + the
    per-ticker structure detection was what made /quant/dashboard time out).
    """
    global _long_bars_cache, _long_bars_ts
    if _long_bars_cache and (time.monotonic() - _long_bars_ts) < _LONG_BARS_TTL:
        return _long_bars_cache
    try:
        from engine.alpaca_client import get_multi_bars
        bars = get_multi_bars(tickers, "1Day", 365) or {}
        if bars:
            _long_bars_cache = bars
            _long_bars_ts    = time.monotonic()
    except Exception as e:
        logger.debug(f"[quant] long bars fetch failed: {e}")
    return _long_bars_cache


def _safe(val, default: float = 0.0) -> float:
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _normalize(value: float, lo: float, hi: float) -> float:
    """Map value from [lo, hi] to [0, 100]."""
    if hi == lo:
        return 50.0
    return float(np.clip((value - lo) / (hi - lo) * 100, 0, 100))


def _empty_dashboard(warming: bool = False) -> dict:
    """Full dashboard SHAPE with empty buckets — served on the web when the
    precomputed cache isn't ready, so we never block on an inline rebuild. The
    app renders empty buckets gracefully (and can show a 'warming up' hint)."""
    return {
        "marketRegime": {
            "label":       "Neutral",
            "description": "Refreshing market data…" if warming else _regime_description("Neutral"),
            "color":       _regime_color("Neutral"),
        },
        "topMomentum": [], "pullbacks": [], "breakouts": [], "breakdowns": [],
        "highVolumeUp": [], "highVolumeDown": [], "vwapReclaim": [],
        "oversoldBounce": [], "turnaround": [], "peak": [], "squeeze": [], "allScored": [],
        "warming": warming,
    }


def get_quant_dashboard(symbols: Optional[list[str]] = None, force: bool = False) -> dict:
    """
    Returns the full quant dashboard payload:
    {
      marketRegime: {...},
      topMomentum:  [...],
      pullbacks:    [...],
      breakouts:    [...],
      highVolume:   [...],
      vwapReclaim:  [...],
      oversoldBounce: [...],
    }
    Results are cached for QUANT_CACHE_TTL seconds.
    """
    global _cache, _cache_ts

    now = time.monotonic()
    if not force and (now - _cache_ts < _CACHE_TTL) and _cache:
        return _cache

    # Cross-process precompute: the worker rebuilds on a schedule (force=True) and
    # stores the result in Redis; the web endpoint serves THAT instantly instead
    # of crunching ~150 names on the request path (which was timing out →
    # "engine unreachable"). Only the scheduled refresh actually rebuilds.
    if not force:
        try:
            cached = cache.kv.get_json(_REDIS_KEY)
            if cached:
                _cache    = cached
                _cache_ts = now
                return cached
        except Exception:
            pass
        # Cold cache (no in-process + no Redis) on a NON-force (web) call: do NOT
        # rebuild inline. Scoring ~150 names — incl. the heavy turnaround / peak /
        # cycle_context detectors — blocks the request 10-20s and trips the app's
        # 6s timeout ("engine unreachable", e.g. the Cycle screen right after a
        # deploy). Serve a lightweight 'warming' payload instead; the worker
        # precompute (force=True, every 3 min) fills the real cache shortly. Only
        # force, or an explicit custom symbol list, builds inline.
        if symbols is None:
            return _empty_dashboard(warming=True)

    result = _build_dashboard(symbols or _scan_universe())
    if result:
        try:
            _enrich_breakouts(result)
        except Exception as e:
            logger.debug(f"[quant] breakout enrich skipped: {e}")
        _cache    = result
        _cache_ts = now
        try:
            cache.kv.set_json(_REDIS_KEY, result, _CACHE_TTL * 10)
        except Exception:
            pass
    return result


# Compact fields the watchlist needs to render a plain-English setup line.
_SNAPSHOT_KEEP = (
    "price", "ma20", "atrPct", "rsi", "relativeVolume", "dayChangePct",
    "trendScore", "momentumScore", "macdRising", "momentumAccelerating",
    "setupType", "setupReason", "watchStatus", "finalQuantScore",
    "breakoutLevel", "breakdownLevel", "distToBreakoutPct",
    "turnaroundStage", "peakStage",
    "wk52High", "wk52Low", "wk52Pct",
    "regimeCategory", "rsVsSpy",
    "cmf", "cmfState", "cmfCross", "cmfHistory",
    "adx", "adxState", "squeezeState", "squeezeBias", "mfi", "mfiState",
    "adrPct", "atrStopPct",
)


def snapshot(tickers: list[str]) -> dict:
    """Compact per-ticker quant read for the watchlist (price + current setup
    signals) so each row can show a plain-English "latest setup" line WITHOUT a
    per-ticker round trip. Reuses the cached full universe scan for names already
    scored; scores any remaining (custom) tickers on demand in one batched fetch.
    Best-effort — missing/failed tickers are simply omitted.
    """
    out: dict = {}
    syms = [t.upper().strip() for t in (tickers or []) if t and t.strip()][:50]
    if not syms:
        return out

    # 1) Reuse the cached full scan (refreshed every few min by the worker).
    cached: dict = {}
    try:
        for r in (cache.kv.get_json(_SCORED_KEY) or []):
            tk = r.get("ticker")
            if tk:
                cached[tk] = r
    except Exception:
        cached = {}

    # 1b) Per-ticker on-demand cache: custom (non-universe) tickers scored on a PRIOR load.
    #     This is what stops the watchlist re-fetching + re-scoring them every time.
    pt_cached: dict = {}
    for tk in syms:
        if tk in cached:
            continue
        try:
            r = cache.kv.get_json(_SNAP_KEY + tk)
            if r:
                pt_cached[tk] = r
        except Exception:
            pass

    # 2) Score the ones we have NEITHER in the universe NOR the per-ticker cache, batched —
    #    and write each result to the per-ticker cache so the next load serves it warm.
    missing = [t for t in syms if t not in cached and t not in pt_cached]
    scored_missing: dict = {}
    if missing:
        try:
            from engine.alpaca_client import get_multi_bars, get_latest_prices
            from engine import regime_detector
            daily      = get_multi_bars(missing, timeframe="1Day", days=60) or {}  # ≥20 trading bars for ma20 (see _build_dashboard)
            daily_long = _get_long_bars(missing) or {}
            intraday   = get_multi_bars(missing, timeframe="5Min", days=2) or {}
            prices     = get_latest_prices(missing) or {}
            try:
                regime_type = (regime_detector.detect() or {}).get("regime_type")
            except Exception:
                regime_type = None
            spy_long = daily_long.get("SPY") if isinstance(daily_long, dict) else None
            for tk in missing:
                try:
                    row = _score_ticker(
                        tk, prices.get(tk), daily.get(tk), intraday.get(tk),
                        daily_long_df=daily_long.get(tk), regime_type=regime_type,
                        spy_long_df=spy_long,
                    )
                    if row:
                        scored_missing[tk] = row
                        try:
                            cache.kv.set_json(_SNAP_KEY + tk, row, _SNAP_TTL)   # warm for next load
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"[quant.snapshot] scoring missing tickers failed: {e}")

    for tk in syms:
        r = cached.get(tk) or pt_cached.get(tk) or scored_missing.get(tk)
        if r:
            out[tk] = {k: r.get(k) for k in _SNAPSHOT_KEEP}

    # 3) Enrich the requested (watchlist) tickers with slow-changing fundamentals
    #    — market cap, P/E (price ÷ trailing EPS), next earnings quarter. 24h-cached
    #    per ticker, so this is ~free after the first daily miss; never blocks output.
    try:
        from engine import ticker_fundamentals
        for tk, row in out.items():
            try:
                f = ticker_fundamentals.get(tk) or {}
                if f.get("market_cap"):
                    row["marketCap"] = f["market_cap"]
                if f.get("earnings_period"):
                    row["earningsPeriod"] = f["earnings_period"]
                if f.get("earnings_date"):
                    row["earningsDate"] = f["earnings_date"]   # actual ISO date for "10 Mar 2026"
                eps, px = f.get("eps"), row.get("price")
                if eps and px and eps > 0:
                    row["peRatio"] = round(float(px) / float(eps), 1)
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"[quant.snapshot] fundamentals enrich failed: {e}")
    return out


def cached_score(ticker: str) -> tuple[Optional[dict], Optional[str]]:
    """Return (full _score_ticker row, asOf-ISO) for a ticker from the cached
    universe scan, or (None, None) if it isn't in the scan.

    This is the SAME snapshot the watchlist one-liner reads (quant:scored:v1), so
    when the hub serves from here the two views can't disagree. The worker keeps
    it warm (~3 min); asOf lets the hub show "updated HH:MM". A live recompute is
    still done on demand (?refresh=1) and for tickers outside the scan.
    """
    tk = (ticker or "").upper().strip()
    if not tk:
        return None, None
    try:
        rows = cache.kv.get_json(_SCORED_KEY) or []
        as_of = cache.kv.get_json(_SCORED_TS_KEY)
    except Exception:
        return None, None
    for r in rows:
        if (r.get("ticker") or "").upper() == tk:
            return r, as_of
    return None, None


_FULL_SNAP_KEY = "quant:snapfull:"   # + TICKER → full _score_ticker row for the ticker HUB
_FULL_SNAP_TTL = int(os.environ.get("QUANT_FULL_SNAP_TTL", "180"))   # 3 min


def cached_full_single(ticker: str) -> tuple[Optional[dict], Optional[str]]:
    """Ticker-hub full-row cache for NON-UNIVERSE tickers (GOOG, custom adds).
    Returns (full_row, asOf) or (None, None). Keeps a buzz-alerted / custom ticker
    from being a full cold recompute on every tap (the /overview timeout)."""
    tk = (ticker or "").upper().strip()
    if not tk:
        return None, None
    try:
        blob = cache.kv.get_json(_FULL_SNAP_KEY + tk)
        if blob and isinstance(blob, dict) and blob.get("row"):
            return blob["row"], blob.get("asOf")
    except Exception:
        pass
    return None, None


def store_full_single(ticker: str, row: dict, as_of: str) -> None:
    """Cache a freshly-scored full row for the ticker hub. Best-effort."""
    tk = (ticker or "").upper().strip()
    if not tk or not row:
        return
    try:
        cache.kv.set_json(_FULL_SNAP_KEY + tk, {"row": row, "asOf": as_of}, _FULL_SNAP_TTL)
    except Exception:
        pass


def _enrich_breakouts(result: dict) -> None:
    """Add lifecycle state + catalyst + R:R to each breakout row, plus a top-level
    Watch Accuracy summary. Best-effort — never breaks the dashboard."""
    import os
    from datetime import datetime, timezone

    rows = result.get("breakouts") or []
    sb = None
    try:
        from supabase import create_client
        from engine.runner import _supabase_key
        sb = create_client(os.environ["SUPABASE_URL"], _supabase_key())
    except Exception:
        sb = None

    # Lifecycle state per ticker (open episodes).
    state_by: dict = {}
    if sb is not None and rows:
        try:
            eps = (sb.table("breakout_watch_history")
                     .select("ticker,state,entered_at,triggered_at,trigger_price,enter_price")
                     .eq("bucket", "breakouts")
                     .is_("exited_at", "null").execute().data) or []
            now = datetime.now(timezone.utc)
            for e in eps:
                age = None
                try:
                    age = int((now - datetime.fromisoformat(
                        e["entered_at"].replace("Z", "+00:00"))).total_seconds() // 60)
                except Exception:
                    pass
                state_by[e["ticker"]] = {
                    "state": e.get("state"), "ageMin": age,
                    "enteredAt": e.get("entered_at"), "triggeredAt": e.get("triggered_at"),
                    "triggerPrice": e.get("trigger_price"), "enterPrice": e.get("enter_price"),
                }
        except Exception:
            pass

    try:
        from engine.runner import _has_recent_news
    except Exception:
        _has_recent_news = lambda _t: False

    for r in rows:
        tk = r.get("ticker"); lvl = r.get("breakoutLevel"); px = r.get("price")
        st = state_by.get(tk) or {}
        r["watchState"] = st.get("state") or (
            "TRIGGERED" if (lvl and px and px > lvl) else "WATCHING")
        r["watchAgeMin"] = st.get("ageMin")
        # The point our watch MARKED the breakout (for a chart arrow): trigger
        # if it broke out, else entry. Only present once an episode exists.
        r["markedAt"]    = st.get("triggeredAt") or st.get("enteredAt")
        r["markedPrice"] = st.get("triggerPrice") or st.get("enterPrice")
        try:
            r["catalyst"] = bool(_has_recent_news(tk))
        except Exception:
            r["catalyst"] = False
        # Rough breakout R:R — entry≈price, stop just below the level, target +3%.
        try:
            if lvl and px:
                stop = round(lvl * 0.99, 2); target = round(lvl * 1.03, 2)
                risk = px - stop
                r["stopRef"] = stop
                r["targetRef"] = target
                r["riskReward"] = round((target - px) / risk, 1) if risk > 0 else None
        except Exception:
            pass

    # Top-level Watch Accuracy (judged episodes, last 14d).
    if sb is not None:
        try:
            from engine import breakout_validator
            result["breakoutWatch"] = breakout_validator.watch_accuracy(sb, days=14)
        except Exception:
            result["breakoutWatch"] = {"judged": 0, "accuracy_pct": None}
    else:
        result["breakoutWatch"] = {"judged": 0, "accuracy_pct": None}


def _build_dashboard(tickers: list[str]) -> dict:
    from engine.alpaca_client import get_multi_bars, get_latest_prices
    from engine import regime_detector

    try:
        # 60 CALENDAR days ≈ 40 TRADING bars. The old days=25 yielded only ~16
        # trading bars (weekends + holidays) — below the 20 needed for the 20-day
        # MA, so _score_ticker fell back to ma20=price, which collapsed trendScore
        # and skewed the 20-day high/low for the WHOLE cached scan. That made the
        # watchlist one-liner (fed by this scan) read "no clear setup" while the
        # live hub (400-day fetch) correctly read e.g. "healthy uptrend". Keep this
        # safely above 20 trading bars so the cache matches the hub.
        daily_bars    = get_multi_bars(tickers, timeframe="1Day", days=60)
        # Turnaround scoring needs ~1y of dailies (200-day trend gate, drawdown,
        # swing structure) — a separate, longer fetch from the 25-bar set above.
        daily_long    = _get_long_bars(tickers)
        intraday_bars = get_multi_bars(tickers, timeframe="5Min", days=2)
        latest_prices = get_latest_prices(tickers)

        # Market regime up front — the turnaround falling-knife gate needs the
        # regime during per-ticker scoring (reused for the label block below).
        try:
            regime_raw = regime_detector.detect()
        except Exception:
            regime_raw = {}
        regime_type = (regime_raw.get("regime_type") or regime_raw.get("regime") or "NEUTRAL")

        scored: list[dict] = []
        for ticker in tickers:
            try:
                row = _score_ticker(
                    ticker,
                    latest_prices.get(ticker),
                    daily_bars.get(ticker),
                    intraday_bars.get(ticker),
                    daily_long_df=daily_long.get(ticker),
                    regime_type=regime_type,
                    spy_long_df=daily_long.get("SPY"),
                )
                if row:
                    scored.append(row)
            except Exception as e:
                logger.debug(f"[quant] {ticker}: {e}")

        # ── Market regime label (reuse regime_raw fetched above) ──────────────
        try:
            regime_map = {
                "TRENDING_BULL":   "Bullish",
                "TRENDING_BEAR":   "Bearish",
                "CHOPPY":          "Choppy",
                "HIGH_VOLATILITY": "Volatile",
                "RANGING":         "Risk-Off",
                "NEUTRAL":         "Neutral",
            }
            regime_label = regime_map.get(regime_type, "Neutral")
            regime_desc  = _regime_description(regime_label)
        except Exception:
            regime_label = "Neutral"
            regime_desc  = "Market regime data unavailable."

        # ── Bucket the scored stocks ──────────────────────────────────────────
        sorted_by_quant = sorted(scored, key=lambda x: x["finalQuantScore"], reverse=True)

        # Bucket thresholds intentionally permissive so dashboard stays useful
        # outside market hours when realized volume drops the scores. During
        # live RTH the thresholds still cluster the strongest setups at the top.
        top_momentum = [
            x for x in sorted_by_quant
            if x["momentumScore"] >= 55 and x["finalQuantScore"] >= 45
        ][:_BUCKET_LIMIT]

        pullbacks = [
            x for x in scored
            if x["setupType"] == "pullback"
        ][:_BUCKET_LIMIT]

        breakouts = [
            x for x in scored
            if x["setupType"] == "breakout"
        ][:_BUCKET_LIMIT]

        breakdowns = [
            x for x in scored
            if x["setupType"] == "breakdown"
        ][:_BUCKET_LIMIT]

        # High volume split by DIRECTION — accumulation (up day) vs distribution
        # (down day). Tracked separately so each gets a meaningful accuracy.
        _high_vol = sorted(
            [x for x in scored if x["volumeScore"] >= 50],
            key=lambda x: x["volumeScore"], reverse=True,
        )
        high_volume_up   = [x for x in _high_vol if (x.get("dayChangePct") or 0) > 0][:_BUCKET_LIMIT]
        high_volume_down = [x for x in _high_vol if (x.get("dayChangePct") or 0) < 0][:_BUCKET_LIMIT]

        vwap_reclaim = [
            x for x in scored
            if x["setupType"] == "vwap_reclaim"
        ][:_BUCKET_LIMIT]

        oversold_bounce = [
            x for x in scored
            if x["setupType"] == "oversold_bounce"
        ][:_BUCKET_LIMIT]

        # Turnaround (swing-low reversal) — both Watch and confirmed Buy Zone,
        # ranked by turnaround score. Filtered on the lifecycle STAGE (not
        # setupType) so a confirmed turn surfaces regardless of its base setup.
        turnaround = sorted(
            [x for x in scored if x.get("turnaroundStage") in ("watch", "buyzone")],
            key=lambda x: -(x.get("turnaroundScore") or 0),
        )[:_BUCKET_LIMIT]

        # Peak (swing-high / distribution) — Watch + confirmed Peak, ranked by
        # peak score. Take-profit / hedge primary; short/puts optional.
        peak = sorted(
            [x for x in scored if x.get("peakStage") in ("watch", "peak")],
            key=lambda x: -(x.get("peakScore") or 0),
        )[:_BUCKET_LIMIT]

        # Squeeze (volatility coil → breakout) — universe scan for names in a TTM
        # squeeze. FIRED (just released, actionable now) first, then COILING (the
        # watch pipeline), each ranked by ADX (a coil resolving into a strong trend
        # is the higher-quality setup). Display/discovery only — not a fired signal.
        _sq_rank = {"fired": 0, "on": 1}
        squeeze = sorted(
            [x for x in scored if x.get("squeezeState") in ("on", "fired")],
            key=lambda x: (_sq_rank.get(x.get("squeezeState"), 9), -(x.get("adx") or 0)),
        )[:_BUCKET_LIMIT]

        # Persist the FULL scored universe (every name, not just the top-20
        # allScored) so universe-wide consumers — the breakdown/heavy-selling
        # alerts — can read each ticker's current state and reuse THIS scan
        # instead of re-fetching ~90 names every 15 min.
        try:
            from datetime import datetime as _dt, timezone as _tz
            cache.kv.set_json(_SCORED_KEY, scored, _SCORED_TTL)
            # Stamp when this scan ran so the hub + watchlist can show "updated HH:MM"
            # and prove they're reading the same snapshot.
            cache.kv.set_json(_SCORED_TS_KEY, _dt.now(_tz.utc).isoformat(), _SCORED_TTL)
        except Exception:
            pass

        return {
            "marketRegime": {
                "label":       regime_label,
                "description": regime_desc,
                "color":       _regime_color(regime_label),
            },
            "topMomentum":    top_momentum,
            "pullbacks":      pullbacks,
            "breakouts":      breakouts,
            "breakdowns":     breakdowns,
            "highVolumeUp":   high_volume_up,
            "highVolumeDown": high_volume_down,
            "vwapReclaim":    vwap_reclaim,
            "oversoldBounce": oversold_bounce,
            "turnaround":     turnaround,
            "peak":           peak,
            "squeeze":        squeeze,
            "allScored":      sorted_by_quant[:20],
        }

    except Exception as e:
        logger.error(f"[quant] _build_dashboard failed: {e}")
        return {}


def _recent_max_rsi(closes, lookback: int = 10, period: int = 14) -> float:
    """Max RSI(period) over the last `lookback` bars. A TOP confirms as RSI FALLS, so gating the
    peak detector on CURRENT RSI alone blinds it exactly through the rollover — MSFT hit RSI 73 at
    its 466 top, fell below 60 by the time it broke down, and the detector stopped being evaluated
    (the 466→370 miss). This lets a recently-overbought name keep being evaluated. numpy-only/cheap."""
    try:
        c = np.asarray(closes, dtype=float)
        if c.size < period + lookback + 1:
            return 0.0
        d = np.diff(c)
        best = 0.0
        for i in range(lookback):
            w = d[-(period + i):(-i if i else None)]
            if w.size < period:
                continue
            g = float(w[w > 0].sum()); l = float(-w[w < 0].sum())
            r = 100.0 - 100.0 / (1.0 + (g / (l if l > 0 else 1e-10)))
            if r > best:
                best = r
        return best
    except Exception:
        return 0.0


def _cmf(daily_df, period: int = 20, hist: int = 30) -> tuple[Optional[float], list]:
    """
    Chaikin Money Flow (CMF-N) — money flowing IN vs OUT over N sessions.
      Money Flow Multiplier = ((C-L) - (H-C)) / (H-L)   (0 if H==L)
      Money Flow Volume      = MFM * volume
      CMF(N)                 = Σ MFV(N) / Σ vol(N)   → range [-1, +1]
    Positive = accumulation (buyers in control), negative = distribution. Returns
    (latest_cmf, last `hist` CMF values for a sparkline). None if insufficient data.
    Never raises.
    """
    try:
        if daily_df is None or "high" not in daily_df or "low" not in daily_df or "volume" not in daily_df:
            return None, []
        h = daily_df["high"].values.astype(float)
        l = daily_df["low"].values.astype(float)
        c = daily_df["close"].values.astype(float)
        v = daily_df["volume"].values.astype(float)
        if len(c) < period + 1:
            return None, []
        rng = np.where((h - l) == 0, np.nan, h - l)
        mfm = ((c - l) - (h - c)) / rng
        mfv = np.nan_to_num(mfm) * v                      # 0-range bars contribute 0 flow
        # rolling CMF over `period` (cumsum trick), keep the last `hist` points
        n = len(c)
        series = []
        start = max(period, n - hist + 1)
        for i in range(start, n + 1):
            win_mfv = float(np.sum(mfv[i - period:i]))
            win_vol = float(np.sum(v[i - period:i]))
            series.append(round(win_mfv / win_vol, 4) if win_vol > 0 else 0.0)
        latest = series[-1] if series else None
        return latest, series
    except Exception:
        return None, []


def _cmf_cross(hist: list, buffer: float = 0.05, lookback: int = 2) -> Optional[str]:
    """Recent CMF zero-line cross → 'bullish' (money rotating IN) / 'bearish'
    (distribution starting) / None. Requires a decisive cross past ±buffer within
    the last `lookback` sessions (filters chop right at the zero line)."""
    try:
        if not hist or len(hist) < 3:
            return None
        recent  = hist[-(lookback + 1):]
        latest  = recent[-1]
        earlier = recent[:-1]
        if latest >= buffer and min(earlier) < 0:
            return "bullish"
        if latest <= -buffer and max(earlier) > 0:
            return "bearish"
        return None
    except Exception:
        return None


def _cmf_state(cmf: Optional[float]) -> str:
    """Plain-English money-flow read from the CMF value."""
    if cmf is None:
        return "unknown"
    if cmf >= 0.10:
        return "accumulation"          # money flowing in (institutional buying)
    if cmf >= 0.05:
        return "mild_accumulation"
    if cmf <= -0.10:
        return "distribution"          # money flowing out (institutional selling)
    if cmf <= -0.05:
        return "mild_distribution"
    return "neutral"


def _adx(daily_df, period: int = 14) -> Optional[float]:
    """Wilder ADX — trend STRENGTH (not direction). >25 trending, <20 chop."""
    try:
        import pandas as pd
        h = daily_df["high"].astype(float); l = daily_df["low"].astype(float); c = daily_df["close"].astype(float)
        if len(c) < period * 2 + 1:
            return None
        up = h.diff(); dn = -l.diff()
        plus_dm  = (((up > dn) & (up > 0)) * up.clip(lower=0))
        minus_dm = (((dn > up) & (dn > 0)) * dn.clip(lower=0))
        tr  = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        a   = 1.0 / period
        atr = tr.ewm(alpha=a, adjust=False).mean().replace(0, 1e-10)
        pdi = 100 * plus_dm.ewm(alpha=a, adjust=False).mean()  / atr
        mdi = 100 * minus_dm.ewm(alpha=a, adjust=False).mean() / atr
        dx  = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, 1e-10)
        adx = dx.ewm(alpha=a, adjust=False).mean()
        return round(float(adx.iloc[-1]), 1)
    except Exception:
        return None


def _adx_state(adx: Optional[float]) -> str:
    if adx is None:  return "unknown"
    if adx >= 25:    return "trending"      # trend strong enough to trade
    if adx >= 20:    return "developing"
    return "choppy"                          # no trend — chop/avoid


def _squeeze(daily_df, length: int = 20, bb_mult: float = 2.0, kc_mult: float = 1.5) -> tuple[str, str]:
    """TTM Squeeze — Bollinger Bands inside Keltner Channels = volatility coil.
    Returns (state, bias): state ∈ on (coiling) / fired (just released) / off;
    bias ∈ bull / bear / flat (price vs the basis = expected release direction)."""
    try:
        import pandas as pd
        # Compute on the last COMPLETED daily bar — drop today's still-forming bar so
        # the state is a SETTLED daily read that changes at most once/day (at the
        # close), instead of flickering fired↔coiling intraday as today's bar builds.
        df = daily_df
        try:
            from datetime import datetime as _dt
            from zoneinfo import ZoneInfo as _ZI
            _et = _ZI("America/New_York")
            _last = df.index[-1]
            if getattr(_last, "tzinfo", None) is not None and _last.tz_convert(_et).date() == _dt.now(_et).date():
                df = df.iloc[:-1]
        except Exception:
            pass
        c = df["close"].astype(float); h = df["high"].astype(float); l = df["low"].astype(float)
        if len(c) < length + 2:
            return "unknown", "flat"
        basis = c.rolling(length).mean()
        dev   = bb_mult * c.rolling(length).std()
        up_bb, lo_bb = basis + dev, basis - dev
        tr  = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        rng = tr.rolling(length).mean()
        up_kc, lo_kc = basis + kc_mult * rng, basis - kc_mult * rng
        on = (lo_bb > lo_kc) & (up_bb < up_kc)
        on_now, on_prev = bool(on.iloc[-1]), bool(on.iloc[-2])
        state = "on" if on_now else ("fired" if on_prev else "off")
        mom = float(c.iloc[-1] - basis.iloc[-1])
        bias = "bull" if mom > 0 else ("bear" if mom < 0 else "flat")
        return state, bias
    except Exception:
        return "unknown", "flat"


def _mfi(daily_df, period: int = 14) -> Optional[float]:
    """Money Flow Index — volume-weighted RSI (0-100). >80 overbought, <20 oversold."""
    try:
        h = daily_df["high"].values.astype(float); l = daily_df["low"].values.astype(float)
        c = daily_df["close"].values.astype(float); v = daily_df["volume"].values.astype(float)
        if len(c) < period + 1:
            return None
        tp  = (h + l + c) / 3.0
        rmf = tp * v
        d   = np.diff(tp)
        pos = float(np.sum(np.where(d > 0, rmf[1:], 0.0)[-period:]))
        neg = float(np.sum(np.where(d < 0, rmf[1:], 0.0)[-period:]))
        if neg == 0:
            return 100.0 if pos > 0 else 50.0
        return round(100 - 100 / (1 + pos / neg), 1)
    except Exception:
        return None


def _mfi_state(mfi: Optional[float]) -> str:
    if mfi is None: return "unknown"
    if mfi >= 80:   return "overbought"
    if mfi <= 20:   return "oversold"
    return "neutral"


def _atr_adr(daily_df, price: float, period: int = 14) -> tuple[Optional[float], Optional[float]]:
    """(adr_pct, atr_stop_pct): average daily range % + a 1.5×ATR stop distance as
    % of price (a suggested swing-stop width / sizing aid)."""
    try:
        import pandas as pd
        h = daily_df["high"].astype(float); l = daily_df["low"].astype(float); c = daily_df["close"].astype(float)
        if len(c) < period + 1 or not price:
            return None, None
        tr  = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        atr = float(tr.tail(period).mean())
        adr_pct  = round(float(((h - l) / c * 100).tail(period).mean()), 2)
        stop_pct = round(1.5 * atr / price * 100, 2)
        return adr_pct, stop_pct
    except Exception:
        return None, None


def _regime_category(closes, current: float, ma20: float, rsi: float,
                     spy_long_df, peak_stage: str = "none",
                     setup_type: str = "") -> tuple[str, Optional[float]]:
    """
    Classify a name for the watchlist "Game Plan" tap-filter — a two-sided read
    of what to do in the current tape. Returns (category, rs_vs_spy_pts). Never
    raises. category ∈ {rs_pullback, rs_leader, short_setup, knife, neutral}.

    LONG side — mirrors relative_strength.is_rs_leader (outperforming SPY 20d +
    rising 20-EMA + above 50-SMA): rs_pullback (leader at the 20-MA = +EV long
    even in a weak tape) vs rs_leader (extended → wait).

    SHORT side — "short STRENGTH, not weakness": short_setup = a confirmed top
    (peak) OR a momentum breakdown that ISN'T yet washed out (rsi > 42). We do
    NOT short the falling knife — a measured backtest had shorting the oversold
    low at -0.08R (you short into the squeeze), so a deeply-oversold downtrender
    stays a "knife" (avoid BOTH sides), not a short.
    """
    try:
        import pandas as pd
        if len(closes) < 21 or spy_long_df is None or len(spy_long_df) < 21:
            return "neutral", None
        spy_c     = spy_long_df["close"].values.astype(float)
        ret20     = (closes[-1] - closes[-21]) / closes[-21] if closes[-21] else 0.0
        spy_ret20 = (spy_c[-1]  - spy_c[-21])  / spy_c[-21]  if spy_c[-21]  else 0.0
        rs_vs_spy = round((ret20 - spy_ret20) * 100, 1)
        _ema       = pd.Series(closes).ewm(span=20, adjust=False).mean().values
        ema_rising = bool(_ema[-1] > _ema[-6]) if len(_ema) >= 6 else False
        sma50      = float(np.mean(closes[-50:])) if len(closes) >= 50 else ma20
        above50    = current > sma50
        is_leader  = (rs_vs_spy > 0) and ema_rising and above50
        near_ma    = (ma20 * 0.97) < current < (ma20 * 1.01)

        # SHORT STRENGTH: confirmed top, or a breakdown that's not yet oversold.
        short_top = peak_stage == "peak"
        short_brk = (setup_type == "breakdown") and rsi > 42
        if (short_top or short_brk) and not (is_leader and near_ma):
            return "short_setup", rs_vs_spy

        if is_leader and near_ma and 40 <= rsi <= 62:
            return "rs_pullback", rs_vs_spy   # actionable long even in a weak tape
        if is_leader:
            return "rs_leader", rs_vs_spy     # strong but extended → wait for a pullback
        if (not above50) and (not ema_rising):
            return "knife", rs_vs_spy         # downtrend + washed out → avoid BOTH sides
        return "neutral", rs_vs_spy
    except Exception:
        return "neutral", None


def _score_ticker(
    ticker: str,
    latest_price: Optional[float],
    daily_df,
    intraday_df,
    daily_long_df=None,
    regime_type: Optional[str] = None,
    spy_long_df=None,
) -> Optional[dict]:
    """Compute all quant scores for one ticker. Returns None if data insufficient."""
    import pandas as pd
    from datetime import datetime, timezone

    if daily_df is None or len(daily_df) < 10:
        return None

    closes = daily_df["close"].values.astype(float)
    volumes = daily_df["volume"].values.astype(float)

    current = _safe(latest_price or closes[-1])
    if current <= 0:
        return None

    # ── Trend Score (0-100) ───────────────────────────────────────────────────
    # Based on: price vs 10-day MA, price vs 20-day MA, MA slope
    ma10 = float(np.mean(closes[-10:])) if len(closes) >= 10 else current
    ma20 = float(np.mean(closes[-20:])) if len(closes) >= 20 else current
    ma5  = float(np.mean(closes[-5:]))  if len(closes) >= 5  else current

    trend_raw  = 0.0
    trend_raw += 40 if current > ma20 else -40     # above/below 20MA
    trend_raw += 30 if current > ma10 else -30     # above/below 10MA
    trend_raw += 30 if ma5 > ma20    else -30      # short MA > long MA = uptrend
    trend_score = _normalize(trend_raw, -100, 100)

    # ── Momentum Score (0-100) ────────────────────────────────────────────────
    # RSI(14) + rate of change
    if len(closes) >= 15:
        deltas = np.diff(closes[-15:])
        gains  = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = float(np.mean(gains)) if gains.any() else 0
        avg_loss = float(np.mean(losses)) if losses.any() else 1e-10
        rs  = avg_gain / avg_loss
        rsi = 100 - 100 / (1 + rs)
    else:
        rsi = 50.0

    roc_5  = (current - _safe(closes[-5]))  / _safe(closes[-5],  1) * 100 if len(closes) >= 5  else 0
    roc_10 = (current - _safe(closes[-10])) / _safe(closes[-10], 1) * 100 if len(closes) >= 10 else 0

    momentum_raw = rsi * 0.5 + _normalize(roc_5,  -10, 10) * 0.3 + _normalize(roc_10, -15, 15) * 0.2
    momentum_score = float(np.clip(momentum_raw, 0, 100))

    # ── MACD(12,26,9) histogram — momentum ACCELERATION signal ────────────────
    # The histogram (MACD line − signal line) is the single best read on whether
    # momentum is BUILDING vs FADING, independent of how far price has already
    # travelled. A rising histogram while price is extended above its average is
    # the "trend in motion" signature — it's what lets the Game Plan and the
    # quant verdicts distinguish "extended + still accelerating → ride it" from
    # "extended + stalling → wait for the pullback". Needs ~35 bars to settle.
    macd_hist = 0.0
    macd_rising = False
    macd_hist_series: list[float] = []
    if len(closes) >= 35:
        try:
            c_ser       = pd.Series(closes)
            ema12       = c_ser.ewm(span=12, adjust=False).mean()
            ema26       = c_ser.ewm(span=26, adjust=False).mean()
            macd_line   = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            hist        = (macd_line - signal_line).values.astype(float)
            macd_hist   = float(hist[-1])
            tail        = hist[-3:]
            # rising = latest bar higher than the prior bar AND the 3-bar window
            # is in an up-sequence (momentum genuinely building, not a one-bar blip)
            macd_rising = bool(
                len(tail) >= 2 and tail[-1] > tail[-2] and tail[-1] >= tail[0]
            )
            macd_hist_series = [round(float(x), 4) for x in hist[-6:]]
        except Exception:
            pass

    # Composite "accelerating" read used by the hub + quant verdicts + entry gate:
    # a strong, intact uptrend whose MACD histogram is still building. This is the
    # condition under which an EXTENDED name should be read as "ride it", not "wait".
    momentum_accelerating = bool(
        macd_rising and momentum_score >= 60 and trend_score >= 60
    )

    # ── Volume Score (0-100) ──────────────────────────────────────────────────
    # Relative volume keyed to the ET TRADING SESSION (not the UTC calendar date) so it
    # doesn't reset to 1.0x overnight (the UTC date rolls over at 8 PM ET), and pace-
    # projected via the EMPIRICAL front-loaded curve ONLY during live RTH — which avoids
    # the naive elapsed/390 over-projection at the open (HOOD 9:46am "2.3x" false accum,
    # 2026-06-04). Outside RTH it shows the realized session ratio. Shared single source of
    # truth in volume_curve (same value the heatmap display + signal gating use).
    from engine.volume_curve import session_relvol
    avg_vol = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else (float(np.mean(volumes)) if len(volumes) else 0.0)
    rel_vol = session_relvol(intraday_df, avg_vol)

    volume_score = float(np.clip(_normalize(rel_vol, 0.5, 3.0), 0, 100))

    # ── Breakout Score (0-100) ────────────────────────────────────────────────
    # How close is price to 20-day high? Above = breakout candidate
    high_20 = float(np.max(daily_df["high"].values[-20:])) if "high" in daily_df else float(np.max(closes[-20:]))
    dist_to_high_pct = (current - high_20) / high_20 * 100  # negative = below high
    breakout_score = _normalize(dist_to_high_pct, -5, 2)

    # ── Breakdown Score (0-100) — bearish mirror of the breakout ───────────────
    # How close is price to its 20-day LOW? At/below = breakdown candidate (a
    # risk/avoid heads-up, not a long setup).
    low_20 = float(np.min(daily_df["low"].values[-20:])) if "low" in daily_df else float(np.min(closes[-20:]))
    dist_to_low_pct = (current - low_20) / low_20 * 100 if low_20 else 0.0  # positive = above the low
    breakdown_score = _normalize(-dist_to_low_pct, -5, 2)  # high when at/below the 20-day low

    # 52-week range (uses the long daily df already fetched for MA200; falls back
    # to the short window). Powers the watchlist range bar + "near 52w high/low".
    wk52_high = wk52_low = wk52_pct = None
    try:
        _lref = daily_long_df if (daily_long_df is not None and "high" in daily_long_df) else daily_df
        if "high" in _lref and "low" in _lref:
            wk52_high = round(float(np.max(_lref["high"].values[-252:])), 2)
            wk52_low  = round(float(np.min(_lref["low"].values[-252:])), 2)
            if wk52_high > wk52_low:
                wk52_pct = round((current - wk52_low) / (wk52_high - wk52_low) * 100, 1)  # 0=low, 100=high
    except Exception:
        pass

    # Day change vs prior close — the DIRECTION of today's move. Critical for
    # reading High Volume (up-vol = accumulation, down-vol = distribution).
    prev_close = float(closes[-2]) if len(closes) >= 2 else current
    day_change_pct = (current - prev_close) / prev_close * 100 if prev_close else 0.0

    # ── Mean Reversion Score (0-100) ──────────────────────────────────────────
    # RSI < 35 + price well below MA = oversold bounce candidate
    oversold_score = _normalize(100 - rsi, 0, 100)  # higher when RSI low
    dist_below_ma  = (ma20 - current) / ma20 * 100    # positive = below MA
    mean_rev_raw   = oversold_score * 0.6 + _normalize(dist_below_ma, -3, 5) * 0.4
    mean_reversion_score = float(np.clip(mean_rev_raw, 0, 100))

    # ── Risk Score (0-100; higher = riskier) ─────────────────────────────────
    # Based on ATR% (average true range as % of price).
    # True Range for bar i needs prev_close (close[i-1]). For 14 bars of
    # TR we need 14 prev_closes — closes[-15:-1] is the correct slice
    # (length 14, aligned with highs[-14:]). Previous code used
    # closes[-14:-1] which is length 13 → numpy broadcast failure that
    # silently zeroed out the entire quant dashboard.
    if len(closes) >= 15 and "high" in daily_df and "low" in daily_df:
        highs       = daily_df["high"].values[-14:].astype(float)
        lows        = daily_df["low"].values[-14:].astype(float)
        prev_closes = closes[-15:-1]                # length 14, aligned with highs/lows
        tr = np.maximum.reduce([
            highs - lows,
            np.abs(highs - prev_closes),
            np.abs(lows  - prev_closes),
        ])
        atr     = float(np.mean(tr))
        atr_pct = (atr / current * 100) if current > 0 else 2.0
    else:
        atr_pct = 2.0  # default 2% if not enough history

    risk_score = _normalize(atr_pct, 0.5, 5.0)

    # ── Final Quant Score ─────────────────────────────────────────────────────
    final = (
        0.25 * trend_score
      + 0.25 * momentum_score
      + 0.20 * volume_score
      + 0.15 * breakout_score
      + 0.15 * mean_reversion_score
      - 0.20 * risk_score
    )
    final_score = float(np.clip(round(final), 0, 100))

    # ── VWAP for intraday context ─────────────────────────────────────────────
    vwap = None
    if intraday_df is not None and not intraday_df.empty:
        try:
            from engine.volume_curve import latest_session_bars
            t_df  = latest_session_bars(intraday_df)   # ET session, not UTC date (overnight-safe)
            if not t_df.empty:
                vol_col = t_df["volume"]
                typ_col = (t_df["high"] + t_df["low"] + t_df["close"]) / 3
                total_v = _safe(vol_col.sum())
                if total_v > 0:
                    vwap = round(_safe((typ_col * vol_col).sum() / total_v), 2)
        except Exception:
            pass

    # ── Setup type classification ─────────────────────────────────────────────
    setup_type, setup_reason = _classify_setup(
        current, ma20, rsi, rel_vol, breakout_score, mean_reversion_score, vwap,
        breakdown_score, trend_score,
    )

    # ── Watch status ──────────────────────────────────────────────────────────
    if final_score >= 65 and risk_score < 60:
        watch_status = "actionable"
    elif final_score >= 45 or setup_type != "none":
        watch_status = "watch"
    else:
        watch_status = "avoid"

    # ── Risk label ────────────────────────────────────────────────────────────
    if risk_score < 35:
        risk_label = "Low"
    elif risk_score < 65:
        risk_label = "Medium"
    else:
        risk_label = "High"

    # ── Breakout Quality (0-100) ──────────────────────────────────────────────
    # A breakout-SPECIFIC score for the Breakout Watch bucket. Unlike
    # finalQuantScore it does NOT subtract the volatility/risk term (breakouts
    # NEED volatility to run, so penalizing it is misleading here) and it
    # weights the three things that actually define a real breakout:
    #   proximity to/above the 20-day high (40%), volume confirmation (30%),
    #   and trend alignment (20%). A mild momentum-health term (10%) keeps the
    #   score honest — it trims when RSI is in overbought/chase territory (>80).
    if rsi >= 80:
        mom_health = max(0.0, 100.0 - (rsi - 80.0) * 4.0)   # chase penalty above 80
    elif rsi >= 55:
        mom_health = 100.0
    else:
        mom_health = float(np.clip(momentum_score, 0, 100))
    breakout_quality = float(np.clip(round(
        0.40 * breakout_score
      + 0.30 * volume_score
      + 0.20 * trend_score
      + 0.10 * mom_health
    ), 0, 100))

    # ── Turnaround (swing-low reversal) — scored off ~1y dailies + regime gate ──
    turnaround_score = 0.0
    turnaround_stage = "none"
    turnaround_reasons: list[str] = []
    # Pre-filter: turnarounds only form from oversold/pullback — skip the
    # expensive structure detection for clearly-strong names (build-cost cut).
    if rsi <= 52:
        try:
            from engine import turnaround_detector
            _ta = turnaround_detector.score_turnaround(
                daily_long_df if daily_long_df is not None else daily_df,
                regime_type=regime_type,
            )
            if _ta:
                turnaround_score   = _ta["score"]
                turnaround_stage   = _ta["stage"]
                turnaround_reasons = _ta.get("reasons", [])
        except Exception as _te:
            logger.debug(f"[quant] {ticker} turnaround: {_te}")

    # ── Peak (swing-high / distribution top) — mirror of turnaround ──────────
    peak_score = 0.0
    peak_stage = "none"
    peak_reasons: list[str] = []
    # Pre-filter: tops form from overbought — but a top CONFIRMS as RSI falls, so evaluate a name
    # that is overbought NOW *or was recently* (≥70 in the last ~10 bars). Without the latch the
    # detector goes silent the moment a name starts rolling over — exactly when we'd want the short
    # (MSFT 466→370: "watch" at the RSI-73 top, then un-evaluated all the way down to the 377 break).
    if rsi >= 60 or _recent_max_rsi(closes) >= 70.0:
        try:
            from engine import peak_detector
            _pk = peak_detector.score_peak(
                daily_long_df if daily_long_df is not None else daily_df,
                regime_type=regime_type,
            )
            if _pk:
                peak_score   = _pk["score"]
                peak_stage   = _pk["stage"]
                peak_reasons = _pk.get("reasons", [])
        except Exception as _pe:
            logger.debug(f"[quant] {ticker} peak: {_pe}")

    # ── Cycle-context differentiators — only for staged turnaround/peak names ──
    cyclicality_score = None
    market_driven_pct = None
    driver_label      = None
    expected_pain_pct = None
    if turnaround_stage != "none" or peak_stage != "none":
        try:
            from engine import cycle_context
            _cx = cycle_context.compute(
                daily_long_df if daily_long_df is not None else daily_df, spy_long_df)
            cyclicality_score = _cx.get("cyclicalityScore")
            market_driven_pct = _cx.get("marketDrivenPct")
            driver_label      = _cx.get("driverLabel")
            expected_pain_pct = _cx.get("expectedPainPct")
        except Exception as _ce:
            logger.debug(f"[quant] {ticker} cycle_context: {_ce}")

    # ── Regime "Game Plan" category (watchlist tap-filter) ────────────────────
    regime_category, rs_vs_spy = _regime_category(
        closes, current, ma20, rsi, spy_long_df,
        peak_stage=peak_stage, setup_type=setup_type,
    )

    # ── Chaikin Money Flow (institutional accumulation/distribution) ──────────
    cmf_val, cmf_hist = _cmf(daily_df)
    cmf_state = _cmf_state(cmf_val)
    cmf_cross = _cmf_cross(cmf_hist)

    # ── ADX (trend strength) · TTM Squeeze (coil) · MFI (vol-weighted RSI) · ATR/ADR ──
    adx_val = _adx(daily_df)
    squeeze_state, squeeze_bias = _squeeze(daily_df)
    mfi_val = _mfi(daily_df)
    adr_pct, atr_stop_pct = _atr_adr(daily_df, current)

    return {
        "ticker":              ticker,
        "price":               round(current, 2),
        "cmf":                 round(cmf_val, 3) if cmf_val is not None else None,
        "cmfState":            cmf_state,
        "cmfCross":            cmf_cross,        # 'bullish' | 'bearish' | None (recent zero-line cross)
        "cmfHistory":          cmf_hist,
        "adx":                 adx_val,
        "adxState":            _adx_state(adx_val),      # trending | developing | choppy
        "squeezeState":        squeeze_state,            # on (coiling) | fired | off
        "squeezeBias":         squeeze_bias,             # bull | bear | flat
        "mfi":                 mfi_val,
        "mfiState":            _mfi_state(mfi_val),       # overbought | oversold | neutral
        "adrPct":              adr_pct,                   # avg daily range %
        "atrStopPct":          atr_stop_pct,             # 1.5×ATR stop width, % of price
        "dayChangePct":        round(day_change_pct, 2),   # today's move vs prior close (direction)
        "regimeCategory":      regime_category,            # rs_pullback | rs_leader | knife | neutral
        "rsVsSpy":             rs_vs_spy,                  # 20d return vs SPY (pts)
        "vwap":                vwap,
        "rsi":                 round(rsi, 1),
        "relativeVolume":      round(rel_vol, 2),
        # Component scores (0-100)
        "trendScore":          round(trend_score, 1),
        "momentumScore":       round(momentum_score, 1),
        "macdHist":            round(macd_hist, 4),       # latest MACD histogram bar
        "macdRising":          macd_rising,               # histogram building 2+ bars
        "macdHistSeries":      macd_hist_series,          # last 6 bars (for a sparkline)
        "momentumAccelerating": momentum_accelerating,    # strong trend + MACD still building
        "volumeScore":         round(volume_score, 1),
        "breakoutScore":       round(breakout_score, 1),
        "breakoutLevel":       round(high_20, 2),        # 20-day high being tested
        "distToBreakoutPct":   round(dist_to_high_pct, 2),  # negative = below the high
        "ma20":                round(float(ma20), 2),     # 20-day avg — the rising trend support / "buy-the-dip" anchor
        "atrPct":              round(float(atr_pct), 2),   # daily range as % of price (for a dip-zone band)
        "wk52High":            wk52_high,                  # 52-week high / low / position (watchlist range bar)
        "wk52Low":             wk52_low,
        "wk52Pct":             wk52_pct,                   # 0 = at 52w low, 100 = at 52w high
        "breakoutQuality":     breakout_quality,         # breakout-specific 0-100 (no vol penalty)
        "breakdownScore":      round(breakdown_score, 1),
        "breakdownLevel":      round(low_20, 2),          # 20-day low being tested
        "distToBelowPct":      round(dist_to_low_pct, 2), # positive = above the low
        "meanReversionScore":  round(mean_reversion_score, 1),
        "turnaroundScore":     round(turnaround_score, 1),
        "turnaroundStage":     turnaround_stage,
        "turnaroundReasons":   turnaround_reasons,
        "peakScore":           round(peak_score, 1),
        "peakStage":           peak_stage,
        "peakReasons":         peak_reasons,
        "cyclicalityScore":    cyclicality_score,
        "marketDrivenPct":     market_driven_pct,
        "driverLabel":         driver_label,
        "expectedPainPct":     expected_pain_pct,
        "riskScore":           round(risk_score, 1),
        "finalQuantScore":     round(final_score, 1),
        # Qualitative outputs
        "setupType":           setup_type,
        "setupReason":         setup_reason,
        "watchStatus":         watch_status,
        "riskLevel":           risk_label,
        # Confidence: maps final score to "low/medium/high" label
        "confidence":          "high" if final_score >= 70 else ("medium" if final_score >= 50 else "low"),
    }


def _classify_setup(
    price: float, ma20: float, rsi: float, rel_vol: float,
    breakout_score: float, mean_rev_score: float, vwap: Optional[float],
    breakdown_score: float = 0.0, trend_score: float = 50.0,
) -> tuple[str, str]:
    """Return (setup_type, plain-English reason)."""

    # Momentum: strong RSI + above MA. Volume requirement loosened so
    # after-hours / pre-market still surfaces top movers from RTH session.
    if rsi >= 55 and price > ma20 and rel_vol >= 1.1:
        return "momentum", f"RSI {rsi:.0f} — momentum above 20-day average"

    # Breakout candidate: near 20-day high. Volume requirement loosened
    # for the same reason as momentum.
    if breakout_score >= 60 and rel_vol >= 1.0:
        return "breakout", "Price approaching 20-day high — breakout candidate"

    # Breakdown (risk/avoid): breaking BELOW the 20-day low. Checked before the
    # oversold-bounce branch so a name making fresh lows reads as a breakdown to
    # avoid, not a dip to buy.
    if breakdown_score >= 60 and rel_vol >= 1.0:
        return "breakdown", "Price breaking below its 20-day low — breakdown (consider avoiding / exiting longs)"

    # VWAP reclaim: price just crossed back above VWAP
    if vwap and price > vwap and price < vwap * 1.005:
        return "vwap_reclaim", "Price reclaiming VWAP — intraday structure improving"

    # Pullback: healthy stock in an UPTREND pulling back to its 20-day average.
    # RSI band is 40–62 (not <50): a shallow pullback in a strong trend keeps RSI
    # mid-range — the old rsi<50 gate only caught DEEP pullbacks (which usually
    # mean the trend is breaking) and missed the good buy-the-dip setup (HOOD
    # held its rising 20-EMA at ~93 with RSI 55). trend_score>=50 keeps this from
    # mislabeling a bear-flag at a FALLING MA as a buyable pullback (the MSFT trap).
    if (price > ma20 * 0.97 and price < ma20 * 1.01
            and 40 <= rsi <= 62 and trend_score >= 50):
        return "pullback", f"Pulling back to 20-day average in an uptrend — possible support zone at {ma20:.2f}"

    # Oversold bounce: RSI < 35, well below MA
    if rsi <= 35 and price < ma20 * 0.97:
        return "oversold_bounce", f"RSI {rsi:.0f} — oversold setup detected, watching for stabilisation"

    return "none", "No specific setup detected — monitoring"


def _regime_description(label: str) -> str:
    descriptions = {
        "Bullish":  "Broad market trending higher. Momentum setups historically perform better.",
        "Bearish":  "Market under pressure. Risk management elevated. Setups may fail faster.",
        "Choppy":   "No clear direction. Breakout setups may whipsaw. Patience recommended.",
        "Volatile": "High volatility detected. Wider price swings expected. Sizing carefully.",
        "Risk-Off": "Markets in risk-off mode. Defensive posture. Setups carry higher uncertainty.",
        "Neutral":  "Mixed signals. No strong directional bias detected.",
    }
    return descriptions.get(label, "Market data being analysed.")


def _regime_color(label: str) -> str:
    colors = {
        "Bullish": "#22C55E", "Bearish": "#EF4444",
        "Choppy": "#F59E0B",  "Volatile": "#F59E0B",
        "Risk-Off": "#6366F1", "Neutral": "#9CA3AF",
    }
    return colors.get(label, "#9CA3AF")
