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
        daily_bars    = get_multi_bars(tickers, timeframe="1Day", days=25)
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
            "allScored":      sorted_by_quant[:20],
        }

    except Exception as e:
        logger.error(f"[quant] _build_dashboard failed: {e}")
        return {}


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

    # ── Volume Score (0-100) ──────────────────────────────────────────────────
    avg_vol    = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(np.mean(volumes))
    today_vol  = 0.0
    if intraday_df is not None and not intraday_df.empty:
        today = datetime.now(timezone.utc).date()
        try:
            today_mask = intraday_df.index.date == today
            today_vol  = _safe(intraday_df[today_mask]["volume"].sum())
        except Exception:
            today_vol  = _safe(intraday_df["volume"].tail(78).sum())

    if avg_vol > 0 and today_vol > 0:
        now_utc = datetime.now(timezone.utc)
        elapsed  = max(1, now_utc.hour * 60 + now_utc.minute - (13 * 60 + 30))
        proj_vol = today_vol / max(0.05, elapsed / 390)
        rel_vol  = proj_vol / avg_vol
    else:
        rel_vol = 1.0

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
            today = datetime.now(timezone.utc).date()
            t_df  = intraday_df[intraday_df.index.date == today]
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
        breakdown_score,
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
    # Pre-filter: tops only form from overbought — skip the rest.
    if rsi >= 60:
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

    return {
        "ticker":              ticker,
        "price":               round(current, 2),
        "dayChangePct":        round(day_change_pct, 2),   # today's move vs prior close (direction)
        "vwap":                vwap,
        "rsi":                 round(rsi, 1),
        "relativeVolume":      round(rel_vol, 2),
        # Component scores (0-100)
        "trendScore":          round(trend_score, 1),
        "momentumScore":       round(momentum_score, 1),
        "volumeScore":         round(volume_score, 1),
        "breakoutScore":       round(breakout_score, 1),
        "breakoutLevel":       round(high_20, 2),        # 20-day high being tested
        "distToBreakoutPct":   round(dist_to_high_pct, 2),  # negative = below the high
        "ma20":                round(float(ma20), 2),     # 20-day avg — the rising trend support / "buy-the-dip" anchor
        "atrPct":              round(float(atr_pct), 2),   # daily range as % of price (for a dip-zone band)
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
    breakdown_score: float = 0.0,
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

    # Pullback: healthy stock pulling back to MA
    if price > ma20 * 0.97 and price < ma20 * 1.01 and rsi < 50:
        return "pullback", f"Pulling back to 20-day average — possible support zone at {ma20:.2f}"

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
