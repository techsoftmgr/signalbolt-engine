"""
Community Insights — the analysis layer that turns raw social trending data
into a *read*.

This is SignalBolt's differentiator. StockTwits / Reddit (and every app that
mirrors them) just tell you what's loud — they ARE the crowd. SignalBolt owns
a price feed, a quant engine, and a manipulation detector, so we can audit the
crowd instead of amplifying it: for every loud name we answer the question
those apps never do — *should you believe it?*

Per-ticker enrichment attached to each trending row:
  • buzzVsPrice  — is the chatter confirmed by the tape? (price 1d/5d change)
  • goingViral   — mention z-score vs the ticker's OWN baseline (needs history)
  • velocity     — 24h mention change %
  • sparkline    — recent mention history (from social_snapshots)
  • engine       — active signal + quant score + manipulation flag
  • catalyst     — latest news headline ("why is it trending"), top names only
  • verdict      — combined label: Real momentum / Hype fading / Pump risk /
                   Crowd trap / Under the radar / Mixed

Tab-level:
  • community_pulse(...)  — rule-based narrative digest
  • whats_changed(...)    — new / climbers / fallers / first-time vs ~24h ago
  • track_record(...)     — did trending names beat SPY? (from snapshots)

Heavy work (Alpaca bars, manipulation, news) is bounded to the trending set
and cached, so the public endpoint stays cheap.
"""

from __future__ import annotations

import logging
import statistics
from datetime import datetime, timezone, timedelta
from typing import Optional

from engine import cache

logger = logging.getLogger("signalbolt.social_insights")

# ── Tunables ─────────────────────────────────────────────────────────────────
_ENRICHED_CACHE_KEY = "social_insights:enriched:v1"
_ENRICHED_TTL       = 600          # 10 min (matches the upstream trending cache)
_PULSE_CACHE_KEY    = "social_insights:pulse:v1"
_PULSE_TTL          = 600
_CHANGED_CACHE_KEY  = "social_insights:changed:v1"
_CHANGED_TTL        = 900          # 15 min
_TRACK_CACHE_KEY    = "social_insights:track:v1"
_TRACK_TTL          = 6 * 3600     # 6 h

_CATALYST_TOP_N  = 10              # fetch "why trending" news for the top N only
_SPARK_POINTS    = 24             # sparkline resolution
_VIRAL_Z         = 2.0            # z-score threshold for "going viral"
_VIRAL_MIN_ABS   = 40             # ignore tiny-mention names regardless of z
_BUZZ_UP_PCT     = 20.0          # mentions change considered "rising"
_BUZZ_HOT_PCT    = 100.0         # mentions change considered "spiking"
_PRICE_UP_PCT    = 1.0           # 1d price move considered "up"
_PRICE_DN_PCT    = -1.0          # 1d price move considered "down"
_VOL_CONFIRM     = 1.5           # rel-volume to confirm a move is real (not drift)

# Verdict definitions: key → (label, tone, blurb). tone drives UI color.
#   good = confirmed/healthy, warn = caution, bad = avoid, info = neutral-early
VERDICTS = {
    "REAL_MOMENTUM": ("Real momentum", "good",
                      "Buzz is confirmed by price — the crowd is right so far."),
    "HYPE_FADING":   ("Hype — no follow-through", "warn",
                      "Loud on social but the tape isn't confirming. Classic trap setup."),
    "PUMP_RISK":     ("Pump risk", "bad",
                      "Spiking chatter with manipulation signatures. Treat with caution."),
    "CROWD_TRAP":    ("Crowd trap", "bad",
                      "Euphoric sentiment while price rolls over — looks like distribution."),
    "UNDER_RADAR":   ("Under the radar", "info",
                      "Price moving with little chatter — ahead of the crowd."),
    "MIXED":         ("Mixed", "neutral",
                      "No clear edge from buzz, price, or our engine right now."),
}


# ──────────────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────────────

def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_ts(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _f(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Data loaders (bounded + defensive)
# ──────────────────────────────────────────────────────────────────────────────

def _load_snapshots(sb, tickers: list[str], days: int = 10) -> dict[str, list[dict]]:
    """Return {ticker: [snapshot rows asc by captured_at]} for the last `days`."""
    if not tickers:
        return {}
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    try:
        res = (
            sb.table("social_snapshots")
            .select("ticker,captured_at,rank,reddit_mentions,reddit_sentiment,price")
            .in_("ticker", tickers)
            .gte("captured_at", since)
            .order("captured_at", desc=False)
            .limit(8000)
            .execute()
        )
        out: dict[str, list[dict]] = {}
        for row in (res.data or []):
            out.setdefault(row["ticker"], []).append(row)
        return out
    except Exception as e:
        # Table may not exist yet (migration not run) — degrade silently.
        logger.info(f"[insights] snapshot load skipped: {e}")
        return {}


def _daily_bars(tickers: list[str], days: int = 30) -> dict:
    """{ticker: DataFrame(open,high,low,close,volume)} via Alpaca; {} on failure."""
    try:
        from engine import alpaca_client
        return alpaca_client.get_multi_bars(tickers, "1Day", days) or {}
    except Exception as e:
        logger.info(f"[insights] daily bars failed: {e}")
        return {}


def _active_signals(sb) -> dict[str, dict]:
    """{ticker: {direction, confidence, strategy}} for ACTIVE engine signals."""
    try:
        res = (
            sb.table("signals")
            .select("ticker,direction,confidence_score,strategy_type")
            .eq("status", "active")
            .execute()
        )
        out: dict[str, dict] = {}
        for r in (res.data or []):
            t = (r.get("ticker") or "").upper()
            if t and t not in out:   # keep the first (most recent enough)
                out[t] = {
                    "direction":  r.get("direction"),
                    "confidence": r.get("confidence_score"),
                    "strategy":   r.get("strategy_type"),
                }
        return out
    except Exception as e:
        logger.info(f"[insights] active signals load failed: {e}")
        return {}


def _quant_map() -> dict[str, dict]:
    """{ticker: quant row} from the (cached) quant dashboard's allScored bucket."""
    try:
        from engine import quant_score_service
        dash = quant_score_service.get_quant_dashboard() or {}
        rows = dash.get("allScored") or []
        return {(r.get("ticker") or "").upper(): r for r in rows if r.get("ticker")}
    except Exception as e:
        logger.info(f"[insights] quant map failed: {e}")
        return {}


def _latest_news(ticker: str) -> Optional[dict]:
    """Latest headline behind the buzz, or None."""
    try:
        from engine import alpaca_client
        items = alpaca_client.get_news(ticker, 1) or []
        if not items:
            return None
        n = items[0]
        return {
            "headline":  n.get("headline"),
            "source":    n.get("source"),
            "url":       n.get("url"),
            "createdAt": n.get("created_at"),
        }
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Per-ticker computations
# ──────────────────────────────────────────────────────────────────────────────

def _price_changes(df) -> dict:
    """
    1d/5d % change + relative volume (latest daily bar vol vs its 20-day avg).
    NOTE: mid-session the latest daily bar's volume is partial, so relVolume is
    understated during RTH (conservative — fewer false "confirmed" reads).
    """
    out = {"price1dPct": None, "price5dPct": None, "lastClose": None, "relVolume": None}
    try:
        if df is None or len(df) < 2:
            return out
        d = df.sort_index()
        closes = d["close"].dropna()
        if len(closes) < 2:
            return out
        last = float(closes.iloc[-1])
        out["lastClose"] = round(last, 2)
        out["price1dPct"] = round((last / float(closes.iloc[-2]) - 1) * 100, 2)
        if len(closes) >= 6:
            out["price5dPct"] = round((last / float(closes.iloc[-6]) - 1) * 100, 2)
        if "volume" in d.columns:
            vols = d["volume"].dropna()
            if len(vols) >= 6:
                recent = float(vols.iloc[-1])
                base = vols.iloc[-21:-1] if len(vols) > 21 else vols.iloc[:-1]
                avg = float(base.mean()) if len(base) else 0.0
                if avg > 0:
                    out["relVolume"] = round(recent / avg, 2)
    except Exception:
        pass
    return out


def _going_viral(snaps: list[dict], current_mentions: Optional[int]) -> dict:
    """
    z-score of the latest mention count vs this ticker's OWN historical baseline.
    Needs enough history; degrades to {viral:False, z:None} otherwise.
    """
    out = {"viral": False, "zScore": None, "baselineAvg": None}
    try:
        hist = [int(s["reddit_mentions"]) for s in snaps
                if s.get("reddit_mentions") is not None]
        cur = current_mentions if current_mentions is not None else (hist[-1] if hist else None)
        if cur is None or len(hist) < 5:
            return out
        base = hist[:-1] if len(hist) > 5 else hist   # exclude latest from baseline
        mean = statistics.mean(base)
        sd = statistics.pstdev(base)
        out["baselineAvg"] = round(mean, 1)
        if sd > 0:
            z = (cur - mean) / sd
            out["zScore"] = round(z, 2)
            out["viral"] = bool(z >= _VIRAL_Z and cur >= max(_VIRAL_MIN_ABS, mean * 1.5))
    except Exception:
        pass
    return out


def _sparkline(snaps: list[dict], points: int = _SPARK_POINTS) -> list[int]:
    """Mention counts over time, downsampled to ~`points` values."""
    vals = [int(s["reddit_mentions"]) for s in snaps if s.get("reddit_mentions") is not None]
    if len(vals) <= points:
        return vals
    # even downsample
    step = len(vals) / points
    return [vals[min(len(vals) - 1, int(i * step))] for i in range(points)]


def _manipulation(df, ticker: str, direction: str) -> Optional[dict]:
    """Run the manipulation detector on daily bars. None if not enough data."""
    try:
        if df is None or len(df) < 20:
            return None
        from engine import manipulation_detector
        m = manipulation_detector.detect(df.sort_index(), ticker, direction)
        flagged = bool(
            manipulation_detector.is_blocking(m)
            or m.get("momentum_ignition")
            or m.get("stop_raid_risk")
        )
        return {
            "flagged":  flagged,
            "flags":    m.get("flags") or [],
            "score":    m.get("score"),
            "stopRaid": bool(m.get("stop_raid_risk")),
            "ignition": bool(m.get("momentum_ignition")),
        }
    except Exception:
        return None


def _engine_take(ticker: str, sentiment: Optional[float],
                 sig_map: dict, quant_map: dict, manip: Optional[dict]) -> dict:
    """Combine active signal + quant score + manipulation into an 'engine read'."""
    sig = sig_map.get(ticker)
    q = quant_map.get(ticker)
    quant_score = None
    setup = None
    if q:
        quant_score = q.get("finalQuantScore")
        setup = q.get("setupType")

    crowd_long = sentiment is None or sentiment >= 0.5
    confirmed = False
    conflict = False
    if sig and sig.get("direction"):
        sig_long = str(sig["direction"]).upper().startswith("L")
        confirmed = (sig_long == crowd_long)
        conflict = not confirmed
    if quant_score is not None:
        if quant_score >= 70:
            confirmed = True
        elif quant_score < 35:
            conflict = True or conflict

    return {
        "hasSignal":   bool(sig),
        "signalDir":   sig.get("direction") if sig else None,
        "signalConf":  sig.get("confidence") if sig else None,
        "quantScore":  quant_score,
        "setupType":   setup,
        "manipulation": manip,
        "confirmed":   bool(confirmed),
        "conflict":    bool(conflict),
    }


def _verdict(*, mentions_chg: Optional[float], price1d: Optional[float],
             price5d: Optional[float], sentiment: Optional[float],
             viral: bool, engine: dict, rel_vol: Optional[float] = None) -> dict:
    """The combined read. Order matters — strongest warning wins."""
    manip = engine.get("manipulation") or {}
    p1 = price1d if price1d is not None else 0.0
    p5 = price5d if price5d is not None else 0.0
    mc = mentions_chg

    buzz_up       = (mc is not None and mc >= _BUZZ_UP_PCT) or viral
    buzz_hot      = (mc is not None and mc >= _BUZZ_HOT_PCT) or viral
    price_up      = p1 >= _PRICE_UP_PCT or (price5d is not None and p5 >= 3.0)
    price_strong  = p1 >= 5.0 or (price5d is not None and p5 >= 10.0)
    price_dn      = p1 <= _PRICE_DN_PCT
    vol_confirmed = rel_vol is not None and rel_vol >= _VOL_CONFIRM

    if manip.get("flagged") and buzz_hot:
        key = "PUMP_RISK"
    elif sentiment is not None and sentiment >= 0.80 and price_dn and p5 >= 2.0:
        key = "CROWD_TRAP"
    elif (buzz_up and price_up) or (price_strong and (mc is None or mc >= 0)):
        # buzz + price aligned, OR a strong price move that chatter isn't fighting
        key = "REAL_MOMENTUM"
    elif buzz_up and not price_up:
        key = "HYPE_FADING"
    elif price_up and not buzz_up:
        # only "ahead of the crowd" if REAL volume backs the move; a quiet rise
        # on below-average volume is just drift, not accumulation → Mixed.
        key = "UNDER_RADAR" if vol_confirmed else "MIXED"
    else:
        key = "MIXED"

    label, tone, blurb = VERDICTS[key]
    # Show the evidence instead of asserting a cause.
    if key == "UNDER_RADAR" and rel_vol is not None:
        blurb = (f"Up on {rel_vol:g}× normal volume while chatter cools — "
                 f"moving before the crowd.")
    # Upgrade the blurb when our own engine agrees with a momentum call.
    if key == "REAL_MOMENTUM" and engine.get("confirmed"):
        blurb = "Buzz, price, AND our engine all line up here."
    if key in ("REAL_MOMENTUM", "MIXED") and engine.get("conflict"):
        blurb = "Crowd's loud, but our engine doesn't see the setup."
    return {"key": key, "label": label, "tone": tone, "blurb": blurb}


# ──────────────────────────────────────────────────────────────────────────────
# Public: enriched trending
# ──────────────────────────────────────────────────────────────────────────────

def get_enriched_trending(sb, limit: int = 30, force: bool = False) -> dict:
    """
    Base trending feed + per-ticker insight layers. Cached 10 min.
    Returns {trending:[...enriched...], last_updated, sources_used, ...}.
    """
    if not force:
        try:
            cached = cache.kv.get_json(_ENRICHED_CACHE_KEY)
            if cached:
                return cached
        except Exception:
            pass

    from engine import social_sentiment
    base = social_sentiment.get_trending(limit=limit, force=force) or {}
    rows = base.get("trending") or []
    if not rows:
        return base

    tickers = [r.get("ticker") for r in rows if r.get("ticker")]
    bars      = _daily_bars(tickers)
    snaps     = _load_snapshots(sb, tickers, days=10)
    sig_map   = _active_signals(sb)
    quant_map = _quant_map()

    for i, r in enumerate(rows):
        t = (r.get("ticker") or "").upper()
        df = bars.get(t)
        sentiment = _f(r.get("reddit_sentiment"))
        mentions_chg = _f(r.get("reddit_change_pct"))

        pc = _price_changes(df)
        viral = _going_viral(snaps.get(t, []), r.get("reddit_mentions"))
        spark = _sparkline(snaps.get(t, []))
        direction = "LONG" if (sentiment is None or sentiment >= 0.5) else "SHORT"
        manip = _manipulation(df, t, direction)
        engine = _engine_take(t, sentiment, sig_map, quant_map, manip)
        verdict = _verdict(
            mentions_chg=mentions_chg,
            price1d=pc["price1dPct"], price5d=pc["price5dPct"],
            sentiment=sentiment, viral=viral["viral"], engine=engine,
            rel_vol=pc.get("relVolume"),
        )

        r["price1dPct"]  = pc["price1dPct"]
        r["price5dPct"]  = pc["price5dPct"]
        r["lastClose"]   = pc["lastClose"]
        r["relVolume"]   = pc.get("relVolume")
        r["goingViral"]  = viral["viral"]
        r["viralZ"]      = viral["zScore"]
        r["baselineAvg"] = viral["baselineAvg"]
        r["sparkline"]   = spark
        r["engine"]      = engine
        r["verdict"]     = verdict
        r["catalyst"]    = _latest_news(t) if i < _CATALYST_TOP_N else None

    payload = {
        "trending":     rows,
        "last_updated": base.get("last_updated") or _iso_now(),
        "sources_used": base.get("sources_used") or [],
        "generated_at": _iso_now(),
    }
    try:
        cache.kv.set_json(_ENRICHED_CACHE_KEY, payload, _ENRICHED_TTL)
    except Exception:
        pass
    return payload


# ──────────────────────────────────────────────────────────────────────────────
# Public: what changed (vs ~24h ago)
# ──────────────────────────────────────────────────────────────────────────────

def whats_changed(sb, force: bool = False) -> dict:
    """New entrants / climbers / fallers / first-time-trending vs ~24h ago."""
    if not force:
        try:
            cached = cache.kv.get_json(_CHANGED_CACHE_KEY)
            if cached:
                return cached
        except Exception:
            pass

    out = {"newToday": [], "goingViral": [], "climbers": [],
           "fallers": [], "droppedOff": [], "generated_at": _iso_now()}
    try:
        since = (datetime.now(timezone.utc) - timedelta(hours=40)).isoformat()
        res = (
            sb.table("social_snapshots")
            .select("ticker,captured_at,rank,reddit_mentions")
            .gte("captured_at", since)
            .order("captured_at", desc=False)
            .limit(12000)
            .execute()
        )
        data = res.data or []
        if not data:
            return out

        by_ticker: dict[str, list[dict]] = {}
        for row in data:
            by_ticker.setdefault(row["ticker"], []).append(row)

        now = datetime.now(timezone.utc)
        latest_rank: dict[str, int] = {}
        rank_24h: dict[str, int] = {}
        first_seen: dict[str, datetime] = {}
        for t, rows in by_ticker.items():
            rows.sort(key=lambda r: r.get("captured_at") or "")
            latest_rank[t] = rows[-1].get("rank") or 999
            ts0 = _parse_ts(rows[0].get("captured_at"))
            if ts0:
                first_seen[t] = ts0
            # closest snapshot to ~24h ago
            target = now - timedelta(hours=24)
            best = None
            best_dt = None
            for r in rows:
                dt = _parse_ts(r.get("captured_at"))
                if not dt:
                    continue
                if best is None or abs((dt - target).total_seconds()) < abs((best_dt - target).total_seconds()):
                    best, best_dt = r, dt
            if best is not None and best_dt is not None and abs((best_dt - target).total_seconds()) <= 8 * 3600:
                rank_24h[t] = best.get("rank") or 999

        for t, lr in sorted(latest_rank.items(), key=lambda kv: kv[1]):
            if lr > 30:
                continue
            pr = rank_24h.get(t)
            fs = first_seen.get(t)
            is_new = fs is not None and (now - fs).total_seconds() <= 26 * 3600
            if is_new and lr <= 25:
                out["newToday"].append({"ticker": t, "rank": lr})
            elif pr is not None:
                delta = pr - lr            # +ve = climbed
                if delta >= 5:
                    out["climbers"].append({"ticker": t, "rank": lr, "from": pr, "delta": delta})
                elif delta <= -5:
                    out["fallers"].append({"ticker": t, "rank": lr, "from": pr, "delta": delta})

        # dropped off: was top-20 ~24h ago, now absent from latest top-30
        for t, pr in rank_24h.items():
            if pr <= 20 and latest_rank.get(t, 999) > 30:
                out["droppedOff"].append({"ticker": t, "from": pr})

        out["newToday"]   = out["newToday"][:8]
        out["climbers"]   = sorted(out["climbers"], key=lambda x: -x["delta"])[:8]
        out["fallers"]    = sorted(out["fallers"], key=lambda x: x["delta"])[:8]
        out["droppedOff"] = out["droppedOff"][:8]
    except Exception as e:
        logger.info(f"[insights] whats_changed skipped: {e}")

    try:
        cache.kv.set_json(_CHANGED_CACHE_KEY, out, _CHANGED_TTL)
    except Exception:
        pass
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Public: community pulse (rule-based narrative)
# ──────────────────────────────────────────────────────────────────────────────

def community_pulse(sb, force: bool = False) -> dict:
    if not force:
        try:
            cached = cache.kv.get_json(_PULSE_CACHE_KEY)
            if cached:
                return cached
        except Exception:
            pass

    enriched = get_enriched_trending(sb, limit=30)
    rows = enriched.get("trending") or []
    changed = whats_changed(sb)

    bullets: list[str] = []
    headline = "Quiet on the socials right now."
    if rows:
        top = [r["ticker"] for r in rows[:3] if r.get("ticker")]
        headline = "Retail's loudest: " + ", ".join(top) + "."

        counts: dict[str, int] = {}
        for r in rows[:15]:
            k = (r.get("verdict") or {}).get("key")
            if k:
                counts[k] = counts.get(k, 0) + 1
        if counts.get("REAL_MOMENTUM"):
            bullets.append(f"\U0001F680 {counts['REAL_MOMENTUM']} name(s) have buzz confirmed by price.")
        if counts.get("HYPE_FADING"):
            bullets.append(f"⚠️ {counts['HYPE_FADING']} look like hype with no follow-through.")
        if counts.get("PUMP_RISK"):
            bullets.append(f"\U0001F6A9 {counts['PUMP_RISK']} flagged for pump / manipulation risk.")
        if counts.get("UNDER_RADAR"):
            bullets.append(f"\U0001F440 {counts['UNDER_RADAR']} moving under the radar (price up, low chatter).")

        viral = [r["ticker"] for r in rows if r.get("goingViral")][:4]
        if viral:
            bullets.append("Going viral: " + ", ".join(viral) + ".")

        confirmed = sum(1 for r in rows[:10] if (r.get("engine") or {}).get("confirmed"))
        if confirmed:
            bullets.append(f"Our engine agrees with {confirmed} of the top 10.")

    if changed.get("newToday"):
        bullets.append("New today: " + ", ".join(c["ticker"] for c in changed["newToday"][:4]) + ".")

    out = {"headline": headline, "bullets": bullets, "generated_at": _iso_now()}
    try:
        cache.kv.set_json(_PULSE_CACHE_KEY, out, _PULSE_TTL)
    except Exception:
        pass
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Public: trending → returns track record
# ──────────────────────────────────────────────────────────────────────────────

def track_record(sb, days: int = 30, horizon_days: int = 5, force: bool = False) -> dict:
    """
    Did trending names actually pay? For each matured snapshot we compare the
    price at capture vs `horizon_days` later, and net out SPY over the same
    window. Aggregated by rank bucket. Cached 6h.
    """
    if not force:
        try:
            cached = cache.kv.get_json(_TRACK_CACHE_KEY)
            if cached:
                return cached
        except Exception:
            pass

    out = {"ready": False, "message": "Not enough matured data yet — building history.",
           "buckets": [], "horizonDays": horizon_days, "windowDays": days,
           "generated_at": _iso_now()}
    try:
        now = datetime.now(timezone.utc)
        lo = (now - timedelta(days=days + horizon_days)).isoformat()
        hi = (now - timedelta(days=horizon_days)).isoformat()   # matured only
        res = (
            sb.table("social_snapshots")
            .select("ticker,captured_at,rank,price")
            .gte("captured_at", lo)
            .lte("captured_at", hi)
            .order("captured_at", desc=False)
            .limit(20000)
            .execute()
        )
        data = [r for r in (res.data or []) if r.get("price") and r.get("rank")]
        if len(data) < 20:
            cache.kv.set_json(_TRACK_CACHE_KEY, out, _TRACK_TTL)
            return out

        # Dedupe to one observation per ticker per calendar day (first of day).
        seen: set[tuple] = set()
        obs: list[dict] = []
        tickers: set[str] = set()
        for r in data:
            dt = _parse_ts(r.get("captured_at"))
            if not dt:
                continue
            key = (r["ticker"], dt.date())
            if key in seen:
                continue
            seen.add(key)
            obs.append({"ticker": r["ticker"], "dt": dt,
                        "rank": int(r["rank"]), "price": float(r["price"])})
            tickers.add(r["ticker"])

        bars = _daily_bars(sorted(tickers) + ["SPY"], days=days + horizon_days + 8)
        spy_df = bars.get("SPY")

        def _fwd_return(df, entry_dt: datetime) -> Optional[float]:
            if df is None or len(df) < 2:
                return None
            d = df.sort_index()
            try:
                idx = d.index
                # first bar on/after entry date
                pos = idx.searchsorted(entry_dt)
                if pos >= len(d):
                    return None
                fwd = pos + horizon_days
                if fwd >= len(d):
                    return None
                p0 = float(d["close"].iloc[pos])
                p1 = float(d["close"].iloc[fwd])
                if p0 <= 0:
                    return None
                return (p1 / p0 - 1) * 100
            except Exception:
                return None

        buckets_def = [("#1", lambda r: r == 1),
                       ("Top 3", lambda r: r <= 3),
                       ("Top 10", lambda r: r <= 10),
                       ("11–30", lambda r: 11 <= r <= 30)]
        agg: dict[str, list] = {b[0]: [] for b in buckets_def}

        for o in obs:
            ret = _fwd_return(bars.get(o["ticker"]), o["dt"])
            if ret is None:
                continue
            spy_ret = _fwd_return(spy_df, o["dt"]) if spy_df is not None else None
            edge = (ret - spy_ret) if spy_ret is not None else None
            for name, pred in buckets_def:
                if pred(o["rank"]):
                    agg[name].append((ret, edge))

        results = []
        for name, pred in buckets_def:
            vals = agg[name]
            if not vals:
                continue
            rets = [v[0] for v in vals]
            edges = [v[1] for v in vals if v[1] is not None]
            results.append({
                "bucket":     name,
                "n":          len(rets),
                "avgReturn":  round(statistics.mean(rets), 2),
                "winRate":    round(sum(1 for x in rets if x > 0) / len(rets) * 100),
                "edgeVsSpy":  round(statistics.mean(edges), 2) if edges else None,
            })

        if results:
            out = {"ready": True, "buckets": results, "horizonDays": horizon_days,
                   "windowDays": days, "sampleTickers": len(tickers),
                   "generated_at": _iso_now()}
    except Exception as e:
        logger.info(f"[insights] track_record skipped: {e}")

    try:
        cache.kv.set_json(_TRACK_CACHE_KEY, out, _TRACK_TTL)
    except Exception:
        pass
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Public: spike detection (for push alerts)
# ──────────────────────────────────────────────────────────────────────────────

def detect_spikes(sb, tickers: list[str],
                  current: dict[str, Optional[int]],
                  min_z: float = 2.5) -> list[dict]:
    """
    Return tickers whose latest mention count is an abnormal spike vs their OWN
    baseline (z-score >= min_z). Uses a stricter threshold than the on-screen
    "going viral" tag (2.0) so push alerts only fire on genuine outliers.
    `current` = {ticker: latest_mentions}.
    """
    if not tickers:
        return []
    snaps = _load_snapshots(sb, tickers, days=10)
    out: list[dict] = []
    for t in tickers:
        v = _going_viral(snaps.get(t, []), current.get(t))
        z = v.get("zScore")
        if v.get("viral") and z is not None and z >= min_z:
            out.append({"ticker": t, "z": z, "mentions": current.get(t)})
    out.sort(key=lambda x: -(x["z"] or 0))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Public: single-ticker detail (tap-through)
# ──────────────────────────────────────────────────────────────────────────────

def ticker_detail(sb, ticker: str) -> dict:
    """Detail for the tap-through screen: mention history + news + price."""
    ticker = (ticker or "").upper().strip()
    snaps = _load_snapshots(sb, [ticker], days=14).get(ticker, [])
    history = [{
        "t":        s.get("captured_at"),
        "mentions": s.get("reddit_mentions"),
        "rank":     s.get("rank"),
        "price":    s.get("price"),
    } for s in snaps]

    news = []
    try:
        from engine import alpaca_client
        for n in (alpaca_client.get_news(ticker, 8) or []):
            news.append({
                "headline":  n.get("headline"),
                "source":    n.get("source"),
                "url":       n.get("url"),
                "createdAt": n.get("created_at"),
                "summary":   n.get("summary"),
            })
    except Exception:
        pass

    return {
        "ticker":    ticker,
        "history":   history,
        "sparkline": _sparkline(snaps, points=48),
        "news":      news,
        "generated_at": _iso_now(),
    }
