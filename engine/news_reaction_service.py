"""
News Reaction Service — fetches Alpaca news, scores sentiment, and measures
price/volume reaction after each headline.

Pipeline per news item:
  1. Fetch headlines via Alpaca News API (batch for active tickers)
  2. Compute simple keyword sentiment (fast, free)
  3. Optionally run Claude AI summary for urgency/context (Pro+ tier)
  4. Get price 1-min, 5-min, 15-min after publish time via Alpaca bars
  5. Calculate newsReactionScore
  6. Flag items that touch active signals

Cache: 5-minute TTL (news is event-driven but we don't want per-request
Alpaca hammering).
"""

import logging
import time
import os
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger("signalbolt.news")

# ── Cache ─────────────────────────────────────────────────────────────────────
_cache: list[dict] = []
_cache_ts: float   = 0.0
_CACHE_TTL: int    = int(os.environ.get("NEWS_CACHE_TTL", "300"))  # 5 min

# Bullish / bearish keyword sets for fast sentiment
_BULLISH_WORDS = {
    "beats", "beat", "raises", "raised", "upgrade", "upgraded",
    "outperforms", "strong", "surge", "surges", "jumps", "rallies",
    "record", "growth", "accelerates", "partnership", "wins", "awarded",
    "positive", "approval", "approves", "breakthrough", "buyback",
}
_BEARISH_WORDS = {
    "misses", "miss", "missed", "lowers", "lowered", "downgrade", "downgraded",
    "disappoints", "weak", "plunges", "drops", "falls", "warns",
    "investigation", "recall", "loss", "losses", "cut", "cuts",
    "delay", "delayed", "lawsuit", "fine", "fined", "suspension",
    "negative", "rejection", "rejects", "concern",
}

# Urgency keyword boosters
_URGENCY_WORDS = {
    "breaking", "alert", "urgent", "just in", "fda", "sec", "doj",
    "earnings", "guidance", "acquisition", "merger", "layoffs", "ceo",
}


def get_news_feed(
    tickers: Optional[list[str]] = None,
    active_signals: Optional[dict[str, str]] = None,
    limit: int = 20,
) -> list[dict]:
    """
    Return processed news items for the given tickers.
    Uses 5-minute cache to avoid hammering Alpaca News API.
    """
    global _cache, _cache_ts

    now = time.monotonic()
    if now - _cache_ts < _CACHE_TTL and _cache:
        items = _cache
    else:
        from engine.heatmap_service import DEFAULT_TICKERS
        watch_tickers = tickers or DEFAULT_TICKERS
        items = _build_news_feed(watch_tickers, limit)
        if items:
            _cache    = items
            _cache_ts = now

    # Enrich with active signal linkage on every call (signals change more often)
    if active_signals:
        for item in items:
            linked = [
                {"ticker": t, "signalId": active_signals[t]}
                for t in item.get("tickers", [])
                if t in active_signals
            ]
            item["linkedSignals"] = linked

    return items[:limit]


def _build_news_feed(tickers: list[str], limit: int) -> list[dict]:
    """Fetch and process fresh news from Alpaca."""
    from engine.alpaca_client import get_multi_news, get_multi_bars

    try:
        raw_news = get_multi_news(tickers, limit=limit)
        if not raw_news:
            return []

        # Fetch daily bars for price-reaction lookup (1Day last 2 days)
        all_news_tickers = list({
            sym
            for item in raw_news
            for sym in (item.get("symbols") or [])
            if sym in tickers
        })
        # 5-min bars for intraday reaction
        intraday = get_multi_bars(all_news_tickers, timeframe="5Min", days=2) if all_news_tickers else {}

        processed: list[dict] = []
        for item in raw_news:
            try:
                processed.append(_process_item(item, intraday))
            except Exception as e:
                logger.debug(f"[news] process item failed: {e}")

        # Sort by urgency then time
        processed.sort(key=lambda x: (x["urgencyScore"], x["publishedAt"]), reverse=True)
        return processed

    except Exception as e:
        logger.error(f"[news] _build_news_feed failed: {e}")
        return []


def _process_item(raw: dict, intraday_bars: dict[str, any]) -> dict:
    """Turn one raw Alpaca news item into a processed news card."""

    headline   = raw.get("headline", "")
    summary    = raw.get("summary",  "")
    source     = raw.get("source",   "Unknown")
    symbols    = raw.get("symbols",  [])
    url        = raw.get("url",      "")
    author     = raw.get("author",   "")

    # Parse published time
    pub_str = raw.get("created_at", raw.get("updated_at", ""))
    try:
        pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
    except Exception:
        pub_dt = datetime.now(timezone.utc)

    # ── Sentiment analysis (keyword-based, fast) ──────────────────────────────
    text_lower = (headline + " " + summary).lower()
    words      = set(text_lower.split())

    bullish_hits = len(words & _BULLISH_WORDS)
    bearish_hits = len(words & _BEARISH_WORDS)
    urgency_hits = len(words & _URGENCY_WORDS)

    if bullish_hits > bearish_hits:
        sentiment       = "bullish"
        sentiment_score = min(100, bullish_hits * 20)
    elif bearish_hits > bullish_hits:
        sentiment       = "bearish"
        sentiment_score = min(100, bearish_hits * 20)
    else:
        sentiment       = "neutral"
        sentiment_score = 0

    # Urgency 0-100
    urgency_score = min(100, urgency_hits * 25 + (20 if len(symbols) <= 2 else 0))

    # ── Price reaction (if 5-min bars available) ──────────────────────────────
    price_reaction = _compute_price_reaction(pub_dt, symbols, intraday_bars)

    # ── Suggested action ──────────────────────────────────────────────────────
    action = _suggested_action(sentiment, price_reaction, urgency_score)

    # ── Reaction score ────────────────────────────────────────────────────────
    reaction_score = _news_reaction_score(sentiment, sentiment_score, price_reaction, urgency_score)

    return {
        "id":              raw.get("id", ""),
        "headline":        headline,
        "summary":         summary[:300] if summary else "",
        "source":          source,
        "author":          author,
        "url":             url,
        "publishedAt":     pub_dt.isoformat(),
        "tickers":         symbols,
        "sentiment":       sentiment,
        "sentimentScore":  sentiment_score,
        "urgencyScore":    urgency_score,
        "priceReaction":   price_reaction,
        "suggestedAction": action,
        "reactionScore":   reaction_score,
        "linkedSignals":   [],  # filled in by get_news_feed()
    }


def _compute_price_reaction(
    pub_dt: datetime,
    symbols: list[str],
    intraday_bars: dict,
) -> Optional[dict]:
    """
    Find the 5-min bar at the time of news publication and compute:
    priceBefore, priceAfter1m (approx), priceAfter5m, priceAfter15m,
    pctChange5m, volumeSpike.
    """
    if not symbols or not intraday_bars:
        return None

    # Use first symbol with bar data
    ticker = None
    df     = None
    for sym in symbols:
        if sym in intraday_bars and intraday_bars[sym] is not None:
            ticker = sym
            df     = intraday_bars[sym]
            break

    if df is None or df.empty:
        return None

    try:
        # Find bar nearest to publish time
        pub_ts = pub_dt.replace(tzinfo=timezone.utc) if pub_dt.tzinfo is None else pub_dt
        idx    = df.index

        # Bars after news
        after_mask = idx >= pub_ts
        after_bars = df[after_mask]

        if after_bars.empty:
            return None  # news is from the future or very recent — no reaction bars yet

        # Price before = close of bar immediately before news
        before_mask  = idx < pub_ts
        before_bars  = df[before_mask]
        price_before = float(before_bars["close"].iloc[-1]) if not before_bars.empty else float(after_bars["open"].iloc[0])

        price_5m  = float(after_bars["close"].iloc[0])  if len(after_bars) >= 1 else None
        price_15m = float(after_bars["close"].iloc[2])  if len(after_bars) >= 3 else None

        vol_at_news = float(after_bars["volume"].iloc[0]) if len(after_bars) >= 1 else 0
        avg_vol     = float(df["volume"].mean())
        vol_spike   = round(vol_at_news / avg_vol, 2) if avg_vol > 0 else 1.0

        pct_5m  = round((price_5m  - price_before) / price_before * 100, 2) if price_5m  else None
        pct_15m = round((price_15m - price_before) / price_before * 100, 2) if price_15m else None

        return {
            "ticker":       ticker,
            "priceBefore":  round(price_before, 2),
            "priceAfter5m": round(price_5m,  2) if price_5m  else None,
            "priceAfter15m":round(price_15m, 2) if price_15m else None,
            "pctChange5m":  pct_5m,
            "pctChange15m": pct_15m,
            "volumeSpike":  vol_spike,
        }

    except Exception as e:
        logger.debug(f"[news] price reaction failed: {e}")
        return None


def _suggested_action(
    sentiment: str,
    price_reaction: Optional[dict],
    urgency: int,
) -> str:
    """Plain-English suggested action — never financial advice."""
    if price_reaction is None:
        if urgency >= 60:
            return "high_risk_event"
        return "monitor"

    pct5 = price_reaction.get("pctChange5m") or 0
    vol  = price_reaction.get("volumeSpike", 1.0)

    # Confirm or contradict
    if sentiment == "bullish" and pct5 >= 0.5:
        return "possible_continuation"
    if sentiment == "bearish" and pct5 <= -0.5:
        return "possible_continuation"
    if sentiment == "bullish" and pct5 <= -0.3 and vol >= 1.5:
        return "possible_reversal"
    if sentiment == "bearish" and pct5 >= 0.3 and vol >= 1.5:
        return "possible_reversal"
    if abs(pct5) >= 2.0:
        return "avoid_chasing"

    return "monitor"


def _news_reaction_score(
    sentiment: str,
    sentiment_score: int,
    price_reaction: Optional[dict],
    urgency: int,
) -> int:
    """
    newsReactionScore 0-100:
    Higher = more significant news event.
    Does NOT indicate trade direction.
    """
    score = 0.0

    # Urgency component (40%)
    score += urgency * 0.4

    # Sentiment strength (30%)
    score += sentiment_score * 0.3

    # Price reaction magnitude (30%)
    if price_reaction:
        pct5 = abs(price_reaction.get("pctChange5m") or 0)
        vol  = min(price_reaction.get("volumeSpike", 1.0), 5.0)
        score += min(30, pct5 * 5 + (vol - 1) * 3)

    return int(min(100, round(score)))
