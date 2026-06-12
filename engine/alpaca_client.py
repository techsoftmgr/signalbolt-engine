"""
Shared Alpaca client singleton — used by all engine modules.

Why a singleton? Creating StockHistoricalDataClient is expensive (sets up
a connection pool + auth round-trip). With 5 strategy scans × 150 tickers
per cycle, re-creating the client per call wastes ~300 ms/cycle and burns
Alpaca rate-limit budget. One shared client handles everything.

Usage:
    from engine.alpaca_client import get_latest_price, get_latest_prices, get_bars, get_news

All functions return None / empty dict / [] on failure so callers can fall
back to yfinance without crashing.
"""

import logging
import os
from typing import Optional

import pandas as pd

logger = logging.getLogger("signalbolt.alpaca")

# ── Module-level singleton ────────────────────────────────────────────────────
_client = None
_ok     = False


def _init() -> None:
    """Lazy-initialise the shared client once."""
    global _client, _ok
    if _ok:
        return
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        key    = os.environ.get("ALPACA_API_KEY", "")
        secret = os.environ.get("ALPACA_SECRET_KEY", "")
        if not key or not secret:
            raise ValueError("Alpaca API keys not set")
        _client = StockHistoricalDataClient(key, secret)
        _ok     = True
        logger.info("[alpaca] Singleton client initialised (SIP feed)")
    except Exception as e:
        logger.warning(f"[alpaca] Client init failed: {e} — will rely on yfinance fallback")


# ── Crypto support (additive; isolated from the stock path) ─────────────────────
# Alpaca crypto needs NO API key/entitlement. A separate historical client serves
# crypto bars/quotes; stock symbols never touch it. Routing decided by crypto_assets.
_crypto_client = None
_crypto_ok     = False


def _init_crypto() -> None:
    global _crypto_client, _crypto_ok
    if _crypto_ok:
        return
    try:
        from alpaca.data.historical import CryptoHistoricalDataClient
        _crypto_client = CryptoHistoricalDataClient()   # keyless
        _crypto_ok     = True
        logger.info("[alpaca] Crypto client initialised")
    except Exception as e:
        logger.warning(f"[alpaca] Crypto client init failed: {e}")


def _crypto_tf(timeframe: str):
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    return {
        "1Min":  TimeFrame(1,  TimeFrameUnit.Minute),
        "5Min":  TimeFrame(5,  TimeFrameUnit.Minute),
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "1Hour": TimeFrame(1,  TimeFrameUnit.Hour),
        "1Day":  TimeFrame(1,  TimeFrameUnit.Day),
        "1Week": TimeFrame(1,  TimeFrameUnit.Week),
    }.get(timeframe, TimeFrame(1, TimeFrameUnit.Day))


def _crypto_bars(bases: list[str], timeframe: str, days: int) -> dict[str, pd.DataFrame]:
    """{base_symbol: OHLCV df} for crypto. Keyed by the BASE symbol (e.g. 'BTC')
    so callers see the same key they passed in. {} on any failure."""
    _init_crypto()
    if not _crypto_ok or _crypto_client is None or not bases:
        return {}
    try:
        from datetime import datetime, timezone, timedelta
        from alpaca.data.requests import CryptoBarsRequest
        from engine import crypto_assets
        pair_to_base = {}
        for b in bases:
            p = crypto_assets.to_pair(b)
            if p:
                pair_to_base[p] = crypto_assets.base(b)
        if not pair_to_base:
            return {}
        req = CryptoBarsRequest(
            symbol_or_symbols=list(pair_to_base.keys()),
            timeframe=_crypto_tf(timeframe),
            start=datetime.now(timezone.utc) - timedelta(days=days),
        )
        df = _crypto_client.get_crypto_bars(req).df
        if df is None or df.empty:
            return {}
        out: dict[str, pd.DataFrame] = {}
        for pair, b in pair_to_base.items():
            try:
                t_df = df.xs(pair, level=0).copy() if isinstance(df.index, pd.MultiIndex) else df.copy()
                t_df.columns = [c.lower() for c in t_df.columns]
                t_df.index   = pd.to_datetime(t_df.index, utc=True)
                out[b] = t_df
            except Exception:
                pass
        return out
    except Exception as e:
        logger.debug(f"[alpaca] crypto bars {bases} failed: {e}")
        return {}


def _crypto_latest_prices(bases: list[str]) -> dict[str, float]:
    """{base_symbol: last trade price} for crypto. {} on failure."""
    _init_crypto()
    if not _crypto_ok or _crypto_client is None or not bases:
        return {}
    try:
        from alpaca.data.requests import CryptoLatestTradeRequest
        from engine import crypto_assets
        pair_to_base = {}
        for b in bases:
            p = crypto_assets.to_pair(b)
            if p:
                pair_to_base[p] = crypto_assets.base(b)
        if not pair_to_base:
            return {}
        resp = _crypto_client.get_crypto_latest_trade(
            CryptoLatestTradeRequest(symbol_or_symbols=list(pair_to_base.keys()))
        )
        out: dict[str, float] = {}
        for pair, b in pair_to_base.items():
            t = resp.get(pair) if hasattr(resp, "get") else resp[pair]
            if t and t.price:
                out[b] = float(t.price)
        return out
    except Exception as e:
        logger.debug(f"[alpaca] crypto latest prices {bases} failed: {e}")
        return {}


def get_crypto_snapshots(bases: list[str]) -> dict[str, dict]:
    """{base: {price, changePercent, session, source}} for crypto — the /prices
    response shape. changePercent is vs the previous completed UTC daily close.
    24/7 market, so session is always 'market'. {} on failure."""
    prices = _crypto_latest_prices(bases)
    if not prices:
        return {}
    daily = _crypto_bars(list(prices.keys()), "1Day", 4)
    out: dict[str, dict] = {}
    for b, px in prices.items():
        chg = None
        try:
            df = daily.get(b)
            if df is not None and len(df) >= 2:
                prev = float(df["close"].iloc[-2])   # last COMPLETED daily close
                if prev > 0:
                    chg = round((px - prev) / prev * 100, 3)
        except Exception:
            pass
        out[b] = {
            "price":         round(px, 2 if px >= 1 else 6),
            "changePercent": chg,
            "session":       "market",
            "source":        "alpaca-crypto",
        }
    return out


# ── Price helpers ─────────────────────────────────────────────────────────────

def get_latest_price(ticker: str) -> Optional[float]:
    """
    Real-time latest trade price (Alpaca SIP).
    Returns None on any failure — caller should fall back to yfinance.
    """
    from engine import crypto_assets
    if crypto_assets.is_crypto(ticker):
        return _crypto_latest_prices([ticker]).get(crypto_assets.base(ticker))
    _init()
    if not _ok or _client is None:
        return None
    try:
        from alpaca.data.requests import StockLatestTradeRequest
        # feed="sip" = full consolidated tape (same as our bars), so the live
        # last-trade matches what ToS/brokers show. Without it the request used
        # the account default (often IEX-only), which lags/differs from ToS.
        resp = _client.get_stock_latest_trade(
            StockLatestTradeRequest(symbol_or_symbols=ticker, feed="sip")
        )
        p = resp[ticker].price
        return float(p) if p else None
    except Exception as e:
        logger.debug(f"[alpaca] latest_price({ticker}) failed: {e}")
        return None


def get_latest_prices(tickers: list[str]) -> dict[str, float]:
    """
    Batch real-time prices for multiple tickers in ONE API call.
    Returns {ticker: price} — missing tickers are omitted (not in response).
    """
    if not tickers:
        return {}
    # Split crypto out to the keyless crypto client; stocks stay on SIP.
    from engine import crypto_assets
    crypto_syms = [t for t in tickers if crypto_assets.is_crypto(t)]
    stock_syms  = [t for t in tickers if not crypto_assets.is_crypto(t)]
    out: dict[str, float] = {}
    if crypto_syms:
        out.update(_crypto_latest_prices(crypto_syms))
    if not stock_syms:
        return out
    _init()
    if not _ok or _client is None:
        return out
    try:
        from alpaca.data.requests import StockLatestTradeRequest
        resp = _client.get_stock_latest_trade(
            StockLatestTradeRequest(symbol_or_symbols=stock_syms, feed="sip")
        )
        for t in stock_syms:
            if t in resp and resp[t].price:
                out[t] = float(resp[t].price)
        return out
    except Exception as e:
        logger.debug(f"[alpaca] batch latest_prices failed: {e}")
        return out


def get_overnight_prices(tickers: list[str]) -> dict[str, float]:
    """Batch latest trade prices from Alpaca's OVERNIGHT feed (the ~8pm-4am ET
    Blue Ocean session). DISPLAY-ONLY — never route these into signal/stop logic:
    the overnight tape is thin and gappy (phantom-stop risk). Requires an Alpaca
    plan that includes the overnight feed; returns {} on ANY error (unsubscribed
    / unsupported SDK / outside session) so the feature stays safely dormant."""
    _init()
    if not _ok or _client is None or not tickers:
        return {}
    try:
        from alpaca.data.requests import StockLatestTradeRequest
        resp = _client.get_stock_latest_trade(
            StockLatestTradeRequest(symbol_or_symbols=tickers, feed="overnight")
        )
        return {
            t: float(resp[t].price)
            for t in tickers
            if t in resp and resp[t].price
        }
    except Exception as e:
        logger.debug(f"[alpaca] overnight prices failed (feed may be unsubscribed): {e}")
        return {}


# ── Quote helpers ─────────────────────────────────────────────────────────────

def get_latest_quote(ticker: str) -> Optional[dict]:
    """
    Real-time NBBO quote (Alpaca SIP). Returns {'bid', 'ask', 'mid', 'spread_pct'}
    or None on failure. Used by entry_gate to reject wide-spread entries.
    """
    _init()
    if not _ok or _client is None:
        return None
    try:
        from alpaca.data.requests import StockLatestQuoteRequest
        resp  = _client.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=ticker)
        )
        q     = resp[ticker]
        bid   = float(q.bid_price) if q.bid_price else 0.0
        ask   = float(q.ask_price) if q.ask_price else 0.0
        if bid <= 0 or ask <= 0:
            return None
        mid   = (bid + ask) / 2
        return {
            "bid":        bid,
            "ask":        ask,
            "mid":        mid,
            "spread_pct": (ask - bid) / mid * 100 if mid > 0 else 0.0,
        }
    except Exception as e:
        logger.debug(f"[alpaca] latest_quote({ticker}) failed: {e}")
        return None


# ── Bar helpers ───────────────────────────────────────────────────────────────

def get_bars(
    ticker: str,
    timeframe: str = "5Min",
    days: int = 2,
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV bars from Alpaca (SIP feed, real-time on paid plan).

    timeframe options: "1Min", "5Min", "15Min", "1Hour", "1Day"
    days: how many calendar days of history to fetch

    Returns a DataFrame with lowercase columns: open, high, low, close, volume
    indexed by timestamp (UTC). Returns None on failure.
    """
    from engine import crypto_assets
    if crypto_assets.is_crypto(ticker):
        return _crypto_bars([ticker], timeframe, days).get(crypto_assets.base(ticker))
    _init()
    if not _ok or _client is None:
        return None
    try:
        from datetime import datetime, timezone, timedelta
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        tf_map = {
            "1Min":  TimeFrame(1,  TimeFrameUnit.Minute),
            "5Min":  TimeFrame(5,  TimeFrameUnit.Minute),
            "15Min": TimeFrame(15, TimeFrameUnit.Minute),
            "1Hour": TimeFrame(1,  TimeFrameUnit.Hour),
            "1Day":  TimeFrame(1,  TimeFrameUnit.Day),
            "1Week": TimeFrame(1,  TimeFrameUnit.Week),
        }
        tf    = tf_map.get(timeframe, TimeFrame(5, TimeFrameUnit.Minute))
        start = datetime.now(timezone.utc) - timedelta(days=days)

        req  = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=tf,
            start=start,
            feed="sip",
            adjustment="split",   # back-adjust price AND volume for splits — else a
                                  # split shows as a phantom -90% gap (false breakdown)
                                  # + split-inflated volume (e.g. KLAC 10:1, 2026-06-12)
        )
        bars = _client.get_stock_bars(req)
        df   = bars.df

        if df is None or df.empty:
            return None

        # Flatten multi-index (symbol, timestamp) → just timestamp index
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(ticker, level=0)

        df.columns = [c.lower() for c in df.columns]
        df.index   = pd.to_datetime(df.index, utc=True)
        return df

    except Exception as e:
        logger.debug(f"[alpaca] get_bars({ticker}, {timeframe}, {days}d) failed: {e}")
        return None


# ── Stop/target breach confirmation ───────────────────────────────────────────

def confirm_level_cross(
    ticker: str,
    level: float,
    is_long: bool,
    kind: str,          # "stop" | "target"
    lookback: int = 5,
) -> bool:
    """
    Confirm a stop/target level was REALLY reached before closing a position.

    Guards against a single bad/out-of-sequence SIP last-trade print (or a brief
    feed glitch) closing a trade at a price the tape never actually printed — the
    2026-06-03 phantom-stop incident (CMCSA booked "stop @ 26.50" while the 1-min
    high was 23.72; BA "stop @ 230" while its whole-day high was 217.72).

    A cross is confirmed if EITHER corroborating source agrees the level was
    touched:
      1) the high/low of the last `lookback` completed 1-min bars, OR
      2) a fresh, independent last-trade read (a real move persists across two
         reads; a one-off bad print reverts).

    Returns False when neither source corroborates — the caller must then NOT
    close (it re-checks next tick/pass). Also False if no data is available at
    all (fail-closed: never fabricate a close).
    """
    def _breach(hi: float, lo: float) -> bool:
        if kind == "stop":
            return (lo <= level) if is_long else (hi >= level)
        return (hi >= level) if is_long else (lo <= level)   # target

    # 1) recent completed 1-min bars (bar aggregation excludes most bad prints,
    #    and is what the user's chart shows)
    try:
        df = get_bars(ticker, timeframe="1Min", days=1)
        if df is not None and len(df) > 0 and {"high", "low"}.issubset(df.columns):
            recent = df.tail(max(1, lookback))
            if _breach(float(recent["high"].max()), float(recent["low"].min())):
                return True
    except Exception as e:
        logger.debug(f"[alpaca] confirm_level_cross bars({ticker}) failed: {e}")

    # 2) fresh independent last-trade read — a sustained move still breaches;
    #    a transient bad print has reverted by now.
    try:
        p2 = get_latest_price(ticker)
        if p2 is not None:
            return _breach(float(p2), float(p2))
    except Exception as e:
        logger.debug(f"[alpaca] confirm_level_cross 2nd-read({ticker}) failed: {e}")

    return False


def sane_close_price(
    ticker: str,
    raw_price: Optional[float],
    lookback: int = 5,
    max_dev_pct: float = 5.0,
) -> Optional[float]:
    """
    Reject a gross bad-print close price. Used by NON-level closes (EOD,
    time-stop, near-expiry, trend exit) that record `get_latest_price()` directly
    — a single bad SIP print there would mis-record P&L the same way the phantom
    stop-outs did.

    If `raw_price` sits more than `max_dev_pct` OUTSIDE the recent `lookback`-bar
    1-min range, clamp it to the nearest real extreme (a real move that large
    would itself appear in the bars, so it would NOT be clamped — only out-of-tape
    prints are). Returns `raw_price` unchanged when within tolerance or when bars
    are unavailable (fail-open: never block a legitimate close).
    """
    if raw_price is None:
        return raw_price
    try:
        df = get_bars(ticker, timeframe="1Min", days=1)
        if df is None or len(df) == 0 or not {"high", "low"}.issubset(df.columns):
            return raw_price
        recent = df.tail(max(1, lookback))
        hi = float(recent["high"].max())
        lo = float(recent["low"].min())
        if raw_price > hi * (1 + max_dev_pct / 100):
            logger.warning(f"[alpaca] {ticker} close price {raw_price:.2f} > recent "
                           f"high {hi:.2f}+{max_dev_pct}% — bad print, clamped to {hi:.2f}")
            return hi
        if raw_price < lo * (1 - max_dev_pct / 100):
            logger.warning(f"[alpaca] {ticker} close price {raw_price:.2f} < recent "
                           f"low {lo:.2f}-{max_dev_pct}% — bad print, clamped to {lo:.2f}")
            return lo
        return raw_price
    except Exception as e:
        logger.debug(f"[alpaca] sane_close_price({ticker}) failed: {e}")
        return raw_price


# ── News helper ───────────────────────────────────────────────────────────────

def get_news(ticker: str, limit: int = 6) -> list[dict]:
    """
    Fetch latest news articles from Alpaca News API.

    Returns a list of dicts with at least {"headline": str, "summary": str}.
    Returns [] on failure so callers can fall back gracefully.
    """
    try:
        import requests as _req
        key    = os.environ.get("ALPACA_API_KEY", "")
        secret = os.environ.get("ALPACA_SECRET_KEY", "")
        if not key or not secret:
            return []
        resp = _req.get(
            "https://data.alpaca.markets/v1beta1/news",
            params={"symbols": ticker, "limit": limit, "sort": "desc"},
            headers={
                "APCA-API-KEY-ID":     key,
                "APCA-API-SECRET-KEY": secret,
            },
            timeout=5,
        )
        if resp.ok:
            return resp.json().get("news", [])
    except Exception as e:
        logger.debug(f"[alpaca] get_news({ticker}) failed: {e}")
    return []


def get_multi_bars(
    tickers: list[str],
    timeframe: str = "1Day",
    days: int = 21,
) -> dict[str, pd.DataFrame]:
    """
    Batch OHLCV bars for multiple tickers in a SINGLE Alpaca API call.
    Used by heatmap and quant services to avoid per-ticker round-trips.

    Returns {ticker: DataFrame(open,high,low,close,volume)} — missing tickers omitted.
    Returns {} on failure so callers can degrade gracefully.
    """
    if not tickers:
        return {}
    # Split crypto out to the keyless crypto client; stocks stay on SIP.
    from engine import crypto_assets
    crypto_syms = [t for t in tickers if crypto_assets.is_crypto(t)]
    stock_syms  = [t for t in tickers if not crypto_assets.is_crypto(t)]
    result: dict[str, pd.DataFrame] = {}
    if crypto_syms:
        result.update(_crypto_bars(crypto_syms, timeframe, days))
    if not stock_syms:
        return result
    tickers = stock_syms
    _init()
    if not _ok or _client is None:
        return result
    try:
        from datetime import datetime, timezone, timedelta
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        tf_map = {
            "1Min":  TimeFrame(1,  TimeFrameUnit.Minute),
            "5Min":  TimeFrame(5,  TimeFrameUnit.Minute),
            "15Min": TimeFrame(15, TimeFrameUnit.Minute),
            "1Hour": TimeFrame(1,  TimeFrameUnit.Hour),
            "1Day":  TimeFrame(1,  TimeFrameUnit.Day),
            "1Week": TimeFrame(1,  TimeFrameUnit.Week),
        }
        tf    = tf_map.get(timeframe, TimeFrame(1, TimeFrameUnit.Day))
        start = datetime.now(timezone.utc) - timedelta(days=days)

        req  = StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=tf,
            start=start,
            feed="sip",
            adjustment="split",   # split-adjust price AND volume (see get_bars note) —
                                  # keeps levels + relativeVolume continuous across splits
        )
        bars = _client.get_stock_bars(req)
        df   = bars.df

        if df is None or df.empty:
            return result   # keep any crypto entries already collected

        for ticker in tickers:
            try:
                if isinstance(df.index, pd.MultiIndex):
                    t_df = df.xs(ticker, level=0).copy()
                else:
                    t_df = df.copy()
                t_df.columns = [c.lower() for c in t_df.columns]
                t_df.index   = pd.to_datetime(t_df.index, utc=True)
                result[ticker] = t_df
            except Exception:
                pass  # ticker not in response — omit silently
        return result

    except Exception as e:
        logger.debug(f"[alpaca] get_multi_bars({len(tickers)} tickers, {timeframe}, {days}d) failed: {e}")
        return result   # keep any crypto entries already collected


def get_multi_news(tickers: list[str], limit: int = 10) -> list[dict]:
    """
    Batch news for multiple tickers from Alpaca News API.
    Returns list sorted newest-first, or [] on failure.
    """
    try:
        import requests as _req
        key    = os.environ.get("ALPACA_API_KEY", "")
        secret = os.environ.get("ALPACA_SECRET_KEY", "")
        if not key or not secret:
            return []
        resp = _req.get(
            "https://data.alpaca.markets/v1beta1/news",
            params={"symbols": ",".join(tickers), "limit": limit, "sort": "desc"},
            headers={
                "APCA-API-KEY-ID":     key,
                "APCA-API-SECRET-KEY": secret,
            },
            timeout=8,
        )
        if resp.ok:
            return resp.json().get("news", [])
    except Exception as e:
        logger.debug(f"[alpaca] get_multi_news failed: {e}")
    return []
