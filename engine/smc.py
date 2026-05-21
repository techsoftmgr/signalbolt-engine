"""
Smart Money Concepts (SMC) analysis engine.
Fetches OHLCV via Alpaca (primary) or yfinance (fallback) and detects:
  - Swing highs / swing lows
  - Break of Structure (BOS)
  - Change of Character (CHoCH)
  - Fair Value Gaps (FVG)
  - Order Blocks (OB)
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

import logging
from engine.config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_DATA_FEED

logger = logging.getLogger("signalbolt.smc")

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    _alpaca_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    _ALPACA_OK = True
except ImportError:
    _ALPACA_OK = False

SWING_WINDOW = 5   # ↑ from 3 — 3 bars = 45-min noise, 5 bars = 75-min real structure

# Alpaca timeframe mapping: interval → (TimeFrame, days_lookback)
_ALPACA_TIMEFRAMES = {
    "5m":  (lambda: TimeFrame(5,  TimeFrameUnit.Minute), 1),
    "15m": (lambda: TimeFrame(15, TimeFrameUnit.Minute), 5),
    "1h":  (lambda: TimeFrame(1,  TimeFrameUnit.Hour),  60),
    # Fix #2: 4H was missing — L5 multiframe always fell back to partial credit
    "4h":  (lambda: TimeFrame(4,  TimeFrameUnit.Hour),  60),
}

# Strategy-specific level sizing parameters
# NOTE: tp1_pct/tp2_pct/sl_fallback are percentage FALLBACKS used only when no
# SMC structure (OB/sweep/swing) is found.  The sl_tp_engine overrides these
# with ATR-based levels for all passing signals, so keep these conservative.
STRATEGY_PARAMS: dict[str, dict] = {
    'scalping': {
        'tp1_pct':     0.004,   # +0.4% (tight scalp)
        'tp2_pct':     0.007,   # +0.7%
        'sl_fallback': 0.003,   # 0.3% fallback SL
        'atr_mult':    0.8,     # ATR × 0.8 below key level (intraday H-L ATR)
        'sweep_bars':  5,
    },
    'day_trade': {
        'tp1_pct':     0.008,   # +0.8%  ↓ from 1.5% — achievable same day
        'tp2_pct':     0.015,   # +1.5%  ↓ from 3.0%
        'sl_fallback': 0.005,   # 0.5%   ↓ from 1.0%
        'atr_mult':    1.0,     # ATR × 1.0 (intraday H-L ATR — no overnight gap)
        'sweep_bars':  5,
    },
    'swing_trade': {
        'tp1_pct':     0.040,   # +4.0%
        'tp2_pct':     0.070,   # +7.0%
        'sl_fallback': 0.020,   # 2.0%
        'atr_mult':    2.0,     # True Range ATR for swing
        'sweep_bars':  7,
    },
    'options_flow': {
        'tp1_pct':     0.008,
        'tp2_pct':     0.015,
        'sl_fallback': 0.005,
        'atr_mult':    1.0,
        'sweep_bars':  5,
    },
    'dark_pool': {
        'tp1_pct':     0.008,
        'tp2_pct':     0.015,
        'sl_fallback': 0.005,
        'atr_mult':    1.0,
        'sweep_bars':  5,
    },
}


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _filter_market_hours(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    Strip pre-market and post-market bars from intraday data.

    Pre/post market bars (4 AM–9:29 AM and 4:01 PM–8 PM ET) have thin
    volume and wide spreads — SMC signals fired on them are unreliable.
    Only applied to 5m and 15m intervals; 1h bars are left as-is.

    Falls back to the full DataFrame if timezone conversion fails or
    fewer than 10 regular-hours bars remain (prevents empty analysis).
    """
    if interval not in ("5m", "15m"):
        return df
    try:
        ts = pd.to_datetime(df["timestamp"], utc=True)
        # Convert to ET — handles EST/EDT automatically
        ts_et = ts.dt.tz_convert("America/New_York")
        # Minutes since midnight in ET
        minutes = ts_et.dt.hour * 60 + ts_et.dt.minute
        # 9:30 AM = 570 min, 4:00 PM = 960 min
        mask = (minutes >= 570) & (minutes < 960)
        filtered = df[mask].reset_index(drop=True)
        # Guard: if filtering removes too many bars, return original
        return filtered if len(filtered) >= 10 else df
    except Exception as e:
        logger.debug(f"[smc] Market-hours filter failed: {e} — returning full bars")
        return df


def _fetch_alpaca(ticker: str, interval: str = "15m") -> pd.DataFrame:
    """Fetch OHLCV from Alpaca. Returns normalised DataFrame or empty DF."""
    if not _ALPACA_OK:
        return pd.DataFrame()
    try:
        tf_factory, days = _ALPACA_TIMEFRAMES.get(interval, _ALPACA_TIMEFRAMES["15m"])
        # Fix: use timezone-aware datetime (datetime.utcnow() is deprecated in 3.12+)
        start = datetime.now(timezone.utc) - timedelta(days=days)
        req = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=tf_factory(),
            start=start,
            feed=ALPACA_DATA_FEED,   # "sip" on paid plan, "iex" on free
        )
        bars = _alpaca_client.get_stock_bars(req)
        df = bars.df
        if df is None or df.empty:
            return pd.DataFrame()

        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(ticker, level="symbol")

        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]

        for candidate in ("timestamp", "datetime", "date"):
            if candidate in df.columns:
                if candidate != "timestamp":
                    df = df.rename(columns={candidate: "timestamp"})
                break

        if "timestamp" not in df.columns:
            df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)

        df = df.sort_values("timestamp").reset_index(drop=True)

        # Strip pre/post market bars for intraday intervals (Fix #2)
        df = _filter_market_hours(df, interval)

        return df
    except Exception as e:
        logger.warning(f"[smc] Alpaca error for {ticker}: {e}")
        return pd.DataFrame()


def _fetch_yfinance(ticker: str, period: str = "10d", interval: str = "1h") -> pd.DataFrame:
    """Fetch OHLCV from yfinance. Returns normalised DataFrame or empty DF."""
    try:
        import yfinance as yf
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        if df.empty or len(df) < 20:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        df.columns = [c.lower() for c in df.columns]
        df = df.reset_index()

        for candidate in ("datetime", "date"):
            if candidate in df.columns:
                df = df.rename(columns={candidate: "timestamp"})
                break

        if "timestamp" not in df.columns:
            df["timestamp"] = df.index

        df = df.sort_values("timestamp").reset_index(drop=True)
        return df
    except Exception as e:
        logger.warning(f"[smc] yfinance error for {ticker}: {e}")
        return pd.DataFrame()


def fetch_candles(ticker: str, period: str = "10d", interval: str = "1h") -> pd.DataFrame:
    """Fetch OHLCV — Alpaca primary, yfinance fallback."""
    df = _fetch_alpaca(ticker, interval)
    if not df.empty and len(df) >= 20:
        return df
    logger.info(f"[smc] Alpaca insufficient for {ticker} — yfinance fallback")
    df = _fetch_yfinance(ticker, period, interval)
    if df.empty or len(df) < 20:
        logger.warning(f"[smc] Insufficient data for {ticker}")
        return pd.DataFrame()
    return df


# ---------------------------------------------------------------------------
# Swing detection
# ---------------------------------------------------------------------------

def detect_swings(df: pd.DataFrame, n: int = SWING_WINDOW) -> pd.DataFrame:
    df = df.copy()
    df["swing_high"] = False
    df["swing_low"] = False
    for i in range(n, len(df) - n):
        window_h = df["high"].iloc[i - n: i + n + 1]
        window_l = df["low"].iloc[i - n: i + n + 1]
        if df["high"].iloc[i] == window_h.max():
            df.at[df.index[i], "swing_high"] = True
        if df["low"].iloc[i] == window_l.min():
            df.at[df.index[i], "swing_low"] = True
    return df


# ---------------------------------------------------------------------------
# BOS / CHoCH
# ---------------------------------------------------------------------------

def detect_structure(df: pd.DataFrame) -> dict:
    """
    Detect BOS and CHoCH with confirmation requirement.

    Confirmation rule: the structural break must have happened at least 1 bar
    ago AND the CLOSE of that confirmation bar must hold above/below the level.
    This filters out wick-only fakeouts where price briefly breaks a level
    but immediately reverses — a common retail trap.

    We also require the break bar to be in the RECENT N bars (max_structure_age),
    so we don't fire signals based on structure breaks from days ago.
    """
    sh_df  = df[df["swing_high"]]
    sl_df  = df[df["swing_low"]]
    sh     = sh_df["high"].values
    sl     = sl_df["low"].values
    price  = df["close"].iloc[-1]
    n      = len(df)

    # The break bar must be within the last 20 bars to be considered "current"
    max_structure_age = 20

    bos_bullish = bos_bearish = False
    choch_bullish = choch_bearish = False

    if len(sh) >= 2:
        level = sh[-1]
        # Find the index of where this swing high was formed
        sh_idx = sh_df.index[-1] if hasattr(sh_df.index, '__len__') else n - 1
        try:
            sh_pos = df.index.get_loc(sh_idx)
        except Exception:
            sh_pos = n - SWING_WINDOW - 1

        bars_since_swing = n - 1 - sh_pos

        # Structure is only valid if the break is recent (within max_structure_age bars)
        if bars_since_swing <= max_structure_age:
            # BOS: current close AND at least 1 bar's close above the swing high
            # (not just a wick poke — requires 2+ consecutive closes above)
            last2_closes = df["close"].iloc[-2:]
            if all(c > level for c in last2_closes):
                bos_bullish = True
            # CHoCH: was making lower highs, now confirmed break above
            if sh[-1] < sh[-2] and bos_bullish:
                choch_bullish = True

    if len(sl) >= 2:
        level = sl[-1]
        sl_idx = sl_df.index[-1] if hasattr(sl_df.index, '__len__') else n - 1
        try:
            sl_pos = df.index.get_loc(sl_idx)
        except Exception:
            sl_pos = n - SWING_WINDOW - 1

        bars_since_swing = n - 1 - sl_pos

        if bars_since_swing <= max_structure_age:
            last2_closes = df["close"].iloc[-2:]
            if all(c < level for c in last2_closes):
                bos_bearish = True
            if sl[-1] > sl[-2] and bos_bearish:
                choch_bearish = True

    return {
        "bos_bullish":   bos_bullish,
        "bos_bearish":   bos_bearish,
        "choch_bullish": choch_bullish,
        "choch_bearish": choch_bearish,
    }


# ---------------------------------------------------------------------------
# Fair Value Gaps
# ---------------------------------------------------------------------------

def detect_fvg(df: pd.DataFrame, max_age_bars: int = 30) -> dict:
    """
    Bullish FVG: candle[i-2].high < candle[i].low   — gap up, middle is impulse candle
    Bearish FVG: candle[i-2].low  > candle[i].high  — gap down
    Returns the FVG nearest to current price for each direction.

    max_age_bars: only consider FVGs formed within the last N bars.
    Stale FVGs (weeks old) are irrelevant to current price action — SMC
    respects freshness. Default 30 bars = ~7.5 hours on 15m chart.
    """
    bullish_fvgs: list[dict] = []
    bearish_fvgs: list[dict] = []

    # Only scan the most recent max_age_bars bars for fresh FVGs
    start_idx = max(2, len(df) - max_age_bars)

    for i in range(start_idx, len(df)):
        c0 = df.iloc[i - 2]
        c2 = df.iloc[i]
        if c0["high"] < c2["low"]:
            bullish_fvgs.append({
                "top":    float(c2["low"]),
                "bottom": float(c0["high"]),
                "ts":     c2.get("timestamp"),
            })
        if c0["low"] > c2["high"]:
            bearish_fvgs.append({
                "top":    float(c0["low"]),
                "bottom": float(c2["high"]),
                "ts":     c2.get("timestamp"),
            })

    price = float(df["close"].iloc[-1])

    def nearest(fvgs: list) -> Optional[dict]:
        if not fvgs:
            return None
        return min(fvgs, key=lambda f: abs(price - (f["top"] + f["bottom"]) / 2))

    return {
        "fvg_bullish": nearest(bullish_fvgs),
        "fvg_bearish": nearest(bearish_fvgs),
    }


# ---------------------------------------------------------------------------
# Order Blocks
# ---------------------------------------------------------------------------

def detect_order_blocks(
    df: pd.DataFrame,
    atr: float = 0.0,
    price: float = 0.0,
) -> dict:
    """
    Bullish OB: last bearish candle before a strong bullish impulse.
    Bearish OB: last bullish candle before a strong bearish impulse.
    Returns the most recently formed OB for each direction.

    Impulse threshold is ATR-relative when ATR/price are provided.
    Institutional OBs need at least 0.8× ATR body move.  Falls back
    to the static 0.6% minimum when ATR data is unavailable.
    """
    bullish_ob: Optional[dict] = None
    bearish_ob: Optional[dict] = None

    # ATR-relative impulse floor: 0.8× ATR but never below 0.6% of price
    # This prevents flagging noise candles as OBs on high-volatility tickers
    # (e.g. NVDA's 0.6% is $3 — that's noise, not institutional intent)
    if atr > 0 and price > 0:
        min_impulse_abs = max(atr * 0.8, price * 0.006)  # whichever is larger
        min_impulse_pct = min_impulse_abs / price
    else:
        min_impulse_pct = 0.006   # static 0.6% fallback

    for i in range(1, len(df) - 1):
        curr = df.iloc[i]
        nxt  = df.iloc[i + 1]

        if curr["close"] < curr["open"]:  # bearish candle
            body_move = (nxt["close"] - nxt["open"]) / nxt["open"] if nxt["open"] else 0
            if body_move > min_impulse_pct:
                bullish_ob = {
                    "top":    float(curr["open"]),
                    "bottom": float(curr["close"]),
                    "ts":     curr.get("timestamp"),
                    "impulse_pct": round(body_move * 100, 3),
                }

        if curr["close"] > curr["open"]:  # bullish candle
            body_move = (nxt["open"] - nxt["close"]) / nxt["open"] if nxt["open"] else 0
            if body_move > min_impulse_pct:
                bearish_ob = {
                    "top":    float(curr["close"]),
                    "bottom": float(curr["open"]),
                    "ts":     curr.get("timestamp"),
                    "impulse_pct": round(body_move * 100, 3),
                }

    return {"ob_bullish": bullish_ob, "ob_bearish": bearish_ob}


# ---------------------------------------------------------------------------
# ATR calculation
# ---------------------------------------------------------------------------

def _calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Average True Range over `period` bars. Returns 0.0 on failure."""
    try:
        high  = df["high"]
        low   = df["low"]
        prev  = df["close"].shift(1)
        tr = pd.concat([
            high - low,
            (high - prev).abs(),
            (low  - prev).abs(),
        ], axis=1).max(axis=1)
        val = tr.rolling(period).mean().iloc[-1]
        return float(val) if not np.isnan(val) else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Liquidity sweep detection  (market-maker stop-raid logic)
# ---------------------------------------------------------------------------

def detect_liquidity_sweep(df: pd.DataFrame, direction: str, lookback: int = 5) -> dict:
    """
    Detect whether market makers raided retail stop-losses just before the
    potential entry candle.

    For LONG  — price must have wicked *below* a prior confirmed swing low
                 in the last `lookback` candles, then *closed back above* it.
                 This is the classic "sell-side liquidity grab" before a rally.

    For SHORT — price wicked *above* a prior swing high then closed back below.
                 Classic "buy-side liquidity grab" before a drop.

    Returns a dict:
        swept       : bool   — True if a valid sweep was found
        sweep_wick  : float  — extreme wick price (new stop anchor)
        swept_level : float  — the swing level that was raided
        candles_ago : int    — how many candles ago the sweep occurred
    """
    no_sweep: dict = {"swept": False}
    min_candles = lookback + SWING_WINDOW + 1
    if len(df) < min_candles:
        return no_sweep

    # "Prior" = everything before the lookback window
    prior  = df.iloc[:-(lookback)]
    recent = df.iloc[-(lookback):]

    if direction == "LONG":
        prior_lows = prior[prior["swing_low"]]["low"].values
        if len(prior_lows) == 0:
            return no_sweep
        swept_level = float(prior_lows[-1])

        for i in range(len(recent)):
            candle = recent.iloc[i]
            # Wick went below the swing low but candle closed above it
            if float(candle["low"]) < swept_level and float(candle["close"]) > swept_level:
                return {
                    "swept":       True,
                    "sweep_wick":  float(candle["low"]),
                    "swept_level": swept_level,
                    "candles_ago": len(recent) - i,
                }

    else:  # SHORT
        prior_highs = prior[prior["swing_high"]]["high"].values
        if len(prior_highs) == 0:
            return no_sweep
        swept_level = float(prior_highs[-1])

        for i in range(len(recent)):
            candle = recent.iloc[i]
            if float(candle["high"]) > swept_level and float(candle["close"]) < swept_level:
                return {
                    "swept":       True,
                    "sweep_wick":  float(candle["high"]),
                    "swept_level": swept_level,
                    "candles_ago": len(recent) - i,
                }

    return no_sweep


# ---------------------------------------------------------------------------
# Entry-level calculation
# ---------------------------------------------------------------------------

def _calculate_levels(
    direction: str,
    price: float,
    obs: dict,
    df: pd.DataFrame,
    strategy_type: str = "day_trade",
    sweep: Optional[dict] = None,
) -> tuple:
    """
    Determine entry, stop-loss, T1, T2.

    Stop priority (best → fallback):
      1. Sweep wick  — stop goes ATR×mult below the wick that performed the raid
      2. Order block — stop goes ATR×mult below the OB bottom (LONG) / above OB top (SHORT)
      3. Swing level — stop goes ATR×mult below the last swing low (LONG) / above swing high (SHORT)
      4. Percentage  — sl_fallback % from entry (last resort)
    """
    p           = STRATEGY_PARAMS.get(strategy_type, STRATEGY_PARAMS["day_trade"])
    tp1_pct     = p["tp1_pct"]
    tp2_pct     = p["tp2_pct"]
    sl_fallback = p["sl_fallback"]
    atr_mult    = p["atr_mult"]

    atr = _calculate_atr(df)
    # Guard: if ATR is unrealistically large (>5% of price) cap the buffer
    atr_buffer = min(atr * atr_mult, price * sl_fallback * 2)
    if atr_buffer == 0:
        atr_buffer = price * sl_fallback

    swing_highs = df[df["swing_high"]]["high"].values
    swing_lows  = df[df["swing_low"]]["low"].values

    if direction == "LONG":
        ob    = obs.get("ob_bullish")
        entry = round(float(ob["top"]) if ob else price, 4)

        # Stop anchor: sweep wick > OB bottom > last swing low > fallback
        if sweep and sweep.get("swept"):
            sl_ref = sweep["sweep_wick"]          # below the raid wick
        elif ob:
            sl_ref = float(ob["bottom"])
        elif len(swing_lows):
            sl_ref = float(swing_lows[-1])
        else:
            sl_ref = price * (1 - sl_fallback)

        stop_loss  = round(sl_ref - atr_buffer, 4)
        target_one = round(entry * (1 + tp1_pct), 4)
        target_two = round(entry * (1 + tp2_pct), 4)

        # Override T1 with nearest swing high if it sits within T2 range
        if len(swing_highs):
            sh = float(swing_highs[-1])
            if entry < sh < entry * (1 + tp2_pct * 1.5):
                target_one = round(sh, 4)

    else:  # SHORT
        ob    = obs.get("ob_bearish")
        entry = round(float(ob["bottom"]) if ob else price, 4)

        # Stop anchor: sweep wick > OB top > last swing high > fallback
        if sweep and sweep.get("swept"):
            sl_ref = sweep["sweep_wick"]          # above the raid wick
        elif ob:
            sl_ref = float(ob["top"])
        elif len(swing_highs):
            sl_ref = float(swing_highs[-1])
        else:
            sl_ref = price * (1 + sl_fallback)

        stop_loss  = round(sl_ref + atr_buffer, 4)
        target_one = round(entry * (1 - tp1_pct), 4)
        target_two = round(entry * (1 - tp2_pct), 4)

        if len(swing_lows):
            sl_val = float(swing_lows[-1])
            if entry * (1 - tp2_pct * 1.5) < sl_val < entry:
                target_one = round(sl_val, 4)

    return entry, stop_loss, target_one, target_two


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def analyze(
    ticker: str,
    interval: str = "1h",
    period: str = "10d",
    strategy_type: str = "day_trade",
) -> Optional[dict]:
    """Run full SMC analysis. Returns analysis dict (with 'candles' key) or None."""
    try:
        df = fetch_candles(ticker, period=period, interval=interval)
        if df.empty:
            return None

        df = detect_swings(df)
        structure = detect_structure(df)

        price = float(df["close"].iloc[-1])
        avg_volume = float(df["volume"].mean())
        last_volume = float(df["volume"].iloc[-1])

        # Compute ATR early so OB detection can use ATR-relative threshold
        atr = _calculate_atr(df)
        fvgs = detect_fvg(df)
        obs  = detect_order_blocks(df, atr=atr, price=price)

        # ── Volume gate for intraday strategies ───────────────────────────────
        # Use the 2nd-to-last bar (last COMPLETED bar) to avoid partial current bar.
        # Skip if the last completed bar is below 40% of the recent average —
        # dead-market conditions produce unreliable SMC signals.
        if strategy_type in ("scalping", "day_trade") and len(df) >= 6:
            recent_avg = float(df["volume"].iloc[-5:-1].mean())  # last 4 completed bars
            last_completed = float(df["volume"].iloc[-2])        # penultimate bar
            # ↑ threshold 40%→60%: 40-59% volume = thin market, SMC patterns unreliable.
            # Raising to 60% filters low-conviction setups without being too restrictive.
            if recent_avg > 0 and last_completed < recent_avg * 0.60:
                logger.debug(
                    f"[smc] {ticker} [{strategy_type}] volume too low "
                    f"({last_completed:.0f} < 60% of avg {recent_avg:.0f}) — skip"
                )
                return None

        # ── Direction: require a genuine structural breakout ─────────────────
        # FVG and OB almost always exist on both sides of price — using them
        # to break ties produces a permanent LONG bias in ranging markets.
        # Only BOS or CHoCH (actual market structure shifts) determine direction.
        has_bullish_struct = structure["bos_bullish"] or structure["choch_bullish"]
        has_bearish_struct = structure["bos_bearish"] or structure["choch_bearish"]

        if not has_bullish_struct and not has_bearish_struct:
            direction = None   # no structural breakout
        elif has_bullish_struct and not has_bearish_struct:
            direction = "LONG"
        elif has_bearish_struct and not has_bullish_struct:
            direction = "SHORT"
        elif structure["choch_bullish"] and not structure["choch_bearish"]:
            direction = "LONG"   # CHoCH overrides conflicting BOS
        elif structure["choch_bearish"] and not structure["choch_bullish"]:
            direction = "SHORT"
        else:
            direction = None    # conflict — let smc_quality resolve below

        # ── SMC Quality enrichment ────────────────────────────────────────────
        # Runs smc_quality.enrich() to:
        #   1. Resolve conflicting structure (instead of just skipping)
        #   2. Score FVG/OB quality (ATR-relative, freshness-aware)
        #   3. Attach quality flags (OB_VIOLATED, FVG_STALE, etc.)
        quality_result = None
        quality_meta: dict = {}
        try:
            from engine import smc_quality
            _partial = {
                "ticker":        ticker,
                "direction":     direction,
                "current_price": price,
                "structure":     structure,
                "fvgs":          fvgs,
                "obs":           obs,
                "candles":       df,
                "strategy_type": strategy_type,
            }
            quality_result = smc_quality.enrich(_partial, df)

            # Use quality resolver to settle conflicts / upgrade ambiguous direction
            if direction is None and quality_result.direction_override not in (None, "AMBIGUOUS"):
                direction = quality_result.direction_override
                logger.info(
                    f"[smc] {ticker}: structure conflict resolved by quality scorer → {direction}"
                )

            # Pack quality metadata into the analysis dict for scorer L1 enrichment
            fq = quality_result.fvg_quality
            oq = quality_result.ob_quality
            sq = quality_result.structure_score
            quality_meta = {
                "fvg_quality_score":   round(fq.score, 1)  if fq else 0.0,
                "ob_quality_score":    round(oq.score, 1)  if oq else 0.0,
                "ob_is_violated":      oq.is_violated       if oq else False,
                "structure_raw_score": round(sq.raw_score, 2) if sq else 0.0,
                "quality_flags":       quality_result.quality_flags,
                "overall_quality":     round(quality_result.overall_quality, 1),
            }
        except Exception as _qe:
            logger.debug(f"[smc] quality enrichment skipped for {ticker}: {_qe}")

        # After conflict resolution, if still no direction → skip
        sweep: dict = {}
        entry, stop_loss, target_one, target_two = (None, None, None, None)
        if direction:
            sweep_bars = STRATEGY_PARAMS.get(strategy_type, STRATEGY_PARAMS["day_trade"])["sweep_bars"]
            sweep = detect_liquidity_sweep(df, direction, lookback=sweep_bars)
            entry, stop_loss, target_one, target_two = _calculate_levels(
                direction, price, obs, df, strategy_type, sweep=sweep
            )

        return {
            "ticker":           ticker,
            "current_price":    price,
            "direction":        direction,
            "entry":            entry,
            "stop_loss":        stop_loss,
            "target_one":       target_one,
            "target_two":       target_two,
            "structure":        structure,
            "fvgs":             fvgs,
            "obs":              obs,
            "liquidity_sweep":  sweep,
            "avg_volume":       avg_volume,
            "last_volume":      last_volume,
            "candles":          df,
            "timeframe":        interval,
            "strategy_type":    strategy_type,
            "atr":              round(atr, 4),
            # SMC quality scores — used by scorer L1 and lifecycle manager
            **quality_meta,
        }
    except Exception as e:
        logger.error(f"[smc] Analysis failed for {ticker}: {e}")
        return None
