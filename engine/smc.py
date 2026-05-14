"""
Smart Money Concepts (SMC) analysis engine.
Fetches OHLCV from yfinance and detects:
  - Swing highs / swing lows
  - Break of Structure (BOS)
  - Change of Character (CHoCH)
  - Fair Value Gaps (FVG)
  - Order Blocks (OB)
"""

from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

SWING_WINDOW = 3


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_candles(ticker: str, period: str = "10d", interval: str = "1h") -> pd.DataFrame:
    """Fetch OHLCV from yfinance. Returns DataFrame with lowercase columns or empty DF."""
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        if df.empty or len(df) < 20:
            print(f"[smc] Insufficient data for {ticker}")
            return pd.DataFrame()

        # Flatten MultiIndex columns if present (older yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        df.columns = [c.lower() for c in df.columns]
        df = df.reset_index()

        # Normalise the datetime index column to "timestamp"
        for candidate in ("datetime", "date"):
            if candidate in df.columns:
                df = df.rename(columns={candidate: "timestamp"})
                break

        if "timestamp" not in df.columns:
            df["timestamp"] = df.index

        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    except Exception as e:
        print(f"[smc] yfinance error for {ticker}: {e}")
        return pd.DataFrame()


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
    sh = df[df["swing_high"]]["high"].values
    sl = df[df["swing_low"]]["low"].values
    price = df["close"].iloc[-1]

    bos_bullish = bos_bearish = False
    choch_bullish = choch_bearish = False

    if len(sh) >= 2:
        # BOS: price breaks the most recent confirmed swing high
        if price > sh[-1]:
            bos_bullish = True
        # CHoCH: previous highs were lower (downtrend) and now price breaks above
        if sh[-1] < sh[-2] and price > sh[-1]:
            choch_bullish = True

    if len(sl) >= 2:
        if price < sl[-1]:
            bos_bearish = True
        # CHoCH: previous lows were higher (uptrend) and now price breaks below
        if sl[-1] > sl[-2] and price < sl[-1]:
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

def detect_fvg(df: pd.DataFrame) -> dict:
    """
    Bullish FVG: candle[i-2].high < candle[i].low   — gap up, middle is impulse candle
    Bearish FVG: candle[i-2].low  > candle[i].high  — gap down
    Returns the FVG nearest to current price for each direction.
    """
    bullish_fvgs: list[dict] = []
    bearish_fvgs: list[dict] = []

    for i in range(2, len(df)):
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

def detect_order_blocks(df: pd.DataFrame) -> dict:
    """
    Bullish OB: last bearish candle before a strong bullish impulse (>0.3% body move)
    Bearish OB: last bullish candle before a strong bearish impulse
    Returns the most recently formed OB for each direction.
    """
    bullish_ob: Optional[dict] = None
    bearish_ob: Optional[dict] = None

    for i in range(1, len(df) - 1):
        curr = df.iloc[i]
        nxt = df.iloc[i + 1]

        if curr["close"] < curr["open"]:  # bearish candle
            body_move = (nxt["close"] - nxt["open"]) / nxt["open"] if nxt["open"] else 0
            if body_move > 0.003:
                bullish_ob = {
                    "top":    float(curr["open"]),
                    "bottom": float(curr["close"]),
                    "ts":     curr.get("timestamp"),
                }

        if curr["close"] > curr["open"]:  # bullish candle
            body_move = (nxt["open"] - nxt["close"]) / nxt["open"] if nxt["open"] else 0
            if body_move > 0.003:
                bearish_ob = {
                    "top":    float(curr["close"]),
                    "bottom": float(curr["open"]),
                    "ts":     curr.get("timestamp"),
                }

    return {"ob_bullish": bullish_ob, "ob_bearish": bearish_ob}


# ---------------------------------------------------------------------------
# Entry-level calculation
# ---------------------------------------------------------------------------

def _calculate_levels(direction: str, price: float, obs: dict, df: pd.DataFrame) -> tuple:
    swing_highs = df[df["swing_high"]]["high"].values
    swing_lows  = df[df["swing_low"]]["low"].values

    if direction == "LONG":
        ob    = obs.get("ob_bullish")
        entry = round(float(ob["top"]) if ob else price, 4)
        sl_ref = float(ob["bottom"]) if ob else (float(swing_lows[-1]) if len(swing_lows) else price * 0.985)
        stop_loss  = round(sl_ref * 0.9985, 4)          # just below OB bottom
        target_one = round(entry * 1.015, 4)             # tighter: +1.5%
        target_two = round(entry * 1.030, 4)             # tighter: +3%
        # Anchor T1 to nearest swing high if closer
        if len(swing_highs):
            sh = float(swing_highs[-1])
            if entry < sh < entry * 1.04:                # only if within 4%
                target_one = round(sh, 4)
    else:
        ob    = obs.get("ob_bearish")
        entry = round(float(ob["bottom"]) if ob else price, 4)
        sl_ref = float(ob["top"]) if ob else (float(swing_highs[-1]) if len(swing_highs) else price * 1.015)
        stop_loss  = round(sl_ref * 1.0015, 4)
        target_one = round(entry * 0.985, 4)             # tighter: -1.5%
        target_two = round(entry * 0.970, 4)             # tighter: -3%
        if len(swing_lows):
            sl = float(swing_lows[-1])
            if entry * 0.96 < sl < entry:
                target_one = round(sl, 4)

    return entry, stop_loss, target_one, target_two


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def analyze(ticker: str, interval: str = "1h") -> Optional[dict]:
    """Run full SMC analysis. Returns analysis dict (with 'candles' key) or None."""
    try:
        df = fetch_candles(ticker, period="10d", interval=interval)
        if df.empty:
            return None

        df = detect_swings(df)
        structure = detect_structure(df)
        fvgs = detect_fvg(df)
        obs = detect_order_blocks(df)

        price = float(df["close"].iloc[-1])
        avg_volume = float(df["volume"].mean())
        last_volume = float(df["volume"].iloc[-1])

        bullish_signals = sum([
            structure["bos_bullish"],
            structure["choch_bullish"],
            fvgs["fvg_bullish"] is not None,
            obs["ob_bullish"] is not None,
        ])
        bearish_signals = sum([
            structure["bos_bearish"],
            structure["choch_bearish"],
            fvgs["fvg_bearish"] is not None,
            obs["ob_bearish"] is not None,
        ])

        if bullish_signals == 0 and bearish_signals == 0:
            direction = None
        elif bullish_signals >= bearish_signals:
            direction = "LONG"
        else:
            direction = "SHORT"

        entry, stop_loss, target_one, target_two = (None, None, None, None)
        if direction:
            entry, stop_loss, target_one, target_two = _calculate_levels(direction, price, obs, df)

        return {
            "ticker":        ticker,
            "current_price": price,
            "direction":     direction,
            "entry":         entry,
            "stop_loss":     stop_loss,
            "target_one":    target_one,
            "target_two":    target_two,
            "structure":     structure,
            "fvgs":          fvgs,
            "obs":           obs,
            "avg_volume":    avg_volume,
            "last_volume":   last_volume,
            "candles":       df,
            "timeframe":     interval,
        }
    except Exception as e:
        print(f"[smc] Analysis failed for {ticker}: {e}")
        return None
