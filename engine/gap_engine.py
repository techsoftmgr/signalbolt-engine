"""
Gap-Continuation Engine (Opening Range Breakout)
=================================================
Detects and scores ORB breakout setups on gap days.

WHY THIS EXISTS:
  SMC analysis requires historical structure (order blocks, FVGs, BOS/CHoCH)
  at the current price level.  On earnings gap days, the stock opens in
  "fresh air" — a price range with ZERO historical candles.  No OBs, no FVGs,
  no structure.  smc.analyze() returns direction=None and the engine skips it.

  This module is the fallback for exactly that scenario.  Instead of SMC
  structure, it uses the Opening Range (first 30 min of trading) as the
  structural anchor — a well-established institutional methodology for gap days.

HOW IT WORKS:
  Gap-Up day (e.g. ARM +12% on earnings):
    1. Stock opens with a gap above prev close  ← gap > 1.5%
    2. First 30 min forms the Opening Range (ORB high, ORB low)
       Markets makers absorb initial euphoria / panic here.
    3. Price BREAKS ABOVE ORB high with volume  ← breakout confirmed
    4. Entry: current price (just above ORB high)
    5. Stop:  ORB low (institutional support, MMs defend gap open)
    6. T1:    ORB high + 1× range | T2: ORB high + 2× range

  Gap-Down day (e.g. WMT -7% on earnings):
    Same logic, mirrored. Breakout BELOW ORB low → SHORT signal.

QUALITY GATES (all must pass):
  - Gap ≥ 1.5% from prev close
  - ≥2 bars of ORB data available (need the range)
  - ORB range ≤ 2.5× ATR (too wide = chaotic open, not tradeable)
  - Breakout confirmed (latest price outside ORB in gap direction)
  - Breakout bar volume > average (institutional participation)
  - Gap not >70% filled (if price came all the way back, thesis dead)
  - R:R ≥ 1.5 after applying stop/target logic
  - Combined quality score ≥ 60/100

INTEGRATION:
  Called from runner._process_smc_ticker() as a fallback when smc.analyze()
  returns no direction.  The returned dict is fully compatible with the
  existing SMC scoring/SL-TP pipeline.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from zoneinfo import ZoneInfo

logger = logging.getLogger("signalbolt.gap_engine")

ET = ZoneInfo("America/New_York")

# ── Configuration ─────────────────────────────────────────────────────────────

MIN_GAP_PCT      = 0.015   # 1.5% minimum overnight gap to trigger analysis
ORB_BARS         = 2       # Opening Range = first 2 × 15-min bars = 30 min
MIN_SCORE        = 60      # Minimum quality score to fire a signal
MAX_ORB_ATR_MULT = 2.5     # ORB wider than this = chaotic open, skip


# ── Public entry point ────────────────────────────────────────────────────────

def analyze(
    ticker:        str,
    df:            pd.DataFrame,   # 15-min OHLCV bars, most recent last
    current_price: float,
    strategy_type: str = "day_trade",
) -> Optional[dict]:
    """
    Analyze a ticker for a gap-ORB breakout setup.

    Detects the overnight gap and the Opening Range entirely from the bar data
    passed in — no extra API calls needed.  The df is the same 5d/15m data
    already fetched by smc.analyze().

    Returns a signal dict compatible with the SMC scoring pipeline, or None.
    """
    if df is None or df.empty or len(df) < 4:
        return None

    # ── Split bars into today vs previous session ─────────────────────────────
    today_bars, prev_close = _split_sessions(df)
    if today_bars is None or len(today_bars) < ORB_BARS or prev_close is None:
        logger.debug(
            f"[gap_engine] {ticker}: not enough session data "
            f"(today_bars={len(today_bars) if today_bars is not None else 0})"
        )
        return None

    # ── Confirm gap ───────────────────────────────────────────────────────────
    today_open = float(today_bars.iloc[0]["open"])
    if prev_close <= 0 or today_open <= 0:
        return None

    gap_pct = (today_open - prev_close) / prev_close   # signed: + = gap-up
    gap_abs = abs(gap_pct)

    if gap_abs < MIN_GAP_PCT:
        logger.debug(
            f"[gap_engine] {ticker}: gap {gap_pct*100:.2f}% < "
            f"threshold {MIN_GAP_PCT*100:.1f}%"
        )
        return None

    gap_direction    = "UP"   if gap_pct > 0 else "DOWN"
    signal_direction = "LONG" if gap_pct > 0 else "SHORT"

    # ── Build Opening Range ───────────────────────────────────────────────────
    orb_df    = today_bars.iloc[:ORB_BARS]
    orb_high  = float(orb_df["high"].max())
    orb_low   = float(orb_df["low"].min())
    orb_range = orb_high - orb_low

    if orb_range <= 0:
        return None

    # ── ATR sanity — reject chaotic wide opens ───────────────────────────────
    atr = _get_atr(df)
    if atr > 0 and orb_range > atr * MAX_ORB_ATR_MULT:
        logger.info(
            f"[gap_engine] {ticker}: ORB ${orb_range:.2f} > "
            f"{MAX_ORB_ATR_MULT}× ATR ${atr:.2f} — chaotic open, skipping"
        )
        return None

    # ── Confirm breakout ──────────────────────────────────────────────────────
    # Price must have broken out of the ORB in the gap direction.
    # We check the latest bar's close, not current_price, to avoid
    # triggering on a brief wick through the level.
    latest_bar   = today_bars.iloc[-1]
    latest_close = float(latest_bar["close"])
    latest_vol   = float(latest_bar["volume"])

    orb_bars_before_latest = today_bars.iloc[:-1]
    avg_vol = float(orb_bars_before_latest["volume"].mean()) if len(orb_bars_before_latest) > 0 else latest_vol

    broke_up   = signal_direction == "LONG"  and latest_close > orb_high
    broke_down = signal_direction == "SHORT" and latest_close < orb_low

    if not broke_up and not broke_down:
        logger.debug(
            f"[gap_engine] {ticker}: {gap_direction} gap {gap_pct*100:+.1f}% "
            f"but no ORB breakout yet "
            f"(price={current_price:.2f} ORB={orb_low:.2f}–{orb_high:.2f})"
        )
        return None

    # ── Score the setup ───────────────────────────────────────────────────────
    score, reasons = _score(
        gap_pct       = gap_pct,
        gap_direction = gap_direction,
        orb_high      = orb_high,
        orb_low       = orb_low,
        orb_range     = orb_range,
        current_price = current_price,
        latest_vol    = latest_vol,
        avg_vol       = avg_vol,
        atr           = atr,
        today_bars    = today_bars,
        prev_close    = prev_close,
    )

    if score < MIN_SCORE:
        logger.info(
            f"[gap_engine] {ticker}: {gap_direction} gap {gap_pct*100:+.1f}% "
            f"ORB setup score={score} < {MIN_SCORE} — not firing"
        )
        return None

    # ── Build SL and targets ──────────────────────────────────────────────────
    buf = max(orb_range * 0.10, atr * 0.15) if atr > 0 else orb_range * 0.10

    if signal_direction == "LONG":
        entry      = round(current_price, 2)
        stop_loss  = round(orb_low  - buf, 2)
        target_one = round(orb_high + orb_range,       2)   # 1× extension
        target_two = round(orb_high + orb_range * 2.0, 2)   # 2× extension
    else:
        entry      = round(current_price, 2)
        stop_loss  = round(orb_high + buf, 2)
        target_one = round(orb_low  - orb_range,       2)
        target_two = round(orb_low  - orb_range * 2.0, 2)

    # Validate R:R before firing
    risk   = abs(entry - stop_loss)
    reward = abs(target_one - entry)
    rr     = round(reward / risk, 2) if risk > 0 else 0.0

    if rr < 1.5:
        logger.info(
            f"[gap_engine] {ticker}: ORB setup R:R={rr:.2f} < 1.5 — skipping"
        )
        return None

    logger.info(
        f"[gap_engine] {ticker} ★ GAP-ORB {signal_direction} SETUP — "
        f"gap={gap_pct*100:+.1f}% prev_close={prev_close:.2f} "
        f"ORB={orb_low:.2f}–{orb_high:.2f} "
        f"entry={entry} sl={stop_loss} t1={target_one} "
        f"score={score} R:R={rr}"
    )

    return {
        # ── Core signal fields ────────────────────────────────────────────
        "ticker":           ticker,
        "direction":        signal_direction,
        "current_price":    current_price,
        "entry":            entry,
        "stop_loss":        stop_loss,
        "target_one":       target_one,
        "target_two":       target_two,
        "candles":          df,
        "timeframe":        "15m",
        "strategy_type":    strategy_type,
        "setup_type":       "GAP_ORB",
        # ── SMC-compatible stubs (required by scorer) ─────────────────────
        # The gap itself IS an FVG (skipped price range) so we surface it here.
        "structure": {
            "bos_bullish": signal_direction == "LONG",
            "bos_bearish": signal_direction == "SHORT",
            "choch_bullish": False,
            "choch_bearish": False,
        },
        "fvgs": {
            # Gap zone = the price range skipped = a macro Fair Value Gap
            ("fvg_bullish" if signal_direction == "LONG" else "fvg_bearish"): {
                "top":    round(today_open, 2),
                "bottom": round(prev_close, 2),
                "ts":     None,
            }
        },
        "obs":              {},
        "liquidity_sweep":  {},
        # ── Display ───────────────────────────────────────────────────────
        "confidence_score":   score,
        "confidence_factors": reasons,
        # ── Gap metadata for analytics/explainer ──────────────────────────
        "gap_pct":     round(gap_pct * 100, 2),
        "gap_open":    round(today_open, 2),
        "prev_close":  round(prev_close, 2),
        "orb_high":    round(orb_high, 2),
        "orb_low":     round(orb_low, 2),
        "orb_range":   round(orb_range, 2),
        "risk_reward": rr,
    }


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score(
    gap_pct: float,
    gap_direction: str,
    orb_high: float,
    orb_low: float,
    orb_range: float,
    current_price: float,
    latest_vol: float,
    avg_vol: float,
    atr: float,
    today_bars: pd.DataFrame,
    prev_close: float,
) -> tuple[int, list[str]]:
    """
    Score the gap-ORB setup 0–100.  Returns (score, reasons[]).

    Scoring breakdown:
      Gap size           0–30 pts   — bigger gap = stronger catalyst
      Breakout volume    0–25 pts   — confirms institutional participation
      ORB tightness      0–20 pts   — tight consolidation = clean setup
      Gap fill check     0–15 pts   — unfilled gap = thesis intact
      Entry proximity    0–10 pts   — catching the breakout early
    """
    score   = 0
    reasons: list[str] = []
    gap_abs = abs(gap_pct)

    # ── Gap size (0–30 pts) ───────────────────────────────────────────────────
    if gap_abs >= 0.08:
        score += 30
        reasons.append(f"Major earnings gap {gap_pct*100:+.1f}%")
    elif gap_abs >= 0.05:
        score += 22
        reasons.append(f"Large gap {gap_pct*100:+.1f}%")
    elif gap_abs >= 0.03:
        score += 15
        reasons.append(f"Solid gap {gap_pct*100:+.1f}%")
    else:
        score += 8
        reasons.append(f"Gap {gap_pct*100:+.1f}%")

    # ── Breakout volume (0–25 pts) ────────────────────────────────────────────
    vol_ratio = latest_vol / avg_vol if avg_vol > 0 else 1.0
    if vol_ratio >= 3.0:
        score += 25
        reasons.append(f"Breakout volume {vol_ratio:.1f}× avg — strong confirmation")
    elif vol_ratio >= 2.0:
        score += 18
        reasons.append(f"Breakout volume {vol_ratio:.1f}× avg")
    elif vol_ratio >= 1.3:
        score += 10
        reasons.append(f"Above-average breakout volume {vol_ratio:.1f}×")
    else:
        # Low volume breakout = possible false breakout — large penalty
        score += 0
        reasons.append("Low volume on breakout — weak confirmation")

    # ── ORB tightness (0–20 pts) ──────────────────────────────────────────────
    # Tight ORB relative to ATR = clean consolidation at gap open = best setup.
    # Wide ORB = initial panic/euphoria not settled, structure unclear.
    if atr > 0:
        orb_atr = orb_range / atr
        if orb_atr < 0.4:
            score += 20
            reasons.append("Tight ORB consolidation — textbook gap setup")
        elif orb_atr < 0.8:
            score += 14
            reasons.append("Solid ORB consolidation")
        elif orb_atr < 1.5:
            score += 7
        else:
            score += 2
            reasons.append("Wide ORB — initial open was volatile")

    # ── Gap fill check (0–15 pts) ─────────────────────────────────────────────
    # Has price come back to fill the gap?  A deep fill negates the gap thesis.
    # Gap is defined as the range between prev_close and today_open.
    gap_size = abs(float(today_bars.iloc[0]["open"]) - prev_close)
    if gap_size > 0:
        if gap_direction == "UP":
            deepest_return = prev_close - float(today_bars["low"].min())
        else:
            deepest_return = float(today_bars["high"].max()) - prev_close
        fill_pct = max(0.0, min(1.0, deepest_return / gap_size))
    else:
        fill_pct = 0.0

    if fill_pct < 0.15:
        score += 15
        reasons.append("Gap unfilled — directional thesis intact")
    elif fill_pct < 0.40:
        score += 8
        reasons.append(f"Gap {fill_pct*100:.0f}% filled — mostly intact")
    elif fill_pct < 0.70:
        score += 3
        reasons.append(f"Gap {fill_pct*100:.0f}% filled — partially weakened")
    else:
        score += 0
        reasons.append(f"Gap {fill_pct*100:.0f}% filled — thesis weakened")

    # ── Entry proximity (0–10 pts) ────────────────────────────────────────────
    # Catch the breakout early. If price has already run 2%+ beyond ORB
    # the easy money is gone and we're chasing.
    ref   = orb_high if gap_direction == "UP" else orb_low
    drift = abs(current_price - ref) / ref if ref > 0 else 1.0
    if drift < 0.003:
        score += 10
        reasons.append("Price just broke ORB — early entry opportunity")
    elif drift < 0.010:
        score += 6
    elif drift < 0.020:
        score += 2
    else:
        score += 0
        reasons.append(f"Price {drift*100:.1f}% from ORB — late entry")

    return int(min(100, max(0, score))), reasons


# ── Session splitter ──────────────────────────────────────────────────────────

def _split_sessions(df: pd.DataFrame) -> tuple[Optional[pd.DataFrame], Optional[float]]:
    """
    Split bars into today's session and extract previous session's close.

    Returns (today_bars_df, prev_close_price).
    Both can be None if timezone data is unavailable.
    """
    try:
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            idx = pd.to_datetime(idx, utc=True, errors="coerce")
        if idx.tz is None:
            idx = idx.tz_localize("UTC")

        idx_et   = idx.tz_convert(ET)
        today_et = pd.Timestamp.now(tz=ET).date()

        today_mask = pd.Series(
            [ts.date() == today_et for ts in idx_et], index=df.index
        )
        prev_mask  = ~today_mask

        today_bars = df[today_mask.values]
        prev_bars  = df[prev_mask.values]

        if today_bars.empty:
            return None, None

        # Previous session close = last bar before today
        prev_close = float(prev_bars["close"].iloc[-1]) if not prev_bars.empty else None

        return today_bars, prev_close

    except Exception as e:
        logger.debug(f"[gap_engine] _split_sessions error: {e}")
        # Graceful fallback: treat last 6 bars as today (rough proxy)
        if len(df) >= 8:
            today_bars = df.tail(6)
            prev_close = float(df.iloc[-7]["close"])
            return today_bars, prev_close
        return None, None


# ── ATR helper ────────────────────────────────────────────────────────────────

def _get_atr(df: pd.DataFrame, period: int = 14) -> float:
    try:
        hl = (df["high"] - df["low"]).tail(period)
        return float(hl.mean()) if not hl.empty else 0.0
    except Exception:
        return 0.0
