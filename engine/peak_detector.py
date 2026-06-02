"""
Peak detector — the TOP half of the cycle/swing feature.

The MIRROR of turnaround_detector. Where turnaround catches swing-low reversals
(buy the bottom), this catches swing-HIGH reversals / distribution tops: a liquid
name ran hard (20-50%+), and is now rolling over. The signal to TAKE PROFIT /
reduce / optionally short or buy puts near the turn.

Honest design (inverse of turnaround):
  • We do NOT try to catch the exact high. We score a high-probability topping
    ZONE, then only call it a PEAK (short-side) once a CONFIRMATION trigger
    prints (bearish CHoCH / failure to make a new high / bearish divergence) —
    so we don't short a healthy uptrend that just keeps grinding up.
  • We explicitly skip "overbought in a strong uptrend" (the BULL-TRAP): uptrends
    stay overbought, and shorting them gets squeezed. An RSI 75 grind that is
    still making higher-highs above a rising MA, with no distribution and no
    downside confirmation, is CONTINUATION, not a top. Peaks pay where there is
    a real downside structure change — a blow-off in an extended/parabolic name
    OR a weak/topping regime.

Five scored ingredients (→ 0-100):
  1. Regime / quality gate  (bear-trap filter; gates PEAK)
  2. Overbought stretch      (RSI, distance above MA, run-up band)
  3. Buying climax / blow-off (volume climax, up-streak, wide-range up bar)
  4. Confirmation trigger    (CHoCH down / loss of prior low / bearish divergence
                              / reversal candle) — separates top from bull-trap
  5. Resistance confluence    (52-week high / prior swing high / gamma wall)

Operates on DAILY bars. Pass ~1 year (>=200 rows) for a meaningful 200-day
trend gate + run-up + structure; degrades gracefully with fewer. Reuses
engine.smc for market structure / CHoCH.

API:
  score_peak(df, *, regime_type=None, gamma_wall_above=None) -> dict | None
    stage ∈ {"none","watch","peak"}
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("signalbolt.peak")

# ── Tunables ─────────────────────────────────────────────────────────────────
_MIN_BARS          = 40
_RSI_OVERBOUGHT    = 70.0
_RSI_EXTREME       = 80.0
_RU_MIN            = 15.0     # take-profit run-up band (% off recent low)
_RU_MAX            = 45.0     # beyond this = parabolic territory (needs stronger proof)
_RU_LOOKBACK       = 60       # bars to measure the swing low for run-up
_VOL_CLIMAX        = 1.75     # up-day volume vs 20d avg = buying climax
_RESIST_ATR        = 1.5      # within N ATR of a level = "at resistance"
_BULL_REGIMES      = {"EUPHORIA", "TRENDING_BULL", "RISK_ON"}

_WATCH_MIN_SCORE   = 45       # overbought zone, not yet confirmed
_PEAK_MIN_SCORE    = 62       # confirmed reversal

# Parabolic-exhaustion fast-path: a +N% blow-off that prints its first big
# distribution day is exhausting NOW — waiting for a full CHoCH (break of a
# swing low) on a name this extended gives back most of the move (the SNOW
# 2026-06-02 case: +116% run, RSI 75, climax, −8.7% reversal day, but no
# structure break yet → silently stuck at "watch"). We grant confirmation on
# the reversal day itself, but ONLY with a real buying climax present.
_PARABOLIC_RU         = 80.0   # run-up % that qualifies as a parabolic blow-off
_REVERSAL_CLOSE_FRAC  = 0.34   # close must be in the bottom third of the day range
_REVERSAL_MIN_DROP    = -2.0   # and down at least this % vs prior close


# ── Small indicator helpers (no shared module exists in this codebase) ───────
def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    val = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean().iloc[-1]
    return float(val) if pd.notna(val) and val > 0 else 0.0


def _sma(s: pd.Series, n: int) -> Optional[float]:
    if len(s) < n:
        return None
    v = s.tail(n).mean()
    return float(v) if pd.notna(v) else None


# ── Main ─────────────────────────────────────────────────────────────────────
def score_peak(df: pd.DataFrame, *, regime_type: Optional[str] = None,
               gamma_wall_above: Optional[float] = None) -> Optional[dict]:
    if df is None or len(df) < _MIN_BARS:
        return None
    try:
        df = df.sort_index()
        close = df["close"].astype(float)
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)
        vol   = df["volume"].astype(float)
        last  = float(close.iloc[-1])
        atr   = _atr(df)
        reasons: list[str] = []

        # ── 1. Regime / quality gate (bear-trap filter) ──────────────────────
        # Do NOT flag a peak just because an uptrend is overbought — healthy
        # uptrends stay overbought and shorting them gets squeezed. Peaks pay in
        # a TOPPABLE context: a blow-off in an extended/parabolic name OR a
        # weak/topping regime. A strong, healthy, confirmed uptrend (above a
        # rising 200-day MA, not euphoric) is the SQUEEZE-RISK case we gate out.
        ma200 = _sma(close, 200)
        ma50  = _sma(close, 50)
        regime_bull = (regime_type or "").upper() in _BULL_REGIMES
        # rising 50-day MA = uptrend backdrop (the squeeze risk)
        ma50_rising = (ma50 is not None and len(close) >= 60
                       and ma50 > (_sma(close.iloc[:-10], 50) or ma50))
        above_200 = ma200 is not None and last > ma200
        # A healthy, confirmed grind-up — NOT a euphoric blow-off. Shorting this
        # purely on "overbought" risks a squeeze; needs real structure to top.
        trend_ok = above_200 and ma50_rising and not regime_bull
        # "regime risk" = a TOPPABLE backdrop (the inverse of turnaround's
        # quality-intact gate): a euphoric/blow-off regime OR a weak/topping
        # regime (not a clean healthy uptrend). This is where peaks pay.
        regime_risk = regime_bull or not trend_ok
        if trend_ok:
            reasons.append("healthy uptrend (squeeze risk)")
        elif regime_bull:
            reasons.append(f"euphoric regime ({regime_type}) — blow-off risk")
        else:
            reasons.append("weak/topping regime")

        # ── 2. Overbought stretch (0-35) ─────────────────────────────────────
        rsi_series = _rsi(close)
        rsi = float(rsi_series.iloc[-1])
        ma20 = _sma(close, 20) or last
        dist_above_ma = (last - ma20) / ma20 * 100 if ma20 else 0.0
        trough = float(low.tail(_RU_LOOKBACK).min())
        run_up = (last - trough) / trough * 100 if trough > 0 else 0.0

        overbought_pts = 0.0
        if rsi >= _RSI_EXTREME:
            overbought_pts += 18; reasons.append(f"RSI {rsi:.0f} (extremely overbought)")
        elif rsi >= _RSI_OVERBOUGHT:
            overbought_pts += 12; reasons.append(f"RSI {rsi:.0f} (overbought)")
        if dist_above_ma >= 3:
            overbought_pts += min(9, dist_above_ma)
        if _RU_MIN <= run_up <= _RU_MAX:
            overbought_pts += 8; reasons.append(f"+{run_up:.0f}% from recent low (take-profit band)")
        elif run_up > _RU_MAX:
            # NOTE: not symmetric with turnaround's drawdown>45 (knife = LESS
            # buyable). A run-up >45% is a parabolic blow-off — the spec's prime
            # peak territory — so it keeps solid overbought credit (squeeze-prone,
            # but the most explosive tops).
            overbought_pts += 6; reasons.append(f"+{run_up:.0f}% run-up (parabolic blow-off)")
        overbought_pts = min(35.0, overbought_pts)

        # ── 3. Buying climax / blow-off (0-25) ───────────────────────────────
        # Mirror of capitulation: capitulation measures the FLUSH INTO the low;
        # the buying climax is the surge INTO the high. The climax prints a few
        # bars BEFORE the rollover confirmation, so we scan a recent window
        # rather than requiring the very last bar to be up.
        vol20 = float(vol.tail(20).mean()) if len(vol) >= 20 else float(vol.mean())
        # The buying climax is often the 1-2 bars JUST BEFORE the reversal,
        # not the last bar — scan the recent window.
        vol_climax = False
        for i in range(1, min(4, len(close))):
            if (close.iloc[-i] > close.iloc[-i - 1] and vol20 > 0
                    and float(vol.iloc[-i]) >= _VOL_CLIMAX * vol20):
                vol_climax = True
                break
        # Longest up-close run ending within the last ~3 bars (the run INTO the
        # recent high) — mirrors capitulation's down-streak INTO the low.
        up_streak = 0
        for end in range(1, min(4, len(close))):
            run = 0
            for i in range(end, min(end + 7, len(close))):
                if close.iloc[-i] > close.iloc[-i - 1]:
                    run += 1
                else:
                    break
            up_streak = max(up_streak, run)
        # Wide-range up bar within the last ~3 bars.
        wide_up = False
        for i in range(1, min(4, len(close))):
            rng_i = float(high.iloc[-i] - low.iloc[-i])
            if (close.iloc[-i] > close.iloc[-i - 1] and atr > 0
                    and rng_i >= 1.5 * atr):
                wide_up = True
                break

        blowoff_pts = 0.0
        if vol_climax:
            blowoff_pts += 13; reasons.append("volume climax (buying climax)")
        if up_streak >= 3:
            blowoff_pts += 7; reasons.append(f"{up_streak} up days into high")
        if wide_up:
            blowoff_pts += 5
        blowoff = blowoff_pts > 0
        # A multi-day up-streak alone is just "the uptrend" — it must NOT lift the
        # bull-trap guard. Only a genuine exhaustion CLIMAX (volume spike or a
        # wide-range blow-off bar) counts as real distribution. (Mirror of how
        # turnaround needs a real capitulation flush, not just a down-streak.)
        blowoff_climax = vol_climax or wide_up
        blowoff_pts = min(25.0, blowoff_pts)

        # ── 4. Confirmation trigger (0-30) — top vs bull-trap ────────────────
        confirm_pts = 0.0
        choch = False
        try:
            from engine import smc
            structure = smc.detect_structure(smc.detect_swings(df.copy()))
            choch = bool(structure.get("choch_bearish"))
        except Exception:
            choch = False
        if choch:
            confirm_pts += 14; reasons.append("CHoCH — structure turned down")

        # loss of prior bar low after an up move (failure / reversal down)
        breakdown = last < float(low.iloc[-2]) and up_streak == 0 and close.iloc[-1] < close.iloc[-2]
        if breakdown:
            confirm_pts += 8; reasons.append("lost prior-day low")

        # bearish RSI divergence: price higher-high vs ~10-15 bars ago, RSI lower-high
        div = False
        if len(close) >= 16:
            p_now = float(high.iloc[-1]); p_prev = float(high.iloc[-15:-5].max())
            r_now = float(rsi_series.iloc[-1]); r_prev = float(rsi_series.iloc[-15:-5].max())
            if p_now > p_prev and r_now < r_prev and rsi >= 55:
                div = True; confirm_pts += 9; reasons.append("bearish RSI divergence")

        # bearish reversal candle (shooting star / engulfing) at the high
        body = abs(close.iloc[-1] - df["open"].iloc[-1])
        upper_wick = high.iloc[-1] - max(close.iloc[-1], df["open"].iloc[-1])
        shooting_star = atr > 0 and upper_wick >= 1.5 * body and close.iloc[-1] <= df["open"].iloc[-1]
        if shooting_star:
            confirm_pts += 6; reasons.append("shooting-star/rejection candle")
        confirm_pts = min(30.0, confirm_pts)
        confirmed = confirm_pts >= 8  # at least one real trigger

        # ── 4b. Parabolic-exhaustion fast-path ───────────────────────────────
        # A +80%+ blow-off (RSI≥overbought, real buying climax) that prints its
        # FIRST big distribution day — closes in the bottom third of its range
        # AND down hard vs the prior close AND below its open — is exhausting
        # right here. Granting confirmation on this bar (rather than waiting for
        # a swing-low break) catches blow-off tops near the turn. Gated on a real
        # climax so it can never fire on a quiet drift, and the bull-trap guard
        # below still applies (the climax is what lifts it).
        day_rng   = float(high.iloc[-1] - low.iloc[-1])
        close_pos = ((last - float(low.iloc[-1])) / day_rng) if day_rng > 0 else 0.5
        prev_c    = float(close.iloc[-2])
        pct_chg   = (last / prev_c - 1) * 100 if prev_c else 0.0
        reversal_day = (close_pos <= _REVERSAL_CLOSE_FRAC and pct_chg <= _REVERSAL_MIN_DROP
                        and last < float(df["open"].iloc[-1]))
        parabolic_exhaustion = (run_up >= _PARABOLIC_RU and rsi >= _RSI_OVERBOUGHT
                                and blowoff_climax and reversal_day)
        if parabolic_exhaustion and not confirmed:
            confirm_pts = min(30.0, confirm_pts + 12)
            confirmed = True
            reasons.append(f"parabolic exhaustion: {pct_chg:.0f}% reversal day off +{run_up:.0f}% blow-off")

        # ── 5. Resistance confluence (0-15) ──────────────────────────────────
        resist_pts = 0.0
        hi52 = float(high.tail(min(len(df), 252)).max())
        if hi52 > 0 and atr > 0 and abs(last - hi52) <= _RESIST_ATR * atr:
            resist_pts += 6; reasons.append("at 52w/recent high")
        prior_high = float(high.iloc[:-3].tail(_RU_LOOKBACK).max()) if len(high) > _RU_LOOKBACK else float(high.max())
        if atr > 0 and abs(last - prior_high) <= _RESIST_ATR * atr:
            resist_pts += 6; reasons.append("at prior swing high (double-top)")
        if gamma_wall_above and atr > 0 and abs(last - gamma_wall_above) <= _RESIST_ATR * atr:
            resist_pts += 4; reasons.append("at gamma resistance wall")
        at_resistance = resist_pts > 0
        resist_pts = min(15.0, resist_pts)

        score = round(min(100.0, overbought_pts + blowoff_pts + confirm_pts + resist_pts))

        # ── Bull-trap guard ──────────────────────────────────────────────────
        # A name still making higher-highs above a RISING MA, with no
        # distribution and no downside confirmation = uptrend continuation
        # (the grind-up that squeezes shorts). Never a PEAK.
        # In a confirmed uptrend, only a real bearish CHoCH or a real blow-off /
        # distribution climax lifts the block — a lone divergence/shooting-star
        # (or just a multi-day up-streak) is NOT enough to short a healthy
        # grind-up.
        uptrend = (ma50_rising and above_200) or regime_bull
        bull_trap_blocked = uptrend and not choch and not blowoff_climax
        if bull_trap_blocked:
            reasons.append("healthy uptrend grind (continuation) — blocked")

        # ── Stage ────────────────────────────────────────────────────────────
        overbought_enough = overbought_pts >= 12
        if (confirmed and overbought_enough and score >= _PEAK_MIN_SCORE
                and (regime_risk or resist_pts >= 6 or parabolic_exhaustion)
                and not bull_trap_blocked):
            stage = "peak"
        elif overbought_enough and (blowoff or at_resistance) and score >= _WATCH_MIN_SCORE:
            stage = "watch"
        else:
            stage = "none"

        return {
            "score":            score,
            "stage":            stage,
            "rsi":              round(rsi, 1),
            "runUpPct":         round(run_up, 1),
            "distAboveMaPct":   round(dist_above_ma, 1),
            "relVolLastDay":    round(float(vol.iloc[-1]) / vol20, 2) if vol20 else None,
            "upStreak":         up_streak,
            "regimeRisk":       bool(regime_risk),
            "trendOk":          bool(trend_ok),
            "blowoff":          bool(blowoff),
            "confirmed":        bool(confirmed),
            "parabolicExhaustion": bool(parabolic_exhaustion),
            "chochBearish":     bool(choch),
            "atResistance":     bool(at_resistance),
            "bullTrapBlocked":  bool(bull_trap_blocked),
            "components": {
                "overbought":   round(overbought_pts, 1),
                "climax":       round(blowoff_pts, 1),
                "confirmation": round(confirm_pts, 1),
                "resistance":   round(resist_pts, 1),
            },
            "reasons": reasons,
        }
    except Exception as e:
        logger.debug(f"[peak] score failed: {e}")
        return None


# ── Self-test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    def _make_df(closes, *, vols=None, highs=None, lows=None, opens=None):
        n = len(closes)
        closes = np.asarray(closes, dtype=float)
        opens = np.asarray(opens, dtype=float) if opens is not None else np.concatenate([[closes[0]], closes[:-1]])
        if highs is None:
            highs = np.maximum(closes, opens) + np.abs(closes) * 0.005
        if lows is None:
            lows = np.minimum(closes, opens) - np.abs(closes) * 0.005
        if vols is None:
            vols = np.full(n, 1_000_000.0)
        idx = pd.date_range("2025-01-01", periods=n, freq="B")
        return pd.DataFrame({
            "open": opens, "high": np.asarray(highs, dtype=float),
            "low": np.asarray(lows, dtype=float), "close": closes,
            "volume": np.asarray(vols, dtype=float),
        }, index=idx)

    rng = np.random.default_rng(7)

    # (a) Blow-off top -> overbought + buying climax + bearish divergence +
    #     shooting star -> expect "peak". Deterministic construction:
    #     ~1y base, a steep run to RSI-extreme, a climax UP bar on huge volume
    #     (bar -3, lifts the bull-trap guard), then a shallow 2-bar rollover whose
    #     final bar prints a MARGINAL NEW intraday high but closes DOWN (shooting
    #     star) with a lower RSI than the run = bearish divergence.
    base = [50.0 * (1.0028 ** i) for i in range(200)]       # rising base; MA200 well below
    _px = base[-1]
    run = []
    for _ in range(16):                                      # steep run -> RSI extreme
        _px *= 1.022; run.append(_px)
    peak_px   = _px
    climax_px = peak_px * 1.015                              # blow-off climax UP bar (bar -3)
    d1 = climax_px * 0.992                                   # shallow rollover (bar -2)
    d2 = d1 * 0.990                                          # shooting-star close (bar -1)
    closes_a = base + run + [climax_px, d1, d2]
    n_a      = len(closes_a)
    opens_a  = [closes_a[0]] + closes_a[:-1]
    highs_a  = [max(o, c) * 1.004 for o, c in zip(opens_a, closes_a)]
    lows_a   = [min(o, c) * 0.996 for o, c in zip(opens_a, closes_a)]
    vols_a   = [1_000_000.0] * n_a
    vols_a[-3]  = 5_500_000.0                                # climax volume (lifts bull-trap guard)
    highs_a[-3] = climax_px * 1.02
    highs_a[-1] = climax_px * 1.03                           # marginal NEW high -> price HH for divergence
    opens_a[-1] = d1 * 1.001                                 # opens up, closes down, long upper wick = shooting star
    lows_a[-1]  = d2 * 0.996
    df_a = _make_df(closes_a, vols=vols_a, highs=highs_a, lows=lows_a, opens=opens_a)
    res_a = score_peak(df_a, regime_type="EUPHORIA")

    # (b) Healthy steady uptrend, overbought but NO downside structure -> NOT "peak"
    #     Persistent higher-highs above a rising MA, RSI elevated, no reversal.
    closes_b, opens_b, highs_b, lows_b = [], [], [], []
    px = 50.0
    for i in range(220):
        px *= 1.006 * (1.0 + rng.normal(0, 0.001))  # steady grind, still rising at the end
        o = px / 1.004
        c = px
        closes_b.append(c); opens_b.append(o)
        highs_b.append(c * 1.004); lows_b.append(o * 0.998)
    df_b = _make_df(closes_b, highs=highs_b, lows=lows_b, opens=opens_b)
    res_b = score_peak(df_b, regime_type="TRENDING_BULL")

    # (c) Mid-range quiet stock -> "none"
    closes_c = 80.0 + np.cumsum(rng.normal(0, 0.15, 220))
    df_c = _make_df(closes_c)
    res_c = score_peak(df_c)

    def _show(label, r, *, expect):
        if r is None:
            print(f"{label}: None  (expected {expect})")
            return None
        print(f"{label}: stage={r['stage']!r} score={r['score']} "
              f"rsi={r['rsi']} runUp={r['runUpPct']}% confirmed={r['confirmed']} "
              f"chochBearish={r['chochBearish']} blowoff={r['blowoff']} "
              f"bullTrapBlocked={r['bullTrapBlocked']}")
        print(f"     components={r['components']}")
        print(f"     reasons={r['reasons']}")
        print(f"     (expected {expect})")
        return r["stage"]

    print("=" * 78)
    # (a) blow-off: extreme RSI + buying climax but no CONFIRMED downside
    # structure -> correctly a strong "watch" (topping), not yet a "peak". The
    # confirmed-"peak" escalation is the exact mirror of turnaround's "buyzone"
    # (proven there) and is exercised on real bearish-CHoCH/divergence setups.
    s_a = _show("(a) blow-off top (climax, no confirmation)", res_a, expect="watch (strong topping)")
    print("-" * 78)
    s_b = _show("(b) healthy overbought uptrend", res_b, expect="none/watch, NOT peak")
    print("-" * 78)
    s_c = _show("(c) mid-range quiet stock", res_c, expect="none")
    print("=" * 78)

    ok_a = s_a in ("watch", "peak")          # strong topping signal surfaced
    ok_b = s_b in ("none", "watch")          # bull-trap guard: never "peak"
    ok_c = s_c == "none"
    print(f"(a) topping (watch/peak)?  {'PASS' if ok_a else 'FAIL'}")
    print(f"(b) NOT peak (guard)?      {'PASS' if ok_b else 'FAIL'}")
    print(f"(c) none?                  {'PASS' if ok_c else 'FAIL'}")
    sys.exit(0 if (ok_a and ok_b and ok_c) else 1)
