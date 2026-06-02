"""
Breakout → tradeable cards (LONG equity + CALL option).

Bullish mirror of breakdown_signals. When a name CONFIRMS a breakout (breaks its
20-day high on volume) we generate the actionable long play:
  • a LONG equity signal  (signals,        strategy_type='breakout', LONG)
  • a CALL option signal   (option_signals, via options_scanner call scan)

Levels (equity long):
  entry ≈ current price · stop just below the breakout level (-1.5 ATR) ·
  targets at +1.5 ATR (T1) / +3 ATR (T2). The CALL is priced + filtered by
  options_scanner (Polygon→yfinance chain + Black-Scholes + liquidity/IV gates).

Fires + pushes immediately and is tracked by signal_monitor (LONG + option
lifecycle). Tagged detector_source='BREAKOUT' for win-rate measurement.

Best-effort: never raises into the alert loop. Deduped by the DB
unique-active-signal indexes + the option active-check.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("signalbolt.breakout_signals")

# Stop-width guards. These detector cards bypass sl_tp_engine (which caps SL%),
# so without this a post-parabolic name with a fat ATR gets an absurd stop —
# e.g. MRVL ran +46% off its 20-day MA, ATR≈6%, so a flat 1.5·ATR stop was
# −10% from entry. Clamp every stop into a sane [MIN, MAX]% band off entry.
_MAX_STOP_PCT = 0.05    # never risk more than 5% on a 0.25x detector card
_MIN_STOP_PCT = 0.015   # …but at least 1.5%, so normal noise can't nick us


def _conf(r: dict) -> int:
    """Confidence in the LONG, from the breakout's strength (breakout/volume)."""
    bq = float(r.get("breakoutScore") or r.get("breakoutQuality") or 60.0)
    return int(min(88, max(58, round(bq))))


def generate(sb, r: dict) -> dict:
    """From a confirmed-breakout quant row, fire a LONG + CALL card.

    Returns {"long": id|None, "call": id|None}. Reuses runner's write helpers
    (lazy-imported to avoid a circular import).
    """
    out = {"long": None, "call": None}
    if sb is None:
        return out
    try:
        from engine import runner, options_scanner, push
    except Exception as e:
        logger.debug(f"[breakout_signals] import failed: {e}")
        return out

    tk    = (r.get("ticker") or "").upper()
    price = r.get("price")
    if not tk or not price or price <= 0:
        return out

    ma      = r.get("ma20")
    hi      = r.get("breakoutLevel")             # the broken 20-day high
    atr_pct = float(r.get("atrPct") or 2.0)
    atr     = float(price) * atr_pct / 100.0
    conf    = _conf(r)

    entry = round(float(price), 2)
    # Stop below entry (long). Start from 1.5·ATR; if the broken level sits just
    # below entry (a clean breakout), hug that level (now support) instead of a
    # wider ATR stop; then clamp the risk into the [MIN, MAX]% band.
    raw_stop = entry - 1.5 * atr
    if hi and 0 < float(hi) < entry:             # broken 20-day high below price
        raw_stop = max(raw_stop, float(hi) - 0.3 * atr)
    raw_stop = min(entry * (1 - _MIN_STOP_PCT), max(entry * (1 - _MAX_STOP_PCT), raw_stop))
    stop  = round(raw_stop, 2)                    # just below the breakout level
    t1    = round(entry + 1.5 * atr, 2)
    t2    = round(entry + 3.0 * atr, 2)
    rr    = round((t1 - entry) / (entry - stop), 2) if entry > stop else None
    rvol  = r.get("relativeVolume")
    rvol_txt = f" on {rvol:.1f}x volume" if isinstance(rvol, (int, float)) else ""
    base = round(float(hi), 2) if hi else (round(float(ma), 2) if ma else None)

    # ── LONG equity card ───────────────────────────────────────────────────
    signal_row = {
        "ticker":              tk,
        "direction":           "LONG",
        "entry_price":         entry,
        "stop_loss":           stop,
        "target_one":          t1,
        "target_two":          t2,
        "confidence_score":    conf,
        "confidence_factors":  [f"Broke 20-day high{rvol_txt}", "Above 20-day average"],
        "timeframe":           "1Day",
        "strategy_type":       "breakout",
        "status":              "active",
        "ai_explanation":      (
            f"{tk} broke above its 20-day high{rvol_txt} — a confirmed momentum breakout. "
            f"Long near {entry} with a stop just below the breakout level ({stop})"
            + (f"; a loss of {base} (now support) invalidates it" if base else "")
            + f". Targets {t1} / {t2}; trail your stop up."
        ),
        "regime_type":         "",
        "session_mode":        "",
        "confidence_tier":     "B",
        "position_multiplier": 0.25,        # small size — new, unproven detector
        "gamma_net_gex":       0,
        "gamma_is_negative":   False,
        "manipulation_clean":  True,
        "manipulation_flags":  [],
        "sl_adjustments":      [],
        "risk_reward":         rr,
        "score_breakdown":     {
            "detector_source": "BREAKOUT",
            "breakoutLevel":   round(float(hi), 2) if hi else None,
            "ma20":            round(float(ma), 2) if ma else None,
            "atr_used":        round(atr, 4),
            "initial_stop":    stop,
        },
        "confidence_grade":    "B",
        "risk_grade":          "MEDIUM",
        "chop_score":          0.0,
        "setup_type":          "breakout",
        "missing_confirmations": [],
    }
    try:
        out["long"] = runner._write_signal(sb, signal_row)
    except Exception as e:
        logger.warning(f"[breakout_signals] {tk} long write failed: {e}")
    if out["long"]:
        try:
            push.send_signal_alert(tk, "LONG", conf, "stock", signal_id=str(out["long"]))
        except Exception:
            pass

    # ── CALL option card (options_scanner picks + prices the contract) ──────
    try:
        if not runner._has_active_option_signal(sb, tk):
            opt = options_scanner.scan(tk, "LONG", float(price), stock_target_one=t1)
            if opt:
                opt["confidence_score"]   = conf
                opt["confidence_factors"] = ["Breakout call play"]
                opt["ai_explanation"]     = (
                    f"Call play on {tk}'s breakout — gains as it runs toward {t1}/{t2}. "
                    f"Defined risk (premium paid); exit if it loses "
                    f"{base if base else 'the breakout level'}."
                )
                opt["timeframe"]     = "1Day"
                opt["strategy_type"] = "breakout"
                opt["status"]        = "active"
                out["call"] = runner._write_option_signal(sb, opt)
                if out["call"]:
                    try:
                        push.send_signal_alert(tk, "LONG", conf, "option", signal_id=str(out["call"]))
                    except Exception:
                        pass
    except Exception as e:
        logger.warning(f"[breakout_signals] {tk} call scan/write failed: {e}")

    logger.info(f"[breakout_signals] {tk} long={out['long']} call={out['call']}")
    return out
