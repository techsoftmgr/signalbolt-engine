"""
Peak → tradeable cards (SHORT equity + PUT option).

When a name reaches the PEAK stage (the cycle detector's confirmed topping /
distribution stage) we generate the actionable bearish reversal play:
  • a SHORT equity signal  (signals,        strategy_type='peak', SHORT)
  • a PUT option signal      (option_signals, via options_scanner put scan)

Bearish twin of turnaround_signals; structurally a breakdown_signals clone but
anchored to the recent HIGH (resistance) instead of the broken low. Levels
(equity short):
  entry ≈ current price · stop just above the recent high (+1.5 ATR) ·
  targets at -1.5 ATR (T1) / -3 ATR (T2). The PUT is priced + filtered by
  options_scanner (chain + Black-Scholes + liquidity/IV gates).

Tagged detector_source='PEAK' for win-rate measurement on the scorecard. Small
size (0.25x) — new, unproven detector expression. Best-effort: never raises
into the caller. Deduped by the DB unique-active indexes + the options
active-check + runner._has_active_signal upstream.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("signalbolt.peak_signals")

# Stop-width guards (these cards bypass sl_tp_engine, which caps SL%).
_MAX_STOP_PCT = 0.05    # never risk more than 5% on a 0.25x detector card
_MIN_STOP_PCT = 0.015   # …but at least 1.5%, so normal noise can't nick us


def _conf(r: dict) -> int:
    """Confidence in the SHORT, from the peak's strength."""
    ps = float(r.get("peakScore") or 60.0)
    return int(min(85, max(58, round(ps))))


def generate(sb, r: dict) -> dict:
    """From a peak-stage quant row, fire a SHORT + PUT card.

    Returns {"short": id|None, "put": id|None}. Reuses runner's write helpers
    (lazy-imported to avoid a circular import).
    """
    out = {"short": None, "put": None}
    if sb is None:
        return out
    try:
        from engine import runner, options_scanner, push
    except Exception as e:
        logger.debug(f"[peak_signals] import failed: {e}")
        return out

    tk    = (r.get("ticker") or "").upper()
    price = r.get("price")
    if not tk or not price or price <= 0:
        return out

    ma      = r.get("ma20")
    hi      = r.get("breakoutLevel")             # recent 20-day high = resistance
    atr_pct = float(r.get("atrPct") or 2.0)
    atr     = float(price) * atr_pct / 100.0
    conf    = _conf(r)

    entry = round(float(price), 2)
    # Stop above entry (short). Start from 1.5·ATR; if the recent high sits just
    # above entry (price rolling over right under resistance), hug that level
    # instead of a wider ATR stop; then clamp the risk into the [MIN, MAX]% band.
    raw_stop = entry + 1.5 * atr
    if hi and float(hi) > entry:                 # recent high above price
        raw_stop = min(raw_stop, float(hi) + 0.3 * atr)
    raw_stop = max(entry * (1 + _MIN_STOP_PCT), min(entry * (1 + _MAX_STOP_PCT), raw_stop))
    stop  = round(raw_stop, 2)
    t1    = round(entry - 1.5 * atr, 2)
    t2    = round(entry - 3.0 * atr, 2)
    rr    = round((entry - t1) / (stop - entry), 2) if stop > entry else None
    rvol  = r.get("relativeVolume")
    rvol_txt = f" on {rvol:.1f}x volume" if isinstance(rvol, (int, float)) else ""
    inval = round(float(hi), 2) if hi else (round(float(ma), 2) if ma else None)

    # ── SHORT equity card ──────────────────────────────────────────────────
    signal_row = {
        "ticker":              tk,
        "direction":           "SHORT",
        "entry_price":         entry,
        "stop_loss":           stop,
        "target_one":          t1,
        "target_two":          t2,
        "confidence_score":    conf,
        "confidence_factors":  [f"Peak / topping stage{rvol_txt}", "Distribution / reversal stage"],
        "timeframe":           "1Day",
        "strategy_type":       "peak",
        "status":              "active",
        "ai_explanation":      (
            f"{tk} reached its peak / topping stage{rvol_txt} — a confirmed bearish reversal. "
            f"Short near {entry} with a stop just above the recent high ({stop})"
            + (f"; a reclaim of {inval} invalidates the top" if inval else "")
            + f". Cover into {t1} / {t2}."
        ),
        "regime_type":         "",
        "session_mode":        "",
        "confidence_tier":     "B",
        "position_multiplier": 0.25,        # small size — new, directional, unproven
        "gamma_net_gex":       0,
        "gamma_is_negative":   False,
        "manipulation_clean":  True,
        "manipulation_flags":  [],
        "sl_adjustments":      [],
        "risk_reward":         rr,
        "score_breakdown":     {
            "detector_source": "PEAK",
            "breakoutLevel":   round(float(hi), 2) if hi else None,
            "ma20":            round(float(ma), 2) if ma else None,
            "atr_used":        round(atr, 4),
            "initial_stop":    stop,
        },
        "confidence_grade":    "B",
        "risk_grade":          "HIGH",
        "chop_score":          0.0,
        "setup_type":          "peak",
        "missing_confirmations": [],
    }
    try:
        out["short"] = runner._write_signal(sb, signal_row)
    except Exception as e:
        logger.warning(f"[peak_signals] {tk} short write failed: {e}")
    if out["short"]:
        try:
            push.send_signal_alert(tk, "SHORT", conf, "stock", signal_id=str(out["short"]))
        except Exception:
            pass

    # ── PUT option card (options_scanner picks + prices the contract) ───────
    try:
        if not runner._has_active_option_signal(sb, tk):
            opt = options_scanner.scan(tk, "SHORT", float(price), stock_target_one=t1)
            if opt:
                opt["confidence_score"]   = conf
                opt["confidence_factors"] = ["Peak put play"]
                opt["ai_explanation"]     = (
                    f"Put play on {tk}'s peak — gains as it falls toward {t1}/{t2}. "
                    f"Defined risk (premium paid); exit if it reclaims "
                    f"{inval if inval else 'the recent high'}."
                )
                opt["timeframe"]     = "1Day"
                opt["strategy_type"] = "peak"
                opt["status"]        = "active"
                out["put"] = runner._write_option_signal(sb, opt)
                if out["put"]:
                    try:
                        push.send_signal_alert(tk, "SHORT", conf, "option", signal_id=str(out["put"]))
                    except Exception:
                        pass
    except Exception as e:
        logger.warning(f"[peak_signals] {tk} put scan/write failed: {e}")

    logger.info(f"[peak_signals] {tk} short={out['short']} put={out['put']}")
    return out
