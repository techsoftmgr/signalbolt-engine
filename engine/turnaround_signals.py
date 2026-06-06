"""
Turnaround → tradeable cards (LONG equity + CALL option).

When a name reaches the turnaround BUY-ZONE (the cycle detector's confirmed
bottoming stage) we generate the actionable bullish reversal play:
  • a LONG equity signal  (signals,        strategy_type='turnaround', LONG)
  • a CALL option signal    (option_signals, via options_scanner call scan)

Bullish twin of peak_signals; structurally a breakout_signals clone but anchored
to the recent LOW (support) instead of the broken high. Levels (equity long):
  entry ≈ current price · stop just below the recent low (-1.5 ATR) ·
  targets at +1.5 ATR (T1) / +3 ATR (T2). The CALL is priced + filtered by
  options_scanner (chain + Black-Scholes + liquidity/IV gates).

Tagged detector_source='TURNAROUND' for win-rate measurement on the scorecard.
Small size (0.25x) — new, unproven detector expression. Best-effort: never
raises into the caller. Deduped by the DB unique-active indexes + the
options active-check + runner._has_active_signal upstream.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("signalbolt.turnaround_signals")


def _live_regime() -> str:
    """Current regime to stamp at fire (regime-sliceable). Never empty/raises."""
    try:
        from engine import signal_telemetry
        return signal_telemetry.live_regime_type()
    except Exception:
        return "RANGING"

# Stop-width guards (these cards bypass sl_tp_engine, which caps SL%).
_MAX_STOP_PCT = 0.05    # never risk more than 5% on a 0.25x detector card
_MIN_STOP_PCT = 0.015   # …but at least 1.5%, so normal noise can't nick us


def _conf(r: dict) -> int:
    """Confidence in the LONG, from the turnaround's strength."""
    ts = float(r.get("turnaroundScore") or 60.0)
    return int(min(85, max(58, round(ts))))


def generate(sb, r: dict) -> dict:
    """From a turnaround buy-zone quant row, fire a LONG + CALL card.

    Returns {"long": id|None, "call": id|None}. Reuses runner's write helpers
    (lazy-imported to avoid a circular import).
    """
    out = {"long": None, "call": None}
    if sb is None:
        return out
    try:
        from engine import runner, options_scanner, push
    except Exception as e:
        logger.debug(f"[turnaround_signals] import failed: {e}")
        return out

    tk    = (r.get("ticker") or "").upper()
    price = r.get("price")
    if not tk or not price or price <= 0:
        return out

    ma      = r.get("ma20")
    lo      = r.get("breakdownLevel")            # recent 20-day low = support
    atr_pct = float(r.get("atrPct") or 2.0)
    atr     = float(price) * atr_pct / 100.0
    conf    = _conf(r)

    entry = round(float(price), 2)
    # Stop below entry (long). Start from 1.5·ATR; if the recent low sits just
    # below entry (a tight bounce off support), hug that level instead of a
    # wider ATR stop; then clamp the risk into the [MIN, MAX]% band.
    raw_stop = entry - 1.5 * atr
    if lo and 0 < float(lo) < entry:             # recent low below price
        raw_stop = max(raw_stop, float(lo) - 0.3 * atr)
    raw_stop = min(entry * (1 - _MIN_STOP_PCT), max(entry * (1 - _MAX_STOP_PCT), raw_stop))
    stop  = round(raw_stop, 2)
    t1    = round(entry + 1.5 * atr, 2)
    t2    = round(entry + 3.0 * atr, 2)
    rr    = round((t1 - entry) / (entry - stop), 2) if entry > stop else None
    rvol  = r.get("relativeVolume")
    rvol_txt = f" on {rvol:.1f}x volume" if isinstance(rvol, (int, float)) else ""
    inval = round(float(lo), 2) if lo else (round(float(ma), 2) if ma else None)

    # ── LONG equity card ───────────────────────────────────────────────────
    signal_row = {
        "ticker":              tk,
        "direction":           "LONG",
        "entry_price":         entry,
        "stop_loss":           stop,
        "target_one":          t1,
        "target_two":          t2,
        "confidence_score":    conf,
        "confidence_factors":  [f"Turnaround buy-zone{rvol_txt}", "Bottoming / reversal stage"],
        "timeframe":           "1Day",
        "strategy_type":       "turnaround",
        "status":              "active",
        "ai_explanation":      (
            f"{tk} reached its turnaround buy-zone{rvol_txt} — a confirmed bottoming reversal. "
            f"Long near {entry} with a stop just below the recent low ({stop})"
            + (f"; a loss of {inval} invalidates the turn" if inval else "")
            + f". Targets {t1} / {t2}; trail your stop up."
        ),
        "regime_type":         _live_regime(),
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
            "detector_source": "TURNAROUND",
            "breakdownLevel":  round(float(lo), 2) if lo else None,
            "ma20":            round(float(ma), 2) if ma else None,
            "atr_used":        round(atr, 4),
            "initial_stop":    stop,
        },
        "confidence_grade":    "B",
        "risk_grade":          "MEDIUM",
        "chop_score":          0.0,
        "setup_type":          "turnaround",
        "missing_confirmations": [],
    }
    try:
        out["long"] = runner._write_signal(sb, signal_row)
    except Exception as e:
        logger.warning(f"[turnaround_signals] {tk} long write failed: {e}")
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
                opt["confidence_factors"] = ["Turnaround call play"]
                opt["ai_explanation"]     = (
                    f"Call play on {tk}'s turnaround — gains as it recovers toward {t1}/{t2}. "
                    f"Defined risk (premium paid); exit if it loses "
                    f"{inval if inval else 'the recent low'}."
                )
                opt["timeframe"]     = "1Day"
                opt["strategy_type"] = "turnaround"
                opt["status"]        = "active"
                out["call"] = runner._write_option_signal(sb, opt)
                if out["call"]:
                    try:
                        push.send_signal_alert(tk, "LONG", conf, "option", signal_id=str(out["call"]))
                    except Exception:
                        pass
    except Exception as e:
        logger.warning(f"[turnaround_signals] {tk} call scan/write failed: {e}")

    logger.info(f"[turnaround_signals] {tk} long={out['long']} call={out['call']}")
    return out
