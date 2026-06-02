"""
Breakdown → tradeable cards (SHORT equity + PUT option).

When a name CONFIRMS a breakdown (breaks its 20-day low on volume) we don't just
push an FYI — we generate the actionable bearish trade so users can act:
  • a SHORT equity signal  (signals,        strategy_type='breakdown', SHORT)
  • a PUT option signal     (option_signals, via options_scanner put scan)

Levels (equity short):
  entry ≈ current price · stop just above the broken level (+1.5 ATR) ·
  targets at -1.5 ATR (T1) / -3 ATR (T2). The PUT is priced + filtered by
  options_scanner (Polygon→yfinance chain + Black-Scholes + liquidity/IV gates).

Both fire + push immediately (per product decision) and are tracked by
signal_monitor — SHORT direction + option lifecycle are already supported.
Everything is tagged detector_source='BREAKDOWN' so the detector-scorecard can
measure their realized win-rate (and we can cut them if they don't earn it).

Best-effort: never raises into the alert loop. Dedup is handled by the DB
unique-active-signal indexes (one active per ticker/strategy) + the option
active-check, so re-running on the same episode is a no-op.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("signalbolt.breakdown_signals")


def _conf(r: dict) -> int:
    """Confidence in the SHORT, derived from the breakdown's selling pressure."""
    bd = float(r.get("breakdownScore") or 60.0)
    return int(min(85, max(58, round(bd))))


def generate(sb, r: dict) -> dict:
    """From a confirmed-breakdown quant row, fire a SHORT + PUT card.

    Returns {"short": id|None, "put": id|None}. Reuses runner's write helpers
    (lazy-imported to avoid a circular import) so fired signals get the same
    stream-subscription + logging as every other fire path.
    """
    out = {"short": None, "put": None}
    if sb is None:
        return out
    try:
        from engine import runner, options_scanner, push
    except Exception as e:
        logger.debug(f"[breakdown_signals] import failed: {e}")
        return out

    tk    = (r.get("ticker") or "").upper()
    price = r.get("price")
    if not tk or not price or price <= 0:
        return out

    ma      = r.get("ma20")
    lo      = r.get("breakdownLevel")            # the broken 20-day low
    atr_pct = float(r.get("atrPct") or 2.0)
    atr     = float(price) * atr_pct / 100.0
    conf    = _conf(r)

    entry = round(float(price), 2)
    stop  = round(entry + 1.5 * atr, 2)          # just above the broken level
    t1    = round(entry - 1.5 * atr, 2)
    t2    = round(entry - 3.0 * atr, 2)
    rr    = round((entry - t1) / (stop - entry), 2) if stop > entry else None
    rvol  = r.get("relativeVolume")
    rvol_txt = f" on {rvol:.1f}x volume" if isinstance(rvol, (int, float)) else ""
    inval = round(float(lo), 2) if lo else (round(float(ma), 2) if ma else None)

    # ── SHORT equity card ──────────────────────────────────────────────────
    signal_row = {
        "ticker":              tk,
        "direction":           "SHORT",
        "entry_price":         entry,
        "stop_loss":           stop,
        "target_one":          t1,
        "target_two":          t2,
        "confidence_score":    conf,
        "confidence_factors":  [f"Broke 20-day low{rvol_txt}", "Below 20-day average"],
        "timeframe":           "1Day",
        "strategy_type":       "breakdown",
        "status":              "active",
        "ai_explanation":      (
            f"{tk} broke below its 20-day low{rvol_txt} — a confirmed bearish breakdown. "
            f"Short near {entry} with a stop just above the broken level ({stop})"
            + (f"; a reclaim of {inval} invalidates it" if inval else "")
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
            "detector_source": "BREAKDOWN",
            "breakdownLevel":  round(float(lo), 2) if lo else None,
            "ma20":            round(float(ma), 2) if ma else None,
            "atr_used":        round(atr, 4),
            "initial_stop":    stop,
        },
        "confidence_grade":    "B",
        "risk_grade":          "HIGH",
        "chop_score":          0.0,
        "setup_type":          "breakdown",
        "missing_confirmations": [],
    }
    try:
        out["short"] = runner._write_signal(sb, signal_row)
    except Exception as e:
        logger.warning(f"[breakdown_signals] {tk} short write failed: {e}")
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
                opt["confidence_factors"] = ["Breakdown put play"]
                opt["ai_explanation"]     = (
                    f"Put play on {tk}'s breakdown — gains as it falls toward {t1}/{t2}. "
                    f"Defined risk (premium paid); exit if it reclaims "
                    f"{inval if inval else 'the broken level'}."
                )
                opt["timeframe"]     = "1Day"
                opt["strategy_type"] = "breakdown"
                opt["status"]        = "active"
                out["put"] = runner._write_option_signal(sb, opt)
                if out["put"]:
                    try:
                        push.send_signal_alert(tk, "SHORT", conf, "option", signal_id=str(out["put"]))
                    except Exception:
                        pass
    except Exception as e:
        logger.warning(f"[breakdown_signals] {tk} put scan/write failed: {e}")

    logger.info(f"[breakdown_signals] {tk} short={out['short']} put={out['put']}")
    return out
