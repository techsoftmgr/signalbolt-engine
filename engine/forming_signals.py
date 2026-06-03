"""
FORMING signals — EARLY, anticipatory tracked cards for the four lifecycle
detectors, fired BEFORE the confirmed stage so we capture the swing earlier.

Variants (each a separate, measured experiment — tagged distinctly so the Quant
scorecard + Analytics track its expectancy on its own):

  • BREAKOUT_FORMING  (LONG)  — pressing the 20-day high on volume, NOT yet broken
  • BREAKDOWN_FORMING (SHORT) — lost the 20-day average on heavy down-volume,
                                 NOT yet broke the 20-day low
  • PEAK_FORMING      (SHORT) — topping `watch` + exhaustion + first lower high
  • TURN_FORMING      (LONG)  — bottoming `watch` + first higher low / reclaim

Risk profile (vs the confirmed detector cards): EARLIER entry, WIDER stop (the
unconfirmed swing whipsaws more, so give it room — beyond the structural
invalidation + a 0.5·ATR buffer, in a 2.5–9% band), but SMALLER size (0.25×) so
the dollar risk stays bounded. Wider targets (+2.5 / +5 ATR) since the early
entry leaves more of the move to capture.

Fires a stock card (signals) + an option card (options_scanner). Best-effort —
never raises into the alert loop. Deduped per (ticker, forming strategy) so it
fires once per episode and can co-exist with the CONFIRMED version on the same
name (that's the A/B: early entry vs confirmed entry on the identical move).
"""
from __future__ import annotations

import logging

logger = logging.getLogger("signalbolt.forming_signals")

# Wider risk band than the confirmed cards (1.5–5%) — room for the early swing.
_MAX_STOP_PCT = 0.09
_MIN_STOP_PCT = 0.025
_ATR_STOP_MULT = 2.0     # base stop distance in ATRs (vs 1.5 confirmed)
_BUF_ATR       = 0.5     # buffer beyond the structural level
_T1_ATR        = 2.5
_T2_ATR        = 5.0

# kind → (direction, detector_source, strategy_type, contract_dir)
_CONFIG = {
    "breakout":  ("LONG",  "BREAKOUT_FORMING",  "breakout_forming",  "LONG"),
    "breakdown": ("SHORT", "BREAKDOWN_FORMING", "breakdown_forming", "SHORT"),
    "peak":      ("SHORT", "PEAK_FORMING",      "peak_forming",      "SHORT"),
    "turn":      ("LONG",  "TURN_FORMING",      "turn_forming",      "LONG"),
    # Volume-leads-price: the EARLIEST tell, before any structural break.
    "accum":     ("LONG",  "ACCUM_FORMING",     "accum_forming",     "LONG"),
    "distrib":   ("SHORT", "DISTRIB_FORMING",   "distrib_forming",   "SHORT"),
}


def _has_active_forming(sb, ticker: str, strategy: str) -> bool:
    """One FORMING signal per (ticker, strategy) episode. Does NOT block the
    confirmed version (different strategy_type) — by design, so we can compare
    early-vs-confirmed on the same name."""
    try:
        res = (sb.table("signals").select("id")
               .eq("ticker", ticker).eq("strategy_type", strategy)
               .eq("status", "active").execute())
        return bool(res.data)
    except Exception:
        return False


def _levels(kind: str, price: float, atr: float, r: dict) -> tuple[float, float, float]:
    """Return (stop, t1, t2) for a forming entry — wider stop beyond the
    structural invalidation, wider targets."""
    is_long = _CONFIG[kind][0] == "LONG"
    ma   = float(r.get("ma20") or 0) or None
    hi   = float(r.get("breakoutLevel") or 0) or None    # 20-day high (top side)
    lo   = float(r.get("breakdownLevel") or 0) or None    # 20-day low  (bottom side)

    if is_long:
        # Invalidation below: a forming breakout fails back under its base/MA;
        # a forming turn fails back under the recent low.
        struct = (lo - _BUF_ATR * atr) if (kind == "turn" and lo) else \
                 ((ma - _BUF_ATR * atr) if ma else None)
        atr_stop = price - _ATR_STOP_MULT * atr
        raw = min(struct, atr_stop) if struct else atr_stop          # lower = wider
        stop = min(price * (1 - _MIN_STOP_PCT), max(price * (1 - _MAX_STOP_PCT), raw))
        t1 = price + _T1_ATR * atr
        t2 = price + _T2_ATR * atr
    else:
        # Invalidation above: a forming top fails on a new high (blow-off high ≈
        # 20-day high); a forming breakdown fails back above the lost MA.
        struct = (hi + _BUF_ATR * atr) if (kind == "peak" and hi) else \
                 ((ma + _BUF_ATR * atr) if ma else None)
        atr_stop = price + _ATR_STOP_MULT * atr
        raw = max(struct, atr_stop) if struct else atr_stop          # higher = wider
        stop = max(price * (1 + _MIN_STOP_PCT), min(price * (1 + _MAX_STOP_PCT), raw))
        t1 = price - _T1_ATR * atr
        t2 = price - _T2_ATR * atr
    return round(stop, 2), round(t1, 2), round(t2, 2)


def _conf(kind: str, r: dict) -> int:
    score_field = {
        "breakout":  "breakoutScore", "breakdown": "breakdownScore",
        "peak":      "peakScore",     "turn":      "turnaroundScore",
        "accum":     "volumeScore",   "distrib":   "volumeScore",
    }[kind]
    raw = float(r.get(score_field) or r.get("volumeScore") or 58.0)
    return int(min(80, max(55, round(raw))))   # capped below confirmed — unproven


def generate(sb, r: dict, kind: str) -> dict:
    """From a quant row at the EARLY stage of `kind`, fire a FORMING stock + option
    card. Returns {"stock": id|None, "option": id|None}."""
    out = {"stock": None, "option": None}
    if sb is None or kind not in _CONFIG:
        return out
    try:
        from engine import runner, options_scanner, push
    except Exception as e:
        logger.debug(f"[forming_signals] import failed: {e}")
        return out

    direction, src, strat, opt_dir = _CONFIG[kind]
    tk    = (r.get("ticker") or "").upper()
    price = r.get("price")
    if not tk or not price or price <= 0:
        return out
    if _has_active_forming(sb, tk, strat):
        return out

    atr_pct = float(r.get("atrPct") or 2.0)
    atr     = float(price) * atr_pct / 100.0
    entry   = round(float(price), 2)
    stop, t1, t2 = _levels(kind, entry, atr, r)
    conf    = _conf(kind, r)
    rr      = round(abs(t1 - entry) / abs(entry - stop), 2) if entry != stop else None
    is_long = direction == "LONG"
    rvol    = r.get("relativeVolume")
    rvol_txt = f" on {rvol:.1f}x volume" if isinstance(rvol, (int, float)) else ""

    label = {
        "breakout":  "forming breakout (pressing the 20-day high)",
        "breakdown": "forming breakdown (lost the 20-day average)",
        "peak":      "forming top (exhaustion + first lower high)",
        "turn":      "forming bottom (first higher low / reclaim)",
        "accum":     "high-volume accumulation (big buyers stepping in)",
        "distrib":   "high-volume distribution (heavy selling stepping in)",
    }[kind]

    signal_row = {
        "ticker":              tk,
        "direction":           direction,
        "entry_price":         entry,
        "stop_loss":           stop,
        "target_one":          t1,
        "target_two":          t2,
        "confidence_score":    conf,
        "confidence_factors":  [f"Early {kind} setup{rvol_txt}", "Anticipatory (forming) entry"],
        "timeframe":           "1Day",
        "strategy_type":       strat,
        "status":              "active",
        "management_mode":     "engine",
        "origin":              "engine",
        "ai_explanation":      (
            f"{tk} is showing a {label}{rvol_txt} — an EARLY, anticipatory "
            f"{'long' if is_long else 'short'} ahead of confirmation. "
            f"Entry near {entry}, wider stop {stop} (give the swing room), "
            f"targets {t1} / {t2}. Smaller size — unproven, forming setup."
        ),
        "regime_type":         "",
        "session_mode":        "",
        "confidence_tier":     "B",
        "position_multiplier": 0.25,
        "gamma_net_gex":       0,
        "gamma_is_negative":   False,
        "manipulation_clean":  True,
        "manipulation_flags":  [],
        "sl_adjustments":      [],
        "risk_reward":         rr,
        "score_breakdown":     {
            "detector_source": src,
            "forming":         True,
            "ma20":            round(float(r.get("ma20")), 2) if r.get("ma20") else None,
            "atr_used":        round(atr, 4),
            "initial_stop":    stop,
        },
        "confidence_grade":    "B",
        "risk_grade":          "MEDIUM",
        "chop_score":          0.0,
        "setup_type":          strat,
        "missing_confirmations": [],
    }
    try:
        out["stock"] = runner._write_signal(sb, signal_row)
    except Exception as e:
        logger.warning(f"[forming_signals] {tk} {src} stock write failed: {e}")
    if out["stock"]:
        try:
            push.send_signal_alert(tk, direction, conf, "stock", signal_id=str(out["stock"]))
        except Exception:
            pass

    # ── Option leg (options_scanner prices + filters the contract) ──────────
    try:
        if not runner._has_active_option_signal(sb, tk):
            opt = options_scanner.scan(tk, opt_dir, float(price), stock_target_one=t1)
            if opt:
                opt["confidence_score"]   = conf
                opt["confidence_factors"] = [f"{src} option play"]
                opt["ai_explanation"]     = (
                    f"{'Call' if is_long else 'Put'} play on {tk}'s {label} — "
                    f"early/anticipatory. Defined risk (premium); exit if the setup fails."
                )
                opt["timeframe"]       = "1Day"
                opt["strategy_type"]   = strat
                opt["status"]          = "active"
                opt["management_mode"] = "engine"
                opt["origin"]          = "engine"
                out["option"] = runner._write_option_signal(sb, opt)
                if out["option"]:
                    try:
                        push.send_signal_alert(tk, direction, conf, "option", signal_id=str(out["option"]))
                    except Exception:
                        pass
    except Exception as e:
        logger.warning(f"[forming_signals] {tk} {src} option scan/write failed: {e}")

    logger.info(f"[forming_signals] {tk} {src} stock={out['stock']} option={out['option']}")
    return out
