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

# Stop-width guards. These detector cards bypass sl_tp_engine (which caps SL%),
# so without this a post-parabolic name with a fat ATR gets an absurd stop —
# e.g. MRVL ran +46% off its 20-day MA, ATR≈6%, so a flat 1.5·ATR stop was
# −10% from entry. Clamp every stop into a sane [MIN, MAX]% band off entry.
_MAX_STOP_PCT = 0.05    # never risk more than 5% on a 0.25x detector card
_MIN_STOP_PCT = 0.015   # …but at least 1.5%, so normal noise can't nick us


def _conf(r: dict) -> int:
    """Confidence in the SHORT, derived from the breakdown's selling pressure."""
    bd = float(r.get("breakdownScore") or 60.0)
    return int(min(85, max(58, round(bd))))


# ── Asset classification (LOGGING-ONLY) ─────────────────────────────────────
# Tags each breakdown card with its instrument class so we can LATER measure
# whether commodity / bond / broad-ETF breakdowns (which mean-revert around
# macro levels) earn their keep vs single-name equity breakdowns. This is PURE
# metadata written to score_breakdown — it does NOT gate, size, re-price, or
# alter any signal in any way. Curated, extensible lists; anything unknown
# defaults to "equity" so a normal stock's behaviour is never changed.
_COMMODITY_ETFS = {
    "GLD", "IAU", "GLDM", "SGOL", "SLV", "SIVR", "PSLV", "GDX", "GDXJ", "SIL",
    "USO", "BNO", "UNG", "UGA", "DBC", "DBA", "DBO", "PPLT", "PALL", "CPER",
    "WEAT", "CORN", "SOYB", "GLTR", "COMT", "PDBC", "USCI",
}
_BOND_ETFS = {
    "TLT", "IEF", "SHY", "IEI", "GOVT", "BIL", "LQD", "HYG", "JNK", "AGG",
    "BND", "BNDX", "TIP", "VTIP", "MUB", "EMB", "TLH", "EDV", "SHV",
}
_BROAD_EQUITY_ETFS = {
    "SPY", "VOO", "IVV", "QQQ", "QQQM", "IWM", "DIA", "VTI", "ITOT", "RSP",
    "MDY", "VTV", "VUG", "SCHB", "SCHX", "SPLG", "IWB", "IWV",
}
_SECTOR_ETFS = {
    "XLE", "XLF", "XLK", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE",
    "XLC", "SMH", "SOXX", "XBI", "IBB", "KRE", "XME", "XOP", "ITB", "XRT",
}


def classify_asset(ticker: str) -> dict:
    """Logging-only instrument classification. Returns {'asset_class', 'is_etf'}.
    Unknown / single-name tickers default to 'equity' so this never reclassifies
    or alters a stock signal."""
    t = (ticker or "").upper()
    if t in _COMMODITY_ETFS:      asset_class = "commodity"
    elif t in _BOND_ETFS:         asset_class = "bond"
    elif t in _BROAD_EQUITY_ETFS: asset_class = "broad_etf"
    elif t in _SECTOR_ETFS:       asset_class = "sector_etf"
    else:                         asset_class = "equity"
    return {"asset_class": asset_class, "is_etf": asset_class != "equity"}


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
    # Sanity-cap the ATR used for SL/TP so a high-ATR / leveraged name can't blow
    # targets past zero (see KORU note in forming_signals). Cap at 8% of price.
    atr     = min(atr, float(price) * 0.08)
    conf    = _conf(r)

    entry = round(float(price), 2)
    # Stop above entry (short). Start from 1.5·ATR; if the broken level sits
    # just above entry (price already below it), hug that level instead of a
    # wider ATR stop; then clamp the risk into the [MIN, MAX]% band.
    raw_stop = entry + 1.5 * atr
    if lo and float(lo) > entry:                 # broken 20-day low above price
        raw_stop = min(raw_stop, float(lo) + 0.3 * atr)
    raw_stop = max(entry * (1 + _MIN_STOP_PCT), min(entry * (1 + _MAX_STOP_PCT), raw_stop))
    stop  = round(raw_stop, 2)                    # just above the broken level
    t1    = round(entry - 1.5 * atr, 2)
    t2    = round(entry - 3.0 * atr, 2)
    rr    = round((entry - t1) / (stop - entry), 2) if stop > entry else None
    rvol  = r.get("relativeVolume")
    rvol_txt = f" on {rvol:.1f}x volume" if isinstance(rvol, (int, float)) else ""
    inval = round(float(lo), 2) if lo else (round(float(ma), 2) if ma else None)

    # Logging-only fire-time telemetry (regime + concentration + sector) for the
    # breakdown-quality study. NEVER gates firing — fails open.
    try:
        from engine import signal_telemetry
        regime_type, study = signal_telemetry.capture(sb, tk, "SHORT", "breakdown")
        if not regime_type:
            regime_type = signal_telemetry.live_regime_type()
    except Exception:
        regime_type, study = "", {}

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
        "regime_type":         regime_type,
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
            # Logging-only metadata for the breakdown-quality study (does NOT
            # gate or change firing). Lets us later segment realized expectancy
            # by entry volume and instrument class.
            "relativeVolume":  round(float(rvol), 2) if isinstance(rvol, (int, float)) else None,
            **classify_asset(tk),
            "study":           study,
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
