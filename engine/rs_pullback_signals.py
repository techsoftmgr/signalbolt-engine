"""
RS-Pullback signals — fire a LONG when a relative-strength leader pulls back to
its rising 20-day average.

This is the ONE long-side edge that measured well (a 2y/858-event backtest:
+0.24R even in a weak/RISK_OFF tape, on par with the same setup in a healthy
market — vs the -EV of catching falling knives). It's "buy strength on a dip,"
the opposite of bottom-fishing. The watchlist already TAGS these
(regimeCategory == 'rs_pullback'); this turns that tag into an actual signal so
the engine acts on it and rides the winner.

Design:
  • Source of truth = the worker's cached scored universe (quant:scored:v1), which
    already carries regimeCategory + rsVsSpy per ticker. No extra data fetch.
  • Fires a swing LONG (timeframe 1Day → trend_ride rides it under the rising
    20-MA automatically). Stop below the 20-MA; small initial size (unproven live).
  • detector_source = 'RS_PULLBACK' so the realized-edge scorecard isolates it
    (group_by=detector) before we trust / upsize it.
  • PANIC-gated (acute crashes flush even leaders) and kill-switchable.
"""
import logging
import os

logger = logging.getLogger("signalbolt.rs_pullback_signals")

_STRAT = "swing_trade"          # recognised as a swing by trend_ride + signal_monitor
_SRC   = "RS_PULLBACK"          # scorecard tag (group_by=detector isolates it)
_MAX_PER_SCAN = 4               # don't flood — best (highest RS) first
_SCORED_KEY = "quant:scored:v1"


def enabled() -> bool:
    """Kill switch — default ON. Set RS_PULLBACK_SIGNALS_ENABLED=false to disable."""
    return os.environ.get("RS_PULLBACK_SIGNALS_ENABLED", "true").strip().lower() not in (
        "0", "false", "no", "off",
    )


def _has_active_long(sb, ticker: str) -> bool:
    """Don't stack longs — skip if any active LONG already exists on the ticker.
    Fails CLOSED (treats a DB error as 'has one') so we never double-fire on error."""
    try:
        res = (sb.table("signals").select("id,direction")
               .eq("ticker", ticker).eq("status", "active").execute())
        return any((s.get("direction") == "LONG") for s in (res.data or []))
    except Exception:
        return True


def build_signal_row(r: dict, regime_type: str = "", study: dict | None = None) -> dict | None:
    """Pure: build the LONG signal row from a scored-universe row. None if unusable.
    Stop sits below BOTH the entry and the 20-MA (where the pullback held); targets
    are 1.5R / 3R but trend_ride lets winners run past them."""
    tk    = (r.get("ticker") or "").upper()
    price = r.get("price")
    ma20  = r.get("ma20")
    if not tk or not price or float(price) <= 0 or not ma20 or float(ma20) <= 0:
        return None
    # Defensive: only a genuine pullback-TO-the-20-MA (price near it). Upstream
    # (_regime_category) already enforces this, but guard so a stray row can't
    # produce a degenerate signal with a hair-thin stop.
    if not (0.95 * float(ma20) <= float(price) <= 1.03 * float(ma20)):
        return None
    entry   = round(float(price), 2)
    atr_pct = float(r.get("atrPct") or 2.0)
    atr     = min(float(price) * atr_pct / 100.0, float(price) * 0.06)   # cap like forming
    stop    = round(min(entry - 1.2 * atr, float(ma20) * 0.99), 2)
    risk    = entry - stop
    if risk <= 0:
        return None
    t1 = round(entry + 1.5 * risk, 2)
    t2 = round(entry + 3.0 * risk, 2)
    rr = round((t1 - entry) / risk, 2)
    rs = r.get("rsVsSpy")
    rs_txt = f"+{rs}" if isinstance(rs, (int, float)) and rs >= 0 else f"{rs}"
    return {
        "ticker": tk, "direction": "LONG", "entry_price": entry,
        "stop_loss": stop, "target_one": t1, "target_two": t2,
        "confidence_score": 62,
        "confidence_factors": [
            f"Relative-strength leader ({rs_txt}% vs SPY, 20d) pulling back to the rising 20-MA",
            "Buying strength on a dip — the measured +EV long even in a weak tape (+0.24R)",
        ],
        "timeframe": "1Day", "strategy_type": _STRAT, "status": "active",
        "management_mode": "engine", "origin": "engine",
        "ai_explanation": (
            f"{tk} is a relative-strength leader (outperforming SPY by {rs_txt}% over 20 days) "
            f"pulling back to its rising 20-day average (~{round(float(ma20), 2)}) — the highest-odds "
            f"long in a weak tape. Entry ~{entry}, stop {stop} (below the 20-MA), targets {t1} / {t2}; "
            f"lets winners ride the rising 20-MA. Smaller size — newly live, tracked."
        ),
        "regime_type": regime_type, "session_mode": "", "confidence_tier": "B",
        "position_multiplier": 0.5, "gamma_net_gex": 0, "gamma_is_negative": False,
        "manipulation_clean": True, "manipulation_flags": [], "sl_adjustments": [],
        "risk_reward": rr,
        "score_breakdown": {
            "detector_source": _SRC, "rs_pullback": True, "rsVsSpy": rs,
            "ma20": round(float(ma20), 2), "atr_used": round(atr, 4), "initial_stop": stop,
            "relativeVolume": (round(float(r.get("relativeVolume")), 2)
                               if isinstance(r.get("relativeVolume"), (int, float)) else None),
            "study": study or {},
        },
        "confidence_grade": "B", "risk_grade": "MEDIUM", "chop_score": 0.0,
        "setup_type": _STRAT, "missing_confirmations": [],
    }


def scan_and_fire(sb) -> int:
    """Read the cached scored universe, fire LONGs on the top RS-pullback names that
    don't already have an active long. Returns the count fired. Never raises."""
    if sb is None or not enabled():
        return 0
    # PANIC gate: no new longs even for leaders in an acute crash (mirrors the RS
    # exemption's PANIC carve-out).
    regime = ""
    try:
        from engine import signal_telemetry
        regime = signal_telemetry.live_regime_type() or ""
    except Exception:
        pass
    if regime == "PANIC":
        return 0
    try:
        from engine import cache
        rows = cache.kv.get_json(_SCORED_KEY) or []
    except Exception:
        rows = []
    cands = [r for r in rows if r.get("regimeCategory") == "rs_pullback"]
    cands.sort(key=lambda r: -(r.get("rsVsSpy") or 0))   # strongest RS first

    from engine import runner
    fired = 0
    for r in cands:
        if fired >= _MAX_PER_SCAN:
            break
        tk = (r.get("ticker") or "").upper()
        if not tk or _has_active_long(sb, tk):
            continue
        try:
            from engine import signal_telemetry
            regime_type, study = signal_telemetry.capture(sb, tk, "LONG", _STRAT)
            if not regime_type:
                regime_type = signal_telemetry.live_regime_type() or regime
        except Exception:
            regime_type, study = regime, {}
        row = build_signal_row(r, regime_type=regime_type, study=study)
        if not row:
            continue
        try:
            sid = runner._write_signal(sb, row)
        except Exception as e:
            logger.debug(f"[rs_pullback] write failed {tk}: {e}")
            continue
        if not sid:
            continue   # untradeable (penny/leveraged) — _write_signal blocked it
        fired += 1
        logger.info(f"[rs_pullback] fired LONG {tk} entry {row['entry_price']} "
                    f"stop {row['stop_loss']} (RS {r.get('rsVsSpy')}% vs SPY)")
        try:
            from engine import push
            push._send_raw(
                title=f"🟢 RS Pullback — {tk}",
                body=(f"{tk}: relative-strength leader pulling back to its rising 20-MA "
                      f"({r.get('rsVsSpy')}% vs SPY) — buyable-dip setup."),
                data={"type": "rs_pullback", "ticker": tk, "signal_id": str(sid)},
            )
        except Exception:
            pass
    return fired
