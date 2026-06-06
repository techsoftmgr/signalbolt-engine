"""
BO_POC — a FIDELITY-FIRST breakout POC.

The fidelity check (2026-06-05) found the live BREAKOUT detector matches the
backtest's winning archetype only ~7% of the time (live fires intraday on the
level + fades; the proven edge is the CONFIRMED daily close above the 20-day
high). So we can't trust the backtest's +1% "BREAKOUT alpha" for the live engine.

BO_POC closes that gap by construction: its live entry condition IS the backtest
predicate — `historical_backtest._breakout` (confirmed daily close ≥ prior 20-day
high on ≥1.5× volume). live == predicate → fidelity ~100%. It fires SMALL-size,
isolated, tagged detector_source='BO_POC', strategy_type='bo_poc', once/day after
the close, regardless of regime (regime tagged for later segmentation). Does NOT
touch the production BREAKOUT detector.

Goal: see whether the PROVEN archetype's edge actually shows up live. If it does,
graduate "confirmed-daily-close breakout" into the product; if not, scrutinize the
backtest's cost/look-ahead assumptions.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("signalbolt.bo_poc")

# Same liquid universe the archetype was validated on (megacap + index ETFs).
UNIVERSE = ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "TSLA", "AMD", "AMZN",
            "AVGO", "CRM", "ADBE", "NFLX", "QCOM", "TXN", "MU", "PANW", "CRWD",
            "MRVL", "SMH", "JPM", "XOM", "HD", "WMT", "COST", "UNH", "V",
            "SPY", "QQQ", "IWM"]

_STOP_PCT = 0.08   # backtest exit baseline (8% stop; BE+trail managed live)


def _has_active(sb, ticker: str) -> bool:
    try:
        r = (sb.table("signals").select("id").eq("ticker", ticker)
             .eq("status", "active").eq("strategy_type", "bo_poc")
             .limit(1).execute().data)
        return bool(r)
    except Exception:
        return False


def scan(sb, universe: list | None = None) -> int:
    """Fire BO_POC LONG for every universe name whose LATEST daily bar confirms a
    20-day-high breakout (the exact backtest predicate). Run once after the close.
    Never raises. Returns the count fired."""
    universe = universe or UNIVERSE
    fired = 0
    try:
        from engine import historical_backtest as hb, signal_telemetry, runner
        from engine.alpaca_client import get_bars
        regime = signal_telemetry.live_regime_type()
        for tk in universe:
            try:
                if _has_active(sb, tk):
                    continue
                df = get_bars(tk, "1Day", 60)
                if df is None or len(df) < 22:
                    continue
                # THE exact backtest predicate — live == backtest, so fidelity ~100%
                if hb._breakout(df) != "LONG":
                    continue
                entry = round(float(df["close"].iloc[-1]), 2)
                if entry <= 0:
                    continue
                level = round(float(df["high"].iloc[-21:-1].max()), 2)   # the broken 20d high
                stop = round(entry * (1 - _STOP_PCT), 2)
                t1, t2 = round(entry * 1.06, 2), round(entry * 1.12, 2)
                row = {
                    "ticker": tk, "direction": "LONG", "entry_price": entry,
                    "stop_loss": stop, "target_one": t1, "target_two": t2,
                    "confidence_score": 65,
                    "confidence_factors": ["Confirmed daily close above the 20-day high on volume",
                                           "BO_POC — fidelity-matched to the backtested breakout archetype"],
                    "timeframe": "1Day", "strategy_type": "bo_poc", "status": "active",
                    "management_mode": "engine", "origin": "engine",
                    "ai_explanation": (
                        f"{tk} closed above its 20-day high ({level}) on volume — a CONFIRMED "
                        f"daily-close breakout (the backtested archetype). Entry {entry}, stop {stop} "
                        f"(−8%), targets {t1}/{t2}. Small size — POC to validate the archetype live."
                    ),
                    "regime_type": regime, "session_mode": "",
                    "confidence_tier": "B", "position_multiplier": 0.25,
                    "gamma_net_gex": 0, "gamma_is_negative": False,
                    "manipulation_clean": True, "manipulation_flags": [],
                    "sl_adjustments": [],
                    "risk_reward": round(abs(t1 - entry) / abs(entry - stop), 2) if entry != stop else None,
                    "score_breakdown": {"detector_source": "BO_POC", "breakout_level": level,
                                        "poc": True, "archetype": "confirmed_daily_close_20d_high"},
                    "confidence_grade": "B", "risk_grade": "MEDIUM", "chop_score": 0.0,
                    "setup_type": "bo_poc", "missing_confirmations": [],
                }
                if runner._write_signal(sb, row):
                    fired += 1
            except Exception as e:
                logger.debug(f"[bo_poc] {tk} skipped: {e}")
        logger.info(f"[bo_poc] scan fired {fired} (regime {regime})")
    except Exception as e:
        logger.error(f"[bo_poc] scan failed: {e}")
    return fired
