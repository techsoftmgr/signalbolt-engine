"""
Crash / deep-value long-term BUY signal (backlog #10) — the COMBINE step.
==========================================================================
Fires ONLY when ALL of these hold (falling-knife defense, layered):
  1. REGIME — the broad market is in an accumulation-window drawdown
     (drawdown_regime: SPY <= -20% off its 52-wk high). Macro fear, not a
     company-specific break.
  2. QUALITY — the name is fundamentally strong (fundamentals quality_score >=
     _MIN_QUALITY): low debt, profitable, FCF+, growing → survives the downturn.
  3. DEEPLY DISCOUNTED — the name itself is >= _MIN_STOCK_DRAWDOWN off its own
     52-wk high (a real markdown, not a mild dip).
  4. TURN CONFIRMED — turnaround_detector stage == 'buyzone': it has STOPPED
     falling and is turning up. THIS is the knife guard — buy quality that has
     stopped falling, not mid-freefall.

Fired with management_mode='manual' so the swing engine never applies tight
stops / 10-day expiry to a months-horizon hold (runner expiry now skips manual
too). Scale-in + exit are the holder's — loose by design. Educational, not advice.

Regime-gated → dormant until a real drawdown (won't fire in a healthy market).
"""
from __future__ import annotations

import logging

logger = logging.getLogger("signalbolt.deep_value")

_MIN_QUALITY        = 4       # fundamentals 0-5 score
_MIN_STOCK_DRAWDOWN = -25.0   # the name itself must be >= 25% off its 52-wk high
_MAX_FIRES          = 15      # normal cap per run — a basket, not the whole list
_MAX_FIRES_DEEP     = 30      # DEEP-BEAR cap (SPY <= -30% off high): bigger basket
_POS_MULT_DEEP      = 1.0     # full size in a deep-bear washout (historically strongest buys)
_POS_MULT_NORMAL    = 0.5     # half size in an ordinary accumulation window — scale in


def _stock_drawdown_and_high(df) -> tuple:
    """(% off 52-wk high [negative], 52-wk high) or (None, None)."""
    if df is None or len(df) < 200 or "high" not in df.columns or "close" not in df.columns:
        return None, None
    hi = float(df["high"].tail(252).max())
    last = float(df["close"].iloc[-1])
    if hi <= 0:
        return None, None
    return round((last / hi - 1) * 100, 1), hi


def generate(sb) -> dict:
    """Run the combine. Returns {fired, count, regime, reason}."""
    from engine import drawdown_regime, fundamentals, runner, push
    from engine import alpaca_client as ac
    from engine.turnaround_detector import score_turnaround

    regime = drawdown_regime.assess()
    if not regime.get("accumulation_window"):
        logger.info(f"[deep_value] window CLOSED (regime={regime.get('regime')}, "
                    f"SPY {regime.get('off_high_pct')}% off) — no fires")
        return {"fired": [], "count": 0, "regime": regime.get("regime"),
                "reason": "accumulation_window_closed"}

    # DEEP-BEAR escalation: SPY <= -30% off its high (regime["deep"]). These are
    # historically the strongest long-term entries → bigger basket + full size.
    deep = bool(regime.get("deep"))
    max_fires = _MAX_FIRES_DEEP if deep else _MAX_FIRES
    pos_mult  = _POS_MULT_DEEP if deep else _POS_MULT_NORMAL

    candidates = fundamentals.get_ranked(sb, min_score=_MIN_QUALITY)
    fired: list[str] = []
    for c in candidates:
        if len(fired) >= max_fires:
            break
        tk = c["ticker"]
        qscore = c.get("quality_score") or 0
        try:
            if runner._has_active_signal(sb, tk, "deep_value"):
                continue
            df = ac.get_bars(tk, "1Day", days=400)
            dd, hi52 = _stock_drawdown_and_high(df)
            if dd is None or dd > _MIN_STOCK_DRAWDOWN:      # not deeply discounted enough
                continue
            ta = score_turnaround(df, regime_type=regime.get("regime"))
            if not ta or ta.get("stage") != "buyzone":      # ── FALLING-KNIFE GATE ──
                continue

            price = round(float(df["close"].iloc[-1]), 2)
            # WIDE structural disaster floor (informational — manual mode, engine
            # won't enforce it): the deeper of -20% or the recent 60-day low.
            recent_low = float(df["low"].tail(60).min())
            stop = round(min(price * 0.80, recent_low * 0.98), 2)
            # Recovery targets toward the prior 52-wk high.
            t1 = round(hi52 * 0.70, 2)
            t2 = round(hi52 * 0.95, 2)
            conf = min(90 if deep else 85, 70 + qscore * 3 + (5 if deep else 0))

            factors = [f"Quality {qscore}/5", f"{dd:.0f}% off 52w high",
                       "Confirmed turn (buyzone)", "Market drawdown regime"]
            if deep:
                factors.append("DEEP-BEAR washout — size up")
            deep_note = (
                f" DEEP-BEAR ({regime.get('off_high_pct')}% off): historically the strongest "
                f"long-term entries — full size, scale in aggressively."
                if deep else
                " Half size — scale in over weeks, don't lump-sum."
            )

            row = {
                "ticker": tk, "direction": "LONG",
                "entry_price": price, "stop_loss": stop, "target_one": t1, "target_two": t2,
                "confidence_score": conf,
                "confidence_factors": factors,
                "timeframe": "1Day", "strategy_type": "deep_value", "status": "active",
                "management_mode": "manual",   # months-horizon hold; engine hands-off
                "position_multiplier": pos_mult,   # 1.0 deep-bear / 0.5 normal accumulation
                "ai_explanation": (
                    f"Deep-value long-term buy: {tk} is {dd:.0f}% off its 52-week high while the "
                    f"market is {regime.get('off_high_pct')}% off (accumulation window). "
                    f"Quality {qscore}/5 (low debt, profitable, FCF+), and it's showing a confirmed "
                    f"turn — not mid-freefall. Multi-month hold.{deep_note} "
                    f"Disaster floor ~{stop}. Educational, not financial advice."
                ),
                "score_breakdown": {
                    "detector_source": "DEEP_VALUE", "quality_score": qscore,
                    "stock_drawdown_pct": dd, "market_off_high_pct": regime.get("off_high_pct"),
                    "turnaround_stage": ta.get("stage"), "turnaround_score": ta.get("score"),
                    "high_52w": round(hi52, 2),
                    "deep_regime": deep, "position_multiplier": pos_mult,
                },
            }
            sid = runner._write_signal(sb, row)
            if sid:
                fired.append(tk)
                logger.info(f"[deep_value] FIRED {tk} q={qscore} dd={dd}% turn=buyzone")
        except Exception as e:
            logger.debug(f"[deep_value] {tk} failed: {e}")

    if fired:
        # Broadcast to ALL users (a rare, market-wide opportunity worth surfacing
        # everyone) + drop it in the in-app Alerts feed. Type 'deep_value' is not
        # in _TYPE_TO_PREF, so _send_raw goes to every registered token.
        deep_tag = " · DEEP-BEAR (size up)" if deep else ""
        title = f"💎 Deep-value buy window OPEN — {len(fired)} names{deep_tag}"
        more = "…" if len(fired) > 10 else ""
        body = (f"Market {regime.get('off_high_pct')}% off highs. Quality names that "
                f"have stopped falling: " + ", ".join(fired[:10]) + more)
        data = {"type": "deep_value", "signal": "deep_value", "count": len(fired),
                "deep": deep, "tickers": fired[:15]}
        try:
            push._record_alert(type_="deep_value", ticker=None, title=title, body=body,
                               stage=("deep" if deep else "open"), data=data, sb=sb)
        except Exception:
            pass
        try:
            push._send_raw(title=title, body=body, data=data)
        except Exception:
            pass
    logger.info(f"[deep_value] done — fired {len(fired)} (regime={regime.get('regime')}, "
                f"deep={deep})")
    return {"fired": fired, "count": len(fired), "regime": regime.get("regime"),
            "reason": "fired" if fired else "no_qualifying_names"}
