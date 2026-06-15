"""
Market Pulse — daily orchestrator + one-time backfill.

run_daily(sb): fetch → compute 5 pillars → resolve regime → upsert one row for the
latest completed trading day. Idempotent (upsert on `date`; A/D uses the prior
day's cumulative, so re-runs don't double-count).

run_backfill(sb, days): seed the cumulative A/D line + populate history by replaying
the last `days` trading days from already-fetched bars (so the A/D line is
meaningful and /history charts aren't empty on day one).
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from . import config as C
from . import constituents, data, guidance, pillars, regime, store

logger = logging.getLogger("signalbolt.market_pulse.job")


def _band_or_none(vix: Optional[dict]) -> tuple:
    if not vix:
        return None, None, None, None
    return vix.get("close"), vix.get("sma10"), vix.get("band"), vix.get("rising")


def _build_row(date_iso: str, dd_spy: int, dd_qqq: int, nh: int, nl: int,
               pct50: float, pct200: float, adv: int, dec: int, cum: int,
               div: bool, vix: Optional[dict], reg: str) -> dict:
    vc, vs, vb, vr = _band_or_none(vix)
    return {
        "date": date_iso,
        "dd_count_spy": int(dd_spy), "dd_count_qqq": int(dd_qqq),
        "new_highs": int(nh), "new_lows": int(nl), "net_nhnl": int(nh - nl),
        "pct_above_50": float(pct50), "pct_above_200": float(pct200),
        "ad_line_cumulative": int(cum), "ad_advancers": int(adv), "ad_decliners": int(dec),
        "ad_divergence": bool(div),
        "vix_close": vc, "vix_sma10": vs, "vix_band": vb, "vix_rising": vr,
        "regime": reg, "guidance_key": reg,
    }


def run_daily(sb) -> dict:
    """Compute + upsert today's Market Pulse row. Returns the row (or {} on no data)."""
    spy = data.index_bars("SPY", days=60)
    qqq = data.index_bars("QQQ", days=60)
    if spy is None or len(spy) < 2:
        logger.error("[market_pulse] no SPY bars — aborting daily run")
        return {}
    date_iso = pd.Timestamp(spy.index[-1]).date().isoformat()

    tickers = constituents.sp500_tickers()
    ubars = data.universe_bars(tickers, days=400)
    if not ubars:
        logger.error("[market_pulse] no constituent bars — aborting daily run")
        return {}

    dd_spy = pillars.distribution_days(spy)
    dd_qqq = pillars.distribution_days(qqq)
    nh, nl, _net = pillars.net_new_highs_lows(ubars)
    pct50, pct200 = pillars.pct_above_mas(ubars)
    adv, dec, adnet = pillars.advance_decline(ubars)

    cum = store.cumulative_before(sb, date_iso) + adnet
    div = pillars.ad_divergence(spy, cum, store.recent_ad_history(sb))

    vix = pillars.vix_read(data.vix_closes())   # isolated; None on failure

    reg = regime.resolve(
        dd_max=max(dd_spy, dd_qqq), net_nhnl=nh - nl,
        pct_above_50=pct50, pct_above_200=pct200, ad_divergence=div,
        vix_level=(vix or {}).get("close"), vix_rising=(vix or {}).get("rising"),
    )
    row = _build_row(date_iso, dd_spy, dd_qqq, nh, nl, pct50, pct200, adv, dec, cum, div, vix, reg)
    store.upsert_daily(sb, row)
    logger.info(f"[market_pulse] {date_iso} regime={reg} dd_max={max(dd_spy, dd_qqq)} "
                f"nh/nl={nh}/{nl} %50={pct50} %200={pct200} vix={(vix or {}).get('close')}")
    return row


def run_backfill(sb, days: int = 120) -> dict:
    """Replay the last `days` trading days from one bulk fetch to seed the cumulative
    A/D line + populate history. Idempotent (cumulative restarts at 0 at the window
    open and is deterministic). VIX is backfilled from its own daily series."""
    spy = data.index_bars("SPY", days=900)
    qqq = data.index_bars("QQQ", days=900)
    if spy is None or len(spy) < C.HL_LOOKBACK + 2:
        logger.error("[market_pulse] backfill: insufficient SPY history")
        return {"written": 0}

    tickers = constituents.sp500_tickers()
    ubars = data.universe_bars(tickers, days=900)
    if not ubars:
        logger.error("[market_pulse] backfill: no constituent bars")
        return {"written": 0}

    vix_full = data.vix_closes(lookback=900)   # may be None

    dates = [pd.Timestamp(d).date() for d in spy.index]
    start_idx = max(C.HL_LOOKBACK, len(dates) - days)   # need a full 52w window behind each replayed day
    cum = 0
    written = 0
    for i in range(start_idx, len(dates)):
        d = dates[i]
        d_ts = pd.Timestamp(d)
        # Slice every series to "as of day d" so each pillar sees only past data.
        spy_d = spy[spy.index <= d_ts]
        qqq_d = qqq[qqq.index <= d_ts]
        ubars_d = {t: df[df.index <= d_ts] for t, df in ubars.items() if df is not None}

        dd_spy = pillars.distribution_days(spy_d)
        dd_qqq = pillars.distribution_days(qqq_d)
        nh, nl, _ = pillars.net_new_highs_lows(ubars_d)
        pct50, pct200 = pillars.pct_above_mas(ubars_d)
        adv, dec, adnet = pillars.advance_decline(ubars_d)
        cum += adnet

        vix_d = None
        if vix_full is not None:
            try:
                vser = vix_full[[pd.Timestamp(ix).date() <= d for ix in vix_full.index]]
                vix_d = pillars.vix_read(vser)
            except Exception:
                vix_d = None

        reg = regime.resolve(
            dd_max=max(dd_spy, dd_qqq), net_nhnl=nh - nl,
            pct_above_50=pct50, pct_above_200=pct200, ad_divergence=False,  # divergence needs forward A/D history; off during seed
            vix_level=(vix_d or {}).get("close"), vix_rising=(vix_d or {}).get("rising"),
        )
        row = _build_row(d.isoformat(), dd_spy, dd_qqq, nh, nl, pct50, pct200, adv, dec, cum, False, vix_d, reg)
        if store.upsert_daily(sb, row):
            written += 1
    logger.info(f"[market_pulse] backfill wrote {written} days")
    return {"written": written}
