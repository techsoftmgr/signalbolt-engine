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


def _build_row(date_iso: str, dd_spy: int, dd_qqq: int, st_spy: int, st_qqq: int,
               nh: int, nl: int, pct50: float, pct200: float, adv: int, dec: int, cum: int,
               div: bool, vix: Optional[dict], reg: str,
               breadth_ratio: Optional[float] = None, breadth_thrust: bool = False) -> dict:
    vc, vs, vb, vr = _band_or_none(vix)
    eff_spy = dd_spy + C.STALL_WEIGHT * st_spy
    eff_qqq = dd_qqq + C.STALL_WEIGHT * st_qqq
    return {
        "breadth_ratio": breadth_ratio, "breadth_thrust": bool(breadth_thrust),
        "date": date_iso,
        "dd_count_spy": int(dd_spy), "dd_count_qqq": int(dd_qqq),
        "stall_count_spy": int(st_spy), "stall_count_qqq": int(st_qqq),
        "effective_dd_spy": round(float(eff_spy), 1), "effective_dd_qqq": round(float(eff_qqq), 1),
        "new_highs": int(nh), "new_lows": int(nl), "net_nhnl": int(nh - nl),
        "pct_above_50": float(pct50), "pct_above_200": float(pct200),
        "ad_line_cumulative": int(cum), "ad_advancers": int(adv), "ad_decliners": int(dec),
        "ad_divergence": bool(div),
        "vix_close": vc, "vix_sma10": vs, "vix_band": vb, "vix_rising": vr,
        "regime": reg, "guidance_key": reg,
    }


def _effective_dd_max(dd_spy: int, dd_qqq: int, st_spy: int, st_qqq: int) -> int:
    """Floored max effective distribution (distribution + STALL_WEIGHT*stalling)
    across SPY/QQQ — what the regime thresholds (5/6) compare against. With zero
    stalling days this equals dd_max, so Phase-1 behavior is unchanged."""
    import math
    eff = max(dd_spy + C.STALL_WEIGHT * st_spy, dd_qqq + C.STALL_WEIGHT * st_qqq)
    return int(math.floor(eff))


def run_daily(sb) -> dict:
    """Compute + upsert the Market Pulse row for the last SETTLED session. Returns
    the row (or {} on no data).

    A daily bar is only final after that day's close. If the latest bar is TODAY
    and the regular session hasn't closed yet (before 4pm ET), it's a FORMING bar —
    we drop it and read the prior completed session, so an early/manual run reflects
    the last close, not 12 minutes of today. The 4:45pm ET cron runs after the
    close, so it naturally keeps the (now settled) same-day bar."""
    spy = data.index_bars("SPY", days=60)
    qqq = data.index_bars("QQQ", days=60)
    if spy is None or len(spy) < 2:
        logger.error("[market_pulse] no SPY bars — aborting daily run")
        return {}

    from datetime import datetime as _dt, timezone as _tz
    try:
        from zoneinfo import ZoneInfo as _ZI
        now_et = _dt.now(_ZI("America/New_York"))
    except Exception:
        now_et = _dt.now(_tz.utc)
    last_date = pd.Timestamp(spy.index[-1]).date()
    forming = last_date == now_et.date() and now_et.hour < 16
    if forming:
        cutoff = pd.Timestamp(now_et.date()).tz_localize("UTC")              # exclude today's forming bar
    else:
        cutoff = pd.Timestamp(last_date).tz_localize("UTC") + pd.Timedelta(days=1)   # keep the latest (settled) bar

    spy = spy[spy.index < cutoff]
    if qqq is not None:
        qqq = qqq[qqq.index < cutoff]
    if spy is None or len(spy) < 2:
        logger.error("[market_pulse] no settled SPY bar — aborting daily run")
        return {}
    date_iso = pd.Timestamp(spy.index[-1]).date().isoformat()

    tickers = constituents.sp500_tickers()
    ubars_raw = data.universe_bars(tickers, days=400)
    if not ubars_raw:
        logger.error("[market_pulse] no constituent bars — aborting daily run")
        return {}
    ubars = {t: df[df.index < cutoff] for t, df in ubars_raw.items() if df is not None}

    dd_spy = pillars.distribution_days(spy)
    dd_qqq = pillars.distribution_days(qqq)
    st_spy = pillars.stalling_days(spy)
    st_qqq = pillars.stalling_days(qqq)
    nh, nl, _net = pillars.net_new_highs_lows(ubars)
    pct50, pct200 = pillars.pct_above_mas(ubars)
    adv, dec, adnet = pillars.advance_decline(ubars)

    cum = store.cumulative_before(sb, date_iso) + adnet
    div = pillars.ad_divergence(spy, cum, store.recent_ad_history(sb))
    br_ratio, br_thrust = pillars.breadth_thrust(store.recent_breadth(sb, 30, before=date_iso) + [(adv, dec)])

    vix = pillars.vix_read(data.vix_closes())   # isolated; None on failure

    eff_max = _effective_dd_max(dd_spy, dd_qqq, st_spy, st_qqq)
    reg = regime.resolve(
        dd_max=eff_max, net_nhnl=nh - nl,
        pct_above_50=pct50, pct_above_200=pct200, ad_divergence=div,
        vix_level=(vix or {}).get("close"), vix_rising=(vix or {}).get("rising"),
    )
    row = _build_row(date_iso, dd_spy, dd_qqq, st_spy, st_qqq, nh, nl, pct50, pct200, adv, dec, cum, div, vix, reg,
                     breadth_ratio=br_ratio, breadth_thrust=br_thrust)
    store.upsert_daily(sb, row)
    logger.info(f"[market_pulse] {date_iso} regime={reg} eff_dd={eff_max} (dd {dd_spy}/{dd_qqq} stall {st_spy}/{st_qqq}) "
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
    # Drop a still-forming final bar (today before the close) — only replay SETTLED
    # sessions, else the backfill writes a partial intraday row for today.
    from datetime import datetime as _dt
    try:
        from zoneinfo import ZoneInfo as _ZI
        _now_et = _dt.now(_ZI("America/New_York"))
    except Exception:
        _now_et = None
    if dates and _now_et is not None and dates[-1] == _now_et.date() and _now_et.hour < 16:
        dates = dates[:-1]
    start_idx = max(C.HL_LOOKBACK, len(dates) - days)   # need a full 52w window behind each replayed day
    cum = 0
    written = 0
    bt_pairs: list[tuple] = []   # rolling (adv, decl) for the breadth-thrust EMA
    for i in range(start_idx, len(dates)):
        d = dates[i]
        # Alpaca bar indexes are tz-aware (UTC); use a tz-aware exclusive cutoff at
        # the NEXT UTC midnight so all of day d's bar is included regardless of its
        # intraday timestamp (a naive cutoff raises a tz-comparison TypeError).
        cutoff = pd.Timestamp(d).tz_localize("UTC") + pd.Timedelta(days=1)
        spy_d = spy[spy.index < cutoff]
        qqq_d = qqq[qqq.index < cutoff]
        ubars_d = {t: df[df.index < cutoff] for t, df in ubars.items() if df is not None}

        dd_spy = pillars.distribution_days(spy_d)
        dd_qqq = pillars.distribution_days(qqq_d)
        st_spy = pillars.stalling_days(spy_d)
        st_qqq = pillars.stalling_days(qqq_d)
        nh, nl, _ = pillars.net_new_highs_lows(ubars_d)
        pct50, pct200 = pillars.pct_above_mas(ubars_d)
        adv, dec, adnet = pillars.advance_decline(ubars_d)
        cum += adnet
        bt_pairs.append((adv, dec))
        br_ratio, br_thrust = pillars.breadth_thrust(bt_pairs[-30:])

        vix_d = None
        if vix_full is not None:
            try:
                vser = vix_full[[pd.Timestamp(ix).date() <= d for ix in vix_full.index]]
                vix_d = pillars.vix_read(vser)
            except Exception:
                vix_d = None

        reg = regime.resolve(
            dd_max=_effective_dd_max(dd_spy, dd_qqq, st_spy, st_qqq), net_nhnl=nh - nl,
            pct_above_50=pct50, pct_above_200=pct200, ad_divergence=False,  # divergence needs forward A/D history; off during seed
            vix_level=(vix_d or {}).get("close"), vix_rising=(vix_d or {}).get("rising"),
        )
        row = _build_row(d.isoformat(), dd_spy, dd_qqq, st_spy, st_qqq, nh, nl, pct50, pct200, adv, dec, cum, False, vix_d, reg,
                         breadth_ratio=br_ratio, breadth_thrust=br_thrust)
        if store.upsert_daily(sb, row):
            written += 1
    logger.info(f"[market_pulse] backfill wrote {written} days")
    return {"written": written}
