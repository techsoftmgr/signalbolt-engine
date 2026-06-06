"""
Module #4 — Market Threat Radar.

A simple, daily-use market-risk dashboard: GREEN / YELLOW / ORANGE / RED + a
0-100 threat score, built from factors the platform already computes (VIX, trend,
breadth, sector rotation, vol regime, earnings risk) + a plain-English summary.

ADDITIVE + READ-ONLY: only reads existing data sources; never writes; never
raises (every fetch fails safe to a neutral value). `_aggregate()` is PURE →
unit-tested. Educational wording only — no advice language.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("signalbolt.phase2.threat_radar")

# breadth universe (liquid, sector-diverse) + sector-rotation ETFs
_BREADTH = ["AAPL", "MSFT", "NVDA", "META", "AMZN", "JPM", "XOM", "UNH", "HD",
            "WMT", "JNJ", "PG", "KO", "CAT", "BA", "DIS", "GS", "AMD", "CRM", "NFLX"]
_DEFENSIVE = ["XLU", "XLP", "XLV"]
_CYCLICAL = ["XLK", "XLY", "XLF"]


# ── pure per-factor threat scores (0 = calm, 100 = max threat) ──

def _vix_threat(vix, chg_pct):
    try:
        vix = float(vix)
    except (TypeError, ValueError):
        return 40
    base = 0 if vix < 15 else 20 if vix < 20 else 45 if vix < 25 else 70 if vix < 30 else 90
    try:
        if chg_pct is not None and float(chg_pct) > 10:   # intraday spike
            base = min(100, base + 20)
    except (TypeError, ValueError):
        pass
    return base


def _trend_threat(above_200ma, off_high_pct):
    o = None
    try:
        o = float(off_high_pct) if off_high_pct is not None else None
    except (TypeError, ValueError):
        o = None
    if above_200ma:
        return 10 if (o is None or o > -5) else 30
    return 70 if (o is None or o > -15) else 90   # below 200-DMA


def _breadth_threat(pct_above_50):
    if pct_above_50 is None:
        return 40
    return int(round(max(0, min(100, (60 - pct_above_50) * 2.0))))   # 60%+ healthy → 0; <30% → high


def _sector_threat(defensive_lead_pct):
    if defensive_lead_pct is None:
        return 40
    return int(round(max(0, min(100, 40 + defensive_lead_pct * 10))))  # defensives leading → risk-off


def _vol_regime_threat(regime_type):
    return {"PANIC": 90, "HIGH_VOL": 70, "TRENDING_BEAR": 65, "RISK_OFF": 60,
            "RANGING": 40, "LOW_VOL": 12, "TRENDING_BULL": 15}.get(regime_type, 40)


def _earnings_threat(n_major):
    if n_major is None:
        return 25
    return int(round(max(0, min(100, n_major * 12))))


def _summary(level, reasons):
    if level == "GREEN":
        return ("Market conditions look calm — low volatility, healthy breadth, and a "
                "constructive trend. A generally supportive backdrop, though risk management "
                "always matters.")
    if level == "YELLOW":
        head = "Mixed conditions — some caution warranted. "
    elif level == "ORANGE":
        head = "Elevated market risk — conditions favor a defensive posture. "
    else:
        head = "High market risk — a hostile environment for trading. Reduced size and tighter "\
               "risk control are worth considering. "
    if reasons:
        head += "Key factors: " + "; ".join(reasons[:3]) + "."
    return head


def _aggregate(factors: list) -> dict:
    """Pure: weighted 0-100 threat score + level + reasons + summary from the
    per-factor scores. factors = [{key,label,threat,weight,detail}]."""
    fs = [f for f in factors if f.get("threat") is not None and f.get("weight")]
    tw = sum(f["weight"] for f in fs)
    score = int(round(sum(f["threat"] * f["weight"] for f in fs) / tw)) if tw else 40
    level = ("GREEN" if score < 25 else "YELLOW" if score < 50
             else "ORANGE" if score < 75 else "RED")
    reasons = [f"{f['label']} ({f['detail']})"
               for f in sorted(fs, key=lambda x: -x["threat"]) if f["threat"] >= 50][:4]
    return {"threat_score": score, "level": level, "reasons": reasons,
            "summary": _summary(level, reasons),
            "factors": [{"key": f["key"], "label": f["label"], "threat": f["threat"],
                         "detail": f["detail"]} for f in factors]}


# ── data gathering (I/O, all fail-safe) ──

def _breadth_pct():
    try:
        from engine.alpaca_client import get_multi_bars
        bars = get_multi_bars(_BREADTH, "1Day", 80)
        if not bars:
            return None
        above = tot = 0
        for df in bars.values():
            if df is None or len(df) < 50:
                continue
            tot += 1
            if float(df["close"].iloc[-1]) > float(df["close"].rolling(50).mean().iloc[-1]):
                above += 1
        return round(100 * above / tot, 0) if tot else None
    except Exception as e:
        logger.debug(f"[threat] breadth failed: {e}")
        return None


def _sector_lead():
    """Defensive 10-day return minus cyclical 10-day return (+ = risk-off rotation)."""
    try:
        from engine.alpaca_client import get_multi_bars
        bars = get_multi_bars(_DEFENSIVE + _CYCLICAL, "1Day", 20)
        if not bars:
            return None

        def _ret(group):
            rs = []
            for tk in group:
                df = bars.get(tk)
                if df is not None and len(df) > 11:
                    c = df["close"]
                    rs.append((float(c.iloc[-1]) / float(c.iloc[-11]) - 1) * 100)
            return sum(rs) / len(rs) if rs else None
        d, c = _ret(_DEFENSIVE), _ret(_CYCLICAL)
        return round(d - c, 2) if (d is not None and c is not None) else None
    except Exception as e:
        logger.debug(f"[threat] sector lead failed: {e}")
        return None


def _earnings_count():
    try:
        from engine import earnings_service
        wk = earnings_service.get_weekly_earnings() or {}
        events = wk.get("events") or wk.get("items") or []
        # "soon" = next ~2 calendar days
        return min(len(events), 12)
    except Exception as e:
        logger.debug(f"[threat] earnings failed: {e}")
        return None


def compute(sb=None) -> dict:
    """Build the threat radar. Never raises."""
    try:
        vix = chg = regime_type = None
        above_200ma = True
        try:
            from engine import regime_detector
            r = regime_detector.detect() or {}
            vix = r.get("vix"); chg = r.get("vix_change_pct")
            regime_type = r.get("regime_type"); above_200ma = r.get("above_200ma", True)
        except Exception as e:
            logger.debug(f"[threat] regime failed: {e}")

        off_high = None
        try:
            from engine import drawdown_regime
            dd = drawdown_regime.assess() or {}
            off_high = dd.get("off_high_pct")
        except Exception as e:
            logger.debug(f"[threat] drawdown failed: {e}")

        breadth = _breadth_pct()
        sector_lead = _sector_lead()
        n_earn = _earnings_count()

        factors = [
            {"key": "vix", "label": "Volatility (VIX)", "weight": 0.25,
             "threat": _vix_threat(vix, chg),
             "detail": f"VIX {round(float(vix),1) if vix else '?'}"
                       + (f", +{round(float(chg),1)}% today" if chg and float(chg) > 0 else "")},
            {"key": "trend", "label": "Market trend", "weight": 0.20,
             "threat": _trend_threat(above_200ma, off_high),
             "detail": ("above 200-DMA" if above_200ma else "below 200-DMA")
                       + (f", {off_high}% off high" if off_high is not None else "")},
            {"key": "breadth", "label": "Breadth", "weight": 0.20,
             "threat": _breadth_threat(breadth),
             "detail": f"{int(breadth)}% of leaders above 50-DMA" if breadth is not None else "n/a"},
            {"key": "sector_rotation", "label": "Sector rotation", "weight": 0.15,
             "threat": _sector_threat(sector_lead),
             "detail": ("defensives leading" if (sector_lead or 0) > 0.5
                        else "cyclicals leading" if (sector_lead or 0) < -0.5 else "neutral")
                       + (f" ({sector_lead:+.1f}% 10d)" if sector_lead is not None else "")},
            {"key": "vol_regime", "label": "Volatility regime", "weight": 0.12,
             "threat": _vol_regime_threat(regime_type),
             "detail": regime_type or "unknown"},
            {"key": "earnings", "label": "Earnings risk", "weight": 0.08,
             "threat": _earnings_threat(n_earn),
             "detail": f"{n_earn} notable this week" if n_earn is not None else "n/a"},
        ]
        out = _aggregate(factors)
        out["enabled"] = True
        return out
    except Exception as e:
        logger.error(f"[threat] compute failed: {e}")
        return {"enabled": True, "threat_score": None, "level": "UNKNOWN",
                "summary": "Threat radar temporarily unavailable.", "factors": [], "error": str(e)}
