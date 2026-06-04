"""
Fundamentals quality screen (SEC EDGAR — free, unlimited, commercial-safe).
============================================================================
Productionizes the prototype: robust XBRL tag normalization, a quality score,
and a quarterly-cached, rolling-refreshed universe. Feeds the crash/deep-value
long-term signal (backlog #10) — the "WHICH names are quality" half.

Why the prototype's margins were wrong (META 148%, MSFT 163%): it grabbed the
FIRST revenue XBRL tag that had data, which is sometimes a partial/segment value
→ tiny revenue → absurd margin. FIX here: for flow items (revenue/NI/OCF/CapEx)
take the LARGEST annual value per fiscal year across all candidate tags (total
revenue > any component), prefer SEC's annual `frame` (CYxxxx) values, and apply
sanity guards (a >80% net margin = wrong revenue → drop the metric).

compute_metrics(facts) is PURE (given a companyfacts dict) so it's unit-testable
without network. screen_ticker()/refresh_universe() do the IO + caching.
"""
from __future__ import annotations

import logging
import os
import time

import requests

logger = logging.getLogger("signalbolt.fundamentals")

_UA = {"User-Agent": "SignalBolt research techsoftmgr@gmail.com"}

# Candidate us-gaap tags (priority + we take the max annual across them for flows).
_REV   = ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
          "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"]
_NI    = ["NetIncomeLoss", "ProfitLoss"]
_EQ    = ["StockholdersEquity",
          "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]
_DEBT_LT  = ["LongTermDebtNoncurrent", "LongTermDebt"]
_DEBT_CUR = ["LongTermDebtCurrent", "DebtCurrent"]
_OCF   = ["NetCashProvidedByUsedInOperatingActivities",
          "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"]
_CAPEX = ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"]

# Sanity bounds — outside these the extracted value is almost certainly the wrong
# XBRL tag, so we drop the metric rather than pollute the score.
_MARGIN_MAX = 80.0      # real net margins top out ~55-60%; >80% = bad revenue
_MARGIN_MIN = -300.0
_GROWTH_MAX = 400.0     # allow hypergrowth (NVDA ~+65%) but reject parsing blowups


# ── XBRL extraction (pure) ─────────────────────────────────────────────────
def _usd_entries(facts: dict, tag: str) -> list[dict]:
    node = (facts.get("facts", {}).get("us-gaap", {}) or {}).get(tag)
    if not node:
        return []
    return node.get("units", {}).get("USD", []) or []


def _annual_by_fy(facts: dict, tags: list[str]) -> dict:
    """{fiscal_year: value} for FLOW items (revenue/NI/OCF/CapEx).
    Uses annual (10-K / fp=FY) entries; across all candidate tags + entries,
    keeps the LARGEST value per fiscal year (total > component), with a slight
    preference for SEC's annual `frame` (CYxxxx, not CYxxxxQx) values."""
    best: dict[int, tuple] = {}   # fy -> (frame_rank, value)
    for tag in tags:
        for e in _usd_entries(facts, tag):
            val = e.get("val")
            fy = e.get("fy")
            if val is None or fy is None:
                continue
            if not str(e.get("form", "")).startswith("10-K"):
                continue
            if e.get("fp") != "FY":
                continue
            frame = str(e.get("frame", ""))
            frame_rank = 1 if (frame.startswith("CY") and "Q" not in frame) else 0
            cand = (frame_rank, float(val))
            if fy not in best or cand > best[fy]:
                best[fy] = cand
    return {fy: v[1] for fy, v in best.items()}


def _latest_two(series: dict) -> tuple:
    """(latest_value, prior_value) by fiscal year, or (None, None)."""
    if not series:
        return None, None
    fys = sorted(series.keys(), reverse=True)
    latest = series[fys[0]]
    prior = series[fys[1]] if len(fys) > 1 else None
    return latest, prior


def _latest_point(facts: dict, tags: list[str]):
    """Most recent balance-sheet value (by 'end' date) across candidate tags."""
    best = None  # (end, value)
    for tag in tags:
        for e in _usd_entries(facts, tag):
            end = e.get("end")
            val = e.get("val")
            if end and val is not None and (best is None or end > best[0]):
                best = (end, float(val))
    return best[1] if best else None


def compute_metrics(facts: dict) -> dict:
    """Pure quality metrics from a companyfacts dict. Network-free → unit-testable."""
    rev_l, rev_p = _latest_two(_annual_by_fy(facts, _REV))
    ni_l, _      = _latest_two(_annual_by_fy(facts, _NI))
    ocf_l, _     = _latest_two(_annual_by_fy(facts, _OCF))
    cap_l, _     = _latest_two(_annual_by_fy(facts, _CAPEX))
    equity       = _latest_point(facts, _EQ)
    debt_lt      = _latest_point(facts, _DEBT_LT)
    debt_cur     = _latest_point(facts, _DEBT_CUR)
    debt = None
    if debt_lt is not None or debt_cur is not None:
        debt = (debt_lt or 0.0) + (debt_cur or 0.0)

    # Net margin (with sanity guard against wrong-revenue parsing)
    net_margin = None
    if ni_l is not None and rev_l and rev_l > 0:
        m = ni_l / rev_l * 100
        net_margin = round(m, 2) if (_MARGIN_MIN <= m <= _MARGIN_MAX) else None

    roe = round(ni_l / equity * 100, 2) if (ni_l is not None and equity and equity > 0) else None
    de  = round(debt / equity, 3) if (debt is not None and equity and equity > 0) else None

    growth = None
    if rev_l is not None and rev_p and rev_p > 0:
        g = (rev_l - rev_p) / rev_p * 100
        growth = round(g, 2) if abs(g) <= _GROWTH_MAX else None

    fcf = (ocf_l - cap_l) if (ocf_l is not None and cap_l is not None) else None

    return {
        "net_margin": net_margin, "roe": roe, "debt_to_equity": de,
        "revenue_growth": growth, "fcf": fcf,
        "fcf_positive": (fcf is not None and fcf > 0),
        "revenue_latest": rev_l, "net_income_latest": ni_l, "equity": equity, "debt": debt,
    }


def quality_score(m: dict) -> int:
    """0-5 pass count. Missing metric = fail (conservative — don't rank on gaps)."""
    return sum(1 for ok in (
        m.get("net_margin") is not None and m["net_margin"] >= 10,
        m.get("roe") is not None and m["roe"] >= 15,
        m.get("debt_to_equity") is not None and m["debt_to_equity"] < 1.0,
        m.get("revenue_growth") is not None and m["revenue_growth"] > 0,
        bool(m.get("fcf_positive")),
    ) if ok)


# ── CIK map (cached in-process, refreshed daily) ───────────────────────────
_cik_cache: dict | None = None
_cik_cache_ts = 0.0


def cik_map() -> dict:
    global _cik_cache, _cik_cache_ts
    if _cik_cache is not None and (time.time() - _cik_cache_ts) < 86400:
        return _cik_cache
    r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=_UA, timeout=20)
    r.raise_for_status()
    _cik_cache = {row["ticker"].upper(): str(row["cik_str"]).zfill(10) for row in r.json().values()}
    _cik_cache_ts = time.time()
    return _cik_cache


def _companyfacts(cik: str) -> dict | None:
    r = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", headers=_UA, timeout=30)
    if r.status_code != 200:
        return None
    return r.json()


def screen_ticker(ticker: str) -> dict | None:
    """Fetch + compute metrics + score for one ticker. None if no EDGAR data."""
    tk = ticker.upper()
    cik = cik_map().get(tk)
    if not cik:
        return None
    facts = _companyfacts(cik)
    if not facts:
        return None
    m = compute_metrics(facts)
    m["ticker"] = tk
    m["quality_score"] = quality_score(m)
    return m


# ── Quality-candidate universe (curated large/mega-caps; extensible to S&P 500) ──
QUALITY_UNIVERSE = [
    # mega-cap tech / comms
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","AVGO","ADBE","CRM","ORCL","CSCO","ACN","TXN",
    "QCOM","AMD","INTU","NOW","AMAT","LRCX","KLAC","MU","ADI","SNPS","CDNS","PANW","CRWD","FTNT",
    "NFLX","DIS","CMCSA","TMUS","V","MA","PYPL","FISV","ADP",
    # consumer
    "COST","WMT","HD","LOW","MCD","SBUX","NKE","TGT","PG","KO","PEP","MDLZ","CL","KMB","EL","MO","PM",
    "BKNG","ABNB","CMG","ORLY","AZO","YUM","TJX","ROST","DG","DLTR",
    # healthcare
    "UNH","LLY","JNJ","ABBV","MRK","PFE","TMO","ABT","DHR","BMY","AMGN","GILD","ISRG","VRTX","REGN",
    "MDT","SYK","BSX","CI","CVS","HCA","ZTS","ELV","HUM",
    # financials
    "BRK.B","JPM","BAC","WFC","GS","MS","C","SCHW","BLK","SPGI","CME","ICE","AXP","CB","PGR","MMC","AON","USB","PNC","TFC",
    # industrials / energy / materials
    "CAT","DE","HON","UNP","UPS","BA","GE","LMT","RTX","NOC","GD","ETN","EMR","ITW","PH","MMM","CSX","NSC",
    "XOM","CVX","COP","SLB","EOG","MPC","PSX","VLO","OXY","WMB","KMI",
    "LIN","APD","SHW","ECL","FCX","NEM","NUE","DOW",
    # utilities / re / staples extra
    "NEE","DUK","SO","D","AEP","EXC","SRE","PLD","AMT","EQIX","CCI","PSA","O","SPG",
]


# ── Cache (Supabase) + rolling refresh ─────────────────────────────────────
def refresh_universe(sb, batch: int = 15, sleep_s: float = 0.2) -> dict:
    """Refresh the `batch` stalest universe tickers into fundamentals_cache.
    Rolling: a few runs/day keeps the whole universe within the quarterly cadence
    fundamentals actually change on. SEC-polite (sleep between fetches)."""
    from datetime import datetime, timezone
    try:
        cached = (sb.table("fundamentals_cache").select("ticker, fetched_at").execute().data) or []
    except Exception as e:
        logger.warning(f"[fundamentals] cache read failed (run supabase-fundamentals-cache.sql?): {e}")
        return {"refreshed": 0, "error": "no_cache_table"}
    seen = {r["ticker"]: r.get("fetched_at") or "" for r in cached}
    # stalest first: never-fetched (not in cache) before oldest fetched_at
    ordered = sorted(QUALITY_UNIVERSE, key=lambda t: seen.get(t.upper(), ""))
    done = 0
    for tk in ordered[:batch]:
        try:
            m = screen_ticker(tk)
            if not m:
                continue
            sb.table("fundamentals_cache").upsert({
                "ticker": m["ticker"],
                "net_margin": m["net_margin"], "roe": m["roe"],
                "debt_to_equity": m["debt_to_equity"], "revenue_growth": m["revenue_growth"],
                "fcf_positive": m["fcf_positive"], "quality_score": m["quality_score"],
                "metrics": {k: m[k] for k in ("revenue_latest","net_income_latest","equity","debt","fcf")},
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }).execute()
            done += 1
        except Exception as e:
            logger.debug(f"[fundamentals] refresh {tk} failed: {e}")
        time.sleep(sleep_s)
    logger.info(f"[fundamentals] refreshed {done}/{batch} (universe={len(QUALITY_UNIVERSE)})")
    return {"refreshed": done, "universe": len(QUALITY_UNIVERSE)}


def get_ranked(sb, min_score: int = 0) -> list[dict]:
    """Cached quality ranking, best first."""
    try:
        rows = (sb.table("fundamentals_cache").select("*")
                .gte("quality_score", min_score)
                .order("quality_score", desc=True)
                .limit(500).execute().data) or []
    except Exception as e:
        logger.warning(f"[fundamentals] get_ranked failed: {e}")
        return []
    rows.sort(key=lambda r: (-(r.get("quality_score") or 0), -((r.get("roe") or -999))))
    return rows
