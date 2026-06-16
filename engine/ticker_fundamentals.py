"""
Per-ticker fundamentals for the watchlist chips — market cap, trailing EPS (for
P/E), and the next earnings quarter. 24h-cached (slow-changing); best-effort,
never raises. Sources (no extra keys beyond the existing Polygon one):
  • market cap + next-earnings quarter — Nasdaq public quote API (no key)
  • trailing annual diluted EPS — Polygon financials (existing key)
P/E is computed downstream as price / EPS (price lives in the snapshot).
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger("signalbolt.ticker_fundamentals")

_TTL = 24 * 3600
_UA = {"User-Agent": "Mozilla/5.0 (SignalBolt)", "Accept": "application/json"}


def _nasdaq(sym: str) -> dict:
    """{market_cap, earnings_period} from Nasdaq. Best-effort."""
    out: dict = {}
    try:
        import httpx
        with httpx.Client(timeout=12, headers=_UA) as c:
            r = c.get(f"https://api.nasdaq.com/api/quote/{sym}/summary", params={"assetclass": "stocks"})
            if r.status_code == 200:
                sd = ((r.json() or {}).get("data") or {}).get("summaryData") or {}
                mc = (sd.get("MarketCap") or {}).get("value") if isinstance(sd.get("MarketCap"), dict) else sd.get("MarketCap")
                if mc:
                    try:
                        out["market_cap"] = int(str(mc).replace(",", "").replace("$", "").strip() or 0) or None
                    except ValueError:
                        pass
            re_ = c.get(f"https://api.nasdaq.com/api/quote/{sym}/eps")
            if re_.status_code == 200:
                arr = ((re_.json() or {}).get("data") or {}).get("earningsPerShare") or []
                nxt = next((e for e in arr if "Upcoming" in str(e.get("type", ""))), None)
                if nxt and nxt.get("period"):
                    out["earnings_period"] = nxt["period"]   # e.g. "Jun 2026" (reporting quarter)
    except Exception as e:
        logger.debug(f"[fundamentals] nasdaq {sym} failed: {e}")
    return out


def _polygon_eps(sym: str):
    """Trailing annual diluted EPS from Polygon financials. None on miss."""
    try:
        import httpx
        key = os.environ.get("POLYGON_API_KEY")
        if not key:
            return None
        with httpx.Client(timeout=12) as c:
            r = c.get("https://api.polygon.io/vX/reference/financials",
                      params={"ticker": sym, "timeframe": "annual", "limit": 1, "apiKey": key})
            if r.status_code != 200:
                return None
            res = (r.json() or {}).get("results") or []
            if not res:
                return None
            inc = (res[0].get("financials") or {}).get("income_statement") or {}
            eps = (inc.get("diluted_earnings_per_share") or inc.get("basic_earnings_per_share") or {})
            v = eps.get("value") if isinstance(eps, dict) else eps
            return round(float(v), 2) if v not in (None, "") else None
    except Exception as e:
        logger.debug(f"[fundamentals] polygon eps {sym} failed: {e}")
        return None


def get(sym: str) -> dict:
    """{market_cap, eps, earnings_period} for a ticker, 24h-cached. {} on total miss."""
    sym = (sym or "").upper().strip()
    if not sym:
        return {}
    ck = f"fundamentals:v1:{sym}"
    try:
        from engine import cache
        c = cache.kv.get_json(ck)
        if c is not None:
            return c
    except Exception:
        pass
    data = _nasdaq(sym)
    # P/E + earnings only make sense for common stocks. The Nasdaq *stocks* summary
    # returns a market cap only for equities — if it's blank the ticker is an ETF/
    # fund/trust, so skip EPS (Polygon returns a meaningless NAV-ish figure for those)
    # and the earnings quarter. Prevents a bogus P/E on GLD/SPY/QQQ etc.
    if data.get("market_cap"):
        eps = _polygon_eps(sym)
        if eps is not None:
            data["eps"] = eps
        # Actual NEXT earnings date (Finnhub, itself 12h-cached) so the watchlist can show
        # "10 Mar 2026" instead of just the reporting quarter. Best-effort; rides along in
        # this 24h-cached payload so the snapshot path makes no extra per-ticker calls.
        try:
            from engine import earnings_service
            ne = earnings_service.get_next_earnings(sym)
            if ne and ne.get("date"):
                data["earnings_date"] = ne["date"]   # ISO yyyy-mm-dd
        except Exception:
            pass
    else:
        data.pop("earnings_period", None)
    try:
        from engine import cache
        cache.kv.set_json(ck, data, _TTL)
    except Exception:
        pass
    return data
