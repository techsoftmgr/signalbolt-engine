"""
SPAC merger (de-SPAC) tracker — which SPAC is merging with which private company.

Source: SEC EDGAR full-text search (https://efts.sec.gov) for de-SPAC business-
combination filings (S-4 / S-4/A). Free, no API key — SEC only requires a
descriptive User-Agent. Each hit carries BOTH parties in `display_names`
(the SPAC registrant + the target), so we extract the pairing, the filing
form/date (= deal stage), and a link to the filing.

Read-only, cached, never raises. NOTE: the exact business-combination CLOSE date
lives in the filing prose, not a structured field — so we surface the announced
pairing + latest filing stage/date + filing link, not a guaranteed listing date.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone, timedelta

logger = logging.getLogger("signalbolt.spac_mergers")

_CACHE_KEY = "markets:spac_mergers:v1"
_CACHE_TTL = 6 * 3600
_EFTS = "https://efts.sec.gov/LATEST/search-index"
_UA = "SignalBolt research techsoftmgr@gmail.com"   # SEC requires a descriptive UA

_SPAC_PAT = re.compile(r"acquisition corp|acquisition company|acquisition holdings|"
                       r"capital corp|blank check|acquisition ltd", re.I)
_STAGE = {
    "425":      "Announced",
    "S-4":      "Registration filed",
    "S-4/A":    "Registration amended",
    "DEFM14A":  "Shareholder vote set",
    "DEF 14A":  "Shareholder vote set",
    "8-K":      "Closing",
}


def _parse_name(s: str) -> dict:
    """'Black Hawk Acquisition Corp  (BKHA, BKHAR)  (CIK 0002000775)' ->
    {name, ticker, cik}. Target rows often have only the (CIK ...) paren."""
    cik = None
    m = re.search(r"\(CIK\s*(\d+)\)", s)
    if m:
        cik = m.group(1)
    ticker = None
    mt = re.search(r"\(([A-Z0-9.,\s]+)\)\s*\(CIK", s)   # ticker paren BEFORE the CIK paren
    if mt:
        ticker = mt.group(1).split(",")[0].strip() or None
    name = re.split(r"\s+\(", s)[0].strip()
    return {"name": name, "ticker": ticker, "cik": cik}


def _filing_url(adsh: str | None, cik: str | None) -> str | None:
    if not adsh or not cik:
        return None
    acc_nodash = adsh.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/{adsh}-index.htm"


def _fetch_hits() -> list[dict]:
    """One EDGAR full-text page of recent de-SPAC S-4 filings. Retries once on a
    transient error (EDGAR intermittently 500s). Fails open to []."""
    start = (datetime.now(timezone.utc) - timedelta(days=120)).strftime("%Y-%m-%d")
    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    params = {"q": '"business combination"', "forms": "S-4", "startdt": start, "enddt": end}
    for attempt in (1, 2):
        try:
            import httpx
            with httpx.Client(timeout=25, headers={"User-Agent": _UA}) as c:
                r = c.get(_EFTS, params=params)
                r.raise_for_status()
                return (r.json().get("hits", {}) or {}).get("hits", []) or []
        except Exception as e:
            logger.debug(f"[spac_mergers] EDGAR fetch attempt {attempt} failed: {e}")
    return []


def get_spac_mergers(force: bool = False) -> dict:
    """De-SPAC pairings (SPAC -> target) with latest filing stage. Cached 6h."""
    from engine import cache
    if not force:
        try:
            cached = cache.kv.get_json(_CACHE_KEY)
            if cached:
                return cached
        except Exception:
            pass

    hits = _fetch_hits()
    by_spac: dict[str, dict] = {}
    for h in hits:
        src = h.get("_source", {}) or {}
        names = [_parse_name(n) for n in (src.get("display_names") or [])]
        spac = next((n for n in names if _SPAC_PAT.search(n["name"] or "")), None)
        if not spac:
            continue   # no SPAC party → skip (ordinary S-4 merger, not a de-SPAC)
        target = next((n for n in names if n is not spac and not _SPAC_PAT.search(n["name"] or "")), None)
        date = src.get("file_date")
        ftype = src.get("file_type") or "S-4"
        key = spac.get("cik") or spac.get("name")
        prev = by_spac.get(key)
        # keep the most RECENT filing per SPAC (= latest deal stage)
        if prev and (prev.get("date") or "") >= (date or ""):
            continue
        by_spac[key] = {
            "spac":        spac.get("name"),
            "spac_ticker": spac.get("ticker"),
            "target":      (target or {}).get("name"),
            "form":        ftype,
            "stage":       _STAGE.get(ftype, _STAGE.get(ftype.split("/")[0], "In registration")),
            "date":        date,
            "filing_url":  _filing_url(src.get("adsh"), (src.get("ciks") or [spac.get("cik")])[0]),
        }

    rows = sorted(by_spac.values(), key=lambda x: x.get("date") or "", reverse=True)
    out = {
        "available": True,
        "source": "sec_edgar",
        "deals": rows[:60],
        "updated": datetime.now(timezone.utc).isoformat(),
        "note": ("De-SPAC business combinations from SEC S-4 filings. Stage = latest "
                 "filing. Exact close/listing date is in the filing text — tap to read."),
    }
    # Only cache a NON-empty result — an empty list usually means a transient
    # EDGAR error, and we don't want to pin that for 6h.
    if rows:
        try:
            cache.kv.set_json(_CACHE_KEY, out, _CACHE_TTL)
        except Exception:
            pass
    return out
