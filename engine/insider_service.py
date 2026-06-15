"""
Open-market insider activity (SEC Form 4) — BUYS and SELLS transacted on the OPEN
MARKET only.

Transaction codes (we keep only P/S; everything else is comp/non-conviction noise):
  • P = open-market (or private) PURCHASE   → bullish conviction
  • S = open-market (or private) SALE        → distribution
  • A (grant/award), M (option exercise), F (tax withholding), G (gift), C, etc. → EXCLUDED

For every kept transaction we compute shares, price/share, and $ value (shares ×
price), then aggregate per ticker over a trailing window: $ bought, $ sold, net $,
distinct buyers/sellers, avg buy/sell price, and a CLUSTER flag (≥2 distinct insiders
buying — the strongest tell).

Data: SEC EDGAR (free), reusing fundamentals' UA + CIK map. The Form 4 raw ownership
XML lives at the BARE filename in the filing folder — the submissions API's
`primaryDocument` points at the XSL-rendered HTML (`xslF345X0n/ownership.xml`), so we
strip that prefix to get the machine-readable XML.
"""
from __future__ import annotations

import logging
import os
import threading
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import requests

logger = logging.getLogger("signalbolt.insider")

_CACHE_KEY = "markets:insiders:v1"
_TTL = 6 * 3600
_inflight = threading.Lock()

_WINDOW_DAYS = int(os.environ.get("INSIDER_WINDOW_DAYS", "30"))   # rolling last ~30 days (1 month); older auto-pruned
_CLUSTER_MIN_BUYERS = int(os.environ.get("INSIDER_CLUSTER_MIN_BUYERS", "2"))
_NOTABLE_BUY_USD = float(os.environ.get("INSIDER_NOTABLE_BUY_USD", "250000"))      # single open-market buy alert floor
_NOTABLE_SELL_USD = float(os.environ.get("INSIDER_NOTABLE_SELL_USD", "1000000"))   # discretionary-sell alert floor (sells noisier → higher bar)


def _ua() -> dict:
    try:
        from engine.fundamentals import _UA
        return _UA
    except Exception:
        return {"User-Agent": "SignalBolt research techsoftmgr@gmail.com"}


def _txt(el, path):
    if el is None:
        return None
    e = el.find(path)
    return e.text.strip() if (e is not None and e.text) else None


def _role(rel) -> str:
    """Human role from the reportingOwnerRelationship block."""
    if rel is None:
        return "Insider"
    parts = []
    if _txt(rel, "officerTitle"):
        parts.append(_txt(rel, "officerTitle"))
    elif _txt(rel, "isOfficer") == "1":
        parts.append("Officer")
    if _txt(rel, "isDirector") == "1":
        parts.append("Director")
    if _txt(rel, "isTenPercentOwner") == "1":
        parts.append("10% Owner")
    return ", ".join(parts) or "Insider"


def parse_form4(xml_bytes: bytes, ticker: str, filing_date: str | None = None) -> list[dict]:
    """Pure: raw Form 4 ownership XML → list of OPEN-MARKET (P/S) transactions with
    shares, price/share, and $ value. Non-P/S rows are dropped. Never raises."""
    out: list[dict] = []
    try:
        root = ET.fromstring(xml_bytes)
    except Exception:
        return out
    owner = _txt(root, ".//reportingOwner/reportingOwnerId/rptOwnerName") or "?"
    role = _role(root.find(".//reportingOwner/reportingOwnerRelationship"))
    # Filing-level context for classifying SELLS as scheduled / comp-driven vs discretionary:
    #  • 10b5-1 plan → the sale was pre-scheduled (footnote/structured flag mentions "10b5-1")
    #  • a same-filing exercise (M) / grant (A) / conversion (C) → the sale is liquidating
    #    comp shares (exercise-and-sell), not a discretionary open-market decision.
    raw = (xml_bytes.decode("utf-8", "ignore").lower())
    is_plan = ("10b5-1" in raw) or ("10b5–1" in raw)
    all_codes = [_txt(tx, "transactionCoding/transactionCode")
                 for tx in (root.findall(".//nonDerivativeTransaction") + root.findall(".//derivativeTransaction"))]
    has_comp = any(c in ("M", "A", "C") for c in all_codes if c)
    for tx in root.findall(".//nonDerivativeTransaction"):
        code = _txt(tx, "transactionCoding/transactionCode")
        if code not in ("P", "S"):
            continue
        try:
            shares = float(_txt(tx, "transactionAmounts/transactionShares/value") or 0)
            price = float(_txt(tx, "transactionAmounts/transactionPricePerShare/value") or 0)
        except (TypeError, ValueError):
            continue
        if shares <= 0:
            continue
        out.append({
            "ticker": ticker,
            "owner": owner,
            "role": role,
            "txn_date": _txt(tx, "transactionDate/value"),
            "code": code,                                  # P=buy, S=sell
            "side": "BUY" if code == "P" else "SELL",
            "shares": round(shares, 2),
            "price": round(price, 4),
            "value_usd": round(shares * price, 2),
            "scheduled": is_plan,                          # 10b5-1 pre-scheduled
            "comp_related": has_comp,                      # exercise/grant/conversion in same filing
            "filing_date": filing_date,
        })
    return out


def aggregate(txns: list[dict], window_days: int = _WINDOW_DAYS) -> dict:
    """Pure: a ticker's open-market transactions → summary (buy/sell $, net, avg price,
    distinct insiders, cluster flag)."""
    def _disc_sell(t):   # a sell that the insider CHOSE to make on the open market now
        return t["code"] == "S" and not t.get("scheduled") and not t.get("comp_related")

    buys = [t for t in txns if t["code"] == "P"]
    sells = [t for t in txns if t["code"] == "S"]
    disc_sells = [t for t in sells if _disc_sell(t)]
    sched_sells = [t for t in sells if not _disc_sell(t)]   # 10b5-1 or exercise/grant liquidation
    buy_usd = round(sum(t["value_usd"] for t in buys), 2)
    sell_usd = round(sum(t["value_usd"] for t in sells), 2)
    disc_sell_usd = round(sum(t["value_usd"] for t in disc_sells), 2)
    sched_sell_usd = round(sum(t["value_usd"] for t in sched_sells), 2)
    buy_sh = sum(t["shares"] for t in buys)
    disc_sell_sh = sum(t["shares"] for t in disc_sells)
    n_buyers = len({t["owner"] for t in buys})
    return {
        "buy_usd": buy_usd, "sell_usd": sell_usd,
        "discretionary_sell_usd": disc_sell_usd,           # the SELL signal (chosen, open-market)
        "scheduled_sell_usd": sched_sell_usd,              # 10b5-1 / comp — noise
        "net_usd": round(buy_usd - sell_usd, 2),
        "net_discretionary_usd": round(buy_usd - disc_sell_usd, 2),   # buys minus only the chosen sells
        "buy_count": len(buys), "sell_count": len(sells), "discretionary_sell_count": len(disc_sells),
        "distinct_buyers": n_buyers,
        "distinct_sellers": len({t["owner"] for t in sells}),
        "distinct_discretionary_sellers": len({t["owner"] for t in disc_sells}),
        "avg_buy_price": round(buy_usd / buy_sh, 2) if buy_sh else None,
        "avg_discretionary_sell_price": round(disc_sell_usd / disc_sell_sh, 2) if disc_sell_sh else None,
        "cluster_buy": n_buyers >= _CLUSTER_MIN_BUYERS,
        "cluster_sell": len({t["owner"] for t in disc_sells}) >= _CLUSTER_MIN_BUYERS,
        "window_days": window_days,
        "transactions": sorted(txns, key=lambda t: (t.get("txn_date") or ""), reverse=True)[:25],
    }


# ── EDGAR I/O ───────────────────────────────────────────────────────────────
def _recent_form4s(cik: str, since: datetime) -> list[tuple]:
    """[(accession, primaryDocument, filing_date)] for Form 4s filed on/after `since`."""
    try:
        r = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=_ua(), timeout=20)
        if not r.ok:
            return []
        rec = r.json().get("filings", {}).get("recent", {})
        out = []
        for form, accn, doc, fdate in zip(rec.get("form", []), rec.get("accessionNumber", []),
                                          rec.get("primaryDocument", []), rec.get("filingDate", [])):
            if form != "4":
                continue
            try:
                if datetime.strptime(fdate, "%Y-%m-%d") < since:
                    continue
            except Exception:
                pass
            out.append((accn, doc, fdate))
        return out
    except Exception as e:
        logger.debug(f"[insider] submissions {cik} failed: {e}")
        return []


def _fetch_form4_xml(cik: str, accession: str, primary_doc: str) -> bytes | None:
    """Raw ownership XML (strip the XSL render prefix from primaryDocument)."""
    bare = (primary_doc or "").split("/")[-1]
    if not bare.endswith(".xml"):
        bare = "ownership.xml"   # near-universal fallback name
    accn = accession.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accn}/{bare}"
    try:
        r = requests.get(url, headers=_ua(), timeout=20)
        return r.content if r.ok else None
    except Exception as e:
        logger.debug(f"[insider] xml {accession} failed: {e}")
        return None


def fetch_ticker(ticker: str, cik: str, window_days: int = _WINDOW_DAYS,
                 seen: set | None = None) -> tuple[list[dict], set]:
    """All open-market transactions for one ticker in the window. `seen` = accessions
    already parsed (so re-runs skip them); returns (new_transactions, accessions_seen)."""
    since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=window_days)
    txns: list[dict] = []
    accns = set()
    for accn, doc, fdate in _recent_form4s(cik, since):
        accns.add(accn)
        if seen and accn in seen:
            continue
        xml = _fetch_form4_xml(cik, accn, doc)
        if xml:
            for t in parse_form4(xml, ticker, fdate):
                t["accession"] = accn
                txns.append(t)
    return txns, accns


# ── Universe (same liquid set as movers/churn) → CIK ────────────────────────
def _universe_ciks() -> list[tuple]:
    """[(ticker, cik)] for the tracked liquid universe, resolved via SEC's CIK map."""
    try:
        from engine import prescreener as ps, momentum_detector as md
        from engine.fundamentals import cik_map
        cm = cik_map()
        syms = sorted(set(ps.EXTENDED_UNIVERSE) | set(md.UNIVERSE))
        return [(s, cm[s]) for s in syms if s in cm]
    except Exception as e:
        logger.error(f"[insider] universe load failed: {e}")
        return []


# ── Table-backed refresh + screen ───────────────────────────────────────────
def _txn_uid(t: dict) -> str:
    import hashlib
    key = f"{t.get('accession')}|{t['owner']}|{t.get('txn_date')}|{t['code']}|{t['shares']}|{t['price']}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]


def refresh_universe(sb, window_days: int = _WINDOW_DAYS) -> dict:
    """Fetch new Form 4s for the universe, upsert open-market transactions, and return
    the freshly-seen NOTABLE buys (cluster or ≥ $threshold) for alerting. Incremental:
    skips filings (accessions) already stored."""
    uni = _universe_ciks()
    stats = {"scanned": 0, "new_transactions": 0, "notable_buys": [], "notable_sells": []}
    if not uni:
        return stats
    # accessions already stored (skip re-parsing)
    seen_by_ticker: dict = {}
    try:
        since_iso = (datetime.now(timezone.utc) - timedelta(days=window_days)).date().isoformat()
        rows = (sb.table("insider_transactions").select("ticker,accession")
                .gte("txn_date", since_iso).execute().data) or []
        for r in rows:
            seen_by_ticker.setdefault(r["ticker"], set()).add(r["accession"])
    except Exception as e:
        logger.debug(f"[insider] seen-accession load failed: {e}")

    for ticker, cik in uni:
        stats["scanned"] += 1
        try:
            new_txns, _ = fetch_ticker(ticker, cik, window_days, seen=seen_by_ticker.get(ticker))
        except Exception as e:
            logger.debug(f"[insider] fetch {ticker} failed: {e}")
            continue
        for t in new_txns:
            row = {
                "txn_uid": _txn_uid(t), "ticker": t["ticker"], "owner": t["owner"], "role": t["role"],
                "txn_date": t.get("txn_date"), "code": t["code"], "side": t["side"],
                "shares": t["shares"], "price": t["price"], "value_usd": t["value_usd"],
                "scheduled": bool(t.get("scheduled")), "comp_related": bool(t.get("comp_related")),
                "accession": t.get("accession"), "filing_date": t.get("filing_date"),
            }
            try:
                sb.table("insider_transactions").upsert(row, on_conflict="txn_uid").execute()
                stats["new_transactions"] += 1
                if t["code"] == "P" and t["value_usd"] >= _NOTABLE_BUY_USD:
                    stats["notable_buys"].append(t)
                elif (t["code"] == "S" and not t.get("scheduled") and not t.get("comp_related")
                      and t["value_usd"] >= _NOTABLE_SELL_USD):
                    stats["notable_sells"].append(t)
            except Exception as e:
                logger.debug(f"[insider] upsert {ticker} failed: {e}")
    # Keep only the last `window_days` — prune everything older (no historical kept).
    try:
        deleted = sb.table("insider_transactions").delete().lt("txn_date", _since_iso(window_days)).execute()
        stats["pruned"] = len(deleted.data or [])
    except Exception as e:
        logger.debug(f"[insider] prune failed: {e}")
    logger.info(f"[insider] refresh: scanned={stats['scanned']} new_txns={stats['new_transactions']} "
                f"notable_buys={len(stats['notable_buys'])} pruned={stats.get('pruned', 0)}")
    return stats


def build_screen(sb, window_days: int = _WINDOW_DAYS, limit: int = 60) -> dict:
    """Aggregate stored transactions (trailing window) into the per-ticker screen,
    ranked by net open-market $ (biggest buying first). Cached."""
    from engine import cache
    empty = {"asOf": datetime.now(timezone.utc).isoformat(), "items": []}
    try:
        since_iso = (datetime.now(timezone.utc) - timedelta(days=window_days)).date().isoformat()
        rows = (sb.table("insider_transactions").select("*").gte("txn_date", since_iso)
                .order("txn_date", desc=True).limit(5000).execute().data) or []
    except Exception as e:
        logger.error(f"[insider] screen query failed: {e}")
        return empty
    by_ticker: dict = {}
    for r in rows:
        by_ticker.setdefault(r["ticker"], []).append(r)
    items = []
    for tk, txns in by_ticker.items():
        agg = aggregate(txns, window_days)
        # Skip names whose only activity is scheduled 10b5-1 / comp selling (no buys,
        # no discretionary sells) — pure noise, would just show as $0 net.
        if agg["buy_usd"] <= 0 and agg["discretionary_sell_usd"] <= 0:
            continue
        items.append({"ticker": tk, **agg})
    items.sort(key=lambda x: -x["net_discretionary_usd"])
    out = {"asOf": datetime.now(timezone.utc).isoformat(),
           "windowDays": window_days, "items": items[:limit]}
    try:
        cache.kv.set_json(_CACHE_KEY, out, _TTL)
    except Exception:
        pass
    return out


def peek() -> dict | None:
    try:
        from engine import cache
        return cache.kv.get_json(_CACHE_KEY)
    except Exception:
        return None


def _since_iso(window_days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=window_days)).date().isoformat()


def summarize_ticker(sb, ticker: str, window_days: int = _WINDOW_DAYS) -> dict:
    """Per-ticker open-market insider summary — table-first, with a live EDGAR fetch
    fallback for tickers OUTSIDE the scanned universe (powers search + the ticker hub).
    Cached per ticker."""
    tk = (ticker or "").upper().strip()
    if not tk:
        return {"ticker": tk, **aggregate([], window_days)}
    from engine import cache
    ck = f"insiders:ticker:{tk}"
    try:
        c = cache.kv.get_json(ck)
        if c:
            return c
    except Exception:
        pass
    rows: list = []
    try:
        rows = (sb.table("insider_transactions").select("*").eq("ticker", tk)
                .gte("txn_date", _since_iso(window_days)).execute().data) or []
    except Exception as e:
        logger.debug(f"[insider] summarize table {tk} failed: {e}")
    if not rows:                                  # not in the universe table → fetch on demand
        try:
            from engine.fundamentals import cik_map
            cik = cik_map().get(tk)
            if cik:
                rows, _ = fetch_ticker(tk, cik, window_days)
        except Exception as e:
            logger.debug(f"[insider] summarize on-demand {tk} failed: {e}")
    out = {"ticker": tk, **aggregate(rows, window_days)}
    try:
        cache.kv.set_json(ck, out, _TTL)
    except Exception:
        pass
    return out


def summary_batch(sb, tickers: list[str], window_days: int = _WINDOW_DAYS) -> dict:
    """Compact per-ticker summaries for the watchlist (TABLE-ONLY → fast; one query).
    Tickers with no stored open-market activity are simply absent."""
    tks = [(t or "").upper().strip() for t in (tickers or []) if t]
    if not tks:
        return {}
    try:
        rows = (sb.table("insider_transactions").select("*").in_("ticker", tks[:200])
                .gte("txn_date", _since_iso(window_days)).execute().data) or []
    except Exception as e:
        logger.debug(f"[insider] summary_batch failed: {e}")
        return {}
    by: dict = {}
    for r in rows:
        by.setdefault(r["ticker"], []).append(r)
    out = {}
    for tk, txns in by.items():
        a = aggregate(txns, window_days)
        out[tk] = {
            "net_discretionary_usd": a["net_discretionary_usd"],
            "buy_usd": a["buy_usd"], "discretionary_sell_usd": a["discretionary_sell_usd"],
            "distinct_buyers": a["distinct_buyers"],
            "cluster_buy": a["cluster_buy"], "cluster_sell": a["cluster_sell"],
        }
    return out
