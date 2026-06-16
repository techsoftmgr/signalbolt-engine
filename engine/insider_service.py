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
import re
import threading
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import requests

logger = logging.getLogger("signalbolt.insider")

_CACHE_KEY = "markets:insiders:v1"
# Short TTL: data now refreshes via the 5-min feed fast-lane, so the cached Insider screen
# must not lag the (uncached, live) watchlist. build_screen also warms per-ticker caches.
_TTL = int(os.environ.get("INSIDER_CACHE_TTL", "600"))
_RECENT_DAYS = int(os.environ.get("INSIDER_RECENT_DAYS", "10"))   # watchlist chip: only surface filings this fresh
_inflight = threading.Lock()

_WINDOW_DAYS = int(os.environ.get("INSIDER_WINDOW_DAYS", "30"))   # rolling last ~30 days (1 month); older auto-pruned
_CLUSTER_MIN_BUYERS = int(os.environ.get("INSIDER_CLUSTER_MIN_BUYERS", "2"))
_NOTABLE_BUY_USD = float(os.environ.get("INSIDER_NOTABLE_BUY_USD", "250000"))      # single open-market buy alert floor
_NOTABLE_SELL_USD = float(os.environ.get("INSIDER_NOTABLE_SELL_USD", "1000000"))   # discretionary-sell alert floor (sells noisier → higher bar)
_FEED_COUNT = int(os.environ.get("INSIDER_FEED_COUNT", "100"))                      # getcurrent atom page size
_FEED_MAX_PAGES = int(os.environ.get("INSIDER_FEED_MAX_PAGES", "3"))               # max pages per feed poll (3×100 = 150 filings of headroom)


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


def _persist_txn(sb, t: dict, stats: dict) -> None:
    """Upsert one parsed open-market transaction (idempotent on txn_uid) and record it as
    a NOTABLE buy/sell if it clears the alert bar. Shared by the per-CIK sweep and the
    fast-lane feed poller so both store + flag for alerting identically."""
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
        stats.setdefault("updated_tickers", set()).add(t["ticker"])
        if t["code"] == "P" and t["value_usd"] >= _NOTABLE_BUY_USD:
            stats["notable_buys"].append(t)
        elif (t["code"] == "S" and not t.get("scheduled") and not t.get("comp_related")
              and t["value_usd"] >= _NOTABLE_SELL_USD):
            stats["notable_sells"].append(t)
    except Exception as e:
        logger.debug(f"[insider] upsert {t.get('ticker')} failed: {e}")


def _bust_ticker_caches(tickers) -> None:
    """Invalidate the per-ticker `summarize_ticker` cache for names whose stored activity
    just changed, so the ticker hub / search recompute live instead of serving a stale
    snapshot. The shared list cache (`build_screen`) is rebuilt separately by the caller.
    Without this, the cached Insider screen could lag the (uncached) watchlist by up to the
    cache TTL when new Form 4s land."""
    if not tickers:
        return
    try:
        from engine import cache
        for tk in tickers:
            cache.kv.delete(f"insiders:ticker:{tk}")
    except Exception as e:
        logger.debug(f"[insider] cache bust failed: {e}")


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
            _persist_txn(sb, t, stats)
    _bust_ticker_caches(stats.get("updated_tickers"))   # keep per-ticker hub/search fresh
    # Keep only the last `window_days` — prune everything older (no historical kept).
    try:
        deleted = sb.table("insider_transactions").delete().lt("txn_date", _since_iso(window_days)).execute()
        stats["pruned"] = len(deleted.data or [])
    except Exception as e:
        logger.debug(f"[insider] prune failed: {e}")
    logger.info(f"[insider] refresh: scanned={stats['scanned']} new_txns={stats['new_transactions']} "
                f"notable_buys={len(stats['notable_buys'])} pruned={stats.get('pruned', 0)}")
    return stats


# ── Fast lane: EDGAR "latest filings" feed → near-real-time detection ────────
_FEED_NS = {"a": "http://www.w3.org/2005/Atom"}


def _issuer_ciks_from_feed(xml_bytes: bytes) -> tuple:
    """Pure: one getcurrent Atom page → (set of ISSUER CIKs int-form, entry_count). EDGAR
    emits TWO entries per Form 4 — one '(Reporting)' (the insider) and one '(Issuer)'. We
    keep only the '(Issuer)' entries; the issuer CIK lives in that entry's link path
    (/Archives/edgar/data/<issuerCik>/...), which is what we key the universe on."""
    out: set = set()
    try:
        root = ET.fromstring(xml_bytes)
    except Exception:
        return out, 0
    entries = root.findall(".//a:entry", _FEED_NS)
    for e in entries:
        title = e.findtext("a:title", default="", namespaces=_FEED_NS) or ""
        if "(Issuer)" not in title:              # skip the paired (Reporting) entry
            continue
        link = e.find("a:link", _FEED_NS)
        href = link.get("href", "") if link is not None else ""
        m = re.search(r"/data/(\d+)/", href)
        if m:
            out.add(str(int(m.group(1))))        # normalize (drop zero-padding)
    return out, len(entries)


def _current_form4_issuer_ciks() -> set:
    """Poll EDGAR's 'getcurrent' latest-filings Atom feed for Form 4s and return the set of
    ISSUER CIKs (int-form strings) that just filed. ONE feed request per page covers the
    whole market — we never iterate per-company here. Capped at `_FEED_MAX_PAGES`."""
    ciks: set = set()
    for page in range(_FEED_MAX_PAGES):
        url = ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=4"
               f"&company=&dateb=&owner=include&count={_FEED_COUNT}&start={page * _FEED_COUNT}"
               "&output=atom")
        try:
            r = requests.get(url, headers=_ua(), timeout=20)
            if not r.ok:
                break
        except Exception as e:
            logger.debug(f"[insider] getcurrent page {page} failed: {e}")
            break
        page_ciks, n = _issuer_ciks_from_feed(r.content)
        if n == 0:
            break                                # past the end of the feed
        ciks |= page_ciks
    return ciks


def refresh_from_feed(sb, window_days: int = _WINDOW_DAYS) -> dict:
    """FAST LANE (near-real-time). Poll EDGAR's latest-filings feed; for any universe issuer
    that just filed a Form 4, fetch + parse ONLY that issuer (reusing fetch_ticker). Cheap
    (one feed request + only matched-issuer fetches) so it can run every few minutes. The
    per-CIK `refresh_universe` remains as a less-frequent completeness backstop."""
    stats = {"feed_ciks": 0, "matched": 0, "new_transactions": 0, "notable_buys": [], "notable_sells": []}
    feed_ciks = _current_form4_issuer_ciks()
    stats["feed_ciks"] = len(feed_ciks)
    if not feed_ciks:
        return stats
    matched = [(t, c) for (t, c) in _universe_ciks() if str(int(c)) in feed_ciks]
    stats["matched"] = len(matched)
    if not matched:
        return stats
    # accessions already stored for the matched tickers → skip re-parse (and re-alert)
    seen_by_ticker: dict = {}
    try:
        since_iso = (datetime.now(timezone.utc) - timedelta(days=window_days)).date().isoformat()
        rows = (sb.table("insider_transactions").select("ticker,accession")
                .in_("ticker", [t for (t, _) in matched]).gte("txn_date", since_iso)
                .execute().data) or []
        for r in rows:
            seen_by_ticker.setdefault(r["ticker"], set()).add(r["accession"])
    except Exception as e:
        logger.debug(f"[insider] feed seen-load failed: {e}")
    for ticker, cik in matched:
        try:
            new_txns, _ = fetch_ticker(ticker, cik, window_days, seen=seen_by_ticker.get(ticker))
        except Exception as e:
            logger.debug(f"[insider] feed fetch {ticker} failed: {e}")
            continue
        for t in new_txns:
            _persist_txn(sb, t, stats)
    if stats["new_transactions"]:
        _bust_ticker_caches(stats.get("updated_tickers"))   # per-ticker hub/search fresh
        build_screen(sb, window_days)            # rebuild the shared list cache so the UI reflects it now
    logger.info(f"[insider] feed refresh: feed_ciks={stats['feed_ciks']} matched={stats['matched']} "
                f"new_txns={stats['new_transactions']} notable_buys={len(stats['notable_buys'])}")
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
        # Warm the per-ticker cache with the SAME shape summarize_ticker returns, so the
        # ticker hub / search and the (uncached, live) watchlist stay consistent with this
        # screen and never serve a stale 6h snapshot when fresh filings land.
        try:
            cache.kv.set_json(f"insiders:ticker:{tk}", {"ticker": tk, **agg}, _TTL)
        except Exception:
            pass
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


def _recent_discretionary(txns: list[dict], days: int = _RECENT_DAYS) -> dict:
    """Pure: open-market DISCRETIONARY (non-10b5-1, non-comp) buys/sells whose EDGAR filing
    is within the last `days`. Powers the watchlist chip — recent, high-conviction activity
    only, with the public (filing) date, dropping anything older. P/S; scheduled excluded."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    def _pub(t):                                   # EDGAR-public date preferred, txn date fallback
        return t.get("filing_date") or t.get("txn_date") or ""
    recent = [t for t in txns if _pub(t) >= cutoff and not t.get("scheduled") and not t.get("comp_related")]
    buys = [t for t in recent if t["code"] == "P"]
    sells = [t for t in recent if t["code"] == "S"]
    dates = [_pub(t) for t in (buys + sells) if _pub(t)]
    return {
        "buy_usd": round(sum(t["value_usd"] for t in buys), 2),
        "sell_usd": round(sum(t["value_usd"] for t in sells), 2),
        "buy_count": len(buys), "sell_count": len(sells),
        "latest_date": max(dates) if dates else None,   # ISO yyyy-mm-dd of the freshest filing
        "window_days": days,
    }


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
            "recent": _recent_discretionary(txns),   # last-10d discretionary buys/sells + date (watchlist chip)
        }
    return out
