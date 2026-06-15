"""Unit tests — insider_service: Form 4 parse (open-market P/S, 10b5-1 + comp flags) + aggregation."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine import insider_service as ins


def _f4(rows: str, footnote: str = "") -> bytes:
    """rows = nonDerivativeTransaction XML; footnote optional (e.g. a 10b5-1 note)."""
    return f"""<?xml version="1.0"?>
<ownershipDocument>
  <issuer><issuerTradingSymbol>HOOD</issuerTradingSymbol></issuer>
  <reportingOwner>
    <reportingOwnerId><rptOwnerName>Malka Meyer</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship><isDirector>1</isDirector></reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>{rows}</nonDerivativeTable>
  {footnote}
</ownershipDocument>""".encode()


def _tx(code, shares, price):
    return f"""<nonDerivativeTransaction>
      <transactionDate><value>2026-05-28</value></transactionDate>
      <transactionCoding><transactionCode>{code}</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>{shares}</value></transactionShares>
        <transactionPricePerShare><value>{price}</value></transactionPricePerShare>
      </transactionAmounts>
    </nonDerivativeTransaction>"""


def test_parse_keeps_open_market_excludes_grant_and_computes_value():
    xml = _f4(_tx("P", 1000, "80.00") + _tx("A", 5000, "0") + _tx("S", 200, "75.00"))
    txns = ins.parse_form4(xml, "HOOD", "2026-05-29")
    assert len(txns) == 2                            # P + S kept, A (grant) dropped
    buy = next(t for t in txns if t["code"] == "P")
    assert buy["value_usd"] == 80000.0 and "Director" in buy["role"]
    sell = next(t for t in txns if t["code"] == "S")
    assert sell["value_usd"] == 15000.0
    # the filing contains a grant (A) → comp_related; no 10b5-1 footnote → not scheduled
    assert sell["comp_related"] is True and sell["scheduled"] is False


def test_parse_flags_10b5_1_plan():
    xml = _f4(_tx("S", 1000, "80.00"),
              footnote="<footnotes><footnote>Sale under a Rule 10b5-1 trading plan.</footnote></footnotes>")
    sell = ins.parse_form4(xml, "HOOD")[0]
    assert sell["scheduled"] is True                 # 10b5-1 → scheduled (noise)


def test_parse_clean_discretionary_sell():
    xml = _f4(_tx("S", 1000, "80.00"))               # lone sell, no grant/exercise, no plan
    sell = ins.parse_form4(xml, "HOOD")[0]
    assert sell["scheduled"] is False and sell["comp_related"] is False


def test_parse_drops_bad():
    assert ins.parse_form4(b"<not-xml", "X") == []


def test_aggregate_splits_discretionary_vs_scheduled():
    txns = [
        {"owner": "A", "code": "P", "shares": 1000, "price": 80, "value_usd": 80000, "txn_date": "2026-05-28"},
        {"owner": "B", "code": "S", "shares": 100, "price": 70, "value_usd": 7000, "txn_date": "2026-05-27"},   # discretionary
        {"owner": "C", "code": "S", "shares": 500, "price": 70, "value_usd": 35000, "txn_date": "2026-05-26", "scheduled": True},   # 10b5-1
        {"owner": "D", "code": "S", "shares": 200, "price": 70, "value_usd": 14000, "txn_date": "2026-05-25", "comp_related": True},  # exercise-and-sell
    ]
    a = ins.aggregate(txns)
    assert a["sell_usd"] == 56000                              # all sells
    assert a["discretionary_sell_usd"] == 7000                # only the chosen one
    assert a["scheduled_sell_usd"] == 49000                   # 10b5-1 + comp
    assert a["distinct_discretionary_sellers"] == 1
    assert a["net_usd"] == 80000 - 56000
    assert a["net_discretionary_usd"] == 80000 - 7000         # buys minus only discretionary sells


def test_aggregate_cluster_buy():
    txns = [
        {"owner": "A", "code": "P", "shares": 100, "price": 10, "value_usd": 1000, "txn_date": "2026-05-28"},
        {"owner": "B", "code": "P", "shares": 100, "price": 10, "value_usd": 1000, "txn_date": "2026-05-27"},
    ]
    a = ins.aggregate(txns)
    assert a["cluster_buy"] is True and a["distinct_buyers"] == 2


def test_summary_batch_compact(monkeypatch):
    from unittest.mock import MagicMock
    rows = [
        {"ticker": "HOOD", "owner": "Meyer", "code": "P", "shares": 1000, "price": 80, "value_usd": 80000, "txn_date": "2026-05-28"},
        {"ticker": "HOOD", "owner": "X", "code": "S", "shares": 100, "price": 70, "value_usd": 7000, "txn_date": "2026-05-27", "scheduled": True},
        {"ticker": "NVDA", "owner": "Stevens", "code": "S", "shares": 1000, "price": 200, "value_usd": 200000, "txn_date": "2026-05-26"},
    ]
    sb = MagicMock()
    sb.table.return_value.select.return_value.in_.return_value.gte.return_value.execute.return_value.data = rows
    out = ins.summary_batch(sb, ["HOOD", "NVDA"])
    assert out["HOOD"]["buy_usd"] == 80000
    assert out["HOOD"]["discretionary_sell_usd"] == 0          # the HOOD sell was 10b5-1
    assert out["HOOD"]["net_discretionary_usd"] == 80000
    assert out["NVDA"]["discretionary_sell_usd"] == 200000     # discretionary sell
    assert out["NVDA"]["net_discretionary_usd"] == -200000


def test_txn_uid_deterministic():
    t = {"accession": "0000950103-26-008745", "owner": "Malka Meyer", "txn_date": "2026-05-28",
         "code": "P", "shares": 1000, "price": 80.0}
    assert ins._txn_uid(t) == ins._txn_uid(dict(t)) and len(ins._txn_uid(t)) == 20


# ── Fast-lane: getcurrent feed parsing ──────────────────────────────────────
_FEED_SAMPLE = b"""<?xml version="1.0" encoding="ISO-8859-1" ?>
<feed xmlns="http://www.w3.org/2005/Atom">
<title>Latest Filings</title>
<entry>
<title>4 - Angelo Michael F (0001404851) (Reporting)</title>
<link rel="alternate" type="text/html" href="https://www.sec.gov/Archives/edgar/data/1404851/000140485126000002/0001404851-26-000002-index.htm"/>
<category scheme="https://www.sec.gov/" label="form type" term="4"/>
</entry>
<entry>
<title>4 - VirnetX Holding Corp (0001082324) (Issuer)</title>
<link rel="alternate" type="text/html" href="https://www.sec.gov/Archives/edgar/data/1082324/000140485126000002/0001404851-26-000002-index.htm"/>
<category scheme="https://www.sec.gov/" label="form type" term="4"/>
</entry>
<entry>
<title>4 - Doe Jane (0009999999) (Reporting)</title>
<link rel="alternate" type="text/html" href="https://www.sec.gov/Archives/edgar/data/9999999/000999999926000001/0009999999-26-000001-index.htm"/>
<category scheme="https://www.sec.gov/" label="form type" term="4"/>
</entry>
<entry>
<title>4 - Microsoft Corp (0000789019) (Issuer)</title>
<link rel="alternate" type="text/html" href="https://www.sec.gov/Archives/edgar/data/789019/000999999926000001/0009999999-26-000001-index.htm"/>
<category scheme="https://www.sec.gov/" label="form type" term="4"/>
</entry>
</feed>"""


def test_feed_extracts_issuer_ciks_only():
    ciks, n = ins._issuer_ciks_from_feed(_FEED_SAMPLE)
    assert n == 4                                  # all entries counted (for pagination)
    # only the (Issuer) entries' CIKs, normalized to int-form (no zero-padding):
    assert ciks == {"1082324", "789019"}
    assert "1404851" not in ciks and "9999999" not in ciks   # the (Reporting) insiders excluded


def test_feed_handles_garbage():
    assert ins._issuer_ciks_from_feed(b"not xml") == (set(), 0)
