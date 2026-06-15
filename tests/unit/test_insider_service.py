"""Unit tests — insider_service: Form 4 parse (open-market P/S only) + aggregation."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine import insider_service as ins

# A Form 4 with: an open-market BUY (P), a grant (A → excluded), and an open-market SELL (S).
FORM4 = """<?xml version="1.0"?>
<ownershipDocument>
  <issuer><issuerTradingSymbol>HOOD</issuerTradingSymbol></issuer>
  <reportingOwner>
    <reportingOwnerId><rptOwnerName>Malka Meyer</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship><isDirector>1</isDirector></reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-05-28</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>1000</value></transactionShares>
        <transactionPricePerShare><value>80.00</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-05-28</value></transactionDate>
      <transactionCoding><transactionCode>A</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>5000</value></transactionShares>
        <transactionPricePerShare><value>0</value></transactionPricePerShare>
      </transactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-05-20</value></transactionDate>
      <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>200</value></transactionShares>
        <transactionPricePerShare><value>75.00</value></transactionPricePerShare>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""


def test_parse_keeps_only_open_market_and_computes_value():
    txns = ins.parse_form4(FORM4.encode(), "HOOD", "2026-05-29")
    assert len(txns) == 2                       # P + S kept; A (grant) dropped
    buy = next(t for t in txns if t["code"] == "P")
    assert buy["side"] == "BUY" and buy["shares"] == 1000 and buy["price"] == 80.0
    assert buy["value_usd"] == 80000.0          # shares * price
    assert buy["owner"] == "Malka Meyer" and "Director" in buy["role"]
    sell = next(t for t in txns if t["code"] == "S")
    assert sell["side"] == "SELL" and sell["value_usd"] == 15000.0


def test_parse_drops_zero_share_and_bad():
    assert ins.parse_form4(b"<not-xml", "X") == []
    # a P with zero shares is dropped
    xml = FORM4.replace("<value>1000</value>", "<value>0</value>")
    assert all(t["code"] != "P" for t in ins.parse_form4(xml.encode(), "HOOD"))


def test_aggregate_dollars_and_cluster():
    txns = [
        {"owner": "A", "code": "P", "shares": 1000, "price": 80, "value_usd": 80000, "txn_date": "2026-05-28"},
        {"owner": "B", "code": "P", "shares": 500, "price": 82, "value_usd": 41000, "txn_date": "2026-05-27"},
        {"owner": "C", "code": "S", "shares": 200, "price": 75, "value_usd": 15000, "txn_date": "2026-05-20"},
    ]
    a = ins.aggregate(txns)
    assert a["buy_usd"] == 121000 and a["sell_usd"] == 15000 and a["net_usd"] == 106000
    assert a["distinct_buyers"] == 2 and a["distinct_sellers"] == 1
    assert a["cluster_buy"] is True             # ≥2 distinct buyers
    assert a["avg_buy_price"] == round(121000 / 1500, 2)
    assert a["transactions"][0]["txn_date"] == "2026-05-28"   # most recent first


def test_aggregate_single_buyer_not_cluster():
    txns = [{"owner": "A", "code": "P", "shares": 100, "price": 10, "value_usd": 1000, "txn_date": "2026-05-28"}]
    a = ins.aggregate(txns)
    assert a["cluster_buy"] is False and a["distinct_buyers"] == 1 and a["sell_usd"] == 0


def test_txn_uid_deterministic():
    t = {"accession": "0000950103-26-008745", "owner": "Malka Meyer", "txn_date": "2026-05-28",
         "code": "P", "shares": 1000, "price": 80.0}
    assert ins._txn_uid(t) == ins._txn_uid(dict(t)) and len(ins._txn_uid(t)) == 20
