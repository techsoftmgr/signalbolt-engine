"""Unit tests — de-SPAC merger tracker. Offline (mocked EDGAR)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine import spac_mergers as sm


def test_parse_name():
    spac = sm._parse_name("Black Hawk Acquisition Corp  (BKHA, BKHAR, BKHAU)  (CIK 0002000775)")
    assert spac["name"] == "Black Hawk Acquisition Corp" and spac["ticker"] == "BKHA" and spac["cik"] == "0002000775"
    tgt = sm._parse_name("Vesicor Therapeutics, Inc.  (CIK 0001993074)")
    assert tgt["name"] == "Vesicor Therapeutics, Inc." and tgt["ticker"] is None and tgt["cik"] == "0001993074"


def test_filing_url():
    u = sm._filing_url("0001829126-26-005870", "0002081300")
    assert u == "https://www.sec.gov/Archives/edgar/data/2081300/000182912626005870/0001829126-26-005870-index.htm"
    assert sm._filing_url(None, None) is None


def test_get_spac_mergers_pairs_and_dedup(monkeypatch):
    hits = [
        {"_source": {"display_names": [
            "Black Hawk Acquisition Corp  (BKHA)  (CIK 0002000775)",
            "Vesicor Therapeutics, Inc.  (CIK 0001993074)"],
            "file_type": "S-4", "file_date": "2026-05-01", "adsh": "0001-26-1", "ciks": ["0002000775"]}},
        # newer amendment for the SAME SPAC -> should replace the older one
        {"_source": {"display_names": [
            "Black Hawk Acquisition Corp  (BKHA)  (CIK 0002000775)",
            "Vesicor Therapeutics, Inc.  (CIK 0001993074)"],
            "file_type": "S-4/A", "file_date": "2026-06-11", "adsh": "0001-26-2", "ciks": ["0002000775"]}},
        # an ordinary (non-SPAC) S-4 -> skipped
        {"_source": {"display_names": [
            "Blockfusion Data Centers, Inc.  (BLDC)  (CIK 0002097508)",
            "Blockfusion USA, Inc.  (CIK 0001910992)"],
            "file_type": "S-4", "file_date": "2026-05-01", "adsh": "0009-26-1", "ciks": ["0002097508"]}},
    ]
    monkeypatch.setattr(sm, "_fetch_hits", lambda: hits)
    out = sm.get_spac_mergers(force=True)
    assert out["available"] is True
    assert len(out["deals"]) == 1                     # deduped to one SPAC; non-SPAC dropped
    d = out["deals"][0]
    assert d["spac"] == "Black Hawk Acquisition Corp" and d["spac_ticker"] == "BKHA"
    assert d["target"] == "Vesicor Therapeutics, Inc."
    assert d["form"] == "S-4/A" and d["stage"] == "Registration amended"   # latest stage kept
    assert d["date"] == "2026-06-11"
    assert d["filing_url"].endswith("0001-26-2-index.htm")
