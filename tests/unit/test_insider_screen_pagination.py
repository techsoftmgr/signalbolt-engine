"""Unit test — insider build_screen must PAGINATE (no silent truncation).

Regression for: NCLH's $25M Jun-1/2 buys vanished from the screen while MELI's newer $200K buy
showed, because a flat .limit(5000) ordered by txn_date desc dropped the OLDEST in-window rows
once the universe exceeded the limit. build_screen now pages through every in-window row."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine import insider_service as ins
from engine import cache as _cache


class _FakeKV:
    def get_json(self, k): return None
    def set_json(self, k, v, ttl=None): pass


class _FakeQuery:
    def __init__(self, pages): self.pages = pages; self._off = 0
    def select(self, *a, **k): return self
    def gte(self, *a, **k): return self
    def order(self, *a, **k): return self
    def range(self, lo, hi): self._off = lo; return self
    def execute(self):
        idx = self._off // 1000
        data = self.pages[idx] if 0 <= idx < len(self.pages) else []
        return type("R", (), {"data": data})()


class _FakeSB:
    def __init__(self, pages): self.pages = pages
    def table(self, name): return _FakeQuery(self.pages)


def _buy(ticker, val, txn_date, owner="X"):
    return {"ticker": ticker, "owner": owner, "role": "Director", "code": "P", "side": "BUY",
            "shares": 1000, "price": val / 1000.0, "value_usd": float(val),
            "scheduled": False, "comp_related": False, "txn_date": txn_date, "accession": f"a-{ticker}-{txn_date}"}


def test_build_screen_paginates_no_truncation(monkeypatch):
    monkeypatch.setattr(_cache, "kv", _FakeKV())
    # Page 0 = a full 1000 rows of newer, small filler activity (would be all a .limit(1000) kept).
    page0 = [_buy("FILL", 5_000, "2026-06-18", owner=f"o{i}") for i in range(1000)]
    # Page 1 = the OLDER, large NCLH buys — only reachable if we page past the first 1000.
    page1 = [_buy("NCLH", 12_400_000, "2026-06-02"), _buy("NCLH", 12_600_000, "2026-06-01")]
    sb = _FakeSB([page0, page1])

    out = ins.build_screen(sb)
    tickers = [it["ticker"] for it in out["items"]]
    assert "NCLH" in tickers                       # not truncated despite being on page 2
    # and it ranks at the TOP by net discretionary $ ( $25M >> filler )
    assert out["items"][0]["ticker"] == "NCLH"


def test_build_screen_stops_at_short_page(monkeypatch):
    monkeypatch.setattr(_cache, "kv", _FakeKV())
    sb = _FakeSB([[_buy("AAA", 1_000_000, "2026-06-10")]])   # single short page → one fetch, then stop
    out = ins.build_screen(sb)
    assert [it["ticker"] for it in out["items"]] == ["AAA"]
