"""ops_diagnostics ring buffer — captures WARNING+ logs so silent failures are
diagnosable via /ops/diagnostics without Fly-log access."""
import logging
from engine import ops_diagnostics as od


def test_ring_captures_warnings_and_filters():
    od._RING.clear()
    od._installed = False
    od.install()
    od.install()  # idempotent
    lg = logging.getLogger("signalbolt.test.ops")
    lg.info("info line — should NOT be captured (below WARNING)")
    lg.warning("[alpaca] get_multi_bars batch 3 failed: rate limited")
    lg.error("boom something broke")

    allw = od.recent(limit=50)
    msgs = " ".join(r["msg"] for r in allw)
    assert "get_multi_bars" in msgs and "boom" in msgs
    assert "should NOT be captured" not in msgs          # INFO filtered out

    # substring + level filters
    assert len(od.recent(contains="alpaca")) == 1
    assert all(r["level"] == "ERROR" for r in od.recent(level="error"))
    assert od.recent(contains="nonexistent-xyz") == []
