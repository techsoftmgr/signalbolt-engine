"""Unit tests — BO_POC uses the EXACT backtest predicate as its live entry
condition (fidelity by construction), and degrades gracefully."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from engine import bo_poc
from engine import historical_backtest as hb


def test_bo_poc_entry_is_the_backtest_predicate():
    # BO_POC's live gate must BE historical_backtest._breakout — same function,
    # so live signals match the backtest archetype ~100%.
    import inspect
    src = inspect.getsource(bo_poc.scan)
    assert "hb._breakout(df)" in src and '"LONG"' in src


def test_universe_nonempty_and_liquid():
    assert len(bo_poc.UNIVERSE) >= 20
    assert "NVDA" in bo_poc.UNIVERSE and "SPY" in bo_poc.UNIVERSE


def test_has_active_handles_bad_sb():
    class _Boom:
        def table(self, *_a, **_k):
            raise RuntimeError("no db")
    assert bo_poc._has_active(_Boom(), "AAPL") is False


def test_scan_never_raises_on_bad_sb():
    class _Boom:
        def table(self, *_a, **_k):
            raise RuntimeError("no db")
    assert bo_poc.scan(_Boom(), universe=["AAPL"]) == 0


def test_breakout_predicate_shared_with_backtest():
    # sanity: the predicate BO_POC calls is the registered backtest BREAKOUT one
    assert hb.DETECTORS["BREAKOUT"] is hb._breakout
