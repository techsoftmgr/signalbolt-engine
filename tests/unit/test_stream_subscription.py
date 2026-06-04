"""
Unit tests — runner._ensure_stream_subscription scheduling from the SYNC scan
thread. Regression for the HOOD freeze (2026-06-04): a fired ticker was only
queued to _pending_tickers (drains on reconnect) instead of actually subscribed,
so while the stream stayed connected the ticker got no ticks and its real-time
stop/target checks went dark. Now it schedules onto the worker's stream loop.
"""
import sys
import os
from unittest.mock import patch, MagicMock
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine import runner, stream


def test_fired_ticker_scheduled_onto_stream_loop():
    fake_loop = MagicMock(); fake_loop.is_running.return_value = True
    cur = MagicMock(); cur.is_running.return_value = False   # this thread: no running loop
    with patch.object(stream, "_stream_loop", fake_loop), \
         patch("asyncio.get_event_loop", return_value=cur), \
         patch("asyncio.run_coroutine_threadsafe") as rcts, \
         patch.object(stream, "subscribe_extra_tickers", MagicMock()):
        runner._ensure_stream_subscription("HOOD")
    # Scheduled onto the worker loop NOW — not left sitting in _pending.
    rcts.assert_called_once()
    assert rcts.call_args[0][1] is fake_loop


def test_falls_back_to_pending_when_stream_not_up():
    cur = MagicMock(); cur.is_running.return_value = False
    stream._subscribed_tickers.discard("ZZZZ"); stream._pending_tickers.discard("ZZZZ")
    with patch.object(stream, "_stream_loop", None), \
         patch("asyncio.get_event_loop", return_value=cur):
        runner._ensure_stream_subscription("ZZZZ")
    # No live stream loop yet → queue for the initial connect to apply.
    assert "ZZZZ" in stream._subscribed_tickers and "ZZZZ" in stream._pending_tickers
    stream._subscribed_tickers.discard("ZZZZ"); stream._pending_tickers.discard("ZZZZ")
