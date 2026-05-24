"""
Small helpers for surfacing silently-swallowed exceptions.

The codebase has many `except Exception: pass` blocks. Most are intentional
graceful-degradation in hot loops (cache updates, optional metadata writes)
where letting the exception escape would kill a scan or close handler that
must keep running.

The downside: when something genuinely breaks (Supabase outage, schema drift,
SDK bug), those blocks hide the failure with zero log output. Debugging
becomes "no signals fired today and there are no error messages".

This module provides:

  swallow(name, level="debug")
      Context manager that runs the block, catches any Exception, and logs
      it at the requested level instead of vanishing it. Use it instead of
      bare try/except: pass.

Usage:

      from engine.safe import swallow

      with swallow("push.signal_close"):
          push._send_raw(...)

      with swallow("supabase.event_insert", level="warning"):
          sb.table("signal_events").insert(row).execute()

Pick the log level based on how much you care:
  - debug   = "I know this might fail, just want it in the log if it does"
  - warning = "if this fails consistently I want to know"
  - error   = "this failing means something is broken" (use try/except instead
              if you want a stack trace)
"""

import logging
from contextlib import contextmanager
from typing import Iterator

_logger = logging.getLogger("signalbolt.safe")

_LEVELS = {
    "debug":   logging.DEBUG,
    "info":    logging.INFO,
    "warning": logging.WARNING,
    "error":   logging.ERROR,
}


@contextmanager
def swallow(name: str, level: str = "debug") -> Iterator[None]:
    """
    Context manager: run the block, catch any Exception, log it.
    Never raises — drop-in replacement for `try: ... except Exception: pass`.
    """
    try:
        yield
    except Exception as e:
        log_level = _LEVELS.get(level, logging.DEBUG)
        _logger.log(log_level, "[swallow:%s] %s: %s", name, type(e).__name__, e)
