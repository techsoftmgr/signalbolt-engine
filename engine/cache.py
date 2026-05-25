"""
Shared key/value cache — Redis-backed when REDIS_URL is set, in-memory
process-local otherwise. Same interface either way so callers don't care.

Why this exists:
  Each Fly app machine has its own Python process. A user request can land
  on any machine, so per-process caches let the same hot data trigger N
  Alpaca calls across N machines instead of 1. Redis collapses that fanout.

Why fall back to in-memory:
  - Local dev doesn't need Redis running
  - Redis outage shouldn't take SignalBolt down — better to over-call
    Alpaca for a few minutes than serve 500s
  - Lets us ship this code today and add Redis later by just setting
    REDIS_URL on Fly; no separate code change required

Provider:
  set REDIS_URL=redis://default:PASSWORD@HOST:6379  on Fly secrets.
  Easiest: `fly redis create` (Fly's Upstash integration) — gives you the
  URL automatically wired as REDIS_URL.

Usage:
    from engine.cache import kv

    val = kv.get_json("prices:AAPL")            # returns dict or None
    kv.set_json("prices:AAPL", {"price": 184.5}, ttl_sec=15)

API surface kept tiny on purpose — get_json / set_json / delete cover
every cache pattern in the engine.
"""

import json
import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger("signalbolt.cache")

_REDIS_URL = os.environ.get("REDIS_URL", "").strip()


class _InMemoryKV:
    """Fallback used when REDIS_URL is unset or Redis can't be reached."""

    def __init__(self) -> None:
        # value, expires_at_monotonic
        self._store: dict[str, tuple[str, float]] = {}

    def get(self, key: str) -> Optional[str]:
        entry = self._store.get(key)
        if not entry:
            return None
        value, expires_at = entry
        if time.monotonic() >= expires_at:
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: str, ttl_sec: int) -> None:
        self._store[key] = (value, time.monotonic() + ttl_sec)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    @property
    def backend(self) -> str:
        return "memory"


class _RedisKV:
    """
    Redis-backed KV. Lazily connects on first use. Any Redis error falls
    back to the in-memory store for that operation only — the next call
    re-tries Redis. So a transient blip doesn't poison the cache.
    """

    def __init__(self, url: str) -> None:
        self._url = url
        self._client = None
        self._fallback = _InMemoryKV()
        self._last_failure_ts: float = 0.0
        # If Redis fails, don't retry for this long — avoid hammering a
        # dead Redis on every request.
        self._failure_cooldown_sec = 30

    def _get_client(self):
        if self._client is None:
            import redis  # imported lazily so the module loads without redis-py
            self._client = redis.Redis.from_url(
                self._url,
                decode_responses=True,
                socket_timeout=2.0,        # don't hang the request on a slow Redis
                socket_connect_timeout=2.0,
                health_check_interval=30,
                retry_on_timeout=False,
            )
        return self._client

    def _is_in_cooldown(self) -> bool:
        return (time.monotonic() - self._last_failure_ts) < self._failure_cooldown_sec

    def get(self, key: str) -> Optional[str]:
        if self._is_in_cooldown():
            return self._fallback.get(key)
        try:
            return self._get_client().get(key)
        except Exception as e:
            self._last_failure_ts = time.monotonic()
            logger.warning(f"[cache] Redis GET failed for {key} — falling back to memory: {e}")
            return self._fallback.get(key)

    def set(self, key: str, value: str, ttl_sec: int) -> None:
        # Always write to local fallback so reads work even if Redis is dead
        self._fallback.set(key, value, ttl_sec)
        if self._is_in_cooldown():
            return
        try:
            self._get_client().setex(key, ttl_sec, value)
        except Exception as e:
            self._last_failure_ts = time.monotonic()
            logger.warning(f"[cache] Redis SETEX failed for {key} — falling back to memory: {e}")

    def delete(self, key: str) -> None:
        self._fallback.delete(key)
        if self._is_in_cooldown():
            return
        try:
            self._get_client().delete(key)
        except Exception as e:
            self._last_failure_ts = time.monotonic()
            logger.warning(f"[cache] Redis DELETE failed for {key}: {e}")

    @property
    def backend(self) -> str:
        return "redis" if not self._is_in_cooldown() else "redis-fallback"


# ── Public KV singleton ──────────────────────────────────────────────────────

class _JsonKV:
    """Wraps a string KV with JSON encode/decode for typed dict storage."""

    def __init__(self, raw) -> None:
        self._raw = raw

    def get_json(self, key: str) -> Optional[Any]:
        s = self._raw.get(key)
        if s is None:
            return None
        try:
            return json.loads(s)
        except Exception:
            return None

    def set_json(self, key: str, value: Any, ttl_sec: int) -> None:
        try:
            self._raw.set(key, json.dumps(value, default=str), ttl_sec)
        except Exception as e:
            logger.debug(f"[cache] set_json failed for {key}: {e}")

    def delete(self, key: str) -> None:
        self._raw.delete(key)

    @property
    def backend(self) -> str:
        return self._raw.backend


def _build_kv() -> _JsonKV:
    if _REDIS_URL:
        logger.info(f"[cache] Redis backend enabled (REDIS_URL set)")
        return _JsonKV(_RedisKV(_REDIS_URL))
    logger.info("[cache] In-memory backend (set REDIS_URL to enable shared cache)")
    return _JsonKV(_InMemoryKV())


# Module-level singleton — import from engine.cache
kv: _JsonKV = _build_kv()
