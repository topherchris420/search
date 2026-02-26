from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

from .retry import retry

logger = logging.getLogger(__name__)

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
    redis = None


@dataclass(slots=True)
class CacheEntry:
    payload: str
    expires_at: float


class MemoryCache:
    def __init__(self) -> None:
        self._store: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()

    def get_json(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if entry.expires_at < time.time():
                self._store.pop(key, None)
                return None
            return json.loads(entry.payload)

    def set_json(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None:
        payload = json.dumps(value, separators=(",", ":"), default=str)
        with self._lock:
            self._store[key] = CacheEntry(payload=payload, expires_at=time.time() + ttl_seconds)


class RedisCache:
    def __init__(self, redis_url: str) -> None:
        self._client = None
        self._memory_fallback = MemoryCache()

        if redis is None:
            logger.warning("redis package unavailable; using in-memory cache")
            return

        try:
            self._client = redis.Redis.from_url(redis_url, decode_responses=True)
            self._ping()
            logger.info("Redis cache enabled: %s", redis_url)
        except Exception as exc:
            logger.warning("Redis unavailable, falling back to in-memory cache: %s", exc)
            self._client = None

    @retry(attempts=2, initial_delay=0.05, retry_on=(Exception,))
    def _ping(self) -> None:
        if self._client is None:
            return
        self._client.ping()

    def get_json(self, key: str) -> dict[str, Any] | None:
        if self._client is None:
            return self._memory_fallback.get_json(key)

        try:
            raw = self._client.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:
            logger.warning("Redis get failed; serving from memory fallback: %s", exc)
            return self._memory_fallback.get_json(key)

    def set_json(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None:
        self._memory_fallback.set_json(key, value, ttl_seconds)
        if self._client is None:
            return

        try:
            payload = json.dumps(value, separators=(",", ":"), default=str)
            self._client.setex(key, ttl_seconds, payload)
        except Exception as exc:
            logger.warning("Redis set failed; memory fallback retained value: %s", exc)


def make_cache_key(prefix: str, payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"
