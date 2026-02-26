from __future__ import annotations

import hmac
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from functools import wraps
from typing import Callable

from flask import Request, g, jsonify, request

from .config import Settings
from .exceptions import RateLimitError, UnauthorizedError

logger = logging.getLogger(__name__)

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None


@dataclass(slots=True)
class RateLimitState:
    count: int
    reset_at: float


class RateLimiter:
    def __init__(self, settings: Settings) -> None:
        self.limit = settings.rate_limit_per_minute
        self.window_seconds = settings.rate_limit_window_seconds
        self._memory_store: dict[str, RateLimitState] = {}
        self._lock = threading.Lock()
        self._redis_client = None

        if redis is None:
            return

        try:
            self._redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
            self._redis_client.ping()
        except Exception as exc:
            logger.warning("Rate limiter redis unavailable; using memory fallback: %s", exc)
            self._redis_client = None

    def _memory_check(self, key: str) -> tuple[bool, int, int]:
        now = time.time()
        with self._lock:
            state = self._memory_store.get(key)
            if state is None or state.reset_at <= now:
                state = RateLimitState(count=0, reset_at=now + self.window_seconds)
                self._memory_store[key] = state

            state.count += 1
            remaining = max(0, self.limit - state.count)
            reset = int(max(0, state.reset_at - now))
            return state.count <= self.limit, remaining, reset

    def _redis_check(self, key: str) -> tuple[bool, int, int]:
        assert self._redis_client is not None
        now = int(time.time())
        bucket = now // self.window_seconds
        redis_key = f"rl:{key}:{bucket}"

        count = int(self._redis_client.incr(redis_key))
        if count == 1:
            self._redis_client.expire(redis_key, self.window_seconds)

        remaining = max(0, self.limit - count)
        reset = self.window_seconds - (now % self.window_seconds)
        return count <= self.limit, remaining, reset

    def check(self, identity: str) -> tuple[bool, int, int]:
        if self._redis_client is not None:
            try:
                return self._redis_check(identity)
            except Exception as exc:
                logger.warning("Redis limiter error; fallback to memory: %s", exc)
        return self._memory_check(identity)


def get_client_ip(req: Request) -> str:
    forwarded = req.headers.get("X-Forwarded-For", "").strip()
    if forwarded:
        return forwarded.split(",")[0].strip()
    return req.remote_addr or "unknown"


def extract_token(req: Request) -> str:
    bearer = req.headers.get("Authorization", "").strip()
    if bearer.lower().startswith("bearer "):
        return bearer[7:].strip().strip('"').strip("'")
    api_key = req.headers.get("X-API-Key", "").strip()
    return api_key.strip('"').strip("'")


def secure_endpoint(settings: Settings, rate_limiter: RateLimiter) -> Callable:
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            g.request_id = str(uuid.uuid4())
            client_ip = get_client_ip(request)

            allowed, remaining, reset = rate_limiter.check(client_ip)
            if not allowed:
                raise RateLimitError(f"Rate limit exceeded. Retry in {reset}s")

            g.rate_limit_remaining = remaining
            g.rate_limit_reset_seconds = reset

            if settings.zero_trust_enabled:
                token = extract_token(request)
                if not token or not hmac.compare_digest(token, settings.api_key):
                    raise UnauthorizedError("Invalid API credential")

            return func(*args, **kwargs)

        return wrapper

    return decorator


def attach_security_headers(response):
    # Zero-trust baseline security headers.
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    response.headers[
        "Content-Security-Policy"
    ] = (
        "default-src 'self'; "
        "script-src 'self' https://cdn.tailwindcss.com; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "object-src 'none';"
    )

    if request.is_secure:
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"

    if hasattr(g, "request_id"):
        response.headers["X-Request-Id"] = g.request_id
    if hasattr(g, "rate_limit_remaining"):
        response.headers["X-RateLimit-Remaining"] = str(g.rate_limit_remaining)
    if hasattr(g, "rate_limit_reset_seconds"):
        response.headers["X-RateLimit-Reset"] = str(g.rate_limit_reset_seconds)

    return response


def unauthorized_response(message: str):
    return jsonify({"error": message}), 401
