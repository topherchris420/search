from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar


T = TypeVar("T")


def retry(
    attempts: int = 3,
    initial_delay: float = 0.15,
    backoff: float = 2.0,
    max_delay: float = 1.5,
    jitter: float = 0.1,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Simple retry decorator with exponential backoff and jitter."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args, **kwargs) -> T:
            delay = initial_delay
            last_error: BaseException | None = None
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retry_on as exc:  # type: ignore[misc]
                    last_error = exc
                    if attempt >= attempts:
                        raise
                    sleep_time = min(max_delay, delay) + random.uniform(0, jitter)
                    time.sleep(sleep_time)
                    delay *= backoff
            if last_error is not None:
                raise last_error
            raise RuntimeError("retry loop exited unexpectedly")

        return wrapper

    return decorator
