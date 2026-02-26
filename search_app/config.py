from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if minimum is not None:
        value = max(minimum, value)
    return value


def _env_list(name: str, default: List[str]) -> List[str]:
    raw = os.getenv(name)
    if raw is None:
        return default
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return items if items else default


@dataclass(slots=True)
class Settings:
    app_name: str
    environment: str
    zero_trust_enabled: bool
    api_key: str
    allowed_origins: List[str]
    redis_url: str
    cache_ttl_seconds: int
    rate_limit_per_minute: int
    rate_limit_window_seconds: int
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    default_page_size: int
    max_page_size: int
    max_request_bytes: int
    data_path: Path

    @classmethod
    def from_env(cls) -> "Settings":
        root = Path(__file__).resolve().parents[1]
        data_path = Path(os.getenv("SEARCH_DATA_PATH", root / "data" / "documents.json"))

        return cls(
            app_name=os.getenv("APP_NAME", "Semantic Ontology Search"),
            environment=os.getenv("APP_ENV", "production"),
            zero_trust_enabled=_env_bool("ZERO_TRUST_ENABLED", True),
            api_key=os.getenv("ZERO_TRUST_API_KEY", "change-me-in-production").strip(),
            allowed_origins=_env_list("ALLOWED_ORIGINS", ["http://localhost:5000"]),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            cache_ttl_seconds=_env_int("CACHE_TTL_SECONDS", 120, minimum=30),
            rate_limit_per_minute=_env_int("RATE_LIMIT_PER_MINUTE", 120, minimum=10),
            rate_limit_window_seconds=_env_int("RATE_LIMIT_WINDOW_SECONDS", 60, minimum=10),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "hash"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            embedding_dimension=_env_int("EMBEDDING_DIMENSION", 384, minimum=32),
            default_page_size=_env_int("DEFAULT_PAGE_SIZE", 10, minimum=1),
            max_page_size=_env_int("MAX_PAGE_SIZE", 50, minimum=1),
            max_request_bytes=_env_int("MAX_REQUEST_BYTES", 1_048_576, minimum=1024),
            data_path=data_path,
        )
