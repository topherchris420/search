from __future__ import annotations

from typing import Any

from flask import Blueprint, jsonify, request

from .cache import RedisCache, make_cache_key
from .config import Settings
from .exceptions import ValidationError
from .semantic_engine import SemanticSearchEngine


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValidationError("filters must be an object")
    return value


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError("page and page_size must be integers") from exc


def build_api_blueprint(
    settings: Settings,
    engine: SemanticSearchEngine,
    cache: RedisCache,
    secure,
) -> Blueprint:
    api = Blueprint("api", __name__, url_prefix="/api")

    @api.get("/health")
    def health() -> Any:
        return jsonify(
            {
                "status": "ok",
                "service": settings.app_name,
                "documents_indexed": engine.document_count,
                "index_version": engine.index_version,
            }
        )

    @api.get("/filters")
    @secure
    def filters() -> Any:
        return jsonify(engine.available_filters())

    @api.post("/search")
    @secure
    def search() -> Any:
        body = request.get_json(silent=True)
        if body is None:
            raise ValidationError("JSON payload is required")

        query = str(body.get("query", ""))
        filters = _as_dict(body.get("filters", {}))
        page = _as_int(body.get("page"), 1)
        page_size = _as_int(body.get("page_size"), settings.default_page_size)
        page_size = min(page_size, settings.max_page_size)

        normalized_filters = {
            "category": str(filters.get("category", "all")),
            "source": str(filters.get("source", "all")),
            "security_tier": str(filters.get("security_tier", "all")),
            "ontology_type": str(filters.get("ontology_type", "all")),
        }

        cache_key = make_cache_key(
            "search",
            {
                "query": query,
                "filters": normalized_filters,
                "page": page,
                "page_size": page_size,
                "index_version": engine.index_version,
            },
        )
        cached = cache.get_json(cache_key)
        if cached is not None:
            cached["cache_hit"] = True
            return jsonify(cached)

        payload = engine.search(
            query=query,
            filters=normalized_filters,
            page=page,
            page_size=page_size,
        )
        payload["cache_hit"] = False
        cache.set_json(cache_key, payload, settings.cache_ttl_seconds)
        return jsonify(payload)

    @api.post("/reindex")
    @secure
    def reindex() -> Any:
        report = engine.reindex_from_disk()
        return jsonify(report)

    return api
