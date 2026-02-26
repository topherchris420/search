from __future__ import annotations

import logging
from typing import Any

from flask import Flask, jsonify, render_template
from flask_cors import CORS

from .api import build_api_blueprint
from .cache import RedisCache
from .config import Settings
from .embeddings import build_embedder
from .exceptions import AppError, ValidationError
from .security import RateLimiter, attach_security_headers, secure_endpoint
from .semantic_engine import SemanticSearchEngine


def create_app(settings: Settings | None = None) -> Flask:
    settings = settings or Settings.from_env()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    app.config["MAX_CONTENT_LENGTH"] = settings.max_request_bytes

    CORS(
        app,
        resources={r"/api/*": {"origins": settings.allowed_origins}},
        supports_credentials=False,
    )

    embedder = build_embedder(settings)
    engine = SemanticSearchEngine(embedder=embedder, data_path=settings.data_path)
    cache = RedisCache(settings.redis_url)
    limiter = RateLimiter(settings)
    secure = secure_endpoint(settings, limiter)

    app.extensions["settings"] = settings
    app.extensions["engine"] = engine
    app.extensions["cache"] = cache

    app.register_blueprint(build_api_blueprint(settings, engine, cache, secure))

    @app.get("/")
    def index() -> Any:
        bootstrap_api_key = ""
        auth_hint = ""
        if settings.zero_trust_enabled:
            auth_hint = "Use the same value configured in ZERO_TRUST_API_KEY."
            if settings.api_key == "change-me-in-production":
                bootstrap_api_key = settings.api_key
                auth_hint = (
                    "Default local key loaded automatically. "
                    "Set ZERO_TRUST_API_KEY to a unique secret for production."
                )

        return render_template(
            "index.html",
            app_name=settings.app_name,
            zero_trust_enabled=settings.zero_trust_enabled,
            default_page_size=settings.default_page_size,
            bootstrap_api_key=bootstrap_api_key,
            auth_hint=auth_hint,
        )

    @app.errorhandler(ValidationError)
    def handle_validation_error(error: ValidationError):
        return jsonify({"error": error.message}), error.status_code

    @app.errorhandler(AppError)
    def handle_app_error(error: AppError):
        return jsonify({"error": error.message}), error.status_code

    @app.errorhandler(404)
    def not_found(_: Any):
        return jsonify({"error": "Not found"}), 404

    @app.errorhandler(Exception)
    def unhandled_error(error: Exception):
        logger.exception("Unhandled error: %s", error)
        return jsonify({"error": "Internal server error"}), 500

    @app.after_request
    def security_headers(response):
        return attach_security_headers(response)

    logger.info(
        "App initialized | docs=%s | zero_trust=%s | provider=%s",
        engine.document_count,
        settings.zero_trust_enabled,
        embedder.name,
    )

    return app
