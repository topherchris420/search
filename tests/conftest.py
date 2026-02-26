from __future__ import annotations

from pathlib import Path
import sys

import pytest


@pytest.fixture()
def app(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    monkeypatch.setenv("ZERO_TRUST_ENABLED", "true")
    monkeypatch.setenv("ZERO_TRUST_API_KEY", "test-token")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6391/0")
    monkeypatch.setenv("SEARCH_DATA_PATH", str(root / "data" / "documents.json"))
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:5000")

    from search_app import create_app

    flask_app = create_app()
    flask_app.config.update(TESTING=True)
    return flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def auth_headers():
    return {"Authorization": "Bearer test-token"}
