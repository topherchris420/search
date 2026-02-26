from __future__ import annotations

from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from search_app.config import Settings


def test_zero_trust_api_key_is_trimmed(monkeypatch):
    monkeypatch.setenv("ZERO_TRUST_API_KEY", "  test-token\n")

    settings = Settings.from_env()

    assert settings.api_key == "test-token"
