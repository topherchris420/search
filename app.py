"""WSGI/ASGI-friendly entrypoint for the semantic search service."""

from __future__ import annotations

import os

from search_app import create_app

app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)


__all__ = ["app", "create_app"]
