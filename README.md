# Vers3Dynamics Search

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/flask-3.x-black)](https://flask.palletsprojects.com/)
[![Tests](https://img.shields.io/badge/tests-pytest-success)](./tests)
[![Redis Cache](https://img.shields.io/badge/cache-redis-red)](https://redis.io/)
[![Zero Trust](https://img.shields.io/badge/security-zero--trust-0f766e)](#zero-trust-security-baseline)

Production-focused semantic search service with embeddings + cosine similarity, Tailwind UI, zero-trust API controls, Redis caching, retries, pagination, and ontology-ready metadata.

![Demo GIF](./docs/demo.gif)

## Demo
- UI: `GET /`
- Health: `GET /api/health`
- Search API: `POST /api/search`

## Core Capabilities
- Semantic search using embeddings + cosine similarity (`hash` fallback, optional `sentence-transformers` or `openai` providers)
- Responsive Tailwind UI with live filters, pagination, dark mode, and retry-aware error UX
- Zero-trust-ready API auth (`Authorization: Bearer <token>` / `X-API-Key`)
- Request rate limiting with Redis-backed counter fallback to in-memory
- Redis result caching with TTL and graceful memory fallback
- Structured filter dimensions: `category`, `source`, `security_tier`, `ontology_type`
- Ontology-ready fields: `ontology_id`, `ontology_type`, `tags`, source provenance
- Pytest coverage for auth, search, pagination, error handling, and reindexing

## Architecture

```text
Client (Tailwind UI) --> Flask API --> Semantic Engine --> Embedding Provider
                           |                |                |
                           |                |                +-- sentence-transformers/openai/hash
                           |                +-- cosine sim + filters + pagination
                           |
                           +-- Zero-trust auth + rate limiter + retries + error handlers
                           +-- Redis cache (fallback memory)
```

## Project Layout

```text
search/
  app.py
  search_app/
    __init__.py
    api.py
    cache.py
    config.py
    embeddings.py
    exceptions.py
    retry.py
    security.py
    semantic_engine.py
  templates/
    index.html
  static/
    app.js
  data/
    documents.json
  tests/
    conftest.py
    test_api.py
  palantir/
    aip_logic.yaml
```

## Quick Start

### 1. Install

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure

```bash
# Required for zero-trust mode
set ZERO_TRUST_ENABLED=true
set ZERO_TRUST_API_KEY=change-me

# Optional
set EMBEDDING_PROVIDER=hash
set REDIS_URL=redis://localhost:6379/0
set SEARCH_DATA_PATH=C:\path\to\documents.json
```

### 3. Run

```bash
python app.py
```

Open `http://localhost:5000`.

## API Examples

### Health

```bash
curl -s http://localhost:5000/api/health
```

### Filters

```bash
curl -s http://localhost:5000/api/filters \
  -H "Authorization: Bearer change-me"
```

### Search

```bash
curl -s http://localhost:5000/api/search \
  -H "Authorization: Bearer change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "mission corridor interference",
    "filters": {
      "category": "all",
      "source": "all",
      "security_tier": "all",
      "ontology_type": "all"
    },
    "page": 1,
    "page_size": 10
  }'
```

### Reindex

```bash
curl -s -X POST http://localhost:5000/api/reindex \
  -H "Authorization: Bearer change-me"
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `APP_NAME` | `Semantic Ontology Search` | Service name |
| `APP_ENV` | `production` | Environment label |
| `ZERO_TRUST_ENABLED` | `true` | Enforce API credential checks |
| `ZERO_TRUST_API_KEY` | `change-me-in-production` | Shared API credential |
| `ALLOWED_ORIGINS` | `http://localhost:5000` | CORS allowlist (comma-separated) |
| `SEARCH_DATA_PATH` | `./data/documents.json` | JSON dataset path |
| `EMBEDDING_PROVIDER` | `hash` | `hash`, `sentence-transformers`, `openai` |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model identifier |
| `EMBEDDING_DIMENSION` | `384` | Hash embedding vector size |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis DSN |
| `CACHE_TTL_SECONDS` | `120` | Search response cache TTL |
| `RATE_LIMIT_PER_MINUTE` | `120` | Requests per identity per window |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate-limit window |
| `DEFAULT_PAGE_SIZE` | `10` | API/UI default page size |
| `MAX_PAGE_SIZE` | `50` | API max page size |

## Zero-Trust Security Baseline
- Credential enforcement on protected endpoints (`/api/filters`, `/api/search`, `/api/reindex`)
- Rate limiting (Redis-backed where available)
- Strict security headers (CSP, frame deny, no-sniff, no-referrer)
- Minimal request body size cap (`MAX_CONTENT_LENGTH`)
- Constant-time token comparison
- Request ID + rate-limit metadata returned in headers

## Testing

```bash
pytest -q
```

Tests cover:
- health endpoint behavior
- auth requirements
- search correctness and cache hit path
- filter + pagination logic
- invalid payload handling
- reindex endpoint

## Palantir AIP Logic Integration

See [`palantir/aip_logic.yaml`](./palantir/aip_logic.yaml) for an ontology-ready pipeline template that maps search anomalies/events into downstream decision workflows.

## Notes
- `hash` embedding mode is deterministic and dependency-light.
- For higher semantic quality, install `sentence-transformers` and set `EMBEDDING_PROVIDER=sentence-transformers`.
- To use remote embeddings, install `openai`, set `OPENAI_API_KEY`, and set `EMBEDDING_PROVIDER=openai`.
