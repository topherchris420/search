from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .embeddings import Embedder
from .exceptions import ValidationError

logger = logging.getLogger(__name__)


class SemanticSearchEngine:
    def __init__(self, embedder: Embedder, data_path: Path) -> None:
        self._embedder = embedder
        self._data_path = data_path
        self._lock = threading.RLock()
        self._documents: list[dict[str, Any]] = []
        self._embedding_matrix = np.zeros((0, embedder.dimension), dtype=np.float32)
        self._index_version = datetime.now(timezone.utc).isoformat()
        self.reindex_from_disk()

    @property
    def index_version(self) -> str:
        return self._index_version

    @property
    def document_count(self) -> int:
        return len(self._documents)

    def _load_documents(self) -> list[dict[str, Any]]:
        if not self._data_path.exists():
            raise ValidationError(f"Data file not found: {self._data_path}")

        with self._data_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if not isinstance(payload, list):
            raise ValidationError("Document dataset must be a JSON array")

        normalized: list[dict[str, Any]] = []
        for idx, item in enumerate(payload):
            if not isinstance(item, dict):
                logger.warning("Skipping document %s: expected object", idx)
                continue

            doc_id = str(item.get("id", f"doc-{idx}"))
            title = str(item.get("title", "Untitled"))
            content = str(item.get("content", ""))
            category = str(item.get("category", "general"))
            source = str(item.get("source", "unknown"))
            security_tier = str(item.get("security_tier", "internal"))
            ontology_type = str(item.get("ontology_type", "entity"))
            ontology_id = str(item.get("ontology_id", doc_id))
            updated_at = str(item.get("updated_at", datetime.now(timezone.utc).isoformat()))
            tags = item.get("tags") or []
            if not isinstance(tags, list):
                tags = []

            normalized.append(
                {
                    "id": doc_id,
                    "title": title,
                    "content": content,
                    "category": category,
                    "source": source,
                    "security_tier": security_tier,
                    "ontology_type": ontology_type,
                    "ontology_id": ontology_id,
                    "tags": [str(tag) for tag in tags],
                    "updated_at": updated_at,
                }
            )

        if not normalized:
            raise ValidationError("No valid documents loaded from dataset")

        return normalized

    @staticmethod
    def _document_text(document: dict[str, Any]) -> str:
        parts = [
            document.get("title", ""),
            document.get("content", ""),
            document.get("category", ""),
            document.get("source", ""),
            document.get("ontology_type", ""),
            " ".join(document.get("tags", [])),
        ]
        return "\n".join(str(part) for part in parts)

    def reindex_from_disk(self) -> dict[str, Any]:
        docs = self._load_documents()
        corpus = [self._document_text(doc) for doc in docs]
        matrix = self._embedder.embed(corpus)

        if len(matrix.shape) != 2 or matrix.shape[0] != len(docs):
            raise ValidationError("Embedding provider returned invalid matrix shape")

        with self._lock:
            self._documents = docs
            self._embedding_matrix = np.asarray(matrix, dtype=np.float32)
            self._index_version = datetime.now(timezone.utc).isoformat()

        logger.info(
            "Index refreshed: documents=%s, dimension=%s, provider=%s",
            len(docs),
            matrix.shape[1],
            self._embedder.name,
        )

        return {
            "status": "ok",
            "documents_indexed": len(docs),
            "dimension": int(matrix.shape[1]),
            "provider": self._embedder.name,
            "index_version": self._index_version,
        }

    @staticmethod
    def _passes_filters(document: dict[str, Any], filters: dict[str, str]) -> bool:
        for key, value in filters.items():
            if not value or value == "all":
                continue
            if str(document.get(key, "")).lower() != value.lower():
                return False
        return True

    @staticmethod
    def _snippet(content: str, limit: int = 240) -> str:
        cleaned = " ".join(content.split())
        if len(cleaned) <= limit:
            return cleaned
        return f"{cleaned[: limit - 3]}..."

    @staticmethod
    def _build_facets(
        ordered_pairs: list[tuple[dict[str, Any], float]],
        pool_size: int = 200,
    ) -> tuple[dict[str, list[dict[str, Any]]], int]:
        facet_keys = ("category", "source", "security_tier", "ontology_type")
        counts: dict[str, dict[str, int]] = {key: {} for key in facet_keys}
        sampled = ordered_pairs[:pool_size]

        for doc, _ in sampled:
            for key in facet_keys:
                value = str(doc.get(key, "unknown"))
                counts[key][value] = counts[key].get(value, 0) + 1

        facets: dict[str, list[dict[str, Any]]] = {}
        for key in facet_keys:
            values = sorted(counts[key].items(), key=lambda item: (-item[1], item[0]))
            facets[key] = [{"value": value, "count": count} for value, count in values]

        return facets, len(sampled)

    def available_filters(self) -> dict[str, list[str]]:
        with self._lock:
            docs = list(self._documents)

        return {
            "category": sorted({doc["category"] for doc in docs}),
            "source": sorted({doc["source"] for doc in docs}),
            "security_tier": sorted({doc["security_tier"] for doc in docs}),
            "ontology_type": sorted({doc["ontology_type"] for doc in docs}),
        }

    def search(
        self,
        query: str,
        filters: dict[str, str],
        page: int,
        page_size: int,
    ) -> dict[str, Any]:
        if page < 1:
            raise ValidationError("page must be >= 1")
        if page_size < 1:
            raise ValidationError("page_size must be >= 1")

        query = (query or "").strip()

        with self._lock:
            docs = list(self._documents)
            matrix = self._embedding_matrix.copy()

        candidate_indices = [
            idx for idx, doc in enumerate(docs) if self._passes_filters(doc, filters)
        ]

        if not candidate_indices:
            return {
                "query": query,
                "filters": filters,
                "page": page,
                "page_size": page_size,
                "total": 0,
                "total_pages": 0,
                "has_next": False,
                "has_prev": page > 1,
                "results": [],
                "facets": {
                    "category": [],
                    "source": [],
                    "security_tier": [],
                    "ontology_type": [],
                },
                "facet_pool_size": 0,
                "index_version": self._index_version,
            }

        candidate_matrix = matrix[candidate_indices]
        candidate_docs = [docs[idx] for idx in candidate_indices]

        if query:
            query_vector = self._embedder.embed([query])[0]
            norms = np.linalg.norm(candidate_matrix, axis=1)
            query_norm = np.linalg.norm(query_vector)
            denom = np.maximum(norms * max(query_norm, 1e-12), 1e-12)
            scores = np.dot(candidate_matrix, query_vector) / denom
            ordered_pairs = sorted(
                zip(candidate_docs, scores.tolist(), strict=False),
                key=lambda pair: pair[1],
                reverse=True,
            )
        else:
            ordered_pairs = sorted(
                [(doc, 0.0) for doc in candidate_docs],
                key=lambda pair: pair[0].get("updated_at", ""),
                reverse=True,
            )

        facets, facet_pool_size = self._build_facets(ordered_pairs, pool_size=200)

        total = len(ordered_pairs)
        total_pages = int(np.ceil(total / page_size))
        page = min(page, max(total_pages, 1))
        start = (page - 1) * page_size
        end = start + page_size
        page_items = ordered_pairs[start:end]

        results = []
        for doc, score in page_items:
            results.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "snippet": self._snippet(doc["content"]),
                    "content": doc["content"],
                    "score": round(float(score), 5),
                    "category": doc["category"],
                    "source": doc["source"],
                    "security_tier": doc["security_tier"],
                    "ontology_type": doc["ontology_type"],
                    "ontology_id": doc["ontology_id"],
                    "tags": doc["tags"],
                    "updated_at": doc["updated_at"],
                }
            )

        return {
            "query": query,
            "filters": filters,
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1,
            "results": results,
            "facets": facets,
            "facet_pool_size": facet_pool_size,
            "index_version": self._index_version,
        }
