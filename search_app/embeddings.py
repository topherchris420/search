from __future__ import annotations

import hashlib
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np

from .config import Settings
from .retry import retry

logger = logging.getLogger(__name__)
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_:-]+")


class Embedder(ABC):
    name: str
    dimension: int

    @abstractmethod
    def embed(self, texts: Iterable[str]) -> np.ndarray:
        raise NotImplementedError


class HashEmbedder(Embedder):
    def __init__(self, dimension: int) -> None:
        self.name = "hash"
        self.dimension = dimension

    def _vectorize(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = _TOKEN_PATTERN.findall(text.lower())
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for i in range(0, len(digest), 4):
                chunk = digest[i : i + 4]
                if len(chunk) < 4:
                    continue
                bucket = int.from_bytes(chunk, "little") % self.dimension
                sign = 1.0 if chunk[0] % 2 == 0 else -1.0
                vector[bucket] += sign

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        return np.vstack([self._vectorize(text) for text in texts])


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        self.name = "sentence-transformers"
        self._model = SentenceTransformer(model_name)
        self.dimension = int(self._model.get_sentence_embedding_dimension())

    @retry(attempts=3, initial_delay=0.1)
    def embed(self, texts: Iterable[str]) -> np.ndarray:
        values = self._model.encode(list(texts), normalize_embeddings=True)
        return np.asarray(values, dtype=np.float32)


class OpenAIEmbedder(Embedder):
    def __init__(self, model_name: str) -> None:
        from openai import OpenAI  # type: ignore

        self.name = "openai"
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._model_name = model_name
        self.dimension = 1536

    @retry(attempts=3, initial_delay=0.2)
    def embed(self, texts: Iterable[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            result = self._client.embeddings.create(model=self._model_name, input=text)
            vectors.append(np.asarray(result.data[0].embedding, dtype=np.float32))
        matrix = np.vstack(vectors)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = np.divide(matrix, np.maximum(norms, 1e-12))
        return matrix


def build_embedder(settings: Settings) -> Embedder:
    provider = settings.embedding_provider.strip().lower()

    if provider == "sentence-transformers":
        try:
            embedder = SentenceTransformerEmbedder(settings.embedding_model)
            logger.info("Using sentence-transformers embedder: %s", settings.embedding_model)
            return embedder
        except Exception as exc:
            logger.warning("Failed to initialize sentence-transformers embedder: %s", exc)

    if provider == "openai":
        try:
            embedder = OpenAIEmbedder(settings.embedding_model)
            logger.info("Using OpenAI embedding model: %s", settings.embedding_model)
            return embedder
        except Exception as exc:
            logger.warning("Failed to initialize OpenAI embedder: %s", exc)

    logger.info("Using hash embedder fallback (dimension=%s)", settings.embedding_dimension)
    return HashEmbedder(settings.embedding_dimension)
