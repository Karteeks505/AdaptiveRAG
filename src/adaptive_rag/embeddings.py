from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class EmbeddingBackend(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return float32 array shape (n, dim), L2-normalized rows for inner product."""


class HashEmbeddingBackend(EmbeddingBackend):
    """Deterministic bag-of-hashes embedding (no external models). For reproducible local runs."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float64)
        for w in text.lower().split():
            h = hashlib.sha256(w.encode("utf-8")).digest()
            for i in range(0, min(len(h), 8)):
                idx = int(h[i]) % self.dim
                v[idx] += 1.0
        n = np.linalg.norm(v) + 1e-12
        return (v / n).astype(np.float32)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts], axis=0)


class SentenceTransformerBackend(EmbeddingBackend):
    def __init__(self, model_id: str, batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_id)
        self._batch_size = batch_size

    def embed(self, texts: list[str]) -> np.ndarray:
        emb = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(emb, dtype=np.float32)


class OpenAIBackend(EmbeddingBackend):
    def __init__(self, model_id: str, dimensions: int | None = None) -> None:
        from openai import OpenAI

        self._client = OpenAI()
        self._model_id = model_id
        self._dimensions = dimensions

    def embed(self, texts: list[str]) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), 64):
            batch = texts[i : i + 64]
            kwargs: dict[str, Any] = {"model": self._model_id, "input": batch}
            if self._dimensions is not None:
                kwargs["dimensions"] = self._dimensions
            resp = self._client.embeddings.create(**kwargs)
            for d in sorted(resp.data, key=lambda x: x.index):
                out.append(list(d.embedding))
        arr = np.asarray(out, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return arr / norms


def build_embedding_backend(cfg: dict[str, Any]) -> EmbeddingBackend:
    emb = cfg["embedding"]
    backend = emb["backend"]
    if backend == "hash":
        return HashEmbeddingBackend(int(emb.get("dim", 384)))
    if backend == "sentence_transformers":
        return SentenceTransformerBackend(emb["model_id"], emb.get("batch_size", 32))
    if backend == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY must be set for openai embedding backend")
        return OpenAIBackend(emb["model_id"], emb.get("dimensions"))
    raise ValueError(f"Unknown embedding backend: {backend}")
