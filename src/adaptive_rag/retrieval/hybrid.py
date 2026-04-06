from __future__ import annotations

import numpy as np

from adaptive_rag.indexing_bm25 import BM25Index
from adaptive_rag.indexing_faiss import FaissIndex
from adaptive_rag.types import Chunk


def reciprocal_rank_fusion(
    dense_order: list[int],
    bm25_order: list[int],
    k: int,
    n: int,
) -> list[int]:
    """RRF over two rankings (indices into chunk list). Returns sorted chunk indices by score desc."""
    scores = np.zeros(n, dtype=np.float64)
    for rank, idx in enumerate(dense_order):
        if 0 <= idx < n:
            scores[idx] += 1.0 / (k + rank + 1)
    for rank, idx in enumerate(bm25_order):
        if 0 <= idx < n:
            scores[idx] += 1.0 / (k + rank + 1)
    return list(np.argsort(-scores))


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        faiss_index: FaissIndex,
        bm25_index: BM25Index,
        rrf_k: int,
    ) -> None:
        self.chunks = chunks
        self._faiss = faiss_index
        self._bm25 = bm25_index
        self._rrf_k = rrf_k
        self._id_to_i = {c.chunk_id: i for i, c in enumerate(chunks)}

    def retrieve(self, query_text: str, query_vec: np.ndarray, top_k: int) -> tuple[list[Chunk], list[float]]:
        n = len(self.chunks)
        dense_chunks, dense_scores = self._faiss.search(query_vec, top_k=min(n, max(top_k * 4, top_k)))
        dense_order = [self._id_to_i[c.chunk_id] for c in dense_chunks]

        bm = self._bm25.scores(query_text)
        bm25_order = list(np.argsort(-bm))[: max(top_k * 4, top_k)]

        fused = reciprocal_rank_fusion(dense_order, bm25_order, self._rrf_k, n)[:top_k]
        out_chunks = [self.chunks[i] for i in fused]
        out_scores = [1.0 / (self._rrf_k + rank + 1) for rank in range(len(out_chunks))]
        return out_chunks, out_scores


class DenseOnlyRetriever:
    def __init__(self, faiss_index: FaissIndex) -> None:
        self._faiss = faiss_index

    def retrieve(self, query_text: str, query_vec: np.ndarray, top_k: int) -> tuple[list[Chunk], list[float]]:
        ch, sc = self._faiss.search(query_vec, top_k)
        return ch, [float(x) for x in sc]


class MMARetriever:
    """Text-only multi-step: hybrid candidate pool + dense re-rank (proxy for multi-agent refinement)."""

    def __init__(
        self,
        hybrid: HybridRetriever,
        chunk_embeddings: dict[str, np.ndarray],
    ) -> None:
        self._hybrid = hybrid
        self._emb = chunk_embeddings

    def retrieve(self, query_text: str, query_vec: np.ndarray, top_k: int) -> tuple[list[Chunk], list[float]]:
        cand_k = min(len(self._hybrid.chunks), max(top_k * 8, 20))
        ch, _ = self._hybrid.retrieve(query_text, query_vec, top_k=cand_k)
        q = query_vec.flatten()
        scored: list[tuple[Chunk, float]] = []
        for c in ch:
            e = self._emb.get(c.chunk_id)
            if e is None:
                continue
            scored.append((c, float(np.dot(q, e))))
        scored.sort(key=lambda x: -x[1])
        out = scored[:top_k]
        return [c for c, _ in out], [s for _, s in out]
