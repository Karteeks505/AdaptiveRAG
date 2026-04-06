from __future__ import annotations

import numpy as np

from adaptive_rag.types import Chunk


class FaissIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        import faiss

        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        self._chunks = list(chunks)
        dim = vectors.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(vectors.astype(np.float32))

    def search(self, query_vec: np.ndarray, top_k: int) -> tuple[list[Chunk], np.ndarray]:
        q = query_vec.reshape(1, -1).astype(np.float32)
        scores, idxs = self._index.search(q, min(top_k, len(self._chunks)))
        picked: list[Chunk] = []
        sc: list[float] = []
        for j, i in enumerate(idxs[0]):
            if i < 0:
                continue
            picked.append(self._chunks[i])
            sc.append(float(scores[0][j]))
        return picked, np.asarray(sc, dtype=np.float32)
