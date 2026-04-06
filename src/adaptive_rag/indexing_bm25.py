from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from adaptive_rag.types import Chunk


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in text.split() if t.strip()]


class BM25Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        corpus = [tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(corpus)

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        s = np.asarray(self._bm25.get_scores(q), dtype=np.float32)
        if s.size == 0:
            return np.zeros(len(self._chunks), dtype=np.float32)
        # normalize to [0,1] for fusion
        smin, smax = float(s.min()), float(s.max())
        if smax - smin < 1e-9:
            return np.zeros_like(s)
        return (s - smin) / (smax - smin)
