from __future__ import annotations

import numpy as np

from adaptive_rag.types import Chunk


def pair_chunks(v0: list[Chunk], v1: list[Chunk]) -> dict[tuple[str, int], tuple[Chunk, Chunk]]:
    m: dict[tuple[str, int], tuple[Chunk, Chunk]] = {}
    v0_map = {(c.doc_id, c.chunk_index): c for c in v0}
    for c1 in v1:
        key = (c1.doc_id, c1.chunk_index)
        if key in v0_map:
            m[key] = (v0_map[key], c1)
    return m


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))


def compute_stale_flags(
    pairs: dict[tuple[str, int], tuple[Chunk, Chunk]],
    emb_v0: dict[str, np.ndarray],
    emb_v1: dict[str, np.ndarray],
    threshold: float,
) -> set[str]:
    """Return v0 chunk_ids that are 'stale' (v1 pair differs beyond threshold)."""
    stale: set[str] = set()
    for _k, (c0, c1) in pairs.items():
        e0 = emb_v0.get(c0.chunk_id)
        e1 = emb_v1.get(c1.chunk_id)
        if e0 is None or e1 is None:
            continue
        sim = cosine_sim(e0, e1)
        if sim < threshold or c0.text.strip() != c1.text.strip():
            stale.add(c0.chunk_id)
    return stale
