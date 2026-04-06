from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    version: Literal["v0", "v1"]
    chunk_index: int
    text: str
    embedding: list[float] | None = None


@dataclass
class Query:
    id: str
    text: str
    category: str
    gold_v1_chunk_ids: list[str]
    gold_v0_chunk_ids: list[str]  # relevant under stable v0 snapshot
    gold_answer_v1: str
    requires_v1_post_amendment: bool = True


@dataclass
class RetrievalResult:
    query_id: str
    chunk_ids: list[str]
    chunk_versions: list[str]
    scores: list[float] = field(default_factory=list)
