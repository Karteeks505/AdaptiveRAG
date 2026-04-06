from __future__ import annotations

import hashlib
from pathlib import Path

from adaptive_rag.types import Chunk


def _word_chunks(text: str, max_words: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    step = max(1, max_words - overlap)
    i = 0
    while i < len(words):
        piece = words[i : i + max_words]
        chunks.append(" ".join(piece))
        i += step
    return chunks


def chunk_markdown_file(
    path: Path,
    version: str,
    max_words: int,
    overlap: int,
    doc_id: str | None = None,
) -> list[Chunk]:
    text = path.read_text(encoding="utf-8")
    doc_id = doc_id or path.stem
    parts = _word_chunks(text, max_words, overlap)
    out: list[Chunk] = []
    for idx, part in enumerate(parts):
        h = hashlib.sha256(f"{doc_id}:{version}:{idx}".encode()).hexdigest()[:12]
        cid = f"{doc_id}_{version}_c{idx}_{h}"
        out.append(
            Chunk(
                chunk_id=cid,
                doc_id=doc_id,
                version=version,  # type: ignore[arg-type]
                chunk_index=idx,
                text=part,
            )
        )
    return out


def load_corpus_dir(glob_pattern: str, version: str, max_words: int, overlap: int, base: Path) -> list[Chunk]:
    from glob import glob

    chunks: list[Chunk] = []
    for p in sorted(glob(str(base / glob_pattern))):
        chunks.extend(chunk_markdown_file(Path(p), version, max_words, overlap))
    return chunks
