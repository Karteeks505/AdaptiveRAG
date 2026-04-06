from __future__ import annotations

from dataclasses import dataclass

from adaptive_rag.types import Chunk, Query


@dataclass
class MetricBundle:
    scr: float
    shr: float
    vrp: float
    va_racc: float
    cas: float


def _micro_scr(chunks: list[Chunk], authoritative_version: str) -> float:
    if not chunks:
        return 0.0
    bad = sum(1 for c in chunks if c.version != authoritative_version)
    return bad / len(chunks)


def compute_query_metrics(
    q: Query,
    retrieved: list[Chunk],
    authoritative_version: str,
    gold_chunk_ids: list[str],
    answer_text: str | None,
    gold_answer: str,
    use_llm_cas: bool,
    cas_from_llm: bool | None,
) -> MetricBundle:
    """authoritative_version is 'v0' (pre) or 'v1' (post)."""
    scr = _micro_scr(retrieved, authoritative_version)

    gold_set = set(gold_chunk_ids)
    shr = 1.0 if any(c.chunk_id in gold_set for c in retrieved) else 0.0
    vrp = 1.0 if any(c.chunk_id in gold_set for c in retrieved) else 0.0

    # VA-RAcc: answer matches gold keywords (simple local judge)
    va_racc = _answer_match(answer_text or "", gold_answer)

    if use_llm_cas and cas_from_llm is not None:
        cas = 1.0 if cas_from_llm else 0.0
    else:
        # proxy: compliance aligned if VRP hit (retrieval grounded in v1 when v1 authoritative)
        cas = vrp if authoritative_version == "v1" else shr

    return MetricBundle(scr=scr, shr=shr, vrp=vrp, va_racc=va_racc, cas=cas)


def _answer_match(answer: str, gold: str) -> float:
    if not gold.strip():
        return 0.0
    g = set(gold.lower().split())
    a = set(answer.lower().split())
    if not g:
        return 0.0
    overlap = len(g & a)
    return min(1.0, overlap / max(1, int(len(g) * 0.5)))


def aggregate_mean(bundles: list[MetricBundle]) -> dict[str, float]:
    if not bundles:
        return {"scr": 0, "shr": 0, "vrp": 0, "va_racc": 0, "cas": 0}
    n = len(bundles)
    return {
        "scr": sum(b.scr for b in bundles) / n,
        "shr": sum(b.shr for b in bundles) / n,
        "vrp": sum(b.vrp for b in bundles) / n,
        "va_racc": sum(b.va_racc for b in bundles) / n,
        "cas": sum(b.cas for b in bundles) / n,
    }
