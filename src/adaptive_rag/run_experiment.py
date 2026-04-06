from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from adaptive_rag.chunking import load_corpus_dir
from adaptive_rag.config import data_root, load_config
from adaptive_rag.drift import compute_stale_flags, pair_chunks
from adaptive_rag.embeddings import build_embedding_backend
from adaptive_rag.indexing_bm25 import BM25Index
from adaptive_rag.indexing_faiss import FaissIndex
from adaptive_rag.llm import ollama_judge_compliance
from adaptive_rag.manifest import write_manifest
from adaptive_rag.metrics import MetricBundle, aggregate_mean, compute_query_metrics
from adaptive_rag.retrieval.hybrid import DenseOnlyRetriever, HybridRetriever, MMARetriever
from adaptive_rag.stats import bootstrap_ci_proportions, clopper_pearson_interval
from adaptive_rag.types import Chunk, Query


def resolve_gold_ids(chunks: list[Chunk], doc_id: str, idx: int) -> list[str]:
    for c in chunks:
        if c.doc_id == doc_id and c.chunk_index == idx:
            return [c.chunk_id]
    return []


def load_queries_flexible(path: Path, chunks_v0: list[Chunk], chunks_v1: list[Chunk]) -> list[Query]:
    """Supports gold_v1_chunk_ids or gold_doc + gold_chunk_index fields."""
    rows: list[Query] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            gv1 = list(r.get("gold_v1_chunk_ids") or [])
            gv0 = list(r.get("gold_v0_chunk_ids") or [])
            if not gv1 and r.get("gold_doc") is not None and r.get("gold_chunk_index") is not None:
                doc = r["gold_doc"]
                idx = int(r["gold_chunk_index"])
                gv1 = resolve_gold_ids(chunks_v1, doc, idx)
                gv0 = resolve_gold_ids(chunks_v0, doc, idx)
            rows.append(
                Query(
                    id=r["id"],
                    text=r["text"],
                    category=r["category"],
                    gold_v1_chunk_ids=gv1,
                    gold_v0_chunk_ids=gv0,
                    gold_answer_v1=r.get("gold_answer_v1", ""),
                    requires_v1_post_amendment=r.get("requires_v1_post_amendment", True),
                )
            )
    return rows


def embed_chunks(backend, chunks: list[Chunk]) -> dict[str, np.ndarray]:
    texts = [c.text for c in chunks]
    mat = backend.embed(texts)
    out: dict[str, np.ndarray] = {}
    for i, c in enumerate(chunks):
        out[c.chunk_id] = mat[i].astype(np.float32)
    return out


def build_stack(
    chunks: list[Chunk],
    emb_map: dict[str, np.ndarray],
    cfg: dict[str, Any],
) -> tuple[FaissIndex, BM25Index, HybridRetriever, DenseOnlyRetriever, MMARetriever]:
    vectors = np.stack([emb_map[c.chunk_id] for c in chunks], axis=0)
    faiss = FaissIndex(chunks, vectors)
    bm25 = BM25Index(chunks)
    hybrid = HybridRetriever(chunks, faiss, bm25, cfg["retrieval"]["hybrid_rrf_k"])
    dense = DenseOnlyRetriever(faiss)
    mma = MMARetriever(hybrid, emb_map)
    return faiss, bm25, hybrid, dense, mma


def answer_from_chunks(chunks: list[Chunk]) -> str:
    if not chunks:
        return ""
    return " ".join(c.text[:400] for c in chunks[:2])


def run_system(
    name: str,
    retriever,
    queries: list[Query],
    emb_backend,
    phase: str,
    cfg: dict[str, Any],
    use_llm_cas: bool,
) -> tuple[list[MetricBundle], list[dict[str, Any]]]:
    rows_out: list[dict[str, Any]] = []
    bundles: list[MetricBundle] = []
    llm_cfg = cfg.get("llm", {})
    base_url = llm_cfg.get("base_url") or ""
    model = llm_cfg.get("model", "llama3.2:3b")
    temp = float(llm_cfg.get("temperature", 0.1))

    for q in queries:
        qv = emb_backend.embed([q.text])[0].astype(np.float32)
        ch, _sc = retriever.retrieve(q.text, qv, cfg["retrieval"]["top_k"])
        auth = "v0" if phase == "pre" else "v1"
        gold_ids = q.gold_v0_chunk_ids if phase == "pre" else q.gold_v1_chunk_ids
        ans = answer_from_chunks(ch)
        cas_llm = None
        if use_llm_cas and base_url and phase == "post" and auth == "v1":
            try:
                cas_llm = ollama_judge_compliance(base_url, model, q.text, ans, q.gold_answer_v1, temp)
            except Exception:
                cas_llm = None

        m = compute_query_metrics(
            q,
            ch,
            auth,
            gold_ids,
            ans,
            q.gold_answer_v1,
            use_llm_cas and cas_llm is not None,
            cas_llm,
        )
        bundles.append(m)
        rows_out.append(
            {
                "system": name,
                "phase": phase,
                "query_id": q.id,
                "retrieved_chunk_ids": [c.chunk_id for c in ch],
                "versions": [c.version for c in ch],
                "scr": m.scr,
                "shr": m.shr,
                "vrp": m.vrp,
                "va_racc": m.va_racc,
                "cas": m.cas,
            }
        )
    return bundles, rows_out


def main() -> None:
    ap = argparse.ArgumentParser(description="Adaptive RAG evaluation harness")
    ap.add_argument("--config", default="configs/local.yaml", help="Path to YAML config")
    ap.add_argument("--phase", choices=["pre", "post", "both"], default="both")
    ap.add_argument("--max-queries", type=int, default=0, help="Limit queries (0=all)")
    ap.add_argument("--out", default="runs/latest", help="Output directory")
    ap.add_argument("--use-llm-cas", action="store_true", help="Use Ollama LLM for CAS when available")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    root = data_root(cfg)

    chunks_v0 = load_corpus_dir(
        cfg["paths"]["corpus_v0_glob"],
        "v0",
        cfg["chunking"]["max_words"],
        cfg["chunking"]["overlap_words"],
        root,
    )
    chunks_v1 = load_corpus_dir(
        cfg["paths"]["corpus_v1_glob"],
        "v1",
        cfg["chunking"]["max_words"],
        cfg["chunking"]["overlap_words"],
        root,
    )

    qpath = root / cfg["paths"]["queries"]
    queries = load_queries_flexible(qpath, chunks_v0, chunks_v1)
    if args.max_queries:
        queries = queries[: args.max_queries]

    run_dir = Path(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)

    emb = build_embedding_backend(cfg)
    emb_v0 = embed_chunks(emb, chunks_v0)
    emb_v1 = embed_chunks(emb, chunks_v1)

    pairs = pair_chunks(chunks_v0, chunks_v1)
    _stale_v0 = compute_stale_flags(pairs, emb_v0, emb_v1, float(cfg["drift"]["cosine_threshold"]))
    (run_dir / "drift_stale_v0_chunk_ids.json").write_text(
        json.dumps(sorted(_stale_v0), indent=2),
        encoding="utf-8",
    )

    _, _, hybrid_v0, dense_v0, mma_v0 = build_stack(chunks_v0, emb_v0, cfg)
    _, _, hybrid_v1, dense_v1, mma_v1 = build_stack(chunks_v1, emb_v1, cfg)

    data_files = [cfg_path, qpath]
    for p in sorted(root.glob("corpus/**/*.md")):
        data_files.append(p)
    write_manifest(run_dir, cfg_path, cfg, data_files)

    phases = ["pre", "post"] if args.phase == "both" else [args.phase]
    all_results: dict[str, Any] = {}

    for phase in phases:
        systems: dict[str, Any] = {}
        if phase == "pre":
            systems = {
                "faiss_static": dense_v0,
                "hybrid_bm25_dense": hybrid_v0,
                "mma_text_proxy": mma_v0,
                "adaptive": hybrid_v0,
            }
            auth = "v0"
        else:
            systems = {
                "faiss_static": dense_v0,
                "hybrid_bm25_dense": hybrid_v0,
                "mma_text_proxy": mma_v0,
                "adaptive": hybrid_v1,
            }

        phase_summary: dict[str, Any] = {}
        for sys_name, retr in systems.items():
            bundles, rows = run_system(
                sys_name,
                retr,
                queries,
                emb,
                phase,
                cfg,
                args.use_llm_cas,
            )
            agg = aggregate_mean(bundles)
            # bootstrap on query-level metric values
            seed = int(cfg.get("bootstrap", {}).get("random_seed", 42))
            n_boot = int(cfg.get("bootstrap", {}).get("n_resamples", 10000))
            cis: dict[str, tuple[float, float]] = {}
            for metric in ["scr", "shr", "vrp", "va_racc", "cas"]:
                vals = [getattr(b, metric) for b in bundles]
                cis[metric] = bootstrap_ci_proportions(vals, n_boot, seed)

            successes_vrp = sum(1 for b in bundles if b.vrp >= 0.999)
            cp = clopper_pearson_interval(successes_vrp, len(bundles))

            phase_summary[sys_name] = {
                "aggregate": agg,
                "bootstrap_ci": {k: {"low": v[0], "high": v[1]} for k, v in cis.items()},
                "clopper_pearson_vrp_ge_1": {"low": cp[0], "high": cp[1], "successes": successes_vrp, "n": len(bundles)},
            }
            (run_dir / f"per_query_{phase}_{sys_name}.jsonl").write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n",
                encoding="utf-8",
            )

        all_results[phase] = phase_summary
        (run_dir / f"summary_{phase}.json").write_text(json.dumps(phase_summary, indent=2), encoding="utf-8")

    (run_dir / "summary_all.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
