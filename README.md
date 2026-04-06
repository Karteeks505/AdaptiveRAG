# Adaptive RAG — experimental harness (local-first)

This repository implements the evaluation protocol described in `Adaptive_RAG_Paper_Rewritten.docx`: paired **v0/v1** corpora, static vs **v1-indexed adaptive** retrieval, and metrics (**SCR, SHR, VRP, VA-RAcc, CAS**) with bootstrap confidence intervals.

## Quick start

```bash
chmod +x scripts/reproduce.sh
./scripts/reproduce.sh
```

Or manually:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python scripts/build_synthetic_data.py
PYTHONPATH=src python -m adaptive_rag.run_experiment --config configs/local.yaml --out runs/latest
```

Outputs appear under `runs/<name>/` (see `runs/README.md`).

## Configuration

- **`configs/local.yaml`** — default **hash embeddings** (no GPU, no downloads), `faiss` + BM25 hybrid, optional Ollama for LLM-judged CAS (`--use-llm-cas` if Ollama is running).
- **`configs/openai.yaml`** — optional; set `OPENAI_API_KEY` and `pip install -e ".[openai]"` for OpenAI embeddings / GPT-4o-mini.

## Systems

| Key | Description |
|-----|-------------|
| `faiss_static` | Dense-only FAISS on frozen v0 index |
| `hybrid_bm25_dense` | RRF fusion of BM25 + dense (Azure-hybrid analogue) |
| `mma_text_proxy` | Hybrid + dense re-rank (text-only multi-step proxy) |
| `adaptive` | Pre: same as hybrid on v0; **Post: v1 index** (fresh retrieval) |

## Syncing measured numbers into the Word paper

After a run:

```bash
PYTHONPATH=src python scripts/sync_paper_results.py --run runs/latest --doc Adaptive_RAG_Paper_Rewritten.docx
```

## Repository layout

- `src/adaptive_rag/` — library + CLI
- `data/` — synthetic corpus + `queries.jsonl` (regenerate via `scripts/build_synthetic_data.py`)
- `configs/` — YAML profiles
- `paper_assets/` — figure PNG generator (optional for the manuscript)
