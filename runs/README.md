# Run outputs

This directory is gitignored. After `scripts/reproduce.sh` or `python -m adaptive_rag.run_experiment`, each run writes:

- `manifest.json` — config hash, optional git commit, data file SHA-256
- `drift_stale_v0_chunk_ids.json` — v0 chunks flagged stale vs v1
- `summary_pre.json` / `summary_post.json` — aggregate metrics + bootstrap CIs
- `summary_all.json` — combined
- `per_query_<phase>_<system>.jsonl` — one line per query
