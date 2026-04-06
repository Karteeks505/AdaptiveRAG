#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -r requirements.txt
pip install -q -e .

python3 scripts/build_synthetic_data.py
OUT="${1:-runs/reproduce_$(date +%Y%m%d_%H%M%S)}"
adaptive-rag-run --config configs/local.yaml --out "$OUT"
echo "Wrote results to $OUT"
