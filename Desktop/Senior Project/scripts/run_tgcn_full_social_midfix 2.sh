#!/usr/bin/env bash
# Re-train TGCN on edges_full_with_social.parquet after MID-correct same-flag social edges.
# Requires: conda env with PyTorch + PyG (see README). Run from repo root.
#
# Usage:
#   chmod +x scripts/run_tgcn_full_social_midfix.sh
#   ./scripts/run_tgcn_full_social_midfix.sh
#
# Outputs:
#   artifacts/tgcn_full_with_social_midfix_maxb1500.json
#   artifacts/tgcn_full_with_social_midfix_maxb1500.csv
# Appends to artifacts/experiment_log.csv and experiment_log.json if present.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/scripts${PYTHONPATH:+:$PYTHONPATH}"
export KMP_DUPLICATE_LIB_OK=TRUE

PY="${PYTHON:-python3}"
if [[ -x "$HOME/miniconda3_arm64/envs/pyg/bin/python3" ]]; then
  PY="$HOME/miniconda3_arm64/envs/pyg/bin/python3"
fi

exec "$PY" scripts/run_tgcn_time_multiseed.py \
  --edges artifacts/edges_full_with_social.parquet \
  --epochs 5 \
  --embedding-dim 32 \
  --lr 0.001 \
  --seeds 1 \
  --use-temporal-node-features \
  --max-train-buckets 1500 \
  --out-report artifacts/tgcn_full_with_social_midfix_maxb1500.json \
  --out-csv artifacts/tgcn_full_with_social_midfix_maxb1500.csv \
  --log --log-csv artifacts/experiment_log.csv --log-json artifacts/experiment_log.json
