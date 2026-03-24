#!/usr/bin/env bash
# Run TGCN on combined (proximity + social) edges with temporal node features.
# Uses arm64 Miniconda pyg env. Run from project root.
set -e
cd "$(dirname "$0")/.."
PYG_PYTHON="${HOME}/miniconda3_arm64/envs/pyg/bin/python3"
if [[ ! -x "$PYG_PYTHON" ]]; then
  echo "Not found: $PYG_PYTHON (install Miniconda arm64 and create pyg env first)"
  exit 1
fi
export PYTHONPATH=scripts
export KMP_DUPLICATE_LIB_OK=TRUE
"$PYG_PYTHON" scripts/run_tgcn_time_multiseed.py \
  --edges artifacts/edges_full_with_social.parquet \
  --epochs 5 \
  --embedding-dim 32 \
  --lr 0.001 \
  --seeds 1 \
  --use-temporal-node-features \
  --out-report artifacts/tgcn_time_temporal_nodes_full_with_social.json \
  "$@"
