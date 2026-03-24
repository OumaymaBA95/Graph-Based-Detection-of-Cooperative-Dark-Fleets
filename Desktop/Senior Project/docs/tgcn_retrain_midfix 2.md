# TGCN re-train after MID-correct social edges

After rebuilding `artifacts/edges_full_with_social.parquet` with `scripts/mmsi_mid.py`, re-run the temporal GNN so JSON/CSV metrics match the current edge file.

## Quick smoke (laptop, ~20–60 s)

Uses the **capped** graph + social edges; metrics are optimistic vs full coverage but validate the stack.

```bash
cd "/path/to/Senior Project"
export PYTHONPATH=scripts
export KMP_DUPLICATE_LIB_OK=TRUE
python3 scripts/run_tgcn_time_multiseed.py \
  --edges artifacts/edges_cap5000_with_social.parquet \
  --epochs 2 \
  --embedding-dim 32 \
  --lr 0.001 \
  --seeds 1 \
  --use-temporal-node-features \
  --max-train-buckets 300 \
  --out-report artifacts/tgcn_smoke_cap5000_with_social_midfix.json \
  --out-csv artifacts/tgcn_smoke_cap5000_with_social_midfix.csv
```

Example smoke output (one seed): ROC AUC ≈ 0.97, AP ≈ 0.98, 50 test buckets evaluated.

## Full laptop run (long; memory-safe)

Trains on **all** edges in `edges_full_with_social.parquet` but limits **training buckets** to 1500 (same idea as README full-coverage recipe).

```bash
./scripts/run_tgcn_full_social_midfix.sh
```

Or manually:

```bash
export PYTHONPATH=scripts
export KMP_DUPLICATE_LIB_OK=TRUE
python3 scripts/run_tgcn_time_multiseed.py \
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
```

Runtime can be **tens of minutes to hours** on a laptop (large node sets + 1500 train buckets × 5 epochs). Monitor with Activity Monitor; ensure AC power.

## Outputs

| Artifact | Role |
|----------|------|
| `artifacts/tgcn_full_with_social_midfix_maxb1500.json` | Full report (per-bucket metrics, overall AUC/AP) |
| `artifacts/tgcn_full_with_social_midfix_maxb1500.csv` | One-row summary CSV |
| `artifacts/experiment_log.csv` / `experiment_log.json` | If `--log` was passed |

After a successful full run, cite these paths in `docs/final_report.md` instead of older `tgcn_*_with_social.json` files unless you explicitly label them as legacy.

## Increasing training data (`--max-train-buckets`)

The script uses the **first N** time-ordered training buckets after the time split. Larger **N** usually gives the model more signal but uses **more RAM** and time; on a Mac you may see **`zsh: killed`** (OOM) if **N** is too high.

**Approach:** increase in steps and use a **new** `--out-report` / `--out-csv` each time, e.g. **400 → 600 → 800 → 1000 → 1200 → 1500**, stopping when the run completes reliably.

```bash
# Example: 800 train buckets (same hyperparameters as before)
export PYTHONPATH=scripts
export KMP_DUPLICATE_LIB_OK=TRUE
PY="$HOME/miniconda3_arm64/envs/pyg/bin/python3"
"$PY" scripts/run_tgcn_time_multiseed.py \
  --edges artifacts/edges_full_with_social.parquet \
  --epochs 3 --embedding-dim 32 --lr 0.001 --seeds 1 \
  --use-temporal-node-features \
  --max-train-buckets 800 \
  --out-report artifacts/tgcn_social_maxb800_ep3.json \
  --out-csv artifacts/tgcn_social_maxb800_ep3.csv
```

**Improvement suite** with a larger bucket budget:

```bash
"$PY" scripts/run_tgcn_improvement_suite.py --python "$PY" \
  --train-buckets 800 --epochs 3 --seeds 1
```

## Ablations to improve AUC

See **`docs/tgcn_improvement_suite.md`** and run:

`python3 scripts/run_tgcn_improvement_suite.py` (proximity vs social, hard negatives, GConvGRU, vessel-day features, alternate LR/embedding).
