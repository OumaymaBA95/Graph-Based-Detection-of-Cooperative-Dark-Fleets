# TGCN improvement / ablation suite

`scripts/run_tgcn_improvement_suite.py` runs a **fixed menu** of experiments aimed at improving or understanding **ROC AUC** and **AP**:

| # | Slug | Edges | Extra |
|---|------|-------|--------|
| 01 | `01_proximity_temporal` | `artifacts/edges_2012_2019_full.parquet` | Proximity **only** (no social edges) |
| 02 | `02_social_temporal` | `artifacts/edges_full_with_social.parquet` | Proximity + **social** (same-flag MID) |
| 03 | `03_social_hardneg` | social | `--use-hard-negatives` |
| 04 | `04_proximity_hardneg` | proximity-only | `--use-hard-negatives` |
| 05 | `05_social_gconvgru` | social | `--model gconvgru` |
| 06 | `06_social_vesselday_temporal` | social | `--use-vessel-day-features` + SST/speed/lat/lon columns |
| 07 | `07_social_tgcn_emb48_lr5e4` | social | `embedding_dim=48`, `lr=5e-4` |

Each run writes:

- `artifacts/tgcn_suite_<slug>.json`
- `artifacts/tgcn_suite_<slug>.csv`

The driver appends rows to:

- **`artifacts/tgcn_improvement_suite_summary.csv`** (one row per run)

## Commands

**Full protocol** (matches your successful lighter run: 400 train buckets, 3 epochs — **can take hours** and may OOM; use `pyg` Python):

```bash
cd "/Users/momoba/Desktop/Senior Project"
export PYTHONPATH=scripts
export KMP_DUPLICATE_LIB_OK=TRUE
export TG_PYTHON="$HOME/miniconda3_arm64/envs/pyg/bin/python3"

"$TG_PYTHON" scripts/run_tgcn_improvement_suite.py --python "$TG_PYTHON"
```

**Quick smoke** (small train buckets, capped test buckets — **not comparable** to thesis numbers, but good for ordering which branch helps):

```bash
"$TG_PYTHON" scripts/run_tgcn_improvement_suite.py --python "$TG_PYTHON" --quick
```

**Subset:**

```bash
"$TG_PYTHON" scripts/run_tgcn_improvement_suite.py --python "$TG_PYTHON" --only 01,02,03
```

**Dry-run:**

```bash
python3 scripts/run_tgcn_improvement_suite.py --dry-run
```

## Interpreting results

- If **01 (proximity)** beats **02 (social)**, dense same-MID social edges may be **hurting** link prediction; consider capping social edges or using proximity-only for ranking.
- If **hard negatives (03/04)** improve AUC vs 01/02, training negatives were too easy before.
- **06** needs `data/features_by_year/*/vessel_day_features.parquet` (present in this project).
- Compare **ROC AUC** and **AP** on the **same** `--train-buckets` / `--epochs` / `--max-test-buckets` settings only.

## Logs

A full suite started in the background may log to **`artifacts/tgcn_improvement_suite.log`**.
