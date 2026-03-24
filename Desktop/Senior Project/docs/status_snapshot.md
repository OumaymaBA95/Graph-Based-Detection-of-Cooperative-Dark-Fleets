# Status Snapshot (Feb 17, 2026)

## What we are doing
We are benchmarking link‑prediction baselines on a temporal co‑movement graph to identify the strongest model under **time‑based splits** (the most realistic setting).

## Current best signals
- **Random split (easier)**: Linear GAE baseline leads overall.
- **Time split (harder)**:
  - **Best temporal GNN so far:** TGCN + temporal node features (3 seeds, 5 buckets) — ROC AUC ≈ 0.784, AP ≈ 0.878.
  - Linear GAE + temporal features remains strong (ROC AUC ≈ 0.628, AP ≈ 0.679).
  - GConvGRU/LSTM variants underperform TGCN in current settings.

## Latest runs (from `artifacts/baseline_comparison.md`)
- TGCN + temporal node features: ROC AUC ≈ 0.784, AP ≈ 0.878
- GConvLSTM time‑split: ROC AUC ≈ 0.478, AP ≈ 0.702
- GConvGRU time‑split: ROC AUC ≈ 0.414, AP ≈ 0.664
- TGCN time‑split (no temporal features): ROC AUC ≈ 0.512, AP ≈ 0.711

## What we should do next
1. **Stabilize temporal GNN results** by evaluating all test buckets (no cap).
2. **Add richer node features** (SST + movement) to temporal GNN inputs.
3. If TGCN + features stays best, **freeze it as the “temporal GNN baseline.”**
