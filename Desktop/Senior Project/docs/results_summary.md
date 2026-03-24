# Results Summary (Feb 18, 2026)

## Executive summary
The final full‑coverage temporal GNN baseline (TGCN + temporal node features) delivers the strongest time‑split performance to date and provides a ranked candidate list suitable for downstream investigation. The project is now ready to shift from modeling to reporting and case‑study validation.

## Final baseline (time‑split)
- **Model:** TGCN + temporal node features
- **Evaluation:** full‑coverage edge list (2012–2019), 876 time buckets
- **ROC AUC:** ≈ 0.679
- **Average Precision:** ≈ 0.695

## Full-coverage with max-train-buckets (Mac-friendly)
- **Config:** max-train-buckets=1500, embedding_dim=32, epochs=5, lr=0.001
- **ROC AUC:** ≈ 0.78 (improved over original full-coverage)
- **Candidates:** `artifacts/tgcn_candidate_scores_fullcoverage.parquet` (top-1000)
- **Validation:** 8 pairs pass 25km ±1 day (from top-100, 100 files/year)
- **Strongest new pair:** 412422375 ↔ 412428225 (2 days within 25km)

## Tuned config (capped graph)
- **Best config:** embedding_dim=32, epochs=5, lr=0.001 (from `artifacts/tgcn_tune_results.csv`)
- **ROC AUC:** ≈ 0.995 on capped graph (single split); ≈ 0.953 (3 seeds)
- **Use:** For laptop runs and ensemble scoring; apply to full-coverage when resources allow.

## Rolling-window cross-validation (capped graph)
- **Config:** 5 folds, train ratios 0.5–0.9, 3 seeds per fold
- **ROC AUC:** 0.927 ± 0.067 across folds
- **Average Precision:** 0.947 ± 0.048 across folds
- **Report:** `artifacts/tgcn_cv_report.json`
- **Full-coverage CV:** Run with `--max-train-buckets 1500 --cv-folds 5` to get mean ± std on the main benchmark (requires ~48 GB RAM).

## Data split (time‑based)
- Sort all `time_bucket` values chronologically.
- Use the last `test_ratio` fraction of buckets as **test** and the earlier buckets as **train**.
- Deduplicate undirected edges and drop any test edges that already appear in train.

## Rolling-window cross-validation (optional)
- Use `--cv-folds N` (N ≥ 2) for rolling-window CV instead of a single split.
- Each fold uses an expanding train window (train_ratio from 0.5 to ~0.9) and tests on the immediate future.
- Report mean ± std ROC AUC and AP across folds for more robust evaluation.
- Example: `--cv-folds 5 --cv-min-train-ratio 0.5` yields 5 folds with train ratios 0.5, 0.6, 0.7, 0.8, 0.9.

This full‑coverage run provides the most representative baseline to date. Earlier capped runs were higher (ROC AUC ≈ 0.883 / AP ≈ 0.921) but used fewer edges, so we treat the full‑coverage result as the final benchmark.

## Key findings
- Time‑split baselines are substantially harder than random splits; temporal features and temporal GNNs help.
- The strongest temporal GNN benefits from temporal interaction features used as node inputs.

## Case studies (Top‑5 pairs)
See `docs/candidate_case_studies.md` for the top‑5 candidate pairs with full‑coverage overlap analysis and track plots.

## Candidate outputs
- Full‑coverage ranked candidates (top-1000): `artifacts/tgcn_candidate_scores_fullcoverage.parquet`
- Full‑coverage ranked candidates (top‑500): `artifacts/tgcn_candidate_scores_fullcoverage_top500.csv`
- Shortlist view (top‑10 + validation notes): `docs/final_shortlist.md`

## Recommended next steps
1. Freeze the baseline and use it for downstream ranking.
2. Validate top candidates with external vessel metadata where available.
3. Consider additional case studies if higher‑confidence pairs emerge.

## Final recommendation
Treat the full‑coverage TGCN + temporal node features run as the final benchmark and focus on interpreting ranked candidates. Use `artifacts/tgcn_candidate_scores_fullcoverage_top500.csv` as the primary ranked output (and `docs/final_shortlist.md` for the top‑10 view + validation evidence).

## Limitations
- Full‑coverage candidate scoring was run with a reduced sampling rate to avoid memory limits.
- Overlap analysis relies on grid‑cell daily positions, which can blur fine‑scale proximity.

## Top‑500 score interpretation
The top‑500 scores are tightly clustered (mean ≈ 14.46, std ≈ 0.42), suggesting the model produces a concentrated band of high‑confidence candidates rather than a long tail. Use the top‑50/100 for manual review, and the top‑500 for broader pattern analysis.

## Final outputs checklist
- **Baseline metrics:** `artifacts/tgcn_time_temporal_nodes_fullcoverage.json` and `.csv`
- **Ranked candidates (full coverage, top‑500):** `artifacts/tgcn_candidate_scores_fullcoverage_top500.csv`
- **Score stats:** `artifacts/tgcn_candidate_score_stats.json`
- **Top-500 MMSI frequency:** `artifacts/tgcn_candidate_top500_mmsi_frequency.csv`
- **Top-500 score distribution:** `artifacts/tgcn_candidate_top500_score_stats.csv`
- **Final shortlist:** `docs/final_shortlist.md`
- **Case studies:** `docs/candidate_case_studies.md`
- **Overlap stats (full coverage):** `artifacts/candidate_pair_overlap_summary_daily_full_ranked.csv`
- **Plots:** `artifacts/plots/candidate_pairs_daily_full/`
