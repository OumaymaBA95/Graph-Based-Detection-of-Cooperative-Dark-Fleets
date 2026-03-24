# Artifacts Manifest (submission / reproducibility)

This file lists the key outputs referenced throughout `docs/` and `README.md`.

## Baseline evaluation (final)
- **Full-coverage edge list**: `artifacts/edges_2012_2019_full.parquet`
- **Capped edge list (laptop)**: `artifacts/edges_2012_2019_cap5000_even30.parquet`
- **Combined (proximity + social) edges** (same-flag social edges use `scripts/mmsi_mid.py`):
  - Full: `python3 scripts/add_social_edges.py --edges artifacts/edges_2012_2019_full.parquet --out artifacts/edges_full_with_social.parquet --max-social-per-bucket 2000` → **53,829** proximity + **607,018** social (**660,847** total; cap controls sampling when many same-MID vessels share a bucket).
  - Capped: `python3 scripts/add_social_edges.py --edges artifacts/edges_2012_2019_cap5000_even30.parquet --out artifacts/edges_cap5000_with_social.parquet --max-social-per-bucket 2000` → **9,495** proximity + **4,189** social (**13,684** total).
- **Final baseline report (JSON)**: `artifacts/tgcn_time_temporal_nodes_fullcoverage.json`
- **Final baseline summary (CSV)**: `artifacts/tgcn_time_temporal_nodes_fullcoverage.csv`
- **Hyperparameter tuning results**: `artifacts/tgcn_tune_results.csv` (best: emb32, ep5, lr0.001)

## Candidate ranking
- **Ranked candidates (full coverage, top-1000)**: `artifacts/tgcn_candidate_scores_fullcoverage.parquet` (1500-bucket model, ROC AUC 0.78)
- **Ranked candidates (top-500)**: `artifacts/tgcn_candidate_scores_fullcoverage_top500.csv`
- **Tuned + ensemble (capped, top-1000)**: `artifacts/tgcn_candidate_scores_tuned_ensemble.parquet`
- **Score distribution stats**: `artifacts/tgcn_candidate_score_stats.json`
- **Top-500 MMSI frequency**: `artifacts/tgcn_candidate_top500_mmsi_frequency.csv`
- **Top-500 score distribution table**: `artifacts/tgcn_candidate_top500_score_stats.csv`

## Validation (overlap evidence)
- **Top-100 proximity validation (25km ±1 day)**: `artifacts/top100_overlap_summary_daily_full_25km_w1.csv`
- **Top-100 proximity validation (50km ±3 days)**: `artifacts/top100_overlap_summary_daily_full_50km_w3.csv`
- **Top-100 proximity validation (100km ±7 days)**: `artifacts/top100_overlap_summary_daily_full_100km_w7.csv`
- **Top-100 region validation (2° bins)**: `artifacts/top100_overlap_summary_daily_full_region2deg.csv`
- **Close-proximity pairs only**:
  - `artifacts/close_pairs_25km_w1.csv`
  - `artifacts/close_pairs_fullcoverage_25km_w1.csv` (1500-bucket model, 8 pairs pass 25km)
  - `artifacts/close_pairs_50km_w3.csv`
  - `artifacts/close_pairs_100km_w7.csv`
- **Case study plots (original 4 pairs)**: `artifacts/plots/case_study_pairs/`
- **Case study plots (1500-bucket model)**: `artifacts/plots/case_study_pairs_fullcoverage/`
- **8 positive pairs – overlap by month**: `artifacts/eight_pairs_overlap_by_month.csv` (and heatmap `artifacts/plots/eight_pairs_overlap_by_month.png`). Generated with: `python3 scripts/overlap_by_month_8pairs.py --pairs artifacts/close_pairs_fullcoverage_25km_w1.csv --daily-root "data/MMSI daily vessels " --all-files --full-months --distance-km 25 --day-window 1 --out-csv artifacts/eight_pairs_overlap_by_month.csv --out-plot artifacts/plots/eight_pairs_overlap_by_month.png`
- **Monthly overlap time-series (8 pairs)**: per-pair and combined plots in `artifacts/plots/pair_overlap_series/`, generated with: `python3 scripts/plot_pair_overlap_time_series.py --overlap-csv artifacts/eight_pairs_overlap_by_month.csv --out-dir artifacts/plots/pair_overlap_series`
- **Six-vessel cluster summary**: `artifacts/six_vessel_cluster_summary.csv` and scatter plot `artifacts/plots/six_vessel_cluster_scatter.png`, generated with: `python3 scripts/analyze_six_vessel_cluster.py --overlap-csv artifacts/eight_pairs_overlap_by_month.csv --max-vessels 6 --out-summary artifacts/six_vessel_cluster_summary.csv --out-plot artifacts/plots/six_vessel_cluster_scatter.png`

## Flag / gear plots (enrichment CSV)
- **Script:** `scripts/plot_flag_gear_enrichment.py`
- **PNGs:** `artifacts/plots/flag_gear_src_dst_gear_heatmap.png`, `flag_gear_same_mid_bar.png`, `flag_gear_timeline_same_mid.png`

## Flag / gear enrichment (top TGCN pair-rows)
- **Enriched CSV**: `artifacts/cooperative_pairs_with_flag_gear.csv` — `python3 scripts/enrich_pairs_with_flag_gear.py`
- **Summary (Markdown + JSON)**: `artifacts/flag_gear_enrichment_summary.md`, `artifacts/flag_gear_enrichment_summary.json` — `python3 scripts/summarize_flag_gear_enrichment.py`
- **MID helper**: `scripts/mmsi_mid.py` (shared with `scripts/add_social_edges.py`)

## Reports (docs/)
- **Shortlist + validation focus**: `docs/final_shortlist.md`
- **Candidate findings table (score + evidence + optional flag/gear merge)**: `docs/candidate_findings.md` — regenerate: `python3 scripts/make_candidate_findings.py` (optional `--enrichment artifacts/cooperative_pairs_with_flag_gear.csv`)
- **Case studies (validation-focused)**: `docs/candidate_case_studies.md`
- **Thesis draft**: `docs/thesis_draft.md`
- **Defense prep (Q&A)**: `docs/defense_prep.md`
- **20-min speaker script**: `docs/20min_speaker_script.md`
- **5-min presentation (PPTX)**: `docs/5_Minute_Dark_Fleets_Presentation.pptx`
- **20-min presentation (PPTX)**: `docs/20_Minute_Dark_Fleets_Presentation.pptx`

## TGCN figures (full-window combined run, maxb1450)
- **Histograms + per-bucket AUC series:** `artifacts/plots/tgcn_social_maxb1450_ep3_bucket_metrics_hist.png`, `artifacts/plots/tgcn_social_maxb1450_ep3_bucket_auc_timeseries.png` — `python3 scripts/plot_tgcn_bucket_metrics.py --report artifacts/tgcn_social_maxb1450_ep3.json --out-dir artifacts/plots`

## TGCN improvement suite (ablations)
- **Doc**: `docs/tgcn_improvement_suite.md`
- **Driver**: `python3 scripts/run_tgcn_improvement_suite.py` (use PyG env; `--quick` for fast ordering of ideas)
- **Summary**: `artifacts/tgcn_improvement_suite_summary.csv`, per-run `artifacts/tgcn_suite_*.json`

## TGCN re-train (MID-correct `edges_full_with_social.parquet`)
- **Recipe & commands**: `docs/tgcn_retrain_midfix.md`
- **Smoke (capped + social)**: `artifacts/tgcn_smoke_cap5000_with_social_midfix.json`, `artifacts/tgcn_smoke_cap5000_with_social_midfix.csv`
- **Full (train buckets capped at 1500)**: `artifacts/tgcn_full_with_social_midfix_maxb1500.json`, `artifacts/tgcn_full_with_social_midfix_maxb1500.csv` — run `./scripts/run_tgcn_full_social_midfix.sh` (long-running on a laptop)

## Cross-validation
- **Capped graph (5-fold CV)**: `artifacts/tgcn_cv_report.json` (ROC AUC 0.927 ± 0.067)
- **Full-coverage (5-fold CV)**: Run manually: `KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/run_tgcn_time_multiseed.py --edges artifacts/edges_2012_2019_full.parquet --epochs 5 --embedding-dim 32 --lr 0.001 --seeds 1 --use-temporal-node-features --max-train-buckets 1500 --cv-folds 5 --out-report artifacts/tgcn_cv_fullcoverage.json`

## Gear classification (movement-based, anonymized training data)
- **Trained gear classifier (RandomForest)**: `artifacts/gear_classifier.joblib`
- **Gear classifier evaluation report**: `artifacts/gear_classifier_report.txt`
- **Training command** (proof-of-concept, using anonymized AIS training data):
  - `python3 scripts/train_gear_classifier.py --data-root "/Users/momoba/Desktop/Senior Project" --max-rows-per-file 200000 --out-model artifacts/gear_classifier.joblib --out-report artifacts/gear_classifier_report.txt`

