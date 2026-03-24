# Graph-Based Detection of Cooperative Dark Fleets: A Temporal Link Prediction Approach

**Senior Project Draft**

---

## Abstract

Illegal, unreported, and unregulated (IUU) fishing often involves vessels that coordinate at sea—transshipment, shared fishing grounds, or rendezvous with vessels that turn off their AIS. We present an unsupervised pipeline to surface candidate vessel pairs that may be coordinating, using publicly available AIS data and a Temporal Graph Convolutional Network (TGCN). We build a temporal co-movement graph where edges connect vessels that were close on the same day, train the TGCN to predict links, and validate high-scoring candidates against raw proximity data. On a full-coverage edge list (2012–2019), the model achieves ROC AUC ~0.78 (single split) with temporal node features. Rolling-window cross-validation on the capped graph yields ROC AUC 0.927 ± 0.067 across 5 folds. We identify 8 pairs that pass a 25 km ±1 day proximity threshold, with the strongest candidate (412422375 ↔ 412428225) showing 2 days within 25 km. The pipeline is reproducible and suitable for downstream investigation by enforcement and research.

**Keywords:** AIS, IUU fishing, temporal graph neural networks, link prediction, unsupervised learning, vessel co-movement

---

## 1. Introduction

### 1.1 Motivation

Marine fisheries face significant challenges from illegal, unreported, and unregulated (IUU) fishing. A common pattern involves vessels that coordinate at sea: transshipment (transferring catch between vessels), shared fishing grounds, or rendezvous with "dark" vessels that turn off their Automatic Identification System (AIS). Detecting these patterns from publicly available AIS data could support enforcement and research.

### 1.2 Goal

We aim to find pairs of vessels that may be coordinating or meeting at sea. **We do not have labels for "cooperative" pairs**—this is unsupervised. We use a link-prediction approach: the model learns which vessel pairs tend to co-occur, then we validate high-scoring candidates against raw proximity data. We **surface candidates** with proximity evidence; we do not claim to "detect" or "prove" cooperation.

### 1.3 Contributions

- A reproducible pipeline from AIS data to ranked candidate pairs
- Temporal GNN (TGCN) with temporal node features for link prediction
- Multi-threshold proximity validation (25 km, 50 km, 100 km) to separate "same region" from "actually close"
- Rolling-window cross-validation for robust evaluation
- Case studies with track plots and interpretation

---

## 2. Related Work

- **AIS and IUU:** AIS data has been used for vessel behavior analysis, transshipment detection, and fleet monitoring. Prior work often relies on rule-based or supervised methods.
- **Temporal GNNs:** TGCN, GConvGRU, and related models combine graph convolutions with recurrent units for temporal link prediction.
- **Link prediction:** Unsupervised link prediction on temporal graphs generalizes to future time periods when using time-based splits.

---

## 3. Methods

### 3.1 Data

- **Vessel positions:** Daily AIS data (latitude, longitude) from 2012–2019. Each vessel has a 9-digit MMSI.
- **Scale:** ~2.38 billion vessel-day records; ~97.5% with sea-surface temperature (SST) from gridded products.

### 3.2 Graph Construction

- **Nodes:** Unique MMSIs (hundreds of thousands of vessels).
- **Edges:** Undirected edge between two vessels if they were within a distance threshold (e.g., 10 km) on the same day.
- **Temporal structure:** Edges grouped into daily time buckets. Full-coverage: 876 buckets over 2012–2019.

### 3.3 Model

- **TGCN:** Temporal Graph Convolutional Network—graph convolution + GRU at each time step.
- **Node features:** Temporal interaction features (interactions count, unique partners, last seen days, mean gap days) plus degree.
- **Task:** Link prediction. Train on positive edges vs. random negatives; score candidates by embedding dot product.

### 3.4 Evaluation

- **Time-split:** First 70% of buckets for training, last 30% for testing.
- **Rolling-window CV:** 5 folds, train ratios 0.5–0.9, mean ± std across folds.
- **Metrics:** ROC AUC, Average Precision.

---

## 4. Results

### 4.1 Baseline Performance

| Setting | ROC AUC | AP |
|---------|---------|-----|
| Full coverage (1500 train buckets) | ~0.78 | ~0.80 |
| Capped graph (single split) | ~0.95 | ~0.96 |
| Capped graph (5-fold CV) | 0.927 ± 0.067 | 0.947 ± 0.048 |

Time-split evaluation is substantially harder than random split. Temporal features and temporal GNNs improve over static baselines.

### 4.2 Candidate Ranking and Validation

- Top-1000 candidates from full-coverage model.
- 8 pairs pass 25 km ±1 day proximity threshold (from top-100, 100 files/year).
- Strongest new pair: 412422375 ↔ 412428225 (2 days within 25 km, mean distance 15.3 km).

### 4.3 Case Studies

See `docs/candidate_case_studies.md` for the top validated pairs with track plots and interpretation. Pair 412000690 ↔ 412325200 has 102 days within 25 km—the strongest validation signal.

![Pair 412000690 and 412325200](../artifacts/plots/case_study_pairs/pair_412000690_412325200.png)

---

## 5. Limitations

- **Grid-cell resolution:** Daily positions blur fine-scale transshipment.
- **No ground truth:** Validation is proximity-based, not confirmed behavior.
- **Temporal coverage:** Results depend on 2012–2019 and edge-construction parameters.

---

## 6. Conclusion

The TGCN + temporal features model surfaces candidate vessel pairs with observable proximity evidence. Multi-threshold validation separates "same region" overlap from "actually close" encounters. The pipeline is reproducible; full-coverage requires substantial RAM; the capped graph runs on typical laptops.

---

## 7. Future Work

- Finer spatial/temporal resolution
- External metadata (flag, gear, ownership)
- Supervised labels if available
- Full-coverage hyperparameter tuning

---

## References

[To be filled: AIS/IUU papers, TGCN/temporal GNN papers, link prediction papers]

---

## Appendix: Reproducibility

Key commands and artifact paths are in `README.md` and `docs/full_presentation.md`. Artifacts: `artifacts/tgcn_time_temporal_nodes_fullcoverage.json`, `artifacts/tgcn_candidate_scores_fullcoverage.parquet`, `artifacts/close_pairs_fullcoverage_25km_w1.csv`.
