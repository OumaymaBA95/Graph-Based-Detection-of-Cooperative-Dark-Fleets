# Defense Preparation: Anticipated Questions & Talking Points

## 0. Thirty-second “scope” script (read from the report)

The **blockquote** under §1 of `docs/final_report.md` states: **(1)** 10 km = training edges; 25 km+ = validation. **(2)** Table 1 **combined** row = headline benchmark; other rows = other experiments. **(3)** August 2017 ≠ 2012–2019 benchmark. Memorize this block.

**Ready 30-second summary (copy/paste):** “We built an unsupervised temporal link-prediction pipeline on daily AIS graphs (2012–2019). On the combined proximity + MID-corrected social graph, a laptop run (1450 training buckets, 3 epochs) reaches ROC AUC = 0.72997 and AP = 0.71381 on 876 held-out future days. Eight high-scoring pairs also pass independent 25 km geographic validation; the pipeline outputs ranked candidates for human review only — no legal or enforcement conclusions are claimed.”

---

## 1. Core Concept

**Q: What problem are you solving?**  
A: IUU fishing often involves vessels coordinating at sea (transshipment, shared grounds, rendezvous with "dark" vessels). We use AIS data and a temporal GNN to surface candidate pairs that may be coordinating. We don't have labels—it's unsupervised. We validate high-scoring candidates against raw proximity data.

**Q: Why link prediction?**  
A: Co-occurrence (vessels close on the same day) forms a graph. Link prediction learns which pairs tend to co-occur. Pairs not in the training graph but scoring high may be "missing links"—vessels that coordinate but weren't observed together in the training window.

---

## 2. Methods

**Q: Why a temporal GNN (TGCN) instead of a static model?**  
A: Vessel co-movement is temporal—patterns repeat over time. The TGCN processes graph snapshots sequentially and learns temporal structure. Time-split evaluation (train on past, test on future) is harder than random split; temporal models help.

**Q: What are temporal node features?**  
A: For each vessel in the training period: interactions count, unique partners, last seen days, mean gap between interactions. These capture activity patterns and help the model distinguish active vs. inactive vessels.

**Q: Why rolling-window cross-validation?**  
A: A single time-split can be lucky or unlucky. CV with 5 folds (train ratios 0.5–0.9) gives mean ± std across different cutoffs, making the evaluation more robust.

---

## 3. Results & Validation

**Q: What is your main TGCN result on the combined (proximity + social) graph?**  
A: On `edges_full_with_social.parquet` with correct three-digit MIDs, temporal node features, **1450** training time buckets (3 epochs, seed 1), we get ROC AUC **0.72997** and AP **0.71381** over **876** held-out test buckets. Artifacts: `artifacts/tgcn_social_maxb1450_ep3.json`. Higher train-bucket counts (1480–1500) were **killed** by the OS on our laptop (memory).

**Q: What is the difference between the August 2017 heatmap and that 0.73 number?**  
A: The **August cooperation heatmap** (Figure 4 in `final_report.md`) is a **short August 2017 window** for visualization—metrics there are **not** the headline. The **0.72997 / 0.71381** result is a **full** 2012–2019 time-split on the combined graph (Table §4.1). Per-bucket variability is summarized **in text** in the report; detailed per-bucket numbers are in `artifacts/tgcn_social_maxb1450_ep3.csv`. Say that clearly so listeners don’t mix scopes.

**Q: Why is full-coverage ROC AUC (~0.78) lower than capped (~0.95)?**  
A: Full-coverage uses many more edges and nodes—it's a harder, more realistic setting. The capped graph is smaller and easier; we use it for reproducibility on laptops. Both show the pipeline works.

**Q: How do you validate?**  
A: We check whether high-scoring pairs were actually close in the raw data. Multiple thresholds: 25 km ±1 day (strongest), 50 km ±3 days, 100 km ±7 days. Pairs that pass 25 km are more credible than those that only pass 100 km.

**Q: Why so few pairs pass 25 km?**  
A: We validate the top-100 model-ranked pairs. Most high-scoring pairs are "same fishing grounds" without close encounters. The validation filters those out—that's the point.

**Q: What are the “flag” and “gear” columns on top pairs?**  
A: The huge combined fleet CSV is **grid-aggregated**—it does not list MMSIs. We join **MMSI-daily** tracks to **single-day cell fleet** files and weight gear by **hours** in each cell. “MID” is the first three digits of the MMSI (national block). About **one-third** of our top August 2017 pair-rows share the same MID on both vessels—consistent with same-country coordination. Gear is **descriptive**, not a vessel registry. See `artifacts/flag_gear_enrichment_summary.md` and `docs/final_report.md` §4.3–4.4.

**Q: What does each gear type (*trawlers*, *set_gillnets*, *fixed_gear*, …) mean?**  
A: Those strings come from the fleet file’s **`geartype`** column (hours-weighted attribution per cell). Plain-language definitions are in the report’s **Appendix A** (same text as **`docs/gear_types.md`** in the repo). Remind the committee: it’s a **coarse proxy** from grid traffic, not a formal gear license per MMSI.

---

## 4. Limitations

**Q: What are the main limitations?**  
A: (1) Daily grid-cell resolution—fine-scale transshipment may be missed. (2) No ground truth—we use proximity heuristics, not confirmed behavior. (3) Results depend on 2012–2019 and edge-construction parameters.

**Q: Could you have false positives?**  
A: Yes. High proximity counts suggest coordination but don't prove it. External metadata (flag, gear, ownership) would strengthen interpretation. We surface candidates for investigation, not final judgments.

---

## 5. Future Work

**Q: What would you do next?**  
A: (1) Finer spatial/temporal resolution. (2) External metadata lookup for top pairs. (3) Supervised labels if enforcement provides them. (4) Full-coverage hyperparameter tuning when resources allow.

---

## 6. Technical Details (if asked)

- **Haversine formula** for great-circle distance between vessel positions
- **BCE loss** for link prediction training
- **Dot product** of embeddings for candidate scoring
- **Mac users:** `KMP_DUPLICATE_LIB_OK=TRUE` for OpenMP; `--max-train-buckets 1500` for full-coverage to avoid OOM

---

## Key Numbers to Remember

| Metric | Value |
|--------|-------|
| **Combined graph TGCN (headline)** | ROC AUC **0.72997**, AP **0.71381** (1450 train buckets, 876 test buckets) |
| Full-coverage **proximity-only** ROC AUC | ~0.78 |
| Capped single-split ROC AUC | ~0.95 |
| Capped 5-fold CV ROC AUC | 0.927 ± 0.067 |
| August 2017 case study (short window) | ~0.56 AUC / ~0.67 AP (not comparable to headline) |
| Pairs passing 25 km | 8 (from top-100) |
| Strongest pair (412000690↔412325200) | 102 days within 25 km |
| Data scale | ~2.38B vessel-day records |
