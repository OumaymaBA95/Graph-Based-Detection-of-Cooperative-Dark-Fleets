---
# Table of contents: enable when knitting to Word/HTML/PDF (RStudio: Knit).
output:
  word_document:
    toc: true
    toc_depth: 3
  html_document:
    toc: true
    toc_float: true
    toc_depth: 3
  pdf_document:
    toc: true
    toc_depth: 3
---
## Graph-Based Screening for Potentially Cooperative Vessel Pairs

**Subtitle:** A temporal link prediction approach using AIS (2012–2019).

### Abstract

Illegal, unreported, and unregulated (IUU) fishing motivates monitoring **coordinated** vessel behavior. This project builds an **unsupervised** pipeline: **daily** graphs from AIS (2012–2019), a **Temporal Graph Convolutional Network (TGCN)** for **temporal link prediction**, and **independent geographic checks** on raw tracks to prioritize high-scoring pairs. **Training edges** use a **10 km** same-day proximity rule; **stricter** distance screens (e.g., **25 km**) are **validation-only**. There are **no labels** for “cooperative” vessels—the model learns from **graph structure and time**, and outputs **candidates for review**, not ground truth.

**Main quantitative result (Table 1):** On the **combined** graph (proximity edges plus same–MMSI-prefix “social” edges, built with a corrected three-digit MID rule), a reproducible laptop run (**1450** training time buckets, 3 epochs, seed 1) achieves **ROC AUC = 0.72997** and **AP = 0.71381** on **876** held-out daily buckets (`artifacts/tgcn_social_maxb1450_ep3.json`). A **proximity-only** full graph reaches higher scores (~**0.78** AUC) under similar memory limits. A separate **short August 2017** window is used only for an **interpretability figure**; its metrics are **not** comparable to the 2012–2019 benchmark. **Eight** candidate pairs pass a strict **25 km, ±1 day** proximity filter. We do **not** claim legal findings or confirmed “dark fleet” membership.

**Keywords:** AIS, IUU fishing, temporal graph neural networks, link prediction, vessel co-movement

---

### 1. Introduction

**Problem.** Fisheries enforcement and research need ways to highlight vessels that may act together—transshipment, shared fishing grounds, or periods with missing AIS (“dark” behavior). Automatic vessel identification (AIS) provides large-scale movement data, but **cooperation is rarely labeled** in the data we use.

**What this work does.** We treat the problem as **temporal link prediction** on daily graphs: if the model assigns a high score to a vessel pair in a future time bucket, that pair is a **candidate** for analysts. We then **validate** candidates with simple proximity rules (distance between daily mean positions) and **case studies** (tracks, monthly overlap).

**What this work does not do.** We do not prove coordination or IUU activity; we provide a **ranked list plus evidence** that human review can interpret.

**Contributions (short).**

1. Reproducible pipeline: AIS → daily graphs → TGCN with temporal node features → time-based train/test split.  
2. **MID-correct** combination of proximity and same-prefix social edges (`scripts/mmsi_mid.py`, `scripts/add_social_edges.py`).  
3. Reported **full-coverage** metrics under realistic **laptop memory** limits (train buckets capped where needed).  
4. Validation and interpretation: proximity filters, eight-pair overlap plots, and an August 2017 interpretability figure with **country and gear** on the vertical axis (from cell-level fleet joins where available).

**Roadmap.** §3 summarizes the **link-prediction setup** (full notation in **Appendix C**), then **data**, **graph rules** (10 km proximity + MID social edges), **training objective**, and **post-hoc** validation distances. §4 gives quantitative results (**Table 1**), geographic and interpretability figures (**Tables 2–3**, Figures 1–4), and **Appendix A** for gear definitions. §5–6 summarize limitations and conclusions; **Appendix B** lists reproducibility commands.

> **For readers and committees (three sentences).** (1) **Training** graphs use **10 km** same-day proximity edges (plus optional MID “social” edges); **25 km** and wider checks are **post-hoc validation**, not class labels. (2) The headline **0.72997 / 0.71381** scores refer to the **combined** full-coverage graph in **Table 1**; **proximity-only** and **capped** rows are **different experiments**—see §3.6. (3) **August 2017** figures illustrate behavior in a **short window**; do not mix their metrics with the 2012–2019 benchmark.

---

### 2. Background and related work

**Fisheries monitoring and AIS.** Illegal, unreported, and unregulated (IUU) fishing and at-sea coordination are active research areas. Large-scale studies use **Automatic Identification System (AIS)** broadcasts to map fishing effort and vessel encounters (e.g., global fisheries footprint and transshipment patterns—see References). AIS does not, by itself, label “cooperative” or illicit behavior; analysts typically combine movement data with **proximity**, **registry**, or **enforcement** context. That gap motivates **unsupervised** tools that rank pairs for review.

**AIS-based fisheries and encounter analytics.** Beyond footprint mapping, vessel-level AIS supports **encounter** and **behavior** analysis (e.g., transshipment and loitering—Miller et al., 2018; monitoring-system assessments—Park et al., 2020). That line of work motivates **movement-based screening** but often relies on **rules** or **supervised** targets when labels exist. Here we stay **unsupervised** at the learning stage and reserve geography for **post-hoc** validation.

**Graph learning and link prediction.** Co-presence is naturally a **graph**: vessels are nodes; edges encode same-day proximity (and optional same-registry structure). **Graph convolutional** encoders (Kipf & Welling, 2017) and **inductive** methods on large graphs (Hamilton et al., 2017) learn vector representations of nodes from topology and attributes. **Link prediction** scores missing or future links (Lü & Zhou, 2011); **temporal** graph models (e.g., spatio-temporal convolutions—Yu et al., 2018; recurrent or memory-based temporal graph networks—Rossi et al., 2020) apply when \(G_t\) **varies** across \(t\). Our implementation uses a **TGCN**-style recurrent update over daily snapshots with **inner-product** scores for pairs (see §3.4).

**Evaluation philosophy.** We use a **chronological train/test split** over daily buckets: train on earlier days, evaluate on **later** days. ROC AUC and AP therefore measure **forecasting**-style generalization to future co-occurrence, not i.i.d. mixing of time points (which can inflate scores when temporal drift is ignored).

---

### 3. Methods

#### 3.1 Problem formulation

Each **calendar day** (bucket) \(t\) defines a snapshot **undirected** graph \(G_t = (V, E_t)\): **nodes** \(V\) are vessel **MMSIs**; **edges** \(E_t\) follow the **construction rules** in §3.3—primarily **same-day** positions within **10 km** (Haversine), plus optional **same three-digit MID** social edges. The learning task is **temporal link prediction**: a **TGCN** produces node embeddings so that **inner-product scores** between pairs rank **observed** edges above **sampled non-edges** under **binary cross-entropy with logits**, with **no** “cooperative” supervision—only structure and time (implementation: `scripts/run_tgcn_time_multiseed.py` and related; details §3.4). **Train and test** partition **time buckets** chronologically (**~70% / ~30%**); **ROC AUC** and **AP** measure ranking quality on **held-out future** buckets.

**Appendix C** lists **symbols**, restates the setup for readers who prefer a **notation table**, and separates **training-edge distance (10 km)** from **validation-only** distances.

#### 3.2 Data and time buckets

We use AIS-derived records spanning **2012–2019**, aggregated to **daily** resolution (~2.38B vessel-day records; sea-surface temperature merged where available). The timeline is divided into **daily buckets**; **876** buckets enter the main edge list used for evaluation.

#### 3.3 Graph construction

- **Nodes** = MMSI identifiers (vessels).  
- **Proximity edges:** same **calendar day** (daily time floor, typically **1D**), pairwise **great-circle (Haversine)** distance at most **10 km** between daily position summaries. The public full-coverage edge list `edges_2012_2019_full.parquet` is built with `scripts/build_temporal_graph_baseline.py` at this scale; builds may also apply a **cap on edges per bucket** to limit memory (see `README.md` for the exact command used per artifact).  
- **Social edges:** undirected links between vessels sharing the same **three-digit ITU MID** (national/organizational MMSI prefix), derived with **`scripts/mmsi_mid.py`** so prefixes are not split on **six** digits. To keep graphs tractable, social edges are **capped** (e.g., **up to 2000 per bucket** in `scripts/add_social_edges.py`).  
- **Combined edge list** for the primary social-augmented benchmark: `artifacts/edges_full_with_social.parquet`.

**Validation distances (not training edges).** Post-hoc proximity checks for candidates use **25 / 50 / 100 km** (and optional **±1 day** windows) on **daily mean** positions—**different** from the 10 km graph edge rule; §3.5.

#### 3.4 Model and training objective

**Architecture.** Temporal Graph Convolutional Network (**TGCN**) with optional **temporal node features** (degree, interaction counts, partner diversity, recency, inter-event gaps—see `scripts/build_temporal_node_features.py` and `README.md`).

**Training details (primary laptop benchmark).** Unless noted otherwise: **embedding dimension 32**, **Adam** optimizer, learning rate **0.001**, **3** training epochs over ordered training buckets, **binary cross-entropy with logits** comparing positive edges to **randomly sampled negative** pairs (same count as positives **per snapshot**). Chronological split ratio **~70/30** (`--test-ratio 0.3` in training scripts). **Train-bucket count** was capped at **1450** for the headline combined-graph run due to **laptop RAM** (higher caps caused OOM).

**Evaluation:** **ROC AUC** and **AP** on **test buckets**; edge deduplication rules avoid trivial leakage of training links into the test objective (see code comments in the TGCN training scripts).

**Computational cost.** All experiments ran on a single laptop (Apple M-series, 16 GB unified RAM). The primary combined-graph run (1450 train buckets, 3 epochs) completed in roughly **20–40 minutes** wall-clock depending on background load; peak resident memory sat near **12–14 GB**, which is why bucket caps above ~1480 triggered the OS out-of-memory killer. The proximity-only full graph trained under similar constraints. Capped-graph runs and the short August 2017 window finished in under 5 minutes each. No GPU was used; all matrix operations ran on CPU via PyTorch. These numbers are approximate—the training scripts do not log wall-clock time or peak RSS automatically, so precise reproducibility of timing requires re-running under controlled conditions.

#### 3.5 Heuristic validation (not supervised labels)

For candidate pairs we compute **Haversine distance** between **daily mean positions** and count days within **25 / 50 / 100 km** (and ±1 day alignment where noted). This is **post-hoc geography**, not a training label.

#### 3.6 Which numbers are “the same experiment”?

Readers should keep the following settings separate (do not mix metrics across rows):

| Setting | Role |
|--------|------|
| **2012–2019 combined graph (proximity + social)** | **Primary benchmark**—full timeline, social edges, reported AUC/AP on 876 test buckets. |
| **Proximity-only graph** | **Ablation**—no social edges; often higher AUC on this codebase; shows contribution of the social layer. |
| **Capped / smaller graphs** | **Sanity checks and tuning**—easier prediction; **very high** AUC; **not** the same difficulty as full global coverage. |
| **August 2017 short window** | **Visualization only**—few buckets; metrics are **not** comparable to the main table. |

---

### 4. Results

§4 presents quantitative link-prediction metrics (**Table 1**), followed by geographic validation and interpretability figures. Recall from §3.6 that table rows represent **different experimental settings**—do not mix metrics across them.

#### 4.1 Quantitative performance (TGCN)

**Table 1** collects results across settings (see §3.6 for comparability notes).

| Setting | Edge list | Train cap / notes | Test eval | ROC AUC | AP |
|---------|-----------|-------------------|-----------|---------|-----|
| Proximity-only, full | `edges_2012_2019_full.parquet` | ~1500 train buckets | time split | **0.77618** | **0.79549** |
| **Combined + social (primary laptop run)** | `edges_full_with_social.parquet` | **1450** train buckets, 3 epochs, seed 1 | **876** test buckets | **0.72997** | **0.71381** |
| Capped graph | capped parquet | tuning / easier task | — | ~0.95 | ~0.96 |
| Capped (5-fold CV) | capped parquet | rolling CV | — | 0.927 ± 0.067 | 0.947 ± 0.048 |
| August 2017 (case-study window) | short window | small | 5 test buckets | ~0.56 | ~0.67 |

**Primary run details (combined row):** embedding dimension 32, learning rate 0.001, temporal node features enabled. Artifacts: `artifacts/tgcn_social_maxb1450_ep3.json` and `.csv`. Increasing allowed training buckets (e.g., 400 → 1450) improved AUC/AP until **memory limits** on the laptop (~1480–1500 buckets caused OOM/`killed`). Further ablations: `artifacts/tgcn_improvement_suite_summary.csv`.

**Proximity-only row source.** The exact row values (0.77618 / 0.79549) come from `artifacts/tgcn_time_temporal_nodes_fullcoverage.json`.

**Variability and uncertainty.** The headline combined run is a **single-seed** report (seed = 1), so cross-seed confidence intervals are not available for the primary row. However, two other sources give a sense of variability:

- **Per-bucket dispersion.** Across the 876 test buckets the per-bucket standard deviation is **±0.103** for ROC AUC and **±0.106** for AP. Some days are easier (denser, more regular graphs) and others harder (sparser activity, regime shifts); the headline **0.72997 / 0.71381** is a macro-average over that distribution, not a claim that every day matches it. Per-bucket values: `artifacts/tgcn_social_maxb1450_ep3.csv`.
- **Cross-seed reference (capped graph).** On the easier capped graph, a 5-fold rolling-window CV with seeds 1–5 yields **0.927 ± 0.067 AUC** and **0.947 ± 0.048 AP** (Table 1). Cross-seed variance is modest there; extending multi-seed runs to the full-coverage graph is a priority for future work.

**Why do social edges lower AUC?** The proximity-only graph scores ~​0.78 AUC versus ~​0.73 for the combined graph. Three factors likely contribute:

1. **Noise from registry grouping.** MID social edges link every pair of vessels sharing the same three-digit national prefix. In a predominantly Chinese-flagged dataset (MID 412), this connects thousands of vessels that never physically co-occur, injecting edges that the model must learn to discount.
2. **Over-smoothing.** Adding dense social edges increases the effective neighborhood size for each GCN layer. When many neighbors are structurally similar but behaviorally unrelated, node embeddings blur toward a common mean, reducing the model’s ability to distinguish genuinely co-present pairs.
3. **Objective dilution.** The training loss treats social edges and proximity edges identically (both are positive examples). The model therefore spends capacity fitting registry structure rather than geographic co-occurrence, which is the signal that carries over to unseen test days.

Disentangling these effects—e.g., by weighting social edges lower, using a heterogeneous-edge architecture, or tightening the MID cap—is listed under future work (§6).

#### 4.2 Geographic validation and case studies

Among workflows that rank and check top candidates, **eight** pairs pass a **25 km, ±1 day** rule. Examples: **412422375 ↔ 412428225** (about **2** days within 25 km); **412000690 ↔ 412325200** (**102** days within 25 km). Detailed narratives and additional plots: `docs/candidate_case_studies.md`.

**Figure 1 — why include a map?** The model assigns abstract scores to vessel pairs. To verify that a high score corresponds to real-world co-movement, we plot raw AIS tracks on a map. **Figure 1** shows our **strongest candidate pair** (**412000690 ↔ 412325200**, 102 days within 25 km) against the **Chinese coastline** in the Yellow Sea / East China Sea. If the model were surfacing random pairs, we would not expect their tracks to overlap geographically — but here they clearly do. This is **visual validation**, not proof of coordination. The **eight-pair** summaries in **Table 2** and Figures 2–3 use a **fixed** full-coverage screen (slightly different pair list); Figure 1 is the **flagship** example.

How to read the figure:

- **Blue lines** trace every daily mean position for vessel **412000690**; **orange lines** trace vessel **412325200**, both spanning 2013 to 2018.
- The **colored background** is a **combined daily presence density**: for each vessel a smoothed density surface (2D kernel density estimation) is computed from all of its daily mean positions, then the two surfaces are **summed** and normalized to a 0–1 scale. The horizontal color bar at the bottom maps this scale:
  - **Dark blue / purple** (0.00–0.15) = low combined presence — neither vessel spent much time there.
  - **Teal / green** (0.15–0.45) = moderate combined presence — at least one vessel was present occasionally.
  - **Yellow / orange / red** (0.45–1.00) = high combined presence — one or both vessels were frequently located there; the warmest colors mark the densest overlap.
- **Black contour lines** with numeric labels (0.1, 0.3, etc.) outline specific density thresholds, making it easy to locate the core shared operating area.
- The **warmest region** (yellow-green, centered near ~121 E, 37–39 N off the Shandong Peninsula) is where daily positions from **both** vessels overlap most, consistent with a **shared fishing ground** in the Yellow Sea.
- **Tan shading** = land (10 m Natural Earth coastline); labeled cities (**Shanghai**, **Qingdao**, **Jinan**, **Nanjing**, **Yancheng**) provide geographic reference along the coast.

![Case study tracks](../artifacts/plots/case_study_pairs/pair_412000690_412325200_contour.png)  
*Figure 1. Vessel pair **412000690** (blue) and **412325200** (orange), Chinese coast, 2013–2018. Background = combined daily presence density (KDE, 0–1 scale; see color bar and "How to read" above). **Takeaway:** the model’s top-ranked pair shows sustained co-presence in the Yellow Sea (~121 E, 37–39 N), confirming the high link-prediction score corresponds to a real shared operating area. Tracks-only version: `pair_412000690_412325200.png`. Reproduce: Appendix B.*

#### 4.3 Monthly overlap and cluster context

**Figure 2** charts the monthly count of days each validated pair's vessels were within **25 km**. **Figure 3** maps those same rendezvous locations geographically and highlights a **six-vessel** subgroup near **~30 N, 122 E**.

How to read Figure 2:

- Each **row** is one vessel pair, sorted from most overlap (top) to least (bottom). The **y-axis label** shows the pair number, the flag state(s) involved (e.g., CHN for China), and a summary line giving the total days and active months.
- Each **bubble** represents one calendar month. **Bubble size is proportional to the number of days** the two vessels were within 25 km that month (see the size legend in the upper right: 1, 5, 10, 20 days).
- **Numbers inside larger bubbles** give the exact day count for that month.
- **Blank spaces** mean zero close approaches that month — the pair was either not active or never came within 25 km.
- **Alternating row shading** (white / light gray) helps visually separate pairs.
- **Color** distinguishes pairs but carries no additional meaning beyond identification.

How to read Figure 3:

- **Faint gray dots** show all rendezvous points (monthly locations where any of the eight pairs were within 25 km).
- **Colored markers** (circles, squares, diamonds, triangles, crosses) each represent one of the **six cluster vessels**. The legend in the upper left identifies each vessel by abbreviated MMSI and flag state (all CHN).
- **Thin gray lines** connect cluster members that appear together as a pair, showing which vessels rendezvous with which.
- The **red X** marks the **weighted hotspot centroid** — the average location of all cluster rendezvous points (~29.7 N, 122.3 E).
- **Tan land shading**, coastlines, rivers, and labeled cities (**Shanghai**, **Hangzhou**, **Ningbo**, **Wenzhou**, **Fuzhou**) provide geographic reference.

##### 4.3.1 Gear type codes (how to read “country · gear”)

Legend labels on Figures 2–4 and Tables 2–3 show **country · gear** per vessel. **Country** comes from the cell-fleet enrichment (`artifacts/cooperative_pairs_with_flag_gear.csv`) when the MMSI appears there, otherwise from the **ITU MID** prefix (e.g., MID111). **Gear** strings (e.g., `trawlers`, `set_gillnets`, `fixed_gear`, `fishing`) are **`geartype` labels** from **cell-aggregated fleet** files, assigned by **hours-weighted** overlap between MMSI daily cells and fleet rows (`scripts/enrich_pairs_with_flag_gear.py`). They indicate **broad fisheries-activity categories**, not a formal gear license per vessel. The **em dash (—)** means no gear could be attributed for that MMSI in the extract—common for the full-coverage pairs, which mostly fall outside the August 2017 enrichment window (see **Table 2**). **Plain-language definitions** of every code are in **Appendix A**.

**Table 2 — Eight validated pairs (country · gear per vessel).** *Format: src MMSI, dst MMSI, then src country · gear | dst country · gear.*

| src | dst | Country · gear (src \| dst) |
|-----|-----|------------------------------|
| 412422375 | 412428225 | CHN · — \| CHN · — |
| 412437423 | 412435485 | CHN · — \| CHN · — |
| 412410128 | 412416248 | CHN · — \| CHN · — |
| 412461376 | 412415321 | CHN · — \| CHN · — |
| 412420679 | 412413383 | CHN · — \| CHN · — |
| 412985698 | 412443375 | CHN · — \| CHN · — |
| 412450427 | 111203412 | CHN · — \| MID111 · — |
| 412461376 | 412427825 | CHN · — \| CHN · — |

**Table 3 — Six-vessel cluster (Figure 3).** *MMSIs selected by `scripts/analyze_six_vessel_cluster.py`; labels match plot legend.*

| MMSI | Country · gear |
|------|----------------|
| 412413383 | CHN · — |
| 412420679 | CHN · — |
| 412427825 | CHN · — |
| 412435485 | CHN · — |
| 412437423 | CHN · — |
| 412461376 | CHN · — |

![All pairs days within 25 km (monthly)](../artifacts/plots/pair_overlap_series/all_pairs_days_within_25km.png)  
*Figure 2. **Bubble chart** of monthly close approaches (days within 25 km) for all eight validated pairs, 2013–2019. Each row is one pair (sorted by total overlap, highest at top); bubble size is proportional to the number of days within 25 km that month; numbers inside larger bubbles show exact counts. **Pair 1** (CHN, 84 days across 9 months) dominates with large, closely spaced bubbles in 2015–2016, indicating **persistent co-presence** over an extended period. Most other pairs are **sporadic** — a handful of small bubbles scattered across different years — suggesting occasional rather than sustained proximity. **Pair 7** (CHN and MID111) is the only cross-flag pair. Y-axis labels show total days and active months per pair.*

![Six-vessel cluster](../artifacts/plots/six_vessel_cluster_scatter.png)  
*Figure 3. **Geographic scatter** of rendezvous locations for all eight validated pairs, plotted against the Chinese coast from Fuzhou to Shanghai. Faint gray dots show every recorded rendezvous point; colored markers (see legend) highlight the **six cluster vessels** (all CHN-flagged, abbreviated MMSIs in legend). Gray lines connect cluster members that co-occur as a pair. The red **X** marks the weighted hotspot centroid (~29.7 N, 122.3 E), sitting in the East China Sea roughly 50–80 km offshore from Ningbo. Most cluster activity is concentrated in a narrow band between 29 N and 30.5 N, while a few outlier points appear further south near Wenzhou. **Takeaway:** six vessels from multiple distinct pairs converge repeatedly on the same offshore area east of Ningbo, suggesting a **shared operating region** rather than coincidence — consistent with the co-presence patterns in Figure 2.*

#### 4.4 Exploratory short-window analysis (August 2017; not the main benchmark)

**Why August 2017?** The main TGCN benchmark (Table 1) spans **2012–2019** but produces abstract link-prediction scores with no fleet metadata attached. The **cell-level fleet enrichment** files that supply **country and gear** labels are only available for **August 2017**—that is the one window where model outputs can be joined to fleet context and interpreted in terms of *what kinds* of vessels were flagged. A separate, smaller TGCN run was therefore trained on this narrow slice (five daily buckets, Aug 7–11) so that cooperation predictions could be cross-referenced with gear type and flag state. The resulting metrics (~0.56 AUC, ~0.67 AP) are **not comparable** to the headline 2012–2019 scores; this window exists purely for **interpretability**.

**Figure 4** summarizes those predictions across all 61 pairs observed during the five-day window. The left panel breaks down cooperative pairs per day by **gear-type group**; the right panel shows the **flag-state** composition of cooperative pair-days. The four gear groups in the chart are:

- **Trawlers** (blue) — both vessels in the pair are classified as **trawlers** (vessels that tow nets through the water or along the seabed). This is the single largest category.
- **Line fishing** (orange) — at least one vessel uses **hook-and-line or generic fishing** methods (includes pairs labeled `fishing`, `fishing | trawlers`, `fishing | set gillnets`, etc.).
- **Set gear** (green) — pairs involving **stationary or deployed gear**: **set gillnets** (nets left in place for fish to gill in the mesh), **set longlines** (long lines with baited hooks left to soak), or **pole-and-line** methods.
- **Other / mixed** (gray) — pairs where gear is **unknown** (em dash) for one or both vessels, or combinations that do not fit the above groups (e.g., `purse seines`).

Full definitions of every individual gear code are in **Appendix A**.

![Cooperation summary](../artifacts/plots/tgcn_daily_cooperation_heatmap.png)  
*Figure 4. **Aggregate cooperation summary** (Aug 7–11, 2017). **Left:** stacked bar chart of cooperative pairs per day, colored by gear-type group. Numbers inside each segment show the count; daily totals are labeled above. **Right:** flag-state breakdown of all cooperative pair-days. **Takeaway:** cooperation predictions are spread across the full five-day window, with **trawler–trawler** pairs (blue) accounting for the largest share on most days and **Chinese-flagged** vessels (CHN, red bar) dominating the country breakdown. Friday (Aug 11) had the most cooperative pairs (14), driven almost entirely by trawlers.*

---

### 5. Limitations

#### 5.1 Threats to validity (why high scores ≠ “guilt”)

- **Link prediction is not a verdict.** ROC AUC / AP measure **ranking** of observed vs sampled non-edges under the **graph definition**. A model can score well when vessels **share fishing grounds**, **ports**, or **traffic lanes**—behavior that is **legal** and common.  
- **AIS selection and compliance.** Vessels may **disable** AIS, use **identity changes**, or have **sparse** coverage; the graph only sees **reported** positions.  
- **Confounding by density.** **Crowded** regions produce many **possible** edges; **degree** and **popularity** effects can inflate link-prediction metrics relative to “rare coordination.”  
- **Metric–semantics gap.** **Good calibration** of the learning objective does **not** imply validated **enforcement** outcomes; external **ground truth** would be needed for that (we have none for cooperation).

#### 5.2 Design and data constraints

- **No behavioral ground truth** for “cooperation” or IUU.  
- **Grid and sampling** choices affect positions and edges.  
- **Laptop memory** caps training buckets (**1450** stable here; higher values failed).  
- **August 2017** is a narrow window—patterns there may not generalize.  
- **Shared geography** vs **coordination:** overlap in dense regions is ambiguous (see §5.1).  
- **Country and gear** on plots come from **cell-aggregated fleet** tables (§4.3.1, **Appendix A**)—descriptive context, not a formal registry. Gear labels should support interpretation only, not enforcement decisions.
- **Single-seed headline numbers.** The primary combined-graph AUC/AP is from one seed; per-bucket dispersion and cross-seed references are discussed in §4.1, but full multi-seed CIs on the headline row remain future work.

---

### 6. Conclusion and future work

We presented an **unsupervised** AIS→graph→TGCN pipeline with a **time-split** evaluation and an explicit **link-prediction** formulation (§3.1; **Appendix C**). The **combined proximity + MID-correct social** graph yields **0.72997 ROC AUC** and **0.71381 AP** on **876** test buckets under the documented laptop protocol (**Table 1**), alongside **proximity-only** and **capped-graph** comparisons. **Eight** pairs pass strict **post-hoc** proximity screening (**Table 2**); interpretability figures add **country and gear** context where data allow (**Appendix A**). For practice, the output is a **ranked candidate list** plus **reproducible** geographic and fleet-context checks—not a determination of wrongdoing. We do **not** claim confirmed coordination or IUU.

**Future work** includes finer spatial/temporal resolution; owner or fleet metadata from registries; **supervised or semi-supervised** learning when reliable labels exist; tighter integration of **movement-based gear** or behavior classifiers; full-coverage **multi-seed** runs and larger training-bucket budgets on **high-memory** hardware; **heterogeneous-edge** or **edge-weighted** architectures to better exploit social edges without the over-smoothing and noise effects described in §4.1; and continued care in separating **benchmark** metrics (2012–2019) from **illustrative** short windows (e.g., August 2017).

---

### References

- Kroodsma, D. A., et al. (2018). *Tracking the global footprint of fisheries*. Science, 359(6378), 904–908.  
- Miller, N. A., et al. (2018). *Identifying global patterns of transshipment behavior*. Frontiers in Marine Science, 5, 240.  
- Park, J., et al. (2020). *A systematic assessment of vessel monitoring data for identifying suspicious transshipment events*. ICES Journal of Marine Science.  
- Kipf, T. N., & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks*. ICLR.  
- Hamilton, W. L., Ying, R., & Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs*. NeurIPS.  
- Yu, B., Yin, H., & Zhu, Z. (2018). *Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting*. IJCAI.  
- Seo, Y., et al. (2018). *Structured Sequence Modeling with Graph Convolutional Recurrent Networks*. ICONIP.  
- Rossi, E., et al. (2020). *Temporal Graph Networks for Deep Learning on Dynamic Graphs*. ICML Workshop on Graph Representation Learning.  
- Lü, L., & Zhou, T. (2011). *Link prediction in complex networks: A survey*. Physica A, 390(6), 1150–1170.  

---

### Appendix A — Gear type codes

*Plain-language definitions for **gear** strings in **Tables 2–3** and **Figures 2–4**. The same content is kept in the repository as **`docs/gear_types.md`** for version control and easy editing.*

#### A.1 Source of labels

In this project, **gear** on figures and tables comes from **cell-level fleet aggregates**, not from a per-vessel registry. For each MMSI–day, we join AIS daily cells to fleet files where each row is **hours** by **flag × geartype** in that cell; the vessel is labeled with the **geartype that receives the most weighted hours** that day (`scripts/enrich_pairs_with_flag_gear.py`). The label is a **coarse, behavioral proxy** for “what kind of fishing that cell’s traffic was attributed to,” not a formal gear certificate for the MMSI.

**Plots use —** when no gear could be attributed (vessel missing from the join, or no overlapping fleet rows).

#### A.2 Codes in `artifacts/cooperative_pairs_with_flag_gear.csv`

These strings are taken **as-is** from the fleet **`geartype`** column (snake_case).

| Code (CSV) | Short gloss |
|------------|-------------|
| **`fishing`** | **Generic / unspecified fishing.** Catch-all category in the source data when activity is attributed to “fishing” without a more specific gear class. Treat as **low specificity** compared to trawlers, nets, etc. |
| **`trawlers`** | **Trawling.** Vessels **tow nets** through the water column or along the bottom (e.g. otter trawl, beam trawl). High mobility; distinct from **fixed** or **set** gear. |
| **`fixed_gear`** | **Stationary gear** fixed to the seabed or structure: pots, traps, stakes, weirs, and similar **non-towed** gear that stays in place. (Name reflects “fixed” position, not “repaired.”) |
| **`set_gillnets`** | **Set gillnets.** Nets **set and left** (anchored, on bottom, or sometimes drifting) so fish **gill** in the mesh. Not actively towed like a trawl. |
| **`set_longlines`** | **Set longlines.** A **long line** with many **baited hooks**, deployed and **left to soak** (demersal or pelagic longline, depending on fishery). |
| **`other_purse_seines`** | **Purse seining (other).** A **surrounding net** used on **schooling fish**; the bottom is **pulled closed** (“pursed”) like a drawstring. “Other” indicates a sub-type bucket in the source taxonomy (not necessarily “miscellaneous quality”). |
| **`pole_and_line`** | **Pole-and-line / baitboat.** Fish caught with **hand-held poles** and **hooks**, often with **live bait**; common in some tuna fisheries. |

#### A.3 How to interpret labels

1. **Same MMSI, different days** can get different geartypes if the vessel moves between cells dominated by different fleet attributions.  
2. **Same geartype on two MMSIs** does not prove they use identical gear—only that both were attributed the same **coarse** class in that **August 2017** window (where enrichment applies).  
3. **AIS “ship type”** codes differ from fleet **geartype** here: the latter is **fisheries-activity** oriented, not the full IMO/AIS ship-type list.

Methodology of the join and weighting: `scripts/enrich_pairs_with_flag_gear.py` and §4.3–4.4 of this report.

#### A.4 Related work (not from the same CSV)

The repository also includes a **movement-based gear classifier** trained on **anonymized** labeled tracks (see project `README.md`, “Gear classification”). Those class labels (e.g. `purse_seines`, `trollers`) are **not** automatically the same strings as the fleet **`geartype`** column above; do not merge them without an explicit mapping.

---

### Appendix B — Reproducibility

| Item | Command / path |
|------|----------------|
| Combined edges | `python3 scripts/add_social_edges.py --edges artifacts/edges_2012_2019_full.parquet --out artifacts/edges_full_with_social.parquet --max-social-per-bucket 2000` |
| Primary TGCN JSON/CSV | `artifacts/tgcn_social_maxb1450_ep3.json`, `.csv` — PyG env, `PYTHONPATH=scripts` |
| Optional per-bucket metric plots (not in main report) | `python3 scripts/plot_tgcn_bucket_metrics.py --report artifacts/tgcn_social_maxb1450_ep3.json --out-dir artifacts/plots` |
| Case study track plots (Figure 1) | `python3 scripts/compute_pair_overlap_from_daily.py --pairs artifacts/case_study_pairs.csv --daily-root "data/MMSI daily vessels " --top-k 4 --distance-km 25 --day-window 1 --max-files-per-year 0 --out-dir artifacts/plots/case_study_pairs --contour` — **`--max-files-per-year 0`** uses all daily CSVs (default); a small number **undercounts** overlap days vs `docs/candidate_case_studies.md`. **`--contour`**: combined daily presence density (sum of two 2D KDE surfaces, `Spectral_r` colormap, labeled iso-lines, colorbar). With **cartopy** installed, the plot includes **coastlines, land fill, rivers, and city labels** for geographic context. First pair → **`pair_<src>_<dst>_contour.png`** + tracks-only **`pair_<src>_<dst>.png`**. Optional `--out-summary`. **`--contour-all-pairs`** for contour files on every pair. |
| Ablations | `scripts/run_tgcn_improvement_suite.py` → `artifacts/tgcn_improvement_suite_summary.csv` |
| August cooperation summary (Figure 4) | `PYTHONPATH=scripts python3 scripts/plot_cooperative_heatmap.py` — two-panel aggregate cooperation summary: stacked bar chart by gear group + flag-state breakdown (Aug 2017 short window). |
| Eight-pair overlap CSV + optional heatmap PNG | Full command in **block below** (overlap CSV feeds **Tables 2–3** / Figures 2–3). |
| Monthly overlap bubble chart (Figure 2) | `PYTHONPATH=scripts python3 scripts/plot_pair_overlap_time_series.py --overlap-csv artifacts/eight_pairs_overlap_by_month.csv --enrichment artifacts/cooperative_pairs_with_flag_gear.csv` — bubble chart of monthly close approaches, one row per pair. |
| Six-vessel cluster map (Figure 3) | `PYTHONPATH=scripts python3 scripts/analyze_six_vessel_cluster.py --overlap-csv artifacts/eight_pairs_overlap_by_month.csv --enrichment artifacts/cooperative_pairs_with_flag_gear.csv` — geographic scatter with cartopy coastline, land fill, cities. |
| Enrichment CSV (for labels) | `artifacts/cooperative_pairs_with_flag_gear.csv` — `scripts/enrich_pairs_with_flag_gear.py` |
| **Gear type definitions (for tables & figures)** | **Appendix A** (mirror for repo edits: `docs/gear_types.md`) |
| **Notation / formal link-prediction setup** | **Appendix C** (symbol table; training vs validation distances) |
| Gear × country pair-count heatmap (updated PNG) | `PYTHONPATH=scripts python3 scripts/plot_flag_gear_enrichment.py --input artifacts/cooperative_pairs_with_flag_gear.csv --out-dir artifacts/plots` |
| Candidates / close pairs | `artifacts/tgcn_candidate_scores_fullcoverage.parquet`, `artifacts/close_pairs_fullcoverage_25km_w1.csv` |

**Eight-pair overlap heatmap — copy/paste (project root):**

```bash
PYTHONPATH=scripts python3 scripts/overlap_by_month_8pairs.py \
  --pairs artifacts/close_pairs_fullcoverage_25km_w1.csv \
  --daily-root "data/MMSI daily vessels " \
  --all-files --full-months \
  --distance-km 25 --day-window 1 \
  --out-csv artifacts/eight_pairs_overlap_by_month.csv \
  --out-plot artifacts/plots/eight_pairs_overlap_by_month.png \
  --enrichment artifacts/cooperative_pairs_with_flag_gear.csv
```

*Faster test run (subsamples daily files; shorter runtime):* omit `--all-files` and add e.g. `--max-files-per-year 30`.

---

### Appendix C — Notation and formal problem setup

*This appendix supports §3.1. It is safe to **omit from slide decks** or to move after appendices A–B in a bound thesis if your program requires a fixed appendix order.*

#### C.1 Symbol glossary

| Symbol | Meaning |
|--------|---------|
| \(t\) | A calendar **day** (time bucket) in the AIS timeline. |
| \(\mathcal{T}\) | The set of days used after filtering (e.g., days with edges). |
| \(V\) | Set of **nodes** (9-digit vessel **MMSIs** appearing in the extract). |
| \(G_t = (V, E_t)\) | **Undirected** snapshot graph for day \(t\). |
| \(E_t\) | **Edges** for day \(t\): proximity (§3.3) ± optional social edges (§3.3). |
| \(u, v\) | Distinct nodes (unordered pair \(\{u,v\}\)). |
| \(s_{u,v}\) | **Scalar score** (logit) for pair \((u,v)\) on a given forward pass / bucket. |
| \(\mathbf{h}_u\) | **Embedding vector** for node \(u\) after the TGCN (dimension **32** in the primary run). |
| \(\mathcal{T}_{\mathrm{train}}, \mathcal{T}_{\mathrm{test}}\) | **Chronological** partition of buckets (**~70% / ~30%**). |

#### C.2 Graph semantics

For each \(t \in \mathcal{T}\), \(G_t = (V, E_t)\) is **undirected**: \(\{u,v\} \in E_t\) means the pair satisfies the **edge construction rule** for that day—**spatial proximity** within the **10 km** same-day threshold (and optional **same three-digit ITU MID** links), not “ground-truth cooperation.” Node set \(V\) may be defined globally from all MMSIs in the study period or induced per implementation; edges are **sparse** and may be **capped per bucket** for memory (§3.3).

#### C.3 Temporal link prediction objective (informal)

The model maps the sequence \(\{G_t\}\) (and optional **temporal node features**) to a recurrent hidden state and node embeddings. For each training snapshot, **positive** pairs are edges in \(E_t \cap \mathcal{T}_{\mathrm{train}}\); **negative** pairs are **sampled** from node pairs that are **not** positive for that snapshot (count matched to positives **per bucket** in code). Scores take the form \(s_{u,v} \propto \mathbf{h}_u^\top \mathbf{h}_v\) (inner product), optimized with **binary cross-entropy with logits** (BCEWithLogitsLoss) against labels \(y \in \{0,1\}\). **There is no label** for IUU, transshipment, or “cooperative”—only **structural** positives and **random** negatives.

**Evaluation.** On \(\mathcal{T}_{\mathrm{test}}\), the model ranks candidate pairs; **ROC AUC** and **Average Precision (AP)** summarize how well **observed** test edges rank above **non-edges** (per the evaluation script’s protocol). High scores mean **plausible future co-occurrence** under the graph definition, not culpability.

#### C.4 Training vs validation distances

| Stage | Typical distance / rule | Role |
|-------|---------------------------|------|
| **Graph edges (training signal)** | **≤ 10 km**, same **day** | Defines \(E_t\) for learning. |
| **Candidate screening (post-hoc)** | **25 / 50 / 100 km**, optional **±1 day** | **Not** used as training labels; validates high-scoring pairs on raw tracks (§3.5). |

This separation is repeated in §3.3–3.5 to avoid conflating **model inputs** with **analyst-facing** thresholds.
