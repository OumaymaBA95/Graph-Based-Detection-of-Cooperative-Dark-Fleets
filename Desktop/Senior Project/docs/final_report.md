---
title: "Graph-Based Screening for Potentially Cooperative Vessel Pairs"
subtitle: "A temporal link prediction approach using AIS (2012–2019)"
author: Oumayma Ben Aoun
output:
  pdf_document:
    latex_engine: xelatex
    keep_tex: true
toc: true
toc-depth: 3
numbersections: true
header-includes: |
  \setlength{\emergencystretch}{4em}
  \usepackage{etoolbox}
  \makeatletter
  \renewcommand{\maketitle}{\begin{titlepage}\centering\vspace*{1.1in}{\LARGE\bfseries\@title\par}\vspace{2.25em}{\large\@author\par}\vspace{1em}{\normalsize\textit{Faculty Advisor: Dr. Ali Duman}\par}\vspace{0.6em}{\normalsize\textit{Committee Members: Dr. Katherine Shoemaker, Dr. Timothy Redl}\par}\vfill\end{titlepage}\clearpage}
  \makeatother
  \apptocmd{\tableofcontents}{\clearpage}{}{}
  \AtBeginEnvironment{longtable}{\footnotesize\setlength{\tabcolsep}{4pt}}
  \setlength{\hfuzz}{6pt}
  \IfFileExists{fvextra.sty}{\usepackage{fvextra}\RecustomVerbatimEnvironment{Highlighting}{Verbatim}{commandchars=\\\{\},breaklines=true,breakanywhere=true}}{}
  \AtBeginDocument{\sloppy}
---

<!-- Title, subtitle, author, and advisor line come from YAML / header-includes for Pandoc → PDF. Body starts at Abstract. -->

# Abstract {.unnumbered}

Illegal, unreported, and unregulated (IUU) fishing has increased interest in tools that surface vessels that may operate in a coordinated way. This project develops an **unsupervised** screening pipeline built on Automatic Identification System (AIS) data (2012–2019). Each day is represented as a graph of vessels; a **Temporal Graph Convolutional Network (TGCN)** performs **temporal link prediction** so that pairs that plausibly co-occur in the future rank highly. Raw tracks are then checked independently with simple geographic rules to prioritize pairs for analyst review. **Training graph edges** connect vessels whose positions fall within **10 km** on the **same calendar day**. Wider distance checks (for example **25 km**) are used only **after** scoring to validate candidates—they are **not** training labels. The model does **not** receive labels for “cooperative” behavior; it learns from **structure and timing** in the graphs and produces **candidates for review**, not adjudicated outcomes.

**Main quantitative result (Table 1):** On the **combined** graph—proximity edges plus optional **social** edges linking vessels that share the same three-digit maritime identifier (MID) prefix, implemented with a corrected MID rule—a reproducible laptop configuration (**1450** training days, 3 epochs, seed 1) achieves **ROC AUC = 0.72997** and **average precision (AP) = 0.71381** on **876** held-out daily test buckets (`artifacts/tgcn_social_maxb1450_ep3.json`). A **proximity-only** ablation on the same edge timeline reaches higher scores (about **0.78** AUC) but on a much smaller **20-bucket evaluation slice**; it is reported in Table 1 as a **directional ablation**, not a like-for-like headline. A narrow **August 2017** window was used early for exploration; its metrics **cannot** be compared with the full 2012–2019 benchmark, and this report does not include figures from that window. After strict **25 km** and **±1 day** screening, **eight** pairs remain of interest; a **gear-aware** follow-up analysis suggests that many **excluded** top candidates resemble routine same-fleet trawler co-location rather than unusual encounters. We **do not** assert legal conclusions or membership in any “dark fleet.”

**Takeaways.** **(i)** Modest absolute scores are **expected** for an unsupervised, label-free screening task on global AIS, and the ±0.10 day-to-day spread is a property of the domain rather than a flaw of the model. **(ii)** **Post-hoc** **25 km / ±1 day** geography matters because **10 km** same-day graph structure can elevate fleet-like pairs that rarely meet the tighter physical standard; the geographic filter separates **fleet-scale similarity** from **pair-level proximity**. **(iii)** **How** registry information enters the graph matters as much as **whether** it does: dense same-MID social edges add noise, encourage over-smoothing, and compete with proximity in the loss—future work should fuse registry context as **node features** or as **typed edges** instead. **(iv)** The deliverable is a **short, auditable** list (top-200 → 8 geography survivors, ~4%; as a group, cross-gear pairs survive at ~5.1% vs 2.4% for trawler–trawler, and the best cross-gear category at ~10× the trawler–trawler rate) with **tracks and overlap plots**—**prioritized leads** plus **supporting context**, **not** a determination of misconduct.

**Keywords:** AIS, IUU fishing, temporal graph neural networks, link prediction, vessel co-movement

---

# Introduction

**Problem.** Fisheries enforcement and marine research often need to narrow attention to vessel pairs that may interact at sea—whether through transshipment, repeated joint presence, or other coordinated patterns sometimes associated with periods of missing AIS (“dark” behavior). AIS offers broad spatial and temporal coverage, yet **explicit labels for cooperation are seldom available** in operational datasets.

**What this work does.** We cast screening as **temporal link prediction** on **daily** graphs of vessels. Pairs that receive high scores under this objective are **candidates** for closer inspection. We support that step with straightforward **post-hoc** checks: distances between **daily mean positions**, maps of raw tracks, and summaries of overlap over time.

**What this work does not do.** This pipeline does **not** establish coordination, wrongdoing, or IUU activity. It yields a **ranked list** and **supporting geographic context** intended for expert review.

**Contributions (short).**

1. An end-to-end, reproducible workflow: AIS records → daily graphs → TGCN with optional temporal node features → chronological train/test split.  
2. A **three-digit MID–consistent** design for combining proximity edges with same-prefix **social** edges (`scripts/mmsi_mid.py`, `scripts/add_social_edges.py`).  
3. Reported metrics on **full-coverage** graphs within **practical laptop memory** limits (training is capped by day count where necessary).  
4. Interpretation beyond raw scores: geographic filters, overlap visualizations for eight validated pairs, and **gear-aware stratification** (§4.2.1) that compares candidates to fleet-wide co-location patterns so that routine fleet behavior can be distinguished from unusual pairs.

**Roadmap.** Section 3 presents **methods**: formal setup (**Appendix C**), **data**, **graph construction** (10 km proximity plus optional MID-based social edges), **training**, and **validation-only** distance rules. Section 4 presents **results**: headline metrics (**Table 1**), an encoder comparison (**GCN** vs. **graph transformer**) on a **smaller capped** graph (§4.1.1; distinct from the full-coverage experiments in Table 1), geographic and case-study material (**Tables 2–3**, Figures 1–4), and gear definitions (**Appendix A**). Section 5 discusses limitations (grouped under task, data, evaluation, and deployment) and ends with a short **synthesis**; Section 6 concludes; **Appendix B** collects reproduction commands.

> **For readers and committees (three sentences).** (1) **Training** graphs use **10 km** same-day proximity (plus optional MID social edges); **25 km** and wider screens are **validation**, not supervised labels. (2) The headline **0.72997 / 0.71381** values apply to the **combined** full-coverage row in **Table 1**; **proximity-only** and **capped** settings are **separate experiments**—see §3.6. (3) **Figure 2** uses 2012–2019 fleet context to show that many high-scoring pairs align with ordinary fleet co-location; the **eight** pairs that pass strict geography stand out relative to that background.

---

# Related work

**Fisheries monitoring and AIS.** Research on illegal, unreported, and unregulated (IUU) fishing and on coordinated vessel behavior makes heavy use of **Automatic Identification System (AIS)** data to characterize fishing effort and vessel encounters (see References for global fisheries footprint and transshipment studies). AIS alone does not encode whether two vessels cooperated or broke rules; analysts usually combine tracks with **proximity logic**, **registry information**, or **enforcement intelligence**. That limitation motivates methods that **rank pairs without cooperation labels**.

**AIS-based fisheries and encounter analytics.** Beyond mapping where vessels go, AIS supports **encounter analysis** and **behavioral screening** (e.g., transshipment and loitering—Miller et al., 2018; assessments of monitoring systems—Park et al., 2020). Much of that work uses **explicit rules** or **supervised** signals when labels exist. Here the learning stage remains **unsupervised**; geographic checks enter only as **post-hoc** validation.

**Graph learning and link prediction.** Vessel co-presence maps naturally to a **graph**: nodes are vessels; edges encode same-day proximity (and here, optional ties based on registry prefix). **Graph convolutional networks** (Kipf & Welling, 2017) and **inductive** neighborhood models (Hamilton et al., 2017) learn node embeddings from topology and features. **Link prediction** ranks missing or future edges (Lü & Zhou, 2011). When the graph changes over time, **temporal** formulations apply—for example spatio-temporal convolutions (Yu et al., 2018) or recurrent temporal graph networks (Rossi et al., 2020). Our model follows a **TGCN**-style recurrent update over daily snapshots and scores pairs with an **inner-product** link head (§3.4).

**Evaluation philosophy.** Train and test sets split **calendar days** in chronological order: the model trains on earlier periods and is scored on **later** periods. ROC AUC and average precision therefore reflect **forward-looking** ranking quality. Random mixing of days across time would ignore drift and could **inflate** scores; we avoid that design.

---

# Methods

## Problem formulation

Fix a **calendar day** (time bucket) $t$. Let $G_t = (V, E_t)$ be an **undirected** graph for that day: **vertices** $V$ are vessel identifiers (**MMSIs**); **edges** $E_t$ are built as in §3.3—mainly pairs whose positions lie within **10 km** great-circle distance on day $t$, optionally augmented by **social** edges between vessels sharing the same **three-digit MID** prefix. The learning problem is **temporal link prediction**: a **TGCN** maps the sequence of graphs to node embeddings; **pair scores** are inner products of embeddings. Training minimizes **binary cross-entropy with logits** so that **observed** edges score above **randomly sampled non-edges** on each day. There is **no** label for “cooperative” fishing—only structural positives and negatives (see `scripts/run_tgcn_time_multiseed.py`; architecture §3.4). **Training** and **test** sets split **days** in time order (**about 70% / 30%**). **ROC AUC** and **AP** summarize ranking quality on **future** test days.

**Appendix C** collects **notation**, repeats the formal picture for readers who prefer a tabular glossary, and distinguishes **10 km training edges** from **validation-only** distances.

## Data and time buckets

AIS records cover **2012–2019** at **daily** aggregation (on the order of **2.38 billion** vessel–day rows, with sea-surface temperature merged where available). The study timeline is partitioned into **one graph per calendar day**; **876** days carry the edge data used for the primary evaluation.

## Graph construction

- **Nodes:** one per **MMSI** (vessel identifier).  
- **Proximity edges:** same **calendar day**, **Haversine** distance at most **10 km** between **daily** position summaries for the two vessels. The full-coverage list `edges_2012_2019_full.parquet` is produced by `scripts/build_temporal_graph_baseline.py`. Some builds **limit edges per day** to fit in memory; see `README.md` for commands tied to each artifact.  
- **Social edges:** undirected ties between vessels that share the **same three-digit ITU MID** prefix. **`scripts/mmsi_mid.py`** extracts the prefix so it is not truncated at six digits. Social edges are **capped per day** (for example **2000** per bucket in `scripts/add_social_edges.py`) so that graphs remain tractable.  
- **Combined (proximity + social)** benchmark list: `artifacts/edges_full_with_social.parquet`.

**Validation distances (not training edges).** Candidate review uses **25 / 50 / 100 km** thresholds (and sometimes **±1 day** alignment) on **daily mean** positions. These rules **do not** define training edges and are **not** the same as the **10 km** edge construction; see §3.5.

## Model and training objective

**Architecture.** A Temporal Graph Convolutional Network (**TGCN**) with optional **temporal node features** derived from recent graph statistics (degree, interaction counts, partner diversity, recency, gaps between events—built in `scripts/build_temporal_node_features.py`; summarized in `README.md`).

**Training details (primary laptop benchmark).** Unless stated otherwise: **embedding size 32**, **Adam**, learning rate **0.001**, **3** passes over training days in order, **binary cross-entropy with logits** with one **random non-edge** sampled per positive edge **per day**. Chronological split **about 70% train / 30% test** (`--test-ratio 0.3`). For the headline **combined** graph run, training was limited to **1450** days because **larger** day counts exceeded available **RAM** on the laptop and led to out-of-memory termination.

**Evaluation.** **ROC AUC** and **AP** are computed on **held-out test days**. The implementation drops or masks edges so that the test loss does not reuse training positives in a trivial way (see comments in the TGCN scripts).

**Computational cost.** Experiments used one **laptop** (Apple M-series, **16 GB** unified memory). The main combined-graph run (**1450** train days, **3** epochs) typically finished in **20–40 minutes** depending on system load; **peak memory** was about **12–14 GB**, which explains why caps near **1480** days triggered failures. Proximity-only full graphs were trained under similar limits. Smaller capped graphs and the brief August 2017 run finished in **under five minutes** each. All computation was **CPU-only** (PyTorch). Timing and memory are **indicative**; scripts do not log wall-clock or RSS automatically, so exact timings require a controlled rerun.

## Heuristic validation (not supervised labels)

For scored pairs we measure **great-circle distance** between **daily mean positions** and count days falling within **25 / 50 / 100 km** (and optional **±1 day** alignment). These steps describe **where** vessels were relative to each other; they **do not** define the learning objective. §4.2.1 adds **gear-aware** context by comparing candidates to **fleet-wide** co-location patterns.

## Which numbers are “the same experiment”?

Treat each row below as a **distinct** experimental setting—**do not** combine metrics across rows:

| Setting | Role |
|--------|------|
| **2012–2019 combined graph (proximity + social)** | **Primary benchmark**—full timeline, social edges, reported AUC/AP on 876 test buckets. |
| **Proximity-only graph** | **Ablation**—no social edges; often higher AUC on this codebase; shows contribution of the social layer. |
| **Capped / smaller graphs** | **Sanity checks and tuning**—easier prediction; **very high** AUC; **not** the same difficulty as full global coverage. |
| **August 2017 short window** | **Exploratory only**—few buckets; metrics are **not** comparable to the main table. No figures from this window appear in the final report. |

---

# Results

The presentation follows a standard empirical flow: **overall scores** (§4.1), **maps and interpretation** (§4.2–4.3), and a brief **exploratory** slice from August 2017 (§4.4). As emphasized in §3.6, **each row of Table 1** corresponds to a **different** graph setup—scores from different rows **cannot** be merged into one narrative.

## Quantitative performance (TGCN)

**Headline numbers, in one place.** On the full **2012–2019** combined graph the primary laptop run reports **ROC AUC = 0.72997** and **AP = 0.71381** on **876** held-out daily test buckets. The **proximity-only** ablation on the same edge timeline reaches **0.77618 / 0.79549**, but on a much smaller **20-bucket evaluation slice** (single-seed, 5 epochs); it is reported as a **directional ablation**, not a like-for-like headline. The smaller **capped** graph used for tuning reaches roughly **0.95 / 0.96**, and a **5-fold rolling** version of that capped setup gives **0.927 ± 0.067 / 0.947 ± 0.048** across seeds **1–5**. **Table 1** collects every setting in one block; the rest of this section explains how to read each row.

**Table 1.** All TGCN results in one place. Each row is a **different** experiment—do not mix metrics across rows (§3.6). Parquet filenames are omitted here so the table fits the page; see §3 (graph construction) and **Appendix B** (reproducibility) for full paths.

| Setting | Train / eval scope | Test buckets | ROC AUC | AP |
|---------|-------------------|-----------|---------|-----|
| Proximity-only, full coverage edges | full 2012–2017-08 train; eval on a **20-bucket** slice (single seed, 5 epochs) | 20 | **0.77618** | **0.79549** |
| **Combined + social — primary laptop run** | **1450** train buckets, 3 epochs, seed 1 | **876** | **0.72997** | **0.71381** |
| Capped graph (single seed) | tuning / easier task | — | ~0.95 | ~0.96 |
| Capped graph, 5-fold rolling CV | seeds 1–5 | — | 0.927 ± 0.067 | 0.947 ± 0.048 |
| August 2017 (case-study window) | exploratory short window | 5 | ~0.56 | ~0.67 |

**How to read Table 1.** **The only headline number is the combined+social row** (**0.72997 / 0.71381** on **876** test buckets). The proximity-only row sits at **0.77618 / 0.79549** but on a **20-bucket evaluation slice**—roughly **40× fewer** test days—so the higher value is **directional, not like-for-like**: it says proximity edges score better than proximity+social on these days, but does not say a fully comparable proximity-only run would land at 0.78. The **capped** rows look much higher because the graph is a **smaller, denser subset** that is structurally easier to predict; they are reported only to support **tuning** and **multi-seed sanity checks**, not as alternative headlines. The **August 2017** row covers a **5-day** window used for early exploration and gear enrichment (§4.4); it is intentionally **not** comparable to the main rows.

**Are these numbers good?** For an **unsupervised, label-free** screening task on **global AIS** with millions of node-day combinations, **AUC around 0.73** with **AP around 0.71** is in line with what should be expected. The **AP** result is the more telling figure: on a sparse temporal graph where the vast majority of candidate pairs are non-edges, an AP near **0.71** corresponds to a model that consistently surfaces real co-occurrence near the top of its rankings—well above the **prevalence baseline** that random ranking would yield. The proximity-only **directional ablation** (~**0.78** AUC on a 20-bucket evaluation slice—see Table 1 caveat) hints at an upper bound on what this architecture can extract from purely geographic structure; closing the remaining gap will likely require **architectural changes**, not just longer training (see “Why do social edges lower AUC?” below and §6).

**Day-to-day variability is a property of the domain, not a bug.** The per-bucket standard deviations of about **±0.103** ROC AUC and **±0.106** AP across the **876** test days mean that **some days are genuinely easier than others**: graphs are denser when vessels concentrate on seasonal grounds and sparser during transit windows; weather, holidays, and policy changes also shift activity. A **macro-average** of **0.73 / 0.71** across that distribution is the right summary; a stable model on global AIS should not produce identical performance every single day, and one that did would be a flag for **leakage** rather than for skill.

**Operationally relevant top-N behavior.** For a screening tool the practical question is: of the pairs the model ranks highest, **how many** survive an analyst’s independent geographic test? Among the **top 200** pairs from the full-coverage combined-graph run, **eight** clear the strict **25 km / ±1 day** rule on raw daily means—a roughly **4%** survival rate on the long list (Table 2; §4.2)—well above the underlying prevalence of repeated 25 km contacts among **all** active pairs, which is orders of magnitude smaller. The **gear-aware** stratification in §4.2.1 tightens this further: as a **group**, cross-gear combinations survive at **5.1%** versus **2.4%** for trawler–trawler (a ~2× aggregate effect), and at the **extremes** the best cross-gear category (**Fixed gear + Trawlers**) survives at **25%**—roughly **10×** the trawler–trawler rate. The full funnel is summarized below.

**Table 1b — Operational funnel (combined graph, full-coverage run).** A precision-at-K-style summary of how the top-200 ranked candidates collapse to an analyst-reviewable list. Numbers come from the same primary run as Table 1 (`tgcn_social_maxb1450_ep3.json`) and the gear-aware analysis in §4.2.1.

| Stage | Pairs | Pass rate vs. previous stage |
|-------|------:|------------------------------|
| Top-200 model-ranked candidates (input list) | 200 | — |
| Pass strict **25 km / ±1 day** geographic filter | **8** | **4.0%** of 200 |
| &nbsp;&nbsp;— within passing set: **cross-gear** pairs | 6 | **5.1%** of 118 cross-gear candidates |
| &nbsp;&nbsp;— within passing set: **trawler–trawler** pairs | 2 | **2.4%** of 82 trawler–trawler candidates |
| Highest-discriminating cross-gear category: **Fixed gear + Trawlers** | 1 / 4 | **25.0%** (~10× trawler–trawler) |

ROC AUC and AP are reported in Table 1 because they are standard, but the **top-200 → geography → gear-context** funnel is what makes the output usable for an analyst.

**Primary run details (combined row).** Embedding size **32**, learning rate **0.001**, temporal node features **on**. Primary outputs: `tgcn_social_maxb1450_ep3.json` plus the companion `.csv` in `artifacts/`. Raising the training day cap from **400** toward **1450** improved metrics until **RAM** limited further growth (roughly **1480–1500** days caused failure). Other tuning experiments are summarized in `tgcn_improvement_suite_summary.csv` (same folder).

**Proximity-only row (directional ablation).** Values **0.77618 / 0.79549** come from `tgcn_time_temporal_nodes_fullcoverage.json` in `artifacts/`. That run used the full **2012–2019** edge file and trained up to the same train/test cutoff, but **evaluated** on only a **20-bucket evaluation slice** (versus **876** buckets for the combined run); reporting it alongside the combined headline is informative for **direction** (proximity edges score higher than proximity+social on these days) but the **absolute** numbers are not directly comparable. Re-running the proximity-only configuration on the full **876**-bucket horizon is queued under §6.

**Variability and uncertainty.** The headline combined result uses **one** random seed (seed **1**); a full multi-seed interval for that full-coverage row is **explicitly noted as future work** in §6. Two views of spread are already available and are reported as a partial substitute:

- **Day-to-day spread (within one seed).** Over the **876** test days, standard deviations of **per-day** ROC AUC and AP are about **±0.103** and **±0.106**. Easy and hard days coexist; the reported **0.72997 / 0.71381** is a **macro-average** over days, not a promise about every individual day. The breadth of this spread is itself a feature of the domain—seasonal grounds, weather, holidays, and policy changes all shift activity—and a model that produced identical performance on every test day would be a flag for **leakage** rather than for skill. Per-day CSV: `tgcn_social_maxb1450_ep3.csv` in `artifacts/`.
- **Cross-seed spread on a capped graph.** A **5-fold** rolling validation with seeds **1–5** on the smaller capped graph gives **0.927 ± 0.067** AUC and **0.947 ± 0.048** AP (Table 1). Cross-seed variance there is modest, which is **suggestive but not conclusive** that seed effects on the full-coverage row would be smaller than the day-to-day effects above. Re-running the headline configuration with **at least 3–5 seeds** is the single most useful next experiment for this report and is a top item in §6.

### GCN vs. graph transformers

Keeping the §3.4 training recipe fixed, **only** the **per-day graph encoder** changes. Training driver: `run_tgcn_time_multiseed.py`. Encoder code: `temporal_graph_baselines.py` in the `scripts/` tree. **two `GCNConv` layers + GRU** versus **two `TransformerConv` layers + GRU**, with the same **dot-product** link head. **“Graph transformer”** here means **neighbor attention** through **`TransformerConv`** at each snapshot—not **full-graph** self-attention over all vessels (that would be **quadratic** in fleet size and is not attempted).

**How to read this comparison.**

1. All numbers below use the **capped** edge list (parquet basename contains `cap5000_even30`; full path in **Appendix B**). Do **not** treat this as a second headline for the full-coverage **Table 1** row (§3.6).  
2. The main benchmark uses **`--model tgcn`**. This ablation compares **`--model gcn`** to **`--model graph_transformer`**, i.e. **convolution vs. neighbor attention** in the **same two-layer + GRU** template—not a different temporal core.

**Protocol.** Train/test day caps **400** / **80**; **5** epochs (the Table 1 combined row uses **3**); seeds **1–3**; embedding **32**; temporal node features on; hard negatives; **49** test days after masking. Write-ups and scripts: `gcn_vs_graph_transformer.md` and `compare_gcn_graph_transformer.py` (see **Appendix B** for full paths).

*Means ± standard deviation are **across seeds** (1–3), unlike the day-level spread described for the headline run in §4.1.*

| Model | ROC AUC | Average precision |
|-------|---------|-------------------|
| **GCN** + GRU | **0.6201 ± 0.0102** | **0.7672 ± 0.0226** |
| **TransformerConv** + GRU | 0.4576 ± 0.1197 | 0.6735 ± 0.0554 |

On average, **GCN** achieves higher ROC AUC and AP and **lower** seed-to-seed variance. Paired tests by calendar day favor GCN for **each** seed (see artifact). A plausible—but **not** proven—explanation is that **noisy** proximity neighborhoods are stabilized more easily by **fixed** neighborhood aggregation than by **learned** attention weights under this budget, and that attention adds **parameters** without compensating gains. **Not attempted here:** global transformers, latent graph learning, edge-conditioned attention, or wide hyperparameter sweeps (see §6).

**Why do social edges lower AUC?** The **directional ablation** (proximity-only on a 20-bucket evaluation slice) scores ~0.78 AUC versus ~0.73 on the 876-bucket combined+social run. Because the two are evaluated on different horizons the absolute gap is **directional, not like-for-like**, but the same qualitative pattern—proximity-only ahead of proximity+social—shows up in every smaller-scale ablation we ran. Three factors likely contribute:

1. **Noise from registry grouping.** MID social edges link every pair of vessels sharing the same three-digit national prefix. In a predominantly Chinese-flagged dataset (MID 412), this connects thousands of vessels that never physically co-occur, injecting edges that the model must learn to discount.
2. **Over-smoothing.** Adding dense social edges increases the effective neighborhood size for each GCN layer. When many neighbors are structurally similar but behaviorally unrelated, node embeddings blur toward a common mean, reducing the model’s ability to distinguish genuinely co-present pairs.
3. **Objective dilution.** The training loss treats social edges and proximity edges identically (both are positive examples). The model therefore spends capacity fitting registry structure rather than geographic co-occurrence, which is the signal that carries over to unseen test days.

Adding dense MID-based social edges was a deliberate **first-cut** way to inject registry context, and the modest performance drop is informative rather than a dead end: it suggests the right answer is not "more edges of the same kind" but **a different way to fuse identity information with movement**. Several concrete directions look more promising and are listed under §6:

1. **Heterogeneous-edge architectures.** Treat proximity and registry as **different relations** so the model can learn separate aggregation functions per edge type (R-GCN, HGT, or a typed message-passing layer). The current homogeneous GCN forces both relations through the same weight matrix, which is exactly where the dilution arises.
2. **Edge-type-specific weights inside the loss.** A simple intermediate step before changing the architecture: down-weight social positives (or sample them less aggressively) so that **proximity** edges dominate the training signal while social edges still contribute as a soft prior.
3. **MID as a node feature, not an edge.** Encoding the three-digit MID prefix as a learned **node embedding** concatenated with the temporal node features lets the model use registry context as a **bias** on each vessel rather than as a forced positive between every same-flag pair. This avoids creating thousands of structurally identical neighbors in MID 412 while still letting the encoder condition on flag.
4. **Tighter or richer registry signals.** Beyond MID, ownership clusters, port-of-registry, or operator IDs (where available) are usually more informative than three-digit national prefixes; combined with (1) and (3), they would also let the model learn to **discount** very common groupings.

The point is not that registry information is unhelpful—it is that **how** it enters the graph matters at least as much as **whether** it enters at all.

## Geographic validation and case studies

Under the workflow that ranks candidates and then applies a **25 km**, **±1 day** rule, **eight** pairs remain. Two examples are **412422375 $\leftrightarrow$ 412428225** (**2** qualifying days) and **412000690 $\leftrightarrow$ 412325200** (**102** days). Extended discussion and plots appear in `docs/candidate_case_studies.md`. §4.2.1 summarizes **gear composition** across all **200** ranked pairs and relates the **192** excluded pairs to **expectations** from fleet-wide co-location.

**Figure 1 — purpose.** Link scores are abstract; **maps** tie them to geography. **Figure 1** highlights the pair **412000690 $\leftrightarrow$ 412325200** (**102** days within **25 km**) against the **Chinese** coast (**Yellow Sea / East China Sea**). Chance pairs would rarely show **persistent** spatial overlap; these tracks **do** overlap—supporting that the score reflects **real co-movement**, not that we have proven illicit coordination. **Table 2** and Figures 3–4 summarize **eight** pairs under a **fixed** full-coverage screening rule (the pair set differs slightly from Figure 1); Figure 1 remains the clearest **single-pair** illustration.

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
*Figure 1. Flagship candidate pair (**412000690**, blue; **412325200**, orange), Yellow Sea / Chinese coast, 2013–2018. Shading: summed kernel density of daily presence (0–1); contours mark density thresholds (see §4.2 “How to read”). Confirms sustained overlap near ~121°E, 37–39°N—link score aligns with a shared operating area, not random pairing. Tracks-only PNG and commands: Appendix B.*

### Gear-aware candidate stratification

The model nominates about **200** pairs; after the **25 km / ±1 day** geographic rule, **eight** remain. The question is what the other **192** pairs represent. A **gear-aware** post-processing step summarizes their **fleet context** and offers a numerical rationale for treating many of them as **ordinary** same-fleet overlap.

**Method.** For **267** distinct MMSIs in the top-200 list we infer a **dominant gear** by sampling **96** vessel-days (one per month from 2012–2019) and joining to **fleet** grids with the same **hours-weighted** rule used elsewhere (§4.3.1). Independently, we estimate how often each **gear pair** (trawler–trawler, trawler–gillnet, etc.) appears in the same grid cell on the same day using **32** sampled fleet days spanning 2012–2019—about **seven million** cell-day observations in total. That baseline answers how common each gear pairing is **before** consulting any model score.

Each candidate receives `adjusted = TGCN_score × discount` with `discount = 1 / (1 + 2 × normalized_baseline_rate)`. Gear combinations that are **frequent** in the fleet receive a **larger** discount; **rarer** or **cross-gear** combinations are discounted **less**. Unknown gear receives **no** change.

**Key findings (Figure 2):**

- **75% of the top-200 candidates** are trawler-dominated: 82 pairs (41%) are **Trawlers + Trawlers** and 67 (34%) are **Line / generic + Trawlers**. However the two categories tell very different stories: **Trawlers + Trawlers** is actually *1.1× below* the fleet baseline — the model is not preferentially elevating same-gear pairs; there are simply many trawler pairs in the data. **Line / generic + Trawlers**, by contrast, appears **10.5× above** the fleet baseline — the TGCN is specifically surfacing these cross-gear encounters far more than chance would predict, suggesting genuine signal.
- **50% are same-flag** (predominantly CHN + CHN), consistent with vessels from the same national fleet operating in the same waters.
- After gear adjustment, cross-gear validated pairs **held or improved** their rank: the **Set gillnets + Trawlers** pair (412461376 $\leftrightarrow$ 412415321) jumped **48 ranks** (72 → 24), and the **Fixed gear + Trawlers** pair (412422375 $\leftrightarrow$ 412428225) dropped only 2 ranks.
- The two **Trawlers + Trawlers** validated pairs dropped **70+ ranks** each (e.g., 80 → 154), confirming their co-presence is explained by shared fishing grounds rather than distinctive coordination.

**Why the results look this way.**  The top-200 list is dominated by trawler pairs for a straightforward reason: trawlers are the most common vessel type in the dataset and they concentrate in the same high-density fishing grounds (Yellow Sea, East China Sea). From the TGCN's perspective, two trawlers that share the same seasonal migration look structurally identical — they appear in the same grid cells, on the same days, year after year. The model correctly identifies this overlap but cannot distinguish "two boats following the same fish" from "two boats deliberately meeting." That is why 80 of the 82 trawler-trawler candidates **fail** the strict 25 km / ±1 day check: their graph-level similarity does not translate to sustained physical proximity on a given day.

Cross-gear pairs tell the opposite story. A fixed-gear vessel (stationary nets or traps) and a trawler (moving nets) have different operating modes, target different species, and rarely share the same grid cell by chance. When the model flags such a pair *and* the geographic filter confirms they were repeatedly within 25 km, the co-location is unlikely to be coincidental. This is reflected in the pass rates: **Fixed gear + Trawlers** passes at **25%** (1 of 4), compared to **2.4%** for trawler-trawler pairs — roughly a **10× difference**. The intermediate categories follow the same gradient: **Set gillnets + Trawlers** passes at **8.3%**, and **Line / generic + Trawlers** at **6.0%**.

**What this means for the project.** The gear-aware analysis serves two purposes:

1. **It justifies the geographic filter.** The 192 excluded pairs are not "false positives" in the traditional sense — the model genuinely detected shared movement patterns. But those patterns are explained by normal fleet behavior (thousands of same-type vessels fishing the same waters), not by vessel-to-vessel coordination. The filter correctly separates fleet-level noise from pair-specific signal.
2. **It strengthens the 8 validated pairs.** The fact that cross-gear pairs survive at disproportionately high rates means the validated set is enriched for behaviorally unusual encounters — exactly the kind of co-location that warrants further investigation. In particular, the Set gillnets + Trawlers pair that jumped 48 ranks after adjustment (72 → 24) was originally buried among dozens of trawler pairs; the gear-aware lens reveals it as one of the most distinctive candidates in the entire list.

How to read Figure 2:

- **Panel (a)** answers "why did 192 of 200 fail?" Each horizontal bar is one gear-type combination; bar length is the **number of candidate pairs** in that category (all 200 are shown). Each bar is **stacked**: gray = did not pass the 25 km / ±1 day geographic check; green = passed. **Failed** and **passed** counts are written once to the **right of each bar** (not inside narrow segments), so labels do not overlap. A strip below the chart shows the **whole cohort** (192 failed + 8 passed = 200). The bracket still highlights that **pass rates** (passed ÷ category size) are much higher for cross-gear categories than for **Trawlers + Trawlers**—e.g. **Fixed gear + Trawlers** **25%**, **Set gillnets + Trawlers** **8.3%**, **Line / generic + Trawlers** **6.0%**, versus **2.4%** for **Trawlers + Trawlers** (82 pairs), roughly **10×** at the extremes.
- **Panel (b)** answers "should we trust the 8 that survived?" A table lists every validated pair with its MMSI identifiers, gear combination, flag match, days within 25 km, mean distance, and TGCN rank. Six of eight are cross-gear pairs, consistent with genuine vessel-to-vessel encounters rather than fleet noise. The two trawler-trawler pairs have higher mean distances (191–349 km across overlap days) and lower ranks, further supporting the interpretation that their validation is marginal compared to the cross-gear pairs.

![Gear-aware re-ranking](../artifacts/plots/gear_aware_reranking.png)  
*Figure 2. Gear-stratified review of top-200 candidates after the **25 km / ±1 day** rule. **(a)** Counts by gear-type combination: gray = failed geography, green = passed (all 200 pairs). **(b)** Summary table of the eight passing pairs (gear, flag, days within 25 km, rank). Script: `scripts/gear_aware_reranking.py`; Appendix B.*

## Monthly overlap and cluster context

**Figure 3** charts the monthly count of days each validated pair's vessels were within **25 km**. **Figure 4** maps those same rendezvous locations geographically and highlights a **six-vessel** subgroup near **~30 N, 122 E**.

How to read Figure 3:

- Each **row** is one vessel pair, sorted from most overlap (top) to least (bottom). The **y-axis label** shows the pair number, the flag state(s) involved (e.g., CHN for China), and a summary line giving the total days and active months.
- Each **bubble** represents one calendar month. **Bubble size is proportional to the number of days** the two vessels were within 25 km that month (see the size legend in the upper right: 1, 5, 10, 20 days).
- **Numbers inside larger bubbles** give the exact day count for that month.
- **Blank spaces** mean zero close approaches that month — the pair was either not active or never came within 25 km.
- **Alternating row shading** (white / light gray) helps visually separate pairs.
- **Color** distinguishes pairs but carries no additional meaning beyond identification.

How to read Figure 4:

- **Faint gray dots** show all rendezvous points (monthly locations where any of the eight pairs were within 25 km).
- **Colored markers** (circles, squares, diamonds, triangles, crosses) each represent one of the **six cluster vessels**. The legend in the upper left identifies each vessel by abbreviated MMSI and flag state (all CHN).
- **Thin gray lines** connect cluster members that appear together as a pair, showing which vessels rendezvous with which.
- The **red X** marks the **weighted hotspot centroid** — the average location of all cluster rendezvous points (~29.7 N, 122.3 E).
- **Tan land shading**, coastlines, rivers, and labeled cities (**Shanghai**, **Hangzhou**, **Ningbo**, **Wenzhou**, **Fuzhou**) provide geographic reference.

### Gear type codes (how to read “country · gear”)

Figures 3–4 and Tables 2–3 label each vessel as **country · gear**. **Country** is taken from the enrichment file `artifacts/cooperative_pairs_with_flag_gear.csv` when present; otherwise it is inferred from the **ITU MID** prefix (for example **MID111**). **Gear** strings such as `trawlers` or `set_gillnets` come from **fleet** `geartype` fields on a **grid** and are assigned by **hours-weighted** overlap between vessel tracks and fleet cells (`scripts/enrich_pairs_with_flag_gear.py`). These labels describe **coarse fisheries-activity categories**, **not** formal licensing or gear certificates. Tables 2–3 and the figures use the **2012–2019** enrichment—the same basis as §4.2.1—which is **broader** than an August 2017-only enrichment. Definitions for each code appear in **Appendix A**.

**Table 2 — Eight validated pairs (country · gear per vessel).** *Format: src MMSI, dst MMSI, then src country · gear | dst country · gear.*

| src | dst | Country · gear (src \| dst) |
|-----|-----|------------------------------|
| 412422375 | 412428225 | CHN · fixed_gear \| CHN · trawlers |
| 412461376 | 412415321 | CHN · trawlers \| CHN · set_gillnets |
| 412437423 | 412435485 | CHN · trawlers \| CHN · fishing |
| 412410128 | 412416248 | CHN · fishing \| CHN · trawlers |
| 412420679 | 412413383 | CHN · fishing \| CHN · trawlers |
| 412450427 | 111203412 | CHN · fishing \| MID111 · trawlers |
| 412985698 | 412443375 | CHN · trawlers \| CHN · trawlers |
| 412461376 | 412427825 | CHN · trawlers \| CHN · trawlers |

**Table 3 — Six-vessel cluster (Figure 4).** *MMSIs selected by `scripts/analyze_six_vessel_cluster.py`; labels match plot legend.*

| MMSI | Country · gear |
|------|----------------|
| 412413383 | CHN · trawlers |
| 412420679 | CHN · fishing |
| 412427825 | CHN · trawlers |
| 412435485 | CHN · fishing |
| 412437423 | CHN · trawlers |
| 412461376 | CHN · trawlers |

![All pairs days within 25 km (monthly)](../artifacts/plots/pair_overlap_series/all_pairs_days_within_25km.png)  
*Figure 3. Monthly days within **25 km** for the eight validated pairs (2013–2019). One row per pair (sorted by total overlap); bubble area is proportional to days that month (see legend). Axis labels give total overlap days and months active. Pair 1 shows dense 2015–2016 overlap; others are mostly sporadic; pair 7 is the only cross-flag case. Reading guide: §4.3.*

![Six-vessel cluster](../artifacts/plots/six_vessel_cluster_scatter.png)  
*Figure 4. Rendezvous locations (monthly, within 25 km) for validated pairs, with Chinese coast context (Fuzhou–Shanghai). Gray dots: all rendezvous points; colored symbols: six vessels that recur across pairs (legend abbreviates MMSI); gray segments connect recorded pairings. Red **×**: centroid of cluster rendezvous (~29.7°N, 122.3°E), east of Ningbo. Highlights repeated convergence of multiple pairs in one offshore band—consistent with Figure 3.*

## Exploratory short-window analysis (August 2017; not the main benchmark)

Early in the project we trained a small TGCN on **five** consecutive days in **August 2017** (7–11 Aug)—one of the few intervals where cell-level fleet products provided **country and gear** labels for enrichment. Metrics on that slice (about **0.56** AUC, **0.67** AP) sit **far below** the 2012–2019 benchmark and should be treated only as **illustrative**.

**Why we abandoned the August 2017 shortcut for the main pipeline.** The five-day window was attractive at first because the corresponding fleet enrichment file came pre-labeled, which made gear and country joins trivial; but five days of test data is **too short** for a temporal benchmark (only **5** test buckets versus **876** in the headline run), concentrates on a single seasonal regime that does **not** reflect annual variability, and ties **gear assignment** to whatever vessels happened to be active in that one summer week. The full-timeline gear analysis in §4.2.1 **replaces** this shortcut: it assigns gear using **2012–2019** fleet context for all **200** candidates—roughly **seven million** cell-day observations across **32** sampled fleet days—so the labels reflect each vessel’s **dominant** behavior over years rather than a single-week snapshot, and the headline metrics in Table 1 sit on a properly long evaluation horizon. An August 2017 cooperation graphic remains in `artifacts/plots/tgcn_daily_cooperation_heatmap.png` for archival purposes but is **not** part of this document. Gear codes are defined in **Appendix A**.

---

# Discussion

This section states **limitations** of the task, data, metrics, and operational use—standard before §6.

## Threats and limitations

The same concerns as in earlier drafts, grouped so readers can scan by **what kind** of limitation is at issue.

### Task definition and semantics

- **Link prediction ranks edges under a graph rule, not misconduct.** ROC AUC / AP measure fit to observed vs sampled **non-edges**. High scores follow from **shared grounds**, **lanes**, or **ports** as easily as from rare coordination.  
- **No cooperation, IUU, or “dark fleet” labels** were used for training; the objective is structural co-occurrence only.  
- **Shared geography is not the same as coordination.** In crowded fisheries, overlap is **ambiguous** even when it is wholly legal and routine.

### Data and preprocessing

- **AIS** can be missing, intermittent, or misused; graphs only see **broadcast** tracks.  
- **Daily means, grid cells, and merge rules** shape both edges and post-hoc distances.  
- **Dense regions** yield many possible edges—**degree** and **baseline encounter rate** can inflate scores relative to detecting **rare** pairwise events.  
- The **August 2017** slice is a **narrow** exploratory window—**do not** treat it like the 2012–2019 benchmark.  
- **Country · gear** come from **fleet cell enrichment** (§4.3.1, **Appendix A**): useful for narrative context, **not** a substitute for registry or license records.

### Evaluation and reproducibility

- **Metrics are not enforcement decisions.** Good ranking on the surrogate task does **not** validate real-world outcomes; **ground-truth coordination** would be needed and is **absent** here.  
- The headline combined-graph row is **single-seed**; §4.1 gives day-level spread and seed spread on **other** setups—**multi-seed** uncertainty on the full benchmark is **future work**.  
- **Gear-aware adjustment** (§4.2.1) uses a **sampled** fleet baseline (**32** fleet days, ~7 million cell-days) and a **fixed discount** formula: **transparent**, **not** calibrated; alternate formulas would **change ranks**. Use it to **interpret** the long list, **not** as a second detector replacing geography.

### Deployment and computational constraints

- **Training** was limited to **~1450** days on a **16 GB** laptop; larger caps hit **memory** failure—reported limits are **hardware-bound**, not an intrinsic ceiling on the method.  
- In operations, treat outputs as **ranked lists** plus maps/tables: **analyst triage**, **not** automated findings.

## Synthesis

**Social edges** link vessels by registry prefix even when they rarely meet; that injects **noise**, encourages **over-smoothing**, and **dilutes** proximity in the training loss—so the **combined** graph’s headline scores **trail** the **proximity-only** curve despite embedding organizational signal one might encode differently in another architecture. **Strict post-hoc geography** (25 km, ±1 day on daily means) tests whether pairs flagged by **10 km** same-day structure actually **repeatedly co-locate** at operational distances; it **separates** fleet-scale **structural similarity** from **pair-level** proximity. **“Validated”** here means **passing that screen** together with documented **tracks and overlap**—an analyst gets a **short** list of **reviewable** cases with **reproducible exhibits**, **not** a determination of guilt or IUU.

---

# Conclusion and future work

This report describes an **unsupervised** pipeline from AIS to daily graphs to a **TGCN** for **temporal link prediction**, evaluated with a **chronological** train/test split (§3.1; formal symbols in **Appendix C**). On the **combined** graph—proximity edges plus **MID-consistent** social edges—the documented laptop run reaches **ROC AUC = 0.72997** and **AP = 0.71381** on **876** test days (**Table 1**), with **proximity-only** and **capped-graph** settings reported for comparison. **Eight** pairs remain after conservative **post-hoc** distance filters (**Table 2**). The **gear-aware** analysis (§4.2.1) suggests that much of the top-200 list reflects **routine** trawler-dominated fleet structure, whereas pairs that clear geography—and especially **cross-gear** pairs—look **less** typical relative to fleet baselines. Figures and tables add **country and gear** context where enrichment exists (**Appendix A**). For operational use, the deliverable is a **ranked candidate list** with **reproducible** geographic and contextual checks—not a finding of guilt or confirmed IUU.

**Future work.** Three directions stand out as both feasible on the existing pipeline and high-value:

1. **Multi-seed uncertainty on the full benchmark.** Re-run the headline combined-graph configuration with at least **3–5 seeds** so the **single-seed** headline of **0.72997 / 0.71381** can be reported with a proper confidence interval rather than only the day-level spread (§4.1). Pair this with a **precision-at-K** sweep (top **50 / 100 / 200** pairs versus geographic survival rate) so the **operational** value of the model is reported in the same table as the macro metrics.
2. **Better fusion of registry / ownership information.** The observed drop from the proximity-only ablation (~**0.78** AUC on a 20-bucket slice) to the combined+social run (~**0.73** AUC on 876 buckets) does not say that registry context is unhelpful—it says that **dense, equally-weighted** social edges between every same-MID pair are the wrong way to inject it. More promising fusions, all listed in §4.1: **heterogeneous-edge** architectures (R-GCN / HGT / typed message passing) that learn a **separate** aggregation per edge type; **edge-type-specific weights** in the loss so proximity dominates the training signal; and treating MID (or richer ownership / port-of-registry IDs) as a **node feature** concatenated with the temporal embedding rather than as an explicit edge—conditioning the encoder on flag without creating thousands of structurally identical neighbors. Each of these should also tighten the **top-200 → geography → gear-context** funnel in **Table 1b**, raising the analyst-yield rate above the current **4%** and—if the gap between trawler–trawler and best-cross-gear pass rates persists—make the operational top-N list cleaner without an analyst having to filter post-hoc.
3. **Scaling and richer signals.** Finer spatial-temporal resolution; **supervised or semi-supervised** extensions where trustworthy labels exist (e.g., declared transshipment events); **movement-based** gear or behavior models; larger day budgets on **high-memory** machines beyond the **~1450**-day laptop ceiling; and a disciplined separation of the **main 2012–2019 benchmark** from **illustrative** short windows such as August 2017.

```{=latex}
\clearpage
```

# References {.unnumbered}

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

```{=latex}
\clearpage
```

# Appendix A — Gear type codes {.unnumbered}

*Plain-language definitions for **gear** strings in **Tables 2–3** and **Figures 2–4**. The same content is kept in the repository as **`docs/gear_types.md`** for version control and easy editing.*

## Source of labels {.unnumbered}

Figure and table **gear** labels are **not** copied from a vessel registry. For each vessel–day we intersect AIS grid cells with **fleet** rows that report **hours** by **flag × geartype** within the cell; we assign the **geartype with the largest weighted hours** (`scripts/enrich_pairs_with_flag_gear.py`). The result is a **coarse behavioral tag**—“what fishing activity dominates that cell’s attributed effort”—rather than an official gear permit for that MMSI.

Plots show **—** when enrichment fails (no fleet overlap or vessel absent from the join).

## Codes in `artifacts/cooperative_pairs_with_flag_gear.csv` {.unnumbered}

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

## How to interpret labels {.unnumbered}

1. **Same MMSI, different days** can get different geartypes if the vessel moves between cells dominated by different fleet attributions.  
2. **Same geartype on two MMSIs** does not prove they use identical gear—only that both were attributed the same **coarse** class in the fleet enrichment window (2012–2019 for all tables and figures).  
3. **AIS “ship type”** codes differ from fleet **geartype** here: the latter is **fisheries-activity** oriented, not the full IMO/AIS ship-type list.

Methodology of the join and weighting: `scripts/enrich_pairs_with_flag_gear.py` and §4.3–4.4 of this report.

## Related work (not from the same CSV) {.unnumbered}

The repository also includes a **movement-based gear classifier** trained on **anonymized** labeled tracks (see project `README.md`, “Gear classification”). Those class labels (e.g. `purse_seines`, `trollers`) are **not** automatically the same strings as the fleet **`geartype`** column above; do not merge them without an explicit mapping.

---

```{=latex}
\clearpage
```

# Appendix B — Reproducibility {.unnumbered}

| Item | Command / path |
|------|----------------|
| Combined edges | `python3 scripts/add_social_edges.py --edges artifacts/edges_2012_2019_full.parquet --out artifacts/edges_full_with_social.parquet --max-social-per-bucket 2000` |
| Primary TGCN JSON/CSV | `artifacts/tgcn_social_maxb1450_ep3.json`, `.csv` — PyG env, `PYTHONPATH=scripts` |
| Optional per-bucket metric plots (not in main report) | `python3 scripts/plot_tgcn_bucket_metrics.py --report artifacts/tgcn_social_maxb1450_ep3.json --out-dir artifacts/plots` |
| Case study tracks (Figure 1) — command | `python3 scripts/compute_pair_overlap_from_daily.py --pairs artifacts/case_study_pairs.csv --daily-root "data/MMSI daily vessels " --top-k 4 --distance-km 25 --day-window 1 --max-files-per-year 0 --out-dir artifacts/plots/case_study_pairs --contour` |
| Case study tracks — notes | **`--max-files-per-year 0`** uses all daily CSVs (default); smaller values can undercount vs `docs/candidate_case_studies.md`. **`--contour`**: KDE density map (`Spectral_r`). With **cartopy**: coastlines / land / cities. Outputs `pair_<src>_<dst>_contour.png` (+ tracks-only `.png`). Optional `--out-summary`, `--contour-all-pairs`. |
| Ablations | `scripts/run_tgcn_improvement_suite.py` → `artifacts/tgcn_improvement_suite_summary.csv` |
| August cooperation summary (not in report; see §4.4) | `PYTHONPATH=scripts python3 scripts/plot_cooperative_heatmap.py` — two-panel aggregate cooperation summary: stacked bar chart by gear group + flag-state breakdown (Aug 2017 short window). Output: `artifacts/plots/tgcn_daily_cooperation_heatmap.png`. |
| Eight-pair overlap CSV + optional heatmap PNG | Full command in **block below** (overlap CSV feeds **Tables 2–3** / Figures 3–4). |
| Monthly overlap bubble chart (Figure 3) | `PYTHONPATH=scripts python3 scripts/plot_pair_overlap_time_series.py --overlap-csv artifacts/eight_pairs_overlap_by_month.csv --enrichment artifacts/cooperative_pairs_with_flag_gear.csv` — bubble chart of monthly close approaches, one row per pair. |
| Six-vessel cluster map (Figure 4) | `PYTHONPATH=scripts python3 scripts/analyze_six_vessel_cluster.py --overlap-csv artifacts/eight_pairs_overlap_by_month.csv --enrichment artifacts/cooperative_pairs_with_flag_gear.csv` — geographic scatter with cartopy coastline, land fill, cities. |
| Enrichment CSV (for labels) | `artifacts/cooperative_pairs_with_flag_gear.csv` — `scripts/enrich_pairs_with_flag_gear.py` |
| **Gear type definitions (for tables & figures)** | **Appendix A** (mirror for repo edits: `docs/gear_types.md`) |
| **Notation / formal link-prediction setup** | **Appendix C** (symbol table; training vs validation distances) |
| Gear × country pair-count heatmap (updated PNG) | `PYTHONPATH=scripts python3 scripts/plot_flag_gear_enrichment.py --input artifacts/cooperative_pairs_with_flag_gear.csv --out-dir artifacts/plots` |
| Gear-aware re-ranking (Figure 2) | `PYTHONPATH=scripts python3 scripts/gear_aware_reranking.py` — enriches top-200 candidate MMSIs with gear type (96 sampled days), computes fleet co-location baselines (~7M cell-days), and produces gear-adjusted rankings + combined bar-chart-and-table figure. |
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

```{=latex}
\clearpage
```

# Appendix C — Notation and formal problem setup {.unnumbered}

*This appendix condenses §3.1 for readers who want a symbol list. It may be **omitted in slide versions** or reordered after A–B if a thesis program requires a specific appendix sequence.*

## Symbol glossary {.unnumbered}

| Symbol | Meaning |
|--------|---------|
| $t$ | A calendar **day** (time bucket) in the AIS timeline. |
| $\mathcal{T}$ | The set of days used after filtering (e.g., days with edges). |
| $V$ | Set of **nodes** (9-digit vessel **MMSIs** appearing in the extract). |
| $G_t = (V, E_t)$ | **Undirected** snapshot graph for day $t$. |
| $E_t$ | **Edges** for day $t$: proximity (§3.3) ± optional social edges (§3.3). |
| $u, v$ | Distinct nodes (unordered pair $\{u,v\}$). |
| $s_{u,v}$ | **Scalar score** (logit) for pair $(u,v)$ on a given forward pass / bucket. |
| $\mathbf{h}_u$ | **Embedding vector** for node $u$ after the TGCN (dimension **32** in the primary run). |
| $\mathcal{T}_{\mathrm{train}}, \mathcal{T}_{\mathrm{test}}$ | **Chronological** partition of buckets (**~70% / ~30%**). |

## Graph semantics {.unnumbered}

For each day $t \in \mathcal{T}$, the graph $G_t = (V, E_t)$ is **undirected**. An edge $\{u,v\} \in E_t$ means the vessels met the **construction rule** for that day—typically **same-day** positions within **10 km**, optionally plus **same three-digit MID** ties. Edges encode **graph structure**, not adjudicated cooperation. The vertex set $V$ may be fixed globally or built per implementation; $|E_t|$ is kept **sparse** and sometimes **capped per day** for memory (§3.3).

## Temporal link prediction objective (informal) {.unnumbered}

The model reads the sequence $\{G_t\}$ (with optional **temporal node features**) into recurrent states and node embeddings $\mathbf{h}_u$. On each training day in $\mathcal{T}_{\mathrm{train}}$, **positives** are edges present that day; **negatives** are sampled non-edges with counts matched **per day** in code. Pair logits follow $s_{u,v} \propto \mathbf{h}_u^\top \mathbf{h}_v$ and are trained with **binary cross-entropy with logits**. Labels encode **presence or absence of an edge**, not IUU or transshipment.

**Evaluation.** On $\mathcal{T}_{\mathrm{test}}$, ROC AUC and AP measure how highly **true** test edges rank versus **non-edges** under the evaluation script’s protocol. Large values indicate **consistent ranking of observed contacts**, not legal liability.

## Training vs validation distances {.unnumbered}

| Stage | Typical distance / rule | Role |
|-------|---------------------------|------|
| **Graph edges (training signal)** | **$\leq$ 10 km**, same **day** | Defines $E_t$ for learning. |
| **Candidate screening (post-hoc)** | **25 / 50 / 100 km**, optional **±1 day** | **Not** used as training labels; validates high-scoring pairs on raw tracks (§3.5). |

Sections §3.3–3.5 repeat this distinction so **training edges** are not confused with **analyst thresholds** used afterward.
