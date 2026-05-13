---
title: "Graph-Based Screening for Potentially Cooperative Vessel Pairs"
subtitle: "A temporal link prediction approach using AIS (2012–2019)"
output:
  powerpoint_presentation:
    slide_level: 2
---

## Agenda

1. Problem & motivation
2. Pipeline overview
3. Graph construction
4. **Deep dive: the TGCN model**
5. Results & validation
6. Gear-aware stratification
7. Case studies
8. Limitations & conclusion

::: notes
Walk through the agenda quickly — emphasize that the professor asked for a deep dive on the TGCN, which is section 4. Total talk is 20 minutes.
:::

# Problem & Motivation

## Why This Matters

- **IUU fishing** costs $10–23 B/year globally
- Enforcement needs tools to flag **coordinated vessel behavior**
- AIS gives billions of position records — but **cooperation is never labeled**
- No ground truth → **supervised learning is not possible**

**Our approach:** unsupervised screening pipeline — ranked candidate list for human review, not a verdict

::: notes
IUU = illegal, unreported, unregulated fishing. AIS = Automatic Identification System — vessels broadcast position, speed, heading. The key constraint is that nobody has labeled which vessel pairs are "cooperating" — so we can't train a classifier. Instead we treat it as temporal link prediction: if two vessels keep appearing near each other across time, the model gives them a high score.
:::

## What This Work Does — and Does Not Do

| Does | Does not |
|------|----------|
| Temporal link prediction on daily graphs | Prove coordination or IUU |
| Ranked candidate list with scores | Replace human analysis |
| Independent geographic validation | Claim outputs are ground truth |
| Gear-aware characterization of excluded pairs | |

::: notes
This slide manages expectations up front. The committee should understand that high model scores mean "plausible future co-occurrence" — not guilt. The output is a list for analysts to review, plus reproducible checks.
:::

# Pipeline Overview

## End-to-End Pipeline

![Pipeline](../artifacts/plots/pipeline_diagram.png)

::: notes
Walk through left to right: (1) We start with 2.38 billion vessel-day AIS records across 2012–2019. (2) Each calendar day becomes a snapshot graph — vessels are nodes, edges connect vessels within 10 km on the same day, plus optional social edges from shared national registry prefix. (3) The TGCN processes this sequence of daily graphs, producing 32-dimensional embeddings per vessel. (4) We score all vessel pairs by inner product of their embeddings — top 200 are candidates. (5) A strict 25 km geographic filter on raw tracks winnows to 8 validated pairs, and gear-aware stratification explains why the other 192 were excluded.
:::

# Graph Construction

## How We Build Daily Graphs

**Proximity edges** (training signal):

- Same calendar day, Haversine distance ≤ **10 km** between daily mean positions
- This defines $E_t$ — the **only** distance used for training

**Social edges** (optional augmentation):

- Same **3-digit ITU MID** prefix (e.g., 412 = China)
- Capped at **2,000 per day** to keep graphs tractable

**Critical distinction:**

| | Distance | Purpose |
|-|----------|---------|
| **Training** | ≤ 10 km, same day | Defines graph edges |
| **Validation** | 25 km, ±1 day | Post-hoc check on raw tracks |

::: notes
The most important thing for the committee to understand: training edges and validation distances are NEVER mixed. The model learns from 10 km same-day proximity. The 25 km check is applied AFTER training to validate candidates on raw AIS tracks — it's a completely independent filter. Stress this — it's what makes the validation credible.
:::

# Deep Dive: The TGCN Model

## What Is a TGCN?

Combines two ideas:

- **GCN** — learns from who is near whom on a **single day**
- **GRU** — learns how that structure **changes across days**

**Key insight:** Replace the GRU's linear layers with **graph convolutions** → the recurrence is *graph-aware*

Each vessel's hidden state is updated based on **its neighbors** in that day's graph

::: notes
A Temporal Graph Convolutional Network. The standard GRU processes sequences of vectors. The TGCN processes sequences of GRAPHS. Inside each GRU gate, instead of a matrix multiply, there's a graph convolution — so the model aggregates information from a vessel's neighbors before deciding how to update its hidden state. We use the TGCN class from PyTorch Geometric Temporal by Rozemberczki et al.
:::

## How GCNConv Works

For each vessel $u$, aggregate neighbor features:

$$\mathbf{h}_u^{\prime} = \sigma\!\left(\sum_{v \in \mathcal{N}(u) \cup \{u\}} \frac{1}{\sqrt{d_u \cdot d_v}} \, \mathbf{W} \, \mathbf{h}_v\right)$$

**In plain English:**

- Look at all vessels within 10 km today
- **Weighted average** of their features (inverse degree normalization)
- Multiply by **learnable weights** $\mathbf{W}$
- Result: each embedding encodes the **local neighborhood structure**

::: notes
This is the Kipf & Welling 2017 GCNConv. The normalization factor 1/sqrt(d_u * d_v) ensures high-degree nodes don't dominate — a vessel in a crowded fishing ground doesn't overwhelm the embedding just because it has many neighbors. The weight matrix W is learned during training. After one GCN layer, vessel u's embedding reflects not just its own features but who it's near. With multiple layers (or the recurrence), information propagates further through the graph.
:::

## Inside the TGCN Cell

![TGCN Architecture](../artifacts/plots/tgcn_architecture.png)

::: notes
Walk through the diagram top to bottom. INPUTS: X_t is the node feature matrix for today (dimension 5 per vessel — degree plus 4 temporal features). H_{t-1} is the hidden state carried from yesterday (32 dimensions per vessel). THREE GATES, each using GCNConv: (1) Update gate Z — sigmoid output between 0 and 1 — decides how much of yesterday's hidden state to keep. (2) Reset gate R — also sigmoid — decides how much of yesterday to forget before computing the new candidate. (3) Candidate state H-tilde — uses tanh — what today's graph says the hidden state should be, with yesterday partially reset. BLEND: The final hidden state H_t is a weighted mix: Z times yesterday plus (1-Z) times the candidate. If Z is close to 1, the model keeps yesterday's state (vessel behavior hasn't changed). If Z is close to 0, the model overwrites with today's graph information. The "Next day" arrow shows this H_t becomes H_{t-1} for tomorrow — that's the recurrence.
:::

## The Three Gates — Equations

**Update gate** — how much to keep from yesterday:

$$Z_t = \sigma\big(\text{Linear}[\text{GCNConv}(X_t) \| H_{t-1}]\big)$$

**Reset gate** — how much of yesterday to forget:

$$R_t = \sigma\big(\text{Linear}[\text{GCNConv}(X_t) \| H_{t-1}]\big)$$

**Candidate state** — what today's graph suggests:

$$\tilde{H}_t = \tanh\big(\text{Linear}[\text{GCNConv}(X_t) \| R_t \odot H_{t-1}]\big)$$

**Final hidden state** — blend:

$$H_t = Z_t \odot H_{t-1} + (1 - Z_t) \odot \tilde{H}_t$$

::: notes
This is the same diagram in equation form — reference the diagram as you walk through it. Each gate takes today's features through a GCNConv (so it's graph-aware), concatenates with yesterday's hidden state, then applies a linear layer and activation. Sigmoid for the gates (output between 0 and 1 — "how much"). Tanh for the candidate (output between -1 and 1 — "what direction"). The element-wise multiply (circle-dot) means each dimension of the 32-dim embedding is gated independently. The key intuition: if a vessel's neighborhood hasn't changed much, Z stays high and the hidden state carries forward. If the neighborhood changes dramatically (new co-occurrences), Z drops and the model incorporates today's graph.
:::

## Node Features (Input to the TGCN)

**5 features per vessel per day** (all z-score standardized):

1. **Degree** (daily) — neighbors within 10 km today
2. **Interaction count** (static) — total edges across training
3. **Unique partners** (static) — distinct co-occurring vessels
4. **Last seen** (static) — days since last appearance
5. **Mean gap** (static) — average interval between appearances

::: notes
These features tell the model about each vessel's behavioral profile. Degree captures today's activity level — it changes every day. The four static features are computed once from training history and capture long-term patterns: a vessel with high interaction count and many unique partners behaves very differently from one with few, sporadic contacts. Last-seen and mean-gap capture temporal regularity. All features are z-score standardized (mean 0, std 1) so no single feature dominates. The combined input is 5-dimensional per vessel per day.
:::

## Scoring: Embeddings → Pair Predictions

After processing day $t$, each vessel $u$ has embedding $\mathbf{h}_u \in \mathbb{R}^{32}$

**Pair score (inner product):**

$$s_{u,v} = \mathbf{h}_u^\top \mathbf{h}_v = \sum_{i=1}^{32} h_{u,i} \cdot h_{v,i}$$

- **High score** → similar neighborhood patterns across time → predicted co-occurrence
- **Low score** → no structural reason to expect co-occurrence

**Why inner product?** Simplest similarity measure — if embeddings point in the same direction, the dot product is large

::: notes
This is where the model makes predictions. After the TGCN processes a day's graph, every vessel has a 32-dimensional embedding vector. To score a pair, we just take the dot product — it's fast and interpretable. Two vessels with similar embeddings (similar neighborhoods, similar temporal patterns) get a high score. The score is a logit — it goes through BCEWithLogitsLoss during training, but at inference we just rank pairs by raw score. The top 200 pairs by score become our candidates.
:::

## Training Loop

**For each of 3 epochs**, iterate through **1,450 days in chronological order:**

1. Build today's graph $G_t$
2. Forward pass: $H_t = \text{TGCN}(X_t, G_t, H_{t-1})$
3. Score **positive pairs** (actual edges) and **negative pairs** (random non-edges, count-matched)
4. **Loss:** BCEWithLogitsLoss — positives should score high, negatives low
5. Backprop + Adam update (LR = 0.001)
6. **Detach** $H_t$ — truncated BPTT (laptop memory constraint)

**No cooperation labels** — only structural positives vs random negatives

::: notes
The training loop processes days in order — this is crucial because the hidden state carries temporal information forward. For each day, we have positive pairs (the actual edges in E_t, meaning vessels within 10 km) and we sample the same number of random non-edges as negatives. BCEWithLogitsLoss trains the model to rank positives above negatives. The key implementation detail is truncated backpropagation through time: after each day, we detach the hidden state from the computation graph. This means gradients only flow through one day at a time, not through the full 1,450-day sequence. We had to do this because the laptop has 16 GB RAM — without detaching, the computation graph would grow until memory runs out. The trade-off is that the model can't learn very long-range dependencies, but in practice the daily recurrence still captures meaningful temporal patterns.
:::

## Evaluation: What the Metrics Mean

**Chronological split:** train on days 1–1,450 / test on days 1,451–2,326

**ROC AUC = 0.73:**
If you pick a random real edge and a random non-edge, there's a **73% chance** the model ranks the real edge higher

**AP = 0.71:**
The precision-recall curve has area 0.71 — reasonable precision as recall increases

**Per-bucket variability:** std ±0.10 across 876 test days — some days are easier (dense graphs), others harder (sparse)

::: notes
The chronological split is important — we're not doing random train/test splitting, which would leak future information. The model trains on earlier days and is evaluated on later days, so the metrics measure genuine forecasting ability. 0.73 AUC is moderate — it means the model is meaningfully better than random (0.50) but far from perfect. This is expected for an unsupervised approach on a hard problem with no labels. The per-bucket standard deviation of 0.10 means some individual days score much higher or lower — the 0.73 is a macro-average. If the committee asks "is 0.73 good enough?" — the answer is that the model's purpose is screening, not classification. Even moderate ranking ability can surface useful candidates when paired with post-hoc validation.
:::

# Results & Validation

## Quantitative Performance (Table 1)

| Setting | ROC AUC | AP |
|---------|---------|-----|
| **Combined + social (primary)** | **0.730** | **0.714** |
| Proximity-only (ablation) | 0.776 | 0.795 |
| Capped graph (sanity check) | ~0.95 | ~0.96 |

::: notes
Three rows, three different experiments — don't mix metrics across them. The combined graph (our primary result) adds social edges from shared national registry prefix. Counter-intuitively, this lowers AUC compared to proximity-only. If the committee asks why: MID 412 (China) connects thousands of vessels into a massive clique — most never physically meet. Three effects: (1) Noise — registry edges that don't correspond to co-occurrence. (2) Over-smoothing — dense neighborhoods blur embeddings. (3) Objective dilution — BCE loss treats social edges identically to proximity edges. The capped graph is a much easier task — don't compare those numbers to the primary result. Future work: heterogeneous-edge architectures that weight edge types differently.
:::

## Geographic Validation: 8 of 200 Survive

![Figure 1 — Flagship pair](../artifacts/plots/case_study_pairs/pair_412000690_412325200_contour.png)

::: notes
Of the top 200 TGCN candidates, we applied a strict 25 km, plus-or-minus 1 day proximity filter on raw daily AIS tracks. Only 8 pairs survive. This figure shows the flagship pair — 412000690 and 412325200 — with 102 days within 25 km. Blue traces are one vessel, orange traces the other. The background heatmap shows combined daily presence density — warmer colors mean both vessels were frequently in that area. The hot spot in the Yellow Sea around 121 East, 37-39 North confirms the model's abstract high score corresponds to real, sustained co-presence. If the model were surfacing random pairs, we would not expect their tracks to overlap like this.
:::

# Gear-Aware Stratification

## Why Did 192 of 200 Candidates Fail?

**Three-step analysis:**

**Step 1 — Gear enrichment:** 267 MMSIs → dominant gear type from 96 sampled fleet days (2012–2019)

**Step 2 — Co-location baseline:** 32 fleet days → ~7M cell-days → how often each gear pair shares a cell **by default**

**Step 3 — Gear-adjusted score:**

$$S_{\text{adj}} = S_{\text{orig}} \times \frac{1}{1 + 2 \cdot \hat{B}}$$

High fleet co-location → larger discount; cross-gear pairs → less affected

::: notes
This is the post-hoc analysis that explains WHY only 8 survived. Step 1: we enrich each vessel with its dominant gear type by joining MMSI daily positions to cell-aggregated fleet data — 96 sampled days across 2012-2019 gives broad coverage. Step 2: separately, we compute a baseline of how often each gear combination shares the same grid cell in the fleet data — about 7 million cell-days. This tells us what's "normal." Step 3: we discount each candidate's TGCN score by their gear pair's baseline rate. The formula uses alpha=2 as the discount strength. Trawler-trawler pairs have high baselines (they're everywhere together), so they get a big discount. Cross-gear pairs like fixed gear plus trawlers have low baselines — their scores are barely affected.
:::

## The 10× Story

**75% of top 200** are trawler-dominated — they fish the same waters by default

**Pass rates tell the real story:**

- **Fixed gear + Trawlers:** 25% (1 of 4 pairs pass)
- **Set gillnets + Trawlers:** 8.3% (1 of 12)
- **Line / generic + Trawlers:** 6.0% (4 of 67)
- **Trawlers + Trawlers:** 2.4% (2 of 82)

**10× difference** between cross-gear and same-gear pass rates

::: notes
This is the key finding. Trawlers dominate the candidate list for a simple reason: they're the most common vessel type, they concentrate in the same fishing grounds, and two trawlers following the same seasonal fish migration look structurally identical to the TGCN. 80 of 82 trawler-trawler pairs fail the 25 km check — their graph similarity doesn't translate to sustained physical proximity on a given day. Cross-gear pairs are the opposite. A fixed-gear vessel (stationary nets or traps) and a trawler (mobile) rarely share the same grid cell by chance. When the model flags such a pair AND the geographic filter confirms repeated 25 km proximity, that co-location is unlikely to be coincidental. That's why fixed gear + trawlers passes at 25% — ten times the trawler-trawler rate.
:::

## Figure 2: All 200 Candidates (counts) + The 8 Validated Pairs

![Figure 2](../artifacts/plots/gear_aware_reranking.png)

::: notes
Panel (a) shows every one of the 200 candidates: each row is a gear combination, bar length is how many pairs fall in that bucket, stacked gray failed versus green passed the 25 km check. The strip at the bottom shows the whole cohort 192 plus 8 equals 200. You can see most of the mass is trawler-heavy categories where almost everyone failed. The bracket still calls out roughly 10x pass-rate difference versus trawler-trawler. Panel (b) is the table of the 8 validated pairs — MMSI identifiers, gear combination, whether they share a flag, days within 25 km, mean distance, and TGCN rank. Notice that 6 of 8 are cross-gear pairs. The two trawler-trawler pairs have higher mean distances (191 and 349 km) and lower ranks — they're the weakest of the 8. The Set gillnets + Trawlers pair jumped 48 ranks after gear adjustment (72 to 24) — originally buried among trawler pairs, revealed as one of the most distinctive candidates.
:::

## What This Means

**Two takeaways:**

1. **The geographic filter is justified** — 192 excluded pairs reflect **normal fleet behavior**, not model failure

2. **The 8 validated pairs are strengthened** — cross-gear pairs survive at disproportionate rates → enriched for **behaviorally unusual** encounters

::: notes
The gear-aware analysis serves as the justification layer. The 192 excluded pairs aren't false positives in the traditional sense — the model genuinely detected shared movement patterns. But those patterns are explained by thousands of same-type vessels fishing the same waters. The filter correctly separates fleet-level noise from pair-specific signal. The fact that cross-gear pairs survive at 10x the rate means our validated set is enriched for the exact kind of encounter that warrants investigation — two vessels with different operating modes repeatedly appearing in close proximity.
:::

# Case Studies

## Monthly Overlap (Figure 3)

![Figure 3](../artifacts/plots/pair_overlap_series/all_pairs_days_within_25km.png)

- **Pair 1:** 84 days across 9 months — **persistent** co-presence (2015–2016)
- Most others: **sporadic** — occasional, not sustained
- **Pair 7:** only cross-flag pair (CHN ↔ MID111)

::: notes
Each row is one validated pair, sorted by total overlap days. Bubble size is proportional to days within 25 km that month. Pair 1 dominates — large bubbles clustered in 2015-2016, showing persistent co-presence over an extended period. Most other pairs are sporadic — a handful of small bubbles scattered across years, suggesting occasional rather than sustained proximity. Pair 7 is the only one involving a non-Chinese vessel (MID111). The pattern differences matter: persistent co-presence (Pair 1) is more suggestive of coordination than occasional encounters.
:::

## Six-Vessel Cluster (Figure 4)

![Figure 4](../artifacts/plots/six_vessel_cluster_scatter.png)

- Six vessels from **multiple distinct pairs** converge on the **same area** east of Ningbo
- Concentrated between 29°N–30.5°N, ~122°E
- Consistent with a **shared operating region**

::: notes
This figure plots rendezvous locations for all eight validated pairs against the Chinese coast. The colored markers highlight six specific vessels that appear in multiple pairs — they form a cluster. The red X marks the weighted centroid at roughly 29.7 North, 122.3 East — about 50-80 km offshore from Ningbo in the East China Sea. The fact that vessels from different pairs converge on the same narrow band suggests a shared operating region rather than coincidence. This is consistent with the co-presence patterns in Figure 3.
:::

# Limitations & Conclusion

## Limitations

- **Link prediction ≠ verdict** — high scores can reflect shared fishing grounds, ports, traffic lanes
- **AIS gaps** — vessels may disable AIS or have sparse coverage
- **Confounding by density** — crowded regions produce many possible edges
- **Single-seed headline** — per-bucket std ±0.10; multi-seed runs are future work
- **Gear labels are coarse** — fleet-level categories, not formal licenses
- **Gear-adjusted score is a heuristic** — different discount functions would shift exact ranks

::: notes
Be upfront about these. The most important one: link prediction is not a verdict. High scores mean "plausible future co-occurrence under the graph definition" — not guilt. AIS compliance is another real issue — vessels can turn off their transponders, so we only see what's reported. The single-seed limitation is honest — we report per-bucket dispersion and cross-seed results on the capped graph, but full multi-seed CIs on the headline number await higher-memory hardware.
:::

## Conclusion

**What we built:**

- Unsupervised AIS → graph → TGCN pipeline with time-split evaluation
- **0.73 ROC AUC**, **0.71 AP** on 876 test days
- **8 pairs** survive strict 25 km post-hoc screening
- Gear-aware stratification: 75% of candidates are fleet noise; validated cross-gear pairs are **10× more likely** to survive

**The output:** a ranked candidate list + reproducible checks — not a determination of wrongdoing

::: notes
Summarize crisply. Emphasize: unsupervised (no labels), time-split (no leakage), reproducible (all code and commands in Appendix B). The gear-aware stratification is the analytical contribution that ties the whole story together — it explains WHY the filter works, not just THAT it works.
:::

## Future Work

- **Multi-seed runs** on high-memory hardware
- **Heterogeneous-edge architectures** for social edges (weight proximity ≠ registry)
- Finer **spatial/temporal** resolution
- **Owner or fleet metadata** from registries
- **Movement-based gear classifiers** integrated into the pipeline

::: notes
If the committee asks what you'd do with more time: multi-seed is the obvious next step to get confidence intervals on the headline number. Heterogeneous edges would address the social-edge AUC drop by letting the model treat proximity and registry edges differently. Finer resolution and registry metadata would improve both the model inputs and the post-hoc validation.
:::

## Thank You — Questions?

**Key takeaway:** The TGCN surfaces candidates from graph structure and time. Geographic validation and gear-aware stratification separate fleet noise from real signal, yielding 8 behaviorally distinctive pairs for review.

::: notes
Open the floor for questions. Have Figure 2 and the TGCN architecture diagram ready to flip back to — those are the two most likely reference slides during Q&A.
:::
