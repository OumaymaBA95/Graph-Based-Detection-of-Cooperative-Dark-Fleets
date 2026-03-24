# Graph-Based Detection of Cooperative Dark Fleets  

---

### Slide 1 – Title & Agenda 
"Good [morning/afternoon]. I'll present my senior project titled Graph-Based Detection of Cooperative Dark Fleets  

The agenda: why this matters, how we prepared the data, the method, results, real examples with pictures, and what's next. Let's begin."

---

### Slide 2 – Motivation 

"IUU fishing — Illegal, Unreported, Unregulated — costs tens of billions and damages the ocean. A key pattern is **coordination at sea**: mother ships receiving fish from smaller boats (transshipment), fleets sharing fishing spots, and boats turning off their AIS to go dark.

**AIS** is the Automatic Identification System: ships broadcast position every few seconds. Criminals switch it off.

The challenge: we have **no labeled examples** of cooperating pairs. So we used an **unsupervised** approach — the computer finds patterns on its own."

---

### Slide 3 – Data Pipeline 

"We used public AIS data from 2012–2019: **2.38 billion rows** — one boat, one day, latitude and longitude. We added sea-surface temperature (97.5% coverage) for quality checks.

The pipeline: merge files → split by year → SST lookup → QC → build the graph.

The **graph**: imagine a social network where each boat is a person. Each day we draw a line between two boats if they were within 10 km. We do this for 8 years — 876 daily snapshots — so we see the network change over time."

---

### Slide 4 – Methods: Graph Construction 

"Each boat ID (MMSI) is a **node**. An **edge** exists only if two boats were close that day. We measure distance with the Haversine formula — standard math for curved Earth distance in km."

---

### Slide 5 – Methods: TGCN Model 

"The model is a **Temporal Graph Convolutional Network** (TGCN).

- **Graph Convolutional**: each day, the AI looks at each boat's neighbors and mixes their information.
- **Temporal**: it remembers past days using a GRU — a memory unit. So it learns patterns like "these two boats keep showing up near each other."

We also feed it four features per boat: how many meetings, how many partners, days since last seen, and average gap between meetings. That helps distinguish active boats from inactive ones.

The task is **link prediction**: the model scores pairs it never saw in training. Higher score = more likely to belong together."

---

### Slide 6 – Methods: Evaluation 

"We train on past years and test on future years — realistic, since you can't use tomorrow's data today.

**ROC AUC** answers: if I pick one real close pair and one random pair, how often does the model rank the real one higher? 0.5 = guessing. 1.0 = perfect.

We also run **5-fold cross-validation** — five different time cutoffs, averaged — to confirm stability."

---

### Slide 7 – Results: Performance (lead with the headline)
"**Main result:** On the **combined** proximity + same-MID social graph—full 2012–2019 time split, **1450** training buckets on a laptop—we get ROC AUC **about 0.73** and average precision **about 0.71** over **876** test buckets. Artifacts are in the repo as `tgcn_social_maxb1450_ep3.json`.

**Separately:** on **proximity-only** full coverage, about **0.78** AUC. On a **smaller capped** graph, about **0.95**; with **5-fold CV**, **0.93 ± 0.07**. Those are different graph sizes and splits—don't compare them to the headline without saying so."

---

### Slide 7b – Results: Figures (optional; 30 seconds)
"The **timeline and heatmap** slides are a **short August 2017** window—for **storytelling**, not the headline number. The **histogram and line plot** of per-bucket AUC show how stable the model is across **876** test buckets for the **full** combined-graph run. **Table** in the report is the authority."

---

### Slide 8 – Results: Validation 
"We took the top 1,000 pairs and checked them against the raw GPS data at three levels: 25 km ±1 day (strong), 50 km ±3 days, 100 km ±7 days (looser).

Only **8 pairs** passed the strictest 25 km test. That's what we want — it filters out boats that are merely in the same region."

---

### Slide 8a – Flag and gear (optional context)

"We also enriched **top August 2017** TGCN candidates by joining **MMSI-daily** positions to **cell-level fleet** tables—about **one-third** of pair-rows share the same **Maritime Identification Digit** on both vessels, and we see common combinations like **trawler–trawler** vs **trawler–fishing**. Gear is hours-weighted in each grid cell, not a formal registry—it's context for analysts. Details: `artifacts/flag_gear_enrichment_summary.md`."

---

### Slide 9 – Understanding the Numbers 
"On the next slides you'll see: **proximity days** (e.g. 102) — how many days we actually found the pair within 25 km in the GPS data — and **candidate score** (e.g. 15.92) — the model's confidence. Higher is stronger evidence."

---

### Slide 10–13 – Case Studies (4 min)
[Show each graph slide]

"**Case Study 1:** 412000690 and 412325200. Score 15.92. 102 days within 25 km, 410 within 50 km. [Show plot] The two lines travel together for years — our strongest candidate.

**Case Study 2:** 8 days within 25 km — intermittent proximity, possible transshipment or shared grounds.

**Case Study 3:** 9 days within 25 km — occasional meetings, useful for geographic diversity.

**Case Study 4 (contrast):** 2 days within 25 km but 864 within 100 km — likely same region, not real meetings. Shows why we use multiple thresholds."

---

### Slide 14 – Limitations 

"Limitations: daily positions only, so brief meetings can be missed. Closeness is a proxy, not proof of illegality. Results depend on our distance thresholds."

---

### Slide 15 – Conclusion 

"We have a reproducible pipeline that turns billions of ship positions into a short list of candidate pairs for investigation. Next steps: finer resolution, vessel metadata, and supervised learning if labels become available.

Thank you. Questions?"
