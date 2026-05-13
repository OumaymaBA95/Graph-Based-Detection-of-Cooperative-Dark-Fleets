# Graph-Based Detection of Cooperative Dark Fleets

## What this project is
This is a senior project that analyzes AIS vessel movement data (2012-2019) to find pairs of vessels that may be moving together in unusual ways.

In simple terms:  we turn vessel movement into a graph, run a temporal graph model, and rank the vessel pairs that look most suspicious for further review.

---
## Why this matters

Illegal or hidden coordinated fishing behavior is hard to detect manually.  
This project helps narrow millions of records into a short list of candidate vessel pairs with supporting evidence.

---

## What the pipeline does
1. Build daily vessel interaction graphs from AIS proximity.
2. Train a temporal graph model (TGCN-style link prediction).
3. Score and rank likely vessel pairs.
4. Validate top pairs using distance/time overlap checks.
5. Export tables/plots used in the final report and presentation.

This is a **screening tool** for analysts, not a legal decision system.

---
## Main files to know
- `docs/final_report.md` - full write-up
- `scripts/build_temporal_graph_baseline.py` - builds graph edges
- `scripts/run_tgcn_time_multiseed.py` - trains/evaluates temporal model
- `scripts/score_tgcn_candidates.py` - scores and ranks candidate pairs
- `docs/candidate_case_studies.md` - case-study summaries

---
## Quick setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
