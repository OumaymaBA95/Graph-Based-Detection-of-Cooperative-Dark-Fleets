# Graph-Based Detection of Cooperative Dark Fleets

This project analyzes AIS vessel movement data (2012-2019) to identify vessel pairs that may be operating in coordinated ways.  

It uses temporal graph learning to rank candidate pairs, then validates those candidates with geographic overlap checks.

## Project goal

This repository is for **screening and prioritization**:
- detect potentially coordinated vessel pairs from large AIS data
- rank candidates for analyst review
- provide supporting evidence (overlap summaries, plots, case studies)

It is **not** a legal decision system.

## What the pipeline does
1. Build temporal vessel graphs from AIS proximity  
2. Train/evaluate temporal graph models (TGCN-style link prediction)  
3. Score and rank likely vessel pairs  
4. Validate top pairs with distance/time overlap checks  
5. Produce report and presentation outputs  

## Key files

### Documentation
- `docs/final_report.md` (main report source)
- `docs/candidate_case_studies.md`
- `docs/gear_types.md`
- `docs/presentation.md`
- `docs/presentation_script.md`

### Core scripts
- `scripts/build_temporal_graph_baseline.py`
- `scripts/run_tgcn_time_multiseed.py`
- `scripts/score_tgcn_candidates.py`
- `scripts/compute_pair_overlap_from_daily.py`
- `scripts/plot_pair_overlap_time_series.py`
- `scripts/analyze_six_vessel_cluster.py`
- `scripts/add_social_edges.py`
- `scripts/enrich_pairs_with_flag_gear.py`
- `scripts/summarize_flag_gear_enrichment.py`

### Build scripts
- `build_final_report_pdf.sh`
- `build_presentation_pdf.sh`
- `build_docs_pdfs.sh`
- `build_tex.sh`

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
