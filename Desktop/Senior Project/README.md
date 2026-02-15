- # Graph-Based Detection of Cooperative Dark Fleets

## Abstract

This repository contains the data-processing and modeling pipeline for a senior research project that aims to detect cooperative "dark" fishing activity. The core idea is to combine vessel movement data with sea-surface temperature (SST) time-series as node attributes, and to use a Graph Auto‑Encoder (GAE) to reconstruct edges and surface likely missing links or bridge vessels.

The pipeline is designed for reproducibility: memory-safe streaming of raw CSVs, vectorized SST extraction from local zarr stores, and a targeted remote fallback (WMTS/ERDDAP) to fill remaining missing SST values.

## Features / implemented components

- Memory-safe combiner for fleet-daily CSVs (supports CSVs inside ZIP archives). Output: `data/combined_fleet_daily_full.csv`.
- Streaming split by year: `data/by_year/combined_<YEAR>.csv`.
- Local gridded data handling: per-year Zarr stores under `data/glorys_by_year/` (if present).
- Vectorized SST extraction: `scripts/extract_sst_vectorized.py` — groups by dataset time index, interpolates via xarray/zarr, applies a nearest-time tolerance and nearest-grid fallback, and writes chunked Parquet outputs under `data/sst_by_year/<year>/`.
- Parallel controller: `scripts/run_parallel_years.py` to run extractors per year with worker control and `--years` filtering.
- QC/status reporting: `scripts/status_report.py` — aggregates parquet outputs, reports missing counts, and produces lightweight SST statistics via reservoir sampling.
- WMTS fallback helper: `scripts/wmts_fallback.py` — simulate mode produces `data/sst_wmts_sample/plan_simulate.json`; sample mode (live) supports a throttled pilot to query Copernicus WMTS GetFeatureInfo for SST when local zarrs are masked or missing.

## Project status (recent results)

- Combined CSV rebuilt for years 2012–2019: `data/combined_fleet_daily_full.csv` (≈1.99B rows written during rebuild).
- Per-year CSV splits: `data/by_year/combined_<YEAR>.csv` for 2012–2019.
- Per-year parquet SST outputs exist under `data/sst_by_year/<year>/` (vectorized extractor runs completed for available years).
- QC (recent run): total rows ≈ 2,383,037,575; SST present ≈ 2,322,638,065; SST missing ≈ 60,399,510 (≈2.53%).

## Reproducibility & environment

We aim for a minimal, reproducible Python environment. Example (macOS / zsh):

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install pandas numpy xarray zarr netCDF4 pyarrow requests tqdm scipy pyproj geopandas rtree
```

Modeling (GNN) dependencies such as PyTorch and `torch-geometric` are environment-specific; install them following their official instructions for your platform.

## Typical workflows and commands

1) Produce the combined CSV (streaming combiner):

```bash
python3 data_loading_exploration.py --combine --overwrite --years 2012 2019
```

2) Split by year (if needed):

```bash
python3 scripts/split_by_year.py --input data/combined_fleet_daily_full.csv --out-dir data/by_year --chunksize 200000
```

3) Run vectorized SST extraction across a selection of years:

```bash
python3 scripts/run_parallel_years.py --by-year-dir data/by_year --zarr-dir data/glorys_by_year --out-root data/sst_by_year --workers 16 --chunk-size 50000 --time-tolerance-days 31 --years 2012,2013,2014
```

4) Run QC/status report:

```bash
python3 scripts/status_report.py --root data/sst_by_year --sample-size 200000
```

## WMTS fallback (recommended procedure)

The WMTS fallback addresses residual missing SSTs after vectorized extraction. Recommended steps:

1. Run simulate mode to generate a stratified sample plan (no network calls):

```bash
python3 scripts/wmts_fallback.py --mode simulate --sample-size 100000
```

2. Inspect `data/sst_wmts_sample/plan_simulate.json` and a few example URLs.

3. Run a conservative live pilot (e.g., 5,000 requests) to measure fill rate and latency:

```bash
python3 scripts/wmts_fallback.py --mode sample --sample-size 5000
```

4. Tune batch size / concurrency / throttle and run the full targeted fallback if the pilot is successful.

The script tries unauthenticated access first; if Copernicus requires credentials, the script reads them from environment variables (see `scripts/wmts_fallback.py` docstring).

## Modeling & experiments (next stage)

After SST extraction and QC:

1. Construct temporal graphs (daily / 6-hour snapshots) where nodes = vessels and edges = co-movement (distance/time thresholds).
2. Create node features from SST sequences (summary statistics, trend features, and time-series embeddings).
3. Train a Graph Auto‑Encoder (GAE) to reconstruct adjacency; score candidate missing edges and aggregate to rank bridge-vessel candidates.
4. Validate using held-out synthetic experiments, manual case studies, and external event overlays (GFW, SAR).

## Data provenance & licensing

- Keep provenance columns for any SST fill (fields such as `sst_source`, `request_url`, `time_tolerance_days`). This allows downstream experiments to exclude or weight fallback-filled values.
- Cite and follow license terms for datasets used (OceanBench, GLORYS, Copernicus, GFW) when publishing results.

## Contributing and contact

If you want to contribute code or data-handling improvements, open an issue or a PR. Include a short description and test data when possible. For questions or coordination, email or message the project owner (maintainer contact in repo metadata).

---

If you want, I can (a) print the simulated WMTS plan (`data/sst_wmts_sample/plan_simulate.json`), (b) re-run the 5k pilot (confirm unauthenticated), or (c) add an ERDDAP fallback option for faster numeric responses.

---

If you want, I can (a) print the simulated WMTS plan (`data/sst_wmts_sample/plan_simulate.json`), (b) re-run the 5k pilot (confirm unauthenticated), or (c) switch the fallback to a public ERDDAP dataset for faster numeric responses.