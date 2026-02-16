# WMTS Fallback Pilot Plan

Goal: run a small, throttled WMTS sampling to estimate fill-rate and latency before a full fallback.

## Preconditions
- Local SST parquet outputs under `data/sst_by_year/` with missing SST positions flagged.
- Sample plan from simulate mode: `data/sst_wmts_sample/plan_simulate.json` (run simulate if absent).
- Environment vars (only if Copernicus requires auth): `WMTS_USERNAME`, `WMTS_PASSWORD`.

## Pilot command (5k requests)
```bash
python3 scripts/wmts_fallback.py --mode sample --sample-size 5000 --throttle-ms 200 --max-concurrent 2 --out-dir data/sst_wmts_sample/pilot_5k
```
Adjust throttle/concurrency if you observe rate limits.

## What to record
- Fill rate: fraction of requests returning valid SST.
- Latency: p50/p90 request time.
- Error codes and retry counts.
- Whether authentication was required.

## Outputs
- Responses/parquet: `data/sst_wmts_sample/pilot_5k/` (written by the script).
- Logs: capture stdout/stderr; summarize fill-rate and latency in a short note.

## After the pilot
- If fill-rate is good and errors low: scale to full plan with cautious throttle.
- If rate-limited or low fill: switch endpoints (e.g., ERDDAP) or widen time/space tolerance.
