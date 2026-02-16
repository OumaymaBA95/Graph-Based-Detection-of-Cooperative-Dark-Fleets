#!/usr/bin/env python3
"""Chunked extraction of SST for the combined fleet CSV using per-year Zarrs.

This script reads the input CSV in chunks, groups rows by year (from timestamp column),
opens the corresponding yearly Zarr under --zarr-dir (default: data/glorys_by_year),
calls the `extract_sst_from_xarray` function (from scripts.extract_sst) to interpolate SST,
and writes parquet chunk files to --out-dir/<year>/chunk_<n>.parquet.

Usage example (test run on small sample):
  python3 scripts/extract_sst_by_year.py --input data/combined_fleet_daily_sample_500.csv --out-dir data/sst_by_year_test --zarr-dir data/glorys_by_year --chunk-size 200 --max-chunks 1

For full run, omit --max-chunks and provide a larger --chunk-size (e.g., 50000).
"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import xarray as xr
import os
import sys
from importlib import import_module

# Ensure repository root is on sys.path so we can import scripts.extract_sst
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
extract_module = import_module('scripts.extract_sst')
extract_fn = getattr(extract_module, 'extract_sst_from_xarray')
find_column = getattr(extract_module, 'find_column')


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def process_chunk(df_chunk, zarr_dir, out_dir, time_col, lat_col, lon_col, mmsi_col, sample_limit=None, chunks={'time':1,'latitude':512,'longitude':512}, time_tolerance_days: int = 31):
    # Ensure timestamp parsed
    df_chunk[time_col] = pd.to_datetime(df_chunk[time_col], errors='coerce')
    df_chunk = df_chunk.dropna(subset=[time_col, lat_col, lon_col])
    if df_chunk.empty:
        return []

    df_chunk['year'] = df_chunk[time_col].dt.year
    out_files = []
    years = df_chunk['year'].unique()
    for year in years:
        subset = df_chunk[df_chunk['year'] == year]
        zarr_path = Path(zarr_dir) / f'glorys_{int(year)}.zarr'
        if not zarr_path.exists():
            print('Zarr for year', year, 'not found at', zarr_path, '; writing NaN rows')
            # write NaN sst rows to parquet
            rows = []
            mmsi_field = mmsi_col or find_column(subset, ['MMSI','mmsi','mmsi_id','mmsi_present'])
            for rec in subset.to_dict('records'):
                rows.append((rec.get(mmsi_field), rec.get(time_col), float(rec.get(lat_col)), float(rec.get(lon_col)), np.nan))
            out_df = pd.DataFrame(rows, columns=['MMSI','timestamp','lat','lon','sst'])
            out_df['sst_missing'] = True
            year_out_dir = Path(out_dir) / str(year)
            ensure_dir(year_out_dir)
            out_file = year_out_dir / f'chunk_missing_{subset.index[0]}.parquet'
            out_df.to_parquet(out_file, engine='pyarrow', index=False)
            out_files.append(str(out_file))
            continue

        print('  Opening zarr for year', year, 'at', zarr_path)
        ds = xr.open_zarr(str(zarr_path), chunks=chunks)
        try:
            out_df = extract_fn(ds, subset, time_col, lat_col, lon_col, mmsi_col=mmsi_col, sample_limit=sample_limit, time_tolerance_days=time_tolerance_days)
        except Exception as e:
            print('  Extraction failed for year', year, e)
            ds.close()
            continue
        ds.close()

        year_out_dir = Path(out_dir) / str(year)
        ensure_dir(year_out_dir)
        out_file = year_out_dir / f'chunk_{subset.index[0]}.parquet'
        out_df.to_parquet(out_file, engine='pyarrow', index=False)
        out_files.append(str(out_file))
    return out_files


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--zarr-dir', default='data/glorys_by_year')
    p.add_argument('--time-col', default=None)
    p.add_argument('--lat-col', default=None)
    p.add_argument('--lon-col', default=None)
    p.add_argument('--mmsi-col', default=None)
    p.add_argument('--chunk-size', type=int, default=50000)
    p.add_argument('--max-chunks', type=int, default=None, help='If set, stop after this many chunks (useful for tests)')
    p.add_argument('--sample-limit', type=int, default=None, help='Pass to extractor to limit per-call samples (for debugging)')
    p.add_argument('--chunks', default="{'time':1,'latitude':512,'longitude':512}")
    p.add_argument('--time-tolerance-days', type=int, default=31, help='Nearest-time tolerance (days) passed to extractor')
    args = p.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print('Input CSV not found:', input_path)
        return

    chunks = eval(args.chunks)

    # Read first small portion to detect column names if not provided
    preview = pd.read_csv(input_path, nrows=50)
    time_col = args.time_col or find_column(preview, ['timestamp','time','datetime','date'])
    lat_col = args.lat_col or find_column(preview, ['lat','latitude','cell_ll_lat'])
    lon_col = args.lon_col or find_column(preview, ['lon','longitude','cell_ll_lon'])
    mmsi_col = args.mmsi_col or find_column(preview, ['MMSI','mmsi','mmsi_id','mmsi_present'])

    if not all([time_col, lat_col, lon_col]):
        print('Required columns not found in preview. Columns:', preview.columns.tolist())
        return

    print('Using columns:', time_col, lat_col, lon_col, 'mmsi:', mmsi_col)

    ensure_dir(args.out_dir)

    reader = pd.read_csv(input_path, parse_dates=[time_col], chunksize=args.chunk_size)
    chunk_i = 0
    total_written = 0
    for chunk in reader:
        print(f'Processing chunk #{chunk_i}, rows={len(chunk)}')
        out_files = process_chunk(chunk, args.zarr_dir, args.out_dir, time_col, lat_col, lon_col, mmsi_col, sample_limit=args.sample_limit, chunks=chunks, time_tolerance_days=args.time_tolerance_days)
        print('  Wrote', len(out_files), 'files for chunk', chunk_i)
        total_written += len(out_files)
        chunk_i += 1
        if args.max_chunks is not None and chunk_i >= args.max_chunks:
            print('Reached max_chunks, stopping')
            break

    print('Done. total parquet files written:', total_written)

if __name__ == '__main__':
    main()
