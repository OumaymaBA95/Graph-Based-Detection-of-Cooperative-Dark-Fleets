#!/usr/bin/env python3
"""
Build a small temporal co-movement edge list for baseline modeling.

Modes:
- Real data: stitch multiple years from `data/by_year/combined_<YEAR>.csv` with a per-year row cap.
- Synthetic (optional): only for smoke tests; default is real data.

Process:
- Reads columns: mmsi, time, latitude, longitude (time parsable by pandas).
- Buckets by time (default 1H), caps rows per bucket, and creates pairwise edges when
    great-circle distance <= --distance-km.
- Outputs edges to Parquet: src, dst, time_bucket, distance_km.
"""
import argparse
import math
import os
import random
from pathlib import Path
from typing import Optional, List
import math as pymath
from glob import glob

import numpy as np
import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def build_edges(df: pd.DataFrame, time_floor: str, max_per_bucket: int, distance_km: float):
    # bucket time
    df = df.copy()
    df['time_bucket'] = df['time'].dt.floor(time_floor)
    edges = []
    # process each bucket with capped rows to avoid O(n^2)
    for bucket, g in df.groupby('time_bucket'):
        if len(g) < 2:
            continue
        # cap rows in bucket for tractability
        if max_per_bucket and len(g) > max_per_bucket:
            g = g.sample(max_per_bucket, random_state=42)
        g = g.reset_index(drop=True)
        coords = g[['latitude', 'longitude']].to_numpy()
        mmsi = g['mmsi'].to_numpy()
        n = len(g)
        # pairwise upper triangle
        for i in range(n):
            lat1, lon1 = coords[i]
            for j in range(i + 1, n):
                if mmsi[i] == mmsi[j]:
                    continue
                lat2, lon2 = coords[j]
                dist = haversine_km(lat1, lon1, lat2, lon2)
                if dist <= distance_km:
                    edges.append((int(mmsi[i]), int(mmsi[j]), bucket, float(dist)))
    if not edges:
        return pd.DataFrame(columns=['src', 'dst', 'time_bucket', 'distance_km'])
    return pd.DataFrame(edges, columns=['src', 'dst', 'time_bucket', 'distance_km'])


def make_synthetic(n_vessels: int = 50, n_times: int = 10, jitter_km: float = 5.0):
    # place vessels on a grid, move slightly per time step
    base_lats = np.linspace(-5, 5, int(math.sqrt(n_vessels)))
    base_lons = np.linspace(30, 40, int(math.sqrt(n_vessels)))
    pts = [(lat, lon) for lat in base_lats for lon in base_lons]
    pts = pts[:n_vessels]
    rows = []
    for t in range(n_times):
        for vid, (lat, lon) in enumerate(pts):
            lat_j = lat + np.random.normal(scale=jitter_km / 110.0)  # approx deg per km
            lon_j = lon + np.random.normal(scale=jitter_km / 110.0)
            rows.append({
                'mmsi': 100000 + vid,
                'time': pd.Timestamp('2018-01-01') + pd.Timedelta(hours=t),
                'latitude': lat_j,
                'longitude': lon_j,
            })
    return pd.DataFrame(rows)


def _choose_files(files: List[str], max_files: Optional[int], sampling: str, seed: int) -> List[str]:
    if not files:
        return []
    if max_files is None or max_files <= 0 or max_files >= len(files):
        return files
    if sampling == 'random':
        rng = random.Random(seed)
        return sorted(rng.sample(files, max_files))
    # even spacing (default)
    step = (len(files) - 1) / max_files
    idx = [round(i * step) for i in range(max_files)]
    return [files[i] for i in idx]


def read_year_sample_mmsi_daily(
    base_dir: Path,
    years: List[int],
    rows_per_year: int,
    max_files_per_year: Optional[int],
    file_sampling: str,
    seed: int,
) -> pd.DataFrame:
    """Read per-day MMSI grid files, sampling up to rows_per_year per year across multiple days."""
    frames = []
    for y in years:
        year_frames = []
        pattern = str(base_dir / f"mmsi-daily-csvs-10-v3-{y}-*.csv")
        files = sorted(glob(pattern))
        files = _choose_files(files, max_files_per_year, file_sampling, seed)
        full_read = rows_per_year <= 0
        per_file = max(1, int(pymath.ceil(rows_per_year / max(1, len(files)))))
        taken = 0
        for fpath in files:
            if not full_read and taken >= rows_per_year:
                break
            if full_read:
                df_chunk = pd.read_csv(fpath, usecols=['date', 'cell_ll_lat', 'cell_ll_lon', 'mmsi'])
            else:
                need = min(per_file, rows_per_year - taken)
                df_chunk = pd.read_csv(fpath, nrows=need, usecols=['date', 'cell_ll_lat', 'cell_ll_lon', 'mmsi'])
            df_chunk = df_chunk.rename(columns={'date': 'time', 'cell_ll_lat': 'latitude', 'cell_ll_lon': 'longitude'})
            df_chunk['time'] = pd.to_datetime(df_chunk['time'])
            df_chunk = df_chunk.dropna(subset=['mmsi', 'time', 'latitude', 'longitude'])
            year_frames.append(df_chunk)
            taken += len(df_chunk)
        if year_frames:
            year_df = pd.concat(year_frames, ignore_index=True)
            frames.append(year_df)
            print(f"Loaded {len(year_df)} rows for {y} from {len(year_frames)} files")
        else:
            print(f"No files found for year {y} with pattern {pattern}")
    if not frames:
        raise FileNotFoundError("No MMSI daily files were loaded; check paths/years.")
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='2018', help='Comma-separated years to stitch (e.g., 2012,2013,2014)')
    ap.add_argument('--mmsi-daily-dir', default='data/MMSI daily vessels ', help='Directory with mmsi-daily-csvs-10-v3-<year>-*.csv')
    ap.add_argument('--rows-per-year', type=int, default=20000)
    ap.add_argument('--max-files-per-year', type=int, default=12, help='Max daily files to sample per year (<=0 uses all)')
    ap.add_argument('--file-sampling', choices=['even', 'random'], default='even')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-edges', default='artifacts/edges_baseline.parquet')
    ap.add_argument('--time-floor', default='1D', help='Time bucket (default 1D since inputs are daily)')
    ap.add_argument('--max-per-bucket', type=int, default=300)
    ap.add_argument('--distance-km', type=float, default=10.0)
    ap.add_argument('--synthetic', action='store_true', help='Use synthetic sample instead of reading CSVs')
    args = ap.parse_args()

    os.makedirs(Path(args.out_edges).parent, exist_ok=True)

    if args.synthetic:
        df = make_synthetic(n_vessels=40, n_times=8, jitter_km=3.0)
    else:
        years = [int(y.strip()) for y in args.years.split(',') if y.strip()]
        df = read_year_sample_mmsi_daily(
            Path(args.mmsi_daily_dir),
            years,
            args.rows_per_year,
            args.max_files_per_year,
            args.file_sampling,
            args.seed,
        )

    edges_df = build_edges(df, args.time_floor, args.max_per_bucket, args.distance_km)
    edges_df.to_parquet(args.out_edges, index=False)
    print(f"Wrote edges: {len(edges_df)} rows to {args.out_edges}")
    if len(edges_df):
        print(edges_df.head())


if __name__ == '__main__':
    main()
