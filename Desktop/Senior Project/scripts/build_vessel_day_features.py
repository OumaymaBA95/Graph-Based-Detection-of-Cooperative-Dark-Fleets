#!/usr/bin/env python3
"""
Build per-vessel, per-day SST feature summaries from sst_by_year parquet chunks.

Outputs columns:
- MMSI, day, sst_mean, sst_std, sst_min, sst_max, sst_count,
  sst_missing_count, sst_missing_rate, lat_mean, lon_mean, total_count,
  distance_km_total, distance_km_mean, speed_mean, speed_std, speed_max,
  stop_rate
"""
import argparse
import math
from pathlib import Path
from typing import Optional, List
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


def _choose_files(files: List[str], max_files: Optional[int], sampling: str, seed: int) -> List[str]:
    if not files:
        return []
    if max_files is None or max_files <= 0 or max_files >= len(files):
        return files
    if sampling == 'random':
        rng = np.random.default_rng(seed)
        return sorted(rng.choice(files, size=max_files, replace=False).tolist())
    step = (len(files) - 1) / max_files
    idx = [round(i * step) for i in range(max_files)]
    return [files[i] for i in idx]


def aggregate_chunk(df: pd.DataFrame, stop_kmh: float) -> pd.DataFrame:
    df = df.copy()
    df['day'] = pd.to_datetime(df['timestamp']).dt.floor('D')

    df = df.sort_values(['MMSI', 'day', 'timestamp'])
    df['next_lat'] = df.groupby(['MMSI', 'day'])['lat'].shift(-1)
    df['next_lon'] = df.groupby(['MMSI', 'day'])['lon'].shift(-1)
    df['next_time'] = df.groupby(['MMSI', 'day'])['timestamp'].shift(-1)
    df['dt_hours'] = (df['next_time'] - df['timestamp']).dt.total_seconds() / 3600.0
    valid_move = df['dt_hours'].gt(0) & df['next_lat'].notna() & df['next_lon'].notna()
    df['segment_km'] = 0.0
    df.loc[valid_move, 'segment_km'] = haversine_km(
        df.loc[valid_move, 'lat'],
        df.loc[valid_move, 'lon'],
        df.loc[valid_move, 'next_lat'],
        df.loc[valid_move, 'next_lon'],
    )
    df['speed_kmh'] = 0.0
    df.loc[valid_move, 'speed_kmh'] = df.loc[valid_move, 'segment_km'] / df.loc[valid_move, 'dt_hours']
    df['is_stop'] = False
    df.loc[valid_move, 'is_stop'] = df.loc[valid_move, 'speed_kmh'] <= stop_kmh

    total = df.groupby(['MMSI', 'day']).agg(
        total_count=('MMSI', 'size'),
        lat_sum=('lat', 'sum'),
        lon_sum=('lon', 'sum'),
    )

    movement = df[valid_move].groupby(['MMSI', 'day']).agg(
        distance_sum=('segment_km', 'sum'),
        distance_count=('segment_km', 'size'),
        speed_sum=('speed_kmh', 'sum'),
        speed_sumsq=('speed_kmh', lambda x: (x * x).sum()),
        speed_max=('speed_kmh', 'max'),
        stop_count=('is_stop', 'sum'),
    )

    missing_mask = df['sst_missing'] | df['sst'].isna()
    missing = df[missing_mask].groupby(['MMSI', 'day']).size().rename('missing_count')

    valid = df[~missing_mask].groupby(['MMSI', 'day']).agg(
        sst_count=('sst', 'size'),
        sst_sum=('sst', 'sum'),
        sst_sumsq=('sst', lambda x: (x * x).sum()),
        sst_min=('sst', 'min'),
        sst_max=('sst', 'max'),
    )

    out = total.join(missing, how='left').join(valid, how='left').join(movement, how='left')
    out['missing_count'] = out['missing_count'].fillna(0).astype(int)
    out['sst_count'] = out['sst_count'].fillna(0).astype(int)
    out['sst_sum'] = out['sst_sum'].fillna(0.0)
    out['sst_sumsq'] = out['sst_sumsq'].fillna(0.0)
    out['distance_sum'] = out['distance_sum'].fillna(0.0)
    out['distance_count'] = out['distance_count'].fillna(0).astype(int)
    out['speed_sum'] = out['speed_sum'].fillna(0.0)
    out['speed_sumsq'] = out['speed_sumsq'].fillna(0.0)
    out['speed_max'] = out['speed_max'].fillna(0.0)
    out['stop_count'] = out['stop_count'].fillna(0).astype(int)
    return out


def merge_aggregates(agg_a: Optional[pd.DataFrame], agg_b: pd.DataFrame) -> pd.DataFrame:
    if agg_a is None:
        return agg_b
    combined = pd.concat([agg_a, agg_b])
    combined = combined.groupby(['MMSI', 'day']).agg(
        total_count=('total_count', 'sum'),
        missing_count=('missing_count', 'sum'),
        sst_count=('sst_count', 'sum'),
        sst_sum=('sst_sum', 'sum'),
        sst_sumsq=('sst_sumsq', 'sum'),
        sst_min=('sst_min', 'min'),
        sst_max=('sst_max', 'max'),
        lat_sum=('lat_sum', 'sum'),
        lon_sum=('lon_sum', 'sum'),
        distance_sum=('distance_sum', 'sum'),
        distance_count=('distance_count', 'sum'),
        speed_sum=('speed_sum', 'sum'),
        speed_sumsq=('speed_sumsq', 'sum'),
        speed_max=('speed_max', 'max'),
        stop_count=('stop_count', 'sum'),
    )
    return combined


def finalize_features(agg: pd.DataFrame) -> pd.DataFrame:
    out = agg.copy().reset_index()
    out['sst_mean'] = out['sst_sum'] / out['sst_count'].replace({0: np.nan})
    out['sst_var'] = (out['sst_sumsq'] / out['sst_count'].replace({0: np.nan})) - (out['sst_mean'] ** 2)
    out['sst_var'] = out['sst_var'].clip(lower=0)
    out['sst_std'] = np.sqrt(out['sst_var'])
    out['sst_missing_rate'] = out['missing_count'] / out['total_count'].replace({0: np.nan})
    out['lat_mean'] = out['lat_sum'] / out['total_count'].replace({0: np.nan})
    out['lon_mean'] = out['lon_sum'] / out['total_count'].replace({0: np.nan})
    out['distance_km_total'] = out['distance_sum']
    out['distance_km_mean'] = out['distance_sum'] / out['distance_count'].replace({0: np.nan})
    out['speed_mean'] = out['speed_sum'] / out['distance_count'].replace({0: np.nan})
    out['speed_var'] = (out['speed_sumsq'] / out['distance_count'].replace({0: np.nan})) - (out['speed_mean'] ** 2)
    out['speed_var'] = out['speed_var'].clip(lower=0)
    out['speed_std'] = np.sqrt(out['speed_var'])
    out['stop_rate'] = out['stop_count'] / out['distance_count'].replace({0: np.nan})
    return out[
        [
            'MMSI', 'day', 'sst_mean', 'sst_std', 'sst_min', 'sst_max',
            'sst_count', 'missing_count', 'sst_missing_rate',
            'lat_mean', 'lon_mean', 'total_count',
            'distance_km_total', 'distance_km_mean',
            'speed_mean', 'speed_std', 'speed_max', 'stop_rate', 'distance_count'
        ]
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sst-root', default='data/sst_by_year')
    ap.add_argument('--years', default='2018', help='Comma-separated years to process')
    ap.add_argument('--out-root', default='data/features_by_year')
    ap.add_argument('--max-files-per-year', type=int, default=0, help='Limit files per year (0 = all)')
    ap.add_argument('--file-sampling', choices=['even', 'random'], default='even')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-rows-per-file', type=int, default=0, help='Limit rows per file (0 = all)')
    ap.add_argument('--stop-kmh', type=float, default=1.0, help='Speed threshold for stop rate')
    args = ap.parse_args()

    years = [y.strip() for y in args.years.split(',') if y.strip()]
    for y in years:
        year_dir = Path(args.sst_root) / y
        if not year_dir.exists():
            print(f"Skip missing year dir: {year_dir}")
            continue
        files = sorted(glob(str(year_dir / '*.parquet')))
        files = _choose_files(files, args.max_files_per_year, args.file_sampling, args.seed)
        if not files:
            print(f"No parquet files found for {y}")
            continue

        agg = None
        for fpath in files:
            if args.max_rows_per_file > 0:
                df = pd.read_parquet(fpath)
                df = df.head(args.max_rows_per_file)
            else:
                df = pd.read_parquet(fpath)
            agg = merge_aggregates(agg, aggregate_chunk(df, args.stop_kmh))

        features = finalize_features(agg)
        out_dir = Path(args.out_root) / y
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / 'vessel_day_features.parquet'
        features.to_parquet(out_path, index=False)
        print(f"Wrote features: {out_path} ({len(features)} rows)")


if __name__ == '__main__':
    main()
