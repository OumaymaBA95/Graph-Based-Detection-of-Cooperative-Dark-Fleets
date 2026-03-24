#!/usr/bin/env python3
"""
Enrich candidate MMSI pairs with per-vessel activity summaries from vessel_day_features.
"""
import argparse
from pathlib import Path
from typing import List
import glob

import pandas as pd


def load_feature_files(root: Path, years: List[str]) -> List[str]:
    if years:
        files = []
        for y in years:
            fpath = root / y / 'vessel_day_features.parquet'
            if fpath.exists():
                files.append(str(fpath))
        return files
    return sorted(glob.glob(str(root / '*' / 'vessel_day_features.parquet')))


def aggregate_features(files: List[str]) -> pd.DataFrame:
    frames = []
    for fpath in files:
        df = pd.read_parquet(fpath)
        frames.append(df)
    if not frames:
        raise FileNotFoundError('No vessel_day_features.parquet files found.')

    data = pd.concat(frames, ignore_index=True)
    summary = data.groupby('MMSI').agg(
        days_count=('day', 'nunique'),
        total_count_sum=('total_count', 'sum'),
        distance_km_total_sum=('distance_km_total', 'sum'),
        sst_mean_mean=('sst_mean', 'mean'),
        speed_mean_mean=('speed_mean', 'mean'),
        stop_rate_mean=('stop_rate', 'mean'),
        lat_mean_mean=('lat_mean', 'mean'),
        lon_mean_mean=('lon_mean', 'mean'),
    ).reset_index()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', default='artifacts/tgcn_candidate_pairs_report.csv')
    ap.add_argument('--features-root', default='data/features_by_year')
    ap.add_argument('--years', default='', help='Comma-separated years to include (blank = all)')
    ap.add_argument('--out', default='artifacts/tgcn_candidate_pairs_enriched.csv')
    ap.add_argument('--top-k', type=int, default=0, help='Optional top-k rows from pairs input')
    args = ap.parse_args()

    years = [y.strip() for y in args.years.split(',') if y.strip()]
    feature_files = load_feature_files(Path(args.features_root), years)

    pairs = pd.read_csv(args.pairs)
    if args.top_k and args.top_k > 0:
        pairs = pairs.head(args.top_k)

    features = aggregate_features(feature_files)

    src_features = features.add_prefix('src_').rename(columns={'src_MMSI': 'src'})
    dst_features = features.add_prefix('dst_').rename(columns={'dst_MMSI': 'dst'})

    enriched = pairs.merge(src_features, on='src', how='left').merge(dst_features, on='dst', how='left')

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(out_path, index=False)
    print(f"Wrote enriched pairs to {out_path}")


if __name__ == '__main__':
    main()
