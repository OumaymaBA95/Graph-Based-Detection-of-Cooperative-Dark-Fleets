#!/usr/bin/env python3
"""
Build per-node temporal interaction features from an edge list with time buckets.

Outputs columns:
- MMSI, interactions_count, unique_partners, last_seen_days, mean_gap_days
"""
import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def load_edges_with_time(path: Path, years: List[int]) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if 'src' not in df.columns or 'dst' not in df.columns or 'time_bucket' not in df.columns:
        raise ValueError('Edge file must contain src, dst, and time_bucket columns')
    df = df[['src', 'dst', 'time_bucket']].dropna()
    df['time_bucket'] = pd.to_datetime(df['time_bucket'])
    if years:
        df = df[df['time_bucket'].dt.year.isin(years)]
    return df[['src', 'dst', 'time_bucket']].astype({'src': int, 'dst': int})


def build_temporal_node_features(train_df: pd.DataFrame, cutoff_time) -> pd.DataFrame:
    edges = train_df[['src', 'dst', 'time_bucket']].copy()

    left = edges[['src', 'dst', 'time_bucket']].rename(columns={'src': 'MMSI', 'dst': 'partner'})
    right = edges[['dst', 'src', 'time_bucket']].rename(columns={'dst': 'MMSI', 'src': 'partner'})
    long = pd.concat([left, right], ignore_index=True)

    interaction_counts = long.groupby('MMSI').size().rename('interactions_count')
    partner_counts = long.groupby('MMSI')['partner'].nunique().rename('unique_partners')

    last_seen = long.groupby('MMSI')['time_bucket'].max()
    last_seen_days = (pd.to_datetime(cutoff_time) - last_seen).dt.total_seconds() / 86400.0
    last_seen_days = last_seen_days.rename('last_seen_days')

    def mean_gap(series: pd.Series) -> float:
        times = np.sort(series.unique())
        if len(times) < 2:
            return np.nan
        diffs = np.diff(times) / np.timedelta64(1, 'D')
        return float(np.mean(diffs))

    mean_gap_days = long.groupby('MMSI')['time_bucket'].apply(mean_gap).rename('mean_gap_days')

    feats = pd.concat([interaction_counts, partner_counts, last_seen_days, mean_gap_days], axis=1).reset_index()
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--edges', required=True)
    ap.add_argument('--years', default='2012,2013,2014,2015,2016,2017,2018,2019')
    ap.add_argument('--test-ratio', type=float, default=0.3)
    ap.add_argument('--out', default='data/temporal_features/node_temporal_features.parquet')
    args = ap.parse_args()

    years = [int(y.strip()) for y in args.years.split(',') if y.strip()]
    df = load_edges_with_time(Path(args.edges), years)
    buckets = sorted(df['time_bucket'].dropna().unique())
    if len(buckets) < 2:
        raise ValueError('Need at least 2 time buckets for a time-based split.')
    n_test = max(1, int(len(buckets) * args.test_ratio))
    split_idx = max(1, len(buckets) - n_test)
    cutoff = buckets[split_idx]

    train_df = df[df['time_bucket'].isin(set(buckets[:split_idx]))]
    feats = build_temporal_node_features(train_df, cutoff)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out_path, index=False)
    print(f"Wrote temporal features: {out_path} ({len(feats)} rows)")


if __name__ == '__main__':
    main()
