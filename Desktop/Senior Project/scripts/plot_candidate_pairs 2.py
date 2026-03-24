#!/usr/bin/env python3
"""
Plot daily mean tracks for candidate MMSI pairs using vessel_day_features and compute overlap stats.
"""
import argparse
from pathlib import Path
from typing import List
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return r * c


def load_features(root: Path, years: List[str]) -> pd.DataFrame:
    if years:
        files = [root / y / 'vessel_day_features.parquet' for y in years]
        files = [f for f in files if f.exists()]
    else:
        files = [Path(p) for p in glob.glob(str(root / '*' / 'vessel_day_features.parquet'))]
    if not files:
        raise FileNotFoundError('No vessel_day_features.parquet files found.')

    frames = [pd.read_parquet(f) for f in files]
    data = pd.concat(frames, ignore_index=True)
    data['day'] = pd.to_datetime(data['day'])
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', default='artifacts/tgcn_candidate_pairs_enriched_top50.csv')
    ap.add_argument('--features-root', default='data/features_by_year')
    ap.add_argument('--years', default='', help='Comma-separated years (blank = all)')
    ap.add_argument('--top-k', type=int, default=5)
    ap.add_argument('--distance-km', type=float, default=10.0)
    ap.add_argument('--out-dir', default='artifacts/plots/candidate_pairs')
    ap.add_argument('--out-summary', default='artifacts/candidate_pair_overlap_summary.csv')
    args = ap.parse_args()

    years = [y.strip() for y in args.years.split(',') if y.strip()]
    features = load_features(Path(args.features_root), years)

    pairs = pd.read_csv(args.pairs).head(args.top_k)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for _, row in pairs.iterrows():
        src = int(row['src'])
        dst = int(row['dst'])

        src_df = features[features['MMSI'] == src][['day', 'lat_mean', 'lon_mean']].dropna()
        dst_df = features[features['MMSI'] == dst][['day', 'lat_mean', 'lon_mean']].dropna()

        if src_df.empty or dst_df.empty:
            continue

        merged = src_df.merge(dst_df, on='day', suffixes=('_src', '_dst'))
        if not merged.empty:
            merged['distance_km'] = haversine_km(
                merged['lat_mean_src'],
                merged['lon_mean_src'],
                merged['lat_mean_dst'],
                merged['lon_mean_dst'],
            )
            close_days = (merged['distance_km'] <= args.distance_km).sum()
            mean_distance = merged['distance_km'].mean()
        else:
            close_days = 0
            mean_distance = np.nan

        summary_rows.append({
            'src': src,
            'dst': dst,
            'overlap_days': len(merged),
            'days_within_km': close_days,
            'mean_distance_km': mean_distance,
        })

        plt.figure(figsize=(6, 4))
        plt.plot(src_df['lon_mean'], src_df['lat_mean'], marker='o', markersize=2, linewidth=1, label=f'{src}')
        plt.plot(dst_df['lon_mean'], dst_df['lat_mean'], marker='o', markersize=2, linewidth=1, label=f'{dst}')
        plt.title(f'Candidate pair {src} vs {dst}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend(loc='best', fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f'pair_{src}_{dst}.png', dpi=150)
        plt.close()

    summary_df = pd.DataFrame(summary_rows)
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.out_summary, index=False)
    print(f"Wrote plots to {out_dir}")
    print(f"Wrote summary to {args.out_summary}")


if __name__ == '__main__':
    main()
