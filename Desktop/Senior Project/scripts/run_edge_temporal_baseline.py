#!/usr/bin/env python3
"""
Edge-level temporal baseline using recency and frequency features.

- Builds pair-level interaction counts and last-seen time from training period.
- Trains a logistic regression on edge features.
- Evaluates ROC AUC and Average Precision on a time-based split.
"""
import argparse
import json
from datetime import datetime, UTC
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


def load_edges_with_time(path: Path, years: List[int]) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if 'src' not in df.columns or 'dst' not in df.columns or 'time_bucket' not in df.columns:
        raise ValueError('Edge file must contain src, dst, and time_bucket columns')
    df = df[['src', 'dst', 'time_bucket']].dropna()
    df['time_bucket'] = pd.to_datetime(df['time_bucket'])
    if years:
        df = df[df['time_bucket'].dt.year.isin(years)]
    return df[['src', 'dst', 'time_bucket']].astype({'src': int, 'dst': int})


def time_split(df: pd.DataFrame, test_ratio: float):
    buckets = sorted(df['time_bucket'].dropna().unique())
    if len(buckets) < 2:
        raise ValueError('Need at least 2 time buckets for a time-based split.')
    n_test = max(1, int(len(buckets) * test_ratio))
    split_idx = max(1, len(buckets) - n_test)
    train_buckets = set(buckets[:split_idx])
    test_buckets = set(buckets[split_idx:])
    train_df = df[df['time_bucket'].isin(train_buckets)]
    test_df = df[df['time_bucket'].isin(test_buckets)]
    return train_df, test_df, buckets[split_idx], buckets[0]


def dedupe_pairs(df: pd.DataFrame) -> List[Tuple[int, int]]:
    df = df[['src', 'dst']].copy().reset_index(drop=True)
    df[['a', 'b']] = pd.DataFrame(
        np.sort(df[['src', 'dst']].to_numpy(), axis=1),
        columns=['a', 'b']
    )
    df = df.drop_duplicates(subset=['a', 'b'])[['a', 'b']]
    return list(map(tuple, df.to_numpy()))


def build_pair_stats(train_df: pd.DataFrame) -> pd.DataFrame:
    df = train_df.copy()
    df[['a', 'b']] = pd.DataFrame(
        np.sort(df[['src', 'dst']].to_numpy(), axis=1),
        columns=['a', 'b']
    )
    stats = df.groupby(['a', 'b']).agg(
        freq=('time_bucket', 'size'),
        last_time=('time_bucket', 'max'),
    ).reset_index()
    return stats


def build_feature_matrix(pairs: List[Tuple[int, int]], stats: pd.DataFrame, cutoff, min_time) -> np.ndarray:
    stat_map = {(row['a'], row['b']): (row['freq'], row['last_time']) for _, row in stats.iterrows()}
    max_recency = (pd.to_datetime(cutoff) - pd.to_datetime(min_time)).days + 1
    features = np.zeros((len(pairs), 2), dtype=float)
    for i, (u, v) in enumerate(pairs):
        a, b = (u, v) if u < v else (v, u)
        freq, last_time = stat_map.get((a, b), (0, None))
        if last_time is None:
            recency_days = max_recency
        else:
            recency_days = (pd.to_datetime(cutoff) - pd.to_datetime(last_time)).days
        features[i, 0] = np.log1p(freq)
        features[i, 1] = 1.0 / (1.0 + recency_days)
    return features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--edges', required=True)
    ap.add_argument('--years', default='2012,2013,2014,2015,2016,2017,2018,2019')
    ap.add_argument('--test-ratio', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-report', default='artifacts/edge_temporal_report.json')
    ap.add_argument('--log', action='store_true')
    ap.add_argument('--log-csv', default='artifacts/experiment_log.csv')
    ap.add_argument('--log-json', default='artifacts/experiment_log.json')
    args = ap.parse_args()

    years = [int(y.strip()) for y in args.years.split(',') if y.strip()]
    df = load_edges_with_time(Path(args.edges), years)
    train_df, test_df, cutoff, min_time = time_split(df, args.test_ratio)

    train_pairs = dedupe_pairs(train_df)
    test_pairs_all = dedupe_pairs(test_df)
    train_set = set((min(u, v), max(u, v)) for u, v in train_pairs)
    test_pos = [(u, v) for u, v in test_pairs_all if (min(u, v), max(u, v)) not in train_set]

    if not train_pairs or not test_pos:
        raise ValueError('Time split produced empty train or test edges.')

    stats = build_pair_stats(train_df)

    nodes = sorted({n for pair in dedupe_pairs(df) for n in pair})
    existing = set((min(u, v), max(u, v)) for u, v in dedupe_pairs(df))
    rng = np.random.default_rng(args.seed)
    neg = []
    nodes_list = list(nodes)
    seen = set()
    while len(neg) < len(test_pos) and len(seen) < len(nodes_list) * len(nodes_list):
        u, v = rng.choice(nodes_list, size=2, replace=False)
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in existing or (a, b) in seen:
            continue
        seen.add((a, b))
        neg.append((a, b))

    X_pos = build_feature_matrix(test_pos, stats, cutoff, min_time)
    X_neg = build_feature_matrix(neg, stats, cutoff, min_time)
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * len(test_pos) + [0] * len(neg))

    clf = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=args.seed)
    clf.fit(X, y)
    scores = clf.predict_proba(X)[:, 1]

    report = {
        'edges_total': len(existing),
        'train_edges': len(train_pairs),
        'test_pos': len(test_pos),
        'test_neg': len(neg),
        'cutoff_time_bucket': str(cutoff),
        'metrics': {
            'roc_auc': roc_auc_score(y, scores),
            'average_precision': average_precision_score(y, scores),
        },
    }

    with open(args.out_report, 'w') as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Report written to {args.out_report}")

    if args.log:
        row = {
            'timestamp': datetime.now(UTC).isoformat().replace('+00:00', 'Z'),
            'model': 'edge_temporal',
            'edges': args.edges,
            'test_ratio': args.test_ratio,
            'seed': args.seed,
            'embedding_dim': 0,
            'use_features': True,
            'features_years': 'edge_temporal',
            'roc_auc': report['metrics']['roc_auc'],
            'roc_auc_std': 0.0,
            'average_precision': report['metrics']['average_precision'],
            'average_precision_std': 0.0,
        }
        log_df = pd.DataFrame([row])
        log_csv_path = Path(args.log_csv)
        if log_csv_path.exists():
            log_df.to_csv(log_csv_path, mode='a', index=False, header=False)
        else:
            log_df.to_csv(log_csv_path, index=False)
        log_json_path = Path(args.log_json)
        if log_json_path.exists():
            with open(log_json_path, 'r') as f:
                payload = json.load(f)
            if not isinstance(payload, list):
                payload = [payload]
        else:
            payload = []
        payload.append(row)
        with open(log_json_path, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"Experiment log updated: {log_csv_path}")
        print(f"Experiment log updated: {log_json_path}")


if __name__ == '__main__':
    main()
