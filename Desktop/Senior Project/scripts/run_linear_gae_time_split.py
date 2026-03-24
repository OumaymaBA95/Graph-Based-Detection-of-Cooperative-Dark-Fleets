#!/usr/bin/env python3
"""
Linear GAE baseline with time-based train/test split.

- Train on earlier time buckets, test on later time buckets.
- Uses truncated SVD on the training adjacency.
- Optional SST feature concatenation (same as standard baseline).
"""
import argparse
import json
import os
from datetime import datetime, UTC
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds

import run_linear_gae_baseline as lgae
from build_temporal_node_features import build_temporal_node_features


def load_edges_with_time(path: Path, years: List[int]) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if 'src' not in df.columns or 'dst' not in df.columns or 'time_bucket' not in df.columns:
        raise ValueError('Edge file must contain src, dst, and time_bucket columns')
    df = df[['src', 'dst', 'time_bucket']].dropna()
    df['time_bucket'] = pd.to_datetime(df['time_bucket'])
    if years:
        df = df[df['time_bucket'].dt.year.isin(years)]
    df = df[['src', 'dst', 'time_bucket']].astype({'src': int, 'dst': int})
    return df[df['src'] != df['dst']].reset_index(drop=True)


def dedupe_pairs(df: pd.DataFrame) -> List[Tuple[int, int]]:
    df = df[['src', 'dst']].copy().reset_index(drop=True)
    df[['a', 'b']] = pd.DataFrame(
        np.sort(df[['src', 'dst']].to_numpy(), axis=1),
        columns=['a', 'b']
    )
    df = df.drop_duplicates(subset=['a', 'b'])[['a', 'b']]
    return list(map(tuple, df.to_numpy()))


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
    return train_df, test_df, buckets[split_idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--edges', required=True)
    ap.add_argument('--years', default='2012,2013,2014,2015,2016,2017,2018,2019')
    ap.add_argument('--embedding-dim', type=int, default=32)
    ap.add_argument('--test-ratio', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-report', default='artifacts/linear_gae_time_report.json')
    ap.add_argument('--features-root', default='data/features_by_year')
    ap.add_argument('--features-years', default='', help='Comma-separated years (optional)')
    ap.add_argument('--use-features', action='store_true')
    ap.add_argument('--use-temporal-features', action='store_true')
    ap.add_argument('--log', action='store_true')
    ap.add_argument('--log-csv', default='artifacts/experiment_log.csv')
    ap.add_argument('--log-json', default='artifacts/experiment_log.json')
    args = ap.parse_args()

    os.makedirs(Path(args.out_report).parent, exist_ok=True)

    years = [int(y.strip()) for y in args.years.split(',') if y.strip()]
    df = load_edges_with_time(Path(args.edges), years)
    train_df, test_df, cutoff = time_split(df, args.test_ratio)

    train_pairs = dedupe_pairs(train_df)
    test_pairs_all = dedupe_pairs(test_df)

    train_set = set((min(u, v), max(u, v)) for u, v in train_pairs)
    test_pairs = [(u, v) for u, v in test_pairs_all if (min(u, v), max(u, v)) not in train_set]

    nodes = sorted({n for pair in dedupe_pairs(df) for n in pair})
    node_set = set(nodes)

    if not train_pairs or not test_pairs:
        raise ValueError('Time split produced empty train or test edges after filtering.')

    node_to_idx = {n: i for i, n in enumerate(nodes)}
    rows = [node_to_idx[u] for u, v in train_pairs]
    cols = [node_to_idx[v] for u, v in train_pairs]
    data = np.ones(len(rows))
    n = len(nodes)
    adj = coo_matrix((data, (rows, cols)), shape=(n, n))
    adj = adj + adj.T

    k = min(args.embedding_dim, max(2, n - 1))
    u, s, vt = svds(adj, k=k)
    order = np.argsort(-s)
    u = u[:, order]
    s = s[order]
    emb = u * s

    feature_chunks = []
    feature_labels = []

    if args.use_features and args.features_years:
        feat_years = [y.strip() for y in args.features_years.split(',') if y.strip()]
        feats = lgae.load_features(Path(args.features_root), feat_years)
        if not feats.empty:
            feat_cols = [c for c in feats.columns if c != 'MMSI']
            feats = feats.set_index('MMSI')
            feat_mat = np.zeros((n, len(feat_cols)), dtype=float)
            for i, m in enumerate(nodes):
                if m in feats.index:
                    feat_mat[i] = feats.loc[m, feat_cols].to_numpy(dtype=float)
            mean = np.nanmean(feat_mat, axis=0)
            std = np.nanstd(feat_mat, axis=0)
            std[std == 0] = 1.0
            feat_mat = (feat_mat - mean) / std
            feat_mat = np.nan_to_num(feat_mat)
            feature_chunks.append(feat_mat)
            feature_labels.append('sst')

    if args.use_temporal_features:
        temporal_feats = build_temporal_node_features(train_df, cutoff)
        if not temporal_feats.empty:
            temporal_feats = temporal_feats.set_index('MMSI')
            temporal_cols = [c for c in temporal_feats.columns]
            temporal_mat = np.zeros((n, len(temporal_cols)), dtype=float)
            for i, m in enumerate(nodes):
                if m in temporal_feats.index:
                    temporal_mat[i] = temporal_feats.loc[m, temporal_cols].to_numpy(dtype=float)
            mean = np.nanmean(temporal_mat, axis=0)
            std = np.nanstd(temporal_mat, axis=0)
            std[std == 0] = 1.0
            temporal_mat = (temporal_mat - mean) / std
            temporal_mat = np.nan_to_num(temporal_mat)
            feature_chunks.append(temporal_mat)
            feature_labels.append('temporal')

    if feature_chunks:
        emb = np.concatenate([emb] + feature_chunks, axis=1)

    existing = set((min(u, v), max(u, v)) for u, v in dedupe_pairs(df))

    test_neg = lgae.negative_sampling(nodes, existing, k=len(test_pairs), seed=args.seed)

    def score(u, v):
        iu, iv = node_to_idx[u], node_to_idx[v]
        return float(np.dot(emb[iu], emb[iv]))

    labels = np.array([1] * len(test_pairs) + [0] * len(test_neg))
    scores = np.array([score(u, v) for u, v in test_pairs + test_neg])

    report = {
        'edges_total': len(existing),
        'train_edges': len(train_pairs),
        'test_pos': len(test_pairs),
        'test_neg': len(test_neg),
        'embedding_dim': int(emb.shape[1]),
    'use_features': bool((args.use_features and args.features_years) or args.use_temporal_features),
        'years': years,
        'cutoff_time_bucket': str(cutoff),
        'metrics': {
            'roc_auc': lgae.roc_auc(labels, scores),
            'average_precision': lgae.average_precision(labels, scores),
        },
    }

    with open(args.out_report, 'w') as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Report written to {args.out_report}")

    if args.log:
        timestamp = datetime.now(UTC).isoformat().replace('+00:00', 'Z')
        row = {
            'timestamp': timestamp,
            'model': 'linear_gae_time',
            'edges': str(args.edges),
            'test_ratio': args.test_ratio,
            'seed': args.seed,
            'embedding_dim': report['embedding_dim'],
            'use_features': report['use_features'],
            'features_years': ','.join([label for label in feature_labels]) if feature_labels else args.features_years,
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
