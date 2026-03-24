#!/usr/bin/env python3
"""
Run the linear GAE time-split baseline with temporal interaction features across multiple seeds.
"""
import argparse
import json
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds

import run_linear_gae_baseline as lgae
import run_linear_gae_time_split as time_split
from build_temporal_node_features import build_temporal_node_features


def build_embedding(train_pairs, nodes, embedding_dim):
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    rows = [node_to_idx[u] for u, v in train_pairs]
    cols = [node_to_idx[v] for u, v in train_pairs]
    data = np.ones(len(rows))
    n = len(nodes)
    adj = coo_matrix((data, (rows, cols)), shape=(n, n))
    adj = adj + adj.T

    k = min(embedding_dim, max(2, n - 1))
    u, s, vt = svds(adj, k=k)
    order = np.argsort(-s)
    u = u[:, order]
    s = s[order]
    emb = u * s
    return emb, node_to_idx


def build_temporal_matrix(nodes, train_df, cutoff):
    temporal_feats = build_temporal_node_features(train_df, cutoff)
    if temporal_feats.empty:
        return None
    temporal_feats = temporal_feats.set_index('MMSI')
    temporal_cols = list(temporal_feats.columns)
    temporal_mat = np.zeros((len(nodes), len(temporal_cols)), dtype=float)
    for i, m in enumerate(nodes):
        if m in temporal_feats.index:
            temporal_mat[i] = temporal_feats.loc[m, temporal_cols].to_numpy(dtype=float)
    mean = np.nanmean(temporal_mat, axis=0)
    std = np.nanstd(temporal_mat, axis=0)
    std[std == 0] = 1.0
    temporal_mat = (temporal_mat - mean) / std
    temporal_mat = np.nan_to_num(temporal_mat)
    return temporal_mat


def run_once(seed, emb, node_to_idx, nodes, test_pairs, existing):
    test_neg = lgae.negative_sampling(nodes, existing, k=len(test_pairs), seed=seed)

    def score(u, v):
        iu, iv = node_to_idx[u], node_to_idx[v]
        return float(np.dot(emb[iu], emb[iv]))

    labels = np.array([1] * len(test_pairs) + [0] * len(test_neg))
    scores = np.array([score(u, v) for u, v in test_pairs + test_neg])

    return {
        'seed': seed,
        'metrics': {
            'roc_auc': lgae.roc_auc(labels, scores),
            'average_precision': lgae.average_precision(labels, scores),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--edges', required=True)
    ap.add_argument('--years', default='2012,2013,2014,2015,2016,2017,2018,2019')
    ap.add_argument('--embedding-dim', type=int, default=32)
    ap.add_argument('--test-ratio', type=float, default=0.3)
    ap.add_argument('--seeds', default='1,2,3')
    ap.add_argument('--out-report', default='artifacts/linear_gae_time_temporal_multiseed.json')
    ap.add_argument('--out-csv', default='artifacts/linear_gae_time_temporal_multiseed.csv')
    ap.add_argument('--log', action='store_true')
    ap.add_argument('--log-csv', default='artifacts/experiment_log.csv')
    ap.add_argument('--log-json', default='artifacts/experiment_log.json')
    args = ap.parse_args()

    years = [int(y.strip()) for y in args.years.split(',') if y.strip()]
    df = time_split.load_edges_with_time(Path(args.edges), years)
    train_df, test_df, cutoff = time_split.time_split(df, args.test_ratio)

    train_pairs = time_split.dedupe_pairs(train_df)
    test_pairs_all = time_split.dedupe_pairs(test_df)
    train_set = set((min(u, v), max(u, v)) for u, v in train_pairs)
    test_pairs = [(u, v) for u, v in test_pairs_all if (min(u, v), max(u, v)) not in train_set]

    if not train_pairs or not test_pairs:
        raise ValueError('Time split produced empty train or test edges.')

    all_pairs = time_split.dedupe_pairs(df)
    nodes = sorted({n for pair in all_pairs for n in pair})
    existing = set((min(u, v), max(u, v)) for u, v in all_pairs)

    emb, node_to_idx = build_embedding(train_pairs, nodes, args.embedding_dim)
    temporal_mat = build_temporal_matrix(nodes, train_df, cutoff)
    if temporal_mat is not None:
        emb = np.concatenate([emb, temporal_mat], axis=1)

    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    results = [run_once(seed, emb, node_to_idx, nodes, test_pairs, existing) for seed in seeds]

    aucs = [r['metrics']['roc_auc'] for r in results]
    aps = [r['metrics']['average_precision'] for r in results]
    summary = {
        'roc_auc_mean': float(np.mean(aucs)),
        'roc_auc_std': float(np.std(aucs)),
        'average_precision_mean': float(np.mean(aps)),
        'average_precision_std': float(np.std(aps)),
    }

    report = {
        'edges': args.edges,
        'years': years,
        'embedding_dim': args.embedding_dim,
        'test_ratio': args.test_ratio,
        'seeds': seeds,
        'summary': summary,
        'per_seed': results,
    }

    with open(args.out_report, 'w') as f:
        json.dump(report, f, indent=2)

    pd.DataFrame([
        {
            'roc_auc_mean': summary['roc_auc_mean'],
            'roc_auc_std': summary['roc_auc_std'],
            'average_precision_mean': summary['average_precision_mean'],
            'average_precision_std': summary['average_precision_std'],
        }
    ]).to_csv(args.out_csv, index=False)

    print(json.dumps(summary, indent=2))
    print(f"Report written to {args.out_report}")
    print(f"CSV written to {args.out_csv}")

    if args.log:
        row = {
            'timestamp': datetime.now(UTC).isoformat().replace('+00:00', 'Z'),
            'model': 'linear_gae_time_temporal_multiseed',
            'edges': args.edges,
            'test_ratio': args.test_ratio,
            'seed': ';'.join(map(str, seeds)),
            'embedding_dim': emb.shape[1],
            'use_features': True,
            'features_years': 'temporal',
            'roc_auc': summary['roc_auc_mean'],
            'roc_auc_std': summary['roc_auc_std'],
            'average_precision': summary['average_precision_mean'],
            'average_precision_std': summary['average_precision_std'],
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
