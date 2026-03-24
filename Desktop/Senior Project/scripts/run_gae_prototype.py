#!/usr/bin/env python3
"""
Prototype GAE-style evaluation on a single year using the linear GAE baseline.

- Filters edges by time_bucket year.
- Optionally adds SST feature summaries for the same year(s).
- Saves embeddings for inspection.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds

from run_linear_gae_baseline import load_edges, load_features, roc_auc, average_precision, negative_sampling, train_test_split_edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--edges', required=True)
    ap.add_argument('--year', help='Single year to filter by (e.g., 2018)')
    ap.add_argument('--years', default='', help='Comma-separated years to include (e.g., 2018,2019)')
    ap.add_argument('--embedding-dim', type=int, default=32)
    ap.add_argument('--test-ratio', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--features-root', default='data/features_by_year')
    ap.add_argument('--use-features', action='store_true')
    ap.add_argument('--out-report', default='artifacts/gae_prototype_report.json')
    ap.add_argument('--out-embeddings', default='artifacts/gae_prototype_embeddings.parquet')
    args = ap.parse_args()

    if not args.year and not args.years:
        raise ValueError('Provide --year or --years.')

    years = []
    if args.years:
        years = [int(y.strip()) for y in args.years.split(',') if y.strip()]
    elif args.year:
        years = [int(args.year)]

    edges_df = pd.read_parquet(args.edges)
    if 'time_bucket' not in edges_df.columns:
        raise ValueError('Edges file must include time_bucket to filter by year.')
    edges_df['time_bucket'] = pd.to_datetime(edges_df['time_bucket'])
    edges_df = edges_df[edges_df['time_bucket'].dt.year.isin(years)]
    if edges_df.empty:
        raise ValueError(f'No edges found for years {years}.')

    edges_df = edges_df[['src', 'dst']].dropna(subset=['src', 'dst']).reset_index(drop=True)
    edges_df = edges_df[edges_df['src'] != edges_df['dst']]
    edges_df[['src', 'dst']] = edges_df[['src', 'dst']].astype(int)
    edges_df[['a', 'b']] = pd.DataFrame(np.sort(edges_df[['src', 'dst']].to_numpy(), axis=1))
    edges_df = edges_df.drop_duplicates(subset=['a', 'b'])[['a', 'b']].rename(columns={'a': 'src', 'b': 'dst'})

    edge_list = list(map(tuple, edges_df[['src', 'dst']].to_numpy()))
    edge_list = [e for e in edge_list if not (pd.isna(e[0]) or pd.isna(e[1]))]
    if len(edge_list) < 2:
        raise ValueError(f'Not enough edges for years {years} after filtering.')
    train_edges, test_pos = train_test_split_edges(edge_list, args.test_ratio, args.seed)

    nodes = sorted(set(edges_df['src']).union(set(edges_df['dst'])))
    nodes = [int(n) for n in nodes if not pd.isna(n)]
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    rows = [node_to_idx[u] for u, v in train_edges]
    cols = [node_to_idx[v] for u, v in train_edges]
    data = np.ones(len(rows))
    n = len(nodes)
    adj = coo_matrix((data, (rows, cols)), shape=(n, n))
    adj = adj + adj.T

    k = min(args.embedding_dim, max(1, n - 1))
    if k >= min(adj.shape):
        k = max(1, min(adj.shape) - 1)
    try:
        u, s, _ = svds(adj, k=k)
        order = np.argsort(-s)
        u = u[:, order]
        s = s[order]
        emb = u * s
    except Exception:
        dense = adj.toarray()
        u, s, _ = np.linalg.svd(dense, full_matrices=False)
        u = u[:, :k]
        s = s[:k]
        emb = u * s

    if args.use_features:
        feats = load_features(Path(args.features_root), [str(y) for y in years])
        if not feats.empty:
            feat_cols = [c for c in feats.columns if c != 'MMSI']
            feats = feats.set_index('MMSI')
            feat_mat = np.zeros((n, len(feat_cols)), dtype=float)
            for i, m in enumerate(nodes):
                if m in feats.index:
                    feat_mat[i] = feats.loc[m, feat_cols].to_numpy(dtype=float)
            if feat_mat.size and np.isfinite(feat_mat).any():
                mean = np.nanmean(feat_mat, axis=0)
                std = np.nanstd(feat_mat, axis=0)
                std[std == 0] = 1.0
                feat_mat = (feat_mat - mean) / std
                feat_mat = np.nan_to_num(feat_mat)
                emb = np.concatenate([emb, feat_mat], axis=1)

    existing = set(edge_list)
    test_neg = negative_sampling(nodes, existing, k=len(test_pos), seed=args.seed)

    def score(u, v):
        iu, iv = node_to_idx[u], node_to_idx[v]
        return float(np.dot(emb[iu], emb[iv]))

    pairs = test_pos + test_neg
    labels = np.array([1] * len(test_pos) + [0] * len(test_neg))
    scores = np.array([score(u, v) for u, v in pairs])

    report = {
        'years': years,
        'edges_total': len(edge_list),
        'train_edges': len(train_edges),
        'test_pos': len(test_pos),
        'test_neg': len(test_neg),
        'embedding_dim': int(emb.shape[1]),
    'use_features': bool(args.use_features),
        'metrics': {
            'roc_auc': roc_auc(labels, scores),
            'average_precision': average_precision(labels, scores),
        },
    }

    with open(args.out_report, 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

    emb_df = pd.DataFrame(emb, columns=[f'emb_{i}' for i in range(emb.shape[1])])
    emb_df.insert(0, 'MMSI', nodes)
    Path(args.out_embeddings).parent.mkdir(parents=True, exist_ok=True)
    emb_df.to_parquet(args.out_embeddings, index=False)
    print(f"Embeddings written to {args.out_embeddings}")
    print(f"Report written to {args.out_report}")


if __name__ == '__main__':
    main()
