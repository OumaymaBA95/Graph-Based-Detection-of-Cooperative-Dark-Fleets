#!/usr/bin/env python3
"""
Linear GAE-style baseline using low-rank adjacency factorization.

- Builds a sparse adjacency from an edge list.
- Fits a low-rank embedding via truncated SVD (svds).
- Scores candidate edges by inner product of embeddings.
- Optionally concatenates aggregated SST features (per MMSI) to embeddings.
"""
import argparse
import json
import os
from datetime import datetime, UTC
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds


def load_edges(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if 'src' not in df.columns or 'dst' not in df.columns:
        raise ValueError('Edge file must contain src and dst columns')
    df = df[['src', 'dst']].astype(int)
    df = df[df['src'] != df['dst']]
    df[['a', 'b']] = pd.DataFrame(np.sort(df[['src', 'dst']].to_numpy(), axis=1))
    return df.drop_duplicates(subset=['a', 'b'])[['a', 'b']].rename(columns={'a': 'src', 'b': 'dst'})


def train_test_split_edges(edges: List[Tuple[int, int]], test_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    edges = edges.copy()
    rng.shuffle(edges)
    n_test = max(1, int(len(edges) * test_ratio))
    return edges[n_test:], edges[:n_test]


def negative_sampling(nodes: List[int], existing: set, k: int, seed: int):
    rng = np.random.default_rng(seed)
    neg = []
    nodes_list = list(nodes)
    seen = set()
    while len(neg) < k and len(seen) < len(nodes_list) * len(nodes_list):
        u, v = rng.choice(nodes_list, size=2, replace=False)
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in existing or (a, b) in seen:
            continue
        seen.add((a, b))
        neg.append((a, b))
    return neg


def hard_negative_sampling(
    nodes: List[int],
    existing: set,
    edges: List[Tuple[int, int]],
    k: int,
    seed: int,
    fallback_random: bool = True,
) -> List[Tuple[int, int]]:
    """
    Sample "hard" negatives: pairs that share a neighbor (2-hop) but are not linked.
    Falls back to random sampling if not enough 2-hop pairs exist.
    """
    neighbors: Dict[int, set] = defaultdict(set)
    for u, v in edges:
        a, b = min(u, v), max(u, v)
        neighbors[a].add(b)
        neighbors[b].add(a)

    two_hop_candidates = []
    for u in nodes:
        for v in neighbors.get(u, []):
            for w in neighbors.get(v, []):
                if u == w:
                    continue
                a, b = (u, w) if u < w else (w, u)
                if (a, b) in existing:
                    continue
                two_hop_candidates.append((a, b))

    two_hop_set = list(dict.fromkeys(two_hop_candidates))
    rng = np.random.default_rng(seed)
    if len(two_hop_set) >= k:
        idx = rng.choice(len(two_hop_set), size=k, replace=False)
        return [two_hop_set[i] for i in idx]

    neg = two_hop_set.copy()
    if fallback_random and len(neg) < k:
        extra = negative_sampling(nodes, existing | set(neg), k - len(neg), seed + 1)
        neg.extend(extra)
    return neg[:k]


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    tp = tp / tp[-1] if tp[-1] > 0 else tp
    fp = fp / fp[-1] if fp[-1] > 0 else fp
    if hasattr(np, 'trapz'):
        return float(np.trapz(tp, fp))
    return float(np.trapezoid(tp, fp))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    labels = labels[order]
    cum_tp = np.cumsum(labels)
    precision = cum_tp / (np.arange(len(labels)) + 1)
    ap = (precision * labels).sum() / max(1, labels.sum())
    return float(ap)


def load_features(features_root: Path, years: List[str]) -> pd.DataFrame:
    frames = []
    for y in years:
        path = features_root / y / 'vessel_day_features.parquet'
        if not path.exists():
            print(f"Skip missing features: {path}")
            continue
        df = pd.read_parquet(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    feats = pd.concat(frames, ignore_index=True)
    agg = feats.groupby('MMSI').agg({
        'sst_mean': 'mean',
        'sst_std': 'mean',
        'sst_min': 'mean',
        'sst_max': 'mean',
        'sst_missing_rate': 'mean',
        'lat_mean': 'mean',
        'lon_mean': 'mean',
    }).reset_index()
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--edges', required=True)
    ap.add_argument('--embedding-dim', type=int, default=32)
    ap.add_argument('--test-ratio', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-report', default='artifacts/linear_gae_report.json')
    ap.add_argument('--features-root', default='data/features_by_year')
    ap.add_argument('--features-years', default='', help='Comma-separated years (optional)')
    ap.add_argument('--use-features', action='store_true')
    ap.add_argument('--log', action='store_true')
    ap.add_argument('--log-csv', default='artifacts/experiment_log.csv')
    ap.add_argument('--log-json', default='artifacts/experiment_log.json')
    args = ap.parse_args()

    os.makedirs(Path(args.out_report).parent, exist_ok=True)

    df = load_edges(Path(args.edges))
    edge_list = list(map(tuple, df[['src', 'dst']].to_numpy()))
    train_edges, test_pos = train_test_split_edges(edge_list, args.test_ratio, args.seed)

    nodes = sorted(set(df['src']).union(set(df['dst'])))
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    rows = [node_to_idx[u] for u, v in train_edges]
    cols = [node_to_idx[v] for u, v in train_edges]
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

    if args.use_features and args.features_years:
        years = [y.strip() for y in args.features_years.split(',') if y.strip()]
        feats = load_features(Path(args.features_root), years)
        if not feats.empty:
            feat_cols = [c for c in feats.columns if c != 'MMSI']
            feats = feats.set_index('MMSI')
            feat_mat = np.zeros((n, len(feat_cols)), dtype=float)
            for i, m in enumerate(nodes):
                if m in feats.index:
                    feat_mat[i] = feats.loc[m, feat_cols].to_numpy(dtype=float)
            # z-score
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
        'edges_total': len(edge_list),
        'train_edges': len(train_edges),
        'test_pos': len(test_pos),
        'test_neg': len(test_neg),
        'embedding_dim': int(emb.shape[1]),
        'use_features': bool(args.use_features and args.features_years),
        'metrics': {
            'roc_auc': roc_auc(labels, scores),
            'average_precision': average_precision(labels, scores),
        },
    }

    with open(args.out_report, 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"Report written to {args.out_report}")

    if args.log:
        timestamp = datetime.now(UTC).isoformat().replace('+00:00', 'Z')
        summary_row = {
            'timestamp': timestamp,
            'model': 'linear_gae',
            'edges': str(args.edges),
            'test_ratio': args.test_ratio,
            'seed': args.seed,
            'embedding_dim': report['embedding_dim'],
            'use_features': report['use_features'],
            'features_years': args.features_years,
            'roc_auc': report['metrics']['roc_auc'],
            'average_precision': report['metrics']['average_precision'],
        }

        log_df = pd.DataFrame([summary_row])
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
        payload.append(summary_row)
        with open(log_json_path, 'w') as f:
            json.dump(payload, f, indent=2)

        print(f"Experiment log updated: {log_csv_path}")
        print(f"Experiment log updated: {log_json_path}")


if __name__ == '__main__':
    main()
