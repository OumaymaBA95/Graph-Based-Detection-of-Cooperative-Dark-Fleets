#!/usr/bin/env python3
"""
Simple link prediction baseline on an undirected graph.

- Input: Parquet edge list with columns src, dst (and optional weight/time_bucket).
- Splits edges into train/test, samples negative edges, computes heuristic scores:
  common neighbors, Jaccard, Adamic-Adar.
- Outputs a JSON report with AUC and Average Precision per heuristic.
- Includes a --synthetic flag to create a tiny random graph for smoke testing.
"""
import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd


def load_edges(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # normalize column names
    cols = {c.lower(): c for c in df.columns}
    def col(name):
        return cols.get(name, name)
    rename = {}
    if 'src' not in df.columns and 'source' in df.columns:
        rename['source'] = 'src'
    if 'dst' not in df.columns and 'target' in df.columns:
        rename['target'] = 'dst'
    if rename:
        df = df.rename(columns=rename)
    if 'src' not in df.columns or 'dst' not in df.columns:
        raise ValueError('Edge file must have src and dst columns')
    df = df[['src', 'dst']].astype(int)
    return df


def make_synthetic(n_nodes: int = 50, p: float = 0.08) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < p:
                edges.append((i, j))
    return pd.DataFrame(edges, columns=['src', 'dst'])


def train_test_split_edges(edges: List[Tuple[int, int]], test_ratio: float = 0.2, seed: int = 42):
    rng = random.Random(seed)
    rng.shuffle(edges)
    n_test = max(1, int(len(edges) * test_ratio))
    test = edges[:n_test]
    train = edges[n_test:]
    return train, test


def build_adj(edges: List[Tuple[int, int]]):
    adj: Dict[int, Set[int]] = {}
    for u, v in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj


def negative_sampling(nodes: List[int], existing: Set[Tuple[int, int]], k: int, seed: int = 42):
    rng = random.Random(seed)
    neg = []
    nodes_list = list(nodes)
    seen = set()
    while len(neg) < k and len(seen) < len(nodes_list) * len(nodes_list):
        u, v = rng.sample(nodes_list, 2)
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in existing or (a, b) in seen:
            continue
        seen.add((a, b))
        neg.append((a, b))
    return neg


def common_neighbors(adj, u, v):
    return len(adj.get(u, set()) & adj.get(v, set()))


def jaccard(adj, u, v):
    a = adj.get(u, set())
    b = adj.get(v, set())
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def adamic_adar(adj, u, v):
    inter = adj.get(u, set()) & adj.get(v, set())
    s = 0.0
    for w in inter:
        deg = len(adj.get(w, []))
        if deg > 1:
            s += 1.0 / np.log(deg)
    return s


def compute_scores(adj, pairs: List[Tuple[int, int]]):
    scores = {
        'common_neighbors': [],
        'jaccard': [],
        'adamic_adar': [],
    }
    for u, v in pairs:
        scores['common_neighbors'].append(common_neighbors(adj, u, v))
        scores['jaccard'].append(jaccard(adj, u, v))
        scores['adamic_adar'].append(adamic_adar(adj, u, v))
    return scores


def _trapz(y, x):
    # numpy 2 renamed trapz -> trapezoid
    if hasattr(np, 'trapz'):
        return np.trapz(y, x)
    return np.trapezoid(y, x)


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    # simple trapezoidal AUC from sorted scores
    order = np.argsort(-scores)
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    tp = tp / tp[-1] if tp[-1] > 0 else tp
    fp = fp / fp[-1] if fp[-1] > 0 else fp
    auc = _trapz(tp, fp)
    return float(auc)


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    labels = labels[order]
    cum_tp = np.cumsum(labels)
    precision = cum_tp / (np.arange(len(labels)) + 1)
    ap = (precision * labels).sum() / max(1, labels.sum())
    return float(ap)


def evaluate(adj, test_pos: List[Tuple[int, int]], test_neg: List[Tuple[int, int]]):
    pairs = test_pos + test_neg
    labels = np.array([1] * len(test_pos) + [0] * len(test_neg))
    scores_dict = compute_scores(adj, pairs)
    metrics = {}
    for name, sc in scores_dict.items():
        sc_arr = np.array(sc, dtype=float)
        metrics[name] = {
            'roc_auc': roc_auc(labels, sc_arr),
            'average_precision': average_precision(labels, sc_arr),
        }
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--edges', default='artifacts/edges_baseline.parquet')
    ap.add_argument('--synthetic', action='store_true', help='Ignore edges file and use synthetic graph')
    ap.add_argument('--test-ratio', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-report', default='artifacts/baseline_linkpred.json')
    args = ap.parse_args()

    os.makedirs(Path(args.out_report).parent, exist_ok=True)

    if args.synthetic:
        df = make_synthetic(n_nodes=60, p=0.06)
    else:
        if not Path(args.edges).exists():
            raise FileNotFoundError(f"Edge file not found: {args.edges}")
        df = load_edges(Path(args.edges))

    # deduplicate edges
    df = df[df['src'] != df['dst']]
    df[['a', 'b']] = pd.DataFrame(np.sort(df[['src', 'dst']].to_numpy(), axis=1))
    df = df.drop_duplicates(subset=['a', 'b'])[['a', 'b']]
    edge_list = list(map(tuple, df.to_numpy()))
    if len(edge_list) < 4:
        print("Not enough edges to evaluate.")
        return

    train, test_pos = train_test_split_edges(edge_list, test_ratio=args.test_ratio, seed=args.seed)
    adj = build_adj(train)
    nodes = list(adj.keys())
    test_neg = negative_sampling(nodes, set(edge_list), k=len(test_pos), seed=args.seed)

    metrics = evaluate(adj, test_pos, test_neg)
    report = {
        'edges_total': len(edge_list),
        'train_edges': len(train),
        'test_pos': len(test_pos),
        'test_neg': len(test_neg),
        'metrics': metrics,
    }
    with open(args.out_report, 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"Report written to {args.out_report}")


if __name__ == '__main__':
    main()
