#!/usr/bin/env python3
"""
Minimal GAE-style training using PyTorch (no torch-geometric).

- Learns node embeddings with a dot-product decoder.
- Trains on positive edges + sampled negative edges.
- Evaluates ROC AUC / AP on a held-out split.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch


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


def train_test_split_edges(edges, test_ratio, seed):
    rng = np.random.default_rng(seed)
    edges = edges.copy()
    rng.shuffle(edges)
    n_test = max(1, int(len(edges) * test_ratio))
    return edges[n_test:], edges[:n_test]


def negative_sampling(nodes, existing, k, seed):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--edges', required=True)
    ap.add_argument('--years', default='2018,2019')
    ap.add_argument('--embedding-dim', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--test-ratio', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-report', default='artifacts/torch_gae_report.json')
    args = ap.parse_args()

    years = [int(y.strip()) for y in args.years.split(',') if y.strip()]

    edges_df = pd.read_parquet(args.edges)
    edges_df['time_bucket'] = pd.to_datetime(edges_df['time_bucket'])
    edges_df = edges_df[edges_df['time_bucket'].dt.year.isin(years)]
    if edges_df.empty:
        raise ValueError(f'No edges found for years {years}.')
    edges_df = edges_df[['src', 'dst']].dropna().astype(int)
    edges_df = edges_df[edges_df['src'] != edges_df['dst']].reset_index(drop=True)
    edges_df[['a', 'b']] = pd.DataFrame(np.sort(edges_df[['src', 'dst']].to_numpy(), axis=1))
    edges_df = edges_df.drop_duplicates(subset=['a', 'b'])[['a', 'b']].rename(columns={'a': 'src', 'b': 'dst'})

    edge_list = list(map(tuple, edges_df[['src', 'dst']].to_numpy()))
    train_edges, test_pos = train_test_split_edges(edge_list, args.test_ratio, args.seed)

    nodes = sorted(set(edges_df['src']).union(set(edges_df['dst'])))
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    train_edges = [(node_to_idx[u], node_to_idx[v]) for u, v in train_edges]
    test_pos_idx = [(node_to_idx[u], node_to_idx[v]) for u, v in test_pos]

    existing = set((min(u, v), max(u, v)) for u, v in edge_list)
    test_neg = negative_sampling(nodes, existing, k=len(test_pos), seed=args.seed)
    test_neg_idx = [(node_to_idx[u], node_to_idx[v]) for u, v in test_neg]

    torch.manual_seed(args.seed)
    emb = torch.nn.Embedding(len(nodes), args.embedding_dim)
    optimizer = torch.optim.Adam(emb.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    pos_edges = torch.tensor(train_edges, dtype=torch.long)

    for _ in range(args.epochs):
        optimizer.zero_grad()
        # sample negatives each epoch
        neg_edges = negative_sampling(nodes, existing, k=len(train_edges), seed=np.random.randint(0, 1_000_000))
        neg_edges = [(node_to_idx[u], node_to_idx[v]) for u, v in neg_edges]
        neg_edges = torch.tensor(neg_edges, dtype=torch.long)

        pos_scores = (emb(pos_edges[:, 0]) * emb(pos_edges[:, 1])).sum(dim=1)
        neg_scores = (emb(neg_edges[:, 0]) * emb(neg_edges[:, 1])).sum(dim=1)

        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        scores = torch.cat([pos_scores, neg_scores])
        loss = loss_fn(scores, labels)
        loss.backward()
        optimizer.step()

    def score_pairs(pairs):
        pairs = torch.tensor(pairs, dtype=torch.long)
        with torch.no_grad():
            scores = (emb(pairs[:, 0]) * emb(pairs[:, 1])).sum(dim=1).detach().cpu().tolist()
        return np.array(scores, dtype=float)

    pos_scores = score_pairs(test_pos_idx)
    neg_scores = score_pairs(test_neg_idx)
    labels = np.array([1] * len(pos_scores) + [0] * len(neg_scores))
    scores = np.concatenate([pos_scores, neg_scores])

    report = {
        'years': years,
        'edges_total': len(edge_list),
        'train_edges': len(train_edges),
        'test_pos': len(test_pos),
        'test_neg': len(test_neg),
        'embedding_dim': args.embedding_dim,
        'epochs': args.epochs,
        'metrics': {
            'roc_auc': roc_auc(labels, scores),
            'average_precision': average_precision(labels, scores),
        },
    }

    os.makedirs(Path(args.out_report).parent, exist_ok=True)
    with open(args.out_report, 'w') as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Report written to {args.out_report}")


if __name__ == '__main__':
    main()
