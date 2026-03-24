#!/usr/bin/env python3
"""
Temporal GCN (TGCN) baseline for time-split link prediction.

- Builds graph snapshots per time bucket.
- Trains a TGCN on training snapshots to predict edges.
- Evaluates on the first test snapshot using embeddings from the final train snapshot.
"""
import argparse
import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import degree
from torch_geometric_temporal.nn.recurrent import TGCN

import run_linear_gae_baseline as lgae


def load_edges_with_time(path: Path, years: List[int]) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if 'src' not in df.columns or 'dst' not in df.columns or 'time_bucket' not in df.columns:
        raise ValueError('Edge file must contain src, dst, and time_bucket columns')
    df = df[['src', 'dst', 'time_bucket']].dropna()
    df['time_bucket'] = pd.to_datetime(df['time_bucket'])
    if years:
        df = df[df['time_bucket'].dt.year.isin(years)]
    return df[['src', 'dst', 'time_bucket']].astype({'src': int, 'dst': int})


def time_split_buckets(buckets: List[pd.Timestamp], test_ratio: float):
    if len(buckets) < 2:
        raise ValueError('Need at least 2 time buckets for a time-based split.')
    n_test = max(1, int(len(buckets) * test_ratio))
    split_idx = max(1, len(buckets) - n_test)
    return buckets[:split_idx], buckets[split_idx:], buckets[split_idx]


def rolling_cv_folds(
    buckets: List[pd.Timestamp],
    n_folds: int,
    min_train_ratio: float = 0.5,
) -> List[Tuple[List[pd.Timestamp], List[pd.Timestamp], pd.Timestamp]]:
    """
    Generate rolling-window CV folds. Each fold uses an expanding train window
    and tests on the immediate future. Train ratios: min_train_ratio, ..., up to
    (1 - 1/n_folds) in equal steps.
    Returns list of (train_buckets, test_buckets, cutoff).
    """
    if len(buckets) < 2 or n_folds < 2:
        raise ValueError('Need at least 2 buckets and 2 folds.')
    folds = []
    for f in range(n_folds):
        # train_ratio: min_train_ratio, min+step, ..., up to ~(1 - 1/n_folds)
        train_ratio = min_train_ratio + (1 - min_train_ratio) * f / n_folds
        split_idx = max(1, int(len(buckets) * train_ratio))
        train_b = buckets[:split_idx]
        test_b = buckets[split_idx:]
        if not test_b:
            continue
        cutoff = buckets[split_idx]
        folds.append((train_b, test_b, cutoff))
    return folds


def dedupe_pairs(df: pd.DataFrame) -> List[Tuple[int, int]]:
    df = df[['src', 'dst']].copy().reset_index(drop=True)
    df[['a', 'b']] = pd.DataFrame(
        np.sort(df[['src', 'dst']].to_numpy(), axis=1),
        columns=['a', 'b']
    )
    df = df.drop_duplicates(subset=['a', 'b'])[['a', 'b']]
    return list(map(tuple, df.to_numpy()))


def build_snapshot_edges(df: pd.DataFrame, buckets: List[pd.Timestamp]) -> Dict[pd.Timestamp, List[Tuple[int, int]]]:
    snapshots: Dict[pd.Timestamp, List[Tuple[int, int]]] = {}
    for bucket in buckets:
        sub = df[df['time_bucket'] == bucket]
        pairs = dedupe_pairs(sub)
        snapshots[bucket] = pairs
    return snapshots


def edges_to_index(edges: List[Tuple[int, int]], node_to_idx: Dict[int, int]) -> torch.Tensor:
    rows = [node_to_idx[u] for u, v in edges]
    cols = [node_to_idx[v] for u, v in edges]
    if not rows:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([rows + cols, cols + rows], dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--edges', required=True)
    ap.add_argument('--years', default='2012,2013,2014,2015,2016,2017,2018,2019')
    ap.add_argument('--embedding-dim', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--test-ratio', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-report', default='artifacts/tgcn_report.json')
    ap.add_argument('--log', action='store_true')
    ap.add_argument('--log-csv', default='artifacts/experiment_log.csv')
    ap.add_argument('--log-json', default='artifacts/experiment_log.json')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    years = [int(y.strip()) for y in args.years.split(',') if y.strip()]
    df = load_edges_with_time(Path(args.edges), years)
    buckets = sorted(df['time_bucket'].unique())
    train_buckets, test_buckets, cutoff = time_split_buckets(buckets, args.test_ratio)

    train_df = df[df['time_bucket'].isin(train_buckets)]
    test_df = df[df['time_bucket'].isin(test_buckets)]

    nodes = sorted({n for pair in dedupe_pairs(df) for n in pair})
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    snapshots = build_snapshot_edges(df, train_buckets)

    model = TGCN(in_channels=1, out_channels=args.embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for _ in range(args.epochs):
        hidden = None
        for bucket in train_buckets:
            edges = snapshots.get(bucket, [])
            edge_index = edges_to_index(edges, node_to_idx)
            if edge_index.numel() == 0:
                continue
            deg = degree(edge_index[0], num_nodes=len(nodes)).unsqueeze(-1)

            hidden = model(deg, edge_index, None, hidden)

            # build training pairs for this snapshot
            pos_edges = edges
            if not pos_edges:
                continue
            existing = set((min(u, v), max(u, v)) for u, v in pos_edges)
            neg_edges = lgae.negative_sampling(nodes, existing, k=len(pos_edges), seed=args.seed)

            def score(pairs):
                scores = []
                for u, v in pairs:
                    iu, iv = node_to_idx[u], node_to_idx[v]
                    scores.append((hidden[iu] * hidden[iv]).sum())
                return torch.stack(scores)

            pos_scores = score(pos_edges)
            neg_scores = score(neg_edges)
            labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
            scores = torch.cat([pos_scores, neg_scores])
            loss = loss_fn(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            hidden = hidden.detach()

    # get embeddings after final train bucket
    hidden = None
    for bucket in train_buckets:
        edges = snapshots.get(bucket, [])
        edge_index = edges_to_index(edges, node_to_idx)
        if edge_index.numel() == 0:
            continue
        deg = degree(edge_index[0], num_nodes=len(nodes)).unsqueeze(-1)
        hidden = model(deg, edge_index, None, hidden)

    if hidden is None:
        raise ValueError('No training snapshots with edges found.')

    # evaluate on first test bucket
    test_bucket = test_buckets[0]
    train_set = set((min(u, v), max(u, v)) for u, v in dedupe_pairs(train_df))
    test_edges_all = dedupe_pairs(test_df[test_df['time_bucket'] == test_bucket])
    test_edges = [(u, v) for u, v in test_edges_all if (min(u, v), max(u, v)) not in train_set]
    if not test_edges:
        raise ValueError('No edges in the first test bucket.')
    existing = set((min(u, v), max(u, v)) for u, v in dedupe_pairs(df))
    test_neg = lgae.negative_sampling(nodes, existing, k=len(test_edges), seed=args.seed)

    scores = []
    labels = []
    for u, v in test_edges + test_neg:
        iu, iv = node_to_idx[u], node_to_idx[v]
        scores.append(float((hidden[iu] * hidden[iv]).sum().detach().cpu().item()))
        labels.append(1 if (u, v) in test_edges else 0)

    labels = np.array(labels)
    scores = np.array(scores)

    report = {
        'edges_total': len(existing),
        'train_edges': len(dedupe_pairs(train_df)),
        'test_pos': len(test_edges),
        'test_neg': len(test_neg),
        'embedding_dim': args.embedding_dim,
        'epochs': args.epochs,
        'metrics': {
            'roc_auc': lgae.roc_auc(labels, scores),
            'average_precision': lgae.average_precision(labels, scores),
        },
        'cutoff_time_bucket': str(cutoff),
    }

    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_report, 'w') as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Report written to {args.out_report}")

    if args.log:
        row = {
            'timestamp': datetime.now(UTC).isoformat().replace('+00:00', 'Z'),
            'model': 'tgcn',
            'edges': str(args.edges),
            'test_ratio': args.test_ratio,
            'seed': args.seed,
            'embedding_dim': args.embedding_dim,
            'use_features': True,
            'features_years': 'tgcn',
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
