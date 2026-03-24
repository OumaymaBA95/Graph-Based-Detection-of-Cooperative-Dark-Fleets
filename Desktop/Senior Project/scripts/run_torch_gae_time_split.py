#!/usr/bin/env python3
"""
Torch GAE prototype with time-based train/test split.
"""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

import train_torch_gae as tgae


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
    ap.add_argument('--embedding-dim', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--test-ratio', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-report', default='artifacts/torch_gae_time_report.json')
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
    train_edges = [(node_to_idx[u], node_to_idx[v]) for u, v in train_pairs]
    test_pos_idx = [(node_to_idx[u], node_to_idx[v]) for u, v in test_pairs]

    existing = set((min(u, v), max(u, v)) for u, v in dedupe_pairs(df))

    test_neg = tgae.negative_sampling(nodes, existing, k=len(test_pairs), seed=args.seed)
    test_neg_idx = [(node_to_idx[u], node_to_idx[v]) for u, v in test_neg]

    torch.manual_seed(args.seed)
    emb = torch.nn.Embedding(len(nodes), args.embedding_dim)
    optimizer = torch.optim.Adam(emb.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    pos_edges = torch.tensor(train_edges, dtype=torch.long)

    for _ in range(args.epochs):
        optimizer.zero_grad()
        neg_edges = tgae.negative_sampling(nodes, existing, k=len(train_edges), seed=np.random.randint(0, 1_000_000))
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
        'edges_total': len(existing),
        'train_edges': len(train_pairs),
        'test_pos': len(test_pairs),
        'test_neg': len(test_neg),
        'embedding_dim': args.embedding_dim,
        'epochs': args.epochs,
        'lr': args.lr,
        'years': years,
        'cutoff_time_bucket': str(cutoff),
        'metrics': {
            'roc_auc': tgae.roc_auc(labels, scores),
            'average_precision': tgae.average_precision(labels, scores),
        },
    }

    with open(args.out_report, 'w') as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Report written to {args.out_report}")

    if args.log:
        row = {
            'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'model': 'torch_gae_time',
            'edges': args.edges,
            'test_ratio': args.test_ratio,
            'seed': args.seed,
            'embedding_dim': args.embedding_dim,
            'use_features': False,
            'features_years': args.years,
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
