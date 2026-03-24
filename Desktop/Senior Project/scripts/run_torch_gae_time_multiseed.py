#!/usr/bin/env python3
"""
Run the Torch GAE time-split baseline across multiple seeds.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import train_torch_gae as tgae
import run_torch_gae_time_split as time_split


def run_once(seed, args, train_pairs, test_pairs, all_pairs, nodes):
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    train_edges = [(node_to_idx[u], node_to_idx[v]) for u, v in train_pairs]
    test_pos_idx = [(node_to_idx[u], node_to_idx[v]) for u, v in test_pairs]

    existing = set((min(u, v), max(u, v)) for u, v in all_pairs)
    test_neg = tgae.negative_sampling(nodes, existing, k=len(test_pairs), seed=seed)
    test_neg_idx = [(node_to_idx[u], node_to_idx[v]) for u, v in test_neg]

    torch.manual_seed(seed)
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

    return {
        'seed': seed,
        'metrics': {
            'roc_auc': tgae.roc_auc(labels, scores),
            'average_precision': tgae.average_precision(labels, scores),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--edges', required=True)
    ap.add_argument('--years', default='2012,2013,2014,2015,2016,2017,2018,2019')
    ap.add_argument('--embedding-dim', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--test-ratio', type=float, default=0.3)
    ap.add_argument('--seeds', default='1,2,3')
    ap.add_argument('--out-report', default='artifacts/torch_gae_time_multiseed.json')
    ap.add_argument('--out-csv', default='artifacts/torch_gae_time_multiseed.csv')
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

    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    results = [run_once(seed, args, train_pairs, test_pairs, all_pairs, nodes) for seed in seeds]

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
        'epochs': args.epochs,
        'lr': args.lr,
        'test_ratio': args.test_ratio,
        'seeds': seeds,
        'cutoff_time_bucket': str(cutoff),
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
            'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'model': 'torch_gae_time_multiseed',
            'edges': args.edges,
            'test_ratio': args.test_ratio,
            'seed': ';'.join(map(str, seeds)),
            'embedding_dim': args.embedding_dim,
            'use_features': False,
            'features_years': ','.join(map(str, years)),
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
