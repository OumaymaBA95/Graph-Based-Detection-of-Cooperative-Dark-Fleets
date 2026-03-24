#!/usr/bin/env python3
"""
Temporal GNN-lite baseline (TGN-inspired) using PyTorch only.

- Maintains node memories updated sequentially by observed edges.
- Uses a simple message MLP + GRUCell for memory updates.
- Scores edges with an MLP on concatenated memories + time deltas.
- Evaluates on a time-based split.
"""
import argparse
import json
from datetime import datetime, UTC
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

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


def dedupe_pairs(df: pd.DataFrame) -> List[Tuple[int, int]]:
    df = df[['src', 'dst']].copy().reset_index(drop=True)
    df[['a', 'b']] = pd.DataFrame(
        np.sort(df[['src', 'dst']].to_numpy(), axis=1),
        columns=['a', 'b']
    )
    df = df.drop_duplicates(subset=['a', 'b'])[['a', 'b']]
    return list(map(tuple, df.to_numpy()))


def recency_feature(last_time, current_time, max_days):
    if last_time is None:
        delta = max_days
    else:
        delta = (current_time - last_time).days
    return 1.0 / (1.0 + max(0.0, delta))


class TGNLite(torch.nn.Module):
    def __init__(self, num_nodes: int, emb_dim: int):
        super().__init__()
        self.emb = torch.nn.Embedding(num_nodes, emb_dim)
        self.msg_mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim * 2 + 1, emb_dim),
            torch.nn.ReLU(),
        )
        self.gru = torch.nn.GRUCell(emb_dim, emb_dim)
        self.score_mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim * 2 + 2, emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, 1),
        )

    def forward_score(self, mem_u, mem_v, rec_u, rec_v):
        feats = torch.cat([mem_u, mem_v, rec_u, rec_v], dim=-1)
        return self.score_mlp(feats).squeeze(-1)

    def update_memory(self, mem_u, mem_v, rec_u, rec_v):
        msg_u = torch.cat([mem_u, mem_v, rec_u], dim=-1)
        msg_v = torch.cat([mem_v, mem_u, rec_v], dim=-1)
        msg_u = self.msg_mlp(msg_u)
        msg_v = self.msg_mlp(msg_v)
        new_u = self.gru(msg_u, mem_u)
        new_v = self.gru(msg_v, mem_v)
        return new_u, new_v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--edges', required=True)
    ap.add_argument('--years', default='2012,2013,2014,2015,2016,2017,2018,2019')
    ap.add_argument('--embedding-dim', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--test-ratio', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-report', default='artifacts/tgn_lite_report.json')
    ap.add_argument('--log', action='store_true')
    ap.add_argument('--log-csv', default='artifacts/experiment_log.csv')
    ap.add_argument('--log-json', default='artifacts/experiment_log.json')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    years = [int(y.strip()) for y in args.years.split(',') if y.strip()]
    df = load_edges_with_time(Path(args.edges), years)
    train_df, test_df, cutoff = time_split(df, args.test_ratio)

    train_df = train_df.sort_values('time_bucket')
    test_df = test_df.sort_values('time_bucket')

    train_pairs = dedupe_pairs(train_df)
    test_pairs_all = dedupe_pairs(test_df)
    train_set = set((min(u, v), max(u, v)) for u, v in train_pairs)
    test_pairs = [(u, v) for u, v in test_pairs_all if (min(u, v), max(u, v)) not in train_set]

    if not train_pairs or not test_pairs:
        raise ValueError('Time split produced empty train or test edges.')

    nodes = sorted({n for pair in dedupe_pairs(df) for n in pair})
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    model = TGNLite(len(nodes), args.embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    max_days = max(1, int((df['time_bucket'].max() - df['time_bucket'].min()).days))

    for _ in range(args.epochs):
        model.train()
        last_time = {n: None for n in nodes}
        mem = model.emb.weight

        losses = []
        for _, row in train_df.iterrows():
            u, v = int(row['src']), int(row['dst'])
            t = row['time_bucket']
            iu, iv = node_to_idx[u], node_to_idx[v]

            rec_u = torch.tensor([[recency_feature(last_time[u], t, max_days)]], dtype=torch.float32)
            rec_v = torch.tensor([[recency_feature(last_time[v], t, max_days)]], dtype=torch.float32)

            mem_u = mem[iu].unsqueeze(0)
            mem_v = mem[iv].unsqueeze(0)
            score_pos = model.forward_score(mem_u, mem_v, rec_u, rec_v)

            # negative sampling
            neg = v
            while neg == v:
                neg = np.random.choice(nodes)
            ineg = node_to_idx[neg]
            mem_neg = mem[ineg].unsqueeze(0)
            rec_neg = torch.tensor([[recency_feature(last_time[neg], t, max_days)]], dtype=torch.float32)
            score_neg = model.forward_score(mem_u, mem_neg, rec_u, rec_neg)

            labels = torch.tensor([1.0, 0.0])
            scores = torch.cat([score_pos, score_neg])
            loss = loss_fn(scores, labels)
            losses.append(loss)

            if len(losses) >= 32:
                optimizer.zero_grad()
                torch.stack(losses).mean().backward()
                optimizer.step()
                losses = []

            with torch.no_grad():
                new_u, new_v = model.update_memory(mem_u, mem_v, rec_u, rec_v)
                mem = mem.clone()
                mem[iu] = new_u.squeeze(0)
                mem[iv] = new_v.squeeze(0)
                last_time[u] = t
                last_time[v] = t

        if losses:
            optimizer.zero_grad()
            torch.stack(losses).mean().backward()
            optimizer.step()

    # evaluation
    model.eval()
    last_time = {n: None for n in nodes}
    mem = model.emb.weight.detach().clone()
    for _, row in train_df.iterrows():
        u, v = int(row['src']), int(row['dst'])
        t = row['time_bucket']
        iu, iv = node_to_idx[u], node_to_idx[v]
        rec_u = torch.tensor([[recency_feature(last_time[u], t, max_days)]], dtype=torch.float32)
        rec_v = torch.tensor([[recency_feature(last_time[v], t, max_days)]], dtype=torch.float32)
        mem_u = mem[iu].unsqueeze(0)
        mem_v = mem[iv].unsqueeze(0)
        new_u, new_v = model.update_memory(mem_u, mem_v, rec_u, rec_v)
        mem[iu] = new_u.squeeze(0)
        mem[iv] = new_v.squeeze(0)
        last_time[u] = t
        last_time[v] = t

    existing = set((min(u, v), max(u, v)) for u, v in dedupe_pairs(df))
    test_neg = lgae.negative_sampling(nodes, existing, k=len(test_pairs), seed=args.seed)

    scores = []
    labels = []
    all_pairs = test_pairs + test_neg
    for idx, (u, v) in enumerate(all_pairs):
        t = test_df['time_bucket'].iloc[-1]
        iu, iv = node_to_idx[u], node_to_idx[v]
        rec_u = torch.tensor([[recency_feature(last_time[u], t, max_days)]], dtype=torch.float32)
        rec_v = torch.tensor([[recency_feature(last_time[v], t, max_days)]], dtype=torch.float32)
        mem_u = mem[iu].unsqueeze(0)
        mem_v = mem[iv].unsqueeze(0)
        score = model.forward_score(mem_u, mem_v, rec_u, rec_v)
        scores.append(float(score.detach().cpu().item()))
        labels.append(1 if idx < len(test_pairs) else 0)

    labels = np.array(labels)
    scores = np.array(scores)

    report = {
        'edges_total': len(existing),
        'train_edges': len(train_pairs),
        'test_pos': len(test_pairs),
        'test_neg': len(test_neg),
        'embedding_dim': args.embedding_dim,
        'epochs': args.epochs,
        'metrics': {
            'roc_auc': lgae.roc_auc(labels, scores),
            'average_precision': lgae.average_precision(labels, scores),
        },
    }

    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_report, 'w') as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Report written to {args.out_report}")

    if args.log:
        row = {
            'timestamp': datetime.now(UTC).isoformat().replace('+00:00', 'Z'),
            'model': 'tgn_lite',
            'edges': str(args.edges),
            'test_ratio': args.test_ratio,
            'seed': args.seed,
            'embedding_dim': args.embedding_dim,
            'use_features': True,
            'features_years': 'tgn_lite',
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
