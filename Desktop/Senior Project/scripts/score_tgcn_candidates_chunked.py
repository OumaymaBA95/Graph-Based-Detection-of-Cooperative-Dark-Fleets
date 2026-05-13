#!/usr/bin/env python3
"""
Chunked candidate scoring for full-coverage edges to avoid OOM.
"""
import argparse
import heapq
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import degree

from build_temporal_node_features import build_temporal_node_features
import run_tgcn_link_pred as tgcn_utils


def _edge_set(pairs: List[Tuple[int, int]]) -> set[tuple[int, int]]:
    return set((min(u, v), max(u, v)) for u, v in pairs)


def _standardize_features(features: np.ndarray) -> np.ndarray:
    means = np.nanmean(features, axis=0)
    stds = np.nanstd(features, axis=0)
    stds = np.where(stds == 0, 1.0, stds)
    standardized = (features - means) / stds
    return np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0)


def _node_inputs(edge_index: torch.Tensor, num_nodes: int, static_features: torch.Tensor | None):
    deg = degree(edge_index[0], num_nodes=num_nodes).unsqueeze(-1)
    if static_features is None:
        return deg
    return torch.cat([deg, static_features], dim=1)


def train_tgcn(
    nodes: List[int],
    node_to_idx: Dict[int, int],
    train_snapshots: Dict[pd.Timestamp, List[Tuple[int, int]]],
    train_buckets: List[pd.Timestamp],
    embedding_dim: int,
    in_channels: int,
    epochs: int,
    lr: float,
    seed: int,
    static_features: torch.Tensor | None,
) -> torch.nn.Module:
    from temporal_graph_baselines import TGCNPure

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = TGCNPure(in_channels=in_channels, out_channels=embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        hidden = None
        for bucket in train_buckets:
            edges = train_snapshots.get(bucket, [])
            edge_index = tgcn_utils.edges_to_index(edges, node_to_idx)
            if edge_index.numel() == 0:
                continue
            x = _node_inputs(edge_index, len(nodes), static_features)
            hidden = model(x, edge_index, None, hidden)

            if not edges:
                continue
            existing = _edge_set(edges)
            neg_edges = tgcn_utils.lgae.negative_sampling(nodes, existing, k=len(edges), seed=seed)

            def score(pairs):
                scores = []
                for u, v in pairs:
                    iu, iv = node_to_idx[u], node_to_idx[v]
                    scores.append((hidden[iu] * hidden[iv]).sum())
                return torch.stack(scores)

            pos_scores = score(edges)
            neg_scores = score(neg_edges)
            labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
            scores = torch.cat([pos_scores, neg_scores])
            loss = loss_fn(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            hidden = hidden.detach()

    return model


def rollout_hidden(
    model: torch.nn.Module,
    nodes: List[int],
    node_to_idx: Dict[int, int],
    snapshots: Dict[pd.Timestamp, List[Tuple[int, int]]],
    buckets: List[pd.Timestamp],
    static_features: torch.Tensor | None,
):
    hidden = None
    for bucket in buckets:
        edges = snapshots.get(bucket, [])
        edge_index = tgcn_utils.edges_to_index(edges, node_to_idx)
        if edge_index.numel() == 0:
            continue
        x = _node_inputs(edge_index, len(nodes), static_features)
        hidden = model(x, edge_index, None, hidden)
    return hidden


def update_stats(stats: Dict[str, float], scores: np.ndarray):
    stats['count'] += scores.size
    stats['sum'] += float(scores.sum())
    stats['sumsq'] += float((scores ** 2).sum())
    stats['min'] = min(stats['min'], float(scores.min()))
    stats['max'] = max(stats['max'], float(scores.max()))


def sample_candidates(nodes: List[int], existing: set[tuple[int, int]], per_node: int, rng: np.random.Generator):
    node_array = np.array(nodes)
    for u in nodes:
        picked = 0
        attempts = 0
        while picked < per_node and attempts < per_node * 10:
            v = int(rng.choice(node_array))
            attempts += 1
            if u == v:
                continue
            pair = (min(u, v), max(u, v))
            if pair in existing:
                continue
            yield u, v
            picked += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--edges', required=True)
    ap.add_argument('--years', default='2012,2013,2014,2015,2016,2017,2018,2019')
    ap.add_argument('--embedding-dim', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--test-ratio', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--candidates-per-node', type=int, default=5)
    ap.add_argument('--top-k', type=int, default=1000)
    ap.add_argument('--batch-size', type=int, default=5000)
    ap.add_argument('--use-temporal-node-features', action='store_true')
    ap.add_argument('--max-train-buckets', type=int, default=None, help='Limit train buckets to reduce memory')
    ap.add_argument('--out', default='artifacts/tgcn_candidate_scores_fullcoverage_top1000.parquet')
    ap.add_argument('--out-stats', default='artifacts/tgcn_candidate_score_stats.json')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    years = [int(y.strip()) for y in args.years.split(',') if y.strip()]
    df = tgcn_utils.load_edges_with_time(Path(args.edges), years)
    buckets = sorted(df['time_bucket'].unique())
    train_buckets, _, cutoff = tgcn_utils.time_split_buckets(buckets, args.test_ratio)
    if args.max_train_buckets is not None and len(train_buckets) > args.max_train_buckets:
        train_buckets = train_buckets[: args.max_train_buckets]

    train_df = df[df['time_bucket'].isin(train_buckets)]
    train_pairs = tgcn_utils.dedupe_pairs(train_df)
    train_set = _edge_set(train_pairs)

    nodes = sorted({n for pair in tgcn_utils.dedupe_pairs(df) for n in pair})
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    static_features = None
    if args.use_temporal_node_features:
        temporal_feats = build_temporal_node_features(train_df, cutoff)
        temporal_feats = temporal_feats.set_index('MMSI')
        feature_cols = ['interactions_count', 'unique_partners', 'last_seen_days', 'mean_gap_days']
        feat_matrix = temporal_feats.reindex(nodes)[feature_cols].to_numpy()
        feat_matrix = _standardize_features(feat_matrix)
        static_features = torch.tensor(feat_matrix, dtype=torch.float32)

    in_channels = 1 + (0 if static_features is None else static_features.shape[1])
    train_snapshots = tgcn_utils.build_snapshot_edges(df, train_buckets)

    model = train_tgcn(
        nodes=nodes,
        node_to_idx=node_to_idx,
        train_snapshots=train_snapshots,
        train_buckets=train_buckets,
        embedding_dim=args.embedding_dim,
        in_channels=in_channels,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        static_features=static_features,
    )

    hidden = rollout_hidden(model, nodes, node_to_idx, train_snapshots, train_buckets, static_features)
    if hidden is None:
        raise ValueError('No training snapshots with edges found.')

    heap: List[Tuple[float, int, int]] = []
    stats = {'count': 0, 'sum': 0.0, 'sumsq': 0.0, 'min': float('inf'), 'max': float('-inf')}

    batch_pairs: List[Tuple[int, int]] = []
    for u, v in sample_candidates(nodes, train_set, args.candidates_per_node, rng):
        batch_pairs.append((u, v))
        if len(batch_pairs) >= args.batch_size:
            scores = []
            for a, b in batch_pairs:
                ia, ib = node_to_idx[a], node_to_idx[b]
                scores.append(float((hidden[ia] * hidden[ib]).sum().item()))
            scores_arr = np.array(scores)
            update_stats(stats, scores_arr)
            for (a, b), score in zip(batch_pairs, scores_arr):
                entry = (score, a, b)
                if len(heap) < args.top_k:
                    heapq.heappush(heap, entry)
                elif score > heap[0][0]:
                    heapq.heapreplace(heap, entry)
            batch_pairs = []

    if batch_pairs:
        scores = []
        for a, b in batch_pairs:
            ia, ib = node_to_idx[a], node_to_idx[b]
            scores.append(float((hidden[ia] * hidden[ib]).sum().item()))
        scores_arr = np.array(scores)
        update_stats(stats, scores_arr)
        for (a, b), score in zip(batch_pairs, scores_arr):
            entry = (score, a, b)
            if len(heap) < args.top_k:
                heapq.heappush(heap, entry)
            elif score > heap[0][0]:
                heapq.heapreplace(heap, entry)

    top = sorted(heap, key=lambda x: -x[0])
    out_df = pd.DataFrame(top, columns=['score', 'src', 'dst'])[['src', 'dst', 'score']]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    stats_out = {
        'count': stats['count'],
        'mean': stats['sum'] / max(1, stats['count']),
        'std': float(np.sqrt(stats['sumsq'] / max(1, stats['count']) - (stats['sum'] / max(1, stats['count'])) ** 2)),
        'min': stats['min'],
        'max': stats['max'],
    }
    stats_path = Path(args.out_stats)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats_out, indent=2))

    print(f"Wrote {out_path}")
    print(f"Wrote {stats_path}")


if __name__ == '__main__':
    main()
