#!/usr/bin/env python3
"""
Score candidate missing edges using the best temporal GNN baseline (TGCN + temporal node features).

This script trains on the time-split train buckets, computes node embeddings, then samples
candidate non-edges and ranks them by dot-product score.
"""
import argparse
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


def _forward_model(model, x, edge_index, hidden):
    return model(x, edge_index, None, hidden)


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
            hidden = _forward_model(model, x, edge_index, hidden)

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
        hidden = _forward_model(model, x, edge_index, hidden)
    return hidden


def sample_candidates(nodes: List[int], existing: set[tuple[int, int]], per_node: int, seed: int):
    rng = np.random.default_rng(seed)
    node_array = np.array(nodes)
    candidates = []
    for u in nodes:
        tries = 0
        while len(candidates) < per_node * len(nodes) and tries < per_node * 10:
            v = int(rng.choice(node_array))
            if u == v:
                tries += 1
                continue
            pair = (min(u, v), max(u, v))
            if pair in existing:
                tries += 1
                continue
            candidates.append((u, v))
            if len([c for c in candidates if c[0] == u]) >= per_node:
                break
        if per_node == 0:
            break
    return candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--edges', required=True)
    ap.add_argument('--years', default='2012,2013,2014,2015,2016,2017,2018,2019')
    ap.add_argument('--embedding-dim', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--test-ratio', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--seeds', default='', help='Comma-separated seeds for ensemble (e.g. 1,2,3); overrides --seed if set')
    ap.add_argument('--candidates-per-node', type=int, default=20)
    ap.add_argument('--top-k', type=int, default=200)
    ap.add_argument('--use-temporal-node-features', action='store_true')
    ap.add_argument('--max-train-buckets', type=int, default=None, help='Limit train buckets to reduce memory')
    ap.add_argument('--out', default='artifacts/tgcn_candidate_scores.parquet')
    args = ap.parse_args()

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

    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()] if args.seeds else [args.seed]
    candidate_seed = seeds[0]  # use first seed for consistent candidate sampling

    candidates = sample_candidates(nodes, train_set, args.candidates_per_node, candidate_seed)
    if not candidates:
        raise ValueError('No candidate pairs sampled.')

    all_scores = []
    for seed in seeds:
        model = train_tgcn(
            nodes=nodes,
            node_to_idx=node_to_idx,
            train_snapshots=train_snapshots,
            train_buckets=train_buckets,
            embedding_dim=args.embedding_dim,
            in_channels=in_channels,
            epochs=args.epochs,
            lr=args.lr,
            seed=seed,
            static_features=static_features,
        )
        hidden = rollout_hidden(model, nodes, node_to_idx, train_snapshots, train_buckets, static_features)
        if hidden is None:
            raise ValueError('No training snapshots with edges found.')
        seed_scores = []
        for u, v in candidates:
            iu, iv = node_to_idx[u], node_to_idx[v]
            seed_scores.append(float((hidden[iu] * hidden[iv]).sum().detach().cpu().item()))
        all_scores.append(seed_scores)

    # Average scores across seeds (ensemble)
    scores = np.mean(all_scores, axis=0) if len(all_scores) > 1 else all_scores[0]

    out_df = pd.DataFrame(candidates, columns=['src', 'dst'])
    out_df['score'] = scores
    out_df = out_df.sort_values('score', ascending=False).head(args.top_k)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print(f"Wrote top-{args.top_k} candidate scores to {out_path}")


if __name__ == '__main__':
    main()
