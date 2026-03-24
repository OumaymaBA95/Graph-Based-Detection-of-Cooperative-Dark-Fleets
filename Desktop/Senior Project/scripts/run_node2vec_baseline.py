#!/usr/bin/env python3
"""
Node2Vec/DeepWalk-style baseline for link prediction.

- Builds random-walk embeddings using Word2Vec.
- Scores candidate edges by dot product of embeddings.
- Supports random train/test split or time-based split.
"""
import argparse
import json
import os
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import networkx as nx
from gensim.models import Word2Vec

import run_linear_gae_baseline as lgae


def load_edges(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if 'src' not in df.columns or 'dst' not in df.columns:
        raise ValueError('Edge file must contain src and dst columns')
    return df[['src', 'dst']].dropna().astype(int)


def load_edges_with_time(path: Path, years: List[int]) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if 'src' not in df.columns or 'dst' not in df.columns or 'time_bucket' not in df.columns:
        raise ValueError('Edge file must contain src, dst, and time_bucket columns')
    df = df[['src', 'dst', 'time_bucket']].dropna()
    df['time_bucket'] = pd.to_datetime(df['time_bucket'])
    if years:
        df = df[df['time_bucket'].dt.year.isin(years)]
    return df[['src', 'dst', 'time_bucket']].astype({'src': int, 'dst': int})


def dedupe_edges(df: pd.DataFrame) -> List[Tuple[int, int]]:
    df = df[['src', 'dst']].copy().reset_index(drop=True)
    df[['a', 'b']] = pd.DataFrame(
        np.sort(df[['src', 'dst']].to_numpy(), axis=1),
        columns=['a', 'b']
    )
    df = df.drop_duplicates(subset=['a', 'b'])[['a', 'b']]
    return list(map(tuple, df.to_numpy()))


def train_test_split_edges(edges: List[Tuple[int, int]], test_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    edges = edges.copy()
    rng.shuffle(edges)
    n_test = max(1, int(len(edges) * test_ratio))
    return edges[n_test:], edges[:n_test]


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


def random_walk(
    rng: np.random.Generator,
    neighbors: Dict[int, List[int]],
    start: int,
    walk_length: int,
    p: float,
    q: float,
) -> List[int]:
    walk = [start]
    while len(walk) < walk_length:
        cur = walk[-1]
        nbrs = neighbors.get(cur, [])
        if not nbrs:
            break
        if len(walk) == 1:
            nxt = rng.choice(nbrs)
        else:
            prev = walk[-2]
            weights = []
            prev_neighbors = set(neighbors.get(prev, []))
            for nbr in nbrs:
                if nbr == prev:
                    weights.append(1.0 / p)
                elif nbr in prev_neighbors:
                    weights.append(1.0)
                else:
                    weights.append(1.0 / q)
            weights = np.array(weights, dtype=float)
            weights = weights / weights.sum()
            nxt = rng.choice(nbrs, p=weights)
        walk.append(int(nxt))
    return walk


def generate_walks(
    nodes: List[int],
    neighbors: Dict[int, List[int]],
    num_walks: int,
    walk_length: int,
    seed: int,
    p: float,
    q: float,
) -> List[List[str]]:
    rng = np.random.default_rng(seed)
    walks: List[List[str]] = []
    for _ in range(num_walks):
        rng.shuffle(nodes)
        for node in nodes:
            walk = random_walk(rng, neighbors, node, walk_length, p, q)
            walks.append([str(n) for n in walk])
    return walks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--edges', required=True)
    ap.add_argument('--years', default='2012,2013,2014,2015,2016,2017,2018,2019')
    ap.add_argument('--embedding-dim', type=int, default=64)
    ap.add_argument('--num-walks', type=int, default=10)
    ap.add_argument('--walk-length', type=int, default=20)
    ap.add_argument('--window-size', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--p', type=float, default=1.0)
    ap.add_argument('--q', type=float, default=1.0)
    ap.add_argument('--test-ratio', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--time-split', action='store_true')
    ap.add_argument('--out-report', default='artifacts/node2vec_report.json')
    ap.add_argument('--log', action='store_true')
    ap.add_argument('--log-csv', default='artifacts/experiment_log.csv')
    ap.add_argument('--log-json', default='artifacts/experiment_log.json')
    args = ap.parse_args()

    os.makedirs(Path(args.out_report).parent, exist_ok=True)

    years = [int(y.strip()) for y in args.years.split(',') if y.strip()]
    cutoff = None
    if args.time_split:
        df = load_edges_with_time(Path(args.edges), years)
        train_df, test_df, cutoff = time_split(df, args.test_ratio)
        train_edges = dedupe_edges(train_df)
        test_edges = dedupe_edges(test_df)
        train_set = set((min(u, v), max(u, v)) for u, v in train_edges)
        test_pos = [(u, v) for u, v in test_edges if (min(u, v), max(u, v)) not in train_set]
        all_edges = dedupe_edges(df)
    else:
        df = load_edges(Path(args.edges))
        all_edges = dedupe_edges(df)
        train_edges, test_pos = train_test_split_edges(all_edges, args.test_ratio, args.seed)

    if not train_edges or not test_pos:
        raise ValueError('Train/test split produced empty edges.')

    G = nx.Graph()
    G.add_edges_from(train_edges)
    nodes = list(G.nodes())
    neighbors = {n: list(G.neighbors(n)) for n in nodes}

    walks = generate_walks(nodes, neighbors, args.num_walks, args.walk_length, args.seed, args.p, args.q)
    model = Word2Vec(
        sentences=walks,
        vector_size=args.embedding_dim,
        window=args.window_size,
        min_count=0,
        sg=1,
        workers=1,
        epochs=args.epochs,
        seed=args.seed,
    )

    def embed(node: int) -> np.ndarray:
        key = str(node)
        if key in model.wv:
            return model.wv[key]
        return np.zeros(args.embedding_dim, dtype=float)

    existing = set((min(u, v), max(u, v)) for u, v in all_edges)
    test_neg = lgae.negative_sampling(nodes, existing, k=len(test_pos), seed=args.seed)

    def score(u: int, v: int) -> float:
        return float(np.dot(embed(u), embed(v)))

    labels = np.array([1] * len(test_pos) + [0] * len(test_neg))
    scores = np.array([score(u, v) for u, v in test_pos + test_neg])

    report = {
        'edges_total': len(all_edges),
        'train_edges': len(train_edges),
        'test_pos': len(test_pos),
        'test_neg': len(test_neg),
        'embedding_dim': args.embedding_dim,
        'num_walks': args.num_walks,
        'walk_length': args.walk_length,
        'window_size': args.window_size,
        'epochs': args.epochs,
        'p': args.p,
        'q': args.q,
        'years': years,
        'time_split': bool(args.time_split),
        'cutoff_time_bucket': str(cutoff) if cutoff is not None else None,
        'metrics': {
            'roc_auc': lgae.roc_auc(labels, scores),
            'average_precision': lgae.average_precision(labels, scores),
        },
    }

    with open(args.out_report, 'w') as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Report written to {args.out_report}")

    if args.log:
        timestamp = datetime.now(UTC).isoformat().replace('+00:00', 'Z')
        model_name = 'node2vec_time' if args.time_split else 'node2vec'
        row = {
            'timestamp': timestamp,
            'model': model_name,
            'edges': str(args.edges),
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
