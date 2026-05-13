#!/usr/bin/env python3
"""
Run a temporal GNN time-split baseline across multiple seeds and test buckets.

Models ``gconvgru`` and ``gconvlstm`` import ``torch_geometric_temporal`` (needs
``torch_sparse`` on many installs). ``tgcn_pyg`` is the same T-GCN cell using only
``GCNConv`` (no ``torch_sparse``). Library ``tgcn`` still uses the optional
``torch_geometric_temporal`` package when that stack is installed correctly.
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

from build_temporal_node_features import build_temporal_node_features
from temporal_graph_baselines import TemporalGCNGRU, TemporalGraphTransformerGRU, TGCNPure

import run_linear_gae_baseline as lgae
import run_tgcn_link_pred as tgcn_utils


def _edge_set(pairs: List[Tuple[int, int]]) -> set[tuple[int, int]]:
    return set((min(u, v), max(u, v)) for u, v in pairs)


def _hidden_tensor(hidden):
    if isinstance(hidden, tuple):
        return hidden[0]
    return hidden


def _forward_model(model, x, edge_index, hidden, model_name: str):
    if model_name == 'gconvlstm':
        if hidden is None:
            return model(x, edge_index, None, None, None)
        return model(x, edge_index, None, hidden[0], hidden[1])
    return model(x, edge_index, None, hidden)


def _score_pairs(hidden, node_to_idx: Dict[int, int], pairs: List[Tuple[int, int]]):
    hidden_tensor = _hidden_tensor(hidden)
    scores = []
    for u, v in pairs:
        iu, iv = node_to_idx[u], node_to_idx[v]
        scores.append((hidden_tensor[iu] * hidden_tensor[iv]).sum())
    return torch.stack(scores)


def _standardize_features(features: np.ndarray) -> np.ndarray:
    means = np.nanmean(features, axis=0)
    stds = np.nanstd(features, axis=0)
    stds = np.where(stds == 0, 1.0, stds)
    standardized = (features - means) / stds
    return np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0)


def _load_vessel_day_features(
    features_root: Path,
    years: List[int],
    feature_cols: List[str],
) -> pd.DataFrame:
    frames = []
    for year in years:
        path = features_root / str(year) / "vessel_day_features.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "MMSI" not in df.columns:
            raise ValueError(f"MMSI column missing in {path}")
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {path}: {missing}")
        df = df[["MMSI"] + feature_cols]
        df = df.groupby("MMSI", as_index=False)[feature_cols].mean()
        frames.append(df)
    if not frames:
        raise ValueError("No vessel_day_features.parquet files found for the selected years.")
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.groupby("MMSI", as_index=False)[feature_cols].mean()
    return merged


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
    model_cls,
    model_name: str,
    cheb_k: int,
    static_features: torch.Tensor | None,
    use_hard_negatives: bool = False,
    graph_transformer_heads: int = 4,
) -> torch.nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)

    if model_name in {'gconvgru', 'gconvlstm'}:
        model = model_cls(in_channels=in_channels, out_channels=embedding_dim, K=cheb_k)
    elif model_name == 'graph_transformer':
        model = model_cls(
            in_channels=in_channels,
            out_channels=embedding_dim,
            heads=graph_transformer_heads,
        )
    elif model_name == 'gcn':
        model = model_cls(in_channels=in_channels, out_channels=embedding_dim)
    else:
        model = model_cls(in_channels=in_channels, out_channels=embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    existing_full = _edge_set([
        pair for bucket in train_buckets
        for pair in train_snapshots.get(bucket, [])
    ])
    all_train_edges = [
        pair for bucket in train_buckets
        for pair in train_snapshots.get(bucket, [])
    ]

    for _ in range(epochs):
        model.train()
        hidden = None
        for bucket in train_buckets:
            edges = train_snapshots.get(bucket, [])
            if not edges:
                continue
            edge_index = tgcn_utils.edges_to_index(edges, node_to_idx)
            if edge_index.numel() == 0:
                continue

            x = _node_inputs(edge_index, len(nodes), static_features)
            hidden = _forward_model(model, x, edge_index, hidden, model_name)

            # positive/negative pairs for this bucket
            pos_pairs = tgcn_utils.dedupe_pairs(pd.DataFrame(edges, columns=['src', 'dst']))
            if use_hard_negatives and all_train_edges:
                neg_pairs = lgae.hard_negative_sampling(
                    nodes, existing_full, all_train_edges, k=len(pos_pairs), seed=seed
                )
            else:
                neg_pairs = lgae.negative_sampling(nodes, existing_full, k=len(pos_pairs), seed=seed)

            pos_scores = _score_pairs(hidden, node_to_idx, pos_pairs)
            neg_scores = _score_pairs(hidden, node_to_idx, neg_pairs)

            labels = torch.cat([
                torch.ones(len(pos_pairs)),
                torch.zeros(len(neg_pairs)),
            ])
            scores = torch.cat([pos_scores, neg_scores])

            loss = loss_fn(scores, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Important: detach hidden state so we don't backprop through
            # the entire history of buckets (prevents "backward through the graph a second time").
            if hidden is not None:
                if isinstance(hidden, tuple):
                    hidden = tuple(h.detach() for h in hidden)
                else:
                    hidden = hidden.detach()

    return model


def rollout_hidden(
    model: torch.nn.Module,
    nodes: List[int],
    node_to_idx: Dict[int, int],
    snapshots: Dict[pd.Timestamp, List[Tuple[int, int]]],
    buckets: List[pd.Timestamp],
    model_name: str,
    static_features: torch.Tensor | None,
):
    hidden = None
    for bucket in buckets:
        edges = snapshots.get(bucket, [])
        edge_index = tgcn_utils.edges_to_index(edges, node_to_idx)
        if edge_index.numel() == 0:
            continue
        x = _node_inputs(edge_index, len(nodes), static_features)
        hidden = _forward_model(model, x, edge_index, hidden, model_name)
    return hidden


def evaluate_buckets(
    model: torch.nn.Module,
    nodes: List[int],
    node_to_idx: Dict[int, int],
    df: pd.DataFrame,
    train_set: set[tuple[int, int]],
    test_buckets: List[pd.Timestamp],
    start_hidden: torch.Tensor,
    max_buckets: int | None,
    seed: int,
    model_name: str,
    static_features: torch.Tensor | None,
    edge_output_path: Path | None = None,
):
    results = []
    edge_rows: list[dict] = []
    hidden = start_hidden
    all_pairs = tgcn_utils.dedupe_pairs(df)
    existing_full = _edge_set(all_pairs)

    model.eval()
    with torch.no_grad():
        for bucket in test_buckets:
            if max_buckets is not None and len(results) >= max_buckets:
                break
            bucket_edges_all = tgcn_utils.dedupe_pairs(df[df['time_bucket'] == bucket])
            bucket_edges = [
                (u, v)
                for u, v in bucket_edges_all
                if (min(u, v), max(u, v)) not in train_set
            ]
            if not bucket_edges:
                continue

            neg_edges = lgae.negative_sampling(nodes, existing_full, k=len(bucket_edges), seed=seed)

            pos_scores = _score_pairs(hidden, node_to_idx, bucket_edges)
            neg_scores = _score_pairs(hidden, node_to_idx, neg_edges)

            labels = np.concatenate([
                np.ones(len(bucket_edges)),
                np.zeros(len(neg_edges)),
            ])
            scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()

            # Optionally collect per-edge outputs for inspection.
            if edge_output_path is not None:
                bucket_str = str(bucket)
                # first positives then negatives, aligned with labels / scores
                for (u, v), label, score in zip(
                    list(bucket_edges) + list(neg_edges),
                    labels,
                    scores,
                ):
                    edge_rows.append(
                        {
                            "src": int(u),
                            "dst": int(v),
                            "bucket": bucket_str,
                            "label": int(label),
                            "score": float(score),
                        }
                    )

            metrics = {
                'roc_auc': lgae.roc_auc(labels, scores),
                'average_precision': lgae.average_precision(labels, scores),
                'bucket': str(bucket),
                'pos_edges': len(bucket_edges),
                'neg_edges': len(neg_edges),
            }
            results.append(metrics)

            # roll hidden forward using this bucket (no training)
            edge_index = tgcn_utils.edges_to_index(bucket_edges, node_to_idx)
            if edge_index.numel() > 0:
                x = _node_inputs(edge_index, len(nodes), static_features)
                hidden = _forward_model(model, x, edge_index, hidden, model_name)

    # If requested, write per-edge CSV once after all buckets are processed.
    if edge_output_path is not None and edge_rows:
        edge_output_path.parent.mkdir(parents=True, exist_ok=True)
        edge_df = pd.DataFrame(edge_rows)
        edge_df.to_csv(edge_output_path, index=False)

    return results


def summarize_bucket_metrics(metrics: List[dict]):
    aucs = [m['roc_auc'] for m in metrics]
    aps = [m['average_precision'] for m in metrics]
    return {
        'roc_auc_mean': float(np.mean(aucs)) if aucs else float('nan'),
        'roc_auc_std': float(np.std(aucs)) if aucs else float('nan'),
        'average_precision_mean': float(np.mean(aps)) if aps else float('nan'),
        'average_precision_std': float(np.std(aps)) if aps else float('nan'),
        'buckets_evaluated': len(metrics),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--edges', required=True)
    ap.add_argument('--years', default='2012,2013,2014,2015,2016,2017,2018,2019')
    ap.add_argument('--embedding-dim', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--test-ratio', type=float, default=0.3)
    ap.add_argument('--seeds', default='1,2,3')
    ap.add_argument(
        '--model',
        default='tgcn',
        choices=['tgcn', 'tgcn_pyg', 'gconvgru', 'gconvlstm', 'gcn', 'graph_transformer'],
        help=(
            "Recurrent graph model. 'tgcn_pyg' = T-GCN cell (GCNConv only, no torch_sparse). "
            "'gcn' = two GCNConv + GRUCell; 'graph_transformer' = two TransformerConv + GRUCell."
        ),
    )
    ap.add_argument(
        '--cheb-k',
        type=int,
        default=2,
        help='Chebyshev order K for GConvGRU / GConvLSTM only.',
    )
    ap.add_argument(
        '--gt-heads',
        type=int,
        default=4,
        help='Attention heads for --model graph_transformer (TransformerConv).',
    )
    ap.add_argument('--use-temporal-node-features', action='store_true')
    ap.add_argument('--use-vessel-day-features', action='store_true')
    ap.add_argument('--use-hard-negatives', action='store_true', help='Use 2-hop hard negative sampling')
    ap.add_argument('--features-root', default='data/features_by_year')
    ap.add_argument('--feature-cols', default='')
    ap.add_argument('--max-test-buckets', type=int, default=None)
    ap.add_argument('--max-train-buckets', type=int, default=None, help='Limit train buckets to reduce memory (uses first N)')
    ap.add_argument('--cv-folds', type=int, default=0, help='Rolling-window CV folds (0=disabled, single split)')
    ap.add_argument('--cv-min-train-ratio', type=float, default=0.5, help='Min train ratio for first CV fold')
    ap.add_argument('--out-report', default='artifacts/tgcn_time_multiseed.json')
    ap.add_argument('--out-csv', default='artifacts/tgcn_time_multiseed.csv')
    ap.add_argument(
        '--out-edges',
        default='',
        help='Optional CSV to write per-edge scores (src,dst,bucket,label,score)',
    )
    ap.add_argument('--log', action='store_true')
    ap.add_argument('--log-csv', default='artifacts/experiment_log.csv')
    ap.add_argument('--log-json', default='artifacts/experiment_log.json')
    args = ap.parse_args()

    years = [int(y.strip()) for y in args.years.split(',') if y.strip()]
    df = tgcn_utils.load_edges_with_time(Path(args.edges), years)
    buckets = sorted(df['time_bucket'].unique())
    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    if args.model == 'tgcn_pyg':
        model_cls = TGCNPure
    elif args.model in {'tgcn', 'gconvgru', 'gconvlstm'}:
        from torch_geometric_temporal.nn.recurrent import GConvGRU, GConvLSTM, TGCN

        model_cls = {
            'tgcn': TGCN,
            'gconvgru': GConvGRU,
            'gconvlstm': GConvLSTM,
        }[args.model]
    elif args.model == 'gcn':
        model_cls = TemporalGCNGRU
    elif args.model == 'graph_transformer':
        model_cls = TemporalGraphTransformerGRU
    else:
        raise ValueError(f'Unknown model: {args.model!r}')

    use_cv = args.cv_folds and args.cv_folds >= 2
    if use_cv:
        fold_tuples = tgcn_utils.rolling_cv_folds(buckets, args.cv_folds, args.cv_min_train_ratio)
    else:
        train_buckets, test_buckets, cutoff = tgcn_utils.time_split_buckets(buckets, args.test_ratio)
        fold_tuples = [(train_buckets, test_buckets, cutoff)]

    nodes = sorted({n for pair in tgcn_utils.dedupe_pairs(df) for n in pair})
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    per_fold = []
    for fold_idx, (train_buckets, test_buckets, cutoff) in enumerate(fold_tuples):
        if args.max_train_buckets is not None and len(train_buckets) > args.max_train_buckets:
            train_buckets = train_buckets[: args.max_train_buckets]

        train_df = df[df['time_bucket'].isin(train_buckets)]
        train_pairs = tgcn_utils.dedupe_pairs(train_df)
        train_set = _edge_set(train_pairs)

        feature_blocks = []
        if args.use_temporal_node_features:
            temporal_feats = build_temporal_node_features(train_df, cutoff)
            temporal_feats = temporal_feats.set_index('MMSI')
            temporal_cols = ['interactions_count', 'unique_partners', 'last_seen_days', 'mean_gap_days']
            temporal_matrix = temporal_feats.reindex(nodes)[temporal_cols].to_numpy()
            feature_blocks.append(temporal_matrix)

        if args.use_vessel_day_features:
            if not args.feature_cols:
                raise ValueError("--feature-cols is required when --use-vessel-day-features is set.")
            feature_cols = [c.strip() for c in args.feature_cols.split(',') if c.strip()]
            vessel_df = _load_vessel_day_features(Path(args.features_root), years, feature_cols)
            vessel_df = vessel_df.set_index('MMSI')
            vessel_matrix = vessel_df.reindex(nodes)[feature_cols].to_numpy()
            feature_blocks.append(vessel_matrix)

        static_features = None
        if feature_blocks:
            feat_matrix = np.concatenate(feature_blocks, axis=1)
            feat_matrix = _standardize_features(feat_matrix)
            static_features = torch.tensor(feat_matrix, dtype=torch.float32)

        in_channels = 1 + (0 if static_features is None else static_features.shape[1])
        train_snapshots = tgcn_utils.build_snapshot_edges(df, train_buckets)
        per_seed = []

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
                model_cls=model_cls,
                model_name=args.model,
                cheb_k=args.cheb_k,
                static_features=static_features,
                use_hard_negatives=args.use_hard_negatives,
                graph_transformer_heads=args.gt_heads,
            )

            hidden = rollout_hidden(
                model,
                nodes,
                node_to_idx,
                train_snapshots,
                train_buckets,
                args.model,
                static_features,
            )
            if hidden is None:
                raise ValueError('No training snapshots with edges found.')

            edge_output_path = Path(args.out_edges) if args.out_edges else None

            bucket_metrics = evaluate_buckets(
                model=model,
                nodes=nodes,
                node_to_idx=node_to_idx,
                df=df,
                train_set=train_set,
                test_buckets=test_buckets,
                start_hidden=hidden,
                max_buckets=args.max_test_buckets,
                seed=seed,
                model_name=args.model,
                static_features=static_features,
                edge_output_path=edge_output_path,
            )

            summary = summarize_bucket_metrics(bucket_metrics)
            per_seed.append({
                'seed': seed,
                'summary': summary,
                'buckets': bucket_metrics,
            })

        seed_aucs = [r['summary']['roc_auc_mean'] for r in per_seed]
        seed_aps = [r['summary']['average_precision_mean'] for r in per_seed]
        fold_overall = {
            'roc_auc_mean': float(np.mean(seed_aucs)) if seed_aucs else float('nan'),
            'roc_auc_std': float(np.std(seed_aucs)) if seed_aucs else float('nan'),
            'average_precision_mean': float(np.mean(seed_aps)) if seed_aps else float('nan'),
            'average_precision_std': float(np.std(seed_aps)) if seed_aps else float('nan'),
            'train_ratio': len(train_buckets) / len(buckets) if buckets else 0,
            'buckets_evaluated': int(np.mean([r['summary']['buckets_evaluated'] for r in per_seed])) if per_seed else 0,
        }
        per_fold.append({
            'fold': fold_idx,
            'cutoff_time_bucket': str(cutoff),
            'in_channels': in_channels,
            'overall': fold_overall,
            'per_seed': per_seed,
        })

    # Single train/test split: std across *seeds* (in per_fold[0].overall). Multi-fold
    # CV: std across *fold means*; seed-level detail stays under each fold.
    if len(per_fold) == 1:
        fo = per_fold[0]['overall']
        overall = {
            'roc_auc_mean': fo['roc_auc_mean'],
            'roc_auc_std': fo['roc_auc_std'],
            'average_precision_mean': fo['average_precision_mean'],
            'average_precision_std': fo['average_precision_std'],
            'seeds': seeds,
            'cv_folds': len(per_fold) if use_cv else 0,
            'buckets_evaluated': fo['buckets_evaluated'],
        }
    else:
        fold_aucs = [f['overall']['roc_auc_mean'] for f in per_fold]
        fold_aps = [f['overall']['average_precision_mean'] for f in per_fold]
        overall = {
            'roc_auc_mean': float(np.mean(fold_aucs)) if fold_aucs else float('nan'),
            'roc_auc_std': float(np.std(fold_aucs)) if fold_aucs else float('nan'),
            'average_precision_mean': float(np.mean(fold_aps)) if fold_aps else float('nan'),
            'average_precision_std': float(np.std(fold_aps)) if fold_aps else float('nan'),
            'seeds': seeds,
            'cv_folds': len(per_fold) if use_cv else 0,
            'buckets_evaluated': int(np.mean([f['overall']['buckets_evaluated'] for f in per_fold])) if per_fold else 0,
        }

    report = {
        'edges': args.edges,
        'years': years,
        'embedding_dim': args.embedding_dim,
        'epochs': args.epochs,
        'lr': args.lr,
        'cheb_k': args.cheb_k,
        'gt_heads': args.gt_heads,
        'use_temporal_node_features': args.use_temporal_node_features,
        'use_vessel_day_features': args.use_vessel_day_features,
        'use_hard_negatives': args.use_hard_negatives,
        'max_train_buckets': args.max_train_buckets,
        'max_test_buckets': args.max_test_buckets,
        'feature_cols': args.feature_cols,
        'test_ratio': args.test_ratio,
        'cv_folds': args.cv_folds if use_cv else 0,
        'cv_min_train_ratio': args.cv_min_train_ratio if use_cv else None,
        'cutoff_time_bucket': str(per_fold[0]['cutoff_time_bucket']) if per_fold and not use_cv else None,
        'in_channels': per_fold[0]['in_channels'] if per_fold else None,
        'model': args.model,
        'overall': overall,
        'per_fold': per_fold,
    }

    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_report, 'w') as f:
        json.dump(report, f, indent=2)

    pd.DataFrame([
        {
            'roc_auc_mean': overall['roc_auc_mean'],
            'roc_auc_std': overall['roc_auc_std'],
            'average_precision_mean': overall['average_precision_mean'],
            'average_precision_std': overall['average_precision_std'],
            'seeds': ';'.join(map(str, seeds)),
            'cv_folds': overall['cv_folds'],
            'buckets_evaluated': overall['buckets_evaluated'],
        }
    ]).to_csv(args.out_csv, index=False)

    print(json.dumps(overall, indent=2))
    print(f"Report written to {args.out_report}")
    print(f"CSV written to {args.out_csv}")

    if args.log:
        features_label = f'{args.model}_time'
        if args.use_temporal_node_features:
            features_label = f'{features_label}+temporal_node'
        if args.use_vessel_day_features:
            features_label = f'{features_label}+vessel_day'
        row = {
            'timestamp': datetime.now(UTC).isoformat().replace('+00:00', 'Z'),
            'model': f'{args.model}_time_multiseed',
            'edges': args.edges,
            'test_ratio': args.test_ratio,
            'seed': ';'.join(map(str, seeds)),
            'embedding_dim': args.embedding_dim,
            'use_features': args.use_temporal_node_features or args.use_vessel_day_features,
            'features_years': features_label,
            'roc_auc': overall['roc_auc_mean'],
            'roc_auc_std': overall['roc_auc_std'],
            'average_precision': overall['average_precision_mean'],
            'average_precision_std': overall['average_precision_std'],
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
