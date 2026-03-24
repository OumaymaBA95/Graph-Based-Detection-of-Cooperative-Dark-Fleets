#!/usr/bin/env python3
"""
Run the linear GAE baseline across multiple seeds and summarize metrics.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_linear_gae_baseline as lgae


def run_once(args, seed):
    class Dummy:
        pass
    opts = Dummy()
    opts.edges = args.edges
    opts.embedding_dim = args.embedding_dim
    opts.test_ratio = args.test_ratio
    opts.seed = seed
    opts.out_report = None
    opts.features_root = args.features_root
    opts.features_years = args.features_years
    opts.use_features = args.use_features
    df = lgae.load_edges(Path(args.edges))
    edge_list = list(map(tuple, df[['src', 'dst']].to_numpy()))
    train_edges, test_pos = lgae.train_test_split_edges(edge_list, args.test_ratio, seed)

    nodes = sorted(set(df['src']).union(set(df['dst'])))
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    rows = [node_to_idx[u] for u, v in train_edges]
    cols = [node_to_idx[v] for u, v in train_edges]
    data = np.ones(len(rows))
    n = len(nodes)
    adj = lgae.coo_matrix((data, (rows, cols)), shape=(n, n))
    adj = adj + adj.T

    k = min(args.embedding_dim, max(2, n - 1))
    u, s, vt = lgae.svds(adj, k=k)
    order = np.argsort(-s)
    u = u[:, order]
    s = s[order]
    emb = u * s

    if args.use_features and args.features_years:
        years = [y.strip() for y in args.features_years.split(',') if y.strip()]
        feats = lgae.load_features(Path(args.features_root), years)
        if not feats.empty:
            feat_cols = [c for c in feats.columns if c != 'MMSI']
            feats = feats.set_index('MMSI')
            feat_mat = np.zeros((n, len(feat_cols)), dtype=float)
            for i, m in enumerate(nodes):
                if m in feats.index:
                    feat_mat[i] = feats.loc[m, feat_cols].to_numpy(dtype=float)
            mean = np.nanmean(feat_mat, axis=0)
            std = np.nanstd(feat_mat, axis=0)
            std[std == 0] = 1.0
            feat_mat = (feat_mat - mean) / std
            feat_mat = np.nan_to_num(feat_mat)
            emb = np.concatenate([emb, feat_mat], axis=1)

    existing = set(edge_list)
    test_neg = lgae.negative_sampling(nodes, existing, k=len(test_pos), seed=seed)

    def score(u, v):
        iu, iv = node_to_idx[u], node_to_idx[v]
        return float(np.dot(emb[iu], emb[iv]))

    pairs = test_pos + test_neg
    labels = np.array([1] * len(test_pos) + [0] * len(test_neg))
    scores = np.array([score(u, v) for u, v in pairs])

    return {
        'seed': seed,
        'edges_total': len(edge_list),
        'train_edges': len(train_edges),
        'test_pos': len(test_pos),
        'test_neg': len(test_neg),
        'embedding_dim': int(emb.shape[1]),
        'metrics': {
            'roc_auc': lgae.roc_auc(labels, scores),
            'average_precision': lgae.average_precision(labels, scores),
        }
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--edges', required=True)
    ap.add_argument('--embedding-dim', type=int, default=32)
    ap.add_argument('--test-ratio', type=float, default=0.3)
    ap.add_argument('--seeds', default='1,2,3,4,5')
    ap.add_argument('--features-root', default='data/features_by_year')
    ap.add_argument('--features-years', default='')
    ap.add_argument('--use-features', action='store_true')
    ap.add_argument('--out-report', default='artifacts/linear_gae_multiseed.json')
    ap.add_argument('--out-csv', default='artifacts/linear_gae_multiseed.csv')
    ap.add_argument('--log', action='store_true')
    ap.add_argument('--log-csv', default='artifacts/experiment_log.csv')
    ap.add_argument('--log-json', default='artifacts/experiment_log.json')
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    results = [run_once(args, seed) for seed in seeds]

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
        'embedding_dim': args.embedding_dim,
        'test_ratio': args.test_ratio,
        'seeds': seeds,
        'use_features': args.use_features and bool(args.features_years),
        'features_years': args.features_years,
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
            'timestamp': lgae.datetime.now(lgae.UTC).isoformat().replace('+00:00', 'Z'),
            'model': 'linear_gae',
            'edges': args.edges,
            'test_ratio': args.test_ratio,
            'seed': ';'.join(map(str, seeds)),
            'embedding_dim': args.embedding_dim,
            'use_features': report['use_features'],
            'features_years': args.features_years,
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
