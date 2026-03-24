#!/usr/bin/env python3
"""
Run link prediction baseline across multiple seeds and summarize metrics.

Outputs:
- JSON report with per-seed results and mean/std summary.
- CSV summary of mean/std per heuristic.
"""
import argparse
import json
import os
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import run_link_prediction_baseline as lp


def dedup_edges(df: pd.DataFrame) -> List[tuple]:
    df = df[df['src'] != df['dst']][['src', 'dst']].copy()
    df[['a', 'b']] = pd.DataFrame(np.sort(df[['src', 'dst']].to_numpy(), axis=1))
    df = df.drop_duplicates(subset=['a', 'b'])[['a', 'b']]
    return list(map(tuple, df.to_numpy()))


def run_once(edge_list: List[tuple], test_ratio: float, seed: int) -> Dict:
    train, test_pos = lp.train_test_split_edges(edge_list, test_ratio=test_ratio, seed=seed)
    adj = lp.build_adj(train)
    nodes = list(adj.keys())
    test_neg = lp.negative_sampling(nodes, set(edge_list), k=len(test_pos), seed=seed)
    metrics = lp.evaluate(adj, test_pos, test_neg)
    return {
        'seed': seed,
        'edges_total': len(edge_list),
        'train_edges': len(train),
        'test_pos': len(test_pos),
        'test_neg': len(test_neg),
        'metrics': metrics,
    }


def summarize(results: List[Dict]) -> Dict:
    summary = {}
    metrics = results[0]['metrics'].keys() if results else []
    for name in metrics:
        aucs = [r['metrics'][name]['roc_auc'] for r in results]
        aps = [r['metrics'][name]['average_precision'] for r in results]
        summary[name] = {
            'roc_auc_mean': float(np.mean(aucs)),
            'roc_auc_std': float(np.std(aucs)),
            'average_precision_mean': float(np.mean(aps)),
            'average_precision_std': float(np.std(aps)),
        }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--edges', default='artifacts/edges_baseline.parquet')
    ap.add_argument('--synthetic', action='store_true', help='Ignore edges file and use synthetic graph')
    ap.add_argument('--test-ratio', type=float, default=0.2)
    ap.add_argument('--seeds', default='1,2,3,4,5', help='Comma-separated seeds')
    ap.add_argument('--out-report', default='artifacts/baseline_linkpred_multiseed.json')
    ap.add_argument('--out-csv', default='artifacts/baseline_linkpred_multiseed.csv')
    ap.add_argument('--log-csv', default='artifacts/experiment_log.csv')
    ap.add_argument('--log-json', default='artifacts/experiment_log.json')
    ap.add_argument('--log', action='store_true', help='Append summary to experiment log files')
    args = ap.parse_args()

    os.makedirs(Path(args.out_report).parent, exist_ok=True)

    if args.synthetic:
        df = lp.make_synthetic(n_nodes=60, p=0.06)
    else:
        if not Path(args.edges).exists():
            raise FileNotFoundError(f"Edge file not found: {args.edges}")
        df = lp.load_edges(Path(args.edges))

    edge_list = dedup_edges(df)
    if len(edge_list) < 4:
        raise ValueError("Not enough edges to evaluate.")

    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    results = [run_once(edge_list, args.test_ratio, seed) for seed in seeds]
    summary = summarize(results)

    report = {
        'edges_total': len(edge_list),
        'test_ratio': args.test_ratio,
        'seeds': seeds,
        'per_seed': results,
        'summary': summary,
    }

    with open(args.out_report, 'w') as f:
        json.dump(report, f, indent=2)

    rows = []
    for name, metrics in summary.items():
        rows.append({'heuristic': name, **metrics})
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)

    print(json.dumps(report['summary'], indent=2))
    print(f"Report written to {args.out_report}")
    print(f"CSV written to {args.out_csv}")

    if args.log:
        timestamp = datetime.now(UTC).isoformat().replace('+00:00', 'Z')
        summary_row = {
            'timestamp': timestamp,
            'edges': str(args.edges),
            'test_ratio': args.test_ratio,
            'seeds': ';'.join(map(str, seeds)),
        }
        for name, metrics in summary.items():
            summary_row[f'{name}_roc_auc_mean'] = metrics['roc_auc_mean']
            summary_row[f'{name}_roc_auc_std'] = metrics['roc_auc_std']
            summary_row[f'{name}_average_precision_mean'] = metrics['average_precision_mean']
            summary_row[f'{name}_average_precision_std'] = metrics['average_precision_std']

        log_df = pd.DataFrame([summary_row])
        if Path(args.log_csv).exists():
            log_df.to_csv(args.log_csv, mode='a', index=False, header=False)
        else:
            log_df.to_csv(args.log_csv, index=False)

        log_json_path = Path(args.log_json)
        if log_json_path.exists():
            with open(log_json_path, 'r') as f:
                payload = json.load(f)
            if not isinstance(payload, list):
                payload = [payload]
        else:
            payload = []
        payload.append(summary_row)
        with open(log_json_path, 'w') as f:
            json.dump(payload, f, indent=2)

        print(f"Experiment log updated: {args.log_csv}")
        print(f"Experiment log updated: {args.log_json}")


if __name__ == '__main__':
    main()
