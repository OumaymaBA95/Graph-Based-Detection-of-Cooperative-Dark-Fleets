#!/usr/bin/env python3
"""
Create a compact baseline summary table and plots from experiment logs.
"""
import argparse
from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt


def _slug(text: str) -> str:
    text = text.replace('.parquet', '')
    text = re.sub(r'[^a-zA-Z0-9_\-]+', '_', text)
    return text.strip('_')


def plot_metrics(row, out_dir: Path):
    heuristics = ['common_neighbors', 'jaccard', 'adamic_adar']
    aucs = [row[f'{h}_roc_auc_mean'] for h in heuristics]
    aps = [row[f'{h}_average_precision_mean'] for h in heuristics]

    label = _slug(Path(row['edges']).stem)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(heuristics, aucs, color='#3b82f6')
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel('ROC AUC (mean)')
    ax.set_title(f'Baseline ROC AUC: {label}')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f'baseline_{label}_auc.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(heuristics, aps, color='#10b981')
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel('Average Precision (mean)')
    ax.set_title(f'Baseline AP: {label}')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f'baseline_{label}_ap.png', dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log-csv', default='artifacts/experiment_log.csv')
    ap.add_argument('--out-dir', default='artifacts/plots')
    ap.add_argument('--out-summary', default='artifacts/baseline_summary.csv')
    ap.add_argument('--out-md', default='artifacts/baseline_summary.md')
    ap.add_argument('--limit', type=int, default=2, help='Number of most recent runs to include')
    args = ap.parse_args()

    log_path = Path(args.log_csv)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    df = pd.read_csv(log_path)
    df = df.sort_values('timestamp', ascending=False)
    df = df.head(args.limit)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        plot_metrics(row, out_dir)

    df.to_csv(args.out_summary, index=False)

    md_lines = [
        '# Baseline Summary',
        '',
        'Recent runs:',
        ''
    ]
    headers = list(df.columns)
    md_lines.append('|' + '|'.join(headers) + '|')
    md_lines.append('|' + '|'.join(['---'] * len(headers)) + '|')
    for _, row in df.iterrows():
        md_lines.append('|' + '|'.join(str(row[h]) for h in headers) + '|')
    md_lines.append('')

    Path(args.out_md).write_text('\n'.join(md_lines))
    print(f"Summary CSV: {args.out_summary}")
    print(f"Summary MD: {args.out_md}")
    print(f"Plots written to: {out_dir}")


if __name__ == '__main__':
    main()
