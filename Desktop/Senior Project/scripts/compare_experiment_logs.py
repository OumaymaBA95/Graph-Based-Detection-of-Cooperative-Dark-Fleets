#!/usr/bin/env python3
"""
Compare experiment log rows and produce a simple summary table.
"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log-csv', default='artifacts/experiment_log.csv')
    ap.add_argument('--out-csv', default='artifacts/experiment_log_summary.csv')
    ap.add_argument('--filter-edges', default='', help='Substring filter for edges path (optional)')
    args = ap.parse_args()

    log_path = Path(args.log_csv)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    df = pd.read_csv(log_path)
    if args.filter_edges:
        df = df[df['edges'].str.contains(args.filter_edges, na=False)]

    if df.empty:
        raise ValueError("No rows matched the filters.")

    df = df.sort_values('timestamp', ascending=False)
    df.to_csv(args.out_csv, index=False)
    print(df.head(10).to_string(index=False))
    print(f"Summary written to {args.out_csv}")


if __name__ == '__main__':
    main()
