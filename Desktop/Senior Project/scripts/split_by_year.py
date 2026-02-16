#!/usr/bin/env python3
"""Split a large combined CSV into per-year CSV files (streaming, memory-safe).

Writes: data/by_year/combined_<YEAR>.csv
"""
from pathlib import Path
import argparse
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--out-dir', default='data/by_year')
    p.add_argument('--chunksize', type=int, default=100000)
    args = p.parse_args()

    inp = Path(args.input)
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # sniff time column from first few rows
    preview = pd.read_csv(inp, nrows=20)
    time_col = None
    for c in ('timestamp', 'time', 'datetime', 'date'):
        if c in preview.columns:
            time_col = c
            break
    if time_col is None:
        time_col = preview.columns[0]

    print('Using time column:', time_col)

    reader = pd.read_csv(inp, parse_dates=[time_col], chunksize=args.chunksize)
    counts = {}
    for i, chunk in enumerate(reader):
        chunk[time_col] = pd.to_datetime(chunk[time_col], errors='coerce')
        chunk = chunk.dropna(subset=[time_col])
        chunk['year'] = chunk[time_col].dt.year
        for year, g in chunk.groupby('year'):
            outp = outdir / f'combined_{int(year)}.csv'
            header = not outp.exists()
            g.drop(columns=['year']).to_csv(outp, mode='a', index=False, header=header)
            counts[year] = counts.get(year, 0) + len(g)
        if (i + 1) % 10 == 0:
            print('Processed chunks:', i + 1)
    print('Done. Rows per year:')
    for y in sorted(counts):
        print(y, counts[y])


if __name__ == '__main__':
    main()
