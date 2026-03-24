#!/usr/bin/env python3
"""Summarize SST extraction outputs under a root directory.

Scans `root/<year>/chunk_*.parquet` files, aggregates:
- total rows
- rows with sst present vs missing
- sst min, mean (streamed), approximate median and 99th (via reservoir sampling)

Use with the project venv python.
"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import random
import math
from typing import Optional


def reservoir_sample_iter(it, k, rng=random.Random()):
    """Yield a reservoir sample (list) of size up to k from iterator of values."""
    sample = []
    for i, v in enumerate(it, start=1):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        if len(sample) < k:
            sample.append(v)
        else:
            j = rng.randrange(i)
            if j < k:
                sample[j] = v
    return sample


def process_year(year_dir, sample_size=1000000, max_files_per_year: int = 0):
    files = sorted(Path(year_dir).glob('chunk_*.parquet'))
    if max_files_per_year and max_files_per_year > 0:
        files = files[:max_files_per_year]
    if not files:
        return None
    total = 0
    missing = 0
    present = 0
    sst_min = None
    sst_max = None
    sst_sum = 0.0
    sst_count = 0
    sample_k = sample_size
    sample_list = []
    rng = random.Random(42)
    # We'll maintain a reservoir incrementally by reading each file and updating
    for fp in files:
        try:
            df = pd.read_parquet(fp, columns=['sst','sst_missing'])
        except Exception as e:
            try:
                # try reading with pyarrow via pandas default
                df = pd.read_parquet(fp)
                if 'sst' not in df.columns:
                    continue
                keep_cols = [c for c in ['sst','sst_missing'] if c in df.columns]
                df = df[keep_cols]
            except Exception as e2:
                print(f"    skip file {fp.name}: {e2}")
                continue
        vals = df['sst'].values
        miss_flag = None
        if 'sst_missing' in df.columns:
            miss_flag = df['sst_missing'].values
        else:
            # infer missing if sst is null/NaN
            miss_flag = np.isnan(vals)

        n = len(df)
        total += n
        # count missing by flag OR sst null
        is_missing = np.array([bool(x) for x in miss_flag])
        missing += int(is_missing.sum())
        present_count = int((~is_missing).sum())
        present += present_count
        # stats for present values
        present_vals = vals[~is_missing]
        if present_vals.size:
            # update min/max/sum/count
            cur_min = float(np.nanmin(present_vals))
            cur_max = float(np.nanmax(present_vals))
            if sst_min is None or cur_min < sst_min:
                sst_min = cur_min
            if sst_max is None or cur_max > sst_max:
                sst_max = cur_max
            sst_sum += float(np.nansum(present_vals))
            sst_count += int(present_vals.size)
            # reservoir sample update
            # create iterator
            for v in present_vals:
                if len(sample_list) < sample_k:
                    sample_list.append(float(v))
                else:
                    i = rng.randrange(total)
                    if i < sample_k:
                        sample_list[i] = float(v)
    # compute mean & quantiles
    mean = None
    median = None
    p99 = None
    if sst_count:
        mean = sst_sum / sst_count
    if sample_list:
        arr = np.array(sample_list)
        median = float(np.nanpercentile(arr, 50))
        p99 = float(np.nanpercentile(arr, 99))
    return {
        'files': len(files),
        'total': total,
        'present': present,
        'missing': missing,
        'sst_min': sst_min,
        'sst_mean': mean,
        'sst_median_approx': median,
        'sst_99_approx': p99,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='data/sst_by_year')
    p.add_argument('--sample-size', type=int, default=200000)
    p.add_argument('--years', help='Comma-separated years to include (e.g., 2018,2019)', default=None)
    p.add_argument('--max-files-per-year', type=int, default=0, help='Optional cap on number of chunk files per year')
    args = p.parse_args()
    root = Path(args.root)
    if not root.exists():
        print('Root path not found:', root)
        return
    years = sorted([p for p in root.iterdir() if p.is_dir()])
    if args.years:
        allow = set([y.strip() for y in args.years.split(',') if y.strip()])
        years = [p for p in years if p.name in allow]
    overall = {
        'files': 0,
        'total': 0,
        'present': 0,
        'missing': 0,
    }
    print('Scanning', len(years), 'year directories under', root)
    for y in years:
        res = process_year(y, sample_size=args.sample_size, max_files_per_year=args.max_files_per_year)
        if res is None:
            print(y.name, 'no parquet chunk files found')
            continue
        overall['files'] += res['files']
        overall['total'] += res['total']
        overall['present'] += res['present']
        overall['missing'] += res['missing']
        print('\nYear', y.name)
        print('  chunk files:', res['files'])
        print('  total rows:', f"{res['total']:,}")
        print('  sst present:', f"{res['present']:,}")
        print('  sst missing:', f"{res['missing']:,}")
        if res['sst_min'] is not None:
            print('  sst min:', f"{res['sst_min']:.3f}")
            print('  sst mean:', f"{res['sst_mean']:.3f}")
            print('  sst median (approx):', f"{res['sst_median_approx']:.3f}")
            print('  sst 99th (approx):', f"{res['sst_99_approx']:.3f}")
    print('\nOverall')
    print('  total chunk files:', overall['files'])
    print('  total rows:', f"{overall['total']:,}")
    print('  sst present:', f"{overall['present']:,}")
    print('  sst missing:', f"{overall['missing']:,}")

if __name__ == '__main__':
    main()
