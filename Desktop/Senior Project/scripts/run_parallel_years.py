#!/usr/bin/env python3
"""Controller to run per-year vectorized extraction in parallel.

It finds CSVs under data/by_year/combined_<YEAR>.csv and corresponding zarrs
under data/glorys_by_year/glorys_<YEAR>.zarr, then launches up to N workers.
"""
from pathlib import Path
import argparse
import subprocess
import multiprocessing
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--by-year-dir', default='data/by_year')
    p.add_argument('--zarr-dir', default='data/glorys_by_year')
    p.add_argument('--out-root', default='data/sst_by_year')
    p.add_argument('--workers', type=int, default=multiprocessing.cpu_count())
    p.add_argument('--chunk-size', type=int, default=50000)
    p.add_argument('--time-tolerance-days', type=int, default=31)
    p.add_argument('--years', type=str, default=None,
                   help='Optional comma-separated list of years to process (e.g. 2015,2017,2018)')
    args = p.parse_args()

    bydir = Path(args.by_year_dir)
    zarrdir = Path(args.zarr_dir)
    csvs = sorted(bydir.glob('combined_*.csv'))
    # filter by years if requested
    if args.years:
        want = set([y.strip() for y in args.years.split(',') if y.strip()])
        csvs = [p for p in csvs if p.stem.split('_')[-1] in want]
    tasks = []
    for c in csvs:
        year = c.stem.split('_')[-1]
        z = zarrdir / f'glorys_{year}.zarr'
        if not z.exists():
            print('Skipping', year, 'no zarr at', z)
            continue
        outdir = Path(args.out_root) / str(year)
        outdir.mkdir(parents=True, exist_ok=True)
        tasks.append((str(c), str(z), str(outdir)))

    if not tasks:
        print('No tasks found')
        return

    print('Found tasks for years:', [t[0] for t in tasks])
    print('Launching up to', args.workers, 'workers')

    procs = []
    i = 0
    while i < len(tasks) or procs:
        # spawn while we have capacity
        while len(procs) < args.workers and i < len(tasks):
            csvp, zarrp, outp = tasks[i]
            cmd = ["/Users/momoba/Desktop/Senior Project/venv/bin/python", "scripts/extract_sst_vectorized.py", "--input", csvp, "--zarr", zarrp, "--out-dir", outp, "--chunk-size", str(args.chunk_size), "--time-tolerance-days", str(args.time_tolerance_days)]
            print('Starting:', ' '.join(cmd))
            p = subprocess.Popen(cmd)
            procs.append((p, csvp))
            i += 1
            time.sleep(0.5)
        # reap finished
        new_procs = []
        for p, c in procs:
            ret = p.poll()
            if ret is None:
                new_procs.append((p,c))
            else:
                print('Task finished for', c, 'exit', ret)
        procs = new_procs
        time.sleep(5)

    print('All tasks complete')

if __name__ == '__main__':
    main()
