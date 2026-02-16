#!/usr/bin/env python3
"""Rebuild a combined fleet CSV from raw CSV files and CSVs inside zip archives.

Writes a memory-safe combined CSV `data/combined_fleet_daily_full.csv` with the
union of headers across all sources.

It will look for directories and zip files under `data/` matching
`fleet-daily-csvs-100-v3-*` and include any .csv files found there.

This is safe to re-run; it streams and does not load entire files into memory.
"""
from pathlib import Path
import argparse
import csv
import zipfile
import io


def iter_csv_files(root_dir):
    root = Path(root_dir)
    # directories with CSVs
    for d in sorted(root.glob('fleet-daily-csvs-100-v3-*')):
        if d.is_dir():
            for f in sorted(d.glob('*.csv')):
                yield ('file', str(f), None)
    # zip archives
    for z in sorted(root.glob('fleet-daily-csvs-100-v3-*.zip')):
        try:
            with zipfile.ZipFile(z, 'r') as zf:
                for name in sorted(zf.namelist()):
                    if name.lower().endswith('.csv'):
                        yield ('zip', str(z), name)
        except Exception:
            continue
    # also include any loose CSVs under data/
    for f in sorted(root.glob('*.csv')):
        yield ('file', str(f), None)


def read_header_from_source(src):
    typ, path, inner = src
    if typ == 'file':
        with open(path, 'r', encoding='utf-8', errors='replace') as fh:
            r = csv.reader(fh)
            try:
                return next(r)
            except StopIteration:
                return []
    else:
        with zipfile.ZipFile(path, 'r') as zf:
            with zf.open(inner) as fh:
                txt = io.TextIOWrapper(fh, encoding='utf-8', errors='replace')
                r = csv.reader(txt)
                try:
                    return next(r)
                except StopIteration:
                    return []


def iter_rows_from_source(src):
    typ, path, inner = src
    if typ == 'file':
        with open(path, 'r', encoding='utf-8', errors='replace') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                yield row
    else:
        with zipfile.ZipFile(path, 'r') as zf:
            with zf.open(inner) as fh:
                txt = io.TextIOWrapper(fh, encoding='utf-8', errors='replace')
                reader = csv.DictReader(txt)
                for row in reader:
                    yield row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data')
    p.add_argument('--out', default='data/combined_fleet_daily_full.csv')
    args = p.parse_args()

    sources = list(iter_csv_files(args.data_dir))
    if not sources:
        print('No CSV sources found under', args.data_dir)
        return

    print('Found', len(sources), 'CSV sources (files and zip entries)')
    # build union of headers
    headers = []
    header_set = set()
    for src in sources:
        try:
            h = read_header_from_source(src)
        except Exception:
            h = []
        for col in h:
            if col not in header_set:
                header_set.add(col)
                headers.append(col)
    if not headers:
        print('No headers found; aborting')
        return

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    print('Writing combined CSV to', outp)
    with open(outp, 'w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        total = 0
        for i, src in enumerate(sources):
            typ, path, inner = src
            print(f'Processing source {i+1}/{len(sources)}:', typ, path, (inner or ''))
            try:
                for row in iter_rows_from_source(src):
                    # ensure output mapping contains all headers
                    out = {k: row.get(k, '') for k in headers}
                    writer.writerow(out)
                    total += 1
                    if total % 1000000 == 0:
                        print('  written', total, 'rows')
            except Exception as e:
                print('  failed to process source', src, 'error:', e)
    print('Done. total rows written:', total)


if __name__ == '__main__':
    main()
