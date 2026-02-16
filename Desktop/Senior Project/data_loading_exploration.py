"""
Data loading and exploration helper.

This script does two things:
- Keeps the previous sampling-based exploration for vessel and fleet daily CSVs.
- Provides a memory-safe combine mode to merge all fleet daily CSVs (from zip archives
  and folders under `data/`) into a single CSV file. Run with `--combine` to enable.

Example:
    python data_loading_exploration.py --combine --years 2012 2019 --output data/combined_fleet_daily.csv --overwrite

The combine routine reads headers first to compute a union of columns, then streams
each CSV in chunks, reindexing to the union and appending to the output file. This
minimizes peak memory usage and tolerates files with different column sets.
"""

import os
import zipfile
import argparse
from typing import List, Tuple

import pandas as pd
import fsspec
import xarray as xr


def sample_vessel_data(vessel_data_dir: str, years: List[str]):
    vessel_files = [f for f in os.listdir(vessel_data_dir) if f.endswith('.csv')]
    vessel_files_2012_2019 = [f for f in vessel_files if any(y in f for y in years)]
    print(f"Processing {len(vessel_files_2012_2019)} vessel files from {years[0]}-{years[-1]} in chunks.")

    vessel_sample_rows = []
    for file in vessel_files_2012_2019:
        file_path = os.path.join(vessel_data_dir, file)
        # Read only first 1000 rows as a sample
        df = pd.read_csv(file_path, nrows=1000)
        vessel_sample_rows.append(df)
        print(f"Read sample from {file}: {df.shape}")
    if vessel_sample_rows:
        vessel_df_sample = pd.concat(vessel_sample_rows, ignore_index=True)
        print(f"Combined vessel sample shape: {vessel_df_sample.shape}")
        return vessel_df_sample
    else:
        print("No vessel samples found.")
        return pd.DataFrame()


def sample_fleet_data(fleet_data_dir: str, years: List[str]):
    fleet_files = [f for f in os.listdir(fleet_data_dir) if f.startswith('fleet-daily-csvs-100-v3-') and f.endswith('.zip')]
    fleet_files_2012_2019 = [f for f in fleet_files if any(y in f for y in years)]
    print(f"Processing {len(fleet_files_2012_2019)} fleet daily zip files from {years[0]}-{years[-1]} in chunks.")

    fleet_sample_rows = []
    for file in fleet_files_2012_2019:
        file_path = os.path.join(fleet_data_dir, file)
        with zipfile.ZipFile(file_path) as z:
            for csv_name in [n for n in z.namelist() if n.lower().endswith('.csv')]:
                with z.open(csv_name) as csvfile:
                    # Read only first 1000 rows as a sample
                    df = pd.read_csv(csvfile, nrows=1000)
                    fleet_sample_rows.append(df)
                    print(f"Read sample from {file}/{csv_name}: {df.shape}")
    if fleet_sample_rows:
        fleet_df_sample = pd.concat(fleet_sample_rows, ignore_index=True)
        print(f"Combined fleet daily sample shape: {fleet_df_sample.shape}")
        return fleet_df_sample
    else:
        print("No fleet samples found.")
        return pd.DataFrame()


def discover_sources(data_dir: str, years: List[str]) -> List[Tuple[str, str, str]]:
    """Discover CSV sources under data_dir for the given years.

    Returns a list of tuples: (source_type, path, member)
    where source_type is 'file' or 'zip'. For 'file' member is ''. For 'zip' member
    will be set when expanding members later.
    """
    sources = []
    # zip files
    for name in os.listdir(data_dir):
        if name.startswith('fleet-daily-csvs-100-v3-') and name.endswith('.zip') and any(y in name for y in years):
            sources.append(('zip', os.path.join(data_dir, name), ''))

    # directories with csvs
    for name in os.listdir(data_dir):
        if name.startswith('fleet-daily-csvs-100-v3-') and os.path.isdir(os.path.join(data_dir, name)) and any(y in name for y in years):
            folder = os.path.join(data_dir, name)
            for root, _, files in os.walk(folder):
                for f in files:
                    if f.lower().endswith('.csv'):
                        sources.append(('file', os.path.join(root, f), ''))
    return sources


def get_zip_members(zip_path: str) -> List[str]:
    with zipfile.ZipFile(zip_path) as z:
        return [n for n in z.namelist() if n.lower().endswith('.csv')]


def expand_zip_sources(sources: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    expanded = []
    for src_type, path, _ in sources:
        if src_type == 'file':
            expanded.append((src_type, path, ''))
        else:
            members = get_zip_members(path)
            for m in members:
                expanded.append(('zip', path, m))
    return expanded


def collect_all_columns(sources: List[Tuple[str, str, str]]) -> List[str]:
    all_cols = []
    for src_type, path, member in sources:
        if src_type == 'file':
            try:
                cols = pd.read_csv(path, nrows=0).columns.tolist()
            except Exception as e:
                print(f"Warning: failed to read header from file {path}: {e}")
                cols = []
            for c in cols:
                if c not in all_cols:
                    all_cols.append(c)
        else:
            try:
                with zipfile.ZipFile(path) as z:
                    # if member provided, read that; otherwise iterate all csv members
                    members = [member] if member else [n for n in z.namelist() if n.lower().endswith('.csv')]
                    for m in members:
                        try:
                            with z.open(m) as f:
                                cols = pd.read_csv(f, nrows=0).columns.tolist()
                        except Exception as e:
                            print(f"Warning: failed to read header from {path}:{m}: {e}")
                            cols = []
                        for c in cols:
                            if c not in all_cols:
                                all_cols.append(c)
            except Exception as e:
                print(f"Warning: failed to open zip {path}: {e}")
    return all_cols


def stream_and_write(sources: List[Tuple[str, str, str]], output_path: str, all_columns: List[str], chunksize: int):
    first_write = True
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for src_type, path, member in sources:
        if src_type == 'file':
            print(f"Processing file {path}")
            try:
                for chunk in pd.read_csv(path, chunksize=chunksize):
                    chunk = chunk.reindex(columns=all_columns)
                    chunk.to_csv(output_path, mode='a', index=False, header=first_write)
                    first_write = False
            except Exception as e:
                print(f"Warning: failed to process {path}: {e}")
        else:
            print(f"Processing zip {path} member {member}")
            try:
                with zipfile.ZipFile(path) as z:
                    members = [member] if member else get_zip_members(path)
                    for m in members:
                        print(f"  member: {m}")
                        try:
                            with z.open(m) as f:
                                for chunk in pd.read_csv(f, chunksize=chunksize):
                                    chunk = chunk.reindex(columns=all_columns)
                                    chunk.to_csv(output_path, mode='a', index=False, header=first_write)
                                    first_write = False
                        except Exception as e:
                            print(f"Warning: failed to process member {m} in {path}: {e}")
            except Exception as e:
                print(f"Warning: failed to open zip {path}: {e}")


def combine_fleet_daily(data_dir: str, years: List[str], output: str, chunksize: int, overwrite: bool, dry_run: bool):
    base_sources = discover_sources(data_dir, years)
    if not base_sources:
        print(f"No fleet sources found under {data_dir} for years {years}")
        return

    sources = expand_zip_sources(base_sources)
    print(f"Discovered {len(sources)} CSV sources to combine.")

    if os.path.exists(output) and not overwrite and not dry_run:
        print(f"Output file {output} already exists. Use --overwrite to replace it.")
        return

    print("Collecting union of all columns (headers only pass)...")
    all_columns = collect_all_columns(base_sources)
    print(f"Total columns discovered: {len(all_columns)}")

    if dry_run:
        print("Dry run: would combine the following sources:")
        for s in sources[:200]:
            print(s)
        print(f"Would write to: {output} with columns: {all_columns}")
        return

    if os.path.exists(output) and overwrite:
        os.remove(output)

    print("Starting streaming and write (this may take a while)...")
    stream_and_write(sources, output, all_columns, chunksize)
    print(f"Finished. Combined CSV saved to {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data', help='Base data directory')
    parser.add_argument('--years', nargs='+', default=[str(y) for y in range(2012, 2020)], help='Years to include (tokens matched in names)')
    parser.add_argument('--combine', action='store_true', help='Combine fleet daily CSVs into one CSV')
    parser.add_argument('--output', default='data/combined_fleet_daily.csv', help='Output CSV path')
    parser.add_argument('--chunksize', type=int, default=100000, help='Rows per chunk when streaming')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite output if exists')
    parser.add_argument('--dry-run', action='store_true', help='Only show what would be done (no writes)')
    args = parser.parse_args()

    # Keep original sampling behaviour
    vessel_dir = os.path.join(args.data_dir, 'MMSI daily vessels ')
    print('\n=== Vessel sampling ===')
    vessel_sample = sample_vessel_data(vessel_dir, args.years)

    print('\n=== Fleet sampling ===')
    fleet_sample = sample_fleet_data(args.data_dir, args.years)

    if args.combine:
        combine_fleet_daily(args.data_dir, args.years, args.output, args.chunksize, args.overwrite, args.dry_run)


if __name__ == '__main__':
    main()


url = "https://minio.dive.edito.eu/project-glonet/public/glorys14_refull_2024/20120327.zarr"
mapper = fsspec.get_mapper(url, anon=True)
print('keys:', list(mapper.keys())[:20])
xr.open_zarr(mapper)   # will fail if no keys or access blocked
