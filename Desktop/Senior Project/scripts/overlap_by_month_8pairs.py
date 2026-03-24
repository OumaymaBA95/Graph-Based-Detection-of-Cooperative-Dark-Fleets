#!/usr/bin/env python3
"""
For the 8 validated vessel pairs (25 km ±1 day), compute overlap per month:
days within 25 km and (optionally) total overlap days, by (src, dst, year-month).

Usage:
  python3 scripts/overlap_by_month_8pairs.py --pairs artifacts/close_pairs_fullcoverage_25km_w1.csv \\
    --daily-root "data/MMSI daily vessels " --distance-km 25 --day-window 1 --out-csv artifacts/eight_pairs_overlap_by_month.csv

  More months: use --all-files to load every daily file, and --full-months to include every month
  in range per pair (zeros where no overlap) for a full-timeline CSV and heatmap.
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from vessel_pair_labels import format_pair_label, load_enrichment_first_row_per_pair


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return r * c


def choose_files(files: list, max_files: int, seed: int) -> list:
    if max_files <= 0 or max_files >= len(files):
        return files
    step = (len(files) - 1) / max_files
    idx = [round(i * step) for i in range(max_files)]
    return [files[i] for i in idx]


def load_daily_data(files: list, mmsi_set: set) -> pd.DataFrame:
    frames = []
    for fpath in files:
        chunk = pd.read_csv(fpath)
        chunk = chunk[chunk['mmsi'].isin(mmsi_set)]
        if chunk.empty:
            continue
        chunk['day'] = pd.to_datetime(chunk['date'])
        frames.append(chunk[['mmsi', 'day', 'cell_ll_lat', 'cell_ll_lon']])
    if not frames:
        return pd.DataFrame(columns=['mmsi', 'day', 'cell_ll_lat', 'cell_ll_lon'])
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description='Overlap by month for 8 validated pairs')
    ap.add_argument('--pairs', default='artifacts/close_pairs_fullcoverage_25km_w1.csv',
                    help='CSV with src,dst and optionally days_within_km (use rows with days_within_km>0)')
    ap.add_argument('--daily-root', default='data/MMSI daily vessels ')
    ap.add_argument('--max-files-per-year', type=int, default=100,
                    help='Max daily files per year to load; 0 = use all files')
    ap.add_argument('--all-files', action='store_true',
                    help='Load all daily files (ignore --max-files-per-year) for full month coverage')
    ap.add_argument('--full-months', action='store_true',
                    help='Include every month in range per pair (zeros where no overlap) for full timeline')
    ap.add_argument('--distance-km', type=float, default=25.0)
    ap.add_argument('--day-window', type=int, default=1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-csv', default='artifacts/eight_pairs_overlap_by_month.csv')
    ap.add_argument('--out-plot', default='artifacts/plots/eight_pairs_overlap_by_month.png', help='Path to heatmap PNG (empty to skip)')
    ap.add_argument(
        '--enrichment',
        default='artifacts/cooperative_pairs_with_flag_gear.csv',
        help='Optional CSV (flag/gear per MMSI pair) to label heatmap rows with country + gear',
    )
    args = ap.parse_args()

    # Load pairs; if CSV has days_within_km, keep only those with at least 1 (the 8 positive)
    pairs_df = pd.read_csv(args.pairs)
    if 'days_within_km' in pairs_df.columns:
        pairs_df = pairs_df[pairs_df['days_within_km'] > 0].drop_duplicates(subset=['src', 'dst'])
    pairs_df = pairs_df[['src', 'dst']].head(20)  # cap in case no days_within_km column
    mmsi_set = set(pairs_df['src']).union(set(pairs_df['dst']))

    all_files = sorted(glob.glob(str(Path(args.daily_root) / 'mmsi-daily-csvs-10-v3-*.csv')))
    if not all_files:
        raise FileNotFoundError('No MMSI daily CSV files found.')
    max_files = 0 if args.all_files else args.max_files_per_year
    files = choose_files(all_files, max_files, args.seed)

    data = load_daily_data(files, mmsi_set)
    if data.empty:
        raise ValueError('No matching MMSI rows in selected daily files.')

    daily = data.groupby(['mmsi', 'day']).agg(
        lat_mean=('cell_ll_lat', 'mean'),
        lon_mean=('cell_ll_lon', 'mean'),
    ).reset_index()

    month_rows = []
    for _, row in pairs_df.iterrows():
        src = int(row['src'])
        dst = int(row['dst'])
        src_df = daily[daily['mmsi'] == src][['day', 'lat_mean', 'lon_mean']].copy()
        dst_df = daily[daily['mmsi'] == dst][['day', 'lat_mean', 'lon_mean']].copy()
        if src_df.empty or dst_df.empty:
            continue

        if args.day_window > 0:
            src_df = src_df.rename(columns={'day': 'day_src', 'lat_mean': 'lat_mean_src', 'lon_mean': 'lon_mean_src'})
            dst_df = dst_df.rename(columns={'day': 'day_dst', 'lat_mean': 'lat_mean_dst', 'lon_mean': 'lon_mean_dst'})
            merged = src_df.merge(dst_df, how='cross')
            merged['day_diff'] = (merged['day_src'] - merged['day_dst']).abs().dt.days
            merged = merged[merged['day_diff'] <= args.day_window]
            merged = merged.drop(columns=['day_diff'])
            merged['day'] = merged['day_src']
        else:
            merged = src_df.merge(dst_df, on='day', suffixes=('_src', '_dst'))

        if merged.empty:
            continue

        merged['distance_km'] = haversine_km(
            merged['lat_mean_src'], merged['lon_mean_src'],
            merged['lat_mean_dst'], merged['lon_mean_dst'],
        )
        merged['within_km'] = (merged['distance_km'] <= args.distance_km).astype(int)
        merged['year_month'] = merged['day'].dt.to_period('M').astype(str)

        # Days within 25 km per month: count unique canonical days when pair was within range
        close = merged[merged['within_km'] == 1].copy()
        if close.empty:
            continue
        by_month = close.groupby('year_month').agg(
            days_within_km=('day', 'nunique'),
            overlap_pairs=('within_km', 'sum'),
            mean_lat_src=('lat_mean_src', 'mean'),
            mean_lon_src=('lon_mean_src', 'mean'),
            mean_lat_dst=('lat_mean_dst', 'mean'),
            mean_lon_dst=('lon_mean_dst', 'mean'),
        ).reset_index()
        # Approximate meeting point (midpoint of the two vessel means)
        by_month['meet_lat'] = 0.5 * (by_month['mean_lat_src'] + by_month['mean_lat_dst'])
        by_month['meet_lon'] = 0.5 * (by_month['mean_lon_src'] + by_month['mean_lon_dst'])
        by_month['src'] = src
        by_month['dst'] = dst
        month_rows.append(by_month)

    if not month_rows:
        print('No monthly overlap found for the selected pairs.')
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=['src', 'dst', 'year_month', 'days_within_km', 'overlap_pairs']).to_csv(out_path, index=False)
        return

    out_df = pd.concat(month_rows, ignore_index=True)
    out_df = out_df[
        [
            'src',
            'dst',
            'year_month',
            'days_within_km',
            'overlap_pairs',
            'mean_lat_src',
            'mean_lon_src',
            'mean_lat_dst',
            'mean_lon_dst',
            'meet_lat',
            'meet_lon',
        ]
    ].sort_values(['src', 'dst', 'year_month'])

    if args.full_months:
        all_months = sorted(daily['day'].dt.to_period('M').astype(str).unique())
        full_rows = []
        for _, pr in pairs_df.iterrows():
            src, dst = int(pr['src']), int(pr['dst'])
            pair_df = out_df[(out_df['src'] == src) & (out_df['dst'] == dst)]
            for ym in all_months:
                row = pair_df[pair_df['year_month'] == ym]
                if row.empty:
                    full_rows.append(
                        {
                            'src': src,
                            'dst': dst,
                            'year_month': ym,
                            'days_within_km': 0,
                            'overlap_pairs': 0,
                            'mean_lat_src': float('nan'),
                            'mean_lon_src': float('nan'),
                            'mean_lat_dst': float('nan'),
                            'mean_lon_dst': float('nan'),
                            'meet_lat': float('nan'),
                            'meet_lon': float('nan'),
                        }
                    )
                else:
                    full_rows.append(row.iloc[0].to_dict())
        out_df = pd.DataFrame(full_rows)
        out_df = out_df.sort_values(['src', 'dst', 'year_month'])

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} ({len(out_df)} rows)")

    if args.out_plot:
        import matplotlib.pyplot as plt
        # Pivot: rows = pair label, cols = year_month, values = days_within_km
        out_df['pair'] = out_df['src'].astype(str) + '–' + out_df['dst'].astype(str)
        wide = out_df.pivot_table(index='pair', columns='year_month', values='days_within_km', fill_value=0)
        wide = wide.reindex(columns=sorted(wide.columns))

        enrich = load_enrichment_first_row_per_pair(args.enrichment)
        pair_labels = []
        for pair in wide.index:
            a, b = pair.split('–', 1)
            pair_labels.append(format_pair_label(int(a), int(b), enrich, multiline=True))

        fig, ax = plt.subplots(figsize=(max(10, wide.shape[1] * 0.4), max(4, wide.shape[0] * 0.55)))
        im = ax.imshow(wide.values, aspect='auto', cmap='YlOrRd', vmin=0)
        ax.set_xticks(range(len(wide.columns)))
        ax.set_xticklabels(wide.columns, rotation=45, ha='right')
        ax.set_yticks(range(len(wide.index)))
        ax.set_yticklabels(pair_labels, fontsize=7)
        ax.set_title('Days within 25 km by month (8 validated pairs; country · gear)')
        plt.colorbar(im, ax=ax, label='Days within 25 km')
        plt.tight_layout()
        plt.savefig(args.out_plot, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Wrote {args.out_plot}")

        # Per-year heatmaps for readable x-axis labels
        out_plot_path = Path(args.out_plot)
        out_stem = out_plot_path.stem
        out_parent = out_plot_path.parent
        years = sorted(set(str(c)[:4] for c in wide.columns))
        for year in years:
            cols_year = [c for c in wide.columns if str(c).startswith(year)]
            if not cols_year:
                continue
            wide_year = wide[cols_year].reindex(columns=sorted(cols_year))
            fig, ax = plt.subplots(figsize=(max(8, len(cols_year) * 0.5), max(4, wide_year.shape[0] * 0.55)))
            im = ax.imshow(wide_year.values, aspect='auto', cmap='YlOrRd', vmin=0)
            ax.set_xticks(range(len(wide_year.columns)))
            ax.set_xticklabels(wide_year.columns.astype(str), rotation=45, ha='right', fontsize=10)
            ax.set_yticks(range(len(wide_year.index)))
            ax.set_yticklabels(pair_labels, fontsize=7)
            ax.set_title(f'Days within 25 km by month (8 validated pairs; country · gear) — {year}')
            plt.colorbar(im, ax=ax, label='Days within 25 km')
            plt.tight_layout()
            year_path = out_parent / f"{out_stem}_{year}.png"
            plt.savefig(year_path, dpi=150, bbox_inches='tight')
            plt.close()
        if years:
            print(f"Wrote per-year heatmaps to {out_parent} ({out_stem}_<year>.png)")


if __name__ == '__main__':
    main()
