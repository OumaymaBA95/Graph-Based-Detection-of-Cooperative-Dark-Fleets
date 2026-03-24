#!/usr/bin/env python3
"""
Compute pair overlap statistics and plots using raw MMSI daily CSVs.

This uses cell_ll_lat/lon as the daily location proxy for each MMSI.
"""
import argparse
import glob
import sys
from pathlib import Path
from typing import List, Optional, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from vessel_pair_labels import (
    load_enrichment_first_row_per_pair,
    load_mmsi_country_gear_map,
    pair_plot_title_country_gear,
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    _HAS_CARTOPY = True
except ImportError:
    _HAS_CARTOPY = False

_PUBLICATION_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica Neue", "Arial", "sans-serif"],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.titleweight": "bold",
    "axes.labelweight": "medium",
    "axes.facecolor": "#ffffff",
    "figure.facecolor": "#ffffff",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.0,
    "axes.labelcolor": "#1a1a2e",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "grid.color": "#cccccc",
    "grid.alpha": 0.4,
    "grid.linewidth": 0.5,
    "grid.linestyle": "-",
}


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return r * c


def _kde_on_grid(
    lon: np.ndarray,
    lat: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    max_points: int = 8000,
    bw: float = 1.35,
) -> Optional[np.ndarray]:
    """Evaluate a 2D KDE on a shared meshgrid. Returns raw Z or None."""
    lon = np.asarray(lon, dtype=float).ravel()
    lat = np.asarray(lat, dtype=float).ravel()
    n = lon.size
    if n < 12:
        return None
    rng = np.random.default_rng(42)
    if n > max_points:
        pick = rng.choice(n, size=max_points, replace=False)
        lon, lat = lon[pick], lat[pick]
    try:
        try:
            kde = gaussian_kde(np.vstack([lon, lat]), bw_method=float(bw))
        except TypeError:
            kde = gaussian_kde(np.vstack([lon, lat]))
    except (np.linalg.LinAlgError, ValueError):
        return None
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    return gaussian_filter(Z, sigma=1.2)


def _histogram_on_grid(
    lon: np.ndarray,
    lat: np.ndarray,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    grid: int,
) -> np.ndarray:
    """Fallback: heavily smoothed 2D histogram when KDE fails."""
    H, _, _ = np.histogram2d(
        lon, lat, bins=grid, range=[[lon_min, lon_max], [lat_min, lat_max]]
    )
    return gaussian_filter(H, sigma=4.5)


def _compute_density(
    lon1: np.ndarray,
    lat1: np.ndarray,
    lon2: np.ndarray,
    lat2: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float]]:
    """Compute combined KDE/histogram density. Returns (X, Y, Z_norm, lon_min, lon_max, lat_min, lat_max)."""
    lon1 = np.asarray(lon1, dtype=float).ravel()
    lat1 = np.asarray(lat1, dtype=float).ravel()
    lon2 = np.asarray(lon2, dtype=float).ravel()
    lat2 = np.asarray(lat2, dtype=float).ravel()
    if lon1.size < 10 or lon2.size < 10:
        return None

    lon_all = np.concatenate([lon1, lon2])
    lat_all = np.concatenate([lat1, lat2])
    span_lon = lon_all.max() - lon_all.min()
    span_lat = lat_all.max() - lat_all.min()
    pad_lon = max(0.18 * span_lon, 0.08) if span_lon > 0 else 0.1
    pad_lat = max(0.18 * span_lat, 0.08) if span_lat > 0 else 0.1
    lon_min = lon_all.min() - pad_lon
    lon_max_ = lon_all.max() + pad_lon
    lat_min = lat_all.min() - pad_lat
    lat_max_ = lat_all.max() + pad_lat

    grid = 200
    xi = np.linspace(lon_min, lon_max_, grid)
    yi = np.linspace(lat_min, lat_max_, grid)
    X, Y = np.meshgrid(xi, yi, indexing="ij")

    Z1 = _kde_on_grid(lon1, lat1, X, Y)
    Z2 = _kde_on_grid(lon2, lat2, X, Y)
    if Z1 is None:
        Z1 = _histogram_on_grid(lon1, lat1, lon_min, lon_max_, lat_min, lat_max_, grid)
    if Z2 is None:
        Z2 = _histogram_on_grid(lon2, lat2, lon_min, lon_max_, lat_min, lat_max_, grid)

    Z_combined = Z1 + Z2
    zmax = float(Z_combined.max())
    if zmax <= 0:
        return None
    Z_combined = Z_combined / zmax
    return X, Y, Z_combined, lon_min, lon_max_, lat_min, lat_max_


def _draw_contours(fig, ax, X, Y, Z_combined, *, use_cartopy: bool = False):
    """Draw filled contour + iso-lines + colorbar on ax."""
    n_levels = 20
    levels = np.linspace(0.0, 1.0, n_levels + 1)

    transform_kw = {"transform": ccrs.PlateCarree()} if use_cartopy else {}

    cf = ax.contourf(
        X, Y, Z_combined,
        levels=levels,
        cmap="Spectral_r",
        antialiased=True,
        zorder=2,
        extend="neither",
        **transform_kw,
    )
    line_levels = levels[2::4]
    cs = ax.contour(
        X, Y, Z_combined,
        levels=line_levels,
        colors="#333333",
        linewidths=0.65,
        zorder=3,
        **transform_kw,
    )
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f", inline_spacing=5)

    cbar = fig.colorbar(
        cf, ax=ax, orientation="horizontal", pad=0.06, shrink=0.65, aspect=35,
    )
    cbar.set_label("Combined daily presence (normalized)", fontsize=10.5, labelpad=6)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_linewidth(0.6)


def choose_files(files: List[str], max_files: int, sampling: str, seed: int) -> List[str]:
    if max_files <= 0 or max_files >= len(files):
        return files
    if sampling == 'random':
        rng = np.random.default_rng(seed)
        return sorted(rng.choice(files, size=max_files, replace=False).tolist())
    step = (len(files) - 1) / max_files
    idx = [round(i * step) for i in range(max_files)]
    return [files[i] for i in idx]


def load_daily_data(files: List[str], mmsi_set: set[int]) -> pd.DataFrame:
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
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', default='artifacts/tgcn_candidate_pairs_enriched_top50.csv')
    ap.add_argument('--daily-root', default='data/MMSI daily vessels ')
    ap.add_argument('--top-k', type=int, default=5)
    ap.add_argument('--distance-km', type=float, default=10.0)
    ap.add_argument('--day-window', type=int, default=0, help='Allow ±N day window for overlap')
    ap.add_argument('--exact-cell', action='store_true', help='Use exact cell overlap instead of distance')
    ap.add_argument('--region-deg', type=float, default=0.0, help='Use region bins (degrees) instead of distance')
    ap.add_argument(
        '--max-files-per-year',
        type=int,
        default=0,
        help='Cap how many daily CSVs are sampled from the full list (evenly spaced). '
        'Use 0 for no cap (all files) — required for validation counts like "days within 25 km" '
        'to match full-coverage reports. Small values (e.g. 5) make plots fast but undercount badly.',
    )
    ap.add_argument('--file-sampling', choices=['even', 'random'], default='even')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-dir', default='artifacts/plots/candidate_pairs_daily')
    ap.add_argument('--out-summary', default='artifacts/candidate_pair_overlap_summary_daily.csv')
    ap.add_argument(
        '--enrichment',
        default='artifacts/cooperative_pairs_with_flag_gear.csv',
        help='CSV with src/dst flag_cell + gear (optional). Missing vessels use ITU MID + — for gear.',
    )
    ap.add_argument(
        '--contour',
        action='store_true',
        help=(
            'Bold filled contour of combined daily presence (2D KDE, turbo colormap, labeled iso-lines, '
            'colorbar). First pair: pair_<src>_<dst>.png (tracks only) + pair_<src>_<dst>_contour.png '
            '(with overlay). Use --contour-all-pairs for contour on every pair.'
        ),
    )
    ap.add_argument(
        '--contour-all-pairs',
        action='store_true',
        help='With --contour, draw contours for every pair, not only the first.',
    )
    args = ap.parse_args()

    mmsi_map = load_mmsi_country_gear_map(args.enrichment)
    enrich_pairs = load_enrichment_first_row_per_pair(args.enrichment)

    pairs = pd.read_csv(args.pairs).head(args.top_k)
    mmsi_set = set(pairs['src']).union(set(pairs['dst']))

    all_files = sorted(glob.glob(str(Path(args.daily_root) / 'mmsi-daily-csvs-10-v3-*.csv')))
    if not all_files:
        raise FileNotFoundError('No MMSI daily CSV files found.')
    files = choose_files(all_files, args.max_files_per_year, args.file_sampling, args.seed)

    data = load_daily_data(files, mmsi_set)
    if data.empty:
        raise ValueError('No matching MMSI rows found in selected daily files.')

    daily = data.groupby(['mmsi', 'day']).agg(
        lat_mean=('cell_ll_lat', 'mean'),
        lon_mean=('cell_ll_lon', 'mean'),
    ).reset_index()
    if args.region_deg > 0:
        daily['region_lat'] = (daily['lat_mean'] / args.region_deg).round(0) * args.region_deg
        daily['region_lon'] = (daily['lon_mean'] / args.region_deg).round(0) * args.region_deg

    days_active = daily.groupby('mmsi')['day'].nunique().to_dict()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for pair_idx, (_, row) in enumerate(pairs.iterrows()):
        src = int(row['src'])
        dst = int(row['dst'])

        if args.region_deg > 0:
            src_df = daily[daily['mmsi'] == src][['day', 'region_lat', 'region_lon']]
            dst_df = daily[daily['mmsi'] == dst][['day', 'region_lat', 'region_lon']]
        else:
            src_df = daily[daily['mmsi'] == src][['day', 'lat_mean', 'lon_mean']]
            dst_df = daily[daily['mmsi'] == dst][['day', 'lat_mean', 'lon_mean']]

        if src_df.empty or dst_df.empty:
            continue

        if args.day_window > 0:
            src_df = src_df.copy()
            dst_df = dst_df.copy()
            src_df['day_key'] = src_df['day']
            dst_df['day_key'] = dst_df['day']
            merged = src_df.merge(dst_df, how='cross', suffixes=('_src', '_dst'))
            merged['day_diff'] = (merged['day_key_src'] - merged['day_key_dst']).abs().dt.days
            merged = merged[merged['day_diff'] <= args.day_window]
            merged = merged.drop(columns=['day_key_src', 'day_key_dst', 'day_diff'])
        else:
            merged = src_df.merge(dst_df, on='day', suffixes=('_src', '_dst'))
        if not merged.empty:
            if args.region_deg > 0:
                merged['same_region'] = (
                    merged['region_lat_src'] == merged['region_lat_dst']
                ) & (
                    merged['region_lon_src'] == merged['region_lon_dst']
                )
                close_days = int(merged['same_region'].sum())
                mean_distance = np.nan
            elif args.exact_cell:
                merged['same_cell'] = (
                    merged['lat_mean_src'].round(4) == merged['lat_mean_dst'].round(4)
                ) & (
                    merged['lon_mean_src'].round(4) == merged['lat_mean_dst'].round(4)
                )
                close_days = int(merged['same_cell'].sum())
                mean_distance = 0.0 if close_days > 0 else np.nan
            else:
                merged['distance_km'] = haversine_km(
                    merged['lat_mean_src'],
                    merged['lon_mean_src'],
                    merged['lat_mean_dst'],
                    merged['lon_mean_dst'],
                )
                close_days = int((merged['distance_km'] <= args.distance_km).sum())
                mean_distance = merged['distance_km'].mean()
        else:
            close_days = 0
            mean_distance = np.nan

        src_days = days_active.get(src, 0)
        dst_days = days_active.get(dst, 0)
        overlap_ratio = close_days / max(1, min(src_days, dst_days))
        overlap_ratio_days = len(merged) / max(1, min(src_days, dst_days)) if len(merged) else 0.0

        summary_rows.append({
            'src': src,
            'dst': dst,
            'overlap_days': len(merged),
            'days_within_km': close_days,
            'mean_distance_km': mean_distance,
            'src_days_active': src_days,
            'dst_days_active': dst_days,
            'overlap_ratio_days': overlap_ratio_days,
            'close_ratio_days': overlap_ratio,
            'files_used': len(files),
        })

        color_src = "#0072B2"
        color_dst = "#D55E00"

        if pair_idx == 0 and args.contour:
            save_modes: List[Tuple[bool, str]] = [
                (False, f"pair_{src}_{dst}.png"),
                (True, f"pair_{src}_{dst}_contour.png"),
            ]
        elif args.contour and args.contour_all_pairs:
            save_modes = [(True, f"pair_{src}_{dst}_contour.png")]
        else:
            save_modes = [(False, f"pair_{src}_{dst}.png")]

        def _track_plot(x, y, color, label, z, *, contour_mode=False, transform=None):
            n = len(x)
            if n == 0:
                return
            xv = x.to_numpy() if hasattr(x, "to_numpy") else np.asarray(x)
            yv = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
            tkw = {"transform": transform} if transform is not None else {}
            if contour_mode:
                ax.plot(
                    xv, yv, color=color, linewidth=2.0, linestyle="-",
                    solid_capstyle="round", alpha=1.0, label=label, zorder=z + 4,
                    path_effects=[pe.withStroke(linewidth=3.8, foreground="white")],
                    **tkw,
                )
                ax.scatter(
                    [xv[0], xv[-1]], [yv[0], yv[-1]],
                    c=color, s=50, alpha=1.0, marker="o",
                    edgecolors="white", linewidths=1.0, zorder=z + 5,
                    **tkw,
                )
            else:
                lw = 2.5 if n < 400 else 2.15
                ax.plot(
                    xv, yv, color=color, linewidth=lw, linestyle="-",
                    solid_capstyle="round", alpha=0.95, label=label, zorder=z,
                )
                if n <= 50:
                    every = 1
                elif n <= 200:
                    every = max(1, n // 45)
                else:
                    every = max(1, n // 55)
                idx = list(range(0, n, every))
                if (n - 1) not in idx and n > 1:
                    idx.append(n - 1)
                ms = 13 if n < 500 else 10
                ax.scatter(
                    xv[idx], yv[idx], c=color, s=ms, alpha=0.88,
                    edgecolors="white", linewidths=0.35, zorder=z + 1,
                )

        src_plot = src_df.sort_values("day")
        dst_plot = dst_df.sort_values("day")
        src_for_range = src_df.sort_values('day') if 'day' in src_df.columns else src_df
        date_range = (
            f"{src_for_range['day'].min().strftime('%Y-%m')} to "
            f"{src_for_range['day'].max().strftime('%Y-%m')}"
            if not src_for_range.empty and 'day' in src_for_range.columns
            else ""
        )

        lat_col_s = "region_lat" if args.region_deg > 0 else "lat_mean"
        lon_col_s = "region_lon" if args.region_deg > 0 else "lon_mean"

        for use_contour, out_name in save_modes:
            with plt.rc_context(_PUBLICATION_RC):
                use_geo = use_contour and _HAS_CARTOPY
                density = None
                if use_contour:
                    density = _compute_density(
                        src_plot[lon_col_s].to_numpy(),
                        src_plot[lat_col_s].to_numpy(),
                        dst_plot[lon_col_s].to_numpy(),
                        dst_plot[lat_col_s].to_numpy(),
                    )

                if use_geo and density is not None:
                    _, _, _, d_lonmin, d_lonmax, d_latmin, d_latmax = density
                    # Wide view: show substantial Chinese coastline for context.
                    span_lon = d_lonmax - d_lonmin
                    span_lat = d_latmax - d_latmin
                    view_lonmin = d_lonmin - 0.7 * span_lon
                    view_lonmax = d_lonmax + 0.15 * span_lon
                    view_latmin = d_latmin - 0.15 * span_lat
                    view_latmax = d_latmax + 0.15 * span_lat
                    proj = ccrs.PlateCarree()
                    fig, ax = plt.subplots(
                        figsize=(14, 9.5), dpi=150,
                        subplot_kw={"projection": proj},
                    )
                    ax.set_extent([view_lonmin, view_lonmax, view_latmin, view_latmax], crs=proj)

                    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#dce8f0", zorder=0)

                    X, Y, Z, *_ = density
                    _draw_contours(fig, ax, X, Y, Z, use_cartopy=True)

                    ax.add_feature(
                        cfeature.LAND.with_scale("10m"),
                        facecolor="#e8e4d8", edgecolor="none", zorder=4,
                    )
                    ax.add_feature(
                        cfeature.COASTLINE.with_scale("10m"),
                        linewidth=1.0, edgecolor="#333333", zorder=5,
                    )
                    ax.add_feature(
                        cfeature.BORDERS.with_scale("10m"),
                        linewidth=0.5, edgecolor="#888888", linestyle=":", zorder=5,
                    )
                    ax.add_feature(
                        cfeature.RIVERS.with_scale("10m"),
                        linewidth=0.35, edgecolor="#88aabb", zorder=5,
                    )
                    gl = ax.gridlines(
                        draw_labels=True, linewidth=0.4,
                        color="#aaaaaa", alpha=0.5, linestyle="--", zorder=6,
                    )
                    gl.top_labels = False
                    gl.right_labels = False

                    # Reference cities along the coast.
                    _cities = [
                        (121.47, 31.23, "Shanghai"),
                        (120.38, 36.07, "Qingdao"),
                        (117.00, 36.65, "Jinan"),
                        (118.80, 32.06, "Nanjing"),
                        (121.61, 38.91, "Dalian"),
                        (119.95, 33.38, "Yancheng"),
                    ]
                    t = ccrs.PlateCarree()
                    for clon, clat, cname in _cities:
                        if view_lonmin <= clon <= view_lonmax and view_latmin <= clat <= view_latmax:
                            ax.plot(
                                clon, clat, marker="s", markersize=5,
                                color="#222222", transform=t, zorder=7,
                            )
                            ax.text(
                                clon + 0.15, clat + 0.15, cname,
                                fontsize=8.5, color="#222222", fontweight="semibold",
                                transform=t, zorder=7,
                                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
                            )

                    _track_plot(
                        src_plot[lon_col_s], src_plot[lat_col_s],
                        color_src, f"Vessel {src}", 2,
                        contour_mode=True, transform=t,
                    )
                    _track_plot(
                        dst_plot[lon_col_s], dst_plot[lat_col_s],
                        color_dst, f"Vessel {dst}", 3,
                        contour_mode=True, transform=t,
                    )
                else:
                    fig, ax = plt.subplots(figsize=(12.5, 8.5), dpi=150)
                    if density is not None:
                        X, Y, Z, *_ = density
                        _draw_contours(fig, ax, X, Y, Z, use_cartopy=False)
                    _track_plot(
                        src_plot[lon_col_s], src_plot[lat_col_s],
                        color_src, f"Vessel {src}", 2, contour_mode=use_contour,
                    )
                    _track_plot(
                        dst_plot[lon_col_s], dst_plot[lat_col_s],
                        color_dst, f"Vessel {dst}", 3, contour_mode=use_contour,
                    )
                    ax.set_xlabel("Longitude (°E)")
                    ax.set_ylabel("Latitude (°N)")
                    ax.grid(True, zorder=-1)
                    ax.set_axisbelow(True)

                title_cg, _ = pair_plot_title_country_gear(src, dst, enrich_pairs, mmsi_map)
                title_line1 = f"Vessels {src} & {dst}  ({title_cg})"
                title_line2 = f"{close_days} days within {args.distance_km:.0f} km  ·  {date_range}"
                ax.set_title(
                    f"{title_line1}\n{title_line2}",
                    pad=12, linespacing=1.4,
                )

                leg = ax.legend(
                    loc="upper left", fontsize=10.5,
                    frameon=True, fancybox=False, shadow=False,
                    framealpha=0.92, edgecolor="#888888", facecolor="#ffffff",
                )
                leg.get_frame().set_linewidth(0.6)

                plt.savefig(
                    out_dir / out_name, dpi=300,
                    bbox_inches="tight", facecolor="#ffffff", pad_inches=0.12,
                )
                plt.close()

    summary_df = pd.DataFrame(summary_rows)
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.out_summary, index=False)
    print(f"Wrote plots to {out_dir}")
    print(f"Wrote summary to {args.out_summary}")


if __name__ == '__main__':
    main()
