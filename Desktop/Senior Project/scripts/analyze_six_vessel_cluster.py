#!/usr/bin/env python3
"""
Analyze the 8 validated pairs and highlight a tight cluster of ~six vessels.

Input:
  - artifacts/eight_pairs_overlap_by_month.csv
    (from overlap_by_month_8pairs.py with --all-files --full-months)

Output:
  - artifacts/six_vessel_cluster_summary.csv
      One row per selected vessel with:
        mmsi, times_in_pairs, total_overlap_days, approx_lat, approx_lon
  - artifacts/plots/six_vessel_cluster_scatter.png
      Geographic scatter of rendezvous points with coastline, land, and city
      labels (cartopy). Highlights the six-vessel cluster.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from vessel_pair_labels import load_mmsi_country_gear_map, mmsi_country_gear_line

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    _HAS_CARTOPY = True
except ImportError:
    _HAS_CARTOPY = False

_PUBLICATION_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica Neue", "Arial", "sans-serif"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.titleweight": "bold",
    "axes.labelweight": "medium",
    "axes.facecolor": "#ffffff",
    "figure.facecolor": "#ffffff",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.labelcolor": "#1a1a2e",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "grid.color": "#cccccc",
    "grid.alpha": 0.4,
    "grid.linewidth": 0.5,
    "grid.linestyle": "-",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
}

_CLUSTER_COLORS = [
    "#e85d04", "#2d6a4f", "#c9184a",
    "#7b2cbf", "#6d4c41", "#d63384",
]

_CLUSTER_MARKERS = ["o", "s", "D", "^", "v", "P"]

_CITIES = {
    "Shanghai":  (121.47, 31.23),
    "Ningbo":    (121.55, 29.87),
    "Wenzhou":   (120.65, 28.02),
    "Fuzhou":    (119.30, 26.07),
    "Hangzhou":  (120.15, 30.25),
    "Qingdao":   (120.38, 36.07),
}


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return r * c


def main():
    ap = argparse.ArgumentParser(description="Find a six-vessel spatial cluster.")
    ap.add_argument(
        "--overlap-csv",
        default="artifacts/eight_pairs_overlap_by_month.csv",
        help="CSV from overlap_by_month_8pairs.py (with locations).",
    )
    ap.add_argument(
        "--max-vessels",
        type=int,
        default=6,
        help="Maximum number of vessels to keep in the cluster.",
    )
    ap.add_argument(
        "--out-summary",
        default="artifacts/six_vessel_cluster_summary.csv",
    )
    ap.add_argument(
        "--out-plot",
        default="artifacts/plots/six_vessel_cluster_scatter.png",
    )
    ap.add_argument(
        "--enrichment",
        default="artifacts/cooperative_pairs_with_flag_gear.csv",
        help="CSV for vessel country · gear labels in legend (optional)",
    )
    args = ap.parse_args()

    mmsi_map = load_mmsi_country_gear_map(args.enrichment)

    df = pd.read_csv(args.overlap_csv)
    # We need rows where there is actual proximity and location estimates
    df = df[
        (df["days_within_km"] > 0)
        & df["meet_lat"].notna()
        & df["meet_lon"].notna()
    ].copy()
    if df.empty:
        raise ValueError("No rows with overlap and valid meet_lat/meet_lon.")

    # Compute a global hotspot: weighted centroid of meeting points
    weights = df["days_within_km"].clip(lower=1)
    lat0 = np.average(df["meet_lat"], weights=weights)
    lon0 = np.average(df["meet_lon"], weights=weights)

    # Distance of each pair-month rendezvous from the hotspot
    df["dist_to_hotspot_km"] = haversine_km(df["meet_lat"], df["meet_lon"], lat0, lon0)

    # Aggregate per pair: where do they usually meet, and how often
    pair_stats = (
        df.groupby(["src", "dst"])
        .agg(
            total_days=("days_within_km", "sum"),
            mean_meet_lat=("meet_lat", "mean"),
            mean_meet_lon=("meet_lon", "mean"),
            mean_dist_to_hotspot_km=("dist_to_hotspot_km", "mean"),
        )
        .reset_index()
    )

    # Sort pairs: closer to hotspot and with more overlapping days
    pair_stats = pair_stats.sort_values(
        ["mean_dist_to_hotspot_km", "total_days"], ascending=[True, False]
    )

    # Greedy selection of vessels from the closest pairs until we have <= max_vessels
    selected_vessels: set[int] = set()
    selected_pairs = []
    for _, row in pair_stats.iterrows():
        src = int(row["src"])
        dst = int(row["dst"])
        # If we already have the max vessels and both are new, skip
        new_mmsis = {src, dst} - selected_vessels
        if selected_vessels and len(selected_vessels) >= args.max_vessels and new_mmsis:
            continue
        selected_pairs.append(row)
        selected_vessels.update([src, dst])
        if len(selected_vessels) >= args.max_vessels:
            # we can still add more pairs involving these same vessels, but no new ones
            continue

    selected_pairs_df = pd.DataFrame(selected_pairs)

    # Per-vessel summary: how often do they appear, total overlap days, and avg location
    vessel_rows = []
    for mmsi in sorted(selected_vessels):
        # rows where this mmsi is either src or dst
        mdf = df[(df["src"] == mmsi) | (df["dst"] == mmsi)]
        if mdf.empty:
            continue
        vessel_rows.append(
            {
                "mmsi": mmsi,
                "times_in_pairs": int(
                    (selected_pairs_df["src"] == mmsi).sum()
                    + (selected_pairs_df["dst"] == mmsi).sum()
                ),
                "total_overlap_days": int(mdf["days_within_km"].sum()),
                "approx_lat": float(mdf["meet_lat"].mean()),
                "approx_lon": float(mdf["meet_lon"].mean()),
            }
        )

    vessel_df = pd.DataFrame(vessel_rows)
    vessel_df = vessel_df.sort_values(
        ["total_overlap_days", "times_in_pairs"], ascending=[False, False]
    )

    # Save summary
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    vessel_df.to_csv(args.out_summary, index=False)
    print(f"Wrote {args.out_summary} ({len(vessel_df)} vessels)")
    print(f"Hotspot approx at lat={lat0:.3f}, lon={lon0:.3f}")

    # --- Geographic scatter plot ---
    Path(args.out_plot).parent.mkdir(parents=True, exist_ok=True)

    all_lons = df["meet_lon"].values
    all_lats = df["meet_lat"].values
    pad_lon, pad_lat = 0.8, 0.8
    extent = [
        min(all_lons.min(), 119.0) - pad_lon,
        all_lons.max() + pad_lon,
        min(all_lats.min(), 26.0) - pad_lat,
        all_lats.max() + pad_lat,
    ]

    cluster_lons = []
    cluster_lats = []
    for mmsi in selected_vessels:
        sub = df[(df["src"] == mmsi) | (df["dst"] == mmsi)]
        cluster_lons.extend(sub["meet_lon"].tolist())
        cluster_lats.extend(sub["meet_lat"].tolist())
    cluster_lons = np.array(cluster_lons)
    cluster_lats = np.array(cluster_lats)

    with plt.rc_context(_PUBLICATION_RC):
        if _HAS_CARTOPY:
            proj = ccrs.PlateCarree()
            fig, ax = plt.subplots(
                figsize=(12, 9.5), subplot_kw={"projection": proj},
                constrained_layout=True,
            )
            ax.set_extent(extent, crs=proj)
            t = proj

            ax.add_feature(cfeature.OCEAN, facecolor="#dbe9f4", zorder=0)
            ax.add_feature(cfeature.LAND, facecolor="#e8e4d8", edgecolor="none", zorder=1)
            ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor="#555555", zorder=6)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="#999999", linestyle="--", zorder=6)
            try:
                ax.add_feature(cfeature.RIVERS, linewidth=0.4, edgecolor="#7fafcf", zorder=6)
            except Exception:
                pass

            gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#aaaaaa",
                              alpha=0.5, linestyle="-")
            gl.top_labels = False
            gl.right_labels = False

            for city, (clon, clat) in _CITIES.items():
                if extent[0] <= clon <= extent[1] and extent[2] <= clat <= extent[3]:
                    ax.plot(clon, clat, "k^", ms=6, transform=t, zorder=8)
                    ax.text(
                        clon + 0.12, clat + 0.1, city, fontsize=8.5,
                        fontweight="bold", color="#222222", transform=t, zorder=8,
                        path_effects=[pe.withStroke(linewidth=3, foreground="white")],
                    )
        else:
            fig, ax = plt.subplots(figsize=(12, 9.5), constrained_layout=True)
            t = None
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

        scatter_kw = {"transform": t} if t else {}

        if len(cluster_lons) >= 3:
            from scipy.spatial import ConvexHull
            try:
                pts_2d = np.column_stack([cluster_lons, cluster_lats])
                hull = ConvexHull(pts_2d)
                hull_pts = pts_2d[hull.vertices]
                hull_pts = np.vstack([hull_pts, hull_pts[0]])
                from matplotlib.patches import Polygon as MplPolygon
                hull_patch = MplPolygon(
                    hull_pts, closed=True,
                    facecolor="#ffa726", edgecolor="#e65100",
                    alpha=0.12, linewidth=1.8, linestyle="--",
                    zorder=2, **({"transform": t} if t else {}),
                )
                ax.add_patch(hull_patch)
            except Exception:
                pass

        ax.scatter(
            all_lons, all_lats,
            s=12, alpha=0.22, color="#90a4ae", edgecolors="none",
            label="All rendezvous", zorder=2, **scatter_kw,
        )

        sorted_vessels = sorted(selected_vessels)
        drawn_edges = set()
        for _, prow in selected_pairs_df.iterrows():
            s, d = int(prow["src"]), int(prow["dst"])
            edge_key = (min(s, d), max(s, d))
            if edge_key in drawn_edges:
                continue
            drawn_edges.add(edge_key)
            s_lat = vessel_df.loc[vessel_df["mmsi"] == s, "approx_lat"]
            s_lon = vessel_df.loc[vessel_df["mmsi"] == s, "approx_lon"]
            d_lat = vessel_df.loc[vessel_df["mmsi"] == d, "approx_lat"]
            d_lon = vessel_df.loc[vessel_df["mmsi"] == d, "approx_lon"]
            if not s_lat.empty and not d_lat.empty:
                ax.plot(
                    [s_lon.iloc[0], d_lon.iloc[0]],
                    [s_lat.iloc[0], d_lat.iloc[0]],
                    color="#666666", linewidth=1.8, linestyle="-",
                    alpha=0.55, zorder=3,
                    path_effects=[pe.withStroke(linewidth=3.5, foreground="white")],
                    **scatter_kw,
                )

        for idx, mmsi in enumerate(sorted_vessels):
            sub = df[(df["src"] == mmsi) | (df["dst"] == mmsi)]
            cg = mmsi_country_gear_line(int(mmsi), mmsi_map)
            short_id = f"...{str(mmsi)[-5:]}"
            color = _CLUSTER_COLORS[idx % len(_CLUSTER_COLORS)]
            marker = _CLUSTER_MARKERS[idx % len(_CLUSTER_MARKERS)]
            ax.scatter(
                sub["meet_lon"], sub["meet_lat"],
                s=90, alpha=0.92, color=color, marker=marker,
                edgecolors="white", linewidths=0.8,
                label=f"{short_id}: {cg}",
                zorder=5, **scatter_kw,
            )

        ax.scatter(
            [lon0], [lat0],
            s=160, c="#d32f2f", marker="X", linewidths=2.0,
            edgecolors="white", label="Hotspot centroid",
            zorder=7, **scatter_kw,
        )

        ax.set_title(
            "Rendezvous locations for eight validated pairs\n"
            "Six-vessel cluster highlighted near the East China Sea",
            fontsize=13, fontweight="bold", pad=12,
        )
        ax.legend(
            fontsize=9, loc="upper left", framealpha=0.94,
            edgecolor="#888888", fancybox=True,
            borderpad=0.8, handletextpad=0.6,
        )

        fig.savefig(args.out_plot, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"Wrote {args.out_plot}")


if __name__ == "__main__":
    main()

