#!/usr/bin/env python3
"""
Compute distances between cooperative MMSI pairs using vessel_day_features.

Input:
- artifacts/cooperative_pairs_labeled.csv  (src,dst,date)
- data/features_by_year/*/vessel_day_features.parquet

Output:
- artifacts/cooperative_pair_distances.csv
    src,dst,overlap_days,days_within_10km,mean_distance_km
"""
from pathlib import Path
from typing import List
import glob

import numpy as np
import pandas as pd


COOP_PATH = Path("artifacts/cooperative_pairs_labeled.csv")
FEATURES_ROOT = Path("data/features_by_year")
OUT_PATH = Path("artifacts/cooperative_pair_distances.csv")


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return r * c


def load_features(root: Path) -> pd.DataFrame:
    files: List[str] = sorted(glob.glob(str(root / "*" / "vessel_day_features.parquet")))
    if not files:
        raise FileNotFoundError(f"No vessel_day_features.parquet files found under {root}")
    frames = [pd.read_parquet(f) for f in files]
    data = pd.concat(frames, ignore_index=True)
    data["day"] = pd.to_datetime(data["day"])
    return data


def main() -> None:
    if not COOP_PATH.exists():
        raise SystemExit(f"Cooperative pairs file not found: {COOP_PATH}")

    coop = pd.read_csv(COOP_PATH)
    features = load_features(FEATURES_ROOT)

    rows = []
    # Work on unique src,dst pairs from cooperative file
    pairs = coop[["src", "dst"]].drop_duplicates()

    for _, row in pairs.iterrows():
        src = int(row["src"])
        dst = int(row["dst"])

        src_df = features[features["MMSI"] == src][["day", "lat_mean", "lon_mean"]].dropna()
        dst_df = features[features["MMSI"] == dst][["day", "lat_mean", "lon_mean"]].dropna()

        if src_df.empty or dst_df.empty:
            continue

        merged = src_df.merge(dst_df, on="day", suffixes=("_src", "_dst"))
        if merged.empty:
            continue

        merged["distance_km"] = haversine_km(
            merged["lat_mean_src"],
            merged["lon_mean_src"],
            merged["lat_mean_dst"],
            merged["lon_mean_dst"],
        )
        overlap_days = len(merged)
        days_within_10 = int((merged["distance_km"] <= 10.0).sum())
        mean_dist = float(merged["distance_km"].mean())

        rows.append(
            {
                "src": src,
                "dst": dst,
                "overlap_days": overlap_days,
                "days_within_10km": days_within_10,
                "mean_distance_km": mean_dist,
            }
        )

    out_df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"Wrote cooperative pair distances to {OUT_PATH} ({len(out_df)} pairs)")


if __name__ == "__main__":
    main()

