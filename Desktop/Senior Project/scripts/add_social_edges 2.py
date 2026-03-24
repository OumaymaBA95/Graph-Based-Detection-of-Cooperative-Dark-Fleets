#!/usr/bin/env python3
"""
Add social-similarity edges (same flag, optional same owner) to a proximity edge list.

Reads an existing parquet with columns src, dst, time_bucket [, distance_km].
Derives "flag" from MMSI Maritime Identification Digits (MID): first 3 digits of 9-digit
MMSI (e.g. 412xxxxxx -> China). Optionally loads mmsi,flag[,owner_id] from a CSV to
override or add ownership.

For each time_bucket, among vessels that appear in that bucket (in the proximity graph),
adds undirected edges (u, v) for same-flag pairs. To keep the graph tractable, caps the
number of social edges per bucket (or per bucket per flag). Output is a combined parquet
with both proximity and social edges (columns: src, dst, time_bucket, edge_type,
distance_km [NaN for social]).

Usage:
  python3 scripts/add_social_edges.py --edges artifacts/edges_2012_2019_cap5000_even30.parquet \\
    --out artifacts/edges_2012_2019_cap5000_with_social.parquet \\
    --max-social-per-bucket 2000
"""
import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np
import pandas as pd

from mmsi_mid import mmsi_to_mid as mmsi_to_flag


def main():
    ap = argparse.ArgumentParser(description="Add same-flag (and optional same-owner) edges.")
    ap.add_argument("--edges", required=True, help="Path to proximity edge parquet (src, dst, time_bucket)")
    ap.add_argument("--metadata", default="", help="Optional CSV: mmsi,flag[,owner_id] to override flag or add owner")
    ap.add_argument("--out", required=True, help="Output parquet (proximity + social edges)")
    ap.add_argument("--max-social-per-bucket", type=int, default=2000, help="Cap social edges per time bucket (sample if over)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_parquet(args.edges)
    if "time_bucket" not in df.columns:
        df = df.rename(columns={"date": "time_bucket"})
    df["time_bucket"] = pd.to_datetime(df["time_bucket"])
    df = df[["src", "dst", "time_bucket"]].drop_duplicates()

    # Assign flag: from metadata CSV or from MMSI MID
    all_mmsi = pd.unique(df[["src", "dst"]].to_numpy().ravel())
    flag_series = pd.Series(index=all_mmsi, dtype=int)
    for m in all_mmsi:
        flag_series.loc[m] = mmsi_to_flag(m)

    if args.metadata and Path(args.metadata).exists():
        meta = pd.read_csv(args.metadata)
        meta = meta.dropna(subset=["mmsi"])
        meta["mmsi"] = meta["mmsi"].astype(int)
        if "flag" in meta.columns:
            meta["flag"] = meta["flag"].astype(int)
            for _, row in meta.iterrows():
                if row["mmsi"] in flag_series.index:
                    flag_series.loc[row["mmsi"]] = row["flag"]
    vessel_flag = flag_series.to_dict()

    # Proximity edges with edge_type
    raw = pd.read_parquet(args.edges)
    df["edge_type"] = "proximity"
    if "distance_km" in raw.columns:
        dist = raw[["src", "dst", "time_bucket", "distance_km"]].drop_duplicates()
        df = df.merge(dist, on=["src", "dst", "time_bucket"], how="left")
    else:
        df["distance_km"] = np.nan

    social_rows = []
    rng = np.random.default_rng(args.seed)

    for t, g in df.groupby("time_bucket"):
        vessels = pd.unique(g[["src", "dst"]].to_numpy().ravel())
        flags = [vessel_flag.get(v, mmsi_to_flag(v)) for v in vessels]
        by_flag = {}
        for v, f in zip(vessels, flags):
            by_flag.setdefault(f, []).append(v)
        bucket_pairs = []
        for flag, nodes in by_flag.items():
            if len(nodes) < 2:
                continue
            nodes = np.array(nodes)
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    u, v = int(nodes[i]), int(nodes[j])
                    bucket_pairs.append((u, v))
        if not bucket_pairs:
            continue
        if len(bucket_pairs) > args.max_social_per_bucket:
            idx = rng.choice(len(bucket_pairs), size=args.max_social_per_bucket, replace=False)
            bucket_pairs = [bucket_pairs[i] for i in idx]
        for u, v in bucket_pairs:
            social_rows.append({
                "src": u,
                "dst": v,
                "time_bucket": t,
                "edge_type": "social",
                "distance_km": np.nan,
            })

    if social_rows:
        social_df = pd.DataFrame(social_rows)
        out_df = pd.concat([df, social_df], ignore_index=True)
    else:
        out_df = df

    out_df = out_df[["src", "dst", "time_bucket", "edge_type", "distance_km"]]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    n_prox = (out_df["edge_type"] == "proximity").sum()
    n_soc = (out_df["edge_type"] == "social").sum()
    print(f"Wrote {args.out}: {n_prox} proximity + {n_soc} social edges ({len(out_df)} total)")


if __name__ == "__main__":
    main()
