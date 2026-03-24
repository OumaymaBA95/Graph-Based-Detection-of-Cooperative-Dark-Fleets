#!/usr/bin/env python3
"""
Export MMSI pairs with cooperation labels and model scores from
artifacts/tgcn_time_temporal_edges.csv.

Outputs:
- artifacts/cooperative_pairs_labeled.csv
- artifacts/cooperative_pairs_top100_by_score.csv
"""
from pathlib import Path

import pandas as pd


IN_PATH = Path("artifacts/tgcn_time_temporal_edges.csv")
OUT_LABELED = Path("artifacts/cooperative_pairs_labeled.csv")
OUT_TOP = Path("artifacts/cooperative_pairs_top100_by_score.csv")


def main() -> None:
    if not IN_PATH.exists():
        raise SystemExit(f"Input edges CSV not found: {IN_PATH}")

    edges = pd.read_csv(IN_PATH)

    # 1) All labeled cooperators (label == 1)
    labeled = edges[edges["label"] == 1].copy()
    labeled_unique = (
        labeled[["src", "dst", "bucket"]]
        .drop_duplicates()
        .sort_values(["bucket", "src", "dst"])
    )
    OUT_LABELED.parent.mkdir(parents=True, exist_ok=True)
    labeled_unique.to_csv(OUT_LABELED, index=False)
    print(f"Wrote labeled cooperative pairs to {OUT_LABELED}")

    # 2) Top 100 pairs by model score (regardless of label)
    top = (
        edges.sort_values("score", ascending=False)
        .drop_duplicates(subset=["src", "dst", "bucket"])
        .head(100)
    )
    top[["src", "dst", "bucket", "label", "score"]].to_csv(OUT_TOP, index=False)
    print(f"Wrote top-100 scored pairs to {OUT_TOP}")


if __name__ == "__main__":
    main()

