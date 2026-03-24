#!/usr/bin/env python3
"""
Clean score plot for the top-100 TGCN-scored MMSI pairs.

- One point per (src,dst,bucket)
- Y-axis: MMSI pair
- X-axis: score
- Color: label (cooperative vs negative)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

IN_PATH = Path("artifacts/cooperative_pairs_top100_by_score.csv")


def main() -> None:
    if not IN_PATH.exists():
        raise SystemExit(f"File not found: {IN_PATH}")

    df = pd.read_csv(IN_PATH)

    # Build a nice pair label
    df["pair"] = df["src"].astype(str) + "–" + df["dst"].astype(str)

    # Sort by score and keep order of pairs by max score
    pair_order = (
        df.groupby("pair")["score"]
        .max()
        .sort_values(ascending=False)
        .index.tolist()
    )
    df["pair"] = pd.Categorical(df["pair"], categories=pair_order, ordered=True)
    df = df.sort_values(["pair", "score"], ascending=[True, False])

    plt.figure(figsize=(10, max(6, len(pair_order) * 0.25)))

    colors = {1: "tab:green", 0: "tab:red"}
    labels = {1: "labeled cooperative (label=1)", 0: "negative sample (label=0)"}

    for lab in (1, 0):
        sub = df[df["label"] == lab]
        if sub.empty:
            continue
        plt.scatter(
            sub["score"],
            sub["pair"],
            c=colors[lab],
            label=labels[lab],
            s=30,
            alpha=0.8,
            edgecolor="k",
            linewidth=0.3,
        )

    plt.xlabel("TGCN score")
    plt.ylabel("MMSI pair")
    plt.title("Top-100 edge scores by TGCN (cooperative vs negative)")
    plt.grid(axis="x", alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

