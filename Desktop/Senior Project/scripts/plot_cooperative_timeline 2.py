#!/usr/bin/env python3
"""
Single-figure daily cooperation timeline for MMSI pairs.

- X-axis: date
- Y-axis: MMSI pair with country + gear (from enrichment when available)
- Color: cooperative (label=1) vs non-cooperative (label=0)
"""
import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import matplotlib.pyplot as plt
import pandas as pd

from vessel_pair_labels import format_pair_label, load_enrichment_first_row_per_pair


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily", default="artifacts/cooperative_timeline_daily.csv")
    ap.add_argument("--enrichment", default="artifacts/cooperative_pairs_with_flag_gear.csv")
    ap.add_argument("--out", default="artifacts/plots/tgcn_daily_cooperation_timeline.png")
    ap.add_argument("--top-n", type=int, default=15)
    args = ap.parse_args()

    daily_path = Path(args.daily)
    if not daily_path.exists():
        raise SystemExit(f"File not found: {daily_path}")

    enrich = load_enrichment_first_row_per_pair(args.enrichment)

    df = pd.read_csv(daily_path)
    df["date"] = pd.to_datetime(df["date"])
    df["pair"] = df["src"].astype(str) + "–" + df["dst"].astype(str)

    totals = (
        df[df["label"] == 1]
        .groupby("pair")["date"]
        .nunique()
        .reset_index(name="coop_days")
        .sort_values("coop_days", ascending=False)
    )
    top_pairs = totals.head(args.top_n)["pair"].tolist()
    sub = df[df["pair"].isin(top_pairs)].copy()
    sub = sub.sort_values(["pair", "date"])

    def pair_key(p: str) -> tuple[int, int]:
        a, b = p.split("–", 1)
        return int(a), int(b)

    label_map = {p: format_pair_label(*pair_key(p), enrich, multiline=False) for p in top_pairs}
    sub["pair_lbl"] = sub["pair"].map(label_map)

    sub["pair_lbl"] = pd.Categorical(sub["pair_lbl"], categories=[label_map[p] for p in top_pairs], ordered=True)

    plt.figure(figsize=(11, max(6, len(top_pairs) * 0.45)))

    colors = {1: "tab:green", 0: "tab:red"}
    labels = {1: "cooperative (label=1)", 0: "non-cooperative (label=0)"}

    for lab in (1, 0):
        chunk = sub[sub["label"] == lab]
        if chunk.empty:
            continue
        plt.scatter(
            chunk["date"],
            chunk["pair_lbl"],
            c=colors[lab],
            label=labels[lab],
            s=20,
            alpha=0.8,
        )

    plt.xlabel("Date")
    plt.ylabel("MMSI pair (country · gear | country · gear)")
    plt.title("Daily cooperation timeline for top MMSI pairs")
    plt.grid(axis="x", alpha=0.3)
    plt.xticks(rotation=45)
    plt.yticks(fontsize=7)
    plt.legend(loc="upper right")
    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
