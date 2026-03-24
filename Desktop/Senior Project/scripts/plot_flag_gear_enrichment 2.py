#!/usr/bin/env python3
"""
Visualize flag/gear enrichment for top TGCN pair-rows.

Reads: artifacts/cooperative_pairs_with_flag_gear.csv (or --input)
Writes PNGs to artifacts/plots/ (or --out-dir):

  - flag_gear_src_dst_gear_heatmap.png   — counts heatmap (country | gear) × (country | gear)
  - flag_gear_same_mid_bar.png           — same MID vs different MID counts
  - flag_gear_timeline_same_mid.png      — scatter: date × row index, colored by same-MID

Requires: pandas, matplotlib
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vessel_pair_labels import vessel_country


def _fill(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().replace("", "(missing)")


def _fill_one(x) -> str:
    t = _fill(pd.Series([x])).iloc[0]
    return t if t != "(missing)" else "—"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="artifacts/cooperative_pairs_with_flag_gear.csv",
        help="Output of enrich_pairs_with_flag_gear.py",
    )
    ap.add_argument("--out-dir", default="artifacts/plots", help="Directory for PNG files")
    ap.add_argument(
        "--top-k-gears",
        type=int,
        default=14,
        help="Limit heatmap to top-K most frequent gear labels (each axis)",
    )
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"Input not found: {inp}")

    df = pd.read_csv(inp)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["src_gear"] = _fill(df["src_gear"])
    df["dst_gear"] = _fill(df["dst_gear"])
    if "src_mid" in df.columns and "dst_mid" in df.columns:
        df["same_mid"] = df["src_mid"] == df["dst_mid"]
    else:
        df["same_mid"] = False

    # Country (flag cell) + gear per vessel — heatmap axes
    df["src_lbl"] = [
        f"{vessel_country(int(r['src']), r.get('src_flag_cell', ''))} | {_fill_one(r['src_gear'])}"
        for _, r in df.iterrows()
    ]
    df["dst_lbl"] = [
        f"{vessel_country(int(r['dst']), r.get('dst_flag_cell', ''))} | {_fill_one(r['dst_gear'])}"
        for _, r in df.iterrows()
    ]

    # --- 1) Heatmap (country | gear) × (country | gear) (counts) ---
    sg = df["src_lbl"]
    dg = df["dst_lbl"]
    top_src = sg.value_counts().head(args.top_k_gears).index.tolist()
    top_dst = dg.value_counts().head(args.top_k_gears).index.tolist()
    sub = df[sg.isin(top_src) & dg.isin(top_dst)]
    if sub.empty:
        sub = df
    ct = pd.crosstab(sub["src_lbl"], sub["dst_lbl"])

    fig, ax = plt.subplots(figsize=(max(10, 0.5 * len(ct.columns)), max(7, 0.45 * len(ct.index))))
    im = ax.imshow(ct.values, aspect="auto", cmap="Blues", vmin=0)
    ax.set_xticks(np.arange(len(ct.columns)))
    ax.set_yticks(np.arange(len(ct.index)))
    ax.set_xticklabels(ct.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(ct.index, fontsize=7)
    ax.set_xlabel("Destination vessel: country | gear")
    ax.set_ylabel("Source vessel: country | gear")
    ax.set_title(
        f"Pair counts (top {args.top_k_gears} labels each axis; N={len(df)} rows)"
    )
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            v = int(ct.values[i, j])
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center", color="black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    fig.tight_layout()
    p1 = out_dir / "flag_gear_src_dst_gear_heatmap.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"Wrote {p1}")

    # --- 2) Same MID vs different ---
    if df["same_mid"].notna().any():
        counts = df["same_mid"].value_counts().sort_index()
        labels = ["Different MID", "Same MID"]
        vals = [int(counts.get(False, 0)), int(counts.get(True, 0))]
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        bars = ax2.bar(labels, vals, color=["steelblue", "darkorange"])
        ax2.set_ylabel("Number of pair-rows")
        ax2.set_title("Top TGCN pair-rows: same maritime MID block (src vs dst)")
        for b, v in zip(bars, vals):
            ax2.text(b.get_x() + b.get_width() / 2, b.get_height(), str(v), ha="center", va="bottom")
        fig2.tight_layout()
        p2 = out_dir / "flag_gear_same_mid_bar.png"
        fig2.savefig(p2, dpi=150)
        plt.close(fig2)
        print(f"Wrote {p2}")

    # --- 3) Timeline: bucket date × row index, color = same_mid ---
    if "bucket" in df.columns:
        df["bucket_ts"] = pd.to_datetime(df["bucket"])
        df = df.sort_values(["bucket_ts", "src", "dst"])
        df["row_idx"] = np.arange(len(df))
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        same = df[df["same_mid"]]
        diff = df[~df["same_mid"]]
        ax3.scatter(
            same["bucket_ts"],
            same["row_idx"],
            c="darkorange",
            s=20,
            alpha=0.85,
            label="Same MID",
        )
        ax3.scatter(
            diff["bucket_ts"],
            diff["row_idx"],
            c="steelblue",
            s=12,
            alpha=0.6,
            label="Different MID",
        )
        ax3.set_xlabel("Bucket date")
        ax3.set_ylabel("Row index (sorted by date, src, dst)")
        ax3.set_title("Enriched top pairs: timeline colored by same-MID (national block)")
        ax3.legend(loc="upper right")
        fig3.tight_layout()
        p3 = out_dir / "flag_gear_timeline_same_mid.png"
        fig3.savefig(p3, dpi=150)
        plt.close(fig3)
        print(f"Wrote {p3}")


if __name__ == "__main__":
    main()
