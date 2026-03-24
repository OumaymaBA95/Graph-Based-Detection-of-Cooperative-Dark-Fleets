#!/usr/bin/env python3
"""
Aggregate cooperation summary: stacked bar chart by day + gear type,
with a country breakdown side panel.

Uses ALL cooperative pairs (not just top-20), showing the full picture
of the model's short-window predictions.
"""
import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from vessel_pair_labels import load_enrichment_first_row_per_pair

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
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.7,
    "axes.labelcolor": "#1a1a2e",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
}

_GEAR_GROUPS = {
    "Trawlers": ["trawlers | trawlers"],
    "Line fishing": [
        "fishing | fishing", "fishing | trawlers", "trawlers | fishing",
        "fishing | fixed gear", "fixed gear | fishing",
        "fishing | set gillnets", "set gillnets | fishing",
        "fishing | \u2014", "\u2014 | fishing",
    ],
    "Set gear": [
        "set gillnets | set gillnets",
        "set longlines | set longlines",
        "set longlines | fishing", "fishing | set longlines",
        "set longlines | trawlers", "trawlers | set longlines",
        "set longlines | pole and line",
        "fixed gear | set gillnets", "set gillnets | fixed gear",
    ],
    "Other / mixed": [],
}

_GEAR_COLORS = {
    "Trawlers": "#1565c0",
    "Line fishing": "#e85d04",
    "Set gear": "#2e7d32",
    "Other / mixed": "#8e8e8e",
}


def _classify_gear(gear_str: str) -> str:
    for group, members in _GEAR_GROUPS.items():
        if group == "Other / mixed":
            continue
        if gear_str in members:
            return group
    return "Other / mixed"


def _gear_combo(src, dst, enrich) -> str:
    from vessel_pair_labels import vessel_country, country_for_mmsi
    key = (src, dst) if src <= dst else (dst, src)
    row = enrich.get(key)
    if row:
        g1 = (row.get("src_gear", "") or "\u2014").replace("_", " ")
        g2 = (row.get("dst_gear", "") or "\u2014").replace("_", " ")
    else:
        g1 = g2 = "\u2014"
    return f"{g1} | {g2}"


def _country_pair(src, dst, enrich) -> str:
    from vessel_pair_labels import vessel_country, country_for_mmsi
    key = (src, dst) if src <= dst else (dst, src)
    row = enrich.get(key)
    if row:
        c1 = vessel_country(src, row.get("src_flag_cell", ""))
        c2 = vessel_country(dst, row.get("dst_flag_cell", ""))
    else:
        c1 = country_for_mmsi(src)
        c2 = country_for_mmsi(dst)
    if c1 == c2:
        return c1
    return f"{c1}\u2013{c2}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily", default="artifacts/cooperative_timeline_daily.csv")
    ap.add_argument(
        "--enrichment",
        default="artifacts/cooperative_pairs_with_flag_gear.csv",
    )
    ap.add_argument("--out", default="artifacts/plots/tgcn_daily_cooperation_heatmap.png")
    args = ap.parse_args()

    daily_path = Path(args.daily)
    if not daily_path.exists():
        raise SystemExit(f"File not found: {daily_path}")

    enrich = load_enrichment_first_row_per_pair(args.enrichment)
    df = pd.read_csv(daily_path)
    df["date"] = pd.to_datetime(df["date"])

    coop = df[df["label"] == 1].copy()
    total_obs = len(df)
    total_coop = len(coop)
    total_pairs = df.groupby(["src", "dst"]).ngroups
    coop_pairs = coop.groupby(["src", "dst"]).ngroups

    coop["gear"] = coop.apply(
        lambda r: _gear_combo(int(r["src"]), int(r["dst"]), enrich), axis=1
    )
    coop["gear_group"] = coop["gear"].apply(_classify_gear)
    coop["country"] = coop.apply(
        lambda r: _country_pair(int(r["src"]), int(r["dst"]), enrich), axis=1
    )

    dates = sorted(coop["date"].unique())
    date_labels = [d.strftime("%a\n%b %d") for d in dates]
    date_range = (
        f"{min(dates).strftime('%b %d')}\u2013"
        f"{max(dates).strftime('%b %d, %Y')}"
    )

    gear_order = ["Trawlers", "Line fishing", "Set gear", "Other / mixed"]
    gear_day_counts = {}
    for gg in gear_order:
        counts = []
        for d in dates:
            n = len(coop[(coop["date"] == d) & (coop["gear_group"] == gg)])
            counts.append(n)
        gear_day_counts[gg] = np.array(counts)

    country_counts = coop["country"].value_counts()

    with plt.rc_context(_PUBLICATION_RC):
        fig = plt.figure(figsize=(13, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1.2], wspace=0.35, figure=fig)

        # --- Left: stacked bar chart by day + gear group ---
        ax_left = fig.add_subplot(gs[0])
        x = np.arange(len(dates))
        bar_w = 0.55
        bottom = np.zeros(len(dates))

        for gg in gear_order:
            vals = gear_day_counts[gg]
            if vals.sum() == 0:
                continue
            ax_left.bar(
                x, vals, bar_w, bottom=bottom,
                color=_GEAR_COLORS[gg], edgecolor="white", linewidth=0.6,
                label=gg, zorder=3,
            )
            for j in range(len(dates)):
                if vals[j] > 0:
                    mid_y = bottom[j] + vals[j] / 2
                    ax_left.text(
                        j, mid_y, str(int(vals[j])),
                        ha="center", va="center", fontsize=9,
                        fontweight="bold", color="white", zorder=4,
                    )
            bottom += vals

        for j in range(len(dates)):
            total = int(bottom[j])
            ax_left.text(
                j, total + 0.3, str(total),
                ha="center", va="bottom", fontsize=11,
                fontweight="bold", color="#333333",
            )

        ax_left.set_xticks(x)
        ax_left.set_xticklabels(date_labels, fontsize=11)
        ax_left.set_ylabel("Cooperative pairs")
        ax_left.set_ylim(0, bottom.max() + 2.5)
        ax_left.set_xlim(-0.5, len(dates) - 0.5)

        ax_left.grid(axis="y", linewidth=0.3, alpha=0.4, zorder=0)
        for spine in ("top", "right"):
            ax_left.spines[spine].set_visible(False)

        ax_left.legend(
            loc="upper left", fontsize=9, framealpha=0.94,
            edgecolor="#aaaaaa", title="Gear type",
            title_fontsize=9,
        )

        ax_left.set_title(
            f"Cooperative pairs per day ({date_range})\n"
            f"{coop_pairs} cooperative pairs out of {total_pairs} observed",
            fontsize=13, fontweight="bold", pad=12,
        )

        # --- Right: country breakdown (horizontal bar chart) ---
        ax_right = fig.add_subplot(gs[1])

        top_countries = country_counts.head(8)
        other_count = country_counts.iloc[8:].sum() if len(country_counts) > 8 else 0
        if other_count > 0:
            top_countries = pd.concat([
                top_countries,
                pd.Series({"Other": other_count}),
            ])

        c_labels = top_countries.index.tolist()
        c_vals = top_countries.values
        c_y = np.arange(len(c_labels))

        c_colors = []
        for lbl in c_labels:
            if lbl == "CHN":
                c_colors.append("#c62828")
            elif lbl == "Other":
                c_colors.append("#8e8e8e")
            else:
                c_colors.append("#5c6bc0")

        ax_right.barh(
            c_y, c_vals, height=0.55,
            color=c_colors, edgecolor="white", linewidth=0.5,
            zorder=3,
        )
        for i, v in enumerate(c_vals):
            ax_right.text(
                v + 0.3, i, str(int(v)),
                va="center", ha="left", fontsize=9,
                fontweight="bold", color="#333333",
            )

        ax_right.set_yticks(c_y)
        ax_right.set_yticklabels(c_labels, fontsize=10)
        ax_right.invert_yaxis()
        ax_right.set_xlabel("Cooperative pair-days")
        ax_right.set_xlim(0, c_vals.max() + 4)
        ax_right.grid(axis="x", linewidth=0.3, alpha=0.4, zorder=0)
        for spine in ("top", "right"):
            ax_right.spines[spine].set_visible(False)

        ax_right.set_title(
            "By flag state",
            fontsize=12, fontweight="bold", pad=10,
        )

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
