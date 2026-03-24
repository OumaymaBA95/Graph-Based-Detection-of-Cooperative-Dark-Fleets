#!/usr/bin/env python3
"""
Create time-series plots of monthly overlap for the 8 validated pairs.

Input:
  - artifacts/eight_pairs_overlap_by_month.csv
      Columns: src, dst, year_month, days_within_km, overlap_pairs, ...

Outputs:
  - artifacts/plots/pair_overlap_series/all_pairs_days_within_25km.png
      Bubble chart: months on x-axis, pairs on y-axis, bubble size = days.
  - artifacts/plots/pair_overlap_series/pair_<src>_<dst>_days_within_25km.png
      Individual per-pair line plots.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from vessel_pair_labels import format_pair_label, load_enrichment_first_row_per_pair

_PUBLICATION_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica Neue", "Arial", "sans-serif"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.titleweight": "bold",
    "axes.labelweight": "medium",
    "axes.facecolor": "#fafafa",
    "figure.facecolor": "#ffffff",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.labelcolor": "#1a1a2e",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "grid.color": "#dddddd",
    "grid.alpha": 0.6,
    "grid.linewidth": 0.4,
    "grid.linestyle": "-",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
}

_PAIR_COLORS = [
    "#2176ae", "#e85d04", "#57cc99", "#c9184a",
    "#7b2cbf", "#e09f3e", "#219ebc", "#6d6875",
]


def _pair_country_label(src: int, dst: int, enrich) -> str:
    """Country pair label, e.g. 'CHN \u2194 CHN' or 'CHN \u2194 MID111'."""
    key = (src, dst) if src <= dst else (dst, src)
    row = enrich.get(key) if enrich else None
    from vessel_pair_labels import vessel_country, country_for_mmsi
    if row:
        c1 = vessel_country(src, row.get("src_flag_cell", ""))
        c2 = vessel_country(dst, row.get("dst_flag_cell", ""))
    else:
        c1 = country_for_mmsi(src)
        c2 = country_for_mmsi(dst)
    if c1 == c2:
        return c1
    return f"{c1} \u2194 {c2}"


def main():
    ap = argparse.ArgumentParser(description="Plot monthly overlap time-series for validated pairs.")
    ap.add_argument(
        "--overlap-csv",
        default="artifacts/eight_pairs_overlap_by_month.csv",
        help="CSV from overlap_by_month_8pairs.py",
    )
    ap.add_argument(
        "--out-dir",
        default="artifacts/plots/pair_overlap_series",
        help="Directory for per-pair time-series plots",
    )
    ap.add_argument(
        "--enrichment",
        default="artifacts/cooperative_pairs_with_flag_gear.csv",
        help="CSV for country/gear labels (optional)",
    )
    args = ap.parse_args()

    enrich = load_enrichment_first_row_per_pair(args.enrichment)

    df = pd.read_csv(args.overlap_csv)
    if df.empty:
        raise ValueError("No rows in overlap CSV.")

    df["year_month"] = pd.PeriodIndex(df["year_month"], freq="M")
    df = df.sort_values(["src", "dst", "year_month"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_months = sorted(df["year_month"].unique())
    month_dates = np.array([p.to_timestamp() for p in all_months])

    pairs = list(df.groupby(["src", "dst"]).groups.keys())
    pair_totals = {
        (s, d): df[(df["src"] == s) & (df["dst"] == d)]["days_within_km"].sum()
        for s, d in pairs
    }
    pairs = sorted(pairs, key=lambda p: pair_totals[p], reverse=True)
    n_pairs = len(pairs)

    # --- Bubble chart ---
    with plt.rc_context(_PUBLICATION_RC):
        fig, ax = plt.subplots(figsize=(14, 7.5), constrained_layout=True)

        global_max = df["days_within_km"].max()
        size_scale = 650.0 / max(global_max, 1)

        y_labels = []
        for i, (src, dst) in enumerate(pairs):
            g = df[(df["src"] == src) & (df["dst"] == dst)].set_index("year_month")
            y_vals = np.array([
                g.loc[m, "days_within_km"] if m in g.index else 0
                for m in all_months
            ], dtype=float)

            mask = y_vals > 0
            active_months = int(mask.sum())
            total = int(pair_totals[(src, dst)])
            country = _pair_country_label(int(src), int(dst), enrich)
            pair_num = i + 1
            y_labels.append(
                f"Pair {pair_num} ({country})\n"
                f"{total} days across {active_months} months"
            )

            if not mask.any():
                continue

            x_active = month_dates[mask]
            s_active = y_vals[mask]
            color = _PAIR_COLORS[i % len(_PAIR_COLORS)]

            ax.scatter(
                x_active, np.full_like(x_active, i, dtype=float),
                s=s_active * size_scale,
                c=color, alpha=0.78, edgecolors="white", linewidths=0.6,
                zorder=3,
            )

            for xd, sv in zip(x_active, s_active):
                if sv >= 3:
                    ax.text(
                        xd, i, f"{int(sv)}",
                        ha="center", va="center", fontsize=7,
                        fontweight="bold", color="white", zorder=4,
                    )

        ax.set_yticks(range(n_pairs))
        ax.set_yticklabels(y_labels, fontsize=9, linespacing=1.3)
        ax.set_ylim(n_pairs - 0.5, -0.5)

        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
        ax.tick_params(axis="x", rotation=0)

        ax.grid(axis="both", zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        for i in range(n_pairs):
            if i % 2 == 1:
                ax.axhspan(i - 0.5, i + 0.5, color="#f0f0f0", zorder=0)

        legend_sizes = [1, 5, 10, 20]
        legend_handles = []
        for ls in legend_sizes:
            if ls <= global_max:
                legend_handles.append(
                    mlines.Line2D(
                        [], [], marker="o", linestyle="None",
                        markersize=np.sqrt(ls * size_scale) / 2,
                        markerfacecolor="#888888", markeredgecolor="white",
                        markeredgewidth=0.5, label=f"{ls} days",
                    )
                )
        ax.legend(
            handles=legend_handles, title="Days within 25 km",
            title_fontsize=9, fontsize=8.5,
            loc="upper right", framealpha=0.94, edgecolor="#aaaaaa",
            handletextpad=1.0, borderpad=0.8,
        )

        ax.set_title(
            "Monthly close approaches for eight validated pairs (2013\u20132019)",
            fontsize=13, fontweight="bold", pad=10,
        )
        ax.set_xlabel("")

        combined_out = out_dir / "all_pairs_days_within_25km.png"
        fig.savefig(combined_out, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # --- Per-pair plots (simple line charts) ---
    for (src, dst), g in df.groupby(["src", "dst"]):
        months = g["year_month"].astype(str).tolist()
        x = range(len(months))
        y = g["days_within_km"].tolist()

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x, y, marker="o", linestyle="-", color="#2176ae")
        ax.set_xticks(x)
        ax.set_xticklabels(months, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Days within 25 km")
        ax.set_xlabel("Year-month")
        ax.set_title(
            f"Monthly overlap: {format_pair_label(int(src), int(dst), enrich, multiline=False)}",
            fontsize=9,
        )
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        out_path = out_dir / f"pair_{src}_{dst}_days_within_25km.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Wrote per-pair plots to {out_dir}")
    print(f"Wrote combined plot to {combined_out}")


if __name__ == "__main__":
    main()

