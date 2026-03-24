#!/usr/bin/env python3
"""
Plot per-test-bucket ROC AUC and AP from a run_tgcn_time_multiseed JSON report.

Example:
  python3 scripts/plot_tgcn_bucket_metrics.py \\
    --report artifacts/tgcn_social_maxb1450_ep3.json \\
    --out-dir artifacts/plots
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="JSON from run_tgcn_time_multiseed.py")
    ap.add_argument("--out-dir", default="artifacts/plots", help="Directory for PNG outputs")
    args = ap.parse_args()

    path = Path(args.report)
    with open(path, encoding="utf-8") as f:
        rep = json.load(f)

    buckets = rep["per_fold"][0]["per_seed"][0]["buckets"]
    aucs = np.array([b["roc_auc"] for b in buckets], dtype=float)
    aps = np.array([b["average_precision"] for b in buckets], dtype=float)
    n = len(aucs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem

    # Histograms
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(aucs, bins=40, color="steelblue", edgecolor="white", alpha=0.9)
    axes[0].axvline(float(rep["overall"]["roc_auc_mean"]), color="darkred", linestyle="--", label=f"Mean AUC = {rep['overall']['roc_auc_mean']:.3f}")
    axes[0].set_xlabel("Per-bucket ROC AUC")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Distribution of bucket ROC AUC (n={n})")
    axes[0].legend(fontsize=8)

    axes[1].hist(aps, bins=40, color="seagreen", edgecolor="white", alpha=0.9)
    axes[1].axvline(float(rep["overall"]["average_precision_mean"]), color="darkred", linestyle="--", label=f"Mean AP = {rep['overall']['average_precision_mean']:.3f}")
    axes[1].set_xlabel("Per-bucket Average Precision")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Distribution of bucket AP (n={n})")
    axes[1].legend(fontsize=8)

    fig.suptitle(f"TGCN test-bucket metrics — {stem}", fontsize=11)
    fig.tight_layout()
    hist_path = out_dir / f"{stem}_bucket_metrics_hist.png"
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {hist_path}")

    # Time-indexed line plot (bucket order = temporal order in evaluation)
    fig2, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(np.arange(n), aucs, lw=0.8, alpha=0.85, label="ROC AUC", color="steelblue")
    ax.axhline(float(rep["overall"]["roc_auc_mean"]), color="darkred", linestyle="--", alpha=0.8)
    ax.set_xlabel("Test bucket index (time order)")
    ax.set_ylabel("ROC AUC")
    ax.set_title(f"Per-bucket ROC AUC over held-out test buckets (n={n}) — {stem}")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    ts_path = out_dir / f"{stem}_bucket_auc_timeseries.png"
    fig2.savefig(ts_path, dpi=150)
    plt.close(fig2)
    print(f"Wrote {ts_path}")


if __name__ == "__main__":
    main()
