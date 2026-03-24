#!/usr/bin/env python3
"""
Plot per-bucket TGCN performance and edge counts from
artifacts/tgcn_time_temporal_nodes_full_with_social.json.
"""
from pathlib import Path
import json

import matplotlib.pyplot as plt


def main() -> None:
    report_path = Path("artifacts/tgcn_time_temporal_nodes_full_with_social.json")
    if not report_path.exists():
        raise SystemExit(f"Report not found: {report_path}")

    with report_path.open() as f:
        report = json.load(f)

    # We only use the single fold / seed this experiment produced.
    buckets = report["per_fold"][0]["per_seed"][0]["buckets"]

    x = [b["bucket"] for b in buckets]
    roc = [b["roc_auc"] for b in buckets]
    ap = [b["average_precision"] for b in buckets]
    pos = [b["pos_edges"] for b in buckets]
    neg = [b["neg_edges"] for b in buckets]

    # Plot AUC / AP over time buckets
    plt.figure(figsize=(10, 5))
    plt.plot(x, roc, marker="o", label="ROC AUC")
    plt.plot(x, ap, marker="s", label="Average precision")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Score")
    plt.title("TGCN performance by time bucket")
    plt.legend()
    plt.tight_layout()

    # Plot positive / negative edge counts per bucket
    plt.figure(figsize=(10, 5))
    plt.bar(x, pos, label="Positive (cooperative) edges")
    plt.bar(x, neg, bottom=pos, alpha=0.7, label="Negative (non-cooperative) edges")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Edge count")
    plt.title("Labeled cooperative vs non-cooperative edges by bucket")
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()

