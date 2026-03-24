#!/usr/bin/env python3
"""
Run a structured set of TGCN experiments to improve / compare ROC AUC:

  - Proximity-only vs full (proximity + social) ablation
  - Hard negative sampling vs uniform negatives
  - Optional GConvGRU instead of TGCN
  - Optional vessel-day features (requires data/features_by_year/*/vessel_day_features.parquet)

Writes per-run JSON/CSV under artifacts/ and a summary table:
  artifacts/tgcn_improvement_suite_summary.csv

Usage (from repo root, use PyG env python — see README):

  export PYTHONPATH=scripts
  export KMP_DUPLICATE_LIB_OK=TRUE
  python3 scripts/run_tgcn_improvement_suite.py

Quick iteration (small buckets, fewer test buckets — NOT comparable to full thesis numbers):

  python3 scripts/run_tgcn_improvement_suite.py --quick

Run a subset:

  python3 scripts/run_tgcn_improvement_suite.py --only 01,02,03

Dry-run (print commands only):

  python3 scripts/run_tgcn_improvement_suite.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# Default feature columns for vessel-day branch (numeric; must exist in parquet)
DEFAULT_VESSEL_FEATURE_COLS = (
    "speed_mean,speed_std,lat_mean,lon_mean,sst_mean"
)


def _build_runs(
    train_buckets: int,
    epochs: int,
    max_test_buckets: int | None,
    seeds: str,
    lr: float,
    embedding_dim: int,
) -> list[tuple[str, str, list[str]]]:
    """(slug, edges_path, extra_cli_args)."""
    base = [
        "--epochs",
        str(epochs),
        "--embedding-dim",
        str(embedding_dim),
        "--lr",
        str(lr),
        "--seeds",
        seeds,
        "--use-temporal-node-features",
        "--max-train-buckets",
        str(train_buckets),
    ]
    if max_test_buckets is not None:
        base += ["--max-test-buckets", str(max_test_buckets)]

    out: list[tuple[str, str, list[str]]] = []

    out.append(
        (
            "01_proximity_temporal",
            "artifacts/edges_2012_2019_full.parquet",
            base.copy(),
        )
    )
    out.append(
        (
            "02_social_temporal",
            "artifacts/edges_full_with_social.parquet",
            base.copy(),
        )
    )
    out.append(
        (
            "03_social_hardneg",
            "artifacts/edges_full_with_social.parquet",
            base + ["--use-hard-negatives"],
        )
    )
    out.append(
        (
            "04_proximity_hardneg",
            "artifacts/edges_2012_2019_full.parquet",
            base + ["--use-hard-negatives"],
        )
    )
    out.append(
        (
            "05_social_gconvgru",
            "artifacts/edges_full_with_social.parquet",
            base + ["--model", "gconvgru"],
        )
    )
    out.append(
        (
            "06_social_vesselday_temporal",
            "artifacts/edges_full_with_social.parquet",
            base
            + [
                "--use-vessel-day-features",
                "--feature-cols",
                DEFAULT_VESSEL_FEATURE_COLS,
            ],
        )
    )
    # Slightly wider embedding + lower LR (often helps optimization)
    out.append(
        (
            "07_social_tgcn_emb48_lr5e4",
            "artifacts/edges_full_with_social.parquet",
            _replace_lr_emb(
                base,
                lr=0.0005,
                emb=48,
            ),
        )
    )
    return out


def _replace_lr_emb(base: list[str], lr: float, emb: int) -> list[str]:
    out = base.copy()
    out[out.index("--embedding-dim") + 1] = str(emb)
    out[out.index("--lr") + 1] = str(lr)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--python",
        default=os.environ.get("TG_PYTHON", sys.executable),
        help="Python with PyG/torch_sparse (default: current interpreter or $TG_PYTHON)",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Small train/test buckets for fast smoke (metrics NOT comparable to full runs)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only",
    )
    ap.add_argument(
        "--only",
        default="",
        help="Comma-separated slug prefixes to run, e.g. 01,02 or 01_proximity",
    )
    ap.add_argument("--train-buckets", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-test-buckets", type=int, default=None, help="Cap test buckets (default: all)")
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--embedding-dim", type=int, default=32)
    args = ap.parse_args()

    if args.quick:
        train_buckets = 120
        epochs = 2
        max_test = 50
        emb = 32
    else:
        train_buckets = args.train_buckets
        epochs = args.epochs
        max_test = args.max_test_buckets
        emb = args.embedding_dim

    runs = _build_runs(
        train_buckets=train_buckets,
        epochs=epochs,
        max_test_buckets=max_test,
        seeds=args.seeds,
        lr=args.lr,
        embedding_dim=emb,
    )

    only = [x.strip() for x in args.only.split(",") if x.strip()]
    if only:

        def _slug_matches(slug: str, patterns: list[str]) -> bool:
            for o in patterns:
                if slug == o or slug.startswith(o + "_") or slug.startswith(o):
                    return True
            return False

        runs = [r for r in runs if _slug_matches(r[0], only)]

    script = ROOT / "scripts" / "run_tgcn_time_multiseed.py"
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(ROOT / "scripts"))
    env["PYTHONPATH"] = str(ROOT / "scripts") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    summary_rows: list[dict[str, Any]] = []
    out_csv = ROOT / "artifacts" / "tgcn_improvement_suite_summary.csv"

    for slug, edges_rel, extra in runs:
        edges_path = ROOT / edges_rel
        if not edges_path.exists():
            print(f"SKIP {slug}: missing {edges_path}", file=sys.stderr)
            summary_rows.append(
                {
                    "slug": slug,
                    "edges": edges_rel,
                    "status": "skipped_missing_edges",
                    "roc_auc_mean": "",
                    "average_precision_mean": "",
                    "buckets_evaluated": "",
                    "report": "",
                }
            )
            continue

        report_path = ROOT / "artifacts" / f"tgcn_suite_{slug}.json"
        csv_path = ROOT / "artifacts" / f"tgcn_suite_{slug}.csv"
        cmd = [
            args.python,
            str(script),
            "--edges",
            str(edges_path.relative_to(ROOT)),
            *extra,
            "--out-report",
            str(report_path.relative_to(ROOT)),
            "--out-csv",
            str(csv_path.relative_to(ROOT)),
        ]

        print("RUN", slug, flush=True)
        if args.dry_run:
            print(" ", " ".join(cmd))
            continue

        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(proc.stderr[-4000:] if proc.stderr else "", file=sys.stderr)
            summary_rows.append(
                {
                    "slug": slug,
                    "edges": edges_rel,
                    "status": f"error_{proc.returncode}",
                    "roc_auc_mean": "",
                    "average_precision_mean": "",
                    "buckets_evaluated": "",
                    "report": str(report_path.relative_to(ROOT)),
                }
            )
            continue

        roc = ap_ = buckets = ""
        if report_path.exists():
            with open(report_path, encoding="utf-8") as f:
                rep = json.load(f)
            ov = rep.get("overall", {})
            roc = ov.get("roc_auc_mean", "")
            ap_ = ov.get("average_precision_mean", "")
            buckets = ov.get("buckets_evaluated", "")
        summary_rows.append(
            {
                "slug": slug,
                "edges": edges_rel,
                "status": "ok",
                "roc_auc_mean": roc,
                "average_precision_mean": ap_,
                "buckets_evaluated": buckets,
                "report": str(report_path.relative_to(ROOT)),
            }
        )
        print(f"  -> ROC AUC={roc} AP={ap_} buckets={buckets}", flush=True)

    if args.dry_run:
        return

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "slug",
                "edges",
                "status",
                "roc_auc_mean",
                "average_precision_mean",
                "buckets_evaluated",
                "report",
            ],
        )
        w.writeheader()
        w.writerows(summary_rows)

    print(f"Wrote {out_csv.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
