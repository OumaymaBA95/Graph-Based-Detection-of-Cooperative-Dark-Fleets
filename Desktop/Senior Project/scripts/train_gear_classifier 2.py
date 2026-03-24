#!/usr/bin/env python3
"""
Train a simple movement-based gear classifier from anonymized AIS training data.

This uses per-point anonymized tracks labeled implicitly by file:
- Anonymized AIS training data/trawlers.csv
- Anonymized AIS training data/purse_seines.csv
- Anonymized AIS training data/fixed_gear.csv
- Anonymized AIS training data/pole_and_line.csv
- Anonymized AIS training data/trollers.csv
- Anonymized AIS training data/unknown.csv

Each CSV has columns like:
  mmsi, timestamp, distance_from_shore, distance_from_port,
  speed, course, lat, lon, is_fishing, source

We train a RandomForest classifier to predict gear_label from
simple movement features (distance_from_shore, distance_from_port,
speed, course). This is *not* linked to your real MMSIs; it is
intended as a proof of concept and for future application once
non-anonymized metadata are available.

Usage (from project root):
  python3 scripts/train_gear_classifier.py \\
    --data-root "/Users/momoba/Desktop/Senior Project" \\
    --max-rows-per-file 200000 \\
    --out-model artifacts/gear_classifier.joblib \\
    --out-report artifacts/gear_classifier_report.txt
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from joblib import dump


GEAR_FILES = {
    "trawler": "Anonymized AIS training data/trawlers.csv",
    "purse_seine": "Anonymized AIS training data/purse_seines.csv",
    "fixed_gear": "Anonymized AIS training data/fixed_gear.csv",
    "pole_and_line": "Anonymized AIS training data/pole_and_line.csv",
    "troller": "Anonymized AIS training data/trollers.csv",
    "unknown": "Anonymized AIS training data/unknown.csv",
}


def load_samples(root: Path, max_rows_per_file: int) -> pd.DataFrame:
    rows = []
    for gear, rel_path in GEAR_FILES.items():
        path = root / rel_path
        if not path.exists():
            print(f"[warn] Missing file for {gear}: {path}")
            continue
        print(f"[info] Loading {gear} from {path}")
        df = pd.read_csv(path, nrows=max_rows_per_file)
        if not {"distance_from_shore", "distance_from_port", "speed", "course"}.issubset(
            df.columns
        ):
            print(f"[warn] Skipping {path} (missing required columns).")
            continue
        df["gear_label"] = gear
        rows.append(
            df[
                [
                    "distance_from_shore",
                    "distance_from_port",
                    "speed",
                    "course",
                    "gear_label",
                ]
            ]
        )
    if not rows:
        raise ValueError("No training rows loaded from any gear files.")
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description="Train movement-based gear classifier.")
    ap.add_argument(
        "--data-root",
        default=".",
        help="Root directory that contains 'Anonymized AIS training data/'",
    )
    ap.add_argument(
        "--max-rows-per-file",
        type=int,
        default=200_000,
        help="Maximum rows to sample from each gear CSV (to keep training manageable).",
    )
    ap.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use for test split.",
    )
    ap.add_argument(
        "--out-model",
        default="artifacts/gear_classifier.joblib",
        help="Path to save trained model.",
    )
    ap.add_argument(
        "--out-report",
        default="artifacts/gear_classifier_report.txt",
        help="Path to save text classification report.",
    )
    args = ap.parse_args()

    root = Path(args.data_root)
    df = load_samples(root, args.max_rows_per_file)

    # Clean up and split
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    X = df[["distance_from_shore", "distance_from_port", "speed", "course"]]
    y = df["gear_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=42,
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        n_jobs=-1,
        random_state=42,
    )
    print("[info] Fitting RandomForest...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, digits=3)
    print(report)

    out_model_path = Path(args.out_model)
    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(clf, out_model_path)
    print(f"[info] Saved model to {out_model_path}")

    out_report_path = Path(args.out_report)
    out_report_path.parent.mkdir(parents=True, exist_ok=True)
    out_report_path.write_text(report)
    print(f"[info] Saved report to {out_report_path}")


if __name__ == "__main__":
    main()

