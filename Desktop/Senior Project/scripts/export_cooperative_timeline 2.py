#!/usr/bin/env python3
"""
Summarize, for each MMSI pair, on which days/months it appears as
cooperative vs non-cooperative according to TGCN outputs.

Input:
- artifacts/tgcn_time_temporal_edges.csv (src,dst,bucket,label,score)

Outputs:
- artifacts/cooperative_timeline_daily.csv
    src,dst,date,label
- artifacts/cooperative_timeline_monthly.csv
    src,dst,month,pos_days,neg_days
"""
from pathlib import Path

import pandas as pd


IN_PATH = Path("artifacts/tgcn_time_temporal_edges.csv")
OUT_DAILY = Path("artifacts/cooperative_timeline_daily.csv")
OUT_MONTHLY = Path("artifacts/cooperative_timeline_monthly.csv")


def main() -> None:
    if not IN_PATH.exists():
        raise SystemExit(f"Input edges CSV not found: {IN_PATH}")

    df = pd.read_csv(IN_PATH)
    # Normalize bucket to date
    df["bucket"] = pd.to_datetime(df["bucket"])
    df["date"] = df["bucket"].dt.date
    df["month"] = df["bucket"].dt.to_period("M").astype(str)

    # 1) Daily: one row per (src,dst,date,label)
    daily = (
        df[["src", "dst", "date", "label"]]
        .drop_duplicates()
        .sort_values(["src", "dst", "date", "label"])
    )
    OUT_DAILY.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUT_DAILY, index=False)
    print(f"Wrote daily timeline to {OUT_DAILY}")

    # 2) Monthly summary: counts of cooperative vs non-cooperative days
    # First collapse to one row per (src,dst,date,label) to avoid double-counting.
    per_day = df[["src", "dst", "date", "month", "label"]].drop_duplicates()
    grouped = (
        per_day.groupby(["src", "dst", "month", "label"])
        .size()
        .reset_index(name="days")
    )

    # Pivot label -> pos_days / neg_days
    monthly = grouped.pivot_table(
        index=["src", "dst", "month"],
        columns="label",
        values="days",
        fill_value=0,
    ).reset_index()
    monthly = monthly.rename(columns={0: "neg_days", 1: "pos_days"})
    # Ensure both columns exist
    if "neg_days" not in monthly.columns:
        monthly["neg_days"] = 0
    if "pos_days" not in monthly.columns:
        monthly["pos_days"] = 0

    monthly = monthly[["src", "dst", "month", "pos_days", "neg_days"]].sort_values(
        ["src", "dst", "month"]
    )
    monthly.to_csv(OUT_MONTHLY, index=False)
    print(f"Wrote monthly summary to {OUT_MONTHLY}")


if __name__ == "__main__":
    main()

