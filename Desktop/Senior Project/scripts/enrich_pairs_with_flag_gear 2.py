#!/usr/bin/env python3
"""
Enrich top TGCN candidate pairs with flag- and gear-like context.

Important: `data/combined_fleet_daily_full.csv` is **cell-aggregated**; the column
`mmsi_present` is a count (often 1), **not** vessel MMSI. You cannot join MMSIs on
that file directly.

This script instead:
1. Loads per-day **MMSI** tracks from `data/MMSI daily vessels /mmsi-daily-csvs-10-v3-<date>.csv`
   (columns include `mmsi`, `cell_ll_lat`, `cell_ll_lon`, `hours`).
2. Loads the matching **cell-aggregated** fleet file for the same day from
   `data/fleet-daily-csvs-100-v3-<year>/fleet-daily-csvs-100-v3-<date>.csv`
   (`flag`, `geartype`, `hours` per cell × flag × gear).
3. Joins on `(cell_ll_lat, cell_ll_lon)` and assigns **geartype** by summing MMSI `hours`
   per `geartype` (vessel time in cells attributed to each gear category).
4. Adds **Maritime Identification Digits (MID)** from each MMSI via `scripts/mmsi_mid.py`
   as `src_mid` / `dst_mid` (3-digit ITU code).

Input:
- artifacts/cooperative_pairs_top100_by_score.csv  (src, dst, bucket, label, score)

Output:
- artifacts/cooperative_pairs_with_flag_gear.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import pandas as pd

from mmsi_mid import mmsi_to_mid

PAIRS_PATH = Path("artifacts/cooperative_pairs_top100_by_score.csv")
MMSI_DAILY_ROOT = Path("data/MMSI daily vessels ")
FLEET_DAILY_ROOT = Path("data")
OUT_PATH = Path("artifacts/cooperative_pairs_with_flag_gear.csv")


def fleet_path_for_date(day: str) -> Path:
    """Single-day fleet aggregate CSV for a calendar day (YYYY-MM-DD)."""
    year = day[:4]
    return FLEET_DAILY_ROOT / f"fleet-daily-csvs-100-v3-{year}" / f"fleet-daily-csvs-100-v3-{day}.csv"


def mmsi_daily_path_for_date(day: str) -> Path:
    return MMSI_DAILY_ROOT / f"mmsi-daily-csvs-10-v3-{day}.csv"


def weighted_geartype_for_mmsi(
    mmsi: int,
    mmsi_day: pd.DataFrame,
    fleet_day: pd.DataFrame,
) -> tuple[str | float, str | float]:
    """
    Returns (best_geartype, dominant_fleet_flag_at_cells) using MMSI hours as weights.
    fleet_day must include cell_ll_lat, cell_ll_lon, flag, geartype, hours.
    """
    sub = mmsi_day[mmsi_day["mmsi"] == mmsi]
    if sub.empty:
        return float("nan"), float("nan")

    # Inner join vessel cells to fleet attribution rows
    merged = sub.merge(
        fleet_day,
        on=["cell_ll_lat", "cell_ll_lon"],
        how="inner",
        suffixes=("_mmsi", "_fleet"),
    )
    if merged.empty:
        return float("nan"), float("nan")

    # Hours from MMSI side (vessel time in cell)
    hm = merged["hours_mmsi"] if "hours_mmsi" in merged.columns else merged["hours_x"]
    merged = merged.assign(_w=hm.fillna(0))

    # Weighted geartype
    by_gear = merged.groupby("geartype", dropna=False)["_w"].sum()
    best_gear = by_gear.idxmax() if len(by_gear) else float("nan")

    # Dominant flag in merged rows (by same weights) — descriptive, not MID
    by_flag = merged.groupby("flag", dropna=False)["_w"].sum()
    best_flag = by_flag.idxmax() if len(by_flag) else float("nan")

    return best_gear, best_flag


def load_day_tables(day: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    mp = mmsi_daily_path_for_date(day)
    fp = fleet_path_for_date(day)
    if not mp.exists():
        raise FileNotFoundError(f"MMSI daily file missing: {mp}")
    if not fp.exists():
        raise FileNotFoundError(f"Fleet daily file missing: {fp}")

    mmsi_day = pd.read_csv(
        mp,
        usecols=["mmsi", "cell_ll_lat", "cell_ll_lon", "hours"],
    )
    mmsi_day["mmsi"] = pd.to_numeric(mmsi_day["mmsi"], errors="coerce").astype("Int64")
    mmsi_day = mmsi_day.dropna(subset=["mmsi"])
    mmsi_day["mmsi"] = mmsi_day["mmsi"].astype(int)

    fleet_day = pd.read_csv(
        fp,
        usecols=["cell_ll_lat", "cell_ll_lon", "flag", "geartype", "hours"],
    )
    return mmsi_day, fleet_day


def main() -> None:
    if not PAIRS_PATH.exists():
        raise SystemExit(f"Pairs file not found: {PAIRS_PATH}")

    pairs = pd.read_csv(PAIRS_PATH)
    pairs["day"] = pd.to_datetime(pairs["bucket"]).dt.strftime("%Y-%m-%d")

    # Cache per-day tables
    cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    src_gear: list[str | float] = []
    dst_gear: list[str | float] = []
    src_flag_cell: list[str | float] = []
    dst_flag_cell: list[str | float] = []
    src_mid: list[int] = []
    dst_mid: list[int] = []

    for _, row in pairs.iterrows():
        day = row["day"]
        s, d = int(row["src"]), int(row["dst"])

        src_mid.append(mmsi_to_mid(s))
        dst_mid.append(mmsi_to_mid(d))

        if day not in cache:
            cache[day] = load_day_tables(day)
        mmsi_day, fleet_day = cache[day]

        g1, f1 = weighted_geartype_for_mmsi(s, mmsi_day, fleet_day)
        g2, f2 = weighted_geartype_for_mmsi(d, mmsi_day, fleet_day)
        src_gear.append(g1)
        dst_gear.append(g2)
        src_flag_cell.append(f1)
        dst_flag_cell.append(f2)

    out = pairs.drop(columns=["day"]).copy()
    out["src_mid"] = src_mid
    out["dst_mid"] = dst_mid
    out["src_flag_cell"] = src_flag_cell
    out["dst_flag_cell"] = dst_flag_cell
    out["src_gear"] = src_gear
    out["dst_gear"] = dst_gear

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote enriched pairs to {OUT_PATH}")


if __name__ == "__main__":
    main()
