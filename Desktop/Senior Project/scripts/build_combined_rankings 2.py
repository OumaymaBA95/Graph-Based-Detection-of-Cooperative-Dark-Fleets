#!/usr/bin/env python3
"""
Build combined rankings for candidate pairs using model score + overlap evidence.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def zscore(series: pd.Series) -> pd.Series:
    if series.std(ddof=0) == 0:
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - series.mean()) / series.std(ddof=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores', default='artifacts/tgcn_candidate_scores_fullcoverage_top500.csv')
    ap.add_argument('--overlap', default='artifacts/top100_overlap_summary_daily_full_25km_w1.csv')
    ap.add_argument('--top-k', type=int, default=100)
    ap.add_argument('--min-id', type=int, default=100_000_000, help='Filter out implausible MMSI-like IDs')
    ap.add_argument('--out-full', default='artifacts/combined_ranked_top100.csv')
    ap.add_argument('--out-top', default='artifacts/combined_ranked_top25.csv')
    args = ap.parse_args()

    scores = pd.read_csv(args.scores).head(args.top_k)
    overlap = pd.read_csv(args.overlap)

    for col in ['src', 'dst']:
        if col in scores.columns:
            scores[col] = pd.to_numeric(scores[col], errors='coerce').astype('Int64')
        if col in overlap.columns:
            overlap[col] = pd.to_numeric(overlap[col], errors='coerce').astype('Int64')
    scores = scores.dropna(subset=['src', 'dst'])
    overlap = overlap.dropna(subset=['src', 'dst'])
    scores = scores[(scores['src'] >= args.min_id) & (scores['dst'] >= args.min_id)]
    overlap = overlap[(overlap['src'] >= args.min_id) & (overlap['dst'] >= args.min_id)]

    merged = scores.merge(overlap, on=['src', 'dst'], how='left')
    merged['overlap_days'] = merged['overlap_days'].fillna(0)
    merged['days_within_km'] = merged['days_within_km'].fillna(0)

    merged['score_z'] = zscore(merged['score'])
    merged['overlap_days_z'] = zscore(merged['overlap_days'])
    merged['close_hit'] = (merged['days_within_km'] > 0).astype(int)

    merged['combined_score'] = (
        merged['score_z']
        + 0.5 * merged['overlap_days_z']
        + 2.0 * merged['close_hit']
    )

    merged = merged.sort_values('combined_score', ascending=False)

    out_full = Path(args.out_full)
    out_full.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_full, index=False)

    out_top = Path(args.out_top)
    merged.head(25).to_csv(out_top, index=False)

    print(f'Wrote {out_full}')
    print(f'Wrote {out_top}')


if __name__ == '__main__':
    main()
