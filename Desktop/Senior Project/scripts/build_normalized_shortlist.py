#!/usr/bin/env python3
"""Build a normalized overlap shortlist from daily overlap summaries."""
import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores', default='artifacts/tgcn_candidate_scores_fullcoverage_top500.csv')
    ap.add_argument('--overlap', default='artifacts/top100_overlap_summary_daily_full_100km_w7.csv')
    ap.add_argument('--top-k', type=int, default=25)
    ap.add_argument('--min-id', type=int, default=100_000_000, help='Filter out implausible MMSI-like IDs')
    ap.add_argument('--out', default='artifacts/high_confidence_normalized_top25.csv')
    args = ap.parse_args()

    scores = pd.read_csv(args.scores)
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

    merged = scores.merge(overlap, on=['src', 'dst'], how='inner')
    merged['overlap_ratio_days'] = merged['overlap_ratio_days'].fillna(0)
    merged['close_ratio_days'] = merged['close_ratio_days'].fillna(0)

    merged = merged.sort_values(
        ['close_ratio_days', 'overlap_ratio_days', 'score'],
        ascending=[False, False, False],
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.head(args.top_k).to_csv(out_path, index=False)
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
