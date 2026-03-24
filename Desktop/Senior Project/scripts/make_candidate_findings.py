#!/usr/bin/env python3
"""
Create a markdown findings report from candidate scores + validation evidence.

This avoids relying on `data/features_by_year/` (which may not be present on all machines)
and instead joins against the overlap/validation CSVs under `artifacts/`.
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


def _load_keyed_csv(path: Path) -> Dict[Tuple[int, int], dict]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if not {'src', 'dst'}.issubset(df.columns):
        return {}
    out: Dict[Tuple[int, int], dict] = {}
    for rec in df.to_dict(orient='records'):
        try:
            key = (int(rec['src']), int(rec['dst']))
        except Exception:
            continue
        out[key] = rec
    return out


def _norm_pair_key(src: int, dst: int) -> Tuple[int, int]:
    """Undirected pair key (order-invariant)."""
    s, d = int(src), int(dst)
    return (s, d) if s <= d else (d, s)


def _load_enrichment_map(path: Path) -> Dict[Tuple[int, int], dict]:
    """
    Load flag/gear enrichment (e.g. cooperative_pairs_with_flag_gear.csv).
    Multiple rows per pair (different buckets) → keep first occurrence.
    """
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if not {'src', 'dst'}.issubset(df.columns):
        return {}
    out: Dict[Tuple[int, int], dict] = {}
    for rec in df.to_dict(orient='records'):
        try:
            s, d = int(rec['src']), int(rec['dst'])
        except Exception:
            continue
        k = _norm_pair_key(s, d)
        if k not in out:
            out[k] = rec
    return out


def _fmt_int(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ''
    try:
        return str(int(float(x)))
    except Exception:
        return str(x)


def _fmt_float(x, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ''
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores', default='artifacts/tgcn_candidate_scores_fullcoverage_top500.csv')
    ap.add_argument('--out', default='docs/candidate_findings.md')
    ap.add_argument('--top-k', type=int, default=20)
    ap.add_argument('--min-id', type=int, default=100_000_000, help='Filter out implausible MMSI-like IDs')
    ap.add_argument('--overlap-25', default='artifacts/top100_overlap_summary_daily_full_25km_w1.csv')
    ap.add_argument('--overlap-50', default='artifacts/top100_overlap_summary_daily_full_50km_w3.csv')
    ap.add_argument('--overlap-100', default='artifacts/top100_overlap_summary_daily_full_100km_w7.csv')
    ap.add_argument('--region2deg', default='artifacts/top100_overlap_summary_daily_full_region2deg.csv')
    ap.add_argument(
        '--enrichment',
        default='artifacts/cooperative_pairs_with_flag_gear.csv',
        help='Optional CSV from enrich_pairs_with_flag_gear.py (src_mid, dst_mid, src_gear, dst_gear); '
        'rows merged when undirected pair matches',
    )
    args = ap.parse_args()

    scores = pd.read_csv(args.scores)
    scores = scores[['src', 'dst', 'score']].dropna()
    scores['src'] = scores['src'].astype(int)
    scores['dst'] = scores['dst'].astype(int)
    scores = scores[(scores['src'] >= args.min_id) & (scores['dst'] >= args.min_id)]
    scores = scores.sort_values('score', ascending=False).head(args.top_k)

    ov25 = _load_keyed_csv(Path(args.overlap_25))
    ov50 = _load_keyed_csv(Path(args.overlap_50))
    ov100 = _load_keyed_csv(Path(args.overlap_100))
    reg2 = _load_keyed_csv(Path(args.region2deg))
    enrich = _load_enrichment_map(Path(args.enrichment))

    headers = [
        'src', 'dst', 'score',
        'overlap_days',
        'within25km_±1d', 'mean_dist_25km',
        'within50km_±3d', 'mean_dist_50km',
        'within100km_±7d', 'mean_dist_100km',
        'region2deg_overlap_ratio', 'region2deg_close_ratio',
        'src_mid', 'dst_mid', 'src_gear', 'dst_gear',
    ]

    md_lines = [
        '# Candidate Pair Findings (Top 20)',
        '',
        f"Source scores: `{args.scores}` (filtered to plausible MMSI IDs)",
        '',
        f"- Proximity validation: `{args.overlap_25}`, `{args.overlap_50}`, `{args.overlap_100}`",
        f"- Region validation: `{args.region2deg}`",
        f"- Optional flag/gear (Aug 2017 TGCN enrichment, merged when pair matches): `{args.enrichment}`",
        '',
        '|' + '|'.join(headers) + '|',
        '|' + '|'.join(['---'] * len(headers)) + '|',
    ]

    for _, row in scores.iterrows():
        key = (int(row['src']), int(row['dst']))
        r25 = ov25.get(key, {})
        r50 = ov50.get(key, {})
        r100 = ov100.get(key, {})
        rr2 = reg2.get(key, {})
        ek = _norm_pair_key(key[0], key[1])
        en = enrich.get(ek, {})

        def _en(col: str) -> str:
            v = en.get(col, '')
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return ''
            return str(v).strip()

        overlap_days = r25.get('overlap_days', '')
        md_lines.append(
            '|'
            + '|'.join([
                str(key[0]),
                str(key[1]),
                _fmt_float(row['score'], 3),
                _fmt_int(overlap_days),
                _fmt_int(r25.get('days_within_km', '')),
                _fmt_float(r25.get('mean_distance_km', ''), 2),
                _fmt_int(r50.get('days_within_km', '')),
                _fmt_float(r50.get('mean_distance_km', ''), 2),
                _fmt_int(r100.get('days_within_km', '')),
                _fmt_float(r100.get('mean_distance_km', ''), 2),
                _fmt_float(rr2.get('overlap_ratio_days', ''), 3),
                _fmt_float(rr2.get('close_ratio_days', ''), 3),
                _en('src_mid'),
                _en('dst_mid'),
                _en('src_gear'),
                _en('dst_gear'),
            ])
            + '|'
        )

    md_lines += [
        '',
        'Notes: overlap/validation rows come from the validated “top‑100” overlap artifacts, so they may be blank for pairs outside that set.',
        'Flag/gear columns are filled only when this pair appears in the Aug 2017 enrichment CSV (undirected match); most full‑coverage top pairs will have these blank.',
        '',
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(md_lines))
    print(f"Wrote {out_path}")


if __name__ == '__main__':
    main()
