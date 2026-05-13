#!/usr/bin/env python3
"""
Run GCN vs graph-transformer (TransformerConv) baselines with the same protocol
as run_tgcn_time_multiseed.py, then write a side-by-side JSON summary.

Example (from repo root; ``cd`` into the project first):

  KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/compare_gcn_graph_transformer.py \\
    --edges artifacts/edges_2012_2019_cap5000_even30.parquet -- \\
    --max-train-buckets 400 --max-test-buckets 80 --epochs 5 \\
    --seeds 1,2,3 --embedding-dim 32 --lr 0.001 \\
    --use-temporal-node-features --use-hard-negatives

Optional third column: add ``--include-tgcn``. That runs ``--model tgcn_pyg``,
the T-GCN gated cell (Zhao et al.) using only ``torch_geometric.nn.GCNConv``—the
same logic as ``torch_geometric_temporal``'s ``TGCN`` class, **without**
``torch_sparse`` or importing the rest of ``torch_geometric_temporal`` (which
often breaks on mixed x86_64/arm64 Mac installs).

If you need the **library** ``--model tgcn`` instead, fix your PyTorch / extension
wheel architecture so ``torch_sparse`` loads; see
https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

Without ``--include-tgcn``, only GCN and graph_transformer run.

Writes **JSON** (``--out``), **Markdown** (default: same basename as ``.md``),
and **CSV** (default: ``*_table.csv``). Subprocess JSON dumps are hidden unless
you pass ``--verbose``. Rankings and ``±`` use **seed std** for single-split runs
(fixed in ``run_tgcn_time_multiseed``).

**Paired statistics (GCN vs graph transformer):** On identical test buckets and
RNG seeds, per-day ROC-AUC (and AP) are aligned and tested with paired
``t``-tests, Wilcoxon signed-rank, and a bucket-wise sign (binomial) test, plus
a bootstrap 95% CI for the mean day-level ROC-AUC gap. This is the strongest
evidence this repo can produce for “GCN ranks higher on held-out days,” but it
is still **statistical** evidence—not logical 100% certainty.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


def _metrics_for_display(payload: dict[str, Any]) -> dict[str, float]:
    """ROC/AP mean and std; prefer seed-std when top-level std is missing."""
    o = payload.get('overall') or {}
    auc_m = float(o.get('roc_auc_mean', float('nan')))
    auc_s = float(o.get('roc_auc_std', 0.0) or 0.0)
    ap_m = float(o.get('average_precision_mean', float('nan')))
    ap_s = float(o.get('average_precision_std', 0.0) or 0.0)
    if auc_s == 0.0 and ap_s == 0.0:
        folds = payload.get('per_fold') or []
        if len(folds) == 1:
            fo = folds[0].get('overall') or {}
            auc_s = float(fo.get('roc_auc_std', 0.0) or 0.0)
            ap_s = float(fo.get('average_precision_std', 0.0) or 0.0)
    return {
        'roc_auc_mean': auc_m,
        'roc_auc_std': auc_s,
        'ap_mean': ap_m,
        'ap_std': ap_s,
        'buckets': int(o.get('buckets_evaluated', 0) or 0),
    }


def _protocol_block(sample: dict[str, Any]) -> dict[str, Any]:
    keys = (
        'edges',
        'years',
        'embedding_dim',
        'epochs',
        'lr',
        'test_ratio',
        'cv_folds',
        'cheb_k',
        'gt_heads',
        'use_temporal_node_features',
        'use_vessel_day_features',
        'use_hard_negatives',
        'max_train_buckets',
        'max_test_buckets',
        'in_channels',
        'cutoff_time_bucket',
    )
    out = {k: sample.get(k) for k in keys if k in sample}
    ov = sample.get('overall') or {}
    if 'seeds' in ov:
        out['seeds'] = ov['seeds']
    return out


def _per_seed_bucket_maps(payload: dict[str, Any]) -> dict[int, dict[str, dict[str, float]]]:
    """Map seed -> calendar bucket -> {roc_auc, average_precision}."""
    per_fold = payload.get('per_fold') or []
    if len(per_fold) != 1:
        raise ValueError(
            'Paired inference needs a single time split (use default --cv-folds 0); '
            f'got len(per_fold)={len(per_fold)}.'
        )
    out: dict[int, dict[str, dict[str, float]]] = {}
    for block in per_fold[0].get('per_seed', []):
        seed = int(block['seed'])
        by_bucket: dict[str, dict[str, float]] = {}
        for m in block.get('buckets', []):
            b = str(m['bucket'])
            by_bucket[b] = {
                'roc_auc': float(m['roc_auc']),
                'average_precision': float(m['average_precision']),
            }
        out[seed] = by_bucket
    return out


def _paired_gcn_vs_graph_transformer(
    ref_payload: dict[str, Any],
    other_payload: dict[str, Any],
    *,
    ref_key: str,
    other_key: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Paired tests on per-test-bucket ROC-AUC (same buckets, same seeds as training runs).

    Returns (summary_dict_for_json, rows_for_csv).
    """
    ref_maps = _per_seed_bucket_maps(ref_payload)
    oth_maps = _per_seed_bucket_maps(other_payload)
    common_seeds = sorted(set(ref_maps) & set(oth_maps))
    csv_rows: list[dict[str, Any]] = []
    if not common_seeds:
        return (
            {
                'error': 'No overlapping seeds between the two model reports.',
                'reference_model': ref_key,
                'other_model': other_key,
            },
            csv_rows,
        )

    per_seed: list[dict[str, Any]] = []
    for seed in common_seeds:
        a, b = ref_maps[seed], oth_maps[seed]
        buckets = sorted(set(a) & set(b))
        if not buckets:
            per_seed.append({'seed': seed, 'error': 'No overlapping bucket keys for this seed.'})
            continue
        n_only_ref = len(set(a) - set(b))
        n_only_oth = len(set(b) - set(a))
        bucket_warning = None
        if n_only_ref or n_only_oth:
            bucket_warning = (
                f'Bucket sets differ: only_in_{ref_key}={n_only_ref}, '
                f'only_in_{other_key}={n_only_oth}; using intersection n={len(buckets)}.'
            )

        x = np.array([a[bb]['roc_auc'] for bb in buckets], dtype=float)
        y = np.array([b[bb]['roc_auc'] for bb in buckets], dtype=float)
        d = x - y
        ap_x = np.array([a[bb]['average_precision'] for bb in buckets], dtype=float)
        ap_y = np.array([b[bb]['average_precision'] for bb in buckets], dtype=float)
        d_ap = ap_x - ap_y

        for bb in buckets:
            csv_rows.append(
                {
                    'seed': seed,
                    'bucket': bb,
                    f'roc_auc_{ref_key}': a[bb]['roc_auc'],
                    f'roc_auc_{other_key}': b[bb]['roc_auc'],
                    f'roc_auc_diff_{ref_key}_minus_{other_key}': a[bb]['roc_auc'] - b[bb]['roc_auc'],
                    f'ap_{ref_key}': a[bb]['average_precision'],
                    f'ap_{other_key}': b[bb]['average_precision'],
                    f'ap_diff_{ref_key}_minus_{other_key}': a[bb]['average_precision'] - b[bb]['average_precision'],
                }
            )

        wins = int(np.sum(d > 0))
        ties = int(np.sum(d == 0))
        losses = int(np.sum(d < 0))
        n_decisive = wins + losses
        binom_p_g = float('nan')
        binom_p_two = float('nan')
        if n_decisive > 0:
            binom_p_g = float(stats.binomtest(wins, n_decisive, 0.5, alternative='greater').pvalue)
            binom_p_two = float(stats.binomtest(wins, n_decisive, 0.5, alternative='two-sided').pvalue)

        wilcox_p_g = wilcox_p_two = float('nan')
        if np.any(d != 0):
            try:
                wilcox_p_g = float(stats.wilcoxon(d, alternative='greater', zero_method='wilcox', mode='auto').pvalue)
                wilcox_p_two = float(
                    stats.wilcoxon(d, alternative='two-sided', zero_method='wilcox', mode='auto').pvalue
                )
            except ValueError:
                pass

        t_p_g = t_p_two = float('nan')
        if len(d) > 1:
            t_p_g = float(stats.ttest_rel(x, y, alternative='greater').pvalue)
            t_p_two = float(stats.ttest_rel(x, y, alternative='two-sided').pvalue)

        boot_mean_ci = (float('nan'), float('nan'))
        if len(d) > 0 and n_bootstrap > 0:
            boot: list[float] = []
            for _ in range(n_bootstrap):
                idx = rng.integers(0, len(d), size=len(d))
                boot.append(float(np.mean(d[idx])))
            boot_mean_ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

        row_out: dict[str, Any] = {
            'seed': seed,
            'n_buckets_paired': len(buckets),
            'mean_roc_auc_diff_ref_minus_other': float(np.mean(d)),
            'median_roc_auc_diff_ref_minus_other': float(np.median(d)),
            'bootstrap_mean_diff_95pct_ci_low': boot_mean_ci[0],
            'bootstrap_mean_diff_95pct_ci_high': boot_mean_ci[1],
            'buckets_ref_higher_roc_auc': wins,
            'buckets_tie_roc_auc': ties,
            'buckets_other_higher_roc_auc': losses,
            'sign_test_binomial_p_ref_greater': binom_p_g,
            'sign_test_binomial_p_two_sided': binom_p_two,
            'wilcoxon_signed_rank_p_ref_greater': wilcox_p_g,
            'wilcoxon_signed_rank_p_two_sided': wilcox_p_two,
            'paired_ttest_p_ref_greater': t_p_g,
            'paired_ttest_p_two_sided': t_p_two,
            'mean_ap_diff_ref_minus_other': float(np.mean(d_ap)),
        }
        if bucket_warning:
            row_out['warning'] = bucket_warning
        per_seed.append(row_out)

    summary = {
        'reference_model': ref_key,
        'other_model': other_key,
        'metric': 'per_test_bucket_roc_auc_paired_across_identical_calendar_days',
        'disclaimer': (
            'P-values are for the stated null hypotheses on this split and graph; '
            'they do not assert universal superiority of one architecture.'
        ),
        'per_seed': per_seed,
    }
    return summary, csv_rows


def _fmt_scientific(x: float) -> str:
    if x != x:  # NaN
        return 'nan'
    return f'{x:.4g}'


def _format_paired_markdown(paired: dict[str, Any]) -> str:
    if paired.get('error'):
        return f"## Paired inference\n\n**Skipped:** {paired['error']}\n"
    lines = [
        '## Paired inference (same test days, same seeds)',
        '',
        f'**Reference:** `{paired["reference_model"]}` vs **other:** `{paired["other_model"]}`.',
        'Positive mean Δ = reference has higher per-day ROC-AUC on paired buckets.',
        '',
        '| seed | n days | mean Δ ROC | 95% boot CI (mean Δ) | Wilcoxon *p* (ref > other) | paired *t* *p* (>) | sign test *p* (>) | wins / ties / losses |',
        '| ---: | ---: | ---: | --- | --- | --- | --- | --- |',
    ]
    for block in paired.get('per_seed', []):
        if block.get('error'):
            lines.append(f"| {block.get('seed', '')} | — | — | — | — | — | — | {block['error']} |")
            continue
        w, t, l = (
            block['buckets_ref_higher_roc_auc'],
            block['buckets_tie_roc_auc'],
            block['buckets_other_higher_roc_auc'],
        )
        lo, hi = block['bootstrap_mean_diff_95pct_ci_low'], block['bootstrap_mean_diff_95pct_ci_high']
        ci_str = f'[{_fmt_scientific(lo)}, {_fmt_scientific(hi)}]' if lo == lo and hi == hi else '[nan, nan]'
        lines.append(
            f"| {block['seed']} | {block['n_buckets_paired']} | "
            f"{block['mean_roc_auc_diff_ref_minus_other']:.5f} | "
            f"{ci_str} | "
            f"{_fmt_scientific(block['wilcoxon_signed_rank_p_ref_greater'])} | "
            f"{_fmt_scientific(block['paired_ttest_p_ref_greater'])} | "
            f"{_fmt_scientific(block['sign_test_binomial_p_ref_greater'])} | "
            f"{w} / {t} / {l} |"
        )
    lines.extend(['', f"*Disclaimer:* {paired.get('disclaimer', '')}"])
    return '\n'.join(lines)


def _write_paired_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _print_table(rows: list[dict[str, Any]], title: str) -> None:
    cols = ('rank', 'model', 'ROC-AUC', 'AP', 'Δ ROC vs best')
    w_rank, w_model, w_auc, w_ap, w_delta = 4, 22, 22, 22, 14
    line = (
        f"+{'-' * (w_rank + 2)}+{'-' * (w_model + 2)}+{'-' * (w_auc + 2)}+"
        f"{'-' * (w_ap + 2)}+{'-' * (w_delta + 2)}+"
    )
    print(title)
    print(line)
    print(
        f"| {cols[0]:^{w_rank}} | {cols[1]:^{w_model}} | {cols[2]:^{w_auc}} | "
        f"{cols[3]:^{w_ap}} | {cols[4]:^{w_delta}} |"
    )
    print(line)
    for r in rows:
        print(
            f"| {r['rank']:^{w_rank}} | {r['display_name']:{w_model}} | "
            f"{r['roc_str']:{w_auc}} | {r['ap_str']:{w_ap}} | {r['delta_str']:{w_delta}} |"
        )
    print(line)


def _write_md(
    path: Path,
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    notes: str | None,
    paired_md: str | None = None,
) -> None:
    lines = [
        '# Graph model comparison (same protocol)',
        '',
        '| Rank | Model | ROC-AUC | AP | Δ ROC vs best |',
        '| ---: | --- | --- | --- | --- |',
    ]
    for r in rows:
        lines.append(
            f"| {r['rank']} | {r['display_name']} | {r['roc_str']} | {r['ap_str']} | {r['delta_str']} |"
        )
    lines.append('')
    lines.append('## Protocol')
    lines.append('')
    lines.append('```json')
    lines.append(json.dumps(protocol, indent=2))
    lines.append('```')
    if notes:
        lines.extend(['', '## Notes', '', notes])
    if paired_md:
        lines.extend(['', paired_md])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        'rank',
        'model_key',
        'backend',
        'roc_auc_mean',
        'roc_auc_std',
        'average_precision_mean',
        'average_precision_std',
        'delta_roc_auc_vs_best',
        'buckets_evaluated',
    ]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--edges', required=True)
    ap.add_argument('--out', default='artifacts/gcn_vs_graph_transformer.json')
    ap.add_argument(
        '--out-md',
        default='',
        help='Markdown summary path (default: same basename as --out with .md)',
    )
    ap.add_argument(
        '--out-csv',
        default='',
        help='CSV summary path (default: same basename as --out with _table.csv)',
    )
    ap.add_argument(
        '--include-tgcn',
        action='store_true',
        help='Also run T-GCN (tgcn_pyg: GCNConv-only cell, no torch_sparse)',
    )
    ap.add_argument(
        '--verbose',
        action='store_true',
        help='Show full stdout from each run_tgcn_time_multiseed subprocess',
    )
    ap.add_argument(
        '--no-paired-inference',
        action='store_true',
        help='Skip paired per-bucket tests (Wilcoxon, paired t, sign test, bootstrap CI).',
    )
    ap.add_argument(
        '--paired-reference',
        default='gcn',
        help='JSON key under models/ for the reference side of paired tests (default: gcn).',
    )
    ap.add_argument(
        '--paired-other',
        default='graph_transformer',
        help='JSON key under models/ for the comparison side (default: graph_transformer).',
    )
    ap.add_argument(
        '--bootstrap-iters',
        type=int,
        default=2000,
        help='Bootstrap resamples for 95%% CI of mean day-level ROC-AUC gap (0 disables).',
    )
    ap.add_argument('--bootstrap-seed', type=int, default=0, help='RNG seed for bootstrap resampling.')
    ap.add_argument(
        '--out-paired-csv',
        default='',
        help='Optional path for per-bucket paired rows (default: <out-stem>_paired_buckets.csv).',
    )
    ap.add_argument(
        'pass_through',
        nargs=argparse.REMAINDER,
        help='Extra args after -- are forwarded to run_tgcn_time_multiseed.py (e.g. --epochs 5 --lr 0.005)',
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    script = Path(__file__).resolve().parent / 'run_tgcn_time_multiseed.py'
    extra = list(args.pass_through)
    if extra and extra[0] == '--':
        extra = extra[1:]

    models = [('gcn', 'gcn'), ('graph_transformer', 'graph_transformer')]
    if args.include_tgcn:
        models.append(('tgcn', 'tgcn_pyg'))

    combined: dict[str, Any] = {
        'edges': args.edges,
        'models': {},
    }
    if args.include_tgcn:
        combined['notes'] = (
            'Table key "tgcn" runs --model tgcn_pyg (T-GCN cell, GCNConv only; no torch_sparse).'
        )

    sub_kw: dict[str, Any] = {'check': True, 'cwd': str(repo_root)}
    if not args.verbose:
        sub_kw['stdout'] = subprocess.DEVNULL

    t0 = time.perf_counter()
    for key, model in models:
        report_path = repo_root / 'artifacts' / f'_compare_tmp_{key}.json'
        cmd = [
            sys.executable,
            str(script),
            '--edges',
            args.edges,
            '--model',
            model,
            '--out-report',
            str(report_path),
            '--out-csv',
            str(repo_root / 'artifacts' / f'_compare_tmp_{key}.csv'),
            *extra,
        ]
        print(f"  [{model}] training + eval …", flush=True)
        t_run = time.perf_counter()
        subprocess.run(cmd, **sub_kw)
        print(f"  [{model}] done in {time.perf_counter() - t_run:.1f}s", flush=True)
        with open(report_path) as f:
            combined['models'][key] = json.load(f)

    first = next(iter(combined['models'].values()))
    protocol = _protocol_block(first)
    combined['protocol'] = protocol

    display_rows: list[dict[str, Any]] = []
    for key, backend in models:
        payload = combined['models'][key]
        m = _metrics_for_display(payload)
        display_name = key
        if key == 'tgcn' and backend == 'tgcn_pyg':
            display_name = 'tgcn (tgcn_pyg)'
        display_rows.append(
            {
                'key': key,
                'display_name': display_name,
                'backend': backend,
                'roc_m': m['roc_auc_mean'],
                'roc_s': m['roc_auc_std'],
                'ap_m': m['ap_mean'],
                'ap_s': m['ap_std'],
                'buckets': m['buckets'],
            }
        )

    best_roc = max(r['roc_m'] for r in display_rows) if display_rows else float('nan')
    ranked = sorted(display_rows, key=lambda r: (-r['roc_m'], r['key']))
    table_rows: list[dict[str, Any]] = []
    for i, r in enumerate(ranked, start=1):
        delta = r['roc_m'] - best_roc if best_roc == best_roc else float('nan')
        table_rows.append(
            {
                'rank': i,
                'display_name': r['display_name'],
                'model_key': r['key'],
                'backend': r['backend'],
                'roc_str': f"{r['roc_m']:.4f} ± {r['roc_s']:.4f}",
                'ap_str': f"{r['ap_m']:.4f} ± {r['ap_s']:.4f}",
                'delta_str': f"{delta:+.4f}" if i > 1 else '—',
                'roc_auc_mean': r['roc_m'],
                'roc_auc_std': r['roc_s'],
                'average_precision_mean': r['ap_m'],
                'average_precision_std': r['ap_s'],
                'delta_roc_auc_vs_best': 0.0 if i == 1 else delta,
                'buckets_evaluated': r['buckets'],
            }
        )

    combined['comparison'] = {
        'ranked_by_roc_auc': table_rows,
        'elapsed_seconds': round(time.perf_counter() - t0, 2),
    }

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    md_path = Path(args.out_md) if args.out_md else out_path.with_suffix('.md')
    if not md_path.is_absolute():
        md_path = repo_root / md_path

    csv_path = Path(args.out_csv) if args.out_csv else out_path.with_name(out_path.stem + '_table.csv')
    if not csv_path.is_absolute():
        csv_path = repo_root / csv_path

    paired_md: str | None = None
    paired_csv_path: Path | None = None
    paired_summary: dict[str, Any] = {}
    if not args.no_paired_inference:
        rk, ok = args.paired_reference, args.paired_other
        if rk in combined['models'] and ok in combined['models']:
            paired_summary, paired_rows = _paired_gcn_vs_graph_transformer(
                combined['models'][rk],
                combined['models'][ok],
                ref_key=rk,
                other_key=ok,
                n_bootstrap=max(0, args.bootstrap_iters),
                rng=np.random.default_rng(args.bootstrap_seed),
            )
            combined['paired_inference'] = paired_summary
            paired_md = _format_paired_markdown(paired_summary)
            pcsv = Path(args.out_paired_csv) if args.out_paired_csv else out_path.with_name(out_path.stem + '_paired_buckets.csv')
            if not pcsv.is_absolute():
                pcsv = repo_root / pcsv
            paired_csv_path = pcsv
            _write_paired_csv(pcsv, paired_rows)
        else:
            miss = [k for k in (rk, ok) if k not in combined['models']]
            paired_summary = {
                'error': f'Paired keys not in run set (missing {miss}).',
                'reference_model': rk,
                'other_model': ok,
            }
            combined['paired_inference'] = paired_summary
            paired_md = _format_paired_markdown(paired_summary)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2)

    notes = combined.pop('notes', None)
    _write_md(md_path, table_rows, protocol, notes, paired_md=paired_md)
    _write_csv(csv_path, table_rows)

    print()
    _print_table(table_rows, 'Comparison (higher ROC-AUC is better)')
    if paired_md and not args.no_paired_inference:
        print()
        print(f'Paired tests: {paired_summary.get("reference_model", "?")} vs {paired_summary.get("other_model", "?")}')
        for block in paired_summary.get('per_seed', []):
            if block.get('error'):
                print(f"  seed {block.get('seed')}: {block['error']}")
                continue
            print(
                f"  seed {block['seed']}: mean ΔROC={block['mean_roc_auc_diff_ref_minus_other']:.5f} "
                f"Wilcoxon p(>)={_fmt_scientific(block['wilcoxon_signed_rank_p_ref_greater'])} "
                f"paired-t p(>)={_fmt_scientific(block['paired_ttest_p_ref_greater'])} "
                f"wins/ties/losses={block['buckets_ref_higher_roc_auc']}/"
                f"{block['buckets_tie_roc_auc']}/{block['buckets_other_higher_roc_auc']}"
            )
    print(f"\nTotal time: {combined['comparison']['elapsed_seconds']}s")
    print(f"JSON:  {out_path}")
    print(f"Markdown: {md_path}")
    print(f"CSV:   {csv_path}")
    if paired_csv_path is not None:
        print(f"Paired buckets CSV: {paired_csv_path}")


if __name__ == '__main__':
    main()
