#!/usr/bin/env python3
"""
Hyperparameter tuning for TGCN + temporal features.

Runs a grid over embedding_dim, epochs, lr, cheb_k, and optional hard negatives.
Uses the capped graph by default (runs on typical hardware).
Outputs a CSV and JSON summary of all configs.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--edges', default='artifacts/edges_2012_2019_cap5000_even30.parquet')
    ap.add_argument('--embedding-dims', default='16,32')
    ap.add_argument('--epochs-list', default='5,10')
    ap.add_argument('--lrs', default='0.001,0.01')
    ap.add_argument('--cheb-ks', default='2,3')
    ap.add_argument('--seeds', default='1,2,3')
    ap.add_argument('--max-test-buckets', type=int, default=20)
    ap.add_argument('--with-hard-negatives', action='store_true')
    ap.add_argument('--out-csv', default='artifacts/tgcn_tune_results.csv')
    ap.add_argument('--out-json', default='artifacts/tgcn_tune_results.json')
    args = ap.parse_args()

    embedding_dims = [int(x) for x in args.embedding_dims.split(',')]
    epochs_list = [int(x) for x in args.epochs_list.split(',')]
    lrs = [float(x) for x in args.lrs.split(',')]
    cheb_ks = [int(x) for x in args.cheb_ks.split(',')]

    results = []
    total = len(embedding_dims) * len(epochs_list) * len(lrs) * len(cheb_ks) * (2 if args.with_hard_negatives else 1)
    run = 0

    for emb in embedding_dims:
        for epochs in epochs_list:
            for lr in lrs:
                for cheb_k in cheb_ks:
                    for use_hard in ([False, True] if args.with_hard_negatives else [False]):
                        run += 1
                        config_id = f"emb{emb}_ep{epochs}_lr{lr}_cheb{cheb_k}" + ("_hard" if use_hard else "")
                        report_path = Path(f"artifacts/tgcn_tune_{config_id}.json")
                        print(f"[{run}/{total}] {config_id} ...", flush=True)

                        cmd = [
                            sys.executable,
                            "scripts/run_tgcn_time_multiseed.py",
                            "--edges", args.edges,
                            "--embedding-dim", str(emb),
                            "--epochs", str(epochs),
                            "--lr", str(lr),
                            "--cheb-k", str(cheb_k),
                            "--seeds", args.seeds,
                            "--use-temporal-node-features",
                            "--max-test-buckets", str(args.max_test_buckets),
                            "--out-report", str(report_path),
                        ]
                        if use_hard:
                            cmd.append("--use-hard-negatives")

                        ret = subprocess.run(cmd, cwd=Path(__file__).resolve().parent.parent, capture_output=True, text=True)
                        if ret.returncode != 0:
                            print(f"  FAILED: {ret.stderr[:500]}", flush=True)
                            results.append({
                                "config": config_id,
                                "embedding_dim": emb,
                                "epochs": epochs,
                                "lr": lr,
                                "cheb_k": cheb_k,
                                "use_hard_negatives": use_hard,
                                "roc_auc_mean": None,
                                "roc_auc_std": None,
                                "ap_mean": None,
                                "ap_std": None,
                                "error": ret.stderr[:200],
                            })
                            continue

                        try:
                            with open(report_path) as f:
                                data = json.load(f)
                            overall = data.get("overall", {})
                            results.append({
                                "config": config_id,
                                "embedding_dim": emb,
                                "epochs": epochs,
                                "lr": lr,
                                "cheb_k": cheb_k,
                                "use_hard_negatives": use_hard,
                                "roc_auc_mean": overall.get("roc_auc_mean"),
                                "roc_auc_std": overall.get("roc_auc_std"),
                                "average_precision_mean": overall.get("average_precision_mean"),
                                "average_precision_std": overall.get("average_precision_std"),
                            })
                            print(f"  ROC AUC = {overall.get('roc_auc_mean', 0):.4f} +/- {overall.get('roc_auc_std', 0):.4f}", flush=True)
                        except Exception as e:
                            results.append({
                                "config": config_id,
                                "embedding_dim": emb,
                                "epochs": epochs,
                                "lr": lr,
                                "cheb_k": cheb_k,
                                "use_hard_negatives": use_hard,
                                "roc_auc_mean": None,
                                "error": str(e),
                            })

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    with open(args.out_json, "w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"Wrote {args.out_json}")

    best = max((r for r in results if r.get("roc_auc_mean") is not None), key=lambda r: r["roc_auc_mean"])
    print(f"\nBest config: {best['config']} (ROC AUC = {best['roc_auc_mean']:.4f})")


if __name__ == "__main__":
    main()
