#!/usr/bin/env python3
"""
Aggregate fold JSON results and produce a summary report.

Usage:
  python scripts/aggregate_results.py --artifacts-dir ./artifacts --out ./report/summary.json --threshold 0.22
"""
from __future__ import annotations
import argparse
import json
import os
from glob import glob
import pandas as pd
from statistics import mean

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts-dir", default="./artifacts")
    p.add_argument("--out", default="./report/summary.json")
    p.add_argument("--threshold", type=float, default=0.22, help="Probability threshold to count signals")
    return p.parse_args()

def main():
    args = parse_args()
    # search recursively for result-fold-*.json within artifacts dir
    files = sorted(glob(os.path.join(args.artifacts_dir, "**", "outputs", "result-fold-*.json"), recursive=True))
    if not files:
        # fallback to top-level pattern
        files = sorted(glob(os.path.join(args.artifacts_dir, "result-fold-*.json")))
    if not files:
        raise SystemExit("no fold result artifacts found")

    all_rows = []
    all_trades = []
    fold_summaries = []
    oos_preds = 0
    oos_signals = 0
    buy_label_values = []

    for fp in files:
        with open(fp, "r", encoding="utf-8") as fh:
            j = json.load(fh)
        if not j.get("ok"):
            fold_summaries.append({"file": fp, "ok": False, "reason": j.get("reason")})
            continue

        # accumulate rows
        for r in j.get("rows", []):
            all_rows.append({"date": r["date"], "probability": r["probability"], "label": r["label"], "file": os.path.basename(fp)})
            if r.get("probability") is not None:
                oos_preds += 1
                if float(r["probability"]) >= args.threshold:
                    oos_signals += 1
                    buy_label_values.append(int(r["label"]))

        # accumulate trades
        all_trades.extend(j.get("trades", []))

        # record fold summary if available
        summary = j.get("summary", {})
        fold_summaries.append({"file": fp, "summary": summary})

        # if fold provided oos_probability summary, accumulate its counts if present
        if j.get("oos_probability"):
            try:
                oos_info = j["oos_probability"]
                # prefer numeric fields if present
                if "predictions" in oos_info:
                    # keep totals in case rows were not included
                    oos_preds += int(oos_info.get("predictions", 0)) - 0  # no-op if already counted
                if "signals" in oos_info:
                    # we already count signals from rows; this is just a fallback
                    oos_signals += int(oos_info.get("signals", 0))
            except Exception:
                pass

    df = pd.DataFrame(all_rows)
    # compute aggregated oos probability metrics using threshold
    if not df.empty:
        df['probability'] = pd.to_numeric(df['probability'], errors='coerce')
        df['label'] = pd.to_numeric(df['label'], errors='coerce')
        total_predictions = int(len(df))
        signals = int((df.probability >= args.threshold).sum())
        buy_precision = float(df.loc[df.probability >= args.threshold, "label"].mean()) if signals > 0 else None
        mean_prob = float(df.probability.mean())
        prob_quantiles = df.probability.quantile([0, 0.25, 0.5, 0.75, 1.0]).to_dict()
    else:
        total_predictions = 0
        signals = 0
        buy_precision = None
        mean_prob = None
        prob_quantiles = {}

    summary = {
        "fold_files": files,
        "n_rows": total_predictions,
        "threshold": float(args.threshold),
        "signals": signals,
        "buy_precision": buy_precision,
        "mean_probability": mean_prob,
        "probability_quantiles": {str(k): float(v) for k, v in prob_quantiles.items()} if prob_quantiles else {},
        "n_trades": len(all_trades),
        "trades": all_trades,
        "fold_summaries": fold_summaries,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print("Wrote", args.out)

if __name__ == "__main__":
    raise SystemExit(main())