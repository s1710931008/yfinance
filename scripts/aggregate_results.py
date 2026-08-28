#!/usr/bin/env python3
"""
Download all fold result JSON files from ./artifacts/ (or passed dir) and aggregate:
- combine rows into a single oos DataFrame
- combine trades from all folds
- compute global trade_metrics and oos probability metrics
- write report/report_summary.json
"""
from __future__ import annotations
import argparse
import json
import os
from glob import glob
from collections import defaultdict
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts-dir", default="./artifacts")
    p.add_argument("--out", default="./report/summary.json")
    return p.parse_args()

def main():
    args = parse_args()
    files = sorted(glob(os.path.join(args.artifacts_dir, "**/outputs/result-fold-*.json"), recursive=True))
    if not files:
        # fallback to non-recursive pattern
        files = sorted(glob(os.path.join(args.artifacts_dir, "result-fold-*.json")))
    if not files:
        raise SystemExit("no fold result artifacts found")
    all_rows = []
    all_trades = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as fh:
            j = json.load(fh)
        if not j.get("ok"):
            continue
        for r in j.get("rows", []):
            all_rows.append({"date": r["date"], "probability": r["probability"], "label": r["label"]})
        all_trades.extend(j.get("trades", []))

    # compute simple aggregated metrics using pandas
    df = pd.DataFrame(all_rows)
    if not df.empty:
        df['probability'] = pd.to_numeric(df['probability'], errors='coerce')
        df['label'] = pd.to_numeric(df['label'], errors='coerce')
    oos_prob = {
        "predictions": int(len(df)),
        "signals": int((df.probability >= 0.0).sum()) if not df.empty else 0,
        "buy_precision": float(df.loc[df.probability >= 0.0, "label"].mean()) if not df.empty else None
    }
    summary = {
        "fold_files": files,
        "n_rows": len(df),
        "oos_prob_summary": oos_prob,
        "n_trades": len(all_trades),
        "trades": all_trades,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print("Wrote", args.out)

if __name__ == "__main__":
    raise SystemExit(main())
