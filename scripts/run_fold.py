#!/usr/bin/env python3
"""
Run a single walk-forward fold using the functions defined in scripts/predict.py.

Produces outputs/result-fold-<fold>.json with:
- fold index
- test rows (date + probability + label)
- fold trades (list)
- fold metrics (trade_metrics)
- fold summary: probability mean/median/std/quantiles, num_above_threshold, threshold
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from importlib import util
from dataclasses import asdict

HERE = os.path.dirname(__file__)
PREDICT_PATH = os.path.join(HERE, "predict.py")


def load_predict_module():
    spec = util.spec_from_file_location("predict", PREDICT_PATH)
    mod = util.module_from_spec(spec)
    sys.modules["predict"] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("ticker")
    p.add_argument("--period", default="10y")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--target", type=float, default=0.04)
    p.add_argument("--adverse", type=float, default=-0.025)
    p.add_argument("--threshold", type=float)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--fold-index", type=int, required=True, help="0-based fold index to run")
    p.add_argument("--min-train", type=int, default=252)
    p.add_argument("--feature-set", choices=["baseline", "all"], default="all")
    p.add_argument("--model", choices=["extra-trees", "logistic"], default="extra-trees")
    p.add_argument("--output", default="outputs/result-fold.json")
    p.add_argument("--no-record", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    mod = load_predict_module()

    # Default threshold consistent with predict.py
    if args.threshold is None:
        args.threshold = 0.22 if args.model == "extra-trees" else 0.70

    # Use the same default context as predict.py when calling download_data
    context = ["2330.TW", "^TWII"]
    primary, contexts, skipped = mod.download_data(args.ticker, context, args.period)
    data, features = mod.build_dataset(primary, contexts, args.horizon, args.target, args.adverse, args.feature_set)
    usable = data.dropna(subset=["label", "ATR"]).copy()
    # Use final-test default of predict.py (20%) here for splitting
    cut = int(len(usable) * (1 - 0.20))
    dev = usable.iloc[:cut]

    # Compute boundaries exactly as predict.walk_forward
    import numpy as np
    boundaries = np.linspace(args.min_train, len(dev), args.folds + 1, dtype=int)
    fold = args.fold_index
    if not (0 <= fold < args.folds):
        raise SystemExit(f"fold-index {fold} outside [0, {args.folds})")

    train = dev.iloc[:max(1, boundaries[fold])]
    test = dev.iloc[boundaries[fold]:boundaries[fold + 1]].copy()
    if test.empty or train.label.nunique() < 2:
        result = {"fold": fold, "ok": False, "reason": "no test rows or insufficient train labels"}
    else:
        # Fit & calibrate on this fold (reuse predict.calibrated_fit_predict)
        prob_values, base, calibrator = mod.calibrated_fit_predict(train, test, features, purge=0, model_kind=args.model)
        test = test.copy()
        test["probability"] = prob_values

        # compute extra summary stats for diagnostics
        probs = test["probability"].to_numpy(dtype=float)
        import numpy as _np
        summary = {}
        if probs.size > 0:
            summary["mean_probability"] = float(_np.nanmean(probs))
            summary["median_probability"] = float(_np.nanmedian(probs))
            summary["std_probability"] = float(_np.nanstd(probs, ddof=0))
            q = _np.nanpercentile(probs, [0, 25, 50, 75, 100]).tolist()
            summary["probability_quantiles"] = {"p0": float(q[0]), "p25": float(q[1]),
                                                "p50": float(q[2]), "p75": float(q[3]), "p100": float(q[4])}
        else:
            summary["mean_probability"] = None
            summary["median_probability"] = None
            summary["std_probability"] = None
            summary["probability_quantiles"] = {}

        summary["threshold"] = float(args.threshold)
        summary["num_above_threshold"] = int((test["probability"] >= args.threshold).sum())

        # simulate trades for this fold (uses provided threshold in simulation later if needed)
        trades = mod.simulate(test, primary, args.threshold, args.horizon, stop_atr=1.5,
                              reward_risk=2.0, commission_bps=14.25, tax_bps=10.0, slippage_bps=5.0,
                              entry_gap_low_atr=None, entry_gap_high_atr=None)
        trades_dicts = [asdict(t) for t in trades]
        metrics = mod.trade_metrics(trades)
        oos_prob = mod.probability_metrics(test, args.threshold)
        # prepare output
        rows = [{"date": str(idx.date()), "probability": float(p), "label": int(l)}
                for idx, p, l in zip(test.index, test.probability, test.label)]
        result = {
            "fold": fold,
            "ok": True,
            "rows": rows,
            "trades": trades_dicts,
            "metrics": metrics,
            "oos_probability": oos_prob,
            "model": args.model,
            "feature_set": args.feature_set,
            "ticker": args.ticker,
            "summary": summary,
        }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())