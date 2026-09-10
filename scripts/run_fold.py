#!/usr/bin/env python3
"""
Run a single walk-forward fold using the functions defined in scripts/predict.py.

Produces outputs/result_fold_<fold>.json with:
- fold index
- test rows (date + probability + label)
- fold trades (list)
- fold metrics (trade_metrics)
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from importlib import util
from dataclasses import asdict

try:
    from strategy_config import (COMMISSION_BPS, ENTRY_GAP_HIGH_ATR, ENTRY_GAP_LOW_ATR,
                                 FINAL_TEST_FRACTION, HORIZON, LABEL_MODE, MIN_TRAIN,
                                 REWARD_RISK, SLIPPAGE_BPS, STOP_ATR, TAX_BPS, THRESHOLD)
except ModuleNotFoundError:
    from scripts.strategy_config import (COMMISSION_BPS, ENTRY_GAP_HIGH_ATR, ENTRY_GAP_LOW_ATR,
                                         FINAL_TEST_FRACTION, HORIZON, LABEL_MODE, MIN_TRAIN,
                                         REWARD_RISK, SLIPPAGE_BPS, STOP_ATR, TAX_BPS, THRESHOLD)

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
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--target", type=float, default=0.04)
    p.add_argument("--adverse", type=float, default=-0.025)
    p.add_argument("--threshold", type=float)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--fold-index", type=int, required=True, help="0-based fold index to run")
    p.add_argument("--min-train", type=int, default=MIN_TRAIN)
    p.add_argument("--label-mode", choices=["legacy-target", "trade-outcome"], default=LABEL_MODE)
    p.add_argument("--feature-set", choices=["baseline", "all"], default="all")
    p.add_argument("--model", choices=["extra-trees", "logistic"], default="extra-trees")
    p.add_argument("--output", default="outputs/result_fold.json")
    p.add_argument("--no-record", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    mod = load_predict_module()

    # Match predict.py defaults for threshold if not provided
    if args.threshold is None:
        args.threshold = THRESHOLD if args.model == "extra-trees" else 0.70

    # Use the same default context as predict.py when calling download_data
    context = ["2330.TW", "^TWII"]
    primary, contexts, skipped = mod.download_data(args.ticker, context, args.period)
    data, features = mod.build_dataset(
        primary, contexts, args.horizon, args.target, args.adverse, args.feature_set,
        args.label_mode, STOP_ATR, REWARD_RISK, COMMISSION_BPS, TAX_BPS,
        SLIPPAGE_BPS, ENTRY_GAP_LOW_ATR, ENTRY_GAP_HIGH_ATR)
    usable = data.dropna(subset=["label", "ATR"]).copy()
    # Use final-test default of predict.py (20%) here for splitting
    cut = int(len(usable) * (1 - FINAL_TEST_FRACTION))
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
        # simulate trades for this fold
        trades = mod.simulate(test, primary, args.threshold, args.horizon, stop_atr=STOP_ATR,
                              reward_risk=REWARD_RISK, commission_bps=COMMISSION_BPS,
                              tax_bps=TAX_BPS, slippage_bps=SLIPPAGE_BPS,
                              entry_gap_low_atr=ENTRY_GAP_LOW_ATR,
                              entry_gap_high_atr=ENTRY_GAP_HIGH_ATR)
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
        }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
