#!/usr/bin/env python3
"""Leakage-aware walk-forward classifier and trading validation for Yahoo data.

This is a research tool, not investment advice.  The final test window is held out
from model selection; CLI parameters must be chosen before looking at its results.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
import sqlite3
import sys
import urllib.parse
import urllib.request
import warnings
from dataclasses import asdict, dataclass
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class Trade:
    signal_date: str
    entry_date: str
    exit_date: str
    entry: float
    exit: float
    probability: float
    gross_r: float
    net_r: float
    reason: str
    regime: str
    net_return: float = 0.0
    position_fraction: float = 1.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ticker", help="Yahoo Finance symbol, e.g. 00631L.TW")
    p.add_argument("--period", default="10y")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--target", type=float, default=0.04)
    p.add_argument("--adverse", type=float, default=-0.025)
    p.add_argument("--threshold", type=float,
                   help="Signal threshold; defaults to 0.22 for ExtraTrees and 0.70 for Logistic")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--final-test", type=float, default=0.20,
                   help="Chronological untouched fraction (0.10-0.40)")
    p.add_argument("--stop-atr", type=float, default=1.5)
    p.add_argument("--reward-risk", type=float, default=2.0)
    p.add_argument("--entry-gap-low-atr", type=float, default=0.15,
                   help="Minimum next-open displacement from signal close, in ATR; negative allows a gap down (default: 0.15)")
    p.add_argument("--entry-gap-high-atr", type=float, default=0.55,
                   help="Maximum next-open displacement from signal close, in ATR (default: 0.55)")
    p.add_argument("--commission-bps", type=float, default=14.25,
                   help="Commission per side in basis points")
    p.add_argument("--tax-bps", type=float, default=10.0,
                   help="Sell-side transaction tax in basis points")
    p.add_argument("--slippage-bps", type=float, default=5.0,
                   help="Slippage per side in basis points")
    p.add_argument("--context", nargs="*", default=["2330.TW", "^TWII"],
                   help="Context symbols; unavailable symbols are reported and skipped")
    p.add_argument("--futures-symbol",
                   help="Optional Yahoo-compatible continuous futures symbol (Taiwan WTX codes are not exposed by yfinance)")
    p.add_argument("--min-train", type=int, default=252)
    p.add_argument("--feature-set", choices=["baseline", "all"], default="all",
                   help="all is the AGENTS.md default; baseline is retained for comparison")
    p.add_argument("--model", choices=["extra-trees", "logistic"], default="extra-trees",
                   help="Prediction model (default: extra-trees experimental candidate)")
    p.add_argument("--output-json", help="Optional path for machine-readable results")
    p.add_argument("--database", default="predictions.sqlite3",
                   help="Append-only prediction history database")
    p.add_argument("--no-record", action="store_true",
                   help="Do not write this run to SQLite (research comparison only)")
    p.add_argument("--shares", type=int, help="Current holding shares; use with --average-cost")
    p.add_argument("--average-cost", type=float,
                   help="Average cost per share including existing buy-side costs")
    p.add_argument("--add-shares", type=int, default=100,
                   help="Maximum shares allowed for one averaging-down add (default: 100)")
    return p.parse_args()


def _clean_ohlcv_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if frame.empty or not set(needed).issubset(frame.columns):
        raise RuntimeError(f"{symbol}: no usable OHLCV data")
    frame = frame[needed].copy().sort_index()
    frame[needed] = frame[needed].apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=needed)
    frame = frame[(frame[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    if frame.empty:
        raise RuntimeError(f"{symbol}: no complete finite OHLCV rows")
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    return frame[~frame.index.duplicated(keep="last")]


def _twse_month_rows(stock_no: str, date: pd.Timestamp) -> dict[pd.Timestamp, dict[str, float]]:
    """Return official TWSE daily OHLCV rows for one calendar month."""
    query = urllib.parse.urlencode({
        "response": "json", "date": date.strftime("%Y%m%d"), "stockNo": stock_no,
    })
    request = urllib.request.Request(
        f"https://www.twse.com.tw/exchangeReport/STOCK_DAY?{query}",
        headers={"User-Agent": "00631L-research-validation/20260819.2"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.load(response)
    if payload.get("stat") != "OK":
        raise RuntimeError(f"TWSE {stock_no}: {payload.get('stat', 'unknown response')}")
    rows: dict[pd.Timestamp, dict[str, float]] = {}
    for values in payload.get("data", []):
        if len(values) < 8:
            continue
        roc_year, month, day = (int(part) for part in values[0].split("/"))
        row_date = pd.Timestamp(roc_year + 1911, month, day)
        try:
            rows[row_date] = {
                "Open": float(values[3].replace(",", "")),
                "High": float(values[4].replace(",", "")),
                "Low": float(values[5].replace(",", "")),
                "Close": float(values[6].replace(",", "")),
                "Volume": float(values[1].replace(",", "")),
            }
        except (TypeError, ValueError):
            continue
    return rows


def _fill_twse_gaps(frame: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, list[str]]:
    """Fill incomplete Yahoo rows from official TWSE data for matching dates only."""
    if not symbol.endswith(".TW") or frame.empty:
        return frame, []
    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.copy()
        frame.columns = frame.columns.get_level_values(0)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not set(needed).issubset(frame.columns):
        return frame, []
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    numeric = frame[needed].apply(pd.to_numeric, errors="coerce")
    incomplete = numeric[needed].isna().any(axis=1)
    gap_dates = list(numeric.index[incomplete])
    if not gap_dates:
        return frame, []
    official: dict[pd.Timestamp, dict[str, float]] = {}
    for month in sorted({date.to_period("M") for date in gap_dates}):
        official.update(_twse_month_rows(symbol.removesuffix(".TW"), month.to_timestamp()))
    filled: list[str] = []
    for gap_date in gap_dates:
        values = official.get(pd.Timestamp(gap_date))
        if values is None:
            continue
        frame.loc[gap_date, needed] = [values[name] for name in needed]
        filled.append(str(pd.Timestamp(gap_date).date()))
    return frame, filled


def _flat_download(symbol: str, period: str) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = yf.download(symbol, period=period, auto_adjust=True, progress=False,
                            actions=False, threads=False)
    frame, fallback_dates = _fill_twse_gaps(frame, symbol)
    cleaned = _clean_ohlcv_frame(frame, symbol)
    cleaned.attrs["twse_fallback_dates"] = fallback_dates
    return cleaned


def download_data(ticker: str, context: Iterable[str], period: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[str]]:
    primary = _flat_download(ticker, period)
    frames: dict[str, pd.DataFrame] = {}
    skipped: list[str] = []
    for symbol in dict.fromkeys(context):
        if symbol == ticker:
            continue
        try:
            frames[symbol] = _flat_download(symbol, period)
        except Exception as exc:  # context must not prevent a primary-symbol run
            skipped.append(f"{symbol} ({exc})")
    return primary, frames, skipped


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def add_technical_features(d: pd.DataFrame) -> list[str]:
    """Add price/volume, KD, MACD and RSI features using current/past bars only."""
    logv = np.log1p(d.Volume)
    d["rsi14"] = rsi(d.Close) / 100

    # Price-volume relationship: abnormal volume, direction agreement, correlation,
    # OBV momentum and money-flow pressure.
    d["volume_z20"] = (logv - logv.rolling(20).mean()) / logv.rolling(20).std()
    d["volume_ratio_5_20"] = d.Volume.rolling(5).mean() / d.Volume.rolling(20).mean() - 1
    d["price_volume_corr20"] = d.ret_1.rolling(20).corr(logv.diff())
    d["return_x_volume"] = d.ret_1 * d.volume_z20
    obv = (np.sign(d.Close.diff()).fillna(0) * d.Volume).cumsum()
    d["obv_momentum20"] = obv.diff(20) / d.Volume.rolling(20).sum().replace(0, np.nan)
    money_flow_multiplier = ((d.Close - d.Low) - (d.High - d.Close)) / (d.High - d.Low).replace(0, np.nan)
    d["cmf20"] = (money_flow_multiplier * d.Volume).rolling(20).sum() / d.Volume.rolling(20).sum()

    # Stochastic KD (9, 3, 3), normalized to 0..1.
    low9, high9 = d.Low.rolling(9).min(), d.High.rolling(9).max()
    rsv = 100 * (d.Close - low9) / (high9 - low9).replace(0, np.nan)
    d["kd_k"] = rsv.ewm(alpha=1 / 3, adjust=False).mean() / 100
    d["kd_d"] = d.kd_k.ewm(alpha=1 / 3, adjust=False).mean()
    d["kd_spread"] = d.kd_k - d.kd_d

    # MACD (12, 26, 9), divided by price so symbols with different prices compare.
    ema12, ema26 = d.Close.ewm(span=12, adjust=False).mean(), d.Close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    d["macd"] = macd / d.Close
    d["macd_signal"] = signal / d.Close
    d["macd_hist"] = (macd - signal) / d.Close
    d["macd_hist_delta"] = d.macd_hist.diff()

    return ["rsi14", "volume_z20", "volume_ratio_5_20", "price_volume_corr20",
            "return_x_volume", "obv_momentum20", "cmf20", "kd_k", "kd_d",
            "kd_spread", "macd", "macd_signal", "macd_hist", "macd_hist_delta"]


def build_dataset(primary: pd.DataFrame, contexts: dict[str, pd.DataFrame],
                  horizon: int, target: float, adverse: float,
                  feature_set: str = "baseline") -> tuple[pd.DataFrame, list[str]]:
    d = primary.copy()
    prev_close = d["Close"].shift()
    tr = pd.concat([(d.High - d.Low), (d.High - prev_close).abs(),
                    (d.Low - prev_close).abs()], axis=1).max(axis=1)
    d["ATR"] = tr.rolling(14).mean()
    d["ret_1"] = d.Close.pct_change(fill_method=None)
    for n in (5, 10, 20, 60):
        d[f"ret_{n}"] = d.Close.pct_change(n, fill_method=None)
    d["sma20_gap"] = d.Close / d.Close.rolling(20).mean() - 1
    d["sma60_gap"] = d.Close / d.Close.rolling(60).mean() - 1
    d["trend_20_60"] = d.Close.rolling(20).mean() / d.Close.rolling(60).mean() - 1
    d["vol_20"] = d.ret_1.rolling(20).std() * math.sqrt(252)
    d["vol_ratio"] = d.ret_1.rolling(10).std() / d.ret_1.rolling(60).std()
    d["range_atr"] = (d.High - d.Low) / d.ATR
    support20 = d.Low.rolling(20).min()
    resistance20 = d.High.rolling(20).max()
    support60 = d.Low.rolling(60).min()
    resistance60 = d.High.rolling(60).max()
    d["support20_gap"] = d.Close / support20 - 1
    d["resistance20_gap"] = resistance20 / d.Close - 1
    d["support60_gap"] = d.Close / support60 - 1
    d["resistance60_gap"] = resistance60 / d.Close - 1

    features = ["ret_1", "ret_5", "ret_10", "ret_20", "ret_60", "sma20_gap",
                "sma60_gap", "trend_20_60", "vol_20", "vol_ratio", "rsi14",
                "volume_z20", "range_atr"]
    technical = add_technical_features(d)
    if feature_set == "all":
        features = list(dict.fromkeys(features + technical + [
            "support20_gap", "resistance20_gap", "support60_gap", "resistance60_gap"
        ]))
    for i, (symbol, ctx) in enumerate(contexts.items()):
        aligned = ctx.Close.reindex(d.index).ffill(limit=3)
        safe = "".join(ch if ch.isalnum() else "_" for ch in symbol).strip("_") or f"ctx{i}"
        for n in (1, 5, 20):
            name = f"ctx_{safe}_ret{n}"
            d[name] = aligned.pct_change(n, fill_method=None)
            features.append(name)
        name = f"ctx_{safe}_trend"
        d[name] = aligned / aligned.rolling(60).mean() - 1
        features.append(name)

    # Labels are for modelling only. All values used here occur after the signal bar.
    future_close = d.Close.shift(-horizon) / d.Close - 1
    future_highs = pd.concat(
        [d.High.shift(-i) / d.Close - 1 for i in range(1, horizon + 1)], axis=1).max(axis=1)
    future_lows = pd.concat([d.Low.shift(-i) / d.Close - 1 for i in range(1, horizon + 1)], axis=1).min(axis=1)
    d["future_return"] = future_close
    d["future_high_return"] = future_highs
    d["future_low_return"] = future_lows
    d["label"] = ((future_close >= target) & (future_lows >= adverse)).astype(float)
    d.loc[d.index[-horizon:], ["label", "future_return", "future_high_return",
                              "future_low_return"]] = np.nan
    d["regime"] = np.select(
        [d.trend_20_60 >= 0, d.trend_20_60 < 0], ["bull", "bear"], default="unknown")
    high_vol = d.vol_20 > d.vol_20.rolling(252, min_periods=60).median()
    d.loc[high_vol & d.regime.ne("unknown"), "regime"] += "_high_vol"
    return d, features


def model(kind: str = "logistic") -> Pipeline:
    if kind == "extra-trees":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", ExtraTreesClassifier(
                n_estimators=500, min_samples_leaf=12, max_features=0.7,
                class_weight="balanced", random_state=42, n_jobs=-1)),
        ])
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000,
                                   random_state=42)),
    ])


def calibrated_fit_predict(train: pd.DataFrame, test: pd.DataFrame,
                           features: list[str], purge: int = 0,
                           model_kind: str = "logistic") -> tuple[np.ndarray, Pipeline, LogisticRegression | None]:
    split = max(int(len(train) * 0.8), len(train) - 126)
    split = min(max(split, 50), len(train) - 20)
    fit, cal = train.iloc[:max(1, split - purge)], train.iloc[split:]
    base = model(model_kind).fit(fit[features], fit.label.astype(int))
    raw_cal = np.clip(base.predict_proba(cal[features])[:, 1], 1e-6, 1 - 1e-6)
    calibrator = None
    if cal.label.nunique() == 2:
        calibrator = LogisticRegression(C=1e3, max_iter=1000).fit(
            np.log(raw_cal / (1 - raw_cal)).reshape(-1, 1), cal.label.astype(int))
    raw = np.clip(base.predict_proba(test[features])[:, 1], 1e-6, 1 - 1e-6)
    if calibrator is None:
        return raw, base, None
    prob = calibrator.predict_proba(np.log(raw / (1 - raw)).reshape(-1, 1))[:, 1]
    return prob, base, calibrator


def return_fit_predict(train: pd.DataFrame, test: pd.DataFrame,
                       features: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit only on known forward returns and return median/10th/90th percentiles."""
    fit = train.dropna(subset=["future_return"])
    if len(fit) < 50:
        raise RuntimeError("insufficient known returns for price model")
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(fit[features])
    x_test = imputer.transform(test[features])
    regressor = ExtraTreesRegressor(
        n_estimators=300, min_samples_leaf=12, max_features=0.7,
        random_state=43, n_jobs=-1)
    regressor.fit(x_train, fit.future_return)
    tree_predictions = np.vstack([tree.predict(x_test) for tree in regressor.estimators_])
    return (np.quantile(tree_predictions, 0.50, axis=0),
            np.quantile(tree_predictions, 0.10, axis=0),
            np.quantile(tree_predictions, 0.90, axis=0))


RESEARCH_RETURN_CANDIDATES = (
    "zero", "historical_mean", "ridge",
    "extra_trees_shrink_0.10", "extra_trees_shrink_0.25",
    "extra_trees_shrink_0.50", "extra_trees_shrink_0.75",
    "extra_trees_shrink_1.00",
)


def research_return_fit_predict(train: pd.DataFrame, test: pd.DataFrame,
                                features: list[str]) -> tuple[np.ndarray, np.ndarray,
                                                              np.ndarray, dict[str, object]]:
    """Select a research-only centre on trailing calibration data and conformalize it."""
    usable = train.dropna(subset=["future_return"])
    if len(usable) < 100:
        raise RuntimeError("insufficient known returns for research price model")
    split = max(int(len(usable) * 0.8), len(usable) - 126)
    split = min(max(split, 50), len(usable) - 30)
    fit, calibration = usable.iloc[:split], usable.iloc[split:]

    imputer = SimpleImputer(strategy="median")
    x_fit = imputer.fit_transform(fit[features])
    x_cal = imputer.transform(calibration[features])
    x_test = imputer.transform(test[features])
    y_fit = fit.future_return.to_numpy(dtype=float)
    y_cal = calibration.future_return.to_numpy(dtype=float)

    extra = ExtraTreesRegressor(
        n_estimators=300, min_samples_leaf=12, max_features=0.7,
        random_state=44, n_jobs=-1)
    extra.fit(x_fit, y_fit)
    extra_cal = extra.predict(x_cal)
    extra_test = extra.predict(x_test)

    ridge = Ridge(alpha=10.0)
    ridge.fit(x_fit, y_fit)
    mean_return = float(np.mean(y_fit))
    calibration_candidates: dict[str, np.ndarray] = {
        "zero": np.zeros(len(calibration)),
        "historical_mean": np.full(len(calibration), mean_return),
        "ridge": ridge.predict(x_cal),
    }
    test_candidates: dict[str, np.ndarray] = {
        "zero": np.zeros(len(test)),
        "historical_mean": np.full(len(test), mean_return),
        "ridge": ridge.predict(x_test),
    }
    for shrink in (0.10, 0.25, 0.50, 0.75, 1.00):
        name = f"extra_trees_shrink_{shrink:.2f}"
        calibration_candidates[name] = extra_cal * shrink
        test_candidates[name] = extra_test * shrink

    calibration_mae = {
        name: float(np.mean(np.abs(y_cal - prediction)))
        for name, prediction in calibration_candidates.items()
    }
    calibration_direction = {
        name: float(np.mean(np.sign(y_cal) == np.sign(prediction)))
        for name, prediction in calibration_candidates.items()
    }
    direction_qualified = [
        name for name in RESEARCH_RETURN_CANDIDATES
        if calibration_direction[name] >= 0.52
    ]
    selection_pool = direction_qualified or list(RESEARCH_RETURN_CANDIDATES)
    selected = min(selection_pool,
                   key=lambda name: (calibration_mae[name], name))
    selected_cal = calibration_candidates[selected]
    selected_test = test_candidates[selected]
    errors = np.abs(y_cal - selected_cal)
    quantile_level = min(1.0, math.ceil((len(errors) + 1) * 0.80) / len(errors))
    try:
        radius = float(np.quantile(errors, quantile_level, method="higher"))
    except TypeError:  # NumPy < 1.22 compatibility.
        radius = float(np.quantile(errors, quantile_level, interpolation="higher"))
    metadata = {
        "selected_model": selected,
        "calibration_samples": len(calibration),
        "calibration_mae": calibration_mae[selected],
        "candidate_mae": calibration_mae,
        "calibration_direction_accuracy": calibration_direction[selected],
        "candidate_direction_accuracy": calibration_direction,
        "direction_constraint": 0.52,
        "conformal_target_coverage": 0.80,
        "conformal_radius": radius,
        "selection_scope": "僅使用該 fold 訓練窗口尾端時間順序校準期",
    }
    return selected_test, selected_test - radius, selected_test + radius, metadata


def walk_forward(dev: pd.DataFrame, features: list[str], folds: int,
                 min_train: int, purge: int = 0,
                 model_kind: str = "logistic",
                 include_research: bool = True) -> pd.DataFrame:
    if len(dev) < min_train + folds * 20:
        raise RuntimeError(f"insufficient history: {len(dev)} usable rows; need at least {min_train + folds * 20}")
    boundaries = np.linspace(min_train, len(dev), folds + 1, dtype=int)
    out = []
    for fold in range(folds):
        train = dev.iloc[:max(1, boundaries[fold] - purge)]
        test = dev.iloc[boundaries[fold]:boundaries[fold + 1]].copy()
        if test.empty or train.label.nunique() < 2:
            continue
        test["probability"], _, _ = calibrated_fit_predict(
            train, test, features, purge, model_kind)
        (test["predicted_return"], test["predicted_return_low"],
         test["predicted_return_high"]) = return_fit_predict(train, test, features)
        if include_research:
            (test["research_predicted_return"], test["research_predicted_return_low"],
             test["research_predicted_return_high"], research_meta) = (
                research_return_fit_predict(train, test, features))
            test["research_model"] = research_meta["selected_model"]
            test["research_conformal_radius"] = research_meta["conformal_radius"]
        test["fold"] = fold + 1
        out.append(test)
    if not out:
        raise RuntimeError("walk-forward produced no valid folds")
    return pd.concat(out).sort_index()


def apply_calibrator(base: Pipeline, calibrator: LogisticRegression | None,
                     x: pd.DataFrame) -> np.ndarray:
    raw = np.clip(base.predict_proba(x)[:, 1], 1e-6, 1 - 1e-6)
    if calibrator is None:
        return raw
    return calibrator.predict_proba(np.log(raw / (1 - raw)).reshape(-1, 1))[:, 1]


def simulate(rows: pd.DataFrame, market: pd.DataFrame, threshold: float, horizon: int,
             stop_atr: float, reward_risk: float, commission_bps: float,
             tax_bps: float, slippage_bps: float,
             entry_gap_low_atr: float | None = None,
             entry_gap_high_atr: float | None = None,
             minimum_predicted_return: float | None = None,
             risk_policy: bool = False) -> list[Trade]:
    loc = {date: i for i, date in enumerate(market.index)}
    trades: list[Trade] = []
    blocked_until = -1
    for signal_date, row in rows.iterrows():
        i = loc.get(signal_date)
        if i is None or i < blocked_until or row.probability < threshold or i + 1 >= len(market):
            continue
        if (minimum_predicted_return is not None
                and ("predicted_return" not in row
                     or not np.isfinite(row.predicted_return)
                     or row.predicted_return < minimum_predicted_return)):
            continue
        position_fraction = strategy_position_fraction(str(row.regime), risk_policy)
        if position_fraction <= 0:
            continue
        entry_i = i + 1
        if entry_gap_low_atr is not None and entry_gap_high_atr is not None:
            gap_atr = (float(market.Open.iloc[entry_i]) - float(row.Close)) / float(row.ATR)
            if not entry_gap_low_atr <= gap_atr <= entry_gap_high_atr:
                continue
        entry = float(market.Open.iloc[entry_i]) * (1 + slippage_bps / 10_000)
        risk = stop_atr * float(row.ATR)
        if not np.isfinite(risk) or risk <= 0:
            continue
        stop, target = entry - risk, entry + reward_risk * risk
        exit_i = min(i + horizon, len(market) - 1)
        reason = "time"
        exit_px = float(market.Close.iloc[exit_i])
        for j in range(entry_i, exit_i + 1):
            # Conservative convention when both levels occur in the same daily bar.
            if market.Low.iloc[j] <= stop:
                exit_i, exit_px, reason = j, stop, "stop"
                break
            if market.High.iloc[j] >= target:
                exit_i, exit_px, reason = j, target, "target"
                break
        exit_px *= 1 - slippage_bps / 10_000
        gross_r = (exit_px - entry) / risk
        costs = entry * (commission_bps / 10_000) + exit_px * ((commission_bps + tax_bps) / 10_000)
        net_r = (exit_px - entry - costs) / risk
        trades.append(Trade(str(signal_date.date()), str(market.index[entry_i].date()),
                            str(market.index[exit_i].date()), entry, exit_px,
                            float(row.probability), gross_r, net_r, reason, str(row.regime),
                            ((exit_px - entry - costs) / entry) * position_fraction,
                            position_fraction))
        blocked_until = exit_i + 1
    return trades


def simulate_averaging(rows: pd.DataFrame, market: pd.DataFrame, threshold: float,
                       horizon: int, stop_atr: float, reward_risk: float,
                       commission_bps: float, tax_bps: float, slippage_bps: float,
                       add_ratio: float, trigger_atr: float = 0.75,
                       entry_gap_low_atr: float | None = None,
                       entry_gap_high_atr: float | None = None) -> tuple[list[Trade], int]:
    """Backtest one fixed-stop add after a trigger decline; R uses total risk at stop."""
    loc = {date: i for i, date in enumerate(market.index)}
    trades: list[Trade] = []
    add_count, blocked_until = 0, -1
    for signal_date, row in rows.iterrows():
        i = loc.get(signal_date)
        if i is None or i < blocked_until or row.probability < threshold or i + 1 >= len(market):
            continue
        entry_i = i + 1
        if entry_gap_low_atr is not None and entry_gap_high_atr is not None:
            gap_atr = (float(market.Open.iloc[entry_i]) - float(row.Close)) / float(row.ATR)
            if not entry_gap_low_atr <= gap_atr <= entry_gap_high_atr:
                continue
        entry = float(market.Open.iloc[entry_i]) * (1 + slippage_bps / 10_000)
        atr = float(row.ATR)
        risk = stop_atr * atr
        if not np.isfinite(risk) or risk <= 0:
            continue
        stop, target = entry - risk, entry + reward_risk * risk
        add_trigger = entry - trigger_atr * atr
        exit_i = min(i + horizon, len(market) - 1)
        exit_px, reason, added = float(market.Close.iloc[exit_i]), "time", False
        for j in range(entry_i, exit_i + 1):
            # Conservative daily-bar rule: an add is filled before a same-bar stop.
            if not added and add_ratio > 0 and market.Low.iloc[j] <= add_trigger:
                added = True
                add_count += 1
            if market.Low.iloc[j] <= stop:
                exit_i, exit_px, reason = j, stop, "stop"
                break
            if market.High.iloc[j] >= target:
                exit_i, exit_px, reason = j, target, "target"
                break
        exit_px *= 1 - slippage_bps / 10_000
        qty = 1 + (add_ratio if added else 0)
        add_value = add_ratio * add_trigger if added else 0
        gross_pnl = qty * exit_px - entry - add_value
        costs = entry * commission_bps / 10_000
        if added:
            costs += add_value * commission_bps / 10_000
        costs += qty * exit_px * (commission_bps + tax_bps) / 10_000
        max_risk = (entry - stop) + (add_ratio * (add_trigger - stop) if added else 0)
        gross_r, net_r = gross_pnl / max_risk, (gross_pnl - costs) / max_risk
        weighted_entry = (entry + add_value) / qty
        trades.append(Trade(
            str(signal_date.date()), str(market.index[entry_i].date()),
            str(market.index[exit_i].date()), weighted_entry, exit_px,
            float(row.probability), gross_r, net_r,
            ("add+" if added else "") + reason, str(row.regime),
            (gross_pnl - costs) / (entry + add_value)))
        blocked_until = exit_i + 1
    return trades, add_count


def trade_metrics(trades: list[Trade]) -> dict[str, float | int | None]:
    if not trades:
        return {"trades": 0, "win_rate": None, "ev_r": None, "profit_factor": None,
                "max_drawdown_r": None, "total_r": 0.0, "wins": 0, "losses": 0,
                "average_win_r": None, "average_loss_r": None, "payoff_ratio": None,
                "max_drawdown_pct": None, "total_return_pct": None}
    r = np.array([t.net_r for t in trades], dtype=float)
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    equity = np.r_[0.0, np.cumsum(r)]
    dd = np.maximum.accumulate(equity) - equity
    avg_win = float(r[r > 0].mean()) if (r > 0).any() else None
    avg_loss = float(r[r <= 0].mean()) if (r <= 0).any() else None
    returns = np.array([t.net_return for t in trades], dtype=float)
    compounded = np.r_[1.0, np.cumprod(1 + returns)]
    peak = np.maximum.accumulate(compounded)
    drawdown_pct = np.divide(peak - compounded, peak, out=np.zeros_like(peak), where=peak > 0)
    return {"trades": len(trades), "win_rate": float((r > 0).mean()), "ev_r": float(r.mean()),
            "profit_factor": float(gains / losses) if losses > 0 else (float("inf") if gains > 0 else None),
            "max_drawdown_r": float(dd.max()), "total_r": float(r.sum()),
            "wins": int((r > 0).sum()), "losses": int((r <= 0).sum()),
            "average_win_r": avg_win, "average_loss_r": avg_loss,
            "payoff_ratio": (avg_win / abs(avg_loss)) if avg_win is not None and avg_loss else None,
            "max_drawdown_pct": float(drawdown_pct.max()),
            "total_return_pct": float(compounded[-1] - 1)}


def formal_validation(oos_rows: pd.DataFrame, market: pd.DataFrame,
                      oos_metrics: dict, final_metrics: dict,
                      args: argparse.Namespace) -> dict[str, object]:
    """Apply the mandatory validation gates without using future data for selection."""
    fold_metrics: list[dict[str, object]] = []
    for fold, rows in oos_rows.groupby("fold", sort=True):
        metrics = trade_metrics(simulate(
            rows, market, args.threshold, args.horizon, args.stop_atr,
            args.reward_risk, args.commission_bps, args.tax_bps,
            args.slippage_bps, args.entry_gap_low_atr, args.entry_gap_high_atr,
            getattr(args, "minimum_predicted_return", None),
            getattr(args, "risk_policy", False)))
        fold_metrics.append({"fold": int(fold), **metrics})
    fold_win_rates = [m["win_rate"] for m in fold_metrics if m["win_rate"] is not None]
    all_folds_have_trades = (len(fold_metrics) >= 2
                             and len(fold_win_rates) == len(fold_metrics))
    fold_std_pp = (float(np.std(fold_win_rates, ddof=0) * 100)
                   if all_folds_have_trades else None)
    win_rate_gap_pp = (abs(float(final_metrics["win_rate"] - oos_metrics["win_rate"])) * 100
                       if final_metrics["win_rate"] is not None
                       and oos_metrics["win_rate"] is not None else None)
    checks = {
        "effective_trades_at_least_30": oos_metrics["trades"] >= 30,
        "walk_forward_winrate_std_at_most_15pp":
            all_folds_have_trades and fold_std_pp is not None and fold_std_pp <= 15,
        "profit_factor_at_least_1_2":
            oos_metrics["profit_factor"] is not None and oos_metrics["profit_factor"] >= 1.2,
        "max_drawdown_at_most_25pct":
            oos_metrics["max_drawdown_pct"] is not None
            and oos_metrics["max_drawdown_pct"] <= 0.249,
        "oos_vs_development_winrate_gap_at_most_10pp":
            win_rate_gap_pp is not None and win_rate_gap_pp <= 10,
        "walk_forward_and_independent_oos_completed":
            bool(fold_metrics) and final_metrics["trades"] > 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "effective_trades": oos_metrics["trades"],
        "fold_winrate_std_pp": fold_std_pp,
        "all_folds_have_trades": all_folds_have_trades,
        "profit_factor": oos_metrics["profit_factor"],
        "max_drawdown_pct": oos_metrics["max_drawdown_pct"],
        "development_walk_forward_winrate": oos_metrics["win_rate"],
        "independent_oos_winrate": final_metrics["win_rate"],
        "oos_vs_development_winrate_gap_pp": win_rate_gap_pp,
        "folds": fold_metrics,
        "max_drawdown_limit": 0.249,
        "drawdown_basis": ("以配置給 00631L 的資金為分母；一般市場投入 85%、"
                           "多頭高波動投入 50%、空頭高波動不進場，扣成本後逐筆複利"
                           if getattr(args, "risk_policy", False) else
                           "每筆交易將配置給 00631L 的資金全額投入，扣成本後逐筆複利"),
        "comparison_basis": "開發期 Walk-Forward OOS 與完全隔離 Final Test 比較",
    }


def strict_entry_validation(data: pd.DataFrame, features: list[str], market: pd.DataFrame,
                            args: argparse.Namespace) -> dict[str, object]:
    """Reject an entry range unless it survives several chronological validations."""
    runs: list[dict[str, object]] = []
    for final_fraction in (0.15, 0.20, 0.25):
        development = data.iloc[:int(len(data) * (1 - final_fraction))]
        for folds in (4, 5, 6):
            rows = walk_forward(development, features, folds, args.min_train,
                                args.horizon, args.model, include_research=False)
            normal = trade_metrics(simulate(
                rows, market, args.threshold, args.horizon, args.stop_atr,
                args.reward_risk, args.commission_bps, args.tax_bps,
                args.slippage_bps, args.entry_gap_low_atr,
                args.entry_gap_high_atr,
                getattr(args, "minimum_predicted_return", None),
                getattr(args, "risk_policy", False)))
            stressed = trade_metrics(simulate(
                rows, market, args.threshold, args.horizon, args.stop_atr,
                args.reward_risk, args.commission_bps * 2, args.tax_bps * 2,
                args.slippage_bps * 2, args.entry_gap_low_atr,
                args.entry_gap_high_atr,
                getattr(args, "minimum_predicted_return", None),
                getattr(args, "risk_policy", False)))
            runs.append({"final_fraction": final_fraction, "folds": folds,
                         "normal_cost": normal, "double_cost": stressed})

    def median_metric(cost: str, key: str) -> float | None:
        values = [run[cost][key] for run in runs
                  if run[cost][key] is not None and np.isfinite(run[cost][key])]
        return float(np.median(values)) if values else None

    positive_runs = sum((run["normal_cost"]["ev_r"] or -math.inf) > 0 for run in runs)
    summary = {
        "runs": len(runs),
        "median_trades": median_metric("normal_cost", "trades"),
        "median_win_rate": median_metric("normal_cost", "win_rate"),
        "median_ev_r": median_metric("normal_cost", "ev_r"),
        "median_profit_factor": median_metric("normal_cost", "profit_factor"),
        "median_max_drawdown_r": median_metric("normal_cost", "max_drawdown_r"),
        "positive_run_ratio": positive_runs / len(runs),
        "double_cost_median_ev_r": median_metric("double_cost", "ev_r"),
        "double_cost_median_profit_factor": median_metric("double_cost", "profit_factor"),
    }
    checks = {
        "median_trades_at_least_60": (summary["median_trades"] or 0) >= 60,
        "median_win_rate_at_least_60pct": (summary["median_win_rate"] or 0) >= 0.60,
        "median_ev_r_at_least_0_15": (summary["median_ev_r"] or -math.inf) >= 0.15,
        "median_profit_factor_at_least_1_40": (summary["median_profit_factor"] or 0) >= 1.40,
        "positive_run_ratio_at_least_70pct": summary["positive_run_ratio"] >= 0.70,
        "median_drawdown_at_most_8r": (summary["median_max_drawdown_r"] or math.inf) <= 8,
        "double_cost_ev_positive": (summary["double_cost_median_ev_r"] or -math.inf) > 0,
        "double_cost_profit_factor_at_least_1_10":
            (summary["double_cost_median_profit_factor"] or 0) >= 1.10,
    }
    return {"passed": all(checks.values()), "checks": checks,
            "summary": summary, "runs_detail": runs}


def grouped_metrics(trades: list[Trade], key: str) -> dict[str, dict]:
    groups: dict[str, list[Trade]] = {}
    for trade in trades:
        value = trade.entry_date[:4] if key == "year" else trade.regime
        groups.setdefault(value, []).append(trade)
    return {name: trade_metrics(group) for name, group in sorted(groups.items())}


def strategy_health(trades: list[Trade], n: int = 20) -> dict[str, float | int | str | None]:
    recent = trades[-n:]
    m = trade_metrics(recent)
    streak = worst = 0
    for t in recent:
        streak = streak + 1 if t.net_r <= 0 else 0
        worst = max(worst, streak)
    prior = trades[-2 * n:-n]
    prior_ev = trade_metrics(prior)["ev_r"]
    drift = None if prior_ev is None or m["ev_r"] is None else float(m["ev_r"] - prior_ev)
    status = "insufficient"
    if len(recent) >= 10:
        status = "healthy" if (m["ev_r"] or 0) >= 0.05 and (m["profit_factor"] or 0) >= 1.10 else "degraded"
    return {"status": status, "sample": len(recent), "profit_factor": m["profit_factor"],
            "average_r": m["ev_r"], "max_losing_streak": worst, "ev_drift_vs_prior_20": drift}


def trading_gate(metrics: dict, annual: dict, health: dict) -> dict[str, object]:
    eligible_years = [v for v in annual.values() if v["trades"] >= 5]
    positive_ratio = (sum((v["ev_r"] or 0) > 0 for v in eligible_years) / len(eligible_years)
                      if eligible_years else 0.0)
    checks = {
        "trades_at_least_30": metrics["trades"] >= 30,
        "ev_r_at_least_0_10": (metrics["ev_r"] or -math.inf) >= 0.10,
        "profit_factor_at_least_1_20": (metrics["profit_factor"] or 0) >= 1.20,
        "positive_year_ratio_at_least_60pct": positive_ratio >= 0.60,
        "health_not_degraded": health["status"] != "degraded",
    }
    return {"passed": all(checks.values()), "checks": checks,
            "positive_year_ratio": positive_ratio}


def holding_analysis(shares: int, average_cost: float, latest_close: float,
                     atr: float, raw_signal: bool, gate_passed: bool,
                     stop_atr: float, reward_risk: float,
                     commission_bps: float, tax_bps: float,
                     slippage_bps: float, add_shares: int = 100,
                     volume_ratio20: float | None = None,
                     horizon: int = 5) -> dict[str, object]:
    """Value an existing long position and produce a deterministic risk action."""
    sell_cost_rate = (commission_bps + tax_bps + slippage_bps) / 10_000
    cost_basis = shares * average_cost
    estimated_proceeds = shares * latest_close * (1 - sell_cost_rate)
    pnl = estimated_proceeds - cost_basis
    break_even = average_cost / (1 - sell_cost_rate)
    risk = stop_atr * atr
    stop = average_cost - risk
    target = average_cost + reward_risk * risk

    if latest_close <= stop:
        action = "停損"
        reason = "最新價格已低於持股停損價"
    elif raw_signal and gate_passed:
        action = "續抱"
        reason = "最新機率達標且歷史交易驗證閘門通過"
    else:
        action = "減碼"
        reason = "目前持股模型未同時通過機率與交易驗證"

    def exit_pnl(price: float) -> float:
        return shares * price * (1 - sell_cost_rate) - cost_basis

    # Historically selected on development data and confirmed on the held-out
    # period: one add of up to 50% after a 0.75 ATR decline, while retaining the
    # original stop and target. --add-shares is a user-controlled hard cap.
    optimized_add_shares = max(1, int(math.floor(shares * 0.50)))
    optimized_add_shares = min(optimized_add_shares, add_shares)
    add_trigger = average_cost - 0.75 * atr
    add_price = add_trigger
    buy_cost_rate = (commission_bps + slippage_bps) / 10_000
    add_total_cost = optimized_add_shares * add_price * (1 + buy_cost_rate)
    after_shares = shares + optimized_add_shares
    after_cost_basis = cost_basis + add_total_cost
    after_average = after_cost_basis / after_shares
    after_break_even = after_average / (1 - sell_cost_rate)
    after_stop = stop
    after_target = target

    def after_exit_pnl(price: float) -> float:
        return after_shares * price * (1 - sell_cost_rate) - after_cost_basis

    can_add = (raw_signal and gate_passed and latest_close > stop
               and latest_close <= add_trigger)
    if latest_close > add_trigger:
        add_reason = f"尚未跌至加碼觸發價 {add_trigger:.2f}"
    elif latest_close <= stop:
        add_reason = "價格已跌破原持股停損，不應再攤平"
    elif not (raw_signal and gate_passed):
        add_reason = "模型訊號或交易驗證閘門未通過"
    else:
        add_reason = "模型條件通過，且價格已到達0.75 ATR加碼區"

    # Dynamic staged plan for an existing position. This is an operational risk
    # framework, not part of the historical single-exit strategy backtest.
    first_defense = stop
    final_stop = latest_close - stop_atr * atr
    rebound_one = latest_close + 0.75 * atr
    rebound_two = break_even
    strong_low = latest_close + reward_risk * stop_atr * atr
    strong_high = max(target, strong_low)
    volume_confirmed = bool(volume_ratio20 is not None and np.isfinite(volume_ratio20)
                            and volume_ratio20 >= 1.5)
    operation_plan = {
        "first_defense": {"low": first_defense, "high": first_defense,
                          "advice": "收盤跌破，次一交易日減碼50%"},
        "final_stop": {"low": final_stop - 0.10 * atr,
                       "high": final_stop + 0.10 * atr,
                       "advice": "收盤跌破區間下緣，剩餘部位退出"},
        "rebound_target_one": {"low": rebound_one - 0.10 * atr,
                               "high": rebound_one + 0.10 * atr,
                               "advice": "反彈至此，減碼30%～50%"},
        "rebound_target_two": {"low": break_even - 0.10 * atr,
                               "high": break_even + 0.10 * atr,
                               "advice": "接近含成本回本價，再減碼"},
        "strong_take_profit": {"low": strong_low, "high": strong_high,
                               "advice": "需先突破反彈目標二且成交量達20日均量1.5倍",
                               "volume_confirmed_now": volume_confirmed},
        "volume_ratio20": volume_ratio20,
        "backtested": False,
    }

    return {
        "shares": shares, "average_cost": average_cost, "cost_basis": cost_basis,
        "estimated_market_value_after_sell_costs": estimated_proceeds,
        "estimated_pnl": pnl, "estimated_return": pnl / cost_basis,
        "break_even_price_after_sell_costs": break_even,
        "action": action, "reason": reason, "holding_stop_price": stop,
        "holding_target_price": target,
        "estimated_pnl_at_stop": exit_pnl(stop),
        "estimated_pnl_at_target": exit_pnl(target),
        "sell_cost_rate": sell_cost_rate,
        "averaging_action": "可考慮加碼" if can_add else "不建議加碼",
        "averaging_reason": add_reason,
        "averaging_rule": "跌0.75 ATR後最多加原持股50%，只加一次，不放寬原停損／停利",
        "averaging_timing": f"本次訊號後{horizon}個交易日內，盤中第一次觸及加碼價",
        "averaging_cancel": "觸及第一防守價、模型閘門失效或超過有效期限即取消",
        "averaging_valid_days": horizon,
        "add_trigger_price": add_trigger,
        "suggested_add_price": add_price, "add_shares": optimized_add_shares,
        "add_shares_cap": add_shares,
        "add_estimated_cost": add_total_cost,
        "after_add_shares": after_shares, "after_add_average_cost": after_average,
        "after_add_break_even_price": after_break_even,
        "after_add_stop_price": after_stop, "after_add_target_price": after_target,
        "after_add_pnl_at_stop": after_exit_pnl(after_stop),
        "after_add_pnl_at_target": after_exit_pnl(after_target),
        "operation_plan": operation_plan,
    }


def probability_metrics(rows: pd.DataFrame, threshold: float) -> dict[str, float | int | None]:
    signal = rows.probability >= threshold
    return {"predictions": len(rows), "signals": int(signal.sum()),
            "buy_precision": float(rows.loc[signal, "label"].mean()) if signal.any() else None,
            "brier": float(brier_score_loss(rows.label.astype(int), rows.probability))}


def validated_score(prob: dict, tm: dict, annual: dict, regimes: dict) -> float:
    precision = prob["buy_precision"] or 0
    brier_quality = max(0.0, 1 - (prob["brier"] or 1) / 0.25)
    ev_quality = float(np.clip(((tm["ev_r"] or 0) + 0.25) / 0.75, 0, 1))
    pf = tm["profit_factor"] or 0
    pf_quality = float(np.clip(pf / 2, 0, 1)) if np.isfinite(pf) else 1.0
    enough = [v for v in annual.values() if v["trades"] >= 5]
    annual_stability = sum((v["ev_r"] or 0) > 0 for v in enough) / len(enough) if enough else 0
    rg = [v for v in regimes.values() if v["trades"] >= 5]
    regime_stability = sum((v["ev_r"] or 0) > 0 for v in rg) / len(rg) if rg else 0
    return round(100 * (0.25 * precision + 0.15 * brier_quality + 0.25 * ev_quality +
                        0.15 * pf_quality + 0.10 * annual_stability + 0.10 * regime_stability), 1)


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return "Infinity" if value > 0 else None
    return value


MODEL_VERSION = "20260828.4"
STRATEGY_VERSION = "20260828.3"
DATABASE_SCHEMA_VERSION = "20260831.1"
TAIPEI = ZoneInfo("Asia/Taipei")

CANDIDATE_20260819_3 = {
    "threshold": 0.15,
    "entry_gap_low_atr": -0.25,
    "entry_gap_high_atr": 0.25,
    "stop_atr": 1.5,
    "reward_risk": 2.5,
    "minimum_predicted_return": 0.005,
    "selection_scope": "最後15%獨立期間之前的預先限制候選集合",
}

RISK_POLICY_20260828_2 = {
    "capital_denominator": "配置給 00631L 的資金",
    "normal_fraction": 0.85,
    "bull_high_vol_fraction": 0.50,
    "bear_high_vol_fraction": 0.0,
    "max_drawdown_limit": 0.249,
    "threshold_selection": "固定沿用預先選定的 0.15；不使用 OOS 事後挑選",
}


def strategy_position_fraction(regime: str, enabled: bool) -> float:
    """Return the precommitted fraction of 00631L-allocated capital."""
    if not enabled:
        return 1.0
    if regime == "bear_high_vol":
        return RISK_POLICY_20260828_2["bear_high_vol_fraction"]
    if regime == "bull_high_vol":
        return RISK_POLICY_20260828_2["bull_high_vol_fraction"]
    return RISK_POLICY_20260828_2["normal_fraction"]


def validate_entry_gap_atr(low: float, high: float) -> None:
    """Validate an ordered, finite next-open displacement interval in ATR."""
    if not math.isfinite(low) or not math.isfinite(high) or low >= high:
        raise ValueError("entry gap ATR values must be finite and satisfy low < high")


def return_forecast_metrics(rows: pd.DataFrame,
                            prefix: str = "predicted_return") -> dict[str, float | int | bool | None]:
    low_col, high_col = f"{prefix}_low", f"{prefix}_high"
    usable = rows.dropna(subset=["future_return", prefix, low_col, high_col])
    if usable.empty:
        return {"samples": 0, "mae": None, "naive_zero_mae": None,
                "direction_accuracy": None, "interval_80_coverage": None,
                "checks": {}, "passed": False}
    actual = usable.future_return.to_numpy(dtype=float)
    predicted = usable[prefix].to_numpy(dtype=float)
    mae = float(np.mean(np.abs(actual - predicted)))
    naive = float(np.mean(np.abs(actual)))
    direction = float(np.mean(np.sign(actual) == np.sign(predicted)))
    coverage = float(np.mean(
        (actual >= usable[low_col].to_numpy(dtype=float))
        & (actual <= usable[high_col].to_numpy(dtype=float))))
    checks = {
        "samples_at_least_30": len(usable) >= 30,
        "mae_better_than_zero_baseline": mae <= naive,
        "direction_accuracy_at_least_52pct": direction >= 0.52,
        "interval_coverage_between_60_and_95pct": 0.60 <= coverage <= 0.95,
    }
    return {"samples": len(usable), "mae": mae, "naive_zero_mae": naive,
            "direction_accuracy": direction, "interval_80_coverage": coverage,
            "checks": checks, "passed": all(checks.values())}


def research_price_forecast(latest_close: float, predicted_return: float,
                            return_low: float, return_high: float,
                            horizon: int, valid_until: str,
                            development_metrics: dict,
                            independent_metrics: dict) -> dict[str, object]:
    """Expose research prices only when both return-model validations pass."""
    validation_passed = bool(
        development_metrics.get("passed") and independent_metrics.get("passed"))
    finite_inputs = all(np.isfinite(value) for value in (
        latest_close, predicted_return, return_low, return_high))
    available = bool(validation_passed and finite_inputs and return_low <= return_high)
    return {
        "available": available,
        "research_only": True,
        "not_trading_advice": True,
        "horizon_trading_days": horizon,
        "valid_until": valid_until,
        "predicted_price": latest_close * (1 + predicted_return) if available else None,
        "predicted_price_low": latest_close * (1 + return_low) if available else None,
        "predicted_price_high": latest_close * (1 + return_high) if available else None,
        "predicted_return": predicted_return if available else None,
        "development_oos": development_metrics,
        "independent_oos": independent_metrics,
        "validation_passed": validation_passed,
        "unavailable_reason": (None if available else
                               "價格模型自身驗證未通過，不顯示中央預測價格或區間"),
    }


def _create_prediction_schema(con: sqlite3.Connection) -> None:
    # Keep every DDL statement inside the caller's explicit transaction.
    # sqlite3.Connection.executescript() commits a pending transaction first.
    statements = (
        """CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            predicted_at TEXT NOT NULL,
            timezone TEXT NOT NULL DEFAULT 'Asia/Taipei',
            symbol TEXT NOT NULL CHECK(symbol = '00631L'),
            market_price REAL NOT NULL,
            action TEXT NOT NULL CHECK(action IN ('買進','持有','賣出','不交易')),
            buy_price REAL, buy_range_low REAL, buy_range_high REAL,
            position_sizing TEXT, stop_loss REAL, take_profit_1 REAL,
            take_profit_2 REAL, risk_reward_ratio REAL,
            model_probability REAL, backtest_winrate REAL,
            valid_until TEXT NOT NULL, model_version TEXT NOT NULL,
            strategy_version TEXT NOT NULL, indicators_snapshot TEXT,
            validation_snapshot TEXT, data_source_snapshot TEXT,
            reasoning TEXT,
            market_state TEXT CHECK(market_state IN ('多頭','空頭','盤整')),
            created_at TEXT NOT NULL DEFAULT (
                strftime('%Y-%m-%dT%H:%M:%S', 'now', '+8 hours') || '+08:00'
            ),
            CHECK(model_probability IS NULL OR model_probability BETWEEN 0 AND 1),
            CHECK(backtest_winrate IS NULL OR backtest_winrate BETWEEN 0 AND 1),
            CHECK(buy_range_low IS NULL OR buy_range_high IS NULL OR buy_range_low <= buy_range_high)
        )""",
        """CREATE TABLE IF NOT EXISTS prediction_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER NOT NULL REFERENCES predictions(id),
            actual_high REAL, actual_low REAL, actual_close REAL,
            hit_stop_loss INTEGER CHECK(hit_stop_loss IN (0,1)),
            hit_take_profit_1 INTEGER CHECK(hit_take_profit_1 IN (0,1)),
            hit_take_profit_2 INTEGER CHECK(hit_take_profit_2 IN (0,1)),
            actual_return_pct REAL, trade_result TEXT,
            prediction_success INTEGER CHECK(prediction_success IN (0,1)),
            resolved_at TEXT NOT NULL, UNIQUE(prediction_id)
        )""",
        """CREATE TRIGGER IF NOT EXISTS prevent_prediction_update
        BEFORE UPDATE ON predictions BEGIN
            SELECT RAISE(ABORT, '原始預測紀錄不得修改，請新增一筆紀錄');
        END""",
        """CREATE TRIGGER IF NOT EXISTS prevent_prediction_delete
        BEFORE DELETE ON predictions BEGIN
            SELECT RAISE(ABORT, '原始預測紀錄不得刪除');
        END""",
    )
    for statement in statements:
        con.execute(statement)


def _market_state_zh(regime: object) -> str:
    value = str(regime or "")
    if value.startswith("bull"):
        return "多頭"
    if value.startswith("bear"):
        return "空頭"
    return "盤整"


def _ensure_taipei_iso(value: object) -> str:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(TAIPEI)
    else:
        stamp = stamp.tz_convert(TAIPEI)
    return stamp.isoformat()


def _migrate_legacy_database(database: str) -> str | None:
    """Preserve a timestamped byte-for-byte backup, then migrate the legacy table."""
    path = Path(database)
    if not path.exists():
        return None
    with sqlite3.connect(path) as check:
        tables = {row[0] for row in check.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if "predictions" not in tables:
            return None
        columns = {row[1] for row in check.execute("PRAGMA table_info(predictions)")}
        if "symbol" in columns and "data_source_snapshot" in columns:
            return None
        if not {"ticker", "market_date"} <= columns:
            raise RuntimeError("unknown predictions schema; automatic migration refused")
    timestamp = dt.datetime.now(TAIPEI).strftime("%Y%m%dT%H%M%S%z")
    backup = path.with_name(f"{path.name}.legacy-{timestamp}.bak")
    # SQLite's backup API includes committed WAL pages and creates a consistent
    # snapshot; copying only the main file can silently omit recent commits.
    with sqlite3.connect(path) as source, sqlite3.connect(backup) as destination:
        source.backup(destination)
    with sqlite3.connect(path) as con:
        con.execute("PRAGMA foreign_keys = ON")
        con.execute("PRAGMA busy_timeout = 5000")
        try:
            con.execute("BEGIN IMMEDIATE")
            con.execute("ALTER TABLE predictions RENAME TO predictions_legacy_20260831_1")
            _create_prediction_schema(con)
            con.row_factory = sqlite3.Row
            rows = con.execute("SELECT * FROM predictions_legacy_20260831_1 ORDER BY id").fetchall()
            for row in rows:
                market_state = _market_state_zh(row["market_regime"])
                source = {"market_date": row["market_date"], "timezone": "Asia/Taipei",
                          "migration": "legacy-20260831.1", "source": "legacy snapshot"}
                validation = {"migration": "legacy validation snapshot unavailable",
                              "backtest_winrate": row["backtest_win_rate"]}
                con.execute("""INSERT INTO predictions (
                    id,predicted_at,timezone,symbol,market_price,action,buy_price,
                    buy_range_low,buy_range_high,stop_loss,take_profit_1,take_profit_2,
                    risk_reward_ratio,model_probability,backtest_winrate,valid_until,
                    model_version,strategy_version,indicators_snapshot,
                    validation_snapshot,data_source_snapshot,reasoning,market_state,created_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (row["id"], _ensure_taipei_iso(row["predicted_at"]), "Asia/Taipei",
                     "00631L", row["market_price"], row["action"], row["suggested_entry"],
                     row["entry_low"], row["entry_high"], row["stop_price"],
                     row["take_profit_1"], row["take_profit_2"], None,
                     row["model_probability"], row["backtest_win_rate"], row["valid_until"],
                     row["model_version"], row["strategy_version"], row["indicators_json"],
                     json.dumps(validation, ensure_ascii=False),
                     json.dumps(source, ensure_ascii=False), row["reason"], market_state,
                     _ensure_taipei_iso(row["predicted_at"])))
                if row["settled_at"] is not None:
                    con.execute("""INSERT INTO prediction_outcomes (
                        prediction_id,actual_high,actual_low,actual_close,hit_stop_loss,
                        hit_take_profit_1,hit_take_profit_2,actual_return_pct,trade_result,
                        prediction_success,resolved_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                        (row["id"], row["actual_high"], row["actual_low"], row["actual_close"],
                         row["stop_touched"], row["target1_touched"], row["target2_touched"],
                         row["actual_return"], row["trade_result"], row["prediction_success"],
                         _ensure_taipei_iso(row["settled_at"])))
            con.commit()
        except Exception:
            con.rollback()
            raise
    return str(backup)


def _settle_expired_predictions(con: sqlite3.Connection, market: pd.DataFrame,
                                latest_date: pd.Timestamp) -> int:
    con.row_factory = sqlite3.Row
    pending = con.execute("""SELECT p.* FROM predictions p
        LEFT JOIN prediction_outcomes o ON o.prediction_id=p.id
        WHERE o.id IS NULL AND p.valid_until<=?""", (str(latest_date.date()),)).fetchall()
    settled = 0
    for row in pending:
        try:
            source = json.loads(row["data_source_snapshot"] or "{}")
        except json.JSONDecodeError:
            source = {}
        market_date = source.get("market_date")
        if not market_date:
            continue
        bars = market.loc[(market.index > pd.Timestamp(market_date)) &
                          (market.index <= pd.Timestamp(row["valid_until"]))]
        if bars.empty:
            continue
        high, low, close = float(bars.High.max()), float(bars.Low.min()), float(bars.Close.iloc[-1])
        stop_hit = target1_hit = target2_hit = False
        ambiguous = False
        for _, bar in bars.iterrows():
            day_stop = row["stop_loss"] is not None and float(bar.Low) <= row["stop_loss"]
            day_t1 = row["take_profit_1"] is not None and float(bar.High) >= row["take_profit_1"]
            day_t2 = row["take_profit_2"] is not None and float(bar.High) >= row["take_profit_2"]
            if day_stop and (day_t1 or day_t2):
                stop_hit, target1_hit, target2_hit, ambiguous = True, day_t1, day_t2, True
                break
            stop_hit, target1_hit, target2_hit = stop_hit or day_stop, target1_hit or day_t1, target2_hit or day_t2
            if day_stop or day_t2:
                break
        basis = row["buy_price"] if row["buy_price"] is not None else row["market_price"]
        actual_return = close / basis - 1 if basis else None
        if row["action"] == "不交易":
            result_text, success = "不交易到期", None
        elif ambiguous:
            result_text, success = "停損停利順序無法判定（保守視為失敗）", 0
        elif stop_hit:
            result_text, success = "停損", 0
        elif target2_hit:
            result_text, success = "第二停利", 1
        elif target1_hit:
            result_text, success = "第一停利", 1
        else:
            result_text, success = "到期", int(actual_return is not None and actual_return > 0)
        con.execute("""INSERT INTO prediction_outcomes (
            prediction_id,actual_high,actual_low,actual_close,hit_stop_loss,
            hit_take_profit_1,hit_take_profit_2,actual_return_pct,trade_result,
            prediction_success,resolved_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (row["id"], high, low, close, int(stop_hit), int(target1_hit), int(target2_hit),
             actual_return, result_text, success, dt.datetime.now(TAIPEI).isoformat()))
        settled += 1
    return settled


def record_prediction(database: str, result: dict, latest: pd.Series,
                      market: pd.DataFrame, args: argparse.Namespace) -> int:
    """Append an immutable snapshot and settle outcomes in a separate table."""
    market_price = result.get("latest_price")
    if not isinstance(market_price, (int, float)) or not np.isfinite(market_price) or market_price <= 0:
        raise ValueError("market_price must be a finite positive number; prediction was not recorded")
    _migrate_legacy_database(database)
    con = sqlite3.connect(database)
    con.execute("PRAGMA journal_mode = WAL")
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA busy_timeout = 5000")
    try:
        con.execute("BEGIN IMMEDIATE")
        _create_prediction_schema(con)
        _settle_expired_predictions(con, market, pd.Timestamp(result["latest_date"]))
        plan, signal = result["execution_plan"], bool(result["signal"])
        indicators = {name: clean_json(float(latest[name])) for name in (
            "rsi14", "kd_k", "kd_d", "macd", "macd_signal", "macd_hist",
            "volume_z20", "volume_ratio_5_20", "trend_20_60", "ATR",
            "support20_gap", "resistance20_gap", "support60_gap", "resistance60_gap"
        ) if name in latest and pd.notna(latest[name])}
        source = dict(result["data_source_snapshot"])
        source["market_date"] = result["latest_date"]
        source["database_schema_version"] = DATABASE_SCHEMA_VERSION
        reason = ("正式驗證、訊號、交易閘門與嚴格進場驗證全部通過" if signal else
                  "正式驗證、機率、交易閘門或嚴格進場驗證未全部通過")
        cur = con.execute("""INSERT INTO predictions (
            predicted_at,timezone,symbol,market_price,action,buy_price,buy_range_low,
            buy_range_high,position_sizing,stop_loss,take_profit_1,take_profit_2,
            risk_reward_ratio,model_probability,backtest_winrate,valid_until,
            model_version,strategy_version,indicators_snapshot,validation_snapshot,
            data_source_snapshot,reasoning,market_state)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (dt.datetime.now(TAIPEI).isoformat(), "Asia/Taipei", "00631L", market_price,
             result["action"], plan["suggested_entry"] if signal else None,
             plan["entry_low"] if signal else None, plan["entry_high"] if signal else None,
             json.dumps(clean_json({"tranches": plan.get("tranches", []),
                                    "capital_denominator": plan.get("capital_denominator")}),
                        ensure_ascii=False),
             plan["stop"] if signal else None, plan["take_profit_1"] if signal else None,
             plan["take_profit_2"] if signal else None,
             plan.get("reward_risk_2") if signal else None, result["latest_probability"],
             result["oos_trading"]["win_rate"], str(result["valid_until"]),
             result["model_version"], result["strategy_version"],
             json.dumps(clean_json(indicators), ensure_ascii=False),
             json.dumps(clean_json(result["validation_snapshot"]), ensure_ascii=False),
             json.dumps(clean_json(source), ensure_ascii=False), reason,
             _market_state_zh(latest.get("regime", "unknown"))))
        prediction_id = int(cur.lastrowid)
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
    return prediction_id


def fmt(value, digits=3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float) and math.isinf(value):
        return "∞"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def pct(value) -> str:
    return "無資料" if value is None else f"{value * 100:.1f}%"


def main() -> int:
    args = parse_args()
    if args.threshold is None:
        args.threshold = 0.22 if args.model == "extra-trees" else 0.70
    if args.model == "extra-trees" and args.feature_set == "all":
        for name, value in CANDIDATE_20260819_3.items():
            if name != "selection_scope":
                setattr(args, name, value)
    args.risk_policy = bool(
        args.ticker.upper() in {"00631L", "00631L.TW"}
        and args.model == "extra-trees" and args.feature_set == "all")
    if not 0.10 <= args.final_test <= 0.40:
        raise SystemExit("--final-test must be between 0.10 and 0.40")
    if args.horizon < 1 or args.folds < 2 or not 0 < args.threshold < 1:
        raise SystemExit("invalid horizon, folds, or threshold")
    if (args.shares is None) != (args.average_cost is None):
        raise SystemExit("--shares and --average-cost must be provided together")
    if args.shares is not None and (args.shares <= 0 or args.average_cost <= 0):
        raise SystemExit("--shares and --average-cost must be positive")
    if args.add_shares <= 0:
        raise SystemExit("--add-shares must be positive")
    try:
        validate_entry_gap_atr(args.entry_gap_low_atr, args.entry_gap_high_atr)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    context_symbols = list(args.context) + ([args.futures_symbol] if args.futures_symbol else [])
    primary, contexts, skipped = download_data(args.ticker, context_symbols, args.period)
    data, features = build_dataset(primary, contexts, args.horizon, args.target,
                                   args.adverse, args.feature_set)
    usable = data.dropna(subset=["label", "ATR"]).copy()
    cut = int(len(usable) * (1 - args.final_test))
    dev, final = usable.iloc[:cut], usable.iloc[cut:]
    oos = walk_forward(dev, features, args.folds, args.min_train,
                       args.horizon, args.model)
    oos_trades = simulate(oos, primary, args.threshold, args.horizon, args.stop_atr,
                          args.reward_risk, args.commission_bps, args.tax_bps,
                          args.slippage_bps, args.entry_gap_low_atr,
                          args.entry_gap_high_atr, args.minimum_predicted_return,
                          args.risk_policy)
    oos_prob, oos_tm = probability_metrics(oos, args.threshold), trade_metrics(oos_trades)
    trade_oos_return_forecast = return_forecast_metrics(oos)
    oos_return_forecast = return_forecast_metrics(oos, "research_predicted_return")
    annual, regimes = grouped_metrics(oos_trades, "year"), grouped_metrics(oos_trades, "regime")
    health = strategy_health(oos_trades)
    gate = trading_gate(oos_tm, annual, health)

    # Purge the last development labels because their forward return touches final-test prices.
    final_train = dev.iloc[:-args.horizon] if args.horizon else dev
    final_prob_values, base, calibrator = calibrated_fit_predict(
        final_train, final, features, args.horizon, args.model)
    final = final.copy()
    final["probability"] = final_prob_values
    (final["predicted_return"], final["predicted_return_low"],
     final["predicted_return_high"]) = return_fit_predict(final_train, final, features)
    (final["research_predicted_return"], final["research_predicted_return_low"],
     final["research_predicted_return_high"], final_research_meta) = (
        research_return_fit_predict(final_train, final, features))
    final_trades = simulate(final, primary, args.threshold, args.horizon, args.stop_atr,
                            args.reward_risk, args.commission_bps, args.tax_bps,
                            args.slippage_bps, args.entry_gap_low_atr,
                            args.entry_gap_high_atr, args.minimum_predicted_return,
                            args.risk_policy)
    final_prob, final_tm = probability_metrics(final, args.threshold), trade_metrics(final_trades)
    trade_final_return_forecast = return_forecast_metrics(final)
    final_return_forecast = return_forecast_metrics(final, "research_predicted_return")
    required_validation = formal_validation(oos, primary, oos_tm, final_tm, args)

    # A quoted entry range must survive several alternative chronological cuts.
    # This deliberately costs more runtime than a single attractive backtest.
    strict_entry = strict_entry_validation(usable, features, primary, args)

    # Refit/calibrate on all labelled history only after final evaluation, solely for today's probability.
    latest = data.iloc[[-1]].copy()
    latest_probability = float(calibrated_fit_predict(
        usable, latest, features, args.horizon, args.model)[0][0])
    latest_predicted_return, latest_return_low, latest_return_high = return_fit_predict(
        usable, latest, features)
    latest_predicted_return = float(latest_predicted_return[0])
    latest_return_low = float(latest_return_low[0])
    latest_return_high = float(latest_return_high[0])
    (latest_research_return, latest_research_low, latest_research_high,
     latest_research_meta) = research_return_fit_predict(usable, latest, features)
    latest_research_return = float(latest_research_return[0])
    latest_research_low = float(latest_research_low[0])
    latest_research_high = float(latest_research_high[0])
    latest_atr = float(latest.ATR.iloc[0])
    latest_close = float(latest.Close.iloc[0])
    if not all(np.isfinite(value) for value in (latest_probability, latest_atr, latest_close)):
        raise RuntimeError("latest market data or model output is not finite; no prediction was recorded")
    if latest_close <= 0 or latest_atr <= 0:
        raise RuntimeError("latest Close and ATR must be positive; no prediction was recorded")
    latest_volume_mean20 = float(data.Volume.rolling(20).mean().iloc[-1])
    latest_volume_ratio20 = (float(latest.Volume.iloc[0]) / latest_volume_mean20
                             if latest_volume_mean20 > 0 else None)
    latest_row = latest.iloc[0]
    recent_year = max((trade.entry_date[:4] for trade in oos_trades), default=None)
    recent_year_trades = ([trade for trade in oos_trades if trade.entry_date[:4] == recent_year]
                          if recent_year else [])
    recent_year_metrics = trade_metrics(recent_year_trades)
    holding = None
    if args.shares is not None:
        holding = holding_analysis(
            args.shares, args.average_cost, latest_close, latest_atr,
            latest_probability >= args.threshold,
            bool(gate["passed"] and strict_entry["passed"]),
            args.stop_atr, args.reward_risk, args.commission_bps,
            args.tax_bps, args.slippage_bps, args.add_shares,
            latest_volume_ratio20, args.horizon)
        add_ratio = holding["add_shares"] / args.shares
        avg_oos_trades, avg_oos_adds = simulate_averaging(
            oos, primary, args.threshold, args.horizon, args.stop_atr,
            args.reward_risk, args.commission_bps, args.tax_bps,
            args.slippage_bps, add_ratio,
            entry_gap_low_atr=args.entry_gap_low_atr,
            entry_gap_high_atr=args.entry_gap_high_atr)
        avg_final_trades, avg_final_adds = simulate_averaging(
            final, primary, args.threshold, args.horizon, args.stop_atr,
            args.reward_risk, args.commission_bps, args.tax_bps,
            args.slippage_bps, add_ratio,
            entry_gap_low_atr=args.entry_gap_low_atr,
            entry_gap_high_atr=args.entry_gap_high_atr)
        holding["averaging_backtest"] = {
            "add_ratio": add_ratio,
            "trigger_atr": 0.75,
            "oos": {**trade_metrics(avg_oos_trades), "adds": avg_oos_adds},
            "final_test": {**trade_metrics(avg_final_trades), "adds": avg_final_adds},
        }
    formal_passed = bool(required_validation["passed"])
    strict_passed = bool(strict_entry["passed"])
    latest_signal = bool(latest_probability >= args.threshold)
    latest_position_fraction = strategy_position_fraction(
        str(latest_row.get("regime", "unknown")), args.risk_policy)
    forecast_validation_passed = bool(
        oos_return_forecast["passed"] and final_return_forecast["passed"])
    trade_forecast_validation_passed = bool(
        trade_oos_return_forecast["passed"] and trade_final_return_forecast["passed"])
    return_signal = latest_predicted_return >= args.minimum_predicted_return
    executable = bool(latest_signal and latest_position_fraction > 0 and return_signal
                      and formal_passed and gate["passed"]
                      and strict_passed and trade_forecast_validation_passed)
    candidate_low = latest_close + args.entry_gap_low_atr * latest_atr
    candidate_high = latest_close + args.entry_gap_high_atr * latest_atr
    candidate_entry = (candidate_low + candidate_high) / 2
    candidate_risk = args.stop_atr * latest_atr
    valid_until = str((pd.Timestamp(latest.index[0]) + pd.offsets.BDay(args.horizon)).date())
    execution_plan = {
        "available": executable,
        "suggested_entry": candidate_entry if executable else None,
        "entry_low": candidate_low if executable else None,
        "entry_high": candidate_high if executable else None,
        "tranches": ([{"price": candidate_high,
                        "capital_ratio": latest_position_fraction / 2},
                       {"price": candidate_low,
                        "capital_ratio": latest_position_fraction / 2}]
                      if executable else []),
        "position_sizing_denominator": "配置給 00631L 的資金",
        "maximum_capital_ratio": latest_position_fraction,
        "stop": candidate_entry - candidate_risk if executable else None,
        "take_profit_1": candidate_entry + candidate_risk if executable else None,
        "take_profit_2": candidate_entry + args.reward_risk * candidate_risk if executable else None,
        "reward_risk_1": 1.0, "reward_risk_2": args.reward_risk,
        "condition": "正式驗證、機率、交易閘門、多重驗證及下一交易日開盤區間全部通過",
        "invalidation": "任一驗證失敗、開盤超出區間、大盤轉空或跌破停損",
        "valid_until": valid_until,
    }
    price_forecast = research_price_forecast(
        latest_close, latest_research_return, latest_research_low, latest_research_high,
        args.horizon, valid_until, oos_return_forecast, final_return_forecast)
    result = clean_json({
        "ticker": args.ticker, "model": args.model, "feature_set": args.feature_set,
        "model_version": (MODEL_VERSION if args.model == "extra-trees" else
                          "20260818.1"),
        "strategy_version": STRATEGY_VERSION,
        "latest_date": str(latest.index[0].date()),
        "latest_price": latest_close, "latest_probability": latest_probability,
        "threshold": args.threshold,
        "threshold_margin": latest_probability - args.threshold,
        "raw_signal": latest_signal,
        "signal": executable,
        "action": "買進" if executable else "不交易",
        "valid_until": valid_until,
        "generated_at": dt.datetime.now(TAIPEI).isoformat(),
        "risk_levels": {"stop_1_5_atr": execution_plan["stop"],
                        "target_2r": execution_plan["take_profit_2"]},
        "entry_plan": {"latest_close_reference": latest_close,
                       "acceptable_low": candidate_low if executable else None,
                       "acceptable_high": candidate_high if executable else None,
                       "gap_low_atr": args.entry_gap_low_atr,
                       "gap_high_atr": args.entry_gap_high_atr,
                       "selection_rule": "OOS至少30筆、EV_R>0、PF>=1.2後，以勝率及保留期穩健性選擇",
                       "oos_backtest": oos_tm,
                       "final_test_confirmation": final_tm,
                       "next_open_known": False},
        "execution_plan": execution_plan,
        "price_forecast": price_forecast,
        "candidate_strategy": {"version": STRATEGY_VERSION,
                               "parameters": CANDIDATE_20260819_3,
                               "risk_policy": RISK_POLICY_20260828_2,
                               "return_signal": return_signal,
                               "trading_return_forecast_validation_passed":
                                   trade_forecast_validation_passed,
                               "activated": executable},
        "research_price_model": {
            "candidates": list(RESEARCH_RETURN_CANDIDATES),
            "latest_selection": latest_research_meta,
            "final_selection": final_research_meta,
            "affects_trading_strategy": False,
        },
        "context_used": list(contexts), "context_skipped": skipped,
        "context_alignment": {
            symbol: {"overlap_rows": int(ctx.index.intersection(primary.index).size),
                     "overlap_ratio": float(ctx.index.intersection(primary.index).size / len(primary))}
            for symbol, ctx in contexts.items()
        },
        "development_window": [str(dev.index.min().date()), str(dev.index.max().date())],
        "final_test_window": [str(final.index.min().date()), str(final.index.max().date())],
        "oos_probability": oos_prob, "oos_trading": oos_tm,
        "recent_year": recent_year, "recent_year_trading": recent_year_metrics,
        "annual_stability": annual, "regime_stability": regimes,
        "strategy_health": health, "trading_gate": gate,
        "validation_snapshot": required_validation,
        "strict_entry_validation": strict_entry,
        "return_forecast_validation": {"development_oos": oos_return_forecast,
                                       "independent_oos": final_return_forecast,
                                       "passed": forecast_validation_passed},
        "trading_return_forecast_validation": {
            "development_oos": trade_oos_return_forecast,
            "independent_oos": trade_final_return_forecast,
            "passed": trade_forecast_validation_passed,
        },
        "validated_score": validated_score(oos_prob, oos_tm, annual, regimes),
        "final_test_probability": final_prob, "final_test_trading": final_tm,
        "costs_bps": {"commission_each_side": args.commission_bps,
                      "tax_sell_side": args.tax_bps, "slippage_each_side": args.slippage_bps},
        "holding": holding,
        "confidence": ("高" if executable else "低"),
        "data_source_snapshot": {
            "source": ("Yahoo Finance via yfinance；缺值列由臺灣證券交易所 STOCK_DAY 補齊"
                       if primary.attrs.get("twse_fallback_dates") else
                       "Yahoo Finance via yfinance"),
            "price_type": "Yahoo auto-adjusted daily OHLCV；TWSE 備援列為官方日成交資料",
            "twse_fallback_dates": primary.attrs.get("twse_fallback_dates", []),
            "market_date": str(latest.index[0].date()),
            "timezone": "Asia/Taipei",
            "downloaded_at": dt.datetime.now(TAIPEI).isoformat(),
            "period": args.period,
            "missing_value_handling": "技術指標暖機列排除；參考市場最多向前填補 3 日",
            "corporate_actions": "yfinance auto_adjust=True",
        },
        "main_risks": ["槓桿ETF波動與複利耗損", "模型跨期不穩定",
                       "跳空造成停損滑價", "歷史勝率不保證未來結果"],
        "oos_trades": [asdict(t) for t in oos_trades],
        "final_test_trades": [asdict(t) for t in final_trades],
    })

    health_zh = {"healthy": "健康", "degraded": "退化", "insufficient": "樣本不足"}
    print("=" * 62)
    print("本分析僅供研究與參考，不構成投資建議；00631L 為槓桿型 ETF，"
          "使用者應自行承擔交易風險與損益。")
    print(f"標的：{args.ticker}　資料日期：{result['latest_date']}　收盤價：{latest_close:.2f}")
    print(f"目前動作：{'等待下一交易日開盤確認' if result['signal'] else '觀望（不買進）'}")
    print(f"預測成功機率：{latest_probability * 100:.1f}%　買進門檻：{args.threshold * 100:.1f}%")
    print(f"買進判定規則：機率至少 {args.threshold * 100:.1f}%、交易驗證閘門通過，"
          "且多重買進價驗證通過，才可考慮建立新部位")
    margin = (latest_probability - args.threshold) * 100
    if margin >= 0:
        print(f"目前機率已高於門檻 {margin:.1f} 個百分點；"
              f"閘門{'通過' if gate['passed'] else '未通過'}")
    else:
        print(f"目前機率仍低於門檻 {abs(margin):.1f} 個百分點；維持觀望")
    if holding is not None:
        print("持股提醒：買進條件是新部位訊號，不代表既有持股必須加碼")
    model_zh = "ExtraTrees 完整技術指標版" if args.model == "extra-trees" else "Logistic 基準版"
    features_zh = "全技術指標" if args.feature_set == "all" else "基礎特徵"
    print(f"模型版本：{model_zh}；{features_zh}")
    print(f"版本代碼：模型 {result['model_version']}；策略 {result['strategy_version']}")
    print(f"交易驗證閘門：{'通過' if gate['passed'] else '未通過'}")
    validation_zh = {
        "effective_trades_at_least_30": "有效交易樣本數 ≥ 30",
        "walk_forward_winrate_std_at_most_15pp": "Walk-Forward fold 勝率標準差 ≤ 15 個百分點",
        "profit_factor_at_least_1_2": "Profit Factor ≥ 1.2",
        "max_drawdown_at_most_25pct": "最大回撤 ≤ 24.9%",
        "oos_vs_development_winrate_gap_at_most_10pp": "OOS 與開發期勝率差異 ≤ 10 個百分點",
        "walk_forward_and_independent_oos_completed": "Walk-Forward 與獨立 OOS 已完成",
    }
    print(f"正式模型驗證：{'通過' if required_validation['passed'] else '未通過'}")
    for name, passed in required_validation["checks"].items():
        print(f"  {validation_zh.get(name, name)}：{'通過' if passed else '未通過／資料不足'}")
    print(f"  有效交易 {required_validation['effective_trades']} 筆；PF "
          f"{fmt(required_validation['profit_factor'])}；最大回撤 "
          f"{pct(required_validation['max_drawdown_pct'])}；勝率差 "
          f"{fmt(required_validation['oos_vs_development_winrate_gap_pp'], 1)} 個百分點")
    strict_summary = strict_entry["summary"]
    print(f"多重買進價驗證：{'通過' if strict_entry['passed'] else '未通過'}（{strict_summary['runs']} 組）")
    print(f"  中位交易 {strict_summary['median_trades']:.0f} 筆；"
          f"中位勝率 {pct(strict_summary['median_win_rate'])}；"
          f"中位平均 {fmt(strict_summary['median_ev_r'])}R；"
          f"中位 PF {fmt(strict_summary['median_profit_factor'])}")
    print(f"  正報酬測試 {strict_summary['positive_run_ratio'] * strict_summary['runs']:.0f}/"
          f"{strict_summary['runs']}；雙倍成本平均 {fmt(strict_summary['double_cost_median_ev_r'])}R；"
          f"PF {fmt(strict_summary['double_cost_median_profit_factor'])}")
    entry_plan = result["entry_plan"]
    print(f"最新收盤參考價：{latest_close:.2f}（不是預測價）")
    forecast = result["price_forecast"]
    dev_forecast = forecast["development_oos"]
    final_forecast = forecast["independent_oos"]
    if forecast["available"]:
        print(f"5 日研究中央預測價格：{forecast['predicted_price']:.2f}；"
              f"80% 區間：{forecast['predicted_price_low']:.2f}～"
              f"{forecast['predicted_price_high']:.2f}；期限：{forecast['valid_until']}")
        print("研究價格預測不是建議買進價，不得單獨作為交易依據")
    else:
        print(f"5 日研究價格預測：未產生；原因：{forecast['unavailable_reason']}")
    print(f"價格模型開發期：MAE {pct(dev_forecast['mae'])}；方向準確率 "
          f"{pct(dev_forecast['direction_accuracy'])}；80% 區間覆蓋率 "
          f"{pct(dev_forecast['interval_80_coverage'])}")
    print(f"價格模型保留期：MAE {pct(final_forecast['mae'])}；方向準確率 "
          f"{pct(final_forecast['direction_accuracy'])}；80% 區間覆蓋率 "
          f"{pct(final_forecast['interval_80_coverage'])}")
    if result["signal"]:
        print(f"歷史回測較穩健的下一交易日買進區間：{entry_plan['acceptable_low']:.2f}～"
              f"{entry_plan['acceptable_high']:.2f}")
        print(f"此成交區間回測：OOS {oos_tm['trades']} 筆、勝率 {pct(oos_tm['win_rate'])}、"
              f"平均 {fmt(oos_tm['ev_r'])}R、PF {fmt(oos_tm['profit_factor'])}")
        print(f"保留期確認：{final_tm['trades']} 筆、勝率 {pct(final_tm['win_rate'])}、"
              f"平均 {fmt(final_tm['ev_r'])}R、PF {fmt(final_tm['profit_factor'])}")
        print(f"開盤高於 {entry_plan['acceptable_high']:.2f}：跳空過大，不追價；"
              f"低於 {entry_plan['acceptable_low']:.2f}：動能確認不足，不進場")
        print("開盤位於區間內：機率與交易閘門已通過，才可考慮建立新部位")
        ep = result["execution_plan"]
        print(f"建議買進價：{ep['suggested_entry']:.2f}；分兩批各使用 50% 資金")
        print(f"建議停損賣出價：{ep['stop']:.2f}（買進價下方 {args.stop_atr:g} ATR）")
        print(f"第一停利價：{ep['take_profit_1']:.2f}（1R）；"
              f"第二停利價：{ep['take_profit_2']:.2f}（{args.reward_risk:g}R）")
        print(f"時間賣出：買進後第 {args.horizon} 個交易日收盤；先碰停損或停利則提前賣出")
        print("注意：若實際買進價不同，停損與停利價必須依實際成交價重新計算")
    else:
        if not result["raw_signal"]:
            reason = "機率未達門檻"
        elif not gate["passed"]:
            reason = "歷史交易驗證閘門未通過"
        else:
            reason = "建議價未通過多重時間切割與成本壓力驗證"
        print(f"買賣點：目前沒有進場點；原因：{reason}")
        print("建議買進價／區間／分批比例／停損停利：不提供（驗證未通過）")
    print(f"預測信心：{result['confidence']}；主要風險：" + "、".join(result["main_risks"]))
    print("=" * 62)
    if holding is not None:
        print("持股分析：")
        print(f"  持股：{holding['shares']:,} 股　平均成本：{holding['average_cost']:.4f}　"
              f"總成本：{holding['cost_basis']:,.0f}")
        print(f"  預估目前損益：{holding['estimated_pnl']:+,.0f}　"
              f"報酬率：{holding['estimated_return'] * 100:+.2f}%（已估賣出成本）")
        print(f"  含賣出成本回本價：{holding['break_even_price_after_sell_costs']:.2f}")
        print(f"  持股建議：{holding['action']}；原因：{holding['reason']}")
        print(f"  持股停損賣出價：{holding['holding_stop_price']:.2f}　"
              f"預估總損益：{holding['estimated_pnl_at_stop']:+,.0f}")
        print(f"  持股停利賣出價：{holding['holding_target_price']:.2f}　"
              f"預估總損益：{holding['estimated_pnl_at_target']:+,.0f}")
        print("  加碼／攤平試算：")
        print(f"    歷史優化規則：{holding['averaging_rule']}")
        print(f"    時間點：{holding['averaging_timing']}")
        print(f"    取消條件：{holding['averaging_cancel']}")
        print(f"    判斷：{holding['averaging_action']}；原因：{holding['averaging_reason']}")
        print(f"    建議加碼價：{holding['add_trigger_price']:.2f}（限價上限；"
              f"不得低於第一防守價 {holding['holding_stop_price']:.2f}）")
        print(f"    建議加碼股數：{holding['add_shares']:,} 股（使用者上限 {holding['add_shares_cap']:,} 股）　"
              f"預估投入：{holding['add_estimated_cost']:,.0f}")
        print(f"    加碼後持股：{holding['after_add_shares']:,} 股　"
              f"平均成本：{holding['after_add_average_cost']:.4f}　"
              f"含賣出成本回本價：{holding['after_add_break_even_price']:.2f}")
        print(f"    加碼後停損價：{holding['after_add_stop_price']:.2f}　"
              f"預估總損益：{holding['after_add_pnl_at_stop']:+,.0f}")
        print(f"    加碼後停利價：{holding['after_add_target_price']:.2f}　"
              f"預估總損益：{holding['after_add_pnl_at_target']:+,.0f}")
        avg_bt = holding["averaging_backtest"]
        print(f"    加碼策略歷史回測（加碼比例 {avg_bt['add_ratio'] * 100:.1f}%）：")
        print(f"      OOS：交易 {avg_bt['oos']['trades']} 筆　實際加碼 {avg_bt['oos']['adds']} 次　"
              f"勝率 {pct(avg_bt['oos']['win_rate'])}　平均 {fmt(avg_bt['oos']['ev_r'])}R　"
              f"PF {fmt(avg_bt['oos']['profit_factor'])}　最大回撤 {fmt(avg_bt['oos']['max_drawdown_r'])}R")
        print(f"      保留期：交易 {avg_bt['final_test']['trades']} 筆　"
              f"實際加碼 {avg_bt['final_test']['adds']} 次　"
              f"勝率 {pct(avg_bt['final_test']['win_rate'])}　"
              f"平均 {fmt(avg_bt['final_test']['ev_r'])}R　"
              f"PF {fmt(avg_bt['final_test']['profit_factor'])}　"
              f"最大回撤 {fmt(avg_bt['final_test']['max_drawdown_r'])}R")
        plan = holding["operation_plan"]
        print("  分批操作方法建議（動態參考值）：")
        print(f"    第一防守：{plan['first_defense']['low']:.2f}　{plan['first_defense']['advice']}")
        print(f"    最終停損：{plan['final_stop']['low']:.2f}～{plan['final_stop']['high']:.2f}　"
              f"{plan['final_stop']['advice']}")
        print(f"    反彈目標一：{plan['rebound_target_one']['low']:.2f}～"
              f"{plan['rebound_target_one']['high']:.2f}　{plan['rebound_target_one']['advice']}")
        print(f"    反彈目標二：{plan['rebound_target_two']['low']:.2f}～"
              f"{plan['rebound_target_two']['high']:.2f}　{plan['rebound_target_two']['advice']}")
        print(f"    強勢停利：{plan['strong_take_profit']['low']:.2f}～"
              f"{plan['strong_take_profit']['high']:.2f}　{plan['strong_take_profit']['advice']}")
        print(f"    目前量能：20日均量的 {fmt(plan['volume_ratio20'], 2)} 倍；"
              f"放量條件目前{'成立' if plan['strong_take_profit']['volume_confirmed_now'] else '未成立'}")
        print("    注意：這套分批操作表尚未納入上方單一出場規則的歷史勝率回測")
        print("=" * 62)
    print(f"走勢外推驗證期間：{result['development_window'][0]} 至 {result['development_window'][1]}；共 {args.folds} 折")
    print(f"樣本外預測：{oos_prob['predictions']} 筆；達門檻訊號：{oos_prob['signals']} 筆；"
          f"訊號勝率（符合預測目標）：{pct(oos_prob['buy_precision'])}；Brier 誤差：{fmt(oos_prob['brier'])}")
    print(f"單一設定樣本外交易：{oos_tm['trades']} 筆（獲利 {oos_tm['wins']}／"
          f"虧損 {oos_tm['losses']}）；交易勝率：{pct(oos_tm['win_rate'])}；"
          f"平均每筆：{fmt(oos_tm['ev_r'])}R；"
          f"獲利因子：{fmt(oos_tm['profit_factor'])}；最大回撤：{fmt(oos_tm['max_drawdown_r'])}R")
    print(f"平均獲利：{fmt(oos_tm['average_win_r'])}R；平均虧損：{fmt(oos_tm['average_loss_r'])}R；"
          f"平均盈虧比：{fmt(oos_tm['payoff_ratio'])}")
    print(f"最近回測年度 {recent_year or '無資料'}：交易 {recent_year_metrics['trades']} 筆；"
          f"勝率 {pct(recent_year_metrics['win_rate'])}")
    print(f"綜合驗證分數：{result['validated_score']}/100；策略健康度："
          f"{health_zh.get(result['strategy_health']['status'], result['strategy_health']['status'])}")
    print(f"最終保留測試：{result['final_test_window'][0]} 至 {result['final_test_window'][1]}；"
          f"交易 {final_tm['trades']} 筆；平均每筆 "
          f"{(fmt(final_tm['ev_r']) + 'R') if final_tm['ev_r'] is not None else '無資料'}；"
          f"獲利因子 {fmt(final_tm['profit_factor'])}")
    print("年度穩定性：")
    for name, metrics in annual.items():
        print(f"  {name}：交易={metrics['trades']}　勝率={pct(metrics['win_rate'])}　"
              f"平均={fmt(metrics['ev_r'])}R　獲利因子={fmt(metrics['profit_factor'])}")
    regime_zh = {"bull": "多頭", "bear": "空頭", "bull_high_vol": "多頭／高波動",
                 "bear_high_vol": "空頭／高波動", "unknown": "未分類"}
    print("市場環境穩定性：")
    for name, metrics in regimes.items():
        print(f"  {regime_zh.get(name, name)}：交易={metrics['trades']}　勝率={pct(metrics['win_rate'])}　"
              f"平均={fmt(metrics['ev_r'])}R　獲利因子={fmt(metrics['profit_factor'])}")
    print(f"成本假設（基點）：手續費每邊 {args.commission_bps}；賣出稅 {args.tax_bps}；"
          f"滑價每邊 {args.slippage_bps}")
    if skipped:
        print("略過的參考市場：" + "; ".join(skipped), file=sys.stderr)
    if not args.no_record:
        prediction_id = record_prediction(args.database, result, latest_row, primary, args)
        result["sqlite"] = {"database": args.database, "prediction_id": prediction_id}
        print(f"SQLite 預測紀錄：{args.database}（ID {prediction_id}，新增且不覆蓋舊預測）")
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
        print(f"完整 JSON 已寫入：{args.output_json}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
