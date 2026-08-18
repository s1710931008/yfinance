import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd


SPEC = importlib.util.spec_from_file_location(
    "predict", Path(__file__).parents[1] / "scripts" / "predict.py")
predict = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = predict
SPEC.loader.exec_module(predict)


def market_frame():
    idx = pd.date_range("2024-01-01", periods=8, freq="B")
    return pd.DataFrame({
        "Open": [100] * 8, "High": [101, 101, 104, 101, 101, 101, 101, 101],
        "Low": [99, 99, 99, 97, 99, 99, 99, 99], "Close": [100] * 8,
        "Volume": [1000] * 8,
    }, index=idx)


def test_clean_ohlcv_removes_incomplete_trailing_row():
    frame = market_frame()
    frame.loc[frame.index[-1], "Close"] = np.nan
    cleaned = predict._clean_ohlcv_frame(frame, "00631L.TW")
    assert len(cleaned) == len(frame) - 1
    assert cleaned.index[-1] == frame.index[-2]
    assert np.isfinite(cleaned[["Open", "High", "Low", "Close", "Volume"]]).all().all()


def test_clean_ohlcv_rejects_no_complete_rows():
    frame = market_frame()
    frame["Close"] = np.nan
    with np.testing.assert_raises_regex(RuntimeError, "no complete finite OHLCV rows"):
        predict._clean_ohlcv_frame(frame, "00631L.TW")


def test_twse_fallback_fills_only_matching_incomplete_date(monkeypatch):
    frame = market_frame()
    gap_date = frame.index[-1]
    frame.loc[gap_date, ["Open", "High", "Low", "Close"]] = np.nan
    official = {gap_date: {
        "Open": 35.88, "High": 35.90, "Low": 34.81,
        "Close": 34.81, "Volume": 167876262.0,
    }}
    monkeypatch.setattr(predict, "_twse_month_rows", lambda *_args: official)
    filled, dates = predict._fill_twse_gaps(frame, "00631L.TW")
    assert dates == [str(gap_date.date())]
    assert filled.loc[gap_date, "Close"] == 34.81
    assert filled.loc[gap_date, "Volume"] == 167876262.0
    assert filled.loc[frame.index[-2], "Close"] == frame.loc[frame.index[-2], "Close"]


def test_twse_fallback_does_not_run_for_complete_yahoo_data(monkeypatch):
    monkeypatch.setattr(
        predict, "_twse_month_rows",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected TWSE request")),
    )
    filled, dates = predict._fill_twse_gaps(market_frame(), "00631L.TW")
    assert dates == []
    pd.testing.assert_frame_equal(filled, market_frame())


def test_simulation_applies_costs_and_target():
    market = market_frame()
    rows = market.iloc[[0]].copy()
    rows["probability"] = 0.9
    rows["ATR"] = 1.0
    rows["regime"] = "bull"
    trades = predict.simulate(rows, market, 0.7, 5, 1.5, 2.0, 10, 10, 0)
    assert len(trades) == 1
    assert trades[0].reason == "target"
    assert trades[0].gross_r == 2.0
    assert trades[0].net_r < trades[0].gross_r


def test_entry_range_rejects_gap_up():
    market = market_frame()
    market.loc[market.index[1], "Open"] = 102
    rows = market.iloc[[0]].copy()
    rows["probability"], rows["ATR"], rows["regime"] = .9, 1., "bull"
    assert not predict.simulate(
        rows, market, .7, 5, 1.5, 2, 0, 0, 0,
        entry_gap_low_atr=0,
        entry_gap_high_atr=.25,
    )


def test_same_bar_stop_and_target_is_conservative():
    market = market_frame()
    market.loc[market.index[1], ["High", "Low"]] = [104, 97]
    rows = market.iloc[[0]].copy()
    rows["probability"], rows["ATR"], rows["regime"] = 0.9, 1.0, "bull"
    trade = predict.simulate(rows, market, 0.7, 5, 1.5, 2.0, 0, 0, 0)[0]
    assert trade.reason == "stop"
    assert trade.net_r == -1.0


def test_averaging_backtest_records_add_and_keeps_stop_first():
    market = market_frame()
    rows = market.iloc[[0]].copy()
    rows["probability"], rows["ATR"], rows["regime"] = 0.9, 1.0, "bull"
    market.loc[market.index[1], ["High", "Low"]] = [104, 97]
    trades, adds = predict.simulate_averaging(
        rows, market, .7, 5, 1.5, 2, 0, 0, 0, .25, .75)
    assert adds == 1
    assert trades[0].reason == "add+stop"
    assert trades[0].net_r == -1.0


def test_metrics_profit_factor_and_drawdown():
    trades = [predict.Trade("", "", "", 1, 1, .8, r, r, "time", "bull")
              for r in [1.0, -0.5, -0.5, 2.0]]
    metrics = predict.trade_metrics(trades)
    assert metrics["ev_r"] == 0.5
    assert metrics["profit_factor"] == 3.0
    assert metrics["max_drawdown_r"] == 1.0


def test_metrics_compounded_drawdown_percentage():
    trades = [predict.Trade("", "", "", 1, 1, .8, r, r, "time", "bull", ret)
              for r, ret in [(1.0, .10), (-1.0, -.20)]]
    metrics = predict.trade_metrics(trades)
    assert round(metrics["total_return_pct"], 3) == -.12
    assert round(metrics["max_drawdown_pct"], 3) == .20


def test_formal_validation_requires_all_mandatory_gates(monkeypatch):
    def fake_simulate(rows, *_args, **_kwargs):
        return [predict.Trade("", "", "", 1, 1, .8, r, r, "time", "bull", r / 100)
                for r in ([1.0] * 6 + [-.5] * 4)]

    monkeypatch.setattr(predict, "simulate", fake_simulate)
    rows = pd.DataFrame({"fold": [1, 2, 3]})
    args = SimpleNamespace(threshold=.2, horizon=5, stop_atr=1.5, reward_risk=2,
                           commission_bps=14.25, tax_bps=10, slippage_bps=5,
                           entry_gap_low_atr=.15, entry_gap_high_atr=.55)
    oos = {"trades": 30, "win_rate": .60, "profit_factor": 1.5,
           "max_drawdown_pct": .10}
    final = {"trades": 10, "win_rate": .55}
    validation = predict.formal_validation(rows, pd.DataFrame(), oos, final, args)
    assert validation["passed"]
    assert validation["fold_winrate_std_pp"] == 0


def test_formal_validation_rejects_empty_walk_forward_fold(monkeypatch):
    def fake_simulate(rows, *_args, **_kwargs):
        if int(rows.fold.iloc[0]) == 1:
            return []
        return [predict.Trade("", "", "", 1, 1, .8, 1, 1, "time", "bull", .01)]

    monkeypatch.setattr(predict, "simulate", fake_simulate)
    rows = pd.DataFrame({"fold": [1, 2]})
    args = SimpleNamespace(threshold=.2, horizon=5, stop_atr=1.5, reward_risk=2,
                           commission_bps=14.25, tax_bps=10, slippage_bps=5,
                           entry_gap_low_atr=.15, entry_gap_high_atr=.55)
    oos = {"trades": 30, "win_rate": .60, "profit_factor": 1.5,
           "max_drawdown_pct": .10}
    final = {"trades": 10, "win_rate": .55}
    validation = predict.formal_validation(rows, pd.DataFrame(), oos, final, args)
    assert not validation["passed"]
    assert validation["fold_winrate_std_pp"] is None
    assert not validation["all_folds_have_trades"]


def test_trading_gate_requires_all_conditions():
    metrics = {"trades": 40, "ev_r": .12, "profit_factor": 1.3}
    annual = {"2022": {"trades": 10, "ev_r": .1},
              "2023": {"trades": 10, "ev_r": .2}}
    health = {"status": "healthy"}
    assert predict.trading_gate(metrics, annual, health)["passed"]
    metrics["profit_factor"] = 1.0
    assert not predict.trading_gate(metrics, annual, health)["passed"]


def test_holding_analysis_includes_sell_costs_and_action():
    h = predict.holding_analysis(2000, 38.0745, 35.95, 1.7, True, True,
                                 1.5, 2, 14.25, 10, 5)
    assert h["estimated_pnl"] < 0
    assert h["break_even_price_after_sell_costs"] > 38.0745
    assert h["action"] == "續抱"
    assert h["estimated_pnl_at_stop"] < 0
    assert h["estimated_pnl_at_target"] > 0
    assert h["after_add_shares"] == 2100
    assert h["after_add_average_cost"] < h["average_cost"]
    assert h["after_add_stop_price"] == h["holding_stop_price"]
    assert h["after_add_target_price"] == h["holding_target_price"]
    assert h["add_trigger_price"] < h["average_cost"]
    assert h["averaging_valid_days"] == 5
    assert not h["operation_plan"]["backtested"]
    assert h["operation_plan"]["final_stop"]["low"] < h["operation_plan"]["first_defense"]["low"]


def test_build_dataset_has_context_and_no_tail_labels():
    idx = pd.date_range("2023-01-01", periods=100, freq="B")
    close = pd.Series(np.linspace(100, 130, 100), index=idx)
    frame = pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1,
                          "Close": close, "Volume": 1000}, index=idx)
    data, features = predict.build_dataset(frame, {"^TWII": frame}, 5, .04, -.025, "all")
    assert any(name.startswith("ctx_TWII") for name in features)
    assert {"rsi14", "kd_k", "kd_d", "macd_hist", "cmf20", "obv_momentum20"} <= set(features)
    assert data.label.tail(5).isna().all()


def test_record_prediction_rejects_nan_market_price_before_sqlite_write(tmp_path):
    result = {"latest_price": float("nan")}
    with np.testing.assert_raises_regex(ValueError, "market_price must be"):
        predict.record_prediction(
            str(tmp_path / "predictions.sqlite3"), result, pd.Series(),
            pd.DataFrame(), SimpleNamespace())
    assert not (tmp_path / "predictions.sqlite3").exists()
