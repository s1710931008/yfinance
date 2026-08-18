import importlib.util
from pathlib import Path
import sys

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
