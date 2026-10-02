import importlib.util
import json
from pathlib import Path
import sqlite3
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


def test_entry_gap_validation_allows_gap_down_interval():
    predict.validate_entry_gap_atr(-0.25, 0.25)


def test_entry_gap_validation_rejects_unordered_or_nonfinite_interval():
    for low, high in ((0.25, 0.25), (0.5, 0.25), (np.nan, 0.25), (-0.25, np.inf)):
        with np.testing.assert_raises_regex(ValueError, "finite and satisfy low < high"):
            predict.validate_entry_gap_atr(low, high)


def test_00631l_risk_policy_scales_or_blocks_regimes():
    assert predict.strategy_position_fraction("bull", True) == .85
    assert predict.strategy_position_fraction("bull_high_vol", True) == .50
    assert predict.strategy_position_fraction("bear_high_vol", True) == 0
    assert predict.strategy_position_fraction("bear_high_vol", False) == 1


def test_risk_policy_blocks_bear_high_vol_trade():
    market = market_frame()
    rows = market.iloc[[0]].copy()
    rows["probability"], rows["ATR"], rows["regime"] = .9, 1., "bear_high_vol"
    assert not predict.simulate(rows, market, .7, 5, 1.5, 2, 0, 0, 0,
                                risk_policy=True)


def test_research_price_forecast_is_independent_of_trading_gate():
    passed = {"passed": True, "mae": .03, "direction_accuracy": .55,
              "interval_80_coverage": .80}
    forecast = predict.research_price_forecast(
        100, .05, -.02, .10, 5, "2026-09-04", passed, passed)
    assert forecast["available"]
    assert forecast["predicted_price"] == 105
    assert forecast["predicted_price_low"] == 98
    assert np.isclose(forecast["predicted_price_high"], 110)


def test_research_price_forecast_keeps_prices_and_marks_failed_validation():
    passed = {"passed": True}
    failed = {"passed": False}
    forecast = predict.research_price_forecast(
        100, .05, -.02, .10, 5, "2026-09-04", passed, failed)
    assert forecast["available"]
    assert forecast["predicted_price"] == 105
    assert forecast["predicted_price_low"] == 98
    assert np.isclose(forecast["predicted_price_high"], 110)
    assert not forecast["validation_passed"]
    assert "研究參考" in forecast["validation_status"]


def test_unvalidated_research_scenario_is_numeric_but_not_actionable():
    scenario = predict.build_research_scenario(
        100, .05, -.10, .15, 2, -.25, .25, 1.5, 2.5,
        "2026-09-08", False)
    assert scenario["available"]
    assert not scenario["validated"]
    assert scenario["not_actionable"]
    assert scenario["raw_estimated_price"] == 105
    assert scenario["scenario_entry"] == 100
    assert scenario["scenario_stop"] == 97
    assert scenario["scenario_take_profit_1"] == 103
    assert scenario["scenario_take_profit_2"] == 107.5
    assert "不可交易" in scenario["warning"]


def test_research_scenario_hides_values_when_inputs_are_invalid():
    scenario = predict.build_research_scenario(
        100, np.nan, -.10, .15, 2, -.25, .25, 1.5, 2.5,
        "2026-09-08", False)
    assert not scenario["available"]
    assert "raw_estimated_price" not in scenario


def test_research_return_model_does_not_read_test_outcomes():
    rng = np.random.default_rng(42)
    train = pd.DataFrame({
        "x": rng.normal(size=220),
        "future_return": rng.normal(scale=.03, size=220),
    })
    test = pd.DataFrame({"x": [-1., 0., 1.], "future_return": [99., 99., 99.]})
    first = predict.research_return_fit_predict(train, test, ["x"])
    test["future_return"] = [-99., -99., -99.]
    second = predict.research_return_fit_predict(train, test, ["x"])
    np.testing.assert_allclose(first[0], second[0])
    np.testing.assert_allclose(first[1], second[1])
    np.testing.assert_allclose(first[2], second[2])
    assert first[3]["selection_scope"].startswith("僅使用")


def test_research_model_uses_rolling_window_and_nested_intervals():
    rng = np.random.default_rng(7)
    train = pd.DataFrame({
        "x": rng.normal(size=1400),
        "future_return": rng.normal(scale=.03, size=1400),
    })
    test = pd.DataFrame({"x": [-1., 0., 1.]})
    _, low, high, metadata = predict.research_return_fit_predict(train, test, ["x"])
    assert metadata["available_training_samples"] == 1400
    assert metadata["rolling_training_samples"] == predict.RESEARCH_TRAIN_WINDOW
    assert metadata["conformal_radius_50"] <= metadata["conformal_radius"]
    assert metadata["selected_model"] in predict.RESEARCH_RETURN_CANDIDATES
    assert (low <= high).all()


def test_same_bar_stop_and_target_is_conservative():
    market = market_frame()
    market.loc[market.index[1], ["High", "Low"]] = [104, 97]
    rows = market.iloc[[0]].copy()
    rows["probability"], rows["ATR"], rows["regime"] = 0.9, 1.0, "bull"
    trade = predict.simulate(rows, market, 0.7, 5, 1.5, 2.0, 0, 0, 0)[0]
    assert trade.reason == "stop"
    assert trade.net_r == -1.0


def test_trade_outcome_label_matches_profitable_target_and_conservative_stop():
    market = market_frame()
    market["ATR"] = 1.0
    labels = predict.trade_outcome_labels(
        market, 5, 1.5, 2.0, 0, 0, 0, -0.25, 0.25)
    assert labels.iloc[0] == 1.0
    market.loc[market.index[1], ["High", "Low"]] = [104, 97]
    labels = predict.trade_outcome_labels(
        market, 5, 1.5, 2.0, 0, 0, 0, -0.25, 0.25)
    assert labels.iloc[0] == 0.0


def test_trade_outcome_label_counts_unfilled_next_open_as_unsuccessful_event():
    market = market_frame()
    market["ATR"] = 1.0
    market.loc[market.index[1], "Open"] = 102.0
    labels = predict.trade_outcome_labels(
        market, 5, 1.5, 2.0, 14.25, 10, 5, -0.25, 0.25)
    assert labels.iloc[0] == 0.0
    assert labels.iloc[-5:].isna().all()


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


def test_probability_metrics_compare_with_unconditional_baseline():
    rows = pd.DataFrame({"probability": [.1, .2, .8, .9], "label": [0, 0, 1, 1]})
    metrics = predict.probability_metrics(rows, .5)
    assert metrics["base_rate"] == .5
    assert metrics["baseline_brier"] == .25
    assert metrics["brier_skill_score"] > 0


def test_return_metrics_reject_one_sided_direction_predictions():
    rows = pd.DataFrame({
        "future_return": [-.04, -.02, .01, .03],
        "research_predicted_return": [.01, .01, .01, .01],
        "research_predicted_return_low": [-.05] * 4,
        "research_predicted_return_high": [.05] * 4,
    })
    metrics = predict.return_forecast_metrics(rows, "research_predicted_return")
    assert metrics["direction_accuracy"] == .5
    assert metrics["balanced_direction_accuracy"] == .5
    assert not metrics["passed"]


def test_bootstrap_ev_interval_is_reproducible_and_ordered():
    trades = [predict.Trade("", "", "", 1, 1, .8, r, r, "time", "bull")
              for r in [1.0, -.5, .25]]
    first = predict.bootstrap_ev_interval(trades, samples=200)
    second = predict.bootstrap_ev_interval(trades, samples=200)
    assert first == second
    assert first["lower_90"] <= first["median"] <= first["upper_90"]


def test_formal_validation_requires_all_mandatory_gates(monkeypatch):
    def fake_simulate(rows, *_args, **_kwargs):
        return [predict.Trade("", "", "", 1, 1, .8, r, r, "time", "bull", r / 100)
                for r in ([1.0] * 6 + [-.5] * 4)]

    monkeypatch.setattr(predict, "simulate", fake_simulate)
    rows = pd.DataFrame({"fold": [1, 2, 3], "probability": [.1, .2, .9],
                         "label": [0, 0, 1]})
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
    rows = pd.DataFrame({"fold": [1, 2], "probability": [.1, .9], "label": [0, 1]})
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


def test_formal_validation_enforces_24_9pct_drawdown_limit(monkeypatch):
    monkeypatch.setattr(
        predict, "simulate",
        lambda *_args, **_kwargs: [
            predict.Trade("", "", "", 1, 1, .8, 1, 1, "time", "bull", .01)
        ],
    )
    rows = pd.DataFrame({"fold": [1, 2], "probability": [.1, .9], "label": [0, 1]})
    args = SimpleNamespace(threshold=.2, horizon=5, stop_atr=1.5, reward_risk=2,
                           commission_bps=14.25, tax_bps=10, slippage_bps=5,
                           entry_gap_low_atr=.15, entry_gap_high_atr=.55)
    oos = {"trades": 30, "win_rate": .60, "profit_factor": 1.5,
           "max_drawdown_pct": .25}
    final = {"trades": 10, "win_rate": .55}
    validation = predict.formal_validation(rows, pd.DataFrame(), oos, final, args)
    assert not validation["checks"]["max_drawdown_at_most_25pct"]
    assert validation["max_drawdown_limit"] == .249


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
    assert data.future_return.tail(5).isna().all()
    assert data.future_high_return.tail(5).isna().all()
    assert data.future_low_return.tail(5).isna().all()


def test_bollinger_candidate_adds_only_expected_features_without_lookahead():
    idx = pd.date_range("2023-01-01", periods=100, freq="B")
    close = pd.Series(np.linspace(100, 130, 100), index=idx)
    frame = pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1,
                          "Close": close, "Volume": 1000}, index=idx)
    _baseline, baseline_features = predict.build_dataset(
        frame, {}, 5, .04, -.025, "all")
    candidate, candidate_features = predict.build_dataset(
        frame, {}, 5, .04, -.025, "all-bollinger")
    assert set(candidate_features) - set(baseline_features) == {
        "bb_percent_b_20_2", "bb_width_20_2"}

    changed = frame.copy()
    changed.loc[idx[-1], "Close"] = 9999
    changed_candidate, _ = predict.build_dataset(
        changed, {}, 5, .04, -.025, "all-bollinger")
    for name in ("bb_percent_b_20_2", "bb_width_20_2"):
        pd.testing.assert_series_equal(
            candidate.loc[:idx[-2], name], changed_candidate.loc[:idx[-2], name])


def test_intraday_bollinger_research_is_non_actionable_and_uses_completed_history():
    idx = pd.date_range("2026-01-01 09:00", periods=120, freq="15min")
    close = pd.Series(100 + np.sin(np.arange(120) / 8), index=idx)
    frame = pd.DataFrame({"Open": close, "High": close + .2, "Low": close - .2,
                          "Close": close, "Volume": 1000}, index=idx)
    result = predict.intraday_bollinger_research(frame, 99, 101)
    assert result["available"]
    assert not result["actionable"]
    assert not result["affects_formal_signal"]
    assert result["validation_status"] == "待驗證"
    assert result["completed_bars"] == 120
    assert result["latest_bar"].endswith("+08:00")
    assert set(result["windows"]) == {"20", "36", "90"}
    assert result["forward_observations"] == 0

    earlier = predict.intraday_bollinger_research(frame.iloc[:-1], 99, 101)
    changed = frame.copy()
    changed.iloc[-1, changed.columns.get_loc("Close")] = 999
    changed_earlier = predict.intraday_bollinger_research(changed.iloc[:-1], 99, 101)
    assert earlier["windows"] == changed_earlier["windows"]


def test_intraday_download_converts_utc_to_taipei_and_drops_open_bar(monkeypatch):
    now = pd.Timestamp.now(tz="Asia/Taipei")
    completed_start = (now - pd.Timedelta(minutes=30)).floor("15min")
    open_start = now.floor("15min")
    utc_index = pd.DatetimeIndex([completed_start, open_start]).tz_convert("UTC")
    frame = pd.DataFrame({"Open": [100, 101], "High": [101, 102],
                          "Low": [99, 100], "Close": [100.5, 101.5],
                          "Volume": [1000, 1200]}, index=utc_index)
    monkeypatch.setattr(predict.yf, "download", lambda *_args, **_kwargs: frame)
    result = predict._flat_intraday_download("00631L.TW")
    assert result.index.tz is None
    assert result.index[-1] == completed_start.tz_localize(None)
    assert open_start.tz_localize(None) not in result.index


def test_simulation_can_require_positive_predicted_return():
    market = market_frame()
    rows = market.iloc[[0]].copy()
    rows["probability"], rows["ATR"], rows["regime"] = .9, 1., "bull"
    rows["predicted_return"] = -.01
    trades = predict.simulate(
        rows, market, .7, 5, 1.5, 2, 0, 0, 0,
        minimum_predicted_return=.01)
    assert trades == []


def test_return_model_produces_ordered_price_interval():
    idx = pd.date_range("2024-01-01", periods=140, freq="B")
    train = pd.DataFrame({
        "x": np.linspace(-1, 1, 120),
        "future_return": np.linspace(-.05, .05, 120),
    }, index=idx[:120])
    test = pd.DataFrame({"x": [-.25, .25]}, index=idx[120:122])
    median, low, high = predict.return_fit_predict(train, test, ["x"])
    assert np.isfinite(median).all()
    assert (low <= median).all()
    assert (median <= high).all()


def test_record_prediction_rejects_nan_market_price_before_sqlite_write(tmp_path):
    result = {"latest_price": float("nan")}
    with np.testing.assert_raises_regex(ValueError, "market_price must be"):
        predict.record_prediction(
            str(tmp_path / "predictions.sqlite3"), result, pd.Series(),
            pd.DataFrame(), SimpleNamespace())
    assert not (tmp_path / "predictions.sqlite3").exists()


def test_unvalidated_scenario_records_levels_but_keeps_action_not_trading(tmp_path):
    database = tmp_path / "predictions.sqlite3"
    result = {
        "latest_price": 100.0, "latest_date": "2026-09-01",
        "signal": False, "action": "不交易", "latest_probability": .3,
        "valid_until": "2026-09-08", "model_version": "20260901.1",
        "strategy_version": "20260901.2", "data_source_snapshot": {},
        "validation_snapshot": {"passed": False},
        "research_scenario": {
            "available": True, "validated": False, "not_actionable": True,
            "warning": "未驗證研究情境；不可交易",
            "raw_estimated_price": 105.0,
            "raw_estimated_price_low": 95.0,
            "raw_estimated_price_high": 115.0,
            "scenario_entry": 100.0,
            "scenario_entry_low": 99.5, "scenario_entry_high": 100.5,
            "scenario_stop": 97.0,
            "scenario_take_profit_1": 103.0,
            "scenario_take_profit_2": 107.5,
            "failed_validations": ["formal_validation"],
            "assumptions": {"stop_atr": 1.5},
            "valid_until": "2026-09-08",
        },
        "oos_trading": {"win_rate": .5},
        "execution_plan": {
            "suggested_entry": 100.0, "entry_low": 99.5,
            "entry_high": 100.5, "tranches": [],
            "position_sizing_denominator": "配置給 00631L 的資金",
            "stop": 97.0, "take_profit_1": 103.0,
            "take_profit_2": 107.5, "reward_risk_2": 2.5,
        },
    }
    prediction_id = predict.record_prediction(
        str(database), result, pd.Series(dtype=float),
        pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]),
        SimpleNamespace())
    with sqlite3.connect(database) as con:
        row = con.execute("""SELECT action,predicted_price,buy_price,buy_range_low,buy_range_high,
            stop_loss,take_profit_1,take_profit_2,validation_snapshot
            FROM predictions WHERE id=?""", (prediction_id,)).fetchone()
        research = con.execute("""SELECT scenario_entry,scenario_stop,
            scenario_take_profit_1,scenario_take_profit_2,not_actionable
            FROM prediction_research_scenarios WHERE prediction_id=?""",
            (prediction_id,)).fetchone()
    assert row[:8] == ("不交易", 105.0, 100.0, 99.5, 100.5, 97.0, 103.0, 107.5)
    snapshot = json.loads(row[8])
    assert snapshot["research_scenario"]["not_actionable"]
    assert research == (100.0, 97.0, 103.0, 107.5, 1)
    with sqlite3.connect(database) as con:
        with np.testing.assert_raises_regex(sqlite3.IntegrityError, "不得修改"):
            con.execute("""UPDATE prediction_research_scenarios
                SET scenario_entry=1 WHERE prediction_id=?""", (prediction_id,))


def test_legacy_database_migration_preserves_rows_and_makes_predictions_immutable(tmp_path):
    database = tmp_path / "predictions.sqlite3"
    with sqlite3.connect(database) as con:
        con.execute("""CREATE TABLE predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, predicted_at TEXT NOT NULL,
            ticker TEXT NOT NULL, market_date TEXT NOT NULL, market_price REAL NOT NULL,
            action TEXT NOT NULL, suggested_entry REAL, entry_low REAL, entry_high REAL,
            stop_price REAL, take_profit_1 REAL, take_profit_2 REAL,
            model_probability REAL, backtest_win_rate REAL, valid_until TEXT NOT NULL,
            model_version TEXT NOT NULL, strategy_version TEXT NOT NULL,
            indicators_json TEXT NOT NULL, reason TEXT NOT NULL, market_regime TEXT NOT NULL,
            actual_high REAL, actual_low REAL, actual_close REAL, stop_touched INTEGER,
            target1_touched INTEGER, target2_touched INTEGER, actual_return REAL,
            trade_result TEXT, prediction_success INTEGER, settled_at TEXT)""")
        con.execute("""INSERT INTO predictions VALUES (
            1,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            ("2026-08-18T16:30:00+08:00", "00631L.TW", "2026-08-18", 34.81,
             "不交易", None, None, None, None, None, None, .33, .45, "2026-08-25",
             "old-model", "old-strategy", "{}", "legacy", "bull", 36.0, 33.0,
             35.0, 0, 0, 0, .01, "到期", None, "2026-08-25T16:30:00+08:00"))
    backup = predict._migrate_legacy_database(str(database))
    assert backup is not None and Path(backup).exists()
    with sqlite3.connect(backup) as backup_con:
        assert backup_con.execute("SELECT count(*) FROM predictions").fetchone()[0] == 1
        assert "ticker" in {
            row[1] for row in backup_con.execute("PRAGMA table_info(predictions)")}
    with sqlite3.connect(database) as con:
        assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert con.execute("SELECT count(*) FROM predictions").fetchone()[0] == 1
        assert con.execute("SELECT count(*) FROM prediction_outcomes").fetchone()[0] == 1
        assert con.execute(
            "SELECT count(*) FROM prediction_research_scenarios").fetchone()[0] == 0
        assert con.execute("SELECT symbol FROM predictions").fetchone()[0] == "00631L"
        with np.testing.assert_raises_regex(sqlite3.IntegrityError, "不得修改"):
            con.execute("UPDATE predictions SET market_price=1 WHERE id=1")
        with np.testing.assert_raises_regex(sqlite3.IntegrityError, "不得刪除"):
            con.execute("DELETE FROM predictions WHERE id=1")
