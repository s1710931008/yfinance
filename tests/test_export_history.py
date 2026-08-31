import importlib.util
import sqlite3
from pathlib import Path
import sys


SPEC = importlib.util.spec_from_file_location(
    "export_history", Path(__file__).parents[1] / "scripts" / "export_history.py")
export_history = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = export_history
SPEC.loader.exec_module(export_history)


def make_database(path):
    with sqlite3.connect(path) as con:
        con.execute("""CREATE TABLE predictions (
            id INTEGER PRIMARY KEY, predicted_at TEXT, market_price REAL,
            action TEXT, model_probability REAL, valid_until TEXT,
            model_version TEXT, strategy_version TEXT, buy_price REAL,
            buy_range_low REAL, buy_range_high REAL, stop_loss REAL,
            take_profit_1 REAL, take_profit_2 REAL, data_source_snapshot TEXT)""")
        con.execute("""CREATE TABLE prediction_outcomes (
            id INTEGER PRIMARY KEY, prediction_id INTEGER, actual_close REAL,
            actual_return_pct REAL, prediction_success INTEGER, resolved_at TEXT)""")
        con.executemany("INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [
            (1, "2026-08-18T16:30:00+08:00", 34.81, "不交易", .33,
             "2026-08-25", "m1", "s1", None, None, None, None, None, None,
             '{"market_date":"2026-08-18"}'),
            (2, "2026-08-19T16:30:00+08:00", 35.0, "買進", .72,
             "2026-08-26", "m2", "s2", 35.2, 35.0, 35.4, 34.0, 36.4, 37.6,
             '{"market_date":"2026-08-19"}'),
        ])
        con.execute("INSERT INTO prediction_outcomes VALUES (?,?,?,?,?,?)",
                    (1, 1, 36.0, .03, None, "2026-08-25T16:30:00+08:00"))


def test_export_history_is_sanitized_and_newest_first(tmp_path):
    database = tmp_path / "predictions.sqlite3"
    make_database(database)
    payload = export_history.export_history(str(database), 30)
    assert [row["id"] for row in payload["records"]] == [2, 1]
    assert payload["records"][0]["settlement_status"] == "等待結算"
    assert payload["records"][0]["counts_as_trade"] is True
    assert payload["records"][0]["trade_levels_available"] is True
    assert payload["records"][0]["suggested_entry"] == 35.2
    assert payload["records"][0]["entry_low"] == 35.0
    assert payload["records"][0]["take_profit_2"] == 37.6
    no_trade = payload["records"][1]
    assert no_trade["settlement_status"] == "已結算"
    assert no_trade["counts_as_trade"] is False
    assert no_trade["actual_return_pct"] is None
    assert no_trade["prediction_success"] is None
    assert no_trade["trade_levels_available"] is False
    assert no_trade["suggested_entry"] is None
    assert no_trade["stop_price"] is None
    assert "disclaimer" in payload
    assert "indicators_json" not in no_trade


def test_export_history_limit_validation(tmp_path):
    database = tmp_path / "predictions.sqlite3"
    make_database(database)
    try:
        export_history.export_history(str(database), 0)
    except ValueError as exc:
        assert "between 1 and 100" in str(exc)
    else:
        raise AssertionError("invalid limit was accepted")
