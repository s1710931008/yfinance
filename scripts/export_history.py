#!/usr/bin/env python3
"""Export a sanitized, read-only prediction history for GitHub Pages."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
from pathlib import Path
from zoneinfo import ZoneInfo


DISCLAIMER = (
    "本分析僅供研究與參考，不構成投資建議；00631L 為槓桿型 ETF，"
    "使用者應自行承擔交易風險與損益。"
)
TAIPEI = ZoneInfo("Asia/Taipei")


def export_history(database: str, limit: int = 30) -> dict[str, object]:
    if limit < 1 or limit > 100:
        raise ValueError("limit must be between 1 and 100")
    uri = f"file:{Path(database).resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("""
            SELECT p.id, p.predicted_at,
                   json_extract(p.data_source_snapshot, '$.market_date') AS market_date,
                   p.market_price, p.action, p.model_probability, p.valid_until,
                   p.model_version, p.strategy_version, p.buy_price AS suggested_entry,
                   p.buy_range_low AS entry_low, p.buy_range_high AS entry_high,
                   p.stop_loss AS stop_price, p.take_profit_1, p.take_profit_2,
                   o.actual_close, o.actual_return_pct AS actual_return,
                   o.prediction_success, o.resolved_at AS settled_at
            FROM predictions p
            LEFT JOIN prediction_outcomes o ON o.prediction_id=p.id
            ORDER BY p.id DESC
            LIMIT ?
        """, (limit,)).fetchall()
    records = []
    for row in rows:
        actionable = row["action"] != "不交易"
        settled = row["settled_at"] is not None
        has_trade_levels = actionable and all(
            row[name] is not None for name in (
                "suggested_entry", "entry_low", "entry_high", "stop_price",
                "take_profit_1", "take_profit_2"))
        records.append({
            "id": row["id"],
            "predicted_at": row["predicted_at"],
            "market_date": row["market_date"],
            "market_price": row["market_price"],
            "action": row["action"],
            "model_probability": row["model_probability"],
            "valid_until": row["valid_until"],
            "model_version": row["model_version"],
            "strategy_version": row["strategy_version"],
            "trade_levels_available": has_trade_levels,
            "suggested_entry": row["suggested_entry"] if has_trade_levels else None,
            "entry_low": row["entry_low"] if has_trade_levels else None,
            "entry_high": row["entry_high"] if has_trade_levels else None,
            "stop_price": row["stop_price"] if has_trade_levels else None,
            "take_profit_1": row["take_profit_1"] if has_trade_levels else None,
            "take_profit_2": row["take_profit_2"] if has_trade_levels else None,
            "settlement_status": "已結算" if settled else "等待結算",
            "actual_close": row["actual_close"] if settled else None,
            "actual_return_pct": (row["actual_return"] if settled and actionable else None),
            "prediction_success": (bool(row["prediction_success"])
                                   if settled and actionable and row["prediction_success"] is not None
                                   else None),
            "counts_as_trade": actionable,
        })
    return {
        "generated_at": dt.datetime.now(TAIPEI).isoformat(),
        "timezone": "Asia/Taipei",
        "limit": limit,
        "count": len(records),
        "disclaimer": DISCLAIMER,
        "privacy": "僅公開查詢必要欄位及原始交易價位；不包含 SQLite、技術指標 snapshot 或內部推理。",
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()
    payload = export_history(args.database, args.limit)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"公開歷史紀錄已寫入：{output}（{payload['count']} 筆）")
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
