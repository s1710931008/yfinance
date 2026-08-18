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
            SELECT id, predicted_at, market_date, market_price, action,
                   model_probability, valid_until, model_version, strategy_version,
                   actual_close, actual_return, prediction_success, settled_at
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()
    records = []
    for row in rows:
        actionable = row["action"] != "不交易"
        settled = row["settled_at"] is not None
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
        "privacy": "僅公開查詢必要欄位；不包含 SQLite、技術指標 snapshot 或內部推理。",
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
