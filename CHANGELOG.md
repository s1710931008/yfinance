# Changelog

本文件記錄 00631L 股票買賣價位預測模型、交易策略、回測規範、資料處理、資料庫 schema 與輸出格式的所有重要變更。

格式參考 Keep a Changelog；模型版本與策略版本使用 `YYYYMMDD.N`，並分開管理。

## [Unreleased]

### Added

- 預留尚未正式發布的修改項目。

## [20260818.1] - 2026-08-18

### Added

- 建立完整 `AGENTS.md` 規範。
- 加入固定技術指標與參數：KD 9,3,3、RSI 14、20 日均量比、5／20／60 日均線、MACD 12,26,9、ATR 14、60 日支撐壓力，以及加權指數與台積電同步比對。
- 加入 Walk-Forward Validation、獨立 out-of-sample 驗證及防止 look-ahead bias 的規則。
- 加入模型驗證門檻：樣本數、fold 勝率標準差、Profit Factor、最大回撤與 OOS 落差。
- 加入模型預測機率與歷史回測勝率分開呈現的規則。
- 加入 SQLite `predictions` 與 `prediction_outcomes` schema、不可更新／刪除 trigger、WAL 與 foreign key 設定。
- 加入模型版本、策略版本、概念漂移、資料庫備份及 PostgreSQL 擴充規則。
- 加入標準分析輸出順序與強制免責聲明。
- 加入 `CHANGELOG.md` 強制維護規範。

### Changed

- 將「模型未通過驗證不得提供建議」明確擴充至行情無法確認、回測失敗、套件缺失及資料來源錯誤等情況。
- 補充交易成本、滑價、槓桿摩擦與資料可重現性要求。
- 補充資金比例必須標明是占 00631L 配置資金或使用者總資金。
- SQLite schema 新增 `timezone`、`risk_reward_ratio`、`validation_snapshot` 與 `data_source_snapshot`，並增加機率與區間檢查。
- 補充同日同時可能觸及停損與停利但無法判斷先後時的保守處理規則。
- 明確規定分析／檢查要求不等於授權修改程式碼或參數。

### Validation

- 文件與 schema 規則已完成靜態檢查。
- 尚未修改或執行實際預測模型程式碼。
- 尚未產生 Walk-Forward 或 out-of-sample 回測結果。
- 模型驗證狀態：**待驗證，不得據此提供買賣建議**。

### Version

- 文件版本：`20260818.1`
- 模型版本：尚未建立或變更。
- 策略版本：尚未建立或變更。
- 使用者確認：本版依 2026-08-18 提供之規格整理；後續程式碼修改仍須另行取得明確確認。
