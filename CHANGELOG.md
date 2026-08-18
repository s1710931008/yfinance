# Changelog

本文件記錄 00631L 股票買賣價位預測模型、交易策略、回測規範、資料處理、資料庫 schema 與輸出格式的所有重要變更。

格式參考 Keep a Changelog；模型版本與策略版本使用 `YYYYMMDD.N`，並分開管理。

## [Unreleased]

### Added

- 新增每日 workflow 完成後的 Discord 通知，包含分析／部署狀態、行情日期、目前價位、正式結論、驗證狀態、儀表板及 Actions 執行連結；Webhook 僅由 GitHub Actions Secret 讀取。
- 新增 GitHub Actions 工作流程，支援每個交易日台北時間 16:30 排程及手動執行。
- 新增依賴清單、測試步驟、分析結果 artifact 與 SQLite cache 延續機制。
- 新增 `.gitignore`，避免將本機虛擬環境與執行暫存檔提交至版本庫。
- 新增 GitHub Pages 發布工作，將最新分析結果提供為固定的 `latest.json` GET 網址。
- 新增公開結果入口頁與強制免責聲明；SQLite 仍只保存在非公開的 workflow artifact/cache。
- 新增中文預測儀表板，直接呈現今日資料、Actions 執行進度、模型機率、OOS／Final Test、驗證門檻、市場分層、風險與 prediction ID。
- 新增扣成本逐筆複利的最大回撤百分比、Walk-Forward 各 fold 勝率穩定性、開發期 Walk-Forward OOS 與獨立 Final Test 勝率差異及正式 `validation_snapshot`。
- 儀表板新增目前價位、條件買進價與區間、分批比例、停損、兩段停利、風險報酬比、成立／失效條件及有效期限欄位。

### Changed

- 模型與策略版本更新為 `20260819.1`；OHLCV 缺值清理明確排除非有限或非正數價格列，報酬計算改用 `pct_change(fill_method=None)`，不再由 pandas 隱含向前填補缺值。
- 自 Git 追蹤中移除 `.venv`；僅影響 repository 大小，不刪除本機虛擬環境。
- README 補充 GitHub Actions 操作方式、保存期限及非永久備份限制。
- README 補充 GET 網址、GitHub Pages 首次啟用方式與公開資料限制。
- Pages workflow 或公開入口檔案推送至 `main` 時會自動執行，避免部署設定更新後仍需額外手動觸發。
- 首頁由單一 JSON 連結改為動態讀取 `latest.json` 與公開 GitHub Actions API；進度與最近一次成功發布結果分開呈現。
- 模型與策略版本更新為 `20260818.1`；正式執行價位必須同時通過完整模型驗證、既有交易閘門、嚴格進場驗證與最新訊號。
- SQLite snapshot 同一 transaction 保存指標、正式驗證、資料來源及執行計畫，並明確使用 Asia/Taipei 時間；啟用 WAL、foreign key 與 busy timeout。
- 將 checkout、setup-python、cache、artifact 與 GitHub Pages 官方 Actions 升級至 Node.js 24 版本，移除 Node.js 20 淘汰與舊 `punycode` 相依警告。

### Fixed

- 修正 Yahoo Finance 回傳尾端不完整行情列時，最新收盤價成為 `NaN` 並造成 SQLite `market_price NOT NULL` 寫入失敗；寫入前新增有限正數防護，禁止以舊值或推測值冒充當日行情。
- 修正進場缺口上限測試的參數傳遞，使測試明確設定區間上下限；正式策略參數與程式邏輯不變。
- 修正同一 workflow run 重跑 job 時產生多個同名 Pages artifact，導致 `deploy-pages` 無法選擇部署檔案；artifact 名稱現在包含 `github.run_attempt`，deploy 只取當次 attempt。

### Impact

- 缺值日期將不產生正式預測紀錄；有效資料列可能略減，但指標參數、交易規則與驗證門檻不變。不預設勝率、PF、回撤或交易頻率改善，且未新增參數搜尋，不增加過度擬合風險。
- Discord 通知只讀取既有分析 artifact，不修改模型、策略、參數、驗證門檻、SQLite 或預測結果；不影響訊號、勝率、交易頻率、報酬、最大回撤、Profit Factor、風險報酬比及過度擬合風險。
- 模型演算法、技術指標、交易規則、參數、資料處理及 SQLite schema 均未變更。
- 預期不影響訊號、勝率、交易頻率、報酬、最大回撤、Profit Factor 或風險報酬比。
- 未新增參數搜尋或模型選擇，因此不增加模型過度擬合風險。
- 發布內容只包含既有 `validation.json`；不公開 SQLite，亦不提供新的推測資料。
- Actions 執行環境相依升級不修改模型或策略行為；可能只影響 CI 啟動、cache 與 artifact 處理方式。
- 儀表板只轉譯既有結果；必要驗證資料缺少時標示「資料不足」並禁止顯示可執行價格。
- 新驗證閘門可能降低交易訊號與頻率；不更動模型超參數或進出場參數，不預設勝率、報酬、PF 或回撤會改善，也未增加參數搜尋造成的過度擬合風險。

### Validation

- `20260819.1` 已完成完整 10 年 Walk-Forward 與獨立 OOS：最新有效行情為 2026-08-17、收盤 35.95，未再取得尾端 `NaN`；68 筆 OOS 交易、勝率 58.8%、PF 1.549、最大回撤 19.8%，Final Test 50 筆、勝率差 7.2 個百分點。
- `20260819.1` 正式驗證仍未通過：5 folds 中 2 folds 無有效交易；嚴格進場驗證中位 PF 0.821、EV -0.087R，雙倍成本 PF 0.585。因此結論維持「不交易」，不提供買進、停損或停利價。
- 本機 14 項測試、workflow YAML、臨時 SQLite 實際寫入與 `PRAGMA integrity_check` 全部通過；寫入紀錄價格 35.95、動作「不交易」、模型與策略版本 `20260819.1`。
- Discord 通知 workflow YAML、shell 結構及本機 11 項測試通過；GitHub-hosted runner 實際通知待推送及設定 repository secret 後確認。
- 本機測試：8 項全部通過。
- GitHub Actions workflow YAML 靜態解析通過。
- 尚未取得首次 GitHub-hosted runner 執行結果；上線後由手動或排程執行確認。
- GitHub Pages GET 發布待推送後進行首次部署驗證。
- 針對首次 GET 取得 404，補上 push 觸發並保留手動與平日排程觸發方式。
- Node.js 24 Actions 升級後，本機 8 項測試與 workflow YAML 靜態檢查均通過；GitHub-hosted runner 驗證待推送後確認。
- Run #8 證實 Node.js 24 Actions 可完成預測與 artifact 上傳；Pages 因同名 artifact 重複而失敗，已加入 run-attempt 去重修正，待新 run 驗證。
- 中文儀表板已通過 HTML 解析、JavaScript 語法、workflow YAML 與本機 8 項測試；GitHub Pages 實際部署與視覺驗證待推送後確認。
- 2026-08-18 完整 10 年 Walk-Forward 與獨立 Final Test：正式驗證未通過；68 筆 OOS 交易、勝率 58.8%、PF 1.549、最大回撤 19.8%，Final Test 51 筆、勝率 62.7%，勝率差 3.9 個百分點。5 folds 中 2 folds 無有效交易，因此 fold 穩定性資料不足／未通過。
- 9 組嚴格進場驗證未通過：中位 60 筆、勝率 45.8%、EV -0.102R、PF 0.778、最大回撤 13.132R；雙倍成本 EV -0.230R、PF 0.565。結論為「不交易」，不得提供買進、停損或停利價。
- 本機 11 項測試、HTML／JavaScript、workflow YAML 與 SQLite transaction／integrity_check 全部通過；GitHub-hosted runner 與 Pages 部署待推送後確認。
- 正式排程執行仍須通過既有 Walk-Forward 與獨立 OOS 驗證門檻；失敗時不得提供買賣建議。

### Version

- 文件／網站版本：`20260819.1`
- 模型版本：`20260819.1`。
- 策略版本：`20260819.1`。
- 使用者確認：2026-08-18，使用者回覆「掛上」。
- 使用者確認：2026-08-18，使用者回覆「確認升級 Node 24 Actions」。
- 使用者確認：2026-08-18，使用者確認建立 latest.json 中文儀表板與 Actions 執行進度。
- 使用者確認：2026-08-18，使用者回覆「確認補齊價位與買賣點驗證」。
- 使用者確認：2026-08-19，使用者要求排程完成後通知 Discord 並附儀表板連結。
- 使用者確認：2026-08-19，使用者回覆「確認修正 NaN」。

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
