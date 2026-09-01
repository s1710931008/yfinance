**# AGENTS.md**

**## 免責聲明**

本文件所定義之所有分析、預測、買賣建議、勝率與回測結果，均基於歷史資料與統計模型計算，**\*\*僅供研究與參考，不構成任何投資建議\*\***。00631L 為槓桿型 ETF，波動風險與追蹤誤差顯著高於一般股票，使用者應自行承擔投資決策之風險與損益。所有輸出內容必須附帶此聲明，不得省略。

\---

**## 一、00631L 股票買賣價位預測模型**

**### 1. 分析指標與固定參數**

預測模型必須綜合分析下列指標，不得只使用單一技術指標判斷。預設參數固定如下；如需調整，必須依「模型修改流程」取得使用者確認，並記錄於模型版本說明與 \`CHANGELOG.md\`。

\| 指標 | 預設參數 |

\|---|---|

\| KD 指標 | 9,3,3 |

\| RSI 指標 | 14 日 |

\| 成交量與價格關係 | 20 日均量比 |

\| 均線與趨勢 | 5／20／60 日均線 |

\| MACD | 12,26,9 |

\| ATR 波動度 | 14 日 |

\| 支撐位與壓力位 | 近 60 日高低點＋近期成交密集區 |

\| 大盤與台積電走勢 | 加權指數、台積電（2330）日線同步比對 |

**### 2. 建議產出前提**

只要必要行情、技術指標與價格計算流程可正常取得／完成，即可直接產生價格預測與買賣價位；**模型驗證是否通過，不再作為輸出價格的前置條件**。

每次分析至少直接提供：

\- **預測價**。

\- **買進價**（可同時提供買進區間）。

\- **停損價**。

\- **停利價**（至少第一停利；如模型有第二目標可一併提供）。

以下情況即使成立，仍可輸出上述價格，但必須如實標示驗證狀態，不得把未完成或未通過的驗證描述為已通過：

\- 回測有效交易樣本數少於 30 筆。

\- 第 4 節任一模型驗證門檻未通過。

\- Walk-Forward Validation 或獨立 out-of-sample 驗證尚未完成。

\- Profit Factor、最大回撤、fold 穩定性或 OOS 差異未達門檻。

只有以下情況不得產生或虛構價格，必須輸出「資料不足」：

\- 即時行情或必要市場資料無法取得、資料延遲到無法判斷，或完整性無法確認。

\- 必要技術指標無法計算。

\- 價格計算程式執行失敗、套件缺失或資料來源錯誤，導致無法形成合理價位。

模型驗證結果屬於**風險與可信度參考資訊**，不是阻擋價格輸出的 gate。若驗證未通過，應明確標示「模型未通過驗證／待驗證」，但仍可正式輸出預測價、買進價、停損價與停利價。

**### 3. 回測規範**

\- 採用 **\*\*Walk-Forward Validation\*\***，避免資料洩漏（look-ahead bias）與過度擬合。

  \- 每個 fold 的訓練窗口至少 252 個交易日（約 1 年）。

  \- 每個 fold 的測試窗口原則上為 60 個交易日（約 1 季）。

  \- 各窗口依時間順序滾動前進，不得隨機切分時間序列。

  \- 參數優化只能在訓練窗口內進行；測試窗口資料不得在優化過程中被使用或觀察。

\- 回測報酬必須納入交易成本：

  \- 買進手續費：預設 0.1425%，若使用折扣費率必須明確記錄。

  \- 賣出手續費：預設 0.1425%，若使用折扣費率必須明確記錄。

  \- 證交稅：依回測當時適用法規設定，預設值需在輸出中揭露。

  \- 預估滑價：必須設定並揭露，不得假設永遠以訊號價成交。

\- 00631L 為槓桿型 ETF，回測須額外考慮槓桿摩擦成本，包括日複利侵蝕、期貨轉倉成本與追蹤誤差。不得僅以現貨報酬乘以 2 估算長期報酬。

\- 參數優化完成後，必須在完全未參與優化的 **\*\*out-of-sample\*\*** 期間（例如最近 3 個月）再次驗證，作為過度擬合檢查。

\- 回測應記錄資料來源、資料期間、資料下載時間、時區、缺值處理、除權息處理及交易成本假設，確保結果可重現。

**### 4. 模型驗證通過標準**

驗證結果用於評估模型可信度、穩定性與風險，**不作為是否可以輸出價格的必要條件**。有資料時應盡量逐項列出以下狀態（通過／未通過／待驗證／資料不足）：

\| 驗證項目 | 通過標準 |

\|---|---|

\| 有效交易樣本數 | ≥ 30 筆；低於門檻標示「樣本不足」 |

\| Walk-Forward fold 勝率穩定性 | 各 fold 勝率標準差 ≤ 15 個百分點 |

\| Profit Factor | ≥ 1.2 |

\| 最大回撤 | ≤ 可承受風險上限；預設 25% |

\| OOS 與訓練期勝率差異 | ≤ 10 個百分點 |

任一項未通過時，整體狀態應標示為 **「模型未通過驗證」**；尚未完成則標示 **「待驗證」**。上述狀態不得被包裝成已驗證，但**不影響預測價、買進價、停損價與停利價的輸出**。

**### 5. 市場狀態分類**

市場狀態供勝率分層統計使用，以 60 日均線斜率與加權指數同步趨勢判斷：

\- **\*\*多頭\*\***：00631L 的 60 日均線向上，且加權指數站上其 60 日均線。

\- **\*\*空頭\*\***：00631L 的 60 日均線向下，且加權指數跌破其 60 日均線。

\- **\*\*盤整\*\***：00631L 的 60 日均線斜率介於 ±0.1% 之間。

\- 若條件互相衝突或資料不足，必須揭露判定依據，不得任意歸類。

**### 6. 分析結果必要欄位**

只要即時行情與必要計算資料可確認，分析結果必須明確提供：

\- 使用的模型版本與策略版本。

\- 資料時間、行情時間、資料來源及時區。

\- 預測價。

\- 建議：買進、持有、賣出或不交易。

\- 建議買進價格及建議買進區間。

\- 分批買進價格與資金比例。

\- 資金比例的分母，必須明確標示為「配置給 00631L 的資金」或「使用者總資金」，不得混用。

\- 加碼或攤平條件。

\- 停損價格。

\- 第一停利價格。

\- 第二停利價格。

\- 風險報酬比及其計算基準。

\- 模型預測勝率（模型輸出機率）。

\- 回測交易勝率（歷史統計結果）。

\- 預測信心、主要風險與訊號失效條件。

\- 完整回測摘要及第 4 節各項通過狀態。

\- SQLite 預測紀錄 ID；若寫入失敗，必須明確標示且不得聲稱已完成紀錄。

**### 7. 價格計算原則**

建議價格必須根據預測當下可取得的即時行情、技術指標、ATR、支撐壓力等可確認資料計算；回測與模型驗證結果若可取得，作為可信度與風險參考，不得因尚未通過驗證而阻擋價格輸出。不得在行情或必要計算資料不足時憑空填入價格。

輸出必須同時說明：

\- 價格成立條件，例如在支撐位附近、RSI 低於特定門檻、成交量符合條件。

\- 價格失效條件，例如跌破關鍵支撐、大盤轉空或超過有效期限。

\- 建議有效期限 \`valid\_until\`。

\- 若市場價格已離開建議區間，不得沿用舊建議，必須重新產生並新增預測紀錄。

**### 8. 勝率呈現規範**

勝率必須來自實際回測，並至少提供：

\- 回測期間。

\- 有效交易樣本數。

\- 獲利交易次數。

\- 虧損交易次數。

\- 整體交易勝率。

\- 最近一年交易勝率。

\- 多頭、空頭、盤整的分層勝率。

\- Profit Factor。

\- 最大回撤。

\- 平均盈虧比。

勝率公式：

\`\`\`text

勝率 = 獲利交易次數 ÷ 有效交易總次數 × 100%

\`\`\`

**\*\*模型預測機率不得當成實際交易勝率。兩者必須分開呈現並明確標示來源。\*\***

**### 9. 不交易與資料不足輸出**

模型驗證未通過、樣本不足或尚未完成 Walk-Forward／OOS 驗證時，**不等於資料不足**。只要必要行情與計算成功，仍應直接輸出：

\- 預測價。

\- 買進價／買進區間。

\- 停損價。

\- 停利價。

\- 驗證狀態與主要風險。

只有行情無法確認、必要指標無法計算或價格計算流程失敗時，才輸出「資料不足」，並說明缺少或失敗項目。

不得為了完成格式而虛構行情、價格、勝率或回測結果。模型驗證未通過時可以有價格，但驗證結果必須照實呈現。

\---

**## 二、模型修改流程**

**### 1. 修改前說明**

修改模型、交易策略、參數、資料處理方式、回測規則、資料庫 schema 或程式碼之前，必須先向使用者說明：

\- 預計修改的內容。

\- 修改原因。

\- 對回測結果、勝率、交易頻率與交易訊號可能造成的影響。

\- 對最大回撤、Profit Factor 與風險報酬比可能造成的影響。

\- 是否可能導致過度擬合。

\- 對應的 Walk-Forward 與 out-of-sample 驗證計畫。

\- 預計使用的新模型版本或策略版本。

**### 2. 使用者確認**

必須取得使用者明確確認後才能修改。使用者僅要求「分析、檢查、提出建議或比較」時，不視為已授權修改程式碼、參數或資料庫。

**### 3. 修改後驗證**

修改完成後應：

\- 重新執行完整 Walk-Forward 回測與獨立 out-of-sample 驗證，以評估修改後的可信度與風險。

\- 比較修改前後的有效樣本數、勝率、總報酬、最大回撤、Profit Factor、平均盈虧比及交易訊號差異。

\- 建立新模型版本或策略版本，不得覆蓋舊版本。

\- 更新 `CHANGELOG.md`，記錄變更內容、原因、影響、驗證結果與版本號。

\- 若修改後尚未驗證或未通過驗證，必須標示「待驗證」或「模型未通過驗證」，**但只要必要行情與價格計算成功，仍可產生預測價、買進價、停損價與停利價**。

\---

**## 三、預測紀錄與模型修正**

**### 1. 紀錄義務**

每次產生 00631L 買賣建議時，必須將預測結果寫入 SQLite，不得只顯示於畫面或終端機。所有業務時間戳記統一使用 **\*\*Asia/Taipei（UTC+8）\*\*** 的 ISO 8601 格式並保留時區偏移，例如 \`2026-08-18T09:30:00+08:00\`。

只有「資料不足」且未形成任何價格預測的診斷輸出，可不寫入 \`predictions\`；但應寫入應用程式日誌，記錄失敗原因。

**### 2. SQLite Schema**

\`\`\`sql

PRAGMA journal\_mode = WAL;

PRAGMA foreign\_keys = ON;

CREATE TABLE IF NOT EXISTS predictions (

    id                   INTEGER PRIMARY KEY AUTOINCREMENT,

    predicted\_at         TEXT NOT NULL,      -- ISO8601, Asia/Taipei

    timezone             TEXT NOT NULL DEFAULT 'Asia/Taipei',

    symbol               TEXT NOT NULL CHECK(symbol = '00631L'),

    market\_price         REAL NOT NULL,

    predicted\_price      REAL,               -- 模型預測目標價

    action               TEXT NOT NULL CHECK(action IN ('買進','持有','賣出','不交易')),

    buy\_price            REAL,

    buy\_range\_low        REAL,

    buy\_range\_high       REAL,

    position\_sizing      TEXT,               -- JSON：分批價格、比例及比例分母

    stop\_loss            REAL,

    take\_profit\_1        REAL,

    take\_profit\_2        REAL,

    risk\_reward\_ratio    REAL,

    model\_probability    REAL,               -- 模型預測機率，0～1

    backtest\_winrate     REAL,               -- 歷史回測勝率，0～1

    valid\_until          TEXT NOT NULL,

    model\_version        TEXT NOT NULL,

    strategy\_version     TEXT NOT NULL,

    indicators\_snapshot  TEXT,               -- JSON：KD/RSI/MACD/ATR/均線等

    validation\_snapshot  TEXT,               -- JSON：各驗證門檻與結果

    data\_source\_snapshot TEXT,               -- JSON：來源、資料期間、行情時間

    reasoning            TEXT,

    market\_state         TEXT CHECK(market\_state IN ('多頭','空頭','盤整')),

    created\_at           TEXT NOT NULL DEFAULT (

        strftime('%Y-%m-%dT%H:%M:%S', 'now', '+8 hours') || '+08:00'

    ),

    CHECK(model\_probability IS NULL OR model\_probability BETWEEN 0 AND 1),

    CHECK(backtest\_winrate IS NULL OR backtest\_winrate BETWEEN 0 AND 1),

    CHECK(buy\_range\_low IS NULL OR buy\_range\_high IS NULL OR buy\_range\_low <= buy\_range\_high)

);

CREATE TABLE IF NOT EXISTS prediction\_outcomes (

    id                    INTEGER PRIMARY KEY AUTOINCREMENT,

    prediction\_id         INTEGER NOT NULL REFERENCES predictions(id),

    actual\_high           REAL,

    actual\_low            REAL,

    actual\_close          REAL,

    hit\_stop\_loss         INTEGER CHECK(hit\_stop\_loss IN (0,1)),

    hit\_take\_profit\_1     INTEGER CHECK(hit\_take\_profit\_1 IN (0,1)),

    hit\_take\_profit\_2     INTEGER CHECK(hit\_take\_profit\_2 IN (0,1)),

    actual\_return\_pct     REAL,

    trade\_result          TEXT,

    prediction\_success    INTEGER CHECK(prediction\_success IN (0,1)),

    resolved\_at           TEXT NOT NULL,      -- ISO8601, Asia/Taipei

    UNIQUE(prediction\_id)

);

CREATE TABLE IF NOT EXISTS prediction\_research\_scenarios (

    prediction\_id              INTEGER PRIMARY KEY REFERENCES predictions(id),

    raw\_estimated\_price        REAL NOT NULL,

    raw\_estimated\_price\_low    REAL NOT NULL,

    raw\_estimated\_price\_high   REAL NOT NULL,

    scenario\_entry             REAL NOT NULL,

    scenario\_entry\_low         REAL NOT NULL,

    scenario\_entry\_high        REAL NOT NULL,

    scenario\_stop              REAL NOT NULL,

    scenario\_take\_profit\_1     REAL NOT NULL,

    scenario\_take\_profit\_2     REAL NOT NULL,

    validated                  INTEGER NOT NULL CHECK(validated IN (0,1)),

    not\_actionable             INTEGER NOT NULL CHECK(not\_actionable = 1),

    warning                    TEXT NOT NULL,

    failed\_validations         TEXT NOT NULL,

    assumptions                TEXT NOT NULL,

    valid\_until                TEXT NOT NULL,

    created\_at                 TEXT NOT NULL

);

CREATE TRIGGER IF NOT EXISTS prevent\_prediction\_update

BEFORE UPDATE ON predictions

BEGIN

    SELECT RAISE(ABORT, '原始預測紀錄不得修改，請新增一筆紀錄');

END;

CREATE TRIGGER IF NOT EXISTS prevent\_prediction\_delete

BEFORE DELETE ON predictions

BEGIN

    SELECT RAISE(ABORT, '原始預測紀錄不得刪除');

END;

\`\`\`

\`predictions\` 為不可修改的原始預測紀錄；\`prediction\_outcomes\` 用於補寫實際結果，不得回寫或覆蓋原始預測。
\`prediction\_research\_scenarios\` 以 prediction ID 一對一保存未驗證研究情境，必須與原始預測在同一 transaction 新增，不得寫入正式交易欄位或交由 outcome 結算。

**### 3. 寫入完整性**

\- 預測資料與必要 JSON snapshot 必須在同一個 transaction 內寫入。

\- JSON 寫入前必須確認可序列化，並保留計算當下的指標及驗證結果。

\- 寫入失敗時必須 rollback，不得輸出不存在的 prediction ID。

\- SQLite 必須啟用 foreign key；單機併發寫入建議啟用 WAL 與合理的 busy timeout。

**### 4. 補寫實際結果**

預測有效期限結束後，應以每日排程或等效機制定期補寫 \`prediction\_outcomes\`：

\- 實際最高價、最低價及收盤價。

\- 是否先觸及停損。

\- 是否觸及第一／第二停利。

\- 實際報酬率。

\- 交易結果。

\- 預測是否成功。

同一交易日內若停損與停利均可能被觸及，而日線資料無法判斷先後順序，必須採保守規則或標示「順序無法判定」，不得選擇對績效較有利的結果。

**### 5. 歷史紀錄不可竄改**

\- 原始預測不得 UPDATE 或 DELETE。

\- 重新預測時必須新增一筆資料。

\- outcome 若因資料修正需要更動，必須保留稽核紀錄；正式實作建議新增 outcome revision 或 audit log，不可靜默覆蓋。

**### 6. 模型修正前的歷史分析**

模型修正前，必須根據 SQLite 歷史預測與結果分析：

\- 整體勝率。

\- 各市場狀態勝率。

\- 各買進區間勝率。

\- 停損及停利觸發比例。

\- 平均報酬率。

\- 平均盈虧比。

\- Profit Factor。

\- 最大回撤。

\- 預測誤差。

\- 最大連續虧損次數。

\- 各模型版本與策略版本的表現差異。

**### 7. 避免 Look-ahead Bias**

\- 所有特徵只能使用預測時間點當下已公布且可取得的資料。

\- 技術指標計算不得包含未完成 K 棒或未來修正值，除非輸出明確標示為盤中模型並有獨立驗證。

\- 標準化、缺值填補、特徵選擇與參數調整，只能在各 fold 的訓練資料中 fit，再套用至測試資料。

\- 不得使用完整資料集先計算門檻、分位數或正規化參數後再切分回測。

**### 8. 模型版本治理**

\- 版本命名採日期版本號 \`YYYYMMDD.N\`，例如 \`20260818.1\`。

\- \`N\` 表示當日第 N 次正式變更。

\- 模型版本與策略版本分開管理，分別寫入 \`model\_version\` 與 \`strategy\_version\`。

\- 多版本並行驗證時，每筆輸出必須明確標示使用版本。

\- 未通過驗證的版本可以保留供研究，但不得設為正式建議版本。

**### 9. 概念漂移監控**

每季至少檢視一次近 3 個月表現。若近 3 個月勝率比歷史整體勝率低 10 個百分點以上，或 Profit Factor、最大回撤明顯惡化，應提示：

\> 市場結構可能已改變，建議重新檢視模型。

提示概念漂移不代表可直接修改模型；仍須依第二節取得使用者確認。

**### 10. 資料庫維運**

\- **\*\*併發寫入\*\***：單機使用 SQLite WAL，降低 outcome 排程與即時預測的鎖定衝突。

\- **\*\*備份\*\***：每日備份 SQLite，檔名包含台北日期時間；定期測試還原。

\- **\*\*完整性檢查\*\***：定期執行 \`PRAGMA integrity\_check;\`。

\- **\*\*擴充路徑\*\***：多使用者、多服務或多主機寫入時，遷移至 PostgreSQL，並保留完整歷史、版本與 schema mapping。

\---

**## 四、標準輸出格式**

所有買賣分析開頭或結尾必須包含：

\> 本分析僅供研究與參考，不構成投資建議；00631L 為槓桿型 ETF，使用者應自行承擔交易風險與損益。

建議依下列順序輸出，**先給價格，再補驗證資訊**：

1\. 免責聲明。

2\. 資料時間、來源、模型版本、策略版本。

3\. **預測價。**

4\. **買進價／買進區間。**

5\. **停損價。**

6\. **停利價（第一停利；如有第二停利一併列出）。**

7\. 最終結論：買進／持有／賣出／不交易／資料不足。

8\. 成立條件、失效條件及有效期限。

9\. 模型驗證狀態、模型預測機率與歷史回測勝率；若未驗證或未通過，照實標示，但不得因此省略價格。

10\. 主要風險、預測信心與 SQLite prediction ID 或寫入失敗說明。

\---

**## 五、CHANGELOG.md 維護規範**

**### 1. 強制更新情況**

下列任一項發生變更時，必須同步更新 \`CHANGELOG.md\`：

\- 模型演算法、特徵或模型超參數。

\- KD、RSI、MACD、ATR、均線等指標參數。

\- 交易訊號、進出場、加碼、攤平、停損或停利規則。

\- 回測窗口、交易成本、滑價或槓桿摩擦假設。

\- 模型驗證門檻或市場狀態分類。

\- 資料來源、清洗、缺值、時區或除權息處理。

\- SQLite schema、trigger、排程或 outcome 判定方式。

\- 輸出格式、免責聲明或風險揭露。

\- 任何會影響預測結果、可重現性或歷史比較的程式碼。

純排版、註解或不影響行為的文字修正，也應記錄於 \`Changed\`，但不一定需要提升模型或策略版本。

**### 2. 紀錄格式**

\`CHANGELOG.md\` 採 Keep a Changelog 類型結構，至少包含：

\- 日期與文件／模型／策略版本。

\- \`Added\`、\`Changed\`、\`Fixed\`、\`Deprecated\`、\`Removed\`、\`Security\` 中適用的分類。

\- 修改原因。

\- 對訊號、勝率、報酬、最大回撤及過度擬合風險的可能影響。

\- 驗證狀態與回測比較結果；尚未驗證時明確標示「待驗證」。

\- 使用者確認紀錄，例如確認日期或關聯工作項目。

**### 3. 不得事後補造結果**

變更完成但尚未回測時，必須寫「待驗證」，不得預填改善後的勝率或績效。回測完成後再新增驗證結果，不得修改成看似當時已知。

\---

**## 六、執行優先順序**

如規則發生衝突，依下列順序處理：

1\. 不得虛構資料、價格、勝率或回測結果。

2\. **只要必要行情與價格計算成功，即使模型未通過驗證或尚未完成驗證，也必須直接輸出預測價、買進價、停損價與停利價。**

3\. 行情、必要指標或價格計算無法確認時，才輸出「資料不足」，不得硬填價格。

4\. 修改前必須取得使用者明確確認。

5\. 原始預測紀錄不得竄改。

6\. 所有正式變更必須建立新版本並更新 `CHANGELOG.md`。

7\. 所有分析輸出必須附上免責聲明。

---

## 七、Git Push 與重新執行規範

### 1. Push 後是否需要重新 Run

只要本次修改包含 `AGENTS.md`、模型規則、策略規則、程式碼、設定檔或任何會影響執行行為的內容，在完成 `git push` 後，**必須重新啟動新的 Run，重新載入最新版本的程式碼與 `AGENTS.md` 規則**。

不得假設已經在執行中的舊 Run 會自動重新讀取 push 後的 `AGENTS.md` 或程式碼變更。

標準流程：

```text
修改檔案
→ 檢查變更
→ git commit
→ git push
→ 重新 Run
→ 確認新 Run 已讀取最新 AGENTS.md 與最新程式碼
→ 再執行預測／回測／分析
```

### 2. 正式服務是否需要 Restart

`git push` 本身只會更新遠端 Git Repository，**不代表正在執行中的 FastAPI、Node.js、Worker、Docker Container 或其他服務已自動套用新版本**。

- 若環境有 CI/CD，且 push 後會自動完成 deploy、restart／rolling restart，則依 CI/CD 流程執行，不需要額外手動 restart。
- 若沒有自動部署機制，則伺服器端完成 `git pull` 或更新程式後，必須重新啟動相關服務。
- 若修改的是 Docker Image 內的程式碼，必須重新 build／deploy Container，除非目前開發環境使用 bind mount 並已確認支援熱更新。
- 若只修改 `AGENTS.md`，不一定需要重啟 FastAPI／Node.js 正式服務；但**AI／Codex 的新工作必須重新 Run，確保讀取最新規則**。

### 3. 執行前版本確認

重新 Run 後，執行預測、回測或模型修改前，應確認目前使用的是最新 commit。

至少確認：

```text
git status
git log -1 --oneline
```

如本地端或執行環境版本不是最新 push 的 commit，不得宣稱已使用最新模型、策略或 `AGENTS.md` 規則。
