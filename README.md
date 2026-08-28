# 股票機率預測與回測工具

這個專案使用 Yahoo Finance 歷史資料，對股票或 ETF 進行時序模型訓練、
walk-forward 樣本外驗證，以及包含交易成本的交易模擬。

預設使用 ExtraTrees 完整技術指標版，包含價格趨勢、波動度、RSI、價量、KD、MACD、
20／60日支撐壓力，以及可取得時的台積電與台灣大盤同步資訊。原本的 Logistic 模型
及 `--feature-set baseline` 基礎特徵仍保留作研究比較。
所有技術指標只使用訊號當日及更早的資料。

> 本工具僅供研究，不構成投資建議。模型出現訊號不代表未來一定獲利。

## 安裝

建議使用獨立虛擬環境：

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install yfinance pandas numpy scikit-learn
```

## GitHub Actions 自動執行

若要在每次排程完成後收到 Discord 通知，請到 repository 的
`Settings → Secrets and variables → Actions → New repository secret`，建立：

- Name：`DISCORD_WEBHOOK_URL`
- Secret：Discord Webhook 的完整網址

Webhook 不可直接寫入 workflow 或其他公開檔案。通知會顯示分析與部署狀態、行情日期、
目前價位、正式結論、驗證狀態，以及儀表板與 Actions 執行連結。

`.github/workflows/predict.yml` 會在每週一至週五台北時間 16:30 自動執行，
也可在 GitHub repository 的 **Actions → 00631L prediction → Run workflow** 手動啟動。
流程會先安裝 `requirements.txt`、執行測試，再以本文件的預設參數完成 00631L
Walk-Forward 與 OOS 分析。

每次執行會上傳 `validation.json`、`predictions.sqlite3` 與 `predict.log`，保存 30 天。
SQLite 也會透過 GitHub Actions cache 帶到下一次成功執行；cache 與 artifact 並非永久備份，
正式使用仍應另行安排每日備份與還原測試。排程採 UTC cron，未排除台灣休市日；休市日
若 Yahoo Finance 沒有新行情，輸出仍須依程式既有的資料完整性與模型驗證規則判斷。

成功執行後，最新 JSON 也會發布到固定 GET 網址：

```text
https://s1710931008.github.io/yfinance/latest.json
```

例如：

```bash
curl --fail https://s1710931008.github.io/yfinance/latest.json
```

首次部署前，repository 必須在 **Settings → Pages → Build and deployment → Source**
選擇 **GitHub Actions**。此網址公開最新 JSON，但不公開 SQLite；GitHub Pages 是靜態發布，
不提供身分驗證或即時計算，內容只會在 workflow 成功後更新。

GitHub Pages 首頁會直接將 `latest.json` 顯示為中文儀表板，包含最新行情日期、模型機率、
OOS／Final Test 回測、驗證門檻、分層結果、風險及 SQLite prediction ID。首頁也會讀取
公開的 GitHub Actions API 顯示最近一次 workflow 的等待、執行、成功或失敗狀態。
若必要驗證資料缺少或未通過，頁面正式結論固定顯示「不交易／資料不足」，且不顯示
可執行的買進、停損或停利價格。

正式驗證通過且最新訊號成立時，首頁才會顯示條件買進價、買進區間、分批比例、停損、
第一／第二停利、風險報酬比、成立與失效條件及有效期限。最大回撤百分比以「每筆交易
將配置給 00631L 的資金全額投入，扣除成本後逐筆複利」計算；開發期 Walk-Forward OOS
勝率會與完全隔離的 Final Test 勝率比較。任一 fold 沒有有效交易時，fold 穩定性視為
資料不足／未通過，不得略過後再計算標準差。

確認依賴已安裝：

```bash
.venv/bin/python -c "import yfinance, pandas, numpy, sklearn; print('dependencies OK')"
```

## 預測 00631L

```bash
.venv/bin/python scripts/predict.py 00631L.TW \
  --period 10y \
  --horizon 5 \
  --target 0.04 \
  --adverse -0.025 \
  --folds 5 \
  --output-json validation.json
```

最後一行後面不要加反斜線 `\`，否則終端會等待繼續輸入。

## 查詢其他股票

只要更換指令中的股票代號。

### 台灣上市股票與 ETF

Yahoo Finance 的上市股票代號通常以 `.TW` 結尾：

```bash
# 台積電
.venv/bin/python scripts/predict.py 2330.TW --period 10y --folds 5

# 鴻海
.venv/bin/python scripts/predict.py 2317.TW --period 10y --folds 5

# 聯發科
.venv/bin/python scripts/predict.py 2454.TW --period 10y --folds 5

# 元大台灣50
.venv/bin/python scripts/predict.py 0050.TW --period 10y --folds 5
```

### 台灣上櫃股票

上櫃股票代號通常以 `.TWO` 結尾：

```bash
.venv/bin/python scripts/predict.py 6488.TWO --period 10y --folds 5
```

### 美國股票

美股直接使用英文代號：

```bash
.venv/bin/python scripts/predict.py AAPL --period 10y --folds 5
.venv/bin/python scripts/predict.py NVDA --period 10y --folds 5
```

## 如何解讀買賣點

預設判斷規則如下：

- ExtraTrees 預設機率門檻為 22%；Logistic 預設為 70%。
- 預測機率達到門檻，而且歷史交易驗證閘門通過，才顯示「符合買進條件」。
- 程式會顯示目前機率高於或低於門檻多少個百分點；「符合買進條件」代表可以考慮
  建立新部位，不代表必須買進，也不代表既有持股必須加碼。
- 參考進場點為下一個交易日開盤價附近。
- 停損為進場價下方 `1.5 ATR`。
- 停利為風險距離的 `2R`。
- 最長持有 5 個交易日；先碰到停損或停利便提前出場。
- 預測機率未達門檻時顯示「觀望」，不建立新的部位。

模型訊號通過後仍要等待下一交易日開盤確認。正式候選策略只接受開盤價位於「最新收盤
減 0.25 ATR」至「最新收盤加 0.25 ATR」之間。這是先要求至少 30 筆 OOS 交易、
正 EV 與 PF 至少 1.2，再比較勝率及保留期穩健性後選出的歷史成交條件。程式會在
買進區間旁列出 OOS 與保留期的交易筆數、勝率、平均 R 及 PF。低於區間代表動能
確認不足，高於區間代表跳空過大；兩者都不進場。最新收盤只是計算基準，不是模型
預測的未來成交價，歷史勝率也不保證未來結果。

程式還會用 3 種歷史保留比例與 3 種 walk-forward 折數，共 9 組時間切割重新驗證
同一成交區間。只有中位交易至少 60 筆、中位勝率至少 60%、中位 EV 至少 0.15R、
中位 PF 至少 1.40、至少 70% 測試為正報酬、中位回撤不超過 8R，而且雙倍成本後
EV 仍為正、PF 至少 1.10，才會顯示建議買進區間。任何一項未通過都只顯示觀望，
不提供建議價。這項嚴格驗證會增加執行時間。

`20260828.2` 對 00631L 採用預先固定的風險配置：一般市場最多投入「配置給 00631L
的資金」之 85%，多頭高波動最多投入 50%，空頭高波動不建立新部位。最大回撤正式
門檻為 24.9%。機率門檻維持 15%，不使用 OOS 結果事後挑選門檻。

### 研究價格預測與交易價位

`20260828.3` 將 5 日研究價格預測與交易建議價完全分開。只有價格模型本身同時通過
開發期 Walk-Forward OOS 與時間隔離保留期驗證時，網頁才顯示中央預測價格及 80%
預測區間；同時列出 MAE、方向準確率與區間覆蓋率。研究價格預測不是建議買進價，
也不會讓未通過的交易策略產生買進、停損或停利價。

價格模型任一檢查未通過時，JSON 與 Page 仍保留可確認的誤差統計，但中央價格與區間
維持 `null`／「未產生」，不得為了完成畫面而填入未驗證的數字。

這裡的賣點是買進後的停損、停利或到期出場，不是放空訊號，也不能單獨用來
判斷既有持股是否應立即賣出。

當訊號與交易驗證閘門同時通過時，輸出會直接列出歷史回測成交區間、區間勝率、
停損賣出價、停利賣出價與時間賣出日。實際成交價要等隔日開盤才知道；若實際成交，
停損與停利價格也必須依成交價重新計算。

## 一般分析
```bash
.venv/bin/python scripts/predict.py 00631L.TW \
  --period 10y \
  --horizon 5 \
  --target 0.04 \
  --adverse -0.025 \
  --folds 5 \
  --output-json validation.json
```


## 既有持股分析

已有持股時，可同時提供股數與含既有買進費用的平均成本：

```bash
.venv/bin/python scripts/predict.py 00631L.TW \
  --period 10y \
  --horizon 5 \
  --target 0.04 \
  --adverse -0.025 \
  --folds 5 \
  --shares 2000 \
  --average-cost 38.0745 \
  --add-shares 100 \
  --output-json validation.json
```

`--shares` 與 `--average-cost` 必須一起提供。持股分析會顯示：

- 扣除預估賣出手續費、交易稅與滑價後的目前損益及報酬率。
- 含預估賣出成本的回本價。
- 續抱、減碼或停損建議及其原因。
- 以平均成本與目前 ATR 計算的持股停損及停利價。
- 在停損或停利價賣出時的預估總損益。
- 股數上限內的加碼價格、加碼後平均成本、回本價、停損／停利及對應總損益。
- 該加碼比例在 OOS 與保留期的交易數、實際加碼次數、勝率、EV_R、PF 及最大回撤。

持股建議規則：價格已低於持股停損價時顯示「停損」；否則最新機率與交易驗證
同時通過時顯示「續抱」；其餘情況顯示「減碼」。實際券商費率、最低手續費與
成交滑價可能不同，因此輸出金額只是估算。

歷史模擬選出的規則為：價格較平均成本下跌 `0.75 ATR` 後，最多加原持股的50%，
只允許加碼一次，並維持原持股停損與停利。`--add-shares` 是使用者允許的股數上限，
預設100股；實際試算股數取「原持股50%」與這個上限的較小值。

時間與成交規則為：本次模型訊號後、`--horizon` 指定的交易日數內，盤中第一次
觸及加碼價時成交。觸及第一防守價、模型閘門失效或超過期限即取消。程式顯示的
加碼價是限價上限；若下一交易日價格已低於第一防守價，不應執行加碼。

開發期及保留期的歷史模擬都顯示這項規則優於不加碼，但不保證未來仍然有效。
程式目前沒有你的總資產與最大虧損預算，因此「可考慮加碼」只代表模型與歷史條件
通過，不代表該股數符合個人風險承受能力。

加碼回測的勝率是依實際完成交易計算，不是模型的分類命中率。回測採每日 OHLC
資料；若同一天同時觸發加碼及停損，會保守假設先加碼、再停損。

### 分批操作方法

提供持股資料後，程式也會依最新價格、ATR、平均成本、含成本回本價與量能產生：

- 第一防守：收盤跌破後，下一交易日減碼50%。
- 最終停損：收盤跌破區間下緣，退出剩餘部位。
- 反彈目標一：分批減碼30%～50%。
- 反彈目標二：接近含賣出成本的回本價時再減碼。
- 強勢停利：突破反彈目標二，且成交量至少為20日均量1.5倍時才成立。

這些價格會隨最新收盤、ATR及平均成本變動，全部是動態參考值。現有歷史回測採
單一停損／停利／時間出場規則，尚未回測這套分批賣出比例，因此分批操作表不應
與上方的歷史勝率混為一談。

## 主要參數

| 參數 | 意義 | 預設值 |
|---|---|---:|
| `--period` | 下載多少歷史資料 | `10y` |
| `--horizon` | 預測及最長持有交易日數 | `5` |
| `--target` | 分類標籤要求的最低漲幅 | `0.04` |
| `--adverse` | 預測期間容許的最大不利跌幅 | `-0.025` |
| `--threshold` | 產生買進訊號的最低機率 | ExtraTrees `0.22`；Logistic `0.70` |
| `--folds` | walk-forward 驗證折數 | `5` |
| `--final-test` | 完全保留的最終測試比例 | `0.20` |
| `--stop-atr` | 停損 ATR 倍數 | `1.5` |
| `--reward-risk` | 停利的 R 倍數 | `2.0` |
| `--entry-gap-low-atr` | 下一日開盤相對訊號收盤的最低 ATR 位移；負值代表低開 | CLI `0.15`；正式候選策略 `-0.25` |
| `--entry-gap-high-atr` | 下一日開盤相對訊號收盤的最高 ATR 位移 | CLI `0.55`；正式候選策略 `0.25` |
| `--feature-set` | `baseline` 基礎版或 `all` 全指標版 | `all` |
| `--model` | `extra-trees` 實驗升級版或 `logistic` 基準版 | `extra-trees` |
| `--database` | 追加保存預測紀錄的 SQLite 路徑 | `predictions.sqlite3` |
| `--no-record` | 研究比較時不寫入 SQLite | 不啟用 |
| `--shares` | 現有持股股數，需搭配平均成本 | 不啟用 |
| `--average-cost` | 每股平均成本，需搭配持股股數 | 不啟用 |
| `--add-shares` | 單次加碼／攤平的股數上限 | `100` |
| `--output-json` | 完整結果及交易明細檔案 | 不輸出 |

一般股票的波動通常低於槓桿 ETF，4% 的五日目標可能過高。例如台積電可以先研究：

```bash
.venv/bin/python scripts/predict.py 2330.TW \
  --period 10y \
  --horizon 10 \
  --target 0.03 \
  --adverse -0.02 \
  --threshold 0.65 \
  --folds 5
```

這只是研究起點，不能因為單次結果較好就認定參數有效。

## 輸出指標

- **訊號命中率**：達到機率門檻後，實際符合分類條件的比例。
- **Brier 誤差**：機率校準誤差，越低越好。
- **平均每筆 R（EV_R）**：扣除成本後，每筆交易平均賺賠的風險倍數。
- **獲利因子（PF）**：總獲利除以總虧損；小於 1 表示整體虧損。
- **最大回撤**：樣本外資金曲線從高點回落的最大 R 倍數。
- **年度／市場環境穩定性**：檢查績效是否只集中在少數年份或行情。
- **綜合驗證分數**：綜合命中率、校準、EV_R、PF 與穩定性的研究評分。
- **策略健康度**：依最近交易的平均 R、PF、連敗與漂移判斷。
- **交易勝率**：獲利交易筆數除以全部有效交易筆數；與模型預測機率分開呈現。
- **平均盈虧比**：平均獲利 R 除以平均虧損 R 的絕對值。

## 技術指標特徵

- **價量關係**：成交量異常值、短長期量比、價量相關、價量交互作用、OBV 動能及 CMF。
- **KD**：9 日 RSV 與平滑後的 K、D，以及 K-D 差距。
- **MACD**：12/26 日 MACD、9 日訊號線、柱狀體及柱狀體變化。
- **RSI**：14 日 RSI。
- **支撐壓力**：收盤距離20日及60日最低／最高價的位置。

加入指標不代表勝率必然提高。是否有效應比較樣本外的訊號命中率、EV_R、獲利因子、
最大回撤與 final test，不能只比較訓練資料的準確率。

依 `AGENTS.md`，預設使用完整技術指標版，但完整不等於一定通過驗證。若9組時間
切割或成本壓力測試失敗，程式仍輸出「不交易」且不提供建議價格。若要比較基礎版：

```bash
.venv/bin/python scripts/predict.py 00631L.TW --period 10y --folds 5 --feature-set baseline --no-record
```

## SQLite 預測紀錄

每次正式執行會把當次預測新增至 `predictions.sqlite3`，包含行情、動作、建議區間、
停損、兩段停利、模型機率、回測勝率、有效期限、版本、技術指標、理由及市場狀態。
原始預測欄位不會被覆蓋；同一天重新執行也會新增一筆。程式執行時會為已到期紀錄
補寫實際最高／最低／收盤、停損停利觸發、報酬與結果。純研究比較可加
`--no-record`，避免把實驗結果寫入正式歷史。

### 要累積多久才有資料

預測有效期預設為5個交易日，因此第一筆實際結果通常要約一週後才會完成。若每個
交易日收盤後執行一次，大約可依下列時間累積：

| 時間 | 每日預測紀錄（約） | 用途 |
|---|---:|---|
| 1個月 | 20筆 | 確認自動紀錄及結果補寫正常 |
| 2.5個月 | 50筆 | 初步觀察，不適合直接下定論 |
| 5個月 | 100筆 | 開始評估近期穩定性 |
| 10個月 | 200筆 | 比較不同市場環境較有參考性 |

這些是每日預測紀錄，不是實際買進交易數。嚴格驗證未通過時會記錄「不交易」，
因此累積50筆真正買進交易可能需要更久，也可能暫時完全沒有買進訊號。

### 查看最新完整結果

```bash
cd /Users/hongzhenghong/Desktop/yfinance
.venv/bin/python -m json.tool validation.json | less
```

按 `q` 離開。Codex自動排程完成後，也會在原本的Codex任務中回報資料日期、目前
動作、模型機率、多重驗證勝率、是否提供買進價及SQLite寫入結果。

### 查詢SQLite最近20筆預測

```bash
sqlite3 -header -column predictions.sqlite3 "
SELECT
  id,
  market_date AS 日期,
  action AS 建議,
  ROUND(model_probability * 100, 1) || '%' AS 模型機率,
  ROUND(backtest_win_rate * 100, 1) || '%' AS 回測勝率,
  trade_result AS 實際結果
FROM predictions
ORDER BY id DESC
LIMIT 20;
"
```

### 查詢已到期的預測結果

```bash
sqlite3 -header -column predictions.sqlite3 "
SELECT
  market_date AS 預測日期,
  action AS 建議,
  actual_high AS 最高價,
  actual_low AS 最低價,
  actual_close AS 到期收盤,
  ROUND(actual_return * 100, 2) || '%' AS 實際報酬,
  trade_result AS 結果,
  prediction_success AS 是否成功
FROM predictions
WHERE settled_at IS NOT NULL
ORDER BY market_date DESC;
"
```

若沒有輸出，通常表示尚未經過5個交易日，或資料庫目前還沒有已結算紀錄。

若要切回原本 Logistic 基準模型：

```bash
.venv/bin/python scripts/predict.py 00631L.TW --period 10y --folds 5 --model logistic
```

如果手動指定 `--threshold`，程式會採用指定值，不再使用模型的自動預設門檻。

交易模擬預設包含雙邊手續費、賣出交易稅及雙邊滑價。

## 持續更新與優化

每次執行時，程式都會下載最新資料並重新訓練，不需要手動更新資料檔案。

可以持續優化，但必須避免為了讓歷史績效漂亮而過度擬合：

1. 只用開發資料調整參數。
2. 不使用 final test 選擇參數或模型。
3. 新資料增加後，才向前滾動重新評估 final test。
4. 不同股票可研究不同的目標漲幅、持有日數與機率門檻。
5. 樣本數必須足夠，不能因一兩筆獲利便判定策略有效。
6. 優先要求 OOS EV_R 大於 0、PF 穩定大於 1，且年度及市場環境表現一致。

修改參數後若 final test 表現很好，也不能反過來繼續針對同一段 final test 調整；
一旦用它選過參數，它就不再是 untouched final test。

## 執行測試

目前測試不依賴 pytest，可直接執行：

```bash
.venv/bin/python - <<'PY'
import tests.test_predict as tests

for name in sorted(n for n in dir(tests) if n.startswith("test_")):
    getattr(tests, name)()
    print("PASS", name)
PY
```
