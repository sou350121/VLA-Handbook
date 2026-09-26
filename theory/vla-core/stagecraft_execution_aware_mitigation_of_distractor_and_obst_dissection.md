# StageCraft：執行期干擾物與阻塞失效的緩解 (StageCraft: Execution Aware Mitigation of Distractor and Obstruction Failures in VLA Models)

> ⚙️ 本文由 Moltbot 自動生成 | 2026-09-26
>
> **論文**: StageCraft: Execution Aware Mitigation of Distractor and Obstruction Failures in VLA Models（IROS 2026）
> **連結**: https://arxiv.org/abs/2603.20659
> **核心定位**: 不改 policy、不微調、不碰權重——用 VLM 在執行「前」改環境初始狀態，把干擾物搬走，換取 VLA 成功率的絕對 +40% 提升

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | 訓練無關（training-free）地推斷該移除哪些干擾物，只搬「必要的」那些，真實機器人三個任務平均絕對 +40% 成功率 |
| 適合精讀 | 若你在做機器人部署、VLA 魯棒性、或用 VLM 做 high-level planner，重點看 §2（object-set 策略）與 §3（數量估算） |
| 可以跳過 | 若你只關心 policy 本身的架構創新，這篇不碰 policy，距離中等 |
| 落地可行性 | 中（plug-and-play，但每次干預加約 200 秒執行時間，且需 SAM3 + 動作 primitive + 校準相機）|
| 主要風險 | 能力上限卡在 VLM 本身；干擾物無法離散化物件化時不適用 |

💡 **X-Ray 開場**
這篇問的是：VLA 在真實工作區遇到訓練時沒見過的干擾物（distractor）與阻塞（obstruction）就崩，能不能不重訓 policy 就救回來？答案是能——用 VLM 看幾段 rollout 影片，推論出場景裡哪些物件「可能害 policy 失敗」，然後請機器人先把這些物件撿走，再執行任務。對 VLA 研究者意味著：魯棒性不一定只能從 policy 內部解，環境本身也是一個可操作的變數。

📍 **研究全景時間線**

```
[2015-2023] 規劃/前置條件研究 (symbolic predicates)
        →  [2024] 以 policy rollouts 做 policy improvement (RL / value guidance)
        →  [2025] ReSET：學人類重排初始狀態（需 policy 訓練資料）
        →  [本文 2026] StageCraft：training-free、VLM in-context、只搬必要干擾物  ← 當前位置
                      限制：能力上界 = VLM；需可離散化物件集
```

## 1. 核心架構/方法總覽 (Overview / Architecture)

### 1.1 系統對比概覽

| 模組 | 輸入 | 輸出 | 頻率/時序 | 訓練/推理 |
|------|------|------|-----------|-----------|
| VLA policy π | 觀測 o（腕部 + 前視相機）| 絕對關節空間動作 | 執行期逐動作 | 已微調好的黑盒，StageCraft 不碰 |
| Rollout buffer B | N 個物件集 × M 次 rollout + 成功標籤 y | 經驗成功率 sr | 部署前一次性收集 | 無訓練 |
| VLM 推理 | 少量 in-context rollout（影像 + y）+ 新初始狀態 | 要移除的物件清單 | 每次任務執行前一次 | in-context，無微調 |
| 環境修改 | 物件語意描述 | pick-and-place 動作 | 執行任務前 | SAM3 偵測 + IK primitive |

### 1.2 關鍵機制 (Key Mechanism)

- **只改環境，不改 policy**：StageCraft 是 plug-and-play decorator，對底層 policy 零假設、零約束，也不需要 policy 的訓練資料或原 MDP。
- **保守干預**：policy 若本來就扛得住某干擾物，StageCraft 就不動它——干預量隨 policy 強度自適應上調或下調。
- **物件集（object-set）而非單物件**：VLM 以「集合」為單位估計成功率，避免開放式推理在不同 rollout 間辨識不一致。

⚡ **Eureka Moment**：不要問「policy 為什麼失敗」，而要問「哪一組物件存在時 policy 會成功」——把「該移除什麼」的開放式問題，轉成「挑一個觀測成功率最高的物件子集」的離散選擇問題。

### 1.3 資訊流/架構圖 (Flow / Diagram)

```
                ┌──────────────────────────────┐
rollouts ─────▶ │ VLM (in-context, 無微調)      │
(video + y)      │ 1. object-set creation        │
                 │ 2. 估 sr(set) → 留 sr_max 集   │
新初始狀態 s0 ─▶ │ 3. set-transition：挑最大子集  │
                 └───────────────┬──────────────┘
                                 │ 要移除的物件描述
                                 ▼
                     SAM3 偵測 → 3D 定位 → IK pick-and-place
                                 │
                                 ▼
                   修改後初始狀態 s0' ──▶ VLA policy π 執行任務
```

## 2. 數學核心 (Math Core)

📌 **Napkin Formula**（一行抓住本質）：

```
X_manip = X_dist \ argmax_{ s ⊆ X_dist, sr(s) ≥ sr_max } |s|
（在「觀測成功率最高的物件集」裡挑最大的可行子集保留；其餘干擾物才需移除）
```

目標：在執行前，從場景物件中挑出最少的干擾物移除，使 policy 成功機率最大。

形式化（論文 §III-A）：

```
MDP:           M = (S, O, A, p, H, ρ0)
初始狀態分解:   s0 = (u0, X0),  X0 = X_rev ∪ X_dist
全集:          X = {x1, ..., xN},  X_dist ⊆ X,  |X_dist| ≤ N
目標:          推斷 X_manip ⊆ X_dist，並最小化 |X_manip|
修改後:         s0' = (u0, X0'),  X0' = {X_rev ∪ X_dist} \ X_manip
動作 primitive: P = {φ1, ..., φK}
```

Rollout 估計（論文 §III-B）：

```
B = { (u0^(i,j), X0^(i), y^(i,j)) },  i = 1..N, j = 1..M, y ∈ {0,1}
```

物件集過濾（論文 §III-C）：

```
S = { s_j ⊆ X | sr(s_j) ≥ sr_max },   sr_max = max_{s ⊆ X} sr(s)
```

變數說明：

| 符號 | 含義 |
|------|------|
| u0 | 非物件因素（機器人配置、環境參數）|
| X_rev | 每個 episode 都在的任務相關物件 |
| X_dist | 干擾物集合 |
| X_manip | 需移除的干擾物子集 |
| y | rollout 成功(1) / 失敗(0) |
| N / M | 不同物件集數 / 每集重複 rollout 次數 |
| sr | 經驗成功率 |
| sr_max | 所有 in-context 集中最高觀測成功率 |

直覺：M 太小 → 成功率估計噪聲大；N 太小 → StageCraft 認不出足夠物件，退化成「全部搬走」。若 policy 強，S 裡會出現含多干擾物的大子集，StageCraft 就少干預；若 policy 弱，S 只剩小子集，就得多搬。

## 3. 帶數字走一遍：玩具例子 (Worked Example)

用論文 RLBench 模擬設定（§IV-C）：任務 "pick the red cup"，環境 Zero / One / Two / Three 分別代表 0 / 1 / 2 / 3 個干擾物。

兩條 policy：
- π_weak：50 個 demo，Zero 環境成功率 78%
- π_strong：250 個 demo，Zero 環境成功率 95%

在 Env Three（3 個干擾物）評估：

```
步驟 1：in-context buffer 填入 Zero/One/Two 的 rollout。
步驟 2：VLM 估每集 sr → 只保留 sr_max 的物件集進 S。
步驟 3：新場景 Three 到來 → 挑「S 中最大的、且其物件都出現在 Three 裡」的子集保留。
        其餘物件 = X_manip → 搬走 → 執行 policy。
```

結果（論文 Fig. 8）：

| policy | StageCraft 平均干預步數（100 episodes）| 成功率變化 |
|--------|----------------------------------------|-----------|
| π_weak | 3.09 步 | 0% → 66%（+66）|
| π_strong | 1.14 步 | 85% → 98%（+13）|

可計算的閉環：弱的 policy 被要求搬更多（3.09 vs 1.14），換來更大增益（+66 vs +13）。這正是「干預量隨 policy 強度自適應」的數字證據。

再看 in-context 樣本數的影響（論文 Fig. 9）：

```
只有 1 個成功的 Env Two rollout → VLM 誤判 policy 對該干擾物魯棒 → 不干預
但 policy 在 Env Two 實際只有 21% 成功率 → 干預缺失導致失敗
→ 樣本增到 20 個：平均干預 1.15 → 2.2 步，成功率 49% → 54%（§IV-C）
```

## 4. 工程視角 (Engineering View)

- **延遲代價**：VLM 推理 + 環境操縱約增加 **~200 秒** 執行時間（論文 Fig. 3 註）。對「任務本身秒級完成」的操作，這是一筆巨大的固定開銷。
- **部署鏈**：VLM → SAM3 偵測 → 框中心投影回 3D（需相機內外參校準）→ IK primitive pick-and-place → 丟進機器人旁的回收箱。任一環失效都會拖垮整體。
- **記憶體 / 上下文**：每個 episode 含影像，token 極重。要更好 Monte Carlo 估計就得更長 context——直接撞 VLM 上下文上限（論文 Limitations）。
- **控制頻率邊界**：StageCraft 是「執行前一次性」干預，不進控制迴路，因此不改變策略控制頻率，但也無法處理執行中才出現的動態干擾。
- **保守性保證**：論文宣稱在兩個假設下（任務相關物件曾出現在成功 rollout 中；VLM 遵守 set-transition prompt），StageCraft 永不誤移任務相關物件。

## 5. 數據與評測 (Data & Eval)

**真實機器人（Franka FR3）**：
- 任務：stack_cups、setup_plate、block_in_bowl（共 3 個）
- 微調資料：10 個任務，每任務 60 個 demo，Mechanical Turk 蒐集語言指令
- 相機：Intel D435 腕部 + 前視共兩路；動作 / 本體感覺用絕對關節空間
- 干擾物：固定 88 個物件（Fig. 4），訓練時從未出現在工作區
- 基座模型：Pi0.5、SmolVLA（LeRobot 預訓練權重全量微調）
- StageCraft 設定：10 個 in-context rollout，15 個評估 rollout
- VLM：gemini-3.1-pro（平均 95% prompt-following 準確率）；消融另用 gemini-2.5-pro、gpt-5.2-pro
- 結果：三任務平均絕對 +40% 成功率，接近 Base（無干擾）水準

**模擬（RLBench）**：見 §3；用 gemini-2.5-pro（延遲較低）。

**基線對照**：naive-prompt（拿掉 object-set 策略）平均搬 1.88 個物件 vs StageCraft 1.14；naive 還在 12/25 個案例誤移了紅杯本身、4 個案例全搬；移除物件數的變異係數 57.8% vs 13.62%（論文 Fig. 9a）。

## 6. 能力與失敗模式 (Capabilities & Failure Modes)

**能做的**：
- 訓練無關地緩解「視覺分佈外」干擾物 / 阻塞造成的執行期失敗
- 依 policy 強度自適應干預量（干預越少代表 policy 越強）
- 可與 RL、adapter、value guidance 等 policy improvement 方法疊加

**做不了 / 高風險場景**：
- 執行「中」才出現的動態干擾（StageCraft 僅執行前干預）
- 干擾物無法離散化成物件集（如流體、光照、紋理干擾）
- 無法操縱的干擾物（搬不動的東西）
- 干擾物數量極多時，VLM 上下文吃不下
- 每次干預新增 ~200 秒執行延遲

### 6.1 隱含假設 (Hidden Assumptions)

- 假設存在可用的低階動作 primitive（pick-and-place、go-to-point 等）與校準相機——真實部署未必現成。
- 假設「任務相關物件曾出現在某些成功 rollout」——若 policy 在某環境從未成功（如 π_weak 在 Env Three 為 0%），此假設在評估環境未必成立。
- 假設 VLM 的 set-transition 推理足夠可靠——但論文自身消融顯示舊世代 VLM 做不到。
- 「移除干擾物即改善」隱含假設：失敗主因是外來物件，而非 policy 本身的動作錯誤。

## 7. 與相關工作對比 (Comparison)

| 方法 | 關注點 | 架構 | 訓練方式 | 適用場景 |
|------|--------|------|----------|----------|
| StageCraft（本文）| 執行期干擾物 | Training-free decorator | 無訓練，VLM in-context | 有干擾物、有動作 primitive 的部署 |
| RL fine-tune / DPPO | Policy 本身 | 改 policy 參數 | RL | 需大量樣本 |
| Value guidance | 引導動作 | 另訓 value function | 訓練 | 需 reward |
| CoT-VLA 等 | 內建推理 | 改架構 + 重註資料 | 預訓練期 | 特定 VLA 模型 |
| ReSET | 初始狀態重排 | 學人類重排 | 需 policy 訓練資料 + 人類資料 | 已知場景 |

**面試 Tip**：被問「VLA 魯棒性只能重訓嗎？」——答：不一定。StageCraft 證明可把問題外移到「環境初始狀態」這一可操作變數，用 VLM in-context 推斷最小干預集，代價是執行前一次性延遲（~200 秒），且不假設 policy 訓練資料。

## 8. 精讀建議 (Reading Guide)

- **值得精讀原文的人**：
  1. 做機器人部署、想在不重訓前提下提升魯棒性的工程師；
  2. 研究 VLM-as-planner / 環境前置條件的研究者；
  3. 評估「VLA 失敗模式分類與緩解」的從業者。
- **建議章節路徑**：先讀 §III-B / §III-C（object-set 策略，全文精華）→ 再看 §IV-B 真實實驗數字 → 可跳 §II 相關工作（除非你要定位與 ReSET 的差異）。
- **不值得精讀的理由**：若你不做機器人學習、或已熟悉 VLM in-context planning，讀摘要 + Fig. 8 / Fig. 9 的數字即可。

---
[← Back to Theory](./README.md)

**關鍵引用**：
- [arXiv:2603.20659](https://arxiv.org/abs/2603.20659)
- [項目頁（含 prompt 全文）](https://stagecraft-decorator.github.io/stagecraft/)
