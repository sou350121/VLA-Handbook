# Claude Code Memory 系統 → VLA 機器人記憶：跨領域架構遷移分析

> ⚙️ 本文由 **Claude Opus 4.6** 生成 | 2026-04-01 | ✍️ 原創分析（非論文拆解）
> **類型**：跨領域架構遷移分析
> **關鍵詞**：Memory System、VLA Memory、Cross-Session Learning、Auto Dream、MemoryVLA、MEM、MemER
> **參考論文**：MemoryVLA (ICLR 2026)、MEM (Physical Intelligence, 2026.03)、MemER (2025.10)、EchoVLA (2025.11)

---

## 核心命題

Claude Code 的 Memory 系統是目前 LLM agent 記憶設計中最成熟的工程實踐之一。VLA（Vision-Language-Action）模型正面臨幾乎相同的記憶困境——跨 episode 遺忘、長時序依賴缺失、多層級經驗無法累積。本文將 Claude Code Memory 的 7 個核心設計模式系統遷移到 VLA 領域，提出一個「分佈式分層機器人記憶架構」。

---

## 一、問題對齊：兩個領域的同構困境

| 困境 | Claude Code | VLA 機器人 |
|------|------------|-----------|
| **跨 session 遺忘** | 每次新 session context window 清零 | 每次新 episode 模型無法記得上次操作經驗 |
| **有限 context** | Context window 有 token 上限 | 推理時序列長度受 GPU 記憶體限制（如 16 frames ≈ 1 min） |
| **多層級知識** | 組織級→使用者級→專案級→子目錄級 | 通用物理知識→環境知識→任務知識→步驟知識 |
| **人機協同** | 人寫規則 + AI 自學習 | 人類示範 + 機器人自主探索 |
| **安全約束** | 權限系統防止危險操作 | 力矩限制、碰撞檢測、OOD 檢測 |

**核心洞察**：這不是巧合。兩者都是「有限記憶容量的 agent 在持續交互環境中如何累積和利用經驗」的問題。

---

## 二、七個設計模式的遷移映射

### 模式 1：雙軌記憶（Human-Written + AI-Written）

**Claude Code 原型**：
- CLAUDE.md = 人類寫的規則和指令
- Auto Memory = Claude 自己累積的學習

**VLA 遷移 → 「示範記憶 + 自學記憶」**：

```
示範記憶（Demonstration Memory）← 人類寫的
├── 專家示範軌跡的壓縮表徵
├── 人類標註的任務語義（"先拿杯子，再倒水"）
├── 安全約束規則（"力矩不超過 5N"、"不碰紅色物體"）
└── 環境佈局先驗知識

自學記憶（Self-Learned Memory）← 機器人自己累積的
├── 成功/失敗經驗的壓縮摘要
├── 物體抓取姿態的偏好（"這個杯子把手朝左時用側抓"）
├── 環境動態模型修正（"這個抽屜需要更大力才能拉開"）
└── 任務分解策略優化
```

**對應最新論文**：
- MEM（Physical Intelligence, 2026.03）的 text-based long-horizon memory 就是一種自學記憶——機器人把已完成動作壓縮為文字摘要（"I picked up the plates"），本質上就是 Claude Code Auto Memory 的機器人版
- MemoryVLA（ICLR 2026）的 working memory + memory bank 雙系統直接對應 Claude Code 的即時 context + 持久化記憶

### 模式 2：分層優先級（Hierarchy with Override）

**Claude Code 原型**：
```
Managed Policy → User → Project → Subdirectory
（越具體的規則，優先順序越高）
```

**VLA 遷移 → 「物理先驗 → 環境知識 → 任務知識 → 步驟知識」**：

```
Layer 0: 物理先驗（Physics Prior）← 類似 Managed Policy，不可覆寫
  │  重力方向、物體持久性、基本力學約束
  │  → 來源：預訓練 + 硬編碼安全規則
  │  → 永遠生效，不可被下層覆蓋
  │
Layer 1: 環境知識（Environment Memory）← 類似 User Memory
  │  空間地圖、物體位置、可通行區域
  │  → 來源：SLAM + 語義分割的長期累積
  │  → 跨所有任務生效
  │
Layer 2: 任務知識（Task Memory）← 類似 Project Memory
  │  "做三明治"的步驟序列、所需物體、成功判據
  │  → 來源：示範學習 + 自我探索
  │  → 在特定任務內生效
  │
Layer 3: 步驟知識（Step Memory）← 類似 Subdirectory Memory
     當前步驟的精細感知、夾爪力回饋、微調策略
     → 來源：即時觀測 + working memory
     → 僅在當前動作步驟有效
     → 最具體的層級，可覆蓋上層的通用策略
```

**關鍵遷移原則**：越具體的記憶優先順序越高（跟 Claude Code 完全一致），但 Layer 0 的物理先驗和安全約束不可被覆蓋（跟 Managed Policy 一樣）。

**對應最新論文**：
- EchoVLA 的 Scene Memory（空間語義地圖）+ Episodic Memory（任務經驗）就是 Layer 1 和 Layer 2 的實現
- MemER（Physical Intelligence）的 hierarchical policy（高層選 keyframe → 低層執行）體現了 Layer 2→3 的優先級關係

### 模式 3：Lazy Loading / 按需載入

**Claude Code 原型**：
- 子目錄 CLAUDE.md 不在啟動時載入，而是 Claude 讀取該目錄文件時才注入
- Path-scoped rules 只在匹配文件時觸發

**VLA 遷移 → 「情境觸發記憶檢索」**：

這是 VLA 記憶系統中**最關鍵也最被忽略**的設計模式。

```
啟動時載入（高成本，高價值）：
├── 當前環境的空間地圖
├── 當前任務的高層計畫
└── 安全約束

按需檢索（低成本，精準觸發）：
├── 看到特定物體 → 觸發該物體的操作記憶
│   （看到紅色杯子 → 檢索"上次抓這個杯子時滑了，要用更大力"）
├── 進入特定區域 → 載入該區域的詳細記憶
│   （走到廚房 → 載入廚房物體佈局和操作歷史）
├── 遇到失敗 → 檢索類似失敗的修復經驗
│   （夾爪滑動 → 檢索所有"滑動"相關的修復策略）
└── 語言指令中的關鍵詞 → 觸發相關技能記憶
    （"小心地"→ 載入精細力控制策略）
```

**為什麼這很重要**：VLA 的推理速度是硬約束（需要 real-time，通常 5-10 Hz）。不能像 Claude Code 一樣把所有記憶都塞進 context——必須按需檢索。MEM 的做法（只保留 16 frames ≈ 1 min 的視覺記憶 + 文字壓縮長期記憶）本質上就是這個模式的工程實現。

**對應最新論文**：
- MemER 的核心機制——高層 policy 從歷史中「選擇相關 keyframes」而非盲目保留所有歷史——就是 lazy loading 的精確對應
- MemoryVLA 的「從 memory bank 中檢索 decision-relevant entries」也是按需載入

### 模式 4：200 行限制 / Context 經濟學

**Claude Code 原型**：
- MEMORY.md 只載入前 200 行 / 25KB
- 詳細內容移到 topic files，MEMORY.md 作為索引
- 超出閾值的內容不載入

**VLA 遷移 → 「記憶壓縮 + 索引-細節分離」**：

```
記憶索引層（永遠在 context 中，嚴格容量限制）：
├── 環境摘要："廚房，3個檯面，冰箱在左側"（text tokens）
├── 任務進度："已完成步驟 1-3，當前步驟 4：切菜"（text tokens）
├── 關鍵經驗索引："物體A→成功策略P1，物體B→注意事項N1"
└── 總 budget：< 固定 token 數（如 MEM 中 ≈ 幾百 tokens）

記憶細節層（按需檢索，不佔常駐 context）：
├── 完整操作軌跡的視覺特徵
├── 詳細的力-位移曲線記錄
├── 歷史 episode 的完整 embedding
└── 高精度空間地圖
```

**MEM 的實現正好驗證了這個模式**：
- 短期記憶：16 frames 的視覺壓縮 → 像 MEMORY.md 的 200 行限制
- 長期記憶：文字摘要（"I picked up the plates"）→ 像 MEMORY.md 的簡明索引
- 當需要細節時，model 可以通過 chain-of-thought 機制回溯 → 像 Claude 按需讀取 topic files

### 模式 5：Auto Dream / 記憶整合

**Claude Code 原型**：
- 觸發條件：24h + 5 sessions
- 四階段：定向 → 收集信號 → 整合 → 修剪索引
- 關鍵操作：相對日期→絕對日期、刪除矛盾事實、合併重疊條目

**VLA 遷移 → 「離線記憶蒸餾」**：

這可能是遷移中**最有價值**的設計模式。

```
觸發條件（雙門控，類似 Claude Code）：
├── 累積 N 個 episode（如 > 50 episodes）
└── 距離上次蒸餾 > T 時間（如 > 8 小時）

四階段蒸餾流程：

Phase 1: 定向（Orientation）
  → 讀取當前記憶目錄：各物體的操作記憶、環境地圖、任務策略
  → 標記「新增/修改/未變」

Phase 2: 收集信號（Signal Mining）
  → 從近期 episode transcripts 中提取：
    - 失敗 → 成功的轉折點（最高價值信號）
    - 反覆出現的困難模式
    - 新發現的物體屬性或環境變化
    - 人類干預/糾正的時刻（類似 Claude Code 提取使用者糾正）

Phase 3: 整合（Consolidation）
  → 核心操作：
    - 合併冗餘經驗（10 次相同抓取 → 1 條壓縮策略）
    - 更新物體模型（"杯子重量從 200g 變為 350g"——有人換了杯子）
    - 刪除過時環境記憶（"桌上的書已經被移走"）
    - 將 episode-specific 知識泛化為 task-level 知識

Phase 4: 修剪與索引（Prune & Index）
  → 更新記憶索引層（保持在 budget 內）
  → 刪除低價值記憶（長期未被檢索的經驗）
  → 按相關性重排序
```

**這在現有論文中幾乎是空白**：目前的 MemoryVLA、MEM、MemER 都關注 episode 內的記憶管理，但**跨 episode 的記憶整合和蒸餾**（即 Auto Dream 的對應物）基本沒有被研究。這是一個重要的研究方向。

### 模式 6：Subagent 記憶（獨立作用域）

**Claude Code 原型**：
- 每個 subagent 有獨立的記憶目錄（user/project/local 三種作用域）
- Subagent 記憶跨 session 累積領域知識
- 主 agent 和 subagent 記憶互不污染

**VLA 遷移 → 「模組化技能記憶」**：

```
主控制器記憶（Main Agent Memory）：
├── 全局環境地圖
├── 任務進度追蹤
└── 高層決策歷史

技能模組記憶（Skill-Specific Memory）：
├── 抓取技能記憶（Grasp Skill Memory）
│   scope: user（跨所有任務可用）
│   └── 各物體最佳抓取姿態、力度配置
│
├── 導航技能記憶（Navigation Skill Memory）
│   scope: project（環境特定）
│   └── 路徑偏好、障礙物記錄、動態物體模式
│
├── 裝配技能記憶（Assembly Skill Memory）
│   scope: local（特定工作站）
│   └── 零件對位精度校正、工具使用順序
│
└── 人機交互記憶（HRI Memory）
    scope: user（跟人走）
    └── 使用者偏好（"這個人喜歡杯子放右邊"）
```

**為什麼 scope 很重要**：
- 抓取技能對各種物體的最佳姿態 → 跨環境可遷移（user scope）
- 導航路徑 → 只對特定環境有效（project scope）
- 工具校正參數 → 只對特定工作站有效（local scope）

這直接對應了 VLA 領域的一個開放問題：**如何在 cross-embodiment transfer 中選擇性遷移記憶**。並非所有記憶都應該跟著機器人走——環境特定記憶應該留在環境中。

### 模式 7：透明可審計（Plain Text Memory）

**Claude Code 原型**：
- 所有記憶都是人類可讀的 Markdown
- `/memory` 命令隨時查看、編輯、刪除
- 完全透明，沒有黑箱

**VLA 遷移 → 「可解釋記憶表徵」**：

這是 VLA 記憶系統中**安全認證的核心需求**。

```
不可解釋的記憶（當前主流，問題大）：
├── 高維向量 embedding → 人無法理解機器人記住了什麼
├── Attention weights → 無法解釋為什麼選擇了某個歷史 frame
└── 隱式記憶 → 出了事無法審計

可解釋的記憶（Claude Code 啟示，方向對）：
├── 文字摘要記憶（MEM 已實現）
│   "I picked up the plates" → 人可讀、可審計、可修正
│
├── 結構化經驗記錄
│   { object: "red_cup", strategy: "side_grasp", force: 3.2N,
│     success: true, note: "handle fragile" }
│   → 運維人員可以直接查看和修改機器人的「經驗」
│
├── 規則化約束記憶
│   "IF force > 5N AND object.material == 'glass' THEN abort"
│   → 安全審計人員可以驗證安全規則
│
└── 記憶變更日誌
    → 類似 git log，記錄每次記憶更新的原因和內容
    → 出問題時可以回溯到「是哪次經驗導致了這個行為」
```

**MEM 的文字記憶是目前最接近這個方向的工作**，但它只用於任務進度追蹤，還沒有擴展到完整的可審計記憶系統。

---

## 三、三視角辯論：這個遷移方案可行嗎？

### 🔴 Bull：為什麼這個方向可能改變 VLA 的遊戲規則

1. **MEM、MemoryVLA、MemER 已經驗證了子模式的可行性**。MEM 的文字記憶 = 模式 4（索引-細節分離）+ 模式 7（可解釋記憶），成功率提升 +62%。MemER 的 keyframe 選擇 = 模式 3（lazy loading）。這些不是理論設想，是已跑通的工程。[來源: MEM paper 2026.03, MemER paper 2025.10]

2. **模式 5（Auto Dream/離線記憶蒸餾）是真正的藍海**。目前所有 VLA 記憶論文都聚焦 episode 內記憶，沒有人系統性地做跨 episode 記憶整合。Claude Code 的 Auto Dream 提供了一個成熟的工程參考。機器人每天工作 8 小時，夜間做記憶蒸餾——完美的時間窗口。

3. **模式 2（分層優先級）解決了 VLA 安全的核心問題**。物理先驗層不可覆蓋 = 硬性安全約束永遠生效。這比現在的"hope the model learns safety"要可靠得多，也更容易通過安全認證（ISO 10218/15066）。

### 🔵 Bear：為什麼這可能只是漂亮的類比

1. **模態差異是根本障礙**。Claude Code 的記憶是文字（Markdown），讀寫成本幾乎為零。VLA 的記憶包含高維視覺特徵、力回饋序列、6-DoF 軌跡——這些東西不能簡單壓縮成文字。MEM 用文字記憶只解決了語義層，精細運動控制的記憶（"用多大力夾這個雞蛋"）必須是向量形式的。[無來源⚠️，推斷]

2. **Real-time 約束與 Claude Code 完全不同**。Claude Code 可以花 500ms 讀一個文件，用戶不會在意。VLA 機器人如果花 500ms 做記憶檢索，手臂已經偏離了安全軌跡。所有的 lazy loading 和按需檢索都受到 5-10 Hz 控制頻率的硬約束。

3. **Auto Dream 的前提是有 transcripts**。Claude Code 可以存 JSONL 的 session transcript，Auto Dream 從中提取信號。VLA 的 "transcript" 是什麼？原始相機流 + 關節角度序列 + 力感測器數據，每小時 TB 級別。不可能全存，存了也不好 grep。[推斷]

4. **可解釋記憶 vs 性能的 trade-off**。文字記憶好審計，但表達能力有限。MemoryVLA 用的是不可解釋的向量 memory bank，反而在精細操作上表現更好（+41% on ShellGameTouch）。安全 vs 性能可能是不可調和的張力。[來源: MemoryVLA ICLR 2026 results]

### 🟢 Arbiter 判決

**整體判斷：70% 置信度 [投注]——這個遷移方向是有價值的，但需要分層實施。**

校準後：70% × 0.9 = 63%

**具體判斷**：

| 模式 | 可行性 | 理由 |
|------|--------|------|
| 1. 雙軌記憶 | ✅ 已驗證 | MEM 和 MemoryVLA 已經在做 |
| 2. 分層優先級 | ✅ 高價值 | 解決安全認證問題，EchoVLA 的雙記憶初步驗證 |
| 3. Lazy Loading | ✅ 已驗證 | MemER 的 keyframe 選擇就是這個 |
| 4. Context 經濟學 | ✅ 已驗證 | MEM 的視覺壓縮 + 文字摘要就是這個 |
| 5. Auto Dream | ⚠️ 未驗證但最有價值 | 研究藍海，需要解決 transcript 存儲問題 |
| 6. Subagent 記憶 | ⚠️ 部分可行 | scope 概念有用，但實現複雜 |
| 7. 可解釋記憶 | ⚠️ 部分可行 | 文字層可以，向量層目前無法 |

**什麼能推翻我**：
- 如果有人證明端到端訓練的隱式記憶（不需要顯式記憶管理）能在 1000+ 個任務上 scale → 整個顯式記憶架構的價值被削弱
- 如果 context window 在 VLA 模型中快速增長到能容納數小時的完整歷史 → lazy loading 和壓縮的必要性降低

**致命實驗**：
花 2 週，在一個中等複雜的長時序任務（如 MEM 的 Recipe Setup）上實現模式 5（離線記憶蒸餾）。對比：
- A 組：每次 episode 從頭開始，無跨 episode 記憶
- B 組：有 MEM 式的 episode 內記憶，但無跨 episode 整合
- C 組：B + 離線記憶蒸餾（每 50 episodes 做一次）

如果 C 在 200 episodes 後顯著優於 B（成功率 > 10% 差距），則驗證了 Auto Dream 遷移的核心假設。

---

## 四、一頁紙架構提案：VLA-Memory OS

```
┌─────────────────────────────────────────────────────┐
│                   VLA-Memory OS                      │
│          (Inspired by Claude Code Memory)            │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────────┐  ┌──────────────┐                 │
│  │ Demonstration │  │ Self-Learned │  ← 雙軌記憶     │
│  │   Memory      │  │   Memory     │                 │
│  │  (人類示範)    │  │  (機器人經驗) │                 │
│  └──────┬───────┘  └──────┬───────┘                 │
│         └────────┬────────┘                          │
│                  ▼                                    │
│  ┌─────────────────────────────────────────┐        │
│  │      Hierarchical Memory Store          │        │
│  │  ┌─────────────────────────────────┐    │        │
│  │  │ L0: Physics Prior (immutable)   │    │        │
│  │  │ L1: Environment Memory          │    │        │
│  │  │ L2: Task Memory                 │    │        │
│  │  │ L3: Step Memory (highest prio)  │    │        │
│  │  └─────────────────────────────────┘    │        │
│  └──────────────┬──────────────────────────┘        │
│                 ▼                                    │
│  ┌─────────────────────────────────────────┐        │
│  │     Context-Aware Memory Router          │        │
│  │  ┌───────────┐  ┌───────────────────┐   │        │
│  │  │  Always-On │  │  On-Demand Fetch  │   │        │
│  │  │  Index     │  │  (Lazy Loading)   │   │        │
│  │  │  (≤budget) │  │  (trigger-based)  │   │        │
│  │  └───────────┘  └───────────────────┘   │        │
│  └──────────────┬──────────────────────────┘        │
│                 ▼                                    │
│  ┌─────────────────────────────────────────┐        │
│  │        VLA Policy (Action Head)          │        │
│  │   memory-conditioned action generation   │        │
│  └──────────────┬──────────────────────────┘        │
│                 ▼                                    │
│  ┌─────────────────────────────────────────┐        │
│  │     Offline Memory Distillation          │        │
│  │        (Auto Dream for Robots)           │        │
│  │  Trigger: N episodes + T hours           │        │
│  │  Signal → Consolidate → Prune → Index    │        │
│  └─────────────────────────────────────────┘        │
│                                                      │
│  ┌─────────────────────────────────────────┐        │
│  │     Auditable Memory Layer               │        │
│  │  Text summaries + structured logs        │        │
│  │  Human-readable, editable, deletable     │        │
│  └─────────────────────────────────────────┘        │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## 五、研究機會排序（按 Impact × Novelty）

| 排名 | 方向 | Impact | Novelty | 理由 |
|------|------|--------|---------|------|
| 🥇 | 離線記憶蒸餾（Auto Dream for Robots） | 極高 | 極高 | 完全空白領域，Claude Code 提供成熟參考 |
| 🥈 | 分層安全記憶（L0 不可覆蓋） | 高 | 高 | 解決 VLA 安全認證的核心卡點 |
| 🥉 | 跨環境記憶遷移（Scope-based Transfer） | 高 | 中 | user/project/local scope 概念新穎 |
| 4 | 雙模態記憶索引（文字索引 + 向量細節） | 中 | 中 | MEM 已做文字部分，需擴展向量部分 |
| 5 | 可審計記憶系統（記憶版 git log） | 中 | 高 | 面向安全認證的工程需求 |

---

## 六、結語

Claude Code 的 Memory 系統之所以優雅，是因為它用最簡單的技術（Markdown + 文件系統）解決了最本質的問題（有限記憶 agent 的跨 session 知識累積）。VLA 領域正在重新發明輪子——MemoryVLA、MEM、MemER 各自解決了碎片化的子問題，但缺少一個統一的架構視角。

Claude Code Memory 的七個設計模式提供了這個統一視角。其中最大的研究機會是**離線記憶蒸餾**（Auto Dream for Robots）——讓機器人「做夢」整合白天的經驗。這不是隱喻，而是一個具體的、可實現的工程方案。

> **記憶截止：2026-03-31。MEM 是截止日期前最新的重要工作（2026.03.03 發佈）。此後可能有新的 VLA 記憶論文改變格局。**
