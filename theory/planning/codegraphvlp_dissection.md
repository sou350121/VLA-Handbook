# CodeGraphVLP：當 VLA 不再假設「看當下就夠」(Code-as-Planner Meets Semantic-Graph State for Non-Markovian VLA)

> **發布時間**：2026-04（arXiv 2604.22238v1）
> **論文題目**：CodeGraphVLP: Code-as-Planner Meets Semantic-Graph State for Non-Markovian Vision-Language-Action Models
> **團隊**：University of Arkansas · Max Planck Research School · Google Research · TU Wien · University of Liverpool  
> **核心定位**：把 VLA 的 Markovian 假設打破——用「持久語義圖 + 一次合成的 Python planner + 抽離雜訊的視覺 prompting」三件套，把 $\pi_0$ 在歷史相關長程任務上從 $30\%$ 拉到 $82\%$，同時把規劃延遲從 $\sim 3\ \text{sec/step}$ 砍到 $0.33\ \text{sec/step}$（$\sim 9\times$）。  

長程操作的核心痛點不在「動作生成」，而在「過去發生了什麼會影響現在該做什麼」。CodeGraphVLP 不擴大 context window 也不堆 memory token，而是把「狀態」顯式寫成圖、把「進度推理」一次性編譯成 Python 程式，VLA 只負責當下子任務的執行。

**X-Ray 開場（非專家可複述）**：
這篇論文解決的是：VLA 模型假設「看現在就夠」，但長程任務（如交換兩個杯子位置）需要記住「最初誰在哪」——一旦相機看不見遮擋物，VLA 就崩潰。作者的解法不是讓 VLA 去記憶，而是把「世界狀態」抽出成一張隨時更新的圖（誰在誰上、誰拿著什麼），再讓 GPT-5 一次性寫出一段 Python 規劃器去查這張圖、決定下一個子任務。對 VLA 研究者意味著：**短程反應式策略 + 結構化規劃層** 可能比「memory-augmented VLA」更務實——延遲低 $9\times$、成功率高 $25$ 個百分點，且 $\pi_0$ 完全不用改。  

---

## 📍 研究全景時間線

```
2023 ─ Code-as-Policies (Liang et al.)
         │  LLM 生程式碼，但程式碼直接控動作（reactive）
         │
2024 ─ ReKep / VoxPoser
         │  程式碼產生「約束/關鍵點」，再交給 motion solver
         │
2024 ─ π₀ / OpenVLA / RT-2  ← 主流 VLA：Markovian (a_t = π(o_t, l))
         │
2025 ─ π₀.₅ / Hi-Robot      ← 雙層：VLM-in-loop 規劃 + VLA 執行
         │  缺點：每步都呼叫 VLM，延遲高，無持久狀態
         │
2025 ─ HAMLET / MemoryVLA / MemER  ← memory-augmented VLA
         │  把歷史壓進 token，scaling history → 計算 ↑
         │
2026 ─ CodeGraphVLP（本文）  ★
         │  持久語義圖（顯式狀態）+ 一次合成的 code planner
         │  → π₀ 不改，只改「給它看什麼 + 給它什麼指令」
         │
未來 ─  • 開放世界圖生成（自動學 schema）
        • 合成程式的形式化驗證
```

**本文在演進中的位置**：把「程式碼」從「直接控制」（CAP）→「產生約束」（ReKep）→**「task-level planner over persistent state」**（本文），是程式化規劃 + 結構化記憶的合流。  

---

## 1. 核心架構/方法總覽 (Overview / Architecture)

### 1.1 系統組件對比 (System Component Comparison)

| 模組 | 工具 / 模型 | 輸入 | 輸出 | 頻率 | 訓練/推理差異 |
|------|------------|------|------|------|--------------|
| Instance Segmenter | YOLOE（fine-tune 10 視訊） | RGB obs `o_t`（多視角） | per-view mask 集 `ℳᵛ` | 每步 | fine-tune 帶機械臂 mask |
| Relevance Filter | VLM + Set-of-Mark | 標號的 mask + 指令 `l` | 任務相關物件子集 | 任務初始 | 一次性 |
| Cross-View Associator | CLIP（cosine + 幾何距離） | 多視角 mask | 同一物的跨視角關聯 | 每步 | 純推理 |
| Online Tracker | Cutie | 上一幀節點 + 新幀 | 持久節點追蹤 | 每步 | 純推理 |
| **Code Planner `𝒫`** | **GPT-5（一次合成）** | 圖 schema + `𝒢₀` + 指令 | Python 程式（含 `policy(graph)`） | **任務初始 1 次** | **只跑 1 次 LLM 呼叫** |
| Planner Runtime | 純 Python | `𝒢_t` + `task_memory` | `(l_t^sub, 𝒪_t^rel)` | 每個 action chunk | 純執行 |
| **VLA Executor** | **$\pi_0$**（fine-tune 過）   | 抹乾淨的觀測 `õ_t` + state `s_t` + 子指令 `l_t^sub` | action chunk `τ_t`（H=10） | 10 Hz | fine-tune 用「子指令 + 抹乾淨圖」 |

### 1.2 關鍵機制 (Key Mechanism)

⚡ **Eureka Moment**：
> **「不是讓 VLA 記住歷史，而是把『當前世界狀態』顯式維護在圖裡——VLA 永遠是 Markovian 的，但它『看到的』和『被告知的』已被規劃器壓縮過。」**

四個設計選擇的因果：
1. **為什麼要圖而不是 token memory？** —— Token memory 隨歷史線性增長，圖只記「物件 + 關係」，且自然支援可查詢的謂詞（`holding(x)` / `in(x, y)` / `on(x, y)`）。
2. **為什麼程式碼一次合成而不是每步重新規劃？** —— $9\times$ 延遲差距（$0.328$ vs $2.967\ \text{sec/step}$）。一次合成把推理成本攤銷，運行時只是純 Python 圖查詢。  
3. **為什麼要 clutter-free prompting？** —— Ablation 顯示去掉它 swap-cups 直接從 85% → 40%（–45 pp）。VLA 的視覺注意力會被無關物件污染。
4. **為什麼 $\pi_0$ 不需要重新設計？** —— 把長程任務切成「子任務 + 子任務相關物件」後，每個子任務都回到 short-horizon Markovian——這正是 $\pi_0$ 已經擅長的範圍。  

### 1.3 信息流 (Flow Diagram)

```
                 ┌──────────────────────────────────────────────┐
                 │           初始化（任務開始時 1 次）              │
                 │                                              │
   o_0 ──► YOLOE ──► VLM (Set-of-Mark) ──► CLIP+幾何 ──► 𝒢₀     │
   l   ─────────────────────────────────────────┐              │
                                                 ▼              │
                                          GPT-5 合成              │
                                            程式 𝒫               │
                 └──────────────────────────────┬───────────────┘
                                                │
   ┌────────────────────────── 每步循環（10 Hz） ────────────────┐
   │                                                            │
   │  o_t ──► Cutie 追蹤 + YOLOE 增量 ──► 更新 𝒢_t                │
   │                          │                                 │
   │                          ▼                                 │
   │                    𝒫(𝒢_t) ──► (l_t^sub, 𝒪_t^rel)            │
   │                                       │                    │
   │                          ┌────────────┴───────────┐        │
   │                          ▼                        ▼        │
   │                    抹乾淨 mask                  子指令       │
   │                    M_t^v = max_{i∈𝒪^rel} m_i^v               │
   │                    Ĩ_t^v = I_t^v ⊙ M_t^v                   │
   │                          │                        │        │
   │                          └─────────┬──────────────┘        │
   │                                    ▼                       │
   │                            π₀（fine-tuned）                  │
   │                                    │                       │
   │                                    ▼                       │
   │                          τ_t = (a_t, ..., a_{t+9})         │
   │                                    │                       │
   │                                    ▼                       │
   │                          機器人執行 → 新 o_{t+10}            │
   └────────────────────────────────────────────────────────────┘
```

---

## 2. 數學核心：圖如何餵 $\pi_0$ (Math Core)  

📌 **Napkin Formula**：
```
τ_t = π₀( I_t ⊙ mask(𝒫(𝒢_t)) ,  s_t ,  l_t^sub from 𝒫(𝒢_t) )
```
> 一行直覺：**$\pi_0$ 的輸入被「圖驅動的規劃器」雙重壓縮——視覺只剩相關物件，語言只剩當前子任務。**  

### 2.1 Markovian vs Non-Markovian 公式對照

**標準 VLA（公式 1）**：
```
τ_t = (a_t, ..., a_{t+k-1}) = π_θ(o_t, s_t, l)
```
假設「當前觀測足夠」。

**最優 non-Markovian 政策（公式 2）**：
```
τ_t = π*(h_t, l)        h_t = (o_{0:t}, s_{0:t}, τ_{0:t-1})
```
但 `h_t` 隨時間線性膨脹——這就是 memory-augmented VLA 的 scaling 瓶頸。

**本文的近似**：用 `𝒢_t` 替代 `h_t`，並用程式 `𝒫` 把 `𝒢_t` 投影到「子任務 + 相關物件」：
```
(l_t^sub, 𝒪_t^rel) = 𝒫(𝒢_t)            # 公式 4
M_t^v = max_{i ∈ 𝒪_t^rel} m_{i,t}^v     # 公式 5（per-view mask 取並集）
Ĩ_t^v = I_t^v ⊙ M_t^v                  # 公式 6（element-wise）
```

### 2.2 訓練目標（公式 7）
```
max_θ  𝔼_{(õ, s, τ, l^sub) ~ 𝒟}  [ log π_θ(τ | õ, s, l^sub) ]
```

> **變數說明**：
> - `õ`：抹乾淨的多視角觀測（注意：訓練時就用抹乾淨的——不是只在測試時才抹）
> - `l^sub`：規劃器產生的短指令（從長程示範自動切出）
> - `𝒟`：把長程示範「子任務化」後的訓練元組
>
> **直覺**：本質上是「重新標註」——把長程示範用 `𝒫` 切成子任務區段，每段都是 `(õ, s, τ, l^sub)`，π₀ 學的是「在乾淨視野下執行短指令」。

### 2.3 多視角關聯距離（公式 3）
```
d(mᵛ, aᵛ) = ‖ 𝐜(mᵛ) − 𝐜(aᵛ) ‖₂
𝐝̃ᵛ(m) = 𝐝ᵛ(m) / max_{a ∈ 𝒜} d(mᵛ, aᵛ)
```
先用 CLIP 視覺距離（語意相同）配對，剩下未配對的退到幾何距離（用 anchor 集做歸一化），閾值 `τ_vis` / `τ_geo` 控制嚴格度。

---

## 3. 帶數字走一遍：玩具例子 (Worked Example — Swap Cups)

任務：3 個盤、2 個杯（黑/藍）隨機放在兩個盤上、第 3 盤空（buffer）。要求**交換**黑藍杯位置。

**初始 $\mathcal{G}_0$**（簡化）：  
```
nodes:  plate_1, plate_2, plate_buffer, cup_black, cup_blue
edges:  cup_black —on→ plate_1
        cup_blue  —on→ plate_2
attrs:  plate_buffer.is_empty = True
```

**GPT-5 合成的 `policy(graph)` 大致樣貌**：
```python
def policy(g):
    if "swap_plan" not in g.task_memory:
        # 一次性規劃出三步搬運
        g.task_memory["swap_plan"] = [
            ("pick_up", "cup_black", "plate_1"),
            ("put_into", "cup_black", "plate_buffer"),
            ("pick_up", "cup_blue",  "plate_2"),
            ("put_into", "cup_blue",  "plate_1"),
            ("pick_up", "cup_black", "plate_buffer"),
            ("put_into", "cup_black", "plate_2"),
        ]
    for step in g.task_memory["swap_plan"]:
        if not satisfied(g, step):
            verb, obj, tgt = step
            return (
                f"{verb.replace('_',' ')} the {obj.replace('_',' ')}"
                f" {'from' if verb=='pick_up' else 'into'} the {tgt.replace('_',' ')}",
                {obj, tgt}              # 𝒪^rel
            )
    return ("done", set())
```

**運行軌跡**：
| step | `𝒢_t` 變化 | `l^sub` | `𝒪^rel` | $\pi_0$ 看到的視覺   |
|------|-----------|---------|---------|---------------|
| 0 | 初始 | `pick up the cup black from the plate 1` | `{cup_black, plate_1}` | 只剩黑杯 + 盤 1 |
| 1 | `holding(cup_black)` | `put into the cup black into the plate buffer` | `{cup_black, plate_buffer}` | 只剩黑杯 + buffer |
| 2 | `cup_black on plate_buffer` | `pick up the cup blue from the plate 2` | `{cup_blue, plate_2}` | 只剩藍杯 + 盤 2 |
| ... | ... | ... | ... | ... |

**為什麼 π₀ 單跑會崩**：第 0 步若 π₀ 看到全場（兩杯三盤），它分不清「該把黑杯放回原盤還是放到 buffer」——因為「buffer 是暫存」這條資訊不在當前畫面裡。CodeGraphVLP 把它**寫進 `l^sub`** + **把無關盤抹掉**，π₀ 的決策瞬間變成 1-NN：「黑杯在我手上，畫面只有 buffer 盤，我就放下去」。

---

## 4. 工程視角：延遲、樣本、可移植性 (Engineering View)

| 維度 | 數值 / 觀察 | 工程含義 |
|------|------------|---------|
| **規劃延遲** | 0.328 sec/step（純 Python 圖查詢） | 對比 VLM-in-loop 的 $2.967\ \text{sec/step}$ → **$9\times$ 加速**；意味著可以高頻重規劃   |
| **VLA 推理頻率** | 10 Hz（action chunk H=10） | 每秒約 1 次 chunk 預測；規劃器跟得上 |
| **訓練成本** | 4× A6000、lr=1e-5、bs=128、50K iter | 中等規模 VLA fine-tune；$\pi_0$ 不從頭訓   |
| **示範量** | 100/100/200 條（per task） | 偏少；但因為訓練時已被「子任務化」，等效樣本更多 |
| **LLM 呼叫** | GPT-5 任務初始 **1 次** | 跨集 amortize；長期部署成本可忽略 |
| **VLM 呼叫** | 只在 `𝒢₀` 初始化用 Set-of-Mark | 不是 per-step，避免 $\pi_{0.5}$ / Hi-Robot 的延遲問題   |
| **硬體** | UR10e + Robotiq 2F-85 + 雙視角（肩 + 腕） | 單臂、平行夾、無觸覺；雙視角是必需（多視角關聯依賴它） |
| **隱藏成本** | YOLOE fine-tune（10 視訊含機械臂 mask） | 換新平台需重 fine-tune YOLOE，這條被論文淡化 |

**部署約束**：
- 需要至少 2 個視角才能跑跨視角關聯——單目部署未驗證
- 程式碼合成依賴 LLM 對任務 schema 的理解——schema 寫得不好程式可能不執行
- 圖更新速率 $\leq$ YOLOE/Cutie 速率，遮擋極端時節點可能丟失  

---

## 5. 數據與評測 (Data & Eval)

### 5.1 三個自設真機任務（皆 history-dependent）

| 任務 | 為什麼是 non-Markovian | 示範量 |
|------|---------------------|--------|
| **Pick-and-Place Twice** | 兩盤外觀近似，第二步要記得「初始」是哪個盤 | 100 |
| **Place-and-Stack** | 立方體放入杯後被遮擋，必須記得它在哪個杯 | 100 |
| **Swap Cups** | 隨機初始配置；buffer 盤是暫存而非目的地 | 200（黑優先 100 + 藍優先 100） |

### 5.2 完整對比表（Table I）

| 方法 | PnP Twice | Place-Stack | Swap Cups | **Avg** |
|------|:---------:|:-----------:|:---------:|:-------:|
| $\pi_0$ FAST   | 0% / 0% | 0% / 0% | 0% / 0% | **0.0%** |
| $\pi_0$   | 0% / 0% | 60% / 40% | 55% / 50% | **30.0%** |
| $\pi_{0.5}$   | 5% / 0% | 35% / 5% | 25% / 10% | **5.0%** |
| Gr00T N1.5 | 50% / 35% | 40% / 40% | 70% / 20% | **31.7%** |
| Gr00T N1.5 + Multi-frame（4 幀，1s 間隔） | 100% / 75% | 50% / 50% | 90% / 45% | **56.7%** |
| **CodeGraphVLP** | **100% / 80%** | **95% / 80%** | **100% / 85%** | **81.7%** |

### 5.3 Ablation（Swap Cups）

| 配置 | Code Planner | Semantic Graph | Success | Latency (s/step) |
|------|:------------:|:--------------:|:-------:|:----------------:|
| 無圖 + VLM-in-loop | ✗ | ✗ | 25% | 2.967 |
| 有圖 + VLM-in-loop | ✗ | ✓ | 55% | 3.142 |
| **有圖 + Code Planner** | **✓** | **✓** | **85%** | **0.328** |

| 視覺 prompting | Success |
|----------------|:-------:|
| 不抹乾淨 | 40% |
| **抹乾淨（full）** | **85%** |

**關鍵發現**：
- 圖貢獻 $+30$ pp（$25 \to 55$），但延遲沒改善  
- Code Planner 取代 VLM-in-loop 同時拿到 $+30$ pp **和** $9\times$ 延遲縮減  
- 抹乾淨視覺單獨貢獻 +45 pp——這是最大單一 ablation 增量

**🧠 評論**：這套 ablation 的順序設計有意思——Gr00T 加 multi-frame 都拿到 +25 pp，說明「給歷史」確實有效；但 CodeGraphVLP 的關鍵不是「給更多」而是**「給更少更精準」**（規劃器 + 抹乾淨）。⚠️ 所有實驗都在自設任務上、UR10e 單一硬體；跨形態/雙臂遷移性未驗證。

---

## 6. 能力與失敗模式 (Capabilities & Failure Modes)

### 6.1 能做什麼
- 顯式 history-dependent 任務（記憶初始配置、buffer 中介、被遮擋目標）
- 高頻規劃（$9\times$ 於 VLM-in-loop）  
- 不改 VLA 主體（$\pi_0$ 即插即用）  
- 把長程示範「子任務化」當資料增強

### 6.2 不能做 / 隱含失敗模式

| 場景 | 為什麼會失敗 | 對應限制 |
|------|------------|---------|
| 未知物件類別（開放世界） | YOLOE 不認得 $\to$ 不入圖 $\to$ 規劃器看不到   | 「閉集」假設 |
| 視角極端 / 鏡頭遮擋 | VLM 屬性/關係推斷對視角敏感（作者承認） | `Limitations` §直引 |
| 規劃器 schema 設計失誤 | GPT-5 生的 Python 不執行 / 邏輯錯 | 「careful prompt design required」（作者承認） |
| 連續關係（軟物、流體） | 圖只支援離散 _on/in/near_ 謂詞 | 規則式關係歸納 |
| 動態擾動（人/物移動） | Cutie 追蹤丟失 $\to$ 圖結構錯亂   | 沒有圖恢復機制 |
| 跨形態（雙臂、人形） | 雙視角關聯 + UR10e fine-tune 強耦合 | 未測試 |

### 6.3 隱含假設 (Hidden Assumptions)

1. **物件可被分割**：YOLOE fine-tune 過——換場景要重訓
2. **任務可被切成離散子任務**：連續控制任務（如平衡）規劃器無法表達
3. **GPT-5 寫得對**：論文沒做合成程式的形式驗證（這正是作者點名的 future work）
4. **示範對齊子任務切分**：訓練時就要把長示範按 `𝒫` 切——意味著 dataset pipeline 也要重做，這條沒被充分強調
5. **雙視角強假設**：跨視角關聯是 graph 完整性的前提

---

## 7. 與相關工作對比 (Comparison)

| 方法類別 | 代表 | 程式碼用法 | 狀態維護 | VLM/LLM 呼叫頻率 | 本文差異 |
|---------|------|----------|---------|-----------------|---------|
| Reactive code | Code-as-Policies | 直接生產動作控制 | 無 | per-step | 本文程式碼是 task planner，不直接控動作 |
| Constraint code | ReKep / VoxPoser | 生關鍵點/約束給 motion solver | 隱式 | 任務級 | 本文針對 task progress，不是 motion |
| Memory VLA | HAMLET / MemoryVLA / MemER | — | 壓縮 token / retrieval | per-step | 本文用顯式圖，scale 不靠 token |
| Hierarchical VLM-VLA | $\pi_{0.5}$ / Hi-Robot   | — | VLM 短時 | **per-step VLM** | 本文 VLM 只在初始；planner 是 Python |
| **CodeGraphVLP（本文）** | — | **task-level planner** | **持久顯式圖** | **任務初始 1 次** | — |

### 🎤 面試 Tip

> **被問「你怎麼處理 VLA 的長程記憶？」** ——
> 不要直接答「memory-augmented」。先問清楚「**任務有沒有 history-dependence？是哪一類？**」。如果是「物件配置記憶 $+$ 中介狀態」這種 CodeGraphVLP 處理的類型，**結構化狀態 $+$ 程式化規劃**比堆 memory token 更務實——延遲低 $9\times$、成功率高 $25$ pp、底層 VLA 可以不動。但要老實補一句：**閉集物件、雙視角假設、跨形態未驗證**——這是真正在工程落地時會踩的坑。  

---

📎 **來源**：
- arXiv: https://arxiv.org/html/2604.22238v1（CodeGraphVLP, 2026-04）
- 對比基線出處：$\pi_0$ (Black et al., 2024) $\cdot$ $\pi_{0.5}$ (Black et al., 2025) $\cdot$ Gr00T N1.5 (NVIDIA, 2025) $\cdot$ Code-as-Policies (Liang et al., 2023) $\cdot$ ReKep (Huang et al., 2024) $\cdot$ MemoryVLA / HAMLET / MemER（綜述參見論文 Related Work）  

🧠 **本文判讀（作者觀點）**：
這篇是 2026 春季 VLA 投稿洪峰中**罕見的「不堆 token、不擴 context」的長程方案**。價值不在絕對數字（自設任務、單一硬體），而在**把「Markovian 假設」這個業界默認前提顯式拆掉**——並提供一個**$\pi_0$ 不用改**的最小侵入方案。最大隱憂是**閉集 $+$ GPT-5 一次合成**，開放世界 $/$ 形式驗證是作者自己點名的 future work——說明他們也清楚瓶頸在哪。

---

[← Back to Planning](./README.md) · [← Back to Theory](../README.md)
