# Xiaomi-Robotics-0：用 RoPE 偏移 + $\Lambda$-mask + 動態 loss 三件套破解 Action Prefix 的「shortcut」 (An Open-Sourced VLA Model with Real-Time Execution)

> **發布時間**：2026-02-13（v1）/ 2026-03-25（v2）（arXiv 2602.12684）；後訓練流程 2026-04-27 開源
> **論文題目**：Xiaomi-Robotics-0: An Open-Sourced Vision-Language-Action Model with Real-Time Execution
> **團隊**：Xiaomi Robotics（23 作者：Rui Cai, Jun Guo … Yuan Zhang, Quanyun Zhou）
> **核心定位**：一個 4.7B 參數的 Qwen3-VL-4B + 16 層 DiT 的 VLA 模型，論點不在「規模更大」而在**異步推理 + Action Prefixing 帶來的「shortcut」如何用三個訓練技巧抵消**——讓模型既能「動作流暢無縫」又能「對視覺信號保持反應」，而不是退化成路徑慣性。Apache 2.0 開源 + HF 權重 + 483 stars。

VLA 部署到真機時最大的兩難：要**異步推理**（一邊執行當前動作一邊算下一段）才能避免延遲卡頓；異步推理要**Action Prefixing**（用上一段動作的尾巴作為下段的條件）才能拼接平滑——但 Action Prefixing 會讓模型「**走捷徑**」，直接複製動作慣性而忽視當前視覺。Xiaomi-Robotics-0 的工程貢獻是把這條 shortcut 用三個技術同時封死。

**X-Ray 開場（非專家可複述）**：
這篇論文要解決的是「機器人連續動作怎麼又快又穩」。傳統做法：每次推理一段動作再執行——但推理 80ms 太慢、動作會卡頓；改成異步推理（執行同時算下一段）+ 把上一段動作的尾巴餵給下一段做開頭——動作變絲滑了，但模型開始**偷懶**：直接根據「之前的動作」推測「下一個動作」，眼睛幾乎不看當前畫面。小米的解法是三個訓練時的小技巧——把動作的位置編號偏移開、加一個 $\Lambda$ 形遮罩限制注意力流向、根據誤差大小動態加重 loss——讓模型即使有 prefix 仍必須看視覺。對 VLA 研究者意味著：**異步推理是落地必經之路，但伴隨的 shortcut 是訓練時就要正面處理的工程議題，不是「換個 backbone」能繞過的**。

---

## 📍 研究全景時間線

```
2023 ─ RT-1 / RT-2 ──────► 同步推理：等動作算完才執行
                              │
2024 ─ ACT / Diffusion Policy ─► 動作 chunk + 重訓
                              │
2024 ─ π0 / OpenVLA ─────► VLA 主流：Flow matching + DiT
                              │
2025 ─ π0.5 / Real-Time Chunking (RTC) ──► 引入 Action Prefixing
                              │  做平滑切換 → shortcut 副作用浮現
                              │
2026-02 ─ Xiaomi-Robotics-0（本文 v1） ★
                              │  正式承認 shortcut 為訓練問題：
                              │  (1) RoPE 位置偏移
                              │  (2) Λ-shape attention mask
                              │  (3) 動態 loss re-weighting
                              │
2026-03 ─ 論文 v2（refine 版本） ★
2026-04-27 ─ 後訓練流程 + 耳機收納 demo 開源 ★
```

**本文在演進中的位置**：把「異步推理 + Action Prefixing」這條落地必經路徑上的副作用（shortcut）正面寫進訓練技巧——不是另一個更大模型，而是**讓現有架構真正能跑**的一組工程修補。

---

## 1. 核心架構/方法總覽 (Overview / Architecture)

### 1.1 系統組件 (System Component)

| 模組 | 設定 | 說明 |
|------|------|------|
| 視覺-語言主幹 | **Qwen3-VL-4B-Instruct**（凍結 KV cache） | 提供 vision + language tokens |
| 動作生成器 | **16 層 Diffusion Transformer (DiT)** | 條件於 VLM KV cache + state token + action prefix |
| 訓練目標 | **Flow matching**（不是 DDPM） | 連續動作向量 |
| Action chunk 長度 `T` | LIBERO/CALVIN: T=10；SimplerEnv: T=4；真機: **T=30** | 任務適配 |
| 部署模式 | **異步推理**：執行當前 chunk 時並行算下一 chunk | 80ms 推理 / RTX 4090 |
| 拼接機制 | **Action Prefixing**：上一 chunk 末尾 w 步作為下一 chunk 的 prefix | 平滑連接 |
| 模型總量 | **4.7B 參數** | — |
| License | **Apache 2.0** | 開源權重 + 代碼 |

### 1.2 關鍵機制 (Key Mechanism)

⚡ **Eureka Moment**：
> **「Action Prefixing 解決了動作平滑問題，但同時打開了一條 shortcut——模型可以光看『前一段動作』就猜出下一段，視覺被忽視。三個小技巧（RoPE 偏移 + $\Lambda$-mask + 動態 loss）的目的都是『**讓 prefix 變得無法被偷懶利用**』。」**

四個設計選擇的因果鏈：

1. **為什麼要異步推理？** —— 同步等推理完才動會卡頓（80ms 推理 + 執行延遲）；異步讓機器人「邊動邊想」，工業部署的硬性要求。
2. **為什麼要 Action Prefixing？** —— 異步推理產生兩段動作，邊界不對齊會抖動；用上一段尾巴作為下一段開頭可平滑過渡——這是 RTC（Real-Time Chunking）標準做法。
3. **為什麼會有 shortcut？** —— Prefix 與目標動作高度相關（時間連續），模型發現「**直接複製 prefix 比看視覺更省事**」，loss 也會降下來。論文原話：「policy learning may take a shortcut by simply copying the action prefix instead of attending to the visual and language inputs」。
4. **為什麼三個技巧而不是一個？** —— 三條 attack vector 互不重疊：
   - **RoPE 偏移**讓模型在位置編碼層面就分清「這是 noisy 待預測動作 vs clean prefix」
   - **$\Lambda$-mask** 在 attention 層面限制 noisy action token 只能看 VLM/state/前 $w$ 步，不能看後段的 prefix
   - **動態 loss re-weighting** 在訓練目標層面對誤差大的樣本加權，避免模型只學「容易的部分」

### 1.3 信息流 (Flow Diagram)

```
   ┌──────────── 異步推理迴路（每 Tₑ 步觸發下一輪） ────────────┐
   │                                                          │
   │ t=0: 推理 chunk_0 (T 步)                                   │
   │       │                                                  │
   │       ▼ 執行 step 0...Tₑ-1                                 │
   │ t=Tₑ: 同時開始推理 chunk_1                                 │
   │       │   ↓                                              │
   │       │   chunk_1 開頭 w 步 = chunk_0 末尾 w 步（prefix）  │
   │       │   ↓                                              │
   │       ▼   ▼                                              │
   │  繼續執行 chunk_0  +  chunk_1 推理中（DiT, 16 layers）     │
   │                                                          │
   └──────────────────────────────────────────────────────────┘

   DiT 內部 attention（Λ-shape mask 簡圖）：
   
       VLM_tokens   state   prefix_w   noisy_action(待預測)
   VLM_tokens     ✓         ✓         ✓
   state          —         ✓         ✓
   prefix_w       —         —         ✓ ← prefix 自己只看自己
   noisy_action   ◀── 透過 KV cache + state + prefix_w 觀察
                  ◀── 但 noisy_action 之間的後段不可看 prefix 之後的動作
                  ╰── 「Λ」形：頂端窄（看少）、下方寬（看多）
```

---

## 2. 數學核心：三招封住 shortcut (Math Core)

📌 **Napkin Formula**：
```
ℓ = Σ_i  w(δ_i) · ||v_θ(x_i, c_i, prefix_i^[+RoPE_offset]) − (ε_i − a_i)||²
              ↑                         ↑
       誤差自適應權重               位置編號偏移避免時間捷徑
                                   + Λ-mask 限制注意力路徑
```
> 一行直覺：**同樣的 flow matching loss，但三個附加件讓「複製 prefix」比「看視覺推理」更難。**

### 2.1 三個技巧的形式化

**(1) RoPE Positional Index Offset**

論文原話：「we simply add an offset to the RoPE positional indices of the noisy action tokens to enable the model to distinguish tokens of noisy actions from those of the clean action prefix.」

```
noisy_action_tok 的 RoPE index = base_index + Δ    (Δ 為固定偏移)
clean_prefix_tok 的 RoPE index = base_index
```

> **直覺**：noisy 與 prefix 在 RoPE 空間被「拉開」——模型不再能透過位置編碼的相鄰性把它們當「自然延續」處理。

**(2) $\Lambda$-Shape Attention Mask**

論文原話：「A noisy action token can only attend to the vision and language tokens via the VLM KV cache, the sink token, the state token, and the action tokens of the previous w timesteps.」

```
mask[i, j] = 1  iff  j ∈ {VLM_KV, sink, state, prefix_{t-w:t}}
mask[i, j] = 0  otherwise
```

> **直覺**：noisy action 不能往「後面」看（後面是 prefix 後續或別的 noisy）——它的注意力被強制送回 VLM/state/前 $w$ 步的「**真信息**」，不能在 noisy 群裡互相抄。「$\Lambda$」形指注意力連通圖頂端窄、底部寬。

**(3) Adaptive Loss Re-weighting**

論文原話：「we dynamically re-weight the flow-matching loss based on the $L_1$ error between the online-predicted actions and the ground-truth actions. This strategy prioritizes samples with larger deviations.」

```
w(δ_i) = f(||a_pred_i − a_gt_i||₁)    (誤差越大，權重越大)
ℓ = Σ_i w(δ_i) · ||v_θ(...) − target_i||²
```

> **直覺**：對誤差大的樣本加重——避免模型沉迷於 prefix 容易的樣本（這些樣本誤差天生小）。f 的具體形式論文未給。

### 2.2 Flow Matching 主目標（不變）

```
ℓ_FM = ||v_θ(x_τ, c, prefix) − (ε − a)||²    (τ ∈ [0,1])
```
其中 `x_τ = (1-τ)·a + τ·ε`，`v_θ` 預測速度場。

> **變數說明**：
> - `a`：ground-truth 動作 chunk（T 步）
> - `prefix`：上一 chunk 末尾 w 步
> - `c`：VLM features + state token
> - `ε ~ N(0, I)`：noise

---

## 3. 帶數字走一遍：$\Lambda$-mask 為什麼是 $\Lambda$ (Worked Example)

考慮一段 chunk，noisy action token 排成 t = 1, 2, ..., 10：

**沒有 mask（常規 transformer self-attn）**：
```
noisy_t=10 可以看到：
  - VLM_tokens   ← 真信息
  - state        ← 真信息
  - prefix       ← 真信息
  - noisy_1..9   ← 但是！這些是 noisy/還在學習的 ← 互相污染
```
→ 模型學到「noisy 互相對齊」=「動作前後一致」=**shortcut**：跟著前面動作走就行，不用看視覺。

**Λ-mask 強制**：
```
noisy_t=10 只能看到：
  - VLM_tokens (KV cache)
  - sink token
  - state token
  - prefix 的 w 步（注意：是 clean prefix，不是 noisy 兄弟）
```
→ 必須**從視覺/語言/state 重建未來動作**，沒有 noisy 兄弟可抄。

**為什麼叫 Λ**：
- 頂端（最早的 noisy）能看的較少（只有 VLM + state + 短 prefix）
- 底端（最晚的 noisy）能看的也是同樣這些「外部信號」
- 但中間的 noisy **無法互看**——形成「兩腳立地、頂部閉合」的 Λ 形連通圖

---

## 4. 工程視角：訓練資源、推理延遲、部署 (Engineering View)

| 維度 | 數值 / 觀察 | 工程含義 |
|------|------------|---------|
| **參數規模** | 4.7B（Qwen3-VL-4B + 16 層 DiT） | $\pi_0$ 同量級 |
| **推理延遲** | **80ms / RTX 4090**（單次 chunk 推理） | 異步推理掩蓋之後實際無感 |
| **預訓練數據** | ~**200M timesteps** robot + **80M** vision-language | 真機 trajectory 量級在 OpenVLA 之上 |
| **數據來源** | DROID + MolmoAct + 內部遙操作（**Lego 338 hr / Towel 400 hr**） | ⚠️ 營銷稿沒提 DROID/MolmoAct |
| **VL : Robot 採樣比** | **1 : 6** | 重 robot，輕 VL |
| **後訓練示範** | **20 hours**（耳機收納 demo） | ⚠️ 耳機 demo **不在論文 v2**，是 04-27 開源時新增 |
| **License** | **Apache 2.0** | 商用友善 |
| **倉庫成熟度** | 483 stars / 50 forks / 6 commits 主分支 | 早期但活躍 |
| **依賴** | PyTorch 2.8 / Transformers $\geq$4.57.1 / FlashAttn 2.8.3 / Python 3.12 / CUDA 12.8 | 新依賴；舊環境需升級 |

**部署約束**：
- 異步推理依賴執行延遲與推理延遲的時序匹配——任務動作頻率 ≠ 80ms 整數倍時抖動仍可能出現
- T=30（真機）chunk 對長程任務不夠——超出 30 步任務需多次拼接
- VLM 凍結意味著新領域（如外科手術、戶外）需重新預訓 VL 部分

---

## 5. 數據與評測 (Data & Eval)

### 5.1 仿真 benchmark 全表

| Benchmark | 子任務 | 成功率 |
|-----------|--------|:------:|
| **LIBERO** | Spatial | 98.8% |
| | Object | **100.0%** |
| | Goal | 98.8% |
| | Long | 97.2% |
| | **Average** | **98.7%** |
| **CALVIN** | ABCD$\to$D 平均完成長度 | **4.80**（vs FLOWER 4.67） |
| | ABC$\to$D 平均完成長度 | **4.75** |
| **SimplerEnv** | Visual Matching | 85.5% |
| | Visual Aggregation | 74.7% |
| | WidowX | 79.2% |

⚠️ **AGENTS.md 仿真飽和警告適用**：LIBERO 已 95-99% 飽和區間，這個數字主要說明「不會比同行差」，不能單獨支撐方法優越性。

### 5.2 真機任務（論文 v2 範圍）

| 任務 | 數據量 | 狀態 |
|------|--------|------|
| Lego Disassembly | 338 hours 遙操作 | 預訓 + post-training |
| Towel Folding | 400 hours 遙操作 | 預訓 + post-training |
| **Earbud Packing** | **20 hours** | ⚠️ **論文 v2 未提**，是 2026-04-27 開源 demo |

⚠️ **數據缺口**：
- **真機任務缺定量成功率**：論文 demo 是質性視訊
- **三技巧 ablation 缺數值表**：論文有「同步 vs 異步」對比，但**缺「有/無 RoPE 偏移」「有/無 Λ-mask」「有/無動態 loss」的逐項數值**——讀者無法判斷各技巧貢獻
- **耳機收納 20 小時聲稱**只有 demo，無 baseline 對照

---

## 6. 能力與失敗模式 (Capabilities & Failure Modes)

### 6.1 能做什麼
- LIBERO 全四子任務 $\geq 97\%$
- CALVIN ABCD$\to$D 4.80（超過 FLOWER）
- 異步推理 80ms / RTX 4090
- 後訓練 20 小時即可學新真機任務（耳機 demo 聲稱）
- Apache 2.0 + HF 權重，學界可直接 fine-tune

### 6.2 失敗模式

| 場景 | 為什麼失敗 |
|------|----------|
| 真機高速衝擊類接觸（非 Lego/Towel 範圍） | 預訓沒有 sufficient 接觸動力學數據 |
| 未在 200M timestep 預訓覆蓋的場景 | 跨形態真機未驗證跨度 |
| T > 30 的長程任務 | chunk 上限；需多次拼接，shortcut 在拼接邊界仍可能漏 |
| 動作頻率與 80ms 不匹配的硬體 | 異步時序對不齊導致抖動 |

### 6.3 隱含假設 (Hidden Assumptions)

1. **VLM 凍結是夠的**：所有視覺語義由 Qwen3-VL-4B 提供——VLM 沒見過的場景（如夜視、紅外）需重新預訓 VL 部分
2. **三技巧協同是必要的**：論文沒做「只用 1/2/3 條」的單獨 ablation——可能其中 1 條已足夠，但無數據驗證
3. **Action Prefixing 寬度 w 是關鍵超參**：論文未公開最優 w 與 T 的比例
4. **DROID + MolmoAct 預訓覆蓋夠廣**：兩者都是公開大規模 robot dataset，但仍偏「日常桌面操作」，極端場景未必涵蓋
5. **Flow matching ≠ 標準 diffusion**：論文用 FM；採樣步數比 DDPM 少，這是延遲 80ms 的部分原因，但 v2 論文沒給 step 數

---

## 7. 與相關工作對比 (Comparison)

| 方法 | VLM | 動作頭 | 異步推理 | Action Prefixing | Shortcut 處理 | LIBERO Avg |
|------|-----|--------|:--------:|:----------------:|:------------:|:----------:|
| OpenVLA | LLaMA-2-7B | autoregressive | ✗ | ✗ | — | 76.5% |
| $\pi_0$ | PaliGemma | flow-matching DiT | 部分 | ✗ | — | 94.2% |
| $\pi_{0.5}$ (RTC) | PaliGemma | DiT | ✓ | ✓ | 部分（不公開細節） | 95.4% |
| Gr00T-N1.5 | Eagle-VL | DiT | ✓ | ✓ | 部分 | ~96% |
| **Xiaomi-Robotics-0** | **Qwen3-VL-4B** | **16 層 DiT, FM** | **✓ (80ms)** | **✓** | **3 招正面處理** | **98.7%** |

### 🎤 面試 Tip

> **被問「VLA 異步推理為什麼會 shortcut？怎麼處理？」** ——
> 三句話答：(1) 異步推理需要 Action Prefixing 拼接平滑，但 prefix 與目標動作高度相關，模型可以「複製 prefix 而忽視視覺」走捷徑；(2) Xiaomi-Robotics-0 的解法是三條獨立路徑：RoPE 位置偏移（位置編碼層拉開）+ $\Lambda$-mask（注意力層限制）+ 動態 loss 加權（loss 層加重困難樣本）；(3) 但要老實補一句：**論文沒給三技巧的逐項 ablation 數字**——只能說「協同有效」，不能說「每條獨立必要」。

---

## 8. 待追問的開放問題

> 來源混合（arXiv 論文 v2 + GitHub + 項目主頁 + 公司營銷稿）：

1. **「HuggingFace 全球 VLA 下載榜第六名」**？論文與項目頁均無此排名數據——是 2026-02 短期排名還是穩定排名？
2. **「Random Masking 隨機遮蔽」是否真存在**？營銷稿原話「自適應加權 + $\Lambda$-mask + **隨機遮蔽**」三招——但論文 v2 的三招實際是「自適應加權 + $\Lambda$-mask + **RoPE 偏移**」。隨機遮蔽**論文中未發現**——是營銷稿翻譯誤植，還是後訓練流程中新增了第四個技巧未在論文體現？
3. **「20 小時耳機收納」demo 真機成功率**？營銷稿稱「連續完成多組」——多組是幾組？成功率如何？論文 v2 不含此任務。
4. **三技巧的逐項 ablation**？論文只給「同步 vs 異步」對比，缺「有 RoPE 偏移 vs 無」「有 $\Lambda$-mask vs 無」「有動態 loss vs 無」的單獨數字——讀者無法判斷各技巧貢獻。
5. **$\Lambda$-mask 的 $w$ 與 $T$ 比例**？$w$ 太大易 shortcut，太小拼接抖動——論文沒給最優搭配。
6. **DROID + MolmoAct 之外是否有未公開來源**？338h Lego + 400h Towel 合計 ~30 天遙操作數據；總體 200M timesteps 是否還有其他內部數據未列？
7. **異步推理在 RTX 4090 之外的硬體表現**？80ms 是 4090 數字；消費級 / 邊緣 GPU（如 5070 Ti laptop、Jetson Orin）的延遲是否仍可滿足異步條件？
8. **Apache 2.0 license vs Qwen3-VL 上游 license**？Qwen3-VL 自有條款（特別是「商用」上的限制需確認）——使用 Xiaomi-Robotics-0 商用時要追溯 VLM 上游。

📎 **內容類型可信度參考**：

| 來源 | 可信度 | 對應內容 |
|------|--------|---------|
| arXiv 論文 v2 | 🟡 中高 | 三技巧定義、LIBERO/CALVIN/SimplerEnv 表、80ms 推理 |
| GitHub README | 🟡 中高 | 4.7B 規模、Apache 2.0、依賴清單 |
| 項目主頁（robotics.xiaomi.com） | 🟡 中 | 預訓數據規模、耳機 demo |
| 公司營銷稿（機器人前瞻） | 🔴 低 | HF 排名第六、「隨機遮蔽」第三技巧名稱、20 小時連續多組 |

---

📎 **來源**：
- arXiv: https://arxiv.org/abs/2602.12684（Xiaomi-Robotics-0, 2026-02-13 v1 / 03-25 v2）
- HTML: https://arxiv.org/html/2602.12684v2
- 項目主頁: https://robotics.xiaomi.com/xiaomi-robotics-0.html
- 代碼: https://github.com/XiaomiRobotics/Xiaomi-Robotics-0（Apache 2.0, 483★, 50 forks）
- 權重: https://huggingface.co/collections/XiaomiRobotics/xiaomi-robotics-0
- 對比基線: $\pi_0$ (Black et al., 2024) · $\pi_{0.5}/\text{RTC}$ (Black et al., 2025) · Gr00T-N1.5 (NVIDIA, 2025) · OpenVLA (Kim et al., 2024) · DROID · MolmoAct

🧠 **本文判讀（作者觀點）**：
這篇論文的真正貢獻**不是「另一個更強的 VLA」**，而是把 RTC（Real-Time Chunking）落地時的隱形痛點（shortcut）正式寫進**訓練處方**。三招（RoPE 偏移 + $\Lambda$-mask + 動態 loss）在概念上互不重疊，**有工程美感**——但缺逐項 ablation 是論文最大的數據漏洞，**讀者無法判斷三招誰是必要、誰是邊際**。

仿真數字飽和（LIBERO 98.7%）不是真正的賣點——真正的賣點是 **「異步 80ms + 真機 20 小時學會新任務 + Apache 2.0 開源」**。但 20 小時 demo 在論文 v2 不存在（只在 04-27 後訓練流程開源時亮相），**真機定量成功率仍缺**——這是 ⚡ 戰略級評估被卡在 🔧 的關鍵原因。

**評級 🔧 可操作（潛在 ⚡）**：開源完整、權重可用、依賴明確、對 shortcut 問題的處方有**工程 reusability**——任何做 RTC 的團隊都可以拿這三招回去用。若 3-6 個月內：(a) 三技巧逐項 ablation 數字補上；(b) 耳機 demo 與基線對比成功率公佈；(c) 跨形態真機驗證——可升級至 ⚡。

---

[← Back to VLA Core](./README.md) · [← Back to Theory](../README.md)
