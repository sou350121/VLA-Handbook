# LDA-1B：把 VLA 與世界模型在 DINO 隱空間裡縫合 (Latent Dynamics Action Model via Universal Embodied Data Ingestion)

> **發布時間**：2026-02-12（arXiv 2602.12215v1）
> **論文題目**：LDA-1B: Scaling Latent Dynamics Action Model via Universal Embodied Data Ingestion
> **團隊**：北京大學 · 銀河通用（Galbot）· NVIDIA · 清華 · CASIA · BAAI · 中山大學（共 23 作者，He Wang / Ming-Yu Liu / Li Yi / Yizhou Wang 領銜）
> **核心定位**：把「世界模型」與「動作策略」統一在 **DINO 隱空間**裡，用 4 個 task head + Multi-Modal DiT 同時學 policy / forward dynamics / inverse dynamics / visual forecasting；藉此**讓低質量數據與無動作視訊都能進入訓練**——3.4 萬小時異構數據養出 1B 模型，RoboCasa-GR1 達 55.4%，把同樣用 VAE 的 UWM 從 20% 直接拉到 55.4%。

長程具身一直在「VLA（純模仿）vs 世界模型（純像素生成）」之間二選一。LDA-1B 的論點是：**選錯空間**才是症結——像素級 VAE 把背景光影也納入學習目標，算力被浪費在不影響操作的細節上；改用 **DINO 視覺特徵**作為動力學的學習空間後，scaling 才走得通。

**X-Ray 開場（非專家可複述）**：
這篇論文要解決的是「機器人怎麼像 GPT 一樣，把網路規模的雜亂數據都吃進去」。傳統做法只用「專家示範」，丟掉低質量數據和無動作的人類視訊；本文把訓練拆成 4 種任務（怎麼動 / 動作後世界會怎變 / 看到變化反推動作 / 純預測未來畫面），讓不同質量的數據各擔一個角色——髒數據只訓「動力學」、無動作視訊只訓「視覺預測」。最關鍵的設計是**不在像素層學世界**，而在 DINO 提取的語義特徵層學——同樣的 1B 模型，用 VAE 拿 20%，換 DINO 拿 55.4%。對 VLA 研究者意味著：**Scaling Law 在具身能跑通的前提，是先換對空間，再談數據量**。

---

## 📍 研究全景時間線

```
2023 ─ RT-1 / RT-2 ──────────► 大規模 BC，純動作克隆
                                 │
2024 ─ OpenVLA / π₀ / Gr00T ────► VLA 鼎立期，但仍是 BC + 短程
                                 │
2024 ─ UniSim / DreamGen ───────► 世界模型派，pixel-level video
                                 │  缺點：動力學被視覺細節稀釋
                                 │
2025-03 ─ PKU+Galbot 非抓握技能論文 ★（營銷稿引述）
                                 │  ⚠️ 「首次系統化 WAM 概念」為營銷稿說法，未獨立驗證
                                 │
2025 ─ UWM (Unified World Model)─► 嘗試 policy+dynamics 共學，但 VAE 為主
                                 │  論文自己的 baseline：UWM-1B 在 RoboCasa 僅 19.3%
                                 │
2026-02 ─ LDA-1B（本文）★
                                 │  WAM 路線的 scaling 落地：
                                 │  • 換 DINO 隱空間（不是 pixel-VAE）
                                 │  • 4 head + register token 統一訓練
                                 │  • 33.8k 小時異構數據按質分工
                                 │
未來 ─  • 視覺 backbone 與動力學 end-to-end 聯訓
        • 多模態（觸覺/力）擴展
        • 自動化「按質分工」
```

**本文在演進中的位置**：在 UWM 試圖統一 policy + dynamics 但仍卡在 VAE 像素層（RoboCasa 19~20%）的背景下，本文用 DINO 隱空間 + 4 head 任務 routing 把同樣的「unified world-action model」想法做到 1B + 33.8k 小時，並用 ablation 把「**像素 vs 語義特徵**」這條 scaling 分水嶺直接展示出來（LDA 55.4% vs UWM 20.0%）。

---

## 1. 核心架構/方法總覽 (Overview / Architecture)

### 1.1 系統組件對比 (System Component Comparison)

| 模組 | 模型 / 工具 | 輸入 | 輸出 | 訓練狀態 | 備註 |
|------|------------|------|------|---------|------|
| 視覺特徵 | **DINO**（凍結） | 多視角 RGB | per-token 語義特徵 | frozen | 不學視覺，只用 |
| 語言條件 | **Qwen3-VL**（凍結） | 文字指令 | 語言 token | frozen | 透過 cross-attn 注入 |
| 主幹 | **MM-DiT**（從頭訓） | 動作 token + 視覺 token + 語言 cross-attn | 預測動作 / 預測未來特徵 | trainable | 1B 參數 |
| Action Encoder/Decoder | 可學 | 連續動作向量 | 動作 token | trainable | 隨 MM-DiT 一起訓 |
| Task Embedding | 4 個可學向量 | 訓練步指定的任務類型 | 加在 diffusion timestep embedding 上 | learnable | 切換 4 種模式 |
| Register Token | 2 個（action / visual） | 任務缺失模態的占位 | 替代缺席 token | learnable | 「填空題」 placeholder |

### 1.2 關鍵機制 (Key Mechanism)

⚡ **Eureka Moment**：
> **「不是讓世界模型多學一點動力學，而是把學習空間從像素換成 DINO 語義——讓模型不再浪費算力預測背景光影，而專注在『物體怎麼互動、狀態怎麼變』。」**

四個設計選擇的因果鏈：

1. **為什麼 DINO 不 VAE？** —— Ablation 直接打臉：相近設置下（MM-DiT + Qwen3vl + EI-30K），VAE 路線的 `UWM(MM-DiT)` 拿 20.0%、`LDA-1B`（DINO）拿 55.4%（**$\Delta \approx +35$ pp，注意 UWM(MM-DiT) 確切參數量論文未明列**）。VAE 的重構目標把算力消耗在像素細節，DINO 的判別性訓練讓特徵對背景不變、對物件結構敏感——**作者主張**後者才是動力學該住的空間（論文未給「DINO 重構訓練 / VAE 判別訓練」的中間態 ablation 來孤立證明這條因果）。
2. **為什麼要 4 個 head 而不是只訓 policy？** —— 只訓 policy 時，加更多髒數據反而下降（$\pi_{0.5}$ 在 mixed-quality 下成功率掉 –10~–20 pp）。把 4 個任務統一後，髒數據可以走「dynamics 分支」做後門訓練，不污染 policy。
3. **為什麼用 Task Embedding + Register Token？** —— 一張網路要同時會 4 件事，需要明確的「**模式開關**」。Register token 的設計很巧：訓 policy 時用 visual register 占未來畫面的位（不預測未來），訓 visual forecasting 時用 action register 占動作的位——把 4 種任務轉成同一個「填空題」的不同 mask 模式。
4. **為什麼用統一末端執行器空間？** —— 不同機器人關節空間沒法共用；但「**手腕 6-DoF + 手指狀態**」是物理通用的——人類也用 MANO 描述手部，跨本體共享。

### 1.3 信息流 (Flow Diagram)

```
  ┌──────────── 輸入流（多源異構） ────────────┐
  │  • 真機示範（高質）                       │
  │  • 真機示範（含次優/重試/暫停） ←── 髒數據  │
  │  • 仿真示範                              │
  │  • 人類第一視角 + MANO 手部標註            │
  │  • 人類無動作視訊                ←── 純視覺 │
  └────────────────┬──────────────────────────┘
                   ▼
           統一末端執行器空間
        （6-DoF wrist Δ + finger keypoints / MANO）
                   │
                   ▼
       ┌───────────────────────────┐
       │  DINO（frozen） 提取視覺特徵  │
       │  Qwen3-VL（frozen） 提取語言  │
       └────────────┬──────────────┘
                    │
                    ▼
       ┌─────────────────────────────────────────┐
       │             MM-DiT (1B params)             │
       │                                           │
       │  ┌─ action stream ──┐    ┌─ visual stream ─┐│
       │  │ 動作 token       │↔↔↔│ 未來特徵 token   ││
       │  └──────────────────┘    └──────────────────┘│
       │       共享 self-attention                    │
       │       語言 cross-attn 注入                    │
       │       AdaLN by (timestep + task_embed)        │
       └─────────────────────────────────────────┘
                    │
       ┌────────────┴────────────────────────────┐
       ▼            ▼            ▼               ▼
   policy      forward dyn   inverse dyn    visual forecast
  (a|o,l)     (o'|o,a,l)     (a|o,o',l)     (o'|o,l)
       ↑            ↑            ↑               ↑
       │ active task embedding 決定哪個 head 反向傳遞 │
       │ register token 占住缺席模態的位置             │
       └─────────────────────────────────────────┘
```

---

## 2. 數學核心：4 個任務一個 loss (Math Core)

📌 **Napkin Formula**：
```
ℓ = ||v_a - (ε_a - a_{t+1:t+k})||² · 𝟙[task uses action]
  + ||v_o - (ε_o - o_{t+1:t+k})||² · 𝟙[task uses obs]
```
> 一行直覺：**flow-matching 速度場分別預測動作與觀測，由 task embedding 開關哪邊算梯度。「不會的就 mask 掉」。**

### 2.1 統一的 flow-matching loss

論文使用 flow matching（不是傳統 diffusion）作為共同訓練目標：

```
ℓ_action = 𝔼 ‖ v_a^θ − (ε_a − a_{t+1:t+k}) ‖²
ℓ_obs    = 𝔼 ‖ v_o^θ − (ε_o − o_{t+1:t+k}) ‖²
ℓ_total  = ℓ_action + ℓ_obs
```

> **變數說明**：
> - `a_{t+1:t+k}`：未來 k 步動作 chunk
> - `o_{t+1:t+k}`：未來 k 步 DINO 特徵（**不是像素**）
> - `ε_a`, `ε_o`：對應的 noise
> - `v_a^θ`, `v_o^θ`：模型預測的速度場
>
> **直覺**：兩條 loss 都在學「從 noisy 狀態流向真實狀態」的速度。task embedding 控制哪一邊被啟動。

### 2.2 4 個 task 對應的啟動模式

| Task | Action 算 loss? | Visual 算 loss? | Register Token 占位 |
|------|:--------------:|:---------------:|--------------------|
| Policy（給觀測產動作） | ✓ | ✗（不預測未來畫面） | visual register 占未來特徵位置 |
| Forward Dynamics（給動作預測未來觀測） | ✗（動作是條件） | ✓ | — |
| Inverse Dynamics（給未來觀測反推動作） | ✓ | ✗（未來觀測是條件） | — |
| Visual Forecasting（無動作純預測未來） | ✗（沒有動作） | ✓ | action register 占動作位置 |

每個 task embedding 加在 diffusion timestep embedding 上，透過 **AdaLN** 進入每一個 MM-DiT block——**不需要為 4 個任務各建一個網路**，只需要 4 個小向量切換。

### 2.3 跨本體統一動作空間

**機器人**：
```
a_robot = [Δwrist_pose ∈ ℝ⁶]  ⊕  [finger_state]
finger_state = {
  parallel-jaw:    width ∈ ℝ¹
  dexterous hand:  keypoints in wrist frame ∈ ℝ^{3·N}
}
```

**人類**：
```
a_human = [wrist_pose ∈ ℝ⁶]  ⊕  [MANO_params]
camera_extrinsics: 保留以解耦頭部 ego-motion 與手部運動
```

> **直覺**：所有「動作」都被投影到「**手腕怎麼動 + 手指怎麼動**」的物理通用層——機器人關節空間（高度設備相關）退到 encoder/decoder 內部處理。

---

## 3. 帶數字走一遍：髒數據如何「不污染」policy (Worked Example)

考慮 mixed-quality fine-tune 場景（論文 TABLE IV）：

```
任務：Place Pen
高質量示範：100 條  →  全部走 4 個 head
低質量示範：30%（含暫停、重試、低效）→ 只走 dynamics + visual forecast head
```

**$\pi_{0.5}$（純 BC baseline）**：
- 高質量 only：60%
- 加 30% 髒：**40%（−20 pp）** ← 髒動作直接污染 policy

**LDA-1B**：
- 高質量 only：70%
- 加 30% 髒：**80%（+10 pp）**

**為什麼 +10 而不是 −10？**
- 髒數據被 task embedding **路由**到 forward dyn / visual forecast head——這兩個 head 學的是「物體掉下去會撞到桌面」「液體會流」「重力會讓未支撐物體下落」這類**與動作對錯無關**的物理規律
- Policy head 只看高質量，不被髒動作干擾
- MM-DiT 的共享 attention 讓「動力學知識」回流到 policy 的決策——同一注意力結構裡，知道「世界會怎樣變」自然有助於「該怎麼動」

**對應的 head 啟動 mask**：

```
高質量條目 (o, a, o') :   policy ✓  forward ✓  inverse ✓  forecast ✓
低質量條目 (o, a*, o') :  policy ✗  forward ✓  inverse ✗  forecast ✓
                                       ↑ a* 是次優動作；但 o→o' 物理仍真實
人類無動作視訊 (o, ?, o'):  policy ✗  forward ✗  inverse ✗  forecast ✓
```

---

## 4. 工程視角：訓練成本、硬體、可移植性 (Engineering View)

| 維度 | 數值 | 工程含義 |
|------|------|---------|
| **訓練計算** | 48 $\times$ H800 / 400k iter / 4,608 GPU-hr | 約 4 天牆鐘；中等規模實驗室可重現（不是百萬 GPU-hr 級） |
| **凍結模塊** | DINO + Qwen3-VL | 主要學主幹 + action encoder/decoder；節省顯存 |
| **參數規模** | 1B（MM-DiT 主幹） | 對比 GR00T-N1.6 / $\pi_{0.5}$ 同量級 |
| **數據規模** | EI-30K = 33.83k 小時 | 8.03k 真機 + 8.6k sim + 7.2k 人類帶動作 + 10k 人類無動作 |
| **真機 fine-tune** | 每 task 100 條 teleop（**50–80% 是專家**，其他是次優） | 直接體現「不要苛求專家數據」的論點 |
| **跨本體 fine-tune** | Galbot G1 約 1 小時數據（營銷稿說法，未在論文嚴格量化） | ⚠️ 此數字未在論文 quantification 直接對應 |
| **硬體覆蓋** | Galbot G1（雙指夾 / Sharpa 22-DoF 靈巧手）+ Unitree G1（BrainCo 10-DoF + Zed Mini） | 至少 3 種末端執行器組合；**所有同一個底模**做 fine-tune |
| **凍結特徵的代價** | DINO 特徵不可微調 | 作者自己列為 limitation：未來方向是視覺與動力學 end-to-end 聯訓 |

**部署約束**：
- 視覺特徵固定 → 對全新視角（ego 之外）泛化未驗證
- 動力學是在 DINO 特徵層學的 → 需要部署時也跑 DINO（推理開銷不可忽略）
- 無動作視訊只貢獻 visual forecast → 嚴格說它不直接提升 policy，但透過 attention 共享間接受益
- 跨本體統一動作空間需「**手動對齊座標系**」（論文原話：`manually aligned`）——換新平台仍需工程介入

---

## 5. 數據與評測 (Data & Eval)

### 5.1 EI-30K 數據組成

| 層 | 規模 | 角色 |
|----|------|------|
| 真機機器人示範（含異構平台） | **8.03k 小時** | 高質量走全 4 head；次優走 dynamics |
| 仿真機器人示範 | 8.6k 小時 | 全 head（質量受控） |
| 人類示範（帶動作標註） | 7.2k 小時 | 經統一動作空間後與機器人共訓 |
| 人類無動作視訊 | 10.0k 小時 | **只**走 visual forecasting |
| **合計** | **33.83k 小時** | — |

🧠 **作者觀點**：論文公開把「按質分工」的比例和角色寫死在表裡——這是少見的「**異構數據策略透明化**」。對比 $\pi_{0.5}$ 之類 monolithic BC 路線，LDA 等於把「數據工程」從黑盒改成可審計的訓練模式。

### 5.2 RoboCasa-GR1 主結果（TABLE II）

| Model | Vis. Rep. | MM-DiT | Success ↑ | 備註 |
|-------|-----------|:------:|:--------:|------|
| GR00T-N1.6 | — | — | 47.6 | Cosmos VLM |
| StarVLA | — | — | 47.8 | Qwen3-VL |
| GR00T-EI30k | — | — | 51.3 | GR00T 用 EI-30K 重訓 |
| UWM-0.1B | VAE | ✗ | 14.2 | 純 VAE 路線 |
| UWM-1B | VAE | ✗ | 19.3 | VAE + DiT |
| UWM(MM-DiT) | VAE | ✓ | **20.0** | **VAE 上限基準** |
| LDA(DiT) | DINO | ✗ | 48.9 | DINO + 普通 DiT |
| LDA-0.5B | DINO | ✓ | 50.7 | DINO + MM-DiT, 0.5B |
| **LDA-1B** | DINO | ✓ | **55.4** | **本文** |

**ablation 拆解**（注意：UWM(MM-DiT) 論文未明列參數量，可能與 UWM-1B 相當）：
- 表徵空間：UWM(MM-DiT) VAE 20.0 → LDA-1B DINO 55.4（**$\Delta \approx +35$ pp**，但混雜了表徵 + 數據策略 + 訓練細節差異）
- DiT → MM-DiT（控制 DINO 與 0.5B）：48.9 → 50.7（+1.8 pp）
- 0.5B → 1B（控制其他）：50.7 → 55.4（+4.7 pp，**模型 scaling 真的給力**）
- 對比 GR00T-EI30k（同數據、不同架構）：51.3 → 55.4（+4.1 pp，架構也有貢獻）

### 5.3 Mixed-Quality Fine-tune（TABLE IV）

| 任務 | $\pi_{0.5}$ 高質 | $\pi_{0.5}$ +髒 | LDA 高質 | LDA +髒 |
|------|:---------:|:--------:|:---------:|:--------:|
| Place pen | 60 | 40（**−20**） | 70 | **80（+10）** |
| Remove lid | 50 | 40（−10） | 50 | **60（+10）** |

> **觀察**：髒數據對 BC 有害（−10~−20 pp），但對 LDA 有益（+10 pp）——驗證「按質分工」假設。

### 5.4 真機長尾任務

| 任務 | 機器人 | LDA-1B | $\pi_{0.5}$ | GR00T |
|------|--------|:------:|:----:|:-----:|
| Pull Nail（低 DoF, BrainCo 手） | Unitree G1 | **80%** | ~0% | — |
| Flip Bread（高 DoF, Sharpa 手） | Galbot G1 | **90%** | 10% | — |
| Clean Rubbish（接觸豐富） | Galbot G1 | **35%** | 0% | 0% |

🧠 **作者觀點**：Pull Nail / Flip Bread 這類**接觸/摩擦主導**的任務上，BC baseline 幾乎全敗（0~10%）；LDA 在這裡的優勢不是「動作模仿更好」，而是「動力學頭學到了接觸物理」——這是把動力學分支從事後補丁變成 first-class 訓練目標的直接回報。

---

## 6. 能力與失敗模式 (Capabilities & Failure Modes)

### 6.1 能做什麼
- 異構數據按質分工（高質 / 髒 / 無動作各歸其位）
- 跨本體 fine-tune（夾爪 / 22-DoF / 10-DoF 同一底模）
- 接觸豐富 / 高 DoF 灵巧任務（Pull Nail / Flip Bread / Clean Rubbish）
- 跨指令重新規劃（演示視訊聲稱）

### 6.2 失敗模式（含論文自承）

| 場景 | 為什麼失敗 | 來源 |
|------|----------|------|
| 全新視覺視角 | DINO 特徵凍結，未訓多視角不變性 | 論文 Limitations 自承 |
| 多模態（觸覺、力、聲音） | 目前只 RGB | 論文自承 |
| 「按質分工」需手動 | 沒有自動學 weight | 論文自承 |
| 視覺與動力學 end-to-end | DINO 凍結 | 論文自承 |
| 雙臂協同 | Pyramid Cup 只在演示視訊出現，未進論文表格 | 待證 |

### 6.3 隱含假設 (Hidden Assumptions)

1. **DINO 已足夠表達物理**：DINO 是判別式預訓，沒看過動力學——論文賭的是「**好的判別特徵 $\approx$ 好的物理表徵**」。這條未被獨立證明，只是經驗性 ablation 支持。
2. **跨本體統一空間需手動對齊**：論文原話 `manually aligned`——換平台不是即插即用。
3. **4 個 task head 的權重平衡**：訓練時 task 是按比例採樣，比例本身是超參——論文未公開最優配比。
4. **無動作視訊只貢獻 visual forecast**：論文沒做「移除無動作視訊」的單獨 ablation，所以「+10k 小時」對 policy 的實際貢獻是間接的。
5. **Flow matching ≠ diffusion**：論文用 flow matching（速度場預測），不是 DDPM——對「diffusion = 慢」的擔憂部分被消解，但採樣步數未明確列出。

---

## 7. 與相關工作對比 (Comparison)

| 方法 | 學什麼 | 表徵空間 | 異構數據策略 | 跨本體 | 真機表現 |
|------|--------|---------|------------|--------|----------|
| $\pi_0$ / $\pi_{0.5}$ | policy | 像素 + token | 只用高質 | 弱 | 短程強、長程/接觸弱 |
| Gr00T-N1.x | policy + 視訊預測 | pixel-VAE | 部分異構 | 中 | 主流 baseline |
| UWM | policy + dynamics | **VAE** | 試圖統一 | 中 | RoboCasa 19~20% |
| WAM 概念類論文（PKU+Galbot 2025） | World+Action 範式（⚠️ 「首次定義」為營銷稿說法） | latent | 概念框架 | — | 概念論文 |
| **LDA-1B（本文）** | **policy + 4 head** | **DINO** | **按質分工 4 角色** | **強（多平台 fine-tune）** | **RoboCasa 55.4%；接觸任務 80~90%** |

### 🎤 面試 Tip

> **被問「VLA 和世界模型哪條路對？」** ——
> 不要選邊。先問「你說的世界模型是 pixel-level 還是 latent-level？」如果是 pixel-level，scaling 通常會被背景細節稀釋；latent-level（特別是 DINO 這類判別式特徵）才是動力學該住的地方。LDA-1B 的核心啟示是：**「VLA vs 世界模型」是假二分；真正的選擇是「在哪個空間共學它們」**。如果面試官追問 scaling law，老實答：DINO + flow-matching MM-DiT + 4 head 任務 routing 是經驗性組合，**不是理論最優**——論文自己就把「視覺 + 動力學 end-to-end 聯訓」列為 future work。

---

## 8. 待追問的開放問題

> 來源混合（peer-review 論文 + 社群營銷稿）。本節用問題形式提出，不帶傾向：

1. **RSS 2026 接收**？營銷稿稱「210 篇之一」，但項目頁與論文 metadata 均未顯示——是已接收但未官宣，還是僅 submit 中？
2. **「$\pi_{0.7}$」對比**？論文 baseline 是 $\pi_{0.5}$（Black et al., 2025），未看到 $\pi_{0.7}$ 的引用——營銷稿可能筆誤或指未發表的 internal 版本。
3. **GEN-1 / Generalist AI 對比**？論文中無此 baseline，純社群敘事。
4. **「1 小時 fine-tune 跨本體」**？營銷稿原話，論文中對應的是「100 條 teleop」這個量化——時長換算與「1 小時」是否對應同一個事實？
5. **無動作視訊（10k 小時）的單獨貢獻**？論文沒做「移除這 10k」的 ablation——它對 policy 成功率的真實貢獻是 +X%？
6. **DINO 為什麼比 VAE 強這麼多（+30 pp）**？論文給的是「判別特徵對背景不變」的直覺解釋，但缺對照：用 DINO**重構式**訓練、或 VAE **判別式**訓練的中間態實驗？
7. **數據污染風險**？EI-30K 中真機 8.03k + 仿真 8.6k 的具體來源（是否包含 RoboCasa 訓練分割）？這會影響 RoboCasa-GR1 評測的可信度。
8. **可重現性**？項目頁明確標 `Data and checkpoints: Coming Soon`——目前只 code，論文表格能否被獨立復現尚未驗證。

📎 **內容類型可信度參考**：

| 來源 | 可信度 | 對應內容 |
|------|--------|---------|
| arXiv 論文（peer-review pending） | 🟡 中高 | 主表格、loss 公式、訓練成本 |
| 項目主頁（pku-epic.github.io/LDA） | 🟡 中 | 演示視訊、硬體列表 |
| 公司營銷稿（機器之心等） | 🔴 低 | RSS 接收、$\pi_{0.7}$ 對比、「1 小時跨本體」 |

---

📎 **來源**：
- arXiv: https://arxiv.org/abs/2602.12215（LDA-1B, 2026-02-12）
- HTML: https://arxiv.org/html/2602.12215v1
- 項目: https://pku-epic.github.io/LDA/
- 代碼: https://github.com/jiangranlv/latent-dynamics-action（**注意**：營銷稿給的 `LDA-1B` 路徑 ≠ 實際 repo 名）
- 對比基線: $\pi_{0.5}$ (Black et al., 2025) · GR00T-N1.6 (NVIDIA, 2025) · UWM (2025) · StarVLA · 早期 WAM 概念類工作（PKU+Galbot, 2025；具體論文名以原文 Related Work 為準）

🧠 **本文判讀（作者觀點）**：
這是 2026 春季少見的**「用 ablation 把整條主流路線（pixel-VAE 世界模型）打死」**的論文——20.0% vs 55.4% 不是優化問題，是路線問題。`「VLA vs 世界模型」是假二分` 這一論斷如果經得起獨立復現，會直接影響未來 12 個月的具身 foundation model 設計。**但要老實看到限制**：DINO/VLM 全凍結意味著視覺學的天花板被外部工具決定，「按質分工」需要人工標註數據質量——scaling 之路還沒到「壓進去就能煉」的 GPT-2 時刻。**🔧 級評估**（未到 ⚡，因 RSS 接收未驗證 + checkpoint 未開源），但若 3 個月內模型權重發布且能被獨立復現，可升級。

---

[← Back to World Model](./README.md) · [← Back to Theory](../README.md)
