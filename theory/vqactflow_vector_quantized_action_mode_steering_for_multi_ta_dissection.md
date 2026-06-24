# VQActFlow：向量量化動作流的多任務機器人操控 (VQActFlow: Vector-Quantized Action Mode Steering for Multi-Task Robot Manipulation)

> ⚙️ 本文由 Moltbot 自動生成 | 2026-06-24
>
> **論文**: VQActFlow: Vector-Quantized Action Mode Steering for Multi-Task Robot Manipulation
> **連結**: https://arxiv.org/abs/2606.21600
> **核心定位**: 用 VQ-VAE 將連續動作 tokenize 成離散 codebook，再以 Variational Flow Matching 生成 code 序列，讓多任務策略在生成過程中始終維持對「動作模式」的明確偏好，從而解決多任務學習中的模式混淆問題。

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | 在 VFM 生成過程中維持對離散動作 codebook 的分類偏好，使 CFG + Codebook Critic 能在推理時精確 steering 動作模式選擇 |
| 適合精讀 | 做多任務機器人策略、探索 VQ-VAE + Flow Matching 結合、需要推理時 guidance 機制的研究者 |
| 可以跳過 | 只做單任務操控、只關心 VLA 語言層面不關心動作生成的讀者 |
| 落地可行性 | 中（需要 VQ-VAE 預訓練 + 兩階段訓練，但 codebook critic 可獨立於策略調整） |
| 主要風險 | 實驗集中在桌面操控和人形機器人拾放，未驗證動態/移動場景；codebook 大小敏感（K=512 最優） |

💡 **X-Ray 開場**
多任務機器人面對同一場景、不同語言指令時，需要從多模態演示分佈中選出正確的動作模式。傳統方法在連續空間生成，模式選擇是隱式的——一旦選錯就是執行錯誤任務。VQActFlow 的核心洞察是：把動作 tokenize 成離散 codebook，讓模式選擇在生成的每一步都是顯式的分類問題，從而讓推理時的 guidance 有明確的「錨點」可以操作。對 VLA 研究者而言，這提供了一條「動作層离散化 + 生成式 steering」的新路徑，與 OpenVLA 的 per-dimension binning 形成對比。

📍 **研究全景時間線**
```
[2023] Diffusion Policy (連續動作空間)
    → [2023] VQ-BeT (殘差 VQ-VAE + 分類頭 + 連續偏移)
    → [2024] OpenVLA (per-dimension scalar binning)
    → [2024] Discrete Policy (VQ-VAE + 連續潛空間 diffusion，末端才 quantize)
    → [2026-06] VQActFlow ← 當前位置：VQ-VAE + VFM 全程分類偏好
    → 局限：僅桌面/人形拾放，未觸及移動/動態場景
```

## 1. 核心架構/方法總覽 (Overview / Architecture)

### 1.1 系統對比概覽 (System Component Comparison)

| 組件 | 輸入 | 輸出 | 訓練方式 | 推理角色 |
|------|------|------|----------|----------|
| VQ-VAE Encoder | 動作 chunk a[1:H] ∈ R^{H×d_a} | 連續潛變量 z ∈ R^{L×d_e} | 重建 + commitment loss | 凍結，產生監督目標 |
| VQ-VAE Decoder | quantized embeddings z_q | 重建動作 chunk | 與 encoder 聯合訓練 | 凍結，將 code 序列解碼為動作 |
| VFM Policy (DiT) | 噪聲 x_t, t, 視覺 o, 語言 c | codebook logits h ∈ R^{L×K} | 交叉熵 (Eq.6) | 核心生成器，每步輸出分類偏好 |
| CFG Module | 條件 logits h_c + 無條件 h_∅ | 引導 logits h̃ | 訓練時 p_drop 置空語言 | 推理時放大語言信號 |
| Codebook Critic | o, c, p, t | 可行性分數 | InfoNCE 對比學習 | 推理時梯度引導 p 向可行模式 |

**關鍵超參數**:

| 超參數 | 數值 | 說明 |
|--------|------|------|
| Codebook 大小 K | 512 | 消融實驗顯示 K=512 最優（LIBERO-90） |
| 時間下採樣 S | 2 | L = H/2^S，壓縮序列長度 |
| DiT 層數 | 12 | hidden=1024, 8 heads |
| Critic 層數 | 3 | hidden=256, 4 heads |
| CFG weight w | 最佳 w=2~4 | LIBERO-90: w=2; LIBERO-Goal: w=4 |
| Critic λ_max | 1.0 | 與 CFG 組合時 λ=1.0 |

### 1.2 關鍵機制 (Key Mechanism)

**為什麼這樣設計？**

傳統多任務策略的痛點：同一視覺觀察 + 不同語言指令 → 需要完全不同的動作。在連續動作空間中，diffusion/flow matching 的 guidance 操作的是連續分佈，模式選擇是隱式的。Discrete Policy 雖然用了 VQ-VAE，但在連續潛空間做 diffusion，只在採樣末端才 nearest-neighbor lookup 回離散 index——生成過程中「沒有明確知道自己要選哪個模式」。

VQActFlow 的解決方案：
1. **VQ-VAE tokenization**：把連續動作 chunk 編碼成 L 個離散 code index（K=512 的 codebook）
2. **VFM 分類訓練**：不訓練速度場回歸（如標準 CFM），而是訓練分類頭預測每個位置的 code index（交叉熵損失）
3. **後驗加權速度**：推理時從分類輸出 p_k^{(l)} 計算後驗均值 μ₁，再用 OT 速度公式推導速度場
4. **統一 guidance 接口**：CFG 和 Codebook Critic 都作用於分類分佈 p，而非連續速度或動作

⚡ **Eureka Moment**：「讓生成過程中的每一步都輸出對 K 個動作模式的分類偏好——這樣 guidance 就有了一個明確的、可操作的錨點，而不是在連續潛空間中盲目地重塑分佈。」

### 1.3 信息流/架構圖 (Flow / Diagram)

```
訓練階段:
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  動作 chunk     │────>│  VQ-VAE Encoder  │────>│  Code Index     │
│  a[1:H]         │     │  (凍結後使用)     │     │  k*[1:L]        │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                    ┌─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  高斯噪聲 x_0   │────>│  OT 插值         │────>│  x_t = (1-t)x_0 │
│  t ~ U(0,1)     │     │  x_t = (1-t)x_0  │     │     + t·z_q     │
│                 │     │  + t·z_q         │     └────────┬────────┘
└─────────────────┘     └──────────────────┘              │
                                                          │
                                                          ▼
                                                 ┌─────────────────┐
                                                 │  VFM Policy     │
                                                 │  f_θ(x_t,t,o,c) │
                                                 │  → logits h     │
                                                 │  → p = softmax(h)│
                                                 └────────┬────────┘
                                                          │
                                                    L_VFM = CE(p, k*)
                                                          │
                                                          ▼
                                                 ┌─────────────────┐
                                                 │  反向傳播更新 θ  │
                                                 └─────────────────┘

推理階段:
┌──────────┐    ┌──────────────────────────────────────────────────┐
│ x_0 ~ N  │───>│  ODE 積分 (Euler steps, t=0 → 1):               │
│ (噪聲)    │    │                                                  │
└──────────┘    │  1. f_θ(x_t, t, o, c) → h_c                     │
                │     f_θ(x_t, t, o, ∅) → h_∅                     │
                │  2. CFG: h̃ = h_∅ + w·(h_c - h_∅)               │
                │  3. p = softmax(h̃)                              │
                │  4. Critic: ∇_p C_ψ → p̃ = Proj[p + λ·∇_p C]   │
                │  5. μ₁ = Σ p_k · e_k                           │
                │  6. v_θ = (μ₁ - x_t) / (1-t)                    │
                │  7. x_{t+Δt} = x_t + Δt · v_θ                   │
                └────────────────────┬─────────────────────────────┘
                                     │
                          x_1 (終端狀態)
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │ position-wise quantize         │
                    │ k*_l = argmin_k ‖x_1^(l) - e_k‖│
                    └────────────┬───────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────────────┐
                    │ VQ-VAE Decoder                 │
                    │ a[1:H] = D(e_{k*_1}, ...,     │
                    │              e_{k*_L})         │
                    └────────────────────────────────┘
```

## 2. 數學核心 (Math Core)

📌 **Napkin Formula**（一行抓住本質）：
```
v_θ(x_t, t) = [Σ_k p_k^{(l)} · e_k - x_t] / (1-t)
其中 p = softmax( CFG引導 logits + Critic梯度 )
```

**目標**：從高斯噪聲生成動作 code 序列，使生成的 code 序列能解碼為正確執行語言指令的動作。

**核心方程**：

訓練損失（分類交叉熵）：
```
L_VFM = -Σ_{l=1}^{L} [h_{k*_l}^{(l)} - log(Σ_j exp(h_j^{(l)}))]
```
其中 h^{(l)} = f_θ(x_t, t, o, c)^{(l)} ∈ R^K 是位置 l 的 logits，k*_l 是 ground-truth code index。

推理速度（從後驗均值導出）：
```
μ₁^{(l)} = Σ_{k=1}^{K} p_k^{(l)} · e_k        (後驗加權均值)
v_θ(x_t, t) = (μ₁ - x_t) / (1-t)              (OT 速度場)
```

CFG 引導（在 logits 空間操作）：
```
h̃ = h_∅ + w · (h_c - h_∅)
p̃_k^{(l)} = exp(h̃_k^{(l)}) / Σ_j exp(h̃_j^{(l)})
```

Codebook Critic 引導（在概率單純形上操作）：
```
p̃ = Proj_Δ [p + λ(t) · Π_T[∇_p C_ψ(o, c, p, t)]]
λ(t) = λ_max · (1 - H(p) / log K)    // 熵自適應權重
```

> 符號與本文保持一致：
> - x_t: OT 插值點 ∈ R^{L×d_e}，t 為 flow 時間步
> - p ∈ R^{L×K}: 每個位置對 K 個 code 的分類概率
> - e_k: codebook 第 k 個 embedding ∈ R^{d_e}
> - o: 視覺觀察編碼; c: 語言指令編碼; ∅: 空語言 embedding
> - H(p): 分類分佈的熵; Π_T: 切空間投影（減均值）; Proj_Δ: 單純形投影

**直覺**：速度場不是直接回歸的，而是從「政策當前偏好哪些動作 mode」的分類分佈中導出。當分佈集中在某個 code 上時，速度指向該 code 的 embedding；當分佈分散時，速度指向多個 code 的加權平均——這自然地表徵了「模式選擇的不確定性」。

## 3. 帶數字走一遍：玩具例子 (Worked Example)

假設一個簡化場景：
- Codebook 大小 K=4（4 個基本動作模式）
- 序列長度 L=2（2 個時間步的 code）
- 當前 ODE 步 t=0.5，x_t 是當前的插值點

**Step 1: Policy 前向**
```
f_θ(x_t, t=0.5, o, c) → h_c ∈ R^{2×4}
f_θ(x_t, t=0.5, o, ∅) → h_∅ ∈ R^{2×4}
```

假設位置 l=1 的 logits：
```
h_c = [2.0, 0.5, -1.0, -0.5]    (條件：「拿起紅球」)
h_∅ = [0.3, 0.2, 0.1, 0.0]     (無條件)
```

**Step 2: CFG 引導**（w=4）
```
h̃ = h_∅ + 4·(h_c - h_∅)
  = [0.3, 0.2, 0.1, 0.0] + 4·[1.7, 0.3, -1.1, -0.5]
  = [7.1, 1.4, -4.3, -2.0]
```

**Step 3: 計算概率**
```
p_unguided = softmax([2.0, 0.5, -1.0, -0.5]) ≈ [0.60, 0.21, 0.04, 0.07]
p_guided   = softmax([7.1, 1.4, -4.3, -2.0]) ≈ [0.95, 0.04, 0.00, 0.00]
```
CFG 將 mode 0 的概率從 60% 提升到 95%——模式選擇變得明確。

**Step 4: Codebook Critic 調整**
假設 critic 認為 mode 0 在當前場景中可行（分數高），mode 1 不可行（與其他 mode 衝突）：
```
∇_p C_ψ ≈ [0.1, -0.05, 0.02, 0.01]
λ(t=0.5) ≈ 0.5（中等熵，中等權重）
p̃ = Proj_Δ[p_guided + 0.5 · ∇_p C_ψ]
```
critic 進一步強化 mode 0，微弱抑制 mode 1。

**Step 5: 計算速度**
假設 4 個 code embedding：
```
e_0 = [1.0, 0.5], e_1 = [-0.5, 1.0], e_2 = [0.3, -0.8], e_3 = [-0.7, -0.3]
μ₁ = 0.95·e_0 + 0.04·e_1 + 0.00·e_2 + 0.00·e_3
   ≈ [0.93, 0.45]
v_θ = (μ₁ - x_t) / (1 - 0.5)
```
速度指向 e_0 的方向——因為模式 0 佔主導。

**Step 6: Euler 更新**
```
x_{t+Δt} = x_t + 0.1 · v_θ
```
經過多個 Euler 步，x_t 逐漸靠近 e_0 和 e_{k*_2} 的區域，最終 quantize 得到 code 序列。

## 4. 工程视角 (Engineering View)

| 工程指標 | 數值/估計 | 含義 |
|----------|-----------|------|
| Policy 參數量 | DiT-12L, d_model=1024, 8 heads | ~100M 量級（標準 DiT-S/X 規模） |
| Critic 參數量 | Transformer-3L, d_model=256, 4 heads | ~10M 量級，遠小於 policy |
| 推理步數 | Euler steps（論文未明確給出，典型 20-50 步） | Flow matching 通常比 diffusion 少步數 |
| 每步開銷 | 1× Policy forward + 1× CFG unconditional + 1× Critic forward | CFG 需額外一次 unconditional 前向 |
| Codebook lookup | O(1) per position（nearest neighbor in R^{d_e}） | K=512, d_e 通常 64-256 |
| 訓練階段 | 兩階段：VQ-VAE 預訓練 → VFM+Critic 聯合訓練 | VQ-VAE 凍結後不參與梯度 |
| 部署約束 | 需要 CLIP-B/32 + ResNet-18 + DiT + Critic | 視覺+語言 encoder 可共享/凍結 |

**工程含義**：
- **CFG 的代價**：每次 ODE 步需要兩次 policy 前向（conditional + unconditional），推理延遲翻倍。w 過高時（>4）性能下降，說明「過引導」會損害執行精度——這是 speed-accuracy 的 trade-off。
- **Critic 的優勢**：每次 guidance 步只需一次小 transformer 前向，無需在內循環中解碼 codebook。可獨立調整目標而不重訓 policy。
- **Codebook 大小的敏感性**：K=128 太粗（詞彙粒度不足），K=2048 太細（分類空間過大難以學習），K=512 是甜蜜點。這意味著 codebook 大小需要針對任務集調優。

## 5. 數據與評測 (Data & Eval)

| 基準 | 場景數 | 任務類型 | 數據來源 | 評估方式 |
|------|--------|----------|----------|----------|
| LIBERO-Goal | 10 tasks | 單任務 + 語言引導 | LIBERO 官方 | 20 rollouts/task, 成功率 |
| LIBERO-90 | 90 scenarios | 多任務混合 | LIBERO 官方 | 從零訓練, 匹配協議 |
| Unitree G1 | 4 tasks | 人形全臂拾放 | 140 VR 遙操演示/task | 20 trials/task, 硬件 |
| ALOHA-style | 未詳述 | 雙臂接觸豐富操作 | 未詳述 | 硬件評估 |

**訓練協議**（仿真基準）：
- 所有基線使用相同的視覺/語言 encoder、訓練數據集、訓練步數、初始化種子
- Discrete Policy 基線使用與 VQActFlow 共享的 VQ-VAE 權重和相同的 DiT backbone
- VLA 模型（OpenVLA, π₀）因預訓練數據集不同而未納入直接比較

**核心結果**：
- LIBERO-Goal: VQActFlow 73.0%（無引導）→ 81.0%（w=4），Discrete Policy 峰值 61.5%（w=6）
- LIBERO-90: VQActFlow + CFG(w=2) + Critic(λ=1.0) = 80.5%，超越 CFM DiT、MT-ACT、VQ-BeT (24.1%)、Discrete Policy+CFG (63.2%)
- 人形硬件: 4 個拾放任務，w=6 時改善正確任務選擇（具體數字待補充，論文截斷）

## 6. 能力與失敗模式 (Capabilities & Failure Modes)

### 能做什么
- **多任務模式選擇**：在相同視覺觀察 + 不同語言指令下，能正確選擇動作模式（LIBERO-Goal 上 CFG 提升 8%）
- **推理時 steering**：無需重訓，通過調整 w 和 λ 即可改變行為
- **場景可行性判斷**：Codebook Critic 提供獨立於語言的可行性信號（+2.0% 單獨增益）
- **跨平台泛化**：從仿真（LIBERO）到雙臂（ALOHA）到人形（G1）均有驗證

### 不能做什么 / 失敗模式
| 失敗類型 | 場景 | 原因 |
|----------|------|------|
| 錯誤任務執行 | w=1（低引導） | 無條件邊緣 h_∅ 抑制不足，政策對錯誤對象執行連貫動作 |
| 抓取失敗 | w=8（過引導） | 激進放大 (h_c - h_∅) 損害執行精度——到達正確對象但抓不準 |
| 天花板任務回退 | 接近天花板任務 | w=4 時「打開中間抽屜」從 90% 降至 75%——引導對已學好的任務反而有害 |
| Codebook 粒度不足 | K=128 | 詞彙太粗，無法區分細粒度動作差異 |
| 分類困難 | K=2048 | 分類空間過大，政策難以學習精確的 K 路分類 |

### 6.1 隱含假設 (Hidden Assumptions)

1. **VQ-VAE codebook 足夠表達所有需要的動作模式**——如果某個任務需要一個 codebook 中不存在的動作原語，政策無法生成它。這是所有 VQ 方法的共性問題。
2. **語言指令能充分區分任務**——CFG 依賴語言條件來區分模式。如果兩個任務需要不同動作但語言描述相似，CFG 可能無法有效區分。
3. **Codebook Critic 的對比負樣本覆蓋了真實場景中的不可行情況**——critic 的 7 種負樣本構造策略（時間打亂、隨機替換、錯誤觀察等）是否覆蓋了部署時的真實分佈偏移，未經驗證。
4. **桌面操控的經驗可遷移到人形機器人**——G1 實驗只做了 4 個簡單的拾放任務（140 演示/任務），遠少於 LIBERO 的訓練規模。人形機器人的全臂控制涉及更複雜的動力學，VQActFlow 在此場景的泛化能力有待更大規模驗證。
5. **Flow matching 的 OT 路徑適合离散 code 生成**——VFM 用連續傳輸 + 分類訓練的混合方案，速度場從分類後驗導出而非直接訓練。這種間接方式是否總能產生良好的速度場，缺乏理論保證。

## 7. 與相关工作對比 (Comparison)

| 方法 | 動作表示 | 生成方式 | 模式選擇 | Guidance 接口 | LIBERO-90 |
|------|----------|----------|----------|---------------|-----------|
| Diffusion Policy | 連續 | DDPM | 隱式 | 連續空間 CFG | ~70%* |
| CFM DiT | 連續 | Flow Matching | 隱式 | 連續空間 CFG | ~72%* |
| OpenVLA | per-dim binning | LM autoregressive | 隱式（per-dim） | 無 | N/A（不同協議） |
| VQ-BeT | 殘差 VQ-VAE | 分類頭 + 連續偏移 | 隱式（分層分類） | 無 | 24.1% |
| Discrete Policy | VQ-VAE | 連續潛空間 diffusion | 末端 quantize | 連續潛空間 CFG | 63.2% |
| **VQActFlow** | **VQ-VAE** | **VFM 全程分類** | **每步顯式分類** | **logits + prob 空間** | **80.5%** |

*數值為近似，基於論文中的相對比較

**面試 Tip**：「如果被問到 VQActFlow 與 Discrete Policy 的區別，回答：兩者都用 VQ-VAE tokenize 動作，但 Discrete Policy 在連續潛空間做 diffusion，只在採樣末端才 quantize——生成過程中沒有明確的模式表示；VQActFlow 用 VFM 直接訓練分類頭，每一步都輸出對 K 個模式的分類偏好，讓 guidance 有明確的操作對象。這是『隱式模式選擇』vs『顯式模式選擇』的本質區別。」

## 8. 精讀建議 (Reading Guide)

- **值得精讀原文的人**：
  1. 做多任務機器人策略、希望改善推理時任務選擇的研究者
  2. 探索 VQ-VAE + Flow Matching 結合的生成模型研究者
  3. 需要設計推理時 guidance 機制（特別是 scene-conditioned feasibility）的工程師

- **建議章節路徑**：先讀 §IV-B（VFM Policy，核心方法）→ 再看 §IV-D（Codebook Critic，創新點）→ §V-B（LIBERO-90 結果，核心實驗）→ 可跳 §III（ preliminaries，如已熟悉 VQ-VAE 和 Flow Matching）

- **不值得精讀的理由**：如果只做單任務操控、已熟悉 VQ-VAE + diffusion 架構、或不關心推理時 guidance——讀摘要和 §IV-A 的 tokenization 部分即可。本文的價值在於「全程顯式模式偏好 + 統一 guidance 接口」的設計，不做多任務的讀者可能從 Diffusion Policy 或 π₀ 獲得更多實用價值。

---
[← Back to Theory](./README.md)
