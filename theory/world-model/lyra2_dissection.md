# Lyra 2.0：用「3D 緩存路由 + 自污染訓練」對抗長軌跡視訊生成的兩大 degeneration (Explorable Generative 3D Worlds)

> **發布時間**：2026-04-14（arXiv 2604.13036v1）
> **論文題目**：Lyra 2.0: Explorable Generative 3D Worlds
> **團隊**：NVIDIA Spatial Intelligence Lab (SIL), Toronto — 15 作者，含 Sanja Fidler / Sherwin Bahmani / Tianchang Shen / Xuanchi Ren（共同通訊）
> **核心定位**：把「生成式 3D 場景」這條 video-then-lift 路線推到「**可長走、可回頭、可在 Isaac Sim 跑機器人**」的程度。論點不在「生得更逼真」，而在**正面解決長軌跡兩大病灶——空間遺忘（spatial forgetting）和時間漂移（temporal drifting）**。Apache 2.0 + 1.8k stars + HF 權重。

VLA 真機調試最大的瓶頸之一是「**沒有夠多樣的高擬真 3D 場景做 sim2real 訓練**」。Lyra 2.0 不是 photo-realistic renderer，它是**「給定相機軌跡，生成一段可重訪的 3D 場景視訊，再 lift 成 3DGS / mesh」**——直接 export 到 Isaac Sim 給機器人跑。對 VLA 工程師意味著：**未來「scene as a service」可能取代手動建場景**。

**X-Ray 開場（非專家可複述）**：
這篇論文要解決的是「AI 生 3D 世界，怎麼讓你能在裡面長時間漫遊不出 bug」。傳統視訊生成只能生 5-10 秒 clip——超過就會 (1) **忘記**剛走過的房間（回頭發現它變了個樣），(2) **漂移**（每幀小誤差累積，越走越糊）。Lyra 2.0 的解法是兩招：(1) 維護一個**「3D 緩存」**，記住每幀的深度+相機，回頭時把舊幀「投影」回當前視角再餵給 video model——所以模型不靠記憶力，靠**幾何路由**；(2) 訓練時故意把模型自己的「壞輸出」當成歷史餵回去，逼它**學會修正自己的錯誤**而非把錯誤累積下去。對 VLA 研究者意味著：**Sim2Real 數據合成的天花板被推高了**——以後給機器人造訓練場景可能比寫 Isaac 腳本還快。

---

## 📍 研究全景時間線

```
2023 ─ Stable Video Diffusion ─► 短 clip 視訊生成
                                  │
2024 ─ CAT3D / GenWarp / ReconX ─► 「video-to-3D」生成式重建範式
                                  │
2024 ─ WonderJourney / WonderWorld ─► 長場景但靠 inpainting 拼接
                                  │
2025 ─ GEN3C ─────────────────► 用點雲做相機控制，但長軌跡仍 drift
                                  │
2025 ─ CaM (Camera-as-Memory) ───► 用相機軌跡做歷史記憶，但 spatial forgetting
                                  │
2025-12 ─ Lyra 1.0（同隊） ★
                                  │  Single-image → 3D/4D 自蒸餾
                                  │
2026-04 ─ Lyra 2.0（本文） ★
                                  │  • 3D cache 做 information routing
                                  │  • Self-augmented training
                                  │  • Wan 2.1-14B DiT backbone
                                  │  • Isaac Sim 直連
                                  │
未來 ─  • 動態場景（人/物移動）
        • 跨域到 outdoor / urban
        • 更短 inference（目前 194s/step）
```

**本文在演進中的位置**：上一波（CAT3D / GenWarp / ReconX）證明「video → 3D」可行；中間一波（WonderJourney / WonderWorld）解決「能不能長」；本文是**第一個把「長 + 幾何一致 + 可回頭」一次解決**的工作，且 baseline 做了 GEN3C / CaM / SPMem 而非 cherry-picking。

---

## 1. 核心架構/方法總覽 (Overview / Architecture)

### 1.1 系統組件 (System Component)

| 模組 | 設定 | 作用 |
|------|------|------|
| **Video DiT** | Wan 2.1-14B（凍結 base + LoRA finetune） | 生成下一段 80 幀 |
| **VAE** | Wan 2.1 VAE（$8\times8$ 空間 / $4\times4$ 時間下採樣） | latent 空間操作 |
| **3D Cache 𝒞** | 全分辨率深度圖 + 相機外參 + 下採樣點雲 P_i ∈ ℝ^((H/d)×(W/d)×3) | 增量積累每幀幾何 |
| **History Retriever** | 從 𝒞 中找出與當前視角重疊的舊幀 | 解決 spatial forgetting |
| **Coordinate Map** | 3 通道 canonical coord ∈ [-1,1]^(3×H×W) + warped depth | 注入幾何約束 |
| **3DGS Decoder** | **Depth Anything v3** (DAv3) feed-forward 3DGS | video → 3D Gaussian |
| **Mesh 提取** | OpenVDB-based 分層稀疏 grid | 大規模 mesh 用 |
| **Inference 時長** | full: 194s/step on GB200；DMD: 15s/step | 主要瓶頸 |

### 1.2 關鍵機制 (Key Mechanism)

⚡ **Eureka Moment**：
> **「不要讓視訊模型『記憶』長歷史——它會崩。把『歷史』顯式存成 3D 幾何，回頭時把舊幀 warp 回當前視角再餵給模型——讓模型只負責它最擅長的（appearance synthesis），幾何由 cache 保證。」**

四個設計選擇的因果鏈：

1. **為什麼用 3D cache 而非 KV cache？** —— KV cache 線性增長 → 長軌跡 OOM；3D cache 是「世界座標系下的點雲 + 深度」，size 與場景而非時間相關。回頭時用相機投影即時取出相關幀，不需保留所有 token。
2. **為什麼幾何只用於「routing」不用於 appearance？** —— 直接用幾何 render 會把深度誤差傳到顏色；用幾何**只決定「該餵哪幀」**，外觀仍由 video model 生成——這是把 generative prior 用對地方。
3. **為什麼要 self-augmented training？** —— 訓練時模型總看到 clean ground-truth 歷史，部署時看到自己生的「污染歷史」，分布不匹配 → drift。讓訓練時主動加噪當歷史，模型學會「**從髒輸入修正回乾淨**」。
4. **為什麼最後 lift 用 DAv3 而非從頭訓 3DGS？** —— DAv3 是預訓 feed-forward 3D foundation model，省掉 per-scene optimization；本文 fine-tune 它對 Lyra 生成的視訊適配——把「研究 3DGS 提取」這個 sub-problem 推給上游。

### 1.3 信息流 (Flow Diagram)

```
                         相機軌跡 {(T_i, K_i)}
                                  │
                                  ▼
   ┌──────── 自迴歸生成迴路（每 80 幀觸發一次） ─────────────┐
   │                                                       │
   │  ┌─────────────────────────────────────────┐           │
   │  │  3D Cache 𝒞  =  {(D_i, T_i, K_i, P_i)}   │ ←── 新幀加入  │
   │  │  • 全分辨率深度 D_i                       │           │
   │  │  • 相機 (T,K)_i                          │           │
   │  │  • 下採樣點雲 P_i                        │           │
   │  └────────────────┬────────────────────────┘           │
   │                   │                                    │
   │     [routing]     │  ←── 給 target 視角，從 cache       │
   │                   │     找最相關的 K 個歷史幀           │
   │                   ▼                                    │
   │  ┌─────────────────────────────────────────┐           │
   │  │  Retrieved frames + canonical coord map │           │
   │  │  + warped depth (4 通道)                 │           │
   │  └────────────────┬────────────────────────┘           │
   │                   │                                    │
   │                   ▼                                    │
   │     ┌───────────────────────────────────┐              │
   │     │  Wan 2.1-14B DiT                  │              │
   │     │  inputs: target cam + retrieved   │              │
   │     │          frames + coord map       │              │
   │     │  output: next 80 frames latent    │              │
   │     └────────────────┬──────────────────┘              │
   │                      │                                 │
   │     [self-aug]       │  ← 訓練時 50% 機率把 hist        │
   │                      │     加 t∈(0,0.5) 的噪聲再餵      │
   │                      ▼                                 │
   │              VAE decode → 80 frames                    │
   │                      │                                 │
   │                      ▼                                 │
   │              更新 3D cache（DAv3 估計 depth）           │
   │                                                        │
   └────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                  ┌─────────────────────────────┐
                  │  3DGS lift（DAv3 fine-tuned） │
                  │  + OpenVDB mesh 提取          │
                  └──────────┬──────────────────┘
                             │
                             ▼
                  ┌─────────────────────────────┐
                  │  NVIDIA Isaac Sim            │
                  │  • 機器人導航/碰撞模擬          │
                  │  • mesh + 物理引擎             │
                  └─────────────────────────────┘
```

---

### 1.4 為什麼用 Wan 2.1-14B 而非 NVIDIA 自家 Cosmos？

這是一個值得單獨拆的設計決定——**NVIDIA SIL 的工作竟然不用 NVIDIA Cosmos**。

📜 **論文原話（§3 Preliminaries + Appendix A.1）**：
> *「We adopt the Wan 2.1 VAE [wan2025wan], which downsamples $8 \times 8$ spatially and $4 \times 4$ temporally...」*
>
> *「We build upon the Wan 2.1-14B DiT [wan2025wan] as our backbone video diffusion model.」*
>
> ⭐ *「**We find that within Wan 2.1, this mechanism alone already delivers accurate camera control even along long trajectories.**」*

Cosmos（`agarwal2025cosmos`）在論文中**只在 intro 引用一次**作為「video-to-3D 範式」的脈絡背景，**未當 baseline 也未做對照**。

**論點背後的技術理由**（部分論文明說，部分為合理推論）：

| 理由 | 證據強度 | 說明 |
|------|:------:|------|
| **相機控制成熟度** | ✅ 論文明說 | Wan 2.1 內建的 camera conditioning 在長軌跡下「**alone already delivers accurate camera control**」——意味著作者**實測過**這條 baseline 已夠用，3D cache 是錦上添花 |
| **VAE 規格匹配** | ✅ 論文需要 | 論文方法強依賴 $8 \times 8$ 空間 / $4 \times 4$ 時間下採樣的 VAE；Wan 2.1 VAE 自帶此規格 |
| **方法 backbone-agnostic 主張** | 🟡 推論 | 用非 NVIDIA backbone 證明「3D cache 路由 + self-aug 訓練」**對 backbone 中立**——這是更強的科學論點：換 Cosmos / Sora-style 都應該適用 |
| **開源生態與 License** | 🟡 推論 | Wan 2.1 是 Apache 2.0 全開源權重，社群有大量 finetune 食譜（CameraCtrl-on-Wan 等）；Cosmos 是 NVIDIA 為 physical AI 推出的閉/半閉模型，外部 fine-tune 友善度待檢驗 |
| **時序問題** | 🟡 推論 | Lyra 2 工作 likely 始於 2025 中後段；當時 Cosmos-1 → Cosmos-2 仍在迭代，Wan 2.1 已穩定 |

🧠 **核心啟示**：
NVIDIA 自家 SIL Lab 公開用阿里巴巴的 backbone 不是政治表態，而是**研究誠實**的展示——「我的方法值得被 mainstream backbone 採用，所以我先在最成熟的開源 video DiT 上驗證」。對 VLA / 3D 研究者意味著：**方法論的 reusability 比 backbone 一致性重要**——將來換 backbone 不是 risk，是 feature。

⚠️ **隱含限制**：Wan 2.1 的訓練數據偏向通用視訊（電影 / 動畫風格） + DL3DV finetune（室內房屋）；若日後底層換 Cosmos（physical AI 預訓），對工業 / 戶外場景可能反而更友善——這是論文沒探索的「未來潛力」軸。

---

## 2. 數學核心：3D 緩存 + flow matching (Math Core)

📌 **Napkin Formula**：
```
z_t^next = DiT( z_t , target_cam , {warp(I_j, T_j → T_target, D_j) for j ∈ Retrieved} )
            ↑                          ↑
     當前 noisy latent      用幾何把舊幀投影回當前視角的「假新幀」
```
> 一行直覺：**video model 不直接看歷史像素，它看「幾何投影過的歷史像素」——這樣它無法漂，因為相機+深度釘住了內容。**

### 2.1 3D Cache 維護

```
𝒞 = { (D_i, T_i, K_i, P_i) }_{i=1..N}
P_i ∈ ℝ^{(H/d) × (W/d) × 3}    # downsampled 點雲
D_i ∈ ℝ^{H × W}                # full-res depth
```

> **直覺**：兩個解析度——下採樣點雲做快速檢索（哪些舊幀重疊當前視角），全分辨率深度做精確 warp。

### 2.2 Self-Augmented Training（公式級）

```
with probability p_aug:
    t ~ U(0, 0.5)
    z_t^hist = (1 − t) · z_0^hist  +  t · ε     # flow matching 加噪
    z_pred = DiT_one_step(z_t^hist)            # 一步去噪
    train history  ←  z_pred                    # 用「污染後的近似重建」當歷史
```

> **直覺**：訓練分布 $\approx$ 推理分布。訓練時看到的歷史不只是 ground-truth，還有「自己生的次優結果」——模型學會在這種分布下保持品質。

### 2.3 Coordinate Map 注入（spatial forgetting 解法核心）

對檢索到的第 j 個歷史幀：
```
C_j ∈ [−1, 1]^{3 × H × W}    # canonical coord map
forward_warp(C_j, D_j, T_j → T_target)  →  per-pixel 對應關係
+ warped depth as 4th channel
→ concat with VAE-encoded I_j → 進 DiT
```

> **直覺**：DiT 不需要「記住」這幀在哪——coord map 在每個像素位置寫明了「這個像素在世界座標系裡是哪」。模型按位置取信息，不靠 attention 跨時間搜尋。

---

## 3. 帶數字走一遍：80 幀 / 194 秒 / 一個房間 (Worked Example)

考慮一個 demo trajectory：起點 → 走進客廳 → 繞沙發一圈 → 走回起點。

```
Step 0:  生 frame 0..79（首次進場）
         • 3D cache 從空填到 80 條目
         • DAv3 估深度，加入 cache
         耗時：194s on 1× GB200（full 35-step CFG）

Step 1:  生 frame 80..159（繞沙發左半）
         • 路由查詢：當前視角與 cache 重疊？→ 有，frame 12-25 與當前視角重合
         • 把這些幀 warp 到當前視角作為 retrieved
         • DiT 在 retrieved 條件下生 80 幀
         耗時：194s

Step 2:  生 frame 160..239（繞沙發右半）

Step 3:  生 frame 240..319（走回起點）
         • 路由：與 frame 0..79 高度重疊（同一視角範圍）
         • Retrieved = frame 0..79 的 warped 版本
         • → DiT 生成內容必須和 frame 0..79 視覺一致
         • 沒有 hallucination「房間長變了」的 bug
```

**為什麼回到起點不會「房間變了」**：
- 沒有 cache → 模型重新生 → 生出的「客廳」和原來不一樣（spatial forgetting）
- 有 cache → frame $0..79$ 被 warp 回當前視角 → 模型看到「應該長這樣」的視覺先驗 → 一致

**為什麼漂移被抑制**：
- 沒有 self-aug → 訓練只見過 clean 歷史 → 部署遇到自己生的髒歷史就崩
- 有 self-aug → 訓練時 $50\%$ 機率餵髒歷史 → 模型學會「我生的有偏差，下次要拉回來」

---

## 4. 工程視角：訓練成本、推理時長、部署 (Engineering View)

| 維度 | 數值 / 觀察 | 工程含義 |
|------|------------|---------|
| **訓練 GPU** | **$64 \times$ NVIDIA GB200**（Blackwell 級） | NVIDIA 專屬硬體；外部團隊需 H200/H100 spike |
| **訓練步數** | 7,000 iterations | 不算特別多，因為 base 是預訓 Wan 2.1-14B |
| **訓練 batch** | 64 across 64 GPUs | 大 batch 但短訓 |
| **參數規模** | Wan 2.1-14B 為主 + 小頭 | 14B 級 |
| **單步推理** | **$194\ \text{s} / 80\ \text{frames}$（full $35$-step CFG）** on $1\times$ GB200 | 慢；不是即時 |
| **DMD 加速** | **15 s / 80 frames（4-step, no CFG）** | $13\times$ 加速；質量輕微下降 |
| **訓練數據** | DL3DV 10K real-world clips | 開源資料集 |
| **Pose 估計** | ViPE（off-the-shelf SLAM/SfM） | 訓練前處理 |
| **Depth 估計** | Depth Anything v3 | 預訓 |
| **Caption** | Qwen3-VL-8B-Instruct | 多模態 |
| **License（代碼）** | Apache 2.0 | 商用友善 |
| **License（模型）** | 各版本不同 | ⚠️ 用前需逐個確認 |
| **倉庫** | nv-tlabs/lyra · $1.8$k★ · $174$ fork | 高關注 |
| **HF 權重** | nvidia/Lyra · nvidia/Lyra-2.0 | 開放下載 |

**部署約束**（從 VLA / robotics 視角）：
- **單步 194 秒不是即時**——適合 offline 場景生成，**不適合線上模擬**
- DMD 變體（15s/step）改善但仍非交互速率
- Isaac Sim export 是論文宣稱的**直連點**，但具體 API / 流程**論文未細說**——可能仍需手動 mesh export + 物理屬性標注
- **DL3DV 訓練 → 主要是室內房屋 / 室內景**——戶外、工業、駕駛場景未驗證
- 動態場景（人/物移動）論文明確**不支持**

---

## 5. 數據與評測 (Data & Eval)

### 5.1 Training / Evaluation 數據

| 用途 | 數據 | 規模 |
|------|------|------|
| Training | **DL3DV** | 10K long video clips, real-world 多樣場景 |
| In-domain Eval | DL3DV-Evaluation split | — |
| OOD Eval | **Tanks-and-Temples** | 經典 3D 場景 benchmark |
| Pose 標註 | ViPE | — |
| Depth 標註 | Depth Anything v3 | — |
| Caption | Qwen3-VL-8B-Instruct | — |

### 5.2 Long Video Generation 主表（Tanks-and-Temples OOD）

| 方法 | SSIM ↑ | LPIPS ↓ | FID ↓ | Subjective ↑ | Style Cons. ↑ | Cam Ctrl ↑ | Reproj Err ↓ |
|------|:------:|:-------:|:------:|:------------:|:-------------:|:----------:|:------------:|
| GEN3C | 0.350 | 0.589 | 79.07 | 21.75 | 75.54 | **70.91** | 0.054 |
| CaM | 0.367 | 0.605 | 59.20 | 34.22 | 82.83 | 31.86 | **0.056** |
| **Lyra 2.0** | **0.384** | 0.552 | 51.33 | **43.35** | **85.07** | 63.87 | 0.069 |
| Lyra 2.0 DMD | 0.362 | **0.545** | **49.71** | 43.02 | 78.91 | 58.12 | 0.077 |

**讀法**：
- **本文在 6/8 指標領先**（SSIM / LPIPS / FID / Subjective / Style / —）
- **Camera Control 落後 GEN3C**（63.87 vs 70.91）—— 因為 GEN3C 直接靠點雲 condition，相機控制更硬
- **Reprojection Error 略高於 GEN3C/CaM** — 模型把「外觀生成」優先級放高於嚴格幾何

### 5.3 3D Scene Generation（Tanks-and-Temples）

| 方法 | LPIPS-P ↓ | LPIPS-G ↓ | FID ↓ | Subjective ↑ |
|------|:--------:|:---------:|:-----:|:------------:|
| SPMem | 0.412 | 0.666 | 94.11 | 9.95 |
| Lyra 2.0 + DAv3 | 0.409 | 0.648 | 79.36 | 14.42 |
| **Lyra 2.0 Full** | **0.372** | **0.629** | **72.47** | **18.80** |

⚠️ **Baseline 缺口**：論文**沒做** CAT3D / GenWarp / ReconX / WonderJourney / WonderWorld 對比——這些是該領域的標準同行。**對比池僅 GEN3C / CaM / SPMem 三個**——讀者要意識到這個 baseline 範圍。

### 5.4 機器人 / VLA 應用聲稱

論文原話：「The 3D Gaussian Splatting representations and meshes generated by our pipeline can be directly exported to physics engines... importing our reconstructed scenes into NVIDIA Isaac Sim, enabling physically grounded robot navigation and interaction」

⚠️ **但**：
- **僅展示「機器人在 Lyra 場景裡走」**，沒有 VLA training data 應用實證
- 沒測「用 Lyra 場景訓練 VLA → 真機表現」
- 「Sim2Real gap 是否變小」這個 VLA 工程師最關心的問題——**論文沒答**

---

## 6. 能力與失敗模式 (Capabilities & Failure Modes)

### 6.1 能做什麼
- 長軌跡（>4 步 80 幀拼接）3D-consistent 視訊
- 視角回環時不 hallucinate
- 生成可 export 到 Isaac Sim 的 3DGS + mesh
- DMD 變體 15s/step（仍非即時，但能用）
- Apache 2.0 + 1.8k stars 開源生態

### 6.2 失敗模式（含論文自承）

| 場景 | 為什麼失敗 | 來源 |
|------|----------|------|
| 動態場景（人/物移動） | 框架假設靜態環境 | 論文 Limitations 自承 |
| 室外 / 駕駛 / 工業 | DL3DV 主要是室內 | 訓練數據限制 |
| 曝光不一致場景 | DL3DV 本身有曝光變化，模型會複製這個 artifact | 論文自承 |
| 即時交互（>1 fps） | full 模型 194s/step；DMD 15s/step | 推理時長 |
| 細結構（薄物、線狀） | 8×8 VAE downsample 限制 | 隱含限制 |

### 6.3 隱含假設 (Hidden Assumptions)

1. **3D cache 內存可承受**：長軌跡點雲線性累積，10000+ 幀的場景是否仍 OK？論文 demo 範圍是 4 步（320 幀），更長未驗證
2. **DAv3 的深度估計足夠準**：cache 的幾何質量上限被 DAv3 釘死——若深度有 outlier，warp 過去的 retrieved frame 會錯位
3. **camera trajectory 是用戶輸入的**：論文有 GUI，但 demo 以人為設計軌跡為主——機器人自主探索（VLA 場景）下相機軌跡品質未測
4. **室內 + 房屋幾何先驗**：Wan 2.1 預訓 + DL3DV fine-tune → 隱式假設「世界看起來像房子」。生成湖泊、隧道、洞穴等可能崩
5. **Self-aug 的 p_aug 是超參**：論文未公開最優值

---

## 7. 與相關工作對比 (Comparison)

| 方法 | 長度 | 相機控制 | 回頭一致性 | 3D 輸出 | Sim 直連 |
|------|:----:|:--------:|:---------:|:-------:|:--------:|
| Stable Video Diffusion | 短 | ✗ | — | ✗ | ✗ |
| CAT3D | 短 | ✓ | ✗（單次） | ✓ NeRF | ✗ |
| GenWarp | 短 | ✓ | 弱 | ✓ | ✗ |
| ReconX | 短-中 | ✓ | 弱 | ✓ NeRF | ✗ |
| WonderJourney | 長（拼接） | ✓ | 弱（每段獨立） | ✓ | 部分 |
| GEN3C | 中 | ✓✓ | 中 | ✓ | ✗ |
| CaM | 中 | ✓ | 中 | ✓ | ✗ |
| **Lyra 2.0**（本文） | **長 + 可回環** | ✓ | **強** | **✓ 3DGS+mesh** | **✓ Isaac Sim** |

### 🎤 面試 Tip

> **被問「video-to-3D 怎麼解決長軌跡 drift？」** ——
> 三句話答：(1) 不要讓 video model 自己記憶——記憶會 OOM 或 drift。(2) 顯式維護「3D 緩存」（depth + camera + 下採樣點雲），回頭時用幾何把舊幀投影回當前視角，**讓 video model 只看「幾何 warp 過的歷史」**。(3) 訓練時故意把模型自己的壞輸出當歷史餵回去，分布匹配 → drift 被抵消。但要老實補：**只在靜態室內場景驗證；對比 baseline 池只有 GEN3C/CaM/SPMem 三個**——CAT3D/WonderWorld 等同行未對比。

---

## 8. 給 VLA 工程師的「工具」視角

把 Lyra 2.0 當工具用，VLA 工作中的 4 個落地點：

### 8.1 場景生成（offline）
- 給 Isaac Sim 造訓練場景的廉價方案
- ⚠️ 目前只室內、靜態——工業/室外/動態任務不行
- 推理 $194\,\text{s}/\text{step}$ 太慢 → 用 DMD 變體 $15\,\text{s}/\text{step}$ 可接受

### 8.2 真機 sim2real 數據增強（推測，論文未驗證）
- 從真機相機重建 → Lyra 重生視訊 → 訓 VLA
- ⚠️ 沒有實證「Lyra 場景訓的 VLA 真機表現如何」——這是最大的開放問題

### 8.3 demo / 視覺化
- 給 BD / PM 看「機器人未來在某種環境下能做什麼」
- 配 Foxglove + Isaac Sim 變成可互動 demo

### 8.4 不該用的場景
- 即時規劃 / 即時模擬（推理太慢）
- 戶外 / 工業 / 駕駛場景（訓練數據不覆蓋）
- 動態多人場景（明確不支援）
- 強物理（碰撞、軟體、流體）—— Lyra 給幾何，物理仍靠 Isaac Sim

---

## 9. 待追問的開放問題

1. **Isaac Sim 直連 API 細節**？論文聲稱「直接 export」，但具體流程（mesh 後處理、collision 標注、material assign）**論文未細說**——這是 VLA 工程師最關心的工程點。
2. **VLA 訓練數據應用實證**？論文示範「機器人在 Lyra 場景裡走」，但沒測「用 Lyra 訓的 VLA 在真機表現如何」——sim2real gap 是否被改善？
3. **長軌跡上限**？論文 demo 是 4 step（320 frames）。10 step（800 frames）/ 50 step 是否仍 consistent？3D cache 內存增長曲線？
4. **與 CAT3D / WonderWorld / ReconX 對比**？這些是 video-to-3D 的標準同行，論文 baseline 池缺它們。
5. **DMD 變體的 quality drop 範圍**？$15\,\text{s}/\text{step}$ 比 $194\,\text{s}$ 快 $13\times$，質量下降在哪個指標上？
6. **動態擴展可行性**？作者列為 future work，但沒給技術 roadmap。
7. **Camera trajectory 是否能由 VLA policy 自主驅動**？論文 demo 是人手設計軌跡——機器人探索式軌跡（不規則、突變）下表現？
8. **`research.nvidia.com` 與 GitHub 的差異**？1.8k stars repo 含 Lyra 1.0 + 2.0 兩版本——VLA 工程師應拿哪個？文檔對「替換」vs「並用」沒明說。

📎 **內容類型可信度**：

| 來源 | 可信度 | 對應內容 |
|------|--------|---------|
| arXiv 論文 v1 | 🟡 中高 | 方法、Tables 1-2、訓練配置 |
| 項目主頁 | 🟡 中高 | demo 視訊、Isaac Sim 集成宣稱 |
| GitHub README | 🟡 中 | 代碼結構、license |
| 通用宣傳 | — | 本文無營銷稿來源 |

---

📎 **來源**：
- arXiv: https://arxiv.org/abs/2604.13036（Lyra 2.0, 2026-04-14）
- HTML: https://arxiv.org/html/2604.13036v1
- 項目主頁: https://research.nvidia.com/labs/sil/projects/lyra2/
- 代碼: https://github.com/nv-tlabs/lyra（Apache 2.0, 1.8k★, 174 forks）
- 權重: https://huggingface.co/nvidia/Lyra-2.0
- 對比基線: GEN3C / CaM / SPMem（論文範圍）+ CAT3D / GenWarp / ReconX / WonderJourney / WonderWorld（領域標準同行，論文未對比）
- 上游依賴: Wan 2.1-14B / Depth Anything v3 / ViPE / Qwen3-VL-8B-Instruct / OpenVDB

🧠 **本文判讀（作者觀點）**：
這篇是 NVIDIA SIL 把「video-to-3D」這條 generative 3D 路線推向「**可長走 + 可回頭 + 可進物理引擎**」的工程化里程碑——**對 VLA 工程師的長期意義**是：未來「給機器人造訓練場景」可能從「Isaac 腳本 + 3D Artist」變成「給軌跡，AI 生」。但**短期還不能用**——推理 194s/step、室內 only、動態不支援、且 sim2real gap 改善缺實證。

**評級 🔧 可操作（潛在 ⚡）**：開源完整、權重開放、對 spatial forgetting + temporal drifting 兩個老問題的處方有工程 reusability。卡在 ⚡ 的關鍵：(a) baseline 池缺 CAT3D / WonderWorld 級同行；(b) 「Isaac Sim 直連」聲稱缺實證細節；(c) 對 VLA 工程師最關心的「sim2real 改善」沒實驗——若 3-6 個月內這三點補上可升級。

---

[← Back to World Model](./README.md) · [← Back to Theory](../README.md)
