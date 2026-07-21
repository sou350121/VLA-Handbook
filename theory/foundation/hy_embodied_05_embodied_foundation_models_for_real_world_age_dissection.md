# HY-Embodied-0.5：具身基础模型实战解析 (HY-Embodied-0.5: Embodied Foundation Models for Real-World Agents)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-04-13
>
> **论文**: HY-Embodied-0.5: Embodied Foundation Models for Real-World Agents
> **链接**: https://arxiv.org/abs/2604.07430
> **核心定位**: 腾讯推出具身专用 VLM 系列，用 MoT 架构 + 视觉 latent tokens + 迭代式后训练，在 2B 小模型上实现超越同尺寸 SOTA 的具身感知与推理能力

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | MoT-2B 在 22 個具身基準上 16 個 SOTA，平均 58.0%；MoE-32B 平均 67.0% 超越 Gemini 3.0 Pro (63.6%) |
| 適合精讀 | 如果你在做 edge 部署的具身 Agent、需要高頻視覺感知 + 推理、或關注 MoT 架構設計 |
| 可以跳過 | 如果你只關心純語言 Agent 或不涉及實時機器人控制的場景 |
| 落地可行性 | 高（2B 版本已開源 HuggingFace，推理代碼完備，支持 Transformers 直接加載） |
| 主要風險 | 實作依賴特定 Transformers 版本（需安裝 git 倉庫特定 commit），vLLM 推理尚未支持 |

💡 **X-Ray 開場**  
這篇論文解決什麼問題？通用 VLM 在具身場景下視覺感知粒度不足、缺乏動作導向的推理能力。  
發現了什麼？用 Mixture-of-Transformers 架構分離視覺/語言計算路徑，加上視覺 latent tokens 和迭代式 RL+RFT 後訓練，能在 2B 參數量下實現超越 4B-7B 模型的具身性能。  
對 VLA 研究者意味著什麼？提供了一個已驗證的 edge-ready 基礎模型，可直接作為 VLA 的「大腦」，且開源權重和推理代碼完備。

📍 **研究全景時間線**
```
[2023] LLaVA / Qwen-VL 通用 VLM 興起 → [2024] RoboBrain / π0 具身專用模型 → [2025] MiMo-Embodied 7B
                                         ↓
                                    [本文 HY-Embodied-0.5] ← 首個 2B 具身模型超越 4B-7B 同儕
                                         ↓
                            局限：實作依賴特定 Transformers 版本，vLLM 尚未支持
```

## 1. 核心架構/方法總覽 (Overview / Architecture)

### 1.1 系統對比概覽 (System Component Comparison)

| 組件 | MoT-2B (Edge) | MoE-32B (Complex) | 設計目的 |
|------|---------------|-------------------|----------|
| 總參數量 | 4B | 407B | - |
| 激活參數 | 2.2B | 32B | 推理時實際計算量 |
| 視覺編碼器 | HY-ViT 2.0 (400M) | 同左 | 原生分辨率支持，任意尺寸輸入 |
| 架構類型 | Mixture-of-Transformers | Mixture-of-Experts | 模態自適應計算 |
| 視覺路徑 | 獨立 QKV + FFN + 雙向 Attention | 同左 | 避免視覺訓練污染語言能力 |
| Latent Tokens | 有（每圖 1 個） | 同左 | 連接視覺與語言語義 |
| 推理速度 | $\approx$ Dense-2B | - | MoT 引入開銷可忽略 |

### 1.2 關鍵機制 (Key Mechanism)

**MoT 架構核心設計**：
- 在預訓練 LLM 基礎上複製 FFN 和 QKV 參數，初始化為原 LLM 權重
- 視覺 token 用複製的參數計算，文本 token 用原始參數計算
- 視覺分支用雙向 Attention（視覺數據無單向性），語言分支用因果 Attention
- 視覺 next-code prediction 任務：用更大 ViT 生成的离散 visual code 監督視覺分支輸出

**視覺 Latent Tokens**：
- 每個視覺元素（圖像/視頻幀）末尾附加 1 個可學習的 latent token
- 預訓練階段用大 ViT 的全局 CLS 特徵監督該 token 輸出
- 作用：提取細粒度語義視覺特徵並與語言概念對齊

⚡ **Eureka Moment**：**模態分離計算 + 視覺 latent tokens** 讓小模型在視覺訓練時不犧牲語言能力，同時通過 latent token 建立跨模態語義橋樑——這是 2B 模型能超越 4B-7B 的關鍵。

### 1.3 信息流/架構圖 (Flow / Diagram)

```
┌─────────────────────────────────────────────────────────────────┐
│                        HY-Embodied-0.5                          │
├─────────────────────────────────────────────────────────────────┤
│  輸入圖像 (任意分辨率)                                           │
│         ↓                                                        │
│  ┌─────────────────┐                                            │
│  │   HY-ViT 2.0    │  (400M, 原生分辨率，蒸馏自更大 ViT)          │
│  │  視覺編碼器      │                                            │
│  └────────┬────────┘                                            │
│           ↓ 視覺 token 序列                                       │
│  ┌────────────────────────────────────────────────────────┐     │
│  │           Mixture-of-Transformers (MoT)                │     │
│  │  ┌──────────────┐    ┌──────────────┐                 │     │
│  │  │ 視覺分支      │    │ 語言分支      │                 │     │
│  │  │ 獨立 QKV/FFN │    │ 原始 LLM      │                 │     │
│  │  │ 雙向 Attention│    │ 因果 Attention│                 │     │
│  │  │ Vision Loss  │    │ LLM Loss     │                 │     │
│  │  └──────┬───────┘    └──────┬───────┘                 │     │
│  │         └──────────┬─────────┘                         │     │
│  │                    ↓                                   │     │
│  │         ┌──────────────────┐                           │     │
│  │         │ Visual Latent    │ ← Global Loss 監督         │     │
│  │         │ Token (每圖 1 個)  │   (與教師 ViT CLS 對齊)      │     │
│  │         └──────────────────┘                           │     │
│  └────────────────────────────────────────────────────────┘     │
│                    ↓                                             │
│         多模態融合表示 → LLM 解碼生成                             │
└─────────────────────────────────────────────────────────────────┘
```

## 2. 數學核心 (Math Core)

📌 **Napkin Formula**（一行抓住本質）：
```
L_total = L_llm + L_vision + L_global
```

**目標**：聯合優化語言生成、視覺感知、跨模態對齊三項能力。

**公式拆解**：

```
(1) 視覺 Loss (Visual Next-Code Prediction):
    L_vision = -1/N_v · Σ_{i=1}^{N_v} log p_i(z_i)
    
    N_v = 視覺 token 數量
    p_i = 第 i 個 token 的預測概率分佈
    z_i = 教師 ViT 生成的目標离散 code

(2) 全局 Loss (Latent Token 對齊):
    L_global = -(f_latent^T · f_teacher) / (||f_latent|| · ||f_teacher||)
    
    f_latent = latent token 的映射隱藏狀態
    f_teacher = 教師 ViT 的全局 CLS 特徵
    → 負餘弦相似度，最大化對齊

(3) 總 Loss (預訓練階段):
    L_total = L_llm + L_vision + L_global
    
    中訓練及微調階段：僅用 L_llm（自回歸語言 Loss）
```

**變量說明**：

| 符號 | 含義 | 來源 |
|------|------|------|
| N_v | 視覺 token 數量 | 由輸入圖像分辨率和 ViT patch size 決定 |
| z_i | 目標离散 visual code | 由更大 ViT 教師模型生成，codebook size=$2k$，每 $8\times8$ patch 壓縮為 1 個 code |
| f_latent | latent token 特徵 | 模型內部可學習 |
| f_teacher | 教師 ViT 全局特徵 | 預計算，固定監督信號 |

**直覺**：視覺 Loss 讓模型學會「預測下一個視覺 patch 是什麼」，類似語言模型的 next-token prediction；全局 Loss 強迫 latent token 吸收整圖語義，成為視覺 - 語言的橋樑。

> 符號與本文/相關文檔保持一致：L_vision 對應論文 Section 3.3 的 vision loss，L_global 對應 global loss。

## 3. 帶數字走一遍：玩具例子 (Worked Example)

假設輸入一張 $512\times512$ 的廚房場景圖像，任務是「找出橙子並給出抓取點」。

**步驟 1：視覺編碼**
```
輸入圖像: 512×512
HY-ViT 2.0 patch size: 14×14 (典型 ViT 設置)
視覺 token 數量: (512/14)² ≈ 1344 個 token
+ 1 個 latent token
= 1345 個視覺 token 輸入 MoT
```

**步驟 2：MoT 前向傳播**
```
視覺 token (1344 個) → 視覺分支 QKV/FFN → 雙向 Attention → 視覺表示
文本 token ("找出橙子並給出抓取點") → 語言分支 QKV/FFN → 因果 Attention → 語言表示
Latent token → 吸收全局語義 → 注入語言分支
```

**步驟 3：視覺 Loss 計算（預訓練階段）**
```
假設 N_v = 1344
教師 ViT 為每個 patch 生成离散 code z_i ∈ {0, 1, ..., 1999} (codebook size=2k)
模型預測每個 patch 的 code 概率分佈 p_i

L_vision = -1/1344 · Σ_{i=1}^{1344} log p_i(z_i)

若模型對某個 patch 預測 p_i(z_i) = 0.8，則該 patch 貢獻 -log(0.8) ≈ 0.22 到 Loss
若預測 p_i(z_i) = 0.1，則貢獻 -log(0.1) ≈ 2.30
→ 鼓勵模型對正確 code 給出高置信度
```

**步驟 4：推理輸出**
```
模型生成 Chain-of-Thought:
<think>
1. 檢測圖像中的水果：發現 3 個橙子，坐標分別為 (120, 340), (280, 350), (450, 360)
2. 評估可抓取性：中間橙子 (280, 350) 無遮擋，抓取點應在橙子頂部偏右
3. 生成抓取坐標：(295, 335)
</think>
最終答案：抓取點坐標為 (295, 335)
```

**步驟 5：Grounding 評價（RL 階段）**
```
若 ground truth 抓取點為 (290, 340)
預測點 (295, 335) 與 GT 的歐式距離 = √((295-290)² + (335-340)²) = √50 ≈ 7.07 像素
歸一化距離 (假設圖像 512×512) = 7.07 / 512 ≈ 0.014
Reward = 1 - 0.014 = 0.986 (接近完美)
```

## 4. 工程視角 (Engineering View)

| 指標 | MoT-2B | 含義 |
|------|--------|------|
| 激活參數 | 2.2B | 推理時實際載入 GPU 的參數量 |
| 總參數 | 4B | 磁盤存儲量（約 8GB BF16） |
| 視覺編碼器 | 400M | HY-ViT 2.0 參數量 |
| 輸入分辨率 | 任意 | 原生支持，無需預處理縮放 |
| 上下文長度 | 32k | 預訓練/中訓練階段 sequence packing |
| 推理速度 | $\approx$ Dense-2B | MoT 引入開銷可忽略（解碼階段主導總時間） |
| 顯存需求 | $\geq16$GB VRAM | 官方推薦，實際 BF16 推理約 10-12GB |
| 推理框架 | Transformers | 需安裝特定 commit (9293856)，vLLM 尚未支持 |

**部署約束**：
- **依賴特定 Transformers 版本**：`pip install git+https://github.com/huggingface/transformers@9293856c419762ebf98fbe2bd9440f9ce7069f1a`
- **推理溫度設置**：官方示例用 temperature=0.8，thinking mode 可選
- **批處理支持**：支持 left-padding 批處理，適合多請求場景
- **Edge 部署**：2B 版本專為 edge 設計，但需 16GB+ RAM（官方建議）

**Trade-off 分析**：
- MoT 架構雙倍參數（4B vs 2B）但激活參數不變 → 訓練時表達力提升，推理時速度不變
- 視覺 latent token 僅 1 個/圖 → 開銷極小，但需額外的 global loss 監督
- 雙向 Attention 用於視覺分支 → 更適合視覺建模，但需獨立實現 attention mask

## 5. 數據與評測 (Data & Eval)

### 5.1 數據組成

**預訓練數據（600B+ tokens）**：
| 類型 | 數據量 | 來源 |
|------|--------|------|
| 通用理解 | 389B tokens | 內部 VLM 數據（caption、STEM、文檔、多輪對話等） |
| 具身 + 感知 | 236B tokens | 空間/機器人數據 43% + 視覺感知數據 57% |

**視覺感知數據（細分）**：
| 任務 | 數據量 | 來源 |
|------|--------|------|
| Omni-Detection (2D/3D) | 62M | OpenImages, Objects365, RefCOCO, SA-1B + VLM 自動標注 |
| Depth Estimation | 36M | 室內/室外 3D 數據集 + 自動駕駛語料 |
| Segmentation | 5M | SA-1B 高質量分割掩碼 |
| Pointing & Counting | 11M | Pixmo-Points + 高密度場景篩選 |

**具身數據（中訓練階段，12M+ QA pairs）**：
- Grounding: Molmo, RoboPoint, RefSpatial + 內部標注
- Affordance: RoboAfford, ShareRobot + VLM 生成指令
- Trajectory: MolmoAct, ShareRobot, FSD + CoTracker3 追蹤提取
- Understanding: Robo2VLM, RoboVQA, RoboRefit, RoboInter-VQA
- Planning: 機器人操作視頻 VLM 標注 + RoboVQA/RoboInter
- Reasoning: 內部構建的長程推理數據集

**空間數據**：
- Correspondence, Geometry, Configuration, Measurement, Dynamics
- 來源：ScanNet, ScanNet++, ARKitScenes + 自採集數據

### 5.2 評測基準（22 個）

**視覺感知（2 個）**：CV-Bench, DA-2K

**具身理解（8 個）**：ERQA, EmbSpatial-Bench, RoboBench-MCQ, RoboBench-Planning, RoboSpatial-Home, ShareRobot-Affordance, ShareRobot-Trajectory, Ego-Plan2

**空間理解（12 個）**：3DSRBench, All-Angles Bench, MindCube, MMSI-Bench, RefSpatial-Bench, SAT, SIBench-mini, SITE-Bench-Image, SITE-Bench-Video, ViewSpatial, VSIBench, Where2Place

### 5.3 主要結果

**MoT-2B vs 同尺寸模型**：
| 模型 | 激活參數 | 22 基準平均分 | Best/Second |
|------|----------|---------------|-------------|
| HY-Embodied-0.5 MoT-2B | 2.2B | **58.0%** | **16/4** |
| Qwen3-VL-4B | 4B | 47.8% | - |
| RoboBrain2.5-4B | 4B | 49.4% | - |
| MiMo-Embodied-7B | 7B | - | - |

**MoE-32B vs Frontier 模型**：
| 模型 | 平均分 | 對比 |
|------|--------|------|
| HY-Embodied-0.5 MoE-32B | **67.0%** | - |
| Gemini 3.0 Pro | 63.6% | -3.4 |
| Seed 2.0 | 66.2% | -0.8 |
| Qwen 3.5 A17B | 66.1% | -0.9 |
| Kimi K2.5 | 61.1% | -5.9 |

**機器人控制實測（20 次試驗成功率）**：
| 任務 | HY-Embodied VLA | $\pi_0$ | $\pi_{0.5}$ |
|------|-----------------|----|----|
| Precision Plug-in Packing | **85%** | 80% | 85% |
| Tableware Stacking | **80%** | 60% | 85% |
| Mug Hanging | **75%** | 45% | 50% |

> 來源：論文 Table 1, Table 2, Figure 13

## 6. 能力與失敗模式 (Capabilities & Failure Modes)

### 6.1 能做什麼

| 能力 | 場景 | 證據 |
|------|------|------|
| 細粒度視覺感知 | 深度估計、物體檢測、計數 | CV-Bench 89.2, DA-2K 92.3 |
| 空間推理 | 3D 關係理解、多視角匹配 | 3DSRBench 57.0, MindCube 66.3 |
| 具身 Grounding | 抓取點定位、bounding box 預測 | RoboSpatial-Home 55.7, RefSpatial 45.8 |
| 動作規劃 | 多步任務分解、軌跡預測 | RoboBench-Planning 54.2, ShareRobot-Traj 73.3 |
| 長程推理 | 複雜場景分析、自我修正 | CoT 可視化顯示 "Wait, no..." 自我糾正 |

### 6.2 不能做什麼 / 局限

| 限制 | 原因 | 影響 |
|------|------|------|
| 依賴特定 Transformers 版本 | MoT 架構未合併到主幹 | 部署需安裝 git commit，無法直接用 pip 版本 |
| vLLM 不支持 | 架構特殊，需適配 | 高吞吐推理需等待官方更新 |
| 實測僅限雙臂 Xtrainer | 機器人實驗平台單一 | 遷移到其他機器人需重新 SFT |
| 部分基準落後 | ShareRobot-Affordance 26.8% (RoboBrain 25.5%) | Affordance 預測仍待改進 |
| 思維模式重複 | 小模型在某些基準產生重複 thinking | 官方註釋 Qwen3.5-VL 有此問題，HY-Embodied 未明確說明 |

### 6.3 隱含假設 (Hidden Assumptions)

- **假設 1：視覺 - 語言模態分離計算總是有益**  
  論文未驗證在極小模型（<1B）或超大模型（>100B）下 MoT 是否仍有收益

- **假設 2：Latent Token 數量 1 個足夠**  
  未探索多個 latent tokens 對複雜場景的影響

- **假設 3：RL 獎勵函數設計覆蓋所有具身能力**  
  獎勵函數分為 Grounding/Regression/Trajectory/Textual 四類，但未驗證邊界案例（如多模態混合輸出）

- **假設 4：5K 小時 UMI 數據足以學習通用表示**  
  機器人控制實驗先用 5K 小時 UMI 預訓練，再 SFT 300-700 episodes，但未驗證 UMI 數據量對最終性能的影響

## 7. 與相關工作對比 (Comparison)

| 模型 | 架構 | 參數量 | 具身專用 | 開源 | 關鍵差異 |
|------|------|--------|----------|------|----------|
| **HY-Embodied-0.5** | MoT + Latent Tokens | 2B/32B | ✅ | ✅ | 模態分離計算，視覺 latent tokens |
| Qwen3-VL | Dense/MoE | 2B-72B | ❌ | ✅ | 通用 VLM，非具身優化 |
| RoboBrain 2.5 | - | 4B | ✅ | ❓ | 專用具身，但性能落後 |
| $\pi_0 / \pi_{0.5}$ | Action Expert | - | ✅ | ✅ | VLA 框架，非基礎模型 |
| MiMo-Embodied | - | 7B | ✅ | ❓ | 小米具身模型，性能居中 |

**面試 Tip**：  
被問到「小模型如何做具身任務」時，可以回答：「HY-Embodied-0.5 用 MoT 架構分離視覺和語言計算路徑，避免視覺訓練污染語言能力，同時用視覺 latent tokens 建立跨模態語義橋樑——這讓 2B 模型在 22 個具身基準上 16 個 SOTA，超越 4B-7B 模型。」

## 8. 精讀建議 (Reading Guide)

### 值得精讀原文的人

1. **做多模態具身 Agent 的研究者**：特別是關注 edge 部署、實時響應的場景
2. **要評估遷移到新機器人平台可行性的工程師**：論文 Section 6 提供 VLA 整合實戰細節
3. **對 MoT/MoE 架構感興趣的模型架構師**：MoT 在 VLM 中的應用是較新的探索

### 建議章節路徑

```
先讀 §1 Introduction → 再看 §2 Model Architecture → §4 Post-training → §5 Evaluation → 可跳 §3 Pre-training（數據細節較瑣碎）→ 最後看 §6 Robot Control（若關心實戰）
```

**理由**：
- §1 快速理解問題定義和核心貢獻
- §2 是架構核心，MoT 和 latent tokens 設計在此
- §4 後訓練策略（RL+RFT+ 蒸餾）是性能關鍵
- §5 驗證效果，表格密集但信息量大
- §3 數據細節可選讀，除非你打算復現預訓練
- §6 僅在需要整合 VLA 時精讀

### 不值得精讀的理由

- 如果你不做機器人學習或具身 Agent，讀摘要 + §1 即可
- 如果你已熟悉 MoE/MoT 架構，§2 可快速瀏覽
- 如果你只關心開源模型使用，直接看 GitHub README 和推理示例

---

## 🔗 關鍵引用

- **論文**: https://arxiv.org/abs/2604.07430
- **GitHub**: https://github.com/Tencent-Hunyuan/HY-Embodied
- **HuggingFace**: https://huggingface.co/tencent/HY-Embodied-0.5
- **技術報告 PDF**: https://github.com/Tencent-Hunyuan/HY-Embodied/blob/master/hy_embodied_tech_report.pdf

---
[← Back to Theory](./README.md)
