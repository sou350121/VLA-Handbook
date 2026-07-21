# RAM：給 VLM 外掛一本可檢索的「三維物體知識庫」 (Retrieval-Augmented Manipulation: VLM Spatial Awareness for Object-Centric Robot Manipulation)

> **發布時間**：2026 年（Science Robotics, Vol. 11, Issue 113, eaea2092）
> **論文題目**：A retrieval-augmented framework enabling VLM spatial awareness for object-centric robot manipulation
> **團隊**：陳凱（CUHK postdoc, 第一作者）· Li C, Tu C, et al.（香港中文大學）
> **核心定位**：不微調 VLM、不重訓 3D 模型，而是把「**類別級三維物體模板**（標準姿態 + 抓取點 + 功能平面）」做成**可檢索的外部記憶**——任務規劃前先 retrieve 對應類別的空間先驗注入 VLM，讓 GPT/Gemini 級語義規劃器**「知道在三維空間中怎麼做」**而不只是「知道要做什麼」。
>
> ⚠️ 核心定位摘自 DeepTech 訪談 + 期刊摘要；未獨立取得 PDF 原文（[Science Robotics 全文需訂閱](https://www.science.org/doi/10.1126/scirobotics.aea2092)）。

長期以來 VLA 領域對 VLM 的批評是「**懂語義不懂三維**」——能拆解「整理桌面」成步驟，但**不知道杯子的把手在哪、餐盤的功能平面朝哪、抽屜要往哪滑**。三條既有路徑都不夠：(a) 純 2D 訓練的 VLM 缺空間先驗；(b) 用 3D 數據微調 VLM 成本高、訓練不穩；(c) 端到端 VLA 又把空間能力埋在權重裡黑盒化。RAM 走第四條路：**把空間知識顯式化、外部化、可查詢**。

**X-Ray 開場（非專家可複述）**：
這篇論文要解決的是「機器人怎麼『看懂』三維世界」。傳統做法：要嘛重訓 VLM 加 3D 數據（貴），要嘛端到端訓 VLA 模型（黑盒）。本文不訓任何模型——它建一個**標準化的物體模板庫**（每個類別一張「說明書」記錄標準姿態、抓取點、功能平面在哪），任務時根據鏡頭看到什麼**檢索**對應模板，把「標準說明書」對齊到當前實際物體上，再把這些三維信息**組成文字塞進 VLM 的 context**。VLM 拿到後規劃就不再是「拿杯子」，而是「用 grasp_point_3 抓杯子的把手，將底部對齊托盤」。對 VLA 研究者意味著：**RAG 範式從 NLP 遷移到 manipulation 是可行的**——預訓 VLM 不夠的部分用「外部 3D 知識庫」補，比硬訓更務實。

---

## 📍 研究全景時間線

```
2023 ─ Code-as-Policies / VoxPoser ───► VLM 直出代碼控制動作
                                         │  痛點：粗粒度，缺顯式空間
                                         │
2024 ─ ReKep / VoxPoser 2 ────────────► 約束式 VLM 規劃（關鍵點 / voxel）
                                         │  進步：顯式空間，但 per-task 設計
                                         │
2024-25 ─ π₀.₅ / Hi-Robot ───────────► VLM-VLA 雙層架構
                                         │  痛點：VLM 仍每步推理，無持久知識
                                         │
2025 ─ CodeGraphVLP ─────────────────► 顯式語義圖 + Code-as-Planner
                                         │  創新：persistent state，但 graph 是 task-specific
                                         │
2026 ─ RAM（本文） ★
                                         │  • 類別級模板庫（11 類起步）
                                         │  • 3D 視覺 grounding（合成→真實泛化）
                                         │  • RAG 注入 VLM context
                                         │  • 不訓 VLM、不訓 3D 模型
                                         │
未來 ─  • 物體類別擴展到開放世界
        • 模板從靜態幾何擴到材質/受力/失敗模式
        • 執行時持續校正（執行階段也用 RAG）
```

**本文在演進中的位置**：把 RAG（Retrieval-Augmented Generation，NLP 領域已成範式）**遷移到 manipulation**——是少見的「**架構橋接**」貢獻而非「更大 VLM」。Science Robotics 收錄反映出社群對這條路徑的認可。

---

## 1. 核心架構/方法總覽 (Overview / Architecture)

### 1.1 三個模組對比 (Three Module Comparison)

> ⚠️ 下表細節合自 DeepTech 訪談；數學形式為作者推論，等待原文確認。

| 模組 | 名稱（中/英） | 輸入 | 輸出 | 是否需訓練 |
|------|--------------|------|------|:--------:|
| **M1** | 物體類別級知識引擎 / Object Category-level Knowledge Engine | 預先建構的類別模板（11 類） | 類別模板（標準姿態、尺寸、對稱性、抓取點、功能平面） | ❌（人工標註模板，類別級而非實例級） |
| **M2** | 三維視覺接地模型 / 3D Visual Grounding | RGB image + 點雲 + 模板 | 模板$\leftrightarrow$當前物體實例的姿態對應、grasp point 投影 | ✅ 主要用**合成數據**訓 |
| **M3** | 檢索增強任務規劃器 / Retrieval-Augmented Task Planner | M2 grounding 結果 + 原始指令 + 圖像 | VLM 增強 context $\to$ 細粒度動作約束 $\to$ 軌跡優化 | ❌（用既有 VLM，不微調） |

### 1.2 關鍵機制 (Key Mechanism)

⚡ **Eureka Moment**：
> **「不要再爭論『要不要 3D 微調 VLM』——把 3D 知識做成『可檢索』的外部資料庫，讓 VLM 在規劃前 retrieve 它需要的空間先驗。VLM 的角色是『推理器』，不是『3D 編碼器』。」**

四個設計選擇的因果鏈：

1. **為何不訓 VLM？** —— 高品質 3D 數據昂貴；訓完仍無法保證泛化到新物體類別。**外部知識庫換新物體只需新增模板，不重訓**。
2. **為何「類別級」而非「實例級」模板？** —— 為每個杯子單獨建模成本爆炸；「類別級 + 視覺 grounding」可以**一個模板適用同類所有實例**——直接降低真實數據採集量。
3. **為何 M2 用合成數據訓？** —— Sim2real 在 grounding 任務上比在 manipulation 任務上**容易得多**（前者只需「物體點雲對齊」，後者要解決物理 dynamics）。**這是關鍵的可行性論點**。
4. **為何 RAG 而非 fine-tune？** —— RAG 在 NLP 已成範式，原因相同：知識更新只需更新庫，不需重訓基模。**作者把這條 NLP 工程經驗搬到 manipulation。**

### 1.3 信息流 (Flow Diagram)

```
   ┌──────────── 預先建構（離線） ────────────┐
   │  人工為 11 類物體標註標準模板           │
   │  • 標準姿態 / 尺寸 / 對稱性               │
   │  • 抓取點（grasp_point_1..N）            │
   │  • 功能平面（functional_plane_1..M）     │
   │  • （可選）鉸接軸 / 形變狀態模板          │
   │  → Knowledge Engine (M1)                │
   └────────────────┬────────────────────────┘
                    │ 模板庫
                    ▼
   ┌──────────── 任務時（在線）─────────────┐
   │                                         │
   │  RGB + 點雲 + 語言 / 圖像指令            │
   │           │                              │
   │           ▼                              │
   │  ┌─────────────────────────────┐         │
   │  │  3D Visual Grounding (M2)   │←── 模板庫 │
   │  │  • 偵測物體 + 類別            │         │
   │  │  • 模板 ↔ 當前實例配對          │         │
   │  │  • 姿態 / 抓取點 / 功能平面投影 │         │
   │  └────────────┬────────────────┘         │
   │               │                           │
   │               ▼                           │
   │  ┌─────────────────────────────┐         │
   │  │  Retrieval-Augmented Task    │         │
   │  │  Planner (M3)                │         │
   │  │  context = (image + lang +   │         │
   │  │   grounded_3D_info)         │         │
   │  │  → VLM (GPT/Gemini)         │         │
   │  │  → 細粒度動作約束            │         │
   │  └────────────┬────────────────┘         │
   │               │                           │
   │               ▼                           │
   │  ┌─────────────────────────────┐         │
   │  │  軌跡優化（trajectory opt）   │         │
   │  └────────────┬────────────────┘         │
   │               ▼                           │
   │       機器人執行                          │
   └─────────────────────────────────────────┘
```

---

### 1.4 從論文到代碼：實作分解（GitHub 取證後新增）

論文層的 3 個模組（M1/M2/M3）對應 GitHub repo 的**4 步腳本管線**——粒度更細：

📦 **代碼倉庫**：[`RetrievalManip/Retrieval-augmented-Manipulation`](https://github.com/RetrievalManip/Retrieval-augmented-Manipulation)（Stars=4 / Forks=0，**新發佈，社群尚未驗證**）

```
論文模組          ↓                      實作腳本（執行順序）
──────────────────────────────────────────────────────────
M2.偵測       ↦  step1_grounding.py    GroundingDINO + SAM2.1
M2.重建       ↦  step1+step2 中        VGGT (point cloud)
M2.姿態+抓取  ↦  step2_ram.py          RAM 模組（核心 IP）
M3.規劃        ↦  step3_planning.py    VLM API 呼叫 + JSON parse
M3.執行        ↦  step4_conducting.py  trajectory + Fairino arm
```

**外部依賴清單**（這是「真開源」與否的關鍵判斷）：

| 元件 | 角色 | License | 訓練狀態 |
|------|------|:------:|:------:|
| **GroundingDINO** | 開放詞彙物體偵測 | Apache 2.0 | 預訓 |
| **SAM2.1** | 分割（每物體 mask） | Apache 2.0 | 預訓 |
| **VGGT** | 點雲重建（NVIDIA 2025） | 各自 | 預訓 |
| **DINOv2** | 視覺語意特徵 | 各自 | 預訓 |
| **VLM API** | 任務規劃（GPT/Gemini/Claude 任選） | 第三方 | API 調用 |
| **RAM 訓練** | `tools/ram_training/train_bop.py` | 待確認 | **本文唯一從零訓練的部分** |
| **CAD meshes** | 物體模板（11 類） | SharePoint 連結 | 人工建模 |

🧠 **觀察**：本論文「**不訓 VLM、不訓 3D foundation**」是真的——RAM 模組本身用 BOP-style 合成數據訓（BlenderProc）。其他全是預訓模型 + API 呼叫。**框架本質是 6 個成熟元件的智能組合**，創新在於**「組合方式 + RAG 範式遷移」**。

⚠️ **License 風險**：repo 主代碼**未指定 license**（README 說「外部元件依各自 license」）。商用前必須逐個元件審查 + 自行洽詢作者。

🔧 **硬體棧**（從 README 確認）：
- 機械臂：**Fairino**（國產，非 Franka / UR）— IP 配置
- 相機：**ZED + RealSense D435/D455**（雙相機，RGB-D）
- OS：Ubuntu 22.04 / Python 3.10 / CUDA 12.8

---

## 2. 數學核心：從模板到約束 (Math Core)

⚠️ **本節為作者推論結構**——具體公式 / 符號等待原文確認。

📌 **Napkin Formula**：
```
plan = VLM_θ( image, language, retrieve(template_K, observation) )
   ↓
constraints = parse(plan) ∈ ℝ^{N × (pose, grasp_pt, align_plane, dir)}
   ↓
trajectory = TrajOpt(constraints)
```
> 一行直覺：**VLM 不直接推 3D，它只負責「推理 + 規劃」。3D 知識來自 retrieve 出來的模板，VLM 只是把模板裡的空間原語拼成具體任務指令。**

### 2.1 類別模板形式（推測）

對類別 K，模板包含：
```
Template_K = {
  canonical_pose:  T_K^c ∈ SE(3),        # 標準參考座標
  bbox:            (W, D, H) ∈ ℝ^3,
  symmetries:      [axis_1, ...],        # 旋轉對稱
  grasp_points:    [(p_i, n_i, type_i)]_{i=1..G},  # 點 + 法向 + 類型
  functional_planes: [(O_j, n_j, role_j)]_{j=1..F},  # 中心 + 法向 + 用途
  (optional) articulated_axis or deform_states
}
```

### 2.2 Grounding 數學形式（推測）

給定觀測物體點雲 `P_obs` 與類別 K 的模板，目標求變換 `T ∈ SE(3)` 使
```
T* = argmin_T  Σ ||T · P_template - NN_obs(T · P_template)||²
                 + λ · feature_dist( CLIP(I_obs), CLIP(I_template_render) )
```
> **直覺**：點雲對齊（幾何）+ 視覺特徵對齊（語意）的混合 ICP——對形變、遮擋更穩。

### 2.3 RAG context 注入

VLM context 結構化為：
```
[user] 指令: "把杯子放到托盤中央"

[retrieved spatial context, JSON]
{
  "objects": [
    {"id": "cup_1", "category": "mug",
     "pose": [...], "grasp_points": [{"id":"handle","loc":[...]}],
     "functional_planes": [{"role":"bottom","normal":[0,0,-1]}]},
    {"id": "tray_1", "category": "tray",
     "pose": [...], "functional_planes": [{"role":"top","center":[...]}]}
  ]
}

[VLM generates]
plan = [
  ("grasp", "cup_1", "grasp_point.handle"),
  ("align", "cup_1.bottom", "tray_1.top.center"),
  ("place", height_offset=0.02)
]
```

**對比之前 VLM 直出**：「拿起杯子放到托盤」——沒有 grasp point，沒有對齊資訊，軌跡優化器無法直接執行。RAM 的 plan **每一步都帶可執行約束**。

---

## 3. 帶數字走一遍：擺餐具 (Worked Example)

任務：給定一張**目標擺放參考圖**（俯拍），機器人把當前桌面的餐具排成圖中布局。

```
Step 0: 觀測
  • 當前場景：3 個盤、2 個杯、1 把叉
  • 參考圖：盤居中，杯左上、叉右下，距離 5cm

Step 1: M2 Grounding（每個物體）
  cup_1: 模板 mug → T_1 ∈ SE(3), grasp=handle, plane=bottom
  cup_2: 模板 mug → T_2, grasp=handle, plane=bottom
  plate_1..3: 模板 plate → T, plane=top
  fork_1: 模板 fork → T, grasp=mid, plane=tine

Step 2: M3 RAG Plan（VLM 生）
  目標佈局：從參考圖解析空間關係
  動作鏈：
    1. plate_2 移到桌面中心（plane.bottom 對齊桌面）
    2. cup_1 移到 plate_2 左上 5cm（cup.bottom 對齊桌面）
    3. fork_1 移到 plate_2 右下 5cm（fork.tine 對齊桌面）

Step 3: TrajOpt
  將上述約束（pose target + plane alignment）解為 6-DoF 軌跡
```

**為什麼 92% 平面成功**（本文 image-guided 結果）：
- 平面場景（桌面同高度）$\to$ grounding 只需對齊 X-Y + 旋轉，誤差容忍大
- 高低平面（**72%**）$\to$ 額外要對齊 Z + 角度，grounding 誤差傳播 $\to$ 失敗率上升

⚠️ 這條 92% / 72% 數字來自 DeepTech 訪談，**未在原文表格逐項驗證**。

---

## 4. 工程視角：實驗規模、泛化、可重現性 (Engineering View)

| 維度 | 數值 / 觀察 | 工程含義 |
|------|------------|---------|
| **實驗任務** | **14 項真機任務** | 中大規模真機驗證（vs 多數論文 3-5 項） |
| **物體類別** | **11 類**（杯、盤、餐具、抽屜、衣物 $\dots$） | 起步覆蓋日常 |
| **物體實例** | **31 個** | 平均 ~3 實例/類 |
| **語言驅動成功率** | **89.17%（120 次）** | 含單物體單步、多物體單步、多物體多步 |
| **多物體多步成功率** | **80%** | 長程任務最具挑戰 |
| **圖像引導 - 平面** | **92%** | 餐具擺放類 |
| **圖像引導 - 高低平面** | **72%** | 立體場景仍有約 25% 提升空間 |
| **桌面清掃 + 工具選擇** | **65%** | 自主規劃借助簸箕等中介工具 |
| **VLM 兼容性** | API 模式（GitHub `languages/` 目錄） | 任意支援 OpenAI-compat 的模型可接入 |
| **訓練資源** | 只訓 RAM 模組（BOP-style 合成數據 via BlenderProc） | 其餘 6 個元件全預訓 — 訓練成本極低 |
| **代碼/資料** | ✅ **已開源**（[RetrievalManip 組織](https://github.com/RetrievalManip/Retrieval-augmented-Manipulation)） | ⚠️ License 未明列；CAD meshes via SharePoint |
| **機械臂** | **Fairino**（國產） | 非 Franka/UR — 跨形態遷移待驗證 |
| **相機棧** | **ZED + RealSense D435/D455** | 雙相機 RGB-D |
| **OS / Python** | Ubuntu 22.04 / Python 3.10 / CUDA 12.8 | 硬性依賴 |
| **倉庫成熟度** | Stars=4, Forks=0 (release 後不久) | 社群驗證尚未發生 |

**部署約束**：
- 11 類起步——日常物體覆蓋夠用，**但開放世界（廚房特殊器具、工業零件）需擴模板庫**
- 模板需人工標準姿態 / 抓取點 / 功能平面——**新類別擴展是人力瓶頸**
- TrajOpt 假設可微 / 連續——**接觸豐富、衝擊類任務未測試**
- VLM 推理延遲未公開

---

## 5. 數據與評測 (Data & Eval)

### 5.1 三類任務分組（語言驅動）

| 任務類型 | 預期難度 | 成功率（聚合） |
|---------|:------:|:-------------:|
| 單物體單步驟 | 低 | （未細列，總體 89.17%） |
| 多物體單步驟 | 中 | 同 |
| **多物體多步驟** | **高** | **80%** |

### 5.2 圖像引導擺放

| 場景 | 成功率 | 對比 |
|------|:----:|------|
| 常規平面（桌面同高） | **92%** | — |
| 複雜高低平面 | **72%** | 突破「**傳統方法依賴俯視參考圖**」限制 |

### 5.3 自主決策（空間約束推理）

| 任務 | 成功率 | 描述 |
|------|:----:|------|
| 桌面清掃 | **65%** | 直接清掃不可行 $\to$ 規劃借助簸箕等中介工具 |

### 5.4 物體類型擴展

| 類型 | 方法 | 任務 |
|------|------|------|
| 鉸接物體（laptop、抽屜） | 多模板匹配（不同開合狀態預設） | 估計旋轉軸 / 推動方向 |
| 柔性物體（衣物） | 折疊步驟拆解 + 分階段模板 | 展開$\to$疊袖（左/右）$\to$疊邊 |
| 觸覺擴展 | 抓取重心偏移時，觸覺反饋觸發 re-grasp | 重新規劃姿勢 |

⚠️ 鉸接 / 柔性的成功率未在訪談中給出，**待原文補**。

---

## 6. 能力與失敗模式 (Capabilities & Failure Modes)

### 6.1 能做什麼
- 跨類別空間操作（11 類起步）
- 語言 + 圖像兩種指令模式都支援
- 自主規劃工具使用（清掃 + 簸箕）
- 鉸接 / 柔性物體初步擴展
- VLM agnostic（不綁定單一模型）

### 6.2 失敗模式（推測 + 訪談部分提及）

| 場景 | 為何失敗 | 來源 |
|------|---------|------|
| 高低平面 image-guided | grounding 誤差在 Z 軸放大 | 數字隱含（92% $\to$ 72%） |
| 桌面清掃 35% 失敗案例 | 工具選擇 / 抓取軌跡 / 推送方向誤差 | 65% 反推 |
| 開放類別物體 | 不在 11 類模板庫內 $\to$ 無法 ground | 訪談明確列為 future work |
| 接觸豐富 / 高速衝擊 | TrajOpt 範式假設不成立 | 隱含 |
| VLM 幻覺 / 邊界推理 | RAG 注入不能完全消除 | 推論 |

### 6.3 隱含假設 (Hidden Assumptions)

1. **類別已知**：M2 grounding 需先知道是哪個類別——**開放詞彙偵測未在框架內**
2. **模板覆蓋有限變體**：標準形態的杯/盤模板對「異形馬克杯」「藝術盤」可能誤匹
3. **單物體 = 單模板**：複合物體（如咖啡機、茶具組）超出當前範式
4. **RAG context 長度足夠**：場景物體多時，serialize 的 JSON 可能逼近 VLM context 上限
5. **TrajOpt 可解**：給定約束**有可行軌跡**——複雜多體碰撞下不一定

---

## 7. 與相關工作對比 (Comparison)

| 方法 | 空間表達 | VLM 訓練 | 物體類別擴展 | 真機規模 | Science Robotics? |
|------|--------|:--------:|:----------:|:------:|:-----------------:|
| Code-as-Policies | 隱式（代碼控制） | ❌ | 動態 | 中 | ❌ |
| ReKep | 顯式（關鍵點約束） | ❌ | per-task 設計 | 中 | ❌ |
| VoxPoser | 顯式（voxel 場） | ❌ | per-task | 中 | ❌ |
| $\pi_{0.5}$ / Hi-Robot | 隱式（VLA 雙層） | ✅（VLA 訓） | 看 VLA 預訓 | 中 | ❌ |
| CodeGraphVLP | 顯式（持久語義圖） | ❌ | 程式碼 ad-hoc | 中 | ❌ |
| **RAM（本文）** | **顯式（類別模板庫）** | **❌（不訓 VLM）** | **新類別 = 新模板** | **大（14 任務 / 31 實例）** | **✅** |

### 🎤 面試 Tip

> **被問「RAG 能搬到 manipulation 嗎？」** ——
> 三句話：(1) RAM 是第一個系統性把 NLP RAG 範式搬到 manipulation 的工作，**不訓 VLM、不訓 3D 模型**，靠類別級模板庫 + 3D grounding + context 注入；(2) 真機 89% 多步成功率 + Science Robotics peer-review，是**強的可行性證據**；(3) 但要老實補：模板覆蓋 11 類是起步、新類別擴展靠人工、開放詞彙偵測在框架外、複雜接觸未測試——**它證明了路徑可行，沒證明 scale 容易**。

---

## 8. 待追問的開放問題

> 取得 GitHub repo 後，部分問題已答；剩餘待**原 PDF + 補充材料**確認：

**已答（從 GitHub README 確認）**：
- ~~代碼是否開源~~ $\to$ ✅ **已開源**（[RetrievalManip 組織](https://github.com/RetrievalManip/Retrieval-augmented-Manipulation)，但 license 未明列）
- ~~VLM API 形式~~ $\to$ ✅ **`languages/` 目錄做 API client + prompt**——任意 OpenAI-compat 模型可換
- ~~硬體~~ $\to$ ✅ **Fairino arm + ZED + RealSense D435/D455**

**仍未答**（待 PDF / 補材確認）：
1. **VLM 對比定量**：repo 是 API 介面，但**論文是否報告 GPT-4o vs Gemini 的成功率差異**？
2. **Baseline 定量對比**：RAM vs VoxPoser / ReKep / $\pi_{0.5}$ / CodeGraphVLP 在 same task suite 的成功率差？訪談沒提
3. **成功率細粒度**：89.17% 中**單步 vs 多步**分布；多步 80% **是 2 步、3 步還是 5 步**？
4. **Sim2real gap 量化**：BOP-style 合成數據訓 RAM $\to$ 真實場景**性能下降百分比**？
5. **觸覺整合成功率**：訪談提及，無數字
6. **VLM context 長度**：場景 10+ 物體時 retrieved JSON 多大？會吃到 context window 上限嗎？
7. **TrajOpt 求解器**：repo `planner/` 目錄但具體用 OMPL / CHOMP / 可微分優化哪一個？收斂率？
8. **長程任務上限**：80% 是「多步」最高難度——具體**幾步**？10 步以上有測嗎？
9. **failure mode 統計**：失敗的 11% / 20% / 35% 各自是 grounding 錯 / 規劃錯 / 執行錯？哪個是主因？
10. **License 商用風險**：repo 未指定 license——是否 MIT/Apache？商用前必須與 CUHK 確認

📎 **內容類型可信度**：

| 來源 | 可信度 | 對應內容 |
|------|--------|---------|
| Science Robotics（peer-reviewed） | 🟢 **高**（但未直接讀到原文） | 論文存在性、3 模組架構、實驗規模 |
| **GitHub 代碼倉庫** | **🟢 高（直接源碼）** | **4 步管線、Fairino 硬體、外部依賴清單、訓練腳本路徑** |
| DeepTech 第一作者訪談 | 🟡 中高 | 具體數字 89.17/80/92/72/65、擴展（鉸接/柔性/觸覺） |
| 期刊封面/標題 | 🟢 高 | 論文題目、DOI、Issue |

---

📎 **來源**：
- 期刊: https://www.science.org/doi/10.1126/scirobotics.aea2092 （**訂閱牆**，本文未直接核對 PDF）
- Cite: Chen K, Li C, Tu C, et al. *Science Robotics*. 2026; 11(113):eaea2092
- **代碼**: https://github.com/RetrievalManip/Retrieval-augmented-Manipulation （Stars=4, Forks=0；license 未指定）
- 媒體訪談: DeepTech 第一作者陳凱訪談（2026 年）
- 上游依賴: GroundingDINO $\cdot$ SAM2.1 $\cdot$ DINOv2 $\cdot$ VGGT (NVIDIA) $\cdot$ BlenderProc $\cdot$ Fairino SDK $\cdot$ ZED/RealSense bindings
- 對比基線: Code-as-Policies (Liang 2023) · ReKep (Huang 2024) · VoxPoser (Huang 2023) · $\pi_{0.5}$ (Black 2025) · CodeGraphVLP (2026) · Hi-Robot

🧠 **本文判讀（作者觀點）**：
這篇是**少見的「不靠 scale 而靠範式遷移」的 manipulation 論文**——把 NLP 已驗證的 RAG 範式搬到具身 AI，**第一個系統性地把「外部空間知識」做成 first-class citizen**。Science Robotics 收錄是強信號（該期刊年錄用 100+ 篇）。

**評級 ⚡ 戰略級**：
- ✅ Peer-reviewed top venue（Science Robotics）
- ✅ 真機規模 14 任務 / 31 實例 / 11 類別（vs 多數 manipulation 論文 3-5 任務）
- ✅ 89.17% 平均 / 80% 多步 / 92% image-guided 都是強表現
- ✅ 範式創新明確（**RAG meets manipulation**）
- ✅ 對 VLA 工程師有方法論啟發：**外部知識庫 + retrieval 是除微調外的第三條路**

但要老實列出**降權因素**：
- ⚠️ 未直接讀 PDF，所有引用基於訪談
- ⚠️ 代碼是否開源未確認
- ⚠️ Baseline 定量對比未在訪談中呈現

若 3-6 個月內代碼開源 + 獨立復現驗證，可在 VLA-Handbook 升級為「mainline reference」之一。

---

[← Back to Planning](./README.md) · [← Back to Theory](../README.md)
