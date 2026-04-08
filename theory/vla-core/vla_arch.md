# VLA 核心架构 (VLA Core Architectures)

> **VLA = Vision + Language + Action**。把一个视觉语言模型（VLM）装上"手"，让机器人看懂世界、听懂指令、做出动作。
>
> 本文是 VLA 架构的**全景导航**，从 2022 年的 RT-1 到 2026 年的最新模型。每个模型附有深度解析链接。

---

## 通用 VLA 架构模板

**所有 VLA 共享同一个骨架**，差异只在三个选择上：

```mermaid
graph TD
    IMG["📷 Image Stream"] --> VE["Vision Encoder<br/><i>ViT / SigLIP / DINOv2</i>"]
    LANG["📝 Language Instruction"] --> LE["Language Encoder<br/><i>LLM / Gemma / Qwen</i>"]
    PROP["🦾 Proprioception"] --> PE["State Encoder<br/><i>MLP / Tokenizer</i>"]

    VE --> FUSION["🔀 Multimodal Fusion<br/><i>Cross-attention / Interleave / MoE</i>"]
    LE --> FUSION
    PE --> FUSION

    FUSION --> BACKBONE["Transformer Backbone<br/><i>1B ~ 55B parameters</i>"]

    BACKBONE --> HEAD["Action Head"]

    HEAD --> T["🎰 Token Head<br/><i>RT-2, FAST</i><br/>离散化为 256 bins"]
    HEAD --> D["🌊 Diffusion Head<br/><i>RDT, Octo</i><br/>DDPM 去噪"]
    HEAD --> F["💨 Flow Head<br/><i>π0, WALL-OSS</i><br/>Flow Matching"]

    T --> ACT["🦾 Robot Action<br/>(x, y, z, rot, gripper)"]
    D --> ACT
    F --> ACT

    style IMG fill:#1a1a2e,stroke:#4361ee,color:#fff
    style LANG fill:#1a1a2e,stroke:#4361ee,color:#fff
    style PROP fill:#1a1a2e,stroke:#4361ee,color:#fff
    style FUSION fill:#0f3460,stroke:#16213e,color:#fff
    style BACKBONE fill:#0f3460,stroke:#16213e,color:#fff
    style HEAD fill:#e94560,stroke:#e94560,color:#fff
    style T fill:#1a1a2e,stroke:#f77f00,color:#fff
    style D fill:#1a1a2e,stroke:#f77f00,color:#fff
    style F fill:#1a1a2e,stroke:#f77f00,color:#fff
    style ACT fill:#2a9d8f,stroke:#2a9d8f,color:#fff
```

**三个关键选择**：

| 选择 | 选项 | 代表模型 | 权衡 |
|------|------|---------|------|
| **VLM Backbone** | 小 (~1B) vs 大 (~55B) | OpenVLA (7B) vs RT-2 (55B) | 推理速度 vs 语义理解 |
| **Action Head** | Token vs Diffusion vs Flow | RT-2 vs RDT vs π0 | 简单性 vs 多模态动作 vs 速度 |
| **训练策略** | 纯 BC vs Co-train vs RL | RT-1 vs RT-2 vs π*0.6 | 数据效率 vs 泛化 vs 性能上限 |

> 💡 **关键洞察**：VLA 的演进不是模型越大越好，而是在这三个维度上找到最优组合。π0 用 3B 参数 + Flow Matching 打败了 RT-2 的 55B + Token，靠的是更好的 Action Head 设计。

---

## 三代演进

### 第一代：专用模型（2022-2023）

> *"先证明 Transformer 能控制机器人。"*

&nbsp;

#### RT-1 — 开山之作

> **核心思想**：把机器人控制建模为 Token 生成问题。

- **架构**：EfficientNet + FiLM conditioning + TokenLearner + Transformer
- **动作表示**：连续动作离散化为 256 bins → 输出离散 Token 序列
- **数据**：130K episodes，17 个月的 Everyday Robots 数据
- **TokenLearner**：把 81 个视觉 token 压缩到 8 个，推理速度快 2x
- **局限**：只在自家数据上有效，换个桌子就不行——没有互联网知识的迁移

&nbsp;

#### RT-2 — VLA 概念的诞生

> **核心思想**：**VLA = VLM + Action Tokens**。直接微调 PaLI-X (55B)，让大语言模型"说出"动作。

- **关键创新**：动作被编码为特殊文本 token（如 `"1 128 90 ..."`），与自然语言共享词表
- **Co-fine-tuning**：混合训练互联网 VQA 数据 + 机器人数据，防止灾难性遗忘
- **涌现能力**：能理解 "pick up the extinct animal"（抓恐龙玩具）——VLM 的语义理解迁移到了动作
- **局限**：55B 参数推理太慢，无法实时控制

> 💡 **这是一个范式转折点**：RT-2 证明了预训练 VLM 的知识可以直接迁移到机器人控制。之后所有 VLA 都沿着这条路走。

&nbsp;

---

### 第二代：开源浪潮（2024）

> *"让所有人都能训练自己的 VLA。"*

&nbsp;

#### ACT — 动作分块变换器

> 📖 **[Deep Dive: ACT 详解](act.md)**

- **核心数学**：**CVAE（条件变分自编码器）** — 假设动作序列由低维"意图"变量 z 生成
- **Action Chunking**：不是预测下一步，而是一次预测未来 k 步的动作块
- **硬件民主化**：ALOHA 平台只需 ~$20K，双臂遥操作
- **影响**：ACT 是最被广泛复现的 VLA baseline，代码简洁优雅

&nbsp;

#### OpenVLA — 7B 开源 VLA

- **架构**：Llama 2 (7B) + DINOv2/SigLIP
- **Action Head**：不用文本 token，而是专门的线性层 + Action Detokenization
- **优化**：支持 4-bit QLoRA 训练，消费级显卡可跑
- **数据**：Open X-Embodiment (OXE) 数据集

&nbsp;

#### RDT — 扩散变换器

> 📖 **[Deep Dive: RDT 详解](rdt.md)**

- **核心思想**：把 Diffusion Transformer（DiT）从图像生成搬到动作生成
- **规模**：RDT-1B 是首个 **10 亿参数**的扩散策略模型
- **Scalable Transformer**：统一处理不同机器人的异构状态/动作空间
- **开发者**：清华 MARS Lab + 字节跳动

&nbsp;

#### FAST — 高效动作 Token 化

> 📖 **[Deep Dive: FAST 详解](fast.md)**

- **问题**：简单分 bin 丢失高频信息，Diffusion 又太慢
- **方案**：DCT（离散余弦变换）+ BPE（字节对编码）— 频域压缩 + 子词粒度
- **效果**：OpenVLA 训练加速 5x，保持动作精度
- **已被 π0.5 采用**（预训练阶段用 FAST，推理阶段切 Flow Matching）

&nbsp;

---

### 第三代：基础模型竞赛（2025-2026）

> *"从'能用'到'好用'——跨形态、双系统、RL 自我提升。"*

这一代的关键趋势：**双系统架构**（快慢分离）、**跨形态**（一个模型控制多种机器人）、**RL 后训练**（超越模仿学习的天花板）。

&nbsp;

#### π0 → π0.5 → π0.6 / π*0.6 — Physical Intelligence

> 📖 Deep Dive: [π0 代码解析](pi0_code_analysis.md) · [π0.5 解剖](pi0_5_dissection.md) · [π0.6 解剖](pi0_6_dissection.md)

**三步演进**：

| 版本 | 核心升级 | Action Head | 关键创新 |
|------|---------|------------|---------|
| π0 | 基础 VLA，Flow Matching | Flow Matching | 首个用 FM 的 VLA，速度 > Diffusion |
| π0.5 | 开放世界泛化 | FAST (pretrain) + FM (infer) | 统一模型同时做语义规划和电机控制 |
| π0.6 | 5B 参数 VLM 升级 | Flow + Action Expert | 专门的"动作专家"模块解决大模型"手笨" |
| π\*0.6 | **RL 后训练** | 同上 + Recap | Offline RL 复盘成功/失败，吞吐量翻倍 |

> 💡 **π\*0.6 的 Recap 算法是一个里程碑**：它证明了 VLA 可以通过 RL 自我提升，而不仅仅是模仿人类示范。
> → 详见 [π0.6 / RECAP 解析](../rl/pi0_6_recap_rl_as_supervised_learning.md)

&nbsp;

#### GR00T-N1.6 — NVIDIA 的人形机器人大脑

> 📖 **[Deep Dive: GR00T-N1.6 解剖](gr00t_n1_6.md)**

- **双系统架构**：
  - **System 2**（慢）：VLM 处理语义指令 → 生成子任务描述（~2Hz）
  - **System 1**（快）：Diffusion Transformer 处理视觉+本体感觉 → 输出关节动作（100Hz）
- **为什么分开**：语义理解不需要 100Hz，运动控制必须达到这个频率。解耦后各自优化。
- **数据**：跨形态训练（Unitree H1、Fourier GR1、自家人形机器人）

&nbsp;

#### Figure Helix 02 — 全身端到端 VLA

> 📖 **[Deep Dive: Helix 02 解剖](figure_helix_02_full_body_autonomy_2026.md)**

- **三层系统**（比 GR00T 多一层）：
  - **S2**（~1Hz）：语义目标
  - **S1**（200Hz）：全身关节目标——视觉 + 触觉 + 掌心相机
  - **S0**（1kHz）：底层扭矩控制——人类运动先验（learned prior）
- **核心突破**：解决了 **loco-manipulation 耦合**——走路和抓东西同时做时不崩溃
- **关键设计**：S0 的人类运动先验承接稳定性，S1 的 visuomotor policy 解决耦合

&nbsp;

#### WALL-OSS — 开源具身 VLA

> 📖 **[Deep Dive: WALL-OSS 详解](wall_oss.md)**

- **架构**：Qwen2.5 VLMoE backbone + **双分支** Action Head
  - **Flow 分支**：精细连续控制（Flow Matching）
  - **FAST 分支**：粗粒度快速动作（离散 token）
- **CoT 推理**：模型先输出中间推理步骤，再生成动作——VLA 级别的思维链
- **定位**："具身 AI 的 Linux"——完全开源，支持 Cross-Embodiment

&nbsp;

#### 更多值得关注的模型

<details>
<summary>展开查看 8 个其他重要模型</summary>

&nbsp;

| 模型 | 定位 | 特色 | Deep Dive |
|------|------|------|----------|
| **Spirit-v1.5** | 小模型高效 VLA | 轻量级架构，适合边缘部署 | [解剖](spirit_v1_5_dissection.md) |
| **SimVLA** | 简洁 baseline | "最简单能跑的 VLA"——实验室复现首选 | [解析](simvla_simple_vla_baseline_robotic_manipulation_2026.md) |
| **Galaxea G0** | 双系统 VLA | 快慢思维分离架构 | [解析](galaxea_g0.md) |
| **OneTwoVLA** | 双系统 VLA | System 1/2 解耦 | [解析](onetwovla.md) |
| **LingBot-VLA** | 实用主义 VLA | 高吞吐训练栈，工程导向 | [解析](lingbot_vla_pragmatic_vla_foundation_model_2026.md) |
| **UnifoLM-VLA-0** | Unitree 开源 VLA | 四足+机械臂的 VLA 实现 | [解析](unifolm_vla_0_unitree_2026.md) |
| **StarVLA** | Lego 式 VLA 代码库 | 模块化研发框架 | [解析](starvla_lego_like_vla_codebase_2026.md) |
| **ABot-M0** | 动作流形学习 | Action Manifold VLA | [解析](abot_m0_action_manifold_learning_vla_foundation_2026.md) |

</details>

&nbsp;

---

## 关键架构趋势

### 趋势 1：双系统架构（快慢分离）

```
System 2 (慢/语义)                    System 1 (快/运动)
┌──────────────────┐                 ┌──────────────────┐
│  VLM (~2Hz)      │  子任务描述 →   │  Action Model    │
│  "把杯子放到架上" │  ─────────→     │  (~100-200Hz)    │
│  → "先抬起杯子"  │                 │  → 关节角度序列   │
└──────────────────┘                 └──────────────────┘
```

**为什么分开？** 语义理解和运动控制的时间尺度差 50-100 倍。一个 7B VLM 跑 100Hz 是不现实的，但 200M 的 Action Model 可以。

**采用的模型**：GR00T-N1.6、Figure Helix 02、Galaxea G0、OneTwoVLA

> → 另见 [小模型 VLA 研究方向](small_vla_models.md)——System 1 天然适合小模型

&nbsp;

### 趋势 2：Action Head 从 Token 走向 Flow

| 方式 | 速度 | 多模态动作 | 精度 | 代表 |
|------|------|-----------|------|------|
| **Token** (离散化) | ⚡⚡⚡ 最快 | ❌ 不行 | 🔶 粗糙 | RT-1, RT-2 |
| **Diffusion** (DDPM) | ⚡ 慢 | ✅ 很好 | 🟢 精细 | RDT, Octo |
| **Flow Matching** | ⚡⚡ 快 | ✅ 很好 | 🟢 精细 | π0, WALL-OSS |
| **FAST** (DCT+BPE) | ⚡⚡⚡ 最快 | 🔶 一般 | 🟢 精细 | FAST tokenizer |

> 💡 **核心权衡**：Token 快但不能处理多模态动作（同一任务的多种做法）。Diffusion 能处理但太慢。Flow Matching 是目前的最优解——速度接近 Token，精度接近 Diffusion。
>
> → 详见 [Diffusion Policy 详解](../diffusion-flow/diffusion_policy.md) · [Flow Matching (π0)](pi0_code_analysis.md) · [Compression Gap](../diffusion-flow/the_compression_gap_why_discrete_tokenization_limits_vision_dissection.md)

&nbsp;

### 趋势 3：RL 后训练成为标配

| 阶段 | 方法 | 效果 | 代表 |
|------|------|------|------|
| Stage 1 | 互联网预训练 | VLM 语义能力 | 所有模型 |
| Stage 2 | 模仿学习 (BC/SFT) | 基本操作能力 | 所有模型 |
| Stage 3 | **RL 后训练** | **超越示范者** | π*0.6, GR-RL |

π*0.6 的 Recap 算法证明了 VLA 可以通过复盘（offline RL）自我提升。这打开了 VLA 的 "post-training" 时代——类似 LLM 从 SFT 到 RLHF 的演进。

> → 详见 [VLA+RL 实战教程](../rl/vla_rl_practical_guide.md) · [GR-RL 解剖](../rl/gr_rl_dissection.md)

&nbsp;

---

## 模型全景对比

| 模型 | 年份 | 参数 | VLM Backbone | Action Head | 训练数据 | 开源 | 特色 |
|------|:----:|-----:|-------------|------------|---------|:----:|------|
| RT-1 | 2022 | ~35M | EfficientNet | Token (256 bins) | Robot only | ❌ | 开山之作 |
| RT-2 | 2023 | 55B | PaLI-X | Text Token | Web + Robot | ❌ | 涌现语义能力 |
| ACT | 2023 | ~80M | — | CVAE Chunk | ALOHA Demo | ✅ | 动作分块，$20K 硬件 |
| Octo | 2024 | ~93M | — | Diffusion | OXE | ✅ | 多形态 Diffusion |
| OpenVLA | 2024 | 7B | Llama 2 + SigLIP | Linear Head | OXE | ✅ | 4-bit QLoRA |
| RDT-1B | 2024 | 1B | — | Diffusion Transformer | OXE | ✅ | 首个 1B 扩散策略 |
| π0 | 2024 | 3B | Gemma | Flow Matching | Cross-Embodiment | ✅ | 首个 FM VLA |
| π0.5 | 2025 | 3B | Gemma | FAST + FM | + YouTube | ✅ | 开放世界泛化 |
| π\*0.6 | 2025 | 5B | Gemma 3 | FM + Action Expert | + RL Data | ✅ | **Recap RL** |
| GR00T-N1.6 | 2025 | ~2B | Custom VLM | Diffusion Transformer | Humanoid Multi | ✅ | 双系统 100Hz |
| WALL-OSS | 2025 | ~7B | Qwen2.5 VLMoE | Dual (Flow + FAST) | Cross-Embodiment | ✅ | CoT + 双分支 |
| Helix 02 | 2026 | ? | Custom | S0/S1/S2 分层 | Humanoid | ❌ | 全身 loco-manipulation |

&nbsp;

---

## FAQ

<details>
<summary><b>VLA 和传统机器人控制有什么区别？</b></summary>

传统控制（PID、MPC）需要人手工建模环境和设计控制器。VLA 从数据中端到端学习——输入是摄像头图像和语言指令，输出是关节角度。不需要人工建模，但需要大量示范数据。

两者不是对立的：Helix 02 的 S0 层就用了传统控制作为安全兜底。
</details>

<details>
<summary><b>VLA 能实时控制吗？</b></summary>

取决于 Action Head 和模型大小：
- **Token Head (RT-1)**：~5ms/step → 200Hz ✅
- **Flow Matching (π0)**：~30ms/step → 30Hz ✅
- **Diffusion (RDT)**：~100ms/step → 10Hz 🔶（需要 action chunking 补偿）
- **大 VLM 推理 (RT-2)**：~500ms+ → 2Hz ❌（不够快）

双系统架构解决了这个问题：大 VLM 跑 2Hz 做语义，小模型跑 100Hz 做动作。
</details>

<details>
<summary><b>我应该选哪个模型作为 baseline？</b></summary>

- **快速原型**：[SimVLA](simvla_simple_vla_baseline_robotic_manipulation_2026.md)——最简单
- **学术研究**：[ACT](act.md)——代码最干净，社区最大
- **工程部署**：[OpenVLA](https://openvla.github.io/)——支持量化，文档好
- **追求 SOTA**：π0.6 或 GR00T-N1.6——性能最强但资源需求大
- **代码框架**：[StarVLA](starvla_lego_like_vla_codebase_2026.md)——模块化 Lego 式
</details>

<details>
<summary><b>VLA 十大挑战是什么？</b></summary>

→ 详见 [VLA 十大挑战](../planning/vla_challenges.md)

简要：泛化、实时性、安全、长程任务、多模态融合、数据效率、Sim2Real、评估标准、可解释性、跨形态迁移。
</details>

&nbsp;

---

## 进一步阅读

| 方向 | 推荐 |
|------|------|
| 研究主线总览 | [VLA 研究主线梳理](vla_research_mainline.md) |
| 动作生成细节 | [动作生成范式详解](../diffusion-flow/action_representations.md) |
| 小模型路线 | [小模型 VLA 深度分析](small_vla_models.md) — 40 页长文 |
| 视觉接地 | [ReconVLA](reconvla_implicit_grounding_by_reconstruction.md) · [FocusVLA](focusvla_focused_visual_utilization_for_vision_language_acti_dissection.md) |
| 语言理解缺口 | [LangGap](langgap_diagnosing_and_closing_the_language_gap_in_vision_la_dissection.md) |
| 双臂操作 | [TwinVLA](twinvla_data_efficient_bimanual_manipulation_with_twin_singl_dissection.md) |
| 效率优化 | [Beyond Attention Magnitude](beyond_attention_magnitude_leveraging_inter_layer_rank_consi_dissection.md) |

---

[← Back to Explorer's Map](./README.md)
