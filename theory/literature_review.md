# VLA 文献核心技术归纳 (Literature Technical Review)

> **快速索引**: [论文索引 (Paper Index)](./paper_index.md) - 多维度查找系统
> **最后更新**: 2026-02-03

本章节对 VLA 领域的核心文献进行**深度技术归纳**，按技术分类组织，适合面试前快速复习模型细节。

---

## 📑 快速导航

| 分类方式 | 跳转链接 |
|:---|:---|
| 📊 [论文索引](./paper_index.md) | 多维度索引（技术/公司/时间） |
| 🎯 [按技术分类](#按技术分类) | 动作生成/训练方法/架构/应用 |
| 🏢 [按公司分类](#按公司机构分类) | Google/Physical Intelligence/ByteDance 等 |
| 📅 [按时间线](#按时间线) | 2023/2024/2025/2026 |
| 📊 [总结对比表](#总结对比表) | 所有模型快速对比 |

---

## 🎯 按技术分类

### -0.5 经典真机数据与闭环抓取（Hand‑Eye Coordination / Grasping）

> 说明：这类工作是“VLA 之前的真机学习底座”：它用可规模化的成功/失败信号，把抓取做成“候选动作评分 + 闭环伺服纠错”的系统工程范式。

- Levine et al. (2016)：Learning Hand‑Eye Coordination for Robotic Grasping with Deep Learning and Large‑Scale Data Collection  
  - 手册解读：[`./classics/levine_hand_eye_coordination_grasping_2016.md`](./classics/levine_hand_eye_coordination_grasping_2016.md)  
  - arXiv：`https://arxiv.org/abs/1603.02199`

- Levine et al. (2016)：End-to-End Training of Deep Visuomotor Policies（像素→扭矩，GPS 把 RL 变成监督学习）  
  - 手册解读：[`./classics/levine_end_to_end_visuomotor_policies_2016.md`](./classics/levine_end_to_end_visuomotor_policies_2016.md)  
  - arXiv：`https://arxiv.org/abs/1504.00702`

- Levine & Abbeel (2014)：Learning Neural Network Policies with Guided Policy Search under Unknown Dynamics（未知动力学 + KL trust region）  
  - 手册解读：[`./classics/levine_gps_unknown_dynamics_2014.md`](./classics/levine_gps_unknown_dynamics_2014.md)  
  - NeurIPS PDF：`https://proceedings.neurips.cc/paper_files/paper/2014/file/6766aa2750c19aad2fa1b32f36ed4aee-Paper.pdf`

### -0.4 基础模型能力注入：具身推理与可提示表征（Embodied CoT / Promptable Representations）

> 说明：这两篇更像“把 VLM 的优势接到控制系统里”的两种方式：  
> (1) **把推理链变成 policy 的显式中间变量（ECoT）**；(2) **把世界知识变成 state representation（PR2L）**。

- Zawalski et al. (2024/2025)：Robotic Control via Embodied Chain-of-Thought Reasoning（ECoT：先推理再出动作，含 bbox/gripper grounding）  
  - 手册解读：[`./frontier/embodied_chain_of_thought_robotic_control_2024.md`](./frontier/embodied_chain_of_thought_robotic_control_2024.md)  
  - arXiv：`https://arxiv.org/abs/2407.08693`

- Chen et al. (2024 arXiv / 2025 TMLR)：Vision-Language Models Provide Promptable Representations for Reinforcement Learning（PR2L：promptable state embedding + RL/BC grounding）  
  - 手册解读：[`./frontier/vlm_promptable_representations_for_rl_pr2l_2025.md`](./frontier/vlm_promptable_representations_for_rl_pr2l_2025.md)  
  - arXiv：`https://arxiv.org/abs/2402.02651`

### -0.3 在线适配：无监督技能预训练与生成式回放（U2O RL / PGR）

> 说明：这两篇更偏“训练配方/系统组件”，但对 VLA 很关键：  
> (1) **U2O RL**：把离线预训练从“任务特定”改成“可复用的无监督技能”，再在线对齐任务；  
> (2) **PGR**：把 replay buffer 从“重采样”升级为“条件生成”，提升在线样本效率并降低过拟合。

- Kim et al. (2024)：Unsupervised-to-Online Reinforcement Learning（U2O RL：无监督离线技能预训练 + bridging + online fine-tuning）  
  - 手册解读：[`./frontier/unsupervised_to_online_reinforcement_learning_u2o_2024.md`](./frontier/unsupervised_to_online_reinforcement_learning_u2o_2024.md)  
  - arXiv：`https://arxiv.org/abs/2408.14785`

- Wang et al. (2024 arXiv / 2025 ICLR)：Prioritized Generative Replay（PGR：条件扩散生成式回放 + relevance guidance，curiosity 是强默认项）  
  - 手册解读：[`./frontier/prioritized_generative_replay_pgr_2025.md`](./frontier/prioritized_generative_replay_pgr_2025.md)  
  - arXiv：`https://arxiv.org/abs/2410.18082`

### 0. 触觉 / 视触觉（Tactile）

> 说明：本综述以 VLA 动作生成与训练范式为主，但在灵巧操作中，“接触相位”的可观测性往往决定系统上限，因此建议把触觉相关工作作为必备补充阅读。

- 触觉为何不可替代（工程盘点）：[`./frontier/tactile_irreplaceable.md`](./frontier/tactile_irreplaceable.md)
- SciRobotics 2026：Visual-tactile pretraining + online multitask learning（单目 + 二值触觉，统一策略覆盖多任务）  
  - 手册笔记：[`./frontier/visual_tactile_pretraining_online_multitask_learning_2026.md`](./frontier/visual_tactile_pretraining_online_multitask_learning_2026.md)  
  - 论文 DOI：`https://doi.org/10.1126/scirobotics.ady2869`  
  - Focus：`https://doi.org/10.1126/scirobotics.aee5782`
- SuperTac + DOVE（Nature Sensors, 2025）：多模态电子皮肤 + 触觉语言模型（把触觉升级为可解释的语义状态）  
  - 论文页：`https://www.nature.com/articles/s44460-025-00006-y`  
  - 手册笔记：[`../deployment/perception/supertac_dove_biomimetic_multimodal_tactile_sensing.md`](../deployment/perception/supertac_dove_biomimetic_multimodal_tactile_sensing.md)
- UniTacHand（2025）：人手→机器人触觉表征统一与技能迁移（MANO UV Map）  
  - 手册解读：[`./frontier/unitachhand.md`](./frontier/unitachhand.md)
- GenForce（Nat Commun 2026）：跨触觉传感器的可迁移力感知（统一 marker 表示 + M2M 条件扩散 + 时序力回归 + 材料补偿）  
  - 深度笔记：[`./tactile/genforce_tactile_force_transfer_2026.md`](./tactile/genforce_tactile_force_transfer_2026.md)  
  - 论文 DOI：`https://doi.org/10.1038/s41467-026-68753-1`

### 0.2 多模态融合与策略共识（Policy Consensus）

> 说明：当触觉是“稀疏但关键”的模态时，特征拼接会把它当噪声；策略共识让每个模态拥有独立专家策略，再在动作层面组合。

- Multi-Modal Manipulation via Policy Consensus (2025)  
  - 手册解读：[`./frontier/policy_consensus_multimodal_manipulation_2025.md`](./frontier/policy_consensus_multimodal_manipulation_2025.md)  
  - arXiv：`https://arxiv.org/pdf/2509.23468`  
  - 项目主页：`https://policyconsensus.github.io/`

### 0.3 物理方程视觉建模（Wave/Heat）

> 说明：用物理方程刻画“特征传播”，为视觉骨干提供可解释的建模偏置。

- WaveFormer: Frequency-Time Decoupled Vision Modeling with Wave Equation (2026)  
  - 手册解读：[`./frontier/waveformer_wave_equation_vision_2026.md`](./frontier/waveformer_wave_equation_vision_2026.md)  
  - arXiv：`https://arxiv.org/abs/2601.08602`

### 0.5 世界模型 / 视频世界模型（World Models / Video）

> 说明：这类路线的目标是把机器人决策从“直接出动作”推向“先想象再行动”——用可学习的世界模型做预测、规划与反事实评估。  
> 注意：落地的第一性约束往往不是画质，而是**物理一致性、可控性、成本与安全**。

- 视频世界模型综述（arXiv 2026）：Video Generation Models in Robotics - Applications, Research Challenges, Future Directions  
  - [手册笔记](./frontier/video_generation_models_in_robotics_survey_2026.md)  
  - 论文：`https://arxiv.org/abs/2601.07823`
- 1X World Model（视频世界模型 + 逆动力学 IDM）：先“想象”再“执行”（公司路线复盘）  
  - [手册笔记](./frontier/one_x_world_model.md)

### 1. 动作生成策略 (Action Generation)

#### 1.1 Diffusion 系列

##### Diffusion Policy (Chi et al., RSS 2023)
> **论文**: [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)
> **机构**: Columbia University

- **核心问题**: 解决传统 MSE 回归在多模态分布 (Multimodal Distribution) 下的平均值问题 (即"撞墙"问题)。
- **核心技术**: **DDPM (Denoising Diffusion Probabilistic Models)**。将动作生成建模为从高斯噪声中逐步去噪的过程。
- **Backbone**:
    - **CNN-based**: 1D Temporal CNN (类似 U-Net)，适合短时序。
    - **Transformer-based**: DiT (Diffusion Transformer)，适合长时序。
- **Action Space**: **连续空间 (Continuous)**。无离散化误差，精度极高。
- **Inference**: 迭代去噪。原始 DDPM 需 100 步，使用 **DDIM** 可加速至 10-15 步。
- **Deep Dive**:
    - **EBM 视角**: Diffusion 实际上是在学习能量地貌 (Energy Landscape)，相比 MSE 的单峰平均，它能捕捉多模态分布 (Multimodal Distribution)。
    - **Conditioning**: 通过 **FiLM** 层将语言/图像特征注入 U-Net。
- **Key Contribution**: 首次将生成式 AI (Generative AI) 引入机器人控制，完美解决了多解问题，并在高精度任务 (如穿针) 上表现卓越。

##### RDT-1B (Liu et al., 2024)
> **论文**: [RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation](https://arxiv.org/abs/2410.07864)
> **机构**: 清华大学 MARS Lab & ByteDance
> **详细解析**: [rdt.md](./rdt.md)

- **核心问题**: 证明机器人学习也存在 Scaling Law，大模型+大数据有效。
- **核心技术**: **DiT (Diffusion Transformer)**，十亿参数级扩散模型。
- **Backbone**: DiT 架构，可扩展到数十亿参数。
- **Action Space**: **连续空间**。
- **Key Contribution**: 首个十亿参数级机器人扩散基础模型，专为双臂操作优化，证明 Scaling Law 在机器人领域也成立。

##### RDT2 (RDT Team, 2026)
> **论文**: [RDT2: Exploring the Scaling Limit of UMI Data Towards Zero-Shot Cross-Embodiment Generalization](https://arxiv.org/abs/2602.03310)  
> **项目主页**: `https://rdt-robotics.github.io/rdt2/`  
> **手册解读**: [`./frontier/rdt2_umi_zero_shot_cross_embodiment_2026.md`](./frontier/rdt2_umi_zero_shot_cross_embodiment_2026.md)

- **核心问题**: 跨本体零样本部署与大规模真机数据成本瓶颈。  
- **核心技术**: **Residual VQ 动作 tokenization + Flow Matching 动作专家 + 单步蒸馏**。  
- **Action Space**: **连续空间**。  
- **Key Contribution**: 依托 UMI 数据规模化与硬件统一，验证 4U 条件下零样本泛化（原文口径，待核验）。

---

#### 1.2 Flow Matching 系列

##### π0 (Physical Intelligence, 2024)
> **论文**: [π0: A Generalist Robot Foundation Model](https://www.physicalintelligence.company/blog/pi0)
> **详细解析**: [pi0_flow_matching.md](./pi0_flow_matching.md) | [代码解析](./pi0_code_analysis.md)

- **核心问题**: 解决 VLM 推理速度慢、难以进行高频 (50Hz) 连续控制的问题。
- **核心技术**: **Flow Matching (流匹配)**。
- **Backbone**: **PaliGemma 3B** (Google 的轻量级 VLM)。
- **Action Space**: **连续空间 (Continuous)**。
    - 不同于 RT-2/OpenVLA 的离散 Token，Pi0 输出连续动作，避免了量化误差。
- **Inference**: 使用 ODE Solver (常微分方程求解器)。相比 Diffusion 的随机游走，Flow Matching 走直线，**1-10 步**即可生成高质量动作。
- **Deep Dive**:
    - **OT-CFM**: 基于 Optimal Transport 构造直线路径 (Wasserstein Geodesic)。
    - **ODE Solver**: 训练时学习向量场，推理时使用 **Euler** (极速) 或 **Heun** (高精) 求解。
- **Key Contribution**: 结合了 VLM 的语义理解和 Flow Matching 的高频精细控制，实现了"大脑"与"小脑"的统一。

##### π0.5 (Physical Intelligence, 2025)
> **核心定位**: **Open-World Explorer (开放世界探险家)**
> **详细解析**: [pi0_5_dissection.md](./pi0_5_dissection.md)

- **核心问题**: 解决机器人无法在从未见过的环境 (Open World) 中泛化的问题。
- **核心技术**: **Unified Model with Hierarchical Inference**。
- **架构创新**:
    - **Latent Thought**: 模型内部生成隐式的高层语义子任务 (Semantic Subtask)，再解码为底层动作。
    - **Hybrid Architecture**: 训练时使用 **FAST Tokenizer** (离散) 加速，推理时使用 **Flow Matching** (连续) 微调。
- **Data Strategy**: **Co-training**。混合 Robot Data (高质量) + Internet Videos (世界模型) + Simulation Data (长序列逻辑)。
- **Key Contribution**: 实现了跨形态 (Cross-Embodiment) 的 Zero-shot 迁移，并显著提升了长序列任务的成功率。

##### π0.6 (Physical Intelligence, 2025)
> **核心定位**: **Self-Improving Master (自我进化大师)**
> **详细解析**: [pi0_6_dissection.md](./pi0_6_dissection.md)

- **核心问题**: 如何超越人类示教的上限，实现极致的熟练度 (Proficiency)。
- **核心技术**: **Recap Algorithm (Offline RL)**。
- **架构升级**:
    - **5B Backbone**: 更强的语义理解。
    - **Action Expert**: 独立的高频动作生成模块 (小脑)，专门负责精细操作。
- **Recap 机制**:
    - 学习失败轨迹 (Failure Cases)，通过 Offline RL 抑制错误动作，奖励成功动作。
    - 实现了 **Data-Driven Self-Improvement**。
- **Key Contribution**: 证明了机器人可以通过自我复盘 (Recap) 在操作速度和鲁棒性上超越人类专家。

##### LingBot-VLA (Robbyant, 2026)
> **论文**: `https://arxiv.org/abs/2601.18692`  
> **代码**: `https://github.com/robbyant/lingbot-vla`  
> **深度解析（手册）**: [lingbot_vla_pragmatic_vla_foundation_model_2026.md](./lingbot_vla_pragmatic_vla_foundation_model_2026.md)

- **核心问题**: 把 VLA 从“能讲清楚”推到“能训得动/能复现吞吐/能部署”，强调真实大规模双臂数据下的工程可训练性。
- **核心技术**: **Flow Matching (流匹配)** + **VLM + Action Expert 融合**（共享 attention 的 token 拼接式融合）。
- **Backbone**: 可换底座（Qwen2.5-VL 或 PaliGemma 路线），便于做 ablation 与工程选型。
- **Action Space**: **连续空间 (Continuous)**（chunked action + padding 维度对齐）。
- **工程主张**: FSDP2、`torch.compile`、packing collator、分布式 checkpoint（dcp），以吞吐为中心指标组织训练栈。
- **可选分支**: Depth 表征对齐/蒸馏（训练侧额外 loss；推理侧可不依赖 depth 模型）。

---

#### 1.3 Tokenization 系列

##### RT-2 (Google DeepMind, 2023)
> **论文**: [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818)

- **核心问题**: 如何让机器人拥有互联网级别的语义理解能力 (泛化到未见过的物体/指令)。
- **核心技术**: **VLA (Vision-Language-Action)** = VLM + Action Tokens。
- **Backbone**: **PaLI-X (55B)** 或 **PaLM-E (12B)**。
- **Action Tokenization**:
    - **Uniform Discretization**: 将动作维度归一化并切分为 **256 个 Bins**。
    - **Text Mapping**: 将这些 Bins 映射为特殊的文本 Token (如 "1", "128")，与自然语言共享词表。
- **Training**: **Co-fine-tuning** (混合微调)。同时训练互联网 VQA 数据 (保持语义) 和机器人操作数据 (学习控制)。
- **Key Contribution**: 涌现出 **Semantic Reasoning** (语义推理) 能力。例如听到 "pick up the extinct animal" 能抓起恐龙玩具，尽管训练数据里只有 "pick up dinosaur"。

##### OpenVLA (Stanford, 2024)
> **论文**: [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246)

- **核心问题**: 复现 RT-2 的能力，但完全开源且高效。
- **核心技术**: **Parameter-Efficient Fine-Tuning (LoRA)**。
- **Backbone**:
    - **Language**: **Llama 2 7B**。
    - **Vision**: **SigLIP** (比 CLIP 更强的视觉编码器)。
    - **Projector**: 2-layer MLP (将视觉 Embedding 映射到语言空间)。
- **Action Output**:
    - 不同于 RT-2 直接输出文本，OpenVLA 使用专门的 **Action Head** (Linear Layer) 预测去离散化的动作 Token。
    - 依然是 **256-bin Discretization**。
- **Optimization**: 支持 **4-bit Quantization (QLoRA)**，使得 7B 模型可以在消费级显卡 (如 RTX 3090/4090) 上运行。
- **Key Contribution**: 提供了第一个性能接近闭源 SOTA 的开源 VLA 模型，并构建了完整的开源训练/部署生态。

##### FAST (Physical Intelligence, 2025)
> **论文**: [FAST: Efficient Action Tokenization for VLA Models (arXiv:2501.09747)](https://arxiv.org/abs/2501.09747)
> **详细解析**: [fast.md](./fast.md)

- **核心问题**: 传统动作 token 化方法（简单分桶）无法处理高频、灵巧的机器人操作。
- **核心技术**: **Frequency-space Action Sequence Tokenization (DCT + BPE)**。
- **工作原理**:
    - **DCT (离散余弦变换)**: 将时域动作序列转换到频域，只保留低频系数（压缩比 2.5:1）。
    - **BPE (字节对编码)**: 类似 GPT，将常见 DCT 系数组合合并为单个 token（压缩比 2.3:1）。
- **效果**: 一个 10 步动作序列从 70 个 token 压缩为 **2-3 个 token**。
- **FAST+**: 在 100 万+真实机器人数据上预训练的通用 tokenizer，跨平台泛化。
- **Key Contribution**: 使 OpenVLA 训练速度提升 **5 倍**，同时保持高频动作精度。

---

#### 1.4 其他动作生成方法

##### ACT (Zeng et al., 2023)
> **论文**: [ACT: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)
> **详细解析**: [act.md](./act.md)

- **核心问题**: 如何用低成本硬件实现精细双臂操作。
- **核心技术**: **CVAE (Conditional Variational Autoencoder)** + **Action Chunking**。
- **Backbone**: CNN-based encoder-decoder。
- **Action Space**: **连续空间**。
- **Key Contribution**: ALOHA 系统的核心算法，证明了 CVAE + 动作分块的有效性。

---

#### 1.5 Latent Action 系列 (潜在动作学习) 🆕

> **核心思想**: 从视频中学习"任务中心"的潜在动作表示，实现跨机器人泛化

```
┌─────────────────────────────────────────────────────────────────┐
│              传统 VLA vs Latent Action VLA                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   传统 VLA:                                                      │
│   Image + Language ──→ Specific Robot Action (7-DoF)            │
│                        └── 绑定特定机器人                        │
│                                                                 │
│   Latent Action VLA:                                            │
│   Video ──→ Latent Action ──→ Decoder ──→ Any Robot Action      │
│             (任务中心表示)      (可插拔)    (不同机器人)          │
│             └── 机器人无关 ─────┘                                │
│                                                                 │
│   优势:                                                         │
│   • 可利用海量互联网视频                                         │
│   • 潜在动作空间更易泛化                                         │
│   • 换机器人只需换 Decoder                                       │
└─────────────────────────────────────────────────────────────────┘
```

##### UniVLA (IJRR 2024)
> **论文**: [UniVLA: Learning to Act Anywhere with Task-centric Latent Actions](https://journals.sagepub.com/doi/full/10.1177/02783649241227559)
> **arXiv**: [2505.06111](https://arxiv.org/abs/2505.06111)

- **核心问题**: 如何利用视频数据训练跨机器人泛化的 VLA 策略。
- **核心技术**: **Task-centric Latent Actions (任务中心潜在动作)**。
- **架构设计**:
    - **Video Encoder**: 从视频序列提取视觉特征
    - **Latent Action Model**: 学习与机器人无关的"任务意图"表示
    - **Robot-specific Decoder**: 将潜在动作解码为具体机器人动作
- **训练数据**: 互联网视频 + 机器人演示数据
- **Key Contribution**: 
    - 首次在 IJRR 顶刊发表的潜在动作 VLA 框架
    - 在操作和导航任务上实现 SOTA
    - 有效的 Sim-to-Real 迁移

##### EvoVLA (2025)
> **论文**: [EvoVLA: Self-Evolving Vision-Language-Action Model](https://arxiv.org/abs/2511.16166)

- **核心问题**: 解决长时程任务中的"阶段幻觉"问题（模型报告进度 > 实际进度）。
- **核心技术**: **自进化框架 (Self-Evolving Framework)**。
- **三大创新**:
    - **SAR (Stage-Aligned Reward)**: 阶段对齐奖励，三元对比学习
    - **POE (Pose-Based Object Exploration)**: 基于姿态的探索（非原始像素）
    - **Long-Horizon Memory**: 选择性上下文保留 + 门控融合
- **性能**: Discoverse-L 基准上平均成功率 **69.2%** (+10.2%)，真机 **54.6%** (+11%)
- **Key Contribution**: 解决长时程操作中的阶段幻觉，自监督持续进化

##### MemoryVLA (2025)
> **论文**: [MemoryVLA: Memory-Augmented VLA for Long-Horizon Manipulation](https://arxiv.org/abs/2508.19236)

- **核心问题**: 长时程任务中的非马尔可夫性问题。
- **核心技术**: **感知-认知记忆系统 (Perception-Cognition Memory)**。
- **架构设计**:
    - **工作记忆**: 短期任务上下文
    - **感知记忆**: 历史视觉观测
    - **认知记忆**: 高层任务语义
- **Key Contribution**: 通过显式记忆机制处理长序列依赖

##### 其他 2025 相关工作

| 模型 | 核心创新 | 论文链接 |
|:---|:---|:---|
| **TTF-VLA** | Temporal Token Fusion，训练无关的多帧融合 | [arXiv:2508.19257](https://arxiv.org/abs/2508.19257) |
| **OmniVLA** | 多传感器感知（红外/雷达/麦克风） | [arXiv:2511.01210](https://arxiv.org/abs/2511.01210) |
| **MergeVLA** | 跨技能模型合并，知识迁移 | [arXiv:2511.18810](https://arxiv.org/abs/2511.18810) |
| **ContextVLA** | 多帧上下文压缩 | 2025 |
| **ReconVLA** | 隐式视觉注意力引导 | [arXiv:2508.10333](https://arxiv.org/abs/2508.10333) |

---

### 2. 训练方法 (Training Methods)

#### 2.1 BC (Behavior Cloning)

##### RT-2 - Co-fine-tuning
- 同时训练互联网 VQA 数据和机器人操作数据
- 保持 VLM 的语义能力，同时学习控制

##### OpenVLA - LoRA Fine-tuning
- 参数高效微调，降低计算成本
- 支持 4-bit 量化部署

##### π0 - Flow Matching Training
- 端到端训练，学习连续动作分布
- 结合 VLM 预训练和 Flow Matching

---

#### 2.2 RL (Reinforcement Learning)

##### GR-RL (ByteDance Seed, 2025)
> **详细解析**: [gr_rl_dissection.md](./gr_rl_dissection.md)

- **核心技术**: **Offline RL + Online RL** 三阶段训练
- **阶段 1**: Critic 筛选高质量演示数据
- **阶段 2**: 形态对称性增强（数据翻倍）
- **阶段 3**: 在线 RL 潜在空间探索（对齐训练-部署差异）
- **Key Contribution**: 首个完成真机穿鞋带任务的 VLA，78% 成功率

##### π*0.6 Recap (Physical Intelligence, 2025)
> **详细解析**: [pi0_6_dissection.md](./pi0_6_dissection.md#recap)

- **核心技术**: **Recap Algorithm (Offline RL)**
- **机制**: 从成功和失败轨迹中学习，抑制错误动作，奖励成功动作
- **Key Contribution**: 超越人类示教水平，实现自我进化

---

#### 2.3 混合训练方法

##### π0.5 - Co-training
- **数据混合**: Robot Data (高质量) + Internet Videos (世界模型) + Simulation Data (长序列逻辑)
- **效果**: 提升跨形态泛化和长序列任务成功率

---

### 3. 架构创新 (Architecture Innovations)

#### 3.1 单模型架构

##### RT-2 / OpenVLA / π0
- VLM Backbone + Action Head
- 统一架构，端到端训练

---

#### 3.2 双系统架构

##### Galaxea G0 (星海图智能, 2024)
> **论文**: [Galaxea Open-World Dataset and G0 Dual-System VLA Model (arXiv:2509.00576)](https://arxiv.org/abs/2509.00576)
> **详细解析**: [galaxea_g0.md](./galaxea_g0.md)

- **核心问题**: 单一 VLA 模型难以同时处理长时域任务的高层规划和低层控制。
- **核心技术**: **Dual-System Architecture (双系统架构)**。
- **架构设计**:
    - **G0-VLM**: 负责多模态规划和高层推理（大脑）。
    - **G0-VLA**: 负责细粒度执行和低层控制（小脑）。
- **训练策略**: **三阶段课程学习**
    1. 跨具身预训练（学习通用世界知识）
    2. 单具身预训练（适配特定机器人）← 核心阶段
    3. 任务后训练（精调复杂技能）
- **Galaxea Open-World Dataset**: 500+ 小时，50 个真实场景，统一具身（R1 Lite），精确子任务标注。
- **Key Contribution**: 在长时域移动操作任务上表现突出，泛化能力强，可解释性高（子任务可见）。

##### π0.6 - VLM + Action Expert
- **VLM (大脑)**: 5B 参数，负责语义理解
- **Action Expert (小脑)**: 轻量级 Transformer，负责高频精细控制
- **详细解析**: [pi0_6_dissection.md](./pi0_6_dissection.md#action-expert)

---

#### 3.3 层级架构

##### WALL-OSS (X², 2025)
> **详细解析**: [wall_oss.md](./wall_oss.md)

- **核心技术**: **Uni-CoT (统一思维链)** + **Dual Heads (Flow + FAST)**
- **架构**: 统一模型内部生成 CoT，双头输出连续和离散动作
- **Key Contribution**: 边想边动，长序列推理能力强

---

### 4. 应用场景 (Application Domains)

#### 4.1 操作任务 (Manipulation)
- RT-2, OpenVLA, π0, GR-RL, ACT

#### 4.2 导航任务 (Navigation)
- (待补充)

#### 4.3 灵巧手 (Dexterous Manipulation)
- GR-RL (穿鞋带), RDT-1B (双臂操作)

---

## 🏢 按公司/机构分类

### Google DeepMind
- **RT-2** (2023)
- **RT-1** (2022)

### Physical Intelligence
- **π0** (2024)
- **π0.5** (2025)
- **π0.6** (2025)
- **FAST** (2025)
- **Knowledge Insulation** (2024)

### ByteDance Seed
- **GR-RL** (2025)
- **RDT-1B** (2024) (与清华合作)

### RDT Team / 清华大学 MARS Lab
- **RDT2** (2026)

### Stanford
- **OpenVLA** (2024)
- **ACT** (2023)

### X² (自变量)
- **WALL-OSS** (2025)

### Galaxea AI
- **G0** (2024)

---

## 📅 按时间线

### 2023（早期探索）

#### Diffusion Policy (RSS 2023)
- 首次将生成式 AI 引入机器人控制
- [详细内容](#diffusion-policy-chi-et-al-rss-2023---a级)

#### RT-2 (ICRA 2023)
- VLA 范式确立
- [详细内容](#rt-2-google-deepmind-2023---s级)

#### ACT (2023)
- CVAE + 动作分块
- [详细内容](#act-zeng-et-al-2023---a级)

---

### 2024（爆发期）

#### OpenVLA (2024.06)
- 首个开源 SOTA VLA
- [详细内容](#openvla-stanford-2024---s级)

#### π0 (2024.10)
- Flow Matching + VLM
- [深度解析](./pi0_flow_matching.md)

#### RDT-1B (2024.10)
- 十亿参数扩散模型
- [详细解析](./rdt.md)

#### Galaxea G0 (2024.09)
- 双系统架构
- [详细解析](./galaxea_g0.md)

#### Knowledge Insulation (2024)
- 梯度隔离防遗忘
- [详细内容](#knowledge-insulation-physical-intelligence-2024---b级)

---

### 2025（最新进展）

#### π0.5 (2025.01)
- 开放世界泛化
- [深度解析](./pi0_5_dissection.md)

#### π0.6 (2025.11)
- 自我进化 (Recap)
- [深度解析](./pi0_6_dissection.md)

#### FAST (2025.01)
- DCT + BPE Tokenization
- [详细解析](./fast.md)

#### GR-RL (2025)
- 三阶段 RL 训练
- [深度解析](./gr_rl_dissection.md)

#### WALL-OSS (2025)
- Uni-CoT + 双头架构
- [详细解析](./wall_oss.md)

#### Policy Consensus (2025)
- 多模态专家策略共识，避免特征拼接稀疏噪声化
- [深度解读](./frontier/policy_consensus_multimodal_manipulation_2025.md)

---

### 2026（最新进展）

#### RDT2 (2026.02)
- UMI 数据规模化 + 跨本体零样本部署  
- [深度解读](./frontier/rdt2_umi_zero_shot_cross_embodiment_2026.md)

#### WaveFormer (2026.01)
- 波动方程视觉建模，频率-时间解耦，保留高频细节
- [深度解读](./frontier/waveformer_wave_equation_vision_2026.md)

---

## 🔧 训练技术 (Training Techniques)

### Knowledge Insulation (Physical Intelligence, 2024)
> **技术**: Pi0 的梯度隔离训练方法

- **核心问题**: VLA 微调时，新增的连续动作专家会破坏 VLM 的预训练语义知识（灾难性遗忘）。
- **核心技术**: **Gradient Isolation (梯度隔离)**。
- **工作原理**:
    - **VLM 分支**: 学习离散动作 token（保持语义理解）。
    - **动作专家分支**: 学习连续动作（使用 `.detach()` 阻止梯度回传到 VLM）。
- **效果**: VLM 的语义知识被"绝缘"保护，同时动作专家独立学习连续控制。
- **Key Contribution**: 防止灾难性遗忘，加速训练，提升泛化能力，为持续学习打好基础。

---

## 📊 总结对比表 (Summary Table)

| 特性 | Diffusion Policy | RT-2 | OpenVLA | π0 | π0.6 | GR-RL | WALL-OSS | Galaxea G0 | FAST |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **核心机制** | Denoising | Token Prediction | Token + LoRA | Flow Matching | Flow + Recap | **MoT + RL** | **Uni-CoT + Dual Heads** | **Dual-System** | **DCT + BPE** |
| **动作空间** | 连续 | 离散 (256) | 离散 (256) | 连续 | 连续 | 连续 | 连续 + 离散 | 连续 | **离散 (压缩)** |
| **Backbone** | CNN/ViT | PaLI-X (55B) | Llama 2 (7B) | PaliGemma (3B) | 5B VLM | **MoT (5B)** | VLM | **VLM + VLA** | **Tokenizer** |
| **推理速度** | 慢 (100 步) | 极慢 | 中等 | 快 (1-10 步) | 快 (1-10 步) | 中等 | 快/精 | 稍慢 (两阶段) | **极快 (5x)** |
| **语义能力** | 弱 | 极强 | 强 | 强 | 强 | 强 | **强 (CoT)** | **强 (分离 VLM)** | N/A |
| **训练方法** | BC | Co-fine-tuning | LoRA | Flow Training | **BC + Recap** | **BC + RL** | BC | Co-training | Tokenizer |
| **适用场景** | 精细操作 | 高层规划 | 通用操作 | 通用控制 | 通用+精细 | **高精度长时程** | 长序列推理 | **长时域移动操作** | **高频 Token 化** |
| **核心优势** | 多模态分布 | 语义涌现 | 开源生态 | 高效推理 | **自我进化** | **三阶段训练** | **统一思维链** | **分层解耦** | **压缩效率** |

---

## 📚 相关资源

- [📊 论文索引](./paper_index.md) - 多维度快速查找
- [🔬 模型深度解析](./README.md#-part-5-模型详解-model-zoo) - 独立深度解析文档
- [📖 理论总览](./README.md) - 返回理论目录

---

**最后更新**: 2026-01-25
[← Back to Theory](./README.md)
