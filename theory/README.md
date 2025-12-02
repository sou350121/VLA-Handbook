# 理论基础 (Theory)

本模块涵盖了机器 VLA 算法岗面试所需的核心理论知识，从基础架构到前沿算法。

# VLA 理论与核心算法 (Theory & Algorithms)

本章节深入探讨 Vision-Language-Action (VLA) 模型的理论基础、核心算法以及前沿模型架构。

> **学习建议**: 建议按照以下 **"数据 -> 架构 -> 算法 -> 模型"** 的逻辑顺序进行学习。
> 🤪 **[太长不看？点这里看人话版 (Funny Version)](./README_FUN.md)**

---

## 📚 Part 1: Foundations (基础基石)
*万丈高楼平地起，数据与动作空间是 VLA 的根基。*

#### Part 1 Index
- **Data Processing** (`theory/data.md`): data formats, loaders, action normalization.
- **Spatial Math** (`theory/spatial_math.md`): coordinate systems, rotation representations.
- **Evaluation** (`theory/evaluation.md`): simulation benchmarks, true-world metrics.
- **Co-training** (`theory/co_training.md`): mixing internet data with robot actions.
- **Action Representations** (`theory/action_representations.md`): continuous vs discrete action spaces.

### 1. 数据与空间 (Data & Space)
- **[数据处理 (Data Processing)](./data.md)**
    - 主流格式对比 (RLDS vs LeRobot vs HDF5)。
    - PyTorch 训练流水线与数据加载。
    - 数据收集工具链 (VR vs Leader-Follower)。
- **[空间智能 (Spatial Intelligence)](./spatial_math.md)**
    - 坐标系 (Base vs Camera vs End-effector)。
    - 旋转表示 (Quaternion vs Euler vs 6D Rotation)。

### 2. 动作与策略 (Action & Strategy)
- **[动作空间 (Action Representations)](./action_representations.md)**
    - 连续控制 (Continuous) vs 离散 Token (Discrete)。
    - 相对控制 (Delta) vs 绝对控制 (Absolute)。
- **[联合训练 (Co-training)](./co_training.md)**
    - 为什么需要混合互联网数据？(防止灾难性遗忘)。
    - 实施策略：数据配比与 Loss Masking。

### 3. 评估与迭代 (Evaluation)
- **[评估体系 (Evaluation Protocols)](./evaluation.md)**
    - Simulation Benchmarks (CALVIN, SIMPLER)。
    - 真机评估指标 (Success Rate, Interventions)。
    - Checkpoint Selection (Loss vs Success Rate)。

## 🎓 Part 2: ML Fundamentals (机器学习基础)
*掌握 VLA 背后的核心机器学习技术。*

#### Part 2 Index
- **Multimodal Models** (`theory/multimodal_models.md`): VLM backbones, fusion strategies, projectors.
- **Self-Supervised Learning** (`theory/self_supervised_learning.md`): contrastive losses, MAE, video SSL.
- **Transfer Learning** (`theory/transfer_learning.md`): cross-embodiment, Sim-to-Real, PEFT.
- **Knowledge Distillation** (`theory/knowledge_distillation.md`): logits, feature, action trajectory distillation.
- **Reinforcement Learning** (`theory/reinforcement_learning.md`): PPO/SAC, Offline RL, Recap.
- **Chain-of-Thought Reasoning** (`theory/chain_of_thought.md`): CoT, ReAct, structured reasoning.

> **ASCII Cheat Sheet**: 所有的 ASCII 图都集中在 [`theory/ascii_cheatsheet.md`](./ascii_cheatsheet.md)，便于复习关键架构和流程。

### 多模态与表示学习 (Multimodal & Representation Learning)
- **[多模态模型 (Multimodal Models)](./multimodal_models.md)** [New]: VLM 架构、融合策略 (Early/Mid/Late Fusion)、视觉编码器选择。
- **[自监督学习 (Self-Supervised Learning)](./self_supervised_learning.md)** [New]: 对比学习 (CLIP/SimCLR)、掩码预测 (MAE)、R3M。

### 迁移与适应 (Transfer & Adaptation)
- **[迁移学习 (Transfer Learning)](./transfer_learning.md)** [New]: 跨形态迁移、Sim-to-Real、域适应、LoRA 微调。
- **[知识蒸馏 (Knowledge Distillation)](./knowledge_distillation.md)** [New]: 软标签蒸馏、特征蒸馏、VLA 模型压缩。

### 学习范式 (Learning Paradigms)
- **[强化学习 (Reinforcement Learning)](./reinforcement_learning.md)** [New]: PPO/SAC 算法、Offline RL、Recap 算法、奖励设计。
- **[思维链推理 (Chain-of-Thought)](./chain_of_thought.md)** [New]: CoT 在 VLA 中的应用、ReAct、分层规划。

## 🧠 Part 3: Architecture & Algorithms (架构与算法)
*理解模型是如何"思考"和"决策"的。*

#### Part 3 Index
- **VLA Architectures** (`theory/vla_arch.md`): VLM + action head design principles.
- **Transformer vs CNN** (`theory/transformer_vs_cnn.md`): why Transformers dominate embodied AI.
- **Policy Generation**: `act.md`, `diffusion_policy.md`, `rdt.md`, `pi0_flow_matching.md`, `fast.md`.
- **Efficiency**: `flash_attention.md`, `peft_lora.md`, `quantization_theory.md`.
### 核心架构
- **[VLA 架构概览 (VLA Architectures)](./vla_arch.md)**: VLM Backbone + Action Head 的主流设计范式。
- **[Transformer vs CNN](./transformer_vs_cnn.md)**: 为什么 Transformer 统治了机器人学习？

### 生成策略 (Policy Generation)
- **[ACT (Action Chunking with Transformers)](./act.md)**: 基于 CVAE 的动作分块预测，ALOHA 项目核心算法。
- **[Diffusion Policy](./diffusion_policy.md)**: 基于扩散模型的动作生成，解决多模态分布问题。
- **[RDT (Robotics Diffusion Transformer)](./rdt.md)**: 十亿参数级扩散基础模型，专为双臂操作优化。
- **[Flow Matching (π0)](./pi0_flow_matching.md)**: 比 Diffusion 更快、更稳定的生成模型，π0 的核心。
- **[FAST (Action Tokenization)](./fast.md)**: 基于频率空间 (DCT) 的动作 Tokenization 技术。

### 效率优化 (Efficiency)
- **[Flash Attention](./flash_attention.md)**: 如何解决长序列 Transformer 的计算瓶颈。
- **[高效微调 (PEFT & LoRA)](./peft_lora.md)**: LoRA / QLoRA 的数学原理，如何用 QLoRA (~6GB) 微调 7B 模型。
- **[量化理论 (Quantization Theory)](./quantization_theory.md)**: Symmetric vs Asymmetric, Per-Channel vs Per-Tensor, AWQ 原理。

## 🚀 Part 4: Advanced Topics (进阶专题)
*解决特定场景下的难题。*

#### Part 4 Index
- **Knowledge Insulation** (`theory/knowledge_insulation.md`): gradient isolation strategies.
- **Tactile VLA** (`theory/tactile_vla.md`): integrating tactile sensing with VLA.

- **[知识绝缘 (Knowledge Insulation)](./knowledge_insulation.md)**: 如何在微调时保护 VLM 的通用常识？
- **[触觉感知 (Tactile VLA)](./tactile_vla.md)**: 引入触觉模态，实现更精细的操作 (e.g., 盲盒摸索)。

## 🦁 Part 5: Model Zoo (模型详解)
*SOTA 模型的深度剖析与实战案例。*

#### Part 5 Index
- **Literature Review** (`theory/literature_review.md`): chronological model summaries.
- **Physical Intelligence (π0 系列)**: `pi0_5_dissection.md`, `pi0_6_dissection.md`.
- **WALL-OSS** (`theory/wall_oss.md`): Uni-CoT dual branch architecture.
- **Galaxea G0** (`theory/galaxea_g0.md`): dual system (VLM + VLA) deep dive.

> **[文献综述 (Literature Review)](./literature_review.md)**: **(必读)** VLA 发展史与主流模型全景图。

- **Physical Intelligence (π0 系列)**
    - **[π0.5 解析](./pi0_5_dissection.md)**: Flow Matching + VLA 的早期探索。
    - **[π0.6 解析](./pi0_6_dissection.md)**: 性能更强的迭代版本。
- **X Square (自变量)**
    - **[WALL-OSS](./wall_oss.md)**: 基于 Uni-CoT 的通用具身大模型。
- **Galaxea AI (星海图)**
    - **[Galaxea G0](./galaxea_g0.md)**: 独特的"小脑+大脑"双系统架构。

---
[← Back to Root](../README.md)

## 学习建议
- **初学者**: 先阅读 **Part 1 基础基石** 部分，理解 VLA 的基本范式 (Tokenization, Co-fine-tuning)。
- **补基础**: 如果 ML 基础不扎实，重点学习 **Part 2 机器学习基础**，特别是多模态、自监督学习和强化学习。
- **进阶**: 深入 **Part 3 架构与算法**，掌握 ACT、Diffusion 和 Flow Matching 的数学原理。
- **前沿**: 关注 **Part 4 进阶专题** 和 **Part 5 模型详解**，特别是 Pi 系列和触觉 VLA，这是大厂面试的差异化竞争点。

## 推荐阅读论文

### VLA 核心
- [RT-1: Robotics Transformer for Real-World Control at Scale](https://arxiv.org/abs/2212.06817)
- [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818)
- [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246)
- [Pi0: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.xxxxx)

### 策略学习
- [ACT: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)
- [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)
- [RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation](https://arxiv.org/abs/2410.07864)

### 机器学习基础
- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [MAE: Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
- [PPO: Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
