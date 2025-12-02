# 🧠 VLA 理论与核心算法

> **Vision-Language-Action** 模型的理论基础、核心算法与前沿架构。

```
┌─────────────────────────────────────────────────────────────────────┐
│                        📖 学习路线图                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Part 1          Part 2          Part 3          Part 4 & 5       │
│   ┌─────┐        ┌─────┐         ┌─────┐         ┌─────┐           │
│   │基础 │ ────▶  │ ML  │ ──────▶ │架构 │ ──────▶ │前沿 │           │
│   │基石 │        │基础 │         │算法 │         │模型 │           │
│   └─────┘        └─────┘         └─────┘         └─────┘           │
│   数据/空间       多模态/RL       Diffusion       π0/G0            │
│   动作/评估       迁移/蒸馏       Flow/FAST       WALL-OSS         │
│                                                                     │
│   ⏱️ ~2天         ⏱️ ~3天         ⏱️ ~3天         ⏱️ ~2天            │
└─────────────────────────────────────────────────────────────────────┘
```

| 🎯 快捷入口 | |
|:---|:---|
| 🤪 **[人话版 (看不下去八股文？)](./README_FUN.md)** | 用类比讲清楚核心概念 |
| 📊 **[ASCII 图鉴](./ascii_cheatsheet.md)** | 一页纸复习所有架构图 |
| 📚 **[文献综述](./literature_review.md)** | VLA 发展史全景图 |

---

## 📚 Part 1: 基础基石 (Foundations)

> *万丈高楼平地起，数据与动作空间是 VLA 的根基。*

| 主题 | 文件 | 核心内容 |
|:-----|:-----|:---------|
| 📦 **数据处理** | [`data.md`](./data.md) | RLDS vs LeRobot vs HDF5、数据加载流水线 |
| 🧭 **空间智能** | [`spatial_math.md`](./spatial_math.md) | 坐标系变换、四元数 vs 欧拉角 vs 6D Rotation |
| 🎮 **动作空间** | [`action_representations.md`](./action_representations.md) | 连续 vs 离散、Delta vs Absolute |
| 🔄 **联合训练** | [`co_training.md`](./co_training.md) | 防止灾难性遗忘、Loss Masking |
| 📝 **评估体系** | [`evaluation.md`](./evaluation.md) | CALVIN/SIMPLER、真机成功率 |

---

## 🎓 Part 2: 机器学习基础 (ML Fundamentals)

> *掌握 VLA 背后的核心 ML 技术，补齐知识短板。*

| 主题 | 文件 | 核心内容 |
|:-----|:-----|:---------|
| 🔮 **多模态模型** | [`multimodal_models.md`](./multimodal_models.md) | VLM 架构、Early/Mid/Late Fusion、SigLIP vs CLIP |
| 🎯 **自监督学习** | [`self_supervised_learning.md`](./self_supervised_learning.md) | 对比学习 (InfoNCE)、MAE、R3M |
| ✈️ **迁移学习** | [`transfer_learning.md`](./transfer_learning.md) | 跨形态迁移、Sim-to-Real、Domain Randomization |
| 📝 **知识蒸馏** | [`knowledge_distillation.md`](./knowledge_distillation.md) | 软标签、Temperature、VLA 压缩 |
| 🎮 **强化学习** | [`reinforcement_learning.md`](./reinforcement_learning.md) | PPO/SAC、Offline RL、Recap 算法 |
| 💭 **思维链** | [`chain_of_thought.md`](./chain_of_thought.md) | CoT/ReAct、Uni-CoT、分层规划 |

---

## 🧠 Part 3: 架构与算法 (Architecture & Algorithms)

> *理解模型是如何"思考"和"决策"的。*

### 🏗️ 核心架构

| 主题 | 文件 | 核心内容 |
|:-----|:-----|:---------|
| 🏛️ **VLA 架构** | [`vla_arch.md`](./vla_arch.md) | VLM Backbone + Action Head 设计范式 |
| ⚔️ **Transformer vs CNN** | [`transformer_vs_cnn.md`](./transformer_vs_cnn.md) | 为什么 Transformer 统治机器人学习 |

### 🎯 动作生成策略 (Policy Generation)

| 算法 | 文件 | 一句话总结 |
|:-----|:-----|:---------|
| **ACT** | [`act.md`](./act.md) | CVAE + 动作分块，ALOHA 核心 |
| **Diffusion Policy** | [`diffusion_policy.md`](./diffusion_policy.md) | 扩散去噪，解决多模态分布 |
| **RDT** | [`rdt.md`](./rdt.md) | 十亿参数扩散模型，双臂操作 |
| **Flow Matching** | [`pi0_flow_matching.md`](./pi0_flow_matching.md) | 比 Diffusion 快 5x，π0 核心 |
| **FAST** | [`fast.md`](./fast.md) | DCT 频域 Tokenization |

### ⚡ 效率优化 (Efficiency)

| 主题 | 文件 | 核心内容 |
|:-----|:-----|:---------|
| 🚀 **Flash Attention** | [`flash_attention.md`](./flash_attention.md) | Tiling + 重计算，显存 O(N²)→O(N) |
| 🔧 **PEFT & LoRA** | [`peft_lora.md`](./peft_lora.md) | 低秩分解，QLoRA ~6GB 微调 7B |
| 📉 **量化理论** | [`quantization_theory.md`](./quantization_theory.md) | INT8/INT4、AWQ 原理 |

---

## 🚀 Part 4: 进阶专题 (Advanced Topics)

> *解决特定场景下的难题，面试差异化竞争点。*

| 主题 | 文件 | 核心内容 |
|:-----|:-----|:---------|
| 👁️ **视觉感知技术** | [`perception_techniques.md`](./perception_techniques.md) | 检测/跟踪/Occupancy/BEV/位姿估计 |
| 🧭 **运动规划** | [`motion_planning.md`](./motion_planning.md) | RRT/PRM、TrajOpt、MoveIt & cuRobo |
| 📡 **状态估计** | [`state_estimation.md`](./state_estimation.md) | EKF/UKF、粒子滤波、IMU+视觉融合 |
| 🛰️ **点云 & SLAM** | [`pointcloud_slam.md`](./pointcloud_slam.md) | 点云语义、配准、Visual/LiDAR SLAM |
| 🤖 **抓取算法 & 仿真** | [`grasp_algorithms.md`](./grasp_algorithms.md) | DexGraspNet/GraspGF、Isaac Sim/SAPIEN 🆕 |
| 🛡️ **知识绝缘** | [`knowledge_insulation.md`](./knowledge_insulation.md) | 微调时保护 VLM 通用常识 |
| 🖐️ **触觉 VLA** | [`tactile_vla.md`](./tactile_vla.md) | GelSight/DIGIT，盲盒操作 |

---

## 🦁 Part 5: 模型详解 (Model Zoo)

> *SOTA 模型的深度剖析，面试必考。*

### 📖 综述

| 文件 | 内容 |
|:-----|:-----|
| 📚 **[文献综述](./literature_review.md)** | **(必读)** RT-1/2 → OpenVLA → π0 发展脉络 |

### 🔬 模型深度解析

| 公司 | 模型 | 文件 | 核心亮点 |
|:-----|:-----|:-----|:---------|
| **Physical Intelligence** | π0.5 | [`pi0_5_dissection.md`](./pi0_5_dissection.md) | Flow Matching + 隐式推理 |
| | π0.6 | [`pi0_6_dissection.md`](./pi0_6_dissection.md) | Recap 自我进化 + Action Expert |
| **X² (自变量)** | WALL-OSS | [`wall_oss.md`](./wall_oss.md) | Uni-CoT 边想边动 |
| **Galaxea AI** | G0 | [`galaxea_g0.md`](./galaxea_g0.md) | 大脑+小脑双系统 |

---

## 🎯 学习建议

```
┌─────────────────────────────────────────────────────────────────────┐
│  👤 你是谁？                    📖 建议路线                          │
├─────────────────────────────────────────────────────────────────────┤
│  🌱 VLA 新手                    Part 1 → Part 3 (ACT/Diffusion)     │
│  📚 ML 基础薄弱                  Part 2 (重点: 多模态、RL)            │
│  🔧 想做工程落地                 Part 3 效率优化 + Part 5 OpenVLA     │
│  🎓 准备大厂面试                 全部 + Part 4 (差异化竞争点)          │
│  ⏰ 只有 1 天                   README_FUN.md + 文献综述             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📄 推荐论文

<details>
<summary><b>VLA 核心 (点击展开)</b></summary>

- [RT-1: Robotics Transformer for Real-World Control at Scale](https://arxiv.org/abs/2212.06817)
- [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818)
- [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246)
- [π0: A Vision-Language-Action Flow Model for General Robot Control](https://www.physicalintelligence.company/blog/pi0)

</details>

<details>
<summary><b>策略学习 (点击展开)</b></summary>

- [ACT: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)
- [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)
- [RDT-1B: A Diffusion Foundation Model for Bimanual Manipulation](https://arxiv.org/abs/2410.07864)

</details>

<details>
<summary><b>机器学习基础 (点击展开)</b></summary>

- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [MAE: Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Chain-of-Thought Prompting Elicits Reasoning in LLMs](https://arxiv.org/abs/2201.11903)
- [PPO: Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

</details>

---

[← 返回主页](../README.md)
