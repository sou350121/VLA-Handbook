# 关键论文时间线 (Key Paper Timeline)

本表梳理了 VLA 领域从多模态基础到最新具身智能模型的演进路线。

## 1. 经典必读 (Classic)
| 时间 | 论文 | 机构 | 核心贡献 |
| :--- | :--- | :--- | :--- |
| **2021.01** | **CLIP** | OpenAI | 奠定了图文对齐的基础，几乎所有 VLA 的视觉编码器都源于此。 |
| **2022.04** | **Flamingo** | DeepMind | 提出了视觉-语言在大模型中的融合方式 (Perceiver Resampler, Gated Cross-Attention)。 |
| **2022.05** | **Gato** | DeepMind | "A Generalist Agent". 证明了 Transformer 可以统一处理文本、图像和动作 Token。 |
| **2022.12** | **RT-1** | Google | **工业界标杆**。TokenLearner + Transformer，大规模真机数据训练。 |
| **2023.03** | **PaLM-E** | Google | Embodied Multimodal Language Model. 将机器人状态作为文本输入。 |
| **2023.07** | **RT-2** | Google | **VLA 范式确立**。Co-fine-tuning，证明了 VLM 的语义能力可以迁移到机器人控制。 |
| **2023.10** | **Octo** | Berkeley | 基于 Diffusion Policy 的开源通用策略模型。 |

## 2. 近半年热点 (Recent - 2024/2025)
> [!TIP]
> 面试加分项：熟悉最新的开源模型和 Pi 公司的进展。

| 时间 | 论文/项目 | 机构 | 核心贡献 |
| :--- | :--- | :--- | :--- |
| **2024.06** | **OpenVLA** | Stanford | **开源 SOTA**。基于 Llama 2 + SigLIP，支持量化部署，社区活跃度极高。 |
| **2024.10** | **π0 (Pi-Zero)** | Physical Intelligence | 强调物理世界理解的基础模型，Flow Matching Action Decoder。**已开源**。 |
| **2024.12** | **MultiBotGPT** | - | 多机器人协作控制，LLM 作为调度中心。 |
| **2024.12** | **WALL-OSS** | X-Square-Robot | [WALL-OSS: Igniting VLMs toward the Embodied Space](https://x2robot.com/en/research/68bc2cde8497d7f238dde690). 双动作分支 (Flow + FAST), COT 推理。**已开源**。 |
| **2025.02** | **VLA-Touch** | - | [VLA-Touch: Enhancing Generalist Robot Policies](https://arxiv.org/abs/2502.xxxxx). 双层反馈机制。 |
| **2025.03** | **OmniVTLA** | - | [OmniVTLA: A Unified Vision-Tactile-Language-Action Model](https://arxiv.org/abs/2503.xxxxx). 视触觉统一模型。 |
| **2025.04** | **π0.5** | Physical Intelligence | **Open-world Generalization**。分层推理 (Hierarchical Inference)，统一高层规划与底层控制。 |
| **2025.04** | **OpenVLA-OFT** | - | Online Fine-Tuning。提升了 OpenVLA 在新任务上的适应速度。 |
| **2025.11** | **π0.6 / π*0.6** | Physical Intelligence | **RL (Recap) + 5B Backbone**。引入 Offline RL 自我进化，长序列任务性能翻倍。 |
| **2026.04** | **π0.7** | Physical Intelligence | **Steerable + Compositional Generalization**。5B 主体（4B Gemma 3 VLM + 400M MEM 视觉编码器 + 860M flow matching action expert）+ 14B BAGEL 世界模型旁路；KI + FAST + RECAP strategy-metadata 蒸馏；主模型 1×H100/38ms，world model 4×H100/1.25s per subgoal。详见 [深度解读](../theory/vla-core/pi0_7_steerable_compositional_generalization_2026.md)。 |

## 3. 综述 (Surveys)
- **2025.01**: *Vision-Language-Action Models for Robotics: A Review Towards Real-World Applications* (IEEE Access)
- **2025.05**: *Vision-Language-Action Models: Concepts, Progress, Applications and Challenges*

## 学习路径建议
1. **入门**: CLIP -> RT-1 -> RT-2
2. **实战**: OpenVLA (代码必读)
3. **进阶**: Octo (Diffusion Policy), Pi Models (物理理解)


---
[← Back to Cheat Sheet](./README.md)
