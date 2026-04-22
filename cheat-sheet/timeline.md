# 关键论文时间线 (Paper Timeline · 2021-2026)

> 2026-04-21 更新 · 覆盖 CLIP 基础 → 2026 Q2 π0.7 / EgoScale / GR00T-N1.7 / WVA / VGA

---

## 🏛️ 1. 基础期 · 2021-2023：VLA 范式的前身

| 时间 | 论文/项目 | 机构 | 唯一重要的一件事 |
|:---:|------|------|------------------|
| **2021.01** | **CLIP** | OpenAI | 图文对齐——几乎所有 VLA 的视觉编码器起点 |
| **2022.04** | **Flamingo** | DeepMind | 视觉 × 语言融合范式（Perceiver Resampler + Gated Cross-Attention） |
| **2022.05** | **Gato** | DeepMind | "Generalist Agent"——Transformer 统一 text / image / action token |
| **2022.12** | **RT-1** | Google | 工业级标杆——TokenLearner + Transformer + 大规模真机 |
| **2023.03** | **ACT** | Stanford | 动作分块（action chunking）的开山 · ALOHA 标配 |
| **2023.03** | **PaLM-E** | Google | 多模态 LLM 吃机器人 state 做控制 |
| **2023.07** | **RT-2** | Google | **VLA 范式确立**——co-fine-tuning 让 VLM 的语义能迁到控制 |
| **2023.10** | **Octo** | Berkeley | Diffusion Policy 通用策略开源 |
| **2023.12** | **Diffusion Policy** | Columbia | 扩散生成动作的奠基工作（Cheng Chi 等） |

## 🔧 2. 扩张期 · 2024：开源 VLA 爆发

| 时间 | 论文/项目 | 机构 | 唯一重要的一件事 |
|:---:|------|------|------------------|
| **2024.06** | **OpenVLA** | Stanford | **开源 SOTA 第 1 代**——Llama 2 7B + SigLIP · LoRA 微调 · 社区标杆 |
| **2024.07** | **RDT** | 清华 | 1B 参数 diffusion transformer · 双臂操作 |
| **2024.10** | **π0** | Physical Intelligence | Flow Matching 动作头 + PaliGemma 3B · 开启高频控制路线 |
| **2024.12** | **RT-H** | Google | 动作层次化（language motion 中间层） |
| **2024.12** | **WALL-OSS** | X-Square | 双分支（Flow + FAST）+ CoT 推理 · [官方页](https://x2robot.com/en/research/68bc2cde8497d7f238dde690) |

## 🚀 3. 加速期 · 2025 H1：VLA 工程化

| 时间 | 论文/项目 | 机构 | 唯一重要的一件事 |
|:---:|------|------|------------------|
| **2025.01** | **FAST** | PI | 动作 tokenization 新方案（频域） |
| **2025.02** | **VLA-Touch** | — | 双层触觉反馈 |
| **2025.02** | **ICLR Scaling Law** (Oral) | — | 📎 [data-scaling-laws](https://data-scaling-laws.github.io/) · **多样性 > 数量**的实证 |
| **2025.03** | **OmniVTLA** | — | 视觉-触觉-语言-动作统一 |
| **2025.03** | **RoboEngine** | — | **+210%** ROI 的背景替换增强 · [arXiv:2503.18738](https://arxiv.org/abs/2503.18738) |
| **2025.04** | **π0.5** | PI | **Open-world Generalization** · 分层推理（高层规划 + 底层控制统一） |
| **2025.04** | **OpenVLA-OFT** | — | Online Fine-Tuning |
| **2025.05** | **SmolVLA** | HuggingFace | 小尺寸（450M）开源 VLA · 消费级 GPU 可跑 |
| **2025.06** | **RSS'25 RoboMIND 2.0** | — | **310K 轨迹** · 含 5K 失败案例 · 739 任务 |
| **2025.07** | **FAVLA** | — | Force-Adaptive Fast-Slow VLA · 接触操作 80.8% |

## 🔬 4. 深化期 · 2025 H2：从"跑通"到"可信"

| 时间 | 论文/项目 | 机构 | 唯一重要的一件事 |
|:---:|------|------|------------------|
| **2025.09** | **AgiBot World 2026** | OpenDriveLab | **1M+ 轨迹 · 2976 hrs** · 最大多模态 · IROS'25 Best Paper 提名（⚠️ CC BY-NC-SA 禁商用）|
| **2025.10** | **LIBERO-PRO** | — | 📎 [arXiv:2510.03827](https://arxiv.org/abs/2510.03827) · **LIBERO 90%→0.0%** 记忆化实证 |
| **2025.10** | **LIBERO-Plus** | — | 📎 [arXiv:2510.13626](https://arxiv.org/abs/2510.13626) · 扰动鲁棒性 |
| **2025.10** | **LIBERO-X** | — | 📎 [arXiv:2602.06556](https://arxiv.org/abs/2602.06556) · 更多扰动维度 |
| **2025.10** | **LingBot-VLA** | Robbyant | **20K hrs 真实数据** 预训练 · GM-100 真机 |
| **2025.11** | **π0.6 / π*0.6** | PI | **RL (RECAP) + 5B backbone** · Advantage Conditioned Policy · 迭代成功率爬升 |
| **2025.11** | **GR00T-N1.7** | NVIDIA | **跨形态基础模型** · 5 demos 起步 · 多 DoF 灵巧手 |
| **2025.11** | **VGA** | — | Vision-Geometry-Action · 挑战 VLM backbone |
| **2025.12** | **WVA** | — | **World-Value-Action** · 价值函数隐式规划 · 真机 75.6%（75.6% on dual-arm Piper） |

## 🌌 5. 前沿期 · 2026 Q1-Q2：新范式涌现

| 时间 | 论文/项目 | 机构 | 唯一重要的一件事 |
|:---:|------|------|------------------|
| **2026.02** | **EgoScale** | NVIDIA GEAR | 📎 [arXiv:2602.16710](https://arxiv.org/abs/2602.16710) · **20,854 hrs 人类视频** log-linear scaling (R²=0.9983) · 22-DoF 手 **+54%** |
| **2026.02** | **OXE-AugE** | — | **4.4M 轨迹** · OXE 3x 增强（CC BY 4.0 商用安全） |
| **2026.03** | **Spark 2.0** | World Labs | 3DGS web renderer 具身表征 |
| **2026.03** | **Latent Space Survey** | — | 潜空间作为具身基础模型的统一接口 |
| **2026.04** | **VLOA** | — | Embodied World Model 3D 点云轨迹 |
| **2026.04** | **HAMLET** | CMU | **历史感知 VLA**——解决即时映射缺陷 |
| **2026.04** | **HazardArena** | — | VLA 语义安全评估 · 填补执行成功率以外的维度 |
| **2026.04** | **π0.7** | PI | **Steerable + Compositional Generalization** · 5B 主体 + 14B BAGEL 世界模型旁路 · KI + FAST + RECAP strategy-metadata 蒸馏 · [深度](../theory/vla-core/pi0_7_steerable_compositional_generalization_2026.md) |
| **2026.04** | **DockAnywhere** | — | 移动操作数据增强（TAMP + 3D 点云编辑 · 0.02s/样本） |
| **2026.04** | **Danfei Xu 访谈**（To Summon a Sensorimotor Ghost） | GT / NVIDIA | **"人类数据 = 伪装的机器人数据"** · 非语言 System 2 · [深度](../theory/foundation/human_data_sensorimotor_ghost_danfei_xu_interview_2026.md) |

## 📜 综述（值得一读）

- **2025.01**: *Vision-Language-Action Models for Robotics: A Review Towards Real-World Applications* (IEEE Access)
- **2025.05**: *Vision-Language-Action Models: Concepts, Progress, Applications and Challenges*
- **2026.03**: *Latent Space Survey* — 前沿潜空间统一接口

---

## 🎯 学习路径建议（三条路线）

### A. 入门 → 工程落地
1. **CLIP → RT-1 → RT-2**（VLA 范式起源）
2. **OpenVLA**（开源代码必读）
3. **ACT / Diffusion Policy**（小规模能跑通）
4. **π0.5 / π0.6**（当前生产路线）

### B. 研究 → 跟踪前沿
1. **π 系列**（π0 → π0.5 → π0.6 → π0.7）
2. **EgoScale + Danfei Xu 系列**（人类数据路线）
3. **WVA / LIBERO-PRO**（架构 × benchmark 双线演进）
4. **HAMLET**（历史感知突破 = 即时映射的第一次认真反驳）

### C. 基础模型 × 跨形态
1. **GR00T-N1.7**（NVIDIA 路线）
2. **OXE + OXE-AugE**（数据基础）
3. **LingBot-VLA**（20K 小时真机路线）
4. **Pi 的跨形态 + 人类迁移**（["Emergence of Human to Robot Transfer"](https://faculty.cc.gatech.edu/~danfei/)）

## 📈 VLA 演进的 3 个关键转折点

1. **2023 RT-2**：VLA 范式确立（VLM + 动作）
2. **2024 π0**：Flow Matching 高频控制路线（脑子慢手快矛盾的第一个答案）
3. **2026 EgoScale**：人类视频 scaling law 实证（数据源从"遥操作"扩到"人类自然行为"）

---

[← Back to Cheat Sheet](./README.md)
