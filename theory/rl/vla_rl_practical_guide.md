# VLA+RL 实战教程：架构、算法与工具链 (Practical VLA+RL Guide)

> **目标**：把 VLA+RL 从“概念”变成“可训练、可评估、可落地”的路径图。  
> **内容**：Transformer VLA 架构、核心 RL 算法、仿真/评测工具链、奖励与安全难点、模型融合范式。

---

## 0. 速览：VLA+RL 全链路

```
Transformer VLA (OpenVLA / pi0.5 / GR00T)
   └─ BC warmstart (先能动)
RL fine-tune (PPO / SAC / TD3 / Q-learning)
   └─ Reward shaping + Critic guidance + Policy lifting
评测基准 (LIBERO) + 训练 infra (RLinf / SimpleVLA-RL)
   └─ 回归测试 + 小规模真机校准
```

```
ASCII 图：VLA+RL 训练-评测主干逻辑

 数据/任务定义
 (Demo + Task Suite: LIBERO)
            |
            v
  +-------------------+          +----------------------+
  |   BC Warmstart    | -------> |    π_base (VLA)      |
  | (SFT / BC init)   |          | (OpenVLA/pi0.5/GR00T)|
  +---------+---------+          +----------+-----------+
            |                               |
            |                               v
            |                    +----------------------+
            |                    |  Rollout (RLinf)     |
            |                    |  + SimpleVLA-RL loop |
            |                    +----------+-----------+
            |                               |
            |                               v
            |     +--------------------+  trajectories  +------------------+
            +---->| Reward Shaping     |--------------->| Critic Q/V       |
                  | (ORM + PRM)        |                | (value guidance) |
                  +--------------------+                +---------+--------+
                                                                 |
                                                                 v
                                                      +------------------+
                                                      | Policy Update    |
                                                      | (PPO/SAC/TD3/Q) |
                                                      +---------+--------+
                                                                |
                                                                v
                                                      +------------------+
                                                      | Policy Lifting   |
                                                      | (reweight/distill)|
                                                      +---------+--------+
                                                                |
                                                                v
                                                      +------------------+
                                                      |  Eval (LIBERO)   |
                                                      |  fixed protocol  |
                                                      +------------------+
```

---

## 1. Transformer VLA 架构（OpenVLA / pi0.5 / GR00T）

**共同点**：
- 视觉编码器 + 语言编码器 + 动作头  
- 动作输出是连续或离散 token，通常带 chunking  
- 训练上先 BC，再 RL 提升上限与长时序稳定性

**差异点（工程视角）**：
| 模型 | 动作头 | 训练路径 | 典型优势 | 参考 |
|---|---|---|---|---|
| OpenVLA | Action token / head | SFT + 可选 RL | 开源基线、生态好 | `theory/vla_arch.md`, `question-bank/openvla_finetuning.md` |
| pi0.5 | Flow + 隐式推理 | Co-training + Flow | 推理快、长时序 | `theory/pi0_5_dissection.md` |
| GR00T-N1.x | 双系统 + DiT | 多阶段训练 | 系统化工程落地 | `theory/gr00t_n1_6.md` |

**学习入口**：
- 架构总览：`theory/vla_arch.md`  
- 研究主线：`theory/vla_research_mainline.md`  
- OpenVLA 实战：`question-bank/openvla_finetuning.md`

---

## 2. 强化学习算法：PPO / SAC / TD3 / Q-learning

| 算法 | 类型 | 适合动作空间 | 优点 | 风险 |
|---|---|---|---|---|
| PPO | On-policy Actor-Critic | 连续/离散 | 稳定、工程友好 | 采样成本高 |
| SAC | Off-policy Actor-Critic | 连续 | 样本效率高 | 超参敏感 |
| TD3 | Off-policy Actor-Critic | 连续 | 抑制 Q 过估计 | 对探索敏感 |
| Q-learning / DQN | Value-based | 离散 | 简洁、可解释 | 难扩展到连续 |

**参考**：详细算法与代码见 `theory/reinforcement_learning.md`

---

## 3. 高阶工具链：LIBERO / RLinf / SimpleVLA-RL

这三者分工不同，但可以组合成稳定流水线：

- **LIBERO**：任务基准与终身学习评测  
- **RLinf**：训练基础设施（rollout、调度、评估回归）  
- **SimpleVLA-RL**：面向 VLA 的 RL 训练框架

**学习入口**：
- 统一讲解：`deployment/simulation_benchmarks_and_tools.md`  
- 仿真选型：`deployment/simulation_environments.md`

---

## 4. 强化学习的难点（ORM/PRM、鲁棒性、安全）

已在 `theory/reinforcement_learning.md` 深度展开：
- **ORM vs PRM**：终局奖励 vs 过程奖励  
- **鲁棒性**：噪声/分布偏移/仿真到真机  
- **安全约束**：CMDP / Lagrangian / Action Projection / Shielding

**学习入口**：`theory/reinforcement_learning.md`

---

## 5. 模型融合范式（Reward Shaping / Critic Guidance / Policy Lifting）

**核心想法**：把多信号融合成“可控训练目标”。

- **Reward Shaping**：ORM + PRM 融合（避免 reward hacking）  
- **Critic Guidance**：用 Q/V 引导动作选择或行为克隆加权  
- **Policy Lifting**：用 critic/规划器“抬升”策略再蒸馏

**学习入口**：`theory/reinforcement_learning.md`（6.3 小节）

---

## 6. 推荐学习顺序（可直接照做）

1) **架构入口**：`theory/vla_arch.md`  
2) **算法细节**：`theory/reinforcement_learning.md`  
3) **工具链**：`deployment/simulation_benchmarks_and_tools.md`  
4) **模型案例**：`theory/pi0_5_dissection.md` + `theory/gr00t_n1_6.md`  
5) **实战脚手架**：`question-bank/openvla_finetuning.md`

---

## 参考索引
- 强化学习总览：`theory/reinforcement_learning.md`  
- VLA 架构：`theory/vla_arch.md`  
- VLA 研究主线：`theory/vla_research_mainline.md`  
- RL 训练基础设施：`theory/rl/rlinf_vla_rl_training.md`  
- 仿真与工具链：`deployment/simulation_benchmarks_and_tools.md`

---
[← Back to Theory](./README.md)
