# RLinf：面向 Embodied / Agentic AI 的 RL 训练基础设施（以及它对 VLA+RL 的意义）

> 参考：项目主页与 README 综述见 [RLinf GitHub](https://github.com/RLinf/RLinf)。

---

## 1) 一句话定位：它解决什么问题？

**RLinf 不是一个“新 RL 算法”，而是一个“把 RL 训练做成可扩展生产线”的基础设施（infra）**：帮你把

- 并行 rollout（采样）
- 数据/统计/回放
- 分布式训练与调度
- 评估、回归测试（CI / E2E）

这些 **最耗工程成本、最容易不稳定** 的环节做成可复用组件，从而更快、更稳地做大规模 RL 训练与迭代。

对具身（embodied）来说，痛点尤其集中在 **rollout 侧吞吐、环境多样性、以及“仿真↔真机”的一致性与可复现**。

---

## 2) 为什么它跟 VLA 相关？（VLA+RL 的典型瓶颈）

当你把 VLA policy 从 “BC 能动” 推到 “RL 更强更稳”，你会遇到两类瓶颈：

- **算法瓶颈**：PPO/Offline RL/在线 fine-tune 怎么做更稳定（KL-to-base、adv/rew 归一化、奖励设计、安全约束）
- **系统瓶颈（更常见）**：rollout 慢、评估不可复现、训练跑一天结果漂移、不同环境/控制频率导致对齐困难

RLinf 的价值主要在第二类：把“系统层不确定性”降到最低，让你能把精力放回“策略改进”本身。

并且 RLinf 的 roadmap 明确提到要支持：

- VLM 训练
- 更多 VLA（例如 WALL-OSS）
- real-world RL

这些方向都与 “VLA + RL 训练管线” 强相关（见 [RLinf GitHub](https://github.com/RLinf/RLinf)）。

---

## 3) 你要同时做“仿真 + 真机”：推荐用 RLinf 的方式（不依赖你先选定 policy）

你现在说“仿真器和真机都做、policy 还不确定”，这反而是最适合先把 infra 视角定下来的情况。

### 3.1 先定三件事（跨 sim/real 的 contract）

- **Observation contract**：RGB / tactile / proprio / language token（哪些必需、哪些可选、时间窗多长）
- **Action contract**：joint pos/vel、Δpose、torque？以及“模型输出频率”与“控制环频率”怎么桥接（chunk + 插值/滤波）
- **Reward contract（仿真） + Safety contract（真机）**：
  - 仿真 reward 可以 dense
  - 真机最重要是 constraint / safety（限位、速度、力矩、接触门控、容错状态机）

> 经验：**contract 没定死之前，不要急着押注 diffusion/flow/tokenization。**

### 3.2 训练阶段建议（最稳路径）

- **Stage A：离线 BC（先让它“能动”）**
  - 连续回归/离散 token/diffusion/flow 都行；先用最稳的方式 warmstart
- **Stage B：仿真上 RL（做大规模改进）**
  - 大规模 rollout + 快速迭代 → 这里 infra 带来的收益最大
- **Stage C：真机只做“小步安全迭代”或“只评估闭环”**
  - 真机更新要保守：KL-to-base、蒸馏、回放、强 safety regularization

> 一句话：**sim 用来探索与放大收益，real 用来校准与验证可靠性。**

---

## 4) 与 VLA loss / 训练目标怎么对齐？（从“手册”映射到“RL infra”）

你可以把 VLA 的训练目标分成 3 层，对应 RLinf 里通常会出现的组件责任：

### 4.1 Policy learning（策略学习层）

- **BC**：MSE/Huber、Token CE、GMM NLL、Diffusion eps loss、Flow matching loss
- **RL fine-tune**：PPO（clip objective + value + entropy）、离线 RL、KL-to-base penalty

对应手册章节：
- `theory/vla_loss_functions_handbook.md`：BC / Diffusion / Flow / RL / Safety / Distill

### 4.2 Representation & grounding（表示/对齐层）

- CLIP/InfoNCE 对齐（vision↔language、vision↔tactile）
- 多模态 fusion 的损失与 mask（co-training）

对应手册章节：
- `InfoNCE`、`Co-training & loss masking`

### 4.3 Safety & controllability（可部署层）

- action smoothness（速度/加速度/jerk）
- barrier/constraint penalty（限位、力矩、workspace）
- contact gating（in-contact 触觉权重上升，避免视觉抖动直接入控制）

对应手册章节：
- `Safety regularization`、`Anti-forgetting`

---

## 5) 如果你想用 RLinf 做 VLA+RL：最重要的 6 个“别翻车”点

1. **控制频率对齐**：policy 的输出频率 vs 真机控制环（常见 10-30Hz vs 125-500Hz）必须用 chunk + 插值桥接，否则 reward/稳定性会被频率问题吞没。
2. **评估协议固定**：seed、初始分布、成功判定、失败类型统计；不然你会得到“每天都像换了任务”的曲线。
3. **KL-to-base 必备**：真机/高风险任务上，任何 RL 更新都要对基线策略加 KL 约束（防策略崩坏）。
4. **奖励不要把你骗了**：仿真 dense reward 可能让策略学会 exploit（投机取巧）；真机一定要保留 “成功/失败 + safety margin” 指标。
5. **把失败当一等数据**：掉帧、打滑、卡死、过热、保护停机 → 必须记录、回放、统计（否则 debug 永无止境）。
6. **先跑通数据面，再谈算法**：rollout→buffer→train→eval 的端到端闭环跑通，比“先写一个更 fancy 的 loss”更重要。

---

## 6) 你该怎么继续看 RLinf（给你一个阅读路径）

打开项目后，建议按这个顺序理解：

- **README / docs**：搞清楚它支持什么场景、以及训练的抽象（见 [RLinf GitHub](https://github.com/RLinf/RLinf)）
- **examples**：看一个最小 end-to-end workflow（env → rollout → train → eval）
- **tests / CI**：看它怎么定义“可复现”与回归（这是 infra 的灵魂）
- **toolkits / ray_utils**：通常是分布式调度与数据面关键实现（吞吐与稳定性在这里）

---

## 7) 参考链接

- RLinf 项目主页：见 [RLinf GitHub](https://github.com/RLinf/RLinf)（README 内也包含 arXiv 引用条目与 roadmap）

---

[← Back to Theory](../README.md)

