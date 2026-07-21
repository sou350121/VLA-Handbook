# RISE: 用组合世界模型实现 VLA 策略自改进 (Self-Improving Robot Policy with Compositional World Model)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-04-30
>
> **论文**: RISE: Self-Improving Robot Policy with Compositional World Model
> **链接**: https://arxiv.org/abs/2602.11075
> **核心定位**: 解决 VLA 在接触丰富和动态操作任务中的脆弱性——通过组合世界模型在"想象空间"中做 on-policy RL，避免真实世界的硬件成本和重置开销，在三个真实机器人任务上实现 +35% ~ +45% 的绝对性能提升。

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | 用组合世界模型（Dynamics + Value）替代真实环境做 on-policy RL，VLA 策略可在想象空间中持续自改进 |
| 適合精讀 | 做 VLA/具身智能 RL 微调的研究者；需要解决 contact-rich 任务失败恢复问题的工程师 |
| 可以跳過 | 只关心纯仿真环境 RL（LIBERO 等）或纯 IL/VLA 预训练的人 |
| 落地可行性 | 中（需大规模预训练 world model：$16\times\text{H100}\times7$天 + $8\times\text{H100}\times3$天；但推理零开销） |
| 主要風險 | 世界模型的误差累积限制连续想象步数（最多 2 步）；泛化到未见任务/机器人平台未验证 |

💡 **X-Ray 开场**
VLA 模型（如 OpenVLA、$\pi_{0.5}$）在预训练时学到了广泛的语义理解，但在接触丰富任务（如动态分拣、背包整理）中，微小的执行偏差会累积成失败。传统的真实世界 RL 因为安全、硬件成本和重置困难而难以规模化。RISE 的核心发现是：一个分解为"动力学预测"和"价值评估"两个模块的组合世界模型，可以在想象空间中生成带 advantage 信号的 on-policy 数据，让 VLA 策略在不接触真实环境的情况下持续改进。对 VLA 研究者意味着——世界模型可以成为真实 RL 的可扩展替代品，而不只是可视化或规划工具。

📍 **研究全景时间线**
```
[2024] VLA 预训练范式确立 (RT-2, OpenVLA, π₀.₅)
    → [2024-2025] IL 局限性暴露：exposure bias、接触任务脆弱性
    → [2025] 仿真 RL 方案 (LIBERO 上 PPO/DSRL)：并行但无法迁移到真实世界
    → [2025] 离线 RL 方案 (RECAP)：用历史数据但受 distribution shift 限制
    → [2026-02] RISE ← 当前位置：组合世界模型做想象 RL，突破离线数据瓶颈
    → [未来?] 多任务泛化、更长的想象 horizon、触觉反馈集成
```

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 组件 | 基础架构 | 输入 | 输出 | 训练数据 | 训练开销 | 推理开销 |
|------|----------|------|------|----------|----------|----------|
| **Dynamics Model (𝒟)** | Genie Envisioner (LTX-Video) + 轻量 action encoder | 历史多视图观测 $O_t$ + 动作序列 $a_t$ | $H$ 步未来多视图观测 $\{\hat{o}_{t+1}, ..., \hat{o}_{t+H}\}$ | Agibot World + Galaxea（大规模机器人数据） | 预训练 $16\times\text{H100}\times7$天；微调 $8\times\text{H100}\times3$天 | <2秒生成 25 帧多视图（比 Cosmos 快 $300\times$） |
| **Value Model (𝒱)** | $\pi_{0.5}$ VLA backbone（冻结/微调） | 单帧多视图观测 $\hat{o}_t$ + 语言指令 $\ell$ | 标量价值分数 $V(\hat{o}_t, \ell) \in [0,1]$ | 任务特定离线数据（成功+失败） | $8\times\text{H100}\times1$天（50k steps） | 前向传播一次（毫秒级） |
| **Policy (π)** | $\pi_{0.5}$ VLA（flow-matching） | 观测 $o_t$ + 指令 $\ell$ + advantage 离散 bin | 动作序列 $a_t$（chunk length $H$） | 离线数据 warm-up + 想象 rollout 自改进 | warm-up + 10k steps self-improving，8×H100 | 与 $\pi_{0.5}$ 相同（world model 不参与推理） |

### 1.2 关键机制 (Key Mechanism)

**为什么分解为 Dynamics + Value 两个模块？**

传统世界模型试图用一个模型同时完成"预测未来"和"评估好坏"两件事。RISE 的核心洞察是这两个子问题需要完全不同的架构和训练目标：

- **Dynamics 需要速度和可控性**：用视频扩散模型（Genie Envisioner），优化生成速度（<2秒/25帧）和对动作的条件控制能力
- **Value 需要判断力**：用 VLA backbone（$\pi_{0.5}$），它已经具有机器人中心的理解力，天然适合做多视图输入的价值评估

分解后，每个模块可以用最适合的架构和优化目标独立训练，互不干扰。

⚡ **Eureka Moment**：与其用一个巨型世界模型同时做"预测"和"评判"，不如把它们拆开——让视频模型负责想象未来，让 VLA 模型负责判断好坏，然后用 advantage = 平均价值提升 作为 RL 信号，让策略在想象空间中自我改进。

### 1.3 信息流/架构图 (Flow / Diagram)

```
                    ┌─────────────────────────────────────────────────┐
                    │              Self-Improving Loop                 │
                    │                                                 │
  oₜ, ℓ ──────► ┌───┴──────────┐                                     │
                │   Policy π    │  以 optimal advantage=1 为条件       │
                │  (rollout)    │  生成动作 aₜ                        │
                └───┬───────────┘                                     │
                    │  aₜ = [aₜ, aₜ₊₁, ..., aₜ₊H₋₁]                  │
                    ▼                                                 │
              ┌─────┴───────────┐                                     │
              │ Dynamics Model 𝒟 │  预测未来 H 步多视图观测            │
              │ (Genie Envisioner)│  ôₜ₊₁, ..., ôₜ₊H = 𝒟(Oₜ, aₜ)     │
              └─────┬───────────┘                                     │
                    │  {ôₜ₊₁, ..., ôₜ₊H}                              │
                    ▼                                                 │
              ┌─────┴───────────┐                                     │
              │ Value Model 𝒱   │  评估每个未来状态的价值              │
              │ (π₀.₅ backbone) │  V(ôₜ₊k, ℓ) for k=1..H             │
              └─────┬───────────┘                                     │
                    │                                                 │
                    ▼                                                 │
              ┌─────┴───────────┐                                     │
              │ Advantage Calc  │  A = (1/H)ΣV(ôₜ₊k) - V(oₜ)         │
              │                 │  → 离散化为 N bins                   │
              └─────┬───────────┘                                     │
                    │  (oₜ, â, A)                                     │
                    ▼                                                 │
              ┌─────┴───────────┐                                     │
              │  Policy Train   │  π(A, oₜ, ℓ) → â                   │
              │ (flow-matching) │  混合离线数据防遗忘                  │
              └─────────────────┘                                     │
                    │                                                 │
                    └────── EMA 更新 π_rollout ────┘                  │
                                                                    │
                    └──────────── 迭代 10k steps ────┘                │
                    ┌─────────────────────────────────────────────────┐
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：
```
Advantage A(oₜ, aₜ, ℓ) = (1/H) · Σ[k=1→H] V(ôₜ₊k, ℓ) − V(oₜ, ℓ)
```

用想象未来 H 步的平均价值，减去当前价值——差值就是这组动作的 advantage。正 = 比现状好，负 = 比现状差。

### 2.1 动力学模型（Dynamics Model）

**目标**：给定历史观测和候选动作，预测未来多视图观测。

```
ôₜ₊₁, ..., ôₜ₊H = 𝒟(Oₜ, aₜ)

其中:
  Oₜ = {oₜ₋N, ..., oₜ₋₁, oₜ}  — 历史窗口（N 帧）
  aₜ = [aₜ, aₜ₊₁, ..., aₜ₊H₋₁] — 动作 chunk（H 步）
  ôₜ₊k                          — 第 k 步预测的多视图观测
```

### 2.2 价值模型（Value Model）

**目标**：学习一个标量价值函数，评估任意状态距任务成功的距离。

```
L_V = L_prog + L_TD

L_prog = E[(V(oₜ, ℓ) − t/T)²]          — 时间进度回归
L_TD   = E[(V(oₜ, ℓ) − yₜ)²]           — TD learning
yₜ = rₜ + γ · V(oₜ₊₁, ℓ)               — TD target

其中:
  t/T    — 当前时间步在总 episode 中的进度比例 [0,1]
  rₜ     — 中间步=0，成功结尾=+1，失败结尾=−1
  γ      — 折扣因子
```

**为什么需要两个 loss？** Progress loss 提供稠密信号但过于平滑（对失败不敏感）；TD loss 用成功/失败终态锚定价值，对接触任务中的细微错误更敏感。两者互补。

### 2.3 Advantage 计算

```
A(oₜ, aₜ, ℓ) = (1/H) · Σ[k=1→H] V(ôₜ₊k, ℓ) − V(oₜ, ℓ)

直觉：如果执行动作序列 aₜ 后，未来 H 步的平均价值比当前高 → advantage 为正 → 这是好动作
```

### 2.4 策略改进（Policy Improvement）

**目标**：用 probabilistic inference 框架，将 advantage 作为条件引导策略生成。

```
π(A_rollout(o, âₜ, ℓ), oₜ, ℓ) → â

优化目标：flow-matching（与 π₀.₅ 一致）
训练数据混合：想象 rollout + 离线标注数据（防 catastrophic forgetting）
```

> 符号与本文保持一致：$\mathcal{D} = \text{dynamics model}$, $\mathcal{V} = \text{value model}$, $\pi = \text{policy}$, $H = \text{action chunk length}$, $N = \text{history window length}$, $\ell = \text{language instruction}$。

## 3. 带数字走一遍：玩具例子 (Worked Example)

假设一个简化的"动态分拣"场景，action chunk length H=4。

**时刻 t**：机器人看到传送带上的一块红色砖块。

```
当前观测 oₜ:  相机看到红色砖在传送带位置 x=3
指令 ℓ:       "把红色砖放到红色箱子"
```

**Step 1 — Policy Rollout**：策略以 advantage=1（最优意图）为条件，生成动作：
```
aₜ = [抓取, 移动, 放置, 等待]  （4 步动作 chunk）
```

**Step 2 — Dynamics 想象未来**：
```
ôₜ₊₁: 机械爪接近砖块
ôₜ₊₂: 砖块被抓起（离开传送带）
ôₜ₊₃: 砖块移动到红色箱子上方
ôₜ₊₄: 砖块放入红色箱子
```

**Step 3 — Value 评估**：
```
V(oₜ, ℓ)    = 0.10  （刚看到砖块，进度 10%）
V(ôₜ₊₁, ℓ)  = 0.25  （接近砖块，进度 25%）
V(ôₜ₊₂, ℓ)  = 0.45  （抓起砖块，进度 45%）
V(ôₜ₊₃, ℓ)  = 0.70  （移动到箱子上方，进度 70%）
V(ôₜ₊₄, ℓ)  = 0.95  （放入箱子，进度 95%）
```

**Step 4 — Advantage 计算**：
```
A = (0.25 + 0.45 + 0.70 + 0.95) / 4 − 0.10
  = 0.5875 − 0.10
  = +0.4875
```

Advantage = +0.49，这是一个高价值动作序列。策略在训练时会学到：当处于类似状态时，生成这个动作序列。

**对比失败场景**：假设动作序列是 [移动, 放置, 抓取, 等待]（顺序错误）：
```
V(ô'ₜ₊₁) = 0.12, V(ô'ₜ₊₂) = 0.08, V(ô'ₜ₊₃) = 0.05, V(ô'ₜ₊₄) = 0.03
A' = (0.12 + 0.08 + 0.05 + 0.03) / 4 − 0.10
   = 0.07 − 0.10
   = −0.03
```

Advantage = −0.03，策略学到这是差动作。通过大量这样的正/负样本，策略在想象空间中逐步改进。

## 4. 工程视角 (Engineering View)

| 维度 | 数值 | 含义 |
|------|------|------|
| **World Model 预训练** | 16×H100 × 7天（dynamics）+ 8×H100 × 1天（value） | 一次性投入，可跨任务复用 dynamics model |
| **任务微调** | 8×H100 × 3天（dynamics FT）+ 8×H100 × 1天（value FT） | 每个新任务需要 |
| **策略 warm-up** | 8×H100，batch=64 | 离线数据微调，类似 RECAP |
| **自改进循环** | 8×H100，batch=64，~10k steps | 纯在想象空间中，不需要真实交互 |
| **Dynamics 推理延迟** | <2秒生成 25 帧多视图 | 比 Cosmos（>10分钟）快 $300\times$，这是 RL 可扩展的关键 |
| **连续想象步数** | 最多 2 步（从每个离线状态出发） | 受限于视频生成模型的误差累积 |
| **推理开销** | 零（world model 不参与推理） | 部署时只需要 policy $\pi$，world model 仅在训练时使用 |

**工程含义**：
- World model 的训练成本不低（总计约 11 天×16 H100），但这是**一次性投入**，dynamics model 预训练后可跨多个任务微调
- 每个任务的完整流程（world model FT + policy warm-up + self-improving）约需 5-6 天×8 H100
- 关键 trade-off：想象步数限制在 2 步内，意味着每个 rollout 只能探索有限状态空间，需要更多初始状态来覆盖
- 部署友好：推理时 world model 完全不参与，策略的 inference cost 与 $\pi_{0.5}$ 完全相同  

## 5. 数据与评测 (Data & Eval)

### 5.1 数据集

| 数据集 | 用途 | 说明 |
|--------|------|------|
| **Agibot World** | Dynamics 预训练 | 大规模真实机器人动作标注数据 |
| **Galaxea** | Dynamics 预训练 | 大规模真实机器人数据（OpenDriveLab 自有） |
| **任务特定离线数据** | Value FT + Policy warm-up | 每个任务包含：专家演示 + 策略 rollout（成功/失败）+ 人工干预矫正（DAgger） |

### 5.2 评测任务（真实世界，双 7-DoF AgileX 机器人）

| 任务 | 难度特征 | 挑战 |
|------|----------|------|
| **Dynamic Brick Sorting** | 动态 + 接触丰富 | 从移动传送带上精确抓取彩色砖块并分类 |
| **Backpack Packing** | 长程 + 柔性物体 | 打开背包 → 放入衣服 → 提起 → 拉拉链   |
| **Box Closing** | 精确双臂协调 | 折叠盒盖并将卡扣精确插入盒子 |

### 5.3 主要结果（论文 Table I）

| 方法 | Dynamic Brick Sorting | Backpack Packing | Box Closing |
|------|----------------------|------------------|-------------|
| $\pi_{0.5}$ (baseline)   | 基线 | 基线 | 基线 |
| $\pi_{0.5}$ + DAgger   | 有提升 | 有提升 | 有提升 |
| $\pi_{0.5}$ + PPO   | 有限提升 | 有限提升 | 有限提升 |
| $\pi_{0.5}$ + DSRL   | 中等提升 | 中等提升 | 中等提升 |
| RECAP ($\pi_{0.5}$)   | 较强提升 | 较强提升 | 较强提升 |
| **RISE (Ours)** | **+35% 绝对提升** | **+45% 绝对提升** | **+35% 绝对提升** |

> 数据来源：论文 Table I + 项目页。RISE 在所有三个任务上显著超越所有基线。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

### 6.1 能力

| 能力 | 场景 | 原因 |
|------|------|------|
| **接触丰富任务改进** | 动态分拣、背包整理 | World model 能模拟接触状态变化，value model 能区分成功/失败接触 |
| **长程任务恢复** | 背包打包（4 个子任务） | TD learning 让价值函数对中间失败敏感，advantage 信号引导恢复 |
| **双臂精确协调** | 盒子关闭 | 多视图输入提供充足的空间信息，value model 能评估精度 |
| **自纠错** | 执行偏差后的恢复 | 想象空间中探索失败路径 → 学到纠正动作   |

### 6.2 失败模式

| 失败模式 | 场景 | 原因 |
|----------|------|------|
| **误差累积** | 连续想象 >2 步 | 视频生成模型的误差随步数累积，预测质量下降 |
| **任务特定过拟合** | 迁移到新任务 | 每个任务需要独立的 world model FT 和 offline data |
| **未见物体/场景** | 新物体类型或布局 | 未报告 zero-shot 泛化能力 |
| **不同机器人平台** | 换到其他机器人 | 实验仅在双 7-DoF AgileX 上验证 |

### 6.3 隐含假设 (Hidden Assumptions)

1. **世界模型的视觉保真度足够支撑价值判断**：假设动力学模型生成的想象帧在视觉和语义上足够准确，让价值模型能做出可靠评估。但如果生成帧出现细微失真（如物体位置偏差几像素），价值评估可能严重偏差。
2. **优势信号在离散化后仍保留足够信息**：advantage 被离散化为 N 个 uniform bins。如果 bin 数量不足，可能丢失关键的细微差异信号。
3. **offline data 的质量决定上限**：warm-up 阶段依赖任务特定的离线数据（专家演示 + rollout）。如果离线数据覆盖不足，想象空间中的探索可能无法突破数据分布的限制。
4. **价值模型的多视图兼容性**：假设从 $\pi_{0.5}$ 初始化的价值模型能自然地处理多视图输入。但 $\pi_{0.5}$ 的多视图能力本身可能有限，这会影响价值评估质量。  

## 7. 与相关工作对比 (Comparison)

| 方法 | 核心思路 | 数据来源 | 是否需真实交互 | 优势 | 局限 |
|------|----------|----------|----------------|------|------|
| **$\pi_{0.5}$ + IL**   | 纯模仿学习 | 专家演示 | 否 | 简单，无需奖励设计 | exposure bias，无法恢复偏差 |
| **$\pi_{0.5}$ + DAgger**   | 在线人工矫正 | 演示 + 人工干预 | 是 | 缓解 exposure bias | 需要持续人工参与，不扩展 |
| **$\pi_{0.5}$ + PPO**   | 真实世界 on-policy RL | 真实交互 | 是 | 理论上最优 | 硬件成本高，安全风险，重置困难 |
| **$\pi_{0.5}$ + DSRL**   | 冻结 VLA + 优化扩散噪声 | 离线数据 | 否 | 样本高效 | 受离线数据分布限制 |
| **RECAP** | advantage-conditioned offline RL | 离线数据 | 否 | 利用 advantage 信号 | 受离线数据 distribution shift 限制 |
| **RISE (Ours)** | 组合世界模型做想象 RL | 离线 + 想象 rollout | 否（仅训练时） | 突破离线数据限制，on-policy 想象 | world model 训练成本高，泛化未验证 |

**面试 Tip**：如果被问到"RISE 和 RECAP 的区别是什么？"——回答：RECAP 完全依赖离线数据（有 distribution shift 问题），RISE 用世界模型在想象空间中生成 on-policy 数据来持续改进，突破了离线数据的限制。两者都用 advantage conditioning，但数据来源不同。

## 8. 精讀建議 (Reading Guide)

- **值得精讀原文的人**：
  1. 做多模态具身 Agent RL 微调的研究者——组合世界模型的 decomposed design 可直接借鉴
  2. 要评估将 VLA 迁移到接触丰富任务（装配、整理、操作柔性物体）可行性的工程师
  3. 关注 world model 作为 RL 训练环境（而非规划工具）的研究者

- **建議章節路徑**：
  - 先讀 §II（Preliminary）：理解 dynamics + value + advantage 的数学形式化
  - 再看 §III-A（Compositional World Model）：核心设计选择，尤其是 Task-Centric Batching 的动机
  - 再看 §III-C（Self-Improving Loop）：rollout + training 的闭环流程
  - 可跳 §IV-A（Experimental Setup）：如果不关心具体机器人硬件配置

- **不值得精讀的理由**：
  - 如果你不做机器人学习（只关注纯视觉/语言任务），这篇的方法论距离较远
  - 如果你已经熟悉 $\pi_{0.6}^*$ 的 probabilistic inference 框架和 RECAP，核心创新主要在 world model 的组合设计上，RL 部分相对标准  

---
[← Back to Theory](./README.md)  
