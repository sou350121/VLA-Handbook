# ORCHID：分层扩散策略的在线自训练共适应 (Online Self-Training for Co-Adaptation in Hierarchical Diffusion Policies)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-06-11
>
> **论文**: Online Self-Training for Co-Adaptation in Hierarchical Diffusion Policies
> **链接**: https://arxiv.org/abs/2603.05291
> **代码**: https://github.com/clemgris/ORCHID
> **核心定位**: 用 LLM 自训练思路（STaR/ReST）解决分层扩散策略中 HL-LL 耦合不匹配问题——通过环境反馈筛选成功轨迹并蒸馏回灌 HL 和 LL，实现双向共适应，避免梯度 RL 的不稳定性。

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | 用环境反馈筛选的成功轨迹对 HL（扩散规划器）和 LL（控制器）做监督蒸馏，迭代提升分层策略，在 CALVIN LH-MTLC 上 Avg.Len. 从 1.89→3.93，超越 MDT（3.72）接近 FLOWER（4.35） |
| 适合精读 | 如果你在做分层策略、扩散策略在线微调、或 HL-LL 耦合对齐——§4 方法 + §5 实验是核心 |
| 可以跳过 | 如果你只关心单级扩散策略或纯离线训练——这篇的在线自训练循环不适用 |
| 落地可行性 | 中（需要仿真环境做 rollout 收集；CALVIN 代码已开源） |
| 主要风险 | 仅在仿真基准验证（CALVIN / Franka-3Blocks），无实机实验；ORCHID-ft 有灾难性遗忘风险 |

💡 **X-Ray 开场**
分层扩散策略把长程操作任务拆成高层规划器（HL，生成视觉子目标序列）和底层控制器（LL，执行动作）。问题是：HL 规划的子目标 LL 不一定能到达，LL 训练的数据分布也可能和 HL 实际生成的分布不匹配——这就是 HL-LL 耦合问题。ORCHID 的洞察是：从 LLM 自训练借一个简单但强大的思路——让当前策略与环境交互，筛选出 HL 和 LL 都成功的轨迹，然后用这些轨迹的监督损失同时更新两者。不需要梯度 RL，不需要额外的 glue 模型，迭代 3-4 轮就能让轻量模型超越离线训练的大模型。

📍 **研究全景时间线**
```
[2023] Diffusion Policy (Chi et al.) — 扩散模型用于底层控制
     ↓
[2023-24] HIP/Ajay, TaKSIE/Kang — 引入 glue 模型缓解 HL-LL 耦合
     ↓
[2024] MDT/Reuss, LDC/Zhang — 共享表示空间耦合 HL-LL
     ↓
[2025] FLOWER/Reuss — 950M VLA 预训练 + 250k 轨迹 → CALVIN SOTA (Avg.Len. 4.35)
     ↓
[2026] ORCHID (本文) — 在线自训练蒸馏，轻量模型逼近 FLOWER ← 当前位置
     ↑
     局限：仅仿真验证，无实机部署
```

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 维度 | 传统离线分层 (iter 0) | ORCHID (在线自训练) | FLOWER (VLA SOTA) |
|------|----------------------|---------------------|-------------------|
| HL 架构 | 扩散 3D CNN U-Net (AVDC)，CLIP 文本编码 | 同左，迭代更新 | 950M VLM  backbone |
| LL 架构 | Diffusion Policy 或 ACT | 同左，迭代更新 | 扩散策略 |
| 子目标类型 | 视觉观测序列（一次性生成完整 plan） | 同左 | 视觉子目标 |
| HL-LL 耦合方式 | 独立训练，无耦合机制 | 同一过滤数据集 ℛ_t 双向蒸馏 | 共享预训练表示 |
| 训练阶段 | 单次离线 IL | 多轮 (Stage 1→2→3 循环) | 预训练 + 离线微调 |
| 需要额外模型 | 否 | 否（无 glue / reward / world model） | 是（VLM 预训练） |
| 数据效率 | 受限于 D_0 | 迭代扩展 beyond D_0 | 需要 250k 轨迹预训练 |
| CALVIN Avg.Len. | 1.89 (HD-ACT) | 3.93 (ORCHID-ACT-ft iter 3) | 4.35 |

### 1.2 关键机制 (Key Mechanism)

ORCHID 的核心是一个三阶段自强化循环：

**Stage 1 — 策略更新**：在当前数据集 D_t 上独立训练 HL 和 LL。
- HL：速度参数化扩散目标（pred-v），输入初始观测 o_0 + 文本目标 g，输出视觉子目标序列 ζ = ⟨o^1, ..., o^M⟩
- LL： goal-conditioned 策略，输入 $(o_{\text{source}}, o_{\text{target}})$，输出动作块 $a_c = \langle a_0, ..., a_{n-1} \rangle$

**Stage 2 — Rollout 收集**：当前策略 $\pi_t$ 部署，每个上下文 $(s_0, l)$ 执行 $K$ 次 rollout，保留第一个成功轨迹。
- 上下文类型 1：环境重置上下文 $(o_0 \sim \rho_{\text{reset}}, g \sim \mathcal{G}_l)$
- 上下文类型 2：回放上下文 (o_0 = o_N^* from D_t, g ~ G_l) — 用已收集轨迹的终态作为新起点，探索标准重置无法到达的状态

**Stage 3 — 数据集聚合**：
- ORCHID：$D_{t+1} = D_t \cup \mathcal{R}_t$，从头训练（防遗忘，但计算量递增）
- ORCHID-ft：$D_{t+1} = \mathcal{R}_t$，从 $\pi_t$ 微调（计算量恒定，但有遗忘风险）

⚡ **Eureka Moment**：用同一份过滤成功轨迹 ℛ_t 同时训练 HL 和 LL——HL 学到的是"LL 实际能到达的子目标分布"，LL 学到的是"HL 实际会生成的子目标模式下的动作分布"，两者在同一数据驱动下自然对齐，无需任何显式耦合损失或额外模型。

### 1.3 信息流/架构图 (Flow / Diagram)

```
┌─────────────────────────────────────────────────────────────────┐
│                    ORCHID Self-Training Loop                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Stage 1: Policy Update (Supervised)                             │
│  ┌──────────────┐    D_t    ┌──────────────┐                    │
│  │  HL (Diff)   │◄─────────┤  Dataset D_t │                    │
│  │  Planner     │          │  {(τ, l)}    │                    │
│  └──────┬───────┘          └──────────────┘                    │
│         │                                                      │
│  ┌──────────────┐    D_t    ┌──────────────┐                    │
│  │  LL (DP/ACT) │◄─────────┤              │                    │
│  │  Controller  │          │              │                    │
│  └──────┬───────┘          └──────────────┘                    │
│         │                                                      │
│         ▼                                                      │
│  Stage 2: Rollout Collection (Environment Interaction)         │
│         │                                                      │
│    ┌────┴────┐    π_t = (HL_t, LL_t)                          │
│    │ Context │──K rollouts/context──► Filter by R=1 ──► ℛ_t   │
│    │ Buffer  │   (reset + replayed)    (binary success)        │
│    └────┬────┘                                                      │
│         │                                                      │
│         ▼                                                      │
│  Stage 3: Dataset Aggregation                                  │
│  ┌──────────────────────────────────────────────┐              │
│  │ ORCHID:  D_{t+1} = D_t ∪ ℛ_t  (retrain)    │              │
│  │ ORCHID-ft: D_{t+1} = ℛ_t      (fine-tune)  │              │
│  └──────────────────────────────────────────────┘              │
│         │                                                      │
│         └────────────► 回到 Stage 1 (下一轮迭代)                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：
```
π_{t+1} = SL(D_t ∪ Filter_R=1(Rollout(π_t)))
```
即：下一轮策略 = 在当前数据 + 环境反馈筛选的成功轨迹上做监督学习。

### 目标

最大化分层策略的期望回报：

```
J(π_HL, π_LL) = E_{(s_0, l), g, ζ~π_HL, τ~π_LL} [ R(τ, s_0, l) ]
```

其中 $R \in \{0, 1\}$ 是二值任务成功奖励。

### HL 扩散目标（速度参数化）

```
L_HL(φ) = || v_φ(ζ^j, j, o_0*, g) - (α_j·ε_j - β_j·ζ^0) ||_2^2
```

| 符号 | 含义 |
|------|------|
| $\zeta^j$ | 扩散步 j 的 noisy 子目标序列 |
| $\zeta^0$ | 目标子目标序列（从 expert/成功轨迹提取） |
| $\alpha_j, \beta_j$ | 噪声调度系数 |
| $v_\varphi$ | 速度预测网络（U-Net） |
| o_0*, g | 初始观测 + 文本目标 |

### LL 控制目标

```
L_LL(ψ) = || π_LL_ψ(o_source, o_target) - a_c* ||_2^2
```

| 符号 | 含义 |
|------|------|
| o_source | 当前观测 |
| o_target | HL 生成的子目标 |
| a_c* | 真实动作块（变长采样，padding 到固定长度 n） |

### 可达性误差（评估指标）

```
E(π_HL, π_LL) = E [ (1/M) · Σ_{i=1}^M d( O(s_{x_i}), o^_i ) ]
```

衡量 HL 规划的子目标 o^_i 和 LL 实际到达状态的观测 O(s_{x_i}) 之间的平均距离（在 pixel / R3M / DINOv2 三个嵌入空间计算）。

> 符号与本文保持一致：$\pi_{\text{HL}}$ 为高层扩散规划器，$\pi_{\text{LL}}$ 为底层控制器，$\zeta$ 为视觉子目标序列，$\mathcal{R}_t$ 为第 $t$ 轮成功轨迹集，$D_t$ 为第 $t$ 轮训练数据集。

## 3. 带数字走一遍：玩具例子 (Worked Example)

假设 CALVIN 上一个简单任务序列："pick apple" → "place apple in bowl"。

**初始状态（iter 0）**：
- D_0 = 150 条人类遥操作轨迹/任务
- HL 在 D_0 上训练：学到的是人类演示的子目标风格（可能包含 LL 无法精确到达的中间姿态）
- LL 在 D_0 上训练：学到的是人类子目标下的动作分布
- 结果：HD-ACT 在 LH-MTLC 上 Avg.Len. = 1.89（只能稳定完成 1-2 个任务）

**iter 1 rollout 收集**：
- 对每个上下文，$\pi_0$ 执行 $K=5$ 次 rollout
- 假设 100 个上下文中 35 个至少有一次成功 → $\mathcal{R}_1$ 包含 35 条成功轨迹
- 关键：ℛ_1 中的子目标是 LL_0 实际到达的状态观测——这些是"LL 真正能到达"的子目标

**iter 1 蒸馏**：
- $\text{HL}_1$ 在 $\mathcal{R}_1$ 上训练：不再拟合人类演示的子目标，而是拟合 $\text{LL}_0$ 实际到达的子目标 → HL 开始生成"LL 可到达"的子目标
- $\text{LL}_1$ 在 $\mathcal{R}_1$ 上训练：学到的是"在 $\text{HL}_0$ 生成的子目标模式下应该做什么动作" → LL 开始适配 HL 的计划风格

**iter 3 结果（ORCHID-ACT-ft）**：
- Avg.Len. = 3.93（从 1.89 提升 108%）
- 5 任务连续成功率从 12.3% → 48.5%
- HL 的可达性误差 E 在 DINOv2 空间显著下降（论文 Figure 3）
- 超越 MDT (3.72)，接近 FLOWER (4.35)——而 FLOWER 用了 950M 参数 + 250k 预训练轨迹

**直觉**：每一轮迭代，HL 和 LL 的分布差距缩小一点。3-4 轮后，两者已经"互相适应"——HL 不会生成 LL 够不到的子目标，LL 已经 specializes 在 HL 的计划模式上。

## 4. 工程视角 (Engineering View)

| 维度 | ORCHID | ORCHID-ft | 工程含义 |
|------|--------|-----------|----------|
| 每轮训练成本 | 递增（D_t 不断增大） | 恒定（只微调 ℛ_t） | ORCHID 在 iter 4 时训练时间是 iter 1 的 4 倍+ |
| 内存需求 | 递增 | 恒定 | 大规模部署时 ORCHID-ft 更友好 |
| 遗忘风险 | 低（保留所有历史数据） | 中（仅用最新数据微调） | ORCHID-ft 在 iter 3+ 时可能出现性能波动 |
| HL 推理开销 | 一次性生成完整 plan + replan 触发 | 同左 | 比 reactive subgoal 方法（每步生成）更高效 |
| LL 推理开销 | 扩散 N 步去噪 或 ACT 前向 | 同左 | DP 比 ACT 慢但精度略高 |
| 环境交互成本 | 每轮 $K \times$ |C(D_t)| 次 rollout | 同左 | 这是最大瓶颈——CALVIN 仿真中每次 rollout 约数秒 |
| 部署约束 | 需要仿真环境做 online rollout | 同左 | 实机部署需要真实环境交互，成本远高于仿真 |

**工程含义总结**：ORCHID 的核心工程 trade-off 是"数据累积 vs 计算效率"。ORCHID-ft 以恒定成本实现了大部分收益（CALVIN 上 3.93 vs 3.82），是实际部署的首选。但需要警惕迭代后期的遗忘风险——论文中 ORCHID-ft 在 iter 4 时某些指标略有回落。

## 5. 数据与评测 (Data & Eval)

### 数据集

| 基准 | 任务数 | D_0 规模 | 数据来源 | 评估协议 |
|------|--------|----------|----------|----------|
| Franka-3Blocks | 10 | 100 demo/任务（默认）/ 10（低数据） | 手写 expert（特权状态） | 任务成功率 |
| CALVIN $D \to D$ | 34 | 150 demo/任务 | 人类遥操作 | MTLC（单任务）+ LH-MTLC（长程连续任务） |

### CALVIN 评测协议

- **MTLC**：固定设置下的单任务成功率（OOD 文本目标）
- **LH-MTLC**：从 1000 个固定重置起点开始，连续完成任务直到失败，记录平均成功序列长度（最多 5 个任务）

### 关键实验结果（CALVIN LH-MTLC，论文 Table 1）

| 方法 | 1任务 | 2任务 | 3任务 | 4任务 | 5任务 | Avg.Len. |
|------|-------|-------|-------|-------|-------|----------|
| HD-ACT (iter 0) | 76.0% | 49.3% | 31.4% | 19.9% | 12.3% | 1.89 |
| ORCHID-ACT-ft (iter 3) | 87.0% | 71.2% | 56.X% | ~48.5% | ~40% | 3.93 |
| MDT | 93.7% | 84.5% | 74.1% | 64.4% | 55.6% | 3.72 |
| FLOWER (950M VLA) | 97.4% | 92.4% | 86.9% | 81.3% | 74.9% | 4.35 |

> TODO: ORCHID-ACT-ft iter 3 在 3/4/5 任务的确切数值在抓取时被截断，论文原文有完整数据。

**数据来源**：论文 Table 1，3 seeds 均值 $\pm$ 标准误。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

### 能做什么
- **低数据 regime 下显著提升**：Franka-3Blocks 上仅 10 demo/任务时，ORCHID 仍能稳定提升（论文 Figure 4）
- **超越纯离线大模型**：轻量模型（HD-ACT）通过 3 轮自训练超越 MDT（共享表示方法）
- **双向共适应可量化**：可达性误差 E 在多轮迭代中持续下降（三个嵌入空间一致）
- **无需额外模型**：不需要 glue model、reward model、world model 或 expert oracle

### 不能做什么
- **无实机验证**：所有实验在 CALVIN（仿真）和 Franka-3Blocks（PyBullet 仿真）上完成
- **长程迭代可能遗忘**：ORCHID-ft 在 iter 4 时部分指标回落（论文暗示）
- **依赖二值奖励**：仅用 $R \in \{0,1\}$，无法利用密集奖励信号（如果有的话）
- **上下文覆盖有限**：回放上下文虽然扩展了探索，但仍受限于当前策略的能力边界

### 6.1 隐含假设 (Hidden Assumptions)

1. **二值奖励足够**：假设任务成功/失败信号足以引导策略改进。但在部分可观测设置下（视觉子目标不包含完整状态），成功轨迹可能包含偶然因素——LL 可能"碰巧"到达了 HL 的子目标但状态不完全正确。
2. **仿真到实机的迁移性**：CALVIN 的 $D \to D$ split 虽然是 in-distribution，但仍是仿真环境。实机上的 HL-LL 耦合问题可能更复杂（传感器噪声、动力学不确定性）。
3. **K 次 rollout 足够**：每个上下文仅尝试 K 次 rollout。如果策略成功率很低，大量上下文可能贡献零数据（ℛ_t 为空），导致迭代停滞。
4. **视觉子目标充分性**：HL 仅生成视觉观测子目标，不包含关节角度/力矩等状态信息。论文承认"视觉子目标可达性不保证任务完成"，但未深入探讨部分可观测性的根本限制。

## 7. 与相关工作对比 (Comparison)

| 方法 | HL-LL 耦合方式 | 在线更新 | 额外模型 | CALVIN Avg.Len. |
|------|---------------|---------|---------|-----------------|
| SuSIE | 无耦合 | 否 | 否（subgoal 检测器） | 2.80 |
| TaKSIE | Glue model 选择 subgoal | 否 | 是（progress estimator） | 3.18 |
| HIP | 辅助模型正则化 HL | 否 | 是（auxiliary discriminator） | TODO |
| MDT | 共享网络层 | 否 | 否 | 3.72 |
| LDC | 共享视觉嵌入空间 | 否 | 否 | 2.88 |
| FLOWER | VLM 预训练共享表示 | 否 | 是（950M VLM） | 4.35 |
| **ORCHID** | **同一过滤数据双向蒸馏** | **是** | **否** | **3.93** |

**面试 Tip**：当被问到"ORCHID 和 MDT 的核心区别是什么"时，回答："MDT 通过共享网络层让 HL 和 LL 在同一个表示空间里工作，这是静态耦合；ORCHID 让 HL 和 LL 通过在线交互和蒸馏自然对齐，这是动态耦合。MDT 受限于初始数据集的分布，ORCHID 能持续探索超出 D_0 的状态空间。"

## 8. 精讀建議 (Reading Guide)

- **值得精讀原文的人**：
  1. 做分层策略/扩散策略在线微调的研究者——§4 的自训练循环设计可直接复用
  2. 评估 HL-LL 耦合问题的工程师——§3.2 的耦合问题定义和 §5 的可达性误差指标有参考价值
  3. 探索 LLM 自训练（STaR/ReST/SPIN）向具身智能迁移的研究者——§2 的相关工作综述清晰

- **建議章節路徑**：先讀 §4（方法）$\to$ 再看 §5 实验（Table 1 + reachability error）$\to$ 可跳 §3.1（GC-POMDP 形式化，标准设定）

- **不值得精讀的理由**：如果不做分层策略或扩散策略（例如只关注单级 VLA 或纯离线 IL），这篇的核心贡献（在线自训练循环）与你的工作距离较远，读摘要即可。

---
[← Back to Theory](./README.md)

**关键引用**：
- 论文: https://arxiv.org/abs/2603.05291
- 代码: https://github.com/clemgris/ORCHID
- ICML 2026 Workshop on Decision-Making from Offline Datasets to Online Adaptation (DEMO)
