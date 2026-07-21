# LARA：潜在动作表示对齐 (Latent Action Representation Alignment for Vision-Language-Action Models)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-06-09
>
> **论文**: LARA: Latent Action Representation Alignment for Vision-Language-Action Models
> **链接**: https://arxiv.org/abs/2606.07100
> **核心定位**: 通过双向表示对齐让 LAM（潜在动作模型）和 VLA 联合训练，解决 LAM 未锚定真实动作、VLA 被冻结 LAM 表示束缚的双重问题

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | 将 LAM 的潜在动作表示 z_t 与扩散 VLA 的中间特征 h_t^θ 通过余弦相似度对齐，双向正则化两者，提升各类 VLA 基准 5%-15% |
| 適合精讀 | 如果你在做 LAM-based VLA 预训练、后训练增强、或用 latent action 做 pseudo-label——这篇的方法即插即用 |
| 可以跳過 | 如果你只关心纯行为克隆 VLA（如 OpenVLA 微调）且不用 LAM，这篇距离中等 |
| 落地可行性 | 高——仅需一个 projection head + cosine loss，不修改 DiT 或 LAM 架构 |
| 主要風險 | 真实世界 G1 人形机器人仅 $2$ 个复合任务 $\times 50$ 次演示，泛化性待更多 real-world 验证 |

💡 **X-Ray 开场**
VLA 模型受限于机器人动作数据稀缺，研究者常用 LAM 从未标注人类视频中提取潜在动作来补充监督信号。但传统做法是两阶段训练——LAM 先预训练，然后冻结作为 pseudo-label 提供者——这导致 LAM 学到的视觉动态与真实机器人动作脱节，VLA 又被冻结的 LAM 表示束缚。LARA 的核心发现是：让 LAM 和 VLA 在训练过程中通过表示对齐互相正则化，两者都能做得更好。

📍 **研究全景时间线**
```
2022 LAPO (Chen) ──→ 2024 LAPA (Ye) ──→ 2025 Moto-GPT (Chen) ──→ 2025 UniVLA (Bu) ──→ [本文 LARA]
   首次提出          用 LAM tokens       用 LAM 做 VLA         LAM 集成到          双向对齐
   潜在动作           做 VLA 监督         预训练               VLA 架构            联合优化
   ← 传统范式：LAM 预训练 → 冻结 → 作为 pseudo-label 或辅助监督 ←
   → 本文范式：LAM 与 VLA 联合训练，表示层双向对齐 ←
```

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 组件 | 传统 LAM-based VLA | LARA |
|------|---------------------|------|
| **LAM 训练** | 单独在 unlabeled 视频上预训练，然后冻结 | 与 VLA 联合训练，持续更新 |
| **LAM 输入** | $I_t$, $I_{t+C}$（连续帧） | 相同 |
| **LAM 输出** | 离散 latent action $z_t^q$（VQ codebook token） | 连续 latent action z_t（量化前）用于对齐 |
| **VLA 架构** | DiT + cross-attention，条件为 VLM 特征 + proprio | 相同 |
| **VLA 监督** | 真实动作 $A_t$ +（可选）预测 $z_t^q$ | 真实动作 A_t + LARA 对齐损失 |
| **训练阶段** | 两阶段：LAM 预训练 $\to$ VLA 训练 | 三阶段：LAM 预训练 $\to$ LARA 联合预训练 $\to$ LARA 联合后训练 |
| **信息流** | 单向：LAM $\to$ VLA（冻结） | 双向：LAM $\leftrightarrow$ VLA（联合优化） |
| **额外参数** | 无 | $1$ 个 projection head $f_\psi$（轻量 MLP） |

### 1.2 关键机制 (Key Mechanism)

LARA 的核心是一个简单的余弦相似度对齐损失：

```
L_LARA(θ, φ, ψ) = -E[A_t, ε, τ][CosSim(z_t^φ, f_ψ(h_t^θ))]
```

其中：
- $z_t^\varphi = \text{LAM}$ 的连续潜在动作（量化前）
- $h_t^\theta = $ 扩散 VLA（DiT）在 $L\text{-}2$ 层的中间特征
- $f_\psi = $ 可学习的 projection head

**为什么这样设计：**

1. **逆向动力学正则化（对 LAM）**：将 LAM 的 latent action 与 VLA 的动作策略表示对齐，迫使 LAM 关注控制相关的视觉变化（物体移动、机械臂运动），而非伪影变化（光照、背景），从而学到更"动作中心"的潜在空间。

2. **前向动力学锚定（对 VLA）**：标准行为克隆只是从观测到动作的模式匹配，不建模动作的物理后果。通过将 DiT 中间特征锚定到 LAM 的前向预测 latent action，VLA 被注入了未来状态演化的显式概念，减少了"运动学合理但功能无效"的幻觉轨迹。

⚡ **Eureka Moment**：不要冻结 LAM——让它和 VLA 在训练中共进化，用一个余弦相似度损失就能实现双向正则化。

### 1.3 信息流/架构图 (Flow / Diagram)

```
┌─────────────────────────────────────────────────────────────┐
│                    LARA Joint Training                       │
│                                                             │
│  ┌──────────┐    I_t, I_{t+C}    ┌──────────┐              │
│  │  Input   │ ─────────────────→ │   LAM    │              │
│  │  Frames  │                    │  (φ)     │              │
│  └──────────┘                    └────┬─────┘              │
│                                      │ z_t (continuous)    │
│                                      ▼                     │
│  ┌──────────┐    A_t, L, s_t     ┌──────────┐             │
│  │  Action  │ ─────────────────→ │   DiT    │             │
│  │  + Cond  │                    │  (θ)     │             │
│  └──────────┘                    └────┬─────┘             │
│                                      │ h_t^θ (mid feature)│
│                                      ▼                    │
│                               ┌─────────────┐             │
│                               │  f_ψ (Proj) │             │
│                               └──────┬──────┘             │
│                                      │                    │
│                    ┌─────────────────┘                    │
│                    ▼                                      │
│          ┌───────────────────┐                           │
│          │  CosSim Alignment │  L_LARA = -CosSim(z, f(h))│
│          └───────────────────┘                           │
│                    │                                      │
│          ┌─────────┴─────────┐                           │
│          ▼                   ▼                           │
│     ∂/∂φ (update LAM)  ∂/∂θ (update DiT)                │
│     + ∂L_LAM             + ∂L_ACT                        │
└─────────────────────────────────────────────────────────────┘
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：
```
L(θ, φ, ψ) = L_ACT(θ) + w₁·(-CosSim[z_t^φ, f_ψ(h_t^θ)]) + w₂·L_LAM(φ)
```

**目标**：联合优化 VLA 动作生成、LAM-VAE 重建、以及两者表示对齐。

**公式拆解**：

| 项 | 含义 | 更新谁 |
|----|------|--------|
| $L_{\text{ACT}}(\theta)$ | Flow matching 动作生成损失：$\mathbb{E}\left[\Vert v_\theta(A_t^\tau, c_t) - (A_t - \varepsilon)\Vert^2\right]$ | $\theta$ (DiT) |
| L_LARA | 余弦相似度对齐：$-\text{CosSim}[z_t^\varphi, f_\psi(h_t^\theta)]$ | $\theta, \varphi, \psi$（三者都更新） |
| $L_{\text{LAM}}(\varphi)$ | VQ-VAE 重建损失：$\Vert I_{t+C} - \hat{I}_{t+C}\Vert^2 + $ commitment loss | $\varphi$ (LAM) |

**变量说明**：
- $\theta$: DiT（扩散动作生成器）参数
- $\varphi$: LAM（IDM + FDM + VQ）参数
- $\psi$: projection head 参数
- $z_t^\varphi$: LAM 的连续 latent action（量化前），维度未明确给出，参考 Moto-GPT 设计
- $h_t^\theta$: DiT 在 $L\text{-}2$ 层的中间特征
- $f_\psi$: 可学习 projection head，将 $h_t^\theta$ 投影到 $z_t$ 的空间
- $w_1$, $w_2$: 损失平衡超参数（论文 Appendix 中有具体值，待补充）

> 符号与本文保持一致：$\theta=\text{DiT}$, $\varphi=\text{LAM}$, $\psi=\text{projection head}$

## 3. 带数字走一遍：玩具例子 (Worked Example)

假设一个简化的 2D 操作场景：

**场景**：机械臂需要将红色方块从左移到右。

**Step 1 — LAM 前向**：
- 输入：$I_t$（方块在左）, $I_{t+C}$（方块在右）
- IDM 输出：z_t = [0.8, -0.3]（表示"向右移动"的潜在动作）
- VQ 量化：$z_t^q = \text{codebook}[7]$（最接近的码本条目）
- FDM 重建：$\hat{I}_{t+C} \approx I_{t+C}$（重建成功，$L_\text{LAM}$ 小）

**Step 2 — VLA 前向**：
- 输入：观察 I_t + 指令 L = "move red block right" + proprio s_t
- DiT 在 L-2 层输出中间特征：$h_t^\theta = [1.2, -0.5]$（未对齐前可能与 $z_t$ 方向不一致）
- Projection head：$f_\psi(h_t^\theta) = [0.7, -0.2]$

**Step 3 — 对齐损失**：
```
CosSim([0.8, -0.3], [0.7, -0.2]) = (0.56 + 0.06) / (sqrt(0.73) × sqrt(0.53)) ≈ 0.91
L_LARA = -0.91
```

如果 CosSim 低（比如 0.3），说明 VLA 的中间表示与 LAM 的潜在动作不一致——VLA 可能在生成"看起来合理但实际无效"的动作。对齐损失会推动两者靠近。

**Step 4 — 联合更新**：
- $\partial L/\partial \theta$: DiT 调整使 $h_t^\theta$ 更接近 $z_t$ 方向
- $\partial L/\partial \varphi$: LAM 调整 IDM 使 $z_t$ 更聚焦于控制相关变化
- 结果：$z_t$ 不再编码光照变化等伪影，$h_t^\theta$ 不再产生无效轨迹

## 4. 工程视角 (Engineering View)

| 维度 | 分析 |
|------|------|
| **额外参数量** | 仅 1 个 projection head $f_\psi$，预计 $< 1\text{M}$ 参数（相对 DiT + LAM 可忽略） |
| **训练开销** | 联合训练 vs 两阶段：多一次前向（LAM 不冻结），但无需额外数据 |
| **推理延迟** | 推理时 LARA 损失不参与，仅需 LAM（可选）+ DiT，延迟不变 |
| **内存占用** | 联合训练时需同时保留 LAM 和 DiT 的梯度，显存约增加 30-50% |
| **部署约束** | 后训练增强模式下，可用预训练 VLA + 预训练 LAM 快速适配新任务 |
| **模块化** | 即插即用——不修改 DiT 或 LAM 内部结构，仅需在训练循环中加入对齐损失 |

**工程含义**：LARA 的设计非常"轻量化"——它不是在架构层面加模块，而是在训练层面加约束。这意味着任何已有的 diffusion-based VLA 都可以通过 LARA 后训练获得提升，无需重新训练整个模型。

## 5. 数据与评测 (Data & Eval)

### 数据集

| 数据集 | 用途 | 说明 |
|--------|------|------|
| OXE (Open-X-Embodiment) | 预训练（OXE-Constrained 设置） | 多机器人演示数据集，论文限制仅使用 OXE 子集 |
| Unlabeled internet videos | LAM 预训练 | 包括人类和机器人交互视频 |
| LIBERO | 评测 | 4 个子任务：Spatial, Object, Goal, Long |
| SIMPLER-ENV | 评测 | 3 个子任务：Pick Coke Can, Object Movement, Drawer |
| GR1-Sim-24(30) | 评测 | 24 个双臂仿真任务，每任务 30 个演示微调 |
| G1-Real(50) | 真实世界评测 | Unitree G1 人形机器人，2 个复合任务 $\times 50$ 次演示 |

### 关键实验结果

**OXE-Constrained 设置（公平对比）**：

| 基准 | LARA (DiT-only) | LARA (full) | 提升 |
|------|-----------------|-------------|------|
| LIBERO Avg | 84.4% | 88.6% | +5.0% |
| SIMPLER-ENV Avg | 55.8% | 65.2% | +16.8% |
| GR1-Sim-24 | 6.4% | 11.4% | +78.1% |
| G1-Real Avg | 56.0% | 74.0% | +32.1% |

**Unconstrained 设置（SOTA 对比）**：

| 模型 | LIBERO Avg | SIMPLER-ENV Avg |
|------|-----------|-----------------|
| GR00T-N1.6 | 95.0% | 78.9% |
| GR00T-N1.6 + LARA | 95.6% | 79.9% |
| 提升 | +0.6% | +1.3% |

> 注意：Unconstrained 设置下 GR00T-N1.6 本身已很强，LARA 的边际提升较小（+1.3%），但在 OXE-Constrained 设置下提升显著（+16.8%），说明 LARA 在数据受限时价值更大。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

### 能做什么

| 能力 | 证据 | 条件 |
|------|------|------|
| 提升小样本 VLA 训练 | OXE-Constrained 下 LIBERO +5%, SIMPLER-ENV +16.8% | 需要 LAM 预训练 + 联合训练 |
| 后训练增强已有 VLA | GR00T-N1.6 + LARA 平均 +1.3% | 需要预训练 LAM |
| 改进 LAM pseudo-label 质量 | LAM refinement 模式下下游 VLA +15% | LAM 需经 LARA 预训练 |
| 跨 embodiment 泛化 | 多 embodiment 预训练后在 G1 人形上测试 | 需要 embodiment-specific MLP |

### 不能做什么 / 局限

| 局限 | 原因 |
|------|------|
| 真实世界验证有限 | G1-Real 仅 2 个任务 $\times 50$ 次演示，远不足以证明泛化 |
| Drawer 任务在 SIMPLER-ENV 上仍弱 | LARA (full) 仅 29.5%，远低于 Pick/Move |
| 对已很强的大模型提升有限 | GR00T-N1.6 在 Unconstrained 下仅 +1.3% |
| 依赖 LAM 预训练质量 | 如果 LAM 预训练不好，对齐起点就低 |

### 6.1 隐含假设 (Hidden Assumptions)

1. **DiT 中间特征与 LAM latent action 空间可对齐**：假设两者在高维空间中存在线性或近线性映射关系（通过 projection head $f_\psi$ 实现）。如果两者表征空间差异过大（例如 DiT 编码了 LAM 完全没学到的语义信息），对齐可能不充分。

2. **CosSim 足以捕捉对齐质量**：余弦相似度只关注方向，忽略幅度。如果 $z_t$ 和 $h_t^\theta$ 的幅度分布差异很大，CosSim 可能给出误导性的"对齐"信号。

3. **联合训练不会导致优化冲突**：$L_{\text{ACT}}$ 和 $L_{\text{LAM}}$ 的梯度方向可能不一致，$w_1$ 和 $w_2$ 的选取对结果敏感。论文未提供消融实验分析超参数的影响。

4. **LAM codebook size = 128 足够**：参考 Moto-GPT 设计，但未分析不同 codebook size 对 LARA 效果的影响。

## 7. 与相关工作对比 (Comparison)

| 方法 | LAM 使用方式 | 训练范式 | 核心创新 | 适用场景 |
|------|-------------|---------|---------|---------|
| **LAPA (Ye 2024)** | LAM 预训练 → 冻结 → tokens 做监督 | 两阶段 | 首次用 LAM tokens 做 VLA 监督 | 基础 LAM-based VLA |
| **Moto-GPT (Chen 2025)** | LAM 预训练 → 冻结 → 生成 pseudo-labels | 两阶段 | LAM + VLA 完整 pipeline | 通用 VLA 预训练 |
| **UniVLA (Bu 2025)** | LAM 集成到 VLA 架构内部 | 端到端 | 架构级集成 | 需要架构修改 |
| **TraceVLA (Zheng 2024)** | 冻结视觉特征做对齐 | 单阶段 | 表示对齐（冻结目标） | 已有 VLA 增强 |
| **LARA (本文)** | LAM 与 VLA 联合训练 | 三阶段 | 双向对齐（可更新目标） | 预训练 + 后训练增强 |

**面试 Tip**：如果被问到"LARA 和 TraceVLA 的区别"，回答：TraceVLA 用冻结的视觉特征做对齐目标（单向正则化），LARA 用可更新的 LAM latent action 做对齐目标（双向共进化），后者让 LAM 也能从 VLA 的真实动作轨迹中受益。

## 8. 精讀建議 (Reading Guide)

- **值得精讀原文的人**：
  1. 做 LAM-based VLA 预训练的研究者——LARA 的联合训练范式可以直接替换现有两阶段流程
  2. 评估后训练增强方案对已有 VLA 模型可行性的工程师——LARA 即插即用，无需改架构
  3. 研究表示对齐在生成模型中应用的研究者——LARA 将 REPA 的思想从图像生成迁移到动作生成

- **建議章節路徑**：先讀 §4（Method，核心对齐机制）→ 再看 §5.2（Full Training 实验）→ 可跳 §2（Related Works，除非需要文献综述）

- **不值得精讀的理由**：如果你不做机器人学习、已经熟悉 LAM 和扩散 VLA 的基础、且只关心纯行为克隆方法，读摘要和 §4.1 即可。


---
[← Back to Theory](./README.md)

**关键引用**：
- 论文: https://arxiv.org/abs/2606.07100
- 代码: https://github.com/lmy1001/LARA（尚未公开）
- LAM 基础: Moto-GPT (Chen et al. 2025), LAPA (Ye et al. 2024)
- 表示对齐灵感: REPA (Yu et al. 2024)