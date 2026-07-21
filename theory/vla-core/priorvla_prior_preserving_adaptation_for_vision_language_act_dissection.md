# PriorVLA：保留先验的 VLA 微调框架 (Prior-Preserving Adaptation for Vision-Language-Action Models)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-05-13
>
> **论文**: PriorVLA: Prior-Preserving Adaptation for Vision-Language-Action Models
> **链接**: https://arxiv.org/abs/2605.10925
> **核心定位**: 将 VLA 微调从"用全量参数拟合下游数据"重构为"保留预训练先验 + 学习利用先验"，仅用全量微调 25% 的参数即实现更强的 OOD 泛化与少样本适应。

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | 冻结 Prior Expert 作为先验源 + 训练 Adaptation Expert 做下游特化，通过 Expert Queries 桥接，比全量微调在 OOD 和少样本场景下表现更好 |
| 適合精讀 | 如果你在研究 VLA 下游适配、灾难性遗忘、或 flow-matching 策略蒸馏，重点看 §3.3-§3.5（方法）和 §4.3（消融） |
| 可以跳過 | 如果你只关心 ID 场景下的最大化性能提升（LIBERO 上全量微调已接近饱和），这篇的增量有限 |
| 落地可行性 | 中（需要双 expert 推理，推理开销约 $2\times$ 前向，但参数量仅 $25\%$ 可训练） |
| 主要風險 | 实验仅在 $\pi_{0.5}$ 上验证；未公开代码；双 expert 推理延迟可能成为部署瓶颈 |

💡 **X-Ray 开场**

全量微调 VLA 时，预训练学到的广泛操作先验会被下游有限数据"拉偏"，导致 OOD 泛化能力下降。PriorVLA 的核心发现是：**把预训练 Action Expert 冻结为只读先验源，同时训练一个平行的 Adaptation Expert，通过可学习的 Query 接口把场景先验和运动先验注入下游策略**——这样只用 25% 的可训练参数，就能在 OOD 和少样本场景下全面超越全量微调。对 VLA 研究者的含义：微调不再必须是"全有或全无"，保留先验本身就是一种正则化。

📍 **研究全景时间线**

```
2023  Diffusion Policy (独立策略) → 2024  RT-1/RT-2 (通用 VLA 出现)
  → 2024-2025  OpenVLA, π0, RDT (大规模预训练)
    → 2025  OpenVLA-OFT, VLA-Adapter, Ki (适配方法：减少参数/桥接特征)
      → 2026-05  PriorVLA ← 当前位置：显式保留预训练前向传播先验 + 可学习接口
        ← 局限：仅 π0.5 验证 / 未开源 / 推理 2× 开销
```

## 1. 核心架构/方法总览 (Overview / Architecture)

PriorVLA 建立在 $\pi_{0.5}$ 架构之上（VLM 视觉编码器 + Flow-Matching Action Expert），引入两个耦合模块：**Dual Action Experts (DAE)** 和 **Expert Queries (EQ)**。

### 1.1 系统对比概览 (System Component Comparison)

| 组件 | 全量微调 (Full FT) | LoRA | PriorVLA |
|------|---------------------|------|----------|
| 可训练参数比例 | 100% | ~5-10% | ~25% |
| VLM 视觉编码器 | 全部微调 | 冻结 + LoRA | 全部微调 |
| VLM 语言/跨模态层 | 全部微调 | 冻结 + LoRA | 冻结（原始权重） |
| Action Expert | 全部微调 | 冻结 + LoRA | 冻结 Prior Expert + 全新 Adaptation Expert |
| 先验保留机制 | 无（参数被覆盖） | 间接（低秩约束） | 显式（冻结 PE 作为只读先验源） |
| 先验利用方式 | 无 | 无 | Expert Queries（SQ/MQ/AQ） |
| 推理时前向次数 | $1\times$ | $1\times$ | $2\times$（PE + AE 各一次） |
| 优势场景 | ID 数据充足 | 参数量受限 | OOD / 少样本 / 数据稀缺 |

### 1.2 关键机制 (Key Mechanism)

**Dual Action Experts (DAE)** 的核心设计哲学是"分离关注点"：

- **Prior Expert (PE)**：冻结的预训练 Action Expert，不参与梯度更新。它的作用不是输出最终动作，而是作为一条只读的前向通路，在其内部表示中提供预训练学到的运动先验。
- **Adaptation Expert (AE)**：从相同预训练权重初始化，专门针对下游任务训练。只有它的去噪输出用于更新动作轨迹和最终动作生成。

两者在相同的去噪轨迹上并行执行，接收相同的噪声动作块，但 PE 的输出被丢弃，仅 AE 的输出驱动推理。

**Expert Queries (EQ)** 是三个可学习的 token 组，充当先验的"接口"：

| Query 类型 | 位置 | 功能 | 注意力模式 |
|-----------|------|------|-----------|
| Scene Queries (SQ) | VLM 输入序列中 | 从 VLM 捕获任务相关的场景先验 | 自注意力 + 读取 OBS tokens |
| Motor Queries (MQ) | 追加到 PE | 从 PE 的去噪表示中捕获运动先验 | 自注意力 + 读取 PE 噪声动作 tokens |
| Action Queries (AQ) | 插入 AE 的噪声动作 tokens 旁 | 整合 SQ 和 MQ 的先验，指导动作生成 | 自注意力 + 读取 OBS + SQ + MQ |

⚡ **Eureka Moment**：预训练的价值不只是初始化权重——预训练模型在一次前向传播中产生的内部表示本身就是可提取、可利用的"先验信号"。保留这些信号并让下游策略学会读取它们，比直接覆盖所有参数更有效。

### 1.3 信息流/架构图 (Flow / Diagram)

```
┌─────────────────────────────────────────────────────────────┐
│                    预训练 VLM (部分冻结)                      │
│  OBS tokens ────────────────────────────────────────────→   │
│                  ↑ 自注意力路径保留                           │
│  SQ tokens ────┘ 读取场景先验                                │
└─────────────────────────────────────────────────────────────┘
                              │ SQ 的 K/V caches
                              ▼
┌─────────────────────┐    ┌─────────────────────────────────┐
│  Prior Expert (冻结) │    │  Adaptation Expert (可训练)      │
│  噪声动作 → 内部表示  │    │  噪声动作 → ──────────────────→ │
│  (输出被丢弃)         │    │    ↑ 读取 SQ + MQ + OBS         │
│         ↑            │    │    AQ 整合多源先验               │
│  MQ 读取 ────────────┘    │    AE 输出更新轨迹 (唯一使用)     │
└─────────────────────────┘    └─────────────────────────────────┘
         │ MQ 的 K/V caches                 │
         └──────────────────────────────────┘
                    注入 AE
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：

```
L = L_FM(a_t:AE)    其中 AE 的输入 = f(SQ(VLM), MQ(PE), OBS)
    且 PE 冻结,  ∂PE/∂θ = 0
```

**目标**：标准 Flow-Matching 均方误差，仅作用于 Adaptation Expert 的去噪预测。

```
L = E[‖AE_denoise(a_tilde, SQ, MQ, OBS) - a_target‖²]
```

**变量说明**：

| 符号 | 含义 |
|------|------|
| `a_tilde` | 去噪步骤 $\tau$ 的噪声动作块 |
| `PE` | 冻结的 Prior Expert，内部表示 h_PE 提供运动先验 |
| `AE` | 可训练的 Adaptation Expert，输出唯一用于动作更新 |
| `SQ` | Scene Queries，从 VLM 捕获的场景先验 token |
| `MQ` | Motor Queries，从 PE 捕获的运动先验 token |
| `OBS` | 原始 VLM 输入 tokens（视觉 + 语言 + 本体感觉） |
| `H` | 动作视界 (action horizon) |

**Scene Queries** 通过自注意力从 VLM 提取场景先验：

```
h_sq^(l+1) = Attn(Q_sq^l, K_obs^l ‖ K_sq^l, V_obs^l ‖ V_sq^l)
```

**Motor Queries** 从冻结 PE 提取运动先验（单向读取，不干扰 PE 前向）：

```
h_mq^(l+1) = Attn(Q_mq^l, K_mq^l ‖ K_a_pe^l, V_mq^l ‖ V_a_pe^l)
```

**Action Queries** 整合多源先验：

```
h_aq,a_ae^(l+1) = Attn(Q, K_aq ‖ K_obs ‖ K_sq ‖ K_mq, V_aq ‖ V_obs ‖ V_sq ‖ V_mq)
```

> 符号与论文保持一致：`‖` 表示 token 拼接，上标 `l` 表示层索引，`a_pe` 表示 Prior Expert 的噪声动作 tokens。

## 3. 带数字走一遍：玩具例子 (Worked Example)

假设一个简化的 2D 抓取任务，动作空间为 (x, y)，我们用具体数值走一遍 PriorVLA 的去噪过程。

**设定**：
- 动作视界 H = 2（预测 2 步动作）
- 去噪步数 = 5 步
- 目标动作 a_target = [0.5, 0.3]（单位：米）
- 初始噪声 a_tilde_0 = [0.1, -0.2]

**Step-by-step 去噪**：

```
τ=0:  a_tilde = [0.10, -0.20]
      PE 前向: h_PE^0 = PE(a_tilde, OBS) → 内部表示 z_PE = [0.45, 0.28] (预训练先验)
      MQ 读取: h_mq = Attn(Q_mq, K_mq‖K_a_pe, V_mq‖V_a_pe) → 提取 z_mq ≈ [0.42, 0.25]
      SQ 读取: h_sq = Attn(Q_sq, K_obs‖K_sq, V_obs‖V_sq) → 提取 z_sq ≈ [0.50, 0.30]
      AQ 整合: h_aq = Attn(Q, K‖K_obs‖K_sq‖K_mq, V‖V_obs‖V_sq‖V_mq)
      AE 输出: f_AE^0 = [0.35, 0.15]
      FM 更新: a_tilde_1 = a_tilde_0 + f_AE^0 * Δτ = [0.10+0.35*0.2, -0.20+0.15*0.2] = [0.17, -0.17]

τ=1:  a_tilde = [0.17, -0.17]
      PE 前向: h_PE^1 → z_PE = [0.44, 0.27] (冻结，先验稳定)
      AE 输出: f_AE^1 = [0.32, 0.18]
      FM 更新: a_tilde_2 = [0.23, -0.13]

τ=2:  a_tilde = [0.23, -0.13]
      AE 输出: f_AE^2 = [0.28, 0.22]
      FM 更新: a_tilde_3 = [0.29, -0.09]

τ=3:  a_tilde = [0.29, -0.09]
      AE 输出: f_AE^3 = [0.22, 0.28]
      FM 更新: a_tilde_4 = [0.33, -0.03]

τ=4:  a_tilde = [0.33, -0.03]
      AE 输出: f_AE^4 = [0.17, 0.33]
      FM 更新: a_final = [0.36, 0.04]
```

**损失计算**（仅 AE）：

```
L = ‖[0.36, 0.4] - [0.5, 0.3]‖² = ‖[-0.14, 0.10]‖² = 0.0196 + 0.0100 = 0.0296
```

**关键观察**：PE 在整个过程中提供稳定的运动先验（$z_{\text{PE}} \approx [0.45, 0.28]$ 始终接近目标），MQ 将其传递给 AE。AE 学会"读取"这个先验并在此基础上做下游特化调整，而不是从零开始学习。这就是 PriorVLA 在少样本下仍能保持合理输出的原因——PE 的先验充当了隐式正则化。

## 4. 工程视角 (Engineering View)

| 维度 | 全量微调 | PriorVLA | 工程含义 |
|------|---------|----------|---------|
| 训练参数量 | 100% | ~25% | GPU 显存占用显著降低，单卡可训练更大模型 |
| 推理前向次数 | $1\times$ | $2\times$ | 推理延迟翻倍，需考虑实时性约束（控制频率 10-20Hz 时可能成为瓶颈） |
| 推理显存 | 模型权重 $\times 1$ | 模型权重 $\times 2$（PE + AE 各一份） | 显存需求翻倍，但 AE 可与 PE 共享部分权重（初始化相同） |
| 训练稳定性 | 全参数更新，可能震荡 | 冻结 PE 提供稳定先验，训练更稳定 | 消融实验证实：让 PE 可训练反而性能下降 |
| 部署复杂度 | 低（单模型） | 中（需维护两个 expert 的前向逻辑） | 需要修改推理引擎以支持双 expert 并行 |
| 量化兼容性 | 标准 | 待验证 | PE 冻结适合 INT8 量化降低显存，但 AE 仍需 FP16 |

**部署建议**：如果目标场景是 ID 数据充足且延迟敏感（如 $50\,\text{Hz}$ 高频控制），全量微调可能仍是更简单的选择。PriorVLA 的价值主要体现在 OOD 泛化和少样本场景——这些场景下 $2\times$ 推理开销通常是可接受的 trade-off。

## 5. 数据与评测 (Data & Eval)

### 数据集

| 基准 | 描述 | 数据规模 | 评估设置 |
|------|------|---------|---------|
| RoboTwin 2.0 | 双臂仿真操作，13 任务子集 | 50 演示/任务（标准） | Easy (ID) / Hard (OOD) |
| LIBERO | Franka 仿真操作，4 套件 | 标准协议 | Spatial / Object / Goal / Long |
| 真实世界 (Franka) | 单臂真实机器人 | 100-300 演示/任务（标准）; 10 演示/任务（少样本） | ID / OOD（光照/背景/物体位置/桌高扰动） |
| 真实世界 (AC-One) | 双臂真实机器人平台 | 同上 | 同上 |

### OOD 扰动因素

论文定义了 4 个 OOD 扰动维度：Light（光照变化）、Background（背景变化）、Object Position（物体位置偏移）、Table Height（桌面高度变化）。这些扰动在真实世界评估中组合使用。

### 训练设置

- 基础模型：$\pi_{0.5}$
- 训练步数：30k（除非另有说明）
- 优化目标：Flow-Matching MSE（仅 AE 输出）
- 可训练组件：Adaptation Expert + 三种 Expert Queries + VLM 视觉编码器

## 6. 能力与失败模式 (Capabilities & Failure Modes)

### 能做什么

| 场景 | 表现 | 原因 |
|------|------|------|
| OOD 泛化（仿真） | RoboTwin Hard: 53% vs $\pi_{0.5}$ 42% (+11) | PE 冻结保留预训练运动先验，防止过拟合 ID 分布 |
| OOD 泛化（真实） | 57% vs $\pi_{0.5}$ 41% (+16) | 先验在真实扰动下比过拟合的 ID 策略更鲁棒 |
| 少样本学习（10 演示） | ID 48% vs $\pi_{0.5}$ 24% (+24); OOD 32% vs $\pi_{0.5}$ 10% (+22) | PE 先验补充了极少量数据的不足 |
| 饱和 ID 基准 | LIBERO 99.1% vs $\pi_{0.5}$ 96.9% | 先验保留在 ID 场景下也不损害性能 |

### 不能做什么 / 局限

| 局限 | 说明 |
|------|------|
| 仅 $\pi_{0.5}$ 验证 | 未在 OpenVLA、RT-1、GROOT-N1 等其他 VLA 架构上验证泛化性 |
| 推理延迟 $2\times$ | PE + AE 双前向，对高频控制（>30Hz）场景可能不实用 |
| 未开源代码 | 截至论文发布，代码未公开，独立复现存在障碍 |
| 单任务独立训练 | 每个任务训练独立模型（RoboTwin 仅 13/50 任务），未评估多任务联合适配 |
| 仅操作任务 | 实验局限于桌面操作，未涉及移动机器人、双臂协调（除 AC-One 基础任务）、人形机器人 |

### 6.1 隐含假设 (Hidden Assumptions)

1. **预训练先验对下游任务始终有益**：论文假设冻结 PE 的先验在任何下游场景都有帮助。但如果预训练数据与下游任务分布差异极大（如预训练主要是桌面操作，下游是室外移动操作），PE 的先验可能成为干扰而非帮助。
2. **Expert Queries 能学会有效提取先验**：SQ/MQ/AQ 的可学习接口假设这些 token 能通过梯度优化学会"读取"正确的先验信息。但在极少样本（如 10 演示）下，Query 本身的训练可能不充分。
3. **$2\times$ 推理开销可接受**：论文未深入讨论推理延迟对实时控制的影响。在需要 50Hz 控制频率的场景，$2\times$ 开销可能需要硬件升级。
4. **$\pi_{0.5}$ 的 Flow-Matching 架构具有代表性**：PriorVLA 完全构建在 $\pi_{0.5}$ 的 FM 架构上。对于基于扩散（Diffusion Policy）或自回归（RT-2）的 VLA，该框架的直接迁移性未知。

## 7. 与相关工作对比 (Comparison)

| 方法 | 核心思路 | 可训练参数 | 先验保留方式 | 适用场景 |
|------|---------|-----------|-------------|---------|
| Full FT (OpenVLA-OFT) | 全量微调所有参数 | 100% | 无（隐式） | ID 数据充足 |
| LoRA | 低秩矩阵注入 | ~5-10% | 间接（低秩约束） | 参数量受限 |
| VLA-Adapter | 桥接冻结 VLM 特征到动作策略 | ~10-20% | 冻结 VLM | VLM 保护优先 |
| Ki (Knowledge Injection) | 桥接冻结 VL 特征到行动策略 | ~10-20% | 冻结 VLM + 注入 | 视觉-语言解耦 |
| MAPS | 约束参数更新幅度 | 100% | 间接（约束优化） | 防止灾难遗忘 |
| Retain | 合并参数更新 | 100% | 间接（权重合并） | 持续学习 |
| **PriorVLA** | **冻结 PE + 平行 AE + Expert Queries** | **~25%** | **显式（冻结 PE 作为先验源 + Query 接口）** | **OOD / 少样本** |

**面试 Tip**：当被问到"PriorVLA 和 LoRA 有什么区别"时，可以这样回答："LoRA 通过低秩约束间接限制参数更新幅度来保留先验，而 PriorVLA 显式地将预训练 Action Expert 冻结为只读先验源，同时训练一个平行的 Adaptation Expert，通过可学习的 Query 接口读取先验。前者是'约束更新'，后者是'分离保留与特化'。在 OOD 和少样本场景下，PriorVLA 的显式先验保留策略效果更显著。"

## 8. 精讀建議 (Reading Guide)

- **值得精讀原文的人**：
  1. 研究 VLA 下游适配/灾难性遗忘的研究者——PriorVLA 的"先验保留"视角提供了不同于正则化/约束的新思路
  2. 需要评估从仿真迁移到真实机器人可行性的工程师——少样本 OOD 结果（10 演示下 48% ID / 32% OOD）对数据稀缺场景有直接参考价值
  3. 对 Flow-Matching 策略蒸馏/双 expert 架构感兴趣的开发者——DAE + EQ 的设计模式可迁移到其他 FM-based 策略

- **建議章節路徑**：先讀 §3.2-§3.5（方法核心：DAE + EQ + 训练）→ 再看 §4.2-§4.3（实验结果 + 消融）→ 可跳 §2（相关工作，除非需要定位论文在文献中的位置）

- **不值得精讀的理由**：如果你不做机器人操作适配、已经熟悉 LoRA/Adapter 类方法且只关心 ID 性能最大化，读摘要和 §1 即可——PriorVLA 的核心贡献在于 OOD 和少样本场景，ID 饱和场景（如 LIBERO）上的增量相对有限。


---
[← Back to Theory](./README.md)

**关键引用**：
- 论文: https://arxiv.org/abs/2605.10925
- 项目页: https://priorvla.github.io/
- 基础模型 $\pi_{0.5}$: https://www.physicalintelligence.company/download/pi0.pdf
- RoboTwin 2.0: https://arxiv.org/abs/2605.10925 (ref [robotwin2])
- LIBERO: https://arxiv.org/abs/2605.10925 (ref [libero])
