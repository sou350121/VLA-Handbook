# 扩散策略长上下文训练与评估深度拆解 (Training and Evaluating Diffusion Policies with Long Context Lengths)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-07-13
>
> **论文**: Training and Evaluating Diffusion Policies with Long Context Lengths
> **链接**: https://arxiv.org/abs/2606.16447
> **项目页**: https://dp-with-long-context.github.io/
> **作者**: Abhinav Agarwal, Adam Wei, Taylan Kargin, Michael Zeng, Cole Becker, Arif Kerem Dayı, Pablo Parrilo, Asuman Ozdaglar, Russ Tedrake — MIT
> **核心定位**: 系统性地重新评估扩散策略（Diffusion Policy）中"长上下文长度是否必然崩溃"这一争议，提出 UNet+Cross-Attention 架构选择 + Variable History Training 算法，在低数据场景下将长上下文策略成功率提升 1.25x-2x。

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | 在合适架构（UNet+Cross-Attention）和足够数据下，扩散策略的上下文长度从短扩展到长并不会灾难性崩溃；低数据场景下 Variable History Training 可显著改善 |
| 適合精讀 | 如果你在构建需要记忆能力的机器人策略（如多步操作、需要回溯历史状态），重点看 §4.1（架构对比）和 §4.2（Variable History Training） |
| 可以跳過 | 如果你只关心短上下文（To≤5）下的单步操作策略，这篇距离较远 |
| 落地可行性 | 高（纯架构改动 + 训练策略修改，无需额外数据或标注） |
| 主要風險 | 推理时延随上下文长度线性增长；所有实验在仿真环境完成，硬件验证仅限 100 条演示的 marshmallows 任务 |

💡 **X-Ray 开场**
扩散策略（Diffusion Policy）是当前机器人模仿学习的最优方法之一，但它通常只 conditioning 在最近几帧观测上——这让它无法处理需要记忆的任务（比如"把物体推回它最初出现的位置"）。此前多篇论文声称"简单地增加上下文长度会灾难性崩溃"，本文用近 200 个策略的系统性实验反驳了这一说法：崩溃不是因为长上下文本身，而是因为架构选择和训练策略不当。

📍 **研究全景时间线**
```
[2024] Diffusion Policy (Chi et al.) — 短上下文 To=16 成为事实标准
    ↓
[2025] Past-Token Prediction (Torne et al.) — 声称长上下文脆弱，用过去动作预测辅助
    ↓
[2026] BPP / MemER / MemoryVLA — 各自用 VLM 过滤/检索/记忆模块扩展上下文
    ↓
[本文] 系统性重评：Naive 扩展 + UNet+Cross-Attention + Variable History Training
    ← 当前位置：挑战"长上下文必然脆弱"的共识，回归简单方案
```

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 维度 | Diffusion Policy 原版 (Chi et al. 2024) | Past-Token Prediction (Torne et al. 2025) | 本文方案 |
|------|----------------------------------------|------------------------------------------|----------|
| 上下文长度 To | 16（短） | 可变（长） | 4-92（覆盖短→长全范围） |
| 去噪骨干 | UNet + FiLM | DiT | UNet + Cross-Attention |
| 条件注入方式 | FiLM（特征级线性调制） | Transformer 自注意力 | Cross-Attention（观测→动作） |
| 过去动作预测 | 是（Tp_past = To） | 是（核心辅助损失） | 是（标准配置，非核心创新） |
| 观测编码器 | 可训练 | 冻结（声称仅为节省显存） | 可训练（默认）/ 冻结（对比实验） |
| 训练策略 | 固定 To | 固定 To | Variable History（课程学习） |
| 数据规模覆盖 | 单规模 | 单规模 | N/2, N, 2N 三档 |
| 实验策略数 | ~10 | ~10 | ~200 |

### 1.2 关键机制 (Key Mechanism)

**为什么 UNet + Cross-Attention 优于 DiT？**

DiT（Diffusion Transformer）将观测 token 与动作 token 拼接后通过自注意力处理。在短上下文下表现尚可，但当 To 增长到 20+ 时，DiT 的参数量和注意力矩阵呈平方增长，且缺乏归纳偏置来区分"哪些历史帧是相关的"。在低数据 regime 下，DiT 在 push-and-return 任务上出现**灾难性崩溃**（成功率趋近 0%）。

UNet + Cross-Attention 则将观测编码后的 embedding 通过 cross-attention 注入到 UNet 的解码层。这种方式：
- 保持 UNet 的空间归纳偏置（对图像特征友好）
- Cross-attention 让模型**自主选择关注哪些历史帧**，而非强制融合所有帧
- 参数量增长更温和，低数据下不易过拟合

⚡ **Eureka Moment**：长上下文策略的崩溃不是"长上下文"本身的错——是 DiT 架构和低数据 regime 共同导致的假象。换对架构（UNet+Cross-Attention）+ 足够数据，naive 扩展上下文长度就能成功。

### 1.3 信息流/架构图 (Flow / Diagram)

```
┌─────────────────────────────────────────────────────────────┐
│                    Observation History                       │
│   o[t-To+1], o[t-To+2], ..., o[t]    (To = 4 ~ 92 frames)  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │  Obs Encoder   │  (ResNet/VIT, frozen or trainable)
              │  e[t-To+1:t]   │
              └────────┬───────┘
                       │ embeddings
                       ▼
┌──────────────────────────────────────────────────────────────┐
│              Denoising Backbone (UNet + Cross-Attention)      │
│                                                               │
│   noisy_action ──► UNet Encoder ──► UNet Decoder              │
│                      │               │                         │
│                      │    Cross-Attn ◄── e[t-To+1:t]          │
│                      │    (conditioning)                       │
│                      ▼                                       │
│              clean_action ~ P(a | o[1:To])                    │
└──────────────────────────────────────────────────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │  Execute on    │
              │  Robot / Sim   │
              └────────────────┘
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：
```
L = E[ || π_θ(o[k-m+1:k]) - a[k-Tp_past(m)+1 : k+Tp] ||² ]
```

其中 m ~ ρ_i 是从课程分布中采样的上下文长度（Variable History Training 时 m 变化；naive scaling 时 m=To 固定）。

**目标**：最小化扩散去噪损失，使策略 π_θ 从观测历史 o[1:To] 中预测动作序列 a。

**变量说明**：

| 符号 | 含义 | 本文取值范围 |
|------|------|-------------|
| To | 上下文长度（观测历史帧数） | 1-92 |
| Tp | 未来预测 horizon | 通常 8-16 |
| Tp_past | 过去动作预测长度 | 通常 = To（naive）/ 小值（VHT） |
| m | 训练时采样的上下文长度 | c ~ To（Variable History） |
| ρ_i | 课程分布（第 i 步） | Random Sprinkle / Progressive |
| c | 最小上下文长度 | 通常 1-4 |

**直觉**：Variable History Training 的核心是让模型在训练时随机看到不同长度的上下文，从而学会"有长上下文时用长上下文，没有时退化为短上下文策略"。这类似于 NLP 中的变长序列训练——避免模型对单一长度过拟合。

> 符号与本文/原始 Diffusion Policy 文档保持一致。扩散损失的具体形式（DDPM 的噪声预测 MSE 或 ELBO）沿用 Ho et al. 2020 标准，本文未做修改。

## 3. 带数字走一遍：玩具例子 (Worked Example)

假设一个简化场景：

**任务**：Grasp-and-Return，上下文长度 To ∈ {4, 8, 16, 24, 48}
**数据**：低数据 regime，N/2 = 50 条演示
**两种训练方式对比**：

### 方式 A：Naive Scaling（固定 To=48）
- 每条演示截取 48 帧，模型始终看到 48 帧
- 50 条 × 48 帧 = 2400 个训练样本
- 模型容易过拟合这 2400 个样本的特定时序模式
- 闭环部署时，分布偏移导致性能急剧下降
- **预期成功率**：~35%（论文图 5a，N/2 数据下 To=48）

### 方式 B：Variable History Training（Progressive 课程）
- 前 50% 训练步：m 从 4 逐步增加到 48
  - 早期：模型主要学习 4-8 帧的短上下文模式（50 条 × 4 帧 = 200 个"有效"样本，信噪比高）
  - 中期：逐步引入 16-24 帧，模型学会利用更多历史
  - 后期：引入 48 帧，但模型已有短上下文基础
- 后 50% 训练步：Random Sprinkle（p=0.8 采样 To=48，0.2 均匀采样 4-47）
- **预期成功率**：~65%（论文图 5a，VHT Progressive vs naive To=48）

**关键差异**：VHT 让模型在低数据下先学会"短上下文够用时就够用"，再逐步学习"什么时候需要长上下文"。这避免了 naive scaling 中模型被迫用 50 条数据拟合 48 帧复杂时序模式的过拟合问题。

## 4. 工程视角 (Engineering View)

| 工程维度 | 短上下文 (To≤16) | 长上下文 (To≥48) | 工程含义 |
|----------|-----------------|-----------------|----------|
| 推理延迟 | 基准 | ~3x 增长（编码 To 帧图像） | 闭环控制频率可能从 10Hz 降至 3-4Hz |
| 显存占用 | 基准 | ~3x 增长（编码 + cross-attention） | 单卡训练 To=80 需 ~24GB VRAM |
| 数据需求 | 50-100 条 | 100-200 条（取决于操作复杂度） |  prehensile（抓握）任务数据效率更高 |
| 量化误差 | 低 | 中高（更多帧 → 更多累积误差） | 部署时可能需要混合精度策略 |
| 模块边界 | 清晰（单帧编码 → 动作） | 模糊（历史帧间的时序依赖） | 需仔细设计观测编码器的时序处理方式 |

**部署约束**：
- 本文所有仿真实验在标准 GPU 上完成，推理延迟未详细报告
- 第 5 节明确提到"inference time overhead due to longer context length remains a challenge"
- 建议方向：蒸馏到轻量架构、动态上下文长度选择（推理时自适应选择 To）

## 5. 数据与评测 (Data & Eval)

### 数据集与规模

| 任务 | 类型 | N（标准数据量） | N/2（低数据） | 2N（高数据） |
|------|------|----------------|---------------|-------------|
| Push-T | 短上下文可解 | 200 条 | 100 条 | 400 条 |
| Square (Robomimic) | 短上下文可解 | 50 条 | 25 条 | 100 条 |
| Lift (Robomimic) | 短上下文可解 | 50 条 | 25 条 | 100 条 |
| Push-and-Return | 需长上下文 | 200 条 | 100 条 | 400 条 |
| Grasp-and-Return | 需长上下文 | 100 条 | 50 条 | 200 条 |
| Marshmallows（硬件） | 需长上下文 | 100 条 | — | — |

### 评测指标

- **Task Success**：任务是否成功完成（二元）
- **Manipulation Completion**：放宽记忆要求后，操作本身是否完成
- **Contextual Success**：Manipulation Completion 中，正确利用历史信息的比例

```
Contextual Success = Task Success / Manipulation Completion
```

这个分解非常关键——它区分了"学不会操作"和"学不会记忆"两种失败模式。

### 评测环境
- 仿真：MuJoCo / Robomimic 标准环境
- 硬件：Franka 机械臂（marshmallows 任务，To=92）

## 6. 能力与失败模式 (Capabilities & Failure Modes)

### 能做什么

| 能力 | 条件 | 证据 |
|------|------|------|
| 长上下文记忆（To=48-80） | UNet+Cross-Attention + N 数据 | 图 4a, 5a |
| 低数据下长上下文学习 | VHT Progressive + 短 Tp_past | 图 8 |
| 硬件部署（有限验证） | To=92, 100 demos | 图 6 |
| 自适应上下文利用 | VHT 训练的模型可退化为短上下文 | 图 8 短上下文性能未下降 |

### 不能做什么

| 失败模式 | 原因 | 场景 |
|----------|------|------|
| DiT 架构在长上下文下崩溃 | 注意力矩阵过大 + 低数据过拟合 | Push-and-return, To≥20 |
| 低数据 + naive scaling | 模型对长时序模式过拟合 | Push-T, N/2, To≥20 |
| 推理延迟高 | 需编码 To 帧图像 | 所有长上下文策略 |
| 多任务泛化未验证 | 本文仅评估单任务策略 | — |

### 6.1 隐含假设 (Hidden Assumptions)

1. **观测编码器质量不变**：本文假设 ResNet/VIT 编码器的特征质量不随上下文长度变化。但实际上，长上下文可能暴露编码器的分布外样本问题。
2. **演示数据质量一致**：所有数据规模假设演示质量相同。现实中，更多演示 = 更多噪声演示。
3. **单任务设定**：所有实验在单任务上训练/评估。多任务场景下，长上下文的样本复杂度会进一步增加。
4. **仿真→真实的 gap 可忽略**：除 marshmallows 外，无硬件验证。仿真中的成功不代表 real-world 可复现。
5. **冻结编码器"无影响"的反驳**：本文发现冻结编码器确实影响性能（图 9），但仅在特定任务上。这个假设的破坏程度因任务而异。

## 7. 与相关工作对比 (Comparison)

| 方法 | 核心思路 | 架构 | 数据效率 | 长上下文能力 | 复杂度 |
|------|---------|------|---------|-------------|--------|
| Diffusion Policy (原版) | 短上下文 To=16 | UNet+FiLM | 高（短上下文） | 不支持 | 低 |
| Past-Token Prediction (Torne et al.) | 过去动作预测辅助 | DiT | 中（需冻结编码器） | 中 | 中 |
| BPP (Mark et al.) | VLM 过滤关键帧 | UNet | 高（减少帧数） | 间接支持 | 高（需 VLM） |
| MemER (Sridhar et al.) | 经验检索增强记忆 | — | 中 | 中 | 高（检索模块） |
| MemoryVLA | 感知-认知双记忆 | VLA | 中 | 高 | 很高 |
| **本文** | **架构选择 + 课程学习** | **UNet+Cross-Attention** | **高（VHT 降低样本复杂度）** | **高（To 最高 92）** | **低（无额外模块）** |

**面试 Tip**：当被问到"长上下文策略为什么总是训练不稳定"时，回答："这不一定是长上下文的问题——先检查你的架构。DiT 在长上下文低数据下会灾难性崩溃，但 UNet+Cross-Attention 可以 naive 扩展到 To=80 而不崩溃。如果数据有限，加上 Variable History Training 课程学习，效果进一步提升 1.25x-2x。"

## 8. 精讀建議 (Reading Guide)

**值得精讀原文的人**：
- 正在构建需要多步记忆能力的机器人策略的研究者（如"回到原点"类任务）
- 评估 Diffusion Policy 架构选择（UNet vs DiT, FiLM vs Cross-Attention）的工程团队
- 对模仿学习中"因果混淆"（causal confusion）问题感兴趣的研究者

**建議章節路徑**：
1. 先读 §3（Evaluating Long-Context）— 理解 naive scaling 在不同任务/数据下的真实表现
2. 再看 §4.1（Cross-Attention）— 架构选择的定量证据
3. 然后 §4.2（Variable History Training）— 低数据场景的实用算法
4. 可跳过 §4.3（Past-Token Prediction 重评）— 除非你对 Torne et al. 的工作特别熟悉

**不值得精讀的理由**：
- 如果你只做短上下文（To≤16）的单步操作，这篇的增量价值有限
- 如果你关注的是多任务/基础策略（foundation policy），本文的单任务设定距离较远
- 如果你关注 VLA（Vision-Language-Action）而非纯视觉扩散策略，本文不涉及语言模态

---
[← Back to Theory](./README.md)

**关键引用**：
- 论文: https://arxiv.org/abs/2606.16447
- 项目页: https://dp-with-long-context.github.io/
- Diffusion Policy (原版): https://arxiv.org/abs/2303.04137
- Past-Token Prediction: https://arxiv.org/abs/2505.09561
