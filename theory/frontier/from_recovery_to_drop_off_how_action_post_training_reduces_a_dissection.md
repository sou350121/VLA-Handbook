# 从恢复到崩塌：动作后训练如何削弱 VLM 的深层深度可解码性 (From Recovery to Drop-off: How Action Post-training Reduces a VLM's Late-Layer Depth Decodability)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-08-13
>
> **论文**: From Recovery to Drop-off: How Action Post-training Reduces a VLM's Late-Layer Depth Decodability
> **链接**: https://arxiv.org/abs/2608.08904
> **核心定位**: 诊断动作后训练（action post-training）对 VLM 空间理解能力的破坏——精确到哪个模块、哪几层、以及为什么。

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | 动作后训练使 VLA 在每一层深度解码能力都下降（floor），且在最后 7 层出现额外崩塌（cliff），根因是 late-layer MLP writes 干扰了几何信息 |
| 適合精讀 | 如果你在做 VLA 架构设计、表征退化分析、或需要理解 backbone 预训练知识在 action tuning 后的保留程度，重点看 §2 和 §4.2 |
| 可以跳过 | 如果你只关心 VLA 端到端任务性能或工程部署，这篇距离中等 |
| 落地可行性 | 中——论文是诊断性研究，不提供修复方案，但为修复指明了方向（保护 late MLP writes） |
| 主要風險 | 仅研究一对模型（Molmo2-ER / MolmoAct2-LIBERO）在一个数据集（LIBERO）上，泛化性待验证 |

💡 **X-Ray 开场**
VLA 通过将预训练 VLM 进行动作后训练获得。但这个过程牺牲了什么？本文用逐层探测发现：VLM 的空间深度理解在变成 VLA 后，每一层都变差了（floor），而且最后 7 层出现额外崩塌（cliff）。因果定位实验揭示：这是最后几层的 MLP 写入（MLP writes）干扰了几何信息——删掉这些 MLP 写入，可以恢复大部分深度可解码性。对 VLA 研究者意味着：动作训练正在系统性地破坏 backbone 的空间表征，且破坏集中在最后几层的 MLP 子层。

📍 **研究全景时间线**
```
[2022] RT-1 开创 VLA 范式 → [2023] RT-2 将动作 token 化 → [2024] π0 引入 flow matching
→ [2025] 多篇报告表征退化，提出对齐/一致性辅助损失
→ [本文 2026] 首次用因果定位精确到模块+层级：late MLP writes 是深度崩塌的根源
← 当前位置：诊断完成，修复方案待探索
```

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 维度 | Molmo2-ER (Base VLM) | MolmoAct2-LIBERO (VLA) |
|------|---------------------|----------------------|
| Backbone | 36 层 decoder, hidden dim=2560 | 同左（weight-matched） |
| 视觉编码器 | SigLIP2 | 同左 |
| 训练目标 | 标准 next-token loss | 三阶段：DCT-BPE action token 训练 → flow-matching expert（backbone frozen）→ LIBERO 全量微调 |
| 动作专家 | 无 | 有，cross-attention 到 backbone 的 KV |
| 深度解码能力（末层 d1） | ~0.76 | ~0.51（下降 0.25） |
| 末层趋势 | 上升（+0.040 from L28） | 下降（-0.123 from L28） |

训练分三阶段：
1. **Action Token 训练**：DCT-BPE 离散化轨迹，在混合机器人-多模态语料上训练 next-token loss
2. **Flow-matching Expert**：添加 action expert cross-attention 到 backbone KV，backbone frozen（knowledge insulation）
3. **LIBERO 全量微调**：释放 insulation，action gradients 到达 backbone

### 1.2 关键机制 (Key Mechanism)

**残差流分解**：每一层的 hidden state 是嵌入加上所有前置模块写入的累加和：

```
h^ℓ = x + Σᵢ₌₀ᵉˡ (aⁱ + mⁱ)
```

其中 aⁱ 是 attention write，mⁱ 是 MLP write。这个分解是本文因果定位的基础——每个 write 可以被单独删除（ablation）来观察下游影响。

**深度探测协议**：对每一层 ℓ ∈ {0, ..., 35}，训练一个容量固定的 Dense Prediction Transformer (DPT) probe，从该层的视觉 token hidden states 预测深度图，监督信号来自 Depth-Anything-3 单目深度估计器。报告 d1 指标（max(d̂/d, d/d̂) < 1.25 的像素比例）。

⚡ **Eureka Moment**：动作后训练将 VLM 最后几层的 MLP 子层"重编程"为动作计算，代价是摧毁了这些层中承载深度信息的最干净通道——MLP writes。

### 1.3 信息流/架构图 (Flow / Diagram)

```
输入图像 I ──→ SigLIP2 ──→ 视觉 tokens x₁:ₖ
指令 c    ──→ E_text  ──→ 文本 tokens xₖ₊₁:ₙ
                                    │
                                    ▼
                    ┌─────────────────────────────────────┐
                    │    Transformer Backbone B_θ          │
                    │  Layer 0 ──→ Layer 1 ──→ ... ──→ L35 │
                    │    Each layer:                        │
                    │      aⁱ = Attnⁱ(LN(hⁱ⁻¹))            │
                    │      mⁱ = MLPⁱ(LN(hⁱ⁻¹ + aⁱ))        │
                    │      hⁱ = hⁱ⁻¹ + aⁱ + mⁱ             │
                    └─────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
              [Probe at L₀]   [Probe at L₁₈]  [Probe at L₃₅]
                    │               │               │
                    ▼               ▼               ▼
              DPT Head        DPT Head        DPT Head
                    │               │               │
                    ▼               ▼               ▼
              Depth Map        Depth Map        Depth Map
                    │               │               │
                    └───────────────┼───────────────┘
                                    ▼
                        Supervised by Depth-Anything-3
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：

```
h^ℓ = x + Σᵢ₌₀ᵉˡ (aⁱ + mⁱ)    →    动作训练后: mⁱ_晚期 干扰深度信号
```

**目标**：量化动作后训练对 VLM 深度可解码性的影响，并定位到具体模块和层级。

**核心方程**：

```
hⁱ₁:ₙ = hⁱ⁻¹₁:ₙ + aⁱ₁:ₙ + mⁱ₁:ₙ
```

其中：
- `hⁱ₁:ₙ`：第 i 层后的 hidden states（n 个 token，维度 dₑ=2560）
- `aⁱ₁:ₙ = Attnⁱ(LN(hⁱ⁻¹₁:ₙ))`：attention 子层的写入
- `mⁱ₁:ₙ = MLPⁱ(LN(hⁱ⁻¹₁:ₙ + aⁱ₁:ₙ))`：MLP 子层的写入
- `LN`：RMSNorm 预归一化

**评估指标 d1**：

```
d1 = fraction of pixels where max(d̂/d, d/d̂) < 1.25
```

经过仿射对齐预测到教师深度图后计算。

**SNR（线性探针信噪比）**：

```
SNR = R² / (1 - R²)
```

R² 来自 ridge 回归，衡量线性可解码的深度方差比例。

> 符号与本文保持一致：aⁱ = attention write, mⁱ = MLP write, hⁱ = residual stream at layer ℓ

## 3. 带数字走一遍：玩具例子 (Worked Example)

让我们用一个具体的数值例子来理解 floor 和 cliff 效应。

**假设场景**：同一张 LIBERO 观察帧，分别输入 Molmo2-ER 和 MolmoAct2-LIBERO。

| 层级区间 | VLM d1 | VLA d1 | 差距 (VLM - VLA) |
|---------|--------|--------|-----------------|
| L0 (初始) | 0.71 | ~0.67 | ~0.04 (floor) |
| L8 (低谷) | 0.66 | ~0.62 | ~0.04 (floor) |
| L20 (中段) | ~0.70 | ~0.66 | ~0.04 (floor 最窄处) |
| L28-35 (末段) | 0.76 | 0.51 | **0.25** |

**关键数字**：
- VLM 从 L28 到 L35：**上升 +0.040**（巩固深度表征）
- VLA 从 L28 到 L35：**下降 -0.123**（崩塌）
- 末层绝对差距：0.76 - 0.51 = **0.25 d1**

**因果定位实验**（ablating L33-35 MLP writes）：

```
VLA clean (L35):  d1 = 0.506
VLA - MLP(L33-35): d1 = 0.584  (+0.078)
```

删除最后 3 层的 MLP 写入，d1 从 0.506 恢复到 0.584，恢复了 cliff 幅度（0.123）的 **63%**。

对比控制组：
- VLM - MLP(L33-35): +0.015（几乎无变化）
- VLA - Attn(L33-35): +0.031（远小于 MLP ablation）

**直觉**：动作训练让最后几层的 MLP 子层"忘记"了如何写入深度信息，转而写入动作相关信息。删掉这些写入，深度信息反而更清晰了。

## 4. 工程视角 (Engineering View)

| 工程维度 | 发现 | 含义 |
|---------|------|------|
| 表征退化幅度 | 末层 d1 下降 0.25（33% 相对下降） | 如果下游任务依赖空间理解（如抓取定位），性能可能显著受损 |
| 恢复可行性 | 删掉 late MLP writes 恢复 63% cliff | 后处理修复可行，但需验证对闭环策略的影响 |
| 计算开销 | DPT probe per layer = 36 个独立训练 | 诊断成本高，但可作为训练时的监控信号 |
| 部署约束 | 消融干预在推理时执行 | 若部署时删除 late MLP，可能影响动作生成质量（trade-off 未评估） |
| 模块特异性 | 仅 late MLP 有显著效应，attention 无 | 修复可精准定位，不需要全局重新训练 |

**工程含义**：本文的发现暗示了一种可能的后处理修复策略——在推理时对 late-layer MLP 施加某种形式的抑制或正则化，以保留空间表征。但这需要在动作性能和空间理解之间找到平衡点，论文未评估这一 trade-off。

## 5. 数据与评测 (Data & Eval)

| 维度 | 设置 |
|------|------|
| 数据集 | LIBERO 框架帧（256px，双相机视角） |
| 监督信号 | Depth-Anything-3 单目深度估计（伪 ground truth） |
| 评估指标 | d1（像素级深度准确率），RMSE（补充） |
| 探针类型 | DPT（Dense Prediction Transformer，非线性）+ Ridge Regression（线性） |
| 探针容量 | 跨所有 36×2 层固定相同容量 |
| 数据分割 | 按 rollout 分割（同一 episode 的相近帧不跨训练/验证） |
| 仿射对齐 | 评估时对预测做仿射变换对齐教师深度 |

**局限**：
- 深度真值来自单目深度估计器而非真实深度传感器
- 仅评估深度（空间理解的一个子集），未评估其他几何原语（如表面法向、3D 位置）
- 消融实验为单种子（single-seed），未报告方差

## 6. 能力与失败模式 (Capabilities & Failure Modes)

### 能做什么

- **精确诊断**：首次将 VLA 表征退化定位到具体模块（late MLP）和层级（L28-35）
- **因果验证**：通过消融实验证明删除 late MLP writes 可恢复深度可解码性
- **机理解释**：用模块级分解（accumulated MLP deposits vs. residual stream）解释了为什么 MLP writes 是几何信息的"更干净载体"

### 不能做什么

- **不提供修复方案**：论文明确声明"does not yet provide a training-time remedy"
- **不评估闭环策略影响**：消融干预是否改善或损害实际机器人性能未知
- **不泛化到其他模型**：仅研究 Molmo2-ER / MolmoAct2-LIBERO 一对模型
- **不覆盖全部空间理解**：仅研究深度感知，未评估其他几何能力

### 6.1 隐含假设 (Hidden Assumptions)

1. **深度是空间理解的充分代理**：论文选择深度作为"spatiogeometric understanding 的原语"，但空间理解可能包含更多维度（如拓扑关系、表面曲率）
2. **DPT probe 的容量足够**：固定容量的 DPT probe 跨所有层比较，但如果某些层的信息需要更大容量的探针才能解码，可能低估了那些层的信息含量
3. **Weight-matched 配对消除了所有混淆**：虽然 Molmo2-ER 和 MolmoAct2-LIBERO 共享初始化，但三阶段训练流程中的数据和超参差异可能引入其他混淆因素
4. **单种子消融的稳定性**：消融实验为单种子，未报告统计显著性

## 7. 与相关工作对比 (Comparison)

| 工作 | 关注点 | 方法 | 关键发现 | 与本文差异 |
|------|--------|------|---------|-----------|
| Kachaev et al. (2025) | 视觉-语言表征退化 | 对齐辅助损失 | 表征退化可用辅助目标缓解 | 探测语义目标，非几何；提供修复 |
| Concurrent work [23] | 层间表征退化 | 时间一致性辅助损失 | 层间退化可用一致性约束缓解 | 探测语义目标；提供修复；无因果定位 |
| Wu et al. (2025) | MLLM adapter 表征 | 逐层探测 + attention knockout | adapter 污染中间层，后期层恢复 | 研究普通 MLLM（非 VLA）；关注语义分割 |
| **本文** | **VLA 几何表征退化** | **逐层 DPT 探测 + 模块级消融** | **late MLP writes 是深度崩塌根因** | **诊断性研究，无修复；因果定位到模块** |

**面试 Tip**：当被问到"VLA 训练如何影响预训练表征"时，回答框架是："存在两层退化——全局 floor（每层都差一点）和晚期 cliff（最后几层额外崩塌）。因果定位表明 cliff 源于 late-layer MLP 子层对几何信息的干扰，而非 attention 子层。"

## 8. 精讀建議 (Reading Guide)

**值得精讀原文的人**：
- 做 VLA 架构设计的研究者，特别是关注 backbone 知识保留问题的
- 从事 Transformer 可解释性/机械分析的研究者，对模块级因果定位方法感兴趣的
- 评估迁移学习策略的工程团队，需要理解预训练表示在下游适配后的退化模式

**建議章節路徑**：
1. 先读 §3.1（Preliminaries）理解 Transformer 残差流分解和 MolmoAct2 三阶段训练流程
2. 再看 §4.1（Depth Probing）理解 floor 和 cliff 的定量发现
3. 然后读 §4.2（Causal Localization）理解模块级消融实验设计
4. 可跳过 §3.3 中关于 SNR 的讨论，如果不关注线性探针细节

**不值得精讀的理由**：
- 如果你不做机器人学习或 VLA 研究，这篇的技术细节可能过于专业化
- 如果你只关心 VLA 的端到端任务性能而非内部表征，摘要足够

---
[← Back to Theory](./README.md)
