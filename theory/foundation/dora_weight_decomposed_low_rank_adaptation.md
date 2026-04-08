# DoRA：权重分解的低秩适配 (DoRA: Weight-Decomposed Low-Rank Adaptation)

> **发布时间**：2024（ICML 2024 Oral；arXiv v1: 2024-02-14，v6: 2024-07-09）  
> **论文题目**：DoRA: Weight-Decomposed Low-Rank Adaptation  
> **团队**：NVIDIA / HKUST  
> **核心定位**：把线性层权重按列分解为 **幅值（magnitude）** 与 **方向（direction）**，让低秩更新主要负责“方向”，同时显式学习“幅值”，从而让 PEFT 的学习模式更接近全参微调（FT），提升精度与训练稳定性，且**不增加推理开销**（可 merge）。

LoRA 的常见痛点不是“省”——而是“有时离 FT 还差一截”。DoRA 的贡献是给出一个可解释的分析视角（幅值/方向更新模式），并提供一个几乎不改变推理形态的改法。

## 0. 1 分钟版

- **LoRA** 学的是 \(\Delta W\) 的低秩近似 \(BA\)，等价于同时在学“幅值变化 + 方向变化”。  
- 论文的 **weight decomposition analysis** 发现：LoRA 的幅值/方向更新往往呈**强正相关**；而 FT 更像**负相关**（更“精细”地只改该改的那一部分）。  
- **DoRA** 把权重按列拆成 \(m\)（幅值）与 \(V\)（方向），让 **LoRA 只负责方向增量**，同时单独学习 \(m\)。  
- **训练更稳**：分解带来的梯度缩放 + 投影（类似 weight normalization 的优化性质）会改善方向更新的条件数。  
- **部署不变**：权重可提前 merge 成普通线性层，不引入额外推理延迟。  

来源：论文与代码（[arXiv](https://arxiv.org/abs/2402.09353)，[PDF](https://arxiv.org/pdf/2402.09353.pdf)，[NVlabs/DoRA](https://github.com/NVlabs/DoRA)）。

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 维度 | Full Fine-tuning (FT) | LoRA | DoRA |
|---|---|---|---|
| 训练参数 | 全量 \(W\) | \(A,B\)（低秩） | \(A,B\)（低秩）+ 每列幅值 \(m\) |
| 更新对象 | 直接更新 \(W\) | 更新 \(\Delta W=BA\) | 更新“方向增量”\(+\) 幅值向量 |
| 学习难点 | 成本高 | 幅值/方向耦合、学习模式偏离 FT | 幅值/方向解耦，更像 FT |
| 推理开销 | 无额外 | 可 merge，无额外 | 可 merge，无额外 |
| 典型收益 | 上限高 | 省算力/显存 | 更接近 FT 的精度与稳定性（论文口径） |

### 1.2 关键机制 (Key Mechanism)

1. **列向量分解**：把 \(W\in\mathbb{R}^{d\times k}\) 的每一列视作一个向量，分为“长度（幅值）+ 单位方向”。  
2. **DoRA 参数化**：让低秩更新主要作用在方向分量上；幅值由一个可训练向量单独更新。  
3. **学习模式对齐**：论文用幅值变化 \(\Delta M\) 与方向变化 \(\Delta D\) 的相关性展示 DoRA 更接近 FT（Fig.2）。  
4. **推理不变**：训练后将 DoRA 权重 merge 回线性层（与 LoRA 同范式）。  

### 1.3 信息流/架构图 (Flow / Diagram)

```text
Pretrained W0
  -> Decompose (column-wise): magnitude m0, direction V0
Train:
  - magnitude vector m (trainable, size 1×k)
  - direction update ΔV via low-rank BA
Merge:
  W' = m * (W0 + BA) / ||W0 + BA||_c
Infer:
  use merged W' as normal Linear weight
```

## 2. 数学核心：把“幅值/方向”拆开学 (Math Core)

### 2.1 权重分解（论文 Eq.(2)）

对权重矩阵 \(W\in\mathbb{R}^{d\times k}\)，按列分解为：

\[
W = m\frac{V}{\|V\|_c}
\]

其中：
- \(m\in\mathbb{R}^{1\times k}\)：幅值向量（每列的范数）
- \(V\in\mathbb{R}^{d\times k}\)：方向矩阵（每列归一化成单位向量）
- \(\|\cdot\|_c\)：按列求范数（vector-wise norm across columns）

初始化：用预训练权重 \(W_0\) 给出 \(m=\|W_0\|_c\)，\(V=W_0\)（论文说明这能避免 weight norm 从零初始化带来的敏感性）。

### 2.2 DoRA 的参数化（论文 Eq.(5)）

DoRA 让方向部分用低秩更新：

\[
W' = m \frac{W_0 + BA}{\|W_0 + BA\|_c}
\]

- \(A\in\mathbb{R}^{r\times k}\), \(B\in\mathbb{R}^{d\times r}\)，\(r\ll \min(d,k)\)
- 训练参数：\(m,A,B\)
- 推理：\(W'\) 可与 \(W_0\) 一样作为普通线性层权重使用（可 merge）

直觉：LoRA 不再承担“幅值 + 方向”两种变化的耦合学习，优化目标被拆成两个更“可控”的旋钮。

### 2.3 梯度直觉：为什么更稳（论文 Eq.(6)(7)）

论文给出：
- 方向梯度会被 **缩放**（与 \(m/\|V'\|_c\) 相关）并通过一个 **投影算子**（把梯度投影到与当前方向正交的子空间），从优化角度改善梯度协方差的条件（与 Weight Normalization 的直觉相似）。  
- 这类性质会“传递”给 \(\Delta V\)（因为 \(V'=V+\Delta V\)，对 \(\Delta V\) 的优化等价于对 \(V'\) 的优化）。

工程含义：在同样 rank 下，DoRA 往往更不那么“挑学习率”，训练更稳定（论文口径）。

## 3. 带数字走一遍：2D 玩具例子 (Worked Example)

设一列预训练权重向量：

- \(w_0 = [3,4]^T\)，\(\|w_0\|=5\)  
- 方向 \(u_0 = w_0/\|w_0\| = [0.6, 0.8]^T\)

现在下游任务需要“方向稍微转一点，但幅值基本不变”：

- **LoRA** 要通过 \(\Delta w\) 一次性实现：既要把方向转开，又要不改变长度，训练时容易出现“幅值和方向一起飘”。  
- **DoRA** 可以更直接地表达：  
  - 幅值 \(m\) 继续接近 5  
  - 方向用 \(BA\) 表示一个小增量，让 \(u\) 发生小角度变化  

这类“只改方向/只改幅值”的细粒度自由度，是论文用 \((\Delta D,\Delta M)\) 相关性解释 DoRA 更像 FT 的核心点之一（Fig.2 + Sec.4）。

## 4. 工程视角：如何在你的训练栈里用 DoRA (Engineering View)

### 4.1 你什么时候优先考虑 DoRA？

- 你已经在用 LoRA，但和 FT 有明显精度差距，且希望 **保持推理无额外开销**。  
- 训练不稳定（loss 抖/发散），你不想靠“暴力调参”解决。  
- 你在做 LLM/LVLM 的指令微调或多任务微调：论文展示 DoRA 在 LLaMA/LLaVA/VL-BART 等上优于 LoRA（见实验表格）。  

### 4.2 训练成本与实现注意点（论文 Sec.4.3）

DoRA 的一个现实问题是：因为归一化项 \(\|W_0+BA\|_c\) 参与计算图，反传会额外占显存。论文给了一个实用技巧：

- **detach 归一化项的梯度**：把 \(\|V'\|_c\) 当作常量参与前向，但不参与反向，可显著降低显存，精度几乎不变（论文表格与消融结论）。

### 4.3 与 QLoRA 的关系

DoRA 是“更新形式”的改造，QLoRA 是“底座权重量化 + 低秩更新”的资源工程。两者不是互斥概念；实际能否组合取决于你的训练框架是否支持（需要以实现为准）。

## 5. 数据与评测 (Data & Eval)

论文覆盖三类代表性设置（这里只列结论口径与方向，具体分数见论文表格）：

- **Commonsense Reasoning（LLaMA/LLaMA2/LLaMA3）**：DoRA 相比 LoRA 有稳定提升；在部分设置中 DoRA 用更低 rank 也可超过 LoRA（Table 1）。  
- **Image/Video-Text Understanding（VL-BART）**：多任务评测中 DoRA 优于 LoRA，接近 FT（Table 2/3）。  
- **Visual Instruction Tuning（LLaVA-1.5-7B）**：DoRA 优于 LoRA（论文实验结论）。  

来源：论文实验部分（[PDF](https://arxiv.org/pdf/2402.09353.pdf)）。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

- **优势更明显的情况**：预训练权重本身“方向大体对”，下游只需在少量方向/幅值上做精细改动时，DoRA 的解耦更占优。  
- **可能不明显的情况**：任务分布与预训练差距巨大，需要大幅重塑表示时，DoRA 的优势可能被“rank 不够/目标层选择不当”掩盖；这时可能需要更高 rank、更多目标模块，甚至回到 FT。  

## 7. 与相关工作对比 (Comparison)

| 方法 | 核心思路 | 推理开销 | 备注 |
|---|---|---|---|
| LoRA | \(\Delta W=BA\) | 无（可 merge） | 最常用 PEFT 基线 |
| LoRA 变体（如 VeRA 等） | 更省参数/共享或重参数化 | 无（可 merge） | 目标常是进一步压参 |
| DoRA | 幅值/方向分解；LoRA 只更新方向 | 无（可 merge） | 目标是缩小与 FT 的精度/稳定性差距 |

**面试 Tip**：一句话回答“DoRA 相对 LoRA 的改进”——**把权重更新拆成幅值与方向两部分：幅值用显式向量学，方向用低秩学，从而让学习模式更像全参微调，但仍可 merge 保持零推理开销。**

## References

- arXiv 摘要页：[https://arxiv.org/abs/2402.09353](https://arxiv.org/abs/2402.09353)  
- 论文 PDF（含公式与梯度推导）：[https://arxiv.org/pdf/2402.09353.pdf](https://arxiv.org/pdf/2402.09353.pdf)  
- 官方代码：`https://github.com/NVlabs/DoRA`（见 arXiv 页面链接）  

---
[← Back to Theory](./README.md)

