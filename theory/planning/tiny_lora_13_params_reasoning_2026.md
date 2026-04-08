# 13 参数推理微调：TinyLoRA (Learning to Reason in 13 Parameters)

> **发布时间**：2026（arXiv）  
> **论文题目**：Learning to Reason in 13 Parameters  
> **核心定位**：用极少参数激活推理能力，RL 信号在低参数预算下显著优于 SFT。

本工作强调“能力激活”而非“知识注入”：在极低参数预算下，RL 的结果监督更能触发推理行为模式。

**一手来源**：
- 论文 PDF：`https://arxiv.org/pdf/2602.04118`

---

## 图示占位（图片待补）

- **图1**：TinyLoRA 权重二维码（含极少参数）  
- **图2**：RL vs SFT 训练曲线（低参数区）  
- **图3**：Scaling Law 趋势（模型越大参数越少）

---

## 0. 先把“可复述结论”写清楚（1 分钟版）
- **一句话**：TinyLoRA 把 LoRA 参数量压到极限，靠 RL 信号激活推理能力。  
- **关键机制**：矩阵更新 -> 向量更新 + 全层共享 + 随机映射（论文口径）。  
- **结果要点**：低参数预算下 RL 明显优于 SFT（数值以论文为准）。  
- **工程价值**：极小补丁可显著提升推理表现，适合边缘部署与快速适配。  
- **潜在风险**：对任务分布/奖励信号依赖强，泛化需验证。

---

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 方案 | 参数形式 | 共享策略 | 优点 | 典型问题 |
|---|---|---|---|---|
| LoRA | 低秩矩阵 A,B | 层内 | 稳定、通用 | 参数仍偏多 |
| LoRA-XS | 低秩 + 固定基 | 层内 | 参数更少 | 表达仍受限 |
| **TinyLoRA** | 向量 v | **跨层全共享** | 极低参数 | 对奖励/任务敏感 |

### 1.2 关键机制 (Key Mechanism)
1) **矩阵 -> 向量**：把可训练矩阵替换为极低维向量（论文口径）。  
2) **跨层全共享**：Attention 与 MLP 模块共享同一向量。  
3) **RL 优于 SFT**：低参数预算下，RL 的结果信号更有效（论文口径）。  
4) **高精度更重要**：极低参数时 FP32 优于 bf16（论文口径）。

### 1.3 信息流/架构图 (Flow / Diagram)

```
Base LLM (7B)
   |
   |  Frozen weights
   v
TinyLoRA Adapter (shared vector v)
   |
   +--> Random projection -> per-layer delta
   |
   v
RL fine-tune (GRPO)  /  SFT baseline
   |
   v
Reasoning performance (GSM8K / AIME / MATH-500)
```

---

## 2. 数学核心：从 LoRA 到 TinyLoRA (Math Core)

**LoRA 基本形式**：

$$
W' = W + A B,\quad A \in \mathbb{R}^{d\times r},\ B \in \mathbb{R}^{r\times k}
$$

**LoRA-XS（示意）**：固定基 + 训练缩放向量（论文口径）。

**TinyLoRA（示意）**：用共享向量替代矩阵更新。

$$
\Delta W_{\ell} = P_{\ell}\,\mathrm{diag}(v)\,Q_{\ell}
$$

- $v$：全模型共享的极低维向量  
- $P_{\ell}, Q_{\ell}$：固定随机/预冻结映射（论文口径）  
- $\ell$：层索引（Attention/MLP 共享同一 $v$）

**直觉**：把“微调能力”压缩为少量全局旋钮，通过随机映射对全模型产生微小但一致的偏置。

---

## 3. 带数字走一遍：玩具例子 (Worked Example)

**场景**：2 层模型、共享向量 $v \in \mathbb{R}^2$。

```
v = [0.2, -0.1] (共享)
层1: ΔW1 = P1 diag(v) Q1
层2: ΔW2 = P2 diag(v) Q2

结果：同一向量在不同层产生不同方向的小幅偏置
```

**直觉**：不是“记住答案”，而是“调节推理风格”。

---

## 4. 工程视角：为什么 RL > SFT (Engineering View)

### 4.1 SFT 的问题
- SFT 要拟合每个 token，信号噪声大。  
- 极低参数时，容量不足以“背诵格式”。

### 4.2 RL 的优势
- RL 只关心结果（对/错），信号更干净。  
- 更像“激活模式”，而非“写入知识”。

### 4.3 精度优先
在极低参数预算下，FP32 往往优于 bf16（论文口径），说明微小扰动需要更高分辨率。

---

## 5. 数据与评测 (Data & Eval)

**任务**：GSM8K / MATH-500 / AIME / Olympiad Bench（论文口径）。  
**结果**：RL 在低参数预算下显著优于 SFT（数值以论文为准）。

> 用户口径提到“13 参数 / 26 字节 / 91%”，建议以论文原文核验。

---

## 6. 能力与失败模式 (Capabilities & Failure Modes)

**能力**：
- 低参数也能显著提升数学推理表现  
- 模型越大所需参数越少（论文口径）

**失败模式**：
- 任务分布偏移时效果下降  
- 奖励设计不当会导致“伪推理”

---

## 7. 与相关方法对比 (Comparison)

| 方法 | 训练信号 | 参数规模 | 低参表现 |
|---|---|---|---|
| SFT | token 级监督 | 中 | 低参易失效 |
| LoRA / LoRA-XS | token 级监督 | 中/小 | 稳定但参数仍多 |
| **TinyLoRA + RL** | **结果级奖励** | **极低** | 低参表现最佳 |

**面试 Tip**：  
“TinyLoRA 的关键不是‘更少参数’，而是用 RL 的结果信号去激活推理模式。”

---

## 参考链接
- 论文 PDF：`https://arxiv.org/pdf/2602.04118`

---
[← Back to Theory](../README.md)
