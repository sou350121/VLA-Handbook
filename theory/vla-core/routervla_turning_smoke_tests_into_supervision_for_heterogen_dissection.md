# RouterVLA：把烟雾测试转化为异构 VLA 选择的监督信号
(RouterVLA: Turning Smoke Tests into Supervision for Heterogeneous VLA Selection)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-06-27
>
> **论文**: RouterVLA: Turning Smoke Tests into Supervision for Heterogeneous VLA Selection
> **链接**: https://arxiv.org/abs/2606.27355
> **核心定位**: 将部署前的烟雾测试（smoke test）从"一次性成本"重构为"可复用的路由监督信号"，用 3 次 probe 将异构 VLA 策略池的 held-out 成功率从 0.4686 提升到 0.6149（+14.64pp），同时证明在标量特征下简单的成功率规则已饱和路由信号。

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | 部署前烟雾测试可转化为策略路由监督信号；3 probes/expert 下成功率规则已饱和标量路由信息 |
| 適合精讀 | 如果你在部署多 VLA 策略池、做模型路由/选择、或关心 commissioning 成本优化 |
| 可以跳過 | 如果你只关心单模型架构改进（如新的 diffusion action head），这篇距离中等 |
| 落地可行性 | 高——无需训练新模型，只需保留 commissioning 阶段的 probe 记录 |
| 主要風險 | 实验仅在 LIBERO-Plus 仿真基准上验证；标量特征无法处理 layout 变化等视觉扰动 |

💡 **X-Ray 开场**
一篇机器人团队在部署 VLA 策略前，通常会对每个候选模型跑几次烟雾测试，然后选一个平均表现最好的全局策略。RouterVLA 问了一个简单但被忽视的问题：这些测试数据能不能不只是"选一个赢家"，而是用来在运行时动态选择最合适的专家？答案是可以——仅用 3 次 probe 的成功率统计，就能把 held-out 成功率提升 14.64 个百分点。更关键的是，研究发现复杂的 learned scorer（逻辑回归、GBDT、MLP）并不比简单的成功率规则更好，因为成功次数本身已经携带了大部分路由信号。

📍 **研究全景时间线**
```
[2023] RT-1/RT-2: 单VLA策略范式
    ↓
[2024] OpenVLA: 开源基础VLA → 模型缩放轴确立
    ↓
[2024-25] RoboRouter/MoIRA/MergeVLA: 策略池路由/合并
    ↓
[2025] LIBERO-Plus: 系统暴露策略性能对扰动的依赖性
    ↓
[2026.06] RouterVLA ← 当前位置：将commissioning从成本转为监督信号
    → 局限: 仅仿真基准; 标量profile无法处理视觉layout变化
```

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

RouterVLA 不是一个新的神经网络架构，而是一个 commissioning-and-selection 框架。它的核心思想是将系统分为两个独立组件：

| 组件 | 输入 | 输出 | 训练方式 | 作用 |
|------|------|------|----------|------|
| **Commissioning Profile (Φ)** | 每个专家的 probe 记录（成功/失败、步数、时长、终止行为）+ 训练集先验 | 14 维特征向量 ϕ_e | 无需训练，确定性计算 | 编码部署前对每个专家的了解 |
| **Selector (g_θ)** | 所有候选专家的 profile 向量 | 选择得分最高的专家 | BCE 损失（learned）或固定规则（transparent） | 基于 profile 决定哪个专家执行 |

**关键设计决策**：profile 构建与 scored execution 严格分离。用于构建 profile 的 probe _trial 绝不能包含被评分 trial 的结果。

### 1.2 关键机制 (Key Mechanism)

**三步流程**：

1. **Commissioning（探测阶段）**：对目标任务-扰动组合 x，对每个可用专家 e 运行 B=3 次 probe trial，记录成功率、步数、时长、超时/早停比例等
2. **Profile 构建**：将 probe 记录汇总为 14 维向量——包含成功率统计（Beta 后验均值/方差）、轨迹摘要（步数/时长均值/方差）、训练集先验、probe 计数和缺失掩码
3. **Selection（选择阶段）**：用透明规则（如 empirical probe success）或 learned scorer 对所有候选专家打分，选最高分者执行 scored trial

⚡ **Eureka Moment**：部署前的烟雾测试不是"沉没成本"——它们是系统学习如何使用已有专家的第一个测量信号。Commissioning 携带路由价值，额外的标量 scorer 容量并不创造它。

### 1.3 信息流/架构图 (Flow / Diagram)

```
                    ┌─────────────────────────────────────┐
                    │  专家池 E = {e1, e2, ..., e28}       │
                    │  (OpenVLA, π0, RDT 等冻结checkpoint) │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │  目标条件 x (任务+扰动)               │
                    │  例如: "pick up red block, camera shifted" │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │  Commissioning: B=3 probes/expert    │
                    │  e1: [成功, 失败, 成功] → 2/3        │
                    │  e2: [成功, 成功, 成功] → 3/3        │
                    │  e3: [失败, 失败, 失败] → 0/3        │
                    │  ...                                 │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │  Profile 构建 Φ                      │
                    │  ϕ_e ∈ R^14:                        │
                    │  [成功率, Beta均值, Beta方差,         │
                    │   步数均值, 步数方差, 时长均值,       │
                    │   时长方差, 超时比例, 早停比例,       │
                    │   训练成功率, 训练延迟先验,           │
                    │   probe计数, 缺失掩码×2]             │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │  Selector g_θ                        │
                    │  透明规则: argmax_e (c_e / B_e)      │
                    │  或 learned: argmax_e g_θ(ϕ_e)       │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │  选定专家 e* 执行 scored trial       │
                    │  记录结果 Y ∈ {0,1}                  │
                    │  ← 此结果不参与 profile 构建!         │
                    └─────────────────────────────────────┘
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：

```
SR(θ) = (1/|I|) · Σ_(x,t)∈I  Y_{ê_θ(x,t), x}^(t)

其中 ê_θ(x,t) = argmax_e g_θ(Φ(P_e,x^(t), a_e))
```

**目标**：在给定目标条件 x 和 probe 预算 B 下，选择能使 held-out 成功率最大化的专家。

**变量说明**：

| 符号 | 含义 |
|------|------|
| E | 冻结专家池（28 个异构 VLA checkpoint） |
| E_x ⊆ E | 条件 x 下可用的专家子集（平均 21.8 个） |
| T_x | 条件 x 的记录 trial 集合（4 个） |
| P_e,x^(t) | 专家 e 在条件 x 上的 probe 集合（排除 trial t） |
| ϕ_e^(t)(x) | 专家 e 的 14 维 commissioning profile |
| g_θ | 标量打分函数（透明规则或 learned scorer） |
| Y_e,x^(t) | 专家 e 在 trial t 上的成功标记 ∈ {0,1} |

**直觉**：公式的核心约束是 outcome disjointness——profile 构建用的 probe 集合明确排除被评分的 trial t。这防止了"用自己的成绩预测自己"的循环论证。

**透明规则**（Empirical Probe Router）：

```
q_e^emp = c_e / B_e    （c_e = 成功 probe 数, B_e = 总 probe 数）
```

在 B=3 的等预算下，empirical、uniform-Beta、hierarchical 三种规则在所有 1,592 个评估行上选择完全相同的专家。

## 3. 带数字走一遍：玩具例子 (Worked Example)

假设一个简化的部署场景：

**条件**：任务 "pick up red block" + 相机偏移扰动
**可用专家**：3 个（OpenVLA-ft, π0-ft, RDT-ft）
**Probe 预算**：B=3 次/专家

**Step 1 — 执行 probes**：

```
OpenVLA-ft:  [成功, 成功, 失败] → c=2, B=3, success_rate=0.667
π0-ft:       [成功, 成功, 成功] → c=3, B=3, success_rate=1.000
RDT-ft:      [失败, 失败, 失败] → c=0, B=3, success_rate=0.000
```

**Step 2 — 构建 profiles**（简化为成功率特征）：

```
ϕ_OpenVLA = [0.667, ...]
ϕ_π0      = [1.000, ...]
ϕ_RDT     = [0.000, ...]
```

**Step 3 — Selection**：

```
argmax_e q_e^emp = argmax_e [0.667, 1.000, 0.000] = π0-ft
```

**Step 4 — Scored execution**（held-out trial，结果在选择时未知）：

```
π0-ft 执行 → 成功 (Y=1) ✓
```

**对比 baseline**（Global Best，不用 probe）：

```
假设 Global Best 选 OpenVLA-ft → 失败 (Y=0) ✗
```

**统计意义**：在 1,592 个评估行上，这个简单的"选 probe 成功率最高的专家"规则将整体成功率从 46.86% 提升到 61.49%。

## 4. 工程视角 (Engineering View)

| 维度 | 数值/分析 | 工程含义 |
|------|-----------|----------|
| **Probe 成本** | 平均 65.5 次 probe 执行/条件（B=3, median 候选 27） |  exhaustive probing 成本过高；需用 shortlist 策略 |
| **Shortlist 效果** | M=12,B=3 → 36 probes, SR=0.6185（几乎无损） | 先用训练先验筛选 top-12，再 probe，省 45% 成本 |
| **极简 shortlist** | M=3,B=1 → 仅 3 probes, SR=0.5597 | 资源极度受限时的最小可行方案 |
| **Selector 复杂度** | 透明规则 ≈ LR ≈ GBDT ≈ MLP | 部署时无需训练复杂模型，成功率规则即可 |
| **Outcome 分离** | 同 trial 复用使 gain 膨胀 1.87× | 评测协议必须严格区分 probe 和 scored trial |
| **Tie 频率** | 59.2% 的行在 probe success 上平局 | 需要二级 tie-break 机制（behavior traces 或 prior） |

**部署约束**：
- 每个专家必须是冻结的、架构无关的 black-box——不需要访问权重或内部表示
- Profile 构建是确定性的，无训练开销
- 核心瓶颈是证据获取（active commissioning），不是 scorer 容量

## 5. 数据与评测 (Data & Eval)

| 维度 | 细节 |
|------|------|
| **数据集** | LIBERO-Plus ledger，34,752 条有效 rollout 记录 |
| **专家池** | 28 个冻结异构 VLA checkpoint（OpenVLA、π0、RDT 等系列变体） |
| **任务-扰动变体** | 398 个（4 个 LIBERO suite × 3 个扰动轴） |
| **每个变体-专家对** | 4 条记录 trial |
| **评估行** | 1,592 个（398 variants × 4 folds） |
| **Primary split** | 3-to-1 trial-disjoint cross-fitting |
| **Generalization split** | Leave-one-suite-out（learned scorer 训练） |
| **不确定性估计** | 10,000 variant-cluster bootstrap |

**关键设计**：候选可用性是数据的一部分，不是事后选择。有效池平均大小 21.8，中位数 27，范围 1-28；仅 0.8% 的变体只有一个可用专家。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

### 能做什么
- **异构策略池路由**：不需要专家共享架构或权重格式，black-box 接口即可
- **低成本 commissioning 复用**：将已有的部署前测试转化为路由信号，无需额外数据采集
- **透明可解释决策**：简单的成功率规则即可达到最优效果，无需黑盒 scorer
- **成本敏感部署**：通过 shortlist 策略在 probe 成本和路由质量间灵活 trade-off

### 不能做什么
- **Layout 变化处理**：在布局扰动轴上路由增益为负（MLP Δ 为负），因为标量 trace 无法区分视觉排列变化
- **Zero-probe 路由**：需要至少 B=1 次 probe/expert；不能直接路由到完全未见过的条件
- **打破标量瓶颈**：在 59.2% 的平局行中，learned behavior tie-break 仅比 success+prior 高 0.0011
- **外推到仿真外场景**：所有实验在 LIBERO-Plus 仿真环境，未验证物理部署中的疲劳、漂移、环境变化

### 6.1 隐含假设 (Hidden Assumptions)

| 假设 | 说明 | 风险 |
|------|------|------|
| **Trial ID ≈ 时间顺序** | 4 个 trial ID 用于 cross-fitting，但 ID 是 ledger 索引而非物理时间戳 | 不模拟真实部署中的时间顺序和漂移 |
| **标量特征足够** | 14 维 profile 仅包含成功/失败、步数、时长等标量统计 | 无法捕获视觉上下文信息（如 layout 变化） |
| **专家池静态** | 28 个专家固定不变，无新专家加入或旧专家淘汰 | 实际部署中专家池是动态演化的 |
| **Probe 成本是沉没成本** | 假设烟雾测试无论如何都要做 | 如果 probes 是额外工作，成本模型完全不同 |
| **LIBERO-Plus 代表性** | 实验仅覆盖桌面操作任务的仿真变体 | 不保证对移动机器人、双臂、人形等场景有效 |

## 7. 与相关工作对比 (Comparison)

| 方法 | 路由依据 | 是否需要训练 | 架构要求 | 适用场景 |
|------|----------|-------------|----------|----------|
| **RouterVLA** | Commissioning probes（成功率等标量） | 可选（透明规则即可） | 架构无关，black-box | 部署前已有烟雾测试的多专家池 |
| **RoboRouter** | 语义相似的历史任务 + 累积执行记录 | 否（retrieval-based） | 需要共享任务嵌入空间 | 有丰富历史执行记录的场景 |
| **MoIRA** | 语言指令相似度或 LM 分类 | 是（LM 训练） | 需要语言-任务对齐 | 语言指令驱动的任务路由 |
| **MergeVLA** | 任务特定 adapter 选择/合并 | 是（adapter 训练） | 需要共享 VLA 架构 | 同一基础模型的多任务变体 |
| **Global Best** | 训练集全局平均成功率 | 否 | 无 | 单专家部署，无 commissioning |

**面试 Tip**：当被问到"RouterVLA 和 RoboRouter 有什么区别"时，回答："RouterVLA 利用部署前的烟雾测试（实际执行记录）作为路由信号，强调 outcome-disjoint 测量；RoboRouter 利用语义相似的历史任务作为路由依据。前者关注 commissioning 证据的复用，后者关注任务相似性的检索。"

## 8. 精讀建議 (Reading Guide)

- **值得精讀原文的人**：
  1. 部署多 VLA 策略池的工程师——本文直接给出了 commissioning 到路由的可行路径
  2. 研究模型选择/路由的学者——outcome separation 的评估原则对 ledger-based routing 研究有方法论价值
  3. 关心部署成本优化的团队——shortlist 策略（M=12,B=3）给出了明确的 cost-quality trade-off

- **建議章節路徑**：先讀 §Problem Setup（理解 outcome-disjoint 定义）→ 再看 §Results Table 2（核心数字）→ 可跳 §Related Work（除非你特别关注 algorithm selection 文献）

- **不值得精讀的理由**：如果你不做机器人部署、已熟悉策略池路由的基本概念、或只关心单模型架构改进，读摘要和 §Discussion 即可


---
[← Back to Theory](./README.md)

**关键引用**：
- 论文: https://arxiv.org/abs/2606.27355
- LIBERO-Plus: https://arxiv.org/abs/2510.13626
- OpenVLA: https://arxiv.org/abs/2406.09231
- π0: https://arxiv.org/abs/2410.24164
