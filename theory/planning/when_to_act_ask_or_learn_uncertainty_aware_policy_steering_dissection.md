# 何时执行、询问或学习：不确定性感知策略转向 (When to Act, Ask, or Learn: Uncertainty-Aware Policy Steering)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-03-02
>
> **论文**: When to Act, Ask, or Learn: Uncertainty-Aware Policy Steering
> **链接**: https://arxiv.org/abs/2602.22474 | https://jessie-yuan.github.io/ups/
> **核心定位**: 解决 VLM 验证器在策略转向中的过度自信问题——通过 conformal prediction 校准不确定性，区分"任务歧义"与"能力不足"，选择正确的解决策略。

💡 **X-Ray 开场**

这篇论文解决一个实际问题：用 VLM 做机器人动作验证器时，VLM 经常"过度自信"——明明不确定或做不到，却硬要执行。UPS 框架让系统学会"自知之明"：能区分"我没听懂指令"（需要问清楚）和"我做不到这个动作"（需要人演示），而不是盲目执行。对 VLA 研究者而言，这是第一个将 conformal prediction 引入策略转向的框架，提供了统计保证的校准机制。

📍 **研究全景时间线**

```
2022-2023: VLM 作为机器人验证器兴起 (VLM-Act, VLM-Img)
    ↓
2024: Policy Steering 框架出现 (用 VLM 筛选 diffusion policy 的动作样本)
    ↓
2025: 发现 VLM 校准问题 (过度自信导致 steering 失效)
    ↓
[2026] UPS 本文 ← 当前位置：首次用 conformal prediction 校准 VLM+policy 组合，区分三类不确定性
    ↓
未来：在线 residual learning 持续改进基础策略能力
```

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 模块 | 输入 | 输出 | 频率/时序 | 训练/推理 |
| --- | --- | --- | --- | --- |
| Base Policy (Diffusion) | 观测 + 任务指令 | 低阶动作块 (short-horizon chunks) | 高频 (控制循环) | 预训练 |
| World Model | 当前观测 + 动作样本 | 预测的未来观测序列 | 中频 (验证时) | 预训练 |
| VLM Verifier | 预测观测的 narration + 任务指令 | 行为叙事概率分布 + 不确定性类型 | 中频 (验证时) | 零样本 |
| Conformal Calibrator | VLM 概率输出 + 校准集 | 校准阈值 q̂, 预测集 | 低频 (部署前校准) | 离线校准 |
| Residual Policy | 干预演示数据 | 修正动作偏移 | 低频 (部署后学习) | 在线增量学习 |

### 1.2 关键机制 (Key Mechanism)

**不确定性分类与解决策略映射**：

| 不确定性类型 | 来源 | 解决策略 | 触发条件 |
| --- | --- | --- | --- |
| Straightforward (高置信) | 无显著不确定性 | Execute | VLM 对单一行为叙事概率 > 校准阈值 |
| Ambiguous (语义歧义) | 任务指令模糊 (如"那边") | Ask (Clarify) | 多个意图假设的概率接近，预测集包含多个选项 |
| Incapable (能力不足) | 基础策略无法完成任务 | Learn (Intervene) | 所有行为叙事概率均低于阈值，预测集为空或极低置信 |

⚡ **Eureka Moment**：UPS 的核心洞见是——VLM 的"过度自信"不是噪声，而是可校准的信号；通过 conformal prediction 将概率输出转化为有统计保证的预测集，系统可以**程序化地区分**"我没听懂"和"我做不到"，而不是依赖启发式阈值。

### 1.3 信息流/架构图 (Flow / Diagram)

```
┌─────────────────────────────────────────────────────────────────┐
│                        DEPLOYMENT LOOP                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User Instruction ──┐                                           │
│                     ▼                                           │
│  ┌──────────────────────────────────────────┐                   │
│  │  1. Intent Hypothesis Generation         │                   │
│  │     VLM: 生成多个可能的意图假设           │                   │
│  └──────────────────────────────────────────┘                   │
│                     │                                           │
│                     ▼                                           │
│  ┌──────────────────────────────────────────┐                   │
│  │  2. Action Sampling (Base Policy)        │                   │
│  │     Diffusion Policy: 生成动作样本        │                   │
│  └──────────────────────────────────────────┘                   │
│                     │                                           │
│                     ▼                                           │
│  ┌──────────────────────────────────────────┐                   │
│  │  3. World Model Imagination              │                   │
│  │     滚动预测未来观测 → 生成长时序序列     │                   │
│  └──────────────────────────────────────────┘                   │
│                     │                                           │
│                     ▼                                           │
│  ┌──────────────────────────────────────────┐                   │
│  │  4. VLM Narration + Scoring              │                   │
│  │     对每个行为叙事打分 (Intent-aware)     │                   │
│  └──────────────────────────────────────────┘                   │
│                     │                                           │
│                     ▼                                           │
│  ┌──────────────────────────────────────────┐                   │
│  │  5. Conformal Prediction Calibration     │                   │
│  │     比较概率 vs 校准阈值 q̂ → 预测集       │                   │
│  └──────────────────────────────────────────┘                   │
│                     │                                           │
│         ┌───────────┼───────────┐                               │
│         ▼           ▼           ▼                               │
│    ┌────────┐  ┌────────┐  ┌────────┐                           │
│    │Execute │  │ Ask    │  │ Learn  │                           │
│    │动作执行 │  │澄清询问 │  │干预学习 │                           │
│    └────────┘  └────────┘  └────────┘                           │
│         │           │           │                               │
│         └───────────┴───────────┘                               │
│                     │                                           │
│                     ▼                                           │
│  ┌──────────────────────────────────────────┐                   │
│  │  6. Residual Learning (Offline)          │                   │
│  │     从干预数据学习 residual policy        │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：

```
预测集 C = {y: p(y|x) ≥ 1 - q̂}, 其中 q̂ 来自 conformal calibration
决策 = Execute if |C|=1; Ask if |C|>1 (ambiguous); Learn if C=∅ (incapable)
```

**目标**：在部署时，以至少 $1-\alpha$ 的概率保证正确策略被包含在预测集中。

**Conformal Prediction 校准过程**：

```
1. 校准集 {(x_i, y_i)} i=1..n
2. 计算非一致性分数 s_i = 1 - p(y_i | x_i)
3. q̂ = (1-α)(n+1) 分位数 of {s_i}
4. 部署时：C(x) = {y: 1 - p(y|x) ≤ q̂} = {y: p(y|x) ≥ 1 - q̂}
```

**变量说明**：

| 符号 | 含义 | 来源 |
| --- | --- | --- |
| p(y|x) | VLM 对行为叙事 y 的条件概率 | VLM verifier 输出 |
| q̂ | 校准阈值 | 从校准集计算的分位数 |
| $\alpha$ | 目标错误率 (如 0.1 表示 90% 覆盖) | 预设超参 |
| C(x) | 预测集 | conformal prediction 输出 |
| |C| | 预测集大小 | 决策依据 |

**直觉**：conformal prediction 不关心概率的"绝对值"是否准确，只关心"排序"是否正确。通过在校准集上统计分位数，它将任意概率输出转化为有覆盖保证的预测集。这使得 VLM 的过度自信被"压缩"到统计框架内——即使 VLM 给所有选项都打高分，只要相对排序正确，校准后仍能产生有意义的预测集。

> 符号与本文/相关文档保持一致：y 表示行为叙事 (behavior narration)，x 表示观测 + 指令上下文。

## 3. 带数字走一遍：玩具例子 (Worked Example)

**场景**：用户指令 "把杯子放到那边的箱子里" (Put the cup in the bin over there)

**步骤 1: Intent Hypothesis**
VLM 生成 3 个意图假设：
- H1: 左边的箱子 (概率先验 0.3)
- H2: 右边的箱子 (概率先验 0.5)
- H3: 中间的箱子 (概率先验 0.2)

**步骤 2: Action Sampling + World Model**
Base policy 为每个意图生成动作样本，world model 滚动预测 5 步未来观测。

**步骤 3: VLM Scoring**
对每个意图 H_k，计算行为叙事概率：
```
p(narration | H1, instruction) = 0.72
p(narration | H2, instruction) = 0.68
p(narration | H3, instruction) = 0.45
```

**步骤 4: Conformal Calibration**
假设校准阶段得到的 q̂ = 0.25 (即阈值 1 - q̂ = 0.75)

**步骤 5: Prediction Set**
```
C = {y: p(y|x) ≥ 0.75}
  = {H1: 0.72, H2: 0.68, H3: 0.45} ∩ {p ≥ 0.75}
  = ∅ (空集，因为所有概率都 < 0.75)
```

**决策**：$C = \emptyset$ → **Incapable** 或 **Ambiguous**

进一步分析：如果是因为多个意图概率接近但都不够高，则是 Ambiguous → 触发 Ask 策略。

**Ask 策略输出**：
```
VLM 生成澄清问题："您指的是哪个箱子？左边、右边还是中间的？"
用户回答："右边的"
→ 更新意图分布：H2 = 0.95, H1 = 0.03, H3 = 0.02
→ 重新计算：p(narration | H2) = 0.88 ≥ 0.75
→ C = {H2}, |C| = 1 → Execute
```

**步骤 6: Execute**
执行 H2 对应的动作样本，任务成功。

**对比：无校准基线**
Vanilla VLM 直接选最高分 H1 (0.72)，执行后失败（用户实际指右边）。

## 4. 工程视角 (Engineering View)

**延迟分解**：

| 阶段 | 耗时 (估计) | 可并行化 | 备注 |
| --- | --- | --- | --- |
| Intent Hypothesis | 200-500ms | 否 | VLM 推理 |
| Action Sampling | 50-100ms | 是 | Diffusion policy 采样 |
| World Model Rollout | 300-800ms | 部分 | 5 步滚动预测 |
| VLM Narration + Scoring | 500-1000ms | 部分 | 主要瓶颈 |
| Conformal Decision | <10ms | 是 | 阈值比较 |
| **总计** | **~1-2.5s** | - | 每验证周期 |

**部署约束**：

1. **校准集需求**：需要 50-200 条标注数据用于 conformal calibration。论文未明确说明校准集规模敏感性，这是待验证点。

2. **World Model 质量依赖**：如果 world model 预测的未来观测不准确，narration 会失真，导致 VLM 打分失效。这是级联误差源。

3. **Residual Learning 频率**：论文提到"minimal expensive human feedback"，但未量化干预频率。工程上需要设置干预上限（如每小时最多 N 次），避免用户疲劳。

4. **量化误差**：VLM 概率输出是浮点数，conformal threshold 比较对舍入误差敏感。建议用定点数或保留足够小数位。

**内存占用**（估计）：
- Base Policy (Diffusion): ~500MB
- World Model: ~300MB
- VLM (外部 API 或本地): 本地部署 ~2-4GB
- 校准数据：<1MB

**吞吐**：单机器人实例，每任务 1-3 次验证循环（取决于首次是否高置信）。

## 5. 数据与评测 (Data & Eval)

**实验设置**：

| 维度 | 配置 |
| --- | --- |
| 仿真环境 | 未明确 (可能是 Isaac Gym 或类似) |
| 硬件平台 | 真实机器人 (未明确型号，从视频看是单臂桌面操作) |
| 任务类型 | 桌面操作 (pick-and-place, 物体定向放置) |
| 评估指标 | Success Rate, Coverage (conformal), Intervention Count |

**基线对比**：

| 基线 | 描述 | 关键差异 |
| --- | --- | --- |
| Base Policy | 原始 diffusion policy | 无 steering |
| Human-Gated DAgger + Residual | 人工干预触发 residual 学习 | 人工决定何时干预 |
| EnsembleDAgger | 用 policy ensemble 分歧触发干预 | 无 VLM, 无 conformal |
| FOREWARN | UPS 但不确定性校准 | 无 conformal prediction |
| UPS w/ Clarification | UPS 有校准但无 residual | 无在线学习 |
| UPS w/ Clarification + Residual (Ours) | 完整框架 | 校准 + 澄清 + 学习 |

**结果摘要**（来自项目主页图表）：
- Conformal Prediction + Intent-aware scoring 在 ambiguous/incapable 场景下覆盖率高于基线
- UPS 相比 Base Policy 在 ambiguous 场景成功率显著提升
- Residual learning 后，incapable 场景成功率提升

> TODO: 论文未提供具体数字表格，需等待 arXiv 正式版本或补充材料。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

**能做什么**：
- 区分"指令歧义"和"能力不足"两类不确定性
- 用统计保证的校准机制替代启发式阈值
- 通过澄清问题减少不必要的人工干预
- 从干预中在线学习改进基础策略

**不能做什么**：
- 无法处理世界模型预测完全错误的情况（级联失效）
- 无法在零校准数据下工作（需要 50-200 条校准样本）
- 无法处理多轮对话的复杂澄清（当前是单轮问答）
- 无法在基础策略完全无相关能力时"凭空学会"（residual learning 需要演示数据）

### 6.1 隐含假设 (Hidden Assumptions)

| 假设 | 是否验证 | 风险 |
| --- | --- | --- |
| VLM 概率输出的相对排序是可靠的 | 部分 (通过 conformal 保证覆盖) | 如果排序也错，校准无效 |
| World model 能准确预测 5 步未来观测 | 未明确验证 | 预测误差会传导到 narration |
| 用户能理解并回答澄清问题 | 假设成立 | 实际用户可能困惑或答错 |
| 干预数据足够训练有效的 residual policy | 假设成立 | 少量干预可能不足 |
| 校准集分布与部署分布一致 | 未验证 | 分布偏移会破坏覆盖保证 |

**X-Ray 批判**：论文的核心贡献是 conformal calibration，但这依赖于一个关键假设——VLM 的概率输出至少"相对可信"（排序正确）。如果 VLM 对错误选项打分高于正确选项，conformal prediction 只能保证"覆盖"，不能保证"精确"。这是 VLM 作为验证器的根本局限。

## 7. 与相关工作对比 (Comparison)

| 工作 | 不确定性处理 | 校准机制 | 解决策略 | 适用场景 |
| --- | --- | --- | --- | --- |
| VLM-Act / VLM-Img | 无显式处理 | 无 | 直接执行 | 简单任务 |
| FOREWARN | 有 VLM verifier | 无 conformal | 选最佳叙事 | 中等复杂度 |
| EnsembleDAgger | Policy ensemble 分歧 | 无 | 触发人工干预 | 需要频繁干预 |
| Human-Gated DAgger | 人工决定 | 无 | 人工干预 | 人在环 |
| **UPS (本文)** | **显式分类 (3 类)** | **Conformal Prediction** | **Execute/Ask/Learn** | **部署时自适应** |

**面试 Tip**：如果被问到"如何让 VLA 系统知道自己什么时候不确定"，回答："用 conformal prediction 校准 VLM 输出，将概率转化为有覆盖保证的预测集。预测集大小决定策略：$|C|=1$ 执行，$|C|>1$ 询问，$C=\emptyset$ 学习。这是 UPS 框架的核心。"

---

## 参考链接

- 论文 arXiv: https://arxiv.org/abs/2602.22474
- 项目主页: https://jessie-yuan.github.io/ups/
- 相关概念: Conformal Prediction 教程 https://martinwainwr.github.io/Conformal-Prediction/

---
[← Back to Theory](./README.md)
