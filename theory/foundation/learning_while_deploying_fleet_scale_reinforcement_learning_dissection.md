# 部署中学习：面向通用机器人策略的车队级强化学习 (Learning While Deploying: Fleet-Scale Reinforcement Learning for Generalist Robot Policies)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-08-26
>
> **论文**: Learning While Deploying: Fleet-Scale Reinforcement Learning for Generalist Robot Policies
> **链接**: https://arxiv.org/abs/2605.00416
> **核心定位**: 解决通用 VLA 策略离线预训练后无法适应真实世界分布偏移的痛点，通过 16 台双臂机器人车队实现 offline-to-online RL 持续自我改进，平均成功率从离线基线提升至 0.95。

## ⚡ 快速判斷

| 維度 | 判斷 |
|------|------|
| 核心結論 | 车队级 offline-to-online RL 可在数小时内将预训练通用 VLA 策略的平均成功率提升至 0.95，长程任务增益最大 |
| 適合精讀 | 如果你在构建部署后持续学习系统、研究 offline-to-online RL 或需要稳定更新 flow-based VLA 策略 |
| 可以跳過 | 如果你只关心纯离线 RL 或单任务 specialist 策略 |
| 落地可行性 | 中（需 16 台机器人车队 + 集中式 learner 基础设施；算法组件可独立复用） |
| 主要風險 | 实验仅在 Agibot G1 双臂平台上验证，泛化到异构机器人平台尚待证明 |

💡 **X-Ray 开场**
离线预训练的 VLA 模型在真实部署中会遇到分布偏移、长尾失败和任务变化——固定数据集无法覆盖。本文提出「部署中学习」(LWD) 框架：让 16 台机器人车队自主运行、收集数据、用 RL 更新策略、再部署，形成数据飞轮。核心算法创新是分布式的隐式价值学习 (DIVL) + 伴随匹配策略提取 (QAM)，在 8 个真实操作任务上达到 95% 平均成功率。对 VLA 研究者意味着：部署不再是训练终点，而是持续改进的数据源。

📍 **研究全景时间线**
```
[2023] VLA 预训练范式兴起 (RT-2, OpenVLA)
    → [2024] 离线 RL 后训练 (π0.6*/RECAP, RLDG)
    → [2024] 在线 RL 微调 specialist 策略 (VLA-RL, RIPT)
    → [2025] 分布式车队执行基础设施 (SOP)
    → [2026-05] **本文 LWD** ← 当前位置：首个 fleet-scale offline-to-online RL 通用 VLA 系统
    → [未来] 异构平台泛化？
```

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 组件 | 职责 | 输入 | 输出 | 训练阶段 | 部署位置 |
|------|------|------|------|----------|----------|
| **Policy (πθ)** — Flow-based VLA | 状态 → 动作块 | 多模态观测 + 语言指令 | 动作块 a_t:t+H (H 步) | 离线全微调 → 在线仅 action expert | 边缘机器人 (异步拉取) |
| **Critic (Qφ)** — Double-Q | 状态-动作价值估计 | 状态表征 z_t + 动作块 a_t | 标量 Q 值 | 离线 + 在线全微调 | 集中式 learner |
| **Distributional Value (Vψ)** — Categorical | 状态条件价值分布 | 状态表征 z_t | 离散类别概率分布 | 离线 + 在线全微调 | 集中式 learner |
| **Reference Flow (fβ)** — 固定 | QAM 策略提取的参考轨迹 | 高斯噪声 + 状态 | 参考流轨迹 | 离线 BC 初始化后冻结 | 集中式 learner |
| **Online Buffer (B_on)** | 车队异步收集的经验 | 自主/干预 rollout 片段 | chunked transitions | 持续写入 | 集中式 learner |

### 1.2 关键机制 (Key Mechanism)

LWD 由两个核心算法组件构成：

**DIVL (Distributional Implicit Value Learning)** — 价值学习
- IQL 用 scalar expectile 回归做隐式价值学习，但车队数据是多模态/重尾的
- DIVL 改用 **categorical 分布模型** 拟合 Q 值的条件分布，用分位数做 TD bootstrap
- 关键创新：**自适应 τ** — 用分布熵调节乐观程度：高不确定性 → 低 τ (保守)；低不确定性 → 高 τ (乐观)

**QAM (Q-learning with Adjoint Matching)** — 策略提取
- Flow policy 多步生成过程直接反传 critic gradient 不稳定且昂贵
- QAM 将 critic gradient 在去噪终点转换为伴随状态，沿参考流轨迹做局部回归
- 保持 fβ (行为克隆初始化) 固定，优化 fθ 向 KL-正则化改进目标靠拢

⚡ **Eureka Moment**：用分布模型替代标量价值估计 + 用伴随匹配替代直接反传——两个看似独立的选择，共同解决了「车队级 heterogeneous 数据上的稳定 RL 更新」这一核心难题。

### 1.3 信息流/架构图 (Flow / Diagram)

```
┌─────────────────────────────────────────────────────────────────┐
│                    CENTRALIZED LEARNER                          │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                   │
│  │   DIVL   │    │   QAM    │    │  TD      │                   │
│  │ Value    │───▶│ Policy   │    │ Target   │                   │
│  │ Learner  │    │ Extractor│    │ Builder  │                   │
│  │ (Vψ,Qφ)  │    │ (fθ)     │    │          │                   │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘                   │
│       │               │               │                          │
│       ▼               ▼               ▼                          │
│  ┌──────────────────────────────────────┐                        │
│  │         Mixed Replay Buffer          │                        │
│  │    B_off  ∪  B_on (fleet-collected)  │                        │
│  └──────────────────────────────────────┘                        │
└───────────────────┬─────────────────────────────┬────────────────┘
                    │ 每 N_sync 步同步             │ 异步上传 rollout
                    ▼                             ▼
        ┌──────────────────────────┐   ┌──────────────────────────┐
        │  16× Agibot G1 Robots    │   │   Human Operator         │
        │  ┌────┐┌────┐┌────┐     │   │  (intervention when      │
        │  │ R1 ││ R2 ││... │     │   │   needed)                │
        │  └────┘└────┘└────┘     │   └──────────────────────────┘
        │  4× grocery + 3×/long   │
        │  30Hz joint-position     │
        └──────────────────────────┘
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：
```
L_total = L_Q(φ) + L_V(ψ) + L_QAM(θ)
  where TD target y_Q = r_t + γ^H · Quant_τ(V_ψ(s_{t+H}))
        and policy gradient g̃_1 = -∇_a[Q_φ(s, a^1)/λ]
```

### 目标
在车队级 heterogeneous offline+online 数据上，稳定地学习一个通用 VLA 策略，使其从稀疏二元奖励中持续改进。

### 核心方程

**DIVL 价值分布拟合**（论文 Eq.12）：
```
L_V(ψ) = E_{(s_t, a_t)~D}[-log p_ψ(Q_φ̄(s_t, a_t) | s_t)]
```
- p_ψ: categorical 分布，拟合 EMA critic 输出的 Q 值在给定状态下的条件分布
- 直觉：不拟合一个标量值，而是拟合一个「可能的 Q 值分布」，保留多模态信息

**自适应 τ 分位数 bootstrap**（论文 Eq.14 + Eq.18）：
```
y_Q = r_t + γ^H · Quant_τ(V_ψ(s_{t+H}))
τ(s) = clip(τ_base - α·H(s), τ_min, τ_max)
```
- H(s): 分布 p_ψ 的归一化熵，衡量不确定性
- 直觉：不确定时降低乐观度（保守），确定时提高乐观度（进取）

**QAM 策略提取**（论文 Eq.9 + Eq.10）：
```
L_QAM(θ) = E[∫_0^1 ||2·f_δ(s, a^w, w)/σ_w + σ_w·g̃_w||^2 dw]
g̃_1 = -∇_a[Q_φ(s, a^1)/λ]
```
- f_δ = f_θ - f_β: 优化流与参考流的差值
- g̃_w: 伴随状态，从 critic gradient 传播
- 直觉：不直接反传 critic，而是把 gradient 信息编码为伴随微分方程，沿参考流做回归

### 变量说明

| 符号 | 含义 |
|------|------|
| s_t | 状态 (观测 o + 语言指令 ℓ_k) |
| a_t | 动作块 a_t:t+H (H 步联合位置指令) |
| r_t | chunk 级奖励 Σ γ^i r_{t+i} |
| Q_φ | critic 网络 (clipped double-Q) |
| V_ψ | 分布价值模型 (categorical) |
| τ | 分位数水平 (自适应) |
| f_θ / f_β | 优化流 / 参考流 (vector field) |
| λ | QAM 温度参数 |
| γ | 折扣因子 |

> 符号与本文保持一致。DIVL 和 QAM 的原始论文使用相同符号约定。

## 3. 带数字走一遍：玩具例子 (Worked Example)

假设一个简化的长程任务场景：

**场景**: 机器人执行「泡茶」子任务——将茶叶放入茶壶（3 步子动作：抓取→移动→放置）

**数据**: replay buffer 中该状态 s 有 100 个历史动作，其中 70 个成功 (Q≈0.9)，30 个失败 (Q≈0.1)

**DIVL 价值分布拟合**:
```
p_ψ(v|s) 的 categorical 支持点: [0.0, 0.1, ..., 0.9, 1.0] (101 个 bin)
拟合结果: p_ψ(0.1|s) ≈ 0.30, p_ψ(0.9|s) ≈ 0.70
```
双峰分布！标量 IQL 会拟合 expectile ≈ 0.75（加权平均），丢失了「成功/失败双模态」信息。DIVL 保留了这个结构。

**自适应 τ**:
```
H(s) = -1/log(101) · Σ p_c · log(p_c) ≈ 0.56 (中等不确定性)
τ(s) = clip(0.8 - 0.3·0.56, 0.5, 0.95) = clip(0.632, 0.5, 0.95) = 0.632
Quant_0.632(V_ψ(s_next)) ≈ 0.7 (第 63 百分位)
```
如果分布更分散（H=0.9），τ 降至 0.53 → 更保守的 bootstrap。如果分布集中（H=0.1），τ 升至 0.77 → 更乐观。

**TD Target**:
```
y_Q = r_t + γ^H · Quant = 0 + 0.95^3 · 0.7 ≈ 0.58
```

**QAM 策略提取**:
```
∇_a Q_φ(s, a^1) 在成功方向有正 gradient
g̃_1 = -∇_a Q / λ → 指向高价值方向
L_QAM 将 f_θ 推向沿参考流的高价值区域
```

**闭环**: 经过 N 次迭代，策略逐渐学会选择那 70% 成功路径上的动作，而非平均化到 0.75 价值的中间地带。

## 4. 工程视角 (Engineering View)

| 工程维度 | 数值/约束 | 含义 |
|----------|-----------|------|
| 车队规模 | 16 台 Agibot G1 双臂机器人 | 需要并行基础设施，单机无法复现数据飞轮效应 |
| 控制频率 | 30 Hz 关节位置控制 | 动作块 H 步 = H/30 秒执行窗口 |
| 在线数据量 | ~60 小时总数据 / 4 小时 wall-clock | 16 台并行 = 每台约 3.75 小时 |
| 策略同步周期 | N_sync = 50 training steps | 每 50 步广播新策略到车队，平衡新鲜度与稳定性 |
| 在线更新范围 | 仅 action expert (Gemma-300M) 微调 | VLM backbone (Gemma-2B + SigLIP) 冻结 → 在线更新高效 |
| 价值网络 | Gemma-3-270M-IT + SigLIP-So400M | 独立于 policy 的 VLM，持续全微调 |
| 离线 n-step TD | n=10 for long-horizon, n=1 for short | 离线阶段加速稀疏奖励传播 |
| 在线 TD | 始终 1-step chunk-level | 避免跨策略/干预边界的 TD 路径污染 |

**工程含义**:
- 离线/在线目标统一是关键的工程决策——避免了 offline critic 过度保守导致的 offline-to-online mismatch
- 在线仅更新 action expert 而非全 VLA backbone，使在线更新计算成本降低约 80%（270M+400M vs 2B+400M）
- 价值网络和策略网络分离部署（集中 vs 边缘），减少了边缘推理的内存占用

## 5. 数据与评测 (Data & Eval)

### 数据集组成（论文 Table IV + §IV-C）

| 数据源 | 类型 | 内容 | 奖励标注 |
|--------|------|------|----------|
| Demonstrations | 离线 | 专家收集的成功轨迹 | 终端 r=1 |
| Rollouts | 离线 | 历史策略生成的成功+失败轨迹 | 终端 r=1/0 |
| Play Data | 离线 | 人工引导的失败模式探索 | 终端 r=1/0 |
| Autonomous Rollouts | 在线 | 当前策略自主执行 | 终端 r=1/0 |
| Human Interventions | 在线 | 操作员介入纠正的片段 | 终端 r=1/0 |

### 评测任务（论文 §V-A1 + Fig.3）

**4 个 Grocery Restocking 任务**（短程，语义泛化）:
- Flat-shelf restocking: 平面货架补货
- Misplaced-item correction: 错位物品纠正
- Freezer restocking: 冰箱门操作补货
- Open-cooler restocking: 开放式冷柜纸箱补货

**4 个 Long-Horizon 任务**（3-5 分钟，精密操作）:
- Brew Gongfu Tea: 功夫茶冲泡（6 子任务）
- Make Fruit Juice: 果汁制作（切割→榨汁）
- Make Cocktail: 鸡尾酒调制（量取→摇匀→装饰）
- Pack Shoes: 鞋子装箱

### 评测指标
- Grocery: 二元成功率（遵循指令 + 按时完成）
- Long-Horizon: 子步骤平均分（1=完全自主成功, 0.5=有小瑕疵/单次重试, 0=多次尝试后失败）
- 额外: cycle time（成功+失败平均执行时间）

### 主要结果（论文 Table I + §V-B）

| 方法 | 平均成功率 | 长程平均 | Grocery 平均 |
|------|-----------|----------|-------------|
| SFT (reference) | ~0.78 | 0.68 | ~0.88 |
| RECAP | ~0.83 | 0.77 | ~0.89 |
| HG-DAgger | ~0.82 | 0.73 | ~0.91 |
| LWD (Offline) | ~0.85 | 0.79 | ~0.91 |
| **LWD (Online)** | **0.95** | **0.91** | **~0.99** |

> 数据来源：论文 Table I。LWD (Online) 在所有 4 个长程任务上均达到最高分。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

### 能做什么
- **长程任务显著增益**: RL 通过多步动态规划传播奖励，将碎片化的部分进展拼接为完整策略——这正是 ICL 方法（误差累积）的短板
- **分布偏移自适应**: 车队数据覆盖新物体实例、新布局、新用户指令，策略持续改进
- **人类干预融合**: 干预片段作为正常 replay transition 存储，无需特殊处理
- **多任务统一**: 单一策略处理 8 种不同类型任务，无需任务特定微调

### 不能做什么
- **异构平台泛化未验证**: 所有实验在 Agibot G1 上完成，迁移到不同自由度/传感器配置的机器人需要额外验证
- **极端长尾场景**: 车队 4 小时/60 小时数据可能仍不足以覆盖所有边缘情况
- **零样本新任务**: 需要至少一些部署数据启动飞轮，不能直接泛化到预训练完全未覆盖的任务域

### 6.1 隐含假设 (Hidden Assumptions)

1. **车队规模足够覆盖分布**: 假设 16 台机器人并行 4 小时能收集到足够多样性的数据。如果任务空间更大（如 50+ 任务），可能需要更多机器人或更长时间
2. **稀疏二元奖励足够**: 仅用终端 r=1/0 信号。对于更复杂的任务（如部分成功也应奖励），可能需要 dense reward shaping
3. **人类干预是可记录的**: 假设操作员干预可以被准确记录并作为有效 transition。但干预质量依赖于操作员水平
4. **价值分布的 categorical 离散化足够**: 101 个 bin 的离散化可能无法精确捕捉多峰分布的精细结构
5. **QAM 的 KL 正则化强度适中**: 温度 λ 的选择影响策略改进幅度——太大则改进慢，太小则破坏预训练表示

## 7. 与相关工作对比 (Comparison)

| 方法 | 策略类型 | 数据利用 | 更新方式 | 部署规模 | 适用场景 |
|------|----------|----------|----------|----------|----------|
| **SFT/BC** | 通用 | 仅专家演示 | 监督学习 | 单任务 | 预训练基线 |
| **RECAP (π0.6*)** | 通用 | 仅离线 rollout | 离线 RL 迭代 | 单任务/离线 | 离线后训练 |
| **VLA-RL** | Specialist | 仅在线 on-policy | 在线 RL | 单机器人/仿真 | 仿真 specialist |
| **RIPT** | Specialist | 仅在线 on-policy | 在线 RL | 单机器人 | 真实 specialist |
| **HG-DAgger** | 通用 | 在线成功+干预 | 在线 ICL | 车队 | 部署后 ICL |
| **LWD (本文)** | **通用** | **离线+在线混合** | **offline-to-online RL** | **16 台车队** | **真实部署持续学习** |

**面试 Tip**: 当被问到「LWD 和 RECAP 的区别」时，回答：「RECAP 是纯离线迭代 RL，遵循 collect-train-deploy 循环，无法即时吸收部署中新数据；LWD 统一了 offline 和 online 阶段的 RL 目标，用分布价值学习 + 伴随匹配实现了 fleet-scale 的持续改进，尤其在长程稀疏奖励任务上优势明显。」

## 8. 精讀建議 (Reading Guide)

- **值得精讀原文的人**:
  1. 构建部署后持续学习系统的工程师——LWD 提供了首个 fleet-scale 的完整参考实现
  2. 研究 offline-to-online RL 的研究者——DIVL 的自适应分位数机制是 IQL 分布化的新思路
  3. 需要稳定更新 flow-based VLA 策略的开发者——QAM 在真实机器人上的应用案例稀缺

- **建議章節路徑**: 先读 §IV（方法核心：DIVL + QAM）→ 再看 §V-B（实验结果，验证 claim）→ 可跳 §II（相关工作，已有背景可略）→ 附录 B2（超参数细节，工程实现参考）

- **不值得精讀的理由**: 如果你不做真实机器人部署、已熟悉 IQL/QAM 的算法细节、或只关心纯离线 RL，读摘要和 Table I 即可。

---

**关键引用**:
- [LWD 项目页面](https://learning-while-deploying.github.io/)
- [arXiv:2605.00416](https://arxiv.org/abs/2605.00416)
- IQL: [Kostrikov et al. 2021](https://arxiv.org/abs/2110.06169)
- QAM: [原文引用 [31]](https://arxiv.org/abs/2605.00416)
- SOP 基础设施: [原文引用 [46]](https://arxiv.org/abs/2605.00416)
- π0.5 flow-based VLA: [原文引用 [5]](https://arxiv.org/abs/2605.00416)

[← Back to Theory](./README.md)
