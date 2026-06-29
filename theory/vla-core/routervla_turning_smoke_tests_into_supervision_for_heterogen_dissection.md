# RouterVLA：将烟雾测试转化为异构 VLA 选择的监督信号 (RouterVLA: Turning Smoke Tests into Supervision for Heterogeneous VLA Selection)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-06-29
>
> **论文**: RouterVLA: Turning Smoke Tests into Supervision for Heterogeneous VLA Selection
> **链接**: https://arxiv.org/abs/2606.27355
> **核心定位**: 将部署前的烟雾测试（smoke tests）从"成本"转化为"监督信号"——用少量 probe 执行记录构建专家档案，从异构 VLA 策略池中选出最适合当前条件的专家，在 LIBERO-Plus 上实现 +14.64pp 的 held-out 成功率提升。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 3 次烟雾测试 per expert + 简单的成功率规则，就能从异构 VLA 池中将 held-out 成功率从 46.86% 提升到 61.49%；更复杂的 learned scorer（LR/GBDT/MLP）在统计上无法超越该规则 |
| 适合精读 | 如果你在做多策略部署/模型路由/ensemble selection，重点看 §2（数学核心）和 §5（数据与评测） |
| 可以跳过 | 如果你只关心单模型 scaling 或训练新 VLA，这篇距离中等——它不训练模型，只调度已有模型 |
| 落地可行性 | 高（无需训练新模型，只需部署前 probe 执行 + 简单规则排序） |
| 主要风险 | 59.2% 的场景下专家间成功率持平（tie），且 layout 扰动下路由完全失效——需要更多证据或视觉上下文 |

💡 **X-Ray 开场**
> 机器人团队在部署 VLA 策略前通常会对每个候选做几次烟雾测试，然后选一个平均最好的"全局冠军"。这篇论文问：能不能把这些测试数据再利用起来，为每个具体场景选择最合适的专家？答案是：可以，而且只需要统计每次 probe 的成功率就够了——更复杂的 ML 模型并不会带来额外收益。关键洞察是"commissioning（部署前测试）本身携带了路由价值"，而不是 scorer 的容量。

📍 **研究全景时间线**
```
[2023] RT-1: 单一 transformer 控制策略
    ↓
[2024] OpenVLA / Octo: 开源通用 VLA，单模型轴扩展
    ↓
[2024] Scaling Laws: 更多数据/容量 → 更强单模型
    ↓
[2025] π0 → π0.5: 单模型能力持续提升
[2025] LIBERO-Plus: 暴露单模型在扰动下的性能差异
    ↓
[2025] MergeVLA / RoboRouter: 模型合并 / 语义检索路由
    ↓
[2026-06] RouterVLA ← 当前位置：部署前 smoke test → 监督路由
    → 局限：仅 scalar profile，layout 扰动下失效，59% tie 率
```

## 1. 核心架构/方法总览 (Overview / Architecture)

RouterVLA 不是一个神经网络架构，而是一个 **commissioning-and-selection 框架**。它将部署流程拆成两个明确阶段：

### 1.1 系统对比概览 (System Component Comparison)

| 维度 | Global Best（基线） | RouterVLA（本文） | Hindsight Bound（上限） |
|------|---------------------|-------------------|------------------------|
| **信息源** | 仅训练集成功率 | Probe 执行记录 + 训练先验 | 所有专家在 scored trial 上的真实结果 |
| **是否需要 probe** | 否 | 是（每专家 B=3 次） | 是（但事后才看） |
| **选择粒度** | 全局单一专家 | 每 variant 独立选专家 | 每 variant 选最优专家 |
| **LIBERO-Plus SR** | 0.4686 | 0.6149 | 0.7393 |
| **Δ vs Global** | — | +14.64pp | +27.07pp |
| **可部署性** | ✅ 是 | ✅ 是 | ❌ 否（需未来信息） |
| **部署成本** | 0 次额外执行 | ~65.5 次 probe/condition | N/A |

### 1.2 关键机制 (Key Mechanism)

RouterVLA 的核心流程：

1. **Commissioning（部署前测试）**：对每个候选专家执行 B 次 probe（默认 B=3），记录 success/failure、rollout length、duration、termination 等标量指标
2. **Profile 构建**：将 probe 记录压缩为 14 维向量 ϕ_e = [S_e, R_e, P_e, M_e]
   - S_e: probe 成功率汇总（经验值、Beta 后验均值/方差）
   - R_e: rollout 轨迹汇总（步数、时长、超时/早停比例）
   - P_e: 训练集先验（训练成功率、延迟先验）
   - M_e: probe 计数和缺失统计 mask
3. **Selector 排序**：用透明规则（empirical success rate）或 learned scorer（LR/GBDT/MLP）对每个专家打分
4. **Scored Execution**：选最高分专家执行一次 held-out trial，记录结果

⚡ **Eureka Moment**：**commissioning 本身携带了路由价值，scalar scorer 的额外容量并不能创造新价值。** 在 B=3 的预算下，简单的成功率计数已经饱和了可用的路由信号——更复杂的模型（GBDT、MLP）在统计上与透明规则不可区分。

### 1.3 信息流/架构图 (Flow / Diagram)

```
┌─────────────────────────────────────────────────────────┐
│                    部署前 (Commissioning)                  │
│                                                          │
│  Variant x (task + perturbation)                         │
│       │                                                  │
│       ├──▶ Expert e₁: 执行 B=3 次 probe ──→ ϕ₁(x)       │
│       ├──▶ Expert e₂: 执行 B=3 次 probe ──→ ϕ₂(x)       │
│       ├──▶ ...                                           │
│       └──▶ Expert eₙ: 执行 B=3 次 probe ──→ ϕₙ(x)       │
│                                                          │
│  Profile 构建: ϕ_e ∈ ℝ¹⁴ = [S_e, R_e, P_e, M_e]        │
│                                                          │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    路由选择 (Selection)                    │
│                                                          │
│  Score: s_e = g(ϕ_e)  for each e ∈ ℰₓ                  │
│       │                                                  │
│       ├── 透明规则: q_emp = c_e / B_e                    │
│       ├── 或 learned: LR / GBDT / MLP                    │
│       │                                                  │
│       ▼                                                  │
│  e* = argmax_e s_e                                       │
│                                                          │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    评分执行 (Scored)                       │
│                                                          │
│  用专家 e* 执行 held-out trial t                         │
│  记录 R = Y_{e*,x}^(t) ∈ {0, 1}                         │
│                                                          │
│  ⚠️ trial t 的结果不进入 profile（outcome-disjoint）       │
└─────────────────────────────────────────────────────────┘
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：
```
e* = argmax_e  c_e / B_e     （在 B 相等时，成功率计数即最优排序）
```

**目标**：在给定 variant x 和 probe 预算 B 的条件下，从候选专家池 ℰ_x 中选择专家 e*，使得 held-out trial t 的成功率最大化。

**核心方程**：

```
Profile:  ϕ_e = Φ(P_e, a_e)         # P_e = probe 记录集, a_e = 训练元数据
Score:    s_e = g(ϕ_e)              # g = 透明规则或 learned scorer
Selection: e* = argmax_{e ∈ ℰ_x} s_e
Reward:   R = Y_{e*,x}^(t) ∈ {0,1}  # held-out trial 的真实结果
```

**透明规则（三种等价）**：

```
经验成功率:    q_emp = c_e / B_e
均匀 Beta 后验: q_uni = (1 + c_e) / (2 + B_e)     # Beta(1,1) 先验
分层 Beta-Binomial: q_hier = (τ·μ_e + c_e) / (τ + B_e)  # τ 在训练集上调
```

当 B_e = B（相等预算）时，三者产生完全相同的专家排序。

**Learned Scorer 训练目标**：

```
L_BCE(θ) = (1/|D_train|) · Σ BCE(Y_e,x^(t), σ(g_θ(ϕ_e^(t)(x))))
```

- 仅用 leave-one-suite-out 的三个训练 suite 的数据
- 标签是 outcome-disjoint 的 held-out 成功率
- GBDT 使用相同的二分类标签

**变量说明**：

| 符号 | 含义 |
|------|------|
| ℰ | 冻结专家池（28 个异构 VLA） |
| ℰₓ | variant x 下可用的子集（均值 21.8） |
| B | 每专家 probe 次数（默认 3） |
| c_e | 专家 e 的成功 probe 数 |
| ϕ_e | 14 维 commissioning profile |
| Y_e,x^(t) | 专家 e 在 trial t 上的 binary 成功结果 |
| SR(θ) | 所有评估行的平均成功率 |
| U_x^(t) | hindsight upper bound = max_e Y_e,x^(t) |

> 符号与本文保持一致。关键约束：profile 构建时不能使用 scored trial t 的结果（outcome-disjoint）。

## 3. 带数字走一遍：玩具例子 (Worked Example)

假设一个 variant x 有 3 个候选专家，每专家 B=3 次 probe：

```
Expert A: 3 次 probe → [成功, 成功, 失败] → c_A = 2, q_emp = 2/3 = 0.667
Expert B: 3 次 probe → [成功, 成功, 成功] → c_B = 3, q_emp = 3/3 = 1.000
Expert C: 3 次 probe → [失败, 失败, 成功] → c_C = 1, q_emp = 1/3 = 0.333
```

**Selector 排序**：B > A > C → 选择 Expert B

**Scored Execution**：用 Expert B 执行 held-out trial t（t 的结果未参与 profile 构建）

假设实际结果：成功 → R = 1

**全局统计**（论文实际数据，Table 2）：

```
Global Best（无 probe）:  SR = 0.4686（选训练集最好的一个专家用于所有场景）
RouterVLA（B=3 probe）:  SR = 0.6149（每场景选最适合的专家）
Hindsight Bound:         SR = 0.7393（上帝视角，每次选真正最好的）

Gap to Hindsight: 0.7393 - 0.6149 = 0.1244（还有 12.44pp 的提升空间）
```

**Feature Ablation 走一遍**（Table 3，MLP 三 seed 诊断）：

```
仅 Prior:          SR = 0.4686  → 等同于 Global Best（无 probe 信息）
仅 Success:         SR = 0.6018  → +13.32pp vs Prior（核心信号）
仅 Rollout Traces:  SR = 0.6116  → 轨迹本身也有信息
Success + Prior:    SR = 0.6051  → baseline
+ Full Traces:      SR = 0.6158  → 仅 +1.07pp（轨迹信号与 success 重叠）
Permuted（对照）:    SR = 0.1011  → 随机，证明信号真实
```

关键发现：Success 特征已经捕获了大部分路由信号，额外轨迹特征的边际增益仅 1.07pp。

## 4. 工程视角 (Engineering View)

### 部署成本分析

| 指标 | 值 | 含义 |
|------|-----|------|
| 平均 probe 数/condition | 65.5 | 21.8 专家 × 3 probe，中位数 81 |
| 短列方案 (M=12, B=3) | 36 次 | 先按先验选 Top-12，再 probe |
| 激进短列 (M=3, B=1) | 3 次 | 仅 Top-3 各 1 次 probe，SR=0.5597 |
| Selector 推理成本 | 可忽略 | LR/GBDT/MLP 对 14 维向量推理 < 1ms |

### 工程含义

1. **Smoke test 是沉没成本**：如果部署前本来就要做烟雾测试，那么 RouterVLA 的路由层是"零边际成本"的——它只是把已有的测试数据重新利用。

2. **瓶颈在证据获取，不在 scorer**：65.5 次 probe/condition 是主要成本。如果这些 probe 是额外工作，真正的工程问题不是"用更大的模型"而是"应该测试谁、何时停止"——即 active commissioning。

3. **短列策略的 trade-off**：
   - M=12, B=3: SR=0.6185（接近全量 0.6149），probe 数减半
   - M=3, B=1: SR=0.5597（下降 ~5.5pp），但仅需 3 次 probe
   - 选择取决于 probe 成本 vs 成功率提升的权衡

4. **Tie 率 59.2%**：近六成场景下多个专家成功率相同，此时只能靠训练先验和固定 ID 顺序打破平局。这意味着在 B=3 时，成功率信号不够细粒度。

5. **Layout 扰动下的脆弱性**：当扰动改变视觉布局时，scalar profile 无法捕捉变化，路由增益为负。需要图像-语言上下文。

## 5. 数据与评测 (Data & Eval)

### 数据集

| 指标 | 值 | 来源 |
|------|-----|------|
| 总记录数 | 34,960 | 多平台 ledger |
| LIBERO-Plus 有效记录 | 34,752 | 本文分析子集 |
| Task-perturbation variants | 398 | 4 suites × 3 扰动轴 |
| 冻结专家数 | 28 | 异构 VLA（不同架构/训练） |
| 每 variant-expert 对的 trial 数 | 4 | ID 0-3 |
| Variant-trial 评估行 | 1,592 | 398 variants × 4 folds |

### 评测协议

- **主协议**：3-to-1 trial-disjoint cross-fitting（3 次 probe 构建 profile，1 次 held-out 评分）
- **泛化协议**：leave-one-suite-out（三个 suite 训练，一个 suite 测试）
- **不确定性**：10,000 次 variant-cluster bootstrap
- **诊断协议**：same-trial reuse（故意允许 scored trial 进入 profile，测量乐观偏差）

### 基线

| 基线 | 说明 |
|------|------|
| Global Best | 选训练集成功率最高的单一专家，忽略 target probe |
| Hindsight Bound | 事后选最优专家（不可部署的上限） |
| Permuted Profile | 随机打乱 profile 特征（验证信号真实性） |

## 6. 能力与失败模式 (Capabilities & Failure Modes)

### 能做什么

| 场景 | 表现 | 原因 |
|------|------|------|
| 相机偏移扰动 | SR +14.64pp | 不同专家对相机偏移的鲁棒性不同，probe 能区分 |
| 物体位置变化 | SR +14.64pp | 专家在物体重排下的成功率差异可被 probe 捕获 |
| 多专家互补 | 有效利用 | 28 个专家的平均可用数 21.8，互补性强 |
| 低成本部署 | 可行 | 无需训练，只需 probe + 简单规则 |

### 不能做什么

| 场景 | 表现 | 原因 |
|------|------|------|
| Layout 扰动 | 路由增益为负（MLP Δ 负值） | Scalar profile 无法捕捉视觉布局变化 |
| B=1 低预算 | SR ≈ 0.5837（下降 3pp） | 单次 probe 信息不足，tie 率更高 |
| 零 probe 新场景 | 不适用 | RouterVLA 假设重复使用场景，非零 probe 路由 |
| 59.2% tie 场景 | 靠先验/ID 打破 | 成功率信号不够细粒度 |

### 6.1 隐含假设 (Hidden Assumptions)

1. **Probe 成本可接受**：假设部署前做 65.5 次 probe（平均）是可行的。对于真实机器人，这可能意味着数十分钟的测试时间。
2. **Trial ID 轮换 ≈ 时间泛化**：3-to-1 cross-fitting 用 ledger ID 轮换而非真实时间顺序。ID 不是时间戳，所以这不等同于部署后的 chronological generalization。
3. **专家池固定**：所有 28 个专家对每个 variant 都可用（仅 0.8% 的 variant 只有 1 个专家）。真实部署中专家可用性可能更稀疏。
4. **Scalar profile 足够**：论文聚焦于 scalar-only profile。这排除了图像/语言特征的潜在价值——作者自己也承认 layout 扰动下需要视觉上下文。
5. **LIBERO-Plus 代表真实部署**：虽然 LIBERO-Plus 包含多种扰动，但它仍是仿真环境。真实机器人的 probe 噪声、机械误差等未建模。

## 7. 与相关工作对比 (Comparison)

| 方法 | 关注点 | 路由依据 | 是否需要训练 | 架构感知 |
|------|--------|----------|-------------|---------|
| **RouterVLA（本文）** | 部署前 smoke test 路由 | Probe 执行记录（scalar） | 否（透明规则即可） | 否（architecture-agnostic） |
| RoboRouter [Chen 2026] | 语义检索路由 | 历史任务相似度 + 执行记录 | 是（检索器） | 否 |
| MoIRA [Kuzmenko 2025] | 语言指令路由 | 文本相似度 / LM | 是 | 否 |
| MergeVLA [Fu 2025] | 模型内部路由 | 任务特定 adapter 选择/合并 | 是 | 是（需共享架构） |
| Algorithm Selection [Rice 1976] | 通用算法选择 | 问题特征描述符 | 是 | N/A |
| LLM Routing [Ong 2024] | LLM 服务路由 | 查询特征 + 质量/成本 | 是 | 否 |

**关键区别**：RouterVLA 是唯一一个将"部署前 commissioning"本身作为路由信号来源的工作。它不依赖语义相似度、不训练新模型、不要求共享架构——它只是把已经要做的烟雾测试数据重新利用。

> **面试 Tip**：如果被问到"RouterVLA 和 RoboRouter 的区别"，回答核心是：RoboRouter 用历史任务的语义相似性做零 probe 路由，RouterVLA 假设已有 probe 预算、用 probe 结果做场景特定路由——前者解决"没测试过怎么选"，后者解决"测试过了怎么更好利用"。

## 8. 精读建议 (Reading Guide)

**值得精读原文的人**：
- 做多策略 VLA 部署的工程团队——需要决定"该保留多少个 checkpoint、如何调度"
- 研究 model routing / ensemble selection 的研究者——本文的 outcome-disjoint 协议是方法论贡献
- 关注 commissioning/qualification 流程的机器人系统工程师——本文把 commissioning 从成本重构为信号

**建议章节路径**：
1. 先读 §Introduction（理解动机：commissioning 作为监督 vs 成本）
2. 再看 §Problem Setup + §RouterVLA（理解 profile 构建和 selector 设计）
3. 重点看 §Results 的 Table 2（主结果）和 Table 3（feature ablation）
4. 可跳过 §Related Work（除非你关心具体引用）

**不值得精读的理由**：
- 如果你只做单模型 scaling（不维护专家池），这篇距离较远
- 如果你已经熟悉 algorithm selection / multi-armed bandit，方法论部分没有太多新内容
- 如果你关注的是训练新 VLA 而非调度已有 VLA，这篇不直接相关

---
[← Back to Theory](./README.md)

**关键引用**：
- [论文 arXiv](https://arxiv.org/abs/2606.27355)
- [LIBERO-Plus](https://github.com/LIBERO-Benchmark/LIBERO) (Fei et al., 2025)
- [OpenVLA](https://arxiv.org/abs/2406.09246) (Kim et al., 2024)
- [π₀.₄](https://www.physicalintelligence.company/) (Physical Intelligence et al., 2025)
