# ChainVLA：通过统一执行状态链式连接 VLA 查询以实现长视界操作 (ChainVLA: Chaining Vision-Language-Action Queries through a Unified Execution State for Long-Horizon Manipulation)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-08-05
>
> **论文**: ChainVLA: Chaining Vision-Language-Action Queries through a Unified Execution State for Long-Horizon Manipulation
> **链接**: https://arxiv.org/abs/2608.02326
> **核心定位**: 解决 action-chunked VLA 在查询边界处的「状态断连」问题——通过统一的执行状态（Progress Context + Motion Tail）将连续查询链式连接，在仅 1.2B 参数下超越 8B+2B 的 Mem-0。

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | 在 receding-horizon VLA 中，同时传递「任务进度」和「未完成运动」两个组件到下一个查询，可显著提升长视界操作成功率 |
| 適合精讀 | 如果你在做多步操作规划、action-chunking VLA、或记忆增强策略，重点看 §3.2-3.4（方法）和 §4.3（消融） |
| 可以跳過 | 如果你只关心单步操作或不需要跨查询记忆，这篇距离中等 |
| 落地可行性 | 中（需要 DiT decoder + 有状态推理框架，但核心思想可迁移到任何 chunked VLA） |
| 主要風險 | 消融实验显示移除任一组件性能崩溃剧烈（3.0% / 11.2%），说明两个组件高度耦合，单独移植其中一个可能无效 |

💡 **X-Ray 开场**
传统的 action-chunked VLA（如 ACT、π0）在每个查询边界都「从零开始」——只看当前观察、 proprioception 和指令，完全丢弃上一轮查询预测的未执行部分，也无法保留已经离开视野的任务证据。ChainVLA 发现：人类做长视界操作时，同时维护「已完成的进度」和「正在进行的运动」两条线索。它把这两条线索编码为一个统一的执行状态，在查询间传递，使每个新预测既继承历史又响应最新观察。结果：在记忆依赖型 RMBench 上达到 62.8%（比 Mem-0 高 10 个百分点），且参数量仅 1.2B。

📍 **研究全景时间线**
```
[2023] ACT (chunked replan, 无状态) → [2024] Temporal Ensemble (解码后融合)
  → [2025] π0 / Real-Time Flow (执行侧连续性) → [2025] MemoryVLA (记忆增强)
  → [2026] Mem-0 (8B+2B, 记忆+动作) → [本文] ChainVLA (统一执行状态, 1.2B)
  ← 当前位置：首次将「任务进度」与「未完成运动」在模型内统一链式传递
```

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 维度 | ACT / Diffusion-Policy | MemoryVLA | Mem-0 | ChainVLA (本文) |
|------|----------------------|-----------|-------|-----------------|
| 查询间状态 | 无（每次从零解码） | 感知-认知记忆库（每 timestep 读写） | 8B 视觉 + 2B 动作模型 | Progress Context + Motion Tail |
| 记忆类型 | 无 | 全量 perceptual-cognitive bank | 外部记忆模块 | Recurrent Working State + Sparse Event Memory |
| 运动连续性 | 无 | 无（仅记忆观察） | 无（仅记忆观察） | Motion Tail（前一轮未执行后缀） |
| 解码方式 | 自回归 / Diffusion | Diffusion | 外部动作模型 | Conditional-Flow + DiT |
| 参数量 | 0.3B-7B | 7B+0.3B | 8B+2B | **1.2B** |
| 训练数据 | 50 demos/task | 多任务 | 多任务 | 50 demos/task |
| RMBench Avg | 6.8-52.8% | 19.6% | **52.8%** | **62.8%** |

### 1.2 关键机制 (Key Mechanism)

ChainVLA 的核心创新是将查询间传递的「执行状态」拆分为两个互补组件：

**组件 1：Progress Context（任务进度）** — 回顾性半部
- **Recurrent Working State**：维护 live tokens（当前执行摘要）+ bounded cache（近期 live tokens 的 FIFO 缓存）
- **Sparse Event Memory**：仅在触发条件满足时写入关键事件（视觉特征 + Stage Info + 时间特征），检索时考虑事件年龄和当前任务阶段估计
- **融合方式**：g_k = Fuse(L_k, Z_k^e)，即 live tokens + 从事件记忆中检索到的证据

**组件 2：Motion Tail（未完成运动）** — 前瞻性半部
- 前一轮预测中**未被执行的后缀**（a_{k, h_exec+1}, ..., a_{k, H}）
- 通过两条模型内路径影响下一轮：① 编码为 tail tokens 进入 Working State；② 对齐后初始化解码器的动作生成
- 关键性质：不是固定命令，而是「可修正的延续先验」——解码器在每个位置都重新生成，可以完全覆盖

⚡ **Eureka Moment**：「运动连续性帮助保留了用于推断任务进度的观察流」——移除 Motion Tail 导致 RMBench 降到 11.2%，移除 Progress Context 降到 3.0%，这种不对称性说明：没有运动连续性，观察流断裂，任务进度也无法正确推断。两个组件不是独立的，而是**级联依赖**的。

### 1.3 信息流/架构图 (Flow / Diagram)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Query k → Query k+1 转换                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  输入: o_k, r_k, ℓ (观察, 本体感受, 指令)                              │
│       ↓                                                              │
│  ┌──────────────┐    ┌──────────────────┐                            │
│  │  Vision Enc. │    │  Proprio Enc.    │                            │
│  │ (Florence-2) │    │   ϕ_r(r_k)       │                            │
│  └──────┬───────┘    └────────┬─────────┘                            │
│         │                     │                                      │
│         ▼                     ▼                                      │
│  ┌─────────────────────────────────────────────┐                    │
│  │           Working State Update               │                    │
│  │  η_k = MLP[ϕ_r, Pool(P_k), Pool(U_k), τ_k]  │                    │
│  │  L̃_k = CrossAttn(Q_W + η_k, X_k)            │                    │
│  │  L_k = LN(L̃_k + m_live · CrossAttn(L̃_k, L_{k-1}))│              │
│  └────────────────────┬────────────────────────┘                    │
│                       │                                             │
│              ┌────────┴────────┐                                    │
│              ▼                 ▼                                    │
│    ┌───────────────┐  ┌──────────────────┐                         │
│    │  Live Tokens  │  │  Summary W̄_k     │                         │
│    │     L_k       │  │  (pool over L,B,P,U)│                        │
│    └───────┬───────┘  └────────┬─────────┘                         │
│            │                   │ retrieval query                    │
│            │          ┌────────▼─────────┐                         │
│            │          │ Sparse Event Mem │                         │
│            │          │  C_k = {(κ,E,b)} │                         │
│            │          └────────┬─────────┘                         │
│            │                   │ top-2 retrieved                    │
│            │                   ▼                                  │
│            │          ┌──────────────────┐                         │
│            │          │  Readout Z_k^e   │                         │
│            │          └────────┬─────────┘                         │
│            │                   │                                   │
│            └────────┬──────────┘                                   │
│                     ▼                                              │
│            ┌──────────────────┐                                    │
│            │ Progress Context │                                    │
│            │   g_k = Fuse     │                                    │
│            └────────┬─────────┘                                    │
│                     │                                              │
│  ┌──────────────────┴──────────────────┐                          │
│  │           Motion Tail u_k            │                          │
│  │  (前一轮未执行后缀 a_{k-1, h_exec+1:H})│                         │
│  │  路径1: 编码为 U_k → Working State   │                          │
│  │  路径2: 对齐为 μ_k → 解码器初始化     │                          │
│  └──────────────────┬──────────────────┘                          │
│                     │                                              │
│                     ▼                                              │
│            ┌──────────────────┐                                    │
│            │  DiT Decoder      │                                    │
│            │ (Conditional-Flow)│                                    │
│            │ 条件: X_k, g_k, u_k │                                  │
│            └────────┬─────────┘                                    │
│                     │                                              │
│                     ▼                                              │
│            ┌──────────────────┐                                    │
│            │  A_k = (a_{k,1},  │                                    │
│            │    ..., a_{k,H}) │                                    │
│            │ 执行前 h_exec 步  │                                    │
│            │ 后缀 → u_{k+1}   │                                    │
│            └──────────────────┘                                    │
└─────────────────────────────────────────────────────────────────────┘
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：
```
Π_θ(· | x_k, s_k) ≈ Π*(· | H_k),  其中 s_k = (g_k, u_k)
```
策略以当前输入 x_k 和执行状态 s_k 为条件，逼近最优历史条件策略。执行状态 s_k 的递归更新：s_{k+1} = F_θ(s_k, x_k)。

**目标**：在 receding-horizon 控制框架下（1 ≤ h_exec < H），每个查询预测 H 步动作，执行前 h_exec 步，然后重新查询。关键问题是：如何让当前查询「知道」之前查询推断出的任务进度和未完成的运动意图？

**核心方程分解**：

```
执行状态: s_k = (g_k, u_k)

Progress Context (回顾性):
  g_k = Fuse(L_k, Z_k^e)
  L_k = LN(L̃_k + m_{k-1}^live · CrossAttn(L̃_k, L_{k-1}))
  L̃_k = CrossAttn(Q_W + η_k, X_k)
  η_k = MLP[ϕ_r(r_k), Pool(P_k), Pool(U_k), τ_k]

  Z_k^e = CrossAttn-Readout({E_i | i ∈ top-2 检索结果})
  ρ_{k,i} = cos(LN(q_k^e), LN(κ_i + Emb_age(clip(k - b_i))))

Motion Tail (前瞻性):
  u_{k+1} = Tail(A_k) = (a_{k, h_exec+1}, ..., a_{k, H})
  μ_{k+1} = I(u_{k+1})  (对齐到完整 horizon 表示)
  Ã_{k+1}^{(0)} = μ_{k+1} + σ_u · ε,  ε ~ N(0, I)

动作生成:
  A_k ~ Π_θ(· | x_k, g_k, u_k)  (DiT conditional-flow decoder)
```

**变量说明**：

| 符号 | 含义 | 更新频率 |
|------|------|---------|
| x_k | (o_k, r_k, ℓ) 当前查询输入 | 每查询 |
| s_k | (g_k, u_k) 执行状态 | 每查询 |
| g_k | Progress Context（任务进度） | 每查询 |
| L_k | Live tokens（当前执行摘要） | 每查询 |
| B_k | Bounded cache（近期 live tokens FIFO） | 每查询 |
| Z_k^e | 从事件记忆检索的证据 | 每查询（只读） |
| C_k | Sparse Event Memory {(κ, E, b)} | 触发写入 |
| u_k | Motion Tail（前一轮未执行后缀） | 每查询 |
| h_exec | 每查询执行步数 | 固定 |
| H | 动作视界长度 | 固定 |

> 符号与本文保持一致。Fuse 为 policy-facing 融合操作（具体结构论文未完全展开，疑似 cross-attention 或 gated fusion）。I 为对齐算子，将后缀对齐到完整 horizon 表示。

## 3. 带数字走一遍：玩具例子 (Worked Example)

假设一个简化的 2D 放置任务，H = 10, h_exec = 3：

**Query 1**：
- 观察：桌上有 3 个积木，指令「把红色放左边」
- 无 Motion Tail（episode 开始，tail path masked）
- 预测 A_1 = [a_1, a_2, ..., a_10]，其中 a_1..a_3 = 「抓取红色积木」
- 执行 a_1..a_3，此时红色积木已被抓起
- 未执行后缀 u_2 = [a_4, ..., a_10] = 「移动到左侧上方 → 释放」

**Query 2**：
- 观察：手中有红色积木，但原始位置已空（关键！原始位置信息已从视野消失）
- Progress Context：Working State 携带 L_1（编码了「已抓取红色」的信息），Sparse Event Memory 可能存储了「红色初始位置」作为锚点事件
- Motion Tail u_2：提供「移动到左侧上方 → 释放」的延续先验
- 解码器初始化：Ã_2^{(0)} = I(u_2) + σ_u · ε（以 u_2 为均值加噪声）
- 预测 A_2 = [a'_1, ..., a'_10]
  - a'_1..a'_3 可能与 u_2 的前 3 步不同（解码器根据最新观察修正了）
  - 但 u_2 提供了合理的起点，避免了从零开始的剧烈跳跃
- 执行 a'_1..a'_3，生成新的 u_3

**关键数值**：论文中 Full 模型在 Put Back Block 上达到 96% 成功率，该任务要求将积木放回已离开视野的垫子上。移除 Motion Tail 后降到 0%，说明没有运动连续性，观察流断裂，Progress Context 也无法正确更新。

## 4. 工程视角 (Engineering View)

| 工程维度 | ChainVLA 设计 | 含义 |
|---------|--------------|------|
| 参数量 | 1.2B（Florence-2-large + DiT） | 比 Mem-0 (10B) 小 8 倍，比 MemoryVLA (7.3B) 小 6 倍 |
| 每查询计算 | 1 次完整前向（encoder + decoder） | 与标准 chunked VLA 相同，额外开销仅 Working State 更新 |
| 状态存储 | Working State (live tokens + bounded cache) + Event Memory | 有界存储，episode 间清空 |
| 推理延迟 | 标准 DiT 解码延迟 + state update 开销 | 增加约 5-10%（cross-attention + retrieval） |
| 控制频率 | h_exec / H 决定查询频率 | h_exec 越小 = 查询越频繁 = Motion Tail 越短但修正越及时 |
| 部署约束 | 需要维护跨查询状态 | 不能是无状态推理服务；需要 episode 级别的状态管理 |
| 训练开销 | 与标准 VLA 相同（unroll 训练） | 额外 overlap-consistency regularizer（轻微） |
| 量化友好度 | DiT + cross-attention | 中等（attention 对量化敏感） |

**工程含义**：ChainVLA 的核心工程 trade-off 是「状态管理复杂度 vs. 长视界性能」。它不要求额外的 planner 或 world model，但要求推理时维护有状态上下文。对于部署而言，这意味着：
- 每个 episode 需要独立的 state 对象
- 查询边界需要缓存未执行后缀
- 事件记忆的读写需要触发逻辑

## 5. 数据与评测 (Data & Eval)

**训练数据**：
- RMBench：5 个记忆依赖型双臂操作任务，每任务 50  demonstrations
- LIBERO：4 个套件（Spatial, Object, Goal, Long），每套件 1 个模型 finetune
- 训练协议：20,000 optimization steps, batch size 4, Florence-2-large backbone

**评测协议**：
- RMBench：每任务 100 rollout episodes，报告成功率
- LIBERO：每任务 50 episodes，报告成功率
- 边界诊断指标：CD_p / CD_R（位姿不连续性）、B2（速度变化）、RMSE_3（前后预测重叠段不一致）

**对比基线**（论文 Table 1）：
- ACT, Diffusion-Policy, π0, π0.5, X-VLA, GR00T-N1
- MemoryVLA (7B+0.3B), Mem-0 (8B+2B), MemoAct, CronusVLA
- 消融变体：w/o Stage Ann., w/o Live Tokens, w/o Event Readout, w/o Progress Ctx., w/o Tail Tokens, w/o Traj. Init., w/o Motion Tail, w/o Both, FIFO Hist. + TE

## 6. 能力与失败模式 (Capabilities & Failure Modes)

### 能做什么
- **记忆依赖操作**：在 RMBench 的 Put Back Block（96%）和 Rearrange Blocks（93%）上表现突出，这些任务需要记住已离开视野的状态
- **跨查询修正**：Motion Tail 提供延续先验但允许完全覆盖， Decoder 可以根据新观察修正任何 horizon 位置
- **小参数高效**：1.2B 参数超越 10B 级的 Mem-0，在 LIBERO 上达到 98.8%（接近天花板）
- **阶段感知事件管理**：有阶段标注数据时，利用阶段转换触发事件写入和检索，提高记忆效率

### 不能做什么 / 失败模式
- **移除任一组件即崩溃**：w/o Progress Ctx. → 3.0%，w/o Motion Tail → 11.2%。两个组件高度耦合，无法独立工作
- **仅靠解码后平滑不够**：Linear Continuation 和 Temporal Ensemble 在 Put Back Block 上分别仅恢复 0% 和 0% 成功率（Table from project page），说明输出级缝合不能替代模型内状态传递
- **单一机器人平台**：所有实验在双臂桌面操作平台上进行，未验证移动机器人、人形或真实世界部署
- **50 demonstrations 设定**：训练数据量较小，未测试大规模数据下的扩展性
- **Episode 间状态清空**：跨 episode 不传递知识，无法利用跨 episode 的累积学习

### 6.1 隐含假设 (Hidden Assumptions)

1. **查询频率固定且已知**：h_exec 和 H 是超参数，论文未探索自适应查询频率
2. **事件写入规则可泛化**：阶段标注数据用阶段转换触发写入，无标注数据用固定周期触发。这两种策略的边界在哪里？论文未深入分析
3. **Motion Tail 的噪声尺度 σ_u 是固定的**：论文未讨论 σ_u 的自适应或学习任务依赖性
4. **DiT decoder 的 conditional-flow 框架是必要的**：论文使用 DiT + conditional-flow，但未对比自回归 decoder 是否也能受益于同样的状态链式机制
5. **50 demos 足够**：在 RMBench 这种记忆密集型任务上，50 demonstrations 是否足够训练出可靠的状态传递能力？未做数据规模消融

## 7. 与相关工作对比 (Comparison)

| 方法 | 核心思路 | 记忆类型 | 运动连续性 | 参数量 | RMBench Avg |
|------|---------|---------|-----------|--------|-------------|
| ACT | chunked replan | 无 | 无 | 0.3B | 6.8% |
| π0.5 | flow-based | 无 | 执行侧融合 | 3.3B | 14.4% |
| MemoryVLA | 全量 perceptual-cognitive bank | 每 timestep 读写 | 无 | 7.3B | 19.6% |
| Mem-0 | 外部记忆 + 动作模型 | 外部记忆模块 | 无 | 10B | 52.8% |
| FIFO Hist. + TE | 观察历史 + 解码后融合 | 固定长度历史 | 解码后 | 1.2B | 35.6% |
| **ChainVLA** | **统一执行状态链式传递** | **Working State + Event Memory** | **Motion Tail（模型内）** | **1.2B** | **62.8%** |

**面试 Tip**：当被问到「ChainVLA 和 MemoryVLA 有什么区别」时，回答：「MemoryVLA 在每一步读写全量记忆库，关注的是观察侧的记忆；ChainVLA 独特地将'前一轮未执行的运动意图'也作为状态传递，并且用稀疏事件记忆替代全量读写。两者的关键差异不是'有没有记忆'，而是'记忆什么'——ChainVLA 记忆的是'任务进度 + 未完成运动'这对互补信号，且通过消融证明它们级联依赖。」

## 8. 精讀建議 (Reading Guide)

**值得精讀原文的人**：
- 做多步操作规划 / receding-horizon VLA 的研究者——§3.1-3.4 的形式化定义和 §4.3 的消融分析必读
- 要评估在现有 VLA 上添加跨查询状态可行性的工程师——§3.2 的执行状态转换和 §4 的工程指标有直接参考价值
- 对 DiT + conditional-flow 在动作生成中应用的感兴趣者——§3.4 的 Motion Tail 初始化和 §3.5 的 overlap-consistency regularizer 是亮点

**建議章節路徑**：
1. 先读 §1 Introduction + Figure 1 — 理解问题动机（为什么 chunked VLA 在查询边界丢失信息）
2. 再看 §3.2 The Execution State + Figure 2 — 掌握核心架构
3. 然后读 §3.3-3.4 — 深入 Progress Context 和 Motion Tail 的细节
4. §4.3 Ablation Analysis — 理解两个组件的级联依赖关系
5. 可跳 §2 Related Work（如已熟悉该领域）和 §4.1 Setup 中的对比方法细节

**不值得精讀的理由**：
- 如果你不做长视界操作或不需要跨查询记忆——读摘要和 §1 即可
- 如果你只关心单步 VLA 或已熟悉 MemoryVLA 类方法——本文的核心贡献（Motion Tail）对你可能不是优先项
- 如果你关注真实世界部署——论文仅在仿真环境评估，未涉及 sim2real


---
[← Back to Theory](./README.md)

**关键引用**：
- 论文: https://arxiv.org/abs/2608.02326
- 项目页: https://muqy1818.github.io/chainvla-web/
- RMBench 基准: Chen et al. 2026a
- Florence-2-large 视觉编码器: Xiao et al. 2024
