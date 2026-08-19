# Zetta ζ：高效闭环具身 Harness 实现物理智能自我进化 (Zetta ζ: An Efficient Closed-Loop Embodied Harness for Self-Evolving Physical Intelligence)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-08-19
>
> **论文**: Zetta ζ: An Efficient Closed-Loop Embodied Harness for Self-Evolving Physical Intelligence
> **链接**: https://arxiv.org/abs/2608.16590
> **核心定位**: 解决现有具身 Agent Harness 开环执行、无法在物理运行中持续学习的根本缺陷，通过三层时间尺度闭环 + 可进化 Harness 实现 VLA 策略冻结条件下的自我进化，在 LIBERO-Pro 和 RoboCasa 上达到 SOTA。

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | 在冻结 VLA 策略参数的前提下，通过可进化的运行时 Critic + Recovery Harness，实现物理执行中的持续闭环学习，成功率随自我探索迭代持续提升 |
| 适合精读 | 如果你在做具身 Agent 框架设计、VLA 后训练、闭环策略优化、或 Embodied Infrastructure，§2（方法）和 §4（实验）值得逐节细读 |
| 可以跳过 | 如果你只关心端到端 VLA 模型训练（如 RT-2、OpenVLA），这篇的 harness 层设计距离较远 |
| 落地可行性 | 中（需要 LLM-as-Judge 基础设施 + 仿真环境批量 rollout 能力；真实机器人部署需额外适配） |
| 主要風險 | 所有实验均在仿真环境完成（LIBERO-Pro / RoboCasa），真实物理部署效果待验证；LLM 驱动的诊断/修复 Agent 成本较高 |

💡 **X-Ray 开场**
现有具身 Agent 在执行任务时是"开环"的——一旦开始 rollout，策略就按固定技能执行，失败后只能在 episode 结束后做事后反思。这种事后反思无法在物理交互过程中实时纠正错误，因为毫秒级的机器人-环境状态变化超出了大模型的决策频率。Zetta 的核心发现是：引入可在线进化的代码级运行时 Critic（监控函数），在动作频率上持续监测执行状态并触发 Recovery，就能在不修改 VLA 策略参数的前提下，让具身系统实现真正的闭环自我进化。对 VLA 研究者而言，这提供了一条"冻结大模型 + 进化小 harness"的实用路径。

📍 **研究全景时间线**
```
[2023] VLA 端到端策略 (RT-1, OpenVLA) ── 强语义理解，但开环推理
       ↓
[2024] 具身 Agent Harness (RoboAgent, RPent) ── LLM 编排工具，但事后反思
       ↓
[2025] 闭环反思尝试 (SkillOpt, EmbodiSkill) ── 代码空间优化，但 episode 级
       ↓
[2026-08] ★ Zetta ζ ── 三层时间尺度闭环 + 可进化 Harness + 专用基础设施
       ← 当前位置：冻结策略 + 进化 harness = SOTA
```

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 组件 | 角色 | 输入 | 输出 | 运行频率 | 是否可进化 |
|------|------|------|------|----------|-----------|
| Action Policy (π) | 底层动作生成 | 观察 s_t, 目标 g | 动作 a_t | 动作级 (~10Hz) | ❌ 冻结 (∇θ=0) |
| Orchestrator Agent (𝒜_orch) | 高层仲裁 | Critic 提议 P_t, R, 𝒯, 任务知识 𝒦 | 执行模式 σ_t | 事件触发 | ❌ 冻结 |
| Runtime Critic (C) | 高频监控 | 轨迹 τ_{0:t} | 提议 P_t = ⟨e_t, σ̂_t⟩ | 动作级 | ✅ 在线进化 |
| Recovery Playbook (R) | 恢复策略库 | 失败证据 e_t | 恢复动作序列 | 事件触发 | ✅ 在线进化 |
| Toolset (𝒯) | 可执行工具集 | 任务需求 | 规划/抓取/恢复等 | 按需 | ✅ 在线进化 |
| Evolutionary Agent (𝒜_evo) | 离线优化 | 失败数据 𝒟_fail, 当前 H | 更新后的 H | 迭代级 | N/A (元控制器) |

### 1.2 关键机制 (Key Mechanism)

Zetta 的核心设计围绕 **三层时间尺度分离的闭环**：

**Loop 1 — Critic-Governed Action Loop（动作级）**
- 运行时 Critic 以动作频率持续扫描轨迹 τ_{0:t}
- 检测到失败前兆（碰撞、停滞、位姿偏差）时生成提议 P_t = ⟨e_t, σ̂_t⟩
- Orchestrator Agent 验证证据后批准模式切换（σ_t 从 0=VLA 切换到 >0=专用工具）
- **关键**: 这是闭环执行的引擎，让系统在物理交互过程中实时干预

**Loop 2 — Rollout-Batch Candidate Optimization Loop（Rollout 批次级）**
- 每批 rollouts 完成后，对失败轨迹进行聚类（基于 Earliest Observable Divergence）
- 对每个聚类选取 medoid seed 进行深度因果诊断
- 通过六层自上而下诊断（Evaluation → Critic → State → Planning → Recovery → Parameter）定位根因层 L*
- 生成候选 Critic 和 Recovery patch

**Loop 3 — Validation-Gated Skill Update Loop（迭代级）**
- 仅在候选 patch 在 hold-out 验证集上提升成功率且具备泛化能力时，才写入技能记忆
- 防止过拟合修复（over-parameterized repair）破坏 VLA 的语义泛化能力

⚡ **Eureka Moment**: 不修改 VLA 策略本身，而是进化一个外置的"监控+恢复"Harness —— 就像给自动驾驶系统加一个不断自我进化的副驾安全员，而不是重新训练驾驶员。

### 1.3 信息流/架构图 (Flow / Diagram)

```
                    ┌─────────────────────────────────────────────┐
                    │              Z-Infra (Rollout Infra)         │
                    │  ┌──────────┐    ┌──────────┐    ┌───────┐  │
                    │  │Env Workers│───▶│Model Wrkrs│───▶│Scheduler│ │
                    │  └──────────┘    └──────────┘    └───────┘  │
                    └─────────────────────────────────────────────┘
                                    │ 并行 Rollouts
                    ┌───────────────▼───────────────┐
                    │   π_VLA (frozen) + ℋ = {C,R,𝒯}│
                    │                               │
                    │  ┌───┐  τ成功 ──▶ I_succ      │
                    │  │π  │                        │
                    │  └───┘  τ失败 ──▶ ℳ_fail       │
                    │                               │
                    │  C monitors at action freq    │
                    │  σ_t = 𝒜_orch(P_t, R, 𝒯, 𝒦)  │
                    └───────────────┬───────────────┘
                                    │ ℳ_fail
                    ┌───────────────▼───────────────┐
                    │   𝒜_evo (Offline Evolution)    │
                    │                                │
                    │  Phase I: 失败画像 + 基线建立   │
                    │    ├─ 确定性调度               │
                    │    ├─ 有效性验证               │
                    │    └─ 成功索引 I_succ / 失败清单│
                    │                                │
                    │  Phase II Stage 1: 因果诊断     │
                    │    ├─ EOD 聚类                  │
                    │    ├─ Medoid 选取               │
                    │    └─ 六层自上而下诊断 → L*      │
                    │                                │
                    │  Phase II Stage 2: Harness 修复  │
                    │    ├─ C* 增强 / R* 起草 / 𝒯* 适配│
                    │    └─ VLA Re-entry Contract Ψ   │
                    │                                │
                    │  Phase III: 泛化 + 打包          │
                    │    ├─ 版本化 Harness Package     │
                    │    └─ Hold-out 泛化验证          │
                    └───────────────┬───────────────┘
                                    │ ℋ^{(k+1)}
                    ────────────────┘ (反馈到执行循环)
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：
```
max_H  J(H) = E_{g,s0~D} [ Success(τ) | π(frozen), 𝒜_orch(frozen), H ]
```

**目标**: 在冻结 VLA 策略 π 和仲裁器 𝒜_orch 的前提下，通过优化可进化 Harness H = {C, R, 𝒯} 来最大化期望任务成功率。

**关键方程**:

```
(1) Harness 定义:
    H = {C, R, 𝒯}

(2) Critic 提议生成:
    P_t = C(τ_{0:t}) = ⟨e_t, σ̂_t⟩

(3) 在线仲裁:
    σ_t = 𝒜_orch(P_t, R, 𝒯, 𝒦)

(4) 离线进化:
    H^{(k+1)} ← 𝒜_evo(𝒟_fail^{(k)}, H^{(k)})

(5) First Missing Milestone:
    m* = min { m_k ∈ M | m_k ∉ {μ_t}_{t=0}^T }

(6) Earliest Observable Divergence:
    t_EOD = min { t | dist(s_t, s_t^ref) > ε, s_t^ref ∈ I_succ(μ_t) }

(7) 六层因果诊断:
    L* = arg max_{L∈ℒ} { IsRootCause(L) | τ证据, 主视角 }
    ℒ = {L_eval, L_crit, L_state, L_plan, L_recv, L_param}

(8) VLA Re-entry Contract:
    Ψ(s_t) = 𝟙(FailureCleared) ∧ 𝟙(Stability(s_t) > γ)
```

**变量说明**:

| 符号 | 含义 |
|------|------|
| π | 底层动作策略（VLA/WAM），参数冻结 |
| 𝒜_orch | 在线仲裁 Agent（固定 multimodal LLM） |
| C | 运行时 Critic 集合（监控函数） |
| R | Recovery Playbook（恢复策略库） |
| 𝒯 | Heterogeneous Toolset（可执行工具集） |
| σ_t | 执行模式（0=VLA, >0=专用工具） |
| e_t | 失败的可审计证据（碰撞/停滞/偏差） |
| m* | 第一个未达成的语义里程碑 |
| t_EOD | 最早可观测发散时间点 |
| L* | 诊断定位的根因层 |
| Ψ | VLA 重新接管控制权的条件谓词 |

**直觉**: 整个系统可以类比为一个"冻结内核 + 热插拔驱动"的操作系统。VLA 策略是内核（不修改），Harness 是驱动层（可以在线更新）。Critic 是中断检测器，Recovery 是错误处理例程，Evolutionary Agent 是自动驱动更新服务。

> 符号与本文保持一致。论文使用严格的数学形式化来描述 harness 的进化和仲裁逻辑。

## 3. 带数字走一遍：玩具例子 (Worked Example)

假设一个 RoboCasa 的 **PnP-Stove** 任务（将锅放到炉灶上），初始 VLA 成功率仅 42%。

**Iteration 0 — 基线采集 (Loop 1)**:
- 在 50 个 development seeds 上纯 VLA 执行
- 结果: 21/50 成功 (42%)
- 失败轨迹收集到 ℳ_fail，包含 29 条失败轨迹

**失败聚类 (Loop 2 Stage 1)**:
- 计算每条失败轨迹的 t_EOD（最早偏离成功分布的时间点）
- 发现两个主要聚类:
  - K1 (17 条): 在"接近炉灶"阶段偏离 → m* = "approach_stove"
  - K2 (12 条): 在"放置"阶段偏离 → m* = "place_on_stove"

**因果诊断 (K1 的 medoid seed)**:
- 六层诊断遍历:
  - L_eval: ✅ 成功标准正确
  - L_crit: ❌ **根因** — 缺少碰撞检测 Critic，机械臂靠近炉灶时未检测到侧向偏移
  - → L* = L_crit

**Harness 修复 (Loop 2 Stage 2)**:
- 生成 C*: 侧向偏移检测函数（监测末端执行器与炉灶边缘的距离 < 5cm 时触发）
- 生成 R*: 侧向修正恢复策略（先后退 3cm，重新对准，再前进）
- 嵌入 VLA Re-entry Contract: Ψ(s_t) = 𝟙(偏移已清除) ∧ 𝟙(接触力稳定 > γ)

**验证 (Loop 3)**:
- 在 30 个 held-out seeds 上测试: 成功率从 38% → 67%
- 通过验证门，C* 和 R* 写入技能记忆

**Iteration 1 — 闭环执行**:
- 新 rollout 中，C* 在动作频率持续监测
- 检测到偏移 → Orchestrator 批准 → 触发 R* 修正 → 交还 VLA
- 结果: 34/50 成功 (68%)，较 Iteration 0 提升 26%

**Iteration 2-4 — 持续进化**:
- 继续发现新的失败模式（如抓取力不足、放置角度偏差）
- 每轮新增 2-3 个 Critic-Recovery 对
- 最终: 45/50 成功 (90%)

这个玩具例子的数字与论文报告的 LIBERO-Pro 34.5%→90.8% 的提升趋势一致。

## 4. 工程视角 (Engineering View)

| 维度 | 数值/约束 | 工程含义 |
|------|-----------|----------|
| 推理加速 | 11.1× 快于 RPent | Critic 是轻量代码函数（非 LLM），大部分时间 VLA 直接执行，仅失败时触发 LLM 仲裁 |
| Rollout 吞吐 | 1.7 → 35.1 episodes/min (20.6×) | Z-Infra 的 worker pool 解耦 + 批量推理是关键；没有这个吞吐，进化循环太慢 |
| Critic 执行频率 | 动作级（~10Hz） | 必须是非 LLM 的代码函数，否则成本不可承受 |
| Orchestrator 调用频率 | 事件触发（仅在 Critic 报警时） | 控制了 LLM 调用成本 |
| 进化迭代周期 | 取决于 rollout 预算 | 每轮需要完整跑完 Loop 1-3；论文未明确单轮耗时 |
| 模型分区 | Z-Infra 支持模型切分 | 大 VLA 可跨 GPU，提高并行度 |
| 异步调度 | 环境 worker 与模型 worker 独立 | 避免 GPU 空闲等待仿真器 |

**部署约束**:
- Critic 必须是确定性代码函数（不能是 LLM），否则无法在动作频率运行
- Orchestrator Agent 是 LLM，但调用频率远低于 Critic（仅事件触发）
- 真实机器人部署需要适配 Z-Infra 的硬件解耦层（当前仅仿真验证）

## 5. 数据与评测 (Data & Eval)

### 数据集

| 基准 | 描述 | 任务数 | 难度特点 |
|------|------|--------|----------|
| LIBERO-Pro | LIBERO 的困难变体 | 未明确（论文 Appendix B 有映射） | 长视野、复杂空间约束 |
| RoboCasa | 厨房场景的具身基准 | 未明确（论文 Appendix A 有映射） | 多步骤、物理交互密集 |

### 评测设置

- **开发集 (𝒟_dev)**: 用于进化循环中的失败画像和 harness 优化
- **Hold-out 测试集**: 严格隔离，仅用于 Loop 3 的泛化验证
- **确定性调度**: 同一批次 rollouts 在相同 GPU (4090)、相同模拟器版本下运行
- **有效性过滤**: 基础设施失败（网络波动/仿真崩溃）强制重跑，确保 𝒱 的统计分布不因非策略因素漂移

### 关键结果

| 指标 | LIBERO-Pro | RoboCasa | 来源 |
|------|-----------|----------|------|
| 初始基线 (纯 VLA) | 34.5% | 73.6% | §4.4 |
| 最终成功率 (Zetta) | **90.8%** | **93.6%** | §4.4 |
| vs 最佳基线提升 | SOTA | SOTA | §4.4 |
| "Aha Moment" 案例 | Wine Bowl: 15%→95% | Cream Cheese: 5%→90% | §4.2 |
| 零样本迁移 (PnP-Stove→其他) | N/A | +14-18% | §4.3 |
| 推理延迟降低 | 91% (11.1×) | 91% (11.1×) | vs RPent |
| Rollout 吞吐提升 | 20.6× (1.7→35.1 ep/min) | 同左 | §4.6 |

## 6. 能力与失败模式 (Capabilities & Failure Modes)

### 能做什么
- **持续同任务改进**: 在固定任务上，成功率随进化迭代持续提升（LIBERO-Pro 34.5%→90.8%）
- **零样本技能迁移**: 在 PnP-Stove 上学到的抓取/重抓/稳定放置技能，零样本迁移到 PnP-Sink/Cabinet/Toaster，提升 14-18%
- **"Aha Moments"**: 某些任务在早期迭代中成功率极低，一旦关键 Critic-Recovery 对被发现，成功率骤升（Wine Bottle in Bowl: 15%→95%）
- **加速推理**: 相比纯 LLM Agent（RPent），推理延迟降低 91%，因为 Critic 拦截了大量本需 LLM 介入的场景

### 不能做什么 / 局限
- **仅仿真验证**: 所有实验在 LIBERO-Pro 和 RoboCasa 仿真环境中完成，真实机器人部署效果未知
- **依赖 LLM 诊断**: Phase II 的因果诊断和修复由 LLM 驱动（𝒜_diag, 𝒜_repr, 𝒜_gen），成本较高且可能有幻觉
- **Rollout 预算限制**: 论文明确说"在当前 rollout 预算下"达到 SOTA，暗示仍有提升空间但受限于计算资源
- **单臂操作**: 实验均为单臂机器人场景，双臂/移动操作未覆盖
- **仿真到现实的 gap**: Z-Infra 设计了解耦架构但仅在仿真中验证，真实物理环境的传感器噪声、延迟、动力学不确定性未讨论

### 6.1 隐含假设 (Hidden Assumptions)

1. **失败模式可聚类**: 假设不同 seed 的失败可以按 t_EOD 聚类成有限个机制。如果失败模式高度连续分布（而非离散聚类），聚类效果可能下降。

2. **Critic 可代码化**: 假设所有关键的失败前兆都能用确定性代码函数检测。对于需要语义理解或复杂视觉推理的失败模式（如"物体放反了"），纯代码 Critic 可能不够。

3. **Orchestrator 仲裁可靠性**: 假设固定 𝒜_orch 能可靠判断是否批准 Critic 的干预提议。如果 Orchestrator 频繁误判（批准不该批准的 / 拒绝该批准的），闭环效果会大打折扣。

4. **仿真到现实的泛化**: 假设在仿真中学到的 Critic-Recovery 对可以零样本迁移到真实机器人。但仿真中的物理建模（接触力、摩擦、形变）与真实世界有 gap。

5. **VLA 策略足够强**: 假设冻结的 VLA 策略本身具备完成任务的基本能力，只是缺乏闭环修正。如果 VLA 策略本身在语义理解层面就错了，Harness 层无法修复。

## 7. 与相关工作对比 (Comparison)

| 系统 | 核心方法 | 闭环学习 | 进化方式 | 训练策略 | 评测基准 |
|------|---------|---------|---------|---------|---------|
| RT-2 (2023) | 端到端 VLA | ❌ 开环推理 | 离线训练 | 大规模演示数据 | Sim + Real |
| OpenVLA (2024) | 开源 VLA | ❌ 开环推理 | 离线微调 | Open X-Embodiment | Sim + Real |
| RoboAgent (2024) | LLM 编排工具 | ❌ 事后反思 | 手动设计技能 | 无需训练 | Sim |
| RPent (2025) | LLM Agent + 反思 | ❌ Episode 级反思 | 手动/半自动 | 无需训练 | Sim |
| SkillOpt (2025) | 代码空间 SGD | ⚠️ Episode 级 | SGD-like 代码优化 | 无需训练 | Sim |
| **Zetta ζ (2026)** | **闭环 Harness + 三层循环** | **✅ 动作级闭环** | **自动化进化** | **冻结策略** | **Sim (LIBERO-Pro, RoboCasa)** |

**面试 Tip**: 如果被问到"Zetta 和传统 VLA 微调有什么区别"，可以答：Zetta 不修改 VLA 策略参数（∇θ=0），而是进化一个外置的监控+恢复 Harness。这避免了过参数化修复破坏 VLA 的语义泛化能力，同时通过三层时间尺度闭环实现了真正的持续学习——不是 episode 结束后才反思，而是在物理执行过程中实时干预。

## 8. 精讀建議 (Reading Guide)

- **值得精讀原文的人**:
  1. 做具身 Agent 框架设计的研究者/工程师（§2 的三层循环架构是可直接复用的设计模式）
  2. 关注 VLA 后训练/持续学习的研究者（"冻结策略 + 进化 harness"是一个有前景的方向）
  3. 需要构建 Embodied Infrastructure 的团队（§3 Z-Infra 的 worker pool 解耦设计有工程参考价值）

- **建議章節路徑**:
  - 先读 §2.1（问题形式化）→ 理解 H={C,R,𝒯} 的架构
  - 再看 §2.4-2.5（因果诊断 + Harness 修复）→ 理解进化机制的核心
  - 然后 §4.2-4.3（Aha Moments + 零样本迁移）→ 理解关键发现
  - 可跳 §3（Z-Infra 基础设施细节，除非你做工程实现）

- **不值得精讀的理由**:
  - 如果你不做具身智能/机器人，只关心纯视觉-语言模型，这篇的 harness 层设计距离较远
  - 如果你已熟悉 SkillOpt / EmbodiSkill 等代码空间优化方法，§2.5 的修复机制不会有太多新信息
  - 所有实验在仿真环境完成，如果你关注真实机器人部署，这篇的直接参考价值有限


---
[← Back to Theory](./README.md)

**关键引用**:
- 论文: https://arxiv.org/abs/2608.16590
- 项目页: https://air-embodied-brain.github.io/zetta
- 机构: 清华大学 AIR (Institute for AI Industry Research) + Z-Trans AI
