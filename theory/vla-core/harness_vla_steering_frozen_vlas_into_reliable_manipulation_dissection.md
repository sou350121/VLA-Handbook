# Harness VLA：通过记忆引导代理将冻结 VLA 转化为可靠操作原语 (Harness VLA: Steering Frozen VLAs into Reliable Manipulation Primitives via Memory-Guided Agents)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-07-16
>
> **论文**: Harness VLA: Steering Frozen VLAs into Reliable Manipulation Primitives via Memory-Guided Agents
> **链接**: https://arxiv.org/abs/2607.08448
> **核心定位**: 不微调 VLA 权重，而是用 Agent 框架将其封装为可重试的接触操作原语，通过记忆引导的组合策略将冻结 VLA 的能力扩展到分布外扰动场景。

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | 将冻结 VLA 暴露为 `vla_act` 原语，配合固定解析原语库 + 双记忆模块（Task-Specific + Global），在分布外扰动下大幅提升操作成功率 |
| 適合精讀 | 如果你在做 VLA 部署/泛化、Agent 框架设计、或需要零微调扩展 VLA 能力 |
| 可以跳過 | 如果你只关心 VLA 预训练方法或纯策略学习 |
| 落地可行性 | 高 — 无需微调 VLA，只需在部署环境跑一次参考种子的探索引导 |
| 主要風險 | 引导阶段成本较高（需 LLM Planner + 环境交互），且依赖 benchmark 特定的成功判定器 |

💡 **X-Ray 开场**
VLA 模型在训练分布内表现优秀，但一旦遇到指令重定向、物体位置交换或布局变化就会大幅退化。Harness VLA 的核心发现是：不需要改 VLA 的权重，而是把它变成一个"可重试的接触专家"——用 Agent Planner 在外部做语义重定位、空间重组和非接触操作，只在需要精细接触时才调用 VLA。结果在 LIBERO-Pro 上比最强基线高出 38.6 个百分点。

📍 **研究全景时间线**
```
2024 Code as Policies (可编程策略) → 2024 ProgPrompt (程序化提示)
  → 2024-25 通用 VLA (RT-2, π0, OpenVLA) → 2025 RATS/Cap-X (VLA+Agent 混合)
    → 【本文 2026.07】Harness VLA — 固定原语库 + 双记忆 + 冻结 VLA 作为可重试原语
      ← 当前位置：从"扩展技能库"转向"学会使用固定技能"
```

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 组件 | 职责 | 输入 | 输出 | 训练/部署阶段 |
|------|------|------|------|---------------|
| **Agentic Planner ($\Pi$)** | 认知编排器，选择原语调用 | 任务描述 ℓ、RGB-D 观测 o_t、机器人状态 q_t、记忆检索结果 | JSON 原语调用 c_t | 部署时在线推理（Codex/CC） |
| **Primitive Library (𝒫)** | 统一操作接口，6 个解析原语 + 1 个 VLA 原语 | JSON 调用参数 | 执行到内部后置条件满足，返回新观测 | 固定不变（部署前确定） |
| **vla_act** | 冻结 VLA 封装的接触原语 | prompt + early-return 谓词 $\tau$ | 动作块序列直到 $\tau$ 满足或预算耗尽 | VLA 权重冻结，仅调用 |
| **Task-Specific Memory** | 存储参考种子探索的成功原语轨迹 | 探索阶段的 JSONL 轨迹 | 参数化后的成功轨迹（空间坐标→感知查询） | 引导阶段写入，部署阶段读取 |
| **Global Memory** | 存储跨任务可复用的成功规则和失败模型 | 探索过程中的启发式经验 | 成功规则 + 失败模型（如空抓取、误检测） | 引导阶段写入，部署阶段读取 |
| **Agentic Harness** | 运行时环境，序列化/执行/日志/检查 | Planner 决策 + 环境反馈 | 观测刷新、进度检查、预算控制 | 始终运行 |

### 1.2 关键机制 (Key Mechanism)

**为什么这样设计？**

1. **职责分离**：传统 VLA 试图在一个策略中吸收语言 grounding、长期组合和底层控制。Harness VLA 将语言/语义/空间推理交给 Planner，将接触精细控制留给 VLA，将非接触操作（移动、旋转、释放）交给解析原语。

2. **固定原语库而非扩展**：现有 coding agent 系统通常通过不断添加新技能来扩展能力。Harness VLA 反其道而行——保持原语库固定（6 解析 + 1 VLA），让 Planner 学会如何编排它们。这降低了部署复杂度，避免了技能膨胀。

3. **VLA 作为可重试原语**：传统 VLA 部署是开环的——一旦启动就执行到底。Harness VLA 将 VLA 封装为 `vla_act`，Planner 可以：布置有利姿态 → 调用 VLA → 检查结果 → 失败则重新布置 → 再调用。这种"尝试-检查-重试"循环大幅提升了接触操作的鲁棒性。

4. **双记忆架构**：
   - **Task-Specific Memory**：存储特定任务的成功轨迹骨架（参数化，非坐标硬编码）
   - **Global Memory**：存储跨任务通用的成功规则和失败模型

⚡ **Eureka Moment**：把 VLA 从"单体轨迹策略"降级为"局部接触专家原语"——Planner 负责在扰动空间中用解析原语导航到 VLA 的训练分布覆盖区域内，然后调用 VLA 完成接触操作。

### 1.3 信息流/架构图 (Flow / Diagram)

```
┌─────────────────────────────────────────────────────────────┐
│                    Agentic Planner (Π)                       │
│  输入: ℓ(任务) + o_t(RGB-D) + q_t(本体) + Memory 检索       │
│  输出: JSON 原语调用 c_t ∈ 𝒫                                 │
└──────────────┬──────────────────────────────┬────────────────┘
               │                              │
               ▼                              ▼
┌─────────────────────────┐    ┌──────────────────────────────┐
│   Analytic Primitives   │    │       vla_act (VLA)          │
│  move_to / move_pose    │    │  冻结 VLA f_θ               │
│  rotate_wrist / pitch   │    │  prompt + stop predicate τ  │
│  set_gripper / release  │    │  → 动作块序列               │
│  navigate_to / move_base│    │                              │
│  (非接触/空间操作)       │    │  (接触丰富操作)              │
└───────────┬─────────────┘    └──────────────┬───────────────┘
            │                                 │
            └────────────┬────────────────────┘
                         ▼
              ┌─────────────────────┐
              │   Physics Engine    │
              │  (MuJoCo/Robosuite) │
              │  执行到后置条件满足   │
              └─────────┬───────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │  o_{t+1}, q_{t+1}   │
              │  + 成功信号 𝒢       │
              └─────────┬───────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │   Memory Update     │
              │  Task-Specific:     │
              │    JSONL 轨迹       │
              │  Global:            │
              │    成功规则+失败模型 │
              └─────────────────────┘
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：

```
π_final = Planner( ℓ, o_t, q_t, Memory_task, Memory_global ; 𝒫_fixed )
```

**直觉**：最终策略 = Planner 在固定原语库上，基于任务描述、当前观测和记忆检索做出的原语选择。VLA 不再是策略本身，而是原语库中的一个可调用的黑盒函数。

**目标**：在部署扰动下（指令重定向、位置交换、布局变化），最大化任务成功率：

```
max_Π E[ 𝒢(τ) | ℓ, ε_perturbed ]
```

其中 $\tau$ 是由 $\Pi$ 在原语库 $\mathcal{P}$ 上编排产生的轨迹，$\mathcal{G}$ 是稀疏成功判定器。

**关键变量说明**：

| 符号 | 含义 |
|------|------|
| $\Pi$ | Agentic Planner（Codex 或 Claude Code） |
| 𝒫 | 固定原语库（6 解析 + 1 VLA） |
| $f_\theta$ | 冻结 VLA 策略（如 $\pi_{0.5\text{-SFT}}$） |
| o_t = (I_rgb, I_d, q_t) | t 时刻多模态观测 |
| ℓ | 自然语言任务描述 |
| 𝒢 | 二元完成判定器（仅 episode 结束时返回） |
| $\tau$ | early-return 谓词（控制 vla_act 调用时长） |
| Memory_task | 任务特定记忆（参数化 JSONL 轨迹） |
| Memory_global | 全局记忆（成功规则 + 失败模型） |

> 符号与本文保持一致。论文未给出显式的优化目标公式，上述公式为对框架的抽象建模。

## 3. 带数字走一遍：玩具例子 (Worked Example)

**场景**：LIBERO-Pro 中的 "place the red block on the blue plate"，但红色方块和蓝色盘子的位置与训练时不同（Position-Swap 扰动）。

**直接 VLA 部署**（无 Harness）：
- VLA 看到场景，但红色方块不在训练时的位置
- VLA 可能仍然朝向训练时的位置移动 → 抓取失败
- 由于是开环执行，失败后无法恢复
- 成功率：~50%（LIBERO-Pro 平均）

**Harness VLA 部署**（有 Memory 引导）：

```
Step 1: Planner 读取 Task-Specific Memory
  → 检索到参考轨迹: [move_to(plate_pos), vla_act("grasp block"), move_to(plate_pos), release]
  → 但轨迹中的坐标是符号化的: [move_to(感知查询:蓝色盘子), vla_act("抓取红色方块"), ...]

Step 2: Planner 从当前 RGB-D 观测中重新 grounding
  → 检测到蓝色盘子在 (0.3, -0.15, 0.45)（参考位置是 0.3, 0.15, 0.45 — 位置变了！）
  → 检测到红色方块在 (-0.2, 0.2, 0.5)（参考位置是 -0.2, -0.2, 0.5 — 位置也变了！）

Step 3: Planner 编排原语调用
  c_1 = {action: "move_to", xyz: [0.3, -0.15, 0.45]}  → 先移动到盘子位置确认
  c_2 = {action: "move_to", xyz: [-0.15, 0.2, 0.55]}  → 移动到方块上方
  c_3 = {action: "vla_act", prompt: "grasp red block", max_chunks: 10, stop: "grasped"}
  → VLA 执行抓取。假设第一次失败（空抓取）

Step 4: Planner 诊断 + 重试
  → Global Memory 中有失败模型: "empty_grasp → 调整预接触姿态"
  → c_4 = {action: "rotate_wrist", target_yaw: 15°}  → 调整手腕角度
  → c_5 = {action: "vla_act", prompt: "grasp red block", max_chunks: 10, stop: "grasped"}
  → VLA 第二次抓取成功！

Step 6: 放置
  → c_6 = {action: "move_to", xyz: [0.3, -0.15, 0.5]}
  → c_7 = {action: "release"}
  → 𝒢 = True → 成功！
```

**关键数字**：
- 直接 VLA：1 次开环执行，失败即结束
- Harness VLA：最多可重试多次 vla_act 调用，每次失败后 Planner 重新定位 + 调整姿态
- 论文 Figure 1 显示：仅需 3-4 次 VLA 调用预算即可达到接近最大性能

## 4. 工程视角 (Engineering View)

| 维度 | 数值/约束 | 工程含义 |
|------|-----------|----------|
| **Planner 推理延迟** | 每次原语调用需 LLM 推理（Codex/CC） | 每步约数秒级延迟，不适合高频控制循环 |
| **VLA 调用次数** | 每 episode 最多 N 次（论文中未明确上限，但实验显示 3-4 次即可） | 相比全轨迹 VLA 推理，调用次数大幅减少 |
| **引导阶段成本** | 每个任务需 1 个参考种子的自主探索 | 一次性成本，但需要 LLM Planner 与环境多轮交互 |
| **部署阶段预算** | 严格步数限制（相比引导阶段大幅缩短） | 适合生产环境部署 |
| **内存占用** | Task-Specific Memory（每任务一个 JSONL 文件）+ Global Memory（共享规则） | 极小，可忽略 |
| **VLA 权重** | 完全冻结，无需任何微调 | 部署时无需 GPU 训练资源，仅需推理 |
| **原语库大小** | 固定 6-8 个原语 | 不随任务数量增长，避免了技能膨胀 |
| **跨平台迁移** | 同一原语接口适配不同机器人（单臂/双臂/移动基座） | 只需更换 VLA 后端 + 添加平台特定原语 |

**Trade-off 分析**：
- **优势**：无需微调 VLA、原语库固定、部署简单、可解释性强（Planner 决策可追溯）
- **代价**：引导阶段需要 LLM 驱动的环境探索、每步有 LLM 推理延迟、依赖 benchmark 特定的成功判定器

## 5. 数据与评测 (Data & Eval)

### 数据集与基准

| 基准 | 类型 | 任务数 | 扰动类型 | 种子数 |
|------|------|--------|----------|--------|
| **LIBERO** (标准) | 桌面操作 | 4 个子集 (Object/Spatial/Goal/LIBERO-10) | 无（分布内） | 10 seeds/task |
| **LIBERO-Pro** | 桌面操作（扰动） | Spatial/Object/Goal/LIBERO-10 | 指令重定向(T) + 位置交换(S) | 10 seeds/task |
| **RoboCasa365** | 厨房操作 | Atomic-Seen / Composite-Seen / Composite-Unseen | 移动基座 + 铰链物体 + 长程组合 | 5-10 seeds |
| **RoboTwin C2R** | 双臂操作 | Clean-to-Randomized 迁移 | 清洁→随机化（零样本） | 未明确 |

### VLA 后端配置

| 基准 | VLA 后端 | 来源 |
|------|----------|------|
| LIBERO / LIBERO-Pro | $\pi_{0.5\text{-SFT}}$ ($\pi_{\text{RLinf}}$) | RLinf 发布 |
| RoboCasa365 | RLDX-1 RoboCasa checkpoint | 论文 [30] |
| RoboTwin C2R | LingBot-VLA | 作者后训练 |

### 评测模式

- **Few-shot**：在 1 个参考种子上引导，存储记忆，然后在未见种子/扰动上评估
- **Zero-shot**：不使用目标设置的任务特定记忆，仅依赖 Planner 在线推理

## 6. 能力与失败模式 (Capabilities & Failure Modes)

### 能做什么

| 能力 | 场景 | 证据 |
|------|------|------|
| 指令重定向鲁棒性 | 任务目标改变但操作类似 | LIBERO-Pro Goal-T: 87%（few-shot）/ 79%（zero-shot） |
| 位置交换鲁棒性 | 物体位置与训练时不同 | LIBERO-Pro Goal-S: 87%（few-shot）/ 31%（zero-shot） |
| 厨房长程操作 | 移动基座 + 铰链物体 + 多步骤 | RoboCasa365: 55.4%（Codex）vs 30.0%（RLDX-1 基线） |
| 双臂零样本迁移 | 清洁→随机化 | RoboTwin C2R: 58.4% vs 50.4%（直接 VLA） |
| 分布内性能保持 | 标准 LIBERO | 96.0%（与冻结 VLA 的 95.3% 相当） |

### 不能做什么 / 失败模式

| 失败模式 | 原因 | 缓解 |
|----------|------|------|
| Zero-shot 位置交换性能低 | 缺乏任务特定记忆时，Planner 难以仅靠在线推理完成空间重定位 | LIBERO-Pro Goal-S zero-shot 仅 31% vs few-shot 87% |
| 引导阶段成本高 | 每个任务需要 LLM Planner 自主探索找到成功轨迹 | 一次性成本，但新任务仍需引导 |
| 依赖成功判定器 | 需要 benchmark 特定的 𝒢 来检测完成 | 真实环境中成功判定可能不可靠 |
| 接触失败重试有限 | 多次 vla_act 调用仍可能失败 | 受步数预算限制 |

### 6.1 隐含假设 (Hidden Assumptions)

1. **成功判定器可用**：论文假设 𝒢 在 episode 结束时可靠返回。但在真实世界中，成功判定可能需要额外的视觉检测模块，且可能出错。

2. **参考种子代表性**：引导阶段仅使用 1 个参考种子。如果参考种子的布局过于特殊，学到的轨迹骨架可能无法泛化到差异较大的扰动。

3. **Planner 推理能力**：假设 Codex/CC 能够可靠地从 RGB-D 观测中提取空间信息并 grounding 符号化轨迹。这对多模态推理能力有较高要求。

4. **原语可靠性**：假设解析原语（move_to、rotate_wrist 等）在目标环境中可靠执行。如果物理引擎或真实机器人的运动学求解器不稳定，这些原语也可能失败。

5. **VLA 后端兼容性**：假设冻结 VLA 在局部接触区域内仍有足够的能力。如果扰动导致接触区域完全超出 VLA 训练分布，vla_act 本身也会失败。

## 7. 与相关工作对比 (Comparison)

| 方法 | 核心思路 | VLA 角色 | 是否需要微调 | 扰动鲁棒性 | 适用场景 |
|------|----------|----------|-------------|-----------|----------|
| **直接 VLA** ($\pi_{0.5}$, RT-2) | 端到端语言-动作映射 | 单体策略 | 预训练 | 低（分布外退化严重） | 分布内操作 |
| **Code as Policies** | LLM 合成执行程序 | 无 VLA | 否 | 中（依赖程序正确性） | 结构化任务 |
| **ProgPrompt** | 程序化提示编排 API | 无 VLA | 否 | 中 | 已有 API 的任务 |
| **RATS** | VLA + 推理时间搜索 | VLA 作为策略 | 否 | 中（43.8% LIBERO-Pro） | 需要搜索的任务 |
| **Cap-X** | VLA + 编码代理 | VLA 作为策略 | 否 | 中（低 zero-shot） | 多扰动场景 |
| **Harness VLA** | 固定原语 + 双记忆 + VLA 作为原语 | VLA 作为可重试原语 | 否（仅需引导） | 高（82.4% LIBERO-Pro） | 分布外扰动部署 |

**面试 Tip**：当被问到"VLA 如何泛化到训练分布外"时，可以回答："Harness VLA 提供了一种不需要微调的解决方案——核心思想是将 VLA 从单体策略降级为可重试的接触原语，用 Agent Planner 在外部处理语义和空间扰动，只在 VLA 训练分布覆盖的局部区域内调用它。这种方法在 LIBERO-Pro 上比最强基线高出 38.6 个百分点，且保持了分布内性能。"

## 8. 精讀建議 (Reading Guide)

**值得精讀原文的人**：
1. 做 VLA 部署和泛化的研究者——本文提供了无需微调即可扩展 VLA 能力的完整框架
2. 设计 Agent 框架的工程师——固定原语库 + 双记忆的架构模式可复用到其他领域
3. 评估 VLA 在真实机器人上可行性的团队——本文证明了冻结 VLA + 智能编排可以达到实用级别的成功率

**建議章節路徑**：
- 先讀 §2（框架设计）→ 理解原语库、记忆模块和两阶段工作流
- 再看 §3.2（基准性能）→ 了解各场景下的具体数字和对比
- 可跳 §3.3（机制分析）→ 如果只关心应用，可略过深入分析

**不值得精讀的理由**：
- 如果你不做机器人操作部署，只关心 VLA 预训练方法
- 如果你已熟悉 Code as Policies 类框架，本文的创新主要在 VLA 集成方式而非 Agent 架构本身

---
[← Back to Theory](./README.md)
