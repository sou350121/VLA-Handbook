# Harness VLA：通过记忆引导代理将冻结 VLA 转化为可靠操作原语 (Harness VLA: Steering Frozen VLAs into Reliable Manipulation Primitives via Memory-Guided Agents)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-07-16
>
> **论文**: Harness VLA: Steering Frozen VLAs into Reliable Manipulation Primitives via Memory-Guided Agents
> **链接**: https://arxiv.org/abs/2607.08448
> **核心定位**: 将冻结 VLA 封装为可重试的"接触操作原语"，通过记忆引导的 Agent 编排器组合少量确定性分析原语，在不微调 VLA 的前提下将其能力扩展到分布外扰动场景。

## ⚡ 快速判斷

| 維度 | 判斷 |
|------|------|
| 核心結論 | 冻结 VLA + 记忆引导 Agent 编排 = 在分布外扰动下显著超越端到端 VLA 和纯 LLM 编程 Agent |
| 適合精讀 | 如果你在探索 VLA 部署策略、Agent 编排框架、或少样本迁移方法，重点看 §2（框架设计）和 §3.2（基准对比） |
| 可以跳過 | 如果你只关心 VLA 预训练数据/架构本身，这篇聚焦的是部署层编排而非训练 |
| 落地可行性 | 中（需要 LLM planner + 固定原语库 + 单样本探索阶段；仿真环境成熟，真机部署待验证） |
| 主要風險 | 探索阶段成本高昂（每个任务需单样本 bootstrap）；Zero-shot 在空间扰动下性能骤降 |

💡 **X-Ray 开场**
这篇论文解决的核心问题是：端到端 VLA 在训练分布内表现优秀，但遇到语义重定向、目标重绑定、空间布局偏移等部署扰动时性能急剧下降。作者发现，与其让 VLA 承担所有责任（语言理解+长期规划+底层控制），不如把它降级为一个"接触操作专家"——只在需要物理接触的阶段调用，其余非接触任务（移动、定位、释放）交给确定性分析原语。对一个聪明的非专家来说，核心 takeaway 是：把 VLA 从"全能司机"变成"换挡技师"，由 Agent 编排器决定何时换挡。

📍 **研究全景时间线**
```
[2024] 通用 VLA (RT-2, OpenVLA) → 端到端策略学习接触丰富控制
       ↓
[2024-25] Code as Policies / ProgPrompt → LLM 编程 Agent 组合感知-控制 API
       ↓
[2025] RATS / Cap-X → 尝试结合 VLA 与 Agent 推理
       ↓
[2026-07] Harness VLA ← 当前位置：固定原语库 + 记忆引导 + VLA 作为可重试原语
       → 局限：探索阶段成本高；零样本空间扰动性能不足
```

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 维度 | 端到端 VLA (π₀.₅等) | LLM 编程 Agent | Harness VLA |
|------|----------------------|----------------|-------------|
| 语言理解 | 内嵌在策略中（弱） | 核心能力（强） | Planner 承担（强） |
| 长期规划 | 无（单轨迹执行） | 核心能力 | Planner 承担 |
| 底层控制 | 直接输出动作 | 通过 API 调用（脆弱） | 原语库封装 |
| 接触操作 | 强项 | 弱项（不规则抓取等） | VLA 原语（强） |
| 非接触操作 | 弱（分布外） | 强（确定性） | 分析原语（强） |
| 部署扰动鲁棒性 | 差 | 中（依赖 API 质量） | 好（记忆引导+重 staging） |
| 微调需求 | 训练时完成 | 无 | 无（VLA 冻结） |
| 探索阶段 | 不需要 | 不需要 | 需要（单样本 bootstrap） |

### 1.2 关键机制 (Key Mechanism)

Harness VLA 的核心设计哲学是**职责分离**：

- **Agentic Planner（编排器）**：负责语义重绑定、空间重定位、非接触执行、VLA 重 staging。使用 Codex 或 Claude Code 作为 LLM 后端。
- **固定原语库（Primitive Library）**：6 个确定性分析原语 + 1 个 VLA 原语。部署时不可扩展。
  - 分析原语：`move_to`、`move_pose`、`rotate_wrist`、`rotate_pitch`、`set_gripper`、`release`
  - VLA 原语：`vla_act` — 接收 prompt + early-return 谓词，冻结 VLA 执行局部接触操作直到谓词满足或 chunk 预算耗尽
- **双层记忆系统**：
  - **Task Specific Memory**：存储单样本探索得到的成功原语调用序列（JSONL 格式，空间坐标参数化为感知查询）
  - **Global Memory**：存储可跨任务复用的成功规则和失败模型

⚡ **Eureka Moment**：不扩展技能库，而是教会 Planner 如何用好一个固定且极小的原语库——VLA 不是策略，而是原语。

### 1.3 信息流/架构图 (Flow / Diagram)

```
┌─────────────────────────────────────────────────────────────┐
│                    Agentic Planner (Π)                       │
│  输入: 任务描述 ℓ + 多模态观察 o_t + 记忆上下文              │
│  输出: JSON 原语调用 c_t ∈ P                                │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
               ▼                              ▼
   ┌───────────────────┐         ┌─────────────────────┐
   │  Task Specific    │         │    Global Memory     │
   │  Memory           │         │  成功规则 + 失败模型  │
   │  成功原语序列      │         └─────────────────────┘
   │  (JSONL, 参数化)   │
   └───────────────────┘
               │
               ▼
   ┌───────────────────────────────────┐
   │     固定原语库 P                   │
   │  ┌────────────┐  ┌──────────────┐ │
   │  │ 分析原语    │  │  VLA 原语    │ │
   │  │ move_to    │  │ vla_act      │ │
   │  │ rotate_*   │  │ (冻结 f_θ)   │ │
   │  │ set_gripper│  │ prompt+stop  │ │
   │  │ release    │  └──────────────┘ │
   │  └────────────┘                   │
   └───────────┬───────────────────────┘
               │
               ▼
   ┌───────────────────────────────────┐
   │    Physics Engine (MuJoCo)         │
   │    执行原语 → 返回 o_{t+1}, q_{t+1} │
   └───────────────────────────────────┘
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：
```
π* = argmax_π E[ G | π 编排 P, f_θ 冻结, M_task + M_global ]
```

**目标**：在 VLA 参数 θ 冻结、原语库 P 固定的约束下，学习一个编排策略 π 使得任务完成概率最大化。

**变量说明**：
- `π`：Agentic Planner（LLM），负责从原语库 P 中选择原语并绑定参数
- `P`：固定原语库，|P| = 6（分析）+ 1（VLA）
- `f_θ`：冻结的 VLA 策略，仅在 `vla_act` 原语中被调用
- `M_task`：Task Specific Memory，单样本探索得到的成功轨迹
- `M_global`：Global Memory，跨任务通用的成功规则与失败模型
- `G`：任务完成谓词（episode 结束时返回的二值信号）

**直觉**：传统 VLA 试图用一个策略网络同时学习"做什么"和"怎么做"。Harness VLA 将这两个问题解耦——Planner 学"做什么"（原语编排），VLA 只负责"怎么做"（接触操作）。记忆系统 M_task 提供结构先验，M_global 提供启发式约束，两者共同缩小 Planner 的搜索空间。

> 符号与本文保持一致：ℓ = 任务描述，o_t = 多模态观察 (I_rgb, I_depth, q)，q_t = 机器人本体状态（末端执行器位姿 + 夹爪状态），τ = early-return 谓词。

## 3. 带数字走一遍：玩具例子 (Worked Example)

假设任务：**"把红色杯子放到蓝色盘子里"**，但部署时杯子位置相对训练时偏移了 15cm。

**端到端 VLA 的失败路径**：
```
t=0: VLA 接收观察 → 输出抓取动作 → 抓到空处（杯子不在训练位置）
t=1: VLA 重复训练行为 → 仍然抓空
...
t=T: 失败（VLA 不知道如何 re-stage）
```

**Harness VLA 的成功路径**：
```
t=0: Planner 读 RGB 观察 → 检测到杯子新位置
     → 调用 move_to(x=杯子新位置)  ← 分析原语，确定性移动
t=1: 到达后调用 vla_act(prompt="抓取红色杯子", max_chunks=10, stop="夹爪接触")
     → VLA 执行局部抓取操作（接触丰富区域，VLA 强项）
     → 夹爪接触谓词满足，提前返回
t=2: Planner 调用 move_to(x=蓝色盘子位置)  ← 分析原语，非接触运输
t=3: Planner 调用 release()  ← 分析原语，释放
结果：成功 ✅
```

**关键数字**：
- VLA 调用次数：1 次（仅在 t=1 的接触阶段）
- 分析原语调用：3 次（move_to × 2, release × 1）
- 总编排步数：4 步（远少于 VLA 从头到尾执行所需的数十步）
- 空间偏移容忍度：由分析原语的感知定位能力决定（不依赖 VLA 的分布内假设）

## 4. 工程视角 (Engineering View)

| 工程维度 | 分析 |
|----------|------|
| **Planner 推理延迟** | 每个 execution turn 需要一次 LLM 调用（Codex/CC）；LIBERO-Pro 上每任务约 5-15 个 turn → 单任务延迟 ~10-60s（取决于 LLM API 速度） |
| **VLA 推理延迟** | vla_act 仅在接触阶段调用，每次 max_chunks 限制（通常 5-10 chunks）；相比端到端 VLA 全轨迹执行，VLA 推理调用次数减少 60-80% |
| **探索阶段成本** | 每个新任务需要单样本 bootstrap（Planner 自主探索直到成功）；这是最大的时间开销，但只需一次 |
| **内存占用** | VLA 冻结（无梯度）+ LLM Planner（API 调用，无本地模型）→ 部署端显存需求低 |
| **Step Budget** | 部署阶段严格限制步数；探索阶段宽松 |
| **可移植性** | 原语库固定 → 迁移到新机器人只需实现相同的 JSON 原语接口，无需重新训练 VLA |
| **失败恢复** | Planner 可检测 VLA 失败并 re-stage（重新定位后再次调用 vla_act）；端到端 VLA 失败即整个轨迹失败 |

**工程含义**：Harness VLA 的架构本质上是一个**分层控制系统**——高层 Planner 以低频率（每个 turn 调用一次 LLM）做决策，低层原语以高频率执行物理动作。这种设计天然适合控制频率分层：Planner 在 0.1-1 Hz 运行，原语内部控制器在 10-100 Hz 运行。

## 5. 数据与评测 (Data & Eval)

### 基准套件

| 基准 | 类型 | 扰动类型 | 评估设置 |
|------|------|----------|----------|
| LIBERO | 桌面操作（标准） | 无（分布内） | 10 任务 × 10 种子 = 100 次/套件 |
| LIBERO-Pro | 桌面操作（扰动） | 指令重定向(T) / 位置交换(S) | 10 任务 × 10 种子 = 100 次/单元格 |
| RoboCasa365 | 厨房操作 | 移动 staging + 铰链器具 | 10 种子(Atomic-Seen) / 5 种子(Composite) |
| RoboTwin C2R | 双臂操作 | Clean→Randomized 零样本迁移 | 50 任务 × 5 随机种子 |

### VLA 后端配置

| 基准 | VLA 后端 | 来源 |
|------|----------|------|
| LIBERO / LIBERO-Pro | π₀.₅-SFT (RLinf 发布) | π_RLinf |
| RoboCasa365 | RLDX-1 RoboCasa | 冻结 checkpoint |
| RoboTwin C2R | LingBot-VLA | 作者后训练 checkpoint |

### Few-shot vs Zero-shot

- **Few-shot**：单样本 bootstrap → 写入 Task Specific Memory → 在新扰动上检索记忆重 grounding
- **Zero-shot**：不检索目标设置记忆，纯靠 Planner 在线推理 + Global Memory

## 6. 能力与失败模式 (Capabilities & Failure Modes)

### 能做什么

| 场景 | 能力 | 原因 |
|------|------|------|
| 指令重定向（语义变化） | 强（Goal-T: 87%） | Planner 语义理解不受空间分布影响 |
| 位置交换（空间变化） | 中-强（Goal-S: 87%，有记忆时） | 分析原语重新定位 + VLA 在接触区域工作 |
| 不规则抓取 | 强 | VLA 作为接触原语处理 |
| 长程复合任务 | 强 | Planner 编排多原语序列 |
| 厨房移动操作 | 强（55.4% vs 30% 基线） | 新增移动原语 + 相同编排逻辑 |

### 不能做什么

| 场景 | 问题 | 原因 |
|------|------|------|
| 零样本 + 空间扰动 | 弱（Goal-S: 31%） | 无 Task Specific Memory 时 Planner 难以发现正确的原语编排顺序 |
| 全新操作类型 | 未知 | 原语库固定，不支持部署时扩展 |
| 真机部署 | 未验证 | 所有实验在仿真器（MuJoCo）中进行 |
| 探索阶段失败 | 无回退 | bootstrap 失败则无记忆可用，任务无法执行 |

### 6.1 隐含假设 (Hidden Assumptions)

1. **单样本足够**：每个任务只需一个参考实例进行 bootstrap。对于复杂任务（如 RoboCasa365 的 Composite-Unseen），单样本可能不足以覆盖所有变体。
2. **原语库完备性**：6+1 个原语足以覆盖所有操作子问题。如果新场景需要原语库中没有的操作（如"拧螺丝"），系统无法处理。
3. **仿真到现实的 gap 可忽略**：所有实验在 MuJoCo 中运行。真机中的传感器噪声、动力学不确定性可能显著降低分析原语的精度。
4. **Planner 可靠性**：假设 LLM Planner 在部署阶段能可靠地将参数化轨迹 re-ground 到当前观察。LLM 的幻觉或推理错误可能导致错误的原语参数绑定。
5. **成功谓词可检测**：假设 `stop` 谓词（如"夹爪接触"）能可靠检测。误检测会导致 vla_act 过早返回或过度执行。

## 7. 与相关工作对比 (Comparison)

| 方法 | 核心思路 | VLA 角色 | 记忆 | 微调 | LIBERO-Pro 整体 |
|------|----------|----------|------|------|-----------------|
| π_RLinf (直接 VLA) | 端到端策略 | 全部 | 无 | 训练时 | 50.0% |
| RATS | VLA + 推理 | 全部 + 推理增强 | 无 | 训练时 | 43.8% |
| Cap-X | Agent + VLA 回调 | Agent 调用 VLA | 无 | 无 | ~33.6% |
| **Harness VLA (CC)** | **Agent 编排固定原语** | **仅接触原语** | **双层记忆** | **无** | **82.4%** |

**关键差异**：
- RATS 在 VLA 内部添加推理模块，但仍让 VLA 承担全局控制责任
- Cap-X 让 Agent 调用 VLA，但没有记忆引导和固定原语库的概念
- Harness VLA 首次将 VLA 完全降级为原语，通过记忆引导的编排实现分布外泛化

💡 **面试 Tip**：当被问到"Harness VLA 和传统 VLA 的区别"时，回答："传统 VLA 把语言理解、长期规划和底层控制全部塞进一个策略网络；Harness VLA 把 VLA 降级为'接触操作原语'，由 Agent 编排器负责其余所有职责。核心创新不是新的 VLA 架构，而是新的职责分工。"

## 8. 精讀建議 (Reading Guide)

**值得精讀原文的人**：
- 探索 VLA 部署策略的研究者（特别是分布外泛化问题）
- 构建 Agent 编排框架的工程师（原语库设计、记忆系统架构）
- 评估少样本迁移到机器人平台可行性的团队

**建議章節路徑**：
1. 先读 §2.2（Harness VLA 架构）→ 理解两阶段生命周期（探索/部署）
2. 再看 §2.3（原语接口）→ 理解固定原语库的设计哲学
3. 然后读 §3.2（基准性能）→ 量化评估收益
4. 可跳 §3.3（机制分析）的附录细节，除非你关心消融实验

**不值得精讀的理由**：
- 如果你不做机器人操作/具身智能，这篇的 engineering 细节可能过于专业
- 如果你已熟悉 Code as Policies 和编程 Agent 框架，核心概念（Agent 编排 + 工具调用）不会让你惊讶
- 如果你关心 VLA 预训练（数据、架构、损失函数），这篇完全不在那个层面

---
[← Back to Theory](./README.md)

**关键引用**：
- 论文: https://arxiv.org/abs/2607.08448
- 项目页: https://harnessvla.github.io/
- VLA 基线 π_RLinf: https://github.com/RLinf-Foundation/RLinf
- LIBERO: https://libero-project.github.io/
- RoboCasa365: https://robocasa365.github.io/
- RoboTwin: https://github.com/robottwin
