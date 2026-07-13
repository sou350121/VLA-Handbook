# Harness VLA: 将冻结 VLA 转化为可靠操作原语（Harness VLA: Steering Frozen VLAs into Reliable Manipulation Primitives via Memory-Guided Agents）

> ⚙️ 本文由 Moltbot 自动生成 | 2026-07-13
>
> **论文**: Harness VLA: Steering Frozen VLAs into Reliable Manipulation Primitives via Memory-Guided Agents
> **链接**: https://arxiv.org/abs/2607.08448
> **核心定位**: 不微调 VLA，而是把冻结 VLA 封装为可重试的"接触操作原语"，通过 LLM Planner + 双记忆模块将其组合到长程、扰动任务中

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | 冻结 VLA + 固定原语库 + 记忆引导的 Agentic Planner 可在无需微调的情况下，将 VLA 的操作范围扩展到分布外扰动任务 |
| 適合精讀 | 如果你在做 VLA 部署优化、长程任务编排、或 LLM-Agent + 机器人控制交叉方向 |
| 可以跳過 | 如果你只关心 VLA 预训练/微调方法本身，这篇是部署层工作 |
| 落地可行性 | 中（需要 LLM Planner 的 API 调用开销；仿真环境验证充分，真机待验证） |
| 主要風險 | 依赖 LLM Planner 的推理质量；两阶段工作流（Bootstrapping → Deployment）需要每任务一次探索 |

💡 **X-Ray 开场**
端到端 VLA 在训练分布内表现强劲，但一旦遇到语义重定向、目标重绑定或空间布局偏移就会失效。Harness VLA 的核心洞察是：不要试图让 VLA 变"聪明"，而是把它降级为一个"接触操作专家"——由上层 Planner 负责语义理解和任务编排，VLA 只负责局部接触丰富的操作阶段。这种职责分离让冻结 VLA 在无需微调的情况下，在 LIBERO-Pro 上比最强基线高出 38.6 个百分点。

📍 **研究全景时间线**
```
2023 Code as Policies (ProgPrompt)    2024 OpenVLA/RT-2    2025 π0/FlowVLA    2026 Harness VLA
       ↓ 用 LLM 生成程序控制机器人           ↓ 端到端 VLA 崛起      ↓ 流匹配动作生成      ← 当前位置
       ↓ LLM 负责一切，原语库不断膨胀        ↓ VLA 负责一切       ↓ 动作分布建模更精细   ↓ 冻结 VLA 作为原语
       ↓ 分析原语无法处理复杂接触             ↓ 分布外泛化差       ↓ 部署扰动下仍脆弱     ↓ Planner 负责编排
```

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 维度 | 端到端 VLA (RLinf/π0) | LLM Coding Agent (Code as Policies) | Harness VLA |
|------|----------------------|-------------------------------------|-------------|
| **语言理解** | VLA 内部隐式学习（弱） | LLM 显式推理（强） | LLM Planner 显式推理（强） |
| **长程编排** | VLA 自身（弱，无记忆） | LLM 生成完整程序 | Planner 逐 step 调用原语 + 双记忆 |
| **接触操作** | VLA 直接输出动作（强） | 分析原语/IK（弱） | vla_act 原语调用冻结 VLA（强） |
| **分布外泛化** | 差（重复训练行为） | 中等（依赖原语覆盖） | 好（Planner 重接地 + 重试） |
| **部署时扩展** | 需要重新训练/微调 | 需要扩展原语库 | 无需扩展，固定原语库 |
| **失败恢复** | 无（单程滚动） | 程序级 try-catch | Planner 重 staging + 重试 vla_act |
| **计算开销** | 单步推理 | LLM 推理 + 原语执行 | LLM 推理 + VLA 推理（交替） |

### 1.2 关键机制 (Key Mechanism)

Harness VLA 的核心设计围绕三个组件展开：

**组件 1: 固定原语库 (Fixed Primitive Library)**
- 6 个分析原语：move_to, move_pose, rotate_wrist, rotate_pitch, set_gripper, release
- 1 个 VLA 原语：vla_act（封装冻结 VLA）
- RoboCasa365 额外 2 个移动基座原语：navigate_to, move_base
- **关键约束**：部署时不可扩展原语库

**组件 2: 双记忆模块 (Dual Memory)**
- **Task Specific Memory**: 存储单任务参考 seed 探索成功的 JSONL 轨迹（参数化坐标 → 符号感知查询）
- **Global Memory**: 跨任务通用的成功规则（如最优 prompt 策略）和失败模型（如空抓取检测、误成功检测）

**组件 3: Agentic Planner (Π)**
- REPL 风格的执行循环：Observe → Retrieve → Compose → Verify & Retry
- 每步输出一个 JSON 格式的原语调用
- 从 RGB-D 图像、任务描述、记忆模块获取上下文

⚡ **Eureka Moment**: VLA 不应该被当作一个"全能策略"来使用——把它降级为一个可重试的"接触操作函数"，由上层 Planner 负责语义理解和任务编排，冻结 VLA 就能在不微调的情况下扩展到分布外任务。

### 1.3 信息流/架构图 (Flow / Diagram)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Agentic Planner (Π)                          │
│                                                                 │
│  输入: 任务描述 ℓ + RGB-D 观察 o_t + 机器人状态 q_t             │
│       + Task Specific Memory (JSONL 轨迹)                       │
│       + Global Memory (成功规则 + 失败模型)                      │
│                                                                 │
│                    ┌────────┐                                   │
│  ┌─────────────┤ Observe  ├──────────────┐                     │
│  │              └────────┘               │                     │
│  ▼                                       ▼                     │
│  ┌────────────┐                    ┌──────────┐               │
│  │  Retrieve  │ ──→ 记忆上下文      │  Compose  │              │
│  └────────────┘                    └────┬─────┘               │
│                                        │                      │
│                              JSON 原语调用                     │
│                              ┌─────────┴──────────┐          │
│                              ▼                    ▼          │
│                    ┌─────────────────┐   ┌──────────────┐    │
│                    │  Analytic       │   │   vla_act    │    │
│                    │  Primitives     │   │  (Frozen VLA)│    │
│                    │  move_to, etc.  │   │   f_θ        │    │
│                    └────────┬────────┘   └──────┬───────┘    │
│                             │                   │            │
│                    ┌────────┴───────────────────┘            │
│                    ▼                                         │
│              Physics Engine (MuJoCo)                         │
│                    │                                         │
│              o_{t+1}, q_{t+1}                                │
│                    │                                         │
│              ┌─────┴──────┐                                  │
│              │  Verify &   │──失败──→ Re-stage + Retry       │
│              │  Terminate  │──成功──→ 下一原语 / 任务完成    │
│              └────────────┘                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：

```
Π(o_t, ℓ, M_task, M_global) → c_t ∈ P
    where c_t = vla_act(prompt, max_chunks, τ)  (接触阶段)
          ∪ analytic_primitives (非接触阶段)
```

**目标**: 在无需微调 VLA 参数 θ 的前提下，通过 Planner 的语义重接地和原语组合，使任务成功率在分布外扰动下最大化。

**执行循环**:

```
给定: 环境 E, 冻结 VLA f_θ, 原语库 P, 任务描述 ℓ
初始化: o_0 = (I_rgb, I_d, q_0), 记忆 M_task, M_global

for t = 0, 1, 2, ...:
    c_t = Π(o_t, ℓ, M_task, M_global)    // Planner 选择原语
    (o_{t+1}, q_{t+1}) = E.execute(c_t)   // 执行原语直到后条件满足
    if G(o_{t+1}) == True: return SUCCESS  // 目标谓词满足
    if t >= T_max: return FAILURE          // 步数预算耗尽
```

**vla_act 原语的内部执行**:

```
vla_act(prompt, max_chunks, τ):
    for k = 1 to max_chunks:
        a_k = f_θ(prompt, o_current)       // VLA 输出动作块
        o_current = E.step(a_k)            // 执行动作块
        if τ(o_current) == True: break     // 早停谓词满足
    return o_current
```

> 符号说明：
> - Π: Agentic Planner（LLM-based，支持 Codex 和 Claude Code 两种后端）
> - f_θ: 冻结的 VLA 策略（如 π0.5-SFT, RLDX-1, LingBot-VLA）
> - P: 固定原语库（8-9 个原语）
> - τ: 早停谓词（如"已抓住物体"、"接触力超过阈值"）
> - G: 任务完成谓词（稀疏奖励，仅在 episode 结束时可观测）
> - M_task: Task Specific Memory（JSONL 轨迹）
> - M_global: Global Memory（成功规则 + 失败模型）

## 3. 带数字走一遍：玩具例子 (Worked Example)

考虑一个 LIBERO-Pro 扰动任务：**"把红色杯子移到蓝色垫子上"**，但物体位置与训练时不同（Position-Swap 扰动）。

**直接 VLA 滚动（失败路径）**:
```
Step 0: VLA 看到场景 → 输出"抓取牛奶盒"动作（训练时记忆）
Step 1-50: 重复训练行为 → 失败（目标已重定向到红色杯子）
结果: FAIL
```

**Harness VLA 执行路径（成功）**:
```
t=0:  Planner Observe — 读取任务"红色杯子→蓝色垫子"
t=1:  Planner Retrieve — 从 M_task 获取参考轨迹:
        [move_to(抓取位), vla_act(抓取), move_to(放置位), set_gripper(open), release]
t=2:  Planner Compose — 从 RGB-D 检测红色杯子当前位置 → move_to(新抓取位)
t=3:  物理引擎执行 move_to → o_3 返回
t=4:  Planner Compose — vla_act(prompt="抓取红色杯子", max_chunks=10, τ="接触力>阈值")
t=5:  VLA 执行 5 个动作块 → τ 满足（已抓住）→ 返回 o_5
t=6:  Planner Observe — 检测到蓝色垫子位置偏移 → move_to(新放置位)
t=7:  Planner Compose — set_gripper(open) → release
t=8:  G(o_8) = True → SUCCESS

VLA 调用次数: 1 次 vla_act
分析原语调用: 4 次 (move_to ×2, set_gripper, release)
总 Planner 决策步数: 8 步
```

**重试场景**（假设第一次抓取失败）:
```
t=4:  vla_act 执行后 → 视觉检测"未抓住"
t=5:  Planner 触发失败模型（来自 M_global: "空抓取检测"）
t=6:  Planner Compose — rotate_wrist(target_yaw=15°)  // 调整姿态
t=7:  move_to(调整后的抓取位)
t=8:  vla_act(prompt="抓取红色杯子", max_chunks=10, τ="接触力>阈值")
t=9:  VLA 执行 → τ 满足 → 成功抓住
```

## 4. 工程视角 (Engineering View)

| 工程维度 | 分析 |
|----------|------|
| **Planner 推理延迟** | 每个执行 turn 需要一次 LLM 调用（Codex 或 Claude Code）。假设每次 2-5 秒，一个 8 步任务需要 16-40 秒的 Planner 推理时间 |
| **VLA 推理延迟** | vla_act 每次调用最多执行 max_chunks 个动作块（论文未明确默认值，典型 5-10 步），每块约 10-20ms 推理，单次调用约 0.1-0.2s |
| **总执行时间** | Planner 延迟主导（LLM API 调用），VLA 推理时间相对可忽略 |
| **内存开销** | 冻结 VLA 加载一次；M_task 每个任务一个 JSONL 文件（KB 级）；M_global 全局共享（KB 级） |
| **Bootstrapping 成本** | 每个新任务需要一次探索阶段（ generous wall-clock budget），失败时可 reset 重试。这是主要的时间成本 |
| **部署成本** | 无需 VLA 微调；需要 LLM API 接入；原语库固定，无需部署时扩展 |
| **模块边界** | Planner ↔ 原语库通过 JSON 接口通信；原语库 ↔ 物理引擎通过环境 API 通信；VLA 被封装在原语内部 |
| **量化/部署约束** | VLA 可独立量化部署；Planner 依赖外部 LLM API（Codex/CC），是主要的部署瓶颈 |

**工程含义**: Harness VLA 的架构将"认知"和"控制"清晰分离。Planner 运行在决策频率（~0.5 Hz，受 LLM 延迟限制），VLA 运行在控制频率（~10-20 Hz，在 vla_act 内部）。这种分层设计使得 VLA 可以用边缘设备部署，而 Planner 可以跑在云端。

## 5. 数据与评测 (Data & Eval)

### 评测基准

| 基准 | 类型 | 扰动类型 | 评估设置 |
|------|------|----------|----------|
| LIBERO (标准) | 桌面操作 | 无（分布内） | 4 个子任务 × 100 trials |
| LIBERO-Pro | 桌面操作 | 指令重定向(T) + 位置交换(S) | 4 个子任务 × 100 trials |
| RoboCasa365 | 厨房操作 | 移动基座 + 铰链物体 + 长程任务 | Atomic-Seen(10 seeds) + Composite(5 seeds) |
| RoboTwin C2R | 双臂操作 | Clean → Randomized 零样本迁移 | 50 tasks × 5 randomized seeds |

### VLA 后端配置

| 基准 | VLA 后端 | 类型 |
|------|----------|------|
| LIBERO / LIBERO-Pro | RLinf π0.5-SFT | 流匹配 VLA |
| RoboCasa365 | RLDX-1 RoboCasa | 扩散策略 VLA |
| RoboTwin C2R | LingBot-VLA | 自研后训练 VLA |

### Few-shot vs Zero-shot

- **Few-shot**: 先在 1 个参考 seed 上 Bootstrapping → 存储 M_task → 在扰动 seed 上评估
- **Zero-shot**: 不提供目标设置的 M_task，仅靠 Planner 实时推理（LIBERO-Pro Goal 子任务）

## 6. 能力与失败模式 (Capabilities & Failure Modes)

### 能做什么

| 能力 | 证据 | 限制 |
|------|------|------|
| 语义重定向 | LIBERO-Pro Task-T: CC 达到 79.0%（Table 5） | 依赖 Planner 的语言理解能力 |
| 空间布局偏移 | LIBERO-Pro Pos-S: CC 达到 31.0%（Table 5） | 高维度空间变化可能超出分析原语覆盖 |
| 长程厨房任务 | RoboCasa365: 55.4%（Codex） vs 30.0%（RLDX-1） | 需要移动基座原语支持 |
| 双臂协调 | RoboTwin C2R: 58.4%（CC） | 需要专门的 LingBot-VLA 后端 |
| 失败重试 | 论文 Key Finding 2 分析 | 重试次数受 step budget 限制 |

### 不能做什么

| 失败模式 | 原因 |
|----------|------|
| 完全零样本新任务 | 无 M_task 时性能显著下降（LIBERO-Pro Goal zero-shot 仅 31-79%） |
| 需要精细力控的任务 | vla_act 的早停谓词 τ 需要手动设计或启发式定义 |
| 高速动态操作 | REPL 循环的 LLM 延迟导致控制频率低 |
| 超出原语库能力的操作 | 原语库固定，无法部署时扩展 |

### 6.1 隐含假设 (Hidden Assumptions)

1. **每个新任务可以承受一次 Bootstrapping 探索** — 实际部署中，对每个新任务运行一次完整的探索可能不现实（时间成本高）
2. **Planner 的视觉理解足够可靠** — 论文假设 LLM Planner 能从 RGB-D 图像中可靠地检测物体位置、判断抓取成功/失败，但未量化 Planner 视觉理解的错误率
3. **早停谓词 τ 可设计** — vla_act 需要任务特定的早停谓词，论文未讨论如何自动学习或泛化这些谓词
4. **分析原语覆盖非接触阶段** — 假设 move_to/rotate/set_gripper 等 6 个分析原语足以覆盖所有非接触操作，但对于复杂铰链物体（如 RoboCasa 中的橱柜），可能需要更多专用原语
5. **M_task 的参数化轨迹可泛化** — 假设将具体坐标替换为符号感知查询后，轨迹能泛化到新的空间布局，但未分析参数化错误的累积效应

## 7. 与相关工作对比 (Comparison)

| 方法 | 核心思路 | VLA 角色 | 部署时扩展 | 分布外泛化 | 适用场景 |
|------|----------|----------|------------|------------|----------|
| **Code as Policies** | LLM 生成程序控制机器人 | 无 VLA | 扩展原语库 | 中等 | 结构化任务 |
| **ProgPrompt** | 预编程 + LLM 组装 | 无 VLA | 手动添加原语 | 低 | 已知任务族 |
| **RATS** | 测试时自适应搜索 | VLA 直接控制 | 搜索策略 | 中等（43.8% LIBERO-Pro） | 扰动鲁棒 |
| **Cap-X** | 多模态 LLM Agent | 无 VLA | 动态生成技能 | 低-中 | 通用任务 |
| **Harness VLA** | 冻结 VLA 作为原语 + 双记忆 | VLA 作为接触原语 | 无需扩展 | 高（82.4% LIBERO-Pro） | 接触丰富任务 |

**面试 Tip**: 当被问到"Harness VLA 和 Code as Policies 的区别"时，回答："CaP 让 LLM 生成完整程序并不断扩展原语库来覆盖新场景；Harness VLA 保持原语库固定，让 LLM 学习如何编排固定的原语（包括将 VLA 封装为一个原语）。前者是'技能扩展'路线，后者是'编排学习'路线。"

## 8. 精讀建議 (Reading Guide)

- **值得精讀原文的人**:
  1. 做多模态具身 Agent 的研究者 — 理解如何将 VLA 集成到 Agent 框架中
  2. 要评估冻结 VLA 部署到扰动场景可行性的工程师 — 本文提供了无需微调的实用路径
  3. 研究 LLM-Robot 接口设计的学者 — 固定原语库 + JSON 接口的统一设计值得参考

- **建議章節路徑**:
  - 先读 §2（框架）→ 理解固定原语库、双记忆、两阶段工作流
  - 再看 §3.2（基准性能）→ 理解四个基准上的定量结果
  - 可跳 §3.1（实验设置细节）— 除非你要复现实验

- **不值得精讀的理由**:
  - 如果你不做机器人学习或 VLA 部署，这篇的工程细节可能过于具体
  - 如果你已熟悉 Code as Policies 类方法，核心思想（LLM 编排 + 原语调用）并不新颖
  - 论文缺乏真机实验，全部在仿真环境验证

---
[← Back to Theory](./README.md)

**关键引用**:
- 论文: https://arxiv.org/abs/2607.08448
- 项目页: https://harnessvla.github.io/
- LIBERO: https://bsergioiv.github.io/
- LIBERO-Pro: https://github.com/LIBERO-Benchmark/LIBERO
- RoboCasa: https://robocasa.ai/
- RoboTwin: https://github.com/TencentRoboticsLab/RoboTwin
