# Agent 架构真机部署攻略：把 VLA 变成“能稳定做完任务”的系统 (Agentic VLA Deployment Guide)

> **适用对象**：做真机落地/系统集成的人（不是只训模型的人）  
> **核心定位**：把 VLA 从“一个模型出动作”升级成“可观测、可恢复、可验收”的 **Agent 系统**：高层负责任务编排与工具调用，低层保证实时执行与安全闭环。  

这份攻略默认你接受一个结论：**大模型推理不可能承担 1kHz 级的接触/平衡闭环**。因此正确的落地方式不是把所有东西塞进一个网，而是把系统切成不同时间尺度的层，并为每层设置接口与验收指标。

---

## 0. 一张图先把“Agent 架构”说清楚

```text
                   slow (0.2–2 Hz)             mid (10–50 Hz)                 fast (200–1000 Hz)
┌───────────────────────────────────┐   ┌──────────────────────────┐   ┌──────────────────────────────┐
│ TaskAgent / Planner (LLM/VLM)     │   │ SkillPolicy / VLA         │   │ RealTimeController (RT)       │
│ - goal -> plan/steps              │-->| - obs -> action targets   │-->| - targets -> torque/pos cmd   │
│ - tool calling / retrieval        │   │ - short-horizon control    │   │ - balance/contact/safety      │
└───────────────┬───────────────────┘   └─────────────┬────────────┘   └─────────────┬──────────────┘
                │                                      │                               │
                v                                      v                               v
     Tools (APIs/ROS services)                 SensorFusion/State                 Motor/Hand/Arm drivers
 - perception, mapping, grasp detect          - sync + filtering                  - EtherCAT/CANFD/RTDE
 - database/memory/logging                    - ring buffer                       - hard limits/estop
 - simulation/eval hooks
```

对照仓库已有内容：  
- “大脑-小脑”分离（远程推理）见 [`deployment/pi0_deployment.md`](./pi0_deployment.md)  
- 多模态同步见 [`deployment/multimodal_data_synchronization.md`](./multimodal_data_synchronization.md)  
- ROS2 性能与实时性见 [`deployment/ros_and_optimization.md`](./ros_and_optimization.md)  
- 末端触觉闭环控制架构见 [`deployment/end_effector_control.md`](./end_effector_control.md)

---

## 1. 为什么 VLA 需要 Agent 架构（而不是“一个模型端到端”）

### 1.1 真机的三类约束

- **时间尺度冲突**：任务规划（秒级）与接触控制（毫秒级）天然不同频  
- **可观测性不完整**：遮挡/滑移/柔顺性差异导致“看懂”不等于“做得稳”  
- **安全与恢复**：真机必然有失败，需要“早停/回退/重试/降级”，这不是一个单网络能替代的

### 1.2 一个实用定义：Agent = 任务编排 + 工具调用 + 恢复策略

你可以把 Agent 当成“把 VLA 变成系统”的那层 glue：  
- **编排**：把长任务拆成可执行步骤（含前置条件与验收条件）  
- **工具调用**：调用感知/地图/抓取检测/数据库等模块，而不是让 VLA 直接“脑补”  
- **恢复**：失败后自动回退到可控状态，不让错误累积到不可逆

---

## 2. 组件拆分：每层吃什么、吐什么、跑多快

### 2.1 推荐组件表（可直接抄进系统设计文档）

| 层 | 典型频率 | 输入 | 输出 | 实现建议 | 验收指标（最小） |
|---|---:|---|---|---|---|
| Planner（LLM/VLM） | 0.2–2 Hz | 任务目标、场景摘要、状态机状态 | 结构化步骤/工具调用 | 独立进程/远程服务 | 步骤正确率、恢复触发率、幻觉率（工具调用失败率） |
| SkillPolicy（VLA） | 10–50 Hz | 对齐后的观测（vision/tactile/proprio）+ 指令 | 动作 targets（delta pose / joint targets / action chunk） | GPU 边缘/远程推理皆可 | 控制延迟（p99）、动作抖动、成功率 |
| RT Controller（S0） | 200–1000 Hz | 当前状态（关节/IMU/触觉/力矩）+ targets | 执行器指令（pos/vel/torque） | C++/实时线程/MCU | jitter（p99/p999）、限幅触发率、跌倒/碰撞保护 |
| Safety Supervisor | 200–1000 Hz | 力/力矩/速度/温度/电流等 | estop / soft stop / safe mode | 硬件+软件双保险 | MTBF、误触发率、最坏情况下停止距离 |
| Observability（日志/回放） | async | 所有关键流 | 可复盘记录 | 独立落盘进程 | 可复盘率（能复现失败）、指标报表完整性 |

### 2.2 数据接口（建议用“typed message”固定下来）

建议把跨进程/跨设备的数据约束成最小协议：  

```text
ObservationPacket:
  - t_ns (monotonic timestamp)
  - images: {cam_id: (H,W,3) or encoded bytes}
  - tactile: (optional) array/tokens
  - proprio: q, dq, effort, imu
  - task: instruction string + task_id

ActionTargets:
  - t_ns
  - representation: delta_pose | joint_targets | action_chunk
  - values: float[]

RTCommand:
  - t_ns
  - mode: position | velocity | torque
  - values: float[]
```

---

## 3. 部署拓扑（最常见的 3 种）

### 3.1 单机单进程（只适合最小 demo）

优点：快；缺点：抖动不可控、故障域大。  
适合：桌面级 demo、验证数据管线。

### 3.2 “大脑-小脑”分离（推荐默认）

参考 [`deployment/pi0_deployment.md`](./pi0_deployment.md) 的 Server/Client 形态：  
- **Server（GPU 工作站）**：跑 Planner/VLA（大模型）  
- **Client（机载计算）**：采集传感器、做同步与缓存、下发 targets  
- **RT（实时控制器/实时线程）**：执行与安全

关键点：网络延迟不是最大问题，**jitter 与队列堆积** 才是。

### 3.3 多机分布式（人形/多臂/多传感器）

当你有多摄像头、多末端、多触觉阵列时：  
- 传感器端做 `Timestamp-at-Source`（见同步文档）  
- 中间件层尽量 zero-copy（见 ROS2 优化文档）  
- RT 层永远不要依赖网络（网络只喂 targets）

---

## 4. 落地步骤（从 0 到可跑）

### 4.1 Step 0：先定“频率预算”和“失败边界”

写在 README 里、贴在墙上：  
- VLA 输出频率目标（例如 20Hz）  
- RT 控制频率（例如 1kHz）  
- 最大允许端到端延迟（例如 p99 < 50ms）  
- 触觉/力矩阈值（软限位/硬限位）  

### 4.2 Step 1：把同步做好（不然训练/部署都会错）

直接按 [`deployment/multimodal_data_synchronization.md`](./multimodal_data_synchronization.md) 的 ring buffer + 对齐策略做：  
- 统一时钟（PTP/NTP）  
- 曝光时间戳口径（曝光中点 vs 到达时间）  
- 记录每条流的 `t_ns`，可回放验证

### 4.3 Step 2：先把 RT 控制层跑稳（再接大模型）

建议先用“手写 targets + RT 执行”跑通：  
- 末端位置/速度限幅  
- jerk 限制  
- 触觉/力矩触发的紧急退让（micro recovery）  

相关可参考：[`deployment/end_effector_control.md`](./end_effector_control.md)

### 4.4 Step 3：接入 VLA（先做短任务闭环）

先做 10–30 秒短任务：  
- pick/place、开合抽屉的一步  
然后再扩大到分钟级任务（需要 Agent/恢复）。

### 4.5 Step 4：接入 Agent（长时程编排 + 工具调用 + 恢复）

推荐最小状态机：  
```text
Idle -> Acquire -> Approach -> Contact -> InContactManip -> Release -> Done
            ^          |          |            |
            |          v          v            v
         Recover <--- Abort <--- Fault <--- NoProgress
```

你不一定要一开始就把 Planner 写得很复杂；但必须把 **NoProgress / Fault / Abort** 做出来，避免“死循环硬顶”。

---

## 5. 常见坑（最容易让 VLA 在真机崩的点）

- **把大模型放进实时环**：一旦卡顿就抖动/跌倒/撞击  
- **只看平均延迟不看 p99/p999**：最坏情况下才决定安全  
- **时间戳口径不一致**：训练对齐错，部署必错  
- **没有回放与失败病历**：你会永远在“感觉变好了”里循环  
- **恢复策略缺失**：长任务的失败不是“偶发”，而是“必然”  

---

## 6. 最小验收清单（建议每周跑一次）

- **实时性**：RT loop jitter p99/p999；VLA 推理延迟 p99；队列是否堆积  
- **同步**：视觉/触觉/本体对齐误差分布（ms）  
- **安全**：软限位触发率、硬停触发率、碰撞/跌倒保护覆盖  
- **可复盘**：给任意一次失败，能在 10 分钟内回放定位到“哪个阶段、哪个传感器、哪个阈值”  
- **长任务稳定性**：分钟级任务的“恢复次数/成功率/平均耗时”曲线是否收敛  

---

## 7. 记忆系统（Memory）：让 Agent 真的“能做长任务”

> 这部分来自你提供的“Agent 记忆系统”文章的工程化抽象：短期（会话内）+ 长期（跨会话）两层，并配套上下文工程（压缩/卸载/隔离）。  
> 关键点：**记忆不是把历史全塞进 prompt**，而是把“可检索、可审计、可更新”的信息变成工具与数据面。

### 7.1 两类记忆：按“是否跨 session”来划分

- **短期记忆（Session / Working Memory）**：当前会话里的消息与工具调用结果（会随 token 增长爆炸）。  
- **长期记忆（Long-Term Memory）**：从短期里提炼出的“事实/偏好/经验/技能/失败病历”，可跨 session 检索并注入。

你可以把它理解成：

```text
ShortTermMemory = 事件流（messages + tool calls + observations）= 可回放
LongTermMemory  = 抽取后的结构化条目（facts/preferences/skills/failures）= 可检索
```

### 7.2 通用集成模式：Retrieve -> Inject -> Act -> Record

```text
Step1 Retrieve:  针对当前 query/任务阶段，从长期记忆检索候选（top-k）
Step2 Inject:    把候选以“可控格式”注入短期上下文（或作为 tool result）
Step3 Act:       Planner/VLA/RT 执行本轮动作
Step4 Record:    从本轮事件流抽取可沉淀的信息，写回长期记忆（带审计日志）
```

对应到常见框架的概念映射：
- Google ADK：`Session`（短期）+ `MemoryService`（长期）→ `add_session_to_memory` / `search_memory`（见 [ADK MemoryService](https://google.github.io/adk-docs/sessions/memory/)）
- LangChain/LangGraph：long-term memory 以 JSON 文档存入 store（namespace + key），工具可读写（见 [LangChain long-term memory](https://docs.langchain.com/oss/python/langchain/long-term-memory)）
- AgentScope：memory 模块提供消息存储原子能力（mark/tag），压缩策略在 agent 层实现（见 [AgentScope 记忆](https://doc.agentscope.io/zh_CN/tutorial/task_memory.html)）

### 7.3 三种上下文工程策略（你文里最重要的落地抓手）

- **Context Reduction（缩减）**：保留预览/摘要，直接减少 token（会丢细节）  
- **Context Offloading（卸载）**：把大块内容移到外部存储，只在上下文保留引用（不丢信息，可按需加载）  
- **Context Isolation（隔离）**：把“搜索/分析/跑脚本”等任务丢给子 agent，主 agent 只拿最终产物（降低主上下文膨胀）

工程上建议默认策略：
- 工具输出（网页/日志/长 JSON）→ **offload**（只留 `artifact_id/path`）  
- 过久的对话历史 → **summarize**（保留最近 N 条原文 + 历史摘要）  
- 大范围检索/代码探索 → **isolate**（子 agent 执行，主 agent 只收摘要+引用）

### 7.4 “长期记忆”的最小实现（建议别一上来就过度设计）

你文里给的组件栈很完整：LLM 抽取 + embedding + vector store + graph store + reranker + SQLite 审计。  
最小可跑版本建议先只做两件事：

1) **Record（写入）**：把事件流抽取成结构化条目（JSON），落地到一个持久化 store，并写审计日志  
2) **Retrieve（检索）**：按 query 检索 top-k 条目，带来源与时间戳返回

推荐存储拆分（从易到难）：

```text
ColdStorage (files): 全量事件流/传感器片段/失败回放（可离线复盘）
SQL (audit):         每次 record/retrieve 的操作日志（可追溯/可回滚）
VectorStore:         可检索条目（facts/preferences/skills/failures）
[Optional]GraphStore:实体-关系（工具链复杂、但能做更强关系推理）
```

### 7.5 把“记忆”接回 VLA 真机：你应该存什么

对 VLA/机器人系统，长期记忆最有价值的不是“聊天事实”，而是以下几类：
- **任务记忆**：某个场景的步骤模板、阶段切换条件、恢复策略（NoProgress/Fault 的处理）  
- **工具记忆**：某类工具的最佳调用参数、失败模式与替代方案（例如相机标定失败怎么 fallback）  
- **失败病历**：失败类型、触发阈值、传感器状态、回放片段引用（对应本指南的可复盘要求）  
- **用户/业务偏好**：安全边界、速度/力上限、容错偏好（适用于“产品化”机器人）  

### 7.6 风险与边界（安全/隐私/污染）

长期记忆会带来两个硬风险：  
- **隐私/合规**：用户偏好与历史可能包含敏感信息（需要访问控制与数据治理）  
- **记忆污染**：错误记忆写入会被反复检索放大（需要审计与可回滚）

最小防线：
- 所有写入必须带 `who/when/why/source` 元数据（可审计）  
- 对“写入长期记忆”的工具加权限门（不是每轮都自动写）  
- 建立“记忆回归测试”：抽样 query，检查是否检索到过期/错误条目

> 延伸阅读：  
> - FlowLLM 的 context 管理读物索引：[`FlowLLM-AI/flowllm/docs/zh/reading`](https://github.com/FlowLLM-AI/flowllm/tree/main/docs/zh/reading)  
> - O-Mem（个性化长程记忆框架）：[`arXiv:2511.13593`](https://arxiv.org/abs/2511.13593)

---

## 8. 评估体系（Evals）：把“能用/稳定/成本”变成可衡量指标

> 这节把你提供的 Anthropic 文章观点工程化：不同类型 Agent 需要不同评分器（grader）组合；同时用 `pass@k` 与 `pass^k` 区分“潜力/可用性”与“一致性/稳定性”。  
> 参考：Anthropic 工程文档 [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)，以及 `τ-Bench`（[arXiv:2406.12045](https://arxiv.org/abs/2406.12045)）、`τ2-Bench`（[arXiv:2506.07982](https://arxiv.org/abs/2506.07982)）、BrowseComp（[arXiv:2504.12516](https://arxiv.org/abs/2504.12516)）。

### 8.1 先对齐一个事实：Agent eval 的单位不是“单次运行”

Agent 行为每次运行都可能变化：同一任务在 run1 成功、run2 失败是常态。  
所以评估要把“重复试验”写进协议。

两个最有用的聚合指标：

```text
pass@k:  k 次尝试里至少成功 1 次的概率（更像“可用性/潜力”）
pass^k:  k 次尝试全部成功的概率（更像“一致性/稳定性”）
```

产品含义：
- **工具型 agent**（一次成功就行）：更关心 `pass@k`
- **面向用户的 agent**（每次都要靠谱）：更关心 `pass^k`

### 8.2 评分器（Grader）清单：从“结果”到“过程”

你给的例子可以抽象成 5 类 grader（可组合）：

- **deterministic_tests**：单测/集成测试/端到端脚本（最硬的真值）
- **state_check**：检查最终状态（数据库/文件系统/订单状态/机器人状态机状态）
- **tool_calls**：工具调用约束（必须调用哪些工具、禁止调用哪些工具、参数范围）
- **static_analysis**：静态分析（lint/typecheck/security/perf）
- **llm_rubric**：用 rubric 评“过程是否合理/表达是否合规/交互是否得体”（对开放任务很关键）

### 8.3 四类 Agent 的评估模板（可直接抄）

#### A) 编码 Agent（Coding Agent）

关键原则：**确定性评分器优先**，过程评分作为补充。

```yaml
task:
  id: "fix-auth-bypass_1"
  desc: "修复当密码字段为空时的认证绕过漏洞"

  graders:
    - type: deterministic_tests
      required:
        - test_empty_pw_rejected.js
        - test_null_pw_rejected.js

    - type: static_analysis
      commands:
        - eslint
        - tsc

    - type: llm_rubric
      rubric: prompts/code_quality.md

  tracked_metrics:
    - type: transcript
      metrics: [n_turns, n_toolcalls, n_total_tokens]
    - type: latency
      metrics: [time_to_first_token, output_tokens_per_sec, time_to_last_token]
```

可用基准：SWE-bench Verified、Terminal-Bench（端到端构建/部署类任务）。

#### B) 对话 Agent（Support/Sales/Coach）

关键原则：**最终状态可验证 + 交互质量可评分**。很多场景需要一个 LLM 扮演用户做多轮对话（`τ-Bench/τ2-Bench` 的核心）。

```yaml
task:
  id: "refund_frustrated_user"
  desc: "处理沮丧用户的退款请求"

  graders:
    - type: state_check
      expect:
        tickets: {status: resolved}
        refunds: {status: processed}

    - type: tool_calls
      required:
        - tool: verify_identity
        - tool: process_refund
          params: {amount: "<=100"}
        - tool: send_confirmation

    - type: llm_rubric
      rubric: prompts/support_quality.md
      assertions:
        - "Agent对客户的沮丧表现出同理心"
        - "解决方案被清晰解释"
        - "Agent的回复基于fetch_policy工具的结果"

    - type: transcript
      max_turns: 10
```

#### C) 研究/搜索 Agent（Research Agent）

关键原则：开放任务很难有“唯一真值”，所以 grader 组合通常是：
- **来源约束**：每个关键主张都应有来源支持（引用可追溯）
- **覆盖度**：来源里的关键信息是否被覆盖/使用（遗漏即扣分）
- **来源质量**：权威性/一手性优先（不能只信 SEO 排名）

BrowseComp 的设计点值得学：问题“答案易验证但难找到”，便于做自动评分与回归。

#### D) 计算机使用 Agent（Computer-Use Agent）

关键原则：不能只看“UI 看起来对了”，要做 **后端状态验证**（WebArena/OSWorld 这类思想：订单是否真的下单、文件是否真的生成、配置是否真的写入）。  

另一个很实用的评估点（Anthropic 提到的）：检查 agent 是否为场景选择了“正确的工具形态”：
- 文本密集 → DOM/结构化读取更省 token
- 电商/布局复杂 → 截图/视觉更省 token

你可以把它做成 `tool_calls` grader：

```yaml
graders:
  - type: tool_calls
    required:
      - tool: read_dom
    forbidden:
      - tool: screenshot_ocr
```

或反过来，视任务而定。

### 8.4 把 eval 接回“真机 VLA”最有用的 3 个落点

对机器人/具身系统（含 VLA）来说，最容易被忽略但最关键的是：**不仅评结果，也评过程与安全边界**。

- **落点 1：state_check = 机器人系统的“后端真值”**  
  不只看“动作完成了吗”，还要检查：状态机是否到达正确阶段、是否触发过保护、是否出现 NoProgress、是否发生过超限。
- **落点 2：过程评分 = 是否在正确时间尺度用对模块**  
  例如 Planner 不应该参与 1kHz 决策；VLA 输出不应绕过 RT safety；触觉/力控应该在接触阶段主导。
- **落点 3：稳定性指标 = `pass^k`（一致性）优先**  
  真机面向用户的系统，`pass@k` 再高，`pass^k` 低也意味着“经常翻车”。

---

[← Back to Deployment](./README.md)

