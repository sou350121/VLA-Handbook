# Benchmark 主线总纲：从任务世界到安全约束，再到世界模型评测器 (Benchmark Mainline Overview)

> 这页的目标不是重复每篇文章的细节，而是把当前 benchmark 专题里最关键的 6 篇笔记串成一条主线，回答一个更“全局”的问题：**如果你想理解具身系统到底该怎么被定义、诊断、比赛、约束和评测，应该按什么顺序看？**

## 相关导读（本专题内）

- **BEHAVIOR-1K**：[`./behavior_1k_human_centered_embodied_ai_benchmark_omnigibson_2024.md`](./behavior_1k_human_centered_embodied_ai_benchmark_omnigibson_2024.md)
- **ENACT**：[`./enact_embodied_cognition_world_modeling_egocentric_interaction_2025.md`](./enact_embodied_cognition_world_modeling_egocentric_interaction_2025.md)
- **2025 BEHAVIOR Challenge 冠军方案**：[`./behavior_challenge_2025_first_place_solution_task_adaptation_vla_2025.md`](./behavior_challenge_2025_first_place_solution_task_adaptation_vla_2025.md)
- **IS-Bench**：[`./is_bench_interactive_safety_vlm_embodied_agents_household_tasks_2025.md`](./is_bench_interactive_safety_vlm_embodied_agents_household_tasks_2025.md)
- **WorldEval**：[`./worldeval_world_model_policy_evaluator_2025.md`](./worldeval_world_model_policy_evaluator_2025.md)
- **Ctrl-World × WorldArena**：[`./ctrl_world_worldarena_embodied_world_model_benchmark_2026.md`](./ctrl_world_worldarena_embodied_world_model_benchmark_2026.md)
- **World Model 主线总纲**：[`../../world_model_mainline.md`](../../world_model_mainline.md)

---

## 0. 全局定位卡

| 维度 | 内容 |
|---|---|
| 所属位置 | `theory/frontier/benchmarks/` 下的专题总纲页 |
| 上游入口 | [`./README.md`](./README.md)、[`../../README.md`](../../README.md) |
| 平行主线 | [`../../vla_research_mainline.md`](../../vla_research_mainline.md) |
| 相邻页面 | [`../../evaluation.md`](../../evaluation.md)、[`../../benchmark_tracker.md`](../../benchmark_tracker.md)、[`../../paper_index.md`](../../paper_index.md)、[`../../literature_review.md`](../../literature_review.md)、[`../../world_model_mainline.md`](../../world_model_mainline.md) |
| 最适合谁看 | 想先抓整体 benchmark 结构，再决定读哪篇单文的人 |
| 最重要用途 | 导航页、复述稿、专题总图 |

这页在全局 `vla-handbook` 里的定位，不是和 [`../../vla_research_mainline.md`](../../vla_research_mainline.md) 平级竞争“谁是总主线”，而是：

- `benchmark / evaluator` 子域里的**二级主线页**
- 单篇 benchmark 笔记之间的**桥接页**
- 把“任务定义 -> 诊断 -> 系统解法 -> 安全 -> evaluator”串起来的专题叙事页

---

## 1. 这条主线到底在回答什么？

很多 benchmark 笔记的问题，不是单篇写得不清楚，而是读者看完以后仍然不知道：

- 这些 benchmark 彼此是什么关系？
- 哪篇在定义问题，哪篇在诊断能力，哪篇在看系统解法？
- 为什么看完 benchmark 还要看 safety 和 evaluator？

这条主线的一个更完整表述是：

```text
先定义任务世界
  -> 再诊断 agent 到底缺什么能力
  -> 再看在 leaderboard 压力下什么系统能活下来
  -> 再问它是不是安全
  -> 最后问能不能用 world model 去评测和筛选这些系统
```

也就是说，这不是 6 篇“并列的论文笔记”，而是一条逐层加约束的研究路径。

---

## 2. 一张总图：benchmark 主线地图

```text
┌──────────────────────────────────────────────┐
│ 1) 问题定义层：BEHAVIOR-1K                  │
│    “真实 household 任务世界到底长什么样？” │
└──────────────────────┬───────────────────────┘
                       │ 定义任务、场景、物理与逻辑
                       v
┌──────────────────────────────────────────────┐
│ 2) 能力诊断层：ENACT                        │
│    “模型是否真的理解动作-状态-观察链条？”   │
└──────────────────────┬───────────────────────┘
                       │ 把失败拆到 cognition 层
                       v
┌──────────────────────────────────────────────┐
│ 3) 系统解法层：2025 BEHAVIOR Challenge      │
│    “在 benchmark 压力下，什么 system recipe │
│      才能真正拿分？”                        │
└──────────────────────┬───────────────────────┘
                       │ 从能力问题落到工程系统
                       v
┌──────────────────────────────────────────────┐
│ 4) 安全约束层：IS-Bench                     │
│    “能做事，不代表能安全地做事。”           │
└──────────────────────┬───────────────────────┘
                       │ 给执行过程加 safety gate
                       v
┌──────────────────────────────────────────────┐
│ 5) 评测器层：WorldEval                      │
│    “能不能先用 world model 排策略、筛版本？”│
└──────────────────────┬───────────────────────┘
                       │ evaluator 角色被单独提出
                       v
┌──────────────────────────────────────────────┐
│ 6) 统一评测层：Ctrl-World × WorldArena      │
│    “world model 不只比像不像，还要比能不能 │
│      当 Data Engine / Policy Evaluator /    │
│      Action Planner。”                      │
└──────────────────────────────────────────────┘
```

一句话总结这张图：

**BEHAVIOR-1K 给出世界，ENACT 诊断能力，Challenge 告诉你系统怎么活下来，IS-Bench 给它加安全约束，WorldEval 和 WorldArena / Ctrl-World 则开始把评测器本身做成研究对象。**

### 再看一张：问题递进图

```text
定义世界          诊断能力          验证系统          约束风险          扩展评测
   │                 │                 │                 │                 │
   v                 v                 v                 v                 v
BEHAVIOR-1K  --->  ENACT  --->  Challenge  --->  IS-Bench  --->  WorldEval / WorldArena

你到底在什么世界做任务？
         -> 你理解交互了吗？
                  -> 你能稳定拿分吗？
                           -> 你会不会危险？
                                    -> 我能不能更便宜地评你？
```

---

## 3. 六层节点一览

| 层级 | 文章 | 它在回答什么问题 | 对手册的意义 |
|---|---|---|---|
| 1 | `BEHAVIOR-1K` | 真实 household 任务世界该如何定义？ | 给 benchmark 主线提供问题定义与任务母体 |
| 2 | `ENACT` | 模型是否理解交互因果与长时程 ego-view 演化？ | 把“任务失败”拆成“认知失败” |
| 3 | `2025 BEHAVIOR Challenge 冠军方案` | 在 leaderboard 压力下，什么系统 trick 真能换分？ | 展示 system recipe，而不只是 paper idea |
| 4 | `IS-Bench` | 任务做成了，过程是否安全？ | 把安全从终局检查推进到过程检查 |
| 5 | `WorldEval` | 能否用 world model 保持策略排序一致性？ | 把 world model evaluator 做成可操作系统 |
| 6 | `Ctrl-World × WorldArena` | world model 到底该怎么统一评测其功能价值？ | 把 evaluator / planner / data engine 统一进 benchmark |

---

## 4. 它和全局 VLA 主线是什么关系？

`vla-handbook` 里已经有一条更偏“模型能力增长”的研究主线：[`../../vla_research_mainline.md`](../../vla_research_mainline.md)。

那条线主要在回答：

- baseline 从哪里起步
- 为什么数据规模化重要
- 感知增强补什么缺口
- 后训练 / recovery 为什么决定真实成功率

而这页的 benchmark 主线主要在回答：

- 这些能力该如何被定义、分层、约束、评测

可以把它们看成两条正交主线：

| 全局研究主线 | benchmark 主线 |
|---|---|
| baseline（ACT / DP） | 问题定义（BEHAVIOR-1K） |
| 数据规模化 | 能力诊断（ENACT） |
| 感知增强 | 系统解法（Challenge winner） |
| 后训练 / recover | 安全约束（IS-Bench） |
| 闭环迭代 | evaluator / planner（WorldEval / WorldArena / Ctrl-World） |

一句话说：

- [`../../vla_research_mainline.md`](../../vla_research_mainline.md) 更像“怎么把 agent 做强”
- `benchmark_mainline.md` 更像“怎么知道 agent 到底强在哪、弱在哪、安不安全、值不值得信”

所以这页不是另起炉灶，而是在全局手册里补上 **评估与约束轴**。

---

## 5. 这条主线为什么成立？

### 3.1 它不是“论文年代排序”，而是“问题递进排序”

如果只按年份看，会容易误以为：

- 这些工作只是不同团队在不同方向做 benchmark

但按问题递进看，就能看到真正的结构：

1. `BEHAVIOR-1K` 先把任务世界定义出来  
2. `ENACT` 追问：失败到底是没理解，还是没控制好  
3. `Challenge 冠军方案` 回答：真正上线 leaderboard，系统要怎么补丁化  
4. `IS-Bench` 追问：这些系统会不会在执行过程中制造风险  
5. `WorldEval / WorldArena` 再往前走：能不能把“评测本身”做得更便宜、更可扩展、更世界模型化

### 3.2 它把“agent 问题”逐层变成“系统问题”

这条线有一个很强的结构感：

```text
task world
  -> cognition
  -> system recipe
  -> safety
  -> evaluator
```

前半段在问 agent 本身：

- 任务是什么？
- 你理解了吗？
- 你做得出来吗？

后半段在问系统外层：

- 你做得安全了吗？
- 我该怎么评你？

这也是为什么这条线越来越像“工程化主线”，而不只是 benchmark 列表。

### 3.3 一张约束漏斗图

```text
候选 agent / system ideas
          │
          v
  [任务世界约束]
  BEHAVIOR-1K
          │
          v
  [认知诊断约束]
  ENACT
          │
          v
  [leaderboard 工程约束]
  Challenge winner
          │
          v
  [安全过程约束]
  IS-Bench
          │
          v
  [评测可扩展性约束]
  WorldEval / WorldArena
          │
          v
真正值得信、值得部署、值得继续迭代的系统
```

---

## 6. 这页覆盖什么，不覆盖什么？

这页主要覆盖的是：

- household / manipulation 导向的 benchmark 世界
- capability diagnosis
- challenge-level system recipe
- interactive safety
- world model evaluator / planner

这页暂时**不直接覆盖**的是：

- 模型 baseline 细拆：这类内容更适合看 `SimVLA` 等 baseline 分析页
- 通用评估协议细节：建议去看 [`../../evaluation.md`](../../evaluation.md)
- 持续榜单追踪：建议去看 [`../../benchmark_tracker.md`](../../benchmark_tracker.md)
- 大范围论文检索：建议去看 [`../../paper_index.md`](../../paper_index.md) 与 [`../../literature_review.md`](../../literature_review.md)

也就是说，这页是“结构页”，不是“协议手册”或“追踪表”。

---

## 7. 三条推荐阅读路线

### A) 你想从“机器人 benchmark 为什么这么难”入门

建议顺序：

1. [`BEHAVIOR-1K`](./behavior_1k_human_centered_embodied_ai_benchmark_omnigibson_2024.md)
2. [`ENACT`](./enact_embodied_cognition_world_modeling_egocentric_interaction_2025.md)
3. [`2025 BEHAVIOR Challenge 冠军方案`](./behavior_challenge_2025_first_place_solution_task_adaptation_vla_2025.md)

这条线会让你先理解：

- 世界为什么难
- cognition 为什么不够
- system recipe 为什么最后不可避免

### B) 你想从“安全与部署前评估”入手

建议顺序：

1. [`2025 BEHAVIOR Challenge 冠军方案`](./behavior_challenge_2025_first_place_solution_task_adaptation_vla_2025.md)
2. [`IS-Bench`](./is_bench_interactive_safety_vlm_embodied_agents_household_tasks_2025.md)
3. [`WorldEval`](./worldeval_world_model_policy_evaluator_2025.md)

这条线更像 deployment 视角：

- 系统先得能跑
- 然后得安全
- 最后还得有便宜、可扩展的评测器

### C) 你想从“world model 到底值不值得信”入手

建议顺序：

1. [`ENACT`](./enact_embodied_cognition_world_modeling_egocentric_interaction_2025.md)
2. [`WorldEval`](./worldeval_world_model_policy_evaluator_2025.md)
3. [`Ctrl-World × WorldArena`](./ctrl_world_worldarena_embodied_world_model_benchmark_2026.md)

这条线的重点是：

- 先看 cognition 诊断
- 再看 world model evaluator
- 最后看统一 benchmark 与更强 action-conditioned 解法

---

## 8. 这 6 篇合起来揭示了哪些核心张力？

### 5.1 `task success` 不等于 `capability understanding`

这是 `BEHAVIOR-1K -> ENACT` 的核心张力。

- 任务没做成，不一定只是控制差
- 也可能是模型没理解动作后果、没记住时序、没维持世界状态

### 5.2 `capability` 不等于 `system performance`

这是 `ENACT -> Challenge winner` 的核心张力。

- 你可以部分理解交互
- 但真进比赛和长时程 benchmark，还是会被 non-Markovian state、recover 数据缺失、dexterity 和 heuristic 问题狠狠干翻

### 5.3 `system performance` 不等于 `safe performance`

这是 `Challenge winner -> IS-Bench` 的核心张力。

- 会做任务
- 不代表会在正确时机处理风险

### 5.4 `good-looking rollout` 不等于 `useful evaluator`

这是 `WorldEval -> WorldArena / Ctrl-World` 的核心张力。

- 视频更真
- 不代表 evaluator 更可靠
- `EWMScore` 和 human correlation 高，不代表 action planning correlation 也高

这条张力恰好把整条主线闭环了：

```text
定义任务
  != 理解任务
  != 完成任务
  != 安全完成任务
  != 可靠评估任务
```

---

## 9. 如果你要把这条线讲给面试官，怎么讲最顺？

一个比较稳的 30 秒版本是：

```text
如果从 benchmark 主线看，
BEHAVIOR-1K 负责定义真实 household 任务世界，
ENACT 负责诊断 VLM 是否真正理解交互因果，
2025 BEHAVIOR Challenge 冠军方案展示在长时程 benchmark 压力下哪些 system recipe 真能拿分，
IS-Bench 再补上执行过程里的安全约束，
而 WorldEval 与 WorldArena / Ctrl-World 则把“评测器”本身做成研究对象，
开始系统回答：能不能用 world model 去筛策略、评策略、甚至辅助规划。
```

如果只允许你说一句话，那就说：

**这条 benchmark 主线，讲的是具身系统如何从“定义任务”一步步走到“安全、可评测、可规模化迭代”。**

---

## 10. 这条主线还缺什么？

当前这 6 篇已经能形成闭环，但还可以继续补三类节点：

- **competition / challenge solution 系列**：继续补更多系统解法层案例
- **interactive safety / policy governance**：把 safety 从 benchmark 推向 deployment protocol
- **world model evaluator / planner 系列**：继续追踪 evaluator 是否真正能替代部分真机回归
- **真实评测协议 / sim2real predictivity 系列**：把这条专题更明确接到 [`../../evaluation.md`](../../evaluation.md)

也就是说，这条线还可以继续延长，但骨架已经有了。

---

## 11. 你在手册里该怎么用这页？

这页最适合三个用途：

1. **当导航页**：新读者先看这页，再决定进哪篇  
2. **当主线页**：把 benchmark 专题从“列表”升级成“结构”  
3. **当复述稿**：面试或写综述时，快速把 6 篇串起来  

如果你已经看过单篇笔记，再回来看这页，重点就不是“再学一遍内容”，而是确认：

- 每篇到底在主线里补哪一层
- 各层之间的张力是什么
- 未来应该继续补哪类 paper

---
[← Back to Benchmark Hub](./README.md)
[← Back to Theory](../../README.md)
