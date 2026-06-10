# Benchmark 主线总纲：从任务世界到安全约束，再到世界模型评测器 (Benchmark Mainline Overview)

> 这页的目标不是重复每篇文章的细节，而是把当前 benchmark 专题里最关键的 6 篇笔记串成一条主线，回答一个更“全局”的问题：**如果你想理解具身系统到底该怎么被定义、诊断、比赛、约束和评测，应该按什么顺序看？**

> **最后更新：2026-06-10。** 六层骨架不变。本次把 2026 年 4-6 月新进的 10 篇深度笔记综合进来：系统解法层出现「显式结构外挂」新范式，安全层从检测推进到运行时恢复，评测器层多了一条「分布 vs 点估计」的新张力——见第 8 节新增张力与第 12 节增量地图。

## 相关导读（本专题内）

- **BEHAVIOR-1K**：[`./behavior_1k_human_centered_embodied_ai_benchmark_omnigibson_2024.md`](./behavior_1k_human_centered_embodied_ai_benchmark_omnigibson_2024.md)
- **ENACT**：[`./enact_embodied_cognition_world_modeling_egocentric_interaction_2025.md`](./enact_embodied_cognition_world_modeling_egocentric_interaction_2025.md)
- **2025 BEHAVIOR Challenge 冠军方案**：[`./behavior_challenge_2025_first_place_solution_task_adaptation_vla_2025.md`](./behavior_challenge_2025_first_place_solution_task_adaptation_vla_2025.md)
- **IS-Bench**：[`./is_bench_interactive_safety_vlm_embodied_agents_household_tasks_2025.md`](./is_bench_interactive_safety_vlm_embodied_agents_household_tasks_2025.md)
- **WorldEval**：[`../world-model/worldeval_world_model_policy_evaluator_2025.md`](../world-model/worldeval_world_model_policy_evaluator_2025.md)
- **Ctrl-World × WorldArena**：[`../world-model/ctrl_world_worldarena_embodied_world_model_benchmark_2026.md`](../world-model/ctrl_world_worldarena_embodied_world_model_benchmark_2026.md)
- **World Model 主线总纲**：[`../world-model/world_model_mainline.md`](../world-model/world_model_mainline.md)
- **2026 年 4-6 月新进笔记**：不在此逐篇罗列，按主题归入第 12 节增量地图（代表作链接在各判断处给出）。

---

## 0. 全局定位卡

| 维度 | 内容 |
|---|---|
| 所属位置 | `theory/planning/` 下的专题总纲页 |
| 上游入口 | [`./README.md`](./README.md)、[`../../README.md`](../../README.md) |
| 平行主线 | [`../vla-core/vla_research_mainline.md`](../vla-core/vla_research_mainline.md) |
| 相邻页面 | [`../foundation/evaluation.md`](../foundation/evaluation.md)、[`../benchmark_tracker.md`](../benchmark_tracker.md)、[`../foundation/paper_index.md`](../foundation/paper_index.md)、[`../foundation/literature_review.md`](../foundation/literature_review.md)、[`../world-model/world_model_mainline.md`](../world-model/world_model_mainline.md) |
| 最适合谁看 | 想先抓整体 benchmark 结构，再决定读哪篇单文的人 |
| 最重要用途 | 导航页、复述稿、专题总图 |

这页在全局 `vla-handbook` 里的定位，不是和 [`../vla-core/vla_research_mainline.md`](../vla-core/vla_research_mainline.md) 平级竞争“谁是总主线”，而是：

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

`vla-handbook` 里已经有一条更偏“模型能力增长”的研究主线：[`../vla-core/vla_research_mainline.md`](../vla-core/vla_research_mainline.md)。

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

- [`../vla-core/vla_research_mainline.md`](../vla-core/vla_research_mainline.md) 更像“怎么把 agent 做强”
- `benchmark_mainline.md` 更像“怎么知道 agent 到底强在哪、弱在哪、安不安全、值不值得信”

所以这页不是另起炉灶，而是在全局手册里补上 **评估与约束轴**。

**两轴正在合流（2026-06 补充）**：4 月时这两条轴还能干净地分开，5-6 月的一批工作表明「能力轴」开始直接消费「评测轴」的产出——诊断层暴露的认知缺陷正被内化成训练目标。三个代表：[VEGA](vega_visual_encoder_grounding_alignment_for_spatially_aware_dissection.md) 把 3D 空间感知在视觉编码器输出层用对齐损失注入（推理零开销，免去 LLM 层的经验性层搜索）；[LARA](lara_latent_action_representation_alignment_for_vision_langu_dissection.md) 让潜在动作模型与 VLA 双向对齐、联合共进化（数据受限设置下 SIMPLER +16.8%，但对已经很强的基座只剩 +1.3%——对齐红利随基座变强而递减）；[LaST-R1](last_r1_reinforcing_robotic_manipulation_via_adaptive_physic_dissection.md) 更进一步，把 RL 奖励直接作用于潜在推理嵌入而非只优化动作（LIBERO 单轨迹 warm-up 达 99.9%）——**评测信号（reward）开始直接雕刻模型的「思考方式」**。这意味着 benchmark 主线不再只是「事后评」，它的信号正在变成训练时的一级输入。

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
- 通用评估协议细节：建议去看 [`../foundation/evaluation.md`](../foundation/evaluation.md)
- 持续榜单追踪：建议去看 [`../benchmark_tracker.md`](../benchmark_tracker.md)
- 大范围论文检索：建议去看 [`../foundation/paper_index.md`](../foundation/paper_index.md) 与 [`../foundation/literature_review.md`](../foundation/literature_review.md)

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
3. [`WorldEval`](../world-model/worldeval_world_model_policy_evaluator_2025.md)

这条线更像 deployment 视角：

- 系统先得能跑
- 然后得安全
- 最后还得有便宜、可扩展的评测器

### C) 你想从“world model 到底值不值得信”入手

建议顺序：

1. [`ENACT`](./enact_embodied_cognition_world_modeling_egocentric_interaction_2025.md)
2. [`WorldEval`](../world-model/worldeval_world_model_policy_evaluator_2025.md)
3. [`Ctrl-World × WorldArena`](../world-model/ctrl_world_worldarena_embodied_world_model_benchmark_2026.md)

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

### 5.5 `risk detection` 不等于 `risk recovery`（2026-06 新增）

4 月时 IS-Bench 层的判断是「把安全从终局检查推进到过程检查」；5 月的 [TAIL-Safe](tail_safe_task_agnostic_safety_monitoring_for_imitation_lear_dissection.md) 表明过程检查仍不够——检测到危险之后还得能**拉回来**。它用三个任务无关指标（可见性/可识别性/可抓取性）+ Lipschitz 连续 Q 函数 + 在线梯度上升，给任意已训练的 IL/VLA 策略加装即插即用的恢复层，不改策略、不需动力学模型。但要看清边界：安全保证是**经验性的**（非 CBF/HJ 级形式化），且只验证了单臂桌面抓取——「benchmark 里的安全分数」与「部署协议级的安全保证」之间还差形式化这一步。

### 5.6 `选对 backend` 不等于 `规划稳`（2026-06 新增）

4 月时 5.4 的张力停在「视频更真不代表 evaluator 更可靠」；[ManiDreams](manidreams_dissection.md) 把这条张力再推进一层：**即使 evaluator/world model 选对了，只要它输出单一确定轨迹，下游规划器仍会被「虚假确定性」骗过去**。它的控制变量 ablation 很干净——同一 backend 同一约束，只把采样实例数从 1 提到 16，成功率 58%→86%（+28pp）。判断：不确定性传播是规划器的责任，不是 backend 的责任；这把「world model 三派之争」（仿真/视频/JEPA）部分消解成了工程权衡。保留怀疑：实测只覆盖 2 种 backend，「三派都能挂」目前是设计目标而非已验证事实。

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
  - *进展（2026-06）*：已部分补上——系统解法层在 4-6 月出现一批结构化方案（见第 12 节），但「leaderboard 压力下的完整 system recipe」案例仍只有一篇。
- **interactive safety / policy governance**：把 safety 从 benchmark 推向 deployment protocol
  - *进展（2026-06）*：[TAIL-Safe](tail_safe_task_agnostic_safety_monitoring_for_imitation_lear_dissection.md) 正是这一步（运行时安全监控 + 恢复），但只给经验性保证；「形式化的部署协议」仍是空位。
- **world model evaluator / planner 系列**：继续追踪 evaluator 是否真正能替代部分真机回归
  - *进展（2026-06）*：[ManiDreams](manidreams_dissection.md) 补上了「分布传播」这一环（见 5.6）；「替代真机回归」本身仍未被任何一篇定量证明。
- **真实评测协议 / sim2real predictivity 系列**：把这条专题更明确接到 [`../foundation/evaluation.md`](../foundation/evaluation.md)
  - *进展（2026-06）*：仍空缺。值得注意的反例信号：[IVLR](thinking_in_text_and_images_interleaved_vision_language_reas_dissection.md) 与 [LARA](lara_latent_action_representation_alignment_for_vision_langu_dissection.md) 都是纯仿真或极少真机验证就给出强结论——sim2real predictivity 的缺位正在让这类强数字难以采信。

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

## 12. 2026 年 4-6 月增量：新证据落在哪一层？

这两个月新进 10 篇笔记。它们不改变六层骨架，但在三层上各形成一个可命名的趋势，并给「能力轴」带来一条合流判断（见第 4 节末）。按主题归纳，不逐篇罗列。

### 12.1 系统解法层：「显式结构外挂」成为新主流 ⚡

4 月时这一层只有一个 Challenge 冠军案例，结论停在「system recipe 不可避免」；5-6 月的证据让 recipe 的形态清晰起来：**赢的不是更大的端到端模型，而是给现成 VLA 外挂显式结构——给它更少、但更精准的输入**。三个代表，三种结构：

- [CodeGraphVLP](codegraphvlp_dissection.md)：持久语义图 + 一次合成的 Python 规划器，正面拆掉 VLA 的 Markovian 假设。π₀ 完全不改，历史相关长程任务平均 30%→81.7%，规划延迟比 VLM-in-loop 低 9 倍。这直接回应了 5.2 张力里「non-Markovian state 把系统狠狠干翻」的判断——答案不是堆 memory token，而是把状态显式写成图。
- [RAM](ram_dissection.md)（Science Robotics）⚡：把 NLP 的 RAG 范式搬到 manipulation——类别级 3D 物体模板库做成可检索的外部知识，不训 VLM、不训 3D 模型，14 项真机任务语言驱动 89.17%、多步 80%。证明了「外部知识库 + retrieval」是微调之外的第三条路；但模板靠人工标注（11 类起步），scale 没被证明容易。
- [IVLR](thinking_in_text_and_images_interleaved_vision_language_reas_dissection.md)：长程任务的中间表示之争有了消融级答案——交错「文本子目标 + 视觉关键帧」轨迹缓存后闭环执行，LIBERO-Long 从 37.7%（无轨迹）提到 92.4%，而 text-only 62.0%、vision-only 68.4%——**两种模态互补而非可替代**。代价是 10 秒前置推理与静态场景假设。

一句话判断：系统解法层正在从「比赛补丁」走向「结构化外挂范式」——共同信条是底层 VLA 不动，改变它看到什么、被告知什么。

### 12.2 部署可行性进入评测话语 🔧

evaluator 层 4 月时只问「能不能便宜地评」；现在「能不能便宜地跑」成为同级问题。[MolmoAct2](molmoact2_action_reasoning_models_for_real_world_deployment_dissection.md) 用 Per-Layer KV 桥接 + Flow Matching 专家把开源 VLA 的「开源/性能/延迟不可能三角」打破：推理 6700ms→180ms（37 倍），7 个基准上超过闭源 π0.5，且权重+数据+代码全开源——开源基线第一次在「可部署」维度追平闭源。更底层的，[OMP](../diffusion-flow/omp_one_step_meanflow_policy_with_directional_alignment_dissection.md)（ICML 2026）把一步推理策略的三大病理（谱偏置/梯度饥饿/内存爆炸）做了理论诊断并给出方向对齐解法，NFE=1、约 6.8ms。判断：当开源系统可复现、推理进入实时区间，benchmark 数字的可信度与可比性都会被抬高——这对评测轴是结构性利好。

### 12.3 安全层与评测器层：见 5.5 / 5.6 新张力

[TAIL-Safe](tail_safe_task_agnostic_safety_monitoring_for_imitation_lear_dissection.md)（检测→运行时恢复）与 [ManiDreams](manidreams_dissection.md)（点估计→分布传播）分别把第 4 层和第 5-6 层向前推了一步，详见 5.5 与 5.6，此处不重复。

### 12.4 增量后的主线一句话

**4 月版本的结论是「定义→诊断→解法→安全→评测」五步闭环；6 月版本的修正是：解法层有了可命名的范式（显式结构外挂），安全层多了恢复一环，评测层多了分布一环——闭环没有被推翻，而是每一环都更厚了。**

---
[← Back to Benchmark Hub](./README.md)
[← Back to Theory](../../README.md)
