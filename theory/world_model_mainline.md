# World Model 主线总纲：从 evaluator 到 planner，再到 world action model

> 这页的目标不是重复解释某一篇 world model 论文怎么做，而是把 `WorldEval`、`Ctrl-World × WorldArena`、`DreamZero` 这些分散在手册里的条目串成一条更清楚的支线，回答一个更高层的问题：**world model 在 VLA / embodied AI 里，正在从“视频模型”演化成什么？**

## 相关导读（仓库内）

- **Benchmark 主线总纲**：[`./frontier/benchmarks/benchmark_mainline.md`](./frontier/benchmarks/benchmark_mainline.md)
- **WorldEval**：[`./frontier/benchmarks/worldeval_world_model_policy_evaluator_2025.md`](./frontier/benchmarks/worldeval_world_model_policy_evaluator_2025.md)
- **Ctrl-World × WorldArena**：[`./frontier/benchmarks/ctrl_world_worldarena_embodied_world_model_benchmark_2026.md`](./frontier/benchmarks/ctrl_world_worldarena_embodied_world_model_benchmark_2026.md)
- **DreamZero**：[`./dreamzero_world_action_models_zero_shot_policies_2026.md`](./dreamzero_world_action_models_zero_shot_policies_2026.md)
- **VLA 研究主线梳理**：[`./vla_research_mainline.md`](./vla_research_mainline.md)
- **评估体系**：[`./evaluation.md`](./evaluation.md)

---

## 0. 全局定位卡

| 维度 | 内容 |
|---|---|
| 所属位置 | `theory/` 下的 world model 支线总纲 |
| 上游入口 | [`./README.md`](./README.md) |
| 平行主线 | [`./vla_research_mainline.md`](./vla_research_mainline.md)、[`./frontier/benchmarks/benchmark_mainline.md`](./frontier/benchmarks/benchmark_mainline.md) |
| 相邻页面 | [`./evaluation.md`](./evaluation.md)、[`./benchmark_tracker.md`](./benchmark_tracker.md) |
| 最适合谁看 | 已经知道 VLA / benchmark 基本框架，想进一步理解 world model 在系统里到底扮演什么角色的人 |
| 最重要用途 | 支线导航页、综述稿、面试复述稿 |

这页在全局 handbook 里的定位，不是“又一条总主线”，而是：

- 把 `world model` 相关条目从 benchmark 专题里再抽出一层
- 形成一条 **evaluator -> planner -> WAM** 的技术支线
- 帮你区分：
  - 哪些工作把 world model 当评测器
  - 哪些工作把它当想象环境
  - 哪些工作进一步把它升级成 action model / policy 本体

---

## 1. 这条主线到底在回答什么？

world model 在机器人里，过去常常被说成一句很空的话：

- “让模型学会预测未来”

但真正落到系统里，关键问题其实是：

- 它是拿来 **评策略** 的？
- 还是拿来 **替代环境 rollout** 的？
- 还是拿来 **生成数据 / 动作** 的？
- 它最后是辅助 policy，还是干脆开始吞掉 policy 的角色？

这条 world model 主线，就是围绕这几个问题展开的。

它的一个更完整表述是：

```text
先把 world model 当 evaluator
  -> 再把它当 environment proxy / planner
  -> 最后把它推进到 world action model，
     让“预测世界”和“预测动作”合一
```

---

## 2. 一张总图：world model 支线地图

```text
┌──────────────────────────────────────────────┐
│ 1) Evaluator 起点：WorldEval                │
│    “能不能先用 world model 排策略、筛版本？”│
└──────────────────────┬───────────────────────┘
                       │ 先解决 ranking proxy 问题
                       v
┌──────────────────────────────────────────────┐
│ 2) 统一评测口径：WorldArena                 │
│    “world model 不只比像不像，还要比功能。”│
└──────────────────────┬───────────────────────┘
                       │ 把 evaluator / planner / data engine
                       │ 写成统一 benchmark
                       v
┌──────────────────────────────────────────────┐
│ 3) 更强模型路线：Ctrl-World                 │
│    “如果 world model 足够 controllable，   │
│      它就更像一个可用 imagination env。”   │
└──────────────────────┬───────────────────────┘
                       │ 从评测器走向 policy-in-the-loop
                       v
┌──────────────────────────────────────────────┐
│ 4) 更激进终点：DreamZero / WAM             │
│    “world model 不只辅助 policy，          │
│      而是开始和 action model 融成一体。”   │
└──────────────────────────────────────────────┘
```

一句话总结：

**WorldEval 在问“能不能先评”，WorldArena 在问“到底该怎么统一评”，Ctrl-World 在答“什么样的 world model 更像可用环境”，DreamZero 则进一步问“既然都能预测世界了，为什么不顺手把动作也一起学掉？”**

### 再看一张：角色升级图

```text
阶段 A：world model = 离线评测器

policy checkpoints ──> world model rollouts ──> ranking / filtering


阶段 B：world model = 环境代理

policy ──> imagined rollout ──> candidate actions / synthetic data
   │
   └──────────────────────> real env verification


阶段 C：world model = 动作生成基底

observation + goal
        │
        v
  world action model
   ├─ predict future video
   └─ predict action chunk
        │
        v
     closed-loop control
```

---

## 3. 四个节点一览

| 层级 | 文章 | 它在回答什么问题 | 关键贡献 |
|---|---|---|---|
| 1 | `WorldEval` | world model 能否做 policy ranking proxy？ | 把 world model evaluator 做成可操作系统 |
| 2 | `WorldArena` | world model 应该如何统一评测？ | 把 `Data Engine / Policy Evaluator / Action Planner` 写进 benchmark |
| 3 | `Ctrl-World` | 什么样的 world model 更适合 policy-in-the-loop？ | multi-view + frame-level action conditioning + memory retrieval |
| 4 | `DreamZero` | world model 是否会吞掉 action model？ | 把 WAM 写成联合预测 video + action 的主监督框架 |

---

## 4. 这条主线为什么成立？

### 4.1 它不是“视频模型发展史”，而是“功能角色演化史”

如果只按模型看，很容易把这些工作误读成：

- 更大的 video diffusion
- 更好的 controllability
- 更长的 rollout

但按功能角色看，它们的真正演化是：

1. **先拿来评估**  
2. **再拿来当环境代理**  
3. **再拿来辅助规划和生成数据**  
4. **最后开始和动作模型边界模糊**

这比“视频更真了”要重要得多。

### 4.2 它揭示了一个越来越清晰的趋势

world model 在 embodied AI 里，正在从：

```text
nice-to-have future predictor
```

变成：

```text
evaluator / proxy environment / data engine / planner / action model substrate
```

也就是说，它已经不只是一个“训练时辅助信号”，而开始进入系统核心。

### 4.3 一张系统插槽图

```text
传统 VLA 闭环

obs ──> VLA policy ──> action ──> real world ──> next obs


加上 evaluator 之后

obs ──> VLA policy ──> action ──> real world ──> next obs
          │
          └────────────> world model evaluator
                           └─ rank / diagnose / screen


加上 proxy environment 之后

obs ──> policy ──> imagined rollout in world model ──> action plan
  │                                                  │
  └──────────────────────────── real world <─────────┘


走到 WAM 之后

obs + goal ──> WAM
               ├─ future video
               └─ action chunk
                    │
                    v
                 real world
```

---

## 5. 它和 benchmark 主线是什么关系？

这条 world model 支线，和 [`./frontier/benchmarks/benchmark_mainline.md`](./frontier/benchmarks/benchmark_mainline.md) 的关系可以这么理解：

- `benchmark_mainline` 更偏“评估与约束轴”
- `world_model_mainline` 更偏“评测器与想象环境轴”

两者的重叠点在：

- `WorldEval`
- `Ctrl-World × WorldArena`

但关注重点不一样：

| benchmark 主线 | world model 主线 |
|---|---|
| world model 是“评测体系的一层” | world model 是“系统功能角色本身” |
| 关心它怎么定义 benchmark / evaluator | 关心它怎么变成 evaluator / planner / WAM |
| 更偏结构与问题定义 | 更偏角色演化与技术路线 |

所以这页不是替代 benchmark 主线，而是从其中拎出一个重要分支单独展开。

---

## 6. 它和全局 VLA 主线是什么关系？

[`./vla_research_mainline.md`](./vla_research_mainline.md) 讲的是：

- baseline
- 数据规模化
- 感知增强
- 后训练 / recovery

而这页讲的是：

- 当 agent 变强以后，world model 在整个闭环里开始承担什么角色

可以把两者看成：

| VLA 研究主线 | world model 主线 |
|---|---|
| 怎么把 policy 做强 | 怎么让系统拥有想象、评测与预测环境能力 |
| 数据 / 感知 / 后训练 | evaluator / planner / WAM |
| 偏 agent 训练 | 偏 agent 外层能力与系统工具 |

一句话说：

- `vla_research_mainline` 更像“训练主线”
- `world_model_mainline` 更像“世界代理主线”

---

## 7. 三条推荐阅读路线

### A) 你想理解“world model 为什么先从 evaluator 起步”

建议顺序：

1. [`WorldEval`](./frontier/benchmarks/worldeval_world_model_policy_evaluator_2025.md)
2. [`Ctrl-World × WorldArena`](./frontier/benchmarks/ctrl_world_worldarena_embodied_world_model_benchmark_2026.md)

这条线适合先建立一个现实感：

- 真机评测太贵
- 所以 evaluator 是第一个最刚需的落点

### B) 你想理解“world model 如何从评测器变成环境代理”

建议顺序：

1. [`WorldEval`](./frontier/benchmarks/worldeval_world_model_policy_evaluator_2025.md)
2. [`Ctrl-World × WorldArena`](./frontier/benchmarks/ctrl_world_worldarena_embodied_world_model_benchmark_2026.md)
3. [`DreamZero`](./dreamzero_world_action_models_zero_shot_policies_2026.md)

这条线最能看到演化：

- evaluator
- proxy environment
- WAM / action-generating model

### C) 你想理解“DreamZero 到底新在哪”

建议顺序：

1. [`Ctrl-World × WorldArena`](./frontier/benchmarks/ctrl_world_worldarena_embodied_world_model_benchmark_2026.md)
2. [`DreamZero`](./dreamzero_world_action_models_zero_shot_policies_2026.md)

因为先理解：

- 为什么 world model 需要 controllability
- 为什么要看 functional utility

再去看 DreamZero 的 `video + action joint modeling`，会更清楚它到底是在接哪条线。

---

## 8. 这条主线里的三个核心张力

### 8.1 `good-looking video` 不等于 `good evaluator`

这是 `WorldEval -> WorldArena` 的核心张力。

- 视频真
- 不代表策略排序一定准

### 8.2 `good evaluator` 不等于 `good planner`

这是 `WorldEval -> Ctrl-World / WorldArena` 的核心张力。

- 能看出哪个 policy 强
- 不代表已经能稳定承担闭环 planning

### 8.3 `world model` 不等于 `auxiliary module`

这是 `Ctrl-World -> DreamZero` 的核心张力。

- 一开始它只是 evaluator / proxy
- 后来它越来越像 policy substrate 本身

也就是说，world model 的角色正在从：

```text
side module
  -> system tool
  -> core control substrate
```

---

## 9. 如果你要把这条线讲给面试官，怎么讲最顺？

一个稳的 30 秒版本是：

```text
world model 在机器人里最早常被当成未来预测器，
但真正有工程价值的第一步其实是 evaluator，
所以 WorldEval 先解决“能不能保持策略排序一致”；
随后 WorldArena 把 world model 的功能角色正式写成 benchmark，
Ctrl-World 则代表更强的 action-conditioned、policy-in-the-loop 路线，
而 DreamZero 更进一步，把 world model 和 action model 合成 WAM，
说明 world model 正在从评测工具演化成控制系统的一部分。
```

如果只允许你说一句话，那就说：

**这条 world model 主线，讲的是机器人系统如何从“预测未来”一步步走到“评估策略、代理环境、甚至生成动作”。**

---

## 10. 这条主线还缺什么？

当前这条支线已经有骨架，但还可以继续补：

- **更多 evaluator work**：继续追踪 world model ranking proxy 是否真的稳
- **更多 planner / data engine work**：补 world model 作为合成数据引擎的案例
- **更强 WAM / unified model work**：补 video+action 联合建模路线
- **真实协议页对接**：和 [`./evaluation.md`](./evaluation.md) 更明确连起来，回答“什么时候必须回真机”

---

## 11. 你在手册里该怎么用这页？

这页最适合三个用途：

1. **当 world model 导航页**：先看这页，再决定进 `WorldEval / Ctrl-World / DreamZero`
2. **当支线总纲**：把 scattered world model 条目串成结构
3. **当复述稿**：快速解释 world model 在具身系统里角色的演化

如果你已经读过其中单篇，再回来读这页，重点就不是“重学内容”，而是确认：

- 每篇到底在 world model 角色演化里补哪一层
- 哪个工作是在做 evaluator，哪个在做 planner，哪个已经在吞 action model
- 接下来补文献时，应该优先补哪类节点

---
[← Back to Theory](./README.md)
