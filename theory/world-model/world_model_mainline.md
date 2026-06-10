# World Model 主线总纲：从 evaluator 到 planner，再到 world action model

> 这页的目标不是重复解释某一篇 world model 论文怎么做，而是把 `WorldEval`、`Ctrl-World × WorldArena`、`DreamZero` 这些分散在手册里的条目串成一条更清楚的支线，回答一个更高层的问题：**world model 在 VLA / embodied AI 里，正在从”视频模型”演化成什么？**
>
> **最后更新：2026-06-10。** 本次更新综合 4-6 月新进的 13 篇文章：原有的 evaluator → planner → WAM 骨架仍然成立，但重心已移到两处——world model 作为 RL 仿真器的后训练实战（新增 §10），以及 world model 从”推理时模块”向”latent 空间 / 训练期监督”的收缩（新增 §11，与 4 月叙事存在显式张力）。

## 相关导读（仓库内）

- **Benchmark 主线总纲**：[`../planning/benchmark_mainline.md`](../planning/benchmark_mainline.md)
- **WorldEval**：[`./worldeval_world_model_policy_evaluator_2025.md`](./worldeval_world_model_policy_evaluator_2025.md)
- **Ctrl-World × WorldArena**：[`./ctrl_world_worldarena_embodied_world_model_benchmark_2026.md`](./ctrl_world_worldarena_embodied_world_model_benchmark_2026.md)
- **DreamZero**：[`./dreamzero_world_action_models_zero_shot_policies_2026.md`](./dreamzero_world_action_models_zero_shot_policies_2026.md)
- **WM 辅助 VLA 后训练综述（§10 骨架读物）**：[world_model_aided_vla_post_training_deep_dive_2026.md](world_model_aided_vla_post_training_deep_dive_2026.md)
- **VLA 研究主线梳理**：[`../vla-core/vla_research_mainline.md`](../vla-core/vla_research_mainline.md)
- **评估体系**：[`../foundation/evaluation.md`](../foundation/evaluation.md)

---

## 0. 全局定位卡

| 维度 | 内容 |
|---|---|
| 所属位置 | `theory/` 下的 world model 支线总纲 |
| 上游入口 | [`./README.md`](./README.md) |
| 平行主线 | [`../vla-core/vla_research_mainline.md`](../vla-core/vla_research_mainline.md)、[`../planning/benchmark_mainline.md`](../planning/benchmark_mainline.md) |
| 相邻页面 | [`../foundation/evaluation.md`](../foundation/evaluation.md)、[`../benchmark_tracker.md`](../benchmark_tracker.md) |
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

> **2026-06 注**：这张四节点骨架仍然成立——但它只覆盖了主线的前半段。4-6 月新进的工作主要落在两处：节点 3-4 之间的"world model 当 RL 仿真器"实战（§10），以及对节点 4"吞掉 policy"叙事本身的修正（§11）。

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

也就是说，它已经不只是一个”训练时辅助信号”，而开始进入系统核心。

**2026-06 更新：这个趋势在加速，并且开始外溢出机器人操作。** 同一套”world model 进系统核心”的范式正在邻域复刻：自动驾驶里 [CoWorld-VLA](coworld_vla_thinking_in_a_multi_expert_world_model_for_auton_dissection.md) 把世界知识拆成语义/几何/动态/轨迹四个专家 token 做 latent CoT，条件化扩散规划器，在 NAVSIM v1 单目单帧设定下拿到 PDMS 89.8 的 SOTA；材料工程里 [LEIA](leia_learned_environment_for_interactive_architected_materia_dissection.md) 用与机器人世界模型同构的 encode-process-decode + action conditioning 架构做实时应力/形变预测（比 FEM 快 100-300×）——架构同构性说明这是通用范式，不是机器人专利。场景生成侧，[Lyra 2.0](lyra2_dissection.md) 用 3D 缓存路由 + 自污染训练正面解决长轨迹的空间遗忘与时间漂移，生成的 3DGS/mesh 可直接导入 Isaac Sim——“scene as a service”开始可见，但”用生成场景训 VLA 是否真的缩小 sim2real gap”目前没有任何实证，这是该方向最大的悬置问题。

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

这条 world model 支线，和 [`../planning/benchmark_mainline.md`](../planning/benchmark_mainline.md) 的关系可以这么理解：

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

[`../vla-core/vla_research_mainline.md`](../vla-core/vla_research_mainline.md) 讲的是：

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

1. [`WorldEval`](./worldeval_world_model_policy_evaluator_2025.md)
2. [`Ctrl-World × WorldArena`](./ctrl_world_worldarena_embodied_world_model_benchmark_2026.md)

这条线适合先建立一个现实感：

- 真机评测太贵
- 所以 evaluator 是第一个最刚需的落点

### B) 你想理解“world model 如何从评测器变成环境代理”

建议顺序：

1. [`WorldEval`](./worldeval_world_model_policy_evaluator_2025.md)
2. [`Ctrl-World × WorldArena`](./ctrl_world_worldarena_embodied_world_model_benchmark_2026.md)
3. [`DreamZero`](./dreamzero_world_action_models_zero_shot_policies_2026.md)

这条线最能看到演化：

- evaluator
- proxy environment
- WAM / action-generating model

### C) 你想理解“DreamZero 到底新在哪”

建议顺序：

1. [`Ctrl-World × WorldArena`](./ctrl_world_worldarena_embodied_world_model_benchmark_2026.md)
2. [`DreamZero`](./dreamzero_world_action_models_zero_shot_policies_2026.md)

因为先理解：

- 为什么 world model 需要 controllability
- 为什么要看 functional utility

再去看 DreamZero 的 `video + action joint modeling`，会更清楚它到底是在接哪条线。

### D) 你想跟上 2026 年 4-6 月的两条新主线

建议顺序：

1. [WM 辅助 VLA 后训练综述](world_model_aided_vla_post_training_deep_dive_2026.md) —— 先拿到"AC-WM 三大硬伤 + Co-evolution 范式"这张地图
2. [World-VLA-Loop](world_vla_loop_closed_loop_learning_of_video_world_model_and_dissection.md) 或 [Sword](sword_style_robust_world_models_as_simulators_via_dynamic_la_dissection.md) —— 看 RL 仿真器路线的两篇代表实作
3. [LDA-1B](lda_1b_dissection.md) —— 看"像素 vs latent"这条 scaling 分水岭
4. [GaussianDream](gaussiandream_a_feed_forward_3d_gaussian_world_model_for_rob_dissection.md) —— 看"世界模型不在推理时运行"的极端形态

前两步对应 §10，后两步对应 §11。

---

## 8. 这条主线里的三个核心张力

### 8.1 `good-looking video` 不等于 `good evaluator`

这是 `WorldEval -> WorldArena` 的核心张力。

- 视频真
- 不代表策略排序一定准

**2026-06 实证补强**：[Dream.exe](dreamexe_can_video_generation_models_dream_executable_robot_dissection.md) 把这条张力推到极致——把 8 个视频生成模型生成的操作视频提取成 3D 轨迹放进 MuJoCo 直接执行，发现视觉质量与物理可执行性几乎是两个独立维度（视觉评分最高的模型在执行中经常失败）。evaluator 这条线由此从"能不能保持策略排序"升级到"生成内容能不能物理执行"。同方向上 [SWEET](sweet_sparse_world_modeling_with_image_editing_for_embodied_dissection.md) 给出镜像证据：密集视频生成不仅慢 40 倍，关键帧质量反而不如图像编辑模型——"更逼真的视频"在两个独立实验里都没有换来"更有用的世界模型"。

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

### 8.4 `进入系统核心` 不等于 `在推理时运行`（2026-06 新增张力）

这是 4 月叙事与 5-6 月证据之间最重要的张力。4 月时我们判断 world model 正走向 core control substrate，隐含图景是"推理时运行的世界模型越来越中心"。但 5-6 月的一波工作（GaussianDream / MoLA / SWEET / LDA-1B，详见 §11）表明：world model 可以以**训练期监督、latent 动力学、前缀表征**的形式"溶解"进 policy，推理时根本不运行。两种"进入核心"的方式目前并行，谁是终局未定。

### 8.5 `想象出的仿真器` 不等于 `可靠的仿真器`（2026-06 新增张力）

RL 仿真器路线（§10）的核心张力：AC-WM 可以生成视觉上合理的 rollout，却预测出物理错误的状态转移、甚至虚假的成功信号——策略可能学会 hacking 世界模型而非完成任务。当前所有解法（强 action condition / 限制 rollout 步数 / Co-evolution 迭代）都是缓解而非根治。

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

**这条 world model 主线，讲的是机器人系统如何从”预测未来”一步步走到”评估策略、代理环境、甚至生成动作”。**

如果面试官追问 2026 年中的最新进展，再补两句：

```text
4-6 月的重心移到两处——
一是用世界模型当 RL 仿真器给 VLA 做后训练，
   闭环 co-evolution 成为标准范式，但净收益缺对照实验；
二是世界模型本身在”收缩”：
   从像素退到 latent（LDA-1B 的 20% vs 55% ablation），
   从推理期退到训练期（GaussianDream 只留一个前缀）。
```

---

## 10. 新主战场：world model 作为 RL 仿真器，后训练这条线在 4-6 月成型

4 月写这页时，”planner / proxy environment”还只是地图上的一个格子；4-6 月它被一整波工作填满，并收敛成一个明确的工程范式：**用 action-conditioned world model（AC-WM）替代真机和物理模拟器，给 VLA 做 RL 后训练**。

骨架读物是 [World Model 辅助 VLA 后训练综述](world_model_aided_vla_post_training_deep_dive_2026.md)。它把这条线的动机（真机 RL 贵，自动驾驶/全身运控等场景根本不可行）、**三大硬伤**（精细控制失真 / 自回归误差累积 / VLA 探索到 OOD 的 state-action）和当前最优范式（**Co-evolution 迭代**：真机 rollout → 微调 WM → WM 内 RL → 再真机）讲清楚了，并提出全线最尖锐的问题——“像 RL Token 那样学一个足够好的 Q(s,a) 是不是就够了？”换句话说：这条线虽然热，**”WM 辅助 RL vs 纯真机 RL”的净收益几乎没人做对照实验**。

两篇代表作分别攻一个硬伤：

- [World-VLA-Loop](world_vla_loop_closed_loop_learning_of_video_world_model_and_dissection.md)：攻奖励信号——把奖励头直接长在视频世界模型的扩散潜空间里（不外挂 VLM 判分），并用”成功+近成功”（SANS）数据迫使模型学会成功/失败的精细边界；闭环迭代 2 轮后真实任务成功率显著提升（Place Cup 13.3%→36.7%）。但奖励对齐率约 87%，剩下的 13% 就是 reward hacking 的空间。
- [Sword](sword_style_robust_world_models_as_simulators_via_dynamic_la_dissection.md)：攻 OOD 与 exposure bias——结构引导风格增强（深度/分割/任务先验约束下的风格迁移）让模型”别无选择只能学动力学”，动态潜在自举（DLB）让训练时就开始用自己的预测做条件、对齐推理分布。仅在 LIBERO-Spatial 验证，但 DLB 机制对任何自回归世界模型通用。

**判断**：这条线已从”能跑”走向”可靠的重工程”，但三大硬伤都只有 mitigation 没有根治；最现实的落地是混合模式——能做真机 RL 就做真机（精细操作尤其如此），不能做真机的场景（驾驶/全身运控）WM 辅助是唯一出路。把”想象 RL”当 plug-and-play 的预期应该降温。

---

## 11. 另一股收缩力：world model 从”运行时模块”退成”训练期老师”与”latent 空间”

4 月的主线叙事是 world model 一路升级、最终”吞掉 policy”（WAM）。5-6 月出现了一股几乎反方向的力：**多篇工作把 world model 的价值从推理时挪到训练时、从像素空间挪到 latent 空间**——它确实进入了系统核心，但进入方式是”被压缩、被溶解”，而不是”被运行”。三个层面的收缩：

1. **表征空间收缩：像素 → latent。** [LDA-1B](lda_1b_dissection.md) 用一组直接的 ablation 把 pixel-VAE 世界模型路线打到墙上：相近设置下 VAE 路线 20.0% vs DINO 隐空间 55.4%（RoboCasa-GR1）。其 policy / forward / inverse / visual-forecast 四 head 设计还让髒数据与无动作视讯”按质分工”进入训练——加 30% 髒数据让 π0.5 掉 10-20pp，却让 LDA 涨 10pp。它对”VLA vs 世界模型哪条路对”的回答是：假二分，真问题是**在哪个空间共学它们**。[DUST](dual_stream_diffusion_for_world_model_augmented_vision_langu_dissection.md)（ICML 2026）从架构侧补全这条线：双流 MMDiT + 解耦噪声调度，让同一个模型在不同噪声组合下分别学前向/逆向动力学，世界模型目标同样是语义 embedding 而非像素（Franka 真机 +10.4%）。
2. **运行时收缩：推理期 → 训练期。** [GaussianDream](gaussiandream_a_feed_forward_3d_gaussian_world_model_for_rob_dissection.md) 训练期跑完整的 3D 高斯重建 + 未来预测做密集监督，推理期全部丢弃、只保留一个 1024-token 前缀（真实机器人 +15.6pp over π0.5）——世界模型从推理时的”模拟器”变成训练时的”老师”。[MoLA](from_imagined_futures_to_executable_actions_mixture_of_laten_dissection.md) 则在”想象→执行”之间插一层三模态（语义/深度/光流）逆动力学翻译层，出发点同样是承认”好看的未来帧 ≠ 有用的动作信号”。
3. **计算量收缩：密集 → 稀疏。** [SWEET](sweet_sparse_world_modeling_with_image_editing_for_embodied_dissection.md) 用图像编辑模型做稀疏关键帧预测，比密集视频生成快 40 倍（10s vs 400s）且关键帧质量更高——操作任务的本质是几个语义里程碑，不需要中间帧。[WorldKV](worldkv_efficient_world_memory_with_world_retrieval_and_comp_dissection.md) 则发现自回归世界模型的 KV cache 本身已是世界记忆，训练无关的检索+压缩就能在约 2× 吞吐下保住长程一致性（在 Matrix-Game-2.0 上甚至超过 Full KV，因为过滤掉了 OOD 退化的旧 cache）。

**与 4 月判断的张力**（见 §8.4）：4 月时判断 world model 正走向 core control substrate；5-6 月的证据并未推翻它，而是修正了”core”的含义——**进入核心不等于在线运行**。WAM 路线（DreamZero → LDA-1B）与”溶解”路线目前并行，而 LDA-1B 恰好横跨两者：它是 WAM 想法的 scaling 落地，但落地的前提正是放弃像素空间。两条路线最终是否合流（一个 latent 空间里既做 WAM 又能按需”瘦身”成前缀），是接下来 6 个月这条主线最值得盯的问题。

---

## 12. 这条主线还缺什么？（2026-06 盘点）

4 月列出的四个缺口，两个月后的状态：

- **更多 evaluator work**：继续追踪 world model ranking proxy 是否真的稳 → **部分填上**——Dream.exe 把评测推进到”物理可执行性”直接测试（§8.1），但全部在 MuJoCo 仿真，真机闭环验证仍缺。
- **更多 planner / data engine work**：补 world model 作为合成数据引擎的案例 → **已成主战场**（§10），但”WM 辅助 RL vs 纯真机 RL”的净收益对照实验仍然没人做——这是当前整条主线最大的可证伪空白。
- **更强 WAM / unified model work**：补 video+action 联合建模路线 → **已填上**——LDA-1B / DUST 给出分水岭判断（latent 空间），但 LDA-1B 的 checkpoint 尚未开源，独立复现待验证。
- **真实协议页对接**：和 [`../foundation/evaluation.md`](../foundation/evaluation.md) 更明确连起来，回答”什么时候必须回真机” → **仍缺**。

新增缺口（2026-06）：

- **长程误差累积没有根治**：限制 rollout 步数 / 关键帧起步 / chunk-level rollout / DLB 都是缓解；显式不确定性量化或周期性”reset 回真实数据”仍是空白。
- **生成场景的 sim2real 实证为零**：Lyra 2.0 类工作展示了”机器人能在生成场景里走”，但没人测过”用生成场景训的 VLA 真机表现”。
- **”溶解”路线 vs WAM 路线缺正面对比**：同数据同算力下，训练期世界模型前缀和推理期 WAM 谁的收益高，没有实验回答。

---

## 13. 你在手册里该怎么用这页？

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
