# 📏 Benchmark 与评测专题入口

> 目标：把 `benchmark / evaluator / benchmark-oriented analysis` 相关文章单独归档，形成一个清晰入口，避免与 `baseline / 模型拆解` 混放。

---

## 0. 分类边界（先说清楚）

**这里收什么：**

- benchmark suite / evaluator / leaderboard methodology
- 围绕“怎么评测具身系统”的分析文章
- 以 benchmark 本身为主角的环境、任务集、指标体系解读

**这里暂时不收：**

- 以模型或 policy baseline 为主角的文章
- 即使论文里包含 benchmark 结果，但核心贡献是模型设计，也仍放在原模型区

例如：

- **收录**：[`Benchmark 主线总纲`](./benchmark_mainline.md)、[`BEHAVIOR-1K`](./behavior_1k_human_centered_embodied_ai_benchmark_omnigibson_2024.md)、[`ENACT`](./enact_embodied_cognition_world_modeling_egocentric_interaction_2025.md)、[`2025 BEHAVIOR Challenge 冠军方案`](./behavior_challenge_2025_first_place_solution_task_adaptation_vla_2025.md)、[`IS-Bench`](./is_bench_interactive_safety_vlm_embodied_agents_household_tasks_2025.md)、[`Ctrl-World × WorldArena`](./ctrl_world_worldarena_embodied_world_model_benchmark_2026.md)、[`WorldEval`](./worldeval_world_model_policy_evaluator_2025.md)
- **不迁入**：[`SimVLA`](../simvla_simple_vla_baseline_robotic_manipulation_2026.md)（它更像模型 baseline 与 training recipe 分析）

---

## 1. 快速入口

| 类型 | 文件 | 一句话定位 |
| :--- | :--- | :--- |
| Guide | [`Benchmark 主线总纲`](./benchmark_mainline.md) | 把 6 篇 benchmark / evaluator 笔记串成一条“任务世界 -> 认知诊断 -> 系统解法 -> 安全约束 -> 世界模型评测器”的总线 |
| Hub | [`BEHAVIOR-1K × OmniGibson`](./behavior_1k_human_centered_embodied_ai_benchmark_omnigibson_2024.md) | 人类需求驱动的 household benchmark，强调复杂状态与真实任务逻辑 |
| Hub | [`ENACT`](./enact_embodied_cognition_world_modeling_egocentric_interaction_2025.md) | 用 egocentric interaction world modeling 测 VLM 是否真的理解动作-状态-观察链条 |
| Case | [`2025 BEHAVIOR Challenge 冠军方案`](./behavior_challenge_2025_first_place_solution_task_adaptation_vla_2025.md) | benchmark 压力下的 system recipe：Pi0.5 adaptation + stage tracking + correction rules |
| Hub | [`IS-Bench`](./is_bench_interactive_safety_vlm_embodied_agents_household_tasks_2025.md) | 交互式 safety benchmark：测风险识别与 mitigation 是否在正确时机发生 |
| Hub | [`Ctrl-World × WorldArena`](./ctrl_world_worldarena_embodied_world_model_benchmark_2026.md) | 把世界模型评测从“视频像不像”推进到“能不能真干活” |
| Hub | [`WorldEval`](./worldeval_world_model_policy_evaluator_2025.md) | 用世界模型替代部分真机评测，做策略筛选与低成本回归 |
| Tracker | [`benchmark_tracker.md`](../../benchmark_tracker.md) | 自动化 benchmark/SOTA 追踪表，保持根目录路径不变 |

---

## 2. 推荐阅读顺序

### A) 你想理解“家务机器人 benchmark 为什么比常见任务难”

- 先看 [`Benchmark 主线总纲`](./benchmark_mainline.md)
- 先看 [`BEHAVIOR-1K`](./behavior_1k_human_centered_embodied_ai_benchmark_omnigibson_2024.md)
- 再看 [`ENACT`](./enact_embodied_cognition_world_modeling_egocentric_interaction_2025.md)，理解这些任务对 world modeling / memory 提出了什么认知要求
- 再看 [`2025 BEHAVIOR Challenge 冠军方案`](./behavior_challenge_2025_first_place_solution_task_adaptation_vla_2025.md)，理解在 leaderboard 压力下哪些 system trick 真能换分
- 再看 [`IS-Bench`](./is_bench_interactive_safety_vlm_embodied_agents_household_tasks_2025.md)，理解“能做事”和“能安全地做事”之间的差距
- 最后回看 [`benchmark_tracker.md`](../../benchmark_tracker.md)，区分“榜单分数”与“真实任务难度”

### B) 你想理解“世界模型怎么被评测”

- 先看 [`Benchmark 主线总纲`](./benchmark_mainline.md)
- 先看 [`ENACT`](./enact_embodied_cognition_world_modeling_egocentric_interaction_2025.md)，理解 VLM 是否具备交互世界建模能力
- 再看 [`2025 BEHAVIOR Challenge 冠军方案`](./behavior_challenge_2025_first_place_solution_task_adaptation_vla_2025.md)，理解这些能力缺口在真实执行系统里会怎样爆炸
- 再看 [`IS-Bench`](./is_bench_interactive_safety_vlm_embodied_agents_household_tasks_2025.md)，理解为什么 task success 远不等于 safe success
- 再看 [`WorldEval`](./worldeval_world_model_policy_evaluator_2025.md)
- 最后看 [`Ctrl-World × WorldArena`](./ctrl_world_worldarena_embodied_world_model_benchmark_2026.md)

### D) 你想理解“具身系统上线前最后一道关卡是什么”

- 先看 [`Benchmark 主线总纲`](./benchmark_mainline.md)
- 先看 [`2025 BEHAVIOR Challenge 冠军方案`](./behavior_challenge_2025_first_place_solution_task_adaptation_vla_2025.md)
- 再看 [`IS-Bench`](./is_bench_interactive_safety_vlm_embodied_agents_household_tasks_2025.md)
- 最后回看 [`ENACT`](./enact_embodied_cognition_world_modeling_egocentric_interaction_2025.md)，区分“认知缺口”与“安全缺口”

### C) 你想找“baseline 与 benchmark”的边界

- benchmark 入口看本页
- baseline 分析看 [`SimVLA`](../simvla_simple_vla_baseline_robotic_manipulation_2026.md)

---

[← Back to Theory](../../README.md)
