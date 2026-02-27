# VLA Handbook

[![CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
![Auto-updated](https://img.shields.io/badge/内容-每日自动更新-blue)

**全中文 · 工程实战导向的 VLA 知识库。** 理论推导 + 代码级拆解 + 真机部署 + 论文自动追踪，一个仓库覆盖从入门到落地的完整路径。

> 79 理论文档 · 9+ Feb 2026 深度解析 · 每日自动 pipeline（⚡ 论文评分 · 深度拆解 · 社交情报）

---

## 三句话说清楚这个 Handbook 的价值

1. **不只是摘要**：每篇给出入口脚本、关键超参、shape 口径的 sanity-check——"看懂"和"跑通"之间的坑，都标出来了。
2. **Robotics 的脏活写清楚**：多模态同步、Sim2Real 断点、动作空间对齐、触觉/灵巧手硬件选型——这些在其他地方要么没有，要么藏在论文附录里。
3. **活的知识库**：自动 pipeline 每天抓最新 VLA 论文（⚡/🔧/📖 评级），精选后生成深度解析写入仓库，不是六个月没人维护的静态文档。

---

## 先看这几篇（30 分钟内建立正确框架）

| # | 文档 | 你会得到什么 | 时间 |
|---|---|---|---|
| ① | [`theory/README.md`](./theory/README.md) | 学习路线图：把"要学什么"降维成可走的路径 | 10–20 min |
| ② | [`theory/pi0_flow_matching.md`](./theory/pi0_flow_matching.md) | Flow Matching 原理 + 工程折中，可直接迁移到 VLA action head | 15–30 min |
| ③ | [`theory/spirit_v1_5_dissection.md`](./theory/spirit_v1_5_dissection.md) | Qwen3-VL + DiT + ODE/Euler 端到端复现入口，代码级拆解 | 20–40 min |
| ④ | [`theory/unifolm_vla_0_unitree_2026.md`](./theory/unifolm_vla_0_unitree_2026.md) | 开源训练栈：数据管线 + 部署要点 + 30 分钟验收清单 | 30–60 min |
| ⑤ | [`deployment/README.md`](./deployment/README.md) | 真机落地总入口：硬件选型 · 多模态同步 · 调参 checklist | 按需查阅 |
| ⑥ | [`theory/world_action_models_are_zero_shot_policies_dissection.md`](./theory/world_action_models_are_zero_shot_policies_dissection.md) | World Action Model 如何做到零样本策略迁移（Feb 2026 精选） | 20 min |

---

## 快速导航

| 目标 | 入口 | 说明 |
|---|---|---|
| 补理论 / 刷面试 | [`theory/README.md`](./theory/README.md) | 路线图 + 核心概念索引 |
| 找论文 / 做综述 | [`theory/paper_index.md`](./theory/paper_index.md) · [`theory/literature_review.md`](./theory/literature_review.md) | 多维索引 + 发展史全景图 |
| 真机落地 | [`deployment/README.md`](./deployment/README.md) | 硬件选型 · 多模态同步 · Sim-to-Real |
| 公司 / 求职 | [`companies/README.md`](./companies/README.md) | 公司指南 + 产业报告 digest |
| 双周前沿报告 | [`reports/biweekly/README.md`](./reports/biweekly/README.md) | VLA / 触觉 / 人形 / 基准 · 含预测回顾 |
| 周报 + 风向洞察 | [`reports/weekly/README.md`](./reports/weekly/README.md) | 每周论文精选 + SOTA + 趋势 Layer 分析 |
| 最新论文解析 | [`theory/`](./theory/) + [`scripts/SCRIPTS.md`](./scripts/SCRIPTS.md) | 每日自动 pipeline ⚡ 精选深度拆解 |
| 变更记录 | [`CHANGELOG.md`](./CHANGELOG.md) | 从 git 历史提炼，避免 README 膨胀 |

---

## 项目结构

```
VLA-Handbook/
├── theory/          # 理论基础（79 个文档，含每日自动深度解析）
├── deployment/      # 真机与部署（硬件选型 · 多模态同步 · Sim-to-Real）
├── reports/
│   ├── biweekly/    # 双周推理报告（含 reflection 预测回顾）
│   └── weekly/      # 周报：论文精选 + SOTA + 趋势风向洞察
├── scripts/         # 自动化 pipeline（SCRIPTS.md 含完整 DAG）
├── question-bank/   # 面试题库与代码实战
├── companies/       # 机器人公司与求职指南
├── cheat-sheet/     # 速查表（时间线 · 核心公式）
└── book/            # 电子书版本
```

---

## Feb 2026 新增深度解析（自动 pipeline 生成）

> ⚡ = 战略必读  🔧 = 工程可用  每日持续更新中

| 论文 | 方向 | 链接 |
|---|---|---|
| World Action Models are Zero-shot Policies | 零样本泛化 | [→](./theory/world_action_models_are_zero_shot_policies_dissection.md) |
| TwinVLA: Data-Efficient Bimanual Manipulation | 双臂操作 | [→](./theory/twinvla_data_efficient_bimanual_manipulation_with_twin_singl_dissection.md) |
| Scaling Verification > Scaling Policy Learning | Test-Time Compute | [→](./theory/scaling_verification_can_be_more_effective_than_scaling_poli_dissection.md) |
| RoboGene: Diversity-Driven VLA Pre-training | 预训练数据 | [→](./theory/robogene_boosting_vla_pre_training_via_diversity_driven_agen_dissection.md) |
| Olaf-World: Latent Actions for Video WM | 视频世界模型 | [→](./theory/olaf_world_orienting_latent_actions_for_video_world_modeling_dissection.md) |
| MIND: Memory & Action Control Benchmark | World Model 评估 | [→](./theory/mind_benchmarking_memory_consistency_and_action_control_in_w_dissection.md) |
| Agent World Model: Infinity Synthetic Envs | 合成环境 | [→](./theory/agent_world_model_infinity_synthetic_environments_for_agenti_dissection.md) |
| CausalGDP: Causality-Guided Diffusion Policy | 因果 + 扩散策略 | [→](./theory/causalgdp_causality_guided_diffusion_policies_for_reinforcem_dissection.md) |
| TaCo: Tactile Codec Benchmark | 触觉数据 | [→](./theory/taco_a_benchmark_for_lossless_and_lossy_codecs_of_heterogene_dissection.md) |

---

## 最近在更新什么

- **每日论文解析**：`theory/` 目录（⚡ 评级论文自动拆解，每天更新）
- **周报**：[`reports/weekly/`](./reports/weekly/)（论文精选 + SOTA + 风向洞察）
- **双周前沿推理**：[`reports/biweekly/`](./reports/biweekly/)（技术收敛判断 + 可验证预测）
- **变更记录**：[`CHANGELOG.md`](./CHANGELOG.md)

---

## 贡献

欢迎提 Issue 和 PR：补论文解读 · 真机经验 · 面试题。见 [`CONTRIBUTING.md`](./CONTRIBUTING.md)。

## 许可证

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.zh)
