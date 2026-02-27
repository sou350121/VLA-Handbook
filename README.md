# VLA Handbook（Vision-Language-Action）

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

> **VLA（Vision-Language-Action）领域的结构化知识库与工程实战手册。**
> 覆盖理论基础、模型解析、真机部署、论文索引与题库。

---

## 为什么这个 Handbook 值得关注（但不灌水）

- **以“可复现/可验收”为第一原则**：每篇尽量给出入口脚本、关键超参、shape/口径的 sanity-check（避免“看起来懂了，跑起来全错”）。  
- **把 Robotics 的“脏活”写清楚**：多模态同步、动作空间对齐、Sim2Real 断点、触觉/灵巧手硬件选型。  
- **持续更新，但不让 README 膨胀**：前沿用 Biweekly 回链，细节进深度笔记，改动进 Changelog。每日自动 pipeline（VLA 论文评分、社交情报、深度解析）持续写入内容，见 [`scripts/`](./scripts/SCRIPTS.md)。

## 先看这 5 篇（高信号精选）

- **① 先拿到“地图”（10–20 分钟）**：[`theory/README.md`](./theory/README.md)  
  - 你会得到：学习路线图 + 模块化索引（把“要学什么”降维成可走的路径）

- **② 把动作生成学到“能用”（15–30 分钟）**：[`theory/pi0_flow_matching.md`](./theory/pi0_flow_matching.md)  
  - 你会得到：Flow Matching 为什么更快、怎么采样、工程折中（可直接迁移到 VLA action head）

- **③ 看一次代码级拆解（20–40 分钟）**：[`theory/spirit_v1_5_dissection.md`](./theory/spirit_v1_5_dissection.md)  
  - 你会得到：Qwen3-VL + DiT + ODE/Euler 的端到端入口、关键超参与复现清单

- **④ 直连真实开源训练栈（30–60 分钟）**：[`theory/unifolm_vla_0_unitree_2026.md`](./theory/unifolm_vla_0_unitree_2026.md)  
  - 你会得到：数据管线/部署要点 + “30 分钟验收”清单（避免跑偏）

- **⑤ 真机落地总入口（按需查阅）**：[`deployment/README.md`](./deployment/README.md)  
  - 你会得到：硬件选型、多模态同步、控制与调参 checklist（把工程坑提前标出来）

- **⑥ 追前沿解析（每日自动更新）**：[`theory/world_action_models_are_zero_shot_policies_dissection.md`](./theory/world_action_models_are_zero_shot_policies_dissection.md)  
  - 你会得到：World Action Model 如何做到零样本策略迁移（Feb 2026 精选之一）；更多解析见 [theory/ 目录](./theory/)

## 🚀 快速开始（按你的目标选入口）

| 你现在想做什么 | 从这里开始 | 你会得到什么 |
|---|---|---|
| **补齐理论/刷面试** | [`theory/README.md`](./theory/README.md) | 学习路线图 + 核心概念索引 |
| **找论文/做综述** | [`theory/paper_index.md`](./theory/paper_index.md)、[`theory/literature_review.md`](./theory/literature_review.md) | 多维索引 + 发展史全景图 |
| **真机落地/跑通闭环** | [`deployment/README.md`](./deployment/README.md) | 硬件选型、多模态同步、控制与调参清单 |
| **行业/公司信息** | [`companies/README.md`](./companies/README.md) | 公司与求职指南 + 产业报告 digest |
| **追前沿（每两周）** | [`reports/biweekly/README.md`](./reports/biweekly/README.md) | 每期要点 + 深度笔记回链 |
| **周报 + 风向洞察** | [`reports/weekly/README.md`](./reports/weekly/README.md) | 每周论文精选 + SOTA + 趋势风向（Layer 分析） |
| **看最新论文解析** | [`theory/`](./theory/) + [`scripts/SCRIPTS.md`](./scripts/SCRIPTS.md) | 每日自动 pipeline 生成 ⚡ 精选论文深度拆解 |

> 更细的学习路线与分 Part 结构，请直接看 [`theory/README.md`](./theory/README.md) 的“学习路线图”。

---

## 📂 项目结构

### 顶层目录

```
VLA-Handbook/
├── theory/          # 理论基础（核心）
├── deployment/      # 真机与部署
├── reports/         # 双周/周期性前沿报告
├── scripts/         # 自动化 pipeline 脚本（内容生成与推送）
├── book/            # 电子书版本
├── cheat-sheet/     # 速查表
├── question-bank/   # 题库与实战
├── product/         # 机器人产品大百科
├── system-design/   # 系统设计
└── companies/       # 机器人公司与求职
```

### 完整目录树

<details>
<summary>展开完整目录树</summary>

```
VLA-Handbook/
├── README.md                   # 项目主页
├── theory/                     # 理论基础
│   ├── README.md               # 索引
│   ├── dexterous_hand_mechanics.md # 🆕 灵巧手机械学深度解析
│   ├── math_for_vla.md         # VLA 必备数学基础
│   ├── vla_arch.md             # VLA 核心架构
│   ├── pi0_flow_matching.md    # Flow Matching（π0 核心）
│   ├── pi0_code_analysis.md    # π0 源码导读
│   ├── spirit_v1_5_dissection.md # 🆕 Spirit-v1.5（RoboChallenge Table30 #1）代码级拆解
│   ├── tactile_vla.md          # 触觉 VLA 与 SaTA 专题
│   └── ...                     # 更多文档见 theory/README.md
├── deployment/                 # 真机与部署
│   ├── README.md               # 索引
│   ├── robot_hardware_selection_pricing.md # 🆕 硬件选型与前沿流派对比
│   ├── embodied_data_collection_overview.md # 🆕 具身数据采集概览 (POV/Sim2Real/RL)
│   ├── multimodal_data_synchronization.md # 🆕 多模态数据同步技术
│   ├── dexterous_hand_wuji.md  # 无极手（舞肌/Wuji）深度解析
│   ├── dexterous_hand_applications.md # 灵巧手实战案例集 (VisionOS)
│   └── ...                     # 更多文档见 deployment/README.md
├── reports/                     # 周期性前沿报告
│   ├── biweekly/                # 双周报告（含 reflection_*.md 预测回顾）
│   └── weekly/                  # 周度报告：论文精选 + SOTA + 趋势风向洞察
├── scripts/                     # 自动化 pipeline 脚本
│   ├── SCRIPTS.md               # 脚本参考文档（命名规则与 DAG 拓扑）
│   ├── vla-trend-snapshot.py    # VLA 趋势快照生成器
│   └── backfill-vla-history.py  # 历史数据回填工具
├── book/                       # 电子书版本
├── question-bank/              # 题库与实战
└── companies/                  # 机器人公司与求职
```

</details>

---

## 🎯 核心入口

| 模块 | 链接 | 说明 |
|:-----|:-----|:-----|
| **📚 Theory 总索引** | [`theory/README.md`](./theory/README.md) | 理论基础、核心算法、前沿架构 |
| **🔍 论文索引** | [`theory/paper_index.md`](./theory/paper_index.md) | 多维度查找（技术/公司/时间） |
| **📖 文献综述** | [`theory/literature_review.md`](./theory/literature_review.md) | VLA 发展史全景图（按技术分类） |
| **🗓️ Biweekly 前沿报告** | [`reports/biweekly/README.md`](./reports/biweekly/README.md) | 每两周更新：VLA / 触觉 / 人形 / 基准 / 工程 |
| **📅 Weekly 周报** | [`reports/weekly/README.md`](./reports/weekly/README.md) | 每周：论文精选 + SOTA 动态 + 趋势风向洞察 |
| **🚀 真机部署** | [`deployment/README.md`](./deployment/README.md) | 硬件选型、多模态同步、Sim-to-Real |
| **🏢 公司与产业** | [`companies/README.md`](./companies/README.md) | 公司/求职指南 + 产业报告 digest（含人形整机图谱） |
| **💡 题库与实战** | [`question-bank/README.md`](./question-bank/README.md) | 面试真题、代码实战、微调指南 |
| **📋 速查表** | [`cheat-sheet/README.md`](./cheat-sheet/README.md) | 时间线、核心公式 |
| **⚙️ 自动化脚本** | [`scripts/SCRIPTS.md`](./scripts/SCRIPTS.md) | Pipeline 脚本参考：VLA 日报/双周/评分/社交情报/深度解析 |
| **📝 变更记录** | [`CHANGELOG.md`](./CHANGELOG.md) | 从 git 历史提炼的 Changelog |

---

## 🧠 Theory 快速推荐（高信号入口）

按“你当下要解决的问题”选入口（每条都是**能落地/可复用**的高信号内容）：

### 机械与硬件（把控制与可控性讲清楚）

- [`theory/dexterous_hand_mechanics.md`](./theory/dexterous_hand_mechanics.md)  
  **关键词**：Grübler / Jacobian / 阻抗控制｜**适合**：补硬核数学与控制直觉
- [`deployment/robot_hardware_selection_pricing.md`](./deployment/robot_hardware_selection_pricing.md)  
  **关键词**：直驱 vs 绳驱 vs 液压｜**适合**：做硬件选型与方案 trade-off

### 前沿模型（看“模型怎么长出来”，而不是看宣传）

- [`theory/pi0_5_dissection.md`](./theory/pi0_5_dissection.md)  
  **关键词**：π0.5 / 开放世界泛化｜**适合**：理解分层推理与泛化机制
- [`theory/pi0_6_dissection.md`](./theory/pi0_6_dissection.md)  
  **关键词**：π0.6 / Recap / Action Expert｜**适合**：追最新结构与训练口径
- [`theory/spirit_v1_5_dissection.md`](./theory/spirit_v1_5_dissection.md)  
  **关键词**：Qwen3-VL + DiT + ODE/Euler｜**适合**：看一次代码级复现入口
- [`theory/tactile/README.md`](./theory/tactile/README.md)  
  **关键词**：VTLA / SaTA / 传感器选型 / 落地 checklist｜**适合**：把触觉相关内容单独拎出来一口气看完

### 前沿系统（全栈闭环：从“能演示”到“能稳定跑完任务”）

- [`theory/frontier/figure_helix_02_full_body_autonomy_2026.md`](./theory/frontier/figure_helix_02_full_body_autonomy_2026.md)  
  **关键词**：Helix 02 / S0-S1-S2 / 200Hz-1kHz｜**适合**：全身 loco-manipulation 系统抽象
- [`theory/unifolm_vla_0_unitree_2026.md`](./theory/unifolm_vla_0_unitree_2026.md)  
  **关键词**：开源训练栈 / 数据管线 / 部署｜**适合**：按清单做“可执行 sanity-check”

### 动作生成（把 distribution 学成可采样的动作）

- [`theory/pi0_flow_matching.md`](./theory/pi0_flow_matching.md)  
  **关键词**：Flow Matching｜**适合**：从原理到工程折中（π0 核心）
- [`theory/diffusion_policy.md`](./theory/diffusion_policy.md)  
  **关键词**：Diffusion｜**适合**：理解多模态动作分布与去噪采样

### 效率优化（让训练/推理跑得动）

- [`theory/flash_attention.md`](./theory/flash_attention.md)  
  **关键词**：FlashAttention｜**适合**：显存/吞吐优化的第一性原理
- [`theory/peft_lora.md`](./theory/peft_lora.md)  
  **关键词**：PEFT / LoRA / QLoRA｜**适合**：小显存微调与工程落地

### 每日自动解析（Feb 2026 论文精选）

> 由自动 pipeline 每日抓取 ⚡ 评级论文并生成深度拆解，持续更新中。

- [`theory/world_action_models_are_zero_shot_policies_dissection.md`](./theory/world_action_models_are_zero_shot_policies_dissection.md)  
  **关键词**：World Action Model / 零样本策略｜**适合**：理解"世界模型即策略"的第一性原理
- [`theory/twinvla_data_efficient_bimanual_manipulation_with_twin_singl_dissection.md`](./theory/twinvla_data_efficient_bimanual_manipulation_with_twin_singl_dissection.md)  
  **关键词**：TwinVLA / 双臂操作 / 数据效率｜**适合**：双臂任务与数据高效训练
- [`theory/olaf_world_orienting_latent_actions_for_video_world_modeling_dissection.md`](./theory/olaf_world_orienting_latent_actions_for_video_world_modeling_dissection.md)  
  **关键词**：OLAF / 潜在动作 / 视频世界模型｜**适合**：无标注视频学动作表示
- [`theory/scaling_verification_can_be_more_effective_than_scaling_poli_dissection.md`](./theory/scaling_verification_can_be_more_effective_than_scaling_poli_dissection.md)  
  **关键词**：Scaling Verification / Test-Time Compute｜**适合**：理解验证扩展 vs 策略扩展的 trade-off
- [`theory/robogene_boosting_vla_pre_training_via_diversity_driven_agen_dissection.md`](./theory/robogene_boosting_vla_pre_training_via_diversity_driven_agen_dissection.md)  
  **关键词**：RoboGene / 多样性驱动 / VLA 预训练｜**适合**：提升 VLA 预训练数据质量的方法论

---

<details>
<summary><b>✨ 为什么值得看（知识库价值）</b></summary>

1. **硬件-模型全链路**：不仅讲 π0，还讲如何选灵巧手、如何解决 1000Hz 传感器同步。
2. **硬核数学推导**：包含雅可比矩阵、阻抗控制、Flow Matching 等核心数学第一性原理。
3. **2026 前沿视野**：持续同步最新硬件（Sharpa Wave, LEAP V2 Adv）与前沿研究（Feb 2026 新增 9 篇论文深度解析）。
4. **全中文 + 工程导向**：专业术语保留英文对照，聚焦 Robotics 特有挑战（如 Hysteresis、Backlash）。
5. **每日自动 pipeline**：VLA 论文评分（⚡/🔧/📖）、社交情报、深度解析由自动化脚本持续写入，见 [`scripts/SCRIPTS.md`](./scripts/SCRIPTS.md)。

</details>

---

<details>
<summary><b>🛠️ VLA 开发必备知识</b></summary>

### 机器人控制
| 方法 | 原理 | 适用场景 |
| :--- | :--- | :--- |
| **PID** | 误差反馈 | 底层关节控制 |
| **阻抗控制** | 弹簧-阻尼行为 | 接触任务、柔顺抓取、人机协作 |
| **前馈控制** | 动力学补偿 | 高频响应、抵消重力/摩擦力 |
| **MPC** | 滚动优化 | 轨迹优化、避障 |

### 硬件控制接口
| 硬件 | 通信协议 | 代表品牌 | 流派 |
| :--- | :--- | :--- | :--- |
| **灵巧手** | CAN/EtherCAT | Wuji, RealerHand, Sharpa | **电机直驱派** (高透明度) |
| | Tendon-driven | LEAP Hand, Shadow Hand | **绳驱线控派** (物理柔顺) |
| | Hydraulic | Sanctuary AI Phoenix | **液压重载派** (极致力量) |
| **机械臂** | EtherCAT, TCP/IP | UR, Franka, AgileX | 工业级 / 具身协作级 |

### Vision Language Models (VLM) - VLA 训练参考
> **最后更新**: 2026年1月12日

| 模型 | 参数量 | 优势 | HuggingFace |
| :--- | :--- | :--- | :--- |
| **Qwen2.5-VL** 🆕 | 3B-72B | **2025 SOTA**，多分辨率/长视频支持 | [Qwen/Qwen2.5-VL](https://huggingface.co/Qwen) |
| **PaliGemma 3B** | 3B | π0, OpenVLA 首选 Backbone | [google/paligemma-3b](https://huggingface.co/google) |
| **SigLIP** | 400M-2.6B | VLA 首选视觉编码器 | [google/siglip](https://huggingface.co/google) |

</details>

---

## 📝 最近更新怎么看？

- **每日论文解析**：看 [`theory/`](./theory/) 目录（自动 pipeline 每日抓取 ⚡ 论文并生成深度拆解）  
- **工程/研究前沿**：看双周报告索引 [`reports/biweekly/README.md`](./reports/biweekly/README.md)（每期都链回深度笔记）  
- **周度论文精选 + 趋势风向**：看 [`reports/weekly/README.md`](./reports/weekly/README.md)  
- **自动化原理**：看 [`scripts/SCRIPTS.md`](./scripts/SCRIPTS.md)（pipeline DAG 与脚本参考）  
- **全量变更记录**：看 [`CHANGELOG.md`](./CHANGELOG.md)（从 git 历史提炼，避免 README 长期膨胀）

---

## 🤝 贡献 (Contributing)

欢迎提交 Issue 和 Pull Request！
- 补充最新的 VLA 论文解读 / 真机部署经验 / 面试真题。

## 📄 许可证 (License)

内容采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.zh) 许可协议。
