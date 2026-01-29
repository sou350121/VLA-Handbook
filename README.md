# VLA Handbook（Vision-Language-Action）

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

> **VLA（Vision-Language-Action）领域的结构化知识库与工程实战手册。**
> 覆盖理论基础、模型解析、真机部署、论文索引与题库。

---

## 为什么这个 Handbook 值得关注（但不灌水）

- **以“可复现/可验收”为第一原则**：每篇尽量给出入口脚本、关键超参、shape/口径的 sanity-check（避免“看起来懂了，跑起来全错”）。  
- **把 Robotics 的“脏活”写清楚**：多模态同步、动作空间对齐、Sim2Real 断点、触觉/灵巧手硬件选型。  
- **持续更新，但不让 README 膨胀**：前沿用 Biweekly 回链，细节进深度笔记，改动进 Changelog。

## 先看这 6 篇（高信号精选）

| 方向 | 推荐入口 | 为什么值得点开 |
|---|---|---|
| **从 0 建立框架** | [`theory/README.md`](./theory/README.md) | 学习路线图 + 模块化索引（把“要学什么”降维成可走的路径） |
| **Flow Matching 入门到可用** | [`theory/pi0_flow_matching.md`](./theory/pi0_flow_matching.md) | 把“为什么快/怎么采样/工程折中”讲成可用抓手 |
| **代码级拆解（生成式动作头）** | [`theory/spirit_v1_5_dissection.md`](./theory/spirit_v1_5_dissection.md) | Qwen3-VL + DiT + ODE/Euler 的端到端入口与复现清单 |
| **真实开源 VLA 训练栈** | [`theory/unifolm_vla_0_unitree_2026.md`](./theory/unifolm_vla_0_unitree_2026.md) | Unitree UnifoLM-VLA-0：数据管线/部署与“30 分钟验收” |
| **触觉/灵巧操作范式** | [`theory/frontier/visual_tactile_pretraining_online_multitask_learning_2026.md`](./theory/frontier/visual_tactile_pretraining_online_multitask_learning_2026.md) | SciRobotics 2026：单目+二值触觉，含任务级失败模式与指标 |
| **真机落地总入口** | [`deployment/README.md`](./deployment/README.md) | 硬件选型、多模态同步、控制与调参 checklist |

## 🚀 快速开始（按你的目标选入口）

| 你现在想做什么 | 从这里开始 | 你会得到什么 |
|---|---|---|
| **补齐理论/刷面试** | [`theory/README.md`](./theory/README.md) | 学习路线图 + 核心概念索引 |
| **找论文/做综述** | [`theory/paper_index.md`](./theory/paper_index.md)、[`theory/literature_review.md`](./theory/literature_review.md) | 多维索引 + 发展史全景图 |
| **真机落地/跑通闭环** | [`deployment/README.md`](./deployment/README.md) | 硬件选型、多模态同步、控制与调参清单 |
| **行业/公司信息** | [`companies/README.md`](./companies/README.md) | 公司与求职指南 + 产业报告 digest |
| **追前沿（每两周）** | [`reports/biweekly/README.md`](./reports/biweekly/README.md) | 每期要点 + 深度笔记回链 |

> 更细的学习路线与分 Part 结构，请直接看 [`theory/README.md`](./theory/README.md) 的“学习路线图”。

---

## 📂 项目结构

### 顶层目录

```
VLA-Handbook/
├── theory/          # 理论基础（核心）
├── deployment/      # 真机与部署
├── reports/         # 双周/周期性前沿报告
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
│   └── biweekly/                # 双周报告
│       └── README.md            # 索引
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
| **🚀 真机部署** | [`deployment/README.md`](./deployment/README.md) | 硬件选型、多模态同步、Sim-to-Real |
| **🏢 公司与产业** | [`companies/README.md`](./companies/README.md) | 公司/求职指南 + 产业报告 digest（含人形整机图谱） |
| **💡 题库与实战** | [`question-bank/README.md`](./question-bank/README.md) | 面试真题、代码实战、微调指南 |
| **📋 速查表** | [`cheat-sheet/README.md`](./cheat-sheet/README.md) | 时间线、核心公式 |
| **📝 变更记录** | [`CHANGELOG.md`](./CHANGELOG.md) | 从 git 历史提炼的 Changelog |

---

## 🧠 Theory 快速推荐（高信号入口）

| 主题 | 文档 | 一句话总结 |
|:-----|:-----|:---------|
| **机械与硬件** | [`dexterous_hand_mechanics.md`](./theory/dexterous_hand_mechanics.md) | 🆕 Grubler 公式、雅可比对偶性与阻抗控制数学基础 |
| | [`robot_hardware_selection_pricing.md`](./deployment/robot_hardware_selection_pricing.md) | 🆕 直驱 vs 绳驱 vs 液压流派对比与典型操纵难点解析 |
| **前沿模型** | [`pi0_5_dissection.md`](./theory/pi0_5_dissection.md) | π0.5 开放世界泛化，分层推理机制 |
| | [`pi0_6_dissection.md`](./theory/pi0_6_dissection.md) | π0.6 Recap 自我进化 + Action Expert |
| | [`spirit_v1_5_dissection.md`](./theory/spirit_v1_5_dissection.md) | 🆕 Spirit-v1.5：Qwen3-VL + DiT，RoboChallenge 代码级复现与入口拆解 |
| | [`tactile_vla.md`](./theory/tactile_vla.md) | 🆕 触觉反馈 VLA、DTA 动态触觉阵列与 SaTA 研究 |
| **前沿系统** | [`figure_helix_02_full_body_autonomy_2026.md`](./theory/frontier/figure_helix_02_full_body_autonomy_2026.md) | 🆕 Helix 02：全身端到端 VLA，S0/S1/S2 分层闭环（200Hz/1kHz） |
| | [`visual_tactile_pretraining_online_multitask_learning_2026.md`](./theory/frontier/visual_tactile_pretraining_online_multitask_learning_2026.md) | 🆕 SciRobotics 2026：单目 + 二值触觉（VT 预训练 + 在线多任务学习），含失败模式/未来方向 |
| | [`unifolm_vla_0_unitree_2026.md`](./theory/unifolm_vla_0_unitree_2026.md) | 🆕 Unitree UnifoLM-VLA-0：开源训练/数据管线/部署，含可执行 sanity-check |
| **动作生成** | [`pi0_flow_matching.md`](./theory/pi0_flow_matching.md) | Flow Matching（比 Diffusion 快 5x，π0 核心） |
| | [`diffusion_policy.md`](./theory/diffusion_policy.md) | 扩散去噪，解决多模态分布 |
| **效率优化** | [`flash_attention.md`](./theory/flash_attention.md) | Tiling + 重计算，显存 O(N²)→O(N) |
| | [`peft_lora.md`](./theory/peft_lora.md) | 低秩分解，QLoRA ~6GB 微调 7B |

---

<details>
<summary><b>✨ 为什么值得看（知识库价值）</b></summary>

1. **硬件-模型全链路**：不仅讲 π0，还讲如何选灵巧手、如何解决 1000Hz 传感器同步。
2. **硬核数学推导**：包含雅可比矩阵、阻抗控制、Flow Matching 等核心数学第一性原理。
3. **2026 前沿视野**：同步 2026 年 1 月最新硬件（Sharpa Wave, LEAP V2 Adv）与研究（SaTA）。
4. **全中文 + 工程导向**：专业术语保留英文对照，聚焦 Robotics 特有挑战（如 Hysteresis、Backlash）。

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

- **工程/研究前沿**：看双周报告索引 [`reports/biweekly/README.md`](./reports/biweekly/README.md)（每期都链回深度笔记）  
- **全量变更记录**：看 [`CHANGELOG.md`](./CHANGELOG.md)（从 git 历史提炼，避免 README 长期膨胀）

---

## 🤝 贡献 (Contributing)

欢迎提交 Issue 和 Pull Request！
- 补充最新的 VLA 论文解读 / 真机部署经验 / 面试真题。

## 📄 许可证 (License)

MIT License
