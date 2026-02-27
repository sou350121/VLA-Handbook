# VLA Handbook

[![CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
![Auto-updated](https://img.shields.io/badge/内容-每日自动更新-blue)

### 全中文 · 工程实战导向的 VLA 知识库

VLA 论文每天几十篇，真正能用的工程细节零散在 GitHub Issue 和论文附录里。  
这个 Handbook 做一件事：**把"看懂论文"和"跑通代码"之间的坑，全部填平。**

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
| ① | `theory/README.md` | 学习路线图：把"要学什么"降维成可走的路径 | 10–20 min |
| ② | `theory/pi0_flow_matching.md` | Flow Matching 原理 + 工程折中，可直接迁移到 VLA action head | 15–30 min |
| ③ | `theory/spirit_v1_5_dissection.md` | Qwen3-VL + DiT + ODE/Euler 端到端复现入口，代码级拆解 | 20–40 min |
| ④ | `theory/unifolm_vla_0_unitree_2026.md` | 开源训练栈：数据管线 + 部署要点 + 30 分钟验收清单 | 30–60 min |
| ⑤ | `deployment/README.md` | 真机落地总入口：硬件选型 · 多模态同步 · 调参 checklist | 按需查阅 |
| ⑥ | `theory/world_action_models_are_zero_shot_policies_dissection.md` | World Action Model 如何做到零样本策略迁移（Feb 2026 精选） | 20 min |

---

## 快速导航

| 目标 | 入口 | 说明 |
|---|---|---|
| 补理论 / 刷面试 | `theory/README.md` | 路线图 + 核心概念索引 |
| 找论文 / 做综述 | `theory/paper_index.md` · `theory/literature_review.md` | 多维索引 + 发展史全景图 |
| 真机落地 | `deployment/README.md` | 硬件选型 · 多模态同步 · Sim-to-Real |
| 公司 / 求职 | `companies/README.md` | 公司指南 + 产业报告 digest |
| 双周前沿报告 | `reports/biweekly/README.md` | VLA / 触觉 / 人形 / 基准 · 含预测回顾 |
| 周报 + 风向洞察 | `reports/weekly/README.md` | 每周论文精选 + SOTA + 趋势 Layer 分析 |
| 最新论文解析 | `theory/` + `scripts/SCRIPTS.md` | 每日自动 pipeline 精选深度拆解 |
| 变更记录 | `CHANGELOG.md` | 从 git 历史提炼，避免 README 膨胀 |

---

## 自动更新时刻表（北京时间）

| 内容 | 更新时间 | 说明 |
|------|---------|------|
| ⚡ 论文评分（⚡/🔧/📖/❌）| 每日 09:15–10:00 | arXiv 当日新论文自动评级 + 热点筛选 |
| 🔬 理论深度解析 | 周二 / 四 / 六 15:30 | ⚡/🔧 精选论文生成代码级拆解，写入 `theory/` |
| 📡 SOTA · Release · 社交情报 | 每日 10:10–10:30 | 榜单变动 · 新版本 · 社区争议 |
| 📋 周报 + 风向洞察 | 每周日 10:30 | 论文精选 + 4-Layer 趋势分析 |
| 📊 双周推理报告 | 每两周 | 技术收敛判断 + 可验证预测 + 上期回顾 |

---

## 项目结构

\`\`\`
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
\`\`\`

---

## Feb 2026 新增深度解析（自动 pipeline 生成）

> ⚡ = 战略必读  🔧 = 工程可用  每日持续更新中

| 论文 | 方向 | 链接 |
|---|---|---|
| World Action Models are Zero-shot Policies | 零样本泛化 | → |
| TwinVLA: Data-Efficient Bimanual Manipulation | 双臂操作 | → |
| Scaling Verification > Scaling Policy Learning | Test-Time Compute | → |
| RoboGene: Diversity-Driven VLA Pre-training | 预训练数据 | → |
| Olaf-World: Latent Actions for Video WM | 视频世界模型 | → |
| MIND: Memory & Action Control Benchmark | World Model 评估 | → |
| Agent World Model: Infinity Synthetic Envs | 合成环境 | → |
| CausalGDP: Causality-Guided Diffusion Policy | 因果 + 扩散策略 | → |
| TaCo: Tactile Codec Benchmark | 触觉数据 | → |

---

## pipeline 如何工作

每天凌晨到上午，[照见 Pulsar](https://github.com/sou350121/Pulsar-KenVersion) 系统自动完成这条链路：

\`\`\`
arXiv 当日论文
      │
      ▼
  ⚡/🔧/📖/❌ 评分
  （相关性 · 工程价值 · 机构权重）
      │
  精选 ⚡/🔧 论文
      │
      ▼
  三段式 LLM 拆解
  （原理 · 关键代码 · 工程折中）
      │
      ▼
  写入 theory/ · 推送 Telegram
      │
  每周汇总 → 周报风向洞察
      │
  每两周 → 预测报告 + 上期验证
\`\`\`

不是人工维护——是一个会自我校准的知识积累系统。

---

## 最近在更新什么

- **每日论文解析**：`theory/` 目录（⚡ 评级论文自动拆解，每天更新）
- **周报**：`reports/weekly/`（论文精选 + SOTA + 风向洞察）
- **双周前沿推理**：`reports/biweekly/`（技术收敛判断 + 可验证预测）
- **变更记录**：`CHANGELOG.md`

---

## 贡献

欢迎提 Issue 和 PR：补论文解读 · 真机经验 · 面试题。见 `CONTRIBUTING.md`。

## 许可证

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.zh)
