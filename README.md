<p align="center"><img src="docs/banner.svg" width="100%" alt="VLA Handbook"></p>

# VLA Handbook

[![CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
![Auto-updated](https://img.shields.io/badge/内容-每日自动更新-blue)
[![PULSAR 照见](https://img.shields.io/badge/PULSAR_照见-每日精选-FF6B35.svg)](https://sou350121.github.io/pulsar-web/)
[![Agent-Playbook](https://img.shields.io/badge/Agent_Playbook-配套-8B5CF6.svg?logo=github&logoColor=white)](https://github.com/sou350121/Agent-Playbook)
[![VLA Expert Skill](https://img.shields.io/badge/VLA_Expert_Skill-Claude_×_Cursor-00C7B7.svg?logo=github&logoColor=white)](https://github.com/sou350121/VLA-expert-skill)
[![RSS 订阅](https://img.shields.io/badge/RSS-订阅-FFA500.svg?logo=rss&logoColor=white)](https://sou350121.github.io/pulsar-web/subscribe)
[![Daily Pulse](https://img.shields.io/badge/📊_Daily_Pulse-15方法族趋势-D4910A.svg)](PULSE.md)
[![Spatial-Intelligence-Handbook](https://img.shields.io/badge/姊妹仓库-Spatial_Intelligence_Handbook-6B8AFE.svg?logo=github&logoColor=white)](https://github.com/sou350121/Spatial-Intelligence-Handbook)

📊 **[PULSAR 照见](https://sou350121.github.io/pulsar-web/)** · 每日北京时间 12:00 更新，减少信息焦虑 — `sou350121.github.io/pulsar-web`

📈 **[Daily Pulse](PULSE.md)** · 15 个 VLA 方法族的论文流量 + 加速度 + 30 日趋势图 · 每日自动生成

📡 **[RSS 订阅](https://sou350121.github.io/pulsar-web/subscribe)** · 4 个主题 feed + 一键 OPML 导入 — 新文章/⚡🔧 论文/SOTA/周报直达你的阅读器 · [完整使用说明](docs/SUBSCRIBE.md)

🤖 **[VLA Expert Skill](https://github.com/sou350121/VLA-expert-skill)** · 一键让 AI 编程助手变成 VLA 专家 — 支持 Claude Code / Cursor / Codex / OpenCode

🛰️ **[Spatial-Intelligence-Handbook](https://github.com/sou350121/Spatial-Intelligence-Handbook)** · 姊妹仓库 — VLA 管 action policy，Spatial 管 world representation（3DGS / VGGT / depth foundation + 跨 embodiment 横向对比）。两者交集是 3D-aware VLA。

🏭 **[行业情报雷达](companies/industry_mainline.md)** · 🆕 每日扫描机器人/具身公司动态（融资 · 产品 · IPO · 合作）→ qwen 联网搜索 + 机械核验后追加到公司档案；周五出[行业判断地图](companies/industry_mainline.md)，原始记录见 [`memory/blog/archives/industry-radar/`](memory/blog/archives/industry-radar)。

> **🆕 2026-06-11 重大更新**
> - **理论总纲两个月来首次全面同步**：10 个主题 `*_mainline.md` 综合了 4-6 月新进的 ~109 篇深度解析（保留旧判断并显式记录张力），71 篇散落文章归位，200+ 失效内链修复，全库 254 → **380 篇**。
> - **行业情报层上线**：`companies/*.md` 不再静态——新增自动雷达每日补充融资/产品/IPO 动态（已回填过去 2 个月积压），周度生成行业判断地图。

### 全中文 · 工程实战导向的 VLA 知识库

VLA 论文每天几十篇，真正能用的工程细节零散在 GitHub Issue 和论文附录里。
这个 Handbook 做一件事：**把"看懂论文"和"跑通代码"之间的坑，全部填平。**

> 525 篇理论文档（10 个主题目录，每日自动新增 2-3 篇深度解析） · 165 篇英文社区实战笔记 · 300+ 条中文社区蒸馏 · 47 条 GitHub Issues 工程经验 · 24 期双周推理报告 · **行业情报雷达**（公司动态 · 融资 · IPO 每日追踪） · 每日自动 pipeline（⚡ 论文评分 · 深度拆解 · 社交情报）

---

## 三句话说清楚这个 Handbook 的价值

1. **不只是摘要**：每篇给出入口脚本、关键超参、shape 口径的 sanity-check——"看懂"和"跑通"之间的坑，都标出来了。
2. **Robotics 的脏活写清楚**：多模态同步、Sim2Real 断点、动作空间对齐、触觉/灵巧手硬件选型——这些在其他地方要么没有，要么藏在论文附录里。
3. **活的知识库**：自动 pipeline 每天抓最新 VLA 论文（⚡/🔧/📖 评级），精选后生成深度解析写入仓库，不是六个月没人维护的静态文档。

### 🔥 社区实战笔记 — 论文不会告诉你的事

这三份笔记是 Handbook 最独特的部分：**中英文社区 + GitHub Issues 的一手踩坑经验，自动采集、人工蒸馏，每周持续更新。**

| | 来源 | 内容 | 更新 |
|---|---|---|---|
| **[📕 中文社区（小红书）](deployment/community_field_notes_xiaohongshu.md)** | 300+ 条中文社区蒸馏（帖1-200 + 可追溯索引 40 条 + 黑话辞典 28 条） | LingBot-VLA / UnifoLM / π0.6 真机 RL / Figure Helix / GR00T N2 + ACT 训练成本实测 + 国标发布 + 薪资 80-120 万 + 社区黑话辞典。 | 每 3 天自动增量 |
| **[📘 英文社区（HF Blog / Discord）](deployment/community_field_notes_english.md)** | 165 篇可追溯条目（HF Blog · 厂商博客 · LeRobot Discord） | SmolVLA / DiffusionVLA / π0-FAST / GR00T N1.5 / OpenVLA-OFT / Figure Helix / Cosmos Policy / ACTSmooth 等前沿论文 + 真实部署经验。 | 每周五 11:00 |
| **[🔧 GitHub Issues](deployment/community_field_notes_github.md)** | 6 大核心仓库 47 条高互动 Issues 蒸馏 | GPU 兼容矩阵（RTX 50 系列 / Jetson）、Pi0 微调陷阱、GR00T 显存优化、训练收敛失败根因、跨仓库收敛信号。 | 每周自动扫描 |

## 📡 订阅（让新内容主动找到你）

不想每天来刷网站？订阅 RSS feed，新内容自动推送到你的阅读器：

| Feed | 内容 | 链接 |
|------|------|------|
| 🧠 **VLA 新文章** | theory/ 每日新增深度解读 | [vla-theory.xml](https://sou350121.github.io/pulsar-web/rss/vla-theory.xml) |
| ⚡ **VLA 每日信号** | ⚡🔧 级论文 + SOTA 榜（过滤 ❌📖） | [vla-daily.xml](https://sou350121.github.io/pulsar-web/rss/vla-daily.xml) |
| 📘 **AI 每日** | AI Agent 生态精选 + 深度解读 | [ai-daily.xml](https://sou350121.github.io/pulsar-web/rss/ai-daily.xml) |
| 📚 **周/双周报告** | 前瞻侦察 + 回顾分析 | [weekly.xml](https://sou350121.github.io/pulsar-web/rss/weekly.xml) |

**🎁 一键全订阅**：[OPML 导入文件](https://sou350121.github.io/pulsar-web/rss/opml.xml)（支持 Feedly / Inoreader / NetNewsWire 等）

**📖 完整使用说明**：[docs/SUBSCRIBE.md](docs/SUBSCRIBE.md) — 含各阅读器教学、CC BY 4.0 说明、FAQ

---

## 和这些方式相比

读 VLA 领域信息，大多数人用的是这四种方式——先说各自真正好在哪：

**看公众号**（机器之心 / 量子位 / PaperWeekly）：中文团队写的可读综述，编辑质量有保证，适合碎片时间、移动端阅读。
**查 GitHub Awesome 列表 / 公开综述**：整理好的资源书签，方便快速找到经典论文和开源项目。
**刷 X/Twitter 跟踪 VLA 作者**：第一时间看到作者反应和社区讨论，实时感强。
**刷小红书**：一线从业者的真实踩坑、复现参数、失败复盘——论文和公众号里绝对看不到的一手经验，评论区比正文更有料。

**选 VLA Handbook**：需要工程级深度——论文怎么跑通、部署怎么踩坑、Sim2Real 断在哪，而且每天自动更新，永久可查。**小红书的社区经验也会被自动收集、蒸馏后写入 Handbook。**

| 维度 | 公众号 ML 文章 | Awesome 列表 / 综述 | X/Twitter 速报 | 小红书社区 | **VLA Handbook** |
|------|-------------|-------------------|--------------|-----------|-----------------|
| **最擅长** | 可读中文综述，移动端友好 | 资源书签，快速入手 | 实时讨论，作者第一反应 | 一线踩坑、真实参数、失败复盘 | 工程实战 + 每日自动深度解析 |
| **工程细节** | ❌ 媒体视角 | ❌ 链接汇总 | ❌ 碎片化 | ⚠️ 有但散落在评论区 | ✅ 入口脚本 · 关键超参 · shape 校验 |
| **一手踩坑** | ❌ | ❌ | ⚠️ 偶尔 | ✅ 评论区大量真实经验 | ✅ [中文蒸馏](deployment/community_field_notes_xiaohongshu.md) · [英文蒸馏](deployment/community_field_notes_english.md) |
| **更新频率** | 不定期 | 月 / 季度 | 实时 | 实时（但搜索困难） | 每日自动 + 中英文社区每周 + GitHub 每周扫描 |
| **历史可查** | ❌ 90 天后限流失效 | ✅ 静态存档 | ❌ 算法埋没 | ❌ 搜索质量差，帖子易沉 | ✅ Git 永久记录，全文 grep |
| **生产踩坑** | ❌ | ❌ | ❌ | ✅ 但需要自己挖掘整理 | ✅ Sim2Real · 多模态同步 · 硬件选型 |
| **趋势预测验证** | ❌ 无追踪 | ❌ | ❌ | ❌ | ✅ 双周 ✅/❌ 历史追踪（12 期） |


---

## 先看这几篇（30 分钟内建立正确框架）

按依赖顺序排列——每一篇回答上一篇读完后自然产生的问题。

**第一层：VLA 是什么、核心设计选择是什么（~15 min）**

**① [VLA 架构总览](theory/vla-core/vla_arch.md)** `5 min`
先建立全局图：输入（视觉 + 语言）→ Backbone → Action Head → 机器人动作。RT-1 → RT-2 → OpenVLA → π0 的演化逻辑，读完你知道每个模块的作用和它们怎么拼在一起。

**② [动作生成三范式](theory/diffusion-flow/action_representations.md)** `10 min`
①读完你会问：Action Head 到底怎么输出动作？这篇回答：离散 Token（快但粗）→ Diffusion（精但慢）→ Flow Matching（又快又精）。这是 VLA 领域最关键的分叉点，后面所有论文都在这三条路上选边。

**第二层：当前赢家为什么赢、整个领域怎么演化（~15 min）**

**③ [Flow Matching 原理拆解](theory/diffusion-flow/pi0_flow_matching.md)** `10 min`
②告诉你 Flow Matching 胜出，这篇解释为什么：ODE 直线路径 vs Diffusion 的曲线去噪，5-20 步推理实现 50Hz 控制。π0 的工程实现细节。

**④ [研究主线梳理](theory/vla-core/vla_research_mainline.md)** `5 min`
拉远一步看全局：为什么 ACT/DP 仍是 baseline、数据规模化 → 感知增强 → RL 后训练三条改进主线怎么交汇。读完你有一张完整的领域地图。

**第三层：对接现实（按需深入）**

**⑤ [社区实战笔记（中文）](deployment/community_field_notes_xiaohongshu.md)** 🔥
论文不会告诉你的事——300+ 条中文社区蒸馏的真实参数、真实失败、真实吐槽。ACT 50 episodes 就能 work、Sim2Real 八成是物理参数没校准。

**⑥ [社区实战笔记（英文）](deployment/community_field_notes_english.md)** 🔥
165 条英文社区精选（HF Blog + 厂商博客 + LeRobot Discord）。前沿论文深度拆解 + 真实部署经验、训练配方、推理延迟实测。

**⑦ [真机部署总入口](deployment/README.md)**
硬件选型 · 多模态同步 · Sim-to-Real · 调参 checklist。准备上真机时再看。

**想深入特定模型？**

| 方向 | 入口 | 适合谁 |
|------|------|--------|
| 开源训练栈复现 | [UnifolM-VLA-0](theory/vla-core/unifolm_vla_0_unitree_2026.md) | 想在宇树机器人上跑通 |
| 端到端代码级理解 | [Spirit-v1.5 解析](theory/vla-core/spirit_v1_5_dissection.md) | 想读懂每行 shape 变换 |
| 最新零样本迁移 | [World Action Model](theory/world-model/world_action_models_are_zero_shot_policies_dissection.md) | 想了解 2026 前沿方向 |
| 触觉 + 力控对齐 | [TAF-VLA](theory/tactile/taf_vla_tactile_force_alignment_2026.md) | 想做触觉融合 |
| VLA 知识蒸馏 | [Shallow-Pi](theory/foundation/shallow_pi_knowledge_distillation_flow_vla_2026.md) | 想做模型压缩 / 边缘部署 |
| 完整学习路线 | [学习路线图](theory/README.md) | 想系统性地从头学 |

---

## 自动更新时刻表（北京时间）

| 内容 | 更新时间 | 去哪看 |
|------|---------|--------|
| ⚡ 论文评分（⚡/🔧/📖/❌） | 每日 09:15–10:00 | [theory/](theory/) |
| 🛰️ VLA 社交情报 | 每日 09:30 | [vla-social-intel/ →](https://github.com/sou350121/VLA-Handbook/tree/main/memory/blog/archives/vla-social-intel)（30 期存档） |
| 🔬 理论深度解析 | 周一 / 三 / 五 15:30 | [theory/](theory/) |
| 📕 小红书社区经验收集 | 每 3 天自动增量 | [社区实战笔记（中文）](deployment/community_field_notes_xiaohongshu.md) |
| 📘 英文社区经验收集 | 每周五 11:00 | [社区实战笔记（英文）](deployment/community_field_notes_english.md) |
| 📋 周报 + 风向洞察 | 每周日 10:30 | [reports/weekly/](reports/weekly/README.md)（29 期存档） |
| 📊 双周推理报告 | 每两周 | [reports/biweekly/](reports/biweekly/README.md)（12 期存档） |

---

## 项目结构

| 目录 | 内容 |
|------|------|
| [`theory/`](theory/) | 525 篇理论文档：vla-core 105 · foundation 78 · world-model 72 · planning 60 · tactile 43 · frontier 38 · diffusion-flow 37 · deployment 35 · rl 31 · perception 24 |
| [`deployment/`](deployment/) | 真机部署：硬件选型 · 多模态同步 · Sim-to-Real + 三份社区实战笔记 |
| [`reports/biweekly/`](reports/biweekly/) | 24 期双周推理报告（含预测回顾 ✅/❌ 打分）|
| [`reports/weekly/`](reports/weekly/) | 29 期周报 + 每日 digest + SOTA + 风向洞察 |
| [`memory/blog/archives/`](memory/blog/archives/) | 30 期 VLA 社交情报 + 小红书原始数据 |
| [`scripts/`](scripts/) | 自动化 pipeline（SCRIPTS.md 含完整 DAG）|
| [`question-bank/`](question-bank/) | 15 份面试题库与代码实战 |
| [`companies/`](companies/) | 12 份机器人公司分析与求职指南 |
| [`cheat-sheet/`](cheat-sheet/) | 速查表（时间线 · 核心公式）|
| [`book/`](book/) | 电子书版本 |

---

## 快速导航

| 目标 | 入口 | 说明 |
|---|---|---|
| 补理论 / 刷面试 | [`theory/README.md`](theory/README.md) | 路线图 + 核心概念索引 |
| 找论文 / 做综述 | [`theory/paper_index.md`](theory/foundation/paper_index.md) | 多维索引 + 发展史全景图 |
| 真机落地 | [`deployment/README.md`](deployment/README.md) | 硬件选型 · 多模态同步 · Sim-to-Real |
| 社区踩坑（中文） | [`deployment/community_field_notes_xiaohongshu.md`](deployment/community_field_notes_xiaohongshu.md) | 🔥 300+ 条中文社区蒸馏 · 帖1-200 + 索引 + 黑话辞典 |
| 社区踩坑（英文） | [`deployment/community_field_notes_english.md`](deployment/community_field_notes_english.md) | 🔥 165 条可追溯条目 · HF Blog + Discord + 厂商博客 |
| GitHub 工程经验 | [`deployment/community_field_notes_github.md`](deployment/community_field_notes_github.md) | 🔧 6 大仓库 47 条高互动 Issues · GPU 兼容矩阵 |
| 公司 / 求职 | [`companies/README.md`](companies/README.md) | 12 份公司分析 + 产业报告 digest |
| 双周前沿报告 | [`reports/biweekly/README.md`](reports/biweekly/README.md) | 12 期 · VLA / 触觉 / 人形 · 含预测回顾 |
| 周报 + 风向洞察 | [`reports/weekly/README.md`](reports/weekly/README.md) | 29 期 · 每周论文精选 + SOTA + 趋势分析 |
| 变更记录 | [`CHANGELOG.md`](CHANGELOG.md) | 从 git 历史提炼 |

---

## 2026 前沿深度解析（30 篇 · 自动 pipeline 持续生成中）

> ⚡ = 重要进展  🔧 = 工程可用  每日持续更新中

| 论文 | 方向 |
|------|------|
| **架构与范式** | |
| [Figure Helix 0.2: Full-Body Autonomy](theory/vla-core/figure_helix_02_full_body_autonomy_2026.md) | S1+S2 双系统人形 VLA |
| [SimVLA: Simple VLA Baseline](theory/vla-core/simvla_simple_vla_baseline_robotic_manipulation_2026.md) | 极简 VLA 基线 |
| [ABOT-M0: Action Manifold Learning](theory/vla-core/abot_m0_action_manifold_learning_vla_foundation_2026.md) | 动作流形学习 |
| [StarVLA: LEGO-like VLA Codebase](theory/vla-core/starvla_lego_like_vla_codebase_2026.md) | 模块化架构 |
| [LingBot: Pragmatic VLA Foundation Model](theory/vla-core/lingbot_vla_pragmatic_vla_foundation_model_2026.md) | 语言引导 VLA |
| [ViTRA: Scalable VLA Pretraining from Human Videos](theory/vla-core/vitra_scalable_vla_pretraining_human_activity_videos_2026.md) | 人类视频预训练 |
| **世界模型** | |
| [DreamZero: World Action Models Zero-shot](theory/world-model/dreamzero_world_action_models_zero_shot_policies_2026.md) | 零样本迁移 |
| [AtomVLA: Offline Post-Training + Predictive World Models](theory/world-model/atomvla_offline_post_training_predictive_latent_world_models_2026.md) | 离线后训练 |
| [VLAW: Iterative Co-Improvement VLA + World Model](theory/world-model/vlaw_iterative_co_improvement_vla_world_model_2026.md) | VLA-WM 协同 |
| [WAM: Three Routes to Video Pretraining](theory/world-model/wam_three_routes_video_pretraining_vs_vla_2026.md) | 视频预训练路线 |
| [Video Generation Models in Robotics Survey](theory/world-model/video_generation_models_in_robotics_survey_2026.md) | 视频生成综述 |
| **强化学习后训练** | |
| [GigaBrain: World Model RL Ramp](theory/world-model/gigabrain_0_5m_star_world_model_based_rl_ramp_2026.md) | 世界模型 RL |
| [Evo-RL: Open Real-World RL](theory/rl/evo_rl_open_real_world_rl_recap_pistar06_so101_2026.md) | 真机 RL 闭环 |
| **触觉与感知** | |
| [TAF-VLA: Tactile-Force Alignment](theory/tactile/taf_vla_tactile_force_alignment_2026.md) | 触觉力控对齐 |
| [TacRefineNet: Tactile-Only Grasp Refinement](theory/tactile/tacrefinenet_tactile_only_grasp_refinement_2026.md) | 纯触觉抓取 |
| [TouchGuide: Inference-Time Touch Steering](theory/tactile/touchguide_inference_time_steering_touch_guidance_2026.md) | 推理时触觉引导 |
| [Visual-Tactile Pretraining + Online Multitask](theory/tactile/visual_tactile_pretraining_online_multitask_learning_2026.md) | 视觉-触觉预训练 |
| **效率与部署** | |
| [Shallow-Pi: Knowledge Distillation for Flow VLA](theory/foundation/shallow_pi_knowledge_distillation_flow_vla_2026.md) | VLA 知识蒸馏 |
| [QVLA: Action-Centric Quantization](theory/foundation/qvla_action_centric_quantization_2026.md) | 动作量化 |
| [RDT2-UMI: Zero-Shot Cross-Embodiment](theory/foundation/rdt2_umi_zero_shot_cross_embodiment_2026.md) | 跨具身零样本 |
| [RoboPocket: Robot-Free Policy Iteration via Phone](theory/deployment/robopocket_robot_free_instant_policy_iteration_phone_2026.md) | 手机端策略迭代 |
| **训练栈与基础设施** | |
| [UnifolM: Open-source VLA Training Stack](theory/vla-core/unifolm_vla_0_unitree_2026.md) | 开源训练栈 |
| [RynnBrain: Open Embodied Foundation Models](theory/vla-core/rynnbrain_open_embodied_foundation_models_2026.md) | 开源基础模型 |
| [Physical Intelligence Layer: Robot API](theory/deployment/physical_intelligence_layer_robot_api_2026.md) | PI Robot API |
| [NVIDIA Physical AI + Autonomous Driving](theory/deployment/nvidia_physical_ai_autonomous_driving_2026.md) | NVIDIA 物理 AI |
| [NVIDIA AI 5-Layer Cake Infrastructure](theory/deployment/nvidia_ai_5_layer_cake_infrastructure_2026.md) | NVIDIA 基建 |
| **视觉与推理** | |
| [Thinker-VLM: Embodied Intelligence](theory/planning/thinker_vlm_embodied_intelligence_2026.md) | VLM 具身推理 |
| [WaveFormer: Wave Equation Vision](theory/perception/waveformer_wave_equation_vision_2026.md) | 波动方程视觉 |
| [FAST Foundation Stereo](theory/perception/fast_foundation_stereo_real_time_zero_shot_stereo_matching_2026.md) | 零样本立体匹配 |
| [GeOPT: Geometric Pretraining for Physics Sim](theory/world-model/geopt_lifted_geometric_pretraining_physics_simulation_2026.md) | 几何预训练 |

---

## 背后的系统：照见 Pulsar

VLA Handbook 的每日内容由 [照见 Pulsar](https://github.com/sou350121/Pulsar-KenVersion) 自动驱动。Pulsar 不只是一组定时脚本——它是一个**自我进化的系统**。

**自我进化**，是指它真的会改变自己的判断：系统维护 19 条 VLA 领域假设，每条带置信度分数。每个月，它统计哪些假设被真实数据反复触发、哪些长期没有支撑，然后自动调整置信度。判断偏了的假设进入 watch-list，下一周期系统会主动注入更多相关信号去验证它——不是人工干预，是系统在自己给自己补课。双周报告的每一条预测，下一期必须打分（✅ 已验证 / ❌ 落空 / ⏳ 待观察），正确率有完整历史记录（已积累 12 期）。

在这之上：

- **自愈 Watchdog** — 23 项健康检查，RSS 中断 · 评分缺失 · LLM 超时，故障自动恢复，不会静默丢数据
- **评分前置** — 每天 30+ 篇论文先经 ⚡/🔧/📖/❌ 评级，精选才进 LLM，节省 80%+ 推理成本
- **全自动** — 33 个 cron job，每天 09:00 开始，无需人工触发

## 🤖 让 AI 编程助手变成 VLA 专家

想让你的 AI 助手直接拥有本 Handbook 525 篇理论文档的知识？

**[VLA Expert Skill](https://github.com/sou350121/VLA-expert-skill)** 把 Handbook 的知识压缩成一个即插即用的 AI Skill，支持多个平台：

| 平台 | 安装方式 |
|------|---------|
| **Claude Code / Cowork** | 复制到 `.claude/skills/vla-expert/` |
| **Cursor** | 复制到 `.cursor/rules/` |
| **Codex / OpenCode** | 作为 system prompt 加载 |

安装后你的 AI 助手能做到：对抗性三视角辩论（Bull/Bear/Arbiter）、论文价值评估、研究方向判断、产业分析、部署指南——每日自动同步最新知识。

👉 **[github.com/sou350121/VLA-expert-skill](https://github.com/sou350121/VLA-expert-skill)**

---

## 贡献

欢迎提 Issue 和 PR：补论文解读 · 真机经验 · 面试题。见 `CONTRIBUTING.md`。

## 许可证

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.zh) · 由 [照见 Pulsar](https://github.com/sou350121/Pulsar-KenVersion) 系统自动驱动
