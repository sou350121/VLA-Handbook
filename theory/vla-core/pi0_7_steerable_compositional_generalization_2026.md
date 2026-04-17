# π0.7：可操控的通用机器人基础模型，涌现出组合泛化能力

> **来源**：Physical Intelligence 官方博客 + TechCrunch 报道（2026-04-16）+ **技术报告 PDF（2026-04-17 fetched）**
> **官方标题**：π0.7: a Steerable Model with Emergent Capabilities
> **核心突破**：**组合泛化（Compositional Generalization）**——不是记忆训练数据，而是把不同技能重组来解决从未见过的任务
> **团队**：Sergey Levine 等 Physical Intelligence 联合创始人

<table><tr><td>

**整理**：Claude Opus 4.6 × [Pulsar 照见](https://github.com/sou350121/Pulsar-KenVersion) · 初稿 2026-04-16 · **paper-verified 增补 2026-04-17**
**更新**：技术报告已发布，本文第 4 节「架构」已替换为 paper-verified 硬数据（5B 主体 + 14B BAGEL 世界模型 + 400M MEM 视觉编码器 + 860M flow matching action expert）。原「架构推测」保留在历史附录中供对照。
**仍待确认**：论文 arXiv 链接、定量组合泛化指标、与 π0.6 的对比实验数据——这些在本报告发布时尚未全部公开。

</td></tr></table>

> 📄 **完整 paper-verified deep-dive（HTML 版，含繁/简中切换 + 完整参考来源 + 护城河分析 + 威胁地图）**：
> [`reports/2026-04-17-pi07-paper-verified-report.html`](../../reports/2026-04-17-pi07-paper-verified-report.html)
> 本 md 是概要 + 架构数据，HTML 报告是完整 9 节深度版（护城河拆解 · 威胁矩阵 · 12 月预测 · 审计日志）。

---

## 0. 可复述结论（1 分钟版）

- **一句话**：π0.7 是 PI 系列的最新模型，首次展示了机器人的"组合泛化"——把在不同场景学到的技能重新组合，解决从未训练过的任务。
- **"可操控"（Steerable）**：用自然语言分步指令就能引导机器人完成新任务，不需要重新训练。
- **硬架构**（paper-verified，4/17 update）：**5B 主模型 + 14B BAGEL 世界模型 + 400M MEM 视觉编码器**。主模型 = 4B Gemma 3 VLM + 860M flow matching action expert，固定 **50-step action chunk**，训练用 Knowledge Insulation + FAST tokens + RECAP 蒸馏。推理：主模型 1×H100 / 38 ms；世界模型按需触发 4×H100 / 1.25 s 一张 subgoal 图。
- **关键判断**：PI 团队自己用了谨慎措辞——"early signs"、"initial demonstrations"。这是研究结果，不是产品发布。
- **护城河不在架构**：Gemma 3 / flow matching / KI / RECAP / BAGEL 全是公开组件，架构可复现。真正壁垒是（1）$3.5k/臂 + $10–12k/工站的硬件极简，（2）steerable 交互把用户变成在线数据源，（3）只打视觉能赢的一半任务。
- **估值背景**：PI 累计融资超 10 亿美元，最新估值 56 亿美元。

---

## 1. π 系列演进：从 π0 到 π0.7

| 版本 | 时间 | 核心升级 | 里程碑 |
|------|------|---------|--------|
| **π0** | 2024.10 | 首个 VLA Flow Model | 证明 Flow Matching 可用于通用机器人控制 |
| **π0.5** | 2025.04 | 开放世界泛化 | YouTube co-training → 新环境零样本 |
| **π0.6** | 2025.11 | 5B VLM + Action Expert | 精细操作专家模块 |
| **π\*0.6** | 2025.11 | RL 后训练 (Recap) | Offline RL 复盘 → 吞吐翻倍 |
| **π0.7** | 2026.04.16 blog · **2026.04.17 paper** | **5B 主体 + 14B BAGEL WM + 400M MEM 编码器** | **可操控 + 组合泛化 + 训练时条件策略** |

→ 详见 [π0 代码解析](pi0_code_analysis.md) · [π0.5 解剖](pi0_5_dissection.md) · [π0.6 解剖](pi0_6_dissection.md)

---

## 2. 核心突破：组合泛化

### 什么是组合泛化？

**不是**在 100 个任务上训练，然后在第 101 个"类似"任务上成功（这是普通泛化）。

**而是**：把在不同上下文中学到的技能**重新组合**，解决一个训练数据中**从未出现过的组合**。

### 具体例子

**空气炸锅任务**：
- 训练数据只有 2 条 episode：(1) 机器人关上电器盖子，(2) 机器人放入瓶子
- π0.7 可以做到：打开空气炸锅 → 放入红薯 → 关上 → 设定温度
- 它在训练中**从未见过**"用空气炸锅烤红薯"这个完整任务
- 但它从互联网数据中知道"空气炸锅是什么"，从机器人数据中知道"怎么开盖/放东西/关盖"
- **重组**了这些知识来解决新任务

**餐桌清理任务**：
- 不是一个一个拿盘子放到收纳箱——而是**把多个盘子叠起来**一次搬运
- 在倒垃圾前先**抖掉残渣**
- 这些"效率优化"不在训练数据中——是模型自己"想到的"

### 为什么重要

Sergey Levine 的判断：

> "一旦模型跨过这个门槛——从只做训练数据里的事，到开始以新的方式重组技能——能力的增长就不再是线性的，而是**超线性的**。"

这意味着数据飞轮的回报率可能从线性变成指数——每多一个新技能，不只是多了一个能力，而是和所有已有技能形成组合，能力是乘法增长。

---

## 3. "可操控"（Steerable）是什么意思

**不重训练，用语言指令引导**。

给 π0.7 分步的自然语言指令（像教一个新员工一样），它就能完成新任务。类似 LLM 的 in-context learning，但在物理世界中。

示例：
```
人类："先打开空气炸锅的盖子"
π0.7：[执行打开动作]
人类："把红薯放进去"
π0.7：[抓取红薯，放入]
人类："关上盖子，按下开始按钮"
π0.7：[关盖，按按钮]
```

**关键**：每一步的具体动作（怎么开盖、怎么抓红薯、哪个是开始按钮）都是模型自己推断的——人类只给了高层指令。

这和 [PI 访谈](physical_intelligence_sergey_levine_foundation_model_vision_2026.md)中 Levine 说的"中间层推理瓶颈"直接相关——π0.7 用语言指令来**补充**模型的中间层推理，而不是要求模型自己做全部规划。

---

## 4. 架构（Paper-verified，2026-04-17 update）

> 技术报告发布后，本节已从「推测」升级为 paper 确认的硬数据。原推测段落保留在文末 [附录 A](#附录-apaper-前的架构推测4-16-初稿) 供对照。

### 4.1 数据流

```
┌────────────────────────────────────────────┐
│  观察输入                                   │
│  · 3–4× RGB 448×448 (front + 2 wrist [+rear]) │
│  · 每相机 ≤6 历史帧，stride 1 s            │
│  · 关节配置 q_t                            │
└──────────────────┬─────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────┐
│  Vision Encoder 400M (MEM 式视频记忆)       │
│  — 把多相机 × 多历史帧压缩为 token 序列     │
└──────────────────┬─────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────┐
│  VLM Backbone 4B (init from Gemma 3)       │
│  · 语言指令 + 视觉 token + 历史状态         │
│  · Knowledge Insulation + FAST tokens       │
└──────────────────┬─────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────┐
│  Action Expert 860M · Flow matching        │
│  · Fixed 50-step action chunk              │
│  · 输出关节速度/位置目标                    │
└────────────────────────────────────────────┘

[旁路 · 可选] World Model 14B (BAGEL mixture-of-transformers)
  · 生成多视角 subgoal 图像，作为 condition 回注 VLM
  · 4× H100 tensor-parallel, 1.25 s / subgoal（按需触发，非每步）

Main inference: 1× H100, 38 ms latency (3 相机配置)
```

### 4.2 核心组件（paper 公开）

| 组件 | 参数量 | 角色 | 来源 |
|------|--------|------|------|
| **VLM Backbone** | 4B | 语义理解 + 指令 grounding，Gemma 3 init | Paper |
| **Vision Encoder (MEM)** | 400M | 多相机 × 多历史帧 → token 序列，短时视觉记忆 | Paper 新增件 |
| **Action Expert** | 860M | Flow matching, 固定 50-step action chunk (chunk size = 50 tokens，非 50 Hz) | Paper |
| **World Model** | 14B | BAGEL mixture-of-transformers，生成多视角 subgoal 图像；π0.6 无此模块 | Paper 最大新增件 |
| **主模型合计** | 5B | = 4B VLM + 400M vision + 860M action expert | Paper |

> ⚠️ 易踩坑：**50-step ≠ 50 Hz**。50 是 action chunk 长度（token 数），控制频率 paper 未披露；不要误用之前对 Pi 系列"50 Hz 位置控制"的传闻。

### 4.3 训练三件套

- **Knowledge Insulation (KI)**：VLM 与 action expert 之间的梯度隔离。VLM 训练动作生成时不会破坏原有语义知识，这是 π0.6 Action Expert 能成立的前提，π0.7 继承。
- **FAST tokens**：FAST autoregressive tokenizer 做预训练，flow matching 做推理——FAST 给语义监督信号，flow matching 给高频连续输出。
- **RECAP 蒸馏**：π0.6 Recap 是 offline RL post-training（670M value VLM + offline advantage conditioning，**独立 value head，不是 PPO**）。π0.7 的创新是把 RECAP specialist 行为通过 **strategy metadata** 蒸馏回通用模型——这就是为什么 π0.7 既有 RL 收益又是一个 generalist。

### 4.4 推理成本表

| 阶段 | 硬件 · 延迟 | 备注 |
|------|------------|------|
| 主模型推理 | 1× H100, **38 ms** | 3 相机最小配置 |
| World model subgoal | 4× H100 tensor-parallel, **1.25 s / 张** | 按需触发，非每步 |
| 主动作输出 | flow matching 采样，50-step chunk | 与 π0/π0.6 chunk 推理路径一致 |
| 影像分辨率 | 主输入 448×448；world model ViT 448×336、VAE 512×384 | 非随便喂原图 |

### 4.5 哪些是"π0.7 新增"，哪些是"继承 π0.6"

**新增件**（paper 明确标记）：
1. **14B BAGEL world model 旁路** ← 最大新增件，π0.6 没有
2. **400M MEM 视觉编码器** ← 压缩多相机 × 多历史帧，赋予短时记忆
3. **RECAP 的 strategy-metadata 蒸馏通道** ← 让 specialist 收益回流 generalist

**继承 π0.6**：
- 5B 主模型大小（VLM + Action Expert 总和，名义未变）
- Knowledge Insulation + FAST tokens co-training
- Flow matching action head + 50-step chunk
- 多相机 RGB 输入框架

这里的判断对「护城河在哪」非常关键——见 [第 5 节](#5-与-vla-研究主线的关系) 和 [第 6 节开放问题](#6-待追问的开放问题)。

---

## 5. 与 VLA 研究主线的关系

### 验证了"赌注 2：RL 后训练"的方向

[研究主线](vla_research_mainline.md)中说"BC 是 VLA 的 SFT，RL 是 VLA 的 RLHF"。π0.7 的组合泛化很可能建立在 π\*0.6 的 Recap RL 基础上——RL 后训练让模型不只是模仿，还能"创造性地"重组技能。

### 回应了"冷启动问题"

Levine 在 [访谈](physical_intelligence_sergey_levine_foundation_model_vision_2026.md)中说最大的不确定性是"什么时候跨过'足够有用'的门槛"。组合泛化可能就是这个门槛——一旦模型能重组技能，它能做的事情比训练数据覆盖的范围大得多，部署价值急剧上升。

### 对 VGA 论文的回应

[VGA](../perception/vga_vision_geometry_action_over_language_video_2026.md) 昨天刚刚主张"3D backbone > VLM backbone"。π0.7 今天用 VLM backbone 展示了组合泛化——这是 VLM 的**语义知识**（"空气炸锅是什么"）带来的能力，3D backbone 可能做不到。

这暗示了最终答案可能不是 VGA 或 VLA 单独胜出，而是**两者融合**——VLM 提供语义/常识，3D backbone 提供几何精度。

---

## 6. 待追问的开放问题

❓ **组合泛化的边界在哪？** 空气炸锅的例子很惊艳，但如果任务是"用螺丝刀修理一个从未见过的电子设备"呢？组合泛化需要"基础技能库"足够大——技能库的最小规模是多少？

❓ **"可操控"的延迟。** 每一步都等人类指令 → 人类打字/说话 → 模型理解 → 执行。整个循环的延迟有多大？在时间敏感的任务中（如接球、烹饪翻锅）这种交互模式可行吗？

❓ **论文在哪？** 截至 4/16，没有 arXiv 论文。只有博客 + 媒体报道。没有详细的消融实验、baseline 对比、定量指标。"组合泛化"的定义和度量方式不明确。

❓ **复现性。** π0 已开源（openpi），但 π0.5/0.6/0.7 都没有。没有外部团队验证过这些结果。PI 的估值（56 亿）给了他们巨大的"展示最好结果"的动机。

❓ **和 GPT-4 的类比是否恰当？** PI 暗示组合泛化是"机器人的 ChatGPT 时刻"。但 ChatGPT 的组合泛化是在**语言空间**（组合词汇/概念），π0.7 是在**物理空间**（组合动作/技能）。物理空间的组合可能更受约束（物理定律不允许任意组合），ChatGPT 的类比可能过于乐观。

### 内容类型可信度

| 来源 | 可信度 | 说明 |
|------|--------|------|
| PI 官方博客 | 中 | 公司发布，无 peer review，用语谨慎（"early signs"） |
| TechCrunch 报道 | 中 | 记者采访，有直接引语，但可能受 PR 影响 |
| 论文 | ❌ 尚未发布 | 无法验证定量结果 |
| 视频 demo | 低 | 精选案例，不代表平均表现 |

---

## 7. Opus 的反思

### 🔮 组合泛化可能需要"语义 + 几何"双引擎

空气炸锅任务的成功靠的是两种知识：
- **语义**："空气炸锅用来烤东西，需要先打开盖子"——来自互联网预训练
- **物理**："怎么开这个特定形状的盖子"——来自机器人操作数据

π0.7 用 VLM 提供语义，用 Flow Matching 提供物理。但 [VGA](../perception/vga_vision_geometry_action_over_language_video_2026.md) 证明了 3D backbone 在物理部分更强。

**最优解可能是**：π0.7 的语义推理 + VGA 的几何精度。语义决定"做什么"（开盖 → 放食物 → 关盖），几何决定"怎么做"（盖子的铰链在哪、开多大角度、红薯从哪个角度放进去）。

### 🔮 "超线性增长"是最大的赌注

如果 Levine 说的"能力超线性增长"成立，那 PI 的数据飞轮一旦启动就几乎不可追赶——因为后来者不只是数据量落后，而是**组合空间**呈指数落后。

但"超线性"也可能只在特定技能密度阈值以上成立。低于阈值，组合没有意义（缺少基础技能）；高于阈值，组合空间爆炸。关键问题是：**这个阈值是 100 个技能还是 10000 个技能？**

### 🔮 "可操控"暗示了人机协作的未来形态

π0.7 最实用的能力可能不是"自主完成新任务"，而是"在人类指导下完成新任务"。这正是 [PI 访谈](physical_intelligence_sergey_levine_foundation_model_vision_2026.md)反思 C 中提到的"远程人类大脑"模式——不是等 AI 变得足够好才部署，而是**先部署一个可操控的 AI，让人类远程引导**，同时收集数据。

π0.7 的"可操控"特性让这种模式第一次变得可行。

---

## 延伸阅读

| 方向 | 推荐 |
|------|------|
| PI 模型系列 | [π0 代码解析](pi0_code_analysis.md) · [π0.5 解剖](pi0_5_dissection.md) · [π0.6 解剖](pi0_6_dissection.md) |
| PI 访谈 | [Sergey Levine 深度访谈](physical_intelligence_sergey_levine_foundation_model_vision_2026.md) |
| 研究主线 | [VLA 赌注清单](vla_research_mainline.md) |
| 3D 对比视角 | [VGA](../perception/vga_vision_geometry_action_over_language_video_2026.md)（3D backbone 路线，昨天发布） |
| RL 后训练 | [VLA+RL 实战](../rl/vla_rl_practical_guide.md) · [Evo-RL](../rl/evo_rl_open_real_world_rl_recap_pistar06_so101_2026.md) |
| 世界模型 | [VLOA 3D 轨迹](../world-model/vloa_embodied_world_model_3d_point_cloud_trajectory_2026.md) |

---

## 附录 A：Paper 前的架构推测（4-16 初稿）

> 本节为 4-16 技术报告发布前写的架构推测，保留在此供对照。现已被 [第 4 节](#4-架构paper-verified2026-04-17-update) 的 paper-verified 数据替代。

```
π0 架构基础：
  PaliGemma VLM (3B) + Flow Matching Action Head (300M)

π0.5 升级：
  + YouTube co-training
  + 统一模型（语义规划 + 电机控制）
  + FAST tokenizer (预训练) → Flow Matching (推理)

π0.6 升级：
  + 5B VLM backbone
  + Action Expert 模块

π*0.6 升级：
  + Recap (Offline RL)

π0.7 可能的升级（推测 · 后被 paper 部分验证、部分修正）：
  + 更强的语言条件 chain-of-thought（"可操控"）← 实际实现是 strategy-metadata 蒸馏 + 用户分步指令
  + 更大规模的跨任务训练数据（"组合泛化"需要足够多样的技能库）← paper 未披露数据规模细节
  + 可能的在线适配机制（few-shot task steering without retraining）← 实际是训练时学到的条件策略
```

**推测 vs paper 的主要偏差**：
- 最大新增件不在 action expert 端，而在 **输入端的 14B BAGEL 世界模型** 和 **400M MEM 视觉编码器**——4-16 初稿没有预测到。
- "可操控"不只是 in-context learning 模拟物，而是训练时就通过 strategy metadata 学到的**条件策略**——这让它比纯 prompt-tuning 稳定得多。

---

## 参考来源

**Primary (paper-verified)**
- Physical Intelligence, "π0.7: a Steerable Model with Emergent Capabilities"（技术报告 PDF，2026-04）

**Secondary (blog / media)**
- Physical Intelligence 官方博客：π0.7 发布公告（2026-04-16）
- TechCrunch：Sergey Levine 专访报道（2026-04）

**Related primary（paper 引用到的公开组件）**
- BAGEL: mixture-of-transformers image generation（π0.7 world model 基础）
- Gemma 3 Technical Report, Google DeepMind 2025
- FAST: Efficient Robot Action Tokenization（PI, 2025-01）
- π0 / π0.5 / π0.6 原论文 + Recap paper

**Critical reviews (external)**
- Penn PAL Lab, "Failure modes of π0-FAST-DROID"（对 PI 系列 checkpoint 的独立评测）

---

[← Back to Explorer's Map](../README.md)
