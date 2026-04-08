# 物理现实锚定的具身基础模型：RynnBrain (RynnBrain: Open Embodied Foundation Models)

> **发布时间**：2026（项目页 BibTeX：`rynnbrain2026`）
> **项目/模型**：RynnBrain（Dense 2B/8B；MoE 30B-A3B；后训练：Plan/Nav/CoP）
> **机构**：阿里达摩院（Alibaba DAMO Academy）
> **核心定位**：把“具身理解”从被动观测升级为**自我中心（ego-centric）认知 + 时空定位（spatiotemporal grounding）+ 物理空间推理 + 物理感知规划（physics-aware planning）**的一体化“brain model”，输出可被下游 VLA 直接消费的**指向/轨迹/规划中间产物**（项目主页与 GitHub 口径）。

**一手来源**：
- GitHub：`https://github.com/alibaba-damo-academy/RynnBrain`
- 项目主页：`https://alibaba-damo-academy.github.io/RynnBrain.github.io/`
- HuggingFace collection：`https://huggingface.co/collections/Alibaba-DAMO-Academy/rynnbrain`
- RynnBrain-Bench（repo 内 README）：`https://raw.githubusercontent.com/alibaba-damo-academy/RynnBrain/main/rynnbrain-bench/README.md`
- RynnScale（训练/评测口径）：`https://raw.githubusercontent.com/alibaba-damo-academy/RynnScale/main/projects/rynn_brain/README.md`

---

## 0. 先把“可复述结论”写清楚（1 分钟版）

- **一句话**：RynnBrain 把“视频理解/定位/指向/规划”做成一个统一的 encoder-decoder 脑模型接口，强调“把推理锚定在物理世界”，并用 **RynnBrain-Bench** 系统化评估 object/spatial/grounding/pointing 四类能力（GitHub + 项目主页）。
- **它是什么**：偏“brain / planner”一侧的基础模型（更像给下游 VLA 提供中间产物与可执行指令），而不是直接输出低层动作控制（GitHub 明确输出包括 spatial trajectories / physical pointing / action planning）。
- **它发布了什么**：Dense（2B/8B）+ MoE（30B-A3B），以及三个后训练专项模型：
  - **RynnBrain-Plan**：操作/任务规划（manipulation planning）
  - **RynnBrain-Nav**：视觉语言导航（vision-language navigation）
  - **RynnBrain-CoP**：Chain-of-Point 空间推理（spatial reasoning）
  （见 GitHub Model Zoo 与 cookbooks）。
- **一个关键数字（来自 RynnBrain-Bench leaderboard）**：在同一套 bench 上，RynnBrain-2B/8B/30B-A3B 在 Object Cognition、Spatial Cognition、Grounding、Area/Affordance/Trajectory 等指标上给出显著领先的分数（见 bench README 的 leaderboard 表格）。

---

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 维度 | 传统“具身理解”M-LLM（常见） | RynnBrain（本文） | 工程含义 |
|---|---|---|---|
| 目标 | 以观测/问答为主 | “理解 + 定位 + 指向/轨迹 + 规划”一体化（项目主页） | 更像 brain→VLA 的接口层 |
| 输入 | 单/少视角图片/视频 + 指令 | **omni-vision** + 文本指令（GitHub） | 更强调视角覆盖与自我中心认知 |
| 输出 | 文本回答 | 多模态输出：**spatial trajectories / physical pointing / action planning**（GitHub） | 下游可直接消费“可执行”中间产物 |
| 训练数据 | 通用多模态 + 少量具身 | 大规模时空/物理空间数据 + 通用知识（GitHub） | “时空定位”能力可能来自数据配方 |

### 1.2 关键机制 (Key Mechanism)

1) **物理空间推理（Physical-space reasoning）= 文本推理与空间落点交错**
- 项目主页强调 interleaved reasoning：在文本推理与空间 grounding 间交替，让推理“落地”。这与单纯“先想后答”的 VLM 不同：它把空间指向当作推理的一部分（项目主页）。

2) **规划输出显式融合“定位到的 affordance / object / area 信息”**
- GitHub 与 RynnScale 说明：模型能输出（或在后训练中学习输出）包含坐标/点序列的结果，并把这些信息嵌入规划结果；这使得规划更像“带物理锚点的指令”，方便下游执行。

3) **统一 encoder-decoder：Dense 与 MoE 一套范式**
- GitHub 指出同一 encoder-decoder 架构支持 Dense 与 MoE 变体；对工程落地，意味着接口协议更稳定：模型大小变化不必改 I/O 形态。

### 1.3 信息流/架构图 (Flow / Diagram)

```text
Inputs:
  - omni-vision (video / multi-view)
  - instruction (text)

Encoder-Decoder (Dense or MoE)
  -> multi-modal outputs:
       - physical pointing (points / areas)
       - spatial trajectories
       - task planning (stepwise plan)

Downstream:
  - VLA / policy consumes: located affordances + plan
```

---

## 2. 输出空间与任务族（Plan / Nav / CoP）

RynnBrain 把“具身问题”拆成三类典型下游：规划、导航、空间推理（GitHub Model Zoo）。从工程角度，这三个头的价值在于：它们把“语言→动作”中最容易失控的一段（对齐环境、对齐物体、对齐目标区域）拆成可观测的中间变量。

- **CoP（Chain-of-Point reasoning）**：用点/区域作为推理的可视化锚点，强调“推理要落在物理空间里”（项目主页）。
- **Plan（Manipulation planning）**：将目标分解成步骤，并显式融合已定位的 affordance/objects/areas（GitHub 与 RynnScale）。
- **Nav（Vision-language navigation）**：以视觉与语言形成路径/目标描述，强调在真实场景视频中的时空一致性（GitHub）。

---

## 3. RynnBrain-Bench：把“理解 + 定位”做成可衡量的基准

项目同时给出 **RynnBrain-Bench**，用四大支柱衡量“具身理解”的关键能力（bench README）：

- **Object Cognition**：细粒度属性理解 + 计数（评估包含 GPT-4o 打分口径；bench README）。
- **Spatial Cognition**：从 ego-centric 视频推导 3D 空间关系（数值题用 MRA/RoA；文本题用 GPT-4o 打分；bench README）。
- **Grounding**：先找关键帧，再在该帧预测空间坐标；指标为 Accuracy@0.5（bench README）。
- **Pointing**：预测 area/affordance/trajectory 等点序列（Area 命中率、Affordance 欧氏距离、Trajectory 用离散 Fréchet 距离 DFD；bench README）。

数据规模（bench README）：
- **3,616** 个视频 clips
- **12,000** 个 open-ended questions
- **21** 个子能力

> 工程启示：bench 把“先找关键帧 + 再定位/指向/轨迹”的两阶段要求写进评测定义，这对“把推理锚定于物理世界”的口径很关键。

---

## 4. 训练与评测口径（可复现入口）

RynnBrain 的训练与评测细节主要落在 RynnScale（GitHub 指向 RynnScale；RynnScale README）：

### 4.1 数据格式：坐标/点序列的统一表达
RynnScale 给出坐标序列的统一格式（RynnScale README）：
- Object bbox：`; (x1,y1),(x2,y2)`
- Area points：`; (x1,y1),(x2,y2),...`
- Affordance point：`; (x1,y1)`
- Trajectory points：`; (x1,y1),(x2,y2),...`
- Grasp pose bbox：`; (x1,y1),(x2,y2)`
并指出图像类任务中坐标会被归一化到 0–1000 等范围（RynnScale README）。

### 4.2 评测脚本：benchmarks 列表与关键参数
RynnScale README 给出 `rynn_scale.api.eval` 的评测入口与参数（fps=2、max_frames=512、像素上限、温度等），并包含 RynnBrain-Bench 对应的 `RynnBrainCog` / `RynnBrainLoc` 评测（RynnScale README 与 bench README）。

---

## 5. 工程视角：对 VLA Handbook 的意义（怎么用这套观点）

- **把“规划/导航/空间推理”当作可复用的 brain 接口**：RynnBrain 的输出（pointing/trajectory/plan）天然可作为下游 VLA 的条件输入，避免把所有不确定性塞进动作 head。
- **用 bench 把“物理锚定”变成可回归测试**：RynnBrain-Bench 把关键帧定位与空间点序列预测纳入评估维度，适合当作“时空定位能力”的回归集。
- **与 handbooks 里的分层观点对齐**：它更像 S2/S1 的 planner/brain（理解+计划），而不是 S0 的安全执行控制；在系统设计上，应把它放在“高层决策/中间表征”，低层仍需传统控制兜底（参见 `theory/robot_control.md` 的分层观点）。

---

## 6. 与相关工作对比（快速定位）

- 与 **Thinker**（具身 VLM）相比：Thinker 重点是“视角与视频末端信息”的输入协议与数据配方（见 `theory/frontier/thinker_vlm_embodied_intelligence_2026.md`）；RynnBrain 更强调“理解+定位+指向/轨迹+规划”的统一输出接口与 bench 评测闭环（项目主页/bench）。
- 与一般 **VLA**（直接出动作）相比：RynnBrain 更像把动作前的“空间对齐/目标锚定”做扎实；工程上可作为 VLA 的上游模块，减少动作空间量化/误差放大的压力（见 [FAST, 2025](https://arxiv.org/abs/2501.09747) 关注的动作 token 化问题，与本条目关注点不同）。

---

[← Back to Theory](../README.md)
