# VLA 论文索引 (Paper Index)

> **快速查找**: 多维度索引系统，帮助快速定位相关论文
> **最后更新**: 2026-01-28

---

## 📋 使用指南

> **Industry Note**: Jim Fan 2025 年度复盘（硬件可靠性 / 评测可复现 / VLM→VLA 路线反思）见 [`./frontier/jim_fan_2025_robotics_lessons.md`](./frontier/jim_fan_2025_robotics_lessons.md)

| 查找方式 | 跳转链接 |
|:---|:---|
| 按技术分类 | [技术分类索引](#技术分类索引) |
| 按公司/机构 | [公司分类索引](#公司分类索引) |
| 按时间线 | [时间线索引](#时间线索引) |
| 完整综述 | [literature_review.md](./literature_review.md) |

---

## 🎯 快速索引表

### 按技术分类索引

| 技术类别 | 论文 | 链接 |
|:---|:---|:---|
| **Diffusion** | Diffusion Policy | [详细](#diffusion-policy) |
| | RDT-1B | [详细](#rdt) |
| | DKT Perception | [深度解析](./frontier/dkt_transparency_perception.md) |
| **Tactile / Visuotactile** 🆕 | VT Pretraining + Online Multitask (SciRobotics 2026) | [深度解析](./frontier/visual_tactile_pretraining_online_multitask_learning_2026.md) |
|  | GenForce (Nat Commun 2026) | [深度笔记](./tactile/genforce_tactile_force_transfer_2026.md) |
| **Flow Matching** | π0 | [深度解析](./pi0_flow_matching.md) |
| | π0.5 | [深度解析](./pi0_5_dissection.md) |
| | π0.6 | [深度解析](./pi0_6_dissection.md) |
| | LingBot-VLA | [深度解析](./lingbot_vla_pragmatic_vla_foundation_model_2026.md) |
| **Tokenization** | RT-2 | [深度解析](#rt-2) |
| | OpenVLA | [深度解析](#openvla) |
| | FAST | [详细](#fast) |
| **RL 训练** | GR-RL | [深度解析](./gr_rl_dissection.md) |
| | π*0.6 Recap | [深度解析](./pi0_6_dissection.md#recap) |
| | RLinf（VLA+RL Infra） | [工具解读](./frontier/rlinf_vla_rl_training.md) |
| **架构创新** | WALL-OSS | [深度解析](./wall_oss.md) |
| | Galaxea G0 | [详细](#galaxea-g0) |
| **训练技术** | Knowledge Insulation | [摘要](#knowledge-insulation) |
| **Latent Action** | UniVLA | [详细](#univla) |
| | EvoVLA | [详细](#evovla) |
| | MemoryVLA | [详细](#memoryvla) |
| **NeurIPS 2025** 🆕 | Artificial Hivemind | [详细](./neurips_2025_insights.md#1-artificial-hivemind-语言模型的同质化问题) |
| | Gated Attention | [详细](./neurips_2025_insights.md#2-gated-attention-门控注意力机制) |
| | 1000 Layer Networks | [详细](./neurips_2025_insights.md#3-1000-layer-networks-深层自监督-rl) |
| | Diffusion Generalization | [详细](./neurips_2025_insights.md#4-diffusion-models-的泛化机制) |
| | Superposition Scaling | [详细](./neurips_2025_insights.md#5-superposition-表示叠加与神经缩放) |
| | RL Reasoning Limits | [详细](./neurips_2025_insights.md#6-rlvr-的局限性rl-真的能扩展推理能力吗) |

### 按公司分类索引

| 公司/机构 | 论文 | 链接 |
|:---|:---|:---|
| **Google DeepMind** | RT-2 | [深度解析](#rt-2) |
| | RT-1 | [详细](#rt-1) |
| **Physical Intelligence** | π0 | [深度解析](./pi0_flow_matching.md) |
| | π0.5 | [深度解析](./pi0_5_dissection.md) |
| | π0.6 | [深度解析](./pi0_6_dissection.md) |
| | FAST | [详细](#fast) |
| | Knowledge Insulation | [摘要](#knowledge-insulation) |
| **ByteDance Seed** | GR-RL | [深度解析](./gr_rl_dissection.md) |
| | RDT-1B | [详细](#rdt) |
| **Stanford** | OpenVLA | [深度解析](#openvla) |
| | ACT | [详细](#act) |
| **X² (自变量)** | WALL-OSS | [深度解析](./wall_oss.md) |
| **Galaxea AI** | G0 | [详细](#galaxea-g0) |
| **Robbyant Team** 🆕 | LingBot-VLA | [深度解析](./lingbot_vla_pragmatic_vla_foundation_model_2026.md) |
| **Columbia** | Diffusion Policy | [详细](#diffusion-policy) |
| **浙江大学** 🆕 | DKT Perception | [深度解析](./frontier/dkt_transparency_perception.md) |
| | VT Pretraining + Online Multitask (SciRobotics 2026) | [深度解析](./frontier/visual_tactile_pretraining_online_multitask_learning_2026.md) |
| **华盛顿大学** 🆕 | Artificial Hivemind | [详细](./neurips_2025_insights.md#1-artificial-hivemind-语言模型的同质化问题) |
| **阿里千问** 🆕 | Gated Attention | [详细](./neurips_2025_insights.md#2-gated-attention-门控注意力机制) |
| **普林斯顿** 🆕 | 1000 Layer Networks | [详细](./neurips_2025_insights.md#3-1000-layer-networks-深层自监督-rl) |
| **巴黎 PSL** 🆕 | Diffusion Generalization | [详细](./neurips_2025_insights.md#4-diffusion-models-的泛化机制) |
| **MIT** 🆕 | Superposition Scaling | [详细](./neurips_2025_insights.md#5-superposition-表示叠加与神经缩放) |
| **清华大学** 🆕 | RL Reasoning Limits | [详细](./neurips_2025_insights.md#6-rlvr-的局限性rl-真的能扩展推理能力吗) |

### 按时间线索引

| 年份 | 论文 | 链接 |
|:---|:---|:---|
| **2023** | Diffusion Policy | [详细](#diffusion-policy) |
| | RT-2 | [深度解析](#rt-2) |
| | ACT | [详细](#act) |
| **2024** | OpenVLA | [深度解析](#openvla) |
| | π0 | [深度解析](./pi0_flow_matching.md) |
| | RDT-1B | [详细](#rdt) |
| | DKT Perception | [深度解析](./frontier/dkt_transparency_perception.md) |
| | Galaxea G0 | [详细](#galaxea-g0) |
| | Knowledge Insulation | [摘要](#knowledge-insulation) |
| | UniVLA | [详细](#univla) |
| **2025** | π0.5 | [深度解析](./pi0_5_dissection.md) |
| | π0.6 | [深度解析](./pi0_6_dissection.md) |
| | FAST | [详细](#fast) |
| | GR-RL | [深度解析](./gr_rl_dissection.md) |
| | WALL-OSS | [深度解析](./wall_oss.md) |
| | EvoVLA | [详细](#evovla) |
| | MemoryVLA | [详细](#memoryvla) |
| | TTF-VLA | [详细](#ttf-vla) |
| | OmniVLA | [详细](#omnivla) |
| | MergeVLA | [详细](#mergevla) |
| | **NeurIPS 2025 Best Papers** 🆕 | [专题解读](./neurips_2025_insights.md) |
| **2026** 🆕 | LingBot-VLA | [深度解析](./lingbot_vla_pragmatic_vla_foundation_model_2026.md) |
| | VT Pretraining + Online Multitask (SciRobotics 2026) | [深度解析](./frontier/visual_tactile_pretraining_online_multitask_learning_2026.md) |
| | Video Generation Models in Robotics (Survey) | [前沿笔记](./frontier/video_generation_models_in_robotics_survey_2026.md) |

---

## 🔍 详细分类索引

### 技术分类索引

#### 1. 动作生成策略

##### Diffusion 系列
- **Diffusion Policy** (2023)
  - 技术: DDPM, U-Net/DiT
  - 动作空间: 连续
  - [详细内容](./literature_review.md#1-diffusion-policy)

- **RDT-1B** (2024)
  - 技术: DiT, 十亿参数
  - 动作空间: 连续
  - [详细内容](./rdt.md)

##### Flow Matching 系列
- **π0** (2024)
  - 技术: Flow Matching, ODE Solver
  - 动作空间: 连续
  - [深度解析](./pi0_flow_matching.md) | [代码解析](./pi0_code_analysis.md)

- **π0.5** (2025)
  - 技术: Flow Matching + 隐式推理
  - 动作空间: 连续
  - [深度解析](./pi0_5_dissection.md)

- **π0.6** (2025)
  - 技术: Flow Matching + Action Expert
  - 动作空间: 连续
  - [深度解析](./pi0_6_dissection.md)

- **LingBot-VLA** (2026)
  - 技术: Flow Matching + VLM + Action Expert（共享 attention）；可选 depth distill
  - 动作空间: 连续
  - [深度解析](./lingbot_vla_pragmatic_vla_foundation_model_2026.md)

##### Tokenization 系列
- **RT-2** (2023)
  - 技术: Action Tokenization, 256-bins
  - 动作空间: 离散
  - [详细内容](./literature_review.md#2-rt-2)

- **OpenVLA** (2024)
  - 技术: Action Tokenization + LoRA
  - 动作空间: 离散
  - [详细内容](./literature_review.md#3-openvla)

- **FAST** (2025)
  - 技术: DCT + BPE Tokenization
  - 动作空间: 离散（压缩）
  - [详细内容](./fast.md)

##### 其他
- **ACT** (2023)
  - 技术: CVAE + Action Chunking
  - 动作空间: 连续
  - [详细内容](./act.md)

##### Latent Action 系列 🆕
- **UniVLA** (IJRR 2024)
  - 技术: Task-centric Latent Actions
  - 核心: 从视频学习机器人无关的动作表示
  - [详细内容](./literature_review.md#univla-ijrr-2024)

- **EvoVLA** (2025)
  - 技术: Self-Evolving + Stage-Aligned Reward
  - 核心: 解决长时程"阶段幻觉"问题
  - [详细内容](./literature_review.md#evovla-2025)

- **MemoryVLA** (2025)
  - 技术: Perception-Cognition Memory
  - 核心: 感知-认知记忆系统
  - [详细内容](./literature_review.md#memoryvla-2025)

- **TTF-VLA** (2025)
  - 技术: Temporal Token Fusion
  - 核心: 训练无关的多帧融合

- **OmniVLA** (2025)
  - 技术: Multi-sensor Perception
  - 核心: 红外/雷达/麦克风多传感器融合

- **MergeVLA** (2025)
  - 技术: Cross-skill Model Merging
  - 核心: 跨技能知识迁移

#### 2. 训练方法

##### BC (Behavior Cloning)
- **RT-2** - Co-fine-tuning
- **OpenVLA** - LoRA Fine-tuning
- **π0** - Flow Matching Training

##### RL (Reinforcement Learning)
- **GR-RL** (2025)
  - 技术: Offline + Online RL
  - [深度解析](./gr_rl_dissection.md)

- **π*0.6 Recap** (2025)
  - 技术: Offline RL
  - [深度解析](./pi0_6_dissection.md#recap)

- **RLinf** (2025)
  - 定位: Reinforcement Learning Infrastructure（面向 Embodied / Agentic AI）
  - 关联: VLA+RL 训练框架（RLinf-VLA）、真机 RL、分布式 rollout 与可复现评估
  - [项目主页](https://github.com/RLinf/RLinf) | [手册解读](./frontier/rlinf_vla_rl_training.md)

##### 混合方法
- **π0.5** - Co-training (Robot + Internet + Sim)

#### 3. 架构创新

##### 单模型架构
- **RT-2** - VLM + Action Tokens
- **OpenVLA** - VLM + Action Head
- **π0** - VLM + Flow Matching

##### 双系统架构
- **Galaxea G0** (2024)
  - G0-VLM (大脑) + G0-VLA (小脑)
  - [详细内容](./galaxea_g0.md)

- **π0.6** - VLM (大脑) + Action Expert (小脑)
  - [深度解析](./pi0_6_dissection.md#action-expert)

##### 层级架构
- **WALL-OSS** (2025) - A级
  - Hierarchical CoT + Dual Heads
  - [详细内容](./wall_oss.md)

#### 4. 应用场景

##### 操作任务
- RT-2, OpenVLA, π0, GR-RL

##### 导航任务
- (待补充)

##### 灵巧手
- GR-RL (穿鞋带), RDT-1B (双臂操作)
- UniTacHand (2025): MANO UV Map 统一触觉表征，实现人手→机器人灵巧手策略迁移（零样本/小样本）
  - [手册解读](./frontier/unitachhand.md)
  - [论文（HTML）](https://arxiv.org/html/2512.21233v2)

##### 触觉 / 视触觉 (Tactile)
- 触觉为什么不可替代（行业/研究盘点）：[笔记](./frontier/tactile_irreplaceable.md)
- **SuperTac + DOVE** (Nature Sensors, 2025): 多模态电子皮肤（多光谱+摩擦电+IMU+温度/接近/振动）+ 触觉语言模型
  - [论文页](https://www.nature.com/articles/s44460-025-00006-y)
  - [手册笔记](./frontier/supertac_dove_multimodal_tactile_sensor.md)
- **VLA-Touch** (2025): Dual-Level Tactile Feedback
  - [论文](https://arxiv.org/abs/2507.17294)
- **OmniVTLA** (2025): Semantic-Aligned Tactile Sensing
  - [论文](https://arxiv.org/abs/2508.08706)
- **GelSLAM** (2025): Real-time High-Fidelity 3D Tactile SLAM
  - [论文](https://arxiv.org/abs/2508.15990)
- **Tactile Robotics: An Outlook** (2025): 触觉机器人综述与路线图
  - [论文](https://arxiv.org/abs/2508.11261)

---

### 公司分类索引

#### Google DeepMind
- **RT-2** (2023)
- **RT-1** (2022)

#### Physical Intelligence
- **π0** (2024)
- **π0.5** (2025)
- **π0.6** (2025)
- **FAST** (2025)
- **Knowledge Insulation** (2024)

#### ByteDance Seed
- **GR-RL** (2025)
- **RDT-1B** (2024) (与清华合作)

#### Stanford
- **OpenVLA** (2024)
- **ACT** (2023)

#### X² (自变量)
- **WALL-OSS** (2025)

#### Galaxea AI
- **G0** (2024)

---

### 时间线索引

#### 2023（早期探索）
- Diffusion Policy (RSS 2023)
- RT-2 (ICRA 2023)
- ACT (2023)

#### 2024（爆发期）
- OpenVLA (2024.06)
- π0 (2024.10)
- RDT-1B (2024.10)
- Galaxea G0 (2024.09)
- Knowledge Insulation (2024)

#### 2025（最新进展）
- π0.5 (2025.01)
- π0.6 (2025.11)
- FAST (2025.01)
- GR-RL (2025)
- WALL-OSS (2025)

#### 2026（世界模型 / 视频生成综述）
- Video Generation Models in Robotics (Survey, arXiv 2026): 视频生成模型在机器人中的应用、挑战与未来方向  
  - [笔记](./frontier/video_generation_models_in_robotics_survey_2026.md)
  - [论文](https://arxiv.org/abs/2601.07823)

---

## 📝 论文添加指南

### 如何添加新论文？

1. **确定文档类型**
   - 深度解析: 开创性、影响大 → 创建独立 `*_dissection.md`
   - 详细章节: 有创新点 → 在 `literature_review.md` 详细章节
   - 简短摘要: 参考价值 → 简短摘要
   - 仅索引: 保持关注 → 仅列标题+链接

2. **分类标记**
   - 技术分类: Diffusion/Flow/Tokenization/RL/架构
   - 公司分类: 按机构归类
   - 时间分类: 按年份

3. **更新索引**
   - 在 `paper_index.md` 相应分类中添加
   - 更新 `literature_review.md`（如需要）
   - 更新 `README.md`（如创建新文档）

### 分类标准

**技术分类**:
- 动作生成: Diffusion, Flow Matching, Tokenization, CVAE, 其他
- 训练方法: BC, RL (Offline/Online), 混合
- 架构: 单模型, 双系统, 层级
- 应用: 操作, 导航, 灵巧手

---

## 🔗 相关资源

- [完整文献综述](./literature_review.md) - 详细技术归纳
- [模型深度解析](./README.md#-part-5-模型详解-model-zoo) - 独立深度解析文档
- [理论总览](./README.md) - 返回理论目录

---

**最后更新**: 2025-12-06
**维护者**: VLA Handbook Team

