# 🧠 VLA 理论与核心算法

> **Vision-Language-Action** 模型的理论基础、核心算法与前沿架构。

```
┌─────────────────────────────────────────────────────────────────────┐
│                        📖 学习路线图                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Part 1          Part 2          Part 3          Part 4 & 5       │
│   ┌─────┐        ┌─────┐         ┌─────┐         ┌─────┐           │
│   │基础 │ ────▶  │ ML  │ ──────▶ │架构 │ ──────▶ │前沿 │           │
│   │基石 │        │基础 │         │算法 │         │模型 │           │
│   └─────┘        └─────┘         └─────┘         └─────┘           │
│   数据/空间       多模态/RL       Diffusion       π0/G0            │
│   动作/评估       迁移/蒸馏       Flow/FAST       WALL-OSS         │
│                                                                     │
│   ⏱️ ~2天         ⏱️ ~3天         ⏱️ ~3天         ⏱️ ~2天            │
└─────────────────────────────────────────────────────────────────────┘
```

| 🎯 快捷入口 | |
|:---|:---|
| 🤪 **[人话版 (看不下去八股文？)](./README_FUN.md)** | 用类比讲清楚核心概念 |
| 📊 **[ASCII 图鉴](./ascii_cheatsheet.md)** | 一页纸复习所有架构图 |
| 📚 **[文献综述](./literature_review.md)** | VLA 发展史全景图（按分类组织） |
| 🧭 **[研究主线梳理](./vla_research_mainline.md)** | 从 ACT/DP baseline 到「数据×感知×后训练」闭环 🆕 |
| 🧩 **[弹性模组化架构 Table 生成器](./modular_pipeline_table_generator.md)** | 从 `theory/` 文本抽象可拼装 pipeline：Stages + Table + 路径图 🆕 |
| 🔍 **[论文索引](./paper_index.md)** | 🆕 多维度快速查找（技术/公司/时间） |
| 🧠 **[RynnBrain 开源解读](./frontier/rynnbrain_open_embodied_foundation_models_2026.md)** | 阿里达摩院：物理现实锚定的具身基础模型（Plan/Nav/CoP + RynnBrain-Bench） 🆕 |
| 🖐️ **[TacRefineNet 开源解读](./frontier/tacrefinenet_tactile_only_grasp_refinement_2026.md)** | 小米机器人：tactile-only 抓取执行微调（goal-conditioned refinement） 🆕 |
| 🚀 **[DreamZero 深度解读](./dreamzero_world_action_models_zero_shot_policies_2026.md)** | NVIDIA：WAM 联合生成 video+action；DreamZero-Flash（1-step）+ 38× 加速，实现 7Hz 闭环控制 ⭐ |
| 🆕 **[VLA 十大挑战](./vla_challenges.md)** | NTU/斯坦福 2025 研究方向 |
| 🔥 **[小模型 VLA 研究](./small_vla_models.md)** | 边缘部署、SmolVLA、蒸馏压缩 |
| 🏆 **[NeurIPS 2025 解读](./neurips_2025_insights.md)** | 6 篇最佳论文的具身智能视角 |
| 🦿 **[Unitree UnifoLM-VLA-0 开源解读](./unifolm_vla_0_unitree_2026.md)** | Qwen2.5-VL + Flow Matching DiT 动作头 + LeRobot→RLDS 管线 + 真机 server 部署 🆕 |
| 🧭 **[产业路线：通用性与“元学习”路径](./frontier/industry_paths_to_generalization.md)** | 全栈整合 / 垂直突破 / 生态平台三路径（策略视角） |

---

## 📚 Part 1: 基础基石 (Foundations)

> *万丈高楼平地起，数据与动作空间是 VLA 的根基。*

| 主题 | 文件 | 核心内容 |
|:-----|:-----|:---------|
| 📦 **数据处理** | [`data.md`](./data.md) | RLDS vs LeRobot vs HDF5、数据加载流水线 |
| 🧭 **空间智能** | [`spatial_math.md`](./spatial_math.md) | 坐标系变换、四元数 vs 欧拉角 vs 6D Rotation |
| 🧮 **数学必备** | [`math_for_vla.md`](./math_for_vla.md) | 🆕 实现 VLA 必备的线代、概率、控制与几何直觉 |
| 🧩 **凸优化与 DCP** | [`dcp_convexity_rules.md`](./dcp_convexity_rules.md) | 🆕 CVX/CVXPY 的凸性检测规则与建模心法 |
| ⚙️ **动力学分类** | [`robot_dynamics_classification.md`](./robot_dynamics_classification.md) | 🆕 约束完备性、浮动基座与惯量矩阵 |
| 🖐️ **灵巧手机械学** | [`dexterous_hand_mechanics.md`](./dexterous_hand_mechanics.md) | 🆕 自由度分配、减速器、驱动流派与雅可比桥梁 |
| 🎮 **动作空间** | [`action_representations.md`](./action_representations.md) | 连续 vs 离散、Delta vs Absolute |
| 🔄 **联合训练** | [`co_training.md`](./co_training.md) | 防止灾难性遗忘、Loss Masking |
| 🧮 **Loss Functions 手册** | [`vla_loss_functions_handbook.md`](./vla_loss_functions_handbook.md) | BC/GMM/Diffusion/Flow/RL、安全正则、对齐损失 🆕 |
| 📝 **评估体系** | [`evaluation.md`](./evaluation.md) | CALVIN/SIMPLER、真机成功率 |

---

## 🎓 Part 2: 机器学习基础 (ML Fundamentals)

> *掌握 VLA 背后的核心 ML 技术，补齐知识短板。*

| 主题 | 文件 | 核心内容 |
|:-----|:-----|:---------|
| 🔮 **多模态模型** | [`multimodal_models.md`](./multimodal_models.md) | VLM 架构、Early/Mid/Late Fusion、SigLIP vs CLIP |
| 🎯 **自监督学习** | [`self_supervised_learning.md`](./self_supervised_learning.md) | 对比学习 (InfoNCE)、MAE、R3M |
| ✈️ **迁移学习** | [`transfer_learning.md`](./transfer_learning.md) | 跨形态迁移、Sim-to-Real、Domain Randomization |
| 📝 **知识蒸馏** | [`knowledge_distillation.md`](./knowledge_distillation.md) | 软标签、Temperature、VLA 压缩 |
| 🎮 **强化学习** | [`reinforcement_learning.md`](./reinforcement_learning.md) | PPO/SAC、Offline RL、Recap 算法 |
| 💭 **思维链** | [`chain_of_thought.md`](./chain_of_thought.md) | CoT/ReAct、Uni-CoT、分层规划 |

---

## 🧠 Part 3: 架构与算法 (Architecture & Algorithms)

> *理解模型是如何"思考"和"决策"的。*

### 🏗️ 核心架构

| 主题 | 文件 | 核心内容 |
|:-----|:-----|:---------|
| 🏛️ **VLA 架构** | [`vla_arch.md`](./vla_arch.md) | VLM Backbone + Action Head 设计范式 |
| ⚔️ **Transformer vs CNN** | [`transformer_vs_cnn.md`](./transformer_vs_cnn.md) | 为什么 Transformer 统治机器人学习 |

### 🎯 动作生成策略 (Policy Generation)

| 算法 | 文件 | 一句话总结 |
|:-----|:-----|:---------|
| **ACT** | [`act.md`](./act.md) | CVAE + 动作分块，ALOHA 核心 |
| **Diffusion Policy** | [`diffusion_policy.md`](./diffusion_policy.md) | 扩散去噪，解决多模态分布 |
| **RDT** | [`rdt.md`](./rdt.md) | 十亿参数扩散模型，双臂操作 |
| **RDT2** | [`./frontier/rdt2_umi_zero_shot_cross_embodiment_2026.md`](./frontier/rdt2_umi_zero_shot_cross_embodiment_2026.md) | UMI 数据规模化 + 跨本体零样本 + 单步蒸馏 |
| **Flow Matching** | [`pi0_flow_matching.md`](./pi0_flow_matching.md) | 比 Diffusion 快 5x，π0 核心 |
| **FAST** | [`fast.md`](./fast.md) | DCT 频域 Tokenization |
| **MM-ACT** | [`./frontier/vla_unified_token_space.md`](./frontier/vla_unified_token_space.md) | 🆕 全模态共享 Token 空间 |
| **传统方法** | [`traditional_action_generation.md`](./traditional_action_generation.md) | 🆕 MSE 回归与 GMM (基础) |

### ⚡ 效率优化 (Efficiency)

| 主题 | 文件 | 核心内容 |
|:-----|:-----|:---------|
| 🚀 **Flash Attention** | [`flash_attention.md`](./flash_attention.md) | Tiling + 重计算，显存 O(N²)→O(N) |
| 🧠 **KV Cache（推理）** | [`kv_cache_llm_inference.md`](./kv_cache_llm_inference.md) | Prefill/Decode、Paged/Radix、分层缓存与传输 |
| 🔧 **PEFT & LoRA** | [`peft_lora.md`](./peft_lora.md) | 低秩分解，QLoRA ~6GB 微调 7B |
| 📉 **量化理论** | [`quantization_theory.md`](./quantization_theory.md) | INT8/INT4、AWQ 原理 |
| 🧪 **QVLA 量化（2026）** | [`./frontier/qvla_action_centric_quantization_2026.md`](./frontier/qvla_action_centric_quantization_2026.md) | 🆕 动作空间敏感量化与通道级比特分配 |

---

## 🚀 Part 4: 进阶专题 (Advanced Topics)

> *解决特定场景下的难题，沉淀可复用的工程抓手与方法论。*

### 🛰️ 具身感知与定位 (Embodied Perception)
*侧重于机器人如何“看”和“感觉”物理世界。*

| 主题 | 文件 | 核心内容 |
| :--- | :--- | :--- |
| 👁️ **视觉感知技术** | [`perception_techniques.md`](./perception_techniques.md) | 检测/跟踪/Occupancy/BEV/位姿估计 |
| 🛰️ **点云 & SLAM** | [`pointcloud_slam.md`](./pointcloud_slam.md) | 点云语义、配准、Visual/LiDAR SLAM |
| 📡 **状态估计** | [`state_estimation.md`](./state_estimation.md) | EKF 噪声抑制（Q/R 调参）、UKF、粒子滤波、IMU+视觉融合 |
| 🖐️ **触觉专题入口** | [`./tactile/README.md`](./tactile/README.md) | 触觉路线图：VTLA / SaTA / 硬件选型 / 落地 checklist 🆕 |
| 🧠 **语言塑形感知** | [`./frontier/language_shapes_perception.md`](./frontier/language_shapes_perception.md) | “灰度香蕉”启示：语义先验影响视觉表征与属性推断 🆕 |
| 🧪 **Physics of AI** | [`./frontier/physics_of_ai_liuziming.md`](./frontier/physics_of_ai_liuziming.md) | 不赌规模：用“现象-观测量-规律”研究神经网络，沉淀工程抓手 🆕 |
| 🧊 **Zero-1-to-3（2023）** | [`./frontier/zero_1_to_3_zero_shot_one_image_to_3d_object_2023.md`](./frontier/zero_1_to_3_zero_shot_one_image_to_3d_object_2023.md) | 🆕 单图像零样本新视角合成 + 3D 重建先验 |
| 📈 **Classic GNNs（2024）** | [`./frontier/classic_gnns_strong_baselines_node_classification_2024.md`](./frontier/classic_gnns_strong_baselines_node_classification_2024.md) | 🆕 节点分类基线再评估：调参后经典 GNN 仍很强 |
| 🌍 **WorldEval（2025）** | [`./frontier/worldeval_world_model_policy_evaluator_2025.md`](./frontier/worldeval_world_model_policy_evaluator_2025.md) | 🆕 世界模型评测器：用生成式 rollout 评估机器人策略 |
| 🌍 **Ctrl-World × WorldArena（2026）** | [`./frontier/ctrl_world_worldarena_embodied_world_model_benchmark_2026.md`](./frontier/ctrl_world_worldarena_embodied_world_model_benchmark_2026.md) | 🆕 ICLR 2026 Ctrl-World + WorldArena 统一评测：视频质量×功能任务（Data Engine/Policy Eval/Action Planner） |
| 🌊 **WaveFormer（2026）** | [`./frontier/waveformer_wave_equation_vision_2026.md`](./frontier/waveformer_wave_equation_vision_2026.md) | 波动方程视觉建模：频率-时间解耦，保留高频细节 🆕 |
| 🐙 **软体本体感知** | [`./frontier/soft_robot_proprioception_gvs_sensitivity_ellipsoid.md`](./frontier/soft_robot_proprioception_gvs_sensitivity_ellipsoid.md) | GVS + 灵敏度椭球：把“不可观测性”变成可视化指标，并用于感知驱动规划 🆕 |

### 🧭 决策、规划与抓取 (Decision & Execution)
*侧重于机器人如何“思考”路径并进行“精细操作”。*

| 主题 | 文件 | 核心内容 |
| :--- | :--- | :--- |
| 🧭 **运动规划** | [`motion_planning.md`](./motion_planning.md) | RRT/PRM、TrajOpt、MoveIt & cuRobo |
| 🗺️ **具身导航 (VLN)** | [`vln_dualvln.md`](./vln_dualvln.md) | 🆕 DualVLN：慢规划/快执行的异步双系统 |
| 🤖 **抓取算法** | [`grasp_algorithms.md`](./grasp_algorithms.md) | DexGraspNet/GraspGF、抓取位姿生成 |

### 🧪 仿真底座与训练强化 (Sim & Augmentation)
*侧重于高效数据生产与模型性能保护。*

| 主题 | 文件 | 核心内容 |
| :--- | :--- | :--- |
| 🎮 **Isaac Lab** | [`isaac_lab.md`](./isaac_lab.md) | 🔥 GPU 仿真框架，单卡百万 FPS 🆕 |
| 🛡️ **知识绝缘** | [`knowledge_insulation.md`](./knowledge_insulation.md) | 微调时保护 VLM 通用常识，防止“智障” |
| 🧪 **VLA+RL 实战教程** | [`vla_rl_practical_guide.md`](./vla_rl_practical_guide.md) | Transformer VLA + RL 算法 + 工具链的落地路径 🆕 |

---

## 🦁 Part 5: 模型详解 (Model Zoo)

> *SOTA 模型的深度剖析，面试必考。*

### 📖 综述

| 文件 | 内容 |
|:-----|:-----|
| 📚 **[文献综述](./literature_review.md)** | **(必读)** 按技术分类组织，RT-1/2 → OpenVLA → π0 发展脉络 |
| 🔍 **[论文索引](./paper_index.md)** | 🆕 多维度索引系统（技术/公司/时间） |

### 🔬 模型深度解析

| 公司 | 模型 | 文件 | 核心亮点 |
|:-----|:-----|:-----|:---------|
| **Physical Intelligence** | π0 ⭐ | [`pi0_flow_matching.md`](./pi0_flow_matching.md) | **开源 (OpenPI)**, Flow Matching 核心 |
| | | [`pi0_code_analysis.md`](./pi0_code_analysis.md) | OpenPI 代码架构深度解析 |
| | π0.5 | [`pi0_5_dissection.md`](./pi0_5_dissection.md) | Flow Matching + 隐式推理 |
| | π0.6 | [`pi0_6_dissection.md`](./pi0_6_dissection.md) | Recap 自我进化 + Action Expert |
| **GigaBrain Team / GigaAI** | GigaBrain-0.5M* | [`gigabrain_0_5m_star_world_model_based_rl_ramp_2026.md`](./gigabrain_0_5m_star_world_model_based_rl_ramp_2026.md) | 🆕 RAMP：世界模型条件化 RL（future latent + value）+ HILR 自我进化闭环；长时程任务相对 RECAP ~+30% |
| **ByteDance Seed** | GR-RL | [`gr_rl_dissection.md`](./gr_rl_dissection.md) | MoT 架构 + 三阶段 RL 训练 |
| **Microsoft** | VITRA | [`./frontier/vitra_scalable_vla_pretraining_human_activity_videos_2026.md`](./frontier/vitra_scalable_vla_pretraining_human_activity_videos_2026.md) | 🆕 真实人类手部视频 → VLA 预训练（1M+ episodes），zero-shot 手部动作预测，少量真机数据可适配 |
| **NVIDIA** | GR00T-N1.6 | [`gr00t_n1_6.md`](./gr00t_n1_6.md) | 🆕 双系统 DiT 架构 + Isaac Lab 仿真 |
| | DreamZero | [`dreamzero_world_action_models_zero_shot_policies_2026.md`](./dreamzero_world_action_models_zero_shot_policies_2026.md) | 🆕 WAM：联合预测视频+动作，14B video diffusion 做到 7Hz 闭环控制 |
| **Spirit AI** | Spirit-v1.5 | [`spirit_v1_5_dissection.md`](./spirit_v1_5_dissection.md) | 🆕 Qwen3-VL + DiT，RoboChallenge Table30 代码级复现指南 |
| **优必选（UBTECH Robotics）** | Thinker（4B/7B） | [`./frontier/thinker_vlm_embodied_intelligence_2026.md`](./frontier/thinker_vlm_embodied_intelligence_2026.md) | 🆕 具身任务规划 VLM：ego-view 数据配方 + “末帧 keyframe + 全视频”输入协议，缓解视角混淆与 ending 忽略 |
| **Robbyant Team** | LingBot-VLA | [`lingbot_vla_pragmatic_vla_foundation_model_2026.md`](./lingbot_vla_pragmatic_vla_foundation_model_2026.md) | 🆕 实用主义 VLA：可换 VLM（Qwen2.5-VL / PaliGemma）+ Action Expert + Flow Matching + 高吞吐训练栈 |
| **StarVLA** | StarVLA | [`../deployment/starvla_lego_like_vla_codebase_2026.md`](../deployment/starvla_lego_like_vla_codebase_2026.md) | 🆕 Lego-like VLA 研发底座：同一套 bench 上对比 FAST/OFT/Flow/双系统等框架（Framework/Dataloader/Trainer/Eval 解耦） |
| **X² (自变量)** | WALL-OSS | [`wall_oss.md`](./wall_oss.md) | Uni-CoT 边想边动 |
| **Galaxea AI** | G0 | [`galaxea_g0.md`](./galaxea_g0.md) | 大脑+小脑双系统 |

### 🧪 研究前沿与特定案例 (Research Frontier)

| 模型 | 文件 | 核心亮点 |
|:-----|:-----|:---------|
| **LLM 推理专题** | [`./llm_reasoning/README.md`](./llm_reasoning/README.md) | 🆕 分治推理 / 具身 CoT / 可提示表征与推理迁移 |
| **Data Flywheel** | [`./frontier/data_flywheel_and_cross_modal.md`](./frontier/data_flywheel_and_cross_modal.md) | 🆕 互联网视频学习、跨模态迁移与数据演进 |
| **Physical Intelligence Layer（2026）** | [`./frontier/physical_intelligence_layer_robot_api_2026.md`](./frontier/physical_intelligence_layer_robot_api_2026.md) | 🆕 “机器人基础模型 API / 物理智能层”：以自治率/干预/吞吐定义 SLO，Weave/Ultra 真机部署验证 |
| **RynnBrain（2026）** | [`./frontier/rynnbrain_open_embodied_foundation_models_2026.md`](./frontier/rynnbrain_open_embodied_foundation_models_2026.md) | 🆕 物理现实锚定的具身基础模型：指向/轨迹/规划统一输出 + RynnBrain-Bench（Object/Spatial/Grounding/Pointing） |
| **TacRefineNet（2026）** | [`./frontier/tacrefinenet_tactile_only_grasp_refinement_2026.md`](./frontier/tacrefinenet_tactile_only_grasp_refinement_2026.md) | 🆕 仅触觉的抓取执行末端微调：多指触觉 + 本体融合，目标驱动回归 wrist 6DoF 增量 |
| **RDT2（2026）** | [`./frontier/rdt2_umi_zero_shot_cross_embodiment_2026.md`](./frontier/rdt2_umi_zero_shot_cross_embodiment_2026.md) | 🆕 UMI 数据规模化 + 跨本体零样本部署 + 单步蒸馏 |
| **Shallow-π（2026）** | [`./frontier/shallow_pi_knowledge_distillation_flow_vla_2026.md`](./frontier/shallow_pi_knowledge_distillation_flow_vla_2026.md) | 🆕 Flow-based VLA 层深蒸馏，端侧推理加速 |
| **DAC-RL（2026）** | [`./llm_reasoning/dac_rl_divide_conquer_reasoning_2026.md`](./llm_reasoning/dac_rl_divide_conquer_reasoning_2026.md) | 🆕 分治推理训练，提升测试时可扩展性 |
| **TinyLoRA（2026）** | [`./llm_reasoning/tiny_lora_13_params_reasoning_2026.md`](./llm_reasoning/tiny_lora_13_params_reasoning_2026.md) | 🆕 13 参数推理微调：RL 信号激活推理模式 |
| **IntelliFold 2（2026）** | [`./frontier/intellifold_2_surpassing_alphafold3_structural_consistency_2026.md`](./frontier/intellifold_2_surpassing_alphafold3_structural_consistency_2026.md) | 🆕 生成式科学智能：共折叠结构预测；latent scaling + 多尺度一致性 +（Pro）PPO 稳采样 + 难例加权 |
| **Reward Discovery** | [`./frontier/reward_discovery_rl.md`](./frontier/reward_discovery_rl.md) | 🆕 Nature Comm: 遗憾最小化元学习奖励发现 |
| **Vicarious Maps** | [`./frontier/vicarious_body_maps.md`](./frontier/vicarious_body_maps.md) | 🆕 Nature 2025: 视触觉“感同身受”的神经基础 |
| **UniTacHand** | [`./frontier/unitachhand.md`](./frontier/unitachhand.md) | 🆕 arXiv 2025: MANO UV Map 统一触觉表征，实现人手→机器人零样本迁移 |
| **Tactile Outlook** | [`./frontier/tactile_irreplaceable.md`](./frontier/tactile_irreplaceable.md) | 🆕 触觉为何不可替代：力-形-质、闭环控制与产品化瓶颈 |
| **Policy Consensus（2025）** | [`./frontier/policy_consensus_multimodal_manipulation_2025.md`](./frontier/policy_consensus_multimodal_manipulation_2025.md) | 🆕 策略级多模态融合：专家策略共识替代特征拼接，解决触觉稀疏噪声化 |
| **SuperTac + DOVE** | [`../deployment/perception/supertac_dove_biomimetic_multimodal_tactile_sensing.md`](../deployment/perception/supertac_dove_biomimetic_multimodal_tactile_sensing.md) | 🆕 多模态电子皮肤（多光谱+摩擦电+IMU+温度/接近/振动）+ 触觉语言模型：让接触相位更可观测、更可解释 |
| **Jim Fan 2025** | [`./frontier/jim_fan_2025_robotics_lessons.md`](./frontier/jim_fan_2025_robotics_lessons.md) | 🆕 行业复盘：硬件可靠性、评测可复现性、VLM→VLA 路线反思 |
| **OneTwoVLA** | [`./frontier/onetwovla.md`](./frontier/onetwovla.md) | 🆕 统一模型 + 自适应推理切换 |
| **1X World Model** | [`./frontier/one_x_world_model.md`](./frontier/one_x_world_model.md) | 🆕 视频世界模型 + 逆动力学（IDM）：先“想象”再“执行” |
| **Figure Helix 02 (2026)** | [`./frontier/figure_helix_02_full_body_autonomy_2026.md`](./frontier/figure_helix_02_full_body_autonomy_2026.md) | 🆕 全身端到端 VLA：S2 语义目标 → S1 200Hz 全身关节目标 → S0 1kHz 执行（人类运动先验替代大量手写控制） |
| **VT Pretraining + Online Multitask (SciRobotics 2026)** | [`./frontier/visual_tactile_pretraining_online_multitask_learning_2026.md`](./frontier/visual_tactile_pretraining_online_multitask_learning_2026.md) | 🆕 单目 + 二值触觉：视触觉自监督预训练 + RL + 在线模仿，统一策略覆盖多项灵巧手任务 |
| **UniVTAC（2026）** | [`./tactile/univtac_unified_visuo_tactile_simulation_platform_2026.md`](./tactile/univtac_unified_visuo_tactile_simulation_platform_2026.md) | 🆕 统一视触觉仿真底座：多传感器数据合成 + UniVTAC Encoder + 8 任务 Benchmark（含真机提升验证） |
| **Active Tactile Exploration（ICRA 2026）** | [`./tactile/active_tactile_exploration_rigid_body_pose_shape_eig_2026.md`](./tactile/active_tactile_exploration_rigid_body_pose_shape_eig_2026.md) | 🆕 仅触觉：violation-implicit loss 同时学刚体轨迹+凸几何；EIG 主动探索加速收敛（<10s） |
| **TaF-VLA（2026）** | [`./frontier/taf_vla_tactile_force_alignment_2026.md`](./frontier/taf_vla_tactile_force_alignment_2026.md) | 🆕 触觉-力对齐：把触觉变成物理力语义，提升接触阶段稳定性 |
| **Video World Model Survey (2026)** | [`./frontier/video_generation_models_in_robotics_survey_2026.md`](./frontier/video_generation_models_in_robotics_survey_2026.md) | 🆕 综述：视频生成模型作为具身世界模型（数据生成/视觉规划/策略评估）；难点是物理幻觉、指令偏差、成本与安全 |
| **GenieReasoner / ERIQ / FACT** | [`./llm_reasoning/geniereasoner_eriq_fact.md`](./llm_reasoning/geniereasoner_eriq_fact.md) | 🆕 量化“推理→动作”传递损耗：推理基准 + 动作分词器 + 统一自回归 |
| **GR-Dexter** | [`./frontier/gr_dexter_bimanual_dexterous_vla.md`](./frontier/gr_dexter_bimanual_dexterous_vla.md) | 🆕 ByteDance Seed：把 VLA 扩展到 21-DoF 灵巧手的全栈框架（硬件/遥操作/跨形态数据） |
| **开可乐/发牌（灵巧手）** | [`./frontier/dexterous_hands_open_can_cards_data_pyramid.md`](./frontier/dexterous_hands_open_can_cards_data_pyramid.md) | 🆕 为什么“开可乐/发扑克牌”比“倒酒/洗碗机”难一个数量级：硬件三路线 × 触觉 × 数据金字塔 |
| **中金（灵巧手）：工程约束→可计算变量** | [`./frontier/dexterous_hand_industry_cicc_05.md`](./frontier/dexterous_hand_industry_cicc_05.md) | 🆕 将产业“工程化瓶颈”映射为热/惯量/可观测性约束，并回链到 `companies/industry_reports` 的报告 digest |
| **Ken Goldberg 对谈** | [`./frontier/ken_goldberg_data_quality_infrastructure.md`](./frontier/ken_goldberg_data_quality_infrastructure.md) | 🆕 GOFE 回归：瓶颈时刻数据、VLM 数据治理、Fog Robotics 与基础设施价值 |
| **Levine 抓取（2016）** | [`./classics/levine_hand_eye_coordination_grasping_2016.md`](./classics/levine_hand_eye_coordination_grasping_2016.md) | 🆕 近百万级真实抓取数据 + 动作评分器 + 闭环视觉伺服：用“候选+评分+小步纠错”把抓取跑稳 |
| **Levine 像素→扭矩（2016）** | [`./classics/levine_end_to_end_visuomotor_policies_2016.md`](./classics/levine_end_to_end_visuomotor_policies_2016.md) | 🆕 End-to-End visuomotor：用 GPS 把 RL 变成监督学习，学习像素→电机扭矩策略 |
| **Levine GPS（2014）** | [`./classics/levine_gps_unknown_dynamics_2014.md`](./classics/levine_gps_unknown_dynamics_2014.md) | 🆕 未知动力学 + KL trust region 的轨迹优化，并蒸馏到神经网络策略（训练更安全、样本更高效） |
| **Embodied CoT（2024）** | [`./llm_reasoning/embodied_chain_of_thought_robotic_control_2024.md`](./llm_reasoning/embodied_chain_of_thought_robotic_control_2024.md) | 🆕 具身思维链：训练 VLA 先推理（plan/subtask/move/bbox/gripper）再出动作，显著提升真实世界泛化，并支持语言纠错 |
| **PR2L（2024/2025）** | [`./llm_reasoning/vlm_promptable_representations_for_rl_pr2l_2025.md`](./llm_reasoning/vlm_promptable_representations_for_rl_pr2l_2025.md) | 🆕 VLM 可提示表征：用 task-relevant prompt 把常识注入 state embedding，再用 RL/BC 做 grounding（Habitat CoT 1.5× 提升） |
| **U2O RL（2024）** | [`./frontier/unsupervised_to_online_reinforcement_learning_u2o_2024.md`](./frontier/unsupervised_to_online_reinforcement_learning_u2o_2024.md) | 🆕 Unsupervised-to-Online RL：用无监督离线技能预训练替代任务特定离线预训练，提升复用性与在线微调稳定性 |
| **PGR（2024/2025）** | [`./frontier/prioritized_generative_replay_pgr_2025.md`](./frontier/prioritized_generative_replay_pgr_2025.md) | 🆕 Prioritized Generative Replay：用条件扩散生成式回放 densify+guide 关键经验，提升在线样本效率并支撑更高 UTD |
| **DKT Perception** | [`./frontier/dkt_transparency_perception.md`](./frontier/dkt_transparency_perception.md) | 🆕 基于视频扩散先验的透明物体深度/法向估计 |
| **MM-ACT** | [`./frontier/vla_unified_token_space.md`](./frontier/vla_unified_token_space.md) | 全模态共享 Token 空间 |
| **SGTM** | [`./frontier/vla_intrinsic_safety.md`](./frontier/vla_intrinsic_safety.md) | 本质安全与知识屏蔽 |
| **RLinf（VLA+RL Infra）** | [`./frontier/rlinf_vla_rl_training.md`](./frontier/rlinf_vla_rl_training.md) | 🆕 VLA+RL 训练“生产线”：rollout / 数据面 / 评估可复现 |
| **ReconVLA** | [`./frontier/reconvla_implicit_grounding_by_reconstruction.md`](./frontier/reconvla_implicit_grounding_by_reconstruction.md) | 🆕 隐式视觉接地：用“重建 gaze region”逼注意力对准目标 |
| **鸽子磁感（前庭电流检测）** | [`./frontier/pigeon_magnetoreception_vestibular_electrosense.md`](./frontier/pigeon_magnetoreception_vestibular_electrosense.md) | 🆕 生物磁导航：内耳前庭可能是“磁感”器官（Science 2025 + Nature news） |
| **皮层下“控制旋钮”** | [`./frontier/subcortical_control_knobs_neuropeptides_temporality.md`](./frontier/subcortical_control_knobs_neuropeptides_temporality.md) | 🆕 神经肽/调制×时间尺度×输入门控：从生存行为理解“皮层下计算” |

---

## 🛡️ Part 6: 安全、对齐与本质安全 (Safety & Alignment)

> *不仅要动得准，更要动得稳。探讨 VLA 在物理世界的最后一道防线。*

| 主题 | 文件 | 核心内容 |
|:-----|:-----|:---------|
| 🛡️ **本质安全 (SGTM)** | [`./frontier/vla_intrinsic_safety.md`](./frontier/vla_intrinsic_safety.md) | 🆕 参数级“脑切除”，平衡安全与学习能力 |
| ⚖️ **对齐技术** | [`alignment_vla.md`](./alignment_vla.md) | 具身 RLHF、受限马尔可夫决策过程 (CMDP) |
| 🛡️ **知识绝缘** | [`knowledge_insulation.md`](./knowledge_insulation.md) | 微调时保护 VLM 通用常识 |
| 🖐️ **灵巧手案例** | [`../deployment/dexterous_hand_wuji.md`](../deployment/dexterous_hand_wuji.md) | 🆕 Wuji (独立驱动) vs. [Optimus V2](../deployment/optimus_hand_v2.md) (肌腱驱动) |

---

## 🛠️ Part 7: 实战案例与部署 (Case Studies & Deployment)

> *理论与实践的交汇点。展示如何将 VLA 与感知算法落地到真实硬件。*

| 主题 | 文件 | 核心内容 |
| :--- | :--- | :--- |
| 🖐️ **手势控制灵巧手** | [`../deployment/mediapipe_wujihand_project.md`](../deployment/mediapipe_wujihand_project.md) | **MediaPipe + WujiHand**: 实时控制系统设计与 **500ms -> 50ms** 延迟优化实践 🆕 |
| 🛡️ **灵巧手硬件对比** | [`../deployment/dexterous_hand_wuji.md`](../deployment/dexterous_hand_wuji.md) | Wuji (独立驱动) vs. [Optimus V2](../deployment/optimus_hand_v2.md) (肌腱驱动) |

---

## 🎯 学习建议

```
┌─────────────────────────────────────────────────────────────────────┐
│  👤 你是谁？                    📖 建议路线                          │
├─────────────────────────────────────────────────────────────────────┤
│  🌱 VLA 新手                    Part 1 → Part 3 (ACT/Diffusion)     │
│  📚 ML 基础薄弱                  Part 2 (重点: 多模态、RL)            │
│  🔧 想做工程落地                 Part 3 效率优化 + Part 5 OpenVLA     │
│  🎓 准备大厂面试                 全部 + Part 4 (工程抓手/方法论)       │
│  ⏰ 只有 1 天                   README_FUN.md + 文献综述             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📄 推荐论文

<details>
<summary><b>VLA 核心 (点击展开)</b></summary>

- [RT-1: Robotics Transformer for Real-World Control at Scale](https://arxiv.org/abs/2212.06817)
- [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818)
- [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246)
- [π0: A Vision-Language-Action Flow Model for General Robot Control](https://www.physicalintelligence.company/blog/pi0)

</details>

<details>
<summary><b>策略学习 (点击展开)</b></summary>

- [ACT: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)
- [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)
- [RDT-1B: A Diffusion Foundation Model for Bimanual Manipulation](https://arxiv.org/abs/2410.07864)

</details>

<details>
<summary><b>机器学习基础 (点击展开)</b></summary>

- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [MAE: Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Chain-of-Thought Prompting Elicits Reasoning in LLMs](https://arxiv.org/abs/2201.11903)
- [PPO: Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

</details>

---

[← 返回主页](../README.md)
