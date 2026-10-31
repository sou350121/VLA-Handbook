# 🌊 Flow River — 扩散策略与 Flow Matching

> 机器人的动作是「画」出来的，不是「算」出来的。

📚 下方为导读选编；[完整目录与当前文档数](#theory-complete-index)见文末。

**阅读建议**：先用[动作表示总览](./action_representations.md)比较离散、扩散与 Flow 路线，再读 [Diffusion Policy](./diffusion_policy.md) 和 [Flow Matching](./pi0_flow_matching.md)。[CFG-Policy](./cfg_policy_unified_diffusion_policy_learning_framework_via_c_dissection.md)可作为进一步选读。

## 🚀 从哪里开始？

| 你是谁 | 推荐 | 为什么 |
|--------|------|--------|
| 🆕 新手 | [Diffusion Flow Mainline](./diffusion_flow_mainline.md) | 动作生成的完整研究地图 |

## 📚 主线 / 工具文章

- [**Diffusion Flow Mainline**](./diffusion_flow_mainline.md)

---

## 🗂️ 深度文章（按主题分组）

### Diffusion / Flow 主线 <code>2</code>

- [Π0 Flow Matching](./pi0_flow_matching.md)
- [Snapflow One Step Action Generation For Flow Matching VLAs V](./snapflow_one_step_action_generation_for_flow_matching_vlas_v_dissection.md)

### ⭐ Flow Matching 实战 <code>1</code>

- [Flow With The Force Field Learning 3D Compliant Flow Matchin](./flow_with_the_force_field_learning_3d_compliant_flow_matchin_dissection.md)

### Diffusion Policy 系列 <code>1</code>

- [Diffusion Policy](./diffusion_policy.md)

### Action Tokenization vs Continuous <code>1</code>

- [The Compression Gap Why Discrete Tokenization Limits Vision](./the_compression_gap_why_discrete_tokenization_limits_vision_dissection.md)

### 🗃️ 其他深度文章 <code>10</code>

- [Action Representations](./action_representations.md)
- [Characterizing VLA Models Identifying The Action Generation](./characterizing_vla_models_identifying_the_action_generation_dissection.md)
- [Closed Loop Action Chunks With Dynamic Corrections For Train](./closed_loop_action_chunks_with_dynamic_corrections_for_train_dissection.md)
- [Como Learning Continuous Latent Motion From Internet Videos](./como_learning_continuous_latent_motion_from_internet_videos_dissection.md)
- [Contractive Diffusion Policies Robust Action Diffusion Via C](./contractive_diffusion_policies_robust_action_diffusion_via_c_dissection.md)
- [Neural Implicit Action Fields From Discrete Waypoints To Con](./neural_implicit_action_fields_from_discrete_waypoints_to_con_dissection.md)
- [Pixel Motion Diffusion Is What We Need For Robot Control](./pixel_motion_diffusion_is_what_we_need_for_robot_control_dissection.md)
- [Traditional Action Generation](./traditional_action_generation.md)
- [VLA Unified Token Space](./vla_unified_token_space.md)
- [X Diffusion Training Diffusion Policies On Cross Embodiment](./x_diffusion_training_diffusion_policies_on_cross_embodiment_dissection.md)

---

## 🔗 跨 Realm 導航

[← Back to Theory Index](../README.md) · [↑ VLA-Handbook 主頁](../../README.md)

[🏛️](../vla-core/README.md) · [🌍](../world-model/README.md) · [🧠](../planning/README.md) · [🎮](../rl/README.md) · [🤚](../tactile/README.md) · [👁️](../perception/README.md) · [🏗️](../foundation/README.md) · [🔧](../deployment/README.md) · [🔬](../frontier/README.md)

---

<!-- VLA:THEORY-NAV:BEGIN -->
<a id="theory-complete-index"></a>
## 完整目录（自动更新）

> 上方保留人工导读与主题分组。本节列出本目录内全部 Git 已追踪 Markdown 文件，包含综述与索引，不含 README。按文件路径排序。

**共 52 份文档。**

- [动作生成范式详解 (Action Representations &amp; Generation)](./action_representations.md)
- [动作到动作流匹配 (Action-to-Action Flow Matching)](./action_to_action_flow_matching_dissection.md)
- [ActSafeGuard：把硬约束注入流匹配策略的可微「射线缩放」安全层 (ActSafeGuard: Differentiable and Training-Aligned Constraint Enforcement for Flow-Matching Policies)](./actsafeguard_differentiable_and_training_aligned_constraint_dissection.md)
- [AdaVLA：免训练自适应步数流匹配，让 VLA 推理提速 2 倍 (AdaVLA: Adaptive Step Flow Matching for Training-free Acceleration of Vision-Language-Action Models)](./adavla_adaptive_step_flow_matching_for_training_free_acceler_dissection.md)
- [AsyncVLA：异步流匹配视觉-语言-动作模型 (AsyncVLA: Asynchronous Flow Matching for Vision-Language-Action Models)](./asyncvla_asynchronous_flow_matching_for_vision_language_acti_dissection.md)
- [分块自适应缓存加速扩散策略 (Block-wise Adaptive Caching for Accelerating Diffusion Policy)](./block_wise_adaptive_caching_for_accelerating_diffusion_polic_dissection.md)
- [CF-VLA：高效粗到细动作生成 (CF-VLA: Efficient Coarse-to-Fine Action Generation for Vision-Language-Action Policies)](./cf_vla_efficient_coarse_to_fine_action_generation_for_vision_dissection.md)
- [CFG-Policy：用 Classifier-free Guidance 统一扩散策略的采样、探索与 RL 微调 (CFG-Policy: Unified Diffusion Policy Learning Framework via Classifier-free Guidance)](./cfg_policy_unified_diffusion_policy_learning_framework_via_c_dissection.md)
- [瓶颈定位：VLA 模型动作生成的边缘架构困境 (Characterizing VLA Models: Identifying the Action Generation Bottleneck for Edge AI Architectures)](./characterizing_vla_models_identifying_the_action_generation_dissection.md)
- [闭环动作块动态校正：训练-free 扩散策略实时适应 (Closed-Loop Action Chunks with Dynamic Corrections for Training-Free Diffusion Policy)](./closed_loop_action_chunks_with_dynamic_corrections_for_train_dissection.md)
- [CoMo：从互联网视频学习连续潜在运动 (CoMo: Learning Continuous Latent Motion from Internet Videos for Scalable Robot Learning)](./como_learning_continuous_latent_motion_from_internet_videos_dissection.md)
- [收缩扩散策略：通过收缩微分方程实现鲁棒动作扩散 (Contractive Diffusion Policies: Robust Action Diffusion via Contractive Score-Based Sampling with Differential Equations)](./contractive_diffusion_policies_robust_action_diffusion_via_c_dissection.md)
- [域自适应扩散策略 (Domain Adaptive Diffusion Policy)](./dadp_domain_adaptive_diffusion_policy_dissection.md)
- [解耦式早退：為 Flow-Matching VLA 做任務相關的算力分配 (Decoupled Early Exits for Task-Dependent Compute Allocation in Flow-Matching VLAs)](./decoupled_early_exits_for_task_dependent_compute_allocation_dissection.md)
- [DFM-VLA：基于离散流匹配的迭代动作细化 (DFM-VLA: Iterative Action Refinement for Robot Manipulation via Discrete Flow Matching)](./dfm_vla_iterative_action_refinement_for_robot_manipulation_v_dissection.md)
- [二分扩散策略优化 (Dichotomous Diffusion Policy Optimization)](./dichotomous_diffusion_policy_optimization_dissection.md)
- [🌊 扩散与流匹配 — 动作生成主线总纲](./diffusion_flow_mainline.md)
- [扩散策略详解 (Diffusion Policy)](./diffusion_policy.md)
- [DreamTrajectory：基于轨迹引导与世界模型对齐的移动操作 VLA (DreamTrajectory: Trajectory-Guided Action Generation with World Model Alignment for Mobile Manipulation)](./dreamtrajectory_trajectory_guided_action_generation_with_wor_dissection.md)
- [DSSP：基于全历史编码的扩散状态空间策略 (Diffusion State Space Policy with Full-History Encoding)](./dssp_diffusion_state_space_policy_with_full_history_encoding_dissection.md)
- [力场流匹配：从单演示生成力觉数据学习 3D 顺应性策略 (Flow with the Force Field: Learning 3D Compliant Flow Matching Policies from Force and Demonstration-Guided Simulation Data)](./flow_with_the_force_field_learning_3d_compliant_flow_matchin_dissection.md)
- [Freeze, Share, Shrink：冻结、共享、压缩——重思扩散策略的动作骨干 (Freeze, Share, Shrink: Rethinking the Action Backbone in Diffusion Policies)](./freeze_share_shrink_rethinking_the_action_backbone_in_diffus_dissection.md)
- [GeCO：时间无条件流匹配用于自适应鲁棒机器人控制 (Generative Control as Optimization: Time Unconditional Flow Matching for Adaptive and Robust Robotic Control)](./generative_control_as_optimization_time_unconditional_flow_m_dissection.md)
- [Hyper-DP3：频域感知的3D扩散策略轻量化重构 (Frequency-Aware Right-Sizing of 3D Diffusion Policies for Visuomotor Control)](./hyper_dp3_frequency_aware_right_sizing_of_3d_diffusion_polic_dissection.md)
- [IMLE-VLA：用条件隐式最大似然估计实现单步动作生成 (IMLE-VLA: Fast Single-Step Action Generation for Vision-Language-Action Policies)](./imle_vla_fast_single_step_action_generation_for_vision_langu_dissection.md)
- [从二值可行性 Oracle 学习安全的 Flow-Matching 机器人策略 (Learning Safe Flow-Matching Robot Policies from Binary Feasibility Oracles)](./learning_safe_flow_matching_robot_policies_from_binary_feasi_dissection.md)
- [Lie 群 B 样条扩散策略：让扩散策略的轨迹真正平滑 (LieSpline-DP: Lie-Group B-Spline Diffusion Policy for Smooth Robot Manipulation)](./liespline_dp_lie_group_b_spline_diffusion_policy_for_smooth_dissection.md)
- [LTLf 约束引导的多智能体扩散策略 (LTLDiff: Finite Linear Temporal Logic-Guided Data Generation and Diffusion Policies for Multi-agent Robotic Manipulation)](./ltldiff_finite_linear_temporal_logic_guided_data_generation_dissection.md)
- [MOSS：面向机器人流匹配策略的模式导向源分布塑造 (Mode-Oriented Source Shaping for Robot Flow Matching Policies)](./moss_mode_oriented_source_shaping_for_robot_flow_matching_po_dissection.md)
- [神经隐式动作场：从离散路点到连续函数 (Neural Implicit Action Fields: From Discrete Waypoints to Continuous Functions for Vision-Language-Action Models)](./neural_implicit_action_fields_from_discrete_waypoints_to_con_dissection.md)
- [噪声空间归因与分块边界伪影控制 (Noise-Space Attribution and Control of Chunk-Boundary Artifact)](./noise_space_attribution_and_control_of_chunk_boundary_artifa_dissection.md)
- [OMP：单步均值流策略与方向对齐 (One-step MeanFlow Policy with Directional Alignment)](./omp_one_step_meanflow_policy_with_directional_alignment_dissection.md)
- [ORCHID：分层扩散策略的在线自训练共适应 (Online Self-Training for Co-Adaptation in Hierarchical Diffusion Policies)](./online_self_training_for_co_adaptation_in_hierarchical_diffu_dissection.md)
- [Pi0 (π0) 代码解构：Flow Matching for VLA](./pi0_flow_matching.md)
- [Pixel Motion Diffusion is What We Need for Robot Control (DAWN)](./pixel_motion_diffusion_is_what_we_need_for_robot_control_dissection.md)
- [势函数引导的 Flow Matching 用于 VLA 策略优化 (Potential-Guided Flow Matching for Vision-Language-Action Policy Improvement)](./potential_guided_flow_matching_for_vision_language_action_po_dissection.md)
- [VLA 的 RL 精调突破：Flow Policy Optimization (FPO)](./reinforcement_fine_tuning_of_flow_matching_policies_for_visi_dissection.md)
- [通过屏障增强流匹配实现安全的视觉语言动作模型 (Safe Vision Language Action Models via Barrier Enhanced Flow Matching)](./safe_vision_language_action_models_via_barrier_enhanced_flow_dissection.md)
- [SCRIPT：面向语言驱动仿真人形控制的可扩展扩散策略 (Scalable Diffusion Policy with Multi-stage Training for Language-driven Physics-Based Humanoid Control)](./script_scalable_diffusion_policy_with_multi_stage_training_f_dissection.md)
- [语义锚定：把 VLA 的 action 表示拉回语义流形 (Semantic Anchoring for Robotic Action Representations)](./semantic_anchoring_for_robotic_action_representations_dissection.md)
- [集合监督扩散策略：通过修正学习动作分块扩散 (Set-Supervised Diffusion Policy: Learning Action-Chunking Diffusion through Corrections)](./set_supervised_diffusion_policy_learning_action_chunking_dif_dissection.md)
- [SnapFlow：流匹配 VLA 的单步动作生成 (SnapFlow: One-Step Action Generation for Flow-Matching VLAs via Progressive Self-Distillation)](./snapflow_one_step_action_generation_for_flow_matching_vlas_v_dissection.md)
- [STEP：时空一致性预测的热启动视觉运动策略 (Warm-Started Visuomotor Policies with Spatiotemporal Consistency Prediction)](./step_warm_started_visuomotor_policies_with_spatiotemporal_co_dissection.md)
- [压缩鸿沟：为何离散 Tokenization 限制 VLA 模型Scaling (The Compression Gap: Why Discrete Tokenization Limits Vision-Language-Action Model Scaling)](./the_compression_gap_why_discrete_tokenization_limits_vision_dissection.md)
- [流匹配不确定性的几何本质：零成本不确定性代理及其在流式 VLA 失败检测中的应用 (The Geometry of Flow-Matching Uncertainty: A Cost-free Uncertainty Proxy and Its Application in Flow-based VLA Failure Detection)](./the_geometry_of_flow_matching_uncertainty_a_cost_free_uncert_dissection.md)
- [TraceFlow：用成功/失败轨迹引导冻结的流匹配机器人策略 (TraceFlow: Guiding Frozen Flow-Matching Robot Policies with Success and Failure Traces)](./traceflow_guiding_frozen_flow_matching_robot_policies_with_s_dissection.md)
- [传统动作生成方法 (Traditional Action Generation)](./traditional_action_generation.md)
- [统一对象中心世界模型与扩散策略：多阶段机器人任务的分层框架 (Unifying Object-Centric World Models and Diffusion Policy: A Hierarchical Framework for Multi-Stage Robotic Tasks)](./unifying_object_centric_world_models_and_diffusion_policy_a_dissection.md)
- [全模态共享 Token 空间：以 MM-ACT 为例的 VLA 进化论](./vla_unified_token_space.md)
- [一步 Euler 够不够？Flow-Matching VLA 的步数等价性实证 (When One Step Is Enough: An Empirical Study of Euler Integration in Flow-Matching Policies)](./when_one_step_is_enough_an_empirical_study_of_euler_integrat_dissection.md)
- [X-Diffusion: 跨具身人类演示训练扩散策略 (X-Diffusion: Training Diffusion Policies on Cross-Embodiment Human Demonstrations)](./x_diffusion_training_diffusion_policies_on_cross_embodiment_dissection.md)
- [XS-VLA: 粗粒度空间蒸馏 + 潜在流匹配的轻量级 VLA 控制 (XS-VLA: Coupling Coarse-grained Spatial Distillation with Latent Flow Matching for Lightweight Robotic Control)](./xs_vla_coupling_coarse_grained_spatial_distillation_with_lat_dissection.md)

<!-- VLA:THEORY-NAV:END -->
