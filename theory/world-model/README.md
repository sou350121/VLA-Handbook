# 🌍 World Realm — 世界模型与仿真

> 让机器人在「脑子里」先试一遍。从 evaluator 到 planner 到 World-Action Model。

📚 下方为导读选编；[完整目录与当前文档数](#theory-complete-index)见文末。

**阅读建议**：先看[世界模型主线](./world_model_mainline.md)，区分用于预测、评估、规划和动作生成的不同角色，再选择具体模型；真机迁移问题可接着读 [Sim-to-Real](../../deployment/sim_to_real_transfer_strategies.md)。

## 🚀 从哪里开始？

| 你是谁 | 推荐 | 为什么 |
|--------|------|--------|
| 🆕 新手 | [World Model Mainline](./world_model_mainline.md) | 从 evaluator 到 World-Action Model 演进 |
| 🛠️ 工程师 | [Lda 1B](./lda_1b_dissection.md) | DINO 隐空间 WAM scaling — 实战代表 |
| 🔬 研究者 | [Lyra2](./lyra2_dissection.md) | NVIDIA 长轨迹生成 + Isaac Sim 直连 |

## 📚 主线 / 工具文章

- [**World Model Mainline**](./world_model_mainline.md)

---

## 🗂️ 深度文章（按主题分组）

### ⭐ World-Action Model (WAM) <code>9</code>

- [Dreamzero World Action Models Zero Shot Policies 2026](./dreamzero_world_action_models_zero_shot_policies_2026.md)
- [Drivedreamer Policy A Geometry Grounded World Action Model F](./drivedreamer_policy_a_geometry_grounded_world_action_model_f_dissection.md)
- [Lda 1B](./lda_1b_dissection.md)
- [Motubrain An Advanced World Action Model For Robot Control](./motubrain_an_advanced_world_action_model_for_robot_control_dissection.md)
- [Ropedreamer A Kinematic Recurrent State Space Model For Dyna](./ropedreamer_a_kinematic_recurrent_state_space_model_for_dyna_dissection.md)
- [Wam Three Routes Video Pretraining Vs VLA 2026](./wam_three_routes_video_pretraining_vs_vla_2026.md)
- [World2Vlm Distilling World Model Imagination Into Vlms For D](./world2vlm_distilling_world_model_imagination_into_vlms_for_d_dissection.md)
- [World Action Models Are Zero Shot Policies](./world_action_models_are_zero_shot_policies_dissection.md)
- [World Value Action Model Implicit Planning For Vision Langua](./world_value_action_model_implicit_planning_for_vision_langua_dissection.md)

### ⭐ 生成式 3D / Video DiT <code>4</code>

- [Learning Physics From Pretrained Video Models A Multimodal C](./learning_physics_from_pretrained_video_models_a_multimodal_c_dissection.md)
- [Lyra2](./lyra2_dissection.md)
- [Olaf World Orienting Latent Actions For Video World Modeling](./olaf_world_orienting_latent_actions_for_video_world_modeling_dissection.md)
- [Video Generation Models In Robotics Survey 2026](./video_generation_models_in_robotics_survey_2026.md)

### Sim2Real / 仿真蒸馏 <code>6</code>

- [Egosim Egocentric World Simulator For Embodied Interaction G](./egosim_egocentric_world_simulator_for_embodied_interaction_g_dissection.md)
- [From Seeing To Simulating Generative High Fidelity Simulatio](./from_seeing_to_simulating_generative_high_fidelity_simulatio_dissection.md)
- [Geopt Lifted Geometric Pretraining Physics Simulation 2026](./geopt_lifted_geometric_pretraining_physics_simulation_2026.md)
- [Simulation Distillation Pretraining World Models In Simulati](./simulation_distillation_pretraining_world_models_in_simulati_dissection.md)
- [Teaching Robots Build Simulations Of Themselves Self Model](./teaching_robots_build_simulations_of_themselves_self_model_dissection.md)
- [Wanderland Geometrically Grounded Simulation For Open World](./wanderland_geometrically_grounded_simulation_for_open_world_dissection.md)

### 物理记忆 / 状态预测 <code>4</code>

- [Dial Decoupling Intent And Action Via Latent World Modeling](./dial_decoupling_intent_and_action_via_latent_world_modeling_dissection.md)
- [Lifting Embodied World Models For Planning And Control](./lifting_embodied_world_models_for_planning_and_control_dissection.md)
- [Physmem Scaling Test Time Physical Memory For Robot Manipula](./physmem_scaling_test_time_physical_memory_for_robot_manipula_dissection.md)
- [Rise Self Improving Robot Policy With Compositional World Mo](./rise_self_improving_robot_policy_with_compositional_world_mo_dissection.md)

### 🗃️ 其他深度文章 <code>14</code>

- [Agent World Model Infinity Synthetic Environments For Agenti](./agent_world_model_infinity_synthetic_environments_for_agenti_dissection.md)
- [Atomvla Offline Post Training Predictive Latent World Models 2026](./atomvla_offline_post_training_predictive_latent_world_models_2026.md)
- [Causal World Modeling For Robot Control](./causal_world_modeling_for_robot_control_dissection.md)
- [Ctrl World Worldarena Embodied World Model Benchmark 2026](./ctrl_world_worldarena_embodied_world_model_benchmark_2026.md)
- [Dailyart Discovering Articulation From Single Static Images](./dailyart_discovering_articulation_from_single_static_images_dissection.md)
- [Gigabrain 0 5M Star World Model Based Rl Ramp 2026](./gigabrain_0_5m_star_world_model_based_rl_ramp_2026.md)
- [Goal VLA Image Generative Vlms As Object Centric World Model](./goal_vla_image_generative_vlms_as_object_centric_world_model_dissection.md)
- [H Wm Robotic Task And Motion Planning Guided By Hierarchical](./h_wm_robotic_task_and_motion_planning_guided_by_hierarchical_dissection.md)
- [Mind Benchmarking Memory Consistency And Action Control In W](./mind_benchmarking_memory_consistency_and_action_control_in_w_dissection.md)
- [One X World Model](./one_x_world_model.md)
- [VLAw Iterative Co Improvement VLA World Model 2026](./vlaw_iterative_co_improvement_vla_world_model_2026.md)
- [Vloa Embodied World Model 3D Point Cloud Trajectory 2026](./vloa_embodied_world_model_3d_point_cloud_trajectory_2026.md)
- [World Model Aided VLA Post Training Deep Dive 2026](./world_model_aided_vla_post_training_deep_dive_2026.md)
- [Worldeval World Model Policy Evaluator 2025](./worldeval_world_model_policy_evaluator_2025.md)

---

## 🔗 跨 Realm 導航

[← Back to Theory Index](../README.md) · [↑ VLA-Handbook 主頁](../../README.md)

[🏛️](../vla-core/README.md) · [🧠](../planning/README.md) · [🎮](../rl/README.md) · [🤚](../tactile/README.md) · [👁️](../perception/README.md) · [🏗️](../foundation/README.md) · [🌊](../diffusion-flow/README.md) · [🔧](../deployment/README.md) · [🔬](../frontier/README.md)

---

<!-- VLA:THEORY-NAV:BEGIN -->
<a id="theory-complete-index"></a>
## 完整目录（自动更新）

> 上方保留人工导读与主题分组。本节列出本目录内全部 Git 已追踪 Markdown 文件，包含综述与索引，不含 README。按文件路径排序。

**共 109 份文档。**

- [AcceRL：面向 VLA 的分布式异步强化学习与世界模型框架 (AcceRL: A Distributed Asynchronous Reinforcement Learning and World Model Framework for Vision-Language-Action Models)](./accerl_a_distributed_asynchronous_reinforcement_learning_and_dissection.md)
- [AeroAct: 动作中心世界-动作模型用于语言条件四旋翼飞行 (AeroAct: Action-Centered World-Action Models for Language-Conditioned Quadrotor Flight)](./aeroact_action_centered_world_action_models_for_language_con_dissection.md)
- [Agent World Model：无限合成环境赋能智能体强化学习 (Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning)](./agent_world_model_infinity_synthetic_environments_for_agenti_dissection.md)
- [AtomVLA：世界模型驱动的离线后训练，把长程任务拆成“原子步骤” (AtomVLA: Scalable Post-Training for Robotic Manipulation via Predictive Latent World Models)](./atomvla_offline_post_training_predictive_latent_world_models_2026.md)
- [因果动作增益世界动作模型 (CAG-WAM: Causal Action-Gain World Action Models for Robot Manipulation)](./cag_wam_causal_action_gain_world_action_models_for_robot_man_dissection.md)
- [因果世界建模用于机器人控制 (Causal World Modeling for Robot Control)](./causal_world_modeling_for_robot_control_dissection.md)
- [接触前执行监控：基于动作条件潜在世界模型的 VLA 安全卫士 (ContactGuard: Pre-Contact Execution Monitoring with Action-Conditioned Latent World Models)](./contactguard_pre_contact_execution_monitoring_with_action_co_dissection.md)
- [CoWorld-VLA：多专家世界模型中的思考式推理 (CoWorld-VLA: Thinking in a Multi-Expert World Model for Autonomous Driving)](./coworld_vla_thinking_in_a_multi_expert_world_model_for_auton_dissection.md)
- [Ctrl-World × WorldArena：世界模型终于不只比“像不像”，而开始比“能不能真干活” (Ctrl-World &amp; WorldArena)](./ctrl_world_worldarena_embodied_world_model_benchmark_2026.md)
- [DailyArt: 从单张静态图像发现关节结构 (DailyArt: Discovering Articulation from Single Static Images via Latent Dynamics)](./dailyart_discovering_articulation_from_single_static_images_dissection.md)
- [DIAL：通过潜在世界建模解耦意图与动作的端到端 VLA (DIAL: Decoupling Intent and Action via Latent World Modeling for End-to-End VLA)](./dial_decoupling_intent_and_action_via_latent_world_modeling_dissection.md)
- [把交互动力学蒸馏进单步去噪：DIDO 让世界行动模型跑得更快 (DIDO: Distilling Interaction-Centric Dynamics into One-Step Denoising for World Action Models)](./dido_distilling_interaction_centric_dynamics_into_one_step_d_dissection.md)
- [世界模型真的让机器人更强吗？——预测式具身智能评测基准综述 (Do World Models Make Better Robots? A Survey of Evaluation Benchmarks for Predictive Embodied Intelligence)](./do_world_models_make_better_robots_a_survey_of_evaluation_be_dissection.md)
- [Dream.exe：视频生成模型能否"梦想"可执行的机器人操作？(Dream.exe: Can Video Generation Models Dream Executable Robot Manipulation?)](./dreamexe_can_video_generation_models_dream_executable_robot_dissection.md)
- [DreamX-Phi 1.0：动作条件视频世界模型用于机器人操作 (DreamX-Phi 1.0: Action-Conditioned Video World Model for Robotic Manipulation)](./dreamx_phi_10_action_conditioned_video_world_model_for_robot_dissection.md)
- [DreamZero：世界动作模型即零样本策略 (World Action Models are Zero-shot Policies)](./dreamzero_world_action_models_zero_shot_policies_2026.md)
- [DriftWorld：通过漂移实现快速世界模型 (DriftWorld: Fast World Modeling through Drifting)](./driftworld_fast_world_modeling_through_drifting_dissection.md)
- [DRILL：训练世界模型以改进策略，而非预测像素 (DRILL: Training World Models to Improve Policies, Not Predict Pixels)](./drill_training_world_models_to_improve_policies_not_predict_dissection.md)
- [DriveDreamer-Policy：几何落地的世界 - 动作统一模型 (DriveDreamer-Policy: A Geometry-Grounded World-Action Model for Unified Generation and Planning)](./drivedreamer_policy_a_geometry_grounded_world_action_model_f_dissection.md)
- [双流扩散世界模型增强 VLA (Dual-Stream Diffusion for World-Model Augmented VLA)](./dual_stream_diffusion_for_world_model_augmented_vision_langu_dissection.md)
- [解耦视频生成世界模型 (DVG-WM: Disentangled Video Generation Enables Efficient Embodied World Model for Robotic Manipulation)](./dvg_wm_disentangled_video_generation_enables_efficient_embod_dissection.md)
- [EgoSim: 第一人称闭环世界模拟器 (EgoSim: Egocentric World Simulator for Embodied Interaction Generation)](./egosim_egocentric_world_simulator_for_embodied_interaction_g_dissection.md)
- [推理期仿真在环精修：让真实布料操作可以「现场纠错」 (Enabling Robust Cloth Manipulation via Inference-Time Simulator-in-the-Loop Refinement)](./enabling_robust_cloth_manipulation_via_inference_time_simula_dissection.md)
- [冻结世界模型的内部预测状态里，已经写好了失败信号 (FARM: Reading Failure Signals from the Internal Predictive States of a Frozen Robotic World Model)](./farm_reading_failure_signals_from_the_internal_predictive_st_dissection.md)
- [ForgeWM：渐进因果训练实现少步动作条件视频世界模型 (ForgeWM: Progressive Causal Training for Few-Step Action-Conditioned Video World Models)](./forgewm_progressive_causal_training_for_few_step_action_cond_dissection.md)
- [从想象未来到可执行动作：混合潜在动作机器人操作 (From Imagined Futures to Executable Actions: Mixture of Latent Actions for Robot Manipulation)](./from_imagined_futures_to_executable_actions_mixture_of_laten_dissection.md)
- [从看到仿真：用数字表亲生成高保真仿真环境 (From Seeing to Simulating: Generative High-Fidelity Simulation with Digital Cousins for Generalizable Robot Learning and Evaluation)](./from_seeing_to_simulating_generative_high_fidelity_simulatio_dissection.md)
- [GaussianDream：前馈式 3D 高斯世界模型赋能机器人操作 (GaussianDream: A Feed-Forward 3D Gaussian World Model for Robotic Manipulation)](./gaussiandream_a_feed_forward_3d_gaussian_world_model_for_rob_dissection.md)
- [GeoPT：动力学提升的几何预训练，扩展物理仿真 (GeoPT: Lifted Geometric Pre-Training for Physics Simulation)](./geopt_lifted_geometric_pretraining_physics_simulation_2026.md)
- [GigaBrain-0.5M\*：世界模型原生的 VLA 自我进化范式 (GigaBrain-0.5M\*: World Model-Based RL for VLA)](./gigabrain_0_5m_star_world_model_based_rl_ramp_2026.md)
- [GlanceWAM：稀疏测试时想象用于世界-动作模型 (GlanceWAM: Sparse Test-Time Imagination for World-Action Models)](./glancewam_sparse_test_time_imagination_for_world_action_mode_dissection.md)
- [Goal-VLA：图像生成式 VLM 作为物体中心世界模型赋能零样本机器人操作 (Goal-VLA: Image-Generative VLMs as Object-Centric World Models Empowering Zero-shot Robot Manipulation)](./goal_vla_image_generative_vlms_as_object_centric_world_model_dissection.md)
- [Grounded-Dreamer：以物理錨定的世界模型合成資料訓練機器人操作策略 (Grounded-Dreamer: Robot Learning from World Model Synthetic Data with Physical Grounding)](./grounded_dreamer_robot_learning_from_world_model_synthetic_d_dissection.md)
- [紧凑世界模型中的空间关系 Grounding：指令泄漏与无目标动力学修复 (Grounding Spatial Relations in a Compact World Model: Instruction Leakage and a Goal-Free Dynamics Fix)](./grounding_spatial_relations_in_a_compact_world_model_instruc_dissection.md)
- [H-WM：分层世界模型引导的机器人任务与运动规划 (H-WM: Robotic Task and Motion Planning Guided by Hierarchical World Model)](./h_wm_robotic_task_and_motion_planning_guided_by_hierarchical_dissection.md)
- [ImageWAM：世界动作模型真的需要视频生成吗？(ImageWAM: Do World Action Models Really Need Video Generation, or Just Image Editing?)](./imagewam_do_world_action_models_really_need_video_generation_dissection.md)
- [用接地潜在动作世界模型从异构演示中学习 (Imitation from Heterogeneous Demonstrations using Grounded Latent-Action World Models)](./imitation_from_heterogeneous_demonstrations_using_grounded_l_dissection.md)
- [保留未来，丢掉推演：RIFT 让世界动作模型一步生成未来缓存 (Keep the Future, Drop the Rollout: RIFT for World Action Models)](./keep_the_future_drop_the_rollout_rift_for_world_action_model_dissection.md)
- [LaST0.5：以交互为中心隐空间推理的通用世界动作模型 (LaST0.5: A Generalist World Action Model with Interaction-Centric Latent Reasoning)](./last05_a_generalist_world_action_model_with_interaction_cent_dissection.md)
- [LaWAM：隐空间世界动作模型 (Latent World Action Models for Efficient Dynamics-Aware Robot Policies)](./lawam_latent_world_action_models_for_efficient_dynamics_awar_dissection.md)
- [LDA-1B：把 VLA 與世界模型在 DINO 隱空間裡縫合 (Latent Dynamics Action Model via Universal Embodied Data Ingestion)](./lda_1b_dissection.md)
- [从视频生成模型学习物理：多模态连续序列世界交互模型用于机器人操作 (Learning Physics from Pretrained Video Models: A Multimodal Continuous and Sequential World Interaction Models for Robotic Manipulation)](./learning_physics_from_pretrained_video_models_a_multimodal_c_dissection.md)
- [LEIA：交互式架构材料的世界模型 (LEIA: Learned Environment for Interactive Architected Materials)](./leia_learned_environment_for_interactive_architected_materia_dissection.md)
- [LIBERO-Occ：通过视点想象克服场景遮挡的 VLA 框架 (LIBERO-Occ: Evaluating and Improving Vision-Language-Action Models under Scene-Induced Occlusion via Viewpoint Imagination)](./libero_occ_evaluating_and_improving_vision_language_action_m_dissection.md)
- [提升具身世界模型用于规划与控制 (Lifting Embodied World Models for Planning and Control)](./lifting_embodied_world_models_for_planning_and_control_dissection.md)
- [LiLa-WAM：轻量级潜在推理世界-动作模型用于机器人操作 (Lightweight Latent Reasoning World-Action Model for Robotic Manipulation)](./lila_wam_lightweight_latent_reasoning_world_action_model_for_dissection.md)
- [Lyra 2.0：用「3D 緩存路由 + 自污染訓練」對抗長軌跡視訊生成的兩大 degeneration (Explorable Generative 3D Worlds)](./lyra2_dissection.md)
- [Magic-W0：把世界模型重构为「决策导向」的三段式结构（Magic-W0: A Structured World–Action Foundation Model for Physical Intelligence）](./magic_w0_a_structured_world_action_foundation_model_for_phys_dissection.md)
- [掩码世界模型：预测什么对机器人策略学习最重要 (Mask World Model: Predicting What Matters for Robust Robot Policy Learning)](./mask_world_model_predicting_what_matters_for_robust_robot_po_dissection.md)
- [掩码视觉动作：统一世界模型的像素级控制接口 (Masked Visual Actions for Unified World Modeling)](./masked_visual_actions_for_unified_world_modeling_dissection.md)
- [MemoryWAM：高效世界动作模型与持久记忆 (Efficient World Action Modeling with Persistent Memory)](./memorywam_efficient_world_action_modeling_with_persistent_me_dissection.md)
- [MIND: 世界模型记忆一致性与动作控制评估基准 (MIND: Benchmarking Memory Consistency and Action Control in World Models)](./mind_benchmarking_memory_consistency_and_action_control_in_w_dissection.md)
- [MIND-V：分层世界模型与RL物理对齐的长程机械操作 (MIND-V: Hierarchical World Model for Long-Horizon Robotic Manipulation with RL-based Physical Alignment)](./mind_v_hierarchical_world_model_for_long_horizon_robotic_man_dissection.md)
- [MotionWAM：迈向实时人形机器人世界动作模型 (MotionWAM: Towards Foundation World Action Models for Real-Time Humanoid Loco-Manipulation)](./motionwam_towards_foundation_world_action_models_for_real_ti_dissection.md)
- [MotuBrain：面向机器人控制的世界-动作统一生成模型 (MotuBrain: An Advanced World Action Model for Robot Control)](./motubrain_an_advanced_world_action_model_for_robot_control_dissection.md)
- [μ0：可扩展的 3D 交互痕迹世界模型（$\\mu\_0$: A Scalable 3D Interaction-Trace World Model）](./mu0_a_scalable_3d_interaction_trace_world_model_dissection.md)
- [Olaf-World：面向视频世界建模的潜在动作定向 (Olaf-World: Orienting Latent Actions for Video World Modeling)](./olaf_world_orienting_latent_actions_for_video_world_modeling_dissection.md)
- [1X World Model（1XWM）：把“视频世界模型”用在人形 NEO 上](./one_x_world_model.md)
- [基于本体 grounding 的世界模型：物理 AI 系统的故障诊断与闭环修复 (Ontology-Grounded World Models for Failure Diagnosis and Closed-Loop Repair in Physical AI Systems)](./ontology_grounded_world_models_for_failure_diagnosis_and_clo_dissection.md)
- [克服动作可控世界模型中的统计偏差 (Overcoming Statistical Bias in Action-Controllable World Models)](./overcoming_statistical_bias_in_action_controllable_world_mod_dissection.md)
- [PhyLatent：为 JEPA 世界模型学习动力学相关表示 (PhyLatent: Learning Dynamics-Relevant Representations for JEPA World Models)](./phylatent_learning_dynamics_relevant_representations_for_jep_dissection.md)
- [PhysMem: 测试时物理记忆扩展 (Scaling Test-time Physical Memory for Robot Manipulation)](./physmem_scaling_test_time_physical_memory_for_robot_manipula_dissection.md)
- [PLaW-VLA：面向视觉-语言-动作策略的预测性潜在世界建模 (Predictive Latent World Modeling for Vision-Language-Action Policies)](./plaw_vla_predictive_latent_world_modeling_for_vision_languag_dissection.md)
- [Q-WAM：面向世界动作模型的 4-bit 量化与动作子空间保护 (Q-WAM: 4-Bit Quantization of World Action Models with Action-Subspace Protection)](./q_wam_4_bit_quantization_of_world_action_models_with_action_dissection.md)
- [QuantWAMs：在世界动作模型上以正确粒度校准 (QuantWAMs: Calibrating at the Right Granularity for World Action Models)](./quantwams_calibrating_at_the_right_granularity_for_world_act_dissection.md)
- [记住该记住的：长程机器人世界模型的上下文调度 (Remembering What Matters for Long-Horizon Robotic World Models)](./remembering_what_matters_for_long_horizon_robotic_world_mode_dissection.md)
- [在表征空间里直接规划：Representation World Model (Learning States, Transition and Executable Plans in Representation)](./representation_world_model_learning_states_transition_and_ex_dissection.md)
- [Riemann-1.0：面向物理 AI 的具身世界动作模型 (Riemann-1.0: An Embodied World Action Model for Physical AI)](./riemann_10_an_embodied_world_action_model_for_physical_ai_dissection.md)
- [RISE: 用组合世界模型实现 VLA 策略自改进 (Self-Improving Robot Policy with Compositional World Model)](./rise_self_improving_robot_policy_with_compositional_world_mo_dissection.md)
- [机器人因子化解耦世界模型：通过机器人渲染实现动作条件化 (Robot-Factored World Models via Robot Rendering)](./robot_factored_world_models_via_robot_rendering_dissection.md)
- [滚动世界-动作模型：把联合去噪摊到重规划周期 (Rolling-WAM: World Action Models with Rolling Imagination)](./rolling_wam_world_action_models_with_rolling_imagination_dissection.md)
- [RopeDreamer：基于运动学递归状态空间模型的柔性线性物体动力学预测 (RopeDreamer: A Kinematic Recurrent State Space Model for Dynamics of Flexible Deformable Linear Objects)](./ropedreamer_a_kinematic_recurrent_state_space_model_for_dyna_dissection.md)
- [Simulation Distillation: Pretraining World Models in Simulation for Rapid Real-World Adaptation](./simulation_distillation_pretraining_world_models_in_simulati_dissection.md)
- [SkeleWAM：骨架世界-動作建模 — 用稀疏 3D 骨架做高效機器人操作 (Skeleton World-Action Modeling for Efficient Robotic Manipulation)](./skelewam_skeleton_world_action_modeling_for_efficient_roboti_dissection.md)
- [SmolWAM：面向边缘设备的隐空间世界-动作模型（SmolWAM: A Latent World-Action Model for Robotic Manipulation on Edge Devices）](./smolwam_a_latent_world_action_model_for_robotic_manipulation_dissection.md)
- [SoMA：面向软体机器人操作的 Real-to-Sim 神经模拟器 (SoMA: A Real-to-Sim Neural Simulator for Robotic Soft-body Manipulation)](./soma_a_real_to_sim_neural_simulator_for_robotic_soft_body_ma_dissection.md)
- [StageWAM：联合嵌入阶段预测的世界-动作模型 (StageWAM: Joint-Embedding Stage Prediction for World-Action Models in Robot Manipulation)](./stagewam_joint_embedding_stage_prediction_for_world_action_m_dissection.md)
- [SWEET：用图像编辑做稀疏世界模型 (SWEET: Sparse World Modeling with Image Editing for Embodied Task Execution)](./sweet_sparse_world_modeling_with_image_editing_for_embodied_dissection.md)
- [Sword: 风格鲁棒的世界模型模拟器 (Sword: Style-Robust World Models as Simulators via Dynamic Latent Bootstrapping for VLA Policy Post-Training)](./sword_style_robust_world_models_as_simulators_via_dynamic_la_dissection.md)
- [机器人"照镜子"学会模拟自己：从单目视频到运动学自模型](./teaching_robots_build_simulations_of_themselves_self_model_dissection.md)
- [潛在世界模型的規劃極限 (The Planning Limits of Latent World Models)](./the_planning_limits_of_latent_world_models_dissection.md)
- [迈向统一机器人学习：桥接表示、VLA 与世界模型 (Toward Unified Robot Learning: Bridging Representation, Vision-Language-Action, and World Models)](./toward_unified_robot_learning_bridging_representation_vision_dissection.md)
- [TS-WAM：时间潜在动作与世界-动作耦合的视觉语言机器人控制 (TS-WAM: Temporal Latent Actions and World-Action Coupling for Vision-Language Robot Control)](./ts_wam_temporal_latent_actions_and_world_action_coupling_for_dissection.md)
- [UniWALM：把语言变成世界-动作模型的工作记忆 (UniWALM: Unified World Action Language Model for Embodied Robot Control)](./uniwalm_unified_world_action_language_model_for_embodied_rob_dissection.md)
- [统一世界-动作模型：把「懂物理」和「会推理」焊进同一条 Transformer (UniWAM: Unified World-Action Model)](./uniwam_unified_world_action_model_dissection.md)
- [Vector-Quantized World Model：用控制器可达性做离散 latent 规划的机器人操作框架 (Vector-Quantized World Model: Latent Feedback Motion Planning for Robot Manipulation)](./vector_quantized_world_model_latent_feedback_motion_planning_dissection.md)
- [VeriVLA：基于潜世界模型的自适应测试时验证，用于流式 VLA (VeriVLA: Adaptive Test-Time Verification for Flow-Based VLAs with Latent World Models)](./verivla_adaptive_test_time_verification_for_flow_based_vlas_dissection.md)
- [视频生成模型在机器人中的应用：近 300 篇体系化综述提炼（2026）(Video Generation Models in Robotics, 2026 Survey)](./video_generation_models_in_robotics_survey_2026.md)
- [视觉-语言-动作还是世界-动作？其实只需动作模型 (Vision-Language-Action or World-Action? Just Action Models)](./vision_language_action_or_world_action_just_action_models_dissection.md)
- [VLAW：世界模型 × VLA 协同进化 (Iterative Co-Improvement of VLA Policy and World Model)](./vlaw_iterative_co_improvement_vla_world_model_2026.md)
- [VLOA 具身世界模型：用 3D 点云轨迹打开物理认知的黑箱](./vloa_embodied_world_model_3d_point_cloud_trajectory_2026.md)
- [VLA 将死，WAM 当立？2026 三路线综述 (Will WAM Replace VLA? A 2026 Three-Route Overview)](./wam_three_routes_video_pretraining_vs_vla_2026.md)
- [用人类玩耍视频在测试时转向世界-动作模型（WAM-TTT: Steering World-Action Models by Watching Human Play at Test Time）](./wam_ttt_steering_world_action_models_by_watching_human_play_dissection.md)
- [Wanderland：开放世界具身 AI 的几何 grounding 仿真平台 (Wanderland: Geometrically Grounded Simulation for Open-World Embodied AI)](./wanderland_geometrically_grounded_simulation_for_open_world_dissection.md)
- [Wh0: 用生成式世界模型合成第一人称手部操作数据 (Wh0: Generative World Models as Scalable Sources of Egocentric Human Hand Manipulation Data)](./wh0_generative_world_models_as_scalable_sources_of_egocentri_dissection.md)
- [当仿真器而非策略失败：审计仿真机器人基准中的接触伪影 (When the Simulator Fails, Not the Policy: Auditing Contact Artifacts in Simulated Robot Benchmarks)](./when_the_simulator_fails_not_the_policy_auditing_contact_art_dissection.md)
- [WISE：世界模型引导的想象调度，用于高效 VLA 后训练 (World-model-guided Imagination Scheduling for Efficient Post-training of Vision-Language-Action Models)](./wise_world_model_guided_imagination_scheduling_for_efficient_dissection.md)
- [将世界模型想象力蒸馏进 VLM：面向动态空间推理的训练框架 (World2VLM: Distilling World Model Imagination into VLMs for Dynamic Spatial Reasoning)](./world2vlm_distilling_world_model_imagination_into_vlms_for_d_dissection.md)
- [世界动作智能体：让 VLM 在「世界动作预演」中直接驾驶机器人 (World Action Agent: Harnessing VLMs for Robot Manipulation via World Action Rehearsal)](./world_action_agent_harnessing_vlms_for_robot_manipulation_vi_dissection.md)
- [World Action Models are Zero-shot Policies (DreamZero)](./world_action_models_are_zero_shot_policies_dissection.md)
- [World Model 辅助 VLA 后训练：研究进展与问题拆解（2026）](./world_model_aided_vla_post_training_deep_dive_2026.md)
- [World Model 主线总纲：从 evaluator 到 planner，再到 world action model](./world_model_mainline.md)
- [世界-价值-动作模型：VLA 系统的隐式规划 (World-Value-Action Model: Implicit Planning for Vision-Language-Action Systems)](./world_value_action_model_implicit_planning_for_vision_langua_dissection.md)
- [World-VLA-Loop：视频世界模型与 VLA 策略的闭环联合学习 (Closed-Loop Learning of Video World Model and VLA Policy)](./world_vla_loop_closed_loop_learning_of_video_world_model_and_dissection.md)
- [WorldCycle：自验证强化学习用于长视界视频世界模型 (WorldCycle: Self-Verifiable Reinforcement Learning for Long-Horizon Video World Models)](./worldcycle_self_verifiable_reinforcement_learning_for_long_h_dissection.md)
- [世界模型评估机器人策略：WorldEval (World Model as Real-World Robot Policies Evaluator)](./worldeval_world_model_policy_evaluator_2025.md)
- [WorldKV：通过检索与压缩实现高效世界记忆 (Efficient World Memory with World Retrieval and Compression)](./worldkv_efficient_world_memory_with_world_retrieval_and_comp_dissection.md)
- [把世界模型当“可靠但不完美”的 RL 仿真器：WoVR 论文拆解 (WoVR: World Models as Reliable Simulators for Post-Training VLA Policies with RL)](./wovr_world_models_as_reliable_simulators_for_post_training_v_dissection.md)
- [X-Foresight：通过预测世界模型实现视觉-动作联合因果预测网络 (X-Foresight: A Joint Vision-Action Causal Forecasting Network via Predictive World Modeling)](./x_foresight_a_joint_vision_action_causal_forecasting_network_dissection.md)

<!-- VLA:THEORY-NAV:END -->
