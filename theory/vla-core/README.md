# 🏛️ Central Keep — VLA 核心架构

> 所有 VLA 模型的「解剖室」。VLA 的核心思想很简单：拿一个 VLM，给它装上「手」——一个动作生成头。但魔鬼在细节。

📚 下方为导读选编；[完整目录与当前文档数](#theory-complete-index)见文末。

**阅读建议**：先读 [VLA 架构](./vla_arch.md)，再比较[动作表示](../diffusion-flow/action_representations.md)；选具体模型时，回到[开源 VLA 指南](./open_source_vla_guide.md)核对代码、权重与训练条件。

## 🚀 从哪里开始？

| 你是谁 | 推荐 | 为什么 |
|--------|------|--------|
| 🆕 新手 | [VLA Arch](./vla_arch.md) | 一篇看完所有 VLA 的共同设计模式 |
| 🛠️ 工程师 | [Open Source VLA Guide](./open_source_vla_guide.md) | 10 个真开源 vs 假开源拆解 + 按场景选型 |
| 🔬 研究者 | [VLA Research Mainline](./vla_research_mainline.md) | VLA 研究主线 — 该赌哪条路 |

## 📚 主线 / 工具文章

- [**Open Source VLA Guide**](./open_source_vla_guide.md)
- [**VLA Research Mainline**](./vla_research_mainline.md)

---

## 🗂️ 深度文章（按主题分组）

### ⭐ π / Physical Intelligence 系列 <code>5</code>

- [Physical Intelligence Sergey Levine Foundation Model Vision 2026](./physical_intelligence_sergey_levine_foundation_model_vision_2026.md)
- [Π0 5](./pi0_5_dissection.md)
- [Π0 6](./pi0_6_dissection.md)
- [Π0 7 Steerable Compositional Generalization 2026](./pi0_7_steerable_compositional_generalization_2026.md)
- [Π0 Code Analysis](./pi0_code_analysis.md)

### ⭐ NVIDIA Gr00T / Google RT 系列 <code>2</code>

- [Gr00T N1 6](./gr00t_n1_6.md)
- [Groot N1 7 Nvidia Open Foundation Model 2026](./groot_n1_7_nvidia_open_foundation_model_2026.md)

### ⭐ 中国团队 VLA <code>2</code>

- [Unifolm VLA 0 Unitree 2026](./unifolm_vla_0_unitree_2026.md)
- [Xiaomi Robotics 0](./xiaomi_robotics_0_dissection.md)

### ACT / Diffusion Policy <code>2</code>

- [Act](./act.md)
- [Unilact Depth Aware Rgb Latent Action Learning For Vision La](./unilact_depth_aware_rgb_latent_action_learning_for_vision_la_dissection.md)

### 动作表示 / Tokenizer <code>2</code>

- [Abot M0 Action Manifold Learning VLA Foundation 2026](./abot_m0_action_manifold_learning_vla_foundation_2026.md)
- [From Action Labels To Sets Rethinking Action Supervision For](./from_action_labels_to_sets_rethinking_action_supervision_for_dissection.md)

### VLA 注意力 / 架构创新 <code>2</code>

- [2D Or 3D Who Governs Salience In VLA Models Tri Stage Token](./2d_or_3d_who_governs_salience_in_vla_models_tri_stage_token_dissection.md)
- [Beyond Attention Magnitude Leveraging Inter Layer Rank Consi](./beyond_attention_magnitude_leveraging_inter_layer_rank_consi_dissection.md)

### VLA 视角 / 多模态 <code>3</code>

- [Clutter Robust Vision Language Action Models Through Object](./clutter_robust_vision_language_action_models_through_object_dissection.md)
- [From Noise To Intent Anchoring Generative VLA Policies With](./from_noise_to_intent_anchoring_generative_vla_policies_with_dissection.md)
- [Gazevla Learning Human Intention For Robotic Manipulation](./gazevla_learning_human_intention_for_robotic_manipulation_dissection.md)

### 综述 / 通用架构 <code>1</code>

- [VLA Arch](./vla_arch.md)

### 🗃️ 其他深度文章 <code>29</code>

- [Fast](./fast.md)
- [Figure Helix 02 Full Body Autonomy 2026](./figure_helix_02_full_body_autonomy_2026.md)
- [Focusvla Focused Visual Utilization For Vision Language Acti](./focusvla_focused_visual_utilization_for_vision_language_acti_dissection.md)
- [Galaxea G0](./galaxea_g0.md)
- [Genie Sim Panorecon Fast Immersive Scene Generation From Sin](./genie_sim_panorecon_fast_immersive_scene_generation_from_sin_dissection.md)
- [Geopredict Leveraging Predictive Kinematics And 3D Gaussian](./geopredict_leveraging_predictive_kinematics_and_3d_gaussian_dissection.md)
- [Hamlet Switch Your Vision Language Action Model Into A Histo](./hamlet_switch_your_vision_language_action_model_into_a_histo_dissection.md)
- [Hif VLA Hindsight Insight And Foresight Through Motion Repre](./hif_vla_hindsight_insight_and_foresight_through_motion_repre_dissection.md)
- [Instructvla Vision Language Action Instruction Tuning From U](./instructvla_vision_language_action_instruction_tuning_from_u_dissection.md)
- [Langgap Diagnosing And Closing The Language Gap In Vision La](./langgap_diagnosing_and_closing_the_language_gap_in_vision_la_dissection.md)
- [Learning Structured Robot Policies From Vision Language Mode](./learning_structured_robot_policies_from_vision_language_mode_dissection.md)
- [Lingbot VLA Pragmatic Foundation Model 2026](./lingbot_vla_pragmatic_foundation_model_2026.md)
- [Lingbot VLA Pragmatic VLA Foundation Model 2026](./lingbot_vla_pragmatic_vla_foundation_model_2026.md)
- [Mash Spread Slice Learning To Manipulate Object States Via V](./mash_spread_slice_learning_to_manipulate_object_states_via_v_dissection.md)
- [Onetwovla](./onetwovla.md)
- [Rdt](./rdt.md)
- [Reconvla Implicit Grounding By Reconstruction](./reconvla_implicit_grounding_by_reconstruction.md)
- [Reflection Based Task Adaptation For Self Improving VLA](./reflection_based_task_adaptation_for_self_improving_vla_dissection.md)
- [Roboneuron A Middle Layer Infrastructure For Agent Driven Or](./roboneuron_a_middle_layer_infrastructure_for_agent_driven_or_dissection.md)
- [Rynnbrain Open Embodied Foundation Models 2026](./rynnbrain_open_embodied_foundation_models_2026.md)
- [Simvla Simple VLA Baseline Robotic Manipulation 2026](./simvla_simple_vla_baseline_robotic_manipulation_2026.md)
- [Small VLA Models](./small_vla_models.md)
- [Spirit V1 5](./spirit_v1_5_dissection.md)
- [Starvla Lego Like VLA Codebase 2026](./starvla_lego_like_vla_codebase_2026.md)
- [Twinvla Data Efficient Bimanual Manipulation With Twin Singl](./twinvla_data_efficient_bimanual_manipulation_with_twin_singl_dissection.md)
- [Uav Track VLA Embodied Aerial Tracking Via Vision Language A](./uav_track_vla_embodied_aerial_tracking_via_vision_language_a_dissection.md)
- [Vitra Scalable VLA Pretraining Human Activity Videos 2026](./vitra_scalable_vla_pretraining_human_activity_videos_2026.md)
- [Vln Dualvln](./vln_dualvln.md)
- [Wall Oss](./wall_oss.md)

---

## 🔗 跨 Realm 導航

[← Back to Theory Index](../README.md) · [↑ VLA-Handbook 主頁](../../README.md)

[🌍](../world-model/README.md) · [🧠](../planning/README.md) · [🎮](../rl/README.md) · [🤚](../tactile/README.md) · [👁️](../perception/README.md) · [🏗️](../foundation/README.md) · [🌊](../diffusion-flow/README.md) · [🔧](../deployment/README.md) · [🔬](../frontier/README.md)

---

<!-- VLA:THEORY-NAV:BEGIN -->
<a id="theory-complete-index"></a>
## 完整目录（自动更新）

> 上方保留人工导读与主题分组。本节列出本目录内全部 Git 已追踪 Markdown 文件，包含综述与索引，不含 README。按文件路径排序。

**共 190 份文档。**

- [2D 还是 3D：谁主导 VLA 模型中的显著性？—— 三阶段 Token 剪枝框架与模态显著性感知 (2D or 3D: Who Governs Salience in VLA Models? -- Tri-Stage Token Pruning Framework with Modality Salience Awareness)](./2d_or_3d_who_governs_salience_in_vla_models_tri_stage_token_dissection.md)
- [3DThinkVLA：通过3D思维引导协同训练赋予VLA模型隐式3D空间推理能力 (3DThinkVLA: Endowing Vision-Language-Action Models with Latent 3D Priors via 3D-Thinking-Guided Co-training)](./3dthinkvla_endowing_vision_language_action_models_with_laten_dissection.md)
- [ABot-M0：动作流形学习的 VLA 基础模型 (ABot-M0: VLA Foundation Model with Action Manifold Learning)](./abot_m0_action_manifold_learning_vla_foundation_2026.md)
- [ACT: 动作分块变换器 (Action Chunking with Transformers)](./act.md)
- [ActionGround：冻结 VLA 策略的免训练运行时精炼 (ActionGround: Training-Free Runtime Refinement of Frozen VLA Policies)](./actionground_training_free_runtime_refinement_of_frozen_vla_dissection.md)
- [ActionPiece：重新思考自回归 VLA 的动作分词 (ActionPiece: Rethinking Action Tokenization for Autoregressive Vision-Language-Action Models)](./actionpiece_rethinking_action_tokenization_for_autoregressiv_dissection.md)
- [异步 VLA 如何让无人机在推理延迟中继续飞：AeroDASH 精读 (AeroDASH: Asynchronous Vision-Language-Action Control for Continuous UAV Flight)](./aerodash_asynchronous_vision_language_action_control_for_con_dissection.md)
- [ALAM: 代数一致潜在动作模型 (Algebraically Consistent Latent Action Model for Vision-Language-Action Models)](./alam_algebraically_consistent_latent_action_model_for_vision_dissection.md)
- [闭环崩塌解剖学：一次压缩 VLA 策略的因果个案研究 (Anatomy of a Closed-Loop Collapse: A Causal Case Study of a Compressed VLA Policy)](./anatomy_of_a_closed_loop_collapse_a_causal_case_study_of_a_c_dissection.md)
- [AnySlot：零样本槽位级放置的目标条件 VLA (Goal-Conditioned Vision-Language-Action Policies for Zero-Shot Slot-Level Placement)](./anyslot_goal_conditioned_vision_language_action_policies_for_dissection.md)
- [AFCE：为 VLA 学习“未来操作目标”的效果监督 (Beyond Actions: Learning Future Operation Targets for Vision-Language-Action Models)](./beyond_actions_learning_future_operation_targets_for_vision_dissection.md)
- [超越注意力幅度：利用层间秩一致性实现高效 VLA 模型 (Beyond Attention Magnitude: Leveraging Inter-layer Rank Consistency for Efficient Vision-Language-Action Models)](./beyond_attention_magnitude_leveraging_inter_layer_rank_consi_dissection.md)
- [重建误差不等于控制性能：自回归 VLA 的动作 Tokenization 该看什么 (Beyond Reconstruction Error: Analytical and Data-Driven Action Tokenization for Autoregressive Vision-Language-Action Models)](./beyond_reconstruction_error_analytical_and_data_driven_actio_dissection.md)
- [超越 Token 重要性：为高效 VLA 推理保留空间骨架 (Beyond Token Importance: Preserving Spatial Scaffolds for Efficient Vision-Language-Action Inference)](./beyond_token_importance_preserving_spatial_scaffolds_for_eff_dissection.md)
- [位翻轉攻擊 VLA：動作解碼架構決定脆弱性 (Bit-Flip Attacks on Vision-Language-Action Models: Action-Decoding Architecture Shapes the Vulnerability)](./bit_flip_attacks_on_vision_language_action_models_action_dec_dissection.md)
- [弥合语义-动作鸿沟：面向高效 VLA 推理的视觉 Token 剪枝 (Bridging the Semantic-Action Gap in Visual Token Pruning for Efficient VLA Inference)](./bridging_the_semantic_action_gap_in_visual_token_pruning_for_dissection.md)
- [拿我的杯子！用视觉注意力提示个性化 VLA 模型 (Bring My Cup! Personalizing Vision-Language-Action Models with Visual Attentive Prompting)](./bring_my_cup_personalizing_vision_language_action_models_wit_dissection.md)
- [显式物理可行性能否提升 VLA 学习？一项实证研究 (Can Explicit Physical Feasibility Benefit VLA Learning? An Empirical Study)](./can_explicit_physical_feasibility_benefit_vla_learning_an_em_dissection.md)
- [VLA 持续学习：真实世界数据下的抗遗忘实证研究 (Can Vision-Language-Action Models Learn from Real-World Data Continually without Forgetting?)](./can_vision_language_action_models_learn_from_real_world_data_dissection.md)
- [能力与鲁棒性不可兼得：VLA 模型的信息论边界 (Capability and Robustness Cannot Both Be Free: An Information-Theoretic Bound for Vision-Language-Action Models)](./capability_and_robustness_cannot_both_be_free_an_information_dissection.md)
- [CertVLA：面向 VLA 的物理视觉攻击可认证防御 (CertVLA: Certified Defense against Physical Visual Attacks for Vision-Language-Action Models)](./certvla_certified_defense_against_physical_visual_attacks_fo_dissection.md)
- [ChainVLA：通过统一执行状态链式连接 VLA 查询以实现长视界操作 (ChainVLA: Chaining Vision-Language-Action Queries through a Unified Execution State for Long-Horizon Manipulation)](./chainvla_chaining_vision_language_action_queries_through_a_u_dissection.md)
- [CLARE: VLA 持续学习通过适配器路由与扩展 (Continual Learning for Vision-Language-Action Models via Autonomous Adapter Routing and Expansion)](./clare_continual_learning_for_vision_language_action_models_v_dissection.md)
- [把末端执行器「隐身」：Cloak 用几何遮挡实现零样本跨本体操作 (Cloak: Zero-Shot Cross-Embodiment Manipulation by Masking the End-Effector from the VLA)](./cloak_zero_shot_cross_embodiment_manipulation_by_masking_the_dissection.md)
- [关闭人形 VLA 的闭环：持久化 3D 对象令牌用于可验证的移动操作 (Closing the Loop in Humanoid VLA: Persistent 3D Object Tokens for Verifiable Loco-Manipulation)](./closing_the_loop_in_humanoid_vla_persistent_3d_object_tokens_dissection.md)
- [通过对象中心与几何接地提升杂乱环境下的 VLA 鲁棒性 (Clutter-Robust Vision-Language-Action Models through Object-Centric and Geometry Grounding)](./clutter_robust_vision_language_action_models_through_object_dissection.md)
- [CoFreeVLA：短时域无碰撞双臂操作通过VLA与风险估计 (CoFreeVLA: Short-Horizon Collision-Free Dual-Arm Manipulation via Vision-Language-Action Model and Risk Estimation)](./cofreevla_short_horizon_collision_free_dual_arm_manipulation_dissection.md)
- [CogVLA：认知对齐的视觉-语言-动作模型 (CogVLA: Cognition-Aligned Vision-Language-Action Model via Instruction-Driven Routing &amp; Sparsification)](./cogvla_cognition_aligned_vision_language_action_model_via_in_dissection.md)
- [CommitFlow：长程 VLA 执行的语义承诺验证与局部纠偏 (CommitFlow: Semantic Commitment Verification and Local Correction for Long-Horizon Robot Manipulation VLA Execution)](./commitflow_semantic_commitment_verification_and_local_correc_dissection.md)
- [持续演化 VLA 技能知识 (Continually Evolving Skill Knowledge in Vision Language Action Model)](./continually_evolving_skill_knowledge_in_vision_language_acti_dissection.md)
- [用 EMG 与视觉标注做连续任务条件化：VLA 可以不止靠语言 (Continuous Conditioning of VLAs with Augmenting EMG and Visual Task Descriptors)](./continuous_conditioning_of_vlas_with_augmenting_emg_and_visu_dissection.md)
- [从人类视频到机器人操作的跨本体因果迁移 (Cross-Embodiment Causal Transfer from Human Videos to Robot Manipulation)](./cross_embodiment_causal_transfer_from_human_videos_to_robot_dissection.md)
- [CycleVLA：通过子任务回溯与最小贝叶斯风险解码实现 VLA 主动自我修正 (CycleVLA: Proactive Self-Correcting Vision-Language-Action Models via Subtask Backtracking and Minimum Bayes Risk Decoding)](./cyclevla_proactive_self_correcting_vision_language_action_mo_dissection.md)
- [从 VLA 表征中解码任务进度 (Decoding Task Progress from VLA Representations)](./decoding_task_progress_from_vla_representations_dissection.md)
- [解耦视觉、语言与动作：高效多任务机器人策略 (Decoupling Vision, Language, and Action for Efficient Multi-Task Robot Policies)](./decoupling_vision_language_and_action_for_efficient_multi_ta_dissection.md)
- [Deltoris：通过比特级稀疏与推测解码实现 VLA 实时推理 (Deltoris: Enabling Real-time VLA Inference in Embodied AI via Bit-level Sparsity and Speculative Inference)](./deltoris_enabling_real_time_vla_inference_in_embodied_ai_via_dissection.md)
- [检测并抑制：面向 VLA 对抗补丁的机制级防御 (Detect and Suppress: A Mechanistic Defense against Adversarial Patches in VLA Models)](./detect_and_suppress_a_mechanistic_defense_against_adversaria_dissection.md)
- [诊断 Agent 编排 VLA 技能组合中的语义交接失败 (Diagnosing Semantic Handoff Failures in Agent-Orchestrated Vision-Language-Action Skill Composition)](./diagnosing_semantic_handoff_failures_in_agent_orchestrated_v_dissection.md)
- [动作表示的「方向–尺度解耦」（Direction-Scale Decomposition, DSD）：重新审视离散 token VLA 该 tokenize 什么](./direction_scale_decomposition_in_action_representation_rethi_dissection.md)
- [分而记之：面向长时序 VLA 策略的递归动作相关记忆 (Divide-and-Remember: Recursive Action-Relevant Memory for Long-Horizon VLA Policies)](./divide_and_remember_recursive_action_relevant_memory_for_lon_dissection.md)
- [VLA 连常识都不知道了？测量视觉-语言-动作模型中的常识与世界知识保留 (Does VLA Even Know the Basics? Measuring Commonsense and World Knowledge Retention in Vision-Language-Action Models)](./does_vla_even_know_the_basics_measuring_commonsense_and_worl_dissection.md)
- [双流记忆：把 VLA 的“执行状态”和“视觉证据”拆开 (Dual-Stream Memory for Vision-Language-Action Policies)](./dual_stream_memory_for_vision_language_action_policies_dissection.md)
- [VLA 动态执行承诺：A3 自适应动作接受机制 (Dynamic Execution Commitment of Vision-Language-Action Models)](./dynamic_execution_commitment_of_vision_language_action_model_dissection.md)
- [动态混合渐进式参数高效专家库用于终身机器人学习 (Dynamic Mixture of Progressive Parameter-Efficient Expert Library for Lifelong Robot Learning)](./dynamic_mixture_of_progressive_parameter_efficient_expert_li_dissection.md)
- [Echo：VLA 动作与任务状态信号的耦合去噪 (Echo: Coupled Action and Task Status Signal Denoising for Vision Language Action Models)](./echo_coupled_action_and_task_status_signal_denoising_for_vis_dissection.md)
- [动静解耦高效长视界 VLA (Static-Dynamic Disentanglement for Efficient Multi-Frame Vision-Language-Action Models)](./efficient_long_horizon_vision_language_action_models_via_sta_dissection.md)
- [具身可解释性：因果理解驱动 VLA 泛化 (Embodied Interpretability: Linking Causal Understanding to Generalization in Vision-Language-Action Models)](./embodied_interpretability_linking_causal_understanding_to_ge_dissection.md)
- [具身中间训练：用数据邻近度弥合 VLM 与 VLA 之间的鸿沟 (EmbodiedMidtrain: Bridging the Gap between Vision-Language Models and Vision-Language-Action Models via Mid-training)](./embodiedmidtrain_bridging_the_gap_between_vision_language_mo_dissection.md)
- [Evo-World：用光流监督未来运动预测的轻量 VLA (Evo-World: A Lightweight Vision-Language-Action Model with Future Motion Prediction via Optical-Flow Supervision)](./evo_world_a_lightweight_vision_language_action_model_with_fu_dissection.md)
- [FAST: 高效动作 Token 化](./fast.md)
- [只思考一次的 VLA：Fast-VLA 的任务记忆与闭环解耦 (Fast-VLA: You Only Think Once for Robot Manipulation)](./fast_vla_you_only_think_once_for_robot_manipulation_dissection.md)
- [FastHarness：用经验调度 GPT-6 复核，降低 VLA 执行监控成本](./fastharness_experience_driven_review_scheduling_for_vision_l_dissection.md)
- [FastVLA：任务条件化与时间自适应的 VLA 视觉 Token 压缩（FastVLA: Task-Conditioned and Temporally Adaptive Visual Token Compression for Efficient Vision-Language-Action Models）](./fastvla_task_conditioned_and_temporally_adaptive_visual_toke_dissection.md)
- [Figure Helix 02：全身端到端 VLA 的“运动-操作一体化”架构 (Helix 02: Full-Body Autonomy)](./figure_helix_02_full_body_autonomy_2026.md)
- [单鱼眼解耦覆盖度与锐度：Fisheye-VLA (Fisheye-VLA: Decoupling Coverage and Acuity for Manipulation with a Single Fisheye Camera)](./fisheye_vla_decoupling_coverage_and_acuity_for_manipulation_dissection.md)
- [FlashVLA：流式动作解码实现高速异步 VLA 推理 (Streaming Action Decoding for Fast and Asynchronous VLA Inference)](./flashvla_streaming_action_decoding_for_fast_and_asynchronous_dissection.md)
- [FluxVLA Engine：面向具身智能的一站式 VLA 工程平台 (FluxVLA Engine: A One-Stop VLA Engineering Platform for Embodied Intelligence)](./fluxvla_engine_a_one_stop_vla_engineering_platform_for_embod_dissection.md)
- [FocusVLA：聚焦视觉利用的 Vision-Language-Action 模型 (FocusVLA: Focused Visual Utilization for Vision-Language-Action Models)](./focusvla_focused_visual_utilization_for_vision_language_acti_dissection.md)
- [FocusVLA：未来意图引导的多视角动态路由 (FocusVLA: Future-Intent Guided Dynamic View Routing for Vision-Language-Action Models)](./focusvla_future_intent_guided_dynamic_view_routing_for_visio_dissection.md)
- [从动作标签到动作集合：重新思考纠正反馈下的模仿学习动作监督 (From Action Labels to Sets: Rethinking Action Supervision for Imitation Learning from Corrective Feedback)](./from_action_labels_to_sets_rethinking_action_supervision_for_dissection.md)
- [从噪声到意图：用残差桥接锚定生成式 VLA 策略 (From Noise to Intent: Anchoring Generative VLA Policies with Residual Bridges)](./from_noise_to_intent_anchoring_generative_vla_policies_with_dissection.md)
- [Galaxea G0: 双系统 VLA 框架](./galaxea_g0.md)
- [GazeVLA：用人类注视学习操作意图 (Learning Human Intention for Robotic Manipulation)](./gazevla_learning_human_intention_for_robotic_manipulation_dissection.md)
- [Genie Sim PanoRecon：从单张全景图快速生成沉浸式 3D 场景 (Genie Sim PanoRecon: Fast Immersive Scene Generation from Single-View Panorama)](./genie_sim_panorecon_fast_immersive_scene_generation_from_sin_dissection.md)
- [用去噪轨迹的几何自适应切分动作块 (GeoAAC: Geometry-Based Adaptive Action Chunking from Denoising Trajectories in VLA Policies)](./geoaac_geometry_based_adaptive_action_chunking_from_denoisin_dissection.md)
- [GeoPredict：利用预测运动学与 3D 高斯几何实现精确 VLA 操作 (GeoPredict: Leveraging Predictive Kinematics and 3D Gaussian Geometry for Precise VLA Manipulation)](./geopredict_leveraging_predictive_kinematics_and_3d_gaussian_dissection.md)
- [GR00T-N1.6 手册篇 / GR00T-N1.6 Handbook Entry](./gr00t_n1_6.md)
- [GR00T-N1.7：NVIDIA 的开源通用机器人基础模型——从人形到任意形态](./groot_n1_7_nvidia_open_foundation_model_2026.md)
- [GS-VLA：即插即用视点规范化 (GS-VLA: Plug-and-Play Viewpoint Canonicalization for Frozen VLA Policies via Gaussian Splatting)](./gs_vla_plug_and_play_viewpoint_canonicalization_for_frozen_v_dissection.md)
- [HAMLET：将视觉 - 语言 - 动作模型转换为历史感知策略 (HAMLET: Switch your Vision-Language-Action Model into a History-Aware Policy)](./hamlet_switch_your_vision_language_action_model_into_a_histo_dissection.md)
- [Harness VLA：通过记忆引导代理将冻结 VLA 转化为可靠操作原语 (Harness VLA: Steering Frozen VLAs into Reliable Manipulation Primitives via Memory-Guided Agents)](./harness_vla_steering_frozen_vlas_into_reliable_manipulation_dissection.md)
- [隐于无形：基于扩散模型的无限制 VLA 机器人攻击 (Hidden in Plain Sight: Diffusion-Based Unrestricted Robotic Attacks on Vision-Language-Action Models)](./hidden_in_plain_sight_diffusion_based_unrestricted_robotic_a_dissection.md)
- [HiF-VLA：通过运动表示实现后见、洞察与前瞻的视觉 - 语言 - 动作模型 (HiF-VLA: Hindsight, Insight and Foresight through Motion Representation for Vision-Language-Action Models)](./hif_vla_hindsight_insight_and_foresight_through_motion_repre_dissection.md)
- [HiFOR：分层未来预测与动作引导世界检索的 VLA 拆解 (HiFOR: Hierarchical Foresight with Action-Guided World Grounding for Vision-Language-Action Model)](./hifor_hierarchical_foresight_with_action_guided_world_ground_dissection.md)
- [把人类在环后训练搬下机器人：HIL-UMI 的机器人免部署 VLA 后训练 (HIL-UMI: Bringing Human-in-the-Loop Post-Training of Vision-Language-Action Models to Universal Manipulation Interface)](./hil_umi_bringing_human_in_the_loop_post_training_of_vision_l_dissection.md)
- [ICI-VLA：基于时空对齐演示的上下文模仿学习 (ICI-VLA: In-Context Imitation with Spatiotemporally Aligned Demonstrations for Vision-Language-Action Models)](./ici_vla_in_context_imitation_with_spatiotemporally_aligned_d_dissection.md)
- [INSIGHT: 推理时序列自省生成人工辅助触发器 (INference-time Sequence Introspection for Generating Help Triggers in VLA Models)](./insight_inference_time_sequence_introspection_for_generating_dissection.md)
- [InstructVLA：从理解到操作的视觉 - 语言 - 动作指令微调 (InstructVLA: Vision-Language-Action Instruction Tuning from Understanding to Manipulation)](./instructvla_vision_language_action_instruction_tuning_from_u_dissection.md)
- [成功率就够了么？输入扰动如何改变 VLA 在桌面操作中的行为 (Is Success All You Need? Investigating the Impact of Input Perturbations on VLA Behaviour in Tabletop Manipulation Tasks)](./is_success_all_you_need_investigating_the_impact_of_input_pe_dissection.md)
- [JevVLA：用同一个能量标量为离散候选与连续动作块做轻量适配 (JevVLA: Jev-Inspired Energy-Based Adaptation for Vision-Language-Action Policies)](./jevvla_jev_inspired_energy_based_adaptation_for_vision_langu_dissection.md)
- [Juno：驯服预测隐变量用于 VLA 模型 (Juno: Taming Predictive Latents for Vision-Language-Action Models)](./juno_taming_predictive_latents_for_vision_language_action_mo_dissection.md)
- [最小可觉差建模用于 VLA Token 压缩 (Just Noticeable Difference Modeling for Token Compression in Vision-Language-Action Models)](./just_noticeable_difference_modeling_for_token_compression_in_dissection.md)
- [知道何时停止：基于 VLA 内部交叉注意力动态的自适应动作分块 (Knowing When to Stop: Adaptive Action Chunking via Internal Cross-Attention Dynamics in VLAs)](./knowing_when_to_stop_adaptive_action_chunking_via_internal_c_dissection.md)
- [LACY：基于视觉语言模型的双向语言-动作循环，实现机器人自我改进操作 (LACY: A Vision-Language Model-based Language-Action Cycle for Self-Improving Robotic Manipulation)](./lacy_a_vision_language_model_based_language_action_cycle_for_dissection.md)
- [LangGap：诊断与修复 VLA 模型的语言理解缺口 (Diagnosing and Closing the Language Gap in Vision-Language-Action Models)](./langgap_diagnosing_and_closing_the_language_gap_in_vision_la_dissection.md)
- [用运行时反馈自进化失败银行：让 VLA 安全约束"学进"策略里 (Learning from Runtime Feedback through Failure-Bank Self-Evolution for Vision-Language-Action Models)](./learning_from_runtime_feedback_through_failure_bank_self_evo_dissection.md)
- [学习语义原子技能用于多任务机器人操作 (Learning Semantic Atomic Skills for Multi-Task Robotic Manipulation)](./learning_semantic_atomic_skills_for_multi_task_robotic_manip_dissection.md)
- [从视觉语言模型学习结构化机器人策略 via 合成神经符号监督 (Learning Structured Robot Policies from Vision-Language Models via Synthetic Neuro-Symbolic Supervision)](./learning_structured_robot_policies_from_vision_language_mode_dissection.md)
- [通过自适应视觉 Token 缓存加速 VLA 模型 (Learning to Accelerate Vision-Language-Action Models through Adaptive Visual Token Caching)](./learning_to_accelerate_vision_language_action_models_through_dissection.md)
- [LIBERO-PRO: 超越死记硬背的 VLA 鲁棒与公平评估 (LIBERO-PRO: Towards Robust and Fair Evaluation of Vision-Language-Action Models Beyond Memorization)](./libero_pro_towards_robust_and_fair_evaluation_of_vision_lang_dissection.md)
- [Lift3D-VLA：将 VLA 模型提升至 3D 几何与动力学感知操作 (Lift3D-VLA: Lifting VLA Models to 3D Geometry and Dynamics-Aware Manipulation)](./lift3d_vla_lifting_vla_models_to_3d_geometry_and_dynamics_aw_dissection.md)
- [LingBot-VLA：20,000 小时真实数据预训练的实用主义 VLA](./lingbot_vla_pragmatic_foundation_model_2026.md)
- [LingBot-VLA：实用主义 VLA 基座模型与高吞吐训练栈 (LingBot-VLA: A Pragmatic VLA Foundation Model)](./lingbot_vla_pragmatic_vla_foundation_model_2026.md)
- [看哪里才重要：VLA 自适应视觉细化 (Look Where It Matters: Adaptive Visual Refinement for Vision-Language-Action Models)](./look_where_it_matters_adaptive_visual_refinement_for_vision_dissection.md)
- [LTM：面向 VLA 的潜转移上下文记忆（Latent Transition Based In-Context Memory for Vision-Language-Action Models）](./ltm_latent_transition_based_in_context_memory_for_vision_lan_dissection.md)
- [多臂协作的VLA：原子动作分配与组合泛化 (MA-VLA: Multi-Arm Vision-Language-Action Model for Collaboration and Compositional Generalization)](./ma_vla_multi_arm_vision_language_action_model_for_collaborat_dissection.md)
- [压碎、涂抹、切片！通过视觉空间进展学习物体状态操作 (Mash, Spread, Slice! Learning to Manipulate Object States via Visual Spatial Progress)](./mash_spread_slice_learning_to_manipulate_object_states_via_v_dissection.md)
- [评估 VLA 的「记忆」：MIKASA-Robo-VLA 长时序操作基准 (MIKASA-Robo-VLA: Benchmarking Memory in VLA Models for Long-Horizon Manipulation)](./mikasa_robo_vla_benchmarking_memory_in_vla_models_for_long_h_dissection.md)
- [运动聚焦潜在动作实现跨具身 VLA 人类视频训练 (Motion-Focused Latent Action Enables Cross-Embodiment VLA Training from Human EgoVideos)](./motion_focused_latent_action_enables_cross_embodiment_vla_tr_dissection.md)
- [以運動為中心的未來動力學：MotionWeave 給 VLA 的世界模型減負 (MotionWeave: Learning Motion-Centered Future Dynamics for Vision-Language-Action Policies)](./motionweave_learning_motion_centered_future_dynamics_for_vis_dissection.md)
- [MuseVLA：自适应多模态感知视觉-语言-动作模型 (MuseVLA: An Adaptive Multimodal Sensing Vision-Language-Action Model for Robotic Manipulation)](./musevla_an_adaptive_multimodal_sensing_vision_language_actio_dissection.md)
- [神经动作编解码器：用音频 Codec 重做 VLA 的动作分词 (NAC: Neural Action Codec for Vision-Language-Action Models)](./nac_neural_action_codec_for_vision_language_action_models_dissection.md)
- [叙事流：用机器人速度场做流匹配的 VLA 模型 (NarrativeFlow: Flow-Based Vision-Language-Action Model Using Robot Velocity Fields)](./narrativeflow_flow_based_vision_language_action_model_using_dissection.md)
- [NebulaVLA: 双频率视觉-语言-动作模型 (NebulaVLA: A Dual-Frequency Vision-Language-Action Model With Guide Action for Robotic Manipulation)](./nebulavla_a_dual_frequency_vision_language_action_model_with_dissection.md)
- [Nexus-VLA：用共享分层语义隐空间桥接人类与机器人经验 (Nexus-VLA: Bridging Human and Robot Experiences via a Shared Hierarchical Semantic Latent Space)](./nexus_vla_bridging_human_and_robot_experiences_via_a_shared_dissection.md)
- [噪声空间学习：冻结 flow-matching VLA 的轻量适配 (Noise Space Learning for VLA Adaptation)](./noise_space_learning_for_vla_adaptation_dissection.md)
- [不是所有特征都生而平等：VLA 的机制可解释性研究 (Not All Features Are Created Equal: A Mechanistic Study of Vision-Language-Action Models)](./not_all_features_are_created_equal_a_mechanistic_study_of_vi_dissection.md)
- [神经符号 VLA：把操作拆成「原始動作計劃」再餵給連續控制器 (NS-VLA: Towards Neuro-Symbolic Vision-Language-Action Models)](./ns_vla_towards_neuro_symbolic_vision_language_action_models_dissection.md)
- [OGPO：面向可恢复 VLA 的离线目标条件策略优化 (Offline Goal-conditioned Policy Optimization for Recoverable Vision-Language-Action Models)](./ogpo_offline_goal_conditioned_policy_optimization_for_recove_dissection.md)
- [OneTwoVLA 模型解构 (Dissecting OneTwoVLA)](./onetwovla.md)
- [完全开源 VLA 选型指南：谁是真开源，谁在"开源洗"](./open_source_vla_guide.md)
- [PDRVLA：把物理动力学表征压进 VLA，而不是在控制回路里生成视频 (PDRVLA: A Vision–Language–Action Model with Physical Dynamics Representations)](./pdrvla_a_vision_language_action_model_with_physical_dynamics_dissection.md)
- [Physical Intelligence：构建通用具身基础模型 — Sergey Levine 深度访谈解析](./physical_intelligence_sergey_levine_foundation_model_vision_2026.md)
- [Pi 0.5 模型解剖 (Dissecting $\\pi\_{0.5}$)](./pi0_5_dissection.md)
- [Pi 0.6 &amp; π\*0.6 模型解剖 (Dissecting π0.6)](./pi0_6_dissection.md)
- [π0.7：可操控的通用机器人基础模型，涌现出组合泛化能力](./pi0_7_steerable_compositional_generalization_2026.md)
- [π0 模型代码深度解析](./pi0_code_analysis.md)
- [PriorVLA：保留先验的 VLA 微调框架 (Prior-Preserving Adaptation for Vision-Language-Action Models)](./priorvla_prior_preserving_adaptation_for_vision_language_act_dissection.md)
- [精密強迫：讓通用 VLA 策略跨過亞毫米容差門檻 (PRISM: Precision Forcing for Generalist Vision-Language-Action Policies)](./prism_precision_forcing_for_generalist_vision_language_actio_dissection.md)
- [Progress Probes P²：用 VLA 内部激活做轻量运行时监控 (Progress Probes P²: Lightweight Runtime Monitoring for Vision-Language-Action Models)](./progress_probes_p2_lightweight_runtime_monitoring_for_vision_dissection.md)
- [RDT: 机器人扩散变换器 (Robotics Diffusion Transformer)](./rdt.md)
- [REACT：滚动去噪与双重解耦的 VLA 反应式机器人控制（REACT: Rolling Denoising and Dual Decoupling for Reactive Robot Control with VLA Models）](./react_rolling_denoising_and_dual_decoupling_for_reactive_rob_dissection.md)
- [ReconVLA：用“重建式监督”做隐式视觉接地（让 VLA 注意力不跑偏）](./reconvla_implicit_grounding_by_reconstruction.md)
- [通过质量多样性环境生成对 VLA 做红队评测 (Red-Teaming of Vision-Language-Action Models via Quality Diversity Environment Generation)](./red_teaming_of_vision_language_action_models_via_quality_div_dissection.md)
- [基于反思的任务适应：自改进 VLA 框架 (Reflection-Based Task Adaptation for Self-Improving VLA)](./reflection_based_task_adaptation_for_self_improving_vla_dissection.md)
- [Reflex: 面向反应关键型操作的快速预测 VLA (Reflex: Enabling Fast and Predictive Vision-Language-Action Models for Reaction-Critical Manipulation)](./reflex_enabling_fast_and_predictive_vision_language_action_m_dissection.md)
- [记住你做过什么：动作历史记忆 + 双专家去噪的长时序 VLA 策略 (Remember What You Did: Action-History Memory with Dual-Expert Denoising for Long-Horizon VLA Policies)](./remember_what_you_did_action_history_memory_with_dual_expert_dissection.md)
- [改写先行：刻画并缓解 VLA 的语言敏感性 (Rephrase Before You Act: Characterizing and Mitigating Language Sensitivity in Vision-Language-Action Models)](./rephrase_before_you_act_characterizing_and_mitigating_langua_dissection.md)
- [RePO-VLA：把失败 rollout 拆成恢复监督的 VLA 策略优化 (Recovery-Driven Policy Optimization for Vision-Language-Action Models)](./repo_vla_recovery_driven_policy_optimization_for_vision_lang_dissection.md)
- [报警值不值得响应：VLA 运行时监控器的反事实评估 (RESCUE: Is This Alarm Worth Acting On? Counterfactual Evaluation of VLA Runtime Monitors)](./rescue_is_this_alarm_worth_acting_on_counterfactual_evaluati_dissection.md)
- [通过免训练注意力重校准恢复 VLA 模型的语言 grounding (Restoring Linguistic Grounding in VLA Models via Train-Free Attention Recalibration)](./restoring_linguistic_grounding_in_vla_models_via_train_free_dissection.md)
- [检索替代微调：测试时扩展 VLA 至新任务 (Retrieve, Don't Retrain: Extending Vision Language Action Models to New Tasks at Test Time)](./retrieve_dont_retrain_extending_vision_language_action_model_dissection.md)
- [按时检索，频域校正：冻结 VLA 的免训练测试时修正框架 (Retrieve in Time, Correct in Frequency)](./retrieve_in_time_correct_in_frequency_dissection.md)
- [检索即行动：运动基元图检索提升 VLA 鲁棒执行 (Retrieve to Act: Motion Primitive Graph Retrieval for Robust VLA Execution)](./retrieve_to_act_motion_primitive_graph_retrieval_for_robust_dissection.md)
- [RhinoVLA 技术报告：面向端侧实时部署的跨本体 VLA 系统 (RhinoVLA Technical Report)](./rhinovla_technical_report_dissection.md)
- [RoboNeuron：连接 Agent 工具调用与 ROS2 执行的中间件架构 (RoboNeuron: A Middle-Layer Infrastructure for Agent-Driven Orchestration in Embodied AI)](./roboneuron_a_middle_layer_infrastructure_for_agent_driven_or_dissection.md)
- [RoboTwin-Phys：WAM 与 VLA 真的理解物理世界吗？ (RoboTwin-Phys: Do WAMs and VLAs Understand the Physical World?)](./robotwin_phys_do_wams_and_vlas_understand_the_physical_world_dissection.md)
- [RotVLA：旋轉潛動作——讓 Latent Action 住進 SO(n) (RotVLA: Rotational Latent Action for Vision-Language-Action Model)](./rotvla_rotational_latent_action_for_vision_language_action_m_dissection.md)
- [用运动学路由，用观察执行：MoE 增强 VLA 中的运动学监督专家路由 (Route by Kinematics, Act by Observation: Kinematics-Supervised Expert Routing in MoE-Augmented VLA)](./route_by_kinematics_act_by_observation_kinematics_supervised_dissection.md)
- [RouterVLA：将烟雾测试转化为异构 VLA 选择的监督信号 (RouterVLA: Turning Smoke Tests into Supervision for Heterogeneous VLA Selection)](./routervla_turning_smoke_tests_into_supervision_for_heterogen_dissection.md)
- [物理现实锚定的具身基础模型：RynnBrain (RynnBrain: Open Embodied Foundation Models)](./rynnbrain_open_embodied_foundation_models_2026.md)
- [给冻结 VLA 装一圈「安全带」：SafeLoop 的风险预测 + 回滚恢复 (SafeLoop: Risk-Aware Rollback for Vision-Language-Action Manipulation)](./safeloop_risk_aware_rollback_for_vision_language_action_mani_dissection.md)
- [SaiVLA-JEPA：在 JEPA 条件化的 VLA 控制中检验未来隐表示预测 (SaiVLA-JEPA: Evaluating Future Latent Prediction in JEPA-Conditioned VLA Control)](./saivla_jepa_evaluating_future_latent_prediction_in_jepa_cond_dissection.md)
- [SEAM：动作块平滑执行 (Smooth Execution of Action-Chunked Motion for Vision-Language-Action Policies)](./seam_smooth_execution_of_action_chunked_motion_for_vision_la_dissection.md)
- [自適應 VLA：讓部署後的策略用「自己的失敗軌跡」逐步補償硬體漂移 (Self-Adaptive VLA for Robust Robot Deployment)](./self_adaptive_vla_for_robust_robot_deployment_dissection.md)
- [把幾何接進動作專家：sensVLA 的空間接地 VLA 架構 (sensVLA: Spatially-Grounded Vision-Language-Action Model for Autonomous Wheel Loader)](./sensvla_spatially_grounded_vision_language_action_model_for_dissection.md)
- [SimVLA：为什么一个“简单 VLA baseline”反而值得全行业重看？ (SimVLA: A Simple VLA Baseline for Robotic Manipulation)](./simvla_simple_vla_baseline_robotic_manipulation_2026.md)
- [小模型 VLA 研究方向深度分析](./small_vla_models.md)
- [紧凑 VLA 的持久循环记忆：SmoLSTM (SmoLSTM: A Compact Vision-Language-Action Model with Recurrent Memory that Persists)](./smolstm_a_compact_vision_language_action_model_with_recurren_dissection.md)
- [Spirit-v1.5 模型解剖（Dissecting Spirit-v1.5）](./spirit_v1_5_dissection.md)
- [Splat2Act：从捕获场景到个性化 VLA 操作（Splat2Act: From Captured Scenes to Personalized VLA Manipulation）](./splat2act_from_captured_scenes_to_personalized_vla_manipulat_dissection.md)
- [异步回放锚定的在线 VLA 后训练 (Stable and Efficient Real-World Online VLA Post-Training via Asynchronous Replay-Anchored Policy Improvement)](./stable_and_efficient_real_world_online_vla_post_training_via_dissection.md)
- [StageCraft：執行期干擾物與阻塞失效的緩解 (StageCraft: Execution Aware Mitigation of Distractor and Obstruction Failures in VLA Models)](./stagecraft_execution_aware_mitigation_of_distractor_and_obst_dissection.md)
- [StarVLA：Lego-like 的 VLA 研发代码库 (StarVLA: A Lego-like Codebase for VLA)](./starvla_lego_like_vla_codebase_2026.md)
- [驯服执行误差下的 VLA：部署期自补偿与压力测试 (Taming VLAs under Robot Execution Errors: Self-Compensation and Stress Testing)](./taming_vlas_under_robot_execution_errors_self_compensation_a_dissection.md)
- [TANGO: 在杂乱的室内环境中用全身 VLA 导航 (Humanoid Navigation in Cluttered Environments with a Whole-Body Vision-Language-Action Model)](./tango_humanoid_navigation_in_cluttered_environments_with_a_w_dissection.md)
- [TemporalFlow-VLA：学习物理接地的执行历史以支持长时程机器人操作 (TemporalFlow-VLA: Learning Physically Grounded Execution History for Long-Horizon Robot Manipulation)](./temporalflow_vla_learning_physically_grounded_execution_hist_dissection.md)
- [TempoVLA：速度可控的 Vision-Language-Action 策略 (TempoVLA: Learning Speed-Controllable Vision-Language-Action Policies)](./tempovla_learning_speed_controllable_vision_language_action_dissection.md)
- [视觉预见VLA的测试时训练 (Test-Time Training for Visual Foresight Vision-Language-Action Models)](./test_time_training_for_visual_foresight_vision_language_acti_dissection.md)
- [按需思考：选择性慢路径干预的 Prompt-Authority 控制 (Think Only When Needed: Prompt-Authority Control for Selective Slow-Path Intervention in Vision-Language-Action Manipulation)](./think_only_when_needed_prompt_authority_control_for_selectiv_dissection.md)
- [TIDAL：时间交错扩散与动作循环实现高频 VLA 控制 (TIDAL: Temporally Interleaved Diffusion and Action Loop for High-Frequency VLA Control)](./tidal_temporally_interleaved_diffusion_and_action_loop_for_h_dissection.md)
- [迈向类人物理智能：面向机器人操作的终身视觉-语言-动作学习 (Towards Human-like Physical Intelligence: Lifelong Vision-Language-Action Learning for Robotic Manipulation)](./towards_human_like_physical_intelligence_lifelongvision_lang_dissection.md)
- [传输差异作为 VLA 模型的可靠性信号 (Transport Discrepancy as a Reliability Signal for Vision-Language-Action Models)](./transport_discrepancy_as_a_reliability_signal_for_vision_lan_dissection.md)
- [TwinVLA：用双臂"孪生"实现数据高效的双手机器人操作 (TwinVLA: Data-Efficient Bimanual Manipulation with Twin Single-Arm Vision-Language-Action Models)](./twinvla_data_efficient_bimanual_manipulation_with_twin_singl_dissection.md)
- [UAV-Track VLA：具身无人机跟踪的视觉 - 语言 - 动作模型 (UAV-Track VLA: Embodied Aerial Tracking via Vision-Language-Action Models)](./uav_track_vla_embodied_aerial_tracking_via_vision_language_a_dissection.md)
- [流匹配 VLA 的不确定性量化 (Uncertainty Quantification for Flow-Based Vision-Language-Action Models)](./uncertainty_quantification_for_flow_based_vision_language_ac_dissection.md)
- [统一噪声引导：高效人类指导的 VLA 适配 (Unified Noise Steering for Efficient Human-Guided VLA Adaptation)](./unified_noise_steering_for_efficient_human_guided_vla_adapta_dissection.md)
- [统一视觉运动目标：超越物理动作的 VLA 监督 (Unified Visuomotor Targets: Supervising VLAs Beyond Physical Actions)](./unified_visuomotor_targets_supervising_vlas_beyond_physical_dissection.md)
- [UnifoLM-VLA-0：Unitree 的 VLA 训练与部署开源实现 (UnifoLM-VLA-0: Unitree’s VLA Open-Source Stack)](./unifolm_vla_0_unitree_2026.md)
- [UniLACT：深度感知 RGB 潜在动作学习 (UniLACT: Depth-Aware RGB Latent Action Learning for VLA Models)](./unilact_depth_aware_rgb_latent_action_learning_for_vision_la_dissection.md)
- [UniSteer：统一噪声引导的高效人类指导 VLA 自适应 (UniSteer: Unified Noise Steering for Efficient Human-Guided VLA Adaptation)](./unisteer_unified_noise_steering_for_efficient_human_guided_v_dissection.md)
- [UniTexture：跨任务通用对抗纹理攻击 VLA 模型 (UniTexture: Cross-Task Universal Adversarial Textures for Vision-Language-Action Models)](./unitexture_cross_task_universal_adversarial_textures_for_vis_dissection.md)
- [USIM and U0：面向通用水下机器人的 VLA 数据集与模型 (USIM and U0: A Vision-Language-Action Dataset and Model for General Underwater Robots)](./usim_and_u0_a_vision_language_action_dataset_and_model_for_g_dissection.md)
- [ValueFormer：阶段感知价值标签，让 VLA 的「静默失败」可被在线读出 (ValueFormer: Stage-Aware Value Labels for Online Failure Detection and VLA Post-Training)](./valueformer_stage_aware_value_labels_for_online_failure_dete_dissection.md)
- [把测试时计算蒸馏进策略：Verify, Act, Distill (Turning Test-Time Compute into Test-Time Adaptation for Vision-Language-Action Models)](./verify_act_distill_turning_test_time_compute_into_test_time_dissection.md)
- [VersaCamVLA：面向机器人操作的相机可配置 VLA 策略 (VersaCamVLA: Camera-Configurable VLA Policies for Robotic Manipulation)](./versacamvla_camera_configurable_vla_policies_for_robotic_man_dissection.md)
- [VGAS：价值引导的动作块选择用于少样本 VLA 适配 (Value-Guided Action-Chunk Selection for Few-Shot Vision-Language-Action Adaptation)](./vgas_value_guided_action_chunk_selection_for_few_shot_vision_dissection.md)
- [VITRA：用真实人类手部视频做可扩展 VLA 预训练 (Scalable Vision-Language-Action Model Pretraining with Real-Life Human Activity Videos)](./vitra_scalable_vla_pretraining_human_activity_videos_2026.md)
- [VLA 核心架构 (VLA Core Architectures)](./vla_arch.md)
- [VLA 研究主线梳理](./vla_research_mainline.md)
- [VLA-Trace：通过表征与行为追踪诊断 VLA 模型 (VLA-Trace: Diagnosing Vision-Language-Action Models through Representation and Behavior Tracing)](./vla_trace_diagnosing_vision_language_action_models_through_r_dissection.md)
- [云边交错：VLA-ULAP 用 7.4M 轻量预测器砍掉近半云端 VLA 调用 (VLA-ULAP: Interleaving Cloud VLA Calls with Ultra-Lightweight Local Action Prediction at the Edge)](./vla_ulap_interleaving_cloud_vla_calls_with_ultra_lightweight_dissection.md)
- [VLANeXt：构建强 VLA 模型的配方 (VLANeXt: Recipes for Building Strong VLA Models)](./vlanext_recipes_for_building_strong_vla_models_dissection.md)
- [VLA 受限于训练但具备新指令泛化能力 (VLAs are Confined yet Capable of Generalizing to Novel Instructions)](./vlas_are_confined_yet_capable_of_generalizing_to_novel_instr_dissection.md)
- [DualVLN：Ground Slow, Move Fast —— VLN 的“快慢系统”基础模型](./vln_dualvln.md)
- [VQActFlow：向量量化動作流的多任務機器人操控 (VQActFlow: Vector-Quantized Action Mode Steering for Multi-Task Robot Manipulation)](./vqactflow_vector_quantized_action_mode_steering_for_multi_ta_dissection.md)
- [WALL-OSS: 点燃 VLM 走向具身空间](./wall_oss.md)
- [机器人策略该学什么预测？重审 VLA 学习的监督目标 (What Should Robot Policies Learn to Predict? Rethinking Supervision Targets in Vision-Language-Action Learning)](./what_should_robot_policies_learn_to_predict_rethinking_super_dissection.md)
- [World Pilot：用世界-动作先验引导 VLA 决策 (World Pilot: Steering Vision-Language-Action Models with World-Action Priors)](./world_pilot_steering_vision_language_action_models_with_worl_dissection.md)
- [Xiaomi-Robotics-0：用 RoPE 偏移 + $\\Lambda$-mask + 動態 loss 三件套破解 Action Prefix 的「shortcut」 (An Open-Sourced VLA Model with Real-Time Execution)](./xiaomi_robotics_0_dissection.md)

<!-- VLA:THEORY-NAV:END -->
