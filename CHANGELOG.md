# Changelog

All notable changes to the **VLA Handbook** project are documented here, derived directly from the repository's git history.

---

## [1.8.0] - 2026-01-11 to 2026-01-12 🆕
### Added
- **Spirit-v1.5 深度拆解**: 新增 [`theory/spirit_v1_5_dissection.md`](./theory/spirit_v1_5_dissection.md)，包含：
    - **核心架构**：Qwen3-VL (大脑) + DiT (小脑) + ODE Euler 积分 (执行)。
    - **多样化数据采集 (Diverse Collection)**：深度解析 Spirit AI 为什么认为“干净数据是伟大模型的敌人”。
    - **RoboChallenge Table30**：首个超越 π0.5 登顶榜单的代码级复现指南与数据对齐细节（包含不同机器人型号的后处理逻辑）。
- **灵巧手机械学深度专题**: 新增 [`theory/dexterous_hand_mechanics.md`](./theory/dexterous_hand_mechanics.md)：
    - 机构学基础 (Grubler's Criterion)、四连杆机构、行星减速器、锥齿轮分析。
    - 传动学对比 (直驱 vs 线驱 vs 液压/Sanctuary AI Phoenix)。
    - 运动学 Jacobian 矩阵与动力学阻抗控制数学表达。
- **具身数据采集概览**: 新增 [`deployment/embodied_data_collection_overview.md`](./deployment/embodied_data_collection_overview.md)：
    - 覆盖第一视角 POV (EgoScale)、Sim2Real 规模化、真机 RL、专家示教 (GELLO/Manus) 等五大路径对比。
    - 深入探讨了“脏数据”、“废数据”与“有效信息密度”的工业界定义。
- **多模态数据同步技术**: 新增 [`deployment/multimodal_data_synchronization.md`](./deployment/multimodal_data_synchronization.md)，解决视觉 (30Hz) 与控制 (1000Hz) 的毫秒级对齐、PTP 协议与硬件触发同步。
- **公司手册更新**: 加入 **Lumos Robot (鹿明机器人)** 专题，解析其“超级数据工厂”定位、鹿明指数与 FastUMI Pro 系统。
- **硬件选型更新**: 
    - 新增 **Sharpa Wave** (DTA 动态触觉阵列)、**LEAP Hand V2 Adv**、**RealerHand (睿尔灵)** 详情。
    - **灵巧手典型挑战**：开可乐罐 (指甲利用与杠杆原理)、抓取手机薄片 (桌面碰撞与柔顺性)、重载抓取 (远端关节扭矩要求)。

### Changed & Optimized
- **全站汉化**: 完成所有 140+ 文档的简体中文翻译与本地化，统一技术术语。
- **部署架构重组**: 重新梳理 [`deployment/README.md`](./deployment/README.md) 索引，按硬件选型、感知同步、机械臂控制、灵巧手专题、仿真数采五大板块分类。
- **公式渲染适配**: 全面优化 `math_for_vla.md` 与 `vla_loss_functions_handbook.md`，使用 `$$` 块与空行强制适配 GitHub Markdown 数学渲染。
- **清理不相关内容**: 移除了与机器人技术不相关的 `grade7b_math` 考试目录。

---

## [1.7.0] - 2026-01-09 to 2026-01-10
### Added
- **VLA 必备数学基础**: 新增 [`theory/math_for_vla.md`](./theory/math_for_vla.md)，系统整理了从线性代数、空间表示 (SE3) 到扩散模型、流匹配的完整数学链条。
- **VLA 损失函数手册**: 新增 [`theory/vla_loss_functions_handbook.md`](./theory/vla_loss_functions_handbook.md)，包含 NLL、KL 散度、ELBO 等公式大白话翻译与 PyTorch 实现。
- **前馈与反馈控制**: 在 [`theory/robot_control.md`](./theory/robot_control.md) 中增加 Feedforward vs Feedback 专题，辅以“抓取透明水瓶”的工程案例。
- **灵巧手实战案例集**: 将 VisionOS (Webcam 遥操作) 与 Wuji 手的 retargeting 实战整合至 [`deployment/dexterous_hand_applications.md`](./deployment/dexterous_hand_applications.md)。

---

## [1.6.0] - 2026-01-06
### Added
- **实战案例：手势控制灵巧手**: 新增 [`deployment/mediapipe_wujihand_project.md`](./deployment/mediapipe_wujihand_project.md)，包含：
    - MediaPipe + WujiHand 联动架构（WebSocket + USB SDK）。
    - **延迟优化实战**：记录从 500ms 降至 50ms 的优化路径（移除软件滤波、启用硬件 LowPass、非阻塞写入）。
    - **面试 Q&A 模拟**：针对实时控制、延迟优化、多维映射等核心问题的专业话术建议。
- **Jim Fan 2025 年度复盘**: 新增 [`theory/frontier/jim_fan_2025_robotics_lessons.md`](./theory/frontier/jim_fan_2025_robotics_lessons.md)。
- **触觉不可替代性与视触觉前沿**: 
    - 新增 [`theory/frontier/tactile_irreplaceable.md`](./theory/frontier/tactile_irreplaceable.md)。
    - 新增 [`theory/frontier/unitachhand.md`](./theory/frontier/unitachhand.md)（人手→机器人灵巧手策略零样本迁移）。
    - 更新 [`theory/tactile_vla.md`](./theory/tactile_vla.md) 的 Q&A 部分，增加「Demo 为何翻车」与「工程 Checklist」。

### Changed
- **理论索引更新**: 在 [`theory/README.md`](./theory/README.md) 中新增「Part 7: 实战案例与部署」板块。

---

## [1.5.0] - 2025-12-29
### Added
- **AI Coding Agent Design Deep Dive**: New document [`system-design/ai_coding_agent_design.md`](./system-design/ai_coding_agent_design.md) covering:
  - User prompt preprocessing (@context, slash commands).
  - MCP (Model Context Protocol) architecture and tool discovery.
  - SubAgent implementation and context isolation.
  - Spec-driven development (OpenSpec interpretation).
- **Data Flywheel & Cross-modal Transfer**: New document [`theory/frontier/data_flywheel_and_cross_modal.md`](./theory/frontier/data_flywheel_and_cross_modal.md) analyzing:
  - Data scarcity solutions in robotics.
  - Humanoid foundation models using internet video data.
  - Boundless Intelligence (智在无界) case study.
- **Robot Startup Category**: Added [`companies/startups.md`](./companies/startups.md) featuring Boundless Intelligence (智在无界) and CEO Zongqing Lu's team.
- **Nature Communications Reward Discovery**: Added [`theory/frontier/reward_discovery_rl.md`](./theory/frontier/reward_discovery_rl.md).

### Changed
- **ROS2 Section Reinforcement**: Major updates to [`deployment/ros_and_optimization.md`](./deployment/ros_and_optimization.md):
  - Added Zero-Copy (Iceoryx) and CycloneDDS tuning.
  - Added Component Containers and WaitSet real-time executor models.
  - Added Unicast/Peers-list discovery for distributed deployment.
  - Added safety certification (Apex.OS) content.
- **Robot Dynamics Classification Refinement**: Updated [`theory/robot_dynamics_classification.md`](./theory/robot_dynamics_classification.md) with English terms and algorithm-friendly analogies.

---

## [1.4.0] - 2025-12-26 to 2025-12-28
### Added
- **OneTwoVLA Adaptation**: Real-world migration guide from 7-DOF Franka to 6-DOF UR5 in `ur5_control_guide.md`.
- **Research Frontier Restructuring**: Organized model-specific analyses into `theory/frontier/`.
- **Co-training Examples**: Added Robot vs Internet data comparison to π0.5 dissection.

### Changed & Fixed
- **GitHub Math Rendering Global Fix**: Optimized LaTeX layout across all 40+ documents, strictly enforcing blank lines around `$$` blocks for correct web display.
- **Architecture Diagrams**: Replaced Mermaid with ASCII and enhanced diagrams for GR00T-N1.6, π0, and π0.5.
- **Math Deep Dives**: Significantly enhanced explanations for Diffusion Policy and Flow Matching.

---

## [1.3.0] - 2025-12-21 to 2025-12-25
### Added
- **GR00T-N1.6 Deep Dive**: Detailed analysis of NVIDIA's humanoid foundation model, DiT architecture, and conditioning.
- **Tesla Optimus V2 Analysis**: Added hardware analysis of the latest Optimus hand and compared it with the Wuji hand.
- **MM-ACT (Unified Token Space)**: Analysis of multi-modality unified token space.
- **SGTM (VLA Intrinsic Safety)**: Added chapter on safety and alignment.

---

## [1.2.0] - 2025-12-15 to 2025-12-18
### Added
- **VLN DualVLN**: Introduced Vision-Language Navigation with Dual-system (Fast/Slow).
- **Traditional Action Generation**: Added MSE/GMM explanation to `theory/traditional_action_generation.md`.
- **LeetCode for Beginners**: Added oral-style LeetCode training diary to the question bank.

### Changed
- **Handbook Restructuring**: Renamed to "VLA Handbook", optimized README with theory-first logic and collapsible sections.
- **First Principles Math**: Added dedicated math sections to core theory documents.

---

## [1.1.0] - 2025-12-10 to 2025-12-13
### Added
- **NeurIPS 2025 Insights**: Decoded top papers from an Embodied AI perspective (Artificial Hivemind, Gated Attention, etc.).
- **GELLO Deployment**: Added guide for assembly and UR5 teleoperation using GELLO.
- **SaTA (Tactile Awareness)**: Deep analysis added to `tactile_vla.md`.
- **Isaac Lab**: Added documentation for the GPU-accelerated simulation framework.

### Fixed
- **GELLO Pricing**: Corrected assembled price to ~¥2000 and added Taobao purchase links.

---

## [1.0.0] - 2025-12-01 to 2025-12-08
### Added
- **Small VLA Research**: Comprehensive analysis of Evo-1, SmolVLA, and Latent Action Learning (UniVLA, MemoryVLA).
- **ByteDance GR-RL**: Dissection of the first VLA to achieve 78% shoe-tying success.
- **Tactile VLA Module**: Added GelSight, Digit, DM-Tac, and GelStereo sensor analyses.
- **VLM Comparison Tables**: Added PaliGemma 3B, SigLIP, and Qwen2-VL comparison for VLA backbones.
- **Motion Planning & SLAM**: Added foundational modules for perception and navigation.

### Changed
- **README_FUN.md**: Introduced "Robot Growth School" analogy for easier onboarding.
- **ASCII Cheat Sheet**: Added visual diagrams for core concepts like LoRA, CoT, and Flash Attention.

---

## [0.8.0] - 2024-11-25 to 2024-11-30
### Added
- **Product Encyclopedia**: Detailed specs and pricing for 50+ humanoid robots, arms, hands, and sensors.
- **Evaluation Protocols**: Math definitions for success rates, A/B testing, and bench details.
- **Simulation Guide**: Comprehensive guide on Isaac Sim, MuJoCo, and SAPIEN.

### Changed
- **Co-training Refactor**: Extracted Co-training to a dedicated document `theory/co_training.md`.
- **Data Pipeline**: Detailed format comparison (RLDS vs LeRobot).

---

## [Initial Release] - 2024-11-15
### Added
- **Core VLA Theory**: Transformer vs CNN, RT-1/RT-2 architectures, and action representations.
- **Hardware Pricing**: Initial hardware guide and RealSense calibration basics.
- **Action Tokens**: Initial research on discrete vs continuous action spaces.
