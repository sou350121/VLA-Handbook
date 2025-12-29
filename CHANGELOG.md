# Changelog

All notable changes to the **VLA Handbook** project are documented here, derived directly from the repository's git history.

---

## [1.5.0] - 2025-12-29 🆕
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
