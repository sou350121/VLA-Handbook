# Changelog

All notable changes to the **VLA Handbook** project will be documented in this file.

---

## [Unreleased] - 2025-12-29

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

### Changed
- **ROS2 Section Reinforcement**: Major updates to [`deployment/ros_and_optimization.md`](./deployment/ros_and_optimization.md):
  - Added Zero-Copy (Iceoryx) and CycloneDDS tuning.
  - Added Component Containers and WaitSet real-time executor models.
  - Added Unicast/Peers-list discovery for distributed deployment.
- **Robot Dynamics Classification Refinement**: Updated [`theory/robot_dynamics_classification.md`](./theory/robot_dynamics_classification.md) with:
  - English terminology for all professional terms.
  - "Algorithm-friendly" analogies for AI engineers.
  - Expanded standard dynamics equation analysis.

---

## [1.2.0] - 2025-12-28

### Added
- **Robot Dynamics Classification**: New document [`theory/robot_dynamics_classification.md`](./theory/robot_dynamics_classification.md) covering:
  - Over-constrained, Under-constrained, and Fully-constrained systems.
  - Floating-base vs Grounded systems.
  - Inertia completeness and numerical stability.
- **Reward Discovery RL**: New document [`theory/frontier/reward_discovery_rl.md`](./frontier/reward_discovery_rl.md) analyzing Nature Communications 2025 paper.
- **OneTwoVLA Dissection**: New document [`theory/frontier/onetwovla.md`](./theory/frontier/onetwovla.md) on adaptive reasoning/action tokens.
- **MM-ACT (Unified Token Space)**: New document [`theory/frontier/vla_unified_token_space.md`](./theory/frontier/vla_unified_token_space.md).
- **SGTM (Intrinsic Safety)**: New document [`theory/frontier/vla_intrinsic_safety.md`](./theory/frontier/vla_intrinsic_safety.md).

### Changed
- **UR5 Control Guide**: Added real-world adaptation from 7-DOF Franka to 6-DOF UR5.
- **Main README**: Refactored into a research-oriented landing page with "Theory Fast Track".

---

## [1.1.0] - 2025-12-26

### Added
- **VLN Special Topic**: Added [`theory/vln_dualvln.md`](./theory/vln_dualvln.md) - Dual-system for Vision-Language Navigation.
- **UR5 Python Control 实战**: New guide [`deployment/ur5_control_guide.md`](./deployment/ur5_control_guide.md) with real-time kernel config and `ur_rtde` examples.
- **Python OOP for Robotics**: Added OOP design patterns and safety decorators to question bank.

---

## [1.0.0] - 2025-12-13

### Added
- **NeurIPS 2025 Insights**: Added [`theory/neurips_2025_insights.md`](./theory/neurips_2025_insights.md) covering top papers from an Embodied AI perspective.
- **NVIDIA GR00T-N1.6**: Deep dive into the latest DiT-based humanoid foundation model.

---

## [0.9.0] - 2024-12-08

### Added
- **Small VLA Models**: Added research on Evo-1 (770M) and SmolVLA (450M).
- **Latent Action Learning**: New section on UniVLA, EvoVLA, and MemoryVLA.
- **ByteDance GR-RL**: Analysis of the first VLA to achieve 78% shoe-tying success.

---

## [0.8.0] - 2024-12-05

### Added
- **Multimodal Foundation**: Comprehensive guide on 2025 VLM models (Qwen2.5-VL, Eagle 2.5, etc.).
- **π0.6 Action Expert**: Dissection of Recap self-evolution mechanism.

---

## [0.7.0] - 2024-12-01

### Added
- **Perception & Planning**: Added Visual Perception, Motion Planning, and SLAM modules.
- **Sensor Integration**: Added tactile sensor and end-effector control guides.

---

## [Initial Release] - 2024-11-15

### Added
- Foundation documents: Transformer vs CNN, RL basics, and early VLA architectures (RT-1, RT-2).
- Hardware pricing tables and camera calibration basics.

