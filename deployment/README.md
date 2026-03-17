# 真机与部署 (Real-world & Deployment)

本模块关注 VLA 算法在真实物理世界中的落地与应用，涵盖从硬件选型、感知对齐到大规模数据采集的完整工程链路。

---

## ⚡ 卡住了？先看这个

**[📕 社区实战笔记・中文 (小红书)](./community_field_notes_xiaohongshu.md)** — 250+ 条中文社区蒸馏（帖1-161 + 可追溯索引 220 条）。论文不会告诉你的真实参数、真实失败、真实吐槽。新增 2026 Q1：LingBot-VLA / UnifoLM / π*0.6 真机 RL / Figure Helix / GR00T N2 等。

> 带着问题来的话，直接看笔记顶部的「你现在卡在哪？」速查表。
> 需要深入理解的话，每个 section 末尾有对应专题文件链接。
> **每周五 10:00 自动增量更新。**

**[📘 社区实战笔记・英文 (HF Blog / Discord / 厂商博客)](./community_field_notes_english.md)** — 300 条英文社区精选（Blog 165 + Discord 135）：SmolVLA / π0-FAST / GR00T N1.5 / ACT / DiffusionVLA / RD-VLA / SeedPolicy / EasyMimic 等最新论文拆解 + LeRobot Discord 真实部署经验、训练配方、推理延迟实测、数据迭代踩坑。

> 165 条 Blog 深度拆解（arXiv 前沿论文 + HuggingFace、NVIDIA、Phospho、ML6 等厂商博客）+ 135 条 LeRobot Discord 一手情报（#show-us-what-you-built / #robotics-papers）。
> 每条 Discord 原文保留英文全量展示，方便直接参考。
> **每周五 11:00 自动增量更新。**

**[🔧 GitHub 社区实战笔记 (GitHub Issues)](./community_field_notes_github.md)** — 200+ 条高互动 Issues 蒸馏。硬件兼容矩阵、训练收敛失败根因、官方 checkpoint 复现问题、跨仓库收敛信号。

> 21 个仓库（lerobot/openpi/GR00T/ManiSkill/IsaacLab/Genesis...），每条结论附 Issue 链接。
> **Pulsar GitHub Issues Sensor 每周自动扫描更新。**

---

## 目录

### 1. 硬件选型与通讯架构 (Hardware & Infrastructure)
- **[硬件选型与成本 (Hardware & Pricing)](./robot_hardware_selection_pricing.md)**: 包含 **Sharpa Wave**, **LEAP V2**, **Wuji** 等主流灵巧手/机械臂/传感器对比。📕 [社区选型指南](./community_field_notes_xiaohongshu.md#94-机械臂选型指南)
- **[可复现人形开源部署：Roboto_Origin](./roboto_origin_reproducible_humanoid_deployment.md)**: 从装配→ROS2 bring-up→仿真训练→Sim2Real→测试矩阵的落地清单。
- **[VLA 模型边缘部署优化 (VLA Edge Deployment)](./vla_model_edge_deployment.md)**: 量化 (GPTQ, AWQ) 与边缘推理 (TensorRT-LLM, vLLM)。📕 [社区加速实测](./community_field_notes_xiaohongshu.md#93-边缘部署与模型压缩)
- **[ROS 集成与算法优化 (ROS & Optimization)](./ros_and_optimization.md)**: ROS2 零拷贝、组件容器与 DDS 分布式调优。
- **[Agent 架构真机部署攻略 (Agentic VLA Deployment)](./agent_architecture_deployment_guide.md)**: 把 VLA 变成"能稳定做完任务"的系统：Planner/Policy/RT 控制分层、延迟预算、安全与恢复。📕 [社区推理卡顿解法](./community_field_notes_xiaohongshu.md#21-推理延迟与执行卡顿)

### 2. 感知、标定与多模态同步 (Sensing, Calibration & Sync)
- **[Perception 总索引](./perception/README.md)**: 感知系统工程（传感器选型、标定、同步、数据质量与评估）的主入口与跨文档索引。
- **[相机标定与手眼对齐 (Camera Calibration)](./camera_calibration_eye_in_hand.md)**: Eye-in-Hand vs Eye-to-Hand 标定实战。📕 [社区精度陷阱](./community_field_notes_xiaohongshu.md#22-硬件相关陷阱)
- **[多模态数据同步技术 (Multimodal Sync)](./multimodal_data_synchronization.md)**: 解决 RGB-D 与高频控制（1000Hz）的时间对齐难题。
- **[触觉集成挑战 (Tactile Integration)](./tactile_sensor_integration_challenges.md)**: 触觉传感器与夹爪集成的工程难点。
- **[运动传感器测量差异与 IEEE P3716 质量评估标准](./ieee_p3716_sports_tracking_quality_standard.md)**: 场地运动中人体/物体追踪测量一致性难题与标准化进展。

### 3. 机械臂控制与遥操作部署 (Robot Arm & Teleoperation)
- **[UR5 Python 控制实战 (UR5 Control Guide)](./ur5_control_guide.md)**: 实时内核配置、`ur_rtde` 高频控制与保护性停止恢复。
- **[GELLO 遥操作部署 (GELLO Deployment)](./gello_deployment.md)**: 低成本 3D 打印遥操作手柄配置与 LeRobot 格式转换。
- **[手势控制灵巧手：MediaPipe + WujiHand 实战项目](./mediapipe_wujihand_project.md)**: 感知→映射→控制的闭环工程骨架与低延迟优化抓手。
- **[Pi0 真机部署 (Pi0 Deployment)](./pi0_deployment.md)**: 官方 OpenPI 架构、Remote Inference 与硬件要求。📕 [社区 π0 微调配方](./community_field_notes_xiaohongshu.md#13-π0--openpi)

### 4. 灵巧手深度专题 (Dexterous Hand Deep Dive)
- **[灵巧手通讯与部署实战 (DexHand Communication)](./dexterous_hand_communication_deployment.md)**: 通讯架构 (CANFD, EtherCAT)、Retargeting 与线缆管理。
- **[灵巧手实战案例集 (DexHand Applications)](./dexterous_hand_applications.md)**: VisionOS 遥操作、跨设备动作映射与 Sim2Real 案例。
- **[Wuji 灵巧手深度解析 (Wuji Hand Deep Dive)](./dexterous_hand_wuji.md)**: 20-DOF 非拉索、全电机集成驱动技术方案。
- **[DexRobot DexHand021 量产版深度解析 (DexHand021 Production)](./dexrobot_dexhand021_production_dexhand_2026.md)**: 19-DOF + CANFD + ROS1/ROS2/micro-ROS + 多模态触觉落地拆解。
- **[Optimus Hand V2 解析](./optimus_hand_v2.md)**: Tesla Optimus 灵巧手技术特点分析。

### 5. 仿真、数据采集与 Sim2Real (Data, Sim & Training)
- **[具身智能数据采集概览 (Embodied Data Collection)](./embodied_data_collection_overview.md)**: POV 第一视角 (EgoScale)、Sim2Real 规模化与真机 RL。📕 [社区数据量门槛](./community_field_notes_xiaohongshu.md#41-数据量门槛)
- **[Evo-RL 仓库部署审计 (Evo-RL Repo Analysis)](./evo_rl_repo_analysis.md)**: 可运行性、稳定性、隐含假设与真机复现门槛。📕 [社区 RECAP 复现](./community_field_notes_xiaohongshu.md#16-rl-后训练-post-training)
- **[灵巧手数据采集方案 (DexHand Data Collection)](./dexterous_hand_data_collection.md)**: 结构化 Episode 定义、Retargeting 算法与数据回放验证。
- **[仿真环境详解 (Simulation Environments)](./simulation_environments.md)**: Isaac Sim vs MuJoCo vs PyBullet 选型指南。📕 [社区 MuJoCo 踩坑](./community_field_notes_xiaohongshu.md#33-mujoco-环境搭建)
- **[仿真基准与训练工具 (Sim Benchmarks & Tooling)](./simulation_benchmarks_and_tools.md)**: LIBERO / RLinf / SimpleVLA-RL 的分工与落地路径。
- **[Sim-to-Real 迁移策略 (Sim-to-Real Transfer)](./sim_to_real_transfer_strategies.md)**: Domain Randomization 与 Reality Gap 应对策略。📕 [社区 Sim2Real 真实根因](./community_field_notes_xiaohongshu.md#31-sim2real-gap-的真实根因)
- **[末端执行器控制系统 (End-Effector Control)](./end_effector_control.md)**: 数据驱动与触觉闭环控制软件架构设计。
- **[StarVLA：Lego-like VLA 研发底座](./starvla_lego_like_vla_codebase_2026.md)**: FAST/OFT/Flow/双系统框架对比。

---

## 学习建议
- **刚入门**: 先看 **[📕 社区笔记速查表](./community_field_notes_xiaohongshu.md#速查你现在卡在哪)**，找到你的问题，再跳到对应专题文件。
- **硬件党**: **[硬件选型](./robot_hardware_selection_pricing.md)** + **[📕 社区机械臂避坑](./community_field_notes_xiaohongshu.md#22-硬件相关陷阱)**。
- **工程党**: **[VLA 边缘部署](./vla_model_edge_deployment.md)** + **[📕 社区推理加速实测](./community_field_notes_xiaohongshu.md#93-边缘部署与模型压缩)** + **[📘 英文社区推理延迟全景](./community_field_notes_english.md)**。
- **算法党**: **[Sim-to-Real 策略](./sim_to_real_transfer_strategies.md)** + **[📕 社区 Sim2Real 失败根因](./community_field_notes_xiaohongshu.md#31-sim2real-gap-的真实根因)**。
- **英文读者**: 直接看 **[📘 English Community Field Notes](./community_field_notes_english.md)** — Blog 深度拆解 + Discord 原文，不需要翻译。
