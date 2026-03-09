# 真机与部署 (Real-world & Deployment)

本模块关注 VLA 算法在真实物理世界中的落地与应用，涵盖从硬件选型、感知对齐到大规模数据采集的完整工程链路。

---

## 目录

### 1. 硬件选型与通讯架构 (Hardware & Infrastructure)
- **[硬件选型与成本 (Hardware & Pricing)](./robot_hardware_selection_pricing.md)**: 包含 **Sharpa Wave**, **LEAP V2**, **Wuji** 等主流灵巧手/机械臂/传感器对比。
- **[可复现人形开源部署：Roboto_Origin](./roboto_origin_reproducible_humanoid_deployment.md)**: 从装配→ROS2 bring-up→仿真训练→Sim2Real→测试矩阵的落地清单。
- **[VLA 模型边缘部署优化 (VLA Edge Deployment)](./vla_model_edge_deployment.md)**: 量化 (GPTQ, AWQ) 与边缘推理 (TensorRT-LLM, vLLM)。
- **[ROS 集成与算法优化 (ROS & Optimization)](./ros_and_optimization.md)**: ROS2 零拷贝、组件容器与 DDS 分布式调优。
- **[Agent 架构真机部署攻略 (Agentic VLA Deployment)](./agent_architecture_deployment_guide.md)**: 把 VLA 变成“能稳定做完任务”的系统：Planner/Policy/RT 控制分层、延迟预算、安全与恢复。

### 2. 感知、标定与多模态同步 (Sensing, Calibration & Sync)
- **[Perception 总索引](./perception/README.md)**: 感知系统工程（传感器选型、标定、同步、数据质量与评估）的主入口与跨文档索引。
- **[相机标定与手眼对齐 (Camera Calibration)](./camera_calibration_eye_in_hand.md)**: Eye-in-Hand vs Eye-to-Hand 标定实战。
- **[多模态数据同步技术 (Multimodal Sync)](./multimodal_data_synchronization.md)**: 解决 RGB-D 与高频控制（1000Hz）的时间对齐难题。
- **[触觉集成挑战 (Tactile Integration)](./tactile_sensor_integration_challenges.md)**: 触觉传感器与夹爪集成的工程难点。
- **[运动传感器测量差异与 IEEE P3716 质量评估标准](./ieee_p3716_sports_tracking_quality_standard.md)**: 场地运动中人体/物体追踪测量一致性难题与标准化进展（对多传感器融合与评估的启示）。

### 3. 机械臂控制与遥操作部署 (Robot Arm & Teleoperation)
- **[UR5 Python 控制实战 (UR5 Control Guide)](./ur5_control_guide.md)**: 实时内核配置、`ur_rtde` 高频控制与保护性停止恢复。
- **[GELLO 遥操作部署 (GELLO Deployment)](./gello_deployment.md)**: 低成本 3D 打印遥操作手柄配置与 LeRobot 格式转换。
- **[手势控制灵巧手：MediaPipe + WujiHand 实战项目](./mediapipe_wujihand_project.md)**: 感知→映射→控制的闭环工程骨架与低延迟优化抓手。
- **[Pi0 真机部署 (Pi0 Deployment)](./pi0_deployment.md)**: 官方 OpenPI 架构、Remote Inference 与硬件要求。

### 4. 灵巧手深度专题 (Dexterous Hand Deep Dive)
- **[灵巧手通讯与部署实战 (DexHand Communication)](./dexterous_hand_communication_deployment.md)**: 通讯架构 (CANFD, EtherCAT)、Retargeting 与线缆管理。
- **[灵巧手实战案例集 (DexHand Applications)](./dexterous_hand_applications.md)**: VisionOS 遥操作、跨设备动作映射与 Sim2Real 案例。
- **[Wuji 灵巧手深度解析 (Wuji Hand Deep Dive)](./dexterous_hand_wuji.md)**: 20-DOF 非拉索、全电机集成驱动技术方案。
- **[DexRobot DexHand021 量产版深度解析 (DexHand021 Production)](./dexrobot_dexhand021_production_dexhand_2026.md)**: 🆕 19-DOF（12 主动+7 被动）+ CANFD + ROS1/ROS2/micro-ROS + 多模态触觉（含滑觉/接近）落地拆解与对接清单。
- **[Optimus Hand V2 解析](./optimus_hand_v2.md)**: Tesla Optimus 灵巧手技术特点分析。

### 5. 仿真、数据采集与 Sim2Real (Data, Sim & Training)
- **[具身智能数据采集概览 (Embodied Data Collection)](./embodied_data_collection_overview.md)**: POV 第一视角 (EgoScale)、Sim2Real 规模化与真机 RL。
- **[Evo-RL 仓库部署审计 (Evo-RL Repo Analysis)](./evo_rl_repo_analysis.md)**: GPT 5.4 High 审计：可运行性、稳定性、隐含假设、未知项与真机复现门槛。
- **[灵巧手数据采集方案 (DexHand Data Collection)](./dexterous_hand_data_collection.md)**: 结构化 Episode 定义、Retargeting 算法与数据回放验证。
- **[仿真环境详解 (Simulation Environments)](./simulation_environments.md)**: Isaac Sim vs MuJoCo vs PyBullet 选型指南。
- **[仿真基准与训练工具 (Sim Benchmarks & Tooling)](./simulation_benchmarks_and_tools.md)**: LIBERO / RLinf / SimpleVLA-RL 的分工、组合与落地路径。
- **[Sim-to-Real 迁移策略 (Sim-to-Real Transfer)](./sim_to_real_transfer_strategies.md)**: Domain Randomization 与 Reality Gap 应对策略。
- **[末端执行器控制系统 (End-Effector Control)](./end_effector_control.md)**: 数据驱动与触觉闭环控制软件架构设计。
- **[StarVLA：Lego-like VLA 研发底座（训练/评测/策略服务脚手架）](./starvla_lego_like_vla_codebase_2026.md)**: 在同一套 bench 上对比 FAST/OFT/Flow/双系统等框架（Framework/Dataloader/Trainer/Eval 解耦）。

---

## 学习建议
- **硬件党**: 直接看 **[硬件选型](./robot_hardware_selection_pricing.md)**，了解最新的灵巧手和机器人平台。
- **工程党**: 重点研读 **[VLA 边缘部署](./vla_model_edge_deployment.md)** 与 **[多模态同步](./multimodal_data_synchronization.md)**。
- **算法党**: **[Sim-to-Real 迁移策略](./sim_to_real_transfer_strategies.md)** 与 **[数据采集](./dexterous_hand_data_collection.md)** 是核心重点。
