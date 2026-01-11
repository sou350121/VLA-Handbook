# 真机与部署 (Real-world & Deployment)

本模块关注 VLA 算法在真实物理世界中的落地与应用，涵盖从硬件选型、感知对齐到大规模数据采集的完整工程链路。

---

## 目录

### 1. 硬件选型与通讯架构 (Hardware & Infrastructure)
- **[硬件选型与成本 (Hardware & Pricing)](./robot_hardware_selection_pricing.md)**: 灵巧手/机械臂/传感器参考价格表与选型对比。
- **[VLA 模型边缘部署优化 (VLA Edge Deployment)](./vla_model_edge_deployment.md)**: 量化 (GPTQ, AWQ) 与边缘推理 (TensorRT-LLM, vLLM)。
- **[ROS 集成与算法优化 (ROS & Optimization)](./ros_and_optimization.md)**: ROS2 零拷贝、组件容器与 DDS 分布式调优。

### 2. 感知、标定与多模态同步 (Sensing, Calibration & Sync)
- **[相机标定与手眼对齐 (Camera Calibration)](./camera_calibration_eye_in_hand.md)**: Eye-in-Hand vs Eye-to-Hand 标定实战。
- **[多模态数据同步技术 (Multimodal Sync)](./multimodal_data_synchronization.md)**: 解决 RGB-D 与高频控制（1000Hz）的时间对齐难题。
- **[触觉集成挑战 (Tactile Integration)](./tactile_sensor_integration_challenges.md)**: 触觉传感器与夹爪集成的工程难点。

### 3. 机械臂控制与遥操作部署 (Robot Arm & Teleoperation)
- **[UR5 Python 控制实战 (UR5 Control Guide)](./ur5_control_guide.md)**: 实时内核配置、`ur_rtde` 高频控制与保护性停止恢复。
- **[GELLO 遥操作部署 (GELLO Deployment)](./gello_deployment.md)**: 低成本 3D 打印遥操作手柄配置与 LeRobot 格式转换。
- **[Pi0 真机部署 (Pi0 Deployment)](./pi0_deployment.md)**: 官方 OpenPI 架构、Remote Inference 与硬件要求。

### 4. 灵巧手深度专题 (Dexterous Hand Deep Dive)
- **[灵巧手通讯与部署实战 (DexHand Communication)](./dexterous_hand_communication_deployment.md)**: 通讯架构 (CANFD, EtherCAT)、Retargeting 与线缆管理。
- **[灵巧手实战案例集 (DexHand Applications)](./dexterous_hand_applications.md)**: VisionOS 遥操作、跨设备动作映射与 Sim2Real 案例。
- **[Wuji 灵巧手深度解析 (Wuji Hand Deep Dive)](./dexterous_hand_wuji.md)**: 20-DOF 非拉索、全电机集成驱动技术方案。
- **[Optimus Hand V2 解析](./optimus_hand_v2.md)**: Tesla Optimus 灵巧手技术特点分析。

### 5. 仿真、数据采集与 Sim2Real (Data, Sim & Training)
- **[具身智能数据采集概览 (Embodied Data Collection)](./embodied_data_collection_overview.md)**: POV 第一视角 (EgoScale)、Sim2Real 规模化与真机 RL。
- **[灵巧手数据采集方案 (DexHand Data Collection)](./dexterous_hand_data_collection.md)**: 结构化 Episode 定义、Retargeting 算法与数据回放验证。
- **[仿真环境详解 (Simulation Environments)](./simulation_environments.md)**: Isaac Sim vs MuJoCo vs PyBullet 选型指南。
- **[Sim-to-Real 迁移策略 (Sim-to-Real Transfer)](./sim_to_real_transfer_strategies.md)**: Domain Randomization 与 Reality Gap 应对策略。
- **[末端执行器控制系统 (End-Effector Control)](./end_effector_control.md)**: 数据驱动与触觉闭环控制软件架构设计。

---

## 学习建议
- **硬件党**: 直接看 **[硬件选型](./robot_hardware_selection_pricing.md)**，了解最新的灵巧手和机器人平台。
- **工程党**: 重点研读 **[VLA 边缘部署](./vla_model_edge_deployment.md)** 与 **[多模态同步](./multimodal_data_synchronization.md)**。
- **算法党**: **[Sim-to-Real 迁移策略](./sim_to_real_transfer_strategies.md)** 与 **[数据采集](./dexterous_hand_data_collection.md)** 是核心重点。
