# 真机与部署 (Real-world & Deployment)

本模块关注 VLA 算法在真实物理世界中的落地与应用，涵盖从硬件选型、感知对齐到大規模數據採集的完整工程鏈路。

---

## 目录

### 1. 硬件选型与通讯架构 (Hardware & Infrastructure)
- **[硬件选型与成本 (Hardware & Pricing)](./hardware.md)**: 灵巧手/机械臂/传感器参考价格表与选型对比。
- **[模型优化与边缘部署 (Optimization)](./optimization.md)**: 量化 (GPTQ, AWQ) 与边缘推理 (TensorRT-LLM, vLLM)。
- **[ROS 集成与算法优化 (ROS & Optimization)](./ros_and_optimization.md)**: ROS2 零拷贝、组件容器与 DDS 分布式调优。

### 2. 感知、标定与多模态同步 (Sensing, Calibration & Sync)
- **[相机标定 (Camera Calibration)](./calibration.md)**: Eye-in-Hand vs Eye-to-Hand 標定實戰。
- **[多模态数据同步 (Multimodal Sync)](./multimodal_sync.md)**: 解决 RGB-D 与高频控制（1000Hz）的时间对齐难题。
- **[Sensor Integration Challenges](./sensor_integration.md)**: 触觉传感器与夹爪集成的工程难点。

### 3. 机械臂控制与遥操作部署 (Robot Arm & Teleoperation)
- **[UR5 Python 控制实战 (UR5 Control Guide)](./ur5_control_guide.md)**: 实时内核配置、`ur_rtde` 高频控制与保护性停止恢复。
- **[GELLO 遥操作部署 (GELLO Deployment)](./gello_deployment.md)**: 低成本 3D 打印遥操作手柄配置与 LeRobot 格式转换。
- **[Pi0 真机部署 (Pi0 Deployment)](./pi0_deployment.md)**: 官方 OpenPI 架构、Remote Inference 与硬件要求。

### 4. 灵巧手深度专题 (Dexterous Hand Deep Dive)
- **[灵巧手部署实战 (Dexterous Hand Guide)](./dexterous_hand_guide.md)**: 通讯架构 (CANFD, EtherCAT)、Retargeting 與線纜管理。
- **[灵巧手实战案例集 (DexHand Applications)](./dexterous_hand_applications.md)**: VisionOS 遥操作、跨設備動作映射與 Sim2Real 案例。
- **[Wuji 灵巧手深度解析 (Wuji Hand Deep Dive)](./dexterous_hand_wuji.md)**: 20-DOF 非拉索、全電機集成驅動技術方案。
- **[Optimus Hand V2 解析](./optimus_hand_v2.md)**: Tesla Optimus 靈巧手技術特點分析。

### 5. 仿真、数据采集与 Sim2Real (Data, Sim & Training)
- **[具身智能数据采集与训练方案 (Data Collection & Training)](./data_collection_solutions.md)**: POV 第一视角 (EgoScale)、Sim2Real 規模化與真機 RL。
- **[灵巧手数据采集方案 (DexHand Data Collection)](./dexterous_hand_data_collection.md)**: 結構化 Episode 定義、Retargeting 算法與數據回放驗證。
- **[仿真环境详解 (Simulation Environments)](./simulation_environments.md)**: Isaac Sim vs MuJoCo vs PyBullet 选型指南。
- **[Sim-to-Real Guide](./sim_to_real.md)**: Domain Randomization 与 Reality Gap 应对策略。
- **[末端执行器控制系统 (End-Effector Control)](./end_effector_control.md)**: 数据驱动与触觉闭环控制軟件架構設計。

---

## 学习建议
- **硬件党**: 直接看 **[硬件选型](./hardware.md)**，了解最新的灵巧手和机器人平台。
- **工程党**: 重点研读 **[模型优化](./optimization.md)** 与 **[多模態同步](./multimodal_sync.md)**。
- **算法党**: **[Sim-to-Real Guide](./sim_to_real.md)** 与 **[數據採集](./dexterous_hand_data_collection.md)** 是核心重點。
