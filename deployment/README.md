# 真机与部署 (Real-world & Deployment)

本模块关注 VLA 算法在真实物理世界中的落地与应用。

## 目录
1. **[硬件选型与成本 (Hardware & Pricing)](./hardware.md)**
    - 灵巧手/机械臂/传感器 **参考价格表**
    - 选型对比 (参数 vs 价格)
2. **[相机标定 (Camera Calibration)](./calibration.md)** [New]
    - Eye-in-Hand vs Eye-to-Hand
    - Aruco 标定实战
3. **[Pi0 真机部署 (Pi0 Deployment)](./pi0_deployment.md)**
    - 官方 OpenPI 架构
    - 硬件要求 (4090 vs Orin)
    - Remote Inference 架构
4. **[灵巧手部署实战 (Dexterous Hand Guide)](./dexterous_hand_guide.md)**
    - 通讯架构 (CANFD, EtherCAT)
    - 软件栈 (Retargeting, Teleop)
    - 真实案例 (线缆管理, 散热)
5. **[模型优化与边缘部署 (Optimization)](./optimization.md)**
    - 量化 (Quantization): GPTQ, AWQ
    - 边缘推理: TensorRT-LLM, vLLM
6. **[仿真环境详解 (Simulation Environments)](./simulation_environments.md)** [New]
    - Isaac Sim vs MuJoCo vs PyBullet
    - 选型指南: 什么时候用什么？
7. **[Sim-to-Real Guide](./sim_to_real.md)**: 仿真到真机的迁移指南。
    - Domain Randomization
    - Reality Gap 应对策略
8. **[Sensor Integration Challenges](./sensor_integration.md)**: 触觉传感器与夹爪集成的五大工程难点。
9. **[末端执行器控制系统 (End-Effector Control)](./end_effector_control.md)**
    - 数据驱动与触觉闭环控制
    - 软件架构设计 (分层架构, 实时控制)
    - 数据采集与模型训练
    - 软件工程实践 (测试, CI/CD, 容器化)
10. **[GELLO 遥操作部署 (GELLO Deployment)](./gello_deployment.md)** 🆕
    - UR5 机械臂配置 (RTDE 通信)
    - Dynamixel 电机标定
    - 数据采集与 LeRobot 格式转换
    - 踩坑记录与最佳实践
11. **[UR5 Python 控制实战 (UR5 Control Guide)](./ur5_control_guide.md)** 🆕
    - Linux 环境配置 (Real-time kernel)
    - `ur_rtde` 高频控制代码范例
    - 保护性停止 (Protective Stop) 自动恢复
    - VLA 模型推理与控制线程架构
12. **[ROS 集成与算法优化 (ROS & Optimization)](./ros_and_optimization.md)** 🆕
    - ROS2 在新型机器人中的主导地位
    - DDS 通信性能与 QoS 实时调优
    - 功能安全认证 (Apex.AI) 与量产鸿沟
    - Python 性能优化 (Zero-Allocation, Numba JIT)

## 学习建议
- **硬件党**: 直接看 [硬件选型](./hardware.md)，了解最新的灵巧手和机器人平台。
- **工程党**: 重点研读 [模型优化](./optimization.md)，掌握如何在 Jetson 上跑大模型。
- **算法党**: [Sim-to-Real Guide](./sim_to_real.md) 是必考题，务必理解 Domain Randomization 的细节。
