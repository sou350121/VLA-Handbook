# VLA (Vision-Language-Action) 算法岗面试手册

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

> **专注机器 VLA (Vision-Language-Action) 算法岗位的面试准备指南。**
> 从理论基础到真机部署，从灵巧手选型到 Sim-to-Real 实战。

## 📖 项目简介 (Introduction)

随着具身智能 (Embodied AI) 的爆发，VLA (Vision-Language-Action) 模型成为连接数字世界与物理世界的关键。本项目旨在为致力于进入该领域的算法工程师提供一份**全中文、实战导向**的面试与学习手册。

不同于通用的 CV/NLP 面试指南，本项目**聚焦于 Robotics 特有的挑战**：
- **Action Tokenization**: 如何将连续动作离散化？
- **Sim-to-Real**: 如何跨越仿真与真机的鸿沟？
- **Deployment**: 如何在边缘设备 (Jetson) 上部署大模型？
- **Hardware**: 灵巧手与机械臂的选型与控制。

## ✨ 项目亮点 (Highlights)

1. **全中文内容**: 所有文档均使用简体中文编写，专业术语保留英文对照。
2. **最新技术覆盖**:
    - 🆕 **2025 最新模型**: **Evo-1** (770M, LIBERO 94.8%), **SmolVLA** (450M, 60Hz), **ControlVLA** (10-shot 少样本)
    - 🆕 **潜在动作学习**: **UniVLA** (IJRR 2024), **EvoVLA**, **MemoryVLA** - 跨机器人泛化新范式
    - 🆕 **小模型 VLA 研究**: 边缘部署、模型压缩、知识蒸馏、210M 超越 55B 的惊人发现
    - 🆕 **ByteDance GR-RL**: 三阶段 RL 训练、形态对称增强、首个真机穿鞋带 VLA
    - 🆕 **论文索引系统**: 多维度查找（技术/公司/时间）、按分类组织的文献综述
    - 包含了 **Physical Intelligence (Pi)** 的 π0, π0.5, π0.6 模型深度解析
    - 涵盖了 **OpenVLA**, **WALL-OSS** (X Square), **Galaxea G0** (星海图) 等开源 SOTA 模型
    - 详解了 **FAST** 动作 Token 化（DCT + BPE，5倍训练加速）
    - 深入讲解 **Knowledge Insulation**（梯度隔离，防止灾难性遗忘）
    - 新增 **高效微调** (LoRA/QLoRA) 和 **量化理论** (AWQ, GPTQ) 详解
    - 新增 **空间数学** (坐标系转换, 旋转表示) 和 **评估体系** (Benchmarks, Metrics)
    - 新增 **[视觉感知技术](./theory/perception_techniques.md)** (检测/跟踪/Occupancy/BEV/位姿估计)
    - 新增 **[运动规划](./theory/motion_planning.md)** (RRT/PRM、TrajOpt、MoveIt)
    - 新增 **[状态估计与传感器融合](./theory/state_estimation.md)** (Kalman/UKF、Particle、VIO)
    - 新增 **[点云理解与 SLAM](./theory/pointcloud_slam.md)** (PointNet/KPConv、LOAM/LIO-SAM)
    - 新增 **[抓取算法与仿真平台](./theory/grasp_algorithms.md)** (DexGraspNet/GraspGF、Isaac Sim/SAPIEN)
3. **硬件选型指南**:
    - 重点加强了 **灵巧手 (Dexterous Hands)** 的介绍 (Shadow, Inspire, Unitree)。
    - 新增 **触觉传感器 (Tactile Sensors)** 深度解析 (GelSight, Tashan, Parsen)。
    - 提供了 **Unitree, Agibot, Fourier** 等中国头部机器人公司的详细参数与价格参考。
    - 新增 **国际机器人公司** 和 **亚洲机器人公司** 对比表。
    - 新增 **[传感器集成难点](./deployment/sensor_integration.md)** 专题。
    - 新增 **[末端执行器控制系统](./deployment/end_effector_control.md)** (数据驱动 + 触觉闭环)。
4. **实战导向**:
    - 提供了 **Sim-to-Real** 的具体技术路线 (Domain Randomization, Co-training)。
    - 提供了 **边缘部署** 的实战代码片段 (vLLM, Quantization)。
    - 新增 **仿真环境对比** (Isaac Sim vs MuJoCo vs PyBullet) 选型指南。
    - 新增 **相机标定** 实战指南 (Eye-in-Hand, Eye-to-Hand, Aruco)。
    - 新增 **[大规模模型训练](./system-design/large_scale_training.md)** (FSDP, 3D 并行, 训练稳定性)。

## 📂 项目结构 (Project Structure)

```
/opt/vla-interview-handbook/
├── README.md                   # 项目主页 (Introduction & Roadmap)
├── theory/                     # 理论基础
│   ├── README.md               # 索引
│   ├── README_FUN.md           # 人话版索引 (轻松理解版)
│   ├── paper_index.md          # 🆕 论文索引 (多维度查找: 技术/公司/时间)
│   ├── literature_review.md    # 🆕 核心文献技术归纳 (按技术分类组织)
│   ├── small_vla_models.md     # 🆕 小模型 VLA 研究 (Evo-1, SmolVLA, 边缘部署)
│   ├── vla_challenges.md       # 🆕 VLA 十大挑战 (NTU/Stanford 2025)
│   ├── vla_arch.md             # VLA 核心架构 (RT-1, RT-2, OpenVLA, Pi, WALL-OSS)
│   ├── transformer_vs_cnn.md   # Backbone 对比 (ViT vs ResNet, SigLIP)
│   ├── action_representations.md # 动作生成范式 (Tokenization vs Diffusion vs Flow)
│   ├── fast.md                 # FAST 动作 Token 化 (DCT + BPE, 5倍加速)
│   ├── diffusion_policy.md     # 扩散策略详解 (DDPM, DDIM, EBM)
│   ├── flash_attention.md      # 性能优化 (Kernel Fusion)
│   ├── pi0_flow_matching.md    # Pi0 代码解构 (Flow Matching)
│   ├── pi0_5_dissection.md     # Pi0.5 模型解剖 (Unified Model)
│   ├── pi0_6_dissection.md     # Pi0.6 模型解剖 (Recap RL)
│   ├── gr_rl_dissection.md     # 🆕 GR-RL 深度解析 (ByteDance Seed, 三阶段 RL)
│   ├── wall_oss.md             # WALL-OSS 深度解析 (Uni-CoT, X Square Robot)
│   ├── galaxea_g0.md           # Galaxea G0 双系统 VLA (星海图智能)
│   ├── knowledge_insulation.md # 知识绝缘技术 (防止灾难性遗忘)
│   ├── co_training.md          # 联合训练 (Co-training) 详解
│   ├── tactile_vla.md          # 触觉感知与 VLA
│   ├── motion_planning.md      # 运动规划 (RRT/TrajOpt/MoveIt)
│   ├── state_estimation.md     # 状态估计与融合 (Kalman/Particle/VIO)
│   ├── pointcloud_slam.md      # 点云理解 & SLAM (LOAM/LIO-SAM)
│   ├── grasp_algorithms.md     # 抓取算法 & 仿真 (DexGraspNet/Isaac)
│   ├── data.md                 # 数据处理 (RLDS, Co-training)
│   ├── spatial_math.md         # 空间数学 (坐标系, 旋转表示)
│   ├── evaluation.md           # 评估体系 (Benchmarks, Metrics)
│   ├── peft_lora.md            # 高效微调 (LoRA/QLoRA 原理)
│   └── quantization_theory.md  # 量化理论 (AWQ, GPTQ)
├── product/                    # 🆕 机器人产品大百科
│   ├── README.md               # 产品索引
│   ├── humanoids.md            # 具身智能本体 (Tesla, Unitree)
│   ├── hands.md                # 灵巧手 (Shadow, Inspire)
│   ├── arms.md                 # 科研机械臂 (Franka, UR)
│   ├── grippers.md             # 平行夹爪 (Robotiq, DH)
│   ├── mobile_bases.md         # 移动底盘 (AgileX)
│   └── sensors.md              # 触觉与感知 (GelSight, Tashan)
├── deployment/                 # 真机与部署
│   ├── README.md               # 索引
│   ├── hardware.md             # 硬件选型与价格参考
│   ├── sensor_integration.md   # 触觉传感器集成难点
│   ├── end_effector_control.md # 🆕 末端执行器控制系统
│   ├── calibration.md          # 相机标定指南
│   ├── pi0_deployment.md       # Pi0 真机部署
│   ├── dexterous_hand_guide.md # 灵巧手部署实战
│   ├── optimization.md         # 模型优化 (量化, TensorRT)
│   ├── simulation_environments.md # 仿真环境详解 (Isaac Sim, MuJoCo, PyBullet)
│   └── sim_to_real.md          # Sim-to-Real 技术
├── system-design/              # 系统设计
│   ├── README.md               # 索引
│   ├── data_pipeline.md        # 数据闭环设计
│   ├── cloud_infrastructure.md # 云端基础设施
│   ├── large_scale_training.md # 🆕 大规模模型训练
│   └── evaluation.md           # 评估系统设计
├── cheat-sheet/                # 速查表
│   ├── README.md               # 索引
│   ├── timeline.md             # 关键论文时间线
│   └── formulas.md             # 核心公式
├── question-bank/              # 题库与实战
│   ├── README.md               # 索引
│   ├── questions.md            # 面试真题
│   ├── openvla_finetuning.md   # OpenVLA 微调实战
│   └── interviewer_guide.md    # 考官视角指南
└── companies/                  # 🆕 机器人公司与求职
    ├── README.md               # 求职指南索引
    ├── china.md                # 中国机器人公司
    ├── international.md        # 国际机器人公司
    ├── asia.md                 # 亚洲机器人公司 (SG/JP/TW/KR)
    └── embodied_ai.md          # 具身智能软件平台
```

## 🚀 快速开始 (Getting Started)

### 📚 推荐学习路径

#### 学习者/应届生
1. **基础入门**: 先看 [理论基础 (Theory)](./theory/README.md) 的 **Part 1: Foundations**，理解数据格式、动作空间、评估体系。
2. **架构理解**: 深入学习 **Part 2: Architecture & Algorithms**，掌握 VLA 核心架构和生成策略。
3. **产品认知**: 浏览 [产品汇总 (Products)](./product/README.md)，了解主流机器人硬件参数。
4. **实战准备**: 学习 [真机部署 (Deployment)](./deployment/README.md)，掌握硬件选型和模型优化。
5. **求职规划**: 参考 [公司名录 (Companies)](./companies/README.md)，了解行业格局和岗位要求。

#### 在职转岗/跳槽者
1. **速查复习**: 先看 [速查表](./cheat-sheet/README.md) 快速回顾核心概念和公式。
2. **深度补充**: 针对性阅读 [理论基础](./theory/README.md) 中的薄弱环节
   - **最新技术**: FAST, Knowledge Insulation, LoRA/QLoRA, 量化理论
   - **前沿模型**: Galaxea G0, WALL-OSS 双系统架构对比
   - **评估体系**: 理解 Benchmarks 和 Metrics 的设计原理
3. **实战强化**: 重点学习 [真机部署](./deployment/README.md)
   - 仿真环境选型 (Isaac Sim vs MuJoCo)
   - 模型优化与边缘部署
   - Sim-to-Real 技术路线
4. **面试准备**: 刷 [题库](./question-bank/README.md) 模拟真实面试场景。
5. **目标公司**: 在 [公司目录](./companies/README.md) 中锁定意向公司和岗位方向。

#### 面试官/技术Leader
1. **题库设计**: 参考 [面试官视角](./question-bank/interviewer_guide.md)
2. **技术深度**: 查阅 [文献综述](./theory/literature_review.md) 了解前沿
3. **系统设计**: 学习 [系统设计](./system-design/README.md) 评估候选人架构能力

## 🛠️ VLA 开发必备知识 (Development Essentials)

### 数据格式 (Data Formats)

| 格式 | 框架 | 优势 | 使用场景 |
| :--- | :--- | :--- | :--- |
| **LeRobot** (推荐) | PyTorch | Transformers 生态集成 | OpenVLA, WALL-OSS, Galaxea G0 |
| **RLDS** | TensorFlow | Open X-Embodiment 标准 | RT-1, RT-2, Octo |
| **HDF5 / NPZ** | 通用 | 跨平台，读写快 | 自定义数据集 |

### 仿真环境 (Simulation)

| 平台 | 速度 | 适用场景 | 文档 |
| :--- | :--- | :--- | :--- |
| **Isaac Lab** (推荐) | 极快 | 大规模训练，GPU 加速 | [GitHub](https://github.com/NVIDIA-Omniverse/Isaac-Lab) |
| **MuJoCo** | 极快 | 快速迭代，算法验证 | [Docs](https://mujoco.readthedocs.io/) |
| **Isaac Sim** | 快 | 高保真渲染，Sim-to-Real | [Docs](https://docs.omniverse.nvidia.com/apps/isaacsim/latest/) |
| **SAPIEN** | 中等 | 抓取算法，复杂操作 | [GitHub](https://github.com/haosulab/SAPIEN) |
| **PyBullet** | 中等 | 学术研究，教学 | [Docs](https://pybullet.org/) |
| **Gazebo** | 慢 | ROS 集成，移动机器人 | [Tutorial](http://gazebosim.org/tutorials) |

### 深度学习框架 (DL Frameworks)

| 类别 | 工具 | 说明 |
| :--- | :--- | :--- |
| **训练** | PyTorch (主流), JAX (Pi0/Google) | 动态图，生态丰富 |
| **部署** | TensorRT, ONNX Runtime, vLLM | GPU 优化，大模型服务 |
| **分布式** | PyTorch FSDP, DeepSpeed | 大模型训练，显存优化 |
| **量化** | bitsandbytes, AWQ, GPTQ | QLoRA 训练，推理加速 |
| **优化** | Flash Attention, torch.compile, KV-Cache | 内存优化，编译加速 |

### RL 框架 (RL Frameworks)

| 框架 | 定位 | 适用场景 |
| :--- | :--- | :--- |
| **Stable Baselines3** | 易用、稳定 | 快速实验、教学 |
| **RLlib (Ray)** | 分布式、可扩展 | 大规模训练 |
| **SKRL** | Isaac Lab 集成 | 机器人 RL |
| **CleanRL** | 单文件实现 | 学习、研究 |
| **TorchRL** | PyTorch 官方 | 生产级应用 |

### 机器人控制 (Robot Control)

| 方法 | 原理 | 适用场景 |
| :--- | :--- | :--- |
| **PID** | 误差反馈 | 底层关节控制 |
| **阻抗控制** | 弹簧-阻尼行为 | 接触任务、人机协作 |
| **MPC** | 滚动优化 | 轨迹优化、避障 |
| **Computed Torque** | 动力学补偿 | 高精度任务 |

### 机器人中间件 (Robotics Middleware)

| 工具 | 定位 | 说明 |
| :--- | :--- | :--- |
| **ROS 2** | 工业标准 | Python/C++ API，硬件抽象 |
| **Isaac Lab** | NVIDIA 框架 | 仿真-真机，GPU 加速 |
| **LeRobot** | VLA 工具链 | Hugging Face 端到端开发 |
| **MoveIt 2** | 运动规划 | 路径规划，碰撞检测 |

### 硬件控制接口 (Hardware Control)

| 硬件 | 通信协议 | 常用工具/SDK |
| :--- | :--- | :--- |
| **灵巧手** | CAN Bus, USB, EtherCAT | Shadow/Inspire/Unitree SDK |
| **机械臂** | EtherCAT, TCP/IP | ROS MoveIt, Franka/UR SDK |
| **移动底盘** | CAN, Serial | ROS Navigation Stack |
| **传感器** | USB, Ethernet | ROS cv_bridge, PCL |

### 版本控制与实验管理 (Version Control & Experiment)

| 类别 | 工具 | 说明 |
| :--- | :--- | :--- |
| **代码** | Git + GitHub/GitLab | 分布式版本控制 |
| **大文件** | Git LFS, DVC | 模型权重，数据集 |
| **实验** | Weights & Biases (推荐), TensorBoard | 实验跟踪，可视化 |
| **模型** | MLflow, HuggingFace Hub | 模型注册，部署 |

### 开发环境 (Development Environment)

| 类别 | 工具 | 说明 |
| :--- | :--- | :--- |
| **Python** | Conda (推荐), venv | 环境隔离，依赖管理 |
| **容器** | Docker + NVIDIA Container | 环境复现，部署 |
| **GPU** | CUDA 11.8+, cuDNN, NCCL | PyTorch 兼容，分布式通信 |
| **IDE** | VS Code, PyCharm | 调试，远程开发 |

### 调试与性能分析 (Debug & Profiling)

| 类别 | 工具 | 说明 |
| :--- | :--- | :--- |
| **调试** | pdb, ipdb, VS Code Debugger | Python 调试 |
| **性能** | torch.profiler, NVIDIA Nsight | GPU 性能分析 |
| **内存** | nvidia-smi, memory_profiler | 显存/内存监控 |

### Vision Language Models (VLM) - VLA 训练参考

> **最后更新**: 2025年12月5日

| 模型 | 参数量 | 开源 | VLA 应用案例 | 适用性 | HuggingFace |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PaliGemma 3B** | 3B | ✅ Apache 2.0 | π0, OpenVLA | ⭐⭐⭐⭐⭐ 最常用 | [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224) |
| **SigLIP** | 400M-2.6B | ✅ Apache 2.0 | OpenVLA, RDT (Vision Encoder) | ⭐⭐⭐⭐⭐ VLA 首选视觉编码器 | [google/siglip-*](https://huggingface.co/models?search=siglip) |
| **LLaVA 1.5/1.6** | 7B/13B | ✅ Apache 2.0 | OpenVLA (Llama 2 + SigLIP) | ⭐⭐⭐⭐ 成熟稳定 | [llava-hf/llava-1.5-*](https://huggingface.co/models?search=llava) |
| **LLaVA-NeXT** | 7B/13B/34B | ✅ Apache 2.0 | - | ⭐⭐⭐⭐ 最新版本，性能提升 | [llava-hf/llava-next-*](https://huggingface.co/models?search=llava-next) |
| **Qwen2.5-VL** 🆕 | 3B/7B/32B/72B | ✅ Apache 2.0 | - | ⭐⭐⭐⭐⭐ **2025 SOTA**，中文首选 | [Qwen/Qwen2.5-VL-*](https://huggingface.co/models?search=Qwen2.5-VL) |
| **Eagle 2.5** 🆕 | 8B | ✅ Apache 2.0 | - | ⭐⭐⭐⭐ 长上下文多模态 | [nvidia/Eagle-*](https://huggingface.co/models?search=Eagle) |
| **Seed 1.5-VL** 🆕 | 20B | ✅ | - | ⭐⭐⭐⭐ GUI 交互强 | [ByteDance/Seed-*](https://huggingface.co/models?search=Seed) |
| **GLM-4.5V** 🆕 | 106B (12B 激活) | ✅ Apache 2.0 | - | ⭐⭐⭐⭐ 3D 空间推理 | [THUDM/GLM-4.5V](https://huggingface.co/models?search=GLM-4) |
| **Llama 4** 🆕 | MoE (16-128专家) | ✅ Meta Llama | - | ⭐⭐⭐⭐ 10M token 上下文 | [meta-llama/Llama-4](https://huggingface.co/models?search=llama-4) |
| **Qwen2-VL** | 2B/7B/72B | ✅ Apache 2.0 | - | ⭐⭐⭐⭐ 2024 版本 | [Qwen/Qwen2-VL-*](https://huggingface.co/models?search=Qwen2-VL) |
| **MiniCPM-V** | 2.4B | ✅ Apache 2.0 | - | ⭐⭐⭐ 超轻量级 | [openbmb/MiniCPM-V](https://huggingface.co/openbmb/MiniCPM-V) |
| **CogVLM** | 17B | ✅ Apache 2.0 | - | ⭐⭐⭐ 视觉理解强 | [THUDM/cogvlm-*](https://huggingface.co/models?search=cogvlm) |
| **InternVL** | 2B-26B | ✅ Apache 2.0 | - | ⭐⭐⭐ 多分辨率支持 | [OpenGVLab/InternVL-*](https://huggingface.co/models?search=InternVL) |
| **InternVL2** | 2B/4B/8B/26B | ✅ Apache 2.0 | - | ⭐⭐⭐⭐ 最新版本，多模态能力增强 | [OpenGVLab/InternVL2-*](https://huggingface.co/models?search=InternVL2) |
| **SmolVLA** | 450M | ✅ Apache 2.0 | - | ⭐⭐⭐ 超轻量级，研究用 | [huggingface/smolvla](https://huggingface.co/models?search=smolvla) |
| **PaLI-X** | 55B | ❌ | RT-2 | ⭐⭐ 闭源，难以部署 | - |

> **选择建议**: VLA 训练首选 **PaliGemma 3B**（轻量高效）或 **SigLIP**（作为 Vision Encoder）。中文任务推荐 **Qwen2.5-VL**（🆕 2025 SOTA）。详细对比见 [多模态模型基础](./theory/multimodal_models.md#56-主流-vlm-对比表vla-训练参考)。

### 🔥 π0 / OpenPI - 开源实用度最高的 VLA 模型

> **Physical Intelligence** 于 2025 年 2 月开源了 **π0 (Pi-Zero)** 系列模型，是目前**工程落地首选**的 VLA 方案。

| 特性 | 说明 |
| :--- | :--- |
| **GitHub** | [OpenPI](https://github.com/Physical-Intelligence/openpi) (⭐ 3.5k+) |
| **HuggingFace** | [physicalintelligence/pi0](https://huggingface.co/physicalintelligence) |
| **LeRobot 集成** | 直接通过 `lerobot` 库加载和微调 |
| **Backbone** | PaliGemma 3B (轻量高效) |
| **核心技术** | Flow Matching (比 Diffusion 快 5-10x) |
| **动作空间** | 连续 (无量化误差，精度高) |
| **推理速度** | 1-10 步 ODE Solver，支持高频控制 (50Hz) |
| **许可证** | Apache 2.0 (商业友好) |

**为什么 π0 是首选？**
1. **开源完整**: 模型权重 + 训练代码 + 数据处理全开源
2. **工程成熟**: Physical Intelligence 是 VLA 领域最强团队，代码质量高
3. **性能 SOTA**: Flow Matching 架构在精度和速度上优于 Diffusion Policy
4. **生态完善**: 与 LeRobot / HuggingFace 深度集成，开箱即用
5. **商业可用**: Apache 2.0 许可，可用于商业产品

**快速开始**:
```bash
# 安装
pip install lerobot

# 加载预训练模型
from lerobot.common.policies import Pi0Policy
policy = Pi0Policy.from_pretrained("physicalintelligence/pi0-base")

# 推理
action = policy.select_action(observation)
```

**深度学习资源**:
- [π0 Flow Matching 原理](./theory/pi0_flow_matching.md) - 核心算法详解
- [π0 代码解析](./theory/pi0_code_analysis.md) - OpenPI 源码导读
- [π0.5 模型解剖](./theory/pi0_5_dissection.md) - 开放世界泛化
- [π0.6 模型解剖](./theory/pi0_6_dissection.md) - Recap 自我进化

### 学习资源 (Resources)

| 类型 | 链接 |
| :--- | :--- |
| **官方文档** | [PyTorch](https://pytorch.org/docs/) · [ROS 2](https://docs.ros.org/) · [Isaac Sim](https://docs.omniverse.nvidia.com/apps/isaacsim/latest/) |
| **开源项目** | [OpenVLA](https://github.com/openvla/openvla) · [LeRobot](https://github.com/huggingface/lerobot) · [Octo](https://github.com/octo-models/octo) · [Pi0](https://github.com/physint-ai/openpi) |
| **数据集** | [Open X-Embodiment](https://robotics-transformer-x.github.io/) · [RLDS](https://github.com/google-research/rlds) |
| **社区** | ROS Discourse · PyTorch Forums · Stack Overflow · GitHub Discussions |

## 📝 更新日志 (Changelog)

### 2025-12-13 🆕
- **NeurIPS 2025 最佳论文解读**: 新增 [neurips_2025_insights.md](./theory/neurips_2025_insights.md)
  - 6 篇获奖论文的具身智能视角解读
  - Artificial Hivemind: 模型同质化与机器人行为多样性
  - Gated Attention: 门控注意力机制与边缘部署
  - 1000 Layer Networks: 深层自监督 RL
  - Diffusion Generalization: 扩散模型泛化机制
  - Superposition Scaling: 表示叠加与多技能统一模型
  - RL Reasoning Limits: RLVR 局限性分析
  - **未来发展方向**: 技术趋势、待解决问题、突破口预测

### 2025-12-08
- **小模型 VLA 研究**: 新增 [small_vla_models.md](./theory/small_vla_models.md)
  - Evo-1 (770M, LIBERO 94.8% SOTA)
  - SmolVLA (450M, 45-60Hz 推理)
  - ControlVLA (10-20 shot 少样本适配)
  - 核心发现：210M SmolVLA 超越 55B RT-2-X
- **潜在动作学习**: 新增 Latent Action Learning 章节
  - UniVLA (IJRR 2024): 从视频学习跨机器人动作
  - EvoVLA, MemoryVLA, TTF-VLA, OmniVLA, MergeVLA
- **论文索引系统**: [paper_index.md](./theory/paper_index.md)
  - 多维度索引（技术/公司/时间）
  - 按技术分类的文献综述重构
- **ByteDance GR-RL**: [gr_rl_dissection.md](./theory/gr_rl_dissection.md)
  - 三阶段 RL 训练、形态对称增强
  - 首个真机穿鞋带 VLA (78% 成功率)
- **VLA 十大挑战**: [vla_challenges.md](./theory/vla_challenges.md)
  - NTU/Stanford 2025 研究方向

### 2025-12-06
- 新增 VLA 总工程师研究方案系列
- 工程师 vs 生物学家三轮辩论
- 技术路线顾问细节研发方案

### 2025-12-05
- 新增多模态模型详解，包含 2025 最新 VLM
- π0.6 Action Expert 深度解析

### 2025-12-01
- 新增视觉感知、运动规划、SLAM 专题
- 新增传感器集成与末端执行器控制

## 🤝 贡献 (Contributing)

欢迎提交 Issue 和 Pull Request！
- 补充最新的 VLA 论文解读。
- 分享你的真机部署经验。
- 提供更多面试真题。

## 📄 许可证 (License)

MIT License
