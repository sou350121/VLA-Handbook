# VLA Handbook（Vision-Language-Action）

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

> **VLA（Vision-Language-Action）领域的结构化知识库与工程实战手册。**
> 覆盖理论基础、模型解析、真机部署、论文索引与题库。

---

<details open>
<summary><b>🚀 建议阅读路线 (Suggested Reading Path)</b></summary>

### 新手入门
1. [Theory 总索引](./theory/README.md) → **Part 1: Foundations**（数据格式、动作空间、评估体系）
2. **Part 2: Architecture & Algorithms**（VLA 核心架构、Diffusion Policy、Flow Matching）
3. [真机部署索引](./deployment/README.md)（硬件选型、模型优化）

### 研究导向
1. [论文索引](./theory/paper_index.md) + [文献综述](./theory/literature_review.md)（快速定位相关论文）
2. [Theory 总索引](./theory/README.md) → **Part 5: Model Zoo**（π0、GR-RL、WALL-OSS 深度解析）
3. [VLA 十大挑战](./theory/vla_challenges.md)（NTU/Stanford 2025 研究方向）

### 工程落地
1. [真机部署索引](./deployment/README.md)（UR5 控制、ROS、优化）
2. [Theory 总索引](./theory/README.md) → **效率优化**（Flash Attention、LoRA、量化）
3. [题库与实战](./question-bank/README.md)（代码实战、微调指南）

> 💡 **详细路线**：查看 [Theory 总索引](./theory/README.md) 获取完整学习路径

</details>

---

## 🎯 核心入口

| 模块 | 链接 | 说明 |
|:-----|:-----|:-----|
| **📚 Theory 总索引** | [`theory/README.md`](./theory/README.md) | 理论基础、核心算法、前沿架构 |
| **🔍 论文索引** | [`theory/paper_index.md`](./theory/paper_index.md) | 多维度查找（技术/公司/时间） |
| **📖 文献综述** | [`theory/literature_review.md`](./theory/literature_review.md) | VLA 发展史全景图（按技术分类） |
| **🚀 真机部署** | [`deployment/README.md`](./deployment/README.md) | 硬件选型、ROS、优化、Sim-to-Real |
| **💡 题库与实战** | [`question-bank/README.md`](./question-bank/README.md) | 面试真题、代码实战、微调指南 |
| **📋 速查表** | [`cheat-sheet/README.md`](./cheat-sheet/README.md) | 时间线、核心公式 |
| **📘 电子书** | [`book/README.md`](./book/README.md) | 合并版 Markdown/PDF/HTML |

---

## 🧠 Theory 快速推荐

> **优先阅读**：以下文档覆盖 VLA 核心概念与最新进展

| 主题 | 文档 | 一句话总结 |
|:-----|:-----|:---------|
| **架构总览** | [`vla_arch.md`](./theory/vla_arch.md) | VLM Backbone + Action Head 设计范式 |
| **动作生成** | [`diffusion_policy.md`](./theory/diffusion_policy.md) | 扩散去噪，解决多模态分布 |
| | [`pi0_flow_matching.md`](./theory/pi0_flow_matching.md) | Flow Matching（比 Diffusion 快 5x） |
| | [`act.md`](./theory/act.md) | CVAE + 动作分块，ALOHA 核心 |
| **效率优化** | [`flash_attention.md`](./theory/flash_attention.md) | Tiling + 重计算，显存 O(N²)→O(N) |
| | [`peft_lora.md`](./theory/peft_lora.md) | 低秩分解，QLoRA ~6GB 微调 7B |
| **前沿模型** | [`pi0_6_dissection.md`](./theory/pi0_6_dissection.md) | Recap 自我进化 + Action Expert |
| | [`gr_rl_dissection.md`](./theory/gr_rl_dissection.md) | ByteDance 三阶段 RL，真机穿鞋带 |
| **导航专题** | [`vln_dualvln.md`](./theory/vln_dualvln.md) | DualVLN：慢规划/快执行的异步双系统 |

> 💡 **更多推荐**：查看 [Theory 总索引](./theory/README.md) 获取完整学习路线图

---

<details>
<summary><b>✨ 为什么值得看（知识库价值）</b></summary>

1. **模型谱系完整**：覆盖 RT-2 → OpenVLA → π0 → π0.6 → GR-RL → WALL-OSS 等主流模型
2. **数学第一性原理**：15+ 篇核心文档包含 "Main Mathematical Idea" 章节（RL、Diffusion、Flow Matching、LoRA、Flash Attention 等）
3. **真机部署踩坑**：UR5 控制、ROS 集成、Python 性能优化、Protective Stop 恢复等实战经验
4. **论文索引系统**：多维度查找（技术/公司/时间），按分类组织的文献综述
5. **2025 最新进展**：Evo-1、SmolVLA、DualVLN、GR-RL、NeurIPS 2025 解读等
6. **全中文 + 工程导向**：专业术语保留英文对照，聚焦 Robotics 特有挑战

</details>

---

<details>
<summary><b>📂 项目结构</b></summary>

### 顶层目录

```
VLA-Handbook/
├── theory/          # 理论基础（核心）
├── deployment/      # 真机与部署
├── book/            # 电子书版本
├── cheat-sheet/     # 速查表
├── question-bank/   # 题库与实战
├── product/         # 机器人产品大百科
├── system-design/   # 系统设计
└── companies/       # 机器人公司与求职
```

### 完整目录树

<details>
<summary>展开完整目录树</summary>

```
VLA-Handbook/
├── README.md                   # 项目主页
├── theory/                     # 理论基础
│   ├── README.md               # 索引
│   ├── README_FUN.md           # 人话版索引
│   ├── paper_index.md          # 论文索引（多维度查找）
│   ├── literature_review.md     # 文献综述（按技术分类）
│   ├── vla_arch.md             # VLA 核心架构
│   ├── diffusion_policy.md     # 扩散策略详解
│   ├── pi0_flow_matching.md    # Flow Matching（π0 核心）
│   ├── act.md                  # ACT（CVAE + 动作分块）
│   ├── flash_attention.md      # Flash Attention
│   ├── peft_lora.md            # LoRA/QLoRA 原理
│   ├── vln_dualvln.md          # 视觉语言导航（VLN）
│   └── ...                     # 更多文档见 theory/README.md
├── deployment/                 # 真机与部署
│   ├── README.md               # 索引
│   ├── ur5_control_guide.md    # UR5 Python 控制实战
│   ├── ros_and_optimization.md # ROS 集成与性能优化
│   └── ...                     # 更多文档见 deployment/README.md
├── book/                       # 电子书版本
│   ├── README.md               # 索引
│   └── output/                 # 合并版输出（Markdown/PDF/HTML）
├── cheat-sheet/                # 速查表
├── question-bank/              # 题库与实战
├── product/                    # 机器人产品大百科
├── system-design/              # 系统设计
└── companies/                  # 机器人公司与求职
```

</details>

</details>

---

<details>
<summary><b>🛠️ VLA 开发必备知识</b></summary>

### 数据格式

| 格式 | 框架 | 优势 | 使用场景 |
| :--- | :--- | :--- | :--- |
| **LeRobot** (推荐) | PyTorch | Transformers 生态集成 | OpenVLA, WALL-OSS, Galaxea G0 |
| **RLDS** | TensorFlow | Open X-Embodiment 标准 | RT-1, RT-2, Octo |
| **HDF5 / NPZ** | 通用 | 跨平台，读写快 | 自定义数据集 |

### 仿真环境

| 平台 | 速度 | 适用场景 | 文档 |
| :--- | :--- | :--- | :--- |
| **Isaac Lab** (推荐) | 极快 | 大规模训练，GPU 加速 | [GitHub](https://github.com/NVIDIA-Omniverse/Isaac-Lab) |
| **MuJoCo** | 极快 | 快速迭代，算法验证 | [Docs](https://mujoco.readthedocs.io/) |
| **Isaac Sim** | 快 | 高保真渲染，Sim-to-Real | [Docs](https://docs.omniverse.nvidia.com/apps/isaacsim/latest/) |
| **SAPIEN** | 中等 | 抓取算法，复杂操作 | [GitHub](https://github.com/haosulab/SAPIEN) |
| **PyBullet** | 中等 | 学术研究，教学 | [Docs](https://pybullet.org/) |
| **Gazebo** | 慢 | ROS 集成，移动机器人 | [Tutorial](http://gazebosim.org/tutorials) |

### 深度学习框架

| 类别 | 工具 | 说明 |
| :--- | :--- | :--- |
| **训练** | PyTorch (主流), JAX (Pi0/Google) | 动态图，生态丰富 |
| **部署** | TensorRT, ONNX Runtime, vLLM | GPU 优化，大模型服务 |
| **分布式** | PyTorch FSDP, DeepSpeed | 大模型训练，显存优化 |
| **量化** | bitsandbytes, AWQ, GPTQ | QLoRA 训练，推理加速 |
| **优化** | Flash Attention, torch.compile, KV-Cache | 内存优化，编译加速 |

### RL 框架

| 框架 | 定位 | 适用场景 |
| :--- | :--- | :--- |
| **Stable Baselines3** | 易用、稳定 | 快速实验、教学 |
| **RLlib (Ray)** | 分布式、可扩展 | 大规模训练 |
| **SKRL** | Isaac Lab 集成 | 机器人 RL |
| **CleanRL** | 单文件实现 | 学习、研究 |
| **TorchRL** | PyTorch 官方 | 生产级应用 |

### 机器人控制

| 方法 | 原理 | 适用场景 |
| :--- | :--- | :--- |
| **PID** | 误差反馈 | 底层关节控制 |
| **阻抗控制** | 弹簧-阻尼行为 | 接触任务、人机协作 |
| **MPC** | 滚动优化 | 轨迹优化、避障 |
| **Computed Torque** | 动力学补偿 | 高精度任务 |

### 机器人中间件

| 工具 | 定位 | 说明 |
| :--- | :--- | :--- |
| **ROS 2** | 工业标准 | Python/C++ API，硬件抽象 |
| **Isaac Lab** | NVIDIA 框架 | 仿真-真机，GPU 加速 |
| **LeRobot** | VLA 工具链 | Hugging Face 端到端开发 |
| **MoveIt 2** | 运动规划 | 路径规划，碰撞检测 |

### 硬件控制接口

| 硬件 | 通信协议 | 常用工具/SDK |
| :--- | :--- | :--- |
| **灵巧手** | CAN Bus, USB, EtherCAT | Shadow/Inspire/Unitree SDK |
| **机械臂** | EtherCAT, TCP/IP | ROS MoveIt, Franka/UR SDK |
| **移动底盘** | CAN, Serial | ROS Navigation Stack |
| **传感器** | USB, Ethernet | ROS cv_bridge, PCL |

### Vision Language Models (VLM) - VLA 训练参考

> **最后更新**: 2025年12月5日

| 模型 | 参数量 | 开源 | VLA 应用案例 | 适用性 | HuggingFace |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PaliGemma 3B** | 3B | ✅ Apache 2.0 | π0, OpenVLA | ⭐⭐⭐⭐⭐ 最常用 | [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224) |
| **SigLIP** | 400M-2.6B | ✅ Apache 2.0 | OpenVLA, RDT (Vision Encoder) | ⭐⭐⭐⭐⭐ VLA 首选视觉编码器 | [google/siglip-*](https://huggingface.co/models?search=siglip) |
| **LLaVA 1.5/1.6** | 7B/13B | ✅ Apache 2.0 | OpenVLA (Llama 2 + SigLIP) | ⭐⭐⭐⭐ 成熟稳定 | [llava-hf/llava-1.5-*](https://huggingface.co/models?search=llava) |
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

### 学习资源

| 类型 | 链接 |
| :--- | :--- |
| **官方文档** | [PyTorch](https://pytorch.org/docs/) · [ROS 2](https://docs.ros.org/) · [Isaac Sim](https://docs.omniverse.nvidia.com/apps/isaacsim/latest/) |
| **开源项目** | [OpenVLA](https://github.com/openvla/openvla) · [LeRobot](https://github.com/huggingface/lerobot) · [Octo](https://github.com/octo-models/octo) · [Pi0](https://github.com/physint-ai/openpi) |
| **数据集** | [Open X-Embodiment](https://robotics-transformer-x.github.io/) · [RLDS](https://github.com/google-research/rlds) |
| **社区** | ROS Discourse · PyTorch Forums · Stack Overflow · GitHub Discussions |

</details>

---

<details>
<summary><b>📝 更新日志（最近更新）</b></summary>

### 2025-12-18 🆕
- **VLN 专题**: 新增 [`vln_dualvln.md`](./theory/vln_dualvln.md) - DualVLN 快慢系统（首个 VLN 基础模型）
- **首页优化**: 重构为研究型 landing page，Theory 优先，长内容折叠收纳

### 2025-12-16 🆕
- **数学第一性原理**: 为 15+ 篇核心理论文档增加了 "Main Mathematical Idea" 章节
- **真机部署实战**: 新增 UR5 Python 控制、ROS 集成与 Python 性能优化
- **Python OOP 实战**: 题库新增面向对象编程在机器人控制中的应用题

### 2025-12-13 🆕
- **NeurIPS 2025 最佳论文解读**: 新增 [`neurips_2025_insights.md`](./theory/neurips_2025_insights.md)
  - 6 篇获奖论文的具身智能视角解读
  - Artificial Hivemind、Gated Attention、1000 Layer Networks 等

<details>
<summary>查看更多历史更新</summary>

### 2025-12-08
- **小模型 VLA 研究**: 新增 [`small_vla_models.md`](./theory/small_vla_models.md)
  - Evo-1 (770M, LIBERO 94.8% SOTA)
  - SmolVLA (450M, 45-60Hz 推理)
  - 核心发现：210M SmolVLA 超越 55B RT-2-X
- **潜在动作学习**: 新增 Latent Action Learning 章节（UniVLA, EvoVLA, MemoryVLA）
- **论文索引系统**: [`paper_index.md`](./theory/paper_index.md) - 多维度索引（技术/公司/时间）
- **ByteDance GR-RL**: [`gr_rl_dissection.md`](./theory/gr_rl_dissection.md) - 三阶段 RL 训练，首个真机穿鞋带 VLA (78% 成功率)

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

</details>

</details>

---

## 🤝 贡献 (Contributing)

欢迎提交 Issue 和 Pull Request！
- 补充最新的 VLA 论文解读
- 分享你的真机部署经验
- 提供更多面试真题

## 📄 许可证 (License)

MIT License
