# VLA Handbook（Vision-Language-Action）

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

> **VLA（Vision-Language-Action）领域的结构化知识库与工程实战手册。**
> 覆盖理论基础、模型解析、真机部署、论文索引与题库。

---

## 🚀 建议阅读路线 (Suggested Reading Path)

### 新手入门
1. [Theory 总索引](./theory/README.md) → **Part 1: Foundations**（数据格式、动作空间、评估体系、灵巧手机械学）
2. **Part 2: Architecture & Algorithms**（VLA 核心架构、Diffusion Policy、Flow Matching）
3. [真机部署索引](./deployment/README.md)（硬件选型、模型优化）

### 研究导向
1. [论文索引](./theory/paper_index.md) + [文献综述](./theory/literature_review.md)（快速定位相关论文）
2. [Theory 总索引](./theory/README.md) → **Part 5: Model Zoo**（π0、GR-RL、WALL-OSS 深度解析）
3. [VLA 十大挑战](./theory/vla_challenges.md)（NTU/Stanford 2025 研究方向）

### 工程落地
1. [真机部署索引](./deployment/README.md)（硬件选型、UR5 控制、ROS、多模态同步）
2. [Theory 总索引](./theory/README.md) → **效率优化**（Flash Attention、LoRA、量化）
3. [题库与实战](./question-bank/README.md)（代码实战、微调指南）

> 💡 **详细路线**：查看 [Theory 总索引](./theory/README.md) 获取完整学习路径

---

## 📂 项目结构

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
│   ├── dexterous_hand_mechanics.md # 🆕 灵巧手机械学深度解析
│   ├── math_for_vla.md         # VLA 必备数学基础
│   ├── vla_arch.md             # VLA 核心架构
│   ├── pi0_flow_matching.md    # Flow Matching（π0 核心）
│   ├── pi0_code_analysis.md    # π0 源码导读
│   ├── tactile_vla.md          # 触觉 VLA 与 SaTA 专题
│   └── ...                     # 更多文档见 theory/README.md
├── deployment/                 # 真机与部署
│   ├── README.md               # 索引
│   ├── robot_hardware_selection_pricing.md # 🆕 硬件选型与前沿流派对比
│   ├── embodied_data_collection_overview.md # 🆕 具身数据采集概览 (POV/Sim2Real/RL)
│   ├── multimodal_data_synchronization.md # 🆕 多模态数据同步技术
│   ├── dexterous_hand_wuji.md  # 舞肌手 (Wuji) 深度解析
│   ├── dexterous_hand_applications.md # 灵巧手实战案例集 (VisionOS)
│   └── ...                     # 更多文档见 deployment/README.md
├── book/                       # 电子书版本
├── question-bank/              # 题库与实战
└── companies/                  # 机器人公司与求职
```

</details>

---

## 🎯 核心入口

| 模块 | 链接 | 说明 |
|:-----|:-----|:-----|
| **📚 Theory 总索引** | [`theory/README.md`](./theory/README.md) | 理论基础、核心算法、前沿架构 |
| **🔍 论文索引** | [`theory/paper_index.md`](./theory/paper_index.md) | 多维度查找（技术/公司/时间） |
| **📖 文献综述** | [`theory/literature_review.md`](./theory/literature_review.md) | VLA 发展史全景图（按技术分类） |
| **🚀 真机部署** | [`deployment/README.md`](./deployment/README.md) | 硬件选型、多模态同步、Sim-to-Real |
| **💡 题库与实战** | [`question-bank/README.md`](./question-bank/README.md) | 面试真题、代码实战、微调指南 |
| **📋 速查表** | [`cheat-sheet/README.md`](./cheat-sheet/README.md) | 时间线、核心公式 |
| **📝 更新日志** | [`CHANGELOG.md`](./CHANGELOG.md) | 🆕 每日详细更新记录 |

---

## 🧠 Theory 快速推荐

| 主题 | 文档 | 一句话总结 |
|:-----|:-----|:---------|
| **机械与硬件** | [`dexterous_hand_mechanics.md`](./theory/dexterous_hand_mechanics.md) | 🆕 Grubler 公式、雅可比对偶性与阻抗控制数学基础 |
| | [`robot_hardware_selection_pricing.md`](./deployment/robot_hardware_selection_pricing.md) | 🆕 直驱 vs 绳驱 vs 液压流派对比与典型操纵难点解析 |
| **前沿模型** | [`pi0_5_dissection.md`](./theory/pi0_5_dissection.md) | π0.5 开放世界泛化，分层推理机制 |
| | [`pi0_6_dissection.md`](./theory/pi0_6_dissection.md) | π0.6 Recap 自我进化 + Action Expert |
| | [`tactile_vla.md`](./theory/tactile_vla.md) | 🆕 触觉反馈 VLA、DTA 动态触觉阵列与 SaTA 研究 |
| **动作生成** | [`pi0_flow_matching.md`](./theory/pi0_flow_matching.md) | Flow Matching（比 Diffusion 快 5x，π0 核心） |
| | [`diffusion_policy.md`](./theory/diffusion_policy.md) | 扩散去噪，解决多模态分布 |
| **效率优化** | [`flash_attention.md`](./theory/flash_attention.md) | Tiling + 重计算，显存 O(N²)→O(N) |
| | [`peft_lora.md`](./theory/peft_lora.md) | 低秩分解，QLoRA ~6GB 微调 7B |

---

<details>
<summary><b>✨ 为什么值得看（知识库价值）</b></summary>

1. **硬件-模型全链路**：不仅讲 π0，还讲如何选灵巧手、如何解决 1000Hz 传感器同步。
2. **硬核数学推导**：包含雅可比矩阵、阻抗控制、Flow Matching 等核心数学第一性原理。
3. **2026 前沿视野**：同步 2026 年 1 月最新硬件（Sharpa Wave, LEAP V2 Adv）与研究（SaTA）。
4. **全中文 + 工程导向**：专业术语保留英文对照，聚焦 Robotics 特有挑战（如 Hysteresis、Backlash）。

</details>

---

<details>
<summary><b>🛠️ VLA 开发必备知识</b></summary>

### 机器人控制
| 方法 | 原理 | 适用场景 |
| :--- | :--- | :--- |
| **PID** | 误差反馈 | 底层关节控制 |
| **阻抗控制** | 弹簧-阻尼行为 | 接触任务、柔顺抓取、人机协作 |
| **前馈控制** | 动力学补偿 | 高频响应、抵消重力/摩擦力 |
| **MPC** | 滚动优化 | 轨迹优化、避障 |

### 硬件控制接口
| 硬件 | 通信协议 | 代表品牌 | 流派 |
| :--- | :--- | :--- | :--- |
| **灵巧手** | CAN/EtherCAT | Wuji, RealerHand, Sharpa | **电机直驱派** (高透明度) |
| | Tendon-driven | LEAP Hand, Shadow Hand | **绳驱线控派** (物理柔顺) |
| | Hydraulic | Sanctuary AI Phoenix | **液压重载派** (极致力量) |
| **机械臂** | EtherCAT, TCP/IP | UR, Franka, AgileX | 工业级 / 具身协作级 |

### Vision Language Models (VLM) - VLA 训练参考
> **最后更新**: 2026年1月12日

| 模型 | 参数量 | 优势 | HuggingFace |
| :--- | :--- | :--- | :--- |
| **Qwen2.5-VL** 🆕 | 3B-72B | **2025 SOTA**，多分辨率/长视频支持 | [Qwen/Qwen2.5-VL](https://huggingface.co/Qwen) |
| **PaliGemma 3B** | 3B | π0, OpenVLA 首选 Backbone | [google/paligemma-3b](https://huggingface.co/google) |
| **SigLIP** | 400M-2.6B | VLA 首选视觉编码器 | [google/siglip](https://huggingface.co/google) |

</details>

---

<details open>
<summary><b>📝 更新日志（最近更新）</b></summary>

### 2026-01-12 🆕
- **前沿灵巧手硬件专题**:
    - 新增 **Sharpa Wave** (22-DOF, DTA触觉) 与 **LEAP Hand V2 Adv** (可折叠手掌) 深度调研。
    - 新增 **RealerHand (睿尔灵)** 硬件选型参考。
    - 深度对比 **直驱 vs 绳驱 vs 液压** 三大传动流派对 VLA 学习的影响。
- **机械学专题**:
    - 新增 [`dexterous_hand_mechanics.md`](./theory/dexterous_hand_mechanics.md)，涵盖 Grubler 公式、雅可比对偶性、阻抗控制数学推导。
- **操控实战分析**:
    - 补充“开可乐（杠杆原理与指甲利用）”与“抓手机（环境物理对抗）”等工程细节。

### 2026-01-06
- **实战案例：手势控制灵巧手**: 新增 [`mediapipe_wujihand_project.md`](./deployment/mediapipe_wujihand_project.md)。

</details>

---

## 🤝 贡献 (Contributing)

欢迎提交 Issue 和 Pull Request！
- 补充最新的 VLA 论文解读 / 真机部署经验 / 面试真题。

## 📄 许可证 (License)

MIT License
