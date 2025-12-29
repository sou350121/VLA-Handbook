# 机器人公司与求职指南

> 本目录整理了全球领先的机器人公司信息，帮助算法工程师了解行业格局、选择职业方向。

## 📂 目录结构

- **[中国头部机器人公司](./china.md)**: Unitree, Agibot, Fourier 等 15+ 家公司
- **[新创公司 (Startups)](./startups.md)**: 智在无界等具身智能新势力 🆕
- **[亚洲机器人公司](./asia.md)**: 新加坡、日本、台湾、韩国的领先公司
- **[国际机器人公司](./international.md)**: Tesla, Figure AI, Boston Dynamics 等
- **[具身智能软件与平台](./embodied_ai.md)**: Physical Intelligence, Covariant, Hugging Face 等

## 🎯 求职方向指南

### 算法岗位分类

#### 1. VLA/具身智能算法
**核心技术**: Vision-Language-Action 模型，端到端学习
**推荐公司**: 
- **国内**: Agibot, X Square (WALL-OSS), Galbot, Robot Era
- **国际**: Physical Intelligence, Figure AI, Covariant

**关键技能**:
- VLA 模型训练与部署 (OpenVLA, Pi0)
- Diffusion Policy / Flow Matching
- Sim-to-Real 技术
- 数据收集与标注

#### 2. 运动控制/强化学习
**核心技术**: Model-Based RL, Whole-Body Control, MPC
**推荐公司**:
- **国内**: Unitree, LimX Dynamics, Deep Robotics
- **国际**: Boston Dynamics, Agility Robotics

**关键技能**:
- 强化学习 (PPO, SAC, TD3)
- 运动规划 (MPC, Trajectory Optimization)
- 动力学建模与仿真

#### 3. 感知/SLAM
**核心技术**: 视觉SLAM, 多传感器融合, 3D重建
**推荐公司**:
- **国内**: Xiaomi, AgileX, Deep Robotics
- **国际**: 1X Technologies, Sanctuary AI

**关键技能**:
- ORB-SLAM, LSD-SLAM
- LiDAR/Vision Fusion
- Point Cloud Processing

#### 4. 灵巧手控制
**核心技术**: 高精度力控，触觉反馈
**推荐公司**:
- **国内**: Agibot (灵犀 X1), Fourier
- **国际**: Sanctuary AI (Phoenix), Figure AI

**关键技能**:
- 触觉传感器处理
- Impedance Control
- 精细操作策略

## 🚀 求职建议

### 技术栈准备
- **必备**: PyTorch, Python, ROS/ROS2
- **加分**: JAX/Flax (用于 VLA 训练), Isaac Sim, MuJoCo
- **开源贡献**: OpenVLA, LeRobot, Diffusion Policy

### 简历亮点
- 顶会论文 (CoRL, RSS, ICRA, IROS)
- 开源项目 (GitHub Star >100)
- 真机部署经验（非纯仿真）
- 竞赛获奖 (DARPA, RoboCup)

### 面试准备
1. **理论基础**: 参考 [理论基础](../theory/README.md)
2. **实战项目**: 参考 [题库](../question-bank/README.md)
3. **硬件知识**: 参考 [硬件选型](../deployment/hardware.md)

## 🌍 地区选择

### 中国 (国内机会)
**优势**: 行业爆发期，岗位多，语言无障碍  
**热门城市**: 北京、上海、深圳、杭州、苏州  
**薪资水平**: 具竞争力，头部公司股权激励丰厚

### 美国 (海外机会)
**优势**: 技术前沿，顶尖团队，学术氛围  
**热门城市**: Bay Area, Austin, Boston, Seattle  
**签证**: H1B需抽签，O-1需杰出成就

### 亚洲其他地区
**优势**: 区域领先，生活成本相对合理  
**推荐**: 新加坡 (创新), 日本 (工业), 韩国 (协作机器人)

## 📈 行业趋势

### 2024-2025 热点
- **人形机器人商业化**: Tesla Optimus, Figure AI 进工厂
- **VLA 模型爆发**: Pi0, OpenVLA, WALL-OSS 等开源
- **Software-First**: 软件公司估值超越硬件公司
- **中国弯道超车**: Unitree, Agibot 等快速崛起

### 技术方向
- **端到端学习**: VLA 取代传统 pipeline
- **Sim-to-Real**: Domain Randomization + 真机微调
- **多模态感知**: 视觉 + 触觉 + 本体感知
- **长时域任务**: 从单步操作到多步骤规划

---
[← Back to Main](../README.md)
