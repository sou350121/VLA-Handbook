# GitHub 社区实战笔记

> 从 21 个 VLA/具身智能仓库的高互动 Issues 中蒸馏的工程知识。每条结论附 Issue 链接作为 evidence。
>
> **数据来源**: 2024-03 至 2026-03，累计扫描 ~200 条高互动 Issues（≥10 comments）
>
> **更新频率**: Pulsar GitHub Issues Sensor 每周自动扫描，人工季度审核

---

## 1. 硬件兼容性矩阵

### 1.1 GPU 选型

| GPU | 可用场景 | 已知限制 | Evidence |
|-----|---------|---------|----------|
| **H100/A100 80GB** | 全量微调+视觉编码器 | 最佳选择，无已知问题 | [GR00T #24](https://github.com/NVIDIA/Isaac-GR00T/issues/24) |
| **4090 24GB** | LoRA 微调，推理 | 全量微调 OOM；GR00T 需 `--no-tune-visual --batch-size 1` | [openpi #379](https://github.com/Physical-Intelligence/openpi/issues/379), [GR00T #24](https://github.com/NVIDIA/Isaac-GR00T/issues/24) |
| **3090 24GB** | LoRA 微调，推理 | 同 4090，bfloat16 支持 | [openpi #379](https://github.com/Physical-Intelligence/openpi/issues/379) |
| **RTX 5090/5070** | 需手动编译 | FlashAttention 不支持 SM120；需 flash-attn 2.8.3 源码编译 + CUDA 12.8 | [GR00T #309](https://github.com/NVIDIA/Isaac-GR00T/issues/309), [lerobot #2217](https://github.com/huggingface/lerobot/issues/2217) |
| **4090D (中国版)** | GR00T 微调 | 需 `model.bfloat16()` 仅对 diffusion+projector | [GR00T #24](https://github.com/NVIDIA/Isaac-GR00T/issues/24) |

**RTX 50 系列用户必读**: torch 2.7.1 不支持 SM120 架构。解决方案：宽松版本约束 `torch>=2.2.1,<=2.8.0` 并从 cu128 索引安装。([lerobot #2217](https://github.com/huggingface/lerobot/issues/2217), 10 人确认)

### 1.2 边缘部署（Jetson 系列）

| 设备 | Pi0/Pi0.5 | GR00T | 推理延迟 | Evidence |
|------|-----------|-------|---------|----------|
| **Jetson AGX Orin 64GB** | ✅ 可用 | ✅ 可用 | ~1.1s (JAX), ~300-500ms (Thor) | [openpi #386](https://github.com/Physical-Intelligence/openpi/issues/386) |
| **Jetson Orin NX 16GB** | ❌ OOM | 未测试 | — | [openpi #386](https://github.com/Physical-Intelligence/openpi/issues/386) |
| **Jetson Orin Nano 8GB** | ❌ OOM | ❌ OOM | — | [openpi #386](https://github.com/Physical-Intelligence/openpi/issues/386) |
| **Jetson AGX Thor** | ✅ ~300ms | ✅ 需特殊 Docker | 理论 4x 快于 Orin | [openpi #386](https://github.com/Physical-Intelligence/openpi/issues/386), [GR00T #343](https://github.com/NVIDIA/Isaac-GR00T/issues/343) |

**Jetson 部署要点**:
- JAX 统一内存预分配导致 OOM：设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false` ([openpi #386](https://github.com/Physical-Intelligence/openpi/issues/386))
- Thor Docker 镜像：`johnnync/isaac-gr00t:r38.2.arm64-sbsa-cu130-24.04` ([GR00T #411](https://github.com/NVIDIA/Isaac-GR00T/issues/411))
- pytorch3d 在 aarch64 无预编译 wheel，需源码编译或注释掉（SO101 场景可跳过）([GR00T #411](https://github.com/NVIDIA/Isaac-GR00T/issues/411))
- TensorRT FP8 不支持 Orin NX（硬件不支持 FP8 指令），NVFP4 缺少自定义 plugin ([GR00T #575](https://github.com/NVIDIA/Isaac-GR00T/issues/575))

### 1.3 IsaacLab 渲染

- **RTX 5090 Ubuntu 渲染模糊**：570.x 驱动在 RTX Real-Time 模式下产生像素化画面，Windows 正常。等待驱动修复。([IsaacLab #2188](https://github.com/isaac-sim/IsaacLab/issues/2188), 20 comments)
- **IsaacSim 版本升级可能破坏 URDF 导入**：新版 IsaacSim 改变了 URDF spawner 行为，需锁定版本。([IsaacLab #1760](https://github.com/isaac-sim/IsaacLab/issues/1760), 36 comments)

---

## 2. 训练配方与陷阱

### 2.1 Pi0/Pi0.5 微调

**收敛失败的三大根因**:

1. **Action 标准差为零** → 损失爆炸到百万级。数据采集时某些维度（如 roll/pitch）始终不变，导致归一化除以零。修复：替换零 std 为小 epsilon (2e-5 到 2e-4)。([openpi #814](https://github.com/Physical-Intelligence/openpi/issues/814), 22 comments)

2. **transformers 版本不兼容** → 权重静默加载失败（`strict=False` 不报错）。transformers ≥4.52.0 重命名了 PaliGemma 的 key (`language_model.model` → `model.language_model`)，导致预训练权重全丢。症状：loss 从 ~0.5 跳到 ~4.5。([lerobot #1406](https://github.com/huggingface/lerobot/issues/1406), 38 comments, 多人真机从 10%→80% SR)

3. **Gripper 归一化错误** → 机械臂行为异常但不报错。Agilex Cobot Magic 需注释掉 gripper normalize/unnormalize 函数。([openpi #405](https://github.com/Physical-Intelligence/openpi/issues/405), 32 comments)

**Pi0 LoRA 配方**:
- 4090 24GB 可用，需低内存配置 + 短 `max_token_len` ([openpi #379](https://github.com/Physical-Intelligence/openpi/issues/379))
- Pi0.5 LoRA：`Pi0Config(pi05=True)` + `paligemma_variant="gemma_2b_lora"` + `action_expert_variant="gemma_300m_lora"` ([openpi #672](https://github.com/Physical-Intelligence/openpi/issues/672))
- `extra_delta_transform`：delta EEF 数据集（LIBERO）设 False，绝对 EEF（DROID）设 True ([openpi #637](https://github.com/Physical-Intelligence/openpi/issues/637))

**Pi0.5 子任务生成**:
- 基础 checkpoint 生成乱码（VLA 微调降低了语言理解）；需额外微调 <100 步 ([openpi #701](https://github.com/Physical-Intelligence/openpi/issues/701))
- 隐藏状态提取位置错误导致前几个词丢失：应用 `jnp.max(jnp.where(prefix_mask, seq_indices, -1))` 找最后有效 token ([openpi #701](https://github.com/Physical-Intelligence/openpi/issues/701))

### 2.2 GR00T 微调

**官方 checkpoint 复现困难**:
- PnPAppleToPlate 官方微调 checkpoint 0% 成功率，2 个独立团队确认。左臂反转(饱和 ~154°)，手从不激活。结论：发布的 checkpoint 可能基于内部版本训练。([GR00T #574](https://github.com/NVIDIA/Isaac-GR00T/issues/574), 2 独立确认)
- RoboCasa 微调后性能反而低于 zero-shot (~40%)。外部论文 (villa-X, MolmoAct) 也报告类似问题。([GR00T #308](https://github.com/NVIDIA/Isaac-GR00T/issues/308))
- 关键修复：缺失的 trained token 问题 (PR #246)，修复后 24 任务平均 ~45% SR (8×H100, bs=120, lr=3e-5, 30k steps) ([GR00T #249](https://github.com/NVIDIA/Isaac-GR00T/issues/249))

**GR00T GPU 最低要求**:
- 最低：2×4090 + `--no-tune-visual --batch-size 1` + LoRA
- 实用：4×4090D + `model.bfloat16()` 仅 diffusion+projector
- 舒适：H100/A100 全量微调
- `--no-tune-visual` 大幅降低显存但限制新 embodiment 适应能力 ([GR00T #24](https://github.com/NVIDIA/Isaac-GR00T/issues/24))

### 2.3 SmolVLA / Diffusion Policy

**数据加载瓶颈（影响所有非 ACT 策略）**:
- 视频解码占数据加载时间的 92%（H100 上实测）。ACT 不受影响。([lerobot #1488](https://github.com/huggingface/lerobot/issues/1488), 19 comments, profiling 数据)
- torchcodec 单样本 3x 快（2.4ms vs 7.1ms），但多进程下有 forked decoder state 问题
- 每 `num_workers` 步卡一次，GPU 利用率跌至 0%
- 数据录制也慢：15s episode 需 30-40s 处理 ([lerobot #1434](https://github.com/huggingface/lerobot/issues/1434))

**SmolVLA LIBERO 复现**:
- 关键发现：论文用 **单步推理**（action chunk = 1），不是完整 chunk 执行。改为单步后 LIBERO-spatial 从 66.8%→82% ([lerobot #1316](https://github.com/huggingface/lerobot/issues/1316), 54 comments)
- Batch size 显著影响结果：4/8 差，32+ 好
- MuJoCo 版本重要：用 3.3.2 避免颜色渲染差异

### 2.4 多 GPU / 分布式训练

- JAX 多 GPU (8+) JIT 编译可能无限挂起。根因：CUDA 12.2 不兼容，升级到 12.8 解决。([openpi #480](https://github.com/Physical-Intelligence/openpi/issues/480))
- orbax-checkpoint 异步保存在 7% 进度时崩溃。修复：禁用异步保存 或 降级 orbax 到 0.11.1。([openpi #505](https://github.com/Physical-Intelligence/openpi/issues/505))
- 多节点训练：社区开源 SLURM 方案 `openpi_multinode` ([openpi #480](https://github.com/Physical-Intelligence/openpi/issues/480))

---

## 3. 推理与部署

### 3.1 Action Chunking 陷阱

**GR00T 官方示例只用第一个 action，丢 10% 性能**:
- 模型输出 16 步 action chunk，但 eval 代码只用 `action_chunk[0]`
- 使用全部 16 步：LIBERO Spatial 90%→96%, Long 80%→90%
- NVIDIA 承认示例代码有误，正在修复 ([GR00T #424](https://github.com/NVIDIA/Isaac-GR00T/issues/424), 3 comments with benchmarks)

### 3.2 LIBERO 基准部署注意

**GR00T × LIBERO gripper 映射 bug**:
- LIBERO sim 用 [-1,1]，但训练归一化到 [0,1]
- 修复：`action_chunk[:,-1] = -2*action_chunk[:,-1] + 1` ([GR00T #136](https://github.com/NVIDIA/Isaac-GR00T/issues/136))
- 图像需旋转：`np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])`
- 最佳结果：bs=64, 20k steps → spatial 94.4%, object 97.6%

### 3.3 Pi0 Checkpoint 兼容性

- Pi0-fast LIBERO checkpoint 在新 repo 版本不工作（成功率 0%）。最后可用 commit：`e4580662` (2025-09-07)。从 base 重新微调也是 0%。([openpi #849](https://github.com/Physical-Intelligence/openpi/issues/849), 4 人确认)

### 3.4 量化可行性

- PaliGemma VLM 的 MLP 做 FP8/NVFP4 PTQ：LIBERO 2000 episode 无显著精度损失。但当前只是权重量化+反量化回 BF16，无实际加速。([openpi #897](https://github.com/Physical-Intelligence/openpi/issues/897))
- TensorRT 导出需求强烈但尚无官方支持。SO-101 + Jetson Orin Nano 是目标场景。([lerobot #3146](https://github.com/huggingface/lerobot/issues/3146))

---

## 4. 模拟器与基准

### 4.1 ManiSkill

- **EE pose 控制器比 joint-space 慢 5x**：200 次迭代 IK 求解器（pytorch_kinematics）。delta EE 控制器快得多（单步 Jacobian）。([ManiSkill #955](https://github.com/haosulab/ManiSkill/issues/955))
- **灵巧手仿真不稳定**：InspireHand URDF 在 ManiSkill 中出现 NaN（MuJoCo 正常），原因不明。([ManiSkill #925](https://github.com/haosulab/ManiSkill/issues/925))
- **运动规划器静默失败**：RRTConnect 有时不报错但轨迹不执行；腕关节 360° 旋转导致问题。([ManiSkill #1249](https://github.com/haosulab/ManiSkill/issues/1249))
- **像素 RL 基线**：SAC + RGBD, 32 envs, 64x64, 1M steps, 1-1.5h on 4090 ([ManiSkill #667](https://github.com/haosulab/ManiSkill/issues/667))

### 4.2 Genesis

- **URDF 柔性体仿真差异**：同一 URDF 在 Gazebo 中是柔性线缆，在 Genesis 中是刚性杆。默认 joint damping/armature 参数不同。([Genesis #1135](https://github.com/Genesis-Embodied-AI/Genesis/issues/1135), 42 comments)
- **内存扩展非线性**：10K envs 用 7GB，20K envs 崩溃。疑似 int32 索引限制。([Genesis #1740](https://github.com/Genesis-Embodied-AI/Genesis/issues/1740))
- **Mac M1 OpenGL 问题**：viewer 无法打开，需 Vulkan backend 或 headless 模式 ([Genesis #11](https://github.com/Genesis-Embodied-AI/Genesis/issues/11), [#36](https://github.com/Genesis-Embodied-AI/Genesis/issues/36))

### 4.3 IsaacLab

- **粗糙地形奖励函数有 bug**：`base_height_l2` 数学上恒等于 `(-target_height)^2`，因为传感器高度和 root link 高度相消。应改用 height scanner 射线命中点中心。([IsaacLab #1698](https://github.com/isaac-sim/IsaacLab/issues/1698))
- **非 Panda 机器人抓取困难**：UR10e 需仔细调 EE frame Z-offset。([IsaacLab #1225](https://github.com/isaac-sim/IsaacLab/issues/1225))
- **训练中模型消失（NaN）**：IsaacLab 2294，与浮点溢出相关 ([IsaacLab #2294](https://github.com/isaac-sim/IsaacLab/issues/2294))

---

## 5. 机器人硬件与数据采集

### 5.1 SO-100/SO-101 常见问题

- **电机校准反转**：肩关节（2号电机）经常反向。根因：编码器值在边界处翻转。解决：(1) 反序校准 3,2,1; (2) 组装时关节放中间位置; (3) 手动填校准值。([lerobot #930](https://github.com/huggingface/lerobot/issues/930), 26 comments)
- **Feetech 电机断连**：遥操作中 follower 断连（`sync_read` 失败）。降到 30Hz 减少但不消除。Feetech 电机本身不稳定。变通：在 `get_observation()` 加 try/catch。([lerobot #3131](https://github.com/huggingface/lerobot/issues/3131))
- **编码器溢出**：`Magnitude 2114 exceeds 2047` 校准错误。([lerobot #1296](https://github.com/huggingface/lerobot/issues/1296), 23 comments)

### 5.2 GR00T × SO100 真机

- 社区报告抖动、抓取不良 ([GR00T #130](https://github.com/NVIDIA/Isaac-GR00T/issues/130))
- LoRA 性能低于全量微调
- **相机位置极度敏感**：1cm 偏移可破坏任务完成
- 腕部相机在多相机设置中影响最大

### 5.3 深度图与触觉

- LeRobot 当前不原生支持深度图、点云、触觉。社区强烈需求。([lerobot #1144](https://github.com/huggingface/lerobot/issues/1144), 27 comments)
- 变通：存为 tensor 在 parquet；或伪装为 RGB 图像
- 进行中：PR #2604 支持未压缩 UINT16 深度图 + hue-depth-encoding 压缩

---

## 6. 跨仓库收敛信号

以下问题在 **2+ 个仓库** 独立出现，代表 VLA 领域的系统性挑战：

| 收敛主题 | 出现仓库 | 核心矛盾 |
|---------|---------|---------|
| **Action 表示混乱** | openpi, GR00T, lerobot | delta vs absolute, EEF vs joint, 归一化不一致 |
| **Jetson 部署 OOM** | openpi, GR00T | 统一内存 + JAX/PyTorch 预分配冲突 |
| **官方 checkpoint 不可复现** | openpi, GR00T | 内部训练管线 ≠ 开源版本 |
| **视频解码瓶颈** | lerobot, GR00T | torchvision_av 内存泄漏 + mp4 解码慢 |
| **RTX 50 系列不兼容** | lerobot, GR00T, IsaacLab | FlashAttention SM120, torch SM120, 驱动渲染 |
| **LIBERO gripper 映射** | openpi, GR00T | [-1,1] vs [0,1] 归一化不匹配 |

---

## 7. 框架采纳度信号（2026-07-24 快照）

| 框架 | 7d Issues | 采纳阶段 | DFI | 信号 |
|------|-----------|---------|-----|------|
| **lerobot** | 33 | 早期探索 | 0.04 (low) | 主要摩擦: hardware |
| **isaaclab** | 19 | 早期探索 | 0.04 (low) | 主要摩擦: hardware |
| **genesis** | 15 | 早期探索 | 0.00 (low) | 主要摩擦: deploy |
| **gr00t** | 4 | 混合 | 0.00 (low) | 主要摩擦: deploy |
| **openpi** | 2 | 混合 | 0.15 (low) | 主要摩擦: deploy |
| **internvla** | 2 | 混合 | 0.00 (low) | 主要摩擦: deploy |
| **diffpol** | 1 | 混合 | 0.00 (low) | 主要摩擦: deploy |
| **mujoco** | 1 | 混合 | 0.00 (low) | 主要摩擦: deploy |
| **openvla** | 0 | **停滞** | 0.00 (low) | 几乎无活动 |
| **rdt** | 0 | **停滞** | 0.00 (low) | 几乎无活动 |
| **octo** | 0 | **停滞** | 0.00 (low) | 几乎无活动 |
| **magma** | 0 | **停滞** | 0.00 (low) | 几乎无活动 |
| **evo-rl** | 0 | **停滞** | 0.00 (low) | 几乎无活动 |
| **act** | 0 | **停滞** | 0.00 (low) | 几乎无活动 |
| **aloha** | 0 | **停滞** | 0.00 (low) | 几乎无活动 |
| **maniskill** | 0 | **停滞** | 0.00 (low) | 几乎无活动 |
| **libero** | 0 | **停滞** | 0.00 (low) | 几乎无活动 |
| **robosuite** | 0 | **停滞** | 0.00 (low) | 几乎无活动 |
| **simplerenv** | 0 | **停滞** | 0.00 (low) | 几乎无活动 |
| **unitree-rl** | 0 | **停滞** | 0.00 (low) | 几乎无活动 |
| **gello** | 0 | **停滞** | 0.00 (low) | 几乎无活动 |

> 数据来源: Pulsar GitHub Issues Sensor, 77 issues analyzed in 7-day window

---

*本文档由 Pulsar GitHub Issues Sensor 自动采集 + 人工蒸馏。Issue 链接为 evidence，可直接点击查看原始讨论。*

*最后更新: 2026-07-24*
