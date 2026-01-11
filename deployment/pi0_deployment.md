# Pi0 真机部署指南 (Pi0 Real-World Deployment Guide)

> **基于官方 OpenPI 仓库**: 本指南基于 Physical Intelligence 开源的 `openpi` 架构编写。
> **适用版本**: π0 (Base), π0-FAST (High Frequency).

## 1. 真机硬件配置 (Real-World Hardware Configuration)

> **官方参考**: 基于 Physical Intelligence 内部使用的 DROID 和 ALOHA 平台配置。

### 1.1 机器人平台 (Supported Robots)
虽然 Pi0 是通用的，但官方仓库 `openpi` 重点适配了以下平台：
- **DROID (Single Arm)**: 基于 **Franka Emika Panda** 机械臂。
    - *特点*: 高精度力控，适合精细操作。
- **ALOHA (Dual Arm)**: 基于 **ViperX 300** (Trossen Robotics) 或自研低成本机械臂。
    - *特点*: 双臂协同，适合双手操作 (如叠衣服)。
- **Trossen WidowX**: 低成本桌面级机械臂，适合入门和测试。

### 1.2 视觉系统 (Vision System)
Pi0 极其依赖多视角视觉输入。标准的官方推荐配置是 **4-Camera Setup**：

| 视角 (View) | 推荐相机 | 安装位置 | 作用 |
| :--- | :--- | :--- | :--- |
| **High Cam** | Logitech C920 / RealSense D435 | 机器人正上方 (俯视) | 全局规划，定位物体位置。 |
| **Low Cam** | Logitech C920 | 机器人正前方 (平视) | 补充视角，观察物体侧面细节。 |
| **Left Wrist** | Arducam / RealSense | 左手腕部 | **关键视角**。用于对准物体，解决遮挡问题。 |
| **Right Wrist** | Arducam / RealSense | 右手腕部 | 同上。 |

> **注意**: 所有相机必须进行 **外参标定 (Extrinsic Calibration)**，确保坐标系对齐。

### 1.3 计算单元架构 (Compute Architecture)
为了解决 Pi0 (3B VLM) 推理重的问题，**强烈推荐** 采用 "大脑-小脑" 分离架构：

- **大脑 (Server)**: 高性能工作站
    - **GPU**: **NVIDIA RTX 4090 (24GB)** 或 A6000。
    - **任务**: 运行 Pi0 模型，接收图像，生成 Action Chunk (50Hz)。
    - **连接**: 通过千兆以太网 (LAN) 连接机器人。

- **小脑 (Client)**: 机载电脑 (Onboard Computer)
    - **推荐型号**: **Intel NUC 12/13 Pro (i7 版本)** 或 **Jetson Orin AGX (64GB)**。
    - **配置要求**:
        - **CPU**: i7-1260P 或更高 (需处理 4 路相机流编解码)。
        - **RAM**: **32GB DDR4/DDR5** (16GB 勉强，32GB 稳妥)。
        - **IO**: 必须有 **Thunderbolt 4** 或 USB 3.2 Gen 2 (用于扩展 PCIe 卡)。
    - **任务**: 读取相机数据，发送给 Server；接收 Action，控制电机。
    - **优势**: 机载电脑只需要负责 I/O，不需要跑大模型，大大降低了机载硬件成本和散热压力。

## 2. 软件环境搭建 (Software Setup)

推荐使用 Ubuntu 22.04 + Python 3.11。

### 2.1 安装 OpenPI
使用 `uv` 进行依赖管理 (官方推荐)：

```bash
# 1. 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 克隆仓库 (包含子模块)
git clone --recursive https://github.com/Physical-Intelligence/openpi.git
cd openpi

# 3. 创建虚拟环境并安装
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

### 2.2 下载模型权重 (Download Weights)

模型权重托管在 Hugging Face，通常需要申请访问权限 (Gated Model)。

#### 步骤 1: 准备 Hugging Face 账号与 Token
1.  注册/登录 [Hugging Face](https://huggingface.co/)。
2.  访问 `physical-intelligence/pi0` 模型页面，点击 **"Request Access"** 并同意用户协议。
3.  进入 [Settings -> Access Tokens](https://huggingface.co/settings/tokens)。
4.  点击 **"New token"**，创建一个类型为 `Read` 的 Token，并复制它 (以 `hf_` 开头)。

#### 步骤 2: 命令行登录
在终端中运行以下命令，粘贴刚才复制的 Token：

```bash
huggingface-cli login
# 提示 "Token:" 时粘贴 (输入不会显示)，按回车。
# 提示 "Add token as git credential?" 选 Y。
```

#### 步骤 3: 下载模型
推荐使用 `huggingface-cli` 下载，支持断点续传。

```bash
# 1. 安装 CLI 工具 (如果还没装)
pip install -U "huggingface_hub[cli]"

# 2. 下载 π0 Base 模型 (推荐下载到指定目录)
huggingface-cli download physical-intelligence/pi0 \
    --local-dir checkpoints/pi0 \
    --resume-download

# [可选] 如果只需要下载特定文件 (例如只下载权重，不下载 safe tensors)
# huggingface-cli download physical-intelligence/pi0 --include "*.pt" --local-dir checkpoints/pi0
```

> **提示**: 模型文件较大 (约 6-10GB)，建议在 `screen` 或 `tmux` 会话中下载，防止网络中断。

## 3. 部署架构：Remote Inference (推荐)

由于 Pi0 计算量较大，在机载电脑 (如 Orin) 上直接运行可能无法达到 50Hz 的控制频率。官方推荐 **Server-Client** 架构。

### 3.1 为什么不能直接在 Orin 上跑? (Why Server-Client?)
- **算力瓶颈**: Pi0 是一个 **3B (30亿参数)** 的视觉语言模型。在 Jetson Orin 上推理一次可能需要 100ms+ (即 <10Hz)。
- **控制要求**: 灵巧手等复杂机器人需要 **50Hz** (即 20ms 一次) 的控制频率才能保证动作流畅、不抖动。
- **解决方案**: **"大脑-小脑" 分离 (Brain-Cerebellum Split)**。
    - **大脑 (Server)**: 放在桌子下的高性能工作站 (RTX 4090)。负责"思考" (运行大模型)，算力无限。
    - **小脑 (Client)**: 放在机器人肚子里的 Orin/NUC。负责"反射" (收发信号)，实时性强。
    - **连接**: 两者通过 **千兆局域网 (Gigabit LAN)** 连接，延迟通常 <1ms，完全可以忽略。

### 3.2 架构图 (Architecture)
```mermaid
graph LR
    subgraph Robot [机器人端 (Client/Cerebellum)]
        Camera[相机] --> |图像| Network
        Network --> |动作| Motor[电机驱动]
    end
    
    subgraph Workstation [工作站端 (Server/Brain)]
        Network[局域网] --> |图像| Model[Pi0 Model (4090)]
        Model --> |Action Chunk| Network
    end
```

### 3.2 Server 端 (4090 工作站)
启动推理服务器：
```bash
python -m openpi.serving.server --model_path checkpoints/pi0 --port 8000
```

### 3.3 Client 端 (机器人/Orin)
连接服务器并执行动作：
```python
from openpi.serving.client import Pi0Client

client = Pi0Client(url="ws://192.168.1.100:8000")

while True:
    # 1. 获取机器人图像
    img = robot.get_camera_image()
    
    # 2. 发送请求并获取动作 (Blocking or Async)
    action = client.predict(image=img, instruction="Pick up the apple")
    
    # 3. 执行动作
    robot.execute(action)
```

## 4. 微调工作流 (Fine-tuning Workflow)

> **注意**: 目前 OpenPI 主要支持 **Full Fine-tuning** (全量微调)。LoRA 支持尚在开发中。

### 4.1 数据准备 (Data Preparation)
Pi0 训练需要 **RLDS (Reinforcement Learning Datasets)** 格式的数据。

1.  **采集数据**: 使用 Teleoperation 采集机器人操作数据 (Images, Joint States, Actions)。
2.  **转换为 RLDS**:
    - OpenPI 没有一键转换脚本，需要基于 `tensorflow_datasets` (TFDS) 编写自定义 Builder。
    - **核心步骤**:
        1. 创建 `my_dataset_builder.py` 继承 `tfds.core.GeneratorBasedBuilder`。
        2. 实现 `_info()` 定义数据结构 (Features: image, action, language)。
        3. 实现 `_generate_examples()` 读取原始数据并生成样本。
    - **构建**: 运行 `tfds build` 生成 TFRecord 文件。

### 4.2 训练配置 (Training Configuration)
修改 `configs/pi0_finetune.yaml` (或类似配置文件)：

```yaml
model:
  type: "pi0"
  # ... model params ...

train:
  batch_size: 8  # 根据显存调整 (A100 80G 可开大)
  learning_rate: 1e-4
  max_steps: 10000
  save_interval: 1000

dataset:
  name: "my_custom_dataset"
  data_dir: "/path/to/rlds_data"
```

### 4.3 启动训练 (Start Training)
使用 `torchrun` 启动单机多卡训练：

```bash
torchrun --nproc_per_node=8 -m openpi.training.train \
    --config configs/pi0_finetune.yaml \
    --exp_name my_finetune_exp
```

### 4.4 评估与测试 (Evaluation)
训练过程中会保存 Checkpoints。
1.  **加载 Checkpoint**: 修改推理脚本指向新的 `model.pt`。
2.  **真机测试**: 在机器人上运行相同的任务，观察成功率。
    - *提示*: 也可以使用 `openpi` 的 Policy Evaluation 脚本在仿真环境中预测试 (如果有 Sim 环境)。

## 5. 常见问题 (FAQ)

### Q1: 为什么我在 Orin 上跑只有 5Hz?
**A**: Pi0 是 3B VLM，Orin 的推理能力有限。请开启 **RTC (Real-Time Chunking)** 或使用 Remote Inference。

### Q2: 什么是 RTC (Real-Time Chunking)?
**A**: 这是一种在端侧部署时的优化技术。模型一次推理生成多个未来的动作块 (Chunk)，机器人依次执行这些动作，从而掩盖推理延迟。

### Q3: Pi0 支持多卡训练吗?
**A**: 支持。使用 `torchrun` 启动多卡训练，可以显著减少显存压力 (Model Parallelism)。

## 6. RealSense 避坑指南 (Troubleshooting)

> **问题**: "RealSense 內容爆炸" (USB Bandwidth Overflow / Frame Drops) 是 4-Camera Setup 最常见的问题。

### 6.1 现象
- 报错: `RuntimeError: Frame didn't arrive within 5000`
- 报错: `RS2_USB_STATUS_OVERFLOW`
- 现象: 相机掉线，或者 FPS 极低 (只有 15fps 甚至更低)。

### 6.2 根本原因 (Root Cause)
- **USB 带宽瓶颈**: 4 个 RealSense D435 跑 640x480@30fps 需要约 1.5Gbps 带宽。如果它们插在同一个 USB Root Hub 上，会瞬间挤爆总线。
- **供电不足**: 4 个相机同时启动瞬间电流很大，可能导致电压骤降，相机重启。

### 6.3 解决方案 (Solutions)

#### 方案 A: 物理隔离 (Physical Isolation) [推荐]
不要把所有相机插在机箱前面板或主板自带的 USB 口上 (它们通常共享一个 Controller)。
1.  **购买 PCIe USB 扩展卡**: 推荐带有 **独立控制器 (Independent Controllers)** 的扩展卡 (如 StarTech 4-Port Dedicated 5Gbps)。
2.  **分配**: 
    - High Cam + Low Cam -> 主板 USB 口
    - Left Wrist + Right Wrist -> PCIe 扩展卡

#### 方案 B: 降低规格 (Reduce Specs)
如果无法加硬件，必须降低数据量：
1.  **降低分辨率**: 从 848x480 降级到 **640x480** 甚至 **424x240** (Pi0 对分辨率不敏感)。
2.  **降低帧率**: 强制设置为 **15fps** 或 **30fps** (不要用 60fps)。
3.  **关闭深度流**: 如果只用 RGB，务必在 Launch 文件中关闭 Depth Stream。

#### 方案 C: 硬件同步 (Hardware Sync)
多相机之间会有红外干扰 (IR Interference)。
1.  **Master-Slave 模式**: 设置一个相机为 Master (发送触发信号)，其他为 Slave。
2.  **Sync Cable**: 使用同步线连接各相机的 Sync 接口。
3.  **Librealsense 设置**:
    ```python
    # Master
    sensor.set_option(rs.option.inter_cam_sync_mode, 1)
    # Slave
    sensor.set_option(rs.option.inter_cam_sync_mode, 2)
    ```

## 7. 相机标定 (Camera Calibration)

> **重要**: 这是一个通用的部署步骤。请参考专门的 **[相机标定指南 (Camera Calibration Guide)](./camera_calibration_eye_in_hand.md)**。

它涵盖了：
- **Eye-to-Hand** (High/Low Cam)
- **Eye-in-Hand** (Wrist Cam)
- **Aruco 标定流程**

---
[← Back to Deployment](./README.md)
