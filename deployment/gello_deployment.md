# GELLO 遥操作部署指南 (UR5)

> **论文**: [GELLO: A General, Low-Cost, and Intuitive Teleoperation Framework for Robot Manipulators](https://arxiv.org/abs/2309.13037)
> **GitHub**: [wuphilipp/gello_software](https://github.com/wuphilipp/gello_software)
> **硬件 CAD**: [wuphilipp/gello_mechanical](https://github.com/wuphilipp/gello_mechanical)
> **本指南适用**: UR5 / UR5e 机械臂

---

## 0. 官方仓库与硬件采购

### 0.1 官方仓库结构

GELLO 项目分为两个仓库：

```
wuphilipp/
├── gello_software/          # 软件代码
│   ├── gello/
│   │   ├── agents/          # 策略接口
│   │   ├── cameras/         # 相机驱动 (RealSense, USB)
│   │   ├── dynamixel/       # Dynamixel 电机控制
│   │   └── robots/          # 机器人驱动
│   │       ├── ur.py        # ⭐ UR5 驱动
│   │       ├── franka.py    # Franka 驱动
│   │       └── xarm.py      # xArm 驱动
│   ├── scripts/
│   │   ├── gello_get_offset.py   # 标定脚本
│   │   └── launch_nodes.py       # 启动脚本
│   └── experiments/         # 实验脚本
│
└── gello_mechanical/        # 硬件设计
    ├── stl/                 # 3D 打印文件
    │   ├── ur/              # ⭐ UR5 专用零件
    │   ├── franka/
    │   └── xarm/
    ├── cad/                 # SolidWorks 源文件
    └── BOM.md               # 物料清单
```

### 0.2 硬件采购清单 (淘宝)

| 零件 | 规格 | 数量 | 淘宝参考价 | 备注 |
|:---|:---|:---|:---|:---|
| **Dynamixel XM430-W350** | 伺服电机 | 3 | ~¥800/个 | 大关节 (J1-J3) |
| **Dynamixel XL330-M288** | 伺服电机 | 4 | ~¥200/个 | 小关节 (J4-J6) + 夹爪 |
| **U2D2** | USB-Dynamixel 适配器 | 1 | ~¥300 | 官方适配器 |
| **U2D2 Power Hub** | 电源分配板 | 1 | ~¥150 | 可选，方便供电 |
| **12V 5A 电源** | DC 电源 | 1 | ~¥50 | XM430 供电 |
| **5V 3A 电源** | DC 电源 | 1 | ~¥30 | XL330 供电 |
| **3D 打印件** | PLA/PETG | 1套 | ~¥200-500 | 淘宝代打印 |
| **轴承 6800ZZ** | 10×19×5mm | 若干 | ~¥5/个 | 关节轴承 |
| **螺丝套装** | M2/M2.5/M3 | 1套 | ~¥30 | 内六角螺丝 |

**总成本**: 约 **¥3500-4500** (不含机械臂)

**淘宝搜索关键词**:
- Dynamixel 伺服电机
- U2D2 Dynamixel
- 3D打印代工 PLA

> ⚠️ **注意**: Dynamixel 电机有国产仿制品，价格约 1/3，但精度和寿命较差，建议正品。

### 0.3 3D 打印说明

```bash
# 下载 UR5 专用 STL 文件
git clone https://github.com/wuphilipp/gello_mechanical.git
cd gello_mechanical/stl/ur

# 需要打印的零件:
ls *.stl
# base.stl
# link1.stl
# link2.stl
# link3.stl
# link4.stl
# link5.stl
# link6.stl
# gripper_mount.stl
```

**打印参数建议**:
| 参数 | 推荐值 |
|:---|:---|
| 材料 | **PETG** (强度好) 或 PLA |
| 层高 | 0.2mm |
| 填充 | 30-50% |
| 壁厚 | 3 层 |
| 支撑 | 需要 (部分零件) |

---

## 1. GELLO 概述

### 1.1 什么是 GELLO?

GELLO (General, Low-cost, and Intuitive Teleoperation) 是一种**低成本、便携式**的遥操作装置，用于收集机器人操作数据。

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GELLO 系统架构                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────────┐         ┌─────────────────┐                  │
│   │   GELLO 主臂    │ ──────▶ │   UR5 从臂      │                  │
│   │ (3D打印+Dynamixel)│  关节   │  (真机/仿真)    │                  │
│   │                 │  映射   │                 │                  │
│   └────────┬────────┘         └────────┬────────┘                  │
│            │                           │                            │
│            │ 读取关节角度                │ 执行动作                   │
│            ▼                           ▼                            │
│   ┌─────────────────────────────────────────────────────┐          │
│   │              数据采集脚本 (collect_data.py)          │          │
│   │                                                     │          │
│   │  ┌─────────┐  ┌─────────┐  ┌─────────┐            │          │
│   │  │关节角度 │  │图像观测 │  │夹爪状态 │            │          │
│   │  └─────────┘  └─────────┘  └─────────┘            │          │
│   │                     │                              │          │
│   │                     ▼                              │          │
│   │              LeRobot / HDF5 数据集                 │          │
│   └─────────────────────────────────────────────────────┘          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 GELLO vs ALOHA 主从臂

| 维度 | GELLO | ALOHA (主从臂) |
|:---|:---|:---|
| **成本** | ~$500 (3D打印+Dynamixel) | ~$4000+ (WidowX 250 × 2) |
| **便携性** | **极高** (可手持) | 低 (桌面固定) |
| **力反馈** | 无 | 有 (重力补偿) |
| **精度** | 较低 (无编码器反馈) | **较高** (电机编码器) |
| **适用场景** | 快速原型、数据采集 | 高精度操作、双臂任务 |
| **维护成本** | 低 (零件便宜) | 高 (电机昂贵) |

### 1.3 为什么选择 GELLO?

1. **低成本**: 材料成本约 $500，适合预算有限的实验室
2. **便携**: 可手持操作，不占用桌面空间
3. **直观**: 1:1 关节映射，无需学习曲线
4. **通用**: 支持 UR5, Franka, xArm 等多种机械臂
5. **开源**: 完整的 CAD 文件和软件代码

---

## 2. 环境配置

### 2.1 系统要求

| 组件 | 要求 |
|:---|:---|
| **操作系统** | Ubuntu 20.04 / 22.04 |
| **Python** | 3.8+ |
| **UR5 控制器** | Polyscope 5.0+ (支持 RTDE) |
| **Dynamixel** | U2D2 适配器 + XL330/XM430 电机 |

### 2.2 创建 Conda 环境

```bash
# 创建环境
conda create -n gello python=3.10 -y
conda activate gello

# 克隆仓库
git clone https://github.com/wuphilipp/gello_software.git
cd gello_software

# 安装依赖
pip install -e .
pip install ur_rtde  # UR5 RTDE 通信
pip install dynamixel_sdk  # Dynamixel 电机控制
pip install opencv-python  # 图像采集
pip install h5py  # HDF5 数据存储
```

### 2.3 Dynamixel 配置

#### 2.3.1 查找 USB 端口

```bash
# 插入 U2D2 适配器后
ls /dev/ttyUSB*
# 输出: /dev/ttyUSB0

# 设置权限 (避免每次 sudo)
sudo usermod -aG dialout $USER
# 重新登录生效
```

#### 2.3.2 配置电机 ID

GELLO 使用 **7 个 Dynamixel 电机** (6 关节 + 1 夹爪)，需要为每个电机设置唯一 ID：

```python
# scripts/set_motor_id.py
from dynamixel_sdk import *

# 配置
BAUDRATE = 1000000
DEVICENAME = '/dev/ttyUSB0'
PROTOCOL_VERSION = 2.0

# 初始化
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

portHandler.openPort()
portHandler.setBaudRate(BAUDRATE)

# 扫描并设置 ID (一次只连接一个电机!)
OLD_ID = 1  # 出厂默认
NEW_ID = 2  # 目标 ID (1-7)

# 写入新 ID
ADDR_ID = 7
packetHandler.write1ByteTxRx(portHandler, OLD_ID, ADDR_ID, NEW_ID)
print(f"Motor ID changed: {OLD_ID} -> {NEW_ID}")
```

**电机 ID 映射 (UR5)**:

| 电机 ID | 对应关节 | Dynamixel 型号 |
|:---|:---|:---|
| 1 | Base (J1) | XM430 |
| 2 | Shoulder (J2) | XM430 |
| 3 | Elbow (J3) | XM430 |
| 4 | Wrist 1 (J4) | XL330 |
| 5 | Wrist 2 (J5) | XL330 |
| 6 | Wrist 3 (J6) | XL330 |
| 7 | Gripper | XL330 |

### 2.4 UR5 通信配置

#### 2.4.1 方案选择: RTDE vs ROS

| 方案 | 优点 | 缺点 |
|:---|:---|:---|
| **RTDE** (推荐) | 低延迟 (~8ms)、无需 ROS 依赖 | 功能相对简单 |
| **ROS/ROS2** | 生态丰富、可视化好 | 配置复杂、延迟较高 |

#### 2.4.2 RTDE 配置

```python
# config/ur5_config.yaml
ur5:
  ip: "192.168.1.100"  # UR5 控制器 IP
  rtde_frequency: 125  # Hz
  
  # 关节限位 (rad)
  joint_limits:
    lower: [-6.28, -6.28, -3.14, -6.28, -6.28, -6.28]
    upper: [6.28, 6.28, 3.14, 6.28, 6.28, 6.28]
  
  # 速度限制
  max_joint_velocity: 1.0  # rad/s
  max_joint_acceleration: 2.0  # rad/s^2
```

#### 2.4.3 UR5 控制器设置

在 UR5 示教器上:
1. **设置** → **系统** → **网络设置**
2. 设置静态 IP (如 `192.168.1.100`)
3. **安装** → **URCaps** → 确保 **External Control** 已安装
4. 创建程序: **外部控制** → 设置主机 IP 为你的电脑 IP

---

## 3. UR5 特定配置

### 3.1 关节映射

GELLO 有 **7-DoF** (6 关节 + 夹爪)，UR5 有 **6-DoF**:

```python
# gello/robots/ur.py
class UR5Robot:
    """UR5 机械臂驱动"""
    
    def __init__(self, ip: str = "192.168.1.100"):
        import rtde_control
        import rtde_receive
        
        self.rtde_c = rtde_control.RTDEControlInterface(ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(ip)
        
        # 关节映射: GELLO ID -> UR5 Joint Index
        self.joint_mapping = {
            1: 0,  # Base
            2: 1,  # Shoulder  
            3: 2,  # Elbow
            4: 3,  # Wrist 1
            5: 4,  # Wrist 2
            6: 5,  # Wrist 3
        }
        # ID 7 为夹爪，单独处理
    
    def get_joint_positions(self) -> np.ndarray:
        """获取当前关节角度 (6-DoF)"""
        return np.array(self.rtde_r.getActualQ())
    
    def move_joints(self, target_q: np.ndarray, speed: float = 0.5):
        """关节空间移动"""
        self.rtde_c.moveJ(target_q.tolist(), speed, 0.5)
    
    def servo_joints(self, target_q: np.ndarray, dt: float = 0.008):
        """伺服模式 (低延迟跟随)"""
        self.rtde_c.servoJ(
            target_q.tolist(),
            velocity=0.5,
            acceleration=0.5,
            dt=dt,
            lookahead_time=0.1,
            gain=300
        )
```

### 3.2 标定流程

#### 3.2.1 零位标定

GELLO 和 UR5 的零位定义不同，需要标定偏移:

```python
# scripts/calibrate_gello.py
import numpy as np
from gello.dynamixel import DynamixelDriver
from gello.robots.ur import UR5Robot

def calibrate():
    """标定 GELLO 零位偏移"""
    
    gello = DynamixelDriver(port='/dev/ttyUSB0', motor_ids=[1,2,3,4,5,6,7])
    ur5 = UR5Robot(ip='192.168.1.100')
    
    print("=== GELLO 零位标定 ===")
    print("1. 将 UR5 移动到 Home 位置 (所有关节为 0)")
    print("2. 调整 GELLO 使其与 UR5 姿态对齐")
    input("按 Enter 继续...")
    
    # 读取当前 GELLO 角度
    gello_angles = gello.get_positions()[:6]  # 忽略夹爪
    ur5_angles = ur5.get_joint_positions()
    
    # 计算偏移
    offsets = ur5_angles - gello_angles
    
    print(f"\n标定偏移 (rad):")
    for i, offset in enumerate(offsets):
        print(f"  Joint {i+1}: {offset:.4f} ({np.degrees(offset):.2f}°)")
    
    # 保存到配置文件
    np.save('config/gello_offsets.npy', offsets)
    print("\n偏移已保存到 config/gello_offsets.npy")
    
    return offsets

if __name__ == "__main__":
    calibrate()
```

#### 3.2.2 验证标定

```python
# scripts/verify_calibration.py
def verify():
    """验证标定结果"""
    
    offsets = np.load('config/gello_offsets.npy')
    gello = DynamixelDriver(...)
    ur5 = UR5Robot(...)
    
    print("=== 标定验证 ===")
    print("移动 GELLO，观察 UR5 是否同步跟随")
    print("按 Ctrl+C 退出")
    
    try:
        while True:
            # 读取 GELLO 角度并应用偏移
            gello_q = gello.get_positions()[:6]
            target_q = gello_q + offsets
            
            # 伺服跟随
            ur5.servo_joints(target_q)
            
            # 打印误差
            actual_q = ur5.get_joint_positions()
            error = np.abs(target_q - actual_q)
            print(f"跟随误差 (deg): {np.degrees(error).round(2)}", end='\r')
            
            time.sleep(0.008)
    except KeyboardInterrupt:
        print("\n验证结束")
```

### 3.3 夹爪配置

```python
# gello/robots/gripper.py
class Robotiq2F85:
    """Robotiq 2F-85 夹爪驱动"""
    
    def __init__(self, ur_rtde_control):
        self.rtde_c = ur_rtde_control
        self.min_width = 0.0    # 完全闭合
        self.max_width = 0.085  # 85mm 开口
    
    def set_position(self, width: float, speed: float = 0.1):
        """设置夹爪开口宽度 (m)"""
        width = np.clip(width, self.min_width, self.max_width)
        # Robotiq 通过 UR 的 Tool Communication 控制
        self.rtde_c.moveToolPosition([0, 0, 0, 0, 0, 0], speed)
    
    def from_gello(self, gello_gripper_angle: float) -> float:
        """将 GELLO 夹爪角度映射到 Robotiq 开口"""
        # GELLO 夹爪角度范围: 0 (闭) ~ 1.5 (开)
        normalized = gello_gripper_angle / 1.5
        return normalized * self.max_width
```

---

## 4. 数据采集流程

### 4.1 采集脚本

```python
# scripts/collect_data.py
import h5py
import cv2
import numpy as np
from datetime import datetime
from gello.dynamixel import DynamixelDriver
from gello.robots.ur import UR5Robot

class DataCollector:
    def __init__(self, output_dir: str = "data"):
        self.gello = DynamixelDriver(
            port='/dev/ttyUSB0',
            motor_ids=[1,2,3,4,5,6,7]
        )
        self.ur5 = UR5Robot(ip='192.168.1.100')
        self.camera = cv2.VideoCapture(0)
        
        # 加载标定偏移
        self.offsets = np.load('config/gello_offsets.npy')
        
        self.output_dir = output_dir
        self.episode_data = []
        
    def collect_episode(self, task_name: str):
        """采集一个 Episode"""
        
        print(f"\n=== 采集 Episode: {task_name} ===")
        print("按 'r' 开始录制, 's' 停止并保存, 'q' 退出")
        
        recording = False
        episode_data = []
        
        while True:
            # 读取 GELLO
            gello_q = self.gello.get_positions()
            joint_q = gello_q[:6] + self.offsets
            gripper_q = gello_q[6]
            
            # 伺服跟随
            self.ur5.servo_joints(joint_q)
            
            # 读取图像
            ret, frame = self.camera.read()
            
            # 显示
            display_frame = frame.copy()
            if recording:
                cv2.putText(display_frame, "RECORDING", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("GELLO Data Collection", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r') and not recording:
                recording = True
                episode_data = []
                print("开始录制...")
                
            elif key == ord('s') and recording:
                recording = False
                self._save_episode(episode_data, task_name)
                print(f"保存完成，共 {len(episode_data)} 帧")
                
            elif key == ord('q'):
                break
            
            if recording:
                episode_data.append({
                    'timestamp': datetime.now().timestamp(),
                    'joint_positions': joint_q.copy(),
                    'gripper_position': gripper_q,
                    'image': frame.copy()
                })
        
        cv2.destroyAllWindows()
    
    def _save_episode(self, data: list, task_name: str):
        """保存为 HDF5 格式"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/{task_name}_{timestamp}.hdf5"
        
        with h5py.File(filename, 'w') as f:
            n_frames = len(data)
            
            # 创建数据集
            f.create_dataset('timestamps', data=[d['timestamp'] for d in data])
            f.create_dataset('joint_positions', 
                           data=np.array([d['joint_positions'] for d in data]))
            f.create_dataset('gripper_positions',
                           data=np.array([d['gripper_position'] for d in data]))
            
            # 图像 (压缩存储)
            images = np.array([d['image'] for d in data])
            f.create_dataset('images', data=images, compression='gzip')
            
            # 元数据
            f.attrs['task_name'] = task_name
            f.attrs['n_frames'] = n_frames
            f.attrs['robot'] = 'ur5'
            f.attrs['frequency'] = 30  # Hz (估计)
        
        print(f"保存到: {filename}")

if __name__ == "__main__":
    collector = DataCollector(output_dir="data/raw")
    collector.collect_episode("pick_and_place")
```

### 4.2 转换为 LeRobot 格式

```python
# scripts/convert_to_lerobot.py
import h5py
import pandas as pd
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def convert_hdf5_to_lerobot(hdf5_path: str, output_dir: str):
    """将 HDF5 转换为 LeRobot 格式"""
    
    with h5py.File(hdf5_path, 'r') as f:
        n_frames = f.attrs['n_frames']
        task_name = f.attrs['task_name']
        
        # 构建 Episode 数据
        episode = {
            'observation.state': f['joint_positions'][:],
            'action': np.diff(f['joint_positions'][:], axis=0),  # Delta action
            'observation.images.camera': f['images'][:],
        }
        
        # 补齐 action (最后一帧复制)
        episode['action'] = np.vstack([
            episode['action'],
            episode['action'][-1:]
        ])
    
    # 保存为 LeRobot 格式
    # ... (具体实现依赖 LeRobot 版本)
    
    print(f"转换完成: {hdf5_path} -> {output_dir}")
```

### 4.3 批量采集脚本

```bash
#!/bin/bash
# scripts/batch_collect.sh

TASK_NAME=$1
NUM_EPISODES=${2:-10}

echo "=== 批量采集: $TASK_NAME ==="
echo "计划采集 $NUM_EPISODES 个 Episode"

for i in $(seq 1 $NUM_EPISODES); do
    echo ""
    echo "--- Episode $i / $NUM_EPISODES ---"
    python scripts/collect_data.py --task "$TASK_NAME" --episode "$i"
    
    read -p "按 Enter 继续下一个 Episode (或 Ctrl+C 退出)..."
done

echo ""
echo "=== 采集完成 ==="
echo "数据保存在: data/raw/"
```

---

## 5. 与 VLA 训练集成

### 5.1 数据格式对齐

不同 VLA 框架需要不同的数据格式:

| 框架 | 格式 | 转换方法 |
|:---|:---|:---|
| **ACT** | HDF5 (ALOHA 格式) | 直接兼容 |
| **Diffusion Policy** | Zarr | `convert_to_zarr.py` |
| **OpenVLA** | RLDS/LeRobot | `convert_to_lerobot.py` |
| **π0 (OpenPI)** | LeRobot | `convert_to_lerobot.py` |

### 5.2 ACT 训练示例

```python
# 使用 GELLO 采集的数据训练 ACT
# 参考: https://github.com/tonyzhaozh/act

# 1. 数据结构
# data/
# ├── episode_0.hdf5
# ├── episode_1.hdf5
# └── ...

# 2. 训练命令
python train.py \
    --task_name pick_and_place \
    --ckpt_dir checkpoints/gello_act \
    --policy_class ACT \
    --batch_size 8 \
    --num_epochs 2000 \
    --lr 1e-5 \
    --seed 0
```

### 5.3 Diffusion Policy 训练示例

```python
# 使用 GELLO 数据训练 Diffusion Policy
# 参考: https://github.com/real-stanford/diffusion_policy

# 1. 转换数据格式
python scripts/convert_to_zarr.py \
    --input_dir data/raw \
    --output_dir data/zarr

# 2. 训练
python train.py \
    --config-name=train_diffusion_unet_image_workspace \
    task=ur5_gello \
    training.num_epochs=1000
```

---

## 6. 踩坑记录与最佳实践

### 6.1 常见问题

#### 问题 1: Dynamixel 通信超时

**症状**: `[TxRxResult] There is no status packet!`

**原因**: 波特率不匹配或电机 ID 错误

**解决**:
```python
# 检查波特率
BAUDRATE = 1000000  # 默认 1Mbps

# 扫描所有 ID
for motor_id in range(1, 255):
    result = packetHandler.ping(portHandler, motor_id)
    if result[0] == COMM_SUCCESS:
        print(f"Found motor at ID: {motor_id}")
```

#### 问题 2: UR5 跟随抖动

**症状**: 机械臂运动时有明显抖动

**原因**: 伺服增益设置不当

**解决**:
```python
# 调整 servoJ 参数
ur5.rtde_c.servoJ(
    target_q,
    velocity=0.3,        # 降低速度
    acceleration=0.3,    # 降低加速度
    dt=0.008,           # 控制周期
    lookahead_time=0.1, # 增加预见时间
    gain=200            # 降低增益 (100-500)
)
```

#### 问题 3: GELLO 零漂

**症状**: 长时间使用后，GELLO 位置逐渐偏移

**原因**: Dynamixel 电机温度变化导致内部电阻改变

**解决**:
1. 每次使用前重新标定
2. 定期复位到 Home 位置
3. 使用软件低通滤波

```python
class LowPassFilter:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.value = None
    
    def filter(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value
```

#### 问题 4: 延迟过高

**症状**: GELLO 和 UR5 之间有明显延迟

**目标**: 端到端延迟 < 50ms

**优化**:
```python
# 1. 使用更高的通信频率
rtde_frequency = 500  # Hz (而不是 125)

# 2. 减少不必要的处理
# 避免在控制循环中打印日志

# 3. 使用独立线程读取 GELLO
import threading

class AsyncGelloReader:
    def __init__(self):
        self.latest_q = None
        self.lock = threading.Lock()
        
    def _read_loop(self):
        while True:
            q = self.gello.get_positions()
            with self.lock:
                self.latest_q = q
    
    def start(self):
        thread = threading.Thread(target=self._read_loop, daemon=True)
        thread.start()
    
    def get_latest(self):
        with self.lock:
            return self.latest_q.copy()
```

### 6.2 最佳实践

#### 实践 1: 数据质量检查

```python
def validate_episode(hdf5_path: str) -> bool:
    """验证采集的数据质量"""
    
    with h5py.File(hdf5_path, 'r') as f:
        joint_q = f['joint_positions'][:]
        
        # 检查关节速度是否合理
        velocities = np.diff(joint_q, axis=0) * 30  # 假设 30Hz
        max_velocity = np.max(np.abs(velocities))
        
        if max_velocity > 2.0:  # rad/s
            print(f"警告: 检测到异常高速 ({max_velocity:.2f} rad/s)")
            return False
        
        # 检查是否有跳变
        jumps = np.where(np.abs(np.diff(joint_q, axis=0)) > 0.5)[0]
        if len(jumps) > 0:
            print(f"警告: 检测到位置跳变 at frames {jumps}")
            return False
    
    return True
```

#### 实践 2: 安全限位

```python
class SafetyWrapper:
    """安全包装器，防止危险动作"""
    
    def __init__(self, robot):
        self.robot = robot
        
        # UR5 关节限位 (rad)
        self.joint_limits = np.array([
            [-2*np.pi, 2*np.pi],   # Base
            [-2*np.pi, 2*np.pi],   # Shoulder
            [-np.pi, np.pi],       # Elbow
            [-2*np.pi, 2*np.pi],   # Wrist 1
            [-2*np.pi, 2*np.pi],   # Wrist 2
            [-2*np.pi, 2*np.pi],   # Wrist 3
        ])
        
        # 最大关节速度 (rad/s)
        self.max_velocity = 1.0
    
    def safe_servo(self, target_q: np.ndarray):
        # 限位检查
        for i, (low, high) in enumerate(self.joint_limits):
            if target_q[i] < low or target_q[i] > high:
                print(f"警告: Joint {i} 超限! ({target_q[i]:.2f})")
                target_q[i] = np.clip(target_q[i], low, high)
        
        # 速度限制
        current_q = self.robot.get_joint_positions()
        velocity = (target_q - current_q) * 125  # 假设 125Hz
        
        if np.max(np.abs(velocity)) > self.max_velocity:
            scale = self.max_velocity / np.max(np.abs(velocity))
            target_q = current_q + (target_q - current_q) * scale
        
        self.robot.servo_joints(target_q)
```

#### 实践 3: 采集 Checklist

```markdown
## GELLO 数据采集检查清单

### 采集前
- [ ] UR5 已开机并初始化
- [ ] GELLO Dynamixel 电源已连接
- [ ] 相机已连接并测试
- [ ] 运行标定验证脚本
- [ ] 工作空间已清理

### 采集中
- [ ] 确保动作流畅，无急停
- [ ] 避免碰撞和奇异点
- [ ] 每 10 个 Episode 检查数据质量

### 采集后
- [ ] 运行数据验证脚本
- [ ] 备份原始数据
- [ ] 记录采集笔记 (环境、物体、异常)
```

---

## 7. 参考资源

| 资源 | 链接 |
|:---|:---|
| **GELLO 论文** | [arXiv:2309.13037](https://arxiv.org/abs/2309.13037) |
| **软件仓库** | [wuphilipp/gello_software](https://github.com/wuphilipp/gello_software) |
| **硬件 CAD** | [wuphilipp/gello_mechanical](https://github.com/wuphilipp/gello_mechanical) |
| **UR RTDE** | [ur_rtde 文档](https://sdurobotics.gitlab.io/ur_rtde/) |
| **Dynamixel SDK** | [ROBOTIS Dynamixel](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_sdk/overview/) |
| **LeRobot** | [huggingface/lerobot](https://github.com/huggingface/lerobot) |

---

[← Back to Deployment](./README.md)

