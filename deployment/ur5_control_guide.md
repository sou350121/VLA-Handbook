# UR5/UR5e 机械臂 Python 控制实战指南 (Linux)

> **适用机型**: UR3, UR3e, UR5, UR5e, UR10, UR10e, UR16e
> **核心库**: `ur_rtde` (推荐), `urx` (已过时但简单), `dashboard_client`

---

## 0. 为什么选择 Linux + Python?

1. **实时性**: Linux 可以配置实时内核 (RT Preempt)，满足 VLA 模型对低延迟 (125Hz - 500Hz) 的要求。
2. **生态系统**: `ur_rtde` 库提供 C++/Python 绑定，直接与 UR 控制器的 RTDE 接口通信，避开了 ROS 的复杂性。
3. **部署简单**: 相比 ROS2 繁琐的编译和环境配置，Python + `ur_rtde` 可以实现“即装即用”。

---

## 1. 硬件连接与配置

### 1.1 网络连接

1. 使用网线将电脑与 UR 控制箱的 LAN 口连接。
2. **电脑端设置**:
   - 设置静态 IP: `192.168.1.10`
   - 子网掩码: `255.255.255.0`
   - 网关: `0.0.0.0`
3. **UR 示教器设置**:
   - 设置 → 系统 → 网络。
   - 设置静态 IP: `192.168.1.100` (需与电脑在同一网段)。

### 1.2 启用远程控制

1. 设置 → 系统 → 远程控制 → 选择 **“启用”**。
2. 安装 → 外部控制 (需安装 URCap `External Control`)。
   - 主机 IP: `192.168.1.10`
   - 端口: `50002`

---

## 2. 软件环境安装

### 2.1 安装 `ur_rtde` (推荐)

`ur_rtde` 是目前最稳定、功能最全的 UR 控制库。

```bash
# Ubuntu 安装依赖
sudo apt-get install libboost-all-dev

# 安装 Python 库
pip install ur_rtde
```

### 2.2 安装 Dashboard Client

用于控制机械臂的上电、刹车释放、加载程序等操作。

```bash
pip install ur_dashboard_control
```

---

## 3. 核心控制逻辑

### 3.1 基础连接与状态读取

```python
import rtde_receive
import rtde_control

ROBOT_IP = "192.168.1.100"

# 初始化接收和控制接口
# 默认频率: e-series 500Hz, CB3 125Hz
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)

# 读取关节位置 (rad)
actual_q = rtde_r.getActualQ()
print(f"当前关节角度: {actual_q}")

# 读取 TCP 笛卡尔位姿 (x, y, z, rx, ry, rz)
actual_tcp = rtde_r.getActualTCPPose()
print(f"当前末端位姿: {actual_tcp}")

# 读取 TCP 受力 (Fx, Fy, Fz, Mx, My, Mz)
actual_force = rtde_r.getActualTCPForce()
print(f"当前受力: {actual_force}")
```

### 3.2 关节空间控制 (`moveJ`)

适用于大幅度轨迹移动或复位。

```python
# target_q: [base, shoulder, elbow, wrist1, wrist2, wrist3] (rad)
target_q = [0, -1.57, 1.57, -1.57, -1.57, 0]

# 参数: 目标位置, 速度(rad/s), 加速度(rad/s^2), 异步(True/False)
rtde_c.moveJ(target_q, speed=0.5, acceleration=1.0)
```

### 3.3 笛卡尔空间控制 (`moveL`)

适用于直线插补运动。

```python
# target_pose: [x, y, z, rx, ry, rz] (单位: m, rad)
target_pose = [0.4, 0.1, 0.4, 0, 3.14, 0]

# 参数: 目标位姿, 速度(m/s), 加速度(m/s^2), 异步
rtde_c.moveL(target_pose, speed=0.2, acceleration=0.5)
```

### 3.4 VLA 核心: 伺服模式 (`servoJ`)

**VLA 模型每 8ms-20ms 推理一次动作，`servoJ` 是实现实时跟随的关键。**

```python
import time

# 模拟 VLA 循环
while True:
    # 1. 假设 VLA 模型输出了下一个目标关节位置
    # next_q = vla_model.predict(observation)
    
    # 2. 发送伺服指令
    # lookahead_time: 0.03-0.2 (平滑度), gain: 100-2000 (跟踪紧密度)
    rtde_c.servoJ(next_q, velocity=0, acceleration=0, 
                  dt=0.008, lookahead_time=0.1, gain=300)
    
    # 3. 严格控制循环时间 (例如 125Hz)
    time.sleep(0.008)
```

---

## 4. 夹爪控制 (Robotiq 示例)

UR 控制夹爪通常有两种方式：通过控制器 IO 或通过透明传输。

```python
# 示例: 通过 IO 控制简单的二进制夹爪 (开/关)
# 设置数字输出口 0
rtde_c.setStandardDigitalOut(0, True)  # 闭合
time.sleep(0.5)
rtde_c.setStandardDigitalOut(0, False) # 打开
```

如果是 Robotiq 2F-85/140，通常使用 `robotiq_gripper` 库或编写脚本通过 TCP 端口 `63352` 发送指令。

---

## 5. 安全与异常处理

```python
try:
    # 检查机器人模式
    if rtde_r.getRobotMode() != 7: # 7 是 RUNNING 模式
        print("机器人未就绪，请先在示教器上启动程序")
    
    # 执行控制
    rtde_c.moveJ(...)

except Exception as e:
    print(f"发生错误: {e}")
    # 紧急停止
    rtde_c.stopJ(2.0)
finally:
    # 断开连接
    rtde_c.disconnect()
    rtde_r.disconnect()
```

---

## 6. 面试常见问题 (Deployment/Control)

**Q1: 如何保证 Python 控制 UR 的实时性？**
- **A**: 使用 `ur_rtde` 接口，它基于 RTDE (Real-Time Data Exchange) 协议，工作在控制器的优先级线程。同时，Python 代码应部署在配置了 `PREEMPT_RT` 内核的 Linux 系统上，并尽可能减少循环内的复杂计算（如图像处理应放在独立进程）。

**Q2: 如果网络波动导致丢包，机械臂会发生什么？**
- **A**: `ur_rtde` 有看门狗 (Watchdog) 机制。如果在规定时间内 (如 10ms) 没有收到新指令，机械臂会根据安全配置执行保护性停止。

**Q3: `servoJ` 和 `speedJ` 有什么区别？**
- **A**: `servoJ` 是位置跟随，给出一个目标位置，机械臂会在 `dt` 时间内尝试到达；`speedJ` 是速度跟随，给出一个关节速度向量，机械臂会持续以该速度运行直到收到新指令或停止指令。VLA 任务中通常使用 `servoJ` 以获得更好的位置确定性。

---

[← Back to Deployment](./README.md)
