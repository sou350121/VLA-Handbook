# UR5/UR5e Python 控制实战指南 (Linux)

> **实战背景**: 在 VLA (Vision-Language-Action) 研究与部署中，UR5/UR5e 是最常见的验证平台。相比于庞大的 ROS 体系，直接使用 Python 接口进行轻量级控制往往更高效，尤其是在数据采集和模型推理阶段。

---

## 1. 环境准备 (Linux Setup)

### 1.1 系统要求
- **OS**: Ubuntu 20.04 / 22.04 LTS (推荐)
- **Kernel**: 标准内核即可，但在高频控制 (500Hz+) 下建议使用 **PREEMPT_RT** 实时内核以减少抖动。
- **Network**: **有线连接**是必须的。Wi-Fi 的抖动会导致 `Protective Stop`。

### 1.2 网络配置
UR 控制器通常静态 IP (如 `192.168.1.100`)。PC 端需配置同网段静态 IP。

```bash
# /etc/netplan/01-network-manager-all.yaml (示例)
network:
  version: 2
  ethernets:
    enp3s0:  # 你的网卡名，用 ip addr 查看
      dhcp4: no
      addresses:
        - 192.168.1.101/24
```
`sudo netplan apply` 生效。
**验证**: `ping 192.168.1.100` 延迟应 < 0.5ms 且无丢包。

---

## 2. Python 库选型

### 2.1 `ur_rtde` (⭐⭐⭐ 强烈推荐)
- **来源**: SDU Robotics (University of Southern Denmark)
- **特点**: 基于 C++ 绑定，性能极高，官方维护，支持 500Hz 通信。
- **安装**: `pip install ur_rtde`

### 2.2 `python-urx` (⚠️ 已过时)
- **特点**: 纯 Python 实现，简单易懂，但**已停止维护**多年。
- **缺点**: 在 Polyscope 5.x+ 上不稳定，不支持 RTDE 高频接口。
- **建议**: 仅用于维护旧代码，新项目勿用。

---

## 3. 实战代码模式 (基于 `ur_rtde`)

### 3.1 基础连接与状态读取
```python
import rtde_receive
import rtde_control
import time

ROBOT_IP = "192.168.1.100"

# 1. 连接接收接口 (读取状态)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
# 2. 连接控制接口 (发送指令)
rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)

# 读取当前关节角度 (rad)
actual_q = rtde_r.getActualQ()
# 读取当前末端位姿 (TCP: x, y, z, rx, ry, rz)
actual_tcp = rtde_r.getActualTCPPose()

print(f"Current Joint: {actual_q}")
```

### 3.2 运动控制模式对比

| 模式 | 指令 | 适用场景 | 备注 |
| :--- | :--- | :--- | :--- |
| **MoveJ** | `moveJ(q, speed, accel)` | 点到点移动 (回 Home) | 关节空间插值，路径非直线 |
| **MoveL** | `moveL(pose, speed, accel)` | 抓取/插拔 | 笛卡尔空间直线运动 |
| **ServoJ** | `servoJ(q, ...)` | **VLA模型推理/遥操作** | 实时流式控制，延迟最低 |
| **SpeedJ** | `speedJ(qd, ...)` | 视觉伺服/摇杆控制 | 速度控制，平滑但需积分 |

#### 实战：使用 ServoJ 进行平滑跟随 (VLA 常用)
模型通常以 10Hz-50Hz 输出目标关节角，直接 `moveJ` 会卡顿，必须用 `servoJ`。

```python
def smooth_servo_loop(target_trajectory):
    # 参数调优是关键
    # velocity/acceleration: 限制最大值防急停
    # lookahead_time: 越小响应越快但易抖动 (0.03-0.2s)
    # gain: 比例增益 (100-500)
    
    dt = 1.0/500  # 2ms 控制周期
    
    for target_q in target_trajectory:
        start = time.time()
        
        rtde_c.servoJ(
            target_q, 
            velocity=0.5, 
            acceleration=0.5, 
            dt=dt, 
            lookahead_time=0.1, 
            gain=300
        )
        
        # 严格控制循环时间
        diff = time.time() - start
        if diff < dt:
            time.sleep(dt - diff)
            
    rtde_c.servoStop()
    rtde_c.stopScript()
```

### 3.3 IO 控制 (夹爪/吸盘)
大部分夹爪 (Robotiq) 或吸盘通过控制器的数字 IO 触发。

```python
# 设置数字输出 DO0 为高电平 (例如：闭合夹爪)
rtde_c.setStandardDigitalOut(0, True)

# 读取数字输入 DI0 (例如：夹爪到位信号)
state = rtde_r.getDigitalInState(0)
```

---

## 4. 踩坑经验 (Troubleshooting)

### 4.1 保护性停止 (Protective Stop)
**现象**: 机器人突然停止，示教器弹窗报错。
**常见原因**:
1. **奇异点 (Singularity)**: 经过肩关节/腕关节奇异区域，导致关节速度无穷大。
   - *解法*: 在路径规划层增加奇异点检测 (`manipulability index`)。
2. **加速度过大**: 模型输出的轨迹不平滑，两帧之间跳变太大。
   - *解法*: 输出端加低通滤波 (LPF) 或插值。
3. **负载未设置**: 抓起重物后未更新 Payload，动力学模型不匹配。
   - *解法*: `rtde_c.setPayload(mass, cog)`

### 4.2 自动复位脚本
在无人值守测试时，自动从 Protective Stop 恢复是必须的。

```python
def try_recover():
    # 检查是否处于保护性停止
    if rtde_r.isProtectiveStopped():
        print("Protective Stop detected! Attempting recovery...")
        # 1. 解锁保护性停止
        rtde_c.unlockProtectiveStop() 
        # 2. 重新上电并松刹车
        # 注意: 需要足够的延时
        return True
    return False
```

### 4.3 实时性与抖动
**现象**: 机器人动作一卡一卡。
**原因**: Python 垃圾回收 (GC) 或网络抖动导致控制周期不稳定。
**优化**:
- 关闭 GC: `gc.disable()` (在关键循环中)。
- 绑核: `taskset -c 0 python control.py`。
- **不要在控制循环中 print**: I/O 是最大的延迟源。

---

## 5. 进阶：结合 VLA 模型
在 VLA 系统中，通常架构如下：

1. **Inference 线程**: GPU 推理，输出 `target_q` 写入队列。
2. **Control 线程**: 500Hz 从队列读取最新 `target_q`，执行 `servoJ`。

```python
import threading
import queue

q_queue = queue.Queue(maxsize=1)

def control_thread():
    while True:
        if not q_queue.empty():
            target = q_queue.get()
            rtde_c.servoJ(target, ..., lookahead_time=0.1)
        else:
            # 队列空时保持当前姿态 (防止掉电)
            rtde_c.servoJ(rtde_r.getActualQ(), ...)
        time.sleep(0.002)

# 启动控制线程
t = threading.Thread(target=control_thread)
t.start()

# 主线程跑模型
while True:
    img = camera.read()
    action = model.predict(img)
    q_queue.put(action)  # 仅放入最新动作
```

---

## 6. 进阶阅读
关于 ROS 集成、OOP 架构设计以及高频控制的性能优化技巧，请参考：
👉 **[ROS 集成与算法优化 (ROS Integration & Algorithm Optimization)](./ros_and_optimization.md)**



