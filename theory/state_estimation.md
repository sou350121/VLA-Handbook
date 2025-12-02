# 状态估计与传感器融合 (State Estimation & Sensor Fusion)

> **面试场景**: “如何利用 IMU + 相机做状态估计？Kalman Filter 与 Particle Filter 有哪些区别？”

---

## 🧭 状态估计分层

```
┌──────────────────────────────────────────────────────────────────────────┐
│   IMU 惯导 (低层) → VIO/VINS (中层) → SLAM/GNSS 融合 (高层)              │
└──────────────────────────────────────────────────────────────────────────┘
```

- **系统模型**: 状态随时间演化 (运动模型)
- **观测模型**: 传感器输出与状态之间的关系
- **滤波/优化器**: KF 家族、Particle Filter、因子图

---

## 1. Kalman Filter 家族

| 滤波器 | 适用场景 | 特点 |
|:-------|:---------|:-----|
| KF | 线性系统 | 闭式解、计算量小 |
| EKF | 弱非线性 | 线性化，需雅可比 |
| UKF | 强非线性 | Sigma Points，无需雅可比 |

### 1.1 EKF 推导

- 预测: \( x^- = f(x) \); 协方差 \( P^- = FPF^T + Q \)
- 更新: \( K = P^- H^T (HP^-H^T + R)^{-1} \)
- 状态: \( x = x^- + K(z - h(x^-)) \)

### 1.2 UKF 要点

- 选择 sigma 点，将其通过非线性函数传播
- 重新组合得到新的均值/协方差
- 对高度非线性的姿态/四元数系统更稳定

---

## 2. 粒子滤波 (Particle Filter)

1. 根据运动模型采样粒子
2. 计算观测似然作为权重
3. 重采样以避免粒子退化
4. 估计状态 (加权平均或最高权重)

优点：可表示任意分布；缺点：高维下粒子数爆炸。

---

## 3. 传感器融合模式

| 组合 | 描述 | 代表系统 |
|:-----|:-----|:---------|
| IMU + Camera | Visual-Inertial Odometry | VINS-Mono, OKVIS |
| IMU + LiDAR | LiDAR-Inertial SLAM | LIO-SAM |
| Wheel + IMU | 移动底盘 | robot_localization |
| IMU + GNSS | 车辆导航 | RTK/INS |

### 3.1 VIO 流程

```
IMU 高频积分 → EKF 预测
      ↑              ↓
相机关键帧 → 特征匹配 → EKF 更新
```

### 3.2 因子图优化

- 节点：状态 (位姿、速度、偏置)
- 因子：IMU 预积分、视觉重投影、里程计
- 使用 GTSAM/Ceres 做滑动窗口优化

---

## 4. 工程技巧

- **预积分**：减少重复积分，IMU 约束仅依赖状态增量
- **零偏建模**：将加速度/陀螺零偏纳入状态
- **创新检验**：Mahalanobis 距离判断观测是否异常
- **时间同步**：硬件触发或 PTP，确保 IMU/相机时钟一致
- **robot_localization**：ROS2 EKF/UKF 节点，配置 `imu0_config` 等参数

---

## 5. 代码片段

```python
# 简化 EKF 结构
class EKF:
    def __init__(self, n, m):
        self.x = np.zeros((n, 1))
        self.P = np.eye(n)
        self.Q = np.eye(n) * 1e-3
        self.R = np.eye(m) * 1e-2

    def predict(self, f, F, u):
        self.x = f(self.x, u)
        self.P = F(self.x) @ self.P @ F(self.x).T + self.Q

    def update(self, z, h, H):
        y = z - h(self.x)
        S = H(self.x) @ self.P @ H(self.x).T + self.R
        K = self.P @ H(self.x).T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(len(self.x)) - K @ H(self.x)) @ self.P
```

```python
# 粒子滤波骨架
class ParticleFilter:
    def __init__(self, n, motion, sensor):
        self.particles = np.zeros((n, 3))
        self.weights = np.ones(n) / n
        self.motion = motion
        self.sensor = sensor

    def predict(self, u, dt):
        noise = np.random.normal(0, [0.01, 0.01, 0.005], self.particles.shape)
        self.particles = self.motion(self.particles, u, dt) + noise

    def update(self, z):
        self.weights *= self.sensor.likelihood(self.particles, z)
        self.weights += 1e-300
        self.weights /= np.sum(self.weights)

    def resample(self):
        idx = np.random.choice(len(self.particles), len(self.particles), p=self.weights)
        self.particles = self.particles[idx]
        self.weights.fill(1.0 / len(self.weights))
```

---

## 6. 面试 Q&A

1. **EKF 为什么会发散？** 线性化误差大、噪声矩阵不准；可减小时间步或采用 UKF。
2. **IMU + 相机融合难点？** 时间同步、外参标定、IMU 噪声建模、滑窗优化计算量。
3. **如何检测传感器失效？** 监控创新、输出方差，或多传感器互检。
4. **粒子数量如何选？** 根据维度和覆盖范围，使用 ESS 自适应调整。

---

## 📚 推荐

- *Probabilistic Robotics*
- *State Estimation for Robotics*
- VINS-Mono / VINS-Fusion
- robot_localization

---

[← Back to Theory Index](./README.md)
