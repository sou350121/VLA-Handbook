# 状态估计与传感器融合 (State Estimation & Sensor Fusion)

> **面试场景**: “如何利用 IMU + 相机做状态估计？Kalman Filter 与 Particle Filter 有哪些区别？”

本章整理 Kalman/EKF/UKF、Particle Filter 以及多传感器融合的工程实践，帮助在面试中自信回答状态估计相关问题。

---

## 🧭 状态估计任务分层

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         状态估计层次                                        │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   低层 (IMU 惯导)                     中层 (组合导航)                    │
│   ───────────────                     ───────────────                    │
│   • 姿态/角速度估计                   • VIO/VINS (IMU + Camera)          │
│   • 零偏校正                          • Wheel Odometry + IMU             │
│                                                                          │
│   高层 (全局定位)                     语义层 (高阶)                      │
│   ───────────────                     ───────────────                    │
│   • SLAM (EKF-SLAM, Graph-SLAM)       • 对象级跟踪                       │
│   • GNSS-RTK + 惯导组合               • 语义状态估计                     │
│                                                                          │
│   核心构件:                                                                │
│   - Motion Model (系统模型)                                               │
│   - Sensor Model (观测模型)                                               │
│   - Filter / Optimizer                                                    │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Kalman Filter 家族

### 1.1 线性 KF

系统模型 (离散):
\[
x_k = A x_{k-1} + B u_k + w_k,\quad w_k \sim \mathcal{N}(0, Q)
\]
观测模型:
\[
z_k = H x_k + v_k,\quad v_k \sim \mathcal{N}(0, R)
\]

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         Kalman Filter 循环                                  │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   预测 (Predict)                                                         │
│   ────────────────────────────────────────────────────────────────────   │
│   x̂⁻ₖ = A x̂ₖ₋₁ + B uₖ                                                   │
│   P⁻ₖ = A Pₖ₋₁ Aᵀ + Q                                                    │
│                                                                          │
│   更新 (Update)                                                          │
│   ────────────────────────────────────────────────────────────────────   │
│   Kₖ = P⁻ₖ Hᵀ (H P⁻ₖ Hᵀ + R)⁻¹                                           │
│   x̂ₖ = x̂⁻ₖ + Kₖ (zₖ - H x̂⁻ₖ)                                           │
│   Pₖ = (I - Kₖ H) P⁻ₖ                                                    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 1.2 扩展 Kalman Filter (EKF)

- 适用于非线性系统：\( x_k = f(x_{k-1}, u_k) + w_k \), \( z_k = h(x_k) + v_k \)
- 用雅可比矩阵线性化：
  - \( F_k = \frac{\partial f}{\partial x} \big|_{x=\hat{x}_{k-1}} \)
  - \( H_k = \frac{\partial h}{\partial x} \big|_{x=\hat{x}_{k}^{-}} \)
- 缺点：线性化误差大时会发散，需要保持小时间步或使用重线性化。

### 1.3 无迹 Kalman Filter (UKF)

- 不再线性化，而是用 **Sigma Points** 捕获均值/协方差传播。
- 步骤：
  1. 选择 \( 2n + 1 \) 个 sigma 点 \( \chi_i \)
  2. 通过非线性函数传播 \( \chi_i' = f(\chi_i) \)
  3. 加权重构均值 \(\hat{x}\) 与协方差 \(P\)
- 优点：对高度非线性的系统更稳健，不需要求雅可比。
- 缺点：计算量稍高，参数 (α, β, κ) 需调节。

### 1.4 选择指南

| 场景 | 推荐滤波器 | 备注 |
|:-----|:-----------|:-----|
| 线性系统 + 高频更新 | KF | 如电机位置闭环 |
| 轻微非线性 + 高速运行 | EKF | 航姿系统、VIO |
| 强非线性 + 噪声复杂 | UKF | 视觉角度 + IMU |

---

## 2. 粒子滤波 (Particle Filter)

### 2.1 原理

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     Particle Filter 流程                                   │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│ 1. 采样: 从上一时刻粒子集合根据运动模型采样新的粒子                      │
│ 2. 加权: 根据观测模型计算粒子权重 (似然)                                 │
│ 3. 归一化: 权重归一化                                                     │
│ 4. 重采样: 根据权重重新采样，避免退化                                    │
│ 5. 状态估计: 求加权平均或选最大权重粒子                                 │
│                                                                          │
│   常用技巧:                                                               │
│   - 低方差重采样 (Low-variance resampling)                               │
│   - 自适应粒子数 (KLD-Sampling)                                          │
│   - Log-Likelihood 累积避免下溢                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.2 KF vs Particle Filter

| 对比项 | Kalman 系列 | Particle Filter |
|:-------|:------------|:----------------|
| 状态分布假设 | 高斯 | 任意分布 |
| 维度扩展 | 易受维度诅咒影响较小 | 粒子数随维度指数增加 |
| 计算量 | 低 (矩阵运算) | 高 (大量粒子) |
| 收敛速度 | 快 | 慢 (依赖粒子权重) |
| 适用场景 | 机器人姿态、VIO | 全局定位、非高斯噪声 |

---

## 3. 传感器融合模式

### 3.1 常见组合

| 组合 | 说明 | 典型系统 |
|:-----|:-----|:---------|
| IMU + Encoder | 惯性测量 + 轮速，补偿滑移 | 地面移动机器人 |
| IMU + Camera | Visual-Inertial Odometry (VIO) | VINS-Mono, OKVIS |
| IMU + LiDAR | LiDAR-Inertial Odometry | LIO-SAM |
| Wheel + GNSS + IMU | 车辆定位 | 自动驾驶 |
| Vision + Tactile | 末端执行器状态估计 | 触觉闭环控制 |

### 3.2 融合架构

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     典型 VIO (EKF-based)                                   │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   IMU 高频积分 → 状态预测                                                 │
│          │                                                               │
│          ▼                                                               │
│   EKF 预测步骤 (Propagate)                                               │
│          │                                                               │
│   相机关键帧提取 → 特征匹配                                              │
│          │                                                               │
│          ▼                                                               │
│   EKF 更新步骤 (Update)                                                   │
│          │                                                               │
│          ▼                                                               │
│   输出: 位姿、速度、偏置                                                 │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Graph-based 融合

- 使用因子图 (Factor Graph) 或 Bundle Adjustment 表述状态估计。
- 节点: 机器人状态 (姿态、速度、偏置)。
- 因子: IMU 预积分、视觉重投影误差、里程计约束等。
- 优点：可离线批处理 / 滑动窗口，精度高；缺点：实时性需优化。

---

## 4. 代码实现片段

### 4.1 PyTorch EKF 结构

```python
import torch

class ExtendedKalmanFilter:
    def __init__(self, state_dim, meas_dim):
        self.state_dim = state_dim
        self.meas_dim = meas_dim
        self.x = torch.zeros(state_dim, 1)
        self.P = torch.eye(state_dim)
        self.Q = torch.eye(state_dim) * 1e-3
        self.R = torch.eye(meas_dim) * 1e-2

    def predict(self, f, F, u=torch.zeros(0)):
        # f: 状态转移函数, F: 雅可比
        self.x = f(self.x, u)
        self.P = F(self.x) @ self.P @ F(self.x).T + self.Q

    def update(self, z, h, H):
        y = z - h(self.x)
        S = H(self.x) @ self.P @ H(self.x).T + self.R
        K = self.P @ H(self.x).T @ torch.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (torch.eye(self.state_dim) - K @ H(self.x)) @ self.P
```

### 4.2 粒子滤波 Python 原型

```python
import numpy as np

class ParticleFilter:
    def __init__(self, num_particles, motion_model, sensor_model):
        self.n = num_particles
        self.motion_model = motion_model
        self.sensor_model = sensor_model
        self.particles = np.zeros((num_particles, 3))  # x, y, theta
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, u, dt):
        noise = np.random.normal(0, [0.01, 0.01, 0.005], size=self.particles.shape)
        self.particles = self.motion_model(self.particles, u, dt) + noise

    def update(self, z):
        self.weights *= self.sensor_model.likelihood(self.particles, z)
        self.weights += 1e-300  # 避免全 0
        self.weights /= np.sum(self.weights)

    def resample(self):
        cumulative = np.cumsum(self.weights)
        cumulative[-1] = 1.0
        indexes = np.searchsorted(cumulative, np.random.rand(self.n))
        self.particles = self.particles[indexes]
        self.weights.fill(1.0 / self.n)

    def estimate(self):
        return np.average(self.particles, weights=self.weights, axis=0)
```

---

## 5. 工程实践

### 5.1 预积分 (IMU Pre-integration)

- Avoid integrating IMU between every pair of frames.
- 在 VIO 中预计算 IMU 约束，使优化只跟状态增量相关。
- 使用 `gtsam::PreintegratedImuMeasurements` 等工具。

### 5.2 零偏估计 (Bias Estimation)

- IMU 零偏随时间漂移，必须纳入状态向量：
  \[
  x = \begin{bmatrix} p & v & q & b_a & b_g \end{bmatrix}
  \]
- 噪声模型中添加随机游走：
  \[
  b_{k} = b_{k-1} + w_b
  \]

### 5.3 观测异常检测

- **Mahalanobis Distance**: 判断测量是否异常：
  \[
  \nu = z - h(\hat{x});\quad d^2 = \nu^\top S^{-1} \nu
  \]
  若 \( d^2 > \chi^2_{n,\alpha} \)，则拒绝该观测。
- **Innovation Gating**: 只在创新位于阈值内时更新。

### 5.4 ROS2 / robot_localization

- `ekf_node`: 融合 IMU + Wheel + GPS。
- `ukf_node`: 支持 UKF。
- 需要正确配置 `imu0_config`, `odom0_config`, `two_d_mode` 等参数。

---

## 6. 面试 Q&A

### Q1: EKF 为什么可能会发散？如何缓解？

- 线性化误差大：使用小时间步、重复线性化或改用 UKF。
- 噪声协方差设定不合理：增大 Q/R 使滤波更保守。
- 初始协方差过小：导致信任预测过度，建议放大 \(P_0\)。

### Q2: Particle Filter 如何选择粒子数？

- 经验：每维 50~100 个粒子。
- 可用 **自适应粒子数**：当 ESS (effective sample size) 高于阈值时减少粒子，低于时增加。

### Q3: IMU + 相机融合的关键难点？

1. 时间同步 (时间戳对齐，PTP/触发)。
2. IMU 到相机的外参标定 (手眼标定)。
3. IMU 噪声模型 (加速度/陀螺零偏、随机游走)。
4. 滑动窗口优化的准实时实现 (Ceres / GTSAM)。

### Q4: 如何检测传感器失效？

- 监控创新 (Innovation) 大小。
- 监控输出方差是否异常增大。
- 多传感器互相验证 (Cross-check)：例如视觉失败时降级为轮速 + IMU。

---

## 📚 推荐资源

- *Probabilistic Robotics* — Thrun et al.
- *State Estimation for Robotics* — Timothy D. Barfoot
- VINS-Mono: [github.com/HKUST-Aerial-Robotics/VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono)
- robot_localization: [docs.ros.org/en/noetic/api/robot_localization](http://docs.ros.org/en/noetic/api/robot_localization/html/index.html)
- GTSAM: [gtsam.org](https://gtsam.org/)

---

[← Back to Theory Index](./README.md)

