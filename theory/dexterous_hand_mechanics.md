# 灵巧手机械学深度解析 (Dexterous Hand Mechanics)

灵巧手是机械电子工程中集成度最高的部件之一。要在接近人手尺寸的体积内实现 20+ 自由度，需要综合运用多种机械原理。

---

## 1. 机构学：构型与自由度 (Mechanism & DOF)

### 1.1 自由度分配 (Degrees of Freedom)

*   **计算公式 (Grubler's Criterion)**：
    对于三维空间的机构，自由度 $M$ 的计算如下：

    $$
    M = 6(n - j - 1) + \sum_{i=1}^{j} f_i
    $$

    - $n$: 连杆总数（含机架）
    - $j$: 关节总数
    - $f_i$: 第 $i$ 个关节的自由度

*   **仿生构型**：主流灵巧手模仿人手的 21-27 个自由度。
    *   **手指 (Fingers)**：通常为 3-4 自由度（弯曲、侧摆）。
    *   **拇指 (Thumb)**：最为关键，通常需要 4-5 自由度以实现**对掌 (Opponency)**。

*   **全驱动 vs. 欠驱动 (Under-actuated)**：
    *   **全驱动**：每个关节由独立电机驱动（如 Wuji, Sharpa）。优势是控制绝对精准，劣势是重量大、成本极高。
    *   **欠驱动**：利用滑轮、拉索或弹簧使一个电机带动多个关节。优势是具备**结构柔顺性 (Structural Compliance)**，能自动包络物体。

---

## 2. 传动学：动力输出的艺术 (Transmission)

### 2.1 精密减速器 (Reducers)

*   **物理关系**：减速比 $G$ 直接决定了输出扭矩 $\tau_{out}$ 与电机扭矩 $\tau_{in}$ 的关系：

    $$
    \tau_{out} = G \cdot \eta \cdot \tau_{in}, \quad \omega_{out} = \frac{\omega_{in}}{G}
    $$

    - $G$: 减速比 (Gear Ratio)
    - $\eta$: 传动效率 (Efficiency)

*   **行星减速器 (Planetary Gear)**：高功率密度，结构紧凑。
*   **齿轮传动 (Gear Train)**：
    *   **锥齿轮 (Bevel Gear)**：用于 90 度转角传动。
    *   **蜗轮蜗杆 (Worm Gear)**：具有自锁功能，但效率较低。

---

## 3. 运动学与空间计算 (Kinematics)

### 3.1 坐标系映射 (Coordinate Mapping)

*   **正向运动学 (FK)**：从电机角度计算指尖坐标 ($x, y, z$)。
*   **逆向运动学 (IK)**：给定目标抓取点，计算各个电机的转角。

### 3.2 雅可比矩阵 (Jacobian)

雅可比矩阵是关节空间 $q$ 与笛卡尔空间 $x$ 之间的差分映射：

*   **速度映射**：

    $$
    \dot{x} = J(q)\dot{q}
    $$

*   **静力学对偶性 (Statics Duality)**：
    这是触觉反馈控制的核心公式。它将末端接触力 $f_{ext}$ 转换为关节力矩 $\tau$：

    $$
    \tau = J^T(q)f_{ext}
    $$

*   **物理意义**：
    - $J(q)$：决定了手指在当前姿态下各方向的运动灵敏度。
    - $J^T$：决定了各关节电机需要输出多少力矩来维持指尖的抓取力。
    - **奇异状态 (Singularity)**：当 $det(J) = 0$ 时，手指在某些方向失去自由度。

---

## 4. 动力学与控制特性 (Dynamics & Control)

### 4.1 动力学基本方程 (Equations of Motion)

灵巧手的运动遵循欧拉-拉格朗日方程：

$$
M(q)\ddot{q} + C(q, \dot{q})\dot{q} + G(q) = \tau - J^T f_{ext}
$$

- $M(q)$: 惯性矩阵 (Inertia Matrix)
- $C(q, \dot{q})$: 离心力与科氏力 (Coriolis & Centrifugal)
- $G(q)$: 重力项 (Gravity)

### 4.2 阻抗控制 (Impedance Control)

为了实现“柔顺抓取”，通常在任务空间建立二阶阻抗模型：

$$
f_{ext} = M_d(\ddot{x}_d - \ddot{x}) + D_d(\dot{x}_d - \dot{x}) + K_d(x_d - x)
$$

其对应的关节控制输出 $\tau$ 为：

$$
\tau = J^T(q)[M_d(\Delta\ddot{x}) + D_d(\Delta\dot{x}) + K_d(\Delta x)] + \hat{G}(q)
$$

- $K_d, D_d$: 期望刚度和阻尼，决定了手指的“弹性”程度。

---

## 5. 总结：DexHand 的三大工程悖论

1.  **高自由度 vs. 有限空间**：如何在成人手指粗细的管内塞进电机和减速器。
2.  **大抓取力 vs. 极小自重**：力矩密度 ($Torque Density$) 的极限挑战。
3.  **高精度控制 vs. 物理鲁棒性**：如何保证手指既能穿针引线，又在撞击桌面时不崩齿。

---
[← 返回理论目录](./README.md)
