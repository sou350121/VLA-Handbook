# 空间智能与坐标系 (Spatial Intelligence & Coordinate Systems)

对于 AI 背景的同学来说，机器人学最令人头大的往往不是深度学习模型，而是**坐标变换 (Coordinate Transformations)**。理解空间关系是训练 VLA 模型的基础。

## 0. 主要数学思想 (Main Mathematical Idea)

> **第一性原理**: **Rigid Body Invariance (刚体不变性 / Lie Group SE(3))**

物理世界中的物体是"刚性"的，移动或旋转它们不会改变它们的形状或内部距离。这意味着我们不能用普通的加法来处理旋转（$R_1 + R_2$ 没有物理意义），而必须遵循**群论 (Group Theory)** 的规则。

- **核心数学工具**: **Lie Groups (李群 SE(3))** 与 **Lie Algebras (李代数 se(3))**。
- **解题逻辑**:
    1.  **非欧几何**: 旋转空间是弯曲的（流形）。为了在神经网络中处理它，我们需要找到合适的表示（如四元数、6D表示），以避免奇异性（万向节死锁）和不连续性。
    2.  **相对性**: 所有的位姿都是相对的 ($T_A^B$)。数学核心是**链式法则** ($T_A^C = T_A^B \times T_B^C$)，这对应于在不同坐标系之间无损地传递信息。

## 1. 核心坐标系 (Core Frames)

在机器人操作中，我们必须时刻清楚数据是在哪个坐标系下定义的。

### 1.1. World Frame (世界坐标系) $\{W\}$
- **定义**: 全局固定的参考系，通常是机器人底座固定的桌子角，或者房间的某个角落。
- **作用**: 多机器人协作时的统一基准。

### 1.2. Base Frame (基座坐标系) $\{B\}$
- **定义**: 机器人底座 (Base Link) 的中心。
- **作用**: **绝大多数单臂机器人的 Action 都是相对于 Base Frame 定义的**。
- **注意**: 如果机器人是移动的 (Mobile Manipulator)，Base Frame 本身是在 World Frame 中移动的。

### 1.3. End-effector Frame (末端坐标系 / TCP) $\{E\}$
- **定义**: 机械臂末端执行器 (Gripper) 的中心点 (Tool Center Point, TCP)。通常 $Z$ 轴指向夹爪延伸方向，$X$ 轴指向夹爪闭合方向。
- **作用**: 描述“手”的位置和朝向。
- **Delta Action**: 很多时候模型预测的是 TCP 相对于**当前时刻 TCP** 的移动量 (即在自己的坐标系下往前走 1cm)。

### 1.4. Camera Frame (相机坐标系) $\{C\}$
- **定义**: 以相机光心为原点，$Z$ 轴通常指向前方，$X$ 轴向右，$Y$ 轴向下 (OpenCV 标准)。
- **作用**: 视觉输入 (RGB/Depth) 最初都是在 Camera Frame 下的。

---

## 2. 齐次变换矩阵 (Homogeneous Transformation Matrix)

为了统一描述旋转 (Rotation) 和平移 (Translation)，我们使用 $4 \times 4$ 的齐次变换矩阵 $T$。

$$
T_{A}^{B} = \begin{bmatrix} R_{3\times3} & t_{3\times1} \\ 0_{1\times3} & 1 \end{bmatrix} \in SE(3)
$$

- **物理含义**: 描述了坐标系 $\{A\}$ 相对于坐标系 $\{B\}$ 的位姿。
- **点的变换**: 如果点 $P$ 在 $\{A\}$ 中的坐标是 $P_A = [x, y, z, 1]^T$，那么它在 $\{B\}$ 中的坐标是：

$$
P_B = T_{A}^{B} \times P_A
$$

### 2.1. 逆变换 (Inverse)
求逆矩阵对应于反向变换 $T_{B}^{A} = (T_{A}^{B})^{-1}$：

$$
(T_{A}^{B})^{-1} = \begin{bmatrix} R^T & -R^T t \\ 0 & 1 \end{bmatrix}
$$

> **注意**: 旋转矩阵是正交矩阵，所以 $R^{-1} = R^T$，计算非常快。

### 2.2. 链式法则 (Chain Rule)
这是机器人学中最强大的工具。例如，已知相机相对于世界的外参 $T_{C}^{W}$，以及物体相对于相机的位姿 $T_{O}^{C}$，求物体在世界系下的位姿：

$$
T_{O}^{W} = T_{C}^{W} \times T_{O}^{C}
$$

---

## 3. 旋转表示深度解析 (Rotation Representations Deep Dive)

位置 $t \in \mathbb{R}^3$ 很好表示，但旋转 $R \in SO(3)$ 是非线性的流形结构，神经网络很难直接预测。

### 3.1. 欧拉角 (Euler Angles)
- **表示**: $(roll, pitch, yaw)$ 或 $(\alpha, \beta, \gamma)$。
- **致命缺陷**: **万向节死锁 (Gimbal Lock)**。当中间轴旋转 90 度时，第一轴和第三轴重合，丢失一个自由度。
- **不连续性**: $\pi$ 和 $-\pi$ 是同一个角度，但数值相差巨大。这会导致 Loss 爆炸。
- **结论**: ❌ **VLA 模型严禁直接预测欧拉角**。

### 3.2. 四元数 (Quaternion)
- **表示**: $q = w + xi + yj + zk$，通常写为向量 $[w, x, y, z]$。
- **性质**: 必须满足单位模约束 $\|q\|_2 = 1$。
- **双倍覆盖 (Double Cover)**: $q$ 和 $-q$ 表示完全相同的旋转。
    - **训练坑点**: 如果真值是 $q$，模型预测的是 $-q$ (实际上是对的)，但 MSE Loss 会很大。
    - **解决方案**: 在计算 Loss 前，如果 $\langle q_{pred}, q_{gt} \rangle < 0$，则将 $q_{gt}$ 翻转为 $-q_{gt}$。
- **结论**: ✅ **主流选择** (如 RT-1)，但需要处理归一化和双倍覆盖。

### 3.3. 6D 旋转表示 (6D Rotation Representation) [SOTA]
- **来源**: *On the Continuity of Rotation Representations in Neural Networks* (Zhou et al., CVPR 2019).
- **核心思想**: 神经网络预测 $3 \times 3$ 旋转矩阵的前两列 $r_1, r_2$ (共 6 个数)。
    - $r_1$ 是 $X$ 轴方向 (未归一化)。
    - $r_2$ 是 $Y$ 轴方向 (未归一化，且不一定垂直于 $X$)。
- **还原步骤 (Gram-Schmidt 正交化)**:
    1. 归一化 $x = \frac{r_1}{\|r_1\|}$
    2. 计算 $z = x \times r_2$ (叉乘得到垂直于 $x, r_2$ 平面的轴)
    3. 归一化 $z = \frac{z}{\|z\|}$
    4. 计算 $y = z \times x$ (得到完美的正交系)
    5. 组装 $R = [x, y, z]$
- **优点**: **连续性 (Continuity)**。这是唯一一种在欧几里得空间中连续的旋转表示，最适合神经网络回归。
- **结论**: ✅✅ **Diffusion Policy / Pi0 的首选**。

#### PyTorch 实现
```python
import torch
import torch.nn.functional as F

def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:, 0:3]
    y_raw = ortho6d[:, 3:6]
    
    x = F.normalize(x_raw, dim=-1)
    z = torch.cross(x, y_raw, dim=-1)
    z = F.normalize(z, dim=-1)
    y = torch.cross(z, x, dim=-1)
    
    matrix = torch.stack([x, y, z], dim=-1) # (B, 3, 3)
    return matrix
```

---

## 4. 相机投影几何 (Camera Projection)

VLA 模型如何理解 3D 世界？通过相机内参。

### 4.1. 针孔相机模型 (Pinhole Model)
将 3D 点 $P_C = [X, Y, Z]^T$ 投影到 2D 像素平面 $p = [u, v]^T$：

$$
Z \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K \begin{bmatrix} X \\ Y \\ Z \end{bmatrix}
$$

其中 $K$ 是 **相机内参矩阵 (Intrinsics Matrix)**：

$$
K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
$$

- $f_x, f_y$: 焦距 (Focal Length)，单位是像素。
- $c_x, c_y$: 光心 (Principal Point)，通常是图像中心。

### 4.2. 深度反投影 (Deprojection)
如果我们有深度图 $D(u, v) = Z$，可以将像素 $(u, v)$ 还原为 3D 点：

$$
X = \frac{(u - c_x) \cdot Z}{f_x}, \quad Y = \frac{(v - c_y) \cdot Z}{f_y}
$$

这就是 **PointCloud** 生成的原理。

---

## 5. 运动学 (Kinematics)

### 5.1. 正运动学 (Forward Kinematics, FK)
- **输入**: 关节角度 $\theta = [\theta_1, \dots, \theta_7]$。
- **输出**: 末端位姿 $T_{E}^{B}$。
- **计算**: 简单的矩阵连乘。$T_{E}^{B} = T_{1}^{B}(\theta_1) \times T_{2}^{1}(\theta_2) \dots$

### 5.2. 逆运动学 (Inverse Kinematics, IK)
- **输入**: 期望的末端位姿 $T_{target}$。
- **输出**: 关节角度 $\theta$。
- **难点**: 
    - **多解**: 同一个位置，机械臂可能有多种姿态到达 (Elbow up / Elbow down)。
    - **无解**: 目标点超出工作空间。
- **VLA 中的应用**: 
    - 通常 VLA 模型预测 **End-effector Pose** (Task Space)，然后通过 **IK Solver** (如 PyBullet IK 或 Franka IK) 计算出关节角度发送给电机。

---

## 6. 面试高频考点

**Q: 为什么 VLA 模型通常预测 Delta Pose 而不是 Absolute Pose？**
A: 
1. **泛化性**: Delta Action (如“向前移动 1cm”) 在不同位置、不同机器人尺寸下更通用。
2. **误差解耦**: 绝对坐标强依赖于 Base Frame 的标定。如果相机外参 $T_{C}^{B}$ 有 1cm 的误差，预测绝对坐标就会全程偏差 1cm。而 Delta Action 配合高频闭环 (Closed-loop)，模型可以像人眼一样，“看着手”进行微调，对外参误差不敏感。

**Q: 为什么旋转矩阵不适合直接作为神经网络输出？**
A: 旋转矩阵必须满足正交约束 $R^T R = I$ 和行列式 $\det(R)=1$。这是一个位于流形 (Manifold) 上的约束。神经网络输出的是无约束的 $\mathbb{R}^9$ 向量，很难保证正交性。强行正交化 (SVD) 会破坏梯度。

**Q: 解释一下 6D Rotation 的连续性优势。**
A: 在 3D 旋转空间 $SO(3)$ 中，欧拉角有奇异点，四元数有双倍覆盖 (拓扑上是球面的双倍覆叠)。这导致从神经网络的欧几里得空间 $\mathbb{R}^n$ 到 $SO(3)$ 的映射在某些点是不连续的。6D 表示通过舍弃冗余约束 (正交性由后处理保证)，实现了 $\mathbb{R}^6 \to SO(3)$ 的连续映射，使得训练更稳定。
