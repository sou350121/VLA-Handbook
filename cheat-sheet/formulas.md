# 核心公式速查（Core Formulas）

> 2026-04-21 更新 · 补充：6D 旋转 · Action Chunking · Flow Matching 推导 · RL 基础 · Advantage

面试可能要手推的核心算法公式。

---

## 1. Transformer / Attention

### 1.1 Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- **Q** 查询 · **K** 键 · **V** 值
- $\sqrt{d_k}$：缩放因子，避免点积过大导致 softmax 梯度消失

### 1.2 Multi-Head Attention

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$

其中 $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

### 1.3 Flash Attention（在线 softmax）

$$m_{\text{new}} = \max(m_{\text{old}}, m_{\text{block}})$$
$$l_{\text{new}} = e^{m_{\text{old}} - m_{\text{new}}} l_{\text{old}} + e^{m_{\text{block}} - m_{\text{new}}} l_{\text{block}}$$

核心：逐块更新，内存 $O(N)$ 而非 $O(N^2)$。

---

## 2. Diffusion Policy

### 2.1 前向（加噪）

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

### 2.2 反向（去噪 → 生成动作）

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

### 2.3 训练 loss（ε-prediction）

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \lVert \epsilon - \epsilon_\theta(x_t, t) \rVert^2 \right]$$

- 机器人输入观察 $O$，从 $x_T \sim \mathcal{N}(0, I)$ 起步逐步去噪到 $x_0$

---

## 3. Flow Matching（π 系列核心）

### 3.1 ODE 定义

$$\frac{dx}{dt} = v_t(x)$$

学一个**速度场** $v_\theta$，从噪声 $x_1$ 确定性推到数据 $x_0$。

### 3.2 训练 loss（CFM · Conditional Flow Matching）

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, x_0, x_1} \left[ \lVert v_\theta(x_t, t) - (x_1 - x_0) \rVert^2 \right]$$

- 目标速度：直线方向 $x_1 - x_0$（Optimal Transport）
- **$t \sim \mathcal{U}[0,1]$，$x_t = (1-t)x_0 + t x_1$**

### 3.3 推理（Euler 积分）

$$x_{t+\Delta t} = x_t + v_\theta(x_t, t) \cdot \Delta t$$

通常 4-10 步就够 → **高频控制**（10-50 Hz）的关键。

### 3.4 为什么比 Diffusion 快

| 维度 | Diffusion | Flow Matching |
|------|-----------|--------------|
| 数学 | SDE（随机） | ODE（确定性） |
| 训练目标 | 预测噪声 | 预测速度（$x_1-x_0$） |
| 推理步数 | 通常 50-1000 | 4-10 |
| 多峰能力 | ✅ 强 | ✅ 强（通过 flow 保持） |

---

## 4. 动作表示 · 旋转编码

### 4.1 四元数（Quaternion）

$$q = w + xi + yj + zk, \quad \lVert q \rVert = 1$$

### 4.1.1 四元数**双覆盖问题** 🚩

- $q$ 和 $-q$ 表示**同一姿态**
- 直接做 MSE 会让网络在 $q$ 和 $-q$ 之间震荡 → 训练不稳
- **解法**：取最近半球（$\langle q, q_{target} \rangle \geq 0$）

### 4.1.2 Slerp（球面线性插值）

$$\text{Slerp}(q_1, q_2, t) = \frac{\sin((1-t)\theta)}{\sin\theta} q_1 + \frac{\sin(t\theta)}{\sin\theta} q_2$$

其中 $\cos\theta = q_1 \cdot q_2$。

### 4.2 **6D 连续旋转表示**（📎 Zhou et al. 2019, VLA 推荐）

旋转矩阵 $R \in SO(3)$ 的前两列 $[b_1, b_2]$ 是**最友好的神经网络表示**（连续、无万向节、无双覆盖）。

反投影（Gram-Schmidt）：
$$b_3 = b_1 \times b_2, \quad R = [b_1, b_2, b_3]$$

### 4.3 Euler 角风险

- 万向节死锁（Gimbal Lock）
- 不连续（$\pi \to -\pi$ 翻转）
- **仅用于人类可读的存储** · 训练中避免

### 4.4 Geodesic Loss（旋转 loss 的正确做法）

$$\mathcal{L}_{\text{geo}}(R_{\text{pred}}, R_{\text{true}}) = \arccos\left(\frac{\text{trace}(R_{\text{pred}}^T R_{\text{true}}) - 1}{2}\right)$$

- **不要用 MSE on rotation**，几何不对称

---

## 5. Action Chunking（ACT 核心）

### 5.1 核心思想

同时预测未来 $k$ 步动作 $[a_t, a_{t+1}, ..., a_{t+k-1}]$，而不是仅预测 $a_t$。

### 5.2 公式

$$\pi(a_{t:t+k} | o_t) \quad \text{vs.} \quad \pi(a_t | o_t)$$

### 5.3 Temporal Ensemble（推理时 smooth）

每一步 $t$ 执行的动作是多个 chunk 的加权平均：

$$a_t^{\text{exec}} = \sum_{i=0}^{k-1} w_i \cdot a_t^{(i)}$$

其中 $a_t^{(i)}$ 是第 $i$ 步前预测的该时刻动作。

### 5.4 为什么有效

- 减少**累积误差**（一次预测多步，避免每步都推理出错）
- 更**平滑**（smoothing over overlapping chunks）
- 训练更**高效**（一次前向学多步）

---

## 6. LoRA（OpenVLA 微调必备）

$$W' = W + \Delta W = W + BA$$

- $W \in \mathbb{R}^{d \times k}$：冻结的预训练权重
- $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$：可训练，$r \ll d, k$
- **可训练参数从 $dk$ 降到 $r(d+k)$**
- 典型 $r \in \{8, 16, 32, 64\}$
- **推理**：merge 回 $W'$ 即可，零延迟增加

---

## 7. 归一化（坑比想像多）

### 7.1 百分位归一化（OpenVLA / π 系列标准）

$$\tilde{x} = \frac{x - q_{1\%}}{q_{99\%} - q_{1\%}}$$

- **不用** Min-Max：一条异常轨迹污染全集
- **不用** Z-score：动作分布通常多峰，高斯假设错误
- **per-dim** 独立归一化（维度量纲差别大）

### 7.2 Revolute 关节的 unwrap

连续两帧关节角 $q_t, q_{t+1}$，若 $|q_{t+1} - q_t| > \pi$ 判定为**穿越 0**，做 $q_{t+1} \mp 2\pi$ 修正。**归一化前做**。

---

## 8. 机器人学基础

### 8.1 坐标变换（齐次变换）

$$^A P = {}^A T_B \cdot {}^B P$$

$$^A T_B = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix}$$

- $R$：3×3 旋转矩阵 · $t$：3×1 平移向量

### 8.2 FK（Forward Kinematics）

$$T_{\text{ee}} = T_{\text{base}} \cdot \prod_{i=1}^{N} T_i(q_i)$$

### 8.3 Jacobian

$$\dot{x} = J(q) \dot{q}$$

$J \in \mathbb{R}^{6 \times N}$ 把关节速度映到末端线速度+角速度。

---

## 9. RL 基础（后训练必备）

### 9.1 Return（带折扣）

$$G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$$

### 9.2 State Value

$$V^\pi(s) = \mathbb{E}_\pi [G_t | s_t = s]$$

### 9.3 Action Value (Q)

$$Q^\pi(s, a) = \mathbb{E}_\pi [G_t | s_t = s, a_t = a]$$

### 9.4 Advantage（核心！）

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

- "动作 a 相对于平均策略好多少"
- **π*0.6 等 ACP 方法用这个做 condition**

### 9.5 Policy Gradient

$$\nabla J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot A^\pi(s_t, a_t) \right]$$

- **需要** $\log \pi(a|s)$ —— 这在 flow-based VLA 上很难

### 9.6 PPO Clipped Objective

$$\mathcal{L}^{\text{PPO}}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$

### 9.7 Advantage Conditioned Policy（π*0.6 做法）

$$\pi(a | s, A^{\text{norm}}) \quad \text{学监督 with advantage as condition}$$

- Advantage 归一化到 $[0, 1]$
- 训练：$\mathbb{E}[(\pi_\theta(a|s,A) - a_{\text{label}})^2]$
- 推理：$A=1$（最优动作）

---

## 10. 评估指标

### 10.1 Success Rate

$$\text{SR} = \frac{\text{成功 episodes}}{\text{总 episodes}}$$

### 10.2 控制频率

$$f = \frac{1}{\Delta t} \quad (\text{Hz})$$

- 高层 VLA：3-10 Hz
- 底层关节控制：500-1000 Hz（由 PD 控制器处理）

### 10.3 Scaling Law（📎 EgoScale · ICLR 2025）

$$\log(\text{validation loss}) = -\alpha \log N + C$$

其中 $N$ 是数据量（小时 or episodes），**log-linear 关系**。📎 EgoScale 实证 R² = 0.9983。

---

## 11. 常用红旗（面试/论文阅读）

| 现象 | 对应公式错误 |
|------|-------------|
| 四元数 MSE 回归 | 违反双覆盖（应取近半球 / 用 6D） |
| Euler 角做回归 | 万向节 / 不连续 |
| 关节角未 unwrap 归一化 | $\pi / -\pi$ 附近数据被错误拉开 |
| Min-Max 归一化 | 一条异常污染整个数据集 |
| MSE 做旋转 loss | 几何不对称（应用 geodesic） |
| Diffusion/Flow 算法混用 | 两者 RL 路线完全不同 |

---

[← Back to Cheat Sheet](./README.md)
