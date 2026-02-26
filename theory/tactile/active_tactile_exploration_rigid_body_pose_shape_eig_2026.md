# Active Tactile Exploration：仅触觉的刚体位姿+形状估计 (Active Tactile Exploration for Rigid Body Pose and Shape Estimation)

> **会议**：ICRA 2026（项目页注明 *Accepted*，camera-ready pending）  
> **论文**：Active Tactile Exploration for Rigid Body Pose and Shape Estimation  
> **核心定位**：在“物体会被触碰推走”的真实场景里，只用触觉/本体数据，在 <10s 接触数据内同时估计 **刚体轨迹（pose over time）+ 形状（cuboid/convex polytope）**，并用 **EIG（Expected Information Gain）** 主动选动作加速收敛。  
> **一手来源**：arXiv HTML `https://arxiv.org/html/2510.13595v2`；项目页 `https://dairlab.github.io/activetactile`；arXiv `https://arxiv.org/abs/2510.13595`

这篇文章重要在于：它把“触觉主动探索”从静态物体/离散类别，推进到 **动态、连续几何族、并且可被扰动** 的设定；并给了一个工程上可实现的闭环：**violation-implicit loss（好优化）+ EIG（选动作）**。

### X-Ray 开场（非专家也能复述）

触觉在真实世界很强，但也很“抠门”：只有接触到的局部才有信息，而且每次触碰可能把物体推走，导致你不仅要学形状，还要学“物体此刻在哪、怎么动”。作者的关键做法是：用一个 **violation-implicit loss** 把刚体约束/接触约束“软地”写进优化目标，避免刚体接触带来的数值僵硬；然后用 **EIG** 选择下一步最有信息的新触碰方向，让模型在很少动作里变准。

### 📍 研究全景时间线（它补了哪块短板）

```text
2015-2023  触觉用于“已知物体/已知环境”的定位与微调（localization/refinement）
2016-2024  主动探索：多用于静态物体、或离散候选集合（分类/检索）
2024-2025  接触丰富系统辨识：更强调“可优化的物理损失/可微仿真”
2026       本文：动态未知刚体（会被推走）+ 连续凸几何（cuboid/polytope）
           ├─ violation-implicit loss：同时学 pose trajectory + geometry
           └─ EIG：把“下一步碰哪里”做成信息最大化问题
```

### ⚡ Eureka Moment（THE 关键洞见一句话）

**用 EIG（基于 Fisher/observed information 的比值）替代显式 belief：在“物体会被扰动”导致 belief 传播很难时，EIG 更稳、更可算；再配合 violation-implicit loss 让“轨迹+几何”的梯度优化可行。**

---

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览：它和“静态触觉建模”差在哪？

| 维度 | 传统触觉建模（常见假设） | 本文（Active Tactile Exploration） | 关键差异 |
|---|---|---|---|
| **物体是否会动** | 常假设静止/固定 | **会被触碰推走** | 必须同时估计 pose trajectory |
| **形状空间** | 离散类别或已知 CAD | **连续几何族**：cuboid / convex polytope | 不是分类，而是连续参数优化 |
| **数据稀疏** | 被动采集 | **主动探索**（EIG） | 用更少动作拿到更快收敛 |
| **优化难点** | 可能忽略接触刚体约束 | **violation-implicit loss** 软约束 | 避免 rigid contact 数值僵硬/梯度爆炸 |

### 1.2 交互协议（对应论文 Figure 2）

```text
Loop:
  1) Learning:
       minimize violation-implicit loss -> estimate (shape θ, pose trajectory x_0:T)
  2) Exploration:
       sample candidate trajectories r_T:H
       simulate forward -> score EIG
       execute best trajectory -> acquire tactile data
  3) Append data and repeat
```

### 1.3 两个核心模块

- **Learning（Sec. IV）**：只用触觉接触布尔值 + 接触法向（以及本体/末端位姿）来拟合 `θ`（几何参数）和 `x_t`（每时刻刚体位姿）。
- **Exploration（Sec. V）**：选动作不是“去不确定处”这种启发式，而是最大化 **EIG**：你下一次触碰预计带来的信息，相对你已观测到的信息能增加多少。

---

## 2. 数学核心：Violation-Implicit Loss + EIG (Math Core)

> Napkin Formula：把“物理约束难优化”变成“先对接触冲量做内层最小化”，用 envelope theorem 避免显式反传；再用 `log det(F * I_obs^{-1} + I)` 作为 EIG。

### 2.1 触觉观测是什么？

在每个时间步 `t`，每个触觉末端传感器给出：

- `c_t ∈ {0,1}`：是否接触
- `n_hat_{t,m} ∈ S^2`：测得接触法向方向（从触觉图像/CoP 映射得到）
- `r_t`：末端位姿（机器人本体/正运动学得到）

目标：用尽量少的动作，估计最终时刻的物体位姿 `x_T` 和几何 `θ`（cuboid 或 convex polytope）。

### 2.2 为什么要 violation-implicit？

如果你直接做“最大似然 + 刚体接触动力学硬约束”，会遇到典型问题：

- 某些区域梯度接近 0（学不动）
- 接触边界附近梯度近似不连续（数值僵硬）

论文借鉴 contact-rich system identification 的思路，把接触冲量 `λ_t` 作为内层优化变量，把动力学与互补约束等“物理项”变成 penalty，构成 **violation-implicit loss**：

```text
L_v(θ, x_{0:T}, D_obs)
  = Σ_t  min_{λ_t ∈ friction_cone}
          [  L_t(sensor; θ, x_t, r_t)
           + || x_{t+1} - f_θ(x_t, r_t, λ_t) ||^2
           + g_θ(physical_constraints; x_t, r_t, λ_t)
          ]
```

这里 `g_θ` 包含论文列出的几类物理约束 penalty（互补、最大耗散、非穿透、非弹性等）。关键点是：内层 `min_{λ_t}` 可以写成一个 QP/SOCP，并且可用 envelope theorem 让外层梯度更“温和”。

### 2.3 EIG：为什么不用显式 belief？

物体会被扰动时，显式 belief 传播（多模态、非高斯、接触不稳定）很难做。论文的立场是：**EIG 可以绕开显式 belief**，只需要当前的点估计（MLE/MAP 附近）以及 Hessian 近似的“信息量”。

论文给的 EIG 形式（省略常数）：

```text
EIG ∝ log det( F(D_acq) * I(D_obs)^{-1} + I )

I(D_obs): observed information  (≈ Hessian of loss at MLE)
F(D_acq): Fisher information    (expected observed info of future data)
```

直觉：下一段动作预计带来的信息 `F`，和你目前已经观测到的信息 `I_obs` 比一比——如果你总去碰“已经看过很多的面”，EIG 就低；去碰“没看过的面”，EIG 就高。

---

## 3. 带数字走一遍：一个最小直觉例子 (Worked Example)

**场景**：物体是个未知 cuboid，且会在推触时滑动；你只有两指触觉，只能拿到“是否接触 + 法向”。

- **第 1 次随机接触**：你可能只得到一个面的大致法向信息，几何上只约束了一个平面。
- **Learning**：优化 `L_v` 后，你能得到一个粗糙的 cuboid（很多面仍“没被看见”），并得到一个可解释的 pose 轨迹（物体被推走的位移/旋转）。
- **Exploration（EIG）**：下一步如果再从同方向接近，预计拿到的“新增信息”很小；而绕到对侧去触碰一个“未观测面”，EIG 会显著更大，因此会被选中。

结果是：相对随机探索，EIG 能更快把 “未观测面” 变成 “已观测面”，在更少动作里把几何与位姿估计压到更低误差。

---

## 4. 工程视角：实现时你真正需要做什么 (Engineering View)

### 4.1 你需要的接口

- **感知输入**：
  - 触觉：`c_t` 与 `n_hat_{t,m}`（论文用 Densetact 的图像流推 CoP 与法向）
  - 本体：末端位姿 `r_t`
- **优化模块**：
  - 外层：对 `θ` 与 `x_{0:T}` 做梯度下降（多初始化避免局部极小）
  - 内层：对每个 `t` 解一个带摩擦锥约束的 QP/SOCP（`λ_t`）
- **规划模块（探索）**：
  - 用交叉熵法（CEM）采样候选轨迹
  - forward simulate 到 horizon `H` 并打分 EIG

### 4.2 关键工程权衡

- **为什么用 convex / cuboid / convex polytope**：保证接触几何更可控、碰撞点更“唯一”，让 loss 和信息量的计算更稳定。
- **为什么 EIG + CEM**：不用维护 belief；同时 CEM 可以在非凸评分函数下做“够用的”黑箱优化。
- **局部最小**：论文明确提到 MLE 可能落在坏的局部最小；需要多初始化、以及更鲁棒的动作空间。

---

## 5. 实验设置与结果 (Data & Eval)

### 5.1 结果的主要结论（从摘要/项目页抓重点）

- **<10s 触觉数据**就能在接触后学到 cuboid 或 convex polytope 的几何近似，并同时估计位姿轨迹。
- **EIG 探索**比随机探索更快、更稳定（仿真与真机都显著改善）。

### 5.2 硬件与评估口径

论文实验使用双指 Trifinger 改造平台 + Densetact；并用相机跑 FoundationPose 作为 **仅评估** 的 “ground truth pose”，最终用 **双向 Chamfer 距离（bCH）** 衡量形状误差。

---

## 6. 能力与失败模式 (Capabilities & Failure Modes)

### 6.1 能做什么

- **动态未知物体**：接触会推走也能学（估计 pose trajectory + geometry）
- **高数据效率**：触觉稀疏但用主动探索把动作数压下来

### 6.2 不能做什么 / 风险点

- **非凸 / 非连续形状**：当前形状族是凸集合（或 cuboid），非凸对象需要额外分解/表示。
- **强扰动/翻滚**：论文讨论中提到，EIG 的某些近似（例如对状态敏感度的处理）在“更混乱的扰动”下可能失效。
- **局部最小**：EIG 是局部信息度量，若当前 MLE 落在坏解附近，探索会被误导。

---

## 7. 与相关路线对比 (Comparison)

| 路线 | 典型做法 | 本文差异 |
|---|---|---|
| 触觉建模（静态） | 形状/位姿在固定参考系里估计 | 本文显式处理“触碰导致物体移动” |
| Bayes/MI 主动探索 | belief + MI（难算、易多模态） | 用 EIG 绕开显式 belief |
| Fisher-only | 只最大化未来信息 | EIG 是“未来信息 / 已观测信息”的相对增益，更适合多步序列 |

### 面试 Tip

如果面试官问“这篇 paper 的创新点是什么”，最稳的一句话是：**它把“动态未知物体的触觉建模”拆成可优化的 violation-implicit loss，并用 EIG 做主动探索，从而在 <10s 触觉数据里同时学 pose trajectory 与凸几何近似。**

---

## References

- arXiv (HTML): `https://arxiv.org/html/2510.13595v2`  
- Project Page (DAIRLab): `https://dairlab.github.io/activetactile`  
- arXiv: `https://arxiv.org/abs/2510.13595`  

---

[← Back to Tactile Index](./README.md)

