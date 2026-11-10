# CompVLA：面向接触丰富操作的可变柔顺 VLA (CompVLA: A Variable Compliance Vision-Language-Action Model for Contact-rich Manipulation)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-11-10
>
> **论文**: CompVLA: A Variable Compliance Vision-Language-Action Model for Contact-rich Manipulation
> **链接**: https://arxiv.org/abs/2609.23614
> **核心定位**: 把 VLA 的动作从“纯运动学 twist + gripper”扩展为“twist + gripper + diagonal stiffness + virtual displacement”，再用 geometric impedance control 执行，从而在接触丰富任务中同时做方向柔顺与力调制。

> **证据口径**：本稿依据 arXiv:2609.23614v2 正文（页面标注 06 Nov 2026；本次提供的是 HTML 全文）。候选摘要片段在 “Ou” 处截断，不是完整摘要，因此不用于补写论文主张；下文数字与机制均按正文 §/Table 标注。若后续同号 PDF 与页面元数据摘要出现不一致，应以同号 PDF 正文为准。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | CompVLA 让策略联合预测 motion、diagonal stiffness 与 virtual displacement，经 GIC 实现方向柔顺和力调制；三个 contact-rich 任务平均 67.1%，高于 ForceVLA 的 54.4%、π0 的 31.5% |
| 适合精读 | 如果你在做 contact-rich manipulation、variable impedance、VLA action space 设计，重点看 §4.1、§4.2 与 Appendix A.2 |
| 可以跳过 | 如果你只关心纯运动学 pick-and-place 或大规模通用 VLA 预训练，这篇距离中等 |
| 落地可行性 | 中：方法不要求推理时外部 F/T 传感器，但依赖高质量 reference trajectory、F/T 标签构造与可 torque/阻抗执行的机器人接口 |
| 主要风险 | Cup Dispense 上 CompVLA 35.0% 低于 ForceVLA 50.0%；作者也承认 compliance 不能补偿大轨迹误差，controller 是简化 GIC，稳定性不是保证 |

💡 **X-Ray 开场**

这篇论文解决的是：现有 VLA 大多只输出运动学目标，到了擦白板、pivot、按压出杯这类任务，机器人不知道“该多硬、该往哪个方向让、该用多大力”。CompVLA 的做法不是外挂一个力控制器，而是让 VLA 直接输出 `k_t` 和 `d_t`：`k_t` 决定各轴刚度，`d_t` 既把虚拟目标推进接触面产生力，又定义柔顺主轴。对 VLA 研究者来说，这意味着 action space 本身可以成为“物理交互参数”的接口，但前提是 reference motion 足够准、controller 能可靠执行。

📍 **研究全景时间线**

```
1984 Hogan 阻抗控制
  → 2024 π0：flow-matching action expert 的 VLA
  → 2025 ACP：从数据学习近似 compliance
  → 2025 TA-VLA：torque-aware VLA
  → 2026 CompVLA：把 stiffness + virtual displacement 纳入 VLA action
```

## 1. 核心架构/方法总览

### 1.1 系统对比概览

| 组件 | 输入 | 输出 | 时序/频率 | 训练/推理差异 |
|------|------|------|-----------|----------------|
| VLM backbone | RGB、language | key-value 表示 | 与 policy chunk 同步 | 与两个 expert 联合 fine-tune |
| Action Expert | VLM KV、robot state、flow time、noisy action | reference TCP twist `V_t`、gripper `g_t` | 预测 `T=25` 的 motion chunk | 训练用 GT action；推理从噪声 denoise 10 步 |
| Compliance Expert | VLM KV、relative TCP trajectory condition `Z_t`、gripper | diagonal stiffness `k_t`、virtual displacement `d_t` | 与 action chunk 同 horizon | 训练用 GT TCP trajectory；推理用 Action Expert 预测轨迹 |
| GIC / controller | reference trajectory、`k_t`、`d_t`、当前 TCP pose | task-space wrench、joint torque `τ_imp` | 高频执行；论文未给出具体控制频率 | 推理时不用外部 F/T 测量 |

关键点：Compliance Expert 的输入不是外部力传感器，而是 VLM context 与 reference trajectory。F/T 用于构造 compliance labels，并在评测中作为接触力测量证据；论文明确说 evaluation 时 policy/controller 不使用 ATI F/T 读数。

### 1.2 关键机制

- **Directional compliance**：不是全轴统一低刚度，而是沿接触方向低刚度、正交方向高刚度。标签构造时用 measured force/torque 估计接触方向，建立 compliance frame；第一轴对齐接触方向，另外两轴取最接近 TCP 的 shortest geodesic orientation。刚度取 `K_0 = diag(k_low, k_max, k_max)`。
- **Force modulation**：仅有低刚度不能产生按压力。若目标位姿与接触位姿重合，spring deflection 为零，改变刚度也不产生力。CompVLA 用 `d_t` 把 equilibrium pose 推进接触面，靠 spring term 产生力；位移幅值由 demonstrated force/torque 除以 `k_low` 得到。
- **Trajectory-conditioned compliance**：Compliance Expert 先看 Action Expert 预测的参考轨迹，再预测 `k_t` 与 `d_t`。这把低频 motion generation 与高频 interaction control 串起来。
- **Controller-aware action**：`d_t` 在控制器中同时承担两件事：平移/旋转虚拟目标，以及确定 compliance frame 的主轴方向。因此 `k_t` 与 `d_t` 必须成对使用，消融中“只用 k 不用 d”反而比 position-only 更差。

⚡ **Eureka Moment**：把 compliance 从控制器外挂参数变成 VLA 的 action token；`d_t` 既推目标产生力，又定义柔顺主轴，`k_t` 决定各轴让多少。

### 1.3 信息流/架构图

```text
language l + image I_t
        │
        ▼
   VLM backbone (PaliGemma)
        │  key-value
        ├──────────────► Action Expert（另接 state s_t）
        │                   │
        │                   ├─► V_t, g_t
        │                   │      │
        │                   │      └─► integrate to reference trajectory X_d
        │                   │                   │
        │                   └──────────────┐    │
        │                                  ▼    ▼
        └──────────────► Compliance Expert ← Z_t
                              │  (relative TCP trajectory + gripper)
                              ├─► k_t
                              └─► d_t
                                     │
                                     ▼
                          GIC / impedance controller
                                     │
                                     ▼
                              τ_imp → robot
```

## 2. 数学核心

📌 **Napkin Formula**：

```text
action token = (V_t, g_t, k_t, d_t)
d_t 推目标产生力，k_t 定各轴柔顺；二者经 GIC 变成 τ_imp
```

先给目标：策略在时刻 `t` 接收当前观测 `O_t = {I_t, s_t}` 与语言 `l`，预测未来 motion chunk；每个 token 不只是运动，还包含 stiffness 与 virtual displacement。

```text
M_t = (m_t, m_{t+1}, ..., m_{t+T-1})
m_t = (V_t, g_t, k_t, d_t)
```

其中 `T=25`，`V_t ∈ R^6` 是 reference TCP twist，`g_t ∈ {0,1}` 是 gripper command，`k_t ∈ R^6` 是 diagonal stiffness，`d_t ∈ R^6` 是 virtual displacement。注意：这是未来动作 chunk，不是历史上下文或记忆窗口。

标签构造分两步。方向柔顺先用 measured force/torque 确定接触轴，并对 `k_low` 做分段线性调制：

```text
k_low(|f|) =
  k_max,                                                   |f| < f_min
  k_max - (k_max - k_min) * (|f| - f_min)/(f_max - f_min), f_min <= |f| <= f_max
  k_min,                                                   |f| > f_max
```

平移空间取 `(k_max, k_min, f_min, f_max) = (500 N/m, 200 N/m, 1 N, 15 N)`；旋转空间取 `(30 Nm/rad, 10 Nm/rad, 0.1 Nm, 2.5 Nm)`。力调制则令：

```text
|d_t| = |f_t| / k_low(|f_t|)
```

方向由接触方向给出。也就是说，`d_t` 的方向和大小共同定义虚拟目标偏移；`k_t` 定义对应轴的刚度。

控制器实现层用 virtual target `X^v` 代替 reference `X_d`：

```text
R^v = R_d Exp(d^R)
p^v = p_d + R_d d^p

K_d^R = R~^R diag(k^R) (R~^R)^T
K_d^p = R~^p diag(k^p) (R~^p)^T
K_d = blkdiag(K_d^R, K_d^p)
```

误差相对 virtual target 计算：

```text
R_e = Log((R^v)^T R)
p_e = R^T (p - p^v)
F = -K_d X_e - D_d V_e
τ_imp = J^T F
```

> 这里要区分概念层与实现层：§3 的 `X_e = log(X_d^{-1}X)^∨`、`V_e = V^b - Ad_{X^{-1}X_d}V_d^b` 是 GIC 的概念公式；Appendix A.2 的实际实现用 `X^v` 构造 decoupled pose error，`V_e` 也相对 virtual target。不能把实现层误写成相对 `X_d`。

Damping 用 critical-damping heuristic：

```text
c_i = 2 sqrt(k_i)
D_d = blkdiag(R~^R diag(c^R)(R~^R)^T, R~^p diag(c^p)(R~^p)^T)
```

训练用 flow matching，两个 expert 各自回归目标 vector field：

```text
Y~(τ) = τ ε + (1 - τ) Y
L_a = E || v_a(Y~, τ | O, l) - (ε - Y_a) ||_2^2
L_c = E || v_c(Y~, τ | O, l, Z) - (ε - Y_c) ||_2^2
L = L_a + L_c
```

变量说明：

| 符号 | 含义 |
|------|------|
| `V_t` | reference TCP twist |
| `g_t` | gripper command |
| `k_t` | 6 维 diagonal stiffness，分 rotational/translational |
| `d_t` | 6 维 virtual displacement，分 rotational/translational |
| `X^v` | 由 reference pose 加 `d_t` 得到的 virtual target pose |
| `K_d` | 由 `k_t` 经 compliance frame 旋转得到的 task-space stiffness matrix |
| `D_d` | 由 `k_t` 经 critical damping 规则得到的 damping matrix |
| `Z_t` | relative TCP trajectory + gripper 的 conditioning sequence |

直觉：`k_t` 决定“遇到外力时让多少”，`d_t` 决定“往哪里让、往哪里压”。`d_t` 不是简单的目标偏移，因为它同时改变 spring deflection 和 compliance frame 的主轴；所以消融中移除 `d_t` 会破坏 `k_t` 的训练配对。

## 3. 带数字走一遍：玩具例子

> 以下为假设性 2D 平移例子，用来验算公式，不是论文实验。忽略旋转、阻尼、摩擦与控制器动态，只看法向按压。

假设接触方向为 `+x`，正交方向为 `y`。取论文 Appendix B.2 的平移参数：`k_max = 500 N/m`，`k_min = 200 N/m`，`f_min = 1 N`，`f_max = 15 N`。目标按压力 `|f| = 3 N`。

先算 `k_low`：

```text
k_low = 500 - (500 - 200) * (3 - 1)/(15 - 1)
      = 500 - 300 * 2/14
      = 500 - 42.857
      = 457.143 N/m
```

再算 virtual displacement 幅值：

```text
|d_x| = 3 / 457.143 = 0.00656 m = 6.56 mm
```

假设参考目标 `p_d = (0, 0)`，接触方向 `+x`，则 virtual target：

```text
p^v = (6.56 mm, 0)
K = diag(457.143, 500) N/m
```

当前 TCP 在 `p = (0, 0)`，相对 virtual target 的误差：

```text
p_e = p - p^v = (-6.56 mm, 0)
F = -K p_e = (457.143 * 0.00656, 0) ≈ (3.0 N, 0)
```

验算：`457.143 × 0.00656 ≈ 3.0 N`，与目标按压力一致；`y` 方向误差为 0，所以 `y` 方向虽然刚度为 500 N/m，但不产生力。若去掉 `d_t`，`p^v = p_d = (0,0)`，`p_e = 0`，spring term 为 0，改变刚度也压不出 3 N。若只保留 `d_t` 但把 `k` 错配成 200 N/m，力约为 `200 × 0.00656 = 1.31 N`，说明 `k` 与 `d` 必须配对。

这个例子也说明为什么“统一低刚度”不够：`x` 方向要软以吸收接触误差，`y` 方向要硬以保持轨迹；各向异性来自 `k_t` 与 compliance frame，而不是单一 scalar gain。

## 4. 工程视角

**低频策略与高频控制的分界。** 论文的 policy 输出 `T=25` 的 action chunk；数据转换把所有模态统一到 10 Hz。若按 10 Hz 与 `T=25` 计算，`M_t = (m_t, ..., m_{t+T-1})` 含当前 token，首末 token 跨度为 `(25-1) × 0.1 = 2.4 s`。若改写成“从当前参考 t 预测 t+1 至 t+H”的 H 步未来，则终点距 t 是 `H × 0.1 s`；`H=25` 时为 2.5 s。两种口径不能混用。论文没有给出 GIC 的具体控制频率，因此不能把 10 Hz 当作控制器频率，也不能把 `T=25` 写成所有实验共有的记忆窗口。

**历史帧与未来 chunk 要分开。** 论文定义的是当前观测 `O_t = {I_t, s_t}` 与未来 motion chunk `M_t`；它没有披露视觉历史帧数量，也没有提出显式记忆模块。`M_t` 是未来预测区间，不能直接当作历史上下文或 memory window。工程上若要复现，需要另行确认图像历史长度、状态历史长度与 chunk 执行策略；这些不是本文已给出的属性。

**推理成本。** Action Expert 与 Compliance Expert 都用 flow matching；推理从噪声开始 denoise 10 步。训练是 100,000 steps、batch size 32、bfloat16、AdamW、peak lr `2.5e-5`、warmup 1,000 steps、后续 30,000 steps 衰减到 `2.5e-6`。这些是训练设置，不是每个控制周期的计算量；论文未报告端到端延迟、吞吐或 controller 抖动。

**控制器简化与稳定性。** Appendix A.2 明确：实现省略了 feedforward 项 `Λ(θ) dot(V_d^*) + μ(θ,V) V_d^*`，依赖机器人低层 gravity compensation，并使用 decoupled pose error；由于这些简化和 time-varying stiffness，Seo et al. 的稳定性结果只是“motivates”该控制器，而不是保证其稳定。工程落地时，这是比“成功率”更值得先验证的边界。

**传感器边界。** 数据采集时有 ATI Axia F/T、joint torque、三相机；F/T 用于构造 compliance labels，但 evaluation 时 policy/controller 不用外部 F/T 读数。不要把 TA-VLA 说成需要外部 F/T 传感器：论文描述 TA-VLA 使用 joint torque 输入，并把 future torque 作为辅助训练信号；CompVLA 的优势是推理时不用 external force/torque sensor measurements，而不是“所有基线都需要 F/T 传感器”。

**数据开放范围。** 论文声称发布 raw contact-rich demonstrations 与 data conversion code，用于后续 compliant manipulation 研究；本稿按引用约束不列额外仓库 URL。这个发布声称只属于 CompVLA 的数据/代码，不能外推到 ForceVLA、TA-VLA、ForceVLA2 或 CompliantVLA-adaptor 的开放范围。

## 5. 数据与评测

### 5.1 预训练与模型初始化

CompVLA 不是从零预训练。论文说它基于 LeRobot v0.4.3 的 π0 实现，VLM backbone 用 PaliGemma，Action Expert 用 Gemma-based model；Compliance Expert 复制 Action Expert 的权重初始化。模型从 `pi0_base` checkpoint 初始化，然后 VLM backbone、Action Expert、Compliance Expert 联合 fine-tune 100,000 steps。训练时还把 joint angles 作为 auxiliary prediction targets，以保留预训练 checkpoint 信息，推理时丢弃这些辅助输出。

### 5.2 数据

论文采集了带 F/T 的遥操作演示数据。平台是两套 7-DoF Franka Research 3：leader-follower，follower 用于数据采集与评测；三台 Intel RealSense D435i 分别位于 front、side、wrist；手腕装 ATI Axia F/T；夹爪换成 UMI fingertip design。每个任务采集 100 条 demonstrations 用于训练。

原始采样率：joint angles 100 Hz、F/T 100 Hz、gripper 15 Hz、三路相机 30 fps。所有模态按 global timestamp 同步，并降采样到统一 10 Hz；连续状态用 time-aware linear interpolation，视频取最近帧。F/T 先做 gravity/inertial compensation，再用 adjoint 变换到 TCP frame，随后 Butterworth low-pass filter 与 magnitude clipping。这些处理属于数据构造，不是推理时的在线力反馈。

### 5.3 仿真评测

本文未报告仿真评测。所有结果来自真实机器人实验；因此不能把本文结论外推到仿真基准或未见过的机器人平台。

### 5.4 真实机器人设置

真实任务有四个：Erase Whiteboard、Pivot、Cup Dispense、Pick and Place。前三个是 contact-rich 任务，分别覆盖 surface tracking under pose uncertainty、continuously varying compliance direction、force-regulated pressing；Pick and Place 用来检查标准操作是否被破坏。每个任务 100 条训练 demonstrations；评测时环境初始化随机化，每个 run 每任务 20 episodes。VLA 方法用同一个训练模型做 3 次独立 evaluation runs，报告 mean ± std；DP 与 ACT 只做 1 次 run。Erase 用 character-based success，其余任务用 episode success。episode 会因超时或 FR3 因过大接触力触发安全停而终止。

### 5.5 主要结果

论文 Table 1 的成功率如下（%）；VLA 结果为 3 次 evaluation runs 的 mean ± std，DP/ACT 为单次 run：

| 方法 | Erase | Pivot | Cup Dispense | Contact-rich 平均 | PnP |
|------|-------|-------|--------------|-------------------|-----|
| π0 | 17.8 ± 7.1 | 31.7 ± 10.4 | 45.0 ± 0.0 | 31.5 ± 5.2 | 80.0 ± 10.0 |
| ForceVLA | 68.1 ± 4.1 | 45.0 ± 15.0 | **50.0 ± 8.7** | 54.4 ± 4.3 | 45.0 ± 15.0 |
| TA-VLA | 52.8 ± 9.7 | 26.7 ± 10.4 | 0.0 ± 0.0 | 26.5 ± 2.0 | **83.3 ± 12.6** |
| CompVLA | **92.9 ± 3.7** | **73.3 ± 10.4** | 35.0 ± 10.0 | **67.1 ± 7.2** | 76.7 ± 5.8 |
| DP | 0.0 | 0.0 | 10.0 | 3.3 | 50.0 |
| ACT | 0.0 | 15.0 | 30.0 | 15.0 | 0.0 |

论文的主张是：CompVLA 在三个 contact-rich 任务上平均 67.1%，比最强基线高 12.7 个百分点；并且在不需 compliance 的 Pick and Place 上与 π0 可比（76.7% vs 80.0%）。但必须同时看到：Cup Dispense 上 CompVLA 35.0% 低于 ForceVLA 50.0%，作者把瓶颈归因于“准确到达小按钮”而非按压阶段。

### 5.6 消融与阶段分析

Table 2 在 Erase Whiteboard 上做 10 episodes 消融：完整 CompVLA 96.0%；去掉 virtual target shift 后 7.0%；去掉整个 Compliance Expert（position-only）后 15.3%。作者解释：`k` 与 `d` 是联合训练的，只保留 `k` 而不使用配对 `d` 会引入 position bias，接触力不足。

Table 3 在 Pivot 上做 10 episodes 消融：固定 stiffness（旋转 30 Nm/rad、平移 500 N/m）但保留 `d_t`，成功率 40%；完整 CompVLA 70%。这说明 variable stiffness 对 compliance direction 变化的任务有作用。

Table 4 做 Cup Dispense 阶段分析，两个 run、每阶段 20 trials：Reaching 阶段 π0 52.5%、CompVLA 45.0%；Pushing 阶段 π0 67.5%、CompVLA 92.5%。这与作者“reaching 是瓶颈、pushing 受益于 compliance”的解释一致，但也说明 CompVLA 的收益不是全任务无差别提升。

## 6. 能力与失败模式

### 6.1 能力

- 能在接触方向变化时保持柔顺，同时用正交高刚度维持轨迹；Pivot 与 Erase 的结果支持这一点。
- 能用 virtual displacement 产生按压力，并在 Cup Dispense 的 pushing 阶段把成功率从 67.5% 提到 92.5%。
- 推理时不需要 external F/T 传感器；F/T 用于训练标签构造，也用于评测中的接触力测量证据。
- 在 standard Pick and Place 上没有明显破坏，76.7% 与 π0 80.0% 接近。

### 6.2 失败模式

- **依赖 reference motion**：作者明确说 compliance 不能补偿大轨迹误差；Cup Dispense 的小按钮 reaching 只有 45.0%，低于 π0 的 52.5%。这说明“先到接触区”仍是瓶颈。
- **标签依赖 F/T**：compliance targets 从 force/torque 测量推导；接触建立后工作较好，但接触前的期望柔顺缺乏直接监督。
- **主导方向假设**：当前方法聚焦一个 dominant compliance direction；多方向柔顺或 rotational/translational 强耦合任务可能不足。
- **控制器稳定性不是保证**：实现是简化 GIC，省略 feedforward、依赖低层 gravity compensation、使用 decoupled error，且 stiffness 时变；论文只声称稳定性结果“motivates”控制器。
- **任务与平台范围有限**：四个真实任务、单一 FR3 平台、每任务 100 demos；没有仿真评测，也没有外部复现。DP/ACT 只跑一次，VLA 基线虽跑三次但仍是小样本 evaluation。
- **未比较的基线**：VTLA 因平台无 tactile sensing 被排除；ForceVLA2 与 CompliantVLA-adaptor 因实现未公开未比较。因此不能把 CompVLA 说成全面优于所有 compliance-aware VLA。

### 6.3 隐含假设

- 视觉、机器人状态与 reference trajectory 足以让 Compliance Expert 在推理时推断接触方向；论文没有证明在视觉遮挡或接触前预测时这一假设总成立。
- 训练时的 F/T 标签能代表评测时的接触物理；传感器校准、滤波、夹爪与工件变化可能改变标签分布。
- 单一 dominant compliance axis 足以覆盖任务；对多接触点、多方向耦合任务未必成立。
- 基线共用同一 GIC 且 stiffness 固定为 CompVLA 的最大值；这提供了受控比较，但也意味着基线不是“各自最优的 compliance 控制器”。
- 未来 action chunk 不被当作历史记忆；论文没有显式 memory 模块。若任务需要长历史，不能从 `T=25` 推导出记忆能力。

## 7. 与相关工作对比

| 工作 | 关注点 | 动作/输出 | 柔顺机制 | 本论文中的角色 |
|------|--------|-----------|----------|----------------|
| π0 | 通用 VLA | kinematic twist + gripper | 无显式 compliance 输出；基线共用 GIC，stiffness 固定最大 | 基线；contact-rich 平均 31.5% |
| ForceVLA | force-aware VLA | pose 动作；6-axis wrench 作为输入增强 | 输入增强，不直接输出 stiffness/d | 基线；平均 54.4% |
| TA-VLA | torque-aware VLA | 动作 + future torque 辅助训练；joint torque 输入 | 不直接输出 stiffness/d | 基线；平均 26.5% |
| ForceVLA2 | hybrid force-position | pose + target wrench，Cross-Scale MoE | 输出 target wrench，不是 `k/d` | 未评测（实现未公开） |
| CompliantVLA-adaptor | variable impedance | 额外 VLM 选择 stiffness/damping gains | 外挂 VLM，基于 task phase 与 F/T | 未评测（实现未公开） |
| CompVLA | VLA + variable compliance | `V_t, g_t, k_t, d_t` | Compliance Expert 预测 `k/d`，GIC 执行方向柔顺与力调制 | 本文方法；平均 67.1% |

**面试 Tip**：被问“CompVLA 和 ForceVLA/TA-VLA 的区别”时，不要只说“它用力传感器”。准确答法是：CompVLA 把 stiffness 与 virtual displacement 变成策略输出，`d_t` 同时负责力调制和柔顺主轴；它推理时不用 external F/T，但训练标签来自 F/T；controller 是简化 GIC，稳定性不是论文保证。

## 8. 精读建议

- **值得精读原文的人**：做 contact-rich manipulation 的研究者；设计 VLA action space 的工程师；想把 variable impedance 接到真实机器人上的控制/学习交叉背景读者。
- **建议章节路径**：先读 §4.1 看 `k_t` 与 `d_t` 如何从演示构造 → 再看 §4.2 看 dual-expert 与 trajectory condition → 然后读 Appendix A.2 看 virtual target 与 decoupled error 的实现 → 最后看 §5 的 Table 1/2/3/4，注意 Cup Dispense 的反例。
- **不值得精读的理由**：如果你不做机器人学习、不关心 contact-rich 任务，或已经熟悉 variable impedance 与 GIC，读摘要与 Table 1 即可；如果你只关心大规模通用 VLA 预训练，这篇的重点不在预训练规模，而在 action representation 与 controller 接口。

---

[← Back to Theory](./README.md)

**关键引用**：论文链接 https://arxiv.org/abs/2609.23614