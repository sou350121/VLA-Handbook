# VLA 数学必备：从直觉到实作

> **这不是数学教材**，而是"为了能看懂 VLA 论文、写出 loss、调好机器人"的实践清单。
> 每个公式都配有 **大白话解释** + **在 VLA 里怎么用**。

---

## 目录

- [1. 线性代数](#1-线性代数空间与变换)
- [2. 微积分与优化](#2-微积分与优化)
- [3. 3D 几何与旋转](#3-3d-几何与旋转)
- [4. 概率与信息论](#4-概率与信息论)
- [5. Transformer 数学](#5-transformer-数学)
- [6. 生成模型](#6-生成模型扩散与-flow-matching)
- [7. 强化学习](#7-强化学习)
- [8. 机器人控制](#8-机器人控制)
- [9. 训练工程](#9-训练工程)
- [速成清单](#-2-周速成清单)

---

## 1. 线性代数：空间与变换

### 内积与相似度

$$
\text{sim}(A, B) = A \cdot B = \sum_i A_i B_i = \|A\| \, \|B\| \cos\theta
$$

**大白话**：两个特征向量的"夹角"。对应项相乘再加总，结果大 = 方向一致 = "长得像"。

**VLA 里怎么用**：Attention 机制中 Q 和 K 的匹配就是在算内积。CLIP 对比学习也是用余弦相似度把"图像"和"文字"拉近。

---

### 线性变换 (Linear Layer)

$$
y = Wx + b
$$

- `x`：输入特征（如视觉 embedding）
- `W`：权重矩阵——一台"空间搬运机"
- `y`：输出特征（如动作空间 embedding）

**大白话**：通过矩阵 `W` 把信息从一个空间搬到另一个空间。VLA 里就是把"看到的东西"搬到"动作的语言"。

---

### 低秩分解 (LoRA)

$$
\Delta W = AB, \quad A \in \mathbb{R}^{d \times r}, \; B \in \mathbb{R}^{r \times k}, \quad r \ll d
$$

**大白话**：巨大的权重更新 `ΔW` 被拆成两个"窄瓶颈"矩阵相乘。只学最核心的变化方向，省 90%+ 显存。

**VLA 里怎么用**：OpenVLA 用 QLoRA（4-bit 量化 + LoRA）在消费级显卡上微调 7B 模型。π0.5 用 LoRA 做跨任务适配。
→ 详见 [PEFT & LoRA](peft_lora.md) · [DoRA](dora_weight_decomposed_low_rank_adaptation.md)

---

### SVD（奇异值分解）

$$
A = U \Sigma V^\top
$$

- `U`：左奇异向量（数据的主方向）
- `Σ`：奇异值对角矩阵（每个方向的"重要程度"）
- `V`：右奇异向量

**大白话**：把任何矩阵拆成"旋转 → 拉伸 → 旋转"三步。扔掉小的奇异值 = 去噪 + 压缩。

**VLA 里怎么用**：模型压缩（只保留前 k 个奇异值）、特征分析（找数据的主成分）。

---

## 2. 微积分与优化

### 链式法则 (Backpropagation 的核心)

$$
\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial y}{\partial x}
$$

**大白话**：最终错误 `L` 一级级往回追究责任。每个参数对错误贡献多少？一层层乘回去就知道了。整个反向传播就是在用链式法则。

---

### 雅可比矩阵 (Jacobian)

$$
J(q) = \frac{\partial f(q)}{\partial q}, \quad \dot{x} = J(q)\,\dot{q}
$$

- `q`：关节角度向量
- `x`：末端执行器位置
- `J`：关节→末端的"换算率"

**大白话**：关节转了一点点，手指在空间里飘了多远、往哪飘？Jacobian 就是这个"换算矩阵"。

**VLA 里怎么用**：
- 逆运动学（IK）：已知手指要去哪，反推关节怎么转
- 力控制：关节扭矩 → 末端力的映射
- VLA 输出的 `Δx`（末端位移）需要通过 `J⁻¹` 转成关节命令

---

### 梯度下降

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}
$$

**大白话**：站在山坡上，往最陡的下坡方向走一步（步长 η）。重复直到走到谷底。所有深度学习训练都是在做这件事。

---

## 3. 3D 几何与旋转

### SE(3) 齐次变换

$$
T = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix} \in SE(3), \quad P_{\text{world}} = T_{\text{cam}}^{\text{world}} \cdot P_{\text{cam}}
$$

- `R`：3×3 旋转矩阵（SO(3)）
- `t`：3×1 平移向量
- `T`：4×4 齐次变换矩阵

**大白话**：相机看到一个点在 `P_cam`，通过变换矩阵 `T` 就能算出它在世界坐标系的位置。这是机器人"手眼协调"的数学基础。

**VLA 里怎么用**：
- 相机标定：`T_cam_to_base` 把像素坐标变成机械臂坐标
- VLA 输出的 action 通常是 `(Δx, Δy, Δz, Δroll, Δpitch, Δyaw)` 的 SE(3) 增量

---

### 旋转表示：为什么不用欧拉角？

| 表示 | 维度 | 连续性 | 万向锁 | VLA 常用？ |
|------|:----:|:------:|:------:|:---------:|
| 欧拉角 (RPY) | 3 | ❌ 不连续 | ❌ 有 | 仅人类界面 |
| 四元数 | 4 | 🔶 需归一化 | ✅ 无 | 一般 |
| 旋转矩阵 | 9 | ✅ 连续 | ✅ 无 | 一般 |
| **6D 旋转** | 6 | ✅ 连续 | ✅ 无 | **推荐** |
| 轴角 (Axis-Angle) | 3 | ❌ 不连续 | ✅ 无 | π0 用 |

**6D 旋转** 取旋转矩阵的前两列（6 个数），第三列通过叉积恢复：

$$
c_3 = c_1 \times c_2, \quad R = [c_1 \mid c_2 \mid c_3]
$$

**大白话**：神经网络预测 6 个连续的数，比预测 3 个不连续的欧拉角容易得多。这就是为什么几乎所有 VLA 论文都用 6D 旋转表示。

> 💡 **实践坑**：π0 系列用轴角（axis-angle）+ 归一化，OpenVLA 用离散化欧拉角，RDT 用 6D。选错表示会导致训练不收敛。

---

## 4. 概率与信息论

### VLA 的概率本质

$$
\pi(a \mid s, g) = P(\text{action} \mid \text{image}, \text{instruction})
$$

**大白话**：VLA 的本质就是一个条件概率模型。给定当前画面 `s` 和语言指令 `g`，输出每个动作 `a` 的可能性。训练就是让这个概率分布尽量接近专家的分布。

---

### KL 散度

$$
D_{\text{KL}}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)}
$$

**大白话**：衡量两个概率分布 `p` 和 `q` 有多不一样。KL = 0 表示完全一致。

**VLA 里怎么用**：
- **CVAE 训练**（ACT 模型）：KL 约束隐变量 z 不要偏离先验太远
- **RL 后训练**（π\*0.6 RECAP）：约束 RL 策略不要偏离 BC 策略太远
- **知识蒸馏**：学生模型的输出分布要接近教师模型

---

### 信息熵

$$
H(X) = -\sum_i P(x_i) \log P(x_i)
$$

**大白话**：熵大 = 模型很迷茫（乱猜），熵小 = 非常有把握。

**VLA 里怎么用**：Token 类 VLA（RT-2）用 temperature 控制采样熵。温度低 → 只选最高概率的动作（保守）；温度高 → 多探索。

---

### ELBO（变分下界）

$$
\log p(x) \geq \mathbb{E}_{q(z|x)} \big[\log p(x|z)\big] - D_{\text{KL}}\big(q(z|x) \| p(z)\big)
$$

**大白话**：我们想最大化数据的概率 `p(x)`，但直接算太难。ELBO 给了一个可以优化的下界：第一项是"重建得多好"，第二项是"隐变量 z 别太离谱"。

**VLA 里怎么用**：**ACT 模型的核心**。编码器把示范轨迹压缩成隐变量 z（意图），解码器从 z 重建动作序列。ELBO 平衡重建精度和隐空间正则化。
→ 详见 [ACT 详解](../vla-core/act.md)

---

## 5. Transformer 数学

### 缩放点积注意力

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

- `Q`（Query）：我在找什么？（来自当前 token）
- `K`（Key）：这里有什么？（来自所有 token）
- `V`（Value）：如果匹配上了，给我什么信息？
- `√d_k`：缩放因子，防止内积太大导致 softmax 饱和

**大白话**：Query 去每个 Key 那里"问一嘴"——你跟我相关吗？相关度高的 Key 对应的 Value 权重就大。最后加权求和得到输出。

**VLA 里怎么用**：
- **Self-attention**：图像 patch 之间互相看（ViT）
- **Cross-attention**：语言 token 去图像里找对应物体（vision-language fusion）
- **Action cross-attention**：动作 query 去视觉特征里找"该往哪伸手"（π0 的 action head）

---

### Flash Attention

$$
\text{tiling}: \quad Q_i K_j^\top \text{ 在 SRAM 中计算，不存完整 } N \times N \text{ 矩阵}
$$

**大白话**：标准 Attention 要存一个 N×N 的注意力矩阵（N = 序列长度），显存爆炸。Flash Attention 把它切成小块在 GPU 高速缓存里算，显存从 O(N²) 降到 O(N)，速度还快 2-4x。

→ 详见 [Flash Attention](flash_attention.md)

---

## 6. 生成模型：扩散与 Flow Matching

> 这是 VLA 的 **Action Head** 核心数学。动作不是"预测"出来的，是"生成"出来的。

### 为什么需要生成模型？

同一个任务往往有多条正确路径（从左边绕 or 从右边绕）：

```
MSE 回归：预测两条路的"平均" → 撞到障碍物 💥
生成模型：采样其中一条完整路径 → 安全到达 ✅
```

这就是为什么 VLA 不用简单回归，而用 Diffusion 或 Flow Matching 来**生成**动作。

---

### Diffusion（扩散模型）

**前向过程**（加噪）：

$$
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

**反向过程**（去噪 = 生成）：

$$
\hat{\epsilon}_\theta(x_t, t) \approx \epsilon, \quad x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\hat{\epsilon}_\theta\right) + \sigma_t z
$$

**训练目标**：

$$
\mathcal{L}_{\text{diffusion}} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \hat{\epsilon}_\theta(x_t, t)\|^2\right]
$$

**大白话**：
1. 把干净的动作 `x₀` 逐步加噪声，变成纯噪声 `x_T`
2. 训练一个网络学会"去噪"——给它带噪声的 `x_t`，让它预测加了多少噪声 `ε`
3. 生成时：从纯噪声开始，反复去噪，一步步"画出"干净的动作

**VLA 里怎么用**：RDT、Octo、Diffusion Policy 都用这个。缺点：需要多步去噪（10-100 步），推理慢。

---

### Flow Matching（流匹配）

**核心思想**：在噪声 `x₁` 和目标 `x₀` 之间画一条直线，学习沿着这条线的"速度场"。

**速度场**：

$$
v_t = x_0 - x_1 \quad \text{(最简单的线性插值)}
$$

$$
x_t = (1-t)\,x_1 + t\,x_0
$$

**训练目标**：

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, x_0, x_1}\left[\|v_\theta(x_t, t) - (x_0 - x_1)\|^2\right]
$$

**生成**（ODE 求解）：

$$
\frac{dx}{dt} = v_\theta(x_t, t), \quad x_0 = x_1 + \int_0^1 v_\theta(x_t, t)\,dt
$$

**大白话**：
1. 在噪声和目标之间拉一条直线
2. 训练网络预测"当前位置该往哪走"（速度场 v）
3. 生成时：从噪声出发，沿着学到的速度场"滑"到目标

**vs Diffusion**：Flow Matching 的路径是直线（不是弯曲的扩散路径），所以只需要 1-5 步就能生成，比 Diffusion 快 10x+。

**VLA 里怎么用**：**π0 系列的核心**。π0 用 Flow Matching 作为 Action Head，实现 30Hz 的实时动作生成。
→ 详见 [Diffusion Policy](../diffusion-flow/diffusion_policy.md) · [π0 Flow Matching](../vla-core/pi0_flow_matching.md) · [Compression Gap](../diffusion-flow/the_compression_gap_why_discrete_tokenization_limits_vision_dissection.md)

---

## 7. 强化学习

### 贝尔曼方程

$$
V(s) = \max_a \left[R(s,a) + \gamma \, V(s')\right]
$$

**大白话**：现在的价值 = 现在的甜头 + 打折后的未来期望。`γ`（折扣因子）越小，机器人越"急功近利"。

---

### 策略梯度

$$
\nabla_\theta J = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot A(s,a)\right]
$$

- `π_θ(a|s)`：策略（模型输出的动作概率）
- `A(s,a)`：优势函数（这次比平均好多少）

**大白话**：结果比平均好（A > 0）→ 给这次动作的概率加仓；搞砸了（A < 0）→ 减仓。这就是 REINFORCE 算法。

---

### PPO 裁剪

$$
\mathcal{L}_{\text{PPO}} = -\min\left(r_t A_t, \; \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\,A_t\right)
$$

$$
r_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

**大白话**：策略更新时，如果新策略和旧策略差太多（比率 `r` 偏离 1 太远），就裁掉。防止一步迈太大翻车。

**VLA 里怎么用**：π\*0.6 的 RECAP 算法用了类似思路——offline RL 复盘时约束新策略不要偏离 BC 策略太远。
→ 详见 [VLA+RL 实战](../rl/vla_rl_practical_guide.md)

---

## 8. 机器人控制

### PD 控制

$$
u = K_p \, e + K_d \, \dot{e}, \quad e = x_{\text{target}} - x_{\text{current}}
$$

**大白话**：
- `Kp · e`：离目标远就使劲追（比例项）
- `Kd · ė`：快到了就踩刹车（微分项 = 阻尼）

**VLA 里怎么用**：VLA 输出的是"目标位置"（position setpoint），底层控制器用 PD 把关节驱动到那里。Figure Helix 02 的 S0 层就是 1kHz 的 PD 控制。

---

### 卡尔曼滤波

$$
\hat{x}_k = \hat{x}_k^- + K_k(z_k - H\hat{x}_k^-)
$$

- `K_k`：卡尔曼增益（"该信谁"的权重）
- `z_k`：传感器测量值
- `ĥx⁻_k`：模型预测值

**大白话**：模型预测说手臂在 A，传感器说手臂在 B。谁准听谁的——卡尔曼增益 K 自动算出最优折中。

---

### 动作平滑

$$
\mathcal{L}_{\text{smooth}} = \sum_t \|a_{t+1} - a_t\|^2 + \lambda \|a_{t+2} - 2a_{t+1} + a_t\|^2
$$

**大白话**：惩罚相邻动作的突变（一阶）和加速度的突变（二阶）。让机器人动起来像丝绸一样顺滑，不抖。

---

## 9. 训练工程

### 梯度裁剪

$$
g \leftarrow g \cdot \min\left(1, \frac{c}{\|g\|}\right)
$$

**大白话**：梯度太大就等比例缩小到阈值 `c`，防止权重被"踢飞"。几乎所有 VLA 训练都开梯度裁剪。

---

### Adam 优化器

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(动量)}
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{(自适应学习率)}
$$

$$
\theta_t = \theta_{t-1} - \eta \frac{m_t}{\sqrt{v_t} + \epsilon}
$$

**大白话**：
- `m`：看看之前几步的梯度方向（惯性），避免来回摆
- `v`：看看这个参数的梯度波动大不大，波动大的走慢点
- 结合起来：方向稳 + 步长自适应。VLA 训练标配。

---

### Temperature Scaling

$$
P(a_i) = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}
$$

**大白话**：温度 `τ` 控制动作选择的"保守 vs 冒险"：
- `τ → 0`：只选最高分动作（贪心）
- `τ = 1`：按原始概率采样
- `τ → ∞`：完全随机

---

## 📅 2 周速成清单

| 天数 | 主题 | 实践任务 |
|:----:|------|---------|
| 1-2 | 线性代数 | 手推 LoRA 的参数节省量；用 NumPy 实现 SVD 压缩一张图 |
| 3-4 | 3D 几何 | 实现 SE(3) 坐标变换；写 6D→旋转矩阵的转换代码 |
| 5-6 | 概率论 | 推导 MSE = 高斯 NLL 的等价性；实现一个带重参数化的 VAE |
| 7-8 | Transformer | 手绘 Q·K·V 匹配过程；从零实现 scaled dot-product attention |
| 9-10 | 生成模型 | 对比 Diffusion vs Flow Matching 的代码（<50 行）；理清前向过程 |
| 11-12 | RL | 理解 PPO clip 的逻辑；写一个 CartPole 的 REINFORCE |
| 13-14 | 联调 | 在一个训练循环中加入梯度裁剪 + EMA + 动作平滑，观察曲线变化 |

---

## 进一步阅读

| 主题 | 推荐 |
|------|------|
| Loss 函数大全 | [VLA Loss Functions Handbook](vla_loss_functions_handbook.md) |
| LoRA / DoRA 微调 | [PEFT & LoRA](peft_lora.md) · [DoRA](dora_weight_decomposed_low_rank_adaptation.md) |
| 量化推理 | [量化理论](quantization_theory.md) |
| 扩散策略 | [Diffusion Policy 详解](../diffusion-flow/diffusion_policy.md) |
| Flow Matching | [π0 代码解析](../vla-core/pi0_code_analysis.md) |
| ACT 的 CVAE | [ACT 详解](../vla-core/act.md) |
| 机器人控制 | [机械臂控制](../deployment/robot_control.md) |
| VLA+RL | [VLA+RL 实战](../rl/vla_rl_practical_guide.md) |

---

[← Back to Explorer's Map](../README.md)
