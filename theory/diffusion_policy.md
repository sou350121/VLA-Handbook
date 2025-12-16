# 扩散策略详解 (Diffusion Policy)

> **核心论文**: [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137) (Cheng Chi et al., RSS 2023)
> **代表模型**: **Octo**, **MimicGen**, **Toyota HPT**

## 0. 主要數學思想 (Main Mathematical Idea)

> **第一性原理**: **Constructing Structure from Chaos (从混沌中构建秩序)**

物理世界中，熵增（有序变无序）是自然趋势，如墨水在水中扩散。Diffusion Model 试图在数学上**逆转**这一时间过程。

- **核心数学工具**: **Langevin Dynamics (朗之万动力学)** 与 **Score Matching**。
- **解题逻辑**:
    1.  **逆向思维**: 如果我们知道数据是如何一步步变成噪声的（前向过程），那么只要学会每一步的"逆操作"（去噪），就能从纯噪声中恢复出数据。
    2.  **梯度指引**: 学习数据分布的梯度场（Score Function，$\nabla_x \log p(x)$）。这就像在迷雾（噪声）中，每一步都沿着"数据密度更高"的方向走一小步，最终必然会走到数据流形上（生成合理的动作）。
    3.  **多模态**: 不像回归模型寻找"平均值"，扩散模型学习的是整个地形（Landscape），因此可以从同一个噪声起点走到不同的终点（解决多解问题）。

## 1. 为什么需要 Diffusion Policy? (Why?)

在 VLA 出现之前，主流的动作生成方式是 **MSE Regression** (均方误差回归) 或 **GMM** (高斯混合模型)。

### 1.1 多模态分布问题 (The Multimodality Problem)
机器人经常面临"多解"情况。例如，绕过障碍物可以**从左绕**也可以**从右绕**。
- **MSE (均值回归)**: 会预测出左和右的平均值 -> **直直撞向障碍物**。
- **Diffusion**: 可以完美拟合双峰分布，随机采样出"左"或"右"的一条完整轨迹，而不会取平均。
    - **Energy-Based Model (EBM) 视角**: 我们可以把 Diffusion 看作是在学习一个能量函数 $E(x)$。真实数据的能量低，噪声的能量高。
    - MSE 试图最小化单一模态的误差，相当于在两个低谷之间强行找一个"平均低谷" (往往是能量很高的高地)。
    - Diffusion 则是学习整个地貌 (Landscape)，允许存在多个分离的低谷 (Modes)。

### 1.2 连续空间的高精度 (High Precision)
相比于 Tokenization (RT-1) 将动作离散化为 256 个桶，Diffusion 直接在连续空间生成浮点数，精度理论上无限，非常适合**穿针引线、精密装配**等任务。

## 2. 数学原理 (Mathematical Formulation)

Diffusion Policy 将动作生成建模为一个 **条件去噪过程 (Conditional Denoising Process)**。

### 2.1 前向过程 (Forward Process / Diffusion)
将真实的动作轨迹 $x_0$ 逐步加噪，变成纯高斯噪声 $x_T$。

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)
$$

经过 $t$ 步后，可以直接写出 $x_t$ 与 $x_0$ 的关系：
$$
q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)
$$

其中 $\alpha_t = 1 - \beta_t$, $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$。

### 2.2 噪声调度器 (Noise Scheduler)
$\beta_t$ 的选择至关重要，通常有两种策略：
- **Linear Schedule**: $\beta_t$ 从 $\beta_{min}=10^{-4}$ 线性增加到 $\beta_{max}=0.02$。
    - $\beta_t = \beta_{min} + \frac{t}{T}(\beta_{max} - \beta_{min})$
- **Cosine Schedule**: $\beta_t$ 随余弦函数变化，能更好地保留中间时刻的信息，防止噪声过早"淹没"信号。
    - $\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos^2 \left( \frac{t/T + s}{1+s} \cdot \frac{\pi}{2} \right)$
    - 这种调度在 $t$ 较小时噪声增加得很慢，保留了更多原始信号，对微小动作的生成更有利。

### 2.3 逆向过程 (Reverse Process / Denoising)
训练一个神经网络 $\epsilon_\theta(x_t, t, \text{Obs})$ 来预测噪声。
- **输入**: 当前带噪动作 $x_t$，时间步 $t$，观测条件 $\text{Obs}$ (图像/语言)。
- **输出**: 预测的噪声 $\hat{\epsilon}$。

去噪公式 (DDPM):

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t, \text{Obs}) \right) + \sigma_t z
$$

其中 $z \sim \mathcal{N}(0, I)$ 是随机噪声 (但在最后一步 $t=0$ 时设为 0)。$\sigma_t$ 是方差项，通常取 $\sqrt{\beta_t}$ 或 $\tilde{\beta}_t$。

### 2.4 损失函数 (Loss Function)
非常简单，就是预测噪声与真实噪声的 MSE：

$$
\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t, \text{Obs}) \|^2 \right]
$$


## 3. 网络架构 (Network Architecture)

Diffusion Policy 的核心是那个预测噪声的网络 $\epsilon_\theta$。主要有两种流派：

### 3.1 CNN-based (1D Temporal CNN / U-Net)
- **原理**: 将动作轨迹看作是一个 1D 的时间序列 (Sequence Length = $T_p$, Action Dim = $D_a$)。
- **结构**: 使用类似于 U-Net 的结构，但在时间维度上进行下采样和上采样 (Downsample/Upsample)。
    - **Conditioning**: 图像特征 (ResNet/ViT) 和 语言特征 (CLIP) 通常通过 **FiLM (Feature-wise Linear Modulation)** 层注入到 U-Net 的每个 Residual Block 中。
    - **Downsample**: Conv1d + GroupNorm + Mish
    - **Upsample**: ConvTranspose1d
    - **Skip Connection**: 将 Encoder 的特征拼接到 Decoder，保留高频细节。
- **特点**: 计算效率高，适合处理短时序依赖，是 Diffusion Policy 论文中的默认选择。

### 3.2 Transformer-based (DiT / Octo)
- **原理**: 将动作轨迹切成 Patch，或者直接作为 Token 输入 Transformer。
- **结构**: 标准的 Transformer Encoder/Decoder (如 DiT - Diffusion Transformer)。
- **特点**: 
    - 能够处理更长的时间依赖。
    - **多模态融合**: 可以直接 Cross-Attention 图像 Patch 和 语言 Token。
    - **Scalability**: 参数量可以做得很大 (e.g., Octo-Base 93M, Octo-Small 27M)，适合作为 Foundation Model。

## 4. 推理加速 (Inference Acceleration)

Diffusion 的最大缺点是慢。DDPM 需要 100 步去噪，推理一次可能要几百毫秒，无法满足机器人 50Hz 的控制要求。

### 4.1 DDIM (Denoising Diffusion Implicit Models)
- **原理**: 将随机游走过程变为确定性过程 (Deterministic)，跳过中间步骤。DDIM 重新定义了前向过程，使得它是一个非马尔可夫过程，从而允许更大的步长。
- **公式**:

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\left( \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta}{\sqrt{\bar{\alpha}_t}} \right)}_{\text{predicted } x_0} + \underbrace{\sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \epsilon_\theta}_{\text{direction pointing to } x_t} + \sigma_t \epsilon_t
$$

- **效果**: 可以将步数从 100 步压缩到 **10-15 步**，同时保持较高的生成质量。

### 4.2 Receding Horizon Control (RHC)
- **原理**: 模型一次预测未来 $H$ 步的动作 (Action Chunk)，但机器人只执行前 $M$ 步 (例如 $M=8$)，然后重新推理。
- **优势**: 掩盖了推理延迟，保证动作的连贯性。

## 5. 面试常见问题 (Q&A)

**Q: Diffusion Policy 和 GAN 有什么区别?**
A: GAN 容易模式坍塌 (Mode Collapse)，即只学会一种解；Diffusion 训练更稳定，能覆盖所有模态 (Mode Coverage)。

**Q: 为什么 Diffusion 推理慢? 如何解决?**
A: 因为它是迭代去噪。解决方法包括使用 DDIM 减少步数，或者使用 Consistency Distillation (一致性蒸馏) 将步数压缩到 1 步。

**Q: 什么是 Action Chunking?**
A: 一次预测一段未来的动作序列，而不是只预测下一步。这利用了动作的时间相关性，提高了平滑度。


---
[← Back to Theory](./README.md)
