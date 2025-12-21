# 传统动作生成方法 (Traditional Action Generation)

在 Diffusion Policy 和 Action Tokenization 成为主流之前，机器人模仿学习 (Imitation Learning) 主要依赖 **MSE 回归** 和 **GMM (高斯混合模型)**。理解这些方法的原理及其局限性，有助于深刻理解为什么要引入 Diffusion。

## 0. 主要数学思想 (Main Mathematical Idea)

> **第一性原理**: **Deterministic vs. Probabilistic Modeling (确定性 vs. 概率建模)**

- **MSE**: 假设动作是观测的确定性函数 $a = f(s)$。这对应于假设数据服从**单峰高斯分布**且方差固定。
- **GMM**: 假设动作服从**多峰高斯分布** $p(a|s) = \sum w_i \mathcal{N}(\mu_i, \sigma_i)$。这是一种参数化的密度估计方法。

---

## 1. MSE 回归 (Mean Squared Error Regression)

这是最简单、最直观的行为克隆 (Behavior Cloning, BC) 方法。

### 1.1 核心思想
直接训练一个神经网络（如 ResNet + MLP），输入观测 $s$，输出动作 $a$。训练目标是最小化预测动作与专家动作之间的欧几里得距离。

### 1.2 数学公式
损失函数 (Loss Function) 为 L2 Loss：

$$
\mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^N \| \pi_\theta(s_i) - a_i \|^2
$$

这本质上等价于最大似然估计 (MLE)，假设动作服从固定方差的单峰高斯分布：

$$
p(a|s) = \mathcal{N}(\pi_\theta(s), \sigma^2 I)
$$

### 1.3 优缺点
- **优点**:
    - **极简**: 实现简单，训练速度快，推理极快 (一次前向传播)。
    - **平滑**: 输出是连续值，通常比较平滑。
- **缺点**:
    - **多模态灾难 (Multimodal Failure)**: 这是 MSE 最大的死穴。
        - **场景**: 面前有一个障碍物，专家数据中有 50% 次从左绕，50% 次从右绕。
        - **结果**: MSE 会学习两者的**平均值** —— 直接撞向障碍物中间。
        - **原因**: 两个合法动作的均值通常是不合法的。

---

## 2. 高斯混合模型 (GMM / MDN)

为了解决多模态问题，研究者引入了 **混合密度网络 (Mixture Density Networks, MDN)**，其核心输出是 GMM。

### 2.1 核心思想
不直接预测一个动作，而是预测 **$K$ 个高斯分布的参数** (权重、均值、方差)。这使得模型可以同时表示多个可能的动作模式（如"左绕"模式和"右绕"模式）。

### 2.2 数学公式
概率密度函数定义为 $K$ 个高斯分量的加权和：

$$
p(a|s) = \sum_{k=1}^K \pi_k(s) \mathcal{N}(a; \mu_k(s), \Sigma_k(s))
$$

其中神经网络的输出包括：
1.  **混合系数 (Weights)** $\pi_k$: 满足 $\sum \pi_k = 1$ (通过 Softmax)。
2.  **均值 (Means)** $\mu_k$: 每个分量的中心。
3.  **方差 (Variances)** $\Sigma_k$: 每个分量的离散程度 (通常输出 log 方差并取 exp)。

**损失函数**: 负对数似然 (Negative Log-Likelihood, NLL)：

$$
\mathcal{L}_{GMM} = - \log \left( \sum_{k=1}^K \pi_k \mathcal{N}(a_{expert}; \mu_k, \Sigma_k) \right)
$$

### 2.3 优缺点
- **优点**:
    - **解决多模态**: 可以分别拟合"左绕"和"右绕"两个波峰，推理时可以选择概率最大的那个，而不是取平均。
    - **不确定性估计**: 可以通过方差判断模型对当前动作的信心。
- **缺点**:
    - **数值不稳定**: 训练过程中 $\Sigma$ 容易塌缩到 0 或发散，导致 Loss NaN。需要大量 Trick (如 min variance clipping)。
    - **模式塌缩 (Mode Collapse)**: 模型往往倾向于只使用其中一两个高斯分量，忽略其他分量。
    - **超参数敏感**: $K$ 的数量难以确定。$K$ 太小不够用，$K$ 太大难以训练。
    - **高维困难**: 在高维动作空间 (如双臂 14 DoF) 中，拟合 GMM 非常困难。

---

## 3. 为什么 VLA 转向了 Diffusion?

下表对比了传统方法与现代 Diffusion Policy：

| 特性 | MSE Regression | GMM / MDN | Diffusion Policy |
| :--- | :--- | :--- | :--- |
| **多模态支持** | ❌ (取平均) | ✅ (有限) | ✅✅ (任意分布) |
| **训练稳定性** | ⭐⭐⭐⭐⭐ | ⭐⭐ (易 NaN) | ⭐⭐⭐⭐ |
| **高维扩展性** | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **推理速度** | 🚀 极快 | 🚀 极快 | 🐢 较慢 (需迭代) |
| **表达能力** | 仅单峰 | 受限于 K | 极强 (非参数化) |

> **面试 Tip**: 如果被问到 "既然 GMM 也能解决多模态，为什么还要用 Diffusion?"
> **答**: GMM 虽然理论上可行，但在工程上极难训练（数值不稳定），且在高维空间表现不佳。Diffusion 通过**迭代去噪**的方式，将复杂的分布拟合问题转化为了简单的 MSE 回归问题（预测噪声），既保留了多模态能力，又获得了类似 MSE 的训练稳定性。

---
[← Back to Theory](./README.md)

























































