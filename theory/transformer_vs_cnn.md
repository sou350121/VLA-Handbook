# Transformer vs CNN: 核心架构对比

## 主要数学思想 (Main Mathematical Idea)

> **"Global Attention vs. Local Connectivity (归纳偏置的权衡)"**

Transformer 和 CNN 的核心对立在于如何处理信息：
1.  **CNN (卷积)**: 假设 "相邻的像素是相关的" (**Local Connectivity**) 和 "特征在哪里并不重要" (**Translation Invariance**)。这是一种强烈的**归纳偏置 (Inductive Bias)**。
2.  **Transformer (注意力)**: 假设 "任何两个像素之间都可能相关" (**Global Attention**)。它没有预设的偏置，完全依赖数据驱动的关系发现 (**Content-based Interactions**)。

从数学上讲，CNN 的卷积核是一个**固定**的局部算子 $y_i = \sum_j w_{i-j} x_j$，而 Transformer 的注意力是一个**动态**的全局算子 $y_i = \sum_j \text{softmax}(q_i^T k_j) v_j$，其中权重取决于输入本身。

---

在 VLA (Vision-Language-Action) 面试中，理解 Backbone (骨干网络) 的差异至关重要。虽然现在的趋势是 Transformer (ViT) 一统天下，但 CNN (ResNet, EfficientNet) 依然在 RT-1 等经典模型中扮演重要角色。

## 1. 核心差异一览表

| 特性 | CNN (卷积神经网络) | Transformer (自注意力机制) |
| :--- | :--- | :--- |
| **核心操作** | Convolution (卷积) | Self-Attention (自注意力) |
| **感受野 (Receptive Field)** | 局部 (Local) -> 逐渐扩大 | 全局 (Global) |
| **归纳偏置 (Inductive Bias)** | 强 (平移不变性, 局部性) | 弱 (需海量数据学习) |
| **计算复杂度** | $O(N)$ (线性) | $O(N^2)$ (平方级) |
| **并行性** | 高 | 极高 |
| **擅长领域** | 图像纹理, 边缘检测 | 语义理解, 长距离依赖 |
| **代表模型** | ResNet, EfficientNet | ViT, BERT, GPT |

## 2. CNN (Convolutional Neural Networks)

### 原理
CNN 模仿生物视觉皮层，通过**滑动窗口 (Sliding Window)** 提取特征。
- **局部性 (Locality)**: 每次只看一小块区域 (e.g., 3x3 像素)。
- **平移不变性 (Translation Invariance)**: 猫在图片左上角还是右下角，识别出的特征是一样的。
- **层级结构**: 浅层学边缘/纹理，深层学形状/物体。

### 在 VLA 中的应用
- **RT-1**: 使用 **EfficientNet-B3** 作为视觉编码器。
- **优势**: 训练收敛快，对小数据集友好 (因为有很强的归纳偏置)。
- **劣势**: 难以捕捉长距离关系 (比如：桌子左边的杯子和桌子右边的壶之间的关系，CNN 需要堆叠很多层才能"看"到两者)。

## 3. Transformer (Attention Is All You Need)

### 原理
Transformer 抛弃了卷积，完全依赖 **Self-Attention (自注意力机制)**。
- **全局感受野**: 每一个 Token (像素块) 都能直接"关注"到图像中的其他所有 Token。
- **动态权重**: 卷积核的权重是固定的 (训练好后)，而 Attention Map 是根据输入动态生成的。

### 在 VLA 中的应用
- **RT-2 / OpenVLA / Pi0**: 使用 **ViT (Vision Transformer)** 或 **SigLIP**。
- **优势**:
    - **多模态统一**: 图像 Patch 和文本 Token 可以被同等对待，直接拼接输入 Transformer。
    - **Scaling Law**: 数据越多，模型越大，效果越好 (由弱归纳偏置决定)。
- **劣势**: 训练极其昂贵，需要海量数据 (JFT-300M, LAION-5B) 才能超越 CNN。

## 4. 为什么 VLA 转向 Transformer?

1.  **多模态融合**: 机器人需要同时处理视觉 (Vision) 和语言 (Language)。Transformer 是目前唯一能完美统一这两种模态的架构 (Early Fusion)。
2.  **语义理解**: 机器人不再只是"执行动作"，而是需要"理解环境"。Transformer 在语义提取上远强于 CNN。
3.  **时序建模**: 动作序列 (Action Sequence) 本质上是时间序列。Transformer (GPT 风格) 天生适合处理序列预测问题。

## 5. 深度解析: ViT & SigLIP 技术细节
在 OpenVLA 和 Pi0 等现代模型中，ViT (Vision Transformer) 通常搭配 **SigLIP** 预训练目标使用。

### 5.1 Vision Transformer (ViT) 核心组件
ViT 将图像视为一系列 Patch 的序列，完全摒弃了卷积。

1.  **Patchify & Linear Projection (切片与线性映射)**:
    - 输入图像 $x \in \mathbb{R}^{H \times W \times C}$ 被切分为 $N$ 个 $P \times P$ 的 Patch $x_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$。
    - **公式**:

$$
z_0 = [x_p^1 E; x_p^2 E; \cdots; x_p^N E] + E_{pos}

$$
其中 $E \in \mathbb{R}^{(P^2 \cdot C) \times D}$ 是可学习的线性投影矩阵，$E_{pos} \in \mathbb{R}^{(N+1) \times D}$ 是位置编码。
    - **关键细节**: 这一步等价于一个 `Conv2d(in_channels=3, out_channels=D, kernel_size=P, stride=P)` 操作。

2.  **Positional Embedding Interpolation (位置编码插值)**:
    - **问题**: VLA 任务中，输入图像分辨率可能变化 (e.g., 224x224 -> 384x384)，导致 Patch 数量 $N$ 变化。
    - **解决方案**: 预训练的 1D 位置编码不能直接用。需要将其 reshape 成 2D 网格，进行 **双线性插值 (Bicubic/Bilinear Interpolation)** 到新的尺寸，再展平回 1D。这是 ViT 能处理不同分辨率的关键。

3.  **CLS Token vs Average Pooling**:
    - **CLS Token**: 原始 ViT 在序列开头加一个特殊的 `[CLS]` Token，其输出作为整张图的特征 (BERT 风格)。
    - **Average Pooling (GAP)**: 现代 ViT (如 SigLIP) 往往去掉 CLS Token，直接对所有 Patch 的输出取平均 (Global Average Pooling)。
      - **优势**: 能够利用全图信息，且对 Learning Rate 更鲁棒 (MAP 论文指出 GAP 优于 CLS)。

### 5.2 SigLIP (Sigmoid Loss for Language Image Pre-training)
OpenVLA 的视觉编码器使用的是 **SigLIP** (来自 Google DeepMind)，而非传统的 CLIP。

#### 1. 为什么不用 CLIP (Softmax Loss)?
传统的 CLIP 使用 **InfoNCE Loss** (基于 Softmax)，需要维护巨大的负样本对 (Negative Pairs)。

$$
L_{CLIP} = -\frac{1}{N} \sum_{i=1}^N \log \frac{e^{x_i \cdot y_i / \tau}}{\sum_{j=1}^N e^{x_i \cdot y_j / \tau}}

$$
- **通信瓶颈**: 分母 $\sum e^{...}$ 需要聚合所有 GPU 上的所有样本 (Global Reduction)。在分布式训练中，这会导致巨大的通信开销。

#### 2. SigLIP 的创新 (Sigmoid Loss)
SigLIP 将 $N \times N$ 的匹配问题转化为 **$N^2$ 个独立的二分类问题**。

$$
L_{SigLIP} = - \frac{1}{N} \sum_{i=1}^N \sum_{j=1}^N \left[ \mathbb{I}_{i=j} \log \sigma(x_i \cdot y_j / \tau + b) + \mathbb{I}_{i \neq j} \log (1 - \sigma(x_i \cdot y_j / \tau + b)) \right]

$$
- **$\mathbb{I}_{i=j}$**: 正样本对 (对角线)，标签为 1。
- **$\mathbb{I}_{i \neq j}$**: 负样本对 (非对角线)，标签为 0。
- **优势**:
    1.  **无需全局同步**: 每个 GPU 只需处理自己手头的负样本，梯度计算是局部的。
    2.  **Batch Size 独立**: 性能不再强依赖于超大 Batch Size (CLIP 需要大 Batch 提供足够负样本，SigLIP 对 Batch 大小不敏感)。

#### 3. 关键实现细节: Bias Initialization
SigLIP 引入了一个可学习的 Bias $b$ (通常初始化为 $- \log N$)。
- **原因**: 在训练初期，正样本极少 (1个)，负样本极多 ($N-1$ 个)。如果 Bias 为 0，Sigmoid 输出 0.5，会导致巨大的初始 Loss (因为大部分应该是 0)。
- **Trick**: 初始化 $b = -10$ 或 $- \log N$，强制初始概率接近 0，匹配负样本占主导的先验分布，极大地稳定了训练。

## 6. 自注意力机制详解 (Self-Attention Deep Dive)

### 6.1 计算公式

自注意力机制的核心公式：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V

$$
其中：
- $Q = XW_Q$, $K = XW_K$, $V = XW_V$ (线性投影)
- $d_k$: Key 的维度 (用于缩放，防止点积过大导致 softmax 饱和)
- $X \in \mathbb{R}^{N \times d}$: 输入序列 (N 个 Token，每个 d 维)

```
┌─────────────────────────────────────────────────────────────────┐
│                   Self-Attention 计算流程                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   输入 X [N, d]                                                 │
│       │                                                         │
│       ├──▶ W_Q ──▶ Q [N, d_k]                                   │
│       ├──▶ W_K ──▶ K [N, d_k]                                   │
│       └──▶ W_V ──▶ V [N, d_v]                                   │
│                                                                 │
│   Step 1: QK^T / √d_k  ──▶  Attention Scores [N, N]             │
│   Step 2: softmax(...)  ──▶  Attention Weights [N, N]           │
│   Step 3: Weights × V   ──▶  Output [N, d_v]                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 计算复杂度分析

| 步骤 | 操作 | 时间复杂度 | 空间复杂度 |
| :--- | :--- | :--- | :--- |
| 线性投影 | $XW_Q, XW_K, XW_V$ | $O(Nd^2)$ | $O(Nd)$ |
| $QK^T$ | 矩阵乘法 | $O(N^2 d)$ | $O(N^2)$ ⚠️ |
| Softmax | 按行归一化 | $O(N^2)$ | $O(N^2)$ |
| $\text{Attn} \times V$ | 矩阵乘法 | $O(N^2 d)$ | $O(Nd)$ |
| **总计** | | **$O(N^2 d)$** | **$O(N^2)$** |

**关键洞察**:
- **时间复杂度**: $O(N^2 d)$，对序列长度 $N$ 是平方级
- **空间复杂度**: $O(N^2)$，需要存储完整的 $N \times N$ 注意力矩阵
- **瓶颈**: 当 $N$ 很大时 (如 VLA 中多帧图像 + 语言 Token)，显存成为主要瓶颈
- **解决方案**: Flash Attention (参见 [flash_attention.md](./flash_attention.md))

### 6.3 Multi-Head Attention

将注意力分成多个"头"，每个头关注不同的特征子空间：

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O

$$
$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

$$
**优势**:
- 不同头可以关注不同类型的关系 (位置、语义、纹理等)
- 参数量不变 ($d_k = d / h$)，但表达能力更强

---

## 7. 面试常见问题

**Q: 自注意力机制是什么？计算复杂度怎么算？**
A: 自注意力让序列中每个位置都能直接关注其他所有位置。计算 $QK^T$ 需要 $O(N^2 d)$ 时间和 $O(N^2)$ 空间，其中 $N$ 是序列长度，$d$ 是特征维度。这是 Transformer 处理长序列时的主要瓶颈。

**Q: 为什么要除以 $\sqrt{d_k}$？**
A: 当 $d_k$ 较大时，$QK^T$ 的点积值会很大，导致 softmax 输出接近 one-hot (梯度消失)。除以 $\sqrt{d_k}$ 可以稳定梯度。

**Q: 为什么 ViT 需要比 ResNet 更多的数据?**
A: 因为 ViT 缺乏 **归纳偏置 (Inductive Bias)**。CNN 天生知道"相邻像素相关"和"平移不变"，而 ViT 必须从数据中自己学习这些规律。

**Q: 什么是 Patchify?**
A: 将一张图片切成一个个小方块 (e.g., 16x16 像素)，拉平成向量，作为 Transformer 的输入 Token。这相当于 NLP 中的分词 (Tokenization)。

---
[← Back to Theory](./README.md)
