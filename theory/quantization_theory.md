# 量化理论 (Quantization Theory)

量化 (Quantization) 是将高精度浮点数 (FP32/FP16) 映射到低精度整数 (INT8/INT4) 的过程。它是 VLA 模型边缘部署的核心技术。

## 0. 主要數學思想 (Main Mathematical Idea)

> **第一性原理**: **Resolution vs. Range (分辨率与范围的权衡)**

世界是连续的，计算机是离散的。量化就是用有限的"桶"（Buckets，如 INT8 的 256 个桶）去装无限的实数。

- **核心数学工具**: **Affine Mapping (仿射映射)** 与 **Rounding Error Minimization**。
- **解题逻辑**:
    1.  **映射**: 将连续区间 $[min, max]$ 线性映射到整数区间 $[0, 2^{b}-1]$。公式：$q = \text{round}(x/S + Z)$。
    2.  **权衡**: Scale ($S$) 决定了桶的大小（分辨率）。Scale 越小，分辨率越高，但能覆盖的范围（Range）越小。
    3.  **挑战**: 如果数据中有离群值（Outliers），为了包住它，Scale 必须很大，导致大部分正常数据的分辨率极低（所有桶都空了）。量化的艺术就在于如何处理这些"捣乱"的离群值。

## 1. 基础原理
│                                                                 │
│   FP32 权重                    INT8/INT4 权重                   │
│   ┌─────────┐                  ┌─────────┐                      │
│   │ 0.0234  │                  │   3     │                      │
│   │ 0.1567  │   ──────────▶    │   20    │                      │
│   │-0.0891  │    量化 (Q)      │  -11    │                      │
│   │ 0.2103  │                  │   27    │                      │
│   └─────────┘                  └─────────┘                      │
│       ▲                             │                           │
│       │                             │                           │
│       └─────────────────────────────┘                           │
│              反量化 (DeQ)                                        │
│                                                                 │
│   存储: 32-bit ────────▶ 4-bit (8x 压缩)                        │
│   精度: 高 ────────────▶ 低 (有损失)                            │
└─────────────────────────────────────────────────────────────────┘
```

## 1. 基础原理

### 1.1. 映射公式
将浮点数 $x$ 映射到整数 $q$：

```math
q = \text{round}\left( \frac{x}{S} + Z \right)
```

反量化 (Dequantization)：

```math
\hat{x} = S(q - Z)
```
其中：
- $S$ (Scale): 缩放因子，决定了量化的粒度。
- $Z$ (Zero-point): 零点偏移，用于对齐零点。

### 1.2. 对称 vs 非对称 (Symmetric vs Asymmetric)

```
对称量化 (Symmetric)              非对称量化 (Asymmetric)
        Z = 0                           Z ≠ 0

    -127 ◀──────▶ +127              0 ◀──────────▶ 255
      │     0     │                 │      Z       │
      ├─────┼─────┤                 ├──────┼───────┤
      │     │     │                 │      │       │
   ───┴─────┴─────┴───           ──┴──────┴───────┴──
     min   0    max               min    0       max
      └─────┬─────┘                └───────┬───────┘
            │                              │
       数据分布对称                    数据分布偏移
     (适合权重 weights)           (适合激活值 activations)
```

#### 对称量化 (Symmetric)
- **特点**: $Z=0$。映射范围是对称的 (e.g., INT8: $[-127, 127]$)。
- **公式**: $q = \text{round}(x / S)$。
- **优点**: 计算快 (不需要加减 Zero-point)。
- **缺点**: 如果数据分布严重不对称 (e.g., ReLU 后的激活值全是正的)，会浪费一半的量化范围。

#### 非对称量化 (Asymmetric)
- **特点**: $Z \neq 0$。映射范围可以适配数据分布 (e.g., $[min, max]$ 映射到 $[0, 255]$)。
- **优点**: 精度更高，充分利用 bit 位。
- **缺点**: 计算稍慢。

---

## 2. 粒度 (Granularity)

量化参数 $S$ 和 $Z$ 是怎么算的？取决于粒度。

### 2.1. Per-Tensor (Layer-wise)
- 整个 Tensor 共享一组 $(S, Z)$。
- **优点**: 硬件实现简单。
- **缺点**: 精度最差。如果 Tensor 里有一个极大的离群值 (Outlier)，整个 Tensor 的 Scale 都会被拉大，导致小数值全部变成 0。

### 2.2. Per-Channel (Channel-wise)
- 每一行 (或每一列) 有一组 $(S, Z)$。
- **优点**: 精度显著提升，是 CNN/LLM 权重量化的标准做法。
- **缺点**: 存储 Scale 需要额外空间 (但在大模型中可忽略)。

### 2.3. Per-Token / Per-Group
- **Per-Token**: 针对激活值 (Activation)，每个 Token 动态计算 Scale。
- **Per-Group**: 将权重每 128 个数分为一组 (Group)，每组一个 Scale。这是 **4-bit 量化 (如 AWQ, GPTQ)** 的标配，因为 4-bit 精度太低，必须用细粒度 Scale 来补。

---

## 3. 进阶难题：离群值 (Outliers)

LLM / VLA 模型有一个特性：**激活值中存在极端的离群值** (Outliers)。这些值虽然很少，但对模型性能至关重要。

### 3.1. 为什么直接量化会失败？
如果直接做 INT8 量化，为了包住那个巨大的 Outlier，Scale 会变得很大，导致其他 99.9% 的正常数值都被压缩到了 0，模型直接傻掉。

### 3.2. 解决方案：SmoothQuant / AWQ
- **SmoothQuant**: 数学等价变换。
  $$ Y = X W = (X \cdot s^{-1}) \cdot (s \cdot W) $$
  把激活值 $X$ 里的 Outlier "平摊" (Smooth) 到权重 $W$ 上。让 $X$ 变小，让 $W$ 变大。因为权重通常比较均匀，更容易量化。
- **AWQ (Activation-aware Weight Quantization)**:
  不量化那些对应重要激活值 (Salient Weights) 的权重，或者保留更高的精度。

---

## 4. 面试高频考点

**Q: 为什么 4-bit 量化通常比 8-bit 量化更难？**
A: 4-bit 只有 16 个格子。如果 Scale 稍微没选好，误差就会巨大。所以 4-bit 通常需要 **Per-Group** 量化 (Group Size=128) 和更复杂的校准算法 (如 GPTQ)。

**Q: Post-Training Quantization (PTQ) vs Quantization-Aware Training (QAT)**
- **PTQ**: 拿训练好的模型直接量化 (加少量校准数据)。简单，主流。
- **QAT**: 训练时就模拟量化误差 (Fake Quantization)。精度最高，但训练成本高，VLA 领域较少用。

**Q: 权重量化 (Weight-only) vs 激活量化 (Activation Quantization)**
- **W4A16**: 权重 4-bit，激活 FP16。省显存，推理速度受限于反量化带宽。
- **W8A8**: 权重 8-bit，激活 8-bit。可以使用 INT8 Tensor Core 加速计算，真正的推理加速。
