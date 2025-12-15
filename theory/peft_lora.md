# 高效微调理论 (PEFT & LoRA)

在 VLA 时代，我们通常基于 7B+ 的大模型进行微调。全量微调 (Full Fine-tuning) 极其昂贵，因此参数高效微调 (PEFT) 成为了必修课。

## 0. 主要數學思想 (Main Mathematical Idea)

> **第一性原理**: **Redundancy of Information (信息的冗余 / Intrinsic Dimension)**

一个拥有 70 亿参数的通用模型，在学习一个特定技能（如"拿起杯子"）时，并不需要改变所有 70 亿个自由度。任务的本质变化通常发生在极低维的子空间中。

- **核心数学工具**: **Low-Rank Matrix Decomposition (低秩矩阵分解 / SVD)**。
- **解题逻辑**:
    1.  **假设**: 权重矩阵的变化量 $\Delta W$ 是低秩的 (Low Rank)。即 $\text{rank}(\Delta W) \ll \min(d, k)$。
    2.  **分解**: 任何低秩矩阵都可以分解为两个小矩阵的乘积 ($B \times A$)。
    3.  **高效**: 我们不直接训练巨大的 $\Delta W$ ($d \times k$)，而是训练微小的 $A$ 和 $B$ ($r \times (d+k)$)。这就像用几个主成分（Principal Components）来近似复杂的变换。

## 1. LoRA (Low-Rank Adaptation)

```
┌─────────────────────────────────────────────────────────────────┐
│                    LoRA 架构示意图                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│         输入 x                                                  │
│           │                                                     │
│     ┌─────┴─────┐                                               │
│     │           │                                               │
│     ▼           ▼                                               │
│  ┌──────┐   ┌──────┐                                            │
│  │  W₀  │   │  A   │  ← 低秩矩阵 (r << d)                       │
│  │      │   │ r×k  │    可训练                                  │
│  │ d×k  │   └──┬───┘                                            │
│  │      │      │                                                │
│  │ 冻结 │      ▼                                                │
│  └──┬───┘   ┌──────┐                                            │
│     │       │  B   │  ← 低秩矩阵                                │
│     │       │ d×r  │    可训练                                  │
│     │       └──┬───┘                                            │
│     │          │                                                │
│     │    ΔW = B·A                                               │
│     │          │                                                │
│     └────┬─────┘                                                │
│          │  W = W₀ + α·BA                                       │
│          ▼                                                      │
│        输出 h                                                   │
│                                                                 │
│  参数量: d×k (冻结) + r×(d+k) (训练) ≈ 0.1% ~ 1%                │
└─────────────────────────────────────────────────────────────────┘
```

### 1.1. 核心思想
大模型的权重矩阵 $W \in \mathbb{R}^{d \times k}$ 虽然参数很多，但在特定任务 (如机器人控制) 上，其**内在维度 (Intrinsic Dimension)** 其实很低。
我们不需要更新整个 $W$，只需要学习一个低秩的增量矩阵 $\Delta W$。

### 1.2. 数学原理
假设预训练权重为 $W_0$，微调后的权重为 $W_0 + \Delta W$。
我们将 $\Delta W$ 分解为两个低秩矩阵 $A$ 和 $B$ 的乘积：

```math
W = W_0 + \Delta W = W_0 + B A
```
其中：
- $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$
- $r \ll \min(d, k)$ 是秩 (Rank)，通常取 8, 16, 32。
- **参数量对比**: $d \times k$ (全量) vs $r \times (d + k)$ (LoRA)。对于 7B 模型，LoRA 参数量通常不到 1%。

### 1.3. 训练与推理
- **初始化**: $A$ 使用高斯初始化，$B$ 初始化为 0。这样初始状态下 $\Delta W = 0$，模型输出与预训练模型一致。
- **训练**: 冻结 $W_0$，只更新 $A$ 和 $B$。
- **推理**: 可以将 $BA$ 加回到 $W_0$ 中 (Merge)，推理速度与原模型完全一致，无额外延迟。

```math
  W_{merged} = W_0 + \alpha \cdot BA
```

  ($\alpha$ 是缩放系数，通常 $\alpha/r$ 用于归一化)。

---

## 2. QLoRA (Quantized LoRA)

### 2.1. 痛点
LoRA 虽然减少了可训练参数，但**基础模型 $W_0$ 依然需要以 FP16 加载到显存中**。对于 65B 的模型，光加载就需要 130GB 显存，单卡 4090 根本跑不动。

### 2.2. 核心创新
QLoRA 结合了 **4-bit 量化** 和 **LoRA**，使得 65B 模型可以在 48GB 显存上微调。

1.  **4-bit NormalFloat (NF4)**: 一种理论最优的 4-bit 量化数据类型，专门针对正态分布的权重设计。
2.  **Double Quantization**: 对量化常数 (Quantization Constants) 再进行一次量化，进一步节省显存 (每参数平均节省 0.37 bit)。
3.  **Paged Optimizers**: 利用 CPU 内存来缓存优化器状态 (Optimizer States)，防止显存 OOM。

### 2.3. 显存计算 (7B Model)
- **Full Fine-tuning (Adam)**: ~112 GB (权重+梯度+优化器状态)
- **LoRA (FP16)**: ~16 GB (权重) + ~1 GB (LoRA) = ~17 GB
- **QLoRA (4-bit)**: ~4 GB (权重) + ~1 GB (LoRA) = ~5 GB (可以在 RTX 3060 上跑！)

---

## 3. 其他 PEFT 方法

### 3.1. Adapters (Houlsby et al.)
- 在 Transformer 的每一层中插入小的全连接网络 (Adapter Layers)。
- **缺点**: 增加了推理延迟 (因为是串行的层)，无法像 LoRA 那样合并权重。

### 3.2. Prefix Tuning / P-Tuning
- 在 Input Token 前面拼接一组可学习的 Virtual Tokens。
- **缺点**: 占用了宝贵的 Context Window 长度。

### 3.3. P-Tuning v2
- 在每一层都加入可学习的 Prefix，而不仅仅是输入层。
- **优点**: 比 P-Tuning v1 效果更好，接近全量微调。
- **缺点**: 仍然占用 Context Window。

---

## 4. PEFT 方法对比 (Comparison)

```
┌─────────────────────────────────────────────────────────────────┐
│                    PEFT 方法对比一览                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   方法          原理              推理延迟   可合并   参数量     │
│   ─────────────────────────────────────────────────────────     │
│   LoRA          低秩分解 ΔW=BA    无额外     ✅       ~0.1-1%   │
│   Adapter       串行插入 MLP      有额外     ❌       ~1-5%     │
│   P-Tuning      可学习 Prompt     占 Context ❌       ~0.01%    │
│   Prefix-Tuning 可学习 KV 前缀    占 Context ❌       ~0.1%     │
│   P-Tuning v2   每层 Prefix       占 Context ❌       ~0.1-1%   │
│                                                                 │
│   💡 VLA 首选: LoRA (可合并，无推理延迟)                         │
└─────────────────────────────────────────────────────────────────┘
```

| 方法 | 原理 | 推理延迟 | 可合并权重 | 参数量 | 效果 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LoRA** | 低秩分解 $\Delta W = BA$ | 无 | ✅ | ~0.1-1% | ⭐⭐⭐⭐ |
| **Adapter** | 串行插入 MLP | 有 | ❌ | ~1-5% | ⭐⭐⭐ |
| **P-Tuning** | 可学习 Soft Prompt | 占 Context | ❌ | ~0.01% | ⭐⭐ |
| **Prefix-Tuning** | 可学习 KV 前缀 | 占 Context | ❌ | ~0.1% | ⭐⭐⭐ |
| **P-Tuning v2** | 每层 Prefix | 占 Context | ❌ | ~0.1-1% | ⭐⭐⭐⭐ |

### LoRA vs P-Tuning vs Adapter 核心差异

| 对比维度 | LoRA | P-Tuning | Adapter |
| :--- | :--- | :--- | :--- |
| **修改位置** | 权重矩阵 (W) | 输入层 / 每层 | 层间插入 |
| **推理时** | 可合并，无延迟 | 需保留 Prompt | 需保留 Adapter |
| **Context 占用** | 无 | 有 (几十~几百 Token) | 无 |
| **适用场景** | 通用首选 | 小模型 / 特定任务 | 多任务切换 |

---

## 5. LoRA 参数选择指南 (Hyperparameter Guide)

### 5.1 Rank (秩 $r$) 的影响

| $r$ 值 | 参数量 | 适用场景 | 风险 |
| :--- | :--- | :--- | :--- |
| **4-8** | 最少 | 简单任务 (指令跟随、风格迁移) | 欠拟合 |
| **16-32** | 适中 | 通用微调 (VLA 推荐) | 平衡 |
| **64-128** | 较多 | 复杂推理、知识注入 | 过拟合 |
| **256+** | 接近全量 | 几乎等价于全量微调 | 失去 LoRA 优势 |

### 5.2 Alpha ($\alpha$) 的影响

```math
W = W_0 + \frac{\alpha}{r} \cdot BA
```

- **$\alpha / r$ 比值**: 控制 LoRA 更新的"强度"
- **常见设置**: $\alpha = 2r$ (即 $\alpha/r = 2$)
- **$\alpha$ 大**: 更激进的更新，收敛快但可能不稳定
- **$\alpha$ 小**: 更保守的更新，稳定但收敛慢

### 5.3 目标层选择

| 配置 | 目标层 | 参数量 | 效果 |
| :--- | :--- | :--- | :--- |
| **最小配置** | `q_proj, v_proj` | ~0.1% | 基础效果 |
| **推荐配置** | `q_proj, k_proj, v_proj, o_proj` | ~0.2% | 较好效果 |
| **完整配置** | Attention + MLP 全部 | ~1% | 最佳效果 |

```python
# PEFT 配置示例
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,                    # 秩
    lora_alpha=32,           # alpha = 2r
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"       # MLP
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

---

## 6. 面试高频考点

**Q: LoRA 的原理是什么？与 P-Tuning、Adapter 的异同点？**
A: 
- **LoRA**: 将权重增量 $\Delta W$ 分解为低秩矩阵 $BA$，冻结原始权重只训练 $A, B$。推理时可合并回原权重，无额外延迟。
- **P-Tuning**: 在输入前加可学习的 Soft Prompt，占用 Context Window。
- **Adapter**: 在 Transformer 层间插入小型 MLP，推理时有额外延迟。
- **核心区别**: LoRA 修改权重本身，P-Tuning 修改输入，Adapter 修改结构。LoRA 是唯一可以"无痕合并"的方法。

**Q: LoRA 的参数选择对模型性能有何影响？**
A:
- **Rank $r$**: 控制表达能力。$r$ 小 (4-8) 适合简单任务，$r$ 大 (64+) 适合复杂任务但易过拟合。
- **Alpha $\alpha$**: 控制更新强度。通常 $\alpha = 2r$。
- **目标层**: 只训练 `q_proj, v_proj` 是最小配置，加上 MLP 效果更好。

**Q: LoRA 的秩 $r$ 越大越好吗？**
A: 不一定。对于简单的任务 (如指令跟随)，$r=8$ 足矣。对于复杂的逻辑推理或知识注入，$r$ 可能需要设大一点 (64 或 128)。但过大容易过拟合，且失去参数高效的优势。

**Q: 为什么 QLoRA 比 LoRA 慢？**
A: 因为 QLoRA 在计算梯度时，需要将 4-bit 权重**反量化 (Dequantize)** 回 FP16/BF16 进行计算。这个反量化过程增加了计算开销。

**Q: VLA 模型微调应该微调哪些层？**
A: 通常微调 **Attention (q_proj, v_proj)** 和 **MLP (gate_proj, up_proj, down_proj)** 效果最好。只微调 Attention 有时不够。
