# 高效微调理论 (PEFT & LoRA)

在 VLA 时代，我们通常基于 7B+ 的大模型进行微调。全量微调 (Full Fine-tuning) 极其昂贵，因此参数高效微调 (PEFT) 成为了必修课。

## 1. LoRA (Low-Rank Adaptation)

### 1.1. 核心思想
大模型的权重矩阵 $W \in \mathbb{R}^{d \times k}$ 虽然参数很多，但在特定任务 (如机器人控制) 上，其**内在维度 (Intrinsic Dimension)** 其实很低。
我们不需要更新整个 $W$，只需要学习一个低秩的增量矩阵 $\Delta W$。

### 1.2. 数学原理
假设预训练权重为 $W_0$，微调后的权重为 $W_0 + \Delta W$。
我们将 $\Delta W$ 分解为两个低秩矩阵 $A$ 和 $B$ 的乘积：

$$
W = W_0 + \Delta W = W_0 + B A
$$
其中：
- $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$
- $r \ll \min(d, k)$ 是秩 (Rank)，通常取 8, 16, 32。
- **参数量对比**: $d \times k$ (全量) vs $r \times (d + k)$ (LoRA)。对于 7B 模型，LoRA 参数量通常不到 1%。

### 1.3. 训练与推理
- **初始化**: $A$ 使用高斯初始化，$B$ 初始化为 0。这样初始状态下 $\Delta W = 0$，模型输出与预训练模型一致。
- **训练**: 冻结 $W_0$，只更新 $A$ 和 $B$。
- **推理**: 可以将 $BA$ 加回到 $W_0$ 中 (Merge)，推理速度与原模型完全一致，无额外延迟。

  $$
  W_{merged} = W_0 + \alpha \cdot BA
  $$

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

---

## 4. 面试高频考点

**Q: LoRA 的秩 $r$ 越大越好吗？**
A: 不一定。对于简单的任务 (如指令跟随)，$r=8$ 足矣。对于复杂的逻辑推理或知识注入，$r$ 可能需要设大一点 (64 或 128)。但过大容易过拟合。

**Q: 为什么 QLoRA 比 LoRA 慢？**
A: 因为 QLoRA 在计算梯度时，需要将 4-bit 权重**反量化 (Dequantize)** 回 FP16/BF16 进行计算。这个反量化过程增加了计算开销。

**Q: VLA 模型微调应该微调哪些层？**
A: 通常微调 **Attention (q_proj, v_proj)** 和 **MLP (gate_proj, up_proj, down_proj)** 效果最好。只微调 Attention 有时不够。
