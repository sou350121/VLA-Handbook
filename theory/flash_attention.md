# Flash Attention: 高效 Transformer 推理的关键

Flash Attention 是 Transformer 模型（包括 VLA）在部署时的核心优化技术，解决了标准 Attention 的内存瓶颈问题。

```
┌─────────────────────────────────────────────────────────────────┐
│              Flash Attention vs 标准 Attention                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   标准 Attention (内存瓶颈)           Flash Attention (分块)    │
│                                                                 │
│   ┌───────────────────┐               ┌───────────────────┐     │
│   │      Q · K^T      │               │   Q₁·K₁  Q₁·K₂   │     │
│   │   ┌─────────────┐ │               │  ┌─────┐┌─────┐  │     │
│   │   │             │ │   ────▶       │  │ 块1 ││ 块2 │  │     │
│   │   │   N × N     │ │   分块        │  └─────┘└─────┘  │     │
│   │   │  (巨大!)    │ │               │  ┌─────┐┌─────┐  │     │
│   │   │             │ │               │  │ 块3 ││ 块4 │  │     │
│   │   └─────────────┘ │               │  └─────┘└─────┘  │     │
│   └───────────────────┘               └───────────────────┘     │
│   内存: O(N²) ❌                       内存: O(N) ✅             │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                        内存层级优化                              │
│                                                                 │
│   ┌────────────────┐     ┌────────────────┐                     │
│   │      HBM       │     │     SRAM       │                     │
│   │   (显存)       │     │   (L2 Cache)   │                     │
│   │   24GB+        │ ◀─▶ │    ~20MB       │                     │
│   │   慢 1TB/s     │     │   快 19TB/s    │                     │
│   └────────────────┘     └────────────────┘                     │
│          │                      │                               │
│          │   标准: 多次读写      │   Flash: 一次加载             │
│          │   Q,K ──▶ S ──▶ P    │   全部在 SRAM 计算            │
│          │   每步写回 HBM       │   只写最终结果                 │
│          │                      │                               │
│   结果: 2-4x 加速, 显存 O(N²) → O(N)                            │
└─────────────────────────────────────────────────────────────────┘
```

## 1. 为什么需要 Flash Attention?

### 标准 Attention 的问题
标准的 Scaled Dot-Product标准 Attention 计算公式：

```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

- **内存瓶颈**: 计算 $QK^T$ 需要存储 $N \times N$ 的注意力矩阵，其中 $N$ 是序列长度。
- **实例**: 对于 ViT (序列长度 196), $196 \times 196 = 38,416$ 个浮点数。VLA 中若包含多帧图像，$N$ 可能达到数千。
- **显存占用**: $O(N^2)$ 的内存占用使得长序列推理几乎不可行。

## 2. Flash Attention 的核心思想

### Tiling (分块计算)
Flash Attention 通过 **分块 (Tiling)** 避免实际存储完整的 $QK^T$ 矩阵。

**算法流程**:
1. 将 $Q, K, V$ 分割成小块 (Tiles)。
2. 逐块计算 Attention，在 SRAM (L2 Cache) 中完成。
3. 使用 **在线 Softmax (Online Softmax)** 技术增量更新归一化。
4. 最终合并结果，避免将中间结果写回 HBM (High Bandwidth Memory)。

### 2.1 Kernel Fusion (算子融合): IO-Aware Computing
Flash Attention 的核心洞察是：**Transformer 的瓶颈不在计算 (FLOPs)，而在显存读写 (HBM IO)**。

-   **HBM vs SRAM**:
    -   **HBM (High Bandwidth Memory)**: 显存，容量大 (24GB+)，但速度慢 (1-2 TB/s)。
    -   **SRAM (L2 Cache)**: 片上缓存，容量小 (几十 MB)，但速度极快 (19 TB/s+)。
-   **标准 Attention**: 需要多次读写 HBM。
    1.  读 $Q, K$ -> 算 $S = QK^T$ -> 写回 HBM ($N^2$)。
    2.  读 $S$ -> 算 $P = \text{softmax}(S)$ -> 写回 HBM ($N^2$)。
    3.  读 $P, V$ -> 算 $O = PV$ -> 写回 HBM。
-   **Flash Attention Fusion**:
    -   将上述所有步骤融合进**同一个 CUDA Kernel**。
    -   数据一旦从 HBM 加载到 SRAM，就在 SRAM 中完成 $QK^T$, Softmax, $PV$ 的所有计算，只把最终结果 $O$ 写回 HBM。
    -   **结果**: HBM 读写量从 $O(N^2)$ 降低到 $O(N)$，尽管 FLOPs 没变，但端到端速度提升了 2-4 倍。

### 2.2 Recomputation (重计算): 换取显存的艺术
在训练时的反向传播 (Backward Pass) 中，通常需要保存前向传播的中间激活值 (Activations) 来计算梯度。

-   **标准做法**: 保存巨大的 $N \times N$ 注意力矩阵 $P$。这直接导致了 OOM (Out of Memory)。
-   **Flash Attention 做法**:
    -   **不保存** $P$ 矩阵。
    -   在反向传播时，利用保存在 SRAM 中的 $Q, K, V$ 块，**重新计算**一遍 Attention。
-   **为什么更快?**
    -   直觉上，重计算会增加 FLOPs，应该变慢。
    -   但由于 Attention 是 **IO-Bound** (受限于带宽) 的，重计算带来的额外 FLOPs 开销，远小于从 HBM 读取巨大矩阵 $P$ 的时间开销。
    -   **结论**: Recomputation 不仅省了显存，反而因为减少了 IO 而变快了。

### 数学推导：在线 Softmax

标准 Softmax 需要两次扫描序列（一次求和，一次归一化）。Flash Attention 使用增量更新：

```
softmax(x)_i = exp(x_i) / Σ exp(x_j)  (j=1 到 N)
```

通过维护运行中的 **最大值 m** 和 **累积和 l**，可以逐块更新：

```
m_new = max(m_old, m_block)

l_new = exp(m_old - m_new) * l_old + exp(m_block - m_new) * l_block
```

## 3. 在 VLA 中的应用

### Wall-X / OpenVLA
- **Wall-X**: requirements.txt 中明确依赖 `flash-attn==2.7.4`。
- **OpenVLA**: 支持 Flash Attention 2 加速推理，尤其在处理长历史序列时。

### Pi0
- Pi0 使用 Flow Matching，推理时需要多步 ODE Solver。Flash Attention 在每一步都能显著减少显存占用。

### 性能提升
- **速度**: 2-4x 加速（相比标准 Attention）。
- **显存**: 内存占用从 $O(N^2)$ 降至 $O(N)$。
- **部署**: 使得在消费级 GPU (e.g., RTX 4090) 上部署 7B VLA 成为可能。

## 4. Flash Attention vs 其他优化

| 技术 | 内存复杂度 | 精度 | 适用场景 |
| :--- | :--- | :--- | :--- |
| **Standard Attention** | $O(N^2)$ | 精确 | 短序列 |
| **Flash Attention** | $O(N)$ | 精确 | 长序列，真机推理 |
| **Sparse Attention** | $O(N \log N)$ | 近似 | 超长文本 (不适合 VLA) |
| **Linear Attention** | $O(N)$ | 近似 | 研究阶段 |

## 5. 面试常见问题

**Q: Flash Attention 是如何做到精确计算的？**
A: 通过在线 Softmax 和分块计算，Flash Attention 在数学上等价于标准 Attention，只是改变了计算顺序和内存访问模式，没有引入任何近似。

**Q: 为什么不能直接用 Sparse Attention?**
A: VLA 的注意力模式通常是密集的（视觉 Patch 之间关系紧密），稀疏假设不成立。Flash Attention 保持了密集计算，只是优化了内存。

**Q: Flash Attention 对训练和推理都有效吗？**
A: 是的。训练时通过 Recomputation 节省显存，推理时通过 Kernel Fusion 加速。Wall-X 等模型在两者都使用。


---
[← Back to Theory](./README.md)
