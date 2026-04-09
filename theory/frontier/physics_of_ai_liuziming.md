# Physics of AI：把神经网络当物理系统来研究

> **核心主张**：不赌规模，而是像物理学家研究自然一样研究 AI——定义现象、设计观测量、归纳规律。这条路可能是 Scaling Law 之外通往 AGI 的"科学路径"。
>
> 本文整理自 MIT 刘子鸣（Ziming Liu）的研究纲领和代表性成果，并把每个概念翻译成 VLA 研究者能直接用的工程抓手。

---

## 为什么 VLA 研究者应该关心这个？

VLA 是所有 AI 系统中最像"真实物理系统"的——它涉及连续力学、接触动力学、3D 几何、多模态感知，输出直接作用于物理世界。而 Physics of AI 提供的正是理解和诊断这类复杂系统的方法论。

**如果你遇到过以下问题，这篇文章会有帮助**：
- 训练到一半突然"开窍"或突然崩溃，不知道为什么
- 模型在仿真里 95% 成功率，到真机只有 30%，不知道哪一层出了问题
- VLA 的 Action Head 选 Token/Diffusion/Flow 没有理论依据，只能试
- 想知道模型到底"学到了什么"，而不只是看最终成功率

---

## 1. 三条 AGI 路径

刘子鸣给出了一个清晰的分类：

| 路径 | 策略 | 优势 | 风险 |
|------|------|------|------|
| **Scaling** | 更多数据 + 更大模型 + 更多算力 | 短期见效快，工程路径清晰 | 能量/数据/算力瓶颈，可能撞墙 |
| **Agent** | 接受黑箱，在外部叠加记忆/工具/规划 | 工程可用性高，立即有产品 | 系统越复杂越脆弱，缺乏理论保证 |
| **Physics of AI** | 像研究自然一样研究 AI 的内部规律 | 用更少资源达到更好效果，可解释 | 进展慢，需要长期投入 |

> 💡 **VLA 的实际处境**：目前主流走的是 Scaling（π0 系列）+ Agent（双系统架构）。Physics of AI 是第三条路——不取代前两者，但为前两者提供**理论地基和诊断工具**。

---

## 2. 核心方法论：像物理学家一样

Physics of AI 不是"用物理公式替代神经网络"，而是借用物理学的**研究方法**：

```
观察现象 → 定义可测量的观测量 → 控变量实验 → 归纳规律 → 用规律指导设计
```

### 2.1 现象 (Phenomena)

训练神经网络时会出现可复现的"涌现行为"，类似物理学中的相变：

| 现象 | 描述 | 类比物理 |
|------|------|---------|
| **Grokking（顿悟）** | 训练 loss 早已收敛，但测试 accuracy 突然跳升 | 一阶相变（过冷液体突然结晶） |
| **Double Descent** | 模型越大先变差再变好，呈 U-W 形曲线 | 临界点附近的非单调行为 |
| **Neural Scaling Laws** | 性能随参数/数据/算力的幂律缩放 | 临界指数（renormalization group） |
| **Representation Geometry** | 训练过程中嵌入从随机→有结构（环、簇、流形） | 对称性自发破缺 |
| **Loss Landscape Topology** | 局部最优之间有"隧道"相连 | 能量面上的鞍点和通道 |

### 2.2 观测量 (Observables)

**不只盯最终 loss/accuracy**，要定义能刻画中间过程的量：

| 观测量 | 测什么 | 怎么算 | 对 VLA 有什么用 |
|--------|--------|--------|----------------|
| 表征秩 (Effective Rank) | 特征空间用了多少维度 | SVD of activation matrix → 奇异值分布的熵 | 判断 VLA 是否在"压缩"还是"记忆" |
| 层间对齐 (CKA) | 不同层的表征有多相似 | Centered Kernel Alignment | 诊断 VLM→Action Head 的信息传递是否断裂 |
| 梯度信噪比 | 梯度的信号 vs 噪声 | mean(grad) / std(grad) | 判断训练是否有效（SNR 太低 = 白学了） |
| 信息平面 | 中间层保留了多少输入信息 vs 输出信息 | Mutual Information I(X;T) vs I(T;Y) | 可视化 VLA 的"信息瓶颈"在哪一层 |
| 动作分布的模态数 | 输出有几个"峰" | GMM 拟合或 kernel density | 判断 Action Head 是否能处理多模态（→ 选 Token vs Diffusion vs Flow） |

### 2.3 控变量实验 (Toy Models)

> *"把大象拆成可观测的侧面。"*

不要一上来就在完整 VLA 上做实验。先用 toy 版本：
- **Toy 数据**：2D 抓取、单步 pick-and-place
- **Toy 架构**：2 层 Transformer + 小 Action Head
- **Toy 任务**：只改一个因素（语言监督强度 / 触觉通道 / 动作分词长度 / 正则化）
- **追踪**：训练全过程的表征结构变化，不只是最终指标

---

## 3. 代表性成果

### 3.1 KAN（Kolmogorov-Arnold Networks）

**动机**：MLP 的万能逼近定理是存在性定理——保证"能拟合"，但不保证"怎么拟合最高效"。KAN 用另一个数学定理（Kolmogorov-Arnold 表示定理）重构网络基元。

**核心区别**：

| | MLP | KAN |
|--|-----|-----|
| 激活函数位置 | 在节点上（固定的 ReLU/GELU） | 在边上（可学习的样条函数） |
| 拟合方式 | 叠加简单非线性 | 组合可学习的一维函数 |
| 可解释性 | 低（黑箱） | 高（每条边都是可视化的函数） |
| 擅长场景 | 大规模通用任务 | 符号公式、科学计算、高精度需求 |

**对 VLA 的启示**：
- VLA 的 Action Head 是否可以用 KAN 替代 MLP？在需要精细力控的任务中，KAN 的高精度可能有优势
- KAN 的可解释性可以帮助诊断"动作生成到底学到了什么函数"

---

### 3.2 Grokking（顿悟现象）

**现象**：模型在训练集上已经 100% 准确（完全记忆），但过了很久之后测试集准确率突然从随机跳到接近完美。

**两种互补解释**：

1. **表征几何演化**：
   - 嵌入从随机初始化逐步演化出有意义的几何形态（如模运算中出现环状结构——"The Clock and the Pizza"）
   - 泛化跃迁发生在几何结构"结晶"的瞬间

2. **表达能力压缩**：
   - 正则化（weight decay）迫使模型从"记忆训练集"转向"压缩出更简单的算法"
   - 一旦找到该算法，性能突变

**对 VLA 的启示**：
- VLA 训练中是否也有"顿悟"？比如某个任务长期不会，突然在某个 epoch 学会？
- 如果有，可以用表征几何追踪来**预测**顿悟何时发生，而不是盲等
- Weight decay 的强度可能决定了 VLA 是在"记忆演示"还是在"学习策略"

---

### 3.3 物理启发的生成模型

刘子鸣的一个探索方向：**不局限于 Diffusion/Flow，用其他物理过程做生成**。

| 物理过程 | 生成模型 | 核心思想 |
|---------|---------|---------|
| 热扩散 | DDPM / Score-based | 加噪 = 热传导，去噪 = 时间反演 |
| 电场 | **PFGM（Poisson Flow）** | 数据点是"电荷"，生成 = 沿电场线流动 |
| 波动 | WaveFormer（实验性） | 波动方程驱动的特征传播 |
| 最优传输 | Flow Matching / Rectified Flow | 噪声→数据的最短路径（ODE） |

**关键洞察**：Diffusion/Flow Matching 已经 IS 物理启发方法。它们成功的原因正是因为利用了正确的物理直觉（连续动力学、概率流 ODE）。下一个突破可能来自找到更适合**动作空间**的物理过程——动作不是图像，它有时间结构、物理约束（关节限位、力矩极限）、因果性。

→ 详见 [Diffusion Policy](../diffusion-flow/diffusion_policy.md) · [π0 Flow Matching](../vla-core/pi0_code_analysis.md) · [WaveFormer](../perception/waveformer_wave_equation_vision_2026.md)

---

## 4. 统计力学视角：理解 VLA 的训练动力学

> 这一节是专访之外的延伸，帮助理解 Physics of AI 更广泛的理论框架。

### 4.1 Loss Landscape = 能量面

把 loss function 想象成一个高维的"地形"：

- **局部最小值** = 盆地（训练会陷进去）
- **鞍点** = 山口（SGD 可以翻过去）
- **平坦最小值** = 宽阔的盆地（泛化好）
- **尖锐最小值** = 窄窝（泛化差）

**对 VLA 的启示**：
- 大 batch size 倾向于找到尖锐最小值（泛化差）
- 小 batch size / 大学习率产生更多"热噪声"，帮助跳出窄盆地
- **EMA（指数移动平均）** 平滑权重轨迹，等于在"能量面上做低通滤波"——这就是为什么几乎所有 VLA 训练都用 EMA

### 4.2 相变与临界点

神经网络训练中的突变行为（grokking、double descent、sudden capability emergence）可以用统计力学的**相变理论**理解：

- **温度** ≈ 学习率 × batch noise
- **秩序参数** ≈ 表征结构的某种度量（如有效秩）
- **相变** = 秩序参数的突变，对应训练行为的质变

**VLA 场景的"相变"**：
- 从"随机动作"突然到"有结构的动作轨迹"
- 从"单任务记忆"到"跨任务泛化"的突变
- 从"看到但不动"到"看到就动"的感知-动作耦合

### 4.3 重整化群（Renormalization Group）与特征层级

物理学中，重整化群描述的是"不同尺度上的等效描述"。神经网络的深度类似于重整化的尺度：

- **浅层**：局部特征（边缘、颜色）
- **中层**：物体部件（手柄、杯口）
- **深层**：语义概念（"一个可以抓的杯子"）

**对 VLA 的启示**：
- VLA 的 Vision Encoder 天然是一个重整化过程（从像素到语义）
- 双系统架构（GR00T、Helix）本质上是把不同"尺度"（语义/运动）分配给不同的子系统——这和重整化群的精神一致
- 如果某一层的"重整化"做得不好（信息丢失太多），下游的动作生成就会崩

---

## 5. Physics of AI × VLA：诊断与设计

### 5.1 VLA 失败诊断框架

把 VLA 的失败拆成可诊断的**中间节点**（类似物理学的观测量）：

```
输入 → [感知] → [语言理解] → [融合] → [规划] → [动作生成] → [执行] → 输出
         ①          ②          ③        ④          ⑤          ⑥
```

每个节点都可以定义观测量：

| 节点 | 观测量 | 诊断方法 |
|------|--------|---------|
| ① 感知 | 特征对齐度、物体检测 mAP | Grad-CAM / 注意力可视化 |
| ② 语言 | 指令→embedding 的区分度 | 不同指令的 embedding 余弦距离 |
| ③ 融合 | 跨模态注意力权重分布 | Cross-attention map 可视化 |
| ④ 规划 | 子任务分解的一致性 | CoT 输出的语义评估 |
| ⑤ 动作 | 生成分布的模态数/方差 | 多次采样的轨迹聚类 |
| ⑥ 执行 | 关节跟踪误差 | PD 控制器的 tracking error |

> 这个框架和 ERIQ（AgiBot 的推理→动作传递损耗度量）异曲同工。
> → 详见 [GenieReasoner/ERIQ](../planning/geniereasoner_eriq_fact.md)

### 5.2 设计原则

从 Physics of AI 视角推导出的 VLA 设计建议：

| 物理原则 | VLA 设计建议 | 实例 |
|---------|-------------|------|
| **最小作用量** | 动作轨迹应该是"最短路径"（最小能量） | Flow Matching 的 OT 路径 |
| **对称性** | 如果任务有对称性，模型应该尊重它 | EquiBim 的等变策略 |
| **信息守恒** | 每一层不应丢失关键信息 | ReconVLA 的重建监督 |
| **尺度分离** | 不同时间尺度的控制应该解耦 | 双系统架构（GR00T, Helix） |
| **热力学稳定性** | 训练应该趋向宽阔最小值 | EMA + 适当的学习率 warmup |

---

## 6. 方法论工具箱：Physics of AI 风格的实验

### 6.1 Toy Model 实验模板

```python
# 1. 选一个 toy VLA 任务（如 2D 抓取）
# 2. 训练时追踪这些观测量：
observables = {
    "effective_rank": compute_svd_entropy(activations),   # 表征秩
    "weight_norm":    model.parameters().norm(),           # 权重范数
    "gradient_snr":   grad.mean() / grad.std(),           # 梯度信噪比
    "action_modes":   fit_gmm(action_samples).n_components, # 动作模态数
    "cka_alignment":  compute_cka(layer_i, layer_j),      # 层间对齐
}
# 3. 只改一个因素（如 Action Head 类型），对比观测量变化
# 4. 归纳：哪些观测量的变化与成功率相关？
```

### 6.2 日常研究习惯

刘子鸣的建议（也适用于 VLA 研究者）：

- **每天记录"小洞察"**：训练 log 里的异常现象、不符合直觉的结果
- **用博客代替只写论文**：很多"细碎但重要"的观察不够一篇论文，但对社区有价值
- **建立"知识网络"**：把发现的现象之间的关系画成图，而不是堆在笔记本里

---

## 7. 延伸阅读

### 仓库内相关文章

| 方向 | 推荐 |
|------|------|
| 扩散/Flow 的物理直觉 | [Diffusion Policy](../diffusion-flow/diffusion_policy.md) · [Flow Matching (π0)](../vla-core/pi0_code_analysis.md) |
| 波动方程视觉 | [WaveFormer](../perception/waveformer_wave_equation_vision_2026.md) |
| 等变策略（对称性） | [EquiBim](../deployment/equibim_learning_symmetry_equivariant_policy_for_bimanual_ma_dissection.md) |
| 推理→动作诊断 | [GenieReasoner/ERIQ](../planning/geniereasoner_eriq_fact.md) |
| 压缩 vs 记忆 | [Compression Gap](../diffusion-flow/the_compression_gap_why_discrete_tokenization_limits_vision_dissection.md) |
| 鸽子磁感（生物物理→感知） | [Pigeon Magnetoreception](pigeon_magnetoreception_vestibular_electrosense.md) |
| 皮层下控制（神经科学→反射） | [Subcortical Control](subcortical_control_knobs_neuropeptides_temporality.md) |

### 外部参考

- **刘子鸣专访原文**：《另辟蹊径，不赌规模：Physics of AI 是通往 AGI 的"科学路径"》
- **KAN**：[Kolmogorov-Arnold Networks (arXiv:2404.19756)](https://arxiv.org/abs/2404.19756)
- **Grokking / The Clock and the Pizza**：[arXiv:2306.17844](https://arxiv.org/abs/2306.17844)
- **PFGM (Poisson Flow)**：[arXiv:2209.11178](https://arxiv.org/abs/2209.11178)
- **Neural Scaling Laws**：[arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
- **Information Bottleneck**：Shwartz-Ziv & Tishby, 2017

---

[← Back to Explorer's Map](../README.md)
