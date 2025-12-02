# 面试题库 (Question Bank)

本题库涵盖了 VLA 算法岗面试的高频问题，分为概念题、场景题和代码题。

---

## 🔥 高频八问 (Top 8 Must-Know Questions)

以下是面试中最常被问到的 8 道核心问题，点击链接可跳转到详细解答。

| # | 问题 | 详细解答位置 | 一句话答案 |
| :--- | :--- | :--- | :--- |
| 1 | 自注意力机制是什么？计算复杂度怎么算？ | [transformer_vs_cnn.md](../theory/transformer_vs_cnn.md#6-自注意力机制详解-self-attention-deep-dive) | $O(N^2 d)$ 时间，$O(N^2)$ 空间 |
| 2 | KV-Cache 如何加速推理？ | [flash_attention.md](../theory/flash_attention.md#5-kv-cache-推理加速-kv-cache-for-inference) | 缓存历史 K/V，每 Token $O(N^2) \to O(N)$ |
| 3 | LoRA 原理？与 P-Tuning/Adapter 异同？ | [peft_lora.md](../theory/peft_lora.md#4-peft-方法对比-comparison) | 低秩分解 $\Delta W=BA$，可合并无延迟 |
| 4 | RLHF 流程？与 DPO 差异？ | [reinforcement_learning.md](../theory/reinforcement_learning.md#61-rlhf-完整流程-rlhf-pipeline) | RLHF 三阶段，DPO 跳过 Reward Model |
| 5 | TP/PP/DP 分别是什么？ | [large_scale_training.md](../system-design/large_scale_training.md#q0-分布式训练中的-tpppDP-分别是什么) | DP 切数据，TP 切矩阵，PP 切层 |
| 6 | Flash Attention 原理？ | [flash_attention.md](../theory/flash_attention.md#6-面试常见问题) | Tiling + Kernel Fusion + Online Softmax |
| 7 | 视觉误判如何语言纠错？ | [multimodal_models.md](../theory/multimodal_models.md#q6-如果视觉模块误判如何通过语言纠错) | 闭环反馈 / CoT 自检 / 多模态一致性 |
| 8 | 如何构建 Evaluation Pipeline？ | [evaluation.md](../theory/evaluation.md#5-evaluation-pipeline-构建-building-evaluation-pipeline) | 数据 → 推理 → 指标 → 日志，CI/CD 集成 |
| 9 | Model-Based vs Model-Free 区别？ | [reinforcement_learning.md](../theory/reinforcement_learning.md#31-model-free-vs-model-based) | Model-Free 直接学策略，Model-Based 先学环境模型 |
| 10 | 马尔可夫性是什么？ | [reinforcement_learning.md](../theory/reinforcement_learning.md#22-马尔可夫性-markov-property) | 下一状态只依赖当前状态，与历史无关 |
| 11 | 为什么最优价值函数就是最优策略？ | [reinforcement_learning.md](../theory/reinforcement_learning.md#25-最优价值函数与最优策略) | $\pi^*(s) = \arg\max_a Q^*(s,a)$，贪心即最优 |
| 12 | 策略迭代 vs 值迭代？ | [reinforcement_learning.md](../theory/reinforcement_learning.md#32-策略迭代-vs-值迭代-policy-iteration-vs-value-iteration) | 策略迭代先评估再改进，值迭代直接迭代 Bellman |

### 快速回顾

<details>
<summary>点击展开 8 道题的简答</summary>

**Q1: 自注意力机制是什么？计算复杂度怎么算？**
- **公式**: $\text{Attention}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d_k})V$
- **复杂度**: 时间 $O(N^2 d)$，空间 $O(N^2)$（存储注意力矩阵）
- **瓶颈**: 序列长度 $N$ 大时显存爆炸 → Flash Attention 解决

**Q2: KV-Cache 如何加速推理？**
- **问题**: 自回归生成时每个 Token 都要重算历史 K/V
- **方案**: 缓存已计算的 K/V，新 Token 只算增量
- **效果**: 每 Token 计算 $O(N^2 d) \to O(Nd)$，N=1000 时约 1000x 加速
- **代价**: 额外显存 $O(LNd)$

**Q3: LoRA 原理？与 P-Tuning/Adapter 异同？**
- **LoRA**: $W = W_0 + BA$，低秩分解，推理时可合并，无额外延迟
- **P-Tuning**: 可学习 Soft Prompt，占用 Context Window
- **Adapter**: 层间插入 MLP，有推理延迟
- **核心差异**: LoRA 是唯一可"无痕合并"的方法

**Q4: RLHF 流程？与 DPO 差异？**
- **RLHF 三阶段**: SFT → Reward Model → PPO
- **DPO**: 跳过 Reward Model，直接从偏好数据优化
- **对比**: RLHF 需 4 模型，DPO 只需 2 模型，更稳定但效果略差

**Q5: TP/PP/DP 分别是什么？**
- **DP (Data Parallel)**: 切数据，All-Reduce 梯度
- **TP (Tensor Parallel)**: 切矩阵，All-Reduce 激活
- **PP (Pipeline Parallel)**: 切层，点对点传输
- **选择**: 7B 用 FSDP，70B+ 用 3D 并行

**Q6: Flash Attention 原理？**
- **Tiling**: 分块计算，避免存储 $N \times N$ 矩阵
- **Kernel Fusion**: QK^T → softmax → ×V 融合进单个 Kernel
- **Online Softmax**: 增量更新归一化
- **效果**: 内存 $O(N^2) \to O(N)$，速度 2-4x

**Q7: 视觉误判如何语言纠错？**
- **闭环反馈**: 用户语言指令纠正 ("不对，是左边那个")
- **CoT 自检**: 输出推理链，发现矛盾
- **多模态一致性**: 语言-视觉 Embedding 相似度检查
- **主动询问**: 低置信度时请求确认

**Q8: 如何构建 Evaluation Pipeline？**
- **数据**: CALVIN/SIMPLER 标准测试集
- **推理**: 多 Checkpoint 并行评估
- **指标**: SR/MSS/IR + Wilson 置信区间
- **日志**: W&B/TensorBoard + 失败案例分析
- **CI/CD**: 训练后自动触发评估

**Q9: Model-Based 和 Model-Free 的区别？**
- **Model-Free**: 直接学习策略或价值函数，不尝试理解环境
- **Model-Based**: 先学习环境动力学 $P(s'|s,a)$，再利用模型规划
- **Trade-off**: Model-Free 简单但样本效率低，Model-Based 高效但有模型误差

**Q10: 马尔可夫性是什么？**
- **定义**: $P(s_{t+1}|s_t, a_t, s_{t-1}, ...) = P(s_{t+1}|s_t, a_t)$
- **含义**: 当前状态是对历史的"充分统计量"，未来只依赖现在
- **重要性**: Bellman 方程成立的前提，简化计算

**Q11: 为什么最优价值函数就是最优策略？**
- **核心**: $\pi^*(s) = \arg\max_a Q^*(s,a)$
- **原因**: $Q^*(s,a)$ 表示执行 $a$ 后按最优策略行动的期望回报
- **结论**: 对 $Q^*$ 贪心就是最优的，两者是"一体两面"

**Q12: 策略迭代和值迭代的区别？**
- **策略迭代**: 评估 (计算 $V^\pi$) + 改进 (贪心更新 $\pi$) 交替进行
- **值迭代**: 直接迭代 Bellman 最优方程 $V(s) \leftarrow \max_a [R + \gamma \sum P V']$
- **效率**: 策略迭代每轮迭代次数少但每轮计算多，值迭代相反

</details>

---

## 1. 概念题 (Conceptual Questions)

### Q1: 解释一下 RT-2 的 Co-fine-tuning 策略，为什么它很重要？
- **参考答案**:
    - RT-2 将机器人动作编码为文本 Token，与互联网 VQA 数据混合训练。
    - **重要性**: 纯机器人数据量太小，容易导致模型遗忘预训练 VLM 的语义知识 (Catastrophic Forgetting)。混合训练让模型既能听懂 "抓恐龙" (语义)，又能输出动作 (控制)，实现了 Zero-shot 泛化。

### Q2: VLA 模型中，Action Tokenization 和 Continuous Regression 有什么区别？
- **参考答案**:
    - **Tokenization (离散化)**: 将连续动作空间划分为 bins (e.g., 256个)，作为分类问题处理。
        - *优点*: 可以建模多模态分布 (Multimodal Distribution)，即同一个状态下可能有多种合理的动作。Transformer 擅长处理离散序列。
        - *缺点*: 精度受限于 bin 的数量，高频控制可能不平滑。
    - **Regression (回归)**: 直接预测连续数值 (MSE Loss)。
        - *优点*: 精度高，适合精细操作。
        - *缺点*: 假设动作分布是单峰高斯 (Unimodal Gaussian)，难以处理多解情况 (e.g., 从左边抓还是右边抓)。

### Q3: 什么是 Sim-to-Real Gap？如何解决？
- **参考答案**:
    - 仿真与真机在视觉 (光照、纹理) 和动力学 (摩擦、质量) 上的差异。
    - **解决方法**:
        1. **Domain Randomization**: 在仿真中随机化各种参数，让真机成为分布中的一种。
        2. **Domain Adaptation**: 使用 GAN 或特征对齐技术。
        3. **System Identification**: 辨识真机参数反馈给仿真。

### Q4: Transformer vs CNN 在 VLA 中的选择：什么时候用 ViT，什么时候用 ResNet？
- **参考答案**:
    - **ViT (优选)**:
        - **多模态统一**: 需要将视觉、语言、动作全部 Tokenize 并拼接 (e.g., RT-2, OpenVLA)。
        - **全局上下文**: 需要理解画面中远距离的物体关系 (e.g., "桌子左边的杯子和右边的壶")。
        - **Scaling Law**: 有大量数据时 ViT 效果更好。
    - **ResNet (优选)**:
        - **小样本**: 数据不足时 CNN 的归纳偏置有助于快速收敛。
        - **纹理特征**: 需要高频细节（如 GelSight 触觉传感器）。
        - **推理速度**: CNN 通常比 ViT 轻量，适合边缘设备。

### Q5: 什么是触觉 VLA (Tactile VLA)？为什么视觉不够？
- **参考答案**:
    - **Tactile VLA**: 融合了触觉传感器 (e.g., GelSight) 的 VLA 模型，能够感知接触、材质、滑移等视觉无法获取的信息。
    - **为什么视觉不够**:
        1. **遇挡 (Occlusion)**: 机械手抓取时，手掌会挡住摄像头。
        2. **物理属性**: 视觉无法直接判断软硬、摩擦力。
        3. **微米级控制**: 触觉传感器提供更高精度的反馈。
    - **代表模型**: VLA-Touch (2025), OmniVTLA (2025)。

### Q6: VLA-Touch 和 OmniVTLA 有什么区别？
- **参考答案**:
    - **VLA-Touch**: 双层反馈机制。
        - **High-level**: 使用 Tactile-Language Model (TLM) 将触觉信号翻译成语言，辅助 VLM 决策。
        - **Low-level**: 通过 FiLM 将触觉特征注入 Diffusion Policy。
        - **优势**: 无需重训整个 VLA，即插即用。
    - **OmniVTLA**: 统一模型 (Unified Tokenization)。
        - **架构**: 将视觉、触觉、语言全部 Token 化，输入同一个 Transformer。
        - **语义对齐**: 使用 InfoNCE Loss 拉近触觉 Embedding 与材质描述文本的距离。
        - **优势**: 能执行跨模态推理 (e.g., "Pick up the softest object")。

## 2. 场景题 (Scenario Questions)

### S1: 只有 100 条真机演示数据，如何训练一个鲁棒的抓取策略？
- **参考答案**:
    - **数据增强**: 旋转、裁剪、颜色抖动。
    - **Sim-to-Real**: 先在仿真中训练一个基础策略，用这 100 条数据做 Fine-tuning。
    - **Co-training**: 混合大规模开源数据集 (如 OXE) 进行训练，但这 100 条数据赋予更高的采样权重。
    - **使用预训练模型**: 基于 OpenVLA 或 RT-1 预训练权重进行微调 (LoRA)。

### S2: 机器人抓取透明物体 (Transparent Object) 总是失败，怎么办？
- **参考答案**:
    - **传感器层面**: RGB-D 相机的深度光在透明物体上会失效 (穿透/反射)。考虑使用立体视觉 (Stereo) 或 补全深度图 (Depth Completion)。
    - **算法层面**:
        - 在训练数据中加入大量透明物体。
        - 使用 **末端触觉 (Tactile Sensor)**: 视觉可能看不准，但摸到了就知道。
        - **多视角融合 (Multi-view Fusion)**: 移动机械臂从不同角度观察，利用镜面反射特征。

## 3. 代码题 (Coding Questions)

### C1: 实现一个简单的 PID 控制器
```python
class PIDController:
    def __init__(self, kp, ki, kd, dt):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral = 0
        self.prev_error = 0

    def update(self, target, current):
        error = target - current
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
        self.prev_error = error
        return output
```

### C2: 计算两个 3D 点之间的欧氏距离 (NumPy)
```python
import numpy as np

def distance(p1, p2):
    # p1, p2: np.array [x, y, z]
    return np.linalg.norm(p1 - p2)
```

### C3: 旋转矩阵转欧拉角 (概念)
- 面试官可能会问转换公式或万向节死锁问题。建议复习 `scipy.spatial.transform.Rotation` 的用法。


---
[← Back to Question Bank](./README.md)
