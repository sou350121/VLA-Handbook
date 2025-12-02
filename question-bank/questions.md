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

## 📚 30 道进阶题 (Advanced Questions)

### 类别 A: 策略生成与动作表示 (Policy & Action)

| # | 问题 | 详细解答位置 | 一句话答案 |
| :--- | :--- | :--- | :--- |
| A1 | Diffusion Policy 如何解决多模态分布问题？ | [diffusion_policy.md](../theory/diffusion_policy.md) | 学习能量函数，允许多个低谷 (Modes)，不取平均 |
| A2 | DDPM 和 DDIM 的区别？为什么 DDIM 更快？ | [diffusion_policy.md](../theory/diffusion_policy.md#41-ddim) | DDIM 是确定性过程，可跳步；DDPM 是随机的 |
| A3 | 什么是 Action Chunking？为什么有效？ | [act.md](../theory/act.md#21-动作分块-action-chunking) | 一次预测多步动作，减少误差累积和决策频率 |
| A4 | ACT 的 Temporal Ensemble 如何工作？ | [act.md](../theory/act.md#22-时间集成-temporal-ensemble) | 重叠预测 + 指数加权平均，平滑轨迹交界 |
| A5 | FAST 为什么用 DCT 而不是 FFT？ | [fast.md](../theory/fast.md#313-为什么-dct-而不是-fft) | DCT 只有实数，边界友好，能量集中性更好 |
| A6 | FAST 的 BPE 如何压缩动作序列？ | [fast.md](../theory/fast.md#32-字节对编码-bpe) | 统计高频 DCT 系数组合，合并为单个 token |
| A7 | Flow Matching 比 Diffusion 好在哪？ | [pi0_flow_matching.md](../theory/pi0_flow_matching.md#12-为什么比-diffusion-好) | 轨迹更直 (ODE)，推理更快 (<10步)，更稳定 |
| A8 | Flow Matching 的损失函数是什么？ | [pi0_flow_matching.md](../theory/pi0_flow_matching.md#23-损失函数) | $\|v_\theta(x_t, t) - (x_1 - x_0)\|^2$，预测速度场 |
| A9 | 为什么 VLA 通常预测 Delta Pose 而非 Absolute？ | [spatial_math.md](../theory/spatial_math.md#6-面试高频考点) | 泛化性好，对外参误差不敏感，配合闭环更鲁棒 |
| A10 | 6D Rotation 比四元数好在哪？ | [spatial_math.md](../theory/spatial_math.md#33-6d-旋转表示) | 连续性！无双倍覆盖问题，训练更稳定 |

### 类别 B: 训练技术与优化 (Training & Optimization)

| # | 问题 | 详细解答位置 | 一句话答案 |
| :--- | :--- | :--- | :--- |
| B1 | 知识蒸馏中温度参数 T 的作用？ | [knowledge_distillation.md](../theory/knowledge_distillation.md#22-软标签) | T 越大分布越平滑，保留更多"暗知识" |
| B2 | 知识蒸馏的软标签为什么比硬标签好？ | [knowledge_distillation.md](../theory/knowledge_distillation.md) | 软标签包含类间关系信息（如"猫像狗"） |
| B3 | 对比学习 InfoNCE 损失的直觉？ | [self_supervised_learning.md](../theory/self_supervised_learning.md#211-infonce-损失) | 拉近正样本对，推远负样本对 |
| B4 | MAE 和 SimCLR 的核心区别？ | [self_supervised_learning.md](../theory/self_supervised_learning.md) | MAE 是掩码重建，SimCLR 是对比学习 |
| B5 | Co-training 为什么能防止灾难性遗忘？ | [co_training.md](../theory/co_training.md#11-防止灾难性遗忘) | 混合 Web 数据保持 VLM 通用语义知识 |
| B6 | Co-training 的 Loss Masking 如何实现？ | [co_training.md](../theory/co_training.md#22-loss-计算) | 机器人数据算 Action Loss，Web 数据算 Text Loss |
| B7 | Sim-to-Real 的 Domain Randomization 原理？ | [transfer_learning.md](../theory/transfer_learning.md) | 仿真中随机化参数，让真机成为分布中的一种 |
| B8 | 什么时候用 Feature Extraction vs Fine-tuning？ | [transfer_learning.md](../theory/transfer_learning.md#22-冻结特征提取) | 数据极少 (<50) 用冻结，数据多用微调 |
| B9 | 量化中 Per-Tensor vs Per-Channel 的区别？ | [quantization_theory.md](../theory/quantization_theory.md) | Per-Channel 每通道独立量化，精度更高 |
| B10 | AWQ 如何找到重要权重？ | [quantization_theory.md](../theory/quantization_theory.md) | 观察激活值分布，激活大的权重更重要 |

### 类别 C: 模型架构与推理 (Architecture & Reasoning)

| # | 问题 | 详细解答位置 | 一句话答案 |
| :--- | :--- | :--- | :--- |
| C1 | VLA 中 Early/Mid/Late Fusion 如何选择？ | [multimodal_models.md](../theory/multimodal_models.md#q3-多模态融合中-early--mid--late-fusion-如何选择) | Mid Fusion (Cross-Attention) 是 VLA 首选 |
| C2 | FiLM 调制是什么？在 VLA 中如何应用？ | [multimodal_models.md](../theory/multimodal_models.md#34-vla-中的主流方案film-调制) | $\gamma \cdot \text{feature} + \beta$，注入条件信息 |
| C3 | SigLIP 比 CLIP 好在哪？ | [transformer_vs_cnn.md](../theory/transformer_vs_cnn.md#52-siglip) | Sigmoid 替代 Softmax，无需全局同步，Batch 独立 |
| C4 | CoT 在 VLA 中有什么价值？ | [chain_of_thought.md](../theory/chain_of_thought.md#12-cot-的价值) | 可解释性、复杂任务分解、错误纠正、泛化增强 |
| C5 | 显式 CoT 和隐式 CoT 的区别？ | [chain_of_thought.md](../theory/chain_of_thought.md) | 显式输出语言推理，隐式在 Latent 空间推理 |
| C6 | RT-2 的 Co-fine-tuning 为什么重要？ | [vla_arch.md](../theory/vla_arch.md#2-rt-2) | 混合 Web+Robot 数据，保持语义理解+学习动作 |
| C7 | OpenVLA 的 Action Head 是如何设计的？ | [vla_arch.md](../theory/vla_arch.md#openvla) | 专门的 Linear Layer 预测去离散化的动作 Token |
| C8 | π0.5 的分层推理架构是什么？ | [pi0_5_dissection.md](../theory/pi0_5_dissection.md) | 高层 VLM 规划 + 低层 Flow 执行 |
| C9 | WALL-OSS 的 Uni-CoT 有什么特点？ | [wall_oss.md](../theory/wall_oss.md) | 统一的视觉-语言-动作跨层思维链 |
| C10 | Galaxea G0 的"大脑+小脑"架构是什么？ | [galaxea_g0.md](../theory/galaxea_g0.md) | VLM 大脑规划 + VLA 小脑执行，分层 CoT |

<details>
<summary>点击展开 30 道题的简答汇总</summary>

### 类别 A: 策略生成与动作表示

**A1: Diffusion Policy 如何解决多模态分布问题？**
- MSE 回归会预测多解的平均值（撞墙）
- Diffusion 学习能量函数，允许多个低谷存在
- 随机采样出完整的"左绕"或"右绕"轨迹

**A2: DDPM 和 DDIM 的区别？**
- **DDPM**: 随机游走去噪，需要 100 步
- **DDIM**: 确定性 ODE，可跳步到 10-15 步
- DDIM 通过非马尔可夫重参数化实现加速

**A3: 什么是 Action Chunking？**
- 一次预测 $k$ 步动作 (如 $k=100$)
- 减少决策频率 50Hz → 0.5Hz
- 隐式任务分解，轨迹更平滑

**A4: ACT 的 Temporal Ensemble？**
- 每步都预测完整 chunk，形成重叠
- 指数加权平均：$w_i = \exp(-m \cdot i)$
- 新预测权重大，平滑轨迹交界

**A5: FAST 为什么用 DCT？**
- DCT 只有实数（FFT 有复数）
- 边界对称延拓，无高频伪影
- 能量集中性更好（JPEG 也用 DCT）

**A6: FAST 的 BPE 压缩？**
- 统计高频 DCT 系数组合
- 合并为单个 token（如 [42,15] → token_256）
- 类似 GPT 文本 BPE，压缩比 2.3:1

**A7: Flow Matching 优势？**
- 学习直线轨迹（最优传输）
- 确定性 ODE，推理 <10 步
- 动作更平滑，无高频抖动

**A8: Flow Matching 损失？**
- $\mathcal{L} = \|v_\theta(x_t, t, \text{cond}) - (x_1 - x_0)\|^2$
- 预测从数据到噪声的速度向量
- 目标是恒定向量，非常直观

**A9: 为什么预测 Delta Pose？**
- 泛化性：在不同位置/机器人通用
- 对外参误差不敏感
- 配合高频闭环可自我校正

**A10: 6D Rotation 优势？**
- 连续映射 $\mathbb{R}^6 \to SO(3)$
- 无四元数双倍覆盖问题
- Gram-Schmidt 后处理保证正交性

### 类别 B: 训练技术与优化

**B1: 温度参数 T 的作用？**
- T=1 正常 softmax
- T>1 分布更平滑，保留暗知识
- 暗知识：错误类别的概率也有信息

**B2: 软标签为什么好？**
- 包含类间关系（"猫"和"狗"比"猫"和"车"更像）
- 提供更丰富的监督信号
- 学生模型学到更泛化的特征

**B3: InfoNCE 直觉？**
- 分子：正样本对相似度
- 分母：所有样本对相似度之和
- 最大化正样本比例 = 拉近正样本

**B4: MAE vs SimCLR？**
- **MAE**: 掩码 75% patch，重建像素
- **SimCLR**: 同图两视图，对比学习
- MAE 更高效，SimCLR 需要大 batch

**B5: Co-training 防遗忘？**
- 机器人数据是 Narrow Domain
- Web 数据保持 Wide Domain 知识
- 混合训练让 VLM 不忘"泰勒斯威夫特是谁"

**B6: Loss Masking？**
- 机器人数据：Action Head Loss
- Web 数据：Text Token Loss
- 互不干扰，各学各的

**B7: Domain Randomization？**
- 随机化仿真参数（光照、纹理、质量）
- 真机成为随机分布中的一种
- 模型学会对这些变化鲁棒

**B8: Feature Extraction vs Fine-tuning？**
- 数据 <50：冻结 backbone，只训练 head
- 数据 >500：全量或 LoRA 微调
- 防止小数据过拟合预训练特征

**B9: Per-Tensor vs Per-Channel？**
- Per-Tensor：整个张量共享 scale/zero
- Per-Channel：每个通道独立量化
- Per-Channel 精度高但存储开销大

**B10: AWQ 找重要权重？**
- 观察激活值分布
- 激活值大的通道更重要
- 对重要权重保持高精度

### 类别 C: 模型架构与推理

**C1: Fusion 策略选择？**
- Early：模态相似（多相机）
- Mid (Cross-Attention)：VLA 首选
- Late：需要模块化解释性

**C2: FiLM 调制？**
- $\text{output} = \gamma \cdot \text{feature} + \beta$
- $\gamma, \beta$ 由条件信息生成
- RT-1 用语言调制视觉，Diffusion 用时间步调制

**C3: SigLIP 优势？**
- Sigmoid 替代 Softmax
- 每对独立计算，无需全局同步
- 对 Batch Size 不敏感

**C4: CoT 在 VLA 的价值？**
- 可解释：知道机器人"在想什么"
- 任务分解：复杂任务拆成子步骤
- 错误纠正：推理中发现矛盾

**C5: 显式 vs 隐式 CoT？**
- 显式：输出语言推理步骤
- 隐式：在 Latent 空间推理
- π0.5 用隐式 CoT，更快

**C6: RT-2 Co-fine-tuning？**
- 混合 Web VQA + Robot 数据
- 动作编码为文本 token
- 保持语义理解 + 学习控制

**C7: OpenVLA Action Head？**
- 专门的 Linear Layer
- 输出离散化动作 token
- 通过 Detokenization 还原连续值

**C8: π0.5 分层推理？**
- 高层：VLM 生成计划/子任务
- 低层：Flow Matching 生成动作
- 类似人类"想清楚再动手"

**C9: WALL-OSS Uni-CoT？**
- 统一的跨模态思维链
- 视觉→语言→动作全程推理
- 双输出头 + 流匹配控制

**C10: Galaxea G0 大脑+小脑？**
- VLM 大脑：高层规划
- VLA 小脑：低层执行
- 三阶段课程学习

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
