# 面试题库 (Question Bank)

本题库涵盖了 **VLA Handbook** 中与面试强相关的高频问题，按三个核心类别组织：**策略生成与动作表示**、**训练技术与优化**、**模型架构与推理**。

每个问题都包含：
- **核心概念**：一句话定义
- **为什么重要**：解决的问题/价值
- **如何实现**：关键技术点
- **实际例子**：VLA 中的应用
- **详细解答**：链接到理论文件的具体章节

---

## 📚 类别 A: 策略生成与动作表示 (Policy & Action)

### 🔥 高频题 (Top Questions)

#### A1: 自注意力机制是什么？计算复杂度怎么算？

**核心概念**: 自注意力机制是 Transformer 的核心，通过计算序列中每个位置与其他位置的相似度来建模全局依赖关系。

**为什么重要**: 
- 解决了 CNN 感受野受限的问题，能够直接建模长距离依赖
- 是 VLA 中多模态融合（视觉-语言-动作）的基础

**如何实现**:
- **公式**: $\text{Attention}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d_k})V$
- **复杂度**: 时间 $O(N^2 d)$，空间 $O(N^2)$（存储注意力矩阵）
- **瓶颈**: 序列长度 $N$ 大时显存爆炸 → Flash Attention 解决

**实际例子**: 
- ViT 中每个 Patch 都能"看到"图像中的所有其他 Patch
- VLA 中语言 Token 可以关注到所有视觉 Token

**详细解答**: [transformer_vs_cnn.md](../theory/transformer_vs_cnn.md#6-自注意力机制详解-self-attention-deep-dive)

---

#### A2: 什么是 Action Chunking？为什么有效？

**核心概念**: Action Chunking 是一次性预测多步动作（如 100 步），而非逐帧预测。

**为什么重要**: 
- 减少误差累积：单步预测的误差会在时间上传播放大
- 降低决策频率：从 50Hz → 0.5Hz，减少计算开销
- 隐式任务分解：模型自动学习将复杂任务分解为子动作序列

**如何实现**:
- ACT 中一次预测 $k$ 步动作（$k=100$）
- 使用 CVAE 建模多模态分布
- Temporal Ensemble 平滑轨迹交界

**实际例子**: 
- 抓取任务：模型一次性预测"伸手→张开夹爪→闭合→抬起"的完整序列
- 相比逐帧预测，轨迹更平滑，成功率更高

**详细解答**: [act.md](../theory/act.md#21-动作分块-action-chunking)

---

#### A3: ACT 的 Temporal Ensemble 如何工作？

**核心概念**: Temporal Ensemble 通过重叠预测和指数加权平均来平滑动作轨迹的交界处。

**为什么重要**: 
- 解决 Chunk 边界不连续问题：相邻 Chunk 可能预测不同的动作
- 提高轨迹平滑性：减少高频抖动，对机器人控制至关重要

**如何实现**:
- 每步都预测完整 Chunk，形成重叠窗口
- 指数加权平均：$w_i = \exp(-m \cdot i)$，新预测权重大
- 最终动作 = $\sum w_i \cdot \text{action}_i$

**实际例子**: 
- 第 1 步预测 [t0-t100]，第 2 步预测 [t1-t101]
- 在 t1 时刻，使用两个预测的加权平均，新预测权重更高

**详细解答**: [act.md](../theory/act.md#22-时间集成-temporal-ensemble)

---

#### A4: 为什么 VLA 通常预测 Delta Pose 而非 Absolute？

**核心概念**: Delta Pose 是相对于当前位姿的增量变化，而非世界坐标系下的绝对位置。

**为什么重要**: 
- **泛化性**: 在不同位置/机器人上都能工作，不依赖绝对坐标系
- **鲁棒性**: 对外参（相机标定）误差不敏感
- **闭环控制**: 配合高频闭环可以自我校正累积误差

**如何实现**:
- 预测 $\Delta x, \Delta y, \Delta z, \Delta \text{roll}, \Delta \text{pitch}, \Delta \text{yaw}$
- 执行时：$\text{new\_pose} = \text{current\_pose} + \Delta \text{pose}$

**实际例子**: 
- 抓取任务：模型预测"向前移动 5cm"，而非"移动到 (1.2, 0.5, 0.8)"
- 即使机器人位置改变，指令依然有效

**详细解答**: [spatial_math.md](../theory/spatial_math.md#6-面试高频考点)

---

#### A5: 6D Rotation 比四元数好在哪？

**核心概念**: 6D Rotation 表示用两个 3D 向量（6 个数）表示旋转，通过 Gram-Schmidt 正交化还原为旋转矩阵。

**为什么重要**: 
- **连续性**: 这是唯一在欧几里得空间中连续的旋转表示
- **无双倍覆盖问题**: 四元数 $q$ 和 $-q$ 表示相同旋转，导致训练不稳定
- **训练友好**: 神经网络回归更稳定，Loss 不会因为符号翻转而爆炸

**如何实现**:
- 预测 $r_1, r_2 \in \mathbb{R}^3$（未归一化）
- Gram-Schmidt: $x = r_1 / \|r_1\|$, $z = (x \times r_2) / \|x \times r_2\|$, $y = z \times x$
- 组装 $R = [x, y, z]$

**实际例子**: 
- Diffusion Policy 和 π0 都使用 6D Rotation
- 相比四元数，训练 Loss 更稳定，收敛更快

**详细解答**: [spatial_math.md](../theory/spatial_math.md#33-6d-旋转表示)

---

### 📖 进阶题 (Advanced Questions)

#### A6: Diffusion Policy 如何解决多模态分布问题？

**核心概念**: Diffusion Policy 学习一个能量函数，允许多个低谷（Modes）存在，而非像 MSE 回归那样预测平均值。

**为什么重要**: 
- **多解问题**: 同一状态下可能有多种合理动作（如"左绕"或"右绕"障碍物）
- MSE 回归会预测两者的平均值（撞墙），Diffusion 可以采样出完整轨迹

**如何实现**:
- 前向过程：逐步添加噪声 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$
- 逆向过程：学习去噪网络 $p_\theta(x_{t-1}|x_t, \text{obs})$
- 采样：从噪声开始，逐步去噪得到动作

**实际例子**: 
- 抓取任务：从左边抓和从右边抓都是合理的
- Diffusion 可以随机采样出其中一种，而非两者的平均（无效动作）

**详细解答**: [diffusion_policy.md](../theory/diffusion_policy.md#11-多模态分布问题-the-multimodality-problem)

---

#### A7: DDPM 和 DDIM 的区别？为什么 DDIM 更快？

**核心概念**: DDPM 是随机过程，需要 100 步去噪；DDIM 是确定性 ODE，可跳步到 10-15 步。

**为什么重要**: 
- **推理速度**: DDIM 可以大幅加速推理（10x），适合实时控制
- **确定性**: ODE 轨迹是确定的，动作更平滑

**如何实现**:
- **DDPM**: $x_{t-1} = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta) + \sigma_t z$（随机项 $z$）
- **DDIM**: $x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}} \epsilon_\theta$（确定性）

**实际例子**: 
- Diffusion Policy 推理：DDPM 需要 100 步（~2 秒），DDIM 只需 10 步（~0.2 秒）
- 实时性要求高的场景必须用 DDIM

**详细解答**: [diffusion_policy.md](../theory/diffusion_policy.md#41-ddim)

---

#### A8: FAST 为什么用 DCT 而不是 FFT？

**核心概念**: DCT（离散余弦变换）只有实数输出，边界友好，能量集中性更好，更适合动作序列压缩。

**为什么重要**: 
- **实数信号**: 机器人动作是实数，FFT 会产生复数，浪费存储
- **边界处理**: DCT 假设对称延拓，避免边界不连续导致的高频伪影
- **压缩效率**: DCT 的能量集中性比 FFT 更好（JPEG 也用 DCT）

**如何实现**:
- 对每个关节的时间序列独立应用 1D DCT
- 保留前 $K$ 个低频系数，丢弃高频
- 压缩比 2.5:1（10 步 → 4 个系数）

**实际例子**: 
- FAST 将 10 步动作序列压缩为 4 个 DCT 系数
- 类似 JPEG 压缩图像，利用频域稀疏性

**详细解答**: [fast.md](../theory/fast.md#313-为什么-dct-而不是-fft)

---

#### A9: FAST 的 BPE 如何压缩动作序列？

**核心概念**: BPE（字节对编码）统计高频 DCT 系数组合，合并为单个 Token，类似 GPT 的文本 BPE。

**为什么重要**: 
- **进一步压缩**: DCT 后仍有大量系数，BPE 可以进一步减少 Token 数
- **利用统计模式**: 某些系数组合频繁出现，可以合并

**如何实现**:
- 初始化词汇表 = $\{0, 1, ..., 255\}$（量化后的 DCT 系数）
- 迭代：统计最高频的相邻对（如 `[42, 15]`），合并为新 Token
- 重复直到词汇表达到目标大小（如 8000）

**实际例子**: 
- 原始序列: `[42, 15, 3, 1]` (4 tokens)
- BPE 后: `[token_256, 3, 1]` (3 tokens)，其中 `token_256 = [42, 15]`
- 压缩比 2.3:1

**详细解答**: [fast.md](../theory/fast.md#32-字节对编码-bpe)

---

#### A10: Flow Matching 比 Diffusion 好在哪？

**核心概念**: Flow Matching 学习确定性的 ODE 轨迹（直线），而非 Diffusion 的随机游走，推理更快更稳定。

**为什么重要**: 
- **推理速度**: Flow Matching 只需 <10 步，Diffusion 需要 50-100 步
- **轨迹平滑**: 直线轨迹比随机游走更平滑，减少高频抖动
- **稳定性**: 确定性 ODE 比随机过程更稳定

**如何实现**:
- 学习速度场 $v_\theta(x_t, t, \text{cond})$，预测从数据到噪声的速度向量
- 损失函数: $\mathcal{L} = \|v_\theta(x_t, t) - (x_1 - x_0)\|^2$
- 推理：从噪声 $x_1$ 开始，沿着速度场积分到 $x_0$

**实际例子**: 
- π0 使用 Flow Matching，推理速度比 Diffusion Policy 快 10x
- 动作更平滑，适合精细控制

**详细解答**: [pi0_flow_matching.md](../theory/pi0_flow_matching.md#12-为什么比-diffusion-好)

---

#### A11: Flow Matching 的损失函数是什么？

**核心概念**: Flow Matching 的损失函数是预测速度场与目标速度（常数向量）的 L2 距离。

**为什么重要**: 
- **简单直观**: 目标是恒定向量 $x_1 - x_0$，非常直观
- **易于优化**: L2 Loss 比 Diffusion 的 Score Matching 更稳定

**如何实现**:
- **目标速度**: $u_t = x_1 - x_0$（恒定向量）
- **预测速度**: $v_\theta(x_t, t, \text{cond})$
- **损失**: $\mathcal{L} = \mathbb{E}_{t, x_0, x_1} \|v_\theta(x_t, t) - (x_1 - x_0)\|^2$

**实际例子**: 
- 训练时：给定真实动作 $x_0$ 和噪声 $x_1$，模型学习预测速度方向
- 推理时：从噪声开始，沿着预测的速度场积分得到动作

**详细解答**: [pi0_flow_matching.md](../theory/pi0_flow_matching.md#23-损失函数)

---

#### A12: VLA 模型中，Action Tokenization 和 Continuous Regression 有什么区别？

**核心概念**: Tokenization 将动作离散化为分类问题，Regression 直接预测连续值。

**为什么重要**: 
- **多模态分布**: Tokenization 可以建模多解（如"左抓"或"右抓"），Regression 假设单峰高斯
- **Transformer 优势**: Tokenization 更适合 Transformer 的离散序列建模能力

**如何实现**:
- **Tokenization**: 将连续空间划分为 bins（如 256 个），作为分类问题
- **Regression**: 直接预测连续值，使用 MSE Loss

**实际例子**: 
- RT-2 / OpenVLA 使用 Tokenization，可以处理多解情况
- 某些精细操作任务使用 Regression，精度更高

**详细解答**: [action_representations.md](../theory/action_representations.md)

---

## 🚀 类别 B: 训练技术与优化 (Training & Optimization)

### 🔥 高频题 (Top Questions)

#### B1: LoRA 原理？与 P-Tuning/Adapter 异同？

**核心概念**: LoRA（Low-Rank Adaptation）通过低秩分解 $\Delta W = BA$ 来微调大模型，只需训练少量参数。

**为什么重要**: 
- **参数效率**: 只需训练 0.1-1% 的参数，显存和计算大幅减少
- **可合并**: 推理时可以将 $BA$ 合并回 $W_0$，无额外延迟
- **多任务**: 可以为不同任务训练不同的 LoRA，灵活切换

**如何实现**:
- **数学**: $W = W_0 + \alpha \cdot BA$，其中 $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$，$r \ll d$
- **初始化**: $A$ 高斯初始化，$B$ 初始化为 0（初始状态 $\Delta W = 0$）
- **推理**: 可以合并 $W_{merged} = W_0 + \alpha \cdot BA$

**实际例子**: 
- 7B 模型全量微调需要 112GB 显存，LoRA 只需 5GB
- OpenVLA 使用 LoRA 微调，可以在消费级显卡上训练

**详细解答**: [peft_lora.md](../theory/peft_lora.md#4-peft-方法对比-comparison)

---

#### B2: RLHF 流程？与 DPO 差异？

**核心概念**: RLHF（Reinforcement Learning from Human Feedback）通过人类偏好数据训练奖励模型，再用 PPO 优化策略；DPO 跳过奖励模型，直接从偏好数据优化。

**为什么重要**: 
- **对齐问题**: 如何让模型输出符合人类价值观（如"安全"、"有用"）
- **DPO 优势**: 更简单、更稳定，无需训练奖励模型

**如何实现**:
- **RLHF 三阶段**:
  1. SFT（Supervised Fine-tuning）：在高质量数据上微调
  2. Reward Model Training：从人类偏好数据训练奖励模型
  3. PPO：使用 PPO 最大化奖励模型输出
- **DPO**: 直接优化偏好损失：$\mathcal{L} = -\log \sigma(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)})$

**实际例子**: 
- ChatGPT 使用 RLHF 对齐，需要 4 个模型（SFT、Reward、Actor、Critic）
- DPO 只需 2 个模型（Actor、Reference），更易部署

**详细解答**: [reinforcement_learning.md](../theory/reinforcement_learning.md#61-rlhf-完整流程-rlhf-pipeline)

---

#### B3: TP/PP/DP 分别是什么？

**核心概念**: 三种分布式训练策略：DP（数据并行）、TP（张量并行）、PP（流水线并行）。

**为什么重要**: 
- **模型规模**: 70B+ 模型无法单卡训练，必须分布式
- **通信效率**: 不同策略的通信开销不同，需要权衡

**如何实现**:
- **DP (Data Parallel)**: 切数据，每卡一份模型副本，All-Reduce 梯度
- **TP (Tensor Parallel)**: 切矩阵，层内切分（如 Attention 的 Q/K/V），All-Reduce 激活
- **PP (Pipeline Parallel)**: 切层，层间切分，点对点传输

**实际例子**: 
- 7B 模型：单机 8 卡用 FSDP（ZeRO-3）足够
- 70B 模型：TP=8（节点内）+ PP=N（跨节点）+ DP

**详细解答**: [large_scale_training.md](../system-design/large_scale_training.md#q0-分布式训练中的-tpppDP-分别是什么)

---

#### B4: Flash Attention 原理？

**核心概念**: Flash Attention 通过 Tiling（分块）、Kernel Fusion（算子融合）和 Online Softmax（在线归一化）将内存从 $O(N^2)$ 降到 $O(N)$。

**为什么重要**: 
- **内存瓶颈**: 标准 Attention 需要存储 $N \times N$ 矩阵，长序列会 OOM
- **速度提升**: 减少 HBM 读写，速度提升 2-4x

**如何实现**:
- **Tiling**: 分块计算，避免存储完整矩阵
- **Kernel Fusion**: 将 QK^T → softmax → ×V 融合进单个 CUDA Kernel
- **Online Softmax**: 增量更新归一化，支持分块计算

**实际例子**: 
- ViT 序列长度 196，标准 Attention 需要 38K 个浮点数
- Flash Attention 只需存储最终输出，内存减少 100x

**详细解答**: [flash_attention.md](../theory/flash_attention.md#6-面试常见问题)

---

#### B5: KV-Cache 如何加速推理？

**核心概念**: KV-Cache 缓存已计算的 K/V，新 Token 只需计算增量，将每 Token 计算从 $O(N^2)$ 降到 $O(N)$。

**为什么重要**: 
- **自回归生成**: 每个 Token 都要重算历史 K/V，计算冗余
- **加速效果**: N=1000 时约 1000x 加速

**如何实现**:
- **无 Cache**: 每步重算 $QK^T$，复杂度 $O(N^2 d)$
- **有 Cache**: 缓存 $K_{1:t}, V_{1:t}$，新 Token 只算 $Q_t K_{1:t}^T$，复杂度 $O(Nd)$

**实际例子**: 
- GPT 生成 1000 Token：无 Cache 需要 1000 次 $O(N^2)$ 计算
- 有 Cache：只需 1000 次 $O(N)$ 计算，加速 1000x

**详细解答**: [flash_attention.md](../theory/flash_attention.md#5-kv-cache-推理加速-kv-cache-for-inference)

---

### 📖 进阶题 (Advanced Questions)

#### B6: 知识蒸馏中温度参数 T 的作用？

**核心概念**: 温度参数 $T$ 控制 Softmax 分布的平滑程度，$T>1$ 时分布更平滑，保留更多"暗知识"。

**为什么重要**: 
- **暗知识**: 错误类别的概率也包含信息（如"猫"和"狗"比"猫"和"车"更像）
- **泛化能力**: 软标签比硬标签提供更丰富的监督信号

**如何实现**:
- **Softmax with Temperature**: $p_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$
- $T=1$: 正常 Softmax
- $T>1$: 分布更平滑，暗知识更明显

**实际例子**: 
- 教师模型输出: `[0.7, 0.2, 0.1]`（猫、狗、车）
- $T=1$: 学生学到"猫是 0.7"
- $T=5$: 学生学到"猫是 0.7，但狗也有 0.2，说明猫和狗相似"

**详细解答**: [knowledge_distillation.md](../theory/knowledge_distillation.md#22-软标签)

---

#### B7: 知识蒸馏的软标签为什么比硬标签好？

**核心概念**: 软标签包含类间关系信息，提供更丰富的监督信号，学生模型学到更泛化的特征。

**为什么重要**: 
- **类间关系**: "猫"和"狗"的概率都高，说明它们相似
- **泛化能力**: 学生模型学到更鲁棒的特征表示

**如何实现**:
- **硬标签**: `[1, 0, 0]`（one-hot）
- **软标签**: `[0.7, 0.2, 0.1]`（教师模型的输出）

**实际例子**: 
- 硬标签只告诉学生"这是猫"
- 软标签告诉学生"这是猫，但和狗也有相似之处"，学到更泛化的特征

**详细解答**: [knowledge_distillation.md](../theory/knowledge_distillation.md)

---

#### B8: 对比学习 InfoNCE 损失的直觉？

**核心概念**: InfoNCE 通过拉近正样本对、推远负样本对来学习表示，最大化正样本在分母中的比例。

**为什么重要**: 
- **自监督学习**: 无需人工标注，从数据中学习表示
- **表示质量**: 学到的特征对下游任务（如分类、检测）很有用

**如何实现**:
- **公式**: $\mathcal{L} = -\log \frac{\exp(\text{sim}(z_i, z_j^+)/\tau)}{\sum_k \exp(\text{sim}(z_i, z_k)/\tau)}$
- **分子**: 正样本对相似度
- **分母**: 所有样本对相似度之和
- **最大化**: 正样本比例 = 拉近正样本，推远负样本

**实际例子**: 
- SimCLR：同一张图的两个数据增强视图是正样本对
- 模型学习将同一物体的不同视图映射到相近的表示

**详细解答**: [self_supervised_learning.md](../theory/self_supervised_learning.md#211-infonce-损失)

---

#### B9: MAE 和 SimCLR 的核心区别？

**核心概念**: MAE 是掩码重建任务，SimCLR 是对比学习任务，两者都是自监督学习方法。

**为什么重要**: 
- **不同范式**: MAE 是生成式，SimCLR 是对比式
- **适用场景**: MAE 更高效，SimCLR 需要大 Batch

**如何实现**:
- **MAE**: 掩码 75% 的 Patch，重建像素，使用 MSE Loss
- **SimCLR**: 同图两视图，对比学习，使用 InfoNCE Loss

**实际例子**: 
- MAE：掩码大部分图像，模型学习重建，学到语义特征
- SimCLR：同一张图的两个视图，模型学习相似表示

**详细解答**: [self_supervised_learning.md](../theory/self_supervised_learning.md)

---

#### B10: Co-training 为什么能防止灾难性遗忘？

**核心概念**: Co-training 混合 Web 数据和机器人数据训练，Web 数据保持 VLM 的通用语义知识，防止遗忘。

**为什么重要**: 
- **灾难性遗忘**: 纯机器人数据是 Narrow Domain，容易遗忘预训练的 Wide Domain 知识
- **语义理解**: VLA 需要理解"泰勒斯威夫特是谁"，这需要 Web 数据

**如何实现**:
- **混合训练**: Web 数据（VQA）+ 机器人数据（Action）
- **Loss Masking**: 机器人数据算 Action Loss，Web 数据算 Text Loss
- **采样权重**: 可以给机器人数据更高权重

**实际例子**: 
- RT-2 Co-fine-tuning：混合 Web VQA 和机器人数据
- 模型既能理解"抓恐龙"（语义），又能输出动作（控制）

**详细解答**: [co_training.md](../theory/co_training.md#11-防止灾难性遗忘)

---

#### B11: Co-training 的 Loss Masking 如何实现？

**核心概念**: Loss Masking 根据数据类型选择不同的 Loss，机器人数据算 Action Loss，Web 数据算 Text Loss。

**为什么重要**: 
- **任务不同**: 机器人数据需要学习动作，Web 数据需要学习语言理解
- **互不干扰**: 各学各的，不会互相干扰

**如何实现**:
- **机器人数据**: `loss = ActionHeadLoss(pred_action, gt_action)`
- **Web 数据**: `loss = TextTokenLoss(pred_text, gt_text)`
- **混合**: `total_loss = \lambda_1 \cdot action_loss + \lambda_2 \cdot text_loss`

**实际例子**: 
- RT-2：机器人数据学习动作 Token，Web 数据学习 VQA
- 两者共享 Transformer Backbone，但 Loss 分开计算

**详细解答**: [co_training.md](../theory/co_training.md#22-loss-计算)

---

#### B12: Sim-to-Real 的 Domain Randomization 原理？

**核心概念**: Domain Randomization 在仿真中随机化各种参数（光照、纹理、质量），让真机成为随机分布中的一种。

**为什么重要**: 
- **Sim-to-Real Gap**: 仿真和真机在视觉和动力学上存在差异
- **鲁棒性**: 模型学会对这些变化鲁棒，泛化到真机

**如何实现**:
- **随机化参数**: 光照强度、纹理、物体质量、摩擦系数等
- **训练**: 模型在随机化的仿真中训练
- **推理**: 真机成为分布中的一种，模型自然适应

**实际例子**: 
- 抓取任务：随机化光照、物体颜色、桌面纹理
- 模型学会对这些变化鲁棒，迁移到真机时效果更好

**详细解答**: [transfer_learning.md](../theory/transfer_learning.md)

---

#### B13: 什么时候用 Feature Extraction vs Fine-tuning？

**核心概念**: Feature Extraction 冻结 Backbone 只训练 Head，Fine-tuning 全量或部分微调。

**为什么重要**: 
- **数据量**: 数据少时冻结防止过拟合，数据多时微调效果更好
- **计算资源**: Feature Extraction 更省显存和计算

**如何实现**:
- **Feature Extraction**: 冻结 Backbone，只训练 Action Head
- **Fine-tuning**: 全量微调或 LoRA 微调

**实际例子**: 
- 数据 <50：Feature Extraction（防止过拟合）
- 数据 >500：Fine-tuning（效果更好）

**详细解答**: [transfer_learning.md](../theory/transfer_learning.md#22-冻结特征提取)

---

#### B14: 量化中 Per-Tensor vs Per-Channel 的区别？

**核心概念**: Per-Tensor 整个张量共享 scale/zero，Per-Channel 每个通道独立量化。

**为什么重要**: 
- **精度**: Per-Channel 精度更高，但存储开销大
- **硬件支持**: 某些硬件只支持 Per-Tensor

**如何实现**:
- **Per-Tensor**: $W_{quant} = \text{round}(W / s) + z$，$s, z$ 整个张量共享
- **Per-Channel**: 每个通道有独立的 $s_i, z_i$

**实际例子**: 
- 7B 模型 INT8 量化：Per-Tensor 精度下降 2%，Per-Channel 精度下降 0.5%
- 但 Per-Channel 需要存储更多 scale/zero

**详细解答**: [quantization_theory.md](../theory/quantization_theory.md)

---

#### B15: AWQ 如何找到重要权重？

**核心概念**: AWQ 观察激活值分布，激活值大的通道更重要，对这些通道保持高精度。

**为什么重要**: 
- **精度权衡**: 全量 INT8 精度下降，AWQ 对重要权重保持 FP16
- **硬件友好**: 混合精度，硬件支持良好

**如何实现**:
- **观察激活**: 统计每个通道的激活值分布
- **选择重要通道**: 激活值大的通道更重要
- **混合量化**: 重要通道 FP16，其他通道 INT8

**实际例子**: 
- 7B 模型：AWQ 对 10% 的通道保持 FP16，其他 INT8
- 精度接近 FP16，速度接近 INT8

**详细解答**: [quantization_theory.md](../theory/quantization_theory.md)

---

#### B16: 只有 100 条真机演示数据，如何训练一个鲁棒的抓取策略？

**核心概念**: 小样本训练需要数据增强、Sim-to-Real、Co-training 和使用预训练模型。

**为什么重要**: 
- **数据稀缺**: 真机数据收集昂贵，需要高效利用
- **泛化能力**: 小数据容易过拟合，需要正则化

**如何实现**:
- **数据增强**: 旋转、裁剪、颜色抖动
- **Sim-to-Real**: 先在仿真中训练，再用真机数据 Fine-tuning
- **Co-training**: 混合开源数据集（如 OXE），100 条数据更高权重
- **预训练模型**: 基于 OpenVLA 或 RT-1 预训练权重微调（LoRA）

**实际例子**: 
- 100 条抓取数据：先用 OXE 数据集 Co-training，再用 100 条数据 Fine-tuning
- 效果接近全量数据训练

**详细解答**: [transfer_learning.md](../theory/transfer_learning.md)

---

## 🏗️ 类别 C: 模型架构与推理 (Architecture & Reasoning)

### 🔥 高频题 (Top Questions)

#### C1: 视觉误判如何语言纠错？

**核心概念**: 通过闭环反馈、CoT 自检、多模态一致性检查和主动询问来纠正视觉误判。

**为什么重要**: 
- **鲁棒性**: 视觉模块可能误判，需要语言反馈纠正
- **人机交互**: 支持多轮对话，提高成功率

**如何实现**:
- **闭环反馈**: 用户语言指令纠正（"不对，是左边那个"）
- **CoT 自检**: 输出推理链，发现矛盾
- **多模态一致性**: 计算语言-视觉 Embedding 相似度
- **主动询问**: 低置信度时请求确认

**实际例子**: 
- 场景：视觉误判"红色杯子"为"橙色杯子"
- 用户："不对，是红色的那个"
- VLA：重新定位 → 修正目标

**详细解答**: [multimodal_models.md](../theory/multimodal_models.md#q6-如果视觉模块误判如何通过语言纠错)

---

#### C2: 如何构建 Evaluation Pipeline？

**核心概念**: Evaluation Pipeline 包括数据管理、推理服务、指标计算、可视化和 CI/CD 集成。

**为什么重要**: 
- **模型选择**: 训练过程中需要评估多个 Checkpoint
- **持续改进**: 自动化评估流程，提高效率

**如何实现**:
- **数据**: CALVIN/SIMPLER 标准测试集
- **推理**: 多 Checkpoint 并行评估
- **指标**: SR/MSS/IR + Wilson 置信区间
- **日志**: W&B/TensorBoard + 失败案例分析
- **CI/CD**: 训练后自动触发评估

**实际例子**: 
- 训练 100 个 Epoch，每个 Epoch 后自动评估
- 选择 SR 最高的 Checkpoint 部署

**详细解答**: [evaluation.md](../theory/evaluation.md#5-evaluation-pipeline-构建-building-evaluation-pipeline)

---

#### C3: Model-Based vs Model-Free 区别？

**核心概念**: Model-Free 直接学习策略或价值函数，Model-Based 先学习环境模型再规划。

**为什么重要**: 
- **样本效率**: Model-Based 更高效，但需要准确的模型
- **适用场景**: Model-Free 简单但需要大量交互

**如何实现**:
- **Model-Free**: 直接学习 $\pi(a|s)$ 或 $Q(s,a)$
- **Model-Based**: 先学习 $P(s'|s,a)$，再利用模型规划

**实际例子**: 
- Model-Free：PPO、SAC，需要大量交互
- Model-Based：MCTS、MPC，样本效率高但需要模型

**详细解答**: [reinforcement_learning.md](../theory/reinforcement_learning.md#31-model-free-vs-model-based)

---

#### C4: 马尔可夫性是什么？

**核心概念**: 马尔可夫性指下一状态只依赖于当前状态和动作，与历史无关。

**为什么重要**: 
- **Bellman 方程**: 马尔可夫性是 Bellman 方程成立的前提
- **简化计算**: 不需要记忆整个历史，只需当前状态

**如何实现**:
- **定义**: $P(s_{t+1}|s_t, a_t, s_{t-1}, ...) = P(s_{t+1}|s_t, a_t)$
- **充分统计量**: 当前状态是对历史的"充分统计量"

**实际例子**: 
- 机器人状态包含位置+速度，满足马尔可夫性
- 若只有位置，则不满足（需要历史速度信息）

**详细解答**: [reinforcement_learning.md](../theory/reinforcement_learning.md#22-马尔可夫性-markov-property)

---

#### C5: 为什么最优价值函数就是最优策略？

**核心概念**: 给定最优价值函数 $Q^*$，最优策略可以直接导出：$\pi^*(s) = \arg\max_a Q^*(s,a)$。

**为什么重要**: 
- **理论保证**: 最优价值函数和最优策略是"一体两面"
- **算法设计**: 值迭代和策略迭代都基于这个原理

**如何实现**:
- **核心定理**: $\pi^*(s) = \arg\max_a Q^*(s,a)$
- **原因**: $Q^*(s,a)$ 表示执行 $a$ 后按最优策略行动的期望回报
- **结论**: 对 $Q^*$ 贪心就是最优的

**实际例子**: 
- Q-Learning：学习 $Q^*$，然后贪心选择动作
- 这就是最优策略

**详细解答**: [reinforcement_learning.md](../theory/reinforcement_learning.md#25-最优价值函数与最优策略)

---

#### C6: 策略迭代 vs 值迭代？

**核心概念**: 策略迭代先评估再改进，值迭代直接迭代 Bellman 最优方程。

**为什么重要**: 
- **算法选择**: 不同场景适合不同算法
- **效率权衡**: 策略迭代每轮迭代少但每轮计算多，值迭代相反

**如何实现**:
- **策略迭代**: 评估（计算 $V^\pi$）+ 改进（贪心更新 $\pi$）交替进行
- **值迭代**: 直接迭代 $V(s) \leftarrow \max_a [R + \gamma \sum P V']$

**实际例子**: 
- 小状态空间：策略迭代更快
- 大状态空间：值迭代更实用

**详细解答**: [reinforcement_learning.md](../theory/reinforcement_learning.md#32-策略迭代-vs-值迭代-policy-iteration-vs-value-iteration)

---

### 📖 进阶题 (Advanced Questions)

#### C7: VLA 中 Early/Mid/Late Fusion 如何选择？

**核心概念**: Early Fusion 在输入层融合，Mid Fusion 在中间层融合（Cross-Attention），Late Fusion 在输出层融合。

**为什么重要**: 
- **融合时机**: 不同时机适合不同任务
- **VLA 首选**: Mid Fusion（Cross-Attention）是 VLA 的主流选择

**如何实现**:
- **Early Fusion**: 模态相似（如多相机图像）
- **Mid Fusion**: Cross-Attention，动态建模模态间关系
- **Late Fusion**: 各模态任务独立，需要模块化解释性

**实际例子**: 
- RT-2 / OpenVLA：Mid Fusion（Cross-Attention）
- 视觉 Token 和语言 Token 可以互相关注

**详细解答**: [multimodal_models.md](../theory/multimodal_models.md#q3-多模态融合中-early--mid--late-fusion-如何选择)

---

#### C8: FiLM 调制是什么？在 VLA 中如何应用？

**核心概念**: FiLM（Feature-wise Linear Modulation）通过 $\gamma \cdot \text{feature} + \beta$ 注入条件信息。

**为什么重要**: 
- **条件注入**: 将语言/时间步等条件信息注入视觉特征
- **轻量高效**: 比 Cross-Attention 更轻量

**如何实现**:
- **公式**: $\text{output} = \gamma \cdot \text{feature} + \beta$
- **生成**: $\gamma, \beta$ 由条件信息（语言/时间步）生成
- **应用**: RT-1 用语言调制视觉，Diffusion 用时间步调制

**实际例子**: 
- RT-1：语言指令生成 $\gamma, \beta$，调制视觉特征
- Diffusion Policy：时间步生成 $\gamma, \beta$，调制动作特征

**详细解答**: [multimodal_models.md](../theory/multimodal_models.md#34-vla-中的主流方案film-调制)

---

#### C9: SigLIP 比 CLIP 好在哪？

**核心概念**: SigLIP 使用 Sigmoid 替代 Softmax，无需全局同步，Batch 独立，更适合大规模训练。

**为什么重要**: 
- **分布式训练**: SigLIP 无需全局同步，通信开销小
- **Batch 大小**: 对 Batch Size 不敏感，CLIP 需要大 Batch

**如何实现**:
- **CLIP**: Softmax + Cross-Entropy，需要全局归一化
- **SigLIP**: Sigmoid + Binary CE，每对独立计算

**实际例子**: 
- CLIP：需要 All-Reduce 同步所有 GPU 的负样本
- SigLIP：每对独立计算，无需同步

**详细解答**: [transformer_vs_cnn.md](../theory/transformer_vs_cnn.md#52-siglip)

---

#### C10: CoT 在 VLA 中有什么价值？

**核心概念**: Chain-of-Thought（思维链）让模型输出推理步骤，提高可解释性、任务分解和错误纠正能力。

**为什么重要**: 
- **可解释性**: 知道机器人"在想什么"
- **复杂任务**: 将复杂任务分解为子步骤
- **错误纠正**: 推理中发现矛盾，自我纠正

**如何实现**:
- **显式 CoT**: 输出语言推理步骤（"我看到一个红色物体..."）
- **隐式 CoT**: 在 Latent 空间推理（π0.5）

**实际例子**: 
- RT-2：输出推理步骤 "我看到一个红色杯子，用户要求抓取，我将执行抓取动作"
- 提高可解释性和成功率

**详细解答**: [chain_of_thought.md](../theory/chain_of_thought.md#12-cot-的价值)

---

#### C11: 显式 CoT 和隐式 CoT 的区别？

**核心概念**: 显式 CoT 输出语言推理步骤，隐式 CoT 在 Latent 空间推理。

**为什么重要**: 
- **速度**: 隐式 CoT 更快（无需生成文本）
- **可解释性**: 显式 CoT 更易理解

**如何实现**:
- **显式 CoT**: 模型输出 "Step 1: ... Step 2: ..."
- **隐式 CoT**: 在 Transformer 的 Hidden States 中推理

**实际例子**: 
- RT-2：显式 CoT，输出语言推理
- π0.5：隐式 CoT，在 Latent 空间推理，更快

**详细解答**: [chain_of_thought.md](../theory/chain_of_thought.md)

---

#### C12: RT-2 的 Co-fine-tuning 为什么重要？

**核心概念**: RT-2 混合 Web VQA 数据和机器人数据训练，保持语义理解能力的同时学习动作控制。

**为什么重要**: 
- **防止遗忘**: 纯机器人数据会遗忘预训练的语义知识
- **Zero-shot**: 能够理解未见过的指令（如"抓恐龙"）

**如何实现**:
- **混合训练**: Web VQA + Robot 数据
- **动作编码**: 将动作编码为文本 Token，与语言 Token 共享词表

**实际例子**: 
- RT-2 能理解 "pick up the extinct animal" → 抓恐龙玩具
- 这得益于 Web 数据的语义理解能力

**详细解答**: [vla_arch.md](../theory/vla_arch.md#2-rt-2)

---

#### C13: OpenVLA 的 Action Head 是如何设计的？

**核心概念**: OpenVLA 使用专门的 Linear Layer 预测去离散化的动作 Token，通过 Detokenization 还原为连续值。

**为什么重要**: 
- **精度**: 相比文本 Token，专门的 Action Head 精度更高
- **效率**: 比 Diffusion Policy 更快

**如何实现**:
- **Action Head**: Linear Layer，输出离散化动作 Token
- **Detokenization**: 将 Token 还原为连续动作值

**实际例子**: 
- OpenVLA：Action Head 输出 7 个 Token（x, y, z, roll, pitch, yaw, gripper）
- 每个 Token 对应 256 个 bin，Detokenization 还原为连续值

**详细解答**: [vla_arch.md](../theory/vla_arch.md#openvla)

---

#### C14: π0.5 的分层推理架构是什么？

**核心概念**: π0.5 使用高层 VLM 规划 + 低层 Flow Matching 执行的分层架构。

**为什么重要**: 
- **任务分解**: 高层规划复杂任务，低层执行精细动作
- **效率**: 类似人类"想清楚再动手"

**如何实现**:
- **高层**: VLM 生成计划/子任务（"先抓杯子，再倒水"）
- **低层**: Flow Matching 生成动作序列

**实际例子**: 
- π0.5：高层输出 "Step 1: 抓取杯子, Step 2: 移动到水壶上方"
- 低层根据每个子任务生成动作序列

**详细解答**: [pi0_5_dissection.md](../theory/pi0_5_dissection.md)

---

#### C15: WALL-OSS 的 Uni-CoT 有什么特点？

**核心概念**: WALL-OSS 的 Uni-CoT（统一思维链）实现视觉→语言→动作的跨模态思维链。

**为什么重要**: 
- **统一推理**: 视觉、语言、动作全程推理，提高一致性
- **可解释性**: 知道模型在每个阶段的推理过程

**如何实现**:
- **双输出头**: 语言推理头 + 动作生成头
- **流匹配控制**: 使用 Flow Matching 生成动作

**实际例子**: 
- WALL-OSS：视觉输入 → 语言推理 "我看到一个红色杯子" → 动作生成 "抓取动作"

**详细解答**: [wall_oss.md](../theory/wall_oss.md)

---

#### C16: Galaxea G0 的"大脑+小脑"架构是什么？

**核心概念**: Galaxea G0 使用 VLM 大脑（高层规划）+ VLA 小脑（低层执行）的分层架构。

**为什么重要**: 
- **分工明确**: 大脑负责规划，小脑负责执行
- **课程学习**: 三阶段课程学习，逐步提升能力

**如何实现**:
- **VLM 大脑**: 高层规划，生成任务计划
- **VLA 小脑**: 低层执行，生成动作序列

**实际例子**: 
- Galaxea G0：大脑规划 "倒水任务"，小脑执行 "抓杯子→移动到水壶→倒水"

**详细解答**: [galaxea_g0.md](../theory/galaxea_g0.md)

---

#### C17: Transformer vs CNN 在 VLA 中的选择：什么时候用 ViT，什么时候用 ResNet？

**核心概念**: ViT 适合多模态统一和全局上下文，ResNet 适合小样本和纹理特征。

**为什么重要**: 
- **架构选择**: 不同场景适合不同架构
- **性能权衡**: ViT 效果更好但需要大数据，ResNet 小数据友好

**如何实现**:
- **ViT**: 多模态统一、全局上下文、Scaling Law
- **ResNet**: 小样本、纹理特征、推理速度

**实际例子**: 
- RT-2 / OpenVLA：ViT（多模态统一）
- RT-1：ResNet（小数据友好）

**详细解答**: [transformer_vs_cnn.md](../theory/transformer_vs_cnn.md)

---

#### C18: 什么是触觉 VLA (Tactile VLA)？为什么视觉不够？

**核心概念**: Tactile VLA 融合触觉传感器，感知接触、材质、滑移等视觉无法获取的信息。

**为什么重要**: 
- **遮挡问题**: 抓取时手掌会挡住摄像头
- **物理属性**: 视觉无法判断软硬、摩擦力
- **精度**: 触觉传感器提供微米级控制

**如何实现**:
- **VLA-Touch**: 双层反馈机制（High-level TLM + Low-level FiLM）
- **OmniVTLA**: 统一 Tokenization，触觉-视觉-语言统一建模

**实际例子**: 
- 抓取透明物体：视觉可能看不准，触觉传感器能感知接触
- 判断材质：视觉无法判断软硬，触觉传感器能感知

**详细解答**: [tactile_vla.md](../theory/tactile_vla.md)

---

#### C19: VLA-Touch 和 OmniVTLA 有什么区别？

**核心概念**: VLA-Touch 使用双层反馈机制，OmniVTLA 使用统一 Tokenization。

**为什么重要**: 
- **架构选择**: 不同架构适合不同场景
- **集成方式**: VLA-Touch 即插即用，OmniVTLA 需要重训

**如何实现**:
- **VLA-Touch**: TLM 翻译触觉为语言 + FiLM 注入触觉特征
- **OmniVTLA**: 触觉-视觉-语言统一 Tokenization，InfoNCE 对齐

**实际例子**: 
- VLA-Touch：无需重训整个 VLA，即插即用
- OmniVTLA：能执行跨模态推理（"Pick up the softest object"）

**详细解答**: [tactile_vla.md](../theory/tactile_vla.md)

---

#### C20: 机器人抓取透明物体总是失败，怎么办？

**核心概念**: 透明物体抓取失败的原因包括深度失效、反射干扰和材质判断困难，需要多传感器融合。

**为什么重要**: 
- **实际需求**: 透明物体在家庭和工业场景中常见
- **技术挑战**: 纯视觉方法难以处理

**如何实现**:
- **传感器**: 立体视觉、深度补全、触觉传感器
- **算法**: 多视角融合、触觉反馈、数据增强

**实际例子**: 
- RGB-D 相机：深度光在透明物体上失效
- 解决方案：使用触觉传感器，摸到了就知道位置

**详细解答**: [perception_techniques.md](../theory/perception_techniques.md)

---

### 💻 代码题 (Coding Questions)

#### C21: 实现一个简单的 PID 控制器

**核心概念**: PID 控制器通过比例（P）、积分（I）、微分（D）三项组合来控制系统输出。

**为什么重要**: 
- **基础控制**: PID 是机器人控制的基础
- **面试常见**: 考察对控制理论的理解

**如何实现**:
```python
class PIDController:
    def __init__(self, kp, ki, kd, dt):
        self.kp = kp      # 比例系数
        self.ki = ki      # 积分系数
        self.kd = kd      # 微分系数
        self.dt = dt      # 时间步长
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

**实际例子**: 
- 机械臂位置控制：目标位置 vs 当前位置
- PID 输出控制信号，驱动电机

---

#### C22: 计算两个 3D 点之间的欧氏距离

**核心概念**: 欧氏距离是两点之间的直线距离，使用 L2 范数计算。

**为什么重要**: 
- **基础操作**: 机器人学中经常需要计算距离
- **空间关系**: 判断物体是否在抓取范围内

**如何实现**:
```python
import numpy as np

def distance(p1, p2):
    # p1, p2: np.array [x, y, z]
    return np.linalg.norm(p1 - p2)
```

**实际例子**: 
- 抓取任务：计算夹爪到物体的距离
- 如果距离 < 5cm，执行抓取动作

---

#### C23: 旋转矩阵转欧拉角（概念）

**核心概念**: 旋转矩阵可以转换为欧拉角，但需要注意万向节死锁问题。

**为什么重要**: 
- **表示转换**: 不同场景需要不同的旋转表示
- **万向节死锁**: 欧拉角的致命缺陷

**如何实现**:
- 使用 `scipy.spatial.transform.Rotation` 进行转换
- 注意万向节死锁：当中间轴旋转 90 度时，第一轴和第三轴重合

**实际例子**: 
- 机器人位姿：通常用旋转矩阵表示
- 显示给用户：转换为欧拉角（更直观）

---

## 🧪 代码实战突击 (30 题+解答)

面向 VLA 具身算法岗常考的代码/工程题，按主题分组。每题给出考察点与解答要点，便于速练。

### Python 基础 (Q1-Q5)

#### Q1: 实现 LRU Cache（O(1) get/put）
- **考察点**: 哈希 + 双向链表、边界处理。
- **解答要点**: 用 `OrderedDict` 或自建双向链表；get/put 访问需移动到表头；容量超限删除尾节点；注意空缓存、重复 key。

#### Q2: 流式读取 1GB 日志统计字段频次
- **考察点**: IO/内存控制、生成器。
- **解答要点**: `with open(..., buffering=...)` + 行迭代；按列解析计数；避免一次性 load；可用 `collections.Counter.update`；提供 top-k 输出。

#### Q3: 变长序列的滑动窗口生成器
- **考察点**: 迭代器协议、边界。
- **解答要点**: `yield seq[i:i+win]`，窗口不足时提前退出；可选 stride；测试空序列/窗口大于长度。

#### Q4: 有限状态机解析指令流（含错误处理）
- **考察点**: 状态设计、异常。
- **解答要点**: 用 `Enum` 标状态，dict 映射转移；非法转移抛自定义异常；附最小单测覆盖正常/异常路径。

#### Q5: CLI 扫描大文件并确认删除
- **考察点**: `argparse`、安全性。
- **解答要点**: 参数含目录、阈值、dry-run；用 `pathlib.rglob` 过滤；输出大小排序；删除前交互确认。

### PyTorch 训练 (Q6-Q12)

#### Q6: 自定义 Dataset + collate_fn 处理变长序列
- **考察点**: 数据管线、padding。
- **解答要点**: `__getitem__` 返回原始序列；collate 里计算 max_len，`pad_sequence`；返回 lengths 供 RNN/Transformer mask。

#### Q7: 训练循环含 AMP、梯度裁剪、LR 调度
- **考察点**: 训练细节。
- **解答要点**: `autocast` + `GradScaler`；`clip_grad_norm_`；scheduler.step() 时机（常在 optimizer.step 后）；记录 loss/grad norm。

#### Q8: checkpoint 保存/恢复
- **考察点**: 状态管理。
- **解答要点**: 保存 `model/optimizer/scheduler/scaler/state_dict` 与 epoch/step；恢复后 `model.train()`；处理缺失文件与设备映射。

#### Q9: DDP 最小示例
- **考察点**: 多卡并行。
- **解答要点**: `torch.distributed.init_process_group`；`DistributedSampler`；`DDP(model, device_ids=[rank])`；设置随机种子确保可复现。

#### Q10: 自定义 Autograd Function (可微 clamp)
- **考察点**: 前向/反向。
- **解答要点**: `ctx.save_for_backward`；反向对区间外梯度截断；用数值梯度检验 `gradcheck`。

#### Q11: 推理性能分析与优化
- **考察点**: profiler 与优化手段。
- **解答要点**: `torch.profiler.profile` 找热点；优化：避免重复 `.to()`、改用 `torch.compile`/JIT、合并小张量操作、使用 `pin_memory`/`non_blocking=True`。

#### Q12: 简单权重衰减与梯度分组
- **考察点**: 优化器配置。
- **解答要点**: 将 `bias/LayerNorm` 排除 weight decay；param_groups 配置；验证参数量与学习率正确下发。

### Git 协作 (Q13-Q16)

#### Q13: 清理脏工作树并同步远端
- **考察点**: 安全同步流程。
- **解答要点**: `git status`→`git stash push -u`→`git fetch`→`git rebase origin/main` 或新分支→`git stash pop` 解决冲突→提交。

#### Q14: `.gitignore` 设计避免泄漏
- **考察点**: 配置与安全。
- **解答要点**: 忽略 `.pt/.ckpt/.h5`、数据目录、`__pycache__`、`*.so`、日志；说明理由（体积/隐私/可重建）。

#### Q15: `rebase -i` 压缩/改写提交
- **考察点**: 历史整理。
- **解答要点**: `git rebase -i HEAD~N` 选择 `squash/fixup/edit`；冲突用 `git status` 定位，解决后 `git rebase --continue`；必要时 `--abort`。

#### Q16: pre-commit 钩子示例
- **考察点**: 自动化检查。
- **解答要点**: `.pre-commit-config.yaml` 启动 `black/ruff/mypy`；安装 `pre-commit install`；提交时阻止未格式化代码。

### SLAM / 视觉里程计 (Q17-Q23)

#### Q17: 特征匹配 + RANSAC 估计单应/本质矩阵
- **考察点**: 前端估计。
- **解答要点**: ORB/SIFT + `cv2.BFMatcher`；`findHomography` 或 `findEssentialMat` with RANSAC；输出内点掩码，剔除外点后姿态恢复。

#### Q18: PnP 位姿估计
- **考察点**: 3D-2D 求解。
- **解答要点**: `cv2.solvePnP` / `solvePnPRansac`；给内参、畸变；用 EPnP 初值；验证重投影误差。

#### Q19: 两帧视觉里程计 (EP & Pose Recovery)
- **考察点**: 几何推导。
- **解答要点**: 归一化相机坐标→`findEssentialMat`→`recoverPose`；尺度不可观，需 IMU/里程计融合。

#### Q20: 简易闭环检测
- **考察点**: Place Recognition。
- **解答要点**: ORB BoW/NetVLAD 向量；余弦相似度阈值 + temporal consistency；触发回环约束。

#### Q21: Pose Graph 优化
- **考察点**: 后端优化。
- **解答要点**: 节点=位姿，边=相对约束；误差项 `log(T_ij^-1 * T_i^-1 * T_j)`；用 G2O/Ceres 优化；鲁棒核抑制外点。

#### Q22: 关键帧策略
- **考察点**: 前端管理。
- **解答要点**: 新帧加入条件：视差阈值/跟踪内点数/时间间隔；淘汰：冗余、覆盖率低；维护共视图更新边。

#### Q23: 立体匹配 + 视差优化
- **考察点**: 立体几何。
- **解答要点**: 构造代价体（SAD/ Census），WTA 或 SGM；子像素插值；左右一致性检查 + 中值滤波。

### 运动控制 / 轨迹规划 (Q24-Q30)

#### Q24: 离散 PID（抗饱和）
- **考察点**: 控制基础。
- **解答要点**: `u = kp*e + ki*Σe*dt + kd*Δe/dt`；积分分离/限幅；死区与微分先行；仿真阶跃响应。

#### Q25: Pure Pursuit 跟踪
- **考察点**: 低速路径跟踪。
- **解答要点**: 选前视点，曲率 `2*y_l/Ld^2`；转向角 `delta = atan(L*curvature)`；前视距离随速度调节。

#### Q26: Stanley 控制器
- **考察点**: 横向控制。
- **解答要点**: 转向 = 航向误差 + `atan(k * crosstrack / v)`；低速抖动用增益限制或加前馈。

#### Q27: A* / Hybrid A* 轨迹规划
- **考察点**: 路径搜索。
- **解答要点**: 栅格 + 启发式（Manhattan/Euclid）；Hybrid A* 采样朝向和 kinematic step；记录父节点复原路径；碰撞检测膨胀障碍。

#### Q28: MPC (单轨模型 QP)
- **考察点**: 预测控制。
- **解答要点**: 线性化单轨模型，构建 `x_{k+1}=Ax+Bu`；代价包含跟踪误差与控制增量；用 OSQP/qpOASES 求解；软约束处理边界。

#### Q29: 碰撞检测与安全距离
- **考察点**: 几何判定。
- **解答要点**: 车体矩形→圆或多边形；障碍膨胀 `inflate_radius`; 用 SAT/圆距离检测路径可行性；提前退出提升性能。

#### Q30: 控制回路延迟/噪声仿真
- **考察点**: 鲁棒性。
- **解答要点**: 在仿真中注入时延/噪声；观察超调/震荡；改进：滤波、前馈、降低增益、加预测补偿。

---

## 📝 总结

本题库按三个核心类别组织，覆盖 **VLA Handbook** 的面试高频问题。每个问题都包含：
- **核心概念**：快速理解
- **为什么重要**：解决什么问题
- **如何实现**：技术细节
- **实际例子**：VLA 中的应用
- **详细解答**：深入学习的链接

建议按照类别系统学习，先掌握高频题，再深入进阶题。

---

[← Back to Question Bank](./README.md)
