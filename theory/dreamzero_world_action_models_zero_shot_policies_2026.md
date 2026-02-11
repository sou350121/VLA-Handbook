# DreamZero：世界动作模型即零样本策略（World Action Models are Zero-shot Policies）

> **来源（权威一手）**：DreamZero 官方项目页、论文 PDF；对比参照 DreamGen 官方页  
> - DreamZero Project: `https://dreamzero0.github.io/`  
> - DreamZero Paper (PDF): `https://dreamzero0.github.io/DreamZero.pdf`  
> - DreamGen (NVIDIA Research): `https://research.nvidia.com/labs/gear/dreamgen/`  
> **关键词**：World Action Model (WAM)、video diffusion、autoregressive DiT、flow matching、teacher forcing、KV cache、closed-loop、7Hz、cross-embodiment

---

## 0. 1 分钟版（能复述给面试官的版本）

- **DreamZero 的核心主张**：把“策略”改写成“世界动作模型（WAM）”——**联合预测未来视频（世界状态演化）+ 动作**，用视频作为“密集的世界动态表征”，让模型从**预训练视频扩散模型**继承物理/时序先验，从而在**新任务、新环境、甚至新本体**上实现更强泛化。
- **为什么它可能比 VLA 更泛化**：VLA 往往从静态 VLM 继承语义，但缺“怎么做”的时空动力学表征；WAM 通过视频预测直接学习“世界怎么变”，动作学习更像在做 **inverse dynamics**：给定（预测的）未来世界，倒推出该怎么动。
- **落地关键**：他们把一个 **14B autoregressive video diffusion** 模型做到了**闭环 7Hz**（约 150ms/动作 chunk），靠一整套算法/系统/内核级优化，论文称总体 **38×** 加速。

---

## 1. 核心架构/方法总览（WAM vs VLA）

### 1.1 WAM 的定义：把“视频世界模型”与“策略”绑成一个端到端分布

DreamZero 不是先“生成视频/计划”再“用单独的 IDM 或 policy 执行”，而是直接学习联合分布（论文式子 (1)）：

\[
\pi_0(o_{l:l+H}, a_{l:l+H} \mid o_{0:l}, c, q_l)
= \underbrace{\pi_0(o_{l:l+H} \mid o_{0:l}, c, q_l)}_{\text{video prediction}}
\cdot
\underbrace{\pi_0(a_{l:l+H} \mid o_{0:l+H}, q_l)}_{\text{IDM}}
\]

- \(o\)：视频观测（世界状态的可视化轨迹）
- \(a\)：动作序列
- \(c\)：语言指令
- \(q\)：本体状态（proprio）
- \(H\)：固定预测/执行 horizon

论文强调：虽然可以分解成“视频预测 + IDM”，但 DreamZero **用一个端到端模型**做联合建模，以获得更强的 video-action 对齐与泛化。

### 1.2 Figure 4 架构要点（输入编码 + AR DiT + 双解码头）

论文 Figure 4（架构描述）要点可以浓缩为：

- **三类输入**：
  - **视觉上下文**：视频帧先经 **VAE 编码**到 latent（降低扩散维度，保留视频先验）
  - **语言指令**：经 **text encoder**
  - **本体状态**：经 **state encoder**
- **主干**：**autoregressive DiT backbone**，训练目标用 **flow matching**
- **输出**：通过**分别的 decoder**联合预测：
  - **未来视频 latent（再可解码为帧）**
  - **未来动作（连续动作）**
- **多视角**：若训练数据有多视角，论文写法是**把多视角拼接到同一帧**，尽量不改 backbone 结构。

### 1.3 为什么选 autoregressive（而非双向/定长）？

论文给的动机非常工程化：

- **KV cache**：AR 架构推理可缓存历史，降低长上下文开销。
- **保留原始 FPS**：双向扩散常要求定长序列，容易做 subsampling，破坏 video-action 对齐；AR 可持续滚动，保持“原生帧率”更利于对齐动作。
- **闭环修正**：AR 视频生成的典型问题是误差累积，但 DreamZero 在机器人闭环里可以“作弊”——执行完一个 chunk 后，用**真实观测**覆盖 KV cache 里的预测帧（见 3.1）。

---

## 2. 数学核心（Math Core）：联合视频+动作的 Flow Matching + Chunk-wise Teacher Forcing

### 2.1 Chunk-wise（像训练 LLM 一样训练视频序列）

DreamZero 把视频按 chunk 切分：每个 chunk 有固定长度 \(K\)（对应动作 horizon），可在**变长轨迹**上训练（论文类比 LLM 变长 token）。

训练时用 **teacher forcing**：当前 noisy chunk 的去噪条件包含**前面 chunk 的干净（clean）上下文**。

### 2.2 Flow Matching 目标（论文式子 (2)(3) 的直觉版）

对第 \(k\) 个 chunk，采样 \(t_k \in [0,1]\)（同一 chunk 内共享同一 \(t_k\)），对 video latent \(z\) 与 action \(a\) 做线性插值噪化（论文式子 (2)）：

\[
z^{k}_{t_k} = t_k z^{k}_{1} + (1-t_k) z^{k}_{0}, \quad
a^{k}_{t_k} = t_k a^{k}_{1} + (1-t_k) a^{k}_{0}
\]

其中 \(z^{k}_{0}, a^{k}_{0} \sim \mathcal{N}(0, I)\)，而 \(z^{k}_{1}, a^{k}_{1}\) 是干净样本。

将前序 chunk 的 clean 上下文记作：
\[
\mathcal{C}_k = \{(z^{j}_{1}, a^{j}_{1})\}^{k-1}_{j=1}
\]

训练目标是让模型 \(u_\theta\) 预测 joint velocity（论文式子 (3)）：

\[
\mathcal{L}(\theta)
= \mathbb{E}\_{z,a,\{t_k\}}
\Bigg[
\frac{1}{K}\sum_{k=1}^{K} w(t_k)
\left\|
u_\theta\big([z^k_{t_k}, a^k_{t_k}]; \mathcal{C}_k, c, q_k, t_k\big) - v^k
\right\|^2
\Bigg]
\]

其中 \(v^k := [z^k_1, a^k_1] - [z^k_0, a^k_0]\)。

**直觉**：

- 把“未来世界怎么演化（视频）”和“该怎么动（动作）”放在同一个连续生成/去噪过程里对齐。
- teacher forcing 让模型学会在真实历史条件下生成下一段未来，而不是完全自由发散。

### 2.3 “IDM 分解”的解释（为什么联合建模会像在做 inverse dynamics）

论文把联合建模视作“视频预测 + IDM”的分解。你可以把它理解成：

- 视频预测部分像一个**隐式视觉规划器**：给出“如果按指令做对了，世界接下来应该长什么样”。
- 动作生成部分在联合训练下被迫与视频一致：等价于在学习“**要达到这个视觉未来，需要哪些动作**”，这就是 IDM 的直觉。

---

## 3. 推理与闭环执行：KV cache + 用真实观测替换预测帧 + 异步执行

### 3.1 关键技巧：闭环里“用真实观测覆盖预测”，消除 AR 误差累积

论文明确写到：推理时 DreamZero 依赖 KV cache（加速），并利用闭环特性：

- 每执行完一个 action chunk
- **用真实摄像头观测替换 KV cache 里对应的预测帧**

这让 AR 视频生成中常见的 compounding error（越滚越歪）被大幅缓解——这是“把世界模型当策略”在闭环控制里独有的优势点。

### 3.2 异步执行（Asynchronous closed-loop execution）

论文的“反直觉点”之一：要做到反应式控制，推理不必阻塞执行。

- 控制器持续执行“最新可用的 chunk”
- 推理在后台用最新观测并行生成下一 chunk
- 延迟约束从“必须在动作开始前完成推理”变成“必须在当前 chunk 过期前完成推理”

论文给的部署例子：控制频率 30Hz、chunk horizon 48 steps，则 chunk 时长约 1.6s；因此希望推理延迟 <~200ms，才能有充足重叠保证平滑响应。

---

## 4. 实时性与优化：从 5.7s/chunk 到 150ms/chunk（38×）

### 4.1 Reactivity Gap：为什么扩散策略天然慢？

论文提到 naive 的 DreamZero（单 GPU）需要约 **5.7 秒/动作 chunk**，瓶颈包括：

- 扩散迭代去噪（例如 16 steps）
- 14B DiT 主干计算成本
- 同步执行导致机器人“等模型算完才动”

### 4.2 三层优化（论文给出的分类）

论文将优化分成三类，我们按“能复述、能追问”来记：

- **算法级**：
  - **DreamZero-Flash**：解耦 video 与 action 的去噪 schedule（项目页也提到用于少步推理保持效果）
- **系统级（并行 + 缓存）**：
  - **CFG 并行**：classifier-free guidance 需要 conditional/unconditional 两次前向；论文写到把两次分摊到两张 GPU，上下文里给出**每 step 延迟降低 47%**。
  - **DiT caching**：利用 flow matching 下 velocity 的方向一致性；当相邻 velocity 余弦相似度高于阈值，复用缓存，论文描述可把“有效 DiT steps”从 16 降到 4，且动作质量损失很小。
- **实现/内核级**：
  - **torch.compile + CUDA Graphs**：减少 CPU 开销、算子融合；静态 shape 的重编译主要发生在第一条轨迹。
  - **后训练量化**：论文写到在 Blackwell 上把权重/激活量化到 **NVFP4**，但保留 QKV/Softmax 为 FP8，非线性为 FP16（敏感算子保精度）。
  - **Kernel/scheduler 优化**：attention 用 cuDNN 后端；把 scheduler 操作搬到 GPU，减少 CPU-GPU 同步 stall。

### 4.3 最终可用性：7Hz 闭环控制

项目页总结为：通过“model + system + implementation”优化，实现 **38×** 加速，将推理做到约 **150ms/动作 chunk**，从而达到 **7Hz closed-loop control**。

---

## 5. 评测设置与关键结论（项目页给的 6 大 setting + 关键数字）

项目页强调 DreamZero 覆盖 **6 类 setting**（其中 5 类测泛化，1 类测实时部署）：

1. **AgiBot Pretraining**：10 seen + 10 unseen tasks；在**新环境/新物体**上 zero-shot 测试  
2. **DROID Pretraining**：Franka；20 seen + 20 “unseen verbs”（动作在 DROID 中缺失）  
3. **Post-Training**：在 3 个下游任务上 finetune，同时要求保留 OOD robustness  
4. **New Embodiment Adaptation**：只用 **30 分钟 play data（55 条轨迹）**迁移到新本体（YAM）  
5. **Interactive Prompting**：现场让人随口提新任务，zero-shot 执行（“prompt robot foundation models”）  
6. **Real-Time Inference**：用 38× speedup 达成 7Hz

项目页给出的**关键数字（可直接背）**：

- **总体结论**：相对 SOTA VLA，在真实机器人上对“新任务+新环境”的泛化有 **>2×** 提升。
- **AgiBot（seen tasks, novel env/object）**：
  - DreamZero 平均 task progress **62.2%**
  - 最强预训练 VLA baseline **27.4%**
- **AgiBot（unseen tasks）**：DreamZero task progress **39.5%**（项目页举例：untie shoelace / shake hands 等）
- **DROID（unseen verbs）**：DreamZero **49%** task progress；SOTA VLA **25–32%**
- **跨本体迁移（video-only demonstrations）**：仅用 **10–20 分钟**的人类或其他机器人视频，未见任务表现**相对提升 >42%**
- **新本体少样本适配**：迁移到 **YAM**，只用 **30 分钟** play data，同时保留 zero-shot 泛化

---

## 6. 与 DreamGen / 1XWM 等路线对齐（你应该怎么比较）

### 6.1 DreamGen（2025）：世界模型“生成数据”，再训练策略

DreamGen 官方页把流程明确写成 **4-stage pipeline**（关键词：neural trajectories）：

1) finetune video world model 到目标机器人本体  
2) prompt 生成机器人视频（含新行为/新环境）  
3) 用 latent action model 或 IDM 提取伪动作  
4) 用“视频+伪动作”训练下游 visuomotor policy

对比 DreamZero：DreamGen 更像“用世界模型扩增数据”，最终仍落到单独 policy；DreamZero 则把“世界模型 + 动作”合成一个 WAM，直接作为闭环策略（并重点解决实时推理）。

### 6.2 1X World Model（Handbook 内路线）：先想象再执行（IDM + rejection sampling）

Handbook 的 `./frontier/one_x_world_model.md`（内部笔记）总结了一条“imagine then execute”的典型范式：世界模型生成未来，再用 IDM 执行动作，并常结合采样/筛选策略。

对比 DreamZero：DreamZero 强调**端到端联合建模 video+action**以及闭环下**用真实观测覆盖预测**来避免 AR 漂移；而“世界模型 + 独立 IDM/policy”的组合更容易在模块间产生对齐/误差传递问题（论文也在 related work 里讨论了模块化系统的 compounding errors 风险）。

---

## 7. 局限与开放问题（论文/项目页明确提到的 + 你可追问的点）

### 7.1 论文明确写到的限制

- **记忆类任务未系统评估**：论文脚注提到“本工作未显式评测或 post-train 仅能靠 memory 成功的任务”，留给未来。
- **“只生成动作是否更快？”并不明显**：论文指出在 14B 规模下，速度主要由 diffusion steps 与 DiT blocks 决定；而且 video/action 联合训练后，简单减少 action 去噪 steps 会伤质量，这推动了 DreamZero-Flash。

### 7.2 面试可追问（基于文中机制推导，不额外引入外部来源）

- **对齐机制的可解释性**：联合 flow-matching 是否会出现“视频看起来对，但动作不对”的失配？用什么诊断指标衡量 video-action alignment？
- **闭环替换观测的边界**：当观测存在遮挡/延迟/噪声时，用真实观测覆盖 KV cache 是否会造成分布突变？需要额外滤波或置信机制吗？
- **跨本体迁移的表示瓶颈**：video-only demonstrations 的迁移效果依赖什么？是视频先验、语言指令对齐，还是 action decoder 的可塑性？
- **实时部署成本**：CFG 并行写到了用两张 GPU 分担；在边缘部署/单卡场景下，DreamZero-Flash（少步推理）能把 trade-off 拉到什么程度？

---

## References

- DreamZero Project Page: `https://dreamzero0.github.io/`  
- DreamZero Paper (PDF): `https://dreamzero0.github.io/DreamZero.pdf`  
- DreamGen (NVIDIA Research): `https://research.nvidia.com/labs/gear/dreamgen/`  

[← Back to Theory](./README.md)

