# 从像素到动作：NEO-VLA 的原生 VLA 策略拆解 (From Pixels to Actions – Towards Native Vision-Language-Action Policies)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-11-05
>
> **论文**: From Pixels to Actions – Towards Native Vision-Language-Action Policies
> **链接**: https://arxiv.org/abs/2611.06918
> **核心定位**: 把 VLA 的视觉接口从“固定编码器输出”改成“backbone 多层 K/V 可选读出”，并用 encoder-free 预训练让早期视觉计算对控制可用；匹配动作专家和训练数据下，冻结骨干 LIBERO 平均 90.9%，全量微调 95.3%。

> **版本说明**：本文依据 arXiv:2611.06918v1 的 HTML 全文（2026-11-04）。候选摘要片段在 “evolving” 处截断，正文摘要完整；二者开头一致，未发现实质性主张差异。以下数字、表格与主张均以正文为准，不用候选摘要补写。

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | 动作策略不必只继承一个固定的视觉终点；NEO-VLA 让 flow-matching 动作专家跨深度读取 40 层 backbone 的 K/V，冻结骨干 LIBERO 平均 90.9%，全量微调 95.3%。但优势不只来自“去掉编码器”，而来自 encoder-free 预训练把早期视觉表示塑造成 control-ready。 |
| 適合精讀 | 如果你在做 VLA 架构、视觉编码器接口、动作头读出深度，重点看 §3.1–3.3、§3.4、§5.1–5.2。 |
| 可以跳過 | 如果你只关心真实机器人大规模部署或长时记忆机制，这篇距离中等：真实世界只有 3 个任务、每任务 20 trials。 |
| 落地可行性 | 中：2B backbone + 28/40 层 expert，RTX 3090 上 10 步 Euler 编译后 282.5 ms；但需要三阶段训练和大量机器人数据。 |
| 主要風險 | 预训练和模型容量差异不能完全归因于 encoder removal；LIBERO-Pro 位置/任务扰动仍弱；真实世界样本小。 |

💡 **X-Ray 开场**

传统 VLA 通常先用 CLIP/SigLIP/DINOv2 把图像压成一个固定视觉特征，再让语言骨干和动作头基于这个特征预测。NEO-VLA 问：能不能让动作头直接读取 backbone 不同深度的视觉计算？论文的回答是能，而且冻结骨干时早期层特别有用；但这不是简单“去掉编码器”——给 Qwen3-VL 直接接中间视觉特征，冻结时反而从 79.5% 掉到 64.9%（Table 3）。对 VLA 研究者意味着：视觉路径的“形成方式”和“暴露方式”可能比“有没有独立编码器”更关键。

📍 **研究全景时间线**

```
2023 RT-2 ──→ 2024 OpenVLA / π0 ──→ 2025 π0.5 ──→ 2026 NEO-VLA
离散动作 token   连续动作头/开源 VLA     开放世界泛化      encoder-free + 跨深度读出
```

本文位置：在“连续动作头”之后，进一步把问题推到视觉接口本身；局限是空间配置和任务目标变化时仍弱，真实世界验证规模有限。

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

先区分**已观测历史帧**与**未来预测帧**：NEO-VLA 的输入是 previous/current 多视角图像和 previous/current 机器人状态；输出是未来 30 步动作 chunk。30 步是未来预测区间，不是历史上下文，也不是所有实验共有的记忆窗口。

| 模块 | 输入 | 输出 | 时序/频率 | 训练/推理差异 |
|------|------|------|-----------|----------------|
| 共享卷积 patch embedder | 原始图像 patch | visual tokens，总 stride 32，每 32×32 区域一个 token，每视角最多 88 个 | 每帧、每视角 | 所有相机、视角、时间步共享；无 ViT、autoencoder、vector-quantised codebook |
| geometry MLPs | 相机内参、旋转 | geometry tokens | 每视角 | 轻量 MLP；**不是** patch embedder |
| state MLPs | previous/current Arm 0、Arm 1、shared body 状态 | 6 个 state tokens | 每时间块 | 联合编码测量值和 validity mask；缺失组件被 mask |
| 40 层 backbone | visual + text + geometry + state tokens | 分层 K/V | 每 query 一次 prefix pass | 三阶段预训练；视觉 token 在视觉块内双向，文本 token 因果 |
| action expert | 选定 backbone 层 K/V + noisy action tokens + flow time τ | velocity field | 每 query 10 个 Euler steps | 28 或 40 层；随机初始化；flow matching 训练 |

> 归属提醒：只有图像进入共享卷积 patch embedder；相机内参/旋转走 geometry MLPs，机器人状态走 state MLPs。三者不能合并成“共享 patch embedder 编码一切”。

### 1.2 关键机制 (Key Mechanism)

- **为什么 encoder-free**：论文的动机是，操作任务依赖局部几何、边界、深度和细微状态变化，而这些信号在语义压缩中容易丢失。NEO-VLA 让像素直接进入 backbone token 空间，视觉和语言表示在 40 层中共同演化。
- **为什么跨深度读出**：动作专家不只看最后一层，而是从层映射 M=(m₁,…,m_d) 指定的 backbone 层读取 cached K/V。这样“视觉读出深度”成为一个显式架构选择。
- **为什么匹配 action expert**：比较 native 与 encoder-based backbone 时，使用相同 action expert 和机器人训练语料，尽量隔离视觉路径的影响。
- **为什么 co-train 理解数据**：Table 6 显示，action-only 微调后模型失去生成有效文本的能力，RoboTwin C2R 从 co-train 的 69.4% 掉到 10.7%。作者据此主张多模态理解与鲁棒动作不是独立能力。
- **为什么 flow matching**：连续动作 chunk 避免离散 token 量化；推理用 10 个 Euler steps，在延迟和动作质量之间折中。

⚡ **Eureka Moment**：动作策略不该只继承一个固定的视觉终点，而应能读取 backbone 计算层级中不同深度的视觉表示；但前提是这些表示由 native 预训练塑造成 control-ready，而不是简单把编码器中间层接给动作头。

### 1.3 信息流/架构图 (Flow / Diagram)

```
previous/current multi-view images
        │
        ├─→ shared conv patch embedder (stride 32) → visual tokens (≤88/view)
        │
camera intrinsics/rotations ─→ geometry MLPs ─→ geometry tokens
robot previous/current states ─→ state MLPs ─→ 6 state tokens
task instruction / control frequency / embodiment ─→ text tokens
        │
        ▼
40-layer native backbone
  visual tokens: bidirectional within each visual block
  text tokens: causal
        │
        ├─→ cached K/V from selected layers M=(m1,...,md)
        │
noisy action tokens + flow time τ ─→ action expert (28 or 40 layers)
        │
        ▼
velocity field vθ(aτ, τ, C)
        │
10 Euler steps (τ=0→1), fixed prefix KV cache
        │
        ▼
30-step action chunk → denormalize → execute
```

层映射的四个设置（§3.3）：

| 设置 | 层映射 | 骨干深度 | 专家层数 | 说明 |
|------|--------|----------|----------|------|
| first-28 | M=(0,…,27) | 28 | 28 | 只读早期层；同时缩短骨干 |
| skip-28 | 28 层在 40 层中近似均匀采样 | 40 | 28 | 保留全部 40 层 |
| end-28 | M=(12,…,39) | 40 | 28 | 只读后期层 |
| full-40 | 全部 40 层 | 40 | 40 | 更高容量，单独比较 |

前三个设置使用相同 28 层 expert 和 hidden width，参数接近；full-40 更深，作者单独作为更高容量设置。

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：

```
a_τ = (1-τ)ε + τa,  target velocity = a - ε
L_action = E[ Σ_g w_g ℓ_g / Σ_g w_g ],  w_g=(2,1,1,1)
```

**目标**：学一个从高斯噪声到真实动作 chunk 的速度场，推理时从纯噪声积分回动作。

**训练公式**：

```
ε ~ N(0, I)
τ = 0.999u,  u ~ Beta(1, 1.5)
a_τ = (1-τ)ε + τa
v_θ(a_τ, τ, C) 的目标 = a - ε
L_action = E[ Σ_{g∈G} w_g ℓ_g / Σ_{g∈G} w_g ]
```

**变量说明**：

| 符号 | 含义 |
|------|------|
| a | 真实动作 chunk：arm 2×30×26，shared body 30×8 |
| ε | 高斯噪声，ε ~ N(0,I) |
| τ | flow time，τ=0.999u，u ~ Beta(1,1.5) |
| a_τ | 噪声与真实动作的线性插值 |
| C | multimodal prefix：图像、几何、语言、机器人状态 |
| v_θ(a_τ, τ, C) | 动作专家预测的速度场 |
| ℓ_g | 组 g 上有效维度的均方速度误差 |
| G | 有有效监督的动作组集合 |
| w_g | 四组权重 (2,1,1,1)：end-effector motion+gripper、arm joints、hand joints、body actions |

**推理公式**：

```
a_{k+1} = a_k + v_θ(a_k, τ_k, C)·Δτ
τ_k = k/10,  Δτ = 0.1,  k = 0..9
初始 a_0 = ε
```

推理从 τ=0 到 1 用 10 个 Euler steps，prefix KV cache 固定，不应用 classifier-free guidance。得到的 chunk 再反归一化回原动作空间。

**直觉**：flow matching 不直接回归动作，而是回归“从噪声指向真实动作”的直线速度。训练时随机采样 τ，让模型在噪声和真实动作之间的任意插值点上都能预测正确方向；推理时从纯噪声出发，沿学到的速度场走 10 步。

## 3. 带数字走一遍：玩具例子 (Worked Example)

> 以下为**假设的一维例子**，用于验算公式，不是论文实验数据。

假设真实动作 a=2，采样噪声 ε=-1，flow time τ=0.75。

**训练插值**：

```
a_τ = (1-0.75)×(-1) + 0.75×2
    = -0.25 + 1.5
    = 1.25
target velocity = a - ε = 2 - (-1) = 3
```

假设模型预测 v=2.8，则均方速度误差：

```
ℓ = (2.8 - 3)^2 = 0.04
```

**推理积分**：从 τ=0 到 1，10 步，Δτ=0.1，初始 a_0=ε=-1。

如果速度完美为 3：

```
a_10 = -1 + 10×0.1×3 = -1 + 3 = 2
```

正好命中真实动作。

如果速度有偏 2.8：

```
a_10 = -1 + 10×0.1×2.8 = -1 + 2.8 = 1.8
```

误差 0.2。这个例子说明：速度场的系统性偏差会在积分中累积；Euler 步数越少，离散化误差和速度误差的交互越明显。论文 Table 11 也显示，2 步推理更快（expert 32.4 ms vs 163.4 ms），但会改变采样动作，所以 Table 10 的所有成功率都用 10 步。

## 4. 工程视角 (Engineering View)

**延迟（Table 11，RTX 3090，BF16，raw 30-step chunks）**：

| 阶段 | 10 steps | 2 steps |
|------|----------|---------|
| Patch embedder | 3.1 ms | 2.7 ms |
| Backbone prefix（40 层） | 84.8 ms | 84.7 ms |
| Action expert sampling | 163.4 ms | 32.4 ms |
| Forward pass, eager | 515.4 ms | 210.7 ms |
| Forward pass, compiled | 282.5 ms | 147.8 ms |
| Robot: request → action chunk | 261.9 ms | 153.5 ms |
| Robot: request → executable plan | 286.2 ms | 178.5 ms |

**工程含义**：

- **KV cache 是核心**：backbone prefix 84.8 ms 每 query 只跑一次；10 个 Euler steps 复用同一份 prefix KV cache，所以主要成本在 action expert（163.4 ms）。2 步时 expert 降到 32.4 ms，prefix pass 反而成为主要成本。
- **编译收益**：torch.compile + 缓存 conditioning inputs，forward 从 515.4 ms 降到 282.5 ms；预测位置变化最多 0.002 cm。这个量级对厘米级操作是否可忽略，取决于任务容差，论文没有进一步做敏感性分析。
- **机器人端**：request → chunk 261.9 ms，request → executable plan 286.2 ms（含 inverse kinematics）。这是单次 query 的延迟，不是控制频率；控制频率作为 structured prompt 的一部分输入，论文没有给出具体数值，不能把 30 步 chunk 直接换算成固定控制频率。
- **动作 chunk 执行**：每次 query 返回一个 raw 30-step chunk，按预测直接执行，不做 real-time chunking。30 步是未来动作预测区间，不是历史记忆窗口。
- **内存**：论文只说明 RTX 3090 24 GB、BF16；没有给显存占用或吞吐数据，不能补写。

**Trade-off**：Euler 步数↓ → 延迟↓，但采样动作改变；论文默认 10 步，2 步仅作参考。action expert 层数 28 vs 40、backbone 读取深度 first-28/skip-28/end-28/full-40，都是容量与延迟/参数量的权衡。first-28 同时缩短 backbone 深度，所以它和 skip-28/end-28 的对比不只是“读出位置”，还混入了“骨干深度和计算量”的变化（论文 Table 4 明确提醒）。

## 5. 数据与评测 (Data & Eval)

### 5.1 预训练

九个公开数据集家族：AgiBot-World、Bimanual YAM、RoboCOIN、Open X-Embodiment、Galaxea Open-World、RoboMIND、RH20T、DROID、InternData-A1 (sim.)。总计 2,231,583 episodes、928.40M rows、7,444 views、9,927.7 hours（Table 1）。数据质量过滤包括：关节读数 |θ|>2π 或与五行动态中位数差 >π 标记无效；至少 90% episodes 中恒定的 joint/hand channels 被 mask；未解决的 joint discontinuities >π 按 embodiment mask；action outliers 按扩展的 per-embodiment bounds 检查；缺失/不可读数据、不一致记录形状、无效值、步数太少的 episode 被丢弃；相机标定检查。

**三阶段训练**：

| 阶段 | 可训练部分 | Steps | Batch size | Peak LR | 备注 |
|------|------------|-------|------------|---------|------|
| Stage 1 | action expert + conditioning modules | 113,400 optimizer steps | 8192 | 2×10⁻⁴ | 冻结 VLM backbone |
| Stage 2 | 全模型 | 75,000 steps | 8192 | expert 2×10⁻⁴；VLM 5×10⁻⁵ | 与多模态理解数据 co-train，每两个 optimizer step 加 packed understanding examples，auxiliary NTP loss 权重 0.1 |
| Stage 3 | 全参数 | 75,000 steps | 8192 | 同 stage 2 | 从 stage-2 初始化，用 in-house action data |

优化器 AdamW，β₁=0.9，β₂=0.95，ε=10⁻⁸，weight decay 10⁻¹⁰；LR 线性 warmup 2,000 steps，cosine decay 到各自 peak 的 10%；bfloat16 混合精度，梯度范数 clip 1.0。

### 5.2 仿真评测

| 基准 | 设置 | NEO-VLA 结果 | 备注 |
|------|------|--------------|------|
| LIBERO（Table 7） | step 27,500，每 suite 2,000 episodes | 平均 97.4 | Spatial 98.4 / Object 99.4 / Goal 95.6 / Long 96.2 |
| LIBERO-plus（Table 5） | 7 个扰动轴，task-weighted total | 89.7 | Camera 94.5 / Robot 72.6 / Language 91.7 / Light 96.2 / Background 96.6 / Noise 96.9 / Layout 82.4 |
| LIBERO-Pro（Table 8） | Object / Position / Semantic / Task | 平均 47.9 | Object 84.3 / Position 3.0 / Semantic 98.0 / Task 6.3 |
| RoboTwin 2.0（Table 9） | official 2.5K episodes | Clean 75.78 / Clean2Rand 69.40 | Clean 落后 Qwen-RobotManip 89.58；C2R 在开源中最好 |
| 骨干对比（Table 2） | final 30,000-step checkpoint，2,000 episodes | Frozen 90.9 / Full FT 95.3 | Qwen3-VL frozen 79.5 / full FT 90.8；Gemma-3 frozen 84.5 / full FT 91.0 |
| 层映射消融（Table 4） | training step 30k，每配置 2,000 rollouts | Frozen: first-28 94.9 / skip-28 90.9 / end-28 86.7；Finetuned: first-28 94.6 / skip-28 95.3 / end-28 92.5 | first-28 同时缩短骨干深度 |

> 注意：Table 2 的 checkpoint 是 final 30,000-step；Table 4 是 step 30k；Table 7 是 step 27,500。三者不能混为同一个评测点。Table 9 是 official 2.5K episodes，不是 2,000。

### 5.3 真实机器人

三个任务：Task 1 把网球放到罐子上；Task 2 把胶带卷放碗里，再把网球放进胶带卷；Task 3 把水平罐子转正插入胶带卷。每个任务只用 50 个 demonstration episodes 微调，评测 20 trials（10 in-domain + 10 out-of-domain），Table 10：

| 模型 | Size | Task 1 | Task 2 | Task 3 |
|------|------|--------|--------|--------|
| StarVLA | 3B | 1/20 | 2/20 | 0/20 |
| π0.5 | 3B | 10/20 | 14/20 | 13/20 |
| NEO-VLA | 2B | 12/20 | 17/20 | 13/20 |

真实世界设置与仿真评测分开：这里是 50 demos/task、20 trials/task，不是 LIBERO 的 2,000 episodes。样本量小，不能据此声称广泛泛化。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

**能做什么**：

- 在 LIBERO 分布内保持强性能（97.4），但作者自己说最强方法差距小，LIBERO 越来越难区分当前策略，所以主要用来证明“没有掉点”。
- 对视觉外观/相机/语言扰动较鲁棒：LIBERO-plus camera 94.5、language 91.7、noise 96.9。
- 冻结骨干迁移强：90.9 平均，超过 Qwen3-VL frozen 79.5 和 Gemma-3 frozen 84.5；LIBERO-Long frozen 85.4，超过两个 encoder-based 模型的 full-FT 结果。
- 真实世界小样本微调有竞争力：12/20、17/20、13/20，匹配或超过 π0.5，明显优于 StarVLA。

**不能做什么 / 弱在哪里**：

- LIBERO-Pro Position 3.0、Task 6.3：空间配置和任务目标变化时显著下降。作者归因于 native backbone 提供感知/语义鲁棒性，但灵活空间和任务级适应仍是关键挑战。
- LIBERO-plus Robot 72.6、Layout 82.4：embodiment 或场景配置变化时增益较小；作者说这符合“对感知方式变化更鲁棒，而非对所有分布偏移鲁棒”。
- RoboTwin clean 75.78，落后 Qwen-RobotManip 89.58；优势主要在 Clean2Rand 69.40。
- 真实世界只有 3 个任务、每任务 20 trials，统计功效有限；不能声称对移动、双臂、人形或开放世界普遍有效。
- 没有明确的长时记忆机制；previous/current 观测是有限历史，30 步 action chunk 是未来预测区间，不能当作记忆窗口。

### 6.1 隐含假设 (Hidden Assumptions)

- **匹配 action expert 和训练数据足以公平比较 backbone**：作者在 §5.1 承认预训练和模型容量差异 preclude attributing the gains solely to encoder removal。Qwen3-VL/Gemma-3 的预训练数据、视觉分辨率、语言能力与 NEO 的 2B backbone 不同。
- **Open X-Embodiment 75k iters 足以让 encoder-based 模型适应机器人**：这是控制变量，但未必是每个 backbone 的最优适配；冻结/全量微调的结论依赖这个预算。
- **“early” 的定义依赖具体层映射**：first-28 同时把 backbone 从 40 层缩到 28 层，混淆了读出位置和骨干深度；first-14/last-14 的 disjoint 对比也有同样问题。skip-28 vs end-28 是更干净的读出位置对比。
- **真实世界 20 trials 的统计功效有限**：12/20 vs 10/20 这类差异不宜过度解读。
- **30 步 chunk 是固定预测区间**：论文没有验证更长历史、闭环记忆或可变 chunk 长度；不能把 30 步写成所有实验共有的记忆窗口。
- **动作专家读取 K/V 的“深度”不等于“时间记忆”**：层映射是视觉计算层级，不是历史帧窗口。

## 7. 与相关工作对比 (Comparison)

| 模型 | 视觉路径 | 动作表示 | 读出深度 | 开放状态（仅表格标注） | 关键数字/备注 |
|------|----------|----------|----------|------------------------|----------------|
| RT-2 | 预训练视觉编码器 | 离散文本 token | 最终层 | 本文未列表格 | 2023，早期 VLA 代表 |
| OpenVLA | 预训练视觉编码器 | 离散 token | 最终层 | ✓（Table 7） | 7B，LIBERO 76.5 |
| π0 | 预训练视觉编码器 | 连续 flow matching | 最终层 | ✓（Table 5） | 3B，LIBERO-plus 53.6；LIBERO-Pro 45.3（Table 5/8）；LIBERO 96.8 属 π0.5（Table 7） |
| π0.5 | 预训练视觉编码器 | 连续 | 最终层 | ✓（Table 5/7/9） | 3B，LIBERO 96.8；LIBERO-plus 84.4；RoboTwin C2R 46.00 |
| Qwen3-VL + action expert | 预训练视觉编码器 | 连续 | 最终层；另测 encoder 中间层 | 论文对比，未标 Open | 2B，frozen 79.5；full FT 90.8；encoder access frozen 64.9 |
| Gemma-3 + action expert | 预训练视觉编码器 | 连续 | 最终层 | 论文对比，未标 Open | 4B，frozen 84.5；full FT 91.0 |
| NEO-VLA | encoder-free patch embedder | 连续 flow matching | 可选 0–39 层 | ✓（Table 5/7/9） | 2B，LIBERO 97.4；LIBERO-plus 89.7；RoboTwin C2R 69.40 |

> 逐模型归属提醒：上表的“开放状态”只按论文表格中的 Open 列填写；Qwen3-VL 和 Gemma-3 在 Table 2 是骨干对比，不是 Table 7 的 Qwen-RobotManip。Table 5 中 Qwen-RobotManip 标 ✗，不能把 Qwen3-VL 的公开模型身份混进去。StarVLA 在 Table 5 是 4B，在 Table 10 是 3B，也不能合并。

**面试 Tip**：如果被问“NEO-VLA 是不是就是去掉视觉编码器？”答：不完全是。它的关键是把 action expert 接到 backbone 的多层 K/V 上，并且用 encoder-free 预训练让早期层保留 control-ready 的几何/边界信息；Table 3 显示，给 Qwen3-VL 直接接中间视觉特征，冻结时反而从 79.5 掉到 64.9，说明“怎么形成和暴露表示”比“有没有编码器”更重要。

## 8. 精讀建議 (Reading Guide)

- **值得精读原文的人**：
  - 做 VLA 架构设计、视觉编码器接口、动作头读出深度的研究者；
  - 想复现“encoder-free 预训练 + flow-matching action expert”的工程师；
  - 需要理解 LIBERO-plus/LIBERO-Pro/RoboTwin 评测条件和扰动轴的人。
- **建议章节路径**：先读 §3.1–3.3（视觉接口、multimodal prefix、层映射）→ §3.4（flow matching 目标和损失）→ §5.1–5.2（骨干对比和层消融）→ §5.4（评测条件）→ 可跳 §2 相关工作（如果熟悉）和 §5.5.3 延迟（如果只关心算法）。
- **不值得精读的理由**：如果你不做机器人学习、只关心通用 VLM，或已经非常熟悉 flow matching 和 LIBERO 评测，读摘要和表格即可；真实世界部分只有 3 个任务、每任务 20 trials，不足以支撑广泛部署判断。

---

[← Back to Theory](./README.md)

**关键引用**：

- 论文：https://arxiv.org/abs/2611.06918
- HTML 全文：https://arxiv.org/html/2611.06918