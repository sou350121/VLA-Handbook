# Ego4Gen：把几何接地运动 token 当作世界-动作接口（Ego4Gen: Generating Geometry-Grounded Motion Tokens as a World-Action Interface）

> ⚙️ 本文由 Moltbot 自动生成 | 2026-11-25
>
> **论文**: Ego4Gen: Generating Geometry-Grounded Motion Tokens as a World-Action Interface
> **链接**: https://arxiv.org/abs/2611.27513
> **核心定位**: 把 MotionCrafter 的 geometry-motion latent 当作 world-action interface：单张 egocentric image + text 生成未来 4 秒 RGB/点图/场景流，再用轻量 FlowInverse 解码动作；卖点是 OOD Hard 泛化，不是 Easy SOTA。

> 来源核验：本文依据 arXiv:2611.27513v1 的 HTML/PDF 正文（2026-11-20 提交）。候选摘要与 PDF 摘要开头一致，但候选摘要截断于 “…modality rou”；以下主张均以 PDF 正文为准，未用候选摘要补写。论文未提供代码、权重或数据下载链接，文中不杜撰仓库地址。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 在 RoboTwin 2.0 仿真中，geometry-motion latent 作为动作解码接口，Hard 设置下比 LAM/video latent 更稳；但闭环 mean 不是全场最高。 |
| 适合精读 | 做 world-action model、latent action interface、4D 生成的人；重点看 §3.1–3.3、§4.1、§4.3。 |
| 可以跳过 | 只关心真机部署或 Easy SOTA 的人：这篇没有真机，Easy 也不是最强。 |
| 落地可行性 | 中低：仿真闭环有数字，但无真机、per-task fine-tune、推理延迟未报告。 |
| 主要风险 | 单帧条件、无显式记忆；依赖 MotionCrafter pseudo-label；Mixed 设置导致 Easy 下降；闭环对比条件不完全统一。 |

💡 **X-Ray 开场**

这篇论文问的是：世界模型预测了未来，到底用什么表示去驱动动作？RGB latent 太吃外观，latent action 又可能混入与动作无关的视觉噪声。Ego4Gen 的答案是：用 MotionCrafter 的 geometry-motion latent 作为中间接口，先生成未来 4 秒的点图和场景流，再用一个轻量 inverse dynamics 网络解码动作。对 VLA 研究者，它的价值不在生成画质，而在验证“几何-运动 latent 比 RGB/latent action 更抗 OOD”这个假设。

📍 **研究全景时间线**

```
2025  TesserAct：4D embodied world model（ICCV 2025）
  │
2026  MotionCrafter：4D VAE 提供 geometry/motion latent（CVPR 2026）
  │
2026  Cosmos3-Nano：omni-modal world model backbone
  │
2026  Ego4Gen：geometry-motion latent → FlowInverse → 仿真闭环  ← 本文
```

> 时间线只列论文参考文献中有明确年份的节点；不罗列完整参考文献表。

## 1. 核心架构/方法总览

### 1.1 系统对比概览（逐模型归属）

| 组件/模型 | 角色 | 输入 | 输出 | 训练状态 | 数据归属 | 记忆/上下文 |
|---|---|---|---|---|---|---|
| Cosmos3-Nano | 预训练生成 backbone | text + 图像 latent | vision latent 及 MoT 内部特征 | 冻结；禁用 audio/action 分支 | 论文未详述其预训练数据 | 无显式记忆；Ego4Gen 只用它做单帧条件生成 |
| Ego4Gen generator | 在 Cosmos3-Nano 上加 G/M 分支与 LoRA | I0 + text | z_V、z_G、z_M | 训练 LoRA 与 G/M 输入输出头 | EgoDex、AgiBot、RoboTwin 2.0 clean-task trajectories | 无历史帧堆叠；4 秒是未来预测 horizon，不是记忆窗口 |
| MotionCrafter 4D VAE | 提供 geometry/motion latent 空间与伪标签 | RGB 帧 | 点图/场景流 latent 或重建 | 冻结 | 论文未述其训练数据；本文只作 frozen encoder/predictor | 不参与闭环决策 |
| FlowInverse | 轻量 inverse dynamics | z_G、z_M + current proprioceptive state | joint-action increments、gripper states；辅助 ee displacement | 两阶段训练；Ego4Gen 冻结 | 阶段一：RoboTwin 2.0 50 clean tasks 的 expert videos；阶段二：每个目标任务 Ego4Gen 生成 latents + 动作 | 输入 current state + future latents；无历史动作序列 |
| Wan2.2 VAE | vision VAE | RGB 帧 | vision latent | 冻结 | 论文未述 | 不参与决策 |

**关键归属说明**：Ego4Gen 的“4 秒预测窗口”是未来生成区间，不是历史上下文，也不是记忆窗口。模型输入只有单张当前 egocentric image I0 和 text prompt；R 序列把 I0 复制为第一帧，其余为未来帧。FlowInverse 只接收 current proprioceptive state 与未来 G/M latents，不接收历史动作序列。论文没有提出显式 memory 模块。

### 1.2 关键机制

1. **为什么选 geometry-motion latent**：作者认为 RGB 视频把任务相关状态变化与颜色、纹理、光照、背景混在一起；latent action 虽然紧凑，但可能编码 exogenous noise。Geometry/motion latent 更接近控制相关结构，且 MotionCrafter 的 4D VAE 能同时解码出点图和场景流。

2. **三模态共享 backbone，modality-routed LoRA**：vision、geometry、motion 三路 token 共享 Cosmos3-Nano 的生成 backbone，但每个 modality 有独立的低秩适配器。Text token 用冻结的 base 参数。这样在数据有限时减少对预训练先验的破坏。

3. **Asymmetric attention + stop-gradient**：RGB token 只 attend 到 text 和 RGB；point/flow token attend 到所有模态，但对 RGB 的 K/V 做 stop-gradient。信息流是单向的：G/M 可以利用 RGB 上下文，RGB 生成不被几何监督干扰。

4. **时空对齐**：vision 侧保留 Wan2.2 VAE latent，未来帧按时间压缩 4 倍；geometry/motion 侧用 MotionCrafter latent，无时间压缩但空间压缩因子不同。通过 modality-specific spatiotemporal patchification 对齐到相同空间 token 网格：vision 25 个 temporal slots，geometry 和 motion 各 24 个 slots。Motion 表示相邻 geometry 帧之间的 3D 位移，因此时间坐标偏移 0.25 个 temporal unit。

5. **FlowInverse 两阶段训练**：阶段一在 RoboTwin 2.0 的 50 个 clean tasks 上预训练，latents 从 expert videos 用 frozen MotionCrafter 提取；阶段二对每个目标任务单独 fine-tune，输入换成 Ego4Gen 生成的 G/M latents 与对应动作轨迹。Ego4Gen 全程冻结。

6. **无动作标签的 world-model post-training**：Mixed 设置从同一 warmup checkpoint 出发，在 clean + randomized videos 上继续训练 Ego4Gen 5k steps，不使用动作标签；FlowInverse 权重不变。作者想验证：只更新世界模型也能提升下游 Hard 泛化。

⚡ **Eureka Moment**：把“未来 4D 几何-运动 latent”而不是 RGB latent 或压缩的 latent action，作为世界模型与动作解码器之间的瓶颈；并用 stop-gradient 让几何分支单向读取 RGB，避免几何监督污染视频生成。

### 1.3 信息流/架构图

```
I0（单张 egocentric RGB）+ text c
        │
        ▼
Cosmos3-Nano backbone（frozen）+ modality-routed LoRA
        │
        ├── V branch → z_V → Wan2.2 VAE decoder → RGB frames
        │                         （96 帧 @24 FPS，含 I0）
        ├── G branch → z_G → MotionCrafter decoder → dense point maps
        │                         （48 帧 @12 FPS）
        └── M branch → z_M（+ z_G）→ dense scene flow
                                  （48 帧 @12 FPS）
        │
        ▼
FlowInverse：z_G、z_M + current proprioceptive state
        │
        ▼
joint-action increments + gripper states
（辅助监督：end-effector displacement）
```

**先区分历史与未来**：I0 是已观测的当前帧；R 中的 I0 是复制进目标序列的 conditioning frame；P/F 从 t=0 开始预测未来。模型没有多帧历史输入，也没有历史动作输入。闭环时每 40 个 control steps 用当前 head-camera observation 重新生成一次未来 latents，这是 receding-horizon 执行，不是记忆机制。

## 2. 数学核心

📌 **Napkin Formula**：

```
I0, c → p(R, P, F | I0, c) → z_GM → FlowInverse → a
```

目标：给定单张图像和文本，联合建模未来 RGB、点图、场景流的条件分布；再把 G/M latent 映射为动作。

```
p_θ(R, P, F | I0, c)
```

其中：

```
R = {I_i}_{i=0}^{T_V-1} ∈ R^{T_V×3×H×W}
P = {P_j}_{j=0}^{T_GM-1} ∈ R^{T_GM×3×H×W}
F = {F_j}_{j=0}^{T_GM-1} ∈ R^{T_GM×3×H×W}
T_V = 96 @ 24 FPS
T_GM = 48 @ 12 FPS
```

LoRA 路由：

```
W̃_s^(m) = W_s + (α_m / r_m) · B_s^(m) A_s^(m)
m ∈ {V, G, M},  s ∈ {Q, K, V, O}
```

Asymmetric attention：

```
O_V   = Attn(Q_V, [K_T; K_V], [V_T; V_V])
O_G/M = Attn(Q_G/M,
             [K_T; sg(K_V); K_G; K_M],
             [V_T; sg(V_V); V_G; V_M])
```

`sg(·)` 是 stop-gradient。RGB 只读 text 和 RGB；G/M 读所有模态，但 RGB 的 K/V 不回传梯度。

多模态 rectified-flow 损失：

```
L = L_V + L_G + L_M
L_m = MSE(v̂_m, ε_m − z_m^0)
```

FlowInverse 的加权 MSE：

```
L = λ_joint · L_joint + λ_gripper · L_gripper + λ_ee · L_ee
```

| 符号 | 含义 |
|---|---|
| I0 | 初始 egocentric RGB 图像 |
| c | text prompt |
| R, P, F | 未来 RGB 帧、点图、稠密场景流 |
| z_V, z_G, z_M | 三模态 clean latents |
| v̂_m | 模型预测的速度场 |
| ε_m | 高斯噪声 |
| z_m^0 | 对应模态的 clean latent |
| W_s | 预训练 attention 投影 |
| A_s^(m), B_s^(m) | 低秩分解因子 |
| r_m, α_m | LoRA rank 与 scaling |
| λ_joint, λ_gripper, λ_ee | 损失权重；论文未给具体数值 |

**直觉**：RGB 负责外观，G/M 负责几何与运动；G/M 可以看 RGB 但不能反向干扰 RGB；动作解码只读 G/M + proprio，避免外观噪声进入控制。作者在 §1 中明确把“减少 task-irrelevant visual variation”作为接口设计的出发点。

## 3. 带数字走一遍：玩具例子

> 以下数值为假设性演算，用于说明时间跨度与重规划关系，不是论文实验设置。

### 3.1 采样帧首末跨度 vs 预测终点偏移

假设 geometry/motion 序列均匀采样，K = 48 帧，Δt = 1/12 秒（12 FPS）。

```
首末采样帧时间戳差 = (K − 1) × Δt
                  = 47 × (1/12)
                  = 3.917 秒
```

如果从当前参考 t 预测 t+1 至 t+H，H = 48：

```
预测终点距当前参考 = H × Δt
                  = 48 × (1/12)
                  = 4.000 秒
```

两者差 1 个 Δt：前者是“48 个采样帧的首末跨度”，后者是“48 个未来步长覆盖的终点偏移”。Ego4Gen 的“4 秒预测窗口”对应后者；不要把 48 帧的首末跨度 3.917 秒误写成 4 秒，也不要把 4 秒预测窗口当作历史上下文。

再看 latent slot：G/M 有 24 个 slots，base temporal rate 为 6，单位 1/6 秒。首末 slot 中心差：

```
(24 − 1) × (1/6) = 23/6 ≈ 3.833 秒
```

24 个 slot 覆盖 24 × 1/6 = 4 秒；首末中心差小于 4 秒，因为 slot 有宽度。这说明“slot 数 × 单位”与“首末中心差”不是同一个量。

### 3.2 重规划周期（假设控制频率）

论文写“every 40 control steps”重新生成未来 latents，但未给控制频率。假设 f = 50 Hz（仅作演算）：

```
重规划周期 = 40 / 50 = 0.8 秒
一个 4 秒预测窗口内可重规划 4 / 0.8 = 5 次
```

若 f = 20 Hz，则周期 2 秒，4 秒窗口内重规划 2 次。这个例子说明：4 秒是预测 horizon，40 control steps 是执行时的重规划间隔；两者都不是记忆窗口。论文没有报告控制频率，所以不能把 0.8 秒写成论文实测。

## 4. 工程视角

| 项目 | 论文设置 | 工程含义 |
|---|---|---|
| 训练硬件 | 8× NVIDIA RTX PRO 6000 | 论文未给 GPU hours；无法直接估算成本 |
| 训练步数 | 40K optimizer steps，global batch 24 | 这是预训练 Ego4Gen 的设置；不是 epoch 数 |
| LoRA | vision (r, α) = (32, 64)；G/M (r, α) = (64, 128) | 三模态独立适配器；text 用 frozen base |
| 数据窗口 | 4 秒训练窗口 | 所有预训练数据组织成 4 秒窗口 |
| 视频采样 | T_V = 96 @ 24 FPS；T_GM = 48 @ 12 FPS | 同一预测窗口，不同采样率 |
| Latent slots | vision 25；G/M 各 24 | 空间 token 形状一致；motion 时间偏移 0.25 |
| 闭环重规划 | 每 40 control steps | 具体控制频率论文未给；不能换算成固定秒数 |
| FlowInverse | 轻量 inverse dynamics | 论文未给参数量、FLOPs、推理延迟 |
| 生成采样 | rectified flow 多步去噪 | 论文未给采样步数；部署延迟关键未知 |
| 真实机器人 | 无 | 所有闭环数字来自 RoboTwin 2.0 仿真 |

**工程判断**：这套系统的模块边界清晰——世界模型生成 G/M latent，FlowInverse 解码动作，二者解耦训练。优点是可以用无动作标签视频更新世界模型而保持 FlowInverse 不变；缺点是 FlowInverse 仍需 per-task fine-tune，且世界模型的多步生成延迟未报告。若部署在真实机器人上，需要额外验证相机标定、latent 尺度对齐、生成失败时的回退策略。论文的 TACO 评测用相机内外参把点图转深度，说明几何输出对相机参数敏感。

## 5. 数据与评测

### 5.1 预训练

- 数据：EgoDex、AgiBot、RoboTwin 2.0 中 ALOHA-AgileX 的 clean-task trajectories。
- 组织方式：全部切成 4 秒训练窗口。
- 目标：联合去噪 RGB、点图、场景流三个模态。
- 监督：geometry/motion latent targets 来自 frozen MotionCrafter predictor，属于 pseudo-label，不是真实 3D 标注。
- 训练：8 GPU，global batch 24，40K optimizer steps，WSD learning-rate schedule。

### 5.2 仿真评测

**(a) 4D 生成：TACO**

TACO 不参与训练，提供 RGB、depth、相机内参和逐帧外参。论文在 2 秒和 4 秒 horizon 上评测；4DNeX 只生成 2 秒，所以 2 秒表包含它，4 秒表不包含。

| 方法 | 2s PSNR↑ | 2s LPIPS↓ | 2s Depth AbsRel↓ | 2s Points Rel_p↓ | 4s PSNR↑ | 4s LPIPS↓ | 4s Depth AbsRel↓ | 4s Points Rel_p↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Wan2.2 | 14.93 | 0.440 | – | – | 13.69 | 0.513 | – | – |
| Cosmos3-Nano | 15.89 | 0.398 | – | – | 14.69 | 0.461 | – | – |
| RynnWorld-4D | 16.27 | 0.413 | 0.174 | – | 14.79 | 0.493 | 0.184 | – |
| 4DNeX | 15.04 | 0.465 | – | 30.34 | – | – | – | – |
| TesserAct | 15.71 | 0.423 | 0.144 | – | 14.35 | 0.491 | 0.160 | – |
| Ego4Gen | 16.06 | 0.394 | 0.140 | 23.21 | 14.83 | 0.456 | 0.153 | 26.77 |

**准确表述**：2 秒 PSNR 上 Ego4Gen 16.06 低于 RynnWorld-4D 16.27，不是最高；但 LPIPS 0.394 优于 RynnWorld-4D 0.413，Depth AbsRel 0.140 优于 TesserAct 0.144 和 RynnWorld-4D 0.174，Points Rel_p 23.21 优于 4DNeX 30.34。4 秒 PSNR 14.83 略高于 RynnWorld-4D 14.79，LPIPS 0.456 最低，Depth AbsRel 0.153 优于 RynnWorld-4D 0.184 和 TesserAct 0.160。作者声称“leading performance in 4D generation”，但严格说不是所有指标、所有 horizon 都第一。

Motion 评测没有真实 3D motion，论文用 VGGT-Ω + AllTracker 在同一 Ego4Gen 生成 RGB 视频上构造 pseudo-references，并在 12 Hz 的有效动态点上比较。MotionCrafter 的 EPE 为 1.430 cm、APD_0.01 48.67%；Ego4Gen 为 1.452 cm、47.22%。两者非常接近，Ego4Gen 略差；Rel_p 和 δ_p 上 Ego4Gen 略好（10.77/92.69 vs 11.04/92.41）。作者用“closely matches”描述，不应写成“超过 MotionCrafter”。

**(b) 闭环控制：RoboTwin 2.0**

设置：ALOHA-AgileX embodiment；5 个任务；Easy 为 clean，Hard 为 randomized。所有闭环 baseline 只用 clean RoboTwin 数据训练。Ego4Gen clean-only 先在 50 个 clean tasks 上 post-train 10k steps；FlowInverse 阶段一在 50 个 clean tasks 上预训练，阶段二每个目标任务单独 fine-tune。执行时每 40 control steps 用当前 head-camera observation 生成未来 latents。

| 方法 | Grab Roller Easy/Hard | Place Empty Cup Easy/Hard | Pick Diverse Bottles Easy/Hard | Place Shoe Easy/Hard | Place Bread Basket Easy/Hard | Mean Easy/Hard |
|---|---:|---:|---:|---:|---:|---:|
| DP | 100 / 3 | 15 / 0 | 8 / 1 | 20 / 0 | 9 / 0 | 30.4 / 0.8 |
| TurboVLA | 93 / 3 | 88 / 1 | 40 / 4 | 51 / 0 | 44 / 0 | 63.2 / 1.6 |
| starVLA | 96 / 38 | 84 / 0 | 43 / 8 | 42 / 0 | 71 / 2 | 67.2 / 9.6 |
| Fast-WAM | 100 / 1 | 92 / 0 | 64 / 0 | 88 / 0 | 81 / 0 | 85.0 / 0.2 |
| π0 | 96 / 80 | 37 / 11 | 27 / 6 | 28 / 6 | 17 / 4 | 41.0 / 21.4 |
| Ego4Gen | 91 / 20 | 61 / 20 | 37 / 13 | 26 / 13 | 18 / 10 | 46.6 / 15.2 |
| Ego4Gen (Mixed) | 78 / 58 | 46 / 23 | 22 / 21 | 27 / 21 | 13 / 9 | 37.2 / 26.4 |

**准确表述**：Ego4Gen clean-only 的 Hard 成功率在 Place Empty Cup、Pick Diverse Bottles、Place Shoe、Place Bread Basket 四个任务上最高（例如 Place Empty Cup 20 vs π0 11、starVLA 0；Pick Diverse 13 vs π0 6、starVLA 8）。但 Mean Hard 15.2 低于 π0 的 21.4，因为 π0 在 Grab Roller Hard 上达到 80。Mean Easy 46.6 也低于 Fast-WAM 85.0、starVLA 67.2、TurboVLA 63.2。作者没有声称 Easy SOTA；论文的卖点是 Hard 泛化和无动作标签适配。

**Mixed 设置**：从同一 warmup checkpoint 出发，在 clean + randomized videos 上无动作标签 post-train 5k steps，FlowInverse 冻结。Mean Hard 从 15.2 升到 26.4（+11.2 个百分点），Mean Easy 从 46.6 降到 37.2（−9.4 个百分点）。作者认为 Easy 下降可能部分源于更新后的 generator latents 与 FlowInverse 训练时所用 latents 不匹配。这是作者的解释，不是实验证明的因果结论。

**(c) 消融**

Table 6 在 EgoDex test clips 上训练 5k steps，比较 shared LoRA、routed LoRA、asymmetric attention（AA）、gradient isolation（GI）。Full 的 PSNR 23.55、SSIM 0.820、LPIPS 0.180 最好；NC 0.01472 略高于 Routed+AA 的 0.01446，不是最低。作者结论是组合有助于在数据有限时保留预训练先验。

Table 7 去掉 FlowInverse 的 50-task 预训练，Mean 从 46.6/15.2 降到 18.4/7.4，说明预训练阶段贡献明显。但 Grab Roller Hard 去掉预训练后反而更高（31 vs 20；有预训练为 20），说明预训练并非对所有任务都正收益。

### 5.3 真实机器人设置

**论文没有真实机器人闭环实验。** 所有闭环数字来自 RoboTwin 2.0 仿真，embodiment 为 ALOHA-AgileX。没有 sim-to-real 迁移、没有真机成功率的报告。这是评价这篇论文时必须明确的边界。

## 6. 能力与失败模式

| 能力 | 证据/条件 |
|---|---|
| 单图 + 文本生成未来 4 秒 RGB/点图/场景流 | §3.1；三模态时间对齐 |
| G/M latent 可显式解码为 3D 结构 | §3.2 解码器；TACO 点图指标 |
| G/M latent 可映射为动作 | §3.3 FlowInverse；Table 1、Table 5 |
| OOD Hard 下比 LAM/video latent 稳 | Table 1：GM Hard 51/37；DreamDojo/UniVLA/Wan2.2 Hard 多为 0 |
| 无动作标签视频更新世界模型可提升 Hard | Table 5：Mixed Hard mean 15.2→26.4；FlowInverse 冻结 |

| 失败模式 | 原因/条件 |
|---|---|
| 无真机验证 | 只有 RoboTwin 2.0 仿真；sim-to-real 未知 |
| clean-only 闭环 mean 非最强 | Easy 低于 Fast-WAM/starVLA；Hard mean 低于 π0（但 Ego4Gen (Mixed) Hard mean 26.4 高于 π0 21.4） |
| 单帧条件、无历史记忆 | 输入只有 I0；没有多帧历史或动作历史；遮挡/长时任务风险 |
| 依赖 pseudo-label | G/M targets 来自 MotionCrafter；几何质量受伪标签限制 |
| 依赖相机标定 | TACO 评测用内外参把点图转 depth；真实部署需标定 |
| per-task fine-tune | FlowInverse 阶段二每个目标任务单独训练；不是 zero-shot 通用策略 |
| Mixed 导致 Easy 下降 | 更新后的 latents 与 FlowInverse 训练分布不匹配；作者解释，未做因果验证 |
| 生成错误级联 | 动作解码依赖未来 G/M latents；世界模型生成错误会传给 FlowInverse |
| 4D 生成并非全面 SOTA | 2 秒 PSNR 低于 RynnWorld-4D；motion EPE 略差于 MotionCrafter |

### 6.1 隐含假设

1. **单帧 I0 + text 足以决定未来 4 秒**：模型没有历史帧堆叠，也没有把 robot state 输入世界模型。对于需要历史上下文的任务（遮挡恢复、多阶段操作），这个假设可能不成立。
2. **G/M latent 保留控制相关信息，且外观噪声被充分去除**：这是全文的核心假设。Table 1 支持它在 OOD Hard 下有用，但没有证明“所有控制相关信息都在 G/M 中”。
3. **仿真 Hard 随机化能代表真实 OOD**：RoboTwin 2.0 的 randomized 设置是论文的主要泛化证据；真实世界的光照、材质、相机抖动未必被覆盖。
4. **MotionCrafter pseudo-label 足够可信**：训练目标是伪标签；TACO 评测也用 VGGT-Ω + AllTracker 构造 pseudo-reference。几何真值的缺失是系统性限制。
5. **4 秒窗口足够覆盖任务相关动态**：更长任务需要 receding-horizon 重规划；论文没有研究窗口长度对闭环的影响。
6. **FlowInverse 的 per-task fine-tune 可接受**：实际部署若任务频繁切换，需要为每个任务收集数据并重训，工程成本不低。

## 7. 与相关工作对比

| 方法 | 论文中的定位 | 接口/输出 | 在本文中的角色 | 关键结果/备注 |
|---|---|---|---|---|
| DreamDojo | LAM baseline | latent action | Table 1 表征对比 | Easy Place Empty Cup 88 最高，但 Hard 0；说明 OOD 崩 |
| UniVLA | LAM baseline | task-centric latent action | Table 1 表征对比 | Hard 0/4；OOD 崩 |
| Wan2.2 | video VAE baseline | video latent | Table 1 表征对比；4D 生成 baseline | Hard 0/0；PSNR 较低 |
| RynnWorld-4D | 4D world model baseline | RGB + depth（用初始 depth） | 4D 生成对比 | 2s PSNR 16.27 最高；Hard 闭环未测 |
| 4DNeX | 4D generation baseline | 2 秒 4D 生成 | 2s 点图对比 | 只支持 2 秒；Rel_p 30.34 |
| TesserAct | 4D embodied world model | 4D 生成 | 4D 生成对比 | Depth AbsRel 0.144/0.160；点图未报告 |
| DP | diffusion policy baseline | 动作扩散 | 闭环对比 | Mean 30.4/0.8 |
| TurboVLA | VLA baseline | 动作 | 闭环对比 | Mean 63.2/1.6 |
| starVLA | VLA baseline | 动作 | 闭环对比 | Mean 67.2/9.6 |
| Fast-WAM | world action model baseline | 动作 | 闭环对比 | Mean Easy 85.0 最高；Hard 0.2 |
| π0 | VLA flow model baseline | 动作 | 闭环对比 | clean-only 对照中 Mean Hard 21.4 最高；Grab Roller Hard 80。注意 Ego4Gen (Mixed) Hard mean 26.4 更高 |
| Ego4Gen | 本文 | G/M latent → FlowInverse | 主角 | Mean 46.6/15.2；Hard 四任务最高 |

**面试 Tip**：如果被问“Ego4Gen 和普通视频世界模型有什么区别”，记住三点：接口是 geometry-motion latent，不是 RGB latent；动作解码器 FlowInverse 很轻，且与世界模型解耦训练；卖点是 OOD Hard 泛化，不是 Easy SOTA。真机没做，别把它说成通用机器人策略。

## 8. 精读建议

- **值得精读原文的人**：
  - 做 world-action model、latent action interface 的研究者：§3.1–3.3 的三模态对齐和 stop-gradient 设计值得细看。
  - 要评估“几何 latent 做动作解码”可行性的工程师：重点看 §4.1 Table 1 和 §4.3 Table 5，注意实验条件差异。
  - 做仿真闭环 OOD 泛化的人：§4.3 的 Easy/Hard 设置和 Mixed 实验有参考价值。

- **建议章节路径**：
  先读 §3.1–3.2，搞清 T_V=96、T_GM=48、25/24/24 slots 和 motion offset；再看 §3.3 FlowInverse 的两阶段训练；然后读 §4.1 Table 1 和 §4.3 Table 5；§4.2 的 4D 生成指标可跳读，除非你做生成评测；§4.4 消融看 Table 6/7 即可。

- **不值得精读的理由**：
  如果你不做机器人学习、只关心真实机器人部署，或者已经熟悉 LAM 且不关心几何接口，读摘要即可。这篇没有真机实验，闭环也只在仿真；Easy 设置下 Fast-WAM 明显更强。

---

[← Back to Theory](./README.md)

**关键引用**：论文链接 https://arxiv.org/abs/2611.27513。本文所有数字均来自该 PDF 正文的 Table 1–7 与 §3–§4；未引用其他外部 URL。