# 一个先验，两种动力学：复用视频扩散 Transformer 做自监督潜动作学习 (One Prior, Two Dynamics: Repurposing Video Diffusion Transformers for Self-Supervised Latent Action Learning)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-11-21
>
> **论文**: One Prior, Two Dynamics: Repurposing Video Diffusion Transformers for Self-Supervised Latent Action Learning
> **链接**: https://arxiv.org/abs/2611.26617
> **核心定位**: 把同一个预训练视频 DiT 分别改造成 IDM 和 FDM，通过控制两者的信息访问，从无动作标注视频中学出可迁移的 32-D latent action；相比“从零学 IDM”或“只把视频生成器当 FDM”的做法，它让两种动力学共享同一个视频先验。

**版本说明**：本文依据 arXiv:2611.26617v1（2026-11-19，cs.CV）正文写作。来源记录中的候选摘要为截断片段（止于 “We initialize both models from the”），未发现与正文的实质冲突；本文不以该片段补写正文未出现的主张。文中所有数字均来自该版本正文的 Table/Fig，作者主张与我的条件性分析会分开标注。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 同一个预训练视频 DiT 的先验可以同时支撑 inverse 和 forward dynamics；关键不在换更大模型，而在控制两个模型各自能访问什么信息。 |
| 适合精读 | 做 latent action、视频世界模型、视频生成先验复用的人；重点看 §3.3–3.5、§5.2–5.4。 |
| 可以跳过 | 只关心端到端 VLA 策略、真实机器人闭环、高频控制的人；这篇没有策略层验证，也没有真实机器人执行。 |
| 落地可行性 | 中低。2×5B 模型 + LoRA，训练 16 GPU、5k updates；FDM 推理是视频扩散采样，不适合高频闭环。 |
| 主要风险 | 评测以 offline linear probing 和生成代理指标为主；只用高噪声训练会压低生成质量，且 probing 未一致优于 mixed noise；latent action 可解释性弱。 |

💡 **X-Ray 开场**

这篇论文问三个问题：视频里有“世界怎么变”，但没有“什么动作导致它变”，latent action learning 想从视觉变化里反推动作，可现有方法大多从零学 dynamics；预训练视频生成器里已经有时序先验，能不能同时拿来当 IDM 和 FDM？作者发现可以，但必须限制信息通路——IDM 的 action query 只读 transition 的两个端点帧（video tokens 仍因果注意前序帧），FDM 必须去掉 clean-history 捷径并强调高噪声，否则 latent action 会被绕过。对 VLA 研究者意味着：视频生成先验不只是特征提取器或生成器，还可以作为一种“动力学预训练”被拆成两个角色复用；但 probing 分数高不等于可控性好，必须看下游生成评测。

📍 **研究全景时间线**

```
2019 Edwards et al. 从观测中模仿 latent policy
  → 2024 Genie / "Learning to act without actions" 从视频学可控世界模型
  → 2025 LAPA 用 latent action 预训练 VLA；Wan 系列视频生成模型开源
  → 2026 Olaf-World 用外部视频表示对齐 latent action；DreamDojo 大规模人类视频机器人世界模型
  → 2026 本文 O2LaF：同一个视频 DiT 先验同时做 IDM 和 FDM
```

（节点年份取自论文参考文献；本文不展开整张参考文献表。）

## 1. 核心架构/方法总览

### 1.1 系统对比概览

| 模块 | 初始化 | 角色 | 输入 | 输出 | 训练/推理差异 |
|------|--------|------|------|------|----------------|
| IDM | Wan2.2-TI2V-5B + rank-128 LoRA + action queries + Gaussian head | 逆动力学：从 transition 推断 latent action | `x̃_IDM_{≤i+1}`，`t_IDM` | 每个 transition 一个 `z_i ∈ R^32` | 单次前向；主实验 `t_IDM = 0`；推理用后验均值 `μ_i` |
| FDM | 同一个 Wan2.2-TI2V-5B + rank-128 LoRA + action embedder | 前向动力学：预测动作的视觉后果 | clean `x_0`、未来 noisy states、`z̄`、`t_FDM`、固定文本 `c_0 = "a video"` | 所有未来 state 的 velocity `v̂_i` | frame-causal joint flow matching；`t_FDM` 混合高噪声；推理 Euler 积分 `t=1→0` |
| Video VAE / text encoder | 冻结 | 编码视频、文本 | RGB / 文本 | latent / embedding | 不训练 |
| Two-stage FDM | 冻结 IDM 后另训一个双向 FDM | 提升生成质量 | 81 RGB frames / 20 latent actions | 视频 | 全量 fine-tune，832×480 |

注意：IDM 和 FDM 是**两个独立适配的模型副本**，共享的是同一个初始化 checkpoint，不是共享权重。架构归属要逐模型看：Wan2.2 同时用于 IDM 和 FDM；V-JEPA2、VideoChat3、ST-Transformer 只在 §5.3 中作为 **IDM backbone 替代**，其中 VideoChat3 在 Table 4 中没有 world model 数字，不能把它的属性合并到生成式 FDM 上。

### 1.2 关键机制

- **为什么用同一个先验**：IDM 需要 transition reasoning，FDM 需要预测视觉变化，两者都依赖时序结构；视频 DiT 的大规模预训练正好提供这种结构。
- **IDM 的非对称 attention**：每个 action query `q_i` 只读 `(x_i, x_{i+1})` 两个端点；query 之间不互相注意，video token 也不注意 query。这样 `q_i` 是只读的 transition register，不会变成辅助通信通道。
- **FDM 的 frame-causal joint prediction**：把所有未来 state 放在一次前向里联合预测，并禁止 `x_i` 注意 `x_{>i}`。若用 autoregressive 且历史帧是 clean 的，模型可以复制上一帧绕过 latent action；joint prediction 去掉了这个 clean-history 捷径。
- **高噪声采样**：低/中噪声下，noisy future 仍保留大量目标信息，FDM 可以不依赖 `z`；高噪声迫使它依赖动作。论文给出端点直觉：`t_FDM=1` 时 flow matching 等价于从 `x_0` 和 `z̄` 重建未来 latent。
- **变分瓶颈**：32-D 连续高斯，`β = 10^-4`，用 KL 把后验拉向标准高斯，限制动作通道的信息量，避免 `z` 编码外观。

⚡ **Eureka Moment**：不是“用视频模型当 world model”，而是“同一个视频 DiT 先验可以被切成两个信息访问受限的角色：一个用 action query 只读 transition 两端、但视频 token 仍因果读取前序帧来做逆动力学，一个以 clean 初始帧、噪声化未来状态和 latent action 做因果联合前向预测，两者通过 32-D 瓶颈端到端耦合”。一句话：**先验共享，信息通路分离**。

### 1.3 信息流/架构图

```
无标注视频 clip V
  → frozen VAE → latent states x_0 ... x_K
  → IDM（Wan2.2 + LoRA + action queries）
       对每个 transition (x_i, x_{i+1}) 输出 z_i ~ N(μ_i, σ_i²)
  → z̄ = [0, z_0, ..., z_{K-1}]
  → FDM（Wan2.2 + LoRA + action embedder）
       输入 clean x_0 + noisy future x_1..x_K + z̄ + t_FDM
       输出 v̂_1..v̂_K
  → flow matching loss + KL
  → 梯度经 z_i 回传 IDM

推理：
  源视频 → IDM → z̄
  目标首帧 + z̄ → FDM → Euler 采样 → 生成视频
```

## 2. 数学核心

📌 **Napkin Formula**（一行抓住本质）：

```
L = E[ (1/K) Σ_i w(t_FDM)·||v̂_i - (ε_i - x_i)||² ] + (β/K) Σ_i KL[q_φ(z_i|·) || N(0,I)]
```

直觉：用高噪声前向动力学逼 IDM 把“能解释未来变化”的信息压进 32 维动作。

**目标**：给定无动作标注视频，为每个 transition 学一个 compact `z_i`，使得 FDM 在只看 `x_0` 和 `z̄` 时能预测未来。

**耦合形式**（论文 Eq. 1）：

```
z_i ~ q_φ(z_i | x̃_IDM_{≤i+1}, t_IDM)
v̂ = f_θ(x̃_FDM, t_FDM, c_0, z̄)
```

**变分头**（论文 Eq. 2）：

```
[μ_i, log σ_i²] = g_φ(LN(h_i))
q_φ(z_i | ·) = N(μ_i, diag(σ_i²))
z_i = μ_i + σ_i ⊙ ξ_i,  ξ_i ~ N(0, I)
```

**FDM 输入**（论文 §3.4）：

```
x̃_FDM = [x_0, x_1^{t_FDM}, ..., x_K^{t_FDM}]
x_i^{t_FDM} = (1 - t_FDM)·x_i + t_FDM·ε_i
```

**损失**（论文 Eq. 3）：

```
L = E[ (1/K) Σ_{i=1}^K w(t_FDM)·||v̂_i - (ε_i - x_i)||² ]
    + (β/K) Σ_{i=0}^{K-1} D_KL[ q_φ(z_i|·) || N(0,I) ]
```

> 符号与本文一致：`K` 是 clip 内未来 latent states 数；`d_z = 32`；`β = 10^-4`；主实验 `t_IDM ~ δ_0`，`t_FDM ~ 0.7 π_base + 0.3 π_high`，`supp(π_high) ⊆ [0.9, 1)`；`c_0 = "a video"`。`w(t_FDM)` 的具体函数形式正文未展开，标 TODO。

**直觉**：
- KL 把 `z` 拉向标准高斯，32 维远小于 `D`，形成 bottleneck。
- 高噪声下 noisy future 几乎不含目标信息，FDM 只能靠 `z`；这给 IDM 的梯度信号更“干净”。
- `t_FDM=1` 时，`v* = ε - x`，定义 `x̂ = ε - v̂`，则 `||v̂ - v*||² = ||x̂ - x||²`，等价于重建未来 latent（论文 §3.4 的推导）。

**时间跨度核算（条件性分析）**：论文 Table 6 给出 E2E 训练 41 RGB frames / 10 latent actions，two-stage 81 frames / 20 latent actions。若按等间隔且首帧为参考，则每个 action 对应 4 帧，E2E 的终点距参考是 `10×4 = 40` 帧（加参考帧共 41），two-stage 是 `20×4 = 80` 帧（共 81）。这是对配置的算术解读，不是论文对 VAE 压缩率的显式声明。要点是：这里的 `K` 是**单个训练 clip 的未来 chunk 长度**，不是历史上下文窗口，也不是跨 clip 的记忆窗口；论文没有提出 memory 机制。均匀采样 `K` 个帧时首末跨度是 `(K-1)×Δt`；但本文是从当前参考 `t` 预测 `t+1` 至 `t+K`，终点距参考是 `K×Δt`，两个公式不要混。

## 3. 带数字走一遍：玩具例子

> 以下数值为**假设**，仅用于验算公式，不代表论文实测。论文实际 `d_z=32`，这里缩到 2 维便于手算。

假设取 `K=3` 个未来 latent states，`d_z=2`，VAE 时间压缩 `Δt=4` 帧。IDM 输出：

```
μ_0 = (0.5, -0.2), σ_0 = (0.1, 0.1)
μ_1 = (0.6, -0.1), σ_1 = (0.1, 0.1)
μ_2 = (0.7,  0.0), σ_2 = (0.1, 0.1)
```

训练时采 `ξ=(0,0)`，则 `z_i = μ_i`。

**KL 验算**：一维高斯对标准高斯的 KL 为

```
D_KL = 0.5·(μ² + σ² - 1 - log σ²)
```

对 `μ=0.5, σ=0.1`：`0.5·(0.25 + 0.01 - 1 + 4.60517) = 1.9326`。
对 `μ=-0.2, σ=0.1`：`0.5·(0.04 + 0.01 - 1 + 4.60517) = 1.8276`。
`z_0` 的二维 KL 合计 `3.7602`；`z_1` 为 `1.9876 + 1.8126 = 3.8002`；`z_2` 为 `2.0526 + 1.8076 = 3.8602`；三组总和约 `11.42`（按 z_0..z_2，对应 Eq. 3 的 i=0..K-1）。按 Eq. 3 的 `(β/K)Σ`，乘 `β=10^-4` 并除以 `K=3` 后 KL 项约 `0.000381`，相对 flow matching loss 很小。这说明在该设置下 KL 是弱正则，不是强约束；信息瓶颈主要靠 32 维维度和 FDM 的预测压力。

**时间跨度验算**：`K=3` 个 action，`Δt=4`，终点距参考 `3×4=12` 帧；若把 `x_0..x_3` 看作 4 个采样帧，首末跨度 `(4-1)×4=12` 帧，一致。若误把 `K=3` 当成历史窗口，就会得出错误结论——它只是未来 chunk 长度。

**高噪声端点验算**：设某一维 `x=0.2`，`ε=1.0`，则 `v* = 0.8`。若预测 `v̂=0.6`，则 `x̂ = ε - v̂ = 0.4`，误差 `(0.6-0.8)² = 0.04`，也等于 `(0.4-0.2)² = 0.04`；若 `v̂=0.8`，误差为 0。这验证了 `t_FDM=1` 时 flow matching 等价于重建未来 latent。

## 4. 工程视角

| 项目 | E2E 训练 | Two-stage FDM |
|------|----------|----------------|
| Backbone | Wan2.2-5B ×2 | Wan2.2-5B |
| 可训练权重 | LoRA-128 + heads | 全量 FDM |
| RGB frames | 41 | 81 |
| Latent actions | 10 | 20 |
| Global batch | 256 | 128 |
| GPUs | 16 | 16 |
| 峰值显存 | 65.49 GiB | 55.08 GiB |
| Updates | 5k | 5k |
| Learning rate | 5e-5 | 5e-5 |
| Warmup steps | 50 | 100 |

（来源：论文 Table 6。）

**工程含义**：
- **吞吐/延迟**：IDM 是单次前向，适合在线动作推断；FDM 是视频扩散采样，需要 Euler 积分 `t=1→0`，论文未给采样步数（TODO），延迟远高于 IDM。不要把 FDM 当成高频控制器。
- **显存**：E2E 同时驻留两个 5B 副本，峰值 65.49 GiB；若部署只要 IDM，理论上可只保留一个 5B + LoRA，成本减半——这是条件性推断，论文未给 IDM-only 数字。
- **步数/单位**：论文的 “5k updates” 是优化器更新步数，不是 epoch；`t_IDM`/`t_FDM` 是 flow timestep，不是 RL time step；`K` 是 clip 内未来 chunk 长度，不是跨 clip 记忆窗口。
- **抖动/量化**：连续高斯 latent，无量化；推理用 `μ_i`，确定性。KL 权重小，latent 空间未必严格各向同性，论文未分析。
- **trade-off**：high-noise-only 训练相比 mixed noise 明显降低生成质量，而 probing 没有一致提升（Table 5：High/High 的 1st-P k=1/50 为 72.28/85.76，Mixed/All 为 72.81/87.95；1st-P ViPE Trans 2.419 vs 1.937）；two-stage 可恢复生成质量，但要多训一个双向 FDM。

## 5. 数据与评测

**预训练**：IDM/FDM 都从 Wan2.2-TI2V-5B 初始化；VAE 和 text encoder 冻结；LoRA 适配。这里的“预训练”是视频生成预训练，不是动作预训练。

**Latent action 训练数据**：Navigation 用 MiraData，Manipulation 用 EgoDex（人类 egocentric 演示）；训练不使用动作标注。Manipulation 只用了 DreamDojo 大规模 egocentric 数据中的 EgoDex 子集（作者强调这一点）。

**离线评测设置**：
- MIND：四个 ternary controls（两平移、两旋转）的 joint exact-match；label budgets 1–50 videos；独立、class-balanced logistic regression probes。
- In-lab GR-1：DreamDojo 的 In-lab Eval benchmark；Ridge regression 预测双臂 wrist translation/rotation；报告 EEF `R²`；label budgets 1–32 episodes per task。
- Zero-shot navigation：24 条 reference trajectories（12 first-person、12 third-person）→ 50 和 20 个新视觉上下文，共 840 次 cross-context generations；ViPE 估计轨迹，Sim(3)-aligned translation NRMSE 和 rotation RPE；VBench 评视觉质量。
- Zero-shot manipulation：85 对视频、20 个任务；human↔robot 直接 transfer 只做定性；定量用 self-reconstruction 和 cross-embodiment cycle consistency（PSNR/SSIM/LPIPS）。

**真实机器人/仿真边界**：主评测是离线视频数据集上的 probing 和生成，不是闭环仿真，也不是真实机器人执行。Appendix C.2 的 MuJoCo/robosuite GR-1 是“FDM as policy verifier”的定性案例：从单张真实照片重建场景，比较 FDM rollout 和 physics rollout；论文明确说“without executing the trajectory we cannot tell which prediction is right”。不能把它写成真实机器人闭环验证。

**关键数字（论文实测）**：
- EEF `R²`：O2LaF 0.277 vs DreamDojo 0.149（GR-1，§4.2）。
- Table 1（1st-P / 3rd-P）：O2LaF Img 0.698/0.682，Trans 1.289/0.511，Rot 0.786/0.330；Olaf-World 0.690/0.648，1.627/0.985，1.300/0.701；AdaWorld 更差。
- Table 2：Human Self PSNR 26.72 vs DreamDojo 23.38；Cross-cycle PSNR 24.30 vs 22.61；Robot Self 19.92 vs 18.73。
- Table 3(a)：Head-only 1st-P `k=1/50` 43.88/39.33；LoRA-128 72.81/87.95；3rd-P LoRA-16 在 `k=1` 是 55.12，高于 LoRA-128 的 52.08——不要简化成“rank 越大越好”。
- Table 3(b)：固定 1.3B IDM 时，1.3B FDM 的 1st-P 为 67.14/80.76、3rd-P 为 59.68/89.52；换成 14B FDM 后为 72.72/85.98、61.73/92.19，支持 FDM scaling 提升 probing；5B/5B 是 Wan2.2 参考（IDM/FDM 均不同），不能用来推断同一 FDM scaling 趋势，且其 3rd-P `k=1` 52.08 低于 1.3B/14B 的 61.73。
- Table 3(c)：Causal joint 1st-P 72.81/87.95，3rd-P 52.08/94.37；AR (`h=1`) 55.45/70.13，45.79/79.49。
- Table 4：V-JEPA2 MIND 88.62/96.14，Color `R²` 0.514/0.816/0.742；Wan2.2 87.95/94.37，Color `R²` -0.297/0.081/0.223。作者解释：V-JEPA2 probing 更高但 source-appearance leakage 更明显。
- Table 5：E2E Mixed/All 1st-P VBench Img 0.649、Trans 1.937；High/High Trans 2.419；Two-stage E2E-init Img 0.711、Trans 1.260。注意 two-stage 行的 probing 分数是 gray/shared，不参与排名，不能当作 two-stage 的独立提升。
- Table 7：Mira only MIND 87.95/94.37、EEF left/right 0.135/0.257；EgoDexMix 76.70/91.66、0.227/0.386；Joint 70/30 85.94/93.96、0.163/0.341。
- Fig. 9：5×5 `(t_IDM, t_FDM)` sweep；navigation 两个视角偏好 `(0.3, 0.98)`，manipulation 不同手臂有不同最优。
- Appendix C.1：CameraBench 上 O2LaF-Scratch (270M) +44.9 AP，O2LaF-DiT (5B) +9.2 AP（Table 8）。

## 6. 能力与失败模式

**能做什么**：
- 从无动作标签视频学 latent action，离散 navigation 和连续 manipulation 都能 linear probe。
- zero-shot cross-context：同一 `z` 序列可条件到不同首帧，迁移相机运动/机器人运动。
- cross-embodiment：human↔robot cycle reconstruction 优于 DreamDojo（Table 2）。
- FDM 可当 action-conditioned world model；two-stage 可提升生成质量。
- IDM 可当 motion annotator（CameraBench 增益，Appendix C.1）。

**不能做什么/失败模式**：
- **没有策略层验证**：论文自己把 “efficient planning and closed-loop control” 列为 future work；Appendix D 明确 dynamics 为视觉预测优化，不直接暴露可执行动作或 reward。
- **生成质量与动作学习 trade-off**：high-noise-only 训练相比 mixed noise 压低 VBench/ViPE，且 probing 没有一致提升；需要 two-stage 额外训练。
- **latent action 可解释性弱**：只能靠 reference video inspection；没有 compositional/appearance-invariant 保证。
- **数据域依赖**：MiraData-only 在 navigation 最强，EgoDexMix 在 robot probing 最强，没有一个配置全面最优（Table 7）。
- **评测以 offline 为主**：zero-shot manipulation 直接 transfer 只做定性；cycle reconstruction 是代理指标，不是任务成功率。
- **计算成本高**：2×5B，视频扩散采样慢。
- **probing 不是充分指标**：V-JEPA2 的 probing 更高但 appearance leakage 更严重。

### 6.1 隐含假设

- 假设视频 DiT 的时序先验可迁移到 action inference；Fig. 3 的 cross-attention 诊断只是定性的，没有定量验证。
- 假设 32-D 高斯 + 小 `β` 足以形成 bottleneck；正文没有 `d_z` 的 sweep。
- 假设 frame-causal + high-noise 能阻止 shortcut；但论文自己说 maximal noise 不总是最优，高方差 velocity target 会削弱监督。
- 假设 human egocentric 视频和 robot 视频共享运动结构；cross-embodiment 结果支持，但只是 cycle consistency。
- 假设 MIND 四轴 ternary 和 In-lab EEF 能代表真实控制；覆盖范围有限。
- 假设同一个先验初始化两个模型优于两个不同先验；Table 4 有不同 backbone 对比，但不是完全 controlled 的同架构不同先验实验。

## 7. 与相关工作对比

| 方法 | 先验进入方式 | IDM | FDM | 动作空间 | 评测 |
|------|--------------|-----|-----|----------|------|
| AdaWorld | 从零/视频世界模型 | 从零 | 从零 | latent | navigation |
| Olaf-World | 外部视频表示对齐（辅助监督） | 从零 | —（外部表示仅作辅助监督） | latent | navigation |
| VideoWorld2 | 视频生成器当 FDM | 从零 | 预训练视频生成器 | latent | manipulation |
| DreamDojo | 大规模人类视频世界模型 | — | — | — | manipulation |
| LAPA-DINOv3 | 冻结视觉特征 | 从零/冻结特征 | — | latent | manipulation |
| VILLA-X | VLA 中的 latent action | — | — | — | manipulation |
| O2LaF | 同一个视频 DiT 同时做 IDM/FDM | 视频 DiT + action queries | 视频 DiT + causal joint | 32-D continuous | navigation + manipulation |

（表中“—”表示论文未在该行给出对应信息；不要把这些基线的方法细节外推。）

**面试 Tip**：被问“这篇和 Olaf-World/VideoWorld2 的区别”时，答：Olaf-World 用外部视频表示对齐 latent action，但 IDM 仍从零学；VideoWorld2 把视频生成器当 FDM，但 IDM 从零，且标准扩散 recipe 下生成器可能忽略 action；O2LaF 的关键是**同一个视频 DiT 初始化两个 dynamics**，再用 causal 信息访问控制让 FDM 必须依赖 action。

## 8. 精读建议

- **值得精读的人**：做 latent action / 视频世界模型 / 视频先验复用的人；想做无动作标签预训练再接 VLA 的人；关心 information bottleneck 和 causal factorization 的人。
- **建议章节路径**：先读 §3.2–3.5（方法核心）→ §5.2（噪声/时序消融）→ §4.2–4.3（评测协议和数字）→ Appendix B.2（数据组成）→ Appendix D（局限）。§2 的参考文献细节可跳。
- **不值得精读的理由**：只关心端到端 VLA 策略、真实机器人闭环、高频控制的人；这篇没有策略层验证和真实机器人执行，读摘要和 §6 即可。

---
[← Back to Theory](./README.md)

**关键引用**：[arXiv:2611.26617](https://arxiv.org/abs/2611.26617) · [HTML 正文](https://arxiv.org/html/2611.26617)