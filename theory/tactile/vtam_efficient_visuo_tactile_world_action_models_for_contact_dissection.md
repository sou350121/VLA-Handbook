# VTAM：无需触觉预训练的高效视觉-触觉世界-动作模型 (VTAM: Efficient Visuo-Tactile World-Action Models for Contact-Rich Manipulation without Tactile Pretraining)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-11-07
>
> **论文**: VTAM: Efficient Visuo-Tactile World-Action Models for Contact-Rich Manipulation without Tactile Pretraining
> **链接**: https://arxiv.org/abs/2603.23481
> **核心定位**: 把触觉当作视频世界模型的额外视角，用未来视觉-触觉 latent 预测和 flow-force proxy 联合训练动作专家；不做触觉/动作预训练，也不在推理时生成视频。
>
> **证据说明**: 本文以 arXiv:2603.23481v2 [cs.RO] 05 Nov 2026 的 HTML/PDF 正文为证据。候选元数据摘要片段在 “RGB and” 处截断，未用其补写或替代正文主张；下文数字均取自该版本正文表格与章节。标注“论文实测”的是正文报告；“作者主张”是论文解释；“编辑分析”是条件性判断。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 论文实测：VTAM 仅用任务 demonstrations，在 UniVTAC 八任务平均 83.4%，与用 30K+ 小时数据预训练的 N0-TWAM（84.5%）和 N0-VTLA（83.1%）接近；真实机器人三任务平均 90.0%，Hanoi 75%。 |
| 适合精读 | 如果你在做触觉 VLA / 触觉 WAM、接触丰富操作、低延迟世界模型，重点看 §3 方法、§5.4 消融、§5.6 延迟。 |
| 可以跳过 | 如果你只关心语言指令泛化或开放世界 VLA，这篇不是通用 VLA 论文；它更偏任务级世界-动作模型。 |
| 落地可行性 | 中高：无需触觉预训练、推理不生成视频；但每任务单独训练，触觉反馈率被 1 Hz 策略推理限制。 |
| 主要风险 | 真实评测任务数有限；Stage I 并非所有任务都提升；flow-force proxy 不是真实力传感器。 |

💡 **X-Ray 开场**
这篇论文解决的是“触觉 VLA/WAM 要么不会用触觉，要么依赖大规模触觉预训练，要么推理太慢”的三重痛点。它的答案是：让触觉图像和 RGB 共享同一个视频 VAE 与视频 DiT，把“预测未来视觉-触觉 latent”作为自监督训练目标，再用光流导出的 flow-force proxy 把接触信号绑进动作专家。对 VLA 研究者的含义是：触觉不一定需要独立编码器和海量触觉预训练，可以先借视频世界模型的表征，再用少量任务 demonstrations 适配。

📍 **研究全景时间线**

```
2017  GelSight 高分辨率光学触觉传感器（论文参考文献[43]）
  → 2025  触觉 VLA / 视触觉策略出现（如 Tactile-VLA，论文参考文献[15]）
  → 2026  UniVTAC 仿真基准、N0 系列大规模视触觉预训练（论文参考文献[4][24][25]）
  → 2026  VTAM（本文）：共享视频 DiT，无触觉/动作预训练，Stage I 触觉世界建模 + Stage II 动作专家
```

## 1. 核心架构/方法总览

### 1.1 系统对比概览

| 模块/阶段 | 输入 | 输出 | 训练/参数 | 推理/时序 | 备注 |
|---|---|---|---|---|---|
| 共享视频 VAE | RGB 帧、触觉图像 | 每路 latent | 冻结；RGB 与触觉共用 | 每次观测编码 | 论文称 touch 不经过独立触觉编码器；摘要里的 “tactile encoder” 在正文实验中对应冻结视频 VAE 的触觉 latent |
| 视频 DiT backbone | 多视角 latent token | 未来视觉/触觉 latent 特征 | 由 Genie Envisioner GE-base 初始化；Stage I、II 微调 | 推理只做一次 forward，不做完整视频去噪 | 论文 §3.1：LTX-Video backbone，28 个 block |
| Stage I | 观测条件 c_obs、未来 latent 噪声 | 预测未来 RGB/触觉 latent | 50,000 training steps，lr 3e-4 | 只训练时使用 | 损失排除 conditioning frames |
| Action expert | backbone 特征 h、当前状态 c_k、联合噪声 y_τ | 动作、状态、proxy | 随机初始化；28 层；Stage II 与 backbone 联合微调 20,000 steps，lr 5e-5 | 少量 Euler steps 积分 flow | 与 backbone 并行，不是视频生成器 |
| Flow-force proxy | 触觉光流 | 3 维力样代理 | 由触觉图像计算，无需力传感器 | 作为 Stage II 辅助预测目标 | 不是真实力标签；论文称无需校准或学习估计器 |
| 默认输入配置 | 2 路 RGB + 1 路触觉 + 状态 | — | — | — | Hanoi 例外：1 路 RGB + Xense Photon 触觉（论文 §3、§5.2） |

> 架构归属提醒：不能把 N0-TWAM 的“专用触觉专家 + 30K+ 小时预训练”归给 VTAM；VTAM 只用视频模型预训练，触觉和动作均从任务 demonstrations 学习。也不能把 VTAM 说成“完全无预训练”：它初始化自 GE-base 视频模型。

### 1.2 关键机制

- **触觉作为额外视角，而不是额外编码器**：触觉帧进入预训练视频 DiT，和 RGB 共享 transformer 权重；论文明确说 touch 不增加 modality-specific transformer 参数（§3.1）。
- **Stage I 先做视觉-触觉世界建模**：用 flow matching 预测未来 RGB 与触觉 latent，让表征在动作学习前先适配接触动态；损失排除条件帧。
- **Stage II 联合去噪动作、状态、proxy**：动作专家与 backbone 端到端微调，目标 y0 = [a; f; s]，每个时间步含 7 维动作、3 维 flow-force proxy、16 维 proprioceptive state。
- **推理不生成视频**：部署时未来 latent 槽位设为 τ=1 的噪声，只做一次 backbone forward，缓存每层特征；动作专家用少量 Euler steps 生成 action chunk。
- **flow-force proxy 的作用要分开看**：论文表 5 中“冻结 VAE 触觉特征直接送 expert、但保留 proxy”仍在三个真实任务上 0%，说明 proxy 不能替代 Stage I 触觉世界建模；而“保留世界模型、去掉 proxy”成功率大降，说明 proxy 在动作学习阶段促进触觉使用。两者不是同一归因。

⚡ **Eureka Moment**：把触觉当成视频世界模型的“另一个视角”，用未来视觉-触觉 latent 预测做自监督，再把光流力代理塞进动作专家的联合去噪目标；推理时只跑一次 backbone，不生成视频。

### 1.3 信息流/架构图

```
默认配置（V=3：2 RGB + 1 tactile；Hanoi 为 V=2：1 RGB + 1 Xense Photon）
已观测帧 I_k^1..I_k^V, 状态 s_k
        │
        ▼
共享冻结视频 VAE E
        │  z_k^v = E(I_k^v)
        ▼
视频 DiT（28 blocks）
  ├─ in-view self-attention（各流共享）
  └─ cross-view attention（流间交换；Hanoi 只有 2 流）
        │
        ├── Stage I：预测未来视觉/触觉 latent
        │     条件 c_obs；未来帧加噪 x_τ；目标 v* = ε - x0
        │
        └── Stage II：与 28 层 action expert 联合微调
              输入：backbone 特征 h + c_k=[0_10; s_k] + 噪声联合目标 y_τ
              输出：动作 a、状态 s、flow-force proxy f
                       │
                       ▼
              推理：未来槽位设噪声 τ=1 → 一次 backbone forward → 缓存 h
                    → action expert 少量 Euler steps → action chunk
```

## 2. 数学核心

📌 **Napkin Formula**：

```
Stage I：L1 = E[ || vθ(xτ, τ | c_obs) - (ε - x0) ||² ]
Stage II：L2 = L_action + λ1·L_state + λ2·L_force，λ1=λ2=1
```

### 2.1 Stage I：未来视觉-触觉 latent 的 flow matching

论文用 flow matching 训练未来观测预测。对干净未来 latent x0、噪声 ε、flow time τ∈[0,1]：

```
xτ = (1 - τ)·x0 + τ·ε
v* = ε - x0
L_stage1 = E[ || vθ(xτ, τ | c_obs) - v* ||² ]
```

| 符号 | 含义 |
|---|---|
| x0 | 未来 RGB/触觉观测的干净 latent 序列 |
| c_obs | 条件观测，即已观测历史帧与状态 |
| ε | 噪声样本 |
| τ | flow time，不是 observation time k |
| vθ | 视频 DiT 预测的速度场 |
| v* | 回归目标 ε - x0 |

直觉：模型不是直接重建未来帧，而是学习“从噪声指向干净未来 latent”的速度场；条件帧不计损失，因此监督集中在未来预测段。

### 2.2 Flow-force proxy：从触觉光流到 3 维代理

论文在无接触参考图 I_ref 与当前触觉图 I_k^3 之间计算稠密光流 u_k=(u_x,k, u_y,k)，再做空间平均：

```
f_x,k = E_p[ u_x,k(p) ]
f_y,k = E_p[ u_y,k(p) ]
f_z,k = E_p[ ∇·u_k(p) ]
F_k^v = [f_x,k, f_y,k, f_z,k]^T
```

| 分量 | 含义 |
|---|---|
| f_x,k, f_y,k | 切向图像位移的均值 |
| f_z,k | 形变场的膨胀/收缩，用散度概括 |
| F_k^v | flow-force proxy，3 维；不是真实力 |

直觉：触觉形变场的方向和散度可以粗略反映接触加载；它不需要力传感器、校准或学习一个力回归器。但它是“力样代理”，不是物理力。

### 2.3 Stage II：动作、状态、proxy 的联合去噪

联合目标：

```
y0 = [a; f; s]      # 按预测 horizon 拼接
每个时间步：a ∈ R^7，f ∈ R^3，s ∈ R^16
c_k = [0_10; s_k]   # 动作与 proxy 位置零填充
```

损失：

```
L_action = E[ || vφ^a(yτ, τ | h, c_k) - v*^a ||² ]
L_state  = E[ || vφ^s(yτ, τ | h, c_k) - v*^s ||² ]
L_force  = E[ || vφ^f(yτ, τ | h, c_k) - v*^f ||² ]
L_stage2 = L_action + λ1·L_state + λ2·L_force
```

论文报告 λ1=λ2=1。动作、状态、proxy 是同一个联合去噪目标的不同切片；proxy 由生成动作的同一个 expert 学习，因此它既是辅助监督，也是动作表征的一部分。

### 2.4 时序口径：不要把未来 chunk 当历史记忆

论文的设置里必须区分三件事：

```
历史采样：K 帧，均匀间隔 Δt → 首末跨度 = (K - 1)·Δt
未来预测：从当前参考 t 预测 t+1 ... t+H → 终点距 t = H·Δt
动作 chunk：54-step action chunk 是执行 horizon，不是触觉历史窗口
```

- 论文 §4.1 说所有任务用 **nine-frame video chunks** 和 **54-step action chunks**；这不是说模型有 9 帧历史记忆或 54 步触觉记忆。
- 若假设 Δt=0.1 s，K=9 帧的首末跨度是 8×0.1=0.8 s；若从 t 预测 H=9 步，终点距 t 是 9×0.1=0.9 s。论文正文没有给出九帧视频的 Δt，所以不能把 9 帧直接换算成具体秒数；这里只是口径演示。
- 54-step action chunk 在 30 Hz 执行下是 54/30=1.8 s；论文图 8 也把 open-loop action chunk 标为 1.8 s。但 Put Bottle 使用 24-step execution horizon（§4），因此不能把 54-step 写成所有实验共有的动作窗口或记忆窗口。
- 论文没有提出跨 episode 的外部记忆机制；Stage I 的条件帧、Stage II 的 action chunk、部署时的缓存特征，是三件不同层面的事。

## 3. 带数字走一遍：玩具例子

> 以下为假设数值，用于验算 flow matching 的损失口径，不是论文实测。

假设一维干净未来 latent x0=2，噪声 ε=-1，flow time τ=0.25。则：

```
xτ = (1 - 0.25)×2 + 0.25×(-1) = 1.5 - 0.25 = 1.25
v* = ε - x0 = -1 - 2 = -3
```

若模型预测 vθ=-2.5，则：

```
L = || -2.5 - (-3) ||² = || 0.5 ||² = 0.25
```

若模型预测 vθ=-3，则 L=0。这个例子说明 Stage I 的监督是“速度场回归”，不是像素重建；条件帧不进入损失。再按 §2.4 的口径：若历史 9 帧、Δt=0.1 s，则历史跨度 0.8 s；若预测未来 9 步，则终点距当前 0.9 s。两者不能混为同一个“记忆窗口”。

## 4. 工程视角

| 项目 | 论文报告 | 工程含义 |
|---|---|---|
| 推理延迟（UniVTAC，L40S，BF16，batch=1） | VTAM：588 ms eager，145 ms compiled，每 action chunk；8 个任务模型平均 | 不做完整视频去噪是主要加速来源 |
| 对比延迟 | N0-TWAM：3,393 ms eager / 5,136 ms compiled；LingBot-VA：9,380 / 10,404 ms；Fast-WAM：548 / 127 ms；Tactile-VLA：364 / 230 ms；FTP-1：345 / 187 ms；N0-VTLA：335 / 225 ms | 未编译时论文称 VTAM 约比 N0-TWAM 快 6×、比 LingBot-VA 快 16×；编译后接近 Fast-WAM 的 127 ms |
| 真实机器人延迟 | 五个 Euler steps，141 ms/chunk；论文折算约 213 actions/s（对应 30 个执行动作） | 与仿真延迟的 10 Euler steps 口径不同，不能混用 |
| 控制频率 | 策略推理 1 Hz；数据采集与底层动作执行 30 Hz；触觉反馈率 1 Hz | 接触丰富任务仍受 1 Hz 触觉更新限制；底层 30 Hz 不等于触觉 30 Hz |
| 动作/视频 chunk | 训练用 54-step action chunks；Put Bottle 用 24-step execution horizon；视频用 nine-frame chunks | 不能把 54-step 或 9-frame 写成所有实验共有的记忆窗口 |
| 视频去噪 | 推理不做完整视频去噪；一次 backbone forward，未来槽位设噪声 τ=1，缓存每层特征 | 训练时仍有未来 latent 预测目标；推理时把“想象”压缩成特征 |

工程含义：VTAM 的卖点不是“完全不需要预训练”，而是“不需要触觉/动作预训练，且推理不生成视频”。它把触觉世界模型当作训练期正则和表征适配器，把部署成本压到一次 backbone forward + 少量 Euler steps。代价是：触觉反馈率只有 1 Hz；对需要高频接触反射的任务，仍需外部低层控制或更高频策略。

## 5. 数据与评测

### 5.1 预训练与训练设置

- 初始化：Genie Envisioner GE-base，LTX-Video backbone（§3.1）。
- 预训练范围：只有视频模型预训练；触觉与动作均不预训练。论文称 Stage I/II 只用任务 demonstrations。
- Stage I：50,000 training steps，lr 3×10^-4，训练视觉与触觉未来 latent 目标。
- Stage II：20,000 training steps，lr 5×10^-5，backbone 与 28 层 action expert 联合微调；VAE 冻结。
- 硬件与 batch：4×A100，每 GPU batch size 16，1,000 warmup steps，AdamW。
- 输入规格：192×256 图像，nine-frame video chunks，54-step action chunks。
- 真实任务 demonstrations：chip 100 条、peeling 61 条、wiping 105 条；Hanoi 100 条。每个设置训练单独的任务策略。

### 5.2 仿真评测：UniVTAC 八任务

论文在 UniVTAC 八个 contact-rich 任务上评测，每任务 **100 seeds**，官方 **300-step budget**，Insert HDMI 为 **600 steps**。VTAM 只用官方 demonstrations，不做触觉/动作预训练。

| 模型 | 触觉预训练 | 仿真平均成功率 | 备注 |
|---|---:|---:|---|
| π0.5（vision-only） | 否 | 41.4% | 视觉基线 |
| LingBot-VA | 否 | 31.4% | 全视频去噪，延迟高 |
| Fast-WAM | 否 | 48.0% | 视觉 only，跳过全视频去噪 |
| ACT + UniVTAC | 否 | 48.0% | 视触觉但无大规模触觉预训练 |
| Tactile-VLA | 否（按论文表 1 分组） | 45.4% | 触觉 VLA 基线 |
| Tactile-WAM | 否（按论文表 1 分组） | 30.0% | 论文称其报告 20 trials/task |
| FTP-1 | 是，约 3K 小时 | 64.8% | 触觉预训练基线 |
| N0-VTLA | 是，30K+ 小时 | 83.1% | 大规模视触觉预训练 |
| N0-TWAM（20%） | 是，部分数据 | 65.4% | 预训练数据比例消融 |
| N0-TWAM | 是，30K+ 小时 | 84.5% | 专用触觉专家，全视频去噪 |
| **VTAM** | **否** | **83.4%** | 仅任务 demonstrations；Put Bottle 用 24-step execution horizon |

论文实测的关键对比：VTAM 83.4% 高于所有无触觉预训练基线（最高 48.0%），也高于 FTP-1 的 64.8% 和 N0-TWAM 20% 的 65.4%，与 N0-VTLA 83.1%、N0-TWAM 84.5% 接近。诚实地说，它不是全面最好：Lift Bottle 为 91%，低于 π0.5 的 100%；Insert Tube 为 97%，低于 Fast-WAM 的 98%；Grasp Classify 与 Tactile-VLA 同为 100%。

### 5.3 真实机器人设置

**平台 A：xArm6 + GelSight Mini + 两路 RealSense D455**

- 评测：80 trials = 20 chip + 20 peeling + 40 wiping（flat 20 + tilted 20）。
- 训练：每任务单独策略；chip 100、peeling 61、wiping 105 条 demonstrations。
- 结果（论文表 2）：VTAM 为 chip 90%、peel 85%、wipe 95%，三任务未加权平均 90.0%。

| 模型 | Chip | Peel | Wipe | Avg. |
|---|---:|---:|---:|---:|
| Genie Envisioner | 0% | 0% | 2.5% | 0.8% |
| π0.5（Vision） | 10% | 0% | 0% | 3.3% |
| π0.5 + Tactile | 5% | 0% | 0% | 1.7% |
| RDP | 0% | 0% | 50% | 16.7% |
| Tactile-VLA | 0% | 20% | 10% | 10.0% |
| **VTAM** | **90%** | **85%** | **95%** | **90.0%** |

论文还报告 wiping 的触觉统计（表 3）：VTAM 的 stable contact ratio 为 0.945，高于 RDP 的 0.750 和 π0.5 的 0.317；TV/frame 为 0.144，略低于 RDP 的 0.153。作者据此主张更持续接触、更少触觉抖动；这是传感器统计，不是单纯成功率。

**平台 B：Flexiv Rizon 4 + Xense Photon + 单 wrist camera（Hanoi Disk Insertion）**

- 100 条 demonstrations，20 trials；VTAM 15/20=75%。
- 对比：FTP-1 55%、π0.5 25%、Genie Envisioner 5%、Tactile-VLA 0%。
- 注意：Hanoi 的策略仍用 100 条 Xense demonstrations 训练；论文所谓“tactile encoder zero-shot across sensors”主要体现在冻结视频 VAE 的触觉 latent 探针上，不应解读为策略零样本跨传感器迁移。

### 5.4 消融：Stage I、冻结 backbone、proxy 各有条件

**真实机器人组件消融（表 5）**

| 变体 | 触觉特征 | Chip | Wipe | Hanoi |
|---|---|---:|---:|---:|
| Tactile VAE → expert | Frozen VAE（保留 proxy） | 0% | 0% | 0% |
| w/o flow-force proxy | 世界模型（去 proxy） | 10% | 10% | 55% |
| VTAM | 世界模型 + proxy | 90% | 95% | 75% |

解读：冻结 VAE 触觉特征直接送入 expert、即使保留 proxy 也全 0，说明“绕过触觉世界建模”不可行；去掉 proxy 但保留世界模型，成功率大幅下降，说明 proxy 在动作学习阶段有作用。两者应分开归因。

**Stage I 在 UniVTAC 的影响（表 6）**

- 完整 VTAM：83.4%；w/o Stage I：65.8%。
- 多数任务提升，例如 Lift Bottle 27%→91%，Put Bottle 54%→100%，Lift Can 49%→75%，Pull-out Key 72%→92%。
- 例外：Insert Hole 67%→95%（w/o Stage I 更好），Grasp Classify 饱和。作者把 Insert Hole 留给未来研究。

**Backbone 适配（表 7，节选）**

| Stage I | Stage II backbone | Lift Bottle | Put Bottle | Insert Hole |
|---|---|---:|---:|---:|
| × | frozen | 4% | 6% | 53% |
| × | fine-tuned | 27% | 54% | 95% |
| ✓ | frozen | 18% | 10% | 0% |
| ✓ | fine-tuned（VTAM） | 91% | 100% | 67% |

这组结果说明：视频 backbone 不能只当固定特征提取器；Stage I 与 Stage II 端到端微调互补，但 Insert Hole 上 Stage I 反而伤害表征。

### 5.5 零样本触觉 latent：是 VAE 探针，不是策略零样本

论文图 11 用 **冻结视频 VAE** 的触觉 latent 做 t-SNE 和线性探针；该 VAE “never trained on touch”。线性探针区分 loaded/unloaded 的 balanced accuracy 为真实 90%、仿真 91%；proxy Spearman 相关为 0.76/0.84；原始像素为 88%/89% 和 0.73/0.83。仿真中 episode progress 的 Spearman 总体 0.74，八个任务中五个在 0.75–0.97。

编辑分析：这支持“预训练视频 VAE 的视觉表征已经保留接触相关信息”，但不等于 VTAM 的策略可以零样本迁移到新传感器或新任务；Hanoi 仍需该平台 demonstrations。

## 6. 能力与失败模式

**能做什么**

- 在四个真实任务、两个平台、两种触觉传感器上取得高于论文所测基线的成功率；尤其 chip 与 wiping 的接触建立/保持明显优于视觉基线。
- 在 UniVTAC 八任务上以少量任务 demonstrations 接近大规模触觉预训练模型。
- 推理不生成视频，compiled 145 ms/chunk；真实机器人 141 ms/chunk（五个 Euler steps）。
- 共享视频 VAE 的触觉 latent 在不同传感器上有结构分离；可作为触觉世界建模的起点。

**不能做什么 / 风险**

- 不是通用语言指令 VLA：方法输入是 RGB、触觉、proprioception，没有语言指令；真实实验每任务单独训练策略。
- 只在桌面接触丰富操作上验证；不能外推到移动、双臂、人形或开放世界任务。
- 触觉反馈率 1 Hz；对需要高频反射的接触任务可能不足。
- Stage I 并非总是有益：Insert Hole 上 w/o Stage I 更高；冻结 Stage I backbone 在 Insert Hole 为 0%。
- flow-force proxy 不是力传感器；它依赖无接触参考图与光流，接触模式变化时相关性可能下降。
- 真实评测每个设置 20 trials 或 40 wiping trials；统计置信区间有限，不能把 90% 当作普适上限。
- 部分 baseline 数字来自先前工作而非同一协议重跑；Tactile-WAM 报告 20 trials/task，比较时需保留条件。

### 6.1 隐含假设

- 触觉图像与 RGB 在视频 VAE/DiT 中共享统计结构；视频预训练表征可迁移到触觉形变。
- 存在可用的无接触参考图，光流散度能粗略代理接触力。
- 1 Hz 策略推理 + action chunk 开环执行足以覆盖所测接触任务。
- 每任务约 100 条 demonstrations 足以适配 Stage I/II；大规模触觉数据的主要价值可能在分布外泛化，而非分布内边际提升（作者在结论中如此主张）。
- 传感器安装与触觉图像外观在任务内稳定；跨传感器只验证了冻结 VAE latent 探针与 Hanoi 任务训练，不是广泛零样本策略迁移。

## 7. 与相关工作对比

| 方法 | 触觉使用 | 触觉/动作预训练 | 推理全视频去噪 | 类别/设置 | 论文报告数字 |
|---|---|---|---|---|---|
| π0.5（Vision） | 无 | 视觉 VLA 预训练 | 否 | 视觉 only | UniVTAC 41.4%；真实 chip 10% |
| Fast-WAM | 无 | 视觉世界模型 | 否 | 视觉 WAM | UniVTAC 48.0%；127 ms compiled |
| Tactile-VLA | 有 | 按表 1 分组为无触觉预训练 | 否（表 8 勾选） | 触觉 VLA | UniVTAC 45.4%；230 ms compiled |
| Tactile-WAM | 有 | 按表 1 分组为无触觉预训练 | 未在表 8 列 | 触觉 WAM | UniVTAC 30.0%（20 trials/task） |
| FTP-1 | 有 | 约 3K 小时触觉预训练 | 否（表 8 勾选） | 触觉策略 | UniVTAC 64.8%；187 ms compiled |
| N0-VTLA | 有 | 30K+ 小时视触觉预训练 | 否（表 8 勾选） | 触觉 VLA | UniVTAC 83.1%；225 ms compiled |
| N0-TWAM | 有，专用触觉专家 | 30K+ 小时视触觉预训练 | 是 | 触觉 WAM | UniVTAC 84.5%；3,393/5,136 ms |
| **VTAM** | 有，共享视频 DiT | **无触觉/动作预训练**；仅视频模型初始化 | **否** | 视触觉 WAM | UniVTAC 83.4%；588/145 ms |

> 面试 Tip：被问“VTAM 为什么不用触觉预训练还能打”，一句话答：它不是训练一个独立触觉编码器，而是把触觉作为视频 DiT 的额外视角做未来 latent 预测，再用 flow-force proxy 把接触信号绑进 action expert 的联合去噪；推理只跑一次 backbone，不生成视频。

## 8. 精读建议

- **值得精读的人**：做触觉 VLA/WAM 的研究者；要在低数据、低延迟下部署接触丰富操作的工程师；研究多模态世界模型共享 backbone 的人。
- **建议章节路径**：先读 §3.1–§3.3 搞清 Stage I/II 与 proxy；再看 §5.4 消融确认各组件条件性收益；最后看 §5.6 延迟与 §4.1 频率口径。Related Work 可快速扫。
- **不值得精读的理由**：如果你不做机器人操作、只关心语言指令泛化，或需要高频触觉闭环控制，这篇的覆盖范围不够；读摘要与表 1/表 2 即可。

---
[← Back to Theory](./README.md)

**关键引用**：[论文](https://arxiv.org/abs/2603.23481)（本文所有数字与主张来源；未引用其他 URL）