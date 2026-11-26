# DexWAC：把世界模型、Actor 与 Critic 绑进一个共享前缀 (DexWAC: A Unified World–Actor–Critic Model for Dexterous Robot Manipulation)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-11-26
>
> **论文**: DexWAC: A Unified World–Actor–Critic Model for Dexterous Robot Manipulation
> **链接**: https://arxiv.org/abs/2611.32654
> **核心定位**: 用一次机器人预训练同时学习 actor、动作条件 world model 和 TD critic，让未来端点预测与价值监督共享表示，并可在真实 rollout 上适配后用于想象式 policy improvement。
>
> 版本说明：本文依据 `arXiv:2611.32654v1`（2026-11-24）的 HTML 正文（对应 PDF v1）写作；候选摘要为截断文本，止于 “learn dynamics and valu”，未用于补写。正文摘要与候选摘要已覆盖部分未发现实质冲突，但候选摘要缺少方法与结果后半段；以下数字与主张均以正文表格、公式和附录为准。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 作者主张：把动作条件视觉预测、奖励/继续预测和 TD 价值学习放进同一个共享前缀，可以同时提升策略表示与后续 policy improvement 的效率。 |
| 适合精读 | 如果你在做 world model + RL 的机器人策略改进、接触丰富灵巧操作、或 VLA 的 test-time 价值筛选，重点看 §4.1–4.3、§5.3–5.6 和 Appendix A/C。 |
| 可以跳过 | 如果你只关心纯 VLA 架构扩展、长程记忆机制或大规模多机器人预训练，这篇距离中等；论文明确排除了 3 个长程记忆任务，也未验证大规模机器人预训练。 |
| 落地可行性 | 中。世界模型只预测 chunk 端点、3 步去噪、单次动作生成约 115.9 ms，具备工程吸引力；但 BoN 需要 32 个候选，延迟约 3116.0 ms，且真实机器人只测了两个任务、未做 policy improvement。 |
| 主要风险 | 端点世界模型在接触丰富任务上的误差会被 PPO/BoN 放大；多任务共享策略存在任务间干扰；真实机器人证据范围窄。 |

💡 **X-Ray 开场**

这篇论文解决的是：机器人演示里不只有“该做什么动作”，还有“动作如何改变世界”和“哪些变化导致成功”。DexWAC 的发现是：把未来端点预测与 TD 价值学习绑在同一个共享前缀上，world model 就不只是辅助损失，而能直接参与想象式策略改进。对 VLA 研究者，关键不是“又做了一个视频世界模型”，而是“端点预测 + critic 共训练 + 真实 rollout 适配”这个接口。

📍 **研究全景时间线**

```text
[2024] π0：VLA flow 控制路线
   → [2025] π0.5 / Motus：更强 VLA 与 world-action model 路线
   → [2026] FACT：failure-aware world-action model，值来自进度标签而非 TD critic
   → [2026] DexWAC：world–actor–critic 单阶段联合预训练 ← 本文
```

本文位置：把 WAM 的未来预测从“辅助监督”推进到“可与 critic 共训练、可适配、可用于 PPO/BoN 的模拟器”。局限：长程记忆任务被排除；大规模机器人预训练未验证；真实机器人只覆盖两个接触丰富任务。

## 1. 核心架构/方法总览

### 1.1 系统对比概览

先按论文可确认的信息逐模型归属，避免把一组模型的属性合并给每个模型。

| 模型/方法 | 论文中的角色与架构 | 数据/开放范围（按论文） | 记忆机制（按论文） |
|---|---|---|---|
| DexWAC | Mixture-of-Transformers（MoT）：FLUX.2-klein-base-4B 世界 backbone + width-1024 actor–critic expert；共享 attention K/V，但权重分离；world model 只预测 chunk 端点 | DexJoCo 800 条专家演示；RoboTwin 2.0 25,000 条 randomized + 2,500 条 clean；真实任务每任务 300 条演示。正文未声明代码/数据开放，候选记录 repo_url 为空，不推断开源状态 | 论文未描述显式历史帧堆叠或记忆模块；上下文 `C_t=(ℓ,z_t,s_t)` 只列指令、当前视觉 latent、当前本体状态；DexJoCo 明确排除 3 个长程记忆任务 |
| π0.5 | 大规模预训练 VLA 基线；论文未展开其 world model/critic 结构 | 论文未展开其数据开放范围 | 论文未描述其记忆机制 |
| Motus | WAM 基线；论文将其归入“统一理解、未来预测、动作生成”的 world-action 路线，动力学预测主要作为辅助监督 | 论文未展开其数据开放范围 | 论文未描述其记忆机制 |
| FACT | WAM 基线；failure-aware causal training；值监督使用轨迹衍生的归一化进度标签，而非 TD critic；有 Stage1/Stage2 | 论文未展开其数据开放范围 | 论文未描述其记忆机制 |
| Ctrl-World | 可控生成式 world model，用作视觉预测基线；50 步去噪 | 论文未展开其数据开放范围 | 论文未描述其记忆机制 |
| VLA-MBPO | 面向 VLA 的 world-model-based RL 基线；50 步去噪；原设置在本论文实验中导致 policy collapse | 论文未展开其数据开放范围 | 论文未描述其记忆机制 |

再按模块看 DexWAC 的输入输出与时序。

| 模块 | 输入 | 输出 | 时序/频率 | 训练/推理差异 |
|---|---|---|---|---|
| World backbone `θ` | `C_t=(ℓ,z_t,s_t)` + clean action chunk | 端点 `z_{t+H}, s_{t+H}`；奖励 `R̂`；继续 `ĉ` | 每个 imagined chunk transition 一次端点预测；论文未给控制频率 | 预训练与真实 rollout 适配时训练；PPO/BoN 中 FLUX backbones 冻结 |
| Actor expert `φ` | 共享 K/V 前缀 | 动作 chunk `a_t...a_{t+H-1}` | 每次生成一个 chunk；DexWAC 共享实现 `H=16` | 预训练做 BC；PPO 阶段更新；BoN 阶段冻结 |
| Critic `Vφ` | `C_t`，attention mask 保证 action-independent | 状态价值 `V(C_t)` | 每个 chunk 级 transition 一个值 | 预训练即用 TD 学习；适配时继续训练 |
| `Qφ` 分支 | `C_t` + clean action chunk + `Q` token | 动作价值 `Q(C_t,a_t)` | 用于 BoN 排序 | 适配阶段启用；BoN 时用想象 transition 继续训练 |

### 1.2 关键机制

- **共享前缀，不共享全部权重**：world backbone 与 actor–critic expert 通过 attention K/V 共享特征，但保持独立权重，目的是减少对 backbone 预训练视觉先验的干扰。论文 §4.1 明确说“share features through attention but keep separate weights”。
- **只预测 chunk 端点，不生成中间视频帧**：给定 `C_t` 和动作 chunk，world model 只预测 `t+H` 的视觉 latent 与本体状态。这样每个 imagined step 只需一次预测，避免逐帧视频生成的高延迟。
- **奖励与继续也由 world model 给出**：奖励头读取端点上下文和区间长度，继续头判断价值 bootstrap 是否继续。于是 imagined transition 的“状态 + 奖励 + 终止”都来自 world model。
- **TD critic 从预训练第一天就参与表示学习**：不是先训策略再外挂 critic，而是用 chunk 级 TD 目标让价值监督塑造共享前缀。论文强调这能减少后续 policy optimization 时“从零学 critic”的成本。
- **真实 rollout 适配**：用专家演示 + 预训练 actor 的成功/失败 rollout 适配 world model 和 critic，actor 保持不变。适配目标仍包含预测、奖励、继续、价值和可选的 `Q`。
- **两种 policy improvement**：PPO 在想象 rollout 上更新 actor，并用专家动作 BC 正则；BoN 冻结 actor 与 world model，只训练 critic 对 32 个候选动作 chunk 排序。

⚡ **Eureka Moment**：把“动作会改变什么”和“什么改变导致成功”压进同一个共享前缀，并让 world model 直接输出 chunk 级端点、奖励和继续信号，从而把未来预测变成可用的 imagined simulator，而不是只做辅助 loss。

### 1.3 信息流/架构图

```text
预训练阶段（DexWAC 共享实现）：

已观测上下文（当前帧）：
  ℓ（语言指令）
  o_t --VAE--> z_t（当前视觉 latent）
  s_t（当前本体状态）
        │
        ▼
  C_t = (ℓ, z_t, s_t)
        │  共享 attention K/V
        ├──────────────► actor expert φ ──► 动作 chunk a_t ... a_{t+H-1}
        │
        ├──────────────► world backbone θ
        │                  输入：C_t + clean action chunk
        │                  输出：仅端点 z_{t+H}, s_{t+H}
        │                        + R̂(C_e,d), ĉ(C_e)
        │
        └──────────────► critic Vφ(C_t)
                           训练目标：chunk 级 TD

未来预测帧的边界：
  只预测 t+H 的端点；不生成 t+1 ... t+H-1 的中间视频帧。
  H=16 是动作 chunk 长度/端点预测跨度，不是历史上下文窗口。
```

```text
三阶段流程：

Stage I  联合预训练
  专家演示 ──► actor BC + world 端点预测 + reward/continuation + TD critic
       │
       ▼
Stage II 真实 rollout 适配
  专家演示 + 成功/失败 rollout ──► 适配 world model + critic（actor 不变）
       │
       ▼
Stage III 策略改进
  路线 A：PPO on imagined rollouts + BC 正则
  路线 B：critic-guided best-of-N（BoN）排序
```

## 2. 数学核心

📌 **Napkin Formula**：

```text
共享前缀 C_t + 动作 chunk → 端点 (z_{t+H}, s_{t+H}, R, c)；同一前缀上的 V(C_t) 用 TD 目标学习
```

先给目标。论文把语言条件操作建模为 MDP，策略在状态 `x_t` 与指令 `ℓ` 下采样动作 chunk：

```text
a_t = (a_t, ..., a_{t+H-1}) ~ π(· | x_t, ℓ)
J(π) = E[ Σ_{t=0}^{T-1} γ^t r_t ]
```

核心训练目标不是单一行为克隆，而是六项联合损失：

```text
L_sup = λ_A L_A + λ_I L_I + λ_S L_S + λ_R L_R + λ_U L_U + λ_V L_V
```

其中 `L_A` 是动作 flow matching，`L_I` 是视觉 latent 预测，`L_S` 是本体状态预测，`L_R` 是奖励回归，`L_U` 是继续分类，`L_V` 是 TD 价值回归。

**动作、视觉、本体三者的生成式监督**都用 conditional flow matching。给定干净目标 `x0` 与高斯噪声 `ε`：

```text
x_τ = (1-τ)x0 + τε
L_FM = E[ || vθ(x_τ, τ) - (ε - x0) ||² ]
```

应用到动作、视觉 latent、本体状态，分别得到 `L_A, L_I, L_S`。

**奖励平滑**：机器人任务常只有稀疏成功信号。论文借 DreamSmooth 思路，把终端成功反馈用单边 Gaussian kernel 分配到前面 `D_T=min(W_R,T)` 步：

```text
g_t = exp(-(T-t)² / (2σ²)) / Z_T,   0 ≤ T-t ≤ D_T
g_t = 0,                            其他
D_T = min(W_R, T)
Z_T = Σ_{k=0}^{D_T} exp(-k² / (2σ²))
```

附录给出 `W_R=48`、`σ=16`，归一化后每个成功回合总奖励为 1；失败回合全程零奖励。

**chunk 区间奖励**：对 anchor `t` 与端点 `e=min(t+H,T)`，令 `d=e-t`：

```text
R_{t:e} = Σ_{i=1}^{d} γ_step^(i-1) · r̃_{t+i}
```

**TD 价值目标**：锚点上下文 `C_t` 与端点上下文 `C_e` 成对前向；端点价值 detach：

```text
y_t = R_{t:e} + γ_step^d · c_e · sg[Vφ(C_e)]
L_V = E[ hκ(Vφ(C_t) - y_t) ]
```

其中 `sg` 是 stop-gradient，`hκ` 是 Huber loss，`c_e` 是继续标签。论文明确说不需要单独 target network 或 EMA copy。

**奖励与继续的类别平衡**：

```text
L_R = α/|P| Σ_{i∈P}(R̂_i-R_i)² + (1-α)/|N| Σ_{i∈N}(R̂_i-R_i)²
L_U = -1/(2|U1|) Σ_{i∈U1} log ĉ_i - 1/(2|U0|) Σ_{i∈U0} log(1-ĉ_i)
```

论文设置 `α=0.5`；若某一类为空，退化为普通 MSE/BCE。

**适配阶段**去掉动作模仿，保留预测、奖励、继续、价值和可选 `Q`：

```text
L_adapt = λ_I L_I + λ_S L_S + λ_R L_R + λ_U L_U + λ_V L_V + λ_Q L_Q
L_Q = E[ (Qφ(C_t, a_t) - y_t)² ]
```

PPO 使用 flow policy 的某一步去噪转移似然比：

```text
ρ_j(ψ) = p_ψ(x_τ' | x_τ, τ, Ĉ_j) / p_old(x_τ' | x_τ, τ, Ĉ_j)
L_π = -E[ min(ρ Â, clip_ε(ρ) Â) ] + β L_reg
```

BoN 在部署时从冻结 actor 采样 `N` 个候选，用 `Q` 排序：

```text
a^(n) ~ π_pre(· | C), n=1...N
n* = argmax_n Qω(C, a^(n))
a* = a^(n*)
```

> 符号说明：`H` 是动作 chunk 长度；`d=e-t` 是实际执行的 chunk 区间控制步数；`γ_step=0.99`；完整 chunk 的折扣约为 `0.99^16≈0.8515`；`c_e` 表示是否继续 bootstrap；`sg` 表示 stop-gradient。论文的 `H=16` 是共享实现中的动作 chunk 设置，不应写成所有实验共有的记忆窗口。

直觉：视觉预测迫使共享前缀编码“动作如何改变场景与机器人状态”，价值监督迫使它编码“哪些状态接近任务成功”。两者共同约束 actor 使用的特征，而不是各训一个独立模型再拼接。

## 3. 带数字走一遍：玩具例子

> 以下为假设的极短回合，仅用于验算公式，不代表论文实验设置。论文实际使用 `H=16`、`γ_step=0.99`、`W_R=48`、`σ=16`。

假设一个成功回合 `T=2`，动作 chunk 仍取 `H=16`，因此端点 `e=min(t+16,2)`。成功终端在 `t=2`。计算奖励平滑：

```text
D_T = min(48, 2) = 2
w0 = exp(0) = 1
w1 = exp(-1/512) ≈ 0.99805
w2 = exp(-4/512) ≈ 0.99222
Z_T = 1 + 0.99805 + 0.99222 = 2.99027

g2 = 1 / 2.99027 ≈ 0.33442
g1 = 0.99805 / 2.99027 ≈ 0.33377
g0 = 0.99222 / 2.99027 ≈ 0.33181
```

对 anchor `t=0`，端点 `e=2`，`d=2`：

```text
R_{0:2} = g1 + 0.99 · g2
        = 0.33377 + 0.99 × 0.33442
        ≈ 0.66485
```

若 `t=2` 是成功终端，继续标签 `c_2=0`，则：

```text
y_0 = R_{0:2} + 0.99² · 0 · V(C_2)
    = 0.66485
```

若假设该端点不是终止、`c_2=1`、且 `V(C_2)=0.5`，则：

```text
y_0 = 0.66485 + 0.9801 × 0.5
    ≈ 1.15490
```

对 anchor `t=1`，端点仍为 `2`，`d=1`：

```text
R_{1:2} = g2 ≈ 0.33442
y_1 = 0.33442 + 0.99 · c_2 · V(C_2)
```

若 `c_2=0`，则 `y_1≈0.33442`；成功终端 anchor `t=2` 的价值目标为零。这个玩具例子验证了三件事：奖励平滑按时间距离衰减；区间奖励按 `γ_step` 折扣；TD 目标只在未终止时 bootstrap。

## 4. 工程视角

**延迟与步数**。论文 Table 4 报告动作条件 world model 的端到端预测延迟：DexWAC 3 步去噪、`0.214 s`；FACT 20 步、`1.816 s`；Ctrl-World 50 步、`4.372 s`；VLA-MBPO 50 步、`63.988 s`。论文称 DexWAC 比 FACT、Ctrl-World、VLA-MBPO 分别快约 `8.5×`、`20×`、`300×`。这里的关键不是“视频生成更快”，而是“只预测 chunk 端点”，把 imagined transition 压缩成一次预测。

**动作生成延迟**。Table 10 在 RTX 5090、batch size 1、每任务 10 次调用共 80 次调用下测量：DexWAC Stage1 平均 `115.9 ms`，PPO 后 `118.0 ms`；π0.5 `352.4 ms`；Motus `626.7 ms`；FACT Stage1 `2665.4 ms`。BoN 因为生成并排序 32 个候选，平均 `3116.0 ms`，约为 Stage1 直接采样的 `26.9×`。工程含义：BoN 的精度收益必须用延迟预算换取；在需要高频闭环的场景，BoN 更适合作为低频重规划或离线筛选，而不是每个控制周期都跑。

**H=16 的边界**。DexWAC 共享实现中动作 chunk 含 `H=16` 个绝对控制目标；world model 预测 `t+H` 端点。这个 `H` 是动作 chunk 长度/未来端点跨度，不是历史上下文窗口。均匀间隔采样 `K` 帧时，首末跨度是 `(K-1)×Δt`；若从当前 `t` 预测 `t+1` 至 `t+H`，终点距当前参考是 `H×Δt`。连续未来 chunk 的预测区间不能直接当作历史上下文或记忆窗口。论文也没有把 `H=16` 写成所有实验共有的记忆窗口；Table 10 中 π0.5 chunk 为 30、Motus 为 16、FACT 为 48，说明不同方法的 chunk 设置并不相同。

**部署约束**。论文未报告控制频率，因此不能从 `115.9 ms/chunk` 直接反推 Hz。条件性分析：若执行器以固定控制周期消费 16 步 chunk，推理延迟必须小于 chunk 执行时间；BoN 的 `3116.0 ms` 会显著压缩可用控制预算。论文也未报告量化误差、显存占用和不同硬件上的抖动，这些是落地前需要补测的工程项（TODO/待补）。

## 5. 数据与评测

### 5.1 预训练设置

| 阶段 | 数据 | 训练量 | 备注 |
|---|---|---|---|
| DexJoCo 联合预训练 | 8 个任务、800 条专家演示 | 10,000 optimizer updates；global batch 256 | 单个语言条件 DexWAC 模型跨 8 任务联合训练 |
| DexJoCo 真实 rollout 适配 | 800 条演示 + 400 条 policy rollouts（每任务 50 条，含成功/失败） | 5,000 updates；batch 256 | actor 不变；适配 world model 与 critic |
| PPO / BoN 策略改进 | 想象 rollout；任务均衡 roots | 500 updates | PPO 用专家动作 BC 正则；BoN 用 32 个候选排序 |
| RoboTwin 2.0 预训练 | 50 个任务；25,000 条 randomized + 2,500 条 clean 轨迹 | 论文未给统一 update 数 | 评测 clean/random 两种设置 |
| 真实机器人 | 每个任务 300 条演示 | 论文未给 update 数 | 未做后续 policy improvement |

注意：DexJoCo 使用 11 个任务中的 8 个，明确排除 3 个需要长程记忆的任务。因此本文不能支持“长程记忆已解决”的结论；`H=16` 也不能被解释为记忆窗口。

### 5.2 仿真评测

DexJoCo 主表 Table 1 使用 seeds 0、1、2，每任务每 seed 50 episodes，报告三 seed 均值与样本标准差。DexWAC Stage1 总体 `65.50±2.2%`；PPO `67.75±1.8%`；BoN `66.58±3.7%`；π0.5 `57.92±5.8%`；Motus `66.83±3.2%`；FACT 两个变体为 `49.17%` 与 `51.08%`。PPO 的提升主要来自 bimanual：`47.78%→52.89%`；single-arm 仅 `76.13%→76.67%`。这说明多任务共享策略的收益并不均匀。

RoboTwin 2.0 使用 50 个任务，每任务 100 episodes，clean/random 两种设置。DexWAC Stage1 平均 `88.1%`（clean `88.1%`、random `88.2%`）；Motus `87.8%`；FACT `87.5%`；Gigaworld-Policy `86.0%`；π0.5 `79.8%`；π0 `62.2%`。论文据此主张 DexWAC 在 randomized 和 average 上最高，但差距很小，不能夸大为全面碾压。

World model 评测在 40 条 held-out DexJoCo 轨迹、767 个窗口上，预测 `t+16`；所有模型接收相同当前观测与记录动作。Table 3 中 DexWAC 在 head/wrist 的 LPIPS、PSNR、SSIM 六项指标均最好；wrist LPIPS 从 FACT 的 `0.1373` 降到 `0.0853`，head LPIPS 为 `0.0287`。Table 4 的延迟见 §4。

Ablation 需要小心口径：Table 6 的 DexJoCo DexWAC 为 `68.00`，与 Table 1 的三 seed 均值 `65.50` 不同；Table 11 显示 seed 0 的 Stage1 为 `68.00`。因此 Table 6 应看表内相对差，不能把 `68.00` 当作主表均值。表内：去掉 world prediction，RoboTwin 从 `88.1` 降到 `82.8`；去掉 critic 降到 `87.4`；DexJoCo 去掉 critic 从 `68.00` 降到 `60.75`，去掉 reward & continuation 降到 `66.25`，去掉视觉预测降到 `65.50`。作者据此主张 critic 监督对接触丰富任务尤其重要。

### 5.3 真实机器人设置

真实机器人有两个任务：`Transfer egg`（单臂在两个蛋托间转移鸡蛋）与 `Extract cup`（双臂从杯堆中取出一只杯子并放回桌面）。每任务 300 条演示，每任务 30 trials，覆盖 5 个初始物体位置，用于空间泛化评测。由于资源限制，真实机器人未做后续 policy improvement。结果：DexWAC `50%` 与 `100%`，π0.5 `20%` 与 `70%`（Table 5）。这是论文实测，但样本只有两个任务、每任务 30 trials，不能外推到更多接触模式或更多本体。

## 6. 能力与失败模式

| 能力/失败模式 | 论文证据 | 条件与边界 |
|---|---|---|
| 联合预训练策略有竞争力 | DexJoCo Stage1 `65.50%`，高于 π0.5 `57.92%`，接近 Motus `66.83%` | 8 任务子集；三 seed、每任务每 seed 50 episodes |
| RoboTwin 平均成功率高 | Stage1 `88.1%` 平均，random `88.2%` | 50 任务、每任务 100 episodes；与 Motus/FACT 差距小 |
| world model 更准更快 | 六项图像指标最好；wrist LPIPS `0.0853`；`0.214 s`/端点 | 预测 `t+16`；40 条 held-out 轨迹、767 窗口；RTX 5090 batch 1 |
| PPO 可提升多任务策略 | 均值 `65.50→67.75`；bimanual `47.78→52.89` | 仅 500 updates；50 条/任务执行轨迹适配；single-arm 提升很小 |
| BoN 可排序候选 | 均值 `66.58%`；延迟 `3116.0 ms` | 32 候选；冻结 actor 与 world model；延迟约为 Stage1 的 `26.9×` |
| 接触丰富任务对 critic 敏感 | 去掉 critic，DexJoCo `68.00→60.75` | Table 6 表内口径；不能与 Table 1 均值混用 |
| 多任务干扰 | 作者在 §5.4/Appendix E 指出部分任务提升、部分回归 | 共享策略优化不保证保留每个预训练技能 |
| 想象式过优化 | 作者指出 world model 误差会被 PPO/BoN 利用；BoN 更明显 | 仅 50 条/任务执行数据；接触结果对位姿/时序敏感 |
| 直接离线 RL 不稳定 | IQL 在 Table 11 为 `54.75%`，低于同表 Stage1 `68.00%`；原 VLA-MBPO 设置导致 collapse | seed 0 口径；作者归因于 Q 值不稳定与 world model 误差累积 |
| 大规模预训练未验证 | 作者在 §6 明确说受算力限制，未做大规模机器人预训练 | 不能声称可扩展到更广数据集/任务/本体 |
| 长程记忆未覆盖 | DexJoCo 排除 3 个长程记忆任务 | 不能把 `H=16` 当记忆窗口，也不能声称解决长程记忆 |

### 6.1 隐含假设

1. **稀疏成功信号可用且终端标签可靠**。奖励平滑依赖成功终端；若成功检测本身有噪声，平滑会把错误信号分配到前序步骤。
2. **端点 latent + 本体状态足以支撑奖励与价值**。论文不生成中间帧，默认 `t+H` 端点能代表 chunk 内的接触结果；对接触时序极敏感的任务，这个假设可能不成立。
3. **共享前缀的收益大于任务间干扰**。Ablation 支持 world/critic 监督有用，但 §5.4 已显示任务间有提升也有回归。
4. **50 条/任务真实 rollout 足以覆盖部署状态**。Appendix E 明确说覆盖不足会导致模型利用；这是论文自认的边界。
5. **冻结 world model 后，PPO/BoN 仍能可靠改进**。作者报告过优化后期收益消失甚至过优化，说明“冻结即安全”不成立。
6. **`H=16` 的端点预测可跨任务复用**。论文共享实现使用 `H=16`，但不同基线 chunk 不同；不能把它当作所有实验共有的时序窗口。

## 7. 与相关工作对比

| 方法 | 关注点 | 架构/训练方式（按论文） | 适用场景 | 与 DexWAC 的关键差异 |
|---|---|---|---|---|
| π0.5 | 大规模 VLA 策略 | 预训练 VLA 基线；论文未展开 world/critic | 通用 VLA 评测基线 | DexWAC 显式共训 world model + TD critic，并用其做 policy improvement |
| Motus | 统一 latent action world model | WAM；动力学预测主要作为辅助监督 | world-action 路线基线 | DexWAC 增加 TD critic 共训练、真实 rollout 适配与 PPO/BoN |
| FACT | failure-aware world-action model | WAM；值用归一化进度标签，不是 TD critic；Stage1/Stage2 | 世界动作模型基线 | DexWAC 用 bootstrapped TD critic，并让 critic 参与共享表示学习 |
| Ctrl-World | 可控生成式 world model | 生成式视频世界模型；50 步去噪 | 视觉预测基线 | DexWAC 不生成中间视频帧，只预测 chunk 端点，延迟更低 |
| VLA-MBPO | VLA 的 world-model-based RL | 世界模型 + RL；50 步去噪 | world-model RL 基线 | 原设置在本论文实验中 collapse；DexWAC 用适配 + BC 正则稳定 PPO |
| DexWAC | 统一 world–actor–critic | MoT 共享 K/V；端点预测；TD critic；PPO/BoN | 接触丰富灵巧操作、多任务策略改进 | 核心是把未来预测与价值监督绑在同一条表示通路上 |

**面试 Tip**：如果被问“DexWAC 和视频世界模型有什么不同”，一句话答：它不生成中间视频帧，而是用动作 chunk 预测端点 latent/本体，并把奖励、继续和 TD critic 放在同一个共享前缀上；`H=16` 是动作 chunk/端点跨度，不是记忆窗口。

## 8. 精读建议

- **值得精读原文的人**：
  - 做 world model + RL 策略改进的研究者：重点看 §4.1–4.3、Appendix A/C，尤其是 TD 目标、适配目标和 PPO/BoN 的接口。
  - 做接触丰富灵巧操作、多任务策略的工程师：重点看 §5.4–5.6 与 Appendix E，理解任务间干扰和过优化风险。
  - 评估部署延迟的工程团队：重点看 §5.3、Table 4、Table 10，区分 world model 延迟、动作生成延迟和 BoN 延迟。

- **建议章节路径**：先读 §4.1（共享前缀 + 端点预测 + TD）→ 再看 §4.2–4.3（适配与 PPO/BoN）→ 然后 §5.2–5.6（数字与 ablation）→ 最后 Appendix A/C（超参、flow PPO、BoN 细节）。§2 Related Work 可在熟悉 WAM/VLA 后跳读。

- **不值得精读的理由**：如果你只关心纯 VLA 架构扩展、长程记忆机制或大规模多机器人预训练，这篇不覆盖；如果你已经非常熟悉 world model + TD critic 的套路，读摘要和 Table 1/3/6 即可判断是否与你的场景相关。

---

[← Back to Theory](./README.md)

关键引用：[arXiv:2611.32654](https://arxiv.org/abs/2611.32654)