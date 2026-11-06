# JADE：让 VLM 成为机器人动作的联合去噪器 (Vision-Language Models Are Joint Denoisers in Robotic Action Generation)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-11-06
>
> **论文**: Vision-Language Models Are Joint Denoisers in Robotic Action Generation
> **链接**: https://arxiv.org/abs/2611.01215
> **核心定位**: 把 VLM 从“静态条件编码器”改成“每步读取 noisy action latents 的联合去噪器”；在 π0.5 基线上提升 OOD 与精细操作，代价是每步推理更贵。
>
> **证据版本说明**: 本笔记依据 arXiv:2611.01215v1 的 HTML 全文（对应 PDF v1，页面标注 31 Oct 2026）。候选元数据标记为 v2 replace，且摘要片段被截断；本笔记未用候选摘要补写或替代 v1 正文主张。若 v2 正文与 v1 有差异，待 v2 正文核验后再更新；当前所有数字与机制均取自 v1 正文。

## ⚡ 快速判断（30 秒读完）

| 维度 | 判断 |
|------|------|
| 核心结论 | JADE 让 VLM 在每个 denoising step 读取当前 noisy action 的 OAT latents，并通过 gated cross-attention 指导 action expert；同时用 clean action latents 监督 VLM。 |
| 适合精读 | 如果你在做 flow/diffusion VLA、action tokenization、少步采样或 OOD 操作，重点看 §3.2–§3.4、§4.3–§4.5。 |
| 可以跳过 | 如果你只关心“又刷了一个 benchmark SOTA”，而不管 VLM 与 action expert 的耦合方式，这篇可以只看摘要。 |
| 落地可行性 | 中：机制清晰、模块可插拔，但 10-step JADE 单次 policy query 203 ms，明显慢于 native π0.5 的 74 ms；Balanced/2-step 可折中。 |
| 主要风险 | 绝对成功率仍低；无显式长时记忆；OAT 需按 benchmark/task 预训练并冻结；真实机器人只覆盖 PiPER 双臂四任务。 |

💡 **X-Ray 开场**

这篇论文问的是：既然 VLM 已经能理解视觉和语言，为什么它在 VLA 里只能当“静态条件”，而不能在去噪过程中看着当前动作 candidate 一起修正？JADE 的答案是：把当前 noisy action 压成紧凑 OAT latents，每步喂给 VLM，让 VLM 的输出通过门控 cross-attention 注入 action expert，并额外监督 VLM 从噪声 latents 恢复 clean latents。对 VLA 研究者来说，这意味着“VLM 参与控制”可以发生在采样循环内部，而不只是 backbone 特征提取。

📍 **研究全景时间线**

```
2022 RT-1（离散动作 token 路线） 
  → 2023 RT-2（VLM 迁移到机器人动作） 
  → 2024 π0（VLM + flow action expert 的级联范式） 
  → 2025 π0.5（开放世界泛化基线；本文的 base） 
  → 2026 JADE（VLM 进入去噪循环；本文）
```

> 仅列本文参考文献可确认的节点，不罗列完整参考文献表。本文的位置是：在 π0.5 的“VLM 静态条件 + flow action expert”范式中，把 VLM 推入每一步去噪。

## 1. 核心架构/方法总览

### 1.1 系统组件对比（JADE 内部）

| 模块 | 输入 | 输出 | 频率/时序 | 训练/推理差异 |
|------|------|------|-----------|----------------|
| 观测前缀 | 3 路 RGB、语言指令、robot state | VLM prefix tokens 的 K/V，记为 C(o_t) | 每个 policy query 计算一次并缓存 | 训练与推理都用；prefix 不 attend action registers |
| OAT encoder | 当前 noisy action 的物理维度 A_τ^phys | Z_τ ∈ R^{8×4} 连续 latents | 每个 denoising step 重算 | 冻结；stop-gradient；按 benchmark 或 task 预训练 |
| 输入适配器 P_η | Z_τ、τ | X_τ，宽度 2048 | 每个 step | 可训练；加 noise-level、register-identity、action-modality embedding |
| VLM register 处理 | X_τ + C(o_t) | 每层 R_τ^(l)，最终 R_τ^out | 每个 step | VLM 可训练；prefix K/V 复用，register 特征逐步重算 |
| 对齐头 G_ψ | R_τ^out | 预测的 clean OAT latents Z_0 | 仅训练时使用 | 线性 2048→4；SmoothL1 |
| 融合模块 | action hidden H_τ^(l)、VLM 特征 R_τ^(l) | cross-attn 更新 U_τ^(l)，再门控残差 | 每个 expert layer、每个 step | 插在 native attention 残差之后、FFN 之前 |
| 门控 α | Avg(H)、Avg(U)、τ 的 time embedding | 每 sample、每 layer 一个标量 | 每个 step | 零初始化，初始时融合残差为 0 |
| Action expert | 融合后的 action hidden | velocity v_Θ(A_τ, τ, o_t) | 每个 step | 保留 native flow-matching 目标 |
| 采样器 | velocity | 更新后的 A_τ | 默认 10 个 Euler steps；另有 2-step、Balanced | 自适应规则只改步长，不改模型权重与执行 horizon |
| 执行 | 50-step action chunk | 执行前 25 个动作后 replan | 每个 chunk | 不是记忆窗口；是未来动作缓冲 |

### 1.2 关键机制

- **Noisy action 进入 VLM**：JADE 不是只把图像和语言给 VLM，而是把当前去噪步的 action latents 也塞进 VLM。这样 VLM 的 guidance 会随着动作 candidate 变化。
- **VLM 与 action expert 的分工**：VLM 不直接输出 action，而是输出 observation-conditioned register features；action expert 仍在原生 action space 里预测 velocity。
- **直接动作监督**：除了 flow-matching loss，JADE 还让 VLM 从 noisy latents 预测 clean OAT latents Z_0。作者认为这给 VLM 一条直接的动作学习信号，使 perception 与 control 更耦合。
- **门控注入**：cross-attention 的输出不是无条件加进去，而是乘一个 per-sample、per-layer 的 tanh gate；gate 零初始化，保证训练初期不破坏 native expert。
- **少步采样**：作者主张每步更有信息，因此 JADE 可用更少 NFE；Balanced 自适应步长进一步在性能与延迟间折中。

⚡ **Eureka Moment**

THE 关键洞见是：**只要 action 被表示成紧凑、有语义结构的 latents，VLM 就能在每一步“看见”当前带噪动作，并像评估器一样动态指导去噪，而不是只当一次性条件。**

### 1.3 信息流/架构图

```text
o_t = {3×RGB, language, robot state}
        │
        ├──→ VLM prefix ──→ C(o_t) = prefix K/V（每个 policy query 缓存一次）
        │
A_τ ──→ frozen OAT ──→ Z_τ ∈ R^{8×4}
        │
        └──→ P_η(Z_τ, τ) ──→ X_τ
                              │
                  VLM registers attend to C(o_t) 与彼此
                              │
                  R_τ^(l), R_τ^out（每个 denoising step 重算）
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
训练时：G_ψ(R_τ^out) → Z_0，SmoothL1               推理时：不计算对齐头
        │                                           │
        └──────────────→ action expert ←────────────┘
                              │
                  gated cross-attn 注入每层
                              │
                        v_Θ(A_τ, τ, o_t)
                              │
                     Euler step：A_τ ← A_τ - h·v
                              │
                         循环到 τ=0
```

**时间口径（编辑说明，不是论文新增实验）**

- 论文明确写 action chunk 为 A = [a_t, …, a_{t+H-1}]，H=50；这是从当前时刻起算的 50 步动作块，最后一项距 t 为 49Δt（按均匀间隔定义）。
- 若历史帧均匀采样 K 个，首末跨度是 `(K-1)×Δt`；若从当前参考 t 预测 t+1 至 t+H，终点距当前参考是 `H×Δt`。
- **连续未来 chunk 的预测区间不能直接当作历史上下文或记忆窗口。** JADE 的 H=50、执行前 25 个动作、10/2/Balanced NFE 都是任务与采样设置；不能把单一设置的 chunk 长度或步长写成所有实验共有的记忆窗口。
- JADE 的 prefix K/V cache 只在一次 policy query 内复用，不是跨 query、跨 episode 的长时记忆。

## 2. 数学核心

📌 **Napkin Formula**

```text
每步：Z_τ = OAT(A_τ) → VLM(o_t, Z_τ) → gated cross-attn → v_Θ；并用 Z_0 监督 VLM
```

**目标**：在 flow-matching VLA 中，让 VLM 的 condition 依赖当前 noisy action，而不是固定于初始观测；同时给 VLM 一个从 noisy latent 恢复 clean latent 的辅助目标。

**Flow matching 与联合损失**

```text
A_τ = (1 - τ) A + τ ε,   ε ~ N(0, I)

L_flow = MSE(v_Θ(A_τ, τ, o_t), ε - A)

Z_τ = OAT(A_τ^phys),   Z_0 = OAT(A^phys)

L_align = SmoothL1(G_ψ(R_τ^out), Z_0)

L = L_flow + 0.1 · L_align
```

**VLM 特征与融合**

```text
{R_τ^(l)}_{l=1..L}, R_τ^out = F_φ(X_τ; C(o_t))

U_τ^(l) = CrossAttn_l(RMSNorm(H_τ^(l)), T_l(R_τ^(l)))

H̃_τ^(l) = H_τ^(l) + α_τ^(l) · U_τ^(l)

α_τ^(l) = s_g · tanh(b_l + g_ω(Avg(H_τ^(l)), Avg(U_τ^(l)), e_g(τ)))
```

**自适应采样**

```text
v_i = v_Θ(A_τ_i, τ_i, o_t)
q_i = -τ_i · v_i^phys

m_i = β · m_{i-1} + (1-β) · q_i
s_i = β · s_{i-1} + (1-β) · q_i^2

c_i = ||[s_i - m_i^2]_+||_F^(1/2)

h_i = min{τ_i, clip(λ_step / (c_i + ε_c), h_min, h_max)}

A_τ_{i+1} = A_τ_i - h_i · v_i
τ_{i+1} = τ_i - h_i
```

**变量说明**

| 符号 | 含义 |
|------|------|
| τ | flow time；τ=1 为高斯噪声端，τ=0 为 clean action 端 |
| A | clean action chunk |
| ε | 高斯噪声 |
| A_τ | 当前 noisy action |
| v_Θ | action expert 预测的 velocity |
| Z_τ / Z_0 | noisy / clean action 的 OAT latents |
| R_τ^(l) / R_τ^out | VLM 第 l 层 / 最终 register features |
| α_τ^(l) | 门控残差系数 |
| q_i | 朝当前一步 clean estimate 的位移 |
| c_i | 最近 correction 的波动统计量 |
| h_i | 当前 Euler 步长 |

**直觉**

- flow matching 学的是从噪声指向 clean action 的 velocity；JADE 让这个 velocity 的 condition 里包含“当前动作 candidate 长什么样”。
- 对齐损失不是替代 flow loss，而是让 VLM 的 register features 对“clean action latent”有预测能力，从而把 perception 与 control 绑得更紧。
- 自适应规则用 correction 的 EMA 波动来决定步长：波动小则迈大步，波动大则迈小步。论文明确说 c_i 不是校准过的局部误差估计器，不能据此证明积分误差容限；它只是采样启发式。

## 3. 带数字走一遍：玩具例子

> **假设示例，非论文实验。** 下面用一维动作演示公式如何闭合，所有数值按论文公式手算验算。

设 clean action `A = 1.0`，噪声 `ε = 0`，当前 flow time `τ = 0.75`：

```text
A_τ = (1 - 0.75) × 1.0 + 0.75 × 0 = 0.25
target = ε - A = -1.0
```

假设 action expert 当前预测 `v = -0.8`：

```text
L_flow = (-0.8 - (-1.0))^2 = 0.04
clean estimate = A_τ - τ·v = 0.25 - 0.75×(-0.8) = 0.85
q = -τ·v = 0.6
```

也就是说，当前 velocity 把 0.25 朝 0.85 推了 0.6。

再用 Balanced 的超参做一次 EMA 演示：`β=0.3`，`λ_step≈0.4217`，`h_min=0.1`，`h_max=0.5`，`h_0=0.1`。为简化，假设连续两步 `q=0.6`（这不是实际轨迹，只用于验算 EMA）：

```text
i=0:
m_0 = 0.3×0 + 0.7×0.6 = 0.42
s_0 = 0.3×0 + 0.7×0.36 = 0.252
V_0 = 0.252 - 0.42^2 = 0.0756
c_0 = sqrt(0.0756) ≈ 0.2749
但第一步用固定 h_0=0.1，不用 c_0

i=1:
m_1 = 0.3×0.42 + 0.7×0.6 = 0.546
s_1 = 0.3×0.252 + 0.7×0.36 = 0.3276
V_1 = 0.3276 - 0.546^2 = 0.029484
c_1 = sqrt(0.029484) ≈ 0.1717

h_1 = min{0.9, clip(0.4217 / 0.1717, 0.1, 0.5)}
    = min{0.9, 0.5}
    = 0.5
τ_2 = 0.9 - 0.5 = 0.4
```

这个例子说明两件事：第一，`q` 是“朝当前 clean estimate 的位移”，不是最终动作；第二，`c_i` 会因零初始化与 EMA 瞬态而变化，论文附录也明确说即使 correction 恒定，早期 `c_i` 仍可能为正。因此不能把 `c_i` 直接解释成曲率或成功概率。

## 4. 工程视角

| 指标 | 论文数值 | 条件/口径 |
|------|----------|-----------|
| 训练 update 时间 | 162.7 → 200.6 ms，+23.3% | 2×H100 80GB、BF16、FSDP、global batch 2、5 次 warm-up 后计时 30 次同步 update；不含编译与数据加载 |
| 每样本矩阵/卷积 FLOPs | 17.349 → 17.448 TF，+0.57% | 含 forward、backward、recomputation；仅该 microbenchmark |
| 10-step native π0.5 延迟 | 约 74 ms/policy query | 生成 50-action chunk；不含通信、仿真、机器人执行 |
| 10-step JADE 延迟 | 203 ms/policy query | 同上 |
| Balanced JADE 延迟 | 132 ms/policy query | 平均约 5 次 NFE |
| 2-step JADE 延迟 | 78 ms/policy query | 2 次 NFE |
| RoboTwin NFE/成功率 | 10-step: NFE 10，Avg 66.9%；2-step: NFE 2，Avg 66.0%；Balanced: NFE 4.995，Avg 65.2% | 固定 checkpoint 与 horizon，只变 NFE |
| RoboDojo NFE/成功率 | 10-step: NFE 10，SR 12.67%；2-step: NFE 2，SR 9.67%；Balanced: NFE 4.985，SR 12.48% | Balanced 保留 10-step SR 的约 98.5% |
| Balanced NFE 理论范围 | 3 ≤ N_eval ≤ 10 | h_0=h_min=0.1，h_max=0.5；终止步可能更小 |

**工程含义**

- **每步更贵，但步数可更少**：JADE 把 VLM 拉进每个 denoising step，单步成本上升；作者用 2-step 与 Balanced 证明“少步”可以部分抵消。2-step 在 RoboTwin 几乎追平 10-step（66.0 vs 66.9），但在 RoboDojo 掉到 9.67%，说明少步并非所有 benchmark 都无损。
- **prefix cache 是关键**：观测前缀 K/V 每个 policy query 只算一次；每步重算的是 action registers。8×4 的紧凑 latents 降低了 VLM 输入长度。
- **门控零初始化利于稳定**：初始融合残差为 0，训练先从 native expert 行为出发，再逐步学习 VLM guidance。
- **控制频率不能从论文直接推出**：论文给的是 policy query 延迟与 50-step chunk、执行前 25 步；没有给实际控制频率。部署时要结合动作执行耗时与 replan 周期计算。
- **内存边界要说清**：JADE 没有显式跨 episode 记忆；prefix K/V 是单次 query 内的缓存。RoboDojo 的 Memory 类别是评测能力维度，不等于模型有长时记忆模块。

## 5. 数据与评测

### 5.1 预训练/表示学习

- JADE 在 π0.5 基础上构建，使用 3 路 RGB、语言指令、robot state，action chunk 长度 H=50。
- 每个 benchmark 单独预训练并冻结一个 OAT encoder；OAT 使用 8 个 4 维 registers，即 8×4 连续 latents。
- OAT 预训练使用 FSQ levels `(8,5,5,5)` 与 4 层 decoder、width 256、batch 256、lr `5e-5`；策略训练与推理只使用 FSQ rounding 前的连续 latents，OAT 冻结并 stop-gradient。
- 动作物理维度 D_p=14，padding 到 D=32；flow loss 覆盖 50×32，OAT 与执行只用物理维度。
- 真实机器人上，每个任务额外使用 task-specific OAT，先在演示上预训练再冻结。
- **开放范围**：论文正文给出代码链接，但未声明训练数据、OAT 权重、策略权重是否开放；本笔记不把代码链接当作数据开放证据，也不转引外部 URL。

### 5.2 仿真评测

| Benchmark | 训练设置 | 评测设置 | JADE | π0.5 基线 |
|-----------|----------|----------|------|-----------|
| RoboTwin 2.0 | 50 个双臂任务，每任务 50 个 clean demonstrations，共 2,500 个演示；单多任务策略 | Clean / Random 两种条件；每任务每条件 100 trials，即每条件 5,000 trials；Random 未在训练中出现 | Clean 79.3%，Random 54.4%，Avg 66.9% | Clean 68.5%，Random 46.0%，Avg 57.3% |
| RoboDojo | 3,500 个 demonstrations，35 个训练任务 | 42 个评测任务，5 类能力；162 个 evaluation cells，6,300 trials；官方 score 0–100 与 full-task SR | Score 18.29，SR 12.67% | Score 11.41，SR 6.91% |

- RoboTwin 上 JADE 的 Random 比 π0.5 高 8.4 个百分点，Avg 高 9.6 个百分点（论文 Table 1）。
- RoboDojo 上 JADE 在五类能力均高于 π0.5；Precision SR 16.33% vs 5.50%，Long-Horizon SR 24.00% vs 14.67%，Memory SR 6.56% vs 4.56%，Open SR 2.42% vs 1.67%（论文 Table 2）。
- **指标口径提醒**：正文 RoboDojo SR 12.67% 是五类等权平均；附录 D.1 说明 archived ten-step JADE 的 episode-weighted success 为 `825/6300=13.10%`，与类别加权值不同。引用时必须说明是哪种平均。

### 5.3 真实机器人设置

- 平台：PiPER 双臂机器人，1 个 overhead camera + 2 个 wrist cameras；输入 RGB、robot state、语言指令；输出 50-action chunk，每个动作含 12 个绝对关节角目标与 2 个 gripper 命令。
- 四个任务：Tool Placement、Bimanual Handover、Object Arrangement、Tabletop Organization。
- 每个任务用相同 100 个 teleoperated demonstrations 分别训练 π0.5 与 JADE；每个策略每任务评测 30 trials。
- 结果：JADE 在四个任务上均优于 π0.5，总体成功率从 49.2% 提升到 70.0%，即 +20.8 个百分点（论文 Figure 5）。
- 归因限制：真实实验只覆盖一个双臂平台与四个桌面任务，不能外推到移动机器人、人形或更多任务域。

## 6. 能力与失败模式

**能做什么**

- 在 RoboTwin 的视觉随机化下提升 Random 成功率；作者认为原因是 VLM 的 condition 会随动作 candidate 更新。
- 在 RoboDojo 的 Precision 与 Long-Horizon 上收益最大；论文实测 Precision SR 从 5.50% 到 16.33%，Long-Horizon 从 14.67% 到 24.00%。
- 在真实双臂任务上提升整体成功率；尤其 Tool Placement、Bimanual Handover、Tabletop Organization 等需要精细放置、双臂协调或多阶段执行的任务。
- 支持少步采样：2-step JADE 在 RoboTwin 上 Avg 66.0%，接近 10-step 的 66.9%；Balanced 在 RoboDojo 上以约 5 NFE 保留约 98.5% 的 10-step SR。

**不能做什么/失败模式**

- **绝对成功率仍低**：RoboDojo full-task SR 12.67%，Memory 类 6.56%，Open 类 2.42%；per-task 表里多个任务为 0。它不是通用操作解决方案。
- **没有显式长时记忆**：JADE 的 prefix K/V cache 只在单次 policy query 内复用；action chunk 是未来动作缓冲，不是历史上下文。不能把 H=50 或 10-step 说成记忆窗口。
- **每步 VLM 成本高**：10-step JADE 203 ms vs native π0.5 74 ms；Balanced 132 ms，2-step 78 ms。实时性要求高的平台需要重新权衡 NFE。
- **依赖 OAT 表示质量**：OAT 按 benchmark/task 预训练并冻结；如果目标域动作分布与 OAT 训练分布不匹配，VLM 读到的是失真的 latents。论文没有给出 OAT 失配的鲁棒性实验。
- **自适应规则不是误差保证**：论文附录明确说 c_i 不是校准的局部误差估计器，不能 certify 积分误差容限；性能–延迟权衡是经验评测。
- **真实实验范围窄**：PiPER 双臂、四任务、每任务 100 demos/30 trials；不能据此声称对所有人形/移动/多机器人有效。

### 6.1 隐含假设

1. **OAT latents 在噪声水平变化时仍保留任务相关语义**：VLM 能从不完全的、带噪的 latents 中读出有用结构；论文用对齐损失强化这一点，但没有逐噪声级消融。
2. **VLM 能处理 8×4 的紧凑 register 序列而不被 prefix 干扰**：设计中 prefix 不 attend registers，registers attend prefix 与彼此；这假设了这种非对称 attention 足以让 VLM 聚焦动作。
3. **动态 guidance 的收益大于每步重算成本**：作者用 2-step/Balanced 支持，但不同硬件、不同控制频率下未必成立。
4. **门控与 cross-attention 的容量足够**：零初始化保证稳定起步，但论文没有分析 gate 是否会在训练后饱和或忽略 VLM。
5. **benchmark 的 Random/Precision/Memory 等类别能代表 OOD 与长时能力**：这是 benchmark 设计假设，不是 JADE 独有的保证。

## 7. 与相关工作对比

> 下表只写本文能确认的属性。基线方法的架构细节、数据开放范围与记忆机制，论文没有展开的，标为“本文未展开”，不把 JADE 的 OAT/gating/prefix cache 归给它们。不同方法的预训练规模与数据不可比，数字仅代表本文官方设置下的报告。

| 方法 | 本文中的角色 | 架构/VLM 参与方式（本文可确认） | 数据/开放范围（本文可确认） | 记忆机制（本文可确认） | 关键指标（RoboTwin Avg / RoboDojo SR） |
|------|--------------|----------------------------------|------------------------------|--------------------------|----------------------------------------|
| JADE | 本文方法 | π0.5 + frozen OAT + 每步 VLM register + gated cross-attn；VLM 参与去噪 | 正文给出代码链接；权重/数据开放未声明 | 无显式长时记忆；prefix K/V 仅单次 query 内缓存 | 66.9% / 12.67% |
| π0.5 | 主要 base 与基线 | flow VLA；论文将其描述为静态条件范式；10-step 对照 | 本文未声明 | 本文未展开 | 57.3% / 6.91% |
| π0 | 基线 | flow VLA；本文只给数值与少量引用 | 本文未声明 | 本文未展开 | 31.4% / 1.53% |
| GR00T-N1.7 | 基线 | flow action expert；预训练 VLM 特征作为连续 action expert 的条件（本文 §1/§2.1） | 本文未声明 | 本文未展开 | 32.2% / 1.31% |
| X-VLA | 基线 | 本文只给 benchmark 数值 | 本文未声明 | 本文未展开 | 44.5% / 6.52% |

**面试 Tip**：被问“JADE 和 π0.5 的本质区别”，一句话答：不是换 backbone，而是把当前 noisy action 的 OAT latents 每步喂给 VLM，用 gated cross-attention 注入 action expert，并加 clean latent 对齐；所以它能把“VLM 理解”变成随动作 candidate 变化的动态 guidance，代价是每步更贵、需要少步采样来抵消。

## 8. 精读建议

- **值得精读原文的人**：
  - 做 flow/diffusion VLA 架构，想把 VLM 更深地接入控制回路的研究者；
  - 做 action tokenization，关心“什么样的 action 表示适合 VLM 读取”的工程师；
  - 做 OOD 操作、精细操作、少步采样与推理延迟权衡的工程团队。
- **建议章节路径**：先读 §3.2–§3.3（noisy action 表示、对齐、融合）→ 再看 §3.4（自适应采样）→ 然后 §4.3–§4.5（消融与 dynamic/static/lagged 分析）→ 需要理论边界时看 Appendix C.2。§2 相关工作可在熟悉 VLA 脉络后跳过。
- **不值得精读的理由**：如果你只关心 benchmark 排名，或不做机器人学习、不关心 action expert 与 VLM 的耦合方式，读摘要与表格即可；如果你已经非常熟悉 π0.5/flow matching，本文的新增点主要集中在“每步 action-conditioned VLM guidance”这一处。

---
[← Back to Theory](./README.md)

关键引用：[论文](https://arxiv.org/abs/2611.01215)（v1 HTML 全文）