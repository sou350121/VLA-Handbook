# 指令黑洞与最优传输虫洞：VLA 执行中改指令的机制拆解 (Escaping the Instruction Black Hole of Vision-Language-Action Policies via Optimal-Transport Wormholes)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-11-26
>
> **论文**: Escaping the instruction black hole of vision-language-action policies via optimal-transport wormholes
> **链接**: https://arxiv.org/abs/2611.34280
> **核心定位**: 把“VLA 一旦进入任务执行就吸收新指令”定义成 instruction black hole，并用 training-free 的 OT relocation 插件把机器人送到新任务的 committed region，让状态而不是语言触发新任务。LIBERO 仿真上，committed-switch success 从不超过 1.2% 提到 93.1%。

**证据口径**：本文只依据 arXiv:2611.34280v1（2026-11-25）正文。来源记录未提供“官方 PDF 与元数据摘要不一致”的验证结果；候选摘要与正文摘要一致，因此不引入摘要之外的主张。下文“论文实测”指正文表格/段落，“作者主张”会单独标注，“条件性分析”是笔者基于论文约束的推断。

---

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 静态模仿在 committed region 对语言条件不敏感；Wormhole 不改策略权重，用 OT 选新任务 approach-phase 状态并限速搬运，让状态替指令发令。 |
| 适合精读 | 做 VLA 指令跟随、执行中切换、人机交互中断恢复的读者；重点看 §1.2、§2、§5。 |
| 可以跳过 | 只关心真实机器人落地、或只关心 autoregressive VLA 的读者；本文只有 LIBERO 仿真与两个 flow-based backbone。 |
| 落地可行性 | 中：插件不需要训练，但需要同工作空间的新任务成功执行记忆、绝对末端位姿与自由空间路径；真实机器人未验证。 |
| 主要风险 | 记忆与 far mouth 的绝对位姿假设；LIBERO-Goal 上 retreat 在 t_s=20 更强；π0 上自训练 AW 失败。 |

💡 **X-Ray 开场**  
这篇论文问的是：为什么机器人执行到一半时，用户改指令它却不听？作者发现，VLA 不是没读到新词，而是执行状态已经“说出”了旧任务，语言在 committed region 变成冗余输入；静态模仿甚至无法区分“会听的策略”和“聋的策略”。对 VLA 研究者而言，这意味着评测不能只看静态成功率，还要看切换时刻的 escape curve，且“把机器人放在哪”可能比“把指令放大多少”更关键。

📍 **研究全景时间线**

```
2023  LIBERO / RT-2：VLA 与长程操作基准成形
  → 2024  π0 / OpenVLA：flow-based 与开源 VLA  backbone 推进
  → 2025  π0.5 / SwitchVLA：开放世界泛化与执行中切换开始被讨论
  → 2026  本文：把“切换失败”从注意力问题改写成状态可辨识性问题
```

---

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

先明确时序口径：论文里的 `s_t` 是**已观测历史/执行上下文**，`A_t = {a_{t+1}, ..., a_{t+K}}` 是**未来动作 chunk**。Wormhole 的 Mouth A 是从旧任务策略采样出的**未来短 horizon exits**；Mouth B 是记忆中的**过去成功执行新任务的状态**。三者不是同一个窗口，不能把未来 chunk 长度当成历史上下文或记忆窗口。

| 组件/模型 | 角色 | 输入 | 输出/时序 | 训练/推理 | 数据开放与记忆 |
|---|---|---|---|---|---|
| π0 | flow-based VLA backbone | 图像、指令、proprio/history | flow-matching action expert 生成 action chunk；论文用 10 步 flow integration | 论文使用 released LIBERO checkpoint，不重新预训练 | checkpoint 公开；自身输入含执行上下文 `s_t`（仅过去历史），策略输出未来 action chunk，无外部状态记忆 |
| π0.5 | 同族 flow-based VLA backbone | 同上 | 同上 | released LIBERO checkpoint | checkpoint 公开；输入/输出口径同 π0，无外部状态记忆 |
| Wormhole | training-free plug-in | 当前 `χ_t`、旧/新指令、新任务成功执行记忆 `M_new` | 采样 N=16 个旧任务 exits，取 M=64 个新任务 entries，OT 选 far mouth，限速 traversal 后交还策略 | 不训练、不访问权重；每次切换计算一次 | 论文称代码与切换协议公开（本文不展开外部链接）；记忆是外部 `M_new`，不是模型参数记忆 |
| AW（attention wormhole） | π0.5 的自训练变体 | 图像、指令、可学习 `[LOC]` token | 指令只通过 `[LOC]` 进入 action expert | LoRA 适配器 + `[LOC]`；6000 steps；切换数据来自 oracle relocation teacher | 只在 π0.5 上验证；`[LOC]` 是 grounding token，不是记忆窗口 |
| retreat-and-restart | scripted 基线 | 当前状态 | 开爪、抬升、回到初始 pose、重启 | 无学习 | 不需要新任务记忆，但需要 home pose |
| action-space bridge | 对照插件 | 当前状态、旧/新指令 | 共享 Mouth A/B，但不移动机器人，执行 2–6 个 10-step chunks | training-free 或 learned bridge head | 需要状态兼容的运动段检索，不是状态记忆 |

**逐模型归属要点**：π0 与 π0.5 都是 flow-based VLA，但论文没有把二者的所有属性合并；Wormhole 对两者都做 training-free 评测；AW 只在 π0.5 上成立，π0 上作者明确报告“self-trained wormhole 未验证成功”。记忆机制也要分开：π0/π0.5 自身只把执行上下文 `s_t`（过去历史）作为输入，并输出未来 action chunk；Wormhole 额外引入 `M_new` 这个外部状态记忆；AW 的 `[LOC]` 是学到的 grounding 通路，不是记忆。

### 1.2 关键机制 (Key Mechanism)

论文把切换失败拆成三个行为签名：

1. **capture**：旧任务照常完成，新指令像没发生。
2. **freezing**：机器人停在 horizon 附近，既不完成旧任务也不执行新任务。
3. **tidal disruption**：切换成功但动作流被撕裂，出现大跳变与 jerk。

机制解释不是“注意力没了”。论文实测显示，在切换前最后一个 query，action queries 对指令的 attention 仍保留约 92%；但 transmission ratio 已经塌缩。也就是说，**指令仍被读到，但没有进入动作**。

Wormhole 的做法分三步：

- **Mouth A（旧任务出口）**：在 `c_old` 下采样 N=16 条短 horizon continuations，表示“旧黑洞正把机器人往哪拉”。
- **Mouth B（新任务入口）**：从 `M_new` 取新任务成功执行的 approach-phase 状态，要求至少在 gripper 首次闭合前 m=10 步；每个 entry 带 time-to-go。
- **Throat（OT 耦合）**：用 entropic OT 在“出口终点”和“入口状态”之间匹配，成本同时考虑路径长度与剩余完成时间；由 coupling 定义 far mouth `y*`。
- **Traversal（限速搬运）**：开爪、抬升、跨越、落到 `y*`，每步位姿动作变化受 `r` 限制；期间不查询策略。
- **Hand-back**：把机器人交还原始策略，在 `l_new` 下继续。

⚡ **Eureka Moment**：不要教策略在旧任务的黑洞里“听话”；把机器人搬进新任务的黑洞，让 committed state 自己替新指令发令。

### 1.3 信息流/架构图 (Flow / Diagram)

```
t_s: l_old → l_new
  │
  ├─ 历史/执行上下文 s_t = {q_{t-r:t}, a_{t-r:t-1}, g_{t-r:t}, e_{t-r:t}}
  │     （只进入条件，不是未来帧）
  │
  ├─ Mouth A: πθ(·|c_old) 采样 N=16 条未来短 horizon exits
  │     终点 q_i，均值 x̂
  │
  ├─ Mouth B: M_new 中取新任务 approach-phase entries y_j
  │     权重 b_j ∝ exp(-E(χ_ts, y_j)/κ)
  │
  ├─ Throat: γ* = Sinkhorn(a, b, C)
  │     C_ij = L(q_i, y_j)/v̄ + λ(T_j - k_j)
  │
  ├─ Far mouth: y* = Σ (ω_i/a_i) γ*_ij y_j
  │
  ├─ Traversal: open gripper → lift → transit → settle
  │     ||a_k - a_{k-1}|| ≤ r；最多 90 steps；到达半径 ε=2 cm
  │
  └─ Hand-back: 原策略在 l_new 下继续
```

---

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**：

```
ρθ(χ) = <Δθ(χ), d*(χ)> / ||d*(χ)||²
目标 ρ→1；黑洞 ρ≤ε；事件视界 t_h = inf{t : χ_t ∈ B_ε}
```

先给目标：用 transmission ratio 量化“策略对新指令的响应有多少落在真正需要的位移方向上”。`ρ=1` 是完全传输，`ρ=0` 是完全吸收。

**committed region 与静态模仿的盲点**：

```
C = {χ : p_D(l|χ) = 1[l = g(χ)]}
```

在 `C` 内，演示数据已经由状态唯一决定任务，语言没有额外信息。Proposition 1 给出：

```
I_D(A; L | χ) = 0,  χ ∈ C
J_D(π̃) = J_D(π),  π̃(·|χ,l) = π(·|χ,g(χ))
ρπ̃(χ) = 0,  χ ∈ C 且 d*(χ) ≠ 0
```

直觉：在 committed region 里，把语言输入替换成“状态推断出的任务”不会改变损失；因此静态模仿无法惩罚“聋”的策略，甚至更偏好这种更简单的解。

**OT 选 far mouth**：

```
γ* = argmin_{γ∈Π(a,b)} Σ_i Σ_j γ_ij C_ij - η H(γ)

C_ij = L(q_i, y_j)/v̄ + λ(T_j - k_j)

b_j ∝ exp(-E(χ_ts, y_j)/κ)
E(χ, y_j) = L(χ, y_j)/v̄ + λ(T_j - k_j)

y* = Σ_i Σ_j (ω_i/a_i) γ*_ij y_j
```

**限速与 tidal disruption 指标**：

```
||a_k - a_{k-1}|| ≤ r

D_disc = ||a_ts - a_ts-1||₂

J_trans = (1/(H-2)) Σ_{k=ts}^{ts+H-3} ||a_{k+2} - 2a_{k+1} + a_k||₂,  H=20
```

变量说明：

| 符号 | 含义 |
|---|---|
| `χ_t = (o_t, q_t, s_t)` | 执行状态：观测、proprio、历史上下文 |
| `c_t = (χ_t, l)` | 完整条件 |
| `d*(χ)` | 新任务相对旧任务所需的期望位移 |
| `Δθ(χ)` | 策略在 `l_new` 与 `l_old` 下的均值动作差 |
| `ρθ` | 传输比；沿正确方向表达了所需重定向的比例 |
| `C` | committed region；状态已唯一决定任务 |
| `q_i` | 第 i 个旧任务 exit 的终点 |
| `y_j` | 第 j 个新任务 approach-phase entry |
| `T_j - k_j` | 从 entry 到任务完成的剩余步数 |
| `L` | traversal 路径长度 |
| `v̄, λ, κ` | 名义速度、time-to-go 权重、温度 |
| `η` | entropic regularization |
| `ω_i, σ²` | 锚定权重与带宽 |
| `r, ε` | 位姿变化率上限、到达/门控半径 |

**时序口径**：论文查询策略的频率是每 5 个 control steps 一次，flow integration 为 10 步；Wormhole 的 exit horizon 是 10 个 control steps，action-space bridge 的 chunk 是 10 steps。论文在 Table 4 报告了统一的 `r=0.3` per step（6-D 位姿变化范数），也没有把这些步长写成所有模型的记忆窗口。均匀采样下：

```
K 个采样帧的首末跨度 = (K-1) × Δt
从当前参考 t 预测 t+1 至 t+H 的终点距当前 = H × Δt
```

因此，连续未来 chunk 的预测区间不能直接当作历史上下文；Wormhole 的 10-step exit horizon 也不是 π0/π0.5 的通用记忆窗口。

---

## 3. 带数字走一遍：玩具例子 (Worked Example)

> 以下为**假设数值**，用于验算公式，不是论文实验。

**玩具 A：传输比**  
设当前状态要求的位移 `d* = (10, 0)`。策略在 `l_new` 与 `l_old` 下的均值动作差为 `Δ = (9, 0)`，则：

```
ρ = <(9,0), (10,0)> / ||(10,0)||²
  = 90 / 100
  = 0.9
```

若 `Δ = (0, 0)`，则 `ρ = 0`；若 `Δ = (2, 0)`，则 `ρ = 0.2`。这验证了 `ρ` 是“沿正确方向的投影比例”，不是动作幅度本身。

**玩具 B：OT 入口加权**  
设 `v̄ = 1.5 cm/step`，`λ = 1`，`κ = 5`。当前末端在 `(0,0)`，两个新任务 entries：

- `y1 = (15,0)`，`T1 = 10`
- `y2 = (30,0)`，`T2 = 20`

先算 energy：

```
E1 = 15/1.5 + 10 = 20
E2 = 30/1.5 + 20 = 40
```

权重：

```
b1 ∝ exp(-20/5) = exp(-4) ≈ 0.0183
b2 ∝ exp(-40/5) = exp(-8) ≈ 0.000335
```

归一化：

```
b1 ≈ 0.0183 / (0.0183 + 0.000335) ≈ 0.982
b2 ≈ 0.000335 / 0.0183 ≈ 0.018
```

若只有一个 exit，OT 退化为 b-weighted entry：

```
y* ≈ 0.982 × (15,0) + 0.018 × (30,0)
   ≈ (15.27, 0)
```

验算：`b1/b2 = exp(-(20-40)/5) = exp(4) ≈ 54.6`，归一化后 `b1 ≈ 54.6/55.6 ≈ 0.982`，与上面一致。若 rate limit `r=0.3`，仅沿 x 轴至少需要 `15.27/0.3 ≈ 51` 步；论文的 lift–transit–settle 路径更长。这个例子说明：time-to-go 会让“更远但更快完成”的 entry 获得权重，但距离仍通过 `L/v̄` 进入成本。

---

## 4. 工程视角 (Engineering View)

| 项 | 论文数值 | 工程含义 |
|---|---|---|
| 采样 N=16 exits | 901 ms（π0.5 server，idle NVIDIA L20）；普通 query 102 ms | 切换时一次性开销约为 9 次普通 query；不是每步都付 |
| far mouth 选择 | 约 10 ms，单 CPU 核，2441 个 states | OT 本身很轻；瓶颈在策略采样与 traversal |
| traversal 中位数 | π0.5 43 steps；π0 44 steps | 期间不查询策略，约省 9 次 query |
| 完成步数 | Wormhole 138/141；retreat 136/138（π0.5/π0） | 前进式重定位不一定更慢，因为交还在 approach phase |
| 切换 discontinuity | Wormhole 0.32/0.30；retreat 1.49/1.44 | 限速把 tidal disruption 压到约五分之一 |
| transition jerk | Wormhole 0.09；retreat 0.30/0.29 | 动作流更平滑 |
| 到达半径 / 预算 | ε=2 cm；最多 90 traversal steps | 远端口不可达时会超时或停短 |
| rate limit | `r=0.3` per step（6-D 变化范数） | 用动作接口实现“ tidal regularizer” |
| 查询/集成 | 每 5 control steps 查询；flow integration 10 steps | 控制频率与策略查询频率不是同一概念 |
| episode 上限 | 220（Spatial）、300（Goal）、280（Object）steps | 评测协议参数，不是模型记忆窗口 |

**工程含义**：

- Wormhole 是**切换时的一次性规划 + 一段闭环 traversal（期间不查询策略）**，适合低频中断，不适合每步重规划。
- 它要求策略暴露 delta end-effector pose 动作接口；真实机器人上还要有碰撞检查与显式 release/place。
- `M_new` 是外部状态记忆，需要同工作空间的成功执行；物体大位移时，绝对末端位姿 `y*` 会脆弱。
- 位置门控只检查“是否已接近新任务 approach phase”，可能跳过本应执行的 traversal；论文在 LIBERO-Goal 的 wine bottle → bowl 上就遇到这个问题。
- 条件性分析：如果真实场景没有“工作空间上方自由空间”，固定 lift–transit–settle 路径会失效；论文的消融显示去掉 lift/clearance 在 LIBERO 上仍成功，但这不能直接外推到拥挤真实场景。

---

## 5. 数据与评测 (Data & Eval)

### 5.1 预训练与基座设置

- **π0 / π0.5**：论文使用 released LIBERO checkpoints，不重新预训练；flow integration 10 steps。两者都是 flow-based VLA，但论文没有把它们的属性合并。
- **AW**：在 π0.5 上训练 LoRA adapters 与 `[LOC]`，53.2M / 3404.5M 参数，6000 steps；损失为 flow-matching + 0.1×grounding KL + 0.2×readout loss。训练数据来自 oracle relocation teacher：原始数据 596 episodes / 98,894 frames；safe-exit 数据 589 episodes / 97,397 frames。2–3 次训练 run，测试集每个模型只评一次。
- **π0 上的 AW**：作者也训练了，但结果低于 BC；论文明确“self-trained wormhole 只在 π0.5 上验证”。π0 的 teacher success 为 63.1%，π0.5 为 91.7%。

### 5.2 仿真评测设置

| 场景 | 切换对 | 初始状态 | 切换步 | 试验数/指标 |
|---|---|---|---|---|
| LIBERO-Spatial pair | 两个相同碗：stove ↔ wooden cabinet | 0–19 | early 2；committed 12/22/32/42；late 52/62 | 每行 40 early、160 committed、80 late；success/capture/freeze；Wilson 区间 |
| LIBERO-Goal | 6 个换物体对 + 2 个同物体换目的地对 | 0–7 | 20、40 | 每格 8 trials；entry memory 用 20–29 |
| LIBERO-Object | alphabet soup ↔ salad dressing | 0–9（消融） | 0、8、16 | 每 bar 10 trials |

评测协议：策略接收第三人称与 wrist 图像（224×224），每 5 个 control steps 查询；切换时最多 4 个已排队动作仍执行；成功由 BDDL goal predicate 判定；旧任务先完成算 capture，超时算 freeze；flow noise 由初始状态与 query index 播种，保证同一初始状态的 paired counterfactual。统计用 95% Wilson 区间与 exact McNemar；自训练策略的比较是 pre-registered。

**论文实测主结果（Table 1，Spatial pair，committed 160 trials/行）**：

| 方法 | π0 committed | π0.5 committed | capture（π0/π0.5） | late（π0/π0.5） |
|---|---|---|---|---|
| vanilla | 0.0% [0.0, 2.3] | 1.2% [0.3, 4.4] | 49.4% / 71.9% | 0.0% / 0.0% |
| action-space bridge（4 chunks） | 23.1% | 33.1% | 15.0% / 31.2% | 1.2% / 1.2% |
| retreat-and-restart | 83.8% [77.3, 88.7] | 90.0% [84.4, 93.8] | 0.0% / 0.0% | 72.5% / 92.5% |
| Wormhole | 93.1% [88.1, 96.1] | 93.1% [88.1, 96.1] | 0.0% / 0.0% | 91.2% / 86.2% |

Wormhole 在两者上都达到 93.1%，且没有 capture；freeze 为 6.9%。Completion 中位数 141（π0）/138（π0.5），与 retreat 的 138/136 接近。Discontinuity 0.30/0.32，jerk 0.09，均低于 retreat。作者主张：Wormhole 以约五分之一的动作不连续性，匹配或超过 scripted retreat。

**LIBERO-Goal 的限定**：Wormhole 在 π0.5 上把换物体切换从 25.0%/14.6% 提到 70.8%/91.7%（t_s=20/40），在 π0 上从 4.2%/0% 提到 64.6%/79.2%；但 t_s=20 时 retreat 更强（97.9%/89.6%）。作者归因于两个弱点：stove → drawer 的 far mouth 在 32 trials 中都停短 6–9 cm；wine bottle → bowl 的早期路径重叠，gate 直接交还策略后 freeze 12/16。

**自训练结果（Table 3，π0.5，safe-exit 数据，3 runs）**：AW 在 committed 上 92.1%，BC 85.2%；late 76.7% vs 63.8%。Pre-registered 前两 runs：AW 419 / BC 372 of 480，+9.8 points，95% CI [5.9, 13.7]，p=1.5×10⁻⁶。作者也指出：safe-exit 数据让 AW 提高 2.3 points（不显著），但让 BC 降低 4.0 points（显著），所以部分差距来自 BC 变差。

### 5.3 真实机器人设置

**没有**。论文明确说实验在 LIBERO 仿真中，使用两个同族 flow-based backbone；autoregressive policies 未评测；真实机器人带语音中断的验证是下一步。因此，任何“真实机器人可用”的推断都只能是条件性分析，不是论文实测。

---

## 6. 能力与失败模式 (Capabilities & Failure Modes)

**能做什么**：

- 在 LIBERO-Spatial 的 committed switches 上，Wormhole 对 π0/π0.5 都达到 93.1%，且从未 capture；对 late switches 也分别达到 91.2%/86.2%。
- 不需要切换演示、不需要训练、不访问权重；可与 retreat 组合，作为 far mouth 不可达时的 fallback。
- 能生成切换数据，让 π0.5 通过 AW 学会“听”；`[LOC]` 的 causal 诊断显示，隐藏 `[LOC]` 会让 committed success 从 95.0% 掉到 0.0%（development split）。
- 提供 escape curve 评测协议，把“静态成功”与“执行中切换成功”分开。

**不能做什么 / 失败模式**：

- Wormhole 在 Spatial pair 每个 backbone 的 160 个 committed switches 中失败 11 个；从不 capture，但会 freeze 或 traversal 超时。π0.5 有 3 个、π0 有 2 个在 90 步预算内未到达；π0.5 有 1 个 gate 在 2 cm 内直接交还后 freeze；其余在交付后 2 cm 内 freeze，π0 的 B→A 方向更常见。
- LIBERO-Goal 上 far mouth 可能不可达；位置门控可能跳过必要 traversal；早期 approach 路径重叠时直接交还会 freeze。
- 需要同工作空间的新任务成功执行记忆；far mouth 是绝对末端位姿，物体大位移会失效；固定 lift–transit–settle 假设上方自由空间。
- late switches 会隐式释放已抓物体；真实机器人需要显式 place/release。
- 只在仿真、两个 flow-based backbone 上验证；autoregressive VLA 未评测；真实机器人未验证。
- OT coupling 在这些场景中没有改变 far mouth（与 b-weighted entry 中位差 0.01 mm），所以“OT 对多峰 continuations 的收益”尚未展示。
- Proposition 1 假设 committed region 尖锐；真实数据是软视界，transmission ratio 能测但不能完全量化。
- AW 只在一个 pair 上、2–3 runs；π0 上低于 BC；safe-exit teacher 本身成功率从 250/260 降到 243/260。

### 6.1 隐含假设 (Hidden Assumptions)

1. **任务可从状态辨识**：horizon 与“无语言线性 probe 能读出任务”的时刻重合；如果任务在整条轨迹上都不可由状态区分，黑洞论证不成立。
2. **新指令从初始状态可执行**：协议要求两条指令都能从初始状态执行；真实场景中可能存在物理上不可达的切换。
3. **记忆可得**：Wormhole 需要同工作空间的新任务成功执行；没有记忆时，Mouth B 无法构造。
4. **绝对末端位姿可迁移**：far mouth 是绝对 pose；物体位置变化大时，需要相对物体的 entry 表达。
5. **自由空间与固定路径**：lift–transit–settle 假设工作空间上方无碰撞；真实拥挤场景需要路径规划。
6. **仿真成功谓词可代表任务完成**：BDDL predicate 不检查真实机器人上的物体状态、抓取稳定性与人类安全。
7. **teacher 使用 simulator 物体位置**：AW 的数据来自 oracle relocation teacher，真实部署不能假设知道物体精确位置。

---

## 7. 与相关工作对比 (Comparison)

| 方法/系统 | 关注点 | 架构/训练 | 切换方式 | 适用场景/局限 |
|---|---|---|---|---|
| RT-2 / OpenVLA | 通用 VLA | 离散 action tokens；静态指令 | 无专门切换机制 | 静态评测强；执行中改指令会落入黑洞 |
| π0 / π0.5 | flow-based VLA | flow-matching action expert；action chunk | 直接替换指令 | 本文 backbone；committed 后 transmission 塌缩 |
| SayCan / YAY Robot / Hi Robot | 语言重规划/纠正 | 高层规划或层级 VLA | 假设低层策略会听从 | 不解决低层 committed region 的“聋” |
| SwitchVLA | 执行中切换 | 在 contact-phase 分段演示上训练专用策略，带 rollback/advance | 插值到 canonical start pose | 需要切换演示；接近 retreat 思路 |
| HAMLET | 历史感知 | 给 VLA 加 history awareness | 改善历史利用 | 不直接解决状态重定位 |
| action-space bridge | 动作空间桥接 | 共享 Mouth A/B，执行 2–6 个 chunks | 不移动机器人，只改动作 | committed 14.4–33.1%；仍会 capture |
| retreat-and-restart | 后退重启 | scripted reset 到初始 pose | 向后重定位 | 83.8%/90.0%；动作跳变大，需要 home pose |
| Wormhole | 前进式重定位 | training-free OT + 限速 traversal | 搬到新任务 approach phase | 93.1%；需要记忆与自由空间；真实机器人未验证 |
| AW | 把虫洞蒸进策略 | π0.5 + `[LOC]` + LoRA | 指令只经 `[LOC]` 进入动作 | π0.5 92.1%；π0 失败；需要切换数据 |

**面试 Tip**：如果被问“VLA 为什么执行中改指令不灵”，不要只答“注意力不够”。先讲 committed region 让静态模仿对语言不敏感，再讲 Wormhole 用 OT 把机器人搬到新任务 basin，最后补一句“评测要报 escape curve，不然静态成功率会高估语言使用”。

---

## 8. 精读建议 (Reading Guide)

- **值得精读原文的人**：  
  1. 做 VLA 指令跟随、执行中切换、人机交互中断恢复的研究者；  
  2. 要设计机器人策略评测协议的工程师，尤其是想区分“静态成功”与“切换成功”的团队；  
  3. 做 flow-based policy 插件、状态重定位、OT 规划的研究者。

- **建议章节路径**：先读 §1.2 与 §2，抓住 instruction black hole、transmission ratio、Proposition 1/2；再看 §5 与 Table 1，理解 paired-counterfactual 协议与主要数字；然后看 §4 与 §6，评估工程约束与失败模式；参考文献可跳读。

- **不值得精读的理由**：如果你只关心真实机器人部署、只做 autoregressive VLA、或不接受仿真结论，读摘要与 §6 即可；论文没有真实机器人结果，也没有跨本体泛化证据。

---

[← Back to Theory](./README.md)

关键引用：[论文](https://arxiv.org/abs/2611.34280)