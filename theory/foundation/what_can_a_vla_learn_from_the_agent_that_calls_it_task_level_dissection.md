# 当 VLA 学会“调用它的 Agent”：任务级经验蒸馏（What Can a VLA Learn from the Agent That Calls It? Task-Level Experience Distillation for Embodied Navigation）

> ⚙️ 本文由 Moltbot 自动生成 | 2026-11-15
>
> **论文**：What Can a VLA Learn from the Agent That Calls It? Task-Level Experience Distillation for Embodied Navigation
> **链接**: https://arxiv.org/abs/2611.16246
> **依据版本**：arXiv:2611.16246v1 HTML 全文（2026-11-12，cs.RO）。候选元数据摘要与正文摘要一致，未发现冲突；本文数字均取自该 HTML 正文的表格与段落，未使用候选摘要补写。
> **核心定位**：把“LLM planner 调用 VLA”这段分层系统中**成功 episode 的任务级经验**蒸馏进 VLA 本体，测试时去掉 planner，单模型导航。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 8B NavGPT VLA 在 REVERIE-CE val-unseen 的 3189 个 episode 上，SR 从 35.0% 提到 44.9%（+9.9 点），测试时不再调用 LLM |
| 适合精读 | 在做“LLM planner + VLA tool”分层导航、想砍掉推理期 planner 的人；重点看 §3.3 三条规则、§4.2 主表、§4.4 增益分解 |
| 可以跳过 | 只关心端到端 VLA 预训练、真实机器人部署、或操作（manipulation）任务的人——这篇只在 Habitat 仿真里做导航 |
| 落地可行性 | 中：方法只需要“已有 agent session 记录 + 一次 LoRA 微调”，但前提是能把 planner 工具步回放/渲染成观测 |
| 主要风险 | 单 benchmark、单 VLA、单 planner；作者自己承认“脱轨后不会恢复”；session 数据未声明公开 |

💡 **X-Ray 开场**

这篇论文问了一个很具体的问题：如果一个 LLM planner 每次都能“带”着 VLA 走对，那 VLA 能不能把这段经验学走，之后自己走？作者的答案是能，但**不能直接把 session 当 demonstration 用**——因为 session 里既有 planner 自己开的工具步，也有绕路。H2E 的三条规则就是把这个“脏数据”洗成 VLA 能学的监督信号。对 VLA 研究者的意义：**蒸馏的对象可以从 planner 换成被调用的 VLA 本体**，推理期直接省掉一个 LLM。

📍 **研究全景时间线**

```
2020  REVERIE / R2R-CE：连续环境 VLN 基准
  │
2022  LoRA：低秩适配，让大模型微调变便宜
  │
2024  NaVid（video VLA 导航）、InstructNav（zero-shot 指令导航）
  │
2025  NaVILA、Dynam3D、Uni-NaVid、g3D-LF
  │
2026  StreamVLN、NavGPT-3 分层 runtime → H2E（本文）← 当前位置
  │
局限：单 benchmark、单 VLA、单 planner；未验证真机与跨 planner 迁移
```

（以上年份均来自论文参考文献表中明确标注的发表年份；未在来源中确认的节点不列出。）

## 1. 核心架构/方法总览

### 1.1 系统对比概览

论文里有三个需要**逐模型归属**的实体，不能把它们的属性混在一起：

| 组件 | 输入 | 输出/动作 | 查询频率 | 训练/推理 |
|------|------|-----------|----------|-----------|
| **Base VLA**（8B NavGPT video VLA） | 最近 16 帧 RGB 观测 `o_{t-15:t}` + 指令 `ℓ` | 8 个 waypoints `ŷ ∈ R^{8×3}`（平面偏移 + 朝向），执行第 1 个 | 每执行 1 个 waypoint 后重新查询 | 推理（released model） |
| **Agent planner**（gpt-6-astra） | VLA 的 `h_t` + place graph + 完整 episode 历史 `z_t` | 给 VLA 发指令/子指令；自己移动机器人；走回 place graph 中的位置；看全景/地图；结束 episode | 每个 planner 调用 | 推理 |
| **H2E 蒸馏后的 VLA** | 同 base VLA | 同 base VLA，但 stop 行为被蒸馏 | 同 base VLA | LoRA + action head 微调后推理 |

关键点：

- Base VLA **没有 stop head**。它“停”的条件是**连续两次输出全零**；移动和停止来自同一个回归输出。
- Agent 的 planner 能看到 VLA 看不到的 `z_t`（place graph + 完整历史）；这正是“特权教师”的信息差。
- H2E 不改变 VLA 结构，只改 LoRA 权重和 action head；`BA` 可合并回 `W_0`，部署大小与 base VLA 相同。

### 1.2 关键机制

**① Task-level labels（任务级标签）**

不管这一步是 VLA 执行的、planner 发子指令让 VLA 执行的、还是 planner 用自己的工具移动的，**统统标成“在原任务指令 ℓ 下、VLA 应该输出的动作”**。目标就是 session 接下来实际走的位姿：

```
R(τ) = { (o_≤t, ℓ, y_t) }_{t<T}
y_t  = Φ_K(p_t; p_{t+1:t+K})
```

标签**不依赖** `c_t`（谁移动的）和 `ℓ_t`（当时收到的子指令）。这把“混合来源的运动”统一到测试时的接口上。

**② Progress masking（进度掩码）**

session 里有绕路。step `t` 进入 loss（`m_t = 1`）的条件是：

```
d(p_{t+K}) ≤ d(p_t) + δ,  δ = 0.25 m
```

或者 `p_t, …, p_{t+K}` 中有一个落在目标 1 m 内（Eq. 4 的 stop 区域）。原地转向保持 `d` 不变，留在 loss 里。**被 mask 的步只从 loss 里去掉，仍留在 history `o_≤t` 中**。`d` 只用于造标签，VLA 推理时看不到。

**③ Stop labels（停止标签）**

统一规则，对所有数据源（包括 geodesic expert）：

```
y_t = 0   若 d(p_t) ≤ r_arr,  r_arr = 1 m
```

1 m 比 3 m 成功区更紧，把 stop 目标放在成功区内部。这条规则还会给“VLA 路过但没停”的状态补 stop 标签。

⚡ **Eureka Moment**：把 agent session 里**每一步运动都当成 VLA 在原指令下的动作**，再用“是否朝目标前进”决定这一步进不进 loss——于是 planner 的工具步和绕路不再是噪声，而是“哪些该学、哪些不该学”的信号。

### 1.3 信息流/架构图

```
训练场景（57 scenes）
┌──────────────────────────────────────────────┐
│ Episode 起始位姿                              │
│      ↓                                        │
│ gpt-6-astra planner ⇄ place graph             │
│      │ ① 原指令/子指令  ② 工具运动  ③ 回退     │
│      ↓                                        │
│ 8B NavGPT VLA（frozen）                       │
│      ↓ 执行第 1 个 waypoint，重复查询          │
│ 成功 session τ = {(p_t, o_t, ℓ_t, c_t)}       │
└──────────────────────────────────────────────┘
              ↓ H2E 三条规则
D_H：332,405 个 labeled steps
（task-level labels + progress mask + stop labels）
              ↓ 混合 D_H : D_DART : D_R2R = 0.60 : 0.25 : 0.15
LoRA（r=16, α=32）+ action head，9000 updates
              ↓ 合并 BA 回 W_0
测试时：VLA 单独运行，无 LLM、0 token
```

## 2. 数学核心

📌 **Napkin Formula**

```
min_θ  Σ_s w_s · E_{x_t~D_s}[ m_t · || π_θ(o_≤t, ℓ) - y_t ||_1 ]
```

一句话直觉：**让 VLA 在原指令下回归 agent 实际走过的位姿，但只用“朝目标走”的步来算 loss。**

**目标**：在 base VLA 上用 LoRA 做一次微调，测试时去掉 LLM planner。

**核心公式**（全部用代码块表示）：

```
Eq.2  R(τ) = { (o_≤t, ℓ, y_t) }_{t<T}
      y_t  = Φ_K(p_t; p_{t+1:t+K})

Eq.3  m_t = 1  若 d(p_{t+K}) ≤ d(p_t) + δ  或  p_t..p_{t+K} 中有状态在 1m 内
      δ = 0.25 m

Eq.4  y_t = 0  若 d(p_t) ≤ r_arr
      r_arr = 1 m

Eq.5  L(θ) = Σ_s w_s · E_{x_t~D_s}[ m_t · || π_θ(o_≤t, ℓ) - y_t ||_1 ]
      w_H = 0.60,  w_DART = 0.25,  w_R2R = 0.15

Eq.6  W = W_0 + (α/r) · B A
      r = 16,  α = 32
```

**变量说明**：

| 符号 | 含义 | 取值/来源 |
|------|------|-----------|
| `τ` | 一条 agent session 记录 | Eq. 1 |
| `c_t ∈ {exec, tool}` | 这一步是 VLA 还是 planner 工具移动的 | 只用于统计，不进标签 |
| `ℓ_t` | 当时 VLA 收到的指令（可能是子指令） | 不进标签 |
| `K` | 未来 waypoint 数 | 8 |
| `Φ_K` | 把未来 K 个位姿表达成机器人坐标系下的 waypoints | 与 VLA 的 waypoint 间距一致 |
| `δ` | progress mask 的容差 | 0.25 m |
| `r_arr` | stop label 半径 | 1 m |
| `m_t` | 是否进 loss | 0/1 |
| `w_s` | 三个数据源的混合权重 | 0.60/0.25/0.15 |

**关于时间窗口的两个不同量（容易混）**：

- **历史观测**：`o_{t-15:t}` 是 **16 帧**，均匀间隔下首末跨度是 `(16-1)×Δt = 15×Δt`。这是 VLA 的输入上下文，不是“16 步记忆”。
- **未来预测**：`y_t = Φ_K(p_t; p_{t+1:t+K})`，`K=8`，从当前参考 `p_t` 预测 `p_{t+1}` 到 `p_{t+8}`，终点距当前参考是 `8×Δt`（假设均匀）。这是**预测区间**，不能当作历史上下文或记忆窗口。
- 论文只报了这一个设置（16 帧历史、8 waypoints）；不能把它写成“所有实验共有的记忆窗口”。

**Proposition 1（为什么 L1 + stop 规则能教会停）**：

```
固定 history h，y 是 m=1 的训练样本目标。
E[ ||π(h) - y||_1 | h ] 的每个最小化器 = y 在 h 条件下的逐坐标中位数。
若 P(y = 0 | h) > 1/2，则最小化器唯一，等于 0。
```

直觉：L1 的逐坐标最优是中位数；stop 规则让 1 m 内每个访问状态都有 stop 目标，stop 目标在该处占多数，于是最优输出**精确为 0**。若换成平方损失，最优是均值，只要还有样本在动，均值就不为 0，停不下来。

## 3. 带数字走一遍：玩具例子

> 下面例 A 是**假设的 2D 数值**，用于说明 masking 与 stop 规则；例 B 用论文 Table 2/Table 7 的**实测数字**复算 Eq. 12。

**例 A（假设）**：目标 `g` 在原点，当前 `p_t = (2.0, 0)`，`d(p_t) = 2.0 m`，`K = 8`。

- 情形 1（朝目标走）：未来 8 个位姿沿 -x 每步 0.05 m，`p_{t+8} = (1.6, 0)`，`d = 1.6 m`。`d(p_{t+8}) - d(p_t) = -0.4 m ≤ 0.25 m`，且中间没有进 1 m 圈 → **`m_t = 1`**，进 loss。
- 情形 2（绕路）：`p_{t+8} = (2.4, 0)`，`d = 2.4 m`，增加 0.4 m > 0.25 m，中间最近只到 1.9 m → **`m_t = 0`**，不进 loss，但 `o_≤t` 仍留在 history。
- 情形 3（进 stop 圈）：若 `p_t = (0.9, 0)`，`d = 0.9 m ≤ 1 m` → **`y_t = 0`**；0.9 m 也在 3 m 成功区内，停在这里 SR = 1。

**Proposition 1 验算（假设）**：某 history `h` 下 3 个样本，`y` 分别为 `(0,0,0)`、`(0,0,0)`、`(0.5,0,0)`。`P(y=0|h) = 2/3 > 1/2`。逐坐标中位数：x = median(0,0,0.5) = 0，y = 0，z = 0 → 最优 `π(h) = (0,0,0)`。若用 L2，均值是 `(0.167,0,0)`，非零，不会精确停。

**例 B（用论文实测数字复算 Eq. 12）**：

```
SR^c - SR^b = (OSR^c - OSR^b)·C^c  +  OSR^b·(C^c - C^b)
                reach term              stop term
C = SR / OSR
```

H2E vs Base VLA（Table 2、Table 7）：

- `C^c = 44.9/50.1 = 0.896`，`C^b = 35.0/52.0 = 0.673`
- reach term = `(50.1 - 52.0) × 0.896 = -1.70` → 论文写 `-1.7` ✓
- stop term = `52.0 × (0.896 - 0.673) = 11.60` → 论文写 `+11.6` ✓
- 合计 `-1.7 + 11.6 = +9.9` → 论文 `+9.9` ✓

H2E vs Geodesic adaptation：

- `C^b = 34.9/38.5 = 0.906`
- reach = `(50.1 - 38.5) × 0.896 = 10.40` → 论文 `+10.4` ✓
- stop = `38.5 × (0.896 - 0.906) = -0.39` → 论文 `-0.4` ✓
- 合计约 `+10.0`，论文写 `+9.9`（舍入）✓

结论：**对 base VLA，增益几乎全来自“停得准”；对 geodesic adaptation，增益几乎全来自“找得到”。**

## 4. 工程视角

**延迟模型**（Eq. 7）：

```
t_H   = T·c_π + Σ_{k=1}^{K} c_LLM(n_k)
t_H2E = T·c_π
```

`c_π` 是一次 VLA 步时间，`K` 是 planner 调用次数，`c_LLM(n_k)` 是 `n_k` token 的调用时间。H2E 的 LLM 项为 0。

**实测（Table 6，100 个 episode，中位数）**：

| 指标 | Agent + base VLA | Base VLA | Geodesic adapt. | H2E |
|------|------------------|----------|-----------------|-----|
| SR | 53 | 38 | 37 | **57** |
| OSR | 70 | 59 | 50 | 64 |
| NE (m) | 5.50 | 8.60 | 9.29 | 5.96 |
| 时间 (s) | 423 | 196 | 194 | **114** |
| LLM token | 70k | 0 | 0 | **0** |

- H2E 比 agent 快 `423/114 ≈ 3.7×`；100 个 episode 上 agent 总共用 16.7M token。
- H2E 甚至比 base VLA 还快，因为 episode 更短：val-unseen 上 H2E 中位数 49 步/episode，base VLA 66 步。
- LoRA 可合并（Eq. 6），**部署大小与单步成本同 base VLA**。
- 论文未给 `c_π` 的绝对值、显存占用、量化误差、抖动数据；这些不能编。

**训练成本**：主结果 9000 updates、约 28 小时、4×NVIDIA L40。Table 4 的受控对比用 3000 updates（= 48,000 samples，samples per update = 16），每 500 updates 存一次 checkpoint，在 30 个其他训练 episode 上选最终 checkpoint。注意：**9000 updates 与 3000 updates 是两个不同实验设置**，不要混写。

**控制频率**：VLA 每次查询输出 8 个 waypoints，机器人执行第一个 waypoint 后再次查询；附录 B 称 VLA 最多运行 500 个 low-level steps。评测频率是“最终 checkpoint”（Table 4 是选出的 checkpoint）。

## 5. 数据与评测

**预训练 vs 微调（分开写）**：

- 论文没有给 base VLA 的预训练数据构成，只说它是 released 8B NavGPT video VLA [51]。H2E 和 geodesic adaptation 都是在这个 base 上做 LoRA 微调，不是从头预训练。
- Table 2 里 Dynam3D（4M+ samples）、D3D-VLP（10M samples）是**别人的基线**，不是本文 VLA 的预训练量，不能混用。

**仿真评测**：

- Benchmark：REVERIE-CE，Habitat + MP3D。
- 主结果：官方 val-unseen 全部 **3189 个 episode，10 个未见场景**。
- 与 agent 对比：**100 个 episode，来自 26 个训练集外场景**（Table 6）。
- 成功判据：停在 `d(p_T) ≤ 3 m`；每 episode 最多 500 步、2400 s 时限。
- 指标：SR、OSR、SPL、NE（Eq. 8）。

**真实机器人**：论文**没有**真实机器人实验。所有结果都在 Habitat 仿真中，不能外推到真机。

**数据组成**：

| 数据 | 规模 | 用途 |
|------|------|------|
| Agent sessions | 2395 条，57 个训练场景 | H2E 主数据 `D_H` |
| Labeled agent steps | 332,405 | 其中 VLA 原指令 53.9%、子指令 20.5%、工具 24.6%、tail 1.1%；masked 23.3%；stop labels 30,181 |
| Table 4 受控对比 | 504 条 session | 与 geodesic adaptation 同 recipe 对比 |
| Table 6 对比 agent | 63 条 session | 从 63 条蒸馏出 57% SR |
| DART + R2R-CE | 48k samples | 混合数据，权重 0.25 / 0.15 |

**标注类型**：task-level labels（Eq. 2）、progress mask（Eq. 3，δ=0.25 m）、stop labels（Eq. 4，r_arr=1 m）。

**统计**：scene bootstrap 重采样 10 个场景 10000 次，主结果 95% 区间 3.9–16.3 点；McNemar `p < 1e-22`。1198 个两者都到达的 episode 上，H2E 成功率高 24.5 点（95% CI 17.4–32.3）。

**数据开放范围**：论文未声明发布 agent session 数据或 H2E 权重，也未给代码仓库。REVERIE-CE / Habitat / MP3D / R2R-CE 是公开资源；DART 是噪声注入/模仿学习方法（用于生成 D_DART），不是数据集或 benchmark。本文没有给链接，此处不补。

## 6. 能力与失败模式

**能做什么（论文实测）**：

- **停得准**：base VLA 只有 67% 的 goal-region visit 转成 success（544 个 episode 到了区域但停在区域外）；H2E 提到 89%，late misses 降到 168。
- **找得到新目标**：H2E 到达了 401 个 base VLA 从未到达的 goal region，其中 355 个成功。
- **比 agent 便宜**：SR 57 vs 53，0 token，114 s vs 423 s。
- **小 adapter 够用**：rank 16 在 861 个 episode 上 SR 45.6%，rank 64 只有 33.2%（Table 5）。

**不能做什么 / 失败模式**：

- **脱轨后不会恢复**：作者在 Limitations 明确说，蒸馏后的 VLA 还不会从偏离中恢复，这需要 agent 的地图和记忆。Table 9 的 Rec. 列：H2E 6/40，base VLA 5/39，geodesic 0/45。
- **早停**：geodesic adaptation 学会了停，但停太早，OSR 只有 38.5%，H2E 是 50.1%。
- **只在仿真、单 benchmark、单 VLA、单 planner**：换 planner（gpt-6-astra 之外）或换 VLA 是否成立，论文未验证。
- **agent session 的“脏”程度**：45.1% 的运动来自子指令或工具（20.5% + 24.6%），23.3% 的步被 mask；如果换一个更“干净”或更“脏”的 agent，比例会变，方法是否仍成立未知。

### 6.1 隐含假设

1. **“下一步位姿”就是好监督**：假设 agent 走出来的路径在局部是可模仿的，即使它包含工具步。论文用 progress mask 缓解，但没有证明“被 mask 的步”一定不该学。
2. **1 m stop 半径跨本体可迁移**：`r_arr = 1 m` 是针对 REVERIE-CE 的 3 m 成功区设计的；换机器人尺寸或换 benchmark 是否仍适用，未验证。
3. **16 帧历史足够**：VLA 只看到 `o_{t-15:t}`，planner 看到完整历史 + place graph。论文说“VLA 学会了找新目标”，但没有分解“哪些找目标能力来自历史窗口内的信息”。
4. **L1 + 多数票机制在数据分布偏移下仍成立**：Proposition 1 是逐 history 的条件结论；实际中 `P(y=0|h) > 1/2` 是否在每个 1 m 内状态都成立，论文没有逐状态验证。
5. **单次 LoRA 微调不会破坏 base 能力**：论文报告了 SR/OSR/SPL/NE，但没有报告 base VLA 在其他任务上的回归测试。

## 7. 与相关工作对比

| 方法 | 蒸馏对象 | 测试时是否调 LLM | 训练数据 | 本文报告的结果 |
|------|----------|------------------|----------|----------------|
| **H2E（本文）** | **VLA 本体** | **否** | agent sessions + DART + R2R | SR 44.9（val-unseen 3189） |
| NavGPT-3 [51] | 不蒸馏，VLA frozen | 是 | – | 作为 base VLA 来源 |
| TinyAstral [9] | planner → 4B student | 是（student） | – | 论文未给可比数字 |
| Guava [21] | agent → compact agent | 是 | – | 论文未给可比数字 |
| OmniBrain-0 [41] | planner → 更小模型 | 是 | – | 论文未给可比数字 |
| Geodesic adaptation | VLA | 否 | DART + R2R，48k samples | SR 34.9（同 3189） |
| Base VLA | – | 否 | released model | SR 35.0（同 3189） |

（表中“论文未给可比数字”指本文没有在相同 3189 episode 设置下报告这些方法的 SR，不是它们没有结果；不能替它们编数字。）

**与 geodesic adaptation 的关键差异**：两者用同样的 LoRA recipe、同样的场景、同样的 updates；H2E 只是**多加了 agent session**。Table 4 用 504 条 session 就到 43.2%，geodesic 停在 34.9%——说明增益来自 session 本身，不是“在 REVERIE-CE 上多训”。

**面试 Tip**：被问到“这篇和 planner 蒸馏有什么区别”时，一句话答：**“它蒸馏的是被调用的 VLA，不是调用者；测试时 LLM 完全移除，所以省的是推理期 token 和延迟，不是模型参数。”**

## 8. 精读建议

- **值得精读原文的人**：
  1. 正在搭“LLM planner + VLA tool”分层导航系统、想把 planner 砍掉的研究者；
  2. 关心“特权教师（place graph + 完整历史）能教给学生什么、教不会什么”的蒸馏研究者；
  3. 需要评估“把 agent session 转成 VLA 训练数据”工程可行性的工程师。

- **建议章节路径**：先读 §3.3 三条规则（Eq. 2–4）→ 再看 §3.4 的 Proposition 1 和 Eq. 5 → 然后 §4.2 主表 + §4.4 的 Eq. 12 分解 → 最后 §4.5 ablation 和 Appendix D 的数据统计。§2 Related Work 可在读完方法后回看，用来定位与 TinyAstral / Guava / OmniBrain-0 的区别。

- **不值得精读的理由**：如果你不做具身导航、或已经熟悉“agent session → 学生策略”这类蒸馏、或只关心真机部署，读摘要和 §4.2 两张表就够了；这篇没有真实机器人实验，也没有跨 benchmark 的泛化验证。

---

[← Back to Theory](./README.md)

**关键引用**：https://arxiv.org/abs/2611.16246