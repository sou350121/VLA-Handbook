# 从失败中学更多：VLA 后训练的事后强化学习 (Learning More from Less: Reinforcement Learning from Hindsight)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-07-14
>
> **论文**: Learning More from Less: Reinforcement Learning from Hindsight
> **链接**: https://arxiv.org/abs/2607.09042
> **作者**: Iris Xu, Sunshine Jiang, John Marangola, Nitish Dashora, Richard Li, Thomas Liu, Zexue He, Yuheng Zhi, Alex Pentland, Pulkit Agrawal, Zhang-Wei Hong
> **机构**: MIT, MIT-IBM Computing Research Lab, Stanford, UC San Diego
> **核心定位**: 用 VLM 对 RL 后训练中失败的 rollout 进行语言级事后重标签，将"无用失败"转化为跨任务训练信号，在 LIBERO-PRO 上实现 5 倍样本效率提升。

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | 用单个 VLM 同时重标签失败 rollout 的指令和奖励，让 GRPO 从原本丢弃的零奖励组中提取学习信号 |
| 適合精讀 | 在做 VLA 后训练/RL 精调、面临样本效率瓶颈的研究者；关注 HER 在语言空间扩展的工作 |
| 可以跳過 | 只关心模仿学习或离线 RL、不涉在线 RL 后训练的场景 |
| 落地可行性 | 中（需要一个大 VLM 做 relabeler，推理成本较高；但算法本身即插即用） |
| 主要風險 | VLM relabeler 的质量直接决定效果；对完全无意义的随机行为无法生成有效 hindsight 指令 |

💡 **X-Ray 开场**
VLA 的 RL 后训练面临一个冷酷现实：早期策略几乎每次 rollout 都失败，稀疏奖励让所有失败组都被 GRPO 丢弃。LfH 的核心洞察是——一个任务的失败可能是另一个任务的成功。用一个 VLM 给失败轨迹"重新命名"：机器人本想关微波炉却拿了杯子，那就把它重标签为"拿起杯子"的成功样本。这样，同样的 rollout 数据，训练信号翻倍。

📍 **研究全景时间线**
```
[2017] HER (Andersen et al.) — 视觉空间目标重标签 → [2020+] 图像/状态级 hindsight 扩展
→ [2025] 语言级 hindsight 在游戏环境初探 → [2026-07] LfH ← 当前位置：VLM 驱动的
语言+奖励双重重标签，首次在 VLA RL 后训练中验证有效性
→ (局限) 依赖 VLM relabeler 质量，尚未在更大规模真实机器人上验证
```

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 组件 | 输入 | 输出 | 训练/推理 | 频率 |
|------|------|------|-----------|------|
| VLA 策略 π_θ | 观测 o_t + 指令 g | 动作 a_t | 训练 (GRPO) | 每步推理 |
| VLM relabeler M_ψ (指令) | 锚点轨迹 τ_i* (RGB序列) | hindsight 指令 g' | 推理 | 每组失败 rollout 调用 1 次 |
| VLM relabeler M_ψ (奖励) | 轨迹 τ_i + g' | 奖励 R̃_i ∈ {0, 0.5, 1} | 推理 | 每组 K 次调用 |
| GRPO 优化器 | 原始组 G + hindsight 组 G̃ | 策略参数更新 | 训练 | 每 batch |

### 1.2 关键机制 (Key Mechanism)

**为什么这样设计？**

1. **语言级重标签而非状态级**：传统 HER 在状态空间重标签目标，但 VLA 的观测是高维 RGB，状态空间重标签困难且难以泛化。语言是 VLA 的天然接口——VLA 本身就是语言条件化的，重标签在语言空间进行可以直接利用 VLA 的跨指令泛化能力。

2. **单 VLM 双职责**：用同一个 VLM（Qwen3-VL-235B）既生成 hindsight 指令又评分奖励，简化了 pipeline，避免了多模型间的误差累积。

3. **仅对低奖励组激活**：高奖励组（平均奖励 ≥ η）已经有可靠的训练信号，重标签反而会引入噪声。LfH 只在"否则会被丢弃"的组上工作，不干扰正常学习。

4. **重要性修正**：轨迹是在原始指令 g 下采样的，但优化目标是 hindsight 指令 g'。通过重要性采样比 π_θ(a|o,g') / π_θ_old(a|o,g) 修正分布偏移。

⚡ **Eureka Moment**：一个任务的失败 rollout，本质上是另一个（未命令的）任务的成功样本——只要你能用语言正确描述它实际做了什么。

### 1.3 信息流/架构图 (Flow / Diagram)

```
┌─────────────────────────────────────────────────────────────┐
│                    GRPO Rollout Phase                       │
│  g ~ P_g  →  π_θ_old(o_t, g) → a_t  →  τ = (o_0,a_0,...) │
│  R(τ, g) = {0 or 1}  (sparse binary reward)               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │  mean(R) < η ? │ ──No──→ 标准 GRPO 训练
              └────────┬───────┘
                       │ Yes (low-signal group)
                       ▼
        ┌──────────────────────────────┐
        │   VLM Relabeler M_ψ          │
        │                              │
        │  Step 1: 选锚点 τ_i*         │
        │          (R_i*=0 的随机样本)  │
        │                              │
        │  Step 2: g' ~ M_inst(τ_i*)   │
        │          生成 hindsight 指令   │
        │                              │
        │  Step 3: R̃_i = M_rew(τ_i,g') │
        │          对组内所有轨迹评分    │
        │          R̃ ∈ {0, 0.5, 1}     │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │   Hindsight GRPO Update       │
        │                               │
        │  Â_i = (R̃_i - μ̃_R) / (σ̃_R+δ)│
        │  r̃_i,t = π_θ(a|o,g') /       │
        │            π_old(a|o,g)       │
        │  ℒ_H-GRPO = PPO-clip + KL    │
        └──────────────┬───────────────┘
                       │
                       ▼
        ℒ_LfH = ℒ_GRPO + λ · ℒ_H-GRPO
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：
```
ℒ_LfH(θ) = ℒ_GRPO(原始组) + λ · ℒ_GRPO(VLM重标签的hindsight组)
```

**目标**：从原本被 GRPO 丢弃的零奖励组中恢复学习信号。

**核心方程组**：

```
// 1. GRPO 原始优势计算 (组内归一化)
Â_i = (R_i - μ_R) / (σ_R + δ)

// 2. 指令重标签 (VLM 生成)
g' ~ M_ψ^inst(· | τ_i*),  i* ~ Unif({i : R_i = 0})

// 3. 奖励重标签 (VLM 评分)
R̃_i = M_ψ^rew(τ_i, g') ∈ {0, 0.5, 1}

// 4. Hindsight 优势计算
Ã_i = (R̃_i - μ̃_R) / (σ̃_R + δ)

// 5. 重要性修正 (跨指令分布偏移)
r̃_i,t(θ) = π_θ(a_i,t | o_i,t, g') / π_θ_old(a_i,t | o_i,t, g)

// 6. 联合优化目标
ℒ_LfH(θ) = ℒ_GRPO(θ) + λ · ℒ_H-GRPO(θ)
```

**变量说明**：

| 符号 | 含义 |
|------|------|
| g | 原始命令指令 |
| g' | VLM 生成的 hindsight 指令 |
| τ_i | 第 i 条轨迹 (o_0, a_0, ..., a_{T-1}, o_T) |
| R_i | 原始稀疏奖励 (0 或 1) |
| R̃_i | hindsight 奖励 (0/0.5/1) |
| η | 激活阈值 (平均奖励 < η 时触发重标签) |
| λ | hindsight 损失权重 |
| i* | 锚点轨迹索引 (从失败轨迹中随机选) |
| K | 组大小 (每组轨迹数) |

> 符号与论文保持一致。VLM relabeler M_ψ 包含两个子模块：M_inst 生成指令，M_rew 评分奖励。

**直觉**：LfH 不改变 GRPO 的核心优化机制——它只是在"否则无信号可学"的时候，用 VLM 创造一个新的学习任务，让同一条轨迹同时为原始任务和 hindsight 任务服务。

## 3. 带数字走一遍：玩具例子 (Worked Example)

**场景**：命令 g = "关闭微波炉"，组大小 K = 4。

**Rollout 阶段**：
```
τ_1: 碰到杯子 → R_1 = 0
τ_2: 拿起杯子 → R_2 = 0
τ_3:  bump 杯子 → R_3 = 0
τ_4: 碰到微波炉但没关 → R_4 = 0
```

**标准 GRPO**：μ_R = 0, σ_R = 0 → 所有 Â_i = 0 → 该组被丢弃，无梯度。

**LfH 介入**：

1. 检测到 mean(R) = 0 < η → 激活重标签
2. 选锚点 i* = 2 (τ_2: 拿起杯子)
3. VLM 生成 g' = "拿起杯子"
4. VLM 评分：
   ```
   R̃_1 = 0   (碰到杯子 ≠ 拿起)
   R̃_2 = 1   (成功拿起杯子)
   R̃_3 = 0   (bump ≠ 拿起)
   R̃_4 = 0.5 (模糊：碰到了但没拿起)
   ```
5. Hindsight 优势：
   ```
   μ̃_R = (0+1+0+0.5)/4 = 0.375
   σ̃_R ≈ 0.443
   Ã_1 = (0 - 0.375) / (0.443 + δ) ≈ -0.80
   Ã_2 = (1 - 0.375) / (0.443 + δ) ≈ +1.41
   Ã_3 = (0 - 0.375) / (0.443 + δ) ≈ -0.80
   Ã_4 = (0.5 - 0.375) / (0.443 + δ) ≈ +0.27
   ```
6. 现在 Ã_2 > 0，τ_2 在 g' = "拿起杯子" 下获得正优势 → 策略学会"拿起杯子"
7. 同时 ℒ_GRPO 仍在优化原始指令 g → 策略同时学习两个任务

**关键**：同一条轨迹 τ_2，在原始指令下是失败 (R=0)，在 hindsight 指令下是成功 (R̃=1)。一条数据，两份信号。

## 4. 工程视角 (Engineering View)

| 维度 | 分析 |
|------|------|
| VLM Relabeler 成本 | 每次 GRPO step 需要调用 VLM：指令生成 1 次/组 + 奖励评分 K 次/组。使用 Qwen3-VL-235B，单次推理约数秒。如果每组 4 条轨迹，每 step 约 5 次 VLM 调用 |
| 吞吐影响 | VLM 推理是瓶颈。论文中 GRPO 本身也需要 rollout 收集（物理机器人更慢），VLM 推理可与 rollout 并行 |
| 内存 | 需要同时维护原始组和 hindsight 组的 buffer，内存翻倍。但 buffer 大小通常不大（几百条轨迹） |
| 部署约束 | VLM relabeler 仅在训练时需要，推理时不需要。部署的 VLA 策略 π_θ 大小不变 |
| 稳定性 | 重要性修正 r̃_i,t 可能引入方差——当 g 和 g' 差异大时，π_θ(a|o,g') 和 π_θ_old(a|o,g) 分布偏移大，重要性比可能很大。论文通过 KL penalty 和 clip 控制 |
| λ 超参 | hindsight 损失权重。论文未详细讨论调参，但直觉上 λ 过大可能让策略偏离原始任务 |

**工程含义**：LfH 不改变部署时的 VLA 架构——它只是一个训练技巧。训练时需要一个大 VLM 做 relabeler，但推理时只需要策略网络。对于资源受限的场景，可以用较小的 VLM（如 Qwen2.5-VL-7B）替代 235B 版本，可能牺牲一些重标签质量。

## 5. 数据与评测 (Data & Eval)

**数据设置**（论文 §5）：

| 设置 | 详情 |
|------|------|
| 初始化策略 | RLinf-Pi0.5-LIBERO-SFT (在 4 个 LIBERO 套件上用少量示范训练) |
| 评测基准 | LIBERO-PRO (OOD 任务，保持场景但改变任务规格) |
| 训练步数 | 40 steps (π_0.5) / 200 steps (GR00T) / 60 steps (OpenVLA-OFT) |
| 组大小 K | 论文未明确给出，但 GRPO 通常 K=4 |
| 物理机器人 | Franka FR3, 10 个 SFT 任务 (每任务 20 个 SpaceMouse 示范) + 1 个 held-out 任务 |

**评测指标**：
```
Gain(t) = SR_t / SR_0 - 1
```
其中 SR_0 是初始策略成功率，SR_t 是训练 t 步后的成功率。这个指标衡量相对初始策略的改进速度。

**关键结果**（论文 Figure 3）：

| 方法 | LIBERO-PRO 样本效率 | 保持组比例 |
|------|---------------------|-----------|
| GRPO (baseline) | 1× (30 steps 达到最终性能) | 20-40% |
| GRPO + RoboMETER | ~2× | 20-40% |
| **GRPO + LfH** | **~5×** (8 steps 达到 GRPO 最终性能) | **70-80%** |

**物理机器人结果**（论文 §5.4）：
- 128 rollouts: LfH ≈ 2× GRPO 成功率
- 160 rollouts: LfH 56% vs GRPO 22%

## 6. 能力与失败模式 (Capabilities & Failure Modes)

### 能做什么

| 能力 | 证据 | 条件 |
|------|------|------|
| 从失败中提取信号 | Figure 2(a): 保留 70-80% 组 vs GRPO 的 20-40% | 失败轨迹必须有"可描述的行为" |
| 跨 VLA 骨干泛化 | Figure 3(c): π_0.5, GR00T, OpenVLA-OFT 均有提升 | VLA 需有语言泛化能力 |
| 物理机器人迁移 | Figure 4(b): Franka FR3 上 2.5× 提升 | 需要 SFT 初始化策略 |
| 超越密集奖励 | 优于 GRPO+RoboMETER | 在低成功率 regime 下优势明显 |

### 不能做什么（失败模式）

| 失败模式 | 原因 |
|----------|------|
| 对完全随机行为无效 | VLM 无法从无意义运动中提取 "meaningful behavior"，这些组被标记为 "uninteresting" 并丢弃 |
| 对与原始任务完全无关的行为可能有害 | 如果 hindsight 指令与原始任务语义距离太远，重要性修正可能不稳定 |
| 依赖 VLM 质量 | 使用 Qwen3-VL-235B；小模型可能生成不准确的重标签 |
| 20% 的组仍然被丢弃 | 论文 Figure 2(a) 显示仍有 20% 的组无法被重标签 |

### 6.1 隐含假设 (Hidden Assumptions)

1. **VLA 有足够的语言泛化能力**：LfH 假设策略能在 hindsight 指令 g' 下从在 g 下采样的轨迹中学习。如果 VLA 的语言泛化能力弱（即 π(a|o,g') 和 π(a|o,g) 差异很大），重要性修正会引入大方差。

2. **失败轨迹包含"可描述的行为"**：LfH 假设大部分失败不是完全随机的，而是"做了一件不同的事"。如果策略早期输出接近随机噪声，VLM 无法生成有意义的 hindsight 指令。

3. **VLM relabeler 的评分是可靠的**：M_rew 输出 {0, 0.5, 1} 三值奖励。如果 VLM 对"是否完成了 g'"判断不准，会引入噪声奖励信号。论文未对 relabeler 准确率做独立评估。

4. **hindsight 任务与原始任务有某种关联**：虽然论文 Figure 2(b) 显示 hindsight 指令可能与目标任务语义不相关（"关闭微波炉" → "拿起杯子"），但 relabeling 仍然有帮助。这暗示即使不相关的任务也能提供"对比 grounding 信号"。但这个机制尚未被严格验证。

## 7. 与相关工作对比 (Comparison)

| 方法 | 重标签空间 | 在线/离线 | 适用场景 | 与 LfH 关系 |
|------|-----------|-----------|----------|------------|
| HER (2017) | 状态/图像目标 | 在线 RL | 低维状态空间 | LfH 是 HER 的语言空间推广 |
| 图像 hindsight (2020+) | 图像目标 | 在线 RL | 视觉任务 | 需要目标生成器；LfH 用 VLM 直接生成语言 |
| RoboMETER (2025) | 密集进度奖励 | 在线 RL | 机器人操作 | 互补：RoboMETER 给同一任务更密集反馈，LfH 创建新任务 |
| 语言数据增强 (IL/离线 RL) | 指令合成/改写 | 离线 | 数据扩充 | LfH 是在线的，且同时重标签奖励 |
| 推理时引导 (Wu et al.) | 动作选择 | 推理时 | VLA 推理 | LfH 将信号烘焙到权重中，非测试时引导 |

**面试 Tip**：当被问到"LfH 和 HER 的区别"时，回答："HER 在状态空间重标签目标——把实际到达的状态作为新目标。LfH 在语言空间做同样的事，但用 VLM 把轨迹'翻译'成语言指令。关键区别是：HER 需要目标空间与状态空间对齐，而 LfH 利用 VLA 天然的语言条件化接口，不需要显式的目标表示。"

## 8. 精讀建議 (Reading Guide)

**值得精讀原文的人**：
- 做多模态具身 Agent RL 后训练的研究者——LfH 直接解决了冷启动问题
- 关注 HER 在语言/视觉空间扩展的研究者——这是 HER 思想在 VLA 时代的首次完整实现
- 需要评估 VLA 后训练样本效率的工程团队——5× 提升在实际机器人部署中意义重大

**建議章節路徑**：
1. 先读 §3 (Preliminaries) + §4 (Method)：理解 GRPO + LfH 的数学框架
2. 再看 §5.2 (Sample Efficiency) + §5.3 (Ablation)：验证核心 claim 的稳健性
3. 可跳 §2 (Related Work) 除非你特别关心 HER 谱系

**不值得精讀的理由**：
- 如果你只做模仿学习或离线 RL，不涉及在线 RL 后训练，这篇的方法不直接适用
- 如果你已经熟悉 HER 且不做 VLA，核心思想（事后重标签）没有太多新内容——创新主要在工程实现（VLM relabeler + 重要性修正）

---
[← Back to Theory](./README.md)

**关键引用**：
- 论文: https://arxiv.org/abs/2607.09042
- HER (原始): https://arxiv.org/abs/1707.01495
- GRPO: https://arxiv.org/abs/2607.09042 (引用 [24])
- RoboMETER (dense reward baseline): https://arxiv.org/abs/2607.09042 (引用 [13])
- LIBERO-PRO: https://arxiv.org/abs/2607.09042 (引用 [32])
- RLinf 框架: https://rlinf.readthedocs.io/
