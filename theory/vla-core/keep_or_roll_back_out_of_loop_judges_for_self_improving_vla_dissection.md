# 环外裁判：自提升 VLA 与端到端具身策略的保留/回滚判定 (Keep or Roll Back? Out-of-Loop Judges for Self-Improving VLA and End-to-End Embodied Policies)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-11-27
>
> **论文**: Keep or Roll Back? Out-of-Loop Judges for Self-Improving VLA and End-to-End Embodied Policies
> **链接**: https://arxiv.org/abs/2611.33794
> **核心定位**: 把“新策略是否更好”的判定权从环内 judge 移到 improver 看不见的 sealed judge，用 anytime-valid 序列检验给出错误率上界；并把假增益拆成 judge error 与 transfer gap。

**版本说明**：本文依据 arXiv:2611.33794v1 的 HTML 全文（2026-11-25）撰写。来源记录未提供独立的官方 PDF 元数据摘要（`abstract_verification` 为空），因此不存在“用候选摘要替换 PDF 主张”的情况；下文数字与主张均取自 HTML 正文的章节、表格与附录。论文在 §4 与 §B.1 的 run 计数口径略有差异，文中会单独注明。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 环内 judge 即使在自己数据上无误差，也测不到 transfer gap；sealed judge + 序列检验才能把“保留一个更差策略”的概率控制在 α 以内 |
| 适合精读 | 做 VLA/驾驶策略在线自提升、RL fine-tuning、模型验收协议的人；重点看 §3、§5.2、§5.4 |
| 可以跳过 | 只关心 VLA 架构、动作 chunk、记忆机制的人——这篇不提供这些 |
| 落地可行性 | 中：协议清晰、统计保证明确，但需要独立评测池与 2× paired rollout，实车成本高 |
| 主要风险 | sealed pool 的代表性；sealed judge 本身仍是 proxy；未决即回滚会损失真增益 |

💡 **X-Ray 开场**

先问三个问题：自提升闭环里，judge 既提供训练信号又决定“是否保留”，证据不独立会怎样？论文在 LIBERO、NAVSIM 和一辆全尺寸实车上各跑了一遍，发现环内 judge 会把失败判成成功，也会把“只在自己评测帧上成立”的增益当成真增益。对 VLA 研究者意味着：不要把“新 checkpoint 在旧评测集上更高”当成提升；需要一个 improver 看不见的 sealed pool，并用有错误率保证的检验来决定 keep/rollback。

📍 **研究全景时间线**

```
2023  LIBERO 基准 / reward overoptimization 缩放律
  → 2024  NAVSIM / betting confidence sequence
  → 2025  HIL-SERL（环内 classifier + human）
  → 2026  本文：sealed judge + anytime-valid acceptance（v1 2026-11-25）
```

只列来源可确认的节点；本文位置在 2026。局限：loop 主要在仿真中运行，实车只测一个已发表改进步。

## 1. 核心架构/方法总览

### 1.1 系统对比概览

| 维度 | LIBERO manipulation（仿真） | NAVSIM driving（仿真） | 全尺寸实车 |
|------|------|------|------|
| 任务/benchmark | LIBERO-Object，3 个任务（alphabet soup / cream cheese / BBQ sauce 放入篮子） | NAVSIM，unseen city（Boston） | 三条真实路线：主路线 144.4 m（左弯约 55°）、直线 60.0 m、右弯 143.1 m |
| start policy 架构 | OpenVLA-OFT（LIBERO-Spatial checkpoint，LoRA few-shot 适配） | DiffusionDrive（从 scratch 训练） | ReCogDrive-2B，IL 与 RL 两版；共享 VLM backbone，只差 diffusion planner 权重 |
| 预训练/数据 | released OpenVLA-OFT checkpoint；每任务用少量 demonstrations 适配 | 20,000 帧 Las Vegas / Pittsburgh / Singapore，20 epochs | 论文未说明两版在自车数据上 fine-tune；实车为零样本部署 |
| improver | rejection-sampling fine-tuning；每代 300 个 fresh start conditions/task，LoRA rank 32、2,000 steps、batch 8、lr 1e-4 | best-of-20 self-distillation；每代 20,000 帧、每帧 20 个 proposal、critic 选 1 个、训练 5 epochs、与 expert 帧 1:1 混合 | 已发表的 RL 改进步（单步，不是多代 loop） |
| 环内 judge | success classifier，读 episode 末帧，每代用自己的标签重训 | PDMS critic，从 frozen encoder 特征预测 PDMS；每代用 true labels 或 own pairwise preferences 刷新 | simulator score（NAVSIM PDMS） |
| sealed judge | simulator success predicate，unseen start conditions | Boston 的 PDMS，16,685 帧 / 280 logs | 实车：第一次 intervention 前完成的路线比例 |
| 决策规则 | 序列检验；预算 300 pairs（环内）/最多 1,000 pairs（sealed） | 序列检验；环内 5,000 帧 / sealed 16,685 帧 | 盲配对驾驶；序列检验在 6 pairs 后仍未决 |
| runs | 主对比 9 sealed + 9 classifier = 18 runs × 5 代；另有 9 个 true-success run × 5 代 | 3 seeds × 3 variants = 9 main runs × 5 代；3 control runs × 5 代 | 1 个改进步；主路线 6 pairs、直线 3 pairs、右弯 1 pair |
| 记忆机制（论文是否描述） | 未描述 | 未描述 | 未描述 |
| 数据开放范围 | LIBERO-Object 为公开 benchmark | NAVSIM 为公开 benchmark；Boston 帧只作 sealed pool、不参与训练 | 论文未说明实车数据公开；正文未给代码仓库地址 |

> **计数口径说明**：§4 写“3 tasks × 3 seeds × 3 variants”，§B.1 写“eighteen runs of five generations”，Table 3 按 variant 各列 9 runs。按 Table 3 与 §5.1 的“45 classifier-decided generations / 90 generations”推算，主对比是 9 sealed + 9 classifier = 18 runs × 5 代 = 90 代；第三 variant（true success 决策）另有 9 runs × 5 代 = 45 代。本文按这个口径叙述，不把 27 当成 18。这是来源内部的口径差异，不影响结论，但引用时要注意。

### 1.2 关键机制

- **环内 judge 有两个失效源，不是一个。** 作者把环内报告增益与 sealed 增益之差记为 judge gap，并拆成 `E`（judge error：judge 在自己数据上就错）与 `T`（transfer gap：在自己数据上真涨，但换分布不涨）。更准的 judge 只能压 `E`；任何只看 `D_in` 的规则都看不见 `T`（Proposition A.1）。
- **sealed judge 的关键不是“换一个更强的模型”，而是“换一个 improver 影响不到的评测分布”。** 论文的 sealed pool 在 improver 运行前固定，数据、参数、抽样 schedule 对 improver 隐藏；driving 的 850 个 improver steps、manipulation 的 114/114 代都做了文件访问检查（Appendix A）。
- **用序列检验而不是固定样本检验，是为了“随时可读、错误率仍然成立”。** 每读一对条件就更新 confidence sequence `C_t`；`L_t > 0` 接受，`U_t < 0` 拒绝，否则继续到预算 `N`，未决就 rollback。这样“保留更差策略”的概率不超过 `α`，且不依赖“读多少次”。
- **双跑设计是测量工具。** 同一个 loop 分别由 `J_in` 和 `J_out` 决策，两个 judge 都给每一代打分，才能同时得到 `Δ̂_g` 与 `Δ_g`，进而算出 `E_g` 和 `T_g`。只跑一次环内 judge，只能看到 `Δ̂_g`。
- **保守回滚是协议的一部分。** 未决不等于接受；论文在 manipulation 的 sealed 变体中 kept 12 代，平均 certified gain `+0.147`，但也出现 classifier 把 3 个 certified gains 回滚掉（Table 3）。这是“宁可少升、不可错升”的代价。

⚡ **Eureka Moment**：决定 keep/rollback 的 judge 必须位于 improver 无法影响的环外；环内证据再准也测不到 transfer gap，所以需要 sealed judge + 有错误率保证的序列检验，而不是再训练一个更强的环内 judge。

### 1.3 信息流/架构图

```
π_g ──(I, J_in)──> π_{g+1}
  │                    │
  │                    ├──> J_in on D_in ──> Δ̂_g（环内报告增益）
  │
  └──> SealedLoop（每代四步）
        1. Seal: 固定 P、s、抽样规则
        2. Improve: 只用 J_in 产生 π_{g+1}；J_out 不可见
        3. Decide: 在 paired conditions c_t 上跑 π_g 与 π_{g+1}
                   d_t = s_{c_t}(π_{g+1}) − s_{c_t}(π_g)
                   C_t = [L_t, U_t]
                   L_t > 0 → accept；U_t < 0 → reject；否则继续到 N
        4. Reveal: accept 则 keep，否则 rollback
                   同时记录 J_out on P ──> Δ_g
                   G_g = Δ̂_g − Δ_g = E_g + T_g
```

## 2. 数学核心

📌 **Napkin Formula**（一行抓住本质）：

```
G_g = Δ̂_g − Δ_g = (Δ̂_g − Δ^in_g) + (Δ^in_g − Δ_g) = E_g + T_g
```

**目标**：判断第 `g` 代候选 `π_{g+1}` 是否真的优于 `π_g`；把“环内报告增益”与“环外 sealed 增益”的差拆成可分别测量的两项。

**核心公式**（代码块，非 LaTeX）：

```
Δ̂_g = J_in(π_{g+1}; D_in) − J_in(π_g; D_in)          # 环内报告增益
Δ_g = S(π_{g+1}; P) − S(π_g; P)                      # sealed 增益
G_g = Δ̂_g − Δ_g                                      # judge gap
Δ^in_g = S(π_{g+1}; D_in) − S(π_g; D_in)             # 环内数据上的 true gain
E_g = Δ̂_g − Δ^in_g                                   # judge error
T_g = Δ^in_g − Δ_g                                   # transfer gap
G_g = E_g + T_g
```

**序列检验**：

```
d_i = s_{c_i}(π_{g+1}) − s_{c_i}(π_g) ∈ [−1, 1]
u_i = (d_i + 1) / 2 ∈ [0, 1]
m̃ = (m + 1) / 2

μ̂_{i−1} = (1/2 + Σ_{j<i} u_j) / i
σ̂²_{i−1} = (1/4 + Σ_{j<i} (u_j − μ̂_{j−1})²) / i
λ_i = sqrt( 2 log(2/α) / (σ̂²_{i−1} · i · log(1+i)) )

λ⁺_i(m) = min{ λ_i, c / m̃ }
λ⁻_i(m) = min{ λ_i, c / (1 − m̃) }        # c = 0.75

K_t(m) = ½ Π_{i=1..t} (1 + λ⁺_i(m)(u_i − m̃))
       + ½ Π_{i=1..t} (1 − λ⁻_i(m)(u_i − m̃))

C_t = { m : max_{r≤t} K_r(m) < 1/α }
L_t = inf C_t，U_t = sup C_t
```

**变量说明**：

| 符号 | 含义 |
|------|------|
| `π_g` | 第 g 代策略 |
| `I` | improver，用 `J_in` 产生候选 |
| `J_in` / `D_in` | 环内 judge / 环内评测集 |
| `J_out` / `P` | sealed judge / sealed pool |
| `s_c(π)` | 条件 c 上策略 π 的 true score，取值 [0,1] |
| `S(π;D)` | 在数据集 D 的条件上对 `s_c` 取均值 |
| `Δ̂_g` / `Δ_g` | 环内报告增益 / sealed 增益 |
| `Δ^in_g` | 环内数据上的 true gain（loop 自己不用） |
| `E_g` / `T_g` | judge error / transfer gap |
| `d_i` | 第 i 对条件的 paired difference |
| `C_t` / `α` / `N` / `n_0` | confidence sequence / 错误率 / 预算 / 首次读数（论文取 10） |

**直觉**：`E` 是“尺子不准”，`T` 是“在这把尺子的量程内量着涨、出了量程不涨”。换更贵的尺子只能修 `E`；要看见 `T`，必须换一个 improver 没见过的量程。Proposition 1 给出：在条件独立于两个策略的假设下，序列检验接受一个 `Δ_g ≤ 0` 的候选（或拒绝一个 `Δ_g ≥ 0` 的候选）的概率不超过 `α`；Corollary A.1 给出 `K` 代的期望错误决策数不超过 `αK`。Proposition A.1 则说明：任何只依赖 `D_in` 上行为的规则，都存在两个 sealed pool 打分函数，使规则行为完全相同、但一个 `Δ_g > 0`、一个 `Δ_g < 0`——所以环内规则无法分辨 `T`。

## 3. 带数字走一遍：玩具例子

> 以下数值为**假设**，用于验算分解式；不是论文实验数据。论文的实测数字在 §5 引用。

**假设**一个 driving-like 场景：

```
D_in 上 true gain        Δ^in = +0.10
环内 judge 报告           Δ̂   = +0.16
sealed pool 上 true gain  Δ    = −0.02
```

分解验算：

```
E = Δ̂ − Δ^in = 0.16 − 0.10 = +0.06
T = Δ^in − Δ  = 0.10 − (−0.02) = +0.12
G = Δ̂ − Δ    = 0.16 − (−0.02) = +0.18
E + T = 0.06 + 0.12 = 0.18  ✓
```

这个例子里，环内 judge 偏乐观 `+0.06`，同时“真涨”的部分在 sealed pool 上反转了 `−0.12`；两者叠加，看起来是 `+0.18` 的大涨。**即使把环内 judge 换成 exact judge**（`E = 0`），`Δ̂ = Δ^in = +0.10`，`G = +0.12 = T`，仍然会 keep 一个在 sealed pool 上更差的策略。这正对应论文的实测：在 driving control runs 上，即使把 loop 自己评测帧上的 true score 喂给检验，也会接受 8/15 代，但同一批候选在 unseen logs 上没有一个被接受（§5.2）。

**错误预算验算**：`α = 0.05`，`K = 5` 代，则 `E[W] ≤ αK = 0.25` 次错误决策；`Pr(W ≥ 1) ≤ 0.25`。论文在 manipulation 上用 189,000 个模拟代测量，无真实增益的池上误接受率最高 1.2%，5 代至少一次误接受 5.7%，低于 union bound 25%（Appendix A，Table 5）。这是“模拟校核”，不是形式证明；形式保证仍依赖 Lemma A.2 的独立抽样假设。

**成本验算**：driving 全池 750,825 帧；序列检验读 354,754 帧。`354754 / 750825 ≈ 0.4725`，即 47%，与论文“47%”一致（§5.4）。注意这是**评测帧预算**，不是策略的观测窗口或动作 chunk 长度。

## 4. 工程视角

| 工程问题 | 论文做法/数据 | 含义 |
|------|------|------|
| 额外 rollout 成本 | 每个 condition 要跑 `π_g` 与 `π_{g+1}` 两遍，paired 比较 | 评测算力约 2×；实车上意味着同一路线两版各跑一次，论文用 back-to-back + 随机顺序 + blind labels 控制顺序效应 |
| 评测预算 | manipulation：环内 300 pairs、sealed 最多 1,000 pairs；driving：环内 5,000 帧、sealed 16,685 帧 | 预算按 testbed 设定，不能把某一设置的步长/帧数写成所有实验共有 |
| 提前停止 | manipulation sealed 中位 191 pairs；32 代用满预算；driving 38 代在 5,316 帧后决定、7 代未决 | 序列检验的收益来自“容易的代早停”，难的代仍可能吃满预算 |
| 未决的代价 | driving 45 代中 7 代未决；manipulation 77 个未决测试需 133–48,024 额外 pairs，中位 4.7×（sealed）/7.4×（classifier）预算 | 未决即 rollback，增益稀疏时会拖慢 loop；论文的投影是“若趋势持续”的估算，不是实测继续读的结果 |
| 隔离成本 | 文件系统 mask；driving 850 steps、manipulation 114/114 代检查 | 这是工程隔离，不是密码学保证；论文没有声称能防侧信道 |
| 控制频率 | 实车每 0.80 s 规划一次，4 s 八点轨迹，5 diffusion steps，限速 4 m/s | sealed judge 不在控制回路内；实车评分是离线按 drive-by-wire 记录算的 |
| 统计口径 | driving 以 log 为单位读 paired difference；per-frame 单独检验会误接受 10/1,499，per-log 为 0/1,499 | 同一 log 内帧相关，不能把帧当独立样本；分层/整群抽样要在协议里写清楚 |

**时序口径（重要）**：本文的时序单位是 generation（代）、run（次运行）、episode（manipulation 的一次 rollout）、log/frame（driving 的评测单位）、pair（一对条件/一次配对驾驶）。论文**没有**把“每代 300 pairs / 5,000 frames / 16,685 frames”定义为策略的历史观测窗口，也没有定义动作 chunk 长度或记忆窗口。实车的 0.80 s 是规划周期，不是所有 testbed 共有的时间步。因此不能把 sealed pool 的帧数读成记忆机制，也不能把某一设置的预算写成所有实验共有的窗口。

**笔者的条件性分析**：如果部署分布与 sealed pool 差异很大，`Δ_g` 的“certified”只对 pool 分布成立；论文自己在 manipulation 上看到，suite start states 上最终策略平均 `+0.18`（sealed）/`+0.21`（classifier），换到 new start states 就缩到 `+0.06`/`+0.02`（Appendix B.5）。工程上应把 sealed pool 当作“部署分布的代理”，并定期用真实部署数据做外部校验，而不是把一次 certified 当成永久结论。

## 5. 数据与评测

### 5.1 预训练与起点（仿真）

- **LIBERO manipulation**：起点是 OpenVLA-OFT 的 LIBERO-Spatial checkpoint；论文用少量 demonstrations 做 LoRA 适配，三个任务的 start policy 在 1,000 个 sealed conditions 上分别为 `0.661 / 0.746 / 0.491`（§4、§B.1）。论文明确说 released OpenVLA-OFT checkpoint 已能解 LIBERO-Object，所以改用 LIBERO-Spatial checkpoint 以便“还能提升”。
- **NAVSIM driving**：起点 DiffusionDrive 从 scratch 在 20,000 帧上训练 20 epochs；主起点是在 3,731 个 Boston 帧上 PDMS 最高的六个配置之一（0.698），这 3,731 帧是 sealed pool 的 22%。为检查“用部分 sealed pool 选起点”的影响，论文在其余 12,954 帧上重跑每代的 sealed test，45 个候选中 43 个决策不变，2 个 rejection 变 undecided，没有候选被接受（Appendix C.1）。
- **数据边界**：Boston 帧完全不参与 loop 训练；driving 的 loop 只看 Las Vegas / Pittsburgh / Singapore。manipulation 的 sealed pool 由 50 个 start states × 200 seeds = 10,000 conditions 组成，每代抽互不相交、按 start state 分层的集合。

### 5.2 仿真评测

| 指标 | LIBERO manipulation | NAVSIM driving |
|------|------|------|
| 环内 vs sealed 的分歧 | 5/90 代报告 gain 而 sealed 发现 loss；median run 的 judge gap 首次超过 0.10 在第 3 代 | 每一代、每个 seed 的 transfer gap 都为正；第一代 judge gap `+0.095 [0.087, 0.104]`（own-label refresh）、`+0.045 [0.037, 0.052]`（true-label refresh） |
| 环内 judge 的错 | 一代中 196 个失败被 classifier 接受 65 个；FP rate 随代从 0.000–0.227（第一代）到 0.000–1.000（最后一代） | critic 在 Boston 的 FP rate 从 0.164 升到 0.383，其 pick 比策略自己的选择差 `−0.042`；control runs 在 loop 帧上 `+0.047~+0.058`，Boston `−0.107~−0.075` |
| sealed 决策 | kept 12 代，平均 certified gain `+0.147`；classifier kept 1 个 worse、rolled back 3 个 certified gains（Table 3） | sealed 拒绝全部 15 个第一代候选（平均 sealed gain `−0.028`）；main runs 45 个候选 0 accept、38 reject、7 undecided |
| 已知增益的检验 | — | 76 对已知 sealed gain 的策略对：12 个真增益、59 个损失、5 个近零；序列检验接受 11/12 真增益、0/59 损失、0/5 近零 |
| 专门构建的改进 loop | — | 32 个候选中 11 个是 first step，sealed gain `+0.029~+0.043`；sealed 接受全部 11 个，learned critic 只接受 1 个；held policy `0.674`（sealed）vs `0.649`（critic）vs `0.689`（loop 帧 true score 的 paired test），起点 `0.638` |
| 更大/off-the-shelf judge | Qwen3-VL-32B 在 196 个失败上只接受 3%，但 TPR 0.69；loop classifier 与 success predicate 同对决策一致 35/38 代，其他 judge 最多 27 | — |
| 评测成本 | — | 序列检验读 354,754 / 750,825 帧 = 47% |

**关键区分**：

- **judge error 的实测**：manipulation 的 classifier 会把“物体还在篮筐边缘”判成成功；论文展示了一代 196 个失败中 65 个被接受，且这些失败中物体从未到达目标区域（§B.3、Fig. 11）。另外 3 代出现“refresh 时没有失败，下一代的失败被判成功”的失明模式（§B.3）。
- **transfer gap 的实测**：driving 的候选在 loop 自己的评测帧上涨，在 training cities 的 unseen logs 上已经跌 `−0.064~−0.038`，占从 loop 帧到 Boston 跌幅的 70%；所以不是“换新城市”才发生（§5.2、Appendix C.3）。即使把 target 换成每帧 true score 最高的 proposal（平均 0.965 vs 策略自己选择的 0.887），第一代候选在三个学习率下 Boston 仍为 `−0.018 / −0.008 / −0.108`（Appendix C.3）。
- **作者主张**：环内证据不足以批准更新；sealed judge 应自己作决定，而不是只提供一个“独立测试集”。论文的对照是：in-loop variants 明明能看到 sealed reading，仍然 keep 了 worse policy（Conclusion）。
- **笔者的条件性分析**：论文的 `αK` 保证假设每代条件独立于策略；实际是 stratified without replacement，作者用模拟（189,000 代）校核误接受率 1.2%，这比形式证明弱，但在本文设置下可接受。换到其他 benchmark 时，应重新做这类模拟，而不是直接套 `α=0.05`。

### 5.3 真实车辆设置

- **平台**：full-scale drive-by-wire 电动 van，RTK-corrected INS，on-board NVIDIA RTX A6000；两版共享 VLM backbone，只差 diffusion planner 权重；每 0.80 s 从同一前摄像头规划一次 4 s 八点轨迹，5 diffusion steps，限速 4 m/s（§4、Appendix D.2）。
- **评分**：`min(D, L) / L`，`D` 是第一次 intervention（brake / steering / throttle override、timeout、stall）时沿路线的距离，`L` 是路线长度；干预由 safety driver 执行，driver 与 operator 只看 blind labels。
- **仿真侧的两版**：ReCogDrive 的 IL 与 RL 在 NAVSIM navtest 的 12,118 个场景上分别为 `0.861` 与 `0.904`，paired gain `+0.043`（95% interval `+0.039` 到 `+0.048`）；PDMS 分项显示 RL 主要赢在 drivable-area compliance（0.946→0.976）与 ego progress（0.804→0.858）；driving-direction compliance 从 0.981 降到 0.967，但 Table 10 标明该项不属于 PDMS。
- **实车结果**：主路线 0/12 完成；safety driver 在弯道前半段 30–57 m 介入；RL 在 6/6 对中走得更远，平均 `+10.4 m`（`+0.072` route fraction），exact sign test one-sided `p=0.016`；序列检验 6 对后仍未决，confidence sequence `[−0.50, +0.57]`（§5.5、Appendix D.3）。直线路线 6/6 完成；右弯 1 对，IL 83.1 m、RL 98.2 m。
- **VL judge 看驾驶视频**：Qwen2.5-VL-3B / 7B、Qwen3-VL-8B 对 20 次 scored drives 的 intervention 预测 AUROC 为 `0.17 / 0.07 / 0.11`，低于 0.5，说明它们把“完成的直线”评得比“弯道”更可能需要干预（Appendix D.3、Fig. 19）。这支持作者“off-the-shelf judge 不能替代 sealed judge”的主张。

## 6. 能力与失败模式

**能做什么**

- 在仿真中把“环内报告涨、sealed 不涨”的候选识别出来，并分别归因到 `E` 或 `T`。
- 在已知增益的 76 对 driving 策略对上，序列检验接受 11/12 真增益、0/59 损失，读 53% 的帧；47% 是九个 main runs 上 354,754/750,825 帧的比例。
- 在一个专门构建的 driving loop 中，把 learned critic 只接受 1/11 的 first-step gains 全部认证，并把 held policy 从 0.638 提到 0.674（critic 为 0.649）。
- 在实车上验证一个已发表改进步的**方向**：RL 在 6/6 盲配对中走得更远。

**不能做什么**

- 不能保证 sealed pool 代表部署分布；manipulation 的 certified gain 换到 new start states 会缩水（`+0.18/+0.21` → `+0.06/+0.02`）。
- 不能替代真实部署：实车两版都未完成主路线，干预发生在弯道前半段；论文自己把原因归为两版共享的 deployment gap，而不是 RL 步的问题。
- 不能处理非平稳 pool：sealed pool 一旦固定，部署分布漂移后需要重新 sealed；论文没有给出在线更新 pool 的协议。
- 不能在预算内决定“近零增益”：manipulation 77 个未决测试需要 133–48,024 额外 pairs；driving 7 代未决。未决即 rollback，会损失真增益（classifier 3 个 certified gains 被回滚、VL judge 2 个）。
- 不能把 off-the-shelf VLM 当 sealed judge：Qwen3-VL-32B FPR 低但 TPR 0.69；Qwen2.5-VL-7B 作 sealed 替代时 AUC 0.85、TPR 0.82 / FPR 0.24，sensitivity+specificity−1 仅 0.585，任务间 0.47/0.36/0.94（Appendix F.2）。
- 实车只测一个改进步、一辆车、无障碍路线；长任务、on-policy RL、硬件上多代 loop 都留给未来（Appendix G）。

### 6.1 隐含假设

1. **sealed pool 的分布足够接近部署分布**。论文用 Boston 作 unseen city、用 new start states 做外推检查，但没有证明 pool 覆盖部署的长尾。
2. **条件抽样独立于两个策略**。Lemma A.2 要求 i.i.d.；实际是 stratified without replacement，作者用 189,000 个模拟代校核误接受率，而不是形式证明。
3. **`s_c` 是可信的 true score**。manipulation 用 simulator predicate，driving 用 PDMS，实车用 route completion；这些都是 proxy，不是绝对真值。论文没有声称 `s_c` 无误差，只声称 sealed judge 的**数据与 schedule** 对 improver 隐藏。
4. **improver 无法通过侧信道影响 sealed pool**。论文用文件系统 mask 和访问检查（850/850、114/114），但这不是密码学隔离。
5. **每代只比较相邻两个策略**。`αK` 控制的是“每代错误决策”的期望数，不是“多代累积漂移”的界；loop 的长期行为仍需外部监控。
6. **paired evaluation 可行**。同一 condition 上两个策略都要能跑；实车上 back-to-back 驾驶存在顺序、路况、驾驶员疲劳等混杂，论文用随机顺序 + blind labels + 6 pairs 缓解，但样本很小。
7. **未决回滚是保守但可接受的策略**。在增益稀疏或预算不足时，这会拖慢 loop；论文没有给出“加预算 vs 回滚”的最优停止规则。

## 7. 与相关工作对比

| 路线 | 代表工作（论文引用） | 判定者 | 是否环外 | 逐代曲线 | 量化 exploitation |
|------|------|------|------|------|------|
| 环内 classifier / reward model | HIL-SERL、SARM2、Robo-Dopamine | classifier / reward model / human | 否 | 多为 n.s. 或 partial | 多为描述或仅 judge error |
| simulator score | SimpleVLA-RL、ReCogDrive、GIGAFLOW | simulator rules / PDMS | 否 | 多为单次或 n.s. | 少 |
| learned critic | EvaDrive、AD-R1 | learned critic / world-model critic | 否 | EvaDrive partial（≥4 iterations，best at 2–3） | 少 |
| 软件 agent 的 sealed evaluation | hidden-rollout harness、test windows、error-controlled verifiers、pre-registered analyses | 环外 harness / test suite | 是 | 非具身 | 不适用 |
| 本文 SealedLoop | — | sealed judge + sequential test | 是 | 是（5–8 代） | 是（quantified） |

论文 Table 1 的调查口径：56 个具身自提升 loop 中，2 个报告完整 per-generation curve、9 个 partial、11 个量化 judge error、0 个同时报告 curve 与量化的 exploitation。这是作者对文献的调查结果，不是本文实验；引用时注意区分。

**与最接近的基线相比**：

- 与“换更强 judge”相比：本文的对照是 Qwen3-VL-32B 等 off-the-shelf judge，结论是更大模型只在 FPR 上更好、TPR 更差，不能替代 sealed judge（§5.3、Fig. 5）。
- 与“独立测试集”相比：本文的 in-loop variants 明明能看到 sealed reading，仍然 keep 了 worse policy；所以“有独立测试集”不等于“由独立测试集决定”（Conclusion）。
- 与“固定样本检验”相比：在 76 对已知增益上，50 logs 的 bootstrap 接受 10.8/12 真增益、读 22% 帧；序列检验接受 11.2/12、读 53% 帧。固定预算下 bootstrap 更省，但序列检验的错误率在“边读边停”时仍成立（Appendix C.4、Table 8）。

**面试 Tip**：如果被问“self-improvement 怎么防止 reward hacking”，一句话答：把决定 keep/rollback 的 judge 放到 improver 看不见的环外，用 anytime-valid 序列检验控制错误率；并且要意识到，即使 judge 在环内数据上完美，transfer gap 仍然存在，所以 sealed pool 必须与训练分布不同。

## 8. 精读建议

- **值得精读原文的人**：
  1. 做 VLA / 驾驶策略在线自提升、RL fine-tuning 的研究者：§3 的协议、§5.2 的分解、§C.4 的序列检验实操。
  2. 做闭环评测与模型验收的平台工程师：§4 的预算/未决率、§5.4 的 acceptance rules、Appendix A 的误接受率模拟。
  3. 要在真实机器人上做“改进步是否可信”实验的人：§5.5 与 Appendix D 的盲配对、评分、顺序控制。
- **建议章节路径**：先读 §3（方法与公式）→ 再看 §5.2（E/T 分解的实测）→ 然后 §5.4 与 §C.4（序列检验的召回/成本）→ 最后 §5.5 / §D（实车）。§2 的相关工作可以在读完 §3 后再看，避免被 56 个 loop 的调查表淹没。
- **不值得精读的理由**：如果你只关心 VLA 架构、动作 chunk、记忆机制或某个 benchmark 的 SOTA，这篇不提供这些；它是一篇评测协议论文，价值在“怎么判定改进”，不在“怎么设计策略”。如果你已经熟悉 anytime-valid inference 与 robot policy comparison，读摘要 + §3 + §5.2 即可。

---

[← Back to Theory](./README.md)

**关键引用**：论文全文与所有数字见 https://arxiv.org/abs/2611.33794（arXiv:2611.33794v1，HTML 版，2026-11-25）。