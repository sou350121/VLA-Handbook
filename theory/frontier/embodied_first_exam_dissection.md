# Embodied First Exam：把 coding agent 放进物理约束下的机器人考场 (Embodied First Exam)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-11-14
>
> **论文**: Embodied First Exam
> **链接**: https://arxiv.org/abs/2611.12177
> **依据版本**: arXiv:2611.12177v1 的 HTML 正文（2026-11-09）。来源记录未提供摘要核验差异；本文不以候选摘要替代正文。摘要中的 34% 对应 456 个 Standard tests，§3.2 的 30% 对应 391 个同时有 Privileged/Standard 的 tasks，分母不同。
> **核心定位**: 它不是新 VLA 模型，而是一个 agent-native 评测框架：让 coding agent 自选 direct control / code to control / train a policy，并在 Privileged、Standard、Realtime 三种物理约束下测同一批机器人任务。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 把“物理约束”拆成状态访问、reset、实时三个可消融的接口变量；最强 agent 在 Privileged 上 77%，同一批 391 个 task 的 Standard 只有 30% |
| 适合精读 | 做 agentic robot 评测、harness 设计、coding agent + robot、sim-to-real 评测的人，重点看 §2.1、§3.2、§3.4、§3.5、Appendix C/H |
| 可以跳过 | 只关心 VLA policy 架构或预训练的人；这篇不训练基座，评的是现成 model + harness |
| 落地可行性 | 中：框架和任务定义可复用，但全仿真、每 test 单次 attempt、真实机器人只有 1 个 demo |
| 主要风险 | 结论依赖 simulator 保真度与 agent 标注的 stage labels；core set 筛选可能对 GPT-6 家族有偏 |

💡 **X-Ray 开场**

这篇论文问的是：通用 coding agent 能不能在物理约束下控制多种机器人？它发现：给 agent 完整的 simulator 状态和无限 reset，最强 agent 能解 77% 的 391 个 hard core tasks；但只给传感器观测、默认不许 reset、仿真在请求间暂停时，同一个 agent 只解 30%。对 VLA 研究者这意味着：评测的单位可能从“policy 在固定 observation-action 接口上的成功率”，变成“agent 在接口权限、reset 预算、时间压力下的任务完成率”。

📍 **研究全景时间线**

```
2022  Code as Policies / SayCan：LLM 写机器人程序、调技能
  → 2023  SWE-bench：coding agent 在真实 repo 上被系统评测
  → 2024  BEHAVIOR-1K / OpenVLA / HumanoidBench：机器人 benchmark 与 VLA 规模化
  → 2025  Gemini Robotics / GR00T N1：通用机器人策略
  → 2026  CaP-X / RobotWorld / EmbodiedSWE / BiGym 2.0：agentic robot 评测出现
  → 2026-11  EFE：47 个机器人、三种物理约束 setting、三种控制方式 ← 本文
局限：全仿真为主；每 test 一次 attempt；真实机器人只有 Appendix I 的 1 个 buzz-wire demo
```

## 1. 核心架构/方法总览

### 1.1 系统对比概览

EFE 把每个 task 打包成 Harbor 格式：agent 有独立 workspace，可装包、写程序、训练；task 从冻结初始状态开始；verifier 在 agent 控制之外的容器里判分。

| 维度 | Privileged | Standard | Realtime |
|------|------------|----------|----------|
| 观测 | 同容器可读完整 simulator state，包括 object pose | 只通过 API 拿本体状态（joint pos/vel/torque）与 RGB-D；不给 object pose、segmentation、scene file、seed、success rule；contact force 只以 safety telemetry 形式给 | 同 Standard |
| reset | 无限 | 默认禁止；9 个 suite 给 3–50 次 budget | 同 Standard |
| 仿真时间 | 请求之间暂停 | 请求之间暂停 | 不暂停；按 slowdown factor S≥1 运行，S=1 当 simulator 至少 3× real time |
| 答案形式 | low-level trajectory，或某些 task 的 program | 在线 action chunk（T 行 × d 维） | 同 Standard |
| 判分 | trajectory 在 fresh process 中 replay（默认 1 次；BEHAVIOR-1K、AM-Bench、RoboDojo、RoboLab 最多 3 次）；program 类答案在 fresh simulator 中运行 | live episode；通常 verifier 读 simulator service 的记录，但 RoboEngineer 与 Embodied Olympics - Match 的 Standard test 使用自己的 runtime | 同 Standard；但 Embodied Olympics - Match 的 Realtime test 也使用自己的 runtime，不经过 simulator service |
| 适用 | 近似上界、生成训练数据 | 主评测，模拟真实机器人部署 | 仅动态任务；54 个 Realtime tests |

关键点：Privileged 的判分不能概括成“一律 replay”。trajectory 类答案才是 replay；Embodied Olympics - Match、EmbodiedSWE 等 program 类答案是在 fresh simulator 中运行程序。Standard/Realtime 则是在线判 live episode，不 replay。

agent 是 model + harness 的组合，结果按组合报告，不能把某个 harness 的数字归给模型本身：

| agent | model | harness | 明确报告的结果 |
|-------|-------|---------|----------------|
| GPT-6 Astra | GPT-6 Astra | Codex | 456 Standard：33.6%（图 7；摘要/§3.4 四舍五入 34%）；391 同时有 Privileged/Standard：77% vs 30% |
| Claude Opus 5.5 | Claude Opus 5.5 | Claude Code | 456 Standard：25.0%（Appendix I）；Realtime 4/54（§3.1） |
| Gemini 4 Argon | Gemini 4 Argon | Antigravity CLI | 图 7 给逐 agent 条形；Realtime 5/54（§3.1） |
| Kimi 3.1 Pro | Kimi 3.1 Pro | Claude Code | 391 tasks 的 Standard：9%（34 tests），是三个最强之外最好的 |
| 其余四个弱 agent | Qwen3.8 Max、Gemini 3.8 Flash、GLM 5.3 Flash、DeepSeek V4.1 Flash | 各自 harness | 论文图 7 给逐 agent 数字；本文不逐一复述未核对值 |

### 1.2 关键机制

- **统一容器 + 独立 verifier**：agent 可以在 workspace 里写代码、装包、训练，但不能改 success rule；verifier 在自己的容器里、无网络、只接收 task 声明的文件。
- **三种控制方式留给 agent 自选**：Direct control（每次发一个 action chunk，看图像再决定下一个）、Code to control（写 open-loop / closed-loop 程序或参数搜索）、Train a policy（在仿真里训练策略再控制）。论文明确说，不同 setting 和资源约束下 agent 会换方式。
- **物理约束 = 接口权限消融**：Privileged 给全状态 + 无限 reset；Standard 只给传感器 + 默认无 reset；Realtime 再加“世界不等人”。三者共享初始场景、success rule、时间限制，所以可以配对比较。
- **资源约束**：2–16 CPU cores、4–39 GiB memory、30 min–4 h wall clock、token budget 从未被触及；9 个 suite 给 reset budget；部分 task 给 GPU。GPU 会改变 agent 的控制方式选择，例如 soccer 的 GPU 版本有 17/20 次训练了 policy。
- **反 reward hacking**：每个 task 必须“机器人不动时失败”；有 reference solution 的 task 必须通过；加 suite 前让 agent 尝试“不解题而通过”，若能通过就修 verifier；RoboDojo 的 Standard test 中 success 不结束 episode，避免 agent 把“episode 结束”当成成功信号。

⚡ **Eureka Moment**：这篇论文最核心的想法不是“再做一个机器人 benchmark”，而是把“物理世界难”拆成三个可开关的接口变量——全状态、reset、实时——从而把“agent 在真实机器人上能不能行”变成可消融实验。

### 1.3 信息流/架构图

```
Privileged:
[agent container: model + harness + simulator library]
        │  提交 trajectory 或 program
        ▼
[verifier container: replay trajectory / 在 fresh simulator 跑 program]
        │
        ▼
     pass / fail

Standard / Realtime:
[agent container: model + harness + client]
        │  HTTP API：关节状态、RGB-D、action chunk
        ▼
[simulator service container] ──live episode record──▶ [verifier container]
        ▲                                              │
        └──────── images / state / time left ──────────┘
                                                       ▼
                                                    pass / fail
```

Standard/Realtime 中 simulator 与 verifier 都和 agent 分离；agent 读不到 success rule，也不能直接改 simulator state。Realtime 只是“不暂停”，不是另一个模型架构。

## 2. 数学核心

📌 **Napkin Formula**：

```
SR(setting) = solved_tests / tests
fail_rate(type) = failed_stages(type) / reached_stages(type)
D = Σ_k n_k(n_k-1) / (n(n-1))
E[D | random] = Σ_k q_k²
```

| 符号 | 含义 | 注意 |
|------|------|------|
| `SR` | 成功率 | 分母是 test 数；每个 agent 每个 test 只做 1 次 attempt，不是多次重复 |
| `fail_rate(type)` | 某类 stage 的失败率 | 分母是“被 attempt 到达的该类 stage 数”，不是所有 stage |
| `D` | failure stage 的集中度（Simpson index） | `n` 个 failed attempts，`n_k` 个失败在同一 stage `k`；衡量不同 agent 是否在同一处翻车 |
| `q_k` | 随机基线中 stage `k` 的概率 | 由其他 task 的失败位置直方图给出；不是均匀分布 |
| `T × dt` | 一个 action chunk 的开环执行时长 | `dt` 是 control period 的倍数且 ≤0.5 s；T 因 suite 而异，论文没有统一值 |

直觉：`D` 高说明失败不是随机散落，而是集中在少数 stage；`Σq_k²` 是“如果 agent 只按任务位置随机失败”的基线。EFE 的 Standard 集中度 61%，随机基线 37%，说明不同 agent 经常在同一 stage 失败。

Privileged 到 Standard 的 gap 用 ablation 拆：

```
SR_priv ≈ 77%
SR_std(baseline) ≈ 28%
SR_std + object-state query ≈ 45%
SR_std + unlimited resets ≈ 53%
SR_std + both ≈ 68%
```

这是 391 个同时有 Privileged/Standard 的 core tasks、GPT-6 Astra in Codex 的受控实验。作者的解释是：总体上 reset 比 exact state 更能补 gap；在 coarse stages 上，state 单独相对 no-success baseline 有用（34% vs 14%），但 reset 仍更高（44%），因此不能说 state 在 coarse 上比 reset 更有用。这是作者的条件性分析，不是因果分解，因为两个 feature 可能互相促进。

**时间换算（条件性，不是论文公式）**：

```
chunk_duration = T × dt
K-frame span = (K-1) × Δt
H-step endpoint = H × Δt
```

`T` 是 action chunk 行数，`dt` 是相邻行的时间间隔；`K` 是均匀采样帧数；`H` 是从当前参考 `t` 预测 `t+1` 到 `t+H` 的步数。EFE 没有 future frame prediction，所以后两行只是接口换算，不是 EFE 的实验设置。论文只给 `dt ≤ 0.5 s`、control rate 5–500 Hz、physics step 1 ms–1/60 s；没有给统一 T。

## 3. 带数字走一遍：玩具例子

### 3.1 分母对齐（真实数字，非假设）

- 456 Standard + 391 Privileged + 54 Realtime = 901 core tests。验算：456 + 391 + 54 = 901，与 Table 1 的 901 core tests 一致。
- 456 个 core tasks 每个都有 Standard test；其中 391 个有 Privileged test；54 个有 Realtime test。因此“456 上 34%”和“391 上 30%”不是矛盾，是分母不同。
- 用 391 作分母验算 ablation：391×0.28 = 109.48；×0.45 = 175.95；×0.53 = 207.23；×0.68 = 265.88；×0.77 = 301.07。论文说 reset run 有 206 个 success、Privileged 有 303 个 solve；差异来自百分比四舍五入和取整。不能把 28%/45%/53%/68%/77% 当成精确计数。

### 3.2 precise/coarse failures 的验算（假设分母）

论文 Table 14 明确给 GPT-6 Astra 的 labeled failures：precise 91 个、coarse 97 个。假设用这两个数作分母，则百分比对应的次数约为：

| run | precise 91 | coarse 97 |
|-----|------------|-----------|
| no success signal | 19% ≈ 17 | 14% ≈ 14 |
| + object-state query | 21% ≈ 19 | 34% ≈ 33 |
| + unlimited resets | 36% ≈ 33 | 44% ≈ 43 |
| + both | 47% ≈ 43 | 67% ≈ 65 |
| Privileged | 73% ≈ 66 | 60% ≈ 58 |

验算：91×0.19 = 17.29；91×0.21 = 19.11；91×0.36 = 32.76；91×0.47 = 42.77；91×0.73 = 66.43；97×0.14 = 13.58；97×0.34 = 32.98；97×0.44 = 42.68；97×0.67 = 64.99；97×0.60 = 58.2。论文只给百分比，次数是本文按分母推算，不是论文表格原值。

这个玩具例子说明：在 precise failures 上，no-success baseline 已解约 17 个（19%）；state query 只比它多约 2 个（21% vs 19%），reset 比它多约 16 个（36% vs 19%）。在 coarse failures 上，state 单独解约 33 个（34%），而 baseline 约 14 个（14%）。与作者“reset 比 state 更能补 precise gap；在 coarse 上 state 单独相对 no-success baseline 有用，但 reset 仍更高”的结论方向一致。

### 3.3 action chunk 时间换算（假设）

假设某 suite 的 control period = 20 ms（50 Hz），`dt = 0.5 s`（25 个 control period，满足“是 control period 的倍数且 ≤0.5 s”），`T = 8` 行：

- chunk 执行时长 = 8 × 0.5 = 4.0 s。
- 若均匀采样 `K = 5` 帧、`Δt = 0.5 s`，首末跨度 = (5−1) × 0.5 = 2.0 s。
- 若从当前 `t` 预测 `t+1` 到 `t+H`，`H = 5`，终点距当前 = 5 × 0.5 = 2.5 s。

验算：8×0.5 = 4.0；4×0.5 = 2.0；5×0.5 = 2.5。标记：T、K、H 都是假设，不是 EFE 的统一设置。EFE 的 control rate 跨 suite 从 5 Hz 到 500 Hz，physics step 从 1 ms 到 1/60 s；论文没有给统一 T。4.0 s 只是这个假设下的 chunk 执行时长，不能当作 memory window，也不能当作历史上下文长度。

## 4. 工程视角

- **延迟与实时性**：Standard 在 API 请求之间暂停仿真，所以 agent 的推理延迟不计入仿真时间；Realtime 不暂停，按 `S≥1` 的 slowdown factor 运行。`S=1` 只当 simulator 至少 3× real time，否则 S 更大。Realtime 目前只评动态任务，54 个 tests，不能把它当成所有任务都有实时压力。
- **控制频率与步长**：control rate 5–500 Hz，physics step 1 ms–1/60 s；`dt` 是 control period 的倍数且 ≤0.5 s；一个 action chunk 是 T 行 × d 维。不同 suite 的步长不同，不能把某一 suite 的 `dt` 写成所有实验共有的记忆窗口。一个 API call 最多 15 min，返回图像数量也有限制，防止单次 call 占满服务。
- **判分频率**：success rule 在每个 action chunk 和每个推进 physics 的 observation 之后检查；Embodied Olympics - Fun 和 RoboDynamicExam 每 1 ms physics step 检查；DexHandExam 从 contact pairs 和 contact forces 计算 checkpoint。这是判分频率，不是 agent 决策频率；agent 决策频率取决于 control rate 和 chunk 长度。
- **隔离与完整性**：verifier 独立容器、无网络；agent 不能读 reference solution 或 success rule；Privileged 的 trajectory 类答案只 replay 动作，program 类答案（如 Embodied Olympics - Match、EmbodiedSWE）由 verifier 在 fresh simulator 中运行；直接改 simulator state 对 replay 无效。Standard/Realtime 通常由 simulator service 记录 live episode，但 RoboEngineer 与 Embodied Olympics - Match 的 Standard（以及 Match 的 Realtime）使用自己的 runtime，不经过 simulator service。RoboDojo 的 Standard test 中 success 不结束 episode，防止 agent 把“episode 结束”当成功信号反复试探。
- **资源与成本**：2–16 CPU cores、4–39 GiB memory、30 min–4 h；9 个 suite 给 3–50 次 reset；token budget 从未被触及。GPT-6 Astra 的 Standard attempt 平均 $27.1（按 2026 年 11 月 list price 计，不是账单），约为 Claude/Gemini 的 4 倍；在 Privileged 与 Standard Setting 中，对每个 agent，失败 attempt 的中位成本高于成功 attempt。
- **记忆机制**：论文没有 memory module，也没有以 memory window 为变量的实验。Standard 中 agent 可以把图像存成二进制文件、不经过模型 context；Privileged 中 save/load state 是跨 episode 的状态复用；Realtime 中程序要自己维护控制闭环。action chunk 是未来要执行的动作，不是历史上下文，也不是记忆窗口。
- **已观测历史帧 vs 未来预测帧**：EFE 给 agent 的是当前/历史传感器帧（Standard/Realtime）或完整状态（Privileged）；action chunk 是未来动作序列。论文没有 future frame prediction，也没有 world model 的预测帧。因此不能把 direct control 的 T 行 chunk 误读成“未来帧预测”，也不能把它的执行时长当成 agent 的记忆窗口。

## 5. 数据与评测

### 5.1 预训练

EFE 不训练基座模型。它评的是现成 coding agent（model + harness），模型本身是预训练好的通用模型。Soccer 里的 RL 训练是 agent 在 test-time 选择 “train a policy”，不是 EFE 的预训练。Privileged 中 GPU 版 soccer 有 5/8 次跑了 RL 代码，但所有 5 个胜利都复用了对手的 walking network；不能算“训练后获胜”。

### 5.2 仿真评测

- **规模**：1,156 tasks / 35 suites / 47 robots / 9 类机器人 / 2,288 tests；9 个新 suite 411 tasks；26 个改编 suite 745 tasks；618/745 个改编 task 用单臂、双臂或移动 manipulator。
- **Core set**：456 tasks / 901 tests；427 个 task 有 stage graph（2,411 stages，平均 5.6/task）；core set 排除“作者判断为 easy”和“GPT-6 Luna 解出”的 task；作者承认这可能对 GPT-6 家族有偏。392 个 task 没有 reference solution。
- **任务类型**：relation stages 最多；新 suite 覆盖 spatial puzzle、dexterous manipulation、multi-robot interaction、dynamic whole-body control、sim-to-sim engineering、safety。
- **主结果**：456 Standard 上 GPT-6 Astra 33.6%（图 7；摘要/§3.4 四舍五入 34%）；391 个同时有 Privileged/Standard 的 task 上 77% vs 30%；三个最强 agent 的并集解 48% Standard tests（不是成功率相加）；Realtime 54 tests 中 GPT 17、Claude 4、Gemini 5，63% 未被任何 agent 解。
- **Ablation**：baseline 28% → +state 45% → +reset 53% → +both 68% → Privileged 77%（391 tasks，GPT-6 Astra in Codex）。
- **失败率**：precise stages 从 22% 升到 56%，coarse 从 22% 升到 38%（within-suite pooled）；precise 升 24–46 个百分点，coarse 升 1–20 个百分点；material 66%、body 45%、mechanism 20%（两 setting pooled，Table 18）。
- **集中度**：Standard 61% vs 随机 37%；三个最强 56% vs 31%；跨 setting 同一最常见 failure stage 在 83 个 task 上 57% 匹配 vs 9% 随机（去掉两边都是第一 stage 的 task）。
- **弱 agent**：在到达的 contact stages 上，无接触失败 9.1–18.5%，三个最强 2.7–3.7%；zero progress 44–64% vs 27–36%。
- **Harness**：GPT 34% Codex vs 22% Claude Code；Gemini 29% Codex / 25% Claude Code / 24% Antigravity CLI；Claude 26% Codex / 25% Claude Code；最大 12 个百分点。跨 provider 经 adapter，只近似 native setup；Antigravity CLI 在 Gemini 内容过滤时直接结束 session，可能影响 Gemini 的数字。
- **控制方式**：1,489 个 sampled attempts 中，Standard 61% direct control；Privileged 82% 写程序；Realtime 97% 写程序，82% 的程序自己跑 control loop；6 次 train policy。Soccer：GPU 的 20 次（12 Standard + 8 Realtime）中 17 次训练、14 次用训练 policy 打、9 次进球、4 次赢；CPU 的 20 次 0 训练、3 进球、0 赢。Privileged 的 8+8 次中 GPU 2 胜、CPU 3 胜，但所有 5 胜都复用对手 walking network，不能与 Standard/Realtime 的“trained policy wins”合并。

### 5.3 真实机器人设置

Appendix I 的 buzz wire 是唯一真实机器人实验：1 个任务、1 个机械臂、2 个 agent，各 8 trials。GPT-6 Astra 前四平均 65.0%、后四 93.8%；Claude Opus 5.5 前四 35.0%、后四 62.5%；GPT 在 5/8 trials 达到 ≥80%，Claude 2/8；GPT 在 trial 7 和 8 走完全程，Claude 没有。作者明确说：这不是 exam 的一部分，结果不计入任何 success rate；一个任务、一个机器人、每个 agent 一条 8-trial 序列，不能分离学习效应和 run variance，也不能估计 sim-to-real 相关性。

## 6. 能力与失败模式

| 能力/失败模式 | 论文实测 | 条件 |
|---------------|----------|------|
| 全状态 + 无限 reset 下解任务 | 77%（303/391 左右） | Privileged，GPT-6 Astra in Codex；只是集合上的近似上界，单 task 可反转 |
| 只给传感器、默认无 reset | 30%（391 tasks）/33.6%（456 tasks） | Standard；GPT-6 Astra in Codex |
| 实时动态任务 | 63% Realtime tests 未被任何 agent 解 | 54 tests；比同 54 个 task 的 Standard 50% 未解更难 |
| 精确 stage | 失败率 22%→56% | precise 定义：tight fit、target pose/path、moving object contact；与 suite/robot 混淆，不能因果解释 |
| 材料/身体 stage | 66% / 45% 失败率 | 两 setting pooled；material 多来自 RoboPaint/Safety Exam，body 多来自 legged/humanoid |
| 弱 agent 接触前失败 | 9.1–18.5% vs 2.7–3.7% | 5 个弱 agent vs 3 个强 agent；按到达的 contact stage 计 |
| 换 harness | 最多 12 个百分点 | GPT-6 Astra Codex 34% vs Claude Code 22%；跨 provider 经 adapter，只近似 native |
| 给 GPU + RL library 时训练 policy | soccer 20 次中 17 次训练，4 胜全部来自 trained policy | 只这一个 task；两版本还差 GPU、library、training copies、instruction，不能归因到单一因素 |

### 6.1 隐含假设

1. **模拟器保真度足够代表物理约束**：全部主结果在仿真里；真实机器人只有 1 个 buzz-wire demo。作者承认 sim-to-real 相关性仍开放。
2. **单次 attempt 代表能力**：每个 agent 每个 test 只做 1 次 attempt；作者承认无法分离 agent 能力与 run variance，future work 会重复多次并给 confidence interval。
3. **stage labels 可信**：427 个 task 的 stage graph 和失败标签由 agent 写，GPT-6.1 Sol 与 GPT-6 Astra 同族，可能对比较有偏；47% 的 root cause 只到 “Control” 类别；perception 只在 4% 的 Standard failures 中被命名。
4. **core set 筛选不偏**：core set 排除 GPT-6 Luna 解出的 task；Luna 与 Astra 同族，作者承认这可能让 core set 对 GPT-6 家族更不友好。
5. **Privileged 是上界**：作者明确说只是“一组 task 上的近似上界”，不是每个 task 的上界；GPT-6 Astra 有 4 个 task 在 Standard 解出但 Privileged 失败，Gemini 4 Argon 有 18 个。
6. **harness 可比**：跨 provider 要经 adapter，只近似 native setup；Antigravity CLI 的内容过滤会结束 session。论文把 harness 当作测量对象，而不是“无关变量”。
7. **action chunk 不等于 memory**：论文没有 memory window 实验；不能把 T×dt 写成历史上下文长度，也不能把未来 action chunk 当成未来帧预测。
8. **成功率跨 suite 可比**：时间限制 30 min–4 h 因 suite 而异；Embodied Olympics - Fun 开放网络；部分 task 无 reference solution。作者在 limitations 中承认这些差异。

## 7. 与相关工作对比

| 工作 | 关注点 | agent 自由度 | setting/规模 | 与 EFE 的差异 |
|------|--------|--------------|--------------|----------------|
| CaP-X | coding agent 组合 perception/control primitives 的 abstraction level | 在给定 abstraction 下写程序 | 机器人 manipulation benchmark | EFE 让 agent 自选 direct/code/train，并把状态访问、reset、实时做成三 setting |
| RobotWorld | agent 在 manipulation、locomotion、driving、aerial 上的任务 | 多 embodiment | 84 tasks（论文引用） | EFE 1,156 tasks / 35 suites / 47 robots，并加 Privileged vs Standard 的权限消融 |
| EmbodiedSWE | coding agent 做长程 dexterous 任务 | 写代码控制 | 长程 dexterous | EFE 覆盖 9 类机器人，并把 sim-to-sim engineering、safety 纳入 suite |
| BiGym 2.0 | humanoid household，learned policy vs agent-developed policy | agent 开发程序 | humanoid household | EFE 跨 47 robots，且区分 full-state 与 sensor-only |
| LIBERO / BEHAVIOR-1K / RoboCasa 等 | 固定 observation-action 接口，评 learned policy | 低，policy 接收固定观测 | 桌面/家居/人形 | EFE 评 coding agent 的 whole loop：读接口、选控制方式、执行、修正；agent 决定动作怎么产生 |
| **EFE** | coding agent 在物理约束下的能力边界 | 高：三种控制方式自选 | 1,156 tasks / 35 suites / 47 robots / 2,288 tests | 三 setting 配对；verifier 独立；stage-level failure labels |

**面试 Tip**：被问到 EFE 时，先答“它把物理约束拆成状态访问、reset、实时三个可消融变量”，再补“456 与 391 分母不同，model 与 harness 要分开报”，最后说“全仿真、每 test 单次 attempt、真实机器人只有一个 demo，所以它是能力边界探测，不是 sim-to-real 定论”。

## 8. 精读建议

- **值得精读原文的人**：
  - 做 agentic robot 评测、benchmark 设计、harness 设计的研究者/工程师；
  - 想把 coding agent 接到仿真或真实机器人上，需要知道接口权限、reset、实时性怎么影响成功率的人；
  - 做 sim-to-real 评测，想区分“模型能力”和“评测环境权限”的人。
- **建议章节路径**：先读 §2.1（三 setting 定义）→ §2.2（资源约束）→ §3.2（Privileged vs Standard 的 ablation）→ §3.4/§3.5（agent 差异与控制方式）→ Appendix C（接口、判分、物理）→ Appendix H/J（失败标签与局限）。如果只关心结果，读 §3.1 + 图 7/10/11 即可。
- **不值得精读的理由**：如果你只关心 VLA policy 架构、预训练数据配方、动作 tokenization，这篇不提供这些；如果你需要真实机器人上的统计结论，这篇只有 1 个 buzz-wire demo，不够。

---

[← Back to Theory](./README.md)

**关键引用**：[Embodied First Exam](https://arxiv.org/abs/2611.12177)（arXiv:2611.12177v1，HTML 正文：https://arxiv.org/html/2611.12177）。