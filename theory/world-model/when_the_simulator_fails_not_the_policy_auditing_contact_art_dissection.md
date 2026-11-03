# 当仿真器而非策略失败：审计仿真机器人基准中的接触伪影 (When the Simulator Fails, Not the Policy: Auditing Contact Artifacts in Simulated Robot Benchmarks)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-11-03
>
> **论文**: When the Simulator Fails, Not the Policy: Auditing Contact Artifacts in Simulated Robot Benchmarks
> **链接**: https://arxiv.org/abs/2610.41857
> **核心定位**: 它不提出更强的 VLA，而是把“策略已放好、仿真器却把物体弹飞”的接触伪影做成可记录、可逐位重放、可单参数重跑的审计工具，并用 PhysX 阈值修复验证仿真分数偏差。
>
> **证据版本说明**: 本文依据 arXiv:2610.41857v1 的 HTML 正文（对应 v1 PDF，2026-10-30）写作。候选元数据中的 abstract_snippet 在 “We presen” 处截断，未用于补写论文主张；正文摘要与候选片段在可读部分一致，但候选片段不完整。所有数字、主张与限制以正文和表格为准。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 论文实测：在 RoboTwin 2.0 五个薄壁容器任务中，LingBot-VA 的 44 个失败有 40 个含接触伪影；把 PhysX 5.3.1 的 GJK 退化三角形阈值改到 5.4.0 后，100 个闭环 episode 中带伪影从 12 降到 0，成功率从 89% 升到 98%（Table 3）。 |
| 适合精读 | 如果你在做 VLA/世界动作模型的仿真评测、benchmark 复现、物理引擎接触调试，重点看 §3、§4.2、§4.3 与 Appendix D.1/D.3。 |
| 可以跳过 | 如果你只关心 VLA 模型结构、预训练数据配比或真机部署，这篇距离中等：它不提供新 VLA 架构，也没有真机实验。 |
| 落地可行性 | 中到高：若你用 SAPIEN/PhysX 5.3.1 且碰撞体在厘米级，修复路径明确；若引擎不可改、资产管线不同，只能做审计与区间估计。 |
| 主要风险 | 修复只在一个引擎路径和五个任务上闭环验证，闭环样本小；RoboDojo 的假接触原因未定位；检测 precision 只在一个任务上测。 |

💡 **X-Ray 开场**

这篇论文问三个问题：仿真 benchmark 的成功率，失败到底算策略的，还是算仿真器的？论文发现，策略把物体放进容器后，PhysX 会在一帧 4 ms 内把物体以几十米每秒弹飞，而机器人并没有碰到它；更关键的是，benchmark 自己的 scripted demonstration 也会以同样方式失败。对 VLA 研究者意味着：仿真分数 `S/N` 不是“策略能力”的纯净度量，至少要先审计接触伪影，再决定是报告区间、排除伪影 episode，还是修引擎后重测。

📍 **研究全景时间线**

```
1988 GJK 距离算法
  → 2012 MuJoCo 软接触模型
  → 2023 LIBERO
  → 2024 RoboCasa / PhysX 5.4.1 文档
  → 2025 RoboTwin 2.0
  → 2026 SimuGuard 审计（本文）← 当前位置
局限：主要修复路径只在 RoboTwin + PhysX 5.3.1 + 厘米级碰撞体上闭合；
RoboDojo 的假接触原因未定位，MuJoCo 基准只做了较小规模审计。
```

## 1. 核心架构/方法总览

### 1.1 系统对比概览

SimuGuard 不是策略模型，而是挂在 benchmark 评测脚本旁边的审计插件。它不改变 evaluator、benchmark 和 policy 的代码，只包住 physics step。

| 模块 | 输入 | 输出 | 时序/频率 | 训练/推理差异 |
|------|------|------|-----------|----------------|
| Adapter | benchmark 自带 evaluator；SAPIEN/Isaac Sim/robosuite 的 step 函数 | 包住 physics step 的 wrapper | 每个 physics step | 无训练；只在评测期旁路运行 |
| Recorder | 每步 actuation、body pose/velocity、contacts | 逐 step 记录；可按 seed 重放 | RoboTwin 4 ms/step；LIBERO/RoboCasa 2 ms/step；RoboDojo 4 ms/step | 无训练；记录开销 7.5–18.4 ms/step 不等 |
| Detector | 记录的速度、接触、位移 | 候选接触伪影事件 | 每个事件；阈值对所有任务一致 | 无训练；三阶段规则 |
| Replayer | seed、记录 actuation、benchmark 写入的状态 | 逐位一致的重放轨迹 | 从第 1 个 physics step 重建 | 无训练；必须从开头重放 |
| Rerun | 重放 + 一个仿真参数改变 | 同动作、同 success check 的新结果 | 从第 1 步，或伪影前 0.1 s 切换 | 无训练；单参数干预 |
| Scorer | 原始结果、flagged episodes、重跑结果 | reported / replay / exclusion / closed-loop 与区间 | 每个 evaluation | 无训练；统计与报告层 |

### 1.2 关键机制

论文的机制可以拆成四步：

1. **逐 physics step 记录，而不是只看视频。** 视频帧间隔大于 physics step，弹飞发生在两帧之间；只看视频只能看到“物体飞了”，看不到是谁给的冲量。
2. **用三阶段检测器筛“接触启动的异常运动”。** 先找最近 40 ms 内有接触、速度超过 0.5 m/s、且一个 4 ms step 内速度变化至少 0.35 m/s 的物体；再在 240 ms 内确认位移至少 4 cm，并满足自由飞行至少 16 ms 或与容器剧烈接触；最后用重力账和载体速度账排除“掉落”和“机器人通过容器间接带动”。
3. **逐位重放。** 同一 seed、同一 actuation、从第 1 个 physics step 开始，重建场景并施加记录动作；论文报告 96/96、400/400、13/13 等重放检查（Table 8）。
4. **单参数重跑。** 在伪影前 0.1 s 打开一个限制（如 PhysX 最大 depenetration velocity），或直接改引擎常数；用 benchmark 自己的 success check 打分。若结果翻转，且切换前逐位一致，就把这次翻转归因于该物理参数。

⚡ **Eureka Moment**：这篇论文最核心的想法不是“检测快速运动”，而是**把单个接触报告与 episode 结果连起来**：同 seed、同动作、只改一个物理参数，若失败变成功，则这个失败至少在这条路径上属于仿真器，而不是策略。

### 1.3 信息流/架构图

```
benchmark evaluator
      │  调用 policy，执行 action
      ▼
policy (LingBot-VA / HyVLA / π0.5 / XR-1)
      │  action
      ▼
simulator physics step  ←── SimuGuard Adapter 包住 step
      │  每步记录 actuation / pose / velocity / contacts
      ▼
Recorder ──► Detector ──► flagged episodes A
      │                         │
      │                         ├─► Replayer：同 seed + 同 actuation，逐位重放
      │                         └─► Rerun：一个参数改变，benchmark success check 打分
      ▼
Scorer：reported S/N；replay rate；exclusion rate；closed-loop rate；区间
```

## 2. 数学核心

📌 **Napkin Formula**：

```
分数偏差 = 被接触伪影翻转的 episode 数 / N
```

更完整地，论文用重放率把 flagged episode 的原始结果 `y_e` 替换为重跑结果 `r_e`：

```
s_hat = (1/N) * ( S + Σ_{e∈A} (r_e - y_e) )
```

先给目标：在不知道“没有伪影的物理”下真实成功率是多少时，用记录和重跑来估计。变量说明：

| 符号 | 含义 |
|------|------|
| `N` | 被计分的 episode 数；scripted demonstration 失败的 seed 不计入 |
| `S` | 原始成功率分子，即成功 episode 数 |
| `A` | 含接触伪影的计分 episode 集合 |
| `S_A` | `A` 中原本成功的 episode 数 |
| `y_e` | episode `e` 的原始结果，成功 1，失败 0 |
| `r_e` | 同记录动作、在伪影前切换一个参数后的重跑结果 |
| `|A|` | flagged episode 数 |

若只有 flagged episode 可能改变，成功率区间是：

```
lower = (S - S_A) / N          # 所有 flagged episode 都失败
upper = (S - S_A + |A|) / N    # 所有 flagged episode 都成功
```

论文在五个 RoboTwin 容器任务上给出：LingBot-VA 的区间为 `[75.8%, 98.5%]`，围绕报告的 `83.1%`；HyVLA 为 `[86.0%, 96.0%]`，围绕 `90.0%`（§3.6）。这个区间的唯一假设是“未 flagged 的 episode 不变”；如果检测器漏检，假设就破了。

检测器的三阶段阈值可以写成：

```
候选：最近 40 ms 内有接触
      且 speed > 0.5 m/s
      且一个 4 ms step 内 Δv >= 0.35 m/s

确认：240 ms 内位移 >= 4 cm
      且（自由飞行 >= 16 ms 或与容器剧烈接触）

重力排除：peak_speed / sqrt(2 * g * Δz) > 1.5

载体排除：peak_speed / fastest_carrier_speed >= 1.5
```

论文特别说明：速度变化阈值按 4 ms step 给出，在其他引擎上随 physics step 缩放；检测器阈值对所有任务一致（Table 7）。

GJK 退化三角形阈值是本文的“一个常数”：

```
PhysX 5.3.1：|ab × ac|^2 <= 2^(-23) ≈ 1.19e-7
PhysX 5.4.0：比较 (2^(-23))^2 ≈ 1.42e-14
```

直觉：这个阈值以米为单位，却不随几何尺度缩放。RoboTwin 把容器拆成 1–2 cm 的凸片；在厘米尺度，合法三角形也可能低于阈值，GJK 提前退出，接触生成把“间隙 + 片厚”当成穿透深度，solver 在同一个 step 内把深度消掉，物体就被弹飞。

## 3. 带数字走一遍：玩具例子

### 3.1 论文 Figure 1 的能量账（论文实测数据，算术验算）

论文记录：10 g 的 can 在 4 ms 内从 0.2 m/s 升到 22 m/s；同一 step 没有机器人 link 接触它，最快的机器人 link 低于 0.1 m/s。验算动能变化：

```
m = 0.01 kg
v1 = 0.2 m/s
v2 = 22 m/s
ΔKE = 0.5 * m * (v2^2 - v1^2)
    = 0.5 * 0.01 * (484 - 0.04)
    = 2.4198 J ≈ 2.42 J
```

这与论文写的“gain of 2.4 J”一致。再看重力能否解释：一个 4 ms step 的自由落体速度增量约 `g * 0.004 = 0.0392 m/s`，远小于 21.8 m/s 的跳变。因此这个 step 的动能不可能来自重力或机器人接触；论文把它归因于 PhysX 的假接触与 solver 的深度移除。

### 3.2 假设的检测阈值例子（以下为假设，不是论文实验）

假设一个 10 g 物体在 4 ms 内从 0.5 m/s 到 0.85 m/s，刚好满足 `Δv = 0.35 m/s` 的候选条件。验算：

```
ΔKE = 0.5 * 0.01 * (0.85^2 - 0.5^2)
    = 0.005 * (0.7225 - 0.25)
    = 0.0023625 J ≈ 2.36 mJ
```

再看确认条件：240 ms 内位移至少 4 cm，等价于平均速度至少 `0.04 / 0.24 = 0.1667 m/s`。这个假设例子说明：检测器不是只看“速度快”，而是看“接触后一个 step 内的速度跳变 + 后续位移/飞行”。它也会把“掉落”与“弹飞”分开，因为重力阶段会做 `sqrt(2gΔz)` 的对照。

### 3.3 GJK 阈值的尺度验算（论文数据，算术验算）

论文说“a right triangle with legs of 1.8 cm is already below it”。验算：

```
leg = 0.018 m
|ab × ac| = leg * leg = 3.24e-4
|ab × ac|^2 = (3.24e-4)^2 = 1.04976e-7
阈值 = 2^(-23) = 1.19209e-7
1.04976e-7 < 1.19209e-7
```

所以这个直角三角形会被 5.3.1 判为退化；这正好解释了为什么厘米级碰撞片会触发问题，而米级几何不一定触发。

## 4. 工程视角

### 4.1 开销与记录量

论文的审计是“每个 physics step 都记录”，不是抽样。开销如下（论文 §C.1）：

| 平台 | 记录开销 | 记录量 | 备注 |
|------|----------|--------|------|
| SAPIEN / RoboTwin | 7.5–11.6 ms/step | 85–105 bytes/step | physics 本身约 0.5 ms；审计模式明显变慢 |
| MuJoCo / LIBERO | 2.8 ms/step | 355 bytes actuation + 304 bytes object states = 659 bytes/step | CPU 渲染时 wall time +23% |
| Isaac Sim / RoboDojo | 6.9–18.4 ms/step | 未给 bytes | 占 episode wall time 的 3–7% |

工程含义：SimuGuard 适合“审计/复现模式”，不适合把监控常开当成生产评测。4 ms step 上增加 7.5–11.6 ms，意味着 physics 不再是瓶颈，记录 I/O 和状态序列化才是。

### 4.2 精确重放的工程约束

论文报告的重放不是“差不多”，而是逐位一致；但它需要几个不显然的细节：

- PhysX 不能在接触后精确恢复保存状态：通过 state interface 恢复的轨迹会偏毫米到厘米，所以必须从第 1 个 physics step 重建并重放。
- benchmark 代码会在两个 physics step 之间改状态：RoboCasa fixture 会设门 joint；robosuite 1.4 会在每步前重算 dynamics，改写 solver warm start；这些都要记录并写回。
- RoboTwin 的 success check 会读“命令的 gripper opening”，所以要从记录的 drive target 恢复夹爪命令。
- LIBERO 的 robosuite 1.4.0 用未设种子的随机生成器放 fixture，场景不能仅靠 seed 复现；需要把记录的 body pose 写回。

这些约束说明：**“可重放”是审计工具的核心能力，也是它最大的工程成本**。如果只记录视频或最终状态，无法把一次弹飞归因到单个接触报告。

### 4.3 单参数重跑的边界

重跑有三种形式：

1. 从第 1 个 physics step 就改参数；
2. 在第一个伪影前 0.1 s 打开参数，切换前逐位一致；
3. 保持参数开，做新的 closed-loop rollout。

第二种最适合因果归因：切换前完全一致，切换后第一个事件就是伪影。论文报告 273/273 次切换前重放逐位一致（Table 8）。但论文也提醒：第三种不再是逐 episode 可比，因为策略会采样新动作；它测的是“修好物理后的新 rollout”，不是同一动作的反事实。

## 5. 数据与评测

### 5.1 先区分历史帧、未来帧与记忆窗口

本文没有研究任何模型的记忆机制。必须先区分：

- **已观测历史帧**：策略输入的历史观测；
- **未来预测帧**：world action model 预测的未来视频帧；
- **动作 chunk**：一次 query 执行多少个 action；
- **physics step**：仿真器推进的时间步。

论文只描述 LingBot-VA 是 world action model，“predicts future video together with its actions”；它没有给出历史帧数 `K_hist`、采样间隔 `Δt_hist`、未来预测 horizon `H`、未来帧间隔 `Δt_future`，也没有给出任何模型的记忆窗口。HyVLA 的 “seven actions executed per query”、π0.5 的 “five actions per query”、RoboDojo 的 “10 physics steps per action” 都是执行配置，不是记忆窗口。

若未来要审计这类时序设置，通用核算是：

```
历史帧首末跨度 = (K - 1) × Δt
未来预测终点距当前参考 = H × Δt
```

这里 `K` 是均匀间隔采样帧数，`Δt` 是帧间隔；若从当前参考 `t` 预测 `t+1` 至 `t+H`，终点距当前参考是 `H × Δt`。这是通用核算，不是论文实验设置。连续未来 chunk 的预测区间不能直接当作历史上下文或记忆窗口；也不能把 RoboTwin 4 ms、LIBERO/RoboCasa 2 ms、RoboDojo 4 ms 这些单一设置的步长写成所有实验共有的记忆窗口。

### 5.2 逐模型归属：架构、检查点、评测与开放范围

| 模型 | 论文中的类别/角色 | 预训练/检查点披露 | 仿真评测 | 真实机器人 | 记忆/时序窗口 |
|------|------------------|------------------|----------|------------|----------------|
| LingBot-VA | world action model；预测未来视频与动作 | 论文只给 RoboTwin checkpoint 来自 ModelScope；未报告预训练数据、训练步数 | RoboTwin 2.0 全部 50 任务，N=2060，成功 1870（90.8%）；五个容器任务报告 83.1% | 无 | 未报告 K、H、Δt 或记忆窗口 |
| HyVLA | VLA | 论文只给 Hy-Embodied-0.5-VLA-RoboTwin checkpoint，经 XPolicyLab；7 actions/query | 五个容器任务 + 六个其他任务；Table 4 共 220 episodes，10 个 artifact；五个容器任务报告 90.0% | 无 | 未报告记忆窗口；7 actions/query 是动作执行块 |
| π0.5 | VLA | 论文只给 RLinf-Pi05-LIBERO-130-fullshot-SFT checkpoint | LIBERO 400 episodes，成功 381（95.2%）；2 个 artifact | 无 | 未报告记忆窗口；5 actions/query 是动作执行块 |
| XR-1 | 被测策略（论文未展开架构） | 论文只给 RoboCasa 与 RoboDojo 官方 checkpoint | RoboCasa 300 episodes，5 个 artifact；RoboDojo 40 episodes，16 成功，7 个 artifact | 无 | 未报告记忆窗口；RoboDojo 10 physics steps/action 是仿真执行配置 |

**预训练与检查点边界**：论文没有报告这些模型的预训练数据组成、训练 token 数、训练步数、模型参数量或许可；因此不能从本文推断预训练规模，也不能把某个模型的 checkpoint 来源写成所有模型共有的数据开放范围。

**数据开放范围**：论文称代码、检测器、重放/重跑工具、episode manifest 和表格脚本已提供，录制数据将随代码发布；本文不引用其仓库地址。模型权重方面，论文只给 checkpoint 来源：LingBot-VA 来自 ModelScope，HyVLA 经 XPolicyLab，π0.5 为 RLinf checkpoint，XR-1 为官方 checkpoint；许可与完整复现条件未在正文展开。

### 5.3 仿真评测设置

**RoboTwin 2.0（主实验）**：SAPIEN + PhysX 5.3.1，4 ms physics step；干净配置（无背景、桌子、光照随机化；seen instructions）。每个 seed 先跑 benchmark 自己的 scripted demonstration，失败则跳过该 seed；只有通过的 seed 才计入 `N`。LingBot-VA 在全部 50 个任务上跑 2060 个计分 episode；五个薄壁容器任务是失败集中区。HyVLA 在五个容器任务和六个其他任务上跑 220 个 episode。

**LIBERO**：MuJoCo 3.2.3，2 ms physics step，25 physics steps/action；π0.5，400 episodes，成功率 95.2%。论文还审计了 2000 个存储初始状态：650 个一开始就位于支撑物内部；LIBERO-Object 的 500 个初始状态全部触发 contact ejection，MuJoCo 在最初 10 ms 内把物体以最高 1.38 m/s 弹开。到策略实际动作的 0.75 s 时物体已静止；130 个从这些状态开始的策略 episode 成功率 96.2%，其余 270 个为 94.8%。策略动作期间只有 2 个 artifact：其中一个 MuJoCo 报告 242.8 mm 深、412 N 的接触，但自身距离查询只给 1.4 mm，mug 以 1.57 m/s 离开夹爪并失败；论文明确说“这个接触是否导致失败未测试”。

**RoboCasa**：MuJoCo 3.3.1，2 ms step，XR-1，300 episodes，5 个 artifact，最高速度 1.19 m/s，全部为失败 episode。

**RoboDojo**：Isaac Sim 5.1.0 + Isaac Lab 0.54.3，PhysX GPU dynamics，4 ms step，10 physics steps/action；XR-1，40 episodes，16 成功，7 个 artifact，全部为失败。论文追踪到一个假接触：tube 被报告 11 mm 进入 rack 壁，接触点却在 tube 凸包外 7.6 mm；depenetration limit 1 m/s 把出射速度从 4.31 降到 1.88 m/s，但论文说这条路径的原因“尚未定位”。

**真实机器人设置**：本文没有真实机器人实验。所有结论来自仿真内审计、逐位重放和重跑；论文引用 SimplerEnv/RoboArena 作为外部背景，但没有报告自己的真机迁移或真机排序验证。因此不能把本文结果外推为“真机成功率也会从 89% 到 98%”。

## 6. 能力与失败模式

### 6.1 能做什么

- **定位假失败**：把“策略放好物体后仿真器弹飞”与“策略动作错误”分开；论文的核心证据是 benchmark 自己的 scripted demonstration 也会以同样方式失败，而修复后不再失败。
- **逐位重放与单参数重跑**：同一 seed、同一 actuation，切换前逐位一致；这让“结果翻转”可以归因到切换后的物理。
- **跨 benchmark 挂载**：LIBERO、RoboTwin、RoboCasa、RoboDojo 都有 adapter；不改变 evaluator、benchmark、policy 代码。
- **给出有条件的分数**：reported、replay、exclusion、closed-loop，以及 `[lower, upper]` 区间；在不能修引擎时，至少不把 `S/N` 当成无偏真值。
- **修引擎后重测**：在 RoboTwin 上把 PhysX 5.3.1 的阈值改成 5.4.0，五个容器任务 100 个闭环 episode 中 artifact 从 12 到 0，LingBot-VA 从 89% 到 98%；HyVLA 从 90% 到 94%，仍有 1 个 artifact。

### 6.2 不能做什么/失败模式

- **artifact 不等于失败**：LingBot-VA 五个任务中有 17 个成功 episode 含 artifact，16 个在 Place Can Basket，弹飞的 can 可能又落回篮子。检测到 artifact 只说明“有异常接触”，不自动说明“这个 episode 一定失败”。
- **检测器有精度边界**：Place Can Basket 上 40 个检测事件中 37 个被五位机器人工程师判为物理上不合理（precision 0.925），20 个 hard negative 中 0 个被判不合理；但 precision 只在这一个任务上测，其他任务靠 60 个 episode 的人工一致性检查。
- **原因定位不完整**：RoboTwin 的假接触被追到 GJK 阈值；RoboDojo 的假接触是另一条 GPU contact generation 路径，论文明确说“原因尚未定位”；MuJoCo 上 artifact 少且慢，但不是没有。
- **修复不是“调 solver 参数”**：加重物体、增加 solver iterations 不能消除 artifact；depenetration limit 能减慢但不能修复，且整段 rollout 常开会让 Place Object Basket 从 90.0% 掉到 50.0%（Table 1 CL），而修复引擎后是 20/20。论文的结论是改引擎常数，不是把限制常开。
- **闭环样本小**：修复引擎的闭环测量是 2 个策略 × 5 个任务 × 20 seeds；LingBot-VA 在 83 个共享 seed 上，8 个只在修复引擎成功、2 个只在原引擎成功，sign test `p=0.11`，论文自己说这个差异单独看不显著；显著的是 artifact 差异。
- **scripted demonstration 的门控效应**：RoboTwin 先用 scripted demonstration 筛 seed；Place Can Basket 有 22 个 seed 因 demonstration 失败被跳过，其中 16 个含 artifact，13 个在限制下通过。也就是说，artifact 不只影响策略计分，还影响“哪些 seed 进入评测”。

### 6.3 隐含假设

1. **只有 flagged episode 会变**：区间 `[lower, upper]` 依赖检测器不漏检；漏检会让真实区间更宽。
2. **重放的动作仍是策略的选择**：replay rate 假设去掉弹飞后，记录动作仍代表策略在该状态下的意图；closed-loop rate 则假设参数改变不影响任务所需物理。
3. **exclusion rate 假设 flagged episode 与其他 episode 一样难**：如果 artifact 集中在更难的任务，排除后会高估。
4. **修一个常数不会改变其他接触**：论文只改了 PhysX 5.3.1 的 GJK 阈值，并报告 4 个真实穿透保持深度；但没有在完整后续版本上跑 RoboTwin 全量。
5. **benchmark 的 success check 是可信的**：本文审计的是 physics 导致的失败，不是 success check 本身的正确性。
6. **仿真排序可外推到真机**：本文没有真机实验，这个假设没有被检验。
7. **检测器定义等于“仿真器故障”**：论文用人工评审作为最终标准，检测器是候选筛选，不是法律判决。

## 7. 与相关工作对比

| 维度 | 引擎对比/接触仿真研究 | 基准可靠性研究 | SimuGuard（本文） |
|------|------------------------|----------------|-------------------|
| 关注点 | 不同引擎在相同场景下的接触力/轨迹差异；solver 与接触模型 | 成功率如何聚合、方差、捷径解、分布偏移 | 单个接触报告如何导致 episode 结果翻转 |
| 代表工作 | Erez et al. 2015；Le Lidec et al. 2023；Acosta et al. 2022 | RoboEval；LIBERO-PRO；LIBERO-Plus；Jiang et al. 2026 | 本文 |
| 是否连接“一个接触报告”到“该 episode 的结果” | 论文将其定位为否：多在隔离场景比较引擎 | 论文将其定位为否：多把每个 episode 结果当给定 | 是：逐位重放 + 单参数重跑 + benchmark 自己的 success check |
| 是否修复原因后重测 | 通常不涉及固定 benchmark 的分数 | 通常不修物理引擎 | 是：RoboTwin 的 PhysX 阈值修复后重测；RoboDojo 只定位到假接触，未修因 |
| 适用场景 | 物理引擎选型、接触模型研究 | 评测方法学、统计报告 | 仿真 benchmark 维护者、VLA 评测复现、接触伪影审计 |

**面试 Tip**：被问“VLA 仿真分数可信吗”，先答：分数是 `S/N`，但失败可能被接触伪影翻转；用同 seed、同 actuation 的重放加单参数重跑，把责任分到策略或引擎，再报告区间或修复后分数。不要一上来就说“仿真不可信”，要说“哪些 episode 的失败不可归因于策略”。

## 8. 精读建议

- **值得精读原文的人**：
  - 做 VLA/世界动作模型仿真评测、需要复现 RoboTwin/LIBERO/RoboCasa 分数的研究者；
  - 维护 benchmark、需要判断“失败是策略还是仿真器”的工程师；
  - 做物理引擎接触调试、关心 GJK/contact generation 的仿真工程师。
- **建议章节路径**：
  先读 §3.1–§3.6（记录、检测、重放、重跑、评分）→ 再看 §4.2（GJK 阈值因果链）→ 再看 §4.3（修复后分数与区间）→ 需要复现细节时读 Appendix D.1/D.3 与 Table 8；相关工作可最后扫。
- **不值得精读的理由**：
  如果你只关心 VLA 模型架构、预训练数据配方、真机部署，或你已经在做非 PhysX 的接触模型研究，这篇的边际收益较低；读摘要与 §4.3 的分数表即可。

---

[← Back to Theory](./README.md)

**论文链接**：https://arxiv.org/abs/2610.41857