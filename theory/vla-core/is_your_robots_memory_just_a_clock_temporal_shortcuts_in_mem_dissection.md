# 你的机器人记忆只是时钟吗？记忆增强 VLA 的时间捷径拆解 (Is Your Robot's Memory Just a Clock? Temporal Shortcuts in Memory-Augmented VLA Policies)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-11-11
>
> **论文**: Is Your Robot's Memory Just a Clock? Temporal Shortcuts in Memory-Augmented VLA Policies
> **链接**: https://arxiv.org/abs/2611.14488
> **证据版本**: arXiv:2611.14488v1 的 HTML 全文（2026-11-10）。候选元数据摘要为截断片段（止于 “Such a null event”），可见部分与全文摘要一致，未发现冲突；下文所有数字取自 HTML 正文表格与附录，未用候选摘要补写。
> **核心定位**: 它不提出新记忆架构，而是用 visible/hidden null event 与时间戳 rescale 两个干预，证明记忆增强 VLA 的失败来自“记忆如何被索引”，而不只是“有没有时间编码”。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 按时间均匀采样并打时间戳的记忆，会把“等待”误读成“工作”，产生 phantom progress；去掉时间编码仍会失败，因为 null 帧仍占记忆槽 |
| 适合精读 | 做长程/记忆 VLA、要设计记忆索引、要给 benchmark 加 shortcut 检验的团队；重点看 §4、§5.3、§6 |
| 可以跳过 | 只关心新模型架构或真实机器人部署的读者；本文没有真机策略评测，因果证据全在仿真 |
| 落地可行性 | 中：event index 是 test-time 干预、无需重训，但依赖仿真特权信号，真实部署未验证 |
| 主要风险 | 评测以 50 episodes/cell 为主，训练方差与部分效应同量级；多数模型单 seed；null event 为脚本注入 |

💡 **X-Ray 开场**

这篇论文问三个问题：记忆 VLA 真的在“记住发生了什么”吗？如果机器人原地停顿 5 秒，什么也没改变，策略会不会以为任务又完成了一次？答案是：会，而且失败几乎全是“提前按按钮”。对 VLA 研究者的意义是：以后评记忆，不能只看成功率，要先排除“读时钟”这条捷径；记忆的索引方式比是否加时间戳更根本。

📍 **研究全景时间线**

```
2025  π0.5 (CoRL 2025) 提供 VLA 基座
  │
2026  RoboMME (ICML 2026) 标准化记忆评测，发布 MME-VLA 家族
  │
2026  MemER / MemoryVLA 等记忆设计并行推进（ICLR 2026）
  │
2026  本文：不换架构，先用 null event 诊断“记忆是否在计事件”
  ▼
  局限：全部因果证据在仿真；真实部署未测；事件检测器用仿真信号
```

时间线只列来源确认的节点；更早的记忆网络与视频压缩工作不展开。

## 1. 核心架构/方法总览

### 1.1 系统对比概览

先明确一个边界：本文讨论的 memory 是**已观测历史帧**（past frames），不是未来预测帧。FrameSamp 的 32 帧覆盖从 episode 开始到当前步的历史；它不是 t+1 至 t+H 的未来 chunk。动作 chunk 是策略一次输出的动作块，null event 注入在 action-chunk 边界，但 chunk 长度不是记忆窗口。各模型的记忆容量、采样频率、时间编码必须逐模型归属，不能合并。

| 模型/记忆 | 索引方式 | 容量/采样 | 时间编码 | 训练/推理 | 本文实测（null event） |
|---|---|---|---|---|---|
| FrameSamp-Modul / Expert / Context（MME-VLA，π0.5 基座） | time | 32 帧，全 episode 均匀采样 | 每帧绝对时间步的 sinusoidal encoding | 预训练 π0.5；本文只做推理 | Pick 47→16，Swing 44→10（两次 5 s pause，论文 Fig. 2）；300 步 pause 下 Swing 44→12（Table 2） |
| TokenDrop-Modul | pixel change | 按像素变化选 token | 同样有时间编码 | 只推理 | Pick pause 39→30，fidget 39→21；Swing pause 45→27（Table 2） |
| MemER-style keyframes | sub-goal | keyframe memory | 论文未述 | 只推理 | Pick pause 38→39；air-grasp 38→27（Table 2、Appendix G） |
| Text sub-goals | sub-goal | 在线 Qwen-VL 预测子目标 | 论文未述 | 只推理 | Pick pause 46→48；Swing 在 floor（1/50），不解读 |
| TTT（recency window） | recency | test-time-training memory | 论文未述 | 只推理 | Pick 23→22；Swing 17→16，本身计数能力弱 |
| π0.5（无记忆） | none | 当前观测 | 无 | 只推理 | Pick 18→19；Swing 15→13 |
| SimpleMemVLA | every frame at 2 Hz | 120 帧固定窗口，约 60 s | 每帧前加文本时间戳 | 独立 checkpoint，Qwen3.5-4B + 约 0.9B action head | Pick pause 47（hidden 50），fidget 38（hidden 50）；Swing pause 36（hidden 48），fidget 33（hidden 49）（Table 24、25） |
| LTM-π | every step motion token | 每对相邻帧一个 latent transition token，ordinal position | 无显式时间戳 | 独立 checkpoint，修改 OpenPI π0.5，十倍 RoboMME 演示 | Pick pause 50→7，fidget 50→15；Swing pause 50→9，fidget 50→10（Table 26） |
| Clock（从零训练） | time | 32 槽，保留时间编码 | sinusoidal | 从 π0.5 基座训练 30k steps，batch 64 | Pick 41→17，Swing 48→10（Table 42） |
| NoClock（从零训练） | time 槽位，无时间编码 | 32 槽 | 去掉 τ 分支 | 同 Clock，仅差 use_pos_emb 一行 | Pick 42→21，Swing 47→14（Table 44） |

这张表里最反直觉的是 NoClock：它从未见过时间戳，却仍被 pause 打崩。论文的解释是两条通路——时间戳前进，以及 null 帧占用 32 个槽位。前者是显式时钟，后者是隐式时钟。

### 1.2 关键机制

论文的论证链是：

1. **现象**：在 RoboMME 的两个 counting task 上，注入 5 秒 pause 或 wrist fidget，成功率从 47/50 掉到 16/50（PickXtimes），从 44/50 掉到 10/50（SwingXtimes）。失败几乎全是提前按按钮。
2. **归因**：同样的物理事件，如果其帧不进入记忆（hidden 条件），成功率回到基线附近。所以问题在“记住了什么”，不在“被打断”本身。
3. **机制**：演示节奏稳定，时间步单独就能预测进度（PickXtimes 89%，SwingXtimes 80%，两个真实 teleop 数据集上超过 92%）。模仿学习没有理由不利用这个廉价特征。
4. **干预**：只缩放时间戳，不改变帧。s=1.5 让策略早停，s=0.5 让它多做重复（Fig. 4）。方向随钟走。
5. **反直觉**：从零训练 NoClock，去掉时间编码，pause 后成功率仍只有 21/50 和 14/50（对应无事件 42/50、47/50，即损失 21/50 与 33/50）；hidden 又恢复。说明槽位本身携带了“有事情发生”的信号。
6. **跨设计**：脆弱性跟随索引方式。三个 FrameSamp 变体全崩；TokenDrop 对 pause 伤害小、对 fidget 伤害大；sub-goal 记忆在 PickXtimes 上不受影响。
7. **修复**：在数据里注入 pause 不解决问题，还损害计数；在推理时用 event index 把“世界回到旧状态”的片段折叠，能恢复一半以上损失。

⚡ **Eureka Moment**：记忆 VLA 的 shortcut 不在“时间戳”这个显式特征里，而在“记忆如何被索引”里；只删时间编码，等于只关掉一个时钟，槽位仍在替另一个时钟计时。

### 1.3 信息流/架构图

```
episode 历史帧 x_0 ... x_t
        │
        ├─ FrameSamp：在 [0,t] 上均匀取 32 帧
        │       │
        │       ├─ 每帧加 absolute time step 的 sinusoidal encoding τ_i
        │       └─ 32 个槽位，槽 j 持有第 floor(j·t/31) 帧
        │
        ├─ TokenDrop：按 pixel change 选 token（仍带 τ）
        ├─ MemER-style：按 sub-goal 选 keyframe
        ├─ Text sub-goals：在线预测子目标文本
        ├─ SimpleMemVLA：2 Hz 每帧 + 文本时间戳，120 帧窗口
        └─ LTM-π：每步一个 motion token，ordinal position
                │
                ▼
        策略 π(a_t | o_t, M_t)
                │
                ▼
        停止/按按钮决策：读“已完成事件数”还是“已流逝时间”？
```

null event 进入记忆有两条路：

```
visible:   x_i (i ∈ N) 占槽   +   τ_i 之后整体 +L
hidden:    x_i (i ∈ N) 不占槽 +   τ_i 不前进
```

论文的关键控制是 hidden：物理事件完全相同，只改记忆输入。若 hidden 恢复，说明伤害通过记忆；若 hidden 不恢复，说明还有物理干扰。SwingXtimes 上 Clock 的 hidden 只到 38/50（no event 48），说明 hidden 并非完美反事实，作者在 Appendix B.1 明确讨论了这一点。

## 2. 数学核心

📌 **Napkin Formula**（一行抓住本质）：

```
演示中：progress ≈ elapsed_time
失败中：failure ∝ null_event 进入记忆索引的方式，而非仅时间戳
```

先给目标：判断策略的停止决策由“已完成事件数”还是“已流逝时间”驱动。论文没有提出新的损失函数，它的“数学”是干预与分解。

**采样跨度**。若在 [0,T] 上均匀取 K 帧，帧间间隔与首末跨度为：

```
Δt = T / (K - 1)
首末跨度 = (K - 1) · Δt = T
```

FrameSamp 的 K=32，所以首末跨度是 31 个 Δt，不是 32 个。Appendix F.1 给出租槽规则：槽 j 持有第 floor(j·t/31) 帧，t 是最新存储帧；这与我上面的跨度一致。不要把 32 帧的 31 个间隔误读成“未来 H 步预测范围”。本文没有未来预测头；若从当前参考 t 预测 t+1 至 t+H，终点距 t 才是 H·Δt。两者不能混用。

**时间戳缩放**：

```
τ'_i = s · τ_i
s > 1 → 表观时间变快 → 早停
s < 1 → 表观时间变慢 → 多重复
```

**null event 的两条通路**：

```
M_t = Index( {x_i}_{i≤t} ∪ {x_i}_{i∈N},  {τ_i + 1[i>end(N)]·L}_{i≤t} )   （end(N) 为 null event 的最后一帧索引；visible 时其后的时间戳整体后移 L）
```

visible 时，null 帧进入索引，且后续时间戳整体后移 L；hidden 时，两者都不发生。NoClock 去掉 τ 项，但 `{x_i}_{i∈N}` 仍占槽，所以第二条通路仍在。

**事件索引**（test-time，不改训练）：

```
keep x_t  iff  scene_changed(x_t, last_kept)
              or grip/load_switch
              or not closed_loop(x_t, kept_frame p)

closed_loop: 机器人构型回到 p 的容差内，且外部场景不变
             → 把 p 之后的保留帧折叠为一帧、一个 clock tick
```

变量说明：

| 符号 | 含义 | 本文取值/说明 |
|---|---|---|
| K | 任务要求的重复次数 | PickXtimes K=1..5；SwingXtimes K=1..3 |
| T | episode 工作步总长 | 因任务而异，论文未给统一值 |
| Δt | 均匀采样帧间隔 | T/(K-1)，K 为采样帧数时同理 |
| L | null event 持续步数 | 100 步 = 5 s；dose 研究用 30、300 步 |
| s | 时间戳缩放因子 | 0.5、0.6、0.75、1.1、1.25、1.5 等 |
| τ_i | 第 i 帧的时间戳 | sinusoidal encoding of absolute time step |
| M_t | 记忆 | 由索引函数 I 从历史帧填充 |
| N | null event 的帧集合 | pause / fidget / air-grasp |

直觉：时间戳是“显式时钟”，槽位是“隐式时钟”。只删显式时钟，隐式时钟还在走；所以 NoClock 仍失败。事件索引试图让记忆只在“世界改变”时前进，而不是在“时间经过”时前进。

## 3. 带数字走一遍：玩具例子

> 以下为**假设例子**，数字用于验算机制，不是论文实测。论文的实测数字在括号中标注。

假设一个 PickXtimes episode，K=3，无事件时中位按按钮时间 t0=579 步（论文 Appendix E.3 给出 j=1..5 的中位 press time：279、424、579、729、852 步）。按论文的 139 步/重复估算，579/139≈4.17，但这是“按按钮时间”而不是“完成次数”，只用于方向验算。

**干预一：时间戳缩放 s=1.5**。策略在真实时间 t 读到的表观时间是 1.5t。若它按“表观时间达到 579 就按”行事：

```
1.5 · t = 579  →  t = 386 步
386 / 139 ≈ 2.78 次重复 < K=3
```

所以应在第 3 次完成前按按钮，即 early stop。反向 s=0.5：

```
0.5 · t = 579  →  t = 1158 步
1158 / 139 ≈ 8.33 次重复 > K=3
```

策略会多做重复，即 extra repetitions。论文实测 s=1.5 时 Pick 成功 47→18，32 个失败全是 early press；s=0.5 时 24/50，失败以 extra 为主（23 个 extra、3 个 early）。方向一致。

**干预二：两次 100 步 pause，visible**。null event 等效于局部加速时钟：在真实工作步 t 上，表观时间多了 200 步。若策略在表观时间达到 579 时按：

```
t + 200 = 579  →  t = 379 步
379 / 139 ≈ 2.73 次重复 < 3
```

早停。hidden 时表观时间不加 200，t=579 才按，完成 3 次。论文实测 Pick 47→16（visible），hidden 回到 45/50；Swing 44→10，hidden 47/50。方向一致。

**槽位占用验算**。若 episode 工作步 450，插入两次 100 步 pause，visible 总时长 650，32 槽中 pause 帧约占：

```
200 / 650 × 32 ≈ 9.8 槽
```

论文 Fig. 6b 的例子是“约 450 working steps，10 slots”，与这个估算一致。hidden 时总时长仍 450，pause 不占槽。这个验算说明：即使没有时间戳，null 帧仍占掉约 10/32 的记忆容量。论文 NoClock 在 Pick 上 42→21，hidden 44，说明槽位通路单独就足以造成大部分伤害。

**边界**：以上 t0=579 是论文的 no-event 中位按按钮时间，不是策略内部阈值；真实策略的阈值可能更低。玩具例子只验证“方向”和“量级”，不替代论文的 paired McNemar 结果。

## 4. 工程视角

**记忆容量与频率**。FrameSamp 是 32 帧覆盖整个 episode，采样间隔随 episode 长度变化；它不是固定帧率。SimpleMemVLA 是 2 Hz、120 帧固定窗口，约 60 秒；LTM-π 是每步一个 token，token 数随 episode 线性增长；MIKASA 实验用 64 槽。把这些“记忆窗口”混为一谈会误判延迟和显存。

**推理开销**。FrameSamp 的 32 帧是固定成本，cross-attention 的 key 数固定；SimpleMemVLA 的 120 帧窗口固定，但每帧带文本时间戳，token 数更大；LTM-π 每步加 token，长 episode 下内存线性增长。论文没有给具体延迟数字，不能编造。

**event index 的工程约束**。它是 test-time 干预，不需要重训，但需要四个信号：7 个 arm joint + 2 个 finger joint 的本体感知、前视相机中的机器人 silhouette、gripper width、arm 上的 net external force。论文的检测器在仿真里用 segmentation 和 contact impulse；真实部署需要从关节读数和相机标定渲染 silhouette，作者明确说“we did not implement this rendering”，也没有建模传感器噪声。参数在 Table 31：joint return tolerance 0.02 rad、finger 0.004 m、exit distance 2 units、stillness 0.05 units、scene unchanged <5 px differ by >16/255、silhouette dilation 3 px、stillness grace 5 frames、grip/load debounce 3 frames。所有 16 个 RoboMME 任务共用一套容差。

**动作 chunk 边界**。主实验 MME-VLA 的 null event 注入在 action-chunk boundaries after policy steps 48 and 160；LTM-π 因每 10 步 replan，其注入点改为 policy steps 50 and 160。这些注入点是注入时机，不是记忆窗口。不要把某个模型的 action chunk 长度写成所有实验共有的记忆容量。

**失败代价**。event index 在 PickXtimes 上会把真实重复误折叠：无 null event 时，PickXtimes test 从 47 降到 41（论文 Table 34、36），validation 上 139/150 降到 120/150（17 坏、4 好，permutation p=0.005）。其他 15 个任务总体不亏（316/750 vs 317/750）。工程含义：事件索引不是免费午餐，它在“真实重复也回到旧状态”的任务上会重复一步。

## 5. 数据与评测

必须把预训练、仿真评测、真实机器人设置分开写。

**预训练**。MME-VLA 家族基于 π0.5。Clock 与 NoClock 从 π0.5 base checkpoint 出发，按发布的 recipe 在 RoboMME 训练数据上训练 30k steps，batch 64；两者只差一行 `use_pos_emb`（NoClock 去掉时间编码）。多数从零模型只训练一个 seed（42）；Clock 和 NoClock 另有 seed 43，在第二台机器上训练与评测。SimpleMemVLA 是独立代码库（Qwen3.5-4B + 约 0.9B action head），直接评测 released checkpoint；LTM-π 是修改过的 OpenPI π0.5，训练数据是 RoboMME 演示的十倍，也直接评测 released checkpoint。

**仿真评测**。主战场是 RoboMME 的两个 counting task：PickXtimes（K=1..5，16/10/12/6/6 个 episode）和 SwingXtimes（K=1..3，15/12/23 个 episode）。每个 cell 50 个 test episodes，使用 benchmark 的 environment seeds 和 policy noise seed 7，所有条件按 episode 配对；统计用 exact two-sided McNemar，Appendix L 对 42 个比较做 Holm 校正。另有 50 个 validation episodes 用于 held-out 复现。MIKASA-Robo 是探索性研究：64 槽模型，BlinkCount-Medium 只成功 24/50，低于预先设定的 60% gate，所以所有 MIKASA 结果只作 exploratory，不作主结论。

**真实机器人**。本文没有真实机器人策略评测。所有 causal evidence 来自仿真。真实数据只用于 §5.1 的“时间预测进度”分析：austin_sailor 90 个演示、stanford_hydra dishes 86（3-cycle）/110（all）个演示、stanford_hydra 3 tasks (all) 570 个演示、FMB 1804 个演示、LIBERO-10 207 个演示（sim teleop）。时间单独预测进度的准确率分别是 0.922、0.933、0.824、0.617、0.747、0.916；这些是演示数据的统计，不是真机 rollout 的成功率。作者在 Appendix D 明确说，这不证明在真实数据上训练的策略会使用该相关，也不证明真实机器人会在 null event 下失败。

**数据开放范围**。论文正文给出项目链接，但本次来源未验证仓库内容；按编辑约束，不展开代码/数据开放细节，也不引用额外 URL。

**评测的频率与单位**。50 episodes/cell 是评测次数，不是训练轮次；30k steps 是训练步数，不是 episode 数；100/300 environment steps 是 null event 的持续时间（论文按 20 Hz 换算，100 步=5 s）；“两次 5 秒 pause”是每个 episode 注入两次，不是每个 training step 注入。不要把“50 episodes”写成“50 次训练”。

## 6. 能力与失败模式

**能做什么**。论文给出了一个可复用的诊断协议：visible vs hidden null event、时间戳 rescale、按索引比较不同记忆设计。这个协议不依赖新模型，可以直接加到任何记忆 benchmark。event index 作为一个 test-time 干预，在 7 个 test probe 上把 248/350 从 stock 的 111/350 拉回，hidden 参考是 295/350；在 held-out validation 上 pooled 从 118/350 到 243/350（+35.7 点，p<10^{-4}），6/7 个 probe 显著。

**不能做什么**。

| 失败模式 | 具体表现 | 论文归因 |
|---|---|---|
| phantom progress | pause/fidget 被读成进度，提前按按钮 | 时间戳 + 槽位两条通路 |
| 时间戳 rescale 方向翻转 | s=1.5 早停，s=0.5 多重复 | 策略读表观时间 |
| 去掉时间编码仍失败 | NoClock Pick 42→21，Swing 47→14 | null 帧占槽 |
| pixel-change 记忆对 fidget 更脆 | TokenDrop Pick fidget 39→21，pause 39→30 | fidget 改变像素但不改变世界 |
| event index 误折叠真实重复 | PickXtimes 无事件 47→41（test），validation 139→120/150 | 真实重复也回到旧状态 |
| 单向 air-grasp | Swing 28→40（event index），hidden 48 | 事件结束不在起点，无法 loop closure |
| 随机 micro-stops | Pick 20→27，Swing 17→24，恢复不显著 | 每次 stop 有减速/ settling，既非静止也非闭环 |
| TimedTransfer | 时间本身就是任务，折叠 pause 错误 | 构造性失败 |
| pause augmentation 损害计数 | Pick 无事件 41→24，Swing 48→19；K≥2 几乎全失败 | 策略不再依赖时间，计数能力下降 |

**训练方差**。Clock 同 recipe、同 seed 的两次重训，在 PickXtimes 无事件上相差可达 12 个 episode，hidden 上相差 9 个；第二台机器上的 no-event 是 32（cluster 41）。作者自己说，小于约 10 个 episode 的差异要谨慎。多数从零模型只有一个 seed。这是本文证据强度的一个现实约束。

### 6.1 隐含假设

1. **演示节奏稳定**。时间能预测进度，是因为演示几乎没有 idle stretch；真实数据可能被 trim 过，统计上看起来比原始采集更规律。作者在 Appendix D 明确承认这一点。
2. **null event 是脚本化的**。所有 pause/fidget/air-grasp 都是注入的，部署中未宣布的 null event 频率未知。论文没有测真实机器人上的自然停顿。
3. **hidden 是理想反事实**。hidden 把事件帧从记忆中删掉、时间戳不前进，但物理事件仍发生。SwingXtimes 上 Clock 的 hidden 只到 38/50（no event 48），LTM-π 的 hidden 只到 31/50，说明“隐藏”并不完美；作者用 Appendix B.1 检查了 arm settling，仍未完全解释。
4. **事件检测器用仿真特权信号**。silhouette 来自 simulator segmentation，net force 来自 contact impulse；真实相机、传感器噪声、标定误差都未建模。作者说这是“main open problem”。
5. **评测规模**。50 episodes/cell，paired McNemar；训练方差与部分效应同量级。Holm 校正后主文声称显著的效果仍显著，但一些子组比较（如 SimpleMemVLA 的 history rescale，p=0.031/0.063）本身较弱。
6. **记忆只指历史帧**。本文不涉及未来预测帧，也不把 action chunk 当作记忆窗口；读者不应把 32 帧历史窗口与未来 H 步预测范围混为一谈。

## 7. 与相关工作对比

| 工作 | 关注点 | 记忆索引 | 是否注入 null event | 与本文的关系 |
|---|---|---|---|---|
| RoboMME (ICML 2026) | 标准化记忆评测 | 多种 released designs | 否 | 本文的主 benchmark；本文在其 counting task 上做诊断 |
| MemER (ICLR 2026) | 经验检索 | keyframe / sub-goal | 否 | 本文复现其 keyframe 记忆；pause 不影响，air-grasp 会（38→27） |
| MemoryVLA (ICLR 2026) | 感知-认知记忆 | perceptual memory bank | 否 | 本文在 related work 引用，但未直接评测，不能替它下结论 |
| TokenDrop | 像素变化选 token | pixel change | 本文测 | pause 伤害小，fidget 伤害大；说明“选内容”不等于“选事件” |
| SimpleMemVLA | 原生视频记忆 | 2 Hz 每帧 + 文本时间戳 | 本文测 | 重采样历史会翻转失败；pause 影响小，fidget 大；Swing 上 pause 也显著 |
| LTM-π | latent transition token | 每步 motion token | 本文测 | pause 与 fidget 都崩；Swing hidden 也不完全恢复，作者归因于 hidden 的额外 swing |
| AGM | 把尝试动作记为 progress | achievement memory | 否 | 本文引用其“失败 grasp 变成持久错误”，未直接对比 |
| RoboMME-Interference | 无关 session 挤占记忆 | frame-sampled | 否 | 本文引用其“无关 session 挤占”，未直接对比 |

对比时要小心：本文的贡献不是“event index 新架构”，而是“诊断 + 最小干预”。event index 的检测器是仿真专用的，作者自己说它“tests this account rather than a new architecture”。

**面试 Tip**：被问到“记忆 VLA 怎么评才不被时钟骗”时，答：先做两个干预——visible/hidden null event 和只缩放时间戳；如果失败随 hidden 消失，说明走记忆；如果 rescale 能翻转失败方向，说明读时钟；再看换索引后失败是否跟着索引走。不要只报成功率。

## 8. 精读建议

- **值得精读原文的人**：
  1. 正在设计长程记忆 VLA 的研究者，尤其是用 uniform sampling + time stamp 的人；
  2. 要给记忆 benchmark 加 shortcut 检验的工程师；
  3. 想复现 event index 或评估其真实部署可行性的团队。

- **建议章节路径**：先读 §4（现象与 hidden 控制）→ §5.2（rescale 干预）→ §5.3（NoClock，最关键的反直觉）→ §6（event index 与边界）→ Appendix I（检测器细节与失败归因）。如果只关心评测，读 §3 的 protocol 和 Appendix A 的统计设置。

- **不值得精读的理由**：如果你不做机器人学习、或只关心新模型架构而非评测诊断，读摘要和 §4 即可；本文没有真机实验，也没有提出可直接部署的通用 null-event detector。

---

[← Back to Theory](./README.md)

**关键引用**：论文全文 https://arxiv.org/abs/2611.14488 ；HTML 版本 https://arxiv.org/html/2611.14488 。本文数字均取自该 HTML 正文的 Fig. 2、Fig. 4、Fig. 6、Table 2、Table 18、Table 24–26、Table 34–38、Table 41–44；未使用候选摘要补写。