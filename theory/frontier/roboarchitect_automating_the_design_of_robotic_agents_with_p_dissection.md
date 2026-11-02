# RoboArchitect：自动设计带持久记忆的机器人 Agent Harness（RoboArchitect: Automating the Design of Robotic Agents with Persistent Memory）

> ⚙️ 本文由 Moltbot 自动生成 | 2026-11-02
>
> **论文**: RoboArchitect: Automating the Design of Robotic Agents with Persistent Memory
> **链接**: https://arxiv.org/abs/2610.42141
> **核心定位**: 把「冻结 VLA 外面那层控制代码」本身变成被搜索的对象——用 coding model 读 rollout trace、诊断失败、改写 harness，全程不更新任何权重

**版本说明**：本笔记只依据 arXiv:2610.42141v1（[cs.RO], 30 Oct 2026）的 HTML 全文。该版本正文摘要与来源记录中的摘要文本一致，未发现「官方 PDF 与元数据摘要不一致」的情况，因此没有用任何摘要片段替代正文主张。HTML 版存在若干 LaTeX 渲染残片（如 π0.5 的符号），本文按语义还原为 π0.5、Δ 等纯文本写法。

---

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 冻结策略不变，只让 coding model 迭代改写外围 harness，10 个设置平均 held-out 成功率 unseen scenes 9.7%→43.3%、unseen tasks 12.7%→40.2%（论文 Table 1） |
| 适合精读 | 做 agentic robot、harness engineering、冻结策略的 test-time adaptation 的人；重点看 §3、§4.2、§4.4 与 Appendix A |
| 可以跳过 | 只关心 VLA 预训练目标、动作表示、action chunk 设计的人——本文不碰这些 |
| 落地可行性 | 中：单次演化 <6 小时 coding-model 时间、<$150，但依赖完整逐步 trace、成套感知工具，且全在仿真 |
| 主要风险 | 无真实机器人验证；所有数字为单点、未报告重复运行方差；VLABench 用的是 progress score 却统称 success rate |

💡 **X-Ray 开场**

这篇论文问的是一个工程问题，而不是模型问题：既然大家都知道「冻结 VLA 外面套一层手写控制代码」能显著提升机器人成功率，那这层代码能不能不手写？

作者的做法是让一个 coding model 看机器人跑失败的完整录像式 trace，自己诊断、自己改代码、自己重测。答案是：在 5 次迭代预算内，多数设置上确实能改出比微调更好的冻结策略包装。对 VLA 研究者的含义是——**性能瓶颈可能不在权重里，而在权重外面那层几十到几百行的控制逻辑里**。

📍 **研究全景时间线**（年份取自论文参考文献条目）

```
2023  Code as Policies（代码即策略） / LIBERO 基准
  │
2024  OpenVLA、SimplerEnv、RoboCasa（VLA + 仿真评测基础设施成型）
  │
2025  π0.5、SmolVLA、RoboCerebra、VLABench（更强的通用策略 + 更难的基准）
  │
2026  Harness VLA、VoLo 等「手工 harness 包冻结策略」
  │
2026  RoboArchitect（本文，2026-10-30）：把 harness 设计本身自动化  ← 当前位置
  │
局限  全仿真；≤5 次迭代；单一 coding model；无真实机器人闭环
```

---

## 1. 核心架构/方法总览

### 1.1 系统对比概览

RoboArchitect 不是一个新模型，而是一套「优化协议 + 运行时程序」的组合。下表按角色拆开（依据论文 §3、§4.1、Appendix B）：

| 组件 | 角色 | 输入 | 输出 | 是否更新权重 | 频率/预算 |
|------|------|------|------|--------------|-----------|
| Frozen VLA policy | 动作提议者 | 当前观测 + 语言指令 | 低层 action vector | 否 | 由 harness 按需查询，部分设置中被完全绕过（如通用 X-VLA 在 SimplerEnv WidowX 上退化为纯 code-as-policy）；论文未报告具体控制 Hz，仅提到 Google robot 为 3 Hz |
| Harness program | 控制中介 / 被优化对象 | 观测、指令、memory、工具输出 | 低层 action vector | 否（改的是代码文本） | 每个 control step 执行 |
| Coding model | harness 的作者 | 完整 rollout traces（逐步相机帧、机器人读数、全部工具输出） | 候选 harness 代码 | 否（只在上下文里推理） | 每次 iteration 一次，最多 5 次 |
| Auxiliary tools | 感知与几何替代路径 | 图像、深度、指令 | 检测框、mask、point、grasp pose、3D 轨迹基元 | 否 | 按需调用 |
| Persistent memory | 跨 episode 经验载体 | 执行统计、任务线索、本体标定 | 检索结果 | 否（写的是数据结构） | 每 episode 平均 3.4–13.0 次读（Table 3，仅 5 个设置有条目） |

关键区别：**训练/推理的分离发生在「代码」层面，而不是「参数」层面**。优化阶段在训练 episodes 上跑 harness 并记录 trace；部署阶段 harness 冻结，和普通 Python 控制程序没有区别。

### 1.2 关键机制

- **Pass-through 起点**：初始 harness 把观测原样转发给冻结策略、原样返回动作。所以 iteration 0 就等于「裸策略」的标准评测（论文 §3 明确说明 Table 1 的 Bare Policy 列即此）。
- **完整 trace 是燃料**：每次评测记录逐步相机帧、机器人读数、所有模型与工具输出。诊断质量直接取决于 trace 的完整度。
- **贪心接受判据**：候选 harness 在同一训练集上重测，**只有严格超过当前 harness 才被接受**，因此训练集上呈单调轨迹，最多 5 次迭代（§3）。
- **One-shot vs Iterative**：iteration 1 的产物就是「一次性自动设计」的 agent；后续迭代看的是这个 agent 自己暴露出的新失败模式（§4.2）。
- **记忆无 schema**：作者不给记忆结构、不给存取规则，由 coding model 自己决定存什么、什么形式、怎么取（§3、§4.4）。这是本文与 Harness VLA、AGM 等「预定义记忆方案」最直接的分野。
- **统一 harness**：另有一条实验线，在跨 benchmark 的程序生成训练池上只演化一个 harness，让它同时服务 Franka Panda、WidowX、Google mobile manipulator（§4.1、§4.2）。

⚡ **Eureka Moment**：把「agent harness」从工程脚手架提升为**搜索空间**——冻结策略不动，让代码去承担域适应，于是「微调 vs 不微调」这个二选一被替换成「改权重 vs 改外围程序」。

### 1.3 信息流/架构图

```
                  ┌──────────────────────────────────────────┐
                  │            Coding model (优化期)          │
                  │   读 trace → 诊断失败模式 → 写候选代码     │
                  └───────────────▲──────────────┬───────────┘
                                  │ traces       │ candidate harness
                                  │              ▼
┌──────────┐   观测+指令   ┌───────────────────────────────┐   action   ┌────────┐
│ 仿真环境  │ ────────────▶ │        Harness program        │ ─────────▶ │ 机器人 │
│ (6 个基准)│ ◀──────────── │  ├ 转发给 frozen VLA?          │            │ 控制器 │
└──────────┘   状态回读     │  ├ 调用视觉/检测/抓取工具?      │            └────────┘
                            │  ├ 直接发程序化轨迹?           │
                            │  └ 读写 persistent memory      │
                            └───────────────────────────────┘
                                  │
                          训练集重测：S_train 提升才接受（≤5 次）
                          最终在 held-out episodes 上评一次
```

---

## 2. 数学核心

📌 **Napkin Formula**（一行抓住本质）：

```
H* = argmax_H  S_train(H ; π_frozen)    s.t.  H_{k+1} = A(H_k, trace_k),  仅当 S_train(H_{k+1}) > S_train(H_k) 才接受,  k ≤ 5
```

论文没有给出形式化目标函数，这是我从 §3 的「优化循环 + 接受判据」反推的最简表述。逐项说明：

| 符号 | 含义 | 论文依据 |
|------|------|----------|
| H | harness 程序（一段可执行控制代码） | §3 Harness interface |
| π_frozen | 冻结 VLA 策略，权重全程不变 | §3、§4.1 |
| S_train | 训练 episodes 上的平均成功率（VLABench 为 progress score） | §4.1 Evaluation metrics |
| A | coding model 的「读 trace → 改代码」算子 | §3 Optimization loop |
| k | 迭代序号，0 = 裸策略，上限 5 | §4.1、§6 Limitations |

真正被报告的「增益」是两个比值：

```
G_scene = S_held(H_K) / S_held(H_0)      # unseen scenes 协议
G_task  = S_held(H_K) / S_held(H_0)      # unseen tasks 协议
```

**直觉**：这不是梯度下降，是**带拒绝的贪心爬山（greedy hill-climbing with rejection）**。它的理论保证只有一条——训练集上不退化；泛化完全靠「修订本身写的是几何与传感量，而不是物体名与坐标」这一结构性归纳偏置（作者在 §4.2 如此主张，属作者解释而非消融证明）。

**时序口径（通用工程约定，非论文公式）**，用于防止把不同概念混为一谈：

```
K 个均匀采样帧的首末跨度   = (K - 1) × Δt
从参考时刻 t 预测 t+1 … t+H 的终点距离 = H × Δt
```

> 重要边界：本文的 persistent memory 是**跨 episode 的数据结构**，不是未来动作 chunk，也不是观测历史窗口。论文没有报告任何 action horizon、观测历史长度或 chunk 长度参数（仅 Appendix A.1 提到 Google robot 平台为 3 Hz）。因此不能把任何单一设置的步长写成「所有实验共有的记忆窗口」——这两件事在本文里没有建立过等式。

---

## 3. 带数字走一遍：玩具例子

### 3.1 真实数据复算：Table 1 的 Mean 行能不能对上

这不是假设，是对论文表格的验算（unseen scenes 列，10 个设置）：

```
Bare:      10.0 + 0.0 + 8.0 + 2.0 + 0.0 + 22.0 + 20.0 + 7.3 + 0.0 + 27.9 = 97.2
           97.2 / 10 = 9.72   ≈ 论文 9.7   ✓

Iterative: 32.0 + 26.0 + 24.0 + 40.0 + 38.0 + 55.3 + 49.3 + 52.6 + 87.5 + 27.8 = 432.5
           432.5 / 10 = 43.25 ≈ 论文 43.3   ✓

One-Shot:  12.0 + 2.0 + 4.0 + 2.0 + 0.0 + 20.0 + 34.3 + 32.8 + 77.1 + 29.3 = 213.5
           213.5 / 10 = 21.35 ≈ 论文 21.4   ✓
```

unseen tasks 同样可复算：Bare 合计 127.2 → 12.72 ≈ 12.7；Iterative 合计 402.2 → 40.22 ≈ 40.2。

但倍率对不齐：

```
43.3 / 9.7  = 4.464  → +346%   论文写 +345%
40.2 / 12.7 = 3.165  → +217%   论文写 +216%
```

差异在 1 个百分点量级，最可能是作者用未舍入的原始均值算倍率、再用两位小数展示均值。**结论：Table 1 内部自洽，倍率标注与表格两位小数存在可解释的舍入差**，不构成矛盾，但也提醒读者别把 +345% 当成精确值。

另一个必须点出的完整性问题：Table 1 的 unseen tasks 列，One-Shot 均值是「–」，因为 VLABench 与 SimplerEnv WidowX 在该协议下首次修订无法评估（Fig. 11 用叉号表示）。所以「one-shot 在 unseen tasks 上平均多少」在本文中**没有答案**。

### 3.2 假设例子：接受判据的抗噪能力（标记为假设，含验算）

> 以下为我的条件性分析，不是论文实验。假设：训练集 50 个 episode（= LIBERO-Pro-10 的 5 个优化任务 × 每任务 10 trials，由 Table 4 推得），接受判据只看点估计成功率。

```
当前 harness: 15/50 = 30%
候选 harness: 22/50 = 44%   → 提升 7 个 episode，被接受

二项噪声估算（假设 episode 独立同分布）：
σ = sqrt(p(1-p)/n) = sqrt(0.30 × 0.70 / 50) = sqrt(0.0042) ≈ 0.0648
即约 3.24 个 episode 的标准差

7 / 3.24 ≈ 2.16σ  →  单侧概率约 1.5%
```

含义：单次 +14 个百分点的跳变不太可能是纯噪声；但若某次迭代只提升 2–3 个百分点（约 1 个 episode 量级），在 n=50 下完全落在噪声内，却仍会被「严格大于」判据接受，并被写进训练集单调曲线。论文没有报告任何重复运行或置信区间，因此**「5 次迭代的累计增益」里有多少来自噪声级接受，无法从现有材料判断**。这是我读 Table 1 与 Fig. 10–12 时最想追问的一点。

### 3.3 假设例子：memory 读取量的量级（标记为假设，含验算）

> 假设：只看 SimplerEnv WidowX 的 unseen scenes 协议，2 个 held-out 任务 × 24 trials = 48 个 episode；沿用论文 Table 3 的 3.5 reads/episode、96% actionable。

```
总读取次数 ≈ 48 × 3.5 = 168 次
可命中次数 ≈ 168 × 0.96 ≈ 161 次
```

注意语义边界：「actionable」在论文中的定义是「这次读取找到了已存条目」（Table 3 表注），**不等于该条目导向了正确动作**。所以 96% 是「缓存命中率」，不是「记忆有效率」。这一点论文表述清楚，但很容易在二手转述中被放大成「记忆 96% 有用」。

---

## 4. 工程视角

**成本侧（论文实测，Appendix B）**：完整 5 次迭代「不到 6 小时 coding-model 时间、平均低于 $150」，不采集示范、不更新权重。相对「为新机器人采数据 + 微调」的传统路径，这是本文最硬的工程论据。

**真正的开销在别处**：

| 开销项 | 论文是否量化 | 工程含义 |
|--------|--------------|----------|
| Trace 存储（逐步相机帧 + 深度 + 全部工具输出） | 否 | 长 episode 下这是带宽与磁盘大头，不是算力大头 |
| Trace 转上下文的 token 量 | 否 | 决定 coding model 能否「看完」整段 rollout；论文只说人工难以穷尽检查长录像 |
| 辅助工具推理延迟 | 否 | 部署时 harness 每步可能串行调用检测/分割/指向/抓取生成，直接压低可用控制频率 |
| Coding model 调用 | 是（≤5 次迭代） | 单点故障：换一个 coding model 是否还成立，论文未做消融 |

**控制频率的教训（论文实测，Appendix A.1）**：早期 harness 用固定时间预算结束每段运动，在 3 Hz 的 Google robot 上预算到期时机械臂还在途中。该组修订除改成「按机械臂位置与目标的距离判断到达或停滞」外，还加入「沿测得表面法线从正面接近竖直面把手」；训练池成功率从 11.0% 升到 30.2% 是这组合并改动的结果。其中状态驱动终止条件的工程含义很直接：**harness 里任何写死的时间常数都是 embodiment 泄漏**，跨本体部署必须换成状态驱动的终止条件。

**部署形态**：演化结束后，harness 就是一段普通控制程序，读标准输入（相机、深度、gripper pose/status）。作者据此主张接口「应该」能迁移到真机（§6），但这是主张，不是实验。

---

## 5. 数据与评测

### 5.1 基准与策略（逐项归属，不做合并）

论文的 10 个独立设置各自绑定一个 (benchmark, policy) 对，不能把某个设置的属性推广到全部：

| Benchmark | 机器人 / 特点 | 每任务 trials | 在该基准上被测的策略 |
|-----------|---------------|---------------|----------------------|
| LIBERO-Pro-10 | 交换物体初始位置的 LIBERO-Long，考空间记忆抗性 | 10 | MolmoAct-2、OpenVLA-7B、π0.5、RLinf-π0.5、SmolVLA |
| RoboCerebra | 短任务序列，每 150 个 control step 场景重置 | 10 | π0.5、SmolVLA |
| VLABench | 视觉相似干扰物中定位目标，用官方 progress score | 10 | X-VLA（通用 checkpoint）；另有 X-VLA 微调版作参照 |
| RoboCasa-365 | 移动机械臂 + 厨房场景 | 50 | RLDX-1 |
| SimplerEnv WidowX | WidowX，跨本体 + 视觉域偏移 | 24 | X-VLA（通用）；另有微调版参照 |
| SimplerEnv Google | Google mobile manipulator，仅用于统一 harness 测试台 | 未单列 | 通用 X-VLA / X-VLA 专家 |

**训练来源距离**分四级（§4.1）：in-distribution、swapped（LIBERO-Pro 式位置置换）、cross-benchmark（同仿真器同机器人但换基准）、out-of-domain（只有通用机器人预训练）。这个分级是理解全文数字的钥匙——增益最大的地方正是最右端。

### 5.2 评测协议

| 协议 | 训练/留出差异 | 考什么 |
|------|---------------|--------|
| Unseen scenes | 任务相同，初始场景布置不同 | 不背物体坐标 |
| Unseen tasks | 任务不同，但共享底层操作技能 | 技能级迁移，而非任务记忆 |
| Unified harness | 训练池为程序生成的 2201 个 episode；评测用各基准自己的 2160 个标准 episode | 跨本体共享机制 |

**指标口径必须说清**：VLABench 用的是官方 progress score（抓对物体、满足各目标条件都给部分分），其余基准是成功率；论文为简化把所有数字统称 success rate（§4.1 明说）。因此跨基准平均值是「混合口径均值」，不是同质成功率均值。

### 5.3 关键结果（论文实测）

| 设置 | Bare | One-Shot | Iterative | Δ |
|------|------|----------|-----------|---|
| 10 设置均值，unseen scenes | 9.7 | 21.4 | 43.3 | +345%（论文标注） |
| 10 设置均值，unseen tasks | 12.7 | –（不完整） | 40.2 | +216% |
| VLABench, X-VLA 通用，unseen scenes | 7.3 | 32.8 | 52.6 | +45.2 |
| 同设置，X-VLA 微调版参照 | 35.2 | – | – | 被 harness 超过 |
| SimplerEnv WidowX, X-VLA 通用，unseen scenes | 0.0 | 77.1 | 87.5 | +87.5 |
| RoboCasa-365, RLDX-1, unseen scenes | 27.9 | 29.3 | 27.8 | −0.2 |

统一 harness（Table 2）：通用 X-VLA 在四个基准上从接近 0 抬到 held-out 均值 28.8；但包在 X-VLA 专家上时，held-out 均值只从 45.9 到 47.0（+2%），且有单项负增益（SimplerEnv Google −1.0）。

---

## 6. 能力与失败模式

**能做什么（论文实测）**

- 在策略完全域外时把它「包」成可用专家：SimplerEnv WidowX 上通用 X-VLA 每步都输出全开夹爪，harness 学会**完全不再查询策略**，退化成纯 code-as-policy 视觉抓取 agent，训练集 0%→100%，held-out 87.5%/66.7%。
- 修具体物理失败：卡住时抬臂脱困、被隔板挡住时改水平接近、推门力度不够时重标定动作幅值、移动底盘漂移时按锚点召回（Fig. 3、Fig. 5）。
- 涌现多种记忆形态（Table 3，仅 5 个设置有条目）：X-VLA/VLABench 记「能抬起物体的手指深度」（3.4 次/集，98% 命中）；X-VLA/WidowX 记「按物体实测高度决定的闭合高度」（3.5 次/集，96%）；MolmoAct-2/LIBERO-Pro 记「成功抓取相对物体顶部的平均深度」（9.9 次/集，99%）；π0.5/RoboCerebra 记「150 步场景重置周期」（13.0 次/集，95%）；RLDX-1/RoboCasa 记「哪些指令需要夹住物体」（11.0 次/集，92%）。
- 跨本体自标定：统一 harness 通过开合空夹爪、在深度图里观察运动，把手指闭合方向写进记忆，用于后续抓取朝向（Fig. 4d）。

**不能做什么 / 明确的失败模式**

| 失败模式 | 证据 | 性质 |
|----------|------|------|
| 在已很强的 in-distribution 设置上，unseen scenes 协议下迭代后微降 | RoboCasa-365 RLDX-1，unseen scenes：27.9→27.8；同一设置在 unseen tasks 上为 73.5→77.5（+4.0） | 论文实测 |
| unseen tasks 上部分策略几乎无提升 | π0.5 on LIBERO-Pro：0.0→2.5；RLinf-π0.5：0.0→10.0 | 论文实测 |
| 迭代反而伤 held-out | 统一 harness + 通用 X-VLA，SimplerEnv WidowX held-out：one-shot 45.8 → iterative 24.0 | 论文实测，正文未解释 |
| 包在强专家上几乎不动 | X-VLA 专家 held-out 均值 45.9→47.0，含 −1.0 单项 | 论文实测 |
| 全仿真，无真机 | §6 明确列为 future work | 作者自述 |
| 迭代预算固定 5 次，饱和点未知 | §6 Limitations | 作者自述 |
| 任务类型仍偏桌面/厨房操作，未覆盖更丰富接触动力学 | §6 将灵巧手内操作、工具使用列为未来扩展，并非明确排除；基准也含移动底盘、开关柜门、按按钮、倒酒等 | 作者自述/论文实测 |
| 感知上限会锁死 harness | VLABench 麻将牌面无法可靠区分，只能退化成候选列表逐个试（Fig. 6g） | 论文实测 |

### 6.1 隐含假设

1. **训练集与 held-out 足够接近**：unseen tasks 也要求「共享底层操作技能」（§4.1）。若新任务需要全新技能，这套流程的迁移性未经检验。
2. **coding model 能消化完整 trace**：论文未报告 trace 长度、截断策略或 token 预算。若真实部署的 episode 长达数千步，这套循环是否还成立是空白。
3. **辅助工具在目标域可用**：检测、分割、指向、抓取生成的质量是 harness 的能力上限。VLABench 的麻将案例已经证明，工具不行时 harness 只能绕，不能修。
4. **每步都有完整标准观测**：相机、深度、gripper pose 与 status 齐备。真机上缺任一通道，harness 的几何例程就得重写。
5. **单次运行即可代表性能**：全部数字为单点，无 seed 数、无方差、无置信区间。这在「用点估计做接受判据」的方法里是结构性风险，不只是报告习惯问题。
6. **单一 coding model 足够**：全部实验用 claude-opus-5（Claude Code 客户端，maximum effort）。换模型是否复现，论文未做任何消融。

---

## 7. 与相关工作对比

| 方法 | 是否更新权重 | harness 由谁设计 | 是否含持久记忆 | 与本文的关键差异 |
|------|--------------|------------------|----------------|------------------|
| 传统微调（OpenVLA、π0.5 等） | 是 | 不涉及 | 无 | 需为新任务/新硬件采示范 |
| Harness VLA、VoLo、AGM 等 | 否 | **人工手写** | Harness VLA、AGM 等有且结构预定义；VoLo 未明确持久记忆 | 本文把「设计」本身自动化，记忆结构也交给搜索 |
| CaP-X / ASPIRE / RHO（code-as-policy 系） | 否 | 自动或半自动 | 部分有技能库 | 这些方法用代码**替换**策略；本文让代码自己决定何时用策略、何时接管 |
| Eureka / ENPIRE / EXIMO | 是（间接） | 自动 | 无 | 自动化的产物是奖励函数/训练配置/数据，最终仍落到权重 |
| **RoboArchitect（本文）** | **否** | **coding model 迭代搜索** | **有，结构自由涌现** | 唯一同时做到「不更新权重 + 自动设计 + 记忆自由演化」的组合 |

**面试 Tip**：被问到这篇时，一句话答——「它把 harness 当搜索空间、用 coding model 在 rollout trace 上做带拒绝的贪心爬山，核心数字是 10 设置均值 9.7%→43.3%；但全部在仿真、单一 coding model、单点无方差，且迭代在个别 held-out 设置上会反噬。」

---

## 8. 精读建议

- **值得精读原文的人**：
  1. 做 agentic robot / 冻结策略 test-time adaptation 的研究者——§3 的优化循环与 §4.4 的记忆涌现是可直接借用的设计模板；
  2. 要把某个通用 VLA 快速推到新机器人平台上的工程师——Appendix B 的成本数字（<6 小时、<$150）与 Appendix A 的例程清单最具参考价值；
  3. 研究「评测协议如何防止 harness 过拟合」的人——§4.1 的双协议划分与训练来源距离分级值得单独看。

- **建议章节路径**：先读 §3 建立机制直觉 → 再看 §4.1 的协议定义（否则 Table 1 的数字会被误读）→ 然后 §4.2 + Table 1 看主结果 → 最后 Appendix A.1/A.2 看具体例程与记忆条目。§2 Related Work 可快读，§5 结论与摘要高度重复。

- **不值得精读的理由**：如果你不做机器人学习、或已经熟悉「LLM 自动改代码 + 执行反馈」这套自动研究范式，读摘要与 Table 1 即可；本文在方法论上属于该范式在机器人 harness 上的一次系统性应用，而非提出全新优化算法。

---

[← Back to Theory](./README.md)

**关键引用**：[RoboArchitect, arXiv:2610.42141](https://arxiv.org/abs/2610.42141)（本文所有数字、协议与例程均出自该版本正文的 §3、§4、Appendix A、Appendix B 及 Table 1–4、Fig. 3–12）

> 待补/待证：论文正文提到作者提供 codebase，但本次记录未核验其可访问性与开放范围；模型权重、rollout 数据、演化出的 harness 代码是否公开，论文未逐项说明。