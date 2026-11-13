# RoboACE：机器人自改进的对抗式协同进化 (RoboACE: Adversarial Co-Evolution for Robot Self-Improvement)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-11-13
>
> **论文**: RoboACE: Adversarial Co-Evolution for Robot Self-Improvement
> **链接**: https://arxiv.org/abs/2611.13034
> **正文来源**: arXiv:2611.13034v1 [cs.RO]，09 Nov 2026（HTML/PDF 正文，作者 Delin Zhao、Zhengrong Yue、Junlin He、Yali Wang，MMLab @ SIAT）
> **摘要一致性说明**: 候选元数据摘录与 PDF 正文 Abstract 内容一致（均为 "...around frozen foundation models..."），未出现需以摘要替代正文主张的冲突；本文所有证据均取自 PDF 正文，不用候选摘要补写正文未覆盖的内容。
> **核心定位**: 在冻结基础模型（GPT-6.1 Terra / SAM 3 / π0.5）的前提下，让"练习怎么选"的 curriculum harness 和"任务怎么做"的 execution harness 交替互改，在 LIBERO-family 四个 200 条件面板上超过所有被评估系统；LIBERO-PRO 达 74.0%，比初始部署高 15.0 个百分点。

---

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | 把"选练习的程序"本身也当作可修订的软件对象，与执行侧修订交替进行，就能在冻结 VLA 权重下持续把已解决的弱点替换成新弱点 |
| 適合精讀 | 在做具身 Agent harness、课程学习、执行反馈闭环、VLA 上层编排的人；重点看 §3.1–§3.4 的交替协议与 §4.4 的三个案例 |
| 可以跳過 | 只关心 VLA 模型结构、动作 tokenizer、预训练数据配比的人，这篇不涉及权重层面的方法 |
| 落地可行性 | 中（仿真内可复制；真实机器人未验证，且依赖一个可稳定调用的强 LLM 优化器做 harness 修订）|
| 主要風險 | 单一开发轨迹 + 只在 LIBERO 生态评测，跨环境/跨本体/跨基线的可迁移性尚未建立 |

💡 **X-Ray 开场**

这篇论文解决的是"自改进闭环里被漏掉的那一半"：以往系统（RoboRSI、Zetta、DynaHarness、RAX、RPG）主要让执行流程可修订——技能、提示词、恢复策略；但固定练习源会留下未被探测的弱点，本文因此把“下一步练什么”的选题程序也列为可修订对象。RoboACE 的做法是让 curriculum harness 和 execution harness 交替做 Stable→Candidate 修订，并且用"固定快照"把两侧的比较条件锁死，使"对抗"可判分。对 VLA 研究者意味着：在基座权重冻结的前提下，上层 harness 的软件演化本身就能带来两位数百分点的评测差异，值得把它当成一个独立的研究对象而不是工程补丁。

📍 **研究全景时间线**

```
2019 POET（环境与解共同进化）
   → 2021 Asymmetric Self-Play（proposer/solver 角色分离）
      → 2023 Code as Policies / LIBERO
         → 2024 Voyager（自动课程 + 技能库）、Eureka
            → 2025 OMNI-EPIC、Eurekaverse、OpenVLA / FAST / π0.5
               → 2026 RoboACE（curriculum 与 execution 双侧 harness 共同修订）← 当前位置
```

局限：本文的节点都是被正文引用且年份明确的；更早的具身 Agent 文献谱系本文没有展开，故不在此罗列。

---

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

RoboACE 本身不是模型，而是一套**两侧 harness + 外部优化器 + 低频顾问**的组织方式。下表把它的部件职责拆开（来源：正文 §3.1–§3.5、附录 A.1–A.3）。

| 部件 | 输入 | 输出 | 频率/时序 | 是否训练权重 |
|------|------|------|-----------|--------------|
| Curriculum Harness | 上一轮 Capability Profile `P_k`、任务统计、可用任务/资产描述 | 保留/替换/新增练习条件 | 每个 curriculum phase 运行一次，执行侧冻结 | 否（只改提示词、技能描述、工具、接口、可读记忆） |
| Admission + Judge | 构造世界提案 | 通过/退回修订 | 每个候选条件一次 | 否 |
| Benchmark Curator | 参考覆盖要求、核心锚点 | 内部基准条件 | 每次 handoff 前 | 否 |
| Execution Harness | 固定快照 `T_k` | 任务级结果、轨迹、工具反馈 | 每个 execution phase 在 `T_k` 上跑 Stable 与 Candidate | 否 |
| Execution Optimizer | 失败诊断（技能失败 vs 技能描述误导 vs 工具选择不当） | 采纳/保留决定 | 每个 Candidate 一组独立 run | 否 |
| Experience Consolidator | 允许外泄的任务级结果与教训 | `P_{k+1}` | 每轮一次 | 否 |
| Strategic Advisor | 请求方 optimizer 的许可证据 + 修订历史 | 调查方向建议（不装修订） | 低频、按需 | 否 |

**关键边界（逐项归属，勿合并）**：

- 冻结的是**基础模型权重**：GPT-6.1 (Terra)、SAM 3、π0.5；论文明确"no LLM or VLA weights were trained or fine-tuned by this project"。
- 会变的是**软件资产**：提示词、技能描述、工具、接口、agent 可读记忆；这些按 deployment 版本化，与 episode workspace 分离。
- 记忆机制属于 **RoboACE 自己的 harness**：附录 A.3 说每个技能主描述最多暴露 2500 字符，并支持检索额外的 task/skill manual。这条**不能**归给 RPent 或 VoLo——RPent 的记忆是提示式经验，本文评测时按其"无目标基准经验"变体运行；VoLo 的记忆体现在子目标分解、VLM 监控与 `replan_grasp` 上，二者机制不同。

### 1.2 关键机制 (Key Mechanism)

要点式说明"为什么这样设计"：

1. **为什么必须两侧都改**：正文用抽屉技能举例——单独成功不等于组合成功，练习必须探测组合；而选择这种练习需要读历史结果、识别前置条件、保留能暴露回归的条件。选题程序若不改，遇到已解决失败后就找不到新弱点了。
2. **为什么交替而不是同时**：对抗式生成挑战需要一个稳定的判分参考。执行侧冻结时，curriculum 才能被公平考核；任务快照冻结时，execution revision 才能被公平比较。
3. **为什么固定快照 `T_k` 而不是滚动任务集**：如果任务集随执行能力一起漂移，Candidate 的提升无法归因，采纳决定会退化成噪声。
4. **为什么要有 Strategic Advisor 但不给它装权限**：局部修订容易在邻近失败上打转；跨轮次的长程视角能识别"反复出现的恢复瓶颈"这类共性问题，但采纳责任必须留在本地 optimizer，否则破坏交替协议。
5. **为什么构造世界要通过 Admission + Judge**：编译/物理有效性/可达性检查能挡住不可执行的场景，但正文明确这些检查**不做**用新 VLA rollout 预筛难度，也**不保证**完整动态可执行性——这是设计上的显式取舍。

⚡ **Eureka Moment**：把"练习选择程序"和"执行程序"当成同一类可版本化的软件对象，用交替冻结 + 固定快照给对抗式协同进化造出一个可判分的擂台。

### 1.3 信息流/架构图 (Flow / Diagram)

```
                  ┌────────────────────────────────────────┐
                  │  Frozen Foundation Models               │
                  │  GPT-6.1 (Terra) / SAM 3 / π0.5          │
                  │  权重全程不训练、不微调                  │
                  └───────────────────┬────────────────────┘
                                      │ 被两侧 harness 调用
        ┌─────────────────────────────┴─────────────────────────────┐
        │                                                           │
┌───────▼──────────────┐                              ┌──────────────▼─────────┐
│ Curriculum Harness    │   ── T_k (fixed snapshot) ──▶│ Execution Harness       │
│ 读 P_k → 保留/替换/    │                              │ 规划 / 代码技能 / VLA   │
│ 新增练习条件           │◀──  P_{k+1} (consolidated) ──│ 在 T_k 上比 Stable vs   │
│ Admission + Judge     │                              │ Candidate                │
└───────┬──────────────┘                              └──────────────┬─────────┘
        │                                                             │
        └──────────────────────┬──────────────────────────────────────┘
                               │ 严格交替：一次只动一侧
                     ┌─────────▼──────────┐
                     │ Strategic Advisor   │  低频、按需、只提方向，不装修订
                     └────────────────────┘
```

轮次协议（对应附录 Table 3，四步）：

```
Step 1  curriculum agent 提案 + optimizer 测 harness 修订   参考：E_k 与 P_k 冻结
Step 2  校验、保留/替换条件、curate 参考覆盖                 产出：固定快照 T_k
Step 3  在 T_k 上评 incoming E 与候选修订（任务/初始化/设置/预算不变）
Step 4  采纳 Candidate 或保留 Stable；consolidate 允许证据    产出：E_{k+1} 与 P_{k+1}
```

---

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：

```
C_{k+1} ← Revise(C_k | E_k, P_k)     ; E_{k+1} ← Revise(E_k | T_k)
```

这篇论文**没有**给出可微目标函数、损失项或采样分布；它的"数学"是交替优化的过程形式与准入判据。以下是我按正文 §3.1–§3.4 与附录 A.1 整理的形式化，不是原文公式。

```
T_k       = Admit( Bench( Candidate | E_k, P_k ) )
E_{k+1}   = Select( CandidateExec | T_k )        # 同任务、同初始化、同设置、同预算
P_{k+1}   = Consolidate( Outcomes( E_{k+1} on T_k ) )
```

变量说明：

| 符号 | 含义 | 备注 |
|------|------|------|
| `k` | 轮次索引 | 论文报告保存了 27 条 round record |
| `C_k` | 第 k 轮的 curriculum harness | 含提示词、检索接口、选题逻辑 |
| `E_k` | 第 k 轮的 execution harness | 含规划、代码技能、VLA 切换接口 |
| `P_k` | Capability Profile | 总结已演示能力、适用教训、未解决缺口、不确定性 |
| `T_k` | 固定任务快照 | 由 admitted practice + internal benchmark 条件组成；执行阶段全程不变 |
| `Stable` / `Candidate` | 当前已装版本 / 待测修订 | 每阶段可保留 Stable |
| `Admit` | 编译、物理有效性、端点可达、指令-目标-场景一致性 | 不用新 VLA rollout 预筛难度 |
| `Consolidate` | 允许外泄的任务级结果与教训 → `P_{k+1}` | 原始图像、轨迹、模型/工具调用、实现差异留在执行侧 |

**直觉**：这是一个"交替固定变量"的坐标上升。交替保证了每一侧被评价时另一侧不动，于是改进可归因；固定快照 `T_k` 保证了题目集不被执行侧的进步所污染。代价是收敛慢、每轮必须重跑整套快照，且没有任何关于"一定会变好"的保证——正文自己就保留了"phase can retain its incoming Stable version"这条退路。

> TODO: 正文未给出 `Admit` 的量化判据阈值，也未给出 Candidate 采纳的加权规则（论文只说 optimizer"consider successes, regressions, reuse, and cost"）。这些细节在提供的文本内不可确认。

---

## 3. 带数字走一遍：玩具例子 (Worked Example)

**例子 A：把 LIBERO-PRO 的百分点差换算成成功题数（可验算）**

面板固定 200 条件（正文 Table 1、§4.2），成功率 = 原生判定成功的条件数 / 200。

```
初始 RoboACE:  59.0%  →  0.590 × 200 = 118 个成功
最终 RoboACE:  74.0%  →  0.740 × 200 = 148 个成功
差值 = 148 − 118 = 30 个成功 = 15.0 个百分点 × 200 / 100  ✓ 与正文一致
```

同理 LIBERO 面板：

```
63.5% → 127 个成功 ;  99.0% → 198 个成功 ; 差 71 个 = +35.5 个百分点  ✓
```

这验证了正文"improves by 35.5 percentage points on LIBERO"与"15.0 points on LIBERO-PRO"在 200 条件口径下是自洽的。

**例子 B：动作 chunk 的时间跨度（假设性标注，用于澄清"记忆窗口"误读）**

论文只说 π0.5 "executes continuous five-action chunks"，环境控制频率 20 Hz（附录 A.4、§4.2）。以下计算是**我加的假设性验算**，不是论文的公式：

```
Δt = 1/20 = 0.05 s
若一个 chunk 有 K = 5 个均匀间隔动作，则首末跨度 = (K−1)×Δt = 4 × 0.05 = 0.20 s
该 chunk 覆盖的物理时间 = 5 × 0.05 = 0.25 s（含末端动作自身的驻留）
每 episode 8000 步 → 8000 / 5 = 1600 个 chunk（若按 chunk 边界重规划）
```

**要点**：这 5 个动作是**未来预测动作**，不是历史观测帧；0.20 s / 0.25 s 是执行 chunk 的跨度，**不能**被当作历史上下文窗口或记忆窗口。论文没有报告任何采样帧数、历史帧跨度或记忆长度指标，任何"记忆窗口 = 5 帧"的说法都是对本文的误读。

---

## 4. 工程视角 (Engineering View)

| 维度 | 论文给出的约束 | 工程含义 |
|------|---------------|----------|
| 单 episode 预算 | 8,000 环境控制步 @ 20 Hz = 400 s 仿真时间（6 min 40 s） | 所有被评系统共用同一预算；超时按失败计入分母 |
| 策略调用粒度 | π0.5 连续 5 动作 chunk | 若按 chunk 重规划，单 episode 上界 1600 次策略前向 |
| 练习集规模 | 稳态 30 个练习条件 | curriculum agent 每轮在 30 个槽位上做保留/替换/新增 |
| 内部基准 | 最多 40 个条件 | 执行侧每轮要在这个快照上跑 Stable + 若干 Candidate |
| 开发历史 | 27 条 round record，累计 76 个不同原生 LIBERO 条件 | 条件池远大于单轮快照，说明"换题"是主要动作 |
| 可读记忆 | 每个技能主描述 ≤ 2500 字符 + 可检索 manual | 提示预算有限；诊断必须落到可暴露描述里才算"部署成功"（附录 A.3） |
| 计算开销 | 权重不训练 | 成本几乎全在 LLM 推理与仿真 rollout；每轮重跑整套快照是主要瓶颈 |

**Trade-off 语言**：

- **慢与可归因**：交替冻结把每轮变成"两次大规模评测"，比同时更新慢，但换来采纳决定的可解释性。工程上适合离线迭代，不适合在线适应。
- **固定快照与过时**：`T_k` 在 execution phase 内不变，意味着执行侧进步后可能已经"通关"了部分条件，但快照要到下一轮才更新；这是可判分性的代价。
- **提示预算与记忆**：2500 字符的技能描述上限说明 harness 不是无界增长的；诊断如果写不进暴露描述或可检索 manual，就不会影响后续 agent（正文明确警告"optimizer journal can contain a useful diagnosis without making it accessible"）。
- **部署约束**：所有持久资产按 deployment 版本化，与 episode workspace 分离；这意味着可以回滚到任意 Stable，但也意味着版本管理必须是一等公民。

---

## 5. 数据与评测 (Data & Eval)

### 5.1 三个层次先分开写

| 层次 | 本文的做法 |
|------|-----------|
| **预训练** | 本文不训练、不微调任何 LLM 或 VLA 权重。π0.5、SAM 3、GPT-6.1 都是现成模型；OpenVLA-OFT、π0-FAST、Cosmos Policy、GR00T N1.7 使用各自发布的 checkpoint |
| **仿真评测** | 四个固定面板各 200 条件，共 800 条件一次打分；LIBERO、LIBERO-Plus、LIBERO-PRO、LIBERO-X |
| **真实机器人** | 论文没有任何真实机器人实验；结论节明确把"physical robots"列为 next empirical step |

### 5.2 四个面板的设置（来源：Table 1、§4.1）

| 面板 | 覆盖 | 评测角色 | 开发期是否使用 |
|------|------|----------|----------------|
| LIBERO | 40 个任务 × 5 个保留初始化 | 迁移到保留条件 | 开发用 LIBERO 条件，但保留面板与审计过的开发历史**条件不相交** |
| LIBERO-Plus | 7 个扰动类别 | 迁移到开发来源之外 | 未使用 |
| LIBERO-PRO | 4 个可用扰动维度 | 迁移到开发来源之外 | 未使用 |
| LIBERO-X | 5 个挑战等级，含语言配对条件 | 迁移到开发来源之外 | 未使用 |

**必须标注的限制**：论文自己说 "These panels define the evaluation in this study and do not constitute the complete official benchmark protocols." 也就是说这些是项目自建的固定子集，不能等同于官方全量协议。

### 5.3 测量口径（§4.2、附录 A.5）

- 每个条件只取**一个指定观测**进主表；历史前缀观测保留但不做 best-of-N。
- 超时、基础设施错误、其它一切非成功结果**全部计入分母**并记为失败。
- 只有完整、独立审计过的面板才计分；部分面板从结果中排除。
- 成功率 = 确认的原生成功数 / 200。原生谓词判任务完成，所以"抓到了物体"这种局部成功仍可能整任务失败。

### 5.4 主结果（Table 2，成功率 %）

| System | LIBERO | LIBERO-Plus | LIBERO-PRO | LIBERO-X |
|--------|--------|-------------|------------|----------|
| π0.5 | 96.9 | 92.0 | 60.5 | 11.5 |
| OpenVLA-OFT | 96.8 | 75.5 | 51.0 | 2.5 |
| π0-FAST | 85.5 | 61.6 | 49.0 | 4.5 |
| Cosmos Policy | 98.5 | 88.5 | 52.5 | 2.5 |
| GR00T N1.7 | 97.5 | 89.0 | 48.0 | 2.5 |
| RPent | 92.0 | 84.0 | 32.5 | 12.5 |
| VoLo | 97.5 | 90.0 | 60.5 | 10.0 |
| RoboACE（before evolution） | 63.5 | 64.5 | 59.0 | 19.5 |
| RoboACE（after evolution） | **99.0** | **94.5** | **74.0** | **30.5** |

（表中加粗按各列最优标注；LIBERO 列最优为 RoboACE 最终部署的 99.0，Cosmos Policy 的 98.5 次之。）

**可核对的差值**：

```
vs 初始部署：LIBERO +35.5 / Plus +30.0 / PRO +15.0 / X +11.0（百分点）
vs π0.5 直控：LIBERO +2.1 / Plus +2.5 / PRO +13.5 / X +19.0
vs 最强 agent 基线（每列最佳）：LIBERO +1.5(VoLo 97.5) / Plus +4.5(VoLo 90.0)
                              / PRO +13.5(VoLo 60.5) / X +18.0(RPent 12.5)
```

⚠️ **记录一致性提示（我的核对，非论文主张）**：按 200 条件口径，96.9%、96.8%、61.6% 无法对应整数成功数（分别 ≈193.8、193.6、123.2），其余单元格都可整除。这可能是百分比保留一位小数时的舍入或不同分母造成的，正文未解释，**待作者澄清**；不影响 RoboACE 各单元格（99.0/94.5/74.0/30.5 全部可整除）。

### 5.5 基线身份与接口（附录 A.4，逐模型归属）

| 系统 | checkpoint/接口 | 动作与观测 | 上层编排 |
|------|-----------------|-----------|----------|
| π0.5 | 最终系统官方控制器 | 每预测块连续 5 动作 | 无 LLM、无其他技能 |
| OpenVLA-OFT | released combined spatial/object/goal/10 微调版 | 原生 8 动作 chunk；前视与腕部图像 + 本体感觉 | 无 |
| π0-FAST | 官方 `pi0_fast_libero` | 官方图像/状态/动作变换；从 10 动作 horizon 做 5 动作重规划 | 无 |
| Cosmos Policy | 发布模型 | 正文未在 A.4 展开 | 无 |
| GR00T N1.7 | NVIDIA 发布模型 | 正文未在 A.4 展开 | 无 |
| RPent | 中性提示、无目标经验检索 | 公开观测、无隐藏几何状态 | Terra/high + 官方 π0.5 控制 |
| VoLo | 部署版 adapter | 子目标分解 + VLM 监控 + `replan_grasp` | Terra/high + 官方 π0.5 控制 |
| RoboACE（初/终） | 同一 controller | 公开观测、无隐藏几何状态 | Terra/high agent + harness 修订 |

**这张表要按行读，不能把一行的属性推给另一行**：例如"8 动作 chunk"只属于 OpenVLA-OFT，"10 动作 horizon 重规划出 5 动作"只属于 π0-FAST，"每个技能 2500 字符描述 + 可检索 manual"只属于 RoboACE。

### 5.6 案例证据（§4.4，三个 case，均为定性 + 小样本计数）

| 案例 | 起点 | 修订内容 | 记录到的后续 |
|------|------|----------|--------------|
| Curriculum revision | 一个可选的紧凑证据读取器最初未被使用 | 改提示词与检索接口，加入选择性阅读、覆盖检查、截断后定向恢复 | 后续一次独立运行中 agent 使用了该读取器，产出合法 30 条件课程，含 2 个它自己选的替换 |
| Execution revision（抽屉接地） | 独立抽屉技能组合不可靠 | 区分"前面板定位种子"与"附接把手"，动作前必须解析把手并验证目标抽屉 | 在同一个 54 条件比较快照上，成功数 48 → 50 |
| Strategic review | 抓取、抽屉恢复、运动接口多次局部修订后，恢复仍是共同瓶颈 | 长程审查提出"何时继续 VLA、何时切代码技能、何时介入"的问题；要求保留任务意图、针对具体控制器限制、检查 handoff 前置条件 | Advisor 只提方向；optimizer 测试并决定采纳 |

注意案例没有报告 200 条件面板级增益，48→50 是 54 条件快照上的计数，不能直接外推成主表百分点。

---

## 6. 能力与失败模式 (Capabilities & Failure Modes)

**能做（有实测支撑）**：

- 在 LIBERO-family 四个自建固定面板上，最终冻结部署的分数在表 2 中被评估的所有系统里最高（99.0 / 94.5 / 74.0 / 30.5）。
- 相对初始部署的净提升在四个面板上都是正的，且 LIBERO / LIBERO-Plus 上的提升（+35.5 / +30.0）说明 harness 修订的累积效应很大。
- 案例层面能看到具体的、可复用的修订：证据读取接口、抽屉把手接地、控制器切换规则。

**不能做 / 未验证（必须说清）**：

- **真实机器人零实验**。所有结论只在仿真中成立；结论节自己把 physical robots 列为下一步。
- **单一开发轨迹**。正文明确 "The results characterize one development trajectory and its resulting deployments."，没有多 seed、多 campaign 的方差估计，也没有跨次复现。
- **非官方全量协议**。四面板是自建 200 条件子集，作者声明不构成完整官方协议。
- **跨生态不可知**。开发只在 LIBERO 生态内进行；LIBERO-Plus / PRO / X 在开发期未使用，但它们仍是 LIBERO-family，不能推出对其它仿真器或真实场景有效。
- **没有权重层面方法**。任何"这篇改进了 VLA 表示"的解读都是错的。
- **候选采纳无形式化保证**。交替协议保留 Stable 是唯一的安全阀，但没有收敛性或单调改进的证明。

### 6.1 隐含假设 (Hidden Assumptions)

1. **"更强的 harness 修订"不会在另一侧造成回归**：论文的采纳只看当前快照上的成功、回归、复用与成本，但没有跨轮次的最坏情况分析。
2. **Judge 的语义检查足以过滤坏任务**：正文承认构造任务通过编译/物理/可达/一致性检查后"do not guarantee full dynamic executability"，也没用 VLA rollout 预筛难度。也就是说，一个"检查通过但实际不可做"的条件有可能进入快照并污染执行侧的比较。
3. **固定 200 条件面板的分辨率足够**：在 200 条件下，1 个百分点 ≈ 2 个条件；LIBERO 上 RoboACE 与 VoLo 差 1.5 个百分点 ≈ 3 个条件，这种量级的差异对单次观测的稳定性没有报告。
4. **LLM 优化器产出的修订是"可部署的"**：正文承认存在"journal 里有诊断但 agent 拿不到"的情况，靠 optimizer 自查是否真被使用；这是一个依赖实现质量的软约束。
5. **能力画像 `P_k` 是充分统计量**：curriculum 只读 `P_k` 与任务统计，不读原始轨迹。如果 `P_k` 丢失了某个关键失败模式，选题就会漏。

---

## 7. 与相关工作对比 (Comparison)

| 系统/方法 | 关注点 | 被修订的对象 | 任务来源 | 适用场景 |
|-----------|--------|--------------|----------|----------|
| POET / Enhanced POET | 环境与其解共同进化 | 环境程序 + 解 | 生成式 | 开放式 RL，非机器人 |
| PAIRED | 用 regret 引导环境设计 | 环境 | 生成式 | 无监督环境设计 |
| Asymmetric Self-Play | proposer/solver 角色分离 | 目标分布 | 生成式 | 机器人目标发现 |
| Voyager | 自动课程 + 技能库 | 技能库 | 自动课程 | Minecraft 式开放世界 |
| RPG | 从离线数据构造练习，再在固定任务集上修执行 | 提示词与技能 | 离线数据构造后**固定** | 具身 Agent 自改进 |
| RoboRSI | 可复用技能改进 | 技能 | 固定 | 真实复杂环境自进化（本文沿用其执行基础设施） |
| Zetta | 运行时 critic 与恢复技能 | critic/恢复技能 | 固定 | 冻结策略周围 |
| DynaHarness | 契约 + 失败驱动修订连接语义与执行 | 接口/契约 | 固定 | 长程操作 |
| VoLo | 子目标分解 + VLM 监控 + 抓取重规划 | 编排 | 固定 | 开放词汇长程操作 |
| RPent | 提示式经验增强控制器 | 提示经验 | 固定/目标任务 | memory-guided agent |
| **RoboACE** | **curriculum harness 与 execution harness 双侧共同修订** | **两侧 harness（含选题程序）** | **随 curriculum 演化而变** | **冻结基础模型下的仿真自改进** |

**与最近邻的两条区别（面试常问）**：

- vs **RPG**：RPG 也会从数据构造练习，但构造完就固定任务集，只修执行；RoboACE 把"选与建练习的程序"也列入修订对象。
- vs **POET**：POET 演化的是环境本身与解；RoboACE 演化的是"围绕冻结模型组织挑战生产与机器人执行的软件"，任务库是 curriculum harness 演化的**输出**，不是被直接优化的环境参数。

**面试 Tip**：被问"这和自动课程学习有什么不同"时，答一句话——"自动课程改的是课程内容，RoboACE 改的是产生课程的那段程序；前者换题，后者换出题人。"

---

## 8. 精讀建議 (Reading Guide)

- **值得精讀原文的人**：
  1. 正在搭具身 Agent harness、需要判断"上层编排能带来多少评测差异"的研究者/工程师；
  2. 做课程学习或开放式环境设计、想看清楚"固定快照 + 交替冻结"这套可判分协议怎么落地的人；
  3. 要评估在冻结 π0.5 之上加一层 LLM 编排、在 LIBERO-family 上能拿到什么量级收益的工程团队。

- **建議章節路徑**：先读 §3.1–§3.4（交替协议与四个阶段的边界）→ 再看 §4.1–§4.3 与 Table 2（评测口径与主结果）→ 然后 §4.4 的三个案例（理解"什么算一次有效修订"）→ 附录 A.1–A.3（更新边界、准入判据、可部署记忆接口）。§2 Related Work 与 References 可跳读，本文的对比已在 §7 汇总。

- **不值得精讀的理由**：如果你不做机器人学习、或只关心 VLA 模型结构与预训练数据配比、或已经熟悉 POET/PAIRED 那套协同进化框架且不在意 harness 工程细节，读摘要 + Table 2 + §4.4 即可。

---

[← Back to Theory](./README.md)

**关键引用（仅本文链接）**

- 论文主页：https://arxiv.org/abs/2611.13034
- 正文 HTML：https://arxiv.org/html/2611.13034