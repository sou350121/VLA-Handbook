# AutoResearch：把 AI Agent 变成“小型研究组织”的最小实验框架 (AutoResearch: A Minimal Autonomous Research Org on a Single GPU)

> **发布时间**：2026-03（GitHub 仓库公开时间）  
> **项目名称**：`karpathy/autoresearch`  
> **核心定位**：不是再做一个更大的训练框架，而是把“AI 自动做研究”压缩成一个**极小但真实可跑**的闭环：`单 GPU`、`单可编辑文件`、`固定 5 分钟实验预算`、`自动 keep/discard`。  
> **一句话 takeaway**：AutoResearch 真正新鲜的地方，不是“让 agent 改代码”，而是把“研究组织本身”写进 `program.md`，让人类主要编排研究规则，agent 主要执行实验循环。  
> **主要来源**：GitHub 仓库 [`karpathy/autoresearch`](https://github.com/karpathy/autoresearch)，仓库主说明 [`README.md`](https://raw.githubusercontent.com/karpathy/autoresearch/master/README.md)，研究组织脚本 [`program.md`](https://raw.githubusercontent.com/karpathy/autoresearch/master/program.md)

很多人把 AutoResearch 看成“Karpathy 又做了一个 agent demo”。但如果只这么理解，就低估了它的价值。它其实在试一件更具体的事：**把研究过程从“人类直接改训练代码”改写成“人类写研究章程，agent 在章程内做连续实验”**。  

对 VLA/具身研究者来说，这不直接等于“能自动发 paper”，但它非常像一个可迁移的研究基础设施原型：当你的实验已经足够便宜、目标函数足够清晰、修改边界足够可控时，agent 确实可以开始承担“夜间迭代器”的角色。

## X-Ray（非本领域也能复述）
- AutoResearch 的目标不是让 AI 接管整个研究，而是给它一个**小而真**的训练问题，让它在一夜之间连续试很多小改动。  
- 这个 repo 故意只让 agent 改 `train.py`，不许动 `prepare.py`，也不许改评测规则；这样实验空间被压得足够小，结果才可比较。  
- 对 VLA 团队最大的启发不是“复制 nanochat”，而是学它的**闭环设计**：固定预算、固定评测、有限可编辑面、自动保留有效改动。  

## 📍 研究全景时间线

```text
人工研究时代
  └─ 人类读日志 -> 改代码 -> 跑实验 -> 比结果

脚本化搜索 / HPO
  └─ 搜超参为主，通常不改模型代码结构

Coding Agent 时代
  └─ agent 能改代码，但通常还是“人类逐轮指挥”

AutoResearch
  └─ 把研究循环写成 program.md：
     人类定义组织规则
     agent 在固定预算内反复试验、比较、保留改动

下一步可能的方向
  └─ 从单文件 LLM 训练 -> 多文件系统 -> 仿真策略 -> VLA / world model 小闭环
```

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 维度 | 传统“手动研究” | 普通超参搜索 | AutoResearch |
|---|---|---|---|
| 人类主要工作 | 直接改训练代码 | 设搜索空间与调度器 | 编写/迭代 `program.md` |
| agent/脚本角色 | 很弱或没有 | 跑预定义配置 | 直接改 `train.py` 并做 keep/discard |
| 可编辑范围 | 整个 repo | 通常只改 config | **只改一个文件：`train.py`** |
| 评测规则 | 研究者可随时改 | 通常固定 | **固定在 `prepare.py`，agent 禁止修改** |
| 实验预算 | 人决定 | 调度器决定 | **固定 5 分钟 wall clock** |
| 结果管理 | 人手判断 | 搜索器排名 | `results.tsv` + 分支前进/回退 |

### 1.2 ⚡ Eureka Moment（关键洞见）

**真正值得学的不是“让 agent 改代码”，而是把研究过程改写成一个有边界的自治系统：固定目标、固定评测、固定预算、极窄修改面。**

### 1.3 三文件分工：什么交给人，什么交给 agent

AutoResearch 把 repo 压到只剩三个关键文件：

| 文件 | 角色 | 谁负责 | 为什么重要 |
|---|---|---|---|
| `prepare.py` | 常量、数据准备、评测、dataloader | 近似“只读” | 锁死 ground truth metric，避免 agent 偷改规则 |
| `train.py` | 模型、优化器、训练循环 | agent | 让搜索空间集中在“真正影响结果”的地方 |
| `program.md` | 研究组织规则、实验循环、keep/discard 标准 | 人类 | 把“研究方法论”显式写成文本程序 |

这就是 AutoResearch 最像“研究组织原型”的地方：  
**人类不再直接写实验改动，而是写 agent 该如何做研究。**

### 1.4 信息流/架构图 (Flow / Diagram)

```text
Human
  -> edits program.md
  -> defines rules / scope / acceptance criteria

Agent
  -> reads README.md + prepare.py + train.py + program.md
  -> edits train.py
  -> runs uv run train.py
  -> reads val_bpb / peak_vram_mb
  -> logs results.tsv
  -> keep improvement or reset branch

Loop
  -> baseline
  -> experiment
  -> compare
  -> keep/discard
  -> repeat overnight
```

## 2. 数学核心：固定预算下它到底在优化什么？ (Math Core)

> Napkin Formula：AutoResearch 优化的不是“训练到最好”，而是“**在固定 5 分钟预算内**，哪个 `train.py` 变体的 `val_bpb` 最低”。

### 2.1 目标函数

可以把它抽象成：

```text
given:
  fixed evaluator E in prepare.py
  fixed wall-clock budget T = 5 minutes
  editable space C = all valid edits to train.py

find:
  c* = argmin_c E(train.py = c, budget = T)

metric:
  val_bpb
  lower is better
```

这里最关键的不是公式本身，而是三个“锁死”的条件：

- **评测函数固定**：`evaluate_bpb` 在 `prepare.py`，agent 不能改。  
- **时间预算固定**：每次都是 5 分钟，所以不同模型大小/批量/结构改动还能在同一口径下比较。  
- **修改范围固定**：agent 只能改 `train.py`，避免到处“打补丁刷分”。  

### 2.2 为什么 `val_bpb` 很适合做自治目标？

README 明确强调：`val_bpb`（validation bits per byte）是**vocab-size-independent** 的，因此不同 tokenizer/词表规模变化下也更方便比较。  

这意味着 AutoResearch 不是只优化“某个特定实现的 loss 数字”，而是在尝试维持一个更稳定的跨变体标尺。

### 2.3 keep / discard 的真正逻辑

`program.md` 给出的标准并不是“只要更低就全留”，而是：

```text
if val_bpb improves:
  keep the commit
else:
  reset back

plus:
  simplicity matters
  tiny gains with ugly complexity may not be worth keeping
```

所以它隐含优化的是：

```text
research value = metric gain - complexity cost
```

虽然复杂度惩罚没有被形式化成代码，但它已经被写成了 agent 的行为准则。  
这点很重要，因为真正的研究组织从来不只追求“数值最优”，还追求“可理解、可维护、可继续叠代”。

## 3. 带数字走一遍：一轮实验是怎么闭环的？ (Worked Example)

### 3.1 第一步：先跑 baseline

`program.md` 明确要求：**第一轮必须先跑原始版本**，建立 baseline。

训练完成后，脚本会打印类似：

```text
val_bpb:          0.997900
training_seconds: 300.1
peak_vram_mb:     45060.2
num_steps:        953
num_params_M:     50.3
depth:            8
```

这一步的作用不是“证明 baseline 很强”，而是给后续所有实验建立比较基线。

### 3.2 第二步：agent 提一个小改动

例如 README / `program.md` 里的示意结果：

```text
baseline:
  a1b2c3d  0.997900  44.0  keep     baseline

experiment:
  b2c3d4e  0.993200  44.2  keep     increase LR to 0.04

failed idea:
  c3d4e5f  1.005000  44.0  discard  switch to GeLU activation

crash:
  d4e5f6g  0.000000   0.0  crash    double model width (OOM)
```

虽然这些数字是说明性示例，不一定代表当前 repo 的真实最佳结果，但它把整个自治闭环讲得非常清楚：

- 改动代码  
- 运行 5 分钟  
- 读出核心指标  
- 记录  
- 留下有效分支，丢弃无效分支  

### 3.3 第三步：为什么 branch 前进 / 回退设计很重要？

`program.md` 的 loop 本质上是：

```text
start from current best commit
  -> try one idea
  -> if better: advance branch
  -> else: reset
```

这使整个分支天然像一条“研究进化链”。  
你第二天醒来时，不只是看到一堆日志，而是看到一条**沿着局部最优不断前进的 commit 历史**。

## 4. 工程视角：为什么这个 repo 很小，却很“硬”？ (Engineering View)

### 4.1 单文件可编辑，是为了控制搜索空间

AutoResearch 最聪明的地方之一，是故意让 agent **只改 `train.py`**。  

工程意义：

- diff 足够小，人类第二天容易 review  
- 错误定位更简单  
- 避免 agent 通过乱改数据/评测/依赖“投机取巧”  
- 可以把“研究创新”更明确地归因到模型/优化器/训练循环本身  

### 4.2 固定 5 分钟预算，是为了让实验真的能跑整夜

README 明确写了一个很朴素的工程假设：

- 每次约 5 分钟  
- 每小时大约 12 次实验  
- 一晚可能做到 100 次左右  

这和很多“大而慢”的研究工作流完全不同。  
它押注的是：**便宜、快速、可重复的小实验，比少量昂贵实验更适合 agent 自治。**

### 4.3 `program.md` 其实是“研究组织代码”

README 直接说，`program.md` 本质上是一个超轻量的 “skill”。  

这句话的含义很深：

- 人类写的不是实验结果，而是“组织规则”  
- agent 执行的不是一条命令，而是一个长期循环  
- 研究方法本身被文本化、可版本化、可迭代  

如果把这套思路迁移到 VLA，真正该抽象出来的不是某个具体策略网络，而是：

- 哪些文件可改  
- 哪些评测不可改  
- 一轮 rollout / training / eval 的预算是多少  
- 什么样的改动算“值得保留”  

### 4.4 为什么它目前还不是“通用自动科研系统”？

因为它故意回避了很多困难问题：

- 多文件依赖爆炸  
- 分布式训练  
- 长实验周期  
- 多目标 trade-off  
- 模糊评测  
- 真机安全风险  

也正因为如此，它才真的跑得起来。  

## 5. 数据与评测：它的 benchmark 为什么可信，又为什么有限？ (Data & Eval)

### 5.1 为什么说它的评测口径很干净？

因为 repo 把以下几件事固定住了：

- 数据准备在 `prepare.py`
- tokenizer 训练流程固定
- dataloader 固定
- `evaluate_bpb` 固定
- 时间预算固定

所以 agent 不太容易“动 benchmark 本身的手脚”。

### 5.2 但为什么说它又非常局部？

README 也明确承认：

- 结果**依赖具体计算平台**
- H100 上成立的最优解，不一定在别的 GPU 上仍然最优
- 固定 5 分钟意味着它更偏向“短时预算下的最优结构”，不一定等于长训最优

这就意味着它找到的是：

```text
best train.py for this hardware, this codebase, this 5-minute budget
```

而不是一个普适结论。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

### 6.1 它擅长什么？

- 小代码库  
- 单一核心指标  
- 快速实验  
- 低成本失败  
- 可以靠日志直接判断 keep/discard 的任务  

### 6.2 隐含假设 (Hidden Assumptions)

AutoResearch 真正依赖的前提有这些：

| 假设 | 含义 | 一旦不成立会怎样 |
|---|---|---|
| 任务足够小 | agent 能完整读懂 repo | 搜索空间失控，改动质量下降 |
| 指标足够单一 | 一个标量能代表“更好” | keep/discard 逻辑会变模糊 |
| 实验足够便宜 | crash / discard 的代价低 | agent 自治很快变得昂贵 |
| 评测足够稳定 | 同一 budget 下结果可比较 | 优劣判断会被噪声淹没 |
| 安全风险低 | 崩了最多浪费一轮训练 | 在真机系统里这就不成立 |

### 6.3 失败模式

1. **过拟合固定 5 分钟预算**  
   - 找到的是“短跑冠军”，不一定是长跑冠军。  

2. **过拟合当前硬件**  
   - H100 上最优，不代表在消费级 GPU 或不同 kernel 实现上仍然最优。  

3. **复杂度悄悄膨胀**  
   - 如果 agent 只追小数点收益，代码会越来越 hacky。  

4. **评价函数与真实目标脱节**  
   - `val_bpb` 下降，并不自动等于“样本质量一定更好”或“迁移能力更强”。  

5. **迁移到机器人系统时安全假设崩塌**  
   - 在文本模型训练里，crash 的代价是一次失败；在真机系统里，crash 可能意味着硬件风险。  

## 7. 与相关工作对比 (Comparison)

| 话题 | AutoResearch 的定位 | 对 VLA / 具身研究的启发 |
|---|---|---|
| 与普通 HPO 对比 | 不只扫超参，也改模型代码 | 可以尝试让 agent 改 policy head / loss / tokenizer，而不只调 config |
| 与一般 coding agent 对比 | 不等人逐轮下指令，按 `program.md` 长时间自治循环 | 重点不只是“会写代码”，而是“会持续跑研究闭环” |
| 与大规模自动科研愿景对比 | 极小、真实、单 GPU、单文件 | 说明“自治科研”应先在小闭环里成立，再谈大系统 |

**面试 Tip**：如果被问“AutoResearch 真正新在哪里？”，你可以答：**它不是证明 agent 会调参，而是把研究流程本身压缩成一个可自治的小组织：固定预算、固定评测、单文件改动、program.md 编程研究规则。**

## 8. 对 VLA Handbook 的实际意义：该学什么，不该误学什么？

### 8.1 值得学的

- **固定实验预算**：把 rollout / 训练 / eval 变成可比较单位  
- **冻结评测 harness**：不要让 agent 动 benchmark 本身  
- **缩窄可编辑面**：先只允许改 action head、loss、adapter，别让 agent 一上来改全栈  
- **明确 keep/discard 规则**：不然 agent 只是在“乱试”  

### 8.2 不该误学的

- 不要以为它已经等于“自动发明 VLA”  
- 不要把这种模式直接搬到真机 without safety shell  
- 不要忽略单一指标与固定预算带来的局部性  

更合理的迁移方式是：

```text
先在仿真 / 小模型 / 单策略文件里试 AutoResearch 式闭环
  -> 再扩到更复杂的 VLA 子模块
  -> 最后才考虑真机后训练与持续学习
```

## References

- GitHub 仓库：[karpathy/autoresearch](https://github.com/karpathy/autoresearch)  
- 仓库主说明（README）：[raw README.md](https://raw.githubusercontent.com/karpathy/autoresearch/master/README.md)  
- 研究组织脚本：[raw program.md](https://raw.githubusercontent.com/karpathy/autoresearch/master/program.md)  
- 上游最小训练栈背景：[karpathy/nanochat](https://github.com/karpathy/nanochat)  

---
[← Back to Theory](../README.md)
