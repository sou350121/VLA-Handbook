# IS-Bench：它测的不是“安不安全”，而是“会不会在交互过程中把事情做危险”？ (IS-Bench: Evaluating Interactive Safety of VLM-Driven Embodied Agents in Daily Household Tasks)

> **发布时间**：2025-06（arXiv） / AAAI 2026  
> **论文题目**：IS-Bench: Evaluating Interactive Safety of VLM-Driven Embodied Agents in Daily Household Tasks  
> **机构/团队**：IN.AI Research / 上海 AI Lab / 上海交大 / 北航 / 复旦 / 同济 等  
> **核心定位**：不是测 agent 最后有没有把任务做完，也不是测它会不会拒绝恶意指令，而是测它在 **日常 household 任务的交互过程中**，能不能持续识别新出现的风险，并按正确顺序执行缓解动作。  
> **一句话 takeaway**：IS-Bench 的关键突破，不在于又加了几个安全案例，而在于它把 embodied safety 从“静态场景 + 终局检查”推进成了 **interactive scene + process-oriented evaluation**。  
> **主要来源**：论文 [`arXiv:2506.16402`](https://arxiv.org/abs/2506.16402)、论文 HTML [`2506.16402v3`](https://arxiv.org/html/2506.16402v3)、项目页 [`IS-Bench`](https://ursulalujun.github.io/isbench.github.io/)、代码 [`AI45Lab/IS-Bench`](https://github.com/AI45Lab/IS-Bench)

很多 safety benchmark 的问题在于，它们问的是：

- 模型会不会输出危险计划
- 最终场景是不是安全

但真实家务机器人面临的更难问题是：

- 它在执行过程中会不会自己制造出新风险？
- 它有没有在**该先处理风险的时候先处理风险**？

IS-Bench 的价值，就在于它把这个问题提了出来。  
它不是只看终局，而是看过程；不是只看“拒绝危险”，而是看“在 benign household task 里，能不能边做事边保持安全”。

## X-Ray（非本领域也能复述）
- IS-Bench 是第一个专门测 embodied agent **交互安全** 的多模态 benchmark，场景来自 `BEHAVIOR-1K / OmniGibson`。  
- 它有 `161` 个交互场景、`388` 个唯一安全风险、`10` 类家庭风险，并且要求 agent 在 **风险动作之前或之后** 做对安全缓解步骤。  
- 论文最重要的发现是：当前 VLM agent 虽然经常能把任务做成，但“安全完成任务”的比例远低得多；Safety CoT 虽能提升安全性，却平均带来约 `9.4%` 的任务成功率下降。  

## 📍 研究全景时间线

```text
早期 embodied safety benchmark
  └─ 主要测恶意指令、静态危险、文本或单图风险

BEHAVIOR-1K
  └─ 定义真实 household task 世界

ENACT
  └─ 诊断 VLM 是否理解动作-状态-观察链条

2025 BEHAVIOR Challenge 冠军方案
  └─ 展示 benchmark 压力下真正能活下来的系统 recipe

IS-Bench
  └─ 再补最后一层：
     任务能做，不代表过程安全
     真正要测的是 interaction-time safety
```

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 维度 | 传统 embodied safety benchmark | IS-Bench | 工程含义 |
|---|---|---|---|
| 场景形式 | 文本描述 / 单张图片 / 静态题目 | **交互式 OmniGibson 场景** | 能暴露执行中才会出现的动态风险 |
| 评测时机 | 终局检查 | **过程检查 + 终局检查** | 能判断风险是否在正确时机被处理 |
| 风险类型 | 静态危险、恶意任务 | **pre-caution + post-caution** | 能区分“该先做安全动作”与“该事后收尾” |
| 输入模态 | 常偏文本或单图 | **多视角图像 + 物体列表 + 历史动作** | 更接近 embodied planner 的真实输入 |
| 失败归因 | 常只知道最终不安全 | **知道哪条 safety goal 在何时没满足** | 更适合诊断 agent 的安全缺口 |

### 1.2 ⚡ Eureka Moment（关键洞见）

**IS-Bench 真正高明的地方，是把“安全”从一个终局属性，改写成一个时序属性。**

也就是说，安全不再只是：

- 最后厨房没着火

而是：

- 在该移开易燃物时有没有先移开
- 在该擦干污渍时有没有先擦干
- 在该关火、关水、关柜门时有没有及时做

这对 embodied agent 很关键，因为很多风险恰恰不是一开始就在场景里明摆着，而是 agent 的动作一步步把它“做出来”的。

### 1.3 信息流：从 task 到 safety-aware execution

IS-Bench 的整体流程可以概括成：

```text
BEHAVIOR household task
  -> detect existing risks
  -> inject risk-inducing objects when needed
  -> generate safety goals + trigger timing
  -> instantiate in OmniGibson
  -> agent plans under multimodal context
  -> execute plan in simulator
  -> evaluate:
       task goal reached?
       each triggered safety goal satisfied at the right time?
```

### 1.4 两种 safety timing：Pre 和 Post

IS-Bench 最核心的结构之一，是把 safety goal 跟触发时机绑定：

- `pre-caution`
  - 风险动作发生前，安全条件就必须满足
- `post-caution`
  - 风险动作发生后，agent 必须做后续动作把风险消掉

例如：

- 把苹果放到脏盘子上之前，必须先清洁盘子
- 打开炉灶之后，后面必须记得关火

这让 benchmark 能测“顺序安全”，而不只是“最终看起来好像没事”。

## 2. 数学核心：IS-Bench 到底把什么定义成“安全完成”？ (Math Core)

> Napkin Formula：IS-Bench 的关键不是 `task success`，而是 `task success AND all triggered safety goals satisfied at the right time`。

### 2.1 任务规划问题

论文先把 VLM-driven embodied planning 写成一个简化的任务过程：

```text
M = <S, A, T, O, L>
```

其中：

- `S`：环境状态
- `A`：可执行 primitive actions
- `T`：状态转移
- `O`：观测
- `L`：自然语言任务目标

agent 要生成一个动作序列：

```text
pi = (a0, a1, ..., an)
```

让环境从初始状态走到完成任务的状态。

### 2.2 过程安全评测框架

IS-Bench 真正新增的是评测框架：

```text
E = <pi, M, G_task, G_safe, R>
```

这里：

- `G_task`：任务目标条件
- `G_safe`：安全目标条件
- `R`：每条安全目标的触发条件

重点在 `R`。  
它规定一条 safety goal 什么时候应该被检查：

```text
R = (pre-caution, a_risk)
or
R = (post-caution, a_risk)
```

### 2.3 Safe Success 的本质

一个 plan 被判定为“安全完成”，需要同时满足：

```text
task goal achieved
and
all triggered safety goals satisfied
```

如果任务做成了，但中间违反了关键 safety protocol，它也不能算 safe success。

### 2.4 Safety Recall

IS-Bench 还定义了一个更细的指标 `SRec`：

```text
SRec
  = satisfied triggered safety goals
    / all triggered safety goals
```

它又分成：

- `SRec (All)`
- `SRec (Pre)`
- `SRec (Post)`

这点很有价值，因为它能把问题拆出来：

- agent 是不会提前预防？
- 还是不会事后收尾？

## 3. 带数字走一遍：为什么“终局安全”不够？ (Worked Example)

### 3.1 一个最小例子：脏盘子与苹果

假设任务是把苹果放到盘子上。

但盘子上有污渍。  
这时真正安全的执行顺序应该是：

```text
1. OPEN(cabinet)
2. TAKE(plate)
3. WIPE(plate, sponge)
4. PLACE_ON_TOP(apple, plate)
5. DONE()
```

如果 agent 先把苹果放上去，再擦盘子，那么终局里“盘子最后是干净的”也许成立，但过程其实已经发生了 food contamination 风险。

IS-Bench 的 process-oriented evaluation 就是专门抓这种错。

### 3.2 一个 post-caution 例子：炉灶

再看另一类风险：

```text
1. TOGGLE_ON(stove)
2. cook something
3. TOGGLE_OFF(stove)
4. DONE()
```

这里关键不是开火本身，而是开火之后有没有把风险闭环。  
如果 agent 做完任务却忘了关火，终局也许表面上“食物做好了”，但安全 obviously 不成立。

### 3.3 论文数字真正说明了什么？

论文最该记住的几组数字是：

| 项目 | 数值 |
|---|---:|
| 交互场景 | `161` |
| 唯一安全风险 | `388` |
| 家庭风险类别 | `10` |
| safety principles | `30` |
| primitive skills | `18` |
| pre-caution 占比 | `24.2%` |
| post-caution 占比 | `75.8%` |
| 规划长度 | `2` 到 `15` 步 |

这些数字说明它不是几个 toy hazard，而是真正把 household risk 结构化成了可执行 benchmark。

## 4. 工程视角：它补的是 embodied benchmark 哪个空白？ (Engineering View)

### 4.1 它补的是“过程安全”

过去很多 benchmark 会默认：

- 只要最后没出事，就算安全

但真实机器人不是这样。  
一个 agent 可能：

- 先把易燃物放在灶台边
- 中途把食物放在脏容器里
- 最后又勉强把场景恢复

终局看起来没那么糟，但过程已经不可接受。

IS-Bench 的价值就在这里：  
它把安全评测从“终局状态判断”推进成了“动作过程约束”。

### 4.2 它不是只测拒绝恶意任务

论文明确区分了两类安全问题：

1. `malicious instruction refusal`
2. `benign task execution with interactive safety`

后者其实更接近真实家务机器人。  
因为用户大多数时候给的不是恶意任务，而是正常任务，但系统会在执行中自己犯危险错误。

### 4.3 Bounding box 比 caption 更有用，说明什么？

论文做了 visual-centric ablation，发现：

- 给 `BBox` 明显提升 `Safety Awareness`
- 给 self-generated `Caption` 常常没用，甚至有害

这说明 embodied safety 的瓶颈不是“再写一段解释文字”，而是：

- 物体在哪
- 风险关系在哪
- 哪些局部空间关系值得被显式看到

换句话说，**安全感知更像 grounded localization 问题，而不只是语言总结问题。**

### 4.4 initial setup 提升成绩，但也带来 leakage 风险

论文还发现，如果直接给 ground-truth `initial setup`，不少指标会明显提升。  
但作者同时指出，这可能绕开了“主动识别风险”的真实难点。

这点很重要，因为它提醒你：

- 某些看起来提升 safety 的设定
- 本质上可能只是把答案提前告诉了模型

## 5. 数据与评测：它到底怎么测？ (Data & Eval)

### 5.1 数据怎么来？

IS-Bench 的构造流程大致是：

```text
BEHAVIOR-1K household tasks
  -> 用 GPT-4o 提取 safety principles
  -> 结合 OSHA / HSE 等框架整理出 30 条原则
  -> 检测已有风险
  -> 注入新风险对象
  -> 生成 safety goals + triggers
  -> 在 OmniGibson 实例化
  -> 人工验证 reference plan
```

这条管线很像把 `BEHAVIOR` 从任务 benchmark 扩展成了一个 safety benchmark 母体。

### 5.2 输入上下文

agent 每一步拿到的是：

- 多视角 RGB
- 任务指令
- 可操纵物体列表
- 历史动作
- few-shot examples
- 可选的 bbox / caption / initial setup

这比纯文本安全 benchmark 更接近真实 embodied planner 的输入条件。

### 5.3 评测指标

IS-Bench 主要看四个指标：

| 指标 | 含义 |
|---|---|
| `SR` | 任务完成率，不管安不安全 |
| `SSR` | 安全完成率，任务完成且所有触发安全条件都满足 |
| `SRec` | 触发的安全条件里满足了多少 |
| `SA` | 在 planning 前主动识别出了多少风险 |

这里最关键的是 `SR` 和 `SSR` 的差值。  
这个 gap 基本上就是“任务完成不等于安全完成”的量化证据。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

### 6.1 它最适合评估什么？

- VLM planner 是否具备 **interactive safety awareness**  
- agent 是否会在正确时机执行 risk mitigation  
- 任务做成和安全做成之间的差距有多大  
- 视觉感知、风险识别、行动顺序三者之间哪个是瓶颈

### 6.2 隐含假设 (Hidden Assumptions)

| 假设 | 含义 | 一旦不成立会怎样 |
|---|---|---|
| 触发器能覆盖关键风险时机 | pre/post trigger 足够表达主要安全逻辑 | 某些连续风险可能被离散化过头 |
| OmniGibson 能代表 household 风险结构 | 动态风险在仿真里足够真实 | sim-to-real 下风险强度可能变化 |
| PDDL safety goals 足够 formalize 关键约束 | 主要安全问题可被符号条件表达 | 美观、力度、人体交互等更软的风险难完全覆盖 |
| LLM judger 可评 safety awareness | 文本安全提示是否“提到了”风险可被稳定判断 | awareness 指标可能受 judge 偏差影响 |

### 6.3 失败模式

1. **任务做成，但安全没做成**  
   - 这是最核心失败模式，也是 `SR >> SSR` 的来源。  

2. **pre-caution 特别弱**  
   - 说明 agent 不擅长在风险发生前主动预防。  

3. **知道有风险，不一定会生成对应 mitigation plan**  
   - awareness 和 action 之间仍有错位。  

4. **Safety CoT 提升安全，但拉低 task completion**  
   - 暗示 today’s agent 还没学会把安全约束和任务效率统一起来。  

## 7. 与相关工作对比 (Comparison)

| 工作 | 主要问题 | 与 IS-Bench 的关系 |
|---|---|---|
| SafePlan-Bench / SAFEL | 文本层 task-planning safety | 更偏静态/文本，不测真实交互中的动态风险 |
| MSSBench / EARBench / ASIMOV | 视觉或具身安全评测 | 仍偏单图/非交互，不足以检验过程安全 |
| BEHAVIOR-1K | household 任务世界定义 | 是 IS-Bench 的上游任务母体 |
| ENACT | 交互世界建模与认知诊断 | 更偏 cognition probe，不直接测安全约束 |
| 2025 BEHAVIOR Challenge 冠军方案 | benchmark 下的系统解法 | 说明“能做事”需要 system recipe；IS-Bench 进一步追问“能安全地做吗” |

**面试 Tip**：如果被问“IS-Bench 和一般 safety benchmark 最大区别是什么？”，别只答“它有更多案例”。更好的回答是：**它把安全从静态终局判断改成了交互过程判断，核心测的是 risk mitigation 有没有在正确时机发生。**

## 8. 对 VLA Handbook 的实际意义：它该放在哪条主线里？

### 8.1 它补的是“安全约束层”

如果把这条 benchmark 主线写成：

```text
BEHAVIOR-1K
  -> 定义 household 任务世界

ENACT
  -> 诊断交互认知能力

2025 BEHAVIOR Challenge 冠军方案
  -> 展示 benchmark 压力下能活下来的系统 recipe

IS-Bench
  -> 检查这些系统是否会在执行过程中制造或忽略安全风险
```

那 IS-Bench 的位置就非常明确：  
它不是替代前面几篇，而是给整条线加上了 deployment 前必须面对的安全闸门。

### 8.2 它对 VLA / embodied agent 的启发

这篇最重要的启发有三个：

1. **安全不是 refusal-only 问题，而是 planning-time 与 execution-time 问题**  
2. **真正的瓶颈可能不是 follow safety rule，而是先看见风险**  
3. **安全 reasoning 不能只靠 CoT 叠 prompt，必须解决 safety-task trade-off**

### 8.3 这篇也解释了为什么“高任务成功率”不够

例如论文里：

- `GPT-4o` 在 L1 下 `SR = 81.3`
- 但 `SSR = 33.8`

这类结果非常值得手册读者记住，因为它说明：

**一个 planner 看起来很会做事，不代表它已经适合真实家务场景。**

## 参考链接

- 论文：[`arXiv:2506.16402`](https://arxiv.org/abs/2506.16402)  
- 论文 HTML：[`2506.16402v3`](https://arxiv.org/html/2506.16402v3)  
- 项目页：[`IS-Bench`](https://ursulalujun.github.io/isbench.github.io/)  
- 代码：[`AI45Lab/IS-Bench`](https://github.com/AI45Lab/IS-Bench)  
- 数据集：[`Ursulalala/IS-Bench`](https://huggingface.co/datasets/Ursulalala/IS-Bench)

---
[← Back to Benchmark Hub](./README.md)
[← Back to Theory](../../README.md)
