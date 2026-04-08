# BEHAVIOR-1K：为什么它不是“任务更多的 benchmark”，而是对通用机器人提出了更真实的要求？ (BEHAVIOR-1K: A Human-Centered Embodied AI Benchmark with OmniGibson)

> **发布时间**：2024-03（arXiv: 2403.09227）  
> **论文题目**：BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation  
> **机构/团队**：Stanford / Stanford HAI / Salesforce Research 等  
> **核心定位**：不是只把“任务数量”从 `100` 扩到 `1000`，而是把“人真正想让机器人做什么”作为 benchmark 的起点，并配套一个能模拟 **刚体 + 关节物体 + 流体 + 柔性物体 + 连续状态变化** 的仿真环境 `OmniGibson`。  
> **一句话 takeaway**：BEHAVIOR-1K 的真正意义，不在“任务很多”，而在它第一次比较系统地把 **人类需求、语义任务定义、复杂物理状态、真实仿真** 绑成了一个统一 benchmark。  
> **主要来源**：论文 HTML [`arXiv:2403.09227`](https://arxiv.org/html/2403.09227v1)，项目主页 [`behavior.stanford.edu`](https://behavior.stanford.edu)

很多 benchmark 的问题，不是“不够难”，而是**难得不够像真实世界**。  

有的 benchmark 很会考导航，有的很会考抓取，有的很会考整理，但往往默认了非常少的任务类型、很单一的场景、或者过于理想化的物理。BEHAVIOR-1K 想修正的，正是这件事：**如果你真的想做“通用家庭机器人”，你不能只让它会搬 3 个积木、开 1 个抽屉、清 1 张桌子。**

它的激进之处在于，它先去问人：**你最想让机器人替你做什么？**  
然后再围绕这些答案，构建任务、物体、场景、状态变化和仿真能力。  

这让它不是“研究者拍脑袋设计的 benchmark”，而更像一个“面向真实需求的任务世界”。

## X-Ray（非本领域也能复述）
- BEHAVIOR-1K 先做了一轮大规模问卷，问人们到底想让机器人做哪些日常任务，然后才决定 benchmark 里放什么任务。  
- 它不只给出 1000 个活动名字，还给出 **50 个场景、9000+ 物体、对象属性、状态变化、任务逻辑定义**，并用 `OmniGibson` 去真实模拟。  
- 最重要的结论是：即便给当前方法加了 action primitives 和运动规划，很多任务仍然非常难，这说明“真正的日常通用机器人”比常见 benchmark 难得多。  

## 📍 研究全景时间线

```text
早期 embodied benchmarks
  └─ 导航 / 单任务 / 少量场景 / 较弱物理 realism

BEHAVIOR-100
  └─ 开始把 household activities 写成 BDDL 逻辑定义

Habitat / ManiSkill / ALFRED / VirtualHome / RLBench
  └─ 各自强化某一类能力：
     导航、重排、指令执行、操作、程序化任务

BEHAVIOR-1K
  └─ 把 benchmark 的起点改成：
     “人真正想让机器人做什么？”
     并配套 OmniGibson 提供更接近真实家庭/商店/办公室任务的仿真

它对后续 VLA / world model / sim2real 的意义
  └─ 不再只问“模型能不能成功”
     而是问“它能不能在更长时程、更复杂物理、更真实需求下成立”
```

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 维度 | 常见 embodied benchmark | BEHAVIOR-1K | 工程含义 |
|---|---|---|---|
| 任务来源 | 研究者设计 | **基于 1461 人问卷的人类需求排序** | 任务更像现实“有用任务” |
| 任务规模 | 少量到中量 | **1000 activities** | 更接近“通用家务/服务机器人” |
| 场景类型 | 常常单一 | **50 scenes / 多类场景**：住宅、花园、办公室、商店、餐厅等 | 泛化压力更真实 |
| 物体多样性 | 通常有限 | **9000+ objects / 1900+ categories** | 长尾物体问题被显式引入 |
| 状态变化 | 常见是 rigid-body + 少量开关态 | **流体、柔性材料、布料、温度、湿度、脏污、开关等** | 任务不再只是“搬运 rigid body” |
| 任务表达 | 几何/脚本/奖励居多 | **BDDL 谓词逻辑 + 物体属性 + transition rules** | 任务定义更语义化且可检查 |

### 1.2 ⚡ Eureka Moment（关键洞见）

**BEHAVIOR-1K 最核心的创新不是“任务更多”，而是把“人类需要什么任务”和“仿真能否真实表达这些任务”当成一个联合设计问题。**

### 1.3 两大组成部分：Dataset + OmniGibson

论文非常明确地把系统拆成两部分：

```text
BEHAVIOR-1K Dataset
  - 1000 activities
  - BDDL activity definitions
  - 50 scenes
  - 9000+ object models
  - object properties / state transitions / knowledge base

OmniGibson
  - realistic rendering
  - rigid body simulation
  - deformable body / cloth / fluids
  - extended object states
  - logical predicate checking + sampling
  - transition machine
```

这个拆法非常重要，因为它说明 benchmark 不是“给你一堆任务名”，而是给你：

- 任务逻辑如何定义  
- 初始状态如何采样  
- 目标状态如何检查  
- 哪些复杂物理过程必须被模拟  

### 1.4 信息流/架构图 (Flow / Diagram)

```text
Human survey
  -> rank everyday activities by usefulness

Activity selection
  -> top 909 tasks from survey
  -> + 91 tasks from BEHAVIOR-100

Knowledge base construction
  -> objects
  -> object properties
  -> state transitions
  -> BDDL activity definitions

3D assets + scenes
  -> 50 scenes
  -> 9000+ objects

OmniGibson
  -> simulate initial states
  -> render observations
  -> evolve object states
  -> check goal predicates

Agent evaluation
  -> task success
  -> efficiency
  -> sim-real gap analysis
```

## 2. 数学核心：它到底如何把“家务活动”写成可执行 benchmark？ (Math Core)

> Napkin Formula：BEHAVIOR-1K 的本质不是“给一个奖励函数”，而是把任务写成 `initial predicates + goal predicates + object/state ontology`，再让仿真去判断什么叫“真的完成了任务”。

### 2.1 BDDL：把活动定义成逻辑约束

论文的关键设计之一是使用 **BEHAVIOR Domain Definition Language (BDDL)**。  

可以把一个任务抽象成：

```text
Task
  = {Objects, Initial Conditions, Goal Conditions}

success
  = whether current world state satisfies goal predicates
```

例如：

- 不是简单说“把杯子放好”
- 而是明确写成：
  - 哪些对象存在
  - 初始状态是什么
  - 目标状态需要满足哪些谓词

这让 benchmark 可以区分：

- 任务有没有完成  
- 哪种状态算合法解  
- 初始场景能否自动采样出有效实例  

### 2.2 为什么这比“直接给 reward”更强？

因为很多日常任务并不是一个简单几何目标。  
例如“烤饼干”“清理洗衣房”“清洁熨斗底部”这类任务，不只是空间位姿变化，还涉及：

- 是否煮熟 / 烧焦
- 是否被某种物质覆盖
- 是否装满 / 空了
- 是否 soaked / folded / open / toggled on

如果没有逻辑谓词和对象状态系统，你根本很难把这些任务写清楚。

### 2.3 BEHAVIOR-1K 真正引入了哪些“连续世界状态”？

OmniGibson 维护的 extended states 包括但不限于：

```text
Temperature
MaxTemperature
SoakedLevel
CoveredLevel
ToggledState
SlicedState
BrokenState
```

这意味着很多谓词并不是“硬编码标签”，而是由仿真中的物理/扩展状态推导出来。

例如：

```text
Cooked(o)
  depends on object's historical max temperature

Frozen(o)
  depends on temperature threshold

Soaked(o, liquid)
  depends on absorbed liquid particles

Filled(container, liquid)
  depends on liquid volume inside container
```

### 2.4 Transition Machine 为什么重要？

真实日常任务里，有很多物理过程不是现成引擎能完整模拟的。  

论文因此加入了一个 **Transition Machine**，用于处理：

- 混合食材变成新物质
- 特定清洁剂去除某类脏污
- 加热后某物变成另一物

可以理解成：

```text
if:
  required inputs + machine + conditions are satisfied
then:
  transform objects/states into new outputs
```

这使 benchmark 能覆盖“烹饪、清洁、加工”这类复杂日常活动，而不是只停留在 rigid rearrangement。

## 3. 带数字走一遍：这套 benchmark 到底大到什么程度？ (Worked Example)

### 3.1 从人类需求开始，而不是研究者想象

论文先做了一个问卷：

- **1461 位参与者**
- 每个活动 **50 个评分**
- 初始活动池约 **2090** 个

最后进入 BEHAVIOR-1K 的是：

- 问卷里排名最高的 **909** 个活动
- 加上 BEHAVIOR-100 中的 **91** 个活动

这使它的 1000 个任务不是“研究者觉得有趣的 1000 个任务”，而是更接近“人类真正想外包给机器人”的 1000 个任务。

### 3.2 benchmark 规模不是口号，而是结构性扩容

论文给出的关键规模：

| 维度 | 数量 |
|---|---:|
| Activities | 1000 |
| Scene types | 8 |
| Scenes | 50 |
| Object categories | 1900+ |
| Object models | 9000+ |
| Respondents in survey | 1461 |

相比 BEHAVIOR-100，它不只是任务数量翻 10 倍，连：

- 场景类型
- 物体长尾
- 可模拟状态变化

都一起放大了。

### 3.3 视觉 realism 也被显式测了

论文做了一个视觉真实性 AMT 研究，结果里：

| 环境 | 视觉 realism 分数 |
|---|---:|
| **OmniGibson** | **3.20 ± 1.23** |
| Habitat 2.0 | 1.74 ± 1.33 |
| AI2-THOR | 1.73 ± 1.37 |
| iGibson 2.0 | 1.69 ± 1.24 |
| ThreeDWorld | 1.65 ± 1.23 |

这并不意味着视觉 realism 已经“等同真实世界”，但至少说明作者不是只说“我们看起来很真”，而是试图用用户研究去验证。

## 4. 工程视角：为什么 current SOTA 还是做得很差？ (Engineering View)

### 4.1 论文故意选了三个“看起来没那么夸张”的任务

他们实验里重点分析的是三个任务：

- `CollectTrash`
- `StoreDecoration`
- `CleanTable`

分别覆盖：

- rigid body manipulation
- articulated object manipulation
- flexible material + fluid interaction

这三类其实已经足够代表“日常任务为什么难”。

### 4.2 直接 visuomotor RL 几乎完全失败

论文的 `RL-VMC`（图像到低层控制）在三项任务上：

| 方法 | StoreDecoration | CollectTrash | CleanTable |
|---|---:|---:|---:|
| RL-VMC | 0.0 | 0.0 | 0.0 |

这说明在这种长时程、稀疏奖励、多步依赖的任务里，**纯端到端视觉到低层控制** 几乎学不起来。

### 4.3 即使加入 action primitives，仍然不轻松

有了 action primitives 后，表现明显提升：

| 方法 | StoreDecoration | CollectTrash | CleanTable |
|---|---:|---:|---:|
| RL-Prim. | 0.48 | 0.42 | 0.77 |
| RL-Prim.Hist. | 0.55 | 0.63 | 0.88 |

这说明两件事：

1. **长时程任务确实需要 action abstraction**
2. **即使做了 abstraction，问题也远没有被解决**

特别是 `CollectTrash`，最短也需要至少 `16` 个 primitive step。  
这已经足够让 credit assignment、探索和历史记忆都变成大问题。

### 4.4 记忆为什么重要？

`RL-Prim.Hist.` 比 `RL-Prim.` 更强，论文明确指出原因之一是：

- 长时程任务里存在 observation aliasing
- 例如机器人看到垃圾桶时，仅凭当前视角，不知道哪些位置已经清理过

所以这篇论文虽然不是 VLA 论文，但它实际上给后来的 memory-VLA / history-aware policy 一个非常早的论据：  
**很多日常任务不是“当前帧理解问题”，而是“历史状态追踪问题”。**

## 5. 数据与评测：它到底揭示了哪些硬问题？ (Data & Eval)

### 5.1 benchmark 里最关键的“难”不是单点难，而是组合难

BEHAVIOR-1K 把以下难度叠到一起：

- 长时程
- 多对象
- 多状态变化
- 多场景
- 柔性物体 / 液体 /温度 / 污渍
- 强语义约束

所以它的难不是“某个 task 很 tricky”，而是它更接近日常生活中的**组合爆炸**。

### 5.2 它也很诚实地暴露了仿真训练中的“作弊空间”

论文为了让 primitive-based baseline 训得动，引入了两个简化：

1. **assistive pick primitive**  
   - 抓取时如果满足接触条件，就直接建立刚性连接  

2. **只检查 final feasibility，不完整执行整条 motion trajectory**

这其实非常重要，因为作者没有假装“我们已经完整解决真实机器人问题”，而是明确承认：

- 如果你把 fully physics-based grasping 打开，性能会大跌  
- motion execution 的完整物理执行也会引入额外误差  

这使 benchmark 更像一个**诚实地揭示困难**的系统，而不是靠大量隐藏假设刷出漂亮数字。

### 5.3 success 之外，它还看 efficiency

论文不仅看成功率，还看：

- `distance traveled`
- `time invested`
- `kinematic disarrangement`

这点非常对，因为一个策略就算“成功”，如果：

- 走了很远
- 花了很久
- 把环境弄得更乱

在真实机器人里也不算好策略。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

### 6.1 它最适合评估什么？

- 长时程 household / service robot 任务  
- 多状态、多物理过程、多对象组合任务  
- sim-to-real 研究中的“结构性 gap”  
- 需要 benchmark 不只是看 manipulation，而是看“活动完成”的研究

### 6.2 隐含假设 (Hidden Assumptions)

| 假设 | 含义 | 一旦不成立会怎样 |
|---|---|---|
| 逻辑任务定义足够覆盖人类意图 | BDDL 能表达大多数活动目标 | 会有“人觉得完成了，但逻辑没捕捉到”的情况 |
| 仿真状态足够真实 | fluids / cloth / thermal states 足够接近真实世界 | sim-real gap 仍会很大 |
| perception gap 可被缓解 | 域随机化或更强视觉模型能弥合差距 | 真实部署仍会因为视觉偏差失败 |
| primitive abstraction 合理 | 这些 primitives 足以代表任务分解 | 某些任务会被 action abstraction 人为限制 |

### 6.3 失败模式

1. **visuomotor end-to-end 学不起来**  
   - 长时程 + 稀疏奖励 + 探索难度太高。  

2. **抓取仍是巨大瓶颈**  
   - 论文自己就承认 fully physics-based grasping 一开，性能会掉很多。  

3. **sim-real gap 主要卡在 perception + grasping + navigation noise**  
   - 在 real robot 上，trained policy 甚至是 `0% success`。  

4. **任务越真实，domain knowledge 注入越难避免**  
   - 不引 action primitives 几乎做不动；引了又意味着算法并非完全“通用自主”。  

## 7. 与相关工作对比 (Comparison)

| benchmark / 系统 | 主要特点 | 与 BEHAVIOR-1K 的差异 |
|---|---|---|
| ALFRED / VirtualHome | 指令执行与程序化 household tasks | 任务语义强，但底层物理 realism 相对弱 |
| Habitat 2.0 / Rearrangement | 强导航/重排 | 更偏 rigid rearrangement，日常活动多样性较弱 |
| ManiSkill / RLBench / Meta-World | 强 manipulation benchmark | 操作能力强，但 human-centered activity coverage 不够 |
| BEHAVIOR-100 | 早期 household logic benchmark | 是直接前身，但任务规模、场景、对象和 realism 都低一个量级 |

**面试 Tip**：如果被问“BEHAVIOR-1K 和别的 embodied benchmark 最大区别是什么？”，不要只答“任务更多”。更好的回答是：**它把任务定义的起点从研究者假设改成了人类需求调查，并且配套了一个能表达复杂物理和逻辑状态变化的仿真系统。**

## 8. 对 VLA Handbook 的实际意义：为什么今天还要回头看 2024 的这篇？

### 8.1 它提醒我们：benchmark 不该只测“会不会搬东西”

今天很多 VLA benchmark 仍偏：

- pick-and-place
- 短时程 manipulation
- 相对干净的 rigid body world

但如果目标真是家庭/服务机器人，BEHAVIOR-1K 提醒我们：

**真正需要的 benchmark，应该包含清洁、烹饪、整理、开关设备、处理流体和柔性物体。**

### 8.2 对 VLA / world model 的意义

这篇论文虽然不是 VLA 论文，但它其实非常适合拿来校准今天的 VLA 讨论：

- 如果一个 VLA 只能在 LIBERO 这类短时程任务上强，不代表它能 handle 真实 household activities  
- 如果 world model 不能正确推演布料、液体、开关状态、清洁状态变化，那它也离真正家务还有距离  

### 8.3 对 VTLA / 触觉路线的意义

BEHAVIOR-1K 也间接说明了为什么触觉重要：

- 清洁  
- 布料  
- 抓取  
- 柔性材料  
- 多次接触修正

这些任务都不是纯视觉能轻松解决的。  
从这个角度看，BEHAVIOR-1K 其实是很多后续 VTLA / tactile policy 研究的“问题定义层”前辈。

## 参考链接

- 论文 HTML：[BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation](https://arxiv.org/html/2403.09227v1)  
- 项目主页：[behavior.stanford.edu](https://behavior.stanford.edu)

---
[← Back to Benchmark Hub](./README.md)
[← Back to Theory](../../README.md)
