# Belief Graph — VLA 认知赌注网络

> **v3 角色**：后端内核，不是用户界面。用户看到的是三视角辩论的输出，
> 不是置信度数字。此图的作用是：记忆一致性、传播检查、校准输入。
>
> 核心规则：
> 1. 信念之间有**条件依赖**（不是独立清单）
> 2. 每条信念必须有**带截止日期的致命实验**（不可证伪 = 无效）
> 3. 每条高置信度信念维护**最强反方叙事**
> 4. 更新时**沿图传播**
> 5. **校准纪律**：原始 >80% 置信度乘 0.9；最小更新幅度 ±5%

---

## 图结构概览

```
                    ┌─────────────────┐
                    │ B0: 数据 > 架构   │ (85%)
                    │ (根信念)          │
                    └────┬───────┬────┘
                         │       │
              ┌──────────▼─┐  ┌──▼──────────┐
              │ B1: 数据飞轮│  │ B2: RL后训练 │
              │ 是核心壁垒  │  │ 突破BC天花板 │
              │ (85%)      │  │ (90%)       │
              └──────┬─────┘  └──┬──────────┘
                     │           │
         ┌───────────▼───────────▼───────────┐
         │ B3: 自我改进闭环是终局形态          │
         │ (数据飞轮 + RL 后训练 = 自进化)     │
         │ (80%)                              │
         └────────────────┬──────────────────┘
                          │
              ┌───────────▼───────────┐
              │ B4: World Model 将成为 │
              │ 闭环的关键加速器        │
              │ (50%)                  │
              └───────────────────────┘

    ┌──────────────┐     ┌───────────────────┐
    │ B5: Flow     │     │ B6: 分层架构       │
    │ Matching 主导│────→│ (S0/S1/S2) 标准化  │
    │ action head  │     │ (75%)              │
    │ (88%)        │     └─────────┬──────────┘
    └──────────────┘               │
                                   │ requires
                         ┌─────────▼──────────┐
                         │ B7: Action Expert   │
                         │ 解耦语义与运动      │
                         │ (80%)               │
                         └────────────────────┘

    ┌───────────────┐     ┌──────────────────┐
    │ B8: 触觉      │     │ B9: 小模型VLA    │
    │ 从可选→必选   │     │ 占据边缘部署     │
    │ (55%)         │     │ (60%)            │
    └───────────────┘     └──────────────────┘
```

---

## 信念节点详细定义

### B0: 数据策略 > 模型架构（根信念）
```yaml
置信度: 85% → 校准后: 77%
含义: 2026-2027年，VLA性能提升的主要驱动力来自数据策略
      （规模/质量/飞轮/跨模态），而非模型架构创新。
前置条件: 无（根信念）
后果节点: B1, B2
支持证据:
  - [信号] PI/Tesla核心壁垒在数据规模而非架构专利
  - [信号] π0→π0.5→π0.6的主要提升来自训练范式（Recap），
    不是架构改变（VLM backbone只从3B→5B）
  - [信号] Ken Goldberg对谈：瓶颈时刻数据 > 平均数据
致命实验:
  ❌ [截止 2027-03] 出现一个≤1M episodes训练的新架构，在真机
     长时序任务上碾压用10M+ episodes训练的PI/Tesla系统
  ❌ [截止 2026-12] 某个"纯架构创新"（如全新的action representation）
     在不增加数据量的前提下提升成功率>30% absolute
  ⏰ 截止到期未发生 → 置信度 +5%
最强反方叙事:
  "架构决定了数据效率的上限。一个好架构能用1/10的数据达到
   同样的效果。当前'数据>架构'的表象是因为还没找到正确的
   架构——就像Transformer出现前大家也觉得NLP的瓶颈是数据。
   MM-ACT的统一token空间或某个未出现的'动作空间Transformer'
   可能改变这个等式。"
上次检验: 2026-03-05
```

### B1: 数据飞轮是核心竞争壁垒
```yaml
置信度: 85% → 校准后: 77%
含义: 能建立"部署→收集→训练→部署"闭环的团队将拉开决定性差距。
前置条件: B0 (数据策略主导)
后果节点: B3
支持证据:
  - [信号] π0.6 Recap：真机rollout→自我改进→2x吞吐量
  - [信号] VLAW四步闭环：policy rollout→校准WM→合成数据→更新policy
  - [信号] GigaBrain RAMP：WM条件化RL + HILR自我进化
  - [推断] 飞轮一旦转起来，数据优势指数增长（正反馈循环）
致命实验:
  ❌ 一个没有飞轮的团队（纯一次性训练）在1年后仍能与飞轮团队打平
  ❌ 飞轮团队的数据积累出现严重的分布偏移问题导致性能下降
最强反方叙事:
  "飞轮有冷启动鸡蛋问题，且自我收集的数据有distribution collapse
   风险（自己教自己，越来越窄）。跨模态迁移（从人类视频零成本获取
   数据）可能跳过飞轮直接到达数据充裕状态——此时飞轮的壁垒 = 0。"
上次检验: 2026-03-05
```

### B2: RL后训练突破BC成功率天花板
```yaml
置信度: 90% → 校准后: 81%
含义: BC的compounding error是硬限制，必须通过RL（或RL-like
      的on-policy/self-play方法）来突破。
前置条件: 无（理论上独立成立——DAgger已证明）
          注: B0(数据>架构)的成立会强化B2的重要性（飞轮需要RL驱动），
          但B2即使B0不成立也独立成立（BC的误差累积是数学事实）。
后果节点: B3
支持证据:
  - [论文明确] DAgger (2011)：BC的误差累积是O(T²)
  - [信号] π0.6 Recap：三条独立验证（PI/ByteDance/GigaAI）
  - [信号] GR-RL MoT三阶段：BC warmstart → Sim RL → Real RL
  - [推断] 任何长时序任务（>30步）BC-only方案最终会碰壁
致命实验:
  ❌ [截止 2027-06] 某种非RL方法（如超大规模BC+数据增强+特殊架构）
     在100+步真机任务上达到>80%成功率
  ❌ [截止 2027-03] "大力出奇迹"路线：10B+模型+100M episodes纯BC
     证明数据规模能暴力压过compounding error
  ⏰ 截止到期未发生 → 置信度 +5%
最强反方叙事:
  "BC的compounding error在理论上是O(T²)，但实践中action
   chunking + receding horizon control已经大幅缓解了这个
   问题。如果chunk size足够大（比如50步），有效T就很小。
   结合更好的感知（减少输入噪声）和更强的VLM backbone
   （减少决策错误），BC的实际天花板可能比理论上高很多。
   RL的工程复杂度和不稳定性可能让很多团队止步于BC-only。"
上次检验: 2026-03-05
```

### B3: 自我改进闭环是VLA的终局形态
```yaml
置信度: 85% → 校准后: 77%
含义: 最终胜出的VLA系统将具备"无需人工干预即可持续提升"的能力。
前置条件: B1 AND B2（需要数据飞轮AND RL后训练同时成立）
后果节点: B4
支持证据:
  - [推断] B1+B2 = 自我改进闭环（飞轮提供数据，RL提供改进信号）
  - [信号] π0.6* (Pi-Star)：已初步展示24小时连续自我改进
  - [信号] VLAW：world model作为想象力加速器
  - [信号] Robometer (2026-03): 1M+轨迹通用reward model，跨embodiment，
    支持多种RL范式。与RoboReward (Google, 2026-01) 共同验证VLM-as-reward可行
  - [信号] RoboReward 8B: 真机RL中改进策略学习，接近人工reward效果
致命实验:
  ❌ 自我改进在100次迭代后收敛到次优解且无法逃出（mode collapse）
  ❌ 真机安全约束让自主探索空间太小，导致飞轮无法有效转动
  ❌ reward specification问题无法解决（机器人hack reward而非完成任务）
最强反方叙事:
  "自我改进需要reward signal，但真实世界的reward极难定义。
   AlphaGo可以自博弈因为围棋有完美reward（赢/输）。机器人
   操作没有完美reward——'成功'的定义是模糊的、context-dependent的。
   所以自我改进闭环在围棋上work不等于在机器人上work。
   最终可能回到人类反馈（RLHF for robots）作为主要信号源。"
   反方更新(2026-03-11): Robometer/RoboReward在已知任务上有效，
   但OOD泛化和reward hacking风险仍未排除。核心问题从"能不能定义
   reward"转向"reward model在新任务上是否可靠"。
上次检验: 2026-03-11 (Robometer + RoboReward双信号更新)
```

### B4: World Model成为闭环的关键加速器
```yaml
置信度: 60%
含义: World Model将在2026下半年从"辅助工具"升级为
      自我改进闭环中不可或缺的组件（通过想象力加速数据生成）。
前置条件: B3（闭环形态成立）
后果节点: 无（末端节点）
支持证据:
  - [信号] VLAW +39.2%（通过WM合成数据改进策略）
  - [信号] DreamZero 7Hz闭环（WM直接作为策略）
  - [信号] Ctrl-World/WorldArena统一评测出现
  - [信号] CoWVLA (2026-03): latent motion space WM，绕过pixel-space物理幻觉
  - [信号] Cosmos Policy (NVIDIA, 2026-01): video model直接变policy，
    LIBERO 98.5% + 真机bimanual SOTA。WM = policy的范式转换
  - [信号] AtomVLA (2026-03): subtask-aware latent WM post-training
  - [信号] Interactive World Simulator (2026-03): consistency model WM，
    真机验证 WM 生成数据训练策略 ≈ 真实数据策略效果，单卡 15 FPS
  - [信号] PlayWorld (Princeton, 2026-03): 首个完全自主WM训练pipeline，
    机器人自主play → SVD-based WM → DSRL in WM → 真机RL成功率+65%。
    Pearson 0.8766 WM预测与真实相关性。无需人类数据训练WM。
  - [信号] NVIDIA Cosmos Predict 2.5 (2026-03): 开源WM用于机器人合成数据
    生成和策略评估，降低WM工程门槛至开箱即用
挑战证据:
  - [信号] 物理幻觉仍是核心问题（接触/可变形预测差）
  - [信号] DreamZero 14B → 计算成本过高
  - [推断] 没有WM的纯RL方案（如Recap）也能工作
致命实验:
  ❌ 2026年底前无团队在真机>1000 episodes中验证WM辅助方案
     优于纯BC+RL baseline
  ❌ 物理幻觉问题在接触密集任务中无实质性改善
  ✅ 如果WM辅助方案在真机上显著减少所需真实rollouts数量
     （>5x数据效率提升）→ 置信度升至70%+
最强反方叙事:
  "World Model的计算开销（14B video diffusion）和物理不准确性
   使其成为净负资产——花在WM上的算力不如直接用来做更多真实
   rollouts。WM是'看起来很酷但实际上是绕路'的方案。Recap
   证明了直接从真实数据学习比从想象中学习更可靠。"
   反方更新(2026-03-11): latent WM方案(CoWVLA/Cosmos/AtomVLA)
   绕过了pixel-space物理幻觉，但latent space是否保留
   足够的物理因果信息仍未在接触密集任务中验证。
   Interactive World Simulator声称physically consistent，但
   验证范围（rigid/deformable/piles）未含接触密集精细任务。
   反方更新(2026-03-13): PlayWorld 65%改进来自小规模实验(20 trials/policy,
   8小时数据)，冗余轨迹问题未解，长期幻觉累积仍存在。
   "autonomous play"是否能在工业复杂场景中保持数据质量待验证。
上次检验: 2026-03-13 (PlayWorld + Cosmos Predict 2.5)
```

### B5: Flow Matching主导VLA action head
```yaml
置信度: 88% → 校准后: 79%
含义: 到2026年底，新发布的VLA论文中>60%使用Flow Matching。
前置条件: 无（独立技术判断）
后果节点: B6 (支持分层——FM的高频能力适配分层中的S1)
支持证据:
  - [信号] π0全系列 + LingBot + UnifolM + DreamZero
  - [论文明确] ODE 5-10步 vs Diffusion 50-100步
  - [推断] 确定性ODE无随机抖动→更适合高频闭环
反方新增证据:
  - [信号] AR-VLA (ETH/Van Gool, 2026-03): 自回归Action Expert + 长时记忆,
    SIMPLER 61.5% > π0.5(51%) > CogACT(52%), 真机89%, jerk 7.89 < FM 9.39。
    首个在FM主场(SIMPLER)上打败π0系列的non-FM方案。
    但仅在WidowX单臂简单任务验证，humanoid/双臂空白。
致命实验:
  ❌ [截止 2026-12] >10B FAST-based VLA在真机长时序达到π0.6级别性能
  ❌ [截止 2027-06] 专用推理芯片使Diffusion推理速度追平Flow Matching
  ❌ [截止 2026-12] AR-based action expert在>3个独立团队的humanoid/双臂
     任务上超越FM方案 → B5降至60%
  ⏰ 截止到期未发生 → 置信度 +5%
最强反方叙事:
  "FAST tokenizer方案与LLM next-token基础设施完全兼容——
   当VLA backbone从5B扩展到50B+时，训练效率比推理速度
   更重要。大规模预训练阶段FAST的并行化优势可能压过FM的
   推理优势。Flow Matching的ODE求解器在极端高维动作空间
   （如30+ DoF humanoid）中的稳定性也还没被充分验证。
   新增(2026-03-14): AR-VLA证明自回归+长时记忆可在简单任务上
   打败FM，且jerk更低。如果AR方案在高维/长时序中也成立，
   FM的'唯一正确答案'叙事将瓦解。"
上次检验: 2026-03-14 (AR-VLA反方证据: SIMPLER 61.5% > π0系列)
```

### B6: 分层架构(S0/S1/S2)成为humanoid VLA标准
```yaml
置信度: 75%
含义: 人形机器人VLA将收敛到S2(语义)→S1(运动)→S0(执行)的分层。
前置条件: 无（但B5支持：FM的高频能力适配S1层）
后果节点: B7
支持证据:
  - [信号] Figure Helix 02: S2/S1(200Hz)/S0(1kHz)
  - [信号] Galaxea G0: 大脑+小脑
  - [信号] GR00T-N1.6: 双系统DiT
  - [生物依据] 皮层/小脑/脊髓分工
致命实验:
  ❌ 端到端统一模型（如OneTwoVLA的自适应切换）在humanoid
     全身任务上超越分层方案
  ❌ 分层方案的层间延迟成为精细操作的瓶颈
最强反方叙事:
  "分层是妥协不是目标。理想方案是端到端，当前分层只是因为
   算力不够单模型200Hz全身控制。一旦硬件跟上（专用芯片），
   端到端的简洁性和全局优化能力会碾压分层方案的层间信息损失。"
上次检验: 2026-03-05
```

### B7: Action Expert解耦语义与运动
```yaml
置信度: 78%
含义: VLA应将"理解任务"与"生成运动"解耦为独立模块，而非用
      同一组权重处理语义理解和连续动作生成。Action Expert
      （专用运动模块）能避免灾难性遗忘且提升运动精度。
前置条件: B6（分层架构中S1层自然需要专用运动模块）
后果节点: 无（末端节点，但与B5有协同：FM适配Action Expert）
支持证据:
  - [信号] π0系列：独立action expert + FM，不干扰VLM backbone
  - [信号] GR00T-N1.6：双系统DiT将语义和运动解耦
  - [信号] RoboVLM/SpatialVLA：action expert显著降低灾难性遗忘
  - [推断] LLM backbone的离散token空间与连续运动的高维流形
    本质不同——强行统一是在"语言空间做物理"
新增支持证据:
  - [信号] Samsung DAM-VLA (2026-03): 双头diffusion (arm+gripper)，VLM-guided
    动态路由，SIMPLER 71%最高。VLM路由=语义→运动的显式解耦验证
  - [信号] AR-VLA (ETH/Van Gool, 2026-03): 独立AR Action Expert + DTR异步机制，
    VLM 70ms + Action 29ms解耦执行，SIMPLER 61.5%真机89%。
    Action Expert解耦的又一独立验证（AR路线，非FM/Diffusion）
反方新增证据:
  - [信号] WholeBodyVLA (ICLR 2026): 统一 latent action 无需显式 action expert，
    humanoid +21.3%——"软分层"可能取代"显式解耦"
致命实验:
  ❌ 统一token空间方案（MM-ACT类）在>10个任务的持续学习中
     证明不需要解耦也能避免灾难性遗忘且保持运动精度
  ❌ 某个50B+统一模型内部自然涌现出运动子空间，功能等价于
     显式解耦但无需架构改动
最强反方叙事:
  "显式解耦是工程hack而非最优解。足够大的统一模型会自发形成
   内部功能分区（如LLM中不同layer处理不同层次信息）。强行解耦
   反而限制了语义-运动的联合优化——比如在需要'语义指导运动'
   的精细操作（如'小心地把鸡蛋放进盒子'）中，解耦可能丢失
   关键的跨模态信息流。MM-ACT和LLARVA等工作已经在探索统一
   方案，如果规模足够大，解耦可能是多余的。"
上次检验: 2026-03-14 (AR-VLA: 独立AR Action Expert + DTR异步解耦)
```

### B8: 触觉传感从"可选"→"必选"
```yaml
置信度: 65%
含义: 到2027年中，触觉输入将从"锦上添花"变为精细操作VLA的
      必需通道——没有触觉的VLA在接触密集任务上将显著落后。
前置条件: 无（独立判断，但B3闭环成立会加速：触觉reward信号
          可驱动更高效的RL自我改进）
后果节点: 无（末端节点，但与CONVERGENCE_MAP Phase 3联动）
支持证据:
  - [信号] TaCo benchmark发布 — 触觉基础设施出现
  - [信号] TacRefineNet/TaF-VLA/SuperTac+DOVE — 多团队独立验证触觉价值
  - [信号] UniVTAC统一仿真 — 标准化基础出现
  - [信号] DexHand021产品化出货 — 硬件成本下降中
  - [信号] GenForce (Nature Comm 2026): 首个跨触觉传感器可迁移力感知框架，
    统一marker表征实现跨设备力预测迁移 — 标准化方向的破冰信号
  - [信号] MoDE-VLA (上交/上海AI Lab, 2026-03): 首个VLA框架内力觉+触觉融合+
    量化消融。力觉去除→成功率-11%，触觉去除→-8%。残差注入机制即插即用。
    Contact-rich bimanual任务成功率2x over π₀ baseline。触觉不可替代的最强量化证据
  - [推断] 纯视觉VLA在"接触判断"类任务（软物体抓取、力控装配）
    上有信息论天花板——缺乏接触力信号无法判断抓取是否稳定
致命实验:
  ❌ 2027-03前主流VLA论文中触觉输入占比<15% (GenForce+MoDE-VLA仍不改变此条——需统计)
  ✅ 触觉传感器成本<$100/手指 且统一数据格式→升至70% — GenForce部分满足"统一格式"（仅力感知子集）
  ❌ 某种视觉力估计方案（visual force estimation）在精细操作
     上追平触觉方案 → 触觉硬件变得不必要
  ✅ 触觉传感器成本降至<$100/手指 且有统一数据格式→升至70%
最强反方叙事:
  "触觉传感器的碎片化是结构性的而非暂时性的——不同任务需要不同
   的传感模态（力/滑移/温度/形变），永远不会有一个'ImageNet for
   touch'。更重要的是，视觉+深度已经能覆盖90%的操作场景——触觉
   只在极少数精细任务中必需，但这些任务的商业价值不足以驱动标准化。
   硬件耐久性问题（传感器皮肤磨损）也让工业部署困难重重。触觉会
   一直停留在学术论文中的'nice to have'。"
上次检验: 2026-03-14 (MoDE-VLA量化消融: 力觉-11%, 触觉-8%)
```

### B9: 小模型VLA (<3B)占据边缘部署
```yaml
置信度: 63%
含义: <3B参数的轻量VLA将主导需要低延迟、低功耗、离线运行的
      边缘部署场景（工厂线、消费级机器人、无网络环境）。
前置条件: 无（独立判断，但与B5协同：FM的轻量推理有利于小模型）
后果节点: 无（末端节点）
支持证据:
  - [信号] SmolVLA 256M — 验证了极小模型的可行性
  - [信号] 多家公司蒸馏路线（大模型→小模型部署）
  - [信号] LiteVLA-Edge (2026-03): 首个完整VLA端侧4-bit GGUF量化部署，
    Jetson Orin上150.5ms延迟(~6.6Hz)完全离线运行
  - [推断] 真机控制频率要求（>10Hz）对推理延迟有硬限制
  - [推断] 边缘场景（工厂、家用）无法承受云端依赖和带宽成本
致命实验:
  ❌ 云端推理延迟被5G/边缘计算解决到<10ms → 大模型也能实时
  ❌ 小模型在复杂多步任务上的成功率天花板太低（<50%）
     导致市场被迫回到大模型+云端方案
  ❌ 2026下半年专用推理芯片让7B+模型在端侧达到>10Hz (尚未发生)
最强反方叙事:
  "小模型是过渡方案。专用推理芯片（如Hailo/Qualcomm AI边缘芯片）
   的指数级进步将在2-3年内让7B甚至更大的模型在端侧实时运行。
   届时小模型的'低延迟'优势消失，而其能力天花板（复杂推理、长
   时序规划）成为致命短板。投资小模型蒸馏技术可能是在优化一个
   即将被硬件进步淘汰的方向。不如直接投注大模型+芯片。"
上次检验: 2026-03-09 (LiteVLA-Edge信号更新)
```

---

## 逆共识投注（Contrarian Portfolio）

> 这些是我**刻意维护的反主流信念**。不一定相信，但必须追踪。

### C1: "架构创新即将回来"
```
当前共识: 数据 > 架构（2026年的主旋律）
逆共识: 某个根本性的新architecture（不是对现有范式的微调）将在
        12个月内出现，其数据效率提升10x，重新让架构成为核心变量。
置信度: 20%
追踪信号: 任何在小数据集上显著超越大数据方案的工作
为什么维护: 如果Transformer级别的范式突破出现在VLA领域，
           "数据>架构"的叙事会瞬间翻转。
```

### C2: "World Model是死胡同"
```
当前共识: WM是VLA的重要辅助/加速器
逆共识: WM永远无法达到足够的物理保真度，最终被纯real-data方案淘汰。
置信度: 20% (↓2%, 2026-03-11收敛扫描: Interactive World Simulator进一步削弱C2)
追踪信号: WM辅助方案在真机长期部署中的failure mode分析
为什么维护: 如果物理幻觉问题本质不可解（而非暂时性工程障碍），
           整个WM方向的投入回报率可能为负。
```

### C3: "VLA不需要language"
```
当前共识: VLA = Vision + Language + Action
逆共识: Language作为中间表征是低效的；直接vision→action的端到端
        方案（辅以少量structured goal specification）最终更高效。
置信度: 24% (↑2%, 2026-03-11收敛扫描: XPENG VLA 2.0量产去除L的产业信号)
追踪信号: 无语言VLA方案在跨任务泛化上的表现
新增追踪(2026-03-11): VLM4VLA (ICLR 2026) 系统性证明视觉模块是VLA
  瓶颈，语言贡献有限。但InstructVLA (ICLR 2026) 在复杂指令任务上
  证明语言仍重要——答案可能取决于任务复杂度。
为什么维护: Language引入了"推理→动作"的传递损耗（GenieReasoner/ERIQ
           已量化），如果这个损耗不可消除，去掉L可能更好。
```

---

## 传播规则

当更新任一节点时，执行：

```python
def propagate(updated_node):
    # 1. 检查后果节点
    for child in updated_node.consequences:
        if updated_node.confidence 显著变化:
            重新评估 child.confidence
            propagate(child)  # 递归

    # 2. 检查前置条件
    for parent in updated_node.preconditions:
        if updated_node 被推翻:
            检查 parent 是否仍有其他支撑

    # 3. 检查逆共识
    for contrarian in contrarian_portfolio:
        if updated_node 的变化支持 contrarian:
            contrarian.confidence += Δ
            如果 contrarian.confidence > 40%:
                升格为正式信念节点
```

---

## 变更日志

| 日期 | 变更 |
|------|------|
| 2026-03-05 | v2 重写：从独立假设列表 → 条件依赖图 + 致命实验 + 逆共识 |
| 2026-03-05 | v3 升级：降级为后端内核 + 校准折扣 + 致命实验截止日期 |
| 2026-03-09 | Paper scan: B8 55%→60% (GenForce), B9 60%→63% (LiteVLA-Edge) |
| 2026-03-11 | Paper scan: B3 80%→85% (Robometer+RoboReward reward model), B4 50%→55% (CoWVLA+Cosmos Policy+AtomVLA latent WM), C2 25%→22%, C3 15%→22% (VLM4VLA ICLR'26) |
| 2026-03-11 | 收敛扫描: B4 55%→60% (Interactive World Simulator WM数据≈真实数据), B7 80%→78% (WholeBodyVLA latent action无需显式解耦), C2 22%→20%, C3 22%→24% (XPENG VLA 2.0去除L量产) |
| 2026-03-13 | Daily digest: B4证据+2 (PlayWorld自主play→WM→RL +65%真机, Cosmos Predict 2.5开源WM), B7证据+1 (Samsung DAM-VLA双头diffusion). 置信度无变更——PlayWorld规模有限(20 trials)，未满足kill condition. Phase 4 counter 7→9. |
| 2026-03-14 | Paper scan: B8 60%→65% (MoDE-VLA量化消融: 力觉-11%, 触觉-8%——触觉不可替代最强证据). Phase 3 counter 7→8. PhysiFlow/Mean-Flow One-Step: B5/B6收敛证据，无置信度变更. |
| 2026-03-14 | Daily digest: AR-VLA (2603.10126)——B5反方新增证据(SIMPLER 61.5%>π0.5, jerk<FM), B7+1证据(独立AR Action Expert+DTR). B5新增致命实验(AR在>3团队humanoid超FM→降至60%). 产业: Mind $2B + Sunday $1.15B. 置信度无变更. |

---

*配合 CLAUDE.md v3 使用。此图是后端内核，用户看到的是三视角辩论输出。每次更新必须沿图传播。*
