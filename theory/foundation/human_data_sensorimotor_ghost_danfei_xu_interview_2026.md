# 人类数据是伪装成另一种形式的机器人数据：Danfei Xu 深度访谈（2026）

> ⚙️ 本文由 Moltbot 自动生成 | 2026-04-21
>
> **原文**：专访 Danfei Xu（Georgia Tech 助理教授 / NVIDIA Research）
> **核心观点来源**：Danfei 的博文 [To Summon a Sensorimotor Ghost](https://x.com/danfei_xu/status/2027034645892624528) + 2026 年访谈
> **相关工作**：[EgoMimic](https://faculty.cc.gatech.edu/~danfei/) · [EgoBridge](https://faculty.cc.gatech.edu/~danfei/) · [EMMA](https://faculty.cc.gatech.edu/~danfei/) · [InMimic](https://faculty.cc.gatech.edu/~danfei/) · [EgoScale](https://arxiv.org/abs/2602.16710)
>
> **引用规范**：📎 = Danfei 原话 / 论文数据 · 🧠 = 作者（Opus 4.7）推理、归纳、判断

---

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心论点 | 📎 人类数据 = 伪装成另一种形式的机器人数据（同样的物理法则下，感知→动作的映射） |
| 适合精读 | 做 VLA 基础模型、人类视频预训练、具身 System 2、灵巧操作 retargeting 的团队 |
| 可以跳过 | 只做固定场景遥操作 + 单任务微调的项目，短期内人类数据路径与你无关 |
| 关键洞察 | 🧠 当前"System 2"只是用 LLM 套的权宜之计，真正需要的是**非语言推理**的学习——而这只能从自然人类行为中来 |
| 主要风险 | 🧠 embodiment gap + "最后一牛顿" 差距没被数据驱动方法根本解决，teleoperation 路线可能是局部最优 |
| 时间线预判 | 📎 "类似 GPT-2 水平的通用机器人模型"：2-3 年内 40% 成功率可期 |

💡 **X-Ray 开场**

**这篇访谈解决什么问题？** — 2025 年行业对"人类数据对机器人有没有用"众说纷纭，EgoScale (2026) 首次证明 human video 遵循 log-linear scaling law 之后，下一个问题是：具体怎么用、何时用、和遥操作数据的关系是什么。

**核心发现？** — 📎 Danfei 的论点是把"人"重新定义为"**一个感知-决策主体的输入输出记录**"，而不是"更便宜的机器人数据源"。核心标准只有两个：(1) scalable / portable 能否规模化，(2) 是否真实捕捉人类决策过程。

**对 VLA 研究者意味什么？** — 🧠 从"用遥操作堆数据"转向"采集人类自然行为 + 学 embodiment-agnostic prior + retargeting 到机器人"。这是一条从数据组织方式到训练架构的系统性路径切换。

📍 **研究全景时间线**

```
[2023] 遥操作 + BC（ACT / Diffusion Policy）主导
    → [2024] 人形机器人全身控制 + motion capture 兴起（whole-body from human）
    → [2024 末] Danfei CoRL talk 提出"human data is robot data" 论点
    → [2025 H1] EgoMimic · EgoBridge · EMMA · InMimic 陆续发表
    → [2025 H2] 第一视角数据公司爆发（50-60 家采集公司涌现）
    → [2026 Q1] EgoScale (NVIDIA GEAR) 证明 log-linear scaling (R²=0.9983)
    → [本访谈] 系统化梳理路径选择 ← 当前位置
    → 未来：non-verbal reasoning 学习 + effect-driven retargeting
```

---

## 1. 核心论点：两类利用人类数据的方法论

📎 Danfei 把当前所有"用人类数据做机器人学习"的工作分成两类：

### 1.1 模仿学习路径（把人当作"特殊的机器人"）

- 从人类行为数据直接学 policy
- 再 retargeting 到真正的机器人
- 代表：**EgoMimic / EgoScale / EgoBridge / EMMA / InMimic / Physical Intelligence 的 VLA 工作**

### 1.2 世界模型路径（广义模仿学习）

- 建模"人接下来要完成的任务"（以视频形式生成未来）
- 再基于视频规划，转成具体动作
- 🧠 **作者补充**：世界模型路径严格讲是把"预测"放在前面、"控制"放在后面的两阶段架构，理论上解耦了 embodiment gap 问题，但引入了新问题——视频预测和真实接触动力学之间的"最后一牛顿"差距

### 1.3 第三种视角：Danfei 自己的定位

📎 **核心标准**（Danfei 评价一种 human data 是否值得用，只看两点）：

1. **可扩展性 / 可迁移性**（scalable & portable）—— 不能只在特定实验室做
2. **是否真实捕捉人类决策过程**—— 把人看作"感知输入 → 行为输出"的 agent，把这个过程完整记录

> 📎 "我其实不太相信存在一种'最好的' human data 表示方式...我们更倾向于把人类数据理解为：对一个具备感知和决策能力的个体的输入-输出过程进行记录。"

🧠 **作者观察**：这个定义把"human data"从素材层（motion capture、YouTube 视频、遥操作）抽象到**决策论层**——任何能记录 `(感知, 状态) → 动作` 的设备都等价。这是很关键的思维切换，但也意味着采集设备选择的重要性被降低了。

---

## 2. 数据采集：Project Aria 的技术壁垒

### 2.1 为什么不是随便一个眼镜都行

📎 Project Aria 的 5 个核心能力：
1. **双目 + 多相机系统** · 在线标定（实时补偿相机相对位置偏移）
2. **VIO**（视觉惯性里程计）做空间定位
3. **手部追踪**（high-quality 手部状态估计）
4. **深度信息**（第二代支持高质量 depth）
5. **SLAM 性能**：部分 benchmark 上 96% 精度 —— 📎 Danfei 访谈披露

### 2.2 关键的细节：热膨胀补偿

📎 Aria 的一个 engineering 细节：**设备佩戴后发热 → 摄像头相对位置漂移 → 多相机标定失效** → Aria 把热效应纳入系统建模，根据温度变化动态调 calibration。

🧠 **作者推理**：这个细节说明消费级 AI 眼镜短期内很难替代 Aria 做研究采集——不是因为光学硬件不够，而是**软件栈 + 建模深度差 10 年工程积累**。Meta 在 AR/VR 研发 10+ 年的壁垒主要在这里。

### 2.3 对数据采集者的启示

📎 Danfei 观点：**设备本身的价值 ≈ 其配套软件能力**（SLAM + 重建 + 手部估计）。硬件 5 个相机谁都能做，但没有这些基础监督信号输出，拿到的视频就是"无 label 的一堆帧"。

🧠 **工程暗示**：如果你的实验室自建 egocentric 数据采集方案，优先级应该是：
1. 双目 / 深度相机（非单目）
2. 硬件校准流程 + 热补偿
3. 手部状态估计的 ground truth（用外部 mocap 辅助）
4. 同步的 head pose（IMU + VIO）

---

## 3. 第一视角 vs 第三视角：为什么 Danfei 押注 egocentric

📎 **Danfei 的三条论据**：

1. **机器人自己也是第一视角**：人形机器人不可能在房间角落放相机看自己，视觉传感器必然装在本体上
2. **手-物交互在第一视角最清晰**：大部分 manipulation 任务的关键信号就在第一视角
3. **可扩展性**：相机戴在身上 → 走到哪采集到哪 vs 固定相机范围有限

📎 **一个被打脸的预测**：Danfei 在 2024 年 CoRL talk 上推测——egocentric 数据要等 AR/VR 设备普及才大规模可用。但 **2025 年起突然涌现 50-60 家专门采集第一视角数据的公司**，速度远超预期。

🧠 **作者追问**：这是"先有需求拉动"还是"数据公司在赌未来"？Danfei 自己说——📎 "绝大多数人并不知道应该如何利用这些数据来训练机器学习模型...目前整体上处在一种供给甚至大于需求的状态"。

🧠 **判断**：现阶段数据采集公司和 frontier labs 在做 co-evolution 循环：公司先采一批 → 研究机构反馈 → 公司调整采集方式。这种迭代的终局可能反直觉——最有价值的可能是**非刻意采集的自然行为数据**（例如用脚关冰箱门），而不是精心标注的结构化任务数据。

---

## 4. 刻意采集 vs 顺道采集：哪种真正 scalable

📎 **Danfei 的关键论点**：

> "在人类视角的数据中，真正重要的其实并不是那些刻意设计出来的行为。很多最有价值的行为，往往是人在不经意之间、没有刻意思考时做出来的。"

### 4.1 当前机器人数据采集的扭曲

典型流程（📎 Danfei 描述）：
- 明确指令："用右手推冰箱门的上部，重复 100 次"
- 动作尽量保持一致
- 🧠 这种 "clean" 数据其实丢掉了真实世界中大量的自然变化

### 4.2 "顺道"行为的独特价值

- 用脚关冰箱门（而非用手）
- 单手操作时另一手同时支撑身体
- 视线在任务完成前已经转向下一步
- 🧠 这些行为体现了**人类的分层规划 + 多任务并行 + 身体资源分配**，是严格命令下采集不到的

🧠 **作者评估**：这一点对 VLA 数据采集流程的启示极大。当前大部分团队（包括 DROID / RoboMIND 2.0）都在优化"clean demonstration"的质量和一致性，但按 Danfei 的观点，这可能是南辕北辙——**真正的人类智能表现在"顺便"的细节里**。

---

## 5. 全身 × 局部：两条平行路线的融合

📎 **当前机器人研究的分叉**：

| 路线 | 输入 | 输出 | 局限 |
|------|------|------|------|
| **Whole-body control** | motion capture 动作 | 全身关节 | "盲" —— 不依赖复杂视觉 |
| **Vision-to-action** | RGB + state | 手部动作 | 只管手，忽略全身 |

📎 Danfei 预判：**"接下来可能很快就会开始融合"**。EMMA (Egocentric Mobile Manipulation) 是 Danfei 实验室往这方向的早期尝试——同时控制底盘和手臂。

### 5.1 为什么必须融合

📎 人在现实生活中做任何事都不只是动手：
- 做饭时身体会自然参与运动
- 整理东西时会伴随重心转移
- 即使是桌面操作，也有手-眼-头部协调

🧠 **工程含义**：现在许多数据采集会刻意限制（"不要动头"、"不要动身体"）来简化问题。但一旦要 in-the-wild scale up，这种限制就站不住脚——很多真实行为在约束条件下根本做不出来。

### 5.2 硬件能做到吗

📎 理论上可行——Aria Gen2 前摄 FOV 170°-190°，很多情况下已经能看到身体大部分。
🧠 但实际落地可能还需辅助设备（手环 / 脚部追踪器）获得更准确的身体状态。Quest 等 VR 系统已经实现了一定程度的全身追踪，无需外设。

---

## 6. 核心章节：人类行为建模的真正难点

### 6.1 模仿学习的架构缺陷

📎 **Danfei 的关键批评**：

> "大多数模仿学习模型的方式其实是这样的：模型看到当前的一帧图像（instantaneous observation），然后通过一个很大的神经网络直接生成一个动作；下一帧图像进来，再生成另一个动作。也就是说，它基本是在做一种从当前观察到动作的即时映射。"
>
> "而人类并不是这样工作的。"

### 6.2 人类到底怎么工作

📎 Danfei 的例子：
> "我现在在做饭，正在切菜。突然我想起来烤箱没有关。但这并不是因为我现在看到的画面（比如面前正在切菜的场景）触发了这个动作，而是因为我记得半个小时之前做过的事情。"

📐 **这引出 VLA 的核心未解问题**：

```
现有 VLA 架构：  ┌──────────────────────────────┐
               │  vision[t] + state[t] → policy → action[t]  │
               │         ↑                                   │
               │  仅用当前感知做即时映射                      │
               └──────────────────────────────┘

人类行为本质：   ┌──────────────────────────────┐
               │  vision[t] + 内部世界状态模型(persistent) → action  │
               │                       ↑                             │
               │           记忆 · 长程目标 · 非视觉线索               │
               └──────────────────────────────┘
```

🧠 **作者观察**：这是对当前"堆参数 + 堆数据" 范式最尖锐的批评之一——**架构本身可能学不到人类智能**，因为缺乏显式的持久状态建模。

### 6.3 Danfei 眼中的"圣杯"

📎 "建立这种能够从大量人类输入输出中学习、并生成类似人类行为的系统"——Danfei 认为这是机器人领域的圣杯（holy grail）。

📎 "如果这个问题能够解决，我们可能就离机器人领域的 ChatGPT 时刻不远了。"

---

## 7. System 1 vs System 2：当前方案的"权宜之计"

### 7.1 现在所谓的 System 2 是什么

📎 目前主流做法：用 LLM / VLM 做高层规划 → "给机器人做顿晚饭" → 生成步骤序列

📎 **Danfei 的判断**：
> "这只是一个 **makeshift System 2**。它并没有真正实现人类 System 2 所承担的那种推理功能。"

### 7.2 为什么语言不够

📎 **Danfei 举的例子**：
> "你去拿一个瓶子时，可能需要施加大约两牛顿的力。但你做这个动作时，并不会在脑子里用语言描述'我要施加两牛顿的力'。"

🧠 **作者引申**：大量日常推理是**non-verbal**——具身的、肌肉记忆级的、跨模态的。这类推理占日常行为的 **大部分**，但 LLM 完全无法直接处理。

### 7.3 什么是真正的 System 2

📎 Danfei 的归纳：**所有不能通过语言表达的推理问题**——包括：
- 物理谜题（两个环扣解开）
- 精细力控任务
- 对物体柔性/摩擦的隐式建模
- 长程记忆触发的行为切换

🧠 **作者判断**：真正的 System 2 研究在机器人领域"还没有真正开始"——这是一句很重的话。它意味着：
1. 过去 2 年做的 VLM-as-planner 方向大概率是局部最优
2. 通往真正通用机器人还缺一个范式级的突破
3. 这个突破最可能的路径是**从人类数据中学 non-verbal pattern**

---

## 8. 世界模型能否成为 System 2 基础

📎 Danfei 讨论了两种 world model 形式：

### 8.1 Video prediction model 路线

- 输入：任务描述（"做饭"）
- 输出：未来关键状态（图像 / 环境状态）
- 问题：📎 **无法做抽象**。人想象"揉面团"不会在 pixel space 思考每一个像素，而是在某种抽象空间里思考"面团变成球"

### 8.2 Dynamics model 路线

- 输入：当前状态 + 动作
- 输出：未来状态
- 问题：📎 **现实动力学太复杂**。桌子上手机往前推，稍微歪了一点点轨迹就完全偏离——长时间尺度几乎不可预测

### 8.3 什么样的抽象最合适

📎 **Danfei 的关键洞察**：不同任务需要不同层级的抽象
- "明天飞洛杉矶" → 高层抽象（"订机票"）
- "画一幅画" → 精确到"第一笔落在哪里"

🧠 **作者引申**：这暗示**世界模型可能需要层级 / 可缩放的抽象空间**，而不是固定粒度。这和 JEPA (LeCun) 的 hierarchical prediction 思路有一定呼应，也和 MuZero 风格的 learned abstractions 有交集。但具体怎么学这种 task-adaptive 抽象仍然是开放问题。

---

## 9. Human-Robot Transfer：跨越 embodiment gap

### 9.1 本质问题

📎 "如果我们希望世界中产生一个和人类相同的结果（比如把瓶子拿起来），机器人应该采取什么样的动作？"

这是个 **effect-driven retargeting** 问题——不是复制人类轨迹，而是复制对世界的影响。

### 9.2 Danfei 的"无聊但正确"结论

📎 **两件事必须同时推进**：
1. 造更像人的机器人（减小 embodiment gap）
2. 更强的 retargeting algorithm

### 9.3 当前 retargeting 的三种技术路线

| 路线 | 代表工作 | 特点 |
|------|---------|------|
| **Representation Learning** | EgoMimic / EgoBridge | 学 domain-invariant feature，对齐 human 和 robot 表示 |
| **Real2Sim2Real** | 较新方向 | 把人放进 simulation → robot 替代 → RL 学习等效动作 |
| **Foundation Model + Human Data** | 📎 π × Danfei 合作 ["Emergence of Human to Robot Transfer..."](https://faculty.cc.gatech.edu/~danfei/) | 先训大机器人模型 → 加人类数据 → 减小 gap |

🧠 **Danfei 坦诚的态度**：📎 "现在其实已经有很多不同的方法在尝试解决这个问题，但我也说不好哪一种方法最终会是最好的。"

---

## 10. 通用机器人时间线

### 10.1 Danfei 的具体数字

📎 "40% 成功率的通用机器人——两三年之内应该是可能做到的"

📎 对标：**GPT-2 的水平**。"能用，但很多地方还不行，你还是需要自己做很多修改"。

### 10.2 关键的信念转换

📎 "一旦一个开放性的科学问题被转化成一个工程问题，那其实离真正理解并解决这个问题就不远了。"

🧠 **作者判断**：Danfei 这句话的隐含论断是**核心科学问题已经基本清楚**——剩下主要是数据 + scaling + 架构迭代的工程问题。这个判断比圈内一些悲观派（认为还缺范式突破）乐观，但也比盲目堆参数的乐观派更有节制。

### 10.3 商业落地 vs 研究路径

📎 **Danfei 的分离判断**：
- **短期商业落地（1-2 年）**：找"sweet spot 任务"——传统自动化做不了 + 数据驱动能搞定（叠衣服、插网线）
- **真正通用机器人**：teleoperation data 路线"可能并不是正确路径"

🧠 **关键引用**：
> 📎 "如果我们的目标是做真正具备人类能力的机器人，那我觉得 teleoperation data 可能并不是一条正确的路径。"

这句话对整个机器人学习社区是一种挑战——因为大量资金（包括 Physical Intelligence / Figure / Tesla Optimus 等）都押在 teleoperation scaling 上。

---

## 11. Sensorimotor Ghost 博文的核心拆解

📎 Danfei 博文 [To Summon a Sensorimotor Ghost](https://x.com/danfei_xu/status/2027034645892624528) 的五段核心论点：

### 11.1 现代 AI = 召唤灵魂（knowledge transfer）

- LLM 本质是把人类知识蒸馏到参数里
- 📎 Karpathy 语："训练语言模型 = 召唤灵魂"
- 🧠 机器人要做类似事情——召唤"**感觉运动灵魂**"

### 11.2 灵魂藏在哪里：动物性的 System 1

📎 Danfei 观察：日常行为大量属于反射性 System 1（肩膀被拍→转头、物体掉落→伸手接）

📎 **遥操作抓不到这些**：
- 毫秒级反射被摇杆/VR 延迟压缩
- 人与人的社会性互动（递物、共享工具）完全不在遥操作数据中

### 11.3 如何召唤：大规模自然人类数据 + 行为克隆

📎 与 LLM next-token 预测类比：**感觉运动模型做 next-action 预测**。

📎 **新兴能力预测**：
- 语言模型大规模训练 → 涌现 ICL / reasoning / agency
- 感觉运动模型大规模训练 → 可能涌现**物理常识**（堆叠会倒、重物难移、光滑表面需调整）

📎 **EgoScale 的实证**：log-linear scaling + 一次学习新任务 + 强语言指令理解都已观察到（📎 [EgoScale arXiv:2602.16710](https://arxiv.org/abs/2602.16710) 具体数字见论文）

### 11.4 给灵魂一个身体：不能只预测，还要控制

📎 **关键区分**：预测 ≠ 控制

关键挑战：**effect-driven retargeting**。不是复制人类轨迹，而是：
- 人类抓瓶的手法对 7-DoF 夹爪可能失败
- 人类依赖毫秒级反射，机器人控制带宽不足
- 需要考虑不同运动学 / 柔顺性 / 延迟 / 力限制

🧠 **"最后一牛顿"问题**（📎 原博文）：
> "几毫米的误差、几毫秒的延迟、或一牛顿的力差，都可能决定物体是滑落还是稳定。"

### 11.5 更大的科学目标

📎 **Danfei 最后的立场**：
> "建模人类不仅仅是为了制造更好的机器人。它本身就是一个重要的科学目标——**从大规模数据中恢复人类的感觉运动智能**。机器人只是其中一个应用场景。"

🧠 **作者观察**：这是一个把 embodied AI 从"工程"拔高到"基础科学"的宣言——类似 LLM 从"NLP 工具"拔高到"认知建模"的定位迁移。如果这个视角被广泛接受，人类数据的研究优先级会再上一层。

---

## 12. 对 VLA 研究者的 takeaway

🧠 **作者总结**（结合访谈 + 博文 + 当前 VLA 领域状态）：

### 12.1 数据策略

- ✅ 短期：继续遥操作 + 少量 ego-centric 人类数据（混合路线）
- ✅ 中期：把**人类视频预训练**当作一级公民（像 EgoScale 一样严肃对待）
- ✅ 长期：采集"非刻意自然行为"远比"干净的示范"有价值
- ❌ 避免：完全押注 "teleoperation scaling 就够了"

### 12.2 架构思考

- 当前 `vision[t] → action[t]` 是根本性不完整
- 需要显式**持久状态建模**（记忆、长程目标、非视觉线索）
- System 2 ≠ 套 LLM，需要学习非语言推理

### 12.3 工程取舍

- 采集设备：硬件 5 相机 ≠ 能用，SLAM + 手部 + 深度的软件栈才是壁垒
- 数据混合比：🧠 参考 EgoScale 建议（人类:人机对齐:机器人 ≈ 100:5:1 小时）
- 硬件方向：往"更像人"的方向做（减小 retargeting 难度）

### 12.4 对投资 / 研究路径的影响

| 赌注 | 风险等级 | Danfei 隐含立场 |
|------|---------|----------------|
| teleoperation scaling | 🟡 中（商业可行，通用性存疑） | 📎 "可能不是正确路径" |
| egocentric data scaling | 🟢 低（已有 log-linear 证据） | 📎 明确押注 |
| VLM-as-System-2 | 🔴 高（局部最优风险） | 📎 "makeshift" |
| world model for reasoning | 🟡 中（抽象层级未解） | 📎 "都很难" |
| humanoid hardware | 🟢 低（减 retargeting gap） | 📎 支持 |

---

## 13. ❓ 待追问的开放问题

🧠 作者提出的、Danfei 访谈中未直接回答但极关键的问题：

1. **人类数据质量 vs 规模**：EgoScale 的 log-linear 是基于干净的 MANUS 手套数据。如果用"自然但脏"的数据（YouTube、家用 AR 眼镜），scaling law 还成立吗？

2. **retargeting 的可学习性下界**：如果 embodiment gap 太大，effect-driven retargeting 是否存在 minimum representation gap？（即 fundamentally 无法跨越的界限）

3. **非语言推理的训练 objective**：如果 LLM 靠 next-token prediction，感觉运动模型靠 next-action prediction——那 non-verbal System 2 靠什么？（next-plan？next-goal？next-world-state？）

4. **规模 vs 多样性的 crossover**：目前 EgoScale 20k+ hours 还看不到平台。什么时候 scaling 会让位给多样性（环境 / 物体 / 任务类型）？

5. **"顺便"行为的数据价值证明**：Danfei 说自然行为 > 刻意采集。但这是定性直觉，还没有论文做过头对头对照实验。有人可以做这个实验吗？

---

## 14. 延伸阅读

| 主题 | 推荐 |
|------|------|
| 人类视频预训练 scaling law（实证） | [EgoScale (arXiv:2602.16710)](https://arxiv.org/abs/2602.16710) |
| VLA 数据工程全链路 | [VLA 数据工程指南](vla_data_engineering_guide.md) |
| Physical Intelligence 基础模型路线 | [Sergey Levine 深度访谈](../vla-core/physical_intelligence_sergey_levine_foundation_model_vision_2026.md) |
| 潜空间表示与抽象 | [潜空间综述 2026](latent_space_survey_foundation_evolution_mechanism_ability_2026.md) |
| VLA 架构主线 | [VLA 架构总览](../vla-core/vla_arch.md) |
| 开源 VLA 审计 | [完全开源 VLA 指南](../vla-core/open_source_vla_guide.md) |
| Danfei Xu 主页 | [faculty.cc.gatech.edu/~danfei/](https://faculty.cc.gatech.edu/~danfei/) |
| Sensorimotor Ghost 博客 | [x.com/danfei_xu/status/2027034645892624528](https://x.com/danfei_xu/status/2027034645892624528) |

---

[← Back to Explorer's Map](../README.md)
