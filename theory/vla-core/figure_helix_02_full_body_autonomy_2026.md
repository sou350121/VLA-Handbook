# Figure Helix 02：全身端到端 VLA 的“运动-操作一体化”架构 (Helix 02: Full-Body Autonomy)

> **发布时间**：2026-01-27（Figure 官方发布）  
> **论文题目/模型名**：Helix 02（Figure）  
> **核心定位**：针对 humanoid 长期未解的 **loco-manipulation（移动×操作耦合）**，Helix 02 用 **S0/S1/S2 分层统一系统**把“状态机拼接”改成“全身连续闭环”：**S2（语义目标）→ S1（200Hz 全身关节目标）→ S0（1kHz 扭矩/执行器命令）**，并把触觉与掌心视觉纳入闭环。  

行业痛点不是“会走/会抓”，而是**走与抓一耦合就崩**：拿起物体重心变、迈步可达域变、接触与遮挡让纯视觉控制不稳定。Helix 02 的最强价值在于提供了一个可复用的系统模板：**用高频 learned prior 承接稳定性，用中频 visuomotor policy 解决耦合，用低频语义 latents 承载长时程任务。**

**官方来源（一手）**：[Introducing Helix 02: Full-Body Autonomy](https://www.figure.ai/news/helix-02)

---

## 0. 先把“可复述结论”写清楚（1 分钟版）

- **一句话**：Helix 02 = “全传感器进 / 全关节出”的全身 VLA，但关键不在“端到端口号”，而在 **200Hz vs 1kHz 的分层闭环**与 **S0 人类运动先验**。
- **一张图记住**：S2 管目标，S1 管全身关节目标，S0 管稳定执行（平衡/接触/协调）。
- **三条数字锚点（官方披露）**：
  - S1：**200 Hz** 输出全身关节目标
  - S0：**10M 参数**、**1 kHz** 输出关节级执行器命令
  - S0：**>1000 小时**人类运动数据 + **>200,000 并行仿真环境**训练，替代 **109,504 行 C++**

---

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

> 目标：把“每层做什么、吃什么输入、吐什么输出、跑多快”写成一张可落地的表。

| 组件 | 频率（官方） | 输入 → 输出（官方+最小抽象） | 训练/推理关注点（推断，需实现时确认） |
|---|---:|---|---|
| **System 2 (S2)** | 未给 Hz（只说 “slow”） | scene + language → semantic latents \(z\) | 更像 “语义分解器/任务状态机替代品”：输出的是目标潜变量序列，不是脚步轨迹。 |
| **System 1 (S1)** | **200 Hz** | all sensors \(o\) + latents \(z\) → full-body joint targets \(q^*\) | 重点是 **多模态时间对齐**与 **全身耦合**：输出关节目标而非直接扭矩，降低实时性压力。 |
| **System 0 (S0)** | **1 kHz** | joint state \(x\) + base motion + target \(q^*\) → actuator commands \(u\) | 类似“学习到的 whole-body tracking controller / prior”：稳定性、接触、协调是主战场；必须强实时、抗 jitter。 |

### 1.2 关键机制 (Key Mechanism)

#### 机制 A：把“loco”与“manip”从控制器拼接变成同一耦合系统

Figure 直接批评传统方案：分拆 locomotion 与 manipulation，再用状态机串联（walk→stop→stabilize→grasp→walk），在接触不确定/物体变化时容易崩且不自然。  
Helix 02 的核心设计哲学是：**同一个系统持续看见全身与环境，并持续输出全身动作**（“steerable”，而不是 replay）。

#### 机制 B：All sensors in, all actuators out（把“观测缺口”补齐）

官方列出的 S1 输入：head cameras + palm cameras + fingertip tactile + full-body proprioception；输出：legs/torso/head/arms/wrists/fingers 的全身关节目标。  
这句话的工程含义是：**全身状态估计与末端精细操作不再是两套系统**，而是统一进同一条闭环里。

#### 机制 C：S0 = learned whole-body prior（把“稳定性”外包给一个可高频执行的模型）

S0 的官方披露可以被工程化地解读为三件事：
- 用大量人类运动数据给出“怎样动才像人且稳定”的先验  
- 用大规模仿真 + domain randomization 把先验做成可迁移的 tracking 执行层  
- 用一个神经先验替代大段手写控制逻辑（109,504 行 C++）

### 1.3 信息流/架构图 (Flow / Diagram)

```text
                         Helix 02 (pixels → torque, full-body)
                         ====================================

Sensors (all-in)
  - head cameras
  - palm cameras (in-hand vision)
  - fingertip tactile (force sensitivity ~ 3g)
  - full-body proprioception

   low freq (semantics / goals)           200 Hz (targets)              1 kHz (execution)
┌───────────────────────────────┐   ┌───────────────────────┐   ┌───────────────────────┐
│ System 2 (S2)                  │   │ System 1 (S1)          │   │ System 0 (S0)          │
│ scene + language -> latents z  │-->| o + z -> joint targets │-->| x + q* -> actuator u  │
└───────────────────────────────┘   └───────────────────────┘   └───────────────────────┘
             ▲                                   ▲                          │
             │                                   │                          v
             └────────── (task state) ───────────┴───────────────>  full-body actuators

o: multi-modal observation (vision/touch/proprio)   q*: full-body joint targets   u: joint-level commands
```

---

## 2. 数学核心：用“分层闭环”替代“状态机拼接” (Math Core)

### 2.1 目标：把不同时间尺度的控制问题拆成可闭环的接口

Helix 02 未公开可复现的训练损失/采样过程，但它公开了足够清晰的 **接口分解**：
- S2 输出语义潜变量序列 \(z\)（慢）
- S1 把多模态观测映射为全身关节目标 \(q^*\)（200Hz）
- S0 在 1kHz 下跟踪 \(q^*\) 并处理稳定性（接触/平衡/协调）

最小数学抽象如下：

\[
z_k = \pi_{S2}(\text{scene}, \text{language})
\]
\[
q^*_t = \pi_{S1}(o_t, z_{k(t)})
\]
\[
u_\tau = \pi_{S0}(x_\tau, q^*_{\lfloor \tau / r \rfloor})
\]

其中 \(r=\frac{f_{S0}}{f_{S1}}=\frac{1000}{200}=5\)。

### 2.2 变量说明（对齐本文符号）

| 符号 | 含义 | 频率/时序 |
|---|---|---|
| \(o_t\) | 多模态观测（vision/touch/proprio） | 以 S1 的 200Hz 时间基准对齐（工程上需要重采样/缓存） |
| \(x_\tau\) | 全身关节状态/基座状态等（S0 的内部状态） | 1kHz |
| \(z_k\) | S2 语义潜变量（目标/阶段） | 低频（官方未给 Hz） |
| \(q^*_t\) | 全身关节目标（targets） | 200Hz |
| \(u_\tau\) | 关节级执行器命令（actuator commands） | 1kHz |

### 2.3 直觉：为什么这能解“运动-操作耦合”

- **耦合发生在 1kHz 级别的接触与平衡**：把这部分交给 S0（learned prior）处理，避免上层每一步都纠结稳定性细节。  
- **耦合也发生在 200Hz 的全身姿态目标**：S1 直接输出全身 targets，而不是上半身/下半身各做各的。  
- **长时程只需要语义一致性**：S2 输出 latents 序列即可，不必显式规划足迹与协调动作（由 S1/S0 吸收）。

---

## 3. 带数字走一遍：4 分钟任务在三层系统里的“算账” (Worked Example)

官方演示：4 分钟厨房任务、61 个 loco-manipulation actions、无重置无人干预。

### 3.1 三层的“更新次数”量级（你能一眼看出系统压力在哪里）

- S0：\(4\times60\times1000=240{,}000\) 次输出  
- S1：\(4\times60\times200=48{,}000\) 次 targets 更新  
- S2：未给频率，但可把它理解为 “阶段级 latents”：例如 61 个 actions 对应几十个 latents 片段（推断）

结论：**真正吃实时的不是 S2，而是 S0/S1 的频率与 jitter**。

### 3.2 一个最小可实现的“缓存接口”（工程上怎么把 200Hz 喂给 1kHz）

如果把 S0 写成 1kHz 线程，它每 1ms 做一次：

- 读取最新 \(x_\tau\)（关节与基座状态）
- 从 ring buffer 读最新 \(q^*\)（S1 每 5ms 更新一次）
- 执行 \(u_\tau=\pi_{S0}(x_\tau,q^*)\)，并做安全限幅

这就是 \(r=5\) 的工程含义：**S1 的 targets 需要可重复读取、可锁步、可测延迟**。

---

## 4. 工程视角：快慢路径 / 训练-推理折中 (Engineering View)

### 4.1 “端到端”不是把所有东西塞进一个巨网

Helix 02 给出的可落地定义是：**端到端的是从 sensors 到 actuators 的统一学习闭环**；但实时性与稳定性仍然需要分层。

| 层 | 工程约束 | 典型部署（推断） | 你该测的指标 |
|---|---|---|---|
| S0（1kHz） | 强实时、低 jitter、强安全 | RT CPU / MCU / 实时线程 | jitter（p99/p999）、饱和率、碰撞/跌倒保护触发率 |
| S1（200Hz） | 多模态对齐、GPU/CPU 负载可控 | 边缘 GPU / 车载计算 | 端到端延迟、targets 稳定性（平滑/jerk）、传感缺失鲁棒性 |
| S2（低频） | 语义一致性、长时程状态维护 | GPU 上的 VLM/LLM | 任务完成率、阶段错误率、纠错恢复成功率 |

> 说明：部署位置是推断（官方未写），用于把系统抽象落地成工程 checklist。

### 4.2 掌心相机 + 指尖触觉：把“遮挡与接触”变成可观测变量

官方强调两类硬件：
- palm cameras：解决 in-hand 时 head camera 自遮挡
- tactile：3g 灵敏度（paperclip 级别）

工程上你可以把它等价成两句：
- **观测缺口被补齐**：接触相位不再靠猜
- **力调制变得可学**：策略可以在触觉反馈下调节抓握力/扭矩，而不是固定夹紧

---

## 5. 数据与评测 (Data & Eval)

### 5.1 训练数据（官方披露）

| 项 | 官方披露 | 你应该追问/补齐（TODO） |
|---|---|---|
| S0 人类运动数据 | **>1000 小时** joint-level retargeted human motion | 数据覆盖哪些动作分布？是否包含负重/携物/弯腰取物？ |
| S0 仿真训练 | **>200,000 并行环境** + domain randomization | 随机化维度有哪些（摩擦/质量/关节阻尼/触觉噪声）？ |
| 模型规模 | S0 **10M** params | S1/S2 参数量、输入分辨率、token 序列长度未披露 |

### 5.2 演示任务（官方披露）

- 4 分钟厨房任务（卸载/装载洗碗机）
- 触觉+掌心视觉支持的精细操作：拧瓶盖、取药丸、推注射器 5ml、杂物堆取件

> 注：这是演示主张，不是公开 benchmark。把它变成“可比较评测”需要：成功标准、失败类型、重复次数、统计指标。

---

## 6. 能力与失败模式 (Capabilities & Failure Modes)

### 6.1 能力（把“能做什么”讲成场景 + 原因）

- **长时程任务（分钟级）**：S2 维护语义状态，S1/S0 在低层吸收动态误差（官方称 implicit error recovery）。
- **跨尺度动作**：同一闭环既能做毫米级手指动作，也能做房间级移动（官方称跨四个数量级）。
- **遮挡下精细操作**：掌心相机 + 触觉让 in-hand 阶段可观测，降低纯视觉策略的盲区崩溃。

### 6.2 失败模式（把“会怎么死”讲具体）

- **Sim2Real 断点（接触长尾）**：S0 声称“纯仿真→真机”，真正的难点是接触/摩擦/柔顺性长尾是否被 domain randomization 覆盖。
- **传感退化**：掌心相机污渍/反光、触觉漂移/断线会把关键相位变回不可观测，策略会退化为“强夹紧/保守动作”。
- **全局耦合带来的安全问题**：全身闭环意味着错误会传播到全身（跌倒/撞击风险），需要强约束与监控（官方未披露具体 safety layer）。

---

## 7. 与相关工作对比 (Comparison)

### 7.1 对比表（关注“系统形态”而不是宣传句）

| 维度 | Helix 02（Figure，官方披露） | 传统 humanoid（拆分+状态机，通用范式） | 桌面/上半身 VLA（通用范式） |
|---|---|---|---|
| 控制形态 | **统一分层闭环**（S2→S1→S0） | 多控制器 + 状态机 handoff | 多数不覆盖动态平衡与全身协调 |
| 频率结构 | S1 **200Hz**，S0 **1kHz** | 频率分散、切换带来不连续 | 常见 10–50Hz（依实现） |
| 传感闭环 | 视觉+触觉+本体 → 全关节 | 感知/控制割裂、反馈有限 | 通常缺触觉（接触相位不可观） |
| 恢复能力 | 主张“连续可 steer + implicit recovery” | 常见“失败→重置/退回状态” | 接触/遮挡处易崩溃 |

### 7.2 面试 Tip（可直接背诵）

被问“Helix 02 的核心创新是什么？”——答：
> **不是更大模型，而是把运动-操作耦合问题写成分层闭环接口**：S2 给语义目标潜变量，S1 在 200Hz 下把多模态观测映射到全身关节目标，S0 用 10M 参数的 1kHz learned prior 做平衡/接触/协调的稳定执行，从 pixels 走到 torque，替代状态机拼接。

---

## 8. “System 0”是不是人形机器人出现的新技术方向？（对照 Sharpa/CraftNet）

**结论（更精确的说法）**：更像是一个正在收敛的 **系统形态/工程抽象**，而不是全新算法名词——行业开始把“接触之后的高频闭环层”显式化成 **System 0**，并把触觉/力反馈与本体感知纳入这一层来处理 **稳定性、接触、滑移与误差当场消化**。  
它的“新”不在分层本身（传统控制一直分层），而在于：**把最低层从手写控制器/规则，替换为可训练、可迁移、可端侧强实时的 learned controller/prior**，并把它作为 VLA 的必要底座。

### 8.1 两家公司在“同一个抽象”上的对齐点

你贴的材料把 Figure（Helix 02）与 Sharpa（CraftNet / VTLA）都描述成 **S2/S1/S0**，核心重合点是：

```text
S2: slow reasoning / task state (language, scene)  -> semantic latents / plan
S1: fast visuomotor / whole-body targets          -> joint targets / coarse motion
S0: ultra-fast feedback control (touch/force)     -> stable execution + contact refinement
```

- **共同主张**：机器人智能不只靠 VLM/VLA 的“理解+粗动作”，还必须有 “接触之后” 的高频闭环层（System 0）。
- **共同关键模态**：触觉/力反馈 + proprio（全身本体感知），用于在接触过程中持续纠错。

### 8.2 分歧点：System 0 的“主战场”不同（但互补）

- **Figure / Helix 02 的 S0**：更偏 **whole-body stability / tracking prior**  
  - 输入：全身关节状态 + base motion + 上层 targets  
  - 输出：1kHz 执行器命令  
  - 目标：把平衡、接触、全身协调做成“不会抖/不会摔/可连续 steer”的底座（官方：10M 参数、>1000h 人类运动数据、>200k 并行仿真环境训练）
- **Sharpa / CraftNet 的 S0**（材料口径）：更偏 **last millimeter / contact reflex（LMI）**  
  - 主战场：接触后用触觉与力反馈做高频精修，压住滑移/对齐误差，让长链条装配/精细操作不因误差累积而崩

一句话：**Figure 把 S0 用来“稳住整个人”，Sharpa 把 S0 用来“稳住最后一毫米”**；两者都在把“接触阶段的闭环”从可选项变成必选项。

### 8.3 为什么这会成为“新方向焦点”：长时程任务把痛点从“会不会”推到“能不能稳定做完”

你贴的案例（4 分钟厨房任务、61 个动作序列、双手占用时用髋部/脚完成交互；以及 30+ 步风车装配）都指向同一个行业迁移：

- **从单点技能** → **长链条任务完成率**  
- **从 open-loop 模仿** → **接触中闭环纠错（误差当场消化）**  
- **从“看得懂”** → **“接触后仍能稳定推进”**

因此 System 0 的价值不是“多一层”，而是它让系统具备：
- **抗遮挡/抗摩擦变化/抗柔顺性差异** 的即时纠错能力  
- 抑制 **误差累积**（长任务链的核心杀手）  
- 把“触觉驱动的交互数据”变成可规模化训练对象（因为接触中纠错可产生更密的监督/回报信号）

### 8.4 你可以用来判定“是否真是新技术路线”的 3 个核验点（最小可复盘）

- **核验 1：没有 S0（或把 S0 降频/去触觉）时，长链条成功率是否断崖**  
  不是看单步成功，而是看 20–60 step 的任务：失败是否主要来自接触后误差累积（滑移/对齐漂移/卡死）。
- **核验 2：S0 是否真在“闭环纠错”，还是只是在 replay 一段先验动作**  
  给触觉/力反馈加扰动（阈值漂移、噪声、短时丢包）或换摩擦材质，观察 S0 是否还能稳定收敛到可执行轨迹。
- **核验 3：S0 的部署形态是否具备“端侧强实时 + safety 约束”**  
  1kHz 级 jitter（p99/p999）与限幅/保护触发率如果不达标，上层再强也会被现实打回状态机。

[← Back to Theory](../README.md)
