# 让策略先“看见”，再“摸准”：TouchGuide 触觉推理引导 (TouchGuide: Inference-Time Steering of Visuomotor Policies via Touch Guidance)

> **发布时间**：2026（arXiv）  
> **论文题目**：TouchGuide: Inference-Time Steering of Visuomotor Policies via Touch Guidance  
> **核心定位**：不是把触觉硬塞进 base policy 里重训，而是在**推理阶段**用 task-specific Contact Physical Model（CPM）对动作采样做“接触物理引导”，让已有的 Diffusion Policy / π0.5 在接触密集任务里更稳。

很多视触觉工作都在问“怎么融合模态”，TouchGuide 换了个角度：**先让视觉策略给出一个大致可行的动作，再让触觉在最后几步把它往“物理上更可行”的方向拉。** 这使它更像一种“接触期外挂”，而不是一套必须从头重训的全新 VLA。

**X-Ray 开场**：这篇论文解决的是“触觉很关键，但又稀疏、昂贵、难直接训进大模型”这个问题。它的发现是，触觉不一定要在表征层和视觉平权，很多时候只要在**动作空间的关键采样窗口**里提供正确梯度，就足以显著抬高成功率。对 VLA / VTLA 研究者来说，这意味着：**触觉的第一步未必要做成新 backbone，也可以先做成 test-time steering 模块。**

**一手来源**：
- 论文 HTML：[https://arxiv.org/html/2601.20239v3](https://arxiv.org/html/2601.20239v3)

---

## 📍 研究全景时间线

```text
Feature concatenation
    -> 触觉常被视觉“淹没”，关键接触相位不稳定

Policy Consensus
    -> 每个模态有专家策略，在策略层做组合

TaF-VLA
    -> 把触觉对齐到“力语义”，补 VLA 的力盲区

TouchGuide
    -> 不重训 base policy
    -> 在 action sampling 后段引入 tactile guidance
    -> 本质是“动作空间 steering”，不是特征拼接
```

TouchGuide 的位置很清楚：它不是要替代 VLA，而是给现有视觉策略加一层**接触期修正器**。

---

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 方案 | 触觉放在哪一层 | 是否要重训 base policy | 优点 | 典型问题 |
|---|---|---|---|---|
| 视觉+触觉特征拼接 | 表征层 | 通常需要 | 实现直观 | 触觉稀疏时容易被当噪声 |
| Policy Consensus | 策略层 | 需要训练多专家+router | 模块化、可降级 | 训练与推理链条更长 |
| TaF-VLA | 物理语义层 | 需要 adapter / 对齐训练 | 把触觉升级成“力语义” | 更偏表示学习 |
| **TouchGuide** | **动作采样层** | **不需要重训 base policy** | **适合把已有视觉策略快速升级到 contact-rich 场景** | 仍需任务级 CPM 与触觉示教 |

### 1.2 关键机制 (Key Mechanism)

1. **两阶段动作生成**  
前半段由 base policy 仅根据视觉生成“粗动作”，保证语义和几何上大致合理。

2. **CPM 评估动作是否符合当前接触物理**  
CPM 接收视觉、触觉、当前 noisy action，输出 feasibility score。

3. **在采样末段做 steering**  
不是全程干预，而是在后几步采样里根据 score 的梯度修正动作，避免过早把触觉噪声放大。

4. **跨 policy 复用**  
论文里直接接到了 Diffusion Policy 和 π0.5 的 action expert 上，说明它更像“外接修正头”。

### 1.3 信息流/架构图 (Flow / Diagram)

```text
             visual observation V_t
                      |
                      v
      pre-trained base policy (DP / pi0.5 action expert)
                      |
         early sampling -> coarse action A_t^k
                      |
                      +-----------------------------+
                      |                             |
                      v                             |
 tactile observation T_t                    Contact Physical Model
                      |                     input: (V_t, T_t, A_t^k)
                      +-----------> fused latent -> score s_phi
                                                    |
                                                    v
                                  grad wrt action: grad_A s_phi
                                                    |
                                                    v
                           late sampling steering -> refined action
```

TacUMI 则是这条链的上游数据来源：给人类示教提供更直接、更低延迟的触觉反馈，减少“采得差导致策略学不好”。

---

## 2. 数学核心：TouchGuide 如何做动作空间引导 (Math Core)

**Napkin Formula**：先给动作，再问“这个动作和当前触觉/视觉是否物理一致”；如果不一致，就沿着 score 梯度把动作往更可行的方向推。

**目标**：把触觉融合从“表征拼接”改成“动作可行性评分 + 采样修正”。

**核心量**：

```text
feasibility score:
s = O_t^T a_t

其中
O_t = visual-tactile observation latent
a_t = action latent
```

直觉上，`s` 越高，表示“当前观测下，这个动作越像专家会做出来的动作”。

**Diffusion Policy 的 steering**：

```text
eps_hat = eps - eta * sqrt(1 - alpha_bar_k) * grad_A s_phi(V_t, T_t, A_t^k)
```

**Flow Matching / π0.5 action expert 的 steering**：

```text
u_hat = u - eta * (k / (1 - k)) * grad_A s_phi(V_t, T_t, A_t^k)
```

**变量解释**：
- `A_t^k`：第 `k` 步的 noisy action
- `s_phi`：CPM 输出的 feasibility score
- `eta`：guidance scale
- `grad_A s_phi`：告诉策略“往哪个方向改动作更符合当前接触物理”

**关键直觉**：
- 视觉负责“把手伸到哪里、朝哪个大方向走”
- 触觉负责“接触后到底有没有卡准、捏稳、插到底”
- TouchGuide 不重写 base policy，只在最容易出错的后半段做纠偏

---

## 3. 带数字走一遍：一个锁孔插钥匙的玩具例子 (Worked Example)

**场景**：Lock Opening。视觉看到钥匙已经对准锁孔，于是 base policy 给出一个“插入不够深但准备开始旋转”的动作。

```text
base policy coarse action:
insert = 0.3
rotate = 0.8

current tactile:
still no "bottom-out" signal
contact pattern says key is not fully seated
```

这时 CPM 会给这个动作一个较低的 feasibility score，因为“未完全插到底就旋转”不符合专家示教里的接触模式。

经过 steering 后，动作会被往更合理的方向推：

```text
refined action:
insert = 0.7
rotate = 0.2
```

也就是说，TouchGuide 做的不是重新理解任务，而是把“最后一厘米”的接触闭环改对。对插入、剥皮、易碎物交接这类任务，这一步往往比上层语义更关键。

---

## 4. 工程视角：快慢路径 / 训练-推理折中 (Engineering View)

### 4.1 为什么这条路对工程团队很友好

- **不用重训整套 base policy**：如果你已经有一个能工作的视觉策略，TouchGuide 更像可插拔升级件。
- **推理期改造小**：论文给出的伪代码本质上只是在采样循环里多了几行 score + gradient。
- **适合“触觉是关键但不总出现”的任务**：比如插入、交接、擦拭、锁开启，真正决定成败的是接触窗口，而不是全程。

### 4.2 推理速度代价并不夸张

论文给出的 RTX PRO 6000 Blackwell 结果：

| Base Policy | 原始速度 | 加 TouchGuide 后 | 变化 |
|---|---:|---:|---:|
| π0.5 | 18.52 fps | 17.24 fps | -6.91% |
| Diffusion Policy | 12.82 fps | 12.35 fps | -3.67% |

这说明它不是“为了触觉把实时性全丢了”，而是用较小延迟代价换明显更高的接触成功率。

### 4.3 TacUMI：为什么论文专门做了一套采集系统

作者很明确地指出：很多 contact-rich 任务学不好，不只是 policy 的问题，而是**示教质量先坏掉了**。TacUMI 的定位是一个低成本、高精度、带直接触觉反馈的 handheld gripper：

| 设计点 | 论文口径 |
|---|---|
| 定位方式 | Vive Tracker + Lighthouse |
| 基础成本 | 约 720 美元（不含触觉传感器） |
| 重量 | 约 540g |
| 关键特性 | rigid fingertip，直接触觉反馈 |
| 采集频率 | 视觉 + 触觉 30Hz |

**工程含义**：TouchGuide 不只是一个算法，它同时在回答“你怎么采到足够好的视触觉示教”。

### 4.4 对 VTLA 的意义

如果你的目标是做 VTLA，TouchGuide 更像：

```text
已有视觉策略 / VLA
        +
task-specific tactile guidance
        +
高质量 contact-rich data collection
```

它非常适合当“先把接触期做稳”的中间层，但还不是完整的通用 VTLA 终局。因为它没有解决跨任务统一触觉表征，也没有把触觉真正做进通用语言-动作 backbone。

---

## 5. 数据与评测 (Data & Eval)

### 5.1 任务设置

论文在 5 个高接触密集任务上验证：

| 任务 | 典型难点 | demos |
|---|---|---:|
| Shoe Lacing | 穿孔方向、抓持位置、滑移 | 100 |
| Chip Handover | 易碎交接、空中 handover、姿态对齐 | 50 |
| Cucumber Peeling | 双臂协同、刀口接触、稳定施力 | 50 |
| Vase Wiping | 曲面接触、接触点控制、过力风险 | 30 |
| Lock Opening | 抓钥匙、插入角、插到底再旋转 | 20 |

### 5.2 主结果

**平均表现（论文表 1）**：

| 方法 | 平均表现 |
|---|---:|
| Diffusion Policy | 16.3% |
| TouchGuide (DP, tactile image) | 36.2% |
| π0.5 | 35.9% |
| TouchGuide (π0.5, tactile image) | 58.0% |

这很重要，因为它说明：
- TouchGuide 对小一些的 DP backbone 有帮助
- 对更强的 π0.5 也能继续叠加提升
- 触觉不是“替换视觉策略”，而是“把强策略最后一段接触修正做对”

### 5.3 关键消融

**Noise pretraining 很关键**：CPM 如果只见过干净专家动作，推理时面对 noisy action 会不适配。论文里给出的消融显示，在 Chip/Cucumber/Lock 三个任务平均值上，加入这一步可从 `39.17%` 提升到 `62.50%`。

**Vision 与 Touch 两边都重要**：去掉视觉或去掉触觉，平均值都会从 `62.50%` 掉到约 `43%` 左右，说明它并不是“纯触觉接管”，而是“视觉先提议、触觉后校正”。

---

## 6. 能力与失败模式 (Capabilities & Failure Modes)

**它真正擅长的事**：
- 给已有视觉策略快速补一层接触闭环，不必重训主干
- 在“第一下接触就决定成败”的任务里提升明显
- 对触觉这种“稀疏但关键”的模态更友好，因为它不要求触觉全程主导
- 结合 TacUMI，形成“数据采集 -> CPM 训练 -> test-time steering”闭环

**它没有解决的事**：
- CPM 仍是**任务级**的，不是通用跨任务触觉世界模型
- 需要额外训练 CPM，也需要相应触觉示教
- 指导超参数 `eta / guidance steps` 需要任务调节
- 更像“接触期外挂”，还不是统一的 VTLA foundation model

**典型失败边界**：
- 任务没多少接触阶段，TouchGuide 的收益可能有限
- 触觉延迟、漂移、标定不稳时，guidance 可能给错方向
- 如果示教本身质量差，CPM 学到的也是差的 contact prior

---

## 7. 与相关工作对比 (Comparison)

| 方法 | 核心问题 | 触觉角色 | 训练代价 | 最适合的使用方式 |
|---|---|---|---|---|
| Feature Concatenation | 怎么把触觉喂进策略 | 额外输入模态 | 中到高 | 任务简单、模态稳定 |
| Policy Consensus | 怎么让稀疏模态在策略层有话语权 | 独立专家策略 | 高 | 多传感器模块化系统 |
| TaF-VLA | 怎么把触觉升级成“力语义” | 物理力表征 | 中到高 | 追求更强力感知语义 |
| **TouchGuide** | **怎么在不重训主策略时用好触觉** | **动作可行性 steering 信号** | **中** | **先把接触密集任务做稳** |

**一句话定位**：
- Policy Consensus 是“多专家怎么投票”
- TaF-VLA 是“触觉怎么变成力语义”
- TouchGuide 是“已有策略怎么在最后几步被触觉拉回正确轨道”

**面试 Tip**：  
“TouchGuide 最值得记住的不是它加了 tactile，而是它把触觉的作用点从 feature fusion 改到了 action sampling。它告诉我们：在接触密集任务里，触觉不一定要全程主导，但必须在关键几步有纠偏权。”

---

## 参考链接

- 论文 HTML：[https://arxiv.org/html/2601.20239v3](https://arxiv.org/html/2601.20239v3)

---
[← Back to Theory](../README.md)
