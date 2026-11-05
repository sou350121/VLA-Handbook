# 动作专家该不该训练 VLM？动作 Token 监督与梯度路由 (Should the Action Expert Train the VLM? Action-Token Supervision and Gradient Routing in Mixture-of-Transformer VLAs)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-11-05
>
> **论文**: Should the Action Expert Train the VLM? Action-Token Supervision and Gradient Routing in Mixture-of-Transformer VLAs
> **链接**: https://arxiv.org/abs/2611.05249
> **核心定位**: 在 π0 式 Mixture-of-Transformer VLA 中，把“动作 token 监督”和“动作专家梯度是否回传 VLM”解耦，检验 insulation 是否真是默认正确配方。
>
> **证据说明**: 本文依据 arXiv:2611.05249v1 正文（2026-11-03，CC BY 4.0）写作。候选元数据摘要为截断片段（止于 “VLM supervis”），来源记录未给出摘要核验结果；因此我只采用正文可核对内容，不用截断摘要补写主张。正文摘要与候选片段在可核对部分未见结论冲突，差异主要是截断。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 作者主张：动作 token 监督与专家梯度是部分可替代的动作学习信号；insulation 的价值取决于是否有动作 token 监督、训练顺序和评测设置，不应默认开启 |
| 适合精读 | 如果你在做 MoT VLA 训练配方、π0 式 K/V 条件、动作专家与 VLM 的梯度路由，重点看 §3、§4.1、§4.3 与 Appendix A.1/A.3 |
| 可以跳过 | 如果你只关心 VLA 架构创新或真实机器人部署，这篇是训练配方与消融研究，距离中等 |
| 落地可行性 | 中：改动集中在 backward 路由与 CE mask，工程接口清晰；但闭环驾驶与操作均为仿真评测、开环驾驶为离线数据评测，每个条件一次训练，真实机器人未验证 |
| 主要风险 | 结论有条件：有 action token 时开环差异小；闭环收益与安全指标不一定同向；不能把 15/16 步 action chunk 当记忆窗口 |

💡 **X-Ray 开场**

这篇论文问的是一个很具体的训练问题：动作专家的 flow-matching loss，要不要把梯度回传到 VLM？常见做法是 stop-gradient，也就是 insulation，理由是保护预训练 VLM；但很多系统同时又在训练 VLM 预测离散动作 token。论文把这两个因素交叉，发现没有动作 token 时，耦合明显更好；有动作 token 时，开环驾驶差异很小，但闭环驾驶和操作任务仍可能从耦合受益。对 VLA 研究者意味着：不要只问“冻结还是反传”，要先问“VLM 还有没有别的动作监督信号、训练顺序是什么、评测是开环还是闭环”。

📍 **研究全景时间线**

```
2024  π0：VLM + flow-matching action expert，专家条件于 VLM 的 K/V
2025  FAST / π0.5：离散动作 token 监督 + 两阶段、绝缘式配方流行
2026  Knowledge Insulation / MolmoAct2 / Wall-OSS-0.5 / APT：
      绝缘与耦合路线并存，但训练配方、初始化、数据不同，结果不可直接比
2026-11  本文：在驾驶中交叉 action-token supervision × gradient routing × training schedule；操作只复现 concurrent，变化 FAST 监督 × routing
```

## 1. 核心架构/方法总览

### 1.1 系统对比概览

论文不是提出一个新 VLA 架构，而是在固定架构下做受控消融。三个模型家族的属性必须分开看，不能把驾驶模型、π0.5 重实现、MolmoAct2 的不同属性合并成“所有模型都如此”。

| 模型/实验 | VLM 主干 | 动作专家与接口 | 动作表示与时序 | 数据与开放范围（按论文） |
|---|---|---|---|---|
| 驾驶模型 | Qwen3-VL-2B-Instruct | 同宽、非因果 flow-matching expert；每层专家 attend 对应 VLM 层的 K/V；VLM 不 attend 专家的连续动作输入 | 专家预测 64 个加速度—曲率控制对，10 Hz，6.4 s；VLM 在动作 token 监督开启时预测同一未来控制序列的 128-token 离散化 | 使用 Physical AI Autonomous Vehicles 数据集（Public Golden 1,434 例、OOD 273 例）与 AlpaSim；正文未声明驾驶模型权重开放 |
| π0.5 重实现 | 正文称 PaliGemma-3B；Table 7 将 VLM backbone 记为 Gemma 2B、18 层、宽 2,048 | Gemma 300M 随机初始化专家，18 层、宽 1,024；专家条件于 VLM 层间 K/V | 专家预测 16 步 action chunk，32 维 padded 空间，前 8 维为 7 关节 + 1 夹爪；训练时关节为 delta、夹爪绝对；闭环以 8 步块开环执行 | 使用 OpenPi 与 FAST 实现；VQA 与机器人数据期望各半；机器人流 90% DROID、10% PolaRiS；正文未声明该重实现权重开放 |
| MolmoAct2 | Molmo2-ER，36 层、宽 2,560 | 随机初始化 MolmoAct2 expert，36 层、宽 768；专家条件于层间 K/V | 专家预测 15 步 action chunk，8 个 active 维在 32 维 padded 空间；关节与夹爪绝对；闭环以 8 步块开环执行 | 从官方 Molmo2-ER checkpoint 初始化，使用 released MolmoAct2 implementation 与 FAST tokenizer；数据按公开来源重建，18 个 SO 仓库不可用，约 0.398% loader mass 在 SO 组内重分配 |

> 注：π0.5 处正文写 “PaliGemma-3B VLM”，Table 7 的 “VLM Backbone” 写 “Gemma 2B”。这可以理解为 PaliGemma 的语言 backbone 维度，但本文不把 3B 总参数与 2B 语言 backbone 混为同一事实；涉及参数时以表格维度为准。

**先区分已观测历史帧与未来预测帧。** 驾驶模型的输入上下文包含当前观测、语言、16 个 waypoint 的 ego 轨迹历史，以及每相机 4 个时间帧；VLM context length 为 16,384 token。未来部分是 64 个控制对 / 128 个动作 token，属于预测区间，不是历史上下文。若等间隔采样 K 帧、间隔 Δt：

```
首末采样点跨度 = (K-1) × Δt
若从当前参考 t 预测 t+1 至 t+H，终点距当前参考 = H × Δt
```

按论文的 64 个控制对、10 Hz 做算术核验：64 × 0.1 s = 6.4 s；首末采样点跨度是 (64-1) × 0.1 s = 6.3 s。这是对论文 64 步 / 10 Hz 的算术解释，不是论文额外声明。操作的 16 步 / 15 步 action chunk 没有给出统一 Δt，不能换算成秒，更不能当作记忆窗口；闭环实际以 8 步块开环执行。

**记忆机制逐模型归属。** 论文没有提出跨 episode memory。驾驶模型的“历史”是输入中的 ego 轨迹历史和多帧观测；π0.5 重实现与 MolmoAct2 的观测是外部 RGB、腕部 RGB、关节与夹爪值，论文未描述额外记忆模块。预测 action chunk 是未来动作区间，不是历史上下文或记忆窗口。

### 1.2 关键机制

论文的核心设计是 2×2 因子交叉，再加三种训练顺序：

- 因子一：是否给 VLM 未来动作 token 的 CE 监督（action-token supervision on/off）。
- 因子二：动作专家的 flow loss 是否反传 VLM（coupled / insulated）。
- 训练顺序：concurrent（VLM 与随机专家一起训练 300k steps）；expert-first（专家先对冻结 VLM 训练 100k steps，再联合 200k steps）；VLM-first（VLM 先单独训练 100k steps，再接新专家联合 200k steps）。所有 schedule 总优化步数都是 300k。

关键控制：

1. **Insulation 在主实验中是 stop-gradient，不是所有阶段都冻结 VLM。** 前向 K/V 不变，只阻断 flow loss 进入 VLM；VLM 仍由 CE 更新。Expert-first 的 100k warm-up 才是 VLM 冻结。
2. **动作 token 关闭时只 mask 未来动作 value token 的 CE。** 这些 token 仍 teacher-forced 留在序列中；轨迹分隔符、past-trajectory token、VQA、reasoning 仍受监督；CE 分母固定为 mask 前监督位置数，避免其他 CE 项被加权。
3. **专家看不到未来动作 value token 状态。** 专家面对的 K/V cache 在 `<future_start>` 后立即结束，不包含未来 value-token 状态和 `<future_end>`，所以 teacher-forced 未来值不会泄漏给专家。
4. **评测时未来动作 token 从不解码。** VLM 生成 rationale，在 start token 停止；专家基于该 K/V cache 预测连续控制。动作 token 只作为训练时的辅助信号。

⚡ **Eureka Moment**：这篇论文最核心的洞见不是“耦合一定更好”，而是“动作 token 监督和专家梯度是两条部分可替代的动作相关信号；只比较 gradient routing 会把 action-token supervision 的效果混进来”。

### 1.3 信息流/架构图

```
驾驶模型：
图像/语言/ego 历史
      ↓
Qwen3-VL-2B-Instruct ── CE ──> VQA / rationale / 结构 token
      │                         └─ 可选：未来动作 token CE
      │ 每层 K/V
      ↓
flow-matching action expert ──> 64 个加速度-曲率控制对
      │                         ↓
      │                  unicycle 积分 → waypoints
      │
      └─ 路由选择：
         coupled: flow loss 梯度可回 VLM
         insulated: sg(K/V)，flow loss 只更新专家

操作模型：
外部/腕部 RGB + 关节/夹爪 + 指令
      ↓
VLM backbone ── CE ──> VQA / 语言 / 可选 FAST action token
      │ 层间 K/V
      ↓
flow-matching expert ──> 16 或 15 步 action chunk
      │
      └─ 闭环实际以 8 步块开环执行
```

## 2. 数学核心

📌 **Napkin Formula**：

```
动作相关表征 ≈ CE(action tokens) 提供的梯度 + flow 梯度回传提供的梯度；两者部分可替代
```

目标很简单：VLM 的 K/V 要携带对连续控制有用的信息，但 VLM 预训练时没有为 flow-matching 目标优化。于是有两条信号：动作 token 的交叉熵，或专家 flow loss 反传。

```text
L = L_CE + L_flow

L_CE = -E_(x,y)~D [
  (1/N) * (
    Σ_{i∈I_base} log p_θ(y_i | x, y_<i)
    + 1_action * Σ_{i∈I_action} log p_θ(y_i | x, y_<i)
  )
]

L_flow = E_(x,a)~D_action, t, ε [
  || v_φ(a_t, t; H̃_θ) - (a - ε) ||_2^2
]

a_t = t a + (1 - t) ε
b ~ Beta(1.5, 1.0)
t = 0.999(1 - b)
ε ~ N(0, I)

coupled:    H̃_θ = H_θ
insulated:  H̃_θ = sg(H_θ)
```

变量说明：

| 符号 | 含义 |
|---|---|
| `θ` | VLM 参数 |
| `φ` | 动作专家参数 |
| `x` | 当前观测、语言上下文、past trajectory history |
| `y` | 自回归目标序列 |
| `a` | 连续未来轨迹/控制目标 |
| `H_θ` | VLM 提供给专家的层间 K/V 表示 |
| `sg(·)` | stop-gradient，前向不变、反向阻断 |
| `1_action` | 动作 token CE 开关；0 时未来动作 value token 被 mask |
| `I_base` | 始终监督的 token：VQA、driving rationale、轨迹分隔符等 |
| `I_action` | 未来轨迹 value token |

动作 token 关闭时的实现损失更精确地写成：

```text
L_CE^(p) = ( Σ_{i∈O} ℓ_i + p * Σ_{i∈F} ℓ_i ) / ( |O| + |F| )

p = 1：动作 token 监督开启
p = 0：动作 token 监督关闭
O：其他监督位置
F：未来动作 value-token 目标位置
ℓ_i：位置 i 的 next-token CE
```

直觉：`p=0` 只把动作 token 的分子项去掉，分母仍用 mask 前监督位置数，所以其他 CE 项的权重不会因为关闭动作 token 而变大。flow 与 CE 在含轨迹的 batch 上各取单位权重。

## 3. 带数字走一遍：玩具例子

> 以下数值是假设的，只用于验算公式与时间语义，不是论文实验数据。

**A. 时间跨度验算。** 假设等间隔采样 4 个未来控制点，`Δt = 0.1 s`：

```text
首末采样点跨度 = (4-1) × 0.1 = 0.3 s
若从当前参考 t 预测 t+1 至 t+4，终点距当前参考 = 4 × 0.1 = 0.4 s
```

所以“4 个点”不等于“首末跨度 0.4 s”；首末采样点跨度是 0.3 s，终点距当前参考才是 0.4 s。这个区别在把论文的 64 个控制对解释成 6.4 s 时很关键。

**B. Flow loss 验算。** 假设一维目标 `a = 1.0`，噪声 `ε = -0.2`，`t = 0.5`：

```text
a_t = 0.5 × 1.0 + 0.5 × (-0.2) = 0.4
目标向量场 a - ε = 1.0 - (-0.2) = 1.2
若专家预测 v = 0.9
L_flow = (0.9 - 1.2)^2 = 0.09
```

在 coupled 条件下，这个 `0.09` 的梯度可以穿过 K/V 回传 VLM；在 insulated 条件下，前向 K/V 仍是同一数值，但 `sg` 阻断该梯度。若动作 token CE 也开启，VLM 还收到离散动作 token 的 CE 梯度；若关闭，VLM 只靠其他 CE 项，动作相关信号只剩 coupled 的 flow 梯度。

**C. CE mask 验算。** 假设 2 个 base 监督位置、4 个未来动作 value 位置，`p=1`：

```text
分子 = base 的 2 项 + 动作的 4 项
分母 = 2 + 4 = 6
```

`p=0` 时：

```text
分子 = base 的 2 项
分母仍为 6
```

因此 base 项没有被放大；动作 token 只是从分子中移除。这与论文 Appendix A.1 的描述一致。

## 4. 工程视角

**训练成本与路由成本。** 驾驶实验每个训练 run 使用 64 张 H100，平均约 15,000 GPU-hours；全局 batch 128，总 300k optimizer steps。论文没有报告 coupling 相对 insulation 的显存、吞吐或延迟数字。我的条件性分析是：coupling 要让 flow loss 穿过层间 K/V 回传到 VLM，反向图需要保留更多上游激活，显存和反向计算通常高于 stop-gradient；但这是工程推断，不是论文实测。

**推理成本。** 驾驶专家从高斯噪声开始，用 10 个 Euler steps 生成连续控制；评估时 VLM 先生成 rationale，再在 start token 停止，随后专家预测。操作模型同样用 10 个 flow-integration steps。闭环执行不是把 16/15 步 chunk 全量开环执行，而是以 8 步块开环执行；论文没有给出控制频率的额外换算，因此不能把 15/16 步直接说成秒数。

**动作 chunk ≠ 记忆窗口。** 驾驶的 64 个控制对是未来预测区间；操作的 16/15 步是预测 action chunk；它们都不是历史上下文或记忆窗口。驾驶的历史窗口是输入中的 16-waypoint ego history 与每相机 4 帧；VLM context length 16,384 token 是另一个维度。工程落地时如果把预测 chunk 长度误当执行窗口，会把“预测 16 步、执行 8 步”错写成“执行 16 步”。

**Schedule 的更新预算不等。** 总步数相同不代表组件更新次数相同。论文 Table 6 给出期望组件更新：concurrent 为 VLM 300k、专家约 145.5k；expert-first 为 VLM 200k、专家约 197k；VLM-first 为 VLM 300k、专家约 97k。因此 schedule 比较是“总优化步数对齐”，不是“每个组件梯度暴露完全对齐”。这是读结果时必须保留的条件。

## 5. 数据与评测

**预训练设置。** 本文不从零预训练 VLA；驾驶模型从 Qwen3-VL-2B-Instruct 初始化，π0.5 重实现使用 PaliGemma 系 VLM，MolmoAct2 从官方 Molmo2-ER checkpoint 初始化。预训练不是本文变量；本文变量是动作专家引入后的监督信号与梯度路由。

**评测设置。** 驾驶开环在 Public Golden（1,434 例）与 OOD（273 例）这两个公开驾驶数据集上做离线 ADE 评测；每个例子采样 6 条候选轨迹，主指标是第一条候选的 ADE，minADE 是用真值选 6 条中最好的一条，属于 oracle 选择指标。闭环在 AlpaSim 的 827 个场景上评，每个场景 5 次随机 rollout，先在场景内平均。操作在 PolaRiS 的三个任务上评：BlockStackKitchen、FoodBussing、PanClean；每个条件每个环境 50 次 rollout，匹配初始状态，450 步 episode，指标是 rubric 归一化任务进度。驾驶 rationale 用冻结 Lingo-Judge 评 first-sample pass rate（分数 > 0.5），并报告 top-k 与 avg-k。

**真实机器人设置。** 论文明确说没有在物理机器人上测试操作策略；这是作者列出的限制。因此所有操作结论只属于 PolaRiS 仿真，不应外推为真实机器人增益。

**实验次数单位。** 训练是 optimizer steps：驾驶 300k；π0.5 重实现 400k；MolmoAct2 300k 后再 1k adaptation steps。评测是 scenes / examples / rollouts：驾驶闭环 827 scenes × 5 rollouts；操作每条件每环境 50 rollouts；开环 bootstrap 是 10,000 次 paired clip-cluster resamples。不要把“50 rollouts”误写成 50 次训练，也不要把“5 rollouts per scene”当成 5 个训练种子。

## 6. 能力与失败模式

**论文实测的能力。** 驾驶开环：有动作 token 时，insulated 与 coupled 差异小，论文称 insulated 在 Public Golden 略低 ADE、OOD 差异较小；无动作 token 时，coupled 相对 insulation 在 Public Golden 降低约 26% ADE、OOD 降低约 24%，摘要概括为约 25%。闭环驾驶：concurrent 下路线进度从 0.600 到 0.703（有 token），从 0.425 到 0.601（无 token）；后者绝对 +0.176、相对约 +41%，摘要写约 40%。操作：π0.5 重实现中，FAST on 从 19.5% 到 30.7%，FAST off 从 10.3% 到 32.9%；MolmoAct2 中 FAST on 从 22.0% 到 24.3%，FAST off 从 14.5% 到 36.5%。这些是论文表格中的观测值，不是我的复现。

**作者主张。** 作者主张动作 token 监督与专家梯度是部分可替代信号；insulation 不是默认正确配方；concurrent training 是一个强而简单的起点。作者也明确说这是 empirical study，不证明背后机制。

**失败模式与边界。**
- 有 action token 时，开环驾驶的 routing 差异很小；如果只看开环 ADE，可能得出“insulation 够用”的结论，但闭环路线进度仍可能从 coupling 受益。
- VLM-first 且有 token 时，闭环路线进度几乎持平（0.648 vs 0.646），说明训练顺序会改变 routing 的收益大小。
- 安全指标不同向：论文提到有 token 的 staged coupled 训练 at-fault collision 更高，尽管路线进度更高或相近；不能只用 route progress 判断配方优劣。
- 生成 rationale 的 Lingo-Judge 通过率不等于广泛 VLM 能力保持。作者只把它当 driving rationale 质量诊断，并明确该结果不能证明更广泛的 VLM 能力保留；论文也没有用 VQA 评测来诊断能力保持。
- 每个条件只训练一次，未测训练种子方差；bootstrap 区间只反映固定 checkpoint 下的评测 clip 变化。
- 闭环驾驶（AlpaSim）与操作（PolaRiS）仅仿真，开环驾驶 ADE 为 Public Golden/OOD 离线数据评测；未做真实机器人操作验证。驾驶只有一个模型家族，操作只有两个模型家族，架构多样性有限。

### 6.1 隐含假设

1. **动作 token 与连续专家编码同一未来轨迹，因此 CE 可以作为动作相关信号的替代。** 这在驾驶中成立得较直接：128-token 离散化对应同一未来控制序列；在操作中 FAST tokenization 与连续 action chunk 的关系更依赖 tokenizer 和动作空间，不能无条件外推。
2. **Stop-gradient 只改 backward，前向 K/V 不变，因此 routing 比较是干净的。** 前向确实不变，但优化轨迹会变；总步数对齐不等于组件更新预算对齐。
3. **语义 co-training 能提供足够的 VLM 能力保护，因此 insulation 的额外保护未必明显。** 论文用 Lingo-Judge 只诊断 driving rationale，不能证明 VQA 或广泛语义能力保持；这是条件性结论。
4. **Concurrent training 的绝对表现最好，因此可作为默认起点。** 论文中 concurrent coupled + token 的 0.703 是最高观测路线进度，但这只是在论文的模型、数据和仿真设置下；不能直接说所有 VLA 都应 concurrent。
5. **Action chunk 长度不是执行窗口，也不是记忆窗口。** 论文的 16/15 步是预测块，闭环执行是 8 步块；把两者混用会误读工程约束。

## 7. 与相关工作对比

| 工作/配方 | 关注点 | 梯度路由 | 动作 token | 训练顺序 | 证据与边界 |
|---|---|---|---|---|---|
| π0.5 / MolmoAct2 类配方 | VLM + flow expert 的规模化 | 常见 insulation / 分阶段 | FAST 或离散动作 token | 常为 VLM-first 两阶段 | 报告收益，但未隔离 insulation 与 action-token supervision |
| Knowledge Insulation | 保护 VLM、加速训练 | insulation | FAST + 视觉语言目标 | 分阶段 | 未评估“无 FAST 时 insulation 是否仍好” |
| Wall-OSS-0.5 | 耦合优化 | coupled | 保留离散动作与多模态目标 | 不同配方 | 报告更强动作表现，但与上述工作数据/初始化不同，不能直接比 |
| APT | 专家预训练 | 比较 insulated/coupled | 视配方 | expert-first 动机 | 发现专家准备会改变 routing 相对表现 |
| 本文 | 解耦 action-token supervision × routing × schedule | coupled vs stop-gradient | on/off 独立变化 | 驾驶：concurrent / expert-first / VLM-first；操作：仅 concurrent | 驾驶（开环离线数据 + AlpaSim 闭环）与操作（PolaRiS 仿真）受控消融；每条件一次训练 |

**面试 Tip**：被问“π0 式 VLA 要不要 stop-gradient？”先反问三件事：VLM 有没有动作 token CE、训练顺序是 concurrent 还是 staged、评测是开环 ADE 还是闭环任务进度。本文的回答是条件性的：无动作 token 时耦合显著更好；有动作 token 时开环差异小，但闭环仍可能受益，且安全指标不一定同向。

## 8. 精读建议

- **值得精读原文的人**：正在设计 MoT VLA 训练配方的研究者；要做 π0.5 / MolmoAct2 类模型复现与消融的工程师；关心自动驾驶 VLA 闭环评测与安全指标的人。
- **建议章节路径**：先读 §3 的 2×2 设计与公式，再读 §4.1 的 concurrent 结果与 Table 3，随后看 §4.3 的 Table 4 与 Appendix A.3 的数据/架构归属；如果关心安全权衡，补 Appendix E 的 Table 11–13；Related Work 可在熟悉 π0/FAST/Knowledge Insulation 后跳读。
- **不值得精读的理由**：如果你不做机器人学习、不需要改 VLA 训练流程，或者只关心真实机器人部署，读摘要与 §6 的失败模式即可；论文没有真实机器人实验，也没有提出新架构。

---
[← Back to Theory](./README.md)

**关键引用**：本文论文链接 https://arxiv.org/abs/2611.05249（arXiv:2611.05249v1，2026-11-03）。