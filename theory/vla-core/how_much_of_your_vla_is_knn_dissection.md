# 你的 VLA 有多少是 kNN？(How Much of Your VLA is kNN)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-11-13
>
> **论文**: How Much of Your VLA is kNN
> **链接**: https://arxiv.org/abs/2611.13160
> **核心定位**: 用 training-free kNN 检索示范动作 chunk，量化 LIBERO 高分中有多少可由“近邻记忆”解释；并给出离线诊断指标 LAC，用来判断 VLM backbone 作为 retrieval key 的质量。

> **版本与摘要说明**：本笔记依据 arXiv:2611.13160v1（2026-11-09）HTML 正文（对应 v1 PDF）。候选摘要给出 91.7% 与 LIBERO-Plus 最小降幅，但未区分 91.7% 是 multimodal kNN、93.0% 的 LAC key 含线性 key 训练，也省略了逐模型 fine-tuning 分析；本文按 PDF/HTML 正文区分，不用候选摘要补写正文没有的主张。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 在 LIBERO 上，不训练策略、只做示范检索的 kNN 可达 91.7%（multimodal；ResNet 88.7%，带 LAC 线性 key 93.0%），说明基准高分中有很大一块可由“匹配测试状态、复用示范动作”解释。 |
| 适合精读 | 如果你在做 VLA 评测、LIBERO 基线、retrieval/non-parametric policy，重点看 §3、§5、§6.1、§6.3、§6.5。 |
| 可以跳过 | 如果你只关心真实机器人新轨迹泛化，这篇距离中等：它只测 LIBERO/LIBERO-Plus 仿真。 |
| 落地可行性 | 中：作为评测基线容易复现；作为部署策略受 memory 覆盖、检索延迟和新任务示范约束。 |
| 主要风险 | 只在密集示范的 LIBERO 类 benchmark 上验证，不能外推到真实机器人或 memory support 之外。 |

💡 **X-Ray 开场**
这篇论文问：LIBERO 上 95%+ 的成功率，有多少是策略真的泛化，多少只是把测试观测匹配到训练示范并复用动作？作者用不训练策略的 kNN 检索，在 LIBERO 上拿到 91.7%（multimodal），在 LIBERO-Plus 上 82.4% 且降幅 9.3 个百分点，为所有评测策略最小。对 VLA 研究者意味着：报 LIBERO 分数时，应同时报 retrieval baseline，否则很难区分“基准记忆”与“泛化”。

📍 **研究全景时间线**

```
2022  VINN：kNN 视觉模仿基线
2023  LIBERO 发布；Diffusion Policy 的 action chunk 时序
2025  π0 / π0.5 / FAST；LIBERO-PRO 质疑记忆
2026  LIBERO-Plus；本文 v1（2026-11-09）：用 kNN 反推 LIBERO 成功率
```

本文位置：把 retrieval 从“老基线”变成“评测探针”；局限是只覆盖 LIBERO 类仿真，未验证真实机器人。

## 1. 核心架构/方法总览 (Overview / Architecture)

论文把 kNN 同时当两类工具用：

1. **baseline**：用示范检索直接产生动作，不训练 policy；
2. **探针**：把某个 VLA 的 VLM hidden states 当 retrieval key，比较 fine-tuning 前后“这个 backbone 有多适合检索动作”。

### 1.1 系统对比概览 (System Component Comparison)

| 组件/变体 | 输入 | 输出 | 时序与执行 | 训练/更新 |
|---|---|---|---|---|
| kNN (ResNet) | 当前图像，frozen ImageNet ResNet-18 特征 | 未来 action chunk 的加权平均 | 只用图像距离；限制在指令对应任务的示范；k=8；执行完整加权 chunk 后重新检索 | 无 policy 训练；encoder frozen |
| kNN (multimodal) | 图像 + 机器人状态 + 语言 + episode time + 最近 observation history | 同上 | 从全部示范检索；k=13；mode seeking + temporal ensembling；每个 control step 重规划 | 无 policy 训练；超参用 Optuna |
| kNN (LAC key) | 标准化 ResNet grid 特征，经线性投影 | 同上 | 与 multimodal 相同的检索/执行 | 线性 key 用 LAC 的可微代理训练；动作仍来自示范 |
| 九个 VLA | 各自 observation/language 接口 | 各自 action chunk | 逐模型不同：VLA-JEPA 每次重规划执行 7 个动作，Evo-1 执行 14 个，GR00T 执行 16 个预测中的 8 个；X-VLA 用绝对末端位姿 | 使用公开 checkpoint；本文不重新训练 |

> 注意：上表把 kNN 三个变体分开列；九个 VLA 的具体设置不能合并成“所有 VLA 都用同一 chunk 长度/同一记忆窗口”。逐模型线索见 Appendix B 与 Table 4。

### 1.2 关键机制 (Key Mechanism)

- **检索记忆**：对每条示范轨迹，存“已观测帧 o_i + 它之后的未来 action chunk a[t_i:t_i+H] + 来源 demonstration 编号”。论文从 1693 条 LIBERO 示范中每隔一帧取帧，得到约 137k 条 memory entries。
- **多模态距离**：视觉、机器人状态、语言、episode time、最近 history 各算一个距离，按模态尺度归一化后加权求和。语言项很关键：消融中忽略 prompt 会掉 27.1 个百分点。
- **每个 demonstration 最多一个邻居**：防止 top-k 全来自同一条轨迹，降低对单一示范的过拟合。Optuna 的 fANOVA 显示，k 和这条唯一性约束分别解释约 57% 和 25% 的超参重要性。
- **softmax 加权**：越近的邻居权重越大；实现中温度按邻域平均距离缩放，使权重对距离绝对尺度不敏感。
- **执行策略**：ResNet 变体执行完整 chunk 后再检索，查询次数少但反应粗；multimodal 变体每个 control step 重规划，并用 temporal ensembling 合并重叠预测，减少 chunk 边界抖动。
- **LAC**：不跑 rollout，直接看“表示空间里的近邻是否也对应相似未来动作”。它用邻居动作距离相对于同任务平均动作距离的比值，衡量 retrieval key 的局部动作一致性。

⚡ **Eureka Moment**
在密集示范的 benchmark 上，“检索 + 复用示范动作”本身就是一个强基线；VLA 的 backbone 好不好，可以用它当 retrieval key 的 LAC/成功率来诊断，而不只是看端到端任务成功率。

### 1.3 信息流/架构图 (Flow / Diagram)

```
离线构建 memory：
LIBERO demos (1693 条)
  → 每隔一帧取已观测帧 o_i
  → 存 (o_i, 未来动作 chunk a[t_i:t_i+H], demo 编号 m_i)
  → 约 137k 条 entries

kNN 推理：
当前已观测帧 q
  → 图像/状态/语言/time/history 距离 d(q,i)
  → top-k，每个 demo 最多 1 个
  → softmax 权重 α_i
  → π(q) = Σ α_i · a[t_i:t_i+H]
  → 执行未来 chunk
  → 下一控制步或下一 chunk 重新检索

VLA 对照路径：
q → VLM backbone → action head → 未来 action chunk

LAC 离线诊断：
held-out q → 同一 key 的 top-k
  → 邻居动作距离 / 同任务动作距离
  → LAC@k
```

关键区分：memory 里存的是**已观测帧 + 未来动作 chunk**；推理时 query 是**当前已观测帧**；history 项用的是**过去观测**；π(q) 是**未来动作**。三者不能混为同一个“窗口”。

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：

```
检索距离 + 邻居加权 = 未来动作 chunk 的加权平均；
LAC = 1 - 邻居动作距离 / 同任务动作距离
```

**目标**：给定当前观测 q，从 memory 中找相似示范状态，复用它们之后的动作 chunk；同时用 LAC 离线判断 retrieval key 是否把“动作相似”的状态放在近邻。

**核心公式**：

```
M = {(o_i, a[t_i : t_i+H], m_i)}_{i=1..N}

d(q,i) = Σ_r (w_r / s_r) · d_r(q,i)

α_i = exp(-d(q,i)/τ) / Σ_{j∈N_k(q)} exp(-d(q,j)/τ)

π(q) = Σ_{i∈N_k(q)} α_i · a[t_i : t_i+H]

LAC@k(q) = 1 - [ (1/k) Σ_{i∈N_k(q)} ||a_q - a_i||² ]
                / [ (1/|C(q)|) Σ_{j∈C(q)} ||a_q - a_j||² ]
```

**变量说明**：

| 符号 | 含义 |
|---|---|
| M | retrieval memory |
| o_i | 已观测帧（历史） |
| a[t_i:t_i+H] | o_i 之后的未来 action chunk |
| m_i | 来源 demonstration 编号 |
| d_r, w_r, s_r | 模态距离、权重、尺度归一化 |
| N_k(q) | top-k 邻居；每个 demonstration 最多一个 |
| α_i | softmax 邻居权重 |
| τ | 温度；实现中按邻域平均距离缩放 |
| C(q) | 与 query 同任务的全部 memory entries |
| LAC@k | 离线 retrieval key 指标 |

**直觉**：

- LAC 接近 1：邻居动作比“同任务随机配对”更接近 query 动作，说明 key 把动作相似的状态聚在一起。
- LAC 接近 0：邻居动作与同任务平均差不多，retrieval key 没提供额外动作信息。
- LAC 为负：邻居动作反而更远，key 可能坏了。论文中 fine-tuned π0 backbone 的 LAC 为 0.55，而其他 key 为 0.76–0.88，对应其 retrieval 成功率从 86.9% 掉到 17.9%。

**时间跨度公式**（必须区分历史与未来）：

```
设历史采样 K_f 帧，间隔 Δt：
  首末跨度 = (K_f - 1) × Δt

若从当前参考 t 预测 t+1 至 t+H：
  终点距当前参考 = H × Δt
```

论文没有在可见正文给出统一 H、Δt、K_f 的数值；这些是超参数或逐模型设置。**不能把某一个设置的 action chunk 长度写成所有实验共有的记忆窗口**，也不能把未来 chunk 的预测区间直接当作历史上下文或 memory 窗口。

## 3. 带数字走一遍：玩具例子 (Worked Example)

> 以下为**假设数值**，仅用于验算 LAC 与 softmax 加权，不代表论文实验。

设 query 的未来动作 chunk 是二维向量 a_q = [1.0, 0.0]。取 k=3 个邻居：

| 邻居 | 距离 d | 动作 a_i | ||a_q - a_i||² |
|---|---:|---|---:|
| i1 | 0.5 | [1.1, 0.0] | 0.01 |
| i2 | 1.0 | [0.9, 0.1] | 0.02 |
| i3 | 2.0 | [0.2, 0.9] | 1.45 |

同任务 C(q) 有 4 条 memory entries：

| entry | 动作 | ||a_q - a_j||² |
|---|---:|---:|
| j1 | [0.0, 1.0] | 2.00 |
| j2 | [-0.5, 0.5] | 2.50 |
| j3 | [0.3, -0.2] | 0.53 |
| j4 | [0.8, 0.3] | 0.13 |

**验算 LAC@3**：

```
邻居平均动作距离 = (0.01 + 0.02 + 1.45) / 3 = 1.48 / 3 = 0.4933
同任务平均动作距离 = (2.00 + 2.50 + 0.53 + 0.13) / 4 = 5.16 / 4 = 1.29
LAC@3 = 1 - 0.4933 / 1.29 = 1 - 0.3824 = 0.6176
```

解释：邻居动作明显比同任务随机配对更接近 query，所以 LAC 为正且较高。

**验算 softmax 加权**（假设 τ=1.0，d 已归一化）：

```
exp(-0.5)=0.6065, exp(-1.0)=0.3679, exp(-2.0)=0.1353
归一化后 α = [0.5465, 0.3315, 0.1219]

π(q) = 0.5465·[1.1,0.0] + 0.3315·[0.9,0.1] + 0.1219·[0.2,0.9]
     = [0.9239, 0.1429]
```

预测不是直接复制最近邻，而是三个未来 chunk 的加权平均；τ 越小越接近最近邻，τ 越大越平均。

**时间跨度验算**（假设 Δt=0.2 s）：

```
K_f=5 个历史帧：首末跨度 = (5-1)×0.2 = 0.8 s
H=8 个未来步：终点距当前 = 8×0.2 = 1.6 s
```

这里 0.8 s 是**已观测历史**，1.6 s 是**未来预测**；不能把 1.6 s 说成 history 窗口，也不能把 0.8 s 说成 action chunk 长度。

## 4. 工程视角 (Engineering View)

| 工程点 | 论文事实 | 条件性分析 |
|---|---|---|
| memory 规模 | 1693 条示范，每隔一帧，约 137k entries | 存储随相机数、特征维度、action chunk 维度线性增长；需离线构建 |
| 检索成本 | 论文未报告延迟/吞吐 | 暴力 kNN 每 query 约 O(N·D)；若 D=512，137k×512≈70M 乘加，10 Hz 约 700M/s，GPU 通常可行，但必须实测 |
| 执行频率 | ResNet 执行完整 chunk 后重检索；multimodal 每个 control step 重规划 | chunk 越长，查询越少但反应越粗；temporal ensembling 增加计算与缓存 |
| 动作空间 | X-VLA 预测绝对位姿；其他模型动作空间不同 | 不能跨模型直接比较动作距离；论文的 action-space 分析排除 X-VLA |
| 记忆归属 | 只有 kNN baseline 有外部 retrieval memory | 不能把 137k entries 说成 VLA 的内在记忆或所有 VLA 共有的记忆窗口 |
| 新任务 | 需要新 demonstrations | retrieval 不能外推到 memory support 之外；新物体/新轨迹是明确局限 |
| LAC key | 只训练线性投影 | 比训练完整 policy 轻，但仍不是完全 training-free；论文未在可见正文给出训练轮次/时长 |

部署含义：如果任务和示范覆盖密集，kNN 可以作为强 baseline 和快速 sanity check；如果任务需要新轨迹、精细力控或真实机器人闭环，retrieval 的覆盖不足会先于“策略能力”成为瓶颈。

## 5. 数据与评测 (Data & Eval)

### 5.1 预训练与 checkpoint

本文**不重新预训练 VLA**。九个 VLA 均以公开 LIBERO fine-tuned checkpoint 评估；论文 Table 4 给出每个模型的 Hugging Face repository 与 revision，GR00T N1.7 每个 LIBERO suite 一个 checkpoint。论文指出部分复现成功率低于作者原报告（如 π0、SmolVLA），可能反映公开 checkpoint 与原论文 checkpoint 的差异。

kNN 侧：视觉用 frozen ImageNet ResNet-18；语言用 unit-normalized CLIP；VLA retrieval key 用各模型 image-token hidden states 的某一层，层按 held-out 动作预测 R² 选择，使 fine-tuning 前后用同一层。LAC key 只训练一个线性投影。论文没有逐一说明九个 VLA 的预训练数据开放范围，**不能把 LIBERO fine-tuning 数据当成它们的预训练数据**。

### 5.2 仿真评测

LIBERO 共 130 个语言条件 Franka 任务、5 个 suite；本文评测 Spatial、Object、Goal、Long 四个 suite，共 40 个任务，使用 1693 条训练示范。官方 protocol 每个任务 50 episodes，因此每个 policy 2000 次 rollout。LIBERO-Plus 引入 7 类扰动；本文选 420 个 variant（每类 60 个），跨所有 policy 共享，且 held out from training and hyperparameter tuning。

**Table 1 关键结果**（论文实测，单位 %）：

| 策略 | LIBERO avg | LIBERO-Plus avg | Δ |
|---|---:|---:|---:|
| π0.5 | 97.7 | 85.5 | -12.2 |
| MolmoAct2 | 97.5 | 82.6 | -14.9 |
| GR00T N1.7 | 94.9 | 80.2 | -14.6 |
| VLA-JEPA | 97.2 | 74.0 | -23.1 |
| Evo-1 | 95.4 | 71.4 | -24.0 |
| X-VLA | 94.5 | 64.5 | -30.0 |
| π0 | 71.9 | 42.9 | -29.0 |
| SmolVLA | 82.2 | 27.9 | -54.3 |
| π0-FAST | 90.6 | 25.7 | -64.8 |
| kNN (ResNet) | 88.7 | 75.5 | -13.2 |
| kNN (multimodal) | 91.7 | 82.4 | -9.3 |
| kNN (LAC key) | 93.0 | 81.7 | -11.3 |

作者主张：multimodal kNN 在 LIBERO 上与 π0-FAST 同级，并高于 SmolVLA、π0；在 LIBERO-Plus 上超过 9 个 VLA 中的 7 个，且降幅最小。VLA-JEPA、Evo-1、X-VLA 在 LIBERO 超过 94%，但在 Plus 上低于 training-free 的 ResNet 变体（75.5%），说明 LIBERO 高分不自动转化为扰动鲁棒性。

**关键消融**（multimodal 基线 91.7%）：

| 消融 | avg | Δ |
|---|---:|---:|
| 忽略语言 prompt | 64.6 | -27.1 |
| 原始 16×16 像素替代 embedding | 57.2 | -34.5 |
| 只用 10% 示范 | 77.4 | -14.3 |
| k=1，只回放最近示范 | 78.6 | -13.1 |
| DINOv2-S 替代 ResNet-18 | 83.4 | -8.3 |
| 只用 wrist camera | 87.5 | -4.2 |
| 普通加权平均 | 90.7 | -1.0 |
| 无 temporal ensembling | 91.1 | -0.6 |
| 无 history / 无 proprio / 无 time-step | 91.3–91.5 | -0.2 至 -0.4 |

论文报告 95% 置信区间约 ±1.2 个百分点（2000 episodes）；上述小改动多在统计噪声内。k 在 4–32 之间时成功率 89.4–92.7%，k=1 掉到 78.6%。超参用 Optuna 搜了 24 个 trial，每个 400 episodes；调参用每个任务 initial states 40–49，未参与调参的 0–39 上 multimodal 为 91.6%，全 50 个状态为 91.7%，说明调参没有明显抬高结果。

### 5.3 真实机器人设置

本文**没有真实机器人实验**。所有结果在 LIBERO/LIBERO-Plus 仿真中；作者在结论中明确承认这一局限。

### 5.4 数据开放范围与摘要差异

论文 Table 4 列出九个 VLA 的公开 checkpoint 与 revision；作者称 evaluation 与 source code 已公开（见论文摘要），本笔记不展开外链。数据开放范围应逐模型理解：**checkpoint 公开不等于预训练数据公开**；论文没有逐一说明每个模型的预训练数据是否开放。

摘要差异：候选摘要的“91.7% without policy training”对应 multimodal kNN；正文还报告 ResNet 88.7% 和 LAC key 93.0%，后者含线性 key 训练。候选摘要未列这些区分，也未列 fine-tuning 表征分析；本文只以正文为证据。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

| 能力 | 证据 | 边界 |
|---|---|---|
| 用 retrieval 复现 LIBERO 高分 | multimodal 91.7%，ResNet 88.7% | 只在 40 个任务、1693 条示范；示范覆盖密集 |
| Plus 扰动下相对稳 | multimodal 82.4%，Δ=-9.3 为最小 | 扰动保留 underlying task/solution trajectories，不是新轨迹 |
| 诊断 VLM backbone | fine-tuning 后 5/8 模型在 LIBERO、7/8 在 Plus 上 retrieval 提升 | 不是所有模型；π0 backbone 从 86.9% 掉到 17.9%；GR00T VLM frozen，before=after |
| 离线诊断 | 排除在 held-out demonstrations 上训练过的 fine-tuned VLM backbones 后，LAC 与 closed-loop retrieval success 的 Pearson r=0.78 | 同一排除条件下 Spearman ρ=0.61，不能可靠排序质量相近的 key；不能无条件推广到所有 key |
| LAC key 提升 | ResNet kNN 从 88.7% 到 93.0% | key 有训练，不是完全 training-free；不是完整 policy |

**不能做什么**：

- 不能外推到 memory support 之外：新物体、新布局、需要新轨迹的任务，retrieval 没有对应示范。
- 不能证明真实机器人泛化：论文只做仿真。
- 不能证明“VLA 只会 kNN”：论文只说明 LIBERO 类密集示范 benchmark 中，retrieval 能解释很大一部分成功；VLA 的动作接近示范，不等于所有场景都如此。
- 不能把 kNN 的 137k memory 当作 VLA 的内在记忆，也不能把某个模型的 action chunk 长度当作所有实验共有的记忆窗口。

### 6.1 隐含假设 (Hidden Assumptions)

1. **测试观测落在示范支撑附近**：LIBERO 初始状态和示范覆盖密集，retrieval 才强；稀疏示范下不成立。
2. **扰动不改变任务解法**：LIBERO-Plus 主要改观测/指令，underlying task 和 solution trajectory 大体保留；需要新轨迹的扰动未被覆盖。
3. **公开 checkpoint 代表原模型**：论文使用 as-released checkpoint，部分复现成功率低于作者报告，说明 checkpoint 差异会影响结论。
4. **动作 chunk 时间对齐**：检索到的示范 chunk 与当前 query 的时间阶段要可对齐；否则“近邻状态”不等于“近邻未来动作”。
5. **LAC 是成功的线性代理**：排除在 held-out demonstrations 上训练过的 fine-tuned VLM backbones 后，LAC 与 closed-loop retrieval success 总体相关（Pearson r=0.78），但 Spearman 只有 0.61，不能替代闭环评测。
6. **调参没有严重泄漏**：作者用 initial states 40–49 调参，并在 0–39 上检查 91.6% vs 91.7%；这缓解但不消除 benchmark 过拟合的担忧。

## 7. 与相关工作对比 (Comparison)

| 方法 | 关注点 | 是否训练 policy | 是否直接输出动作 | 与本文差异 |
|---|---|---|---|---|
| VINN | kNN 视觉模仿 | 否 | 是 | 用学习视觉特征做最近邻；本文用 action chunk、多模态距离和 LAC |
| Behavior Retrieval | 从离线数据选轨迹 | 是（下游 policy） | 否 | retrieval 用于数据选择，不直接当 policy |
| FlowRetrieval | 光流相似度选示范 | 是 | 否 | 用于 few-shot 数据选择 |
| STRAP | 检索子轨迹增强训练 | 是 | 否 | 不是纯 lookup |
| REGENT / RICL / R+X | 检索作为 inference-time context | 不统一（REGENT 给 learned agent 提供经验；RICL 用 pretrained VLA；R+X 为 in-context imitation） | 否（由学习/预训练策略产生动作） | retrieval 给策略提供上下文，不直接产生动作 |
| He et al. lookup table | 解释 diffusion policy 的动作记忆 | 否 | 是 | 关注 diffusion policy；本文关注 VLA benchmark 与 backbone 诊断 |
| LIBERO-Plus / LIBERO-PRO | 鲁棒性/记忆评测 | 不适用 | 不适用 | 本文补上 retrieval baseline 和 LAC 指标 |
| 本文 | retrieval 作为 baseline + 探针 | 否（LAC key 只训练线性 retrieval key，不训练 policy） | 是 | 用 kNN 量化 benchmark 成功，并逐模型诊断 VLM backbone |

**面试 Tip**：被问“LIBERO 95% 是不是虚高”，先答数字：这篇的 multimodal kNN 在 LIBERO 91.7%、Plus 82.4%，降幅 9.3 个百分点为最小；再补边界：只说明 LIBERO 这类密集示范 benchmark 可被 retrieval 解释很多，不证明真实机器人泛化，也不证明所有 VLA 只会 kNN。

## 8. 精读建议 (Reading Guide)

- **值得精读原文的人**：
  - 做 VLA 评测/基准的人：需要设计 retrieval baseline、held-out 划分和扰动评测。
  - 做 retrieval/non-parametric policy 的人：关心多模态距离、每 demo 唯一约束、temporal ensembling。
  - 想诊断 VLM backbone 的人：关心 LAC、fine-tuning 前后 retrieval key 变化。
- **建议章节路径**：先读 §3 + §5 + Table 1（机制与评测）→ 再看 §6.1/§6.3/§6.5（结果与诊断）→ 需要复现时看 Appendix A.5 与 Appendix B（消融与逐模型设置）。Related Works 可在熟悉背景后跳读。
- **不值得精读的理由**：如果你不做 LIBERO、不关心仿真 benchmark，或只关心真实机器人新轨迹泛化，读摘要和 §6 的失败模式即可；论文没有真实机器人实验，也没有 memory support 之外的证据。

---

[← Back to Theory](./README.md)

**关键引用**：[How Much of Your VLA is kNN, arXiv:2611.13160](https://arxiv.org/abs/2611.13160)