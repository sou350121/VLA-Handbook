# VLA 该多大程度信任自己的推理？逐步引导作为可测量的动作价值函数 (How Much Should a VLA Trust Its Reasoning? Step-Level Guidance as a Measured Action-Value Function)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-11-01
>
> **论文**: How Much Should a VLA Trust Its Reasoning? Step-Level Guidance as a Measured Action-Value Function
> **链接**: https://arxiv.org/abs/2610.40002
> **核心定位**: 不再问“推理 trace 对不对”，而是把“推理该多大程度影响动作解码”变成每个 Euler step 在有限动作集上选取的离散权重，并用可重置生成的一步分支精确测量该权重的动作值。
> **证据版本**: 本次输入为 arXiv:2610.40002v1 的 HTML 全文（2026-10-29），未见官方 PDF。候选元数据摘要为截断片段，未覆盖实验数字，未用于补写；正文表 1–4 的关键数字与摘要中出现的 16.5%、262/300、8.135159 vs 8.079 一致。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 把 reasoning 的信任度建模为逐 Euler step 的离散权重，用 one-step branch 测得的终端分数训练保守控制器；在 Alpasim 和 Waymo 离线评测上超过固定权重与 SAC/GRPO。 |
| 适合精读 | 做 reasoning VLA、flow/diffusion policy guidance、test-time control 的读者，重点看 §3–§4 与 §5.1–§5.2。 |
| 可以跳过 | 只关心 VLA 预训练数据配比、真实机器人闭环或非驾驶域迁移的读者，这篇距离中等。 |
| 落地可行性 | 中：backbone 冻结，在线只需 critics 前向；但离线需要可重置 sampler 与可评分训练场景，跨域未验证。 |
| 主要风险 | 只在 Alpamayo 1.5 + 驾驶域验证；Alpasim 是神经重建仿真，Waymo 是离线 benchmark；hidden test 是官方提交并由 benchmark 组织方评分，但缺独立复现与真实机器人闭环。 |

💡 **X-Ray 开场**

这篇论文解决的是：推理型 VLA 写出的 trace 可能错，也可能“看起来对但把计划带偏”；现有方法只判断 trace 对不对，本文问“它该多大程度影响动作”。它发现，如果把 classifier-free guidance 的权重按 Euler step 调节，并利用冻结 expert + 固定噪声的可重置性，就能把“某一步改用某个权重”的后果精确测成最终分数。对 VLA 研究者，这意味着推理 trace 可以从“要检测的缺陷”变成“可控制的测试时变量”。

📍 **研究全景时间线**

```
2022  CFG（Ho & Salimans）[10]
  → 2023  flow matching（Lipman et al.）[19]
  → 2025  Alpamayo-R1 等推理型驾驶 planner [23]
  → 2026  WOD-E2E 基准 [32] / 本文 v1（2026-10-29）
```

本文位置：在 CFG 与 flow matching 之后，把 guidance 权重从“固定或调度超参”变成“逐 Euler step 的可测控制动作”。局限：目前只在驾驶 VLA 上验证，不能直接外推到机械臂、人形或通用具身 Agent。

## 1. 核心架构/方法总览

### 1.1 系统对比概览

| 对象 | 架构/角色 | 输入→输出 | 时序/频率 | 训练/推理与开放范围 |
|------|-----------|-----------|-----------|---------------------|
| Alpamayo 1.5 VLM | 写 CoC 的 VLM | 观测+指令→reasoning trace，写入 KV cache C | 每次生成 | 本文冻结；预训练细节未展开；权重许可未声明 |
| Alpamayo 1.5 action expert | flow-matching expert | (x_k, t_k, C)→velocity；K 个 Euler step 后解码 plan | K Euler steps；Waymo 明确 K=10，Alpasim 正文未给 K | 本文冻结；不更新 |
| CoC-CFG | guidance 轴 | C_full / C_ref、w_k→guided velocity | 每个 Euler step 一个 w_k | 无训练；w=1 精确恢复原策略 |
| Q-adaptive critics | 小型 critics | s_k→各动作的 Q 值 | 推理时每个 Euler step 前向一次 | 离线用 measured branch 标签训练；参数量、动作集大小未报告 |
| Alpasim | 神经重建仿真器 | 场景→闭环 rollout | 10 Hz 控制；10 s；每 25 控制步一次推理 | 不是策略模型；数据公开性未说明 |
| Waymo benchmark | 数据/评测协议 | 帧→RFS | 离线；hidden test 由 organizers 评分 | 不是模型；访问受 benchmark 协议 |
| Alpamayo-R1 | 引用中的相关 planner | — | — | 本文未用于实验；不能把 1.5 的结果归给它 |

**记忆/上下文归属**：Alpamayo 1.5 的 C 是当前生成的 VLM KV cache，包含观测、指令和 reasoning trace；reasoning span 是 C 中一段连续 token。本文没有提出跨 episode 长期记忆，也没有把 K 个 Euler step 当作历史帧或记忆窗口。K 是去噪伪时间步，不是物理时间步。

### 1.2 关键机制

- **只调 reasoning span 的影响**：C_ref 只移除 reasoning span，观测、指令、prompt 格式和剩余条目的位置都保留。这样 guidance 的差值只归因于 trace，而不是整体 context 变了。
- **逐 Euler step 一个权重**：w_k 可以衰减（小于 1）、恢复原模型（等于 1）或放大（大于 1）推理的影响。它不是整条 trace 一个全局权重，而是解码过程中每一步一个决策。
- **权重选择是有限时域 MDP**：状态 s_k=(k, x_k, w_{k-1}, C)，动作是 w_k，转移是 guided Euler step，终端奖励是 plan 的分数 R(x_K)。
- **值可测而非 bootstrap**：expert 冻结、初始噪声固定，生成可重置到任意 step。对每个动作 a 做一步分支，再按当前控制器 π 完成，得到 G^π(s_k,a)。这不是从后续估计自举，而是直接测量。
- **保守拟合**：目标里加入离开 anchor 的固定成本和比例成本，并用 clip 截尾；多阶段标签按年龄折扣，避免控制器跑到没测过的状态太远。
- **推理时不做 branch rollout**：只对 critics 做前向，取带成本的 argmax；因此在线成本主要来自每个 Euler step 的 critic 前向，而不是完整分支。

⚡ **Eureka Moment**：把“删不删推理”变成“每个 Euler step 给推理多少权重”，并且因为生成可重置，这个权重的作用可以被精确测成动作值。

### 1.3 信息流/架构图

```
观测 + 指令
   │
   ▼
VLM ──► reasoning trace ──► KV cache C_full
                               │
                 去掉 reasoning span，其余不变
                               ▼
                            C_ref
                               │
x_0 ~ noise ──► Euler step k: v_full, v_ref
                               │
        x_{k+1}=x_k+Δt·(v_ref+w_k·(v_full-v_ref))
                               │
                         K 步后 x_K ──► plan ──► R(x_K)

离线测量：
  reset 到 s_k
  对每个 a∈𝒜：应用 a 一步 → 按 π 完成 → 得分 G^π(s_k,a)
  与 anchor 分支配对 → A(s_k,a)=G^π(s_k,a)-G^π(s_k,a0)
  → 拟合 Qbar

推理：
  w_k=argmax_{a∈𝒜}( Qbar(s_k,a)-η·|a-a0| )
```

## 2. 数学核心

📌 **Napkin Formula**

```
信任度不是二值，而是每个 Euler step 的 w_k；w_k 的作用用一步分支的终端分数精确测。
```

**目标**：在训练场景分布上最大化终端分数，同时保持控制器不偏离 anchor 太远。

```
max_{π} E_scene[ R(x_K) ],   w_k=π(s_k)∈𝒜
```

**CoC-CFG 的 Euler step**：

```
x_{k+1}=x_k+Δt·( v_ref + w_k·(v_full-v_ref) )
v_full=v(x_k,t_k,C_full)
v_ref =v(x_k,t_k,C_ref)
Δt=1/K
```

当 w_k=1 时，括号内变为 v_full，恢复原策略；w_k=0 时只用 C_ref；w_k 在 0 与 1 之间衰减，大于 1 放大。

**MDP 定义**：

```
s_k=(k, x_k, w_{k-1}, C),  w_{-1}:=a0
a=w_k
s_{k+1}=Euler(s_k,a)
reward=0 (k<K),  R(x_K) (k=K)
```

**一步分支测得的优势**：

```
G^π(s_k,a)=R(x_K)  | 在 k 应用 a，然后按 π 完成
A(s_k,a)=G^π(s_k,a)-G^π(s_k,a0)
```

**critic 拟合目标**：

```
y(s,a)=clip( A(s,a)-λ_c·1[a≠a0]-λ_m·|a-a0|, -τ, τ )
```

**多阶段损失**：

```
L_n=Σ_{j=1..n} ρ^{n-j}·L_{D_j},  0<ρ≤1
```

**部署规则**：

```
w_k=argmax_{a∈𝒜}( Qbar(s_k,a)-η·|a-a0| )
```

| 符号 | 含义 | 本文口径 |
|------|------|----------|
| C_full | 完整 context | 含 reasoning trace 的 VLM KV cache |
| C_ref | 去掉 reasoning span 的 context | 其余不变 |
| K | Euler 解码步数 | Waymo 为 10；Alpasim 正文未给 |
| w_k | 第 k 个 Euler step 的 guidance 权重 | 有限动作集 𝒜 中的元素 |
| a0 | anchor 权重 | Alpasim 为训练场景上最好的固定 w=0.5；Waymo 为默认 w=1 |
| R(x_K) | 终端分数 | Alpasim 为负 displacement cost；Waymo 为 RFS |
| A(s,a) | 相对 anchor 的优势 | 同状态、同 continuation 下的测量差值 |
| λ_c, λ_m | 离开 anchor 的固定/比例成本 | 论文未给具体数值 |
| τ | 标签 clip 阈值 | 论文未给具体数值 |
| ρ | 旧标签折扣 | 论文未给具体数值 |
| η | 部署时离开 anchor 的惩罚 | 论文未给具体数值 |

**直觉**：A(s,a) 是“这一步改用 a 而不是 anchor，最终分数差多少”。因为 anchor 分支与 a 分支从同一状态出发，场景难度被差值消掉。无 bootstrap 意味着标签是测量，但只对当前 continuation π 精确；π 改进后旧标签会“变旧”，所以用 ρ 折扣。成本项则让控制器在没测过的状态上不要过度偏离 anchor。

## 3. 带数字走一遍：玩具例子

> **【假设例子，不是论文数据】** 以下数值仅用于走通公式。

假设 K=3，Δt=1/3，x_0=0，v_ref=1.0，v_full=1.4 恒定。

- w=1：每步增量=1/3×1.4=0.4667，x_3=1.4。
- w=0.5：每步增量=1/3×(1.0+0.5×0.4)=1/3×1.2=0.4，x_3=1.2。
- w=0：每步增量=1/3×1.0=0.3333，x_3=1.0。

可见 w 在 0 到 1 之间时，plan 落在“去掉推理”和“原模型”之间；大于 1 会推得更远。这只是单步混合的算术，不代表真实 flow field。

再看值标签。假设某状态 s_k，a0=1.0，候选 a=0.5。按 π 完成后：

- a=0.5 的分支 R=8.0；
- anchor a0=1.0 的分支 R=7.0；
- 则 A(s_k,0.5)=8.0-7.0=+1.0。

设 λ_c=0.2，λ_m=0.1，τ=5：

```
y=clip(1.0-0.2-0.1×|0.5-1.0|, -5, 5)
 =clip(1.0-0.2-0.05, -5, 5)
 =0.75
```

部署时设 Qbar(s_k,0.5)=0.8，Qbar(s_k,1.0)=0.0，η=0.1：

```
score(0.5)=0.8-0.1×0.5=0.75
score(1.0)=0.0-0.1×0=0.0
→ 选 0.5
```

若 Qbar(s_k,0.5)=0.04，则 score(0.5)=0.04-0.05=-0.01，小于 0，控制器会退回 anchor。这说明成本项确实会阻止“收益不够大”的偏离。

## 4. 工程视角

**离线测量成本**：每个 visited state 要对 |𝒜| 个权重各做一次 one-step branch，每个 branch 都要完成剩余 K 步生成并评分。若粗略写成：

```
branch 数 ≈ N_scene × K × |𝒜| × n_stage
```

论文给 N=120 个训练场景/帧，Waymo K=10，但未给 |𝒜|。若仅借用表 3 的 7 个固定权重作量级示意（**假设，不是论文动作集**），单阶段约 120×10×7=8400 次完成生成。这是条件性估算，不是论文报告的吞吐。

**在线成本**：推理时不做 branch，只对 critics 做前向，每个 Euler step 一次，取 argmax。critics 是“small critics”，但论文未给参数量、FLOPs 或延迟，不能量化。关键工程约束是：必须能访问 x_k、C 和当前 w_{k-1} 的特征；必须能按 Euler step 插入控制器。

**时序口径**：这是本文最容易被误读的地方。

| 量 | 值 | 含义 | 不能误读为 |
|----|----|------|------------|
| K 个 Euler steps | Waymo 为 10；Alpasim 正文未给 | flow matching 去噪步 | 10 帧历史、10 步未来、记忆窗口 |
| 控制频率 | Alpasim 10 Hz | 闭环控制步 | Euler step 频率 |
| rollout 时长 | Alpasim 10 s | 约 100 个控制步 | 训练轮次 |
| 推理间隔 | Alpasim 每 25 控制步一次 | 按 10 Hz 可推得约 2.5 s 一次推理 | 所有实验共有的动作 chunk 长度 |
| 训练场景/帧 | 120 | 控制器训练数据 | epoch 数 |
| stage n=1..4 | 控制器迭代阶段 | 重新测量+拟合的轮次 | 训练 epoch |

如果按均匀间隔采样历史帧，K 个采样帧的首末跨度是 (K-1)×Δt；若从当前参考 t 预测 t+1 至 t+H，终点距当前参考是 H×Δt。本文的 K 是 Euler 去噪步，不适用这两个换算；不能把 K=10 写成“10 帧历史”或“10 步动作 chunk”。Alpasim 的“每 25 控制步一次推理”也只是该仿真实测的设置，不是 Waymo 或所有实验共有的推理间隔。

**内存与分支**：C_full 与 C_ref 两个分支都要保留 KV cache；C_ref 去掉 reasoning span 后 token 数可能减少，但论文未给显存或 cache 大小。工程上要把“推理 span 的 token 边界”作为一等公民维护，否则 C_ref 构造会改变位置编码或 prompt 格式，破坏反事实解释。

**Trade-off**：
- 动作集 𝒜 越细，控制分辨率越高，但离线 branch 数和在线 critic 输出维度越大；论文未给 𝒜 大小。
- anchor 成本越大，越保守，越不容易在未测状态上崩，但可能放弃有效增益。
- 单次 rollout 的闭环保留了碰撞、离线距离等指标，但论文 closed loop 是 one rollout per scene and arm，未报告多 seed 方差。

## 5. 数据与评测

### 5.1 模型与预训练

本文实验主体是 **Alpamayo 1.5**：VLM 写 CoC 到 KV cache，flow-matching action expert 从该 cache 解码轨迹。本文**不训练 VLM 和 expert**，只训练 Q-adaptive guidance 的 critics。论文没有展开 Alpamayo 1.5 的预训练数据配比、参数规模、训练轮次或许可；不能从本文推断这些细节。Alpamayo-R1 只是引用中的相关 planner，本文未用于实验，不能把 1.5 的结果归给它。

**数据开放范围**：本次来源未提供代码仓库；论文未声明 Alpamayo 1.5 权重、Alpasim 重建数据或训练标签的公开许可。Waymo 数据受 benchmark 协议约束。不能从本文推断可复现性。

### 5.2 Alpasim：仿真开环与闭环

Alpasim 是神经重建的真实驾驶仿真器，不是真车。开环评测把 Alpamayo 1.5 的 plan 与 logged future 对比，不重新规划；分数是负 displacement cost，由 3 s 和 5 s 的平均/最终位移误差加权求和并除以固定尺度。

| 设置 | 数据与条件 | 指标 | 关键结果 |
|------|------------|------|----------|
| 开环 | 120 训练场景，358 held-out 场景 | 负 displacement cost，越高越好 | w=1: -1.1704；best fixed w=0.5: -1.0495（+10.3%）；SAC: -1.0468（+10.6%）；GRPO: -1.0091（+13.8%）；ours stage 3: -0.9767（+16.5%） |
| 闭环 | 300 个训练池外场景；10 s @ 10 Hz；ground-truth warm-up；每 25 控制步一次推理；每场景每 arm 一次 rollout | 通过数、at-fault collision、距 log 最大距离、composite score | w=1: 252/300，13 次碰撞，1.232 m，0.695；anchor w=0.5: 254/300，10 次，1.272 m，0.691；ours: 262/300，7 次，1.118 m，0.712 |

开环中 stage 3 最好（-0.9767），stage 4 略差（-0.9769），所以闭环保留 stage 3。闭环中 ours 在通过数、碰撞数、距 log 距离和 composite score 四项均最好；anchor 的通过数略高于 w=1，但 score 略低，说明不同指标不完全同向。

### 5.3 Waymo：真实数据离线评测

Waymo Open Dataset end-to-end driving benchmark 使用真实驾驶数据，但不是真车闭环。RFS 是 11 个 per-category 分数的平均。478 个计分 validation frames 中，358 个 held-out 用于所有控制器对比，120 个供训练。Alpamayo 1.5 在 Waymo 上用 K=10 个 Euler steps 解码；由于它的训练相机 rig 和车辆与 Waymo 不同，论文用 rule-based camera 和 vehicle conversions 做映射，VLM 与 expert 保持冻结，anchor 为默认 w=1。

| 设置 | 数据与条件 | 指标 | 关键结果 |
|------|------------|------|----------|
| Held-out | 120 训练帧，358 held-out 帧；所有 arm 同一套 conversions | RFS，越高越好 | best fixed w=0.75: 8.2412；default w=1: 8.2363；SAC: 8.2783；GRPO: 8.2493；ours stage 2: 8.3032（比 best fixed +0.0620，比 default +0.0669） |
| Hidden test | organizers 评分；作者提交 stage 2 | RFS | Alpamayo 1.5 + Q-adaptive guidance: 8.135159；表 4 中最高已发表对比方法 DriveMA-4B: 8.079 |

表 3 中，ours 的 stage 1/2/3 分别为 8.2917/8.3032/8.2922，均高于所有固定权重、四种 adaptive guidance、reward fine-tuning、SAC 和 GRPO；stage 4 为 8.2780，略低于 SAC 的 8.2783，所以论文选 stage 2 上 hidden test。Table 3 还包含 scene-level Q（每场景一个 w）8.1105，以及四个 stage 平均 8.1972 的对比口径；这些数字不能混用。

**作者主张**：hidden test 的 8.135159 是“published methods 中的新 SOTA”。这是作者按表 4 汇总的其他论文报告值做的比较，不是独立复现；表 4 明确说每篇论文取最佳报告变体，四舍五入到三位。该分数是官方 hidden-test 提交并由 benchmark 组织方评分，但缺少独立复现，不能概括为“不是第三方验证”。

### 5.4 真实机器人设置

本文**没有真实机器人闭环实验**。Alpasim 是神经重建仿真；Waymo 是真实驾驶数据上的离线 benchmark。两者都不能替代真车或真实机械臂的闭环验证。

**历史与未来口径**：输入 C 来自当前已观测的观测和指令，输出是未来 plan；Alpasim 开环对比 logged future，闭环在重建中按实际位姿重渲染；Waymo RFS 对比人类评分的未来轨迹。论文未给历史帧数；358 是评测帧数，不是历史窗口。K=10 是未来 plan 的去噪步数，不是未来 10 帧。

## 6. 能力与失败模式

| 能做 | 证据 | 不能做/未验证 |
|------|------|----------------|
| 在驾驶域逐 Euler step 衰减/放大 reasoning 的影响 | Alpasim 开环 +16.5%；闭环保留 262/300；Waymo held-out 8.3032；hidden 8.135159 | 不能判断 trace 本身对不对；不替代 hallucination detection |
| 在 backbone 冻结下做 test-time control | VLM/expert 冻结，只训练 critics | 不能跨到未验证的机器人平台或非驾驶域 |
| 用一步分支测出权重作用 | 无 bootstrap，标签来自可重置生成 | 需要可评分训练场景；R 与真实安全不一致时会优化错目标 |
| 在固定权重和 SAC/GRPO 之上取得更好分数 | 表 1、表 3 | hidden test 为作者提交，缺独立复现；closed loop 单次 rollout 方差未量化 |

**具体失败模式**：
- C_ref 去掉 reasoning span 后可能分布外，v_ref 不可靠；如果 span 边界或位置编码处理不当，guidance 差值不再只归因于 trace。
- 论文假设 reasoning span 是一段连续 token；多段推理、交错推理或工具调用痕迹可能需要多组权重，当前单一连续 span 的全局权重不够细。
- 评分 R 是终端奖励；如果 RFS 或 displacement cost 与真实安全不一致，控制器会优化错目标。闭环保留碰撞指标，但训练目标本身不是碰撞约束。
- 单步 branch 的标签依赖 continuation π；π 改进后旧标签有偏差，ρ 折扣只是缓解，不是无偏纠正。
- 动作集 𝒜 太粗时，权重分辨率不足；太大时离线 branch 成本上升。论文未给 𝒜 大小和超参数，复现需自行扫。
- 真实部署中若初始噪声不固定、expert 更新或 sampler 不可重置，“测量值”不再精确，方法退化为近似值估计。
- closed loop 每场景每 arm 只跑一次 rollout；通过数的差异可能受单次方差影响，论文未报告重复实验。

### 6.1 隐含假设

1. reasoning trace 占据 C 的一个连续 span，移除它后其余位置和 prompt 格式不变，C_ref 是合理反事实。
2. expert 冻结且初始噪声固定，生成 deterministic 且可 reset；真实系统中这两个条件不一定成立。
3. 训练场景的 R 能代表部署目标；Alpasim 的负 displacement cost 与 Waymo RFS 是不同目标，两个设置分别训练 critics，不能互相替代。
4. 一步分支后用当前 π 完成，能代表“应用该权重后的值”；这只对 π 精确，不是最优 Q*。
5. 终端分数足够反映计划质量；中间过程的安全约束没有被显式建模。
6. 动作集和超参数“per backbone”固定；不能把 Alpasim 的 anchor=0.5 或动作集直接搬到 Waymo 的 anchor=1。

## 7. 与相关工作对比

| 方法族 | 权重/条件 | 是否更新 backbone | 训练信号 | 本文差异 |
|--------|-----------|-------------------|----------|----------|
| 固定 CFG 权重 | 全程一个 w | 否 | 无 | 本文逐 Euler step 选 w |
| CFG++ / linear ramp / guidance interval / rescale | 调度或改进 guidance | 否 | 超参在训练帧上扫 | 本文把 w 当离散动作，用测量值训练控制器 |
| SAC / GRPO | 学每步权重 | 否 | 任务分数，在线 RL | 本文不做 bootstrap，直接回归测量值 |
| scene-level Q | 每场景一个 w | 否 | 每个固定权重的场景分数 | 本文每 Euler step 一个 w |
| reward fine-tuning | 更新 expert | 是 | 奖励 | 本文冻结 expert，只控 guidance |
| 本文 Q-adaptive guidance | 每 Euler step 一个 w | 否 | one-step branch 的 measured advantage + anchor 成本 | 值可精确测量，控制器保守迭代 |

**面试 Tip**：被问到“CoC-CFG 和普通 CFG 有什么区别”，抓住三点：权重是逐 Euler step 的离散动作；值来自可重置生成的一步分支精确测量，不是 bootstrap；训练是带 anchor 成本的监督回归，推理时只做 critics 前向、不做 branch rollout。

## 8. 精读建议

- **值得精读原文的人**：
  - 做 reasoning VLA、flow/diffusion policy guidance、test-time control 的研究者：重点看 §3 的 CoC-CFG 轴和 §4 的 measured value 构造。
  - 要评估“冻结 backbone + 小控制器”能否迁移到新机器人平台的工程师：重点看 §5.1 的时序口径和 §6.1 的假设。
  - 做驾驶端到端规划、关心 reasoning trace 可控性的读者：重点看 §5.2 的表 1–2 和 §5.3 的表 3–4。

- **建议章节路径**：先读 §3（公式 1，理解 w_k 的作用）→ 再看 §4（MDP、式 2–4，理解为什么值可测）→ 然后 §5.1–§5.2（看 Alpasim 条件与数字）→ 最后 §6.1（检查假设是否适用于你的场景）。Related Work 可在熟悉 CFG/offline RL 后跳读。

- **不值得精读的理由**：如果你不做生成式策略 guidance、不关心驾驶域，或只关心 VLA 预训练数据与真实机器人闭环，这篇的证据边界较窄，读摘要和表 1–4 即可。

---

[← Back to Theory](./README.md)

论文链接：https://arxiv.org/abs/2610.40002

本次来源未提供代码仓库；未使用其他外部链接作为证据。