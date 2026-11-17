# 在仿真中训练推理模型以原生使用真实机器人（Training Reasoning Models in Simulation to Natively Use Real Robots）

> ⚙️ 本文由 Moltbot 自动生成 | 2026-11-17
>
> **论文**：Training Reasoning Models in Simulation to Natively Use Real Robots
> **链接**: https://arxiv.org/abs/2611.18562
> **核心定位**：把 robot-use agent（RUA）做成无 harness 的 pose-level 多轮推理策略，用仿真 RL 替代示范微调，试图让开源 27B 模型接近闭源 frontier RUA。

> 证据版本说明：本文依据 arXiv:2611.18562v1（cs.RO，2026-11-13）的 HTML 正文。候选摘要中的 60%、1.6%→74.4%、57.4% vs 0.75% 均可在正文核对，未发现摘要与正文冲突。但要注意：**57.4% vs 0.75% 是 LIBERO-Pro 的 task perturbation 子项，不是总平均；总平均是 62.1% vs 53.3%**。本文只以该正文为证据，不用候选摘要补写论文主张。
>
> 标记约定：【实测】论文表格/正文数字；【主张】作者解释或假设；【分析】本文编辑的条件性判断。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 用二值任务奖励在多轮仿真里做 RURL，Qwen3.8-27B 学会无 harness 输出 EE pose；LIBERO 从 1.6% 到 74.4%，零样本真机 18/30。 |
| 适合精读 | 如果你在做 RUA、agentic robot、sim-to-real、RL post-training，重点看 §3.1 接口、§3.2 算法与 PEAR、§4.4 + Appendix G 真机。 |
| 可以跳过 | 如果你只关心低层 VLA 动作生成、action chunk、视频预测或实时控制，这篇距离中等：它每轮只输出一个 pose。 |
| 落地可行性 | 中（研究原型）：接口简单、可仿真训练，但每轮长推理约 4.3 s、上下文消耗大、控制器依赖强、真机评测只有 30 trials/模型。 |
| 主要风险 | 真实机器人样本小、只覆盖两个任务族；长程任务常被 context 上限截断；π0.5 对比来自先前工作而非作者复现。 |

💡 **X-Ray 开场**

它解决的是：开源权重模型能不能像 computer-use agent 一样，直接“看相机—推理—发位姿”来操作真机。作者发现：只在仿真里做 RL、不给示范、不加 harness，27B 模型也能把 LIBERO 从 1.6% 提到 74.4%，并在未见过的双臂真机上完成 18/30 次试验。对 VLA 研究者，这意味着“推理模型 + 位姿接口 + 仿真 RL”可能是一条独立于 VLA 示范微调的后训练路线，但速度、上下文和真实评测规模仍是硬约束。

📍 **研究全景时间线**

```
2023  LIBERO 基准（Liu et al., 2023）
  │
2024  DROID 真实机器人数据集（Khazatsky et al., 2024）
  │
2025  π0.5：VLA + 示范监督（Black et al., 2025a）
  │
2026  GPT-6 Astra 作为 RUA 的评测；RoboDojo 基准（OpenAI, 2026b; Chen et al., 2026）
  │
2026-11  本文：RURL + PEAR，仿真 RL → 零样本真机  ← 当前位置
```

本文的位置：把 CUA 式多轮 RL 搬到机器人，强调“无 harness、无示范、仿真训练、真机零样本”。局限：只覆盖两个仿真器、少量任务、27B 基座，真机只有 30 trials/模型。

## 1. 核心架构/方法总览

### 1.1 系统对比概览

| 模块 | 输入 | 输出 | 时序/频率 | 训练/推理差异 |
|------|------|------|-----------|----------------|
| RUA policy（Qwen3.8-27B-PEAR） | 指令 + 当前相机帧 + 机械臂状态 + 历史轮次 | chain of thought + JSON 的 EE pose 增量 | 每轮一次；非固定 Hz，取决于解码与控制器 | RL 训练；评测用 BF16 |
| IK/控制器 | EE pose 增量 | 关节目标/运动 | 每轮执行到到位或阻塞 | 不学习；机器人特定 |
| 仿真器 | 动作 | 下一观测 + 成功信号 | 每轮 | 训练时执行 |
| PEAR | rollouts | token 序列与策略更新 | 异步 | 训练基础设施 |
| 奖励 | 整个 episode | 0/1 | episode 结束 | 二值成功奖励 |

关键点：动作是 **pose-level target**，不是 joint command；同一 JSON 协议在 LIBERO 的单臂 Franka、RoboDojo 的双臂 ARX、真实双臂 YAM 上都适用，因为 IK 和控制器在模型外面。论文明确说没有 perception tools、planner、skill library，也没有 harness。【实测/主张】

### 1.2 关键机制

- **接口统一，换机器人只换 prompt 里的部件和相机描述。** 模型每轮看到相机帧和机械臂状态，输出每个 named part 的位姿增量；控制器把目标执行到位或阻塞为止。
- **没有 harness。** 模型必须自己定位物体、决定怎么操作、发现动作失败。作者的论点是：harness 能帮开源模型完成部分任务，但可能限制动作空间，使另一些任务不可行。【主张】
- **RL only，没有 SFT、没有 demonstrations。** 奖励是 episode 结束时仿真器是否报告成功；训练用 GRPO 的组内标准化优势，配合 DAPO 的 group filter 和 score-centering。【实测】
- **课程学习是必需的。** 二值奖励下，全成功或全失败的组优势为零；基座在 LIBERO-Long 只有 0.4%，在多数 RoboDojo 任务几乎不成功，所以先训练有信号的 Goal/Spatial/Object，再逐步加入 Long 和 RoboDojo。
- **记忆就是 context。** 指令和之前所有轮次的 observation/response 都留在上下文；没有外部记忆模块。上下文长度是硬约束，不是可以随意扩大的“记忆窗口”。
- **异步长序列训练。** PEAR 构建在 veRL + LMCache 上，rollout 与更新全异步；rollout policy 可能落后 1–2 个版本，用 score-centering 校正，不用 PPO clipping。

⚡ **Eureka Moment**

把机器人控制降维成和 computer-use 一样的“看当前屏幕—推理—发一个位姿增量”接口，然后用二值任务奖励在多轮仿真里做 RL；接口不变，所以仿真里学到的策略可以零样本搬到真机。

### 1.3 信息流/架构图

```
训练：
任务指令 + 当前相机帧 + 机械臂状态 + 历史轮次
        │
        ▼
Qwen3.8-27B-PEAR ── CoT ──► JSON: ΔEE pose
        │                         │
        │                         ▼
        │                  IK + controller
        │                         │
        │                         ▼
        │                   simulator step
        │                         │
        └──── next obs + binary success ◄──┘
        （整段 episode 的 token 序列 → GRPO 更新）

部署：
同一协议 → 真实 YAM 的相机/状态 → JSON ΔEE pose → 真机 IK/controller
```

**先区分已观测历史帧与未来预测帧。** 本文输入的是已观测的当前相机帧、机械臂状态和之前轮次；输出是下一 EE pose。它不预测未来视频帧，也不输出未来动作 chunk。所谓 memory 是 context 中保留的 earlier rounds，不是未来 chunk 的预测区间。论文还明确说当前模型不能可靠规划 action chunks，所以每轮只输出下一个 pose。【实测】

概念校验（假设，不是本文实验）：若某视频方法均匀采 K 帧，首末跨度是 `(K−1)×Δt`；若从当前参考 t 预测 t+1 至 t+H，终点距 t 是 `H×Δt`。本文没有 K 帧采样和 H 步未来预测，不能把 19.7 轮平均或 256k token 上限套成帧窗口。

## 2. 数学核心

📌 **Napkin Formula**

```
max_θ E_τ[R(τ)]：二值成功奖励 + 组内标准化优势，把整段 episode 的信用分摊到每个 token
```

目标：在训练任务上最大化 episode 成功率的期望。

```
max_θ  E[R(τ)]，τ ~ πθ
R(τ) = 1  若 episode 结束时仿真器报告成功
R(τ) = 0  否则
```

GRPO 的组内标准化优势：

```
A_i = (R_i − mean_j(R_j)) / (std_j(R_j) + ε)
```

策略梯度估计：

```
g = (1/N) Σ_i A_i Σ_{t,k} [
      ∇θ log πθ(y_i,t,k | c_i,t,k)
    − E_{v~μ(·|c_i,t,k)} ∇θ log πθ(v | c_i,t,k)
    ]
```

| 符号 | 含义 |
|------|------|
| τ | 一个 episode：o1,y1,...,oT,yT |
| o_t | 第 t 轮的观测（相机帧与机械臂状态） |
| y_t | 第 t 轮的模型响应，含 reasoning 和 action |
| c_i,t,k | 第 i 个 episode、第 t 轮、第 k 个 token 的上下文 |
| N | 一个 batch 中生成的 token 总数 |
| A_i | episode 级优势，广播到该 episode 的每个 token |
| μ | 生成该 episode 的 rollout policy，可能落后 1–2 个版本 |
| ε | 小常数，防止除零 |

直觉：

- 二值奖励很粗，但组内比较提供了相对信用：同一任务、同一初始状态下，成功 episode 的所有 token 得到正优势，失败 episode 的所有 token 得到负优势。
- 如果一组全成功或全失败，`std=0`，优势为零，DAPO 式 group filter 直接丢弃并替换。因此训练依赖“混合成功/失败”的组。
- 第二项是 score-centering correction，用来修正异步 rollout 的 off-policy 偏差；论文没有用 truncated importance sampling，也没有 PPO clipping。
- 损失是 token-mean，没有 KL penalty、没有 entropy bonus；这使训练更接近“纯 REINFORCE + 组基线”。

> 论文没有展开 base model 的层数、注意力结构等细节；这里只确认它是 open-weight 27B reasoning model，Qwen3.8-27B。不要补写论文未给出的架构参数。

## 3. 带数字走一遍：玩具例子

> 以下为假设数值，用来验算 GRPO 的组内优势，不是论文实验。

假设一个组有 G=8 个 episode，共享任务和初始状态，奖励为：

```
R = [1, 0, 0, 0, 0, 0, 0, 0]
```

按总体标准差计算（假设，仅用于本例）：

```
mean(R) = 1/8 = 0.125
偏差 = [0.875, −0.125, −0.125, −0.125, −0.125, −0.125, −0.125, −0.125]
方差 = (0.875² + 7×0.125²) / 8
     = (0.765625 + 0.109375) / 8
     = 0.875 / 8
     = 0.109375
std = sqrt(0.109375) ≈ 0.3307
```

优势：

```
成功 episode：A = (1 − 0.125) / 0.3307 ≈ 2.646
失败 episode：A = (0 − 0.125) / 0.3307 ≈ −0.378
```

验算优势和：

```
2.646 + 7×(−0.378) ≈ 2.646 − 2.646 ≈ 0
```

这解释了为什么“混合组”有信号：成功样本被推高，失败样本被压低，组内基线抵消了任务难度带来的偏置。若 8 个 episode 全失败，`mean=0, std=0`，优势为零，这一组被丢弃；若全成功，同理。论文的 curriculum 就是为了尽早制造“混合组”。

再做一个**假设性**的帧跨度校验，防止概念混淆：若均匀采 K=8 帧，Δt=0.5 s，首末跨度是 `(8−1)×0.5=3.5 s`；若从 t 预测 t+1 至 t+8，终点距 t 是 `8×0.5=4.0 s`。本文不采未来帧，而是每轮观测当前帧并输出下一 pose；所以不要把本文的 19.7 轮平均或 256k context 当成视频记忆窗口。

## 4. 工程视角

| 约束 | 论文数字 | 工程含义 |
|------|----------|----------|
| 训练规模 | 约 50,000 H100-equivalent GPU-hours | 不是小实验室可轻松复现的规模；PEAR 的异步和 prefix cache 是必需项 |
| 每步 batch | 128 groups × 8 rollouts，group filter 后计数 | 组内必须混合成功/失败才有梯度；采样成本高 |
| episode 长度 | 平均约 94k tokens（1.8k prompt + 92k response），19.7 轮，39–59 张图 | 长上下文是主成本；训练初期 119k/19.9 轮，末期 82k/18.5 轮 |
| 序列上限 | LIBERO 256k；RoboDojo 160k | 不是所有实验共有的记忆窗口；换环境会变 |
| prefix cache | H100 上 95%+ 命中；160k 上限 64 rollouts，256k 上限 32 rollouts | 多轮共享前缀，缓存命中决定吞吐 |
| 并行 | H100：FSDP8×SP8，32,512 tokens/GPU；GB300：FSDP4×SP4，69,632 tokens/GPU | 长序列用 FSDP + sequence parallelism + gradient checkpointing 才放得下 |
| 推理速度 | 52 tokens/s（M5 Max，NVFP4）；221 tokens/s（RTX 5090）；747 tokens/s（GB300） | 747 tokens/s 下 xhigh 一轮约 4.3 s，low 约 1.1 s |
| 量化 | NVFP4 提速但降低精度、拉长 CoT；所有评测用 BF16 | 报告的成功率不是 NVFP4 的结果 |
| 真机预算 | 每 trial 最多 25 轮；单轮 reasoning 最多 32k tokens | 成功 trial 用 7–24 轮、23k–132k context；长推理会直接失败 |

工程上要抓住三点：

1. **这不是低层实时控制。** 每轮要等完整 CoT 和 JSON action，xhigh 一轮约 4.3 s；复杂动作要拆成多轮。论文明确说没有 action chunk，控制器把手臂移动到单个 pose 后停住。因此“控制频率”不是固定 Hz，而是“每轮推理 + 执行”的节奏。
2. **上下文是硬瓶颈。** 完整 RoboDojo 540 episodes 中，final model 有 370/540（68.5%）因 256k context 耗尽而失败；LIBERO 的 Goal 和 Long 分别有 19% 和 33% 的 episode 因 context 到 256k 结束。【实测】作者建议更强基座、更长上下文或 context 管理 harness，但这些都不是本文已验证的方案。【主张】
3. **控制器质量会影响结果。** 动作是 pose-level target，控制器执行不到位会错过目标；低效控制器还会消耗 step budget。论文在 limitations 中承认这一点。对部署团队，IK/控制器的调参不是“外围工程”，而是成功率的一部分。

## 5. 数据与评测

### 5.1 预训练与后训练

- 基座：Qwen3.8-27B，open-weight 27B reasoning model。论文没有展开其预训练数据与架构细节。
- 作者**没有做预训练**，也**没有做 SFT**；只用 RURL 做 RL post-training。没有 demonstrations，没有 real-robot data。
- 训练课程：LIBERO 的 Goal、Spatial、Object 共 148 steps；第 149–225 steps 加入全部四个 LIBERO suite；再用 115 steps 训练 RoboDojo 的 `general_pickup` 和 `align_blocks`。总 step 数 340。
- 作者观点：机器人数据应在 pretraining/mid-training 阶段加入，而不是对已经 post-trained 的模型做 SFT；但这是主张，不是本文实验结论。

### 5.2 仿真评测

| 基准 | 设置 | 本文结果 | 对照 |
|------|------|----------|------|
| LIBERO | 2,000 episodes（40 任务×50） | base 1.6% → ours 74.4% | GPT-6.1 Sol max 76.2%；GPT-6 Astra max 80.6%；medium 48.1%/49.3% |
| LIBERO 分 suite | 同上 | Spatial 85.8、Object 97.0、Goal 61.4、Long 53.2 | base 1.2/1.8/2.8/0.4 |
| LIBERO-Plus | 10,030 episodes | 比 LIBERO 掉 12.2 分；最敏感是 initial state，掉 36.5 分 | π0.5 掉 11.2 分 |
| LIBERO-Pro | 四类扰动，每类 2,000 episodes | 总平均 62.1% | π0.5 总平均 53.3% |
| LIBERO-Pro task | 同一场景换任务 | 57.4% | π0.5 0.75% |
| LIBERO-Pro position | 交换物体位置 | 57.5% | π0.5 20.8% |
| LIBERO-Pro semantic | 改写指令 | 73.3% | π0.5 95.8% |
| LIBERO-Pro object | 物体换外观 | 60.2% | π0.5 96.0% |
| RoboDojo 两任务 | 每任务 163–200 episodes，95% Wilson | final 71.2%/78.8% | LIBERO-only 39.9%/17.0%；base 10.7%/13.0% |
| RoboLab-120 | 120 任务×10 = 1,200 episodes；final 为两次评测均值 | final 37.3% | base 5.4%；LIBERO-only 22.1%；Cosmos3 36.8%；π0.5 28.0%；DreamZero 25.7%；GR00T N1.6 7.2% |
| 完整 RoboDojo | 540 episodes | final 成功 21 个；68.5% 失败因 context 耗尽 | base 成功 1 个 |

读数注意：

- **摘要的 57.4% vs 0.75% 不是 LIBERO-Pro 总平均。** 它是 task perturbation 子项；总平均是 62.1% vs 53.3%。不要混用。
- π0.5 的结果不是作者自己评测的：LIBERO 来自 openpi，LIBERO-Plus 来自 Yan et al.，LIBERO-Pro 是 Zhou et al. 的 per-suite 结果平均；RoboLab 上的 π0.5 又是另一个 DROID checkpoint。跨表比较要保留这个条件。
- GPT-6.1 Sol / GPT-6 Astra 用同一协议，但 effort 设置不同（medium/max）；这不是等算力比较。ours 在 xhigh 下 74.4%，超过两者 medium，低于两者 max。
- LIBERO 的 2,000 episodes 是评测 episode，不是训练 step；RoboLab 的 1,200 episodes 是 120 任务×10；final model 的 37.3% 是两次评测的均值（2,400 episodes）。

### 5.3 真实机器人设置

- 机器人：真实双臂 YAM，训练中从未见过；没有任何 real-robot data。
- 试验数：每个模型 30 trials。10 个 align-blocks trials（5 个场景×2） + 20 个 general-pickup trials（10 个物体×2）。
- 成功判定：25 轮内完成；单轮 reasoning 不超过 32k tokens；无效动作重试也计一轮。
- 结果：【实测】final 18/30（60%），LIBERO-only 7/30（约 23%），base 0/30。
- 成功 trial：7–24 轮，23k–132k context。失败原因：final 为 6 次轮数耗尽、3 次未对齐、2 次碰倒、1 次球滚走；base 为 25 次轮数耗尽、4 次碰倒、1 次未对齐；LIBERO-only 更分散，还有 32k 单轮、unsafe pose、未完全抬起、抓错物体。
- 两个 reasoning 演示：画《星月夜》的简笔线稿；选酚酞而非碘/乙醇检测碱性。它们是定性展示，不是统计评测。

【分析】18/30 的点估计是 60%，但 30 trials 很小；按 Wilson 95% 近似约 42%–75%（本文编辑计算，非论文报告）。因此“60%”应理解为这次 30 trials 的实测，不宜直接外推为普遍成功率。论文没有给真机 Wilson 区间，只给了 RoboDojo 的区间。

## 6. 能力与失败模式

| 能力 | 证据 | 条件 |
|------|------|------|
| 指令跟随 | LIBERO-Pro task 57.4% vs π0.5 0.75% | 只测四类扰动；semantic/object 上 π0.5 仍更强 |
| 扰动鲁棒 | LIBERO-Plus 掉 12.2 分；position 掉 16.9 分 | initial state 掉 36.5 分，仍敏感 |
| 零样本迁移 | RoboDojo、RoboLab、真实 YAM | 真实只两个任务族；RoboLab 是仿真 |
| 保留通用能力 | 9 个 benchmark 的 95% 区间包含零；均值变化 −0.15 | 不能证明“完全无损”，只能说未测出显著退化 |
| 长程推理 | xhigh 比 low 成功率高很多：74.4% vs 21.9% | 代价是每轮 3.23k vs 0.86k thinking tokens；速度更慢 |

| 失败模式 | 证据 | 原因 |
|----------|------|------|
| context 耗尽 | 完整 RoboDojo 68.5% 失败；LIBERO Goal/Long 有 19%/33% 因 256k 结束 | 每轮都保留历史，长任务迅速吃满上下文 |
| 速度慢 | 747 tokens/s 下 xhigh 一轮约 4.3 s；少数 32k tokens 单轮 >30 s | 每步先长推理再动，不是实时闭环 |
| 无 action chunk | 每轮一个 pose，复杂动作拆多轮 | 论文明确说当前不能可靠规划 chunk；控制器质量影响大 |
| prompt 依赖 |  harder tasks 需要详细描述 embodiment 和任务 | 27B 基座的世界知识/实体知识有限 |
| 规模有限 | 两个仿真器、少量任务、27B、无 SFT | 作者承认没 scale up；GPU 资源有限 |
| 真机统计弱 | 30 trials/模型，两个任务族 | 不能据此声称广泛真实世界泛化 |

### 6.1 隐含假设

- **仿真成功判定与真实任务成功足够一致。** 论文用仿真器二值奖励训练，再零样本上真机；这默认 sim-to-real gap 可被 pose-level 接口和推理吸收，但没有逐项验证仿真/真实成功条件的一致性。
- **二值奖励足以支撑长程学习。** 实际上依赖 curriculum 和混合组；全失败任务没有信号。作者没有证明在更大任务集上仍能稳定扩展。
- **context 可以充当记忆。** 但完整 RoboDojo 的 68.5% 失败是 context 耗尽；记忆机制不是无限的，也不是所有实验共有的窗口。
- **prompt 能提供足够的 embodiment 信息。** 论文承认模型经常缺乏对机器人的知识，需要 prompt 详细描述；这既是能力，也是依赖。
- **IK/控制器可靠。** 动作是 pose-level target，控制器执行不到位会直接造成失败；论文把 controller 质量列为限制。
- **基线比较是公平的。** GPT 基线用同一协议但不同 effort；π0.5 结果来自先前工作、不同 checkpoint；这些比较是“有条件可比”，不是严格等条件复现。

## 7. 与相关工作对比

| 方向 | 代表 | 训练方式 | 是否需 harness | 数据 | 与本文关系 |
|------|------|----------|----------------|------|------------|
| Frontier RUA | GPT-6 Astra、Claude Fable 5.2 | 闭源，细节未披露 | 论文未统一说明 | 闭源 | 本文用同一协议做基线；ours 接近但未超过 max effort |
| Harness RUA | ORA、ASPIRE、Harness VLA、OpenRUA、RAX、PyRUA-Lean | 给开源模型加 perception/skills/memory/code | 是 | 多样 | 本文明确不用 harness；作者认为 harness 可能限制动作空间 |
| CUA RL | DigiRL、WebRL、UI-TARS-2 | 多轮 RL，设备/网页/桌面 | 因环境而异 | 交互数据 | 本文把同类 recipe 搬到仿真机器人 |
| VLA SFT | π0.5 | 示范监督微调 | 否，但是 VLA 策略 | 任务示范 | LIBERO 上更强；任务/位置扰动下明显弱于本文 |
| 本文 | RURL + PEAR | 仿真 RL，无示范、无 harness | 否 | 仿真 LIBERO + RoboDojo | 开源 27B，零样本真机 |

关键差异不是“RL 一定比 SFT 好”，而是**在不同扰动上各有所长**：π0.5 在 LIBERO 原任务、semantic/object 扰动上更强；本文在 task/position 扰动上更强。作者的解释是“模仿示范” vs “推理任务”，但这是解释，不是因果实验。

**面试 Tip**：被问到这篇的核心，不要只说“RL 提点”。要答：它把 RUA 做成无 harness 的 pose-level 多轮 agent，用二值奖励和 GRPO 在仿真里学，接口不变所以能零样本上真机；代价是每轮一次长推理、上下文易爆、真实评测只有 30 trials。

## 8. 精读建议

- **值得精读原文的人**：
  - 做多模态具身 Agent / RUA 的研究者：重点看 §3.1 的 JSON 协议和 Figure 2。
  - 做 sim-to-real RL post-training 的工程团队：重点看 §3.2 的 GRPO 变体、PEAR 的异步/prefix cache、§4.4 和 Appendix G。
  - 要评估“开源模型能否替代闭源 frontier RUA”的人：重点看 §4.1–4.3 的分 suite/扰动/迁移数字，以及 §5 limitations。
- **建议章节路径**：先读 §3.1 + Figure 2（接口）→ 再看 §3.2 + 公式 1–2 + Table 1（算法与基础设施）→ 然后 §4.4 + Appendix G（真机 trial 明细）→ 最后 §5（限制）。如果熟悉 related work，§2 可跳；Appendix B–E 用于核对数字。
- **不值得精读的理由**：如果你只做低层 VLA action chunk、视频预测或高频率实时控制，这篇的“每轮一个 pose + 长 CoT”不是你的设定；读摘要和 §5 即可。如果你已经熟悉 GRPO/DAPO 和 agentic RL，数学部分可以快速扫过，重点看真机与失败模式。

---

[← Back to Theory](./README.md)

关键引用：
- 论文摘要页：https://arxiv.org/abs/2611.18562
- 论文 HTML 正文：https://arxiv.org/html/2611.18562