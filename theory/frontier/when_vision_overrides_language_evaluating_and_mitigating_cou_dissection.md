# 当视觉压过语言：VLA 反事实失败的评测与缓解 (When Vision Overrides Language: Evaluating and Mitigating Counterfactual Failures in VLAs)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-11-06
>
> **论文**: When Vision Overrides Language: Evaluating and Mitigating Counterfactual Failures in VLAs
> **链接**: https://arxiv.org/abs/2602.17659
> **核心定位**: 把 VLA「听得见指令却照着视觉先验做」的失败模式命名为 counterfactual failure，用 LIBERO-CF 量化它，再用一个类似 CFG 的双分支动作外推在推理时把语言信号放大；不改架构、不重训 base VLA，TF 变体零训练，VA 变体效果更好但要额外训一个 vision-only 分支。

> **版本说明**：本次证据来自 arXiv:2602.17659v3 [cs.CV] 04 Nov 2026 的正文渲染。候选摘要片段在 “regardless of language intent” 附近被截断，未包含任何数字；本文所有数字均取自正文（Table 1–7、Fig. 2/4/5/7、§6.1、§7.2、附录 B/C），未用候选摘要补写。候选摘要与正文摘要在定性主张上未见冲突，差异主要是截断导致的信息缺失。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | VLA 在反事实指令下会回退到训练场景的视觉捷径；LIBERO-CF 把这种失败量化出来，CAG 用「条件动作减无条件动作」的位移外推来增强语言接地。 |
| 适合精读 | 如果你在做 VLA 语言接地评测、多模态模态失衡、推理时引导，重点看 §4（基准设计）、§5（CAG 公式）、§6.1 与 §7.2（数字）。 |
| 可以跳过 | 如果你只关心 VLA 预训练数据缩放或新架构，这篇是推理时插件，距离中等；读摘要和 Table 3 即可。 |
| 落地可行性 | 中高：TF-CAG 零训练、内存不变，延迟约 1.6×；VA-CAG 效果更好，但要额外训 VA，内存约 1.9×。 |
| 主要风险 | grounding 指标只测「夹爪接触目标」，可能高估语言理解；固定 ω、真实部署开销仅按 15-step action chunk 测得、真实世界每指令仅 10 trials，泛化性待外部复现。 |

💡 **X-Ray 开场**

这篇论文问的是：VLA 在熟悉场景里，真的会按语言指令做事吗？作者发现，当指令指向训练时没充分监督的替代任务时，模型会反复执行它「最熟」的那个训练任务，哪怕语言明确说了别的目标。对 VLA 研究者意味着：标准 LIBERO 高分不等于语言接地，评测里必须把「视觉上合理但语言不同」的指令单独拆出来；而 CAG 给出了一条不改模型、只改推理的缓解路径。

📍 **研究全景时间线**

```
2023 ── LIBERO / Open X-Embodiment 奠定仿真评测与大规模机器人数据
  │
2024 ── OpenVLA / π_0 / DROID 把 VLA 与真实机器人数据管线推上台面
  │
2025 ── π_0.5 / OpenVLA-OFT 把 LIBERO 指标推到接近饱和
  │
2026 ── 本文：LIBERO-CF + CAG  ← 当前位置
        局限：仅 LIBERO 仿真 + 单一 Franka 真实平台，未覆盖移动/双臂/人形
```

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

论文的核心不是新 VLA 架构，而是一个「评测基准 + 推理时插件」。CAG 有两个变体：TF（training-free）直接用同一个 VLA 丢掉语言输入来近似无条件分支；VA（vision-action prior）单独训练一个只看 obs 的 vision-only 模型作为无条件分支。

| 模块 | 输入 | 输出 | 频率/时序 | 训练/推理差异 |
|------|------|------|-----------|----------------|
| 条件分支 π_cond | 观测 o + 指令 l | action chunk a_cond | 每个 action chunk 一次 | 用官方 VLA 权重；TF 不训练，VA 也不改条件分支 |
| 无条件分支 π_uncond（TF） | 观测 o（丢 l） | action chunk a_uncond | 与条件分支并行 | 不训练，同一 VLA 丢语言输入 |
| 无条件分支 π_uncond（VA） | 观测 o | action chunk a_uncond | 与条件分支并行 | 单独训练 VA，丢语言 token；π_0/π_0.5 全量微调，OpenVLA-OFT 用 LoRA |
| CAG 混合 | a_cond, a_uncond, ω | a_CAG | 每个 chunk 一次 | 无训练，动作空间逐维外推 |
| LIBERO-CF 仿真评测 | 场景 + 反事实指令 | grounding / success | 每任务 50 trials | 评测时不训练（OpenVLA-OFT/π_0/π_0.5 用官方 checkpoint；X-VLA 按 §B.3 在 LIBERO 上微调） |
| 真实机器人评测 | 场景 + 指令 | grounding / success | 每指令 10 trials | π_0.5-DROID 全量微调 10k steps，batch 32 |

需要特别说明：本文不预测未来视频帧，也不使用历史帧记忆机制。CAG 作用于动作 chunk；论文中出现的 15-step 是动作 chunk 长度，**不是**历史上下文窗口，也**不是**所有实验共有的记忆窗口。论文没有报告不同 chunk 长度的消融，因此不能把这个步长外推到其他设置。若类比帧采样，均匀间隔下 K 个采样帧的首末跨度是 (K-1)×Δt，从当前参考 t 预测 t+1 至 t+H 的终点距当前参考是 H×Δt；但本文的动作 chunk 是未来动作序列，不是历史帧采样，两者不能混用。

### 1.2 关键机制 (Key Mechanism)

论文的机制可以拆成四步：

1. **把失败归因于视觉捷径**：在固定场景下，机器人数据往往只覆盖少量任务，观测 o 与训练任务强绑定；语言 token 数量远少于视觉 token，语言条件在动作选择中的影响被边缘化。
2. **用贝叶斯视角形式化**：P(a|o,l) ∝ P(a|o)·P(l|a,o)。如果 P(a|o) 太强，语言项 P(l|a,o) 无法把动作分布推离视觉先验。
3. **估计语言诱导位移**：Δa_l = a_cond - a_uncond。同一个观测下，条件分支和无条件分支的预测差，近似「语言带来的动作改变」。
4. **CFG 式外推**：a_CAG = a_uncond + ω·Δa_l。ω=1 退化为标准条件策略；ω>1 放大语言位移；ω=0 退化为纯视觉。

TF 与 VA 的取舍：

- TF 不需要额外训练，直接复用同一个 VLA，丢语言输入来近似 π(a|o)。缺点是「丢语言」的近似不干净，条件分支和无条件分支共享同一套权重，语言监督的耦合没有被解开。
- VA 显式训练一个 vision-only 分支，与语言监督解耦，提供更干净的 P(a|o)。代价是要多训一个模型，内存和部署成本更高。
- 论文的 Table 4 还对比了 R2（训练时随机语言 dropout）和 R4（LoRA 微调 VA）、R5（全量微调 VA）。R2 表现最差：语言 dropout 会削弱条件分支的语言条件能力，导致模型更偏向视觉主导行为；R4 和 R5 都强于 R3（TF），说明显式 VA 比复用条件分支更有效。

⚡ **Eureka Moment**：把「语言条件的作用」看成条件动作与无条件动作之间的位移，然后像 classifier-free guidance 一样在动作空间放大这个位移——不需要改架构，也不需要额外 demonstrations。

### 1.3 信息流/架构图 (Flow / Diagram)

```
                        ┌──────────────────────────────────────┐
 o（视觉观测） ──┬──────>│ π_cond(o, l)        → a_cond         │──┐
                 │       └──────────────────────────────────────┘  │
                 │                                                  │
 l（语言指令） ──┘       ┌──────────────────────────────────────┐  ├──> a_CAG = a_uncond + ω·(a_cond - a_uncond)
 o（视觉观测） ─────────>│ π_uncond(o)         → a_uncond       │──┘
                         └──────────────────────────────────────┘
                                        │
                                        ▼
                              机器人执行 action chunk
```

TF 变体里，上下两个分支是同一个 VLA，只是下面那次推理丢掉语言输入；VA 变体里，下面分支是单独训练的 vision-only 模型。两个分支并行推理，所以延迟不是简单 2×，论文实测约 1.6–1.7×。

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：

```
a_CAG = a_uncond + ω · (a_cond - a_uncond)
```

**目标**：在推理时增强语言条件，把动作预测从视觉先验拉向指令意图，同时不改 base VLA 的架构和预训练权重。

**形式化**：论文先给出一个概念性的贝叶斯分解：

```
P(a | o, l) ∝ P(a | o) · P(l | a, o)
```

其中 P(a|o) 是 vision-conditioned action prior，P(l|a,o) 衡量动作与观测、指令的兼容性。CAG 不直接对这个分布做采样，而是在动作空间做 CFG 式外推：

```
Δa_l = a_cond - a_uncond
a_CAG = a_uncond + ω · Δa_l
```

> 符号与本文保持一致：o 为视觉观测，l 为语言指令，a 为动作 chunk，a_cond 为条件分支预测，a_uncond 为无条件分支预测，Δa_l 为语言诱导动作位移，ω ≥ 0 为引导尺度。

| 符号 | 含义 |
|------|------|
| o | 视觉观测（图像/多视角） |
| l | 语言指令 |
| a | 动作 chunk（长度按具体设置） |
| a_cond | 条件分支 π_cond(a|o,l) 的预测 |
| a_uncond | 无条件分支 π_uncond(a|o) 的预测 |
| Δa_l | 语言诱导动作位移，a_cond - a_uncond |
| ω | 引导尺度，控制语言条件强度 |

**直觉**：ω=0 时完全忽略语言，退化为纯视觉策略；ω=1 时就是标准条件策略；ω>1 时把「语言带来的动作改变」放大，让策略更坚定地偏离视觉先验。论文发现 ω 过大会损害操作精度，所以每个 backbone 固定一个经验值：π_0 和 π_0.5 用 ω=1.5，OpenVLA-OFT 用 ω=3.0。OpenVLA-OFT 需要更大 ω，与它在 Table 1 中更强的 vision-only 表现（69.1% avg success）一致。

## 3. 带数字走一遍：玩具例子 (Worked Example)

### 3.1 假设的标量外推（假设，非论文实测）

为直观理解 CAG，假设某一维动作 a ∈ R，某时刻：

- 无条件分支预测 a_uncond = 0.20（偏向训练任务物体）
- 条件分支预测 a_cond = 0.60（偏向指令目标）
- 语言诱导位移 Δa_l = 0.60 - 0.20 = 0.40
- 引导尺度 ω = 1.5

则：

```
a_CAG = 0.20 + 1.5 × 0.40
      = 0.20 + 0.60
      = 0.80
```

验算：0.20 + 0.60 = 0.80。语言位移从 0.40 被放大到 0.60，最终动作 0.80。

边界行为：

```
ω = 0   → a_CAG = 0.20（纯视觉先验）
ω = 1   → a_CAG = 0.60（标准条件策略）
ω = 2   → a_CAG = 1.00（更强外推，可能超出可行域）
```

这个例子说明：CAG 不是重新训练一个更听话的模型，而是在推理时把「语言已经带来的那一点改变」放大。ω 太大时，动作可能被推到物理上不可行或精度下降的区域，这与论文「excessive guidance degrades manipulation accuracy」的观察一致。

> 标记：以上是假设的标量例子，真实动作是多步 action chunk 的多维向量，CAG 逐维/逐时间步外推；论文没有给出这组具体数值。

### 3.2 论文实测的 grounding 口径演示（非假设）

论文 Table 9 的 Coke/Sprite/Fanta 场景中，π_0.5 在指令 “Pick up the fanta” 下的分解是：

- 10 次试验中，9 次尝试正确目标（Fanta），其中 6 次成功：记为 6/9；
- 1 次抓了 Coke 且成功：记为 1/1；
- 按论文正文口径，grounding = 9/10 = 90%，success = 6/10 = 60%，biased success = 1/10 = 10%。

这个例子说明两个指标的区别：grounding 只问「夹爪是否接触指令目标」，success 问「任务是否完成」。模型可能碰对了目标但没完成任务，也可能抓错目标却「成功」完成训练任务——后者正是 counterfactual failure 的典型表现。

## 4. 工程视角 (Engineering View)

### 4.1 延迟、内存与训练成本

论文 Table 6 给出真实世界部署的开销（π_0.5 backbone，每次 15-step action chunk）：

| 方法 | GPU 内存 | 延迟 | 额外训练 |
|------|----------|------|----------|
| π_0.5 baseline | 8.5 GB | 80.0 ± 2.2 ms | 无 |
| TF-CAG | 8.5 GB（1.0×） | 126.6 ± 4.4 ms（1.6×） | 无 |
| VA-CAG | 16.5 GB（1.9×） | 132.6 ± 1.9 ms（1.7×） | 约 3 h LIBERO / 1 h 真实，8×H100 |

工程含义：

- **TF-CAG 是「零训练、零额外内存」的选项**：只增加推理时间，适合已经部署了 VLA、想快速提升指令跟随的场景。代价是 1.6× 延迟，控制频率会下降。
- **VA-CAG 是「效果优先」的选项**：需要额外训一个 vision-only 分支，内存接近 2×，延迟 1.7×。论文说 VA 训练在 LIBERO 上约 3 小时、真实数据约 1 小时（8×H100），这个成本对研究团队可接受，对产线部署要看预算。
- **双分支并行**：论文明确说两个分支并行推理，所以延迟不是 2×。但并行意味着 GPU 上要同时容纳两个分支的激活，VA-CAG 的 16.5 GB 是实测值。
- **15-step chunk 是特定设置**：论文只在 15-step action chunk 上报告延迟（约 130 ms）。不能把这个数字外推成「所有 CAG 实验都用 15 步」，也不能把它当作历史上下文或记忆窗口。论文没有报告不同 chunk 长度的消融，这是待补的工程参数。
- **ω 是每个 backbone 固定超参**：π_0/π_0.5 用 1.5，OpenVLA-OFT 用 3.0。论文没有给自适应 ω 的策略，实际部署需要按 backbone 和任务调参。作者在 limitation 里也提到未来会探索基于不确定性或视觉-语言不一致的自适应引导。
- **未量化的风险**：动作空间外推可能放大噪声或导致抖动，论文没有报告动作平滑性、加速度或量化误差。这是从「指标提升」到「可部署」之间需要补的一环。

### 4.2 与「记忆窗口」的区分

本文没有记忆机制，也没有历史帧聚合。CAG 的输入是当前观测 o 和指令 l，输出是未来动作 chunk。所谓 15-step 是动作 chunk 的长度，不是模型记住的历史长度。工程上不要把「CAG 在 15-step chunk 上工作」误读成「CAG 依赖 15 帧历史」或「所有实验共享 15 步记忆窗口」。论文没有做这类消融，任何关于记忆窗口的推断都超出证据范围。

## 5. 数据与评测 (Data & Eval)

### 5.1 预训练与模型来源（逐模型归属）

论文使用四个 VLA 作为基线，架构、预训练数据、动作表示各不相同，不能把一组模型的属性合并归给每个模型：

| 模型 | 参数量/主干 | 预训练数据（论文所述） | 动作表示/解码 | 本文中的 VA 分支训练方式 |
|------|-------------|------------------------|----------------|--------------------------|
| OpenVLA-OFT | 7B，Prismatic VLM 主干 | Open-X-Embodiment | continuous action，L1 回归，parallel decoding + action chunking | LoRA 微调 OpenVLA-7B，batch 64，300k steps，lr 5×10⁻⁴ |
| π_0 | 3B，PaliGemma 主干 | 机器人数据（论文未展开细节） | flow matching | 全量微调，30k steps，batch 32，lr 5×10⁻⁵，EMA 0.999 |
| π_0.5 | π_0 改进版 | 混合机器人 + 非机器人数据 | 继承 π_0 的 flow matching | 全量微调，30k steps，batch 32，lr 5×10⁻⁵，EMA 0.999 |
| X-VLA | 0.9B，soft-prompted transformer | 跨 embodiment 数据 | 绝对末端执行器动作空间 | 同配置丢语言，50k iterations，batch 16，lr 1×10⁻⁴，前 1k steps 冻结 backbone |

需要说明：

- OpenVLA-OFT 的 7B、Prismatic、Open-X-Embodiment 来自论文 §B.1/§B.3；π_0 的 3B、PaliGemma、flow matching 来自 §B.1；π_0.5 的混合预训练来自 §B.1；X-VLA 的 0.9B、soft-prompted transformer 来自 §B.2。
- 论文没有说 LIBERO-CF 的独立下载地址或代码仓库；本次来源只包含 arXiv 正文，未验证项目页内容。因此本文不给出 repo 链接，也不声称基准已开源。
- OpenVLA-OFT、π_0、π_0.5 使用官方 LIBERO-finetuned checkpoint；X-VLA 使用官方 base model（X-VLA-Pt）在 LIBERO 上微调；VA 分支是在 LIBERO 数据上丢语言微调得到的，不是从零预训练。

### 5.2 仿真评测：LIBERO-CF

LIBERO-CF 构建在 LIBERO 场景布局上，作者声称是「首个 VLA 反事实基准」（作者主张，非本文独立验证）。四个 suite：

| Suite | 任务数 | 评测内容 |
|-------|--------|----------|
| CF-Spatial | 15 | 空间语言接地，目标为原背景物体 |
| CF-Object | 10 | 替代指令，目标为不同物体 |
| CF-Long | 10 | 长程语言跟随，涉及新目标 |
| CF-OOD | 15 | 未见物体的泛化 |

评测设置：

- 每个任务 50 trials（§B.1），不是 50 epochs，也不是 50 个训练 step。
- 指标：grounding（夹爪是否接触指令指定目标，不管任务完成）、success（是否完成任务）、Faithful（遵循指令任务）、Biased（回退到原始训练任务）。
- 基线：OpenVLA-OFT、π_0、π_0.5；附录 B 额外加入 X-VLA。
- VA 训练使用 LIBERO 数据，丢语言 token，训练配置与对应 VLA 默认配置一致（除丢语言外）。

关键数字（Table 3，平均列）：

| 模型 | 方法 | Grounding (F.) | Success (F.) | Biased Grounding (B.) | Biased Success (B.) |
|------|------|----------------|--------------|------------------------|---------------------|
| OpenVLA-OFT | baseline | 4.7% | 0.4% | 83.6% | 78.6% |
| OpenVLA-OFT | TF | 5.8% | 0.9% | 81.9% | 76.3% |
| OpenVLA-OFT | VA | 11.3% | 2.1% | 65.4% | 41.0% |
| π_0 | baseline | 28.8% | 9.6% | 63.0% | 45.0% |
| π_0 | TF | 31.9% | 9.6% | 59.3% | 27.6% |
| π_0 | VA | 37.3% | 10.4% | 52.0% | 25.8% |
| π_0.5 | baseline | 30.8% | 13.2% | 65.6% | 60.9% |
| π_0.5 | TF | 40.5% | 16.8% | 59.8% | 43.5% |
| π_0.5 | VA | 46.3% | 21.7% | 52.2% | 36.2% |

读法：

- F. 列是「遵循反事实指令」的比率，B. 列是「回退到训练任务」的比率。两者不是同一指标的正反面，而是两个不同行为维度的统计。
- π_0.5 是 baseline 中最好的，但 grounding 也只有 30.8%、success 13.2%；OpenVLA-OFT 最严重，grounding 4.7%、success 0.4%。
- TF-CAG 对 π_0.5 把 grounding 从 30.8% 提到 40.5%（+9.7 个百分点，论文写作 9.7%），success 从 13.2% 提到 16.8%（+3.6 个百分点）。
- VA-CAG 对 π_0.5 把 grounding 提到 46.3%（+15.5 个百分点），success 提到 21.7%（+8.5 个百分点）；同时 biased grounding 从 65.6% 降到 52.2%，biased success 从 60.9% 降到 36.2%。
- X-VLA（Table 7）：baseline grounding 37.3%、success 13.8%；TF 39.7%/19.5%；VA 41.0%/17.1%。VA 的 grounding 更高，但 success 略低于 TF，说明不同 backbone 的收益结构不同。

Table 2 的「移除训练任务物体」实验进一步支持视觉捷径假设：CF-Spatial 与 CF-Focused 使用相同反事实指令，只去掉训练任务物体，平均 grounding 从 39.0% 提到 64.4%，success 从 20.0% 提到 40.0%。这是很强的机制证据，但也要注意：去掉物体改变了场景分布，提升不可能全部归因于「语言接地变强」。

### 5.3 真实机器人设置

真实实验与仿真分开：

- 平台：Franka Research 3 + Robotiq 2F-85 夹爪；ZED 2i 外部相机 + ZED mini 腕部相机（§7.1）。
- 场景维度：Object Recognition、Spatial Reasoning、Goal Targeting、OOD Generalization；附录 C 增加 Long-Horizon Reasoning。
- 每个场景 3 个任务：1 个 in-domain（20 个专家演示）+ 2 个 under-observed（各 1 个演示，仅作 minimal warm-up）；OOD 任务零样本。
- 每指令 10 trials，随机物体位置。
- 模型：π_0.5-DROID 全量微调 10k steps，batch 32；VA 分支训练约 1 小时，8×H100。
- 指标：grounding 和 success，按场景报告；论文摘要给出平均 +9.4% grounding / +17.2% success（VA-CAG），正文 §7.2 分场景给出具体数字。

分场景关键数字（§7.2）：

- Object Recognition：CAG 在两个场景达到 100% 平均 grounding，成功率各提升 13.3%。
- Spatial Reasoning：grounding 提升 16.6%，success 提升 13.3%。
- Goal Targeting：grounding 到 96.7%，success 提升 36.7%。
- OOD：Cube 的 5 次失败中 4 次是 counterfactual failure；CAG 缓解后零样本泛化被进一步释放。

需要克制的地方：真实世界每指令只有 10 trials，统计粒度有限；场景是桌面操作，不能外推到移动、双臂或人形。

### 5.4 指标口径与评测条件

| 指标 | 定义 | 注意 |
|------|------|------|
| grounding | 夹爪是否接触指令指定目标，不管任务完成 | 可能高估语言理解：碰一下目标不等于按指令完成任务 |
| success | 是否完成任务 | 更严格，但受底层技能限制 |
| Faithful (F.) | 遵循指令任务的比率 | Table 3 的 F. 列 |
| Biased (B.) | 回退到原始训练任务的比率 | Table 3 的 B. 列，不是 F. 的补集 |

评测条件必须和数字一起读：仿真每任务 50 trials，真实每指令 10 trials；仿真中 OpenVLA-OFT/π_0/π_0.5 用官方 checkpoint，X-VLA 用官方 base model 在 LIBERO 上微调，真实用 10k steps 微调的 π_0.5-DROID；VA 分支是单独训练的，不是同一个模型换输入。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

| 能做什么 | 不能做什么 |
|----------|------------|
| TF-CAG 零训练提升 grounding：π_0.5 30.8% → 40.5%，success 13.2% → 16.8% | 技能完全缺失时无效：作者明确说 action-space guidance 不能创造不存在的技能 |
| VA-CAG 跨 VLA 一致提升：OpenVLA-OFT、π_0、π_0.5、X-VLA 都受益 | 不能替代反事实数据：Table 5 显示加数据仍有增益，CAG 与数据互补 |
| 降低 biased 行为：π_0.5 VA 把 biased success 从 60.9% 降到 36.2% | 不能解决所有 OOD：CF-OOD 上 π_0.5 VA 仍只有 36.4% grounding、10.3% success |
| 真实世界保留 in-domain 性能，同时提升 under-observed 任务 | 未验证移动/双臂/人形；只在 LIBERO 仿真 + 单一 Franka 桌面平台 |
| 低延迟：TF 1.6×、VA 1.7×，双分支并行 | 固定 ω，非自适应；每个 backbone 要手动调 |
| 与 CFG 类似，plug-and-play，不改架构和预训练权重 | 未量化动作抖动/平滑性；VA 需要额外内存和训练预算 |

### 6.1 隐含假设 (Hidden Assumptions)

- **反事实任务在物理上可行**：LIBERO-CF 的替代指令是在同一场景布局下可执行的，但「可行」由作者构造保证，不是模型自己判断的。
- **失败主要来自视觉捷径，而不是语言理解不足**：论文用 V-only 输入仍保留高成功率（Table 1）来支持这一点，但 grounding 低也可能部分来自语言-视觉对齐不足，二者没有完全分离。
- **无条件分支能近似 P(a|o)**：TF 用「丢语言」近似，VA 用单独训练近似。VA 更干净，但它也是用同一批场景数据训练的，可能继承视觉偏差；论文没有量化 VA 分支自身的视觉先验强度。
- **动作空间线性外推不破坏底层技能**：CAG 在动作 chunk 上逐维外推，假设技能本身存在且连续；如果动作多模态或强非线性，线性外推的副作用未验证。
- **固定 ω 跨任务/场景可接受**：论文按 backbone 固定 ω，没有按任务自适应；不同反事实任务的视觉歧义程度不同，固定 ω 可能不是最优。
- **grounding 接触目标能代表语言接地**：指标只测接触，不测接触后的行为；模型可能碰对目标但执行错误任务。
- **仿真结论可迁移到真实**：论文做了真实实验，但规模有限（每指令 10 trials、单一机器人），迁移性仍需外部复现。

## 7. 与相关工作对比 (Comparison)

| 工作 | 关注点 | 是否需要额外数据/训练 | 评测 | 方法类型 |
|------|--------|------------------------|------|----------|
| LIBERO-Plus (Fei et al., 2025) | 鲁棒性分析（光照、噪声、布局等） | 否 | LIBERO 变体 | 评测/分析 |
| CAST (Glossop et al., 2025) | 用反事实标签改进指令跟随 | 是（反事实标签） | 指令跟随 | 数据增强 |
| LIBERO-Pro (Zhou et al., 2025) | 鲁棒、公平评估，超越记忆 | 否 | LIBERO 变体 | 评测 |
| 本文 | 反事实指令下的视觉捷径 | TF 不需要；VA 需要训 VA | LIBERO-CF + 真实机器人 | 推理时双分支外推 |

与 CFG 的关系：CAG 借用 classifier-free guidance 的「条件减无条件再外推」形式，但作用对象不是扩散模型的去噪步，而是最终动作 chunk；论文明确说这是「CFG-inspired extrapolation heuristic in action space」，不是严格的概率采样。这个区别很重要：CAG 没有理论保证会采样自某个修正后的分布，它是工程上的动作外推。

与 π_0/π_0.5/OpenVLA-OFT 的关系：CAG 是插件，不是新 VLA。它不改 base 架构、不改预训练权重，条件分支就是原模型；增益来自推理时对语言位移的放大，而不是更强的语言理解。

**面试 Tip**：如果被问到「这篇的核心贡献是什么」，一句话答：「它把 VLA 在反事实指令下回退到视觉捷径的现象量化成 LIBERO-CF，并用一个 CFG 式的双分支动作外推在推理时放大语言信号；TF 零训练但增益有限，VA 要额外训一个 vision-only 分支但效果更好。」

## 8. 精读建议 (Reading Guide)

- **值得精读原文的人**：
  - 做 VLA 语言接地/可靠性评测的研究者：§4 的 suite 设计和指标口径值得逐条看，尤其是 grounding 与 success 的分离。
  - 想低成本提升现有 VLA 指令跟随的工程师：§5 的公式和 §6.1 的 TF/VA 对比是落地核心；Table 6 的开销决定你能不能接受 1.6–1.7× 延迟。
  - 研究多模态模态失衡的人：§3.2 的贝叶斯形式和 Table 1 的 V-only 证据可以作为「视觉先验压制语言」的案例。
- **建议章节路径**：先读 §4（基准和指标）→ §5（CAG 公式与双分支）→ §6.1（Table 3 的数字）→ §7.2（真实世界）→ 附录 B.1/B.3（训练细节）→ 附录 C（长程和逐任务分解）。§2 相关工作可以最后扫，不影响理解主线。
- **不值得精读的理由**：如果你不做机器人学习、不关心 VLA 评测，或者已经非常熟悉 CFG 且只想知道「能不能用」，读摘要和 Table 3 即可；如果你要找新架构或预训练数据配方，这篇是推理时插件，不是架构论文。

---

[← Back to Theory](./README.md)

**关键引用**：论文正文与表格均来自 [arXiv:2602.17659](https://arxiv.org/abs/2602.17659)。本文未使用候选摘要中的截断文本补写任何数字或主张。