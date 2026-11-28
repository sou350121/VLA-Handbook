# FlipVLA：仅靠比特翻转的 VLA 后门攻击（FlipVLA: Backdoor Attacks on Vision–Language–Action Models Only Through Bit Flips）

> ⚙️ 本文由 Moltbot 自动生成 | 2026-11-28
>
> **论文**：FlipVLA: Backdoor Attacks on Vision–Language–Action(VLA) Models Only Through Bit Flips
> **链接**: https://arxiv.org/abs/2611.33810
> **核心定位**：它把“VLA 后门”从数据投毒/恶意微调，推进到“只改少量部署权重 bit”的威胁模型：用 selector–amplifier 在现成 FC–GELU–FC 结构里构造物体触发响应，再放大成动作偏差；在 OpenVLA-OFT 与 GR00T N1.7 的 LIBERO 仿真评测中报告 90.22%–100% ASR，同时 clean SR 最多下降 1.50 个百分点。
>
> **版本与摘要核验**：本文依据 arXiv:2611.33810v1 [cs.CV] 25 Nov 2026 的 HTML/PDF 正文；候选摘要与正文摘要一致，未发现官方 PDF 与元数据摘要的实质差异。以下数字以正文 Table 1–4、S1–S2 为准，不用候选摘要补写或替代 PDF 中的主张。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 论文实测：在 OpenVLA-OFT 与 GR00T N1.7 上，仅用固定 68 个 bit flips（selector 60 + amplifier 8），可在 LIBERO 仿真中实现物体触发后门；ASR 90.22%–100%，clean SR 最多降 1.50 个百分点。 |
| 适合精读 | 如果你在做 VLA 部署、权重分发、模型供应链安全、bit-flip/Rowhammer 威胁建模，重点看 §4 方法、§5 评测、§6 消融。 |
| 可以跳过 | 如果你只关心 VLA 建模能力提升、数据配方或真实机器人泛化，这篇距离中等；它不提出新的 VLA 表示学习或控制算法。 |
| 落地可行性 | 低到中：攻击构造需要 white-box 访问与离线 bit search；论文未在真实 GPU/机器人上演示物理 bit 注入，物理可行性仍是未验证假设。 |
| 主要风险 | 论文的“后门”是仿真评测中的固定权重破坏；真实世界触发物体分布、硬件 bit flip 精度、防御/监控均未验证。 |

💡 **X-Ray 开场**

这篇论文问：不改训练数据、不微调、不投毒，只改少量部署权重 bit，能否让 VLA 在看到某个任务无关物体时失败？论文的答案是：在 OpenVLA-OFT 和 GR00T N1.7 的 LIBERO 仿真评测里可以，而且 clean 场景性能下降很小；但“物理上如何精确翻这些 bit”没有被实验证明。对 VLA 研究者意味着：评测不能只看 clean input，还要把权重完整性、触发器物体和供应链攻击纳入威胁模型。

📍 **研究全景时间线**

```
2014  Rowhammer 硬件故障注入研究（来源：论文参考文献）
  → 2019  Bit-Flip Attack 将权重比特翻转作为攻击面
  → 2025  BadVLA 用数据投毒/恶意微调做 VLA 后门
  → 2026  FlipVLA：仅靠少量固定 bit flips 的 object-triggered VLA 后门（本文）
          当前位置：仿真验证强，物理注入与真实机器人未验证
```

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 对象 | 架构/动作生成 | FlipVLA 注入位置 | 修改预算 | 评测范围 | 真实机器人 |
|------|--------------|------------------|----------|----------|------------|
| OpenVLA-OFT | parallel decoding + L1-regression action head | projector 的 FC2（selector）与 FC3（amplifier） | 默认 60 selector bits + 8 amplifier bits，共 68 bits | LIBERO Spatial/Goal/Object/Long；主 Spatial 每个 policy 18,500 episodes | 正文未报告 |
| GR00T N1.7 | flow-matching diffusion transformer | 连接 VLM 与 action diffusion transformer 的 FC–GELU–FC 模块 | 同一构造，默认 68 bits | LIBERO Spatial/Goal/Object/Long 主实验（Table 2）；组件消融每 variant 1,000 rollouts；触发替换每 trigger 2,000 rollouts | 正文未报告 |
| FlipVLA selector | FC2 的单输出通道 j，参数在视觉 token 间共享 | 改 FC2 权重/偏置的 BF16 bit | 60 bits（默认） | 触发 token 激活 + 背景抑制 | 推理时固定 |
| FlipVLA amplifier | FC3 中与通道 j 相连的单个权重 | 改该权重的 bit | 8 bits（默认） | 最大化 positive set 上的动作 MAE | 推理时固定 |

> 归属说明：OpenVLA-OFT 的注入点是 projector 的 FC2/FC3；GR00T N1.7 的注入点是 VLM 到 action diffusion transformer 的 FC–GELU–FC 模块。不能把 OpenVLA-OFT 的 projector 层名直接套到 GR00T N1.7。两个模型都使用 H=8 的 action horizon，但论文没有报告记忆机制；H=8 是未来动作 chunk 长度，不是历史上下文窗口。

### 1.2 关键机制 (Key Mechanism)

- **威胁模型**：white-box attacker 可访问模型架构、预训练参数、中间激活和动作输出；可收集少量 clean、trigger-present、distractor 观测做离线 bit search；可在固定预算内翻选定参数 bit，并把触发物体放进机器人视野。攻击者不能访问/操纵原训练流程、投毒训练数据、重训练或微调、改架构/指令/控制软件；所有修改在推理时固定，无在线更新。
- **为什么不直接联合搜索**：同一组被改权重同时处理 trigger 与 non-trigger 输入；一个破坏 trigger 行为的 bit 也可能破坏正常操作。论文把问题拆成“先构造触发器响应，再放大响应”，用 selector 负责触发判别与背景抑制，用 amplifier 负责把响应转成动作偏差。
- **为什么选 projector/FC–GELU–FC**：论文的稀疏线性探测显示，projector 若干层和早期 Llama 层只需少量通道就能检测触发物体；projector 主要由 FC 与 GELU 组成，局部激活更易分析。GELU 可把负响应压向零，从而衰减背景 token。
- **selector**：在 FC2 输出通道 j 上构造 `z_t(x)=w^T h_t(x)+b`；对触发 token 用 log-mean-exp 池化，让少数高响应 token 主导；对背景 token 施加负阈值抑制，再经 GELU 衰减。
- **amplifier**：冻结 selector 后，只改 FC3 中与通道 j 相连的单个权重，最大化 positive set 上的平均动作 MAE；论文明确说 amplifier 不再额外约束 clean/distractor，选择性主要由 selector 承担。
- **bit search**：每个候选 bit flip 通过一次前向评估；selector 每轮最多考虑 `8,705×16=139,280` 个候选 bit 位置（8,704 权重 + 1 偏置，BF16 16 个 bit 位），保留之前修改，每个 bit 位最多翻一次，直到预算耗尽。

⚡ **Eureka Moment**：把“后门植入”拆成 selector 制造触发器响应、amplifier 放大该响应，并利用现成 FC–GELU–FC 与 GELU 门控，在极少量离散 bit flips 内完成条件破坏。

### 1.3 信息流/架构图 (Flow / Diagram)

```
clean / trigger image + instruction + proprio
        │
        ▼
visual patches (224×224, 14×14 patch, 512 visual tokens across views)
        │
        ▼
FC1 → GELU → FC2(selector channel j) → GELU → FC3(amplifier weight)
        │                         │
        │                         └─ bit flips: selector 60 bits + amplifier 8 bits
        ▼
OpenVLA-OFT: Llama blocks → L1-regression action head
GR00T N1.7: VLM → FC–GELU–FC → action diffusion transformer
        │
        ▼
A(x) ∈ R^{8×7}: 8 future action steps × (3 translation + 3 rotation + 1 gripper)
```

> 时序口径：这里的 8 是未来动作 chunk 的步数，不是历史帧数。若以 Δt 均匀采样 K 个历史帧，首末跨度是 `(K-1)×Δt`；若从当前参考 t 预测 t+1 至 t+H，终点距当前参考是 `H×Δt`。本文没有报告历史帧采样 K、控制频率或记忆窗口，因此不能把 H=8 当作记忆窗口，也不能把单一设置的动作 chunk 长度写成所有实验共有的记忆窗口。

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**：

```
先选 bit：min L_selector = L_positive + λ·L_supp
再选 bit：max E_{x~P}[MAE(x)]
```

**目标**：在有限 bit 预算下，找一组固定权重修改，使得 trigger-present 输入的动作偏离原始策略，同时 clean/distractor 输入尽量保持原行为。

**动作输出与偏差**：

```
A(x) = f_Θ(x) ∈ R^{H×7}，本文所有评估模型 H=8

MAE(x) = 1/(8×7) · Σ_{s=1}^{8} Σ_{k=1}^{7} |A_{s,k}(x) - A_{0,s,k}(x)|
```

**selector 的单 token 响应**：

```
z_t(x) = w^T h_t(x) + b
```

其中 `h_t(x) ∈ R^{8704}` 是 FC2 输入特征，`w,b` 是通道 j 的参数；`T(x)` 是包含至少 5% 触发像素的 token，`B(x)={0,...,511}\T(x)` 是背景 token。

**触发 token 的 log-mean-exp 池化**：

```
r(x,T(x)) = τ_+ · log( (1/|T(x)|) · Σ_{t∈T(x)} exp(z_t(x)/τ_+) )
```

**触发激活损失**：

```
ℓ_positive(x,T(x)) = ( [m_+ - r(x,T(x))]_+ / m_+ )^2
```

这里 `[a]_+ = max(a,0)`。该损失为零的条件是池化响应 `r(x,T(x)) ≥ m_+`；这保证至少一个 trigger token 的 `z_t ≥ m_+`（因为 log-mean-exp 不大于最大 token 响应），但单个 token 达到 `m_+` 本身不保证损失为零，其他 token 仍可能把池化响应拉低。

**背景 token 的逐 token 抑制误差**：

```
e_t(x) = ( [z_t(x) - m_-]_+ / |m_-| )^2
```

当 `z_t(x) ≤ m_-` 时误差为零；超过 `m_-` 则二次惩罚。阈值在 GELU 之前施加，以便压低非触发位置后续激活。

**背景抑制池化**：

```
ℓ_supp(x,B(x)) = τ_- · log( (1/|B(x)|) · Σ_{t∈B(x)} exp(e_t(x)/τ_-) )
```

**总目标**：

```
L_positive(θ) = E_{x~P}[ℓ_positive(x,T(x);θ)]

L_supp(θ) = (1/3) · Σ_{S∈{P,N,C}} E_{x~S}[ℓ_supp(x,B(x);θ)]

L_selector(θ) = L_positive(θ) + λ·L_supp(θ)
```

**离散 bit 搜索**：

```
θ_selector^(k) = argmin_{θ'∈C_k} L_selector(θ')
```

`C_k` 是对当前 selector 参数施加一个允许 bit flip 后的候选集合；每轮选损失最低者，保留之前修改，直到预算耗尽。

**amplifier 目标**：

```
L_amplifier(W) = - E_{x~P}[ MAE(x;W) ]
```

`W` 是 FC3 中与 selector 通道 j 相连的候选权重；冻结 selector，只改该权重的 bit，最大化 positive set 上的平均动作 MAE。

**评测指标**：

```
Δ_s = SR_0^s - SR_atk^s，s ∈ {+, -, c}

ASR = min(1, (Δ_+/SR_0^+) · (SR_atk^c/SR_0^c))

SG = Δ_+/SR_0^+ - Δ_-/SR_0^-
```

`SR_0^+`、`SR_atk^+` 是原始/被改模型在 trigger 条件下的成功率；`-` 是 distractor，`c` 是 clean。ASR 被截断到 1；SG 可以超过 100%，例如附录 Table S2 中 blue tetrahedron 的 SG 为 100.79%，因为 distractor SR 从 96.00% 升到 97.25%。

> 符号与本文一致：`P`、`N`、`C` 分别表示 trigger-present、distractor-only、clean 观测集合；`m_+`、`m_-` 是激活/抑制阈值；`τ_+`、`τ_-` 是池化温度；`λ` 是抑制项权重。默认 `m_+=4.5`、`m_-=-5`、`τ_+=0.5`、`τ_-=0.25`、`λ=3`。

**直觉**：selector 不是“训练一个触发器分类器”，而是用现有视觉特征把某个通道的响应推向“触发 token 高、背景 token 低”；GELU 把负背景压掉。amplifier 再把这个通道的高响应接到动作头上，让 trigger-present 输入的动作矩阵偏离原始策略。因为修改是离散 bit，搜索必须逐候选前向评估，而不能用梯度更新。

## 3. 带数字走一遍：玩具例子 (Worked Example)

> 【假设例子，仅用于验算公式，不是论文实验数据】假设 selector 通道在某个输入上有 2 个 trigger token，`τ_+=0.5`、`m_+=4.5`，FC2 输出为：

```
z = [5.0, -1.0]
```

计算 log-mean-exp：

```
r = 0.5 · ln( (exp(5.0/0.5) + exp(-1.0/0.5)) / 2 )
  = 0.5 · ln( (exp(10) + exp(-2)) / 2 )
  = 0.5 · ln( (22026.4658 + 0.1353) / 2 )
  = 0.5 · ln(11013.3005)
  ≈ 0.5 × 9.30684
  ≈ 4.65342
```

因为 `r ≈ 4.653 > m_+=4.5`，`ℓ_positive=0`。注意：第二个 token 只有 `-1.0`，远低于 `m_+`，但池化仍过阈值；这说明 log-mean-exp 允许少数高响应 token 主导。反过来，如果两个 token 是 `[4.0, 3.0]`：

```
r = 0.5 · ln( (exp(8) + exp(6)) / 2 )
  = 0.5 · ln( (2980.9580 + 403.4288) / 2 )
  = 0.5 · ln(1692.1934)
  ≈ 0.5 × 7.433
  ≈ 3.7165
```

此时 `r < 4.5`，损失为：

```
ℓ_positive = ((4.5 - 3.7165)/4.5)^2
           ≈ (0.7835/4.5)^2
           ≈ 0.0303
```

再假设背景有 2 个 token，`m_-=-5`、`τ_-=0.25`，`z=[-6.0, -4.0]`：

```
e(-6.0) = 0
e(-4.0) = ([-4.0 - (-5)]_+ / 5)^2 = (1/5)^2 = 0.04

ℓ_supp = 0.25 · ln( (exp(0/0.25) + exp(0.04/0.25)) / 2 )
       = 0.25 · ln( (1 + exp(0.16)) / 2 )
       ≈ 0.25 · ln( (1 + 1.1735) / 2 )
       ≈ 0.25 × 0.0832
       ≈ 0.0208
```

这个例子说明：selector 的损失不是“每个 trigger token 都必须超过阈值”，而是池化响应过阈值；背景抑制则逐个 token 惩罚超过 `m_-` 的响应。实际 bit search 会把 `w,b` 的 BF16 bit 逐位翻转，对每个候选做前向，选 `L_selector` 最低者；这里没有梯度，也没有训练轮次。

## 4. 工程视角 (Engineering View)

| 工程问题 | 论文给出的约束 | 工程含义 |
|----------|----------------|----------|
| 修改预算 | 默认 60 selector bits + 8 amplifier bits = 68 bits | 修改量极小，但搜索是离散组合问题；不是训练，不存在 epoch/learning rate。 |
| 搜索成本 | selector 单轮候选上界 `8,705×16=139,280` 个 bit 位置，每个候选一次前向 | 若 60 轮都全量枚举，条件性上界约 8.36M 次前向；实际有通道筛选与预算终止，论文未报告实际搜索吞吐。 |
| 通道筛选 | 从 4,096 个 FC2 输出通道中逐步增加 trial bit budget，保留 selector loss 最低的候选通道 | 这是降低搜索空间的关键工程步骤；候选通道含 8,704 权重 + 1 偏置。 |
| 推理开销 | 修改后的权重固定，无在线更新；触发物体只需出现在视野 | 运行时没有额外网络分支；但权重完整性已被破坏，可通过 checksum/hash 监控发现。 |
| 硬件可行性 | 论文以 Rowhammer 等硬件故障注入为动机 | 论文没有在真实 GPU 上执行 Rowhammer，也没有证明能精确落位 68 个 bit；这是未验证假设。 |
| 时序 | H=8 是未来动作 chunk；motivation 中扰动在每个 action prediction 施加 | 不能把 H=8 当成历史上下文或记忆窗口；论文未报告控制频率，不能推算真实时间跨度。 |

**条件性分析**：从部署角度看，这种攻击的危险不在“运行时算力”，而在“权重分发链”。如果攻击者能在模型仓库、OTA 更新或 GPU 内存中改 68 个 bit，且没有完整性校验，那么 clean 评测很可能照常通过，因为触发物体不在评测集里。反过来，ECC 内存、安全启动、权重签名、运行时激活监控都会提高攻击成本；论文没有评估这些防御。

## 5. 数据与评测 (Data & Eval)

### 5.1 数据构造

- 以 LIBERO 场景为基础，构造 300 个 observation groups 用于优化、150 个用于验证；每组包含 1 个 clean 观测、24 个 trigger-present 变体、63 个 distractor-only 变体。
- OpenVLA-OFT 主 LIBERO-Spatial 实验按 demonstration 划分：每个 task 用 10 个 demonstration 做优化、5 个不相交 demonstration 做验证；在轨迹 0%、33%、66% 处采样观测。优化集与验证集还使用不同的插入位置池。
- positive 与 negative 变体从同一个 clean 观测、同一个插入位置生成；positive 含指定 trigger，negative 含 distractor 但不含 trigger。distractor 可能与 trigger 共享颜色或形状，用于测选择性。
- 每张 main/wrist 图像 resize 到 224×224，按 14×14 patch 切分，两个视角共 512 个 visual tokens；patch 中至少 5% 像素属于 trigger 的 token 记为 `T(x)`，其余为 `B(x)`。
- 数据开放范围：候选 `repo_url` 为空；正文没有给出代码仓库或攻击构造数据的公开下载链接。可确认的是评测使用 LIBERO 仿真 suite，攻击构造观测由作者基于场景生成；不能推断任一模型开放了攻击数据。

### 5.2 预训练、仿真评测、真实机器人设置分开写

| 阶段 | 论文设置 | 证据/限制 |
|------|----------|-----------|
| 预训练 | 使用 OpenVLA-OFT 与 GR00T N1.7 作为 frozen base；攻击不重训练、不微调、不投毒训练数据 | 论文未报告两个模型的预训练数据配方、参数开放范围或训练细节；不能把“攻击不需要训练”误读为“模型没有预训练”。 |
| 仿真评测 | LIBERO Spatial、Goal、Object、Long 四个 suite；paired rollout 比较原始与 bit-flipped policy，同一 instruction、初始状态、插入物体配置 | 主 Spatial 每个 policy 18,500 episodes：4,320 trigger、13,680 distractor、500 clean；每个插入物体条件每 task 6 个初始状态，每个初始状态 1 个 seen + 1 个 unseen 插入位置；clean 每 task 50 个初始状态。 |
| 真实机器人 | 正文未报告真实机器人 rollout | 摘要中的 real-world environments 是威胁动机，不是实验设置；不能声称已在真实机器人上验证。 |

### 5.3 指标与主结果

- 指标：rollout success rate（SR）、attack success rate（ASR）、selectivity gap（SG）。ASR 同时看 trigger 条件相对下降与 clean 保留；SG 比较 trigger 与 distractor 的相对 SR 下降。
- 主结果（论文 Table 2）：两个模型 × 四个 LIBERO suite，八个 model–suite 组合上 ASR 90.22%–100%，SG 84.60%–99.52%；trigger SR 降到 0–9.86%；clean SR 最多下降 1.50 个百分点；distractor SR 最多下降 6.00 个百分点。
- 基线（论文 Table 3，OpenVLA-OFT + LIBERO-Spatial，同一 68-bit 预算）：FlipVLA ASR 90.22%、SG 84.60%；BadVLA（BFA-adapt）ASR 7.13%、SG 5.47%；Joint PBS 的 ASR 与 SG 均低于 1%；Random Bit Flips 在三种输入条件下 SR 都降到 0，ASR/SG 为 0。
- 计数口径：Table 3 的 “Ours” 复用 Table 2 的主 Spatial 结果，即每个 policy 18,500 episodes；三个基线各用每个 policy 2,000 episodes，原始与修改 policy 使用匹配场景。这里的单位是 episode/rollout，不是训练轮次。
- LIBERO-Long 上，两个模型在 distractor-only 场景的 SR 本来就低于 clean，即使未攻击也存在 baseline gap；作者认为这可能反映原策略对 distractor 放置的敏感性。

### 5.4 选择性与消融（默认 OpenVLA-OFT + LIBERO-Spatial）

| 实验 | 论文实测 | 解释边界 |
|------|----------|----------|
| 颜色选择性 | red mug 的平均 SR 为 0–1.85%；其他颜色为 64.85–99.10%；orange mug 为 64.85–82.85% | 作者认为 orange 与红色相近，可能出现非预期激活；这是选择性不足的例子。 |
| 位置鲁棒性 | 8 个插入位置下 SR 0–9.72% | 触发物体在测试位置范围内仍有效；超出该范围未验证。 |
| 形状选择性 | 红色 distractor 的 SR 81.25–100%，red mug 为 0% | 说明不只是“红色”触发，还依赖形状/物体特征。 |
| 物体组合 | blue mug + red capsule/box 的 SR 95.37%/94.44%；含 red mug 的组合均为 2.78% | 作者主张 selector 响应 trigger 物体特征，而不是红颜色与 mug 形状的简单共现。 |
| 触发尺寸 | 0.5×：ASR 81.03%、SG 75.11%；1.5×：ASR 91.68%、SG 83.22%；1×：ASR 90.22%、SG 84.60% | 缩小触发仍有效但下降；放大提高 ASR 但 SG 略降，1× 更平衡。 |
| bit 预算 | selector 仅 30 bits 时 trigger SR 低于 10%，clean 近 baseline，distractor SR 高于 85%；继续加预算收益变小，distractor SR 一般改善 | 30 bits 已能形成明显触发破坏；预算分析固定 amplifier 8 bits，图 13 使用 λ=2。 |
| λ 消融 | 60 FC2 bits 时 λ=3 的 positive P5 与 negative P95 间隔最大；80–120 bits 时 λ=2 最大；λ=1 negative P95 较高，λ=10 强抑制 negative 但也降低 positive P5 | 触发激活与背景抑制存在 trade-off，最优 λ 随 bit 预算变化。 |
| 扰动传播 | 恢复 visual K：MAE 0.21760 → 0.00255；恢复 visual V：0.10175；只保留 modified text states：0.21737（99.9%）；恢复 text 为 original、保留其余 modified：0.00458（2.1%）；只保留 modified visual states：0.00288（1.3%） | 作者主张：被改 visual keys 是关键，block 0 后 text 表示携带大部分后续动作偏差。该消融用 500 个 trigger-present validation observations。 |

**GR00T N1.7 的组件消融（Table S1）**：每个 variant 1,000 rollouts（400 trigger、400 distractor、200 clean）。baseline trigger/distractor/clean SR 为 95.00/97.00/96.50%；selector-only 为 95.00/95.25/97.00，ASR 0.00、SG -1.80；amplifier-only 为 94.75/96.75/97.50，ASR 0.27、SG 0.01；selector+amplifier 为 7.50/96.25/97.00，ASR 92.58、SG 91.33。原模型与 selector-only 在 400 个 trigger rollouts 中各成功 380 次，amplifier-only 成功 379 次；两者同时存在时 trigger SR 从 95.00% 降到 7.50%（30/400），clean/distractor 与 baseline 差距在 0.75 个百分点内。作者认为两个组件在该设置下功能耦合，单独一个不能复现 stealthy attack。

**GR00T N1.7 的触发替换（Table S2）**：每个 trigger 2,000 rollouts（400 trigger、400 distractor、200 clean，每个模型版本）。red mug：original 95.00/97.00/96.50 → bit-flipped 7.50/96.25/97.00，ASR 92.58、SG 91.33；blue tetrahedron：97.75/96.00/98.50 → 0.50/97.25/98.50，ASR 99.49、SG 100.79；green cube：96.75/96.00/97.00 → 0.25/96.00/97.00，ASR 99.74、SG 99.74。每个 trigger 使用各自修改后的模型；作者报告 trigger-present 的成功→失败变化分别为 389 次（tetrahedron）和 386 次（cube），两个替换 trigger 的 aggregate clean SR 与原值相同，distractor SR 不低于原值。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

| 能力 | 具体表现 | 边界 |
|------|----------|------|
| 条件任务失败 | trigger 物体出现时 SR 降到 0–9.86%（主表范围） | 是 untargeted failure，不指定错误动作；真实机器人未测。 |
| 跨模型 | OpenVLA-OFT（parallel decoding + L1 head）与 GR00T N1.7（flow-matching diffusion transformer） | 注入位置按模型不同；不能把 OpenVLA-OFT 的 projector 结构套到 GR00T N1.7。 |
| 跨 suite | LIBERO Spatial/Goal/Object/Long | 只在 LIBERO 仿真；Long 上 distractor baseline 本身较低。 |
| 选择性 | 颜色、形状、位置、物体组合、触发尺寸均有测试 | orange mug 等相似颜色会部分激活；distractor SR 最多降 6.00 个百分点。 |
| 可解释消融 | visual K 恢复使 MAE 大幅下降；text states 携带后续影响 | 消融只覆盖部分层/位置；不是完整因果回路。 |

**不能做什么/未验证什么**：

- 没有真实机器人 rollout；物理安全风险是作者的威胁动机，不是实测结论。
- 没有在真实 GPU 内存上执行 Rowhammer 或故障注入；bit flips 是离线搜索后直接改权重，物理落位可行性未证明。
- 没有防御/检测评估：权重签名、ECC、运行时监控、触发器检测均未测试。
- 没有 targeted action：攻击目标是让任务失败，不是让机器人执行指定危险动作。
- 没有记忆机制评估；H=8 只是未来动作 chunk，不是历史上下文或记忆窗口。
- 没有报告控制频率、历史帧采样 K、真实部署延迟；不能从 H=8 推算时间跨度。

### 6.1 隐含假设 (Hidden Assumptions)

1. **分布假设**：攻击构造观测与评测场景足够接近；若真实触发物体在光照、遮挡、视角、材质上偏移，selector 的通道响应是否仍过阈值未验证。
2. **可见性假设**：攻击者能把触发物体放进机器人视野，且不被任务指令/场景清理移除；论文假设触发物体任务无关但可见。
3. **白盒与精确落位假设**：攻击者可访问中间激活并精确翻转指定 BF16 bit；物理故障注入能否稳定命中这些 bit 未证明。
4. **固定权重假设**：推理期间权重不被更新、校验或监控；若部署链有完整性校验，攻击会被发现。
5. **保留性假设**：clean/distractor 性能保留由有限 LIBERO suite 代表；真实世界的长尾场景未覆盖。
6. **不可检测假设**：论文没有评估防御方是否能通过激活异常、触发器检测或权重统计发现后门。

## 7. 与相关工作对比 (Comparison)

| 方法 | 是否需要训练数据/微调 | 修改对象 | 触发条件 | 选择性机制 | 论文报告结果（OpenVLA-OFT + LIBERO-Spatial） |
|------|------------------------|----------|----------|------------|-----------------------------------------------|
| BadVLA（BFA-adapt） | 保留 BadVLA 目标/设置，但用单 bit flip 替代连续更新 | 权重 bit | 数据/目标定义的后门 | 沿用 BadVLA 目标 | ASR 7.13%、SG 5.47%（2,000 episodes/policy） |
| Joint PBS | 不需要训练，联合搜索 FC–GELU–FC | 权重 bit | trigger 输入 | 联合最大化 trigger MAE，约束 clean/distractor | ASR 与 SG 均低于 1%（2,000 episodes/policy） |
| Random Bit Flips | 不需要训练 | 随机权重 bit | 无 | 无 | 三种输入条件 SR 都降到 0，ASR/SG 为 0 |
| FlipVLA（本文） | 不需要重训练/微调/投毒；需 white-box 离线 bit search | FC2 selector + FC3 amplifier | 指定 trigger 物体 | selector 触发激活 + 背景抑制；amplifier 放大 positive MAE | ASR 90.22%、SG 84.60%（复用主 Spatial 18,500 episodes/policy） |

**面试 Tip**：被问到“这篇和 BadVLA 有什么区别”时，一句话答：BadVLA 还是数据/目标层面的后门，FlipVLA 把攻击面换成部署权重的少量离散 bit flips，并用 selector–amplifier 把“触发器响应”和“动作破坏”解耦；但它的物理 bit 注入和真实机器人有效性尚未验证。

## 8. 精读建议 (Reading Guide)

- **值得精读原文的人**：
  1. 做 VLA 部署、模型供应链安全、权重完整性/防篡改的工程师；
  2. 研究 bit-flip attack、Rowhammer、模型后门检测的研究者；
  3. 需要评估“clean 评测通过但触发场景失败”这一风险边界的机器人安全团队。
- **建议章节路径**：先读 §4.2–4.3（selector/amplifier 与 bit search）→ 再看 §5.1–5.3（数据构造、paired rollout、主结果/基线）→ 最后看 §6 消融与附录 Table S1/S2（组件耦合、触发替换）。§2 的公式可在需要复现损失时细读；§7 相关工作可快速扫过。
- **不值得精读的理由**：如果你不做机器人学习、不关心模型安全，或者已经熟悉 bit-flip attack 的基本威胁模型，读摘要与 §5 的主结果即可；论文没有提出新的 VLA 表示学习或控制算法，也没有真实机器人实验。

---

**关键引用**：本次论文全文（arXiv:2611.33810v1，2026-11-25）https://arxiv.org/abs/2611.33810

[← Back to Theory](./README.md)