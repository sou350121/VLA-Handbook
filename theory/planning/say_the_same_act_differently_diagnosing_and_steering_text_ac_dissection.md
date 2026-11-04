# 说一样的话，做不同的事：推理 VLA 的文本—动作背离诊断与引导 (Say the Same, Act Differently: Diagnosing and Steering Text-Action Divergence in Reasoning VLAs)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-11-04
>
> **论文**: Say the Same, Act Differently: Diagnosing and Steering Text-Action Divergence in Reasoning VLAs
> **链接**: https://arxiv.org/abs/2611.04166
> **核心定位**: 它把“rationale 可读”拆成可检验的几何问题：动作最敏感方向几乎正交于文本损失子空间，因此文本逐字不变时动作仍可被推入碰撞；TNAS 既做诊断，也做输入级引导。

> **证据口径**：以下只依据 arXiv:2611.04166v1 的 HTML 全文（正文标头 cs.LG，2026-11-03；https://arxiv.org/html/2611.04166）。候选摘要片段与正文摘要一致，未用于补写；来源记录的分类标注 cs.RO/cross 与正文标头 cs.LG 不一致，属于元数据差异。若官方 PDF 与元数据摘要冲突，应以已验证身份的 PDF 正文为准；本文未使用候选摘要替代正文主张。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 论文实测：在 Alpamayo 与 InstructVLA 的共享 hidden state 上，动作最敏感方向与文本损失子空间的范数重叠仅 1.87% / 5.28%；沿文本中性残差扰动可大幅改动作而几乎不改文本。 |
| 适合精读 | 做 reasoning VLA 安全、可解释性、runtime monitor、自动驾驶开环评测的人；重点看 §2 和 §3。 |
| 可以跳过 | 只关心真实机器人闭环控制或训练新 VLA 的人；本文没有真实机器人实验，也没有训练方法。 |
| 落地可行性 | 中：诊断思路和“动作通道直检”可落地；攻击复现受白盒梯度、gated 数据和数字扰动限制。 |
| 主要风险 | 结论只覆盖两个模型与开环评测，不能外推为所有 reasoning VLA 的普遍定律；文本中性是一阶局部性质。 |

💡 **X-Ray 开场**

这篇论文问：模型嘴上说“停车保持距离”，动作一定停吗？它发现视觉扰动可以让 rationale 逐字不变、动作却撞上去；根因是文本和动作共享 hidden state，但动作最敏感方向几乎不在文本损失子空间里。对 VLA 研究者，这意味着 rationale 监控只能当“有用但不充分”的证据，安全门必须直接看动作通道。

📍 **研究全景时间线**

```
2022  RT-1 等早期 VLA
  → 2023  RT-2 / PaLM-E 把 VLM 知识迁到机器人
  → 2025  Alpamayo-r1、InstructVLA 等 reasoning VLA 把 rationale 与动作并列输出
  → 2026  本文 TNAS：诊断文本—动作背离并做输入级引导  ← 当前位置
局限：只验证两个模型；无真实机器人闭环；无记忆机制评估。
```

## 1. 核心架构/方法总览

### 1.1 系统对比概览

| 维度 | Alpamayo | InstructVLA |
|------|----------|-------------|
| 任务域 | 自动驾驶 | 机器人操作 |
| 模型/检查点 | Alpamayo-1.5-10B | InstructVLA checkpoint（VLM base Eagle2-2B） |
| 视觉输入 | 4 相机 × 4 时间帧 = 16 图/clip；cross-left、front-wide、cross-right、front-telephoto | VLM/SigLIP 视觉流；action expert 另持 clean DINO 输入 |
| 文本输出 | 驾驶 rationale（reasoning token） | 任务意图句，平均 9.60 词，一个句子覆盖 16-step chunk |
| 动作头 | flow-matching，10-step Euler；64×2 加速度—曲率控制积分成 2D waypoints | flow-matching action expert；16×7 normalized 7-DoF end-effector |
| 动作读出 | 64 waypoints @ 10 Hz = 6.4 s，单位米 | 16 chunk steps，normalized 单位，裁剪/反归一化前 |
| 本文 hook | 层 32–35，reasoning token | 层 14–17，response token |
| 数据开放 | gated NVIDIA PhysicalAI-Autonomous-Vehicles；checkpoint 非商业研究许可；推理代码 Apache-2.0 | VLA-IT/Bridge 评测样例；资产 license 截至 2026-11 未明确；Eagle2-2B 卡为 CC-BY-NC-4.0 |
| 评测 | 200 clips 探针；154 有障碍标注做开环碰撞；1,181 manifest 做 benign reader | 200 个 Bridge-distribution 样例；fixed-prefix + free generation；无真实机器人 |

### 1.2 关键机制

- **共享接口 + 多对一读出**：文本和动作都读同一 hidden state；rationale 是离散的 many-to-one readout，许多 hidden state 可以映射到同一句话，因此“文本没变”不保证“状态没变”。
- **用动作 Jacobian，而不是动作位移梯度**：动作位移在 clean 点为零，梯度未定义；TNAS 在 Alpamayo 上求动作输出对扰动 δ 的 Jacobian 的主右奇异向量，得到局部动作最敏感方向 d_act；InstructVLA 则用归一化高斯输出向量 w0 的 VJP 构造 d_act = normalize(J_A^T w0)。
- **文本侧选在 clean 点仍有梯度的损失**：teacher-forced KL 在 clean 点梯度为零，hinge 会饱和；论文用 target NLL 与 target-margin drop，二者在 clean 点非零，Gram–Schmidt 后得到 rank-2 文本子空间 T。
- **投影出文本分量**：d_TNAS 是 d_act 去掉 T 后的残差；一阶上它不改变 NLL/margin，却仍能推动动作。
- **输入级引导**：攻击目标加 cos(Δh, d_TNAS) 项，把图像扰动引起的 hidden-state 位移拉向文本中性方向，缓解动作项与文本惩罚项的梯度冲突。

⚡ **Eureka Moment**：不是“文本可能不可信”这种泛泛判断，而是“动作最敏感方向与文本损失子空间几乎正交”——所以可以构造一个一阶文本中性、却能改变动作的方向。

### 1.3 信息流/架构图

```
已观测输入帧（论文称 4 temporal frames，未给 Δt）        未来预测
Alpamayo: 4 相机 × 4 帧 ─┐
InstructVLA: 当前图像 ───┤
                         ▼
                 vision encoder
                         ▼
          shared hidden state（hook 层）
                 ├── text head ──► rationale / response（离散、多对一）
                 └── action head ─► 未来 64 waypoints / 16-step chunk

TNAS:
  J_A → d_act；g_NLL, g_margin → T
  d_TNAS = normalize((I - P_T) d_act)
  image δ 经 PGD，加 λ_cos · cos(Δh, d_TNAS) 引导
```

## 2. 数学核心

📌 **Napkin Formula**：

```
d_TNAS = normalize((I - P_T) d_act),   ρ = ||P_T d_act|| / ||d_act||
```

目标是找“局部动作最敏感、且与文本损失一阶正交”的方向。核心公式如下：

```
J_A = ∂A_hook(δ) / ∂δ  在 δ=0
d_act = normalize(J_A 的主右奇异向量)        # Alpamayo
d_act = normalize(J_A^T w0)                    # InstructVLA，w0 为归一化高斯输出向量

L_NLL(δ) = (1/|R|) Σ_{t∈R} -log p_adv(t, y_t)
L_margin(δ) = (1/|R|) Σ_{t∈R} (m_cln(t) - m_adv(t))
T = span{g_NLL, g_margin}
P_T = 到 T 的正交投影
d_TNAS = normalize((I - P_T) d_act)

ρ = ||P_T d_act|| / ||d_act||
```

| 符号 | 含义 |
|------|------|
| δ | 对 hook 层 hidden state 的加性扰动；每层一个向量，广播到 token span R |
| A_hook | hooked forward 的扁平动作输出：Alpamayo 64×2 waypoints；InstructVLA 16×7 |
| J_A | 动作输出对 δ 的局部 Jacobian |
| d_act | Alpamayo 为 J_A 的主右奇异向量；InstructVLA 为 normalize(J_A^T w0)，局部动作最敏感方向 |
| L_NLL / L_margin | clean 点仍有非零梯度的文本损失；KL 与 hinge 在 clean 点梯度为零/饱和，所以不用 |
| T | 两个文本梯度的 span，rank 2 |
| P_T | 到 T 的正交投影 |
| d_TNAS | 去掉文本分量后的动作方向 |
| ρ | 动作方向中落在文本子空间的范数比例 |

直觉：ρ 小表示文本只“看见”动作方向的一小部分；投影后保留 sqrt(1-ρ²) 的动作范数。论文实测 Alpamayo 平均 ρ=1.87%，InstructVLA 平均 ρ=5.28%，两者都保留平均 >99.8% 的动作方向范数。

动作位移与文本漂移的定义：

```
Δ_ADE(δ) = (1/T) Σ_{t=1..T} ||τ_t(δ) - τ_t(0)||_2
Δ_IVLA(δ) = (1/H) Σ_{t=1..H} ||a_t(δ) - a_t(0)||_2

KL_tf(δ) = (1/|R|) Σ_{t∈R} Σ_v p_cln(t,v) [log p_cln(t,v) - log p_adv(t,v)]
```

其中 T=64、H=16；KL_tf 是 forward teacher-forced KL，单位为 nats，按 active token span 平均。

输入级攻击目标：

```
L_act+txt = L_act - λ_txt [ΔL_NLL + Δm]
L_TNAS = L_act+txt + λ_cos cos(Δh, d_TNAS)
δ^{k+1} = Π_S[δ^k + α sign(∇_δ L(δ^k))]
```

直觉：仍是在 L_act+txt 上优化动作与文本项，并额外加 λ_cos cos(Δh, d_TNAS) 引导；它并未把输入/PGD 更新或优化目标限制在文本中性子空间，而是把 hidden-state 位移方向拉向 d_TNAS；cos 项不直接读 logits。

## 3. 带数字走一遍：玩具例子

> 以下为**假设数值**，仅用于验算投影几何，不对应论文实验。

假设 3 维 hidden state，文本子空间由 e1=(1,0,0) 张成。设动作方向 d_act=(0.1, 0.994987, 0)，已近似归一化。则：

```
ρ = |<e1, d_act>| = 0.1 = 10%
残差 r = d_act - 0.1 e1 = (0, 0.994987, 0)
||r|| = 0.994987 = sqrt(1 - 0.1²)
d_TNAS = r / ||r|| = (0, 1, 0)
保留范数比例 = 99.4987%
```

再设 J_A 在 d_TNAS 方向的增益为 2.0，取 δ=0.01·d_TNAS。动作变化约为：

```
||ΔA|| ≈ 2.0 × 0.01 = 0.02
```

文本 NLL 的一阶变化为 <g_text, δ>；若 g_text 沿 e1，而 δ 沿 e2，则一阶变化为 0。二阶项可能非零，但在小步长下很小。这个例子说明：ρ 不需要接近 0，只要文本梯度与动作方向近似正交，就能在文本一阶不变的情况下推动动作。论文的 ρ 比 10% 更小，因此分离更强；但“文本中性”仍是局部、一阶性质，不是全局保证。

## 4. 工程视角

### 4.1 时序：历史帧、未来预测与“记忆窗口”澄清

```
K 帧均匀采样：首末跨度 = (K-1) × Δt
未来 H 步：终点距当前参考 = H × Δt

Alpamayo：H=64，Δt=0.1 s → 终点距当前参考 6.4 s
Alpamayo：K=4，但论文未给帧间 Δt → 不能算首末跨度
InstructVLA：H=16，论文未给 chunk 频率 → 不能换算秒数
```

必须区分：

- **已观测输入帧**：Alpamayo 的 4 个 temporal frames 是输入观测；论文没有说明其中历史/当前的具体划分，也没有给 Δt，因此不能把“4 帧”算成 0.3 s 或任何具体历史窗口。
- **未来预测帧**：Alpamayo 的 64 waypoints 是未来 6.4 s 的预测轨迹；InstructVLA 的 16-step chunk 是未来动作块。二者都是输出区间，不是历史上下文，也不是记忆窗口。
- **记忆机制**：论文没有提出或评估跨回合记忆机制；不能把 64 waypoints 或 16-step chunk 写成“所有实验共有的记忆窗口”。连续未来 chunk 的预测区间不能直接当作历史上下文或记忆窗口。

### 4.2 计算与步数

- 本文没有训练模型参数，也没有 epoch/轮次；文中的“步数”需区分：输入扰动优化的 PGD steps、报告 readout 的 10-step Euler solver，以及 Alpamayo 攻击优化内部的 4-step differentiable flow rollout（seed 42）。
- 每个攻击跑 **200 PGD steps**；每步 forward+backward 穿过 vision encoder、language backbone、action head。
- 报告 readout 用 **10-step Euler solver**；Alpamayo 的攻击优化内部用 **4-step differentiable flow rollout**（seed 42），InstructVLA 的攻击设置未说明使用 4-step 内部 rollout。两个步数不能混。
- Alpamayo：每个 guidance weight 约 **47 GPU-hours / 200 clips**，约 **14 GPU-min/clip**；Action-only 与 Text-preserving 共享一个 **200-clip、约 85 GPU-hours** 的 run。
- InstructVLA：每个攻击 arm 约 **13 GPU-hours / 200 examples**；Probe 与 clean-state amplification 约 **20–24 GPU-hours**。
- 投影审计：每 clip 抽 **1,000 个随机单位向量**，200 clips 共 **200,000 个样本**；Alpamayo 随机控制用 **16 seeds/clip**。

### 4.3 部署约束与数值单位

- 攻击是**白盒数字扰动**：full-image ε=0.05、α=5×10⁻⁴；patch ε=0.5、α=5×10⁻³；不是打印/相机拍摄验证。
- Alpamayo 的 ADE 单位是**米**；InstructVLA 的 action shift 是 **normalized 7-DoF 输出的 ℓ2**，在部署裁剪与反归一化之前，坐标可超出 [-1,1]。因此 **1.73 不是 1.73 米**，不能与 Alpamayo 的米制 ADE 直接比。
- 开环碰撞是代理指标，不是闭环安全证明；真实部署还需要硬件在环与闭环测试。
- 工程建议（我的条件性判断）：rationale monitor 可以保留，但安全门应直接验证动作通道；exact match 不能单独作为动作安全证书。

## 5. 数据与评测

| 层面 | 论文实际设置 | 备注/局限 |
|------|--------------|-----------|
| 预训练 | 不训练新模型；用 Alpamayo-1.5-10B 与 InstructVLA checkpoint | 未报告预训练数据配比、训练轮次；不能从本文推断预训练配方 |
| 离线/仿真评测 | Alpamayo：open-loop logged-scene；200 clips 探针；154 障碍标注做碰撞；1,181 manifest 做 benign reader。InstructVLA：200 个 Bridge-distribution 样例；fixed-prefix + free generation | 不是闭环仿真；没有真实机器人 rollout |
| 真实机器人 | 无 | 不能声称真机安全效果 |

关键指标的条件必须说清楚：

- **Silent collision**：只在 **154 个有障碍标注的 clips** 上算；定义是“clean plan 无碰撞 ∧ attacked plan 有碰撞 ∧ free-generated rationale 与 clean 逐字相同”三者交集。它不是“攻击后碰撞率”本身。论文 Table 1 中 Alpamayo full-image λ=30 的 silent collision 为 52.6%，Text-preserving PGD 为 48.7%。若把附录 A.3 的 attacked-plan collision counts 换算为比例，两者分别约为 61.7%（95/154）与 61.0%（94/154）；这两个比例不是 Table 1 的列，相关配对检验见表 3。两者 silent collision 的配对检验 p=0.451，不能读成显著提升；显著优势在 exact match（89.5% vs 83.0%，p=0.0192）与 KL_tf（p=1.95×10⁻³²）。
- **Exact match**：free generation 下的逐字相等。Alpamayo 用采样解码（top-p=0.98，temperature=0.6，seed 42）；InstructVLA 用 greedy。它不是 teacher-forced exact match。
- **KL_tf**：teacher-forced forward KL，nats，按 active span 平均；与 free-generation exact match 是不同条件。
- **ADE / action shift**：fixed-prefix readout；Alpamayo 64 waypoints @10 Hz，单位米；InstructVLA 16 steps，normalized 单位。

论文实测的关键数字：

- 等范数 q=2：Alpamayo d_TNAS 动作位移 0.142 m，比文本梯度控制大 44.0×–47.8×，比随机方向大 53.9×；其 KL_tf 为 9.87×10⁻⁵，文本梯度控制的 KL 高 269×–277×。
- InstructVLA q=2：动作位移 0.156，比文本梯度控制大 4.64×–5.51×，比随机方向大 12.6×；KL_tf 为 2.20×10⁻²，文本梯度控制高 21.2×–26.3×。
- Alpamayo full-image λ=30：Exact 89.5%，KL_tf 1.03×10⁻¹，silent collision 52.6%；patch 最高 silent collision 35.1%（λ=10），patch λ=30 为 34.4%。
- InstructVLA full-image λ=0.20：Exact 85.0%，action shift 1.73，KL_tf 2.06×10⁻²；patch λ=0.20：Exact 77.5%，action shift 1.32，KL_tf 2.80×10⁻²。

## 6. 能力与失败模式

| 能力 | 证据 |
|------|------|
| 几何诊断 | ρ=1.87%/5.28%；文本子空间只覆盖动作方向很小一部分 |
| 输入级攻击 | Alpamayo full-image silent 52.6%，patch 35.1%；InstructVLA Exact 85.0%，action 1.73 |
| 防御启示 | 文本逐字不变不能认证动作安全；需动作通道直检 |

失败模式与局限：

- **只覆盖两个模型、两个域**：Alpamayo 自动驾驶与 InstructVLA 操作；不能外推到所有 reasoning VLA。
- **白盒数字扰动**：需要梯度；物理世界打印/拍摄未验证；patch 仍是数字 patch。
- **开环评测**：logged actors 按记录轨迹走，不是闭环反应；碰撞率不是事故率。
- **样本与数据开放**：200 clips/examples、154 碰撞 clips；Alpamayo 数据 gated；InstructVLA 资产 license 未明确，复现受限制。
- **一阶局部性**：d_TNAS 只保证 clean 点一阶文本中性；有限步长可能改变文本；exact match 受采样影响。
- **无记忆机制、无真实机器人、无训练方法**：不能把 chunk 当记忆窗口，也不能声称真机安全提升。
- **单位不可直比**：InstructVLA 的 normalized action shift 不是物理米/弧度。

### 6.1 隐含假设

- 共享 hidden state 是文本与动作的真实共同接口；hook 层选择代表该接口。
- 局部 Jacobian 方向在有限扰动下仍有效；ρ 是一阶量，不是全局不变性证明。
- NLL 与 margin drop 能代表“文本变化”；但 KL_tf 与 exact match 是另外的度量，三者不完全等价。
- 随机单位向量与文本梯度是合适对照；但对照的覆盖仍有限。
- 采样解码的 exact match 能代表 rationale 保持；top-p/temperature/seed 固定，换解码设置可能变。
- 开环碰撞是安全代理；真实闭环中其他车辆会反应，碰撞率可能不同。
- 200 样本足以支持文中比例估计；论文报告了部分配对检验，但并非所有指标都有置信区间。
- 攻击者能同步扰动多相机/多帧；物理可行性未验证。

## 7. 与相关工作对比

| 工作 | 关注点 | 是否改文本 | 层级 | 方法 | 局限 |
|------|--------|------------|------|------|------|
| TRAP（据本文 §4 转述） | 用 patch 劫持 CoT 与动作 | 是，文本可见变化 | 输入/推理 | adversarial patch | 留下文本证据 |
| Trinh et al.（据本文 §4 转述） | 直接破坏推理 trace | 是 | 推理 trace | 替换对象名等 | 留下文本证据 |
| Foutter et al.（concurrent） | 观察 Alpamayo 说让行却不让速 | 不一定 | 输出/后训练 | 语义一致性 critic | 不是 hidden-state 几何诊断 |
| Wu et al. | runtime reasoning-action alignment verification | 监控 | 运行时 | 验证 | 本文认为可与 TNAS 互补 |
| 本文 TNAS | 文本逐字不变、动作变 | 否（目标） | shared hidden state | Jacobian + 投影 + PGD cos 引导 | 白盒、两模型、开环 |

**面试 Tip**：如果被问“reasoning VLA 的 rationale 能不能当安全监控”，答：“能提供信息，但不充分。TNAS 说明动作最敏感方向几乎正交于文本损失子空间；文本不变时动作仍可变，所以监控必须直接验证动作通道。”

## 8. 精读建议

- **值得精读原文的人**：做 reasoning VLA 安全/可解释性的研究者；做自动驾驶 VLA 开环安全评测的工程师；要做 runtime action verification 的工程团队。
- **建议章节路径**：先读 §2.1–2.3（TNAS 构造与两个 probe）→ 再看 §3.1–3.2（攻击设置与主表）→ 需要碰撞协议时看附录 A.3 → 熟悉相关工作可跳 §4。
- **不值得精读的理由**：如果你只关心真实机器人闭环控制、训练新 VLA、或跨回合记忆机制，本文没有这些内容；读摘要与 §6 即可。

---
[← Back to Theory](./README.md)

**论文链接**：https://arxiv.org/abs/2611.04166