# 压缩鸿沟：为何离散 Tokenization 限制 VLA 模型Scaling (The Compression Gap: Why Discrete Tokenization Limits Vision-Language-Action Model Scaling)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-04-07
>
> **论文**: The Compression Gap: Why Discrete Tokenization Limits Vision-Language-Action Model Scaling
> **链接**: https://arxiv.org/abs/2604.03191
> **核心定位**: 揭示动作离散化如何通过信息瓶颈阻断视觉编码器升级的收益，为 VLA 架构选型提供理论依据

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | 离散动作 tokenization（如 OAT）引入固定容量码本瓶颈（~80 bits），使视觉编码器升级无法传播到动作执行；连续表示（Diffusion Policy）无此瓶颈 |
| 適合精讀 | 如果你在做 VLA 架构设计、动作表示选型、或研究 scaling law 在具身 AI 中的适用性 |
| 可以跳過 | 如果你只关心应用层调参、不涉及底层架构决策 |
| 落地可行性 | 中（理论指导性强，但需根据你的任务域验证离散/连续表示的 trade-off） |
| 主要風險 | 离散表示在低质量编码器下仍有优势（结构化补偿），不可一概而论 |

💡 **X-Ray 开场**：这篇论文解决什么问题？—— 为什么升级视觉编码器在 VLA 中不像在 VLM 中那样有效？发现了什么？—— 动作离散化引入的信息瓶颈会阻断上游改进。对 VLA 研究者意味着什么？—— 架构选型时，连续动作表示更能受益于视觉基础模型的快速迭代。

📍 **研究全景时间线**

```
[2020] Scaling Laws for LLMs (Kaplan) 
    → [2022] Compute-Optimal Training (Hoffmann)
    → [2025] Diffusion Policy 成为连续动作基线 (Chi et al.)
    → [2026] OAT 离散 tokenization 提出 (Liu et al.)
    → [本文] 发现离散/连续路径的 scaling 不对称性
    ← 当前位置：Physical AI 的 scaling 需要识别瓶颈位置，而非均匀增加容量
```

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 组件 | Diffusion Policy (连续) | OAT (离散) | 关键差异 |
|------|------------------------|------------|----------|
| 动作表示 | 连续向量 $a \in \mathbb{R}^d$ | 离散 token 序列 $T \in \mathcal{V}^{H_l}$ | 有无量化阶段 |
| 信息流 | $O \to Z \to A$ (全连续) | $O \to Z \to T \to A$ (含量化 $Q$) | OAT 多一个瓶颈 |
| 码本容量 | 无限制 | |V|$H_l \approx 1000^8 \approx 80$ bits | 硬上限 |
| 编码器敏感度 | 高（$\Delta_{\text{enc}} = +21\sim26\%$） | 低（$\Delta_{\text{enc}} = +3\sim10\%$） | 本文核心发现 |
| 模型 scaling | 有效（$M\to L\ +12\sim14\%$） | 无效/不稳定 | 瓶颈位置不同 |

### 1.2 关键机制 (Key Mechanism)

**数据处理不等式（Data Processing Inequality）**：对于马尔可夫链 $O \to Z \to A$：

```
I(O;A) ≤ min(I(O;Z), I(Z;A))
```

即观察 O 到动作 A 的互信息受限于路径中最紧的瓶颈。

**连续路径（DP）**：
```
O →[f_enc] Z →[ε_θ] A
```
所有阶段都是连续的，I(Z;A) 无硬上限，瓶颈通常在 I(O;Z)（视觉编码器）。升级编码器直接放宽瓶颈。

**离散路径（OAT）**：
```
O →[f_enc] Z →[Q] T →[T^-1] A
```
量化阶段 Q 引入硬上限：
```
I(O;A) ≤ I(Z;T) ≤ H_l × log₂|V| ≈ 8 × log₂(1000) ≈ 80 bits
```
即使编码器升级提高了 I(O;Z)，额外的信息也会在量化阶段被丢弃。

⚡ **Eureka Moment**：VLA 的 scaling 行为不取决于模型大小或数据量，而取决于**信息瓶颈的位置**——离散动作表示将瓶颈从视觉编码器转移到了码本，使上游改进失效。

### 1.3 信息流/架构图 (Flow / Diagram)

```
┌─────────────────────────────────────────────────────────────────┐
│                    连续路径 (Diffusion Policy)                   │
│                                                                 │
│  观察 O  ──→ [视觉编码器 f_enc] ──→ 表征 Z ──→ [去噪网络 ε_θ] ──→ 动作 A  │
│            ↑ 瓶颈位置 (I(O;Z))        │ 无硬上限                    │
│            │ 升级编码器有效            │                          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      离散路径 (OAT)                              │
│                                                                 │
│  观察 O  ──→ [视觉编码器 f_enc] ──→ 表征 Z ──→ [量化 Q] ──→ Token T ──→ 动作 A  │
│            ↑ 升级编码器无效          │ 硬瓶颈：I(Z;T) ≤ 80 bits      │
│                                     │ 额外信息在此丢弃              │
└─────────────────────────────────────────────────────────────────┘
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：

```
离散路径：I(O;A) ≤ H_l × log₂|V| ≈ 80 bits（硬上限，与编码器质量无关）
连续路径：I(O;A) ≤ I(O;Z)（瓶颈在编码器，升级有效）
```

**变量说明**：
| 符号 | 含义 | 典型值 |
|------|------|--------|
| I(X;Y) | X 与 Y 的互信息（bits） | - |
| H_l | token 序列长度 | 8 (OAT) |
| |V| | 码本词汇表大小 | 1000 (OAT 默认) |
| O | 原始观察（图像） | - |
| Z | 视觉表征（编码器输出） | 64-d (ResNet) / 1152-d (SigLIP) |
| A | 执行的动作 | 7-d (LIBERO) |
| T | 离散 token 序列 | H_l 个离散值 |

**直觉**：想象一条水管，最窄的那段决定了总流量。离散动作表示在中间加了一个固定直径的阀门（码本），即使你把上游水管（视觉编码器）换得更粗，总流量也不会增加。

## 3. 带数字走一遍：玩具例子 (Worked Example)

假设我们有两个视觉编码器：
- **ResNet-18**: 输出 $64$-d 特征，估计 $I(O;Z) \approx 50$ bits
- **SigLIP**: 输出 $1152$-d 特征，估计 $I(O;Z) \approx 150$ bits

**连续路径（DP）**：
```
ResNet: I(O;A) ≤ min(50, ∞) = 50 bits  → 成功率 36.4%
SigLIP: I(O;A) ≤ min(150, ∞) = 150 bits → 成功率 57.6%
增益: Δ_enc = +21.2%
```

**离散路径（OAT）**：
```
ResNet: I(O;A) ≤ min(50, 80) = 50 bits  → 成功率 53.8%
SigLIP: I(O;A) ≤ min(150, 80) = 80 bits → 成功率 57.4%
增益: Δ_enc = +3.6%
```

关键洞察：OAT 在 ResNet 下表现更好（53.8% vs 36.4%），因为结构化 tokenization 补偿了有限的感知信息。但当编码器升级后，DP 能充分利用额外信息（+21.2%），而 OAT 被码本卡住（+3.6%）。

## 4. 工程视角 (Engineering View)

| 工程考量 | 连续路径 (DP) | 离散路径 (OAT) | 含义 |
|----------|---------------|----------------|------|
| 推理延迟 | 10 步去噪 | 自回归生成 H_l 个 token | OAT 可能更快（H_l << 去噪步数） |
| 内存占用 | 连续特征 + 去噪网络 | 码本 + 自回归解码器 | 相近 |
| 量化误差 | 无 | FSQ 量化损失 | OAT 有固有信息损失 |
| 与 LLM 集成 | 需额外适配 | 天然兼容（token 接口） | OAT 优势 |
| Scaling 潜力 | 高（受益于视觉基础模型） | 低（码本饱和） | 本文核心结论 |
| 低质量编码器下表现 | 较差 | 较好（结构化补偿） | OAT 在资源受限时仍有价值 |

**部署约束**：
- 如果你的系统依赖预训练 LLM 做 prefix decoding，离散表示仍有价值（统一语言 - 动作建模）
- 如果你追求长期 scaling、计划持续升级视觉编码器，连续表示更可持续
- 码本容量可调：本文显示增大 |V| 可部分恢复编码器敏感度（|V|=1920 时 ResNet 性能暴露）

## 5. 数据与评测 (Data & Eval)

**基准**: LIBERO-10（10 个任务，每任务 50 个演示），Franka Emika Panda 机械臂，7-d 动作空间

**实验设计**:
- **因子实验**: $2\times2\times2 = 8$ 条件（动作表示 $\times$ 编码器 $\times$ 模型大小）
- **编码器质量梯度**: 4 个编码器（ResNet-18, SigLIP, DINOv2, SigLIP 2）
- **码本大小实验**: 3 个 |V|（1000, 1920, 4375）

**训练细节**:
- AdamW, lr=5e-5 (policy) / 1e-5 (encoder)
- 300 epochs, 单卡 A100
- 评估：每 50 epochs 500 次 rollout，报告峰值成功率

**关键结果**（Table 1, LIBERO-10）:

| 动作表示 | 模型大小 | ResNet-18 | SigLIP | $\Delta_{\text{enc}}$ |
|----------|----------|-----------|--------|-------|
| DP | M | 36.4% | 57.6% | +21.2% |
| DP | L | 44.0% | 70.0% | +26.0% |
| OAT | M | 53.8% | 57.4% | +3.6% |
| OAT | L | 48.0% | 58.4% | +10.4% |

## 6. 能力与失败模式 (Capabilities & Failure Modes)

**连续路径（DP）适合**：
- 视觉编码器质量高且持续升级的场景
- 追求长期 scaling 的系统
- 不需要与 LLM 统一 token 接口的任务

**离散路径（OAT）适合**：
- 视觉编码器质量有限（嵌入式/边缘部署）
- 需要与语言模型统一接口（prefix decoding）
- 推理速度敏感（自回归 token 生成可能快于扩散去噪）

**失败模式**：
- **DP**: 低质量编码器下表现显著落后（ResNet 时 36.4% vs 53.8%）
- **OAT**: 编码器升级后收益被码本瓶颈阻断，长期 scaling 受限

### 6.1 隐含假设 (Hidden Assumptions)

- **假设 1**: 码本容量固定。实际中可动态调整 |V|，但会改变训练/推理复杂度
- **假设 2**: 信息瓶颈是唯一的 scaling 限制因素。实际中优化难度、数据质量也重要
- **假设 3**: LIBERO 基准的结果可泛化到其他任务域。需验证（移动操作、双臂、人形机器人）
- **假设 4**: 峰值成功率是合适的评估指标。方差、鲁棒性未考虑

## 7. 与相关工作对比 (Comparison)

| 工作 | 关注点 | 架构 | 与本文关系 |
|------|--------|------|------------|
| Diffusion Policy (Chi et al., 2025) | 连续动作扩散基线 | 连续去噪 | 本文的连续路径代表 |
| OAT (Liu et al., 2026) | 有序动作 tokenization | 离散 FSQ 量化 | 本文的离散路径代表 |
| Tong et al. (2026) | 多模态预训练 scaling | VAE vs RAE | 发现类似不对称性（VAE 饱和，RAE 持续 scaling） |
| FAST (Pertsch et al., 2025) | 频域动作压缩 | 离散 | 同样受码本瓶颈限制（本文推测） |

**面试 Tip**：被问到"VLA 中离散 vs 连续动作表示如何选择"时，回答："如果追求长期 scaling 且视觉编码器会持续升级，选连续表示（Diffusion Policy）；如果资源受限或需要与 LLM 统一接口，离散表示（OAT）在低质量编码器下仍有优势——关键看信息瓶颈在哪里。"

## 8. 精讀建議 (Reading Guide)

**值得精讀原文的人**：
1. **VLA 架构设计者**：需要在离散/连续动作表示间做选型决策
2. **具身 AI scaling 研究者**：关心 scaling law 在机器人学习中的适用边界
3. **多模态系统工程师**：设计视觉 - 语言 - 动作流水线，需理解瓶颈位置

**建議章節路徑**：
- 先读 §1 Introduction（理解核心问题）→ §2.2 Information Bottleneck（理论框架）→ §4 Results（实验验证）→ §5 Discussion（含义与局限）
- 可跳 §2.1 Action Representations（若已熟悉 OAT/DP 细节）
- 可跳 §3 Experiment Design（若只关心结论，不关心实验细节）

**不值得精讀的理由**：
- 如果你不做机器人学习、只关心应用层 API 调用
- 如果你已确定使用某种动作表示且不计划更改
- 如果你的任务域与 LIBERO 差异极大（如移动导航、人形全身控制），结论需重新验证

---

## 关键引用

- **论文**: https://arxiv.org/abs/2604.03191
- **OAT 原文**: https://arxiv.org/abs/2602.04215
- **Diffusion Policy**: https://arxiv.org/abs/2303.04137
- **Tong et al. (2026) 多模态 scaling**: https://arxiv.org/abs/2603.03276

---

[← Back to Theory](./README.md)
