# HBVLA：将 VLA 模型推向 1-Bit 后训练量化 (Pushing 1-Bit Post-Training Quantization for Vision-Language-Action Models)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-08-24
>
> **论文**: HBVLA: Pushing 1-Bit Post-Training Quantization for Vision-Language-Action Models
> **链接**: https://arxiv.org/abs/2602.13710
> **核心定位**: 解决 VLA 模型在资源受限机器人上部署的内存瓶颈——通过 1-bit 后训练量化（PTQ），将权重内存压缩约 82%，同时保留 92-94% 的全精度性能，显著超越现有二值化基线。

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | VLA 模型可安全压缩至 1-bit 权重精度（PTQ），在 LIBERO/Simpler 上保留 92-94% 全精度成功率，内存降低 82% |
| 適合精讀 | 如果你在部署 VLA 到边缘设备（Jetson/树莓派等），或研究 LLM/VLM 二值化向 VLA 迁移，重点看 §1.2 和 §2 |
| 可以跳过 | 如果你只关心 VLA 架构创新（如新 backbone、新训练范式），这篇是量化工程方向 |
| 落地可行性 | 高（PTQ 无需重训，仅需少量校准数据；推理时激活保持 BF16，仅权重二值化） |
| 主要風險 | 真机成功率下降仍达 12-23 个百分点；仅量化权重（激活仍 BF16），实际加速受内存带宽限制 |

💡 **X-Ray 开场**
VLA 模型动辄 7B 参数，在机器人边缘设备上跑不动。传统量化方法直接套用 LLM 二值化方案会严重损失动作稳定性——因为 VLA 的误差会在闭环执行中累积放大。HBVLA 的核心发现是：把量化问题从「特征重建」重新定义为「策略保持」，用动作损失引导的 Hessian 矩阵识别关键权重，再对非关键权重做 Haar 小波变换后二值化，最终在 1-bit 极端压缩下仍保持可用控制。对 VLA 研究者意味着：极端量化在具身智能领域终于可行，但真机部署仍有 gap。

📍 **研究全景时间线**
```
[2022] BinaryConnect      → [2023] XNOR-Net → [2024] BiLLM/HBLLM (LLM二值化)
                                                                    ↓
[2025] BiVLM (VLM二值化) → [2025] BitVLA/SQIP (VLA QAT, 需重训)
                                                                    ↓
[2026] HBVLA ← 当前位置: 首个 VLA-tailored 1-bit PTQ，无需重训
                              ↑ 局限: 真机仍有 12-23pp 成功率下降
```

## 1. 核心架构/方法总览 (Overview / Architecture)

HBVLA 将 VLA 量化从「特征重建问题」重新定义为「策略保持问题」。传统 LLM/VLM 二值化关注权重矩阵的 Frobenius 范数逼近或 perplexity 不变，但 VLA 的输出是连续动作，量化误差会经物理系统动力学放大并在长视界闭环执行中累积。HBVLA 的三步流水线直接针对这一特性设计。

### 1.1 系统对比概览 (System Component Comparison)

| 维度 | HBVLA | BiLLM (LLM基线) | HBLLM (LLM基线) | BiVLM (VLM基线) | BitVLA/SQIP |
|------|-------|-----------------|-----------------|-----------------|-------------|
| 量化类型 | PTQ (1-bit weight) | PTQ (1-bit) | PTQ (1-bit) | PTQ (1-bit) | QAT (需重训) |
| 是否需要训练数据 | ❌ 否 | ❌ 否 | ❌ 否 | ❌ 否 | ✅ 是 (大量机器人数据) |
| 激活精度 | BF16 (仅权重二值化) | BF16 | BF16 | BF16 | 通常更低 |
| 权重敏感识别 | 动作感知 Hessian | 通用 Hessian (H=XX^T) | OBQ-based | 通用 Hessian | 训练时学习 |
| 非关键权重处理 | 稀疏正交变换 + Haar | Haar (无排列) | Haar | Haar (无VLA适配) | N/A |
| 关键权重处理 | 残差 + 列Haar | 全局Haar | 全局Haar | 全局Haar | N/A |
| LIBERO (π₀.₅ LM+ViT) | 92.7% | 85.4% | 87.3% | 86.1% | N/A (QAT不同设定) |
| Simpler Visual Matching | 70.0% | 62.3% | 64.1% | 63.5% | N/A |
| 内存压缩 | ~82% | ~80% | ~80% | ~80% | 取决于bit配置 |
| 推理加速 | 2.93x | ~2x (估计) | ~2x (估计) | ~2x (估计) | N/A |

### 1.2 关键机制 (Key Mechanism)

HBVLA 的三个核心组件形成闭环：

1. **策略感知权重分区**：用动作损失反向传播得到的 token 重要性分数修正 Hessian 矩阵，将权重列分为「关键」（salient）和「非关键」（non-salient）两组。关键列的量化误差对动作输出影响最大，需要更精细的处理。

2. **稀疏正交变换（非关键权重）**：VLA 的多模态权重在矩阵空间中交错排列，直接做 Haar 变换会产生跨模态高频噪声。先用贪心配对-链式启发式算法找最优排列矩阵 P，让相似列相邻，再做 Haar 变换，大幅降低高频能量。

3. **Salience-Adaptive Haar 域量化**：关键权重和非关键权重都使用 Haar 小波变换 + 分组 1-bit 量化，但策略不同——非关键权重用行变换 + 共享均值减少元数据开销；关键权重用列变换 + 残差量化补偿非关键部分的逼近误差。

⚡ **Eureka Moment**：VLA 二值化的核心不是「重建权重矩阵」，而是「保持策略输出」——用闭环动作偏差（而非特征重构误差）来指导哪些权重该保留、哪些可以激进压缩，这是 HBVLA 超越所有 LLM/VLM 二值化基线的根本原因。

### 1.3 信息流/架构图 (Flow / Diagram)

```
校准数据 (少量 trajectory)
         │
         ▼
┌─────────────────────────────────────────────────┐
│  Step 1: 策略感知权重分区                         │
│                                                 │
│  Forward:  全精度 Φ(X) → a    vs  二值 Φ̂(X) → â  │
│  Loss:     L_act = Σ ρ̄_j · (â_j - a_j)²         │
│  Backward: ∂L_act/∂Y^(p) → token重要性 s_t         │
│  Hessian:  H̃ = X·S·X^T = Σ s_t · x_t·x_t^T       │
│  Partition: → I_sal (关键列) + I_non-sal (非关键)  │
└─────────────────────────────────────────────────┘
         │                    │
         ▼                    ▼
┌─────────────────┐  ┌──────────────────────────┐
│ Step 2: 非关键   │  │ Step 3: 关键权重          │
│ 权重二值化       │  │ 残差二值化                │
│                 │  │                          │
│ W_filled · P    │  │ R = W - Ŵ_non-sal       │
│ · H_m (行Haar)  │  │ R(:,I_sal)·H_n (列Haar)  │
│ → Q(·) → U_B    │  │ → Q(·) → Ŵ_sal          │
│ → H_m^T·P^T     │  │                          │
└─────────────────┘  └──────────────────────────┘
         │                    │
         └────────┬───────────┘
                  ▼
         Ŵ_l = Ŵ_non-sal + Ŵ_sal
                  │
                  ▼
         量化后 VLA (1-bit weight, BF16 activation)
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：
```
θ̂* = argmin_θ̂  E_x[D_KL(f_θ(·|x) || f_θ̂(·|x))]
     ≈ W_local: min_Ŵ ||WX - ŴX||_F²  (按动作敏感性分区处理)
```

**目标**：找到一组 1-bit 权重 θ̂，使得量化后策略 f_θ̂ 与全精度策略 f_θ 的 KL 散度最小。实际操作中，这被分解为逐层的 Frobenius 范数逼近问题，但按动作敏感性分区处理。

**关键公式链**：

```
(1) 漂移加权动作损失:
    L_act = Σ_{j=1}^{d_a} ρ̄_j · (â_j - a_j)²

(2) 漂移敏感度 (Jacobian 范数归一化):
    ρ_j = E_t [||J_{:,j}^(t)||₂]
    ρ̄_j = ρ_j / (1/d_a · Σ_k ρ_k + ε)

(3) Token 重要性 (从 Q,K,V,O 投影梯度平均):
    G^(p) = ∂L_act / ∂Y^(p)        (p ∈ {Q,K,V,O})
    a_t^(p) = (1/d_p) · ||G^(p)_{:,t}||₂²
    s_t = (1/|P|) · Σ_p a_t^(p)

(4) 修正 Hessian:
    H̃ = X·S·X^T = Σ_t s_t · x_t·x_t^T

(5) Haar 变换 (单层正交归一化):
    H(w) = w·H_m = [w_lo, w_hi]
    w_k^lo = (w_{2k-1} + w_{2k}) / √2
    w_k^hi = (w_{2k-1} - w_{2k}) / √2

(6) 分组 1-bit 量化:
    Q(u) = α_g · sign(u - μ_g)

(7) 非关键权重: Ŵ_non-sal = U_B · H_m^T · P^T
    其中 U = W_non-sal · P · H_m,  U_B = Q(U)

(8) 关键权重 (残差): Ŵ_sal = H_n^{-1} · Q(H_n · R(:, I_sal))
    其中 R = W - Ŵ_non-sal

(9) 最终重建: Ŵ = Ŵ_non-sal + Ŵ_sal
```

**变量说明**：

| 符号 | 含义 | 维度 |
|------|------|------|
| a, â | 全精度/量化后动作向量 | R^{d_a} |
| ρ̄_j | 第 j 维动作的漂移敏感度（归一化 Jacobian 范数） | 标量 |
| X | 校准激活矩阵 | R^{d×N} |
| s_t | 第 t 个 token 的重要性分数 | 标量 |
| H̃ | 修正 Hessian 代理 | R^{d×d} |
| P | 稀疏正交排列矩阵 | {0,1}^{m×m} |
| H_m | 归一化 Haar 变换矩阵 | R^{m×m} |
| μ_g, α_g | 分组均值和缩放因子 | 标量 |

> 符号与本文保持一致。H_m 表示 Haar 变换矩阵（行方向），H_n 表示列方向 Haar 变换。两者都是正交归一化的。

## 3. 带数字走一遍：玩具例子 (Worked Example)

假设一个简化场景：某线性层权重 W ∈ R^{4×4}，校准激活 X ∈ R^{4×3}（3 个 token），动作维度 d_a = 2。

**Step 1: 计算漂移敏感度**

假设 Jacobian 范数：ρ = [3.0, 1.0]（第 1 维动作对末端执行器影响更大）
```
ρ̄ = [3.0/(2.0+ε), 1.0/(2.0+ε)] ≈ [1.5, 0.5]
```

**Step 2: 计算动作损失与 token 重要性**

假设全精度动作 a = [0.8, 0.3]，量化后动作 â = [0.6, 0.5]：
```
L_act = 1.5 × (0.6-0.8)² + 0.5 × (0.5-0.3)²
      = 1.5 × 0.04 + 0.5 × 0.04
      = 0.06 + 0.02 = 0.08
```

反向传播得到 Q 投影梯度 G^(Q) ∈ R^{d_p×3}，假设各 token 的 channel-normalized 敏感度：
```
a_1^(Q) = 0.8,  a_2^(Q) = 0.2,  a_3^(Q) = 0.5
```
平均过 Q,K,V,O 四个投影后得到 token 重要性：
```
s = [0.7, 0.15, 0.45]
```

**Step 3: 修正 Hessian 与权重分区**

```
H̃ = s₁·x₁x₁^T + s₂·x₂x₂^T + s₃·x₃x₃^T
```

token 1 和 3 重要性高 → 对应的权重列被标记为 salient。假设第 1 和第 3 列是 salient：
```
I_sal = {0, 2},  I_non-sal = {1, 3}
```

**Step 4: 非关键权重二值化**

用相邻均值填充 salient 列后，对 non-salient 部分做排列+Haar：
```
W_non-sal ∈ R^{4×2} (列 1 和 3)
→ 排列 P (让相似列相邻)
→ U = W_non-sal · P · H_2
→ 分组量化: U_B = α · sign(U - μ)
→ 重建: Ŵ_non-sal = U_B · H_2^T · P^T
```

**Step 5: 关键权重残差量化**

```
R = W - Ŵ_non-sal
R_sal = R[:, {0,2}]  (提取关键列)
→ 列 Haar: R_sal^c = H_2 · R_sal
→ 量化: Q(R_sal^c)
→ 重建: Ŵ_sal = H_2^{-1} · Q(R_sal^c)
```

**Step 6: 最终重建**

```
Ŵ = Ŵ_non-sal + Ŵ_sal
```

这个两步策略的关键直觉：非关键权重先被粗略量化，关键权重在残差上量化，避免了非关键部分的量化误差干扰关键信号。

## 4. 工程视角 (Engineering View)

| 工程指标 | 数值 | 含义 |
|----------|------|------|
| 权重内存压缩 | ~82% | π₀.₅: 4.6GB→0.83GB; OpenVLA-OFT: 15.2GB→2.74GB; CogACT: 30.5GB→5.50GB |
| 推理加速 | 2.93x | 延迟降低约 65.9%（A800 GPU 上测量） |
| 激活精度 | BF16 | 仅权重二值化，激活保持半精度 |
| 平均权重精度 | 1.02-1.13 bit | 因分组量化元数据（μ_g, α_g）产生少量 overhead |
| 校准数据需求 | 少量 trajectory | 无需大规模训练数据（vs QAT 需要数千 demonstration） |
| 训练需求 | 无 | PTQ 方法，冻结参数上操作 |

**工程含义**：

- **内存带宽是瓶颈**：虽然权重只有 1-bit，但激活仍为 BF16，实际加速比受限于内存带宽而非计算量。在 CPU/边缘设备（如 Jetson Orin）上，权重二值化的 bitwise 运算优势可能更明显。
- **元数据开销可控**：分组量化需要存储每组的 μ_g 和 α_g，但论文证明即使计入这些元数据，平均精度仅 1.02-1.13 bit，存储效率仍然极高。
- **部署约束**：需要实现 Haar 变换 + 二值化的 custom kernel 才能真正获得 2.93x 加速。如果只做软件层面的 weight binarization 而没有 kernel 优化，实际加速会打折扣。
- **校准成本**：需要少量 trajectory 数据做前向+反向传播计算 token 重要性。对于闭源 VLA 或数据受限场景，这可能是一个门槛。

## 5. 数据与评测 (Data & Eval)

**评测设置**：

| 基准 | 环境 | 任务数 | 机器人 | 观察类型 |
|------|------|--------|--------|----------|
| LIBERO | MuJoCo | 100 任务 (4 suites) | Franka Panda | RGB + proprioceptive + delta-action |
| SIMPLER | 高保真仿真 | 4 任务 | Google Robot Arm | RGB |
| Mobile ALOHA | 真机 | 3 任务 | Mobile ALOHA | RGB (30-450 demonstrations/任务) |

**评测模型**：
- π₀.₅（flow matching 架构）
- OpenVLA-OFT（action tokenization 架构）
- CogACT（diffusion policy 架构）

**基线方法**：HBLLM、BiLLM、BiVLM（均为 1-bit PTQ，但非 VLA 专用）

**关键数据点**（来自论文 Tables 1-5）：
- LIBERO π₀.₅ LM+ViT: HBVLA 92.7% vs FP 97.1%（差距仅 4.4pp）
- LIBERO OpenVLA-OFT All: HBVLA 86.0% vs FP 92.2%（差距 6.2pp）
- Simpler Visual Matching CogACT LM+ViT: HBVLA 70.0% vs 最强基线 62.3%（+7.7pp）
- Simpler Visual Matching CogACT All: HBVLA 67.2% vs 最强基线 55.8%（+11.4pp）
- Mobile ALOHA Pick & Place: HBVLA 较 FP 下降 23.4pp（30 trials）
- Mobile ALOHA Sequenced: HBVLA 较 FP 下降 12.5pp（24 trials）

## 6. 能力与失败模式 (Capabilities & Failure Modes)

**能做什么**：
- 将 7B 级 VLA 模型压缩到 1/5 权重内存（82% 压缩），在 Jetson 等边缘设备上可行
- 在仿真环境（LIBERO/Simpler）中保持 90%+ 全精度成功率
- 跨三种 VLA 架构（flow matching、action tokenization、diffusion）通用
- 比通用 LLM/VLM 二值化方法平均多保留 11pp 成功率

**不能做什么**：
- 真机成功率仍有 12-23pp 下降（Mobile ALOHA 实验）—— 对于安全关键任务仍不够可靠
- 长视界任务（LIBERO Long suite）下降更大——量化误差在长 rollout 中仍有累积
- 仅量化权重，激活仍为 BF16——真正的 end-to-end 1-bit inference 仍需等待激活二值化

### 6.1 隐含假设 (Hidden Assumptions)

1. **校准数据代表性**：token 重要性分数 s_t 依赖校准 trajectory 的分布。如果校准数据覆盖的任务/场景与部署场景差异大，saliency 分区可能不准确。
2. **Jacobian 可获取**：漂移敏感度 ρ_j 需要 embodiment-specific Jacobian。对于新机器人平台，需要额外的运动学建模。
3. **局部近似充分性**：方法基于逐层局部优化（Frobenius 范数），而非端到端全局优化。层间误差传播未被显式建模。
4. **BF16 激活假设**：推理时激活保持 BF16。如果未来需要 INT8/INT4 激活量化，当前框架需要扩展。
5. **静态权重假设**：PTQ 只量化静态权重，不处理 LoRA/adapter 等动态参数。对于 fine-tuned VLA，需要额外处理。

## 7. 与相关工作对比 (Comparison)

| 方法 | 类型 | VLA专用 | 需训练 | 核心创新 | 适用场景 |
|------|------|---------|--------|----------|----------|
| HBVLA (本文) | PTQ 1-bit | ✅ | ❌ | 动作感知Hessian + Haar域混合量化 | 边缘部署，无训练数据 |
| BiLLM | PTQ 1-bit | ❌ | ❌ | OBQ-based 二值化 | LLM 压缩 |
| HBLLM | PTQ 1-bit | ❌ | ❌ | Haar 变换 + 分组量化 | LLM 压缩 |
| BiVLM | PTQ 1-bit | ❌ | ❌ | VLM 二值化扩展 | 视觉语言模型 |
| BitVLA | QAT | ✅ | ✅ | 训练时量化 | 有训练数据和算力 |
| SQIP | QAT | ✅ | ✅ | 结构化量化 | 有训练数据和算力 |
| 4-bit VLA QAT | QAT 4-bit | ✅ | ✅ | 低比特训练 | 中等压缩需求 |

**面试 Tip**：当被问到「为什么不能直接把 LLM 二值化方法用在 VLA 上？」时，回答：「LLM 的误差在 token 空间度量，VLA 的误差在连续动作空间且会在闭环执行中累积放大。HBVLA 用动作损失引导的 Hessian 而非通用激活统计来识别关键权重，这是本质区别——量化目标从特征重建变成了策略保持。」

## 8. 精讀建議 (Reading Guide)

**值得精讀原文的人**：
- 需要在边缘设备（Jetson/Raspberry Pi）部署 VLA 的工程师——1-bit 权重压缩 82% 直接决定能否跑在 8GB RAM 的设备上
- 研究 LLM/VLM 二值化向具身智能迁移的研究者——HBVLA 的「策略感知 Hessian」思路可推广到其他模态
- 评估 PTQ vs QAT 技术路线选择的架构师——本文证明 PTQ 在 VLA 上可达 QAT 级别的压缩效果

**建議章節路徑**：
- 先读 §3.1-3.2（策略感知权重分区 + 漂移加权动作损失）——理解核心创新
- 再看 §3.3（Salience-Aware Hybrid Binarization）——理解 Haar 域量化的技术细节
- 可跳 §4.1 实验设置细节，直接看 §4.2 主结果表格 + §4.4 时间内存分析

**不值得精讀的理由**：
- 如果你已有充足算力且不需要边缘部署——QAT 方法（如 BitVLA）可能给出更好的绝对性能
- 如果你不熟悉 Haar 小波变换——附录的数学推导较密集，正文理解核心思想即可
- 如果你关注的是 VLA 模型架构创新（如新 backbone）而非压缩——这篇是正交方向


---
[← Back to Theory](./README.md)

**引用链接**：
- 论文: https://arxiv.org/abs/2602.13710
- DOI: https://doi.org/10.48550/arXiv.2602.13710
- VLA-Handbook Theory Index: https://github.com/sou350121/VLA-Handbook/tree/main/theory
