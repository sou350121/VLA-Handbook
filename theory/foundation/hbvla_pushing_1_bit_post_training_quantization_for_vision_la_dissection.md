# HBVLA：将 VLA 模型推向 1-Bit 后训练量化 (Pushing 1-Bit Post-Training Quantization for Vision-Language-Action Models)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-08-22
>
> **论文**: HBVLA: Pushing 1-Bit Post-Training Quantization for Vision-Language-Action Models
> **链接**: https://arxiv.org/abs/2602.13710
> **核心定位**: 将 VLA 模型权重压缩至 1-bit 精度，在 LIBERO 上保留 92.2% 的全精度性能，同时减少 82% 的权重内存占用，解决 VLA 在资源受限机器人上的部署瓶颈。

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | 1-bit PTQ 可安全用于 VLA 模型：HBVLA 通过策略感知的 Hessian 分区 + Haar 小波变换 + 混合二值化，在 3 个 VLA 架构上平均超越最强基线 11.0 个百分点 |
| 適合精讀 | 如果你在做 VLA 边缘部署、机器人量化、或超低比特模型压缩，重点看 §1.2（方法）和 §5（实验数据） |
| 可以跳過 | 如果你只关心 QAT 量化或 4/8-bit PTQ，这篇距离中等——它专注的是 1-bit 极端压缩 |
| 落地可行性 | 高（纯 PTQ，无需训练数据；推理时激活保持 BF16，仅权重二值化） |
| 主要風險 | 真实机器人上仍有 12.5-23.4pp 成功率下降；salient/non-salient 分区比例需手动调优 |

💡 **X-Ray 开场**
VLA 模型（如 OpenVLA、CogACT）参数巨大，直接跑在机器人上内存和算力都不够。把权重压到 1-bit 是最激进的压缩方式，但现有方法会把动作信息毁掉——量化误差在闭环执行中不断累积，导致机械臂振荡或轨迹漂移。HBVLA 的核心发现是：不是所有权重都同等重要——用动作损失反向传播来识别"对动作生成真正关键的权重"，对这些权重用残差量化精细保护，对非关键权重用 Haar 小波变换压缩。结果：内存减少 82%，推理加速 2.93 倍，仿真性能仅降 4-8 个百分点。

📍 **研究全景时间线**
```
[2021] BinaryConnect (纯 QAT 二值化)
  → [2017] XNOR-Net (权重+激活二值化)
  → [2023] BiLLM/HBLLM (LLM 1-bit PTQ)
  → [2024] Bi-VLM (VLM 二值化，忽略激活结构)
  → [2025] BitVLA/SQIP (VLA QAT，需大量数据)
  → [2026-02] HBVLA ← 当前位置：首个 VLA 专用 1-bit PTQ
    ← 局限：真实环境仍有 12-23pp 下降，仅权重量化
```

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 维度 | HBVLA | BiLLM (LLM PTQ) | HBLLM (LLM PTQ) | Bi-VLM (VLM PTQ) | BitVLA/SQIP (VLA QAT) |
|------|-------|------------------|------------------|-------------------|------------------------|
| 量化类型 | PTQ (1-bit 权重) | PTQ (1-bit) | PTQ (1-bit) | PTQ (1-bit) | QAT (需训练) |
| 激活精度 | BF16（未量化） | 1-bit | 1-bit | 1-bit | 低比特 |
| 核心策略 | 策略感知 Hessian + Haar 小波 | OBQ + 组量化 | Haar 频域分解 | Haar 变换 | 训练时量化 |
| 是否需训练数据 | ❌ 否 | ❌ 否 | ❌ 否 | ❌ 否 | ✅ 是（大量机器人数据） |
| 目标保真度 | 动作分布 KL 散度 | 特征重建误差 | 特征重建误差 | 特征重建误差 | 任务成功率 |
| 适用模型 | VLA 专用 | LLM 通用 | LLM 通用 | VLM 通用 | VLA 专用 |
| 内存节省 | ~82% | ~87% | ~87% | ~85% | ~80-90% |

### 1.2 关键机制 (Key Mechanism)

HBVLA 的方法由三个相互关联的组件构成：

**组件 1：策略感知的修正 Hessian（Policy-Aware Rectified Hessian）**
- 传统 Hessian H = XX^T 只看激活统计量，不考虑量化误差对物理动作的后果
- HBVLA 引入 token 重要性分数 s_t，源自"漂移加权动作损失"（drift-weighted action loss）
- 修正 Hessian: H̃ = XSX^T = Σ_t s_t · x_t · x_t^T，其中 S = diag(s_1, ..., s_N)
- 这使 Hessian 估计与动作空间敏感度对齐，而非被背景干扰物的激活主导

**组件 2：结构感知二值投影（Structure-Aware Binary Projection）**
- VLA 的非显著权重列在权重矩阵空间中交错排列（不同模态参数混合）
- Haar 变换在固定局部窗口上操作，跨模态列配对会产生高频阶跃异常
- 解决：在 Haar 变换前应用稀疏正交变换（排列矩阵 P），将相似列配对
- 最小化高频能量：‖W · P · H_hi‖_F^2 = (1/2) · Σ_k ‖W(:,π(2k-1)) - W(:,π(2k))‖_2^2

**组件 3：显著性自适应 Haar 域量化（Saliency-Adaptive Haar Quantization）**
- 显著权重：在残差上做列方向 Haar 变换 → 1-bit 量化 → 逆变换
- 非显著权重：排列 P → 行方向 Haar 变换 → 组内 1-bit 量化 → 逆变换
- 两组共享量化基元：Q(u) = α_g · sign(u - μ_g)

⚡ **Eureka Moment**：把二值量化从"特征重建问题"重新定义为"策略保留问题"——不是最小化权重重建误差，而是最小化量化后策略与全精度策略的动作分布 KL 散度。

### 1.3 信息流/架构图 (Flow / Diagram)

```
校准输入 X (BF16 激活)
       │
       ├──→ [Forward] 局部注意力块 Φ(X) → 全精度动作 a
       └──→ [Forward] 二值化块 Φ̂(X) → 二值动作 â
                │
       ┌────────┘
       ▼
┌─────────────────────────────────────────────┐
│  Backward: ∂L_act/∂Y^(p) for p ∈ {Q,K,V,O} │  ← 局部反向传播
│  → token 重要性 s_t = mean_p (‖G^(p)_,t‖_2) │
└─────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│  修正 Hessian H̃ = X·diag(s)·X^T            │
│  → 列显著性评分 → 分区 I_sal / I_non-sal    │
└─────────────────────────────────────────────┘
       │
       ├──→ [非显著列 I_non-sal] ──────────────────┐
       │      1. 相邻均值填充显著列缺失值            │
       │      2. 排列矩阵 P (贪心配对)              │
       │      3. 行 Haar 变换 → 组 1-bit 量化       │
       │      4. 逆 Haar → 逆排列 → W̃_non-sal      │
       │                                            │
       └──→ [显著列 I_sal] ────────────────────────┐│
              1. 残差 R = W - W̃_non-sal            ││
              2. 列 Haar 变换 → 组 1-bit 量化      ││
              3. 逆 Haar → W̃_sal                   ││
                                                    ││
              W̃_layer = W̃_non-sal + W̃_sal ◄───────┘│
                                                        │
              所有层重复 → 二值化 VLA 模型 ◄───────────┘
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：
```
θ̂* = argmin_θ̂  E_x[D_KL(f_θ(·|x) || f_θ̂(·|x))]
     ≈ 策略感知分区 + Haar域1-bit量化 + 残差补偿
```

**目标**：找到最优二值参数 θ̂*，使量化后策略 f_θ̂ 的动作分布尽可能接近全精度策略 f_θ。

**核心方程**：

```
全局目标: θ̂* = argmin_θ̂ E_{x~D}[D_KL(f_θ(·|x) || f_θ̂(·|x))]

层目标:   min_W̃ ||WX - W̃X||_F^2

漂移加权动作损失: L_act = Σ_j ρ̄_j · (â_j - a_j)^2

漂移敏感度:   ρ_j = E_t[||J_{:,j}^(t)||_2]
归一化:       ρ̄_j = ρ_j / (mean_k(ρ_k) + ε)

Token 重要性:  s_t = (1/|P|) Σ_{p∈{Q,K,V,O}} (1/d_p)·||G^(p)_{:,t}||_2

修正 Hessian:  H̃ = X·S·X^T = Σ_t s_t · x_t · x_t^T

Haar 变换:    H(w) = [w_lo, w_hi]
             w_k^lo = (w_{2k-1} + w_{2k}) / √2
             w_k^hi = (w_{2k-1} - w_{2k}) / √2

量化基元:    Q(u) = α_g · sign(u - μ_g)

非显著量化:  Û = Q(W_non-sal · P · H_m)
             W̃_non-sal = Û_B · H_m^T · P^T

显著量化:    R = W - W̃_non-sal
             W̃_sal = H_n^{-1} · Q(H_n · R(:, I_sal))

最终重建:    W̃ = W̃_non-sal + W̃_sal
```

> 符号说明：
> - W: 全精度权重矩阵 (d × m)
> - X: 校准激活 (d × N)，N 为 token 数
> - s_t: token t 的重要性分数（来自动作损失反向传播）
> - H̃: 修正 Hessian 代理 (N × N)
> - I_sal / I_non-sal: 显著/非显著列索引集
> - P: 排列矩阵（稀疏正交，{0,1} 元素）
> - H_m: m 维归一化 Haar 变换矩阵
> - α_g, μ_g: 组 g 的缩放因子和均值
> - J^(t): 运动学 Jacobian，映射动作扰动到末端执行器偏差

## 3. 带数字走一遍：玩具例子 (Worked Example)

假设一个简化的 2×4 权重矩阵 W（2 行 4 列），校准激活 X 给出 3 个 token 的重要性分数 s = [0.8, 0.1, 0.6]。

```
步骤 1: 修正 Hessian
H̃ = 0.8·x₁x₁^T + 0.1·x₂x₂^T + 0.6·x₃x₃^T
→ token 1（s=0.8）贡献最大，token 2（s=0.1）几乎被忽略
→ 列显著性评分: [col0=0.72, col1=0.35, col2=0.68, col3=0.25]

步骤 2: 分区（假设阈值取中位数）
I_sal = {col0, col2}   ← 高重要性 token 关联的列
I_non-sal = {col1, col3} ← 低重要性列

步骤 3: 非显著量化
W_non-sal = [[w01, w03], [w11, w13]]
→ 排列 P: 假设 col1 和 col3 差异大，P 交换使相似列配对
→ Haar: w_k^lo = (a+b)/√2, w_k^hi = (a-b)/√2
→ 假设 w01=0.5, w03=-0.3:
     w_lo = (0.5 + (-0.3))/√2 = 0.141
     w_hi = (0.5 - (-0.3))/√2 = 0.566
→ 量化: Q(w_lo) = α·sign(0.141 - μ), Q(w_hi) = α·sign(0.566 - μ)
→ 假设 μ=0.1, α=0.4:
     Q(w_lo) = 0.4·sign(0.041) = +0.4
     Q(w_hi) = 0.4·sign(0.466) = +0.4
→ 逆变换重建: Û_B · H_m^T

步骤 4: 显著残差量化
R = W - W̃_non-sal
→ 对 R(:, I_sal) 做列 Haar + 1-bit 量化
→ 残差补偿了非显著量化引入的误差

步骤 5: 合并
W̃ = W̃_non-sal + W̃_sal
→ 最终 1-bit 权重，推理时 sign(u) 输出 ±1，乘以 α_g 恢复尺度
```

**关键直觉**：显著列（与高动作敏感度 token 关联）获得"残差保护"——非显著量化先做粗近似，显著量化再补残差细节。这类似于 JPEG 压缩中的低频+高频分层编码。

## 4. 工程视角 (Engineering View)

| 指标 | 数值 | 工程含义 |
|------|------|----------|
| 权重内存 (π0.5) | 4.6GB → 0.83GB | 单卡 8GB GPU 可装载 π0.5 全模型（原需 A100-80GB） |
| 权重内存 (OpenVLA-OFT) | 15.2GB → 2.74GB | 可从 8GB 边缘设备（如 Jetson Orin）运行 |
| 权重内存 (CogACT) | 30.5GB → 5.50GB | 扩散策略模型也能边缘部署 |
| 推理加速 | 2.93× | 延迟降低 ~65.9%，接近 10Hz 控制频率 |
| 激活精度 | 保持 BF16 | 仅权重二值化；若激活也量化可进一步加速，但风险未知 |
| 量化元数据开销 | 每组 α_g + μ_g | 组大小影响存储效率：组越小精度越高但元数据越多 |
| 校准数据需求 | 少量（PTQ 典型几十到几百样本） | 不需要大规模机器人数据集（区别于 QAT） |

**部署约束**：
- 当前方案仅量化权重，激活保持 BF16。在边缘设备上，激活的内存占用可能成为新的瓶颈
- Haar 变换引入 O(d) 额外计算，但相比矩阵乘法可忽略
- 排列矩阵 P 的贪心配对需要 O(m^2) 预处理，但只需在量化时做一次

**Trade-off 分析**：
- 显著列比例：论文通过"最小化局部二值化重建目标"自动确定。比例过高 → 压缩率下降；过低 → 动作精度损失
- 组大小：论文未给出具体组大小数值（TODO: 待补充），但提到非显著权重在同一行+频带内共享 μ，以减小元数据开销

## 5. 数据与评测 (Data & Eval)

### 评测环境

| 环境 | 任务数 | 机器人平台 | 观察模态 | 评估指标 |
|------|--------|-----------|---------|---------|
| LIBERO | 100 tasks (4 suites) | Franka Panda | RGB + proprioceptive + delta-action | 成功率 (%) |
| SIMPLER | 4 tasks | Google Robot Arm | RGB | 成功率 (%) (VM + VA) |
| Mobile ALOHA (真实) | 3 tasks | Mobile ALOHA 平台 | RGB | 成功率 (%) (30/24 trials) |

### 主要结果（来自论文 Table 4）

**LIBERO 上 LM+ViT 量化**：
- π0.5: HBVLA 92.7% vs FP 97.1%（差距 4.4pp）
- OpenVLA-OFT: HBVLA 92.2% vs FP 96.5%（差距 4.3pp）

**SIMPLER Visual Matching 上全模型量化**：
- CogACT: HBVLA 67.2% vs 最强基线 55.8%（+11.4pp）

**SIMPLER Variant Aggregation 上全模型量化**：
- CogACT: HBVLA 65.1% vs 最强基线 53.6%（+11.5pp）

**Mobile ALOHA 真实机器人（OpenVLA-OFT 全模型量化）**：
- Pick and Place: HBVLA 较 FP 下降 23.4pp（30 trials）
- Sequenced Instruction: 下降 12.5pp（24 trials）
- Flexible Folding: 下降 16.6pp（24 trials）
- 尽管有差距，HBVLA 在所有二值 PTQ 基线中表现最好

### 组件级敏感度（来自论文 Tables 1-3）

| VLA 组件 | 量化敏感度 | 原因 |
|----------|-----------|------|
| Vision Encoder | 低（最鲁棒） | 视觉特征提取对量化误差容忍度高 |
| Adapt/Projector | 中 | 跨模态对齐，部分敏感 |
| Language Model | 中 | 语言理解有一定冗余 |
| Action Head | 高（最敏感） | 直接输出连续动作，误差无后续补偿 |

## 6. 能力与失败模式 (Capabilities & Failure Modes)

### 能做什么
- **桌面操作的仿真环境**：LIBERO 和 SIMPLER 上成功率下降仅 4-8pp，基本可用
- **大参数 VLA 的边缘部署**：CogACT 30.5GB → 5.5GB，可在 Jetson 级别设备上运行
- **跨架构泛化**：在 π0.5（流匹配）、OpenVLA-OFT（tokenized action）、CogACT（扩散策略）三种不同 VLA 架构上均有效
- **真实机器人基础操作**：Pick and Place 等简单任务在真实环境仍有一定成功率

### 不能做什么
- **复杂真实机器人任务**：Sequenced Instruction 和 Flexible Folding 下降 12-17pp，说明多步骤/柔性操作对量化更敏感
- **长程任务的高可靠执行**：量化误差在闭环中累积，长 horizon 任务风险更高
- **激活量化**：当前仅量化权重，激活保持 BF16。端到端 1-bit 推理尚未实现

### 6.1 隐含假设 (Hidden Assumptions)

1. **校准集代表性假设**：PTQ 依赖校准集计算 Hessian 和 token 重要性。如果校准集的任务分布与部署环境不匹配，量化参数可能次优。论文未讨论校准集大小/多样性的影响。

2. **Jacobian 可获取假设**：漂移敏感度 ρ_j 需要运动学 Jacobian。这假设已知机器人运动学模型。对于未知构型或柔性体机器人，Jacobian 可能难以获取。

3. **BF16 激活假设**：推理时激活保持 BF16。这意味着内存和计算加速主要来自权重压缩，而非端到端低比特推理。如果未来要量化激活，方法需要重新设计。

4. **Salient/Non-salient 二分假设**：将权重简单分为两类可能过于粗糙。实际敏感度可能是连续分布，硬划分可能丢失中间梯度的信息。

5. **真实环境数据有限**：Mobile ALOHA 仅 3 个任务、每任务 24-30 次试验。样本量不足以统计显著地评估真实部署可靠性。

## 7. 与相关工作对比 (Comparison)

| 方法 | 量化类型 | 目标保真度 | 是否需要训练 | VLA 专用 | 平均超越 FP 差距 |
|------|---------|-----------|-------------|---------|----------------|
| **HBVLA** (本文) | PTQ, 1-bit 权重 | 动作分布 KL | ❌ | ✅ | 仿真 4-8pp / 真实 12-23pp |
| BiLLM | PTQ, 1-bit | 特征重建 | ❌ | ❌ | 仿真 15-25pp |
| HBLLM | PTQ, 1-bit | 特征重建 | ❌ | ❌ | 仿真 12-20pp |
| Bi-VLM | PTQ, 1-bit | 特征重建 | ❌ | ❌ | 仿真 14-22pp |
| BitVLA | QAT, 低比特 | 任务成功率 | ✅ | ✅ | 需大量训练数据 |
| SQIP | QAT, 低比特 | 任务成功率 | ✅ | ✅ | 需大量训练数据 |

**面试 Tip**：当被问到"为什么不能直接把 LLM 的二值化方法用到 VLA 上？"——回答："LLM 的量化误差在 token/特征空间衡量，VLA 的误差在物理动作空间衡量。小动作偏差通过接触动力学会被放大并累积，导致振荡或轨迹漂移。HBVLA 用动作损失反向传播来指导量化，而不是用特征重建误差。"

## 8. 精讀建議 (Reading Guide)

- **值得精讀原文的人**：
  1. 做 VLA 边缘部署的研究者/工程师——这是首个 VLA 专用 1-bit PTQ 方法
  2. 做模型量化但想扩展到具身智能领域的研究者——策略感知 Hessian 的思路可迁移
  3. 评估机器人平台硬件需求的系统工程师——内存/延迟数据可直接用于容量规划

- **建議章節路徑**：先讀 §Methodology（公式 1-16 是核心）→ 再看 §Experiment Table 4（SOTA 对比）→ 可跳 §Related Work（背景信息，不影响方法理解）

- **不值得精讀的理由**：如果你不做机器人部署、已熟悉 Haar 变换二值化（HBLLM）、或不关心 1-bit 极端压缩——读摘要和 Table 4 即可


---
[← Back to Theory](./README.md)

**关键引用**：
- 论文: https://arxiv.org/abs/2602.13710
- LIBERO: https://libero-project.github.io
- SIMPLER: https://simpler-env.github.io
- Mobile ALOHA: https://mobile-aloha.github.io
