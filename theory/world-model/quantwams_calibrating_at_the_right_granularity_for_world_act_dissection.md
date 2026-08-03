# QuantWAMs：在世界动作模型上以正确粒度校准 (QuantWAMs: Calibrating at the Right Granularity for World Action Models)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-08-03
>
> **论文**: QuantWAMs: Calibrating at the Right Granularity for World Action Models
> **链接**: https://arxiv.org/abs/2607.28405
> **核心定位**: 针对世界动作模型（WAMs）的迭代去噪+闭环执行特性，提出一套 PTQ 量化框架，在 W4A4 精度下将性能损失控制在 0.2-0.7 个百分点，同时显存降至 FP16 的 29%

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | WAM 的量化决策必须在三个校准上下文中对齐：模型结构（哪些模块可共享统计量）、部署分布（哪些状态在闭环中可达）、联合目标（视频+动作梯度如何交互）——三者缺一即导致精度崩塌 |
| 適合精讀 | 正在做 WAM 部署/模型压缩的研究者；需要把 Diffusion Policy 量化到端侧设备的工程师 |
| 可以跳過 | 只做 VLA 训练不做部署、或只关注 open-loop 量化方法的人 |
| 落地可行性 | 高（已在 AgiBot G2 真机上验证 3 个操作任务；代码开源） |
| 主要風險 | 需要原始 co-training targets 做 backward pass，对闭源 WAM 不适用；校准数据需 benchmark-specific 采集 |

💡 **X-Ray 开场**

WAM（World Action Model）同时预测未来观测和执行动作，但它的迭代去噪+闭环执行让量化部署变得困难——传统 PTQ 方法假设每个前向传播独立评分、模型同构，这在 WAM 上全不成立。QuantWAMs 的核心发现是：**每个量化决策本质上都是部署前的有限样本估计，必须在结构、分布、目标三个轴上与部署上下文对齐**。在 W4A4 量化下，Fast-WAM 在 RoboTwin 2.0 上仅损失 0.2 个百分点，显存降至 FP16 的 29%。

📍 **研究全景时间线**

```
2022 GPTQ (Frantar)     2023 SmoothQuant      2024-25 SVDQuant/Atom
  权重量化               激活平滑             Diffusion PTQ
      │                       │                    │
      └───────────┬───────────┘                    │
                  │                                │
          开放领域 LLM 量化                   扩散模型 PTQ
          假设：独立前向、同构               假设：open-loop 校准
                  │                                │
                  └────────────┬───────────────────┘
                               │
                    2026 QuantWAMs ← 本文
                    关键创新：校准上下文对齐
                    结构×分布×目标 三维约束
                    首次针对 WAM 闭环特性设计
```

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 维度 | GPTQ | SmoothQuant | SVDQuant | Atom | QuantWAMs |
|------|------|-------------|----------|------|-----------|
| 目标模型 | LLM | LLM | Diffusion | Diffusion | **WAM** |
| 校准目标 | 逐层重建 | 激活平滑 | SVD 截断 | 通道统计 | **视频-动作联合梯度** |
| 校准分布 | 独立样本 | 独立样本 | open-loop | open-loop | **闭环 rollout 状态** |
| 模块共享假设 | 同构 | 同构 | 同构 | 同构 | **坐标可容性检验** |
| 去噪步保护 | 无 | 无 | timestep-aware | timestep-aware | **fixed-intervention replay** |
| 理论保证 | 二阶补偿 | 平滑界 | 谱界 | 通道界 | **Prop 1: pooling crossover** |
| W4A4 RoboTwin (Fast-WAM) | N/A | N/A | 63.8% | 71.5% | **91.8%** |

### 1.2 关键机制 (Key Mechanism)

QuantWAMs 围绕一个核心原则构建：**每个 PTQ 决策都是部署前的有限样本估计，必须在校准上下文中保持适当性**。这个上下文由三个维度定义：

1. **结构维度（Structural）**：哪些模块可以共享激活统计量？——只在"坐标可容"（coordinate-admissible）的模块间 pooling
2. **分布维度（Distributional）**：在哪些状态上测量敏感性？——用真实闭环 rollout 状态，而非合成/open-loop 数据
3. **目标维度（Objective）**：用哪个损失函数评分？——用视频-动作联合梯度，而非单流或平方后融合

⚡ **Eureka Moment**：传统 PTQ 把量化当作"模型压缩"问题；QuantWAMs 把它重新定义为"校准上下文对齐"问题——每个决策必须在结构、分布、目标三个轴上与部署环境一致，否则就是无效估计。

### 1.3 信息流/架构图 (Flow / Diagram)

```
                    校准数据准备
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   32 条训练          32 条 FP16       原始 co-
   轨迹 D_cal         closed-loop      training targets
                      rollout D_prof   (backward pass)
        │                │                │
        ▼                ▼                ▼
   ┌─────────┐    ┌───────────┐    ┌──────────┐
   │ Shared- │    │ Fixed-    │    │ Co-Train │
   │ Basis   │    │ Intervention│  │ Saliency │
   │ Outlier │    │ Rollout   │    │ (Weight  │
   │ Calib   │    │ Auditing  │    │  Axis)   │
   └────┬────┘    └─────┬─────┘    └────┬─────┘
        │               │               │
        ▼               ▼               ▼
   Top-K 通道      去噪步保护         层粒度
   混合精度掩码    调度修复           W8/W4 分配
        │               │               │
        └───────────────┼───────────────┘
                        ▼
              ┌─────────────────┐
              │  混合精度 WAM   │
              │  W4A4 + W8 层   │
              │  BF16 异常通道  │
              └────────┬────────┘
                       ▼
              闭环部署 (AgiBot G2)
```

## 2. 数学核心 (Math Core)

### 📌 Napkin Formula（一行抓住本质）

```
每个量化决策 d = (G, D, L) 必须满足:
  G: 证据池化范围与坐标系统一致
  D: 校准分布 ≈ 闭环部署分布
  L: 评分目标包含视频×动作交叉项
任一维度失配 → 部署时决策失效
```

### 2.1 共享基座异常值校准（Shared-Basis Outlier Calibration）

**目标**：决定哪些通道保留高精度（BF16），哪些量化到低精度（A4）。

**核心方程**（Top-K 通道选择）：

```
e_g(c) = Σ_{i∈g} π_i · e_i(c)    // 池化通道能量
Ω_g = TopK_K(e_g)                 // 选 Top-K 保留
```

其中 e_i(c) 是上下文 i 中通道 c 的均方激活，π_i 编码部署暴露权重。

**Pooling Crossover 定理（Proposition 1）**：

```
N* = σ²_c / τ²_c                   // 临界样本数
N_eff = mN / (1 + (m-1)N·τ²_c/σ²_c) // 有效样本数
```

当校准样本数 N < N* 时，pooling 降低风险；当 N > N* 时，独立估计更优。

**变量说明**：

| 符号 | 含义 |
|------|------|
| z_i^(n)(c) | 上下文 i、轨迹 n 中通道 c 的均方激活 |
| e_i(c) | 通道 c 在上下文 i 中的期望能量 |
| e_g(c) | 组 g 中加权池化的通道能量 |
| τ²_c | 跨成员异质性方差 |
| σ²_c | 成员内采样方差 |
| N* | pooling 有益的临界样本数 |

**直觉**：当样本少且成员间差异不大时，pooling 通过增大有效样本量降低估计噪声；但当样本充足或成员差异大时，pooling 会引入偏差。

### 2.2 联合训练目标显著性（Co-Training-Objective Saliency）

**目标**：决定哪些 Linear 层升级到 W8。

**核心方程**（Kronecker 分解的 Empirical Fisher 失真）：

```
D_L(b) = 1/2 · tr[G_L · ε_L^(b) · Σ_L · (ε_L^(b))^T]
B_L = D_L(b_lo) - D_L(b_hi)     // 升级收益
```

**关键洞察**：联合 Fisher 与后融合 Fisher 的差异

```
diag(G_joint - G_fusion) = 2·λ_v·λ_a · E[g_v ⊙ g_a]
```

后融合方法丢失了视频和动作梯度的逐坐标对齐信息。

### 2.3 固定干预 Rollout 审计（Fixed-Intervention Rollout Auditing）

**目标**：修正去噪步保护调度。

**核心方程**（固定干预 replay 剖面）：

```
S_ref^replay(t) = E_{x_t ~ D_t^ref} [ ||f_a(x_t) - f_fp(x_t)||²₂ ]
T_replay = TopK_t(S_ref^replay(t))    // 用 replay 剖面替换保护集
```

**闭环误差传播的一阶近似**：

```
δs_{j+1} = A_j · δs_j + B_j · ε_j
```

下游任务影响取决于转移 Jacobian 的乘积，不仅是局部误差 ε_j。

## 3. 带数字走一遍：玩具例子 (Worked Example)

### 场景：2 层 WAM 的通道 pooling 决策

假设一个 WAM 有 2 个坐标可容的 Linear 层（视频 output projection + action output projection），每层 d=64 通道，用 N=32 条校准轨迹估计。

**步骤 1：估计方差成分**

```
通道 c=1:  σ²=0.01, τ²=0.002  →  N* = 5.0
通道 c=32: σ²=0.01, τ²=0.008  →  N* = 1.25
```

**步骤 2：pooling 决策**

```
N_cal = 32 条轨迹

通道 c=1: 32 > 5.0 → pooling 可能引入偏差，检查 bootstrap 稳定性
通道 c=32: 32 > 1.25 → 异质性大，不宜 pooling，用 per-context 估计
```

**步骤 3：Top-K 选择**

```
假设 ρ=0.02, d=64 → K = ⌊0.02×64⌋ = 1

池化能量 e_g = [0.85, 0.42, 0.38, ..., 0.05]
Top-1 通道 = c=1 (能量 0.85) → 保留 BF16
其余 63 通道 → 量化到 A4
```

**步骤 4：层粒度 W8 分配**

```
假设 30 个候选 Linear 层，预算 B = ⌊0.2×30⌋ = 6 层

各层升级收益 B_L（按 Empirical Fisher 排序）:
  Layer 7:  B=0.034  ← 视频早期层，高敏感
  Layer 15: B=0.028  ← action 中间层
  Layer 3:  B=0.025
  Layer 22: B=0.021
  Layer 11: B=0.018
  Layer 28: B=0.015  ← 第 6 名，刚好入选
  Layer 5:  B=0.012  ← 未入选，保持 W4
```

**结果**：6 层 W8 + 24 层 W4，激活 Top-2 通道 BF16 + 其余 A4 → 综合约 4.8 bit 权重 + 约 4 bit 激活。

## 4. 工程视角 (Engineering View)

| 指标 | 数值 | 说明 |
|------|------|------|
| 峰值显存 | FP16 的 29% | 仅针对 video+action DiT blocks，不含 VAE/embedding |
| Block 级加速 | 1.4-1.6× | 单次 model call 的 block 延迟比，非端到端控制环 |
| 校准数据量 | 32 条轨迹 | 用于所有 PTQ 拟合步骤（激活统计、mask、smoothing、saliency、GPTQ） |
| Profile 数据量 | 额外 32 条 FP16 rollout | 仅用于去噪步调度审计，与校准集 trajectory-disjoint |
| 硬件平台 | SM120 Blackwell (RTX PRO 5000) | W4 用 NVFP4 kernel，W8 用 FP8 kernel，BF16 用 bypass |
| 校准开销 | 需 backward pass | 需要原始 co-training targets，对闭源模型不适用 |

**工程含义**：

- **控制频率**：量化不改变控制频率，但 block 级 1.4-1.6× 加速意味着在相同控制周期内可以跑更多去噪步
- **模块边界**：量化决策在 DiT block 级别 granularity，不影响 VAE/encoder 等外围模块
- **部署约束**：需要 Blackwell 架构 GPU 的 NVFP4 硬件支持；旧架构（如 Ada/Hopper）可能无法复现相同加速比
- **校准数据隔离**：三组 32 条轨迹（校准/审计/验证）完全 disjoint，确保无数据泄露

## 5. 数据与评测 (Data & Eval)

### 数据集

| 数据集 | 规模 | 任务数 | 说明 |
|--------|------|--------|------|
| RoboTwin 2.0 | 2,500 clean + 25,000 random demonstrations | 50+ 双臂操作任务 | 双臂机器人仿真 |
| LIBERO | 4 suites × 10 tasks = 40 tasks | Spatial/Object/Goal/Long | 单臂桌面操作 |

### 评测协议

- **种子**：r ∈ {42, 43, 44}，每个种子独立采样校准/审计/验证集
- **评估规模**：15,000 episodes (RoboTwin 2.0) / 6,000 episodes (LIBERO Fast-WAM) / 1,500 episodes (LIBERO LingBot-VA)
- **指标**：闭环任务成功率（end-to-end protocol）
- **关键细节**：所有方法使用相同的初始状态种子，确保公平比较

### 数据分离协议

```
D_cal (32 trajectories)  → PTQ 拟合（激活统计、mask、smoothing、saliency、GPTQ）
D_prof (32 FP16 rollouts) → 去噪步剖面审计
D_val (独立 rollout)      → 调度 accept/reject
D_test (独立 rollout)     → 最终评测
```

四组数据 trajectory-disjoint（包括初始状态种子），但任务身份可重叠（benchmark-specific 校准，非跨任务迁移）。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

### 能做什么

| 能力 | 证据 |
|------|------|
| W4A4 下接近 FP16 精度 | RoboTwin 2.0: 91.8% vs 91.9% (Fast-WAM) |
| 跨架构通用 | Fast-WAM (dual-stream MoE) 和 LingBot-VA (shared-backbone) 均有效 |
| 真机部署 | AgiBot G2 上 3 个操作任务验证（论文提及但未给出具体数字） |
| 显存大幅降低 | 峰值显存降至 FP16 的 29% |
| 理论保证 | Proposition 1 给出 pooling 的样本复杂度界 |

### 不能做什么 / 局限

| 局限 | 原因 |
|------|------|
| 需要 co-training targets | 闭源 WAM 无法获取原始训练目标，无法做 backward pass |
| Benchmark-specific 校准 | 每个 benchmark 需独立采集 64+ 条轨迹，不跨任务迁移 |
| 依赖 Blackwell GPU | NVFP4 kernel 需要 SM120 架构，旧 GPU 无法复现 |
| 仅覆盖 DiT blocks | 显存/加速数字不含 VAE/embedding/projection，端到端收益更低 |
| 未测试跨域泛化 | 实验限于 RoboTwin 2.0 和 LIBERO，未评估真实世界 distribution shift |
| 非正式统计检验 | 论文明确声明 "These comparisons are not formal non-inferiority or equivalence tests" |

### 6.1 隐含假设 (Hidden Assumptions)

1. **坐标可容性可预先确定**：论文假设可以从架构设计中判断哪些模块共享坐标系统，但这在 shared-backbone 架构中可能不完全明确
2. **32 条轨迹足够**：所有校准步骤共用 32 条轨迹，对于高维 WAM 可能仍偏少（Proposition 1 的 N* 可能超过 32）
3. **FP16 rollout 代表部署分布**：fixed-intervention replay 使用 FP16 模型生成的 rollout 状态，但量化模型的部署分布可能不同（论文承认这一点，见 Δ_prof 诊断）
4. **Blackwell 硬件可用性**：实际部署场景中，端侧机器人可能没有 Blackwell GPU

## 7. 与相关工作对比 (Comparison)

| 方法 | 关注点 | 架构假设 | 校准分布 | 是否理论保证 |
|------|--------|----------|----------|-------------|
| GPTQ | 权重二阶补偿 | 同构 LLM | 独立文本 | 二阶误差界 |
| SmoothQuant | 激活 outlier 迁移 | 同构 LLM | 独立文本 | 平滑界 |
| SVDQuant | SVD 截断 | 扩散模型 | open-loop | 谱误差界 |
| Atom | 通道异常值保护 | 扩散模型 | open-loop | 通道界 |
| **QuantWAMs** | **校准上下文对齐** | **WAM 结构感知** | **闭环 rollout** | **Prop 1: pooling crossover** |

**面试 Tip**：当被问到"QuantWAMs 和传统 PTQ 的核心区别是什么？"——回答："传统 PTQ 假设模型同构、校准分布独立于部署、目标单一；QuantWAMs 认识到 WAM 的三个特性（多流耦合、闭环误差传播、联合训练目标）要求量化决策在结构、分布、目标三个轴上与部署上下文对齐，缺一即失效。"

### 消融实验（Table 3, LIBERO-Long）

| 组件组合 | Fast-WAM | LingBot-VA |
|----------|----------|------------|
| Base (无 QuantWAMs) | 80.8±0.7% | 80.2±0.9% |
| + Shared-Basis | 89.1±0.6% | 89.6±0.8% |
| + Joint (Synthetic) | 90.9±0.7% | 91.8±0.8% |
| + Replay (Full) | **95.0±0.4%** | **98.0±0.3%** |

三个组件依次贡献 +8.3%、+1.8%、+4.1%（Fast-WAM），Shared-Basis 是最大单一贡献者。

## 8. 精讀建議 (Reading Guide)

**值得精讀原文的人**：
- 正在做 WAM/扩散策略部署压缩的研究者，尤其是需要把模型塞进端侧 GPU 的场景
- 研究多流/共享 backbone 架构量化问题的工程师
- 对"校准上下文对齐"这一理论框架感兴趣的量化研究者

**建議章節路徑**：
1. 先讀 §3.1（Shared-Basis）→ 理解 coordinate admissibility 和 pooling crossover 定理
2. 再看 §3.2（Co-Training Saliency）→ 理解联合 Fisher 与后融合的差异
3. 然后 §3.3（Fixed-Intervention）→ 理解为什么 open-loop 校准对 WAM 不够
4. 可跳 §4.4 的細部 ablation（除非你做同类研究）

**不值得精讀的理由**：
- 如果你不做模型量化/部署优化，这篇论文的方法论对你没有直接价值
- 如果你只关注 open-loop 量化（如 LLM PTQ），这篇的核心贡献（闭环感知校准）不适用你的场景
- 如果你不用 WAM 架构（Fast-WAM / LingBot-VA），论文的方法需要大量适配

---

**关键引用**：
- [arXiv 论文](https://arxiv.org/abs/2607.28405)
- [Project Page](https://quantwams.github.io)
- [VLA-Handbook Theory Index](./README.md)

---
[← Back to Theory](./README.md)
