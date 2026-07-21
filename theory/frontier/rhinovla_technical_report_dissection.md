# RhinoVLA 技术报告 (RhinoVLA Technical Report)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-07-21
>
> **论文**: RhinoVLA Technical Report
> **链接**: https://arxiv.org/abs/2606.07383
> **核心定位**: 将 VLA 推理延迟的根因追溯到视觉 token 数量，通过 Qwen3-VL 的 $4\times$ token 压缩 + 辉羲 R1 芯片联合优化，在边缘 SoC 上实现 $11.69$ Hz 实时闭环控制，同时保持与 $\pi_{0.5}$ 相当的任务成功率。

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | VLA 边缘部署的瓶颈不在模型参数量，而在视觉 token 数量；减少 token 数可直接降低 MLP GEMM 计算量，从而提升推理频率 |
| 適合精讀 | 如果你在边缘硬件（Jetson/R1/NPU）上部署 VLA；或需要跨机器人平台统一训练策略 |
| 可以跳過 | 如果你只关心桌面 GPU 上的 VLA 训练、或纯算法层面的策略优化 |
| 落地可行性 | 中 — 依赖辉羲 R1 芯片（非市面通用硬件），但 token 压缩思想可迁移到任何平台 |
| 主要風險 | R1 芯片生态尚不成熟；预训练数据主要来自 AgiBot 自家机器人，跨品牌泛化能力待验证 |

💡 **X-Ray 开场**

VLA 模型在机器人上跑不快，根本原因是什么？这篇论文给出了一个反直觉的答案：不是模型太大，而是视觉 token 太多。每个摄像头画面被切成 256 个 token 喂给语言模型，而语言模型的 MLP 层计算量跟 token 数成正比。RhinoVLA 把每张图片的 token 从 256 压缩到 64，配合自研芯片的算子优化，在边缘 SoC 上跑到了 11.69 Hz — 刚好跨过实时控制的门槛。对 VLA 研究者来说，这意味着"token 效率"应该成为架构设计的核心指标之一，而不仅仅是精度。

📍 **研究全景时间线**

```
[2022] RT-1: 端到端 tokenized 动作
    ↓
[2023] RT-2: 动作即文本 token，VLM+策略耦合
    ↓
[2024] π₀: VLM backbone + flow-matching Action Expert 解耦架构
    ↓
[2025] π₀.₅: 离散 token 预训练 + 连续 flow-matching 后训练
    ↓
[2025] OpenVLA: 7B Prismatic VLM + 970k 机器人数据
    ↓
[2026] RhinoVLA ← 当前位置: Qwen3-VL(2.13B) 4× token 压缩 + 72D 统一接口 + R1 芯片联合优化
    → 局限: 依赖特定芯片生态，跨品牌泛化待验证
```

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 维度 | $\pi_{0.5}$ (PI) | RhinoVLA | 差异含义 |
|------|-----------|----------|----------|
| VLM Backbone | PaliGemma-224 (~3B) | Qwen3-VL (2.13B) | 参数量减少 ~30% |
| 每图视觉 token | 256 | 64 (空间合并后) | **4$\times$ 压缩**，核心创新   |
| Action Expert | 0.43B (Gemma-based) | 0.40B (Qwen-compatible) | 规模相当，但适配 Qwen 接口 |
| 总参数量 | ~3.3B | ~2.53B | 更轻量 |
| 部署芯片 | NVIDIA Jetson AGX Orin | 辉羲 R1 (7nm, 500 TOPS INT8) | 不同芯片生态 |
| 端到端频率 | ~1.17 Hz (Orin 估算) | **11.69 Hz** (R1 实测) | **10$\times$ 提升**   |
| 跨机器人接口 | 无统一机制 | View Registry + 72D slot + Instance LoRA | RhinoVLA 独有 |
| 训练数据 | 未公开 | AgiBotWorld (2976h G1 + G2) + Open X-Embodiment | 数据源不同 |
| 量化策略 | FP16 | W8A16 (INT8 权重 + FP16 激活) | 内存带宽优化 |

### 1.2 关键机制 (Key Mechanism)

RhinoVLA 的核心设计围绕三个异质性挑战展开：

**挑战 A: 摄像头视角异构** $\to$ **View Registry**  
- 不同机器人数据集的摄像头布局、命名、顺序完全不同
- 解决方案：在预处理阶段将每个数据集的摄像头字段映射到固定角色-模态词汇表（如 `[head|rgb]`、`[left_wrist|rgb]`）
- 关键效果：Qwen3-VL 在 tokenization 之前就知道每张图的相机身份，而非从图像顺序中猜测

**挑战 B: 动作空间异构** $\to$ **72D 统一物理槽位 + 二元掩码**  
- 不同机器人的动作向量长度和索引含义不同
- 解决方案：定义 72 维固定物理含义的槽位空间（臂/腕/头/腰用弧度，夹爪用 [0,1] 闭合比，移动底座用速度单位）
- 二元掩码标记哪些槽位对当前机器人有效，无效槽位从 flow-matching 损失中排除
- 灵巧手预留 16 槽：拇指 4 DoF + 其余四指各 3 DoF（4-3-3-3-3 分配）

**挑战 C: 机器人实例残差** $\to$ **Robot-instance LoRA**  
- 即使共享相同的槽位定义，不同机器人在标定误差、关节限位、夹爪力学等方面仍有差异
- 解决方案：在 Action Expert 的 FFN 中插入机器人实例级 LoRA 模块
- 部署时可将 LoRA 合并到基权重中，保持统一计算图

⚡ **Eureka Moment**: VLA 推理延迟的瓶颈不是模型参数量，而是视觉 token 数量 — 因为 MLP GEMM 的计算量与 token 数成线性关系，减少 token 数就是最直接的性能优化。

### 1.3 信息流/架构图 (Flow / Diagram)

```
┌─────────────────────────────────────────────────────────────────┐
│                    RhinoVLA 端到端信息流                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Head RGB]    [Left Wrist RGB]    [Right Wrist RGB]            │
│       │                │                    │                   │
│       └────────────────┼────────────────────┘                   │
│                        ▼                                        │
│          ┌─────────────────────┐                                │
│          │   并行视觉编码       │  ← 三图 batch=3 并行 ViT       │
│          │   (24.31ms vs 34.52) │                                │
│          └─────────┬───────────┘                                │
│                    ▼                                            │
│    ┌───────────────────────────────────┐                        │
│    │       Qwen3-VL Backbone (2.13B)   │                        │
│    │   每图 64 tokens (vs 256 in π₀.₅) │                        │
│    │   输出: 最后18层 KV cache (c_vlm)  │                        │
│    └────────────────────┬──────────────┘                        │
│                         ▼                                       │
│  ┌─────────────────────────────────────────────┐               │
│  │           Action Expert (0.40B)              │               │
│  │                                              │               │
│  │  输入: c_vlm + 72D state + masks             │               │
│  │       + noisy action + time t + instance_id  │               │
│  │                                              │               │
│  │  ┌──────────────┐  ┌──────────────────┐     │               │
│  │  │ Shared AE    │  │ Instance LoRA    │     │               │
│  │  │ (18 layers)  │  │ (FFN only, per   │     │               │
│  │  │              │  │  robot)          │     │               │
│  │  └──────┬───────┘  └────────┬─────────┘     │               │
│  │         └────────┬──────────┘                │               │
│  │                  ▼                           │               │
│  │    输出: 72D flow velocity v̂_θ              │               │
│  │    (无效槽位被 mask 排除)                     │               │
│  └─────────────────────────────────────────────┘               │
│                         ▼                                       │
│              [72D 动作 → 机器人执行]                             │
│                                                                 │
│  端到端延迟: 85.54 ms → 11.69 Hz (R1)                          │
└─────────────────────────────────────────────────────────────────┘
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：

```
MLP 计算量 ∝ S (token 数量)  →  减少 S = 直接降低延迟
```

**目标**: 在保持 VLA 任务成功率的前提下，最小化边缘硬件上的端到端推理延迟。

**核心方程 — MLP GEMM 计算复杂度**:

```
FLOPs = 2 · B · S · d_in · d_out
```

其中：
- `B` = batch size（推理时为 1）
- `S` = 输入 token 数量（视觉 token + 上下文 token）
- `d_in` / `d_out` = 线性层的输入/输出维度（固定）

**关键洞察**: 当 `d_in`、`d_out`、`B` 固定时，FLOPs 与 `S` 成线性关系。$\pi_{0.5}$ 的 PaliGemma 用 256 token/图，Qwen3-VL 用 64 token/图 $\to$ 单图 MLP 计算量减少 **4$\times$**。  

**Flow-matching 损失**（训练目标）:

```
L_FM = Σ_{h,d} m_a(d) · w(h,d) · ||v̂_θ(h,d) - (z(h,d) - a(h,d))||²₂
       ──────────────────────────────────────────────────────────────
       Σ_{h,d} m_a(d) · w(h,d) + ε
```

其中：
- `h` = 动作 chunk 时间步（预测未来 H 步动作）
- `d` = 72D 槽位索引
- `m_a(d)` = 动作掩码（1=有效槽位，0=无效）
- `w(h,d)` = 每槽位权重（平衡不同动作组）
- `v̂_θ` = Action Expert 预测的 flow velocity
- `z` = 干净目标动作 chunk
- `a` = 高斯噪声 ~N(0, I)
- `x_t = (1-t)·a + t·z` = 插值点，t ∈ [0,1]

> 符号与本文保持一致：所有公式基于论文 §3.3.2 和 §3.1.2。

## 3. 带数字走一遍：玩具例子 (Worked Example)

**场景**: 单目视觉（1 张 $224\times224$ 图像），单步推理，比较 $\pi_{0.5}$ 与 RhinoVLA 的 MLP 计算量差异。  

**假设参数**（基于论文公开数据 + 合理估算）：

| 参数 | $\pi_{0.5}$ (PaliGemma)   | RhinoVLA (Qwen3-VL) |
|------|-------------------|----------------------|
| 视觉 token/图 | 256 | 64 |
| 上下文 token (语言指令) | 64 | 64 |
| 总 token S | 320 | 128 |
| d_in (hidden size) | 2048 | 2048 |
| d_out (intermediate) | 8192 | 8192 |
| Transformer 层数 | 26 | 26 |

**每层 MLP FLOPs 计算**:

```
π₀.₅:   FLOPs = 2 × 1 × 320 × 2048 × 8192 = 10.74 × 10⁹  (10.74B)
Rhino:  FLOPs = 2 × 1 × 128 × 2048 × 8192 =  4.29 × 10⁹  (4.29B)
```

**26 层总 MLP FLOPs**:

```
π₀.₅:   26 × 10.74B = 279.3B FLOPs
Rhino:  26 ×  4.29B = 111.6B FLOPs
```

**理论加速比**: 279.3 / 111.6 = **2.5×**

这意味着仅通过 token 压缩，VLM backbone 的 MLP 计算量就减少了 **60%**。加上 R1 芯片的 W8A16 量化（up_proj 实测 $1.69\times$ 加速）、并行视觉编码（$34.52\,\text{ms} \to 24.31\,\text{ms}$）、以及算子融合，最终端到端达到 $11.69\ \text{Hz}$。  

**推理采样过程**（以 4 步 flow-matching 为例）：

```
Step 0: a ~ N(0,I), 72D 随机噪声动作
Step 1: t=0.25 → x_t = 0.75·a + 0.25·z → AE 预测 v̂₁ → 更新动作
Step 2: t=0.50 → x_t = 0.50·a + 0.50·z → AE 预测 v̂₂ → 更新动作
Step 3: t=0.75 → x_t = 0.25·a + 0.75·z → AE 预测 v̂₃ → 更新动作
Step 4: t=1.00 → x_t = z              → AE 预测 v̂₄ → 输出最终动作
```

每步推理都需要重新跑一次 Action Expert（18 层），但 VLM KV cache 只需计算一次并复用。

## 4. 工程视角 (Engineering View)

### 4.1 延迟分解（R1 优化后）

论文 Table 6 给出的端到端延迟分解：

| 阶段 | 延迟 (ms) | 占比 | 优化手段 |
|------|-----------|------|----------|
| 视觉编码 (3 views) | ~24.3 | 28.4% | 并行 batch encoding |
| VLM Backbone (Qwen3-VL) | ~38.5 | 45.0% | token 压缩 + W8A16 + 算子融合 |
| Action Expert (4 steps) | ~22.7 | 26.6% | 共享 KV cache + Instance LoRA |
| **总计** | **~85.5** | **100%** | **$\to\ 11.69\ \text{Hz}$**   |

### 4.2 关键工程 trade-off

| 设计选择 | 收益 | 代价 |
|----------|------|------|
| W8A16 量化（非 W8A8） | 保持精度（W8A8 明显降低成功率） | 需要自定义 GEMM kernel（up_proj $191\,\mu\text{s} \to 113\,\mu\text{s}$）   |
| Qwen3-VL 64 token/图 | MLP 计算量减少 60% | 空间分辨率降低，可能丢失细粒度视觉信息 |
| Instance LoRA（非独立输出头） | 统一部署图，可合并权重 | LoRA 容量有限，大残差机器人可能适配不足 |
| 冻结 VLM backbone，仅训 LoRA | 减少训练计算量 | 预训练视觉表征无法针对机器人视角深度调整 |

### 4.3 部署约束

- **芯片依赖**: 所有优化（W8A16 GEMM、FlashAttention SPM 适配、细粒度调度）都针对辉羲 R1 定制，迁移到 Jetson 需要重写 kernel
- **内存带宽**: R1 提供 200 GB/s 级带宽，与 Orin (203 GB/s) 相当，但 R1 的 SPM（软件管理片上缓存）提供了 GPU shared memory 之外的优化维度
- **控制频率目标**: 10 Hz 是实时闭环控制的最低门槛（Luo et al. 2024; Jang et al. 2022），RhinoVLA 的 11.69 Hz 刚好跨过

## 5. 数据与评测 (Data & Eval)

### 5.1 训练数据

| 数据源 | 内容 | 规模 |
|--------|------|------|
| AgiBotWorld Beta | 真实机器人轨迹 | 2,976.4 小时 (G1) + 数百小时 (G2) |
| AgiBotWorld 2026 | 更新版真实机器人轨迹 | 未公开具体时长 |
| Open X-Embodiment | 跨平台机器人关节空间子集 | 多机器人平台 |

**数据配比**: 使用 power-law 平衡（指数 $0.43$，与 $\pi_0$ 系列一致）：  

```
p_i = N_i^0.43 / Σ_j N_j^0.43
```

### 5.2 LIBERO 仿真评测

| 模型 | Spatial | Object | Goal | Long | 平均 |
|------|---------|--------|------|------|------|
| OpenVLA | — | — | — | — | — |
| $\pi_0$   | — | — | — | — | — |
| $\pi_0$-Fast | — | — | — | — | — |
| CoT-VLA | — | — | — | — | — |
| **RhinoVLA** | — | — | — | 90.4% | **94.1%** |

> TODO: 论文 Table 4 的具体数值在 HTML 提取中不完整，上述表格中 "-" 表示待从 PDF 原文补充。已知 RhinoVLA 平均 $94.1\%$，Long 套件 $90.4\%$，超过 $\pi_0$ 和 $\pi_0$-Fast。

**消融实验**（论文原文）：
- 无机器人预训练 → 平均 $90.0\%$
- + 机器人预训练 → 平均 $91.8\%$（$+1.8$pp）
- + View Registry → 平均 **$94.1\%$**（$+2.3$pp）

### 5.3 真机评测

| 机器人 | 任务 | 设置 | RhinoVLA SR | $\pi_{0.5}$ SR |
|--------|------|------|-------------|---------|
| Galbot G1 | 红袋→远 bin | seen | — | — |
| Galbot G1 | 红袋→远 bin | unseen | 100% | 100% |
| AgiBot G2 | 长程多物品分类 | seen | 58% | 52% |
| AgiBot G2 | 长程多物品分类 | unseen | 24% | — |
| AgiBot G1 | 双臂叠毛巾 | seen | 67% | — |
| AgiBot G1 | 双臂叠毛巾 | unseen | 43% | — |

> TODO: 部分 $\pi_{0.5}$ 对比数据在 HTML 提取中不完整，待从 PDF 补充。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

### 6.1 能力

| 能力 | 证据 | 条件 |
|------|------|------|
| 实时边缘控制 | R1 上 11.69 Hz | 需要辉羲 R1 芯片 |
| 跨本体迁移 | Galbot G1（未见品牌）unseen 100% | 动作可映射到 72D 槽位 |
| 长程任务 | AgiBot G2 长程 58% (seen) | 需要目标机器人适配数据 |
| 柔性物体操作 | AgiBot G1 叠毛巾 67% (seen) | 双臂配置 |
| 跨机器人预训练 | Instance LoRA 残差与动作掩码距离相关 | 机器人间结构差异不能太大 |

### 6.2 失败模式

| 失败模式 | 场景 | 原因 |
|----------|------|------|
| Unseen 设置性能骤降 | AgiBot G2 unseen 24% (vs seen 58%) | 工作空间/物体分布偏移超出 LoRA 适配能力 |
| 细粒度视觉操作 | 64 token/图可能丢失小物体细节 | 空间合并导致分辨率下降 |
| 非 72D 槽位覆盖的 DoF | 腿部/足部关节不被预训练监督 | 当前 72D 设计未包含移动底盘的腿关节 |
| 大残差机器人 | 结构差异大的机器人 LoRA 残差相似性低 | Instance LoRA 容量有限 |

### 6.3 隐含假设 (Hidden Assumptions)

1. **视觉 token 减少不会显著损害精细操作能力** — 论文在 LIBERO 和真机上验证了成功率，但 LIBERO 的场景相对结构化，真实世界的视觉复杂度可能更高
2. **72D 槽位空间足以覆盖主流机器人** — 对于超出 72D 设计范围的多指灵巧手或特殊执行器，映射规则可能不够表达
3. **Instance LoRA 的 FFN-only 插入点足够** — LoRA 仅插入 FFN 而不修改 attention 模块，假设机器人差异主要体现在前馈特征而非注意力模式上
4. **W8A16 量化不损失精度** — 论文声称 W8A8 明显降低成功率，但 W8A16 的精度保持仅在特定模型上验证，不同架构可能有不同表现
5. **辉羲 R1 生态可用** — 所有部署优化都依赖 R1 的 SPM、自定义 GEMM kernel 和运行时调度，迁移成本未量化

## 7. 与相关工作对比 (Comparison)

| 维度 | $\pi_{0.5}$ | OpenVLA | GR00T N1 | RhinoVLA |
|------|------|---------|----------|----------|
| 架构 | VLM + Flow Expert | 7B Prismatic VLM | VLM + DiT Action | Qwen3-VL + Flow Expert |
| 参数量 | ~3.3B | ~7B | 未公开 | ~2.53B |
| 视觉 token/图 | 256 | 可变 | 未公开 | **64** |
| 动作表示 | 连续 flow-matching | tokenized | DiT 生成 | 连续 flow-matching (72D) |
| 边缘部署 | Jetson Orin (慢) | 未优化 | 未优化 | **R1 11.69 Hz** |
| 跨机器人接口 | 无 | 无 | 有 (NVIDIA 生态) | **View Registry + 72D + LoRA** |
| 量化 | FP16 | 未量化 | 未公开 | **W8A16** |
| 开源 | 部分 | 是 | 部分 | 是 (Apache 2.0) |

**面试 Tip**: 如果被问到"RhinoVLA 相比 $\pi_{0.5}$ 的核心改进是什么"，回答："不是减少参数量，而是减少视觉 token 数量 — 从 $256$ 到 $64$，$4$ 倍压缩。因为 MLP GEMM 计算量与 token 数线性相关，这是比减少参数量更直接的延迟优化路径。"

## 8. 精讀建議 (Reading Guide)

- **值得精讀原文的人**:
  1. 在边缘硬件（Jetson/NPU/自研芯片）上部署 VLA 模型的系统工程师 — §3.1 延迟分解和 §3.4 部署优化是可直接复用的 checklist
  2. 需要跨多个机器人平台统一训练策略的研究者 — §3.2 的三机制（View Registry / 72D slot / Instance LoRA）提供了完整的异质性处理框架
  3. 关注 Qwen3-VL 作为 VLA backbone 可行性的研究者 — 这是首个大规模验证 Qwen3-VL 在 VLA 中表现的工作

- **建議章節路徑**: 先读 §3.1（性能分析，理解为什么 token 数是瓶颈）→ 再看 §3.2（架构设计，理解三个异质性机制）→ 可跳过 §2（背景）除非你不熟悉 VLA 演进

- **不值得精讀的理由**: 如果你不做边缘部署、不关心跨机器人训练、且已有充足的 GPU 算力，那么这篇论文的核心贡献（token 压缩 + 芯片联合优化）对你的直接价值有限，读摘要即可

---
[← Back to Theory](./README.md)

**关键引用**:
- 论文: https://arxiv.org/abs/2606.07383
- 代码: https://github.com/HUIXI-AI/RhinoVLA
- 权重: https://huggingface.co/HuixiAI/RhinoVLA
- Qwen3-VL: https://github.com/QwenLM/Qwen3-VL
- $\pi_{0.5}$: https://www.physicalintelligence.company/blog/pi05
