# OxyGen：面向多任务并行的 VLA 统一 KV Cache 管理 (Unified KV Cache Management for VLA Inference under Multi-Task Parallelism)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-05-20
>
> **论文**: OxyGen: Unified KV Cache Management for VLA Inference under Multi-Task Parallelism
> **链接**: https://arxiv.org/abs/2603.14371
> **代码**: https://github.com/air-embodied-brain/OxyGen
> **核心定位**: 将 KV Cache 从"每个任务独立持有"改造为"跨任务/跨帧统一共享资源"，解决 MoT VLA 多任务并行推理时的冗余计算与资源争用，在 Jetson AGX Thor 上实现最高 3.7x 加速。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 统一 KV Cache 管理通过跨任务共享 + 跨帧连续批处理，消除 MoT VLA 多任务并行推理中的冗余 prefill 与资源争用，最高 3.7x 加速 |
| 适合精读 | 如果你在部署 MoT VLA（如 pi0.5）到边缘设备（Jetson/4090），或多任务并发推理遇到延迟瓶颈，重点看 §3 方法 + §4.6 真机部署 |
| 可以跳过 | 如果你只关心单任务 action-only VLA 推理，或模型压缩/量化层面优化，这篇距离较远 |
| 落地可行性 | 高（基于 openpi 实现，开源代码，已在真机人形机器人验证） |
| 主要风险 | 仅验证了 pi0.5 一种 MoT backbone；跨帧调度引入的 overhead 在短解码场景下收益有限 |

💡 **X-Ray 开场**
现代具身智能机器人需要同时做多件事：一边操作物体，一边跟用户对话，一边构建环境记忆。MoT VLA（如 pi0.5）从架构上支持这种多模态输出，但现有推理系统让每个任务独立跑一遍前向传播——同样的视觉输入被编码多次，KV Cache 重复生成。OxyGen 发现根因是 KV Cache 被"孤立管理"，把它变成跨任务、跨帧的统一共享资源后，推理速度最高提升 3.7 倍，同时不损失动作质量。

📍 **研究全景时间线**

```
2023 RT-1/RT-2        → 2024 OpenVLA/CogAct     → 2025 pi0/MoT-VLA 架构涌现
       ↓                       ↓                          ↓
  Action-only           Action-only + 多模型        单模型多专家路由
  单任务推理             推理                    架构支持多任务但推理仍孤立
                                                    ↓
                                              2026 OxyGen ← 当前位置
                                              KV Cache 统一管理
                                              跨任务共享 + 跨帧批处理
                                              ↓
                                        局限：仅 pi0.5 验证
```

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 维度 | Baseline (Sequential Isolated) | Parallel (MPS) | OxyGen |
|------|-------------------------------|----------------|--------|
| KV Cache 管理 | 每个任务独立 prefill，各自持有 | 每个任务独立 prefill，各自持有 | 统一管理器 M，跨任务/跨帧共享 |
| Prefill 次数/帧 | Action 1 次 + Language 1 次 = 2 次 | 2 次（并行但冗余） | 仅 1 次（共享观察编码） |
| Language 解码 | 帧内顺序执行，阻塞 action | 帧内并行，GPU 资源争用 | 跨帧连续批处理，解耦 action 硬截止 |
| Action 频率 (RTX 4090, N=30) | 19.1 Hz | ~22 Hz | **60+ Hz** |
| Language 吞吐 (RTX 4090, N=30) | ~55 tokens/s | ~65 tokens/s | **200+ tokens/s** |
| 峰值显存 | 6.43 GB | 12.49 GB（翻倍） | 7.35 GB（+15%） |
| 能耗/请求 | 117.4 mJ | 120.9 mJ（+3%） | **25.8 mJ（-78%）** |
| 适用场景 | 单任务或低并发 | 资源充裕时的简单加速 | 边缘设备多任务并行部署 |

### 1.2 关键机制 (Key Mechanism)

OxyGen 的核心设计围绕一个抽象：**将 KV Cache 提升为一等公民的共享资源**，由统一管理器 M 负责跨任务分发和跨帧调度。

**优化 1 — 跨任务 KV 共享（Cross-Task KV Sharing）**
- 同一帧的观察 o_t 只 prefill 一次，生成共享 KV Cache K_t
- K_t 分发给 action expert（只读上下文，用于 S 步去噪）和 language expert（作为自回归解码的初始上下文）
- 消除冗余 prefill，短解码场景下带来 1.4x 加速

**优化 2 — 跨帧连续批处理（Cross-Frame Continuous Batching）**
- 语言解码不需要在单帧内完成，可以跨帧继续
- 管理器维护每个活跃请求的可恢复状态 σ_t = (K_t, y_t, δ_t)
- 每帧将 m 个活跃请求批处理为一个联合批次，统一推进 k 步解码
- Action 生成有硬帧截止（如 50 Hz 控制频率），语言解码有软截止（跨帧完成）
- 大 batch size 下硬件并行度充分利用，长解码场景下维持恒定 action 频率

⚡ **Eureka Moment**：KV Cache 不是每个任务的私有财产——它是共享观察的编码结果，天然适合被多个专家跨任务、跨帧复用。

### 1.3 信息流/架构图 (Flow / Diagram)

```
帧 t 的输入: o_t (视觉观察 + 语言指令)
                    │
                    ▼
          ┌─────────────────┐
          │  Prefill (一次)   │  ← 共享 VLM backbone Θ_VLM
          │  生成 K_t         │
          └────────┬────────┘
                   │ K_t (modality-agnostic KV cache)
          ┌────────┴────────┐
          ▼                 ▼
   ┌──────────────┐  ┌──────────────────┐
   │ Action Expert │  │ Language Expert  │
   │ S 步去噪      │  │ 自回归解码 k 步   │
   │ 生成 A_t      │  │ 追加到 σ_t       │
   └──────┬───────┘  └────────┬─────────┘
          │                   │
          ▼                   ▼
     Action A_t        存入管理器 M
     (硬截止完成)         σ_t → 活跃集 R
                              │
                    ┌─────────┴─────────┐
                    │ 跨帧批处理解码      │
                    │ m 个活跃请求联合    │
                    │ 每帧推进 k 步       │
                    └───────────────────┘
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：

```
T_frame = T_prefill + T_denoise + T_decode(B, k)
        ≈ T_shared + T_action + T_token(B) · k
```

核心洞察：跨帧批处理将单 token 解码成本除以 batch size B，使得 language 解码时间从 O(N·T_token) 降为 O((N/k)·T_token/B) 每帧——当 B 增大时，language 时间被摊薄到几乎不影响 action 硬截止。

**变量说明**：

| 符号 | 含义 | 典型值 |
|------|------|--------|
| τ | Language 吞吐（tokens/s） | 200+（OxyGen） |
| B | 平均活跃批大小 | N/k（总步数/每帧步数） |
| k | 每帧语言解码步数 | 1~10（可调） |
| T | 帧端到端推理延迟 | 由 prefill + denoise + decode 组成 |
| f | Action 频率（Hz） | ≥ 50（灵巧操作） |
| H | Action horizon（控制步数） | 10 |
| f_min | 最低控制频率 | 50 Hz |

**直觉**：降低 T（减少冗余计算）和提高 B·k（跨帧批处理放大并行度）是两条正交优化路径。OxyGen 同时走两条路——跨任务 KV 共享降低 T，跨帧批处理提高 B。

**核心方程**：

共享 KV Cache 的生成（一次 prefill）：

```
{(h_{t,l}, K_{t,l}, V_{t,l})}_{l=1}^{L} = Θ_VLM(o_t)
K_t = {(K_{t,l}, V_{t,l})}_{l=1}^{L}
```

两个专家共享 K_t 独立生成：

```
p_{Θ_Act}(A_t | K_t)     — action expert, S 步去噪
p_{Θ_Lang}(y_t | K_t)    — language expert, 自回归解码
```

可恢复状态（跨帧续传的关键）：

```
σ_t = (K_t, y_t, δ_t)
```

其中 δ_t ∈ {0,1} 是终止标志（EOS 或达到最大长度 N）。

> 符号与本文保持一致：o_t = 观察，K_t = KV Cache，A_t = action chunk，y_t = language tokens，S = 去噪步数，N = 最大解码步数，k = 每帧解码步数，H = action horizon。

## 3. 带数字走一遍：玩具例子 (Worked Example)

假设场景：RTX 4090，LIBERO 配置，每帧一个新观察和一个新语言请求。

**参数设定**：
- 总解码步数 N = 30
- 每帧解码步数 k = 5
- Action horizon H = 10，去噪步数 S = 10
- 单帧 prefill 时间 = 30 ms
- 单帧 action denoise 时间 = 150 ms
- 单 token language decode（单请求）= 15 ms

**Baseline（孤立执行）**：

每帧需要：prefill × 2（action + language 各一次）+ denoise + language decode(5 tokens)
```
T_baseline = 2×30 + 150 + 5×15 = 60 + 150 + 75 = 285 ms
f = H / T = 10 / 0.285 ≈ 35.1 Hz
```
当 N 增大（每帧需要 decode 更多 token），language 时间线性增长，f 持续下降。N=30 时实际测得 f ≈ 19.1 Hz。

**OxyGen（统一 KV + 跨帧批处理）**：

每帧需要：prefill × 1（共享）+ denoise + batched language decode(k=5, batch size B)
```
T_oxygen = 30 + 150 + (5×15)/B
```
当 B = 3（3 个活跃请求批处理）：
```
T_oxygen = 30 + 150 + 25 = 205 ms
f = 10 / 0.205 ≈ 48.8 Hz
τ = B×k / T = 3×5 / 0.205 ≈ 73 tokens/s
```
当 N 更大、B 更大时（如 B = 6）：
```
T_oxygen = 30 + 150 + 12.5 = 192.5 ms
f ≈ 52 Hz（几乎恒定）
τ = 6×5 / 0.1925 ≈ 156 tokens/s
```

**关键洞察**：Baseline 的 language decode 时间随 N 线性增长，拖慢 action；OxyGen 通过批处理将单 token 成本除以 B，language 时间被"摊薄"，action 频率几乎不受 N 影响。

## 4. 工程视角 (Engineering View)

### 4.1 部署架构

| 组件 | 技术栈 | 说明 |
|------|--------|------|
| 基础框架 | openpi（pi0.5 官方推理框架，GitHub 10k+ stars） | OxyGen 在其上构建调度层 |
| Backend | JAX（主要）+ PyTorch（2026-05 新增，含 torch.compile） | 双后端支持 |
| 硬件平台 | NVIDIA GeForce RTX 4090（24GB）/ Jetson AGX Thor | 边缘部署典型选择 |
| 操作系统 | Ubuntu 22.04/24.04 + CUDA 13.0 | |
| 包管理 | uv（Python 3.11） | |
| 许可证 | Apache 2.0（Gemma 组件有额外条款） | |

### 4.2 关键 Trade-off

**每帧解码步数 k 的选择**：
- k 小 → 平均 batch size B = N/k 大 → 批处理效率高 → language 吞吐高
- k 小 → 每帧留给 action 的时间预算多 → action 频率高
- 但 k 太小 → 请求跨更多帧完成 → 管理器状态维护 overhead 增加
- 论文建议：k ∈ {1, 5, 10}，根据目标帧率和硬件校准

**跨帧调度 overhead**：
- OxyGen 与"单帧解码上界"之间存在 gap（图 6 虚线），代表跨帧调度的额外开销
- 在 RTX 4090 上，这个 gap 约 5-10 Hz，相对可控
- 在 Jetson AGX Thor 上 gap 略大，但仍显著优于 baseline

**显存 vs 能效**：
- OxyGen 峰值显存 7.35 GB（vs baseline 6.43 GB），增加约 15%
- 但能耗/请求从 117.4 mJ 降至 25.8 mJ（-78%），因为批处理减少了 VLM 权重的内存访问次数
- 对边缘设备而言，能效提升远比显存小幅增加重要

### 4.4 真机部署数据（Jetson AGX Thor 人形机器人）

| 阶段 | Baseline | OxyGen |
|------|----------|--------|
| Prefill + Denoise | 207.5 ms | 198.0 ms |
| Language Generation | 822.3 ms | 195.4 ms（4.2x 加速） |
| Action 执行窗口 | 333 ms | 333 ms |
| 总推理时间 | 1030 ms | 393 ms |
| 是否超出窗口 | ❌ 远超 333 ms | ✅ Action 198 ms < 333 ms |

关键设计：OxyGen 将 language 生成（195 ms）安排在 action 产出之后，利用 action 执行窗口（333 ms）隐藏 language 延迟——language 生成与 action 执行重叠，不阻塞下一控制周期。

## 5. 数据与评测 (Data & Eval)

### 5.1 评测配置

论文未使用完整 rollout 行为，而是匹配 LIBERO/DROID/ALOHA 的观察规格（相机数量、分辨率、控制维度）进行推理速度评测：

| 配置 | 来源 | 特点 |
|------|------|------|
| LIBERO | Liu et al. 2023 | 标准 manipulation benchmark，3 相机 |
| DROID | Khazatsky et al. 2024 | 大规模真实机器人数据集，多相机 |
| ALOHA | Zhao et al. 2023 | 双臂操作，低延迟要求 |

### 5.2 动作质量验证

使用 pi0.5-LIBERO 官方 checkpoint 验证 OxyGen 不降低动作质量：

| 测试集 | openpi 报告 | OxyGen 实测 | 差异 |
|--------|------------|------------|------|
| LIBERO-Spatial | 98.8% | 98.0% | -0.8% |
| LIBERO-Long | 98.2% | 98.6% | +0.4% |
| LIBERO-Goal | 98.0% | 97.4% | -0.6% |
| LIBERO-10 | 92.4% | 93.2% | +0.8% |

差异在统计噪声范围内（±0.8%），确认加速不牺牲质量。

### 5.3 工作负载泛化

论文还测试了三种到达模式：
- **Uniform**：每帧固定数量请求 → 总 action/s 提升 4.4x
- **Poisson**：随机到达（强度 λ 变化）→ 维持更高 action 频率
- **Random-length**：请求长度随机变化 → 恒定 per-request action 频率

## 6. 能力与失败模式 (Capabilities & Failure Modes)

### 6.1 隐含假设 (Hidden Assumptions)

| # | 隐含假设 | 影响 | 验证状态 |
|---|---------|------|---------|
| 1 | Action 和 language 任务共享同一观察 o_t | 如果两个任务使用不同相机视角或不同时间的观察，跨任务 KV 共享不适用 | 论文假设成立但未显式讨论 |
| 2 | pi0.5 的 MoT 架构（VLM backbone + action expert + language expert） | 对其他 MoT backbone（如 WALL-OSS、Xiaomi-Robotics-0）的泛化性未验证 | §3.2 讨论但留作未来工作 |
| 3 | 单 GPU 边缘设备场景 | 多 GPU 部署下资源争用模式不同，优化效果待验证 | 实验仅覆盖单 GPU |
| 4 | 逐 token 自回归解码 | 状态 σ_t 理论上支持 speculative decoding，但未实际验证 | 作者声称兼容，未实验 |
| 5 | 每帧解码步数 k 离线校准 | 动态负载下固定 k 可能不是最优值 | 实验用 k∈{1,5,10} 覆盖但未自适应 |

### 6.2 能力

| 能力 | 证据 |
|------|------|
| 消除冗余 prefill | 跨任务 KV 共享，prefill 从 2 次降为 1 次（ablation 图 6） |
| 解耦 language/action 时间线 | 跨帧批处理，language 不阻塞 action 硬截止 |
| 边缘设备部署 | Jetson AGX Thor 真机验证，action 198 ms < 333 ms 窗口 |
| 能效优化 | 能耗/请求降低 78%（25.8 vs 117.4 mJ） |
| 工作负载泛化 | Uniform/Poisson/Random-length 三种到达模式均有效 |
| 与模型级优化正交 | 可与 token pruning、layer skipping、KV pruning 等叠加 |

### 6.3 失败模式

| 场景 | 原因 | 影响程度 |
|------|------|---------|
| 短解码（N < 10） | 跨帧批处理收益小，调度 overhead 占比高 | 低 — 仍有 1.4x 加速（KV 共享） |
| 单请求/低并发 | Batch size B = 1，无批处理收益 | 中 — 仅靠 KV 共享加速 |
| 每帧多观察 | Prefill 成本主导，per-request action 频率下降 | 中 — 但总 action/s 仍提升 |
| 仅关心 action-only | 无 language 任务时优化无意义 | 高 — 不适用 |

## 7. 与相关工作对比 (Comparison)

| 方法 | 关注点 | 架构/策略 | 适用场景 |
|------|--------|----------|---------|
| **OxyGen** | 多任务推理调度 | 统一 KV Cache 管理 + 跨帧批处理 | MoT VLA 边缘部署 |
| KV-Efficient VLA | 模型级 KV 优化 | 算子级选择性激活 KV cache | 单/多任务 VLA |
| Token pruning / Layer skipping | 模型级压缩 | 跳过 token/层减少计算 | 各种 VLA |
| vLLM / SGLang prefix caching | LLM 服务 | 前缀缓存共享 | 云规模 LLM 推理 |
| RTC / VLA-RAIL | Action 流水线 | 异步 action chunking，推理与执行重叠 | Action-only VLA |
| Naive MPS parallel | 简单并行 | CUDA MPS 多进程 | 资源充裕时 |

**关键区别**：vLLM/SGLang 面向云规模多租户 LLM 服务，目标是最大化总吞吐；OxyGen 面向边缘设备单 GPU 多任务 MoT VLA，目标是在满足 action 硬截止的前提下最大化 language 吞吐。前者直接应用于 VLA 会失败——没有跨模态协调设计，language 解码仍会阻塞 action。

> 💡 **面试 Tip**：如果被问到"OxyGen 和 vLLM 的 prefix caching 有什么区别"，核心回答是：vLLM 假设同构自回归读者（所有请求以相同方式扩展 cache），而 MoT VLA 的 action expert 把 prefill cache 当只读上下文、language expert 逐 token 追加 KV—— naive sharing 会导致 language 的 append 混入 action 的只读视图。OxyGen 的统一管理器在架构层面解决了这个异构语义问题。

## 8. 精讀建議 (Reading Guide)

**值得精讀原文的人**：
- 部署 MoT VLA（pi0.5/WALL-OSS 等）到边缘设备（Jetson/4090）的工程师——§3.2 的统一 KV 管理器设计和 §4.6 真机部署数据直接可用
- 研究多模态模型推理优化的研究者——§3.1 的问题形式化（非对称截止下的联合优化）和 §A.2 的批处理状态定义有方法论价值
- 评估多任务具身 Agent 系统架构的研究者——§1 和 §2 的 motivation 和 related work 提供了清晰的分类框架

**建議章節路徑**：
先读 §1 Introduction（理解 isolated execution 的两个问题）→ 再看 §3 Method（统一 KV 管理器 + 两个优化）→ §4.2 端到端结果 + §4.3 ablation（量化收益）→ §4.6 真机部署（落地参考）→ 可跳 §A（形式化细节，除非你要复现）

**不值得精讀的理由**：
- 如果你不做 MoT VLA 推理优化（只关心单任务 action-only），这篇的优化方向不匹配
- 如果你关注的是模型压缩/量化层面（而非推理调度层），相关工作中的 KV-Efficient VLA 或 token pruning 论文更相关
- 如果你只在云端多 GPU 环境部署，边缘单 GPU 的优化假设不适用


---
[← Back to Theory](./README.md)

**关键引用**：
- 论文: https://arxiv.org/abs/2603.14371
- 代码: https://github.com/air-embodied-brain/OxyGen
- pi0.5 原始论文: https://www.physicalintelligence.company/download/pi0_5.pdf
- openpi 框架: https://github.com/Physical-Intelligence/openpi
