# MemoryWAM：高效世界动作建模与持久记忆 (MemoryWAM: Efficient World Action Modeling with Persistent Memory)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-06-22
>
> **论文**: MemoryWAM: Efficient World Action Modeling with Persistent Memory
> **链接**: https://arxiv.org/abs/2606.20562
> **核心定位**: 解决 WAM 中"记忆-效率"的根本矛盾——用混合记忆机制将推理复杂度从 O(N) 降至 O(N/d)，同时超越全历史 KV 缓存基线 LingBot-VA 4.8 个百分点

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | 混合记忆（滑动窗口 + 锚帧 + gist token）可在保持 87% 成功率（与全注意力持平）的同时，将推理复杂度从 O(N) 降至 O(N/15) |
| 適合精讀 | 如果你在构建长视界机器人策略、需要解决 KV 缓存膨胀问题、或研究世界模型的记忆机制 |
| 可以跳过 | 如果你只做短视界 VLA（如标准 OpenVLA 部署），这篇距离中等 |
| 落地可行性 | 中——依赖 Wan2.2-TI2V-5B 预训练权重，需 8 GPU 训练，但推理阶段无需视频生成 |
| 主要風險 | 仅用 50  demonstrations/task 训练；gist token 压缩比固定为 15×，未探索自适应压缩 |

💡 **X-Ray 开场**
WAM（世界动作模型）面临一个根本矛盾：用滑动窗口做短期记忆效率高但会遗忘长期信息，用全历史 KV 缓存保留完整上下文但推理成本随轨迹长度线性增长。MemoryWAM 受人类认知心理学启发，设计了三层混合记忆——短期滑动窗口、事件边界锚帧、压缩的 gist token——把推理复杂度从 O(N) 降到 O(N/d)（d=15），同时在 RMBench 9 个长期记忆任务上达到 83.0% 平均成功率，比全历史基线 LingBot-VA 还高 4.8%。对 VLA 研究者意味着：世界模型可以在不牺牲推理速度的前提下拥有"长期记忆"。

📍 **研究全景时间线**
```
[2024] VLA 直接映射观察→动作（OpenVLA, RT-2）
  → [2025] WAM 统一视频-动作建模（Cosmos Policy, FastWAM）— 滑动窗口高效但遗忘
  → [2025] 全历史 KV 缓存 WAM（LingBot-VA, DreamZero）— 记忆完整但推理慢
  → [2026-06] MemoryWAM — 混合记忆，兼顾效率与记忆 ← 当前位置
  ← 局限：仅 50 demo/task 训练；gist token 压缩比固定
```

## 1. 核心架构/方法总览 (Overview / Architecture)

MemoryWAM 基于 MoT（Mixture-of-Transformers）架构，包含两个独立的 DiT（Diffusion Transformer）分支：视频 DiT 负责动力学建模与记忆维护，动作 DiT 负责条件化动作生成。训练阶段通过视频预测提供密集监督信号，推理阶段跳过视频生成，仅用动作 DiT 输出动作块。

### 1.1 系统对比概览 (System Component Comparison)

| 维度 | π0.5 / FastWAM（滑动窗口） | LingBot-VA（全历史 KV） | MemoryWAM（混合记忆） |
|------|------|------|------|
| **记忆范围** | 最近 N 帧（有界） | 全部历史帧 | 最近帧 + 锚帧 + gist token |
| **推理复杂度** | O(1)（常数窗口） | O(N) 线性增长 | O(N/d)，d=15 |
| **GPU 内存** | 常数 | 线性增长 | O(N/d) |
| **RMBench 平均成功率** | 10.4% / 5.9% | 78.2% | **83.0%** |
| **长期记忆能力** | ❌ 弱 | ✅ 强 | ✅ 强 |
| **推理延迟** | 低 | 高（1600帧时显著增长） | 低（1600帧时仍低于 RNN/TTT） |
| **模型大小** | ~5B（仅视频DiT） | ~6B | ~6B（视频DiT 5B + 动作DiT 1B） |
| **推理时视频生成** | ❌ 不需要 | ❌ 不需要 | ❌ 不需要 |

### 1.2 关键机制 (Key Mechanism)

MemoryWAM 的混合记忆由三个组件构成：

- **短期记忆（Short-term Memory）**: 最近 N_recent = 4 帧的完整视频 latent KV cache，用于即时闭环控制，捕捉物体运动、接触状态等快速变化信号
- **事件边界记忆（Event-boundary Memory）**: 任务初始 N_init = 2 帧的完整视觉 token，保留场景初始状态（如物体位置、任务指令相关的关键信息），这些信息后续可能被遮挡或移出视野
- **Gist 记忆（Gist Memory）**: 每帧 M = 8 个可学习的 gist token（远少于 L = 120 个视觉 token），压缩长期历史。对于既非锚帧也非近期帧的历史帧，后续 token 不直接 attends 该帧，而是 attends 其 gist token

⚡ **Eureka Moment**: 人类记忆不是单一存储——短期记忆容量有限但保真度高，长期记忆保存抽象 gist 而非逐字细节，事件边界特别突出。MemoryWAM 把这三条认知心理学原理映射到 WAM 的 KV cache 管理中，用 8 个 gist token 替代 120 个视觉 token，实现 15× 压缩而不丢失关键决策信息。

### 1.3 信息流/架构图 (Flow / Diagram)

```
训练阶段:
  观察 o_t → [视频VAE] → z_t (video latent)
    → [视频DiT Φ_v] → 视频预测（监督信号）+ KV cache 更新
    → [动作DiT Φ_a] → 动作块 a_{t:t+h-1}（条件化于视频KV cache）
  损失: L = λ_v·L_video + λ_a·L_action

推理阶段（无视频生成）:
  观察 o_t → [视频VAE] → z_t
    → [视频DiT Φ_v] → 更新 C_t^v（单次前向！）
    → [动作DiT Φ_a] → a_{t:t+h-1}（denoising，attend to C_≤t^v）

混合记忆缓存结构:
  C_≤t^v = C_short^v ∪ C_anchor^v ∪ C_gist^v
           │              │              │
           ├─ 最近4帧      ├─ 初始2帧     ├─ 每帧8个gist token
           ├─ 完整KV cache  ├─ 完整KV cache ├─ 压缩表示（15×）
           └─ 闭环控制      └─ 任务初始    └─ 长期历史
                │              │              │
                └──────────────┴──────────────┘
                        动作DiT 统一 attend
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：
```
C_≤t^v = C_short^v ∪ C_anchor^v ∪ C_gist^v
|C_gist^v| = O(N·M) = O(N·L/d)  其中 d = L/M = 120/8 = 15
```

**目标**: 在保持策略性能（成功率）不变的前提下，最小化推理时的 KV cache 大小和注意力计算量。

**核心方程**:

1. 视频 DiT 更新（推理时单次前向）：
```
C_t^v = Φ_v(z_t, l; C_{<t})
```

2. 动作 DiT 生成（条件化于混合缓存）：
```
a_{t:t+h-1} = Φ_a(x_τ^a, l; C_short^v ∪ C_anchor^v ∪ C_gist^v)
```

3. 总损失（训练时）：
```
L = λ_v · L_video + λ_a · L_action
  = λ_v · MSE(ŷ_video, y_video) + λ_a · MSE(ŷ_action, y_action)
```

**变量说明**:

| 符号 | 含义 |
|------|------|
| z_t | t 时刻视频观测的 VAE latent（48 通道） |
| l | 任务指令（T5 text encoder 编码） |
| Φ_v | 视频 DiT（30 blocks, hidden 3072, 24 heads） |
| Φ_a | 动作 DiT（30 blocks, hidden 1024, 24 heads） |
| C_t^v | t 时刻视频侧 KV cache |
| x_τ^a | τ 扩散时刻的噪声动作 token |
| h | 动作视界 = 16 步 |
| L | 每帧视觉 token 数 = 120 |
| M | 每帧 gist token 数 = 8 |
| d | 压缩比 = L/M = 15 |

> 符号与本文/相关文档保持一致。

## 3. 带数字走一遍：玩具例子 (Worked Example)

假设一个 100 步的长期操作任务，每 16 步输出一个动作块（即 N = 100/16 ≈ 6 个 latent 帧）。

**全历史 KV 缓存方式**（如 LingBot-VA）:
- 每帧 120 个 token，6 帧全部保留
- KV cache token 数 = 6 × 120 = 720
- 注意力复杂度 = O(720²) ≈ O(518,400)

**MemoryWAM 混合记忆方式**:
- 假设第 1-2 帧为锚帧（完整保留）：2 × 120 = 240 token
- 第 5-6 帧为近期帧（完整保留）：2 × 120 = 240 token
- 第 3-4 帧被压缩为 gist token：2 × 8 = 16 token
- KV cache token 数 = 240 + 240 + 16 = 496
- 注意力复杂度 = O(496²) ≈ O(246,016)
- **节省**: (518,400 - 246,016) / 518,400 ≈ 52.5% 的注意力计算量

扩展到 1600 帧（论文中的极端情况）:
- 全历史: 1600 × 120 = 192,000 token → O(192,000²)
- MemoryWAM: 假设 2 锚帧 + 4 近期帧 + 1594 帧压缩为 gist
  - = 2×120 + 4×120 + 1594×8 = 240 + 480 + 12,752 = 13,472 token
  - 节省比例 ≈ 93%

## 4. 工程视角 (Engineering View)

| 工程维度 | 数值/约束 | 含义 |
|------|------|------|
| **模型总参数量** | ~6B（视频DiT 5B + 动作DiT 1B） | 与 LingBot-VA 相当，但推理时动作 DiT 仅需 denoising |
| **视频 DiT 前向次数** | 每帧 1 次（推理时） | 不生成视频，仅更新 KV cache，大幅降低延迟 |
| **动作视界** | h = 16 步 | 每帧 latent 对应 16 个动作步（frame stride 4 × VAE stride 4） |
| **训练 GPU** | 8 GPU，per-GPU batch = 1 | 需要较大显存（Wan2.2-TI2V-5B 的视频 DiT 较大） |
| **推理延迟增长** | 1600 帧时仍低于 RNN/TTT | 论文 Fig.4a 显示混合记忆在长序列下比 RNN/TTT 更快 |
| **GPU 内存增长** | O(N/d) 而非 O(N) | 15× 压缩比下，长轨迹内存增长显著放缓 |
| **输入分辨率** | 384×320 mosaic（3 相机拼接） | head 256×320 + wrists 128×320，联合 VAE 编码 |
| **本体感知** | 14 维关节向量，投影到 text-token 维度 | 追加到动作 expert 的 text context 中 |

**工程含义**: MemoryWAM 的关键工程优势在于推理时不需要视频生成（与 FastWAM 等一致），同时 KV cache 的 15× 压缩使长视界任务可以在单卡或双卡上运行，而非 LingBot-VA 所需的多卡扩展。对于部署到真实机器人平台，这意味着更低的控制延迟——论文特别指出 LingBot-VA 在 Shell Game 任务中因高延迟错过了杯子交换时机。

## 5. 数据与评测 (Data & Eval)

**训练数据**:
- RMBench 基准：每任务 50 个 expert demonstrations
- 真实实验：ARX 双臂机器人 + RealSense D455 RGB 相机
- 数据增强：clean conditioning latent 与高斯噪声线性混合（随机比例 [0,1]，p=1.0，仅视频侧），防止 teacher-forcing 过拟合

**评测设置**:
- **RMBench**（仿真）：9 个双臂操作任务，涵盖不同 Task Memory Complexity 级别；每任务 100 次 rollout 报告成功率
- **真实世界**：2 个定制任务
  - Shell Game：人类交换杯子后识别含立方体的杯子
  - Look and Press：观察两个数字，按对应次数按压左右按钮，最后按后部按钮确认

**关键结果**（来自论文 Table 1 和项目页）:

| 任务 | π0.5 | FastWAM | LingBot-VA | MemoryWAM |
|------|------|---------|------------|-----------|
| Observe and Pick Up | 9% | 0% | 13% | **27%** |
| Rearrange Blocks | 13% | 0% | 100% | **100%** |
| Put Back Block | 11% | 0% | 100% | **100%** |
| Swap Blocks | 24% | 0% | 99% | **100%** |
| Swap T | 15% | 7% | 88% | **94%** |
| Battery Try | 16% | 20% | 41% | **41%** |
| Blocks Ranking Try | 6% | 26% | 100% | **100%** |
| Cover Blocks | 0% | 0% | 79% | **98%** |
| Press Button | 0% | 0% | 84% | **87%** |
| **平均** | **10.4%** | **5.9%** | **78.2%** | **83.0%** |

## 6. 能力与失败模式 (Capabilities & Failure Modes)

**能做什么**:
- 长视界非马尔可夫任务：需要记住早期观测（如初始物体位置、看到的数字）才能做出正确决策
- 部分可观测环境：关键信息被遮挡或移出视野后仍能基于记忆推理
- 实时操作：低推理延迟使其能在 Shell Game 等时效性任务中成功（LingBot-VA 因延迟高而失败）

**不能做什么 / 失败模式**:
- Battery Try 任务仅 41% 成功率（与 LingBot-VA 持平）：涉及电池方向组合导致仪表盘指针旋转，需要更精细的物理动力学建模，超出了当前记忆机制的能力
- 仅 50 demonstrations/task 的训练规模限制了复杂任务的极限性能
- gist token 压缩比为固定 15×，无法根据任务复杂度自适应调整——对于信息密度高的任务可能压缩过度，对简单任务则可能浪费容量

### 6.1 隐含假设 (Hidden Assumptions)

1. **任务初始帧总是最重要的事件边界**: 论文假设 N_init = 2 帧足以捕获所有关键初始信息。但对于多阶段任务（如先观察再操作再观察），中间的事件边界（如物体被移动到新位置）可能同样重要，未被覆盖
2. **gist token 的 15× 压缩比足够通用**: L=120 → M=8 是在 384×320 三相机拼接输入下设定的。对于更高分辨率输入（如 4K 单目相机），120 个 token 可能不够表达，压缩比可能不够
3. **训练时和推理时的 KV cache 可见性完全一致**: 论文声称"training-time visibility exactly reproduces the inference-time KV cache"。这要求训练数据的轨迹长度覆盖推理时的所有可能长度，否则长轨迹推理可能遇到未见过的注意力模式
4. **单机器人平台评估泛化性有限**: 所有实验基于 ARX 双臂机器人（14 维关节）。对于单臂、移动基座或人形机器人，混合记忆的有效性未验证

## 7. 与相关工作对比 (Comparison)

| 方法 | 核心关注点 | 架构 | 记忆方式 | 训练方式 | 适用场景 |
|------|------|------|------|------|------|
| **π0.5** | 直接观察→动作 | Transformer | 无记忆（当前帧） | Flow matching | 短视界、Markov 任务 |
| **FastWAM** | 高效 WAM | Video DiT + Action DiT | 滑动窗口（有界） | Flow matching | 中视界、大部分可见 |
| **LingBot-VA** | 全历史记忆 WAM | Autoregressive Transformer | 全历史 KV cache | Autoregressive | 长视界、但推理慢 |
| **MemoryWAM** | 高效持久记忆 | MoT (Video DiT + Action DiT) | 混合记忆（3层） | Flow matching | 长视界、实时要求 |

**面试 Tip**: 当被问及"为什么不用 RNN/TTT 做长期记忆"时，可以回答：RNN 和 TTT 虽然复杂度是 O(1)，但它们引入额外的网络参数和更新操作，在短序列时延迟反而高于混合记忆；更重要的是，在 Press Button 任务上，RNN 和 TTT 的成功率（~60-70%）显著低于混合记忆（87%），说明过度压缩的状态表示无法保留所有任务相关的细节。

## 8. 精讀建議 (Reading Guide)

- **值得精讀原文的人**:
  1. 构建长视界机器人策略的研究者——混合记忆机制可直接迁移到现有 WAM 架构
  2. 需要部署 WAM 到资源受限平台（单卡/边缘设备）的工程师——15× KV cache 压缩有直接部署价值
  3. 研究认知启发 AI 架构的学者——将人类短期/长期/事件边界记忆映射到 KV cache 管理的思路有启发意义

- **建議章節路徑**:
  - 先读 §3.3 Hybrid Memory（核心方法）→ 再看 §4.2 Memory Mechanism Comparison（机制对比实验）→ 可跳过 §4.1 Implementation Details（除非要复现）

- **不值得精讀的理由**:
  - 如果你不做机器人学习（只做 NLP/CV 的长期序列建模），这篇的 robot-specific 设计（如动作 DiT、VAE latent）距离较远
  - 如果你已熟悉 FastWAM 和 LingBot-VA，这篇的方法论增量主要是"混合记忆"这一设计选择，而非新的架构范式

---
[← Back to Theory](./README.md)

**关键引用**:
- 论文: https://arxiv.org/abs/2606.20562
- 项目页: https://yangsizhe.github.io/MemoryWAM/
- RMBench 基准: [论文引用 12]
- Wan2.2-TI2V-5B 预训练模型: [论文引用 55]
- LingBot-VA 基线: [论文引用 33]
- FastWAM 基线: [论文引用 62]
