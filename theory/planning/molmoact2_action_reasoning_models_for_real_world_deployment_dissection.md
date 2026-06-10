# MolmoAct2：面向真实世界部署的动作推理模型 (MolmoAct2: Action Reasoning Models for Real-world Deployment)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-05-06
>
> **论文**: MolmoAct2: Action Reasoning Models for Real-world Deployment
> **链接**: https://arxiv.org/abs/2605.02881
> **核心定位**: 首个"完全开源 + 可部署 + 高性能 + 快速推理"的四合一 VLA，用 Flow-Matching 专家 + Per-Layer KV 桥接 + 自适应深度推理，在 7 个仿真/真实基准上超越 π0.5，同时释放权重、数据、训练代码。

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | MolmoAct2 是首个在性能上超越闭源 π0.5 的完全开源 VLA，推理延迟低至 180ms（基础）/ 790ms（Think），支持三平台开箱部署 |
| 适合精读 | 如果你在做 VLA 架构设计、Flow Matching 在机器人中的应用、或需要开源可部署基线，重点看 §4（架构）和 §6（评测） |
| 可以跳过 | 如果你只关心纯触觉感知或纯扩散策略加速，这篇距离中等——它关注的是端到端 VLA 而非单一模块 |
| 落地可行性 | 高（权重/数据/代码全开源，支持 YAM/SO-100/DROID 三平台开箱部署） |
| 主要风险 | 自适应深度推理仅在场景变化区域预测 depth token，对静态场景增益有限；视觉遮挡仍是共性问题 |

💡 **X-Ray 开场**
VLA 模型长期面临一个"不可能三角"：开源 vs 高性能 vs 低延迟——三者不可兼得。闭源模型（如 π0.5）性能好但不可复现；开源模型要么性能差、要么延迟高到无法闭环控制。MolmoAct2 的核心突破是用 **Per-Layer KV 桥接 + Flow Matching 专家** 的架构设计，在保持 VLM 预训练知识的同时，将推理延迟从 MolmoAct 的 6700ms 压到 180ms（37x 加速），同时性能超越 π0.5。对 VLA 研究者来说，这意味着开源 VLA 首次在"可部署"维度上追平了闭源方案。

📍 **研究全景时间线**
```
[2024] RT-1 (Google) → 首个端到端 VLA，离散动作token
       ↓
[2024] OpenVLA → 首个开源 VLA 基线，但性能有限
       ↓
[2025] FAST Tokenizer (PI) → 连续动作→离散token的高效编码，但数据未开源
       ↓
[2025] MolmoAct (v1) → 首个 Action Reasoning Model，3D depth reasoning，但推理慢(6.7s)
       ↓
[2025] π0.5 (PI) → 闭源高性能基线，但不可复现
       ↓
[2026] MolmoAct2 ← 当前位置：开源+高性能+低延迟，Per-Layer KV桥接 + Flow Matching
       ↓
    局限：仍依赖单一视觉模态，触觉/力觉未纳入
```

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 维度 | MolmoAct (v1) | π0.5 (闭源) | MolmoAct2 |
|------|---------------|-------------|-----------|
| VLM Backbone | Molmo2-ER 前身 | 闭源 | Molmo2-ER (Qwen3-4B 特化) |
| 动作表示 | 离散 token | 连续 Flow | 离散 token + 连续 Flow 双输出 |
| 动作专家 | 无 | 闭源 Flow | DiT-style Flow Matching，Per-Layer KV 桥接 |
| 推理延迟 | 6700ms | 未公开 | 180ms (base) / 790ms (Think) |
| 动作Tokenizer | 闭源 | 闭源 | OpenFAST (2048 vocab, 5 embodiment 训练) |
| 开源程度 | 权重 | 仅权重 | 权重 + 数据 + 代码 |
| 部署平台 | 单臂 | 闭源 | YAM 双臂 / SO-100/101 / DROID Franka |
| 推理变体 | 无 | 无 | MolmoAct2-Think (自适应深度推理) |

### 1.2 关键机制 (Key Mechanism)

MolmoAct2 的架构创新集中在三个层面：

**（1）Molmo2-ER：具身推理特化 VLM**
- 从 Molmo2 (Qwen3-4B) 出发，用 3.3M 样本的具身推理语料微调
- 采用 **Specialize-then-Rehearse** 两阶段训练：
  - Stage 1 (20K steps): 在具身语料上微调，快速迁移到具身数据流形
  - Stage 2 (1.5K steps): 混合原始 Molmo2 多模态数据（p=0.5 比例），防止灾难性遗忘
- 在 13 个具身推理基准上平均 63.8%，超越 GPT-5 和 Gemini Robotics ER-1.5

**（2）OpenFAST Tokenizer：开源动作分词器**
- 将 1 秒连续动作轨迹 → 频域变换 → 量化 → BPE → 2048-token 词汇表
- 训练数据覆盖 5 种机器人平台（YAM 双臂 / SO-100/101 / DROID Franka / BC-Z / Bridge），100 万条轨迹
- 所有动作维度 pad 到 32D，1-99 百分位归一化，gripper 单独处理

**（3）Per-Layer KV 桥接 + Flow Matching 专家**（核心创新）
- 预训练阶段：VLM 预测离散 action token（标准 next-token prediction）
- 后训练阶段：在 VLM 上嫁接 DiT-style Flow Matching 专家，输出连续动作轨迹
- **关键设计**：专家不直接读取 VLM hidden states，而是通过 **per-layer KV cache** 桥接
  - 对每个 VLM layer ℓ，提取 self-attention 的 Kℓ 和 Vℓ
  - 用可学习投影 PK/PV 映射到专家维度
  - 专家每层 cross-attention 直接 attend 到对应 VLM 层的 KV cache
- 这保证了 VLM 的视觉-语言注意力状态无损传递到连续控制器，同时通过 knowledge insulation 防止 flow loss 回传破坏 VLM

⚡ **Eureka Moment**：用 VLM 的 per-layer KV cache（而非 hidden states）作为连续动作专家的 conditioning 源——既保留了 VLM 预训练知识的完整性（knowledge insulation），又让每个专家层都能直接访问同深度的视觉-语言注意力状态，实现了离散预训练与连续控制的"无缝嫁接"。

### 1.3 信息流/架构图 (Flow / Diagram)

```
                    ┌─────────────────────────────────────────────┐
                    │              INPUT SEQUENCE                  │
  Instruction ─────►│  [IMG] task desc [STATE] <action_output>    │
  Camera(s) ───────►│         ↑ ViT (SigLIP2) → Connector → LLM   │
  Robot State ─────►│                                    │        │
                    └────────────────────────────────────┼────────┘
                                                         │
                              ┌──────────────────────────┼──────────────────┐
                              │    VLM Backbone (36L)     │                  │
                              │  Self-Attention Layers    │                  │
                              │                           │                  │
                              │  Each layer ℓ produces:   │                  │
                              │    K_ℓ^vlm, V_ℓ^vlm       │                  │
                              └───────────┬───────────────┘                  │
                                          │                                  │
                        Per-Layer KV Bridge│                                  │
                        K̃_ℓ = P_K · K_ℓ^vlm                                │
                        Ṽ_ℓ = P_V · V_ℓ^vlm                                │
                                          │                                  │
                              ┌───────────▼───────────────┐                  │
                              │  Flow Matching Expert (36L) │                  │
                              │                           │                  │
                              │  Block ℓ:                 │                  │
                              │    h'_ℓ = h_ℓ + g_sa·SA   │                  │
                              │    h̄_ℓ = h'_ℓ + g_ca·CA   │← attends to K̃_ℓ,Ṽ_ℓ
                              │    h_{ℓ+1} = h̄_ℓ + g_ff·MLP              │
                              │                           │                  │
                              │  Output: continuous       │                  │
                              │  action trajectory        │                  │
                              └───────────────────────────┘                  │
                                                                             │
  Discrete Action Tokens ◄───────────────────────────────────────────────────┘
  (from VLM LM head, next-token prediction)
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：
```
L_post = L_LM + L_flow
     = -Σ log p(action_token | context) + E[||m ⊙ (f_θ(x_t, t, c) - (a - ε))||²₂]
```

**目标**：同时训练离散 token 预测（保持 VLM 能力）和连续 Flow Matching（输出可执行动作），两者共享同一个 VLM backbone。

**Flow Matching 核心方程**：

```
给定: 归一化动作 a, 高斯噪声 ε, 时间 t ∈ [0,1]
插值: x_t = (1-t)·ε + t·a
目标: u = a - ε (速度场)

L_flow = E[||m ⊙ (f_θ(x_t, t, c) - u)||²₂]
```

**变量说明**：

| 符号 | 含义 |
|------|------|
| x_t | t 时刻的噪声动作（插值 between noise and data） |
| t | flow time，均匀采样自 [0,1] |
| ε | 标准高斯噪声 |
| a | 归一化目标动作 chunk（30 steps × 32 dims max） |
| f_θ | DiT-style action expert |
| c | VLM context（任务/视觉/状态/setup-control descriptors） |
| m | mask（padding steps 和 padding dims） |
| K | flow sample 数量（post-training: 4, fine-tuning: 8） |

**直觉**：Flow matching 学习一个从噪声到真实动作的"速度场"。推理时从纯高斯噪声出发，沿学习到的速度场积分，逐步"去噪"得到连续动作轨迹。Per-layer KV 桥接确保每个去噪步骤都能访问 VLM 的视觉-语言注意力状态。

> 符号与本文保持一致：f_θ 为 action expert，K_ℓ^vlm/V_ℓ^vlm 为 VLM 第 ℓ 层的 KV cache，P_K/P_V 为 adapter 投影。

## 3. 带数字走一遍：玩具例子 (Worked Example)

假设一个简化的 2D 抓取任务：

```
场景: 机械臂从 (x=0, y=0) 移动到 (x=10, y=5) 抓取物体
动作维度: 2D (x, y 方向的速度)
控制频率: 10 Hz, 1 秒 = 10 步

Step 1: 归一化
  目标动作 a = [1.0, 0.5, 1.0, 0.5, ..., 1.0, 0.5] (10 steps × 2 dims)
  归一化后: a_norm = [0.8, 0.3, 0.8, 0.3, ...] (1-99 百分位)

Step 2: 加噪 (t=0.5)
  ε = [0.2, -0.1, 0.3, 0.1, ...] (随机高斯)
  x_0.5 = (1-0.5)·ε + 0.5·a_norm
        = [0.5, 0.1, 0.55, 0.2, ...]

Step 3: Expert 预测速度
  f_θ(x_0.5, t=0.5, c) → 预测 u = a_norm - ε
  损失: ||f_θ(x_0.5, 0.5, c) - (a_norm - ε)||²

Step 4: 推理 (从噪声到动作)
  从 ε ~ N(0,I) 出发
  t=0.0 → x_0 = ε
  t=0.1 → x_0.1 = x_0 + 0.1 · f_θ(x_0, 0, c)
  ...
  t=1.0 → x_1 ≈ a_norm (接近目标动作)

Step 5: Per-Layer KV 桥接的作用
  每个去噪步骤 t_i，expert 的第 ℓ 层都通过 cross-attention
  访问 VLM 第 ℓ 层的 K_ℓ^vlm, V_ℓ^vlm
  → 模型"看到"摄像头中的物体位置，决定往哪移动
```

**关键数字**：
- 训练时每个 action chunk 采样 K=4 个不同的 t 值（post-training）/ K=8（fine-tuning）
- 每个 t 值贡献一个 flow loss 项，共享同一个 VLM context
- 推理时需要 O(10-50) 步数值积分（取决于 flow solver 步数）

## 4. 工程视角 (Engineering View)

| 指标 | MolmoAct (v1) | MolmoAct2 (base) | MolmoAct2 (Think) |
|------|---------------|-------------------|-------------------|
| 单次推理延迟 | 6700ms | 180ms | 790ms |
| 加速比 | 1x | 37x | 8.5x |
| 硬件需求 | 1×H100 | 1×H100 | 1×H100 |
| 动作输出频率 | ~0.15 Hz | ~5.5 Hz | ~1.3 Hz |
| 模型深度 | 36 layers | 36+36 (VLM+Expert) | 36+36+depth |
| 训练总 GPU 时 | ~8000h (估算) | ~11,560h | 含在 Think 微调中 |

**训练计算量拆解**：
- Pre-training: 200K steps, batch=128, 64×H100 → ~5,760 GPU hours
- Post-training: 100K steps, batch=128, 64×H100 → ~2,300 GPU hours
- Fine-tuning (YAM): 100K steps, batch=128, 64×H100 → ~2,300 GPU hours
- Fine-tuning (DROID): 100K steps, batch=64, 32×H100 → ~1,150 GPU hours
- Fine-tuning (SO-100/101): 100K steps, batch=64, 32×H100 → ~1,150 GPU hours
- Fine-tuning (LIBERO): 50K steps, batch=64, 32×H100 → ~1,150 GPU hours
- **总计**: ~13,810 GPU hours（约 575 H100 天）

**工程含义**：
1. **180ms 延迟**意味着 ~5.5 Hz 控制频率，接近真实机器人闭环控制需求（通常 10-30 Hz）。虽然仍有差距，但已从"不可用"提升到"可部署"。
2. **Per-Layer KV 桥接**避免了 backprop through VLM，post-training 时 VLM 梯度被 detach，大幅降低显存需求。
3. **Knowledge Insulation**：post-training 阶段 detach KV cache，flow loss 不回传 VLM → 防止连续动作训练破坏预训练的视觉-语言能力。fine-tuning 阶段则不禁用（未发现一致收益）。
4. **Flow sample 数量**：post-training 用 K=4（显存限制），fine-tuning 用 K=8（更密集的 flow 轨迹监督）。

## 5. 数据与评测 (Data & Eval)

### 5.1 训练数据组成

| 数据集 | 规模 | 平台 | 特点 |
|--------|------|------|------|
| MolmoAct2-BimanualYAM | 34.5K demos, 720h, 28+ tasks | YAM 双臂 | 最大开源双臂数据集，家庭/工厂/咖啡场景 |
| MolmoAct2-DROID | 74,604 episodes, 17.8M frames | DROID Franka | 质量过滤，去空闲帧，语言重标注 |
| MolmoAct2-SO100/101 | 38,059 episodes, 19.8M frames, 184h | SO-100/101 | 1,222 社区数据集，4 阶段过滤（含 TOPReward） |
| Open X-Embodiment 子集 | BC-Z + Bridge V2 + RT-1 | 多平台 | 扩展 embodiment 多样性 |
| MolmoAct Dataset | 10.6K trajectories | 单臂 | v1 数据保留 |
| Molmo2-ER 语料 | 3.3M samples | 非机器人 | 具身推理特化（pointing/detection/VQA/ego-exo） |

### 5.2 评测结果（关键数字）

**仿真基准**：

| 基准 | MolmoAct2 | π0.5 | MolmoAct (v1) | 提升 |
|------|-----------|------|---------------|------|
| MolmoBot (all tasks) | 20.6% | 10.3% | — | 2x |
| RoboEval (bimanual) | 0.443 | 0.405 | — | +9% |
| LIBERO (after FT) | 97.2% | — | ~86.6% | +10.6pts |
| LIBERO-Think (after FT) | 98.1% | — | ~86.6% | +11.5pts |

**真实世界零样本（Franka 臂，15 trials/task）**：

| 任务 | MolmoAct2 | MolmoBot | π0.5 |
|------|-----------|----------|------|
| Apple on plate | 100% | — | — |
| Pipette in tray | 86.7% | — | — |
| Red cube in tape roll | 93.3% | — | — |
| Knife in box | 93.3% | — | — |
| Multi-object to bowl | 62% | — | — |
| **平均** | **87.1%** | **48.4%** | **45.2%** |

**第三方评估（Cortex AI，双臂任务）**：

| 模型 | 平均得分 | 8 任务中第几名 |
|------|----------|---------------|
| MolmoAct2 | 0.51 | #1 (7/8 tasks) |
| OpenVLA-OFT | 0.36 | — |
| π0.5 | 0.32 | — |
| Cosmos Policy | 0.16 | — |
| X-VLA | 0.05 | — |

**Molmo2-ER 具身推理基准（13 个基准平均）**：

| 模型 | 平均分 |
|------|--------|
| Molmo2-ER | 63.8 |
| GPT-5 | <63.8 |
| Gemini Robotics ER-1.5 | <63.8 |
| Molmo2 (base) | 46.8 |

> 所有数字来自论文正文和 AI2 博客。论文来源标注为"论文 Sec.6"或"博客"。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

### 6.1 能做什么

| 能力 | 条件 | 证据 |
|------|------|------|
| 开箱双臂操作 | YAM 平台 | 720h 双臂数据预训练，零样本双臂任务 |
| 开箱单臂操作 | DROID Franka / SO-100 | 零样本 Franka 87.1% 成功率 |
| 快速微调适配 | 新 embodiment | LIBERO FT 后 97.2%（50K steps） |
| 3D 深度推理 | 需要空间理解的任务 | MolmoAct2-Think 自适应 depth token |
| 自然语言指令 | 通用任务描述 | VLM re-annotation 提升语言多样性 2x |
| 视觉轨迹引导 | 2D trace steering | 博客提及，早期能力 |

### 6.2 不能做什么

| 失败模式 | 原因 |
|----------|------|
| 自遮挡（gripper 遮挡相机视野） | 纯视觉模态，无深度/触觉冗余 |
| 高频精细操作 | 180ms 推理延迟限制了控制频率上限 |
| 深度轴误差（视觉引导时） | 2D trace 缺乏深度信息 |
| 触觉/力觉任务 | 未纳入触觉模态，纯视觉-语言-动作 |
| 极端场景泛化 | 训练数据集中在桌面/家庭场景 |

### 6.3 隐含假设 (Hidden Assumptions)

1. **Per-layer KV 桥接的层对齐假设**：VLM 第 ℓ 层的注意力状态与 Expert 第 ℓ 层的抽象层级匹配。论文假设 L=36 的对称结构使这种对齐自然成立，但未做消融实验验证非对称结构（如 VLM 28 层 + Expert 36 层）的效果。

2. **Knowledge Insulation 在 post-training 阶段的必要性**：论文 detach KV cache 防止 flow loss 回传 VLM，但在 fine-tuning 阶段发现"未观察到一致收益"后放弃了 insulation。这暗示 insulation 的价值可能随任务阶段变化，但未给出系统性分析。

3. **OpenFAST 的跨 embodiment 泛化假设**：Tokenizer 用 5 种平台 100 万条轨迹训练，假设 2048-token 词汇表能覆盖所有平台的动作空间。但未报告 reconstruction error 或 tokenizer 消融。

4. **自适应深度推理的静态场景假设**：MolmoAct2-Think 仅在"场景变化区域"预测 depth token，假设静态区域的 depth 可以从上一帧复用。但未量化"场景变化"的检测阈值和复用误差。

5. **语言重标注的质量假设**：用 Qwen3.5-27B 重标注机器人数据，假设 VLM 生成的指令比原始标注更准确多样。但未见人工评估重标注质量的实验。

## 7. 与相关工作对比 (Comparison)

| 模型 | 开源程度 | 动作表示 | 推理架构 | 延迟 | 性能 | 部署平台 |
|------|----------|----------|----------|------|------|----------|
| RT-1 (Google) | 权重 | 离散 token | Direct mapping | 未知 | 基线 | Google Robot |
| OpenVLA | 权重+代码 | 连续 (LoRA) | VLM + LoRA head | 未知 | 中 | Franka |
| π0.5 (PI) | 权重 | 连续 Flow | 闭源架构 | 未知 | 高 | 闭源 |
| MolmoAct (v1) | 权重+数据 | 离散 + depth token | VLM + CoT reasoning | 6700ms | 中高 | 单臂 |
| **MolmoAct2** | **权重+数据+代码** | **离散 + 连续 Flow** | **VLM + Per-Layer KV + Flow Expert** | **180ms** | **高** | **YAM/SO-100/DROID** |

**面试 Tip**：当被问到"MolmoAct2 与 π0.5 的核心区别"时，回答："π0.5 是闭源 Flow-based VLA，性能强但不可复现；MolmoAct2 是首个在性能上超越 π0.5 的完全开源方案，核心创新是 Per-Layer KV 桥接——用 VLM 每层的 KV cache 而非 hidden states 连接 Flow Matching 专家，实现了离散预训练与连续控制的无缝嫁接，同时通过 Knowledge Insulation 保护 VLM 预训练知识。"

## 8. 精讀建議 (Reading Guide)

**值得精讀原文的人**：
- 做 VLA 架构设计的研究者——Per-Layer KV 桥接是一个可复用的设计模式，可能影响下一代 VLA 架构
- 需要开源可部署 VLA 基线的工程师——MolmoAct2 是目前唯一"权重+数据+代码"全开源且性能超越 π0.5 的方案
- 研究 Flow Matching 在机器人中应用的人——论文详细描述了 DiT-style expert 与 VLM 的耦合方式

**建议章节路径**：
- 先读 §4（MolmoAct2 架构）→ 理解三阶段训练管线和 Per-Layer KV 桥接
- 再看 §6（评测）→ 了解在 7 个基准上的具体表现
- 可跳 §2（Molmo2-ER）和 §3（数据）→ 如果不关注 VLM 特化和数据工程细节

**不值得精讀的理由**：
- 如果你只做纯触觉感知或纯扩散策略加速，这篇的 VLA 架构设计距离你的核心问题较远
- 如果你已熟悉 MolmoAct v1 和 FAST tokenizer，这篇的增量主要在架构重组而非全新概念

---
[← Back to Theory](./README.md)

**关键引用**：
- 论文: https://arxiv.org/abs/2605.02881
- 项目页: https://allenai.org/blog/molmoact2
- 模型权重: https://huggingface.co/collections/allenai/molmoact2-models
- 数据集: https://huggingface.co/collections/allenai/molmoact2-datasets
- Molmo2-ER 基准对比: 论文 Table 3 / 博客
- Cortex AI 第三方评估: 博客