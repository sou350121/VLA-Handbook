# MotionWAM：迈向实时人形机器人世界动作模型 (MotionWAM: Towards Foundation World Action Models for Real-Time Humanoid Loco-Manipulation)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-06-10
>
> **论文**: MotionWAM: Towards Foundation World Action Models for Real-Time Humanoid Loco-Manipulation
> **链接**: https://arxiv.org/abs/2606.09215
> **核心定位**: 将 WAM 从迭代去噪的慢速瓶颈中解放出来，用单步中间特征实现实时人形机器人全身移动操作，在 9 个真实 Unitree G1 任务上比最强 VLA 基线高出 32% 绝对成功率。

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | 用 Video DiT 的单步中间去噪特征替代完整迭代去噪，配合统一全身动作潜变量，实现 4.9 Hz 实时 WAM 控制，在 9 个真实人形任务上达到 76.1% 成功率 |
| 適合精讀 | 做 WAM/世界模型+策略耦合的研究者；需要实时全身控制的人形机器人工程师；关注 Cosmos 生态迁移的研究者 |
| 可以跳過 | 只关心桌面机械臂操作、不涉及人形/全身控制的读者 |
| 落地可行性 | 中（需要 Cosmos-Predict2.5-2B 预训练权重 + 大量 egocentric 视频 + 200 episodes/任务 teleop 数据） |
| 主要風險 | 仅在 Unitree G1 上验证；Stage 3 数据需求对每个新平台都是瓶颈 |

💡 **X-Ray 开场**
现有 WAM 通过迭代去噪未来视频帧来预测动作——这在桌面机械臂上可行，但对人形机器人来说太慢了。MotionWAM 发现：不需要等视频帧完全去噪，只需在扩散过程的纯噪声端读取 Transformer 的中间激活，就能获得足够的"场景将往哪里去"的信息。这一观察把推理速度提升了 7 倍，同时让双腿从"保持平衡的配角"变成了"参与任务的主动角色"（踢球、踩踏板）。

📍 **研究全景时间线**
```
[2023] ACT/Diffusion Policy (桌面臂) → [2024] π0/GR00T (VLA 通用策略) → [2025] WAM 初代 (迭代去噪, 桌面臂)
  → [2025末] Cosmos Policy (迭代去噪, 0.7 Hz) → [2026-06] MotionWAM ← 当前位置
  (单步中间特征, 4.9 Hz, 全身统一动作空间, 人形 loco-manipulation)
  ← 局限: 仅 Unitree G1, 未见跨平台验证
```

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 组件 | 传统分层方案 | MotionWAM |
|------|-------------|-----------|
| **上层策略** | 仅控制上肢关节目标 | 无分层——单一策略输出全身动作 |
| **下层控制器** | 跟踪粗粒度基座命令（速度/高度/朝向） | SONIC 全身控制器解码统一运动潜变量 |
| **动作空间** | 上肢=精细关节角；下肢=粗粒度基座命令（不一致） | 统一全身运动潜变量（运动+躯干+高度+足部交互+手部操作） |
| **视觉先验** | VLM 静态图像-文本特征 | Video DiT 中间去噪特征（动态视频先验） |
| **推理方式** | 单步前向 | Video DiT 单步前向 + Motion DiT 单步前向 |
| **足部行为** | 仅限保持平衡 | 任务驱动（踢球、踩踏板、扫垃圾） |
| **推理频率** | VLA ~10-20 Hz（但无视频先验） | 4.9 Hz（有视频先验） |

### 1.2 关键机制 (Key Mechanism)

MotionWAM 的核心设计围绕两个关键决策：

**决策 1：用中间去噪特征替代完整去噪**
传统 WAM 需要多步迭代去噪未来视频帧，然后从去噪后的帧中提取特征。MotionWAM 在扩散过程的纯噪声端（$\tau_f \approx 1$）设置一个 forward hook，直接从 Video DiT 的 velocity network 读取中间激活。此时未来帧仍然是高斯噪声，但给定干净的当前条件帧 $z_t^0$ 和语言目标 $l$，Transformer 的激活已经编码了"场景将往哪里去"的信息。这一单步操作是实时性的关键。

**决策 2：统一运动潜变量替代上下肢分离**
基于 SONIC 全身控制器，MotionWAM 用一个统一的运动潜变量 $m_t = (m_t^{\text{cont}}, k_t)$ 覆盖所有身体部位：
- $k_t \in \{-1, -15/16, \dots, 1\}^{64}$：SONIC 的有限标量量化（FSQ）token，64 维离散向量，汇总运动、躯干、高度和足部交互意图  
- $m_t^{\text{cont}}$：连续通道，覆盖 SONIC 未覆盖的灵巧手/夹爪控制  

⚡ **Eureka Moment**：不需要等视频帧完全去噪——在纯噪声端读取 Transformer 中间激活，就足以获得"场景将往哪里去"的语义编码，这一观察把推理速度从 0.7 Hz 提升到 4.9 Hz。

### 1.3 信息流/架构图 (Flow / Diagram)

```
┌─────────────────────────────────────────────────────────────────┐
│                    MotionWAM 端到端信息流                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  当前帧 o_t ──→ VAE 编码 ──→ z_t^0 (干净潜变量)                  │
│       │                                                         │
│       └──→ Language Embedding (Cosmos-Reason1)                   │
│                                                                  │
│  噪声 ε ~ N(0,I) ──→ z_{t+1}^{τf} (纯噪声未来帧, τf≈1)          │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────────────────────────────────┐                   │
│  │         Video DiT (Cosmos-Predict2.5-2B)  │                   │
│  │  输入: z_{t+1}^{τf}, τf | z_t^0, l       │                   │
│  │  输出: h_t^{τf} = H[v_θ](...) ← forward   │                   │
│  │           hook 在纯噪声端读取激活            │                   │
│  └──────────────────────────────────────────┘                   │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────────────────────────────────┐                   │
│  │         Motion DiT (共享主干)              │                   │
│  │  输入: h_t^{τf}, p_t (本体感受),          │                   │
│  │          ε_m (噪声运动潜变量)               │                   │
│  │  输出: m_t = (m_t^cont, k_t)              │                   │
│  └──────────────────────────────────────────┘                   │
│       │                                                         │
│       ▼                                                         │
│  round(k_t) → SONIC 解码 → 关节命令 a_t → Unitree G1            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：
```
h_t^{τf} = H[Video DiT](z_{t+1}^{τf}≈noise, τf≈1 | z_t^0, l)
m_t = Motion DiT(h_t^{τf}, p_t, ε_m)  →  round(k_t) → SONIC → a_t
```

**目标**：用 Video DiT 的单步中间激活替代多步迭代去噪，将视频动态先验注入策略，同时保持实时推理速度。

**核心方程**：

1. **Video DiT 中间特征提取**（Eq. 2）：
```
h_t^{τf} = H[v_θ^{video}](z_{t+1}^{τf}, τf | z_t^0, l)
其中 z_{t+1}^{τf}|_{τf→1} ~ N(0, I)
```
在纯噪声端（$\tau_f \approx 1$）读取 velocity network 的隐藏状态——未来帧尚未去噪，但激活已编码场景演化方向。  

2. **Video 流匹配损失**（Eq. 3）：
```
L_video = E_{τv, z_{t+1}^0, εv} [||v_θ^{video}(z_{t+1}^{τv}, τv | z_t^0, l) - (εv - z_{t+1}^0)||_2^2]
```
学习从噪声到干净未来帧的速度场。

3. **Motion 流匹配损失**（Eq. 4）：
```
L_motion = E_{τa, m_t^0, εm} [||v_φ^{motion}(m_t^{τa}, τa | h_t^{τf}, p_t, e) - (εm - m_t^0)||_2^2]
```
以 Video DiT 中间特征为条件，学习运动潜变量的速度场。

4. **Stage 2/3 联合损失**（Eq. 5）：
```
L = L_motion + L_video
```
Video 损失作为表示正则化，防止动作信号覆盖动态先验。

5. **运动潜变量解码**（Eq. 6）：
```
m_t = (m_t^cont, k_t~) → 流匹配 → m_hat_t = (m_hat_t^cont, k_hat_t~)
→ k_hat_t = round(k_hat_t~) → SONIC → a_t
```
SONIC token 索引作为连续标量在流匹配中回归，推理时最近邻取整后解码为关节命令。

> 符号与论文保持一致：$o_t = \text{egocentric 观测}$, $p_t = \text{本体感受状态}$, $l = \text{语言目标}$, $e = \text{实体索引}$, $\tau = \text{流匹配时间步}$, $H[\cdot] = \text{隐藏状态提取算子}$。  

## 3. 带数字走一遍：玩具例子 (Worked Example)

假设一个简化场景：人形机器人需要执行"踢球"任务。

**输入**：
- 当前帧 o_t：机器人视角看到足球在前方 1.5m
- 语言目标 l = "kick the soccer ball"
- 本体感受 p_t：当前站立姿态，双足着地

**Video DiT 单步前向**：
- $z_t^0 = \text{VAE}(o_t)$：当前帧潜变量  
- $z_{t+1}^{\tau_f} = \mathcal{N}(0, I)$：纯噪声（$\tau_f = 0.99$）  
- 经过 Video DiT 一层前向：$h_t^{\tau_f} \in \mathbb{R}^{N \times D}$（$N=$序列长度, $D=$隐藏维度）  
- 这个激活向量编码了"球将被踢飞、身体前倾、右腿前摆"的语义方向

**Motion DiT 前向**：
- 输入：$h_t^{\tau_f}$（交叉注意力）+ $p_t$（拼接）+ $\varepsilon_m \sim \mathcal{N}(0,I)$（噪声运动潜变量）  
- 流匹配去噪：经过若干步（或单步 flow-matching），输出 m_hat_t
- m_hat_t = (m_hat_t^cont, k_hat_t~) = ([0.3, -0.1, 0.8, ...], 3.7)
- $\text{round}(3.7) = 4$ → SONIC token 4 对应"右腿前摆踢球"的全身运动模式  

**SONIC 解码**：
- Token 4 的 64 维 FSQ 向量 → 全身关节目标  
- 包含：右髋屈曲 30°、右膝伸展、左腿支撑、躯干前倾 15°
- 输出 $a_t$ → 关节控制器执行  

**关键**：整个过程只需 Video DiT 一次前向 + Motion DiT 一次前向，在 A100 上约 200ms，实现 4.9 Hz 控制频率。

## 4. 工程视角 (Engineering View)

| 维度 | 数值/约束 | 工程含义 |
|------|----------|---------|
| **推理频率** | 4.9 Hz (A100) | 满足人形平衡最低要求（~5 Hz），但余量不大 |
| **对比 Cosmos Policy** | 7x 加速（4.9 vs 0.7 Hz） | 核心差异：单步中间特征 vs 迭代去噪 |
| **对比 VLA 基线** | 略低但可比（VLA 通常 10-20 Hz） | 用 2-4x 速度换取视频动态先验，trade-off 值得 |
| **硬件** | 单 RTX 4090 工作站 + 机载控制器 | WebSocket 策略服务器模式，非机载推理 |
| **模型规模** | Video DiT: Cosmos-Predict2.5-2B (2B) + Motion DiT (~数百M) | 总参数量与 GR00T-N1.7 可比 |
| **训练数据** | Stage 1: ~2,136h 视频；Stage 3: 200 episodes/task × 9 tasks   | Stage 1 数据可廉价获取（无动作标注）；Stage 3 是瓶颈 |
| **流匹配步数** | 论文未明确给出推理步数 | 疑似单步或极少步数 flow-matching（需确认） |
| **部署约束** | 需要 Cosmos 生态（VAE + text encoder 冻结） | 迁移到其他平台需适配 Cosmos 预训练权重 |

**工程含义总结**：MotionWAM 的实时性来自于"用信息换计算"——不追求完美的未来帧重建，而是在噪声端提取足够好的语义方向。这是一个经典的工程权衡：牺牲视频重建质量，换取 7x 速度提升和 32% 成功率提升。

## 5. 数据与评测 (Data & Eval)

**数据组成**：
- **Stage 1**（~2,136h）：egocentric 人类视频 + 人形机器人视频，无动作标注，仅用于视频帧预测
- **Stage 2**：Unitree G1 异构数据，覆盖不同末端执行器和动作标注格式
- **Stage 3**：200 episodes/任务 × 9 任务 = 1,800 episodes 全身 teleoperation 数据，通过 PICO VR 三点追踪采集，经 SMPL 重定向到 Unitree G1  

**评测设置**：
- 9 个真实 Unitree G1 全身 loco-manipulation 任务
- 每个任务 20 次试验，报告成功率百分比
- 任务覆盖 5 种核心能力：腰部控制、高度调节、蹲姿移动、任务驱动足部交互、身体-手协调
- 所有基线使用相同的 Stage 3 数据集微调，通过相同的 SONIC 接口输出动作

**任务列表**（论文 Figure 3）：
1. Lift Basket（提篮子）
2. Retrieve Item（取物）
3. Load Cart（装推车）
4. Toss Garbage（扔垃圾）
5. Kick Soccer（踢足球）
6. Wipe Board（擦黑板）
7. Do Laundry（做家务）
8. Step on Pedal（踩踏板）
9. 第 9 个任务名称在截断中未完整显示

## 6. 能力与失败模式 (Capabilities & Failure Modes)

**能做什么**：
- 实时（4.9 Hz）全身 loco-manipulation 控制
- 任务驱动足部交互（踢球、踩踏板）——分层方案无法做到
- 从单目 egocentric 相机实现端到端控制
- 在 9 个任务上全面超越 VLA 基线，整体成功率 76.1% vs 43.9%

**不能做什么 / 局限**（论文 §6 + 实验观察）：
- **仅验证 Unitree G1**：三阶段训练范式未在其他 humanoid 平台上验证迁移性
- **Stage 3 数据需求**：每个新平台/任务集需要 ~200 episodes teleoperation，规模化成本高
- **依赖 Cosmos 生态**：Video DiT 初始化自 Cosmos-Predict2.5-2B，迁移到其他视频生成模型需重新适配
- **4.9 Hz 控制频率余量小**：在 A100 上达到 4.9 Hz，但部署到机载计算平台（如 Jetson）可能进一步下降
- **单目视觉**：仅依赖 head-mounted RealSense D435i RGB，无深度/多视角冗余

### 6.1 隐含假设 (Hidden Assumptions)

1. **中间激活足够编码未来动态**：论文假设 $\tau_f \approx 1$ 时的 Transformer 激活足以指导动作预测，但未系统探索不同 $\tau_f$ 值的性能曲线。如果最优 $\tau_f$ 因任务而异（精细操作 vs 大步移动），固定 $\tau_f$ 可能次优。  

2. **Egocentric 视频预训练足以捕获机器人动态**：Stage 1 使用人类 egocentric 视频 + 少量机器人视频，假设这些视觉动态可以迁移到 Unitree G1 的视角。但未量化"多少机器人视频是必要的"。

3. **SONIC token 足够表达所有任务**：64 维 FSQ token（32 级）假设全身运动可以被离散化为有限词汇。对于超出训练分布的复杂足部交互，token 分辨率可能不足。

4. **流匹配单步推理足够**：论文暗示 Motion DiT 可以单步或少步推理，但未给出不同推理步数下的性能-速度 trade-off 曲线。

## 7. 与相关工作对比 (Comparison)

| 方法 | 关注点 | 架构 | 训练方式 | 适用场景 | 实时性 |
|------|--------|------|---------|---------|--------|
| **MotionWAM** | 人形全身 loco-manipulation | 双 DiT + 中间特征 | 三阶段 | Unitree G1 | 4.9 Hz ✅ |
| **GR00T-N1.7** | 通用人形策略 | VLM + 动作头 | 大规模预训练+微调 | 多平台 | ~10-20 Hz |
| **π₀.₅** | 通用机器人策略 | 流匹配 + VLM | 大规模预训练 | 多平台 | ~10-20 Hz |
| **Cosmos Policy** | WAM 策略耦合 | 迭代去噪 WAM | 视频+动作联合 | 桌面臂 | 0.7 Hz ❌ |
| **ACT** | 桌面臂操作 | Transformer + 去噪 | 演示学习 | 桌面臂 | ~10 Hz |
| **Diffusion Policy** | 桌面臂操作 | Diffusion + CNN | 演示学习 | 桌面臂 | ~10 Hz |
| **Qwen3DiT** (ablation) | VLM vs 视频先验 | Qwen3-VL 2B + Motion DiT | 同 Stage 3 | Unitree G1 | ~10 Hz |

**关键区分**：
- 与 VLA 基线（GR00T-N1.7, $\pi_{0.5}$）的唯一差异是：MotionWAM 用 Video DiT 中间特征替代 VLM 图像-文本特征，其余（相同数据、相同动作空间、相同 SONIC 接口）全部一致
- 与 Cosmos Policy 的核心差异：单步中间特征 vs 迭代去噪 → 7x 速度差异
- Qwen3DiT ablation 证明：VLM 静态先验在运动密集型任务上接近零成功率，视频动态先验是性能差异的来源

> 💡 **面试 Tip**：如果被问到"WAM 和 VLA 在人形控制上的核心区别是什么"，回答：VLA 学习静态图像到动作的映射，缺乏时间演化模型；WAM 通过视频生成先验注入物理动态，但迭代去噪是速度瓶颈。MotionWAM 的关键洞察是——不需要完整去噪，中间激活就足以指导动作，这一观察将 WAM 从桌面带到了人形。

## 8. 精讀建議 (Reading Guide)

- **值得精讀原文的人**：
  1. 研究 WAM/世界模型+策略耦合的研究者——中间特征提取是论文最有迁移价值的技术点
  2. 从事人形机器人实时全身控制的工程师——三阶段训练框架和统一动作空间设计可直接参考
  3. 评估 Cosmos 生态迁移可行性的研究者——了解预训练权重如何适配到机器人任务

- **建議章節路徑**：先讀 §3.2（Model Architecture，理解双 DiT + 中间特征机制）→ 再看 §4.2-4.4（实验结果与消融）→ 可跳 §2（Related Work，已有本文 §7 对比表）

- **不值得精讀的理由**：如果你不做人形机器人控制、已熟悉 WAM 的基本范式（如 Cosmos Policy）、或只关心桌面机械臂操作，读摘要和本文 §7 对比表即可。

---
[← Back to Theory](./README.md)
