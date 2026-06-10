# MotionWAM：面向实时人形移动操作的世界动作模型 (MotionWAM: Towards Foundation World Action Models for Real-Time Humanoid Loco-Manipulation)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-06-10
>
> **论文**: MotionWAM: Towards Foundation World Action Models for Real-Time Humanoid Loco-Manipulation
> **链接**: https://arxiv.org/abs/2606.09215
> **核心定位**: 首次将世界动作模型（WAM）从桌面机械臂扩展到实时人形机器人全身移动操作，通过中间去噪特征读取替代迭代去噪，实现 7× 推理加速。

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | 用 Video DiT 的中间去噪特征直接驱动 Motion DiT，一步生成全身动作，无需迭代去噪未来视频帧，在人形机器人上实现实时 WAM 闭环控制 |
| 适合精读 | 做人形机器人全身控制、世界模型驱动策略、实时推理优化的研究者/工程师 |
| 可以跳过 | 只关心桌面机械臂操作或纯 VLA 语义推理的人 |
| 落地可行性 | 中（需要 Cosmos-Predict2 预训练权重 + 大规模 egocentric 视频数据 + Unitree G1 硬件） |
| 主要风险 | 仅在单一机器人平台验证；未见新物体泛化测试；单目摄像头丢失即失效 |

💡 **X-Ray 开场**

现有 WAM 需要先迭代去噪生成未来视频帧，再从视频帧反推动作——这个流程在桌面机械臂上可以跑，但放到需要高频平衡控制的人形机器人上就太慢了（Cosmos Policy 仅 0.7 Hz）。MotionWAM 的洞察是：不需要等视频完全去噪，在纯噪声阶段（τ ≈ 1）读取 Video DiT 的中间激活，这些特征已经编码了"场景将要往哪走"的信息。一步读取，直接驱动动作生成，推理速度提升 7 倍，同时成功率从 43.9% 提升到 76.1%。

📍 **研究全景时间线**

```
2023  Diffusion Policy (视觉→动作直接映射)
  → 2023  RT-2 (VLA 语义先验注入)
  → 2024  ACT / OpenVLA (VLA 生态成型)
  → 2025  GR00T-N1 / π₀.₅ (通用人形 VLA 基座)
  → 2025  Cosmos Policy (世界模型→迭代去噪视频→动作, 0.7 Hz)
  → 2025  DiT4DiT (Video DiT + Motion DiT 联合建模, 桌面机械臂)
  → [2026-06 MotionWAM] ← 当前位置: 中间特征读取 + 全身统一动作空间 + 实时闭环
  ← 局限: 仅 Unitree G1, 无新物体泛化, 单目摄像头
```

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 维度 | 传统分层策略 (Upper-Lower Split) | Cosmos Policy | MotionWAM |
|------|------|------|------|
| **动作空间** | 上肢关节目标 + 下肢基座速度/高度/朝向 | 离散动作 token | 统一全身 motion latent（64 维离散 + 连续末端执行器） |
| **视觉先验** | 静态图像特征（VLM） | 迭代去噪未来视频帧 | Video DiT 中间去噪特征（单步读取） |
| **推理模式** | 直接映射 | 多步迭代去噪 | 单步 forward pass |
| **推理频率** | 通常 >10 Hz | 0.7 Hz (A100) | 4.9 Hz (A100) |
| **下肢角色** | 仅维持平衡 | 受限于动作空间 | 任务驱动（踢球、踩踏板） |
| **训练阶段** | 通常单阶段 | 单阶段微调 | 三阶段渐进式 |
| **硬件平台** | 多样 | 桌面机械臂 | Unitree G1 人形机器人 |

### 1.2 关键机制 (Key Mechanism)

MotionWAM 的核心设计围绕两个突破：

**突破 1：中间特征读取替代迭代去噪**

传统 WAM 需要完整走完扩散去噪流程（多步）才能从视频帧提取动作。MotionWAM 在 Video DiT 的纯噪声端（τ_f ≈ 1）安装 forward hook，仅一次 forward pass 就读取 velocity network 的隐藏状态 h_t。这些激活编码了"场景将要往哪走"的语义——不需要看到清晰的未来帧，只需知道趋势。

**突破 2：统一全身动作空间**

传统分层策略把身体切成两半：上肢拿精细关节目标，下肢只拿粗粒度基座命令。MotionWAM 用 SONIC 全身控制器共享 latent 作为瓶颈，通过有限标量量化（FSQ）得到 64 维离散 token k_t ∈ {-1, -15/16, ..., 1}^64，覆盖行走、躯干运动、高度调节、脚部交互。连续部分 m_t^cont 负责灵巧手/夹爪。

⚡ **Eureka Moment**：不需要等视频完全去噪——在纯噪声阶段读取 Video DiT 的中间激活，这些特征已经编码了足够的动力学先验来驱动动作生成，一步到位。

### 1.3 信息流/架构图 (Flow / Diagram)

```
┌─────────────────────────────────────────────────────────────┐
│                    MotionWAM 推理流程                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  当前帧 o_t ──→ VAE Encode ──→ z_t^0                        │
│       \                                                      │
│        \  (conditioning)                                     │
│         \                                                    │
│          ▼                                                   │
│  ┌──────────────────────────────────────────────┐           │
│  │           Video DiT (Cosmos-Predict2)         │           │
│  │  Input: z_t^0 (干净) + z_{t+1}^{τ_f} (噪声)   │           │
│  │  Language: Cosmos-Reason1 embedding           │           │
│  │                                              │           │
│  │  → Single forward pass @ τ_f ≈ 1             │           │
│  │  → Forward hook 读取隐藏状态                   │           │
│  └────────────────────┬─────────────────────────┘           │
│                       │                                     │
│                  h_t^{τ_f}                                  │
│            (中间去噪特征)                                     │
│                       │                                     │
│                       ▼                                     │
│  ┌──────────────────────────────────────────────┐           │
│  │           Motion DiT                          │           │
│  │  Input: h_t^{τ_f} (cross-attn)                │           │
│  │        proprioceptive state p_t               │           │
│  │        noisy motion latent m_t^{τ_a}          │           │
│  │  Output: velocity field → motion latent m_t   │           │
│  └────────────────────┬─────────────────────────┘           │
│                       │                                     │
│              m_t = (m_t^cont, k_t)                          │
│                       │                                     │
│                       ▼                                     │
│  ┌──────────────────────────────────────────────┐           │
│  │              SONIC Decoder                    │           │
│  │  round(k_t) → 离散 token → 全身关节指令 a_t   │           │
│  └──────────────────────────────────────────────┘           │
│                       │                                     │
│                       ▼                                     │
│              Unitree G1 执行                                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 2. 数学核心 (Math Core)

### 2.1 问题形式化

MotionWAM 遵循"预测视频动力学 → 反演动作"范式：

```
o_{t+1} ~ p_v(· | o_t, l)          // 视频世界模型预测未来帧分布
m_t   ~ p_a(· | o_t, p_t, H(o_{t+1}^{τ_v}))  // 动作模型从中间特征反演
```

其中 o_t 是自我中心观测，l 是语言目标，p_t 是本体感知状态，H(·) 从生成过程中提取隐藏状态，τ_v → 0 时 o_{t+1}^{τ_v} 收敛到干净未来帧。

### 2.2 中间特征读取

关键创新：不等待完整去噪，在纯噪声端读取：

```
h_t^{τ_f} = H[v_θ^video](z_{t+1}^{τ_f}, τ_f | z_t^0, l)
z_{t+1}^{τ_f} |_{τ_f→1} ~ N(0, I)
```

τ_f ≈ 1 时，z_{t+1}^{τ_f} 几乎是纯高斯噪声，Video DiT 仅一步 forward pass 就输出隐藏状态——这些激活编码了场景演化的方向性信息。

### 2.3 Flow Matching 目标

**Video DiT 损失**（学习未来帧的 velocity field）：

```
L_video = E_{τ_v, z_{t+1}^0, ε_v} [ || v_θ^video(z_{t+1}^{τ_v}, τ_v | z_t^0, l) - (ε_v - z_{t+1}^0) ||^2 ]
```

**Motion DiT 损失**（学习 motion latent 的 velocity field）：

```
L_motion = E_{τ_a, m_t^0, ε_m} [ || v_φ^motion(m_t^{τ_a}, τ_a | h_t^{τ_f}, p_t, e) - (ε_m - m_t^0) ||^2 ]
```

**Stage 2/3 联合损失**：

```
L = L_motion + L_video
```

保留 L_video 作为"表示正则化器"——防止动作信号覆盖已学到的动力学先验。

### 2.4 动作解码

```
m_t = (m_t^cont, k_t~)
  → Flow Matching 预测 → m̂_t = (m̂_t^cont, k̂_t~)
  → k̂_t = round(k̂_t~)  // 最近邻取整恢复离散索引
  → SONIC 解码 → a_t (全身关节指令)
```

> 符号与本文保持一致：o = 观测, p = 本体感知, m = motion latent, z = VAE latent, h = 隐藏状态, v = velocity network, τ = flow timestep, ε = 噪声样本, l = 语言目标, e = embodiment 索引

## 3. 带数字走一遍：玩具例子 (Worked Example)

假设一个简化场景：人形机器人需要从站立状态走到 2 米外的桌子前，拿起杯子。

**Step 1 — 当前观测**：
- o_t: 头载摄像头拍摄的当前帧（机器人视角，看到桌子和杯子在前方）
- p_t: 当前关节角度 [0.1, -0.05, 0.2, ...]（64 维），当前速度 0 m/s

**Step 2 — Video DiT 单步推理**：
- 输入 z_t^0（干净当前帧 VAE latent）+ z_{t+1}^{τ_f=0.99}（接近纯噪声的未来帧 latent）
- 经过 Cosmos-Predict2 的 DiT trunk，在某个 transformer block 的 forward hook 读取 h_t
- 这一步耗时约 100-200ms（单步 forward pass）
- h_t 编码的信息：「前方有桌子，杯子在桌上，机器人需要向前移动」

**Step 3 — Motion DiT 推理**：
- 输入 h_t（cross-attention 注入视觉动力学先验）+ p_t（本体状态）+ 噪声 motion latent
- Flow matching 去噪，输出 m_t = (m_t^cont, k_t~)
- 假设 k_t~ = 0.3 → round(0.3) = 0 → SONIC token index 0（对应"向前走"的全身轨迹）
- m_t^cont = [0.0, 0.0]（夹爪未激活，保持闭合）

**Step 4 — SONIC 解码**：
- Token 0 解码为全身关节指令：腿部关节周期性摆动，躯干前倾 5°，手臂自然摆动
- 输出 a_t 发送到 Unitree G1 关节控制器

**Step 5 — 循环**：
- 每 200ms（≈5 Hz）重复 Step 1-4
- 当摄像头检测到桌子足够近时，h_t 的语义从"向前走"变为"伸手"
- Motion DiT 输出新的 k_t（对应"伸手+夹爪打开"的 token），m_t^cont 变为 [1.0, 1.0]（夹爪打开）

**关键数字**：
- 单次完整推理：~200ms（Video DiT ~150ms + Motion DiT ~50ms）
- 闭环频率：4.9 Hz（实测）
- 对比 Cosmos Policy：需要 5-10 步迭代去噪 → ~1400ms → 0.7 Hz

## 4. 工程视角 (Engineering View)

| 工程维度 | MotionWAM | 备注 |
|----------|-----------|------|
| **推理硬件** | NVIDIA RTX 4090（训练）/ A100（部署测试） | 策略以 WebSocket 服务运行在 workstation 上，机载控制器通过 WebSocket 查询 |
| **推理延迟** | ~200ms/步（单步 forward pass） | 对比 Cosmos Policy 的迭代去噪 ~1400ms |
| **闭环频率** | 4.9 Hz（A100） | 人形机器人平衡控制最低要求约 2-5 Hz，MotionWAM 刚好达标 |
| **模型规模** | Video DiT: Cosmos-Predict2.5-2B (2B 参数) + Motion DiT: 未明确 | 与 Qwen3DiT 基线参数匹配，总参数量级 ~2-3B |
| **内存占用** | 未明确报告 | 预估 Video DiT 2B 参数 ~4-8GB (FP16) + Motion DiT ~1-2GB |
| **训练数据量** | Stage 1: ~2,136 小时 egocentric 视频；Stage 3: 9 任务 × 200 episodes | Stage 1 数据量大但无需动作标注（便宜）；Stage 3 数据量小但需 teleoperation |
| **部署架构** | Off-board 推理 + WebSocket 通信 | 策略不运行在机载计算单元上，存在通信延迟（未量化） |

**工程含义**：
- 单步读取策略使 WAM 首次达到人形机器人实时控制门槛，但 4.9 Hz 余量不大——增加任务复杂度或更换更大模型可能突破实时边界
- Off-board 推理架构意味着通信延迟是隐藏变量，若迁移到机载部署（如 Jetson Orin），需要重新评估实时性
- 三阶段训练的成本分布：Stage 1 最贵（2,136 小时视频预训练），Stage 3 最便宜（每任务 200 episodes teleoperation）

## 5. 数据与评测 (Data & Eval)

### 5.1 训练数据

| 阶段 | 数据类型 | 规模 | 标注需求 |
|------|----------|------|----------|
| Stage 1 | Egocentric 人类视频 + 人形机器人视频 | ~2,136 小时 | 无需动作标注（仅视频帧） |
| Stage 2 | 异构 Unitree G1 数据（不同末端执行器 + 不同动作标注格式） | 未明确 | 需要动作标注 |
| Stage 3 | Teleoperated 全身演示（VR 采集 → SMPL → Unitree G1 retarget） | 9 任务 × 200 episodes | 需要全身关节轨迹 |

### 5.2 评测任务（9 个真实机器人任务）

| 任务 | 核心能力 | MotionWAM 成功率 | 最强基线 (GR00T-N1.7) | 差距 |
|------|----------|------|------|------|
| Lift Basket | 腰部控制 | 未逐项披露 | — | — |
| Retrieve Item | 全身协调 | — | — | +40% |
| Load Cart | 身体-手协调 | — | — | +40% |
| Toss Garbage | 高度调节 + 投掷 | — | — | — |
| Kick Soccer | 脚部任务驱动交互 | — | — | +40% |
| Wipe Board | 躯干运动 + 手部操作 | — | — | +45% |
| Do Laundry | 全身协调 | — | — | +30% |
| （其余 2 个任务） | — | — | — | — |

**总体成功率**：MotionWAM 76.1% vs GR00T-N1.7 43.9%（+32.2% 绝对提升），20 trials/任务。

### 5.3 消融实验（三阶段训练必要性）

| 变体 | Lift Basket | Retrieve Item | Load Cart | Toss Garbage | Kick Soccer | 平均 |
|------|------|------|------|------|------|------|
| 完整三阶段 | 最高 | 最高 | 最高 | 最高 | 最高 | 基准 |
| 去掉 Stage 1 | ↓11% | — | — | — | — | -11% |
| 去掉 Stage 2 | ↓28% | — | — | — | — | -28% |

Stage 2 的贡献（28%）远大于 Stage 1（11%），说明跨 embodiment 动作 grounding 比 egocentric 视频先验更关键。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

### 6.1 能力

| 能力 | 说明 |
|------|------|
| 实时全身控制 | 4.9 Hz 闭环频率，满足人形机器人平衡控制最低要求 |
| 任务驱动脚部交互 | 踢球、踩踏板——分层策略无法实现（下肢仅被分配基座速度） |
| 全身协调运动 | 蹲姿行走、身体-手协同、高度调节 + 操作复合任务 |
| 单摄像头端到端 | 仅需头载 Intel RealSense D435i RGB，无需外部动捕 |
| 多 embodiment 泛化 | Stage 2 跨不同末端执行器/动作格式训练，共享 Motion DiT trunk |

### 6.2 失败模式

| 失败场景 | 原因 | 论文来源 |
|----------|------|----------|
| 目标物体离开摄像头视场 | 单目摄像头无冗余视觉，丢失视觉 grounding 后策略停滞 | §6 Limitations |
| 头摄像头视角偏离训练分布 | 摄像头漂移导致输入分布外推，隐藏状态 h_t 语义失真 | §6 Limitations |
| 新物体泛化未知 | 训练/测试物体视觉相似，未报告严格 OOD 物体成功率 | §6 Limitations |
| 仅 Unitree G1 验证 | 三阶段范式未在其他人形平台上验证迁移性 | §6 Limitations |
| 4.9 Hz 余量有限 | 复杂任务或模型增大可能突破实时边界 | §4.4 Table 3 |

### 6.3 隐含假设 (Hidden Assumptions)

1. **中间去噪特征足够丰富**：假设 τ_f ≈ 1 时的隐藏状态编码了足够的动力学信息。论文通过实验验证了这一点，但未系统探索不同 τ_f 值的影响——可能存在更优的中间 timestep。

2. **SONIC latent 足以表征所有任务**：64 维 FSQ token（32^64 种组合）被假设为足够表达 9 种任务所需的全身运动。但如果任务集扩展到更精细的操作（如灵巧手操作），这个 bottleneck 可能成为限制。

3. **Off-board 推理可接受**：策略运行在 workstation 上通过 WebSocket 通信，假设通信延迟可忽略。若部署到机载计算单元，实时性需要重新评估。

4. **Egocentric 视频动力学可迁移**：Stage 1 用人类 egocentric 视频预训练，假设人类视角的视觉动力学与人形机器人视角足够相似。消融实验显示去掉 Stage 1 仅降 11%，说明这个假设部分成立但不是决定性的。

5. **Flow matching 适合离散动作**：将 SONIC token 索引作为连续标量 k_t~ 在 flow matching 框架中回归，推理时 round() 取整。这个设计巧妙但隐含假设：连续空间中的回归能准确映射到正确的离散 token。

## 7. 与相关工作对比 (Comparison)

| 方法 | 视觉先验 | 动作空间 | 推理模式 | 平台 | 实时性 |
|------|----------|----------|----------|------|--------|
| Diffusion Policy | 静态图像 | 机械臂关节 | 多步扩散 | 桌面机械臂 | ✓ |
| ACT | 静态图像 (VLM) | 机械臂关节 | 单步 transformer | 桌面机械臂 | ✓ |
| GR00T-N1.7 | 静态图像 (VLM) | 全身离散 token | 单步 | 人形机器人 | ✓ |
| π₀.₅ | 静态图像 (VLM) | 全身 | 单步 | 人形机器人 | ✓ |
| Cosmos Policy | 迭代去噪视频 | 离散动作 | 多步迭代去噪 | 桌面机械臂 | ✗ (0.7 Hz) |
| DiT4DiT | Video DiT 联合建模 | 机械臂 | 单步 | 桌面机械臂 | ✓ |
| **MotionWAM** | **Video DiT 中间特征** | **全身统一 motion latent** | **单步** | **人形机器人** | **✓ (4.9 Hz)** |

**面试 Tip**：当被问及"MotionWAM 与 Cosmos Policy 的核心区别"时，回答：「Cosmos Policy 需要迭代去噪生成完整未来视频帧再反推动作（多步，慢），MotionWAM 在纯噪声端一步读取 Video DiT 的中间激活作为动力学先验（单步，快），两者精度相当但推理速度差 7 倍。本质是『是否需要看到清晰未来帧』的取舍。」

## 8. 精讀建議 (Reading Guide)

- **值得精讀原文的人**：
  1. 做人形机器人全身控制的研究者——统一动作空间 + 三阶段训练范式可直接借鉴
  2. 探索世界模型驱动策略的工程师——中间特征读取是一个通用的加速技巧
  3. 评估从桌面操作迁移到人形机器人可行性的团队——三阶段数据策略有参考价值

- **建議章節路徑**：
  - 先读 §3.2（Model Architecture）——理解 dual-DiT + 中间特征读取的核心机制
  - 再看 §3.3（Training Recipe）——三阶段训练的设计哲学和数据策略
  - 然后 §4.2-4.4（实验）——验证核心 claim 的数据
  - 可跳过 §2（Related Work）——如果已熟悉 WAM 和 VLA 生态

- **不值得精讀的理由**：
  - 如果只做桌面机械臂操作，这篇的全身控制架构对你过重
  - 如果已熟悉 DiT4DiT 的 dual-DiT 框架，MotionWAM 的核心创新（中间特征读取）是一个相对小的改动——读 §3.2 即可
  - 如果不做人形机器人，§4 的实验设置和任务设计参考价值有限

---
[← Back to Theory](./README.md)
