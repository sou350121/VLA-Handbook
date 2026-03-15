# VLA Weekly Convergence Scan — 2026-03-11

> **扫描类型**: 每周收敛扫描（vla-upstream-radar 调度任务）
> **扫描窗口**: 2026-03-05 → 2026-03-11
> **上次收敛扫描**: 2026-03-05 (v3 系统建立)
> **Belief Graph 版本**: v3 (2026-03-11 paper-scan 更新后)

---

## Executive Summary

本周是 v3 系统建立后的**首次正式收敛扫描**。核心发现：

1. **ICLR 2026 VLA 爆发**：164 篇 VLA 提交（去年 9 篇），确认 VLA 进入主流化加速期
2. **三条新信号值得追踪**：WholeBodyVLA (ICLR 2026)、Interactive World Simulator (arXiv 2603.08546)、GateFlow (ICLR 2026)
3. **Phase 4 (World Model) 加速确认**：Interactive World Simulator 在真机验证 WM 生成数据 ≈ 真实数据效果
4. **Phase 1 (FM action head) 出现反信号**：ICLR 2026 趋势显示 Discrete Diffusion VLA 方向活跃，XPENG VLA 2.0 跳过 FM 直接 vision→action
5. **B6 (分层架构) 获得新支持**：WholeBodyVLA + LeVERB 两篇 ICLR 2026 论文验证 latent 分层在 humanoid whole-body 上有效
6. **无致命实验被触发**，无逆共识升格

---

## 1. 新信号识别与独立性验证

### 1.1 WholeBodyVLA (ICLR 2026, OpenDriveLab)

```
来源: arXiv 2512.11047, ICLR 2026 accepted
核心: 统一 latent VLA for humanoid whole-body loco-manipulation
方法: Latent Action Model (LAM) 从 action-free 视频学习 + LMO-RL 全身协调
结果: AgiBot X2 humanoid 上比 baseline +21.3%
信息传播链: 独立于 Figure Helix / GR00T — 用 latent action 而非显式分层
独立性判定: ✅ 独立（不同方法论路线）
```

**Belief Graph 冲击**:
- B6 (分层架构): 支持。LAM 的 latent action 本质是软分层（语义→latent→运动）
- B7 (Action Expert): 部分挑战。WholeBodyVLA 用统一 latent 而非显式解耦
- C3 ("VLA不需要language"): 中性。语言指令仍在使用，但 latent action 绕过了显式语言中间表征

### 1.2 Interactive World Simulator (arXiv 2603.08546, March 2026)

```
来源: arXiv 2603.08546
核心: 用 consistency model 构建交互式世界模型，支持策略训练和评估
方法: Consistency model for image decoding + latent dynamics prediction
结果: 单卡 RTX 4090 上 15 FPS 稳定运行 >10 分钟；WM 生成数据训练的策略
      与相同数量真实数据训练的策略性能相当
信息传播链: 未引用 VLAW/CoWVLA/Cosmos — 独立路线
独立性判定: ✅ 独立
```

**Belief Graph 冲击**:
- B4 (World Model): **强支持**。首次有证据表明 WM 生成数据 ≈ 真实数据效果，且计算成本合理（单卡运行）
- 约束松弛 #1 (数据采集成本): 直接相关——WM 合成数据作为"绕过"路径的证据
- 约束松弛 #3 (WM物理保真度): Consistency model 是否能保持物理一致性？论文声称"physically consistent"但需要更多验证

### 1.3 GateFlow (ICLR 2026)

```
来源: OpenReview qOSy2PX4xS, ICLR 2026
核心: 用 Wasserstein distance 门控抑制 VLA 中的 shortcut learning
方法: 测量 observation features 和 action targets 之间的传输距离，
      低距离（语义理解）增强梯度，高距离（捷径）抑制梯度
信息传播链: 基于 flow matching 框架的改进，引用 π0
独立性判定: ✅ 独立（新方法论贡献，非跟随复现）
```

**Belief Graph 冲击**:
- B5 (FM 主导): 支持。GateFlow 建立在 FM 之上，扩展其能力边界
- 通用贡献：解决了 VLA FM head 的 ELBO-NLL gap 问题，提升 FM 可靠性

### 1.4 ICLR 2026 宏观趋势信号

```
来源: Moritz Reuss 分析 164 篇 VLA 提交
核心发现:
  - VLA 提交从 9 → 164 (ICLR 2025 → 2026)，18x 增长
  - 趋势: Discrete Diffusion VLAs, Reasoning VLAs, RL for VLAs,
    Efficient VLAs, Cross-Action-Space Learning
  - Discrete Diffusion VLA 作为趋势类别出现 ← B5 (FM) 的反信号
独立性: N/A（元分析）
```

**Belief Graph 冲击**:
- B5 (FM 主导): ⚠️ 潜在反信号。Discrete Diffusion VLA 作为 ICLR 2026 趋势意味着学术界未完全收敛到 FM
- 通用: "Reasoning VLAs" 和 "Embodied Chain-of-Thought" 作为趋势 → 与 B7 (Action Expert 解耦) 相关

### 1.5 XPENG VLA 2.0 (产业信号)

```
来源: XPENG AI Day, Q1 2026 量产部署
核心: "Vision-Implicit Token-Action" 路径，去除语言翻译步骤
结果: 首个量产 VLA 系统，大众为首个全球合作伙伴
信息传播链: 产业独立路线（非学术跟随）
独立性判定: ✅ 独立（产业独立开发）
```

**Belief Graph 冲击**:
- C3 ("VLA不需要language"): **强支持**。首个量产系统选择去除 L，产业验证
- B5 (FM 主导): 中性/反信号。XPENG 未使用 FM，而是"隐式 token"方案

---

## 2. Phase Counter 更新

### Phase 1: Action Head → Flow Matching

```
状态: ████████████████████░░ 85% → 保持 85%（无重大变化）
```

**本周变化**: 无新独立 FM 采用信号。

**新反信号**:
- ICLR 2026 "Discrete Diffusion VLAs" 作为趋势类别出现
- XPENG VLA 2.0 选择 implicit token 而非 FM
- GateFlow 虽建立在 FM 上但暴露了 FM 的 shortcut learning 问题

**判断**: FM 在产业前沿（PI 系列）仍主导，但学术界出现分化。85% 不变，但需要在 Q2 重新审视——如果 Discrete Diffusion VLA 在真机验证超越 FM，需要下调。

**独立收敛计数**: 仍为 4/4（无新增）

### Phase 2: 训练范式 → RL 后训练

```
状态: ██████████████░░░░░░░░ 65% → 68%（ICLR 趋势确认）
```

**本周变化**: ICLR 2026 将 "RL for VLAs" 列为趋势类别 + WholeBodyVLA 使用 LMO-RL

**新信号**:
| # | 收敛信号 | 日期 | 独立? |
|---|---------|------|------|
| 5 | WholeBodyVLA LMO-RL (ICLR 2026) | 2026-03 | ✅ |
| 独立收敛计数 | **5/5** |

### Phase 3: 触觉传感 → 标准化

```
状态: ████████░░░░░░░░░░░░░ 35% → 保持 35%
```

**本周变化**: 无新独立信号。GenForce (上次扫描已计入) 仍是最新。

**注意**: 多轴触觉传感器论文（humanoid 工业应用）出现，但属于硬件而非标准化方向。

### Phase 4: World Model → 闭环实用化

```
状态: ████████░░░░░░░░░░░░░ 40% → 45%（Interactive World Simulator 关键信号）
```

**本周变化**: Interactive World Simulator 提供了 WM 闭环实用化的**最强证据之一**——WM 生成数据训练的策略与真实数据策略"性能相当"，且单卡可运行。

**新信号**:
| # | 收敛信号 | 日期 | 独立? |
|---|---------|------|------|
| 7 | Interactive World Simulator: consistency model, WM 数据 ≈ 真实数据 | 2026-03 | ✅ |
| 独立收敛计数 | **7/7** |

**重要**: 这是第一篇在真机多种任务（rigid/deformable/piles）上验证 WM 生成数据可替代真实数据的工作。之前的 VLAW +39.2% 是"WM 辅助提升"，这篇是"WM 可替代"——质的区别。

### Phase 5: 跨形态泛化 → 统一基座

```
状态: ████████░░░░░░░░░░░░░ 40% → 42%
```

**本周变化**: WholeBodyVLA 展示了 latent action 在 humanoid 上的泛化能力，但仅限单一 embodiment (AgiBot X2)。

**新信号**:
| # | 收敛信号 | 日期 | 独立? |
|---|---------|------|------|
| 7 | WholeBodyVLA: latent action from action-free video, humanoid | 2026-03 | ✅ |
| 独立收敛计数 | **7/7** |

---

## 3. 收敛交叉检测

### 3.1 现有交叉更新

**Phase 2 × Phase 4 ("在想象中自我改进")**:
- Interactive World Simulator 直接验证了这个交叉的核心假设：WM 生成的数据可以训练有效策略
- **交叉点紧迫度从"理论可能"提升至"工程验证中"**
- 阻碍因素更新：物理保真度 → consistency model 声称已解决，但接触密集任务仍未验证

**Phase 3 × Phase 2 ("从接触失败中学习")**:
- 无新信号。仍然是被低估的交叉。

### 3.2 新观察到的交叉

```
新交叉: Phase 2 (RL后训练) × Phase 5 (跨形态泛化)
    ↓                              ↓
  "从错误中学习"          ×    "统一动作空间"
    ↓                              ↓
    ←──── 交叉点 ────→
    "跨形态 RL：在一个 embodiment 上的 RL 经验
     迁移到另一个 embodiment"

WholeBodyVLA 的 latent action 空间 + LMO-RL 暗示：
如果 latent action 空间是跨 embodiment 统一的，
RL 经验可以在 latent space 中跨形态迁移——
这比 Phase 5 当前追踪的"数据混训"路线更根本。
```

---

## 4. 约束松弛状态检查

| 排名 | 约束 | 本周变化 | 松弛方向 |
|------|------|---------|---------|
| **1** | 真机数据采集成本 | ⬆️ Interactive World Simulator 证明 WM 数据可替代真实数据 | **绕过路径 A (WM) 获得关键验证** |
| **2** | 实时推理延迟 | → 无新信号。4-bit 量化 (LiteVLA-Edge) 仍是最新 | 硬件路线进展待观察 |
| **3** | WM 物理保真度 | ⬆️ Consistency model 声称 physically consistent | 但接触密集验证缺失 |
| **4** | 跨形态动作空间 | ⬆️ WholeBodyVLA latent action 提供新路线 | 但仅单 embodiment 验证 |
| **5** | 触觉传感标准化 | → 无新信号 | 等待更多标准化进展 |

**关键洞察**: 约束 #1 (数据采集成本) 本周获得了最大松弛信号。Interactive World Simulator 的"WM 数据 ≈ 真实数据"如果可复现，将是**整个 VLA 领域的拐点之一**——数据不再是稀缺资源。

---

## 5. 时间套利窗口检查

### 套利 1: "触觉 × RL = 精细操作的下一跳"
```
窗口状态: 仍然开放（6-12 个月）
Kill 条件检查: ❌ 未触发（无触觉 reward 信号噪声过大的报告）
本周变化: 无
```

### 套利 2: "约束松弛 #2 即将被硬件解决"
```
窗口状态: 仍然开放（6 个月）
Kill 条件检查: ❌ 未触发（芯片未延期报告）
本周变化: 无新芯片发布消息
⚠️ 注意: LiteVLA-Edge 的 4-bit 量化路线可能让"小模型+量化"先于"大模型+芯片"到达
         如果 4-bit 3B 模型在复杂任务上够用，套利2 的前提（需要大模型）被削弱
```

### 套利 3: "自我改进闭环的 reward specification 是被忽视的瓶颈"
```
窗口状态: 6-9 个月（上次更新已缩短）
Kill 条件检查: ⚠️ 接近触发。Robometer 在已知任务上表现好，但 OOD 可靠性未验证。
              如果 Q2 出现 Robometer/RoboReward 在 OOD 场景的正面真机验证 → 窗口关闭
本周变化: 无新 reward model 论文
```

### 套利 4: "视觉编码器的控制感知注入"
```
窗口状态: 6-9 个月
Kill 条件检查: ❌ 未触发
本周变化: RoboVLMs (Nature Machine Intelligence) 的 >600 实验确认 backbone 选择很重要，
          但未明确支持或反对 Spatial Forcing 路线
```

---

## 6. Adversarial Triad: Interactive World Simulator 的含义

这是本周最高 ΔI 的新信号，需要三视角辩论。

### 🔴 Bull: WM 数据 ≈ 真实数据 → 约束 #1 正在被绕过

Interactive World Simulator 的核心结果——在 rigid objects, deformable objects, object piles 上，WM 生成的数据训练的策略与**相同数量**真实数据策略性能相当——这是 B4 最需要的证据。之前的 WM 工作（VLAW +39.2%）只证明了"WM 辅助有用"，这篇证明了"WM 可替代"。

更重要的是**计算成本合理**：单卡 RTX 4090, 15 FPS, 稳定 >10 分钟。这不是 DreamZero 14B 那种计算成本让部署不可行的情况。Consistency model 的推理效率（比扩散模型快 10-50x）是关键差异。

如果这个结果可推广，**VLA 领域的数据瓶颈可能在 2026 下半年被大幅缓解**。B4 应从 55% 上调至 60%+。

### 🔵 Bear: "性能相当"的实验条件需要严格审查

Bear 直接质疑"WM 数据 ≈ 真实数据"的强结论：

1. **"相同数量"是多少？** 如果实验用了 100 episodes 对比，那说明 WM 在小数据 regime 有效——但 VLA 需要的是 10K-100K 级别的 scale。WM 数据在大规模下是否仍然 ≈ 真实数据？
2. **任务复杂度天花板**。Rigid objects + deformable + piles 是 manipulation 的"中等难度"。接触判断精细度要求高的任务（如插座插入、螺丝拧紧）——WM 物理保真度够吗？Consistency model 的物理一致性声明缺乏在这些 corner case 的验证。
3. **长期自我改进中的误差累积**。WM 数据在单次训练中有效不等于在**迭代自我改进**（B3 闭环）中有效。WM 的微小物理误差在多轮迭代中可能被放大——这是 Phase 2 × Phase 4 交叉的核心风险。
4. **Recap 的竞争**。PI 的 Recap 路线不需要 WM 就实现了自我改进。Interactive World Simulator 需要证明比 Recap 更高效（更少真机交互），否则 WM 路线的额外复杂度不值得。

### 🟢 Arbiter: B4 上调 +5%，但 Phase 4 的临界条件仍未满足

综合 Bull 和 Bear：

- **B4 从 55% → 60%**（校准后仍为 60%，因 <80% 不折扣）。理由：Interactive World Simulator 提供了"WM 可替代真实数据"的首个真机证据，但实验规模和任务复杂度限制了结论强度。
- **Phase 4 状态从 40% → 45%**。新增独立信号（7/7），但临界条件（真机 >1000 episodes + 显著优于纯 BC+RL）**仍未满足**——需要看到大规模真机验证。
- **用户行动建议**：
  1. 如果你有访问 RTX 4090 级别算力的能力，值得复现 Interactive World Simulator 在你关注的具体任务上的结果
  2. 关键实验：WM 生成 1000 episodes vs 真实 1000 episodes，在你的 benchmark 上对比
  3. 如果 WM 方案的数据效率提升 <3x → 暂时 stay with Recap 路线

---

## 7. Adversarial Triad: WholeBodyVLA 对分层架构的含义

### 🔴 Bull: Latent action 是"软分层"的优雅解——比显式 S0/S1/S2 更灵活

WholeBodyVLA 用 Latent Action Model 从 action-free 视频学习统一 latent 表征，然后解码为 dual-arm + locomotion。这不是传统的显式分层（S2→S1→S0），而是"学出来的软分层"——模型自己决定如何在 latent space 中组织语义和运动信息。

AgiBot X2 上 +21.3% 的提升说明这种方法有效。更重要的是，它能从**无动作标注的视频**中学习——这直接缓解了 humanoid 数据稀缺的问题。

B6 (分层架构) 应该更新为包含"软分层"路线：不只是 Figure Helix 的显式 S0/S1/S2，还有 WholeBodyVLA 的 latent 分层。

### 🔵 Bear: +21.3% 在单一 embodiment 上不足以确认路线

Bear 指出：
1. **单一 embodiment (AgiBot X2)**。不知道 latent action 是否能迁移到其他 humanoid（Figure、NVIDIA、XPENG）
2. **Baseline 可能太弱**。"比 baseline +21.3%"的 baseline 是什么？如果是简单 BC baseline，这个提升不令人意外
3. **与 Helix 02 的直接对比缺失**。Helix 02 在真机 humanoid 上已展示了成熟的 S0/S1/S2 方案——WholeBodyVLA 需要在相同或更难的任务上超越 Helix 02 才能证明"软分层 > 显式分层"

### 🟢 Arbiter: B6 保持 75%，但添加"软分层"作为竞争路线

- B6 不变（75%），但支持证据中增加 WholeBodyVLA 作为"latent 软分层"路线
- B7 (Action Expert) 受到轻微挑战——WholeBodyVLA 不使用显式 action expert
- **实际建议**：关注 ICLR 2026 后续是否有更多 latent action 方案出现。如果 6 个月内出现 3+ 独立团队采用 latent action for humanoid → 考虑调整 B6 的定义

---

## 8. Belief Graph 更新摘要

| 节点 | 上次 | 本次 | 变化 | 原因 |
|------|------|------|------|------|
| B0 (数据>架构) | 77% | 77% | → | 无新信号 |
| B1 (数据飞轮) | 77% | 77% | → | 无新信号 |
| B2 (RL后训练) | 81% | 81% | → | WholeBodyVLA LMO-RL 支持但不足以再上调 |
| B3 (自我改进闭环) | 77% | 77% | → | 无新 reward model 信号超越上次扫描 |
| B4 (World Model) | 55% | 60% | **+5%** | Interactive World Simulator 真机验证 WM 数据可替代 |
| B5 (FM action head) | 79% | 79% | → | GateFlow 支持，但 Discrete Diffusion 趋势是反信号，互相抵消 |
| B6 (分层架构) | 75% | 75% | → | WholeBodyVLA 支持但路线不同(软分层 vs 显式分层) |
| B7 (Action Expert) | 80% | 78% | **-2%** | WholeBodyVLA 展示不需要显式 action expert 的路线 |
| B8 (触觉) | 60% | 60% | → | 无新信号 |
| B9 (小模型VLA) | 63% | 63% | → | 无新信号 |
| C1 (架构创新回来) | 20% | 20% | → | 无新信号 |
| C2 (WM死胡同) | 22% | 20% | **-2%** | Interactive World Simulator 进一步削弱 C2 |
| C3 (VLA不需要L) | 22% | 24% | **+2%** | XPENG VLA 2.0 量产部署去除 L 的产业信号 |

**传播检查**:
- B4 ↑ → B3 检查：B4 加速器升至 60% 间接支持 B3，但 B3 的独立条件（B1+B2）未变，保持 77%
- B7 ↓ → B6 检查：B6 的支持不依赖于"必须显式 action expert"，只要分层存在即可，保持 75%
- C2 ↓ → B4 一致（WM 方向获得支持，"WM 死胡同"逆共识减弱）

---

## 9. 保守偏误自检

**上次降低置信度的日期**: 2026-03-11（B7 从 80% → 78%，C2 从 22% → 20%）
**连续无下调天数**: 0 天 ✅（本次有下调）

**引用来源时间分布检查**:
- 本周引用: Interactive World Simulator (2026-03), WholeBodyVLA (2025-12/ICLR 2026), GateFlow (ICLR 2026), XPENG VLA 2.0 (2025-11 发布/2026-Q1 部署)
- 非近期引用: 无
- ⚠️ 所有引用均来自最近 4 个月 → 需要在下次扫描中主动寻找**失败案例**和**反信号**

---

## 10. 下周关注重点

1. **ICLR 2026 会议 (如在进行中)**：关注 VLA 相关 oral/spotlight 论文的反响和讨论
2. **Interactive World Simulator 复现**：是否有独立团队验证 WM 数据替代效果
3. **Discrete Diffusion VLA**：是否有真机验证超越 FM 的结果
4. **XPENG VLA 2.0 量产反馈**：Q1 部署效果如何
5. **Robometer/RoboReward OOD 验证**：套利 3 的 kill condition 监测

---

## 变更日志

| 日期 | 变更 |
|------|------|
| 2026-03-11 | 首次正式收敛扫描。Phase 2: 65%→68% (+WholeBodyVLA). Phase 4: 40%→45% (+Interactive World Simulator). B4: 55%→60%. B7: 80%→78%. C2: 22%→20%. C3: 22%→24%. 新增 Phase 2×5 交叉观察。 |

---

*配合 CLAUDE.md v3 + BELIEF_GRAPH.md + CONVERGENCE_MAP.md + EPISTEMICS.md 使用。*
