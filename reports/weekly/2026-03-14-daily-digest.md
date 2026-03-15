# VLA Daily Digest — 2026-03-14

> **扫描类型**: 每日信号摘要（vla-daily-digest 调度任务）
> **扫描窗口**: 2026-03-13 → 2026-03-14
> **Belief Graph 版本**: v3 (2026-03-14 paper-scan 基线: B8 65%)

---

## Executive Summary

今日综合 paper scan 和新发现信号，核心关注：

1. **[已处理] MoDE-VLA** (2603.08122): B8 60%→65%，触觉不可替代的最强量化消融。详见 `2026-03-14-paper-scan.md`。
2. **[新发现] AR-VLA** (2603.10126): **首个具有长时记忆的自回归 Action Expert**——SIMPLER 61.5%（超 π0.5 的 51%、CogACT 的 52%），真机 89%，轨迹平滑度优于 FM。直接挑战 **B5（FM 主导）** 并强力支持 **B7（Action Expert 解耦）**。需要三视角辩论。
3. **[已处理] PhysiFlow / Mean-Flow One-Step**: B5/B6 收敛证据，无置信度变更。
4. **产业信号**: 2 天内人形机器人领域累计融资超 $3B（Mind Robotics $2B + Sunday Robotics $1.15B），Figure Helix 02 家庭场景自主演示。

**Belief Graph 变更**: B8 60%→65%（paper scan 已执行）。AR-VLA 对 B5 构成挑战但未达调整阈值（见辩论）。B7 +1 支持证据。

---

## 1. ΔI 信号筛选

### 通道 1（主流信念 B0-B9）+ 通道 2（逆共识 C1-C3）

| # | 信号 | ΔI | 冲击 | 处置 |
|---|------|-----|------|------|
| 1 | MoDE-VLA: 力觉+触觉融合VLA, 量化消融 (2603.08122) | **⚡高** | B8↑, Phase 3 +1 | ✅ paper scan 已辩论 |
| 2 | **AR-VLA: 自回归Action Expert, SIMPLER 61.5%, 真机89% (2603.10126)** | **⚡高** | B5⚠️挑战, B7支持 | **三视角辩论** |
| 3 | PhysiFlow: 三脑humanoid VLA, latent FM 50Hz (2603.05410) | 🔧中 | B6支持, B5支持 | ✅ paper scan 已处理 |
| 4 | Mean-Flow One-Step VLA: 单步FM, 83.9x faster (2603.01469) | 🔧中 | B5支持, B9相关 | ✅ paper scan 已处理 |
| 5 | OmniStream: 统一streaming视觉backbone+VLA适配 (2603.12265) | 📖低 | 工程/基础设施 | 记录 |
| 6 | 产业: Mind Robotics $2B A轮 + Sunday Robotics $1.15B B轮 | 📊产业 | 宏观 | 简评 |
| 7 | 产业: Figure Helix 02 家庭场景自主演示 | 📊产业 | B6 确认性 | 记录 |

---

## 2. Adversarial Triad: AR-VLA — 自回归 Action Expert 挑战 FM 主导叙事

**核心事实**: AR-VLA (ETH Zurich / Luc Van Gool 团队) 提出独立的自回归 Action Expert：不同于 FM/Diffusion 每次观测重置时间上下文，AR-VLA 通过长时记忆维护动作历史。关键创新：Dynamic Temporal Re-anchoring (DTR) 机制——用 RoPE 位置编码处理"慢 VLM (~70ms) + 快动作 (~29ms)"的异步频率不匹配，数学上消除感知陈旧性。

**量化结果**:
- **SIMPLER (BridgeV2泛化)**: 61.5% → 超 π0.5 (51.0%), CogACT (52.1%), π0-Fast (49.0%)
- **真机 WidowX**: 89% 平均成功率，cup-on-plate 和 lobster 任务 100%
- **Specialist (ALOHA)**: Cube-Script 97.33% vs ACT 86% vs Diffusion Policy 33.33%
- **轨迹平滑度**: 平均 jerk 7.89 vs OpenVLA 10.13 vs Flow-Matching 9.39 (×10² rad/s³)
- **延迟**: 有效 46.25ms/action（VLM 69.56ms + Action Expert 28.86ms）

### 🔴 Bull: AR-VLA 证明"FM 是唯一正确的 action head"是错的——自回归+记忆可能是更优解

三个核心论点：

1. **SIMPLER 61.5% 打败 π0 系列**——这不是在自己的小 benchmark 上表演，而是在 FM 系列的主场（BridgeV2 → SIMPLER 泛化评测）上赢了 π0.5 和 π0-Fast。如果 FM 是"唯一正确答案"，为什么一个 AR 方案在 FM 的地盘上赢了 10+ 个百分点？

2. **轨迹平滑度优于 FM**——B5 的核心论据之一是"ODE 无随机抖动"让 FM 适合高频闭环。但 AR-VLA 的 jerk 7.89 低于 FM 的 9.39。如果 AR 方案不仅速度够快（29ms/action）还比 FM 更平滑，FM 在平滑度上的叙事优势就不存在了。

3. **记忆是被忽视的维度**——当前所有 FM/Diffusion action head 都是 reactive 的（每次观测重置），无法利用动作历史。AR-VLA 在 PushT2 和 Stack3 等需要记忆的任务上大幅领先。这意味着在**长时序任务**中，FM 的 reactive 架构可能是结构性短板。B5 的致命实验问的是"FAST-based VLA能否达到π0.6级别"——AR-VLA 不是 FAST，但它证明了 non-FM 方案能打败 π0 系列。

**对 B5 的冲击**: 不是"FM 错了"，而是"FM 不是唯一正确答案"——B5 说的是">60% 新论文用 FM"，但如果 AR 方案在性能上持续打赢 FM，adoption 趋势可能翻转。

### 🔵 Bear: SIMPLER 61.5% 是 sim 数据——真机对比才有意义，而且 AR 方案的 scaling 存疑

直接反驳 Bull 的三点：

1. **SIMPLER 是仿真评测，且 baseline 可能不公平**。AR-VLA 用 BridgeV2 训练在 SIMPLER 上测，但 π0.5 和 π0-Fast 的 SIMPLER 数据可能不是最优配置（这些模型是多任务大模型，不是为 SIMPLER 专门优化的）。真机 WidowX 89% 更有意义，但 WidowX 是单臂简单任务——和 π0.6 的真机双臂长时序不是一个量级。

2. **Jerk 差异不大且任务太简单**。FM jerk 9.39 vs AR 7.89——差距 ~16%。在 WidowX 桌面 pick-and-place 上的平滑度差异，不能推广到 humanoid 全身 30+ DoF 控制。在高维动作空间中，FM 的 ODE 流场理论优势可能更显著。PhysiFlow 刚在 humanoid 上用 FM 达到 50Hz——AR 方案在 humanoid 上的表现完全是空白。

3. **记忆优势真实但有代价**。AR-VLA 的长时记忆确实在 PushT2/Stack3 上赢了，但这引入了 history 管理的工程复杂度和潜在的 distribution shift 风险（长序列中错误累积 through 历史记忆）。ablation 显示 history mask rate 0.0 → 成功率 0%，说明系统对历史依赖极度敏感——如果真机部署中历史被污染（如传感器故障），系统可能整体崩溃。FM 的 reactive 模式反而更鲁棒。

4. **B5 的核心不是"FM 最强"而是"FM 被最多人采用"**。B5 说的是 ">60% 新论文采用 FM"——这是 adoption 趋势判断，不是 performance 判断。一个 ETH 团队的优秀工作不会改变整个领域的 adoption 趋势。FM 的生态优势（π0 开源、训练基础设施、社区惯性）远大于单篇论文的影响。

### 🟢 Arbiter: B5 不调整，B7 +1 证据，但 AR-VLA 标记为 B5 的重要监测信号

综合判断：

- **B5 保持 79%（校准后）**。Bear 说得对：B5 是 adoption 趋势判断。AR-VLA 证明了 AR 方案能在性能上竞争，但不改变 FM 的 adoption 惯性。目前 π0 系列、LingBot、UnifolM、DreamZero、PhysiFlow 等都选择 FM——一篇论文不翻转趋势。**但标记为 B5 的重要监测信号**——如果后续 2-3 个独立团队在 humanoid/双臂上复现 AR 方案优势，B5 需要下调。

- **B7 +1 支持证据**。AR-VLA 的核心架构就是 Action Expert 解耦——独立的 AR Action Expert + 异步 VLM backbone。这是继 π0、GR00T N1.6、Samsung DAM-VLA 之后又一个独立团队选择显式解耦的案例。DTR 机制是解耦方案处理异步频率的优雅解法。B7 保持 78%（已充分支撑）。

- **对 C1（架构创新回来）的微弱相关**: AR-VLA 的 DTR 机制是一个有趣的架构创新，但它不是 C1 追踪的"10x 数据效率新架构"。不调整 C1。

- **关键代理指标**:
  1. AR-VLA 在 humanoid/双臂/长时序（>30步）任务上的表现（目前空白）
  2. 其他团队是否采纳 AR action expert 架构
  3. 记忆机制在真机长期部署中的鲁棒性

- **用户行动建议**:
  - AR-VLA 的 DTR 机制（异步 VLM + 快速 action）值得学习——这是所有 VLA 面临的工程问题
  - 如果你在做 action head 选型，AR 方案现在是 FM/Diffusion 之外的第三个值得认真评估的选项
  - **不要因为 AR-VLA 在 SIMPLER 上赢了就放弃 FM**——在 humanoid 和大规模部署中 FM 的验证更充分

---

## 3. 产业信号简评

### $3B+ 融资潮（2 天内）
- **Mind Robotics** ($2B A轮, Rivian 拆分): 汽车制造→机器人的跨界信号，验证资本对通用机器人的下注
- **Sunday Robotics** ($1.15B B轮, 估值): 目标 2026 感恩节前推出家庭机器人，消费级时间线进一步具体化

**对 Belief Graph 的含义**: 资本信号不直接改变技术判断，但加速了 B1（数据飞轮）的实现条件——更多真机部署 = 更多数据飞轮的潜在启动点。

### Figure Helix 02 家庭演示
单一神经模型指导多步家务（整理客廳、捡玩具、擦拭）。这是 B6（分层架构）的确认性产业信号——Figure 的 S2/S1/S0 架构在家庭场景中的端到端演示。不改变置信度。

---

## 4. 低 ΔI 信号记录

| 信号 | ΔI | 备注 |
|------|-----|------|
| OmniStream (2603.12265): 统一 streaming 视觉 backbone | [Δ低] | 多任务视觉基础模型，含 VLA 适配。工程基础设施信号，不改变范式。 |
| Microsoft × Hexagon Robotics 合作 | [Δ低] | 工业 humanoid 部署合作。生态信号。 |

---

## 5. Belief Graph 变更摘要

| 节点 | 变更 | 理由 |
|------|------|------|
| **B8** | **60% → 65%** | MoDE-VLA 量化消融（paper scan 已执行） |
| B5 | 不变 (79%) | AR-VLA 构成挑战但 adoption 趋势未变，标记为重要监测信号 |
| B7 | 不变 (78%) | AR-VLA +1 支持证据（独立 AR action expert + DTR 异步机制） |
| B6 | 不变 (75%) | PhysiFlow + Figure Helix 02 均为收敛确认 |
| B9 | 不变 (63%) | Mean-Flow One-Step 间接相关，未验证 |
| 其余 | 不变 | — |

## 6. Convergence Map 变更摘要

| Phase | 变更 | 新信号 |
|-------|------|--------|
| **Phase 3 (触觉)** | **7→8** | MoDE-VLA (paper scan 已更新) |
| Phase 1 (FM) | 不变 (4/4) | AR-VLA 是 FM 的**竞争信号**而非收敛信号——标记为 Phase 1 的反相变力量 |
| Phase 2 (RL) | 不变 (5/5) | — |
| Phase 4 (WM) | 不变 (9/9) | — |
| Phase 5 (跨形态) | 不变 (7/7) | — |

**Phase 1 反相变更新**: AR-VLA 加入 Phase 1 反相变信号列表（与 FAST tokenizer 并列）。AR 方案首次在 SIMPLER 上打败 FM 系列——如果此模式被更多团队验证，Phase 1 的"FM 成为事实标准"判断需要修正为"FM + AR 双轨并行"。

## 7. 致命实验状态检查

| 节点 | 致命实验 | 截止 | 距今 | 状态 |
|------|---------|------|------|------|
| B0 | ≤1M ep 新架构碾压 10M+ 系统 | 2027-03 | ~12月 | ❌ 未触发 |
| B0 | 纯架构创新 >30% absolute | 2026-12 | ~9月 | ❌ 未触发 |
| B2 | 非RL方法 100+步真机 >80% | 2027-06 | ~15月 | ❌ 未触发 |
| B2 | 10B+ 纯BC暴力压过 compounding error | 2027-03 | ~12月 | ❌ 未触发 |
| B4 | 无团队真机 >1000ep 验证 WM > BC+RL | 2026-12 | ~9月 | ❌ 未触发 |
| **B5** | **>10B FAST-based VLA 达 π0.6 级别** | **2026-12** | **~9月** | ❌ 未触发。**注: AR-VLA 不是 FAST-based 但开辟了 non-FM 的第三条路线** |
| B5 | 推理芯片使 Diffusion 追平 FM | 2027-06 | ~15月 | ❌ 未触发 |
| B8 | 主流VLA论文触觉占比 <15% | 2027-03 | ~12月 | ❌ 未触发 |

**建议**: 为 B5 增加新的致命实验：
> ❌ [截止 2026-12] AR-based action expert 在 >3 个独立团队的 humanoid/双臂任务上超越 FM 方案 → B5 降至 60%

**无致命实验在 30 天内到期。**

---

## 8. 逆共识检查

| 逆共识 | 今日信号 | 变更 |
|--------|---------|------|
| C1 (架构创新回来) | AR-VLA DTR 机制有创新性但非"10x 数据效率"级别 | 无变更 (20%) |
| C2 (WM是死胡同) | 无新信号 | 无变更 (20%) |
| C3 (VLA不需要L) | 无新信号 | 无变更 (24%) |

## 9. 保守偏误自检

- **上次下调日期**: 2026-03-11（B7: 80%→78%, C2: 22%→20%）
- **连续无下调天数**: 3 天 ✅（无警报，阈值 30 天）
- **今日应下调但未下调的信号?**:
  - AR-VLA 对 B5 的挑战是否应下调 B5？→ 否。单篇论文在仿真评测上赢不足以改变 adoption 趋势判断。需等 humanoid 验证 + 多团队采纳。但标记为监测信号。
  - 产业融资潮是否上调 B1？→ 否。资本信号不直接改变技术判断。

## 10. 时间套利窗口状态

| 套利 | 窗口 | 今日变化 |
|------|------|---------|
| #1 触觉×RL | 6-12月 | MoDE-VLA 强化了感知端价值，RL reward 端仍空白 |
| #2 推理延迟被硬件解决 | 6月 | 无变化 |
| #3 Reward specification | 6-9月 | 无变化 |
| #4 视觉编码器控制感知注入 | 6-9月 | 无变化 |

---

## 11. Belief Graph 状态快照

| 节点 | 校准后置信度 | 上次更新 | 今日变化 |
|------|------------|---------|---------|
| B0 数据>架构 | 77% | 03-05 | → |
| B1 数据飞轮 | 77% | 03-05 | → |
| B2 RL后训练 | 81% | 03-05 | → |
| B3 自我改进闭环 | 77% | 03-11 | → |
| B4 World Model | 60% | 03-13 | → |
| B5 FM action head | 79% | 03-05 | → (**AR-VLA 标记为监测信号**) |
| B6 分层架构 | 75% | 03-05 | → |
| B7 Action Expert | 78% | 03-14 | → (证据+1 AR-VLA) |
| B8 触觉 | **65%** | **03-14** | ↑ (MoDE-VLA, paper scan) |
| B9 小模型VLA | 63% | 03-09 | → |
| C1 架构创新回来 | 20% | 03-05 | → |
| C2 WM死胡同 | 20% | 03-11 | → |
| C3 VLA不需要L | 24% | 03-11 | → |

---

## 12. 一句话记忆锚点

- **AR-VLA**: "首个长时记忆AR Action Expert: SIMPLER 61.5%打败π0系列, jerk优于FM, B5第一个真正的竞争信号"
- **MoDE-VLA**: "VLA内力觉-11%、触觉-8%不可替代性消融——B8升至65%"
- **产业**: "2天$3B+融资: Mind Robotics $2B + Sunday Robotics $1.15B, 家庭机器人时间线2026感恩节"

---

## 13. 明日关注

1. **AR-VLA 社区反应**: ETH + Luc Van Gool 团队有影响力，如果获广泛关注可能改变 action head 选型讨论
2. **ICLR 2026 VLA 趋势报告**（Moritz Reuss 博客）: 164 篇 VLA 提交的系统分析，可能包含 FM vs AR adoption 的统计数据
3. **Cosmos Predict 2.5 社区采用速度**: 开源后 48h+ 的反应
4. **B5 致命实验更新**: 考虑是否增加 AR-based 致命实验

---

*配合 CLAUDE.md v3 + BELIEF_GRAPH.md + CONVERGENCE_MAP.md + EPISTEMICS.md 使用。*
*今日 paper scan (2026-03-14-paper-scan.md) 已单独执行，本 digest 整合 paper scan 结果 + AR-VLA 新信号 + 产业情报。*
