# VLA Daily Digest — 2026-03-13

> **扫描类型**: 每日信号摘要（vla-daily-digest 调度任务）
> **扫描窗口**: 2026-03-12 → 2026-03-13
> **Belief Graph 版本**: v3 (2026-03-13 更新后)

---

## Executive Summary

今日发现 **1 条高 ΔI 信号**（PlayWorld: 首个完全自主 WM 训练闭环）和 **1 条中-高 ΔI 信号**（NVIDIA 开源 Cosmos Predict 2.5 + GR00T N1.6 LeRobot 集成）。核心关注：

1. **PlayWorld (Princeton, arXiv 2603.09030)**: 首个完全自主的"自主 play → WM → RL in WM → 真机部署"pipeline。真机成功率 +65%，WM 预测与真实 Pearson=0.88。**这是 Phase 2×4 交叉（"在想象中自我改进"）的首次端到端验证。** 但规模有限（8 小时数据，20 trials/policy）。
2. **NVIDIA Cosmos Predict 2.5 开源 + GR00T N1.6 × LeRobot 集成**: WM 和 VLA 工具的民主化信号。Cosmos Predict 2.5 将 WM 用于机器人策略训练的门槛从"自研"降至"开箱即用"。
3. **Samsung DAM-VLA**: 双头 diffusion (arm+gripper) + VLM 路由。SIMPLER 71% 最高。Action Expert 解耦方向的企业级验证。
4. **Phase 4 (World Model) 收敛计数 7→9，状态 45%→50%**。B4 置信度保持 60%（PlayWorld 规模不足以触发上调）。

---

## 1. ΔI 信号筛选

### 通道 1（主流信念 B0-B9）+ 通道 2（逆共识 C1-C3）

| # | 信号 | ΔI | 通道 | 处置 |
|---|------|-----|------|------|
| 1 | PlayWorld: 自主play→WM→RL, 真机+65% (Princeton) | **高** | B4 (WM), Phase 4, Phase 2×4 交叉 | 三视角辩论 |
| 2 | NVIDIA Cosmos Predict 2.5 + GR00T N1.6 × LeRobot | **中-高** | B4, B6, Phase 4 生态 | 简评 + Phase 4 计数 |
| 3 | Samsung DAM-VLA: 双头diffusion + VLM路由 | **中** | B7 (Action Expert) | 简评 |
| 4 | CGVD: training-free VLA去噪, 77.5% vs 43% (arXiv 2603.10340) | **低** | 工程/推理技巧 | 记录 |
| 5 | MetaWorld-X: VLM-orchestrated experts for humanoid (arXiv 2603.08572) | **低-中** | B6, B7 | 记录 |
| 6 | AGPS: agent-guided policy search替代human-in-loop (arXiv 2602.11978v2) | **低-中** | B3, Phase 2 | 记录 |
| 7 | BayesianVLA: Information Collapse诊断+修复 (arXiv 2601.15197) | **低** | C3 边缘相关 | 记录 |

---

## 2. Adversarial Triad: PlayWorld — 首个"自主play→WM→RL→真机"端到端闭环

**背景**: PlayWorld (Princeton, Majumdar lab) 实现了一个完全无需人类演示的 pipeline: VLM 提议多样化任务 → VLA 执行 → 收集自主 play 数据 → 训练 SVD-based video WM → 在 WM 中用 DSRL 做 RL → 部署到真机。在 DROID 平台上，RL-in-WM 训练的策略在真机上成功率提升 65%。WM 预测与真实成功率的 Pearson 相关达 0.8766。

### 🔴 Bull: 这是 Phase 2×4 交叉的首次端到端验证——"在想象中自我改进"不再是假说

CONVERGENCE_MAP 中最危险的交叉是 Phase 2 (RL后训练) × Phase 4 (WM)——"在想象中自我改进"。PlayWorld 是**首个完整验证这条路线的工作**：

1. **完全自主数据收集**：VLM 提议任务 + VLA 执行 + 自动安全机制。8 小时无人值守连续收集。这意味着数据飞轮的冷启动问题（B1 的反方叙事）可以被绕过——不需要人类演示来启动飞轮。
2. **WM 预测质量惊人**：Pearson 0.8766 意味着 WM 对不同策略好坏的排序与现实高度一致。这直接挑战了 C2（"WM 是死胡同"）——如果 WM 能可靠地预测策略表现，它就不是死胡同。
3. **RL in WM 有效且高效**：DSRL 冻结基础 diffusion model，只学习轻量噪声策略。65% 真机改进是实打实的。这与 VLAW 的 +39.2% 和 Interactive World Simulator 的"WM 数据 ≈ 真实数据"共同构成 WM-RL 路线的第三个独立验证。
4. **对 B4 的含义**：Phase 4 收敛从 7 个独立信号爆增到 9 个，且 PlayWorld 是最接近"端到端自主闭环"的信号。B4 已从 03-05 的 50% 升到 60%，趋势明确。

### 🔵 Bear: 小规模实验 + 已知局限 = 过早庆祝

Bear 直接反驳四点：

1. **规模极小**：每个策略仅 20 次真机试验。8 小时数据收集。B4 的 kill condition 要求 >1000 episodes。PlayWorld 连 100 episodes 都没到。65% 改进在 20 trials 的统计显著性存疑——置信区间可能很宽。
2. **任务简单**：Bowl/carrot/polar bear，rectangular block，towel——全是桌面简单物体操作。DROID 平台是固定单臂。这与 B4 关心的"真机长时序"和"复杂任务"相去甚远。
3. **幻觉未消除**：作者自己承认"does not eliminate hallucinations entirely"和"prediction errors accumulate over long horizons"。WM-RL 的核心风险正是幻觉在 RL 循环中被放大——PlayWorld 在简单任务上掩盖了这个问题。
4. **Recap 路线仍然更简单**：π0.6 Recap 不需要 WM 就实现了真机自我改进，且已经在更大规模（2x 吞吐量提升）上验证。PlayWorld 用更复杂的方案（VLM + VLA + SVD WM + DSRL）达到的效果，在简单任务上可能不如 Recap 的直接 RL。**WM 的"想象力"价值需要在 Recap 路线搞不定的任务上验证——比如稀有失败模式或危险操作。**

### 🟢 Arbiter: Phase 4 计数上调，B4 保持，关键代理指标明确

综合判断：

- **Phase 4 收敛计数 7→9**（+PlayWorld, +Cosmos Predict 2.5）。状态从 45% 升至 50%。WM 方向的证据密度在快速增加。
- **B4 保持 60%**。PlayWorld 方向正确但规模不足以满足 kill condition（>1000 episodes + 显著优于 BC+RL）。需要看到同类工作扩展到 >100 tasks、>1000 episodes、长时序任务。
- **Phase 2×4 交叉从"假说"升级为"首次验证"**。PlayWorld 证明"在想象中自我改进"的端到端 pipeline 可以 work。但从"实验室 demo work"到"比 Recap 更好"还有很大距离。
- **关键代理指标**：
  1. PlayWorld 或同类工作在 >10 个不同任务类型上复现 WM-RL 优势
  2. 与 Recap 路线的 head-to-head 对比（同任务、同规模）
  3. 长时序任务（>30 步）中 WM 幻觉累积是否导致 RL 失效
- **用户行动建议**：
  1. PlayWorld 的 VLM-proposed autonomous play pipeline 值得关注——这是数据收集的新范式（无需人类参与）
  2. 不要因为 Phase 4 计数快速增长就过度投入 WM——Recap 仍是验证更充分的路线
  3. 观察 NVIDIA Cosmos Predict 2.5 在社区中的采用速度——如果快速普及，WM 的工程门槛问题将不再成立

---

## 3. 简评: NVIDIA Cosmos Predict 2.5 + GR00T N1.6 × LeRobot

**ΔI: 中-高** | 影响节点: B4 (WM), B6 (分层), Phase 4, Phase 5

NVIDIA 发布三件套：Cosmos Predict 2.5（开源可定制 WM，用于机器人合成数据生成 + 策略评估）、Cosmos Reason 2（开源推理 VLM）、GR00T N1.6（开源 humanoid VLA）。同时与 Hugging Face 合作将 GR00T N 和 Isaac Lab-Arena 集成进 LeRobot 框架。

**对 Belief Graph 的含义**：
- **B4 (WM)**：Cosmos Predict 2.5 是 WM 的**民主化信号**——从 NVIDIA 自研工具变成社区可用基础设施。这不改变 WM 的技术可行性判断，但大幅降低工程门槛，加速 Phase 4 的"方法论竞争 → 早期共识"过渡。计入 Phase 4 收敛（独立信号 #9）。
- **B6 (分层)**：GR00T N1.6 是确认性信号（已在 B6 证据中），开源 + LeRobot 集成 = 分层 humanoid VLA 的标准化在推进。
- **生态含义**：NVIDIA 2M 开发者 + Hugging Face 13M 开发者 = VLA/WM 工具的可及性跃升。这对 Phase 2 (RL 民主化, 与 Evo-RL 同向) 和 Phase 5 (跨形态) 都是正向信号。

不更新节点置信度。Phase 4 计数已更新。

---

## 4. 简评: Samsung DAM-VLA

**ΔI: 中** | 影响节点: B7 (Action Expert 解耦)

Samsung Research 发布 DAM-VLA：双头 diffusion model（arm action head + gripper action head），由 VLM 高层语义驱动动态路由。Google Robot 83%/81%，WidowX SIMPLER 71%（最高）。

**对 B7 的含义**：DAM-VLA 的"VLM → 动态路由 → 专用 action head"架构是 B7（Action Expert 解耦）的企业级验证。VLM 负责语义理解，diffusion action heads 负责运动生成——显式解耦且性能强。与 WholeBodyVLA 的"统一 latent"形成对照。B7 增加一条支持证据，置信度保持 78%。

---

## 5. 低 ΔI 信号记录

| 信号 | 来源 | ΔI | 备注 |
|------|------|-----|------|
| CGVD: 训练-free VLA去噪 | arXiv 2603.10340 | [Δ低] | 77.5% vs 43% 在杂乱环境。训练-free + 模型无关 = 工程价值，但不改变范式。推理时的"视觉蒸馏"思路可能对部署有用。 |
| MetaWorld-X: VLM编排专家层级 | arXiv 2603.08572 | [Δ低-中] | 人形机器人的 VLM → expert routing → 运动。与 B6 (分层) 和 B7 (解耦) 一致。Princeton + 张建伟团队。增量确认。 |
| AGPS: Agent替代human-in-loop RL | arXiv 2602.11978v2 | [Δ低-中] | VLM 作为"语义 WM"提供 RL 引导。与 PlayWorld 的 VLM-proposed play 方向一致。修订版 Mar 7。 |
| BayesianVLA: Information Collapse | arXiv 2601.15197 | [Δ低] | 发现 VLA 训练中语言信号被"坍缩"的问题。与 C3 (VLA不需要L) 微弱相关——如果语言在训练中自然被忽略，可能说明语言贡献确实有限。但解法是加强语言条件化而非去除。 |

---

## 6. Kill Condition 截止日期检查

| 节点 | 致命实验 | 截止日期 | 距今 | 状态 |
|------|---------|---------|------|------|
| B0 | ≤1M episodes 新架构碾压 10M+ 系统 | 2027-03 | ~12 月 | ❌ 未触发 |
| B0 | 纯架构创新提升 >30% absolute | 2026-12 | ~9 月 | ❌ 未触发 |
| B2 | 非 RL 方法在 100+ 步真机 >80% 成功率 | 2027-06 | ~15 月 | ❌ 未触发 |
| B2 | 10B+ 纯 BC 暴力压过 compounding error | 2027-03 | ~12 月 | ❌ 未触发 |
| B4 | 无团队真机 >1000 ep 验证 WM 优于 BC+RL | 2026-12 | ~9 月 | ❌ 未触发 |
| B5 | >10B FAST-based VLA 达 π0.6 级别 | 2026-12 | ~9 月 | ❌ 未触发 |
| B5 | 推理芯片使 Diffusion 追平 FM | 2027-06 | ~15 月 | ❌ 未触发 |
| B8 | 主流 VLA 论文触觉占比 <15% | 2027-03 | ~12 月 | ❌ 未触发 |

**无致命实验在 30 天内到期。**

---

## 7. 保守偏误自检

- **上次下调日期**: 2026-03-11（B7: 80%→78%, C2: 22%→20%）
- **连续无下调天数**: 2 天 ✅（无警报）
- **今日是否有应下调但未下调的信号**:
  - Bear 对 PlayWorld 的规模质疑是否应影响 C2？→ 否。PlayWorld 的 65% 真机改进（即使小规模）是对 C2 的证据，不是支持 C2 的证据。C2 保持 20%。
  - BayesianVLA 的 Information Collapse 是否支持 C3？→ 极微弱。作者的解法是强化语言条件化，而非去除语言。不调整。

---

## 8. Belief Graph 状态快照

| 节点 | 校准后置信度 | 上次更新 | 今日变化 |
|------|------------|---------|---------|
| B0 数据>架构 | 77% | 03-05 | → |
| B1 数据飞轮 | 77% | 03-05 | → |
| B2 RL后训练 | 81% | 03-05 | → |
| B3 自我改进闭环 | 77% | 03-11 | → |
| B4 World Model | 60% | 03-13 | → (证据+2, 置信度不变, Phase 4 count 7→9) |
| B5 FM action head | 79% | 03-05 | → |
| B6 分层架构 | 75% | 03-05 | → |
| B7 Action Expert | 78% | 03-13 | → (证据+1 Samsung DAM-VLA, 置信度不变) |
| B8 触觉 | 60% | 03-09 | → |
| B9 小模型VLA | 63% | 03-09 | → |
| C1 架构创新回来 | 20% | 03-05 | → |
| C2 WM死胡同 | 20% | 03-11 | → |
| C3 VLA不需要L | 24% | 03-11 | → |

---

## 9. 相变计数器快照

| Phase | 上次 (03-12) | 本次 (03-13) | 变化 | 说明 |
|-------|-------------|-------------|------|------|
| Phase 1 (FM→action head) | 4/4 | 4/4 | 无 | — |
| Phase 2 (RL后训练) | 5/5 | 5/5 | 无 | AGPS 是 RL 引导优化，非新范式 |
| Phase 3 (触觉标准化) | 7/7 | 7/7 | 无 | — |
| Phase 4 (World Model) | 7/7 | **9/9** | **+2** | PlayWorld + Cosmos Predict 2.5 |
| Phase 5 (跨形态) | 7/7 | 7/7 | 无 | — |

**Phase 4 是当前加速最快的相变方向。** 从 03-05 的 4 个信号到 03-13 的 9 个信号（8 天 +5），独立收敛密度惊人。

---

## 10. 时间套利窗口状态

| 套利 | 窗口 | Kill 检查 | 今日变化 |
|------|------|----------|---------|
| #1 触觉×RL | 6-12月 | ❌ 未触发 | 无变化 |
| #2 推理延迟被硬件解决 | 6月 | ❌ 未触发 | 无变化 |
| #3 Reward specification | 6-9月 | ⚠️ 接近 | PlayWorld 用 progress-based reward（简单但有效），AGPS 用 agent 生成引导 — reward 自动化的替代路线在增加 |
| #4 视觉编码器控制感知注入 | 6-9月 | ❌ 未触发 | 无变化 |

---

## 11. 关键洞察：Phase 4 加速的含义

Phase 4 (WM→闭环实用化) 在 8 天内从 4 个独立信号增长到 9 个，这是所有 Phase 中增速最快的。值得注意的模式：

- **路线多样化**：pixel WM (VLAW, DreamZero) → latent WM (CoWVLA, AtomVLA) → consistency model WM (Interactive World Simulator) → SVD WM (PlayWorld) → 商业 WM (Cosmos, Runway GWM-1)
- **从辅助到替代**：早期信号是"WM 辅助训练"，现在出现"WM 替代真实数据"（Interactive WS）和"WM 中做 RL"（PlayWorld/VLAW）
- **工程门槛在急速下降**：Cosmos Predict 2.5 开源 + Runway GWM SDK + PlayWorld 的自主 play pipeline

**但 Bear 的核心质疑仍成立**：所有信号都在相对简单的任务上。真机长时序 + 接触密集 + >1000 episodes 的验证仍缺。如果 WM 在"硬任务"上的幻觉问题暴露出来，Phase 4 的叙事可能快速反转。

**对用户的建议**：Phase 4 值得持续追踪，但不要因为信号密度高就立刻 all-in WM 方向。Recap (纯 RL，无 WM) 仍是验证更充分、工程更简单的路线。WM 的价值需要在"Recap 搞不定的场景"中证明——比如稀有失败模式模拟、危险操作预演。

---

## 12. 明日关注

1. **PlayWorld 社区反应**：Princeton Majumdar lab 的影响力可能让这篇快速获得关注
2. **NVIDIA Cosmos Predict 2.5 + LeRobot 社区采用**：开源后多快有人复现？
3. **ICLR 2026 持续讨论**：VLA track 的非正式结果和讨论
4. **Samsung DAM-VLA 技术深度**：双头 diffusion + VLM 路由的具体架构值得审阅

---

*配合 CLAUDE.md v3 + BELIEF_GRAPH.md + CONVERGENCE_MAP.md + EPISTEMICS.md 使用。*
