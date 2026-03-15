# VLA Daily Digest — 2026-03-12

> **扫描类型**: 每日信号摘要（vla-daily-digest 调度任务）
> **扫描窗口**: 2026-03-11 → 2026-03-12
> **Belief Graph 版本**: v3 (2026-03-11 收敛扫描后，无变更)

---

## Executive Summary

今日无重大 Belief Graph 更新。新信号以**产业/融资**为主，技术信号为已有方向的确认性证据。核心关注：

1. **Rhoda AI $450M A轮 + FutureVision 平台**：首个以"视频→动作"（WAM 路线）为核心叙事的独角兽级融资，验证产业界对 video-pretrained policy 的信心
2. **Samsung DAM-VLA 开源**：企业实验室 VLA 框架发布，具身智能技术向产业扩散的又一信号
3. **新理论文档入库**：Evo-RL（RL 后训练民主化）、WAM 三路线综述、SimVLA（简单 baseline 的力量）
4. **无致命实验触发，无逆共识升格，无 Belief Graph 节点变更**

---

## 1. ΔI 信号筛选

### 通道 1（主流信念 B0-B9）+ 通道 2（逆共识 C1-C3）

| # | 信号 | ΔI | 通道 | 处置 |
|---|------|-----|------|------|
| 1 | Rhoda AI FutureVision ($450M, $1.7B估值) — 视频→动作模型平台 | **中高** | B4 (WM), Phase 4 | 三视角辩论 |
| 2 | Evo-RL — 低成本机械臂上的开源 RECAP/π*0.6 工程化 | **中** | B2/B3, Phase 2 | 简评 |
| 3 | SimVLA — 0.5B 极简 VLA baseline 在 LIBERO 达 98.6 | **中** | B0, B7 | 简评 |
| 4 | WAM 三路线综述 — 视频预训练 vs VLA 范式分析 | **中** | B4, 范式思考 | 纳入辩论背景 |
| 5 | Samsung DAM-VLA — 企业 VLA 框架开源 | **低** | 产业扩散信号 | 记录 |
| 6 | Qualcomm × Neura — IQ10 处理器合作 | **低-中** | B9, 约束#2 | 记录 |
| 7 | XGSynBot Z1 — 轮式人形工业机器人 | **低** | 产业硬件 | 记录 |
| 8 | AutoResearch (Karpathy) — 单 GPU 自动研究闭环 | **低** | 方法论/元工具 | 记录 |
| 9 | Lightning Grasp — Contact Field 灵巧手抓取合成 | **低** | B8 边缘相关 | 记录 |

---

## 2. Adversarial Triad: Rhoda AI FutureVision — WAM 路线获得独角兽级资本验证

**背景**: Rhoda AI 从隐身模式走出，$450M A轮（$1.7B 估值），核心产品 FutureVision 基于"数百万公开视频训练的直接视频→动作模型"。这不是一个 VLA 公司，而是一个 **WAM (World Action Model) 公司**。

### 🔴 Bull: 这是 WAM 路线获得产业背书的里程碑

$450M + $1.7B 估值意味着顶级投资人相信"视频预训练 → 动作策略"这条路线有真实商业价值。结合本周入库的 WAM 三路线综述文档的分析框架：视频模型天然携带时间连续性、物体运动轨迹、接触与形变、因果顺序——这些正是 VLM 预训练缺失的"物理直觉"。

Cosmos Policy (NVIDIA) 在学术端已展示 video model 直接变 policy 的可行性（LIBERO 98.5%），现在 Rhoda AI 在产业端用 $1.7B 估值投注同一方向。这是 **Phase 4 (World Model → 闭环实用化)** 的强产业验证信号。

B4 应继续上行关注。如果 Rhoda AI 在 Q2-Q3 公布真机部署结果，B4 可能需要再次上调。

### 🔵 Bear: 融资 ≠ 技术验证，WAM 的工程挑战仍在

Bear 直接回应三点：

1. **融资是预期信号，不是验证信号**。2025-2026 具身 AI 赛道整体融资热潮中，$450M 反映的是**资本对叙事的信心**，不是对技术可行性的证明。Mujin、Figure 等公司的高估值也是建立在预期而非已证明的大规模部署之上。

2. **"数百万公开视频"的分布偏移问题**。公开视频 ≠ 机器人第一视角视频。从 YouTube 视频学到的"物理直觉"在迁移到特定机器人形态和工业任务时，可能面临严重的 domain gap。Rhoda AI 尚未公布任何真机部署数据。

3. **WAM 路线的计算成本仍未解决**。DreamZero 14B 的教训——video model 的推理成本远高于纯 action model。Cosmos Policy 的 LIBERO 结果是在简单桌面任务上，不代表复杂工业场景也能 work。

**关键质疑**: Recap 路线（PI）不需要 video model 就实现了自我改进。WAM 需要证明自己在"真机 + 复杂任务 + 合理算力"条件下优于 Recap，否则就是"用更贵的方案做一样的事"。

### 🟢 Arbiter: 不改 B4，但标记 WAM 为独立追踪方向

综合判断：

- **B4 保持 60%**。Rhoda AI 是产业信心信号而非技术验证信号。B4 的临界条件（真机 >1000 episodes + 显著优于纯 BC+RL）仍未满足。需要看到 Rhoda AI 或同类公司的**真机部署对比数据**才能调整。
- **新增追踪**: WAM 作为一条与"VLM→VLA"平行的范式路线，值得在 CONVERGENCE_MAP 中追踪。但目前不创建独立 Phase——等有 3+ 独立产业团队采用 WAM 路线再升格。
- **用户行动建议**:
  1. 关注 Rhoda AI Q2-Q3 的技术公开（论文/真机 demo）
  2. WAM 三路线综述文档值得深读——它提出的"VLA 学的是 Pr(a|l,o)，WAM 学的是 Pr(a, o_future|l, o_now)"框架对理解范式分歧很有帮助
  3. 不需要现在 pivot——但应该理解 WAM 路线的核心论点，作为 B4 和 C2 的监测背景

---

## 3. 简评：Evo-RL — RL 后训练的民主化

**ΔI: 中** | 影响节点: B2 (RL 后训练), B3 (自我改进闭环), Phase 2

Evo-RL 把 π*0.6/RECAP 的真机 RL 后训练路线工程化到 SO101 低成本机械臂 + LeRobot 开源栈。核心价值不是"RL 有效"（已知），而是**降低了"第一次跑通真机 RL"的门槛**。

**对 Belief Graph 的含义**: B2 (RL 后训练) 获得确认但不足以上调——Phase 2 的收敛已很强（5/5 独立信号），Evo-RL 是工程化信号而非新证据。但值得注意的是：**RL 后训练从"少数顶级团队的秘密武器"变成"社区可复现工程"的转变正在加速。** 这对 Phase 2 的"早期多数采纳 → 晚期多数"过渡有推动作用。

不更新 B2 置信度。Phase 2 状态保持 68%。

---

## 4. 简评：SimVLA — 简单 baseline 的力量

**ΔI: 中** | 影响节点: B0 (数据>架构), B7 (Action Expert 解耦)

SimVLA 用 0.5B backbone + 感知控制显式解耦 + flow matching action head，在 LIBERO 上达到 98.6（超过 OpenVLA-OFT 97.1、π0.5 96.9），训练显存仅 9.3 GB。

**核心洞察**: SimVLA 系统性证明 **data shuffling、action normalization、LR schedule、action chunk horizon 等"训练配方"对性能的影响大于模块创新**——这直接支持 B0 (数据策略>架构) 的叙事。

**对 B7 (Action Expert) 的含义**: SimVLA 采用"VLM 作感知编码器 + 轻量 transformer action head"的显式解耦——与 B7 一致。但上周 WholeBodyVLA 展示了统一 latent 也可以工作。两者并存说明**解耦的最佳粒度仍在探索中**。

不更新任何节点。SimVLA 是确认性证据，不是方向性变化。

---

## 5. Kill Condition 截止日期检查

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

## 6. 保守偏误自检

- **上次下调日期**: 2026-03-11（B7: 80%→78%, C2: 22%→20%）
- **连续无下调天数**: 1 天 ✅（无警报）
- **今日是否有应下调但未下调的信号**: 否。今日信号均为确认性或产业信号，无明确反方证据。

---

## 7. Belief Graph 状态快照（无变更）

| 节点 | 校准后置信度 | 上次更新 | 今日变化 |
|------|------------|---------|---------|
| B0 数据>架构 | 77% | 03-05 | → |
| B1 数据飞轮 | 77% | 03-05 | → |
| B2 RL后训练 | 81% | 03-05 | → |
| B3 自我改进闭环 | 77% | 03-11 | → |
| B4 World Model | 60% | 03-11 | → |
| B5 FM action head | 79% | 03-05 | → |
| B6 分层架构 | 75% | 03-05 | → |
| B7 Action Expert | 78% | 03-11 | → |
| B8 触觉 | 60% | 03-09 | → |
| B9 小模型VLA | 63% | 03-09 | → |
| C1 架构创新回来 | 20% | 03-05 | → |
| C2 WM死胡同 | 20% | 03-11 | → |
| C3 VLA不需要L | 24% | 03-11 | → |

---

## 8. 时间套利窗口状态

| 套利 | 窗口 | Kill 检查 | 今日变化 |
|------|------|----------|---------|
| #1 触觉×RL | 6-12月 | ❌ 未触发 | 无变化 |
| #2 推理延迟被硬件解决 | 6月 | ❌ 未触发 | Qualcomm×Neura 合作是方向一致的弱信号 |
| #3 Reward specification | 6-9月 | ⚠️ 接近 | 无新 reward model 论文 |
| #4 视觉编码器控制感知注入 | 6-9月 | ❌ 未触发 | 无变化 |

---

## 9. 明日关注

1. **Rhoda AI 技术细节**: FutureVision 平台的架构/数据/真机结果是否有更多公开信息
2. **Samsung DAM-VLA**: 框架技术细节值得快速审阅——"Dynamic Action Model"可能与 B7 (Action Expert) 相关
3. **ICLR 2026 后续**: 会议期间可能出现更多 VLA 论文的讨论和非正式结果
4. **Evo-RL 社区反馈**: LeRobot 社区对 Evo-RL 的复现进展

---

*配合 CLAUDE.md v3 + BELIEF_GRAPH.md + CONVERGENCE_MAP.md + EPISTEMICS.md 使用。*
