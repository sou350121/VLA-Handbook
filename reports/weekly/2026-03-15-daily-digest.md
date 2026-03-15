# VLA Daily Digest — 2026-03-15

> **扫描类型**: 每日信号摘要（vla-daily-digest 调度任务）
> **扫描窗口**: 2026-03-14 → 2026-03-15
> **Belief Graph 版本**: v3 (2026-03-15 双周自省基线: B7 75%, B9 65%)
> **特殊事件**: 今日同步执行了 v3 首次双周自省审计（reflection_2026-03-15.md）

---

## Executive Summary

**低信号日**。无⚡高 ΔI 信号，无三视角辩论触发。主要活动：

1. **双周自省审计已执行**（详见 `biweekly/reflection_2026-03-15.md`）。两处置信度纪律修正：B7 78→75%、B9 63→65%。重大发现：生存者偏误严重（零失败案例记录）、致命实验截止日期已补齐、最小更新幅度 ±5% 规则被频繁违反。
2. **新论文信号**：CGVD (2603.10340) 和 GST-VLA (2603.09079) 均为增量工程贡献，ΔI 低。
3. **生态信号**：LeRobot v0.5.0 发布（Pi0-FAST + Real-Time Chunking for FM + Unitree G1 humanoid），开源工具链同步推进 FM 和 FAST 两条路线。
4. **产业信号**：Tesla Optimus Gen 3 Austin 演示 + 2026 夏季量产宣言。
5. **Meta 信号**：Moritz Reuss ICLR 2026 VLA 博客发布（164 篇 VLA 提交，同比 9→164 爆炸增长），但内容无法访问——**标记为高优先级跟进**。

**Belief Graph 变更**: 无（双周自省的 B7/B9 修正已在今日早些时候执行）。

---

## 1. ΔI 信号筛选

### 通道 1（主流信念 B0-B9）+ 通道 2（逆共识 C1-C3）

| # | 信号 | ΔI | 冲击 | 处置 |
|---|------|-----|------|------|
| 1 | CGVD (2603.10340): 训练免推理框架，VLA 在杂乱场景 77.5% vs 43% baseline | 📖低 | 工程/推理优化 | 记录 |
| 2 | GST-VLA (2603.09079): 3D Gaussian 空间 token + depth-aware CoT | 📖低-中 | 套利4(视觉编码器)相关 | 记录 |
| 3 | LeRobot v0.5.0: Pi0-FAST + RTC(FM) + Unitree G1 humanoid | 🔧中 | B5 生态(FM+FAST并行), B6/B9 工具链 | 简评 |
| 4 | Tesla Optimus Gen 3 量产宣言 (Austin demo) | 📖低 | 产业/宏观 | 记录 |
| 5 | Moritz Reuss ICLR 2026 blog (164 VLA submissions) | 🔧中-⚡高 (无法访问) | B5 (FM adoption 统计?) | **跟进** |

**逆共识通道**：
- C1 (架构创新): GST-VLA 的 3D Gaussian tokenizer 是新表征方式，但不构成"10x 数据效率"级别创新。ΔI = 0。
- C2 (WM 死胡同): 无新信号。ΔI = 0。
- C3 (VLA 不需要 L): 无新信号。ΔI = 0。

---

## 2. 信号详评

### 2.1 CGVD — Concept-Gated Visual Distillation (2603.10340)

**核心**: 训练免、模型无关的推理框架。解决 VLA 在杂乱环境中的"精度-推理缺口"。通过指令解析→目标精炼→Fourier inpainting 去除语义干扰物，保留空间几何。

**结果**: 77.5% vs 43.0% baseline（杂乱操作任务）。

**对 Belief Graph 的影响**: 无。这是推理阶段的工程优化，不改变训练范式或架构判断。与 B0-B9 无直接因果关系。

**价值**: 实用工程技巧。如果做真机部署且场景杂乱，CGVD 的 Fourier inpainting 方法值得借鉴。

### 2.2 GST-VLA — 3D Gaussian Spatial Tokens (2603.09079)

**核心**: 用 128 个各向异性 3D Gaussian 基元替代传统 2D patch token，编码深度和表面法向信息。附带 Depth-Aware Chain-of-Thought 监督（3D 物体定位 → 抓取 affordance → 度量距离 → SE(3) 路径点）。

**对 Belief Graph 的影响**: 与套利 4（视觉编码器控制感知注入）方向一致。SpatialVLA → Spatial Forcing → GST-VLA 形成了"向视觉编码器注入 3D 监督"的持续信号流。但 GST-VLA 来自小团队（Yeungnam University / KAIST），需看真机验证。

**记录为**: 套利 4 的又一支持信号（累计 3+），但不调整任何置信度。

### 2.3 LeRobot v0.5.0 — 开源生态里程碑

**核心新增**:
- **Pi0-FAST 策略**: FAST tokenizer 的自回归 VLA 现已进入主流开源框架
- **Real-Time Chunking (RTC)**: 显著降低 Flow Matching 策略的推理延迟
- **Unitree G1 humanoid 全身控制**: LeRobot 首个 humanoid 集成
- **200+ PRs, 50+ 新贡献者**: 社区健康度指标

**对 Belief Graph 的影响**:
- **B5 (FM 79%)**: 双面信号。RTC 强化 FM 部署能力 → FM 生态优势增强。但 Pi0-FAST 被纳入同一框架 → FAST 的可及性与 FM 同等 → FM 不再是唯一"开箱即用"的选择。净效应中性。
- **B6 (分层 75%)**: Unitree G1 humanoid 支持 → humanoid VLA 工具链成熟中。生态信号，不改变置信度。
- **B9 (小模型 65%)**: RTC 降低推理延迟有助于中等规模模型上机，间接压缩小模型的独占优势。微弱挑战，不调整。

**关键观察**: LeRobot v0.5.0 同时支持 FM (via RTC) 和 FAST (via Pi0-FAST)，反映开源社区的策略是**不选边**。这可能预示 action head 的未来是"工具箱"而非"唯一正确答案"——与 AR-VLA 昨日的信号形成呼应，进一步弱化 B5 的"FM 唯一性"叙事。标记为 B5 长期监测信号。

### 2.4 Tesla Optimus Gen 3

Austin Autonomy Pop-Up 公开展示 + 韩国驾驶测试视频。Musk 宣布 2026 夏季量产。

**对 Belief Graph 的影响**: 无直接技术 VLA 信号。Musk 量产时间线历史上频繁跳票。产业信号记录，不调整。

### 2.5 Moritz Reuss ICLR 2026 VLA Blog ⚠️

**已知信息**: 164 篇 VLA 提交（ICLR 2025 仅 9 篇 → 18x 增长）。涵盖 discrete diffusion VLAs、推理模型、benchmark 趋势（LIBERO/CALVIN/SIMPLER）、前沿 vs 学术差距。

**无法访问**: 博客域名 (mbreuss.github.io) 被网络代理阻断。LinkedIn 和知乎镜像同样不可达。

**为什么重要**: 这是目前最全面的 VLA 领域统计分析。如果包含 FM vs Diffusion vs AR 的 adoption 百分比，将直接冲击 B5（FM 主导叙事）。164 篇论文的统计比任何单篇论文更有分量。

**行动**: **高优先级跟进**。下次 digest 尝试其他渠道获取统计数据。如果数据显示 FM adoption <60%，B5 需要重新评估。

---

## 3. Failure Signals 段落 🔴（新增——执行双周自省建议）

> 自省审计发现系统存在严重生存者偏误（零失败案例记录）。从本 digest 起，每次增加 Failure Signals 段落。

**今日主动搜寻结果**:
- **DreamZero 14B 部署成本**: 无后续信号。14B video diffusion model 的推理成本在真实部署中是否可行仍然未知。标记为 B4 的开放风险。
- **VLAW +39.2% 复现**: 无其他团队报告复现 VLAW 的 co-evolution 结果。标记为 B4 证据的未验证状态。
- **Being-H0.5 30 种 embodiment 泛化**: 无后续真机验证报告。标记为 Phase 5 证据的未验证状态。
- **触觉标准化实际障碍**: MoDE-VLA 和 GenForce 各用不同传感器和数据格式，标准化进展为零。Phase 3 的"信号密集但未收敛"判断仍然准确。

---

## 4. Belief Graph 变更摘要

| 节点 | 变更 | 理由 |
|------|------|------|
| B7 | 78% → **75%** | 双周自省纪律修正（±5% 最小更新规则）。已执行。 |
| B9 | 63% → **65%** | 双周自省纪律修正（±5% 最小更新规则）。已执行。 |
| 其余 | 不变 | 今日无高 ΔI 信号 |

## 5. Convergence Map 变更摘要

| Phase | 变更 | 理由 |
|-------|------|------|
| 全部 | 不变 | 今日无新收敛信号 |

## 6. 致命实验状态

**无致命实验在 30 天内到期。** 最近到期：2026-12（B0、B4、B5 多条实验，~9 个月）。

双周自省已补齐 B1/B3/B6/B7 的致命实验截止日期（均设为 2027-06 或 2027-03）。

## 7. 逆共识检查

| 逆共识 | 今日信号 | 变更 |
|--------|---------|------|
| C1 (架构创新回来) | GST-VLA 微弱相关，非 10x 级别 | 无变更 (20%) |
| C2 (WM是死胡同) | 无新信号 | 无变更 (20%) |
| C3 (VLA不需要L) | 无新信号 | 无变更 (24%) |

## 8. 保守偏误自检

- **上次下调日期**: 2026-03-15（双周自省：B7 78→75%）
- **连续无下调天数**: 0 天 ✅
- **今日应下调但未下调的信号?**: 否。今日无高 ΔI 信号。

## 9. 时间套利窗口状态

| 套利 | 窗口 | 今日变化 |
|------|------|---------|
| #1 触觉×RL | 6-12月 | 无变化 |
| #2 推理延迟被硬件解决 | 6月 | 无变化 |
| #3 Reward specification | 6-9月 | 无变化 |
| #4 视觉编码器控制感知注入 | 6-9月 | GST-VLA 为又一支持信号（3D Gaussian spatial tokens） |

---

## 10. Belief Graph 状态快照 (2026-03-15, post-reflection)

```
Belief Graph (校准后):
  B0 数据>架构       77%  [上次检验 03-05, PI锚定风险]
  B1 数据飞轮       77%  [上次检验 03-05]
  B2 RL后训练       81%  [上次检验 03-05]
  B3 自我改进闭环   77%  [上次检验 03-11]
  B4 World Model    60%  [上次检验 03-13, 独立性待验证]
  B5 FM action head  79%  [上次检验 03-14, AR-VLA监测+PI锚定+LeRobot双轨]
  B6 分层架构       75%  [上次检验 03-05]
  B7 Action Expert  75%  [修正自78%, 上次检验 03-15]
  B8 触觉→必选      65%  [上次检验 03-14]
  B9 小模型VLA      65%  [修正自63%, 上次检验 03-15]

Contrarian Portfolio:
  C1 架构创新回来    20%  [信号饥饿, 建议扩展定义]
  C2 WM是死胡同     20%  [正常削弱, 下限15%]
  C3 VLA不需要L     24%  [活跃, 距升格阈值16pp]

Convergence Map:
  Phase 1 FM         85%  [4/4独立, 反相变: FAST+AR-VLA]
  Phase 2 RL后训练   68%  [5/5独立]
  Phase 3 触觉标准化  35%  [8/8独立]
  Phase 4 World Model 50%  [9/9声称独立, 待验证]
  Phase 5 跨形态     40%  [7/7独立]
```

---

## 11. 一句话记忆锚点

- **CGVD**: "训练免 Fourier inpainting 去杂乱：77.5% vs 43% baseline。工程技巧，不改范式"
- **GST-VLA**: "3D Gaussian 空间 token + depth CoT。套利4(视觉编码器3D注入)又一信号"
- **LeRobot v0.5.0**: "FM(RTC) + FAST(Pi0-FAST) + G1 humanoid 同时支持——开源不选边，B5 唯一性叙事进一步弱化"
- **Moritz Reuss**: "164篇VLA@ICLR2026（9→164 YoY）。无法访问详细统计——高优先级跟进"

---

## 12. 明日关注

1. **Moritz Reuss ICLR 2026 blog**: 寻找其他渠道获取 FM vs AR vs FAST 的 adoption 统计。如果 FM <60% → B5 需重新评估
2. **AR-VLA 社区后续**: 是否有更多团队讨论/复现 AR action expert？
3. **失败案例主动搜寻**: DreamZero 部署成本、VLAW 复现、触觉标准化障碍
4. **Phase 4 引用链验证**: CoWVLA ↔ AtomVLA ↔ Cosmos 独立性确认（双周自省建议 #5）

---

*配合 CLAUDE.md v3 + BELIEF_GRAPH.md + CONVERGENCE_MAP.md + EPISTEMICS.md 使用。*
*今日为低信号日。主要事件为双周自省审计执行及纪律修正。首次增加 Failure Signals 段落。*
