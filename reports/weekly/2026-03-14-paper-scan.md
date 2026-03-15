# VLA Paper Scan — 2026-03-14

> **扫描类型**: 定时论文扫描（vla-paper-scanner 调度任务）
> **扫描窗口**: 2026-03-13 → 2026-03-14
> **Belief Graph 版本**: v3 (2026-03-13 基线)

---

## Executive Summary

今日扫描发现 **1 条高 ΔI 信号** 和 **2 条中 ΔI 信号**：

1. **MoDE-VLA** (2603.08122): 首个将力觉+触觉多模态融入预训练 VLA backbone 的系统，含量化消融——力觉-11%、触觉-8%。直接冲击 **B8（触觉从可选→必选）** 和 **B7（Action Expert 解耦）**。Phase 3 收敛计数 +1。
2. **PhysiFlow** (2603.05410): 三脑架构（皮层/基底核/小脑）humanoid 全身 VLA，latent flow matching 50Hz，比 autoregressive 快 126x。支持 **B6（分层架构标准化）** 和 **B5（FM 主导）**。
3. **Mean-Flow One-Step VLA** (2603.01469): 单步 FM 推理，比 Diffusion Policy 快 83.9x。支持 **B5（FM 主导）** 和 **B9（小模型边缘部署）**。

**Belief Graph 变更**: B8 60%→65%（MoDE-VLA 量化消融是触觉必选的强证据）。其余节点无变更。

---

## 1. ΔI 信号筛选

### 通道 1（主流信念 B0-B9）+ 通道 2（逆共识 C1-C3）

| # | 信号 | ΔI | 冲击 | 处置 |
|---|------|-----|------|------|
| 1 | MoDE-VLA: 力觉+触觉融合VLA, contact-rich任务成功率2x (2603.08122) | **⚡高** | B8↑, B7支持, Phase 3 +1 | 三视角辩论 |
| 2 | PhysiFlow: 三脑humanoid VLA, latent FM 50Hz, 74.9% vs 65% (2603.05410) | **🔧中** | B6支持, B5支持 | 快速三视角 |
| 3 | Mean-Flow One-Step VLA: 单步FM推理, 83.9x faster (2603.01469) | **🔧中** | B5支持, B9相关 | 快速三视角 |
| 4 | UltraDexGrasp: 20M frame双手灵巧抓取数据集 (2603.05312) | **📖低** | B0边缘 | 记录 |
| 5 | UniHM: VLM引导统一灵巧手操作 (2603.00732) | **📖低** | — | 记录 |
| 6 | APPLV: VLA→导航规划参数 (2603.08862) | **❌零** | 非操作领域 | 跳过 |
| 7 | Two-Stage Reward Curriculum (2603.05113) | **📖低** | B3边缘 | 记录 |

---

## 2. Adversarial Triad: MoDE-VLA — 首个力觉+触觉多模态 VLA 含量化消融

**核心事实**: MoDE-VLA (上海交大/上海AI Lab) 将力觉和触觉传感器通过 Mixture-of-Dexterous-Experts 架构融入预训练 π₀ VLA backbone。关键设计：残差注入机制（modality-specific heads 产生 contact-aware correction，自由空间运动时校正自然衰减至零）。在4个 contact-rich bimanual 任务上：苹果削皮30%、管重排30%、齿轮装配60%、充电器插拔15%，平均34%。Baseline π₀ 平均15%。还包含 IMCopilot: RL-based teleoperation共享自主系统。

**量化消融关键数据**:
- 去除力觉：平均成功率 **-11%**（插入类任务影响最大：力信号是接触检测的主要线索）
- 去除触觉：平均成功率 **-8%**（触觉提供指尖形变和接触状态，视觉/臂力矩无法替代）

### 🔴 Bull: MoDE-VLA 是触觉从"学术玩具"走向"VLA 标配"的关键转折

这是第一次在 VLA 框架内对力觉和触觉做了**独立的量化消融**——不是"加了触觉好了一点"，而是**精确测量了每种模态的不可替代性**。力觉-11%、触觉-8%，合计约19%的绝对成功率差距。这在 contact-rich 任务上几乎是"有没有"的区别。

更重要的是架构设计——残差注入机制让触觉/力觉作为"correction signal"融入预训练 VLA，不破坏 backbone 的预训练能力。这意味着**任何现有 VLA 都可以用类似方式加装触觉模块**。这不是一个全新架构，而是一个即插即用的扩展模式。

结合 IMCopilot (RL-augmented teleoperation)，它还解决了数据采集瓶颈——高 DoF 灵巧手的遥操作本来极难，RL 共享自主降低了人类操作难度。

**对 B8 的直接冲击**: 这是"触觉在 VLA 中不可替代"的最强量化证据。之前的 TaF-VLA、SuperTac+DOVE 都只展示了"加触觉有帮助"，没有做干净的消融来证明"没有触觉不行"。MoDE-VLA 补上了这个缺口。B8 应该从 60% 升至 65-70%。

### 🔵 Bear: 34%平均成功率暴露了 contact-rich VLA 的残酷现实——触觉只是杯水车薪

让我们面对现实：MoDE-VLA 的平均成功率只有 **34%**。即使是最好的齿轮装配也只有 60%。充电器插拔 15%。这些数字说明 contact-rich bimanual dexterous manipulation **本身就极难**，触觉只是把"完全不行"(15%) 变成了"勉强能做"(34%)。

Bull 说的"-11%力觉 -8%触觉"是在一个 34% 基线上的消融——如果基线成功率是 80%，这些数字才有说服力。在 34% 基线上，这可能只是说明"系统太脆弱，拿掉任何东西都会崩"。

另外，这是在 **4个高度特化的 contact-rich 任务**上做的。VLA 论文的 90%+ 工作是在桌面 pick-and-place 类任务上做的——那些任务**不需要触觉**。MoDE-VLA 证明的是触觉在**极端场景**下有价值，不是在**通用场景**下必需。B8 的命题是"触觉从可选→必选"——MoDE-VLA 只证明了"在某些任务中有帮助"，离"必选"还很远。

Bull 说"任何 VLA 都能即插即用"——但残差注入需要力觉和触觉传感器的硬件支持。当前 99% 的机器人平台没有配备这些传感器。**硬件生态不支持 = 再好的软件方案也推不动**。

### 🟢 Arbiter: B8 小幅上调合理，但"必选"门槛未达到——真正的信号是架构模式

Bear 说得对，34% 基线让消融数据的解读需要谨慎。但 Bull 的核心论点不在于绝对数字，而在于**两个不可替代性的独立证据**：力觉和触觉各自贡献了不同的、视觉/力矩无法替代的信息通道。这是 B8 "信息论天花板"假说的直接验证。

**B8 更新建议**: 60% → 65%（+5%，达到最小更新幅度）。理由：量化消融是 B8 自创建以来最强的直接证据。但不到70%，因为 Bear 的两个论点成立：(1) 只在极端场景验证；(2) 硬件生态瓶颈未消除。

**B7 影响**: 残差注入机制是 Action Expert 解耦思路的变体——模态专用 expert 作为 refinement，不干扰 backbone。这支持 B7（解耦语义与运动），但不改变置信度——已在 Samsung DAM-VLA 中看到类似模式。

**Phase 3 (触觉标准化) 更新**: +1 收敛信号，计数 7→8。MoDE-VLA 是来自上交/上海AI Lab 的独立信号（不引用 PI 系列触觉工作，从灵巧手操作方向独立推导）。独立性: ✅

**用户行动建议**:
- 如果你的研究方向涉及 contact-rich 操作 → MoDE-VLA 的残差注入模式值得复制
- 即使不做触觉 → 记住"modality-specific residual correction"这个架构模式，它对任何新模态融入 VLA 都适用
- **时间套利**: 力觉+触觉作为 RL reward signal 的研究几乎为零（见套利 1）。MoDE-VLA 证明了感知端有价值，但在 RL 闭环中用触觉做 reward 的论文还没出现。这个窗口仍然开放。

---

## 3. Quick Adversarial Triad: PhysiFlow — 三脑 Humanoid VLA

**核心事实**: PhysiFlow (上交 IRMV Lab) 提出三脑架构：(1) 新皮层脑：SigLIP+LoRA → 256维语义 latent；(2) 基底核脑：latent flow matching + Gemma decoder → 50Hz motion sequences；(3) 小脑脑：teacher-student RL + BC → 物理约束执行。Unitree G1 真机验证。总体成功率 74.9% vs LeVERB 65.0%。FM 推理延迟 18.65ms（比 DDPM 快 5.3x，比 autoregressive 快 126x）。

### 🔴 Bull: 三脑=生物分层的忠实映射，74.9%真机成功率是humanoid VLA新SOTA

PhysiFlow 的三脑架构直接映射到人类运动控制的神经解剖学：大脑皮层(语义)→基底核(运动规划)→小脑(执行校准)。这不是随便的工程分层，而是有生物学依据的功能分区。50Hz FM 推理 + 物理感知跟踪器 = **首次在 humanoid 全身控制中实现快速且稳定的 VLA 闭环**。9.9% 绝对提升 over LeVERB，且在复杂导航任务上提升翻倍（31.2%→63.6%）。

### 🔵 Bear: LeVERB 是弱 baseline，真正的挑战在于 vs Figure Helix / GR00T

74.9% vs 65.0% (LeVERB)。LeVERB 是学术 baseline，不是产业 SOTA。真正的对手是 Figure Helix 02 (S2/S1 200Hz/S0 1kHz) 和 GR00T-N1.6。PhysiFlow 没有和这些系统对比。而且"三脑"和 Figure 的三层、GR00T 的双系统本质上是同一个思路——分层架构——只是用不同的神经科学术语包装。这不是新范式，是已知范式的又一个实例。

### 🟢 Arbiter: 支持 B6 但不改变置信度——收敛信号而非突破

PhysiFlow 是 B6（分层架构标准化）的又一个独立收敛信号——来自不同团队，用不同术语（三脑 vs S0/S1/S2 vs 大脑/小脑），但本质相同。这加强了"分层是 humanoid VLA 的默认架构"的判断，但不改变 B6 的 75% 置信度（已经被 Figure/Galaxea/GR00T 充分支撑）。

FM 推理速度数据（126x faster than autoregressive）支持 B5，但也不是新信息。

**不做置信度调整。记录为 B6 收敛证据。**

---

## 4. Quick Adversarial Triad: Mean-Flow One-Step VLA — FM 推理效率的极限

**核心事实**: Mean-Flow One-Step VLA (arXiv 2603.01469) 通过修改 flow matching 训练目标（直接学习区间平均速度），消除噪声引发的多步约束，实现**单步**生成机器人动作。推理速度比 Diffusion Policy 快 83.9x，比 SmolVLA 快 8.7x。真机验证。

### 🔴 Bull: 单步推理是 FM 的"最终形态"——延迟问题彻底解决

如果能在单步生成高质量动作，FM 的推理速度优势就不再是"5-10x over Diffusion"而是"**80-100x**"。这对 B9（小模型边缘部署）意味着：即使用更大的模型，单步推理也可能满足实时要求。可能改变"小模型必须"的等式。

### 🔵 Bear: 单步推理的质量 tradeoff 没有充分验证

83.9x 速度数据印象深刻，但关键问题是：单步生成的动作质量是否在所有任务复杂度下都保持？在长时序、高精度任务中，多步 ODE 求解的优势可能在于"迭代修正"——单步方案可能在简单任务上足够，复杂任务上崩溃。论文没有在高难度 contact-rich 或长时序任务上验证。

### 🟢 Arbiter: 技术路线确认，不改变信念——但记住这个趋势

Mean-Flow One-Step 和之前的 FlowPolicy (consistency flow matching) 是同一个趋势：**FM 正在从"多步ODE"走向"单步/少步"**。这支持 B5（FM 主导），但 B5 已经在 79%（校准后），没有新信息改变置信度。

**对 B9 的间接影响**: 如果大模型也能单步实时推理，"小模型是必需的"这个假设就被削弱。但这需要更多验证。暂不调整 B9。

**不做置信度调整。记录为 B5 演化方向（FM → one-step FM）。**

---

## 5. Low-ΔI 快速记录

| 论文 | 一句话 | 信念相关 |
|------|--------|---------|
| UltraDexGrasp (2603.05312) | 20M frame 双手灵巧抓取数据集，1000物体，合成数据生成 | B0 (数据>架构) 边缘支持 |
| UniHM (2603.00732) | VLM引导统一灵巧手操作框架，vision+language→dexterous | C3 反证据（语言在灵巧操作中有用） |
| Two-Stage Reward Curriculum (2603.05113) | 解耦任务reward和行为reward的两阶段RL课程 | B3 边缘（reward engineering） |

---

## 6. Belief Graph 变更摘要

| 节点 | 变更 | 理由 |
|------|------|------|
| **B8** | **60% → 65%** | MoDE-VLA 量化消融（力觉-11%, 触觉-8%）是触觉不可替代的最强直接证据 |
| B5 | 不变 (79%) | Mean-Flow One-Step 支持趋势但无新信息 |
| B6 | 不变 (75%) | PhysiFlow 是又一收敛实例，非突破 |
| B7 | 不变 (78%) | MoDE-VLA 残差注入支持解耦，但类似模式已见 |
| B9 | 不变 (63%) | One-step FM 间接削弱"小模型必需"，但未验证 |

## 7. Convergence Map 变更摘要

| Phase | 变更 | 新信号 |
|-------|------|--------|
| **Phase 3 (触觉)** | **计数 7→8** | MoDE-VLA: VLA 框架内力觉+触觉融合+量化消融 (独立: ✅ 上交/上海AI Lab) |
| Phase 1 (FM) | 不变 | Mean-Flow One-Step 和 PhysiFlow FM 是 Phase 1 内演化，非新收敛 |

## 8. 致命实验状态检查

- **B8 致命实验**: "2027-03前主流VLA论文中触觉输入占比<15%" → MoDE-VLA 是又一个触觉 VLA 论文，但单独一篇不改变占比统计。继续追踪。
- **B5 致命实验**: "2026-12前出现>10B FAST-based VLA达到π0.6级别" → 无新信号。未触发。
- 其余致命实验：均未触发。

## 9. 逆共识检查

| 逆共识 | 今日信号 | 变更 |
|--------|---------|------|
| C1 (架构创新即将回来) | Mean-Flow One-Step: 推理效率突破但不是"数据效率10x的新架构" | 无变更 (20%) |
| C2 (WM是死胡同) | 无新信号 | 无变更 (20%) |
| C3 (VLA不需要language) | UniHM: VLM引导灵巧手操作 → 语言在复杂操作中仍有用 | 微弱反证据，不调整 (24%) |

## 10. 时间套利更新

**套利 1 (触觉×RL) 状态: 窗口仍然开放，信号更强**

MoDE-VLA 在感知端证明了触觉的不可替代性，但它的 RL 组件 (IMCopilot) 仅用于数据采集辅助，**没有用触觉作为 RL reward signal**。"触觉 reward → RL 精细操作学习"这个交叉研究方向仍然几乎无人探索。窗口估计: 6-12 个月。

---

## 11. 一句话记忆锚点

- **MoDE-VLA**: "首次量化：VLA中力觉-11%、触觉-8%的不可替代性消融"
- **PhysiFlow**: "三脑humanoid VLA = B6分层架构又一收敛实例，FM 126x faster"
- **Mean-Flow One-Step**: "FM走向单步推理，83.9x faster，B5演化方向确认"

---

*扫描完成。下次扫描: 2026-03-15。*
*配合 CLAUDE.md v3 + BELIEF_GRAPH.md + CONVERGENCE_MAP.md 使用。*
