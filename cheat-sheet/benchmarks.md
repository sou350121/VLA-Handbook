# Benchmark 地图（30+ benchmark · 7 大类）+ 可信度警告

> 2026-04-21 深度更新 · 🚨 **报单一 benchmark 分数要慎重**——所有 LIBERO 系列都有记忆化风险

---

## 🚨 首先读这里：6 条 benchmark 可信度警告

### ⚠️ 警告 1 · LIBERO 高分 = 记忆化（不是泛化）

📎 **[LIBERO-PRO (arXiv:2510.03827)](https://arxiv.org/abs/2510.03827)** 实证：
- 标准 LIBERO：**90%+**
- LIBERO-PRO 泛化设定：**0.0%**

### ⚠️ 警告 2 · VLA 完全忽略语言指令

📎 **[LIBERO-PRO](https://arxiv.org/abs/2510.03827) + [LIBERO-Para (arXiv:2603.28301)](https://arxiv.org/abs/2603.28301)**：
- Vision-only（mask 语言）：**44.6%**
- Language-conditioned：**47.8%**（≈ vision-only）

### ⚠️ 警告 3 · 轻微扰动导致崩溃

📎 **[LIBERO-Plus (arXiv:2510.13626)](https://arxiv.org/abs/2510.13626) · [LIBERO-X (arXiv:2602.06556)](https://arxiv.org/abs/2602.06556)**：

| 扰动 | 标准 | 扰动后 |
|------|:---:|:------:|
| 换物体 | 90%+ | <30% |
| 换初始位置 | 90%+ | ~0% |
| 换指令措辞 | 90%+ | 不变（因为不读指令） |
| 换环境 | 90%+ | ~0% |

### ⚠️ 警告 4 · 仿真排名 ≠ 真机排名

| 模型 | LIBERO 仿真 | 真机 | 跌幅 |
|------|:---------:|:----:|:----:|
| π0.5 | 96.9% | 52-77% | -20~45 |
| VGA | 98.1% | 58-75% | -23~40 |
| WVA | 99.6% | 75.6% | -24 |

### ⚠️ 警告 5 · 真机 benchmark 也有覆盖缺陷

- **RoboChallenge Table30**：仅 30 个桌面任务
- **GM-100**：仅 3 个机器人平台
- 都不含：移动操作 / 柔软物体 / 动态环境

### ⚠️ 警告 6 · 评测泄漏（🧠 作者提出，尚无 VLA 专论）

📎 类比 [How Contaminated Is Your Benchmark (arXiv:2502.00678)](https://arxiv.org/abs/2502.00678)（LLM 领域已成熟议题）：OXE 子集可能与 LIBERO/CALVIN 有视觉相似性。VLA 领域尚无系统 pHash 研究。

---

## 🏛️ 一、通用操作仿真 Benchmark

| Benchmark | 任务数 | 机器人 | 测什么 | 许可证 | 状态 |
|-----------|:-----:|-------|-------|:------:|:----:|
| **[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)** | 4×10 | Franka | 知识迁移、终身学习 | MIT ✅ | ⚠️ **饱和** |
| **[LIBERO-PRO](https://arxiv.org/abs/2510.03827)** | LIBERO+扰动 | Franka | **鲁棒性**（必报） | MIT ✅ | 2025 |
| **[LIBERO-Plus](https://arxiv.org/abs/2510.13626)** | 扰动扩展 | Franka | 更多扰动维度 | MIT ✅ | 2025 |
| **[LIBERO-X](https://arxiv.org/abs/2602.06556)** | 扰动 | Franka | 多维扰动 | MIT ✅ | 2025 |
| **[LIBERO-Para](https://arxiv.org/abs/2603.28301)** | 扰动 | Franka | 语言指令多变 | MIT ✅ | 2026 |
| **[CALVIN](https://github.com/mees/calvin)** | 34 | 桌面 | **长程语言条件** | MIT ✅ | 活跃 |
| **[RLBench](https://github.com/stepjam/RLBench)** | **100** | Franka | 手工多样任务 | MIT ✅ | 经典 |
| **[Meta-World](https://meta-world.github.io/)** | 50 | Sawyer | RL + 多任务标准 | MIT ✅ | 经典 RL 测试 |
| **[ManiSkill v3](https://github.com/haosulab/ManiSkill)** | 多种 | 多种 | GPU 加速仿真 | Apache-2.0 ✅ | RSS'25 |
| **[RoboCasa](https://robocasa.ai/)** | 厨房 | 移动操作 | 家庭场景泛化 | MIT ✅ | 活跃 |
| **[BEHAVIOR-1K](https://behavior.stanford.edu/)** | **1000** | 多种 | 开放世界最全面 | 未明确 ⚠️ | 活跃 |
| **[VLABench](https://arxiv.org/abs/2502.09587)** | 大规模 | 多种 | 长程语言条件 | 未明确 ⚠️ | ICCV'25 |
| **[HazardArena](https://arxiv.org/abs/2604.XXXXX)** | 语义安全 | 多种 | ⚡ 安全评估（填补空白）| 未明确 ⚠️ | 2026 |

## 🤖 二、真机 Benchmark（物理世界）

| Benchmark | 任务数 | 机器人 | 许可证 | 为什么重要 |
|-----------|:-----:|-------|:------:|-----------|
| **[RoboChallenge Table30](https://robochallenge.ai/)** | **30** | 标准化 | 未明确 ⚠️ | **首个在线真机评测**，可远程提交 |
| **[GM-100](https://huggingface.co/datasets/robbyant/lingbot-GM-100)** | **100** | 3 平台 | Apache-2.0 ✅ | 最多任务真机 benchmark |
| **[RoCo Challenge](https://arxiv.org/abs/2603.15469)** | 多任务 | 协作双臂 | 未明确 ⚠️ | **工业装配** · AAAI 2026 |
| **[AutoEval](https://auto-eval.github.io/)** ⭐ | 真机 24/7 | 多种 | — | 🟢 自动评测减少 99% 人工监督；评测结果与人评 ground truth 吻合 |
| **[RoboArena](https://arxiv.org/abs/2506.18123)** ⭐ | 分布式 | 多种 | — | 2025：**分布式真实评测**，多方联合测 VLA |

## 🔄 三、Real-to-Sim 评测工具

> 🧠 **为什么重要**：Sim → Real 的 gap 已知，但 **Real → Sim 验证**是新兴方向——即"在仿真里得到的排名，真机是否成立？"

| Benchmark | 核心方法 | 价值 |
|-----------|---------|------|
| **[SimplerEnv](https://simpler-env.github.io/)** | Visual Matching + Variant Aggregation | 📎 CoRL 2024 · Sim-Real 强相关性实证（MMRV + Pearson） |
| **[RobotArena ∞ (arXiv:2510.23571)](https://arxiv.org/abs/2510.23571)** ⭐ | Real-to-Sim translation + VLM 自评分 + 众包偏好 | 📎 ICLR 2026：**在大规模数字孪生中评测 VLA** |
| **[REALM (arXiv:2512.19562)](https://arxiv.org/abs/2512.19562)** ⭐ | Real-to-Sim 验证的 generalization benchmark | 2025：**专门测泛化** |
| **[TERM-Bench (arXiv:2601.18723)](https://arxiv.org/abs/2601.18723)** ⭐ | Eval-Actions + AutoEval 架构 | 2026：可信度评测体系 |

## 🤲 四、灵巧手 / Dexterous

| Benchmark | 任务 | 许可证 | 价值 |
|-----------|------|:------:|------|
| **[DexArt (arXiv:2305.05706)](https://arxiv.org/abs/2305.05706)** | 灵巧手 × 铰接物体 | 未明确 ⚠️ | CVPR'23 · **铰接物体泛化** |
| **[DexYCB](https://dex-ycb.github.io/)** | 手部抓取 + 姿态估计 | 未明确 ⚠️ | 手-物交互标注 |
| **HandoverSim** | 人-机交接（DexYCB 入仿真） | 未明确 ⚠️ | 仿真标配 |
| **[DexH2R](https://arxiv.org/abs/2509.XXXXX)** ⭐ | 动态人-机抓取交接（灵巧手） | 未明确 ⚠️ | ICCV 2025：**首个真实世界灵巧手 handover** |
| **[VTDexManip](https://openreview.net/forum?id=jf7C7EGw21)** | 视觉-触觉预训练 × RL | 未明确 ⚠️ | 首个视觉-触觉灵巧操作数据集 |

## 🦿 五、人形全身 / Humanoid

| Benchmark | 任务 | 机器人 | 价值 |
|-----------|:----:|-------|------|
| **[HumanoidBench (arXiv:2403.10506)](https://arxiv.org/abs/2403.10506)** ⭐ | 15 操作 + 12 locomotion | Unitree H1 + Shadow Hand | MuJoCo · **高维 RL 基准** · 发现 RL 多数失败 |
| **Ego Humanoid Manipulation Benchmark** ⭐ | **12 任务**（短至长程） | Unitree H1 + Inspire Hand | Isaac Lab · **reproducible egocentric** 测试台 |

## 🏠 六、移动操作 / Mobile Manipulation

| Benchmark | 任务数 | 机器人 | 价值 |
|-----------|:-----:|-------|------|
| **[HomeRobot OVMM (arXiv:2306.11565)](https://arxiv.org/abs/2306.11565)** ⭐ | **open-vocabulary** | Hello Robot Stretch | 仿真+真机双组件 · NeurIPS'23 Challenge |
| **[Habitat-Matterport HM3D](https://aihabitat.org/datasets/hm3d/)** | 1000 建筑级扫描 | 多种 | 最大室内 3D 数据集 |
| **[HM3D-OVON](https://ram81.github.io/docs/papers/OVON_IROS.pdf)** | 开放词汇导航 | 多种 | IROS 2024 · 物体目标导航 |

## 🧸 七、柔软物体 / Deformable

| Benchmark | 任务 | 许可证 | 价值 |
|-----------|------|:------:|------|
| **[SoftGym](https://sites.google.com/view/softgym)** | 布、绳、流体 | MIT ✅ | FleX 仿真 · 经典 deformable |
| **[PlasticineLab (arXiv:2104.03311)](https://arxiv.org/abs/2104.03311)** | 橡皮泥 | — | **可微分物理** · DiffTaichi |

## 🤚 八、触觉 / Tactile

| Benchmark | 任务 | 价值 |
|-----------|------|------|
| **[ManiFeel (RSS 2025)](https://www.robot-manipulation.org/events/workshops/rss-2025)** ⭐ | 视觉-触觉 policy 学习 | RSS 2025 · Purdue |
| **[POEMPEL (RSS 2025)](https://www.robot-manipulation.org/events/workshops/rss-2025)** ⭐ | **力感知操作** | RSS 2025 · KIT |
| **[TaF-Dataset (arXiv:2601.20321)](https://arxiv.org/abs/2601.20321)** | 10M 触觉-力对 | 6 种触觉传感器 |

## 🎯 九、VLA 专用

| Benchmark | 特点 | 价值 |
|-----------|------|------|
| **[VLA-Arena](https://vla-arena.github.io/)** ⭐ | 4 维评测：Safety / Distractor / Extrapolation / Long Horizon | **综合 leaderboard** + task store · 2026 |

---

## 🧠 十、**智能 / 推理 / 长程规划**（最新方向 · 2025-2026）

> **为什么单列**：传统 benchmark 测"机器人能不能把动作做对"——这些测"机器人能不能**像智能体**一样思考、规划、自我修正"。

### 10.1 具身推理（Embodied Reasoning）

| Benchmark | 任务 | 价值 |
|-----------|------|------|
| **ERIQ**（Embodied Reasoning Intelligence Quotient）⭐ | **6K+ QA 对** · 4 维推理 | 解耦推理与执行 · 📎 发现**推理能力与 VLA 泛化强相关** |
| **[COIN (arXiv:2604.16886)](https://arxiv.org/html/2604.16886)** ⭐ | Chain of Interaction · 推理 × 具身交互 | 2026 · 长程 state maintenance + 自适应规划 |
| **[RoboBench (arXiv:2510.17801)](https://arxiv.org/html/2510.17801v1)** ⭐ | MLLM 作为 **embodied brain** | 📎 评测"符号规划 + 具身可行性"双维度 · 暴露 affordance reasoning / failure diagnosis 的空白 |
| **[ECoT (OpenReview)](https://openreview.net/forum?id=S70MgnIA0v)** ⭐ | Embodied Chain-of-Thought | 📎 **+28% OpenVLA** 泛化任务 · 推理中间态（bbox/EE pos）作为监督 |

### 10.2 长程任务（Long-Horizon）

| Benchmark | 特点 | 价值 |
|-----------|------|------|
| **[LoHoVLA (arXiv:2506.00411)](https://arxiv.org/html/2506.00411v1)** ⭐ | 统一架构 · 长程任务 | 📎 超越 hierarchical VLA baseline |
| **VLABench**（已列 §1） | **复合任务 >500 步**（vs primitive 120 步）| ICCV 2025 · 多技能 + 多步逻辑推理 |
| **PsiBot R1（Psi R1）**⭐ | Chain of Action Thought (CoAT) | 📎 麻将 demo **30+ 分钟连续推理** · 真机长程 |

### 10.3 世界模型智能（World Model Intelligence）

| Benchmark | 测什么 | 价值 |
|-----------|--------|------|
| **PhysicsMind**（2026.01）⭐ | Sim + Real 机械学 · 物理推理 + 预测 | 评测**基础 VLM + 世界模型**的物理理解 |
| **RBench**（2026.01）⭐ | Video 生成模型作为 embodied world | 📎 "Rethinking Video Generation Model for the Embodied World" |
| **MobileWorldBench**（2025.12）⭐ | **语义世界建模** × 移动 agent | 从视觉预测跨越到语义级世界理解 |
| **[World Model Bench](https://worldmodelbench.github.io/)**（CVPR 2025）⭐ | 通用世界模型 benchmark | CVPR 2025 workshop 正式 benchmark |
| **PointWorld** ⭐ | 3D 世界模型 scaling | ICLR 2026 · 3D 表示路线 |

### 10.4 组合泛化（Compositional Generalization · π0.7 路线）

> 🧠 **为什么重要**：真智能 = 技能组合成新任务。测试模型能不能把学过的基元 **重组**出未见过的任务。

| 评测维度 | 代表工作 |
|---------|---------|
| **Steerable** 可引导性 | 📎 [π0.7](../theory/vla-core/pi0_7_steerable_compositional_generalization_2026.md)——通过 metadata 引导行为方向 |
| **Compositional** 技能组合 | π0.7 · LoHoVLA · VLABench composite tasks |
| **Causal / Recovery** 因果 + 自我修正 | F6 动作偏斜测试（见 [failure-modes](./failure-modes.md)）+ RoboMIND 2.0 失败数据 |

### 10.5 真世界持续评测基础设施

| 平台 | 定位 |
|------|------|
| **[ManipulationNet](https://manipulation-net.org/)** ⭐ | **社区治理** · 全球真实世界持续 benchmark · 物理操作 + 多模态推理 |

---

## 📐 "智能"层次的分层评测建议

```
L1  基础动作正确              →  LIBERO（标准）/ CALVIN / RLBench
     （能不能抓起来）
     │
L2  扰动下稳定                 →  LIBERO-PRO / Plus / X / Para
     （换个物体还能不能抓）           VLA-Arena Distractor
     │
L3  真机验证                   →  GM-100 / Table30 / AutoEval / RoboArena
     （仿真到真机不崩）
     │
L4  长程 + 多步                →  CALVIN / VLABench 复合 / LoHoVLA
     （能不能做 500 步的任务）
     │
L5  推理 × 规划               →  ERIQ / COIN / RoboBench / ECoT
     （能不能对未见任务推理）
     │
L6  世界理解                  →  PhysicsMind / RBench / World Model Bench
     （物理 / 语义 / 3D 建模正确）
     │
L7  自主智能体                →  PsiBot R1 CoAT / ManipulationNet
     （30 分钟连续推理 + 自我修正）
```

**关键洞察**：论文常在 L1-L2 刷分，L5-L7 才是通往"真正智能"的北极星 benchmark。2025-2026 的趋势是从 L1 往 L5+ 迁移。

⭐ 表示 2025-2026 新增。

---

## 🎯 Benchmark 选型决策树（深度版）

```
你要做什么？
│
├─ 快速验证想法（1 天内）
│   └─ LIBERO（⚠️ 必须同时报 PRO 对照）
│
├─ 长程语言任务
│   ├─ 标准 → CALVIN（34 任务）
│   └─ 开放词汇 → HomeRobot OVMM
│
├─ 全面评估能力
│   ├─ 100 任务 → RLBench
│   ├─ 1000 任务 → BEHAVIOR-1K
│   └─ 综合 4 维 → VLA-Arena
│
├─ Sim → Real 可信度
│   ├─ 标准 → SimplerEnv
│   ├─ 数字孪生 → RobotArena ∞
│   ├─ 验证泛化 → REALM
│   └─ 24/7 真机自动评测 → AutoEval
│
├─ 人形全身
│   ├─ RL 基准 → HumanoidBench（15+12）
│   └─ egocentric 操作 → Ego Humanoid Benchmark
│
├─ 灵巧手
│   ├─ 铰接物体 → DexArt
│   ├─ 人机交接 → DexH2R（最新真实）
│   └─ 视-触联合 → VTDexManip
│
├─ 移动操作
│   ├─ 开放词汇 → HomeRobot OVMM
│   └─ 导航 → HM3D-OVON
│
├─ 柔软物体
│   ├─ 布/绳/流体 → SoftGym
│   └─ 橡皮泥/形状变形 → PlasticineLab
│
├─ 触觉研究
│   ├─ 力感知 → POEMPEL
│   ├─ 视-触 policy → ManiFeel
│   └─ 跨触觉预训练 → TaF-Dataset
│
├─ 双臂
│   └─ RoboTwin 2.0 + GM-100
│
├─ 工业装配
│   └─ RoCo Challenge
│
├─ 真机排名
│   ├─ 在线提交 → RoboChallenge Table30
│   ├─ 分布式联合 → RoboArena
│   └─ 自动评测 → AutoEval
│
├─ 语义安全
│   └─ HazardArena · VLA-Arena Safety 维度
│
├─ 具身推理能力
│   ├─ QA 级别 → ERIQ（6K+ QA 对）
│   ├─ 交互级 → COIN
│   ├─ MLLM brain → RoboBench
│   └─ CoT 效果测 → ECoT
│
├─ 长程 + 多步智能
│   ├─ 30+ 分钟推理 → PsiBot R1（CoAT）
│   ├─ 统一架构 → LoHoVLA
│   └─ 复合任务（500+ 步）→ VLABench composite
│
├─ 世界模型智能测评
│   ├─ 物理预测 → PhysicsMind
│   ├─ 视频生成质量 → RBench
│   ├─ 语义世界 → MobileWorldBench
│   └─ 通用 WM → World Model Bench
│
└─ 真世界社区级持续评测
    └─ ManipulationNet（全球社区治理）
```

---

## 💡 Benchmark 报告的"黄金三件套"

报成绩时**必须同时给出**：

1. **主 benchmark 分数**（LIBERO / CALVIN / etc.）
2. **鲁棒性对照**（LIBERO-PRO / Plus / X 或 VLA-Arena Extrapolation）
3. **真机验证**（≥1 个真机 benchmark：GM-100 / Table30 / AutoEval）

只给第 1 个 = 不完整。

---

## 🏆 对研究者的 5 条硬建议

1. ❌ **不要只报 LIBERO 标准分**——必须同时报 LIBERO-PRO/Plus/X 扰动
2. ❌ **不要省略语言消融实验**——遮蔽语言后如果成功率不变，说明模型没用到语言
3. ✅ **必须报真机数字**——仿真 95%+ 的论文如果没有真机验证，价值存疑
4. ✅ **报失败案例 + 恢复能力**——RoboMIND 2.0 的 5K 失败案例是好参考
5. ✅ **披露数据混合来源**——预训练语料的子集构成要可审计

---

## ❗ 报成绩的对比范例

**不充分**：
> "我们在 LIBERO 上做到 97.2%"

**充分**：
> "LIBERO 标准 97.2%，LIBERO-PRO 泛化设定 **32.5%**——差距说明模型学到一部分模式，但在新物体/新位置下仍脆弱。
> vision-only baseline 消融：成功率从 97% 跌到 44%，说明语言确实起作用。
> 真机验证：GM-100 上 **68%**（10 任务平均），接触丰富子集 45%。"

—— 扰动 + 消融 + 真机三件套齐全，可信度完全不一样。

---

## 📊 按"可信度 × 规模 × 覆盖面"分层

| 层级 | benchmark | 特点 |
|------|----------|------|
| **S 级**（可信+广） | AutoEval · RobotArena ∞ · REALM · ManipulationNet | 2025-2026 最新 · 真机或数字孪生 · 规模化 |
| **A 级**（广但仿真） | RLBench · BEHAVIOR-1K · VLA-Arena · ManiSkill v3 · VLABench composite | 成熟、覆盖广、社区接受 |
| **B 级**（细分领域强） | DexArt · HumanoidBench · OVMM · PlasticineLab | 各自领域内经典 |
| **智能级**（🧠 新前沿） | ERIQ · COIN · RoboBench · LoHoVLA · PhysicsMind · RBench · ECoT | 2025-2026 · 推理 / 长程 / 世界模型 |
| **C 级**（⚠️ 已饱和） | LIBERO（纯）· CALVIN（纯）| 仍可用但需搭配扰动版本 |

---

## 📚 延伸阅读

- [VLA 数据工程指南](../theory/foundation/vla_data_engineering_guide.md) · 含完整 benchmark + 6 条警告详解
- [失效模式 F1-F6](./failure-modes.md) · 真机出问题时怎么定位
- [模型对比](./model_comparison.md) · 各模型的 benchmark 成绩（注意解读方式）
- [VLA-Arena 官方](https://vla-arena.github.io/) · 实时 leaderboard
- [Awesome Robot Manipulation](https://github.com/BaiShuanghao/Awesome-Robotics-Manipulation) · 社区维护 · 最新论文追踪

---

[← Back to Cheat Sheet](./README.md)
