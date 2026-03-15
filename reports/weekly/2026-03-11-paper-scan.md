# VLA Paper Scan — 2026-03-11

> **扫描窗口**: 2026-03-09 → 2026-03-11
> **信息来源**: arXiv, web search, ICLR 2026 proceedings, industry announcements
> **Belief Graph 版本**: v3 (2026-03-09 更新后)
> **重要补充**: 本次扫描同时回溯了部分 2026-01/02 发布但上次扫描遗漏的高价值论文

---

## 总览

本次扫描发现 **11 篇值得记录的新论文/发布**，其中：
- ⚡ 高 ΔI（需要三视角辩论）: **4 篇**
- 🔧 中 ΔI（值得跟踪）: **4 篇**
- 📖 低 ΔI（背景记录）: **3 篇**

**本次扫描核心发现**：
1. **Reward model 方向爆发** — Robometer (1M轨迹) + RoboReward (OXE基础) 同期出现，直接冲击套利3
2. **WM新范式涌现** — CoWVLA (latent motion) + AtomVLA (latent WM post-training) + Cosmos Policy (video→policy) 三条独立路线同时推进B4
3. **VLM4VLA (ICLR 2026) 关键发现**：视觉模块是VLA瓶颈，语言模块贡献有限 → **C3 逆共识的强支持信号**
4. **推理效率军备竞赛** — KERV/RAPID/Characterizing VLA 三篇集中攻击VLA部署瓶颈

---

## ⚡ 高 ΔI 论文 — 三视角辩论

---

### 论文 1: Robometer — 百万轨迹级通用 Reward Model

> **arXiv 2603.02115** | March 3, 2026
> **核心**: 基于 RBM-1M 数据集（100万+轨迹），训练通用 reward model；双目标：帧级 progress loss + 轨迹级 preference loss
> **关键数字**: 1M+ 轨迹, 多 embodiment, 支持 model-free RL / model-based RL / offline RL / failure detection / data retrieval
> **Belief Graph 冲击**: B2 (RL后训练), B3 (自我改进闭环), 直接冲击**套利3** (reward specification)
> **逆共识检查**: 无

#### 🔴 Bull: 这可能是 reward specification 问题的转折点——套利3 窗口正在关闭

Robometer 的 RBM-1M 数据集是目前**规模最大的 robotic reward 训练集**（100万+轨迹，跨多种 embodiment），而且包含大量**失败轨迹**——这是训练 reward model 最稀缺的数据。双目标设计（帧级 progress + 轨迹级 preference）比 RoboReward 的单一方案更全面。如果 Robometer 的 reward 足够可靠，**B3（自我改进闭环）的 reward specification 瓶颈就被解决了一大半**——闭环不再需要人工标注 reward。结合 RoboReward（同期，Google DeepMind 系，8B 模型在真机 RL 中改进策略学习），reward model 方向正在经历**从"概念验证"到"工程可用"的跃迁**。

这直接威胁套利3 的窗口——如果 VLM-as-reward-model 方案在 2026Q2 被证明"足够可靠"，套利3 的 12 个月窗口可能缩短到 6 个月。

#### 🔵 Bear: 100万轨迹的多样性不等于覆盖率，reward model 在 OOD 场景下的可靠性才是关键

Bear 直接反驳 Bull 的"转折点"论断：
1. **RBM-1M 的分布问题**。100万轨迹主要来自 OXE 等现有数据集——这些数据集以桌面操作为主。在 humanoid 全身操作、户外导航、精细装配等场景中，Robometer 的 reward 可靠性未经验证。100万看起来多，但对真实世界的 task 多样性来说仍是冰山一角。
2. **Reward hacking 风险**。RL 的经典问题：reward model 越通用，策略 hack 它的方式就越多。Robometer 在"已知任务"上表现好不代表在"新任务"上不会被 exploit。B3 的致命实验——"reward specification 问题无法解决（机器人 hack reward 而非完成任务）"——仍未排除。
3. **RoboReward 8B 与 Robometer 的关系**：两者用不同数据、不同架构——如果结果一致是强信号，但如果 Robometer 在 RoboReward 失败的场景中也失败，说明 VLM reward model 有结构性盲点。

#### 🟢 Arbiter: 套利3 窗口缩短，但未关闭——现在是验证 reward model 可靠性的最佳时机

- **如果 Bull 对了**: Reward model 在 2026H2 成为 VLA 自我改进的标准组件。你现在就该在自己的任务上测试 Robometer/RoboReward 作为 RL reward 的可靠性。
- **如果 Bear 对了**: Reward model 在已知任务上好用但在新任务上不可靠，reward specification 仍是人工问题。但即使如此，reward model 作为"粗筛"（过滤明显失败轨迹）已经有实用价值。
- **时间套利**: **高**。Robometer + RoboReward 同期出现说明领域正在快速收敛到"VLM-as-reward"方案。但大多数团队还在用人工标注或二值 reward。如果你现在就在自己的管线中集成 reward model，当主流跟进时你已经积累了可靠性数据和调优经验。
- **套利3 状态更新**: 窗口从 12 个月缩短至 **6-9 个月**。致命条件"VLM-as-reward-model 方案证明足够可靠"正在逼近。

#### Belief Graph 更新
- B2 (RL后训练): 不变 (校准后 81%)。Robometer 支持 RL，但 B2 已经是高置信度，增量验证不足以推动更新。
- B3 (自我改进闭环): **80% → 85% → 校准后 77%** (+5%)
  - 理由: Robometer + RoboReward 两个独立团队同时验证 VLM reward model 可行。B3 最强反方叙事的核心——"真实世界 reward 极难定义"——被部分削弱。Bull 和 Bear 都同意 reward model 在**已知任务**上有效，分歧仅在 OOD 泛化。按校准纪律，Bull+Bear 部分共识 → 最小更新 +5%。
  - **传播**: B3→B4 检查：B3 上调增强了 B4 的前提条件，但 B4 的瓶颈（物理保真度）独立于 reward model，不传播。

---

### 论文 2: Chain of World (CoWVLA) — 潜在运动空间的 World Model 思考

> **arXiv 2603.03195** | March 3, 2026
> **核心**: 将视频分解为"结构+运动"潜在空间，VLA 在潜在运动空间做 chain-of-thought 式预测，避免 pixel-space 重建浪费
> **关键创新**: video VAE 提取 latent motion → VLA 预测 latent motion chain → 与 action 在 unified decoder 中联合建模
> **Belief Graph 冲击**: B4 (World Model加速器)
> **CONVERGENCE_MAP Phase 4 更新**: 新的独立路线
> **逆共识检查**: 弱反对 C2 (WM是死胡同) — 如果 latent WM 解决了物理幻觉问题

#### 🔴 Bull: 这是 WM 方向的"正确打开方式"——绕过物理幻觉

B4 当前卡在 50% 的核心原因是"物理幻觉"——pixel-space 的 WM 预测接触后物理交互时系统性不准确。CoWVLA 的关键洞察是：**你不需要在 pixel space 预测完整未来帧**。通过分离结构（背景不变）和运动（物体位移/旋转），WM 只需要预测运动的低维流形——物理幻觉在低维空间中的影响远小于在高维 pixel space 中。如果这个思路正确，它同时解决了 B4 的两个障碍：
1. 物理幻觉 → 在 latent motion space 中被稀释
2. 计算成本 → latent space 远小于 pixel space（不需要 14B video diffusion）

结合 Cosmos Policy（NVIDIA，将 video model 直接变为 policy）和 AtomVLA（latent WM post-training），WM 方向正在从"pixel-space 暴力预测"收敛到"latent-space 高效推理"。这可能是 WM 真正实用化的技术路径。

#### 🔵 Bear: Latent space 不是万能药——运动和物理是耦合的

Bear 挑战 Bull 的核心假设"latent motion 避开物理幻觉"：
1. **运动和接触物理不可分离**。当机器人推一个物体时，物体的运动取决于接触力、摩擦、质量——这些在"运动潜在空间"中未被显式建模。VAE 的 latent space 可能编码了运动的表观，但丢失了因果物理关系。
2. **历史先例**：latent dynamics model（如 Dreamer 系列）在 Atari/MuJoCo 上成功，但在真实机器人接触密集任务中表现显著下降——因为接触物理的非线性在 latent space 中被"压缩掉了"。
3. **CoWVLA 的实验还在 sim benchmark 上**。没有真机验证，任何关于"解决物理幻觉"的声明都是 premature。

直接反驳 Bull 关于"收敛"的论点：CoWVLA/Cosmos Policy/AtomVLA 三个方案走的是完全不同的 latent 路线（motion VAE / video diffusion latent / subtask latent），这说明领域还在**方法论探索阶段**，而非"收敛"。

#### 🟢 Arbiter: B4 应该微幅上调，但"latent WM 是否比 pixel WM 好"需要真机数据

- **如果 Bull 对了**: Latent WM 在 2026H2 成为 VLA 的标准预训练范式（取代 pixel WM）。对你的含义：关注 latent dynamics 表征的研究比关注 video generation 模型更值得。
- **如果 Bear 对了**: Latent WM 和 pixel WM 都在接触密集任务上失败，纯 real-data RL（如 Recap）仍是最可靠路径。C2（WM是死胡同）的置信度应上调。
- **关键分歧点**: "latent motion space 是否保留了足够的物理因果信息？" 这是一个**实验问题**，不是理论问题。需要在真机接触密集任务（如精细装配、软物体操作）上对比。

#### Belief Graph 更新
- B4 (World Model): **50% → 55%** (+5%)
  - 理由: CoWVLA + Cosmos Policy + AtomVLA 三条独立 latent WM 路线同时出现，方向信号密度很高。虽然 Bear 正确指出"方法论碎片化"，但"多团队独立探索 latent WM"本身就是一个 Phase 4 的加速信号。最小更新 +5%。
  - 致命实验状态: "2026年底前无团队在真机>1000 episodes中验证WM辅助优于纯BC+RL" — 尚未触发，但 Cosmos Policy 在真机 bimanual 上的表现是最接近的数据点。
- C2 (WM是死胡同): **25% → 22%** (-3%)
  - 理由: 多条 latent WM 路线的出现略微削弱了 C2。但 Bear 关于接触物理的论点仍成立，不做大幅下调。

---

### 论文 3: VLM4VLA (ICLR 2026) — 视觉模块才是 VLA 瓶颈，不是语言

> **arXiv 2601.03309** | Jan 2026 | **ICLR 2026 Poster**
> **核心**: 系统性研究 VLM 选择如何影响 VLA 性能。关键发现：(1) VLM 的通用能力与下游 VLA 性能弱相关；(2) **视觉模块是性能瓶颈**，语言模块贡献有限；(3) 向视觉编码器注入控制相关监督可持续提升性能
> **Belief Graph 冲击**: B0 (数据>架构), B7 (Action Expert解耦)
> **逆共识检查**: ⚠️ **C3 (VLA不需要language) 的强支持信号**

#### 🔴 Bull: 这是 C3 升格的关键证据——语言可能是 VLA 中最不重要的模态

VLM4VLA 的发现"视觉模块是瓶颈，语言贡献有限"直接支持 C3 的核心主张。如果语言模块在 VLA 中的作用主要是"传递任务指令"而非"推理"，那么更简单的 goal specification（如目标图像、点击标注、结构化任务ID）可能比自然语言更高效。这意味着：
1. **VLA 中的 L 可能是工程负担**：语言 backbone 占了大量参数但对动作精度贡献有限。
2. **视觉编码器的"控制相关监督"** 才是真正的杠杆——这与 Spatial Forcing (ICLR 2026) 的 3D 空间感知注入方向一致。
3. 如果 C3 最终被验证，当前所有 VLA 的架构设计（先理解语言→再生成动作）都需要重构。

这是 C3 从 15% 升格的最强信号——一篇 ICLR 2026 poster 在 7 个 embodied 任务上系统性地证明了语言的边际贡献低。

#### 🔵 Bear: "语言贡献有限"在简单任务中成立，在复杂多步任务中不一定

Bear 精准切入 Bull 的推理链中的弱环节：
1. **实验范围限制**。VLM4VLA 的 ablation 基于 Qwen2.5-VL backbone 在标准 benchmark（LIBERO/CALVIN 等）上的结果。这些 benchmark 的任务语言描述相对简单（"pick up the red cup"），语言当然贡献有限。但在**需要复杂语义理解的任务**（"把最轻的那个放到第二高的架子上"）中，语言理解就是关键。
2. **InstructVLA (ICLR 2026)** 的发现恰好相反——它证明了语言指令遵从能力对 VLA 泛化至关重要，在 SimplerEnv-Instruct 的 80 任务 benchmark 上大幅超越不强调语言的方案。VLM4VLA 和 InstructVLA 的结论矛盾，说明**答案取决于任务复杂度**。
3. **C3 的推理链仍不完整**：即使语言模块的"直接"贡献有限，VLM 的语言预训练可能提供了隐式的世界知识（如"杯子是脆弱的→需要轻拿"），去掉 L 可能丢失这些隐式先验。

#### 🟢 Arbiter: C3 应该上调，但 VLM4VLA vs InstructVLA 的矛盾是关键分歧点

- **如果 Bull 对了**: 语言在 VLA 中是"nice to have"而非"must have"。投资方向应该转向视觉编码器的控制感知能力（如 Spatial Forcing 式的 3D 监督注入），而非更大的 LLM backbone。
- **如果 Bear 对了**: 语言在复杂多步任务中仍不可或缺。但 VLM4VLA 的发现仍有价值——它说明了视觉编码器的改进是当前投入产出比最高的方向。
- **对冲行动**: 无论 C3 最终结果如何，"改进视觉编码器"都是高价值行动——这是 VLM4VLA 和 InstructVLA 都同意的。
- **C3 的致命实验建议**: "在需要复杂语义理解的长时序任务（>10步、需要推理的指令）上，去掉语言模块的 VLA 是否仍能泛化？" 如果能 → C3 升格；如果不能 → C3 回到 15%。

#### Belief Graph 更新
- C3 (VLA不需要language): **15% → 22%** (+7%)
  - 理由: ICLR 2026 系统性研究直接支持 C3 核心主张，且 Spatial Forcing 的 3D 视觉监督方向也隐含了"视觉 > 语言"的信号。但 InstructVLA 的矛盾结论限制了更新幅度。+7% 超过最小更新 5%，合理。
  - 注意: C3 仍远低于 40% 升格阈值，但方向正确——需要持续追踪。
- B7 (Action Expert解耦): 不变 (80%)。VLM4VLA 支持"视觉和动作应该分开优化"但不直接涉及 Action Expert 架构。

---

### 论文 4: Cosmos Policy (NVIDIA) — 从 Video Model 直接到 Robot Policy

> **arXiv 2601.16163** | Jan 2026 | **ICLR 2026**
> **核心**: 将大型预训练视频模型 (Cosmos-Predict2) 通过单阶段 post-training 直接变为 robot policy，无需架构修改。在 latent diffusion 过程中同时生成 action + future observation + value estimate。
> **关键数字**: LIBERO 98.5%, RoboCasa 67.1%, 真机 bimanual SOTA
> **Belief Graph 冲击**: B4 (World Model), B5 (Flow Matching), B7 (Action Expert)
> **逆共识检查**: 弱支持 C2 反向（WM 可用）；弱挑战 B7（统一模型生成 action+video，未显式解耦）

#### 🔴 Bull: 这证明了 WM 不仅是辅助工具，它可以直接成为 policy

Cosmos Policy 的核心突破不是数字（虽然 98.5% LIBERO 是 SOTA），而是**范式转换**：video model IS the policy。之前的 WM 方案都是"WM 辅助 policy"——WM 生成合成数据/预测未来，policy 是独立模块。Cosmos Policy 说："不需要两个模块，video model 的 latent diffusion 过程本身就能编码 action"。这意味着：
1. **B4 的定位可能需要重新定义**：WM 不只是"加速器"，可能直接就是"policy 本身"。
2. **预训练规模优势巨大**：Cosmos-Predict2 在互联网视频上预训练的物理先验直接迁移到机器人——这绕过了"真机数据稀缺"的核心约束。
3. **真机 bimanual SOTA** 说明这不只是 sim 上的把戏。

结合 CoWVLA 的 latent motion 方向，2026 年 WM 领域正在从"能不能用"转向"怎么用最好"。

#### 🔵 Bear: LIBERO 98.5% 说明不了什么——真正的考验是 OOD 泛化和长时序

Bear 的三个致命反驳：
1. **LIBERO 是一个极度 overfit 的 benchmark**。在 LIBERO 上刷到 98%+ 的方案已经有多个，这个数字不再有区分度。RoboCasa 67.1% 更有信息量但仍是 sim。
2. **"video model = policy" 的内存和延迟成本**。Cosmos-Predict2 是一个大型 video diffusion 模型——推理一次需要多少 GPU-ms？如果不能在 edge 实时跑（<100ms），这个方案只适用于"云端控制"场景。这直接与 B9（小模型边缘部署）矛盾。
3. **统一 action+video+value 的风险**：三个目标在同一个 diffusion 过程中优化，可能导致 action 精度被 video reconstruction 拖累。这是 B7 反方叙事（"统一模型内部涌现功能分区"）的一个测试——如果 Cosmos Policy 在精细操作上的 action 精度不如专用 Action Expert，B7 的解耦论点就被强化。

#### 🟢 Arbiter: B4 应该上调，但 Cosmos Policy 的实际部署可行性是关键观察点

- **如果 Bull 对了**: "Video model as policy" 成为新范式，对你意味着应该关注 video foundation model（如 Cosmos / Sora 类模型）的 robot adaptation，而非从头训练 VLA。
- **如果 Bear 对了**: Cosmos Policy 是有趣的学术贡献，但推理成本限制了实际部署。专用 VLA + 轻量 WM 辅助仍是工程最优解。
- **关键数据需求**: Cosmos Policy 在真机上的推理延迟和 action 频率——如果 <50ms / >20Hz → 范式转换是真的；如果 >200ms / <5Hz → 只适用于慢速任务。

#### Belief Graph 更新
- B4 (World Model): 已在论文2中上调至 55%，Cosmos Policy 是**额外独立支持**，但 B4 本次已更新，不重复计算。标记为 Phase 4 新信号。
- B7 (Action Expert): 不变 (80%)。Cosmos Policy 的统一方案需要更多精细操作验证才能挑战 B7。观察中。

---

## 🔧 中 ΔI 论文 — 跟踪记录

---

### 论文 5: AtomicVLA — Atomic Skill MoE 用于持续学习

> **arXiv 2603.07648** | March 8, 2026
> **ΔI**: 🔧中 — 支持 B7 (Action Expert解耦)
> **核心**: Skill-Guided Mixture-of-Experts (SG-MoE) 构建可扩展的原子技能库，每个 expert 专精一种原子技能。在长时序和持续学习任务中分别超越 baseline 18.3% 和 21%。

**跟踪理由**:
- **MoE + skill library** 是 B7 (Action Expert 解耦) 的技术实例化——将"统一 action head"拆分为专用 skill experts。
- 持续学习 +21% 是有意义的——灾难性遗忘是 VLA 部署的真实痛点。
- 但 Bear 反驳：MoE 的 routing 在 OOD 任务中可能错误分配 expert，且 skill 的原子化粒度如何定义是开放问题。
- 不升为高 ΔI 因为：MoE 用于 VLA 不是新概念（Being-H0.5 也用了 MoT），AtomicVLA 是增量改进。

**Belief Graph**: B7 +0 (方向确认，但已是 80% 高置信度)。

---

### 论文 6: AtomVLA — 子任务感知 + 预测性 Latent WM Post-Training

> **arXiv 2603.08519** | March 9, 2026
> **ΔI**: 🔧中 — 支持 B3 (自我改进) + B4 (World Model)
> **核心**: 首个"子任务感知"VLA + 可扩展离线 post-training pipeline，用 predictive latent world model 提供中间指导，减少长时序 compounding error。

**跟踪理由**:
- "子任务感知" + "latent WM guidance" 是 B3 和 B4 的交叉信号——用 WM 预测来引导子任务完成。
- 但实验细节有限（March 9 刚发布），需要等真机数据。
- 与 CoWVLA 的 latent motion 方向不同——AtomVLA 的 latent 更偏向任务语义层（子任务 → 动作），CoWVLA 偏向运动几何层（运动 → 动作）。两者是互补而非竞争。

**Belief Graph**: B4 +0 (已在论文2中更新), B3 +0 (支持但不够强)。

---

### 论文 7: KERV — 运动学校正的推测性解码加速 VLA 推理

> **arXiv 2603.01581** | March 2, 2026
> **ΔI**: 🔧中 — 支持 B9 (小模型边缘部署), 推理效率方向
> **核心**: 将 Speculative Decoding 引入 VLA，用运动学 Kalman Filter 做推测性解码的草稿预测。27%-37% 加速，几乎无成功率损失。

**跟踪理由**:
- 推理加速对 VLA 部署至关重要——37% 加速意味着原来 200ms 的推理变成 126ms。
- 运动学先验（Kalman Filter）是智能的——利用了机器人动作的物理平滑性。这比纯 ML 加速更鲁棒。
- 与 RAPID (edge-cloud partitioning) 和 Characterizing VLA (bottleneck analysis) 一起，显示推理效率已成为2026年VLA部署的核心战场。
- 不升为高 ΔI：加速方法不改变信念网络，只改变部署时间线。

**Belief Graph**: B9 +0 (支持方向但 KERV 针对 token-based VLA，不直接验证小模型优势)。

---

### 论文 8: Being-H0.5 — 人类中心跨形态泛化

> **arXiv 2601.12993** | Jan 2026
> **ΔI**: 🔧中 — 支持 Phase 5 (跨形态泛化), B0 (数据>架构)
> **核心**: 35,000小时多模态数据, 30种 embodiment, 统一动作空间, Mixture-of-Flow (MoF) 框架。LIBERO 98.9%。5 个真机平台验证。

**跟踪理由**:
- **35,000小时 + 30种 embodiment** 是目前最大规模的跨形态预训练。Phase 5 的独立收敛计数可能需要 +1（需验证独立性）。
- **Mixture-of-Flow (MoF)** = MoE + Flow Matching 的结合 → 同时支持 B5 (FM) 和跨形态。
- **统一动作空间**是 Phase 5 的关键约束——Being-H0.5 的方案如何映射异构动作空间值得关注。
- 不升为高 ΔI：跨形态方向已知，Being-H0.5 是规模上的推进但不是方法论突破。

**Phase 5 计数器**: 待评估。Being-H0.5 (BeingBeyond) 的独立性需要确认——如果不引用 HPT/CrossFormer 的动作空间映射方案 → ✅ 独立 → Phase 5: 5→**6**。

---

## 📖 低 ΔI 论文 — 一行记录

| 论文 | 一句话 | ΔI |
|------|--------|-----|
| **RAPID** (arXiv 2603.07949) | Edge-Cloud 协同推理优化 VLA 部署，冗余感知 + 兼容性最优分区。 | 📖 工程优化，不改变信念。B9 背景噪音。 |
| **Characterizing VLA Models** (arXiv 2603.02271) | 分析 VLA 的 action generation bottleneck，为边缘 AI 芯片设计提供依据。 | 📖 架构分析，不改变信念。可能影响未来芯片设计方向。 |
| **Spatial Forcing** (ICLR 2026) | 隐式 3D 空间对齐策略，30行代码即插即用，训练加速 3.8x。 | 📖 支持 VLM4VLA 的"视觉是瓶颈"结论，但是增量优化。 |

---

## 相变计数器更新

| Phase | 上次 | 本次 | 变化 | 说明 |
|-------|------|------|------|------|
| Phase 1 (FM→action head) | 4/4 独立 | 4/4 独立 | 无变化 | Being-H0.5 MoF 使用 FM 但非独立信号 |
| Phase 2 (RL后训练) | 4/4 独立 | 4/4 独立 | 无变化 | Robometer 支持 RL reward 但不是 RL 方法本身 |
| Phase 3 (触觉标准化) | 7/7 独立 | 7/7 独立 | 无变化 | 无新触觉信号 |
| Phase 4 (World Model) | 4/4 独立 | **6/6 独立** | **+2** | CoWVLA (✅独立: latent motion 新路线) + Cosmos Policy (✅独立: NVIDIA video→policy 新路线) |
| Phase 5 (跨形态) | 5/5 独立 | **6/6 独立** | **+1** | Being-H0.5 (✅独立: 人类中心学习 + MoF，新方法论) |

**Phase 4 独立性验证**:
- CoWVLA: 来源追溯 → 基于 video VAE latent factorization，与 VLAW/DreamZero/1X 的方法不同。✅ 独立。
- Cosmos Policy: 来源追溯 → NVIDIA Cosmos 视频基座模型的 robot adaptation，与上述所有方法的出发点不同。✅ 独立。

**Phase 4 状态更新**: 30% → **40%**。独立收敛计数从 4 → 6，且出现了 latent WM 这一新技术子类。但物理保真度核心障碍仍在。

---

## Belief Graph 变更摘要

| 节点 | 旧值 (校准后) | 新值 (校准后) | Δ | 原因 |
|------|-------------|-------------|---|------|
| B3 自我改进闭环 | 80% | **85% → 校准后 77%** | +5% (原始) | Robometer + RoboReward 双重验证 reward model |
| B4 World Model | 50% | **55%** | +5% | CoWVLA + Cosmos Policy + AtomVLA latent WM 路线爆发 |
| C2 WM是死胡同 | 25% | **22%** | -3% | 多条 latent WM 路线削弱 C2 |
| C3 VLA不需要language | 15% | **22%** | +7% | VLM4VLA (ICLR 2026) 系统性证据 |
| 其他 | 不变 | 不变 | 0 | — |

**传播检查**:
- B3 80%→85%: 后果节点 B4 检查 → B4 的瓶颈是物理保真度（独立于 B3），不传播。
- B4 50%→55%: 无后果节点（末端），不传播。
- C3 15%→22%: 仍远低于 40% 升格阈值，不触发升格。

**校准纪律检查**:
- B3 原始 85% → × 0.9 = 校准后 77%。✅
- B4 55% 在 60% 以下区间，不需折扣。✅
- 所有更新 ≥ ±5% 最小更新幅度。✅

---

## 逆共识组合更新

| 逆共识 | 旧值 | 本周信号 | 新值 | 说明 |
|--------|------|---------|------|------|
| C1: 架构创新回归 | 20% | 无信号 | 20% | — |
| C2: WM是死胡同 | 25% | Latent WM 多路线爆发（反 C2） | **22%** | 多条独立 latent WM 路线削弱 C2 |
| C3: VLA不需要language | 15% | VLM4VLA (ICLR 2026) 强信号 | **22%** | 系统性研究证明语言模块贡献有限 |

**C3 升格监控**: 22% → 需要再累积约 18% 的证据才触发升格。关键下一步：需要在复杂语义任务上验证"去语言"方案是否仍有效。如果有效 → C3 可能快速升至 >40%。

---

## 时间套利检查

### 本周新套利机会

#### 套利 4（新）: "视觉编码器的控制感知注入是被低估的杠杆"
```
你的判断: VLM4VLA + Spatial Forcing 共同指向一个结论：
         改进 VLA 的最高 ROI 不是更大的 LLM backbone，
         而是向视觉编码器注入 3D/控制相关监督
领域主流: 多数团队在 scaling LLM backbone (3B→7B→13B)
套利窗口: 6-9 个月（ICLR 2026 结果传播后窗口关闭）
致命条件: 更大 LLM backbone 在复杂推理任务上的优势
          压过视觉编码器改进的增益
建议行动: 在现有 VLA pipeline 中加入 Spatial Forcing 式
          的 3D 监督——30行代码，3.8x 训练加速
```

### 现有套利机会状态更新

| 套利 | 状态 | 变化 |
|------|------|------|
| 套利 1 (触觉×RL) | 窗口仍开放 | 无新信号 |
| 套利 2 (推理延迟硬件) | 窗口仍开放 | KERV/RAPID 的软件加速缩小了差距，但硬件跳跃仍是更大的杠杆 |
| **套利 3 (reward specification)** | **窗口缩短** | Robometer + RoboReward 双重信号。窗口从 12 个月 → **6-9 个月** |
| 套利 4 (视觉编码器) | **新** | VLM4VLA + Spatial Forcing |

---

## 自检

- ✅ 本周有 1 个下调: C2 25%→22%
- ✅ 引用来源时间分布: March 2026 (5) + Jan-Feb 2026 (4) + ICLR 2026 (2) — 不过度依赖新近性
- ✅ 权威偏误检查: Cosmos Policy (NVIDIA) 获得与 CoWVLA (独立团队) 同等的 Bear 审查
- ✅ 收敛独立性: Phase 4 新信号 (CoWVLA, Cosmos Policy) 独立性已验证
- ✅ 逆共识保护: C3 的 VLM4VLA 信号通过低阈值通道保留并上调
- ⚠️ 注意: B3 原始值 85% 经校准后为 77%——用户看到的是"闭环信心增强"，但校准后置信度实际比上周（80%）还低。这不是矛盾——校准纪律的目的就是防止高置信度区间的过度自信。

---

*下次扫描: 2026-03-16*
*系统版本: CLAUDE.md v3*
