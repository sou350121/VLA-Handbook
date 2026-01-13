# 灵巧手数据采集方案 (Dexterous Hand Data Collection)

在具身智能（Embodied AI）尤其是 VLA (Vision-Language-Action) 模型的训练中，数据质量是决定模型上限的核心因素。相比平行夹爪，灵巧手（Dexterous Hand）由于其高自由度和复杂的接触动力学，对数据采集的精度、同步性和多模态覆盖有更高要求。

## 1. 数据单元定义 (Episode & Timestep)

高质量的数据集必须遵循严格的结构化定义，以便于行为克隆（Behavior Cloning）或扩散策略（Diffusion Policy）的学习。

### Episode (轨迹级) 结构
- **Episode ID**: 唯一标识。
- **Task Description**: 语言指令（如 "Pick up the bottle and unscrew the cap"）。
- **Success Label**: 二元标签（由人工标注或自动评测得出）。
- **Timesteps**: 一系列时序记录。

### Timestep (步级) 核心信号
| 信号类别 | 核心字段 | 说明 |
| :--- | :--- | :--- |
| **视觉 (Vision)** | `rgb_static`, `rgb_wrist` | 场景全局视角 + 腕部第一人称视角（POV）。 |
| **本体 (Proprio)** | `joint_positions`, `joint_velocities` | 灵巧手 N 个关节的绝对位置与速度。 |
| **示教 (Action)** | `joint_targets` | 专家示教的下一帧目标位置（通常是 $\Delta$ 增量或绝对位置）。 |
| **触觉 (Tactile)** | `pressure_map`, `contact_force` | 关键触碰点的压力分布（若硬件支持）。 |

---

## 2. 三大数采路线 (Acquisition Routes)

### 方案 A：专家遥操作 (Expert Teleoperation) — "Gold Standard"
- **原理**: 专家佩戴 VR 头显（如 Apple Vision Pro）或数据手套（Manus/SenseGlove），通过 Retargeting 算法实时驱动灵巧手。
- **优点**: 包含完整的示教逻辑，数据符合人类先验。
- **挑战**: 成本极高，Retargeting 的运动学映射（Mapping）可能存在伪影。

### 方案 B：脚本化/半自动化采数 (Scripted-HITL) — "Scalable"
- **原理**: 针对特定任务（如抓取），编写启发式脚本（Heuristic Script）完成大部分动作，人在关键点（Human-in-the-loop）通过按键切换状态或微调动作。
- **优点**: 效率高，可在大规模环境下快速生成基础轨迹。
- **挑战**: 泛化能力弱，仅适用于结构化任务。

### 方案 C：混合启动方案 (Bootstrap / Filtered Play)
- **原理**: 利用现有的基础模型（如 OpenVLA）进行自主探索，专家仅在模型报错或即将碰撞时介入纠偏（Intervention）。
- **优点**: 采集到的数据专门针对模型的“盲区”，训练效率极高（参考 DAgger 思想）。

---

## 3. 核心技术点

### 3.1 运动学重定向 (Kinematic Retargeting)
将人手的 21+ 自由度映射到机器手的 N 个关节（如 Wuji Hand 的 20-DOF）。
- **常用算法**: 逆运动学（IK）映射、几何比例映射（Finger-to-Finger）。
- **关键**: 必须处理“奇异点”和“自碰撞”保护，防止损坏硬件。

### 3.2 触觉代理 (Tactile Proxies)
若硬件无触觉传感器，可使用**电流/扭矩反馈 (Effort)** 作为触觉代理信号，判断是否已稳固抓取。

### 3.3 数据回放验证 (Replay Validation)
数采完成后，必须在真机上**重放（Replay）**记录的 `joint_targets`。如果重放无法达成 Success，说明记录的数据存在延迟或精度丢失，属于“脏数据”。

---

## 4. 质量控制 (QC) 与 安全

- **频率一致性**: 视觉（30Hz）与控制（1000Hz）必须严格对齐（详见 [多模态同步](./multimodal_data_synchronization.md)）。
- **力度上限 (Torque Limit)**: 示教过程中若专家用力过猛，采集程序需自动截断动作并报警。
- **看门狗 (Watchdog)**: 任何数采程序必须配备“急停（E-Stop）”按钮，物理切断电机电源。

---

## 4.1 对标 DYNA 的 “RM-in-loop” ：把「质量」做成可规模化的信号

DYNA 的公开叙事里，一个很关键的工程点是：在连续部署（streaming、没有清晰 episode 边界）的场景下，用 **Reward Model (RM)** 去做「进度评估、质量蒸馏、错误恢复与数据切段」，从而让机器人能 **长时无干预运行**，并把部署数据“自动变成训练数据”。这套思路对灵巧手非常适配，因为灵巧操作的关键不是“某一次成功”，而是“接触密集状态下持续稳定 + 快速自救”。

参考：
- DYNA-1 强调 RM 能提供细粒度反馈，并支持 *Intentional Error Recovery* 与 *High-Quality Dataset Creation and Curation*：<https://www.dyna.co/dyna-1/research>
- DYNA 的“第一性原理”里也把 RM 作为质量度量与可规模化改进的一部分：<https://www.dyna.co/blog/dyna-robotics-closes-120m-series-a>

下面把 DYNA 的模块拆成「DexHand 可落地」的接口与数据字段。

### A) RM 的输入：DexHand 要补哪些“可判别质量”的观测

把 RM 理解成一个函数 \(r_t = \text{RM}(o_{0:t}, a_{0:t})\)（不一定输出奖励，也可以输出 *progress/quality* 分数）。为了让 RM 能可靠地区分“稳/不稳、可恢复/不可恢复”，DexHand 的观测需要比纯视觉更“接触可见”：

- **视觉**: `rgb_static`, `rgb_wrist`（你已有）
- **本体**: `joint_positions`, `joint_velocities`（你已有）
- **力/扭矩代理（强烈建议加入）**: `joint_effort` / `motor_current`（用来判别接触、夹持、卡死、打滑）
- **接触 proxy（可选但很有价值）**: `tactile`（若有传感器），或 `contact_flags`（阈值化的接触事件）
- **环境状态（若可获取）**: 物体是否仍在手中（通过视觉跟踪/秤/开关量），作为 RM 的弱监督信号

> 直觉：DexHand 的“失败”往往不是看不到，而是 **接触后物理状态变了**（滑、夹歪、卡、扭矩异常），这些必须进入 RM 的判别空间。

### B) RM 的输出：不只一个标量 reward，而是一组“部署可用”的头

DYNA 叙述里 RM 的价值之一是“进度估计 + 质量度量”，对 DexHand 我建议直接把 RM 设计成多头输出（你可以先用规则生成伪标签训练）：

- **progress**: \(\hat{p}_t \in [0,1]\)（任务进度/阶段）
- **quality**: \(\hat{q}_t \in [0,1]\)（动作是否“生产级稳定”）
- **recoverability**: \(\hat{c}_t \in [0,1]\)（当前是否处于可自救状态）
- **failure_mode**: 分类（`slip`, `jam`, `self_collision_risk`, `occlusion`, `unknown`…）

这比“成功/失败”更贴近灵巧手的真实需求：**你需要知道怎么救**，而不是只知道失败了。

### C) “RM-in-loop” 训练与部署闭环：对应到三条数采路线

把 DYNA 的闭环翻译成你这份文档里的 A/B/C 三条数采路线：

- **方案 A（专家遥操作）**：用专家轨迹提供“正例段”，并刻意采集 **纠偏段（recovery）**，让 RM 学到“偏离后如何回归”。  
- **方案 B（脚本/HITL）**：脚本负责把系统推进到高密度接触区间，人类只在关键时刻介入；这些“介入点”天然就是 RM 的高价值训练数据（边界态）。  
- **方案 C（Bootstrap）**：把 RM 当作过滤器与教练：
  - **在线过滤**：当 \(\hat{q}_t\) 太低或 \(\hat{c}_t\) 太低，触发安全策略/人工接管  
  - **自动收集 hard cases**：把 RM 判为“低质但可恢复”的片段打上优先级，进入后续训练池

### D) Streaming 自动切段（解决“没有 episode 边界”）

DYNA 明确提到 continuous deployment 数据没有天然 episode 边界，并发展了“自动分段 + subtask labeling”的做法（见其 DYNA-1 页面）。对 DexHand，你可以用 RM 的 `progress` 与 `failure_mode` 做事件分割：

- **start**：首次进入接触（`effort`/`contact_flag` 上升）
- **subtask switch**：\(\hat{p}_t\) 斜率变化 / `failure_mode` 切换（例如从 `approach` 到 `grasp` 到 `recover`）
- **end**：物体稳定放置/任务完成；或 recoverability 很低并触发人工接管

最终把“连续流”变成可训练的 episode：
- `episode.boundaries = [(t0,t1), (t1,t2), ...]`
- `episode.subtask = ["approach","pinch","rotate","recover",...]`

### E) 最小可落地：先用规则做 RM 的伪标签（工程强可行）

如果你现在没有足够标注训练 RM，仍可快速启用 RM-in-loop 的思路：

- 用规则生成 `failure_mode`：例如 `motor_current` 异常升高 + `joint_velocity` 近 0 → `jam`
- 用可复现性生成 `quality`：如你在 3.3 的 replay validation，能 replay 成功则 \(\hat{q}\) 高
- 用滑移 proxy 生成 `recoverability`：`effort` 下降 + 视觉抓取框漂移 → `slip`（可恢复窗口）

这能让你“先跑起来”，再用累积的数据训练更强的 RM。

### F) 一个贴近真实的最小案例：DexHand 抓起桌面“手机/卡片”（薄片光滑物体）

这个任务**很贴近真实落地**：薄、滑、和桌面有真实碰撞与对抗。你很容易采到“失败片段”（滑、卡、夹歪、顶到桌面），因此非常适合用规则生成伪标签，让 RM-in-loop 先跑起来。

#### 任务与传感最小集合（不靠额外外设也能做）

- **任务**：把手机（或卡片）从桌面抓起，抬离桌面 \(h \ge 2\text{cm}\)，稳定保持 2 秒。
- **观测（最小）**：
  - `rgb_wrist`（腕相机，判断薄片边缘、是否离桌）
  - `joint_positions`, `joint_velocities`
  - `motor_current` 或 `joint_effort`（必须有一个；用来判别接触/卡死/滑移）
- **控制**：
  - `joint_targets`（位置控制即可；推荐限速 + 电流阈值保护）

> 为什么这个集合“够用”：薄片抓取的本质是“边缘接触 + 力学对抗”，电流/扭矩是你最便宜的“触觉替身”。

#### 子任务分段（streaming → episode/subtask）

用“事件”把连续流切成 4 段（可直接作为 RM 的 progress 伪标签）：

- **S0 approach**：手指张开，靠近桌面上目标边缘（`motor_current` 低）
- **S1 wedge/contact**：指尖接触桌面/物体边缘，出现轻微电流上升
- **S2 pinch + micro-lift**：缓慢闭合并尝试抬离（电流上升 + 关节速度下降）
- **S3 lift + hold**：物体离桌并稳定保持（视觉上有离桌间隙，电流稳定）

#### 伪标签规则（可实现、可调参）

下面这些规则都能用现有信号实现，且与“物理现象”强对应：

- **success（episode 级标签）**：
  - `lift_height >= 2cm` 且 `hold_time >= 2s`
  - `lift_height` 可以先用 `rgb_wrist` 做一个很粗的“离桌判别”（例如桌面边缘线与物体下缘的像素间隙 > 阈值）
- **failure_mode（步级/段级标签）**：
  - `jam`（卡死/顶住）：
    - `motor_current` 高于阈值（例如 > P95）持续 \(> 200\text{ms}\)
    - 且 `joint_velocities` 近 0（例如 \(\|\dot{q}\|_\infty < \epsilon\)）
  - `slip`（滑移）：
    - 处于 S2/S3
    - `motor_current` 明显下降（夹持力掉了）或闭合继续增加但 `lift_height` 无提升
  - `table_collision_risk`（桌面强对抗风险）：
    - 腕相机看到“手指/物体”与桌面边界重叠增加（粗规则即可）
    - 且 `motor_current` 出现尖峰（spike）
- **recoverability（可自救窗口）**：
  - `recoverable = 1`：`slip` 发生但电流未到危险阈值，且仍能看到物体（未完全丢失）
  - `recoverable = 0`：`jam` 且持续时间长、或物体完全离开视野/掉落
- **quality（生产级质量伪标签）**：
  - 先用最硬核的工程判据：**Replay Validation**
    - 把 `joint_targets` 在真机上重放
    - 若 “成功 + 轨迹无保护触发（无 jam spike）” → `quality=high`
    - 若 “首次成功但重放失败” → `quality=low`（典型是延迟/同步问题导致不可复现）

#### RM-in-loop 怎么用在“采数当天”（立刻可用）

1) **在线过滤（安全 + 省时间）**
- 若检测到 `jam` → 立即停止闭合，执行“回退 5mm + 重新对齐 approach”
- 若 `table_collision_risk` → 降速（速度上限减半），或切换到“更高抬手高度再接触”

2) **自动挑 hard cases（让数据更值钱）**
- 把 `recoverable=1` 的 `slip` 片段单独存成 `recovery_clip`（训练“纠偏”最有用）
- 把“成功但质量低（重放失败）”的 episode 标记为 `sync_suspect=1`，回头优先排查时间戳/对齐

3) **从伪标签到真 RM（几天内可迭代）**
- 第 1 周：完全用规则跑闭环（先把系统做稳）
- 第 2 周：用规则生成的 `(progress, failure_mode, recoverability, quality)` 训练一个小 RM（例如轻量 transformer/1D temporal CNN）
- 第 3 周：RM 替换部分规则（减少手工阈值依赖），继续收集更多“边界态”数据

> 这就是 DYNA “用可规模化信号把部署变成训练”的最小翻译版：先用规则把闭环跑通，再逐步把规则学习化。

## 5. 参考来源与论文

### 平台与硬件方案
- **RAPID Hand (2025)**: 低成本、全感知集成、低于 7ms 延迟的灵巧数采平台。 [arXiv:2506.07490](https://arxiv.org/abs/2506.07490)
- **MagiClaw (2025)**: 基于视觉的软体灵巧手，支持 iPhone POV 数采。 [arXiv:2509.19169](https://arxiv.org/abs/2509.19169)
- **SmartHand (2021)**: 嵌入式触觉感知系统与大规模触觉数据集。 [arXiv:2107.14598](https://arxiv.org/abs/2107.14598)
- **Soft Humanoid Hand (2020)**: 指端视觉感知（In-finger Vision）方案。 [arXiv:2006.03537](https://arxiv.org/abs/2006.03537)

### 算法与数据集
- **Learning Hand-Eye Coordination (Levine et al., 2016)**: 大规模（80万次抓取）真实机器人数据采集鼻祖。 [arXiv:1603.02199](https://arxiv.org/abs/1603.02199)
- **DexGraspNet (2022)**: 基于仿真生成的超大规模（130万+）灵巧抓取数据集. [arXiv:2210.02697](https://arxiv.org/abs/2210.02697)
- **Rotating without Seeing (2023)**: 仅依赖触觉（无需视觉）的手内操作数据集与策略. [arXiv:2303.10880](https://arxiv.org/abs/2303.10880)
