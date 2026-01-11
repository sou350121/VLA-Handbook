# 灵巧手数据采集方案 (Dexterous Hand Data Collection)

在具身智能（Embodied AI）尤其是 VLA (Vision-Language-Action) 模型的训练中，数据质量是决定模型上限的核心因素。相比平行夹爪，灵巧手（Dexterous Hand）由于其高自由度和复杂的接触动力学，对数据采集的精度、同步性和多模态覆盖有更高要求。

## 1. 数据单元定义 (Episode & Timestep)

高质量的数据集必须遵循严格的结构化定义，以便于行为克隆（Behavior Cloning）或扩散策略（Diffusion Policy）的学习。

### Episode (轨迹級) 结构
- **Episode ID**: 唯一标识。
- **Task Description**: 语言指令（如 "Pick up the bottle and unscrew the cap"）。
- **Success Label**: 二元标签（由人工标注或自动评测得出）。
- **Timesteps**: 一系列时序记录。

### Timestep (步級) 核心信号
| 信号类别 | 核心字段 | 说明 |
| :--- | :--- | :--- |
| **视觉 (Vision)** | `rgb_static`, `rgb_wrist` | 场景全局视角 + 腕部第一人称视角（POV）。 |
| **本体 (Proprio)** | `joint_positions`, `joint_velocities` | 灵巧手 N 个关节的绝对位置与速度。 |
| **示教 (Action)** | `joint_targets` | 专家示教的下一帧目标位置（通常是 $\Delta$ 增量或絕對位置）。 |
| **触觉 (Tactile)** | `pressure_map`, `contact_force` | 关键触碰点的压力分布（若硬件支持）。 |

---

## 2. 三大数采路线 (Acquisition Routes)

### 方案 A：专家遥操作 (Expert Teleoperation) — "Gold Standard"
- **原理**: 专家佩戴 VR 头显（如 Apple Vision Pro）或数据手套（Manus/SenseGlove），通过 Retargeting 算法实时驱动灵巧手。
- **优点**: 包含完整的示教逻辑，数据符合人类先验。
- **挑战**: 成本极高，Retargeting 的運動學映射（Mapping）可能存在伪影。

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

## 5. 参考来源与论文

### 平台与硬件方案
- **RAPID Hand (2025)**: 低成本、全感知集成、低于 7ms 延迟的灵巧数采平台。 [arXiv:2506.07490](https://arxiv.org/abs/2506.07490)
- **MagiClaw (2025)**: 基于视觉的软体灵巧手，支持 iPhone POV 数采。 [arXiv:2509.19169](https://arxiv.org/abs/2509.19169)
- **SmartHand (2021)**: 嵌入式触觉感知系统与大规模触觉数据集。 [arXiv:2107.14598](https://arxiv.org/abs/2107.14598)
- **Soft Humanoid Hand (2020)**: 指端视觉感知（In-finger Vision）方案。 [arXiv:2006.03537](https://arxiv.org/abs/2006.03537)

### 算法与数据集
- **Learning Hand-Eye Coordination (Levine et al., 2016)**: 大规模（80万次抓取）真实机器人数据采集鼻祖。 [arXiv:1603.02199](https://arxiv.org/abs/1603.02199)
- **DexGraspNet (2022)**: 基于仿真生成的超大规模（130万+）灵巧抓取数据集。 [arXiv:2210.02697](https://arxiv.org/abs/2210.02697)
- **Rotating without Seeing (2023)**: 仅依赖触觉（无需视觉）的手内操作数据集与策略。 [arXiv:2303.10880](https://arxiv.org/abs/2303.10880)
