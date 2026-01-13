# 灵巧手 DexHand 的 Sensing System 设计（面向部署 × 采数 × 自恢复）

> 目标：设计一套 **可落地、可维护、可规模化采数** 的 DexHand 传感系统，让模型不仅“会做”，还能 **长时稳定运行（no-reset / no-intervention）**，并把部署数据持续转化为训练数据闭环。

---

## 0. 先定“系统 KPI”（否则传感会越堆越乱）

灵巧手的 sensing system 不应追求“传感器最多”，而应追求 **能支撑闭环与数据质量**。建议把 KPI 写成可测量项：

- **稳定性 KPI**：连续运行 \(T\) 分钟内，`recover_event` 次数、`human_intervention` 次数、`drop` 次数。
- **质量 KPI**：同一轨迹 **Replay 可复现率**（详见 [多模态同步](./multimodal_data_synchronization.md)）。
- **效率 KPI**：单位时间可采集的“有效 episode”数量（排除不可复现/不可对齐的废数据）。
- **安全 KPI**：电流/温度超限次数、近碰撞次数（软硬限位触发统计）。

> 直觉：DexHand 的工程胜负手不是单次成功率，而是“长时稳定 + 可自救 + 可复现数据”。

---

## 1. 设计原则（把复杂系统压缩成 5 条）

- **P1：Timestamp-at-source**：所有模态在“产生瞬间”打统一时钟戳（PTP 优先），避免到 PC 才打戳造成不可逆偏差（见 [多模态同步](./multimodal_data_synchronization.md)）。
- **P2：多频闭环**：不要幻想用 30Hz 视觉闭环解决 1kHz 接触不稳定；必须分层（见下文第 3 节）。
- **P3：接触可见**：仅靠 RGB 很难判断“滑/卡/夹歪/打滑”；必须把 **电流/扭矩/触觉 proxy** 纳入观测（这一点对 RM 更关键）。
- **P4：可维护优先**：触觉（尤其视触觉）是易损耗件，系统设计必须考虑快拆、布线寿命、标定频率（见 [触觉集成挑战](./tactile_sensor_integration_challenges.md)）。
- **P5：先做 MVS（Minimum Viable Sensing）**：先跑通“可复现采数 + 自恢复”，再升级传感器，而不是一开始堆满硬件导致不可维护。

---

## 2. 传感栈分层（从 MVS 到 Pro）

### 2.1 Level-0（MVS：最小可行传感）

这一层目标：**用最低成本获得接触态判别能力 + 可复现数据**。

- **视觉**：
  - `rgb_static`（外部/俯视/侧视任一）
  - `rgb_wrist`（腕相机，强烈建议）
- **本体**：
  - `q`（关节位置）、`\dot{q}`（关节速度）
- **接触代理（必选其一）**：
  - `motor_current` 或 `joint_effort`
- **事件信号（软件生成即可）**：
  - `contact_flag`（由电流/扭矩阈值化）
  - `safety_trip`（软限位/电流限/温度限触发）

这套配置已经能支撑：
- 抓取/旋拧/捏取的 **滑移/卡死** 检测
- **Replay Validation**（判断数据是否可复现）
- RM 的最小伪标签（failure_mode / recoverability）

### 2.2 Level-1（加触觉：解决“视觉看不见的接触细节”）

优先级建议：

- **先上“低带宽触觉”再上“高带宽视触觉”**：
  - MEMS 压力阵列 / 电阻阵列：帧率更高、带宽更好、布线更容易
  - 视触觉（GelSight/DIGIT）：信息密度高，但带宽/布线/标定/耐久都更难（见 [触觉集成挑战](./tactile_sensor_integration_challenges.md)）
- **布点策略**（灵巧手很关键）：
  - 指尖（distal）优先：滑移与接触几何最敏感
  - 指腹/掌心其次：包覆抓取与大物体支撑

### 2.3 Level-2（工程化 Pro：为“长时部署与健康管理”补齐）

- **热与健康监测**：
  - 电机/驱动板温度 `temp_motor_i`
  - 估算磨损指标：`over_current_count`, `stall_time_ms`
- **力/扭矩（可选）**：
  - 手腕 F/T 或指端力传感（用于更可靠的 slip/press 判别）

---

## 3. 三层闭环（频率预算 + 每层用什么信号）

建议把系统拆成 3 个“时间尺度”，每层各司其职：

### 3.1 Level-A：安全/反射层（500–1000Hz，必须实时）

**目标**：防损坏、防卡死、防过热、毫秒级止损。

- 输入：`motor_current/joint_effort`、`\dot{q}`、温度、软限位
- 输出：限流/限速/急停、微退避（backoff）
- 典型规则：
  - `current > I_hard` 且 `|\dot{q}| < eps` 持续 \(>200\)ms → `jam` → backoff

### 3.2 Level-B：触觉协调层（60–200Hz）

**目标**：防滑（slip control）、夹持力调度、微调接触几何。

- 输入：触觉阵列 / 视触觉特征 / 电流 proxy
- 输出：夹持力（或关节闭合量）微调、接触姿态微调
- 典型能力：在物体刚开始滑时增加夹持、或调整接触点

### 3.3 Level-C：语义策略层（5–15Hz）

**目标**：任务理解、子任务切换、与 VLA/LLM 的高层决策对接。

- 输入：`rgb_static/rgb_wrist`（以及可选语言指令/任务状态）
- 输出：子目标（grasp pose / open-close schedule / retreat policy）、以及对 Level-A/B 的参数下发（例如夹持力上限、速度上限）

> 关键点：Level-C 不直接管“接触瞬间怎么稳住”，它管的是“什么时候该进入接触、失败后走哪条恢复路径”。

---

## 3.4 从“人形”角度看手的角色：手是「物理因果接口」

对人形机器人来说，手不是“夹住物体就完事”，而是负责把意图转成**物理世界会响应的原因**：施加力、力矩、接触约束，进而改变物体的速度、姿态、接触状态。

如果不先归纳“手如何运用力影响物体”，系统很容易出现两类典型失败：
- **任务语义与控制量脱节**：策略说“拉开/推开/拧开”，底层却只有“关节角变化”，缺少可解释的力学接口。
- **感知信号不够用**：你想判断“是推还是拉”，本质是在判断“接触约束 + 摩擦预算”下，哪种力学原语能让物体产生你要的响应。

### A) 把“手的动作”抽象成末端 Wrench（力 + 力矩）

对物体的直接作用可以写成 6 维 wrench：

$$
w = \begin{bmatrix} f \\ \tau \end{bmatrix}
= \begin{bmatrix} F_x & F_y & F_z & \tau_x & \tau_y & \tau_z \end{bmatrix}^T
$$

它比“关节角”更贴近任务语义：
- **push / pull**：主要改变切向力分量（$F_x,F_y$）
- **press**：主要控制法向力（$F_z$）
- **twist / turn**：主要控制力矩（$\tau$）
- **lever / pry（撬动/指甲）**：需要“接触点 + 力臂”，本质是输出力矩

> 系统设计上：Level-C 的高层意图，最好能落到“目标 wrench / 目标接触模式”，再由 Level-A/B 实现。

### B) “判定物件滑动”的本质：识别接触模式与摩擦边界

接触时可粗分三种模式（它们决定你该怎么用力）：
- **Sticking（粘着/不滑）**：切向力会转化为物体运动（推/拉/搬运）
- **Sliding（滑动）**：切向力超过摩擦能力，接触点在滑（需要增大夹持或改变接触几何）
- **Rolling / Pivot（滚动/转轴）**：接触点不滑，但物体绕边/点旋转（翻边、拨动、旋拧中的“定位阶段”）

最核心的物理边界是摩擦锥（简化）：

$$
\|f_t\| \le \mu f_n
$$

- $f_n$：法向力（夹持/按压）
- $f_t$：切向力（推/拉所需摩擦）
- $\mu$：摩擦系数

**滑移检测**就是在估计：当前是否进入 sliding，以及离边界有多近（`slip_risk`）。

### C) “推还是拉”的决策：其实是在选择「力学原语」

很多人形手任务更像是在选一种“让物体按物理原因反应”的原语（Force Primitive）：

- **推（push）**：用切向力把物体推离你；依赖足够摩擦与可用接触面
- **拉（pull）**：需要可抓住/可挂住的接触点，把物体拉向你；更适合“越过障碍/从缝里拉出”
- **按（press）**：用法向力触发机构（按钮/卡扣）
- **拖（drag）**：允许受控滑动，用较低 $f_n$ 维持“可控摩擦”
- **楔入（wedge）**：先用几何制造接触，再把模式从 sliding 变成 sticking（薄片/手机是典型）
- **扭（twist/turn）**：需要稳定 $f_n$ + 足够 $\tau$；否则表现为“拧不动但打滑”
- **撬（lever/pry）**：核心是力臂与力矩输出（小空间大力矩输出、指甲类动作）

> 所以“推/拉”不是纯语义选择，而应由：可用接触点、摩擦预算、可用力矩、以及失败模式（滑/卡/顶）共同决定。

### D) 为了做出这些决策，你需要哪些“可观测信号”（对应 sensing 取舍）

要让手能判定“滑动/接触模式/该推还是拉”，至少需要四类可观测量：
- **法向力代理 $f_n$**：`motor_current/joint_effort`（最低配），最好有指尖压力阵列
- **切向/剪切趋势 $f_t$**：剪切触觉阵列最直接；没有触觉时只能弱依赖视觉 + 电流残差
- **接触几何**：接触面积/接触点位置（视触觉最强；低分辨率阵列次之）
- **外部力矩/约束反力**：手腕 F/T 最直接（尤其对拧/撬/推拉门类任务）

这也解释了“高频低带宽（电流/压力/剪切）+ 低频高信息（视触觉或 F/T）”的组合：前者支撑 Level-B 快控制，后者支撑“因果判定 + 数据闭环”。

### E) 落到 VLA/RM-in-loop：把“力学原语”写成数据与标签

一旦你把力学原语库写进数据/标签体系，模型学到的就不只是动作轨迹，而是“物理因果”：
- `contact_mode`: `sticking/sliding/rolling/pivot`
- `primitive`: `push/pull/press/drag/wedge/twist/pry`
- `slip_risk`: 0~1（离摩擦边界的风险）

RM 的价值也就更清晰：它不只是评估成功/失败，而是评估“接触模式是否正确、滑移风险是否上升、是否需要切换原语（例如从 push 切到 wedge）”。

---

## 4. 时间同步与数据管线（让数据“可训练、可复现、可追责”）

### 4.1 统一时钟与时间戳

- **统一时钟**：优先 PTP（IEEE 1588），至少做到“控制板/相机/主机”同一时钟域。
- **Timestamp-at-source**：每个传感器在产生数据的那一刻打戳，而不是 ROS 收到时打戳（见 [多模态同步](./multimodal_data_synchronization.md)）。

### 4.2 Ring Buffer 对齐（主频建议以视觉为 anchor）

训练样本通常以视觉帧为 anchor（30–60Hz），为每帧 \(t\) 取：
- 最近邻或插值后的 `q(t), \dot{q}(t), effort(t)`
- 触觉特征（若有）`tactile(t)`
- 动作标签 `a(t)`（来自示教/策略输出）

这类“对齐”不只是为了训练，也直接决定了 **Replay 可复现率**。

### 4.3 记录两条动作流（非常重要）

DexHand 的“动作”应同时记录：
- **命令流**：`cmd_joint_target`（你下发的目标）
- **执行流**：`measured_q, measured_effort`（实际执行结果）

否则你无法判断失败到底是“策略错”，还是“执行/延迟/饱和/保护触发”导致的。

---

## 5. 标定与一致性（把“传感器漂移”当成必然）

### 5.1 相机与手眼标定

- 参考：[相机标定与手眼对齐](./camera_calibration_eye_in_hand.md)
- **建议**：把标定结果作为版本化资产管理：`calib_version`, `calib_hash`，并写入每条 episode 的 metadata（便于追责）。

### 5.2 触觉标定（如果做 Level-1）

触觉（尤其软材料/视触觉）会磨损、老化、零点漂移，这是常态（见 [触觉集成挑战](./tactile_sensor_integration_challenges.md)）。

可落地做法：
- 每天/每班次做一次 **零点校准**（无接触时的 baseline）
- 每更换蒙皮/指尖模块，强制 `recalib_required=1`
- 把“标定失败/漂移超限”写成可观测事件：`sensor_health=degraded`

---

## 6. RM-in-loop 如何嵌入 DexHand sensing system（从伪标签到可部署 RM）

把 RM 当成“质量裁判”，让系统在部署时不仅能做任务，还能：
- 自动切段（streaming → episode）
- 自动挑选 hard cases（对训练最值钱）
- 自动触发恢复策略（减少人工介入）

### 6.1 RM 的最小输入（不需要昂贵触觉也能跑）

- `rgb_wrist`
- `q, \dot{q}`
- `effort/current`
- 事件：`contact_flag`, `safety_trip`

### 6.2 RM 的最小输出（建议多头）

- `progress`：子任务进度（可用规则先做伪标签）
- `quality`：生产级质量（Replay 成功 + 无保护触发 = 高）
- `recoverability`：是否处于可自救窗口
- `failure_mode`：`slip/jam/occlusion/...`

> 这套写法与 DYNA 的公开叙事一致：用 RM 支撑 long-horizon 的鲁棒运行与数据闭环（参考 [DYNA-1 Research](https://www.dyna.co/dyna-1/research) 及 [DYNA Series A 博文](https://www.dyna.co/blog/dyna-robotics-closes-120m-series-a)）。

---

## 7. 最小落地版本（你今天就能做出来的 DexHand sensing system）

如果你只有一个 DexHand + 2 个相机（静态 + 腕部），没有触觉：

- **硬件**：
  - wrist RGB（30–60Hz）
  - static RGB（30Hz）
  - 电机电流/关节 effort（>=200Hz，越高越好）
  - 关节位置速度（>=200Hz，越高越好）
- **软件**：
  - PTP/统一时钟 + Timestamp-at-source
  - Ring Buffer 对齐（视觉 anchor）
  - Level-A 安全反射（限流/卡死检测/退避）
  - RM 伪标签（规则 + Replay）先跑闭环
- **数据**：
  - 记录命令流 + 执行流
  - 每条 episode 写入 `calib_version`、`sync_status`、`replay_ok`

> 这就是“先把闭环跑通，再升级触觉”的工程路径：先把系统做稳、做可复现、做可规模化采数。

---

## 8. 迭代路线图（从 MVS 到 Pro）

- **阶段 1（1–2 周）**：MVS + Replay Validation + 规则 RM
- **阶段 2（2–4 周）**：把规则伪标签训练成轻量 RM（替换阈值）
- **阶段 3（1–2 月）**：加入指尖触觉（Level-1），把 slip/recovery 做成可泛化能力
- **阶段 4（长期）**：健康管理 + 自动维护（传感器漂移检测、耗材更换策略）

---

## 参考链接（与本仓库内容的关系）

- 多模态同步方法论：[`multimodal_data_synchronization.md`](./multimodal_data_synchronization.md)
- 触觉集成工程难点：[`tactile_sensor_integration_challenges.md`](./tactile_sensor_integration_challenges.md)
- DexHand 采数与 RM-in-loop 落地案例：[`dexterous_hand_data_collection.md`](./dexterous_hand_data_collection.md)

