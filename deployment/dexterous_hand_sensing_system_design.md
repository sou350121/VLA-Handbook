# 灵巧手 DexHand 的 Sensing System 设计（面向部署 × 采数 × 自恢复）

> 这是一份“可落地的系统设计文档”：从 **任务→力学因果→可观测信号→多频闭环→数据闭环（RM-in-loop）** 一次讲清楚，并给出从 MVS 到“超越人类能力”的路线图。

---

## 目录

- 1. 系统目标与 KPI
- 2. 总体架构（传感栈 × 多频闭环 × 数据闭环）
- 3. 传感栈分层（MVS→Pro）与最佳分布
- 4. 三层闭环（Level-A/B/C）与职责边界
- 5. 人形视角：手是“物理因果接口”（力学原语、接触模式、滑移）
- 6. RM-in-loop：让系统自动切段、挑 hard cases、触发自恢复
- 7. 时间同步与数据管线（可训练、可复现、可追责）
- 8. 标定、健康监测与可维护性
- 9. 最小落地版本 + 迭代路线图（走向超越人类）

---

## 1. 系统目标与 KPI（把方向锁死）

DexHand 的 sensing system 不应追求“传感器最多”，而应追求 **长时稳定 + 可自救 + 数据可复现**：

- **稳定性 KPI**：连续运行 \(T\) 分钟内，`recover_event` 次数、`human_intervention` 次数、`drop` 次数。
- **质量 KPI**：同一轨迹 **Replay 可复现率**（详见 [多模态同步](./multimodal_data_synchronization.md)）。
- **效率 KPI**：单位时间可采集的“有效 episode”数量（排除不可复现/不可对齐的废数据）。
- **安全 KPI**：电流/温度超限次数、近碰撞次数（软硬限位触发统计）。

> 直觉：DexHand 的工程胜负手不是单次成功率，而是“长时稳定 + 快速自救 + 可复现数据飞轮”。

---

## 2. 总体架构（传感栈 × 多频闭环 × 数据闭环）

三件事必须同时成立：

1) **传感栈分层**：高频低带宽信号兜底、少量高信息传感器负责“理解接触”。  
2) **多频闭环分工**：1kHz 止损、100Hz 防滑、10Hz 原语切换。  
3) **数据闭环**：把部署数据自动切段、打标签、挑 hard cases，再反哺训练（RM-in-loop）。

```
           ┌──────────────┐
           │  Level-C     │  5–15Hz  语义策略 / 原语切换（push/pull/wedge/twist…）
           └──────┬───────┘
                  │  下发参数/子目标
                  ▼
           ┌──────────────┐
           │  Level-B     │  60–200Hz  触觉协调（防滑、夹持力调度、微调接触几何）
           └──────┬───────┘
                  │  夹持/姿态微调
                  ▼
           ┌──────────────┐
           │  Level-A     │  500–1000Hz  安全反射（限流、卡死、退避）
           └──────┬───────┘
                  │  电机/关节命令
                  ▼
             DexHand 硬件

  数据闭环（旁路）：timestamp→对齐→episode切段→RM打分→hard cases→再训练
```

设计原则（压缩成 5 条）：

- **P1：Timestamp-at-source**：所有模态在“产生瞬间”打统一时钟戳（PTP 优先）（见 [多模态同步](./multimodal_data_synchronization.md)）。
- **P2：多频闭环**：不要用 30Hz 视觉去硬做 1kHz 接触稳定；必须分层。
- **P3：接触可见**：仅靠 RGB 很难判断“滑/卡/夹歪”；必须引入 **电流/扭矩/触觉 proxy**。
- **P4：可维护优先**：触觉（尤其视触觉）是易损耗件（见 [触觉集成挑战](./tactile_sensor_integration_challenges.md)）。
- **P5：先做 MVS**：先跑通“可复现采数 + 自恢复”，再升级传感器，而不是一开始堆满硬件。

---

## 3. 传感栈分层（MVS→Pro）与最佳分布

### 3.1 Level-0（MVS：最小可行传感）

目标：**低成本获得接触态判别能力 + 可复现数据**。

- **视觉**：`rgb_static` + `rgb_wrist`（腕相机强烈建议）
- **本体**：`q`、`\dot{q}`
- **接触代理（必选其一）**：`motor_current` 或 `joint_effort`
- **事件信号（软件生成）**：`contact_flag`、`safety_trip`

这套配置已经能支撑：
- **卡死/顶住检测**（强可行）
- **接触开始/夹持强弱的粗闭环**（强可行）
- **Replay Validation**（决定数据能不能用于训练）

### 3.2 Level-1（触觉层：让“滑移/接触几何”可观测）

先讲结论：为了兼顾 KPI，最推荐的空间分布是 **“2 + 1 + 1”**：

- **2 个指尖做高频剪切/压力阵列（100–200Hz）**：拇指 + 食指（防滑闭环主力）
- **1 个指尖做高信息几何（视触觉 30–60Hz）**：拇指或食指（接触点/面积/边缘/微滑动）
- **1 个手腕 6D F/T（>=200Hz）**：推/拉/扭/撬的因果判定（外部力矩/约束反力）

为什么不是“全指视触觉”：全铺会把带宽/布线/耐久/标定维护 KPI 拉爆；而用“少量高信息老师 + 少量高频工兵 + 全关节兜底”，能在信息量与工程可维护之间达到最优点（见 [触觉集成挑战](./tactile_sensor_integration_challenges.md)）。

### 3.3 Level-2（工程化 Pro：长时部署与健康管理）

- **热与健康监测**：`temp_motor_i`、`over_current_count`、`stall_time_ms`
- **冗余观测**：关键关节 `effort/current` 的统计漂移（用于 `sensor_health`）

---

## 4. 三层闭环（Level-A/B/C）与职责边界

### 4.1 Level-A：安全/反射层（500–1000Hz，必须实时）

目标：防损坏、防卡死、防过热、毫秒级止损。

- 输入：`motor_current/joint_effort`、`\dot{q}`、温度、软限位
- 输出：限流/限速/急停、微退避（backoff）
- 典型规则：`current > I_hard` 且 `|\dot{q}| < eps` 持续 \(>200\)ms → `jam` → backoff

### 4.2 Level-B：触觉协调层（60–200Hz）

目标：防滑（slip control）、夹持力调度、微调接触几何。

- 输入：**剪切/压力阵列（主）** + `effort/current`（冗余） + （可选）视触觉特征
- 输出：夹持力（或关节闭合量）微调、接触姿态微调
- 典型能力：在物体刚开始滑时增加夹持、或调整接触点

### 4.3 Level-C：语义策略层（5–15Hz）

目标：任务理解、子任务切换、与 VLA/LLM 的高层决策对接。

- 输入：`rgb_static/rgb_wrist`（以及可选语言指令/任务状态） + （可选）手腕 F/T 统计特征
- 输出：子目标（grasp pose / open-close schedule / retreat policy）、以及对 Level-A/B 的参数下发（夹持力上限、速度上限、恢复路径）

> 关键点：Level-C 不直接管“接触瞬间怎么稳住”，它管的是“什么时候该进入接触、失败后走哪条恢复路径”。

---

## 5. 人形视角：手是“物理因果接口”（力学原语、接触模式、滑移）

对人形机器人来说，手的职责是：把意图转成物理世界会响应的原因（力、力矩、接触约束）。

### 5.1 Wrench（力 + 力矩）是任务语义的最小接口

对物体的直接作用可以写成 6 维 wrench：

$$
w = \begin{bmatrix} f \\ \tau \end{bmatrix}
= \begin{bmatrix} F_x & F_y & F_z & \tau_x & \tau_y & \tau_z \end{bmatrix}^T
$$

push/pull/press/twist/pry 本质是在控制 \(w\) 的不同分量。

### 5.2 接触模式 + 摩擦边界（滑移判定的物理本质）

接触模式可粗分：sticking / sliding / rolling(pivot)。

摩擦边界（简化）：

$$
\|f_t\| \le \mu f_n
$$

滑移检测就是在估计：是否进入 sliding，以及离边界有多近（`slip_risk`）。

### 5.3 力学原语（Force Primitives）：推/拉不是语义，而是物理策略

- `push / pull / press / drag / wedge / twist / pry`

> 所以“推/拉”不是纯语义选择，而应由：可用接触点、摩擦预算、可用力矩、以及失败模式（滑/卡/顶）共同决定。

### 5.4 为了做出这些决策：四类必备可观测信号

- **法向力代理 \(f_n\)**：`motor_current/joint_effort`（最低配）→ 压力阵列（更好）
- **切向/剪切趋势 \(f_t\)**：剪切触觉阵列最直接；没有触觉时只能弱依赖视觉 + 电流残差
- **接触几何**：接触面积/接触点位置（视触觉最强；低分辨率阵列次之）
- **外部力矩/约束反力**：手腕 F/T 最直接（尤其对拧/撬/推拉门类任务）

---

## 6. RM-in-loop：让系统自动切段、挑 hard cases、触发自恢复

把 RM 当成“质量裁判”，让系统在部署时不仅能做任务，还能：

- 自动切段（streaming → episode/subtask）
- 自动挑选 hard cases（对训练最值钱）
- 自动触发恢复策略（减少人工介入）

最小建议：RM 做多头输出，而不是只输出一个标量：

- `progress`：子任务进度（可用规则先做伪标签）
- `quality`：生产级质量（Replay 成功 + 无保护触发 = 高）
- `recoverability`：是否处于可自救窗口
- `failure_mode`：`slip/jam/occlusion/...`
- （强建议加）`primitive/contact_mode/slip_risk`：把物理因果写进标签体系

> 这套思路与 DYNA 的公开叙事一致：用 RM 支撑 long-horizon 的鲁棒运行与数据闭环（参考 [DYNA-1 Research](https://www.dyna.co/dyna-1/research) 及 [DYNA Series A 博文](https://www.dyna.co/blog/dyna-robotics-closes-120m-series-a)）。

---

## 7. 时间同步与数据管线（可训练、可复现、可追责）

- **统一时钟**：优先 PTP（IEEE 1588）
- **Timestamp-at-source**：设备端产生数据时立即打戳（见 [多模态同步](./multimodal_data_synchronization.md)）
- **Ring Buffer 对齐**：以视觉帧为 anchor（30–60Hz），对齐 `q,\dot{q},effort/current,tactile`
- **记录两条动作流（非常重要）**：
  - 命令流：`cmd_joint_target`
  - 执行流：`measured_q, measured_effort`

否则你无法判断失败到底是“策略错”，还是“执行/延迟/饱和/保护触发”导致的。

---

## 8. 标定、健康监测与可维护性

- **相机与手眼标定**：见 [相机标定与手眼对齐](./camera_calibration_eye_in_hand.md)
- **触觉标定**：零点漂移、材料老化是常态（见 [触觉集成挑战](./tactile_sensor_integration_challenges.md)）
- **可落地做法**：
  - 每天/每班次零点校准（baseline）
  - 每次更换蒙皮/指尖模块：`recalib_required=1`
  - 把健康状态写成可观测事件：`sensor_health=degraded`

---

## 9. 最小落地版本 + 迭代路线图（走向超越人类）

### 9.1 最小落地版本（你今天就能做出来）

如果你只有 DexHand + 2 个相机（静态 + 腕部），没有触觉：

- **硬件**：wrist RGB（30–60Hz）、static RGB（30Hz）、`effort/current`（>=200Hz）、`q,\dot{q}`（>=200Hz）
- **软件**：PTP/统一时钟、Ring Buffer 对齐、Level-A 安全反射、Replay Validation
- **数据**：命令流 + 执行流、episode metadata（`calib_version/sync_status/replay_ok`）

### 9.2 路线图：从 MVS 到“超越人类能力”

“超越人类”不是堆传感器，而是定义超能力并做系统闭环：

- **超人类带宽**：1kHz 反射 + 100–200Hz 防滑
- **超人类动态范围**：从轻触到强对抗，实时管理 `slip_risk`
- **超人类可解释性**：输出 `contact_mode/wrench_est/constraint_state`
- **超人类接触几何理解**：1 个高信息“老师”（视触觉）+ 2–3 个高频“工兵”（剪切/压力）+ effort/current 兜底
- **超人类自标注飞轮**：RM-in-loop 自动切段/挑 hard cases/触发自恢复
- **超人类鲁棒性**：漂移/磨损/温升都进入 `sensor_health` 与降额逻辑

建议迭代节奏：
- **阶段 1（1–2 周）**：MVS + Replay Validation + 规则 RM
- **阶段 2（2–4 周）**：把规则伪标签训练成轻量 RM（替换阈值）
- **阶段 3（1–2 月）**：加入 2 个剪切/压力指尖（Level-B 变强）
- **阶段 4（长期）**：加入 1 个视触觉 + 1 个手腕 F/T（建立“因果判定 + 教师传感”）

---

## 参考链接（与本仓库内容的关系）

- 多模态同步方法论：[`multimodal_data_synchronization.md`](./multimodal_data_synchronization.md)
- 触觉集成工程难点：[`tactile_sensor_integration_challenges.md`](./tactile_sensor_integration_challenges.md)
- DexHand 采数与 RM-in-loop 落地案例：[`dexterous_hand_data_collection.md`](./dexterous_hand_data_collection.md)
