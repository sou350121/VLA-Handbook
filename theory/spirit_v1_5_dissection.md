# Spirit-v1.5 模型解剖（Dissecting Spirit-v1.5）

> **写作风格说明**：本文刻意参考 `pi0_flow_matching.md` / `pi0_5_dissection.md` 的叙事逻辑：
> **Main Mathematical Idea → 架构信息流（ASCII）→ 数学/推理细节 → 代码入口走读 → 复现 checklist → 与 π0/π0.5 对比**。

- 官方仓库：[`Spirit-AI-Team/spirit-v1.5`](https://github.com/Spirit-AI-Team/spirit-v1.5)

---

## 0. 主要数学思想 (Main Mathematical Idea)

> **第一性原理**：把“动作生成”写成 **条件 ODE 的轨迹积分**，并辅以 **“脏数据”预训练带来的物理常识**。

Spirit-v1.5 的成功很大程度上归功于其 **多样化数据采集（Diverse Collection）** 范式。它打破了传统具身智能依赖“干净/精心编排”数据集（如 OXE）的束缚，主张从真实、凌乱、非脚本化的数据中学习。

- **Action Head**：它更像 π0 的 Flow Matching，让网络预测“速度场” $v_t$，通过 Euler 积分推回动作。
- **数据核心**：Spirit AI 认为“干净数据是伟大机器人基础模型的敌人”。其预训练数据包含大量失败-重试、任务切换和环境干扰，这赋予了模型极强的 **物理常识（Physical Common Sense）** 和恢复能力。

---

## 1. 核心架构：Qwen3-VL（大脑）+ DiT（小脑）+ ODE（执行）

Spirit-v1.5 的仓库目录结构中，明确写了主文件：
- `model/modeling_spirit_vla.py`：主模型架构（Qwen3-VL backbone + DiT head + policy API）
（见：[`Spirit-AI-Team/spirit-v1.5`](https://github.com/Spirit-AI-Team/spirit-v1.5)）。

### 1.1 整体信息流（ASCII）

```
                    Spirit-v1.5 端到端信息流（推理）
┌──────────────────────────────────────────────────────────────┐
│ 输入端：多视角图像 + 机器人状态 + 任务文本                    │
│  - observation.images.{cam_high,cam_left_wrist,cam_right_wrist}│
│  - observation.state                                          │
│  - task / robot_type                                          │
└───────────────┬──────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────┐
│ 1) 文本+视觉预处理（Qwen3-VL Processor + Prompt Template）     │
│  - 把多张 <image> placeholder 写进 prompt                      │
│  - 把 robot_type 与 task 组织成对话（chat_template）            │
└───────────────┬──────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────┐
│ 2) VLM Backbone：Qwen3VLForConditionalGeneration               │
│  - 输出 hidden_states（取最后 K 层拼接作为条件）                │
└───────────────┬──────────────────────────────────────────────┘
                │ cond = hidden_states_lastK
                ▼
┌──────────────────────────────────────────────────────────────┐
│ 3) Action Head：DiT（BaseDiT）                                 │
│  输入：state_emb + noisy_action_emb + time(t) + cond           │
│  输出：v_t（动作更新方向 / 速度场）                             │
└───────────────┬──────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────┐
│ 4) ODE Solver：Euler Integration（num_steps 次）               │
│  x <- x + dt * v_t                                             │
└───────────────┬──────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────┐
│ 5) 输出端：action chunk（反归一化）→ executor 做 robot-specific  │
│    后处理 → RoboChallenge 下发动作                              │
└──────────────────────────────────────────────────────────────┘
```

### 1.2 一组“最小必要的变量/张量形状”速查（读代码时非常省脑）

> 下面的形状/默认值来自 `model/modeling_spirit_vla.py` 的 `SpiritVLAConfig` 与默认 feature 定义；你在读 `executor.py` / `select_action()` 时可以对照。

- **三路图像**：默认 feature shape 是 **(3, 240, 320)**（`cam_high/left_wrist/right_wrist`）。推理时 batch 张量为 **(B, 3, 240, 320)**。  
  - 在 `executor.py` 里：`_img_byte_to_tensor(...) -> (3, 240, 320)`，再 `unsqueeze(0)` 变 **(1, 3, 240, 320)**。  
- **状态**：默认 `observation.state` feature shape 是 **(14,)**。  
  - 在 `prepare_state()` 中：若输入是 **(B, 14)** 会先变为 **(B, 1, 14)**，再 `pad_vector(..., max_state_dim=32)` → **(B, 1, 32)**。  
- **动作 chunk（采样空间）**：`n_action_steps=50`、`max_action_dim=32`，所以内部采样 `x_t` 是 **(B, 50, 32)**。  
  - 最后会裁剪到 `original_action_dim`（来自 `output_features["action"].shape`），再 Unnormalize 得到真实动作维度。  
- **ODE 迭代步数**：`num_steps=10`（即每次推理大约 10 次 Euler 更新）。  
- **DiT 默认规模**：`dit_hidden_size=1024`，`dit_num_heads=8`，`dit_num_layers=18`。  
  - 参考源码：[`model/modeling_spirit_vla.py`](https://raw.githubusercontent.com/Spirit-AI-Team/spirit-v1.5/main/model/modeling_spirit_vla.py)

---

## 2. 数学/推理细节：它到底“走了几步”，每一步做什么

下面完全按 `model/modeling_spirit_vla.py` 的实现口径解释。

### 2.0 推理时“条件”到底是什么（cond 的来源与意义）

Spirit-v1.5 的核心不是“直接从图像回归动作”，而是：

- 先用 Qwen3-VL 把多视角观测 + 任务文本编码成 **hidden states**
- 再让 DiT action head cross-attend 到这些 hidden states，把它们当成条件 \(\\mathrm{cond}\\)

从抽象上看，就是：

$$
v_\\theta = f_\\theta(x_t, t, \\underbrace{h_{\\text{VLM}}}_{\\mathrm{cond}})
$$

这里的 \(h_{\\text{VLM}}\) 是“这个时刻看到什么、要做什么、机器人是什么类型”的综合表征。

### 2.1 状态与动作的归一化：MIN_MAX → [-1, 1]

在 `SpiritVLAPolicy.select_action(...)` 内，先执行：
- `Normalize` inputs（state）
- `Normalize` targets（action space 的定义）

这一步依赖 MIN_MAX 统计量（min/max）。代码里 `Normalize(..., stats=None)`，但 forward 里会 assert min/max 不是 `inf`，因此这些 buffer 必须来自 checkpoint 的 `state_dict`。

- 对应实现：[`model/utils.py`](https://raw.githubusercontent.com/Spirit-AI-Team/spirit-v1.5/main/model/utils.py)

Normalize 的具体公式（源码逐行对应）是：

$$
x' = \\frac{x - x_{\\min}}{x_{\\max} - x_{\\min} + 10^{-8}}
$$

再把 \([0,1]\) 拉到 \([-1,1]\)：

$$
\\hat{x} = 2x' - 1
$$

#### 2.1.1 这一步为什么重要（工程直觉）

- **动作 head 的 ODE 迭代**默认在一个“相对规整”的数值空间里工作。把 state/action 统一缩放到 \([-1, 1]\)，可以显著降低不同机器人/任务尺度差异带来的数值不稳定。
- **坑点**：如果你迁移到自家机器人，最容易踩的是：`stats` 不完整或维度对不上 → Normalize buffer 里出现 `inf` → 推理直接 assert 失败。

### 2.2 条件 ODE 的离散化：Euler 积分

在 `_sample_actions_unified(...)` 内：
- `x_t` 初始化为高斯噪声（action chunk）
- `dt = -1.0 / num_steps`
- `time` 从 1.0 递减到 0 附近
- 每步调用 `_denoise_step(...)` 得到 `v_t`，然后更新：

$$
 x \leftarrow x + dt \cdot v_t
$$

这和 `pi0_flow_matching.md` 里介绍的“用 ODE solver 走 1~10 步”属于同一类推理范式。

#### 2.2.1 代码层“走几步”的确定方式

- `num_steps`：在 `SpiritVLAConfig` 里默认是 10（仓库实现）；也就是一次推理会执行大约 10 次更新。
- `dt = -1.0 / num_steps`：时间从 1.0 往 0 走，**负号**是关键（表示“从噪声流回数据”）。

#### 2.2.2 一段极简伪代码（对应 `_sample_actions_unified`）

```
state_emb = state_proj(pad(state))
cond = proj_vlm_output(vlm_hidden_lastK)

x = N(0, I)                    # noisy action chunk
t = 1.0
dt = -1 / num_steps

while t >= 0:
  suffix = concat(state_emb, action_in_proj(x))
  v = action_out_proj( DiT(suffix, cond, timestep=t) )
  x = x + dt * v
  t = t + dt

return x
```

### 2.3 读代码时最有用的“调用链速览”（ASCII callgraph）

```
scripts/run_robochallenge.sh
  └─ python -m robochallenge.run_robochallenge
      └─ RoboChallengeExecutor.infer(...)
          ├─ _prepare_batch(...)                 # observation -> model input dict
          ├─ policy.select_action(item)          # core inference API
          │   ├─ preprocess_rb_batch(...)        # images + state + task -> tensors
          │   ├─ Qwen3VLForConditionalGeneration # produce hidden_states (cond)
          │   └─ _sample_actions_unified(...)    # noisy chunk -> Euler steps -> chunk
          └─ _post_process_action(...)           # chunk -> robot specific action format
```

这张图的目的：你在仓库里搜这些函数名，基本就能把推理通路串起来。

### 2.4 多视角 + 任务文本：prompt 是怎么拼出来的（非常关键）

Spirit-v1.5 的 `preprocess_rb_batch(...)` 有两点非常“工程化”：

- **固定使用 3 个 `<image>` placeholder**（即便某个相机缺失，也会用全 0 placeholder 补上）
- **把 `robot_type` 写进 prompt**，让 VLM 明确“这是哪种机器人/动作语境”

源码里 user prompt 的模板是（逐字）：

```
<image> <image> <image>
The current robot type is {robot_type}. What is the current task?
```

然后它把 `TASK_INFO` 里的 `task` 当作 “assistant 的回答”（gpt turn），一起送入 `preprocess_qwen_visual(...)`。
直觉上，这等价于：让 VLM hidden states 同时吸收 **视觉证据 + 机器人类型 + 任务语义**，再交给 action head 去生成动作。

参考源码：[`model/modeling_spirit_vla.py`](https://raw.githubusercontent.com/Spirit-AI-Team/spirit-v1.5/main/model/modeling_spirit_vla.py)

### 2.5 cond 的构造：hidden_states 怎么取、怎么拼

在 `select_action(...)` 内：

1. 调用 `self.qwen.forward(..., output_hidden_states=True)` 得到 `vlm_outputs.hidden_states`（按层存的列表）。
2. `num_vlm_last_embd` 默认是 1（在 `BaseDiTConfig` 里设置），并取 `min(1, len(hidden_states))`。
3. 把最后 1 层沿 **序列维度**拼起来：

$$
h_{\\text{VLM}} = \\mathrm{cat}\\big(\\text{hidden\\_states}[-1:],\\ \\text{dim}=1\\big)
$$

随后进入 DiT 前会做一次投影：

- 如果 VLM hidden size == `dit_hidden_size`（1024）则 `proj_vlm_output = Identity()`
- 否则用 `Linear(vlm_hidden_size -> 1024)` 把条件投到 DiT 的宽度上

### 2.6 DiT head 如何用 cond（cross-attn / interleave self-attn）

Spirit-v1.5 的 `BaseDiT` 由一串 `BasicTransformerBlock` 组成。一个关键开关是：

- `dit_interleave_self_attention` 默认是 `False`。
  - 也就是：每一层都会把 `encoder_hidden_states=vlm_last_embed` 作为 cross-attn 的条件。
- 如果你把它设成 `True`：奇数层会变成纯 self-attn（`encoder_hidden_states=None`），形成 “cross / self / cross / self ...” 的交错模式。

对应源码（逻辑等价）：

```
if idx % 2 == 1 and interleave_self_attention:
  encoder_hidden_states = None
else:
  encoder_hidden_states = vlm_last_embed
```

### 2.7 “suffix token”：state 与 noisy action 如何拼成 DiT 的输入序列

DiT 的 `hidden_states` 不是 VLM token，而是由下面两段拼接出来的“动作序列 token”：

- `state_proj(pad(state))`：把 **(B, 1, 32)** 投到 **(B, 1, 1024)**，作为第一个 token
- `action_in_proj(noisy_actions)`：把 **(B, 50, 32)** 投到 **(B, 50, 1024)**，作为后续 50 个 token

所以 `hidden_states` 的序列长度大致是：

$$
L = 1 + 50 = 51
$$

源码还有一个非常容易忽略的细节：在 `_embed_suffix(...)` 里会执行：

```
state[:, :, [2, 9]] = 0
```

这意味着 state 的某两维在进入 DiT 前被硬置零。
如果你迁移到自家 state layout，一定要注意这行代码，否则你会“以为模型用到了某些维度”，但实际上被置零了。

---

## 3. 代码入口：从脚本到模型的一条最短路径

下面按“怎么跑起来”来走读。

### 3.1 `scripts/run_robochallenge.sh`：最小 launcher

- 脚本：[`scripts/run_robochallenge.sh`](https://raw.githubusercontent.com/Spirit-AI-Team/spirit-v1.5/main/scripts/run_robochallenge.sh)
- 环境变量：
  - 必填：`TASK_NAME`, `ROBOCHALLENGE_JOB_ID`, `USER_TOKEN`, `CKPT_PATH`
  - 可选：`USED_CHUNK_SIZE`（默认 60）

脚本最终执行：
- `python -m robochallenge.run_robochallenge --single_task ... --used_chunk_size ...`

### 3.2 `robochallenge/run_robochallenge.py`：评测循环入口

- 文件：[`robochallenge/run_robochallenge.py`](https://raw.githubusercontent.com/Spirit-AI-Team/spirit-v1.5/main/robochallenge/run_robochallenge.py)

关键点：
- `RoboChallengeExecutor(cfg)`：封装 ckpt 加载 + policy 推理
- `job_loop(..., duration=1/15)`：约 15Hz 的下发频率
- `image_type` 会依据 `TASK_INFO[task][robot_type]` 做选择

#### 3.2.1 频率非常关键：15Hz 的含义

`duration=1/15` 表示它按 15Hz 下发动作（或动作 chunk 的一部分）。这意味着：

- 模型推理 + 后处理 + 网络 I/O 必须在 **~66ms** 内稳定完成，才能不掉帧。
- 因为模型输出是 chunk（多步动作），15Hz 更像“每帧更新一段计划”，再由底层控制器去执行更高频的伺服。
- 这也是很多 VLA 系统追求 action chunking 的工程原因（把“高成本推理”变成“低频更新”）。

### 3.3 `robochallenge/runner/task_info.py`：任务元数据

- 文件：[`robochallenge/runner/task_info.py`](https://raw.githubusercontent.com/Spirit-AI-Team/spirit-v1.5/main/robochallenge/runner/task_info.py)

它把：
- 任务名 → 任务文本（prompt 用）
- 任务名 → robot_type / action_type
- 任务名 → 三路相机 key 映射

### 3.4 `robochallenge/runner/executor.py`：I/O 适配与动作后处理

- 文件：[`robochallenge/runner/executor.py`](https://raw.githubusercontent.com/Spirit-AI-Team/spirit-v1.5/main/robochallenge/runner/executor.py)

你要重点看两段：
- `_prepare_batch(...)`：把 RC 的 observation（图像 bytes + state）整理成 policy 需要的 batch
- `_post_process_action(...)`：把模型 action chunk 转成不同 robot_type 的下发动作格式（UR5/Franka/ARX5/aloha）

#### 3.4.1 `_prepare_batch` 到底在“对齐什么”

这段代码的本质是：把 RoboChallenge 的 observation（每种机器人各不相同）统一投影到 Spirit 期望的 **14 维 state + 3 路图像** 输入格式。

它做的“对齐”主要有三类：

1) **统一 state 的语义布局（强约定）**

`executor.py` 会先创建 `item["observation.state"] = torch.zeros(14)`，再按 robot_type 把字段填进去：

- **ARX5 / Franka（单臂末端控制）**
  - `state[0:3]`：末端位置 xyz
  - `state[3:6]`：末端旋转 rotvec（注意：ARX5 的输入来自 euler，Franka 的输入来自 quat，都会转成 rotvec）
  - `state[6]`：gripper（可选用 stats 做归一化映射到 0.1 尺度）

- **UR5（关节控制）**
  - `state[0:6]`：6 个关节角
  - `state[6]`：gripper（用 `0.1 - ...` 的形式对齐到 0.1 尺度）

- **aloha（双臂）**
  - 左臂：`state[0:3]` xyz，`state[3:6]` rotvec，`state[6]` gripper
  - 右臂：`state[7:10]` xyz，`state[10:13]` rotvec，`state[13]` gripper

并且会保留一份 `observation.state.before_norm`（用于后处理时叠加 delta）。

2) **统一图像的处理方式（固定分辨率 + 缺失相机容错）**

- `_img_byte_to_tensor(...)`：把 byte -> PIL -> RGB -> resize 到 `(320, 240)`，最终变成 `(3, 240, 320)`。
- 若某路相机缺失：只允许 **UR5 的 right_wrist 缺失**（源码里有 assert），会用 `zeros_like(cam_high)` 补一个全 0 图。

3) **统一 batch 的格式**

- `observation.state` 最终是 `(1, 14)`（并放到 device 上）
- `task` / `robot_type` 以 list[str] 的形式传入（后续 prompt 会用到）

如果你要把 Spirit-v1.5 迁移到自家机器人，这一段就是你最需要“复刻/替换”的位置。

参考源码：[`robochallenge/runner/executor.py`](https://raw.githubusercontent.com/Spirit-AI-Team/spirit-v1.5/main/robochallenge/runner/executor.py)

#### 3.4.2 `_post_process_action`：为什么它是“跨机器人可用”的关键

模型输出的是 **delta-like 的 action chunk**（每一步是一条 action 向量）。`_post_process_action(...)` 会把这些 delta 叠加到当前 state 上，变成 RoboChallenge 需要的“绝对控制量”。

它的核心套路是统一的：

- **平移**：`target = delta + current`
- **旋转**：`R_target = R(delta_rotvec) * R(current_rotvec)`（然后按不同机器人输出 euler/quat）
- **夹爪**：根据 robot_type 做方向/尺度变换（并可用 `raw_embodiment_stats` 把 0.1 尺度映射回真实硬件单位）

你可以把各 robot_type 的“输出格式”理解成下面这张表：

- **ARX5**：`[x,y,z] + [euler_xyz] + [gripper]`（旋转：rotvec 复合后转 euler）
- **Franka**：`[x,y,z] + [quat_xyzw] + [gripper]`（旋转：rotvec 复合后转 quat）
- **UR5**：`[joint1..joint6] + [gripper]`（关节：delta + current；夹爪：`0.1 - delta`，无 stats 时会映射到 0~255）
- **aloha**：左臂 `xyz+quat+gripper` + 右臂 `xyz+quat+gripper`（总计 16 维）

这能解释一个重要事实：**Spirit-v1.5 的“跨机器人能力”有一部分来自工程层的 action space 设计与后处理，而不完全是模型本身魔法。**

参考源码：[`robochallenge/runner/executor.py`](https://raw.githubusercontent.com/Spirit-AI-Team/spirit-v1.5/main/robochallenge/runner/executor.py)

---

## 4. RoboChallenge 复现 checklist（工程版）

> 官方仓库 README 也给了简洁版本，这里是“工程可复现”口径的 checklist。

1. Python 版本：3.11+
2. 安装依赖：按仓库 README（uv 或 pip）
3. 准备 checkpoint：确保 `CKPT_PATH/` 下至少存在：
   - `config.json`
   - `model.safetensors`
4. 设置环境变量：
   - `TASK_NAME`：必须在 `TASK_INFO` 中出现
   - `ROBOCHALLENGE_JOB_ID`
   - `USER_TOKEN`
   - `CKPT_PATH`
   - （可选）`USED_CHUNK_SIZE`：默认 60
5. 运行：执行 `scripts/run_robochallenge.sh`

引用：
- 仓库 README：[`Spirit-AI-Team/spirit-v1.5`](https://github.com/Spirit-AI-Team/spirit-v1.5)

### 4.1 常见坑位（建议你真的跑一次就能立刻对上）

- **缺 stats**：Normalize buffer 里 min/max 为 `inf` → assert 失败（见 2.1 节）。
- **CKPT_PATH 不对**：脚本会检查 `model.safetensors` 是否存在。
- **chunk_size 不一致**：脚本默认 60，但 `executor.py` 对某些任务会把 `used_chunk_size` 覆盖为 40（见 `TASKS_USE_LESS_CHUNK_SIZE`）。
- **图像 key 不匹配**：`TASK_INFO` 决定了哪个相机映射到 `cam_high/left_wrist/right_wrist`。
- **UR5 的状态断言**：`_prepare_batch` 中对 UR5 有 `assert state_tmp[6] > 1`（意味着输入的夹爪字段期望是“像 0~255 这种量级”的值），如果你喂的是 0~0.1 会直接炸。
- **夹爪单位容易搞错**：UR5 的 `_post_process_action` 在没有 `raw_embodiment_stats` 时，会把 `0.1` 尺度映射到 0~255；而有 stats 时走另一条 min/max 映射逻辑。
- **图像缺失的容错是有条件的**：源码里只允许 UR5 的 `cam_right_wrist` 缺失，其他情况会 assert（见 3.4.1）。
- **推理 dtype**：`executor.infer` 在 CUDA 上默认启用 `torch.autocast(..., dtype=torch.bfloat16)`；如果你本地环境不支持 bfloat16/driver 不一致，可能会遇到性能或兼容问题（尤其是 Windows 环境）。

### 4.2 “迁移到自家机器人”最小改造清单（非常实用）

如果你不是跑 RoboChallenge，而是要把 Spirit-v1.5 接到自己的机器人上，通常只需要改 3 处（改太多反而容易引入 train-deploy mismatch）：

1. **观测对齐（你自己的 `_prepare_batch`）**
   - 保证输入字典里有：三路图像 + state + task/robot_type（或你定义的等价条件）
   - 把你机器人的状态向量拼成模型期望的 state layout（位置/旋转/夹爪等）

2. **动作对齐（你自己的 `_post_process_action`）**
   - 把模型输出的 delta-like action（position/rotvec/gripper）映射到你的控制接口
   - 注意旋转组合顺序（左乘/右乘）和单位（rad / meter）

3. **归一化统计量（stats）**
   - 最稳的做法：沿用 checkpoint 里随模型发布的 stats
   - 如果你必须换 action/state 定义：需要重新生成匹配维度的 min/max，否则会出现 2.1 节的 assert 问题

---

## 5. 训练范式深度：多样性数据采集（Diverse Collection）

根据 Spirit AI 官方 Blog [Spirit-v1.5: Clean Data Is the Enemy](https://www.spirit-ai.com/en/blog/spirit-v1-5)，该模型的核心竞争力来自对数据质量定义的重构。

### 5.1 为什么要用“脏数据”？
- **传统方式（Clean Data）**：任务脚本化、物体摆放预测、路径线性化。结果是机器人只学会了“实验室内的完美路径”，一旦遇到部分遮挡或微小偏差就会失败。
- **Spirit 方式（Diverse Data）**：
    - **随机性**：操作员即兴发挥，不设固定脚本（如“今天我用机器人调一杯鸡尾酒”，具体步骤自定）。
    - **连续性**：在一次会话中覆盖多种原子技能（抓取、扭转、插入、双臂协作）。
    - **容错性**：数据中包含大量“失败-重试”循环，使模型学会了如何从错误中恢复。

### 5.2 消融实验发现：多样性溢价（Diversity Premium）
Spirit AI 进行了 Group A（精心编排数据）与 Group B（多样化数据）的对比：
- **迁移效率**：在相同数据量下，多样化模型（Group B）在微调阶段达到相同性能所需的迭代次数 **减少了 40%**。
- **可扩展性**：由于不依赖精细的任务设计，数据采集效率提升了 **200%**，研究人员的关注度需求降低了 **60%**。

---

## 6. RoboChallenge Table30 评测深度

Spirit-v1.5 登顶的 Table30 榜单是具身智能领域的高难度实机测评：
- **30 个真实任务**：包括插花（arrange_flowers）、做素食三明治（make_vegetarian_sandwich）、插网线（plug_in_network_cable）等。
- **多维度挑战**：精确 3D 定位、遮挡处理、长程时序依赖。
- **跨平台一致性**：要求模型在 Franka、Arx5、UR5 和 ALOHA 等异构平台上表现稳定。

---

## 7. 与 π0 / π0.5 的“写作逻辑一致”的对比（只写可验证点）

这里刻意遵循 `pi0_flow_matching.md` / `pi0_5_dissection.md` 的对比方式：只写“代码/公开材料能确认”的点。

### 7.1 Backbone
- Spirit-v1.5：Qwen3-VL（仓库 README 与 `SpiritVLAConfig.backbone`）
  - 参考：[`Spirit-AI-Team/spirit-v1.5`](https://github.com/Spirit-AI-Team/spirit-v1.5)

### 7.2 Action 生成范式
- Spirit-v1.5：DiT head + Euler 迭代（ODE 风格）生成 action chunk
  - 参考：[`model/modeling_spirit_vla.py`](https://raw.githubusercontent.com/Spirit-AI-Team/spirit-v1.5/main/model/modeling_spirit_vla.py)

### 7.3 Benchmark 集成方式
- Spirit-v1.5：把评测入口明确固化成 `scripts/` + `robochallenge/` 模块（可复现工程结构很清晰）
  - 参考：[`scripts/run_robochallenge.sh`](https://raw.githubusercontent.com/Spirit-AI-Team/spirit-v1.5/main/scripts/run_robochallenge.sh)
  - 参考：[`robochallenge/run_robochallenge.py`](https://raw.githubusercontent.com/Spirit-AI-Team/spirit-v1.5/main/robochallenge/run_robochallenge.py)

### 7.4 “今天打败 π0.5 登顶”如何表述更严谨
- **性能数据**：截至 2026-01-11，Spirit-v1.5 在 Table30 总分 66.09，成功率 50.33%，具有统计学显著的领先优势。
- **最强引用**：仓库 README 的 Table30 #1 声明（截至 2026-01-11）
  - 参考：[`Spirit-AI-Team/spirit-v1.5`](https://github.com/Spirit-AI-Team/spirit-v1.5)
- **官方 Blog 深度解析**：
  - 参考：[Spirit-v1.5: Clean Data Is the Enemy of Great Robot Foundation Models](https://www.spirit-ai.com/en/blog/spirit-v1-5)
- **交叉引用**：中文媒体 2026-01-12 的榜单叙述（用于“今天打败 π0.5”这一口径）
  - 参考：[`m.sohu.com` 报道](https://m.sohu.com/a/975015519_610300)
  - 参考：[`stcn.com` 报道](https://www.stcn.com/article/detail/3586134.html)

---

## 参考与延伸阅读

- Spirit v1.5 官方仓库：[`Spirit-AI-Team/spirit-v1.5`](https://github.com/Spirit-AI-Team/spirit-v1.5)
- Spirit v1.5 官方 Blog：[Spirit-v1.5: Clean Data Is the Enemy](https://www.spirit-ai.com/en/blog/spirit-v1-5)
- Spirit v1.5 HuggingFace（仓库 README 提供）：[`Spirit-AI-robotics/Spirit-v1.5`](https://huggingface.co/Spirit-AI-robotics/Spirit-v1.5)
- Spirit v1.5 官方演示视频（见 Blog）：涵盖了调酒、素描、积木堆叠等多样化技能演示。
