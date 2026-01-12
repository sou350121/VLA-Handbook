# Spirit-v1.5 模型解剖（Dissecting Spirit-v1.5）

> **写作风格说明**：本文刻意参考 `pi0_flow_matching.md` / `pi0_5_dissection.md` 的叙事逻辑：
> **Main Mathematical Idea → 架构信息流（ASCII）→ 数学/推理细节 → 代码入口走读 → 复现 checklist → 与 π0/π0.5 对比**。

- 官方仓库：[`Spirit-AI-Team/spirit-v1.5`](https://github.com/Spirit-AI-Team/spirit-v1.5)

---

## 0. 主要数学思想 (Main Mathematical Idea)

> **第一性原理**：把“动作生成”写成 **条件 ODE 的轨迹积分**。

Spirit-v1.5 的 action head 不是一步回归，也不是离散 token 采样；它更像 π0 的 Flow Matching：

- 让网络预测一个“速度/更新方向” $v_t$（代码中命名为 `v_t`）
- 从噪声动作块 $x_t$ 出发
- 用 Euler 积分把 $x_t$ 沿着 $v_t$ 逐步推回到“干净动作块”

在代码里，这个过程长得非常像：

$$
\frac{dx}{dt} = v_\theta(x,t,\mathrm{cond})
$$

以及离散化的 Euler 更新：

$$
 x_{t+\Delta t} = x_t + \Delta t \cdot v_\theta(x_t,t,\mathrm{cond})
$$

其中条件 $\mathrm{cond}$ 来自 VLM（Qwen3-VL）的 hidden states。

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

---

## 2. 数学/推理细节：它到底“走了几步”，每一步做什么

下面完全按 `model/modeling_spirit_vla.py` 的实现口径解释。

### 2.1 状态与动作的归一化：MIN_MAX → [-1, 1]

在 `SpiritVLAPolicy.select_action(...)` 内，先执行：
- `Normalize` inputs（state）
- `Normalize` targets（action space 的定义）

这一步依赖 MIN_MAX 统计量（min/max）。代码里 `Normalize(..., stats=None)`，但 forward 里会 assert min/max 不是 `inf`，因此这些 buffer 必须来自 checkpoint 的 `state_dict`。

- 对应实现：[`model/utils.py`](https://raw.githubusercontent.com/Spirit-AI-Team/spirit-v1.5/main/model/utils.py)

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

---

## 5. 与 π0 / π0.5 的“写作逻辑一致”的对比（只写可验证点）

这里刻意遵循 `pi0_flow_matching.md` / `pi0_5_dissection.md` 的对比方式：只写“代码/公开材料能确认”的点。

### 5.1 Backbone
- Spirit-v1.5：Qwen3-VL（仓库 README 与 `SpiritVLAConfig.backbone`）
  - 参考：[`Spirit-AI-Team/spirit-v1.5`](https://github.com/Spirit-AI-Team/spirit-v1.5)

### 5.2 Action 生成范式
- Spirit-v1.5：DiT head + Euler 迭代（ODE 风格）生成 action chunk
  - 参考：[`model/modeling_spirit_vla.py`](https://raw.githubusercontent.com/Spirit-AI-Team/spirit-v1.5/main/model/modeling_spirit_vla.py)

### 5.3 Benchmark 集成方式
- Spirit-v1.5：把评测入口明确固化成 `scripts/` + `robochallenge/` 模块（可复现工程结构很清晰）
  - 参考：[`scripts/run_robochallenge.sh`](https://raw.githubusercontent.com/Spirit-AI-Team/spirit-v1.5/main/scripts/run_robochallenge.sh)
  - 参考：[`robochallenge/run_robochallenge.py`](https://raw.githubusercontent.com/Spirit-AI-Team/spirit-v1.5/main/robochallenge/run_robochallenge.py)

### 5.4 “今天打败 π0.5 登顶”如何表述更严谨
- **最强引用**：仓库 README 的 Table30 #1 声明（截至 2026-01-11）
  - 参考：[`Spirit-AI-Team/spirit-v1.5`](https://github.com/Spirit-AI-Team/spirit-v1.5)
- **交叉引用**：中文媒体 2026-01-12 的榜单叙述（用于“今天打败 π0.5”这一口径）
  - 参考：[`m.sohu.com` 报道](https://m.sohu.com/a/975015519_610300)
  - 参考：[`stcn.com` 报道](https://www.stcn.com/article/detail/3586134.html)

---

## 参考与延伸阅读

- Spirit v1.5 官方仓库：[`Spirit-AI-Team/spirit-v1.5`](https://github.com/Spirit-AI-Team/spirit-v1.5)
- Spirit v1.5 HuggingFace（仓库 README 提供）：[`Spirit-AI-robotics/Spirit-v1.5`](https://huggingface.co/Spirit-AI-robotics/Spirit-v1.5)
- Spirit v1.5 官方 blog（仓库 README 的 BibTeX 指向）：[`spirit-ai.com/en/blog/spirit-v1-5`](https://www.spirit-ai.com/en/blog/spirit-v1-5)
