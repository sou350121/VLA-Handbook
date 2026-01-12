# Spirit-v1.5 深度拆解（Spirit AI）

> 本文基于 Spirit AI 官方开源仓库进行**代码级**阅读与整理，重点覆盖：模型结构、推理数据流、RoboChallenge 运行入口、关键文件导航与可复现 checklist。

- 官方仓库：[`Spirit-AI-Team/spirit-v1.5`](https://github.com/Spirit-AI-Team/spirit-v1.5)

---

## 1. 一句话结论：为什么它值得你马上看

Spirit-v1.5 是 Spirit AI 开源的 VLA（Vision-Language-Action）机器人基础模型实现。其仓库 README 明确声明：截至 2026-01-11，**Spirit-v1.5 在 RoboChallenge Table30 榜单位列第一**（并提供了推理代码与 checkpoint 下载方式）
（引用见仓库 README：[`Spirit-AI-Team/spirit-v1.5`](https://github.com/Spirit-AI-Team/spirit-v1.5)）。

同时，多家中文媒体在 2026-01-12 报道了“Spirit v1.5 超越 Pi0.5 登顶 Table30”的消息（用于交叉验证“今天打败 π0.5”这一叙述）：
- [`m.sohu.com` 报道](https://m.sohu.com/a/975015519_610300)
- [`stcn.com` 报道](https://www.stcn.com/article/detail/3586134.html)

---

## 2. 模型架构概览（从代码视角）

仓库 README 对模型结构的描述非常直接：
- `model/modeling_spirit_vla.py`：主模型架构（**Qwen3-VL backbone + DiT head + policy API**）
（见仓库目录结构说明：[`Spirit-AI-Team/spirit-v1.5`](https://github.com/Spirit-AI-Team/spirit-v1.5)）。

### 2.1 数据流（推理时）

```mermaid
flowchart TD
  obs[Obs_Images+State+TaskText] --> preprocess[preprocess_rb_batch]
  preprocess --> qwen[Qwen3VLForConditionalGeneration]
  qwen --> vlmHidden[HiddenStates_LastK]
  obs --> statePad[pad_vector(State->max_state_dim)]
  statePad --> stateProj[state_proj]
  noise[Gaussian_Noise_ActionChunk] --> actionIn[action_in_proj]
  vlmHidden --> proj[proj_vlm_output]
  stateProj --> ditIn[Concat(StateEmb+ActionEmb)]
  actionIn --> ditIn
  proj --> dit[BaseDiT_CrossAttn]
  dit --> vPred[action_out_proj]
  vPred --> ode[ODE_Euler_Integration]
  ode --> actionChunk[ActionChunk_TxD]
  actionChunk --> unnorm[Unnormalize]
  unnorm --> post[Executor_PostProcess_To_RobotAction]
```

---

## 3. 关键代码入口逐文件拆解

下面按仓库的“最短运行路径”来拆：`scripts/run_robochallenge.sh` → `robochallenge/run_robochallenge.py` → `executor.py` → `modeling_spirit_vla.py`。

### 3.1 `scripts/run_robochallenge.sh`：最小启动脚本

- 入口脚本：[`scripts/run_robochallenge.sh`](https://raw.githubusercontent.com/Spirit-AI-Team/spirit-v1.5/main/scripts/run_robochallenge.sh)
- 关键点：
  - 通过 `PYTHONPATH` 把 repo root 加入路径
  - 要求环境变量：`TASK_NAME`, `ROBOCHALLENGE_JOB_ID`, `USER_TOKEN`, `CKPT_PATH`，以及可选 `USED_CHUNK_SIZE`（默认 60）
  - 最终执行：`python -m robochallenge.run_robochallenge ...`

### 3.2 `robochallenge/run_robochallenge.py`：RoboChallenge 进程入口

- 文件：[`robochallenge/run_robochallenge.py`](https://raw.githubusercontent.com/Spirit-AI-Team/spirit-v1.5/main/robochallenge/run_robochallenge.py)
- 主要逻辑：
  - 解析 CLI 参数：`--single_task --robochallenge_job_id --ckpt_path --user_token --used_chunk_size`
  - 构造 `RoboChallengeExecutor(cfg)`
  - 创建 `InterfaceClient(cfg.user_token)` 并进入 `job_loop(...)`
  - 控制循环频率：`duration = 1 / 15`（即 15Hz 推理/下发节奏）

### 3.3 `robochallenge/runner/task_info.py`：任务与机器人映射表

- 文件：[`robochallenge/runner/task_info.py`](https://raw.githubusercontent.com/Spirit-AI-Team/spirit-v1.5/main/robochallenge/runner/task_info.py)
- 你需要注意的点：
  - `TASK_INFO[task_name]` 给出：`task`（自然语言任务）、`robot_type`（ARX5/Franka/UR5/aloha）、`action_type`（leftpos/leftjoint/pos）、以及三路相机 key 的映射。
  - `TASKS_USE_LESS_CHUNK_SIZE = ["move_objects_into_box"]`：某些任务默认用更小的 chunk（在 executor 里会覆盖为 40）。

### 3.4 `robochallenge/runner/executor.py`：I/O 适配与动作后处理

- 文件：[`robochallenge/runner/executor.py`](https://raw.githubusercontent.com/Spirit-AI-Team/spirit-v1.5/main/robochallenge/runner/executor.py)

核心链路：
- `_prepare_batch(...)`：把 RC 的图像 bytes + state → `observation.images.*` + `observation.state` + `task` + `robot_type`
- `policy.select_action(item)`：推理出 action chunk（B×T×D）
- `_post_process_action(...)`：按 robot_type 把 action chunk 变成 RC 需要的动作格式（UR5/Franka/ARX5/aloha 的差异很大）

### 3.5 `model/modeling_spirit_vla.py`：SpiritVLAPolicy 的推理核心

- 文件：[`model/modeling_spirit_vla.py`](https://raw.githubusercontent.com/Spirit-AI-Team/spirit-v1.5/main/model/modeling_spirit_vla.py)

关键点（只讲“能复现/能读懂”的）：

- **Backbone**：`Qwen3VLForConditionalGeneration.from_pretrained(backbone)`
- **视觉输入**：三路图像（high/left_wrist/right_wrist），通过 Qwen3-VL 的 image processor + vision tower 获取 embeddings
- **prompt 组织**：把 `robot_type` 与 `task` 以 chat_template 形式组织进输入，使 VLM hidden states 吸收任务语义
- **DiT head**：`BaseDiT` 作为动作生成器，cross-attend 到 VLM hidden states
- **动作生成**：ODE/Euler 迭代形式的连续去噪/生成（`x_t += dt * v_t`），输出 action chunk
- **归一化**：STATE/ACTION 默认走 MIN_MAX；注意 stats/buffer 需要随 checkpoint 加载，否则会在 Normalize 中报错（这一点对复现很关键）

相关工具函数在：[`model/utils.py`](https://raw.githubusercontent.com/Spirit-AI-Team/spirit-v1.5/main/model/utils.py)

---

## 4. RoboChallenge 复现 checklist（按官方仓库最小路径）

### 4.1 安装
官方 README 要求：Python 3.11+，并推荐 `uv` 或 `pip`。

- 参考：[`Spirit-AI-Team/spirit-v1.5`](https://github.com/Spirit-AI-Team/spirit-v1.5)

### 4.2 下载 checkpoint
- Base：[`Spirit-AI-robotics/Spirit-v1.5`](https://huggingface.co/Spirit-AI-robotics/Spirit-v1.5)
- 详细 checkpoint 表见仓库 README：[`Spirit-AI-Team/spirit-v1.5`](https://github.com/Spirit-AI-Team/spirit-v1.5)

### 4.3 运行
按官方脚本：[`scripts/run_robochallenge.sh`](https://raw.githubusercontent.com/Spirit-AI-Team/spirit-v1.5/main/scripts/run_robochallenge.sh)

你需要准备：
- `TASK_NAME`：必须出现在 `TASK_INFO` 里（见：[`task_info.py`](https://raw.githubusercontent.com/Spirit-AI-Team/spirit-v1.5/main/robochallenge/runner/task_info.py)）
- `ROBOCHALLENGE_JOB_ID`
- `USER_TOKEN`
- `CKPT_PATH`（要求目录下存在 `model.safetensors` 与 `config.json`）
- `USED_CHUNK_SIZE`（默认 60；某些任务在 executor 内会改为 40）

---

## 5. 与 π0.5 的“可验证”对比（不做无依据推断）

这里只写**代码层面可验证**的差异点（而不是凭感觉解释为什么得分更高）：

- **Backbone 选择**：Spirit-v1.5 明确是 **Qwen3-VL** 体系（见仓库 README 与 `SpiritVLAConfig.backbone` 默认值：[`Spirit-AI-Team/spirit-v1.5`](https://github.com/Spirit-AI-Team/spirit-v1.5)）。
- **动作生成范式**：它用 DiT + ODE/Euler 迭代生成 action chunk（见：[`model/modeling_spirit_vla.py`](https://raw.githubusercontent.com/Spirit-AI-Team/spirit-v1.5/main/model/modeling_spirit_vla.py)）。
- **评测集成方式**：它把 RoboChallenge 的 “job polling / executor / post-process” 做成独立模块（见：[`run_robochallenge.py`](https://raw.githubusercontent.com/Spirit-AI-Team/spirit-v1.5/main/robochallenge/run_robochallenge.py)，[`executor.py`](https://raw.githubusercontent.com/Spirit-AI-Team/spirit-v1.5/main/robochallenge/runner/executor.py)）。

至于“今天打败 π0.5 登顶 Table30”，这里引用公开叙述，但不做无依据归因：
- [`m.sohu.com` 报道](https://m.sohu.com/a/975015519_610300)
- 仓库自身的 Table30 #1 声明：[`Spirit-AI-Team/spirit-v1.5`](https://github.com/Spirit-AI-Team/spirit-v1.5)

---

## 6. 你可以怎么用它（两条路线）

- **路线 A：复现 RoboChallenge**
  - 直接按 `scripts/run_robochallenge.sh` 跑
  - 先从单一任务验证 pipeline（例如 `move_objects_into_box`）

- **路线 B：迁移到自家机器人/数据格式**
  - 让你的观测对齐到 `observation.images.*` 与 `observation.state`
  - 复用 `SpiritVLAPolicy.select_action(...)` 产出 action chunk
  - 自己实现一个 `_post_process_action` 适配你的下发协议

---

## 参考与延伸阅读

- Spirit v1.5 官方仓库：[`Spirit-AI-Team/spirit-v1.5`](https://github.com/Spirit-AI-Team/spirit-v1.5)
- Spirit v1.5 官方 blog：[`spirit-ai.com/en/blog/spirit-v1-5`](https://www.spirit-ai.com/en/blog/spirit-v1-5)
- HuggingFace checkpoints：[`Spirit-AI-robotics/Spirit-v1.5`](https://huggingface.co/Spirit-AI-robotics/Spirit-v1.5)
