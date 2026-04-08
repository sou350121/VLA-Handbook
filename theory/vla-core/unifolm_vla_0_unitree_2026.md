# UnifoLM-VLA-0：Unitree 的 VLA 训练与部署开源实现 (UnifoLM-VLA-0: Unitree’s VLA Open-Source Stack)

> **发布时间**：2026-01-29（开源代码/权重发布日）  
> **模型名**：UnifoLM-VLA-0（含 `UnifoLM-VLM-Base` / `UnifoLM-VLA-Base` / `UnifoLM-VLA-LIBERO` checkpoint）  
> **核心定位**：以 **Qwen2.5-VL-7B** 为 VLM 主干，把 VLM 从“理解/对话”继续训练成“动作条件编码器”，再接一个 **Flow-Matching + DiT** 动作头，做 **action chunking** 的长时域控制；配套给出 **LeRobot→HDF5→RLDS** 数据管线与 **server-side 推理** 的真机部署模板。

**一手来源**（本文只写可核验内容）：  
- 项目主页：`https://unigen-x.github.io/unifolm-vla.github.io/`  
- GitHub：`https://github.com/unitreerobotics/unifolm-vla`  
- 模型/数据集集合页（指针）：`https://huggingface.co/unitreerobotics/models`、`https://huggingface.co/unitreerobotics/datasets`

---

## 0. 结论先行（工程视角）

- **这套开源最值得学的不是“又一个 SOTA 数字”**，而是它把 VLA 的关键工程闭环补齐了：  
  - **训练**：accelerate + deepspeed（zero2），统一 RLDS 读数，多数据集 mixture；  
  - **动作建模**：不是“直接回归动作向量”，也不是逐步去噪式 diffusion policy，而是 **flow matching（速度场）+ DiT（cross-attn 到 VLM hidden state）**；  
  - **部署**：明确的 **server / client** 接口（FastAPI `/act`）+ 归一化/反归一化（`dataset_statistics.json`）。
- **你可以把它理解成**：OpenVLA-style 的“VLM 条件动作模型” + Isaac GR00T/CogACT 系的“生成式动作头”（但这里明确落成了可运行代码与数据管线）。

---

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 模块 | 在代码里叫什么 | 输入 → 输出 | 关键实现证据 | 你需要盯的工程变量 |
|---|---|---|---|---|
| **VLM 主干** | `model/modules/vlm/QWen2_5.py` | 图像(可多帧/多相机) + 指令 → VLM 最后层 hidden states | `Qwen2_5_VLForConditionalGeneration` + `flash_attention_2` + bfloat16 | 上下文长度、相机数量、窗口大小 `window_size`、是否冻结 VLM |
| **动作头（生成式）** | `FlowmatchingActionHead`（`DiT_ActionHeader.py`） | 条件 `vl_embs`(+ proprio) → 预测动作序列 | DiT + cross-attn（`encoder_hidden_states=vl_embs`） + flow matching loss | `NUM_ACTIONS_CHUNK`、`num_inference_timesteps`、`repeated_diffusion_steps` |
| **动作分块 (chunking)** | `NUM_ACTIONS_CHUNK`（`constants.py`）+ `future_action_window_size`（`datasets.py`） | 预测长度为 \(H\) 的动作块 | G1 默认 \(H=25\)；LIBERO \(H=8\) | chunk 太长→误差积累；太短→规划不足 |
| **数据管线** | LeRobot→HDF5→RLDS(TFDS) | 离线示范 → RLDS episodes | `prepare_data/convert_lerobot_to_hdf5.py` + `prepare_data/hdf5_to_rlds/.../rlds_dataset.py` | 相机 key 对齐、动作/状态维度对齐、归一化统计 |
| **真机推理** | FastAPI server | `observations[]` → `action` | `deployment/model_server/run_real_eval_server.py` | 时延/带宽、归一化一致性、payload schema 稳定性 |

### 1.2 关键机制 (Key Mechanism)

**机制 A：把 VLM 当成“条件编码器”，不做 LM loss**  
训练脚本 `train_unifolm_vla.py` 的 forward 里只用：
- `qwenvl_outputs.hidden_states[-1]` 作为条件 `vl_embs`
- 动作损失来自 action head（`action_loss`），没有语言生成监督

**机制 B：动作头是 Flow Matching（速度场）+ DiT（跨注意力到 VLM）**  
`FlowmatchingActionHead.forward()` 的核心是：
- 采样噪声 `noise`，用时间 \(t\) 混合得到 `noisy_trajectory`
- 目标是速度 `velocity = actions - noise`
- 模型预测 `pred_velocity`，用 MSE 拟合速度场（这是“flow matching”的常见实现形态）

**机制 C：proprio 不是可选装饰，而是动作分布对齐的一部分**  
server 侧显式对 `state` 做 normalize，并和图像一起喂入模型；训练侧也可通过 `--trainer.use_proprio True` 打开。

### 1.3 信息流/架构图 (Flow / Diagram)

```text
        (RLDS batch)                              (Model)
  images(window, multi-cam)  ───────┐
  instruction(text)          ───────┼──> Qwen2.5-VL (flash-attn, bf16) ──> last_hidden [B,L,H]
  proprio(state)             ───────┘                                  \
                                                                      cross-attn
                                                                         \
                                                                  FlowMatchingActionHead (DiT)
                                                                  - sample t, noise
                                                                  - predict velocity field
                                                                  -> actions_chunk [B,H,action_dim]
```

---

## 2. 数学核心：Flow Matching 动作头 + 动作分块如何工作 (Math Core)

### 2.1 目标：预测一段动作块而不是下一步

设当前时刻（或窗口末端）条件特征为 \(c\)（由 VLM hidden state + proprio 编码得到），动作块为：

\[
\mathbf{a}_{1:H} \in \mathbb{R}^{H \times D}
\]

其中 \(H=\texttt{NUM\_ACTIONS\_CHUNK}\)，\(D=\texttt{ACTION\_DIM}\)。

在代码中（`constants.py`）：
- LIBERO：\(H=8, D=7\)
- Unitree G1（EE 6D 路线常用）：\(H=25, D=23\)（`G1_EE_6D_CONSTANTS`）

### 2.2 Flow Matching 的损失：拟合速度场

实现对应 `DiT_ActionHeader.py`：

- 采样噪声 \(\epsilon \sim \mathcal{N}(0, I)\)
- 采样时间 \(t \in (0,1)\)（代码里来自 Beta 分布并离散到 `num_timestep_buckets`）
- 构造噪声轨迹：

\[
\tilde{\mathbf{a}} = (1-t)\epsilon + t\mathbf{a}
\]

- 目标速度：

\[
\mathbf{v} = \mathbf{a} - \epsilon
\]

- 模型预测速度 \(\hat{\mathbf{v}}_\theta(\tilde{\mathbf{a}}, c, t)\)，优化：

\[
\mathcal{L} = \mathbb{E}\left[\lVert \hat{\mathbf{v}}_\theta - \mathbf{v} \rVert_2^2 \right]
\]

### 2.3 推理：Euler 积分得到动作（离散步数由 `num_inference_timesteps` 控制）

`predict_action()` 用 \(N\) 步欧拉积分：

\[
\mathbf{a}^{(k+1)} = \mathbf{a}^{(k)} + \Delta t \cdot \hat{\mathbf{v}}_\theta(\mathbf{a}^{(k)}, c, t_k),\quad \Delta t=\frac{1}{N}
\]

工程含义：\(N\) 越大推理越慢但可能更稳；\(N\) 太小会更“贪快”，在接触丰富任务上更容易抖。

---

## 3. 带数字走一遍：一个最小动作 chunk 例子 (Worked Example)

以 **G1_EE_6D** 为例（按仓库常量）：  
- 动作维度 \(D=23\)：左右臂各 \(3\)（xyz）+\(6\)（rotation 6D）共 \(18\) 维，外加 **左右夹爪 2 维 + 腰部 3 维**（合计 \(5\) 维）  
- chunk 长度 \(H=25\)

你可以把模型输出理解为：

```text
action_chunk = [
  a_t, a_{t+1}, ..., a_{t+24}
]
每个 a 是 23 维连续向量（训练时按统计量归一化到 [-1,1] 或 q01-q99 区间）
```

数据侧（`datasets.py`）用：
- `traj_transform_kwargs.future_action_window_size = NUM_ACTIONS_CHUNK - 1`

工程解读：RLDS 管线在 `chunk_act_obs` 里把轨迹切片为「观测窗 + 未来动作窗」；在本仓库默认配置中，未来动作窗大小通过 `future_action_window_size` 与 `NUM_ACTIONS_CHUNK` 绑定，用于 action chunking。  
建议你在本地训练前先打印一次 batch 的 `action` shape，确认“实际训练喂给 action head 的动作序列长度”与当前平台常量一致（避免 silent mismatch）。

---

## 4. 工程视角：快慢路径 / 训练-推理折中 (Engineering View)

### 4.1 训练入口与关键超参

训练脚本（`scripts/run_scripts/run_unifolm_vla_train.sh`）关键字段：
- `base_vlm`：初始化的 VLM 权重（例如 `UnifoLM-VLM-Base` 本地路径）
- `data_mix`：数据混合名（例如 `Unitree_all_task` 或 `g1_stack_block`）
- `window_size`：输入帧窗口（1 或 2；LIBERO eval 脚本用过 `window_size=2`）
- `--trainer.use_wrist_image True`：配置层面宣称支持腕相机（但见本文“验收备注”，当前 `RLDSBatchTransform` 默认只取 `image_primary`）
- `--trainer.use_proprio True`：显式打开 proprio
- `--trainer.max_train_steps 150000`，`lr=4e-5`

### 4.2 混合数据集不是“拼起来就行”：要靠 mixture 权重把任务分布拉平

`mixtures.py` 里给了 `Unitree_all_task` 的权重（并非平均），这决定了训练过程中各任务被采样的频率，是多任务稳定性的第一旋钮。

### 4.3 归一化/反归一化：真机稳定性的“隐藏地基”

训练会保存 `dataset_statistics.json`（`train_unifolm_vla.py` 调 `save_dataset_statistics(...)`）。  
推理 server 侧用它做：
- `normalize_proprio(...)`
- `unnormalize_action(...)`

如果你迁移到新机器人/新动作定义，**最容易错的不是模型，而是统计量与维度对不上**。

### 4.4 迁移必看：平台常量/维度/归一化对齐表

这张表用来回答一个工程上最致命的问题：**你现在训练/推理使用的 action/proprio 维度与归一化口径到底是哪一套？**

| 平台常量 | 触发方式（代码） | `NUM_ACTIONS_CHUNK` | `ACTION_DIM` | `PROPRIO_DIM` | 归一化 | 典型 `unnorm_key`（示例） |
|---|---|---:|---:|---:|---|---|
| `LIBERO_CONSTANTS` | `constants.py`：命令行参数含 `libero` | 8 | 7 | 8 | `BOUNDS_Q99` | `libero_spatial_no_noops` 等 |
| `G1_EE_6D_CONSTANTS` | `constants.py`：命令行参数含 `ee_6d`；否则默认也会走到该分支 | 25 | 23 | 23 | `BOUNDS_Q99` | `g1_stack_block` 等 |
| `G1_CONSTANTS`（joint） | `constants.py`：命令行参数含 `joint` | 25 | 16 | 16 | `BOUNDS` |（取决于你自己的 joint 数据集 key） |

两条硬规则：
- **`unnorm_key` 必须与训练时保存的 `dataset_statistics.json` 里 key 完全一致**（server 侧会用它做 normalize/unnormalize）。  
- **动作维度的“名义定义”必须与标准化 transform 输出对齐**：例如 Unitree G1 的 EE6D 路线在标准化后会把 `observation.state` / `action` 映射到 `*_6d`（总维度 23）。

---

## 5. 数据与评测 (Data & Eval)

### 5.1 数据集与格式：LeRobot → HDF5 → RLDS

仓库给出两步转换：

- **LeRobot → HDF5**：`prepare_data/convert_lerobot_to_hdf5.py`  
  - 读 `LeRobotDataset`，写每个 episode 一个 `episode_{i}.hdf5`  
  - 组织了 `qpos` / `ee_qpos` / `action` / `ee_action` 与多路相机图像

- **HDF5 → RLDS(TFDS)**：`prepare_data/hdf5_to_rlds/rlds_dataset/rlds_dataset.py`  
  - TFDS builder 里目前 hardcode 了 glob：`glob.glob(\"/path/to/save/the/converted/data/directory/*.hdf5\")`（你需要改为自己的路径）
  - 定义了 observation 的 image/state/ee_state 形状，并提供 RPY→6D 的批处理转换

### 5.2 仿真：LIBERO

eval 脚本：`scripts/eval_scripts/run_eval_libero.sh`  
核心要改的字段：
- `your_ckpt`（`pytorch_model.pt`）
- `vlm_pretrained_path`
- `task_suite_name`（`libero_spatial`/`goal`/`object`/`long` 等）
- `unnorm_key`
- `window_size`

项目主页给出的 LIBERO 汇总（Average=98.7）可作为对标读法，但你在复现时要确认：**任务套件、no-noops 版本、窗口大小是否一致**（否则数值不可直接对比）。

### 5.3 真机：12 类任务单策略（声明级）

项目主页宣称：在 Unitree G1 上，用单一策略 checkpoint 覆盖 12 类复杂操作，并展示抗干扰。这里属于演示与系统声明，本文不扩写未开源的实验细节。

---

## 6. 能力与失败模式 (Capabilities & Failure Modes)

### 6.1 你可以期待的能力

- **长时域动作生成**：chunking + 生成式动作头天然偏长时域
- **多相机融合（推理侧明确）**：server 侧示例把 `full_image + wrist_*` 以 multi-image prompt 喂给 VLM；训练侧是否同样启用多相机，需以本地 batch transform 实际拼接为准
- **多任务统一**：通过 mixture 权重与统一 action head 训练实现

### 6.2 你应该预期的失败模式（按代码结构推导）

| 失败模式 | 典型表现 | 最小诊断 | 常见修复 |
|---|---|---|---|
| **动作/状态维度不匹配** | server 报 shape error 或动作发散 | 检查 `constants.py` 平台选择与 dataset feature shape | 明确选择 `G1_EE_6D`/`G1` 常量；重建统计量 |
| **统计量不一致** | 真机动作幅度明显不对（过大/过小） | 对比训练保存的 `dataset_statistics.json` 与部署用的 key | 统一 `unnorm_key`；重新生成 stats |
| **多任务负迁移** | 某任务学会了另一个变差 | 看 `mixtures.py` 的权重与采样占比 | 调权重；拆 task family；分 head/adapter |
| **推理时延过大** | 控制频率不够导致抖动 | 打印 server inference time | 减少 `num_inference_timesteps`；减窗口/相机；量化/更强 GPU |

---

## 7. 与相关工作对比 (Comparison)

| 维度 | UnifoLM-VLA-0（本仓库） | OpenVLA/OFT 类 | 传统 BC 回归 |
|---|---|---|---|
| VLM 角色 | 条件编码器（取 hidden states） | 多为 VLM + action head | 常见为纯视觉/状态编码 |
| 动作头 | Flow matching + DiT cross-attn | 常见为回归或 diffusion/transformer policy | MSE/GMM |
| 长时域 | chunk（平台相关：G1=25） | 依实现而定 | 往往弱 |
| 部署 | server/client + unnorm stats | 不一定提供端到端模板 | 工程化成本高 |

**面试 Tip（1 句话）**：被问“它和 Diffusion Policy 有什么本质区别？”——答：它用 **flow matching 直接学速度场**，推理靠 **少步 Euler 积分**，本质上仍是“生成式动作模型”，但训练目标与采样过程更像“连续时间 ODE”视角而不是逐步去噪。

---

## 8. 验收备注：哪些断言已被代码证明，哪些需要你本地再确认

下面是我对照仓库代码做的验收清单（保证本文“强断言”可追溯）：

- **VLM 主干与精度/注意力实现（已核验）**：`QWen2_5.py` 明确使用 `Qwen2_5_VLForConditionalGeneration` + `flash_attention_2` + bfloat16 + `device_map="cuda"`。  
- **训练确实不做 LM 生成监督（已核验）**：`train_unifolm_vla.py` 的 `Unifolm_VLA.forward()` 只返回 `action_loss`，没有 `labels` 的 LM loss。  
- **动作头确实是 flow matching + DiT cross-attn（已核验）**：`DiT_ActionHeader.py` 明确构造 `velocity = actions - noise` 并最小化 MSE；`DiT(..., encoder_hidden_states=vl_embs)`。  
- **RLDS 的 chunking 机制与动作窗（已核验）**：`rlds_dataloader/datasets/rlds/dataset.py` 的 `chunk_act_obs(window_size, future_action_window_size)` 会把 `action` 扩成 `window_size + future_action_window_size` 的序列。  
- **需要你本地再确认的点（避免踩坑）**：  
  - `RLDSBatchTransform` 当前实现默认只读取 `observation.image_primary`（未显式拼接 wrist 图像），因此**训练侧多相机是否生效**取决于你本地版本是否改过这段逻辑。  
  - `RLDSBatchTransform` 里 `actions` 字段只在 `window_size > 1` 时赋值；如果你按 `run_unifolm_vla_train.sh` 把 `window_size=1` 直接跑，需要先本地确认该分支是否会导致取不到 `actions`（建议训练前打印 batch keys/shape 做一次 sanity check）。  
  - `RLDSBatchTransform` 里给到 VLM 的文本 prompt 带有“预测 10 个关键轨迹点”的描述，这更像残留 prompt 模板；它不影响 action head 的训练目标，但可能影响 VLM hidden states 的语义分布，建议你在复现/改造时把 prompt 口径统一为“预测动作块”。

### 8.1 Sanity Check（30 分钟）：把“对齐”变成可执行核验

目标：在你真正跑 150k steps 之前，先用 30 分钟把最容易翻车的 4 件事核验掉：
- **(A) 训练 batch 里到底有没有 `action`，shape 是多少？**
- **(B) `window_size` 是否真的生效，动作窗长度是不是你以为的 `H`？**
- **(C) 训练侧到底喂了几路相机？（primary only vs primary+wrist）**
- **(D) `unnorm_key` / 统计量 / 维度是否一致？**

#### A) 训练 batch 的最小打印点（直接验证 `action` 是否存在）

在 Unitree repo 的训练入口 `src/unifolm_vla/training/train_unifolm_vla.py`，找到训练循环前或第一步之后插入打印（只跑 1 次即可）。你要看的就是 3 个 shape：

- `batch["action"]`：应为 `[B, T, ACTION_DIM]`
- `batch["state"]`：若开启 proprio，应为 `[B, PROPRIO_DIM]` 或 `[B, 1, PROPRIO_DIM]`（取决于后续 `unsqueeze`）
- `NUM_ACTIONS_CHUNK / ACTION_DIM / PROPRIO_DIM`：应与你期望的平台常量一致

建议打印内容（示意）：

```python
# 在拿到 batch_vla 后立刻打印一次（只打印 rank0）
if (not dist.is_initialized()) or dist.get_rank() == 0:
    print("batch keys:", batch_vla.keys())
    print("action shape:", None if "action" not in batch_vla else tuple(batch_vla["action"].shape))
    print("state shape:", None if batch_vla.get("state", None) is None else tuple(batch_vla["state"].shape))
    print("constants:", NUM_ACTIONS_CHUNK, ACTION_DIM, PROPRIO_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE)
    raise SystemExit("Sanity check done.")
```

判定标准：
- 看到了 `action shape == (B, T, ACTION_DIM)`，且 `ACTION_DIM` 与你选的平台一致 → 过
- `action` 不存在/shape 不对 → **先别训练**，先按 C/D 修

#### B) 动作窗长度到底是多少（window_size / future_action_window_size 的真实语义）

repo 的 RLDS 管线会在 `rlds_dataloader/datasets/rlds/dataset.py` 里执行：
- `chunk_act_obs(window_size, future_action_window_size)`

这会把 action 的时间维扩成：
- \(T = window\_size + future\_action\_window\_size\)

而本仓库的 `datasets.py` 默认设置：
- `future_action_window_size = NUM_ACTIONS_CHUNK - 1`

因此“你以为的 chunk 长度 \(H\)”与“真实喂给模型的 action 序列长度 \(T\)”之间可能差 1（或更多）。  
**最硬核的核验方式**：不要靠推理，直接看 A 里打印出来的 `T`。

#### C) 训练侧多相机是否生效（primary + wrist 还是 primary only）

两个层面要核验：
- **数据加载层**：`datasets.py` 会根据 `data_mix` 选择 `load_camera_views`，Unitree/G1 默认是 `("primary", "left_wrist", "right_wrist")`。这只说明 RLDS 会把 wrist 图像解出来。  
- **喂给 VLM 的层**：`RLDSBatchTransform.__call__` 目前只从 `rlds_batch["observation"]["image_primary"]` 构造 `images` 列表（你在 A 的打印里也能侧面验证：`pixel_values` 对应的 image 数量）。

核验方法：
- 直接在 `RLDSBatchTransform.__call__` 里打印 `rlds_batch["observation"].keys()`，确认是否含 `image_left_wrist` / `image_right_wrist` 等 key。  
- 再打印被拼进 prompt 的 images 数量（例如 `len(images)`）。

判定标准：
- `observation` 里确实有 wrist key，但 `len(images)==window_size`（只吃 primary）→ 说明“训练侧多相机没启用”；你要么接受，要么按 8.2 的补丁点改造。

#### D) 统计量与 `unnorm_key`（最常见的真机发散来源）

server 侧 `deployment/model_server/run_real_eval_server.py` 会：
- `self.norm_stats_action = vla.norm_stats[self.args.unnorm_key]['action']`
- `self.norm_stats_proprio = vla.norm_stats[self.args.unnorm_key]['proprio']`

核验方法：
- 打开你训练输出目录下的 `dataset_statistics.json`，确认包含你将要传入的 `unnorm_key`（比如 `g1_stack_block`）。  
- 确认其中 `action` / `proprio` 的统计量维度与 `ACTION_DIM / PROPRIO_DIM` 一致（mask/q01/q99 的长度）。

### 8.2 最小“补丁点”清单：你真要改，多半就改这 3 处

> 这不是让你现在就改代码，而是把“如果不一致，应该改哪里”一次性说清楚，避免盲改。

1) **训练侧加入 wrist 图像**（让训练与真机 multi-image 对齐）  
文件：`src/unifolm_vla/rlds_dataloader/datasets/datasets.py` 的 `RLDSBatchTransform.__call__`  
要点：从 `rlds_batch["observation"]` 把 `image_left_wrist`/`image_right_wrist`（或对应 key）追加进 `images` 列表，然后保持顺序一致（primary → left_wrist → right_wrist）。

2) **让 `window_size=1` 时也能拿到 `actions`**  
文件：同上 `RLDSBatchTransform.__call__`  
要点：目前只有 `window_size>1` 才设置 `batch_input['actions']`。如果你常用 `window_size=1`，就要在该分支也把动作序列塞进去（至少 `actions = actions[:, :]`），否则训练侧 `collate_fn` 很可能拿不到 `actions`。

3) **统一 prompt 口径为“predict action chunk”**  
文件：同上 `RLDSBatchTransform.__call__`  
要点：把“10 key trajectory points”的残留 prompt 改为与训练目标一致的描述（例如“predict an action chunk of length H”）；这会改变 VLM hidden states 的语义分布，属于“有收益也有风险”的改动，建议你先固定其它变量跑一个小 ablation（比如 5k steps）再决定。

[← Back to Theory](./README.md)

