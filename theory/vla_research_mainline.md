# VLA 研究主线梳理：从 ACT/DP 基线到“数据 × 感知 × 后训练”的工程化闭环

> 这份笔记的目标不是重复解释 ACT / Diffusion Policy（DP）怎么实现，而是回答一个更“研究-工程都关心”的问题：**为什么 ACT 和 DP 仍然是最常用 baseline，以及真正能把真实世界成功率拉上去的主线改进在哪里**。

## 相关导读（仓库内）

- **ACT**：[`act.md`](./act.md)
- **Diffusion Policy**：[`diffusion_policy.md`](./diffusion_policy.md)
- **VLA 架构总览**：[`vla_arch.md`](./vla_arch.md)
- **动作表示（Delta pose / Absolute pose 等）**：[`action_representations.md`](./action_representations.md)
- **数据与数据格式**：[`data.md`](./data.md)
- **损失函数手册（BC / Diffusion / Flow / RL）**：[`vla_loss_functions_handbook.md`](./vla_loss_functions_handbook.md)
- **感知技术总览（含位姿估计、跟踪、多视角/点云）**：[`perception_techniques.md`](./perception_techniques.md)

---

## 0. 为什么 ACT 和 Diffusion Policy 仍是“默认 baseline”

在机器人操控里，ACT/DP 之所以长期占据 baseline，不是因为“论文少”，而是因为它们分别覆盖了两类最常见的工程约束：

- **ACT（动作分块/序列建模）**：推理路径短、部署简单、对小数据和实时控制更友好。可参考 LeRobot 的 ACT 说明与开源实现。([LeRobot ACT 文档](https://huggingface.co/docs/lerobot/act?utm_source=openai))
- **Diffusion Policy（动作生成分布）**：天然适配 **多峰（multi-modal）动作分布**、能输出更平滑的动作序列；在标准 benchmark 上已被反复验证为强基线。([Diffusion Policy 项目页](https://diffusion-policy.cs.columbia.edu/?utm_source=openai))

> 现实里很多“改进工作”最后都在做同一件事：**让 ACT 更稳、更泛化，或让 diffusion 更快、更可控**。

---

## 1) 数据规模化：从“训练一个任务”到“预训练 + 适配”

### 1.1 这条主线在解决什么

把范式从「针对单任务训练一个 policy」推向「**先用大规模、多任务、多形态数据预训练**，再对具体平台/任务做适配」。

直觉上，它在提升两种能力：

- **跨任务的零样本/少样本能力**：把“语言/视觉理解 + 常识动作先验”做进模型。
- **组合泛化**：能在“新背景 + 新物体 + 新指令组合”下保持一定成功率。

OpenVLA 是这条路线的代表性开源落点之一。([OpenVLA 项目页](https://openvla.github.io/?utm_source=openai))

### 1.2 为什么“成功率不一定暴涨”（你提到的关键点）

你观察到：在一些高精度任务上，大模型预训练并不一定比“专用小模型”强，常见原因包括：

- **Embodiment mismatch（机体不一致）**：不同机械臂的动作空间、控制频率、夹爪/手型差异，会让同一条动作轨迹在另一个平台上变成“不可执行/不稳定”。
- **数据混合带来的动作语义漂移**：同一个语言指令或视觉场景，在不同机器人上可能对应不同的低层控制策略（动作等价类不同）。
- **动作表示不统一**：例如一个平台用 joint-space position control，另一个用 end-effector delta pose + impedance；如果不做“动作空间对齐”，预训练优势会被稀释。

### 1.3 工程上怎么把“数据规模化”做扎实

- **对齐动作表示**：优先把不同平台映射到统一的 action representation（例如 end-effector delta pose + gripper），并对控制频率做 resample/对齐（详见 `action_representations.md`）。
- **把“失败/恢复”也当作数据**：不要只收“干净成功 demo”，要显式收集偏离后的 recover 行为（与第 3 条主线强相关）。
- **记录元数据用于再训练**：相机位姿、末端执行器类型、控制模式、延迟（latency）、相机帧率等必须进数据 schema（见 `data.md`）。

---

## 2) 感知模块增强：把“看得更稳”作为成功率下限

### 2.1 这条主线在解决什么

很多真实世界失败并不是“动作模型不会做”，而是 **输入端不稳定**：遮挡、反光、背景干扰、视角变化、滚动快门/运动模糊、以及对称物体的姿态歧义。

所以这条线在做的是：

- **更强的视觉 backbone / 表征**（例如更好的 VFM/VLM），降低 domain shift。
- **更强的几何约束/3D 表示**，提升空间泛化与可控性。

DP3（3D Diffusion Policy）是“几何/3D 输入增强带来显著泛化收益”的代表之一。([DP3 RSS 论文页](https://www.roboticsproceedings.org/rss20/p067.html?utm_source=openai))

> 你的判断很关键：**感知增强经常提升鲁棒性下限，但不会自动“发明新的动作逻辑”**。

### 2.2 这条主线在工程里如何落地

- **把感知不确定性显式化**：输出 pose/confidence/uncertainty，让下游 policy 能做 risk-aware 决策（例如触发重观测/换视角/重初始化）。
- **从 object-centric 做起**：bbox→mask→keypoints/pose，尽量让 policy 看到的是“目标物体的状态”，而不是整张图的纹理噪声。
- **多视角/多帧融合**：对遮挡、对称物体、多解姿态特别有效（这部分可结合你已写的 6D pose 多视角融合方法论）。

---

## 3) RL 后训练 / On-policy 数据：让模型学会“从错误里爬回来”

### 3.1 你指出的核心矛盾：BC 的专家分布过窄

ACT/DP（以及大多数 BC）最常见的系统性失败是：

- 训练数据是“专家演示”，覆盖的是一个很窄的成功流形；
- 真实世界有噪声，系统 inevitably 会进入 **分布外（OOD）状态**；
- 长任务中误差累积，缺少 recover 行为会导致不可逆失败。

这个问题在 imitation learning 里有经典论述：**DAgger** 通过“在学习者分布上采集数据”来缓解 compounding error。([DAgger arXiv:1011.0686](https://arxiv.org/abs/1011.0686?utm_source=openai))

> 你写的“关键不是扩大完美数据，而是高效采集 on-policy 数据流水线”——这是非常工程化、也非常符合近两年的趋势判断。

### 3.2 这里的“RL 后训练”到底在做什么

把它拆成三个可操作的子问题：

- **(A) 怎么采集 on-policy 数据**：在线 rollout，系统性覆盖“滑了、偏了、遮住了、没抓稳”等状态。
- **(B) 怎么定义 reward / 成功信号**：成功率很稀疏时，需要引入 dense proxy（例如位姿误差、接触事件、阶段性里程碑）。
- **(C) 怎么让训练稳定**：把离线专家数据当作锚点（防崩坏），同时逐步扩大 on-policy 占比。

### 3.3 你可以直接复用的“on-policy 数据流水线”模板（工程视角）

- **触发式采集**：只在“置信度下降/接触异常/末端偏差超阈值”时开启高频记录，降低数据成本。
- **分阶段标注**：把任务拆成 approach / pre-contact / contact / manipulate / retreat，多阶段 reward 更好写。
- **Recover skill 库**：把“重新定位、轻微抖动、换抓取点、退回重来”等 recover 作为独立技能学习与评估。
- **安全护栏**：速度/力/关节限位的 safety controller 永远在 policy 外层兜底。

---

## 4) 一张“研究主线地图”（把三条主线接成闭环）

```text
        ┌──────────────────────────────┐
        │  Baseline: ACT / Diffusion   │
        └──────────────┬───────────────┘
                       │
                       v
┌──────────────────────────────┐
│ 1) 数据规模化：预训练+适配     │
│ - 多任务/多形态数据            │
│ - 动作表示对齐                 │
└──────────────┬───────────────┘
               │ feeds
               v
┌──────────────────────────────┐
│ 2) 感知增强：鲁棒性下限        │
│ - VFM/VLM 特征更稳             │
│ - 3D/几何/多视角约束           │
└──────────────┬───────────────┘
               │ provides uncertainty
               v
┌──────────────────────────────┐
│ 3) 后训练：on-policy + recovery │
│ - 覆盖 OOD 状态                │
│ - 学会“从错里回到正轨”          │
└──────────────┬───────────────┘
               │ updates
               v
        ┌──────────────────────────────┐
        │  新一轮数据与模型迭代（闭环） │
        └──────────────────────────────┘
```

---

## 5) 你这段梳理在面试里的“可追问点”清单

如果面试官深挖，你可以沿着这些问题展开（每个都能落到工程细节）：

- **数据规模化**：动作空间怎么对齐？不同控制频率怎么对齐？embodiment mismatch 怎么缓解？
- **感知增强**：怎么输出置信度？遮挡/对称物体怎么消歧？多视角融合怎么做权重？
- **后训练**：on-policy 数据怎么高效采？触发条件是什么？reward 怎么设计？怎么保证安全？

---

## 参考链接（外部）

- **Diffusion Policy 项目页**：`https://diffusion-policy.cs.columbia.edu/` ([链接](https://diffusion-policy.cs.columbia.edu/?utm_source=openai))
- **DP3 RSS 论文页**：`https://www.roboticsproceedings.org/rss20/p067.html` ([链接](https://www.roboticsproceedings.org/rss20/p067.html?utm_source=openai))
- **LeRobot ACT 文档**：`https://huggingface.co/docs/lerobot/act` ([链接](https://huggingface.co/docs/lerobot/act?utm_source=openai))
- **DAgger (Dataset Aggregation)**：`https://arxiv.org/abs/1011.0686` ([链接](https://arxiv.org/abs/1011.0686?utm_source=openai))
- **OpenVLA 项目页**：`https://openvla.github.io/` ([链接](https://openvla.github.io/?utm_source=openai))
