# LaST0.5：以交互为中心隐空间推理的通用世界动作模型 (LaST0.5: A Generalist World Action Model with Interaction-Centric Latent Reasoning)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-11-12
>
> **论文**：LaST0.5: A Generalist World Action Model with Interaction-Centric Latent Reasoning
> **链接**: https://arxiv.org/abs/2611.14048
> **核心定位**：在 world model 与 action generation 之间插入 interaction-centric latent bottleneck，并用 progress consistency 自适应决定执行 horizon；论文表 1-3 中相对 LaST0 提升 RLBench 82.0→88.0、LIBERO 98.1→99.5、真实平均 43.8→63.8。
>
> **证据版本**：arXiv:2611.14048v1（2026-11-10）HTML 全文，对应 v1 PDF 正文。候选摘要仅用于定位；来源记录未给出摘要验证结果（abstract_verification=null），候选 snippet 在 “local robot-object int...” 处截断，本文未用其补写任何主张。论文未给出代码仓库，候选 repo_url 为空；本文不写仓库链接。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 世界模型的全局视频表示对局部机器人-物体交互不够敏感；LaST0.5 用交互线索监督一个 compact latent，再用 value 单调性决定执行多长。 |
| 适合精读 | 做 WAM/VLA 架构、latent reasoning、value-guided execution 的读者，重点看 §1.2、§2、§3.3 和表 1-4。 |
| 可以跳过 | 只关心纯 VLA、不做 world model 的读者，读摘要和表 1-3 即可。 |
| 落地可行性 | 中：模块边界清晰，但依赖 2B 级 world model、离线交互标注和多阶段推理；真实平均成功率 63.8%，长程末段仅 30.0%。 |
| 主要风险 | LIBERO 报告训练中最佳成功率；真实任务仅 4 个、每任务 20 rollouts；AWAT 依赖 progress 单调假设。 |

💡 **X-Ray 开场**

这篇论文解决什么问题？现有 WAM 把 world model 的全局物理表示直接迁移到动作生成，但操作任务真正需要的是机器人-物体局部交互，二者存在表示错配。

发现了什么？作者在未来帧上叠加末端运动曲线和目标物体 affordance box，用这些交互线索监督一个 3-token 的 latent 空间，作为 world expert 与 action expert 之间的中间层；再用 progress value 判断 imagined future 是否可靠。

对 VLA 研究者意味着什么？世界模型不是越大越全局越好；“交互中心抽象 + 可靠性门控”可能比单纯增大视频预测更影响操作成功率。这是作者主张，本文的实验支持主要来自 LIBERO-Spatial 消融和真实四任务。

📍 **研究全景时间线**

```
2023  LIBERO 基准
  → 2024  OpenVLA / Mixture-of-Transformers / UMI
  → 2025  π0.5 / Cosmos-Predict2.5 / Janus
  → 2026  LaST0 / Fast-WAM / Cosmos Policy / Motus
  → 2026-11  LaST0.5（本文，arXiv v1 2026-11-10）
```

本文位置：在 “world model → action” 之间加一个 interaction-centric latent，并用 value 做执行时可靠性判断。局限：真实任务平均 63.8%，长程最后阶段 30.0%，LIBERO 取训练中最佳，尚不能证明跨 embodiment 通用。

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 模块 | 归属/初始化 | 输入 → 输出 | 时序单位 | 训练 vs 推理 |
|------|-------------|-------------|----------|--------------|
| World Expert | Cosmos-Predict2.5-2B，预训练 diffusion transformer | 5 个历史主视角帧 t-4..t、语言、噪声未来 12 帧 → 未来视频 latent / KV | 未来帧 t+1..t+12，256×256 | 训练保留 Cosmos 生成目标 L_world；推理 2 个 denoising steps；训练时对 latent/action 屏蔽 K/V 梯度 |
| Latent Expert | 从 Janus 初始化 | 语言 tokens + M 个 learnable query tokens，条件于 world future → M 个 interaction-centric latent tokens | 默认 M=3，对应 t+4、t+8、t+12 的摘要 | 训练用 cosine 对齐 SigLIP 目标；推理单次并行前向 |
| Action Expert | 从与 latent expert 相同的语言 backbone 初始化，但独立参数 | 图像 tokens、timestamp tokens、24 个连续 action tokens → 12 步 action chunk | 12 个动作步，每步 2 个 token（wrist + actuator） | 训练 flow-matching L_act；推理 10 个 denoising steps（默认 n=10） |
| Value Head | 轻量 MLP，接在 world expert 去噪后的未来帧特征上 | world DiT hidden tokens H_i → 每段 1 个 progress value | 默认 M=3 段，每段 4 个 future frames | 冻结 base model 后回归训练；推理用于 AWAT |
| AWAT | 执行策略，不是独立网络 | value 序列 v̂_1..v̂_M → 执行/截断/重采样 | 默认每 4 个动作步一个 value | 仅推理时生效；训练时 value head 单独训练 |

这张表要逐模型读：Cosmos 只属于 world expert，Janus 只用于初始化 latent expert，action expert 虽然与 latent expert 同语言 backbone 但参数独立。论文没有说三者共享权重；相反，MoT 明确“共享 global attention stream，保持 independent expert weights”。

### 1.2 关键机制 (Key Mechanism)

- **为什么需要 latent**：作者认为 world expert 的视频预测被背景和大范围环境特征主导，直接把它喂给 action expert 会稀释局部交互。latent expert 把未来帧压成 3 个 token，并用末端运动曲线和目标物体 box 监督，使这 3 个 token 更聚焦“机器人正在操作什么”。
- **为什么用 MoT + blockwise-causal attention**：三个 expert 在同一个 attention stream 里，但各自有独立权重；信息流被限制为 world → latent → action。World expert 内部仍双向可见，latent 段 L_i 只看 W_1..W_i 和 L_1..L_i，action 段 A_i 只看 W_1..W_i、L_1..L_i、A_1..A_i。作者在附录 A.2 特别说明：这不是严格时间隔离，因为 world expert 内部双向注意力会让后续 world 段间接影响早期表示；mask 限制的是跨 expert 的直接条件。
- **为什么 detach world K/V**：训练时把 world expert 的 keys/values 从 latent 和 action 的梯度中断开，把预测的未来当作只读上下文，保护预训练物理先验。这是一个工程上很关键的选择：action loss 不会反向破坏 world model。
- **为什么 value 不是外部 reward**：AWAT 的 value head 接在 world expert 的未来帧特征上，预测的是“第 i 段未来帧对应的任务进度”，不是环境 reward，也不是候选动作排序器。作者强调它与 forward model 表示时间对齐，用来判断 imagined future 是否还值得执行。
- **为什么反转/重复帧增强**：专家演示的进度基本单调，但模型生成的未来可能停滞或回退。训练时把一部分 clip 时间反转（进度递减）或重复帧（进度恒定），让 value head 见过非单调模式。
- **为什么 latent 并行生成**：M 个 latent token 在一次前向中全部生成，而不是自回归。附录 E.3 的消融显示并行 99.6、自回归 99.4、teacher forcing 97.5；作者据此选择并行以降低推理延迟。

⚡ **Eureka Moment**：这篇论文最核心的想法不是“再训练一个更大的 world model”，而是**在 world model 和 action 之间加一个由交互视觉线索监督的 compact latent bottleneck，并把 progress 单调性当作执行可靠性的门控信号**。前者解决表示错配，后者解决 imagined future 不可靠时仍被执行的问题。

### 1.3 信息流/架构图 (Flow / Diagram)

```
语言 l + 历史帧 I_t (t-4..t, 5 帧)
        │
        ▼
World Expert (Cosmos-Predict2.5-2B)
  预测 V: t+1..t+12（12 帧，256×256）
  K/V detach（训练时阻断梯度）
        │
        ├──────────────► Value Head g_φ
        │                 每段 1 个 progress v̂_i
        ▼
Latent Expert (Janus 初始化)
  语言 tokens + M=3 learnable queries
  → Z：3 个 interaction-centric latent tokens
        │
        ▼
Action Expert (flow matching)
  图像 tokens + timestamp tokens + 24 action tokens
  → a：12 步 action chunk
        │
        ▼
AWAT：v̂_i > v̂_{i-1} 才继续执行；下降则截断并重观测；
      初始值低于上一周期则并行采样多个 world futures，选 value 最高者
```

注意图中的“未来 12 帧”是预测帧，不是历史上下文；“3 个 latent tokens”是未来摘要，不是记忆窗口；“12 步 action chunk”是未来动作，不是历史缓存。

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：

```
video → interaction latent → action，且 progress 单调才执行
```

先给目标：学习一个策略，使动作生成同时受益于世界模型的物理先验和交互中心的紧凑表示；再用 value 判断世界模型当前预测是否可靠。

联合预测分布：

```
p(a, Z, V | c_t) = p(V | c_t) · p(Z | V, c_t) · p(a | Z, V, c_t)
```

latent 监督：

```
L_lat = 1 - (1/M) · Σ_{i=1..M} cos(ẑ_i, z_i_GT)
```

总损失：

```
L = λ1·L_world + λ2·L_lat + λ3·L_act
```

value 预测与监督：

```
v̂_i = g_φ(Pool(H_i))
v_i_GT = t_i / T
L_value = (1/M) · Σ_{i=1..M} ℓ_reg(v̂_i, v_i_GT)
```

动作空间标准化：

```
δT = T_t^(-1) · T_{t+1}
δR = R_t^T · R_{t+1}
δt = R_t^T · (t_{t+1} - t_t)
```

| 符号 | 含义 |
|------|------|
| c_t | 当前观测-指令上下文 (I_t, l) |
| V | 未来视频，默认 12 帧 |
| Z | interaction-centric latent 序列，默认 M=3 |
| a | action chunk，默认 12 步 |
| M | 时间对齐段数，默认 3 |
| H | action horizon，默认 12 步 |
| z_i_GT | 由带交互线索的未来帧经冻结 SigLIP + 可训练 MLP 得到的目标 |
| v̂_i | 第 i 段 progress 预测 |
| t_i | 第 i 段 4 个 future frames 的中位环境时间步 |
| T | episode 长度 |
| H_i | world DiT 中与第 i 个 video latent 关联的 hidden tokens |
| g_φ | 轻量 MLP value head |
| ℓ_reg | 回归损失，论文未指定具体形式 |

**直觉**：分解式强制信息流按 video → interaction → action 走；cosine 损失让 3 个 latent token 对齐“带运动曲线和物体框”的未来帧表示，而不是重建整张图；value 回归把 episode 时间进度压成一个标量，AWAT 再用这个标量的单调性决定执行到哪一段。

### 2.1 时间跨度与记忆边界

这是全文最容易混淆的地方，必须把“已观测历史帧”和“未来预测帧”分开：

```
已观测历史：t-4, t-3, t-2, t-1, t
  K=5，均匀间隔 Δt，首末跨度 = (5-1)·Δt = 4Δt

未来预测：t+1, t+2, ..., t+12
  H=12，终点距当前参考 t = 12·Δt

latent 默认采样点：t+4, t+8, t+12
  只有 3 个 token，但跨度覆盖 12Δt；这是未来摘要，不是历史

action chunk：12 个动作步，每步 2 个 action token
  动作步与视频帧不是同一单位，不能直接等同
```

若假设 Δt=0.1 s（对应语料 10 Hz），则历史 5 帧跨度 0.4 s，未来 12 帧终点距当前 1.2 s。但论文没有明确 world expert 未来帧的采样率，也没有给真实控制频率；因此这只是**条件性换算**，不能写成实测延迟。

逐模型归属记忆边界：

- **World Expert**：历史上下文是 5 个主视角帧 t-4..t；causal video tokenizer 把 5 个历史帧压成 2 个 latent frames，把 12 个未来帧压成 3 个 latent frames。这是 tokenizer 压缩，不是“记忆窗口”。
- **Latent Expert**：M 个 learnable query tokens 是推理 token，不是历史帧缓存；它们条件于 world expert 的未来表示，不存储过去。
- **Action Expert**：12 步 action chunk 是未来动作；它通过 attention 访问 W_1..W_i 和 L_1..L_i，这是条件上下文，不是独立记忆模块。
- **AWAT value**：每段 value 对应未来 4 个帧/4 个动作步的进度估计，不是历史观测。

**不能把连续未来 chunk 的预测区间直接当作历史上下文或记忆窗口。** 也不能把本文默认的“每 4 步一个 value、M=3”写成所有实验共有的记忆窗口；正文图 3(c) 的 temporal granularity 消融明确试过 1、2、4、12 步的不同粒度，默认 4 步只是本文设置。

## 3. 带数字走一遍：玩具例子 (Worked Example)

### 3.1 假设的 AWAT 执行例子

以下为**假设数值**，不是论文实测，只用于验算 AWAT 的截断逻辑。

设 action chunk 为 12 步，M=3，每段 4 步；上一周期 progress 估计 v0=0.20。当前预测：

```
v̂_1 = 0.32
v̂_2 = 0.41
v̂_3 = 0.40
```

按论文规则“v̂_i > v̂_{i-1} 才继续执行”：

- v̂_1=0.32 > 0.20：执行第 1 段 4 步；
- v̂_2=0.41 > 0.32：执行第 2 段，再 4 步；
- v̂_3=0.40 < 0.41：停止，不执行第 3 段。

所以本次执行 8 步后重新观测。若按 10 Hz 动作步长假设换算，执行时长约 0.8 s；若全部 12 步执行则约 1.2 s。这个换算是条件性的，论文没有给真实控制频率。

若 v̂_1=0.18 < v0=0.20，则按论文“初始预测值低于上一周期，整个预测不可靠”的规则，执行 0 步，并行采样多个 world futures，选 value 最高的 KV cache 重跑。论文没有给候选数量，因此重采样延迟不能量化。

### 3.2 论文实测平均成功率的验算

真实四任务平均成功率按表 3 的全任务完成口径计算：

```
LaST0.5: (80.0 + 95.0 + 50.0 + 30.0) / 4 = 63.75 ≈ 63.8
π0.5:    (75.0 + 85.0 + 30.0 + 20.0) / 4 = 52.50
Cosmos:  (60.0 + 85.0 + 20.0 + 10.0) / 4 = 43.75 ≈ 43.8
LaST0:   (60.0 + 75.0 + 25.0 + 15.0) / 4 = 43.75 ≈ 43.8
```

验算与论文表 3 的 Avg. 列一致。说明平均是按四个任务的全任务成功率等权平均，Screw Fastening 取 S2、Parcel-Box Packing 取 S3 作为全任务完成口径。这个细节很重要：不能把 S1 的 90.0 或 100.0 当成该任务的全任务成功率。

## 4. 工程视角 (Engineering View)

| 工程维度 | 论文信息 | 条件性分析 |
|----------|----------|------------|
| 推理阶段 | world expert 2 个 denoising steps；latent expert 1 次并行前向；action expert 10 个 denoising steps（默认 n=10） | 比纯 VLA 多 world + latent 两段；但 action 只 10 步，latent 不是自回归，省去逐 token 解码 |
| 执行步数 | action chunk 12 步；AWAT 按 value 单调性截断 | 默认 M=3、每段 4 步，因此可执行 0/4/8/12 步；这是对规则的解读，论文未逐项列出执行步数 |
| 控制频率 | 预训练语料统一 10 Hz；真实控制频率未给 | 若按 10 Hz 换算，12 步约 1.2 s，4 步约 0.4 s；不能把 10 Hz 语料频率直接当成部署控制频率 |
| 重采样成本 | 初始 value 低于上一周期时，并行采样多个 world futures | 候选数未给；延迟随候选数近似线性增长，这是条件性推断 |
| 显存/吞吐 | 8×NVIDIA H20 训练；未给推理显存、延迟、吞吐 | 不能编具体数字；2B world expert + Janus + action backbone 的部署成本需要实测 |
| 分辨率 | world expert 256×256；VLM 分支 224×224 | 两个分支分辨率不同，部署时要分别预处理 |
| 数据管线 | 训练时离线生成 2D 运动曲线和 Grounding DINO 物体框；推理不需要 | 训练数据标注依赖相机参数、末端 TCP 投影和检测器质量；推理管线可省，但训练分布会受影响 |
| 抖动/阈值 | value 每 4 步一个；1 步粒度 98.9，低于 4 步 99.7 | 作者解释细粒度会引入高频波动；工程上可加平滑/滞后阈值，但论文未做 |

**工程含义**：LaST0.5 的模块边界适合“world model 只读、latent 轻量、action 高频”的分工。World expert 的 K/V detach 让 action loss 不破坏预训练物理先验；latent expert 只有 3 个 token，适合做低带宽瓶颈；action expert 用 flow matching 一次生成 12 步，AWAT 再决定执行多长。真正的部署瓶颈在 world expert 的 2B 规模和重采样策略，而不是 latent 或 action 头。

## 5. 数据与评测 (Data & Eval)

### 5.1 预训练

论文预训练语料为 2,479,441 episodes、536M frames、14,892.6 小时，统一到 10 Hz。组成如下（论文表 5）：

| 来源 | 模态 | Episodes | Hours |
|------|------|----------|-------|
| Genrobot | UMI | 656,993 | 9,088.0 |
| AgiBot | Dual-arm | 996,381 | 2,479.0 |
| InternData-A1 | Sim + Real | 344,607 | 1,429.2 |
| DROID | Single-arm | 94,030 | 766.2 |
| EgoDex | Human ego | 314,839 | 727.4 |
| RoboCOIN | Dual-arm | 42,828 | 313.2 |
| RoboMIND | Dual/Single | 29,763 | 89.6 |
| 合计 | — | 2,479,441 | 14,892.6 |

UMI 数据占 9,088/14,892.6≈61.0%，是主体；真实双臂遥操作、单臂、仿真和人类 egocentric 数据混合。所有源的动作标签被丢弃，按统一末端执行器坐标重新推导：wrist 用相对 SE(3) 变换 δT，gripper 用开合趋势，hand 用下一帧 5×3 指尖表示。这个标准化是跨 embodiment 训练的关键工程细节。

**数据开放范围**：论文只描述语料来源与统计，未声明代码、模型权重或数据开放；候选记录 repo_url 为空。本文不写仓库链接，也不推断开源。

### 5.2 仿真评测

| 基准 | 任务/数据 | 输入 | 评测协议 |
|------|-----------|------|----------|
| RLBench | 10 个任务，Franka Panda，CoppeliaSim；每任务 100 条演示 | 单前视 RGB，VLM 分支 224×224 | 3 次独立 run，每次每任务 20 rollouts，共 60 rollouts/task；报告 mean ± std |
| LIBERO | Spatial / Object / Goal / Long 四个 suite，官方训练数据 | agent-view + wrist-view，224×224 | 报告训练过程中最佳成功率；100 epochs |

【论文实测】RLBench 平均 88.0±3.0，LIBERO 平均 99.5（Spatial 99.7、Object 100.0、Goal 99.1、Long 99.3）。【作者主张】这说明 interaction-centric latent + AWAT 在仿真上达到 SOTA。

【条件性分析】LIBERO 的“best success rate obtained during training”不是固定最后 checkpoint，和基线表 2 的可比性需要谨慎；表 2 标题已说明该表报告训练中最佳成功率，基线也按该口径列入；但 best-during-training 不是固定最后 checkpoint，跨方法比较仍需注意选择细节。RLBench 的 3×20 rollouts 有明确单位，但 10 个任务的难度差异很大，均值会被 Close Box 等简单任务拉高。

### 5.3 真实机器人设置

| 项目 | 设置 |
|------|------|
| 机器人 | 7-DoF Tianji Marvin 双臂，平行夹爪 |
| 相机 | 头部 ZED 2i + 两个腕部 Insta360 GO 3S，三路 RGB，224×224 |
| 演示 | PICO VR 遥操作，每任务 150 条轨迹 |
| 任务 | Bag Pulling、Object Organization、Screw Fastening（S1 工具使用/S2 插入）、Parcel-Box Packing（S1 放置/S2 折叠/S3 封箱） |
| 评测 | 每任务 20 rollouts；全任务成功率用于平均；Screw 取 S2、Parcel 取 S3 |

【论文实测】真实平均 63.8%，π0.5 52.5%，Cosmos Policy 43.8%，LaST0 43.8%（论文表 3）。按表计算，LaST0.5 比 LaST0 高 20.0 个百分点，比 π0.5 高 11.3 个百分点。长程任务最后阶段仍是瓶颈：Parcel S3 为 30.0%，Screw S2 为 50.0%。

【条件性分析】真实任务只有 4 个、每任务 20 rollouts，置信区间不会很窄；平均 63.8% 说明还有大量失败。作者展示的失败案例包括拉链卡住、黄瓜碰倒篮子、胶带打滑、毫米级螺丝错位，分别对应机械阻力、几何建模、夹爪摩擦和视觉伺服精度。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

| 能力 | 证据 | 条件 |
|------|------|------|
| 多任务仿真 | LIBERO 99.5，RLBench 88.0 | 微调后；LIBERO 取训练中最佳；RLBench 3 runs ×20 rollouts |
| 真实双臂长程 | 平均 63.8%；Object Organization 95.0% | 4 个任务、每任务 20 rollouts、150 demos |
| OOD 位置/物体/背景 | Position 40.0 vs π0.5 15.0；Background 72.5；Object 55.0 | 两个真实任务上的位置/实例/背景变化 |
| 交互注意力集中 | 附录 F.1 的 action-to-vision attention maps | 定性证据；不是成功率 |
| 失败预警 | value 下降对应失败案例 | 附录 F.2 的可视化；未给定量 AUROC 等指标 |

**不能做什么**：

- 不能证明跨 embodiment 通用：真实实验只在一个 Tianji 双臂平台，预训练虽混合多 embodiment，但没有在更多真实机器人上验证。
- 不能证明开放世界泛化：OOD 只测位置、物体实例、背景三类，不是全新任务或全新物理交互。
- 不能证明力控/接触-rich 任务可靠：失败案例明确提到拉链阻力、胶带摩擦、螺丝毫米级对齐，说明纯视觉策略仍有物理边界。
- 不能把 LIBERO 99.5 当成“普遍 99.5%”：这是四个 suite 的平均，且取训练中最佳；真实平均只有 63.8%。

### 6.1 隐含假设 (Hidden Assumptions)

1. **世界模型未来与动作时间对齐**：AWAT 假设第 i 段 future frames 的 value 能代表第 i 段动作的可靠性；如果 world model 的时间对齐有偏，value 会误导执行。
2. **progress 标签线性有效**：v_i_GT=t_i/T 假设 episode 内任务进度与时间线性；对长程、阶段式任务，这个假设可能过强。反转/重复增强只是启发式，不等于真实失败分布。
3. **交互线索足够代表交互**：2D 末端轨迹 + Grounding DINO 物体框能覆盖动作相关区域；如果检测器漏检或相机标定有误，latent target 会偏。
4. **统一动作空间不丢关键信息**：所有 embodiment 被压到相对末端执行器 + 夹爪/手表示；这可能丢失力、触觉或关节级约束。
5. **10 Hz 足够**：预训练统一 10 Hz，但真实高频接触任务可能需要更高控制频率；论文未给真实控制频率。
6. **LIBERO best-during-training 口径**：表 2 标题已说明该表报告训练中最佳成功率，基线也包含在内；仍应注意 best-during-training 不是固定最后 checkpoint。
7. **value 单调性是可靠性判据**：AWAT 假设 progress 上升等于可靠、下降等于不可靠；如果 world model 自信地预测错误但 value 仍上升，门控会失效。
8. **预训练分布覆盖测试**：61% UMI 数据可能让动作先验偏向 UMI 式末端控制，真实双臂任务的 grounding 是否充分，论文未做数据配比消融。

## 7. 与相关工作对比 (Comparison)

| 方法 | 论文中的定位 | 世界模型/动作生成 | 交互/进度机制 | 论文列出的结果 |
|------|--------------|-------------------|---------------|----------------|
| OpenVLA-OFT | VLA 基线 | 直接视觉-语言-动作 | 无 | RLBench 45.0；LIBERO 97.1 |
| π0.5 | VLA 基线 | 直接 VLA | 无 | RLBench 65.0；LIBERO 96.9；真实 52.5 |
| HybridVLA | VLA 基线 | 扩散 + 自回归 | 无 | RLBench 74.0；LIBERO 未列 |
| Cosmos Policy | WAM 基线 | 视频模型微调为策略 | 无本文式 latent | RLBench 77.0；LIBERO 98.5；真实 43.8 |
| Fast-WAM | WAM 基线 | 世界动作模型 | 无本文式 latent | RLBench 63.0；LIBERO 97.6 |
| Motus | WAM 基线 | 统一 latent action world model | 无本文式 latent | LIBERO 97.7；RLBench 未列 |
| LaST0 | 前代基线 | 论文称 VLA/前代 | 无本文式 AWAT | RLBench 82.0；LIBERO 98.1；真实 43.8 |
| LaST0.5 | 本文 | Cosmos world + Janus latent + flow action | interaction-centric latent + AWAT | RLBench 88.0；LIBERO 99.5；真实 63.8 |

这张表只写论文明确列出的数字和定位，不推断基线内部架构。LaST0.5 的相对优势在 RLBench（88.0 vs 82.0）和真实平均（63.8 vs 43.8）更明显；LIBERO 上从 98.1 到 99.5 是小幅提升，但论文的 best-during-training 口径需要留意。

**面试 Tip**：被问“LaST0.5 和一般 WAM 有什么区别”，一句话答：**它不是把全局视频表示直接喂给动作，而是先用交互视觉线索监督一个 3-token latent bottleneck，再用 progress 单调性决定执行 horizon；代价是多一个 world+latent 推理阶段和一个 value head。**

## 8. 精读建议 (Reading Guide)

- **值得精读原文的人**：
  - 做 WAM/VLA 架构的研究者：重点看 §1.2 的 MoT 信息流、§2 的分解式与 latent 监督、附录 A.1/A.2 的 token 与 attention mask。
  - 做机器人部署的工程师：重点看 §3.3 的 AWAT 执行规则、§4 的推理步数、§5.3 的真实设置与失败案例。
  - 研究 value-guided execution 的人：重点看 §1.2 的 AWAT 与 §2 的 value 监督、正文图 3(c) 的粒度消融、附录 F.2 的 value 曲线。

- **建议章节路径**：先读 §1.2 + 图 2，理解三个 expert 的职责；再读 §2 + §2.1，分清历史帧、未来帧、latent token、action chunk 的单位；然后读 §3.3 + 表 1-4，判断实验条件；最后看附录 A.1/A.2 和附录 E/G。参考文献和附录 B.2 的坐标变换细节可以后置。

- **不值得精读的理由**：如果你不做机器人学习、不关心 world model，或者已经非常熟悉 MoT/latent action 且只想知道结果，读摘要和表 1-3 即可。论文的真实任务只有 4 个，平均 63.8%，不足以支撑“通用操作”的强结论。

---

[← Back to Theory](./README.md)

**关键引用**：论文 https://arxiv.org/abs/2611.14048 ；HTML 全文 https://arxiv.org/html/2611.14048 。本文所有数字来自该论文全文；未引用外部仓库或额外 URL。