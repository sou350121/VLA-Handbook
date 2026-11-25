# 让行为流起来：Koopman 行为模型作为视觉运动灵巧操作的伪规划器 (Going with the Flow: Koopman Behavioral Models as Pseudo Planners for Visuo-Motor Dexterity)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-11-25
>
> **论文**：Going with the Flow: Koopman Behavioral Models as Pseudo Planners for Visuo-Motor Dexterity
> **链接**: https://arxiv.org/abs/2602.07413
> **核心定位**：把灵巧技能建模为“动作流 + 视觉流”的耦合潜在线性动力系统，用开环 rollout 充当伪规划器，并用预测视觉流偏差触发重规划；目标不是替代通用大模型，而是在特定任务族上以低推理成本换取时序一致性与可监控性。

**版本与证据说明**：本文依据来源提供的 arXiv:2602.07413v4 正文（HTML 版，标注 2026-11-20）。候选元数据摘要片段在 “creating a rigid trade-off between temporal” 处截断，不能覆盖正文摘要，故未用其补写。来源未提供可独立核验的官方 PDF 正文，因此本文不声称已核验 PDF；若需正式引用，应以同版本 PDF 复核。以下数字与主张均来自该正文，未使用外部材料。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | K-UBM 在 7 个仿真任务和 4 个真实任务上与 Diffusion、ACT 等基线可比；作者报告仿真平均成功率为第二高，真实平均成功率 85.0% 低于 ACT 的 92.5%，但轨迹更平滑、推理更快，并支持事件触发重规划。 |
| 适合精读 | 如果你在做多指灵巧操作、动作分块时序一致性、Koopman/动力系统策略、在线监控与重规划，重点看 §2、§3、§4.2、Appendix F/G/I/K。 |
| 可以跳过 | 如果你只关心大规模 VLA、视频世界模型的跨任务泛化，这篇的距离中等；它是任务特定行为模型，不是通用策略。 |
| 落地可行性 | 中：线性 rollout 推理便宜，但需要任务专用视觉特征、每任务演示、初始条件对齐和触发阈值校准；真实泛化尚未验证。 |
| 主要风险 | 线性动力学平滑接触不连续；开环伪规划在强扰动/变化物体动力学下脆弱；遮挡鲁棒性主要是机制性论证，缺同条件定量对照。 |

💡 **X-Ray 开场**

这篇论文问的是：灵巧操作一定要用 Diffusion/Transformer 这类高表达、固定 action chunk 的反应式策略吗？作者的答案是：可以把技能看成一个“行为动力系统”，让动作流和视觉流在 latent 空间里共同演化，再用线性 Koopman rollout 生成完整计划。对 VLA 研究者而言，它的意义不在“又一个 SOTA”，而在于提示了一条替代路径：预测视觉流不仅可以辅助动作，还可以作为运行时监控信号，让策略在“深思熟虑的开环计划”和“事件触发的反应”之间切换。

📍 **研究全景时间线**

```
1931 Koopman 算子
  → 2013 DMP 把技能视为动力系统
  → 2023 ACT（动作分块）/ KODex（Koopman 灵巧操作）
  → 2024 KOROL（视觉物体特征 + Koopman rollout）
  → 2025 Diffusion Policy 期刊版 / UVA（视频-动作联合）
  → 2026 K-UBM：动作流+视觉流统一 Koopman 伪规划器（本文）
```

本文位置：把 Koopman 从“手-物体状态预测器”推进到“动作流 + 视觉流联合行为模型”，并把视觉预测用于重规划。局限：单任务族、每任务训练、线性接触假设，真实任务仅 4 个。

## 1. 核心架构/方法总览

### 1.1 系统对比概览

K-UBM 的模块可以拆成“视觉特征 → 统一行为状态 → 谱编码 → 线性动力学 → 动作读出 → 运行时监控”。

| 模块 | 输入 | 输出 | 训练/推理时序 | 备注 |
|------|------|------|----------------|------|
| 视觉特征 | RGB 图像序列 | Object Flow 128 维特征，或 DynaMo embedding | 先从所有演示中提取/训练 | 不是策略本体；Table 1 的推理时间不含该步骤 |
| 统一行为状态 | 动作 `a_t`、视觉特征 `φ_t` | `ξ_t = [a_t, φ_t]` | 每时间步 | 视觉特征按平均 L2 范数缩放，平衡量纲 |
| 谱编码器 `g_θ` | `ξ_t` | `ψ_t` | 与 K 联合训练 | MLP；Table 7 隐层 `(128,256)`，lifting 维 256 |
| 状态包含 latent | `ξ_t`、`ψ_t` | `z_t = [ξ_t, ψ_t]` | 每时间步 | 动作直接嵌在 `z_t` 中，支持 decoder-free 读出 |
| Koopman 矩阵 `K` | `z_t` | `z_{t+1} = K z_t` | 多步 latent 预测训练 | 初始化为 identity；单独较小学习率 |
| 动作读出 | `z_t` | `a_t` | rollout 时直接取子向量 | 通用 UBM 需 `f_dec`；K-UBM 用 state-inclusive 避免显式 decoder |
| 监控/重规划 | 预测视觉流、观测视觉流 | 触发信号、新计划 | 在线每步监控 | flow 用 centroid 距离跳变；DynaMo 用相似度变化 |

**时序口径必须分清**：若均匀采样 `K` 帧，首末跨度是 `(K-1)×Δt`；若从当前参考 `t` 预测 `t+1` 至 `t+H`，终点距当前参考是 `H×Δt`。本文中 Diffusion 的 `T_o=2` 是历史观测长度，不是未来预测长度；`T_p/T_a` 是未来预测/执行长度；ACT 的 chunk 10/20/40 是未来动作块长度；K-UBM 的 `H=15/50` 是多步 latent 预测训练长度，不是历史记忆窗口。连续未来 chunk 的预测区间不能直接当作历史上下文或记忆窗口。

### 1.2 关键机制

- **技能是耦合动力系统，不是反应式映射**：标准策略学 `o_t → a_t` 或 `o_{t-H:t} → a_{t:t+C}`；UBM 学 `z_{t+1}=f_UBM(z_t)`，其中 `z_t` 同时编码动作与视觉行为状态。
- **伪规划器**：给定初始条件 `ξ_0`，K-UBM 可 rollout 任意长度 `t_p ≤ T`，一次生成整段 nominal 动作与视觉流；horizon 可灵活变化，不需要固定 action chunk。
- **state-inclusive lifting**：把 `ξ_t` 本身放进 `z_t`，使动作成为 latent 的子向量，推理时可直接读出；同时让线性动力学“锚定”在行为状态上，减少长程漂移。
- **线性 rollout 换效率**：`z_{t+1}=K z_t`，多步预测是 `K^l z_t`；相比 Diffusion 的迭代去噪或 Transformer 的序列推理，动作估计更便宜。
- **预测视觉流即运行时监控**：模型不仅预测动作，还预测视觉特征；预测与观测偏差超过阈值时，重初始化 `z_t` 并重规划。这是“模型作为自己的 runtime monitor”的关键。

⚡ **Eureka Moment**：把“技能”从观察→动作的映射，改成“动作流与视觉流共同演化的线性潜动力学”；视觉预测不是辅助输出，而是触发反应式重规划的信号。

### 1.3 信息流/架构图

```text
训练：
演示 {I_t, q_t, a_t}
  → 视觉特征 φ_t（Object Flow 或 DynaMo）
  → ξ_t = [a_t, φ_t]
  → ψ_t = g_θ(ξ_t)
  → z_t = [ξ_t, ψ_t]
  → 学习 K，使 z_{t+1} = K z_t
  → 多步损失：Σ_l ||K^l z_t - z_{t+l}||²

推理：
ξ_0 → z_0 → K^l z_0 → 读出 a_t 与预测 φ_t
          ↘ 与观测视觉流比较
              → 偏差跳变超阈值 → 重初始化 z_t → 新 rollout
```

注意：通用 UBM 定义需要 action decoder `f_dec`；K-UBM 的 decoder-free 来自 state-inclusive 设计，不应推广成“所有 UBM 都不需要 decoder”。

## 2. 数学核心

📌 **Napkin Formula**：

```text
z_{t+1} = K z_t,  z_t = [ξ_t; ψ_t],  ξ_t = [a_t; φ_t]
```

先给目标：学一个 latent 线性系统，使动作流与视觉流在联合 latent 中按 Koopman 算子演化；再用多步预测损失让编码器和 `K` 共同线性化。

核心公式：

```text
ξ_t = [a_t^T, φ_t^T]^T
ψ_t = g_θ(ξ_t)
z_t = [ξ_t^T, ψ_t^T]^T
z_{t+1} = K z_t
K^l z_t = l 步 latent 预测
```

多步线性一致性损失：

```text
L_K-coherence = E_t [ Σ_{l=1}^{H} || K^l z_t - z_{t+l} ||_2^2 ]
```

flow 特征的重规划触发规则：

```text
d_t = || centroid(observed_flow_t) - centroid(predicted_flow_t) ||
trigger if |d_t - d_{t-1}| > 3 × running_mean(previous jumps)
```

变量说明：

| 符号 | 含义 |
|------|------|
| `t` | 时间步 |
| `a_t` | 机器人目标关节命令，维度 `d_q` |
| `q_t` | 机器人关节状态；`o_t={q_t,I_t}` |
| `φ_t` | 视觉特征；Object Flow 为 128 维，DynaMo 为 ResNet-18 embedding |
| `ξ_t` | 统一行为状态，维度 `d_ξ=d_q+d_φ` |
| `ψ_t` | 谱编码器输出，Table 7 的 lifting 维为 256 |
| `z_t` | state-inclusive latent，维度 `d_z=d_ξ+d_ψ` |
| `K` | 可学习 Koopman 矩阵，`d_z×d_z` |
| `H` | 多步 latent 预测训练长度；仿真 15，真实 50（Appendix F） |

直觉：如果 `K` 学得准，`K^l z_t` 一次给出未来第 `l` 步的联合行为状态；动作和视觉流一起外推，因此时序一致性是“构造出来的”，不是靠 action chunk 平均出来的。代价是：线性系统会平滑尖锐接触，`K` 的谱性质也决定 rollout 是否漂移。

训练稳定性要点（Appendix F/J）：`K` 初始化为 identity；encoder 学习率 `5e-4`，Koopman 学习率 `5e-5`；使用 gradient clipping。作者解释：多步矩阵幂会放大梯度，`K` 更新慢一些、从 identity 开始，能让 latent 动力学逐步长出来。

## 3. 带数字走一遍：玩具例子

> 以下为**假设玩具例子**，只用于验算机制，不是论文实验数据。

**Toy A：线性 rollout 与多步损失**

假设 `d_ξ=1, d_ψ=1`，`z_t=[ξ_t, ψ_t]^T`，学到的

```text
K = [[1, 0.5],
     [0, 1  ]]
z_0 = [1.0, 0.2]^T
```

则：

```text
z_1 = K z_0 = [1.0 + 0.5×0.2, 0.2] = [1.1, 0.2]
z_2 = K z_1 = [1.1 + 0.1, 0.2]     = [1.2, 0.2]
z_3 = K z_2 = [1.2 + 0.1, 0.2]     = [1.3, 0.2]
```

若 `H=2`，真实目标恰为 `z_1=[1.1,0.2]`、`z_2=[1.2,0.2]`，则两项误差都为 0，loss 为 0。若真实 `z_2=[1.19,0.2]`，则第二项误差平方为 `(1.2-1.19)^2=0.0001`，第一项为 0，单样本 loss 为 `0.0001`；取期望后还要对样本平均。这里 `H=2` 是未来两步预测的训练长度，不是“2 帧历史记忆”。

**Toy B：flow 触发重规划**

假设某 episode 的 flow centroid 距离序列为：

```text
d = [0.20, 0.30, 0.25, 2.10]
```

到 `t=4` 时，之前的 jump 为：

```text
|0.30-0.20| = 0.10
|0.25-0.30| = 0.05
running_mean = (0.10+0.05)/2 = 0.075
3×running_mean = 0.225
latest_jump = |2.10-0.25| = 1.85
```

因为 `1.85 > 0.225`，触发重规划。这个例子验算了 Appendix K 的规则：用 episode 自身历史归一化，不需要绝对像素阈值。但它也暴露条件：如果扰动造成的 2D flow 变化不明显，可能漏触发。

## 4. 工程视角

| 工程项 | 论文给出的条件 | 工程含义 |
|--------|----------------|----------|
| 仿真 per-step 推理 | K-UBM Flow `0.42±0.05 ms`，DynaMo `0.47±0.14 ms`；Diffusion `29.95±1.4` / `36.31±2.3 ms`；UVA `98.6±26.9 ms`（raw images） | 线性 rollout 的动作估计很便宜；但 Table 1 明确排除 visual feature extraction，不能把该数字当作端到端闭环延迟 |
| 真实 per-step 推理 | K-UBM 作者报告 `0.017 ms/step` | 这是 per-step policy query time；未含视觉前端、监控、重规划。不能据此推断真实控制频率 |
| 平滑性 | 真实任务 arm jerk 平均 `4.98±2.01`，ACT `28.42±8.14`，Diffusion `44.69±23.11`；hand jerk `4.14±1.71` vs ACT `16.56±12.56`、Diffusion `14.16±4.84` | 动力系统式 rollout 天然更连续；但 jerk 单位是 `rad/s^3`，Table 10 的 jerk 为 10 次运行 mean±std，不是成功率运行数 |
| 训练成本 | K-UBM 不使用大策略预训练；视觉前端在任务内训练；每任务训练 | 数据效率是作者主张，但正文主要给成功率/推理成本，未给统一的数据 Scaling 曲线 |
| 记忆机制 | state-inclusive 当前状态 + `K` 外推；无显式 recurrent memory | 不能把 `H=15/50` 说成记忆窗口；它是多步训练预测长度。长历史窗口消融在摘录中未见 |
| 重规划阈值 | flow 用 `3×` 历史 jump 均值；DynaMo 用相似度变化 | 规则无绝对像素阈值，但仍依赖特征、相机和 episode 历史；工程上需校准 |

**工程判断（我的条件性分析）**：K-UBM 适合“任务族固定、初始条件可对齐、视觉特征可稳定提取、接触不连续不极端”的场景。它的优势是 rollout 便宜、可解释、可监控；弱点是视觉前端和重规划监控的延迟没有进入主表，真实部署的闭环周期需要重新测。

## 5. 数据与评测

### 5.1 预训练与视觉特征（逐模型归属）

| 模型/模块 | 架构 | 训练数据/预训练 | 开放范围 |
|-----------|------|------------------|----------|
| K-UBM + Object Flow | SAM3 语言提示分割 + CoTracker 跟踪 256 点 + 卷积 autoencoder 压到 128 维；MLP 谱编码器 + 线性 `K` | 每个任务用演示训练；autoencoder 用 flow 重建损失；策略与 `K` 联合训练 | 论文未声明代码/数据开放范围 |
| K-UBM + DynaMo | DynaMo ResNet-18 编码器 + 自监督 latent dynamics；MLP 谱编码器 + 线性 `K` | DynaMo 在目标任务演示图像上 in-domain 自监督预训练；K-UBM 每任务训练 | 论文未声明代码/数据开放范围 |
| Diffusion 基线 | CNN 1D temporal U-Net 条件去噪；FiLM 注入 Flow/DynaMo 特征 | 每任务训练；论文未提大规模预训练 | 论文未声明 |
| ACT 基线 | CVAE + Transformer encoder/decoder | 每任务训练；chunk 10/20/40 逐任务选 | 论文未声明 |
| UVA 基线 | MAR ViT + 冻结 KL-16 VAE；视频/动作双 diffusion head | 从 `mar_base` 初始化，约 262M trainable parameters；300 epochs | 论文未声明 |
| KODex 基线 | 预定义 polynomial/sinusoidal lifting + 最小二乘 | 每任务；解析解，不走五 seed | 论文未声明 |
| KOROL 基线 | 冻结 Flow/DynaMo + 可学习 MLP 投影 + Koopman 最小二乘 | 每任务；输出特征 8 维；prediction horizon 15 | 论文未声明 |

### 5.2 仿真评测

- **任务**：7 个仿真任务。DexArt 3 个：Bucket、Laptop、Toilet，SAPIEN，XArm6 6-DoF + Allegro 16-DoF。Adroit 4 个：Door opening、Tool use、In-hand reorientation、Object relocation，MuJoCo，Adroit 30-DoF 系统（24-DoF hand + 6-DoF floating wrist）。
- **演示**：DexArt 30 条训练、30 条测试；Adroit 50 条训练、30 条测试；用预训练 RL experts 生成成功轨迹。
- **基线**：Diffusion、ACT、UVA、KODex、KOROL。Diffusion 调 `(T_a,T_p)∈{(4,8),(8,16)}`，`T_o=2`；ACT 调 chunk 10/20/40；UVA 条件 4 帧、动作 chunk 8。
- **次数口径**：每条策略 5 个随机种子，KODex 解析解除外；架构消融 7 任务 × 3 seeds × flow 特征。成功率在 unseen test set 上报告；Table 1 推理成本在 Bucket 单任务上测，per-step policy query time 按总时间步平均，排除 visual feature extraction。
- **结果**：作者报告 K-UBM 平均成功率为第二高；UVA 略高但推理显著更慢；KODex/KOROL 表现较差且方差大，作者归因于最小二乘对高维视觉 latent 噪声敏感。消融中 K-UBM 在 state-inclusive lifting + linear latent dynamics 下平均成功率最高（Fig. 4，7 任务、3 seeds、flow）。

### 5.3 真实机器人设置

- **平台**：7-DoF Franka Research 3 + 16-DoF LEAP-V1 左手，23 维状态空间。
- **任务**：Flower Arrangement、Chips Pouring、Whiteboard Erasing、Dinner Serving。
- **数据**：每任务 20 条 kinesthetic teaching 演示；固定第三人称 webcam；后处理对齐初始关节配置。
- **基线**：Diffusion、ACT；UVA 因训练时间过长省略。
- **结果**：K-UBM 平均成功率 `85.0%`，ACT `92.5%`，Diffusion `72.5%`；K-UBM 不是最高成功率，但作者报告最快推理和最平滑轨迹。Table 10 中 jerk 为 10 次运行 mean±std；成功率是任务级表值，不应与 jerk 的运行次数混淆。

## 6. 能力与失败模式

**能做什么（论文实测/作者主张）**：

- 在 7 个仿真任务上与强基线可比，平均成功率为第二高；真实 4 任务平均 85.0%，低于 ACT 但高于 Diffusion。
- 轨迹更平滑：真实 arm jerk 约比 Diffusion/ACT 低 6–9 倍，hand jerk 低 3–4 倍（作者表述；Table 10 数值可复核）。
- 推理便宜：仿真 per-step 0.42/0.47 ms（不含视觉特征提取）；真实 0.017 ms/step（policy query）。
- 支持 flexible horizon：rollout 长度可调，不需要固定 action chunk。
- 支持事件触发重规划：反应性实验 7 任务 × 30 rollouts = 210 次；K-UBM trigger ratio `0.88±0.16`，task success `0.58±0.20`；Diffusion `0.41±0.28`，ACT `0.41±0.26`。
- 视觉流预测可作为监控/验证信号：Appendix H 报告 flow RMSE 与 DynaMo cosine similarity 趋势，失败 rollout 的预测误差更大/相似度更低。

**不能做什么/失败模式**：

- **接触不连续**：线性 `K` 会平滑尖锐接触；作者明确说高频冲击任务可能需要 switching Koopman/hybrid modeling。
- **开环伪规划的脆弱性**：高度随机或物体动力学变化大的环境，单次开环 rollout 不够，可能需要在线适应或 residual correction。
- **遮挡 + 扰动同时发生**：重规划依赖视觉反馈；同时遮挡和扰动会延迟触发，作者建议引入 tactile feedback。
- **触发失败**：2D flow centroid 启发式对某些 3D 偏差不敏感，任务不变阈值可能漏触发。
- **重规划失败**：触发后可能需要数据中未出现的恢复行为（如物体翻倒后的恢复、避碰）。
- **初始条件依赖**：Appendix G 用 `a_0=q_1`、`I_0=I_1`，并假设演示初始关节配置一致；真实数据也后处理对齐初始关节配置。我的条件性分析：这降低了初始条件多样性，任意初始位姿/跨任务泛化未验证。
- **任务特定**：每任务训练、每任务视觉前端；不是通用 VLA，也不是可任意 action-conditioned 的世界模型。

### 6.1 隐含假设

1. **初始条件可对齐**：K-UBM 以 `ξ_0` 为起点；若初始状态分布很宽，开环计划的前提变弱。
2. **视觉特征能稳定表达任务相关物体运动**：Object Flow 依赖 SAM3 分割与 CoTracker 跟踪；跟踪失败或遮挡会污染监控信号。
3. **线性 latent 动力学足以覆盖技能**：适合粗粒度行为，不适合强接触不连续/高频冲击。
4. **扰动会造成可检测的视觉流偏差**：如果扰动在 2D flow 上不明显，触发规则会漏检。
5. **基线调参公平但任务特定**：论文对 Diffusion/ACT 做了逐任务 horizon/chunk 搜索；这使比较更公平，但也说明“固定 chunk 长度”不是所有实验共有的记忆窗口。

## 7. 与相关工作对比

| 方法 | 架构 | 训练数据/预训练 | 历史/未来时序 | 记忆机制 | 本文角色/适用场景 |
|------|------|------------------|----------------|----------|-------------------|
| K-UBM | MLP 谱编码器 + 线性 Koopman `K`；state-inclusive latent | 每任务演示；视觉前端 Object Flow/DynaMo 任务内训练；无大策略预训练 | 以初始条件 rollout 到任意 `t_p`；训练多步 `H=15/50` | 当前 `ξ_t` + `K` 外推；无显式历史缓存 | 任务特定灵巧行为伪规划器；强调时序一致、快速推理、可监控 |
| Diffusion | CNN 1D temporal U-Net 条件去噪 | 每任务；无大规模预训练声明 | `T_o=2` 历史观测；`(T_a,T_p)=(4,8)` 或 `(8,16)` 逐任务调 | 反应式 receding horizon；无长期记忆 | 强表达基线；推理慢，chunk/horizon 需调 |
| ACT | CVAE + Transformer | 每任务 | chunk 10/20/40 逐任务调 | 反应式 action chunk；无显式长期记忆 | 强基线；真实任务平均成功率最高 |
| UVA | MAR ViT + 冻结 VAE + 双 diffusion head | `mar_base` 预训练，约 262M trainable；300 epochs | 条件 4 帧；动作 chunk 8 | 条件帧 + 动作 chunk；无长期记忆描述 | 视频-动作联合基线；仿真略高但推理慢，真实因训练久省略 |
| KODex | 预定义 lifting + 最小二乘 | 每任务；解析解 | 一步预测目标；不支持多步损失 | 无显式历史窗口 | Koopman 基线；高维视觉 latent 下方差大 |
| KOROL | 冻结 Flow/DynaMo + MLP 投影 + 最小二乘 | 每任务；输出 8 维特征 | prediction horizon 15 | 无显式长期记忆 | Koopman+视觉基线；表现弱于 K-UBM |

**面试 Tip**：被问到时先答——“K-UBM 不是世界模型，也不是通用 VLA；它是任务特定的行为动力学伪规划器，用线性 Koopman rollout 换时序一致性和可监控性。代价是接触不连续、初始条件对齐和任务特定训练。不要把 `H=15/50` 说成记忆窗口，它们是多步训练预测长度；也不要把 0.017 ms 说成含视觉的闭环延迟。”

## 8. 精读建议

- **值得精读原文的人**：
  - 做多指灵巧操作、动作分块时序一致性、平滑控制的研究者；
  - 想评估 Koopman/动力系统策略能否落地的工程师；
  - 对“预测视觉流作为运行时监控”的在线重规划机制感兴趣的人。
- **建议章节路径**：先读 §2 的 UBM 定义与 §3 的 K-UBM；再看 §4.1–4.2 的评测条件与 Table 1；然后读 Appendix F（超参）、Appendix G（初始条件）、Appendix I/K（反应性与触发）、Appendix L（视觉损坏鲁棒性）。Appendix D 可在需要复现基线调参时再看。
- **不值得精读的理由**：如果你不做机器人学习，或只关心大规模视频生成/VLA 的跨任务泛化，读摘要即可；如果你已熟悉 Koopman 且只想要 SOTA 数字，本文的核心新意主要在“统一行为流 + 伪规划 + 事件监控”的组合，而不是单点算法突破。

---

[← Back to Theory](./README.md)

**关键引用**：[论文 arXiv:2602.07413](https://arxiv.org/abs/2602.07413)