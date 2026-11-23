# 潜空间世界建模的紧凑机器人策略：La-JEPA 精读 (La-JEPA: Compact Robot Policies with Latent World Modeling)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-11-23
>
> **论文**: La-JEPA: Compact Robot Policies with Latent World Modeling
> **链接**: https://arxiv.org/abs/2611.08156
> **核心定位**: 不在像素空间重建未来，而是在预训练 V-JEPA 2.1 特征空间里做“训练时未来预测 + 推理时纯动作生成”，用 0.10–1.14B 可训练参数逼近甚至超过更大的生成式 WAM/VLA。

> **证据版本说明**：本文依据 arXiv:2611.08156v2（2026-11-20）的 HTML 正文。来源记录未提供可核验的官方 PDF；候选摘要为截断片段，不能用于补写正文主张。可核验的正文摘要与片段开头一致，但片段不完整；未发现可核验的“PDF 与元数据摘要不一致”。以下数字均取自正文表格与附录，并尽量标注条件。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 冻结 V-JEPA 2.1 特征空间 + 从零训练的 World/Action DiT，可在不生成未来画面的情况下获得强控制性能；在无 embodied pretraining 的策略中，La-JEPA 在 RoboTwin 2.0 与修正指令后的 LIBERO-Plus 上领先，且参数更少、延迟更低。 |
| 适合精读 | 做 VLA/WAM 架构、潜空间世界模型、低成本策略部署的人，重点看 §3 架构、§4 设计分析、§5.2 指令泄漏。 |
| 可以跳过 | 只关心语言模型预训练或纯视觉生成的人，这篇距离中等；它不训练视频生成器。 |
| 落地可行性 | 中高：推理只跑 Action DiT，S/B 策略调用 17.78/25.16 ms；但依赖 V-JEPA 2.1 与 FLAN-T5-large 冻结编码器，真实机器人只验证了 4 个桌面任务。 |
| 主要风险 | 无 embodied pretraining，真实机器人仅一个 Franka 平台、4 个任务；LIBERO-Plus 原始指令被污染，排行榜结论对指令协议敏感。 |

💡 **X-Ray 开场**

这篇论文问的是：机器人策略一定要继承视频生成大模型或 VLM 的架构吗？作者的做法是把“世界模型”从像素生成器改成训练时的特征预测正则：策略在冻结的 V-JEPA 2.1 特征空间里从零训练，训练时联合预测未来特征与动作，推理时把未来 token 全部丢掉，只跑动作去噪。对 VLA 研究者的意义是：**潜空间选择与策略原生训练，可能比继承一个更大的生成式骨干更重要**；但“无 embodied pretraining”也意味着它尚未证明能替代大规模机器人预训练。

📍 **研究全景时间线**

```
2024  V-JEPA：视频自监督的特征预测
  → 2025  V-JEPA 2：扩大视频自监督规模
  → 2026  V-JEPA 2.1：改善稠密特征，适合下游机器人
  → 2026  Fast-WAM / Faster-WAM：训练时预测未来、推理时只生成动作
  → 2026  La-JEPA（本文）：在冻结 JEPA 特征空间从零训练紧凑 World/Action DiT
  ← 当前位置：强仿真结果 + 少量真机验证；尚未做大规模 embodied pretraining
```

## 1. 核心架构/方法总览

### 1.1 系统对比概览

| 模块/模型 | 初始化与可训练性 | 输入 | 输出/推理时是否预测未来 | 时序与频率 | 训练/推理差异 |
|---|---|---|---|---|---|
| V-JEPA 2.1 ViT-L 编码器 | 冻结；La-JEPA 可加 LoRA | 每路相机最近观测帧；训练时还有未来帧 | 输出 z_ctx 与 z_fut | 每路相机独立编码；观测为两帧连续片段 | 训练时未来目标由冻结编码器预先计算；推理不跑未来分支 |
| FLAN-T5-large | 冻结 | 语言指令 | 条件 token | 每次策略调用重新编码 | 训练与推理一致 |
| World DiT | 从零训练 | z_ctx、z_fut^τf、条件 | 预测未来特征速度 v_f | 训练时使用；推理丢弃未来 token | 训练时与 Action DiT 联合；推理只保留观测 KV |
| Action DiT | 从零训练 | 动作噪声 A^τa、观测 KV、条件 | 预测动作速度 v_a | 推理时做 Euler 去噪 | 动作 token 不看未来 token |
| La-JEPA 整体 | 编码器冻结/LoRA；DiT 从零 | 多相机、proprio、语言 | 动作 chunk A=(a_t,...,a_{t+H-1}) | 默认预测 32 个动作；LIBERO 执行 10 个、RoboTwin 执行 24 个后重规划 | 训练联合 flow matching；推理只跑 Action DiT |
| Fast-WAM | 视频生成器初始化 | 论文所述视频/动作输入 | 训练时预测未来，推理只生成动作 | 部署仍保留生成骨干 | 与 La-JEPA 的“从零 DiT”不同 |
| Faster-WAM | 视频生成器初始化 | 视频骨干缓存 KV + 单层动作专家 | 训练时预测未来，推理只生成动作 | 部署仍保留视频骨干 | 动作模块更浅，但仍继承生成骨干 |
| JEPA-WAM | VLM 监督初始化 | V-JEPA 特征 + 语言 | 保留预训练预测器 | 论文未在正文展开全部时序细节 | 与 La-JEPA 最接近，但预测器非从零 |
| LeapBot-WA | 视频生成器初始化，且做机器人数据预训练 | V-JEPA 对齐空间 | 保留预训练策略骨干 | 论文未在正文展开全部时序细节 | 部署 10.4B，仍有 embodied pretraining |

> 逐模型归属说明：上表中“冻结编码器 + 从零 DiT”只属于 La-JEPA；“保留视频生成骨干”属于 Fast-WAM/Faster-WAM/LeapBot-WA；“保留 VLM 预训练预测器”属于 JEPA-WAM。不能把 La-JEPA 的“推理不生成未来”直接归给所有 WAM。

### 1.2 关键机制

- **潜空间而非像素空间**：未来目标 z_fut 是 V-JEPA 2.1 的特征，不是 RGB 帧；损失在特征空间算 MSE，不做像素重建。
- **策略原生训练**：World DiT 与 Action DiT 都从零初始化，架构与容量可以按控制需求选，不受视频生成骨干的深度/宽度约束。
- **训练时世界建模，推理时纯动作**：未来 token 只用于训练；推理时丢弃未来 token，把 World DiT 的观测 KV 缓存一次，每个去噪步只跑 Action DiT。
- **注意力掩码隔离未来**：观测 token 只看观测；未来 token 看观测与未来；动作 token 只看观测与动作。动作路径永远不直接读未来特征。
- **鲁棒性配方**：图像增强 + 编码器 LoRA + 相机/proprioception dropout；消融显示图像增强主要贡献 OOD 鲁棒性。

⚡ **Eureka Moment**：把“世界模型”从推理时必须生成未来的大模块，降级为**训练时的特征预测正则**；策略在预训练 JEPA 空间里从零学控制，推理只保留动作分支。

### 1.3 信息流/架构图

```text
训练时：
  多相机最近帧 ──► V-JEPA 2.1（冻结/LoRA）──► z_ctx ─┐
  多相机未来帧 ──► V-JEPA 2.1（冻结）──────► z_fut ─┤
  语言 ──► FLAN-T5-large（冻结）──► z_cond ──────────┤
  proprio ──► 学习投影 ──► z_cond ───────────────────┤
                                                     ▼
                                      World DiT（从零）──► v_f 预测
                                                     │
                                      Action DiT（从零）──► v_a 预测
                                                     │
                                     注意力掩码：动作不看未来 token

推理时：
  最近观测帧 ──► V-JEPA 编码 ──► 观测 KV 缓存
  语言/proprio ──► 条件
  动作噪声 ──► Action DiT 反复去噪（默认 10 步）──► 动作 chunk
  ──► 执行前 k 个动作（LIBERO k=10；RoboTwin k=24）──► 新观测重规划
```

## 2. 数学核心

📌 **Napkin Formula**：

```text
L = MSE(v_f, z_fut - ε_f) + MSE(v_a, A - ε_a)
推理：只对动作速度场做 Euler 积分，未来分支不执行
```

**目标**：联合学习未来特征预测与动作生成；训练时未来分支给共享的 World DiT 观测表示提供额外监督，推理时只生成动作。

**前向加噪**：

```text
z_fut^τf = (1 - τf)·ε_f + τf·z_fut
A^τa    = (1 - τa)·ε_a + τa·A
τf, τa ~ Uniform(0,1)
ε_f, ε_a ~ Gaussian
```

**训练损失**：

```text
L = E[ MSE( v̂_f(z_ctx, z_fut^τf, A^τa, z_cond; τf), z_fut - ε_f )
     + MSE( v̂_a(z_ctx, z_fut^τf, A^τa, z_cond; τa), A - ε_a ) ]
```

**推理**：

```text
丢弃 z_fut 分支；固定 τ_ctx = 0
从 τa=0 到 τa=1 做 Euler 积分：
A^{τ+Δτ} = A^τ + Δτ · v̂_a(A^τ, z_ctx, z_cond; τ_ctx=0)
```

**变量说明**：

| 符号 | 含义 |
|---|---|
| t | 当前决策时刻 |
| H | 动作 chunk 的预测长度；La-JEPA 默认预测 32 个动作 |
| A | 动作 chunk (a_t,...,a_{t+H-1}) |
| z_ctx | 最近观测帧的 V-JEPA 特征 |
| z_fut | 未来帧的 V-JEPA 特征，仅训练使用 |
| z_cond | 语言与 proprio 形成的条件序列 |
| τf, τa | 未来/动作 flow matching 的独立时间步 |
| εf, εa | 高斯噪声 |
| v̂f, v̂a | 网络预测的未来/动作速度 |
| k | 每次重规划实际执行的动作数；LIBERO 10，RoboTwin 24 |

**直觉**：未来分支和动作分支各自做 flow matching，但通过共享 World DiT 参数与注意力耦合。动作 token 不能直接看未来 token，所以未来预测不是“推理时规划器”，而是**训练时表示正则**；它改变动作 token 读取的观测表示。

> 时序口径必须分清：若均匀采样 K 帧、间隔 Δt，首末跨度是 (K-1)·Δt；若从当前参考 t 预测 t+1 至 t+H，终点距 t 是 H·Δt。La-JEPA 的动作 chunk 是 32 个预测动作，执行 k=10/24 后重规划；这不是“历史记忆窗口”。论文正文提到未来目标“32 frames ahead”，附录又写 tube-motion 联合编码“16 future frames”，两者存在数字口径张力，精读时应以具体实验配置核对，不能把任一时长写成所有实验共有的记忆窗口。

## 3. 带数字走一遍：玩具例子

> 以下为**假设数值**，只用于验算公式，不代表论文实验设置。

假设均匀采样 5 个历史帧，帧间隔 Δt=0.2 s：

```text
首末跨度 = (K-1)·Δt = (5-1)×0.2 = 0.8 s
```

若从当前参考 t 预测 t+1 至 t+16，终点距当前参考：

```text
H·Δt = 16×0.2 = 3.2 s
```

这 3.2 s 是“未来预测区间”，不是历史上下文窗口。

再走一遍 flow matching。设某个未来特征标量 z_fut=1.0，噪声 ε_f=-0.2，采样 τf=0.25：

```text
z_fut^τf = (1-0.25)×(-0.2) + 0.25×1.0 = 0.1
目标速度 = z_fut - ε_f = 1.0 - (-0.2) = 1.2
若网络预测 v̂_f = 1.1
未来损失 = (1.1 - 1.2)^2 = 0.01
```

动作分支同理。设 A=0.8，ε_a=0.2，τa=0.5：

```text
A^τa = 0.5×0.2 + 0.5×0.8 = 0.5
目标速度 = 0.8 - 0.2 = 0.6
若 v̂_a = 0.55
动作损失 = (0.55 - 0.6)^2 = 0.0025
总损失 = 0.01 + 0.0025 = 0.0125
```

关键检查：动作 query 的注意力掩码只允许看观测与动作，不能看未来 token。因此未来损失再大，也不会让动作 token 直接“抄未来”；它只能通过共享 World DiT 参数改变观测表示，间接影响动作分支。

## 4. 工程视角

### 4.1 延迟与参数

论文 Table 16 测的是**完整策略调用**，不是控制循环；单张消费级 GPU，batch size 1，20 次 warm-up 后 100 次调用的中位数。默认 10 个去噪步。

| 策略 | 步数 | 优化构建延迟 (ms) | 原始 kernel CUDA graphs (ms) | Eager (ms) |
|---|---:|---:|---:|---:|
| La-JEPA S | 10 | 17.78 | 34.20 | 99.96 |
| La-JEPA B | 10 | 25.16 | 42.35 | 98.21 |
| La-JEPA L | 10 | 59.72 | 88.87 | 167.93 |
| Fast-WAM | 10 | 88.09 | OOM | 193.44 |
| Faster-WAM | 10 | 50.96 | OOM | 62.70 |
| π0.5 | 10 | 71.89 | — | — |
| La-JEPA S | 3 | 11.04 | 19.59 | — |
| La-JEPA B | 3 | 13.40 | 22.54 | — |
| La-JEPA L | 3 | 26.30 | 39.37 | — |

工程含义：

- **推理只跑 Action DiT**：观测 KV 缓存一次，每个去噪步不重复跑 World DiT；这是低延迟的主要来源。
- **3 步去噪是可用操作点**：论文称 3 步与 10 步在 LIBERO/LIBERO-Plus 上相差约 1 个百分点以内；但这是特定 checkpoint 的评测，不应外推为所有任务都无代价。
- **执行 chunk 不能无限缩短**：LIBERO-Plus 上只执行 5 个动作会让 L 降 3.29 分、B 降 3.45 分；更短 chunk 增加重规划频率，延迟与成功率要一起看。
- **延迟比较不是完全同条件**：La-JEPA 用两路 256×256、两帧、FLAN-T5-large，计时 FP32；Fast-WAM/Faster-WAM 用两路 224×224 单帧、BF16 UMT5-XXL。作者称在同一优化程度下比较，但读者仍应把编码器/输入差异视为条件性因素。

### 4.2 参数账本

| 模型 | 可训练参数 (B) | 部署参数 (B) | 说明 |
|---|---:|---:|---|
| La-JEPA S（LIBERO 配置） | 0.10 | 0.74 | World/Action DiT 0.094B + 编码器 LoRA 0.002B；部署含 V-JEPA 2.1 L 0.305B + FLAN-T5-large 0.341B |
| La-JEPA B（LIBERO 配置） | 0.34 | 0.98 | 同上，DiT 0.335B |
| La-JEPA L（LIBERO 配置） | 1.14 | 1.78 | DiT 1.131B + 全深度 LoRA 0.006B |
| La-JEPA S（RoboTwin 配置） | 0.11 | 0.75 | token 投影不同导致略增 |
| La-JEPA B（RoboTwin 配置） | 0.37 | 1.01 | — |
| La-JEPA L（RoboTwin 配置） | 1.17 | 1.82 | — |
| La-JEPA XL（RoboTwin 全量训练） | 2.14 | 2.79 | 全量训练；S/B/L/XL 全量训练值均为单 seed |
| Fast-WAM | 6.0 | 12.4 | 视频生成器初始化 |
| Faster-WAM | 5.1 | 11.4 | 视频生成器初始化 |
| JEPA-WAM | 0.55 | 1.35 | VLM 监督初始化 |
| LeapBot-WA | 2.7 | 10.4 | 视频生成器 + 机器人预训练 |
| π0.5 | 3.35 | 3.35 | embodied pretraining |

> “部署参数”包含冻结的视觉/语言编码器，是规模度量，不是内存占用或每动作算力。对 La-JEPA S/B，部署参数的大头是冻结编码器；但 L 的 World/Action DiT 约 1.131B，大于冻结 V-JEPA 2.1 L 与 FLAN-T5-large 合计约 0.646B，XL 更明显，不能一概而论。

## 5. 数据与评测

### 5.1 预训练、数据与开放范围

- **预训练**：La-JEPA 不使用目标演示之外的机器人数据做 embodied pretraining；视觉编码器用公开 V-JEPA 2.1，语言编码器用公开 FLAN-T5-large。策略 DiT 从零训练。
- **仿真数据**：LIBERO 原始 2,000 条演示，按“回放动作必须成功”过滤后保留 1,777 条；RoboTwin 2.0 的 2,500 条 clean 演示中，排除 62 条关节角跨 ±π 的轨迹，保留 2,438 条（538,662 帧，每任务至少 41 条）。
- **真实机器人数据**：作者用 SpaceMouse 遥操作采集 4 个 Franka 桌面任务，每任务 100 条演示，共 400 条；10 Hz 记录相机、机器人状态、末端位姿与夹爪状态。论文声明不发布新数据集。
- **代码开放**：论文声明提供特征预计算、训练与评测代码及 S/B/L 配置；本文不额外给出代码仓库链接。

### 5.2 仿真评测设置

| 基准 | 评测内容 | 规模/协议 | 备注 |
|---|---|---|---|
| LIBERO | 4 个标准 suite | 每 suite 上游步数限制（220/280/300/520），10 步 settling | La-JEPA 三个 size 各 3 个训练 seed |
| LIBERO-Plus | 7 类扰动、10,030 个变体 | 修正指令或原始指令两套协议 | 原始评测器给 8,493/10,030 个变体追加扰动参数 |
| RoboTwin 2.0 | 50 个任务，clean 与 randomized | 主结果每条件 5,000 episodes；消融常用 1,000 episodes | clean-only 训练与 full 训练分开报告 |
| JEPA-WAM 对比 | JEPA-WAM 的 20 个任务 | 每条件、每 training seed 2,000 episodes | 同一 20 任务训练；全 50 任务训练会降低这 20 任务上的随机化成功率 |

**关键结果（论文实测）**：

| 设置 | La-JEPA S | B | L | XL | 对照 |
|---|---:|---:|---:|---:|---|
| LIBERO | 97.9 | 98.0 | 98.1 | — | 与最佳基线差距 ≤0.6 分 |
| LIBERO-Plus 修正指令 | 86.3 | 87.3 | 88.0 | — | NowWAM 86.2；L 领先 1.8 分 |
| LIBERO-Plus 原始指令 | 80.0 | 82.5 | 84.1 | — | NowWAM 86.2 领先；说明指令污染会改变排序 |
| RoboTwin clean-only：clean / randomized / mean | 85.5 / 33.9 / 59.7 | 86.2 / 35.4 / 60.8 | 86.0 / 32.1 / 59.0 | — | clean-only 下 size 不单调 |
| RoboTwin full：clean / randomized / mean | 89.2 / 89.4 / 89.3 | 92.6 / 91.5 / 92.0 | 93.3 / 92.7 / 93.0 | 94.2 / 93.2 / 93.7 | full 数据下 S→XL 单调；S/B/L/XL 全量训练值均为单 seed、每条件 5,000 episodes |
| JEPA-WAM 20 任务：clean / randomized | — | 95.2 / 56.5 | — | — | JEPA-WAM 79.9 / 36.9 |

**设计分析（论文消融）**：

- 去掉未来预测：RoboTwin clean -4.8、randomized -2.9；LIBERO -2.2；LIBERO-Plus -4.5。即使推理不预测未来，训练时未来监督仍显著影响策略。
- 把 V-JEPA 特征换成 Wan2.2 VAE latent：RoboTwin clean -13.7、randomized -28.4；LIBERO-Plus -10.7；LIBERO -1.4。作者认为“潜空间选择”是最大单一因素。
- 去掉图像增强：LIBERO-Plus -13.0、RoboTwin randomized -15.1；in-distribution 只 -2.4。说明增强主要买 OOD 鲁棒性。
- 去掉编码器 LoRA：LIBERO-Plus -4.7、RoboTwin clean -4.0。

> 条件性分析：这些消融以 S 配方、seed 42、150k updates 为参照，部分比较的评测实现或数据划分不同；表 6 的绝对值不能简单横向加到主表上。作者也说明 recipe 未做系统性超参搜索。

### 5.3 真实机器人设置

- 平台：单台 Franka Panda 桌面机械臂，平行夹爪；腕部 RealSense D405 + 外部 RealSense D435；通过 Polymetis/Franka Control Interface 执行末端增量与夹爪命令。
- 任务：cube placement、fruit picking、Push T、dish manipulation（长时序多步，含餐具位置互换）。
- 训练：La-JEPA B 从零训练，带编码器 LoRA、不加图像增强；与 Fast-WAM、Faster-WAM、π0.5、LingBot-VA 比较。
- 评测：每任务 20 次 rollout；LingBot-VA 的 dish 为 10 次，故其 average 省略。

| 策略 | 延迟 (ms) | Fruit | Cube | Push T | Dish Position | Average |
|---|---:|---:|---:|---:|---:|---:|
| π0.5（embodied PT） | 71.89 | 20 | 20 | 11 | 18 | 17.25 |
| La-JEPA（无 embodied PT） | 25.16 | 19 | 20 | 15 | 16 | 17.50 |
| Fast-WAM | 88.09 | 19 | 19 | 14 | 12 | 16.00 |
| Faster-WAM | 50.96 | 18 | 20 | 13 | 14 | 16.25 |

论文实测：La-JEPA 在无 embodied pretraining 的策略中平均成功率最高，与 π0.5 可比；Push T 最强，但长时序 dish 上 π0.5 更好。作者也指出这只覆盖 4 个任务、一个机器人平台，不能证明更广泛的操作 parity。

## 6. 能力与失败模式

**能做什么**：

- 在不生成未来帧的前提下，利用未来特征预测改善控制；推理只跑 Action DiT。
- 以较小可训练参数获得强仿真结果：S 0.10B、B 0.34B、L 1.14B 可训练参数；部署参数 0.74–1.78B（LIBERO 配置）。
- 在 LIBERO-Plus 修正指令后，S/B/L 均超过或匹配复评基线；L 88.0 对 NowWAM 86.2。
- 全量 RoboTwin 训练下，S→XL 单调提升；XL 93.7 mean，接近 ImageWAM 93.4 但部署参数约少 3 倍。
- 真实机器人 4 任务平均 17.50/20，与 π0.5 的 17.25/20 可比。

**不能做什么/失败模式**：

- **没有 embodied pretraining**：clean-only RoboTwin 的 randomized 成功率仍低于最强 embodied-pretrained 方法；真实机器人只验证 4 个任务。
- **依赖 V-JEPA 2.1 特征**：换 VAE latent 大幅下降；若没有合适的预训练 JEPA 编码器，这套配方不能直接复现。
- **未来预测不是推理时规划**：它不提供显式未来 rollout；对需要长时序闭环规划的任务，优势是否保持未知。
- **指令协议敏感**：LIBERO-Plus 原始指令被污染，Fast-WAM 修正后 +19.3 分、LeapBot-WA +8.1 分；排行榜排序会变。La-JEPA 自身 +3.9 至 +6.3 分。
- **真实机器人范围窄**：单 Franka、桌面、4 任务、每任务 20 次 rollout；没有移动/人形/多机器人验证。
- **延迟不是控制循环**：Table 16 是策略调用，不含仿真、传输、执行；真实控制频率需另测。
- **配方未系统调优**：作者明确未系统搜索学习率、weight decay、dropout、LoRA 配置、World/Action DiT 容量比。

### 6.1 隐含假设

- **假设 1**：V-JEPA 2.1 的冻结稠密特征已经包含控制所需的物体、空间与运动信息；LoRA 足以适配目标域。
- **假设 2**：训练时未来特征预测能通过共享 World DiT 改善观测表示，而推理时不需要显式未来；这对长时序任务是否成立，论文未充分验证。
- **假设 3**：仿真 randomized 场景与真实机器人鲁棒性有足够相关性；但真实只测 4 个任务。
- **假设 4**：修正 LIBERO-Plus 指令后，复评基线与原论文数字可比；但并非所有基线都有修正指令分数。
- **假设 5**：动作 chunk 开环执行 k 步后重规划，控制频率足够；论文的延迟比较未覆盖闭环控制抖动。

## 7. 与相关工作对比

| 方法 | Transformer 初始化 | Embodied PT | 可训练/部署参数 (B) | 推理是否生成未来 | 关键结果/定位 |
|---|---|---|---:|---|---|
| La-JEPA S/B/L | 从零 | 否 | 0.10/0.74；0.34/0.98；1.14/1.78 | 否 | LIBERO-Plus 修正 86.3/87.3/88.0；RoboTwin clean-only mean 59.7/60.8/59.0 |
| Fast-WAM | 视频生成器 | 否 | 6.0/12.4 | 否（训练时预测） | LIBERO-Plus 修正 70.8；RoboTwin clean-only mean 39.9 |
| Faster-WAM | 视频生成器 | 否 | 5.1/11.4 | 否（训练时预测） | LIBERO 98.5；RoboTwin clean-only mean 35.4；真实平均 16.25 |
| JEPA-WAM | VLM 监督 | 否 | 0.55/1.35 | 保留预训练预测器 | LIBERO-Plus 修正 77.8；20 任务 RoboTwin 79.9/36.9 |
| LeapBot-WA | 视频生成器 | 是 | 2.7/10.4 | 保留预训练策略骨干 | LIBERO 97.3；LIBERO-Plus 修正 79.7（released 71.7） |
| π0.5 | VLM | 是 | 3.35/3.35 | 否（VLA） | LIBERO-Plus 修正 85.7；真实平均 17.25 |

**面试 Tip**：如果被问“La-JEPA 和世界动作模型有什么区别”，一句话答：**它不是推理时生成未来的 WAM，而是把未来特征预测当作训练正则、推理只跑动作分支的紧凑策略；真正的卖点是潜空间与从零策略架构，而不是更大的生成骨干。**

## 8. 精读建议

- **值得精读原文的人**：
  - 做 VLA/WAM 架构选型，想知道“生成式骨干是否必要”的研究者；
  - 要在低延迟、低参数预算下部署桌面操作策略的工程师；
  - 做机器人评测协议、尤其是 LIBERO-Plus 指令泄漏问题的评测研究者。
- **建议章节路径**：
  1. 先读 §3.1–§3.2 与图 2，搞清 z_ctx / z_fut / 注意力掩码；
  2. 再看 §4.1 的特征空间与未来预测消融，理解“为什么潜空间重要”；
  3. 然后读 §5.1–§5.2 与表 1/2，注意 clean-only 与 full training 的区别；
  4. 最后读 §5.2 指令泄漏与 Appendix B 的评测协议，避免把排行榜数字当成同一条件；
  5. Appendix C 的延迟与参数账本适合工程落地前核对。
- **不值得精读的理由**：如果你不做机器人学习、不关心 JEPA/世界模型，或已经熟悉“训练时未来预测、推理时只生成动作”的 Fast-WAM 路线，读摘要与表 1/2 即可；本文的核心增量是“潜空间 + 从零紧凑策略”，不是全新的生成式建模方法。

---

[← Back to Theory](./README.md)

**关键引用**：论文链接 https://arxiv.org/abs/2611.08156（本文所有数字与主张均取自该论文 v2 正文；未使用其他外部来源）。