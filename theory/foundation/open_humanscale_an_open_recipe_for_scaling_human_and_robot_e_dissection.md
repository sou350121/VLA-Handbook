# 从 100 到 50,000 小时：Open-HumanScale 的自我中心预训练配方 (Open-HumanScale: An Open Recipe for Scaling Human and Robot Experience for Embodied Foundation Models)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-11-05
>
> **论文**: Open-HumanScale: An Open Recipe for Scaling Human and Robot Experience for Embodied Foundation Models
> **链接**: https://arxiv.org/abs/2611.06817
> **核心定位**: 它把“第一视角人类视频能迁移到机器人”从单点实验推进到可复现的 50,000 小时数据配方与闭环缩放研究，并给出纯人类预训练与人类-机器人混合预训练两套世界-动作模型。

**来源说明（必读）**：本文只依据 arXiv:2611.06817v1（2026-11-04）的 HTML 全文，该 HTML 对应论文 PDF 正文。候选元数据摘要与正文摘要一致，未发现“摘要与 PDF 主张冲突”的情况；因此下文数字均取自正文表格、图注与实验设置，而不是候选摘要。论文声明开源数据、训练代码与 checkpoint，但本文不引用论文之外的仓库 URL。

---

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 纯第一视角人类视频从 100 小时扩到 50,000 小时，后训练后的机器人闭环成功率呈经验 log-linear 增长；再加入机器人数据做混合预训练，可进一步提升零样本与 OOD 表现。 |
| 适合精读 | 如果你在做 egocentric 预训练、数据 curation、世界-动作模型、跨 embodiment 动作空间，重点看 §2、§3.2–§3.4、§5.2–§5.3。 |
| 可以跳过 | 如果你只关心高层语言推理、任务规划或 VLA 的语义泛化，这篇的距离中等；它的主战场是动作接口与数据缩放。 |
| 落地可行性 | 中：数据配方和动作对齐思路可复用，但 50k 小时级预训练、100k optimizer steps、8,192 batch 的算力门槛很高。 |
| 主要风险 | 缩放律是有限数据点上的经验拟合；“每个评测设置取最佳 checkpoint”会带来乐观偏差；真实机器人每任务仅 10 trials。 |

💡 **X-Ray 开场**

这篇论文问的是：第一视角人类视频到底能不能像文本一样“堆规模就变强”？它给出的答案是：在统一的双臂末端执行器动作空间下，可以；而且增益主要体现在分布外泛化，而不是演示分布内的拟合。对 VLA 研究者来说，这意味着“人类视频预训练”不再只是辅助表征学习，而是可以进入闭环策略评测的主线；但必须同时盯住动作对齐、数据多样性和评测协议，否则规模红利会被接口不一致吃掉。

📍 **研究全景时间线**

```
2022  R3M：视觉表示预训练用于机器人操作
  → 2024  π0：VLA flow 模型，机器人策略基础模型化
  → 2025  RoboTwin 2.0 / EgoDex：双臂基准与第一视角操作数据规模化
  → 2026  HumanScale / EgoScale / Dyna-2：自我中心预训练的缩放律与大规模验证
  → 2026-11  本文：50k 小时开放配方 + 纯人类 / 人机混合两套世界-动作模型
```

本文的位置：把“人类数据缩放”从验证损失推进到闭环任务成功率，并把数据多样性、动作对齐、样本效率拆开做控制实验。局限：闭环评测仍以 RoboTwin 2.0 为主，真实机器人平台与任务数有限；缩放律的外推边界未验证。

---

## 1. 核心架构/方法总览

### 1.1 系统对比概览

论文实际给出两个模型，必须逐模型归属，不能把两者的属性合并。

| 维度 | HumanWAM-Vanilla | HumanWAM-Goal |
|------|------------------|---------------|
| 骨干 | Wan2.2-TI2V-5B 视频 backbone + 同深度 action expert，Mixture-of-Transformers | Cosmos3-Nano-16B：frozen Qwen3-VL Reasoner + shared video-action Generator |
| 动作空间 | 16-D 双臂 EEF：位置 3 + 四元数 4 + 夹爪 1，左右各 8 | 49-D hybrid：关节 23 + 底盘 3 + EEF 16 + 头部位姿 7；头部位姿只作几何元数据并被 mask |
| 动作表示 | 绝对 camera-frame EEF 位姿（最终配置）；探索阶段也用过 relative | chunk-relative 位移 + 米制夹爪开口；底盘相对当前位姿 |
| 预测视野 | 32-step action chunk；每个样本 9 帧视频，每 4 个 action steps 采一帧 | 48-step action chunk；12 个未来视频帧 |
| 条件输入 | 语言 + 当前观测 + proprio | 语言 + 当前观测 + proprio + 可选历史帧 H_t + goal image g |
| 已观测 vs 预测 | 当前帧为已观测；9 帧中其余为未来视频帧；32 步为未来动作 | o_t 与可选 H_t 为已观测；12 帧与 48 步为未来预测 |
| 记忆机制 | 未加入专门记忆模块；RoboDojo Memory 组仅 6.00% / 7.00 | 可选历史帧 H_t 作为 clean visual tokens，带 temporal RoPE；论文未给历史窗口长度 |
| 预训练数据 | 50,000 小时纯 egocentric 人类视频 | 14,618 小时机器人 + 15,901 或 45,938 小时人类视频；总预算约 30.5k / 60.6k 小时 |
| 后训练 | RoboTwin 2.0：2,500 demos，24,000 optimizer steps，batch 512；RoboDojo：3,500 demos，50,000 steps；真实机器人：50 demos/task，10,000 steps，batch 256 | RoboTwin：50 clean demos/task，共 2,500 条轨迹；真实机器人任务特定后训练 |
| 开放范围 | 论文声明开源 50K 语料、curation/预训练 pipeline、预训练与后训练 checkpoint、推理代码、仿真评测套件 | 论文声明开源 HumanWAM checkpoints；backbone 依赖 Cosmos3-Nano-16B |

> 关键区分：Vanilla 的 32 步 chunk 与 Goal 的 48 步 chunk 都是“未来动作预测区间”，不是历史上下文，也不是记忆窗口。Goal 的 12 个未来视频帧同样不是历史帧；历史帧只有可选的 H_t，且论文没有给出其帧数或时间跨度。

### 1.2 关键机制

**（1）统一双臂 EEF 动作空间。** 人类手腕位姿提供 EEF 位置与朝向，夹爪开合由手部标注推导。这样人类视频与机器人动作共享同一个 16-D 接口，不需要人类和机器人共享关节构型。论文明确把动作空间限制在双臂操作，不加入移动维度，原因是第一视角数据中的身体/头部运动大多与任务无关，且跨源标注不完整。

**（2）跨源对齐三件套。** 相机系对齐、腕轴对齐、夹爪归一化。论文在 1,000 小时四源语料上做了 Aligned vs Raw 控制实验：对齐后 clean 从 72.0% 到 77.5%，scene-ood 从 43.2% 到 52.2%，random 从 18.0% 到 26.4%。这说明“动作接口不一致”不是细节，而是会直接吃掉规模收益的工程问题。

**（3）数据 curation 三关。** 手部可见性、动作与相机稳定性、空闲过滤。手部不可见或只有边缘手指的片段会被剔除；轨迹抖动、相机跳变会触发切分；32 帧滑窗内 EEF 变化低于阈值视为 inactive，且一个 chunk 中 active 窗口少于 70% 会被丢弃。论文没有给出所有阈值的具体数值，这是复现时的一个不确定点。

**（4）世界-动作联合建模。** Vanilla 用 isolated attention 或 IDM attention；IDM 允许动作 token 关注未来视频帧，训练时用带噪的真实未来视频 latent，推理时先生成未来视频再预测动作。Goal 则用 goal image 作为 clean conditioning token，替换两阶段 IDM 推理，改为单阶段 joint rectified-flow 采样，视频与动作 token 共享 Transformer 权重并双向交互。

**（5）缩放律与多样性。** 论文在 100–50,000 小时上拟合 `SR = a + b·log10(H)`，并在 isolated attention 与 IDM attention 两种配置下都观察到 log-linear 趋势。多样性实验进一步表明：在相同小时数下，任务/场景覆盖随规模扩张的语料，其拟合斜率比固定覆盖的语料更陡。

⚡ **Eureka Moment**：THE 关键洞见是——把“人类视频时长”变成可复现的闭环缩放实验，前提是先做一个跨源一致的双臂 EEF 动作接口；没有这个接口，规模越大，异构标注的几何歧义越会被放大。

### 1.3 信息流/架构图

```
Open-HumanScale-50K（5 个来源）
        │
        ├─ 手部可见性 / 稳定性 / 空闲过滤
        ├─ 相机系 + 腕轴 + 夹爪对齐
        ▼
统一 16-D 双臂 EEF 动作 + 语言 + 视频
        │
        ▼
HumanWAM-Vanilla 预训练（50k h，单头视角）
        │
        ├─ EgoPro mid-training（多视角适配）
        ▼
机器人后训练（RoboTwin / RoboDojo / 真实机器人）
        │
        ▼
闭环评测：clean / scene-ood / random

机器人 14.6k h ─┐
人类 45.9k h  ─┼→ HumanWAM-Goal 联合预训练（49-D hybrid + goal image）
                │
                ├─ zero-shot：直接部署，无参数更新
                └─ post-training：50 demos/task 后微调
```

---

## 2. 数学核心

📌 **Napkin Formula**（一行抓住本质）：

```
SR = a + b·log10(H)   且   预测视野 ≠ 记忆窗口：K 帧跨度 = (K-1)·Δt，H 步未来终点 = H·Δt
```

**目标**：把人类视频小时数 H 与机器人闭环成功率 SR 建立起可检验的经验关系，同时把“动作接口”和“预测视野”定义清楚，避免把未来 chunk 误读为历史上下文。

**统一动作向量**：

```
a_t = [p_t^L, q_t^L, g_t^L, p_t^R, q_t^R, g_t^R] ∈ R^16
```

| 符号 | 含义 |
|------|------|
| p_t^h ∈ R^3 | 手 h 的 EEF 位置，单位米 |
| q_t^h ∈ R^4 | 单位四元数，wxyz 顺序 |
| g_t^h ∈ [0,1] | 夹爪开合，0 全闭、1 全开 |
| h ∈ {L,R} | 左手 / 右手 |

**相机系变换**：

```
T_{C_cv E}^h(t) = T_{W C_cv}(t)^{-1} · T_{W E}^h(t)
p_CE^h(t) = S · p_{C_cv E}^h(t)
R_CE^h(t) = S · R_{C_cv E}^h(t)
S = diag(1, -1, -1)
```

直觉：把世界系手腕轨迹搬到对应视频帧的相机系；再把相机轴统一为 +x 右、+y 上、+z 后。S 用于翻转 +y 向下、+z 向前的源坐标系。

**腕轴对齐**：

```
R_aligned^h(t) = R_camera^h(t) · C_{s,h}
```

其中 `C_{s,h} ∈ SO(3)` 是每个数据源、每只手的常量旋转，用正交 Procrustes 或投影腕轴 + 单目手部关键点确定。直觉：不同数据集对“手腕坐标轴”的定义不同，必须先统一，否则同样的物理动作会有不同的数值标签。

**夹爪归一化**：

```
g = clip((d - d_closed) / (d_open - d_closed), 0, 1)
```

d 是拇指-食指指尖距离；`d_closed`、`d_open` 是每个数据源分别给定的参考距离。Goal 进一步改成米制物理开口，用 0.12 m 作为统一参考宽度，以便零样本跨机器人。

**49-D 混合动作**：

```
a_ext = [ j_t (23), b_t (3), a_t (16), p_t^H (7) ] ∈ R^49
```

j_t 是机器人关节状态，未使用自由度零填充并 mask；b_t 是平面底盘平移与偏航；a_t 是 16-D EEF；头部位姿只作为几何元数据，不进入状态条件与动作损失。

**chunk-relative 位移**：

```
Δp(τ|t) = R_H(t)^T · (p(τ) - p(t))
```

直觉：把未来第 τ 步的 EEF 位置表示为“相对当前观测时刻 t 的位移”，并投影到当前头相机坐标系。这样即使相机移动，整个 chunk 仍有一个静止、与 embodiment 无关的参考系。

**Goal 条件建模**：

```
p_θ(O_t^+, A_t | c_t)
c_t = (ℓ, o_t, s_t, H_t, g)
O_t^+ = 12 个未来视频帧
A_t ∈ R^{48×49}
```

ℓ 是语言，o_t 是当前观测，s_t 是 proprio，H_t 是可选历史帧，g 是 goal image。注意：O_t^+ 是未来预测帧，H_t 才是已观测历史；两者不能混为一谈。

**经验缩放律**：

```
SR = a + b · log10(H)
```

H 是预训练小时数；b 的物理意义是“每十倍小时数带来的成功率增益（pp/decade）”。论文在 isolated attention 下报告 R² = 0.971 / 0.949 / 0.877（clean / scene-ood / random），对应斜率 4.43 / 15.88 / 10.24 pp/decade。

**真实机器人进度分**：

```
P_{k,r} = Σ_{s=1}^{M_k} w_{k,s} · b_{k,r,s}
P_{k,r} ∈ [0,10]
Σ_s w_{k,s} = 10
```

b 是第 s 个阶段是否完成，w 是预设权重。直觉：二值成功率会把“接近完成”判为零，进度分保留部分完成信号；但论文明确说二值成功与进度分是分开评的，成功不一定意味着满分。

> 符号与本文保持一致：`H` 指预训练小时数，不是历史帧数；`h` 作上标时指左手/右手，作变量时指小时，按上下文区分。

---

## 3. 带数字走一遍：玩具例子（假设与验算）

以下例子中，标注“假设”的部分是我为了让计算闭环而设的；标注“验算”的部分是用论文给定数字复算。

**例 1：9 帧视频的跨度验算。**

论文给出 Vanilla 的 action chunk 为 32，视频帧每 4 个 action steps 采一帧，每个样本 9 帧。假设 9 帧均匀间隔、第一帧对应当前观测 t，则首末跨度为：

```
(K - 1) × Δt = (9 - 1) × 4 = 32 个 action steps
```

验算结果正好等于 32-step action chunk。这说明 9 帧覆盖的是“当前帧 + 未来 32 步”的预测视野，而不是 32 步历史上下文。若误把 9 帧当历史窗口，就会把未来信息泄漏进“记忆”概念里。

**例 2：48 步 chunk 的时间换算。**

论文在真实机器人上给出 48-step action chunk 对应 1.6 s（Astribot S1）。假设控制频率恒定，则：

```
单步时间 = 1.6 s / 48 ≈ 0.0333 s
控制频率 ≈ 30 Hz
终点距当前参考 = 48 × 0.0333 s = 1.6 s
```

这是“未来动作终点”的跨度，不是历史窗口。Goal 的 12 个未来视频帧没有给出采样间隔，因此不能据此计算视频预测的时间跨度；只能确认它们是未来帧。

**例 3：缩放律斜率复算。**

以 isolated attention 的 clean 为例，H 从 100 到 10,000 是 100 倍，log10 差为 2；论文拟合斜率为 4.43 pp/decade，则预测增益：

```
4.43 × 2 = 8.86 pp
```

论文实测从 71.2% 到 80.0%，增益 8.8 pp，与拟合值吻合。scene-ood 的 15.88 × 2 = 31.76 pp，实测 25.2% → 58.1% 为 32.9 pp；random 的 10.24 × 2 = 20.48 pp，实测 8.4% → 28.7% 为 20.3 pp。差异来自最小二乘拟合不是逐点过点，属于正常残差。

**例 4：真实机器人 40 次试验反推。**

Vanilla 真实机器人实验为 4 个任务 × 每个任务 10 trials × 每个 setting，因此每个模型每个 setting 共 40 trials。No-Ego 的 OOD 平均成功率为 2.5%，反推：

```
2.5% × 40 = 1 次成功
```

HumanWAM-Vanilla 的 OOD 平均为 67.5%，反推：

```
67.5% × 40 = 27 次成功
```

这与论文“No-Ego 在 40 次 OOD 试验中只成功 1 次”的表述一致。注意：这是按总数反推，不代表每个任务都是整数成功；任务级数字见论文 Table 9。

---

## 4. 工程视角

**训练成本与 batch 调整。** 论文明确说固定训练预算会让小语料过训练、大语料欠训练，因此按语料大小调 global batch：100 h 用 512，50k h 用 8,192；学习率 1e-4，cosine 100k steps，前 5% 线性 warm-up。每 10k pretraining steps 存 checkpoint，并对每个 checkpoint 做后训练与评测，最后按评测设置分别取最佳。这意味着“50k 小时模型”的评测结果不是单一 checkpoint 的全程表现，而是 checkpoint 选择后的结果；工程复现时必须把这一选择协议算进成本。

**推理延迟。** Vanilla 在 isolated attention 下，动作 token 只 attend 当前视频帧，因此推理可以跳过未来视频生成，只 denoise action chunk；IDM 配置则要先生成未来视频 latent，再预测动作，论文自己说这是两阶段串行采样。Goal 用 goal image 作为 clean token，改成单阶段 joint rectified-flow，作者声称采样延迟减半，并可跳过 VAE decoder 做实时控制。这些是作者主张或架构推断，论文没有给统一的端到端毫秒数；真实机器人部署提到 PhyAI 带来 2.2× 推理加速（相对 PyTorch + CUDA Graphs 基线），但未给绝对延迟。

**动作 chunk 与 replan。** RoboTwin 上 Vanilla 预测 32 步，执行前 24 步再 replan；真实机器人 Goal 预测 48 步（1.6 s）。这两个数字是不同模型的设置，不能写成所有实验共有的“记忆窗口”或“控制周期”。Vanilla 的 32 步没有给出对应秒数，因此不能假设 30 Hz 去换算。

**数据管线的工程负担。** 50K 语料从 104,289.41 小时降到 50,000.00 小时，保留 31,396,799 个 physical episodes、35,934,235 个连续 clips、5,400M 帧。curation 不是简单截断，而是按手部可见性、轨迹稳定性、空闲窗口切分；跨源还要做相机系、腕轴、夹爪三套对齐。真正可复用的资产是这套“清洗 + 对齐”的管线，而不是某个模型结构。

**部署接口。** Goal 的零样本部署依赖 goal image；真实机器人用 held-out 机器人演示或人类演示的目标图。仿真评测的 goal image 由 GPT Image 2.5 合成，并用 ChatGPT Astra High 审核；7,032 张新生成图中首轮 90.7% 通过、7.7% 失败、1.6% 不确定，修复后全部通过。这个流程说明 goal image 不是“免费接口”，它本身有生成与审核成本。

---

## 5. 数据与评测

### 5.1 预训练数据

| 来源 | curation 前小时 | curation 后小时 | curation 后 episodes |
|------|-----------------|-----------------|----------------------|
| HumanNet 子集 | 90,275.25 | 40,743.58 | 30,833,247 |
| EgoSuite-Open100K | 11,193.99 | 7,081.65 | 348,724 |
| EgoDex | 829.45 | 335.43 | 163,537 |
| Gen-HumanEgo | 1,847.39 | 1,814.53 | 44,632 |
| EgoVerse | 143.33 | 24.81 | 6,659 |
| 合计 | 104,289.41 | 50,000.00 | 31,396,799 |

标注类型包括手部运动、语言描述、相机位姿；不同源提供的关键点、MANO 参数、深度图等不完全一致。论文把动作统一到 16-D 双臂 EEF，并在 Goal 中扩展到 49-D hybrid。

### 5.2 仿真评测设置

| 实验 | 后训练数据 | optimizer steps | batch | 评测 episodes |
|------|------------|-----------------|-------|----------------|
| 纯人类缩放探索 | RoboTwin 50 clean demos/task | 12k | 512 | 50 tasks × 50 episodes/setting = 2,500/setting |
| HumanWAM-Vanilla RoboTwin | 2,500 demos | 24k | 512 | 100 episodes/task × 50 tasks = 5,000/setting |
| HumanWAM-Vanilla RoboDojo | 3,500 demos | 50k | 512 | 6 组，按论文表 8 |
| HumanWAM-Goal zero-shot | 无 | 无 | 无 | 50 episodes/task × 50 tasks = 2,500/setting |
| HumanWAM-Goal post-training scaling | 50 clean demos/task | 未逐项给 | 未逐项给 | 100 episodes/task × 50 tasks = 5,000/setting |
| HumanWAM-Goal leaderboard | 最佳配置 + augmentation | 未逐项给 | 未逐项给 | 100 episodes/task × 50 tasks × 2 settings = 10,000 |

评测条件分三档：clean 是演示分布；scene-ood 只随机背景与桌面纹理；random 是完整 RoboTwin 随机化，包括背景、杂物、光照、桌面高度抖动（最多 3 cm）。论文明确说，clean 最好的 checkpoint 不一定 OOD 最好，所以每个设置分别取最佳 checkpoint；这会让数字偏乐观，读者需要知道。

### 5.3 真实机器人设置

| 实验 | 平台 | 任务数 | trials | 后训练 |
|------|------|--------|--------|--------|
| HumanWAM-Vanilla | Cobot Magic 双臂 | 4 | 10 trials/task/setting；共 40 trials/model/setting | 50 demos/task，10k steps，batch 256 |
| HumanWAM-Goal zero-shot | Astribot S1 轮式移动双臂 | 20（10 easy + 10 hard） | 10 trials/task；共 200 rollouts/checkpoint | 无 |
| HumanWAM-Goal 任务后训练 | Astribot S1 | 长程桌面整理、移动家务等 | 定性 rollout + 任务特定评测 | 重置 optimizer、LR、sampler；保留 co-training 归一化统计 |

真实机器人 Vanilla 的 OOD 设置是更换桌布、灯光及其组合；每个任务 10 次 in-distribution + 10 次 OOD，OOD 中 4 次换桌布、4 次换灯光、2 次组合。这个设计能分离视觉偏移，但 trials 数少，置信区间宽。

### 5.4 关键结果（只列论文实测）

**纯人类缩放（isolated attention）**：100 h → 10k h，clean 71.2% → 80.0%，scene-ood 25.2% → 58.1%，random 8.4% → 28.7%；无人类预训练基线为 69.5% / 16.2% / 2.1%。

**纯人类缩放（IDM attention）**：100 h → 50k h，clean 87.4%–89.0% 基本饱和，scene-ood 49.1% → 73.2%，random 21.5% → 42.9%；无人类预训练为 87.9% / 29.2% / 7.7%。

**多样性**：EgoSuite-Open100K 上，scaled-diversity 相对 matched-diversity 的拟合斜率增加 2.02 / 4.73 / 4.50（clean / scene-ood / random）。

**样本效率**：10 demos/task 时，clean 从 No-Ego 34.5% → 500 h 44.2% → 5k h 50.1%；random 从 4.1% → 8.3%。scene-ood 上，No-Ego 50 demos 的成绩被 500 h + 25 demos、5k h + 5 demos 超过。

**人类 vs 机器人预训练**：500–5,000 h 区间人类预训练全面优于机器人预训练；5k h 时 scene-ood 52.3% vs 45.6%，random 27.6% vs 22.4%；1k h 人类模型在所有三个设置上超过 5k h 机器人模型。

**动作对齐**：1,000 h 四源语料，Aligned vs Raw：clean 72.0 → 77.5，scene-ood 43.2 → 52.2，random 18.0 → 26.4。

**人类动作监督**：1,000 h 语料，20k steps 时，视频-only vs 50% 人类动作监督：clean 70.4 → 75.2，scene-ood 13.0 → 34.8，random 3.6 → 16.3。

**RoboTwin 2.0 最终对比（论文 Table 14）**：

| 方法 | Clean2Clean | Clean2Random | Average |
|------|-------------|--------------|---------|
| Qwen-RobotManip-Context | 84.7 | 69.4 | 77.1 |
| GO-3 Nano Preview | 88.54 | 81.48 | 85.01 |
| PICA-WAM | 90.68 | 74.70 | 82.69 |
| HumanWAM-Vanilla | 92.08 | 68.88 | 80.48 |
| HumanWAM-Goal | 92.70 | 84.60 | 88.65 |

注意：Goal 的 92.70 / 84.60 使用了最佳混合 checkpoint + photometric augmentation + 后训练配方改进，与 Figure 19 的受控缩放曲线分开；不能把它当成“纯缩放”结果。

**RoboDojo（HumanWAM-Vanilla）**：mean success 14.17%，mean progress 19.00；对比 Galaxea G0.5 14.60 / 20.10，DM0.5 17.96 / 23.37。Memory 组仅 6.00 / 7.00，论文明确说没有加专门记忆模块。

**真实机器人 Vanilla**：in-distribution mean 57.5% → 87.5%；OOD mean 2.5% → 67.5%。任务级：paper cup stacking OOD 0% → 80%，towel folding 10% → 60%，drawer storage 0% → 70%，desk tidying 0% → 60%。

**真实机器人 Goal zero-shot**：20 任务平均 20.0% → 26.0% → 30.5%（robot-only → +15.9k h human → +45.9k h human）；easy 32% → 34% → 42%，hard 8% → 18% → 19%；progress 2.535 → 3.365 → 4.205；coverage 10/20 → 14/20 → 16/20。60.6k h 配置下，robot-balanced 采样把 hard 从 19% 提到 27%（+8 pp），但论文没有给该配置的 20 任务总平均，不能外推。

---

## 6. 能力与失败模式

**能做什么：**

- 在 RoboTwin 2.0 上，纯人类预训练 + 50 demos/task 后训练即可达到 92.08% clean；说明人类视频可以支撑双臂桌面操作。
- OOD 增益显著大于 in-distribution 增益；缩放主要改善 scene-ood / random，而不是 clean 拟合。
- 数据多样性是独立缩放维度；相同小时数下，扩大任务/场景覆盖比重复固定覆盖更有效。
- 人类预训练能提高后训练样本效率；在 500–5,000 h 区间，人类数据比同小时机器人数据更有利于闭环成功率。
- Goal 模型支持零样本执行、视觉 goal 的 in-context 行为切换、以及长程多阶段任务；真实机器人上可用人类演示的 goal image 跨 embodiment 指定目标。

**不能做什么 / 弱在哪里：**

- **记忆不是强项。** Vanilla 没有专门记忆模块，RoboDojo Memory 组 6.00 / 7.00；Goal 虽支持可选历史帧，但论文没有给历史窗口长度，也没有专门记忆基准。不能把 48-step chunk 或 12 future frames 说成记忆机制。
- **多视角直接后训练会伤 OOD。** isolated attention 下，Single-view scene-ood 46.08%，Multi-view conditioning 34.08%，Multi-view prediction 33.00%；random 22.48% → 12.92% / 14.84%。EgoPro mid-training 能缓解，但 IDM 下 Multi-view conditioning 的 scene-ood 61.20% 仍略低于 Single-view 62.00%。
- **Goal 的 hard-task 零样本会 plateau。** 真实机器人 hard 从 18% 到 19%，标准数据比例采样下增益很小；robot-balanced 采样能到 27%，但这是另一个配置。
- **真实机器人 trials 少。** Vanilla 每任务每 setting 10 trials，Goal 每任务 10 trials；任务级百分比的分辨率是 10%，不能过度解读小差距。
- **仿真 goal image 是合成+审核的。** 7,032 张新图中首轮 9.3% 未通过；虽然修复后全部通过，但生成与审核引入额外变量。
- **跨 embodiment 仍有限。** 预训练覆盖人类与多种机器人，但闭环评测主要是 RoboTwin Aloha-AgileX、Cobot Magic、Astribot S1；不能声称对任意人形/移动平台有效。

### 6.1 隐含假设

1. **人类手部 EEF 可以近似机器人 EEF。** 论文用位置、四元数、夹爪开合对齐，但没有建模接触力、手指柔顺性、工具使用中的力闭合；这对精密插入、 deformable 操作是潜在缺口。
2. **相机系绝对动作在单一目标 embodiment 后训练时可对齐。** Vanilla 最终用绝对 camera-frame，Goal 改用 chunk-relative 以支持零样本；两种表示的适用边界是条件性的，不是普适最优。
3. **数据 curation 阈值可迁移。** 手部可见性、稳定性、空闲阈值没有完整给出，换数据集是否需要重调，论文未做敏感性分析。
4. **缩放律可外推。** 拟合点有限（isolated attention 到 10k h，IDM 到 50k h），R² 高不等于外推到 100k h 仍成立；论文自己也只是说“经验”缩放律。
5. **每个设置取最佳 checkpoint 不偏倚比较。** 这在工程上合理，但会让不同规模模型的比较偏向乐观；尤其是小模型可能在某个 checkpoint 偶然表现好。
6. **goal image 是充分的任务规格。** 仿真 goal 由生成模型合成并审核，真实机器人 goal 来自 held-out 演示；这避开了语言歧义，但也意味着评测的是“看图执行”，不是“纯语言理解”。
7. **10 trials/task 足以比较真实机器人。** 对 20%–30% 的成功率，10 次试验的抽样误差很大；论文的趋势判断应视为方向性证据，而不是精确排名。

---

## 7. 与相关工作对比

| 工作 | 关注点 | 数据规模 | 评测 | 与本文的区别 |
|------|--------|----------|------|--------------|
| EgoScale | 第一视角人类数据缩放，验证损失 log-linear | 20,000+ 小时 | 验证损失 + 机器人任务 | 本文把证据推进到闭环成功率，并加入多样性控制 |
| HumanScale | 人类视频预训练后验证动作损失缩放律 | 未在本文给出 | 验证动作损失 | 本文用闭环 success，且比较人类 vs 机器人同小时 |
| Dyna-2 | 一百万小时人类视频，零样本动作预测缩放 | 1M 小时 | 零样本动作预测 + 后训练 | 本文规模小一个量级，但做闭环与真实机器人 |
| OpenWAM | 世界-动作信息流与预训练 | 未在本文给出 | 多基准 | 本文重点是数据缩放，不是架构消融 |
| Fast-WAM | 训练时视频预测、推理时省略 | 未在本文给出 | 多基准 | 本文借用其 Optional-IDM 变体做消融 |
| π0.5 | VLA 开放世界泛化 | 未在本文给出 | 多基准 | 本文是世界-动作模型，不是纯 VLA |

**面试 Tip**：被问“这篇和 HumanScale/EgoScale 有什么区别”时，一句话答：前者主要看验证损失，本文把同一套人类数据缩放问题推到闭环任务成功率，并证明多样性、动作对齐和后训练样本效率是三个独立杠杆；但本文的缩放律仍是有限点上的经验拟合，不能当定律背。

---

## 8. 精读建议

- **值得精读原文的人**：
  1. 正在搭建 egocentric 预训练数据管线的人——§2 的 curation 与三套对齐是可直接借鉴的工程清单。
  2. 做世界-动作模型、需要决定 action chunk、视频预测视野、attention 配置的人——§3.1、§4.4、§5.3 有明确的消融与取舍。
  3. 要评估“人类数据值不值得买”的团队——§3.4、§3.7 给出了样本效率与人类 vs 机器人的同小时比较。
- **建议章节路径**：先读 §2（数据与动作接口）→ §3.2–§3.4（缩放律、多样性、样本效率）→ §4.1–§4.2（Vanilla 最终配置与 RoboTwin 结果）→ §5.2–§5.3（Goal 的 49-D 动作与 goal conditioning）→ §6（真实机器人与失败模式）。§4.4 的消融可以按需回看，§7 相关工作可快速扫过。
- **不值得精读的理由**：如果你不做机器人学习，或者只关心 VLA 的语言/推理能力，这篇的核心是数据与动作接口，读摘要和 §3.2 的缩放律即可；如果你已经非常熟悉世界-动作模型架构，且只想要一个新模块，这篇没有提出全新的注意力或生成范式，它的贡献在配方与实证。

---

[← Back to Theory](./README.md)

**关键引用**：论文全文与所有数字均来自 [Open-HumanScale, arXiv:2611.06817v1](https://arxiv.org/abs/2611.06817)。本文未使用论文之外的代码仓库或额外 URL。