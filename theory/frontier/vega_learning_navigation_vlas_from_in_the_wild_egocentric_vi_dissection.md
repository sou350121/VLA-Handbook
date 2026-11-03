# 从野生第一视角视频学导航 VLA：VEGA 的几何轨迹监督 (VEGA: Learning Navigation VLAs from In-the-Wild Egocentric Video with Geometric Trajectory Supervision)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-11-03
>
> **论文**: VEGA: Learning Navigation VLAs from In-the-Wild Egocentric Video with Geometric Trajectory Supervision
> **链接**: https://arxiv.org/abs/2606.18426
> **核心定位**: 用单目几何把无标注第一视角视频转成避障轨迹监督，训练一个推理时只看 RGB + goal 的 flow-matching 导航 VLA；相比依赖机器人演示的导航 VLA，作者在真实四足机器人试验中报告了大幅降低碰撞并提高成功率。

> **来源说明**：本文以 arXiv:2606.18426v3（2026-10-30）全文（HTML 渲染）为证据。候选摘要片段在“obstacle-awar”处截断，未用于补写任何主张；若官方 PDF 与元数据摘要存在差异，以 v3 全文正文为准。文中所有数字均标注论文表格或章节位置；论文未报告的项目写“论文未报告”，不做推算填充。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 几何规划可以只在训练时出现：用 MoGe-2 + ESDF + Dijkstra 生成避障 waypoint 轨迹，再用 flow matching 蒸馏进 VLA；推理只需 RGB + goal。 |
| 适合精读 | 做 navigation VLA、learning from video、自动数据引擎、几何监督、sim-to-real 的读者；重点看 §IV-A、§IV-B、§V-E。 |
| 可以跳过 | 只做 manipulation、或不关心几何监督的读者；读摘要 + Table II/III 即可。 |
| 落地可行性 | 中高：训练管线重（单目重建、BEV、ESDF、Dijkstra、轨迹优化），推理轻；但真实世界只测了四足、4 个场景、每场景 20 trials。 |
| 主要风险 | memoryless、静态 ESDF 不建模动态障碍轨迹、单目几何误差、真实试验样本小、统一接口下基线（尤其 NaVILA 的 LiDAR 避障被禁用）并非各系统最佳性能。 |

💡 **X-Ray 开场**

导航 VLA 缺大规模机器人演示，而互联网第一视角视频很多却没有动作标签和 goal 标签。VEGA 的做法是：先用单目几何把每帧变成多个 goal 的避障轨迹，再把这些轨迹蒸馏进一个 flow-matching VLA。对 VLA 研究者来说，这意味着“几何监督”可以只存在于离线数据管线，部署时退回纯 RGB 策略——这是一条可复制的“无机器人数据训导航 VLA”路径，但它的边界也很清楚：没有记忆、没有动态障碍预测、真实试验规模有限。

📍 **研究全景时间线**

```
2024  LeLaN：用预训练导航策略给野生视频打轨迹标签
  │
2025  NaVILA / OmniVLA：机器人导航数据监督的导航 VLA
  │
2026  TIC-VLA：动态环境下的 think-in-control 导航 VLA
  │
2026  VEGA（本文）：几何监督 + 无机器人演示 + 多模态 goal
```

时间线只列论文正文明确引用且年份可确认的节点；不把参考文献表全部展开。

---

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 阶段 | 输入 | 输出 | 关键组件 | 训练/推理差异 |
|------|------|------|----------|----------------|
| 数据生成 | 无标注 egocentric RGB 帧 I_t | 点图、BEV、ESDF、多模态 goal、参考 waypoint 轨迹 | MoGe-2、RANSAC、Florence-2、heading-aware Dijkstra、ESDF 轨迹优化 | 仅训练/离线；需要几何 |
| 策略训练 | (I_t, goal, 参考轨迹) | flow-matching 损失 | π0.5 组件：SigLIP、Gemma、waypoint encoder、action expert、2D waypoint MLP | 训练；几何只作为标签 |
| 推理 | 当前 RGB I_t + goal | 未来 H=50 个 2D waypoint 的 chunk | 同一个 VLA；不需要重建/ESDF | 推理；无几何、无深度、无 LiDAR |
| 执行 | waypoint chunk | 机器人运动 | 共享控制栈（论文引用 [15]），不额外规划/避障 | 部署；允许在完成预测前从更新观察 replan |

> 论文未报告控制频率、推理延迟、flow matching 积分步数；表中“频率/时序”不填具体数字，避免把 H=50 误写成 50 个控制周期或 50 秒。

### 1.2 关键机制 (Key Mechanism)

- **几何转监督**：对每帧 I_t，MoGe-2 输出相机坐标系下的 dense metric point map P_t(u,v)。RANSAC 拟合局部地面，把点图旋转到机器人坐标系（xy 平面为地面，z 向上）。在机器人周围定义 BEV 域 Ω_BEV，把点投影到地面；按高度区间 [z_min, z_max] 过滤出障碍点 P_obs，栅格化成 occupancy grid M_occ；再对 organized point map 做 raycasting 得到 visibility mask M_vis，把 BEV 分成 known free、occupied、unknown 三类。最后计算 visibility-aware ESDF。
- **多模态 goal 锚定**：Florence-2 检测物体标签和 bbox；对每个检测 j，在 MoGe-2 点图中查询对应区域并投影到地面，得到 waypoint goal g_j^wp = (x_j, y_j)。这样同一目标同时有 text goal、image-region goal、waypoint goal 三种对齐标注。论文还从观测到的 free space 中采样辅助 waypoint goals，给“任意可通行位置”提供监督。
- **避障轨迹生成**：对每个 waypoint goal，先在 visibility-aware ESDF 上做 heading-aware Dijkstra，得到一条 collision-free、朝向目标的初始路径；不可达时选最近可达端点，超出局部 ESDF 的 goal 丢弃。然后固定起点和终点，只优化 interior waypoints：一项惩罚 clearance 不足，一项惩罚不平滑。论文设置 d_safe = 0.5 m（对应机器人半径），waypoint chunk horizon H = 50。
- **蒸馏进 VLA**：训练 tuple 为 (I_t, g_j, m_j, ŷ_{t:t+H,j})，其中 m_j ∈ {text, box, wp}。策略用 flow matching 学这个轨迹分布；推理时从高斯噪声积分学到的 vector field，输出未来 waypoint chunk。几何管线在推理时全部消失。

⚡ **Eureka Moment**：这篇论文最核心的想法不是“用视频训 VLA”，而是**把经典几何规划器当成离线标注器**——让 ESDF/Dijkstra 在训练时提供避障监督，然后把这个能力蒸馏进一个推理时不需要几何的 RGB 策略。

### 1.3 信息流/架构图 (Flow / Diagram)

```
无标注 egocentric video（walking tours / bike rides / first-person navigation）
        │
        ▼
固定帧率均匀采样 → 当前帧 I_t（每帧视为一个 local navigation scene）
        │
        ├─ MoGe-2 → dense point map P_t
        │       → RANSAC 地面对齐 → BEV 投影 → 高度滤波 → occupancy M_occ
        │       → raycasting visibility → M_vis
        │       → visibility-aware ESDF
        │
        ├─ Florence-2 → 物体标签 + bbox
        │       → 点图区域投影 → waypoint goal g_j^wp
        │       → 对齐 text / image-region / waypoint goals
        │       → 另从 free space 采样辅助 waypoint goals
        │
        ▼
heading-aware Dijkstra（ESDF 上）→ 初始避障路径
        │
        ▼
ESDF 轨迹优化（固定端点，优化 interior waypoints）
        │
        ▼
训练 tuple：(I_t, g_j, m_j, ŷ_{t:t+H,j})
        │
        ▼
flow-matching VLA：SigLIP + Gemma（冻结）+ waypoint encoder / action expert（从零训练）
        │
        ▼
推理：I_t + goal → 未来 H=50 个 2D waypoint → 共享控制栈 → 机器人
```

---

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：

```
用 ESDF 规划出的未来 waypoint 轨迹，去回归一个 flow-matching 策略：π_θ(ŷ_{t:t+H} | I_t, g)
```

先给目标：VEGA 要学的不是一个确定性回归器，而是一个以当前 RGB 和 goal 为条件、能表示多模态未来轨迹的生成策略。几何规划器提供目标分布样本，flow matching 负责把这个分布蒸馏进网络。

**策略与未来 chunk**（论文 Eq. 1）：

```
π_θ(a_{t:t+H} | I_t, g)
```

- I_t：当前 RGB 观测。论文的 VEGA 是 memoryless，输入只有当前帧，没有历史帧堆叠。
- g：goal，支持 text、image-region、waypoint 三种模态。
- a_{t:t+H}：未来动作 chunk；在 VEGA 的具体实现中，它被具体化为机器人坐标系下的 2D waypoint 序列。
- H：waypoint chunk 的 horizon，论文设置为 50。

> 注意：H=50 是**未来预测 chunk 的 waypoint 数**，不是历史上下文长度，也不是记忆窗口。论文没有给 waypoint 之间的时间间隔或空间间隔，因此不能把 H=50 换算成秒数或米数。若从当前参考 t 预测 t+1 至 t+H，终点距当前参考是 H×Δt；但论文未报告 Δt，所以这里只保留符号关系，不填具体值。

**Flow matching**（论文 Eq. 2–3）：

```
y_τ = (1 - τ) y_0 + τ y_1,   τ ∈ [0, 1]

L_FM(θ) = E[ || v_θ(y_τ, τ, I_t, g) - (y_1 - y_0) ||₂² ]
```

- y_1：目标轨迹，即 ESDF 生成的参考 waypoint 轨迹。
- y_0 ~ N(0, I)：与 y_1 同维的高斯噪声。
- y_τ：线性插值路径上的点；τ 是 flow 的插值时间，不是机器人时间步。
- v_θ：网络学到的 vector field，条件为 y_τ、τ、I_t、g。
- 推理：从 τ=0 的噪声开始，积分学到的 vector field 到 τ=1，得到一条轨迹样本。

**ESDF 定义**（论文 Eq. 6）：

```
ESDF(x,y) = +d_unsafe(x,y),   (x,y) ∈ X_free
ESDF(x,y) = -d_free(x,y),    (x,y) ∈ X_occ ∪ X_unk
```

- d_unsafe(x,y)：free space 中的点到最近的 occupied 或 unobserved cell 的欧氏距离。
- d_free(x,y)：occupied/unknown 区域中的点到最近的 known-free cell 的距离。
- 正值表示 known free space 内的 clearance；负值表示 planner 应避开的区域。
- 这个符号约定很重要：优化项只会在 ESDF 小于 d_safe 时惩罚，负值会被 max(0, ·) 截断，但 planner 的 Dijkstra 会直接避开负值区域。

**轨迹优化**（论文 Eq. 8）：

```
min_{p_1,...,p_{H-2}}  λ_obs  Σ_{i=1}^{H-2} [max(0, d_safe - ESDF(p_i))]²
                     + λ_smooth Σ_{i=1}^{H-2} || p_i - p̄_i ||₂²
```

- p_i：第 i 个 interior waypoint；起点 p_0 和终点 p_{H-1} 固定。
- p̄_i：邻居 waypoint 的加权平均，用来定义平滑目标。
- d_safe = 0.5 m（论文设置，对应机器人半径）。
- H = 50（与 waypoint chunk horizon 一致）。
- λ_obs、λ_smooth：论文正文未报告具体取值，TODO 待补。不能从论文中编造这两个数。

**变量与符号保持一致**：

| 符号 | 含义 | 备注 |
|------|------|------|
| I_t | 当前 RGB 帧 | 只有当前帧，VEGA 不使用历史帧 |
| g | goal | text / image-region / waypoint |
| H | waypoint chunk 的 horizon | 论文设为 50 |
| y_0 | 高斯噪声 | 与目标轨迹同维 |
| y_1 | ESDF 生成的参考轨迹 | 训练目标 |
| y_τ | flow 插值点 | τ ∈ [0,1] |
| d_safe | 安全距离 | 0.5 m |
| Ω_BEV | 局部 BEV 域 | 25 m × 20 m，0.02 m/cell |

**直觉**：Dijkstra 负责“拓扑上走得通”，ESDF 优化负责“离障碍够远且平滑”，flow matching 负责“把这一族轨迹变成可采样的策略分布”。三者分工明确，几何只在训练时承担“老师”的角色。

---

## 3. 带数字走一遍：玩具例子 (Worked Example)

> ⚠️ 以下为**假设的 2D 例子**，用于验算 ESDF 符号和轨迹优化，不是论文实验，也不代表论文的超参数。

**设定**：忽略栅格量化，把 ESDF 近似为到矩形障碍边界的连续欧氏距离。障碍为 x ∈ [0.8, 1.2]、y ∈ [-0.2, 0.2] 的矩形。d_safe = 0.5 m。H = 3，端点固定为 p_0 = (0,0)、p_2 = (2,0)，唯一 interior waypoint 是 p_1。λ_obs = 1.0，λ_smooth = 0.1。平滑目标 p̄_1 = (p_0 + p_2)/2 = (1,0)。

**ESDF 符号验算**：

- free 点 p_1 = (1, 0.7)：到障碍上边界 y=0.2 的距离是 0.5，所以 ESDF = +0.5。
- occupied 内点 (1, 0.1)：到最近 free 边界 y=0.2 的距离是 0.1，所以 ESDF = -0.1。

这验证了论文的符号约定：free space 为正，occupied/unknown 为负。

**初始路径**：假设 Dijkstra 给出 p_1 = (1, 0.3)，ESDF = 0.1（到上边界 0.2 的距离）。此时 clearance 不足：

```
obs = 1.0 × [max(0, 0.5 - 0.1)]²
    = 1.0 × 0.4²
    = 0.16

smooth = 0.1 × ||(1, 0.3) - (1, 0)||₂²
       = 0.1 × 0.3²
       = 0.009

total = 0.16 + 0.009 = 0.169
```

**优化后候选**：p_1 = (1, 0.7)，ESDF = 0.5，刚好满足 d_safe：

```
obs = 1.0 × [max(0, 0.5 - 0.5)]² = 0

smooth = 0.1 × ||(1, 0.7) - (1, 0)||₂²
       = 0.1 × 0.49
       = 0.049

total = 0 + 0.049 = 0.049
```

**验算结论**：优化把 waypoint 从 clearance 0.1 m 推到 0.5 m，总损失从 0.169 降到 0.049；平滑项阻止它继续远离直线。

**反例（说明 λ 的平衡很重要）**：如果 λ_smooth 改成 1.0，初始总损失变成 0.16 + 1.0×0.09 = 0.25；优化后变成 0 + 1.0×0.49 = 0.49。此时优化器反而宁愿贴着障碍，也不愿偏离直线。也就是说，λ_smooth 太大时，避障项会被平滑项压过。论文没有报告 λ 的具体取值，因此这个反例只说明目标函数的条件性行为，不能推断论文的实际设置。

---

## 4. 工程视角 (Engineering View)

**训练重、推理轻**是 VEGA 最核心的工程 trade-off。数据生成侧包含：MoGe-2 单目重建、RANSAC 地面对齐、BEV 投影、高度滤波、raycasting visibility、ESDF、Florence-2 检测、Dijkstra、轨迹优化。推理侧只剩：RGB + goal → VLA forward → flow 积分 → waypoint chunk。论文没有报告数据生成吞吐、GPU 型号、训练时长、推理延迟、控制频率；这些是 TODO，不能从论文数字反推。

**BEV 规模**：25 m × 20 m，0.02 m/cell，即 1250 × 1000 = 1.25M cells。ESDF 和 Dijkstra 都在这个网格上跑。离线生成 8M 条 goal-conditioned 轨迹时，这个计算量不可忽略。论文没有给并行化或缓存策略，工程落地时需要自己评估。

**输出维度**：H = 50，2D waypoint，即 50 × 2 = 100 维输出。Flow matching 推理需要积分步数，论文未报告；不能把 H=50 当作 50 个控制周期或 50 秒。实际部署时，waypoint chunk 交给共享控制栈执行；论文说所有方法可以在完成预测前从更新观察 replan，但未给 replan 频率。

**记忆机制**：VEGA 是 memoryless，输入只有当前帧 I_t 和 goal。H=50 是未来预测 chunk，不是历史上下文，也不是记忆窗口。作者在 limitation 中明确说：障碍或目标离开视野时，memoryless 架构会脆弱；建议引入 temporal memory 或 recurrent state。这个判断是作者自己提出的，不是实验验证过的改进。

**动态障碍**：训练监督是静态 ESDF，没有时间维度；Isaac-Sim 里有移动行人，但策略没有显式预测行人轨迹。作者建议未来用 motion prediction 或 time-indexed distance fields。这意味着当前版本对“突然横穿”的动态障碍，主要靠当前帧反应，而不是预测。

**部署约束**：推理不需要深度、LiDAR、地图或 ESDF，适合低成本 RGB 平台；但 waypoint goal 仍需 2D 坐标，text/image goal 需要对应编码器。共享控制栈不额外做路径规划或避障，所以避障责任在策略输出本身；这也是作者禁用 NaVILA 原生 LiDAR 避障的原因——他们要评估 VLA 预测本身，而不是下游模块的修正。

**数据规模**：250 小时 egocentric video，约 8M 条 goal-conditioned trajectories。论文说这是最强导航基线训练数据时长的约 2.6%，但没有点名最强基线，也没有给其具体小时数。按 250 ÷ 0.026 反推约 9.6k 小时，只能作为量级参考，不是论文实测值。论文没有给视频采样帧率，因此不能从 250 小时推算帧数；任何“每帧多少条轨迹”的换算都需要额外假设。

---

## 5. 数据与评测 (Data & Eval)

### 5.1 预训练/训练数据与监督生成

| 项目 | 论文所述 | 备注 |
|------|----------|------|
| 视频来源 | online walking tours、bike rides、other first-person navigation footage | 无标注；不是机器人数据 |
| 采样 | 固定帧率均匀采样，每帧视为一个 local navigation scene | 论文未报告具体帧率 |
| 几何 | MoGe-2 dense point map；RANSAC 地面对齐；BEV 25 m × 20 m @ 0.02 m/cell | 仅训练时使用 |
| 障碍表示 | 高度滤波 + occupancy grid + raycasting visibility + ESDF | ESDF 区分 free/occ/unknown |
| Goal | Florence-2 检测 + 点图投影 → text/image-region/waypoint 对齐；free-space 辅助 waypoint | 同一帧多个 goal |
| 轨迹 | heading-aware Dijkstra + ESDF 优化；d_safe=0.5 m；H=50 | 固定端点，优化 interior |
| 数据量 | 约 8M goal-conditioned trajectories；250 小时视频 | 含 point clouds、ESDFs、reference trajectories |
| 架构 | π0.5 组件；SigLIP/Gemma 冻结；waypoint encoder/action expert 从零训练；2D waypoint MLP 可训练 | 论文 §IV-B、Fig. 3 |
| 推理 | 只需 RGB + goal；不需要重建、ESDF、深度、LiDAR | 论文反复强调 |

> 数据开放范围：作者在 §I 末尾声称“will release the trained model weights, dataset, and VEGA-Bench evaluation suite upon publication”。这是**作者声称将发布**，不是截至 v3 已开放；论文正文没有给代码仓库 URL，本文也不杜撰。基线模型的数据开放范围论文未述，不能替它们下结论。

### 5.2 仿真评测

**VEGA-Bench**（论文 §V-D、Table III）：

- 整体 benchmark：250k scenes、约 8M navigation goals，配 scene geometry；指标为 goal progress、collision rate、obstacle clearance。
- 主实验 test set：约 12.5 小时 held-out videos，按 city-level 切分，避免同一视频的帧及其派生 goal/trajectory 同时出现在训练和评测。
- 主比较使用 common text-conditioned subset；modality-specific 结果在真实世界单独报告（见 5.3），不要把 Table III 的 goal-conditioning 列当成 VEGA-Bench 结果。

| 方法 | G ↑ | C ↓ | Cl ↑ | 备注 |
|------|-----|-----|------|------|
| OmniVLA | 0.88 | 6.75 | 2.35 | 最强基线（按 C 和 Cl） |
| TIC-VLA | 0.84 | 9.01 | 1.34 |  |
| NaVILA | 0.77 | 7.40 | 2.18 |  |
| VEGA | 0.89 | 5.52 | 2.87 | goal progress 只高 0.01，论文用“competitive” |

按表验算作者摘要中的相对变化：

```
碰撞相对降低 = (6.75 - 5.52) / 6.75 = 0.1822 ≈ 18.2%
clearance 相对提升 = (2.87 - 2.35) / 2.35 = 0.2213 ≈ 22.1%
```

这两个是**相对变化**，不是百分点。Goal progress 从 0.88 到 0.89，只高 0.01，论文没有把它说成显著提升，这个措辞是准确的。

**几何敏感性**（Table IV）：用 UniDepthV2、UniK3D、Metric3Dv2 重新估计 ESDF 和目标坐标，VEGA 仍保持最低碰撞率、最高 clearance，方法排序保持；Spearman 相关系数在 0.91–1.00 量级。注意 VEGA 在 UniK3D 下的 C=5.39，比 MoGe-2 的 5.52 更低，说明绝对值和具体几何源有关，不能把 5.52 当成固定常数。

**Isaac-Sim**（Table I）：3 个场景，每个方法每场景 50 trials；静态障碍 + 移动行人，行人运动 onset 在各方法间用相同 timing conditions。

| 场景 | 方法 | S ↑ | C ↓ | Cl ↑ |
|------|------|-----|-----|------|
| Scn1 | VEGA | 1.00 | 0.00 | 0.85 |
| Scn1 | NaVILA | 0.80 | 0.00 | 0.84 |
| Scn2 | VEGA | 0.90 | 0.10 | 0.49 |
| Scn2 | NaVILA | 0.90 | 0.02 | 0.53 |
| Scn3 | VEGA | 1.00 | 0.00 | 1.06 |
| Scn3 | NaVILA | 0.00 | 0.00 | 1.52 |

条件性分析：VEGA 并非每个场景每个指标都最好。Scn2 中 NaVILA 的碰撞率更低（0.02 vs 0.10）、clearance 更高（0.53 vs 0.49）；Scn3 中 NaVILA 的成功率为 0，但其 clearance 1.52 是在非碰撞轨迹上统计的，样本可能很少，不能直接解读为“更安全”。论文的总体结论是 VEGA 在 50 simulations per scenario 下整体更好，但读者应保留这些局部反例。

**消融**（Table V）：

| 消融 | 设置 | G ↑ | C ↓ | Cl ↑ |
|------|------|-----|-----|------|
| 几何监督 | ESDF-optimized | 0.92 | 1.2 | 3.12 |
| 几何监督 | Straight-line | 0.94 | 14.8 | 2.14 |
| LiDAR 验证 | SCAND LiDAR 规划 | 0.91 | 0 | 0.492 |
| LiDAR 验证 | MoGe-2 规划 | 0.881 | 3.2 | 0.385 |

- ESDF vs 直线：碰撞从 1.2% 升到 14.8%，clearance 从 3.12 m 降到 2.14 m，goal progress 几乎不变（0.92 vs 0.94）。说明几何规划主要改善避障/clearance，不是 goal 到达本身。
- LiDAR 验证：用同一 LiDAR-ground-truth ESDF 评估，MoGe-2 规划与 SCAND LiDAR 规划的轨迹平均 Fréchet 距离 0.28 m；goal progress 0.881 vs 0.91，碰撞 3.2% vs 0，clearance 0.385 vs 0.492。说明单目几何与 LiDAR 几何接近，但不是零差距。
- 不同几何源生成监督（Table V(a)）：MoGe-2 C=0、UniDepthV2 C=0.10、UniK3D C=0.12、Metric3Dv2 C=0.18；clearance 3.35/3.22/3.27/3.08。监督质量对几何源有一定敏感性。

### 5.3 真实机器人设置

- 平台：四足机器人（论文未给型号）。
- 场景：4 个真实场景，含静态和动态障碍；每场景 20 trials。
- 执行：所有方法共享同一个控制栈（论文引用 [15]），该栈不提供额外路径规划或避障；允许在完成预测前从更新观察 replan。
- Goal 接口：通过每个模型支持的接口给 goal，保持目标一致；NaVILA 的高层导航命令被转成等价的 local waypoints。
- 关键条件：**不使用 NaVILA 原生的 LiDAR-conditioned 避障 locomotion policy**，因为作者要评估 VLA 预测本身的避障，而不是下游模块的修正。这意味着 NaVILA 的比较不是其完整系统的最佳性能。

| 场景 | 方法 | S ↑ | C ↓ | Cl ↑ |
|------|------|-----|-----|------|
| Scn1 | VEGA | 0.95 | 0.05 | 0.93 |
| Scn1 | OmniVLA | 0.15 | 0.85 | 0.35 |
| Scn1 | TIC-VLA | 0.00 | 1.00 | N/A |
| Scn1 | NaVILA | 0.00 | 1.00 | N/A |
| Scn2 | VEGA | 0.75 | 0.25 | 0.56 |
| Scn2 | OmniVLA | 0.00 | 0.65 | 0.32 |
| Scn2 | TIC-VLA | 0.10 | 0.85 | 0.37 |
| Scn2 | NaVILA | 0.00 | 0.80 | 0.35 |
| Scn3 | VEGA | 1.00 | 0.00 | 0.94 |
| Scn3 | OmniVLA | 0.00 | 0.50 | 0.44 |
| Scn3 | TIC-VLA | 0.00 | 1.00 | N/A |
| Scn3 | NaVILA | 0.00 | 0.45 | 0.31 |
| Scn4 | VEGA | 0.95 | 0.05 | 0.52 |
| Scn4 | OmniVLA | 0.00 | 1.00 | N/A |
| Scn4 | TIC-VLA | 0.00 | 1.00 | N/A |
| Scn4 | NaVILA | 0.00 | 0.75 | 0.25 |

按表验算作者摘要中的“至少 65 个百分点、碰撞至少降 61.5%、clearance 至少升 51.4%”：

```
最小 success 差：Scenario 2，VEGA 0.75 - TIC-VLA 0.10 = 0.65
  → 65 个百分点（percentage points），不是 65% 相对提升

最小碰撞相对降低：Scenario 2， (0.65 - 0.25) / 0.65 = 0.6154
  → 61.5% 相对降低

最小 clearance 相对提升：Scenario 2， (0.56 - 0.37) / 0.37 = 0.5135
  → 51.4% 相对提升
```

这三个数字都能从 Table II 复算出来，但要注意单位和参照对象：第一个是百分点差，后两个是相对变化。

**Goal modality 真实世界实验**（Table III 右列）：Scenario 4 布局，移除动态障碍，每个 supported modality 20 trials。

| 方法 | Text ↑ | Image ↑ | WP ↑ |
|------|--------|---------|------|
| OmniVLA | 0.8 | 0.2 | 0.6 |
| TIC-VLA | 0.8 | N/A | N/A |
| NaVILA | 0.6 | N/A | N/A |
| VEGA | 1.0 | 1.0 | 0.8 |

- 这是真实世界 20 trials，不是 VEGA-Bench。
- VEGA 的 WP 是 0.8（16/20），不是完美；text/image 各 1.0。
- TIC-VLA 和 NaVILA 的 N/A 表示实验只测了 supported modality，不代表架构绝对不支持其他模态；论文没有展开它们的模态能力，不能替它们补写。
- OmniVLA 支持 text/image/2D pose（论文 §II-A），这里 WP 列对应 waypoint goal；不要把它和 VEGA 的 image-region/waypoint 混为一谈。

---

## 6. 能力与失败模式 (Capabilities & Failure Modes)

**能做什么**：

- 从无标注第一视角视频生成 goal-conditioned、obstacle-aware 的 waypoint 监督，不需要机器人演示或人工动作标注。
- 支持 text、image-region、waypoint 三种 goal；真实世界 Scenario 4 去动态障碍的 20-trial 实验中，VEGA 三种模态分别达到 1.0/1.0/0.8 成功率。
- 推理时只需要 RGB + goal，不需要深度、LiDAR、地图、ESDF 或在线重建。
- 在作者测试的 4 个真实场景和 3 个 Isaac-Sim 场景中，整体成功率和避障指标优于所比较的导航 VLA 基线。

**不能做什么/未验证什么**：

- **没有记忆**：VEGA 是 memoryless，只有当前帧。障碍或目标离开视野后，策略没有历史状态来恢复；作者在 limitation 中明确建议加 temporal memory/recurrent state，但这不是本文实验。
- **没有动态障碍轨迹预测**：训练监督是静态 ESDF；Isaac-Sim 的移动行人只用于评测，策略没有显式预测行人未来轨迹。作者建议 motion prediction 或 time-indexed distance fields。
- **没有多机器人平台验证**：真实实验只用了四足机器人；不能外推到轮式、无人机、人形或不同动力学。
- **没有户外长距离/复杂语言组合的专项评测**：论文的场景是 cluttered indoor/outdoor 局部导航，不是长程指令跟随。
- **单目几何失效场景未系统评测**：透明、反光、弱纹理、远距离、细结构障碍都可能让 MoGe-2 点图出错；论文用替代几何源做了敏感性分析，但没有针对这些具体材质做专项实验。
- **BEV 外目标被丢弃**：局部 ESDF 是 25 m × 20 m，超出范围的 goal 在数据生成时被丢弃；策略没有学过这种 goal。
- **unknown space 可能过度保守**：ESDF 把 unknown 当负值，planner 会避开；这可能让策略绕开实际可通行但未观测的区域。
- **真实试验样本小**：每场景 20 trials，4 个场景；成功率差 65 个百分点虽然大，但置信区间仍宽，不能当作所有部署环境的保证。

### 6.1 隐含假设 (Hidden Assumptions)

1. **单目几何足够准**：整个监督管线建立在 MoGe-2 点图可信的前提上；论文用 LiDAR 对比和替代几何源做了支持，但不是所有场景都验证过。
2. **局部 BEV 范围足够**：25 m × 20 m 被认为覆盖导航决策所需范围；超出范围的目标直接丢弃，这默认了目标总在局部窗口内。
3. **静态 ESDF 能代表动态场景的避障需求**：训练监督没有时间维度，但评测含移动行人；这默认了“当前帧 + 静态 clearance”足以应对动态障碍。
4. **当前帧足够（memoryless）**：默认部分可观测性不会致命；作者自己指出这在障碍/目标出视野时脆弱。
5. **共享控制栈不引入额外避障**：真实实验的比较被设计成“只评估 VLA 预测本身”；这默认了控制栈不会掩盖或修复策略的错误。
6. **20 trials 能区分方法**：真实世界每场景 20 trials，统计功效有限；作者没有给置信区间。
7. **统一接口下基线公平**：NaVILA 的原生 LiDAR 避障被禁用；作者给了理由，但读者应把它理解为“统一接口下的比较”，不是各系统最佳性能。
8. **250 小时视频分布覆盖部署环境**：训练视频来自 walking tours/bike rides 等，默认与测试室内外 clutter 分布足够接近。
9. **8M 轨迹的 goal 分布覆盖测试 goal**：VEGA-Bench 的 goal 来自场景几何和检测，默认与真实部署的 goal 类型一致。

---

## 7. 与相关工作对比 (Comparison)

| 方法 | 监督来源 | 几何使用 | 支持 goal（论文所述） | 记忆（论文所述） | 评测（论文所述） |
|------|----------|----------|------------------------|------------------|------------------|
| VEGA | 无标注 egocentric video + 几何生成轨迹 | 训练时 MoGe-2/ESDF，推理不用 | text / image-region / waypoint | memoryless（作者 limitation） | VEGA-Bench + Isaac-Sim + 真实四足 |
| LeLaN | 无标注 egocentric video，用预训练 navigation policy 生成轨迹 | 不显式做几何规划；继承预训练策略的避障局限 | 语言条件 | 论文未述 | 论文未展开 |
| OmniVLA | 机器人导航数据（论文 §II-A） | 论文未述 | text / image / 2D pose | 论文未述 | 本文基线 |
| NaVILA | 机器人数据（论文 §II-A） | 论文未述；实验禁用其 LiDAR 避障 locomotion | text（实验） | 论文未述 | 本文基线 |
| TIC-VLA | 论文未展开 | 论文未展开 | text（实验） | 论文未述 | 本文基线 |
| LiMo | robot-collected observations + co-registered elevation maps | 几何（elevation map） | goal-conditioned（细节未展开） | 论文未述 | 论文未作为基线 |

> 这张表刻意逐模型归属属性：不能把 VEGA 的“多模态 goal + memoryless + 训练时几何”合并给所有模型；也不能把 OmniVLA 的 2D pose goal 写成 VEGA 的 waypoint goal。论文没有描述的，写“论文未述/未展开”。

**面试 Tip**：如果被问“VEGA 和 LeLaN 的根本区别”，一句话答：LeLaN 的轨迹标签来自一个预训练导航策略，会继承该策略的避障缺陷；VEGA 用单目重建几何 + ESDF + Dijkstra/优化生成避障轨迹，几何只在训练时用，推理是纯 RGB VLA。

---

## 8. 精读建议 (Reading Guide)

- **值得精读原文的人**：
  - 做 navigation VLA、robot learning from video 的研究者：§IV-A 的数据管线是可复用的“几何标注器”范式。
  - 做自动数据引擎/自动标注的工程师：§IV-A1–A3 的 BEV/ESDF/Dijkstra/优化细节最值得看。
  - 要评估“无机器人数据训 VLA”可行性的工程团队：§V-B、§V-D、§V-E 给出了真实、仿真、消融三层证据，但要注意样本规模和接口条件。
  - 做 sim-to-real 的读者：§V-C 的 Isaac-Sim 动态行人设置和 §VI 的 limitation 值得对照。

- **建议章节路径**：
  - 先读 §IV-A（数据生成）→ 再读 §IV-B（VLA 架构）→ 然后 §V-B/§V-D（真实与 benchmark）→ 接着 §V-E（消融与几何敏感性）→ 最后 §VI（limitation）。
  - 如果只想快速判断是否相关：读摘要 + Table II/III + §VI 即可。

- **不值得精读的理由**：
  - 只做 manipulation、不关心导航的读者：任务域差异大，读摘要即可。
  - 已经熟悉 flow matching + ESDF 规划、只想知道结论的读者：看 Table II/III 和 §VI 就够。
  - 不做机器人学习、只想看多模态大模型架构的读者：VEGA 的架构是 π0.5 的适配，不是新的通用 VLM 架构。

---

[← Back to Theory](./README.md)

**关键引用**：VEGA 论文链接 https://arxiv.org/abs/2606.18426 ；全文 HTML https://arxiv.org/html/2606.18426 。本文只使用该论文链接，不提供额外代码仓库或未在论文中出现的 URL。