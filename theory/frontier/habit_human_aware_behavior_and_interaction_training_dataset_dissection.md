# HABIT：面向人机共存环境的机器人操作示范数据集（HABIT: Human-Aware Behavior and Interaction Training Dataset for Robot Manipulation）

> ⚙️ 本文由 Moltbot 自动生成 | 2026-11-12
>
> **论文**：HABIT: Human-Aware Behavior and Interaction Training Dataset for Robot Manipulation
> **链接**: https://arxiv.org/abs/2606.31682
> **核心定位**：把“人类在场”做成机器人示范数据的一根多样性轴——不是让策略在部署时突然学会与人共处，而是在采集阶段就用人类线索触发机器人动作，使让行、手势跟随、时空同步成为可学习信号。

> 版本说明：本文依据 arXiv:2606.31682v2 的 HTML 全文（标注 10 Nov 2026）写作。候选摘要片段在“three roles capt”处截断，正文摘要完整；两者未发现实质冲突。以下数字、设置与主张均取自该版本正文及附录，未用候选摘要补写或替代正文主张。论文未报告仿真评测，全部评测为真实双臂机器人设置。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 在人类在场示范上微调 VLA，能出现 Robot-only 数据学不到的 human-aware 行为；作者报告 π0.5 与 GR00T N1.6 在可比任务上均受益，Coworker 任务增益最大 |
| 适合精读 | 做 HRI/VLA 数据、双臂操作、共享工作空间安全的人；重点看 §2.3 采集协议、§3 评测框架、§4.2 主结果、§4.4 迁移实验 |
| 可以跳过 | 只关心单臂桌面抓取、纯策略架构创新、或不需要人类在场数据的人；这篇的贡献是数据集与评测协议，不是新网络 |
| 落地可行性 | 中：采集协议和三角色任务设计可复用，但单环境、约 10 名人类操作员、1 对 1 配置限制了外推 |
| 主要风险 | 无误差条/显著性检验；Collaborator 任务没有 Robot-only 基线；“人类在场”与“行为诱导协议”捆绑，难以拆分各自贡献 |

💡 **X-Ray 开场**

这篇论文解决什么问题？现有大规模机器人示范几乎都在无人环境采集，策略在空房间里很能干，一旦有人走进工作空间，就不知道让行、接物、看手势。它发现了什么？把人类放进采集现场还不够，必须让机器人的每个动作都由相机可见的人类线索触发，否则策略学到的是“背流程”而不是“看人反应”。对 VLA 研究者意味着什么？人类在场应被视为数据多样性的一根轴，和场景、物体、 embodi ment 并列，而不是部署时才补的 safety patch。

📍 **研究全景时间线**

```
2003 Scholtz 提出 HRI 角色/评估框架
  → 2021 Onnasch & Roesler 的 HRI 分类学
  → 2023 Pascher 等综述机器人运动意图沟通
  → 2024 DROID / Open X-Embodiment 等大规模机器人数据
  → 2025 AgiBot World、π0.5、GR00T N1 等双臂/通用 VLA
  → 2026 HABIT ← 当前位置：把“人类在场”系统性地加入示范数据
```

> 时间线只列论文参考文献中有明确年份的节点；不展开完整文献表。

## 1. 核心架构/方法总览

### 1.1 系统对比概览

HABIT 本身不是策略模型，而是一个数据资源 + 评测框架；被微调的两个 VLA 各有独立配置。下表逐项归属，避免把不同模型属性合并。

| 对象 | 角色 | 输入 | 输出/动作 | 训练/推理差异 | 记忆/历史机制 | 开放范围 |
|------|------|------|-----------|----------------|----------------|----------|
| HABIT 数据集 | 人类在场双臂示范 | 5 路 RGB、机器人运动学、子任务标注 | 不适用 | 不适用 | 不涉及策略记忆 | 论文称以 CC BY 4.0 发布，人脸模糊；实验只用 3 路 robot-side 相机 |
| π0.5 | 被微调 VLA | 3 路 RGB（front、left-wrist、right-wrist）+ 14-D 状态 | 14-D 动作（双臂各 7-D Cartesian delta），零填充到 32-D | 从 pi05_base 初始化；5000 steps、batch 128；10 步 action chunk | 论文未描述显式记忆模块；action chunk 是未来动作预测跨度，不是历史记忆窗口 | 开源 checkpoint；论文称提供 fine-tuning code，但候选记录无 repo URL，本文不写地址 |
| GR00T N1.6 | 被微调 VLA | 3 路 RGB + 14-D 状态 | 14-D 动作（双臂各 7-D Cartesian delta） | 从 GR00T-N1.6-3B 初始化；冻结视觉编码器，只调 top-4 LLM 层 + 多模态投影 + 扩散动作头；5000 steps、batch 128；16 步 action chunk | 论文未描述显式记忆模块；16 步 chunk 不等于记忆长度 | 开源 checkpoint；论文称提供 fine-tuning code |

补充：两者推理时都只执行每个预测 action chunk 的前 4 步，剩余步骤丢弃；论文称这是为了响应性和跨模型执行频率一致，但未报告具体 wall-clock 控制频率。相机记录为 640×480、10 Hz；不要把 10 Hz 当成策略推理频率。

### 1.2 关键机制

- **三角色任务结构**：Collaborator（人机共同完成同一任务，需直接物理配合）、Coworker（共享空间但各做各的任务，重点是避让）、Supervisor（人通过手势/示范指挥机器人）。每个角色 20 个任务，共 60 个任务。
- **Task workflow 作为有向图**：节点是子任务，标为 H_i 或 R_i；边是执行顺序。论文用 single-agent edge 与 cross-agent edge 的比例来量化人机耦合程度：Collaborator/Supervisor 以 cross-agent 边为主，Coworker 以 single-agent 边为主。
- **Reactive interaction**：采集时禁止操作员“背流程”提前行动；每个操作员必须直接观察伙伴行为后再动作，且禁止使用相机录不到的协调信号。这样每个示范动作都有策略可见的触发线索。
- **Behavior elicitation**：安全优先的让行（机器人遇到人/人持物靠近时主动收臂）、时间适配（人为变化人的移动速度）、手势 grounding（人指向前的等待时间从预定义区间采样，防止策略只靠语言指令抢跑）。
- **Additional diversification**：同一任务内变化衣服颜色、随机化物体操作顺序；跨任务覆盖多名体型不同的操作员。

⚡ **Eureka Moment**：不是“把人类放进画面”就够了，而是用 reactive interaction 把人类反应变成策略可观测的触发信号——让“看人下菜”成为唯一可学的因果路径。

### 1.3 信息流/架构图

```
人类操作员（手势/移动/递物）
        │
        ▼
5 路 RGB（实验只用 3 路 robot-side：front + 左右腕）
        │
        ▼
VLA（π0.5 或 GR00T N1.6）
        │  预测 action chunk（10 或 16 步）
        ▼
只执行前 4 步 → 机器人双臂动作
        │
        ▼
人类观察到机器人动作后反应 → 形成闭环
```

关键点：训练数据里的人类线索与部署时策略可见的输入必须同源；否则策略学到的只是任务顺序先验。

## 2. 数学核心

📌 **Napkin Formula**：

```
Success = (1/N) · Σ_i 1[manipulation ∧ workflow ∧ safety]
```

先给目标：一次 trial 只有在所有机器人子任务同时满足操作正确、流程合规、人类安全时才记为成功。再给公式与变量。

```
Success rate = (1/N) · Σ_{i=1}^{N} 1[τ_i succeeds]

1[A] = 1 if A true, else 0
```

| 符号 | 含义 |
|------|------|
| τ_i | 第 i 次独立 trial |
| N | 每个 (task, model, condition) 单元的 trial 数；本文 N=20 |
| manipulation | 子任务要求的物理操作完成，物体达到目标状态 |
| workflow | 满足任务 workflow 的结构条件：顺序、时空同步或线索跟随 |
| safety | 全程无人机碰撞、未碰到人持物 |

Workflow 的形式化（论文 §2.1）：

```
G = (V, E)
V = {H_1, ..., H_m} ∪ {R_1, ..., R_n}
E ⊆ V × V，边表示子任务执行先后
```

边分两类：single-agent（同代理子任务之间）与 cross-agent（人类子任务到机器人子任务）。论文用这两类边的比例判断角色耦合强度。

事件级指标（论文 §E.4，非主公式）：

```
synchronization lag = 人完成子任务 → 机器人到达下一子任务目标的时间
yield rate = 机器人 visibly retreats 的冲突事件数 / 冲突事件总数
gesture response time = 手势开始 → 机器人到达指示目标的时间
```

> 符号与本文一致；以上均为论文定义或正文直接给出的计算方式。

**采样跨度换算（一般性分析，不是论文实测）**：

```
若在历史观测中均匀取 K 帧，帧间隔 Δt：
  首末帧时间跨度 = (K - 1) × Δt

若从当前参考 t 预测未来 t+1 至 t+H，步长 Δt：
  预测终点距当前参考 = H × Δt
```

这两个式子只用于工程实现时区分“已观测历史帧”和“未来预测帧”。论文没有报告观测历史帧数，也没有把 action chunk 长度定义为记忆窗口；不能把 10 步或 16 步 chunk 直接当作历史上下文长度。

**直觉**：HABIT 的成功判据是合取，不是加权和。操作成功但抢跑（workflow 违规）或撞到人（safety 违规），整次 trial 仍失败。这使“能干”与“会共处”不能被互相抵消。

## 3. 带数字走一遍：玩具例子

> 以下数值为假设，用于验算公式，不是论文某任务的真实结果。

假设某 Supervisor 任务跑 N=20 次 trial：

- 16 次 manipulation 成功；
- 其中 13 次既在手势出现后行动（workflow 合规），又没有碰撞（safety 合规）；
- 另外 3 次虽然抓对了物体，但在手势出现前就动手，记为 gesture-following failure；
- 剩余 4 次是抓取滑落等 manipulation failure。

按公式：

```
Success rate = 13 / 20 = 0.65 = 65%
```

如果只看 manipulation，会得到 16/20 = 80%，高估 15 个百分点。这就是论文坚持三条判据合取的原因。

再看 mid-training 的样本效率（论文 §4.4 报告的数字，不是假设）：Waste Sorting 上，mid-training + 50 个下游示范匹配 direct fine-tuning 200 个示范；换算成样本比是 200/50 = 4×。Shelf Cleaning 上，direct fine-tuning 即使 200 个示范也未超过 45% 成功率，而 mid-training + 100 个示范达到 60%。这两个数字是作者报告值，说明 HABIT 作为先验可以替代一部分下游示范，但不能推断所有任务都按同一比例节省。

再按论文总量做一次换算（本文计算，非论文指标）：164.19 小时 / 10,563 episodes ≈ 55.96 秒/episode；10,563 / 60 tasks ≈ 176.05 episodes/task。评测子集每任务约 189–207 episodes，略高于全数据均值。

## 4. 工程视角

**训练侧**：两个模型都在单节点 8×H100 上微调，5000 steps、batch 128。π0.5 的 mid-training 改为 batch 256、16,500 steps，对应 mid-training 子集 2 epochs；子集由 41 个任务、每任务最多 100 个示范组成，排除 6 个评测任务。注意“5000 steps”是优化步数，不是 epoch 数；论文没有给主实验的 epoch 数。

**动作接口**：14-D 状态 = 每臂 3-D xyz + 3-D rotation + 1-D gripper；14-D 动作 = 每臂 7-D Cartesian delta。π0.5 把 14-D 动作零填充到 32-D 架构维度；GR00T 直接用 14-D。两者都吃 3 路 RGB，尽管数据集发布了 5 路。这意味着人类 ego-view 和外视角在实验中未被策略使用，论文把它列为未来工作。

**推理侧**：π0.5 预测 10 步、GR00T 预测 16 步，但都只执行前 4 步。工程含义是：chunk 越长，动作越平滑，但每次重规划间隔可能越长；只执行前 4 步是在响应性与利用完整 chunk 之间折中。论文没有给推理延迟、控制频率或每次 query 的 wall-clock 时间，因此不能从 chunk 长度反推实际 Hz。

**历史帧与未来帧**：论文没有报告观测历史窗口。若工程实现中取 K 个均匀历史帧、间隔 Δt，历史跨度是 (K-1)Δt；若从当前 t 预测未来 H 步、步长 Δt，预测终点是 t+HΔt。二者不能混用。连续未来 chunk 的预测区间不是历史上下文，也不是记忆窗口；论文中的 10/16 步只说明动作预测跨度。

**数据成本**：5 路相机、10 Hz、640×480，加上双臂运动学和子任务标注；论文没有给存储大小或标注人时。可确认的是采集使用 Meta Quest 3 控制器遥操作双 FR3，遥操作管线基于 DROID codebase；数据采集协议包含安全让行、变速、手势等待时间分箱等人工设计，这些都会增加采集复杂度。

## 5. 数据与评测

### 5.1 数据组成

| 项目 | 数值/设置 |
|------|-----------|
| episodes | 10,563 |
| 总时长 | 164.19 小时 |
| 任务数 | 60（三角色各 20） |
| 机器人子任务 / 人类子任务 / 子任务对 | 157 / 182 / 308 |
| 物体类型 | 101 类，12 大类；每任务平均 4.2 类 |
| 机器人 | 双臂 Franka Research 3 + Robotiq 夹爪 |
| 遥操作 | Meta Quest 3 控制器，基于 DROID 管线 |
| 相机 | 5 路 RGB，640×480，10 Hz；实验只用 3 路 robot-side |
| 人类操作员 | 约 10 名，单环境，1 对 1 |
| 许可 | 论文称 CC BY 4.0，人脸模糊 |

### 5.2 评测任务

| 角色 | 任务 | 关键挑战 | Robot-only 基线 |
|------|------|----------|------------------|
| Collaborator | Table Serving | 人靠近哪盘，机器人抬哪盘；时空同步 | N/A（任务本身需要人） |
| Collaborator | Shelf Cleaning | 人清一层，机器人擦一层；递/还掸子 | N/A |
| Coworker | Waste Sorting | 中等工作空间重叠，平均 2 次让行/ trial | 适用 |
| Coworker | Box Packaging | 高重叠，平均 3 次让行/ trial | 适用 |
| Supervisor | Donut Serving | 单阶段轨迹，指向选甜甜圈 | 适用（两条件都给 indexed language instruction） |
| Supervisor | Food Storage | 两阶段轨迹，指向选容器 | 适用（同上） |

每个 (task, model, condition) 单元 N=20 trials，由同一人类操作员按预定义协议执行。Robot-only 基线使用相同任务、环境、遥操作，唯一差别是无人共处；Coworker 的 Robot-only 只做机器人自己那部分，Supervisor 的两条件都带 indexed language instruction，HABIT 额外有指向手势。因此 Supervisor 的比较隔离的是“手势”而非“是否给出目标”。

### 5.3 主结果

论文 Figure 6 以柱状图报告成功率，正文只给方向性描述：π0.5 在 HABIT 上每个可比任务均优于 Robot-only，Coworker 增益最大；Donut Serving 两者相当，因为单阶段 + indexed instruction 已足够消歧；Food Storage 因两阶段分支点上的指向手势而受益。GR00T N1.6 呈同样模式，绝对值低于 π0.5，但 HABIT 相对 Robot-only 的收益一致。正文没有给出逐任务 Robot-only 的具体百分比，本文不补数字。

可引用的 HABIT 绝对值来自附录 Table 6（π0.5，原始训练数据 vs 人脸匿名化数据，20 trials）：

| 任务 | 角色 | 原始 | 匿名化 |
|------|------|------|--------|
| Table Serving | Collaborator | 95.0% | 95.0% |
| Shelf Cleaning | Collaborator | 45.0% | 50.0% |
| Waste Sorting | Coworker | 65.0% | 70.0% |
| Box Packaging | Coworker | 85.0% | 90.0% |
| Donut Serving | Supervisor | 95.0% | 95.0% |
| Food Storage | Supervisor | 70.0% | 75.0% |

这些是 π0.5 在 HABIT 上的成功率，不是“比 Robot-only 高多少”。人脸匿名化前后最多差一个 trial，说明脸部外观不是主要线索。

### 5.4 失败分析

论文把角色相关失败分为三类：Collaborator 的 precondition violation（人还没做完就抢下一步）、Coworker 的 collision（共享空间同时动作）、Supervisor 的 gesture-following failure（跟错手势目标）。另有与角色无关的 manipulation failure（抓滑、没打中）。作者报告 HABIT 训练显著降低三类角色相关失败，但 Figure 8 未给具体百分比，本文不编数值。

### 5.5 Mid-training 迁移

仅 π0.5 做了该实验；GR00T 未参与。mid-training 子集：41 个任务、每任务最多 100 个示范、排除 6 个评测任务、2 epochs、16,500 steps、batch 256。下游只测 Shelf Cleaning 和 Waste Sorting，各用 50/100/200 个示范微调。

| 任务 | 作者报告的关键结果 |
|------|--------------------|
| Waste Sorting | mid-training + 50 demos 匹配 direct 200；mid-training + 100 demos 超过所有 direct 预算 |
| Shelf Cleaning | direct 200 demos 仍未超过 45%；mid-training + 100 demos 达到 60% |

这支持“HABIT 可作为下游人机交互任务的先验”，但只覆盖两个任务、一个模型，不能外推到所有 HRI 任务。

### 5.6 OOD 评测（仅 π0.5 的汇总表）

| 分布偏移 | 范围 | In-distribution | OOD |
|----------|------|------------------|-----|
| 衣服颜色 | 6 任务平均 | 75.8% | 65.0% |
| 体型 | 6 任务平均 | 75.8% | 64.2% |
| 未见背景 | Waste Sorting | 65.0% | 65.0% |
| 未见背景 | Food Storage | 70.0% | 65.0% |
| 多人类 | Waste Sorting | 65.0% | 65.0% |

多人类设置是两人并排、轮流动作；46 次冲突事件中避免 44 次，两次碰撞都发生在两人同时移动时，而这一冲突模式不在 1 对 1 采集范围内。未见背景只换背景、不换桌面，因此只说明对背景外观不敏感，不说明能迁移到全新环境。

### 5.7 事件级指标

| 角色 | 指标 | 任务 | π0.5 | GR00T N1.6 |
|------|------|------|------|------------|
| Collaborator | sync lag ↓ | Shelf Cleaning | 2.47 ± 0.90 s (n=52) | 4.43 ± 3.19 s (n=31) |
| Collaborator | sync lag ↓ | Table Serving | 4.61 ± 1.80 s (n=79) | 7.56 ± 2.93 s (n=31) |
| Coworker | yield rate ↑ | Waste Sorting | 1.00 (n=43) | 0.89 (n=47) |
| Coworker | yield rate ↑ | Box Packaging | 1.00 (n=49) | 0.95 (n=63) |
| Supervisor | response time ↓ | Food Storage | 3.02 ± 0.62 s (n=15) | 6.37 ± 4.39 s (n=5) |
| Supervisor | response time ↓ | Donut Serving | 7.07 ± 4.65 s (n=19) | 9.89 ± 4.75 s (n=13) |

n 是事件数，不是 trial 数；不要把它和 N=20 混为一谈。

## 6. 能力与失败模式

### 6.1 能做什么

- **Collaborator**：在共享任务中按人的动作时机抬/放物体，表现为更短的同步滞后。
- **Coworker**：在共享空间里主动收臂让行；π0.5 在两组 Coworker 任务的事件级统计中让行率为 1.00，GR00T 为 0.89/0.95。
- **Supervisor**：把手势作为目标选择线索，尤其在两阶段 Food Storage 中受益；Donut Serving 因语言指令已足够，增益不明显。
- **快速适配**：π0.5 在 HABIT 上 mid-train 后，用更少下游示范达到更高成功率；但只验证了两个任务。

### 6.2 不能做什么 / 未验证

- **多人类同时移动**：只测了两人轮流、自然重叠的 Waste Sorting；两次碰撞都发生在同时移动时，作者明确说这超出当前数据范围。
- **全新环境**：未见背景只换背景不换桌面，不能证明迁移到全新房间。
- **显式记忆/长程交互**：论文没有提出记忆模块，也没有报告历史观测窗口；不能把 action chunk 当记忆。
- **统计显著性**：NeurIPS checklist 明确回答“不报告 error bars/置信区间”；N=20 trials 是固定协议，不是随机抽样方差估计。
- **Collaborator 的 Robot-only 对照**：任务本身需要人，N/A；因此 Collaborator 的增益无法用同任务无人基线直接对照。
- **GR00T 的 mid-training/OOD 汇总**：mid-training 只做 π0.5；OOD 汇总表也只汇总 π0.5。不能把 π0.5 的 OOD 数字归给 GR00T。

### 6.3 隐含假设

- **人类在场 + 行为诱导协议 = 可迁移的 human-aware 先验**。论文自己也说，增益应归因于“人类在场、行为诱导的采集协议整体”，而不是单独的人类在场。这意味着换一套采集协议，结果可能不同。
- **相机可见的线索足以支撑协调**。Reactive interaction 禁止录不到的协调信号；但如果真实部署中存在语音、触觉、共享计划等线索，当前数据没有覆盖。
- **同一人类操作员评测 in-distribution  trials 不引入系统性偏差**。论文用同一操作员、固定初始状态、预定义协议来降低方差，但也可能让策略适应这个人的动作风格。
- **动作 chunk 前 4 步执行足以代表部署频率**。论文没有给 wall-clock 频率，跨模型“一致执行频率”是设计目标而非实测结论。
- **单环境、约 10 名操作员、1 对 1 配置**。作者在 Limitations 中承认这会限制体型、动作风格、环境条件和多智能体动态的多样性。

## 7. 与相关工作对比

| 工作 | 关注点 | 人类是否在场 | 角色结构 | 数据/评测开放范围 |
|------|--------|--------------|----------|-------------------|
| DROID | 多场景、多任务真实操作数据 | 否（论文定位为常规无人示范） | 无 | 论文引用为公开数据集 |
| BridgeData V2 | 多场景、多任务 | 否 | 无 | 论文引用为公开数据集 |
| RH20T | 多模态、多任务 | 否 | 无 | 论文引用为公开数据集 |
| Open X-Embodiment | 跨 embodiment 聚合 | 否 | 无 | 论文引用为公开聚合 |
| AgiBot World | 双臂/人形规模化 | 否 | 无 | 论文引用为公开平台 |
| RoboCOIN | 双臂多平台 | 否 | 无 | 论文引用为公开数据 |
| HABIT | 人类在场、人机交互示范 | 是，每条 episode 都有人 | Collaborator / Coworker / Supervisor | 论文称 CC BY 4.0，含 5 路 RGB、运动学、子任务标注 |

> 表中其他数据集的规模数字未在本文引用，避免编造；只写论文对其定位的描述。

面试 Tip：被问到 HABIT 与“往数据里加人”的区别时，答“它用 reactive interaction 保证每个机器人动作都由相机可见的人类线索触发；评测又把 manipulation、workflow、safety 拆成合取判据，所以能把‘会做任务’和‘会与人共处’分开看”。

## 8. 精读建议

- **值得精读原文的人**：
  1. 做 HRI + VLA 数据采集的研究者，想复用三角色任务和 reactive interaction 协议；
  2. 做双臂操作、共享工作空间安全策略的工程师，关心让行、手势 grounding 的评测口径；
  3. 想评估“人类在场数据能否作为 mid-training 先验”的团队。
- **建议章节路径**：先读 §2.3（采集协议，决定数据为何可学）→ §3（成功判据与评测任务）→ §4.2（主结果方向）→ §4.4（迁移实验）→ 需要复现时再看 Appendix C.3 和 D 的超参表。§5 文献综述可快速扫过；Appendix C.3 的逐 trial 排列只在复现时精读。
- **不值得精读的理由**：如果你不做真实机器人、不关心人类在场数据，或者只想要新模型架构，这篇的距离较远；读摘要和 §1 即可。

---

[← Back to Theory](./README.md)

**关键引用**：论文链接 https://arxiv.org/abs/2606.31682（本文所有数字与主张均取自该版本正文及附录；未使用其他 URL）。