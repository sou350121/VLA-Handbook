# ABot-M0：动作流形学习的 VLA 基础模型 (ABot-M0: VLA Foundation Model with Action Manifold Learning)

> **发布时间**：2026-02-11（arXiv v1）  
> **论文题目**：ABot-M0: VLA Foundation Model for Robotic Manipulation with Action Manifold Learning  
> **团队**：AMAP CV Lab（Alibaba Group）  
> **核心定位**：用“数据治理 + 统一动作表征 + AML（动作流形学习）+ 模块化 3D 感知注入”把异构本体/异构数据源统一到同一条 VLA 训练路径，推动 One-Brain, Many-Forms 的可复现工程化落地。

ABot-M0 的亮点不只是单点架构，而是一套端到端的“异构数据→统一表征→通用策略”的系统方案：从 UniACT 数据治理，到动作空间的统一（EEF delta + rotation vector + pad-to-dual），再到将扩散式训练从“预测噪声”改成“预测干净动作序列”的 Action Manifold Learning（AML）。

## 0. 1 分钟版

- **UniACT-dataset**：整合 6 个公开数据集（OXE、OXE-AugE、AgiBot-Beta、RoboCOIN、RoboMind、Galaxea），形成 **600 万+ trajectories / 9500+ 小时 / 20+ embodiments** 的统一操作数据集，并进行清洗、格式统一与采样均衡（[arXiv](https://arxiv.org/abs/2602.11236)）。  
- **动作表征统一**：将动作统一为 **EEF 坐标系下的 delta actions**，旋转统一为 **rotation vector（轴角向量）**；单臂通过 **pad-to-dual** 零填充统一到双臂框架（[PDF](https://arxiv.org/pdf/2602.11236.pdf)）。  
- **AML（Action Manifold Learning）**：提出“动作流形假说”，用 **DiT** 直接预测“干净的连续动作序列”（a-prediction），把学习目标从“拟合噪声”改成“投影到可行动作流形”，提升解码速度与策略稳定性（[arXiv](https://arxiv.org/abs/2602.11236)）。  
- **模块化 3D 感知注入**：双流感知：VLM 语义（Qwen3-VL）+ 可插拔 3D 模块（VGGT / Qwen-Image-Edit）注入几何与多视角先验，无需改 backbone（[arXiv](https://arxiv.org/abs/2602.11236)）。  
- **结果（平均成功率）**：LIBERO **98.6**，LIBERO-Plus **80.5**，RoboCasa GR1 **58.3**，RoboTwin 2.0（randomized）**81.16**（[Project Page](https://amap-cvlab.github.io/ABot-Manipulation)）。  

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 维度 | 常见 VLA 预训练（简化概括） | ABot-M0 |
|---|---|---|
| 数据来源 | 单一平台/少本体；或格式不统一导致“数据孤岛” | UniACT：6 数据集统一治理，20+ 本体覆盖 |
| 动作空间 | joint / abs / delta 混杂；旋转表征不统一；单/双臂不兼容 | EEF delta + rotation vector；pad-to-dual 统一单/双臂 |
| 训练目标 | 扩散式多为 \(\epsilon\)-pred / \(v\)-pred（预测噪声） | AML：a-pred（预测干净动作序列） |
| 感知 | VLM 偏语义，3D/几何弱；多视角注入困难 | 双流 + 可插拔 3D 模块，注入几何/多视角先验 |
| 结论形态 | 论文往往只强调“模型结构”或“数据规模” | 明确主张“数据标准化/架构/训练目标”三者正交、可叠加增益 |

### 1.2 关键机制 (Key Mechanism)

1. **Data Harmony（数据治理）**：将多源数据统一到同一格式（论文将多源格式统一到 LeRobot v2），并处理无效指令、视觉异常、动作异常、频率错配等问题；论文称清洗后约 **16% trajectories 被丢弃**（[PDF](https://arxiv.org/pdf/2602.11236.pdf)）。  
2. **Action Standardization（动作空间统一）**：统一到“可跨本体复用”的 EEF delta + rotation vector，并明确双臂输出接口。  
3. **Action Manifold Learning（AML）**：用 DiT 直接预测干净动作，减少“噪声学习”的无结构负担。  
4. **Dual-stream Perception（双流感知）**：Qwen3-VL 提供语义与视觉表征；3D 模块作为 plug-in expert 注入空间结构与多视角信息。  
5. **Two-stage Training（两阶段）**：大规模统一预训练 + 任务 SFT，兼顾“泛化与精度”（[arXiv](https://arxiv.org/abs/2602.11236)）。  

### 1.3 信息流/架构图 (Flow / Diagram)

```text
UniACT (6 datasets) -> cleaning/standardization/balancing
  -> VLM(Qwen3-VL) + optional3D(VGGT/Qwen-Image-Edit)
  -> ActionExpert(DiT with AML)
  -> predict dual-arm EEF delta actions (pad-to-dual compatible)
  -> evaluate on LIBERO/LIBERO-Plus/RoboCasa/RoboTwin2
```

## 2. 数学核心：动作表征与 AML 的“可计算接口” (Math Core)

### 2.1 统一动作表征：EEF delta + rotation vector（论文 Sec.2.3）

论文将动作统一为末端执行器（EEF）坐标系下的增量动作（delta），并把旋转统一成 rotation vector（轴角向量）：

- rotation vector 定义：\(\mathbf{r}=\theta \mathbf{k}\)，其中 \(\theta\in[0,\pi]\)，\(\|\mathbf{k}\|=1\)，\(\mathbf{r}\in\mathbb{R}^3\)（[arXiv HTML](https://arxiv.org/html/2602.11236v1)）。  
- 每个手臂的 7D 动作为：\([\Delta x,\Delta y,\Delta z,\mathbf{r},gripper]\)。  
- 每个 timestep 输出左右臂各一份（因此是“双臂接口”）。  

### 2.2 pad-to-dual：把单臂统一到双臂接口（论文 Sec.2.3）

对于单臂轨迹，将未使用手臂通道置零（zero padding），并把所有单臂统一成“右臂执行”的双臂格式：

```text
dual_action = [left_7d, right_7d]
single_arm_right:
  left_7d = [0,0,0, 0,0,0, 0]
  right_7d = [Δx,Δy,Δz, rx,ry,rz, g]
```

工程含义：同一模型参数可同时覆盖单臂与双臂任务，并在语言指令约束下学习“何时用一只手 vs 双臂协同”。

### 2.3 AML：从“预测噪声”到“预测干净动作”

ABot-M0 的 AML 来自一个关键假设：有效动作不是散布在全高维空间，而是在物理与任务约束下形成低维、平滑的可行流形。  
因此训练目标从扩散式的 \(\epsilon\)-pred / \(v\)-pred（噪声）迁移到 **a-pred（干净动作序列）**，使学习更像“投影到可行流形”而非“去噪拟合”（[arXiv](https://arxiv.org/abs/2602.11236)）。

> 注：AML 的完整损失与采样细节建议直接以论文 Sec.3/6.3.1 为准（本文不复刻全部公式，避免误读）。

## 3. 带数字走一遍：为什么“统一动作接口”能解异构本体 (Worked Example)

假设你有两个数据源：

- 数据源 A：单臂，动作是 EEF delta + gripper  
- 数据源 B：双臂，动作是左右臂各自的 EEF delta + gripper  

在 pad-to-dual 之前，A 与 B 的动作维度不一致，模型要么做两套 head，要么把数据拆开训练。  
pad-to-dual 之后，A 被“提升”为双臂接口的一种特例（左臂永远 0），于是：

- 模型的输出空间一致  
- 训练 batch 可以混合采样  
- cross-embodiment 的共享发生在“同一语义动作接口”上（而不是关节角这种本体强绑定表征）

这就是 ABot-M0 所谓“让异构原始数据终于可以一起用”的关键工程点之一（[arXiv](https://arxiv.org/abs/2602.11236)）。

## 4. 工程视角：数据治理与训练范式的落地含义 (Engineering View)

### 4.1 Data Harmony 的最小验收点

从论文 Sec.2.2 可提炼出一套最小验收：

- **指令质量**：空指令、乱码、多语言混杂需清洗/翻译；否则模型退化为 VA（vision-action）而不是 VLA。  
- **视频质量**：黑帧、严重模糊、遮挡、无效视角需过滤。  
- **动作质量**：异常轨迹长度、连续大 delta（抖动）、动作频率与帧率严重错配、缺维/语义不明的 action 必须剔除。  
- **格式统一**：多源数据统一到同一存储与加载协议（论文选 LeRobot v2）。  

### 4.2 统一动作表征的 trade-off

- **优势**：EEF delta 是跨本体更稳的“公共接口”，rotation vector 规避欧拉角奇异性。  
- **风险**：不同机器人控制器对 EEF delta 的跟踪能力、频率与滤波可能不同；真实部署需要做控制频率与插值/平滑对齐（这往往决定 sim→real 的稳定性）。  

### 4.3 模块化 3D 感知：把 3D 当成“可插拔专家”

论文强调不改 backbone，而用 plug-in 3D 模块注入空间先验（VGGT / Qwen-Image-Edit + multi-view），工程上更像“perception expert 侧挂”，而非大规模重训 VLM。

## 5. 数据与评测 (Data & Eval)

### 5.1 UniACT 的组成与分布（论文 Table 1 / Fig.2）

- 6 个数据源：OXE、OXE-AugE、AgiBot-Beta、RoboCOIN、RoboMind、Galaxea（[arXiv HTML](https://arxiv.org/html/2602.11236v1)）。  
- 规模：600 万+ trajectories，9500+ 小时，20+ embodiments（[arXiv](https://arxiv.org/abs/2602.11236)）。  
- 分布：论文提到 OXE-AugE 单臂数据约占 67%，OXE 次之；其余四个双臂数据合计约 17.2%，因此训练中需要均衡采样（[arXiv HTML](https://arxiv.org/html/2602.11236v1)）。  

### 5.2 关键结果（Project Page 汇总）

- **LIBERO**：平均 **98.6%**（并给出四套 suite：Spatial/Object/Goal/Long 的分项表）（[Project Page](https://amap-cvlab.github.io/ABot-Manipulation)）。  
- **LIBERO-Plus**（zero-shot，训练仅用 LIBERO）：总分 **80.5%**（[Project Page](https://amap-cvlab.github.io/ABot-Manipulation)）。  
- **RoboCasa GR1 Tabletop（24 tasks）**：平均 **58.3%**（[Project Page](https://amap-cvlab.github.io/ABot-Manipulation)）。  
- **RoboTwin 2.0（50+ tasks）**：randomized 平均 **81.16%**（clean 80.42%）（[Project Page](https://amap-cvlab.github.io/ABot-Manipulation)）。  

## 6. 能力与失败模式 (Capabilities & Failure Modes)

### 6.1 能力画像（从设计出发的可预期收益）

- **跨本体混训**：统一 EEF delta 接口 + pad-to-dual，使“单/双臂、不同机械臂/平台”可在同一策略头下学习。  
- **长序列稳定性**：AML 以“干净动作序列”为目标，理论上更利于长序列动作平滑与减少抖动。  
- **高精度空间任务**：模块化 3D 感知注入，目标是补齐纯 VLM 的 3D 推理短板。  

### 6.2 失败模式（面试常问）

- **数据偏置**：大规模数据里某些数据源占比过高会导致 embodiment bias；论文通过降低某些数据源采样比例与均衡采样缓解，但真实效果依赖实现细节。  
- **动作接口≠控制接口**：EEF delta 在不同控制器上的“可执行性”不一致，真实系统需要做频率、滤波、饱和与安全约束对齐。  
- **3D 模块的可靠性**：plug-in 模块引入新的 failure mode（例如单目 3D 误差、多视角合成伪影），需要消融与回退策略。  

## 7. 与相关工作对比 (Comparison)

| 工作 | 核心差异点 | 与 ABot-M0 的关系 |
|---|---|---|
| OpenVLA / OpenVLA-OFT | 更偏 VLM+action head 与后训练策略 | ABot-M0 强调“数据治理 + 统一动作接口 + AML”一体化 |
| π0 / π0.5 | Flow/FAST 等更快动作生成范式 | ABot-M0 的 AML 是另一条“直接预测干净动作”的范式 |
| GR00T-N1.6 | 双系统与仿真数据管线 | ABot-M0 也强调数据与模块化感知，benchmark 侧做系统对比 |
| X-VLA | 强基线（LIBERO 表现很高） | Project Page 显示 ABot-M0 在 LIBERO 平均 SR 上更高（需注意实验配置一致性） |

**面试 Tip**：一句话回答“ABot-M0 的贡献”——**把异构公开数据做成可混训的统一接口（EEF delta + rotation vector + pad-to-dual），并用 AML 把动作生成从“预测噪声”改为“预测干净动作序列”，再用模块化 3D plug-in 补齐 VLM 的空间短板，从而支撑跨本体泛化。**

## References

- arXiv：`https://arxiv.org/abs/2602.11236`  
- PDF：`https://arxiv.org/pdf/2602.11236.pdf`  
- 项目页：`https://amap-cvlab.github.io/ABot-Manipulation`  
- 代码：`https://github.com/amap-cvlab/ABot-Manipulation`  

---
[← Back to Theory](../README.md)

