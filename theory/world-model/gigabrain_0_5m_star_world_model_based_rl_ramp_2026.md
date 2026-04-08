# GigaBrain-0.5M*：世界模型原生的 VLA 自我进化范式 (GigaBrain-0.5M*: World Model-Based RL for VLA)

> **发布时间**：2026-02-12（arXiv v1）  
> **论文题目**：GigaBrain-0.5M*: a VLA That Learns From World Model-Based Reinforcement Learning  
> **团队**：GigaBrain Team / GigaAI（新闻稿口径常称“极佳视界”，以论文/项目页署名为准）  
> **核心定位**：用 **世界模型预测的未来状态 + 价值** 作为条件，把 VLA 从“看当前就出动作”的反应式控制，升级成“看未来再出动作”的前瞻式决策；并通过 **RAMP + Human-in-the-Loop Rollout** 做持续自我进化，在长时程任务上实现更高鲁棒性。

这篇是“重点文章”的原因很直接：它把世界模型从“数据生成/评测辅助”推到“策略学习的原生条件信号”，并且给出了从公式到工程闭环（HILR）的一整套可复述范式。

## 0. 1 分钟版

- 主张：主流 VLA 多是 **myopic**（只看当前），导致长时程任务弱；而视频世界模型更擅长时空预测，因此把世界模型做成策略条件能显著补足“前瞻性”。（[arXiv](https://arxiv.org/abs/2602.12099)）  
- 方法：提出 **RAMP**（Reinforcement leArning via world Model-conditioned Policy），四阶段迭代闭环：世界模型预训练 → 条件化策略训练 → HILR 真机 rollout → 用筛选后的轨迹持续训练。（[arXiv HTML](https://arxiv.org/html/2602.12099v1)）  
- 关键点：相比 π*0.6 的 **RECAP** 只用稀疏 advantage（0/1），RAMP 额外引入世界模型预测的 **未来 latent state**，信息增益更高，并理论上证明“RECAP 是 RAMP 的特例”。（Eq.(4)，[arXiv HTML](https://arxiv.org/html/2602.12099v1)）  
- 数据：GigaBrain-0.5 预训练 **10,931 小时**（61% 由 GigaWorld 合成，39% 真机）。（[Project Page](https://gigabrain05m.github.io/)）  
- 结果：RAMP 相比 RECAP 在 Laundry Folding / Box Packing / Espresso Preparation 等任务上 **约 +30%**，并在项目页展示“长时程连续无失败”。（[arXiv](https://arxiv.org/abs/2602.12099)，[Project Page](https://gigabrain05m.github.io/)）  

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 维度 | 传统 VLA（简化概括） | RECAP（π*0.6） | RAMP（GigaBrain-0.5M*） |
|---|---|---|---|
| 决策条件 | 主要基于当前观测 \(o_t\) | 额外条件：稀疏 advantage（0/1） | 额外条件：世界模型预测的 **未来状态 latent** + 价值 |
| 信号密度 | 低（对未来缺显式建模） | 低（0/1） | 高（未来 latent + value） |
| 学习范式 | BC/IL 或小规模 RL | advantage-conditioned | world-model-conditioned RL |
| 长时程鲁棒性 | 易累积误差 | 有提升但信息受限 | 显著提升（论文与项目页口径） |

### 1.2 关键机制 (Key Mechanism)

1. **世界模型既预测未来也预测价值**：把 value 作为 latent frame 拼接到视觉 latent 中，无需改 DiT 架构。  
2. **策略显式条件化未来**：policy 不再只“对当前反应”，而是“对未来打分/对未来状态做条件生成”。  
3. **HILR 数据闭环**：用人类在环修正策略 rollout，清理干预边界的过渡伪影，形成可持续训练数据。  
4. **两种推理模式**：通过训练时随机 mask world model token（p=0.2），允许部署时选择“绕过世界模型的高频模式”或“启用世界模型的标准模式”。（[arXiv HTML](https://arxiv.org/html/2602.12099v1)）  

### 1.3 信息流/架构图 (Flow / Diagram)

```text
WorldModel Wϕ (predict future-state latents + value)
      │
      ├─ future latents z_future  ─┐
      └─ value v_t (-> advantage -> I) ─┤
                                       ▼
Policy πθ(a | o_t, z_future, I, language)
      │
      ├─ deploy -> HILR rollouts (autonomous + corrections)
      └─ continual training (world model + policy)
```

## 2. 数学核心：RAMP 如何把“未来”塞进策略条件 (Math Core)

### 2.1 RAMP 的 KL 正则 RL 视角（论文 Sec.3.2.1）

论文从 KL-regularized RL 出发，给出最优策略形式（省略常数）：

\[
\hat{\pi}(a|\mathbf{S}) \propto \pi_{\text{ref}}(a|\mathbf{S})\exp\left(\frac{A^{\pi_{\text{ref}}}(\mathbf{S},a)}{\beta}\right)
\]

关键做法是把优势项的指数形式，重写成“改进事件”条件分布的比值，并最终得到可训练的 NLL 目标（Eq.(3)）：

\[
\mathcal{L}(\theta)=\mathbb{E}_{D}\left[-\log\pi_{\theta}(a|\mathbf{o},\mathbf{z},l)-\alpha\log\pi_{\theta}(a|I,\mathbf{o},\mathbf{z}_{t},l)\right]
\]

其中 \(I=\mathds{1}[A(\mathbf{o},\mathbf{z},l,a)>\epsilon]\) 是二值改进信号。  
直觉：第二项把“偏好/改进”作为条件，让策略更像在学“更好的动作分布”。

来源：Eq.(2)(3)，[arXiv HTML](https://arxiv.org/html/2602.12099v1)。

### 2.2 为什么说“RECAP 是 RAMP 的特例”（论文 Eq.(4)）

论文给出 RECAP 与 RAMP 的关系：

\[
\pi_{RECAP}(a|\mathbf{o},I)=\int_{\mathbf{z}}\pi_{RAMP}(a|\mathbf{o},\mathbf{z},I)p(\mathbf{z}|\mathbf{o},I)d\mathbf{z}
\]

工程含义：  
- **RECAP** 等价于“把未来 latent 积分掉”，因此学到的是对各种未来的平均策略；  
- **RAMP** 直接条件化特定未来 latent，让策略从“猜未来”变成“对准未来”。  

来源：Eq.(4)，[arXiv HTML](https://arxiv.org/html/2602.12099v1)。

### 2.3 世界模型的状态拼接：把 value 当作 latent frame（论文 Sec.3.2.2）

论文构造 latent state：

\[
\mathbf{s}_{t}=\big[\mathbf{z}_{t}\,;\,\Psi(v_{t})\,;\,\Psi(\mathbf{p}_{t})\big]
\]

其中 \(\mathbf{z}_t\) 是未来观测编码后的视觉 latent，\(v_t\) 是标量价值，\(\mathbf{p}_t\) 是本体状态；\(\Psi\) 做空间铺平投影，使其形状对齐视觉 latent。  

来源：Eq.(6)，[arXiv HTML](https://arxiv.org/html/2602.12099v1)。

## 3. 带数字走一遍：价值预测曲线为什么“有用” (Worked Example)

项目页给了一个非常工程化的指标集合：  

- **WM-based (state+value)**：Kendall = 0.8018，MAE = 0.0621，速度 0.25s  
- **WM-based (value only)**：Kendall = 0.7288，MAE = 0.0838，速度 0.11s  
- **VLM-based**：延迟 0.32s/frame（A800），瓶颈在 SigLIP visual encoder  

来源：[Project Page](https://gigabrain05m.github.io/)。

如何把它理解成“策略真的会更稳”？  

把长时程任务拆成 3 个阶段：

1. **探索/对齐阶段**（例如叠衣服前半段反复调整姿态）：价值会有波动（因为未来状态分支多）。  
2. **进入可收敛子空间**（衣物摆正，进入稳定折叠）：价值应当单调上升。  
3. **外界扰动/卡住**（被异物挡住、抓取失稳）：价值骤降，成为强烈的“纠错触发信号”。  

这类“与物理进程对齐的价值演化”，就是 world model-conditioned policy 能把“反应式控制”升级为“带前瞻的纠错控制”的直觉基础。

## 4. 工程视角：四阶段闭环与 HILR 的关键细节 (Engineering View)

### 4.1 四阶段 RAMP（正确的顺序很重要）

> 注意：项目页中文描述的编号顺序有倒序排版，这里按论文的 Stage 1→4 顺序复述。

1. **World Model Pre-training**：用大规模操作数据训练世界模型，联合预测未来状态与价值；论文给出 reward 设定（终止成功 0，失败 -C_fail，其它 -1）。（Eq.(5)，[arXiv HTML](https://arxiv.org/html/2602.12099v1)）  
2. **Policy Training with Conditioning**：用世界模型预测的 \(z_{future}\) 与 value→advantage→\(I\) 作为条件微调策略，并做随机 mask（p=0.2）防止过度依赖世界模型。（[arXiv HTML](https://arxiv.org/html/2602.12099v1)）  
3. **HILR Data Collection**：部署策略真机 rollout，遇到失败由人类纠正；并清理干预边界过渡伪影，保证轨迹时序一致。（[arXiv HTML](https://arxiv.org/html/2602.12099v1)）  
4. **Continual Training**：用筛选后的 HILR 数据持续训练，并强调要同步更新世界模型以防 advantage collapse。（[arXiv HTML](https://arxiv.org/html/2602.12099v1)）  

### 4.2 两种推理模式：可选“高频”与“高前瞻”

论文明确了部署策略：

- **Optimistic control**：推理时固定 \(I=1\)。  
- **Efficient mode**：绕过世界模型（mask future latents），提高控制频率。  
- **Standard mode**：启用世界模型生成 future latents，给长时程任务更多前瞻信息。  

来源：[arXiv HTML](https://arxiv.org/html/2602.12099v1)。

## 5. 数据与评测 (Data & Eval)

### 5.1 预训练数据：10,931 小时（合成/真机配比）

项目页给出 GigaBrain-0.5 的预训练分布：

- 总计 **10,931 小时**  
- **61%（6,653 小时）**：GigaWorld 合成（新纹理/视角/物体配置等）  
- **39%（4,278 小时）**：真机数据  

来源：[Project Page](https://gigabrain05m.github.io/)。

### 5.2 公共榜单信号：RoboChallenge

项目页称截至 2026-02-09：

- 中间模型 **GigaBrain-0.1**：平均 51.67%  
- 对比 **π0.5**：42.67%（高 9 个百分点）  

来源：[Project Page](https://gigabrain05m.github.io/)。

### 5.3 对比基线：AWR / RECAP / RAMP

论文摘要与项目页都强调：RAMP 相比 RECAP 在 Laundry Folding / Box Packing / Espresso Preparation 上 **约 +30%**。（[arXiv](https://arxiv.org/abs/2602.12099)，[Project Page](https://gigabrain05m.github.io/)）

## 6. 能力与失败模式 (Capabilities & Failure Modes)

### 6.1 能力画像

- **长时程稳定执行**：项目页展示 Box Packing / Espresso / Laundry Folding 的 near-perfect success 与连续稳定运行（以视频为证据）。  
- **跨任务适应**：RAMP 的 conditioning（future latent + value）在多任务训练中带来更明显增益（项目页描述 5k→20k steps 持续扩大差距）。  

来源：[Project Page](https://gigabrain05m.github.io/)。

### 6.2 失败模式（工程上必须预案）

- **世界模型误导**：world model 的未来预测若偏离真实，会把策略“导向错误的未来”。因此“efficient mode”与随机 mask 训练是重要保险丝。  
- **HILR 边界伪影**：人工接管会引入分布突变，论文明确需要边界检测与清理，否则会把过渡伪影学进策略。  
- **优势塌缩**：如果只训练策略而不更新世界模型，优势可能坍缩到 0 附近，导致条件信号失效；论文强调联合训练。  

## 7. 与相关工作对比 (Comparison)

| 范式 | 代表 | 核心差异 | 你该记住的一句话 |
|---|---|---|---|
| advantage-conditioned RL for VLA | RECAP（π*0.6） | 条件信号稀疏（0/1） | “用稀疏偏好信号帮 VLA 后训练” |
| world model-conditioned RL for VLA | RAMP（本文） | 条件信号密集（future latent + value） | “把世界模型的未来预测变成策略的原生条件” |
| world model + IDM | DreamGen/ViDAR 等 | 先生成未来视频，再用 IDM 反推动作 | “把动作学习从策略端转移到 IDM 端” |
| world model as data engine | GigaWorld / Cosmos 等 | 合成数据为主 | “世界模型先当数据工厂” |

**面试 Tip**：一句话回答“GigaBrain-0.5M* 的新范式是什么？”——**RAMP：用世界模型预测的未来状态 latent 与价值作为条件信号来训练/后训练 VLA，并配合 HILR 构成行动-纠错-再训练的自我进化闭环；RECAP 是把未来 latent 积分掉的特例。**

## References

- arXiv：`https://arxiv.org/abs/2602.12099`  
- arXiv HTML（含公式与 RAMP 实现细节）：`https://arxiv.org/html/2602.12099v1`  
- 项目页（含视频、数据配比、价值预测指标与基线对比）：`https://gigabrain05m.github.io/`  

---
[← Back to Theory](./README.md)

