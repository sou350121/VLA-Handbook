# PGR：用条件扩散“生成式回放”替代 PER 的稀有样本过拟合 (Prioritized Generative Replay, ICLR 2025)

> **发布时间**：2024 arXiv（`arXiv:2410.18082`），ICLR 2025（Oral）  
> **论文题目**：Prioritized Generative Replay  
> **核心定位**：把 replay buffer 变成一个**可条件生成的参数化记忆**：用扩散模型从真实在线经验中学习 \(p(\tau)\)，再用“相关性函数” \( \mathcal{F}(\tau)\) 做条件/引导生成，把 synthetic replay **推向更有学习价值且更“新颖”的区域**。相比传统 **PER（prioritized experience replay）**，PGR 不只是“多抽稀有样本”，而是 **densify + guide**：在稀有但关键的 transition 周围生成更多“邻域样本”，并用 curiosity 等信号避免对少量高优先级样本过拟合。

**核心来源**：
- 论文（arXiv HTML v2）：`https://arxiv.org/html/2410.18082v2`
- 论文（arXiv）：`https://arxiv.org/abs/2410.18082`
- 项目页：`https://pgenreplay.github.io/`
- 代码：`https://github.com/renwang435/pgr`

---

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 方法 | “记忆”是什么 | 采样/训练数据分布 | 主要问题 | PGR 相比之下 |
|---|---|---|---|---|
| **Uniform replay** | 有限 replay buffer | 与在线访问分布近似一致 | 关键样本稀有→学习慢 | PGR 通过生成把关键区域 densify |
| **PER（优先经验回放）** | 有限 replay buffer + priority | 更偏向 TD-error 等高优先样本 | **稀有样本被反复抽**→更易过拟合 | PGR 用生成做“邻域扩增”，并用 curiosity 维持多样性 |
| **SynthER（无条件生成回放）** | 扩散模型拟合 \(p(\tau)\) | 仍接近原分布（无引导） | 在稀疏奖励/难任务上会失败或不如 model-free | PGR 关键在 **conditioning/guidance**，不是“更好画质” |
| **PGR（本文）** | **条件扩散模型** \(G(\tau\mid \mathcal{F}(\tau))\) | 朝“高相关性/高新颖性”区域偏移 | 需要选择 \( \mathcal{F}\) 与 guidance scale | 实证上 curiosity 是强默认项，并能支撑更高 UTD |

### 1.2 关键机制 (Key Mechanism)

PGR 的两个关键词：

- **Densification**：把有限 replay buffer 变成“无限”参数化 buffer（扩散模型可采样无限 transition）
- **Guidance**：用相关性函数 \( \mathcal{F}(\tau)\) 引导生成，让采样分布偏向更“有用”的 transition 子空间

这里的“有用”不是靠 reward 贪婪筛选（论文明确指出 reward-conditioning 反而更差），而是更偏向：
- TD-error / value frontier
- **curiosity（intrinsic novelty）**：既相关又不容易塌缩到少量样本

### 1.3 信息流/架构图 (Flow / Diagram)

```text
Outer loop (online RL):
  policy π interacts -> collect real transitions τ_real -> D_real
                 │
                 ├─ update relevance function F(τ)  (e.g., curiosity / TD error / return)
                 │
  Inner loop (periodic, every ~10K iters):
      train conditional diffusion G(τ | F(τ)) on D_real
      sample τ_syn from G with classifier-free guidance -> D_syn
      train policy π on mix(D_real, D_syn) with synthetic ratio r
```

---

## 2. 数学核心：PGR 如何把“优先回放”转写为“条件生成”？(Math Core)

> 符号：\(\tau=(s,a,s',r)\) 表示 transition；\(\mathcal{F}(\tau)\) 是相关性函数（标量或低维条件）。

### 2.1 相关性函数：把“priority”变成 conditioning signal

论文在 §4.2 给出 3 类常用 \(\mathcal{F}\)（都很便宜、适合在线）：

**(1) Return / value-based**

\[
\mathcal{F}(s,a,s',r) = Q(s,\pi(s))
\]

优点：已有 Q 网络，成本低；缺点：高 return transition 多样性可能很低，容易过拟合。

**(2) TD-error（PER 的经典 priority）**

\[
\mathcal{F}(s,a,s',r)= r + \gamma Q_{\text{target}}(s', \arg\max_{a'} Q(s',a')) - Q(s,a)
\]

缺点：Q 对 OOD transition 的估计不可靠；贪婪强调高 TD-error 会带来 myopic/低质量估计。

**(3) Curiosity（本文最强默认项）**

基于 ICM（intrinsic curiosity module）的 forward-model prediction error：

\[
\mathcal{F}(s,a,s',r)=\frac{1}{2}\left\lVert g(h(s),a)-h(s')\right\rVert^2
\]

直觉：让生成偏向“模型还没学会”的转移边界，通常意味着**更高新颖性 + 更高多样性**，从而降低 synthetic data 的过拟合风险。

### 2.2 条件扩散 + classifier-free guidance（CFG）

他们用条件扩散模型生成 transition，并用 CFG 在采样时加大对条件的服从（论文 §3）。

训练时随机丢弃条件（概率 \(p_{\text{uncond}}\)）以支持 CFG：

\[
\mathbb{E}\left\lVert \epsilon_\theta\left(x^n,n,(1-p)\cdot y + p\cdot \emptyset\right)\right\rVert^2_2
\]

其中 \(y\) 就是条件（例如 \(y=\mathcal{F}(\tau)\)），\(\emptyset\) 是空条件。

采样时用 guidance scale \(\omega\) 混合 conditional/unconditional 预测：

\[
\hat{\epsilon}=\omega\cdot\epsilon_\theta(x^n,n,y) + (1-\omega)\cdot\epsilon_\theta(x^n,n,\emptyset)
\]

经验口径：\(\omega\) 越大，生成越“贴合”高相关性区域，但也更容易 mode collapse；curiosity 的优势是它天然更分散，不容易塌缩。

### 2.3 训练用数据是“真实+生成”的混合分布

PGR 最终训练 policy 用的是：

\[
\mathcal{D}_{real} \cup \mathcal{D}_{syn}
\]

并用一个 synthetic ratio \(r\in[0,1]\) 控制 batch 里 synthetic 的占比（论文 Algorithm 1）。

---

## 3. 带数字走一遍：为什么 PGR 不等于“更高质量生成”？(Worked Example)

论文在 §5.2 用一个很关键的证据反驳直觉误区：

- “条件生成更强”并不是因为生成更准确（transition fidelity 更高）
- 而是因为生成了“**更对学习有用**”的 transition 类型

他们用一个动力学一致性指标（对生成的 \((s,a,s',r)\) 再进 simulator rollout 得到 ground-truth \((\tilde{s}',\tilde{r})\)，算 MSE）比较：

- SynthER（无条件）
- PGR（curiosity 条件）

结论：两者 **生成质量相近**（MSE 很接近），但 PGR 的学习效果更好。面试可用一句话概括：

> PGR 的关键不是“画得更像”，而是“画到对的区域”。

---

## 4. 工程视角：PGR 的系统参数、成本与可扩展性 (Engineering View)

### 4.1 外层/内层循环：什么时候重训扩散模型？

论文给了一个很工程化的经验值（Appendix D）：**每 10K iterations** 重跑一次 inner loop（重训/刷新 diffusion buffer），在效果与训练耗时之间折中较好。

### 4.2 Buffer 大小与条件丢弃概率

他们的一个实现细节（论文 §4.3 / §5）：

- \(|\mathcal{D}_{real}| \approx 1M\) transitions
- \(|\mathcal{D}_{syn}| \approx 1M\) transitions（后续 scaling 会扩到 2M）
- 训练时以 **0.25** 概率随机丢弃条件（支持 CFG）

### 4.3 成本增量（对比 SynthER）

项目页/论文 Table 3 给的量级是：

- PGR 相对 SynthER **总训练时间增加 <5%**
- 生成 VRAM：6.67GB（相对 SynthER 4.31GB）

面试口径：这是一个“可插拔 replay 组件”，不是把系统复杂度翻倍的 world-model planner。

### 4.4 Scaling：PGR 能支撑更高 UTD（update-to-data ratio）

论文 §5.3 的核心结论是：PGR 的合成数据更“可用”，因此能比 SynthER 更可靠地提升 UTD：

- baseline：UTD=20
- PGR 在更大 policy 网络 + 更高 synthetic ratio 的组合下，可推到 **UTD=40**（SynthER 同设置会退化）

但也有边界：当 synthetic ratio 推到 0.875 时，两者都可能崩（论文 §5.3）。

---

## 5. 数据与评测 (Data & Eval)

### 5.1 基准：DMC-100K / Pixel-DMC-100K / OpenAI Gym

他们遵循 SynthER 的 online RL 评测套件（论文 §5），统一 **100K env steps**（个别 sparse 更难任务用 300K）。

### 5.2 关键结果（表 1 / 表 2）

从 Table 1（DMC-100k，5 seeds）可直接背三组强对比（越直观越好）：

- **Quadruped-Walk**：redq 496.75 → synther 727.01 → **pgr(curiosity) 927.98**
- **Cheetah-Run**：redq 606.86 → synther 729.35 → **pgr(curiosity) 817.36**
- **Reacher-Hard**：synther 838.60 → **pgr(TD error) 917.61**（curiosity 也很强：915.21）

Pixel-DMC（同表 1）里，pgr 也能超过 drq-v2 与 synther：

- **Walker-Walk（pixel）**：drq-v2 514.11，synther 468.53，**pgr(curiosity) 570.99**
- **Cheetah-Run（pixel）**：drq-v2 489.30，synther 465.09，**pgr(curiosity) 529.70**

OpenAI Gym（表 2，3 seeds）：

- **HalfCheetah-v2**：synther 8165.35 → **pgr(curiosity) 9234.61**

### 5.3 样本效率（可复述的“50K vs 100K”）

论文 §5.2 指出：在多个 state-based 任务上，curiosity-PGR 能在 **约 50K env steps** 达到 SynthER 在 100K 的性能水平（典型例子：cheetah-run、quadruped-walk）。

---

## 6. 能力与失败模式 (Capabilities & Failure Modes)

### 6.1 能力

- **比 PER 更强**：PGR 明确在同一 priority（TD-error/curiosity）下，仍优于 PER（论文 Figure 3a）
- **不是探索奖励的替代品**：给 baseline 加 curiosity bonus 仍不如 PGR（论文 Figure 3b / Appendix B）
- **更抗 overfit**：curiosity guidance 让 replay 更 diverse，dormant ratio（DR）更低、更稳定（论文 Figure 6）

### 6.2 失败模式 / 适用边界

- **reward-based conditioning 是反例**：贪婪 densify 高 reward transition 会更差（论文明确给出 Reward-pgr 退化）
- **过高 synthetic ratio 会崩**：0.875 附近无论 synther/pgr 都可能 catastrophic（论文 §5.3）
- **\(\mathcal{F}\) 设计要谨慎**：过于依赖 Q 的 \(\mathcal{F}\) 会把 Q 的偏差放大成生成偏差
- **算力与延迟**：虽然相比 world-model planner 轻很多，但依然需要周期性训练/生成（要有 GPU 预算）

---

## 7. 与相关工作对比 (Comparison)

| 方向 | 代表 | 与 PGR 的差异 |
|---|---|---|
| 优先经验回放 | PER | 只能重采样“已有 transition”，无法 densify；稀有样本重复抽取易过拟合 |
| 无条件生成回放 | SynthER | 有 densification，但缺 guidance，难任务可能不如 model-free |
| model-based RL | MBPO / Dreamer | 直接用 dynamics unroll/plan，耦合更强、更怕 model bias；PGR 是“更弱耦合”的 replay 增强 |

**面试 Tip（一句话）**：被问“PGR 相比 PER 的本质增量是什么？”——答：“PER 只是更频繁地抽稀有样本，容易对少量关键 transition 过拟合；PGR 把 replay 变成条件扩散生成器，在关键 transition 周围 densify 出大量邻域样本，并用 curiosity 这种更分散的 relevance signal 引导生成，从而同时提高样本效率和稳定性，还能支撑更高 UTD。”

---

[← Back to Theory](../README.md)

