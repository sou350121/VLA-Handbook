# Levine：未知动力学下的 Guided Policy Search（Learning Neural Network Policies with GPS under Unknown Dynamics, NeurIPS 2014）

> **发布时间**：2014（NeurIPS）  
> **论文题目**：Learning Neural Network Policies with Guided Policy Search under Unknown Dynamics  
> **核心定位**：用“**局部线性动力学拟合 + KL 约束的轨迹优化**”在未知/接触丰富动力学下学习局部线性高斯控制器，再用 GPS 把它蒸馏到神经网络策略；训练过程里**神经网络策略无需直接上机探索**，更安全、更样本高效。  

**核心来源**：
- NeurIPS PDF：`https://proceedings.neurips.cc/paper_files/paper/2014/file/6766aa2750c19aad2fa1b32f36ed4aee-Paper.pdf`

---

## 1. 核心架构/方法总览 (Overview / Architecture)

这篇更“算法底层”，但对具身非常关键：它回答了一个长期难题：

> 动力学未知且有接触不连续时，如何既样本高效，又能学到复杂策略？

核心拆法：
- 在当前数据附近拟合 **time-varying local linear dynamics**
- 用 KL 约束控制每次更新的幅度，保证线性模型仍然有效
- 得到一个稳定的 **time-varying linear-Gaussian controller**
- 用 GPS 把“局部控制器”蒸馏成“全局神经网络策略”

### 1.1 信息流/架构图 (Flow / Diagram)

```text
Rollouts from local controller p(u|x)
        │
        v
Fit local linear dynamics:  x_{t+1} ≈ A_t x_t + B_t u_t + noise
        │
        v
Trajectory optimization with KL constraint (trust region)
        │
        v
Update local controller p_t (linear-Gaussian)
        │
        v
Supervised learning: train πθ to match p_t
        └────────────── repeat ───────────────┘
```

---

## 2. 数学核心：它到底在优化什么？(Math Core)

如果你只记一件事：**没有 KL 约束（trust region），局部线性拟合会被下一次更新“带离有效区域”，训练直接崩。**

### 2.1 线性高斯轨迹分布（local controller）是什么？

论文在未知动力学下优化的是一个 time-varying linear-Gaussian controller：

```text
p(u_t | x_t) = N(K_t x_t + k_t, C_t)
```

并在每个时间步拟合 local linear-Gaussian dynamics：

```text
p(x_{t+1} | x_t, u_t) = N(f_x,t x_t + f_u,t u_t + f_c,t, F_t)
```

### 2.2 读细后必须会复述的递推式（iLQG / LQR 口径）

论文在预备知识部分给了经典递推（用下标表示导数）：

```text
Q_{xu,xu,t} = ℓ_{xu,xu,t} + f_{xu,t}^T V_{xx,t+1} f_{xu,t}
Q_{xu,t}    = ℓ_{xu,t}    + f_{xu,t}^T V_{x,t+1}

V_{xx,t} = Q_{xx,t} - Q_{ux,t}^T Q_{uu,t}^{-1} Q_{ux,t}
V_{x,t}  = Q_{x,t}  - Q_{ux,t}^T Q_{uu,t}^{-1} Q_{u,t}

K_t = -Q_{uu,t}^{-1} Q_{ux,t}
k_t = -Q_{uu,t}^{-1} Q_{u,t}
```

你在面试里要能把它讲成一句话：  
**“线性拟合 dynamics + 二次展开 cost → backward pass 得到 Q/V → 计算线性反馈 K 和开环项 k。”**

### 2.3 KL 约束怎么“进 DP”？（核心 trick）

论文把“每次更新步长”写成约束（概念式）：

```text
minimize   E_p[ cost(τ) ]
subject to D_KL( p(τ) || p_prev(τ) ) <= ε
```

它跟今天的 TRPO/PPO trust region、以及 MPC 的“滚动小步 + 稳定性约束”同源。

关键 trick 是：在“线性高斯 + 最大熵”的设定下，可以把 KL 约束等价成**修改 cost**再跑 DP：

```text
在 DP 里用 augmented cost:
  ℓ̃(x_t, u_t) = (1/η) ℓ(x_t, u_t)  -  log p_prev(u_t | x_t)
```

直觉：
- \(-\log p_{\text{prev}}\) 把新策略拉回旧策略附近（别一步跳出线性拟合有效域）
- 对偶变量 \(\eta\) 控制“拉回去”的力度（\(\eta\) 越大，步子越小）

### 2.4 接触不连续为什么还能学？

另一个关键点：对接触不连续，局部线性模型会“平均”两侧动态，把确定性不连续变成近似随机平滑——因此在局部优化上反而更可控（这也是它敢碰 contact-rich 的原因）。

---

## 3. 动力学拟合：GMM prior 是怎么省样本的？（读细才会知道）

如果只是每个时间步做一次线性回归，样本需求会随状态维度暴涨。论文用一个非常实用的 trick：

- 把多次迭代、所有 time step 的转移样本 \((x_t, u_t, x_{t+1})\) 汇总
- 拟合 **Gaussian Mixture Model (GMM)** 做“背景动力学分布”（粗糙的 piecewise linear 动态先验）
- 每个时间步的线性动力学拟合时，用 normal-inverse-Wishart 的方式把 GMM 当 prior（降低回归方差）

论文给的经验口径：GMM prior 能把每次迭代所需样本数降低 **4–8 倍**（尤其对接触丰富场景很关键）。

---

## 4. GPS 蒸馏：为什么说“神经网络不必上机探索”？

论文强调的安全性来自训练流程：

- 真正在系统上跑的是 \(p(u_t\mid x_t)\)（线性高斯控制器，行为稳定、可约束）
- 神经网络策略 \(\pi_\theta\) 只在后台做 supervised learning，拟合控制器产生的 state-action 对

这使得即使神经网络初始是随机的，也不会在真实系统上“乱试”。

---

## 5. 实验设置（你应该会说出它测了什么）

论文的实验用来证明两件事：
- 轨迹优化部分在未知动力学下比很多 prior methods 更样本高效
- 把它嵌进 GPS 后，能学到“参数更多的神经网络策略”，而传统 policy search 很难做到

任务包括：
- 2D / 3D peg insertion（接触不连续）
- octopus arm control（高维）
- planar swimming（连续控制）
- bipedal walking（欠驱动）

对比方法包括 REPS、RWR、CEM、PILCO，以及“已知模型的 iLQG”参考线。  
面试复述口径：**在接触不连续与高维情况下，这个“局部线性 + trust region”的混合方案更稳、更省样本。**

## 6. 工程视角：为什么它对具身/手术/高风险任务特别像“正确答案”？(Engineering View)

### 6.1 安全：训练期神经网络策略不必直接上机

面试里这句话很加分：
- “训练交互由线性高斯控制器完成，稳定性更可控；神经网络只是做 supervised distillation。”

对于高风险系统（手术/人形/高功率末端）：
这相当于把危险探索从“黑盒策略”移到了“可控 controller + trust region”里。

### 6.2 样本效率：把全局建模难题降维成局部拟合

不学一个全局动力学模型（太难），只学每个时间步附近的线性模型（更容易），再靠 KL 约束保证下一轮仍在“可拟合区域”。

### 6.3 与现代 VLA 的映射

你可以把它映射成今天的系统语言：
- **local controller**：servo / MPC / contact state machine（可控、可验证）
- **πθ**：VLM+action head / diffusion policy（可泛化）
- **KL trust region**：限制 policy update 幅度 / 限制分布漂移 / 安全 guardrail

---

## 7. 能力与失败模式 (Capabilities & Failure Modes)

### 7.1 能力
- 未知动力学下的样本高效学习
- 对接触不连续更鲁棒（至少在局部优化可行）
- 训练更安全（policy 不直接上机探索）

### 7.2 失败模式
- **局部线性拟合质量不足**：观测噪声大、状态维太高、或 rollout 覆盖太窄
- **ε 设太大**：一步跳太远，线性近似失效；设太小则学习慢
- **部分可观测问题**：需要额外处理（论文里讨论了训练时全观测、测试时部分观测的策略）

---

## 8. 面试 Tip：怎么用它回答“为什么不纯端到端 RL？”

> “Levine 2014 的 GPS under unknown dynamics 用 KL 约束做 trust region，让局部线性动力学拟合在每次迭代都保持有效；训练交互由稳定的线性高斯控制器完成，神经网络策略只做蒸馏，因此样本更高效也更安全。对高风险具身系统，这种‘可控老师 + 可泛化学生’比纯端到端探索更可验收。”

---

[← Back to Theory](../README.md)

