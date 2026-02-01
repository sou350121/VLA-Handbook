# 具身智能算法面经：安克（Anker）All-in 具身智能（RL + MPC）(2025)

> **来源**：用户粘贴面经原文  
> **日期**：2025-12-21  
> **岗位方向**：具身智能 / 机器人 / 强化学习 / 规控（MPC）  
> **形式**：无 coding、无反问；“自我介绍+项目细节”后进入八股与规控

---

## 0. 面试流程概览

- **简历**：自我介绍 + 项目细节深挖
- **八股（RL）**：策略梯度、期望求导、方差问题、baseline 的无偏性与类型、抓取场景的 baseline 选择
- **规控（MPC）**：原理与 LQR/PID 区别、实时求解速度、求解失败兜底
- **coding**：无
- **反问**：无

---

## 1. RL 八股：问题清单 + 可复述答案

### 1.1 说一下策略梯度公式

最经典口径（REINFORCE）：

\[
\nabla_\theta J(\theta)=\nabla_\theta \mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t} r_t\right]
=\mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t}\nabla_\theta \log \pi_\theta(a_t\mid s_t)\,G_t\right]
\]

其中 \(G_t=\sum_{t'\ge t}\gamma^{t'-t}r_{t'}\)。

如果想更贴近 actor-critic：

\[
\nabla_\theta J(\theta)=\mathbb{E}\left[\nabla_\theta \log \pi_\theta(a_t\mid s_t)\,A^\pi(s_t,a_t)\right]
\]

### 1.2 为什么可以对“期望”求导？（期望与梯度交换）

标准答法：
- 满足一定正则条件（可积性/支配收敛一类条件）时，可以交换导数与积分/求和顺序；
- 在 RL 里常用 **log-derivative trick（score function trick）**：

\[
\nabla_\theta \mathbb{E}_{x\sim p_\theta}[f(x)]
=\nabla_\theta \int p_\theta(x) f(x)\,dx
=\int p_\theta(x)\nabla_\theta \log p_\theta(x)\,f(x)\,dx
=\mathbb{E}\left[f(x)\nabla_\theta \log p_\theta(x)\right]
\]

把 \(p_\theta(x)\) 换成轨迹分布 \(p_\theta(\tau)\) 就得到策略梯度推导。

### 1.3 如何解决方差大的问题？

面试常见要点（答 3–5 条就够）：
- **baseline**（值函数/状态值/优势函数）
- **actor-critic / GAE**：用优势估计替代 \(G_t\)，并用 GAE 平衡 bias-variance
- **reward normalization / advantage normalization**
- **更大 batch / 多并行 rollout**（减少估计噪声）
- **更稳的优化**：PPO clip、trust region、entropy regularization
- **减少时序相关性**：比如经验回放（对 off-policy）、或更强的采样策略（但要解释 off-policy 校正）

### 1.4 baseline 为什么能降低方差？引入后会不会产生偏执（bias）？

关键结论：**只要 baseline 不依赖 action，就不改变期望梯度（无偏），但能显著降方差。**

证明口径（写出一行就很加分）：

\[
\mathbb{E}_{a\sim\pi_\theta(\cdot\mid s)}\left[\nabla_\theta \log \pi_\theta(a\mid s)\,b(s)\right]
=b(s)\nabla_\theta \sum_a \pi_\theta(a\mid s)
=b(s)\nabla_\theta 1
=0
\]

因此

\[
\mathbb{E}\left[\nabla_\theta \log \pi_\theta(a\mid s)\,(G_t-b(s_t))\right]
=\mathbb{E}\left[\nabla_\theta \log \pi_\theta(a\mid s)\,G_t\right]
\]

**什么时候会引入 bias？**
- baseline 如果依赖 \(a\)（例如 \(b(s,a)\)）且没做正确校正，会改变期望；
- advantage/critic 估计是近似的，会带来 **估计误差**，但这通常被视为“可控 trade-off”，在 actor-critic/PPO 里是标准做法。

### 1.5 简单介绍一下 baseline 类型

按“baseline 用什么”分类，常见几类：
- **常数 baseline**：用回报均值/滑动均值（最简单）
- **state-value baseline**：\(V(s)\)，得到 advantage \(A=G-V(s)\)
- **Q baseline / advantage baseline**：直接学 \(A(s,a)\) 或 \(Q(s,a)\)
- **GAE**：优势的时序平滑估计（PPO 常用）

按“是否学习”分类：
- **handcrafted baseline**（规则/启发式）
- **learned baseline**（critic / value network）

### 1.6 机械臂抓取用什么 baseline？

更“工程口径”的答法：先明确你在说的是 **RL 训练稳定性** 的 baseline，而不是“抓取算法基线”。

- 如果是 **on-policy（PPO/TRPO）** 做抓取（通常 sparse reward、contact 复杂）：
  - 用 **\(V(s)\) critic** + **GAE**（最常见、最稳）
  - reward/adv normalization
  - 如果任务长时/稀疏，可提 curriculum、reward shaping（但注意区分 baseline）

- 如果是 **off-policy（SAC/TD3）**：
  - 本质上不叫 baseline，而是直接学 Q；但你可以说“用双 Q/target network/温度项”稳定估计

一句话版本（适合面试收尾）：
> “抓取这类接触丰富、奖励稀疏的问题，我会用 actor-critic 的 \(V(s)\) baseline + GAE（PPO 标配）来把方差压下来；如果是 off-policy 则用 Q/target 结构稳定训练。”

---

## 2. 规控 MPC：问题清单 + 可复述答案

### 2.1 说一下 MPC 原理，和 LQR / PID 有什么区别？

**MPC 核心**：
- 每个控制周期，基于当前状态 \(x_t\) 解一个有限时域优化：
  - 目标：最小化未来 \(H\) 步的代价
  - 约束：动力学约束 + 状态/控制约束（速度、力矩、碰撞、安全边界等）
- 只执行第一步控制 \(u_t\)，下一周期滚动重解（receding horizon）

对比口径：
- **PID**：误差反馈的固定结构控制，不显式优化未来，不自然处理复杂约束
- **LQR**：线性系统 + 二次代价的解析解（或高效 DP），通常假设无约束/弱约束
- **MPC**：显式处理约束、滚动优化；代价是需要实时求解器与算力预算

### 2.2 MPC 实时应用中如何保证求解速度？求解失败怎么办？

#### 2.2.1 求解速度（从工程到算法的三层回答）

面试里很常见的“三段式”：

- **(1) 问题形式化**：能线性化/凸化就线性化/凸化  
  - 非线性动力学 → 局部线性化（SQP / iLQR / sequential convex）
  - 把每步变成 QP（尤其是小型 QP）

- **(2) 求解器选择**（你原文给的两点很到位）
  - **qpOASES**：小型 QP、需要快速响应、active-set 思路常见
  - **OSQP**：稀疏/大规模 QP，ADMM 类方法，工程上好用（自动驾驶 MPC 常见）

- **(3) 实现层优化**
  - 稀疏矩阵表示与分解复用（warm start / factorization caching）
  - 并行化（多核）/ SIMD
  - 限制迭代次数、设定实时超时策略

#### 2.2.2 求解失败（fallback/容错）

标准工程答案：
- **备份控制器（backup controller）**：例如 PID/LQR、上一时刻最优解的延拓（hold-last / shift solution）、安全停止策略
- **可行性优先**：软约束（slack variables）、约束松弛策略，保证“有解”
- **安全边界**：超时/发散时切换到安全模式（限速/限力/停机）

一句话版本：
> “实时 MPC 关键是把问题尽量做成结构化 QP 并 warm-start，用合适求解器；失败就切换到备份控制器或安全策略，保证系统稳定与安全。”

---

## 3. 面试复盘：这场的“信号点”

- **RL**：考察你是否真的理解 policy gradient 推导、baseline 的无偏性与方差来源
- **MPC**：考察你能不能把“算法原理 → 实时工程 → 兜底策略”讲成闭环
- **风格**：没有 coding，说明更偏“基础理解 + 工程口径”筛选

---

[← Back to Question Bank](./README.md)

