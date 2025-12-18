# 强化学习 (Reinforcement Learning)

> **核心概念**: 强化学习 (Reinforcement Learning, RL) 是一种通过与环境交互来学习最优策略的方法。智能体通过**试错** (Trial-and-Error) 和**奖励反馈** (Reward Feedback) 不断改进行为。

## 1. 为什么 VLA 需要强化学习? (Why RL for VLA?)

### 1.1 行为克隆 (BC) 的局限

传统 VLA 主要依赖 **行为克隆 (Behavior Cloning)**：模仿人类演示。

| 方法 | 优点 | 缺点 |
| :--- | :--- | :--- |
| **BC** | 简单，数据效率高 | 上限是人类水平，无法超越演示者 |
| **RL** | 可以超越人类，自我进化 | 需要大量交互，稀疏奖励难优化 |

### 1.2 RL 在 VLA 中的价值

- **超越人类示教**: 通过自我博弈/探索发现更优策略
- **长序列优化**: BC 只模仿每步动作，RL 优化整个轨迹的累积回报
- **适应性学习**: 在真机部署时持续自我改进
- **稀疏奖励任务**: 只有任务成功时给奖励的场景（如组装）

### 1.3 VLA 中的 RL 应用案例

- **π*0.6 (Pi-Star)**: 使用 Recap 算法（Offline RL）超越人类示教
- **RT-2**: 使用 RL from Human Feedback (RLHF) 改进语义推理
- **RoboCasa**: 使用 PPO 训练家庭操作策略

## 2. RL 基础概念 (RL Fundamentals)

### 2.1 马尔可夫决策过程 (MDP)

```
MDP = (S, A, P, R, γ)
```

- **S**: 状态空间 (State Space) - 机器人观测到的一切
- **A**: 动作空间 (Action Space) - 可执行的动作
- **P(s'|s,a)**: 状态转移概率 - 环境动力学
- **R(s,a)**: 奖励函数 - 行为好坏的反馈
- **γ**: 折扣因子 - 未来奖励的权重 (通常 0.99)

### 2.2 马尔可夫性 (Markov Property)

**定义**: 下一状态只依赖于当前状态和动作，与历史无关。

```
P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ...) = P(s_{t+1} | s_t, a_t)
```

```
┌─────────────────────────────────────────────────────────────────┐
│                    马尔可夫性图解                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   非马尔可夫:  s₀ → s₁ → s₂ → s₃ → s₄                           │
│               └────────────────────┘                            │
│                  所有历史都影响未来                               │
│                                                                 │
│   马尔可夫:    s₀ → s₁ → s₂ → s₃ → s₄                           │
│                          └───┘                                  │
│                    只有 s₃ 影响 s₄                               │
│                                                                 │
│   💡 "当前状态是对历史的充分统计量"                              │
└─────────────────────────────────────────────────────────────────┘
```

**为什么重要**:
- **简化计算**: 不需要记忆整个历史，只需当前状态
- **Bellman 方程成立**: 价值函数可以递归定义
- **实际应用**: 机器人状态通常包含位置+速度，满足马尔可夫性；若只有位置则不满足

### 2.3 核心目标

最大化**累积折扣回报 (Cumulative Discounted Return)**:

```
G_t = Σ_{k=0}^{∞} γ^k × R_{t+k+1}
```

### 2.4 价值函数 (Value Functions)

**状态价值函数 (State Value)**:

```
V^π(s) = E_π[ G_t | S_t = s ]
```

**动作价值函数 (Action Value / Q-Function)**:

```
Q^π(s, a) = E_π[ G_t | S_t = s, A_t = a ]
```

**Bellman 方程**:

```
Q^π(s, a) = R(s, a) + γ × E_{s' ~ P}[ V^π(s') ]
```

### 2.5 最优价值函数与最优策略

**最优价值函数**:

```
V*(s) = max_π V^π(s)
Q*(s, a) = max_π Q^π(s, a)
```

**最优策略**:

```
π*(s) = argmax_a Q*(s, a)
```

**为什么最优价值函数就是最优策略？**

```
┌─────────────────────────────────────────────────────────────────┐
│                最优价值函数 ↔ 最优策略                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   核心定理: 给定 Q*(s, a)，最优策略可直接导出                    │
│                                                                 │
│   π*(s) = argmax_a Q*(s, a)                                     │
│                                                                 │
│   证明思路:                                                     │
│   1. Q*(s,a) 表示在状态 s 执行动作 a 后，按最优策略行动的期望回报 │
│   2. 要最大化回报，只需在每个状态选择 Q* 最大的动作              │
│   3. 这正是贪心策略，而对于 Q* 贪心就是最优的                    │
│                                                                 │
│   反过来:                                                       │
│   给定最优策略 π*，可以计算出 Q*(s,a) = Q^{π*}(s,a)             │
│                                                                 │
│   💡 最优价值函数和最优策略是"一体两面"                          │
└─────────────────────────────────────────────────────────────────┘
```

**数学推导**:
1. **Bellman 最优方程**: `V*(s) = max_a [ R(s,a) + γ × Σ_{s'} P(s'|s,a) × V*(s') ]`
2. 如果我们知道 `V*`，则最优动作是使上式取最大值的 `a`
3. 这等价于 `π*(s) = argmax_a Q*(s,a)`

### 2.6 策略 (Policy)

策略 `π(a|s)` 定义了在状态 `s` 下采取动作 `a` 的概率。

- **确定性策略**: `a = π(s)`
- **随机策略**: `a ~ π(·|s)`

## 3. RL 算法分类 (RL Algorithm Taxonomy)

```
                        RL 算法
                           │
           ┌───────────────┼───────────────┐
           │               │               │
      Model-Free      Model-Based    Offline RL
           │               │               │
     ┌─────┴─────┐         │          ┌────┴────┐
     │           │         │          │         │
 Value-Based  Policy-Based │       CQL/IQL   Recap
     │           │         │
   DQN/SAC   PPO/TRPO   Dreamer/MBPO
```

### 3.1 Model-Free vs Model-Based

```
┌─────────────────────────────────────────────────────────────────┐
│                Model-Free vs Model-Based                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Model-Free (无模型):                                          │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Agent ←──── 交互 ────→ Environment                     │   │
│   │    │                         │                          │   │
│   │    └── 直接学习 π 或 Q ──────┘                          │   │
│   │        (不关心环境如何工作)                              │   │
│   └─────────────────────────────────────────────────────────┘   │
│   代表: DQN, PPO, SAC                                           │
│   特点: 简单直接，但需要大量交互数据                            │
│                                                                 │
│   Model-Based (基于模型):                                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Agent ←──── 交互 ────→ Environment                     │   │
│   │    │                         │                          │   │
│   │    ├── 学习环境模型 P(s'|s,a) ┘                         │   │
│   │    │                                                    │   │
│   │    └── 在模型中规划/模拟 ──→ 策略                       │   │
│   └─────────────────────────────────────────────────────────┘   │
│   代表: Dreamer, MBPO, MuZero                                   │
│   特点: 样本高效，但模型误差会累积                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

| 类型 | 原理 | 优点 | 缺点 | 适用场景 |
| :--- | :--- | :--- | :--- | :--- |
| **Model-Free** | 直接学习策略/价值 | 简单，无模型偏差 | 样本效率低 | 真机交互成本低 |
| **Model-Based** | 学习环境动力学 $P(s'|s,a)$ | 样本效率高 | 模型误差累积 | 仿真环境、Sim-to-Real |

**关键区别**:
- **Model-Free**: 不尝试理解环境如何工作，只关心"什么动作能获得高回报"
- **Model-Based**: 先学习环境的"规则"（状态转移），再利用规则进行规划

**VLA 中的应用**:
- **Model-Free**: π0.6 的 Recap 算法（直接从数据学习策略）
- **Model-Based**: 世界模型 (World Model) 用于预测未来状态，辅助规划

### 3.2 策略迭代 vs 值迭代 (Policy Iteration vs Value Iteration)

两种经典的动态规划算法，用于求解 MDP。

```
┌─────────────────────────────────────────────────────────────────┐
│              策略迭代 vs 值迭代                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   策略迭代 (Policy Iteration):                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  初始化 π₀                                               │   │
│   │      │                                                   │   │
│   │      ▼                                                   │   │
│   │  ┌──────────────────┐                                    │   │
│   │  │ 策略评估 (PE)    │ ← 计算 V^π (迭代至收敛)            │   │
│   │  │ V^π(s) = E[...]  │                                    │   │
│   │  └────────┬─────────┘                                    │   │
│   │           │                                              │   │
│   │           ▼                                              │   │
│   │  ┌──────────────────┐                                    │   │
│   │  │ 策略改进 (PI)    │ ← π(s) = argmax_a Q(s,a)           │   │
│   │  │ π ← greedy(V)    │                                    │   │
│   │  └────────┬─────────┘                                    │   │
│   │           │                                              │   │
│   │           └──── 重复直到 π 不再变化 ────┘                │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   值迭代 (Value Iteration):                                     │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  初始化 V₀                                               │   │
│   │      │                                                   │   │
│   │      ▼                                                   │   │
│   │  ┌──────────────────────────────────────────────────┐   │   │
│   │  │ V(s) ← max_a [R(s,a) + γ Σ P(s'|s,a) V(s')]      │   │   │
│   │  │       (Bellman 最优方程的迭代更新)                │   │   │
│   │  └──────────────────────────────────────────────────┘   │   │
│   │           │                                              │   │
│   │           └──── 重复直到 V 收敛 ────┘                    │   │
│   │                                                          │   │
│   │  最后: π(s) = argmax_a Q(s,a)                            │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

| 对比 | 策略迭代 | 值迭代 |
| :--- | :--- | :--- |
| **核心操作** | 评估 + 改进交替 | 直接迭代 Bellman 最优方程 |
| **每轮计算** | 策略评估需迭代至收敛 | 只做一次 Bellman 更新 |
| **收敛速度** | 迭代次数少 | 每轮计算量小 |
| **总体效率** | 大状态空间更快 | 小状态空间更快 |
| **策略输出** | 每轮都有显式策略 | 最后才提取策略 |

**直觉理解**:
- **策略迭代**: "先完整评估当前策略有多好，再改进"
- **值迭代**: "直接朝最优价值函数迭代，最后再提取策略"

### 3.3 On-Policy vs Off-Policy

| 类型 | 代表算法 | 特点 |
| :--- | :--- | :--- |
| **On-Policy** | PPO, TRPO | 只用当前策略的数据，稳定但低效 |
| **Off-Policy** | SAC, TD3 | 可复用历史数据，高效但不稳定 |

## 4. VLA 常用 RL 算法 (RL Algorithms for VLA)

### 4.1 PPO (Proximal Policy Optimization)

**核心思想**: 限制策略更新幅度，防止训练崩溃。

**目标函数**:

```
L_CLIP(θ) = E_t[ min( r_t(θ) × Â_t, clip(r_t(θ), 1-ε, 1+ε) × Â_t ) ]
```

其中 `r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)` 是重要性采样比率。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PPO:
    def __init__(self, policy, value_net, clip_epsilon=0.2, lr=3e-4):
        self.policy = policy
        self.value_net = value_net
        self.clip_epsilon = clip_epsilon
        self.optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(value_net.parameters()), lr=lr
        )
    
    def compute_gae(self, rewards, values, dones, gamma=0.99, lam=0.95):
        """Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        return torch.tensor(advantages)
    
    def update(self, states, actions, old_log_probs, rewards, dones):
        # 计算价值和优势
        values = self.value_net(states).squeeze()
        advantages = self.compute_gae(rewards, values.detach(), dones)
        returns = advantages + values.detach()
        
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO 更新
        for _ in range(10):  # 多轮更新
            # 计算新的 log_prob
            dist = self.policy(states)
            new_log_probs = dist.log_prob(actions)
            
            # 重要性采样比率
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Clipped 目标
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 价值损失
            value_loss = F.mse_loss(self.value_net(states).squeeze(), returns)
            
            # 熵正则化 (鼓励探索)
            entropy = dist.entropy().mean()
            
            # 总损失
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
```

### 4.2 SAC (Soft Actor-Critic)

**核心思想**: 最大化奖励的同时最大化策略熵（鼓励探索）。

**目标**:

```
J(π) = E_{τ ~ π}[ Σ_t R(s_t, a_t) + α × H(π(·|s_t)) ]
```

```python
class SAC:
    def __init__(self, actor, critic1, critic2, target_critic1, target_critic2, 
                 alpha=0.2, gamma=0.99, tau=0.005):
        self.actor = actor
        self.critic1 = critic1
        self.critic2 = critic2
        self.target_critic1 = target_critic1
        self.target_critic2 = target_critic2
        self.alpha = alpha  # 熵系数
        self.gamma = gamma
        self.tau = tau
    
    def update(self, batch):
        states, actions, rewards, next_states, dones = batch
        
        # ===== Critic 更新 =====
        with torch.no_grad():
            # 从当前策略采样下一个动作
            next_actions, next_log_probs = self.actor.sample(next_states)
            
            # 目标 Q 值 (取两个 critic 的最小值)
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            
            # TD 目标
            target = rewards + self.gamma * (1 - dones) * target_q
        
        # 当前 Q 值
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)
        
        # ===== Actor 更新 =====
        new_actions, log_probs = self.actor.sample(states)
        q1 = self.critic1(states, new_actions)
        q2 = self.critic2(states, new_actions)
        q = torch.min(q1, q2)
        
        # 最大化 Q - α * log_prob
        actor_loss = (self.alpha * log_probs - q).mean()
        
        # ===== 软更新目标网络 =====
        self.soft_update(self.critic1, self.target_critic1)
        self.soft_update(self.critic2, self.target_critic2)
        
        return critic_loss, actor_loss
    
    def soft_update(self, source, target):
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data.copy_(self.tau * src_param.data + (1 - self.tau) * tgt_param.data)
```

### 4.3 Offline RL (离线强化学习)

**核心问题**: 只有固定的历史数据集，无法与环境交互。

**挑战**: 分布偏移 (Distribution Shift) - 策略可能选择数据集中没见过的动作。

#### 4.3.1 CQL (Conservative Q-Learning)

**思想**: 保守估计 Q 值，惩罚数据集外的动作。

```
L_CQL = α × E_{s ~ D}[ log Σ_a exp(Q(s,a)) - E_{a ~ D}[Q(s,a)] ] + L_TD
```

#### 4.3.2 IQL (Implicit Q-Learning)

**思想**: 使用分位数回归避免显式最大化 Q。

#### 4.3.3 Recap (π*0.6 的核心算法)

+**思想**: 从成功和失败的轨迹中学习（详细机制可参考 [pi0_6_dissection.md](./pi0_6_dissection.md) 中的 Recap 解析）。

```python
class RecapAlgorithm:
    """π*0.6 的 Recap 离线 RL 算法"""
    def __init__(self, policy, value_net):
        self.policy = policy
        self.value_net = value_net
    
    def label_trajectories(self, trajectories):
        """标注轨迹: 成功 vs 失败"""
        labeled = []
        for traj in trajectories:
            success = traj['final_reward'] > 0  # 任务是否成功
            for t, transition in enumerate(traj['transitions']):
                # 关键: 找到失败轨迹中"开始出错"的时刻
                if not success and self.is_critical_failure(traj, t):
                    transition['label'] = 'negative'  # 负样本
                elif success:
                    transition['label'] = 'positive'  # 正样本
                labeled.append(transition)
        return labeled
    
    def is_critical_failure(self, traj, t):
        """判断是否是导致失败的关键时刻"""
        # 使用价值函数估计: 如果 V 骤降，说明这步出错了
        v_t = self.value_net(traj['states'][t])
        v_t1 = self.value_net(traj['states'][t+1])
        return (v_t - v_t1) > threshold
    
    def update(self, labeled_data):
        """对比学习: 提升正样本概率，降低负样本概率"""
        loss = 0
        for sample in labeled_data:
            state, action = sample['state'], sample['action']
            log_prob = self.policy.log_prob(state, action)
            
            if sample['label'] == 'positive':
                loss -= log_prob  # 提升正样本概率
            else:
                loss += log_prob  # 降低负样本概率
        
        return loss / len(labeled_data)
```

## 5. 奖励设计 (Reward Engineering)

### 5.1 稀疏奖励 vs 稠密奖励

| 类型 | 示例 | 优点 | 缺点 |
| :--- | :--- | :--- | :--- |
| **稀疏** | 任务成功 +1，否则 0 | 不需要人工设计 | 难以学习 |
| **稠密** | 距离目标越近奖励越高 | 学习容易 | 可能导致局部最优 |

### 5.2 奖励设计示例

```python
def compute_reward(state, action, next_state, task_type="pick_and_place"):
    """机器人操作任务的奖励函数"""
    
    if task_type == "pick_and_place":
        gripper_pos = state['gripper_position']
        object_pos = state['object_position']
        target_pos = state['target_position']
        
        # 阶段 1: 接近物体
        dist_to_object = np.linalg.norm(gripper_pos - object_pos)
        
        # 阶段 2: 抓取物体
        is_grasping = state['gripper_closed'] and dist_to_object < 0.02
        
        # 阶段 3: 移动到目标
        if is_grasping:
            dist_to_target = np.linalg.norm(object_pos - target_pos)
        else:
            dist_to_target = 1.0  # 惩罚没抓到物体
        
        # 组合奖励
        reward = -0.1 * dist_to_object  # 接近物体
        reward += 0.5 * is_grasping     # 抓取奖励
        reward -= 0.1 * dist_to_target  # 接近目标
        
        # 稀疏成功奖励
        if dist_to_target < 0.05 and is_grasping:
            reward += 10.0  # 任务成功
        
        return reward
```

### 5.3 奖励塑形 (Reward Shaping)

```python
def shaped_reward(state, next_state, potential_func, gamma=0.99):
    """
    基于势函数的奖励塑形 (不改变最优策略)
    F(s, s') = γ * Φ(s') - Φ(s)
    """
    phi_s = potential_func(state)
    phi_s_next = potential_func(next_state)
    shaping = gamma * phi_s_next - phi_s
    return shaping
```

## 6. RL + VLA 的结合 (RL + VLA Integration)

### 6.1 RLHF 完整流程 (RLHF Pipeline)

```
┌─────────────────────────────────────────────────────────────────┐
│                    RLHF 三阶段流程                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   阶段 1: SFT (Supervised Fine-Tuning)                          │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  预训练模型 + 高质量数据 → 基础策略 π_SFT                 │   │
│   └─────────────────────────────────────────────────────────┘   │
│                          │                                      │
│                          ▼                                      │
│   阶段 2: Reward Model Training                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  收集人类偏好 (A > B) → 训练奖励模型 R(s, a)              │   │
│   │  Loss: -log σ(R(y_w) - R(y_l))  (Bradley-Terry)          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                          │                                      │
│                          ▼                                      │
│   阶段 3: RL Fine-tuning (PPO)                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  使用 R(s, a) 作为奖励，PPO 优化策略                      │   │
│   │  + KL 惩罚: 防止偏离 π_SFT 太远                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   需要同时加载: π_SFT (参考), π_θ (训练), R (奖励), V (价值)    │
│   显存需求: 4 个模型 → 非常昂贵!                                │
└─────────────────────────────────────────────────────────────────┘
```

```python
class RLHF_VLA:
    """使用人类反馈强化学习改进 VLA"""
    def __init__(self, vla_model, reward_model):
        self.vla = vla_model
        self.reward_model = reward_model  # 从人类偏好训练
    
    def collect_comparisons(self, states, num_samples=2):
        """收集人类偏好比较"""
        actions = [self.vla.sample(states) for _ in range(num_samples)]
        # 人类选择更好的动作
        human_preference = get_human_preference(states, actions)
        return actions, human_preference
    
    def train_reward_model(self, comparisons):
        """从偏好数据训练奖励模型"""
        for (action_win, action_lose, state) in comparisons:
            r_win = self.reward_model(state, action_win)
            r_lose = self.reward_model(state, action_lose)
            
            # Bradley-Terry 模型
            loss = -torch.log(torch.sigmoid(r_win - r_lose))
            loss.backward()
    
    def rl_finetune(self, states):
        """使用学到的奖励进行 RL 微调"""
        actions = self.vla.sample(states)
        rewards = self.reward_model(states, actions)
        
        # PPO 更新
        self.ppo_update(states, actions, rewards)
```

### 6.2 从演示初始化 RL (Demo-Guided RL)

```python
class DemoGuidedRL:
    """结合 BC 预训练和 RL 微调"""
    def __init__(self, policy):
        self.policy = policy
    
    def phase1_bc_pretrain(self, demonstrations):
        """Phase 1: 行为克隆预训练"""
        for state, action in demonstrations:
            pred_action = self.policy(state)
            loss = F.mse_loss(pred_action, action)
            loss.backward()
    
    def phase2_rl_finetune(self, env, num_episodes=1000):
        """Phase 2: RL 微调超越演示"""
        for episode in range(num_episodes):
            state = env.reset()
            while not done:
                action = self.policy.sample(state)
                next_state, reward, done, _ = env.step(action)
                
                # 存储经验
                self.buffer.add(state, action, reward, next_state, done)
                state = next_state
            
            # SAC/PPO 更新
            self.rl_update()
```

## 7. GR-RL: VLA + RL 融合框架案例 (ByteDance Seed)

> **GR-RL**: 字节跳动 Seed 团队 2025 年发布的机器人学习框架，专为**长时程、灵巧、高精度操作**设计。
> [[官网](https://seed.bytedance.com/en/gr_rl)]

### 7.1 核心洞察

GR-RL 揭示了一个被行业长期忽略的事实：

```
┌─────────────────────────────────────────────────────────────────┐
│               GR-RL 解决的三大核心问题                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   问题 1: 演示数据有噪声                                        │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  人类演示中存在失误（鞋带滑脱、反复尝试）                  │   │
│   │  直接喂给模型 → 学会"怎么犯错"                           │   │
│   │  💡 解决: Offline RL Critic 筛选高质量数据                │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   问题 2: 训练-部署不一致                                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  训练: 输出"原始动作"                                    │   │
│   │  部署: 执行"平滑后处理动作"                              │   │
│   │  💡 解决: 在线 RL 对齐（潜在空间探索）                    │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   问题 3: 数据量有限                                            │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  高精度任务的演示数据采集成本高                           │   │
│   │  💡 解决: 形态对称性增强（镜像翻转数据翻倍）              │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 模型架构

GR-RL 采用 **Mixture-of-Transformer (MoT)** 混合架构，总参数量 **50 亿**。

```
┌─────────────────────────────────────────────────────────────────┐
│                    GR-RL 架构 (5B 参数)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              VLA Policy (π)                              │   │
│   │                                                          │   │
│   │   Vision Encoder ──┐                                     │   │
│   │                    │                                     │   │
│   │   Language Encoder ├──→ MoT Backbone ──→ Action Head     │   │
│   │                    │                                     │   │
│   │   State Encoder ───┘                                     │   │
│   └─────────────────────────────────────────────────────────┘   │
│                            │                                    │
│                            │ 共享特征                           │
│                            ▼                                    │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │           Multi-task Critic (任务进度评估)               │   │
│   │                                                          │   │
│   │   输入: (state, action, task_embedding)                  │   │
│   │   输出: 任务进度值 V(s) ∈ [0, 1]                         │   │
│   │                                                          │   │
│   │   功能:                                                  │   │
│   │   • 评估每一步对整体任务的贡献                           │   │
│   │   • 进展快 → 值高；出错 → 值骤降                        │   │
│   │   • 用于筛选"非最优数据"                                │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 多阶段训练流程

```
┌─────────────────────────────────────────────────────────────────┐
│                 GR-RL 三阶段训练流程                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   阶段 1: 演示轨迹筛选 (Offline RL Critic)                      │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  原始演示数据 (含失误)                                   │   │
│   │        │                                                 │   │
│   │        ▼                                                 │   │
│   │  Critic 评估每步进度值 V(s)                              │   │
│   │        │                                                 │   │
│   │        ├── V 骤降 → 标记为"失败时刻"→ 剔除              │   │
│   │        │                                                 │   │
│   │        └── V 稳步上升 → 保留为高质量数据                 │   │
│   └─────────────────────────────────────────────────────────┘   │
│                          │                                      │
│                          ▼                                      │
│   阶段 2: 形态对称性增强 (Data Augmentation)                    │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  对筛选后的数据进行镜像翻转:                             │   │
│   │  • 图像水平翻转 + 左右手视角交换                         │   │
│   │  • 状态/动作在世界坐标系镜像变换                         │   │
│   │  • 语言指令空间描述调整 ("左边" → "右边")               │   │
│   │                                                          │   │
│   │  效果: 数据量翻倍，无需额外采集成本                      │   │
│   └─────────────────────────────────────────────────────────┘   │
│                          │                                      │
│                          ▼                                      │
│   阶段 3: 在线强化学习优化 (Online RL Alignment)                │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  问题: 训练时输出 ≠ 部署时执行的动作                     │   │
│   │                                                          │   │
│   │  方案: 在扩散模型的潜在噪声空间中进行结构化探索          │   │
│   │        (不在原始动作空间盲目加噪声)                       │   │
│   │                                                          │   │
│   │  效果: 弥合训练-部署鸿沟，稳定长时程精细操作             │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.4 Critic 进度评估原理

```python
class TaskProgressCritic:
    """GR-RL 的任务进度评估器"""
    def __init__(self, encoder, threshold=0.3):
        self.encoder = encoder  # 共享 VLA 编码器
        self.value_head = nn.Linear(hidden_dim, 1)
        self.threshold = threshold  # 进度骤降阈值
    
    def forward(self, state, task_embedding):
        """评估当前状态的任务进度"""
        features = self.encoder(state, task_embedding)
        progress = torch.sigmoid(self.value_head(features))  # [0, 1]
        return progress
    
    def filter_demonstrations(self, trajectory):
        """筛选高质量演示数据"""
        filtered = []
        for t in range(len(trajectory) - 1):
            v_t = self.forward(trajectory[t]['state'], trajectory[t]['task'])
            v_t1 = self.forward(trajectory[t+1]['state'], trajectory[t+1]['task'])
            
            progress_drop = v_t - v_t1
            
            if progress_drop > self.threshold:
                # 进度骤降 → 失败时刻，剔除
                print(f"⚠️ 检测到失败时刻 t={t}, drop={progress_drop:.2f}")
                continue
            
            filtered.append(trajectory[t])
        
        return filtered
```

### 7.5 与其他 VLA 方法对比

| 方法 | 核心思路 | 数据利用 | 在线学习 | 精细操作 |
| :--- | :--- | :--- | :--- | :--- |
| **BC (传统 VLA)** | 纯模仿 | 不筛选 | ❌ | 受限 |
| **π0.6 (Recap)** | Offline RL 筛选 | ✅ 筛选 | ❌ | 中等 |
| **GR-RL** | Offline + Online RL | ✅ 筛选 + 增强 | ✅ | ✅ 高精度 |

### 7.6 GR-RL 的"通才转专家"范式

```
┌─────────────────────────────────────────────────────────────────┐
│               GR-RL "通才转专家" 方法论                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Step 1: 通才预训练 (如 GR-3, 40B VLA)                         │
│           泛化、新环境适应、基础操作能力                         │
│                    │                                            │
│                    ▼                                            │
│   Step 2: 专家微调 (GR-RL)                                      │
│           • RL Critic 筛选高质量专家数据                        │
│           • 形态对称增强提升泛化                                │
│           • 在线 RL 对齐部署差异                                │
│                    │                                            │
│                    ▼                                            │
│   Step 3: 高精度任务 (如穿鞋带)                                 │
│           长时程、灵巧、高精度操作                               │
│                                                                 │
│   💡 核心: RL 不是替代 BC，而是"提纯" + "对齐"                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.7 面试 Q&A

**Q1: GR-RL 与 π0.6 的 Recap 有什么区别？**

A:
| 对比 | π0.6 Recap | GR-RL |
| :--- | :--- | :--- |
| **数据筛选** | 基于成功/失败标签 | 基于进度值骤降检测 |
| **在线学习** | ❌ 纯 Offline | ✅ Offline + Online |
| **探索空间** | - | 扩散模型潜在空间 |
| **数据增强** | 无 | 形态对称镜像 |

**Q2: 为什么在潜在噪声空间探索而不是动作空间？**

A:
- **动作空间噪声**: 可能产生不合理动作（如关节超限）
- **潜在空间噪声**: 结构化，保持动作语义合理性
- **效率**: 潜在空间维度低，探索效率更高

**Q3: GR-RL 的局限性是什么？**

A:
- **实验条件**: 论文中鞋孔处于"理想状态"，真实场景更复杂
- **泛化性**: 对未见过的物体/场景仍需验证
- **计算成本**: 50B 参数 + 在线 RL 训练成本高

## 8. 面试高频问题 (Q&A)

**Q1: VLA 中 BC 和 RL 如何取舍?**

A:
- **优先 BC**: 数据充足、任务简单、需要快速迭代
- **引入 RL**: 需要超越人类、长序列优化、稀疏奖励任务
- **最佳实践**: BC 预训练 + RL 微调 (如 π*0.6)

**Q2: Offline RL 和 Online RL 的核心区别?**

A:
- **Online RL**: 可以与环境交互，探索新状态
- **Offline RL**: 只有固定数据集，需要处理分布偏移
- **VLA 现状**: 因为真机交互成本高，Offline RL 更实用

**Q3: SAC 中温度参数 α 的作用?**

A:
- **α 大**: 更重视熵 → 更多探索 → 策略更随机
- **α 小**: 更重视奖励 → 更多利用 → 策略更确定
- **自动调节**: 可以将 α 设为可学习参数

**Q4: 为什么 PPO 比 TRPO 更流行?**

A:
- **简单**: PPO 只需 Clip，TRPO 需要计算 Fisher 矩阵
- **高效**: PPO 可以用 SGD，TRPO 需要共轭梯度
- **效果**: 在大多数任务上性能相当

**Q5: Recap 算法相比传统 Offline RL 的优势?**

A:
- **利用失败数据**: 传统方法只模仿成功轨迹，Recap 从失败中学习
- **关键时刻识别**: 通过价值函数定位"出错点"
- **简单高效**: 不需要复杂的约束优化

**Q6: RLHF 的基本流程是什么？与 DPO 的差异是什么？**

A: **RLHF 三阶段流程**:
1. **SFT**: 在高质量数据上监督微调，得到基础策略 $\pi_{SFT}$
2. **Reward Model**: 从人类偏好数据训练奖励模型 $R(s, a)$
3. **RL (PPO)**: 使用 $R$ 作为奖励，PPO 优化策略，加 KL 惩罚防止偏离 $\pi_{SFT}$

**DPO (Direct Preference Optimization)** 的差异:
- **跳过 Reward Model**: 直接从偏好数据优化策略
- **数学推导**: 将 RL 目标重参数化为分类问题

```
L_DPO = -E[ log σ( β × log(π_θ(y_w|x) / π_ref(y_w|x)) - β × log(π_θ(y_l|x) / π_ref(y_l|x)) ) ]
```

| 对比 | RLHF | DPO |
| :--- | :--- | :--- |
| **阶段数** | 3 阶段 | 1 阶段 |
| **模型数** | 4 个 (π, π_ref, R, V) | 2 个 (π, π_ref) |
| **稳定性** | PPO 易崩 | 更稳定 |
| **计算量** | 高 | 低 |
| **效果** | 略优 (理论上) | 接近 RLHF |

## 9. 主流 RL 框架 (RL Frameworks)

### 8.1 框架对比

| 框架 | 定位 | 优势 | 适用场景 |
| :--- | :--- | :--- | :--- |
| **Stable Baselines3** | 易用、稳定 | 文档完善，API 简洁 | 快速实验、教学 |
| **RLlib** | 分布式、可扩展 | Ray 生态，多 GPU/节点 | 大规模训练 |
| **CleanRL** | 单文件实现 | 代码清晰，易于修改 | 学习、研究 |
| **TorchRL** | PyTorch 官方 | 与 PyTorch 深度集成 | 生产级应用 |
| **SKRL** | Isaac Lab 集成 | GPU 并行，机器人专用 | 机器人 RL |

### 8.2 Stable Baselines3 (SB3)

**特点**: 最易上手的 RL 库，API 设计优雅。

```python
# Stable Baselines3 快速上手
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

# 创建向量化环境
env = make_vec_env("Pendulum-v1", n_envs=4)

# 创建模型
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    verbose=1,
    tensorboard_log="./logs/"
)

# 训练
model.learn(total_timesteps=100_000, callback=EvalCallback(env))

# 保存/加载
model.save("ppo_pendulum")
model = PPO.load("ppo_pendulum")

# 推理
obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
```

**自定义网络**:

```python
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn

class CustomCNN(BaseFeaturesExtractor):
    """自定义 CNN 特征提取器 (用于图像输入)"""
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # 计算输出维度
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, *observation_space.shape)).shape[1]
        self.linear = nn.Linear(n_flatten, features_dim)
    
    def forward(self, observations):
        return self.linear(self.cnn(observations))

# 使用自定义网络
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),
)
model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs)
```

### 8.3 RLlib (Ray)

**特点**: 分布式训练首选，支持多 GPU/多节点。

```python
# RLlib 分布式训练
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

# 配置
config = (
    PPOConfig()
    .environment("CartPole-v1")
    .framework("torch")
    .rollouts(
        num_rollout_workers=4,      # 并行 worker 数
        num_envs_per_worker=2,      # 每个 worker 的环境数
    )
    .training(
        lr=5e-5,
        gamma=0.99,
        train_batch_size=4000,
        sgd_minibatch_size=128,
        num_sgd_iter=30,
        model={"fcnet_hiddens": [256, 256]},
    )
    .resources(
        num_gpus=1,                  # 训练用 GPU
        num_cpus_per_worker=1,
    )
)

# 训练
algo = config.build()
for i in range(100):
    result = algo.train()
    print(f"Iter {i}: reward = {result['episode_reward_mean']:.2f}")

# 或使用 Ray Tune 进行超参搜索
tune.run(
    "PPO",
    config=config.to_dict(),
    stop={"episode_reward_mean": 200},
    num_samples=4,  # 并行搜索 4 组超参
)
```

**多智能体 RL**:

```python
# RLlib 多智能体
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .environment("MultiAgentCartPole")
    .multi_agent(
        policies={"policy_0", "policy_1"},
        policy_mapping_fn=lambda agent_id, episode, **kwargs: f"policy_{agent_id}",
    )
)
```

### 8.4 SKRL (Isaac Lab 集成)

**特点**: 专为 Isaac Lab 设计，GPU 并行训练。

```python
# SKRL + Isaac Lab
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.wrappers.torch import wrap_env

# 包装 Isaac Lab 环境
env = wrap_env(isaac_lab_env)

# 配置
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 16
cfg["learning_epochs"] = 8
cfg["mini_batches"] = 4
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 3e-4

# 创建 Agent
agent = PPO(
    models={"policy": policy_net, "value": value_net},
    memory=memory,
    cfg=cfg,
    observation_space=env.observation_space,
    action_space=env.action_space,
)

# 训练
trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)
trainer.train()
```

### 8.5 框架选择指南

```
┌─────────────────────────────────────────────────────────────────┐
│                    RL 框架选择决策树                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Q1: 是否需要分布式训练 (多 GPU/多节点)?                        │
│       │                                                         │
│       ├── 是 → RLlib (Ray 生态，分布式首选)                     │
│       │                                                         │
│       └── 否 → Q2: 是否使用 Isaac Lab?                          │
│                    │                                            │
│                    ├── 是 → SKRL (官方推荐)                     │
│                    │                                            │
│                    └── 否 → Q3: 目标是什么?                     │
│                                 │                               │
│                                 ├── 快速实验 → SB3              │
│                                 ├── 学习研究 → CleanRL          │
│                                 └── 生产部署 → TorchRL          │
│                                                                 │
│   💡 VLA 常用组合:                                              │
│   • 仿真训练: Isaac Lab + SKRL/RSL-RL                           │
│   • 大规模: RLlib + Ray Cluster                                 │
│   • 快速验证: SB3 + Gymnasium                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 8.6 TORCS 与自动驾驶 RL

**TORCS** (The Open Racing Car Simulator) 是经典的自动驾驶 RL 测试平台。

```python
# TORCS 环境使用 (gym_torcs)
import gym
import gym_torcs

env = gym.make("Torcs-v0", vision=True, throttle=True)

# 观测空间: 车辆状态 + 可选视觉
# - speed, angle, trackPos, track sensors (19)
# - 可选: RGB 图像 (64x64x3)

# 动作空间: [steering, throttle, brake]
# - steering: [-1, 1]
# - throttle: [0, 1]
# - brake: [0, 1]

obs = env.reset()
for _ in range(1000):
    action = agent.act(obs)  # 你的策略
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
```

**注**: TORCS 主要用于自动驾驶研究，VLA 机器人领域更常用 Isaac Lab/MuJoCo。

## 10. LIBERO 终身学习基准 (Lifelong Learning Benchmark)

> **LIBERO**: Benchmarking Knowledge Transfer for Lifelong Robot Learning [[GitHub](https://github.com/Lifelong-Robot-Learning/LIBERO)] [[Paper](https://arxiv.org/abs/2306.03310)]

### 9.1 为什么关注 LIBERO?

- **主流基准**: 多任务 / 终身学习 / 知识迁移研究默认引用的数据与任务集
- **任务覆盖全面**: 提供受控分布偏移 (Spatial / Object / Goal) 与 entangled 任务 (LIBERO-100)
- **开箱即用**: 打包示范数据、评测脚本、策略网络 (BC-RNN / BC-Transformer / BC-ViLT) 与算法 (base、ER、EWC、PackNet、Multitask)
- **可扩展**: Procedural generation pipeline 支持生成更多 manipulation 任务，方便自定义研究

### 9.2 任务套件总览

| 套件 | 任务数 | 迁移挑战 | 说明 |
| :--- | :--- | :--- | :--- |
| **LIBERO-Spatial** | 30 | 空间关系 | 相同物体，不同空间布置 |
| **LIBERO-Object** | 30 | 物体类型 | 不同物体属性，考验语义+抓取泛化 |
| **LIBERO-Goal** | 30 | 目标变化 | 同场景，多目标组合 |
| **LIBERO-100** | 100 | 混合 (Entangled) | 拆成 **LIBERO-90** (预训练) + **LIBERO-10** (终身学习测试) |

### 9.3 数据与环境

```bash
# 环境安装
conda create -n libero python=3.8.13
conda activate libero
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
            torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .

# 下载示范 (可选 --datasets 指定套件)
python benchmark_scripts/download_libero_datasets.py --use-huggingface
```

### 9.4 训练 / 评测入口

```bash
# 终身学习训练
export CUDA_VISIBLE_DEVICES=0
python libero/lifelong/main.py \
    seed=0 \
    benchmark_name=LIBERO_90 \
    policy=bc_transformer_policy \
    lifelong=ewc

# 脱机评测
python libero/lifelong/evaluate.py \
    --benchmark LIBERO_10 \
    --task_id 0 \
    --algo ewc \
    --policy bc_transformer_policy \
    --ep 50 \
    --device_id 0
```

### 9.5 与 VLA 的结合方式

```
┌─────────────────────────────────────────────────────────────────┐
│                LIBERO + VLA 结合流程                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   1. 数据层: 复用 LIBERO 示范训练 / 微调 VLA 的 Action Head     │
│                                                                 │
│   2. 评测层: 作为"终身学习回归测试集"，检查 Catastrophic        │
│              Forgetting                                         │
│                                                                 │
│   3. 跨域迁移: 先用 LIBERO-90 预训练，再在 LIBERO-10 /          │
│               自定义任务 → 真机验证 Sim-to-Real                  │
│                                                                 │
│   4. 算法组合: 将 RLHF / DPO 等上层优化与 EWC / PackNet 等      │
│               底层正则结合，形成混合 pipeline                    │
│                                                                 │
│   💡 面试常问: "如何评估 VLA 的终身学习能力？"                   │
│      → 使用 LIBERO-90 预训练 + LIBERO-10 测试遗忘程度            │
└─────────────────────────────────────────────────────────────────┘
```

### 9.6 面试 Q&A

**Q: 如何使用 LIBERO 评估 VLA 的终身学习能力？**

A:
1. **预训练**: 在 LIBERO-90 上训练 VLA 策略
2. **终身学习**: 在 LIBERO-10 的 10 个任务上顺序微调
3. **评估遗忘**: 每学完一个新任务后，测试所有旧任务的成功率
4. **对比算法**: 比较 Sequential Fine-tuning (baseline) vs EWC / PackNet / ER
5. **关键指标**: Forward Transfer (新任务学习速度) + Backward Transfer (旧任务遗忘程度)

## 11. 参考资源 (References)

- **PPO**: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- **SAC**: [Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL](https://arxiv.org/abs/1801.01290)
- **CQL**: [Conservative Q-Learning for Offline RL](https://arxiv.org/abs/2006.04779)
- **IQL**: [Offline RL with Implicit Q-Learning](https://arxiv.org/abs/2110.06169)
- **Spinning Up**: [OpenAI Spinning Up in Deep RL](https://spinningup.openai.com/)
- **Stable Baselines3**: [Docs](https://stable-baselines3.readthedocs.io/)
- **RLlib**: [Docs](https://docs.ray.io/en/latest/rllib/)
- **SKRL**: [Docs](https://skrl.readthedocs.io/)

---
[← Back to Theory](./README.md)
