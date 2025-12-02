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

$$
\text{MDP} = (S, A, P, R, \gamma)
$$

- **$S$**: 状态空间 (State Space) - 机器人观测到的一切
- **$A$**: 动作空间 (Action Space) - 可执行的动作
- **$P(s'|s,a)$**: 状态转移概率 - 环境动力学
- **$R(s,a)$**: 奖励函数 - 行为好坏的反馈
- **$\gamma$**: 折扣因子 - 未来奖励的权重 (通常 0.99)

### 2.2 核心目标

最大化**累积折扣回报 (Cumulative Discounted Return)**:

$$
G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

### 2.3 价值函数 (Value Functions)

**状态价值函数 (State Value)**:
$$
V^\pi(s) = \mathbb{E}_\pi \left[ G_t | S_t = s \right]
$$

**动作价值函数 (Action Value / Q-Function)**:
$$
Q^\pi(s, a) = \mathbb{E}_\pi \left[ G_t | S_t = s, A_t = a \right]
$$

**Bellman 方程**:
$$
Q^\pi(s, a) = R(s, a) + \gamma \mathbb{E}_{s' \sim P} \left[ V^\pi(s') \right]
$$

### 2.4 策略 (Policy)

策略 $\pi(a|s)$ 定义了在状态 $s$ 下采取动作 $a$ 的概率。

- **确定性策略**: $a = \pi(s)$
- **随机策略**: $a \sim \pi(\cdot|s)$

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

| 类型 | 原理 | 优点 | 缺点 |
| :--- | :--- | :--- | :--- |
| **Model-Free** | 直接学习策略/价值 | 不需要环境模型 | 样本效率低 |
| **Model-Based** | 学习环境动力学模型 | 样本效率高 | 模型误差累积 |

### 3.2 On-Policy vs Off-Policy

| 类型 | 代表算法 | 特点 |
| :--- | :--- | :--- |
| **On-Policy** | PPO, TRPO | 只用当前策略的数据，稳定但低效 |
| **Off-Policy** | SAC, TD3 | 可复用历史数据，高效但不稳定 |

## 4. VLA 常用 RL 算法 (RL Algorithms for VLA)

### 4.1 PPO (Proximal Policy Optimization)

**核心思想**: 限制策略更新幅度，防止训练崩溃。

**目标函数**:
$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 是重要性采样比率。

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
$$
J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_t R(s_t, a_t) + \alpha H(\pi(\cdot|s_t)) \right]
$$

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

$$
\mathcal{L}_{CQL} = \alpha \mathbb{E}_{s \sim D} \left[ \log \sum_a \exp(Q(s,a)) - \mathbb{E}_{a \sim D}[Q(s,a)] \right] + \mathcal{L}_{TD}
$$

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

## 7. 面试高频问题 (Q&A)

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

$$
\mathcal{L}_{DPO} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]
$$

| 对比 | RLHF | DPO |
| :--- | :--- | :--- |
| **阶段数** | 3 阶段 | 1 阶段 |
| **模型数** | 4 个 (π, π_ref, R, V) | 2 个 (π, π_ref) |
| **稳定性** | PPO 易崩 | 更稳定 |
| **计算量** | 高 | 低 |
| **效果** | 略优 (理论上) | 接近 RLHF |

## 8. 参考资源 (References)

- **PPO**: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- **SAC**: [Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL](https://arxiv.org/abs/1801.01290)
- **CQL**: [Conservative Q-Learning for Offline RL](https://arxiv.org/abs/2006.04779)
- **IQL**: [Offline RL with Implicit Q-Learning](https://arxiv.org/abs/2110.06169)
- **Spinning Up**: [OpenAI Spinning Up in Deep RL](https://spinningup.openai.com/)

---
[← Back to Theory](./README.md)

