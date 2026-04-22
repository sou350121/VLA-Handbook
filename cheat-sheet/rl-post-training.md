# RL 后训练三流派 + World Model 辅助

> 2026-04-21 · 参考 [World Model 辅助 VLA 后训练深度解读](../theory/world-model/world_model_aided_vla_post_training_deep_dive_2026.md)

---

## 🎯 一张速查：该用哪个

| 你的需求 | 推荐流派 | 代表工作 |
|---------|---------|---------|
| 最简 + 稳定收敛 | **ACP（π*0.6 派）** | [π*0.6](https://faculty.cc.gatech.edu/) |
| 精细操作 + 不破坏基模 | **RL Token 派** | [RL Token](https://faculty.cc.gatech.edu/) |
| 想压榨理论极限 | **Flow-matching RL 派** | πRL / π-StepNFT / DPPO |
| 不能做真机 RL（自动驾驶/全身运控）| **WM + Co-evolution** | World-VLA-Loop / VLAW / WoVR |

---

## 📐 预备：RL 核心概念速通

### 奖励定义（主流做法）

$$
r_t = \begin{cases} -1 & \text{任务未成功} \\ -C_{\text{fail}} & \text{任务失败终点} \\ 0 & \text{任务成功终点} \end{cases}
$$

$$V(s_t) = \sum_{t'=t}^T \gamma^{t'-t} r_{t'}$$

归一化到 $[-1, 0]$ → 得到**任务完成百分比**的查询模型。

### Advantage

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

**"动作 a 相对于平均策略好多少"**——ACP / AWR / RL Token 都围绕这个。

---

## 🏛️ 流派 1：传统 RL（为 flow-matching 设计策略梯度）

### 代表工作
- **Diffusion Policy Policy Optimization (DPPO)**
- **πRL** · Online RL Fine-tuning for Flow-based VLAs
- **π-StepNFT** · Wider Space Needs Finer Steps
- **RL for Flow-Matching Policies**

### 核心挑战
Flow Matching 策略 $\pi(\cdot|s)$ 是**多步 ODE 后的分布** → 无法直接解析 $\log \pi$ → 传统 policy gradient 失效。

### 通用解法
把整个**降噪过程建模为 MDP**，每一步降噪作为一个 policy step（用 Gaussian 建模）：

```
Full Flow Policy (implicit)
      │
      ▼ decompose as MDP
┌─────────────────────────────────────┐
│  step 0: π(x_1 → x_{0.9})  (Gaussian) │
│  step 1: π(x_{0.9} → x_{0.8})          │
│  ...                                    │
│  step N: π(x_{0.1} → x_0)              │
└─────────────────────────────────────┘
  ← 每步做标准 PPO，最终策略 = 整个过程
```

### Pros / Cons
- ✅ 理论清晰
- ❌ 训练复杂度高
- ❌ 不稳定（多步 MDP 高维）

---

## 🎓 流派 2：ACP（向监督学习靠齐）

### 核心工作：**π*0.6**

```
1. VLA 在真实环境 rollout
2. Human-in-the-loop intervene + 打成功/失败标签
3. 训 Value Function（V(s) 归一化）
4. Supervised learning with advantage as condition
5. 迭代回 1
```

### Advantage Conditioned Policy 公式

$$\pi(a | s, A^{\text{norm}}) \quad \text{with } A \in [0, 1]$$

- 训练：模型输入 state + advantage，输出动作 → MSE loss
- 推理：固定 $A=1$（"最优"动作）

### Advantage Weighted Regression（相关做法）

$$\mathcal{L}_{\text{AWR}} = \mathbb{E} [w(A) \cdot \lVert a_\theta(s) - a_{\text{label}} \rVert^2]$$

权重 $w(A) = e^{A/\beta}$，高 advantage 样本权重大。

### 实证（📎 π*0.6）
- 迭代次数越多，成功率**逐步提升**
- 非常稳定
- 代价：对 value 函数质量高度敏感

### Pros / Cons
- ✅ **稳定** · 监督学习，易收敛
- ✅ 和现有 VLA 架构兼容
- ❌ 需要好的 value function
- ❌ 不是真正"探索"——只是在已有数据上做条件匹配

---

## 🔧 流派 3：RL Token（独立 Actor）

### 核心工作：**RL Token**

```
 VLA（冻结）─┐                   
            │                   
            │ 提取 Token         
            │                   
            ▼                   
     Actor ←→ Critic            
      ↓                         
   精细调整                      
      ↓                         
  执行动作                       
```

### 关键设计
- **冻结整个 VLA**
- 只训练轻量 **Actor + Critic**
- Actor 基于 VLA 输出做**精细调整**（分层思想）
- Critic 用 **Q(s, a)**（不是 V(s)），支持 off-policy

### 为什么 Q 不是 V？
- Q-learning 支持 off-policy → **采样效率高**
- 适合 online real-robot RL

### 📎 惊艳结果
RL 后模型操作速度**超过人类遥操作速度**。

### Pros / Cons
- ✅ 不破坏基模能力（避免单任务过拟合）
- ✅ Online RL 效率高（微调轻量模块）
- ✅ 适合精细操作任务
- ❌ 不适合"重构策略"的场景

---

## 🌐 World Model 辅助 RL（不能做真机 RL 的唯一出路）

### 基本范式

```
[1] 真实环境或预训练数据 → [2] 训 World Model
                                ↓
          VLA 在 World Model 中 rollout（模拟物理）
                                ↓
          Value/Reward Model 给分 → RL 更新
                                ↓
          真实数据补充 → [1]（Co-evolution）
```

### 🚨 三大硬伤 + 当前解法

#### 1. 精细控制失真（Hallucination）
**问题**：AC-WM 预测视觉合理但**物理不对**（WoVR：预测成功夹取，真实失败）

**解法**：
- **架构**：AdaLN + Cross-Attention 强化 action condition
- **数据**：补齐成功/失败边界数据（World-VLA-Loop）

#### 2. 自回归误差累积
**问题**：rollout 20 次后视频"合理但偏离物理真相"

**解法（全是减 rollout 步数）**：

| 方法 | 具体 |
|------|------|
| **RISE** | 最多 2 次 rollout |
| **WoVR** | 从关键帧开始（非初始状态） |
| **VLA-MBPO** | Chunk-level（每 chunk 生成一帧，1/k 步数） |

#### 3. VLA × AC-WM state-action 分布不一致
**问题**：VLA 探索到 OOD state → WM 预测错误 → VLA 学会 hack WM → 真机崩

**解法：Co-evolution 迭代**

```
收集 VLA rollout → 微调 WM → WM 辅助 VLA → 再收集真机数据 → ...
```

### Co-evolution 代表工作

| 工作 | 关键特点 |
|------|---------|
| **GigaBrain-0.5M*** | 4 阶段 · Latent WM（不是 pixel）· stochastic masking |
| **VLAW** | 4 步迭代 · 真机 + 合成 |
| **WoVR** | 关键帧 rollout · 冻结 VLA 采数据训 WM |
| **World-VLA-Loop** | 4 步迭代 · WM 融合 reward head |
| **RISE** | 真机:合成 = **6:4** · 防灾难性遗忘 |

### 📎 ViVa 的范式转移（2026 Q1）
用 **Video Generation Model 做 Value Model backbone** > VLM 做 backbone。
- 视频预测隐式包含"任务完成度"
- 任务进度信息比"成功/失败"稀疏标签更易收敛

---

## ❓ 四个灵魂拷问（选型前想清楚）

1. **AC-WM 真的带来增益吗？** 🧠 现有工作多数缺对照实验（纯真机 RL vs WM 辅助 RL）
2. **Pixel WM 还是 Latent WM？** 🧠 VLA 决策不需要像素级未来（Danfei Xu 观点），Latent 可能是正确方向
3. **真的需要 WM 吗？** 🧠 如果 Q(s,a) 学得够好（RL Token 路径），WM 可能冗余
4. **ACP 本质 = CFG？** 🧠 Classifier-Free Guidance 的同构——都是"条件控制最优性"

---

## 🔮 未来方向（可能的 Ph.D 题目）

- **World Action Model 统一**：transition + policy + value 合成一个模型
- **RL 之外的后训练**：preference learning (DPO) · self-play · LLM-as-critic
- **更好的 Value Model**：ViVa 证明 video gen > VLM，下一步？
- **混合模式**：精细动作真机 RL + 长程规划 WM RL + Co-evolution

---

## 📊 决策流程图（你该选哪条路）

```
你的约束？
│
├─ 真机 RL 不可行（自动驾驶/全身运控）
│   └─ World Model + Co-evolution（必选）
│
├─ 有真机但成本高
│   ├─ 想迭代数据 → ACP（π*0.6）✅ 最稳
│   ├─ 要精细操作 → RL Token ✅ 最高效
│   └─ 要理论完整 → 传统 RL 派
│
├─ 能做大规模真机 RL
│   └─ 就做真机 RL · WM 是退路，不是首选
│
└─ 数据增强需求
    └─ WM 作为工具（retargeting / 人手→机械手对齐）
```

---

## 📚 延伸阅读

- [World Model 辅助 VLA 后训练深度解读](../theory/world-model/world_model_aided_vla_post_training_deep_dive_2026.md) · 完整文献拆解
- [VLA 架构主线](../theory/vla-core/vla_arch.md) · Flow-based VLA 基础
- [Danfei Xu 访谈](../theory/foundation/human_data_sensorimotor_ghost_danfei_xu_interview_2026.md) · System 2 的真意义

---

[← Back to Cheat Sheet](./README.md)
