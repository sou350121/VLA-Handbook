# ACT: 动作分块变换器 (Action Chunking with Transformers)

> **核心论文**: [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705) (Tony Z. Zhao et al., RSS 2023)
> **代表项目**: **ALOHA**, **Mobile ALOHA**, **ACT++**

## 0. 主要數學思想 (Main Mathematical Idea)

> **第一性原理**: **Compression of Intent (意图的压缩)**

高维的动作空间（肌肉纤维/电机控制）往往是由低维的"意图"（Grab cup）驱动的。

- **核心数学工具**: **Variational Inference (变分推断 / CVAE)**。
- **解题逻辑**:
    1.  **隐变量假设**: 假设复杂的动作序列 $a$ 是由一个不可观测的低维变量 $z$（意图/风格）生成的。
    2.  **平衡**: 我们希望找到这个 $z$。ELBO (Evidence Lower Bound) 损失函数在两件事之间寻找平衡：
        - **重建 (Reconstruction)**: $z$ 必须包含足够的信息来还原出精确的动作（要把活干好）。
        - **正则 (Regularization)**: $z$ 的分布应该尽可能简单（如标准正态分布），以便于我们在推理时可以随意采样（通用性/生成能力）。
    3.  **多模态**: 不同的 $z$ 代表不同的意图，从而生成不同的合理动作序列，解决了"平均化"问题。

## 1. 为什么需要 ACT? (Why?)

传统的行为克隆 (Behavior Cloning, BC) 方法通常采用**单步预测**：每次只预测下一时刻的动作。这种方式存在严重的**误差累积 (Compounding Error)** 问题。

### 1.1 单步预测的致命缺陷

```
单步预测流程:
观测 o_t → 策略 π(o_t) → 动作 a_t → 执行 → 观测 o_{t+1} → ...
```

**问题**:
- **误差累积**: 每一步的预测误差会传递到下一步。如果每步有 1% 的误差，100 步后误差可能达到 100%。
- **分布偏移 (Distribution Shift)**: 训练数据来自专家轨迹，但执行时的状态分布可能偏离专家，导致"越错越离谱"。
- **高频闭环压力**: 需要每一帧都精确预测，对模型要求极高。

### 1.2 ACT 的解决思路

ACT 提出了一个简单而有效的想法：**一次预测一段未来的动作序列 (Action Chunk)**，而不是只预测下一步。


$$
\text{单步 BC}: \pi(o_t) \rightarrow a_t
$$



$$
\text{ACT}: \pi(o_t) \rightarrow [a_t, a_{t+1}, ..., a_{t+k-1}]
$$


**核心优势**:
- **减少决策点**: 如果 chunk 大小 $k=100$，决策频率从 50Hz 降到 0.5Hz，大大降低了误差累积的机会。
- **隐式任务分解**: 预测一整段动作，模型需要"理解"整个子任务的结构 (如"伸手→抓握→抬起")，而不只是盲目模仿下一帧。
- **平滑轨迹**: 一次生成的轨迹天然连贯，避免了单步预测的抖动。

## 2. 核心技术 (Core Techniques)

### 2.1 动作分块 (Action Chunking)

**定义**: 将连续的动作序列分成固定长度 $k$ 的"块"，模型一次预测一整个块。

**数学表示**:
- 输入: 当前观测 $o_t$ (图像 + 本体感知 + 语言指令)
- 输出: 动作序列 $\mathbf{a} = [a_t, a_{t+1}, ..., a_{t+k-1}] \in \mathbb{R}^{k \times D_a}$
- 其中 $D_a$ 是动作维度 (如 7-DoF 机械臂 + 1 夹爪 = 8)

**典型参数**:
- **ALOHA**: $k = 100$ (在 50Hz 控制下，对应 2 秒的动作)
- **Mobile ALOHA**: $k = 50$ (对应 1 秒)

### 2.2 时间集成 (Temporal Ensemble)

仅用 Action Chunking 还不够。如果在 $t=0$ 时预测了 $[a_0, ..., a_{99}]$，然后在 $t=100$ 时再预测 $[a_{100}, ..., a_{199}]$，两段轨迹的交界处可能不连续。

**解决方案**: **重叠预测 + 指数加权平均**

```
时间步 t=0:   预测 [a_0, a_1, ..., a_99]
时间步 t=1:   预测 [a_1', a_2', ..., a_100']
时间步 t=2:   预测 [a_2'', a_3'', ..., a_101'']
...

执行 a_t 时，融合所有对 a_t 的预测:
a_t^{final} = Σ w_i * a_t^{(i)}
```

**指数加权公式**:

$$
w_i = \exp(-m \cdot i)
$$


其中 $m$ 是衰减系数 (通常 $m=0.01$)，$i$ 是预测的"年龄"(越新的预测权重越大)。

**效果**:
- **平滑过渡**: 新旧预测的融合消除了不连续。
- **鲁棒性**: 即使某次预测有误，也会被其他预测"稀释"。

### 2.3 CVAE 架构 (Conditional Variational Autoencoder)

ACT 的另一个核心创新是使用 **CVAE** 来处理动作的**多模态分布**。

#### 2.3.1 为什么需要 CVAE?

与 Diffusion Policy 类似，机器人动作存在**多解**问题 (如绕障碍物可以从左绕或从右绕)。

- **MSE 回归**: 会预测左右的平均值 → 撞向障碍物。
- **CVAE**: 学习动作分布的**隐变量表示**，采样不同的 $z$ 可以生成不同的合理轨迹。

#### 2.3.2 CVAE 数学原理

**训练时 (有真实动作)**:
1. **编码器 (Encoder)** $q_\phi(z | o, a)$: 将观测 $o$ 和真实动作序列 $a$ 编码为隐变量 $z$ 的分布 (均值 $\mu$, 方差 $\sigma^2$)。
2. **解码器 (Decoder)** $p_\theta(a | o, z)$: 从隐变量 $z$ 和观测 $o$ 重建动作序列。

**损失函数**:

$$
\mathcal{L} = \underbrace{\| a - \hat{a} \|^2}_{\text{重建损失}} + \beta \cdot \underbrace{D_{KL}(q_\phi(z|o,a) \| \mathcal{N}(0, I))}_{\text{KL 散度}}
$$


- **重建损失**: 让解码器输出接近真实动作。
- **KL 散度**: 让隐变量分布接近标准正态分布，便于推理时采样。
- **$\beta$**: 权重系数 (ACT 中通常 $\beta = 10$)。

**推理时 (无真实动作)**:
1. 从标准正态分布采样 $z \sim \mathcal{N}(0, I)$。
2. 解码器根据 $o$ 和 $z$ 生成动作序列。

```python
# 训练时
z_mu, z_logvar = encoder(obs, gt_actions)  # 编码真实动作
z = reparameterize(z_mu, z_logvar)         # 重参数化采样
pred_actions = decoder(obs, z)             # 解码

recon_loss = mse(pred_actions, gt_actions)
kl_loss = -0.5 * (1 + z_logvar - z_mu**2 - z_logvar.exp()).sum()
loss = recon_loss + beta * kl_loss

# 推理时
z = torch.randn(batch_size, z_dim)         # 直接从标准正态采样
pred_actions = decoder(obs, z)             # 解码
```

## 3. 网络架构 (Network Architecture)

ACT 使用 **Transformer** 作为骨干网络，具体包括:

### 3.1 观测编码器 (Observation Encoder)

```
输入:
- 图像: [B, T, C, H, W] (可以是多相机)
- 本体感知: [B, D_p] (关节角度、末端位置等)

处理:
- 图像 → ResNet-18 / ViT → 特征向量 [B, T, D_v]
- 本体感知 → Linear → 特征向量 [B, D_p']
- 拼接 → [B, T+1, D]
```

### 3.2 CVAE 编码器 (仅训练时)

```
输入:
- 观测特征: [B, T+1, D]
- 真实动作: [B, k, D_a] (经过 Linear 映射)

结构:
- Transformer Encoder (4 层)
- 输出: z_mu, z_logvar ∈ R^{D_z} (通常 D_z = 32)
```

### 3.3 动作解码器 (Action Decoder)

```
输入:
- 观测特征: [B, T+1, D] (作为 Cross-Attention 的 Key/Value)
- 隐变量 z: [B, D_z] (拼接到每个 Query Token)
- 可学习的 Action Query: [B, k, D] (k 个位置编码)

结构:
- Transformer Decoder (7 层)
- Cross-Attention: Query 关注 观测特征
- 输出: [B, k, D_a] (k 步动作)
```

### 3.4 完整架构图

```
                    ┌─────────────────────────────────────────┐
                    │            CVAE Encoder                 │
                    │  (obs_feat, gt_actions) → z_mu, z_var   │
                    └──────────────────┬──────────────────────┘
                                       │ z (重参数化采样)
                                       ▼
┌──────────────┐    ┌─────────────────────────────────────────┐
│   Images     │───▶│                                         │
│  (多相机)    │    │         Observation Encoder             │
├──────────────┤    │  ResNet-18 / ViT + Positional Emb       │
│  Proprio     │───▶│                                         │
│  (本体感知)  │    └──────────────────┬──────────────────────┘
└──────────────┘                       │ obs_feat [B, T+1, D]
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │          Action Decoder                 │
                    │  (Action Query + z) × obs_feat          │
                    │      Transformer Decoder                │
                    └──────────────────┬──────────────────────┘
                                       │
                                       ▼
                              Action Chunk [B, k, D_a]
```

## 4. 与其他方法的对比 (Comparison)

| 方法 | 预测方式 | 多模态处理 | 推理速度 | 适用场景 |
| :--- | :--- | :--- | :--- | :--- |
| **单步 BC** | 单帧 | 无 (MSE) | 最快 | 简单任务 |
| **ACT (CVAE)** | Chunk ($k$ 步) | CVAE 采样 | **快** | 高频、精细操作 |
| **Diffusion Policy** | Chunk ($k$ 步) | 扩散采样 | 慢 (多步去噪) | 多模态复杂任务 |
| **Flow Matching (π0)** | Chunk ($k$ 步) | ODE 采样 | 中等 | 通用基础模型 |

**ACT 的优势**:
- **推理速度快**: CVAE 只需一次前向传播，Diffusion 需要 10-100 步去噪。
- **简单易实现**: 标准 Transformer + VAE，无需复杂的噪声调度器。
- **数据效率高**: 在 ALOHA 项目中，仅 50 条演示就能学会双臂精细操作。

**ACT 的劣势**:
- **分布覆盖有限**: CVAE 的隐空间容量有限，可能无法覆盖所有动作模态。
- **KL 坍塌风险**: 如果 $\beta$ 设置不当，模型可能忽略隐变量 $z$。

## 5. 实战代码示例 (Code Example)

```python
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerDecoder

class ACTPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        chunk_size: int = 100,
        z_dim: int = 32,
        hidden_dim: int = 512,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 7,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.z_dim = z_dim
        
        # Observation Encoder
        self.obs_encoder = nn.Linear(obs_dim, hidden_dim)
        
        # CVAE Encoder (用于训练)
        self.cvae_encoder = TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8),
            num_layers=num_encoder_layers
        )
        self.z_mu = nn.Linear(hidden_dim, z_dim)
        self.z_logvar = nn.Linear(hidden_dim, z_dim)
        
        # Action Decoder
        self.action_queries = nn.Parameter(torch.randn(chunk_size, hidden_dim))
        self.z_proj = nn.Linear(z_dim, hidden_dim)
        self.action_decoder = TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8),
            num_layers=num_decoder_layers
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)
        
        # Action embedding for CVAE encoder
        self.action_embed = nn.Linear(action_dim, hidden_dim)
    
    def encode(self, obs_feat, actions):
        """CVAE 编码器: 编码观测和动作到隐空间"""
        # actions: [B, k, action_dim]
        action_feat = self.action_embed(actions)  # [B, k, hidden_dim]
        
        # 拼接观测和动作
        combined = torch.cat([obs_feat, action_feat], dim=1)  # [B, T+1+k, D]
        combined = combined.permute(1, 0, 2)  # [T+1+k, B, D]
        
        encoded = self.cvae_encoder(combined)
        pooled = encoded.mean(dim=0)  # [B, D]
        
        return self.z_mu(pooled), self.z_logvar(pooled)
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, obs_feat, z):
        """动作解码器: 从观测和隐变量生成动作序列"""
        batch_size = obs_feat.shape[0]
        
        # Action queries + z
        queries = self.action_queries.unsqueeze(0).expand(batch_size, -1, -1)
        z_expanded = self.z_proj(z).unsqueeze(1)  # [B, 1, D]
        queries = queries + z_expanded  # 广播加法
        
        queries = queries.permute(1, 0, 2)  # [k, B, D]
        obs_feat = obs_feat.permute(1, 0, 2)  # [T+1, B, D]
        
        decoded = self.action_decoder(queries, obs_feat)
        decoded = decoded.permute(1, 0, 2)  # [B, k, D]
        
        return self.action_head(decoded)
    
    def forward(self, obs, actions=None):
        """
        训练时: obs + actions → z → pred_actions
        推理时: obs → sample z → pred_actions
        """
        obs_feat = self.obs_encoder(obs).unsqueeze(1)  # [B, 1, D]
        
        if actions is not None:  # 训练模式
            z_mu, z_logvar = self.encode(obs_feat, actions)
            z = self.reparameterize(z_mu, z_logvar)
            pred_actions = self.decode(obs_feat, z)
            return pred_actions, z_mu, z_logvar
        else:  # 推理模式
            z = torch.randn(obs.shape[0], self.z_dim, device=obs.device)
            pred_actions = self.decode(obs_feat, z)
            return pred_actions


def compute_loss(pred_actions, gt_actions, z_mu, z_logvar, beta=10.0):
    """ACT 损失函数: 重建损失 + β * KL 散度"""
    recon_loss = nn.functional.mse_loss(pred_actions, gt_actions)
    kl_loss = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss
```

## 6. 时间集成的实现 (Temporal Ensemble)

```python
class TemporalEnsemble:
    def __init__(self, chunk_size, action_dim, decay=0.01):
        self.chunk_size = chunk_size
        self.decay = decay
        self.action_buffer = []  # 存储历史预测
        self.weights = []        # 存储对应权重
    
    def update(self, new_chunk):
        """添加新预测的 chunk"""
        # new_chunk: [k, action_dim]
        self.action_buffer.append(new_chunk)
        self.weights.append(1.0)
        
        # 衰减旧权重
        self.weights = [w * (1 - self.decay) for w in self.weights]
        
        # 移除过期的预测
        while len(self.action_buffer) > self.chunk_size:
            self.action_buffer.pop(0)
            self.weights.pop(0)
    
    def get_action(self, t):
        """获取时刻 t 的集成动作"""
        weighted_sum = 0
        total_weight = 0
        
        for i, (chunk, w) in enumerate(zip(self.action_buffer, self.weights)):
            # 计算该 chunk 对时刻 t 的预测
            chunk_start_time = t - len(self.action_buffer) + i + 1
            local_idx = t - chunk_start_time
            
            if 0 <= local_idx < len(chunk):
                weighted_sum += w * chunk[local_idx]
                total_weight += w
        
        return weighted_sum / total_weight if total_weight > 0 else None
```

## 7. 面试常见问题 (Q&A)

**Q1: ACT 和 Diffusion Policy 的核心区别是什么?**

A: 
- **生成机制**: ACT 使用 **CVAE** (一次前向传播)，Diffusion 使用**迭代去噪** (10-100 步)。
- **速度**: ACT 推理更快，适合高频控制 (50Hz)；Diffusion 慢但分布覆盖更全。
- **实现复杂度**: ACT 更简单 (标准 VAE)；Diffusion 需要噪声调度器。

**Q2: 为什么 Action Chunking 能减少误差累积?**

A:
- 减少了**决策点数量**：chunk=100 时，原本 100 次决策变为 1 次。
- 模型需要**规划整个子任务**，而非盲目模仿，隐式学到了任务结构。
- 配合**时间集成**，单次预测误差被多次预测平均掉。

**Q3: CVAE 中的 β 参数如何调整?**

A:
- **β 过大**: KL 散度被过度惩罚，隐变量 $z$ 趋近于标准正态，模型退化为确定性输出 (**KL 坍塌**)。
- **β 过小**: 隐空间不规整，推理时采样的 $z$ 可能落在"空白区"，生成不合理的动作。
- **经验值**: ALOHA 项目中 $\beta = 10$ 效果最佳；可以使用 **β-VAE 退火** (从小到大逐渐增加)。

**Q4: 时间集成 (Temporal Ensemble) 的作用是什么?**

A:
- **平滑轨迹**: 消除相邻 chunk 之间的不连续。
- **提高鲁棒性**: 单次预测的错误被历史预测"稀释"。
- **权衡延迟 vs 平滑**: 衰减系数 $m$ 越大，响应越快但越不平滑。

**Q5: ACT 为什么在 ALOHA 项目中表现出色?**

A:
- **高频双臂协调**: 双臂操作需要 50Hz 精细控制，ACT 的快速推理至关重要。
- **数据效率**: CVAE 的隐空间提供了良好的归纳偏置，50 条演示即可泛化。
- **硬件友好**: 简单架构易于部署到边缘设备。

## 8. 参考资源 (References)

- **论文**: [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)
- **GitHub**: [ALOHA](https://github.com/tonyzhaozh/aloha)
- **项目主页**: [Mobile ALOHA](https://mobile-aloha.github.io/)
- **视频教程**: [ACT 论文精读](https://www.bilibili.com/video/BV1xxx) (B 站)

---
[← Back to Theory](./README.md)

