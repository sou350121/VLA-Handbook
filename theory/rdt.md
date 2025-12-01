# RDT: 机器人扩散变换器 (Robotics Diffusion Transformer)

> **核心论文**: [RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation](https://arxiv.org/abs/2410.07864) (Liu et al., 2024)
> **开发者**: 清华大学 MARS Lab & 字节跳动
> **代表模型**: **RDT-170M**, **RDT-1B**

## 1. 为什么需要 RDT? (Why?)

### 1.1 机器人学习的 Scaling Law 困境

在 NLP 和 CV 领域，**Scaling Law** 已被充分验证：模型越大、数据越多，性能越好。然而，机器人学习领域一直缺乏类似的验证:

- **数据稀缺**: 机器人数据收集成本高，数量级远低于图文数据。
- **分布多样**: 不同机器人 (单臂/双臂/人形) 的状态空间、动作空间差异巨大。
- **缺乏统一架构**: 之前的方法 (ACT, Diffusion Policy) 多为任务特定设计，难以规模化。

### 1.2 RDT 的核心目标

RDT 是首个**十亿参数级**的机器人扩散基础模型，专为解决以下问题：

1. **Scaling**: 证明机器人策略也能从"大模型+大数据"中获益。
2. **跨形态泛化 (Cross-Embodiment)**: 单一模型适配不同机器人 (单臂/双臂)。
3. **双臂操作 (Bimanual)**: 特别优化了需要两只手协调的复杂任务。

## 2. 核心技术 (Core Techniques)

### 2.1 可扩展的 DiT 架构 (Scalable Diffusion Transformer)

RDT 基于 **DiT (Diffusion Transformer)** 架构，将扩散过程与 Transformer 结合。

#### 2.1.1 为什么选择 DiT?

| 架构 | 优点 | 缺点 |
| :--- | :--- | :--- |
| **U-Net (Diffusion Policy)** | 成熟、稳定 | 难以规模化，参数效率低 |
| **DiT** | 与 LLM 架构统一，易于扩展 | 需要更多数据 |

DiT 的核心思想是将扩散模型的去噪网络从 U-Net 替换为 Transformer，这使得：
- 可以直接借鉴 LLM 的训练技术 (FlashAttention, Gradient Checkpointing)。
- 参数量可以平滑扩展到数十亿。

#### 2.1.2 RDT 的架构设计

```
                    ┌──────────────────────────────────────┐
                    │          Condition Encoder           │
                    │  (图像: SigLIP + 语言: T5)           │
                    └───────────────┬──────────────────────┘
                                    │ condition [B, L, D]
                                    ▼
┌──────────────┐    ┌──────────────────────────────────────┐
│  Noisy       │───▶│        DiT Backbone                  │
│  Action      │    │  (带 AdaLN-Zero 的 Transformer)      │
│  x_t         │    │                                      │
├──────────────┤    │  - 输入: x_t + t_emb + cond          │
│  Timestep t  │───▶│  - 输出: 预测噪声 ε 或 v             │
└──────────────┘    └───────────────┬──────────────────────┘
                                    │
                                    ▼
                         Denoised Action x_{t-1}
```

**关键组件**:

1. **AdaLN-Zero (Adaptive Layer Normalization)**:
   - 将时间步 $t$ 和条件信息注入到每个 Transformer Block。
   - 公式: $\text{AdaLN}(h, c) = c_s \odot \text{LayerNorm}(h) + c_b$
   - 其中 $c_s, c_b$ 是从条件 $c$ 预测的 scale 和 bias。

2. **统一的动作表示 (Unified Action Representation)**:
   - **单臂**: $(x, y, z, roll, pitch, yaw, gripper) \in \mathbb{R}^7$
   - **双臂**: 拼接两个单臂 + 可选的躯干自由度 $\in \mathbb{R}^{14+}$
   - 通过 **Padding + Masking** 统一不同形态的动作维度。

### 2.2 大规模预训练数据 (Pre-training Data)

RDT 在多个大规模机器人数据集上预训练：

| 数据集 | 类型 | 规模 | 特点 |
| :--- | :--- | :--- | :--- |
| **Open X-Embodiment** | 真机 | 1M+ episodes | 多种机器人形态 |
| **DROID** | 真机 | 76k episodes | 高质量双臂数据 |
| **RH20T** | 真机 | 20k episodes | 中国场景 |
| **RoboTurk** | 真机 | 2k episodes | 远程遥操 |
| **模拟数据** | 仿真 | 500k+ | RLBench, ManiSkill |

**数据配比策略**:
- **真机数据优先**: 真机数据占 70%，模拟数据占 30%。
- **双臂数据增强**: 对双臂数据进行上采样，弥补其稀缺性。

### 2.3 条件编码 (Condition Encoding)

RDT 使用**多模态条件**来指导动作生成：

```python
class RDTConditionEncoder(nn.Module):
    def __init__(self):
        # 视觉编码器: SigLIP (Google 的 CLIP 变体)
        self.vision_encoder = SigLIPVisionModel.from_pretrained("google/siglip-base")
        
        # 语言编码器: T5-XXL
        self.language_encoder = T5EncoderModel.from_pretrained("google/t5-xxl")
        
        # 本体感知投影
        self.proprio_proj = nn.Linear(proprio_dim, hidden_dim)
        
        # 融合层
        self.fusion = nn.TransformerEncoder(...)
    
    def forward(self, images, language, proprio):
        # images: [B, N_cam, C, H, W] - 多相机
        # language: [B, L_text] - 语言指令
        # proprio: [B, D_proprio] - 本体感知
        
        # 编码
        vis_feat = self.vision_encoder(images.flatten(0, 1))  # [B*N_cam, D]
        vis_feat = vis_feat.view(B, N_cam, -1)
        
        lang_feat = self.language_encoder(language).last_hidden_state
        
        proprio_feat = self.proprio_proj(proprio).unsqueeze(1)
        
        # 融合
        combined = torch.cat([vis_feat, lang_feat, proprio_feat], dim=1)
        return self.fusion(combined)
```

### 2.4 扩散训练与采样 (Diffusion Training & Sampling)

RDT 使用 **DDPM** 框架进行训练，但采用了几个优化:

#### 2.4.1 训练目标

$$
\mathcal{L} = \mathbb{E}_{t, a_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(a_t, t, c) \|^2 \right]
$$

其中:
- $a_0$: 真实动作序列 (Action Chunk)
- $a_t = \sqrt{\bar{\alpha}_t} a_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$: 加噪动作
- $c$: 条件 (图像 + 语言 + 本体感知)
- $\epsilon_\theta$: DiT 网络预测的噪声

#### 2.4.2 推理加速

RDT 使用 **DDIM** 将去噪步数从 100 步压缩到 **10 步**:

```python
@torch.no_grad()
def sample(self, condition, num_steps=10):
    """DDIM 采样"""
    batch_size = condition.shape[0]
    
    # 从纯噪声开始
    x_t = torch.randn(batch_size, self.chunk_size, self.action_dim)
    
    # DDIM 步长
    timesteps = torch.linspace(1000, 0, num_steps + 1).long()
    
    for i in range(num_steps):
        t = timesteps[i]
        t_prev = timesteps[i + 1]
        
        # 预测噪声
        eps_pred = self.dit(x_t, t, condition)
        
        # DDIM 更新
        alpha_t = self.alphas_cumprod[t]
        alpha_prev = self.alphas_cumprod[t_prev] if t_prev > 0 else 1.0
        
        x0_pred = (x_t - torch.sqrt(1 - alpha_t) * eps_pred) / torch.sqrt(alpha_t)
        x_t = torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(1 - alpha_prev) * eps_pred
    
    return x_t
```

## 3. 模型变体 (Model Variants)

RDT 提供了不同规模的模型以适应不同的算力需求：

| 模型 | 参数量 | 层数 | 隐藏维度 | 头数 | 推理速度 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RDT-170M** | 170M | 24 | 1024 | 16 | ~50ms/step |
| **RDT-1B** | 1.2B | 48 | 2048 | 32 | ~200ms/step |

**Scaling Law 验证**:
- RDT-1B 在所有任务上都优于 RDT-170M，验证了机器人领域的 Scaling Law。
- 性能提升呈现**对数线性**关系：参数量翻 10 倍，性能提升约 15-20%。

## 4. 关键创新点 (Key Innovations)

### 4.1 统一的跨形态表示 (Unified Cross-Embodiment Representation)

**问题**: 不同机器人的动作维度不同 (单臂 7D, 双臂 14D, 人形 30+D)。

**解决方案**: **Padded Action Space**

```python
# 统一到最大维度 (例如 32D)
MAX_ACTION_DIM = 32

def pad_action(action, embodiment_type):
    """将不同机器人的动作填充到统一维度"""
    if embodiment_type == "single_arm":
        # [7] -> [32] (后面填 0)
        padded = F.pad(action, (0, MAX_ACTION_DIM - 7))
        mask = torch.cat([torch.ones(7), torch.zeros(MAX_ACTION_DIM - 7)])
    elif embodiment_type == "bimanual":
        # [14] -> [32]
        padded = F.pad(action, (0, MAX_ACTION_DIM - 14))
        mask = torch.cat([torch.ones(14), torch.zeros(MAX_ACTION_DIM - 14)])
    # ...
    return padded, mask
```

训练时，只对有效维度计算损失:
$$
\mathcal{L} = \mathbb{E} \left[ \| m \odot (\epsilon - \hat{\epsilon}) \|^2 \right]
$$

### 4.2 双臂协调优化 (Bimanual Coordination)

双臂操作需要两只手**协同**工作（如折衣服、搬运大物体）。RDT 通过以下设计增强双臂协调:

1. **动作拼接**: 左右臂动作在序列维度拼接，而非独立预测。
2. **Cross-Arm Attention**: 在 Transformer 中，左臂的 Token 可以 Attend 到右臂。
3. **对称性数据增强**: 随机交换左右臂的输入/输出，增加数据多样性。

### 4.3 高效微调 (Efficient Fine-tuning)

预训练的 RDT 可以通过 **LoRA** 高效适配到新任务:

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,                    # LoRA 秩
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # 只微调 QV 投影
    lora_dropout=0.05,
)

rdt_lora = get_peft_model(rdt_model, config)
# 可训练参数: ~10M (vs 1.2B 全量)
```

**微调数据需求**:
- **10-50 条演示**: 基础适配
- **100-500 条演示**: 高性能微调
- **对比**: 从头训练 Diffusion Policy 需要 1000+ 条

## 5. 实验结果 (Experimental Results)

### 5.1 Benchmark 性能

| 任务 | ACT | Diffusion Policy | OpenVLA | RDT-1B |
| :--- | :--- | :--- | :--- | :--- |
| **LIBERO-90** | 68.2% | 73.5% | 75.1% | **82.3%** |
| **Bimanual Fold** | 45% | 52% | 48% | **71%** |
| **Pick & Place** | 82% | 85% | 87% | **93%** |
| **长序列任务** | 35% | 42% | 40% | **58%** |

### 5.2 泛化能力

RDT 在**未见过的场景**上也展现出良好的泛化:
- **新物体**: 成功率下降 < 10%
- **新背景**: 成功率下降 < 5%
- **新机器人**: 零样本迁移成功率 40%+

## 6. 与其他方法的对比 (Comparison)

| 方法 | 架构 | 参数量 | 预训练 | 双臂支持 | 推理速度 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ACT** | CVAE + Transformer | ~10M | 无 | ✅ | 最快 |
| **Diffusion Policy** | U-Net | ~50M | 无 | ❌ | 慢 |
| **OpenVLA** | LLM + Action Head | 7B | ✅ | ❌ | 慢 |
| **π0** | VLM + Flow Matching | 3B+ | ✅ | ✅ | 中等 |
| **RDT-1B** | DiT | 1.2B | ✅ | ✅ (专门优化) | 中等 |

**RDT 的独特优势**:
- **专为双臂设计**: 动作空间和注意力机制都针对双臂优化。
- **验证 Scaling Law**: 首个证明机器人领域 "bigger is better" 的模型。
- **开源友好**: 提供完整的预训练权重和微调代码。

## 7. 实战代码示例 (Code Example)

```python
import torch
from transformers import AutoModel

# 加载预训练的 RDT-1B
model = AutoModel.from_pretrained("thu-ml/RDT-1B")

# 准备输入
images = torch.randn(1, 2, 3, 224, 224)  # 双相机
language = "fold the towel in half"
proprio = torch.randn(1, 14)  # 双臂本体感知

# 推理
with torch.no_grad():
    condition = model.encode_condition(images, language, proprio)
    actions = model.sample(condition, num_steps=10)  # [1, chunk_size, 14]

print(f"Generated actions shape: {actions.shape}")
# 输出: Generated actions shape: torch.Size([1, 64, 14])
```

### 7.1 LoRA 微调示例

```python
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader

# 配置 LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    lora_dropout=0.05,
)

model_lora = get_peft_model(model, lora_config)
print(f"Trainable params: {model_lora.num_parameters(only_trainable=True):,}")
# 输出: Trainable params: 12,582,912

# 微调循环
optimizer = torch.optim.AdamW(model_lora.parameters(), lr=1e-4)

for batch in dataloader:
    images, language, proprio, gt_actions = batch
    
    # 前向传播
    loss = model_lora.compute_loss(images, language, proprio, gt_actions)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## 8. 面试常见问题 (Q&A)

**Q1: RDT 和 Diffusion Policy 的核心区别是什么?**

A:
- **架构**: RDT 使用 **DiT (Transformer)**，Diffusion Policy 使用 **U-Net**。
- **规模**: RDT 是十亿参数级基础模型，Diffusion Policy 通常 < 100M。
- **预训练**: RDT 在大规模多机器人数据上预训练，Diffusion Policy 通常从头训练。
- **跨形态**: RDT 统一处理不同机器人，Diffusion Policy 需为每种形态单独训练。

**Q2: 为什么 RDT 专门优化双臂操作?**

A:
- **数据稀缺**: 双臂数据远少于单臂，需要专门的数据增强。
- **协调性**: 双臂任务需要两只手**同步配合**，不能独立预测。
- **工业需求**: 双臂机器人是工业和家庭场景的主要形态。

**Q3: RDT 的 Scaling Law 是如何验证的?**

A:
- 训练了 **170M, 500M, 1B** 三个规模的模型。
- 保持数据和训练步数相同，只改变模型大小。
- 结果: **参数量 ↑ 10x → 成功率 ↑ 15-20%**，呈对数线性关系。

**Q4: RDT 如何处理不同机器人的动作维度?**

A:
- **统一填充**: 所有动作填充到最大维度 (如 32D)。
- **动态掩码**: 训练和推理时，只对有效维度计算损失/输出。
- **Embodiment Token**: 可选地添加"机器人类型"嵌入。

**Q5: RDT vs π0，哪个更好?**

A:
- **RDT 优势**: 专门优化双臂，开源完整，Scaling Law 验证充分。
- **π0 优势**: VLM 底座更强 (语义理解)，Flow Matching 比 DDPM 更快。
- **选择建议**: 双臂任务选 RDT；需要强语义理解选 π0。

## 9. 局限性与未来方向 (Limitations & Future)

### 9.1 当前局限

- **推理速度**: 10 步 DDIM 仍需 ~100ms，难以满足某些高频控制需求。
- **数据需求**: 预训练需要大规模数据，新形态机器人可能缺乏数据。
- **安全性**: 扩散模型的随机性可能导致不安全动作。

### 9.2 未来方向

- **一致性蒸馏 (Consistency Distillation)**: 将去噪步数压缩到 1 步。
- **人形机器人**: 扩展到 30+ 自由度的人形机器人。
- **在线学习**: 结合 RL 进行实时适应。

## 10. 参考资源 (References)

- **论文**: [RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation](https://arxiv.org/abs/2410.07864)
- **GitHub**: [thu-ml/RDT-1B](https://github.com/thu-ml/RoboticsDiffusionTransformer)
- **Hugging Face**: [thu-ml/RDT-1B](https://huggingface.co/thu-ml/RDT-1B)
- **博客**: [清华 MARS Lab 技术博客](https://mars-lab.github.io/)

---
[← Back to Theory](./README.md)

