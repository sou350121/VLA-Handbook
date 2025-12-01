# 迁移学习 (Transfer Learning)

> **核心概念**: 迁移学习 (Transfer Learning) 是将在**源域 (Source Domain)** 学到的知识应用到**目标域 (Target Domain)** 的技术。在 VLA 领域，迁移学习是实现跨机器人、跨场景泛化的关键。

## 1. 为什么 VLA 需要迁移学习? (Why Transfer Learning?)

### 1.1 机器人学习的迁移挑战

| 迁移类型 | 源域 | 目标域 | 挑战 |
| :--- | :--- | :--- | :--- |
| **跨形态 (Cross-Embodiment)** | 单臂机器人 | 双臂机器人 | 动作空间不同 |
| **跨场景 (Cross-Environment)** | 实验室 | 真实家庭 | 视觉分布偏移 |
| **跨任务 (Cross-Task)** | 抓取物体 | 折叠衣物 | 技能差异大 |
| **仿真到真实 (Sim-to-Real)** | 仿真环境 | 真机 | 物理差异 |

### 1.2 迁移学习的价值

$$
\text{数据收集成本} = \frac{\text{所需数据量}}{\text{数据收集效率}} \propto \frac{1}{\text{迁移能力}}
$$

- **减少数据需求**: 预训练模型只需少量目标域数据微调
- **提高泛化能力**: 学习跨域不变的特征表示
- **加速部署**: 新机器人/场景无需从头训练

## 2. 迁移学习范式 (Transfer Learning Paradigms)

### 2.1 预训练-微调 (Pre-training + Fine-tuning)

```
┌─────────────────────────────────────────────────────────────┐
│                    Phase 1: Pre-training                    │
│                                                             │
│   大规模数据 (ImageNet/CLIP/OXE)                            │
│              │                                              │
│              ▼                                              │
│        预训练模型 (通用特征)                                  │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    Phase 2: Fine-tuning                     │
│                                                             │
│   目标域少量数据 (50-500 episodes)                           │
│              │                                              │
│              ▼                                              │
│        微调后模型 (目标任务特定)                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 冻结特征提取 (Feature Extraction)

只训练任务头，冻结预训练的 backbone。

```python
class FeatureExtraction(nn.Module):
    def __init__(self, pretrained_encoder, action_dim):
        super().__init__()
        self.encoder = pretrained_encoder
        
        # 冻结编码器
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # 只训练策略头
        self.policy_head = nn.Sequential(
            nn.Linear(encoder.output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    
    def forward(self, obs):
        with torch.no_grad():
            features = self.encoder(obs)
        action = self.policy_head(features)
        return action
```

**适用**: 目标域数据极少 (< 50 episodes)

### 2.3 全量微调 (Full Fine-tuning)

解冻所有参数进行训练。

```python
class FullFineTuning(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.model = pretrained_model
        
        # 所有参数可训练
        for param in self.model.parameters():
            param.requires_grad = True
    
    def forward(self, obs):
        return self.model(obs)
```

**适用**: 目标域数据充足，且与源域差异较大

### 2.4 参数高效微调 (Parameter-Efficient Fine-Tuning, PEFT)

只训练少量新增参数即可适配新任务，常见工具包括 LoRA、Prompt Tuning、Adapter 等。

LoRA 通过在投影层中串接低秩增量（$W' = W_0 + BA$）来模拟微调效果，大幅减少可训练参数、减轻显存压力，还可以为不同任务保存多个适配器。

详细的 LoRA 数学推导与实践示例请参考 **[theory/peft_lora.md](./peft_lora.md)**（该文档也是 VLA 中 PEFT 的权威参考），这里只保留高层总结和经验值。

## 3. 跨形态迁移 (Cross-Embodiment Transfer)

### 3.1 动作空间对齐

不同机器人的动作空间差异是跨形态迁移的核心挑战。

```python
class ActionSpaceAdapter(nn.Module):
    """将源域动作映射到目标域"""
    def __init__(self, source_action_dim, target_action_dim):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(source_action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, target_action_dim)
        )
    
    def forward(self, source_action):
        return self.adapter(source_action)
```

### 3.2 统一动作表示

**RDT/OpenVLA 的方案**: 统一填充到最大维度

```python
def unify_action_space(action, embodiment_type, max_dim=32):
    """统一不同机器人的动作空间"""
    action_mapping = {
        "franka_7dof": 7,
        "ur5_6dof": 6,
        "bimanual_14dof": 14,
        "mobile_3dof": 3
    }
    
    original_dim = action_mapping[embodiment_type]
    
    # 填充到统一维度
    padded_action = F.pad(action, (0, max_dim - original_dim))
    
    # 创建有效维度 mask
    mask = torch.zeros(max_dim)
    mask[:original_dim] = 1
    
    return padded_action, mask
```

### 3.3 形态无关特征学习

```python
class EmbodimentInvariantEncoder(nn.Module):
    """学习与机器人形态无关的特征"""
    def __init__(self, obs_encoder, action_encoder):
        super().__init__()
        self.obs_encoder = obs_encoder
        self.action_encoder = action_encoder
        
        # 对抗学习: 分类器试图区分形态
        self.embodiment_classifier = nn.Linear(hidden_dim, num_embodiments)
        self.gradient_reversal = GradientReversalLayer()
    
    def forward(self, obs, action, embodiment_label):
        # 编码观测
        obs_feat = self.obs_encoder(obs)
        
        # 对抗训练: 让特征无法区分形态
        reversed_feat = self.gradient_reversal(obs_feat)
        embodiment_pred = self.embodiment_classifier(reversed_feat)
        adversarial_loss = F.cross_entropy(embodiment_pred, embodiment_label)
        
        return obs_feat, adversarial_loss
```

## 4. 仿真到真实迁移 (Sim-to-Real Transfer)

### 4.1 Domain Randomization (域随机化)

在仿真中随机化各种参数，让模型对变化鲁棒。

```python
class DomainRandomization:
    """仿真环境的域随机化"""
    
    @staticmethod
    def visual_randomization(image):
        """视觉随机化"""
        transforms = [
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            T.GaussianBlur(kernel_size=5),
            T.RandomAdjustSharpness(sharpness_factor=2),
        ]
        for t in transforms:
            if random.random() > 0.5:
                image = t(image)
        return image
    
    @staticmethod
    def physics_randomization(env):
        """物理参数随机化"""
        # 摩擦系数
        env.set_friction(random.uniform(0.5, 1.5))
        # 物体质量
        env.set_object_mass(random.uniform(0.8, 1.2) * default_mass)
        # 电机延迟
        env.set_actuator_delay(random.uniform(0, 0.05))
        return env
    
    @staticmethod
    def camera_randomization(env):
        """相机参数随机化"""
        # 相机位置扰动
        env.camera_pos += np.random.uniform(-0.02, 0.02, size=3)
        # FOV 变化
        env.camera_fov = random.uniform(55, 65)
        return env
```

### 4.2 System Identification (系统辨识)

```python
class SystemIdentification(nn.Module):
    """从真实数据推断仿真参数"""
    def __init__(self):
        self.encoder = nn.LSTM(obs_dim, hidden_dim)
        self.param_predictor = nn.Linear(hidden_dim, physics_param_dim)
    
    def forward(self, trajectory):
        """
        trajectory: [T, obs_dim] - 真实轨迹
        """
        _, (h_n, _) = self.encoder(trajectory)
        predicted_params = self.param_predictor(h_n.squeeze())
        return predicted_params  # e.g., 摩擦系数、质量等
```

### 4.3 Real-to-Sim-to-Real

```
真实数据 (少量)
    │
    ▼ System Identification
仿真参数校准
    │
    ▼ 大量仿真数据生成
    │
    ▼ 仿真训练
    │
    ▼ Sim-to-Real Fine-tuning
真机部署
```

## 5. 域适应 (Domain Adaptation)

### 5.1 对抗域适应 (Adversarial Domain Adaptation)

```python
class DANN(nn.Module):
    """Domain-Adversarial Neural Network"""
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.task_classifier = TaskClassifier()
        self.domain_classifier = DomainClassifier()
    
    def forward(self, source_data, target_data, alpha=1.0):
        # 源域特征
        source_feat = self.feature_extractor(source_data)
        source_task_pred = self.task_classifier(source_feat)
        
        # 目标域特征
        target_feat = self.feature_extractor(target_data)
        
        # 域分类器 (通过梯度反转对抗训练)
        source_domain = self.domain_classifier(
            GradientReversal.apply(source_feat, alpha)
        )
        target_domain = self.domain_classifier(
            GradientReversal.apply(target_feat, alpha)
        )
        
        # 域分类损失: 让特征无法区分来自哪个域
        domain_loss = F.binary_cross_entropy(
            torch.cat([source_domain, target_domain]),
            torch.cat([torch.zeros_like(source_domain), 
                       torch.ones_like(target_domain)])
        )
        
        return source_task_pred, domain_loss


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        # 反转梯度
        return -ctx.alpha * grad_output, None
```

### 5.2 最大均值差异 (MMD)

```python
def mmd_loss(source_features, target_features, kernel='rbf'):
    """Maximum Mean Discrepancy 损失"""
    def rbf_kernel(x, y, sigma=1.0):
        dist = torch.cdist(x, y) ** 2
        return torch.exp(-dist / (2 * sigma ** 2))
    
    K_ss = rbf_kernel(source_features, source_features)
    K_tt = rbf_kernel(target_features, target_features)
    K_st = rbf_kernel(source_features, target_features)
    
    mmd = K_ss.mean() + K_tt.mean() - 2 * K_st.mean()
    return mmd
```

## 6. 零样本/少样本迁移 (Zero/Few-Shot Transfer)

### 6.1 语言引导的零样本迁移

```python
class LanguageGuidedTransfer(nn.Module):
    """通过语言描述实现零样本迁移"""
    def __init__(self, vlm_backbone):
        super().__init__()
        self.vlm = vlm_backbone
        
    def forward(self, image, task_description):
        """
        task_description: "pick up the red cup and place it on the tray"
        无需该任务的训练数据，依靠 VLM 的语义理解
        """
        # VLM 直接理解新任务
        action = self.vlm.generate_action(image, task_description)
        return action
```

### 6.2 少样本学习策略

```python
class FewShotPolicy(nn.Module):
    """少样本学习策略"""
    def __init__(self, base_policy, adaptation_steps=5):
        super().__init__()
        self.base_policy = base_policy
        self.adaptation_steps = adaptation_steps
    
    def adapt(self, support_set, lr=0.01):
        """在少量支持样本上快速适应"""
        # 复制参数
        adapted_params = {k: v.clone() for k, v in self.base_policy.named_parameters()}
        
        for _ in range(self.adaptation_steps):
            # 计算支持集上的损失
            loss = 0
            for obs, action in support_set:
                pred_action = self.base_policy(obs)
                loss += F.mse_loss(pred_action, action)
            loss /= len(support_set)
            
            # 梯度更新
            grads = torch.autograd.grad(loss, adapted_params.values())
            adapted_params = {
                k: v - lr * g 
                for (k, v), g in zip(adapted_params.items(), grads)
            }
        
        return adapted_params
```

## 7. VLA 中的迁移学习实践 (Practical Transfer in VLA)

### 7.1 OpenVLA 的迁移流程

```python
# 1. 加载预训练模型
from transformers import AutoModelForVision2Seq
model = AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b")

# 2. 配置 LoRA
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
)
model = get_peft_model(model, lora_config)

# 3. 在目标数据上微调
for batch in target_dataloader:
    loss = model.compute_loss(**batch)
    loss.backward()
    optimizer.step()

# 4. 保存适配器
model.save_pretrained("./openvla-adapted-myrobot")
```

### 7.2 迁移效果对比

| 方法 | 目标域数据量 | 成功率 | 训练时间 |
| :--- | :--- | :--- | :--- |
| 从头训练 | 1000 episodes | 65% | 10 hours |
| 全量微调 | 100 episodes | 72% | 2 hours |
| LoRA 微调 | 50 episodes | **78%** | **30 min** |
| 冻结+策略头 | 20 episodes | 60% | 10 min |

## 8. 面试高频问题 (Q&A)

**Q1: 迁移学习和域适应的区别是什么?**

A:
- **迁移学习**: 广义概念，任何利用源域知识帮助目标域学习的方法
- **域适应**: 迁移学习的子类，专注于解决源域和目标域**分布不同**的问题
- **关系**: 域适应是实现迁移学习的一种具体技术

**Q2: 为什么 LoRA 在 VLA 中效果好?**

A:
- **数据效率**: 机器人数据稀缺，少量参数更不易过拟合
- **知识保留**: VLM 的预训练知识通过冻结主干被保护
- **多任务适配**: 可以为不同任务/机器人保存独立的 LoRA 适配器
- **部署高效**: 推理时可以合并 LoRA 权重，无额外开销

**Q3: Sim-to-Real 中 Domain Randomization 的局限性?**

A:
- **参数敏感**: 随机化范围需要精心调整，过大会降低性能
- **无法覆盖所有差异**: 有些真实世界的复杂性难以在仿真中建模
- **训练效率**: 极端随机化可能导致学习困难
- **改进方案**: 结合 System Identification 或 Real-to-Sim

**Q4: 跨形态迁移的核心挑战是什么?**

A:
- **动作空间不一致**: 不同机器人 DoF 不同
- **观测空间差异**: 相机位置、分辨率不同
- **动力学差异**: 不同机器人的运动特性不同
- **解决方案**: 统一动作表示 + 形态无关特征学习 + 对抗训练

**Q5: 如何判断是否需要全量微调 vs LoRA?**

A:
- **LoRA 适用**: 目标域与源域相似，数据量少 (< 500 episodes)
- **全量微调适用**: 目标域差异大，有足够数据 (> 1000 episodes)
- **经验法则**: 先尝试 LoRA，效果不佳再考虑全量微调

## 9. 参考资源 (References)

- **LoRA**: [Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **Domain Randomization**: [Domain Randomization for Transferring Deep Neural Networks](https://arxiv.org/abs/1703.06907)
- **DANN**: [Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1505.07818)
- **Open X-Embodiment**: [Open X-Embodiment: Robotic Learning Datasets and RT-X Models](https://arxiv.org/abs/2310.08864)

---
[← Back to Theory](./README.md)

