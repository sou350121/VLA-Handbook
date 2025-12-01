# 知识蒸馏 (Knowledge Distillation)

> **核心概念**: 知识蒸馏 (Knowledge Distillation, KD) 是一种模型压缩技术，通过让小模型（学生）模仿大模型（教师）的行为，实现知识的迁移。在 VLA 领域，知识蒸馏是实现边缘端部署的关键技术。

## 1. 为什么 VLA 需要知识蒸馏? (Why KD for VLA?)

### 1.1 VLA 部署挑战

| 模型 | 参数量 | 推理延迟 | 显存需求 |
| :--- | :--- | :--- | :--- |
| **RT-2 (PaLI-X)** | 55B | ~5s | 100+ GB |
| **OpenVLA** | 7B | ~200ms | 16 GB |
| **目标 (边缘)** | < 1B | < 50ms | < 4 GB |

**现实约束**:
- **实时性**: 机器人控制需要 20-50Hz (20-50ms)
- **硬件限制**: 机载计算通常只有 Jetson Orin (8-32GB)
- **功耗**: 移动机器人对功耗敏感

### 1.2 知识蒸馏的价值

$$
\text{小模型性能} + \text{大模型知识} \approx \text{大模型性能}
$$

- **模型压缩**: 10x 参数减少，性能损失 < 10%
- **推理加速**: 降低延迟，满足实时控制需求
- **降低成本**: 减少部署硬件需求

## 2. 知识蒸馏基础 (KD Fundamentals)

### 2.1 基本框架

```
┌─────────────────────────────────────────────────────────────┐
│                    Teacher Model (大模型)                    │
│                    - 预训练好的 VLA                          │
│                    - 参数冻结                               │
└──────────────────────────┬──────────────────────────────────┘
                           │ Soft Labels / Logits
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     Distillation Loss                       │
│              L = α * L_hard + (1-α) * L_soft                │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Student Model (小模型)                     │
│                    - 目标部署模型                            │
│                    - 可训练                                 │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 软标签 (Soft Labels)

**硬标签 (Hard Labels)**: One-hot 向量，只有正确类别为 1。

**软标签 (Soft Labels)**: 教师模型的 softmax 概率分布。

$$
p_i^{(T)} = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

其中 $T$ 是**温度参数** (Temperature)：
- **T = 1**: 正常 softmax
- **T > 1**: 分布更平滑，保留更多"暗知识"
- **T → ∞**: 均匀分布

### 2.3 蒸馏损失 (Distillation Loss)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
    
    def forward(self, student_logits, teacher_logits, labels):
        """
        student_logits: 学生模型输出
        teacher_logits: 教师模型输出
        labels: 真实标签 (用于硬标签损失)
        """
        # 硬标签损失 (与真实标签)
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # 软标签损失 (与教师)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
        soft_loss = soft_loss * (self.temperature ** 2)  # 温度缩放
        
        # 组合损失
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        return total_loss
```

## 3. VLA 中的知识蒸馏策略 (KD Strategies for VLA)

### 3.1 特征蒸馏 (Feature Distillation)

除了输出层，还蒸馏中间特征。

```python
class FeatureDistillation(nn.Module):
    def __init__(self, teacher, student, feature_layers):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.feature_layers = feature_layers
        
        # 特征投影层 (对齐维度)
        self.projectors = nn.ModuleDict()
        for name in feature_layers:
            t_dim = teacher.get_feature_dim(name)
            s_dim = student.get_feature_dim(name)
            if t_dim != s_dim:
                self.projectors[name] = nn.Linear(s_dim, t_dim)
    
    def forward(self, inputs):
        # 获取教师特征
        with torch.no_grad():
            teacher_features = self.teacher.get_features(inputs, self.feature_layers)
        
        # 获取学生特征
        student_features = self.student.get_features(inputs, self.feature_layers)
        
        # 特征对齐损失
        feature_loss = 0
        for name in self.feature_layers:
            t_feat = teacher_features[name]
            s_feat = student_features[name]
            
            if name in self.projectors:
                s_feat = self.projectors[name](s_feat)
            
            feature_loss += F.mse_loss(s_feat, t_feat)
        
        return feature_loss / len(self.feature_layers)
```

### 3.2 注意力蒸馏 (Attention Distillation)

让学生模型学习教师的注意力模式。

```python
class AttentionDistillation(nn.Module):
    def __init__(self, teacher, student):
        super().__init__()
        self.teacher = teacher
        self.student = student
    
    def forward(self, inputs):
        # 获取注意力权重
        with torch.no_grad():
            _, teacher_attns = self.teacher(inputs, output_attentions=True)
        
        _, student_attns = self.student(inputs, output_attentions=True)
        
        # 注意力对齐损失
        attn_loss = 0
        for t_attn, s_attn in zip(teacher_attns, student_attns):
            # t_attn, s_attn: [B, num_heads, seq_len, seq_len]
            attn_loss += F.mse_loss(s_attn, t_attn)
        
        return attn_loss / len(teacher_attns)
```

### 3.3 动作轨迹蒸馏 (Action Trajectory Distillation)

VLA 特有：蒸馏整个动作序列。

```python
class ActionTrajectoryDistillation(nn.Module):
    """VLA 专用: 蒸馏动作轨迹"""
    def __init__(self, teacher_vla, student_vla, chunk_size=16):
        super().__init__()
        self.teacher = teacher_vla
        self.student = student_vla
        self.chunk_size = chunk_size
    
    def forward(self, obs, instruction):
        """
        obs: 观测 [B, C, H, W]
        instruction: 语言指令
        """
        # 教师生成动作轨迹
        with torch.no_grad():
            teacher_actions = self.teacher.generate_actions(obs, instruction)
            # teacher_actions: [B, chunk_size, action_dim]
        
        # 学生生成动作轨迹
        student_actions = self.student.generate_actions(obs, instruction)
        
        # 轨迹级别的蒸馏损失
        trajectory_loss = F.mse_loss(student_actions, teacher_actions)
        
        # 可选: 时间加权 (更重视近期动作)
        time_weights = torch.exp(-0.1 * torch.arange(self.chunk_size))
        weighted_loss = (trajectory_loss * time_weights.unsqueeze(-1)).mean()
        
        return weighted_loss
```

### 3.4 分布蒸馏 (Distribution Distillation)

对于 Diffusion/Flow Matching 策略，蒸馏动作分布。

```python
class DiffusionDistillation(nn.Module):
    """Diffusion Policy 的蒸馏"""
    def __init__(self, teacher_diffusion, student_diffusion, num_steps=10):
        super().__init__()
        self.teacher = teacher_diffusion
        self.student = student_diffusion
        self.num_steps = num_steps
    
    def forward(self, obs, condition):
        # 教师预测的噪声
        with torch.no_grad():
            x_t = torch.randn(obs.shape[0], self.action_dim)
            for t in reversed(range(self.num_steps)):
                teacher_eps = self.teacher.predict_noise(x_t, t, condition)
                x_t = self.ddim_step(x_t, teacher_eps, t)
            teacher_trajectory = x_t
        
        # 学生预测 (直接预测最终轨迹，一步到位)
        student_trajectory = self.student(obs, condition)
        
        # 蒸馏损失
        loss = F.mse_loss(student_trajectory, teacher_trajectory)
        return loss
```

## 4. VLA 蒸馏的特殊考虑 (Special Considerations)

### 4.1 视觉编码器蒸馏

```python
class VisionEncoderDistillation:
    """蒸馏视觉编码器: SigLIP (L) → MobileViT (S)"""
    
    def __init__(self, teacher_vision, student_vision):
        self.teacher = teacher_vision  # SigLIP-L (400M)
        self.student = student_vision  # MobileViT (6M)
        
        # CLS token 对齐
        self.cls_projector = nn.Linear(student.embed_dim, teacher.embed_dim)
        
        # Patch token 对齐
        self.patch_projector = nn.Conv1d(student.embed_dim, teacher.embed_dim, 1)
    
    def distill(self, images):
        with torch.no_grad():
            t_cls, t_patches = self.teacher(images)
        
        s_cls, s_patches = self.student(images)
        
        # CLS 损失
        cls_loss = F.mse_loss(self.cls_projector(s_cls), t_cls)
        
        # Patch 损失 (需要处理分辨率不匹配)
        t_patches_interp = F.interpolate(t_patches, size=s_patches.shape[-1])
        patch_loss = F.mse_loss(self.patch_projector(s_patches), t_patches_interp)
        
        return cls_loss + 0.5 * patch_loss
```

### 4.2 语言模型蒸馏

```python
class LLMDistillation:
    """蒸馏语言模型: Llama-7B → TinyLlama-1B"""
    
    def __init__(self, teacher_llm, student_llm):
        self.teacher = teacher_llm
        self.student = student_llm
    
    def distill(self, input_ids, attention_mask):
        # 教师输出
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        
        # 学生输出
        student_outputs = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Logit 蒸馏
        logit_loss = self.soft_cross_entropy(
            student_outputs.logits,
            teacher_outputs.logits,
            temperature=2.0
        )
        
        # 隐藏层蒸馏 (跳层对齐)
        # Teacher 32 层 → Student 12 层, 每隔 ~2.7 层对齐
        layer_mapping = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 31]
        hidden_loss = 0
        for s_idx, t_idx in enumerate(layer_mapping):
            t_hidden = teacher_outputs.hidden_states[t_idx]
            s_hidden = student_outputs.hidden_states[s_idx]
            hidden_loss += F.mse_loss(s_hidden, t_hidden)
        
        return logit_loss + 0.1 * hidden_loss
```

### 4.3 多阶段蒸馏 (Progressive Distillation)

```python
class ProgressiveDistillation:
    """渐进式蒸馏: 逐步缩小模型"""
    
    def __init__(self, teacher, intermediate_sizes=[4B, 2B, 1B]):
        self.teacher = teacher
        self.stages = []
        
        prev_model = teacher
        for size in intermediate_sizes:
            student = create_model(size)
            self.stages.append((prev_model, student))
            prev_model = student
    
    def train(self, data):
        for stage_idx, (teacher, student) in enumerate(self.stages):
            print(f"Stage {stage_idx + 1}: {teacher.num_params}B → {student.num_params}B")
            
            for epoch in range(num_epochs):
                for batch in data:
                    loss = self.distill_step(teacher, student, batch)
                    loss.backward()
                    optimizer.step()
            
            # 冻结当前学生，作为下一阶段的教师
            for param in student.parameters():
                param.requires_grad = False
```

## 5. 蒸馏 + 量化联合优化 (KD + Quantization)

### 5.1 量化感知蒸馏 (Quantization-Aware Distillation)

```python
class QADistillation:
    """量化感知蒸馏"""
    
    def __init__(self, teacher, student, quant_bits=4):
        self.teacher = teacher
        self.student = student
        self.quant_bits = quant_bits
    
    def forward(self, inputs):
        # 教师 (FP16)
        with torch.no_grad():
            teacher_out = self.teacher(inputs)
        
        # 学生 (模拟量化)
        student_out = self.quantized_forward(self.student, inputs)
        
        # 蒸馏损失
        loss = F.mse_loss(student_out, teacher_out)
        return loss
    
    def quantized_forward(self, model, inputs):
        """模拟 INT4 量化的前向传播"""
        # 伪量化: 模拟量化误差，但保持梯度可传播
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # 量化权重
                w = module.weight
                scale = w.abs().max() / (2 ** (self.quant_bits - 1) - 1)
                w_quant = torch.round(w / scale) * scale
                
                # Straight-Through Estimator
                module.weight.data = w + (w_quant - w).detach()
        
        return model(inputs)
```

### 5.2 蒸馏后量化 (Post-Distillation Quantization)

```python
def post_distillation_quantize(distilled_model, calibration_data, bits=4):
    """蒸馏后进行 PTQ"""
    from transformers import BitsAndBytesConfig
    
    # 收集激活统计
    activations = collect_activations(distilled_model, calibration_data)
    
    # 计算量化参数
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",  # NormalFloat4
    )
    
    # 量化模型
    quantized_model = quantize_model(distilled_model, quant_config)
    
    return quantized_model
```

## 6. 实战：OpenVLA 蒸馏到 1B 模型

```python
class OpenVLADistillation:
    """将 OpenVLA (7B) 蒸馏到 1B 模型"""
    
    def __init__(self):
        # 教师: OpenVLA 7B
        self.teacher = AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b")
        
        # 学生: TinyLlama 1B + MobileViT
        self.student = TinyVLA(
            vision_encoder="mobilevit_small",
            language_model="TinyLlama/TinyLlama-1.1B",
            action_head_dim=7
        )
        
        # 蒸馏配置
        self.temperature = 4.0
        self.alpha = 0.7  # 软标签权重
    
    def train_step(self, batch):
        images = batch['images']
        instructions = batch['instructions']
        gt_actions = batch['actions']
        
        # 教师预测
        with torch.no_grad():
            teacher_actions, teacher_logits = self.teacher(
                images, instructions, return_logits=True
            )
        
        # 学生预测
        student_actions, student_logits = self.student(
            images, instructions, return_logits=True
        )
        
        # 多任务蒸馏损失
        
        # 1. 动作蒸馏
        action_distill = F.mse_loss(student_actions, teacher_actions)
        
        # 2. Logit 蒸馏 (如果有语言输出)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # 3. 硬标签损失 (与真实动作)
        hard_loss = F.mse_loss(student_actions, gt_actions)
        
        # 组合
        total_loss = (
            self.alpha * soft_loss + 
            (1 - self.alpha) * hard_loss +
            0.5 * action_distill
        )
        
        return total_loss


# 训练脚本
distiller = OpenVLADistillation()
optimizer = torch.optim.AdamW(distiller.student.parameters(), lr=1e-4)

for epoch in range(10):
    for batch in dataloader:
        loss = distiller.train_step(batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 保存蒸馏后的小模型
distiller.student.save_pretrained("./openvla-1b-distilled")
```

## 7. 效果对比 (Performance Comparison)

| 模型 | 参数量 | 延迟 (Jetson) | CALVIN 成功率 | 显存 |
| :--- | :--- | :--- | :--- | :--- |
| **OpenVLA (Teacher)** | 7B | 500ms | 78% | 16GB |
| **OpenVLA-1B (Distilled)** | 1B | 80ms | 72% | 4GB |
| **OpenVLA-1B + INT4** | 1B | 50ms | 70% | 2GB |
| **从头训练 1B** | 1B | 80ms | 58% | 4GB |

**结论**: 蒸馏后的 1B 模型比从头训练提升 14%！

## 8. 面试高频问题 (Q&A)

**Q1: 温度参数 T 的作用是什么? 如何选择?**

A:
- **T 的作用**: 软化 softmax 分布，保留更多"暗知识"（如相似类的关系）
- **T 小 (1-2)**: 分布尖锐，信息主要来自 top-1 预测
- **T 大 (4-10)**: 分布平滑，保留更多类间关系
- **经验值**: VLA 中通常 T=4-6

**Q2: 为什么蒸馏比从头训练小模型效果好?**

A:
- **软标签信息**: 包含类间关系（如"红色苹果"和"绿色苹果"更相似）
- **正则化效果**: 软标签比 one-hot 更平滑，防止过拟合
- **数据效率**: 小模型难以从有限数据学到复杂模式，教师"总结"了知识

**Q3: 蒸馏 VLA 时应该蒸馏哪些组件?**

A:
- **必须蒸馏**: Logits (输出分布)、动作轨迹
- **推荐蒸馏**: 注意力权重、中间特征
- **可选蒸馏**: 视觉编码器特征（如果学生编码器不同）

**Q4: 蒸馏和量化的顺序?**

A:
- **推荐**: 先蒸馏，再量化
- **原因**: 蒸馏保留知识，量化压缩表示；先量化会丢失太多信息
- **高级**: 量化感知蒸馏（QAD），在蒸馏时模拟量化误差

**Q5: Self-Distillation 是什么?**

A:
- **定义**: 模型自己蒸馏自己（教师和学生是同一模型）
- **方法**: 用模型的历史 checkpoint 作为教师
- **效果**: 提升模型的校准性 (Calibration)，常用于大模型训练后期

## 9. 参考资源 (References)

- **Original KD**: [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- **Feature Distillation**: [FitNets: Hints for Thin Deep Nets](https://arxiv.org/abs/1412.6550)
- **Attention Transfer**: [Paying More Attention to Attention](https://arxiv.org/abs/1612.03928)
- **Progressive Distillation**: [On the Efficacy of Knowledge Distillation](https://arxiv.org/abs/1910.01348)

---
[← Back to Theory](./README.md)

