# 自监督学习 (Self-Supervised Learning)

> **核心概念**: 自监督学习 (Self-Supervised Learning, SSL) 是一种从无标签数据中学习有意义表示的方法。通过设计 **Pretext Task (代理任务)**，让模型从数据本身的结构中学习。

## 0. 主要數學思想 (Main Mathematical Idea)

> **第一性原理**: **Structure is Supervision (结构即监督)**

人类学习不需要每时每刻的外部标签，因为数据本身内部蕴含着丰富的**结构信息**（时空连续性、多视角一致性）。

- **核心数学工具**: **Manifold Learning (流形学习)** 与 **Mutual Information Maximization (互信息最大化)**。
- **解题逻辑**:
    1.  **不变性 (Invariance)**: 同一个物体，无论光照、角度如何变化，其本质（语义）不变。通过拉近同一物体不同视图的距离（对比学习），模型学会了忽略无关干扰（如像素噪声），抓住核心语义。
    2.  **预测性 (Predictability)**: 世界是有规律的。如果我知道了现在的状态，我应该能某种程度上预测未来（预测学习）或填补缺失（掩码预测）。这种预测能力迫使模型理解数据的内在逻辑和物理规律。

## 1. 为什么 VLA 需要自监督学习? (Why SSL for VLA?)

### 1.1 机器人数据的困境

| 数据类型 | 规模 | 标注成本 |
| :--- | :--- | :--- |
| 互联网图文 | 数十亿 | 低（网页自动爬取） |
| 视频数据 | 数亿小时 | 极低（无需标注） |
| **机器人操作数据** | **数十万** | **极高（需真机遥操）** |

**问题**: 有标签的机器人数据稀缺，无法支撑大模型训练。

### 1.2 SSL 的价值

$$
\text{大量无标签数据} \xrightarrow{\text{SSL 预训练}} \text{通用表示} \xrightarrow{\text{少量标签微调}} \text{高性能策略}
$$

- **视觉表示**: 从海量图像/视频中学习通用视觉特征
- **动作表示**: 从人类视频中学习动作先验
- **世界模型**: 从视频中学习物理规律

## 2. 自监督学习范式 (SSL Paradigms)

### 2.1 对比学习 (Contrastive Learning)

**核心思想**: 拉近相似样本，推远不同样本。

#### 2.1.1 InfoNCE 损失

$$
\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbf{1}_{k \neq i} \exp(\text{sim}(z_i, z_k) / \tau)}
$$

其中:
- $z_i, z_j$: 同一样本的两个增强视图的表示
- $\tau$: 温度系数 (通常 0.07)
- $\text{sim}(\cdot)$: 余弦相似度

#### 2.1.2 SimCLR 框架

```
         ┌──────────────────────────────────────┐
         │           原始图像 x                  │
         └───────────────┬──────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │ 数据增强 T    │               │ 数据增强 T'
         ▼               │               ▼
      ┌─────┐            │            ┌─────┐
      │ x_i │            │            │ x_j │
      └──┬──┘            │            └──┬──┘
         │               │               │
         ▼               │               ▼
      Encoder f          │            Encoder f
         │               │               │
         ▼               │               ▼
      ┌─────┐            │            ┌─────┐
      │ h_i │            │            │ h_j │
      └──┬──┘            │            └──┬──┘
         │               │               │
         ▼               │               ▼
      Projector g        │            Projector g
         │               │               │
         ▼               │               ▼
      ┌─────┐            │            ┌─────┐
      │ z_i │◀───────────┴───────────▶│ z_j │
      └─────┘       对比损失           └─────┘
                   (最大化相似度)
```

**数据增强策略**:
- 随机裁剪 + 缩放
- 颜色抖动 (Color Jittering)
- 高斯模糊
- 水平翻转

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLR(nn.Module):
    def __init__(self, encoder, proj_dim=128, temperature=0.07):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(encoder.output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, proj_dim)
        )
        self.temperature = temperature
    
    def forward(self, x_i, x_j):
        # 编码
        h_i = self.encoder(x_i)  # [B, D]
        h_j = self.encoder(x_j)
        
        # 投影
        z_i = F.normalize(self.projector(h_i), dim=1)
        z_j = F.normalize(self.projector(h_j), dim=1)
        
        # 计算相似度矩阵
        batch_size = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)  # [2B, D]
        sim_matrix = torch.mm(z, z.t()) / self.temperature  # [2B, 2B]
        
        # 构建标签: 正样本对在对角线上
        labels = torch.arange(batch_size, device=z.device)
        labels = torch.cat([labels + batch_size, labels])  # [2B]
        
        # 移除自身相似度
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim_matrix.masked_fill_(mask, -float('inf'))
        
        # InfoNCE 损失
        loss = F.cross_entropy(sim_matrix, labels)
        return loss
```

#### 2.1.3 CLIP: 视觉-语言对比学习

```python
def clip_loss(image_features, text_features, temperature=0.07):
    """
    image_features: [B, D] - 图像编码
    text_features: [B, D] - 文本编码
    """
    # 归一化
    image_features = F.normalize(image_features, dim=1)
    text_features = F.normalize(text_features, dim=1)
    
    # 相似度矩阵
    logits = torch.mm(image_features, text_features.t()) / temperature
    
    # 对称损失
    labels = torch.arange(len(logits), device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)      # 图像→文本
    loss_t2i = F.cross_entropy(logits.t(), labels)  # 文本→图像
    
    return (loss_i2t + loss_t2i) / 2
```

### 2.2 掩码预测 (Masked Prediction)

**核心思想**: 遮住部分输入，让模型预测被遮住的部分。

#### 2.2.1 MAE (Masked Autoencoder)

```
原始图像 (224x224)
    │
    ▼ Patch 分割 (16x16)
196 个 Patches
    │
    ▼ 随机 Mask (75%)
49 个可见 Patches + 147 个 Mask Tokens
    │
    ▼ ViT Encoder (只处理可见 Patches)
    │
    ▼ 添加 Mask Tokens + 位置编码
    │
    ▼ 轻量 Decoder (重建全部 Patches)
    │
    ▼ MSE Loss (只计算被 Mask 的 Patches)
```

**关键设计**:
- **高 Mask 比例 (75%)**: 任务足够难，强迫学习语义
- **非对称架构**: 重编码器 (ViT-L)，轻解码器 (小 Transformer)
- **只编码可见 Patches**: 大幅减少计算量

```python
class MAE(nn.Module):
    def __init__(self, encoder, decoder, mask_ratio=0.75):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask_ratio = mask_ratio
        
        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, encoder.embed_dim))
    
    def random_masking(self, x, mask_ratio):
        """随机 Mask 策略"""
        B, N, D = x.shape  # [Batch, N_patches, Dim]
        num_keep = int(N * (1 - mask_ratio))
        
        # 随机打乱
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # 保留前 num_keep 个
        ids_keep = ids_shuffle[:, :num_keep]
        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        
        # 生成 mask (1 表示被移除)
        mask = torch.ones(B, N, device=x.device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward(self, images):
        # Patch embedding
        patches = self.encoder.patch_embed(images)  # [B, N, D]
        
        # Random masking
        patches_masked, mask, ids_restore = self.random_masking(patches, self.mask_ratio)
        
        # Encode (只处理可见 patches)
        latent = self.encoder.forward_encoder(patches_masked)
        
        # Decode (补充 mask tokens)
        B, N_vis, D = latent.shape
        mask_tokens = self.mask_token.expand(B, ids_restore.shape[1] - N_vis, -1)
        latent_full = torch.cat([latent, mask_tokens], dim=1)
        latent_full = torch.gather(latent_full, 1, ids_restore.unsqueeze(-1).expand(-1, -1, D))
        
        pred = self.decoder(latent_full)  # [B, N, patch_size^2 * 3]
        
        # Loss (只计算 masked patches)
        target = self.patchify(images)
        loss = (pred - target) ** 2
        loss = (loss.mean(dim=-1) * mask).sum() / mask.sum()
        
        return loss
```

#### 2.2.2 VideoMAE: 视频版 MAE

**关键改进**:
- **时空 Masking**: 对视频的时间和空间维度同时 Mask
- **Tube Masking**: 在时间上连续 Mask 同一位置（更难）
- **运动先验**: 学习时序动态

**对 VLA 的价值**: 从人类视频中学习动作的时序模式

### 2.3 预测学习 (Predictive Learning)

#### 2.3.1 时间对比学习

```python
class TemporalContrastive(nn.Module):
    """学习视频帧的时序关系"""
    def __init__(self, encoder, pred_steps=3):
        super().__init__()
        self.encoder = encoder
        self.predictor = nn.GRU(hidden_size, hidden_size)
        self.pred_steps = pred_steps
    
    def forward(self, video_frames):
        """
        video_frames: [B, T, C, H, W]
        """
        B, T, C, H, W = video_frames.shape
        
        # 编码每帧
        features = []
        for t in range(T):
            feat = self.encoder(video_frames[:, t])
            features.append(feat)
        features = torch.stack(features, dim=1)  # [B, T, D]
        
        # 预测未来帧特征
        loss = 0
        for t in range(T - self.pred_steps):
            context = features[:, :t+1]  # 历史帧
            pred, _ = self.predictor(context)
            pred = pred[:, -1]  # 最后一步的预测
            
            # 对比学习: 正样本是未来帧
            target = features[:, t + self.pred_steps]
            loss += contrastive_loss(pred, target)
        
        return loss / (T - self.pred_steps)
```

#### 2.3.2 R3M: 从人类视频学习机器人表示

**核心思想**: 人类的手部动作与机器人 gripper 类似，可以迁移。

```
人类视频 (Ego4D 数据集)
    │
    ├─▶ 时间对比学习 (Time Contrastive)
    │   - 相邻帧特征应该相似
    │
    ├─▶ 视频-语言对齐 (Video-Language Alignment)
    │   - 视频与文字描述对齐
    │
    └─▶ 语言条件时间对比 (L3)
        - 语言引导的时序预测

    │
    ▼
通用视觉表示 (适用于机器人)
```

## 3. VLA 中的 SSL 应用 (SSL in VLA)

### 3.1 视觉编码器预训练

| 方法 | 数据 | 用于 VLA |
| :--- | :--- | :--- |
| **ImageNet 监督** | 1M 图像 + 标签 | 基础特征 |
| **CLIP** | 400M 图文对 | 语义对齐 |
| **DINOv2** | 142M 无标签图像 | 空间特征 |
| **R3M** | Ego4D 人类视频 | 操作相关特征 |

### 3.2 世界模型预训练

```python
class WorldModelSSL(nn.Module):
    """从视频预测未来帧，学习物理规律"""
    def __init__(self):
        self.encoder = ViTEncoder()
        self.dynamics = TransformerDynamics()
        self.decoder = ViTDecoder()
    
    def forward(self, video, actions=None):
        """
        video: [B, T, C, H, W]
        actions: [B, T-1, A] (可选，动作条件)
        """
        # 编码历史帧
        B, T, C, H, W = video.shape
        latents = []
        for t in range(T):
            z_t = self.encoder(video[:, t])
            latents.append(z_t)
        latents = torch.stack(latents, dim=1)  # [B, T, D]
        
        # 预测下一帧
        pred_latents = self.dynamics(latents[:, :-1], actions)
        
        # 重建
        pred_frames = self.decoder(pred_latents)
        
        # 损失: 预测与真实未来帧的差异
        loss = F.mse_loss(pred_frames, video[:, 1:])
        return loss
```

### 3.3 动作表示学习

```python
class ActionSSL(nn.Module):
    """从视频中学习隐式动作表示"""
    def __init__(self):
        self.encoder = VideoEncoder()
        self.action_predictor = MLP()
    
    def forward(self, frame_t, frame_t1):
        """预测两帧之间的隐式动作"""
        z_t = self.encoder(frame_t)
        z_t1 = self.encoder(frame_t1)
        
        # 预测"动作"使得 z_t → z_t1
        pred_action = self.action_predictor(torch.cat([z_t, z_t1], dim=-1))
        
        # 自监督目标: 动作应该能够预测状态变化
        pred_z_t1 = self.dynamics(z_t, pred_action)
        loss = F.mse_loss(pred_z_t1, z_t1.detach())
        
        return loss, pred_action
```

## 4. 数据增强策略 (Data Augmentation)

### 4.1 图像增强

```python
import torchvision.transforms as T

# VLA 常用增强
train_transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(p=0.5),  # 注意: 机器人任务可能需要禁用
    T.ColorJitter(0.4, 0.4, 0.4, 0.1),
    T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### 4.2 机器人特定增强

```python
class RobotAugmentation:
    """机器人场景的数据增强"""
    
    @staticmethod
    def camera_shift(image, proprio, max_shift=10):
        """模拟相机位置轻微偏移"""
        shift_x = random.randint(-max_shift, max_shift)
        shift_y = random.randint(-max_shift, max_shift)
        image = T.functional.affine(image, angle=0, translate=(shift_x, shift_y), scale=1, shear=0)
        return image, proprio
    
    @staticmethod
    def action_noise(action, noise_std=0.01):
        """给动作添加噪声（用于 BC）"""
        noise = torch.randn_like(action) * noise_std
        return action + noise
    
    @staticmethod
    def temporal_crop(video, crop_ratio=0.8):
        """时间维度裁剪"""
        T = video.shape[0]
        crop_len = int(T * crop_ratio)
        start = random.randint(0, T - crop_len)
        return video[start:start + crop_len]
```

## 5. 对比学习 vs 掩码预测 (Comparison)

| 特性 | 对比学习 (Contrastive) | 掩码预测 (Masked) |
| :--- | :--- | :--- |
| **代表** | SimCLR, CLIP, MoCo | MAE, BEiT |
| **目标** | 学习不变性表示 | 学习重建能力 |
| **数据增强** | 强依赖 | 不依赖 |
| **负样本** | 需要 (大 batch) | 不需要 |
| **计算效率** | 低 (大 batch) | 高 (只编码 25%) |
| **下游任务** | 分类、检索 | 检测、分割 |
| **VLA 适用** | 语义理解 | 空间精细任务 |

## 6. 面试高频问题 (Q&A)

**Q1: 对比学习中温度系数 τ 的作用是什么?**

A:
- **τ 小**: 分布更 sharp，对负样本区分更严格，但容易过拟合噪声
- **τ 大**: 分布更 uniform，对比更"软"，但难以学习细粒度区分
- **经验值**: 0.07 (CLIP), 0.1 (SimCLR)

**Q2: MAE 为什么要 Mask 75% 这么高的比例?**

A:
- **任务难度**: 比例低时任务太简单，通过插值就能重建
- **语义学习**: 高比例强迫模型理解图像的整体语义结构
- **效率**: 只编码 25% patches，计算量减少 4 倍
- **对比**: NLP 的 BERT 只 Mask 15%，因为语言冗余更低

**Q3: R3M 如何从人类视频迁移到机器人?**

A:
- **假设**: 人类手部操作的视觉特征与机器人 gripper 类似
- **数据**: Ego4D (第一人称视频) 含大量手-物交互
- **方法**: 时间对比 + 语言对齐，学习"动作相关"的视觉表示
- **迁移**: 冻结编码器，只训练策略头

**Q4: 自监督学习在 VLA 中最大的价值是什么?**

A:
- **数据效率**: 利用海量无标签数据预训练，减少对稀缺机器人数据的依赖
- **泛化能力**: 预训练特征具有更好的跨域泛化能力
- **世界模型**: 从视频学习物理规律，支持模型预测控制

**Q5: 如何选择 SSL 方法进行 VLA 预训练?**

A:
- **语义理解任务** (如指令跟随): CLIP / 对比学习
- **空间精细任务** (如精密装配): MAE / DINOv2
- **动作预测任务**: R3M / VideoMAE

## 7. 参考资源 (References)

- **SimCLR**: [A Simple Framework for Contrastive Learning](https://arxiv.org/abs/2002.05709)
- **MAE**: [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)
- **CLIP**: [Learning Transferable Visual Models](https://arxiv.org/abs/2103.00020)
- **R3M**: [R3M: A Universal Visual Representation for Robot Manipulation](https://arxiv.org/abs/2203.12601)
- **DINOv2**: [DINOv2: Learning Robust Visual Features](https://arxiv.org/abs/2304.07193)

---
[← Back to Theory](./README.md)

