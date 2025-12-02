# 多模态模型基础 (Multimodal Models)

> **核心概念**: 多模态模型 (Multimodal Models) 是指能够同时处理多种数据模态（如视觉、语言、音频、触觉等）的深度学习模型。在 VLA 领域，多模态能力是连接"看"、"说"、"做"的关键。

## 1. 为什么需要多模态? (Why Multimodal?)

### 1.1 机器人的感知需求

机器人在真实世界中需要同时处理多种信息：

| 模态 | 来源 | 作用 |
| :--- | :--- | :--- |
| **视觉 (Vision)** | RGB 相机、深度相机 | 理解场景、识别物体 |
| **语言 (Language)** | 语音指令、文本 | 理解任务意图 |
| **本体感知 (Proprioception)** | 关节编码器、IMU | 感知自身状态 |
| **触觉 (Tactile)** | 触觉传感器 | 感知接触力、纹理 |
| **音频 (Audio)** | 麦克风 | 环境声音、语音交互 |

### 1.2 单模态的局限性

- **仅视觉**: 无法理解抽象指令（"把那个危险的东西拿走"）
- **仅语言**: 无法定位具体物体（"桌上的红色杯子"在哪？）
- **缺乏本体感知**: 不知道机械臂当前姿态，无法闭环控制

### 1.3 多模态的优势

$$
\text{多模态理解} > \sum \text{单模态理解}
$$

- **语义接地 (Grounding)**: 将语言概念与视觉实体绑定
- **跨模态推理**: "红色的东西"（语言）→ 锁定红色物体（视觉）→ 抓取动作
- **鲁棒性**: 一个模态失效时，其他模态可以补偿

## 2. 多模态架构演进 (Architecture Evolution)

### 2.1 早期：双塔模型 (Dual-Encoder)

```
          ┌─────────────┐      ┌─────────────┐
图像 ────▶│  Image      │      │   Text      │◀──── 文本
          │  Encoder    │      │   Encoder   │
          │  (ResNet)   │      │   (BERT)    │
          └──────┬──────┘      └──────┬──────┘
                 │                    │
                 ▼                    ▼
              img_emb              text_emb
                 │                    │
                 └────────┬───────────┘
                          │
                    Cosine Similarity
```

**代表**: CLIP, ALIGN
**特点**: 图像和文本独立编码，通过对比学习对齐到同一空间
**局限**: 无法进行深度的跨模态交互

### 2.2 中期：融合编码器 (Fusion Encoder)

```
          ┌─────────────┐      ┌─────────────┐
图像 ────▶│  Image      │      │   Text      │◀──── 文本
          │  Encoder    │      │   Encoder   │
          └──────┬──────┘      └──────┬──────┘
                 │                    │
                 └────────┬───────────┘
                          ▼
                 ┌─────────────────┐
                 │  Fusion Module  │
                 │  (Cross-Attn)   │
                 └────────┬────────┘
                          ▼
                   Fused Features
```

**代表**: ViLBERT, LXMERT, UNITER
**特点**: 通过 Cross-Attention 实现深度交互
**改进**: 支持更复杂的多模态推理

### 2.3 现代：统一解码器 (Unified Decoder)

```
          ┌─────────────┐
图像 ────▶│  Vision     │──┐
          │  Encoder    │  │
          └─────────────┘  │
                           │   ┌─────────────────────┐
                           ├──▶│     LLM Decoder     │──▶ 输出
                           │   │  (Unified Token)    │
          ┌─────────────┐  │   └─────────────────────┘
文本 ────▶│  Tokenizer  │──┘
          └─────────────┘
```

**代表**: Flamingo, LLaVA, GPT-4V, Gemini
**特点**: 将视觉特征作为"虚拟 Token"输入到 LLM
**优势**: 利用 LLM 的强大推理能力，支持任意输入输出组合

## 3. VLA 中的多模态融合策略 (Fusion Strategies in VLA)

### 3.1 早期融合 (Early Fusion)

在特征提取阶段就进行融合。

```python
class EarlyFusion(nn.Module):
    def __init__(self):
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.language_proj = nn.Linear(language_dim, hidden_dim)
        self.proprio_proj = nn.Linear(proprio_dim, hidden_dim)
        
    def forward(self, image_feat, text_feat, proprio):
        # 直接拼接
        fused = torch.cat([
            self.vision_proj(image_feat),
            self.language_proj(text_feat),
            self.proprio_proj(proprio)
        ], dim=1)  # [B, L_v + L_t + 1, D]
        return fused
```

**优点**: 简单高效
**缺点**: 不同模态的特征尺度可能不匹配

### 3.2 中期融合 (Mid Fusion / Cross-Attention)

通过注意力机制动态融合。

```python
class CrossModalAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads)
        
    def forward(self, query_feat, context_feat):
        """
        query_feat: 需要被增强的特征 (e.g., 动作 query)
        context_feat: 提供上下文的特征 (e.g., 图像 + 语言)
        """
        # Query attends to Context
        attended, attn_weights = self.cross_attn(
            query=query_feat,
            key=context_feat,
            value=context_feat
        )
        return attended, attn_weights
```

**代表**: RT-1 (TokenLearner)，Octo
**优点**: 动态学习模态间关系
**缺点**: 计算开销大

### 3.3 晚期融合 (Late Fusion)

各模态独立处理后再合并决策。

```python
class LateFusion(nn.Module):
    def __init__(self):
        self.vision_policy = VisionPolicy()
        self.language_policy = LanguagePolicy()
        self.fusion_head = nn.Linear(hidden_dim * 2, action_dim)
        
    def forward(self, image, text):
        vision_out = self.vision_policy(image)
        language_out = self.language_policy(text)
        
        # 决策层融合
        fused = torch.cat([vision_out, language_out], dim=-1)
        action = self.fusion_head(fused)
        return action
```

**优点**: 各模态可以独立优化
**缺点**: 无法学习复杂的跨模态交互

### 3.4 VLA 中的主流方案：FiLM 调制

**FiLM (Feature-wise Linear Modulation)** 是 VLA 中最常用的条件注入方式。

```python
class FiLM(nn.Module):
    """Feature-wise Linear Modulation"""
    def __init__(self, cond_dim, feature_dim):
        self.gamma = nn.Linear(cond_dim, feature_dim)  # Scale
        self.beta = nn.Linear(cond_dim, feature_dim)   # Shift
        
    def forward(self, feature, condition):
        """
        feature: 要调制的特征 [B, L, D]
        condition: 条件信息 [B, C]
        """
        gamma = self.gamma(condition).unsqueeze(1)  # [B, 1, D]
        beta = self.beta(condition).unsqueeze(1)
        
        # 调制: γ * feature + β
        return gamma * feature + beta
```

**应用场景**:
- **RT-1**: 语言特征通过 FiLM 调制视觉特征
- **Diffusion Policy**: 时间步 $t$ 通过 FiLM 注入到 U-Net

## 4. 核心视觉编码器 (Vision Encoders)

### 4.1 ViT (Vision Transformer)

```
图像 [H, W, 3] 
    │
    ▼ Patch Embedding (16x16)
[N_patches, D] where N = (H/16) * (W/16)
    │
    ▼ + Position Embedding
    │
    ▼ Transformer Encoder (L layers)
    │
    ▼
[CLS] token 或 全局平均池化
```

**特点**:
- 将图像切分为 Patch (如 16x16)
- 每个 Patch 作为一个 Token
- 通过 Self-Attention 建模全局关系

### 4.2 SigLIP (Sigmoid Loss for Language-Image Pre-training)

**改进 CLIP**:
- 使用 Sigmoid 替代 Softmax (更好的批量对比学习)
- 支持更大的 batch size
- VLA 首选的视觉编码器 (OpenVLA, RDT)

### 4.3 DINOv2 (Self-supervised Vision Transformer)

**特点**:
- 自监督预训练，无需标签
- 强大的低层视觉特征 (边缘、纹理)
- 适合需要精确空间信息的任务

### 4.4 对比与选择

| 编码器 | 预训练方式 | 特点 | VLA 应用 |
| :--- | :--- | :--- | :--- |
| **ResNet** | 监督学习 | 高效，适合 CNN 策略 | RT-1, Diffusion Policy |
| **ViT** | 监督/自监督 | 全局建模强 | 通用 |
| **CLIP/SigLIP** | 对比学习 | 语义对齐好 | OpenVLA, RDT |
| **DINOv2** | 自监督 | 空间特征强 | 精细操作 |

## 5. 语言编码器 (Language Encoders)

### 5.1 BERT-style (Encoder-only)

```python
from transformers import BertModel

text = "pick up the red cup"
inputs = tokenizer(text, return_tensors="pt")
outputs = bert_model(**inputs)

# 使用 [CLS] token 或平均池化
text_embedding = outputs.last_hidden_state[:, 0, :]  # [B, D]
```

**适用**: 理解型任务，指令嵌入

### 5.2 T5-style (Encoder-Decoder)

**适用**: 需要生成文本的任务 (如 CoT 推理)

### 5.3 LLM-style (Decoder-only)

**代表**: Llama, Gemma, Qwen
**适用**: 现代 VLA 的标准选择，利用强大的 In-context Learning

## 6. 投影层设计 (Projector Design)

将视觉特征映射到语言空间是 VLA 的关键。

### 6.1 简单 MLP

```python
class MLPProjector(nn.Module):
    def __init__(self, vision_dim, language_dim):
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, language_dim),
            nn.GELU(),
            nn.Linear(language_dim, language_dim)
        )
    
    def forward(self, vision_feat):
        return self.proj(vision_feat)
```

### 6.2 Perceiver Resampler (Flamingo)

```python
class PerceiverResampler(nn.Module):
    """将可变数量的视觉 Token 压缩为固定数量"""
    def __init__(self, num_latents=64):
        self.latents = nn.Parameter(torch.randn(num_latents, hidden_dim))
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads)
        
    def forward(self, vision_tokens):
        # vision_tokens: [B, N_patches, D] (N_patches 可变)
        # 输出: [B, num_latents, D] (固定)
        
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)
        output, _ = self.cross_attn(
            query=latents,
            key=vision_tokens,
            value=vision_tokens
        )
        return output  # [B, 64, D]
```

**优势**: 控制视觉 Token 数量，减少 LLM 的计算负担

### 6.3 Q-Former (BLIP-2)

使用可学习的 Query 从视觉编码器中提取与任务相关的特征。

## 7. 实战：构建简单的多模态 VLA

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class SimpleMultimodalVLA(nn.Module):
    def __init__(
        self,
        vision_encoder_name="google/siglip-base-patch16-224",
        language_model_name="meta-llama/Llama-2-7b-hf",
        action_dim=7,
        chunk_size=16
    ):
        super().__init__()
        
        # 视觉编码器 (冻结)
        self.vision_encoder = AutoModel.from_pretrained(vision_encoder_name)
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        
        # 投影层 (可训练)
        vision_dim = self.vision_encoder.config.hidden_size
        language_dim = 4096  # Llama 2 hidden dim
        self.vision_projector = nn.Sequential(
            nn.Linear(vision_dim, language_dim),
            nn.GELU(),
            nn.Linear(language_dim, language_dim)
        )
        
        # 语言模型 (LoRA 微调)
        self.language_model = AutoModel.from_pretrained(
            language_model_name,
            load_in_4bit=True  # QLoRA
        )
        
        # 动作头 (可训练)
        self.action_head = nn.Sequential(
            nn.Linear(language_dim, language_dim // 2),
            nn.ReLU(),
            nn.Linear(language_dim // 2, action_dim * chunk_size)
        )
        
        self.chunk_size = chunk_size
        self.action_dim = action_dim
    
    def forward(self, images, input_ids, attention_mask):
        """
        images: [B, C, H, W]
        input_ids: [B, L]
        attention_mask: [B, L]
        """
        batch_size = images.shape[0]
        
        # 1. 视觉编码
        with torch.no_grad():
            vision_outputs = self.vision_encoder(images)
            vision_features = vision_outputs.last_hidden_state  # [B, N_patches, D_v]
        
        # 2. 投影到语言空间
        vision_tokens = self.vision_projector(vision_features)  # [B, N_patches, D_l]
        
        # 3. 获取语言嵌入
        text_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        # 4. 拼接 [Vision Tokens | Text Tokens]
        inputs_embeds = torch.cat([vision_tokens, text_embeds], dim=1)
        
        # 5. 通过语言模型
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            output_hidden_states=True
        )
        
        # 6. 取最后一个 hidden state 作为动作条件
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # [B, D_l]
        
        # 7. 预测动作
        actions = self.action_head(last_hidden)  # [B, action_dim * chunk_size]
        actions = actions.view(batch_size, self.chunk_size, self.action_dim)
        
        return actions
```

## 8. 面试高频问题 (Q&A)

**Q1: CLIP 和 SigLIP 的区别是什么？**

A:
- **损失函数**: CLIP 使用 Softmax + Cross-Entropy (InfoNCE)，SigLIP 使用 Sigmoid + Binary CE
- **batch 依赖**: CLIP 的 Softmax 需要对比 batch 内所有样本，SigLIP 的 Sigmoid 每对独立计算
- **扩展性**: SigLIP 更适合大 batch 训练，负样本利用更高效

**Q2: 为什么 VLA 普遍选择 Decoder-only LLM 而不是 BERT？**

A:
- **生成能力**: Decoder-only 天然支持自回归生成（包括动作 Token）
- **In-context Learning**: 可以通过 Prompt 引导模型理解新任务
- **规模效应**: 大规模 LLM (7B+) 主要是 Decoder-only 架构，可以直接复用

**Q3: 多模态融合中 Early / Mid / Late Fusion 如何选择？**

A:
- **Early Fusion**: 数据模态相似度高（如多相机图像）
- **Mid Fusion (Cross-Attention)**: 需要动态建模模态间关系（VLA 首选）
- **Late Fusion**: 各模态任务独立性强，或需要模块化解释性

**Q4: 视觉 Token 数量如何选择？**

A:
- **多了**: LLM 计算开销大，长序列 Attention 变慢
- **少了**: 丢失空间细节，影响精细操作
- **常见选择**: 256 tokens (16x16 patches @ 224px)，或使用 Perceiver Resampler 压缩到 64

**Q5: 为什么要冻结视觉编码器？**

A:
- **防止灾难性遗忘**: 视觉编码器的预训练特征很重要
- **计算效率**: 减少可训练参数
- **数据效率**: 机器人数据少，全量训练容易过拟合
- **例外**: 如果视觉任务差异大（如从 ImageNet 迁移到内窥镜），可能需要微调

**Q6: 如果视觉模块误判，如何通过语言纠错？**

A: 这是多模态 VLA 的核心优势之一，有以下几种机制：

```
┌─────────────────────────────────────────────────────────────────┐
│                   视觉误判 → 语言纠错机制                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   场景: 视觉模块误判 "红色杯子" 为 "橙色杯子"                    │
│                                                                 │
│   方案 1: 闭环语言反馈 (Human-in-the-Loop)                      │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  用户: "不对，是红色的那个"                               │   │
│   │  VLA: 重新定位 → 修正目标                                │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   方案 2: Chain-of-Thought 自检                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  VLA 输出: "我看到一个橙色物体..."                        │   │
│   │  用户指令: "抓红色杯子"                                   │   │
│   │  CoT 推理: "指令说红色，但我识别为橙色，可能有误"          │   │
│   │  动作: 请求确认 或 重新感知                              │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   方案 3: 多模态一致性检查                                      │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  计算: sim(语言描述 Embedding, 视觉特征 Embedding)        │   │
│   │  如果 sim < threshold: 触发重新感知/询问                  │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   方案 4: 主动询问 (Uncertainty-aware)                          │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  VLA: "你是指这个吗？" (显示候选物体)                     │   │
│   │  用户: "是的" / "不是，是左边那个"                        │   │
│   └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**实现要点**:
1. **语义接地 (Grounding)**: 语言指令必须与视觉检测结果绑定，而非独立处理
2. **置信度输出**: 视觉模块输出检测置信度，低置信度时触发纠错机制
3. **多轮对话**: VLA 需要支持多轮交互，而非单次指令执行
4. **CoT 推理**: 显式输出推理过程，便于发现矛盾 (参见 [chain_of_thought.md](./chain_of_thought.md))

## 9. 参考资源 (References)

- **CLIP**: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- **LLaVA**: [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)
- **Flamingo**: [A Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198)
- **SigLIP**: [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343)

---
[← Back to Theory](./README.md)

