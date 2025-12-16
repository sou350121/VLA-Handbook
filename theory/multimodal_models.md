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

---

## 5.5 PaliGemma 详解 (VLA 常用 Backbone)

> **论文**: [PaliGemma: A versatile 3B VLM for transfer](https://arxiv.org/abs/2407.07726) (Google, 2024)
> **官方**: [HuggingFace](https://huggingface.co/google/paligemma-3b-pt-224)

PaliGemma 是 Google 推出的轻量级 VLM，已成为 **π0、OpenVLA** 等 VLA 的首选 backbone。

### 为什么 VLA 常用 PaliGemma?

| 优势 | 说明 |
| :--- | :--- |
| **轻量高效** | 3B 参数，可在单卡 (24GB) 微调 |
| **预训练充分** | 在大量图文数据上训练，视觉理解强 |
| **开源友好** | Apache 2.0 许可，可商用 |
| **模块化设计** | Vision Encoder 和 LLM 解耦，易于适配 |
| **多分辨率** | 支持 224/448/896 输入尺寸 |

### PaliGemma 架构

```
┌─────────────────────────────────────────────────────────────┐
│                      PaliGemma 3B                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Image Input                    Text Input                  │
│   [224×224×3]                    "Pick up the cup"           │
│        │                              │                      │
│        ▼                              ▼                      │
│   ┌──────────────┐              ┌──────────────┐            │
│   │   SigLIP     │              │   Gemma      │            │
│   │  ViT-So400m  │              │  Tokenizer   │            │
│   │  (400M)      │              │              │            │
│   └──────┬───────┘              └──────┬───────┘            │
│          │                             │                     │
│   [256 patches]                  [L tokens]                  │
│   [256, 1152]                    [L, 2048]                   │
│          │                             │                     │
│          ▼                             │                     │
│   ┌──────────────┐                     │                     │
│   │  Linear Proj │ (1152 → 2048)       │                     │
│   └──────┬───────┘                     │                     │
│          │                             │                     │
│          └──────────┬──────────────────┘                     │
│                     ▼                                        │
│            [Vision] + [Text Tokens]                          │
│                     │                                        │
│                     ▼                                        │
│   ┌─────────────────────────────────────────────────────────┐│
│   │                 Gemma 2B LLM                            ││
│   │          (18 Transformer Layers)                        ││
│   │                                                         ││
│   │    Self-Attention (Vision + Text 一起处理)               ││
│   │                     ↓                                   ││
│   │              Hidden States                              ││
│   └─────────────────────────────────────────────────────────┘│
│                     │                                        │
│                     ▼                                        │
│              [B, L, 2048]                                    │
│           (送给 Action Head)                                 │
└─────────────────────────────────────────────────────────────┘
```

### 核心组件

#### 1. SigLIP Vision Encoder

```python
# SigLIP vs CLIP
# SigLIP 使用 Sigmoid Loss 而非 Softmax，更适合细粒度理解

# 配置
vision_config = {
    "model": "ViT-So400m",      # 400M 参数
    "image_size": 224,          # 或 448, 896
    "patch_size": 14,           # 16×16 patches
    "hidden_size": 1152,
    "num_layers": 27,
    "num_heads": 16
}

# 输出: [B, 256, 1152] (256 = (224/14)² patches)
```

#### 2. Gemma 2B LLM

```python
# Gemma 是 Google 的轻量级 LLM
llm_config = {
    "hidden_size": 2048,
    "num_layers": 18,
    "num_heads": 8,
    "vocab_size": 256000,
    "max_position": 8192,
    "intermediate_size": 16384  # FFN
}
```

#### 3. 投影层 (Linear Projection)

```python
# 将 SigLIP 特征投射到 Gemma 空间
self.vision_proj = nn.Linear(1152, 2048)

# 投射后，视觉 Token 和文本 Token 在同一空间
vision_tokens = self.vision_proj(siglip_output)  # [B, 256, 2048]
```

### VLA 中的使用方式

```python
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

# 加载模型
model = PaliGemmaForConditionalGeneration.from_pretrained(
    "google/paligemma-3b-pt-224",
    torch_dtype=torch.bfloat16
)
processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")

# 方式 1: 获取 Hidden States (用于 Action Head)
def get_vlm_features(images, text):
    inputs = processor(images=images, text=text, return_tensors="pt")
    outputs = model(
        **inputs,
        output_hidden_states=True
    )
    # 最后一层 hidden states
    hidden = outputs.hidden_states[-1]  # [B, L, 2048]
    return hidden

# 方式 2: 直接生成文本 (用于 CoT)
def generate_text(images, text):
    inputs = processor(images=images, text=text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    return processor.decode(outputs[0])
```

### PaliGemma 版本对比

| 版本 | 参数量 | 输入分辨率 | 适用场景 |
| :--- | :--- | :--- | :--- |
| **paligemma-3b-pt-224** | 3B | 224×224 | VLA 首选，平衡效率 |
| paligemma-3b-pt-448 | 3B | 448×448 | 需要更多细节 |
| paligemma-3b-pt-896 | 3B | 896×896 | 高分辨率任务 |
| paligemma-3b-mix-224 | 3B | 224×224 | 混合任务微调版 |

### PaliGemma vs 其他 VLM

| 模型 | 参数量 | 开源 | VLA 适用性 |
| :--- | :--- | :--- | :--- |
| **PaliGemma** | **3B** | ✅ Apache 2.0 | ⭐⭐⭐⭐⭐ 最常用 |
| LLaVA 1.5 | 7B/13B | ✅ | ⭐⭐⭐⭐ 较大但成熟 |
| Qwen-VL | 7B | ✅ | ⭐⭐⭐⭐ 中文支持好 |
| GPT-4V | ~1T | ❌ | ⭐⭐ API 延迟高 |
| PaLI-X | 55B | ❌ | ⭐ 太大无法部署 |

### 面试常见问题

**Q: 为什么 π0 选择 PaliGemma 而不是更大的 LLaVA?**

A: 三个原因:
1. **效率**: 3B 参数可在单卡训练/推理，满足机器人实时性要求
2. **SigLIP**: 比 CLIP 更好的细粒度视觉理解
3. **模块化**: Vision/Language 解耦，方便接 Action Head

---

**Q: PaliGemma 的 256 个 vision tokens 够用吗?**

A: 对于大多数机器人任务足够:
- 桌面操作: 224×224 分辨率 + 256 tokens 能覆盖关键物体
- 需要精细操作时: 可用 448/896 版本 (1024/4096 tokens)
- Trade-off: 更多 tokens = 更慢推理

---

## 5.6 主流 VLM 对比表（VLA 训练参考）

> **目标**: 为 VLA 开发者提供当前市场上主流 Vision Language Model 的对比，重点关注**已在 VLA 项目中实际使用**的模型。
> 
> **最后更新**: 2025年12月5日

---

### 5.6.1 ✅ 已在 VLA 中实际使用（优先推荐）

| 模型 | 机构 | 发布时间 | Vision Encoder | LLM Backbone | 参数量 | 输入分辨率 | 开源 | 许可证 | VLA 应用案例 | HuggingFace |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PaliGemma 3B** | Google | 2024.07 | SigLIP ViT-So400m | Gemma 2B | 3B | 224/448/896 | ✅ | Apache 2.0 | **π0 (Pi-Zero)**, OpenVLA 变体 | [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224) |
| **SigLIP** | Google | 2023.09 | ViT (Sigmoid Loss) | - | 400M-2.6B | 224-384 | ✅ | Apache 2.0 | **OpenVLA**, **RDT** (Vision Encoder) | [google/siglip-*](https://huggingface.co/models?search=siglip) |
| **LLaVA 1.5/1.6** | - | 2023.10/2024.01 | CLIP/ViT | Llama 2/Vicuna | 7B/13B | 336/672 | ✅ | Apache 2.0 | **OpenVLA** (Llama 2 + SigLIP 组合) | [llava-hf/llava-1.5-*](https://huggingface.co/models?search=llava) |
| **LLaVA-NeXT** | - | 2024.12 | CLIP/ViT | Llama 3/Vicuna | 7B/13B/34B | 672/1344 | ✅ | Apache 2.0 | 最新版本，性能提升 | [llava-hf/llava-next-*](https://huggingface.co/models?search=llava-next) |
| **PaLI-X** | Google | 2023.12 | ViT-22B | PaLM-E | 55B | 224-1024 | ❌ | - | **RT-2** | - |

**选择建议**:
- **PaliGemma 3B**: VLA 训练首选，轻量高效（单卡 24GB 可训练），预训练充分，模块化设计
- **SigLIP**: VLA 首选视觉编码器，比 CLIP 更强的细粒度理解，支持大 batch 训练
- **LLaVA**: 成熟稳定，社区支持好，适合需要更大模型的场景

### 5.6.2 🔄 适合 VLA 训练的开源 VLM（推荐尝试）

#### 🆕 2025年最新发布

| 模型 | 机构 | 发布时间 | Vision Encoder | LLM Backbone | 参数量 | 输入分辨率 | 开源 | 许可证 | 优势 | HuggingFace |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen2.5-VL** | 阿里巴巴 | 2025.03 | Window Attn ViT + MRoPE | Qwen2.5 LLM | 3B/7B/32B/72B | 任意分辨率 | ✅ | Apache 2.0 | **2025 SOTA**，数学推理强，长视频支持 | [Qwen/Qwen2.5-VL-*](https://huggingface.co/models?search=Qwen2.5-VL) |
| **Eagle 2.5** | NVIDIA | 2025.04 | 长上下文 ViT | - | 8B | 长视频 | ✅ | Apache 2.0 | 长上下文多模态，Video-MME 72.4% | [nvidia/Eagle-*](https://huggingface.co/models?search=Eagle) |
| **Seed 1.5-VL** | 字节跳动 | 2025.05 | - | - | 20B (激活) | - | ✅ | - | 媲美 Gemini 2.5 Pro，GUI 交互强 | [ByteDance/Seed-*](https://huggingface.co/models?search=Seed) |
| **PLM** | Meta | 2025.05 | - | - | - | - | ✅ | MIT | 开源视觉语言模型，复杂视觉任务 | [meta-llama/PLM](https://github.com/facebookresearch/PLM) |
| **GLM-4.5V** | 智谱AI | 2025 | 3D-RoPE ViT | GLM-4.5-Air | 106B (12B 激活) | - | ✅ | Apache 2.0 | MoE 架构，3D 空间推理 | [THUDM/GLM-4.5V](https://huggingface.co/models?search=GLM-4) |
| **Llama 4 Scout/Maverick** | Meta | 2025.04 | ViT Patch | MoE Transformer | 16-128 专家 | - | ✅ | Meta Llama | 10M token 上下文，多模态 | [meta-llama/Llama-4](https://huggingface.co/models?search=llama-4) |

#### 2024年发布（仍推荐）

| 模型 | 机构 | 发布时间 | Vision Encoder | LLM Backbone | 参数量 | 输入分辨率 | 开源 | 许可证 | 优势 | HuggingFace |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen2-VL** | 阿里巴巴 | 2024.08 | InternViT | Qwen2 LLM | 2B/7B/72B | 448-1344 | ✅ | Apache 2.0 | 性能大幅提升 | [Qwen/Qwen2-VL-*](https://huggingface.co/models?search=Qwen2-VL) |
| **InternVL2** | 商汤 | 2024.07 | InternViT-6B | InternLM2 | 2B/4B/8B/26B | 448-1344 | ✅ | Apache 2.0 | 多模态能力增强 | [OpenGVLab/InternVL2-*](https://huggingface.co/models?search=InternVL2) |
| **MiniCPM-V 2.6** | 面壁智能 | 2024.08 | ViT | MiniCPM | 8B | 336-1344 | ✅ | Apache 2.0 | 超轻量级，边缘部署 | [openbmb/MiniCPM-V-*](https://huggingface.co/models?search=MiniCPM-V) |
| **LLaVA-NeXT** | - | 2024.06 | CLIP/ViT | Llama 3/Vicuna | 7B/13B/34B | 672/1344 | ✅ | Apache 2.0 | 最新 LLaVA 版本 | [llava-hf/llava-next-*](https://huggingface.co/models?search=llava-next) |
| **SmolVLA** | Hugging Face | 2024.12 | ViT-Small | TinyLlama | 450M | 224 | ✅ | Apache 2.0 | 超轻量级，VLA 研究入门 | [huggingface/smolvla](https://huggingface.co/models?search=smolvla) |

#### 经典模型（仍可用）

| 模型 | 机构 | 发布时间 | Vision Encoder | LLM Backbone | 参数量 | 输入分辨率 | 开源 | 许可证 | 优势 | HuggingFace |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen-VL** | 阿里巴巴 | 2023.11 | CLIP-ViT | Qwen LLM | 7B/72B | 448-1024 | ✅ | Apache 2.0 | 中文支持好 | [Qwen/Qwen-VL](https://huggingface.co/Qwen/Qwen-VL) |
| **CogVLM** | 智谱AI | 2023.10 | EVA2-ViT | GLM | 17B | 490 | ✅ | Apache 2.0 | 视觉理解强，中文支持 | [THUDM/cogvlm-*](https://huggingface.co/models?search=cogvlm) |
| **InternVL** | 商汤 | 2024.01 | InternViT | InternLM | 2B-26B | 448-1024 | ✅ | Apache 2.0 | 多分辨率支持 | [OpenGVLab/InternVL-*](https://huggingface.co/models?search=InternVL) |

**适用场景**:
- **Qwen2.5-VL** (🆕 2025): 中文指令 VLA 首选，数学推理强，支持任意分辨率和长视频
- **Eagle 2.5** (🆕 2025): 长上下文多模态任务，视频理解
- **Seed 1.5-VL** (🆕 2025): GUI 交互、复杂视觉推理
- **GLM-4.5V** (🆕 2025): 3D 空间推理任务
- **Llama 4** (🆕 2025): 超长上下文（10M token），文档分析
- **Qwen2-VL**: 中文支持好（2024 版本）
- **MiniCPM-V**: 边缘设备部署，资源受限场景
- **SmolVLA**: 超轻量级研究，快速原型验证

### 5.6.3 ❌ 闭源 API（参考，不适合直接训练）

| 模型 | 机构 | 发布时间 | 参数量 | 特点 | VLA 适用性 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Gemini 2.5 Pro** 🆕 | Google | 2025.03 | 未公开 | **2025 SOTA**，1M token 上下文，内置思考功能 | ⭐⭐ API 调用，成本高 |
| **Claude 3.7 Vision** 🆕 | Anthropic | 2025.02 | 未公开 | 高精度 OCR，图表解析 | ⭐⭐ API 调用，延迟问题 |
| **GPT-4o** | OpenAI | 2024.05 | ~1T | 多模态理解强，统一 Transformer 架构 | ⭐⭐ API 延迟高，不适合实时控制 |
| **GPT-4o-mini** | OpenAI | 2024.07 | 未公开 | 轻量版 GPT-4o，成本更低 | ⭐⭐ API 调用，延迟仍较高 |
| **Gemini 1.5 Pro** | Google | 2024.02 | 未公开 | 1M token 上下文 | ⭐⭐ API 调用，成本高 |
| **Claude 3.5 Sonnet** | Anthropic | 2024.06 | 未公开 | 视觉理解强，性能提升 | ⭐⭐ API 调用，延迟问题 |

**说明**: 闭源 API 模型虽然能力强，但存在延迟高、成本高、无法本地部署等问题，不适合直接用于 VLA 训练。可作为参考或用于数据标注、CoT 推理等辅助任务。

**2025 年闭源模型趋势**:
- **Gemini 2.5 Pro**: 目前排行榜第一，内置推理思考功能
- **Claude 3.7**: OCR 和图表解析能力大幅提升

### 5.6.4 经典模型（历史参考）

| 模型 | 机构 | 发布时间 | 特点 | VLA 影响 |
| :--- | :--- | :--- | :--- | :--- |
| **BLIP-2** | Salesforce | 2023.01 | Q-Former 架构创新 | ⭐ 早期 VLM，较少直接用于 VLA |
| **Flamingo** | DeepMind | 2022.04 | Perceiver Resampler, Gated Cross-Attention | ⭐⭐ 架构创新影响深远，但未直接用于 VLA |

### 5.6.5 VLA 训练选择指南

#### 快速选择

```
需要轻量级、单卡训练？
  ├─ 是 → PaliGemma 3B (首选)
  └─ 否 → LLaVA 7B/13B

只需要 Vision Encoder？
  └─ SigLIP (VLA 首选)

需要中文支持？
  └─ Qwen-VL 7B

需要边缘部署？
  └─ MiniCPM-V 2.4B

需要高分辨率输入？
  └─ InternVL 或 PaliGemma 896px 版本
```

#### 技术对比

| 特性 | PaliGemma 3B | LLaVA 7B | Qwen-VL 7B | SigLIP (Vision) |
| :--- | :--- | :--- | :--- | :--- |
| **训练效率** | ⭐⭐⭐⭐⭐ 单卡可训练 | ⭐⭐⭐ 需要多卡 | ⭐⭐⭐ 需要多卡 | ⭐⭐⭐⭐⭐ 仅 Vision |
| **推理速度** | ⭐⭐⭐⭐ 快 | ⭐⭐⭐ 中等 | ⭐⭐⭐ 中等 | ⭐⭐⭐⭐⭐ 极快 |
| **视觉理解** | ⭐⭐⭐⭐ 强 | ⭐⭐⭐⭐ 强 | ⭐⭐⭐⭐ 强 | ⭐⭐⭐⭐⭐ 最强 |
| **中文支持** | ⭐⭐ 一般 | ⭐⭐ 一般 | ⭐⭐⭐⭐⭐ 优秀 | - |
| **VLA 生态** | ⭐⭐⭐⭐⭐ 最常用 | ⭐⭐⭐⭐ 成熟 | ⭐⭐⭐ 较少 | ⭐⭐⭐⭐⭐ 最常用 |

#### 实际应用案例

1. **π0 (Pi-Zero)**: 使用 PaliGemma 3B 作为 VLM backbone，结合 Flow Matching 实现高频控制
2. **OpenVLA**: 使用 Llama 2 7B + SigLIP 组合，通过 LoRA 高效微调
3. **RT-2**: 使用 PaLI-X 55B（闭源），证明了 VLM 语义能力可迁移到机器人控制
4. **RDT**: 使用 SigLIP 作为 Vision Encoder，专注于视觉特征提取

### 5.6.6 集成建议

#### 使用 PaliGemma 3B 训练 VLA

```python
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import torch

# 1. 加载预训练模型
model = PaliGemmaForConditionalGeneration.from_pretrained(
    "google/paligemma-3b-pt-224",
    torch_dtype=torch.bfloat16
)
processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")

# 2. 获取多模态特征（用于 Action Head）
def get_vlm_features(images, text_instructions):
    inputs = processor(images=images, text=text_instructions, return_tensors="pt")
    outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[-1]  # [B, L, 2048]
    return hidden

# 3. 接 Action Head
action_head = nn.Linear(2048, action_dim * chunk_size)
actions = action_head(hidden[:, -1, :])  # 使用最后一个 token
```

#### 使用 SigLIP 作为 Vision Encoder

```python
from transformers import AutoProcessor, AutoModel
import torch

# 加载 SigLIP Vision Encoder
vision_encoder = AutoModel.from_pretrained("google/siglip-base-patch16-224")
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

# 提取视觉特征
def extract_vision_features(images):
    inputs = processor(images=images, return_tensors="pt")
    outputs = vision_encoder(**inputs)
    return outputs.last_hidden_state  # [B, N_patches, D]
```

### 5.6.7 Pre-training vs Fine-tuning vs Post-training

> **重要概念**: 在 VLA 训练中，这三个术语有明确的区别和顺序。

#### 训练阶段对比

| 阶段 | 英文 | 中文 | 数据来源 | 训练目标 | 典型方法 | VLA 应用 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Pre-training** | Pre-training | 预训练 | 大规模通用数据 (ImageNet, CLIP, 互联网图文) | 学习通用视觉/语言特征 | 自监督学习、对比学习 | VLM backbone (PaliGemma, SigLIP) |
| **Fine-tuning** | Fine-tuning | 微调 | 目标任务数据 (机器人示教数据) | 适配特定任务 | 监督学习 (BC), LoRA | OpenVLA, π0 在机器人数据上微调 |
| **Post-training** | Post-training | 后训练 | 交互收集的数据 (成功+失败轨迹) | 自我改进，超越示教 | Offline RL (Recap) | π*0.6 的 Recap 算法 |

#### 详细说明

**1. Pre-training (预训练)**

```
大规模数据 (ImageNet/CLIP/互联网图文)
        │
        ▼
  学习通用特征
        │
        ▼
  预训练模型 (如 PaliGemma 3B)
```

- **目标**: 在大规模数据上学习通用的视觉和语言理解能力
- **数据**: 通常不需要标注，使用自监督或对比学习
- **结果**: 得到一个具备基础能力的模型
- **VLA 应用**: 
  - PaliGemma 3B 在互联网图文数据上预训练
  - SigLIP 在图像-文本对上进行对比学习预训练

**2. Fine-tuning (微调)**

```
预训练模型 (PaliGemma 3B)
        │
        ▼
  目标任务数据 (机器人示教)
        │
        ▼
  微调后模型 (适配机器人控制)
```

- **目标**: 在预训练模型基础上，用目标任务数据微调，使其适配特定任务
- **数据**: 需要标注的示教数据 (observation-action pairs)
- **方法**: 
  - **Full Fine-tuning**: 更新所有参数（显存需求大）
  - **LoRA/QLoRA**: 只训练少量参数（推荐）
- **VLA 应用**:
  - OpenVLA: 在机器人数据上 LoRA 微调
  - π0: 在机器人数据上微调 PaliGemma

**3. Post-training (后训练)**

```
微调后模型 (π0.6)
        │
        ▼
  机器人交互收集数据 (成功+失败)
        │
        ▼
  Offline RL (Recap 算法)
        │
        ▼
  改进后模型 (π*0.6, 超越示教)
```

- **目标**: 通过分析成功和失败轨迹，自我改进，超越人类示教水平
- **数据**: 机器人实际运行收集的数据（包含成功和失败案例）
- **方法**: Offline RL (如 Recap 算法)
- **特点**: 
  - 不仅学习"怎么做"，还学习"怎么做得更好"
  - 可以超越人类示教者的水平
- **VLA 应用**: π*0.6 的 Recap 算法

#### 完整训练流程示例 (π0.6 → π*0.6)

```python
# Phase 1: Pre-training (通常由模型提供方完成)
# 使用大规模数据训练 PaliGemma 3B
pretrained_vlm = load_pretrained("google/paligemma-3b-pt-224")

# Phase 2: Fine-tuning (在机器人数据上微调)
# 使用示教数据微调
robot_demos = load_robot_demonstrations()  # 人类示教数据
finetuned_model = fine_tune(pretrained_vlm, robot_demos, method="LoRA")
# 得到 π0.6

# Phase 3: Post-training (Recap, 自我改进)
# 机器人交互收集数据
interaction_data = robot.collect_data()  # 包含成功和失败轨迹

# 使用 Offline RL 改进
improved_model = recap_algorithm(finetuned_model, interaction_data)
# 得到 π*0.6
```

#### 关键区别总结

| 特性 | Pre-training | Fine-tuning | Post-training |
| :--- | :--- | :--- | :--- |
| **数据来源** | 通用大规模数据 | 目标任务示教数据 | 交互收集的成功+失败数据 |
| **训练目标** | 学习通用特征 | 适配特定任务 | 自我改进，超越示教 |
| **学习方式** | 自监督/对比学习 | 监督学习 (BC) | Offline RL |
| **是否必需** | ✅ 是 (模型基础) | ✅ 是 (任务适配) | ⚠️ 可选 (性能提升) |
| **典型时间** | 数周/月 (大规模) | 数小时/天 | 数天/周 (持续改进) |

#### 面试常见问题

**Q: Pre-training 和 Fine-tuning 的区别是什么？**

A:
- **Pre-training**: 在大规模通用数据上学习基础能力（如视觉理解、语言理解）
- **Fine-tuning**: 在预训练模型基础上，用目标任务数据微调，使其适配特定任务（如机器人控制）

**Q: Post-training 和 Fine-tuning 的区别是什么？**

A:
- **Fine-tuning**: 使用人类示教数据，学习"怎么做"（模仿学习）
- **Post-training**: 使用交互收集的成功+失败数据，学习"怎么做得更好"（强化学习），可以超越人类示教水平

**Q: 为什么需要 Pre-training？**

A: 
- 机器人数据稀缺且昂贵，从头训练需要大量数据
- Pre-training 让模型具备通用能力，只需少量机器人数据即可适配
- 类似人类先学基础知识，再学专业技能

### 5.6.8 常见问题

**Q: 为什么 VLA 首选 PaliGemma 3B 而不是更大的 LLaVA?**

A: 三个原因:
1. **效率**: 3B 参数可在单卡 (24GB) 训练/推理，满足机器人实时性要求
2. **SigLIP**: 比 CLIP 更好的细粒度视觉理解
3. **模块化**: Vision/Language 解耦，方便接 Action Head

**Q: SigLIP 和 CLIP 的区别是什么？**

A: 
- **损失函数**: CLIP 使用 Softmax + Cross-Entropy (InfoNCE)，SigLIP 使用 Sigmoid + Binary CE
- **Batch 依赖**: CLIP 的 Softmax 需要对比 batch 内所有样本，SigLIP 的 Sigmoid 每对独立计算
- **扩展性**: SigLIP 更适合大 batch 训练，负样本利用更高效

**Q: 如何选择 Vision Encoder 和 LLM 的组合？**

A:
- **轻量级**: PaliGemma 3B (SigLIP + Gemma 2B)
- **平衡**: LLaVA (CLIP/ViT + Llama 2 7B)
- **自定义**: SigLIP (Vision) + 任意 LLM (Language)

**Q: 中文 VLA 任务应该选择哪个 VLM？**

A: 推荐 **Qwen2.5-VL 7B**（🆕 2025.03），中文支持最好，数学推理能力强，支持任意分辨率和长视频。如果资源受限，可选择 **Qwen2.5-VL 3B** 版本。

**Q: 有哪些 2025 年最新的 VLM 更新值得关注？**

A: 
- **Qwen2.5-VL** (2025.03): 阿里巴巴最新版本，**2025 SOTA**，数学推理强，支持任意分辨率
- **Eagle 2.5** (2025.04): NVIDIA 发布，长上下文多模态，Video-MME 72.4%
- **Seed 1.5-VL** (2025.05): 字节跳动发布，媲美 Gemini 2.5 Pro，GUI 交互强
- **GLM-4.5V** (2025): 智谱AI，MoE 架构，3D 空间推理
- **Llama 4** (2025.04): Meta 发布，10M token 上下文，多模态 MoE 架构
- **PLM** (2025.05): Meta 开源视觉语言模型

**Q: 2025年闭源 API 模型有哪些更新？**

A:
- **Gemini 2.5 Pro** (2025.03): Google 发布，排行榜第一，内置思考功能
- **Claude 3.7 Vision** (2025.02): Anthropic 发布，高精度 OCR 和图表解析

---

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

