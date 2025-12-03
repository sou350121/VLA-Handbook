# 触觉感知与 VLA (Tactile VLA)

在机器人操作中，视觉 (Vision) 往往不足以完成所有任务。对于接触密集型 (Contact-rich) 任务（如在黑暗中摸索物体、精密装配、判断物体材质），**触觉 (Tactile)** 是不可或缺的模态。

2024-2025 年，VLA 领域开始爆发 "Vision-Tactile-Language-Action" 的研究，旨在赋予机器人"触觉语义"理解能力。

## 1. 为什么需要触觉? (Why Tactile?)
- **视觉遮挡 (Occlusion)**: 当机械手抓取物体时，手掌会挡住摄像头视线。此时只有触觉能提供反馈。
- **物理属性感知**: 视觉无法直接判断物体的软硬、摩擦力、重量。
- **微米级控制**: 视觉通常有毫米级误差，而触觉传感器 (如 GelSight) 可以提供微米级的纹理信息。

## 2. 核心传感器技术

### 2.1 GelSight
MIT 开发的高分辨率光学触觉传感器，是触觉 VLA 研究的基石。

| 参数 | 规格 |
| :--- | :--- |
| 分辨率 | ~40 微米 |
| 尺寸 | 约 30×30mm 感知面 |
| 输出 | RGB 图像 + 深度图 |
| 帧率 | 30-60 FPS |
| 价格 | ~$500-1000 |

**原理**: 内部有弹性体 (Elastomer) + LED 光源 + 摄像头。当物体接触弹性体时，表面变形改变光照分布，摄像头捕捉这些变化重建接触几何。

### 2.2 DIGIT (Meta AI)
> **论文**: [DIGIT: A Novel Design for a Low-Cost Compact High-Resolution Tactile Sensor](https://arxiv.org/abs/2005.14679) (RSS 2020)
> **官网**: [digit.ml](https://digit.ml)

Meta AI (FAIR) 开发的**开源**紧凑型触觉传感器，专为机器人手指设计。

**代码库**:

| 仓库 | 说明 | 链接 |
| :--- | :--- | :--- |
| **digit-interface** | Python 驱动接口 | [GitHub](https://github.com/facebookresearch/digit-interface) |
| **TACTO** | DIGIT 仿真器 (PyBullet) | [GitHub](https://github.com/facebookresearch/tacto) |
| **PyTouch** | 触觉 ML 库 | [GitHub](https://github.com/facebookresearch/PyTouch) |
| **Sparsh** | 预训练触觉模型 | [GitHub](https://github.com/facebookresearch/sparsh) |
| **3DCal** | 触觉传感器标定工具 | [GitHub](https://github.com/3DCal/3DCal) |

| 参数 | 规格 |
| :--- | :--- |
| 分辨率 | 640×480 RGB |
| 尺寸 | **20×27×18mm** (极紧凑) |
| 重量 | ~20g |
| 帧率 | 60 FPS |
| 接口 | USB-C |
| 成本 | **~$15** (开源 BOM) |

**优势**:
- **开源硬件**: 完整 CAD 设计、制造指南公开，可自行制作
- **紧凑设计**: 专为机器人手指优化，可安装在 Allegro Hand 等灵巧手上
- **低成本**: 材料成本仅 $15，适合大规模部署
- **高帧率**: 60 FPS 支持实时控制
- **预训练模型**: Meta 提供训练好的触觉表征模型 (如 T3, Sparsh)
- **适配接口**: 官方提供 PyTorch 接口、ROS 驱动，生态完善

**局限性**:
- **一致性差**: 每个传感器的光学特性略有不同，需要单独标定
- **无力输出**: 只提供形变图像，**不直接输出接触力** (需算法估计)
- **弹性体磨损**: 长时间使用后弹性体会老化

**与 GelSight 对比**:

| 特性 | GelSight | DIGIT |
| :--- | :--- | :--- |
| 分辨率 | 更高 (~40μm) | 较低 (像素级) |
| 尺寸 | 较大 | **极紧凑** |
| 成本 | 高 ($500+) | **极低 ($15)** |
| 开源 | 部分 | **完全开源** |
| 力输出 | 可估计 | **无直接输出** |
| 一致性 | 较好 | **传感器间差异大** |
| 生态 | 学术为主 | **Meta 预训练模型** |
| 适用场景 | 精密检测、研究 | 灵巧手、大规模部署 |

### 2.3 千觉 GelStereo (Xense Robotics)
> **公司**: 千觉机器人科技（上海）有限公司，成立于 2024 年 5 月
> **官网**: [xense-robotics.com](https://www.xense-robotics.com)

基于**双目立体视觉 (Binocular Stereo)** 的高分辨率多模态触觉传感器。

| 参数 | 规格 |
| :--- | :--- |
| 感知密度 | **人类手指 800 倍** |
| 模态 | 三维力觉 + 动觉 + 滑觉 + 形貌 |
| 原理 | 双目相机 + 弹性体 |
| 输出 | 3D 点云 + 力分布图 |

**核心技术 - GelStereo 原理**:

```
        左相机                右相机
           \                  /
            \   弹性体变形   /
             \     ↓       /
              ┌─────────┐
              │ 接触区域 │  ← 物体接触
              └─────────┘
                  ↓
           双目视差 → 3D 重建
```

1. **双目立体匹配**: 两个相机从不同角度拍摄弹性体表面，通过视差计算深度
2. **亚像素精度**: 相比单目 GelSight，双目可实现更精确的 3D 形貌重建
3. **实时性**: 无需复杂的光度立体 (Photometric Stereo) 算法，速度更快

**与 GelSight 对比**:

| 特性 | GelSight (单目) | GelStereo (双目) |
| :--- | :--- | :--- |
| 3D 重建方法 | 光度立体 (需多色光) | **立体匹配** (单色即可) |
| 计算复杂度 | 高 | **低** |
| 光照要求 | RGB 三色光 | 单色光 |
| 深度精度 | 依赖标定 | **几何约束更强** |

---

### 2.4 戴盟 DM-Tac (DMRobot)
> **公司**: 戴盟机器人，专注视触觉技术
> **官网**: [dmrobot.com](https://www.dmrobot.com)

全球首款**多维高分辨率高频率**视触觉传感器系列。

#### DM-Tac W 传感器

| 参数 | 规格 |
| :--- | :--- |
| 感知密度 | **4 万单元/cm²** (人类 240/cm²) |
| 模态 | 形貌 + 纹理 + 软硬 + 滑移 + 三维力 |
| 耐久性 | **500 万次按压测试** |
| 厚度 | 毫米级 |

**核心技术 - 单色光视触觉**:

传统 GelSight 使用 **RGB 三色光** + 光度立体 (Photometric Stereo) 重建 3D 形貌。戴盟采用**单色光**方案，简化硬件设计：

| 方案 | 三色光 (RGB) | 单色光 (戴盟) |
| :--- | :--- | :--- |
| 光源 | 红/绿/蓝 LED (多角度) | 单色 LED |
| 3D 重建 | 光度立体 (解析解) | 数据驱动 / 学习方法 |
| 硬件复杂度 | 需精确 LED 布局 | **更简单** |
| 标定 | 需标定 RGB 通道 | **更简单** |

**优势**:
- **硬件简化**: 减少 LED 数量和布局要求
- **工业级耐用**: 500 万次按压，远超学术级传感器
- **高密度**: 4 万单元/cm² 的高密度触点

> 注：单色光损失了光度立体的多方向光照信息，需依赖学习算法补偿。

#### DM-Hand1 灵巧手
集成 DM-Tac 传感器的五指灵巧手，每个指尖都有触觉感知。

#### DM-EXton 遥操作系统
穿戴式数据采集设备，用于收集高质量人类示教数据，为模仿学习提供触觉标注。

---

### 2.5 其他触觉传感器
- **TacTip**: Bristol 大学，基于生物启发的软性触觉传感器
- **BioTac**: SynTouch 公司，多模态传感器 (压力 + 温度 + 振动)
- **ReSkin**: Meta AI，磁性薄膜触觉传感器，可贴附于任意表面
- **Taxim**: CMU，基于有限元仿真的触觉传感器

## 3. 最新模型进展 (2024-2025)

### 3.1 Tactile-VLA / TVL (ICML 2024)
> **论文**: [A Touch, Vision, and Language Dataset for Multimodal Alignment](https://arxiv.org/abs/2402.13232)
> **作者**: Max Fu, Gaurav Datta, et al. (UC Berkeley)
> **代码**: [GitHub - Max-Fu/tvl](https://github.com/Max-Fu/tvl)

**核心贡献**: 首个大规模触觉-视觉-语言对齐数据集和基准模型。

#### 数据集 (TVL Dataset)

| 统计量 | 数值 |
| :--- | :--- |
| 样本数 | **44,000** 三元组 |
| 材质种类 | 100+ 种 |
| 传感器 | GelSight Mini |
| 标注类型 | 自然语言描述 |

**数据示例**:
```
触觉图像: [GelSight 压痕图]
视觉图像: [物体 RGB 图]
语言描述: "The surface feels rough and rigid, like sandpaper"
```

**数据采集流程**:
1. 机器人用 GelSight 传感器接触物体表面
2. 同时拍摄物体的 RGB 图像
3. 人类标注员用自然语言描述触觉感受

#### 模型架构详解

```
┌─────────────────────────────────────────────────────────┐
│                    TVL 模型架构                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   Vision Image ──→ CLIP ViT-L/14 ──→ [CLS] Token       │
│                         ↓                               │
│                   Vision Embedding (768d)               │
│                         ↓                               │
│                   ┌─────────────┐                       │
│                   │  Projection │ (MLP)                 │
│                   │    Layer    │                       │
│                   └─────────────┘                       │
│                         ↓                               │
│                  Joint Embedding Space                  │
│                         ↑                               │
│                   ┌─────────────┐                       │
│                   │  Projection │ (MLP)                 │
│                   │    Layer    │                       │
│                   └─────────────┘                       │
│                         ↑                               │
│                  Tactile Embedding (768d)               │
│                         ↑                               │
│  Tactile Image ──→ ViT-B/16 (MAE pretrained) ──→ [CLS] │
│                                                         │
│                  Joint Embedding Space                  │
│                         ↓                               │
│                   Cross-Attention                       │
│                         ↓                               │
│                  Language Decoder (GPT-2)               │
│                         ↓                               │
│                 "rough and cold..."                     │
└─────────────────────────────────────────────────────────┘
```

#### 训练策略

**Stage 1: Tactile Encoder 预训练 (MAE)**
```python
# 随机 Mask 75% 的 Tactile Patches
masked_patches = random_mask(tactile_image, ratio=0.75)
# 重建被 Mask 的 Patches
reconstruction_loss = MSE(decoder(encoder(masked_patches)), original)
```

**Stage 2: 对比学习对齐 (Contrastive Alignment)**

```math
L_{align} = -log \frac{exp(sim(t_i, v_i) / τ)}{\sum_{j} exp(sim(t_i, v_j) / τ)}
```

- `t_i`: 触觉 Embedding
- `v_i`: 对应的视觉 Embedding
- `τ`: 温度参数 (0.07)

**Stage 3: 语言生成微调**
```python
# 触觉 + 视觉 → 语言描述
loss = CrossEntropy(decoder(tactile_emb, vision_emb), text_tokens)
```

#### 下游任务与性能

| 任务 | 指标 | TVL 性能 |
| :--- | :--- | :--- |
| 材质分类 | Accuracy | 78.3% |
| 触觉-语言检索 | R@1 | 45.2% |
| 触觉问答 | BLEU-4 | 32.1 |
| Zero-shot 材质识别 | Accuracy | 61.7% |

#### 核心洞察

1. **触觉-视觉互补**: 视觉擅长识别物体类别，触觉擅长判断物理属性
2. **语言作为桥梁**: 自然语言描述使触觉语义可解释、可迁移
3. **MAE 预训练有效**: 触觉图像的纹理特征适合 Masked Autoencoder

**意义**: 为 VLA 提供了触觉语义理解的基础，使机器人能够"用语言描述触觉"。

---

### 3.2 VLA-Touch (2025)
> **论文**: [VLA-Touch: Enhancing Generalist Robot Policies with Dual-Level Tactile Feedback](https://arxiv.org/abs/2502.xxxxx)
> **核心思想**: 双层反馈机制 (Dual-level Feedback)。

- **背景**: 现有的 VLA (如 RT-2) 缺乏触觉通道。直接微调大模型成本太高。
- **架构细节**:
    1.  **High-level Planning (VLM)**:
        - **Tactile Encoder**: 使用 **ResNet-50** 或 **ViT-B** 编码 GelSight 图像。
        - **Tactile-Language Model (TLM)**: 预训练一个 Decoder-only Transformer，将触觉 Embedding 翻译成自然语言描述 (e.g., "I feel a smooth, hard surface")。
        - **Prompting**: 将生成的触觉描述作为 Prompt 喂给 VLM (如 GPT-4o 或 Gemini)，辅助其进行推理。
    2.  **Low-level Control (Diffusion)**:
        - **Fusion**: 使用 **FiLM (Feature-wise Linear Modulation)** 将触觉特征注入到 Diffusion Policy 的 U-Net 中。
        - **Action Refinement**: 触觉信号主要用于修正动作的最后几毫米 (Contact Phase)，确保接触力适中。
- **优势**: 无需重新训练整个 VLA，即插即用。

### 3.3 OmniVTLA (2025)
> **论文**: [OmniVTLA: A Unified Vision-Tactile-Language-Action Model](https://arxiv.org/abs/2503.xxxxx)
> **核心思想**: 统一的视触觉语言动作模型 (Unified Vision-Tactile-Language-Action Model)。

- **架构细节**:
    - **Unified Tokenization**:
        - **Vision**: ViT Patch Embeddings.
        - **Tactile**: 同样使用 ViT 处理触觉图像，将其 Patch 化为 Tactile Tokens。
        - **Language**: 文本 Token。
    - **Semantic Alignment (Contrastive Learning)**:
        - 关键创新在于**语义对齐**。它不仅学习触觉的物理特征，还学习触觉的语义描述 (e.g., "slippery", "rough")。
        - **Loss Function**: $\mathcal{L} = \mathcal{L}_{action} + \lambda \mathcal{L}_{align}$。其中 $\mathcal{L}_{align}$ 是 InfoNCE Loss，拉近触觉 Embedding 与对应的材质描述文本 Embedding 的距离。
- **训练**: 使用大规模的多模态数据集 (包含图像、触觉图、语言指令、动作)。
- **能力**: 能够执行 "Pick up the softest object" (抓起最软的物体) 这种需要跨模态推理的任务。

## 4. 挑战与未来
1.  **数据稀缺**: 相比于图像数据，高质量的触觉-语言对 (Tactile-Language Pairs) 非常少。
2.  **Sim-to-Real**: 触觉仿真非常困难 (涉及复杂的软体形变)，目前主要依赖真机数据收集。
3.  **硬件成本**: 高分辨率触觉传感器 (如 GelSight) 依然昂贵且易损耗。

## 5. 深度解析: ResNet vs ViT for Tactile
在触觉 VLA 中，选择 ResNet 还是 ViT 作为触觉编码器 (Tactile Encoder) 是一个关键的设计决策。这不仅仅是"CNN vs Transformer"的问题，而是关乎**触觉信号的物理特性**如何被编码。

### 5.1 ResNet (CNN): 纹理与几何的专家
ResNet 在处理 GelSight 这类**基于光学 (Optical-based)** 的触觉传感器时表现出色，原因在于其**归纳偏置 (Inductive Bias)** 与触觉图像的特性高度契合。

*   **技术细节**:
    *   **平移不变性 (Translation Invariance)**: 触觉特征（如物体表面的凸起）可能出现在传感器的任何位置。ResNet 的**权重共享 (Weight Sharing)** 卷积核保证了无论凸起在哪里，提取的特征都是一致的。
    *   **局部性 (Locality)**: 触觉感知的核心是**接触 (Contact)**。接触通常发生在局部区域。ResNet 的卷积核 (e.g., 3x3) 强制模型关注局部像素的梯度变化，这对于检测**边缘 (Edges)**、**纹理 (Textures)** 和 **滑移 (Slip)** 至关重要。
    *   **层级特征 (Hierarchical Features)**: ResNet 通过 Pooling 不断下采样，自然地形成了从"微观纹理"到"宏观形状"的特征金字塔。这对于判断物体材质（微观）和抓取稳定性（宏观）都很有用。

### 5.2 ViT (Transformer): 全局接触与多模态统一
ViT 在 OmniVTLA 等最新模型中更受欢迎，主要是为了**多模态对齐**和**全局上下文**。

*   **技术细节**:
    *   **Patchify & Linear Projection**: ViT 将 $224 \times 224$ 的触觉图像切分为 $16 \times 16$ 的 Patches，展平后通过线性层映射为 Embedding。这一步彻底打破了像素的网格结构，使其能与文本 Token 在同一向量空间中交互。
    *   **全局感受野 (Global Receptive Field)**: Self-Attention 允许每一个 Tactile Patch 在第一层就与其他所有 Patch 交互。这对于理解**多点接触 (Multi-point Contact)** 非常关键。例如，当手指捏住物体时，指尖两侧的受力分布是相关的，ResNet 需要堆叠多层才能"看"到这种长距离关联，而 ViT 一眼就能看到。
    *   **位置编码 (Positional Encoding)**: 由于 ViT 没有卷积的归纳偏置，它必须依赖可学习的位置编码来理解"哪里是上，哪里是下"。在触觉中，绝对位置往往对应着机械手的具体部位 (e.g., 指尖 vs 指腹)，这对控制很重要。

### 5.3 核心差异对比表

| 特性 | ResNet (Tactile) | ViT (Tactile) |
| :--- | :--- | :--- |
| **归纳偏置** | 强 (平移不变, 局部性) | 弱 (需数据学习) |
| **擅长特征** | **高频纹理** (Texture), 边缘, 局部形变 | **低频分布** (Force Distribution), 全局接触模式 |
| **数据需求** | 低 (几千张图即可收敛) | 高 (需 MAE 预训练或 ImageNet 迁移) |
| **多模态融合** | 需通过 Pooling 压缩成向量后融合 | **Token 级融合** (可与 Text Token 拼接) |
| **典型应用** | 材质识别, 滑移检测 (Slip Detection) | 复杂操作策略 (Manipulation Policy), 跨模态推理 |

## 6. 面试常见问题

**Q1: 触觉图像 (Tactile Image) 和普通 RGB 图像有什么区别?**

触觉图像通常反映的是**几何形状 (Geometry)** 和 **受力分布 (Force Distribution)**，对光照变化不敏感，但对接触极其敏感。处理时通常不需要复杂的颜色增强，但需要关注纹理细节。

---

**Q2: 如何将触觉融入 VLA?**

三种主流方法：
1. **通道拼接**: 将触觉图像作为额外视觉通道 (Concat)
2. **Cross-Attention**: 触觉特征作为 Key/Value 注入 Policy
3. **多模态对齐**: 像 Tactile-VLA/OmniVTLA 那样将触觉、视觉、语言对齐到统一空间

---

**Q3: DIGIT vs GelSight 怎么选?**

| 场景 | 推荐 | 原因 |
| :--- | :--- | :--- |
| 灵巧手操作 | **DIGIT** | 体积小、重量轻、成本低 |
| 精密检测/研究 | **GelSight** | 分辨率更高 |
| 大规模数据采集 | **DIGIT** | 开源、便宜、可批量制作 |
| 工业应用 | 看需求 | GelSight 稳定性更好 |

---

**Q4: 触觉 Sim-to-Real 为什么难?**

1. **软体仿真复杂**: 弹性体形变涉及非线性 FEM，计算成本高
2. **接触模型不精确**: 摩擦、滑移的物理建模困难
3. **传感器特性**: 每个传感器的光学特性略有不同
4. **解决方案**: 域随机化 (Domain Randomization)、真实数据微调

---

**Q5: GelStereo (千觉) vs GelSight 的技术差异?**

| 维度 | GelSight | GelStereo |
| :--- | :--- | :--- |
| 3D 重建 | 光度立体 (Photometric Stereo) | **双目立体匹配** |
| 光源 | RGB 三色 LED | 单色 LED |
| 相机数 | 1 个 | **2 个** |
| 原理 | 根据不同光照下的亮度变化推断法向量 | 根据双目视差直接计算深度 |
| 优势 | 纹理细节更丰富 | **计算更快、几何约束更强** |

---

**Q6: DM-Tac (戴盟) 的单色光技术有什么优势?**

传统 GelSight 使用 RGB 三色光从不同角度照射弹性体，通过**光度立体 (Photometric Stereo)** 算法从颜色通道分离出三个方向的光照信息，重建表面法向量。

单色光方案的 trade-off：

| 维度 | RGB 三色光 | 单色光 |
| :--- | :--- | :--- |
| 信息量 | 3 个光照方向 | 1 个光照方向 |
| 重建精度 | 理论上更高 | 依赖算法补偿 |
| 硬件复杂度 | 需精确 LED 布局 | **更简单** |
| 标定难度 | 需标定每个颜色通道 | **更简单** |
| 成本 | 较高 | **较低** |

**实际考量**: 单色光损失了部分法向量信息，但可通过深度学习端到端补偿。对于抓取任务，高密度触点 (4万/cm²) 可能比精确 3D 重建更重要。

> ⚠️ 注：戴盟的具体算法细节未公开，上述为基于公开信息的推测。

---

**Q7: 如何选择触觉传感器?**

| 场景 | 推荐 | 理由 |
| :--- | :--- | :--- |
| 学术研究/快速原型 | **DIGIT** | 开源、便宜、**Meta 预训练模型** |
| 需要力反馈 | **GelSight / BioTac** | DIGIT 无直接力输出 |
| 高精度 3D 重建 | **GelSight** | 分辨率最高 |
| 灵巧手集成 | **DM-Tac** | 超薄、耐用、高密度 |
| 机械臂末端 | **GelStereo** | 双目 3D、实时性好 |
| 大面积覆盖 | **ReSkin** | 磁性薄膜、可贴附 |
| 批量部署 (一致性要求高) | **GelSight / DM-Tac** | DIGIT 传感器间差异大 |

---

**Q8: DIGIT 没有力输出怎么办?**

两种解决方案：
1. **基于图像的力估计**: 训练 CNN/MLP 从形变图像回归接触力
   ```python
   # 伪代码
   force = ForceEstimator(tactile_image)  # 输出 Fx, Fy, Fz
   ```
2. **外部力传感器融合**: 在手腕处加装六维力传感器 (如 ATI)，与 DIGIT 图像融合

Meta 的 **Sparsh** 模型已包含力估计预训练，可直接使用。


---
[← Back to Theory](./README.md)
