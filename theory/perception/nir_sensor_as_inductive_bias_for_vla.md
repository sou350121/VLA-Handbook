# 传感器选择即架构设计：近红外作为 VLA 的硬件归纳偏置

> 这不是一篇论文解析，而是一个研究方向的 first-principle 论证——为什么近红外（NIR）可能是比 RGB 更适合机器人操作的视觉模态。

<table><tr><td>

**整理**：Claude Opus 4.6 × [Pulsar 照见](https://github.com/sou350121/Pulsar-KenVersion) · 2026-04-15
**前置阅读**：[3D 优先：VGA × Spark 2.0 表征革命](3d_first_principle_vga_spark_embodied_representation_revolution.md) · [VGA 论文解析](vga_vision_geometry_action_over_language_video_2026.md)

</td></tr></table>

---

## 0. 核心主张

> **传感器选择不是工程细节——它是架构设计的第一步。选择 RGB 等于在模型中引入"外观偏置"；选择 NIR 等于引入"结构偏置"。对于物理操作任务，结构偏置是正确的。**

---

## 1. 一个被忽视的问题：为什么所有 VLA 都用 RGB？

从 RT-1 到 π₀.₅，从 OpenVLA 到 GR00T-N1.6，所有 VLA 模型的视觉输入都是 RGB 图像。没有人质疑过这个选择。

原因不是"RGB 最好"，而是：
- RGB 相机最便宜（$20）
- 互联网预训练数据全是 RGB
- 学术界的惯性——"大家都用 RGB"

但 [VGA](vga_vision_geometry_action_over_language_video_2026.md) 刚刚证明了：**换一个更好的 backbone（3D 几何）能用 36K 场景打败万亿 token 训练的 VLM**。如果 backbone 的选择这么重要，为什么没人想过——**传感器的选择**是否同样重要？

---

## 2. First Principle：任务相关信息密度

### 什么是"好的传感器"？

不是通道越多越好。不是分辨率越高越好。而是：

$$
\text{传感器质量} \propto \frac{I(X; Y)}{H(X)}
$$

其中 X 是传感器输出，Y 是任务目标（动作），H(X) 是传感器的总信息量。

**好的传感器 = 在有限带宽中，最大化与任务相关的信息占比。**

### 对操作任务做信息分解

| 信息维度 | 操作是否需要？ | RGB 编码量 | NIR 编码量 |
|---------|:-----------:|:---------:|:---------:|
| **3D 形状** | ✅ 核心需求 | 间接（需从阴影、遮挡推断） | 直接（结构光 → 深度） |
| **物体位置** | ✅ 核心需求 | 有，但受光照干扰 | 有，且光照不变 |
| **表面法线** | ✅ 抓取角度依赖于此 | 从阴影弱推断 | 结构光直接提供 |
| **材质反射率** | ✅ 力控需要 | 被颜色淹没 | NIR 反射率区分金属/塑料/织物 |
| 颜色 | ❌ 红杯蓝杯抓法一样 | **3 通道中 ~60% 带宽** | 0%（单通道无色彩） |
| 纹理细节 | ❌ 木纹不影响抓取 | 大量编码 | 弱化 |
| 光照/阴影 | ❌ 纯噪声 | 严重干扰 | NIR 带通滤波后不受可见光影响 |
| 镜面高光 | ❌ 纯噪声 | 金属/玻璃上严重 | 弱化（NIR 反射更漫） |

**RGB 三通道的大量带宽在编码颜色、纹理、光照——这些对操作任务都是噪声。NIR 的单通道天然过滤掉了这些噪声。**

---

## 3. 信息瓶颈理论：传感器是硬件级压缩

Tishby (2015) 的信息瓶颈原理：

$$
\min_Z \; I(X; Z) - \beta \cdot I(Z; Y)
$$

最优表征 Z 对输入 X 最大程度压缩（小 I(X;Z)），同时对目标 Y 最大程度保留（大 I(Z;Y)）。

**深度学习的本质就是在学这个压缩。** VLM backbone 用万亿 token 学会了"从 RGB 中忽略颜色和光照，提取语义"。VGA 的 VGGT 用 13,824 A100-hours 学会了"从 RGB 中忽略外观，提取 3D 几何"。

**但如果传感器本身就已经完成了这个压缩呢？**

```
RGB 路线：
  RGB（外观+结构混合信号）→ 学习压缩掉外观 → 提取结构 → 动作
  ↑ 需要大量预训练来"学会忽略"

NIR 路线：
  NIR（结构信号为主）→ 不需要压缩外观（传感器已滤除）→ 直接编码 → 动作
  ↑ 传感器完成了硬件级信息瓶颈
```

> **NIR 是一个"硬件级的信息瓶颈"——它在光电转换层面完成了 RGB 需要百万参数、万亿 token 才能学到的压缩。**

VGA 论文最惊人的消融是 random init → 6.4%（去掉 3D 预训练几乎归零）。这 92 分的差距，本质上是"学会从 RGB 中提取几何"的成本。NIR 能免费提供其中多少？

- 忽略颜色 → NIR 单通道 → **免费**
- 忽略光照变化 → NIR 带通滤波 → **免费**
- 忽略镜面高光 → NIR 漫反射特性 → **免费**
- 提取 3D 结构 → NIR 结构光 + RealSense → **免费**

---

## 4. NIR 的物理特性：为什么单通道是优势

### NIR 不是"少了两个通道的 RGB"

| 特性 | RGB | NIR (0.7-1.4μm) |
|------|-----|-----------------|
| 光源 | 被动（依赖环境光） | **主动**（红外投影仪，人眼不可见） |
| 通道数 | 3（R,G,B 编码颜色） | 1（灰度编码反射结构） |
| 分辨率 | 1080p+ | **同级**（标准 CMOS 传感） |
| 暗光性能 | 差（无光则无图） | **强**（主动 NIR 照明） |
| 透明物体 | 几乎不可见 | **部分可见**（NIR 穿透某些塑料） |
| 深度获取 | 需要双目或学习型估计 | **原生**（结构光/ToF） |
| 价格 | ~$20 | **~$0**（RealSense 同时输出 RGB+NIR+Depth） |
| 干扰源 | 光照变化、阴影、高光 | 仅受强阳光干扰（室内不受影响） |

### RealSense D435：零成本获取 NIR

Intel RealSense D435 的硬件架构：

```
                      RealSense D435
    ┌──────────────────────────────────────┐
    │  NIR 投影仪（主动结构光）              │
    │       ↓ 投射红外点阵                  │
    │  NIR 左相机 ←→ NIR 右相机（立体匹配）  │
    │       ↓                              │
    │  深度图（640×480 @ 90Hz）              │
    │                                      │
    │  RGB 相机（独立）                      │
    │       ↓                              │
    │  彩色图（1920×1080 @ 30Hz）            │
    └──────────────────────────────────────┘

    你同时拿到：RGB + NIR 左 + NIR 右 + Depth
    NIR 图像是深度计算的"中间产物"——一直在那里，只是没人用它做 VLA
```

**每个用 RealSense 的 VLA 实验室，都已经在丢弃 NIR 数据。**

---

## 5. 最深的 First Principle："传感器选择即架构设计"

### 三层递进

**Level 1**（[3D-first-principle](3d_first_principle_vga_spark_embodied_representation_revolution.md) 的结论）：
> 瓶颈在 3D 表征。3D backbone > VLM backbone。

**Level 2**（VGA 的推论）：
> 表征应与任务的物理本质对齐。几何表征 > 语义表征。

**Level 3**（本文的推论）：
> **最好的表征是在传感器层面就对齐了的。传感器选择 = 硬件级归纳偏置。**

| 路线 | 做了什么 | 成本 |
|------|---------|------|
| VLM 路线 | 用万亿 token 学习从 RGB 中提取语义 | 巨大（GPT 级预训练） |
| VGA 路线 | 用 13,824 A100-hrs 学习从 RGB 中提取 3D | 大（36K 3D 场景预训练） |
| **NIR 路线** | **传感器直接输出结构信号，跳过"从外观到几何"的学习** | **~零**（换个输入通道） |

---

## 6. NIR + 6 轴力传感：时序互补的物理对

NIR 和 F/T 的关系不是"两个传感器拼在一起"——它们覆盖了操作过程的不同时间阶段：

```
时间 →

┌─── 接近阶段 ───┐  ┌── 接触阶段 ──┐  ┌── 操作阶段 ──┐
│                │  │              │  │              │
│ NIR 主导       │  │ F/T 主导     │  │ 两者协同     │
│ 看到形状、位置  │  │ 感到力、摩擦  │  │ NIR 看全局    │
│ 规划接近路径    │  │ 调整握力      │  │ F/T 感局部    │
│ F/T ≈ 0（未接触）│  │ NIR 被遮挡↑   │  │ 互相验证      │
│                │  │              │  │              │
└────────────────┘  └──────────────┘  └──────────────┘
```

**RGB + F/T 是"外观 + 物理"的跨域融合（两种不同语言）。NIR + F/T 是"远程物理 + 接触物理"的同域融合（同一种语言的两个方言）。**

> 参考：[FAVLA](../tactile/favla_a_force_adaptive_fast_slow_vla_model_for_contact_rich_dissection.md) 用 1×A100 6 小时训练 RGB+F/T 融合，260 轨迹。[TaF-VLA](../tactile/taf_vla_tactile_force_alignment_2026.md) 收集了 1000 万触觉-力配对。但**没有人做过 NIR + F/T 融合**。

---

## 7. 提议的架构：物理优先双分支

```
┌──────────────────────────────────────────────────────┐
│              Physics Branch（物理分支）                │
│                                                      │
│   NIR (1ch)  ──→  Structure Encoder  ──┐             │
│                                        ├→ Physics    │
│   Depth (1ch) ──→  Geometry Encoder  ──┤   Fusion    │
│                                        │   Module    │
│   F/T (6-axis) → TCN/Transformer ──────┘             │
│                        ↓                             │
│                 Physics Features                     │
│           (3D 几何 + 材质 + 力学状态)                  │
└────────────────────────┬─────────────────────────────┘
                         │
                    Top Fusion → Action Head → 动作
                         │
┌────────────────────────┴─────────────────────────────┐
│             Semantic Branch（语义分支）                │
│                                                      │
│   RGB (3ch) + Language ──→ VLM (frozen LoRA)         │
│                        ↓                             │
│                Semantic Features                     │
│           (哪个物体、什么任务)                         │
└──────────────────────────────────────────────────────┘
```

**核心设计原则**：
1. NIR + Depth + F/T 先在物理空间融合（因为它们编码的是同一个物理现实）
2. RGB + Language 在语义空间融合（因为它们编码的是概念/意图）
3. 两个分支在顶层融合
4. 物理分支可以独立预训练（不需要语言标注）

**和"平等三路融合"的区别**：
- 平等融合假设所有模态等价 → 让网络自己学哪些该融合 → 需要更多数据
- 分层融合利用先验知识（物理 vs 语义）→ 减少学习负担 → 更少数据

---

## 8. 实验验证方案

### 实验 0：NIR vs RGB（最关键，1-2 天，零额外硬件）

RealSense D435 同时输出 RGB + NIR + Depth。用同一组 demo，训三个 VLA：

| 模型 | 输入 | 预期 |
|------|------|------|
| VLA-RGB | RGB (3ch) | baseline |
| VLA-NIR | NIR (1ch) | **如果 > RGB → 核心假设成立** |
| VLA-Depth | Depth (1ch) | 上限参考 |
| VLA-NIR+D | NIR + Depth (2ch) | **如果 ≈ Depth → NIR 接近纯几何** |

**如果 VLA-NIR > VLA-RGB**：证明单通道结构信号 > 三通道外观信号（传感器即偏置）。
**如果 VLA-NIR ≈ VLA-Depth**：NIR 单通道已接近纯几何输入的上限。

### 实验 1：NIR + F/T vs RGB + F/T（验证"物理对"假设）

| 模型 | 输入 | 预期 |
|------|------|------|
| VLA-RGB+F/T | RGB + 6轴力矩 | FAVLA baseline |
| **VLA-NIR+F/T** | NIR + 6轴力矩 | **如果收敛更快 → 同域融合更容易** |

比较：收敛速度（需要多少 demo）、最终成功率、接触阶段的成功率。

### 实验 2：物理分支可独立预训练吗？

| 阶段 | 做什么 |
|------|--------|
| Pre-train | 只用 NIR+Depth+F/T（无 RGB、无语言）做自监督 |
| Fine-tune | 加入 RGB+Language，在操作任务上微调 |
| 对比 | vs 从头三模态联合训练 |

如果预训练有效 → 物理表征可迁移，物理分支独立于语义。

### 算力估算（你的 8×A100）

| 实验 | GPU | 时间 |
|------|:---:|:----:|
| 实验 0（NIR vs RGB） | 8×A100 | **2 天** |
| 实验 1（NIR+F/T vs RGB+F/T） | 8×A100 | **3 天** |
| 实验 2（物理分支预训练） | 8×A100 | **5 天** |
| 消融 + 可视化 | 8×A100 | **2 天** |
| **总计** | | **~12 天** |

---

## 9. 待追问的开放问题

❓ **NIR 在室外失效。** 强阳光淹没结构光投影。户外操作场景（农业、建筑）怎么办？可能需要 NIR + LWIR 双红外方案。

❓ **语义信息的损失有多大？** NIR 不能区分"红色杯子"和"蓝色杯子"。如果任务要求颜色区分（"把红色的给我"），纯 NIR 失败。RGB 在语义分支中是否足以弥补？

❓ **NIR 结构光的深度精度有限。** RealSense D435 在 >4m 距离精度急剧下降。对于桌面操作（<1m）够用，但移动操作（房间尺度）可能不够。

❓ **预训练数据的 domain gap。** VGGT 在 RGB 上预训练。如果 VGA 用 NIR 输入，VGGT 的权重能迁移吗？可能需要在 NIR 数据上重新预训练——但 NIR 数据集远小于 RGB。

❓ **"信息瓶颈"假设是否过度简化？** 也许操作任务确实需要颜色信息的某些场景（透过颜色判断水果成熟度、通过颜色区分化学品）。单通道的"简洁"在这些场景可能是劣势。

---

## 10. Opus 的反思

### 🔮 如果 VLA-NIR > VLA-RGB 被验证，意味着什么？

意味着过去 4 年（2022-2026）所有 VLA 论文都在用一个次优的传感器配置。不是因为 RGB 不好，而是因为**没人质疑过默认选择**。

这和 VGA 的发现是平行的：所有人都在用 VLM backbone，没人质疑过是否应该用 3D backbone。VGA 质疑了，发现 3D > VLM。同样：所有人都用 RGB，没人质疑过是否应该用 NIR。

> **科学进步有时不是发现新东西，而是质疑旧假设。**

### 🔮 "传感器即偏置"可能是一个通用原则

不只是 NIR vs RGB。对于不同任务，最优传感器不同：

| 任务 | 最优传感器 | 原因 |
|------|-----------|------|
| 桌面操作 | NIR + F/T | 近距精确几何 + 接触力 |
| 仓库导航 | LiDAR + RGB | 远距全局几何 + 语义 |
| 食品分拣 | RGB + 高光谱 | 颜色 + 成分 |
| 水下操作 | 声呐 + F/T | 光学在水中失效 |

**没有"通用最优传感器"——只有"与任务对齐的最优传感器"。** 这意味着未来的 VLA 不应该假设固定的传感器配置，而应该有一个"传感器适配层"，能从任意传感器组合中提取任务相关信息。

### 🔮 最远的想象：传感器也可以被学习

当前的传感器（RGB、NIR、LiDAR）都是人类设计的。但如果有一个可微的"虚拟传感器"——它的光谱响应曲线、空间分辨率、时间采样率都是可学习的参数呢？

给定一个操作任务，系统自动学出"最优的传感器配置"——也许是在 850nm-950nm 之间的窄带 NIR + 低频 RGB 子采样 + 高频力矩。这是 neural architecture search 在传感器层面的推广——**neural sensor search**。

纯属科幻？也许。但如果"传感器即偏置"的原则被验证，这个方向就有了理论基础。

---

## 延伸阅读

| 方向 | 推荐 |
|------|------|
| 3D 优先原则 | [VGA × Spark 表征革命](3d_first_principle_vga_spark_embodied_representation_revolution.md) |
| VGA 论文 | [Vision-Geometry-Action](vga_vision_geometry_action_over_language_video_2026.md) |
| Spark 2.0 | [3DGS 网页渲染](spark_2_0_3dgs_web_renderer_world_labs_2026.md) |
| 触觉融合 | [FAVLA](../tactile/favla_a_force_adaptive_fast_slow_vla_model_for_contact_rich_dissection.md) · [TaF-VLA](../tactile/taf_vla_tactile_force_alignment_2026.md) · [OmniVTA](../tactile/omnivta_visuo_tactile_world_modeling_for_contact_rich_roboti_dissection.md) |
| 热红外 VLA | [ThermoAct (2026)](https://arxiv.org/abs/2603.25044) · [TIRO Workshop @ ICRA 2025](https://sites.google.com/view/tiro25/) |
| 3D 感知工具 | [点云与 SLAM](pointcloud_slam.md)（含 RealSense、Depth Anything V3、60+ 工具） |
| PI 访谈 | [Sergey Levine](../vla-core/physical_intelligence_sergey_levine_foundation_model_vision_2026.md)（中间层瓶颈） |
| VLA 研究主线 | [赌注清单](../vla-core/vla_research_mainline.md) |

---

[← Back to Explorer's Map](../README.md)
