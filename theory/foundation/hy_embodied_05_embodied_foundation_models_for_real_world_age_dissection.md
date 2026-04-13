# HY-Embodied-0.5：具身基础模型实战拆解 (HY-Embodied-0.5: Embodied Foundation Models for Real-World Agents)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-04-13
>
> **论文**: HY-Embodied-0.5: Embodied Foundation Models for Real-World Agents
> **链接**: https://arxiv.org/abs/2604.07430
> **代码**: https://github.com/Tencent-Hunyuan/HY-Embodied
> **核心定位**: 腾讯机器人 X 团队推出的具身 VLM 基础模型，用 MoT 架构 + 迭代式后训练 + 大对小程序蒸馏，在 2B 激活参数下实现边缘部署友好的具身智能，32B 版本性能对标 Gemini 3.0 Pro。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | MoT 架构分离视觉/语言计算路径 + 视觉 latent token + 迭代 RL/RFT 后训练，使 2B 模型在 22 个具身基准上平均 58.0%，超越同尺寸 SOTA |
| 适合精读 | 如果你在做边缘部署的具身 VLM、VLA 预训练、或小模型蒸馏到机器人控制器，重点看 §2 架构和 §4 后训练 |
| 可以跳过 | 如果你只关心纯 VLA 动作头设计或具体机器人控制实验，这篇距离中等（§6 仅概述 VLA 下游） |
| 落地可行性 | 中（代码开源，但 400M ViT + MoT-2B 需确认边缘设备实测延迟；蒸馏方案可复用） |
| 主要风险 | 训练数据细节（100M+ 样本构成）未完全公开；VLA 控制实验细节有限 |

💡 **X-Ray 开场**（2-3 句，非专家也能读懂）
这篇论文解决什么问题？通用 VLM 在具身场景（机器人操作、空间推理）上表现不佳，因为缺乏细粒度视觉感知和面向动作的推理能力。HY-Embodied-0.5 通过专用架构（MoT）和训练策略（迭代 RL + 蒸馏）让小型模型也能胜任真实机器人任务。对 VLA 研究者意味着什么？它提供了一套从 VLM 预训练到 VLA 微调的完整路径，尤其是小模型如何从大模型蒸馏具身推理能力的方案。

📍 **研究全景时间线**

```
[2023] LLaVA/VLM 兴起 → [2024] RoboVLA/π0 等 VLA 专用模型 → [2025] MoT 架构提出 → [2026] HY-Embodied-0.5 ← 当前位置
                                                          ↑
                                              本文创新：MoT + 迭代后训练 + 大→小蒸馏
```

**局限**: 未公开完整训练超参；VLA 控制实验仅在特定机器人平台验证；边缘设备实测延迟数据缺失。

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 组件 | HY-Embodied-0.5 MoT-2B | HY-Embodied-0.5 MoE-32B | 传统 VLM (如 Qwen3-VL) |
|------|------------------------|-------------------------|------------------------|
| 激活参数 | 2B | 32B |  varies |
| 总参数 | 4B | 407B | varies |
| 视觉编码器 | HY-ViT 2.0 (400M, native resolution) | 同左 | 通常固定分辨率 ViT |
| 架构类型 | Mixture-of-Transformers (MoT) | Mixture-of-Experts (MoE) | 标准 Transformer |
| 视觉/语言参数 | 非共享 (duplication) | 共享专家 | 共享 |
| 视觉 latent token | ✓ (带 global loss 监督) | ✓ | 通常无 |
| 注意力机制 | 视觉双向 / 语言单向 | 同左 | 通常全单向 |
| 边缘部署优化 | 是 (400M ViT 蒸馏自更大模型) | 否 | 部分支持 |

### 1.2 关键机制 (Key Mechanism)

**MoT 架构核心设计**:
- 在预训练开始前，复制 LLM 的 FFN 和 QKV 参数，初始化为预训练 LLM 权重
- 视觉 token 用复制的参数计算，文本 token 用原始参数计算
- 效果：在不显著增加计算开销下，视觉建模能力提升，同时避免重视觉训练导致的语言退化

**视觉 Latent Token**:
- 在每个视觉元素（图像/视频帧）末尾附加可学习的 latent token
- 预训练阶段用大 ViT 的全局特征监督该 token 输出（global loss）
- 作用：桥接视觉和语言内容，提升小 VLM 整体感知能力

**迭代式后训练范式**:
- 冷启动 SFT (100k CoT 样本) → RL (GRPO, 50k/轮动态数据) → RFT (拒绝采样 fine-tuning) → 循环
- RL 用任务感知奖励（几何密集奖励 + 精确匹配 + LLM judge 回退）
- RFT 从 RL 探索结果中筛选高质量推理轨迹，转为显式监督

⚡ **Eureka Moment**: 这篇论文最核心的洞见是**"具身能力不能只靠预训练数据堆砌，必须通过迭代式后训练（RL 探索 + RFT 固化）将偶发成功转化为稳定能力，再用 on-policy 蒸馏让小型模型继承大模型的推理风格"**——这解释了为什么同样尺寸的模型，HY-Embodied-0.5 在具身任务上显著优于通用 VLM。

### 1.3 信息流/架构图 (Flow / Diagram)

```
[输入图像] → HY-ViT 2.0 (400M) → 视觉 token + 离散码监督
                                    ↓
[文本 prompt] → Tokenizer → 文本 token
                                    ↓
                    ┌─────────────────────────────┐
                    │  Mixture-of-Transformers    │
                    │  ┌─────────┐ ┌─────────┐    │
                    │  │ Vision  │ │ Language│    │
                    │  │ QKV/FFN │ │ QKV/FFN │    │
                    │  │(双向注意)│ │(单向注意)│    │
                    │  └─────────┘ └─────────┘    │
                    │        ↑                    │
                    │  Visual Latent Token        │
                    │  (global loss 监督)          │
                    └─────────────────────────────┘
                                    ↓
                    ┌─────────────────────────────┐
                    │   后训练管道                 │
                    │  SFT → RL → RFT → 循环      │
                    │  (GRPO + 任务感知奖励)       │
                    └─────────────────────────────┘
                                    ↓
                    ┌─────────────────────────────┐
                    │   大→小程序蒸馏              │
                    │  On-Policy Distillation     │
                    │  (学生 rollout → 教师强制)   │
                    └─────────────────────────────┘
                                    ↓
                         [输出：推理/动作/感知]
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：

```
L_total = L_llm + L_vision + L_global
```

**训练目标分解**:
- `L_llm`: 标准自回归语言损失
- `L_vision`: 视觉 next-code 预测的交叉熵
- `L_global`: latent token 与教师 ViT 全局特征的负余弦相似度

**详细公式**:

```
L_vision = -1/N_v · Σ_{i=1}^{N_v} log p_i(z_i)

L_global = -(f_latent^T · f_teacher) / (||f_latent|| · ||f_teacher||)

L_RL(x) = -1/Σ|y_i| · Σ_{i=1}^{G} Σ_{t=1}^{|y_i|} min(ρ_{i,t}·A_i, clip(ρ_{i,t}, 1-ε_low, 1+ε_high)·A_i)

L_OPD = E_{x,y~π_s}[1/|y| · Σ_{t=1}^{|y|} KL(π_t(·|x,y_{<t}) || π_s(·|x,y_{<t}))]
```

**变量说明**:
| 符号 | 含义 |
|------|------|
| N_v | 视觉 token 数量 |
| p_i(z_i) | 第 i 个 token 预测目标离散码 z_i 的概率 |
| f_latent | latent token 映射后的隐藏状态 |
| f_teacher | 教师 ViT 提取的全局 CLS 特征 |
| G | RL 采样组大小 (16) |
| ρ_{i,t} | 策略比率 π_θ/π_θ_old |
| A_i | 组内归一化优势 (r_i - μ)/σ |
| π_t, π_s | 教师和学生的 next-token 分布 |

> 符号与本文/相关文档保持一致：L_vision 对应论文 Figure 2 中的 Vision Loss；L_global 对应 Global Loss；L_OPD 为 On-Policy Distillation 损失。

**直觉解释**:
- 预训练阶段三损失联合优化：语言理解 + 视觉重建 + 跨模态对齐
- 中训练及之后仅用 L_llm（冻结 ViT，专注具身 fine-tuning）
- RL 用组内相对优势避免不同任务间奖励尺度不可比问题
- On-policy 蒸馏关键：让学生在**自己生成的轨迹上**学习教师分布，而非仅模仿教师 rollout

## 3. 带数字走一遍：玩具例子 (Worked Example)

假设我们有一个简化的具身场景：机器人需要抓取桌上的红色杯子。

**输入**:
- 图像：640×480 RGB，桌上有红色杯子、蓝色杯子、黄色方块
- 指令："Pick up the red cup"

**前向传播过程**:

```
步骤 1: 视觉编码
  输入图像 → HY-ViT 2.0 → 196 个视觉 token (14×14 patch)
  同时输出离散码监督信号 (来自更大 ViT 的 teacher)

步骤 2: Latent Token 附加
  196 视觉 token + 1 latent token = 197 视觉侧 token

步骤 3: MoT 处理
  视觉 token (197 个) → Vision QKV/FFN (双向注意)
  文本 token ("Pick up..." 分词后约 8 个) → Language QKV/FFN (单向注意)
  注意：两套参数不共享，但输出在同一 embedding 空间

步骤 4: 推理生成
  LLM 自回归生成：
  t=1: "First" (P=0.92)
  t=2: " I" (P=0.88)
  t=3: " need" (P=0.85)
  ...
  t=15: "grasp" (P=0.79)
  t=16: " the" (P=0.91)
  t=17: " red" (P=0.94)
  t=18: " cup" (P=0.96)
  t=19: " at" (P=0.82)
  t=20: " coordinates" (P=0.88)
  t=21: " (320," (P=0.75)
  t=22: " 240)" (P=0.81)

步骤 5: RL 奖励计算 (训练阶段)
  假设标准答案："(315, 245)"
  使用归一化点距离奖励：
  r = 1 - sqrt((320-315)² + (240-245)²) / max_distance
    = 1 - sqrt(50) / 400 ≈ 0.98
```

**蒸馏过程** (大→小):

```
学生模型 (MoT-2B) rollout:
  生成前缀："First I need to grasp the red cup at"
  学生 next-token 分布: coordinates(0.4), position(0.3), location(0.2), ...

教师模型 (MoE-32B) teacher forcing:
  相同前缀下，教师分布: coordinates(0.7), location(0.2), position(0.08), ...

蒸馏损失:
  KL(教师分布 || 学生分布) = Σ p_t · log(p_t / p_s)
  这个损失推动学生分布靠近教师分布
```

## 4. 工程视角 (Engineering View)

| 工程指标 | MoT-2B (边缘) | MoE-32B (云端) | 备注 |
|----------|---------------|----------------|------|
| 激活参数 | 2B | 32B | MoE 总参数 407B，但每 token 仅激活 32B |
| ViT 参数 | 400M | 400M | 蒸馏自更大内部模型，支持任意分辨率 |
| 推理延迟 | 待实测 (目标 <100ms @ Jetson Orin) | 待实测 | 论文未给出边缘设备实测数据 |
| 显存占用 | ~4GB (FP16) | ~64GB (FP16) | 估算值，未考虑 KV cache |
| 训练 token | 600B+ (预训练) + 25M (中训练) | 同左 | 预训练 389B 通用 + 236B 具身/感知 |
| 后训练数据 | 100k SFT + 50k/轮 RL + 300k RFT | 同左 | RL 动态筛选，RFT 从 1M 候选筛到 300k |
| 蒸馏方式 | On-Policy (学生 rollout + 教师强制) | 教师 | 关键：监督学生在自己状态下的分布 |

**部署约束**:
- 边缘设备需确认 400M ViT + MoT-2B 的实际吞吐（论文声称"实时响应"但未给数字）
- 蒸馏后的小模型保留了多少大模型的推理深度？需要实测验证
- VLA 微调阶段需要多少机器人数据？论文 §6 仅概述，未给数据量级

**Trade-off 分析**:
- MoT vs 标准 Transformer: 参数量翻倍 (4B vs 2B)，但计算开销几乎不变（因为视觉/语言 token 分开计算，无交叉）
- Latent Token: 增加 1 个 token 的计算，但换来全局语义对齐，适合小模型
- 迭代后训练：计算成本高（多轮 RL+RFT），但换来推理质量稳定提升

## 5. 数据与评测 (Data & Eval)

### 5.1 数据组成

| 数据类型 | 样本量 | 来源 | 用途 |
|----------|--------|------|------|
| Omni-Detection | 62M | OpenImages, Objects365, RefCOCO, SA-1B + 自动标注 | 2D/3D 检测、物体识别 |
| Depth Estimation | 36M | 室内/室外 3D 数据集 + 自动驾驶语料 | 绝对/相对深度感知 |
| Segmentation | 5M | SA-1B (过滤后) | 细粒度视觉感知、边缘感知 |
| Pointing & Counting | 11M | Pixmo-Points + 高密度场景筛选 | 物体指向与计数 |
| Embodied (Grounding/Affordance/Trajectory) | 未公开 | Molmo, RoboPoint, ShareRobot + 自采 | 具身操作基础 |
| Spatial (Correspondence/Geometry/Configuration) | 未公开 | ScanNet, ScanNet++, ARKitScenes + 自采 | 3D 空间理解 |
| General Understanding | 389B tokens | 内部通用 VLM 数据 | 基础推理与理解 |

**预训练混合**: 600B+ tokens (389B 通用 + 236B 具身/感知，其中空间/机器人占 43%)
**中训练混合**: 25M 样本 (通用：具身：空间 = 12:5:3)

### 5.2 评测基准与结果

**22 个基准覆盖三大类**:

| 类别 | 代表基准 | 评测能力 |
|------|----------|----------|
| 视觉感知 | Visual Genome, RefCOCO, SA-1B | 物体识别、指代理解、分割 |
| 空间推理 | ScanQA, ScanRefer, ARKitScenes | 3D 定位、空间关系、深度估计 |
| 具身理解 | RoboVQA, RoboPoint, Molmo | 机器人场景理解、操作规划 |

**关键结果** (论文 Table 1/2 摘要):

| 模型 | 激活参数 | 22 基准平均 | 备注 |
|------|----------|-------------|------|
| HY-Embodied-0.5 MoT-2B | 2B | 58.0% | 16/22 基准 SOTA |
| Qwen3-VL-4B | 4B | 47.8% | 通用 VLM |
| RoboBrain2.5-4B | 4B | 49.4% | 具身专用 VLM |
| HY-Embodied-0.5 MoE-32B | 32B | 67.0% | 超越 Gemini 3.0 Pro (63.6%) |
| Gemini 3.0 Pro | 未公开 | 63.6% | 前沿闭源模型 |

**分项表现亮点**:
- **视觉感知**: 在 RefCOCO 指代理解任务上达到 78.2%，超越 Qwen3-VL-4B (71.5%)
- **空间推理**: 在 ScanRefer 3D 定位任务上达到 45.8%，较 RoboBrain2.5-4B 提升 8.3%
- **具身理解**: 在 RoboPoint 指点任务上达到 82.1%，接近人类水平 (85.3%)

**VLA 下游实验** (§6):
- 用 HY-Embodied-0.5 作为 VLM 基础训练 VLA 模型
- 在真实机器人物理评估中取得"compelling results"
- 但论文未给出具体任务成功率、对比基线等细节（TODO: 待补充）
- 推测使用 ACT 或 Diffusion Policy 作为动作头，需等待代码开源确认

## 6. 能力与失败模式 (Capabilities & Failure Modes)

### 6.1 能做什么

| 能力 | 场景 | 证据 |
|------|------|------|
| 细粒度物体定位 | 指点、计数、3D 框预测 | Pointing/Counting 数据 11M，基准测试 SOTA |
| 深度感知 | 绝对/相对深度估计 | Depth Estimation 数据 36M |
| 空间关系推理 | 物体相对位置、方向、距离排序 | Configuration/Measurement 数据 |
| 具身规划 | 多步动作序列预测 | Planning 数据 (视频分段 + VLM 标注) |
| 长链推理 | 复杂多步问题 | 迭代 RL+RFT 后训练，100k CoT 冷启动 |

### 6.2 不能做什么 / 局限

| 局限 | 原因 | 影响 |
|------|------|------|
| VLA 控制细节不透明 | §6 仅概述，无具体实验设置 | 难以复现机器人控制结果 |
| 边缘延迟未实测 | 论文声称"实时"但无数据 | 部署前需自行 benchmark |
| 训练数据未完全开源 | 仅开源代码/模型，数据清单不完整 | 难以评估数据偏差 |
| 单一机器人平台验证 | 未说明具体机器人型号/数量 | 泛化性待验证 |

### 6.3 隐含假设 (Hidden Assumptions)

- **假设 1 (视觉监督)**: 视觉 next-code 预测任务能有效监督视觉分支——但离散码来自内部更大 ViT，外部无法复现；该假设依赖教师 ViT 的表征质量
- **假设 2 (RL 数据)**: RL 动态数据筛选能持续提供"可学习前沿"样本——但长期运行是否会导致模式坍塌？需要持续注入新任务类型
- **假设 3 (蒸馏)**: On-policy 蒸馏能保留教师推理风格——但学生容量有限时，是否只能学到表面模式？需验证推理深度是否真正迁移
- **假设 4 (MoT 扩展)**: MoT 参数复制不会导致过拟合——但 2B→4B 的参数量增加是否在所有任务上都有益？可能在简单任务上冗余
- **假设 5 (Sim2Real)**: 论文隐含假设仿真/合成数据训练的感知能力可直接迁移到真实机器人——但未报告 Sim2Real gap 量化实验
- **假设 6 (任务泛化)**: 22 个基准的评测覆盖被认为足以代表"真实世界具身能力"——但实际机器人操作涉及连续控制、接触力学等未评测维度
- **假设 7 (推理链质量)**: RFT 筛选的"高质量推理轨迹"由更强教师模型评判——但该评判标准本身可能存在偏差，且未公开评判 prompt

## 7. 与相关工作对比 (Comparison)

| 工作 | 架构 | 训练策略 | 具身数据 | 边缘优化 | 开源 |
|------|------|----------|----------|----------|------|
| HY-Embodied-0.5 | MoT + Latent Token | 迭代 RL+RFT + On-Policy 蒸馏 | 100M+ | 是 (400M ViT) | 代码/模型 |
| RoboBrain2.5 | 标准 VLM | 标准 SFT | 未公开 | 否 | 部分 |
| Qwen3-VL | 标准 VLM | 通用预训练 | 少 | 部分尺寸 | 是 |
| π0 (VLA) | VLA 专用 | 行为克隆 + 扩散策略 | 机器人轨迹 | 否 | 是 |
| RT-2 | VLA | 行为克隆 | 机器人 + 网络数据 | 否 | 是 |

**关键差异**:
- HY-Embodied-0.5 是**VLM 基础模型**，不是端到端 VLA——它提供感知和推理基础，VLA 动作头需额外微调
- 迭代后训练 (RL+RFT 循环) 是独特贡献，其他工作多为一次性 SFT 或纯 RL
- On-policy 蒸馏方案对小模型部署有参考价值

**面试 Tip**: 被问到"小模型如何做具身任务"时，可以答："HY-Embodied-0.5 展示了三个关键点：(1) 用 MoT 分离视觉/语言计算路径避免能力退化，(2) 用迭代 RL+RFT 将偶发成功转化为稳定推理能力，(3) 用 on-policy 蒸馏让小型模型继承大模型的推理风格而非仅模仿输出。"

## 8. 精读建议 (Reading Guide)

### 值得精读原文的人

1. **做多模态具身 Agent 的研究者**: 尤其是关注小模型边缘部署的团队，§2 架构和 §4 后训练有直接参考价值
2. **要评估迁移到新机器人平台可行性的工程师**: §3 数据构成和 §5 评测结果帮助判断模型是否适合你的任务域
3. **做模型蒸馏/压缩的研究者**: §4.4 On-Policy Distillation 提供了大→小具身能力迁移的新思路

### 建议章节路径

```
先读 §1 Introduction → 再看 §2 Model Architecture → §4 Post-training → 可跳 §3.1 (数据细节过多)
```

**原因**: §1 给出问题定义和贡献概述；§2 是架构核心；§4 是训练策略创新；§3.1 数据细节对大多数读者过于冗长，需要时再查。

### 不值得精读的理由

- 如果你**不做机器人学习**：这篇的具身数据构建和 RL 奖励设计可能过于垂直
- 如果你**已熟悉类似方法**（如 MoT、GRPO、蒸馏）：可直接看 §5 结果和 §6 VLA 下游
- 如果你**只想要端到端 VLA 方案**：这篇是 VLM 基础模型，动作头需额外设计

---

[← Back to Theory](./README.md)

**关键引用**:
- 论文: https://arxiv.org/abs/2604.07430
- 代码: https://github.com/Tencent-Hunyuan/HY-Embodied
- MoT 架构原文: Liang et al., 2024
- GRPO: Shao et al., 2024
- On-Policy Distillation: Agarwal et al., 2024 / Thinking Machines Lab, 2025
