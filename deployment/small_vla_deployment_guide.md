# 小模型 VLA 部署实战经验手册

> **来源**：整理自 [VLA Handbook](https://github.com/sou350121/VLA-Handbook) 三份社区实战笔记（小红书 300+ 条、英文社区 165 条、GitHub Issues 200+ 条）+ 理论文档（SmolVLA / SimVLA / Shallow-Pi / QVLA / Spirit-v1.5 等）。所有结论附社区帖号或论文出处。
>
> **适用对象**：准备用小模型 VLA（<3B 参数）做真机部署的工程师。
>
> **最后更新**：2026-04-15

---

## 目录

1. [小模型 VLA 全景图](#1-小模型-vla-全景图)
2. [已验证的训练配方](#2-已验证的训练配方)
3. [训练超参数深度解读](#3-训练超参数深度解读)
4. [数据量门槛与模型选择策略](#4-数据量门槛与模型选择策略)
5. [GPU 硬体矩阵与生存指南](#5-gpu-硬体矩阵与生存指南)
6. [训练收敛失败排查手册](#6-训练收敛失败排查手册)
7. [推理加速：从 5Hz 到 60Hz](#7-推理加速从-5hz-到-60hz)
8. [量化与模型压缩路线](#8-量化与模型压缩路线)
9. [Action Chunking 陷阱与抖动处理](#9-action-chunking-陷阱与抖动处理)
10. [LoRA vs 全量微调决策树](#10-lora-vs-全量微调决策树)
11. [Sim2Real 的真实根因](#11-sim2real-的真实根因)
12. [真机部署工程陷阱](#12-真机部署工程陷阱)
13. [论文数字 vs 社区实测](#13-论文数字-vs-社区实测)
14. [推荐技术栈与最短复现路径](#14-推荐技术栈与最短复现路径)

---

## 1. 小模型 VLA 全景图

### 1.1 为什么必须用小模型

机器人本地算力的现实：

| 硬体 | 算力 | 能跑的模型 |
|------|------|-----------|
| Jetson Orin NX | ~100 TOPS (INT8) | SmolVLA (450M) |
| RK3588 | ~6 TOPS | 量化后的极小模型 |
| RTX 4090 | ~330 TOPS (FP16) | π0 (3B) 勉强 |
| H100 | ~990 TOPS (FP16) | OpenVLA (7B) 舒适 |

大模型 VLA（7B+）推理延迟 200-500ms，但精细操作需要 20ms 以内。**小模型不是「省钱版」，而是唯一能本地即时控制的路径。**

### 1.2 当前主力小模型对比（2026 年性能快照）

| 模型 | 参数量 | LIBERO Avg | 推理速度 | 边缘部署 | 训练显存 |
|------|--------|-----------|---------|---------|---------|
| **SimVLA** | 0.5B | **98.6%** | 快 | ✅ | **9.3 GB** |
| **Evo-1** | 770M | 94.8% | 快 | ✅ | — |
| **SmolVLA** | 210M-450M | ~80-85% | 45-60 Hz | ✅ | ~24 GB |
| ACT | ~80M | ~90%+ (单任务) | 快 | ✅ | <16 GB |
| π0 | 3B | ~88% | ~15 Hz | ⚠️ 量化后 | ~50 GB |
| OpenVLA | 7B | ~80% | ~5 Hz | ❌ | ~62 GB |
| RT-2 | 55B | ~85% | ~1 Hz | ❌ | — |

**关键事实**：SmolVLA 210M 在 SimplerEnv 上超越 55B 的 RT-2-X（48.2% vs 42.3%），Evo-1 770M 在 LIBERO 上超越所有大模型。SimVLA 0.5B 用 9.3GB 显存达到 98.6%。**机器人任务不需要那么大的模型。**

---

## 2. 已验证的训练配方

### 2.1 SmolVLA 微调配方（社区验证）

来源：小红书帖 55 评论（烧仙草），社区唯一完整公开配方。

```yaml
# SmolVLA Fine-tuning Recipe (Community Verified)
data:
  episodes: 50              # 最小可用量（冻结 backbone 前提下）
  collection_freq: 20 Hz

training:
  batch_size: 64
  learning_rate: 4e-5
  total_steps: 30000        # 30K steps
  chunk_size: 30
  n_action_step: 24
  strategy: freeze_backbone  # 仅微调 expert
  precision: bfloat16

hardware:
  minimum: RTX 3090 (24GB)  # 冻结 backbone
  recommended: RTX 4090

result:
  success_rate: 50-80%      # 训练分布内
```

### 2.2 ACT 训练配方（3 人独立验证）

来源：小红书帖 45/46/55 三人独立验证。

```yaml
# ACT Training Recipe (3x Independent Verification)
data:
  episodes: 50              # 单任务最小可用量
  convergence: ~31K steps

training:
  # ACT 使用 CVAE + Transformer，无需 Flow Matching 配置
  action_chunking: true

hardware:
  minimum: RTX 3090 (24GB)
  cost: "本地 3090 训练 100K steps = 3 小时 / 云端 4090D = 20 元 (AutoDL)"

result:
  success_rate: 90%+        # 训练分布内，单任务
```

### 2.3 SimVLA 配方（论文消融验证）

来源：SimVLA 论文 (arXiv:2602.18224)，0.5B 模型 LIBERO 98.6%。

```yaml
# SimVLA Recipe (Paper Ablation Verified)
architecture:
  vlm_backbone: SmolVLM-0.5B
  action_head: transformer_encoder  # 轻量 action head
  conditioning: token_concatenation  # 比 cross-attention 更强！
  flow_matching: true

training:
  learning_rate: 2e-4        # 消融确认最优
  vlm_lr_multiplier: 0.1     # VLM 用 0.1× 学习率（关键！）
  action_horizon: 10         # chunk 不是越长越好
  data_shuffling: true       # 关掉直接崩到 9.9%（致命）
  action_normalization: true # 关掉直接崩到 12.3%（致命）

hardware:
  training_vram: 9.3 GB      # 单卡消费级 GPU 可训

result:
  libero_avg: 98.6%
```

### 2.4 GR00T 官方配方

来源：GitHub Issues 社区挖掘，官方实验配置。

```yaml
# GR00T Official Recipe
training:
  batch_size: 120
  learning_rate: 3e-5
  total_steps: 30000
  gpus: 8x H100

minimum_viable:
  gpus: 2x RTX 4090
  flags: "--no-tune-visual --batch-size 1"
  method: LoRA
```

### 2.5 π0 系列配方

来源：π0 代码解析 + 小红书帖 20（南柯一手经验）。

```yaml
# π0 Fine-tuning Recipe
architecture:
  vlm_backbone: paligemma-3b-pt-224
  action_expert_layers: 4
  action_expert_heads: 8
  action_expert_dim: 512
  action_horizon: 50
  num_inference_steps: 10
  sigma_min: 0.001

training:
  optimizer: AdamW
  lr_schedule: cosine_with_warmup
  warmup_steps: 1000-5000
  max_grad_norm: 1.0
  precision: bfloat16
  gradient_accumulation: 4

data:
  minimum: 100 episodes      # <200 条大概率失败
  recommended: 5000 episodes  # OOD 成功率可达 97%

hardware:
  lora: "单卡 20GB (bs=1) → A100 40GB 推荐"
  full_finetune: "单卡 70GB (A100/A800) → 8×A800 推荐"
```

---

## 3. 训练超参数深度解读

### 3.1 SimVLA 消融：哪些「静默变量」真正决定生死

这组消融数据是整份文件最有价值的部分，来自 SimVLA 论文：

| 训练细节 | 开启 | 关闭 | 结论 |
|---------|------|------|------|
| **Data shuffling** | 98.6% | **9.9%** | 不打散轨迹 = 直接崩塌 |
| **Action normalization** | 98.6% | **12.3%** | 不归一化 = 直接崩塌 |
| **LR 2e-4 vs 5e-4** | 98.6% | **72.7%** | 学习率差 2.5 倍，性能差 26% |
| **VLM LR ×0.1 vs ×1.0** | 98.6% | **44.2%** | VLM 必须轻调，Action Head 全速 |
| **Horizon 10 vs 30** | 98.6% | **87.3%** | chunk 越长不一定越好 |
| **Token concat vs cross-attn** | 98.6% | **91.5%** | 最简单的反而最好 |

**核心教训**：在比较架构之前，先确认 shuffling、normalization、LR 是否对齐。很多「架构优势」其实只是 recipe 没对齐。

### 3.2 各参数详解

**batch_size（32-64）**：小 batch（4/8）在 Diffusion/Flow 类模型上效果显著差，32+ 才稳定。SmolVLA 社区和大模型社区结论一致。显存不够用梯度累积模拟（bs=8 + accumulate=4 ≈ 等效 32）。

**learning_rate（1e-4 ~ 4e-5）**：SmolVLA 用 4e-5（冻结 backbone），SimVLA 用 2e-4（VLM ×0.1 倍率），GR00T 用 3e-5。规律：冻结越多 lr 可以越高，端到端微调 lr 必须更保守。

**VLM LR multiplier（0.1×）**：SimVLA 消融确认 VLM 和 Action Head 学习率必须分开设。VLM 用 0.1× 轻调保留预训练知识，Action Head 用全速适配新任务。直接统一学习率 → 性能跌到 44.2%。

**action_horizon（10-50）**：不是越长越好。SimVLA 用 10 最优（98.6%），调到 30 掉到 87.3%。SmolVLA 用 30，π0 用 50。**以 20Hz 控制频率算：horizon 10 = 0.5 秒规划，50 = 2.5 秒。** 短任务用短 horizon，长任务再考虑加长。

**warmup_steps（1000-5000）**：小模型收敛快，1000 步通常足够。数据集很小（几千步一个 epoch）可以缩到 500。

**precision（bfloat16）**：没理由不用。显存减半、速度翻倍、bf16 的动态范围比 fp16 大，VLA 数值范围较大的任务更稳定。A100/H100/4090 都支持。

---

## 4. 数据量门槛与模型选择策略

### 4.1 模型选择决策树

```
你有多少 episodes？
│
├─ <50 条 ────→ ACT（唯一选项，其他模型全部失败）
│
├─ 50-200 条 ──→ ACT 最稳（90%+）
│                SmolVLA 可试（冻结 backbone，50-80%）
│                π0 / Diffusion Policy 大概率失败
│
├─ 200-1000 条 → SmolVLA / SimVLA（解锁端到端微调）
│                π0 开始可用
│                ACT 仍然稳定
│
└─ >1000 条 ──→ π0 最强（5000 条时 OOD 97%）
                 SimVLA 0.5B 可达 98.6%（matched setup）
                 大模型路线解锁
```

### 4.2 消融实验数据（帖 55，同条件对比）

| 模型 | <200 条数据 | >5000 条数据 |
|------|-----------|-------------|
| ACT | **90%+ 成功** | 稳定但不再提升 |
| SmolVLA | 失败（前进→回撤振荡） | 可用 |
| π0 | 失败（同上） | **97% OOD** |
| Diffusion Policy | 失败（同上） | 可用 |

**社区结论：数据少用 ACT，数据多用 π0，中间地带用 SmolVLA/SimVLA。**

### 4.3 数据采集实用建议

- **采集频率 20Hz** 是社区验证的甜蜜点
- ACT 50 episodes 在 Franka 上验证可行（3 人独立确认）
- **预测绝对量优于 delta 值**：预测相对 offset 很难 recover（帖 20 南柯 + ACT 论文结论）
- State 归一化必须提前计算全数据集 mean/var（不能边训边算）

---

## 5. GPU 硬体矩阵与生存指南

### 5.1 训练资源矩阵

| 任务 | 最低配置 | 推荐配置 | 成本参考 |
|------|---------|---------|---------|
| ACT 训练 | 3090 (24GB) | 4090 | 3h/100K steps 本地 |
| SmolVLA 微调（冻结） | 3090 (24GB) | 4090 | — |
| SimVLA 训练 | 单卡 ~10GB | 4090 | 显存最友好 |
| π0 LoRA | 20GB (bs=1) | A100 40GB | — |
| π0 全量 | A100 70GB | 8×A800 | — |
| GR00T LoRA | 2×4090 | 4×4090D | — |
| GR00T 全量 | H100 | 8×H100 | — |

### 5.2 显存对比

| 模型 | LoRA 显存 | 全量显存 |
|------|----------|---------|
| ACT (~80M) | N/A | <16 GB |
| SmolVLA (500M) | <16 GB | ~24 GB |
| SimVLA (0.5B) | — | **~9.3 GB** |
| π0 (3B) | ~20 GB | ~70 GB |
| OpenVLA (7B) | ~30 GB | ~100 GB+ |

### 5.3 3090 生存指南（帖 39，127 赞）

**能跑**：ACT、SmolVLA 冻结 backbone、SimVLA、LoRA 微调小模型

**不能跑**：全参数 π0、Motus、大规模 RL

**三件套**：gradient checkpointing + mixed precision + gradient accumulation

**云平台选型**（帖 52）：
- AutoDL：便宜但不支持 docker
- 智星云：按小时租，性价比高
- GPULab：有预装 IsaacSim 镜像但贵

### 5.4 边缘设备部署

| 设备 | 能跑 | 延迟 | 注意事项 |
|------|------|------|---------|
| Jetson AGX Orin 64GB | ✅ | ~1.1s (JAX) | 设 `XLA_PYTHON_CLIENT_PREALLOCATE=false` |
| Jetson Orin NX/Nano | ❌ OOM | — | 不可用 |
| Jetson AGX Thor | ✅ | ~300ms | 理论 4× 快于 Orin |
| RTX 5090 | ⚠️ | — | 需手动编译 FlashAttention，SM120 相容问题 |

### 5.5 RTX 50 系列注意事项

- torch 2.7.1 不支持 SM120 架构
- 解决：`torch>=2.2.1,<=2.8.0` + cu128 索引安装
- FlashAttention 需手动编译
- IsaacLab 在 Ubuntu + 570.x 驱动下有像素化渲染问题（Windows 正常）

---

## 6. 训练收敛失败排查手册

### 6.1 三大静默杀手

**杀手 1：Action 标准差为零 → Loss 爆炸到百万级**

```
症状：Loss 突然跳到 1e6+
根因：数据采集时某些维度（如 roll/pitch）始终不变 → 归一化除以零
修复：把零 std 替换为小 epsilon
      stats["std"][stats["std"] == 0] = 2e-5  # 或 2e-4
```

**杀手 2：Transformers 版本不兼容 → 权重静默加载失败**

```
症状：Loss 从 ~0.5 跳到 ~4.5，真机成功率从 80% 跌到 10%
根因：transformers ≥4.52.0 重命名了 PaliGemma 的 key
      language_model.model → model.language_model
      导致预训练权重全丢但不报错！
修复：锁定 transformers 版本，或手动修复 key mapping
```

**杀手 3：Gripper 归一化映射错误**

```
症状：机械臂行为异常但不报错
根因：LIBERO sim 用 [-1,1]，训练归一化到 [0,1]
修复：action_chunk[:,-1] = -2*action_chunk[:,-1] + 1
      对 Agilex Cobot Magic：直接注释掉 gripper normalize/unnormalize
```

### 6.2 SimVLA 消融确认的致命错误

| 忘了做什么 | 结果 |
|-----------|------|
| Data shuffling | 98.6% → **9.9%** |
| Action normalization | 98.6% → **12.3%** |
| VLM LR multiplier | 98.6% → **44.2%** |

### 6.3 多 GPU 训练问题

- **JAX 8+ GPU JIT 无限挂起**：CUDA 12.2 不兼容，升级到 12.8 解决
- **orbax-checkpoint 7% 进度崩溃**：禁用异步保存或降级 orbax 到 0.11.1
- **视频解码瓶颈**：占数据加载 92% 时间（H100 实测），torchcodec 单样本 3× 快但多进程有 forked decoder state 问题
- **OMP_NUM_THREADS**：不设成 4 会导致 CPU 过载

### 6.4 跨仓库收敛信号（系统性挑战）

| 问题 | 出现仓库 | 核心矛盾 |
|------|---------|---------|
| Action 表示混乱（delta vs absolute） | openpi, GR00T, lerobot | 归一化不一致 |
| Jetson OOM | openpi, GR00T | 统一内存 + 预分配冲突 |
| 官方 checkpoint 不可复现 | openpi, GR00T | 内部管线 ≠ 开源版本 |
| RTX 50 系列不兼容 | lerobot, GR00T, IsaacLab | SM120 支持缺失 |

---

## 7. 推理加速：从 5Hz 到 60Hz

### 7.1 核心结论

> **工程优化（CUDA Graph / 算子融合）比模型压缩更立竿见影。**
> — 帖 65，AI椰青，258 赞

### 7.2 加速技术栈（帖 61，85 赞 130 收藏）

| 技术 | 效果 | 实现难度 |
|------|------|---------|
| CUDA Graph 预记录 kernel | 消除 CPU-GPU 同步开销 | 中 |
| RMS norm + Linear 合并 | 减少 7-8ms | 低 |
| QKV 投影融合 | 减少 kernel launch | 低 |
| Full Streaming Inference（VLM + Action Expert 并行） | GPU 利用率翻倍 | 高 |
| **组合效果** | **单卡 30Hz 推理 + 480Hz 控制** | — |

真机验证：机器人抓住下落的笔，100% 成功率，反应速度接近人类。

### 7.3 异步推理架构

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ 感知          │ ──→ │ 推理          │ ──→ │ 执行          │
│ (相机输入)    │     │ (模型前向)    │     │ (动作下发)    │
└──────────────┘     └──────────────┘     └──────────────┘
                          ↑                      │
                          │    上一 chunk 继续执行 │
                          └──────────────────────┘
推理等待期间继续执行上一个 action chunk，避免动作中断。
```

### 7.4 各模型推理速度实测

| 模型 | 参数量 | 原始速度 | 优化后 |
|------|--------|---------|--------|
| SmolVLA | 210M-450M | 20-30 Hz | **45-60 Hz** |
| SimVLA | 0.5B | — | 快 |
| ACT | ~80M | 快 | — |
| π0 | 3B | ~15 Hz | — |
| OpenVLA | 7B | ~5 Hz | ~10-12 Hz (INT4) |
| RT-2 | 55B | ~1 Hz | — |

### 7.5 KV-Cache 优化

VLM 的语言指令在一个 episode 内不变，用 KV-Cache 只计算一次，后续帧只更新图像。SmolVLA/SimVLA 的小 VLM 天然 KV-Cache 占用少。

---

## 8. 量化与模型压缩路线

### 8.1 三条压缩路径

**路径 1：知识蒸馏（Shallow-Pi）**

把 π 系列 VLM backbone + Action Head 同时从 18 层压到 6 层：
- 成功率下降 <1%
- 推理加速 >2×
- 关键：蒸馏 Flow Matching 的速度场（而非最终动作），用注意力对齐保留跨模态语义
- 蒸馏损失 = 任务监督 + 教师速度场对齐 + action→VL cross-attention KL 散度

**路径 2：结构化剪枝（Layer Skipping）**

SmolVLA 方案：24 层只保留前 12 层，计算量减半，原理是浅层已包含足够的视觉语言对齐。

| 剪枝率 | 参数保留 | 性能保留 |
|-------|---------|---------|
| 25% | 75% | ~98% |
| 50% | 50% | ~92% |
| 75% | 25% | ~80% |

**路径 3：量化（QVLA 通道级）**

| 方案 | 显存 | 精度损失 | 推理速度 |
|------|------|---------|---------|
| FP16 | 14 GB (7B) | 基准 | 5 Hz |
| INT8 | 7 GB | ~1% | 8 Hz |
| INT4 (QLoRA) | 4 GB | ~3% | 10 Hz |
| INT4 (AWQ) | 4 GB | ~2% | 12 Hz |

### 8.2 QVLA 的核心发现：不能无脑统一量化

VLA 各模块对量化的敏感度差异极大：

| 模块 | 量化敏感度 | 建议 |
|------|-----------|------|
| Vision Encoder | **低**（鲁棒） | 可降到 INT4 |
| LLM Backbone | 中 | INT8 安全 |
| Projector | **高** | 保留 FP16 |
| Action Head | **最高** | **必须保留 FP16** |

**通道级混精度量化**（每个通道独立分配 0/2/4/8/16 bit）比全局统一 bit 好得多。校准数据必须覆盖任务关键动作相位，否则长时程任务误差累积导致机器人漂移。

### 8.3 实测可行性

- PaliGemma VLM 的 MLP 做 FP8/NVFP4 PTQ：LIBERO 2000 episode 无显著精度损失
- 但目前只是权重量化 + 反量化回 BF16，**无实际加速**
- TensorRT 导出是社区强烈需求但尚无官方支持

---

## 9. Action Chunking 陷阱与抖动处理

### 9.1 GR00T 的重大发现：别只用第一步

官方示例只用 action chunk 的第一步（`action_chunk[0]`），**丢了 10% 性能**：

| 使用方式 | LIBERO Spatial | LIBERO Long |
|---------|---------------|-------------|
| 只用第一步 | 90% | 80% |
| **用全部 16 步** | **96%** | **90%** |

### 9.2 Action Horizon 不是越长越好

SimVLA 消融：

| Horizon | LIBERO Avg |
|---------|-----------|
| H=10 | **98.6%** |
| H=20 | 92.4% |
| H=30 | 87.3% |

### 9.3 抖动处理方案比较（帖 190，176 赞，ICRA 2026）

| 方案 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| ACT temporal ensemble | 相邻 chunk 加权平均 | 内建，丝滑 | 掩盖模型问题 |
| SmolVLA 加权融合 | chunk 边界处加权过渡 | 简单 | 依赖后处理 |
| RTC 引导生成 | 推理时约束连续性 | 理论干净 | 实现复杂 |
| 在线插值（帖 35 开源） | 约束 jerk/acc/vel 三阶导 | 支持任意频率 | — |
| **ABPolicy（ICRA 2026）** | Flow Matching 建模 B-spline 控制点 | **源头平滑** | 新方法 |

**Chunk 内部抖** → 调大去噪步数。**Chunk 间抖** → 异步推理 + RTC 过渡。

---

## 10. LoRA vs 全量微调决策树

### 10.1 经验法则（帖 34b，算法改进猫博士，86 赞）

```
你有多少数据？
│
├─ <200 条 ──→ LoRA（防过拟合）
│               rank 8-16 (小模型) / 32-64 (大模型)
│               lora_alpha = 2× rank
│               target: q_proj, v_proj（可选加 FFN）
│
├─ 200-500 条 → 看任务与预训练的距离
│               近（同类机器人/任务）→ LoRA
│               远（新动作空间）→ 全量
│
└─ >500 条 ──→ 考虑全量微调
```

### 10.2 LoRA 配置模板

```python
LoraConfig(
    r=16,                    # 小模型 8-16，大模型 32-64
    lora_alpha=32,           # 通常 2× rank
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)
# OpenVLA 7B + LoRA(r=16): 可训练参数只有 ~17M (0.24%)
# 推理时可合并回主模型，零额外开销
```

### 10.3 危险组合：LoRA + RL（帖 34c，马小疼，137 赞）

- LoRA 的低秩限制了 policy 表达能力
- RL 探索容易跳出 LoRA 能表达的范围 → 训练崩溃
- GRPO 在 MOE 模型上更危险（Qwen3 论文确认）
- **建议**：先 LoRA-SFT 稳定后，RL 阶段考虑全量或更高 rank

### 10.4 灾难性遗忘（帖 31/37）

直接微调 VLM 学动作 → OOD 泛化降 ~10%，attention 集中在背景而非目标物体。

解法：
- 冻结视觉主干（SmolVLA 预设做法）
- Visual Representation Alignment：用原始 VLM 做「视觉老师」约束视觉模块不偏移
- 诊断工具：VLAExplain（帖 37，支持 Pi0.5 注意力可视化）

---

## 11. Sim2Real 的真实根因

### 11.1 社区高度一致的结论

> **大多数 Sim2Real 失败是物理参数不准，不是算法问题。**

| 失败根因 | 频率 | 来源 |
|---------|------|------|
| Friction model 未校准 | **最高频** | 帖 41 |
| Domain Randomization 被简化为「加噪声」 | 常见 | 帖 41 |
| 执行器延迟未建模 | 常见 | 帖 42 |
| URDF 参数不准 / 版本不一致 | 常见 | 帖 54 |
| 观测向量维度漏对齐 | 隐蔽 | 帖 33 |

### 11.2 最惨案例

帖 33：Isaac Gym → MuJoCo 迁移，四步内摔倒。排查三周，最终发现是观测向量中漏了初始关节角的调整——**只改了两三个单词**。

### 11.3 推荐方法

**Real2Sim2Real 闭环**：先用真机数据校准仿真 → 再从校准后的仿真训练。不追「仿真中的性能」，追「仿真的真实性」。

---

## 12. 真机部署工程陷阱

### 12.1 SmolVLA 特有问题

- `tokenizer.json` 必须从 HuggingFace 单独下载（仓库不包含，帖 56）
- 跨维度（如 6D→14D）必须冻结视觉主干只训动作头
- LIBERO 复现：单步推理比多步强（66.8% → 82%）
- MuJoCo 必须用 3.3.2（颜色渲染差异影响视觉输入）

### 12.2 Spirit-v1.5 隐藏坑

- `_embed_suffix()` 里 `state[:, :, [2, 9]] = 0` 硬把 state 第 2/9 维置零
- UR5 的 `assert state_tmp[6] > 1` 期望夹爪值 0~255，喂 0~0.1 直接报错
- 归一化 stats 不匹配会 assert 失败（MIN_MAX buffer 出现 inf）

### 12.3 通用真机问题

- **电机冷启动**（帖 12）：温度低 → 摩擦力大 → 预测偏差。解法：**开机空跑 10 分钟热机**
- **重复精度 ≠ 绝对精度**（帖 53）：参数表 ±0.02mm 只是重复精度，TCP 标定/基座标定/热漂移都加误差
- **松灵 Piper 机械臂**（帖 54）：逆解大量可达角度求不出（万向锁附近）、URDF 末端座标系不同、无自锁
- **固件更新**（帖 26）：更新后 RL 模型换了，运动学指标全部得重调

### 12.4 推荐低成本硬体组合

- SO101 + π0：可跑通，最低成本
- XLeRobot (4K RMB) / SO101：入门首选
- 松灵七轴 (15K RMB)：中端

---

## 13. 论文数字 vs 社区实测

| 模型 | 论文声称 | 社区实测 | 落差 | 来源 |
|------|---------|---------|------|------|
| OpenVLA | 85.7% | **62.6%** | -23.1% | 帖 55 |
| π0 (<200 条) | 通用泛化 | 全部失败 | 严重 | 帖 55 |
| π0 零样本跨机器人 | — | **不可能** | 社区共识 | 多帖 |
| GR00T 官方 ckpt | 可复现 | **0% 成功率** | 严重 | 2 团队确认 |
| SmolVLA LIBERO | — | 66.8%→82%（改单步） | — | GitHub Issues |
| SimVLA | 98.6% | 待社区复现 | — | 论文 |

**Pi0-fast LIBERO checkpoint 兼容性**：新版 repo 不工作（成功率 0%），最后可用 commit：`e4580662`（2025-09-07）。

---

## 14. 推荐技术栈与最短复现路径

### 14.1 最短路径：SmolVLA + LeRobot

```bash
# 1. 安装 LeRobot（v0.5.0+，已原生支持 SmolVLA）
pip install lerobot

# 2. 采集 50 episodes（20Hz）
# 用 LeRobot 的标准采集工具

# 3. 训练（单卡 4090，约 2-3 小时）
# 配置：lr=4e-5, bs=64, 30K steps, 冻结 backbone

# 4. 部署推理
# SmolVLA 在 RTX 4070 上可达 45-60Hz
```

### 14.2 性能最优路径：SimVLA

```bash
# 0.5B 模型，9.3GB 显存，LIBERO 98.6%
# 关键 recipe：
#   - lr=2e-4, VLM lr ×0.1
#   - action_horizon=10
#   - token concatenation（不要用 cross-attention）
#   - 必须开 data shuffling + action normalization
```

### 14.3 真机最稳路径：ACT

```bash
# 50 episodes 即可 90%+ 成功率
# 本地 3090 训练 100K steps = 3 小时
# 云端 4090D = 20 元
# 代码最干净，社区最大，踩坑最少
```

### 14.4 推理加速清单（按 ROI 排序）

1. **bfloat16 混合精度** — 零成本，显存减半
2. **KV-Cache** — 语言指令只算一次
3. **CUDA Graph** — 消除 CPU 开销
4. **算子融合**（RMS+Linear, QKV）— 减少 7-8ms
5. **异步推理** — 推理等待期继续执行上一 chunk
6. **通道级量化**（QVLA）— Action Head 保留 FP16

### 14.5 多卡训练 Checklist

- [ ] CUDA 版本 ≥12.8（避免 8+ GPU JIT 挂起）
- [ ] 设定 `OMP_NUM_THREADS=4`
- [ ] 禁用 orbax 异步 checkpoint（或降级到 0.11.1）
- [ ] 用 DeepSpeed ZeRO-2（比 Accelerate 更稳定）
- [ ] HDF5 压缩用 lzf（不要用 gzip，IO 延迟高）
- [ ] Dataloader 调好 num_workers 和 prefetch_factor

---

## 附录：社区高价值贡献者索引

| 贡献者 | 主要贡献 | 帖号 |
|--------|---------|------|
| 南柯 | π0 finetune 完整参数 + 绝对量 vs delta 结论 | 帖 20 |
| 烧仙草 | SmolVLA 50ep 完整训练配方 | 帖 55 评论 |
| 谭谈AI | Motus/WAM 作者，训练参数 | 帖 3 |
| 赵波 (SJTU) | RECAP 完整复现 + 训练成本 | 帖 36 |
| AI椰青 | 单卡 30Hz 推理方案 + Efficient VLA 评估 | 帖 61/65 |
| S!mple | ABPolicy 抖动处理（ICRA 2026） | 帖 190 |
| Claude | LeRobot 框架真机 VLA 复现 | 帖 55/156 |

---

> **本文件整理自 [VLA Handbook](https://github.com/sou350121/VLA-Handbook) 社区实战笔记与理论文档。所有数据均附来源帖号或论文出处，建议交叉验证后使用。**
