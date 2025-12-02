# 大规模模型训练 (Large-Scale Model Training)

> **面试场景**: "请描述你参与过的最大规模的模型训练，遇到了什么挑战？"

大规模训练是 VLA 算法工程师的核心竞争力之一。本文涵盖从集群架构到训练稳定性的完整知识体系。

---

## 📊 规模感知 (Scale Awareness)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    模型规模 vs 训练资源对照表                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   模型规模        显存需求(FP16)    推荐配置           典型训练时间       │
│   ─────────────────────────────────────────────────────────────────     │
│   1B 参数         ~4 GB             1x A100-80G        数小时            │
│   7B 参数         ~14 GB            8x A100-80G        1-3 天            │
│   13B 参数        ~26 GB            8-16x A100-80G     3-7 天            │
│   70B 参数        ~140 GB           32-64x A100-80G    1-2 周            │
│   175B+ 参数      ~350+ GB          128+ H100          数周-数月         │
│                                                                         │
│   💡 VLA 模型通常 3B-8B，但视觉 Token 多导致实际显存占用远超纯 LLM        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1. 训练基础设施 (Infrastructure)

### 1.1 GPU 集群选型

| GPU 型号 | 显存 | FP16 算力 | 互联 | 适用场景 |
| :--- | :--- | :--- | :--- | :--- |
| **H100 SXM** | 80GB HBM3 | 1979 TFLOPS | NVLink 4.0 (900 GB/s) | 超大规模预训练 |
| **A100 SXM** | 80GB HBM2e | 312 TFLOPS | NVLink 3.0 (600 GB/s) | 大规模训练主力 |
| **A100 PCIe** | 40/80GB | 312 TFLOPS | PCIe Gen4 | 性价比之选 |
| **RTX 4090** | 24GB | 330 TFLOPS | PCIe Gen4 | 小规模实验/推理 |
| **L40S** | 48GB | 362 TFLOPS | PCIe Gen4 | 推理优化 |

**面试要点**:
- H100 vs A100: H100 的 Transformer Engine 对注意力机制有 **2-3x 加速**
- SXM vs PCIe: SXM 版本支持 NVLink，多卡通信快 **5-10x**

### 1.2 网络架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    多机多卡网络拓扑                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Node 0 (8x H100)              Node 1 (8x H100)                        │
│   ┌──────────────────┐          ┌──────────────────┐                    │
│   │  GPU0 ─ GPU1     │          │  GPU0 ─ GPU1     │                    │
│   │   │      │       │          │   │      │       │                    │
│   │  GPU2 ─ GPU3     │   IB     │  GPU2 ─ GPU3     │                    │
│   │   │      │       │◄────────►│   │      │       │                    │
│   │  GPU4 ─ GPU5     │ 400Gb/s  │  GPU4 ─ GPU5     │                    │
│   │   │      │       │          │   │      │       │                    │
│   │  GPU6 ─ GPU7     │          │  GPU6 ─ GPU7     │                    │
│   └──────────────────┘          └──────────────────┘                    │
│        NVLink                        NVLink                             │
│       (节点内)                      (节点内)                             │
│                                                                         │
│   💡 节点内 NVLink >> 节点间 InfiniBand >> 普通以太网                    │
└─────────────────────────────────────────────────────────────────────────┘
```

| 互联技术 | 带宽 | 延迟 | 适用场景 |
| :--- | :--- | :--- | :--- |
| **NVLink** | 600-900 GB/s | ~1 μs | 节点内 GPU 通信 |
| **InfiniBand** | 200-400 Gb/s | ~1-2 μs | 节点间通信（首选）|
| **RoCEv2** | 100-200 Gb/s | ~2-5 μs | 成本敏感场景 |
| **Ethernet** | 10-100 Gb/s | ~10+ μs | 不推荐训练 |

### 1.3 存储系统

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    存储层次架构                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────┐                                                       │
│   │  GPU HBM    │ ← 最快，但容量小 (80GB)                               │
│   └─────────────┘                                                       │
│          ↓ PCIe                                                         │
│   ┌─────────────┐                                                       │
│   │  NVMe SSD   │ ← 本地缓存，预加载热数据 (~TB 级)                      │
│   └─────────────┘                                                       │
│          ↓ 网络                                                         │
│   ┌─────────────┐                                                       │
│   │  Lustre/GPFS│ ← 并行文件系统，高吞吐 (数 GB/s)                       │
│   └─────────────┘                                                       │
│          ↓                                                              │
│   ┌─────────────┐                                                       │
│   │  S3/对象存储│ ← 冷数据存档，容量大成本低 (PB 级)                     │
│   └─────────────┘                                                       │
│                                                                         │
│   💡 VLA 训练数据是视频/图片，随机读取性能是瓶颈                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**数据加载优化**:
- **WebDataset/TFRecord**: 小文件打包成 Tar，顺序读取
- **NVIDIA DALI**: GPU 解码图片/视频，释放 CPU 压力
- **预取 (Prefetch)**: DataLoader 多进程预加载下一个 batch

---

## 2. 分布式训练策略 (Distributed Training)

### 2.1 并行策略对比

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    分布式并行策略一览                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   策略               切分维度        显存效率    通信开销    实现复杂度   │
│   ───────────────────────────────────────────────────────────────────   │
│   Data Parallel      数据批次        低          低          ★☆☆        │
│   (DDP)                                                                 │
│                                                                         │
│   FSDP               参数+梯度+      高          中          ★★☆        │
│   (ZeRO-3)           优化器状态                                         │
│                                                                         │
│   Tensor Parallel    矩阵列/行       中          高          ★★★        │
│   (TP)                                                                  │
│                                                                         │
│   Pipeline Parallel  网络层          中          中          ★★★        │
│   (PP)                                                                  │
│                                                                         │
│   3D Parallel        DP + TP + PP    最高        最高        ★★★★       │
│                                                                         │
│   💡 VLA 7B 模型: FSDP 通常够用; 70B+: 需要 3D 并行                     │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Parallel (DDP)

最简单的并行策略，每个 GPU 持有完整模型副本。

```python
# PyTorch DDP 基本用法
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化进程组
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

# 模型包装
model = MyVLAModel().cuda()
model = DDP(model, device_ids=[local_rank])

# 数据采样器
sampler = DistributedSampler(dataset, shuffle=True)
dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_per_gpu)
```

**限制**: 模型必须能放进单卡显存 → 7B FP16 需要 ~14GB，单卡可行

### 2.3 FSDP (Fully Sharded Data Parallel)

PyTorch 原生的 ZeRO-3 实现，**VLA 微调首选**。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FSDP 工作原理                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   传统 DDP:                                                             │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐                                │
│   │ GPU 0   │  │ GPU 1   │  │ GPU 2   │                                │
│   │ 完整模型 │  │ 完整模型 │  │ 完整模型 │   ← 显存浪费!                   │
│   │ 完整梯度 │  │ 完整梯度 │  │ 完整梯度 │                                │
│   │ 完整优化 │  │ 完整优化 │  │ 完整优化 │                                │
│   └─────────┘  └─────────┘  └─────────┘                                │
│                                                                         │
│   FSDP:                                                                 │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐                                │
│   │ GPU 0   │  │ GPU 1   │  │ GPU 2   │                                │
│   │ 参数 1/3 │  │ 参数 2/3 │  │ 参数 3/3 │   ← 切分存储                   │
│   │ 梯度 1/3 │  │ 梯度 2/3 │  │ 梯度 3/3 │                                │
│   │ 优化 1/3 │  │ 优化 2/3 │  │ 优化 3/3 │                                │
│   └─────────┘  └─────────┘  └─────────┘                                │
│        ↓            ↓            ↓                                      │
│        └──── All-Gather 重建完整参数 ────┘                              │
│                    ↓                                                    │
│              Forward/Backward                                           │
│                    ↓                                                    │
│        ┌──── Reduce-Scatter 切分梯度 ────┐                              │
│        ↓            ↓            ↓                                      │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐                                │
│   │ 更新 1/3 │  │ 更新 2/3 │  │ 更新 3/3 │                                │
│   └─────────┘  └─────────┘  └─────────┘                                │
└─────────────────────────────────────────────────────────────────────────┘
```

```python
# FSDP 配置示例
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

# 混合精度配置
mixed_precision_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

# FSDP 包装
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
    mixed_precision=mixed_precision_policy,
    auto_wrap_policy=transformer_auto_wrap_policy,   # 按 Transformer Block 切分
    device_id=torch.cuda.current_device(),
)
```

### 2.4 Tensor Parallelism (TP)

切分矩阵运算，**适合超大模型单层放不下的情况**。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Tensor Parallelism (列切分示例)                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Linear Layer: Y = XW + b                                              │
│                                                                         │
│   ┌───────────────────────────────────────────────────┐                 │
│   │                     W (权重矩阵)                   │                 │
│   │              [hidden, output_dim]                  │                 │
│   └───────────────────────────────────────────────────┘                 │
│                           ↓                                             │
│               按列切分到 N 个 GPU                                        │
│                           ↓                                             │
│   ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐              │
│   │    W_0    │ │    W_1    │ │    W_2    │ │    W_3    │              │
│   │  GPU 0    │ │  GPU 1    │ │  GPU 2    │ │  GPU 3    │              │
│   └───────────┘ └───────────┘ └───────────┘ └───────────┘              │
│        ↓             ↓             ↓             ↓                      │
│      Y_0 = XW_0   Y_1 = XW_1   Y_2 = XW_2   Y_3 = XW_3                  │
│        └─────────────┴─────────────┴─────────────┘                      │
│                           ↓                                             │
│                     All-Gather                                          │
│                           ↓                                             │
│                      Y = [Y_0, Y_1, Y_2, Y_3]                           │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.5 Pipeline Parallelism (PP)

切分模型层，**适合层数多的深层网络**。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Pipeline Parallelism                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   模型层分配:                                                           │
│   GPU 0: [Embedding, Layer 0-7]                                         │
│   GPU 1: [Layer 8-15]                                                   │
│   GPU 2: [Layer 16-23]                                                  │
│   GPU 3: [Layer 24-31, LM Head]                                         │
│                                                                         │
│   Micro-batch 流水线 (减少 Bubble):                                     │
│                                                                         │
│   Time ────────────────────────────────────────────►                    │
│                                                                         │
│   GPU 0: │ F0 │ F1 │ F2 │ F3 │    │    │    │    │ B3 │ B2 │ B1 │ B0 │  │
│   GPU 1: │    │ F0 │ F1 │ F2 │ F3 │    │    │ B3 │ B2 │ B1 │ B0 │    │  │
│   GPU 2: │    │    │ F0 │ F1 │ F2 │ F3 │ B3 │ B2 │ B1 │ B0 │    │    │  │
│   GPU 3: │    │    │    │ F0 │ F1 │ F2 │ F3 │ B2 │ B1 │ B0 │    │    │  │
│                                                                         │
│   F = Forward, B = Backward                                             │
│   Bubble (空闲) 在流水线填充和排空阶段不可避免                           │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.6 3D 并行 (Megatron-LM / DeepSpeed)

**超大规模训练 (100B+) 的标准配置**。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    3D 并行示意 (64 GPUs)                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Data Parallel (DP=8):  8 个相同的模型副本                              │
│        ├── Pipeline Parallel (PP=4): 每个副本分 4 个阶段                 │
│              └── Tensor Parallel (TP=2): 每阶段的矩阵切 2 份             │
│                                                                         │
│   总 GPU = DP × PP × TP = 8 × 4 × 2 = 64                                │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  DP Group 0                                                     │   │
│   │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐ │   │
│   │  │ PP Stage 0  │ │ PP Stage 1  │ │ PP Stage 2  │ │ PP Stage 3│ │   │
│   │  │ GPU0  GPU1  │ │ GPU2  GPU3  │ │ GPU4  GPU5  │ │ GPU6 GPU7 │ │   │
│   │  │  TP=2       │ │  TP=2       │ │  TP=2       │ │  TP=2     │ │   │
│   │  └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘ │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  DP Group 1 ... (同样结构 x 8)                                  │   │
│   └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 训练优化技术 (Training Optimization)

### 3.1 混合精度训练 (Mixed Precision)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    数值精度对比                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   精度        位数    范围           显存     速度     VLA 推荐           │
│   ────────────────────────────────────────────────────────────────────  │
│   FP32        32     ±3.4e38        100%     1x       存优化器状态       │
│   FP16        16     ±65504         50%      2x       ⚠️ 溢出风险        │
│   BF16        16     ±3.4e38        50%      2x       ✅ 推荐 (H100/A100) │
│   TF32        19*    ±3.4e38        ~50%     ~2x      NVIDIA 默认        │
│   INT8        8      ±127           25%      4x       推理加速           │
│                                                                         │
│   💡 BF16 = FP32 的指数范围 + FP16 的速度，是大模型训练首选              │
└─────────────────────────────────────────────────────────────────────────┘
```

```python
# PyTorch AMP (Automatic Mixed Precision)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast(dtype=torch.bfloat16):  # 前向用 BF16
        outputs = model(batch)
        loss = criterion(outputs, targets)
    
    scaler.scale(loss).backward()  # 反向梯度缩放
    scaler.step(optimizer)
    scaler.update()
```

### 3.2 Gradient Checkpointing (梯度检查点)

**用计算换显存**：不保存所有激活值，反向时重新计算。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Gradient Checkpointing                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   传统方式: 保存所有激活值                                               │
│   ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐                     │
│   │ Act0│ Act1│ Act2│ Act3│ Act4│ Act5│ Act6│ Act7│  ← 显存爆炸!        │
│   └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘                     │
│                                                                         │
│   Checkpointing: 只保存检查点，中间重算                                  │
│   ┌─────┬     ┬     ┬─────┬     ┬     ┬     ┬─────┐                     │
│   │ Act0│     │     │ Act3│     │     │     │ Act7│  ← 显存 ÷ 4         │
│   └─────┴     ┴     ┴─────┴     ┴     ┴     ┴─────┘                     │
│              ↑ 反向时从 Act0 重算 Act1, Act2                             │
│                                                                         │
│   代价: 约 30% 额外计算时间                                              │
└─────────────────────────────────────────────────────────────────────────┘
```

```python
# PyTorch 启用 Gradient Checkpointing
from torch.utils.checkpoint import checkpoint_sequential

# 对 Transformer blocks 启用
model.transformer.gradient_checkpointing_enable()

# 或手动
def forward_with_checkpoint(self, x):
    return checkpoint_sequential(self.layers, segments=4, input=x)
```

### 3.3 Gradient Accumulation (梯度累积)

**小显存也能跑大 Batch Size**。

```python
accumulation_steps = 8
effective_batch_size = batch_per_gpu * num_gpus * accumulation_steps

for i, batch in enumerate(dataloader):
    with autocast():
        loss = model(batch) / accumulation_steps  # 缩放 loss
    
    loss.backward()  # 累积梯度
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## 4. 训练稳定性 (Training Stability)

### 4.1 常见问题与解决方案

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    训练异常诊断手册                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   问题              症状                    解决方案                      │
│   ──────────────────────────────────────────────────────────────────    │
│   Loss Spike        Loss 突然飙升           1. 降低学习率                 │
│                     再缓慢恢复              2. 启用梯度裁剪               │
│                                             3. 检查异常数据               │
│                                                                         │
│   Loss NaN          Loss 变成 NaN           1. 检查 FP16 溢出 → 用 BF16  │
│                                             2. 降低学习率                 │
│                                             3. 增加 warmup steps          │
│                                                                         │
│   Loss 不下降       Loss 平台期             1. 调整学习率调度器           │
│                                             2. 检查数据质量               │
│                                             3. 增大 batch size            │
│                                                                         │
│   显存 OOM          CUDA out of memory      1. 减小 batch size            │
│                                             2. 启用 Gradient Checkpointing│
│                                             3. 使用 FSDP/DeepSpeed        │
│                                                                         │
│   GPU 利用率低      < 80%                   1. 增加 DataLoader workers    │
│                                             2. 使用 NVIDIA DALI           │
│                                             3. 检查网络瓶颈               │
│                                                                         │
│   训练速度不稳定    每个 step 时间波动大    1. 禁用动态 shape             │
│                                             2. 检查 GC (垃圾回收)         │
│                                             3. 固定 batch size            │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 梯度裁剪 (Gradient Clipping)

防止梯度爆炸的标准做法。

```python
# 按范数裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 按值裁剪
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

### 4.3 学习率调度 (Learning Rate Schedule)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    常用学习率调度器                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Warmup + Cosine Decay (大模型训练首选):                               │
│                                                                         │
│   LR ▲                                                                  │
│      │    ╭──────╮                                                      │
│      │   ╱        ╲                                                     │
│      │  ╱          ╲                                                    │
│      │ ╱            ╲                                                   │
│      │╱              ╲______                                            │
│      └─────────────────────────────► Steps                              │
│        ↑              ↑                                                 │
│      Warmup         Peak                                                │
│      (1-5%)        (最大 LR)                                            │
│                                                                         │
│   推荐配置:                                                             │
│   - Peak LR: 1e-5 ~ 1e-4 (微调), 1e-4 ~ 3e-4 (预训练)                   │
│   - Warmup: 总步数的 1-5%                                               │
│   - Min LR: Peak 的 1-10%                                               │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.4 Checkpoint 策略

```python
# 定期保存 + 最佳模型保存
def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'rng_state': torch.get_rng_state(),  # 随机数状态，确保可复现
        'cuda_rng_state': torch.cuda.get_rng_state_all(),
    }, path)

# 断点续训
def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    torch.set_rng_state(checkpoint['rng_state'])
    torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
    return checkpoint['epoch'], checkpoint['loss']
```

---

## 5. 监控与调试 (Monitoring & Debugging)

### 5.1 关键指标监控

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    训练监控 Dashboard                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   必监控指标:                                                           │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │  Loss (train/val)     │  梯度范数  │  学习率   │  吞吐量        │   │
│   │  ████████░░ 0.23      │  ▁▂▃▄▅▆▇  │  1e-4     │  1200 tok/s    │   │
│   └────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│   资源监控:                                                             │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │  GPU 利用率        │  显存占用      │  网络带宽     │  磁盘 IO   │   │
│   │  GPU0: 95%        │  72/80 GB      │  350 Gb/s    │  2.1 GB/s  │   │
│   │  GPU1: 94%        │  71/80 GB      │              │            │   │
│   └────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│   工具:                                                                 │
│   - Weights & Biases: 曲线可视化、超参搜索                              │
│   - TensorBoard: PyTorch 原生支持                                       │
│   - nvidia-smi / nvitop: GPU 实时监控                                   │
│   - Prometheus + Grafana: 集群级监控                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Profiling 工具

```python
# PyTorch Profiler
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=tensorboard_trace_handler('./log/profiler'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step, batch in enumerate(dataloader):
        train_step(batch)
        prof.step()

# 查看结果
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### 5.3 显存分析

```python
# 显存使用详情
torch.cuda.memory_summary(device=0, abbreviated=True)

# 找显存泄漏
import gc
gc.collect()
torch.cuda.empty_cache()

# 显存快照 (PyTorch 2.0+)
torch.cuda.memory._record_memory_history()
# ... run some code ...
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
```

---

## 6. 面试高频 Q&A

### Q1: 描述一次你遇到的大规模训练问题，是如何解决的？

**模板回答**:
> "在训练 7B VLA 模型时，我们遇到了 **Loss Spike** 问题。排查发现是某些视频数据帧损坏导致的。
> 
> 解决方案：
> 1. 添加数据预检脚本，过滤异常样本
> 2. 启用梯度裁剪 (max_norm=1.0)
> 3. 添加 Loss Spike 检测，自动跳过异常 batch
> 
> 最终训练稳定完成。"

### Q2: FSDP 和 DeepSpeed ZeRO 有什么区别？

| 维度 | FSDP | DeepSpeed ZeRO |
| :--- | :--- | :--- |
| **实现** | PyTorch 原生 | Microsoft 第三方库 |
| **易用性** | 更简洁，API 统一 | 功能更多，配置复杂 |
| **功能** | ZeRO-3 | ZeRO-1/2/3, Offload, Infinity |
| **社区** | PyTorch 官方维护 | Microsoft 维护 |
| **推荐** | 7B-70B 微调首选 | 100B+ 或需要 Offload 时 |

### Q3: 如何优化 GPU 利用率？

1. **DataLoader 优化**: 增加 workers, 使用 pin_memory
2. **GPU 解码**: 使用 NVIDIA DALI 进行图片/视频解码
3. **减少通信**: 增大 batch size, 使用梯度累积
4. **重叠计算与通信**: FSDP 的 `limit_all_gathers=True`
5. **编译优化**: `torch.compile()` (PyTorch 2.0+)

### Q4: 如何估算训练所需资源？

```python
# 显存粗估公式 (FP16 训练)
model_params = 7e9  # 7B
bytes_per_param = 2  # FP16
gradient_bytes = 2   # FP16
optimizer_bytes = 8  # FP32 momentum + variance

# 模型 + 梯度 + 优化器
base_memory = model_params * (bytes_per_param + gradient_bytes + optimizer_bytes)
# 7B * 12 bytes ≈ 84 GB

# 激活值 (取决于 batch size 和序列长度)
activation_memory = batch_size * seq_len * hidden_dim * num_layers * 2  # 粗估

# FSDP 可将 base_memory 切分到 N 个 GPU
```

### Q5: 混合精度训练为什么需要 Loss Scaling？

FP16 的最小正数是 ~6e-8，梯度值如果太小会变成 0 (**梯度下溢**)。

**Loss Scaling 原理**:
1. **放大 Loss**: `scaled_loss = loss * scale_factor` (如 65536)
2. **梯度也被放大**: 反向传播时梯度同比例放大
3. **更新前缩小**: `grad = grad / scale_factor`
4. **动态调整**: 如果梯度溢出 (Inf/NaN)，减小 scale_factor

BF16 不需要 Loss Scaling，因为它的指数范围和 FP32 相同。

---

## 7. 实战 Checklist

训练大模型前的检查清单：

- [ ] **数据**: 验证数据格式正确，无损坏样本
- [ ] **显存**: 估算显存需求，确认硬件配置
- [ ] **网络**: 多机训练确认 InfiniBand/RoCE 正常
- [ ] **存储**: 数据读取速度足够，不卡 GPU
- [ ] **Checkpoint**: 配置定期保存，支持断点续训
- [ ] **监控**: W&B / TensorBoard 配置完成
- [ ] **日志**: 完整记录超参数和环境版本
- [ ] **回滚**: 保留上一个稳定版本的 checkpoint

---

## 📚 推荐资源

- [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)

---

[← Back to System Design](./README.md)

