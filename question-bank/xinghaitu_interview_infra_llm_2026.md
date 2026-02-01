# 星海图面经：训练系统 & 大模型基础（含手撕）逐题应答稿 (2026)

> **定位**：把面经题目变成“可复述、可推导、可落地”的回答脚本（面试官追问也扛得住）。  
> **题目来源**：小红书（用户整理的面经帖）  

---

## 1) Zero123 是什么？它解决什么问题？

### 建议回答（30 秒版）
**Zero-1-to-3（常被口语叫 Zero123）**是一类“从单张图像合成多视角”的模型：输入一张物体图片 + 目标视角（相机位姿/方位角等），输出该视角下的渲染图。常用于：
- **单视图 → 多视图数据增强**（为 3D 重建、位姿估计、NeRF/GS 等提供伪多视角监督）
- **几何/外观一致性约束**（通过多视角一致性提升下游的 3D 可用性）

### 追问准备
- **它为什么对具身有用**：很多机器人数据是单视角/单相机，Zero123 可“补多视角”，帮助 6D pose / grasp / tracking 这类几何任务的数据覆盖。

**参考**：Zero-1-to-3 论文（arXiv）：`https://arxiv.org/abs/2303.11328`

---

## 2) 你对比过普通 DDP 和 ZeRO-1 吗？4 卡时显存能省多少？

### 一句话结论
**ZeRO-1 的核心收益是把 optimizer state 分片（shard）**，因此显存节省主要来自 optimizer state；参数与梯度仍是复制（replicate）。

### 用“每个参数占多少字节”算一遍（面试官喜欢）
以 Adam 为例（常见实现）：
- 参数 \(W\)：fp16/bf16（2 bytes）
- 梯度 \(\nabla W\)：fp16/bf16（2 bytes，具体取决于实现）
- Adam 一阶/二阶矩 \(m,v\)：通常 fp32（各 4 bytes → 合计 8 bytes）

则 **DDP（不分片）**每参数显存大致：
\[
2 + 2 + 8 = 12\ \text{bytes/param}
\]

**ZeRO-1（N 卡）**把 optimizer state 分片为 \(1/N\)，则每参数显存大致：
\[
2 + 2 + \frac{8}{N}
\]
4 卡时：
\[
2 + 2 + \frac{8}{4} = 6\ \text{bytes/param}
\]

因此在这种典型设定下，**显存大约减半（12 → 6 bytes/param）**。  
注意：实际数值会因 **梯度精度、是否保留 fp32 master weights、是否启用 gradient accumulation/activation checkpointing** 而变化，但“ZeRO-1 主要省 optimizer state”这个结论不变。

---

## 3) CPU offload 是什么？为什么能省显存，代价是什么？

### 建议回答
CPU offload 指把一部分“占显存的大户”放到 CPU 内存（甚至 NVMe）：
- 常见 offload 对象：**optimizer state / gradients / params**（不同 ZeRO stage 支持不同粒度）

### 代价（必须主动说）
- **吞吐下降**：PCIe/NVLink 传输 + CPU 计算更慢，step time 增加
- **抖动增大**：IO/内存带宽波动导致 latency jitter
- **工程复杂度上升**：pin memory、prefetch、分层缓存、容错/恢复

一句话：**用时间换空间**，适合“显存卡死但能接受慢一点”的训练。

---

## 4) BF16 和 FP16 的区别？为什么训练大模型更偏向 BF16？

### 关键点
- **FP16**：指数位更少，动态范围更窄，更容易溢出/下溢；通常强依赖 loss scaling。
- **BF16**：指数位与 FP32 相同（动态范围更大），更抗溢出；通常 **更稳、更少调参**。

### 工程回答
在支持 BF16 的硬件（A100/H100 等）上：
- BF16 往往在稳定性上更好（尤其长训练、分布外 batch、梯度尖峰）
- 速度通常与 FP16 同量级（取决于硬件与 kernel）

---

## 5) & 6) checkpoint 落在 epoch 中间：resume 时如何“无缝衔接”数据，不重复不丢失？

这题面试官核心在问：**“resume 不只恢复模型/优化器，还要恢复数据顺序。”**

### 先给结论（工程可落地）
要做到“无缝衔接”，你至少需要恢复：
- **epoch**（或等价的 `sampler.set_epoch(epoch)` 种子状态）
- **在该 epoch 内已经消费的 batch/样本 offset**（`steps_in_epoch_done`）
- **所有 RNG 状态**（Python/NumPy/PyTorch/CUDA），否则 shuffle/augmentation 可能不一致

### 方案 A（最常见，HF Trainer/Accelerate 类做法）：重建同一顺序 + skip
1. 让 `DistributedSampler(shuffle=True, seed=...)` 在相同 epoch 下生成确定性 index 序列  
2. resume 时根据 `global_step` 计算“已经走过多少个 dataloader batch”  
3. **跳过前 N 个 batch**（skip-first-batches），继续训练

优点：简单、通用；缺点：恢复时可能要“空跑/跳过”一些 batch（但通常可接受）。

### 方案 B（更硬核）：StatefulSampler / 保存 index 游标
把 sampler 做成可 `state_dict()`：
- 保存 `epoch、seed、cursor（当前 index 指针）、shuffle permutation`  
resume 时从 cursor 继续吐 index。

优点：真正无缝；缺点：实现/维护成本更高，且要兼容多卡一致性。

### 你可以补一句“工程取舍”
大多数工程里会选 **方案 A**（确定性 + skip），因为它足够稳且容易与现有 Trainer 集成。

---

## 7) DDP 里同时涉及 DataLoader 和 Sampler：shuffle 应该由谁负责？

### 标准答案
**由 Sampler（尤其是 DistributedSampler）负责 shuffle**。

原因：
- 分布式要保证各 rank **不重叠且覆盖完整数据集**
- DataLoader 的 `shuffle=True` 不知道 world size / rank，容易导致重复或覆盖不全

工程写法：
- `DataLoader(..., shuffle=False, sampler=DistributedSampler(..., shuffle=True))`

---

## 8) DistributedSampler 如何在无通信下生成各进程的 index 且不重叠？

### 关键逻辑（面试官想听的实现细节）
给定 `num_replicas=world_size` 与 `rank`：
1. 先生成一个全局 `indices=[0..len-1]`，若 `shuffle=True` 则用 `seed+epoch` 做确定性打乱  
2. 计算 `total_size = num_samples * num_replicas`，必要时 padding/截断，使可整除  
3. 各 rank 取子序列：
\[
\text{indices}_{rank} = \text{indices}[rank : total\_size : num\_replicas]
\]

这样每个 rank 的切片互不重叠，拼起来覆盖全局序列；全程不需要进程间通信，因为大家用同一套确定性规则生成同一个全局序列。

---

## 9) GQA 是什么？解决什么问题？

### 一句话
**GQA（Grouped-Query Attention）**：让多个 Query head 共享更少的 Key/Value head（\(n_{kv} < n_q\)），从而降低 **KV cache** 的显存与带宽。

### 你可以补一句“为什么重要”
长上下文推理/长任务里，KV cache 往往是显存大头；GQA 用更少的 KV 头显著省显存，同时通常比 MQA（单 KV）更稳。

---

## 10) RMSNorm 和 LayerNorm 的区别？

### 公式直觉
- **LayerNorm**：对每个 token 的 hidden 做均值与方差归一化（减去 mean，再除 std）
- **RMSNorm**：不减均值，只按 RMS（均方根）缩放（更简单、更快，且在大模型里常用）

### 工程差异
RMSNorm 通常：
- kernel 更轻
- 数值更稳定/更常用（取决于架构）

---

## 11) tokenization 放哪里更合理：Dataset/DataLoader 阶段，还是 forward 前？

### 面试回答的“权衡结构”
看你更在意：
- **吞吐与稳定**：尽量前置（Dataset 预处理/离线缓存/packing），训练时只做轻量拼装  
- **灵活性**：放在 forward 前，方便随时换 tokenizer、改 special tokens、做动态 prompt

### 大模型训练里的常见实践
- **文本 tokenization**：倾向于离线预处理 + 缓存（避免 CPU 成为瓶颈）  
- **多模态（图像/视频）处理**：一部分在 DataLoader（resize/normalize），一部分在模型内（视觉 encoder）  

一句话：**高吞吐训练更倾向前置/缓存；研究迭代更倾向模型内动态处理。**

---

## 12) 手撕：省份数量

这题通常是 LeetCode “省份数量 / 朋友圈” 变体：给定 \(N\times N\) 邻接矩阵 `isConnected`，求连通分量数量。

### DFS/BFS 思路
- 遍历每个城市 \(i\)，若未访问则 DFS/BFS 扩展所有与之连通的城市；计数 +1

复杂度：
- 时间 \(O(N^2)\)
- 空间 \(O(N)\)

### 并查集（Union-Find）也可
- 扫描矩阵的上三角，遇到 1 就 union；最后统计根的数量

---

[← Back to Question Bank](./README.md)

