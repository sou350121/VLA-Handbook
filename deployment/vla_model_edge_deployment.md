# 模型优化与边缘部署 (Optimization & Edge Deployment)

VLA 模型通常参数量巨大 (7B+)，而机器人端的计算资源 (如 Jetson Orin) 有限。本章介绍如何将大模型塞进小设备。

## 1. 模型量化 (Quantization)
量化是降低显存占用和加速推理最有效的方法。

### 常用技术
- **GPTQ (Post-Training Quantization)**:
    - 仅需少量校准数据，即可将模型量化为 4-bit 或 8-bit。
    - 精度损失极小，推理速度提升显著。
- **AWQ (Activation-aware Weight Quantization)**:
    - 考虑到激活值的分布，保护重要的权重不被过度量化。
    - 在边缘设备上通常比 GPTQ 更快，且精度更好。
- **QLoRA (Quantized LoRA)**:
    - 训练时的量化技术。在 4-bit 冻结的基础模型上微调 LoRA 适配器。
    - 使得在单张消费级显卡 (e.g., RTX 3090/4090) 上微调 7B+ 模型成为可能。

### 显存需求估算 (7B Model)
- **FP16 (16-bit)**: ~14 GB
- **INT8 (8-bit)**: ~7 GB
- **INT4 (4-bit)**: ~4 GB (可运行在 Jetson Orin NX 16GB 上)

## 2. 边缘推理框架 (Edge Inference Frameworks)

### TensorRT-LLM
- **NVIDIA 官方** 推出的 LLM 推理加速库。
- **特点**: 极致的性能优化，支持 In-flight batching, PagedAttention。
- **部署**: 需要将 PyTorch 模型编译为 TensorRT Engine。
- **适用**: Jetson Orin 系列，追求极致 FPS。

### vLLM
- **特点**: 易用性好，吞吐量高 (PagedAttention)。
- **部署**: 原生支持 HuggingFace 模型，无需复杂的编译过程。
- **适用**: 快速验证，服务器端部署。

### ONNX Runtime
- **特点**: 跨平台，支持多种硬件 (CPU, GPU, NPU)。
- **适用**: 非 NVIDIA 硬件，或需要极高兼容性的场景。

## 3. 部署实战：在 Jetson Orin 上部署 OpenVLA

### 步骤概览
1. **环境准备**: 安装 JetPack 6.0, CUDA, PyTorch。
2. **模型转换**: 使用 `auto_gptq` 或 `llm-awq` 将 OpenVLA 权重转换为 4-bit AWQ 格式。
3. **加载推理**:
    ```python
    from vllm import LLM, SamplingParams

    # 加载 4-bit 量化模型
    llm = LLM(model="openvla-7b-awq", quantization="awq")
    
    # 推理
    prompts = ["Image: <image_features> Instruction: Pick up the apple."]
    outputs = llm.generate(prompts)
    ```
4. **性能调优**: 调整 `max_num_seqs` 和 `gpu_memory_utilization` 以平衡延迟和吞吐量。

## 面试高频考点
1. **量化精度**: 4-bit 量化对 VLA 的动作预测精度有影响吗？(答: 有轻微影响，但在闭环控制中通常可接受，因为反馈回路会修正误差)
2. **延迟 (Latency)**: 机器人控制对延迟的要求是多少？(答: 通常 < 100ms，理想 < 30ms。大模型推理通常是瓶颈)
3. **KV Cache**: 什么是 KV Cache？在机器人连续控制中如何管理？(答: 缓存历史 Token 的 Key/Value，避免重复计算。但在 VLA 中，通常只关注当前帧或短历史，Context Window 不会无限增长)

---
[← Back to Deployment](./README.md)
