# VLA 社交情报 | 2026-11-06

📡 VLA 社交情报 | 2026-11-06

- **[开源项目]** LeRobot 于 2026-11-05 发布 v0.7.0：Hub 上近 300 TB 开放 Physical AI 数据支持流式训练（Hub/Storage Bucket/LanceDB），训练栈面向 5-20B 参数 world action model 重建（FSDP2、HSDP、Torch DCP；8×H200 上 6B FastWAM 的 checkpoint 保存快 3x、恢复峰值 143GB→42GB），新增 interactive rollout（/subtask、/vqa、/autosteer）与 lerobot-policy-server 远程推理。
  来源: [Hugging Face Blog (LeRobot)](https://huggingface.co/blog/lerobot/lerobot-release-v070) | 来源发布：2026-11-05 | 事件日期：2026-11-05
- **[开源项目]** FineART 数据集由 Scale AI 与 Hugging Face 提供：40k+ 双手操作 episodes、1,718 小时、533k+ subtask 标注、1.9 TB LeRobot 格式；FineART-VLA 在 π0.5 基础上加语言头，由高层系统先用语言预测下一子任务（System 2），flow-matching 动作专家再执行（System 1）。原文未说明该数据集的发布日。
  来源: [Hugging Face Blog (LeRobot)](https://huggingface.co/blog/lerobot/lerobot-release-v070) | 来源发布：2026-11-05 | 事件日期：未确认
- **[开源项目]** OpenGalaxea 的 G0.5 已接入 LeRobot：Qwen3.5-2B VLA，把 chain-of-thought（当前子任务、可选 bounding box）与动作写在同一流，可按快速 System 1 运行，也可作为 System 2 推理；四个转换后 checkpoint 覆盖 base、LIBERO、RoboTwin 2.0、SO-101，许可为 G0.5 Community License（非 Apache-2.0）。
  来源: [Hugging Face Blog (LeRobot)](https://huggingface.co/blog/lerobot/lerobot-release-v070) | 来源发布：2026-11-05 | 事件日期：未确认
- **[开源项目]** Robbyant 的 LingBot-VLA 2.0 集成进 LeRobot：Qwen3-VL-4B + 32 experts/top-4 稀疏 MoE 动作专家、55 维跨本体动作空间；原文报告 RoboTwin 2.0 clean 上 93.5% 成功率（π0.5 为 82.7%），CUDA graph 与 torch.compile 把 action chunk 从约 756 ms 降至约 120 ms（bf16，RTX PRO 6000）。
  来源: [Hugging Face Blog (LeRobot)](https://huggingface.co/blog/lerobot/lerobot-release-v070) | 来源发布：2026-11-05 | 事件日期：未确认
- **[开源项目]** LeRobot v0.7.0 收录 NVIDIA Cosmos 3（Nano 16.5B / Edge 4.6B，LIBERO 平均成功率 97.90%/97.95%，支持 FSDP2 全参与 LoRA 微调）与 NVIDIA LongLive 团队的 Long-WAM（以自回归视频预训练做长上下文，IDM 模式 99.00%、COD 模式 98.45%，各 2000 episodes）。
  来源: [Hugging Face Blog (LeRobot)](https://huggingface.co/blog/lerobot/lerobot-release-v070) | 来源发布：2026-11-05 | 事件日期：未确认
- **[开源项目]** Black Forest Labs 在 LeRobot v0.7.0 中释出 FLUX 3 Action：冻结视频 VAE 与文本编码器驱动扩散 transformer，联合预测视频 latent 与动作 chunk（推理时只输出动作）；发布 base、SO-101、DROID 三个 checkpoint，支持推理与 LoRA 微调，新本体需用其 flux-action 仓库训练。
  来源: [Hugging Face Blog (LeRobot)](https://huggingface.co/blog/lerobot/lerobot-release-v070) | 来源发布：2026-11-05 | 事件日期：未确认
- **[开源项目]** Dexmal DM0.5 由 OpenDM 移植至 LeRobot：Gemma3-4B VLM + 680M flow-matching 动作专家，lerobot/dm05_libero 在 LIBERO 复现得 197/200；同一版本收录 X Square Robot 的 4B Wall-OSS-0.5（Qwen2.5-VL + flow-matching），可用于 /vqa 与 /autosteer 交互式 rollout。
  来源: [Hugging Face Blog (LeRobot)](https://huggingface.co/blog/lerobot/lerobot-release-v070) | 来源发布：2026-11-05 | 事件日期：未确认

---

*本報告由 VLA-Handbook 自動生成*
