# VLA/Embodied AI 英文社区实战笔记

> **版本**: v1.0 — 2026-03-15
> **数据来源**: HuggingFace Blog、GitHub Issues、厂商技术博客、Discord 社区
> **对应中文版**: `community_field_notes_xiaohongshu.md`（220+ 篇小红书帖子）
> **采集 skill**: `scripts/en-vla-collector/english-vla-collector-SKILL.md`

---

## 类别说明

| 标签 | 含义 |
|------|------|
| Recipe | 训练配方（config/超参/数据量/结果） |
| Debug | 踩坑调试（bug/错误/修复） |
| Edge | 边缘部署（推理延迟/量化/芯片） |
| Data | 数据工程（采集/格式/规模） |
| Arch | 架构选型（设计决策/benchmark） |
| Strategy | 产业战略（趋势/部署策略） |

---

## 可追溯信息表

| # | 标题 | 来源 | 日期 | 类别 | URL | 核心内容 |
|---|------|------|------|------|-----|---------|
| 1 | SmolVLA: Efficient VLA trained on Community Data | HF Blog | 2025-12-16 | Recipe | [链接](https://huggingface.co/blog/smolvla) | 450M 参数，<30k episodes 即可训练；异步推理比同步快 30%（9.7s vs 13.75s），2× task throughput；社区预训练数据 +26.6% 改进（51.7%→78.3%）；SO100/SO101 硬件；`batch_size=64, steps=20000` |
| 2 | Post-Training GR00T N1.5 for SO-101 Arm | HF Blog (NVIDIA) | 2025-12-05 | Recipe | [链接](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning) | 默认 ~25GB VRAM；`--no-tune_diffusion_model` 省显存；SO-100/SO-101 不在预训练中需用 `new_embodiment` 标签；社区反馈 30 episodes + 6000 steps 可产出合理策略；denoising_steps=16 + action_horizon=16 适合复杂任务；jerky motion 常见问题 |
| 3 | Bringing Robotics AI to Embedded Platforms | HF Blog (NXP) | 2026-03-05 | Edge | [链接](https://huggingface.co/blog/nxp/bringing-robotics-ai-to-embedded-platforms) | NXP i.MX 95 SoC（6× A55 + NPU）；ACT ONNX FP32: 2.86s/96% 准确率 → 优化后 0.32s/89%；SmolVLA ONNX FP32: 29.1s/47% → 优化后 6.15s；量化 action expert（flow matching）严重降精度；120 episodes, 3 cameras, 10 clusters；20% recovery episodes 提升成功率；异步推理必须 inference < execution time |
| 4 | LeRobot v0.5.0: Scaling Every Dimension | HF Blog | 2026-03-09 | Arch | [链接](https://huggingface.co/blog/lerobot-release-v050) | 200+ PRs, 50+ 新贡献者；首个人形支持（Unitree G1 全身控制）；Pi0-FAST 自回归 VLA（Gemma 300M action expert + FAST tokenization）；RTC 实时分块推理（Physical Intelligence）；Wall-X（Qwen2.5-VL + flow matching）；X-VLA（Florence-2）；SARM 长序列奖励建模；PEFT/LoRA 支持；流式视频编码零等待；10× 图像训练加速 |
| 5 | Train ACT on SO-101: Journey, Gotchas, Lessons | HF Blog (Sherry Chen) | 2025-09-30 | Recipe, Debug | [链接](https://huggingface.co/blog/sherryxychen/train-act-on-so-101) | 3 轮迭代：Try1 50ep→woodpecker，Try2 72ep→60% ID/10% OOD，Try3 125ep+旋转→90% ID/75% OOD；ACT 52M 4h on RTX 3080 12GB；关键坑：相机 POV 不固定、标定文件丢失、USB 路径冲突、夹爪电机过力损坏；解决：udev rules、固定曝光、progress score 评估、分层采样；结论：买备用电机 |
| 6 | Fine Tuning SmolVLA for New Environments | Medium (CU Correll Lab) | 2026-01-21 | Recipe, Debug | [链接](https://medium.com/correll-lab/fine-tuning-smolvla-for-new-environments-code-included-af266c56d632) | SmolVLA 450M 在 Franka Panda 仿真上微调；25 demos→学到 dipping motion，125 demos→40% grasp success；关键坑：stats.json 不匹配导致动作失控（denormalization 用错数据集统计量）；数据回放（replay）是最重要验证步骤；RTX 3050 Ti 推理 ~10s/chunk；Google Colab 免费训练 4h/22GB VRAM；方法论 > 硬件 |
| 7 | Evaluating π0 in the Wild: Strengths & Problems | Penn PAL Lab | 2025-12 | Debug, Arch | [链接](https://penn-pal-lab.github.io/Pi0-Experiment-in-the-Wild/) | 300+ trials，平均 task progress 42.3%；"Place can into purple box" 仅 16.7% 成功率；致命弱点：PaliGemma 语义理解弱（不认识陌生物体）、无记忆（memoryless → 多步任务失败）、OOD 物体导致 early stopping；Franka FR3 + Robotiq 2F-85 评估；vibe-checking 方法论暴露零样本泛化的真实上限 |
| 8 | Cosmos Policy for Advanced Robot Control | HF Blog (NVIDIA) | 2026-01-29 | Arch | [链接](https://huggingface.co/blog/nvidia/cosmos-policy-for-robot-control) | 基于 Cosmos Predict-2 WFM 后训练；LIBERO 98.5% SOTA（超 OpenVLA-OFT 97.1%、π0 94.2%）；RoboCasa 67.1%（仅 50 demos/task，超 π0 62.5%@300 demos）；动作/状态/价值统一编码为 latent frames → 单模型同时做 visuomotor control + world modeling + planning；planning 模式比 direct 高 12.5% 完成率；WFM 视频预训练 > VLM 图文预训练 |

---

## Discord 实战情报（LeRobot Discord, 16k+ 成员）

### D1. FAST tokenization 必须用 quantile normalization
- **来源**: `#vla-research` — **Ilya** (2026/2/4 13:50)，回复 @KWang 关于 Pi0.5/Pi FAST fine-tune 的发现
- **原文**: "FAST should definitely use quantile normalization (it is even covered in the original paper). Without it, because of outliers, many of the 'small' actions will be quantized to '0' and other 'central' tokens. Also, the pretrained FAST+ expects [-1,1] numbers. You can look at the FAST reconstruction loss (actions -> fast tokens -> actions and then compute MSE), and it is generally much better for quantile normalization. Not sure about the Diffusion/Flow matching case, there can be benefits as you described. Also, in many cases, the model can perform just fine without outliers, and removing them makes the tasks easier. Also, I would guess (without proof) that it works best for delta_actions, removing outliers from state or absolute actions restricts the policy from going to the edge states, and can impact performance if it is important."
- **类别**: Recipe, Debug
- **关键数据点**: FAST+ 期望 [-1,1] 范围；不用 quantile normalization → 小动作被量化为 "0"；delta_actions 去 outlier 可行，但 absolute actions 去 outlier 会限制 edge states

### D2. X-VLA 有效但 Pi-FAST 在自定义数据集上表现差
- **来源**: `#vla-research` — **eliasab** (2026/2/10 01:08)
- **原文**: "I meant on my own datasets. With xvla I was able to get some good results, but with pi fast, the arm just moves erratically"
- **类别**: Debug
- **关键数据点**: 同一数据集上 X-VLA 能出结果，但 Pi-FAST 动作完全失控（erratic）；可能与 normalization 或 tokenizer 配置有关

### D3. Pi0-FAST LIBERO checkpoint 87.5% SR
- **来源**: `#vla-research` — **Jade @HF LeRobot**（官方）(2026/2/9 17:33)
- **原文**: "This checkpoint of pi-fast you get around 87.5% sr on libero: https://huggingface.co/lerobot/pi0fast-libero"
- **类别**: Recipe
- **关键数据点**: 官方 pi0fast-libero checkpoint，LIBERO 87.5% success rate

### D4. π0.5 复现进展：annotation pipeline + RL extension 开放协作
- **来源**: `#vla-research` — **Jade @HF LeRobot**（频道开场帖）(2026/1/28 19:06)
- **原文**: "Current focus: full reproduction of pi05. Branch: [...] Annotation pipeline reliability: we've seen cases where the VLM annotator outputs incorrect subtask annotations. [...] pi05-full validation: we need to test pi05-full end-to-end in simulation (e.g., LIBERO) and real-world. Since the model expects subtasks as input, dataset correctness for training and evaluation is critical. [...] RL extension: this is still open work. Anyone interested in collaborating on RL for pi05 is welcome."
- **类别**: Arch, Debug
- **关键数据点**: VLM annotator 产出错误 subtask 标注是已知问题；pi05-full 需要 subtask 输入，数据集正确性关键；RL post-training 仍为开放问题

### D5. 3D Depth VLA 论文追踪（RealSense 团队）
- **来源**: `#vla-research` — **chrismatthieu** (2026/1/29 20:06)
- **原文**: "The RealSense team has been tracking 3D depth-powered VLA papers: PerAct, OG-VLA, DepthVLA, Spatial Forcing, Modality-Augmented Fine-Tuning – this is the paper we started implementing. https://locate3d.atmeta.com/demo"
- **类别**: Arch
- **关键数据点**: Modality-Augmented Fine-Tuning (Park et al., Dec 2025) 已开始实现；Depth-only VLA 效果不佳（Liu et al. 论文明确尝试过）

### D6. SmolVLA 固定位置 98% → 扩展多样性后成功率暴跌
- **来源**: `#general-chat`（Discord 搜索 "smolvla training"）— 匿名用户
- **原文**: "I recently built the SO-ARM101 and tried finetuning smoIVLA on a very simple pick and place task. The task was to pick a pen up and place it in a tray with the pen and tray always in the same positions, with the same rotations. I had great success with this and found my finetuned model could do the task with about a 98% success rate, but I think this gave me a false sense of achievement 😅. Once I tried to expand the training data by moving the pen around the desk and changing its rotation, I found the success rate plummeted quite dramatically. It definitely showed some understanding of the task and makes some plausible attempts but mostly fails."
- **类别**: Recipe, Debug
- **关键数据点**: SmolVLA on SO-ARM101，固定位置 98% SR → 加入位置/旋转多样性后暴跌；与 Sherry Chen 的 ACT 经验完全吻合（数据多样性是关键瓶颈）

### D7. SmolVLA eval 后夹爪电机死机 + 校准失败
- **来源**: `#help-general` (Help & Support) — **Tylp** (2026/3/13 23:51) 回复 + `#general-chat` — **Karthick** (2026/1/31 12:50)
- **Karthick 原文**: "Need help with my SO-101 follower arm, gripper motor stopped working during eval. Current Issue: Teleop and calibration now fail because gripper is frozen. Gripper motor won't move at all (even manually). Red LED still blinks when I move the wrist motor, even with power OFF and cable disconnected. What I did: ✅ Calibrated leader & follower arms - working fine. ✅ Teleop, ACT record/train/eval - all good. ✅ SmolVLA train - completed. ❌ SmolVLA eval - got calibration mismatch error -> then gripper motor stopped working"
- **Tylp 解法**: "check if all the motors are recognized using the [feetech debug software] (https://www.feetechrc.com/software.html). As for your second point, I ran into the issue as well and [this comment] (https://github.com/huggingface/lerobot/issues/1296#issuecomment-3383577176) helped me to resolve it"
- **类别**: Debug
- **关键数据点**: SmolVLA eval 触发 calibration mismatch → 夹爪电机彻底冻死（即使断电也无法手动转动）；解法：Feetech debug 软件检测 + GitHub issue #1296；与 Sherry Chen 的电机过力损坏经验一致

### D8. Calibration ValueError: Magnitude exceeds 2047
- **来源**: `#help-general` — **Simas** (2026/1/21 13:01) + `#help-forum` — **ojt___** (2026/1/17 04:06)
- **Simas 原文**: "When I set the leader arm in the pose required in the docs before calibration, it instantly fails, because some position values exceeds some set values in the script"
- **ojt___ 原文**: "Getting this error when trying leader calibration: ConnectionError: Failed to sync read 'Present_Position' on ids=[1, 2, 3, 4, 5, 6] after 1 tries. [TxRxResult] There is no status packet"
- **类别**: Debug
- **关键数据点**: SO-101 校准时 position 值超出硬编码上限 2047；sync read 超时是常见通信故障；多个用户重复报告同类问题

---

## GitHub Issues 周报

> 由 `scripts/en-vla-collector/github_vla_issue_collector.py` 在用户云端生成，
> 输出到 `reports/github_vla_issues_weekly.md`，通过 sync-vla-handbook 同步后可被读取。

---

*v1.1 — 2026-03-15 更新。HF Blog 8 条 + Discord 8 条（D1-D8），共 16 条高价值内容。覆盖 Recipe×5 / Debug×7 / Edge×1 / Arch×5 / Data×0 / Strategy×0。*
