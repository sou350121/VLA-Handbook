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
| 9 | Fine-tuning π0 with AMD ROCm and LeRobot | AMD ROCm Blog | 2025-07-14 | Recipe | [链接](https://rocm.blogs.amd.com/artificial-intelligence/rocm-lerobot/README.html) | AMD MI200 GPU 训练 → Ryzen AI PC（Phoenix）边缘部署；3B 参数 π0 仅需 50 条 20 秒轨迹即可微调 pick-and-place；Koch 双臂 leader-follower + 双 Logitech 摄像头 640×480@15fps；关键经验：位置多样化防过拟合、udev rules 固定 USB 设备映射、ROCm Docker 容器化部署流程；从数据中心训练到桌面推理的完整 pipeline |
| 10 | State of VLA Research at ICLR 2026 | Moritz Reuss Blog | 2025-10 | Arch, Strategy | [链接](https://mbreuss.github.io/blog_post_iclr_26_vla.html) | ICLR 2026 VLA 提交量 164 篇（前年 9 篇，18× 增长）；LIBERO 基本被解 >95% 是标配，不需要 VLA 也能达到；五大趋势：Discrete Diffusion VLA、Embodied Chain-of-Thought、新 Action Tokenizer（FASTer/OmniSAT）、高效 VLA（量化/蒸馏）、RL 微调；**关键洞察**：开源 VLA 在 sim benchmark 上接近/超过 π0.5，但零样本真实环境差距巨大（RoboArena 排行榜仅 Pi 模型有竞争力）；两个被严重忽视的方向：数据质量（OXE 质量差但无人量化）、in-context learning |
| 11 | π*0.6: A VLA that Learns from Experience | Physical Intelligence | 2025-11-17 | Arch, Strategy | [链接](https://www.pi.website/blog/pistar06) | 首个公开的 VLA + RL post-training 工业级结果；基于 π0.5 模型（5B VLM + action expert）；两种经验学习方式：coaching（专家纠正机器人错误轨迹）+ reinforcement（value function 做 credit assignment）；三个真实任务：espresso 制作（5:30am–11:30pm 运行）、50 种衣物折叠、纸箱组装；RL 后训练显著提升 throughput 和成功率；核心挑战：纯模仿学习的 compounding error → 需要 recovery 数据 + RL 修正 |
| 12 | Train ACT on SO-101 with LeRobot: Step-by-Step | Trelis Substack | 2026-02 | Recipe | [链接](https://trelis.substack.com/p/train-an-act-policy-for-an-so-101) | SO-101 + LeRobot ACT 端到端训练指南；推荐 50+ training examples，chunk_size=50，action_steps=15，batch_size=4-32；validation split 10%；关键建议：先在完全固定场景训练（same position/rotation），确认 work 后再扩展；固定光照和相机位置比 data augmentation 更重要；ensembling（temporal ensembling）在某些任务上反而降低准确率；大数据集才值得用 image augmentation |
| 13 | OpenVLA-OFT: Fine-Tuning Recipe for 26× Faster VLA | Stanford (Chelsea Finn Lab) | 2025-02 | Recipe, Arch | [链接](https://openvla-oft.github.io/) | 通过 parallel decoding + action chunking 实现 26× 加速、3× 低延迟；LIBERO 从 76.5%→97.1% SOTA；真机 ALOHA 双臂上超过 π0、RDT-1B、Diffusion Policy、ACT 最高 15%；**核心发现**：L1 regression 比 diffusion 在不完美数据上更鲁棒——diffusion 会精确复现 suboptimal 动作（如勺子插太深），L1 的有限表达力反而起到正则化效果，自动取 median mode；8×A100/H100 训练 1-2 天，50K-150K steps |
| 14 | Figure Helix: 首个全身人形 VLA | Figure AI | 2025-02-20 | Arch, Strategy | [链接](https://www.figure.ai/news/helix) | System 1 + System 2 架构：S2 = 7B VLM@7-9Hz（场景理解）+ S1 = 80M transformer@200Hz（低级控制），通过 latent vector 端到端训练通信；仅 ~500h 遥操数据（不到前人 VLA 数据集的 5%）；35-DoF 全上身控制（含手指）；双机器人协作零样本抓取任何家用物品；**部署关键**：S1/S2 分跑双嵌入式 GPU，训练时注入 temporal offset 模拟推理延迟以消除 train-inference gap；单套权重、无需 task-specific fine-tuning |
| 15 | ACTSmooth: 消除 ACT 在 SO-101 上的抖动 | Giacomo Moran Blog | 2026-03-10 | Recipe, Debug | [链接](https://www.giacomoran.com/blog/act-smooth/) | ACT 在真机上两大问题：推理延迟导致动作过期 + chunk 边界不连续导致抖动；ACTSmooth 方案：prefix conditioning（把上一个 chunk 的尾部动作作为下一个的输入）+ relative action representation + async inference；在 SO-101 + M2 Max MacBook 上实测：加速度均匀性从 170→115（降 32%），任务得分 1.8/2.0；relative actions 是关键（去掉后 smoothness 回退到 168）；past prefix 反而是负面结果（去掉后略微更smooth）；10FPS 下线性插值到 30FPS 命令率是必须的；代码开源 lerobot-policy-act-smooth |

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

### D9. Data augmentation 导致成功率归零
- **来源**: `#general-chat` — **Alli** (2025/11/10 08:20)
- **原文**: "I also tried enabling data augmentation but it brings 0 success rate so bad, increasing a batch size and number of train iterations also did not work and actually led to bad generalization"
- **关键数据点**: LeRobot 内置 data augmentation 对某些任务有害而非有益；增大 batch size 和训练轮数也无法补救；与直觉相反——更多增强 ≠ 更好泛化

### D10. 双臂任务在 SO-100 上极具挑战
- **来源**: `#general-chat` — **nicov** (2025/4/17 18:20)
- **原文**: "I used phosphobot for bimanual control with the meta quest app. Super simple setup. Works well for teleoperation. No issue with data collection. Training AI models is more challenging. Impressive bimanual tasks (eg: passing an object from one arm to another) are difficult for the so100 hardware (very precise tasks). Training models require at least 2x more data"
- **关键数据点**: Phosphobot + Meta Quest VR 遥操作简单好用、数据采集无问题；但 SO-100 双臂精密任务（如传递物体）训练极难；双臂任务所需数据量至少是单臂的 2×

### D11. Pick-and-place 需要多少 episodes？社区典型困惑
- **来源**: `#general-chat` — 匿名用户（Discord 搜索 "episodes needed"）
- **原文**: "I want to perform a pick-and-place task where I pick up an object from a table among other objects and place it into a box. I have a SO101 and want to perform it with the real robot. The workspace is approximately 60x60cm. Do you have any tips or know of any papers that discuss the best way to record and train for tasks like this? For example, for each object, how many episodes do I need to record? How many hours of training? And what are the best models currently capable of reproducing this? Is there any good result that isn't just overfitting?"
- **关键数据点**: 这代表社区最典型的困惑——pick-and-place 到底需要多少数据？结合 handbook 已有内容回答：ACT 50ep 可 work（§1 Sherry Chen）、SmolVLA 固定位置 98% 但多样性后暴跌（D6）、π0 微调 50 条可跑通但泛化有限

### D12. Recovery episodes 怎么录？社区讨论
- **来源**: `#show-us-what-you-built` — **eliasab** (2026/1/22 16:59) + `#general-chat` — **nic** (2024/12/3 17:22)
- **eliasab 原文**: "That's really interesting. I'm also currently working on the error recovery. So your recovery episodes start from the moment you take over? Or the entire episode from start to finish (including both the inference and teleop) is recorded?"
- **nic 原文**: "Has anyone tried to tune a model (ACT or DP) with recovery episodes? Meaning episodes you start in a failed state going to a successful state. For example when you try to grasp an object, start in a position where the gripper missed the object and then retries and grasps the object successfully. The question is more how much it would really improve my model"
- **类别**: Recipe, Debug
- **关键数据点**: Recovery episodes 的录制方式是开放问题——从失败状态开始录还是包含整个推理+接管过程？eliasab 探索两阶段架构（failure detection → recovery model）；nic 提出直接混入 recovery demos 训练 ACT/DP；参考 NXP blog（#3）发现 20% recovery episodes 显著提升成功率

### D13. LeRobot hackathon 经验：Mac 推理延迟是隐形杀手
- **来源**: `#hackathon-reinforcement-learning` — **Alexander Soare** (HF LeRobot 核心贡献者) (2024/10/28 19:16)
- **原文**: "Probs 8? Only 5 teams pushed through till the end though. Others decided to switch to imitation learning (I had warned them that results were not guaranteed). 2 teams had macs and there were latency/inference time issues, without which I'm sure they would have succeeded."
- **类别**: Debug, Edge
- **关键数据点**: 8 队参赛仅 5 队完成；部分队伍放弃 RL 转 imitation learning；**Mac 上的推理延迟问题导致至少 2 队失败**——这与 ACTSmooth blog（#15）的发现一致：即使 ~40ms 延迟也会导致可见抖动；对 Mac 用户的警告：推理延迟必须显式处理（latency matching 或 async inference），否则策略表面上训好了但真机跑不动

---

## GitHub Issues 周报

> 由 `scripts/en-vla-collector/github_vla_issue_collector.py` 在用户云端生成，
> 输出到 `reports/github_vla_issues_weekly.md`，通过 sync-vla-handbook 同步后可被读取。

---

*v1.3 — 2026-03-15 更新。Blog 15 条 + Discord 13 条（D1-D13），共 28 条。新增：Trelis ACT SO-101 训练指南、OpenVLA-OFT 微调配方（L1 > diffusion 的反直觉发现）、Figure Helix S1/S2 人形 VLA 架构、ACTSmooth prefix conditioning 消除 SO-101 抖动、recovery episodes 录制讨论、Mac 推理延迟 hackathon 教训。*
