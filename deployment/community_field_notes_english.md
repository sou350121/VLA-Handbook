# VLA/Embodied AI 英文社区实战笔记

> **版本**: v3.0 — 2026-03-15
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
| 16 | ML6 Field Report: ACT + GR00T-N1 实战 | ML6 Blog | 2025-12 | Recipe, Debug, Strategy | [链接](https://www.ml6.eu/en/blog/ai-robotics-a-field-report-on-imitation-learning-with-lerobot) | 比利时 ML6 团队在 SO-100 上测试 ACT + GR00T-N1；**ACT 结果**：10k frames/20ep→60% SR（训练时间不够），46k/100ep→90%（单轴），137k/340ep→79%（双轴但可泛化）；**GR00T-N1 结果**：pick-place 完全失败（缺乏精度），但布料折叠达 80%（复杂任务反而好）；VLA 推理延迟导致动作间明显卡顿；关键学习：数据精确度 > 数量、受控环境很重要、loss 不能反映真实成功率；hackathon 第3名（用 gaussian splatting 解决相机不稳定） |
| 17 | SmolVLA 在 SO-101 上的三轮数据迭代 | ggando Blog | 2026-03-01 | Recipe, Debug | [链接](https://ggando.com/blog/smolvla-so101/) | 三轮数据采集的深刻教训：v1(50ep/30cm)→手臂接近但抓空，v2(81ep)→引入"nudge trick"导致 20-80% 大方差（混合策略是毒药），v3(75ep/10cm/严格协议)→dual-cam 100% SR；**核心发现**：一致性 > 数量（75 clean > 81 mixed）、workspace 密度比总量重要（50ep@30cm 失败 vs 75ep@10cm 成功）；SmolVLA vs ACT 对比：同数据下 SmolVLA dual-cam 100% vs ACT 80%；RealSense 接 USB 2.0 会静默降速（用 lsusb -t 检查）；遥操延迟会降低 demo 质量（LeRobot record_loop 同步 IO 阻塞） |
| 18 | Phospho SmolVLA 官方训练指南 | Phospho Docs | 2026-02 | Recipe | [链接](https://docs.phospho.ai/learn/train-smolvla) | 官方推荐：~50 episodes 起步，20k steps 约 4h on A100；SmolVLA 是 base model 必须 fine-tune；uv 包管理器推荐（避免依赖冲突）；phosphobot 中间件简化 SO-100/SO-101 控制、数据录制、模型训练全流程；支持 Meta Quest VR 遥操；数据采集到部署的端到端教程 |
| 19 | SO-101 on Jetson AGX Orin: 边缘训练+推理 | Hackster.io (Shahizat) | 2025-07 | Edge, Recipe | [链接](https://www.hackster.io/shahizat/running-lerobot-so-101-arm-kit-using-nvidia-jetson-agx-orin-19b8a4) | Jetson AGX Orin 上端到端运行 LeRobot SO-101；Diffusion Policy 训练约 6h 完成；展示从硬件组装、校准、数据采集、训练到推理的完整 pipeline；Jetson GPU 同时用于训练和推理；边缘设备上的 imitation learning 可行性验证 |
| 20 | HIL-SERL: 1-2h 真机 RL 达到近完美成功率 | HF LeRobot Docs | 2025-10 | Arch, Recipe | [链接](https://huggingface.co/docs/lerobot/en/hilserl) | Human-in-the-Loop Sample Efficient RL；分布式 SAC learner + actor + 人类干预；reward classifier 自动检测成功/失败；1-2h 真机训练即可达到 100% success rate；与纯 IL 对比：IL 受限于 demo 质量上限，HIL-SERL 可超越人类 demo；LeRobot 集成支持；关键限制：需要人一直在旁边看着、干预 |
| 21 | LearnOpenCV VLA 全景 + LeRobot Policy 教程 | LearnOpenCV | 2025-04 | Arch, Strategy | [链接](https://learnopencv.com/vision-language-action-models-lerobot-policy/) | VLA 架构全景：dual-expert 路线（NVIDIA GR00T N1 + Figure Helix）vs generalist 路线（π0）；LeRobot 框架实现 IL/RL/VLA 三类 policy；SmolVLA 特色：社区数据驱动预训练；对初学者友好的 VLA 入门资源 |
| 22 | One-Step Diffusion Policy: 1.5Hz→62Hz 加速 | NVIDIA Research (ICML 2025) | 2025-10 | Arch, Edge | [链接](https://research.nvidia.com/labs/dir/onedp/) | 蒸馏多步 diffusion 为单步生成器；推理频率从 1.5Hz→62Hz（41× 加速）；仅需 2-10% 额外预训练成本；KL divergence 沿 diffusion chain 最小化；6 个仿真 + 4 个 Franka 真机任务验证；解决 diffusion policy 的核心部署瓶颈——迭代去噪太慢 |
| 23 | 2025 Embodied AI Hackathon 回顾 | Seeed Studio Blog | 2025-11-06 | Strategy, Recipe | [链接](https://www.seeedstudio.com/blog/2025/11/06/2025-embodied-ai-hackathon-recap-we-built-home-cooking-robot/) | Seeed Studio + LeRobot + NVIDIA 联合 hackathon；冠军"Matcha Bot"用双 SO-101 臂自动制作抹茶（GR00T N1.5 + Jetson Thor）；参赛队伍横跨 arms/exoskeleton/自由赛道；100h 真实数据 + 100% 成功率挑战赛道；硬件：20 YAM + 20 SO-101 提供给参赛者；展示社区从 hobby 到应用的转型 |
| 24 | GenAI for Robotics: SmolVLA 仿真微调 | Medium (Henry Hu) | 2026-01 | Recipe, Debug | [链接](https://medium.com/@henryhu1607/genai-for-robotics-fine-tuning-smolvla-to-pick-and-place-940b485e6c9b) | SmolVLA 450M 在 Franka Panda 仿真 pick_cube_rl 数据集（25 episodes）上微调；Google Colab A100 约 4h；结果：未能成功抓取，但行为"戏剧性地"改善——手臂主动朝 cube 移动并尝试抓取；25 episodes 不足以学会完整任务；SmolVLA 降低 VLA 入门门槛（学术实验室/个人也可以跑） |
| 25 | 12 Predictions for Embodied AI 2026 | Dylan Bourgeois Blog | 2026-01 | Strategy | [链接](https://dtsbourg.me/en/articles/predictions-embodied-ai) | 12 项预测含：VLA 将成为标配但 sim benchmark 会被解（LIBERO >95% 已不够）；数据飞轮比模型架构更重要；RL post-training 将成为标准流程；边缘部署仍是瓶颈（95%/step → 60%@10步链式任务）；开源 vs 闭源差距在缩小但真机泛化仍差；触觉传感将成为下一个 frontier |
| 26 | NVIDIA Isaac Lab 2.3: 遥操数据采集新标准 | NVIDIA Developer Blog | 2025-12 | Data, Recipe | [链接](https://developer.nvidia.com/blog/streamline-robot-learning-with-whole-body-control-and-enhanced-teleoperation-in-nvidia-isaac-lab-2-3/) | Isaac Lab 2.3 扩展遥操设备支持：Meta Quest VR、Manus 手套；SpaceMouse 比键盘产出更平滑的 demo；全身控制支持人形机器人数据采集；与 LeRobot 集成打通 sim→real pipeline；数据采集效率和质量是 imitation learning 最关键的环节 |
| 27 | AWS Embodied AI: 从边缘到云的 Physical AI | AWS Open Source Blog | 2026-02 | Arch, Edge, Strategy | [链接](https://aws.amazon.com/blogs/opensource/building-intelligent-physical-ai-from-edge-to-cloud-with-strands-agents-bedrock-agentcore-claude-4-5-nvidia-gr00t-and-hugging-face-lerobot/) | Strands Agents + Bedrock + Claude 4.5 + GR00T + LeRobot 集成；GR00T 在 Jetson 边缘硬件上运行控制机械臂；展示 LLM agent 编排 + VLA 执行的分层架构；从云端推理到边缘部署的完整 physical AI stack |
| 28 | Jetson Thor + GR00T N1.5 部署 SO-101 全流程 | Seeed Studio Wiki | 2026-01 | Edge, Recipe | [链接](https://wiki.seeedstudio.com/fine_tune_gr00t_n1.5_for_lerobot_so_arm_and_deploy_on_jetson_thor/) | GR00T N1.5 在 Jetson Thor 上的端到端部署指南；从 SO-101 数据采集 → 微调 → Jetson Thor 部署；Jetson Thor 1200 FP4 TFLOPS/64GB 内存，前代 Orin 2× 性能；社区首个 Jetson Thor + VLA 完整教程；与 hackathon 冠军"Matcha Bot"使用相同技术栈 |
| 29 | VLA-0-Smol: 500M 参数复现大模型级 LIBERO 性能 | Robot Learning Collective | 2025-12 | Recipe, Arch | [链接](https://robot-learning-collective.github.io/vla-0-smol) | 500M 参数（SmolVLM2-500M）在 LIBERO 上达到 94.1% 平均 SR，接近 VLA-0 3B 的 94.7%；系统性 ablation 揭示训练 VLA 的优先级：**LR 是生死线**（5e-6 完全不学，5e-5 最优）、**必须微调 vision encoder**（冻结掉 32 个百分点）、**whole-action masking +7.8%**、相对动作 > 绝对动作；system prompt 在微调后无用；float16 梯度爆炸必须用 bf16；LIBERO 高分主要测记忆力非泛化（LIBERO-Plus/PRO 证实）；消费级 GPU 即可训练 |
| 30 | Behind a Failed Robot Learning Project: 数据工程的教训 | Medium (CU Correll Lab) | 2026-01-21 | Recipe, Debug | [链接](https://medium.com/correll-lab/fine-tuning-smolvla-for-new-environments-code-included-af266c56d632) | SO-100 上 SmolVLA 微调失败复盘；**stats.json 不匹配是致命陷阱**——denormalization 用错数据集统计量导致动作完全失控；数据回放（replay）是最重要的验证步骤；相机视角对策略表现影响巨大；60 条 demo 不足以泛化；LeRobot V3 数据格式需要注意兼容性；RTX 3050 Ti 推理 ~10s/chunk；方法论 > 硬件 |
| 31 | Decoding SmolVLA: 架构设计深度拆解 | Phospho Blog | 2026-02 | Arch | [链接](https://docs.phospho.ai/blog/decoding-smolvla) | SmolVLA 内部机制详解：pixel-shuffling 将每帧压缩到 64 tokens（vs PaliGemma 的 256+）；layer skipping 实现交错 CA+SA 降低计算量；flow matching 动作生成 vs 回归——flow matching 处理多模态动作分布更好；异步推理比同步快 30%（解耦 VLM 编码和动作生成）；450M 参数设计哲学：在 VLM backbone 和 action expert 之间找平衡 |
| 32 | GR00T N1.6 + Cosmos-Reason: Sim-to-Real 零样本 | NVIDIA Developer Blog | 2026-03 | Arch, Recipe | [链接](https://developer.nvidia.com/blog/gr00t-n1-6-sim-to-real/) | GR00T N1.6 核心升级：Cosmos-Reason-2B VLM 替代之前的视觉编码器；DiT 扩大 2× 至 32 层；state-relative actions 替代绝对动作；全身 RL 在 Isaac Lab 中训练；COMPASS 导航模块支持自主导航；**关键突破**：零样本 sim-to-real 迁移——仿真训练直接部署到真机无需额外微调；展示从感知到控制的完整 physical AI stack |
| 33 | Isaac Lab-Arena: 40× 加速的模块化评估框架 | NVIDIA Research | 2026-02 | Arch, Data | [链接](https://developer.nvidia.com/blog/isaac-lab-arena/) | 模块化任务构建：Object + Scene + Embodiment + Task 四维组合；并行评估 40× 加速（0.76h vs 34.9h）；250+ Lightwheel 预置任务；与 LeRobot Environment Hub 集成；支持 ACT/DP/VLA 等多种策略统一评估；解决 VLA 社区缺乏标准化评估的痛点 |
| 34 | X-VLA: 首个软提示跨具身 VLA（ICLR 2026 冠军） | HF LeRobot Docs | 2026-02 | Arch, Recipe | [链接](https://huggingface.co/docs/lerobot/en/xvla) | 0.9B 参数，ICLR 2026 最佳论文；核心创新：domain_id 系统为每个机器人注入 soft prompt，实现跨具身迁移；自动 action mode 检测（delta vs absolute）；LIBERO 93%、布料折叠 100%；Florence-2 视觉编码器 + flow matching action expert；训练只需冻结部分层 + LoRA，单卡可跑 |
| 35 | Robotics Made Simple: SO-101 新手实战 | Val Kamenski Blog | 2025-08-17 | Recipe, Debug | [链接](https://www.kamenski.me/articles/robotics-made-simple-playing-with-lerobot-and-so-101) | 新手向 SO-101 完整上手指南；**硬件建议**：RTX 4090+ 是长期投资（GR00T/SmolVLA 需要 CUDA，Mac/Colab 不够用）；Ubuntu 24.04 裸机安装问题最少；USB 端口需 `sudo chmod 666`（文档容易漏看）；组装陷阱：夹爪装反 → 校准失败；**关键心态**：别期望 VLA 开箱即用——必须先训练；遥操录数据时不要直接看 follower 臂，通过摄像头观察；SO-101 成本 $200-400 |
| 36 | EmbodiFlow: Pi0/Pi0.5 微调可视化平台 | EmbodiFlow (io-ai.tech) | 2026-01 | Recipe | [链接](https://io-ai.tech/platform/en/guides/Pipeline/LeRobot/Pi0/) | 基于 OpenPI 框架的 Pi0/Pi0.5 微调指南；可视化数据管理 + 一键导出 LeRobot 格式；支持自定义机器人和任务的数据导入；降低 Pi0 微调的工程门槛——不需要从零搭建 OpenPI 环境 |
| 37 | ICRA 2026 Workshop: VLA Pipelines for Real Robots | ICRA 2026 | 2026-06 | Strategy | [链接](https://icra2026vlapipeline.github.io/) | ICRA 2026 专题研讨会聚焦 VLA 从仿真到真机的 pipeline 完整性；核心议题：数据引擎（自动采集+质量控制）、domain randomization、human-in-the-loop、标准化评估；反映学术界共识——VLA 的瓶颈已从模型架构转向工程 pipeline |
| 38 | LingBot-VLA: 国产开源"通用大脑"模型 | Robotics & Automation News | 2026-03-13 | Arch, Strategy | [链接](https://roboticsandautomationnews.com/2026/03/13/robbyant-open-sources-lingbot-vla-model-as-a-universal-brain-for-robots/99640/) | Robbyant（灵蚁智能）开源 LingBot-VLA；定位为机器人"通用大脑"；代表中国 VLA 产业化新动态——开源路线与闭源（Physical Intelligence）形成对比；需要关注其在真机上的实测表现 |
| 39 | LoRA VLA: RTX 4060 8GB 上训练 3.1B VLA | arXiv 2512.11921 | 2025-12 | Recipe, Edge | [链接](https://arxiv.org/abs/2512.11921) | 3.1B 参数 VLA 通过 LoRA rank=8 + 4-bit 量化在 8GB VRAM 上训练；仅需 200 条 demo；SO-101 按钮按压任务验证；证明消费级 GPU（RTX 4060）也能参与 VLA 研发；对资源受限的个人/小团队极有价值 |
| 40 | Phosphobot: 一站式 VLA 开发中间件 | Phospho | 2026-02 | Recipe | [链接](https://docs.phospho.ai/learn/train-smolvla) | pip install 一键安装；Meta Quest VR 遥操作；支持云端 GPU 训练；LeRobot 格式原生支持；从数据采集到模型部署的完整 pipeline；显著降低 SO-100/SO-101 用户的入门门槛——不需要写数据管道代码 |

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
- **类别**: Recipe, Debug
- **关键数据点**: LeRobot 内置 data augmentation 对某些任务有害而非有益；增大 batch size 和训练轮数也无法补救；与直觉相反——更多增强 ≠ 更好泛化

### D10. 双臂任务在 SO-100 上极具挑战
- **来源**: `#general-chat` — **nicov** (2025/4/17 18:20)
- **原文**: "I used phosphobot for bimanual control with the meta quest app. Super simple setup. Works well for teleoperation. No issue with data collection. Training AI models is more challenging. Impressive bimanual tasks (eg: passing an object from one arm to another) are difficult for the so100 hardware (very precise tasks). Training models require at least 2x more data"
- **类别**: Recipe, Data
- **关键数据点**: Phosphobot + Meta Quest VR 遥操作简单好用、数据采集无问题；但 SO-100 双臂精密任务（如传递物体）训练极难；双臂任务所需数据量至少是单臂的 2×

### D11. Pick-and-place 需要多少 episodes？社区典型困惑
- **来源**: `#general-chat` — 匿名用户（Discord 搜索 "episodes needed"）
- **原文**: "I want to perform a pick-and-place task where I pick up an object from a table among other objects and place it into a box. I have a SO101 and want to perform it with the real robot. The workspace is approximately 60x60cm. Do you have any tips or know of any papers that discuss the best way to record and train for tasks like this? For example, for each object, how many episodes do I need to record? How many hours of training? And what are the best models currently capable of reproducing this? Is there any good result that isn't just overfitting?"
- **类别**: Recipe
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

### D14. vast.ai 训练 SmolVLA 10/10 成功但无法泛化
- **来源**: `#general-chat` — **Psychonautic** (2026/1/19 23:37)
- **原文**: "Thanks for the advice! I have a leader arm and did exactly as you said; I collected data on a simple pick and place task with a pen and tray. I then used vast.ai to finetune smolvla on the data. It performed the task with a 10/10 success rate! However, since the task was very simple and narrow (pen and tray always in the same place), it of course struggles when there are any changes to the environment. So I'm wondering what the next step is to make something a little more general."
- **类别**: Recipe, Debug
- **关键数据点**: vast.ai 云端训练 SmolVLA 可行；固定场景 10/10 完美；但任何环境变化就失败；与 D6（SmolVLA 98%→暴跌）和 #17（ggando 三轮迭代）完全一致——泛化是所有人的核心瓶颈

### D15. Pi0.5 复现失败：4×H200 仅达 55% vs 官方 95%
- **来源**: `#vla-models` — 匿名用户 (2026/1 左右)
- **原文**: "I attempted to reproduce the results for lerobot/pi05_libero_finetuned by following the provided recipe. Using 4x H200 GPUs, I fine-tuned lerobot/pi05_libero_base with a batch size of 64 for 6k steps. However, the Success Rate (SR) on the libero_10 evaluation was only around 55%. In contrast, when I evaluate the official lerobot/pi05_libero finetuned checkpoint directly, it achieves an SR of approximately 95%. Why am I unable to reproduce the original performance? I used the HuggingFaceVLA/libero dataset for this run."
- **类别**: Debug
- **关键数据点**: 4×H200、batch 64、6k steps 仍只有 55% SR（官方 95%）；可能原因：数据集版本不匹配、超参差异、评估协议不同；**复现 VLA 结果比想象中难得多**——硬件够了不代表结果能对齐

### D16. Pi0.5 人形机器人按电梯按钮：~100 样本微调
- **来源**: `#vla-models` — **Tahsincan Kose** (2026/1/8 20:42)
- **原文**: "Hello everyone. I've finetuned Pi0.5 base on a single-task dataset for a custom humanoid robot on elevator button-pushing task. So my finetuning dataset contains ~100 samples from real-robot..."
- **类别**: Recipe, Arch
- **关键数据点**: Pi0.5 base 在非标准人形机器人上微调；仅 ~100 条真机数据用于电梯按钮任务；展示 VLA 在人形场景的迁移潜力；单任务 fine-tune 数据需求与桌面臂类似

### D17. RealSense 深度数据集成：编码到 RGB 通道的 workaround
- **来源**: `#help-general` — **dsvschvhm** (2025/11/5 09:56)
- **原文**: "I used to try this method: depth camera is Intel Realsense D415, which casts depth data into mm. With my little workspace, I think 16bit is enough so I embed depth data into R and G channel and save them to image. Then saving the raw images after encoding to video. For the loss in image<->video, finally I can only use raw image to get real depth data. This is not a good way, I tried to save depth as other numberic data to .parquet file but it may lead to a very very large file."
- **类别**: Debug, Data
- **关键数据点**: LeRobot 目前不原生支持深度数据；社区 workaround：16bit depth 编码到 R+G 通道存为 image，但 video 压缩会损失精度；parquet 存数值可行但文件巨大；深度集成仍是开放问题

### D18. SmolVLA 颜色泛化测试：抓取 OK 但放置退化
- **来源**: ggando blog 补充数据（[#17](https://ggando.com/blog/smolvla-so101/) 延伸）
- **原文**: "I ran a quick color generalization test with the SmolVLA dual-cam model, since it was trained exclusively on a red cube. Orange: Succeeded but hit the bowl on the way back (rare with the red cube). Blue: Grasped but dropped on the way to the bowl. Green: Grasped and moved toward the bowl but hit it with the gripper, failed. 1/3 success vs 5/5 with the training color. Grasping generalizes across colors reasonably well. It did pick up all three, but the place trajectory degrades. The model seems to have learned color-specific visual features for the red cube rather than a general 'cube' concept. The placing phase is somehow more sensitive. I didn't change the prompt to specify the cube color, so this is purely testing whether the vision backbone generalizes. It partially does for grasping but not for the full task. I'd probably need to add cube color variations in the dataset if I wanted to be robust w.r.t. visual appearance of the cube."
- **类别**: Debug, Recipe
- **关键数据点**: VLA 的视觉泛化是部分的——抓取动作可以跨颜色泛化，但放置轨迹严重依赖训练颜色；需要在数据集中加入颜色变化才能鲁棒；对"VLA 理解语言就能泛化"的期望需要校准

### D19. GR00T-N1 布料操作：复杂任务反而比 pick-place 好
- **来源**: ML6 blog 补充数据（[#16](https://www.ml6.eu/en/blog/ai-robotics-a-field-report-on-imitation-learning-with-lerobot) 延伸）
- **原文**: "We found that GR00T-N1 had surprisingly different success rates across tasks. For the pick-and-place task, it completely failed — the model lacked the precision needed for accurate grasping and placing. However, for cloth folding, it achieved around 80% success rate. The model showed strong global task awareness and would keep trying when it recognized the task wasn't complete, but it lacked fine-grained precision and sub-task awareness. This is a counterintuitive finding: a 'harder' task (cloth folding) worked better than a 'simpler' one (pick-place), possibly because the pretrained model had better priors for deformable object manipulation. The inference latency of the VLA also caused noticeable stuttering between actions."
- **类别**: Debug, Arch
- **关键数据点**: VLA foundation model 的反直觉现象——精确 pick-place 不如"模糊"布料操作；可能因为预训练数据中布料操作有更好的 prior；对 VLA 选型的启示：不是所有任务都适合同一个模型

### D20. ACT vs SmolVLA 训练效率对比
- **来源**: ggando blog 补充数据（[#17](https://ggando.com/blog/smolvla-so101/) 延伸）
- **原文**: "I trained all three variants on the same v3 dataset (75 episodes) for 20k steps each. SmolVLA (dual cam): final loss 0.005, grad norm 0.11, ~10.4h training, ~1.7B params, 100% success rate (5/5). ACT (dual cam): final loss 0.052, grad norm 3.92, ~10.5h training, 52M params, 80% success rate (4/5). SmolVLA converges to ~10x lower loss and ~35x lower gradient norms, which is expected given the pretrained VLM backbone vs ACT training from scratch with only ResNet18 features. ACT matching SmolVLA wrist-only at 80% is notable given it's 52M params trained from scratch vs ~1.7B fine-tuned. For the v3 ACT run I also added an aspect-ratio-preserving resize to 224×224 (which the earlier v1 ACT run was missing), so it could finally run at batch_size=64 without OOM. ACT has no built-in image resize — 640×480 frames produced 602 encoder tokens per forward pass, OOM'd at batch_size=64 on 24GB VRAM."
- **类别**: Recipe, Arch
- **关键数据点**: SmolVLA 预训练 backbone 优势明显：同数据同时间，loss 低 10×、gradient 稳定 35×；ACT 需要手动处理图像分辨率否则 OOM；但 ACT 52M 从零训到 80% 也说明小模型的竞争力

### D21. 遥操数据质量 > 数量：一致策略胜过混合技巧
- **来源**: 多源汇总 — ggando blog [#17](https://ggando.com/blog/smolvla-so101/)、ML6 field report [#16](https://www.ml6.eu/en/blog/ai-robotics-a-field-report-on-imitation-learning-with-lerobot)、Sherry Chen HF blog [#5](https://huggingface.co/blog/sherryxychen/train-act-on-so-101)
- **原文摘要**:
  - ggando: "Lesson learned: consistency in demonstrations matters more than quantity. One clean strategy beats a mix of tricks. I developed a habit of nudging the cube with the gripper's static finger to rotate it into a better angle before grasping. Seemed clever at the time. The policy learned a conditional behavior: sometimes nudge, sometimes grasp directly. It couldn't consistently decide which to do."
  - ML6: "Data accuracy is more important than quantity. 10k frames with controlled, sequential movements taught more than 137k frames with varied execution styles."
  - Sherry Chen: 3 rounds of iteration — Try1 50ep→woodpecker behavior, Try2 72ep→60% in-distribution, Try3 125ep+rotation diversity→90% ID/75% OOD.
- **类别**: Recipe, Data
- **关键数据点汇总**: 三个独立团队的共识——imitation learning 的数据采集应遵循"一种策略做到底"原则，先在窄场景验证，再有计划地扩展。混合策略是毒药。

### D22. VLA 推理延迟全景：从 40ms 到 10s+
- **来源**: 多源汇总 — ACTSmooth blog [#15](https://www.giacomoran.com/blog/act-smooth/)、NXP blog [#3](https://huggingface.co/blog/nxp/bringing-robotics-ai-to-embedded-platforms)、Penn PAL [#7](https://penn-pal-lab.github.io/Pi0-Experiment-in-the-Wild/)、ML6 [#16](https://www.ml6.eu/en/blog/ai-robotics-a-field-report-on-imitation-learning-with-lerobot)、OneDP [#22](https://research.nvidia.com/labs/dir/onedp/)
- **原文摘要**:
  - ACTSmooth: "ACT inference on M2 Max MacBook ~30ms mean, 40ms 95th percentile. At 30fps (33ms per frame), this corresponds to roughly two timesteps of inference delay." — even this small delay causes visible jerkiness.
  - NXP: "ACT ONNX FP32 on i.MX 95: 2.86s → optimized 0.32s. SmolVLA ONNX FP32: 29.1s → optimized 6.15s." — quantizing action expert (flow matching) severely degrades accuracy.
  - ML6: "GR00T-N1 inference latency caused noticeable stuttering between actions during real robot evaluation."
  - ggando: "SmolVLA on RTX 3050 Ti inference ~10s/chunk."
  - OneDP: "Distilled one-step diffusion: 1.5Hz → 62Hz (41× speedup)."
- **类别**: Edge, Arch
- **关键数据点汇总**: ACT ~40ms（仍需 ACTSmooth）；SmolVLA ~10s/chunk on 3050 Ti；NXP i.MX95 ACT 优化后 0.32s；OneDP 蒸馏后 62Hz；**规律**：模型越大延迟越高，但 async inference + action chunking 是通用解药；边缘部署必须把延迟作为第一优先级

### D23. GR00T vs X-VLA vs ACT 对比：社区实测经验
- **来源**: `#vla-models` — **Don** (2026/2 左右)，Discord 搜索 "gr00t training"
- **原文**: 社区用户 Don 对比三个模型在自定义任务上的表现，发现 GR00T 在预训练覆盖的任务上有优势但自定义任务微调困难，X-VLA 凭 soft prompt 在新 embodiment 上表现更灵活，ACT 虽小但在数据充足时仍有竞争力。
- **类别**: Arch, Recipe
- **关键数据点**: 三模型各有最佳适用场景——GR00T 适合预训练覆盖的任务域、X-VLA 适合跨 embodiment 迁移、ACT 适合数据充足的单任务；没有"万能模型"

### D24. GR00T 训练数据量经验：90/140/450 episodes 实测
- **来源**: `#vla-models` — 多个用户 (2026/1-2)，Discord 搜索 "gr00t training"
- **原文**: 社区多人分享 GR00T 微调数据量经验："90 episodes 能跑但抖动严重"、"140 episodes 明显改善"、"450 episodes 接近稳定"。共识是 2-300 episodes 是最优投入产出比区间——低于 100 条质量差，超过 300 条提升不明显。
- **类别**: Recipe, Data
- **关键数据点**: GR00T 微调最优区间 200-300 episodes；低于 100 条动作抖动严重；与 SmolVLA/ACT 的 50-125 条相比，GR00T 因模型更大需要更多数据

### D25. GR00T 数据格式 v3.0 → v2.1 转换问题
- **来源**: `#vla-models` — **andrewr96** (2026/2)，Discord 搜索 "gr00t training"
- **原文**: "I collected data with the latest LeRobot which uses dataset format v3.0, but GR00T fine-tuning scripts expect v2.1. The conversion isn't straightforward and the documentation doesn't cover this gap."
- **类别**: Debug, Data
- **关键数据点**: LeRobot v3.0 数据格式与 GR00T 微调脚本（期望 v2.1）不兼容；社区工具 Forge（见 D28）可解决此类格式转换问题

### D26. chunk_size vs action_steps 参数困惑
- **来源**: `#help-general` — **nic** (2026/1)，Discord 搜索 "chunk_size action_steps"
- **原文**: "I'm confused about the difference between chunk_size and action_steps in the LeRobot config. Are they the same thing? When should I change which one?"
- **类别**: Debug, Recipe
- **关键数据点**: LeRobot 中 chunk_size（模型一次预测多少步）和 action_steps（实际执行多少步再重新预测）是两个不同概念；action_steps ≤ chunk_size；较小的 action_steps 意味着更频繁重新规划（更稳但更慢）；社区常见误区是把两者混为一谈

### D27. Diffusion Policy 训练时间：RTX 4090 上 50k steps 即可
- **来源**: `#alex-koch-arm` — **Xingdong Zuo** (2025/1/6)，Discord 搜索 "diffusion policy steps"
- **原文**: "Diffusion I've heard is slow, but haven't tested. ppl in community come up with quite some ways to speedup diffusion policies (e.g. 1-step DP). [...] 200k steps to reach its max potential, which takes 3.2 hours on my 4090 gpu. Although it does pretty well with just 50k steps which is only 50 minutes."
- **类别**: Recipe, Edge
- **关键数据点**: Diffusion Policy 在 RTX 4090 上 200k steps 需 3.2h 达到最大性能，但 **50k steps（50 分钟）已经 pretty well**——对快速原型验证很有价值；社区已有 1-step DP 加速方案（参考 #22 OneDP）

### D28. Forge: 机器人数据集万能格式转换器
- **来源**: `#datasets` — **Arpit** (2026/1/25-27)，Discord 搜索 "dataset format v2"
- **原文**: "Hub-and-spoke architecture: All formats normalize to Episode/Frame intermediate representation. Adding a reader unlocks all writers (and vice versa) — no N×M conversion logic. Read: RLDS (Open-X/TFDS), LeRobot v2/v3, Zarr (Diffusion Policy/UMI), HDF5, Rosbag."
- **类别**: Data
- **关键数据点**: `arpitg1304/forge` — 开源的机器人数据集格式转换器；支持 RLDS ↔ LeRobot v2/v3 ↔ Zarr ↔ HDF5 ↔ Rosbag 双向转换；hub-and-spoke 架构意味着新增一种格式只需写一个 reader/writer；解决社区最大痛点之一——数据格式碎片化

### D29. LoRA 微调 SmolVLA 的 GPU 需求
- **来源**: `#vla-models` — **pangu** (2025/10/16) + **Inish** (2025/7/30) + **Jiabin** (2025/3/23)，Discord 搜索 "LoRA fine-tune"
- **原文**:
  - pangu: "Do you know the script to do LORA fine-tuning? I don't see it mentions in the Lerobot doc"
  - Inish: "Hi! has anyone done LoRA fine tuning of smolVLA"
  - Jiabin: "Hi Xiaoxuan, just to confirm with you on the GPU, did you LoRA fine tune on a 4090 with 24G VRAM?"
- **类别**: Recipe, Edge
- **关键数据点**: 社区对 LoRA 微调 VLA 有强烈需求但文档缺失；4090 24GB VRAM 是 LoRA 微调的可行配置（参考 #39 LoRA VLA 论文验证 8GB 也可行）；LeRobot v0.5.0 已官方支持 PEFT/LoRA（#4）

### D30. LeIsaac Pipeline: GR00T + Isaac Lab 端到端训练
- **来源**: `#vla-models` — **zeyu.hu** (Lightwheel/NVIDIA) (2026/2)，Discord 搜索 "gr00t training"
- **原文**: zeyu.hu 分享 LeIsaac pipeline 将 Isaac Lab 仿真环境与 GR00T 训练直接打通——仿真中生成的 episode 自动转为 GR00T 训练格式，减少数据管道的手动操作。Lightwheel 贡献了 250+ 预置任务。
- **类别**: Data, Recipe
- **关键数据点**: LeIsaac = Isaac Lab → GR00T 的自动化数据管道；与 Isaac Lab-Arena（#33）配合使用；250+ 任务库降低了仿真数据生成门槛

### D31. GR1 推理代码 Bug + Action Chunking 改造
- **来源**: `#general-chat` — **Zhuoheng Li** (2024/5/20)，Discord 搜索 "diffusion policy steps"
- **原文**: "I just found a bug in the original GR1 inference code. Fixing the bug may slightly raise the success rate. I am also modifying it into a policy that predicts an 'action chunk', just like ACT. Based on that, I will modify it to a diffusion policy step by step."
- **类别**: Debug, Arch
- **关键数据点**: GR1 官方推理代码有 bug（影响成功率）；社区开发者将 GR1 改造为 action chunking + diffusion policy 混合架构；说明开源模型的代码质量值得审计——不能假设官方代码无 bug

### D32. SmolVLA 微调后泛化瓶颈：社区共识
- **来源**: 多源汇总 — D6（SmolVLA 98%→暴跌）、D14（vast.ai 10/10 但无法泛化）、#17（ggando 三轮迭代）、#30（Correll Lab 失败）、#29（VLA-0-Smol LIBERO 记忆 vs 泛化）
- **原文摘要**:
  - D6: "Fixed position 98% SR → expanded diversity → plummeted dramatically"
  - D14: "vast.ai 10/10 success but struggles with any changes to the environment"
  - ggando: "75 clean eps@10cm workspace → 100%, but expanding workspace fails"
  - VLA-0-Smol: "LIBERO-Plus found models often just memorizing training data"
- **类别**: Debug, Strategy
- **关键数据点汇总**: **泛化是 VLA 社区的 #1 瓶颈**。5+ 个独立团队报告相同模式：固定场景高成功率（90-100%）→ 任何变化就崩溃。根因不是模型架构而是数据覆盖度——但扩展数据时混合策略又是毒药。目前最佳实践：先窄场景验证 → 有计划地逐步扩展 workspace → 保持数据采集策略一致。

### D33. VLA 训练设计选择优先级汇总
- **来源**: 多源汇总 — VLA-0-Smol [#29](https://robot-learning-collective.github.io/vla-0-smol)、SmolVLA blog [#1](https://huggingface.co/blog/smolvla)、ggando [#17](https://ggando.com/blog/smolvla-so101/)、Correll Lab [#30](https://medium.com/correll-lab/fine-tuning-smolvla-for-new-environments-code-included-af266c56d632)
- **原文摘要**:
  - VLA-0-Smol: "Learning rate has a dramatic impact — 5e-6 and 1e-5 completely failed (0% SR). Freezing vision encoder cuts performance by 32 percentage points. Whole-action masking +7.8%. Relative actions > absolute. System prompts don't matter after fine-tuning."
  - SmolVLA: "Async inference 30% faster. Community pretrain data +26.6%."
  - ggando: "Consistency > quantity. 75 clean episodes > 81 mixed."
  - Correll Lab: "stats.json mismatch → total action failure. Replay is the most important verification step."
- **类别**: Recipe, Arch
- **关键数据点汇总**: VLA 训练决策的优先级层次——**致命级**：学习率（错了就 0%）、vision encoder 微调（必须）、数据一致性（混合策略是毒药）；**重要级**：action masking、相对动作、异步推理；**可忽略**：system prompt、temporal ensembling（简单任务）。这是社区 10+ 团队实战经验的共识。

### D34. 硬件投资指南：社区实测 GPU 需求矩阵
- **来源**: 多源汇总 — Val Kamenski [#35](https://www.kamenski.me/articles/robotics-made-simple-playing-with-lerobot-and-so-101)、LoRA VLA [#39](https://arxiv.org/abs/2512.11921)、VLA-0-Smol [#29](https://robot-learning-collective.github.io/vla-0-smol)、Sherry Chen [#5](https://huggingface.co/blog/sherryxychen/train-act-on-so-101)、Henry Hu [#24](https://medium.com/@henryhu1607/genai-for-robotics-fine-tuning-smolvla-to-pick-and-place-940b485e6c9b)
- **原文摘要**:
  - Val Kamenski: "RTX 4090 or better will save you a lot of frustration. Most libraries only support CUDA."
  - LoRA VLA: LoRA rank=8 + 4-bit 量化 → 3.1B VLA 在 8GB VRAM 上可训练
  - VLA-0-Smol: 500M 模型在消费级 GPU 上完成全部 ablation
  - Sherry Chen: ACT 52M 在 RTX 3080 12GB 上 4h 训练完成
  - Henry Hu: Google Colab A100 约 4h 训练 SmolVLA
- **类别**: Edge, Recipe
- **关键数据点汇总**: **ACT（52M）**→ RTX 3080 12GB 足够（4h）；**SmolVLA（450M）**→ A100 或 RTX 4090 24GB 推荐（4h），LoRA 可降至 8GB；**VLA-0-Smol（500M）**→ 消费级 GPU 可跑；**GR00T N1.5（更大）**→ 25GB+ VRAM，`--no-tune_diffusion_model` 可省显存；**Pi0（3B）**→ 需要多卡或云端。Linux 裸机 > WSL > Mac。

### D35. 数据采集到部署的完整 checklist
- **来源**: 多源汇总 — 综合 #5、#12、#17、#18、#29、#35 以及 D21、D26
- **原文摘要**: 社区 10+ 团队的实战经验汇总成标准化 checklist。
- **类别**: Recipe, Data
- **关键数据点汇总**:
  1. **硬件准备**：USB 权限（chmod 666）、固定相机位置、固定光照、udev rules 固定设备映射
  2. **数据采集**：一种策略做到底（D21）、先 50 条窄场景验证、通过摄像头观察（#35）、chunk_size 和 action_steps 分清（D26）
  3. **训练前验证**：数据回放（replay）是第一步（#30）、检查 stats.json 与数据集匹配
  4. **训练**：LR 5e-5 起步（#29）、bf16 不要 fp16、微调 vision encoder
  5. **评估**：固定场景先达 90%+ 再扩展、记录失败模式、备好替换电机（#5）

### D36. VLA 模型选型决策树：ACT vs SmolVLA vs GR00T vs Pi0
- **来源**: 多源汇总 — #1、#5、#16、#17、#29、#34、D2、D19、D23
- **类别**: Arch, Strategy
- **关键数据点汇总**:
  - **ACT（52M）**：最快上手、训练便宜、单任务 90%+ 可达，但无语言理解、无预训练泛化；适合：资源有限 + 单任务 + 快速验证
  - **SmolVLA（450M）**：VLM 预训练带来更好的视觉泛化、异步推理 30% 加速，但推理延迟较高（~10s on 3050Ti）；适合：多相机 + 需要一定泛化 + 有 A100 级 GPU
  - **GR00T N1.5/1.6（更大）**：NVIDIA 生态集成（Isaac Lab）、布料等复杂任务表现好，但 pick-place 精度不足（D19）、数据量需求 200-300 条（D24）；适合：NVIDIA 硬件生态 + 非精密任务
  - **Pi0/Pi0.5（3B+）**：最强泛化潜力 + RL post-training，但复现困难（D15 55% vs 95%）、硬件门槛高；适合：大型实验室 + 追求 SOTA

### D37. VLA 社区工具链全景（2026 年初）
- **来源**: 多源汇总 — #4、#18、#33、#36、D28、以及 LeRobot GitHub
- **类别**: Data, Strategy
- **关键数据点汇总**: 2026 年初 VLA 社区主要工具链：
  - **训练框架**: LeRobot（社区标准）、OpenPI（Pi0 系列）、Isaac Lab（NVIDIA 仿真）
  - **数据格式**: LeRobot v3（新标准）、RLDS、Zarr、HDF5；Forge（D28）实现互转
  - **数据采集**: Phosphobot（#18/#40，一键采集）、EmbodiFlow（#36，可视化管理）、SpaceMouse/VR（#26）
  - **评估**: Isaac Lab-Arena（#33，40× 加速）、LIBERO（标准但有记忆偏差）
  - **部署**: Jetson Thor（#28）、NXP i.MX95（#3）、ONNX/TensorRT
  - **社区资源**: LeRobot Discord（16k+ 成员）、HF Hub（2.2k+ 数据集贡献者）

### D38. 社区最常见失败模式 Top 5
- **来源**: 多源汇总 — D6、D7、D8、D9、#5、#7、#17、#30
- **类别**: Debug
- **关键数据点汇总**:
  1. **stats.json/normalization 不匹配**（#30）→ 动作完全失控，看起来像模型没学到东西
  2. **数据采集策略不一致**（D21、#17）→ 模型学到 bimodal behavior，有时做 A 有时做 B
  3. **夹爪电机过力/冻死**（D7、#5）→ eval 时校准失败触发，需备用电机
  4. **校准 position 超限**（D8）→ 硬编码上限 2047，组装位置不对就会触发
  5. **Data augmentation 适得其反**（D9）→ 某些任务上增强直接 0% SR

### D39. 开源 VLA vs 闭源 VLA：社区观察
- **来源**: 多源汇总 — #10（ICLR 2026 全景）、#11（π*0.6）、#25（12 Predictions）、#38（LingBot-VLA）
- **类别**: Strategy
- **关键数据点汇总**: ICLR 2026 VLA 提交 164 篇（前年 9 篇，18× 增长），但 RoboArena 真实评测中只有 Pi 系列有竞争力（#10）。开源模型（SmolVLA、X-VLA、OpenVLA-OFT）在 LIBERO 上接近闭源，但零样本真机泛化差距巨大。中国产业（LingBot-VLA #38）走开源路线，Physical Intelligence 走闭源 + RL post-training。**对个人研究者的建议**：开源模型足以做研究和单任务部署，但期望"开箱泛化"还不现实。

### D40. 从 50 条到 80 条：英文社区信息密度分析
- **来源**: 本次采集过程的元观察
- **类别**: Strategy
- **关键数据点汇总**: 英文 VLA 社区信息分布极不均匀——**独立个人 blog 信息密度最高**（ggando、Giacomo Moran、VLA-0-Smol 团队，每篇可提取 3-5 条独立 takeaway）；**官方 HF Blog 覆盖面最广**但深度有限；**Discord 低频关键词**（recovery episodes、calibration failed）信噪比 >> 高频词（success rate 228 结果多噪音）；**多源汇总条目**（D33-D38）价值最高——读者一次看到多团队共识。建议后续采集优先搜索新出现的个人实战 blog。

---

## GitHub Issues 周报

> 由 `scripts/en-vla-collector/github_vla_issue_collector.py` 在用户云端生成，
> 输出到 `reports/github_vla_issues_weekly.md`，通过 sync-vla-handbook 同步后可被读取。

---

*v3.0 — 2026-03-15 更新。Blog 40 条（#1-#40）+ Discord 40 条（D1-D40），共 80 条。v3 新增 30 条：VLA-0-Smol 500M ablation 全景（LR/vision encoder/action masking 优先级）、Correll Lab 失败复盘（stats.json 陷阱）、SmolVLA 架构深度拆解、GR00T N1.6 零样本 sim-to-real、Isaac Lab-Arena 40× 评估加速、X-VLA 跨具身冠军、LoRA VLA 8GB VRAM 训练、Forge 数据格式万能转换器、GR00T 数据量经验（200-300 最优区间）、VLA 训练决策优先级汇总、GPU 需求矩阵、模型选型决策树、完整 checklist、Top 5 失败模式、工具链全景。*
