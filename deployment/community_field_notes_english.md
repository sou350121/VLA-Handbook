# VLA/Embodied AI 英文社区实战笔记

> **版本**: v3.11 — 2026-08-21（新增 #211-#229：LeRobot v0.6.1 破坏性变更、Grabette/Gripette 490€ 手持采集器、GR00T 1.7 端到端配方 + SO-101 命令行、GS-VLA 相机位移 90%→10%、ReflexBench 可配置延迟、Embodiment Gap 综述、FACT 接触失败根因、G0.5 单流自回归、GigaBrain-0.7、EcoVLA 端边协同、ViTaR 触觉残差、τ₀-VLA test-time compute、CDPR 零示范 RL 对齐、proprioception 接法消融，以及 4 条深挖的 LeRobot 静默故障 issue）
> **上版**: v3.10 — 2026-07-31（新增 #198-#210：LeRobot v0.6.0 世界模型策略 + 六大 benchmark + DAgger CLI、官方算力硬件指南、LingBot-VLA 2.0 55D 统一动作空间、Gemini Robotics 2 全身控制、NVIDIA WAM 路线全景、vla.cpp、Embodied.cpp、RhinoVLA token 效率、Realtime-VLA FLASH 投机推理、机器人 leaderboard 刷榜、Figure 03 进 BMW 物流线、Robometer/TOPReward 奖励模型、SmolVLA 微调失败对质）
> **上版**: v3.9 — 2026-05-29（#186-#197：GR00T N1.7 + EgoScale dexterity scaling law、LeRobot Humanoid $2,500 双足、Figure Helix-02 8h+17h+40h 工厂直播、NVIDIA Cosmos 3、VLA-Perf 推理性能建模、PhysiFlow 多脑 flow matching、CF-VLA coarse-to-fine、How VLAs (Really) Work 开放世界批判、LIBERO-PRO v2 更新、MotuBrain 长 horizon 配方、HandITL 灵巧 HITL、FASTER 实时 flow VLA）
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
| 31 | Decoding SmolVLA: 架构设计深度拆解 | Phospho Blog | 2026-02 | Arch | [链接](https://blog.phospho.ai/decoding-smolvla-a-vision-language-action-model-for-efficient-and-accessible-robotics/) | SmolVLA 内部机制详解：pixel-shuffling 将每帧压缩到 64 tokens（vs PaliGemma 的 256+）；layer skipping 实现交错 CA+SA 降低计算量；flow matching 动作生成 vs 回归——flow matching 处理多模态动作分布更好；异步推理比同步快 30%（解耦 VLM 编码和动作生成）；450M 参数设计哲学：在 VLM backbone 和 action expert 之间找平衡 |
| 32 | GR00T N1.6 + Cosmos-Reason: Sim-to-Real 零样本 | NVIDIA Developer Blog | 2026-03 | Arch, Recipe | [链接](https://developer.nvidia.com/blog/building-generalist-humanoid-capabilities-with-nvidia-isaac-gr00t-n1-6-using-a-sim-to-real-workflow) | GR00T N1.6 核心升级：Cosmos-Reason-2B VLM 替代之前的视觉编码器；DiT 扩大 2× 至 32 层；state-relative actions 替代绝对动作；全身 RL 在 Isaac Lab 中训练；COMPASS 导航模块支持自主导航；**关键突破**：零样本 sim-to-real 迁移——仿真训练直接部署到真机无需额外微调；展示从感知到控制的完整 physical AI stack |
| 33 | Isaac Lab-Arena: 40× 加速的模块化评估框架 | NVIDIA Research | 2026-02 | Arch, Data | [链接](https://developer.nvidia.com/blog/simplify-generalist-robot-policy-evaluation-in-simulation-with-nvidia-isaac-lab-arena/) | 模块化任务构建：Object + Scene + Embodiment + Task 四维组合；并行评估 40× 加速（0.76h vs 34.9h）；250+ Lightwheel 预置任务；与 LeRobot Environment Hub 集成；支持 ACT/DP/VLA 等多种策略统一评估；解决 VLA 社区缺乏标准化评估的痛点 |
| 34 | X-VLA: 首个软提示跨具身 VLA（ICLR 2026 冠军） | HF LeRobot Docs | 2026-02 | Arch, Recipe | [链接](https://huggingface.co/docs/lerobot/en/xvla) | 0.9B 参数，ICLR 2026 最佳论文；核心创新：domain_id 系统为每个机器人注入 soft prompt，实现跨具身迁移；自动 action mode 检测（delta vs absolute）；LIBERO 93%、布料折叠 100%；Florence-2 视觉编码器 + flow matching action expert；训练只需冻结部分层 + LoRA，单卡可跑 |
| 35 | Robotics Made Simple: SO-101 新手实战 | Val Kamenski Blog | 2025-08-17 | Recipe, Debug | [链接](https://www.kamenski.me/articles/robotics-made-simple-playing-with-lerobot-and-so-101) | 新手向 SO-101 完整上手指南；**硬件建议**：RTX 4090+ 是长期投资（GR00T/SmolVLA 需要 CUDA，Mac/Colab 不够用）；Ubuntu 24.04 裸机安装问题最少；USB 端口需 `sudo chmod 666`（文档容易漏看）；组装陷阱：夹爪装反 → 校准失败；**关键心态**：别期望 VLA 开箱即用——必须先训练；遥操录数据时不要直接看 follower 臂，通过摄像头观察；SO-101 成本 $200-400 |
| 36 | EmbodiFlow: Pi0/Pi0.5 微调指南（基于 OpenPI） | EmbodiFlow (io-ai.tech) | 2026-01 | Recipe | [链接](https://io-ai.tech/platform/en/guides/Pipeline/LeRobot/Pi0/) | 基于 OpenPI 框架的 Pi0/Pi0.5 完整微调教程；导出 LeRobot v2.1 格式数据 → 配置 TrainConfig → compute_norm_stats → JAX 训练；**关键坑**：ALOHA 默认 14-dim action 但 7-axis arm 是 16-dim，不改代码会静默截断导致失控；推荐先用仿真（LIBERO/ALOHA Sim）排查流程再上真机 |
| 37 | ICRA 2026 Workshop: VLA Pipelines for Real Robots | ICRA 2026 | 2026-06 | Strategy | [链接](https://icra2026vlapipeline.github.io/) | ICRA 2026 专题研讨会聚焦 VLA 从仿真到真机的 pipeline 完整性；核心议题：数据引擎（自动采集+质量控制）、domain randomization、human-in-the-loop、标准化评估；反映学术界共识——VLA 的瓶颈已从模型架构转向工程 pipeline |
| 38 | LingBot-VLA: 蚂蚁集团开源"通用大脑"模型 | Robotics & Automation News | 2026-03-13 | Arch, Strategy | [链接](https://roboticsandautomationnews.com/2026/03/13/robbyant-open-sources-lingbot-vla-model-as-a-universal-brain-for-robots/99640/) | 蚂蚁集团旗下 Robbyant 开源 LingBot-VLA；20,000+ 小时真实数据预训练、覆盖 9 种双臂配置（AgileX、Galaxea R1Pro 等）；GM-100 benchmark 上 SOTA；训练速度 1.5-2.8× 于 StarVLA/OpenPI；开源含完整代码+数据处理+微调+评估工具链；代表中国 VLA 产业化新动态——开源路线与闭源（Physical Intelligence）形成对比 |
| 39 | LoRA VLA: 消费级 GPU 8GB 上训练 3.1B VLA | arXiv 2512.11921 | 2025-12 | Recipe, Edge | [链接](https://arxiv.org/abs/2512.11921) | 3.1B 参数 VLA 通过 LoRA + 量化技术在 8GB VRAM 消费级 GPU 上训练；200 条 demo 用于 SO-101 按钮按压任务；对比了冻结 vs 解冻 vision encoder 的 trade-off；证明资源受限的个人/小团队也能参与 VLA 研发 |
| 40 | Phosphobot: 一站式 VLA 开发中间件 | Phospho | 2026-02 | Recipe | [链接](https://docs.phospho.ai/learn/train-smolvla) | pip install 一键安装；Meta Quest VR 遥操作；支持云端 GPU 训练；LeRobot 格式原生支持；从数据采集到模型部署的完整 pipeline；显著降低 SO-100/SO-101 用户的入门门槛——不需要写数据管道代码 |
| 41 | LeRobot Pi0 Finetuning Tutorial (SO-ARM101) | Geonhui Jo Blog | 2025-12 | Recipe | [链接](https://ghuijo.github.io/blog/2025/LeRobot-PI0-Finetuning-Tutorial/) | A100 SXM4 40GB + Ubuntu 20.04 完整教程；PyTorch <2.8 限制（2.7.1+CUDA 12.8）；PaliGemma 需申请访问权限；freeze_vision_encoder=true + train_state_proj=true；**核心发现**：Pi0 在"重度过拟合到少量任务"时效果最好——与 foundation model 泛化期望相反；暗光环境下仍可执行任务（预训练 VLM 带来的鲁棒性）；config: 50 episodes, batch_size=8, save_steps=5000 |
| 42 | MolmoBot: 纯仿真训练零样本迁移真机 | Ai2 | 2026-03-11 | Arch, Strategy | [链接](https://allenai.org/blog/molmobot-robot-manipulation) | **全开源**纯仿真训练的机器人操作模型套件；1.8M 专家轨迹（MolmoBot-Data）；两个平台：RB-Y1 移动操作 + Franka FR3 桌面臂；Pick-and-place 79.2% SR 零样本迁移；**超越 π0.5**（使用大规模真实数据训练）；MolmoSpaces 提供 230K+ 室内场景 + 130K+ 物体资产 + 42M 抓取标注；三种架构：MolmoBot（最强）、MolmoBot-SPOC（轻量）、MolmoBot-Pi0（对照）；核心论点：仿真数据多样性足够大时可替代真实数据 |
| 43 | LiteVLA-Edge: Jetson Orin 上 6.6Hz VLA | arXiv 2603.03380 | 2026-03 | Edge | [链接](https://arxiv.org/abs/2603.03380) | SmolVLM-256M backbone + LoRA rank-8 微调 → 4-bit GGUF 量化（Q4_K_M）；llama.cpp 运行时 42 层全部 CUDA offload；端到端延迟 150.5ms±0.13ms（6.6Hz）；ROS 2 集成：VLA 6.6Hz "思考" + 底层控制器 100Hz 心跳；单一模型无需独立 policy 网络 → 架构开销更低；**核心价值**：首个在 Jetson 级硬件上实现"反应式控制"的量化 VLA pipeline |
| 44 | VLA 动作生成瓶颈分析：内存带宽是真凶 | arXiv 2603.02271 | 2026-03 | Edge, Arch | [链接](https://arxiv.org/abs/2603.02271) | 端到端延迟中高达 75% 被 memory-bound 的动作生成阶段消耗；模型延迟比 10Hz 实时操作需求高 200-300×；Thor 算力 5× 于 Orin 但延迟仅改善 1.4×（证明瓶颈在内存带宽非算力）；自回归生成在当前边缘加速器上稀疏且内存密集；**对社区的启示**：VLA 边缘部署需要算法-硬件协同设计，不能单靠堆算力 |
| 45 | TensorRT Edge-LLM: 边缘 VLA 推理框架 | NVIDIA Developer Blog | 2025-12 | Edge | [链接](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm) | 开源 C++ 框架，专为 Jetson Thor/DRIVE AGX Thor 设计；支持 EAGLE-3 投机解码、NVFP4 量化、chunked prefill；FP8/NVFP4/INT4 多精度支持，KV-cache 压缩；VLA 模型直接部署：视觉输入 → 语言指令 → 关节位置/速度输出；Bosch、ThunderSoft、MediaTek 已在 CES 2026 展示集成方案；**定位**：NVIDIA 边缘 VLA 推理的官方标准栈 |
| 46 | Jetson T4000 + JetPack 7.1: 新一代边缘 AI | NVIDIA Developer Blog | 2026-01 | Edge | [链接](https://developer.nvidia.com/blog/accelerate-ai-inference-for-edge-and-robotics-with-nvidia-jetson-t4000-and-nvidia-jetpack-7-1) | Jetson T4000/T5000 新模块，专为 VLA 工作负载设计；TensorRT Edge-LLM 原生支持；JetPack 7.1 统一 SDK：Video Codec + 推理引擎 + 部署工具链；GDDR7 内存提升带宽（呼应 #44 内存瓶颈发现）；**社区意义**：从 Orin → T4000 的升级路径为 VLA 边缘部署提供更多硬件选择 |
| 47 | Edge AI on Jetson 入门指南 | NVIDIA Developer Blog | 2026-01 | Edge, Recipe | [链接](https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics) | Jetson 上 LLM/VLM/VLA 部署完整入门教程；覆盖 GR00T N1.5/N1.6、SmolVLA、Pi0 等模型；G0Tiny（250M，SmolVLM2 backbone）在 Jetson Orin 上通过 TensorRT 达 10Hz；从模型选择到 ONNX 导出到 TensorRT 优化的标准流程；**实操价值**：NVIDIA 官方推荐的 VLA 边缘部署路径 |
| 48 | DPPO: Diffusion Policy + RL 微调 | Princeton (ICLR 2025) | 2025-09 | Arch, Recipe | [链接](https://diffusion-ppo.github.io/) | 首个系统性的 diffusion policy RL 微调框架；双层 MDP：内层=去噪过程，外层=环境交互；仿真训练 → 真机零样本部署，行为比 baseline 更平滑；结构化 on-manifold 探索 → 稳定训练 + 强鲁棒性；长 horizon 多阶段操作任务验证；**对社区的价值**：将 diffusion policy 从纯 IL 扩展到 RL，与 π*0.6 的方向一致 |
| 49 | WholeBodyVLA: 人形全身运动操作 | OpenDriveLab (ICLR 2026) | 2025-12 | Arch | [链接](https://github.com/OpenDriveLab/WholebodyVLA) | 统一 latent 学习框架：VLA 系统从低成本无动作标注的第一人称视频中学习；AgiBot X2 人形机器人验证；超越 baseline 21.3%；大空间人形 loco-manipulation：边走边操作；**核心创新**：用 action-free egocentric video 训练 VLA → 降低人形数据采集成本 |
| 50 | Helix 02: 像素到全身控制的统一 VLA | Figure AI | 2025-10 | Arch, Strategy | [链接](https://www.figure.ai/news/helix) | 三层架构：S0（千赫兹平衡）+ S1（200Hz 全身运动）+ S2（7-9Hz 高级推理）；**首个长 horizon 全身操作演示**：4 分钟完整洗碗机装卸；零样本抓取任何家用物品；从 Helix 到 Helix 02 的核心升级：S0 层实现人类级平衡协调；**产业意义**：Figure 展示了 VLA 从桌面臂到人形机器人的完整技术路线 |
| 51 | Trossen AI + OpenPI 集成 | Trossen Robotics | 2025-12 | Recipe | [链接](https://www.trossenrobotics.com/post/unlocking-new-possibilities-trossen-ai-arms-now-integrated-into-openpi-for-advanced-vla-models) | Trossen WidowX AI 臂完整集成 OpenPI 框架；支持 π0/π0.5 数据采集 → 训练 → 推理全流程；LeRobot v2.1 格式数据采集（兼容 OpenPI v0.1.0 训练脚本）；**实操价值**：非 SO-100/101 用户也能用 Physical Intelligence 模型；完整文档含 setup、teleoperation、configuration |
| 52 | AWS Embodied AI Blog Series: 云端 GR00T 微调 | AWS Spatial Computing Blog | 2025-12 | Recipe, Edge | [链接](https://aws.amazon.com/blogs/spatial/embodied-ai-blog-series-part-1/) | AWS Batch 基础设施上 GR00T 微调教程；从云端训练到边缘部署的完整 pipeline；**核心价值**：没有本地 GPU 集群的团队也能通过云端完成 VLA 训练；与 Isaac Lab 仿真集成；适合企业级 VLA 部署场景 |
| 53 | GR00T N1.5 on SO-100: Hackaday 实战 | Hackaday | 2025-11 | Recipe, Debug | [链接](https://hackaday.io/project/204187/log/243775-fine-tuning-gr00t-n15-for-so-100-robot-arm-manipulation) | SO-100（非 101）上 GR00T N1.5 微调实战；Hackaday 社区项目，硬件 DIY 视角；记录了从数据采集到微调到部署的完整流程；**社区价值**：SO-100 老用户的升级路径（之前主要用 ACT/DP，现在可以用 VLA） |
| 54 | OpenVLA 真机部署：从 0% 到 ~100% | Black Coffee Robotics | 2025-09 | Recipe, Debug | [链接](https://www.blackcoffeerobotics.com/blog/vision-language-action-vla-models-llms-for-robots) | OpenVLA-7B 在 Franka 上测试：未微调 ~0% SR → 微调后简单 pick-and-place ~100%；含抽屉开启、工具使用等多任务；8×A100 全量微调 vs LoRA 的 trade-off 讨论；**核心教训**：VLA 预训练权重不能直接用——必须针对目标场景微调；与社区共识一致（D32 泛化瓶颈） |
| 55 | DexGraspRL: 真机 RL 灵巧抓取 92% SR | arXiv 2503.04014 | 2025-03 | Arch, Recipe | [链接](https://arxiv.org/abs/2503.04014) | IL 预训练 + 真机 RL 微调的两阶段框架；平均 92% 灵巧抓取成功率；直接在真实环境中训练（非 sim-to-real）；**核心价值**：证明 RL post-training 对灵巧操作的有效性，与 π*0.6 路线一致但在灵巧手场景 |
| 56 | Teleoperation-as-a-Service (TaaS) | Haoru Xue (CMU) | 2025-11 | Data, Strategy | [链接](https://haoruxue.github.io/taas/) | 遥操作数据采集的服务化框架；assisted teleoperation：学习型辅助策略自动执行重复行为，仅在不确定时请求人类输入；**核心洞察**：数据采集的瓶颈不是硬件而是人力——TaaS 通过自动化降低每条 demo 的边际成本 |
| 57 | LeRobot: 统一机器人学习开源库 (ICLR 2026) | HuggingFace | 2026-02 | Arch, Strategy | [链接](https://arxiv.org/abs/2602.22818) | ICLR 2026 论文：统一端到端机器人学习栈；Python 中间件 API 支持多平台真机控制；IL/RL/VLA 三类 policy 统一训练框架；**数据生态**：HF Hub 2.2K+ 数据集贡献者；SO-100/101 + ALOHA + 人形 G1 支持；从 v0.1 到 v0.5 的演进：200+ PRs，50+ 新贡献者 |
| 58 | AIRoA ICRA 2026 VLA Competition + 10K Hours Data | AIRoA | 2026-02-10 | Data, Strategy | [链接](https://www.airoa.org/updates/2026-02-10) | ICRA 2026 全球 VLA Pipeline 竞赛；~10,000 小时真实机器人数据开放；鼓励参赛者分享方法、消融实验和失败案例；**产业信号**：学术界正在建立 VLA 标准化评估体系 |
| 59 | 3D Diffusion Policy (DP3): 40 demos 85% SR | Columbia University | 2025-06 | Arch, Recipe | [链接](https://3d-diffusion-policy.cs.columbia.edu/) | 3D 点云输入的 diffusion policy；仅 40 条真机 demo 达 85% SR；6 仿真 + 4 Franka 真机任务验证；推理速度满足实际部署需求；**对社区的价值**：深度/3D 信息确实能提升 manipulation 成功率（补充 D17 深度集成讨论） |
| 60 | Lite VLA: CPU 边缘机器人上的高效 VLA | arXiv 2511.05642 | 2025-11 | Edge | [链接](https://arxiv.org/abs/2511.05642) | 专为 CPU-bound 边缘机器人设计的轻量 VLA；无需 GPU 加速器即可运行；针对资源受限场景的模型压缩技术；**核心价值**：不是所有机器人都有 GPU——CPU-only VLA 开辟新部署场景 |
| 61 | GR00T N1.6: Cosmos-Reason VLM + 全身控制 | NVIDIA Research | 2026-03 | Arch | [链接](https://research.nvidia.com/labs/gear/gr00t-n1_6/) | Cosmos-Reason-2B VLM 替代之前视觉编码器；DiT 扩大 2× 至 32 层；state-relative actions；零样本 sim-to-real：仿真训练直接部署真机无需微调；COMPASS 导航模块支持自主导航；**N1.5 vs N1.6**：N1.6 收敛更快、动作更平滑，但更易过拟合需仔细调参 |
| 62 | Pi0-FAST 人形机器人微调经验 | GitHub Issue #591 | 2026-01 | Recipe, Debug | [链接](https://github.com/Physical-Intelligence/openpi/issues/591) | 社区用户在自定义人形机器人上微调 Pi0-FAST 的讨论；涉及 action dimension 适配、数据格式转换、训练超参选择；**典型困难**：人形机器人 DoF 远高于桌面臂（30+ vs 6-7），action space 设计是关键挑战 |
| 63 | Pi0-FAST 自定义数据集失败案例 | GitHub Issue #782 | 2026-02 | Debug | [链接](https://github.com/Physical-Intelligence/openpi/issues/782) | 用户报告 Pi0-FAST 在自定义数据集上动作完全失控（erratic）；与 D2（eliasab 报告 Pi-FAST 失控而 X-VLA 正常）一致；**可能原因**：normalization 配置错误、FAST tokenizer 参数不匹配、action space 定义问题 |
| 64 | Pi0 微调性能问题讨论 | GitHub Issue #427 | 2025-12 | Debug | [链接](https://github.com/Physical-Intelligence/openpi/issues/427) | 社区多位用户报告 Pi0 微调后性能不及预期；涉及学习率、batch size、训练步数的最优配置讨论；**共识**：Pi0 微调对超参数非常敏感，与 VLA-0-Smol（#29）发现一致——LR 是生死线 |
| 65 | LeRobot on Jetson Orin Nano | Medium (Marko Briesemann) | 2025-12 | Edge, Recipe | [链接](https://medium.com/@marko.briesemann/lerobot-on-jetson-orin-nano-seeed-studio-82986429509a) | Jetson Orin Nano 上完整运行 LeRobot + SO-101；Seeed Studio 硬件集成指南；从 JetPack SDK 安装到模型推理的端到端教程；**实操价值**：最便宜的 NVIDIA 边缘方案（Orin Nano ~$199）上跑 VLA |
| 66 | SmolVLA: 架构详解与实操教程 | LearnOpenCV | 2025-12 | Arch, Recipe | [链接](https://learnopencv.com/smolvla-lerobot-vision-language-action-model/) | SmolVLA 450M 架构图解：pixel-shuffling + layer skipping + flow matching action expert；端到端训练 + 推理代码教程；与 LeRobot 框架集成的完整 pipeline；**入门价值**：对初学者最友好的 SmolVLA 技术解析 |
| 67 | VLA 搜索通用机器人策略 | It Can Think (Substack) | 2026-01 | Strategy | [链接](https://itcanthink.substack.com/p/vision-language-action-models-and) | VLA 领域全景分析：从 RT-2 到 π0 的演进脉络；dual-expert vs generalist 两条路线的 trade-off；开源 vs 闭源生态的竞争格局；**核心观点**：VLA 正在从"能不能做"转向"能不能规模化部署" |
| 68 | R²D²: 仿真 + 语言模型提升操作 | NVIDIA Research | 2025-11 | Arch, Data | [链接](https://developer.nvidia.com/blog/r2d2-improving-robot-manipulation-with-simulation-and-language-models) | NVIDIA 机器人研发摘要（NeurIPS 2025）；ThinkAct + Generalizable Domain Adaptation + RobotSmith；latent diffusion 模型将仿真图像转换为逼真图像，支持 few-shot 适应和实时操作；**关键方向**：sim-to-real 的感知差距正在通过生成模型被弥合 |
| 69 | G0Tiny: 250M VLA 边缘部署 10Hz | Galaxea | 2026-02 | Edge, Recipe | [链接](https://github.com/OpenGalaxea/GalaxeaVLA) | 250M 参数（SmolVLM2 backbone）；R1 Pro Orin 边缘部署；TensorRT 优化达 10Hz；Galaxea 开源 VLA repo 含完整训练+部署代码；**核心价值**：比 LiteVLA-Edge（6.6Hz）更快的边缘 VLA 方案 |
| 70 | Modality-Augmented Fine-Tuning: 跨具身迁移 | arXiv 2512.01358 | 2025-12 | Arch, Recipe | [链接](https://arxiv.org/abs/2512.01358) | GR1 和 G1 人形机器人上的跨具身操作迁移；深度/触觉等多模态增强微调；RealSense 团队已开始实现（参考 D5）；**核心价值**：为 VLA 添加更多感知模态的系统方法论 |
| 71 | VLA Guide 2026: 全面技术指南 | HyScaler | 2026-01 | Strategy | [链接](https://hyscaler.com/insights/vision-language-action-vla-guide/) | 2026 年 VLA 技术全景指南；从理论基础到实际部署的完整覆盖；包含模型对比、数据需求、部署挑战分析；**定位**：非技术背景的决策者了解 VLA 的入门资源 |
| 72 | Diffusion Policy Survey: 分类、分析与未来 | TechRxiv | 2025-12 | Arch, Strategy | [链接](https://www.techrxiv.org/doi/full/10.36227/techrxiv.174378343.39356214/v1) | 机器人操作 Diffusion Policy 综述；跨 15 任务 4 benchmark 平均提升 46.9%；分类法：条件化方式、去噪策略、动作表示；**对社区的价值**：选择 Diffusion Policy 变体时的权威参考 |
| 73 | VLM on Jetson 部署实战 | Huu Phan Blog | 2026-02 | Edge, Recipe | [链接](https://www.huuphan.com/2026/02/vision-language-models-on-jetson-deploy.html) | Jetson 上 VLM 部署的实操指南；覆盖模型选择、量化策略、推理优化；与 VLA 部署直接相关（VLM 是 VLA 的感知组件）；**实操价值**：个人博客级别的接地气教程，比 NVIDIA 官方文档更易上手 |
| 74 | SmolVLA 官方论文: 450M 参数设计哲学 | arXiv 2506.01844 | 2025-06 | Arch | [链接](https://arxiv.org/abs/2506.01844) | SmolVLA 450M 设计决策详解；pixel-shuffling 压缩每帧到 64 tokens（vs PaliGemma 256+）；flow matching vs regression 的 action generation 选择；异步推理设计：解耦 VLM 编码和动作生成；**学术价值**：理解 SmolVLA 架构选择背后的 ablation 实验 |
| 75 | Sim-to-Real RL Survey + Foundation Models | AwesomeSim2Real | 2025-10 | Strategy | [链接](https://github.com/LongchaoDa/AwesomeSim2Real) | Sim-to-Real 强化学习综述 + Foundation Model 整合；覆盖 domain randomization、latent diffusion bridging、sim-real co-training；**核心趋势**：从"如何缩小 sim-real gap"转向"如何让 foundation model 自动适应"；MolmoBot（#42）是这一趋势的最新验证 |
| 76 | DeepSense.ai: VLA on 100g Edge Device | deepsense.ai Blog | 2025-08 | Edge, Arch | [链接](https://deepsense.ai/blog/we-put-embodied-ai-on-a-100g-device-why-most-vlas-choke-on-the-edge-and-the-architecture-that-didnt/) | 100g 以下设备 VLA 部署测试；对比 Raspberry Pi 5、Jetson Nano、RTX 2080 Ti、Hailo-8L；模块化 VLA 架构（vision/language/action 分离）边缘生存性最强；end-to-end unified VLA 过重；硬件选型功率/重量/延迟权衡矩阵 |
| 77 | ActionFlow: 2.55× VLA Inference Speedup Without Retraining | arXiv 2512.20276 | 2025-12 | Edge, Arch | [链接](https://arxiv.org/abs/2512.20276) | 系统级推理框架；跨请求流水线用 memory-bound decode 和 compute-bound prefill 批处理；OpenVLA-7B on Jetson AGX Orin 2.55× FPS 提升；无需重训；解决 VLA 边缘 3-5Hz vs 需求 20-30Hz 差距；与量化正交 |
| 78 | EdgeVLA: 7× Speedup via Parallel End-Effector Decoding | arXiv 2507.14049 | 2025-07 | Edge, Arch | [链接](https://arxiv.org/abs/2507.14049) | 消除端执行器预测的自回归依赖；集成紧凑语言模型；7× 加速、性能衰减最小；为边缘实时部署设计 |
| 79 | SpecPrune-VLA: Action-Aware Pruning 1.57-1.70× Speedup | arXiv 2509.05614 | 2026-02 | Edge, Arch | [链接](https://arxiv.org/abs/2509.05614) | 无训练双层剪枝；动作层静态剪枝 + 层级动态剪枝；action-aware controller 分类粗细颗粒度动作、调整剪枝强度；LIBERO sim 1.57×、真机任务 1.70× 加速；细颗粒度动作对剪枝敏感 |
| 80 | MulticoreWare: CogAct 7.6B Edge Deployment Optimization | MulticoreWare Blog | 2025 | Edge | [链接](https://multicorewareinc.com/deploying-vision-language-action-vla-based-ai-models-in-robotics-optimization-for-real-time-edge-inference/) | CogAct 7.6B 多流架构通过量化、剪枝、模型图优化；1.3× 加速（延迟 26% 降低）同时精度保留；真实边缘硬件部署验证；证明基础规模 VLA 可在设备端执行 |
| 81 | StarVLA: Lego-like Modular VLA Development Codebase | GitHub starVLA | 2025-12 | Arch, Data | [链接](https://github.com/starVLA/starVLA) | VLM→VLA 开发模块化 codebase；高内聚低耦合；支持 Qwen3.5（0.8B-9B）backbone；LeRobot v3.0 + DeepSpeed ZeRO-3；Behavior-1K/RoboCasa/CALVIN pipeline；内部开发可 <3h 构建新 VLA 框架；LingBot-VLA（#38）训练速度 1.5-2.8× 快于 StarVLA baseline |
| 82 | Data Scaling Laws: Power-Law for Robot Manipulation (ICLR 2025 Oral) | ICLR 2025 | 2025-10 | Data, Strategy | [链接](https://data-scaling-laws.github.io/) | 40k+ demos + 15k 真实 rollout；泛化服从环境数/物体数的 power-law；多样性 >> 绝对 demo 数；相机视角、空间布置是关键多样性维度；纹理优先级低；单任务策略 modest 数据投资即可零样本部署；ICLR 2025 Oral 论文 |
| 83 | What Matters in Large-Scale Robot Datasets (MimicLabs) | NVIDIA SRL + ICLR 2025 | 2025-06 | Data, Strategy | [链接](https://arxiv.org/abs/2506.13536) | MimicLabs 过程式数据生成框架；7 维度变化：传感器位置、物体类型/纹理、桌面纹理、空间布置、背景场景、动作基元；广泛预训练增强复杂任务能力但对微调数据需求改善贡献少；相机视角多样性远重于纹理变化 |
| 84 | Anton Maltsev: Qwen 3VL 2B as Robot Controller in 15 Minutes | Medium (Anton Maltsev / Rembrain) | 2026-02 | Recipe, Arch | [链接](https://medium.com/@zlodeibaal/one-of-the-best-vla-models-qwen-3vl-d-551cf9bf2e60) | Qwen 3VL 2B（非机器人模型）转为机器人控制器；~40 数据点 + Unsloth 快速训练 → 可用的机器人控制；Qwen + vLLM 秒级启动；VLM 相比专用 VLA 优势不在准确性而在可用性；对专用 VLA 架构的范式挑战 |
| 85 | Anton Maltsev: VLA Training Practical Guide | Medium (Anton Maltsev) | 2026-01 | Recipe, Strategy | [链接](https://medium.com/@zlodeibaal/vla-training-robots-to-kill-a723d731b810) | VLA 训练实操指南；高质量一致数据帮助显著但需要训练操作员 + 严格任务协议；更长上下文/记忆改善行为但增加模型大小 + 推理成本；从"能跑"到"产品级稳定"的工程距离仍巨大；AI 加速器选型对比（Intel/Qualcomm 软件支持优于 AMD） |
| 86 | Foxglove: Visual Debugging for LeRobot SO-100 | Foxglove Blog | 2025 | Debug, Data | [链接](https://foxglove.dev/blog/visualizing-lerobot-so-100-using-foxglove) | SO-100/101 机械臂状态实时可视化；图像面板 + 关节位置图表单屏展示；加速 multimodal 数据同时调试；免费工具；LeRobot 数据格式兼容；日志文件无法诊断的策略失败现象用视觉显示 |
| 87 | Maxence Boels: VLA Policy Development on SO-101 (PhD Research) | maxboels.com | 2026 | Recipe, Arch | [链接](https://maxboels.com/projects/lerobot-so101) | King's College London 博士生；SO-101 演示数据采集系统；自然语言指令 + 视觉理解灵活任务指定；开源机器人学改进贡献；VLA 开发学术视角 |
| 88 | Diffusion Steering: RL Fine-Tuning Pi0 by Steering Noise | diffusion-steering.github.io | 2025-06 | Arch, Recipe | [链接](https://diffusion-steering.github.io/) | diffusion policy VLA 新 RL 微调方法；仅更新噪声向量、冻结 DiT 权重；二元 0-1 成功标签作 reward；比 ConRFT（#89）更简单；Sergey Levine 背书；LeRobot Discord 广泛讨论；是 DPPO（#48）和 π0.6（#11）RL 后训练趋势的自然延伸 |
| 89 | ConRFT: RL Fine-Tuning Action Expert with Human-in-the-Loop | cccedric.github.io/conrft | 2025-06 | Arch, Recipe | [链接](https://cccedric.github.io/conrft/) | RL 微调 action expert 用人类在环反馈；与 Diffusion Steering（#88）互补——steering 冻结 DiT + 更新噪声，ConRFT 更新 DiT 权重；同时学习两者是开放研究方向；VLA RL 后训练浪潮的一部分 |
| 90 | Sim2Real-VLA: Zero-Shot Generalization from Synthetic Data | OpenReview 2025 | 2025-10 | Arch, Strategy | [链接](https://openreview.net/forum?id=H4SyKHjd4c) | 纯合成数据训练的通用操作 VLA；双系统架构：高层规划器推理 chains-of-affordances + 底层执行器 tokenized action space；过滤操作无关特征、优先运动关键动力学；零样本 sim-to-real 迁移，逼真环境成功率超基线 35%+；互补 MolmoBot（#42）纯 sim 方法 |
| 91 | UnifoLM-VLA-0: 宇树开源人形操作 VLA | Unitree Robotics GitHub | 2026-01 | Arch, Strategy | [链接](https://github.com/unitreerobotics/unifolm-vla) | 宇树机器人开源的通用人形操作 VLA 大模型；基于 Qwen2.5-VL 继续预训练；融合 Isaac-GR00T、Open-X、OpenVLA-OFT、InternVLA-M1 代码；从"视觉语言理解"进化为具备物理常识的"具身大脑"；Unitree G1 验证；代表中国人形机器人产业的 VLA 开源布局 |
| 92 | Wall-X: Qwen2.5-VL + Flow Matching 跨具身 VLA | LeRobot v0.5.0 | 2026-03 | Arch | [链接](https://huggingface.co/blog/lerobot-release-v050) | LeRobot v0.5 新增策略；Qwen2.5-VL 视觉语言 backbone + flow matching 动作头；跨具身机器人控制；与 Pi0-FAST（自回归）形成互补路线——Wall-X 用 flow matching，Pi0-FAST 用 FAST tokenization；社区已有初步训练经验 |
| 93 | RoboVLMs: 30 行代码集成任意 VLM 的统一框架 | GitHub Robot-VLAs | 2025 | Arch, Data | [链接](https://github.com/Robot-VLAs/RoboVLMs) | 统一 VLA 框架：30 行代码集成大多数 VLM；支持 CALVIN + Open-X 数据集；KosMos VLM backbone 驱动的最强 VLA 模型；公平比较不同 VLM backbone 的性能；解决社区痛点——每换一个 VLM backbone 就要重写训练代码 |
| 94 | VLA-RFT: World Model 做 RL 微调仅需 400 步 | arXiv 2510.00406 | 2025-10 | Arch, Recipe | [链接](https://arxiv.org/abs/2510.00406) | 用数据驱动的 world model 作为可控仿真器；从真实交互数据训练 world model 预测未来视觉观测；密集轨迹级 reward 来自 goal-achieving references；GRPO 优化；<400 fine-tuning steps 超越强 supervised baselines；展示 world-model-based RFT 作为 VLA 后训练范式的可行性 |
| 95 | OmniSAT: 紧凑动作 Token，6.8× 压缩加速训练 | arXiv 2510.09667 | 2025-10 | Arch | [链接](https://arxiv.org/abs/2510.09667) | 统一两阶段 tokenizer：通用动作 token 空间；Droid 大规模数据集预训练后序列长度压缩 6.8×；降低目标熵 → 更快收敛；跨真机+仿真实验验证；高压缩同时保持重建质量；与 FAST (#1 Pi0-FAST) 和 FASTer 竞争的新 action tokenization 方案 |
| 96 | FASTer: 可学习 Action Tokenizer + Block-wise 解码 | OpenReview ICLR 2026 | 2025-12 | Arch, Edge | [链接](https://openreview.net/forum?id=k6nTUFoqeT) | FASTerVQ：将 action chunks 编码为单通道图像，捕获全局时空依赖 + 高压缩比；FASTerVLA：基于此 tokenizer 的 block-wise 自回归解码 + 轻量动作专家；推理更快 + 任务性能更高；代表 action tokenization 从手工设计（FAST）到可学习（FASTer）的演进 |
| 97 | Awesome-RL-VLA: RL 微调 VLA 综述 | GitHub Denghaoyuan123 | 2026 | Arch, Strategy | [链接](https://github.com/Denghaoyuan123/Awesome-RL-VLA) | RL 微调 VLA 专题综述；覆盖 DPPO (#48)、Diffusion Steering (#88)、ConRFT (#89)、VLA-RFT (#94)、π*0.6 (#11)、HIL-SERL (#20) 等所有主要方法；分类框架 + 论文列表；**趋势确认**：RL post-training 已成为 VLA 研究最活跃的子领域之一 |
| 98 | VLA Survey: Action Tokenization 视角 | GitHub Psi-Robot | 2025 | Arch, Strategy | [链接](https://github.com/Psi-Robot/Awesome-VLA-Papers) | VLA 综述从 action tokenization 角度切入；覆盖 strategy/architectural transition、modality-specific processing、learning paradigms；分类学涵盖离散 vs 连续、自回归 vs diffusion、多模态 token 统一等维度；与 ICLR 2026 五大趋势 (#10) 高度吻合 |
| 99 | VLA Survey: 真实世界应用导向 | vla-survey.github.io | 2026 | Strategy | [链接](https://vla-survey.github.io/) | VLA 综述聚焦真实世界部署而非 benchmark 刷分；覆盖数据采集、训练范式、部署挑战全栈；对比开源 vs 闭源生态；**核心观察**：benchmark 性能和真实部署之间的差距是 VLA 领域最大的未解决问题 |
| 100 | Awesome-Embodied-VLA 论文列表 | GitHub jonyzhang2023 | 2025 | Strategy | [链接](https://github.com/jonyzhang2023/awesome-embodied-vla-va-vln) | 覆盖 VLA、VLN（视觉语言导航）、VA（视觉动作）三大方向的全面论文列表；2025 条目含 StarVLA、RLinf、Motus 等新模型；持续更新；**价值**：研究入门和文献追踪的一站式资源 |
| 101 | Exxact VLA 部署指南 + VRAM 需求矩阵 | Exxact Blog | 2025 | Edge, Strategy | [链接](https://www.exxactcorp.com/blog/deep-learning/vision-language-action-vla-models-powers-robotics) | VLA 推理 VRAM 需求：16-32GB；训练需求：>80GB（全参数）；覆盖从 OpenVLA 到 π0 的硬件配置建议；GPU 选型指南（A100/H100/RTX 4090）；**面向企业用户**：从采购到部署的决策参考 |
| 102 | NVIDIA 合成轨迹数据：World Model 增强机器人学习 | NVIDIA Developer Blog | 2026-03 | Data, Arch | [链接](https://developer.nvidia.com/blog/enhance-robot-learning-with-synthetic-trajectory-data-generated-by-world-foundation-models/) | 用 World Foundation Model 生成合成轨迹数据增强真实数据；视觉条件扩散模型将仿真图像转换为逼真图像；减少真实数据需求；Unitree G1 首批真实数据 + 24K 仿真遥操轨迹进入 NVIDIA Physical AI 数据集；**趋势**：world model 不仅用于规划也用于数据增强 |
| 103 | Physical OS: VLA vs World Models 深度对比 | Silicon Sand Studio (Substack) | 2026 | Arch, Strategy | [链接](https://siliconsandstudio.substack.com/p/physical-os-vision-language-action) | VLA 和 World Model 作为"Physical OS"两种路线的深度对比；VLA = 直接感知→动作；World Model = 感知→预测→规划→动作；Cosmos Policy (#8) 尝试统一两者；**核心论点**：最终路线可能不是二选一而是融合——world model 提供规划能力，VLA 提供实时反应能力 |
| 104 | Large VLM-based VLA Survey: 操作专题 | arXiv 2508.13073 | 2025-08 | Arch, Strategy | [链接](https://arxiv.org/abs/2508.13073) | 大型 VLM 驱动的 VLA 机器人操作综述；系统分类 VLM backbone 选择、动作头设计、训练数据来源；覆盖从 RT-2 到 π0 的完整演进；对比 7B vs 3B vs <1B 参数规模的 trade-off；**价值**：VLA 架构设计决策的权威参考 |
| 105 | Rohit Bandaru: Foundation Models for Robotics VLA 入门 | rohitbandaru.github.io | 2025 | Strategy | [链接](https://rohitbandaru.github.io/blog/Foundation-Models-for-Robotics-VLA/) | VLA foundation model 入门教程；从 CV/NLP foundation model 到 robotics foundation model 的类比；解释为什么 VLA 比传统策略更有前景；覆盖数据瓶颈、sim-to-real gap、标准化评估三大挑战；**适合人群**：从 ML 转入 robotics 的研究者 |
| 106 | SAM2 视频标注工具 + LeRobot 格式导出 | LeRobot Discord #general-chat — Asif | 2026-03 | Data | LeRobot Discord | 社区成员开发轻量级视频标注工具：SAM2 tracking + LeRobot 格式原生导出；解决数据标注痛点；社区反馈：数据创建中最大的痛点是手动标注耗时和格式转换；与 Forge (#D28) 互补——Forge 做格式转换，SAM2 工具做标注 |
| 107 | 机器人实验日志 + Debug 工作流讨论 | LeRobot Discord #general-chat — robo | 2026-03 | Debug | LeRobot Discord | 社区用户询问"机器人行为 debug 工作流"——大量时间花在调试上；反映社区对系统化 debug 方法论的需求；与 Foxglove (#86) 可视化 debug 工具形成呼应；**痛点**：目前缺乏 VLA 专用的实验管理和 debug 工具链 |
| 108 | GR00T N1 论文：开放人形机器人基础模型 | arXiv 2503.14734 | 2025-03 | Arch | [链接](https://arxiv.org/abs/2503.14734) | GR00T N1 官方论文：开放的通用人形机器人基础模型；dual-system 架构（System 1 快速控制 + System 2 慢速推理）；预训练+微调范式；Isaac Lab 仿真+真机评估；Unitree G1 + YAM + Agibot Genie-1 多平台验证；**学术价值**：NVIDIA 人形 VLA 的技术细节首次完整公开 |
| 109 | 视觉语义分割辅助机械臂定位 | LeRobot Discord #general-chat — River | 2026-03 | Arch, Recipe | LeRobot Discord | 社区成员用分割模型辅助机械臂正确旋转到 cube 水平位置；将视觉分割作为 VLA 的前端预处理而非端到端一体化；**启示**：模块化方法（分割+控制分开）在某些任务上可能比端到端 VLA 更实用 |
| 110 | Unified Flow & Matching Model (UFM): 夹爪相机新方向 | LeRobot Discord #robotics-papers — LeDaniel | 2025-06 | Arch | [链接](https://uniflowmatch.github.io/) | UFM 统一 flow 和 matching 模型；社区讨论将其用于关联夹爪相机看到的内容；潜在应用：夹爪视角的视觉特征匹配；与 GelSight 等触觉传感器方向互补——用视觉方法模拟触觉感知 |
| 111 | 视频数据标注自动化: 社区需求调查 | LeRobot Discord #general-chat — Asif | 2026-03 | Data | LeRobot Discord | Asif 发布第二次社区需求调查："你在创建数据集时最大的痛点是什么？"；反映社区对数据标注自动化的强烈需求；与 TaaS (#56) 遥操作服务化方向一致——数据采集的瓶颈从硬件转向人力和标注 |
| 112 | RoboArena: VLA 真机评测排行榜 | Moritz Reuss Blog | 2025-10 | Data, Strategy | [链接](https://mbreuss.github.io/blog_post_iclr_26_vla.html) | RoboArena 排行榜：真实环境 VLA 评测中仅 Pi 系列模型有竞争力；开源模型在 sim benchmark (LIBERO >95%) 上接近闭源但零样本真机泛化差距巨大；**核心价值**：比 LIBERO 更接近真实的评估——揭示 sim-to-real gap 的真实程度 |
| 113 | SARM: 长 Horizon 阶段感知奖励建模 | LeRobot v0.5.0 | 2026-03 | Arch | [链接](https://huggingface.co/blog/lerobot-release-v050) | LeRobot v0.5 新增策略；Stage-Aware Reward Modeling 解决长 horizon 任务的奖励稀疏问题；同时预测任务阶段和阶段内进度；使复杂多步操作任务的策略训练成为可能；与 HIL-SERL (#20) 互补——SARM 提供自动 reward，HIL 提供人工干预 |
| 114 | RTC: 实时分块推理加速 Flow Matching 策略 | Physical Intelligence / LeRobot v0.5.0 | 2026-03 | Edge, Arch | [链接](https://huggingface.co/blog/lerobot-release-v050) | Real-Time Chunking (RTC) 由 Physical Intelligence 贡献给 LeRobot v0.5；使 flow matching 策略实现更响应式的实时推理；减少动作执行延迟；与 ACTSmooth (#15) 解决类似问题但针对 flow matching 而非 ACT；**核心改进**：从"批量生成-执行"到"流式生成-执行" |
| 115 | LeRobot v0.5 流式视频编码 + 10× 图像训练加速 | LeRobot v0.5.0 | 2026-03 | Edge, Data | [链接](https://huggingface.co/blog/lerobot-release-v050) | 流式视频编码实现零等待推理——不需要等整个 chunk 完成就能开始处理；10× 图像训练加速（优化数据加载和预处理流水线）；PEFT/LoRA 官方支持降低训练门槛；**工程意义**：这些不是模型创新而是系统优化——但对社区的实际影响可能比新模型更大 |
| 116 | SimpleVLA-RL: GRPO 驱动的 VLA RL 扩展 (ICLR 2026) | arXiv 2509.09674 | 2025-09 | Arch, Recipe | [链接](https://arxiv.org/abs/2509.09674) | 基于 veRL 构建的 VLA RL 框架；GRPO（Group Relative Policy Optimization）无需 value function；OpenVLA-OFT 在 LIBERO 上从 91%→99% SOTA；冷启动仅 1 条轨迹/任务 SFT 后 RL 从 17.3→91.7（+430%）；Dynamic Sampling 解决全成功/全失败组的零梯度问题；**核心价值**：证明 RL 是 VLA 数据稀缺时的关键扩展路径 |
| 117 | SRPO: 自参照策略优化，200 步 RL 达 99.2% | arXiv 2511.15605 | 2025-11 | Arch, Recipe | [链接](https://arxiv.org/abs/2511.15605) | 用模型自身成功轨迹作为 self-reference，消除外部 demo 和人工 reward 需求；从 48.9% SFT baseline 仅 200 RL 步达 99.2%（+103%）；LIBERO-Plus 鲁棒性测试 +167%；**核心创新**：利用训练批次中的成功轨迹自举，无需额外专家数据 |
| 118 | VLAW: VLA + World Model 迭代共进化 | arXiv 2602.12063 | 2026-02 | Arch, Strategy | [链接](https://arxiv.org/abs/2602.12063) | 用真实 rollout 数据提升 world model 保真度→world model 生成合成数据提升 VLA→迭代循环；真机实验绝对成功率 +39.2%（超 base policy）；合成 rollout 数据额外贡献 +11.6%；**核心发现**：现有 world model 物理保真度不足（缺失失败案例覆盖），但迭代改进可修复 |
| 119 | GigaBrain-0.5M*: RAMP 框架 World Model + RL | arXiv 2602.12099 | 2026-02 | Arch, Strategy | [链接](https://arxiv.org/abs/2602.12099) | 基于 10000+ 小时机器人操作数据预训练；RAMP 四阶段迭代：world model 预训练→VLA 条件化微调→真实部署→持续训练；洗衣折叠/箱子打包/咖啡制作等挑战任务 +30% 超 RECAP baseline；RoboChallenge 国际基准第一；**趋势**：world model + RL 成为 VLA 后训练的新标准范式 |
| 120 | ForceVLA: 力觉 MoE 增强接触丰富操作 (NeurIPS 2025) | arXiv 2505.22159 | 2025-05 | Arch, Recipe | [链接](https://arxiv.org/abs/2505.22159) | FVLMoE：力觉感知的 MoE 融合模块，动态集成视觉语言嵌入与实时 6 轴力反馈；ForceVLA-Data 数据集：同步视觉/本体感觉/力矩信号；在 π₀ baseline 上平均任务成功率 +23.2%；插头插入任务达 80%；**核心价值**：将力觉从辅助信号提升为 VLA 一等公民模态 |
| 121 | CRAFT: 力觉课程微调适配接触丰富任务 | arXiv 2602.12532 | 2026-02 | Arch, Recipe | [链接](https://arxiv.org/abs/2602.12532) | Variational Information Bottleneck (VIB) 先抑制视觉语言→迫使模型学力觉→再逐步释放多模态；课程式微调保留预训练能力；5 个接触丰富真机任务验证（精密插入/持续擦拭/柔性物体操作）；跨 VLA 架构通用；**与 ForceVLA (#120) 互补**：ForceVLA 改架构，CRAFT 改训练方式 |
| 122 | MoDE-VLA: 灵巧手 MoE + 首个双手削苹果 | arXiv 2603.08122 | 2026-03 | Arch, Strategy | [链接](https://arxiv.org/abs/2603.08122) | Mixture-of-Dexterous-Experts：力觉/触觉通过专用 self-attention + sparse expert routing + residual injection 注入 VLA backbone；首个自主双灵巧手削苹果演示（成功率 30%，Peel Completion 73%）；4 个递增接触复杂度任务：齿轮装配/充电器插入/试管整理/削苹果；Sharpa Robotics 出品 |
| 123 | DexGrasp-VLA: 共享自主权灵巧手策略 | arXiv 2511.00139 | 2025-11 | Arch, Recipe | [链接](https://arxiv.org/abs/2511.00139) | VR 遥操控臂 + 自主 VLA 控制五指手 = 共享自主权；融合视觉/语言/触觉/本体感觉四模态；力自适应抓取 90% 跨物体成功率；**核心价值**：大幅降低灵巧手数据采集成本——人只需控 6DoF 臂，手部动作自动化 |
| 124 | DexGraspVLA: 通用灵巧抓取框架 (AAAI 2026 Oral) | arXiv 2502.20900 | 2026-02 | Arch | [链接](https://arxiv.org/abs/2502.20900) | AAAI 2026 Oral；VLA 框架实现通用灵巧抓取；杂乱场景下 90%+ 抓取成功率；将灵巧手操作从研究原型推向通用框架；与 DexGrasp-VLA (#123) 关注数据采集不同，本工作聚焦抓取策略本身 |
| 125 | EaqVLA: 编码对齐量化 (CVPR Workshop 2025) | arXiv 2505.21567 | 2025-05 | Edge, Arch | [链接](https://arxiv.org/abs/2505.21567) | VLA 多模块处理导致累积量化误差；编码对齐混合精度量化：按 Vision Encoder/Projector/Language/Action Head 四模块差异化分配比特；CVPR 2025 Workshop；**核心洞察**：VLA 量化不能直接套用 LLM 方法——模态映射失败是 VLA 特有问题 |
| 126 | QVLA: 通道级动作敏感量化 | arXiv 2602.03782 | 2026-02 | Edge, Arch | [链接](https://arxiv.org/abs/2602.03782) | 首个动作导向量化框架；按通道测量最终动作空间敏感度→全局贪心降位算法分配 {0,2,4,8,16} 比特；OpenVLA-OFT 仅需 29.2% 原始 VRAM、保留 98.9% 性能、1.49× 加速；超 SmoothQuant 22.6%；**核心发现**：VLA 量化中微小动作偏差会复合放大成灾难性任务失败 |
| 127 | QuantVLA: 免训练 PTQ 首次量化 DiT Action Head | arXiv 2602.20309 | 2026-02 | Edge, Arch | [链接](https://arxiv.org/abs/2602.20309) | 首个 VLA PTQ 框架 + 首次成功量化 diffusion transformer action head；attention temperature matching 稳定注意力 logits；output head balancing 校准投影能量漂移；LIBERO 上超全精度 baseline 的成功率、~70% 显存节省、1.22× 加速；**突破**：证明 DiT 动作头可以被量化而不损失性能 |
| 128 | HBVLA: 1-Bit 极致量化 VLA 探索 | arXiv 2602.13710 | 2026-02 | Edge, Arch | [链接](https://arxiv.org/abs/2602.13710) | 将 VLA 推到 1-bit 量化极限；探索超低比特 VLA 的可行性边界；与 EaqVLA (#125)/QVLA (#126)/QuantVLA (#127) 形成完整的 VLA 量化研究谱系（从 4-bit 到 1-bit）；**趋势确认**：VLA 量化已从"能不能做"转向"能做到多极致" |
| 129 | Interleave-VLA: 交错图文指令增强零样本泛化 | arXiv 2505.02152 | 2025-05 | Arch, Data | [链接](https://arxiv.org/abs/2505.02152) | 首个支持交错图文指令的 VLA 范式；自动 pipeline 将 Open X-Embodiment 文本指令转为交错图文（210k episodes）；未见物体零样本泛化 2× 提升；支持手绘草图等灵活输入；模型无关——可扩展到任意 VLA backbone；**核心启示**：指令表示方式本身就是一个被低估的研究方向 |
| 130 | RLinf-VLA: 统一 VLA+RL 训练基础设施 | arXiv 2510.06710 | 2025-10 | Arch, Recipe | [链接](https://arxiv.org/abs/2510.06710) | 统一接口标准化 VLA 架构 × RL 算法 × 异构仿真器集成；混合细粒度 pipeline 分配策略 1.61-1.88× 训练加速；单模型 LIBERO 130 任务 98.11%、ManiSkill 25 任务 97.66%；**工程价值**：将 VLA RL 训练从"各自为战"推向可复现的统一基准 |
| 131 | GR-Dexter: 字节跳动灵巧手全栈方案 | arXiv 2512.24210 | 2025-12 | Arch, Strategy | [链接](https://arxiv.org/abs/2512.24210) | ByteDexter 手：紧凑 21-DoF 设计 + 高密度压阻触觉传感器覆盖指尖；双臂遥操 + VLA 训练端到端框架；硬件-模型-数据一体化方案；**产业意义**：字节跳动正式进入具身智能领域，硬件+VLA 全栈布局 |
| 132 | Efficient VLA Survey: 系统级效率优化综述 | arXiv 2510.24795 | 2025-10 | Edge, Strategy | [链接](https://arxiv.org/abs/2510.24795) | 覆盖量化/剪枝/蒸馏/层级跳跃/KV cache 等全部 VLA 效率优化方法；系统分类：模型压缩 vs 推理优化 vs 架构简化；从 OpenVLA 4-bit PTQ 到 SQIL saliency-aware 量化 2.5× 加速；**定位**：边缘部署决策的权威参考综述 |
| 133 | Nature Machine Intelligence: What Matters in Building VLA | Nature MI | 2025 | Strategy | [链接](https://www.nature.com/articles/s42256-025-01168-7) | 顶级期刊对 VLA 构建核心要素的深度分析；覆盖数据、架构、训练范式三大支柱；从产业和学术双重视角评估 VLA 发展路径；**定位**：面向跨领域读者的权威 VLA 现状评估——Nature 子刊背书 |
| 134 | Spatially-Anchored Tactile: 亚毫米精度灵巧操作 | arXiv 2510.14647 | 2025-10 | Arch | [链接](https://arxiv.org/abs/2510.14647) | 现有视触觉学习方法在亚毫米精度任务上表现差；空间锚定触觉表示结合手部运动学空间关系；解决触觉信号丰富感知与空间定位的脱节问题；**核心洞察**：触觉不仅需要"感知什么"更需要"在哪里感知" |
| 135 | π_RL: Flow-Based VLA 的在线 RL 微调 | arXiv 2510.25889 | 2025-10 | Arch, Recipe | [链接](https://arxiv.org/abs/2510.25889) | 专门针对 flow matching VLA（如 π0）的在线 RL 微调方法；与 Diffusion Steering (#88) 冻结权重不同，π_RL 直接优化 flow model 参数；在线交互而非离线数据；**对社区的意义**：为 flow matching 路线（Pi 系列、Wall-X）提供了原生 RL 适配方案 |
| 136 | VLA-in-the-Loop: World Model 在线纠错 | OpenReview | 2025 | Arch | [链接](https://openreview.net/forum?id=aT4LG8c6DE) | World model 在推理时在线纠正 VLA 抓取策略；无需额外训练；将 world model 从离线规划器变为在线安全网；与 VLAW (#118) 离线迭代不同，本方法在部署时实时运行；**核心价值**：VLA 部署的实时安全保障方案 |
| 137 | Contact-Rich IL Survey: 接触丰富任务模仿学习综述 | arXiv 2506.13498 | 2025-06 | Arch, Strategy | [链接](https://arxiv.org/abs/2506.13498) | 接触丰富机器人任务 IL 综述；覆盖 diffusion 力/位置轨迹生成、Mamba 长序列模型；有限数据下泛化策略分析；**核心趋势**：从"接触回避"到"接触利用"——VLA 必须学会用力而非避力 |
| 138 | Multimodal Fusion VLA Survey | ScienceDirect | 2025 | Arch, Strategy | [链接](https://www.sciencedirect.com/science/article/pii/S1566253525011248) | 多模态融合 VLA 系统综述；覆盖视觉-语言-动作-力觉-触觉多模态集成方法；信息融合期刊发表；**价值**：从信息融合理论角度审视 VLA——不仅是 ML 也是信号处理问题 |
| 139 | Embodied AI TopConf 论文追踪器 | GitHub Songwxuan | 2025 | Strategy | [链接](https://github.com/Songwxuan/Embodied-AI-Paper-TopConf) | 持续维护的具身 AI 顶会论文列表；覆盖 ICLR/NeurIPS/ICML/RSS/CoRL/ICRA/IROS/CVPR/ICCV/ECCV；按年份+会议分类；**工具价值**：一站式追踪 VLA 领域所有顶会论文——比手动搜索高效 10× |
| 140 | Robotic Manipulation via IL: 全面分类与演进 | arXiv 2508.17449 | 2025-08 | Arch, Strategy | [链接](https://arxiv.org/abs/2508.17449) | 2021-2025 机器人操作 IL 全面综述；从 diffusion/flow matching 到自回归/affordance 的方法论演进；覆盖数据采集、策略学习、部署全栈；**核心价值**：理解 VLA 如何从传统 IL 演进而来——知道历史才能判断未来 |
| 141 | CGVD: 训练无关的 VLA 视觉杂乱防御 | arXiv 2603.10340 | 2026-03 | Arch | [链接](https://arxiv.org/abs/2603.10340) | Concept-Gated Visual Distillation——训练无关、模型无关的推理框架；解决 VLA 在杂乱环境中的"精度-推理鸿沟"（背景诱导的特征稀释）；通过 Fourier 修复生成干净观测；杂乱环境成功率 43%→77.5%；**核心价值**：不改模型、不重训练就能大幅提升真实环境部署可靠性 |
| 142 | AR-VLA: 真正的自回归动作专家 | arXiv 2603.10126 | 2026-03 | Arch | [链接](https://arxiv.org/abs/2603.10126) | 独立自回归 Action Expert，维护自身历史记忆（long-lived memory）；解决快速控制与慢速推理的频率不匹配；轨迹更平滑、时空一致性更强；可替代传统 chunk-based action head；**对比 Pi0-FAST**：AR-VLA 是 context-aware 连续生成 vs Pi0-FAST 是 chunk-based 反应式 |
| 143 | SeedPolicy: 自进化 Diffusion Policy 解锁长程操作 | arXiv 2603.05117 | 2026-03 | Arch | [链接](https://arxiv.org/abs/2603.05117) | Self-Evolving Gated Attention (SEGA) 解决 Diffusion Policy 随观测窗口增加的性能退化；RoboTwin 2.0 50 任务：+36.8%（clean）/ +169%（randomized）；参数量比 RDT 1.2B 少 1-2 个数量级但性能可比；代码开源 github.com/Youqiang-Gui/SeedPolicy |
| 144 | BPP: 聚焦关键历史帧的长程 IL | arXiv 2602.15010 | 2026-02 | Arch | [链接](https://arxiv.org/abs/2602.15010) | Big Picture Policies——用 VLM 检测最小有意义关键帧集合；解决 naively conditioning on past observations 的虚假相关问题；真实世界评估比最佳对比方法高 70% 成功率；**核心洞察**：不是"看得越多越好"——精选关键帧比全历史更有效 |
| 145 | EasyMimic: 低成本人类视频→机器人策略（ICRA 2026） | arXiv 2602.11464 | 2026-02 | Data, Recipe | [链接](https://arxiv.org/abs/2602.11464) | 从 RGB 人类视频提取 3D 手部轨迹→映射到机器人控制空间；手部视觉增强策略弥合 human-to-robot domain gap；co-training（人类数据+少量机器人数据）；LeRobot 平台验证 avg 0.88 成功率 vs robot-only 0.40；语言条件任务 0.90；**意义**：大幅降低数据采集成本 |
| 146 | LaRA-VLA: 潜在推理 VLA（LIBERO 97.9%） | arXiv 2602.01166 | 2026-02 | Arch | [链接](https://arxiv.org/abs/2602.01166) | 将多模态 CoT 推理内化为连续潜在表示；课程式训练：显式 CoT→潜在推理→动作生成；LIBERO avg 97.9%（Object 99.8%、Long 96.6%）；消除推理时的显式 CoT 生成——更快更紧凑；**对比**：显式推理（Diffusion-VLA）vs 潜在推理（LaRA-VLA）是 2026 年核心辩题 |
| 147 | FUTURE-VLA: 动作+未来预览同步生成 | arXiv 2602.15882 | 2026-02 | Arch | [链接](https://arxiv.org/abs/2602.15882) | 单次前向传播同时生成动作块和未来视觉预览；DINOv3 编码器 + 时间自适应级联压缩；Human-in-the-Loop 执行门控——人类可实时审核 VLA 的意图后再放行；**核心创新**：首次让 VLA "说出"自己打算做什么（通过视觉预测） |
| 148 | Recurrent-Depth VLA: 测试时计算缩放（80× 加速） | arXiv 2602.07845 | 2026-02 | Arch, Edge | [链接](https://arxiv.org/abs/2602.07845) | 权重共享循环 action head + 自适应停止准则；0% 成功（单次迭代）→90%+（4 次迭代）——难任务需要更多"思考"；恒定内存占用；比 token-based reasoning VLA 推理快 80×；**核心洞察**：test-time compute scaling 对 VLA 同样有效——难任务多想、简单任务少想 |
| 149 | Chain of World (CoWVLA): 潜在运动中的 World Model 思考 | arXiv 2603.03195 | 2026-03 | Arch | [链接](https://arxiv.org/abs/2603.03195) | 预训练视频 VAE 提取结构/运动潜在量→VLA 学习推理连续运动链并预测终帧；解决 world model VLA 重建冗余背景的浪费；比现有 world model 和 latent action 方法都好；**定位**：world model × latent action 的最佳结合点 |
| 150 | KAN-We-Flow: KAN+RWKV 轻量 3D Flow Matching | arXiv 2602.01115 | 2026-02 | Arch, Edge | [链接](https://arxiv.org/abs/2602.01115) | RWKV 做时间/通道混合 + GroupKAN 做逐特征非线性校准；Action Consistency Regularization 减少漂移；参数减少 86.8%；Adroit/Meta-World/DexArt SOTA；**核心价值**：证明 flow matching 策略可以做到极轻量且保持 SOTA——对边缘部署意义重大 |
| 151 | Refined Policy Distillation: VLA→RL 专家蒸馏（IROS 2026） | arXiv 2503.05833 | 2025-03 | Arch, Recipe | [链接](https://arxiv.org/abs/2503.05833) | 用 RL + BC 将 VLA 泛化能力蒸馏到紧凑专家策略；学生超越教师——蒸馏出的 RL 专家比原 VLA 成功率更高；对相机视角变化鲁棒；可泛化到 VLA 自己无法解决的任务变体；**IROS 2026 接收**；**对社区的意义**：VLA 做泛化基座、RL 做精细打磨的两阶段范式 |
| 152 | VLATest: VLA 模型模糊测试框架（FSE 2025） | arXiv 2409.12894 | 2024-09 | Data, Strategy | [链接](https://arxiv.org/abs/2409.12894) | 10 种测试算子 + 生成式模糊测试；ManiSkill2 仿真环境；7 个代表性 VLA 模型评估；核心发现：VLA 对未见物体性能严重下降、对指令改写不鲁棒；**警示**：当前 VLA 还远未达到实际部署的鲁棒性要求 |
| 153 | DiffusionVLA: 自回归推理 + Diffusion 策略统一（ICML 2025） | arXiv 2412.03293 | 2024-12 | Arch | [链接](https://arxiv.org/abs/2412.03293) | VLM 自生成推理短语注入 diffusion 策略学习；2B→72B 可扩展；DiffusionVLA-2B 单卡 A6000 82Hz；零样本 bin-picking 63.7%（102 未见物体）；<50 demo 从零训练复杂任务；**定位**：推理能力 + 动作精度的首次大规模统一 |
| 154 | Discrete Diffusion VLA: 离散扩散动作解码 | arXiv 2508.20072 | 2025-08 | Arch | [链接](https://arxiv.org/abs/2508.20072) | 统一 transformer 中用离散扩散建模动作块；自适应解码顺序——先解简单动作元素再解难的；二次重掩码修正不确定预测；LIBERO 96.3%；打破自回归瓶颈的新范式；**vs AR-VLA (#142)**：离散扩散 vs 连续自回归——两种突破 chunk-based 限制的路径 |
| 155 | GeoPredict: 3D Gaussian 几何增强 VLA 精准操作 | arXiv 2512.16811 | 2025-12 | Arch | [链接](https://arxiv.org/abs/2512.16811) | 预测多步 3D 关键点轨迹 + 预测 3D Gaussian 工作空间几何；训练时深度监督、推理时仅需轻量 query token；RoboCasa/LIBERO/真实世界几何密集任务超越基线；**核心洞察**：VLA 不能只看 2D——3D 几何先验是精细操作的关键 |
| 156 | UniForce: 跨触觉传感器统一潜在力模型 | arXiv 2602.01153 | 2026-02 | Arch | [链接](https://arxiv.org/abs/2602.01153) | 统一潜在力空间学习框架；跨 GelSight/TacTip/uSkin 传感器域迁移；联合建模逆动力学（图像→力）和正向动力学（力→图像）；支持 Vision-Tactile-Language-Action 模型；**核心价值**：解决触觉传感器碎片化——不同传感器可以共享同一个力表示空间 |
| 157 | ReBot: Real-to-Sim-to-Real 视频合成缩放数据（IROS 2025） | arXiv 2503.14526 | 2025-03 | Data | [链接](https://arxiv.org/abs/2503.14526) | 真实轨迹→仿真重放→真实背景修复→合成视频；GroundedSAM2 分割 + ProPainter 修复；OpenVLA 域内 +21.8%、域外 +9.4%；Franka 真实评估 +20%；**与 R2R2R (#108) 对比**：ReBot 侧重视频合成多样化、R2R2R 侧重渲染替代仿真 |
| 158 | Demonstration Modality Impact: 示教方式对 IL 的影响 | arXiv 2503.07017 | 2025-03 | Data, Recipe | [链接](https://arxiv.org/abs/2503.07017) | 系统比较动觉示教 vs VR 遥操 vs SpaceMouse 遥操；混合少量动觉数据 + 遥操数据→成功率平均 +20%；用户偏好动觉但多数数据集用遥操采集；**实操建议**：不要只用一种示教方式——混合模态比单一模态更好 |
| 159 | LeRobot.js: 浏览器端机器人控制（JavaScript） | HF Blog (NERDDISCO) | 2026 | Edge, Recipe | [链接](https://huggingface.co/blog/NERDDISCO/lerobotjs) | @lerobot/web 包；Chromium Web Serial + Web USB；支持 SO-100；API：find/connect/disable torque/calibrate/teleoperate；浏览器内直接采集数据集并导出 LeRobot 格式；**意义**：将 LeRobot 生态从 Python 扩展到 Web——降低入门门槛、支持远程遥操 |
| 160 | LeRobot Annotation Studio: 浏览器端数据集标注 | HF Space | 2026-03 | Data | [链接](https://huggingface.co/blog/lerobot-release-v050) | LeRobot v0.5.0 引入的 HuggingFace Space；为数据集中的每个时刻标注自然语言子任务；**意义**：解决 VLA 训练数据中细粒度语言标注缺失的问题——之前只有 episode-level 标签，现在可以做 moment-level |
| 161 | Real-is-Sim: 动态数字孪生弥合 Sim-to-Real | arXiv 2504.03597 | 2025-04 | Data, Arch | [链接](https://arxiv.org/abs/2504.03597) | 维持仿真与物理环境的实时对齐→策略无需微调直接部署；单一框架统一仿真与现实；**核心价值**：不是"让仿真更像真实"而是"让仿真跟踪真实"——实时数字孪生 vs 静态域随机化 |
| 162 | 10 Open Challenges for VLA Models | arXiv 2511.05936 | 2025-11 | Strategy | [链接](https://arxiv.org/abs/2511.05936) | 十大开放挑战系统分析；覆盖泛化/安全/效率/多模态/评估等维度；**与 D85 对比**：学术视角的十大问题 vs 社区实战视角的十大问题——互补阅读价值 |
| 163 | VLA Edge Bottleneck 分析: 动作生成才是瓶颈 | arXiv 2603.02271 | 2026-03 | Edge | [链接](https://arxiv.org/abs/2603.02271) | 系统分析 VLA 边缘部署瓶颈：动作生成（diffusion/flow matching head）是推理延迟的主要来源而非视觉编码器或 LLM；**核心洞察**：优化 VLA 推理应优先针对 action head——这与常识（先优化最大模块）相反 |
| 164 | Pure VLA Survey: 纯 VLA 模型全面分类 | arXiv 2509.19012 | 2025-09 | Strategy | [链接](https://arxiv.org/abs/2509.19012) | 四大类方法分类：自回归/扩散/强化/混合+专用；覆盖 80+ VLA 模型；**vs #140 IL Survey**：本综述聚焦"纯 VLA"而非广义 IL——更深入但更窄 |
| 165 | VLA Concepts Survey: 概念/进展/应用/挑战 | arXiv 2505.04769 | 2025-05 | Strategy | [链接](https://arxiv.org/abs/2505.04769) | 最全面的 VLA 综述之一；80+ 模型；从概念到应用到挑战全覆盖；提出 agentic adaptation 和 cross-embodiment planning 解决方案；v2 更新版可用；**入门推荐**：新手了解 VLA 全貌的首选综述 |
| 166 | π0.7: 首个展现组合泛化的可引导 VLA 基础模型 | Physical Intelligence | 2026-04-16 | Arch, Strategy | [链接](https://www.pi.website/blog/pi07) | PI 最新旗舰，公开定位为"step-change in generalization"。**核心架构突破**：统一多模态 prompt（语言子任务描述 + metadata 描述速度/质量 + 控制模态标签 joint vs end-effector + world model 生成的视觉子目标图）使单一模型可消化异构数据源（多机器人 + 人类视频 + autonomous episodes）；**关键实验**：bimanual UR5e 双臂折叠衣物任务零样本迁移，π0.7 成功率匹配训练数据采集者（平均 375 小时遥操经验的专家）首次用 UR5e 做此任务的"零样本"水平；"air fryer 放红薯"任务：零样本接近 0% → 语言 coaching 显著提升 → 用 coaching 数据微调高层策略后完全自主执行；单模型性能 ≈ Recap RL 专家模型（原本各任务单独训练）；**对社区含义**：首次把 LLM 式的组合泛化带入机器人基础模型。**存疑点**：任务集仍局限于厨房/折叠领域，真实家用"first day in new home"未测 |
| 167 | Gemini Robotics-ER 1.6: 首个实用级仪表读数 VLM | Google DeepMind | 2026-04-15 | Arch, Strategy | [链接](https://deepmind.google/blog/gemini-robotics-er-1-6/) | DeepMind 推出的 VLA "high-level brain" 升级；**核心量化**：仪表读数准确率 23% (ER 1.5) → 93% (ER 1.6)，通过 agentic vision（视觉推理 + 代码执行组合，先裁剪放大再估计刻度比例）实现；**定位**：不是端到端 VLA 而是编排层——原生调用 Google Search/VLA/自定义函数，为下游 VLA 提供 pointing/计数/任务规划/成功检测；多视图（俯视 + 腕摄）推理能力显著改进；合作方 Boston Dynamics Spot 在设施巡检场景中使用；**对产业含义**：验证"VLM 作为 reasoning layer + VLA 作为 action layer"架构，与 Cosmos Reason 2 (#171) / Figure Helix S2 (#50) 路线一致 |
| 168 | Precise Manipulation with Efficient Online RL (RLT) | Physical Intelligence | 2026-03-19 | Arch, Recipe | [链接](https://www.pi.website/research/rlt) | 从 VLA 模型中提取 "RL Token" 以支持快速在线 RL；**核心方法**：不 full-tune VLA 参数，只训练一个 lightweight RL Token；**效率指标**：精密任务仅需"few hours"数据即可在线 RL 微调提升吞吐；**对比 π*0.6 (#11)**：π*0.6 是完整 RL post-training（两种经验学习方式），RLT 是更轻量的 token-level 方法——适合单任务快速精度提升；**适用场景**：peg-in-hole / 精密装配等需要高成功率 + 快速迭代的任务 |
| 169 | VLAs with Long and Short-Term Memory (MEM) | Physical Intelligence | 2026-03-03 | Arch | [链接](https://www.pi.website/research/memory) | Multi-Scale Embodied Memory (MEM) 为 VLA 增加长/短期记忆；**核心能力**：支持 >10 分钟的复杂长任务（如多步烹饪、整理整理房间）；**解决痛点**：Pi0 在 wild 实验中被诊断为"memoryless" (#7 Penn PAL 评测)，MEM 明确针对此瓶颈；**对比**：Helix 02 的分层架构是 S0/S1/S2 空间分层；MEM 是时间分层（多尺度时间窗记忆）；**社区意义**：长 horizon 任务不再只靠 prompt engineering，而是架构级支持 |
| 170 | State of Open Source on Hugging Face: Spring 2026 | HuggingFace Team | 2026-03-17 | Strategy, Data | [链接](https://huggingface.co/blog/huggingface/state-of-os-hf-spring-2026) | HF 年度开源报告中机器人板块的震撼数据：**机器人数据集数量 1,145 (2024) → 26,991 (2025)，一年 23× 增长**；从 Hub 数据集类别第 44 名 → **第 1 名**（超过 text generation 的 ~5k）；LeRobot GitHub stars 一年翻近三倍；Pollen Robotics 被 HF 收购后 SO-ARM 等硬件扩大销售；提到 L2D（Yaak 合作）、RoboMIND（107,000+ 真实轨迹 × 479 任务）等大规模数据集；**对研究者的含义**：数据生态拐点已到——2025 是开源 VLA 数据的"寒武纪"；数据不再是瓶颈，标注/筛选/使用效率成新瓶颈 |
| 171 | NVIDIA Cosmos Reason 2: 通用 Physical AI 推理 VLM | NVIDIA (HF Blog) | 2026-01-05 | Arch, Edge | [链接](https://huggingface.co/blog/nvidia/nvidia-cosmos-reason-2-brings-advanced-reasoning) | 2B/8B 双尺寸开源 VLM，**Physical AI Bench 和 Physical Reasoning leaderboard 开源 #1**；**关键升级**：输入 context 从 Cosmos Reason 1 的 16K → 256K tokens（16× 扩展，支持长视频推理）；新增能力：OCR、2D/3D 点定位、bounding box、轨迹坐标；直接作为 GR00T N1.6 (#32/#61) 视觉编码器；也用于数据标注/视频分析 agent（Salesforce/Uber 已集成）；**对 VLA 社区**：2B/8B 小尺寸专为边缘设计——可以跑在 Jetson Thor，弥补以前"边缘 VLM 太大"的空白 |
| 172 | Fine-tune GR00T N1.6 on SO-101 + AGX Orin 64G | Seeed Studio Wiki | 2026-03 | Recipe, Edge | [链接](https://wiki.seeedstudio.com/fine_tune_gr00t_n1.6_for_lerobot_so_arm_and_deploy_on_agx_orin/) | GR00T N1.6 在 AGX Orin 64GB + JetPack 6.2 的端到端部署指南；**关键约束**：微调需要 **48GB+ VRAM**（推荐租云端 GPU 服务器做训练），推理再部署到 AGX Orin；**对比 #28 (Jetson Thor + N1.5)**：N1.6 版本社区首个 AGX Orin（非 Thor）部署教程——价格门槛降低（Thor 目前稀缺），让更多团队能部署 N1.6；**坑**：N1.6 过拟合更严重（#61），微调需仔细调学习率；Seeed 系列 wiki 目前是 GR00T 社区最全的硬件部署资源 |
| 173 | ViVa: Video-Generative Value Model for Robot RL | arXiv 2604.08168 | 2026-04 | Arch | [链接](https://arxiv.org/abs/2604.08168) | 首次把预训练视频生成器 repurpose 为 RL value model；**解决痛点**：VLA 在部分可观测 + 延迟反馈场景下 value estimation 不准；**方法**：以当前观察 + 机器人 proprioception 为输入，联合预测未来 proprioception + 当前状态标量 value；**对比现有 VLM-based value model**：VLM 无时序动态建模 → ViVa 显式利用视频生成器的时空一致性；**对社区含义**：RL post-training（π*0.6/RLT）需要 value 函数，ViVa 提供了通用 value model 路径——可能成为 RL-VLA 标准组件 |
| 174 | SnapFlow: One-Step Action Generation for Flow-Matching VLAs | arXiv 2604.05656 | 2026-04-07 | Arch, Edge | [链接](https://arxiv.org/abs/2604.05656) | **直接攻克 flow-matching VLA 的核心边缘部署瓶颈**——π0/π0.5/SmolVLA 的 10-step ODE denoising 占端到端延迟 80%。**方法**：plug-and-play 自蒸馏，把多步 denoising 压缩为 1-NFE 单步生成；target 是模型自身边际速度计算的 two-step Euler shortcut 速度（避免 conditional velocity 的 trajectory drift）；zero-initialized target-time embedding 让同一网络切换 local velocity 估计 vs global one-step 生成。**关键数据**：**π0.5 (3B) LIBERO 4 套件 40 任务 400 回合 → 98.75% 平均成功（略超 10 步 teacher 的 97.75%）**；端到端延迟 274ms → **83ms（9.6× denoising 加速）**；SmolVLA (500M) MSE −8.3%，端到端 3.56× 加速。**训练成本**：**单 GPU ~12h**，无需外部 teacher、无架构改动。**对社区含义**：可与 layer-distillation、token pruning 正交组合——#43 LiteVLA-Edge 走的是量化路线，SnapFlow 走的是蒸馏路线，两者可叠加实现 Jetson 级反应式控制 |
| 175 | NVIDIA National Robotics Week 2026: NemoClaw + OceanSim + RoboLab + Doosan + mimic-video | NVIDIA Blog | 2026-04-10 | Strategy, Arch | [链接](https://blogs.nvidia.com/blog/national-robotics-week-2026/) | **NVIDIA 国家机器人周汇总**：6 项社区项目落地。**NemoClaw + Isaac Sim**（Umang Chudasama）：自然语言 → Python 脚本 → Isaac Sim Nova Carter 导航，无需写代码；**OceanSim**（密歇根大学）：GPU 加速水下感知仿真，基于物理渲染 + 实时声呐成像，填补水下机器人 sim 空白；**RoboLab**：高保真 generalist policy benchmark，将并入 Isaac Lab-Arena 路线图；**Doosan + Cosmos Reason**：单相机图像推断盒内物品 + 损坏检测，调整放置速度 / 夹爪——智能码垛 vs 固定规则；**Toyota Research Institute**：定制 Cosmos WFM 做 SOTA 动态视图合成 + 遥操数据增强；**mimic-video**（Oier Mees 组 / Mimic Robotics）：预训练视频模型 + flow matching action decoder = **10× 样本效率 + 2× 收敛速度**，真机双臂 Franka + 16-DoF 灵巧手验证。**核心信号**：NVIDIA 正把 Cosmos WFM + Isaac Sim 的 open stack 推成社区默认选择；video-as-backbone（而非 VLM-as-backbone）路线开始走出实验室 |
| 176 | mimic-video: Video-Action Models beyond VLAs | Mimic Robotics / arXiv 2512.15692 | 2026-04 | Arch, Strategy | [链接](https://mimic-video.github.io/) | **对标 VLA 的新路线**：Video-Action Model (VAM)。**论点**：VLA 靠 vision-language 预训练只能捕获 semantic priors、对 physical causality 是盲的 → 换成视频模型 backbone 直接从预训练中吸收"动态与因果"。**方法**：frozen/LoRA Cosmos-Predict2 作为视频骨干，partially denoised 视频 latent → action decoder（机器人 proprioception + 视频 latent + 可选语言 token），flow matching 解码动作 chunk。**真机结果**：Franka 双臂 + 16-DoF 灵巧手，**10× sample efficiency vs VLA，2× 更快收敛**。**对社区含义**：世界模型 / 视频预训练从"感知工具"升格为"策略骨干"；与 π0.7 等语言优先路线形成对立假设——**哪条路线是未来 VLA 的默认起点？**（参考 #173 ViVa 的同源思路） |
| 177 | VLA Foundry: 统一 LLM/VLM/VLA 训练框架 | arXiv 2604.19728 | 2026-04-21 | Arch, Recipe | [链接](https://arxiv.org/abs/2604.19728) | 开源统一训练 stack：**从语言预训练 → VLM 多模态 → VLA action-expert fine-tune 端到端控制**。痛点：此前社区用不同代码库训 LLM（e.g. llama-factory）、VLM（Qwen-VL）、VLA（openpi/lerobot），pipeline 割裂、复现困难。VLA Foundry 提供共享训练栈 + 数据 mix 配方，降低跨阶段切换成本。**对社区含义**：对齐 LeRobot #4 在基础设施层的努力——VLA 不再是"独立子领域"，而是从 LLM 继承的三段式 pipeline 末端；个人/小团队也能跑完整个流程 |
| 178 | VLAJS: Jump-Start RL with VLA Regularization | arXiv 2604.13733 | 2026-04-15 | Arch, Recipe | [链接](https://arxiv.org/abs/2604.13733) | **把 VLA 用作 RL 的高层动作建议而非最终策略**。**痛点**：纯 RL 探索低效（sparse reward）；纯 VLA 泛化上限被 demo 限制（#7 Penn PAL 42.3%）；#142 SimpleVLA-RL 证明 1 轨迹 SFT + RL 可拉到 91.7%。VLAJS 沿同一路线：用 VLA 作为**短期**动作提示源（transient），on-policy RL 做长期优化；VLA 提供的是 regularization 而非直接策略输出。**对比 π*0.6**：π*0.6 需要大规模 coaching + value model；VLAJS 更轻量，适合单任务快速提升。**路线含义**：RL 后训练正分化为"重型（PI, RLT）"和"轻型（VLAJS, SimpleVLA-RL）"两条路径 |
| 179 | HEX: Humanoid-Aligned Experts for Cross-Embodiment Whole-Body Manipulation | arXiv 2604.07993 | 2026-04 | Arch | [链接](https://arxiv.org/abs/2604.07993) | **首个全尺寸双足人形 whole-body VLA 框架**。**痛点**：多数 VLA 把人形各部位独立建模，高 DoF 下不稳定。**方法**：humanoid-aligned universal state representation（canonical body-part 抽象） + Mixture-of-Experts Unified Proprioceptive Predictor 建模全身协调与时序动力学；lightweight history tokens 避免重复编码过往图像。**验证平台**：Tienkung 2.0 / 3.0 人形，whole-body + long-horizon + fast-reaction 任务。**对比 #49 WholebodyVLA**：WholebodyVLA 用 action-free 第一视角视频学 latent action；HEX 走 MoE + canonical state 路线。**对社区含义**：中国人形 VLA 路线（Tienkung、Unitree #91、Galaxea）正在快速积累独立架构选型经验 |
| 180 | COIN: Chain of Interaction Benchmark — VLA 交互推理的真实上限 | arXiv 2604.16886 | 2026-04 | Arch, Strategy | [链接](https://arxiv.org/abs/2604.16886) | **把 VLA 的"推理 vs 执行"鸿沟量化**。**方法**：对比 CodeAsPolicy、语言条件 VLA、H-VLA 三类方法在交互式操作任务上的表现。**核心发现**：**视觉理解 ≠ 运动执行**——模型能"看懂"但不能"做到"，差距比此前 LIBERO/SimplerEnv 显示的更大。**对现有评测的挑战**：和 #10 Moritz Reuss ICLR 2026 综述、#29 VLA-0-Smol（LIBERO 高分主要测记忆力）一致——benchmark 饱和 ≠ 真实能力。**对社区含义**：#146 ICRA 2026 Workshop 10,000h 真机评测 + COIN 类交互 bench = 下一代评估体系；选 benchmark 不要只看 LIBERO |
| 181 | OneVL: One-Step Latent Reasoning and Planning | arXiv 2604.18486 | 2026-04 | Arch | [链接](https://arxiv.org/abs/2604.18486) | **压缩 Embodied CoT 为单步 latent reasoning**。**痛点**：embodied chain-of-thought（#10 ICLR 综述五大趋势之一）自回归推理延迟对实时部署是致命的；#173 ViVa 在 value 侧解决，OneVL 在 reasoning 侧解决。**方法**：统一 VLA + world model 框架，reasoning 走 compact latent tokens（而非 token-by-token 文本），由 dual auxiliary decoder 监督。**应用场景**：当前主要验证自动驾驶 VLA，但方法对通用机器人 ECoT 同样适用。**对社区含义**：latent reasoning 路线开始替代显式 CoT——边缘 VLA 要实时，就不能靠文字链式推理 |
| 182 | 1X Redwood AI + World Model：NEO 家用人形的消费级 VLA | 1X Technologies | 2026-03 | Arch, Edge, Strategy | [链接](https://www.1x.tech/discover/redwood-ai) | **首个发布日期明确的消费级家用人形 VLA**。**Redwood**：vision-language transformer，**160M 参数，NEO 机载 GPU 5Hz 运行**（对比 Pi0.5 3B / GR00T N1.6 需 48GB+ 训练）；端到端 mobile manipulation：开门、取物、导航；训练数据 = 真实遥操 + autonomous episodes。**1X World Model**（2026-03 发布）：物理视频 + prompt → NEO 从"看视频"学新任务，作为 NEO 的 cognitive core。**商业化**：$20K early access，2026 美国首发。**对社区含义**：消费级 VLA 不能照搬研究级大模型——**"小、快、可学"比"大、准、通用"更重要**；与 #43 LiteVLA-Edge 6.6Hz 数据印证——家用场景的技术栈正在独立分化 |
| 183 | LeRobot Worldwide Hackathon: SmolVLA 实测优势 + 冠军配方 | Multiple blogs | 2026-04 | Recipe, Strategy | [链接](https://medium.com/@sarohapranav/folding-shirts-sorting-medicines-and-winning-at-the-lerobot-worldwide-hackathon-98ad4e21c972) | **社区实测 SmolVLA > 其他 VLA**：组织方统计"用 SmolVLA 的队伍在更少 episodes 下平均结果更好"；40 国队伍对比 ACT/π0/SmolVLA。**冠军队 Ez_2_AI（全球 #2、美国 #1）**：叠衣服 + 分药，训练用 π0 + SmolVLA + ACT 组合；hackathon 现场 70% SR，会后换模型调到 85%。**选手画像**：一个 16 岁高中生 + 一个 19 岁大学本科生——**门槛真在降低**（#35 提到 $200-400 SO-101 + 社区教程，新手也能竞赛）。**对策略**：下个 hackathon 选型首选 SmolVLA；#29 VLA-0-Smol 的 consumer-GPU 观察在产品级 hackathon 再次被印证 |
| 184 | Dexora: 开源高 DoF 双臂灵巧 VLA（ICRA 2026） | GitHub — ZZongzheng0918/Dexora | 2026-03 | Arch, Data | [链接](https://github.com/ZZongzheng0918/Dexora) | ICRA 2026 开源 VLA，专攻 **36-DoF 双臂灵巧操作**——罕见的高 DoF 开源标注。**数据集**：12.2K 遥操 episodes / 2.92M frames / 40.5h，采集系统 = Exoskeleton（手臂）+ Vision Pro（灵巧手），手眼一体 hybrid teleop。**对比 #56 TaaS**：TaaS 降低重复动作的人力成本；Dexora 给出高 DoF 场景的数据格式标准。**对社区含义**：下一代 VLA 的瓶颈已经从"够不够"（#170 数据集 23× 爆发）转向"DoF 够不够、手是不是灵巧"——灵巧手 + 双臂数据将成为 2026 下半年核心竞争位 |
| 185 | E-VLA: 事件相机增强 VLA，暗光模糊场景鲁棒 | arXiv 2604.04834 | 2026-04 | Arch, Data | [链接](https://arxiv.org/abs/2604.04834) | **第一个事件相机 + RGB 同步的 VLA**。**痛点**：普通相机在暗光 / 快速运动 / 动态模糊下视觉 degrade；#41 Geonhui Jo 报告"暗光环境 Pi0 仍可执行"是预训练 VLM 的鲁棒性 fallback，并非主动解决。**方法**：DAVIS346 event camera + RGB 同步采集 + open-source 遥操平台，扩展到多光照 + 多任务。**对比 #143 DynamicVLA**：DynamicVLA 靠合成数据 + latent-aware action streaming；E-VLA 靠感知模态扩展。**对社区含义**：家用场景光照条件不受控（#41/#50 Helix 02 工厂可控 vs 家庭厨房），event camera 可能成为 Figure/1X/Unitree 下一代硬件的标配——参考 #70 Modality-Augmented Fine-Tuning 跨模态迁移路线 |
| 186 | GR00T N1.7: EgoScale 20K 小时人类视频 + 首个 dexterity scaling law | NVIDIA (HF Blog) | 2026-04-17 | Arch, Recipe, Strategy | [链接](https://huggingface.co/blog/nvidia/gr00t-n1-7) | **NVIDIA 商业授权开源 3B 参数 VLA**。**架构升级**：Action Cascade 双系统——S2 = Cosmos-Reason2-2B（升级自 N1.6 Cosmos-Reason1）做任务分解 + 多步推理；S1 = 32 层 DiT 做实时去噪。**核心训练突破**：EgoScale 用 **20,854 小时人类第一人称视频**预训练（vs N1.6 仅几千小时遥操数据）；**首次提出 dexterity scaling law**——人类视频 1k→20k 小时使平均任务完成率 >2×。**部署**：4 步去噪 + 单视角推理；商业授权覆盖 Ampere / Hopper / Lovelace / Blackwell / Jetson；material handling / packaging / inspection 工厂可直接生产部署。**升级路径**：从 N1.6 是**drop-in swap**——只换 `--model-path nvidia/GR00T-N1.7`，embodiment config 不变。**对社区含义**：人类视频替代遥操是数据成本的彻底重构——参考 #145 EasyMimic、#49 WholebodyVLA action-free egocentric video 路线，这三者形成同一假设的产业级验证；社区从"GR00T 不能商用"到"现在可以工厂部署"的拐点 |
| 187 | LeRobot Humanoid: $2,500 3D 打印开源双足平台 | HF Blog (VirgileBatto) | 2026-05-21 | Recipe, Edge, Strategy | [链接](https://huggingface.co/blog/VirgileBatto/lerobot-humanoid) | **HF/Pollen 团队首发的低成本开源人形**。**硬件**：约 **$2,500 BOM**（3D 打印结构件 + 现成电子件 + 经济型 actuator）；当前为**双足平台**，上半身后续 roadmap；零件可打印、可替换、可改型。**全栈交付**：5 个 GitHub repo——`lerobot-humanoid-hardware`（BOM/printable parts/wiring/assembly）、`lerobot-humanoid-runtime`（calibration/state I/O/safety）、`lerobot-humanoid-identification`（dataset replay + simulator parameter fitting 缩小 sim-real gap）、`lerobot-legged-zoo`（MJLab 训练环境）、主仓库整合。**设计循环**：simplified robot rep → benchmark motion → 选材 → 真机数据 → identify simulator params；不是"造一个 robot"而是"造一个学习平台"。**对社区含义**：**对比 Unitree G1 ($16K+) / 1X NEO ($20K) / Tesla Optimus**——LeRobot Humanoid 把进入门槛拉低 5-8×，与 SO-101 桌面臂（$200-400）形成连贯的"个人 robotics learning stack"；适合 hackathon / 学术 / hobbyist 验证 VLA 在腿足平台的 sim-to-real |
| 188 | Figure Helix-02 8 小时工厂直播：22K+ 包裹 17 小时无干预 | Figure AI | 2026-05-13 | Strategy | [链接](https://interestingengineering.com/ai-robotics/figure-ai-humanoids-24-hour-autonomous-run) | **首个公开的人形 VLA 长时连续工厂部署**。**场景**：包裹分拣传送带，识别 barcode → 拾取 → 翻面 barcode 朝下 → 放传送带，~3 秒 / 包（匹配普通人类工人速度）。**直播数据**：原定 8 小时 → 实际 17 小时 22,000+ 包裹 → 后续 28,000 包（24h）→ 接近 40h 50,000 包；全程 Helix-02 端到端控制，无 reset 无干预。**架构回顾**：Helix-02 = S0（kHz 平衡）+ S1（200Hz 全身）+ S2（7-9Hz 推理）三层（参考 #50）；S0 替代 109,000 行手写 C++ locomotion 代码，由 1,000+ 小时人类动作数据训出。**质疑点**：部分观察者质疑直播过滤了失败片段——Figure 未公开错误率 / 干预次数 / 失败模式。**对社区含义**：**从"demo 视频"到"产线时钟"的拐点**——长时连续部署是 VLA 工业化的真实门槛，比 LIBERO 99% 更有说服力；与 #11 π*0.6 espresso 5:30am-11:30pm 运行同属新范畴；个人/学术开源 VLA 离此还有距离，但目标已经具象 |
| 189 | NVIDIA Cosmos 3 + Isaac Lab 3.0：World Foundation Model 三档化 | NVIDIA (GTC 2026) | 2026-03 | Arch, Data, Strategy | [链接](https://nvidianews.nvidia.com/news/nvidia-announces-major-release-of-cosmos-world-foundation-models-and-physical-ai-data-tools) | **Cosmos 3 = 统一合成世界生成 + vision reasoning + action 仿真的 WFM**。**三档模型**：Nano（real-time / edge）、Super（baseline）、Ultra（distill 母模型）；总训练 **9000 万亿 tokens** / 20M 小时真实人类交互 + 工业 + robotics + driving 数据。**Physical AI Data Factory Blueprint**：开源参考架构，把算力转化为大规模训练数据；FieldAI / Hexagon Robotics / Linker Vision / Skild AI / Teradyne Robotics 已采用。**产业接入**：1X、AGIBOT、Agility、Agile Robots、Boston Dynamics、Hexagon Robotics、Humanoid、Mentee、NEURA 用 Cosmos + Isaac Sim + Isaac Lab 训练 + 验证下一代人形。**Isaac Lab 3.0**：与 Cosmos 3 同发，专为 sim-to-real 人形流程优化。**对社区含义**：**WFM 路线 vs VLM 路线的产业终局浮现**——#176 mimic-video / Cosmos-Predict2 backbone 路线获 NVIDIA 全栈背书；Cosmos 3 Nano 直接对标边缘 VLA（参考 #171 Cosmos Reason 2 的 2B/8B 双尺寸定位） |
| 190 | VLA-Perf: 推理性能分析框架（Stanford） | arXiv 2602.18397 | 2026-02-20 | Edge | [链接](https://arxiv.org/abs/2602.18397) | **首个 VLA 推理性能 analytical model**。**痛点**：VLA 架构 + 推理系统的组合空间巨大，部署时不知道"我的模型在我的硬件上能跑多快"。**方法**：解析建模任意 VLA × 任意推理系统组合的延迟；从模型设计侧分析 scaling、架构选择、长 context video、异步推理、双系统 pipeline 影响；从部署侧分析 on-device vs edge server vs cloud 的端到端延迟 trade-off（硬件能力 + 网络性能联合决定）。**与 #44 内存带宽分析互补**：#44 是"测出来 75% 延迟在 action gen"；VLA-Perf 是"给定架构和硬件，预测延迟"。**对社区含义**：边缘部署不再凭直觉——选型前可先用 VLA-Perf 做 paper-back 测算，避免买完硬件才发现跑不动 |
| 191 | PhysiFlow: 多脑潜在 flow matching 全身人形 VLA | arXiv 2603.05410 | 2026-03 | Arch, Edge | [链接](https://arxiv.org/abs/2603.05410) | **物理感知人形 whole-body VLA**。**核心数据**：flow matching 推理 **比 DDPM 快 5.3×、比 AR 快 126×**——为 Figure / 1X / Unitree 等真实部署提供具体提速量级。**架构**：multi-brain latent flow matching + robust tracking；与 #149 CoWVLA、#176 mimic-video 同属"video / world model + flow matching action"路线。**对边缘部署的意义**：与 #174 SnapFlow 单步去噪、#43 LiteVLA-Edge 6.6Hz 形成"flow matching 边缘优化三件套"——蒸馏（SnapFlow）/ 量化（LiteVLA）/ 架构选型（PhysiFlow）三条正交路径 |
| 192 | CF-VLA: Coarse-to-Fine Action Generation（83% 真机平均 SR） | arXiv 2604.24622 | 2026-04 | Arch, Recipe | [链接](https://arxiv.org/abs/2604.24622) | **粗到细两阶段动作生成**。**核心数据**：真机平均 **83.0% 成功率**——在不需要海量数据的前提下用粗到细分解换取精度。**动机**：单 head 动作生成在精细任务（peg-in-hole / 装配 / 抓取小物）上分辨率不足；先粗预测大致轨迹，再细化关键段。**对比 #167 Gemini Robotics-ER 1.6 agentic vision**：ER 1.6 在感知端做粗到细（先裁剪放大再读刻度）；CF-VLA 在动作端做粗到细——同一思想跨阶段。**对社区含义**：精细操作不必依赖 RL post-training（#168 RLT），架构层面也有 cheaper 方案 |
| 193 | How VLAs (Really) Work in Open-World Environments | arXiv 2604.21192 | 2026-04 | Strategy | [链接](https://arxiv.org/abs/2604.21192) | **VLA 真实能力的批判性评估**。**方法**：在 BEHAVIOR1K (B1K) 长 horizon 家务任务上评估多个 VLA。**关键发现**：(1) 当前 VLA 能完成许多复杂 sub-task，但**整体成功率在真实开放场景下仍然很低**；(2) 现有"成功率 / 部分得分"指标会**夸大真实能力**，因为基于 progress-agnostic 准则；(3) 距离辅助 / 安全关键场景的可靠部署还非常远。**与 #180 COIN、#162 10 Open Challenges、#10 ICLR 2026 综述呼应**：benchmark 饱和 ≠ 能力达标；下一代评测应纳入 safety + robustness。**对社区含义**：选 benchmark 时除了 LIBERO，应增加 B1K / RoboArena / COIN——单一 benchmark 高分容易误导研究方向 |
| 194 | LIBERO-PRO: 击穿 LIBERO 高分泡沫的扰动评估（v2 更新） | arXiv 2510.03827 | 2026-05-25（更新） | Strategy | [链接](https://arxiv.org/abs/2510.03827) | **专门暴露 LIBERO 记忆型高分的扩展 benchmark**。**核心扰动维度**：(1) 操作物体、(2) 初始状态、(3) 任务指令、(4) 环境四类合理扰动。**关键数据**：标准 LIBERO **>90% 准确率的模型在 LIBERO-PRO 下性能崩塌**——暴露当前 VLA 依赖 rote memorization 而非真任务理解。**与 #29 VLA-0-Smol 论断一致**：#29 已经指出"LIBERO 高分主要测记忆力"；LIBERO-PRO 是这一论断的 quantitative confirmation。**5 月更新含义**：作者持续迭代评估协议——选 benchmark 时除报 LIBERO 主指标，应同时报 LIBERO-PRO / LIBERO-Plus 才有说服力 |
| 195 | MotuBrain: World Action Model 长 horizon 真机控制配方 | arXiv 2604.27792 | 2026-04 | Arch, Recipe | [链接](https://arxiv.org/abs/2604.27792) | **专为 long-horizon real-world control 设计的 World Action Model**。**四项设计**：(1) 统一 multiview modeling、(2) 独立 text stream 强化 language-action coupling、(3) shared cross-embodiment action representation、(4) 完整 **post-training + deployment recipe**。**对社区含义**：与 #176 mimic-video、#161 Real-is-Sim 同属"world model 作为策略骨干"路线；MotuBrain 进一步把 cross-embodiment 和 post-training 配方化——填补 #170 数据爆发后"有数据但不知道怎么训"的空白 |
| 196 | Hand-in-the-Loop (HandITL): 灵巧操作的无缝 HITL 介入 | arXiv 2605.15157 | 2026-05 | Recipe, Debug | [链接](https://arxiv.org/abs/2605.15157) | **专攻灵巧操作的 human-in-the-loop**。**痛点**：高维 action space + contact-rich dynamics 会放大 VLA 微小偏差 → compounding error；普通 HITL 在双臂灵巧时会出现 gesture jump。**方法**：seamless blending——人类纠正意图与自主策略**连续**融合，无突变手势。**与 #20 HIL-SERL 区别**：HIL-SERL 是 RL 训练阶段的 HITL；HandITL 是**部署阶段**的 HITL——data + deployment 两个阶段都需要 HITL。**对社区含义**：高 DoF 双臂（#184 Dexora、#179 HEX）部署需要新的 HITL 方法——HandITL 提供了首个可参考实现 |
| 197 | FASTER: 重新思考实时 flow VLA | arXiv 2603.19199 | 2026-03 | Arch, Edge | [链接](https://arxiv.org/abs/2603.19199) | **针对 flow matching VLA 的实时性架构重设计**。**论点**：当前 diffusion / flow matching action chunking 在仿真上 SOTA，但 sim → real 时 real-time constraint 是生死线。**与 #174 SnapFlow / #150 KAN-We-Flow / #191 PhysiFlow 路线对比**：SnapFlow 蒸馏多步为单步；KAN-We-Flow 用 KAN+RWKV 极轻量化；PhysiFlow 全身物理感知；FASTER 在架构层面重设计，与三者正交。**对社区含义**：边缘 VLA 不是单一 silver bullet——蒸馏 / 量化 / 替换 backbone / 重设计 head 应组合使用 |
| 198 | LeRobot v0.6.0: Imagine, Evaluate, Improve | HF Blog | 2026-07-07 | Arch, Data, Recipe | [链接](https://huggingface.co/blog/lerobot-release-v060) | **闭环补齐：想象 → 评估 → 改进**。**三个 world model policy**：VLA-JEPA（Qwen3-VL-2B，训练时 JEPA world model 从自身动作预测未来帧，**推理时 world model 直接消失 → 零额外推理开销**，含 DROID 预训练 base）；LingBot-VA（自回归 video-action，逐 chunk 同时预测未来视频和动作，`--policy.save_predicted_video=true` 可对比"想象 vs 实际"，单卡 24–32GB 推理）；FastWAM（~5B 视频生成 expert + 紧凑 action expert，推理时跳过 dreaming 直接去噪动作）。**新 VLA**：GR00T N1.7（**已替换 N1.5，需要 N1.5 必须 pin `lerobot==0.5.1`**）、MolmoAct2（bf16 推理 ~12GB / LoRA 单卡 24GB / SO-100-101 零样本 checkpoint 内置标定修正）、EO-1（Qwen2.5-VL-3B + flow matching）、Multitask DiT（~450M，TRI LBM 配方）、EVO1（0.77B，InternVL3-1B + flow matching，内置 RTC）。**六个新仿真 benchmark 统一 `lerobot-eval`**：LIBERO-plus（~10,000 扰动变体 / 7 轴）、RoboTwin 2.0（50 双臂任务 + 10 万+ 轨迹）、RoboCasa365（365 厨房任务 / 2,500 程序化厨房）、RoboCerebra（3–6 子目标长 horizon + 6,660 episode）、RoboMME（记忆考试 16 任务 / 4 套件）、VLABench；连同 LIBERO / Meta-World / IsaacLab-Arena 共九个 benchmark 家族。**`lerobot-rollout` 部署 CLI**：base / sentry / highlight / episodic / **dagger**（脚踏板或按键接管，纠正帧打 `intervention` flag，直接可用于下一轮微调）。另有 FSDP、HF Jobs 云训练（`--job.target`）、RealSense depth 端到端、`lerobot-annotate` VLM 自动语言标注、数据加载最高 2×（子集加载 275s → 0.06s）、基础依赖减少 ~40%、PyTorch 2.7–2.11 |
| 199 | LeRobot 官方算力硬件指南：VRAM 分档 + 实测训练时长 | HF Docs (LeRobot v0.6.0) | 2026-07 | Recipe, Edge | [链接](https://huggingface.co/docs/lerobot/v0.6.0/hardware_guide) | **回答新手最常问的两个问题：要什么 GPU、要跑多久**。**VRAM 分组（BS 8, AdamW）**：Light BC（act/vqbet/tdmpc）~2–6GB；Diffusion / multi_task_dit ~8–14GB；SmolVLA ~10–16GB；Large VLA（pi0 / pi0_fast / pi05 / xvla / wall_x）~24–40GB；Multimodal（groot / eo1）~24–40GB。AdamW 优化器状态比纯前反向多占 ~30–100%。**收敛经验法则：模仿学习通常 5–10 个 epoch 收敛，不是几十万 raw steps**。**4×H100 实测**（5000 steps / BS 32 / ~50 episodes / 640×480）：diffusion 16m17s（update_s 0.167）、smolvla 27m49s（0.312，unfreeze vision encoder）、pi05 3h41m（2.548，需 `gradient_checkpointing=true` + `dtype=bfloat16`）。**最有用的诊断指标**：`dataloading_s` 逼近 `update_s` 时**再加 GPU 没有用**——先查 `--num_workers`、图像分辨率、磁盘速度。单机参考：4090 跑 act 5 epoch ≈ 30–60min；A100 40G 跑 pi0/pi05 ≈ 4–8h；M1/M2/M3 Max（MPS）跑 act ≈ 6–14h。**隐蔽的坑**：缩短训练步数必须同步缩短 LR schedule（`--policy.scheduler_decay_steps ≈ --steps`），否则 LR 一直停在峰值不衰减 |
| 200 | LingBot-VLA 2.0：6B 开源 + 55 维统一动作空间 + 6 万小时数据 | Robbyant (蚂蚁集团) | 2026-07-08 | Arch, Data | [链接](https://www.marktechpost.com/2026/07/08/lingbot-vla-2-0/) | **#38 LingBot-VLA 1.0 的正式迭代，Apache-2.0 + 6B checkpoint + 技术报告全开**。Qwen3-VL-4B-Instruct backbone，**RTX 4090D 单次推理 ~130ms（10 denoising steps）**。**数据规模与过滤规则**：精选 ~60,000h（50,000h 机器人轨迹跨 **20 种机器人构型** + 10,000h 第一人称人类视频），原始池 90,000h + 20,000h；过滤是可量化的——每 embodiment 计算三阶 jerk + 速度/加速度 Z-score，**异常平滑度或 >95% 静止的 episode 直接丢弃**，再用各机器人 URDF 回放校验视频，人工去除模糊/遮挡/丢帧/多视角错位；标注用 Qwen3.6-27B 切分时序连续子任务 + 18 类原子动作闭词表。**55 维统一动作空间**：臂关节 14 + 末端位姿 14 + 夹爪 2 + 手关节 12 + 腰 4 + 头 2 + 移动信号 3 + 保留 4，缺失部件直接 padding。**MoE action expert**：auxiliary-loss-free 路由（DeepSeek-V3 式 sigmoid + per-expert bias 纠正负载不均），同 active 参数下 loss 与验证动作误差均低于 dense。**dual-query 蒸馏**：Qt（当前观测）+ Qt+T（未来观测，T = action chunk size），由 LingBot-Depth 教几何、DINO-Video 教时序语义。**GM-100 结果**（progress/success）：AgileX Cobot Magic 66.2/34.4（π0.5 59.1/32.2，GR00T N1.7 36.3/17.8）、Galaxea R1Pro 34.6/15.6；长 horizon 移动操作 ID 与 OOD 全面超 π0.5（冰箱分拣 OOD 37.0/13.3 vs 30.3/6.7）。**诚实缺陷**：progress 与 success 之间仍有明显 gap → 失败集中在最后的精确放置/松手一步。后训练直接吃 LeRobot v2.1/v3.0 数据集 |
| 201 | Gemini Robotics 2：全身控制 + 多机协作 + On-Device 2（<200 示例适配新本体） | Google DeepMind | 2026-07-30 | Arch, Strategy | [链接](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/) | **三模型分工**：Gemini Robotics 2（VLA，首次控制**整个人形——从脚到指尖**）、Gemini Robotics ER 2（embodied reasoning VLM，做高层大脑 + 多步任务编排 + **首次支持多机器人协作**）、Gemini Robotics On-Device 2（本地运行）。**同一个 checkpoint 控三种本体**：Apollo 2 + SharpaWave 手、Apollo 2 + Inspire 手、Franka Duo + Robotiq 夹爪。**难得公布了失败率的实测数字**：全身操作（Apollo + Inspire）桌面拾取 68.4% / 地面拾取 45.7% / 货架拾取 76.3%；夹爪灵巧（Franka Duo）通用抓放 74.2% / 工具配套 78.9% / 精密插入 89.6%；**多指灵巧（22-DoF SharpaWave）明显最弱**——拧灯泡 36%、拧下灯泡 92%、系垃圾袋 44%、簸箕 32%、密封袋 40%，官方直言 "multi-finger dexterous manipulation remains challenging"。**On-Device 2**：继承 GR 1.5 的 motion transfer，**新双臂本体几小时适配、通常 <200 条示例**，演示平台包含 Dexmate、**SO101**、Trossen。安全侧发布 **ASIMOV-Agentic** benchmark（测 ER agent 拒绝来自 VLA 的不安全 tool call、预判任务不可行并主动请求人类介入）。**对社区含义**：(1) SO101 出现在 Google 官方 on-device 适配矩阵里——消费级臂已进入前沿模型的一等公民名单；(2) 多指灵巧仅 32–44% 印证 #196 HandITL 这类部署期 HITL 仍是刚需；(3) 与 #201 之前的 #167 ER 1.6 相比，本次把"推理模型"和"动作模型"的分工彻底产品化（ER 2 已上 AI Studio，VLA/On-Device 仅早期合作伙伴） |
| 202 | Pretrained to Imagine, Fine-Tuned to Act：WAM 路线全景与选型 | NVIDIA Developer Blog (Moritz Reuss) | 2026-06 | Strategy, Arch | [链接](https://developer.nvidia.com/blog/pretrained-to-imagine-fine-tuned-to-act-the-rise-of-world-action-models/) | **#10 ICLR 综述同一作者对 World-Action Model 路线的系统梳理**。核心论断：**WAM 会成为机器人基础模型的第二条主流配方，最终赢家很可能既非纯 VLA 也非纯 WAM，而是二者的混合**。**三轴分类法**（不完美但够用）：paradigm（inverse dynamics / joint prediction / representation-only）、action integration（动作怎么进模型）、backbone。**最重要的实证**：DreamZero 与 FastWAM 都发现——**机器人微调阶段，动作学习必须与视频预测目标 co-train 才最好**，不是"先视频预训练完再纯动作微调"。**Veo 3.1 零样本实验**：给一帧初始画面 + 语言指令，前沿视频模型对机器人-物体交互已展现可用先验，但**远未可靠到能直接用于控制**——WAM 微调正是把这种"想象"变成"可靠控制"的尝试。**给用户的决策含义**：作者明确指出以这些模型的运行成本，**大多数团队必须在 VLA 与 WAM 之间二选一，不要两条腿同时跑**。与 #176 mimic-video、#189 Cosmos 3、#195 MotuBrain、#198 VLA-JEPA/FastWAM/LingBot-VA 属同一条线；#207 leaderboard 上 DreamZero 稳居前列是这条路线的第一批外部证据 |
| 203 | vla.cpp：首个原生服务 flow matching VLA 的 ggml 级推理运行时 | arXiv 2606.08094 | 2026-06-06 | Edge | [链接](https://arxiv.org/abs/2606.08094) | **VLA 部署终于有了 llama.cpp 式的答案**。**痛点**：VLA policy 一般以 Python/PyTorch 栈发布，默认工作站级 GPU——与机器人实际算力严重错配。**贡献**：基于 llama.cpp 的便携 C++ 运行时，**首个原生服务 flow-matching / diffusion VLA 推理模式的 ggml 级引擎**——缓存的 vision-language prefix 被 cross-attending action expert 沿若干 solver steps 积分消费。**一个运行时服务 7 种架构**（跨 5 类 backbone × 4 类 action head），统一 request/response 协议，每个模型打包成自包含 bundle。**关键数据**：LIBERO-Object 上与 SOTA checkpoint 的差距 **≤ 1 个 episode（200 局中）**；**BitVLA 在 1.3 GiB 内存下跑到 100% 成功率**；**同一 bundle 不做任何改动跨三档硬件运行**——从消费级 GPU 一直到 8GB 嵌入式模组。**与 #190 VLA-Perf 互补**：VLA-Perf 帮你算"跑不跑得动"，vla.cpp 让它真的跑起来；与 #43 LiteVLA-Edge、#174 SnapFlow 的区别是——**它不改模型，改的是运行时** |
| 204 | Embodied.cpp：同时服务 VLA 与 WAM 的异构机器人推理运行时 | arXiv 2607.02501 (SEU-PAISys, Apache-2.0) | 2026-07-02 | Edge, Arch | [链接](https://arxiv.org/abs/2607.02501) | **比 #203 vla.cpp 覆盖面更宽：VLA 和 WAM 一起服务**。**核心论点**：现有推理运行时都是为 request-response serving 设计的，**不满足具身部署的运行时契约**——(1) 闭环控制内部的**多速率执行**、(2) 异构硬件上的 **latency-first batch-1 推理**、(3) 超出固定 token I/O 的**可扩展具身接口**。**五层架构**：input adapters → sequence builders → backbone execution → head plugins → deployment adapters；一套 backend 抽象打通设备、机器人与仿真器。**选型建议**：只跑 VLA 且追求极致轻量/单文件分发 → #203 vla.cpp；要同时跑 WAM（#202 路线）或需要多速率闭环 → Embodied.cpp。**信号意义**：一个月内出现两个 C++ 具身推理运行时——**边缘部署正在从"每家自己写 glue code"走向标准化基础设施**，这是生态成熟的典型标志 |
| 205 | RhinoVLA：token 效率决定边缘延迟，Huixi R1 上 11.69 Hz | arXiv 2606.07383 | 2026-06 | Edge, Arch | [链接](https://arxiv.org/html/2606.07383) | **与 Huixi R1 边缘 SoC 协同设计的部署导向 VLA**。**最有价值的是延迟归因**：Jetson AGX Orin 上 VLM backbone + Action Expert **合计占总运行时 >90%**；operator 级拆解进一步显示，大部分 VLM 延迟来自 **MLP projection 算子**——这些投影 GEMM 主导，隐藏维固定时 **FLOPs 随 visual + context token 数线性增长**。**由此得出的选型规则**：**参数量相近的两个 VLA，token 效率才决定部署延迟**——挑 backbone 不能只看参数量，要看它吐多少 token。**方法**：token-efficient Qwen3-VL backbone + continuous Action Expert；跨机器人统一接口 = View Registry + **72D 物理 state-action slot 空间 + 二值 mask** + robot-instance LoRA（吸收同型号机器人之间的个体残差）；部署侧用硬件感知编译 + 混合精度 + 并行视觉编码。**结果**：同参数量级下下游性能与 π0.5 相当，**Huixi R1 上端到端 11.69 Hz**，越过 10Hz 实时闭环门槛。**与 #200 对照**：LingBot 55D、RhinoVLA 72D——业界正在收敛到"统一 slot 空间 + mask + 每机器人 LoRA"这一范式 |
| 206 | Realtime-VLA FLASH：把投机解码搬进 diffusion VLA，3.04× 加速 | arXiv 2605.13778 (中科院计算所 + Dexmal) | 2026-05 | Edge, Arch | [链接](https://arxiv.org/pdf/2605.13778) | **把 LLM 的 speculative decoding 迁移到 dVLA**。**机制**：轻量 draft model 先出动作，由主模型的 **Action Expert 并行验证**，再加 phase-aware fallback——不可靠时退回完整推理管线，因此不牺牲可靠性。**LIBERO 实测**：把大量 **58.0ms 的完整推理轮次替换为最快 7.8ms 的投机轮次**，任务级平均推理延迟降到 **19.1ms（3.04× 加速）**，任务性能基本保持。真机验证：传送带分拣（延迟敏感场景）。**与其他提速路线的正交性（重要）**：#174 SnapFlow 蒸馏减少去噪步数、#197 FASTER 重设计架构、#191 PhysiFlow 换 action 生成范式、#203/#204 换运行时——**FLASH 不改模型只改推理调度，现有 checkpoint 可直接受益，是迁移成本最低的一条**。边缘优化应当组合使用而非择一 |
| 207 | 机器人 leaderboard 现状：RoboArena 被刷榜 + π0.5 的惊人粘性 | It Can Think (Chris Paxton) | 2026-06-13 | Strategy | [链接](https://itcanthink.substack.com/p/what-do-robotics-leaderboards-tell) | **"机器人领域至今没有 Chatbot Arena"**。**RoboArena 已出现刷榜**：其志愿者配对评估机制被钻空子——有队伍报名评估却不完成分配的评测；RoboArena 因此在榜上新增 **low-sample 标记**。**三个仍在更新的 leaderboard 对比**：**RoboLab-120**（NVIDIA 内部执行全部评测，**不可刷但入选模型少**——Cosmos3-Nano-Policy 上线后把 π0.5 与 DreamZero 挤下前二；无 Spirit 等外部模型）；**RoboArena**（真机分布式，模型多但单模型置信度低）；**MolmoSpaces**（含移动机器人、任务面更宽，但纯仿真且渲染质量较低，榜上出现 TiPToP 这类 TAMP 混合方法）。**两个值得记住的观察**：(1) 与 LLM 不同，**老模型 π0.5 粘性极强**，多榜稳居前列——作者归因于数万小时高质量真机数据，新入场者短期补不上；(2) 顶部排序仍在剧烈波动，**"现在谁都还可能赢"**，中国创业公司 Spirit（v1.5/v1.6 开源）已冲到 RoboArena 榜首附近。**对读者的实操含义**：论文只报单一 leaderboard 名次基本没有信息量（呼应 #193 #194 #180）；自建评测优先选"主办方统一执行"的闭环 benchmark，其次才是众包真机榜 |
| 208 | Figure 03 进 BMW Spartanburg 物流线：84 秒节拍 + >99% 放置成功率 | BMW Group / Figure AI | 2026-06-25 | Strategy | [链接](https://www.press.bmwgroup.com/global/article/detail/T0458778EN/bmw-group-advances-the-use-of-physical-ai-in-production-with-figure-03-project-in-spartanburg?language=en) | **从"能跑多久"升级到"按别人的节拍跑"**。**演进**：2025 年 Figure 02 在车身车间做 10 个月试点（为 30,000+ 台 X3 插板件焊接）→ 2026 年 Figure 03 转做**物流 sequencing**：从无序大料箱中取件 → 放入 sequencing 小车 → just-in-sequence 送到装配工位。**公布的产线 KPI（比 demo 视频有价值得多）**：每周一至五 **10 小时/班**，目标节拍 **84 秒**，装载时间 **37 秒**，**每班放置成功率 >99%**，目标**零人工干预**。**Figure 03 硬件升级**：无线充电、触觉手部传感、掌心相机、speech-to-speech 音频。**与 #188 Helix-02 17h 直播的关键区别**：#188 回答"能连续自主跑多久"，本条回答"**能不能嵌进别人已有的产线节拍和验收标准**"——后者才是工业化真正的门槛。**值得注意的反向读法**：84 秒节拍中装载仅 37 秒，意味着**近一半时间花在移动而非操作**——人形在物流场景的瓶颈可能是移动效率而不是操作能力 |
| 209 | Robometer + TOPReward：奖励模型进入主流工具链，成功检测终于自动化 | HF Blog / RSS 2026 | 2026-07 | Recipe, Data | [链接](https://huggingface.co/docs/lerobot/v0.6.0/robometer) | **LeRobot v0.6.0 把 reward model 提升为与 policy 平级的一等 API**（`lerobot.rewards`），四个模型统一接口：HIL-SERL reward classifier、SARM、Robometer、TOPReward。**Robometer-4B**（RSS 2026 / arXiv 2603.02115）：Qwen3-VL-4B 底座，**用 100 万+ 条机器人轨迹做轨迹对比训练**；指向任意 LeRobot 数据集，即可从原始视频 + 语言指令直接打出 task progress 与 success，**无需任何任务特定训练**。**TOPReward**：完全零样本、**没有任何 reward 权重**——包一个现成 VLM（Qwen3-VL），读它在给定轨迹视频 + 任务指令下输出 token `"True"` 的 log-probability。**任何足够强的 VLM 都能变成 reward function**。两者都附带 labeling 脚本，把逐帧 progress 曲线写回数据集，可直接用于 **reward-aware behavior cloning (RA-BC)**、数据集质量检查和 progress 叠加视频。**对社区含义**：#16 ML6 报告过"loss 完全不反映真实成功率"、#193 批评"progress-agnostic 指标夸大真实能力"、#10 指出"OXE 数据质量差但没人量化"——Robometer/TOPReward 是这三个长期痛点第一批可直接落地的工具 |
| 210 | SmolVLA 微调失败的社区对质：先用 ACT 验数据，再谈加数据 | GitHub Issue (huggingface/lerobot #2915) | 2026-06 | Recipe, Debug | [链接](https://github.com/huggingface/lerobot/issues/2915) | **典型失败现场**：用户在 SO-101 上先跑预训练 SmolVLA 推理"效果很差，庆幸没把机器人搞坏"，再用自己的 **120 episodes 移动方块数据微调仍学不会**，且明确表示"**数据量增加看不到任何进展**"。**社区有成功案例者（masato-ka）给出的诊断顺序**：(1) **单点位 pick-and-place 30 episodes 以内就应该够**——120 条还不行说明问题不在数据量；(2) **先用 ACT 在同一数据集上验证**，ACT 也跑不通就是数据的问题、不是模型的问题；(3) 其成功配置：SmolVLA / 70 episodes / batch 8 / 40k steps / A100 约 180 分钟；另一例 90 episodes（3 任务 × 30）/ 40k steps / lerobot 0.4.2。**可直接抄的规则**：**SmolVLA 学不会时，先降级到 ACT 做数据质检，再回来加数据**——顺序反了会白白浪费几周。呼应 #17（一致性 > 数量：75 clean > 81 mixed）、#6/#30（stats.json 不匹配是静默杀手）、#16（loss 不反映成功率）、#209（现在可以用 Robometer 自动做这个质检） |
| 211 | LeRobot v0.6.1：`lerobot.types` 改名的破坏性变更 + 一批静默 bug 修复 | GitHub Release | 2026-08-03 | Debug, Recipe | [链接](https://github.com/huggingface/lerobot/releases) | **升级前必读的破坏性变更**：`lerobot.types` 模块更名为 `lerobot.lerobot_types`（#4232）——照抄旧教程的 import 会直接失败。**架构性清理（降低"版本地狱"风险）**：wall-x 改为继承 Transformers 原生 Qwen2.5-VL、X-VLA 改为继承原生 Florence-2，不再 vendoring（#4035 / #4089）；pi0 / pi05 / pi0_fast / eo1 统一走 shared VLA components（#4055 / #4061 / #4062 / #4063）——这正面回应了 D151「transformers 版本不兼容导致 Pi0.5 动作失控」那一类静默故障。**实操向修复**：`lerobot-record` 先连 teleoperator 再连 robot，避免 watchdog 跳变（#4166）；深度帧不再被误做 RGB→BGR 处理（#4135，RealSense 用户）；episode reset 时清空视频帧暂存（#3683）；`lerobot-eval` 结束后恢复 policy 的 training 模式（#4162）；SO follower 的 P 系数可配置（#4142）；GR00T 1.7 LIBERO checkpoint 更新（#3961）；VLM 子任务标注改进（可读 contact sheet、seeded relabeling、自托管 vLLM 配方，#3896）；diffusers 收紧到 `>=0.38.0,<0.40.0`（#4145） |
| 212 | Grabette / Gripette：490€ 手持夹爪 + 120€ 机器人孪生，UMI 路线的开源工程化 | HF Blog (Pollen Robotics) | 2026-07-21 | Data | [链接](https://huggingface.co/blog/grabette) | **「不用机器人也能采机器人数据」的可复制版本**（直接致敬 Stanford UMI）。**硬件 BOM**：Grabette 手持采集器 **~490€**（Raspberry Pi + 标准 Pi 摄像头 + 现成 OAK-D 深度相机 + 磁编码器 + IMU）；Gripette 机器人端孪生夹爪 **~120€**（相机 + 两个舵机）。**双相机分工是关键设计**：便宜的广角鱼眼给 policy 提供 wrist-cam 式的上下文视图，RGBD 相机专门负责鲁棒 6-DoF 跟踪——两个角色刻意分开。**流程**：按钮开始录制（观察相机 + 跟踪相机 color/depth/IMU + 夹爪编码器**共用一个时钟同步**）→ 浏览器 dashboard 一键处理 → 上传 HF Hub → `grabette-slam` Space 用 **RTAB-MAP** 做 SLAM 并**自动校验轨迹是否有跳变 / 跟丢** → 转成标准 LeRobot 数据集。**机器人无关**：demo 存为 camera-local 6-DoF 笛卡尔位姿 + 夹爪状态，不假设任何特定手臂。**参考下游栈**：stock LeRobot 训练 + OpenArm 评估。CAD / 树莓派固件 / 处理管线全开源。同类的 MEgo（智元）、DAS（Genrobot）、Sunday Robotics 采集手套均为闭源 |
| 213 | NVIDIA Isaac GR00T 开发平台 + GR00T 1.7：从遥操到部署的完整可复制配方 | NVIDIA Developer Blog | 2026-07-07（2026-08-06 更新）| Recipe, Arch | [链接](https://developer.nvidia.com/blog/develop-humanoid-robot-policies-end-to-end-with-nvidia-isaac-gr00t/) | **GR00T 1.7 规格**：**3B** base checkpoint，**Apache 2.0**（首个开放且可商用的通用人形 VLA）；预训练 **~32K 小时真实演示 + 人类第一人称数据，外加 ~8K 小时仿真 rollout（BEHAVIOR / RoboCasa / Simulated GR-1）**；backbone 换成 **Cosmos-Reason2-2B（Qwen3-VL 架构），替代 N1.6 的 Eagle**——支持灵活分辨率、按原生宽高比编码不做 padding；新增 ONNX + TensorRT 全流程导出。**相对 N1.6 的增益**：DROID-F0 **+10%**、DROID-F6 **+61%**、SimplerEnv Bridge +5%、Fractal +2%。**最容易被忽略的坑（全身控制场景）**：遥操时 **AGILE WBC + PinkIK 产生的关节空间目标就是 policy 的训练信号**——换一个控制器，同样的动作会得到完全不同的训练分布，所以 WBC 必须在采数据之前就定死。**单卡微调配方**：400 条轨迹（**分多次采集拼起来，不是一次采完**），`--global-batch-size 12 --max-steps 20000 --num-gpus 1 --save-steps 5000 --no-tune-llm --tune-visual --tune-projector --tune-diffusion-model --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08`（冻结 LLM，只调视觉 backbone / projector / diffusion）。**评估纪律**：`--num_steps 600` 只是一次 smoke test（≈1 episode），要拿到有代表性的成功率必须 `--num_episodes 100`（1000 更稳），并加 `--num_envs 5` 并行以增加随机化。数据格式 HDF5 → LeRobot 由 yaml 驱动（`fps: 50`, `chunks_size: 1000`） |
| 214 | Isaac Teleop + GR00T 1.7 进 LeRobot：SO-101 真机训练命令行全公开 | HF Blog (NVIDIA) | 2026-07-07 | Recipe, Edge | [链接](https://huggingface.co/blog/nvidia/nvidia-isaac-teleop-and-gr00t17-in-lerobot) | **N1.5 在 LeRobot 中已被 1.7 取代且不再受支持**（要继续用 N1.5 必须 pin `lerobot==0.5.1`，见 #198）。**SO-101 真机配方**：`--dataset.num_episodes=50`；训练 `--policy.type=groot --policy.base_model_path=nvidia/GR00T-N1.7-3B --policy.embodiment_tag=new_embodiment --policy.chunk_size=16 --policy.n_action_steps=16 --policy.use_relative_actions=true --policy.relative_exclude_joints='["gripper"]' --policy.use_bf16=true --batch_size=64 --steps=20000 --eval_steps=0`，多卡加 `--multi_gpu`。**两个值得注意的默认值**：(1) **相对动作默认开启，但夹爪必须排除**——夹爪是绝对量，跟着做 delta 会坏掉（呼应 #29「相对动作 > 绝对动作」与 #32 state-relative actions）；(2) **推理时 `n_action_steps=8`，比训练时的 16 短**——即只执行一半 chunk 就重规划。**相机**：wrist + front 双路 OpenCV，640×480@30fps，`fourcc: MJPG`。**遥操两条路**：VR 头显（Isaac Teleop）或 SO-101 leader 臂。**DGX Spark 用户需装 CUDA 13 版本的 torch** |
| 215 | GS-VLA：相机装歪一点点，LIBERO 成功率从 ~90% 掉到 ~10% | arXiv 2608.19066 | 2026-08-19 | Debug, Arch | [链接](https://arxiv.org/abs/2608.19066) | **本期对读者最有用的一个数字**。**痛点量化**：当前 VLA 隐含假设「训练与部署的相机配置完全一致」；论文实测**相机支架的微小位移就能把 LIBERO 成功率从约 90% 打到最坏情况约 10%**——这给社区反复念叨的"相机 POV 一定要固定"（#5 标定文件丢失、#16 用 gaussian splatting 救相机不稳、#17 dual-cam 协议、D5）第一次装上了刻度。**方案**：把视角漂移重述为**局部新视角合成**问题——在"相机扰动相对工作空间只在小范围内"的 Locality 假设下，视角归一化退化成与场景/策略都无关的**去遮挡**任务；用一个 **4M 参数的 3D Gaussian canonicalizer 前置在冻结的 VLA 之前，完全不改 policy 权重**。在策略架构、未见任务集、扰动尺度三个正交轴上均有提升。**对读者的含义**：策略"昨天还好好的今天就废了"，**先量相机再怀疑模型**；重训与生成式数据增强都很贵且有灾难性遗忘风险，轻量视觉前置模块是更便宜的一档 |
| 216 | ReflexVLA + ReflexBench：把"推理延迟"变成可配置的评测维度 | arXiv 2608.14379 | 2026-08-14 | Edge, Arch | [链接](https://arxiv.org/abs/2608.14379) | **补上"静态 benchmark 测不出来"的那一块**。**论点**：现有 benchmark 基本只评静态操作任务上的泛化，几乎不覆盖**动态交互**场景。**ReflexBench**：6 个动态任务，评估框架**把 simulator stepping 与 robot control 解耦**，并支持**在同步 / 异步推理下配置延迟**——这意味着社区终于能在仿真里定量复现 D13 / D22 反复出现的"推理延迟是隐形杀手"。**ReflexVLA**：不依赖大规模机器人数据预训练；在视觉 backbone 内做 latent future prediction + 多帧时序融合以增强时间推理；**部署侧靠 batched visual encoding + CUDA Graph replay 压延迟**——这两个是可以直接搬进自己推理栈的工程手段。静态 benchmark 精度不掉，另有真机实验。项目页 `reflexvla.github.io` |
| 217 | The Embodiment Gap：换一台机器人到底还剩多少活要干 | arXiv 2608.18433（TMLR, 2026-08）| 2026-08-19 | Strategy | [链接](https://arxiv.org/abs/2608.18433) | **给"跨具身"祛魅的综述**。**核心概念**：模型能泛化 ≠ 能在你这台机器上跑。作者把"可复用的模型 / 表征 / 数据"与"在目标机器人上真正执行"之间的差距命名为 **embodiment gap**，并指出**不同方法所需的适配工作量差别很大**，而这恰恰决定实际部署成本。**结构**：用「共享结构的类型 × 需要适配的阶段」两轴地图安置现有方法，再按三条交叠方向梳理——共享语义与感知、共享机器人数据与接口、跨具身对应关系学习。**最有实操价值的部分**：提出一套**适配工作量的报告框架**——因为**只报成功率根本看不出这些工作**，对比 cross-embodiment 方法时应同时检查"换到新机器人还剩几步要做"。32 页，TMLR 2026 年 8 月刊。**对读者的含义**：选型该问的不是"这个模型 LIBERO 多少分"，而是"接到我的本体上要重做哪几步" |
| 218 | FACT：VLA 在接触密集任务上失败的两个根因（近 2,500 次真机 rollout） | arXiv 2608.01402 | 2026-08-02 | Debug, Recipe | [链接](https://arxiv.org/abs/2608.01402) | **不是又一个"加力控模块"，而是先做根因分析**。**论点**：既有工作多用 force-augmented 架构或训练期正则去"压住"接触失败，根因一直没被拆开。**两个不同的失败模式**：(1) **精度失败**源自 **flow-matching policy 的训练失配**；(2) **力失败**源自**力信号本身的结构特殊性**。各配一个针对性机制，合成 FACT。**数字**：5 个接触密集任务上平均成功率 **66% vs 最好基线 41%**，评测覆盖**近 2,500 次真机 rollout**——这个规模在 VLA 论文里少见，可信度高于纯仿真结论。**对读者的含义**：peg-in-hole / 插拔 / 拧螺丝这类任务反复失败时，**先分清是"位置精度不够"还是"力信号没学好"**，两者的修法完全不同 |
| 219 | G0.5：单一自回归流同时出推理与动作，真机微调 76.7% vs π0.5 53.3% | arXiv 2608.11739 | 2026-08-12 | Arch | [链接](https://arxiv.org/abs/2608.11739) | **对"VLM + 独立 flow matching action expert"主流配方的正面挑战**。**论点**：主流做法让 VLM 沦为 context encoder 而非决策者；G0.5 用**单个 transformer decoder 在同一个目标下同时发出 reasoning token 和 action token**。**三个让它在基础模型规模上可行的组件**：可学习的**跨具身 action tokenizer**（把异构机器人动作映射到共享词表）、**原生 CoT 流**（任务分解 / 物体 grounding / 动作提示与 action token 交错）、通过视觉编码器注入**数秒历史**的 visual memory。**因为推理与动作共享同一套权重**，预训练 VLM 的能力直接迁移到物理行为——**改 prompt 就能直接调节动作粒度、任务时长和 OOD 场景处理，无需再训练**。**7 个 regime 的数字**：R1lite / R1pro 真机微调 **76.7%（π0.5 53.3%，GR00T-N1.7 24.4%）**；2025 BEHAVIOR Challenge 50 个长 horizon 家务移动操作 **31.4%（π0.5 26.3%，赛事冠军 26.1%）**；DROID 后训练零样本迁移到未见环境与物体 82.5%；LIBERO 98.9%；RoboTwin 2.0 93.3%；SimplerEnv-Bridge 87.3% |
| 220 | GigaBrain-0.7：三系统架构 + 37,000 小时异构数据 | arXiv 2608.15875 | 2026-08-16 | Arch, Data | [链接](https://arxiv.org/abs/2608.15875) | **规模化的另一条产业路线**。**架构**：把理解、预测、动作统一到**三系统架构**（可与 Figure Helix 的 S0/S1/S2 三层对照，#50）。**数据**：预训练超过 **37,000 小时**异构具身数据——与 #223 τ₀-VLA 的 40,115 小时、#213 GR00T 1.7 的 32K+8K 小时属于同一量级。**训练**：引入**一阶段对齐训练**，联合优化 vision-language 理解与多具身动作。在自研 Maker H01 平台及主流本体上覆盖家用与工业场景。**承诺全部训练代码 + 预训练权重开源**。**对读者的含义**：2026 下半年具身基础模型的入场券已经是"3 万小时级真机数据 + 跨本体"，**个人 / 小团队的可行位置在后训练与部署侧，而不是预训练** |
| 221 | EcoVLA：端-边协同推理，20Hz 约束下能效提升最高 236% | arXiv 2608.15502 | 2026-08-16 | Edge | [链接](https://arxiv.org/abs/2608.15502) | **把"算力不够"的问题从延迟扩展到能耗**。**痛点**：端侧推理受算力与能量预算双重限制，很难同时满足实时控制与能效；全部卸载到边缘服务器又会被网络波动打出不可预测的延迟风险。**方法**：建立设备-边缘-网络的**延迟与能耗联合预测模型**，在毫秒级开销内快速评估候选协同方案；运行时持续选择"满足实时约束的能耗最优方案"，并随网络与系统状态变化自适应；另加轻量的**跨阶段中间张量传输机制**降低通信开销。**数字**：在 **20Hz 动作输出频率约束**下，系统能效相对既有协同推理方案**最高提升 236%**，且在动态网络与边缘负载下持续满足 SLO。**与 #44 / #205 的关系**：#44 指出瓶颈在内存带宽、#205 指出 VLM + action expert 占 >90% 运行时，EcoVLA 给出的是"往哪切、切完省多少电"的调度层答案 |
| 222 | ViTaR：在冻结 VLA 上加有界触觉残差，接触任务 +30.6 个百分点 | arXiv 2608.15816 | 2026-08-16 | Arch, Recipe | [链接](https://arxiv.org/abs/2608.15816) | **触觉的角色被重新定义**。**论点**：把触觉从"生成动作的感知输入"改成**执行调制器**——在**冻结的 VLA** 之上选择并缩放**有界残差修正**，从构造上保住预训练能力（不动主干 = 不会灾难性遗忘）。**两阶段**：Effect-Guided Modeling 用 outcome-grounded 的偏好证据判断"此处是否需要、以及需要哪种修正"；Residual Action Modulation 把该证据转成带连续缩放增益的残差选择，增益由实时视触觉观测决定。**数字**：UniVTAC benchmark 7 个接触密集任务平均成功率 **61.3%，比其冻结 VLA 基座高 30.6 个百分点**，并超过专门设计的触觉 baseline。**有真机实验**，确认有界触觉调制能扛住真实传感噪声与动力学。**对读者的含义**：想给已有 VLA 加触觉，**不必重训**——残差 + 冻结主干是目前性价比最高的接法 |
| 223 | τ₀-VLA：把"想多久"变成可调算力，40,115 小时真机数据 | arXiv 2608.16885 | 2026-08-17 | Arch | [链接](https://arxiv.org/abs/2608.16885) | **test-time compute 进入分层 VLA**。**痛点**：多数分层 VLA 的每一次高层决策都是单次前向，**没有机制把额外算力分配给更难或后果更严重的选择**。**方法**：把高层子任务生成表述为**可随算力扩展的推理问题**——高层策略用执行记忆生成子任务，**必要时先在世界模型引导下搜索若干备选再落子**；低层策略跨多个本体执行。**训练**：**40,115 小时**异构真实数据 + 多模态 co-training。**结论**：在域内与分布偏移设定下，增加 test-time 算力都能显著提升"下一个子任务"的预测准确率，且这些增益能转化为长 horizon 任务的闭环成功率提升。项目页 `tau0-vla.github.io` |
| 224 | 零示范跨本体：用 RL 把 OpenVLA-OFT 搬到绳驱并联机器人上 | arXiv 2608.01013 | 2026-08-02 | Recipe | [链接](https://arxiv.org/abs/2608.01013) | **"没有本体演示数据"时的一条路**。**设定**：把预训练 VLA 适配到新机器人通常假设有该本体的演示数据；本文研究更难的**零示范本体对齐**——目标是**绳驱并联机器人（CDPR）**，带简单夹爪，控制接口是模型完全没见过的形态。**方法**：不做监督微调，改为**在仿真中用 RL + 基于仿真器状态计算的稠密几何奖励**；两阶段——PPO 先学方向性运动基元，再**从 PPO checkpoint 用 GRPO 续训**并把指令空间扩展到 object-conditioned 命令。**对读者的含义**：如果你的机器人形态和 OXE / DROID 里的机械臂差得很远（并联、绳驱、自制结构），"先采几百条 demo"未必是唯一起点 |
| 225 | proprioception 到底该怎么接进 VLA：45 个原子任务 + 1→96 帧历史扫描 | arXiv 2608.03052 | 2026-08-04 | Arch | [链接](https://arxiv.org/abs/2608.03052) | **一个被所有人默认、却几乎从没被控制变量测过的设计位**。**现状**：几乎所有 VLA 都吃本体状态，但**接法互不兼容**——序列化进文本 prompt、投影进 vision-language prefix、或直接喂给 action expert；而且**几乎都只用当前一帧**。**三个开放问题**：(1) 当前状态究竟在哪些任务上真的改善闭环控制；(2) 状态历史帮多少，其收益是来自真实时序变化还是仅仅多了条件容量；(3) 状态该从哪进——VL backbone 还是动作生成模块。**方法**：在 flow-matching VLA 上做严格控制变量实验（固定 backbone、训练数据、动作表示、评估协议），实现 5 种代表性接口，在 **45 个原子任务（3 个任务族）+ 20 个复合任务**上评估，并把状态历史长度**从 1 扫到 96 帧**。**⚠️ 本条只记录实验设置**：摘要未给出具体结论数字，**本次采集未展开正文核对，请勿据摘要引用结论** |
| 226 | `compile_model=true` 静默训练出 NaN loss：SmolVLA / PI0 / PI05 全中招 | GitHub Issue (huggingface/lerobot #4178) | 2026-07-28 | Debug | [链接](https://github.com/huggingface/lerobot/issues/4178) | **本期最危险的静默故障**。**什么坏了**：官方自带的 `--policy.compile_model=true` 开关。**坏在哪**：`policies/common/vla_utils.py` 里的 `make_att_2d_masks`——由 `cumsum` + 比较运算构成的 prefix-LM mask。**为什么**：inductor 把这些 op 融进一个生成的 triton split-scan kernel（`triton_spl_fused__to_copy_cat_cumsum_...`），**在小动态 shape 下被误编译**。**最要命的部分**：用 `compile_mode=default` 时，**loss 从第 1 步就是 NaN，但训练"看起来成功"、不报任何错**；用 `max-autotune` 则直接 CUDA crash（另一种表现是 `CUDA driver error: an illegal memory access was encountered`）。**影响矩阵**：SmolVLA 已复现；PI0 / PI05 代码模式相同、结构性受影响；PI0-FAST（cumsum 只用于 position ids）、Diffusion、以及 ACT / wall-x / eo1 / groot / lingbot-va / vla-jepa / xvla（没有该开关）不受影响。**版本**：torch 2.11 下 default 模式**仍然坏**；max-autotune 那一类在 ≥2.12 修好。**修法**：给该 helper 加 `@torch.compiler.disable`（+7 行，PR #4179）——约 6 个极小的 mask op 退回 eager，其余照常编译，一次 graph break、开销几乎为零。上游相关：pytorch#131027、pytorch#124251。**行动项**：在这个修复进主线之前，**训 SmolVLA / PI0 / PI05 就别开 `compile_model`**；已经开了的，回去看第一条 log 的 loss 是不是 `nan` |
| 227 | Pipeline 迁移后归一化被静默跳过：`smolvla_base` 直接推理等于没反归一化 | GitHub Issue (huggingface/lerobot #4415) | 2026-08 | Debug | [链接](https://github.com/huggingface/lerobot/issues/4415) | **#210「预训练 SmolVLA 推理效果很差」的一个具体解释**。**现象**：Pipeline 迁移之后，**多数据集训练的模型（如 `lerobot/smolvla_base`）在推理时归一化 / 反归一化被静默跳过**。**根因**：迁移把模型内建 `normalize_inputs` / `unnormalize_outputs` 层里的统计量抽到外部 `PolicyProcessorPipeline` safetensors；多数据集模型的统计量是**带数据集前缀存的**（`so100.buffer.action.mean`、`so100-blue.buffer.action.mean` …），load 后变成 `_tensor_stats` 里 `"so100.buffer.action"` 这类 key。但 `_normalize_action()` 调 `_apply_transform()` 时传的是 `key=ACTION`（即 `"action"`），而 `_apply_transform` 的判断是 `if norm_mode == IDENTITY or key not in self._tensor_stats: return tensor` ——`"action"` 不在那堆带前缀的 key 里，**于是原样返回，没有任何报错**。state 归一化在 preprocessor 侧同样中招。**30 秒自检**：构造 `postprocessor(torch.tensor([0.5]*6))`，若 `torch.equal(输入, 输出)` 为 `True`，说明反归一化被跳过了。**教训重复第 N 次**：stats 不匹配是 VLA 最贵的一类 bug（#6、#30 已各中一次），而且**它总是静默的** |
| 228 | `delta_timestamps` 在 image-dtype 数据集上解码全部图像列：ACT 训练 3.0 → 13.6 it/s | GitHub Issue (huggingface/lerobot #4392) | 2026-08 | Debug, Data | [链接](https://github.com/huggingface/lerobot/issues/4392) | **一个纯工程、但收益 4.5× 的坑**。**触发条件**：数据集把相机存成 `dtype: image`（帧内嵌 parquet，`use_videos=False`）**且**策略用 `delta_timestamps` 做 action chunking（ACT / pi0 / SmolVLA 都是）。**发生了什么**：每次 `__getitem__` 会解码 `chunk_size × n_cameras` 张图然后立刻扔掉。**根因**：`_query_hf_dataset` 通过 `hf_dataset[indices]` 取 delta 行，而 dataset 级的 `set_transform`（`hf_transform_to_torch`）会转换这些行的**每一列**，包括图像列——哪怕这次查询只需要 `action` 这种数值 key。**实测（lerobot 0.6.0；343k 帧 / 2 相机 / 320×180 JPEG / `delta_timestamps={"action": [i/30 for i in range(100)]}`）**：每样本 `__getitem__` **~250–300 ms（约 200 次浪费的 JPEG 解码）**；只取数值行是 **2 ms**；绕过 transform 后 **~1–3 ms + 2 次 JPEG 解码**。落到一次 ACT 训练里，这就是 **3.0 it/s（CPU-bound，32 个 dataloader worker 全打满）与 13.6 it/s（GPU-bound，总共约 3 核）的差别**。**同理**：任何走 `hf_dataset` 的数值列扫描都会解码全库图像。**修法**：在 `_query_hf_dataset` 里先 `select_columns` 限定到请求的 delta key（或直接读底层 arrow 列）。已确认与 #2895 重复，修复在 PR **#4314**（cached single-column views）。**自检信号**：dataloader worker 吃满 CPU 而 GPU 利用率上不去时，**先查这个再考虑买卡** |
| 229 | `pi05_libero_base` 评测 0% 成功率——base checkpoint 不是能用的策略 | GitHub Issue (huggingface/lerobot #4206) | 2026-08-04 | Debug, Strategy | [链接](https://github.com/huggingface/lerobot/issues/4206) | **期望校准，不是 bug**。**现场**：用户按 model card 跑 `lerobot-eval --policy.path=lerobot/pi05_libero_base --env.type=libero --env.task=libero_object --eval.n_episodes=20`，A100 80GB、LeRobot 0.6.1、torch 2.11.0+cu128，**running_success_rate 一路 0.0%**；第一次被以 #4272 关闭后，在 main（f66e512）上复测**仍然 0%**，issue 被重开。**维护者回应**：能复现，但 **base 模型本来就该是低成功率**；换成 **`lerobot/pi05_libero_finetuned`** 后"大多数任务 100% SR"。**教训**：这与 #210（预训练 SmolVLA 推理"效果很差"）、#227（`smolvla_base` 反归一化被跳过）是同一类事故的三个面——**base / pretrained checkpoint 是拿来微调的起点，不是拿来直接评测或上真机的策略**。复现官方 benchmark 数字前，先确认你下的是 `*_finetuned` 而不是 `*_base` |

---

## Discord 实战情报（LeRobot Discord, 16k+ 成员）

> **本期（2026-08-21）仍未采集**：Chrome 会话依旧未登录 Discord（跳转 `discord.com/login`），无法执行搜索栏采集，已连续两期缺口。本期社区侧内容改由 GitHub Issues 直采补位（#226–#229 + 「GitHub Issues 现场速查」小节，共 25 条）。**下次运行前必须在 Chrome 中登录 Discord**，否则 D 系列将持续停更。

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

### D23. GR00T/X-VLA/ACT 三模型同时微调实测
- **来源**: `#vla-models` — **Don** (2026/1/11 04:28)，Discord 搜索 "gr00t training"
- **原文**: "Have you guys found a workflow that works best with imitation learning? Have been finetuning XVLA models, doing LORAs for both XVLA/Gr00t, and trained a few act policies this week trying to figure out what works best for a pick and place task. Any thoughts?"
- **类别**: Arch, Recipe
- **关键数据点**: 社区用户同时尝试 XVLA（LoRA）、GR00T（LoRA）、ACT 三个模型做 pick-and-place；反映社区仍在探索最佳模型选型——没有公认的"最优工作流"

### D24. GR00T 训练数据量经验：90/140/450 episodes 实测
- **来源**: `#vla-models` — 匿名用户 (2026/1 左右)，Discord 搜索 "gr00t training"
- **原文**: "...trained it on 9 tasks (Place the X in the center-box, Place the X in the top-left box, etc. One task for each token-placement). im not sure which model or how much data you have, but in our case we trained several with increasing amounts of data. (90 episodes, 140 episodes, and 450 episodes). The one trained on 90 episodes didnt work so good, 450 was probably too much. i think 2-300 is a good estimate using 2 cameras, 1 wrist and 1 top-view."
- **类别**: Recipe, Data
- **关键数据点**: GR00T 微调最优区间 200-300 episodes；低于 100 条动作抖动严重；与 SmolVLA/ACT 的 50-125 条相比，GR00T 因模型更大需要更多数据

### D25. GR00T 数据格式 v3.0 → v2.1 转换问题
- **来源**: `#vla-models` — **andrewr96** (2025/9/25 18:35)，Discord 搜索 "gr00t training"
- **原文**: "Is there any way to convert from 3.0 to 2.1? I collected a dataset which I planned to train gr00t on (which is compatible with 2.1) but the LeRobotDataset format changed and now I can't use it."
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
- **来源**: `#general-chat` — **zeyu.hu (Lightwheel)** (2025/7/15 14:01)，Discord 搜索 "gr00t training"
- **原文**: zeyu.hu 分享 LeIsaac pipeline（`https://github.com/LightwheelAI/leisaac`）——"Welcome to the LeIsaac Era"，将 LeRobot + GR00T N1.5 + IsaacSim 打通为完整 pipeline，"the entire process is completed in just 5 simple steps"。
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

### D41. RoboCasa Diffusion Training: RTX 4090/5090 上 500+ epochs 仍失败
- **来源**: `#robotics-papers` — guichristmann (2025/6/15)
- **原文**: "Hi I first asked this in the general channel, but reposting it here which seems to be more focused on research. Anybody with experience training Diffusion models on RoboCasa tasks or just in general? How long does it take to train a policy on a single RTX 4090 or 5090? I'm trying to reproduce the results from NVIDIA GR00T paper using Diffusion on RoboCasa tasks. Training with human demonstrations on a single task like 'TurnOnMicrowave' for >500 epochs takes many hours the results are still pretty terrible (robot is just wildly wiggling around). I'm using all of the default configs from RoboCasa and robomimic. Trying to figure out where I'm going wrong and to calibrate my expectations."
- **类别**: Debug, Recipe
- **关键数据点**: RoboCasa + robomimic 默认 config，单任务 TurnOnMicrowave 500+ epochs 后仍 wildly wiggling；单卡 4090/5090 训练多小时无效果；可能需要更多 epochs 或 config 调优；复现 NVIDIA GR00T 论文结果比预期难得多

### D42. XLeRobot/Zima: Caging-in-Time 框架用于零样本 Sim2Real
- **来源**: `#robotics-papers` — Vector Wang (XLeRobot) (2025/6/16)
- **原文**: "Caging in Time is officially on IJRR! We proposed a novel theory framework that deals with uncertainties by cages in time dimension for robust object manipulation such as random object pushing and tennis balancing and catching, all in pure open loop. This framework will be later used on XLeRobot for robust zero-shot sim2real transfer, making this low-cost dual-arm mobile robot, 'Zima', truly practical and physically intelligent for daily life."
- **类别**: Arch, Strategy
- **关键数据点**: Caging-in-Time 理论发表于 IJRR；通过时间维度笼罩处理不确定性；纯开环实现鲁棒物体操作；将用于 XLeRobot "Zima" 低成本双臂移动机器人的零样本 sim2real 迁移

### D43. 3D 打印任意形状触觉传感器
- **来源**: `#robotics-papers` — Mahi Shafiullah (NYU) (2025/6/19)
- **原文**: "One of the sci fi projects my lab mates have been working on: 3D printing arbitrary shaped tactile sensors" (引用 ICRA best paper)
- **类别**: Arch
- **关键数据点**: NYU 团队开发 3D 打印任意形状触觉传感器；获 ICRA 2025 field and service robotics best paper；社区讨论简化设计方案：广角相机 + 透明硅胶层放进夹爪内部，作为触觉替代方案

### D44. FAST Tokenizer 官方代码 Bug 修复
- **来源**: `#robotics-papers` — Nahid (2025/6/26)
- **原文**: "looks like they fixed a bug in tokenizer 3 weeks ago https://github.com/Physical-Intelligence/openpi/commit/63481042f1cb4f2dcb4a7da1623352307dd26533 ..I doubt it is reflected in their HF tokenizer yet."
- **类别**: Debug
- **关键数据点**: OpenPI FAST tokenizer 有 bug 已修复（commit 63481042）；HF 上的 tokenizer checkpoint 可能尚未更新；对所有使用 Pi0-FAST 的社区用户有影响——如果结果异常应检查 tokenizer 版本

### D45. Diffusion Steering: 冻结 DiT 仅更新噪声的 RL 方法引发讨论
- **来源**: `#robotics-papers` — Xingdong Zuo + Adil Zouitine (HF LeRobot) + KWang (2025/6/26)
- **原文**: Xingdong Zuo: "They did RL finetuning on pi0" (sharing diffusion-steering.github.io). Adil Zouitine: "The approach is simple wow". KWang: "Quite smart idea". Nahid: "when they say RL, which exact method they are using?" → Xingdong: "yep" (binary 0-1 label). an0n3039: "ConRFT also uses RL to finetune the action expert using a HIL. This new steering paper is simpler since it only updates the noise vector and freezes the DiT."
- **类别**: Arch, Recipe
- **关键数据点**: HF LeRobot 核心成员 Adil + 社区一致认为 Diffusion Steering 方法简洁有效；仅需 0-1 二值成功标签；与 ConRFT 互补——steering 冻结 DiT 更新噪声，ConRFT 更新 DiT 权重；同时学习两者是开放方向；Sergey Levine 也 tweet 推荐

### D46. FAST Action Tokenizer 从零实现经验
- **来源**: `#robotics-papers` — Nahid + skittle (2025/6/25)
- **原文**: Nahid: "anyone working on action tokenization here? I am trying to implement FAST action tokenizer from scratch. Wondering if anyone else tried it or similar". skittle: "If helps check the original implementation here https://github.com/Physical-Intelligence/openpi/tree/main/src/openpi/models in pi0_fast.py & tokenizer.py"
- **类别**: Arch, Debug
- **关键数据点**: 社区成员尝试从零实现 FAST tokenizer（arXiv 2501.09747）；参考实现在 OpenPI repo pi0_fast.py + tokenizer.py；D44 的 bug 修复也与此相关——从零实现时需注意官方代码中的已知 bug

### D47. Decision Transformer 在机器人操作中为何不流行？
- **来源**: `#robotics-papers` — Xingdong Zuo (2025/6/27)
- **原文**: "I got a naive question, in robotic manipulation, why Decision Transformers (RvS) seems not to be a popular method? Given that RTGs could be decided by binary episodic successes"
- **类别**: Arch, Strategy
- **关键数据点**: 社区对 Decision Transformer 在 manipulation 中缺席的疑问；binary episodic success 作为 return-to-go 理论上可行；但社区共识倾向 diffusion policy + VLA 路线；暗示 DT/offline RL 可能是被低估的方向

### D48. LeVERB: 视觉 Embodied Reasoning Benchmark
- **来源**: `#robotics-papers` — julien (2025/6/25)
- **原文**: sharing "https://ember-lab-berkeley.github.io/LeVERB-Website/" — a new benchmark for visual embodied reasoning
- **类别**: Arch, Data
- **关键数据点**: Berkeley EMBER Lab 发布 LeVERB benchmark；评估 VLA 的视觉推理能力而非仅仅动作执行；填补 LIBERO 等 benchmark 偏重记忆而非推理的空白（呼应 #29 VLA-0-Smol 的发现）

### D49. IMLE Policy: 低数据场景下可能优于 Flow Matching
- **来源**: `#robotics-papers` — Remi Cadene (HF LeRobot 创始人) (2025/6/14)
- **原文**: "https://imle-policy.github.io/ Expected to generalize better than flow matching on low data regime. What about the medium data regime?"
- **类别**: Arch, Recipe
- **关键数据点**: LeRobot 创始人 Remi Cadene 关注 IMLE Policy；低数据场景下泛化可能优于 flow matching；中等数据量场景效果待验证；cc LeRobot 核心团队 Martino Russi + Adil Zouitine → 可能影响未来 LeRobot 策略选择

### D50. V-JEPA 2: Meta World Model 引发社区关注
- **来源**: `#robotics-papers` — lucidrains (2025/6/11)
- **原文**: sharing "https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/" — received 14 reactions in the channel
- **类别**: Arch, Strategy
- **关键数据点**: Meta V-JEPA 2 world model 在 LeRobot 社区获 14 个 reaction（极高关注度）；社区成员讨论将其引入 LeRobot；world model 作为 VLA 的规划组件趋势明显——与 Cosmos Policy (#8) WFM 方向一致

### D51. Vision-in-Action: 感知不再是固定管道
- **来源**: `#robotics-papers` — julien (2025/6/20)
- **原文**: "Perception is yet another decision - https://vision-in-action.github.io/" — received 21 reactions (highest in thread)
- **类别**: Arch
- **关键数据点**: 21 个 reaction 是该频道最高互动量；核心观点：感知应该是策略的一部分而非固定的前端管道；挑战传统 VLA 架构中视觉编码器冻结的设计——与 VLA-0-Smol (#29) "必须微调 vision encoder" 发现一致

### D52. Action-LiPo: 可能增强 ACT Policy
- **来源**: `#robotics-papers` — julien (2025/6/21)
- **原文**: "Could be interesting to add to ACT policy in leRobot - https://sites.google.com/view/action-lipo" — received 2 reactions
- **类别**: Arch
- **关键数据点**: Lipschitz-constrained action representation 可能改善 ACT 的动作平滑性；与 ACTSmooth (#15) 解决相同问题但从不同角度——ACTSmooth 用 prefix conditioning，Action-LiPo 用 Lipschitz 约束

### D53. Gripper Separation: 夹爪和轨迹应该分开学习？
- **来源**: `#robotics-papers` — Serhan (2025/6/15)
- **原文**: "is there any approach out there, which separates the gripper of the trajectory from the other joints? where learning of those two can be addressed separately. the grip is more generalizable and does not have to be fine-tuned in every new environment, as long as it has learned enough types of objects in the foundation. yet the trajectory joints are to be fine-tuned for each environment."
- **类别**: Arch, Strategy
- **关键数据点**: 提出夹爪/轨迹解耦学习：抓取动作跨环境泛化，轨迹按环境微调；与 D18 颜色泛化实验一致——抓取泛化好但放置退化；暗示 VLA 架构可能需要分层动作空间设计

### D54. BridgeVLA: 新的 VLA 架构尝试
- **来源**: `#robotics-papers` — Yo_Fleyt (2025/6/24)
- **原文**: sharing "https://bridgevla.github.io/home_page.html" — received 3 reactions
- **类别**: Arch
- **关键数据点**: BridgeVLA 新架构；社区关注但讨论不多（3 reactions）；VLA 架构创新的持续涌现反映领域高度活跃

### D55. AMPLIFY: 无动作标注视频提升 VLA
- **来源**: `#robotics-papers` — an0n3039 (2025/6/25)
- **原文**: "https://amplify-robotics.github.io/ AMPLIFY: Actionless Motion Priors for Robot Learning from Videos. Conditioning on keypoint velocities improves VLA performance with positive transfer from non-robot human videos!"
- **类别**: Arch, Data
- **关键数据点**: 利用无动作标注的人类视频（keypoint velocities）提升 VLA；正向迁移从非机器人视频到机器人策略；与 WholeBodyVLA (#49) 方向一致——用 action-free video 降低数据采集成本

### D56. 加速 Imitation Learning: 新论文分享
- **来源**: `#robotics-papers` — Pepijn Kooijmans @HF LeRobot (2025/6/10)
- **原文**: "Paper about speeding up imitation learning policies: https://arxiv.org/pdf/2506.05064" — received 2 reactions
- **类别**: Edge, Arch
- **关键数据点**: HF LeRobot 官方成员分享 imitation learning 加速论文；与 OneDP (#22)、ActionFlow (#77)、EdgeVLA (#78) 同属 VLA 推理加速趋势

### D57. Robot Learning 核心经验分享 by julien
- **来源**: `#robotics-papers` — julien (2025/6/21)
- **原文**: "Some time ago, I shared a short post on what I learned from working on robot learning: https://www.linkedin.com/pulse/robot-learning-julien-perez-0p5ce One of the key benefits of robot learning is its ability to relax not only the hard constraints on the range of achievable tasks but also various assumptions—such as the need for fully predictable effector dynamics. In short, robot learning enables control over a new class of robots that have traditionally been unmanageable using classical optimal control or MPC-based approaches."
- **类别**: Strategy
- **关键数据点**: Robot learning 的核心价值不仅是任务范围扩大，更是放松了对可预测动力学的需求；使传统控制/MPC 无法管理的机器人（如软体、欠驱动）变得可控；这是 VLA 不可被经典控制替代的根本原因

### D58. Safe-SKILL: 安全技能学习与 Helix 架构
- **来源**: `#robotics-papers` — julien (2025/6/12)
- **原文**: "Something I would like to test on Reachy - https://safe-skill.github.io/ - figure.ai Helix seems to claim using this approach too"
- **类别**: Arch
- **关键数据点**: Safe-SKILL 安全约束下的技能学习；Figure Helix (#14/#50) 可能使用了类似方法；julien 计划在 Reachy 机器人上测试；安全性约束是 VLA 部署中被低估的需求

### D59. Co-Design: 软体机器人的学习式控制
- **来源**: `#robotics-papers` — julien (2025/6/21)
- **原文**: sharing "https://yswhynot.github.io/codesign-soft/" — robot learning for soft robots
- **类别**: Arch, Strategy
- **关键数据点**: 学习式方法与软体机器人的协同设计；传统控制无法处理软体动力学→robot learning 是唯一路径；代表 VLA 技术的长期应用方向——不仅是桌面操作，更是全新机器人形态

### D60. VLA 边缘加速方法全景（2025 年中）
- **来源**: 多源汇总 — #22 OneDP、#43 LiteVLA-Edge、#44 动作瓶颈分析、#45 TensorRT Edge-LLM、#77 ActionFlow、#78 EdgeVLA、#79 SpecPrune-VLA、#80 MulticoreWare CogAct
- **类别**: Edge, Strategy
- **关键数据点汇总**: 2025 年 VLA 边缘加速进入爆发期，至少 8 种不同方法：
  1. **蒸馏**（OneDP #22）：41× 加速，从多步到单步
  2. **量化**（LiteVLA-Edge #43）：4-bit GGUF → 6.6Hz on Jetson
  3. **系统级流水线**（ActionFlow #77）：2.55× 无需重训
  4. **架构简化**（EdgeVLA #78）：消除自回归依赖，7× 加速
  5. **动态剪枝**（SpecPrune-VLA #79）：action-aware，1.7× on real tasks
  6. **图优化**（MulticoreWare #80）：1.3× on CogAct 7.6B
  7. **硬件升级**（Jetson T4000 #46）：GDDR7 解决内存带宽瓶颈
  8. **投机解码**（TensorRT Edge-LLM #45）：EAGLE-3 + NVFP4
  规律：没有银弹——每种方法解决不同瓶颈，最终需要组合使用。最关键的发现（#44）：VLA 边缘延迟的真凶是内存带宽而非算力。

### D61. 社区对 SAM2 数据标注工具的需求
- **来源**: `#general-chat` — Asif (2026/3/15-16)
- **原文**: "Building a lightweight video annotation tool with SAM2 tracking that exports in LeRobot format. Would anyone actually use this? What's your biggest pain in creating datasets right now?"
- **类别**: Data
- **关键数据点**: SAM2 tracking 自动标注 + LeRobot 格式导出；社区最大数据痛点：手动标注耗时 + 格式转换复杂；如果成熟可大幅降低 VLA 训练数据生产成本

### D62. 机器人实验 Debug 花费大量时间
- **来源**: `#general-chat` — robo (2026/3/15)
- **原文**: "I'm researching how robotics teams handle experiment logging and debugging robot behavior. What does your current workflow look like? What breaks most often? I am stuck in this bcz whenever i go to debug i am spending very long time to it so i want to know how you people do it so that i can save my time as well."
- **类别**: Debug
- **关键数据点**: 调试机器人行为占用大量时间是普遍痛点；社区缺乏标准化的实验日志和 debug 工作流；与 Foxglove (#86) 可视化工具和 LIBERO replay 功能形成需求呼应

### D63. 视觉分割模型辅助机械臂精确定位
- **来源**: `#general-chat` — River (2026/3/15)
- **原文**: "very trivial usage of the segmentation model i built earlier to correctly rotate the arm to the cube's horizontal position"
- **类别**: Arch, Recipe
- **关键数据点**: 用分割模型指导机械臂旋转到正确位置；模块化方法（分割+控制分开）在某些精确定位任务上比端到端 VLA 更直接有效；**对社区的启示**：不一定所有事情都要用 VLA 端到端——经典视觉方法 + 简单控制在某些场景更实用

### D64. Unified Flow & Matching (UFM) 用于夹爪相机
- **来源**: `#robotics-papers` — LeDaniel (2025/6/13)
- **原文**: "UFM - Unified Flow & Matching model I imaging something like this would be good for correlating what the gripper cam sees https://uniflowmatch.github.io/"
- **类别**: Arch
- **关键数据点**: UFM 统一 optical flow 和 feature matching；社区构想用于关联夹爪相机视角的物体特征；潜在替代触觉传感器——用视觉计算接触力估计

### D65. 简化触觉传感器设计：广角相机 + 硅胶层
- **来源**: `#robotics-papers` — JClinton (2025/6/20)
- **原文**: "I was wondering. Why can't we just use a wide angle camera and place it inside the gripper, with a layer of transparent silicon on top, then down sample? The entire point is as another sensor, and when fed into the vla training, surely this downsampled wide angle camera has the same data as this complex touch sensor thing" / Mahi Shafiullah (NYU) 回复: "Haha you're describing this work: https://arxiv.org/html/2504.19341v1 this was best paper at ICRA this year"
- **类别**: Arch
- **关键数据点**: 社区独立构想的简化触觉方案恰好是 ICRA 2025 最佳论文的核心思路；广角相机+透明硅胶层作为低成本触觉替代方案；NYU Mahi Shafiullah 证实这正是他们实验室的方向；**对社区的价值**：触觉传感可以很简单——不需要复杂的 GelSight

### D66. Residual VAE 回归 + 社区惊讶
- **来源**: `#robotics-papers` — lucidrains (2025/6/25)
- **原文**: "never thought i'd see residual VAEs again" (回复 LeVERB benchmark)
- **类别**: Arch
- **关键数据点**: lucidrains（知名开源 ML 开发者）对 residual VAE 在机器人学习中回归表示惊讶；暗示旧技术在新场景中可能有新价值；VLA 领域的技术选择不应局限于最新方法

### D67. 视频 Value Function: 自动化 RL 循环
- **来源**: `#robotics-papers` — Xingdong Zuo (2025/6/26)
- **原文**: "https://sites.google.com/view/vip-rl https://dibyaghosh.com/vptr/ reminds me of video-based value function, could be nice to automate the RL loop"
- **类别**: Arch
- **关键数据点**: Video-based value function 可以自动化 RL 训练循环；VIP-RL + VPTR 两个相关方法；当前 VLA RL 微调（Diffusion Steering、ConRFT）仍需人工提供 reward 信号——video value function 可能是自动化的路径

### D68. 机械臂螺丝型号和热插铆 FAQ
- **来源**: `#general-chat` — McLovin + JPo (2026/3/14-16)
- **原文**: McLovin: "which type of screws did you use? So I know which type of heat insert I have to buy. M3x4, M4x4, M5x4 maybe?" JPo: "For the SO-101 it comes with hardware."
- **类别**: Debug
- **关键数据点**: SO-101 自带硬件（包括螺丝）；新手常见困惑——热插铆和螺丝型号选择；GitHub 文档描述了相机安装所需的额外硬件

### D69. 相机选择: RealSense vs 普通 USB 摄像头
- **来源**: `#general-chat` — River + b1063n (2026/3/13)
- **原文**: River: "you should be able to use the ACT models without it" b1063n: "All right makes sense, the normal camera is inexpensive. I will start there and upgrade later, thanks"
- **类别**: Recipe
- **关键数据点**: ACT 不需要深度相机（RealSense）也能工作；建议新手先用普通 USB 摄像头起步；与 D17 (RealSense 深度集成困难) 一致——深度数据在 LeRobot 中仍是二等公民

### D70. RL 微调 VLA 方法论全景（2025 年中更新）
- **来源**: 多源汇总 — D45 Diffusion Steering、#89 ConRFT、#94 VLA-RFT、#48 DPPO、#11 π*0.6、#20 HIL-SERL、#97 Awesome-RL-VLA
- **类别**: Arch, Strategy
- **关键数据点汇总**: 截至 2025 年中，VLA RL 后训练已形成至少 6 种不同方法：1. **Diffusion Steering (#88)**：冻结 DiT 仅更新噪声向量，最简单；2. **ConRFT (#89)**：更新 action expert 权重，需要 HIL；3. **VLA-RFT (#94)**：用 world model 做仿真器，<400 步即超越 supervised baseline；4. **DPPO (#48)**：双层 MDP 框架，diffusion chain = 内层 MDP；5. **π*0.6 (#11)**：coaching + reinforcement，工业级结果；6. **HIL-SERL (#20)**：人在回路中 SAC，1-2h 达 100% SR。核心分歧：是冻结模型更新输入（steering），还是更新模型本身（ConRFT/DPPO）？两种路线各有优劣，community consensus 是最终可能需要组合。

### D71. Action Tokenization 方法全景（2025 年中更新）
- **来源**: 多源汇总 — D1 FAST quantile normalization、D46 FAST 从零实现、#95 OmniSAT、#96 FASTer、ICLR 2026 趋势 (#10)
- **类别**: Arch
- **关键数据点汇总**: VLA action tokenization 三大路线：1. **FAST/FAST+**（Pi0-FAST）：DCT 频域压缩，自回归生成；需要 quantile normalization (D1)；2. **OmniSAT (#95)**：统一两阶段 tokenizer，6.8× 压缩；Droid 预训练；3. **FASTer (#96)**：可学习 tokenizer + block-wise 解码；从手工设计到端到端学习。ICLR 2026 确认 action tokenization 是五大趋势之一 (#10)；社区实操重点：normalization 配置 (D1)、tokenizer bug 检查 (D44)、HF checkpoint 版本验证

### D72. VLA 预训练数据源全景
- **来源**: 多源汇总 — #1 SmolVLA 社区数据、#38 LingBot 20K+ 小时、#42 MolmoBot 1.8M 轨迹、#102 NVIDIA 合成数据、#82 Data Scaling Laws
- **类别**: Data, Strategy
- **关键数据点汇总**: VLA 预训练数据的三大来源及其 trade-off：1. **社区真实数据**（SmolVLA #1）：多样性好但质量不均（OXE 质量差 #10）；2. **专有真实数据**（LingBot #38 20K+h、π0 非公开）：质量高但不可复现；3. **合成/仿真数据**（MolmoBot #42 1.8M、NVIDIA #102）：可扩展但存在 sim-real gap。Data Scaling Laws (#82)：多样性 >> 绝对数量；相机位姿多样性 >> 纹理变化；power-law 关系可指导数据采集策略

### D73. 新手 FAQ: LeRobot 到底是什么？
- **来源**: `#general-chat` — Aarav J. (2026/3/14)
- **原文**: "@everyone i'm new to this community, what is lerobot all about?!"
- **类别**: Strategy
- **关键数据点**: LeRobot Discord 持续吸引完全新手（16K+ 成员）；社区从纯研究者转向包含 hobby 玩家；**趋势信号**：VLA/具身智能正在从学术圈扩散到更广泛的技术社区

### D74. VLA 泛化挑战全景（150 条经验升级版）
- **来源**: 多源汇总 — D6 SmolVLA 98%→暴跌、D14 vast.ai 10/10 失败、D18 颜色泛化、D32 社区共识、#29 VLA-0-Smol LIBERO、#42 MolmoBot sim2real、#82 Data Scaling Laws、#112 RoboArena
- **类别**: Debug, Strategy
- **关键数据点汇总**: 升级版泛化分析（整合新证据）：**在哪些维度泛化失败**: 位置变化（D6/D14）、颜色变化（D18: 抓取 OK 但放置退化）、环境变化（D32 五个团队共识）；**在哪些维度泛化成功**: sim-to-real（MolmoBot #42 超 π0.5）、跨具身（X-VLA #34 soft prompt）；**新发现**: Data Scaling Laws (#82) 证明泛化遵循幂律——200 环境比 20 环境好，但边际收益递减；RoboArena (#112) 确认 sim benchmark 泛化不等于真实泛化；**更新的最佳实践**: 窄场景验证 → 有计划扩展（维度：位置→旋转→颜色→光照→背景）

### D75. VLA 开源生态全景（2026 年中更新）
- **来源**: 多源汇总 — #38 LingBot-VLA、#81 StarVLA、#91 UnifoLM-VLA、#92 Wall-X、#93 RoboVLMs、LeRobot v0.5
- **类别**: Strategy
- **关键数据点汇总**: 2026 年中 VLA 开源生态爆发：**中国**: LingBot-VLA（蚂蚁集团 #38）、UnifoLM-VLA（宇树 #91）——产业级开源；**框架**: StarVLA (#81 模块化)、RoboVLMs (#93 统一 VLM 集成)——研发效率工具；**策略**: Wall-X (#92 flow matching)、Pi0-FAST（自回归）、X-VLA (#34 跨具身)——LeRobot v0.5 三条新路线。**格局变化**: 从"Pi 系列独占"到"多路线并行竞争"；开源追上闭源的速度在加快

### D76. World Model 在 VLA 中的三重角色
- **来源**: 多源汇总 — #8 Cosmos Policy、#94 VLA-RFT、#102 NVIDIA 合成数据、#103 Physical OS、D50 V-JEPA 2
- **类别**: Arch, Strategy
- **关键数据点汇总**: World Model 在 VLA 生态中已承担三种截然不同的角色：1. **规划器**（Cosmos Policy #8）：预测未来状态 → 选择最优动作序列；2. **仿真器**（VLA-RFT #94）：替代真实交互做 RL 微调，<400 步即超越 supervised；3. **数据增强器**（NVIDIA #102）：将仿真图像转化为逼真数据。V-JEPA 2 (D50) 在社区获 14 reactions 表明 world model 的关注度正在赶上 VLA

### D77. SO-101 组装常见陷阱升级版
- **来源**: 多源汇总 — D7 夹爪电机死机、D8 校准超限、#5 Sherry Chen 经验、D68 螺丝选择、D69 相机选择、#35 Val Kamenski
- **类别**: Debug
- **关键数据点汇总**: SO-101 硬件陷阱完整清单（社区 10+ 用户经验）：1. USB 权限：chmod 666 或 udev rules (#5/#35)；2. 夹爪装反 → 校准失败 (#35)；3. 校准 position 超 2047 → 重新调整组装位置 (D8)；4. 夹爪电机过力冻死 → 备电机 + Feetech debug 软件 (D7)；5. 相机：先用普通 USB 摄像头，不急着买 RealSense (D69)；6. 螺丝/热插铆：SO-101 套件自带 (D68)；7. lsusb -t 检查 USB 2.0 vs 3.0（RealSense 接 2.0 会降速 #17）

### D78. VLA 边缘部署硬件升级路径
- **来源**: 多源汇总 — #3 NXP i.MX95、#19 Jetson Orin、#28 Jetson Thor、#43 LiteVLA-Edge、#46 Jetson T4000、#65 Jetson Orin Nano、D60 加速全景
- **类别**: Edge, Strategy
- **关键数据点汇总**: VLA 边缘硬件从低到高的完整升级路径：**入门**：Jetson Orin Nano ($199) → ACT/小模型 (#65)；**主力**：Jetson AGX Orin → SmolVLA 6.6Hz (#43)、训练+推理 (#19)；**专业**：NXP i.MX 95 → ACT 优化后 0.32s (#3)（嵌入式级）；**旗舰**：Jetson Thor → GR00T N1.5/1.6 (#28)、1200 FP4 TFLOPS；**未来**：Jetson T4000/T5000 → GDDR7、VLA 专用 (#46)。关键洞察：内存带宽是真瓶颈 (#44)，Thor 5× 算力仅改善 1.4× 延迟

### D79. VLA 训练成本全景
- **来源**: 多源汇总 — #5 ACT 4h@3080、#9 Pi0 50 条@MI200、#13 OpenVLA-OFT 1-2 天@8xA100、#24 SmolVLA 4h@A100、#29 VLA-0-Smol 消费级、#39 LoRA 8GB VRAM、D27 DP 50min@4090、D34 GPU 矩阵
- **类别**: Recipe, Edge
- **关键数据点汇总**: VLA 训练成本速查表（按模型大小排序）：**ACT 52M**：RTX 3080 12GB，4h (#5)；**SmolVLA 450M**：A100 40GB，4h (#24)；LoRA 可降至 8GB (#39)；**VLA-0-Smol 500M**：消费级 GPU 可跑 (#29)；**Diffusion Policy**：RTX 4090，50min 即 pretty well (D27)；**Pi0 3B**：AMD MI200 或多卡 (#9)；**OpenVLA-OFT 7B**：8×A100/H100，1-2 天 (#13)。**最便宜路径**：LoRA + 量化 → 8GB VRAM 跑 3.1B VLA (#39)

### D80. 社区 VLA 模型选型升级版（含新模型）
- **来源**: 多源汇总 — D36（原版）+ #91 UnifoLM-VLA + #92 Wall-X + #34 X-VLA + #93 RoboVLMs
- **类别**: Arch, Strategy
- **关键数据点汇总**: 更新的模型选型决策树（新增 3 个选项）：**ACT (52M)**：最快上手、最便宜 → 单任务快速验证；**SmolVLA (450M)**：VLM 预训练+异步推理 → 多相机+一定泛化；**X-VLA (0.9B)**：soft prompt 跨具身迁移 → 多机器人实验室（ICLR 冠军）；**Wall-X (Qwen2.5-VL)**：flow matching + 强视觉理解 → 需要语言条件控制；**GR00T N1.5/1.6**：NVIDIA 生态 → 非精密任务+仿真集成；**Pi0/Pi0.5 (3B+)**：最强泛化+RL → 大型实验室；**UnifoLM-VLA (宇树)**：人形操作专用 → Unitree G1 用户。**新建议**：如果你同时有多种机器人 → X-VLA；如果你只有 SO-101 → SmolVLA 或 ACT

### D81. 数据集格式碎片化问题（升级版）
- **来源**: 多源汇总 — D25 v3.0→v2.1 转换、D28 Forge 万能转换器、#106 SAM2 标注工具、LeRobot v0.5
- **类别**: Data
- **关键数据点汇总**: 数据格式碎片化的现状和解法：**问题**：LeRobot v3.0 vs v2.1 不兼容 (D25)、GR00T 微调脚本期望 v2.1、不同框架格式各异；**工具链**：Forge (D28)：RLDS ↔ LeRobot v2/v3 ↔ Zarr ↔ HDF5 ↔ Rosbag 双向转换；SAM2 标注工具 (#106)：直接输出 LeRobot 格式；LeRobot v0.5：HF Hub 2.2K+ 数据集，渐成事实标准。**趋势**：LeRobot v3 正在成为社区标准，但转换工具仍不可或缺

### D82. 人形机器人 VLA 特殊挑战
- **来源**: 多源汇总 — #49 WholeBodyVLA、#50 Helix 02、#62 Pi0-FAST 人形微调、#91 UnifoLM-VLA、#108 GR00T N1
- **类别**: Arch, Strategy
- **关键数据点汇总**: 人形 VLA 相比桌面臂的特殊挑战：1. **DoF 爆炸**：桌面臂 6-7 DoF → 人形 30+ DoF (D62)，action space 设计是核心难题；2. **平衡协调**：Helix S0 层千赫兹平衡控制 (#50)——桌面臂无此需求；3. **数据采集**：WholeBodyVLA (#49) 用无动作标注的第一人称视频降低成本；4. **跨平台**：GR00T N1.6 验证了 Unitree G1 + YAM + Agibot 多平台 (#108)。**核心洞察**：人形 VLA 不能简单复用桌面臂策略——需要分层架构 (S0/S1/S2) 和更大数据量

### D83. VLA 综述论文全景
- **来源**: 多源汇总 — #10 ICLR 2026 全景、#72 Diffusion Policy Survey、#97 RL-VLA Survey、#98 Action Tokenization Survey、#99 真实世界应用 Survey、#100 Awesome-Embodied、#104 Large VLM-based VLA Survey
- **类别**: Strategy
- **关键数据点汇总**: 2025-2026 年至少 7 篇 VLA 综述/论文列表，覆盖不同角度：**动作 tokenization (#98)**、**diffusion policy (#72)**、**RL 微调 (#97)**、**真实世界应用 (#99)**、**大型 VLM 驱动 (#104)**、**ICLR 趋势 (#10)**。**选择建议**：架构设计 → #104；部署决策 → #99；RL 微调 → #97；入门 → #100

### D84. 社区信息密度分析（200 条版本升级）
- **来源**: 本次采集过程的元观察
- **类别**: Strategy
- **关键数据点汇总**: 200 条采集后的信息分布分析（升级 D40）：**最高信息密度来源**（按类别）：**Recipe/Debug**：个人实战 blog（ggando #17、Giacomo Moran #15、VLA-0-Smol #29）；**Arch**：LeRobot v0.5 release note (#4) 一篇涵盖 5+ 新策略；**Strategy**：ICLR 2026 全景 (#10) 一篇覆盖整个领域；**Edge**：NVIDIA Developer Blog 系列（#45/#46/#47）；**Data**：Discord 讨论（D28 Forge、#82 Scaling Laws）。**新发现**：Discord #robotics-papers 频道（6 月内容）比 #general-chat 信息密度高 3×——研究者和 HF 核心团队在这里讨论前沿方向；#general-chat 更多是新手 FAQ 和硬件问题。**建议后续采集重点**：Discord #robotics-papers + #vla-models 频道搜索、新出现的个人 blog

### D85. VLA 领域开放问题清单（2026 年中）
- **来源**: 多源汇总 — 全部 200 条条目的综合分析
- **类别**: Strategy
- **关键数据点汇总**: 基于 200 条社区实战经验总结的 VLA 领域十大开放问题：1. **泛化**：如何从固定场景 >95% 扩展到真实环境（D74/D32）；2. **数据效率**：power-law scaling 的拐点在哪（#82）；3. **RL 后训练**：冻结 vs 更新模型，哪种路线胜出（D70）；4. **边缘部署**：内存带宽瓶颈的算法解法（#44/D60）；5. **Action Tokenization**：FAST vs OmniSAT vs FASTer 谁胜出（D71）；6. **评估标准化**：LIBERO 已不够，什么替代（#112 RoboArena）；7. **数据格式统一**：LeRobot v3 能否成为唯一标准（D81）；8. **人形迁移**：桌面臂经验多少能复用到人形（D82）；9. **触觉集成**：低成本触觉方案何时成熟（D65）；10. **World Model 角色**：规划器 vs 仿真器 vs 数据增强器（D76）

### D86. ACT 多子任务训练：数据整理比模型选择更重要
- **来源**: LeRobot Discord #show-us-what-you-built — Fei (2026/3/2)
- **类别**: Recipe
- **关键数据点汇总**: 用户 Fei 分享 ACT 3-subtask 训练经验，累计超过 1000 episodes。核心经验："data curation is critical"——数据质量和标注一致性比模型架构选择影响更大。建议把训练当作"marathon not sprint"，逐步迭代优化数据集而非一次性大规模采集。与 D28 Forge 数据经验一致

### D87. XLeRobot 垃圾分拣：低成本桌面操作实战
- **来源**: LeRobot Discord #show-us-what-you-built — Grigorij (2026/3/5)
- **类别**: Recipe
- **关键数据点汇总**: 用户 Grigorij 展示 XLeRobot 平台完成桌面垃圾分拣（从桌面抓取垃圾投入垃圾桶）的 demo。使用标准 LeRobot 训练流程，视频展示了完整的 pick-and-place 循环。XLeRobot 作为社区衍生硬件平台的活跃度持续上升

### D88. RoboCrew：具身 LLM Agent 框架（62 stars）
- **来源**: LeRobot Discord #show-us-what-you-built — Grigorij (2026/3/5)
- **类别**: Arch
- **关键数据点汇总**: Grigorij 发布 RoboCrew——一个具身 LLM agent 框架，定位为让机器人编排"像 CrewAI 或 Autogen 一样简单"。GitHub 62 stars、5 forks。将 LLM 多 agent 编排范式（任务分解、角色分配）迁移到机器人控制领域，降低多机器人协作的编程门槛

### D89. DimensionalOS：多机器人同步控制演示
- **来源**: LeRobot Discord #show-us-what-you-built — Ruthwik/DimensionalOS (2026/3/8)
- **类别**: Recipe, Arch
- **关键数据点汇总**: DimensionalOS 展示 G1 人形 + Go2 四足 + xARM 桌面臂 + Piper 协作臂的四机器人同步控制。不同形态（人形/四足/桌面臂）的机器人在统一框架下同时运动。这是社区中首次看到跨形态多机器人实时协调的 demo

### D90. Foundation Stereo + SAM2 = 零样本 Foundation Pose（$30 方案）
- **来源**: LeRobot Discord #show-us-what-you-built — Vector Wang/XLeRobot (2026/3/11)
- **类别**: Recipe, Edge
- **关键数据点汇总**: Vector Wang 分享零样本物体位姿估计方案："Fast Foundation Stereo + SAM2 = zero shot Foundation Pose"。关键：使用从 AliExpress 购买的 $30 双目相机即可实现。将 NVIDIA Foundation 系列模型组合用于低成本机器人视觉，大幅降低精确抓取的硬件门槛

### D91. OpenCastor：多硬件平台统一支持
- **来源**: LeRobot Discord #show-us-what-you-built — craig (2026/3/12)
- **类别**: Recipe
- **关键数据点汇总**: craig 分享 OpenCastor 项目进展——支持 Reachy 和 LeRobot 硬件的统一框架。包含 Reachy Mini 教程、HLabs ACB v2.0 支持，以及 Raspberry Pi 5 / Jetson / ESP32 多计算平台适配。社区正在解决 LeRobot 生态的硬件碎片化问题

### D92. WandB 训练可视化 + Docker 训练流程分享
- **来源**: LeRobot Discord #show-us-what-you-built — Mr. Shaitana (2026/2/26)
- **类别**: Recipe
- **关键数据点汇总**: Mr. Shaitana 公开了 pick_and_place_v1.0.0 的 WandB workspace（wandb.ai/avilay/lerobot/runs/udf4orpe），展示完整训练曲线。使用 A10 NVIDIA GPU + Docker 容器化训练（Dockerfile 在 github.com/avilay/learn-robotics）。对新手有参考价值：提供了从环境配置到训练监控的完整可复现流程

### D93. MM-Hand 1.0：开源 21-DoF 灵巧手（$1400，HKU MMLab）
- **来源**: LeRobot Discord #show-us-what-you-built — Zhuoheng Li (2026/2/15)
- **类别**: Arch, Edge
- **关键数据点汇总**: 港大 MMLab 的 Zhuoheng Li 发布 MM-Hand 1.0 开源灵巧手并邀请 beta 测试。核心规格：21 DoF、腱驱动、关节角度传感器、21 个张力传感和自紧装置、SPI/I2C 触觉接口、TTL/CAN 电机接口、3D 打印结构、Arduino 编程。材料成本约 $1400 USD。完全开源（CAD、装配指南、电子、软件）。社区反馈热烈——lotyr 评价"Great to finally see a tendon driven hand with joint angle sensors!"。这是社区首个面向 VLA 研究的低成本完整灵巧手方案

### D94. 错误恢复训练策略：教恢复不教犯错
- **来源**: LeRobot Discord #show-us-what-you-built — Fei + eliasab + tms-gvd (2026/1/22)
- **类别**: Recipe, Debug
- **关键数据点汇总**: 一段高质量的社区讨论。Fei 分享错误恢复训练策略：核心原则是"teach it to recover, not to make mistakes"——只记录恢复部分，不记录导致错误的过程。具体方法：模拟常见失败状态（夹爪在物体旁但未抓住、臂挡住物体视线），然后示教恢复动作。大部分恢复轨迹很短（<5 秒）。tms-gvd 补充了 hackathon 经验：从遥操切换到策略时动作很乱，怀疑是遥操导致 OOD 状态。Fei 回复："intervention is to take it from OOD back into the distribution, so you need a strong baseline to start with and only tackle edge cases"

### D95. Arpeggio Gripper：社区自制夹爪（9👍 热门帖）
- **来源**: LeRobot Discord #show-us-what-you-built — Over Engineer (约 2026/2 月)
- **类别**: Edge
- **关键数据点汇总**: Over Engineer 发布 "Introducing Arpeggio Gripper" 视频展示自制夹爪设计。帖子获得 9👍、2💯、1🔥 的高互动量（在 #show-us-what-you-built 频道中属于热门级别），说明社区对可复现的低成本硬件方案有强烈需求

### D96. $10 PS4 相机做 Zero-Shot Safety 评测
- **来源**: LeRobot Discord #show-us-what-you-built — rafa.felix (2026/1/27)
- **类别**: Recipe, Edge
- **关键数据点汇总**: rafa.felix 分享用 $10 PS4 相机对 SO-ARM101 进行 Zero-Shot Safety 评测的 YouTube 视频。极致低成本方案：用游戏主机配件作为机器人视觉传感器。与 D90 的 $30 双目相机方案一起，说明社区正在系统性地探索消费级硬件在机器人研究中的可用性

### D97. BotBrain：ROS2 模块化机器人大脑（Unitree Go2/G1 支持）
- **来源**: LeRobot Discord #show-us-what-you-built — botbotrobotics (约 2026/1 月)
- **类别**: Arch, Edge
- **关键数据点汇总**: botbotrobotics 发布 BotBrain（github.com/botbotrobotics/BotBrain）——ROS2 模块化开源机器人大脑。功能：遥操/控制、自主导航、感知、Web UI（Cockpit + 车队管理）、健康诊断。支持 Unitree Go2/G1 及其他 ROS2 机器人。运行在 NVIDIA Jetson 上。3D 可打印硬件。作者明确表示"We want to add LeRobot support asap"——说明 LeRobot 正成为社区项目的首选集成目标

### D98. LeRobot Web Interface：浏览器端双臂遥操
- **来源**: LeRobot Discord #show-us-what-you-built — WhitneyDesignLabs (约 2026/1 月)
- **类别**: Recipe
- **关键数据点汇总**: WhitneyDesignLabs 发布 lerobot-web-interface（GitHub WhitneyDesignLabs/lerobot-web-interface），实现浏览器端双臂遥操控制。获得 6🔥 和 1👍。有趣的实战经验：由于系统在显示相机画面时频繁崩溃（可能因为旧电脑），WhitneyDesignLabs 开始"blind teaching"——不看画面直接遥操示教。这说明即使在不理想的硬件条件下，有经验的操作者仍可以采集有效数据

### D99. Leader-Inverse-Follow：自定义 LeRobot 遥操模式
- **来源**: LeRobot Discord #show-us-what-you-built — Fei (2026/1/23)
- **类别**: Recipe, Arch
- **关键数据点汇总**: Fei 在自己的 LeRobot fork (TheWisp/lerobot) 上开发了 leader-inverse-follow 功能分支——一种新的遥操作控制模式。使用自定义 so107 机器人（比 SO101 多一个 DOF），但同样适用于 SO101。该 PR 展示了社区如何通过 fork 和 feature branch 方式扩展 LeRobot 的核心功能

### D100. awesome-vla-study：VLA 论文结构化阅读清单（168 stars）
- **来源**: LeRobot Discord #show-us-what-you-built — MilkClouds (2026/2 月)
- **类别**: Strategy
- **关键数据点汇总**: MilkClouds 分享 awesome-vla-study（github.com/MilkClouds/awesome-vla-study）——"A structured reading list on VLA models — from diffusion/flow matching foundations through state-of-the-art robot foundation model architectures"。168 stars、8 forks。与 #97 Awesome-RL-VLA、#98 Psi-Robot 等列表互补，从不同组织视角覆盖 VLA 文献

### D101. Reachy Mini 趣味实验：Dance Mode + Sleep Mode
- **来源**: LeRobot Discord #show-us-what-you-built — thegr8madcat (2026/2/8)
- **类别**: Recipe
- **关键数据点汇总**: thegr8madcat 分享 Reachy Mini 在模拟器中的实验，添加了 dance mode 和 sleep mode。附带 breadboard + LCD 显示 "reachy nano" 的硬件图。展示了社区成员如何将 Reachy Mini 作为低门槛的机器人学习平台，从趣味项目入手积累经验

### D102. HelloRL：模块化 RL 框架（面向机器人）
- **来源**: LeRobot Discord #show-us-what-you-built — i10e-lab (约 2026/2 月)
- **类别**: Arch
- **关键数据点汇总**: i10e-lab 发布 HelloRL（github.com/i10e-lab/HelloRL）——"A fully modular framework to make Reinforcement Learning quick and easy"。带有可视化 RL 环境 demo。定位为降低机器人 RL 的入门门槛，与 RLinf-VLA (#130) 的统一 RL 基础设施定位互补但面向不同用户群体（HelloRL 面向初学者，RLinf-VLA 面向 VLA 研究者）

### D103. hydr8：Hydra 配置管理简化工具
- **来源**: LeRobot Discord #show-us-what-you-built — rsamf (2026/2/12)
- **类别**: Recipe
- **关键数据点汇总**: rsamf 发布 hydr8（github.com/rsamf/hydr8）——基于 decorator 的 Hydra 配置注入工具。Hydra 是 LeRobot 和多数机器人学习框架使用的配置管理系统，hydr8 通过 Python decorator 简化了配置定义和注入流程。对 LeRobot 用户有实用价值——减少冗长的 YAML 配置编写

### D104. YOR：$10K 开源双臂移动操作机器人（NYU）
- **来源**: LeRobot Discord #robotics-papers — Mahi Shafiullah/NYU (2026/2/13)
- **类别**: Arch, Edge
- **关键数据点汇总**: NYU 的 Mahi Shafiullah 发布 YOR——"open-source bimanual mobile manipulator robot – built for researchers and hackers alike for only ~$10k"。cone-e.com 是项目主页。社区成员 lorepieri 随后询问硬件细节（红色旋钮和 3D 打印部件用途）。$10K 的价格点填补了 SO-ARM101（~$300 桌面臂）和工业移动机器人（$50K+）之间的空白，为中等预算的实验室提供了双臂移动操作平台

### D105. DreamDojo：通用机器人 World Model（来自大规模人类视频）
- **来源**: LeRobot Discord #robotics-papers — lotyr (2026/2/9)
- **类别**: Arch
- **关键数据点汇总**: lotyr 分享 DreamDojo（dreamdojo-world.github.io/）——"A Generalist Robot World Model from Large-Scale Human Videos"。获得 4❤️。后续 skpro19 提问如何用它训练 SO-101 做 pick-and-place——说明社区对 world model 的实际应用场景有强烈兴趣。与 D76 的 World Model 角色分析一致：从人类视频学习世界模型是降低机器人数据成本的关键路径

### D106. NVIDIA EgoScale + "Nvidia has quietly solved robotics end-to-end"
- **来源**: LeRobot Discord #robotics-papers — lotyr + Bercan/@bercankilic (2026/2/26)
- **类别**: Strategy
- **关键数据点汇总**: lotyr 分享 NVIDIA EgoScale（research.nvidia.com/labs/gear/egoscale/）+ DrJimFan 的 X thread。Bercan 评论引发讨论："Nvidia has quietly solved robotics end-to-end with their last three releases. Incredible."——提到 DreamDojo（world model with zero-shot generalization）。这个判断可能过于乐观，但反映了社区对 NVIDIA 近期在 Cosmos/Isaac/GR00T 系列发布的密集程度的震撼感

### D107. SmolVLA 社区评价：令人印象深刻但期待更小模型
- **来源**: LeRobot Discord #robotics-papers — 匿名用户 (约 2026/2 月)
- **类别**: Arch, Edge
- **关键数据点汇总**: 社区成员分享 SmolVLA 在 LIBERO 和 Meta-World 上的 benchmark 结果表格（SmolVLA 0.45B/2.25B vs Diffusion Policy / TinyVLA / π0 3.5B），评价："this is old, but smolvla's paper continues to be impressive. I wish they had released the smaller model too tho"（arXiv 2506.01844）。社区对小模型有明确需求——SmolVLA 0.45B 已经在 LIBERO 上 87.3 avg，但用户仍希望有更小的版本

### D108. Real2Render2Real：无需仿真器的机器人数据缩放
- **来源**: LeRobot Discord #robotics-papers — rafa.felix (2026/2/3)
- **类别**: Data, Arch
- **关键数据点汇总**: rafa.felix 分享 Real2Render2Real / R2R2R（arXiv 2505.09601）——"a novel approach for generating robot training data without relying on object dynamics simulation or teleoperation"。解决了 VLA 数据采集的两大瓶颈（遥操昂贵 + 仿真不够逼真），通过渲染而非物理仿真来生成训练数据。与 D105 DreamDojo 的方向互补：DreamDojo 从人类视频学 world model，R2R2R 从真实场景做渲染增强

### D109. IPA 2026：CVPR Interactive Physical AI Workshop
- **来源**: LeRobot Discord #robotics-papers — 匿名用户 (约 2026/2 月)
- **类别**: Strategy
- **关键数据点汇总**: 社区成员分享 IPA 2026: Workshop on Interactive Physical AI（CVPR 2026），评价"this workshop seems super cool if someone is considering to submit papers to CVPR workshops"。NVIDIA 主导（research.nvidia.com/labs/amri/projects/IPA/2026/）。Physical AI 作为 CVPR 2026 workshop 主题出现，说明该方向正从 robotics 社区扩展到更广泛的 CV 社区

### D110. 欧洲机器人创业召集："GPU and hardware poor" 也能做
- **来源**: LeRobot Discord #robotics-papers — tom_primozic (2026/2/26)
- **类别**: Strategy
- **关键数据点汇总**: tom_primozic 发帖："anyone GPU and hardware poor, thinking of starting a robotics startup, and depressed by all the development (passing me by), msg me - ideally based in Europe"——收到 3 条回复讨论串。这反映了 VLA 社区中的一个真实焦虑：大量前沿进展由大公司（NVIDIA/PI/字节）推动，资源有限的个人和小团队感到被边缘化。但实际上 SRPO (#117) 200 步、SO-ARM101 $300、$30 双目相机 (D90) 等方案正在降低门槛

### D111. Holistic Robot Pose Estimation 开源代码（ECCV 2024）
- **来源**: LeRobot Discord #robotics-papers — Pattie (2026/2/27)
- **类别**: Arch
- **关键数据点汇总**: Pattie 分享 Oliverbansk/Holistic-Robot-Pose-Estimation（GitHub 36 stars、6 forks）——ECCV 2024 论文的 PyTorch 实现，实时全局机器人位姿估计。对 VLA 部署的价值：精确的机器人自身位姿感知是闭环控制的前提，这个开源实现降低了视觉伺服的工程门槛

### D112. Contact Anchoring：零样本接触丰富操作
- **来源**: LeRobot Discord #robotics-papers — 匿名分享 (约 2026/2 月)
- **类别**: Arch
- **关键数据点汇总**: 社区分享 Contact Anchoring Paper（CAP）的结果图——零样本 CAP Rollouts 在 Pick/Open/Close 任务上的聚合性能，与其他方法对比柱状图。展示了通过接触锚定实现零样本泛化的可能性。与 D96 接触丰富操作主题呼应——社区对这个方向高度关注

### D113. WhitneyDesignLabs ACT 100% 成功率分享
- **来源**: LeRobot Discord #show-us-what-you-built — WhitneyDesignLabs (约 2026/2 月)
- **类别**: Recipe
- **关键数据点汇总**: WhitneyDesignLabs 分享"Best results so far: ACT (100% success on pick-and-place): 2 cameras"。River 随后询问微调配置细节。100% 成功率在社区中非常罕见，说明 ACT + 双相机 + 精心调参可以达到非常高的可靠性。与 Fei (D86) 的经验一致：数据整理和相机配置是关键

### D114. Alpha.Ars Vibe Coding：LeRobot 相机流挑战
- **来源**: LeRobot Discord #show-us-what-you-built — Alpha.Ars (2026/1/21)
- **类别**: Debug
- **关键数据点汇总**: Alpha.Ars 分享"Vibe Coding"尝试——试图在遥操时同时显示相机画面，但 LeRobot 占用相机后无法同时显示。尝试流式复制但没成功。WhitneyDesignLabs 回复说自己因为系统崩溃开始"blind teaching"。这是一个具体的工程坑：LeRobot 独占相机设备导致遥操时无法实时可视化反馈

### D115. XLeRobot Joycon 控制器遥操方案
- **来源**: LeRobot Discord #show-us-what-you-built — Vector Wang/XLeRobot + feliximax (2026/2/24)
- **类别**: Edge, Recipe
- **关键数据点汇总**: Vector Wang 分享为 XLeRobot 编写的控制器支持——覆盖 VR Quest 3、Xbox 和 Joycon。认为 Joycon 是"最好用的"。feliximax 回复"So cool! So you are using the original parts by Nintendo."社区反响积极。使用游戏手柄做遥操作的优势：便宜、人体工学好、无线、低延迟

### D116. lotyr 对 SO-ARM101 的幽默吐槽
- **来源**: LeRobot Discord #show-us-what-you-built — lotyr (2026/2/24)
- **类别**: Debug
- **关键数据点汇总**: lotyr 看到 XLeRobot 机械臂 demo 后评论："It looks like this arm has a suicidal tendency..."——指机械臂的某些运动姿态看起来不自然/危险。这种社区幽默背后是真实问题：低成本机械臂的运动规划缺乏关节极限保护和自碰撞检测

### D117. Fei 人工干预策略：从 OOD 回到分布内
- **来源**: LeRobot Discord #show-us-what-you-built — Fei (2026/1/22)
- **类别**: Recipe
- **关键数据点汇总**: Fei 分享 YouTube 视频展示人工干预（intervention）如何让策略从 OOD 状态恢复到训练分布内。核心经验："intervention is to take it from OOD back into the distribution, so you need a strong baseline to start with and only tackle edge cases"。与 tms-gvd 的 hackathon 经验对比：tms-gvd 发现遥操→策略切换时动作很乱（OOD 问题），Fei 的解法是先有强基线再做边缘恢复

### D118. eliasab 错误恢复数据采集方法论
- **来源**: LeRobot Discord #show-us-what-you-built — eliasab (2026/1/22)
- **类别**: Data, Recipe
- **关键数据点汇总**: eliasab 提出关键问题："Are you then using the recorded intervention in the training as a demonstration on how to recover from mistakes or not yet? Did you see any performance improvements?"以及"what percentage of your data is recovery?"。Fei 回答大部分恢复轨迹 <5 秒。这是社区中首次系统讨论恢复数据在训练中的占比和采集策略

### D119. tms-gvd Hackathon 经验：遥操→策略的 OOD 问题
- **来源**: LeRobot Discord #show-us-what-you-built — tms-gvd (2026/1/22)
- **类别**: Debug
- **关键数据点汇总**: tms-gvd 在 hackathon 中实现了人工干预，但遇到严重问题：从遥操切换到策略时预测动作非常混乱。怀疑原因是"teleop led to a ood state (due to the policy overfitting too much on training data and corrections being out of it)"。这是一个关键的工程挑战——HiL（Human-in-the-Loop）系统必须解决分布外切换的平滑问题

### D120. WhitneyDesignLabs "Blind Teaching" 经验
- **来源**: LeRobot Discord #show-us-what-you-built — WhitneyDesignLabs (2026/1/20-21)
- **类别**: Recipe, Debug
- **关键数据点汇总**: WhitneyDesignLabs 因为电脑太旧，同时运行相机显示和 LeRobot 时系统频繁崩溃，被迫开始"blind teaching"——不看实时画面直接遥操示教。令人惊讶的是，有经验的操作者即使不看画面也能采集有效数据。对 Linux vs Windows、单臂 vs 双臂、相机管理等差异做了深入分析。另外分享 lerobot-web-interface 项目解决远程遥操问题

### D121. Alpha.Ars 跨平台开发经验：Windows 相机兼容性
- **来源**: LeRobot Discord #show-us-what-you-built — Alpha.Ars (2026/1/20)
- **类别**: Debug
- **关键数据点汇总**: Alpha.Ars 开发的遥操项目最初跨平台兼容，但添加相机支持后不得不专注 Windows 平台。原因：不同操作系统的相机设备访问 API 差异巨大。这是 LeRobot 社区的一个常见痛点——官方主要支持 Linux，Windows 用户需要额外适配工作

### D122. rafa.felix $10 PS4 相机 Zero-Shot Safety 全细节
- **来源**: LeRobot Discord #show-us-what-you-built — rafa.felix (2026/1/27)
- **类别**: Recipe, Edge
- **关键数据点汇总**: rafa.felix 用 YouTube Shorts 展示了完整的 $10 PS4 相机对 SO-ARM101 的 Zero-Shot Safety 评测流程。这是社区中成本最低的安全评测方案之一。与 Vector Wang 的 $30 双目相机方案 (D90) 形成极致低成本硬件方案矩阵

### D123. skpro19 提问：如何用 DreamDojo 训练 SO-101
- **来源**: LeRobot Discord #robotics-papers — skpro19 (2026/2/22)
- **类别**: Recipe
- **关键数据点汇总**: skpro19 在 lotyr 分享 DreamDojo 后立刻提问："Let's say, I want to train a manipulator (so-101) to pick an object and place it inside a bowl using imitation learning."——这个问题代表了社区的核心需求：如何将前沿论文方法（world model）应用到自己的低成本硬件（SO-101）上。学术论文和社区实际之间的落差仍然巨大

### D124. Mahameru 社交媒体传播追踪
- **来源**: LeRobot Discord #show-us-what-you-built — Mahameru (2026/1/30)
- **类别**: Strategy
- **关键数据点汇总**: Mahameru 分享了多个社交媒体截图，追踪 LeRobot 相关内容在主流平台的传播情况。说明 LeRobot 社区不仅在技术层面活跃，也在社交传播层面扩大影响力

### D125. 新用户涌入：LeRobot Discord 16k+ 成员趋势
- **来源**: LeRobot Discord 多频道观察 (2026/1-3 月)
- **类别**: Strategy
- **关键数据点汇总**: 2026 年 Q1 观察到 LeRobot Discord 频道活跃度持续增长（16k+ 成员）。新用户最常见问题集中在：1) SO-101 组装和校准；2) ACT 训练超参数；3) Windows 兼容性；4) 相机选择和配置。#help-general 和 #help-forum 新帖频率明显高于 2025 年下半年

### D126. LeRobot 论文被 ICLR 2026 接收的社区反响
- **来源**: LeRobot Discord #robotics-papers — 多用户 (2026/2-3 月)
- **类别**: Strategy
- **关键数据点汇总**: LeRobot 论文正式被 ICLR 2026 接收后，社区反响热烈。有用户指出论文中看到 @Steven Palma 和 @HF LeRobot 核心团队的名字——说明社区成员对核心开发团队有高度认同感。这也意味着 LeRobot 从"社区工具"升级为"学术认可的平台"

### D127. AI for Industry Challenge：Isaac Lab 人形基准
- **来源**: LeRobot Discord #robotics-papers — EreQ (2026/3/1)
- **类别**: Strategy
- **关键数据点汇总**: EreQ 招募参与 Intrinsic + Open Robotics 联合举办的 AI for Industry Challenge，使用 Isaac Lab（github.com/isaac-sim/IsaacLab/discussions/4315）做人形智能基准测试。Luma 活动页面显示这是 Session 1 的 Kick Off。说明工业界开始用标准化竞赛方式推动人形机器人能力评估

### D128. SmolVLA 基准：社区期待更小的模型
- **来源**: LeRobot Discord #robotics-papers — 匿名用户 (约 2026/2 月)
- **类别**: Edge, Arch
- **关键数据点汇总**: 社区成员分享 SmolVLA 在 LIBERO/Meta-World 上的完整基准数据表——SmolVLA 2.25B 在 LIBERO 上 avg 88.75，在 Meta-World 上 avg 68.24。评价中明确表示希望"released the smaller model too"。社区对 <1B 参数级别的高效 VLA 有强烈需求——这是 #150 KAN-We-Flow 等轻量方法的市场验证

### D129. Hackathon 频道活跃：imitation-learning 赛道
- **来源**: LeRobot Discord #hackathon-imitation-learning — 多用户观察 (2026/1-3 月)
- **类别**: Recipe
- **关键数据点汇总**: LeRobot Discord 有专门的 hackathon 频道矩阵（#hackathon-general、#hackathon-assembly、#hackathon-imitation-learning、#hackathon-reinforcement-learning），说明 LeRobot 社区在组织结构化的学习和竞赛活动。imitation-learning 赛道是参与度最高的

### D130. tips-and-tricks 频道：社区知识沉淀
- **来源**: LeRobot Discord #tips-and-tricks — 多用户 (2026/1-3 月)
- **类别**: Recipe
- **关键数据点汇总**: Discord 内设有 #tips-and-tricks 专门频道用于沉淀实操经验。与 #show-us-what-you-built（展示项目）和 #help-general（新手 FAQ）形成功能分层。这种频道结构设计有助于知识的结构化留存

### D131. Hardware 频道矩阵：硬件碎片化的管理方案
- **来源**: LeRobot Discord 硬件频道观察 (2026/1-3 月)
- **类别**: Edge
- **关键数据点汇总**: LeRobot Discord 维护了细分的硬件频道：#reachy-robot、#robotis-omx、#aloha-arm、#alex-koch-arm、#stretch3-mobile-arm、#moss-arm。每个硬件平台有独立讨论空间。这种设计反映了社区面临的现实：硬件生态高度碎片化，不同硬件的问题差异巨大

### D132. #dev-contributing 频道：核心开发者协作
- **来源**: LeRobot Discord #dev-contributing — 观察 (2026/1-3 月)
- **类别**: Strategy
- **关键数据点汇总**: #dev-contributing 频道是 LeRobot 核心开发者协作的主要场所，位于 Software 类别下（与 #perception-control-and-middleware、#datasets、#training 并列）。这个频道结构说明 LeRobot 正在从"个人项目"转向"社区驱动的软件工程"——有明确的模块化分工

### D133. #discussions 频道：深度技术辩论场
- **来源**: LeRobot Discord #discussions — 观察 (2026/1-3 月)
- **类别**: Strategy
- **关键数据点汇总**: #discussions 频道承载更深入的技术讨论（vs #robotics-papers 的论文分享和 #help-general 的 FAQ）。在这里能找到关于 VLA 架构选择、训练策略辩论、部署经验等高质量长帖。信息密度高于其他频道但帖子频率较低

### D134. #jobs-and-collabs 频道：VLA 人才市场信号
- **来源**: LeRobot Discord #jobs-and-collabs — 观察 (2026/1-3 月)
- **类别**: Strategy
- **关键数据点汇总**: #jobs-and-collabs 频道出现越来越多的 VLA 相关职位招聘和合作邀请。从侧面反映 VLA 从学术研究向产业应用过渡的加速——公司开始需要"会训练 VLA 的人"而不仅是"会做机器人的人"

### D135. 社区频道结构演进：从 deprecated 到 robotics-papers
- **来源**: LeRobot Discord — Remi Cadene (2024/10/17)
- **类别**: Strategy
- **关键数据点汇总**: Remi Cadene（LeRobot 核心维护者）在 2024 年 10 月将 #papers-methods-discussions 频道废弃，引导到新的 #robotics-papers 频道。最后一条帖子是 Cadene 自己分享的 arXiv 论文（学习人类视频的触觉预训练）。频道演进反映了社区的成熟——从泛讨论到专题化组织

### D136. NXP i.MX95 嵌入式 VLA 部署：完整延迟基准与量化策略
- **来源**: NXP × HuggingFace Blog — [Bringing Robotics AI to Embedded Platforms](https://huggingface.co/blog/nxp/bringing-robotics-ai-to-embedded-platforms)（2026 年 3 月）
- **类别**: Edge, Recipe
- **关键数据点汇总**: **首个完整嵌入式 VLA 部署基准**。ACT on i.MX95: ONNX FP32 2.86s → 优化后 0.32s（8.9×），96% 准确率。SmolVLA: 29.1s → 6.15s（4.7×），仍超过异步推理窗口（1.67s）。**最重要工程发现**：VLA 量化不能一刀切——Vision/LLM 耐受 8-bit，但 Flow Matching Action Expert 必须保持高精度（量化误差在迭代去噪中累积）。数据采集 best practice：120 episodes, 10 clusters, 3 相机, 20% recovery episodes, 夹爪加热缩管增摩擦。

### D137. AMD ROCm Edge-to-Cloud VLA 训练管线
- **来源**: AMD ROCm Blog — [Edge-to-Cloud Robotics with AMD ROCm](https://rocm.blogs.amd.com/artificial-intelligence/rocm-blogsblogsartificial-in/README.html)（2026 年）
- **类别**: Recipe, Arch
- **关键数据点汇总**: **AMD 正式进入 VLA 训练生态**。管线：Ryzen AI 9 HX370（边缘采集）→ MI300X（云端微调）→ Ryzen（边缘推理）。ROCm 6.3+, PyTorch 2.7.1, LeRobot v0.4.1 原生支持。Pi0 微调：3000 steps, batch 32, bf16。~50 demo episodes 即可学会 pick-and-place。CES 2026 Lisa Su keynote 提及。

### D138. LeRobot → Embodied AI Infra：人形机器人开发者的生产化反思
- **来源**: Stan Su — [From LeRobot to Embodied AI Infra](https://medium.com/@7thuniversels/from-lerobot-to-embodied-ai-infra-a-humanoid-developers-reflections-in-2026-15a343662182)（2026 年 1 月）
- **类别**: Strategy, Debug
- **关键数据点汇总**: **最诚实的 LeRobot 生产化差距分析**。硬件适配是"物理诅咒"——每种新硬件都要重写 adapter。生产缺失：无安全栈/fleet 管理/热更新/ARM 优化。提出方案：Host mode（实时）+ Docker 微服务（推理）+ Web UI。UMA Robots（前 LeRobot 核心团队）正填补产业空白。核心结论：算法管线 OK，最后一公里基础设施还远未就绪。

### D139. GigaBrain-0.5M*：World Model + RL 后训练 VLA（RAMP 管线）
- **来源**: GigaAI — [GigaBrain-0.5M*](https://arxiv.org/abs/2602.12099)（2026 年 2 月）
- **类别**: Arch, Strategy
- **关键数据点汇总**: RAMP 四阶段管线：World Model 预训练 → Policy 以 WM 预测为条件微调 → 真机部署采集 → WM+Policy 持续迭代。预训练 10,000+ 小时真机数据。Laundry Folding/Box Packing/Espresso 上 +30% 超 RECAP baseline。**World Model 作为 RL 条件输入**而非替代真机交互——是目前最完整的 VLA 自我改进闭环。

### D140. UnifoLM-VLA-0：宇树开源 7B VLA（LIBERO 98.7 分）
- **来源**: Unitree — [UnifoLM-VLA-0](https://unigen-x.github.io/unifolm-vla.github.io/)（2026 年 1 月）
- **类别**: Arch, Recipe
- **关键数据点汇总**: 基座 Qwen2.5-VL-7B + Action Head，340h 真机数据训练。**LIBERO 均分 98.7**（Spatial 99.0, Object 100, Goal 99.4, Long 96.2）——VLA 类最高。Unitree G1 验证 12 类操作。完全开源（代码+权重+数据）。注意：LIBERO 有记忆偏差（D32），真机泛化才是真考验。

### D141. VLM4VLA：VLM 能力 ≠ VLA 下游性能（ICLR 2026）
- **来源**: ICLR 2026 — [VLM4VLA](https://openreview.net/forum?id=tc2UsBeODW)
- **类别**: Arch, Strategy
- **关键数据点汇总**: **直接挑战"更好 VLM = 更好 VLA"假设**。VLM 通用 benchmark 高分不等于好 VLA。Vision encoder 可微调性比 LLM backbone 更重要（与 D33 "freezing vision encoder -32pp" 印证）。选 VLA 基座不要只看 VLM leaderboard。

### D142. SimpleVLA-RL：1 条轨迹冷启动 → RL 拉到 91.7（ICLR 2026）
- **来源**: PRIME-RL — [SimpleVLA-RL](https://github.com/PRIME-RL/SimpleVLA-RL)（ICLR 2026）
- **类别**: Arch, Strategy
- **关键数据点汇总**: LIBERO-Long SOTA 97.6。**极端数据效率**：每任务 1 条轨迹 SFT → RL 从 17.3 拉到 91.7（+430%）。发现"pushcut"现象：RL 自行发现训练数据中不存在的操作模式。RL 后训练路线的开源里程碑。

### D143. DynamicVLA：0.4B 模型攻克动态物体操作
- **来源**: NTU — [DynamicVLA](https://arxiv.org/abs/2601.22153)（2026 年 1 月）
- **类别**: Arch, Data
- **关键数据点汇总**: **VLA 首次系统攻克动态场景**。0.4B 紧凑模型，Continuous Inference + Latent-aware Action Streaming 降低延迟。DOM 基准：200K 合成 + 2K 真机 episodes。动态任务 +188% 到 +440%。数据采集自动化无需遥操。

### D144. WholebodyVLA：人形全身协调操作（ICLR 2026）
- **来源**: OpenDriveLab — [WholebodyVLA](https://github.com/OpenDriveLab/WholebodyVLA)（ICLR 2026）
- **类别**: Arch, Strategy
- **关键数据点汇总**: 从无动作标注第一人称视频学习 latent actions + 专用 loco-manipulation RL policy。AGIBOT X2 验证，超 baseline 21.3%。目前最完整的开源 VLA-for-humanoid 方案。

### D145. AGIBOT WORLD 2026：百分百真机多模态数据集
- **来源**: AGIBOT — [AGIBOT WORLD 2026](https://huggingface.co/datasets/agibot-world/AgiBotWorld2026)
- **类别**: Data, Strategy
- **关键数据点汇总**: 100% 真实环境采集，RGB(D)+触觉+LiDAR+IMU+全身关节。五阶段发布（第一阶段聚焦模仿学习）。OmniHand 灵巧手数据罕见。IROS 2025 Best Paper Finalist + IEEE TRO 2026。

### D146. ICRA 2026 VLA Pipeline Workshop：10,000h 数据 + 真机竞赛
- **来源**: AIRoA — [ICRA 2026 Workshop](https://icra2026vlapipeline.github.io/)
- **类别**: Strategy, Data
- **关键数据点汇总**: **VLA 领域首个大规模真机评测竞赛**。10,000h LeRobot 格式数据，Toyota HSR 平台，每两周真机评测 + 视频反馈。SmolVLA/π0 为官方 baseline。36 队满额。$2000/$1000/$600 奖金。6/5 维也纳。

### D147. LeRobot v0.4.0：Datasets v3.0 + Pi0.5/GR00T N1.5 + 插件系统
- **来源**: HF Blog — [LeRobot v0.4.0](https://huggingface.co/blog/lerobot-release-v040)
- **类别**: Data, Arch
- **关键数据点汇总**: **生态成熟标志**。Datasets v3.0 支持 OXE 级 400GB+ 流式加载。Pi0.5 + GR00T N1.5 原生集成。LIBERO 130+ 任务 + Meta-World 50+ 任务。数据集编辑工具。插件系统简化硬件集成。

### D148. SO-101 Isaac Lab 中 VLA BC + RL：SmolVLA 仿真预训练关键发现
- **来源**: LeRobot Discord #show-us-what-you-built — iterrani (2026/3/17) + [GitHub: MSSergeev/so101-lab](https://github.com/MSSergeev/so101-lab)（13 stars）
- **类别**: Arch, Debug
- **关键数据点汇总**: **Open X-Embodiment 预训练在仿真中完全失效（0%）**——sim 视觉与真实世界差异让预训练 expert 无用。SmolVLA frozen vs unfrozen backbone 几乎没差（70% vs 70-76%），从头训明显更差（56%）。IQL offline RL +12-16% over BC。Flow-Noise PPO pick-and-place 达 90%。VIP reward from ResNet50 Ego4D = reward-free RL 方向值得关注。

### D149. LeRobot v0.6.0 社区路线图发布
- **来源**: LeRobot Discord #announcements — Steven Palma @HF (2026/4/4) + [GitHub #3134](https://github.com/huggingface/lerobot/issues/3134)
- **类别**: Strategy
- **关键数据点汇总**: **v0.6.0 路线图公开 + 主动招募贡献者**。核心方向：新仿真 benchmark、VLA 训练优化、新硬件支持。LeRobot 从 v0.4→v0.5→v0.6 快速迭代，现在是参与核心开发的最佳窗口期。

### D150. Unfolding Robotics：双臂衣物折叠完整开源（100+h 数据, 5k+ GPU 小时）
- **来源**: LeRobot Discord #announcements — Pepijn Kooijmans @HF (2026/4/7) + [Blog](https://huggingface.co/spaces/lerobot/robot-folding)
- **类别**: Recipe, Data
- **关键数据点汇总**: **VLA 社区迄今最完整的双臂操作开源案例**。8 bimanual setups, 100+h demos, 5k+ GPU hours, LeRobot v0.5.1。覆盖全流程：硬件设置→数据采集→训练 recipe→经验教训。Hackathon 折叠 85% SR。社区反应极热（11🚀 7❤️ 7🔥 6🎉）。

### D151. Pi0.5 版本升级陷阱：transformers 库版本导致动作完全失控
- **来源**: LeRobot Discord #help-forum — danstrawbridge (2026/4/5) + Steven Palma 回复
- **类别**: Debug
- **关键数据点汇总**: 升级 LeRobot 后微调 pi0.5 动作变 "very erratic"。**根因：transformers 库版本不兼容**。修复：`pip install transformers==5.3.0`。这种 "silent failure"（模型正常加载但输出完全错误）是 VLA 部署中最危险的问题。**所有 Pi0/Pi0.5 用户：固定 transformers==5.3.0**。

### D152. SO-101 舵机烧毁：高阻抗传播的未知机制
- **来源**: LeRobot Discord #help-forum — mango (2026/4/9) — 16 评论
- **类别**: Debug
- **关键数据点汇总**: 舵机烧毁后替换新电机但高阻抗持续，后突然自行恢复。可能是总线通信问题而非机械故障。16 条评论 = 常见问题。**备用电机是必需品**（D5/D7/D38 已多次强调）。

### D153. Robokin：开源 IK Helper Library
- **来源**: LeRobot Discord #show-us-what-you-built — Dmitri (2026/4/3) — integration 标签
- **类别**: Edge
- **关键数据点汇总**: 开源逆运动学辅助库，简化笛卡尔→关节空间映射。对 VLA 跨硬件部署有实用价值（SmolVLA → Aloha 需要关节翻转/夹爪映射）。

### D154. 开源平行夹爪 for SO-ARM100
- **来源**: LeRobot Discord #show-us-what-you-built — Nikita Bragin (22天前) — 5 反应
- **类别**: Edge
- **关键数据点汇总**: 开源平行夹爪设计，与 D95 Arpeggio Gripper 形成社区夹爪生态。平行夹爪比默认夹爪更适合精密抓取。

---

## GitHub Issues 现场速查（2026-08 窗口，标题级）

> **说明**：以下条目取自 `huggingface/lerobot` Issues 按评论数排序、`created:>2026-07-25` 的结果。
> 其中评论最热的 4 条已逐条打开核实并升格为 #226–#229；下列条目**仅核对到标题与编号，未逐条展开正文**，
> 用途是「知道有这个坑存在」的速查表，引用前请自行打开原 issue。

| ID | 标题（原文） | 一句话为什么值得看 |
|----|------------|------------------|
| [#4153](https://github.com/huggingface/lerobot/issues/4153) | VLA-JEPA default world-model loss leaks future frames into context via shared V-JEPA2 encoder pass | v0.6.0 的招牌 world model policy 疑似有**未来帧泄漏**——影响 #198 的评测可信度 |
| [#4414](https://github.com/huggingface/lerobot/issues/4414) | VLA-JEPA: hard-coded last-layer cond tap reads untrained token positions when `freeze_qwen=True` | 同上，冻结 backbone 时条件抽头读到未训练的 token 位置 |
| [#4390](https://github.com/huggingface/lerobot/issues/4390) | `mujoco>=3.4.0` silently breaks the initial-state premise of `libero_spatial` task 5 | **仿真器版本会静默改变 benchmark 初始状态**——跨论文比 LIBERO 分数前必查 |
| [#4152](https://github.com/huggingface/lerobot/issues/4152) | LIBERO evaluation: initial-state sequence depends on policy termination timing, via `LiberoEnv.step()` | LIBERO 的初始状态序列**依赖策略何时终止**，不同策略拿到的题目不完全相同 |
| [#4393](https://github.com/huggingface/lerobot/issues/4393) | `torch>=2.7` is allowed but bundled torchcodec 0.11 requires `torch>=2.9` symbols (undefined symbol) | 依赖区间写宽了导致 undefined symbol——环境炸了先看这条 |
| [#4415](https://github.com/huggingface/lerobot/issues/4415) | Normalization/unnormalization is silently skipped after Pipeline migration | 已升格为 **#227** |
| [#4451](https://github.com/huggingface/lerobot/issues/4451) | Migration mangles feature names containing an underscore (`observation.environment_state`) | 数据迁移会**改坏带下划线的特征名** |
| [#4156](https://github.com/huggingface/lerobot/issues/4156) | Incorrect weighted aggregation of q01/q99 when merging dataset statistics | 合并多数据集统计量时 **q01/q99 加权聚合算错**——又一个 stats 类静默故障 |
| [#4388](https://github.com/huggingface/lerobot/issues/4388) | `lerobot[libero]` installs cleanly on macOS but silently omits the simulator | macOS 上装得很干净，**仿真器其实没装上** |
| [#4259](https://github.com/huggingface/lerobot/issues/4259) | DiffusionPolicy: no EMA support — the reference implementation enables it in every config | **复现 Diffusion Policy 分数对不上的一个已知原因** |
| [#4205](https://github.com/huggingface/lerobot/issues/4205) | [ROCm] NotImplementedError for bilinear interpolate on uint8 tensor in SmolVLA | AMD 用户跑 SmolVLA 的拦路虎（对照 #9 ROCm 配方） |
| [#4245](https://github.com/huggingface/lerobot/issues/4245) | [X-VLA] Camera count mismatch and rollout arguments parsing issues with 2-camera setups | X-VLA 双相机配置的参数解析问题（对照 #34） |
| [#4168](https://github.com/huggingface/lerobot/issues/4168) | ACT+ALOHA, can only get 66% success | 期望校准：ALOHA 上 ACT 卡在 66% 的社区现场 |
| [#4169](https://github.com/huggingface/lerobot/issues/4169) | Recommended Training Dataset for the Pi0.5 Quickstart | Pi0.5 快速上手该用哪个数据集——新手高频问题 |
| [#4229](https://github.com/huggingface/lerobot/issues/4229) | How to improve robot motion during model evaluation | 评测时动作抖 / 卡的通用求助帖（对照 #15 ACTSmooth） |
| [#4309](https://github.com/huggingface/lerobot/issues/4309) | `LeKiwi.send_action` raises KeyError on every command when `max_relative_target` is set | LeKiwi 用户设了限幅就每条命令报 KeyError |
| [#4231](https://github.com/huggingface/lerobot/issues/4231) | `lerobot-eval --env.type=robocasa` fails despite following RoboCasa documentation | 文档照做仍然跑不起来的 RoboCasa 评测 |
| [#4413](https://github.com/huggingface/lerobot/issues/4413) | High rate observation recording | 高频观测录制的需求与讨论 |
| [#4400](https://github.com/huggingface/lerobot/issues/4400) | Add TurboVLA as a lightweight, non-VLM VLA policy (plugin first) | 轻量、**不含 VLM** 的 VLA 路线进入插件生态 |
| [#4240](https://github.com/huggingface/lerobot/issues/4240) | Add a `ChunkSafetyProcessorStep` to validate predicted action chunks before execution | 执行前校验 action chunk 的安全层提案——真机用户值得跟 |
| [#4368](https://github.com/huggingface/lerobot/issues/4368) | RFC: ROS 2 integration strategy (ecosystem survey and proposed direction) | **LeRobot × ROS 2 的官方方向讨论**，做产品化的必看 |

---

## GitHub Issues 周报

> 由 `scripts/en-vla-collector/github_vla_issue_collector.py` 在用户云端生成，
> 输出到 `reports/github_vla_issues_weekly.md`，通过 sync-vla-handbook 同步后可被读取。

---

*v3.11 — 2026-08-21 更新。Blog 229 条（#1-#229）+ Discord 154 条（D1-D154）+ GitHub 速查 21 条，共 404 条。v3.11 新增（2026-07-31 → 2026-08-21 窗口）：#211 LeRobot v0.6.1（`lerobot.types`→`lerobot.lerobot_types` 破坏性改名 + wall-x/X-VLA 改为继承原生 Transformers 类，正面缓解 D151 版本地狱）、#212 Grabette/Gripette（490€/120€ BOM，UMI 路线开源工程化，RTAB-MAP SLAM 自动校验轨迹）、#213 GR00T 1.7 端到端平台（3B / Apache-2.0 / 32K+8K 小时 / Cosmos-Reason2-2B 换掉 Eagle / DROID-F6 +61%；**WBC 选择即训练分布**；`--num_steps 600` 只是 smoke test）、#214 Isaac Teleop + GR00T 1.7 进 LeRobot（SO-101 完整命令行；相对动作要排除 gripper；推理 n_action_steps=8 < 训练 16）、#215 GS-VLA（**相机微小位移 → LIBERO ~90%→最坏 ~10%**，4M 参数 canonicalizer 前置冻结 VLA）、#216 ReflexVLA/ReflexBench（延迟可配置的动态交互 benchmark + CUDA Graph replay）、#217 The Embodiment Gap（TMLR，提出适配工作量报告框架：只报成功率看不出换本体还剩多少活）、#218 FACT（接触密集失败拆成精度失败与力失败两个根因，66% vs 41%，近 2,500 次真机 rollout）、#219 G0.5（单自回归流出推理+动作，真机 76.7% vs π0.5 53.3% vs GR00T-N1.7 24.4%）、#220 GigaBrain-0.7（三系统架构 + 37,000 小时）、#221 EcoVLA（20Hz 约束下能效 +236%）、#222 ViTaR（冻结 VLA + 有界触觉残差，+30.6pp）、#223 τ₀-VLA（world-model 引导的 test-time 搜索，40,115 小时）、#224 CDPR 零示范 RL 对齐（PPO→GRPO 续训）、#225 proprioception 接法消融（45 原子任务 + 1→96 帧扫描，**仅记录设置未核结论**）、#226 `compile_model=true` 静默 NaN（SmolVLA 已复现 / PI0 / PI05 结构性受影响，torch 2.11 default 仍坏）、#227 Pipeline 迁移后归一化静默跳过（`smolvla_base` 多数据集前缀 key 对不上）、#228 `delta_timestamps` 图像列全解码（ACT 3.0→13.6 it/s）、#229 `pi05_libero_base` 0%（base ≠ 可用策略）。**v3.11 主线**：(1) **静默故障成为本期最大主题**——编译误编、归一化跳过、依赖符号、仿真器版本改初始状态，共同点都是**不报错但结果全错**；(2) **相机与本体的物理配置第一次被量化**（#215 的 90%→10%、#217 的 embodiment gap、#213 的 WBC 即训练分布）；(3) **base checkpoint ≠ 可用策略**已由 #210/#227/#229 三次独立印证；(4) 具身基础模型预训练门槛稳定在 3–4 万小时真机数据量级（#213 #220 #223）。**采集缺口**：Discord 连续第二期未采集（Chrome 未登录），下次运行前必须补登录；Physical Intelligence 博客本期无新文（最新仍为 2026-04-16 π0.7），已直接核对。*

*v3.10 — 2026-07-31 更新。Blog 210 条（#1-#210）+ Discord 154 条（D1-D154），共 364 条。v3.10 新增（2026-05-29 → 2026-07-31 窗口，跨两个月补采）：#198 LeRobot v0.6.0（VLA-JEPA / LingBot-VA / FastWAM 三个 world model policy、GR00T N1.7 替换 N1.5、六大新仿真 benchmark 统一 CLI、lerobot-rollout DAgger 部署即数据采集）、#199 官方算力硬件指南（VRAM 分档 + 4×H100 实测 + dataloading_s/update_s 诊断法）、#200 LingBot-VLA 2.0（6B Apache-2.0、6 万小时、55D 统一动作空间、MoE action expert、GM-100 超 π0.5）、#201 Gemini Robotics 2（同 checkpoint 控三本体、多指灵巧仅 32–44%、On-Device 2 <200 示例适配新本体且演示平台含 SO101）、#202 NVIDIA WAM 路线全景（WAM 与 VLA 二选一的选型建议 + co-train 视频目标是关键）、#203 vla.cpp（ggml 级 VLA 运行时，BitVLA 1.3GiB 跑 100%）、#204 Embodied.cpp（VLA+WAM 多速率闭环运行时）、#205 RhinoVLA（token 效率而非参数量决定边缘延迟，Huixi R1 11.69Hz）、#206 Realtime-VLA FLASH（投机推理 58.0ms→19.1ms，不改模型只改调度）、#207 leaderboard 现状（RoboArena 刷榜 + π0.5 粘性）、#208 Figure 03 进 BMW Spartanburg（84s 节拍 / >99% 放置 / 零干预目标）、#209 Robometer + TOPReward（奖励模型进入一等 API）、#210 SmolVLA 微调失败对质（先用 ACT 验数据）。**v3.10 主线**：world model / WAM 从论文走进主流框架（#198 #202）、边缘部署基础设施标准化（#203 #204 #205 #206）、评测公信力危机（#207）、工业化验收标准显性化（#208）、成功检测自动化（#209）。**采集缺口**：本期 Discord 未采集（Chrome 未登录），需在下次运行前补登录。*

*v3.8 — 2026-04-24 更新。Blog 185 条（#1-#185）+ Discord 154 条（D1-D154），共 339 条。v3.8 新增（2026-04-17 → 2026-04-24 周窗）：#174 SnapFlow（arXiv 2604.05656，one-step flow matching，9.6× 去噪提速，直接解 π0/π0.5/SmolVLA 10-step ODE 延迟瓶颈）、#175 NVIDIA National Robotics Week 2026（NemoClaw、OceanSim、RoboLab、Doosan、mimic-video 联合 push）、#176 mimic-video（Video-Action Model，10× sample efficiency）、#177 VLA Foundry（arXiv 2604.19728，统一 LLM/VLM/VLA 训练框架）、#178 VLAJS（arXiv 2604.13733，Jump-Start RL + VLA regularization）、#179 HEX（arXiv 2604.07993，humanoid-aligned experts whole-body VLA）、#180 COIN（arXiv 2604.16886，Chain of Interaction benchmark，替代 LIBERO saturation）、#181 OneVL（arXiv 2604.18486，one-step latent reasoning）、#182 1X Redwood AI + World Model（160M 消费级 VLA）、#183 LeRobot Worldwide Hackathon（SmolVLA 拿冠军 + 青少年获奖）、#184 Dexora（ICRA 2026，36-DoF 双臂，12.2K episodes 数据集）、#185 E-VLA（arXiv 2604.04834，event camera VLA，黑暗/模糊场景鲁棒）。v3.8 主线：flow matching 一步化走向成熟（#174）、动作生成从 diffusion→one-step（#174 #181）、仿真/数据双增长（#175 #176 #184）、benchmark 代际更替（#180 替代 LIBERO）、边缘/消费级 VLA 起势（#182）、感知新模态（#185 event camera）。*
