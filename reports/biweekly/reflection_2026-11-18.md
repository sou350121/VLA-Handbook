🤔 *双周反思* | 2026-11-05 – 2026-11-18

_读完没立场 = 这两周在消费而不在研究_

本期先划一条边界：可核验窗口只有 11-13 至 11-18 共 6 天，评级样本 280 篇里 ⚡ 为 0。看完下面每个推断，你都要先问一句「这 6 天能支撑到哪一步」。

━━━ 趋势与判断 ━━━

1️⃣ 本期同时出现三条「数据从哪来」的论文线：*ESI* 把 LLM–VLA 协作过程转成后训练数据、*Agentic-Teacher* 用编码 agent 针对失败模式定向造示范、*TouchTwin* 用真实触觉做 real-to-sim-to-real 放大。但你手上有哪一条能接到自己的训练脚本上跑起来？如果一条都接不上，是数据路线不成熟，还是你根本没在跟踪可复现的代码？

2️⃣ 榜单侧：*LIBERO* standard-closed 与 standard-opensource 双双 99.8（*LaST-R1*）、CALVIN ABC-D 4.78（*MMaDA-VLA*）被判饱和；同期 *How Much of Your VLA Is kNN* 指出 LIBERO 上 95%+ 成功率的 VLA 有相当部分能被简单近邻检索复现。你的下一个消融实验，还敢把 LIBERO 分数当主要证据吗？你现在真正信任的 held-out 标准是什么？

3️⃣ 开源权重这边，*Runway Praxis-1* 以 Apache 2.0 放出 27B MoE 世界动作模型，但 46.2% 的 RoboLab-120 成绩全部是自报；*AIRoA-0* 用 π0.5 基座 + 16,151 小时手持 Yubi 夹爪演示；*NVIDIA GR00T 2* 的「5 条演示、50 任务 95%」来自二手新闻稿。三个数字里你选一个当基线做复现，你选哪个？说清你判断可信度的依据，别说「都要看」。

4️⃣ 如果这半年只能押一条数据来源：手持人类演示（AIRoA-0 / *LeRobot* 300TB 流式 Hub）、触觉放大（TouchTwin / *DexSeed* / *TouchBridge*）、还是世界模型当 RL 仿真器（*WoVR* / Praxis-1）？三者都在本期拿到公开证据，选一个，并说出你放弃另外两条的代价。

5️⃣ 触觉族 14 天计数 16、7 天 4，本地评级样本里明显冷却；但校准记录 V-002（触觉将从配件变必需模态）本期仍是 strengthened，置信度升到 0.66。同一批数据得出相反结论，你信哪一个？或者更该问的是：这套方法族计数口径到底在量什么？

━━━ 技术追问 ━━━

🔬 本期 *Outcome-Balanced PPO and Policy-Induced Flow Rectification* 与 *Concertina*、*E0*（Tweedie 离散扩散）、*ODEAction* 全都绕不开「动作头怎么生成」这一层。你能否当场说清：flow matching 策略、离散扩散动作头、回归式 action chunk 三者在多峰动作分布上的取舍差别？说不清就先读 flow matching 的 ODE 采样推导再谈 RL 后训练。

🔬 「冻结主干、只训动作专家」是本期反复出现的配方（*NEO-VLA* 报 90.9% vs Qwen3-VL 79.5%、*Should the Action Expert Train the VLM?* 专文讨论）。请你回答：动作专家应不应该把梯度回传到 VLM 主干？给出你的实验设计和预期结果，不允许写「视情况而定」。

🔬 *Is Your Robot's Memory Just a Clock?* 说记忆增强 VLA 的「记忆」常退化成时间步标记。你当前跑的记忆或长上下文 VLA，能设计一个区分「真回忆」与「时间捷径」的对照实验吗？如果设计不出来，你的记忆模块结论就都不可信。

🔬 推理侧本期聚集了 *VSpark*（投机解码）、*ActPrune*（动作相关 token 剪枝）、*Where Is VLA Inference Time Spent?*（瓶颈剖析）。你的部署里控制频率的瓶颈究竟在视觉编码、骨干前向还是动作生成？在接受任何加速插件前，你得先有自己的剖析数据。
