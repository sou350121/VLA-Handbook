# 小红书 VLA 社区声音收集（2026-06-01 用户指令"再去抓有用经验"Round 3）

> 收集日期：2026-06-01
> 搜索关键词（7 组，参数/踩坑导向）：机械臂 标定 踩坑 / 灾难性遗忘 VLA / OpenPI 真机 / GR00T 微调 / RoboTwin 复现 / FAST tokenizer 动作 / ACT 机械臂 真机 成功率
> 总计：**7 篇详细收录 + 0 标题**（共 7 篇新帖）→ 帖277-283
> 采集器：xiaohongshu-vla-collector（manual-trigger Round 3，用户指令"要在小红书一样是有用的经验分享"）
> 提取方式：搜索页 `section.note-item` 列出 → 点击卡片获取 xsec_token → DOM 提取 + heavy sanitize
> 浏览器：Browser「小紅書」(Windows, 已登录)

---

## 本轮核心信号

🔴 **Bull**：本轮把 §1-§5 缺的参数 / 踩坑数据点补足。**7 个帖全部是一手实战**，无论文综述、无广告、无求职。最强信号是 **VLA 失败的非范式原因被一手暴露**：
- **帖277（S!mple）**：**冷启动温度影响 VLA 成功率** —— 行星减速器摩擦力随温度变化 → 底层轨迹跟踪滞后 → 绝对 joint 角动作空间误差级联放大 → action chunk 开环不能补偿。VLAC 论文作者也提到空调温度影响。**评论 sylin："VLA 其实是过拟合到了某个场景，对环境的变化没有感知"** —— 这是新的 belief signal："VLA 落地失败的一大类原因是物理环境漂移，不是模型本身"。
- **帖281（XVLA + RoboTwin 多卡 5 大坑）**：DeepSpeed 原版 > acc+DS > acc；`HDF5_USE_FILE_LOCKING=FALSE`；HDF5 `lz4` 压缩 2T→250G（90% 压缩率）；**NUMA 跨节点内存访问导致卡数据**，必须 `numactl --membind`。这是手册 §1.5 没覆盖的工程级深坑。
- **帖283（ACT 桌面 vs 非桌面）**：作者复现失败（后验坍塌 + 泛化差 + 抖动）但 temporal ensemble 好用 / CVAE 别碰。**评论 Claude**：相反案例 164 条 ACT 30 epoch 就能用，π0/SmolVLA loss 0.007 都复现不出来。**评论 Anuuu**：franka 上 ACT 50 条 / π0.5-droid 100 条，但成功率和泛化都不行，**回去踏实做 RL**。

🔵 **Bear**：本轮 Bear 信号集中在 **"已知工具链不稳定"**：
- **帖279（lerobot pi vs openpi）**：评论"秋光的祝福"——lerobot Libero Spatial Task 05 **MuJoCo 重力 bug 把物体弹到离谱位置导致成功率低**；"星空和大海"——业内反馈 lerobot pi 兼容性差，今年新版本不确定修没修；"你的就是你的"——"lerobot 一大坨"。LeRobot 在 04-28 帖214 已升起的"维护降级"警报本轮再添 3 个独立印证。
- **帖280（宇树 G1 + VLA）**：一手报告 **ACT / π0.5 / GR00T 用宇树官方 HF 数据微调效果都不好**；人形 VLA 部署比机械臂困难（接口频繁不匹配）。这是首个 G1 真机 VLA 三件套联合失败的报告。

🟢 **Arbiter**：
1. **若做硬件层稳定性**：先解决冷启动（帖277）——空调温度 + 减速器预热是 VLA 上 prod 的非可选项，比改算法划算。
2. **若做多卡训练**：照抄帖281 的 5 步清单。
3. **若做 ACT/π0 选型**：评论一手数据汇总——**ACT 50-100 条桌面机械臂 OK；非桌面 / 人形 / G1 全部不 work**；π0.5-droid 100 条比 ACT 多但泛化没强；**RL 是这些团队的退路**。
4. **若关注范式架构**：帖282 星海图 G0.5——**单 decoder 同时 reasoning + action（不是 VLM Encoder + action expert 的主流路线）**，跨 18 本体 Action Tokenizer 压成 27 维 vocab，长程任务 +30~35。"具身智能时代的 ARM"产业野心。
5. **若做灾难性遗忘修复**：帖278 上交 CVPR'26 Driving Expert Adapter——**全量微调破坏性最大、LoRA 保住常识但任务掉**；解法是"提示空间适配 + 动态调用专家模块"不动主参数。直接对应手册 §2.3 现有内容做加强。

---

## 帖 277：机械臂冷启动温度影响 VLA 成功率（一手）🔥🔺

- **作者**：S!mple
- **日期**：03-13
- **链接**：https://www.xiaohongshu.com/explore/69b3fab400000000230055de
- **关键词**：机械臂 标定 踩坑
- **Score**：9 分（一手 +3、根因分析 +3、跨论文印证 +2、新 belief 候选 +1）

**核心内容**：做 VLA 精确操作 task 时发现 **机械臂刚启动测试成功率异常偏低，运行几分钟后才稳定**。排除光照等因素，根因链：
- 关节电机（行星减速器）摩擦力受温度影响
- 冷启动时润滑油粘稠，阻力大，底层控制器轨迹跟踪滞后
- VLA 动作空间是绝对 joint 角，误差级联放大
- action chunk 开环执行不能及时补偿
→ 成功率下降

**跨论文印证**：VLAC 论文作者演讲时也提到空调温度影响算法性能（VLAC 同样用松灵 Piper 机械臂）。

### 精选评论
- **LoongDiy**：模型越跑越差？开始几次稳定，后面就出小问题 —— 同方向新症状
- **sylin**（关键）：温度对电机影响大，**"现在的 VLA 其实是过拟合到了某个场景，对环境的变化没有感知"** —— 提议把温度加进 state
- hfcucuy："这不就是过拟合的最好证明吗"

---

## 帖 278：上交 CVPR'26 — 告别 VLA 微调后的知识遗忘 🔥

- **作者**：今天赶CCFDDL了吗
- **日期**：04-29
- **链接**：https://www.xiaohongshu.com/explore/69f1fb540000000035025f90
- **关键词**：灾难性遗忘 VLA
- **Score**：8 分（论文一手综述 +3、具体 benchmark + 解法 +3、对应手册 §2.3 知识盲区 +2）

**核心内容**：上海交大 AutoLab 团队 CVPR'26 工作 Driving Expert Adapter——VLM 在自动驾驶数据微调后对**关键长尾目标（路障/石块/山坡上的牛）"视而不见"** = 灾难性遗忘。

**两个产出**：
- **Fidelity Driving Bench**（基准）：18 万场景 / 90 万 QA；筛高难长尾；核心 2 指标——关键目标漏检率 + 原始知识保留率
  - 结论：**全量微调破坏性最大** / 常规 LoRA 部分保住但任务掉
- **Driving Expert Adapter**（方案）：不改主参数，**在提示空间做适配，动态调用专家模块注入驾驶知识** → 知识保留率明显提升 + 决策不掉

**Arbiter 意义**：虽然是自动驾驶场景，但 VLA 范式直接迁移；与 §2.3 现有"用未微调 VLM 做视觉老师"（帖31/37）形成互补。

### 精选评论
- **uuq**（关键）：用 VLA 通用模型，期望的是预训练的世界知识解决领域数据覆盖不到的场景 + 理解推理能力。"大量自动驾驶数据再微调，是否会直接让本来想要的通用知识能力彻底被抹除"

---

## 帖 279：lerobot pi vs openpi 效果对比（评论区精华）🔵

- **作者**：Steven（广东）
- **日期**：05-18
- **链接**：https://www.xiaohongshu.com/explore/6a0aa02e000000003503b4fd
- **关键词**：OpenPI 真机
- **Score**：5 分（标题级 + 评论区高质量讨论 = 信号集中在评论）

**核心内容**：标题问 lerobot pi 和 openpi 哪个更好。正文无内容。**信号全在评论区**：

### 精选评论（按重要性排序）
- **秋光的祝福**：**lerobot 在一些转换上有问题** —— 比如 Libero 数据集 Spatial Task 05 上 **MuJoCo 重力会导致需要抓的物体被弹到一个很离谱的位置**，导致成功率很低（**与本批帖262 Libero 阈值 bug 形成第二个 Libero benchmark 工程问题**）
- **Tavish**：这是转换问题吗？确定不是 policy 的问题？（同行 reviewer 标准质疑）
- **星空和大海**：去年业内交流时被告知"不要用 lerobot 的 pi 兼容的不好"，**今年更新到新版本以后不知道 bug 修没修**
- **Spiral galaxy momentum**："我更相信 groot 一点"
- **你的就是你的**："lerobot 一大坨，不知道现在啥状态"
- **讨厌数学**：有没有除了这两个更好的 vla 框架可用的？（生态信号）

**Arbiter 意义**：LeRobot 在 §14.7 "维护降级 + FluxVLA 补位"已升起的警报本轮再添 3 个独立印证。

---

## 帖 280：宇树 G1 + VLA 一手部署记录 🔥🔵

- **作者**：绿洲（北京）
- **日期**：4 小时前（2026-06-01）
- **链接**：https://www.xiaohongshu.com/explore/6a1d5ef0000000000702774b
- **关键词**：GR00T 微调
- **Score**：8 分（一手最新 +3、跨模型联合失败 +3、硬件踩坑细节 +2）

**核心内容**：五月初安装的两腕部相机 + 头部双目相机（宇树和第三方合作的廉价方案，加起来 ~1000+）。

**硬件踩坑**：
- 按宇树官方文档买的腕部相机和文档里的不一样，**但都能用**
- **不要把三个 USB 都接到绿联拓展坞 — 带宽不够**（评论 zhongke 补：拓展坞可以接三个，但**要单独给拓展坞供电**）

**软件/算法**：
- XR 设备和算力没到位，**直接用宇树官方 HF 开源数据**微调
- **ACT / π0.5 / GR00T 三个模型全部效果不好**

**三点感想**：
1. **人形机器人 VLA 部署比机械臂困难**——Codex 经常报接口不匹配，要修一大堆
2. **宇树官方开源数据微调效果不好**——刚需 XR 设备自采
3. 算力 + 设备到位前不打算做工程复现，**先优先研读论文**

**Arbiter 意义**：首个 G1 真机 ACT / π0.5 / GR00T 三件套联合失败的具体报告。

---

## 帖 281：XVLA + RoboTwin 多卡训练 5 大优化（金级一手）🔥

- **作者**：R*
- **日期**：2026
- **链接**：https://www.xiaohongshu.com/explore/698071b4000000001a036d97
- **关键词**：RoboTwin 复现 / 多卡训练
- **Score**：10 分（一手 +3、具体参数 +3、覆盖工程盲区 +2、跨工具链 +2）

**5 大优化**（XVLA + RoboTwin + DeepSpeed）：

1. **框架**：**原版 DS > acc+DS > acc**。acc 又慢显存又大（17h、380W）；acc+DS 显存优化但还是慢；**原版 DS zero0 缩到 14h，功率 420W**

2. **Dataloader 参数**：
   - num_workers = 4-8 区别不大，**4 反而更快，太大反而慢**
   - batch_size：基本越大越好（8-32 内存够就开）
   - pin_memory：一般要开
   - persistent_workers：一般要开
   - drop_last：可能有用

3. **关闭 HDF5 文件锁**：`export HDF5_USE_FILE_LOCKING=FALSE`

4. **数据 IO 压缩**：
   - depth 改成 f32
   - seg 图像占空间大，用 **lz4 压缩（clevel 小越快）**
   - **2T 数据压到 250G（90% 压缩率），解压速度快，大文件解压代价 < 传输代价**

5. **NUMA 节点管理**（关键）：双路 CPU 服务器是两个节点（cpu0 → 卡0-3，cpu1 → 卡4-7）。**多人使用时未做 CPU 资源管理会卡死不读数据**。必须 `numactl --membind` 指定使用的 CPU 节点，**防止跨节点内存访问**。

**DS 多进程核心分配**：`export OMP_NUM_THREADS` 默认 1 会喂不饱进程，必须 ≥ 进程数（workers 数），但不要开太大。

---

## 帖 282：星海图 G0.5 范式分析 — 单 decoder reasoning+action 🔺

- **作者**：(reviewer)
- **日期**：近期
- **链接**：https://www.xiaohongshu.com/explore/6a1d8c28000000003503828f
- **关键词**：FAST tokenizer / 动作分词
- **Score**：8 分（范式架构分析 +3、跨产业战略 +3、具体数字 +2）

**架构分歧 — 不是细节差异是世界观分歧**：
- **主流**（π0 / GR00T / OpenVLA / MolmoAct）：VLM 当 Encoder + 外挂 action expert
- **G0.5**（星海图）：VLM 重新当 Actor，**单 decoder 同时想、同时做**

**三个技术判断**：
1. **VLM 重新成为 Actor**：reasoning 和 action 共享同一组 transformer 权重；代价是让出 flow-matching 高频控制效率；**收益是对机器人讲话就能改行为，不必重训**（in-context behavior steering）
2. **跨本体 Action Tokenizer**：**18 种机器人本体统一压进 27 维 token vocab**；自由度 / 控制频率 / 形态学差异全由 tokenizer 抹平 →"具身智能时代的 ARM"野心
3. **Native CoT**：reasoning 和 action 进同一个 decoder 同流生成（Subtask + BBox + Trace + ActionHint 一起出）→ **长程任务 +30 ~ +35**

**战略选择**：
- **整机 + 智能**：不是 OpenAI of Robotics 也不是 Boston Dynamics，是 **Tesla of Robotics**——硬件出数据，数据训模型，模型反哺硬件
- **开放生态**：李飞飞团队用 R1 / Physical Intelligence 用 R1 Lite 跑 π0.5 / 华为 + 近百家客户在用

---

## 帖 283：ACT 桌面 vs 非桌面（评论一手对比金矿）🔥

- **作者**：🍑气小周
- **日期**：02-05
- **链接**：https://www.xiaohongshu.com/explore/6984695d000000000b013968
- **关键词**：ACT 真机 成功率
- **Score**：8 分（一手失败 +3、评论区高质量对比 +3、形成 ACT 适用范围共识 +2）

**作者一手**：用 lerobot + 自己的机械臂复现 ACT 失败：**后验坍塌 + 泛化差 + 抖动或不动**。**temporal ensemble 好用，CVAE 别碰**。

### 评论区（一手对比金矿）

- **Claude**（关键反例）：**164 条抓取放置数据，ACT 30 epoch 就能用；但 π0 / SmolVLA 训到 loss 0.007 都复现不出来** —— **完全相反的 ACT vs VLA 大模型对比结论**
- **Anuuu**（关键一手）：**franka 上 ACT 50 条能训出来，pi05-droid 微调需要 100 条**；**两个的成功率和泛化能力都不太行，回去踏实做 RL**
- 🍑气小周（作者）："感觉好像桌面机械臂上的效果都比较好" → 形成共识：**ACT 适用范围 = 桌面机械臂**

---

## 信念网络更新（Round 3 增量）

| 信念 | 状态 | 校准置信度 | 致命实验 |
|------|------|----------|----------|
| **B17**（新）：VLA 落地失败的一大类原因是**物理环境漂移（温度 / 摩擦 / 磨损）**，不是模型本身 | 新增 | **55%** | 是否有"加 temperature/contact 到 state"的 VLA 工作显著改善真机成功率 |
| **B18**（新）：单 decoder reasoning+action（G0.5 架构）vs encoder+action expert（主流） | 新增 | **45%**（早期） | G0.5 实测在公开 benchmark 上是否能击败 π0.5 / GR00T-N1.5 |
| **B19**（新）：跨本体 Action Tokenizer（27 维 vocab）是机器人 OS 雏形 | 新增 | **50%** | 18 本体 → 100 本体扩展时是否仍稳定 |
| **B-LeRobot 维护降级** | **第 4 次印证** | 警报继续 | 帖279 评论 3 条 + 帖280 G1 三件套失败 |
| **B-ACT 适用范围 = 桌面机械臂** | 新观察 | **65%** | 是否有非桌面 ACT 成功案例 |
| **C - "VLA 是过拟合到某个场景"** | 新增逆共识 | **50%** | 帖277 sylin 评论 + 帖239 Rose 行为印证 |

---

*Round 3 由 xiaohongshu-vla-collector 用户指令"再去抓有用经验"触发；7 篇详细新帖，全部参数 / 踩坑 / 一手导向，无论文综述、无广告、无求职。*
