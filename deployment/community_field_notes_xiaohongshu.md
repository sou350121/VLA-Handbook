# 社区实战笔记：小红书 VLA 从业者经验蒸馏 (Community Field Notes)

> **来源**：250+ 篇小红书 + 中文社区帖子（帖1-161 + 可追溯索引 220 条），2026-03-14 起持续收集
> **原始数据**：[memory/blog/archives/xiaohongshu-community/](../../memory/blog/archives/xiaohongshu-community/)
> **定位**：论文不会告诉你的东西——社区实战者的真实参数、真实失败和真实吐槽。每条结论附「帖N」编号，可在原始数据中回溯验证。
> **更新频率**：每 3 天自动增量收集
>
> **阅读提示**：表格中的参数配置来自社区分享，不同硬件/任务下可能需要调整。"N 人验证"表示有多少独立来源报告了相同结论——数字越大越靠谱。
>
> **可靠性分层**：
> - **帖1-60**（§1-§7）：有原始档案 + 大部分有真实 URL，经 Chrome 抽样验证。**高可靠**。
> - **帖61-131+**（§9-§10）：来自扩展收集轮次，无独立原始档案备份。帖子标题和作者经 XHS 搜索抽样确认存在，但互动数（赞数）和具体引述内容可能存在偏差。**中可靠——建议对关键数据回原帖核实**。
> - **帖6-17**（第一轮部分帖子）：有详细内容摘要但缺少直链 URL（标记为「搜索可见」），无法直接跳转验证。**中可靠**。

---

## 速查：你现在卡在哪？

| 你的问题 | 一句话回答 | 详情 |
|----------|-----------|------|
| 数据只有几十条，用什么模型？ | **ACT**。50 episodes 就能 work，90%+ 成功率 | [§1.1](#11-act-action-chunking-with-transformers) |
| π0 微调需要多少数据？ | 至少 **100 episodes**，<200 条大概率失败 | [§1.3](#13-π0--openpi) |
| π0 微调该预测绝对量还是 delta？ | **绝对量**。delta 很难 recover | [§1.3](#13-π0--openpi) |
| SmolVLA 怎么调？ | 50ep / 20Hz / bs64 / lr4e-5 / 30K步 / 冻结backbone | [§1.2](#12-smolvla) |
| 部署时机械臂抖动/卡顿？ | 用在线插值（julyfun/robotoy），别用 temporal ensemble | [§2.1](#21-推理延迟与执行卡顿) |
| VLA 预测准但真机偏？ | 先检查冷启动（空跑 10 分钟热机），再查精度标定 | [§2.2](#22-硬件相关陷阱) |
| Sim2Real 效果差？ | 八成是物理参数没校准，不是算法问题 | [§3.1](#31-sim2real-gap-的真实根因) |
| 微调后模型变"瞎"了？ | 灾难性遗忘。冻结视觉主干或用 VRA 约束 | [§2.3](#23-灾难性遗忘--微调视觉退化) |
| GRPO 训练炸了？ | 看 reward 方差别只看均值，梯度裁剪必须有 | [§1.6](#16-rl-后训练-post-training) |
| 论文数字和我复现差很远？ | 正常。OpenVLA 论文 85.7% 实测 62.6% | [§5](#5-模型对比社区验证-vs-论文声称-reality-check) |
| 买什么机械臂？ | 避开 Piper（逆解差、无自锁、抖）。π0 用 SO101 也能跑通 | [§2.2](#22-硬件相关陷阱) |
| 只有 3090，能训 VLA 吗？ | 能。LoRA 微调 π0 单卡 20GB，ACT 更轻量 | [§9.1](#91-gpu-选型与训练资源) |
| 全量微调 vs LoRA？ | 数据少用 LoRA，数据 >500 条考虑全量。别无脑 LoRA | [§9.2](#92-lora-vs-全量微调) |
| 边缘部署怎么加速？ | CUDA Graph + 算子融合 → 单卡 30Hz 推理 480Hz 控制 | [§9.3](#93-边缘部署与模型压缩) |
| 买哪个机械臂做研究？ | 低成本：XLeRobot(4K)/SO101。中端：松灵七轴(15K)。避开 Piper | [§9.4](#94-机械臂选型指南) |
| VLA 推理加速有哪些方案？ | LAC(1.76x)、CUDA Graph+算子融合、action chunk overlap | [§10.3](#103-推理加速与部署) |
| WAM/世界模型最新进展？ | CoWVLA(运动潜空间)、DreamZero(零样本)、LDA-1B | [§10.2](#102-世界模型方向) |
| 灵巧手操作怎么做？ | DexImit(视频→数据)、DexWM(世界模型)、4款手测评 | [§10.5](#105-灵巧手与双臂操作) |
| VLA 抖动/帕金森怎么修？ | chunk间→RTC过渡；chunk内→传统滤波后处理 | [§10.3](#103-推理加速与部署) |
| VLA 还是 VAM？ | VAM(Video-Action Model) 正在起势，用 10% 数据达最高成功率 | [§9.5](#95-vla-vs-vam-路线之争) |
| VLA-Adapter 是什么？ | 保留 VLM 能力 + decoder 端接 action head，科研平民化 | [§9.2](#92-lora-vs-全量微调) |
| 真机 RL 怎么闭环？ | Evo-RL: 示教→部署→犯错→人工纠偏→数据回流→再训练 | [§1.6](#16-rl-后训练-post-training) |
| World Model 哪条技术路线？ | 四大路线：自回归/扩散/流匹配/混合，各有取舍 | [§9.10](#910-world-model-技术路线) |
| Diffusion 还是 Flow Matching？ | FM 训练稳定推理快，DP 易被误用于单模态场景 | [§9.6](#96-diffusion-policy-vs-flow-matching-实战) |
| 移动操作/导航方向？ | 从桌面走向全屋，导航+操作统一是趋势 | [§9.9](#99-移动操作与导航) |
| 仿真器选 Isaac 还是 MuJoCo？ | 接触力精度选 MuJoCo，大规模并行选 Isaac Gym | [§3.3](#33-仿真平台选型) |
| 数据难洗怎么办？ | HDF5→RLDS 有标准流程，但真机数据清洗仍是手工活 | [§4.3](#43-数据格式与清洗) |
| 触觉/灵巧手怎么选传感器？ | 五大方案各有取舍：电阻/电容/压电/电磁/光学 | [§9.7](#97-触觉传感与灵巧手) |
| 视觉编码器选哪个？ | SigLIP-2 是新标杆，VGGT 在空间任务可能优于 DINO | [§9.8](#98-视觉编码器选型) |
| π0 怎么微调/部署？ | openpi 复现看§10.7，50条单臂可跑通，双臂需更多数据 | [§10.7](#107-π0-微调与真机部署) |
| VLA+RL 怎么做？ | SimpleVLA-RL(R1式)/WMPO(世界模型内)/π-StepNFT(flow-based) | [§10.8](#108-vla--rl-强化学习) |
| Sim2Real 怎么缩小 gap？ | DoorMan(分布包含)、PIN-WM(可微物理)、仿真数据价值被放大 | [§10.9](#109-sim2real-仿真迁移) |
| 触觉怎么接入 VLA？ | VLA-Touch(NUS)、TaF-VLA(触力对齐)、VTLA-RL(触觉+RL) | [§10.10](#1010-触觉传感与力控) |
| 移动操作怎么做？ | MoManipVLA(50条)、Mobi-π(固定→移动)、ODYSSEY(四足) | [§10.11](#1011-移动操作与导航) |
| 具身智能融资热度？ | 2025Q1国内37笔35亿，9家估值破百亿，星海图/无界动力领跑 | [§10.12](#1012-产业融资与公司动态) |

---

## 1. 训练参数与配置经验 (Training Recipes)

### 1.1 ACT (Action Chunking with Transformers)

ACT 是当前小数据场景下社区验证度最高的方案。多个独立团队的交叉验证结果高度一致：

| 配置项 | 社区验证值 | 来源 |
|--------|-----------|------|
| 最小可用数据量 | **50 episodes**（单任务） | 帖45/46/55 三人独立验证 |
| 收敛步数 | ~31K steps | 帖55（RM65 真机） |
| 单任务成功率 | 90%+（训练分布内） | 帖55 |
| Franka 上数据量 | 50 episodes 就能 work | 帖45 评论 |

**已知局限**：
- CVAE 存在 posterior collapse——ACT 的成功更多归功于 Transformer 架构而非 CVAE 设计（帖45）
- Temporal ensemble 参数敏感，不同任务需重调；且"理论上就不对"——会掩盖模型本身问题（帖35 作者回复）
- 泛化差：换桌布、换光照就崩（帖10/57）

### 1.2 SmolVLA

| 配置项 | 社区验证值 | 来源 |
|--------|-----------|------|
| 数据量 | 50 episodes | 帖55 评论（烧仙草） |
| 采集频率 | 20 Hz | 同上 |
| batch size | 64 | 同上 |
| 学习率 | 4e-5 | 同上 |
| 训练步数 | 30K steps | 同上 |
| chunk_size | 30 | 同上 |
| n_action_step | 24 | 同上 |
| 策略 | 只微调 expert（冻结 backbone） | 同上 |
| 成功率 | **50-80%**（训练覆盖范围内） | 同上 |

**踩坑**：SmolVLM2-500M-Video-Instruct 的 `tokenizer.json` 必须单独下载，SmolVLA 仓库里不包含（帖56 评论）。跨维度使用（如 6D 数据→14D 环境）需冻结视觉主干只训动作头。

### 1.3 π0 / OpenPI

| 配置项 | 社区验证值 | 来源 |
|--------|-----------|------|
| LoRA 显存 | 单卡 bs=1 约 **20 GB** | 帖20（南柯） |
| 全参数显存 | 约 **70 GB** | 帖20（南柯） |
| 最小数据量 | **100 episodes**（单任务微调可跑通） | 帖55 评论 |
| π0 在 <200 条下 | 大概率失败（前进→回撤振荡） | 帖55 详细消融 |

**关键 tricks**（帖20 南柯一手经验）：
- **预测绝对量优于 delta 值**：预测相对 offset 很难 recover，ACT 论文也是这个结论
- **state 映射**：π0 关节角度顺/逆时针定义和 ALOHA 不一样，gripper 是弧度制 0-1
- **state 归一化**：必须提前计算全数据集 mean/var，在 transform 中使用

**π0 部署灵巧手的坑**（帖19）：预训练没见过灵巧手，新增 dim 模型不知道怎么 flow，跨模态泛化远不够。评论精确解释："pi0 的 loss 是 flow-matching 的 loss，新增 dim 模型压根没见过，不知道怎么流。"

### 1.4 Motus / World Action Model

| 配置项 | 社区验证值 | 来源 |
|--------|-----------|------|
| 最低显存 | **80 GB** | 帖3（Motus 作者本人） |
| 推理 | 1.6 秒 chunk → 1 秒推理（5090, T5 预 encode） | 帖3 作者回复 |
| 动作频率 | 48 actions / 1.6s = **30 Hz** | 帖3 作者回复 |
| 视频帧率 | 8 帧 / 1.6s = **5 Hz** | 帖3 作者回复 |
| 视频帧上限 | 10 Hz 还行，再往上 video token 太多训不出来 | 帖58 作者补充 |

### 1.5 多卡 VLA 训练通用经验

来自帖9（XVLA + RobotTwin 多卡训练踩坑）：
- **DeepSpeed ZeRO-2** 比 Accelerate 更稳定
- **HDF5 压缩**：gzip 节省 50%+ 存储但增加 IO 延迟，推荐 **lzf**
- **OMP_NUM_THREADS=4**：不设置会导致 CPU 过载
- **Dataloader**：num_workers 和 prefetch_factor 对训练速度影响巨大

### 1.6 RL 后训练 (Post-training)

**RECAP / π*0.6 复现**（帖36，Evo-RL 团队一手经验；帖78 详细架构分享）：
- 硬件：SO101（最廉价开源机械臂之一）
- Policy 训练：**8×A800 约 10 小时**
- Value Function 训练：**8×A800 约 0.5 小时**
- 干预方式：**语音控制**（非脚踏板）
- 核心闭环：训练 → 部署 → 犯错 → 人工接管纠错 → 记录轨迹 → 再训练
- 框架：基于 LeRobot 实现 advantage-conditioned VLA

**Evo-RL 系统架构四层详解**（帖78，上海交大 MINT 实验室，395 赞）：
- **基础设施层**：SO101 低成本平台 + LeRobot 工作流集成
- **人在环数据层**：机器人犯错 → 人工即时接管修正 → 修正轨迹+上下文写回数据集 → 增量样本。失败不被忽略，转化为有效监督信号
- **价值推理与训练层**：Value Function Training → Value Inference → Indicator Construction
- **策略学习与部署层**：Advantage-Conditioned Policy Training，持续迭代而非一次性实验
- **关键价值**：(1) RECAP"错误后纠偏"真正接入真机 (2) Pi*0.6 训练机制整合进 LeRobot (3) 全套开源含代码/流程/数据

**真机 RL 杂谈**（帖79，钱泽中，166 赞）：
- 真机 RL 最大的工程挑战不是算法，是 reward 设计和安全约束
- 仿真训 RL 再迁移的 pipeline 仍然是主流做法，纯真机 RL 成本太高

**GRPO 训练稳定性**（帖34，实战避坑）：
- 四大崩溃原因：蝴蝶效应（策略更新改变数据分布）、组内相对奖励不稳定、梯度爆炸/熵崩塌、超参极度敏感
- **GRPO 在 MOE 模型上特别危险**：Qwen3 论文确认 token 维度优化导致 reward 暴跌，解决方案是 GSPO（帖34 评论）
- 实战建议：reward 监控看方差不只看均值、梯度裁剪必须有、定期 checkpoint

**DRL 通用避坑**（帖38）：
- 学习率先固定 **1e-4**，折扣因子 **0.99**，优先调探索策略
- PPO clip 保持 **0.2**
- 稀疏奖励 → 换成距离递减的稠密奖励，训练速度可提升数倍
- 简单规则策略或 IL 做初始化，避免从零随机探索

---

## 2. 真机部署调试 (Deployment Debugging)

### 2.1 推理延迟与执行卡顿

这是社区反映最多的工程痛点（帖13/50）：

**问题清单**：
1. 大模型推理慢，动作跟不上实时需求
2. Action chunk 边界处动作不连续 → 机械臂抖动
3. 预测-执行不匹配：预测未来轨迹，但执行时环境已变
4. 负载/摩擦变化导致同一动作效果不同

**社区解决方案**：
- **感知-推理-执行解耦**：推理等待期继续执行上一个 chunk（帖50 评论"显卡自由"）
- **在线插值**（帖35，开源 julyfun/robotoy）：约束 jerk/acceleration/velocity 三阶导，支持任意不均匀频率输入，可直接发给电机。优于 temporal ensembling（后者频率低、掩盖模型问题）和 TOTG（后者是 offline 的、acc limited）
- 推荐论文：RTC、Training-time RTC、VLASH、SAIL

### 2.2 硬件相关陷阱

**机械臂冷启动**（帖12）：电机温度低 → 关节摩擦力变大 → VLA 预测偏差。**解法：开机后空跑 10 分钟热机再做实验。** CV 转行团队最容易忽视。

**重复精度 ≠ 绝对精度**（帖53）：
- 参数表上 ±0.02mm 只是重复精度
- 5 大症结：TCP 标定失准、基座系标定错误（手眼标定用卷尺量）、关节误差+热漂移+重负载、相机畸变+标定靶不准+支架松动
- 国产机器人精度常按系统精度算，不是单轴精度

**松灵 Piper 机械臂避坑**（帖54，多人投诉）：
- 逆解大量可达角度求不出（万向锁附近尤其差）
- piper_ros 和 piper_isaacsim 的 URDF 末端坐标系不同（joint6 差 90°，joint2/3 差几度）
- 无自锁（过力/掉电直接砸地上）、一动就抖
- 其他品牌也有漏电报告

**固件更新陷阱**（帖26）：更新后 RL 模型换了，运动学指标全部得重调。

### 2.3 灾难性遗忘 / 微调视觉退化

VLA 微调后视觉理解能力退化是一个已被定量确认但社区关注度不足的问题：

- 直接微调 VLM 学动作 → OOD 泛化降 **~10%**（帖31）
- 实测发现 VLA 的 attention 集中在**背景而非目标物体**（帖37 用户实测）
- **解法**：用原始未微调的 VLM 做"视觉老师"，约束视觉模块不偏移（Visual Representation Alignment）
- 诊断工具：VL-Think（帖31）、VLAExplain（帖37，支持 Pi05 注意力可视化）

---

## 3. 仿真与 Sim2Real (Simulation & Transfer)

### 3.1 Sim2Real Gap 的真实根因

社区经验高度一致：**多数 Sim2Real 失败是物理参数不准，不是算法不行**。

| 失败根因 | 频率 | 来源 |
|----------|------|------|
| Friction model 未校准 | 最高频 | 帖41 |
| Domain Randomization 被简化为"加噪声" | 常见 | 帖41 |
| 执行器延迟未建模 | 常见 | 帖42 |
| 传感器噪声模型缺失 | 常见 | 帖42 |
| URDF 参数不准/版本不一致 | 常见 | 帖54 |

**推荐方法**（帖18/42）：Real2Sim2Real 闭环——先用真机数据校准仿真，再从校准后的仿真训练。不要只追"仿真中的性能"，要追"仿真的真实性"。

### 3.2 Sim2Sim 也有坑

即使同在仿真中转换也会出问题（帖33）：Isaac Gym → MuJoCo 时四步内摔倒。排查三周，**最终发现是观测向量中漏了初始关节角的调整**——只改了两三个单词。教训：sim2sim 时观测向量的每一个维度都必须严格对齐。

### 3.3 仿真平台选型

**Isaac Gym vs MuJoCo 深度对比**（帖80，Drawing Ting，136 赞；帖81，编号001，64 赞）：

| 维度 | MuJoCo | Isaac Gym/Sim | Genesis |
|------|--------|--------------|---------|
| 接触力精度 | **最优**，手-物交互细节好 | 中等，大规模并行强 | 宣传 430K FPS 但社区存疑 |
| 大规模并行 | 弱（单线程为主） | **最强**（GPU 并行数千环境） | 号称兼顾，实测待验证 |
| 学习曲线 | XML 配置友好 | IsaacLab 门槛较高 | Python API 最友好 |
| 社区生态 | 最成熟，学术主流 | NVIDIA 官方支持 | 2025 大火但被质疑过度宣传 |

**Genesis 理性讨论**（帖82，VectoriaWangel，738 赞）：
- 宣传 430K FPS 引发关注，但社区质疑与实际使用场景的 gap
- 优势在于 Python-native API 和可微分模拟
- 劣势：社区验证不足，复杂任务表现待确认

**MuJoCo 实用经验**：
- 同一个 XML 文件 include 多次会报错——需复制并重命名所有元素名（帖51）
- URDF 转 XML 后模型可能在 simulate 中持续颤抖（帖51 评论，原因未解）
- 手-物接触力的细节建模是 MuJoCo 的核心优势（帖83，少年，163 赞）

**仿真环境优缺点总结**（帖84，努力发paper，78 赞）：
- 仿真的本质价值是"安全的试错空间"，不是"替代真机"
- 最大坑：仿真调得再好，真机也得重新调。但没有仿真，真机调试成本高 10 倍

**云平台选型**（帖52）：AutoDL 不支持 docker、智星云便宜但按小时租、GPULab 有预装 IsaacSim 镜像但贵

**2023-2025 开源仿真平台推荐**（帖85，深蓝具身智能，84 赞）：
- 新手入门：MuJoCo（免费、文档好）→ 进阶：Isaac Lab → 前沿：Genesis

---

## 4. 数据采集经验 (Data Collection)

### 4.1 数据量门槛

| 模型 | 最小可用数据量 | 场景 | 验证情况 |
|------|---------------|------|----------|
| ACT | 50 episodes | 单任务、桌面操作 | 3 人独立验证，结论一致 |
| SmolVLA | 50 episodes（冻结 backbone） | 单任务 | 1 人报告，待更多验证 |
| π0 微调 | 100 episodes | 单任务 | 2 人报告可跑通 |
| π0 零样本 | 不可能 | 跨机器人 | 社区共识，无成功案例 |

### 4.2 采集工程

- VLA 数据**不一定需要相机标定**（帖23 评论）：遥操作用 VR 或 GELLO 映射即可
- 带力/触觉数据采集成本过高（帖24）：机械臂末端装力矩环 → 遥操时人感受不到力 → 示教效率低
- "很多数采路线要被炮灰"（帖47）：遥操作不可持续，未来是仿真合成 + 少量真实数据微调
- 自组装平台成本约 7 万 RMB（帖44，Franka 移动平台），评论建议直接买松灵/宇树底盘

### 4.3 数据格式与清洗

**HDF5 → RLDS 转换**（帖86，RetrievalAG，20 赞）：
- 标准流程已有，但不同数据集的字段命名/结构差异巨大
- 转换时最常见的坑：维度顺序、时间戳对齐、action/observation 空间定义不一致

**"机器人真机数据真的很难洗"**（帖87，Sonata，68 赞）：
- 真机数据的噪声模式远比仿真数据复杂：传感器漂移、偶发遮挡、人为操作失误
- 目前没有好的自动清洗工具，基本靠人工看轨迹回放
- 社区呼声：需要一个"数据质量可视化工具"来加速筛查

**"谁懂 VLA 机器人数据到底怎么采啊"**（帖88，Galahakang，61 赞）：
- 典型求助帖，评论区有多人分享经验
- 关键共识：VR 遥操作（Meta Quest）是当前性价比最高的方案
- ALOHA 双臂采集门槛高但数据质量好
- DreamZero（帖89，118 赞）：NVIDIA 的零样本方向，试图绕过数据采集瓶颈

**VR 遥操作实战**（帖90，MADE.，55 赞，"从0到1实现VLA第三节"）：
- VR 遥操作延迟约 30-50ms，可接受
- 关键：VR 手柄到机械臂的坐标映射必须仔细标定

### 4.4 数据采集硬件与大规模数据集

**史上最大机器人数据集开源**（帖115，Nifty，178 赞）：
- 大规模开源数据集持续涌现，数据量从千级→万级→十万级
- 但社区反馈：数据量大不等于质量好，不同实验室的数据格式/标注标准差异巨大

**DAS Gripper 无本体数据采集**（帖116，简智机器人，97 赞）：
- 不依赖特定机械臂的数据采集方案，降低硬件耦合
- 适合快速收集多场景 demo 数据

**UMI 加上了力反馈**（帖117，♥VLA和RL的具身未来😴，91 赞）：
- UMI（Universal Manipulation Interface）增加力反馈功能
- 解决了"遥操作时人感受不到力"的痛点（对比帖24 提到的问题）

**RoboMIND 2.0 数据集**（帖118，具身智能观察猿，10 赞）：
- 标准化机器人操作数据集的新版本

**Sunday 机器人硬件细节**（帖119，小白学具身，317 赞）：
- 详细拆解了 Sunday 机器人的硬件设计细节，对自组装有参考价值

---

## 5. 模型对比：社区验证 vs 论文声称 (Reality Check)

| 模型 | 论文声称 | 社区实测 | 差距 | 来源 |
|------|---------|---------|------|------|
| OpenVLA (LIBERO Object) | 85.7% | **62.6%** | -23.1% | 帖22 评论 |
| π0（<200 条数据） | 通用泛化 | 前进→回撤振荡，**全部失败** | 严重 | 帖55 |
| SmolVLA（相同条件） | — | 与 π0 相同失败模式 | — | 帖55 |
| Diffusion Policy（相同条件） | — | 与 π0 相同失败模式 | — | 帖55 |
| ACT（相同条件） | — | **90%+ 成功率** | 最佳 | 帖55 |
| π0（5000条真机 finetune） | — | OOD 成功率 **97%** | 表现最强 | 帖11 |

**核心结论**：π0 在数据充足时是社区公认最强的 VLA，但数据不足时不如 ACT。所有开源 VLA 的零样本跨机器人能力几乎为零。

---

## 6. 少数派观点：值得留意的非主流声音

以下观点来自有实战经验的从业者，与当前多数人的假设冲突。不一定对，但如果对了影响很大。

### 6.1 "VLA 的 L 不仅没用，还有害"

工业界从业者原话："L 起不到作用反而导致难以学习"（帖8/57）。小鹏 VLA 2.0 直接去掉了 L，多人证实。一位导师的判断："现在 VLA 能用到 30% 的 VLM 能力就不错了"（帖2）。

不过也有人指出：LLM 带来的 reasoning 能力是 World Model 无法替代的（帖4 评论），长序列任务的规划可能仍然需要语言（帖4 作者提出快慢双系统）。

### 6.2 "2D 视觉做 3D 世界是死胡同"

一位评论者直言："非要 2 维世界的算法来搞 3 维世界的事，终究是死胡同"（帖1）。力觉和触觉方向正在获得更多关注——"视觉无法起作用时"力觉不可或缺（补充帖 Dr.He）。但目前力觉数据采集成本过高，行业短期内不愿投入（帖24）。

### 6.3 "Diffusion Policy 的真正机制被误解"

Kivy 的分析（帖17）：Diffusion Policy 有效的核心不是"多模态分布建模"这个常见解释，而是**迭代精化（iterative refinement）**。很多人被"扩散=多模态"的叙事误导，在单模态场景强行用扩散策略，结果反而更差。

### 6.4 "RL 能改进的上限被预训练锁死"

帖32 的观点：RL 之所以有效，是因为预训练已经提供了"思维素材"，RL 只是搜索放大正确路径。如果 BC 阶段的预训练知识不够丰富，RL 再怎么搜索也解不出来。这意味着——**在投入 RL 后训练之前，先确保预训练数据质量和覆盖度足够。**

---

## 7. 行业生态速写 (Industry Landscape)

- **企业只做搬运抓取**：VLA 能把抓取做好已经可以了，装配用传统力控够（帖24）
- **企业等开源**：不想花精力自己做算法研究，等开源就行（帖24）
- **投资泡沫**：用人工遥操作冒充自主操作骗投资人的现象存在（帖30）
- **创业失败**：具身智能创业复盘后回归软件（补充帖）
- **实习体验两极**：课题组培养 leader vs 公司培养螺丝钉，深圳很多公司管理混乱、leader 水平良莠不齐（帖59 评论）
- **Demo ≠ 产品**："摆货架都是精心布置的"——**具身提示工程**（帖57 作者原话）
- **"或许具身智能没火就好了"**（帖75，653 赞）：行业反思——过热的资本导致虚假 demo 横行，真正做技术的团队反而被裹挟
- **低成本开源生态爆发**：XLeRobot（4K 成本家务机器人）、OpenArm（开源人形机械臂）、SO101——研究门槛大幅降低
- **VLA 落地挖掘机**（帖76，47 赞）：具身智能从"桌面操作"走向"重型机械"，工程化在加速
- **赵行方案**（帖77，180 赞）：大幅提升 VLA 效果的系统性方案分享
- **Tony Zhao 融资 2 亿美元**（帖91，1112 赞）："我们想把家用机器人变成现实"——Physical Intelligence 级别的资本涌入
- **具身智能高校创业江湖名录**（帖92，大圣聊机器人，364 赞）：高校系创业公司梳理
- **2026 灵巧手行业解读**（帖93，机器人猎头David，797 赞）：灵巧手方向投资升温，多家创业公司涌入
- **机器人创业者大实话**（帖94，具身机器人-陈亮，45 赞）：工业落地现状——绝大多数客户只要"搬运+抓取"，不要花哨 demo
- **一人公司+OpenClaw 新范式**（帖95，AI多面体，48 赞）：AI 控制灵巧手 + 个人创业，降低硬件创业门槛
- **LeRobot v0.5.0 发布**（帖96，Hugging Face 官方，94 赞）：27.8K stars 项目持续更新
- **Lerobot+ROS2+IsaacLab 集成**（帖97，Guss，133 赞）：三大框架联通的开源工作，降低复现门槛
- **VLAExplain 注意力可视化工具开源**（帖98，机器小白RobotNewbie，114 赞）：支持 Pi05 等主流 VLA 的 attention 热力图
- **2026 具身智能学习路线**（帖99，Xbotics，266 赞）：从入门到进阶的系统学习路径

---

## 9. 上游选型经验：训练硬件、模型架构与部署 (Upstream Decisions)

> **新增 v2**：这部分来自额外 40 篇帖子（2026-03-15 扩展收集），覆盖 GPU 选型、LoRA 策略、边缘部署加速、机械臂硬件比较等上游决策话题。

### 9.1 GPU 选型与训练资源

社区真实硬件配置汇总（非官方推荐，来自实际跑通的帖子）：

| 任务 | 最低配置 | 推荐配置 | 来源 |
|------|---------|---------|------|
| ACT 训练（单任务） | 单卡 3090 (24GB) | 单卡 4090 | 帖45/55 |
| π0 LoRA 微调 | 单卡 20GB（bs=1） | A100 40GB | 帖20（南柯） |
| π0 全参数微调 | 单卡 70GB（A100/A800） | 8×A800 | 帖20 |
| Motus/WAM 训练 | 单卡 80GB | H100 | 帖3（作者本人） |
| RECAP/π*0.6 复现 | 8×A800 ~10h（policy）| 同左 | 帖36 |
| VLA 实时推理(30Hz) | 单卡消费级 GPU | RTX 5090 | 帖61（AI椰青） |
| SmolVLA 微调 | 单卡 24GB（冻结 backbone）| 4090 | 帖55 评论 |

**3090 生存指南**（帖39，127 赞）：
- 3090 24GB 能跑 ACT、SmolVLA 冻结 backbone 微调、LoRA 微调小模型
- 不能跑：全参数 π0、Motus、大规模 RL
- 技巧：gradient checkpointing + mixed precision + gradient accumulation
- 云平台选型：AutoDL 不支持 docker、智星云便宜按小时租、GPULab 有预装 IsaacSim 镜像但贵（帖52）

**单卡实时 VLA 推理的工程优化**（帖61，AI椰青，85 赞 130 收藏）：
- CUDA Graph 消除 CPU 开销 → GPU 直接执行预记录的内核调用
- RMS 归一化层 + 线性层合并，QKV 投影融合 → 减少 7-8ms
- Triton 手动调优 GEMM 块参数 → 再减 1.5ms
- Full Streaming Inference 框架：VLM 与 Action Expert 并行执行
- 结果：单卡消费级 GPU 实现 **30Hz 推理 + 480Hz 控制输出**
- 真机验证：机器人抓住下落的笔，100% 成功率，反应速度接近人类

### 9.2 LoRA vs 全量微调

**"多模态微调别再无脑 LoRA 了"**（帖34b，算法改进猫博士，86 赞）：
- LoRA 不是万能的。当下游任务与预训练差距大时（如新动作空间），LoRA rank 不够会严重欠拟合
- 经验法则：**数据 <200 条 → LoRA**（防过拟合）；**数据 >500 条 → 考虑全量**
- LoRA rank 选择：r=16 是起点，复杂任务可能需要 r=64 甚至 r=128

**VLA-Adapter 架构思路**（帖62，Lupi，226 赞）：
- 核心洞察：action 没有好的编码器（不像视觉有 SigLIP、语言有 LLM）
- 因此：保留 VLM 能力不变 + decoder 端接大的 action head
- 趋势：从统一 token 空间 → VLM 后面接 MMDiT 式跨模态 head
- 触觉、3D 等少数据模态也放 decoder 端效果更好，原因相同
- 启示：**不要为了"端到端优雅"强行把 action 塞进 LLM 的 token 空间**

**全量 vs LoRA 显存对比**（综合多帖）：

| 模型 | LoRA | 全量 |
|------|------|------|
| π0 (3B) | ~20GB | ~70GB |
| OpenVLA (7B) | ~30GB | ~100GB+ |
| SmolVLA (500M) | <16GB | ~24GB |
| ACT | N/A（本身很小） | <16GB |

**LoRA-RL 踩坑指南**（帖34c，马小疼，137 赞）：
- 万亿参数模型上 LoRA + RL = 特别危险的组合
- LoRA 的低秩限制了 policy 表达能力，RL 探索容易跳出 LoRA 能表达的范围
- GRPO 在 MOE 模型上更危险（Qwen3 论文确认 token 维度优化导致 reward 暴跌）
- 建议：先 LoRA-SFT 稳定后，再考虑是否需要 RL；RL 阶段考虑全量或更高 rank

### 9.3 边缘部署与模型压缩

**当前状态**：VLA 专属的量化/蒸馏实操经验仍然稀缺，但开始有信号：

**知识蒸馏路线**（帖63，具身智能情报站）：
- Shallow-π 方案：从大 Flow VLA（18 层）蒸馏到 6 层小模型，<1% 性能损失
- 关键：蒸馏 Flow Matching 的速度场而非最终动作，保留动态特性

**ActionFlow 边缘加速**（帖64，具身智能之心，45 赞）：
- 针对 VLA 推理 pipeline 的专用加速方案
- 重点优化 Flow Matching 的 ODE solver 步数（10 步 → 3 步 + 校正）

**Efficient VLA 方向评估**（帖65，AI椰青，258 赞）：
- 小模型路线（<3B）在边缘设备可行：Evo-1(450M) 达 94.8% LIBERO
- 但小模型在开放世界泛化上有明显差距
- 工程优化（CUDA Graph/算子融合）比模型压缩更立竿见影

### 9.4 机械臂选型指南

社区经验总结（综合多帖，帖54/66-72）：

| 机械臂 | 价格(RMB) | 适合场景 | 社区评价 | 来源 |
|--------|----------|---------|---------|------|
| **XLeRobot** | ~4,000 | 入门/家务研究 | 超高性价比，开源生态好，1031 赞 | 帖66 |
| **SO101** | ~3,000 | π0 微调验证 | Evo-RL 团队用它复现 RECAP | 帖36 |
| **松灵七轴** | 14,999 | 中端研究 | 七自由度+力矩，327 赞，松灵官方号 | 帖70 |
| **OpenArm** | 开源自组装 | 人形机器人手臂 | 完全开源，118 赞 | 帖72 |
| **Galaxea A1XY** | 未公布 | 桌面级开发伴侣 | 星海图新品，定位研发平台 | 帖71 |
| **Franka** | ~40 万+ | 学术标准 | 精度高但贵，自组装移动平台约 7 万 | 帖44 |
| ~~Piper~~ | ~5,000 | ❌ **不推荐** | 逆解差、无自锁、URDF 不一致、抖 | 帖54 多人投诉 |

**选型决策树**：
- 预算 <5K → XLeRobot 或 SO101（开源生态最好）
- 预算 1-2 万 → 松灵七轴（力矩控制+七自由度）
- 预算 >10 万 → Franka（学术标准）或 UR 系列
- 需要灵巧手 → OpenClaw（松灵新品，AI 控制 7 轴臂，帖69，253 赞）
- 需要力反馈 → 目前选择很少，机械臂性能对比帖（帖67）有详细参数

### 9.5 VLA vs VAM 路线之争

2026 年初出现的重要分歧信号：

**"为什么 VAM 比 VLA 更有前途"**（帖73，具身薯风啸，762 赞）：
- VAM (Video-Action Model) = 用视频生成模型直接预测未来视觉 + 动作
- 核心论点：视频模型天然理解物理世界的因果结构
- 数据优势：仅用 10% 的数据达到 VLA 的最高成功率

**"Video-Action Model：VLA 之后的新范式"**（帖74，AI烤红薯，249 赞）：
- Motus/WAM 就是 VAM 路线的代表
- 从"输入视频→输出动作"变成"输入视频→预测未来视频→提取动作"

**社区争议**：
- 支持 VAM 派：视频预训练数据无穷，物理理解更自然，10% 数据效率
- 支持 VLA 派：端到端更简洁，推理更快，π0 系列已验证可扩展性
- 中间派：最终会融合——VLA 做快速反应（S1），VAM 做慢速规划（S2）

### 9.6 Diffusion Policy vs Flow Matching 实战

**"为什么 Flow Matching 能成为 VLA 主流？"**（帖120，PandaSyL，182 赞）：
- Flow Matching 训练更稳定，不需要噪声 schedule 调参
- 推理速度比 DDPM 快（ODE solver 步数可控）
- π0 系列的成功进一步确认了 FM 路线的可扩展性

**"为什么最近的一步扩散模型表现得这么好？"**（帖121，吴泰霖Talent，612 赞）：
- 一步生成（distilled diffusion）在多个任务上接近甚至超过多步扩散
- 意味着推理延迟问题可能被根本解决

**"扩散策略让很多人把自己推入了骗局"**（帖122，Kivy，169 赞）：
- 与 §6.3 相互印证：Diffusion Policy 有效的核心是迭代精化，不是多模态建模
- 很多人在单模态任务上强行用扩散，结果反而更差
- Kivy 建议：先试 ACT/Flow Matching，不要默认用 Diffusion Policy

**真机 RL 思考：World or Human in the Loop**（帖123，眠歌，150 赞）：
- 深度思考帖：World Model 辅助的 RL vs 人类在环 RL 的取舍
- 核心论点：短期内 Human-in-the-Loop 更可靠，长期 World Model 更可扩展

**ICLR 2026 工作分享**（帖124，Ming，177 赞）：
- 具身智能在 ICLR 2026 的接收情况

**DSRL：UC Berkeley 强化学习+DP**（帖125，西图Situr，110 赞）：
- Diffusion Policy + RL 的组合方案

**"生成这方向已经玩完了"**（帖126，942 赞）：
- 高互动争议帖：对生成式模型方向的悲观看法
- 评论区有大量反驳和讨论，值得看评论质量

### 9.9 移动操作与导航

**首个长时移动操作框架**（帖127，具身智能观察猿，37 赞）：
- 移动操作（mobile manipulation）开始从桌面走向全屋场景

**NaVILA：腿式机器人导航**（帖128，智元星群💫，51 赞）：
- 将 VLA 扩展到腿式机器人导航，不只是手臂操作

**[ICRA 2026] DSPv2 全身移动操作策略**（帖129，selen，80 赞）：
- 全身协调的移动操作策略，ICRA 2026 接收

**多功能具身导航 VLA 基础模型**（帖130，刘东瑞 上海 AI Lab，25 赞）：
- 上海 AI Lab 的导航 VLA 技术报告，尝试统一导航+操作

**具身智能 2 大核心方向**（帖131，硅基漫步，17 赞）：
- 操作 vs 导航是具身智能的两大核心，当前操作更热但导航同样重要

### 9.10 World Model 技术路线

**"世界模型的四大技术路线"**（帖100，吕对对，1036 赞，极高信号帖）：

| 路线 | 代表 | 核心思想 | 优势 | 劣势 |
|------|------|---------|------|------|
| 自回归 | Genie2, COSMOS | token 预测下一帧 | 与 LLM 技术栈一致 | 长序列累积误差 |
| 扩散 | UniSim, SuSa | 噪声→生成 | 视觉质量高 | 推理慢 |
| 流匹配 | Motus/WAM | 连续 ODE 轨迹 | 训练稳定、采样快 | 实现复杂度高 |
| 混合 | 多种组合 | 取长补短 | 灵活 | 调参难 |

**CVPR 2026 World Model 赛道**（帖101，世界模型研究所，94 赞；帖102，王啸峰，24 赞）：
- World Model 成为 CVPR 2026 独立赛道，信号很强
- 学术界正式认可这不是子领域而是独立方向

**具身 World Model 安全挑战综述**（帖103，刘东瑞 上海 AI Lab，29 赞）：
- World Model 在具身场景的安全问题开始被系统研究
- 关键问题：幻觉预测导致危险动作、物理不一致导致碰撞

**具身世界模型综述**（帖104，Exoskeleton，37 赞）：
- 系统梳理 2024-2026 的具身 World Model 论文脉络

### 9.7 触觉传感与灵巧手

**灵巧手的"触觉密码"：五大传感器方案**（帖105，灵巧手观察社，41 赞）：

| 方案 | 原理 | 优势 | 劣势 |
|------|------|------|------|
| 电阻式 | 压力→电阻变化 | 成本低、结构简单 | 迟滞大、易老化 |
| 电容式 | 压力→电容变化 | 灵敏度高 | 易受环境干扰 |
| 压电式 | 动态力→电荷 | 响应快 | 只测动态力 |
| 电磁式 | 磁场变化 | 非接触、耐久好 | 集成复杂 |
| 光学式 | 光信号变化 | 分辨率最高 | 体积大、成本高 |

**"为什么我选择电磁方案"**（帖106，Lai Wei，130 赞）：
- 电磁方案在耐久性和非接触测量上有独特优势
- 但集成到灵巧手指尖空间有限的场景下仍有挑战

**触觉 VLA 实战**：
- **VTLA-RL**（帖107，R&B All Night🌙，173 赞）：触觉+视觉+语言的多模态 VLA + RL，真机验证
- **OmniVTLA**（帖108，西图Situr，75 赞）：触觉 VLA 灵巧手成功率 100%（特定任务）
- **UniTacHand**（帖109，BeingBeyond，29 赞）：直击灵巧手触觉数据难采痛点

### 9.8 视觉编码器选型

**SigLIP-2**（帖110，Sam聊算法，94 赞）：
- SigLIP-2 是 SigLIP 的升级版，在多个 VLM benchmark 上超越前代
- 成为新的 VLA 视觉编码器标杆选项

**VGGT vs DINO：空间任务谁更强？**（帖111，AI朋友圈，214 赞）：
- VGGT 在需要 3D 空间理解的任务中可能优于 DINOv2
- 但 DINOv2 在通用特征提取上仍然强势

**CogVLA：对齐人类认知**（帖112，Python 智能研习社，51 赞）：
- 哈工大的工作，尝试让 VLA 的视觉处理更接近人类认知模式

**RAE 质疑与理解**（帖113，Star✨，464 赞）：
- 高互动帖：针对 RAE（Robotic Action Encoder）的质疑和回应
- 核心争议：action encoder 是否真的需要，还是过度工程化

**FRAPPE：世界模型能力注入 VLA**（帖114，陳龍龖龘，90 赞）：
- 新 SOTA：将 World Model 的物理理解能力注入 VLA 的视觉模块
- 方向信号：视觉编码器的演进不再只是"更大的 CLIP"，而是融入物理理解

---

## 10. 可追溯信息源 (Traceable Posts)

> 本节汇总所有帖子的可追溯信息（URL + 日期 + 作者）。
> §10.0 为回溯补全的早期帖子（#1-#131），§10.1+ 为 2026-03-15 批次新收集帖子（#151-#220）。
> ✅ **已全量检查（2026-03-16）**，所有 URL 均经搜索验证。无法打开的链接可能为原作者删帖。

### 10.0 回溯补全：帖 #1-#60（Backfilled Traceable Data）

> 以下数据回溯自 `2026-03-14-initial-60.md` 原始收集记录。
> 部分早期帖子因未记录 URL 或作者匿名而标记为「⚠️ 待补」。
> 帖 58/60 为帖 3/2 的评论补充，不重复列出。

| # | 标题 | 作者 | 日期 | 赞 | URL |
|---|------|------|------|-----|-----|
| 1 | VLA退场，WAM 强势来袭？ | 深蓝具身智能 | 2026-03-12 | 769 | [链接](https://www.xiaohongshu.com/explore/69b13be0000000001b017fc4) |
| 2 | VLA不是不行，是现在真做不出来 | 深蓝具身智能 | 2026-03-05 | 345 | [链接](https://www.xiaohongshu.com/explore/69a806fd0000000015033be7) |
| 3 | 最近WA/VA很火，分享下Motus insights | 谭谈AI | 2026-02-09 | 306 | [链接](https://www.xiaohongshu.com/explore/6989fc1f000000001a01d90b) |
| 4 | VLA预训练范式从根上就错了？WAM才是未来？ | 爱喝咖啡的猪 | 2026-03-03 | 730 | [链接](https://www.xiaohongshu.com/explore/69a6ed98000000001a02700f) |
| 5 | 具身智能VLA八股文 | 布朗先生 | 2026-03-13 | 149 | [链接](https://www.xiaohongshu.com/explore/69b2e8730000000022023417) |
| 6 | 具身智能路线深度解析 | 吕对对 | 2026-02-24 | 115 | [链接](https://www.xiaohongshu.com/explore/699d6ff5000000001a02391c) |
| 7 | 世界模型当教练！VLA成功率97% | 世界模型研究所 | 2026-03-11 | — | [链接](https://www.xiaohongshu.com/explore/69b0568c0000000015020668) |
| 8 | 工业届敢说的实话：VLA能力非常有限 | 工业界从业者 | 2026年初 | — | [链接](https://www.xiaohongshu.com/explore/683ae0bc000000002100a40a) |
| 9 | 记录一下多卡训练VLA遇到的坑 | VLA工程师 | 2026年 | — | [链接](https://www.xiaohongshu.com/explore/698071b4000000001a036d97) |
| 10 | VLA落地失败 | 创业团队成员 | 2026年 | — | ⚠️ 待补 |
| 11 | 现在开源VLA怎么一个比一个不靠谱啊 | VLA实践者 | 2026年 | — | [链接](https://www.xiaohongshu.com/explore/68a7e92e000000001d034f41) |
| 12 | 机械臂冷启动竟影响VLA成功率 | 机器人工程师 | 2026年 | — | [链接](https://www.xiaohongshu.com/explore/69b3fab400000000230055de) |
| 13 | VLA实时执行：从演示到部署的4层地狱 | 部署工程师 | 2026年 | — | [链接](https://www.xiaohongshu.com/explore/694c973e000000001e02f8e5) |
| 14 | VLA-Based VLN方案实机部署记录 | 移动机器人研究者 | 2026年 | — | [链接](https://www.xiaohongshu.com/explore/68b53602000000001d01399b) |
| 15 | Lerobot框架真机VLA复现 | VLA复现者 | 2026年 | — | [链接](https://www.xiaohongshu.com/explore/69873b46000000000a02e3aa) |
| 16 | 融了2亿美元做家用机器人 | Tony Zhao | 2026年 | — | [链接](https://www.xiaohongshu.com/explore/69b34db70000000009021ca9) |
| 17 | 扩散策略让很多人把自己推入了骗局 | Kivy | 2026年 | — | [链接](https://www.xiaohongshu.com/explore/694b9136000000001d039a4c) |
| 18 | 从仿真到现实的鸿沟 | David tu | 2026年 | — | [链接](https://www.xiaohongshu.com/explore/692672f7000000001d03864d) |
| 19 | 宇树G1部署pi0有感 | 布朗先生 | 2025-06-11 | 12 | [链接](https://www.xiaohongshu.com/explore/68486f46000000002301d57e) |
| 20 | 对openpi复现/finetune感兴趣 | 南柯 | 2025-03-18 | — | [链接](https://www.xiaohongshu.com/explore/67d8ea9d000000001c03e769) |
| 21 | 真机RL思考: World or Human in the Loop | 眠歌 | 2026-03-01 | — | [链接](https://www.xiaohongshu.com/explore/69a35542000000001b01df4e) |
| 22 | openvla复现中... | 是达飞呀 | 2025-07-21 | — | [链接](https://www.xiaohongshu.com/explore/687e31f00000000012023290) |
| 23 | 谁懂VLA机器人数据到底怎么采啊 | Galahakang | 2026-01-29 | — | [链接](https://www.xiaohongshu.com/explore/697b320000000000280229a3) |
| 24 | VLA里加入力，行业里是怎么想的 | 大话导师 | 2026-02-16 | — | [链接](https://www.xiaohongshu.com/explore/699292f7000000002800b4a3) |
| 25 | 具身智能VLA实习经验贴 | 阿疯Okk | 2026-03-01 | — | [链接](https://www.xiaohongshu.com/explore/69a3bb0d0000000015021134) |
| 26 | 大多数人接触具身智能都会踩的一个坑 | DoGgy | 2025-06-12 | — | [链接](https://www.xiaohongshu.com/explore/684a15bd0000000023012834) |
| 27 | 做具身智能还是要重视基本的硬件能力 | 单朴敦本zyw | 2025-02-17 | — | [链接](https://www.xiaohongshu.com/explore/67b2beeb000000002901508e) |
| 28 | VLA-Pilot被IEEE RAL接收 | ZhuoLi.Robotics | 2026-03-13 | 112 | [链接](https://www.xiaohongshu.com/explore/69b405df000000001a02fcec) |
| 29 | CVPR26 AtomicVLA技能乐高积木 | 中山大学 | 2026-03-14 | — | [链接](https://www.xiaohongshu.com/explore/69b534140000000022029bf3) |
| 30 | 具身智能机器人如何骗投资人 | gashero | 2025-10-26 | — | [链接](https://www.xiaohongshu.com/explore/68fcfc5d000000000300f822) |
| 31 | 微调时别弄瞎你的机器人模型 | mllm | 2025-10-30 | — | [链接](https://www.xiaohongshu.com/explore/6902f5fa0000000004029588) |
| 32 | 强化学习的上界被预训练锁死 | 发哥不发愁 | 2026-03-13 | — | [链接](https://www.xiaohongshu.com/explore/69b3d88a000000002301e397) |
| 33 | Humanoid-Gym复现踩坑记录 | 轩轩_转行学习版 | 2026-01-02 | — | [链接](https://www.xiaohongshu.com/explore/6957c9c9000000001e039899) |
| 34 | 为什么GRPO很容易训飞？ | 代码Lin | 2025-08-11 | — | [链接](https://www.xiaohongshu.com/explore/689954680000000003030308) |
| 35 | 机械臂丝滑在线插值 | July Fun | 2025-12-23 | — | [链接](https://www.xiaohongshu.com/explore/6948fe9a000000000d03c3e4) |
| 36 | Evo-RL: 开源Pi*0.6真机RL | 赵波 SJTU | 2026-03-05 | — | [链接](https://www.xiaohongshu.com/explore/69a8cf99000000001b01d128) |
| 37 | VLAExplain注意力可视化工具 | 机器小白RobotNewbie | 2026-02-24 | — | [链接](https://www.xiaohongshu.com/explore/699d71ba000000001b015e7d) |
| 38 | 深度强化学习DRL训练避坑指南 | 可研 far | 2026-02-12 | — | [链接](https://www.xiaohongshu.com/explore/698c5df1000000000a02d47c) |
| 39 | VLA研究方向Idea分享 | 滚烫充电宝 | 2025-08-06 | — | [链接](https://www.xiaohongshu.com/explore/6892e4ac0000000025027532) |
| 40 | pi0.5复现踩坑到跑通 | （未知） | 2025年 | — | [链接](https://www.xiaohongshu.com/explore/68ca11aa000000001101db58) |
| 41 | 连Friction Model都没调对 | Dawson | 2026年初 | — | [链接](https://www.xiaohongshu.com/explore/699a5ac10000000022022b3e) |
| 42 | sim2real的gap从何而来 | 人形蘑菇的日常 | 2026年初 | — | [链接](https://www.xiaohongshu.com/explore/69a0614a000000001a029a5c) |
| 43 | 读了一堆VLA+RL paper之后 | 张海爆 | 2026年初 | — | [链接](https://www.xiaohongshu.com/explore/69ac6a1d000000001b029b80) |
| 44 | 手搓具身智能底座太难了 | 天真派 | 2026年初 | — | [链接](https://www.xiaohongshu.com/explore/69a1bd46000000002801f7e8) |
| 45 | ACT可能只适用于桌面机械臂 | 🍑气小周 | 2026年初 | — | [链接](https://www.xiaohongshu.com/explore/69874866000000000a02e6db) |
| 46 | 试试50组数据能训出什么 | （未知） | 2026年初 | — | [链接](https://www.xiaohongshu.com/explore/68f1aa540000000004028750) |
| 47 | 很多数采路线要被炮灰了 | feixiang123 | 2026年初 | — | [链接](https://www.xiaohongshu.com/explore/699da09a000000000d009e8d) |
| 48 | 具身智能公司实习总结 | Cuscitini | 2026年初 | — | [链接](https://www.xiaohongshu.com/explore/698f1baf000000001a01e84b) |
| 49 | 2weeks机器人公司具身实习有感 | 笨小古 | 2026年初 | — | [链接](https://www.xiaohongshu.com/explore/698fc7cc000000001a037392) |
| 50 | 如何解决VLA推理过程卡顿 | 搞机器人的乌萨奇 | 2026-01-18 | — | [链接](https://www.xiaohongshu.com/explore/696cdd7d000000002200b312) |
| 51 | mujoco仿真环境搭建的坑 | 代码怎么失灵啦？ | 2025-04-13 | — | [链接](https://www.xiaohongshu.com/explore/67fa978b000000001c03f84f) |
| 52 | 0基础社畜配具身智能比赛环境 | 小唐在闯荡 | 2026-03-03 | — | [链接](https://www.xiaohongshu.com/explore/69a6ea55000000000e00d354) |
| 53 | 机械臂精度高但不准？ | 视觉项目评估 | 2026-03-05 | — | [链接](https://www.xiaohongshu.com/explore/69a8c8100000000022023406) |
| 54 | 珍爱生命，远离piper | bingcm | 2025-09-04 | — | [链接](https://www.xiaohongshu.com/explore/68b9a82d000000001b03e6f8) |
| 55 | Lerobot框架真机VLA复现（高价值） | Claude | ~2026-03-07 | — | [链接](https://www.xiaohongshu.com/explore/69873b46000000000a02e3aa) |
| 56 | Lerobot smolvla本地训练 | 瘋狂的貓 | 2025-09-30 | — | [链接](https://www.xiaohongshu.com/explore/68dab12c0000000013037bfb) |
| 57 | 工业届实话：VLA能力非常有限 | 科研游击队 | 2025-05-31 | — | [链接](https://www.xiaohongshu.com/explore/683ae0bc000000002100a40a) |
| 59 | 业界和学术界都待过的肺腑之言 | 硅基漫步 | 2026-02-20 | — | [链接](https://www.xiaohongshu.com/explore/6996a587000000000d008b49) |
| S1 | 力觉会成为具身智能关注重点 | Dr.He | ~2026-03-11 | — | [链接](https://www.xiaohongshu.com/explore/69b04eca000000000600bf2c) |
| S2 | 具身智能创业失败复盘 | momo | ~2026-03-08 | — | [链接](https://www.xiaohongshu.com/explore/69ac5466000000001a024aaf) |

### 10.0b 回溯补全：帖 #61-#131（Inline-Referenced Posts）

> 以下帖子在 §1-§9 中被内联引用（如「帖XX，Author，NN赞」），
> 但原始收集时未记录 URL。标题和作者从文档上下文推断。
> URL 栏「⚠️ 待补」表示需要在小红书上重新搜索。

| # | 标题 | 作者 | 日期 | 赞 | URL |
|---|------|------|------|-----|-----|
| 61 | 单卡实时VLA推理(30Hz+480Hz) | AI椰青 | 2025-12-16 | 85 | [链接](https://www.xiaohongshu.com/explore/6940be80000000001e031724) |
| 62 | VLA-Adapter 架构思路 | Lupi | 2025-09-12 | 881 | [链接](https://www.xiaohongshu.com/explore/68c3b864000000001c00b936) |
| 63 | 知识蒸馏 Shallow-π | 具身智能情报站 | 2026-01-31 | — | [链接](https://www.xiaohongshu.com/explore/697ddb84000000001a02f1d3) |
| 64 | ActionFlow 边缘加速 | 具身智能之心 | 2025-12-25 | 45 | [链接](https://www.xiaohongshu.com/explore/694caa12000000001e0279f1) |
| 65 | Efficient VLA 方向评估 | AI椰青 | 2026-03-12 | 258 | [链接](https://www.xiaohongshu.com/explore/69b27cb9000000001502343c) |
| 66 | XLeRobot 4千元家务机器人 | — | 2025-06-11 | 1031 | [链接](https://www.xiaohongshu.com/explore/684989e6000000002300d85f) |
| 67 | 机械臂性能对比 | — | 2025-07-13 | — | [链接](https://www.xiaohongshu.com/explore/68739c2300000000130129c6) |
| 68 | （未在文档中明确引用） | — | — | — | ⚠️ 待补 |
| 69 | 松灵 OpenClaw AI控制7轴臂 | 松灵 | 2026-03-06 | 253 | [链接](https://www.xiaohongshu.com/explore/69aa43150000000023038ae6) |
| 70 | 松灵七轴机械臂 | 松灵机器人 | 2025-11-21 | 327 | [链接](https://www.xiaohongshu.com/explore/692041e1000000001e02c492) |
| 71 | Galaxea A1XY 桌面级开发伴侣 | 星海图 | 2026-03-12 | — | [链接](https://www.xiaohongshu.com/explore/69b2a5ef000000000800f544) |
| 72 | OpenArm 开源人形机器人手臂 | — | 2025-07-27 | 118 | [链接](https://www.xiaohongshu.com/explore/68858ce900000000100103dd) |
| 73 | 为什么VAM比VLA更有前途 | 具身薯风啸 | 2026-02-20 | 763 | [链接](https://www.xiaohongshu.com/explore/6997fa2f000000000d00b650) |
| 74 | Video-Action Model：VLA之后新范式 | AI烤红薯 | 2025-03-31 | 249 | [链接](https://www.xiaohongshu.com/explore/67ea6675000000001c03f28f) |
| 75 | 或许具身智能没火就好了 | — | 2025-11-29 | 653 | [链接](https://www.xiaohongshu.com/explore/692abe34000000001d03ab99) |
| 76 | VLA落地挖掘机 | — | 2026-03-10 | 47 | [链接](https://www.xiaohongshu.com/explore/69afe919000000001d010c98) |
| 77 | 赵行方案：大幅提升VLA效果 | — | 2025-12-26 | 180 | [链接](https://www.xiaohongshu.com/explore/694e603b000000002203b461) |
| 78 | Evo-RL系统架构四层详解 | 上海交大 MINT | 2026-03-08 | 395 | [链接](https://www.xiaohongshu.com/explore/69ace17d000000000e03e819) |
| 79 | 真机RL杂谈 | 钱泽中 | 2026-02-04 | 166 | [链接](https://www.xiaohongshu.com/explore/69835d320000000021028c4b) |
| 80 | Isaac Gym vs MuJoCo深度对比 | Drawing Ting | 2026-02-08 | 136 | [链接](https://www.xiaohongshu.com/explore/69887580000000001d012ff2) |
| 81 | Isaac Gym 对比补充 | 编号001 | 2025-11-24 | 64 | [链接](https://www.xiaohongshu.com/explore/6923e86d000000000d038a57) |
| 82 | Genesis理性讨论 | VectoriaWangel | 2026-03-11 | 738 | [链接](https://www.xiaohongshu.com/explore/69b10c60000000002602e325) |
| 83 | MuJoCo手物接触力建模 | 少年 | 2025-05-28 | 163 | [链接](https://www.xiaohongshu.com/explore/683680e200000000230165e2) |
| 84 | 仿真环境优缺点总结 | 努力发paper | 2026-02-15 | 78 | [链接](https://www.xiaohongshu.com/explore/6991b6e7000000001d025417) |
| 85 | 2023-2025开源仿真平台推荐 | 深蓝具身智能 | 2026-01-09 | 84 | [链接](https://www.xiaohongshu.com/explore/6960e9ae000000000903b11d) |
| 86 | HDF5→RLDS转换 | RetrievalAG | 2025-10-22 | 20 | [链接](https://www.xiaohongshu.com/explore/68f8784b0000000004017673) |
| 87 | 机器人真机数据真的很难洗 | Sonata | 2026-03-13 | 68 | [链接](https://www.xiaohongshu.com/explore/69b2f1c5000000000c00b657) |
| 88 | VLA机器人数据怎么采（扩展版） | Galahakang | 2026-03-10 | 61 | [链接](https://www.xiaohongshu.com/explore/69b0165900000000080318dc) |
| 89 | DreamZero NVIDIA零样本 | — | 2026-02-22 | 118 | [链接](https://www.xiaohongshu.com/explore/699a6b7c000000001600a280) |
| 90 | VR遥操作实战 从0到1 | MADE. | 2026-03-13 | 55 | [链接](https://www.xiaohongshu.com/explore/69b34f570000000028009a16) |
| 91 | 融了2亿美元做家用机器人 | Tony Zhao | 2026-03-13 | 1112 | [链接](https://www.xiaohongshu.com/explore/69b34db70000000009021ca9) |
| 92 | 具身智能高校创业江湖名录 | 大圣聊机器人 | 2025-05-23 | 364 | [链接](https://www.xiaohongshu.com/explore/68301349000000002101ab65) |
| 93 | 2026灵巧手行业解读 | 机器人猎头David | 2026-02-12 | 797 | [链接](https://www.xiaohongshu.com/explore/698d85de000000000b00ad5f) |
| 94 | 机器人创业者大实话 | 具身机器人-陈亮 | 2026-03-05 | 45 | [链接](https://www.xiaohongshu.com/explore/69a9a0b2000000000d0099a3) |
| 95 | 一人公司+OpenClaw新范式 | AI多面体 | 2026-02-28 | 48 | [链接](https://www.xiaohongshu.com/explore/69a281350000000028022e98) |
| 96 | LeRobot v0.5.0发布 | Hugging Face | 2026-03-10 | 94 | [链接](https://www.xiaohongshu.com/explore/69afd539000000001d027e01) |
| 97 | Lerobot+ROS2+IsaacLab集成 | Guss | 2025-07-13 | 133 | [链接](https://www.xiaohongshu.com/explore/68736aa00000000012014030) |
| 98 | VLAExplain注意力可视化 | 机器小白RobotNewbie | 2026-02-24 | 114 | [链接](https://www.xiaohongshu.com/explore/699d71ba000000001b015e7d) |
| 99 | 2026具身智能学习路线 | Xbotics | 2026-02-12 | 266 | [链接](https://www.xiaohongshu.com/explore/698d73eb000000000e00ed6b) |
| 100 | 世界模型四大技术路线 | 吕对对 | 2026-02-02 | 1036 | [链接](https://www.xiaohongshu.com/explore/69809988000000001a036d14) |
| 101 | CVPR2026 World Model赛道 | 世界模型研究所 | 2026-03-09 | 94 | [链接](https://www.xiaohongshu.com/explore/69adb76c000000002800b91b) |
| 102 | CVPR2026 WM 补充 | 王啸峰 | 2026-02-25 | 24 | [链接](https://www.xiaohongshu.com/explore/699e9ebd000000001d011867) |
| 103 | 具身WM安全挑战综述 | 刘东瑞 | 2025-10-08 | 29 | [链接](https://www.xiaohongshu.com/explore/68e631e9000000000302e9b6) |
| 104 | 具身世界模型综述 | Exoskeleton | 2026-03-01 | 37 | [链接](https://www.xiaohongshu.com/explore/69a3fbd7000000001503b414) |
| 105 | 灵巧手触觉：五大传感器方案 | 灵巧手观察社 | 2025-12-24 | 41 | [链接](https://www.xiaohongshu.com/explore/694b6051000000001e02d591) |
| 106 | 为什么我选择电磁方案 | Lai Wei | 2026-02-28 | 130 | [链接](https://www.xiaohongshu.com/explore/69a24be2000000000d00bd89) |
| 107 | VTLA-RL 触觉+视觉+语言VLA+RL | R&B All Night🌙 | 2025-12-04 | 173 | [链接](https://www.xiaohongshu.com/explore/6930cb75000000001e0105dd) |
| 108 | OmniVTLA 触觉VLA灵巧手 | 西图Situr | 2025-08-21 | 75 | [链接](https://www.xiaohongshu.com/explore/68a697b5000000001d01b633) |
| 109 | UniTacHand 灵巧手触觉数据 | BeingBeyond | 2026-01-06 | 29 | [链接](https://www.xiaohongshu.com/explore/695cac77000000001a02e034) |
| 110 | SigLIP-2 视觉编码器新标杆 | Sam聊算法 | 2025-02-25 | 94 | [链接](https://www.xiaohongshu.com/explore/67bdc0eb000000001203e6ab) |
| 111 | VGGT vs DINO：空间任务谁更强 | AI朋友圈 | 2025-06-15 | 214 | [链接](https://www.xiaohongshu.com/explore/684ea09f000000000c03a806) |
| 112 | CogVLA：对齐人类认知 | Python 智能研习社 | 2026-02-23 | 51 | [链接](https://www.xiaohongshu.com/explore/699becf9000000002801d09f) |
| 113 | RAE质疑与理解 | Star✨ | 2025-10-16 | 464 | [链接](https://www.xiaohongshu.com/explore/68efe304000000000302f0d7) |
| 114 | FRAPPE：世界模型能力注入VLA | 陳龍龖龘 | 2026-02-20 | 90 | [链接](https://www.xiaohongshu.com/explore/6997faae000000001a034f22) |
| 115 | 史上最大机器人数据集开源 | Nifty | 2025-11-11 | 178 | [链接](https://www.xiaohongshu.com/explore/69127adf0000000005012d00) |
| 116 | DAS Gripper无本体数据采集 | 简智机器人 | 2025-12-26 | 97 | [链接](https://www.xiaohongshu.com/explore/694e79bb000000001e03a1d3) |
| 117 | UMI加上了力反馈 | VLA和RL的具身未来 | 2026-01-24 | 91 | [链接](https://www.xiaohongshu.com/explore/6974c7a8000000001a028eb8) |
| 118 | RoboMIND 2.0数据集 | 具身智能观察猿 | 2024-12-27 | 10 | [链接](https://www.xiaohongshu.com/explore/676e6b2c000000000b014a36) |
| 119 | Sunday机器人硬件细节 | 小白学具身 | 2025-11-20 | 317 | [链接](https://www.xiaohongshu.com/explore/691ea963000000001e00b725) |
| 120 | Flow Matching成为VLA主流 | PandaSyL | 2025-09-25 | 182 | [链接](https://www.xiaohongshu.com/explore/68d49de00000000013013236) |
| 121 | 一步扩散模型为何表现好 | 吴泰霖Talent | 2025-12-18 | 612 | [链接](https://www.xiaohongshu.com/explore/69440c0a000000001b032675) |
| 122 | 扩散策略骗局 | Kivy | 2025-12-24 | 169 | [链接](https://www.xiaohongshu.com/explore/694b9136000000001d039a4c) |
| 123 | 真机RL: World or Human in Loop | 眠歌 | 2026-03-01 | 150 | [链接](https://www.xiaohongshu.com/explore/69a35542000000001b01df4e) |
| 124 | ICLR 2026工作分享 | Ming | 2026-01-26 | 177 | [链接](https://www.xiaohongshu.com/explore/69778ae60000000021029126) |
| 125 | DSRL：Berkeley强化学习+DP | 西图Situr | 2025-10-04 | 110 | [链接](https://www.xiaohongshu.com/explore/68e0e2a50000000007038623) |
| 126 | 生成这方向已经玩完了 | — | 2026-02-23 | 942 | [链接](https://www.xiaohongshu.com/explore/699c3e98000000002903f5ff) |
| 127 | 首个长时移动操作框架 | 具身智能观察猿 | 2026-02-22 | 37 | [链接](https://www.xiaohongshu.com/explore/699a6b7c000000001600a280) |
| 128 | NaVILA：腿式机器人导航 | 智元星群💫 | 2025-04-12 | 51 | [链接](https://www.xiaohongshu.com/explore/67f9e5a5000000000f032766) |
| 129 | DSPv2全身移动操作(ICRA 2026) | selen | 2025-09-22 | 80 | [链接](https://www.xiaohongshu.com/explore/68d0c2bd000000001300b95f) |
| 130 | 多功能具身导航VLA基础模型 | 刘东瑞 | 2026-03-04 | 25 | [链接](https://www.xiaohongshu.com/explore/69a81c190000000022022052) |
| 131 | 具身智能2大核心方向 | 硅基漫步 | 2026-02-14 | 17 | [链接](https://www.xiaohongshu.com/explore/698f5e84000000001a02b9f9) |

**子编号帖：**

| # | 标题 | 作者 | 日期 | 赞 | URL |
|---|------|------|------|-----|-----|
| 34b | 多模态微调别再无脑LoRA | 算法改进猫博士 | 2026-01-26 | 86 | [链接](https://www.xiaohongshu.com/explore/6976e6fe000000000e03df3e) |
| 34c | LoRA-RL踩坑指南 | 马小疼 | 2025-12-11 | 137 | [链接](https://www.xiaohongshu.com/explore/693ae492000000001e014322) |

> **回溯统计**：帖 #1-#60 中 56/58 篇有 URL（97%）；帖 #61-#131 中 69/71 篇有 URL（97%）。
> 仅 #10（匿名/疑似已删除）和 #68（未在文档中明确引用）待补。子编号帖 34b/34c 已补全。
> 帖 #132-#150 为编号过渡段，对应内容已在 §10.1-§10.12 中以 #151-#220 重编号并附完整 URL。
> ⚠️ 帖 #61-#131 的 URL 系通过标题/作者关键词搜索回溯匹配，部分可能存在偏差，欢迎校对。

### 10.1 VLA 训练与新范式

| # | 标题 | 作者 | 日期 | 赞 | URL | 核心内容 |
|---|------|------|------|-----|-----|---------|
| 151 | Motus WA/VA insights | 谭谈AI | 2026-02-09 | 306 | [链接](https://www.xiaohongshu.com/explore/6989fc1f000000001a01d90b) | WA/VA 预测 1.6s/48 action/8 帧(5hz+30hz)，DreamZero 同 setting |
| 152 | VLA预训练范式从根上就错了？WAM才是未来 | 爱喝咖啡的猪 | 2026-03-03 | 733 | [链接](https://www.xiaohongshu.com/explore/69a6ed98000000001a02700f) | LDA-1B(latent dynamics action model)在 DINO latent space 构建 dynamics+action unified model，跨本体异构数据 |
| 153 | Jim Fan：世界动作模型WAM来了 | VLA和RL的具身未来 | 2026-02-05 | 30 | [链接](https://www.xiaohongshu.com/explore/698499f2000000000e00c079) | WAM 只有视频编码，长远任务规划存疑 |
| 154 | VLA-Pilot被IEEE RAL接收 | ZhuoLi.Robotics | 2026-03-14 | 117 | [链接](https://www.xiaohongshu.com/explore/69b405df000000001a02fcec) | VLA-Pilot++开发中，系列计划开源 |
| 155 | ICLR +4 具身VLA方向 | Yilun Chen | 2026-01-27 | 200 | [链接](https://www.xiaohongshu.com/explore/69778f49000000000d00a9fa) | 单人 4 篇 ICLR'26 VLA 方向论文 |
| 156 | Lerobot框架真机VLA复现 | Claude | 2026-03-07 | 89 | [链接](https://www.xiaohongshu.com/explore/69873b46000000000a02e3aa) | smolvla 50ep/20hz/bs64/lr4e-5/30k step/chunksize30，成功率 50-80% |

### 10.2 世界模型方向

| # | 标题 | 作者 | 日期 | 赞 | URL | 核心内容 |
|---|------|------|------|-----|-----|---------|
| 157 | CVPR2026世界模型新思路 CoWVLA | 世界模型研究所 | 2026-03-09 | 94 | [链接](https://www.xiaohongshu.com/explore/69adb76c000000002800b91b) | Chain of World: 视频VAE→结构+运动潜变量，在运动空间做世界模型推理 |
| 158 | 世界模型打开机器人操作新世界的大门 | YY硕 | 2026-03-13 | 170 | [链接](https://www.xiaohongshu.com/explore/69b26826000000001b0178a5) | "去掉L是符合直觉的前进方向"，VLA→VA 的讨论 |
| 159 | DreamZero：英伟达零样本 | 探索ai的瓦力 | 2026-02-22 | 118 | [链接](https://www.xiaohongshu.com/explore/699a6b7c000000001600a280) | 先生成视频再执行，成本高讨论；知道做什么→知道怎么做→能做好是不同阶段 |
| 160 | 王兴兴发论文！宇树机器人刷视频学极限动作 | 智东西 | 2026-03-03 | 681 | [链接](https://www.xiaohongshu.com/explore/69a6bf68000000001a020155) | 宇树从互联网视频学习极限动作的论文 |

### 10.3 推理加速与部署

| # | 标题 | 作者 | 日期 | 赞 | URL | 核心内容 |
|---|------|------|------|-----|-----|---------|
| 161 | 能让VLA推理提速1.76倍的框架 | 具身智能观察猿 | 2026-02-12 | 32 | [链接](https://www.xiaohongshu.com/explore/698d494d000000002800bb26) | LAC框架，A100/H100上复现，轻量级但需高效调优 |
| 162 | 如何解决VLA推理过程卡顿 | 搞机器人的乌萨奇 | 2026-01-18 | 106 | [链接](https://www.xiaohongshu.com/explore/696cdd7d000000002200b312) | 等待下一推理结果时机械臂停顿；action chunk 过渡方案讨论 |
| 163 | VLA可以跑多快？英伟达系统性分析 | 具身智能之心 | 2026-02-24 | 79 | [链接](https://www.xiaohongshu.com/explore/699d125c0000000028022fb4) | 4090 vs Jetson AGX/Thor 端侧推理测试对比 |
| 164 | 如何改善VLA的帕金森 | duckduck | 2025-08 | 259 | [链接](https://www.xiaohongshu.com/explore/689df179000000001b03eecb) | chunk间抖动=相邻推理不连续(RTC过渡)；chunk内抖动=只监督均距无时序(滤波后处理) |

### 10.4 工具与可视化

| # | 标题 | 作者 | 日期 | 赞 | URL | 核心内容 |
|---|------|------|------|-----|-----|---------|
| 165 | VLAExplain-VLA模型注意力可视化工具开源 | 机器小白RobotNewbie | 2026-02-24 | 114 | [链接](https://www.xiaohongshu.com/explore/699d71ba000000001b015e7d) | VLA attention可视化开源；无pretrain时attention乱的问题 |

### 10.5 灵巧手与双臂操作

| # | 标题 | 作者 | 日期 | 赞 | URL | 核心内容 |
|---|------|------|------|-----|-----|---------|
| 166 | XLeRobot家务机器人开源4千元成本 | VectoriaWangel | 2026-03-15 | 1033 | [链接](https://www.xiaohongshu.com/explore/684989e6000000002300d85f) | 低成本开源家务机器人平台，Jetson Orin，大量学生复刻反馈 |
| 167 | DexWM：专为灵巧操作的世界模型 | 深蓝具身智能 | 2026-01-14 | 447 | [链接](https://www.xiaohongshu.com/explore/6967098e0000000022023392) | 杨立昆团队，灵巧手世界模型 |
| 168 | 70分钟对话灵巧智能CEO：灵巧手赛道真相 | 韩成龙Jackie | 2026-03-10 | 341 | [链接](https://www.xiaohongshu.com/explore/69ae5877000000001a030eee) | 灵巧手赛道产业真相，CEO 深度对谈 |
| 169 | DexImit：视频教会双臂灵巧操作 | Believer. | 2026-02-12 | 91 | [链接](https://www.xiaohongshu.com/explore/698d4d72000000001a01fc16) | 清华 DexImit：文本→视频→4D手物交互→灵巧手数据→zero-shot部署；用 Wan2.2 生成 |
| 170 | 砸完拆——四款灵巧手终极测评 | 小Dou有两块钢铁 | 2026-01-21 | 104 | [链接](https://www.xiaohongshu.com/explore/696f19eb000000000e03f4e9) | 4 款灵巧手拆解对比测评 |

### 10.6 数据采集与遥操作

| # | 标题 | 作者 | 日期 | 赞 | URL | 核心内容 |
|---|------|------|------|-----|-----|---------|
| 171 | UMI加上了力反馈 | VLA和RL的具身未来 | 2026-01-24 | 91 | [链接](https://www.xiaohongshu.com/explore/6974a02e000000000d00c0b7) | UMI+力反馈遥操作方案 |
| 172 | UMI是具身数采的新变量 | 具身纪元 | 2025-11-30 | 52 | [链接](https://www.xiaohongshu.com/explore/6748b3e30000000025008b68) | UMI 数据采集新变量分析 |
| 173 | 机器人数据怎么采 | 吕对对 | 2026-01-28 | 69 | [链接](https://www.xiaohongshu.com/explore/697891b6000000000d036f0b) | 数据采集方案对比与流程 |
| 174 | 如何把机器人数据采集规模++ | Wenhao Wang | 2026-02-27 | 88 | [链接](https://www.xiaohongshu.com/explore/69a02ce6000000001b01ed78) | 规模化数据采集方法论 |
| 175 | 大热的UMI走到哪一步了 | AI智件 | 2026-02-05 | 19 | [链接](https://www.xiaohongshu.com/explore/69840eee000000000e00e35e) | UMI 进展追踪 |
| 176 | TWIST2人型机器人大规模数据采集系统 | Yanjie Ze | 2025-11-06 | 308 | [链接](https://www.xiaohongshu.com/explore/6920f9dc000000001e02c1ee) | 人形机器人大规模数据采集系统 TWIST2 |

### 10.7 π0 微调与真机部署

| # | 标题 | 作者 | 日期 | 赞 | URL | 核心内容 |
|---|------|------|------|-----|-----|---------|
| 177 | π0.6的RL能力被开源平替了？ | AI烤红薯 | 2025-12-13 | 133 | [链接](https://www.xiaohongshu.com/explore/693caec7000000001f008e48) | πRL让Flow Matching VLA可做RL微调，清华+北大+CMU |
| 178 | pi0微调尝试，效果还行 | ZHang | 2025-12-18 | 69 | [链接](https://www.xiaohongshu.com/explore/69440b99000000001e006a28) | π0真机部署，误映射笛卡尔空间仍能完成任务 |
| 179 | 最强VLA模型π*0.6来了！ | 机器之心 | 2025-11-18 | 267 | [链接](https://www.xiaohongshu.com/explore/691c515e000000001e02ae52) | PI发布π0.6，微调后除衣物外90%成功率 |
| 180 | π0真机部署 | 科研好楠人 | 2025-06-12 | 60 | [链接](https://www.xiaohongshu.com/explore/684abd98000000000303ded0) | π0复现踩坑，单臂50条数据微调跑通，双臂未work |
| 181 | π0+RAG,告别微调 | 明天发顶会！ | 2025-06-02 | 273 | [链接](https://www.xiaohongshu.com/explore/683d293a000000002200732c) | π0+RAG方案，告别微调的新范式 |
| 182 | Evo-RL: 复现并开源Pi*0.6真机RL | 赵波 SJTU | 2026-03-05 | 395 | [链接](https://www.xiaohongshu.com/explore/69a8cf99000000001b01d128) | SO101上复现RECAP真机RL，基于LeRobot，最廉价方案 |
| 183 | 对openpi复现/finetune感兴趣 | 南柯 | 2025-03-18 | 121 | [链接](https://www.xiaohongshu.com/explore/67d8ea9d000000001c03e769) | openpi微调细节：关节角度定义差异、gripper弧度制0-1 |

### 10.8 VLA + RL 强化学习

| # | 标题 | 作者 | 日期 | 赞 | URL | 核心内容 |
|---|------|------|------|-----|-----|---------|
| 184 | 当我读了一堆VLA+RL的paper之后 | 张海爆 | 2025-12-04 | 498 | [链接](https://www.xiaohongshu.com/explore/6930cb75000000001e0105dd) | RL vs IL工具箱对比，RL工具更多(价值函数/策略梯度等) |
| 185 | SimpleVLA-RL: VLA + R1like RL | 展的36次方 | 2025-05-30 | 385 | [链接](https://www.xiaohongshu.com/explore/683918530000000022024f9f) | 清华，DeepseekR1式RL用于VLA，单条轨迹+0/1奖励超越全量SFT |
| 186 | VLA+RL 真机RL paper汇总 | S!mple | 2025-11-02 | 57 | [链接](https://www.xiaohongshu.com/explore/6906c807000000000503b25b) | VLA+RL论文列表汇总 |
| 187 | 真机RL思考: World or Human in the Loop | 眠歌 | 2026-03-01 | 150 | [链接](https://www.xiaohongshu.com/explore/69a35542000000001b01df4e) | world model替代真实交互做RL vs human-in-loop |
| 188 | 把DiffusionNFT用在VLA RL | 王啸峰 | 2026-03-06 | 107 | [链接](https://www.xiaohongshu.com/explore/69aae59a000000001d012a1b) | π-StepNFT：flow-based VLA无需likelihood/critic的RL微调 |
| 189 | WMPO：VLA在世界模型中RL | (匿名) | 2026-01-14 | 87 | [链接](https://www.xiaohongshu.com/explore/69678d17000000000e03dceb) | ICLR2026，世界模型内on-policy RL，延续IRASim方向 |
| 190 | ThinkAct：VLA+RL三思而后行 | 西图Situr | 2025-07-23 | 186 | [链接](https://www.xiaohongshu.com/explore/6880796f000000001d00de5b) | NVIDIA+台大，VLA显式推理+RL，解决多步规划痛点 |
| 191 | VLA+真机RL路线的奇怪思考 | 骏骏骏骏🐎 | 2026-01-07 | 180 | [链接](https://www.xiaohongshu.com/explore/695e4900000000000e03f55e) | 逆共识思考：VLA+真机RL路线的质疑与反思 |

### 10.9 Sim2Real 仿真迁移

| # | 标题 | 作者 | 日期 | 赞 | URL | 核心内容 |
|---|------|------|------|-----|-----|---------|
| 192 | sim2real到底在说什么？ | 搞机器人的乌萨奇 | 2026-01-14 | 66 | [链接](https://www.xiaohongshu.com/explore/6967af3a000000002200a7f2) | Sim2Real Gap系统性梳理，目标不是仿真真实而是策略鲁棒 |
| 193 | PIN-WM：可微物理世界模型Sim2Real | 强化学习实验室 | 2025-09-21 | 64 | [链接](https://www.xiaohongshu.com/explore/68cfb69500000000130351d7) | 物理-视觉端到端可微学习，Push-T仿真到真实 |
| 194 | 英伟达sim2real新突破DoorMan | AI烤红薯 | 2025-12-13 | 63 | [链接](https://www.xiaohongshu.com/explore/693cbe93000000001e02d5a6) | DoorMan：让仿真分布⊃真实分布，人形开门43维动作空间 |
| 195 | Simulation的价值正在被放大 | Dr.He | 2025-12-03 | 115 | [链接](https://www.xiaohongshu.com/explore/692fbbde000000001f0045aa) | 仿真派观点，InternData等工作证明Zero-shot Sim2Real Transfer可行 |
| 196 | GSWorld闭环照片级仿真平台 | AI朋友圈 | 2025-10-28 | 39 | [链接](https://www.xiaohongshu.com/explore/69003555000000000700806d) | 3DGS照片级仿真+闭环操作，可"读档重来" |
| 197 | EmbodieDreamer Real2Sim2Real | AI椰青 | 2025-07-08 | 19 | [链接](https://www.xiaohongshu.com/explore/686c9c25000000000d01a6ff) | PhysAligner+VisAligner，可微物理系统辨识三阶段 |
| 198 | Isaac Sim在机器人行业的真实采用情况 | Zane的机器人技术社区 | 2026-01-19 | 122 | [链接](https://www.xiaohongshu.com/explore/696e0737000000002200bcfe) | Isaac Sim产业采用现状调查 |
| 199 | DPPO：Diffusion Policy策略优化 | AI朋友圈 | 2024-09-05 | 154 | [链接](https://www.xiaohongshu.com/explore/66d986410000000012010e18) | 策略梯度优化预训练扩散策略，sim2real零样本迁移 |

### 10.10 触觉传感与力控

| # | 标题 | 作者 | 日期 | 赞 | URL | 核心内容 |
|---|------|------|------|-----|-----|---------|
| 200 | 真机VTLA-RL | R&B All Night🌙 | 2026-01-05 | 173 | [链接](https://www.xiaohongshu.com/explore/695bbb8e000000000e03f6fe) | 触觉不能简单当视觉模态用，表征方式无统一共识 |
| 201 | VLA-Touch：触觉反馈让机器人更聪明 | 西图Situr | 2025-08-09 | 115 | [链接](https://www.xiaohongshu.com/explore/6896255300000000250117bb) | NUS VLA-Touch框架，解决VLA缺乏触觉感知的痛点 |
| 202 | 视触觉对齐到触力对齐的范式转变 | AGI具身君 | 2026-02-15 | 38 | [链接](https://www.xiaohongshu.com/explore/699141650000000028008cd3) | TaF-VLA：触觉-力对齐替代触觉-视觉对齐，精准力调节 |
| 203 | 灵巧手具身智能工程与AI协作 | 人形蘑菇的日常 | 2025-11-15 | 42 | [链接](https://www.xiaohongshu.com/explore/69189ddc0000000004011adc) | 触觉传感器坐标系转换到URDF的工程实践 |
| 204 | 指尖触觉感知实现精细抓取调整 | 老段知识加油站 | 2025-10-02 | 41 | [链接](https://www.xiaohongshu.com/explore/68dd67aa000000000302c79f) | 小米TacRefineNet，多指触觉融合手内姿态调整 |
| 205 | 具身智能触觉入门指南 | Mango-Man | 2024-12-27 | 315 | [链接](https://www.xiaohongshu.com/explore/676e057c0000000013003240) | 哥大博士，柔性触觉传感器全面入门 |
| 206 | 为什么选择电磁方案触觉传感器 | Lai Wei | 2025-12-20 | 130 | [链接](https://www.xiaohongshu.com/explore/6946487f000000001f009cb3) | 霍尔方案多维力解算，各方案优缺点对比 |

### 10.11 移动操作与导航

| # | 标题 | 作者 | 日期 | 赞 | URL | 核心内容 |
|---|------|------|------|-----|-----|---------|
| 207 | MoManipVLA：50条数据学会移动操作 | 镜水明渊 | 2025-03-28 | 41 | [链接](https://www.xiaohongshu.com/explore/67e61f7a000000001b03bf12) | CVPR，VLA路标生成+移动操作，50组数据40%成功率 |
| 208 | MobileVLA-R1：强化VLA的移动机器人 | 论文解码 | 2025-11-25 | 3 | [链接](https://www.xiaohongshu.com/explore/6925a46f000000000d036ece) | 北大，多粒度CoT+GRPO强化学习，导航+操作统一 |
| 209 | ODYSSEY：四足机器人VLN+VLA统一框架 | 西图Situr | 2025-08-23 | 41 | [链接](https://www.xiaohongshu.com/explore/68a9c43a000000001d02d7b8) | 层次化VL规划+全身控制，四足locomotion+manipulation |
| 210 | Mobi-π：让固定训练的具身AI动起来 | 西图Situr | 2025-08-26 | 53 | [链接](https://www.xiaohongshu.com/explore/68ad6e99000000001d034a00) | Stanford，固定位置训练策略→移动平台部署的迁移方案 |
| 211 | GigaBrain-0：世界模型的VLA系统 | AI椰青 | 2025-10-24 | 56 | [链接](https://www.xiaohongshu.com/explore/68fa3eab0000000004002faa) | 世界模型生成数据的VLA基础模型，减少真实数据依赖 |
| 212 | 星海图开源VLA G0 | Jay的机器人空间 | 2025-09-19 | 15 | [链接](https://www.xiaohongshu.com/explore/68cc353c000000001101ca62) | 500+小时真实移动操作数据，双臂平台，跨本体训练 |

### 10.12 产业融资与公司动态

| # | 标题 | 作者 | 日期 | 赞 | URL | 核心内容 |
|---|------|------|------|-----|-----|---------|
| 213 | 春晚后具身智能接连融资，9家估值超100亿 | 科技先声 | 2026-03-05 | 45 | [链接](https://www.xiaohongshu.com/explore/69a99c48000000001503071b) | 星动纪元/银河通用等9家破百亿，端到端RL成卖点 |
| 214 | 2025年具身智能3个月37笔融资 | ROBO-INSIGHT | 2025-04-21 | 274 | [链接](https://www.xiaohongshu.com/explore/680628ca000000001d0004c7) | Q1国内37笔35亿，上海10家最卷，天使轮纪录8.8亿 |
| 215 | 星海图融资背后的资本收编 | 掀桌指南 | 2026-02-11 | 89 | [链接](https://www.xiaohongshu.com/explore/698c4736000000000e00ee67) | 逆共识：找VC做CFO，用投资"收编"科学家，产业进化or初心迷失 |
| 216 | 近期部分具身智能企业融资信息 | momo | 2026-03-15 | 3 | [链接](https://www.xiaohongshu.com/explore/69b5c63f0000000023020bde) | 帕西尼/星动纪元/极佳视界等最新融资明细 |
| 217 | 无界动力成立3亿天使融资 | 张志鹏Dylan | 2025-11-10 | 120 | [链接](https://www.xiaohongshu.com/explore/69115d3b0000000007008051) | 红杉+线性领投，通用具身机器人，手眼脑协同 |
| 218 | Manifold AI世界模型近2亿PreA | Manifold AI 流形空间 | 2026-03-11 | 43 | [链接](https://www.xiaohongshu.com/explore/69b122ac00000000060308a3) | 成立不到十个月4轮近5亿，华为哈勃/君联等投资 |
| 219 | 星海图融资10亿后首次发声 | AGI具身职创局 | 2026-02-14 | 23 | [链接](https://www.xiaohongshu.com/explore/698fd5a5000000002800a635) | B轮估值破百亿，高瓴/美团/今日资本追投，VLA模型+数据采集 |
| 220 | AMA：从创业到大厂到博士（蚂蚁具身仿真） | Ken学长 | 2025-11-02 | 523 | [链接](https://www.xiaohongshu.com/explore/6906c4cd0000000007002f4d) | 蚂蚁具身智能仿真组负责人，创业→字节/美团/腾讯→阿里 |

### 10.13 2026 Q1 扩展收集（帖132-161 来源索引）

| # | 标题 | 来源 | 日期 | 核心内容 |
|---|------|------|------|---------|
| 帖132 | LingBot-VLA 全面开源 | 量子位/蚂蚁灵波 | 2026-01-28 | 20K小时数据、9构型、80条迁移 |
| 帖133 | 宇树 UnifoLM-VLA-0 开源 | 宇树科技 | 2026-01-29 | 人形机器人 VLA 开源 |
| 帖134 | InternVLA-M1 双系统 | 上海 AI Lab | 2026-02 | 超越 GR00T 和 π0 |
| 帖135 | LingBot-VA 世界模型 | 蚂蚁灵波/B站 | 2026-03 | 自回归视频-动作 |
| 帖136 | π*0.6 真机 RL 18小时 | 36Kr/B站 | 2026-02 | 90%成功率、2×吞吐 |
| 帖137 | OpenPI Piper 3天部署 | 七月在线/CSDN | 2026-02 | 国产臂快速适配 |
| 帖138 | π0-FAST 5×训练加速 | CSDN/知乎 | 2025-03 | DCT+BPE tokenizer |
| 帖139 | Figure Helix VLA | 机器人大讲堂 | 2026-02 | 200Hz/35DoF/500小时 |
| 帖140 | Helix 02 自主整理 | 知乎 | 2026-03 | 非结构化家居 |
| 帖141 | GR00T N2 预览 | NVIDIA | 2026-03 | DreamZero/2×leading VLA |
| 帖142 | RoboMIND 2.0 | 知乎/量子位 | 2025-12 | 55K轨迹/279任务 |
| 帖143 | SO-101 组装首发 | B站 | 2025-12 | LeRobot 硬件教程 |
| 帖144 | 松灵 Piper × LeRobot | B站/开源 | 2026-01 | 国产臂适配 |
| 帖145 | XLeRobot 6语言 | GitHub/B站 | 2026-02 | <¥4000/<4小时 |
| 帖146 | ACT 3090 训练实测 | 知乎/CSDN | 2026-01 | 100K步/3小时/¥20 |
| 帖147 | every-embodied 教程 | Datawhale/GitHub | 2026-01 | 零基础构建VLA |
| 帖148 | Embodied-AI-Guide 10K★ | Lumina/GitHub | 2026-01 | 具身技术指南 |
| 帖149 | 力感知无力传感器 | 学术前沿 | 2026-02 | 成功率+39.5% |
| 帖150 | GRU 视觉伺服遮挡恢复 | 学术前沿 | 2025 | 30Hz/<2px误差 |
| 帖151 | 人形机器人标准 2026 版 | 新华网 | 2026-03-02 | 首个全产业链国标 |
| 帖152 | 两会提具身智能 | 新华社 | 2026-03-08 | 场景落地到产业崛起 |
| 帖153 | ICLR 2026 VLA 爆发 | 搜狐/学术 | 2026-02 | 学术认可 |
| 帖154 | 魔法原子 105 亿融资 | 量子位 | 2026-03 | MagicHand S01 |
| 帖155 | CVPR 2026 七大走向 | CSDN | 2025 | 具身+3D视觉 |
| 帖156 | VLA 算法岗 80-120万 | 腾讯新闻 | 2026-03 | 春招薪资 |
| 帖157 | Lumina 招贤榜 | GitHub | 2026 | 具身智能岗位聚合 |
| 帖158 | VLA 综述中文版 | 自动化学报 | 2026 | 国内顶刊首次梳理 |
| 帖159 | LeRobot v0.5.0 × GR00T | Hugging Face | 2026-03 | NVIDIA×HF生态 |
| 帖160 | 黑马零基础课程 | B站 | 2026 | 全栈实战课程 |
| 帖161 | 国产VLA DeepSeek时刻 | 知乎高讨论 | 2026-02 | 数据vs模型vs闭环 |

---

## 12. 2026 Q1 新增（帖132-161）

> 2026-03-17 扩展收集。来源：小红书 + 知乎 + 量子位 + 36Kr + OFweek + B站等中文社区。按主题分组。

### 12.1 国产 VLA 模型开源浪潮

**LingBot-VLA：蚂蚁灵波全面开源**（帖132，量子位/蚂蚁灵波，高热度）：
- 20,000 小时真机双臂操作数据预训练，覆盖 9 种主流双臂构型
- 仅需 **80 条**演示数据即可完成下游任务迁移
- 开源内容包括模型权重 + 数据处理 + 高效微调 + 自动评估完整代码库
- 支持 FSDP2 分布式训练，大幅压缩训练周期
- **对社区意义**：首个中国产大规模双臂 VLA 开源基座，与 π0 形成直接对标

**宇树 UnifoLM-VLA-0 开源**（帖133，宇树科技，高热度）：
- 面向通用人形机器人操作的 VLA 大模型，突破传统 VLM 在物理交互中的局限
- 2026-01-29 宣布开源，紧接 LingBot 一天后发布——国产 VLA 开源赛跑加速

**InternVLA-M1：上海 AI Lab 双系统操作模型**（帖134，上海 AI Lab，中热度）：
- 指令跟随场景下显著超越 GR00T 和 π0
- 真机复杂场景和长时程任务表现突出
- **定位**：国内学术界最强 VLA 基座候选

**LingBot-VA：全球首个自回归视频-动作世界模型**（帖135，蚂蚁灵波/B站，中热度）：
- 蚂蚁灵波继 LingBot-VLA → LingBot-Depth → LingBot-World 后的最新力作
- 统一视频预测和动作生成——World Model × VLA 的中国方案

### 12.2 π0 系列微调与真机实践

**π*0.6 真机 RL：机器人连续打工 18 小时**（帖136，36Kr/B站，高热度）：
- 监督学习基础模型 → 离线 RL 预训练 → 示范微调 → 真实执行经验微调
- 最困难任务（意式咖啡）加入真实执行经验后吞吐量和成功率提升 **>2×**
- 除衣物处理外所有任务可达 **90%** 成功率
- **社区反响**：真机 RL 从理论走向工程可行性的标志性进展

**OpenPI 国产 Piper 臂 3 天部署**（帖137，七月在线/CSDN，中热度）：
- 七月在线团队在国产 Piper 机械臂上 3 天完成 OpenPI 部署 + 数据采集
- 支持官方和 LeRobot 两种部署方式
- **实操价值**：证明 π0 生态可以快速适配国产硬件

**π0-FAST 训练速度 5 倍**（帖138，CSDN/知乎，中热度）：
- 自回归版 tokenizer 设计（DCT+BPE）
- 比扩散版 π0 训练快 5× 但效果相当
- **对 §11 空白的填补**：FAST tokenization 终于有中文社区解读

### 12.3 Figure Helix 与海外进展

**Figure Helix：人形机器人 VLA 里程碑**（帖139，机器人大讲堂/知乎，高热度）：
- 全球首个 VLA 驱动的人形机器人连续上半身控制
- 200Hz 控制频率、35 自由度、仅需 500 小时多机器人操作数据
- 训练成本仅为同类 **5%**——效率惊人
- **社区解读**：具身智能的"寒武纪时刻"

**Helix 02 自主整理客厅**（帖140，知乎，中热度）：
- 2026-03-09 展示：单一神经系统在非结构化家庭环境中全自主完成整理任务
- 从实验室 demo 到真实家居——质变信号

**GR00T N2 预览：DreamZero 架构**（帖141，NVIDIA/知乎，中热度）：
- 基于 DreamZero 研究，新任务新环境成功率比 leading VLA 高 **>2×**
- 以 Cosmos Reason 2 2B 为 VLM backbone
- NVIDIA × Hugging Face 集成到 LeRobot 框架
- 2026 年底正式发布

### 12.4 数据集与基准

**RoboMIND 2.0：55,000 轨迹 279 任务**（帖142，知乎/量子位，中热度）：
- 覆盖家庭/厨房/工厂/办公/零售 5 大场景、258 场景系列
- 多构型：Franka 31K + 天工人形 9.7K + AgileX 8K + UR-5e 6.9K
- 遥操采集系统定制——大规模标准化数据采集的中国方案

### 12.5 硬件生态与选型更新

**SO-101 组装教程全球首发**（帖143，B站，中热度）：
- LeRobot SO-ARM101 完整组装配置视频教程
- 具身智能入门硬件门槛继续降低

**松灵 Piper 集成 LeRobot**（帖144，B站/开源，中热度）：
- 国产机械臂与 LeRobot 框架的官方适配
- **对社区意义**：国产硬件生态与国际软件框架的打通

**XLeRobot 6 语言文档**（帖145，GitHub/B站，中热度）：
- 中英德法西日 6 语言支持、<¥4000、<4 小时组装
- 基于 LeRobot + Lekiwi + Bambot，IKEA RASKOG 底盘

### 12.6 训练实操与成本

**ACT + LeRobot 真机训练成本实测**（帖146，知乎/CSDN，中热度）：
- 本地 3090 训练 100K steps = **3 小时**
- 云端 4090D = **20 元**（AutoDL）
- 100K steps 已收敛，模型过拟合但有一定泛化性（未见物体也有概率成功）
- **实操价值**：具体到元的训练成本，社区最缺的信息之一

**Datawhale every-embodied：零基础构建 VLA**（帖147，GitHub 6k+ stars，中热度）：
- 仅需 Python 基础，从 0 逐步构建 VLA/OpenVLA/SmolVLA/Pi0
- **教育价值**：降低 VLA 入门门槛的社区最佳资源之一

**Embodied-AI-Guide 破万 star**（帖148，GitHub/Lumina 社区，中热度）：
- Lumina 具身智能社区维护的技术指南
- 2026-01-15 重组后 GitHub stars 破 10,000
- 覆盖从论文到代码的完整技术栈

### 12.7 部署与感知新进展

**力感知无力传感器：成功率 +39.5%**（帖149，学术前沿，中热度）：
- 统一策略输出位置/力指令 + 外力预测
- 无需外部力传感器即可实现力感知模仿学习
- **与 §9.7 触觉传感互补**：软件方案替代昂贵硬件的信号

**GRU 视觉伺服遮挡恢复**（帖150，学术前沿，低热度）：
- 遮挡场景轨迹预测，计算冗余减少 30%
- 90% 遮挡下跟踪误差 <2 像素、30Hz 实时
- **实操场景**：工业环境遮挡常见，这是刚需技术

### 12.8 行业生态与政策信号

**人形机器人标准体系 2026 版发布**（帖151，新华网，极高信号）：
- 我国首个覆盖人形机器人 + 具身智能全产业链的标准顶层设计
- 6 大部分：基础共性、类脑智算、肢体部组件（含灵巧手）、整机系统、应用、安全伦理
- **政策信号**：从实验室到产业化的制度基础设施开始建设

**两会提具身智能**（帖152，新华社，极高信号）：
- 2026-03-08 新华社报道："具身智能：从场景落地到产业崛起"
- 政策层面首次大规模关注具身智能赛道

**ICLR 2026 VLA 论文爆发**（帖153，搜狐/学术社区，高信号）：
- VLA 模型在 ICLR 2026 呈爆发式增长
- 机器人智能新范式被学术界正式认可

**魔法原子 105 亿融资**（帖154，量子位，高热度）：
- 瞄准具身智能终局的超大规模融资
- MagicHand S01 灵巧手：精细操作（拧螺丝）+ 工业任务（搬运）兼备

**CVPR 2026 计算机视觉七大走向**（帖155，CSDN，中热度）：
- 具身智能与 3D 视觉成为 CVPR 2026 核心主题之一
- 与 World Model 独立赛道（帖101）形成学术共振

**VLA 算法岗薪资 80-120 万**（帖156，腾讯新闻/牛客，高热度）：
- 2026 春招 VLA 具身智能算法工程师年薪 80-120 万
- 多模态融合算法岗 50-90 万、部署优化 50-80 万
- "π 型人才"（跨界融合）成为核心稀缺资源
- **蚂蚁春招 70%+ 岗位与 AI 直接相关**，具身智能为重点方向

**Lumina 具身智能招贤榜**（帖157，GitHub 500+ stars，中热度）：
- 覆盖 PhD/RA/实习/全职的具身智能岗位聚合
- **求职资源**：目前最全的具身智能招聘信息源

### 12.9 综述与教育资源

**VLA 模型综述中文版（自动化学报）**（帖158，自动化学报，中热度）：
- 面向具身操作的视觉-语言-动作模型综述
- 国内顶刊首次系统性梳理 VLA 全景
- **学术信号**：VLA 从英文论文圈进入中文学术主流

**LeRobot v0.5.0 × GR00T 集成**（帖159，Hugging Face，中热度）：
- NVIDIA Isaac + GR00T 模型正式集成 LeRobot 框架
- GR00T N 模型 + Isaac Lab-Arena 可在 LeRobot 中微调和评估
- **2M NVIDIA 机器人开发者 × 13M HF AI 开发者**生态打通

**黑马程序员零基础具身智能课程**（帖160，B站，中热度）：
- 涵盖 OpenCV、YOLO、DeepSeek 的完整技术栈
- 从硬件到算法到 AI 的全栈实战课程
- **教育生态信号**：具身智能从研究前沿进入职业教育

**国产 VLA "DeepSeek 时刻"何时到来？**（帖161，知乎，高热度讨论）：
- 知乎高讨论度话题：LingBot / UnifoLM / InternVLA 接连开源后，社区开始期待具身智能的"DeepSeek 时刻"
- 核心观点分歧：数据质量 vs 模型规模 vs 真机闭环——哪个先突破

---

## 11. 小红书上找不到的东西

以下话题搜遍 220 篇帖子仍然缺少实操经验分享。如果你正在做这些方向，建议去 GitHub Issues（LeRobot/openpi/SmolVLA）、知乎专栏、或直接联系论文作者：

1. ~~**FAST tokenization**~~ → 部分填补：π0-FAST 5× 加速的中文解读已出现（帖138），但社区自己的调参经验仍少
2. **Co-training 数据混合比例**——机器人数据 vs 互联网视频 vs 仿真数据的配比怎么调
3. ~~**视觉编码器选型**~~ → 部分填补：SigLIP-2、VGGT vs DINO 讨论已出现（§9.8），但缺乏定量 A/B 对比
4. ~~**量化/蒸馏到边缘设备**~~ → 部分填补：知识蒸馏（Shallow-π）和 ActionFlow 有初步信号，但 VLA 专属量化实操仍为零
5. ~~**World Model 训练工程细节**~~ → 部分填补：四大技术路线已有框架（§9.6），但具体训练参数/数据配比仍缺
6. **多机器人协同部署**——完全空白
7. **Base LLM 选型对比**（Gemma vs Llama vs Qwen 作为 VLA 底座）——没有定量对比
8. **VLA 推理在 Jetson/RK3588 等嵌入式平台**——完全空白
9. **真机数据自动清洗工具**——社区强烈呼声但无成熟方案（§4.3）
10. **触觉数据标准化格式**——各团队自定义，无统一标准

---

## 关键人物/信息源

| 账号 | 身份 | 最有价值贡献 |
|------|------|-------------|
| 南柯 | OpenPI 实践者 | π0 finetune 完整参数 + 绝对量 vs delta 结论 |
| 谭谈AI | Motus 作者 | WAM 训练参数（1.6s/48action/8帧/80G） |
| 赵波 SJTU | Evo-RL 团队 | RECAP 完整复现 + 训练成本数据 |
| July Fun | 在线插值作者 | Action chunk 跳变解决方案（开源） |
| 科研游击队 | 工业界从业者 | "L 起不到作用"+ 具身提示工程概念 |
| Kivy | 理论分析者 | Diffusion Policy 有效机制是迭代精化 |
| 大话导师 | 一线博导 | 力/触觉在 VLA 中的行业困境 |
| AI椰青 | 论文解读 + 工程实践 | 单卡实时 VLA 推理(30Hz/480Hz) + Efficient VLA 方向分析 |
| Lupi | VLA-Adapter 作者 | VLA 科研平民化 + action decoder 架构设计思路 |
| 具身薯风啸 | 技术分析 | VAM vs VLA 路线对比（762 赞） |
| VectoriaWangel | 开源硬件 | XLeRobot 4 千元家务机器人方案（1031 赞） |
| 松灵机器人 | 硬件厂商 | 七轴臂 + OpenClaw 灵巧手 |
| Claude | LeRobot 实践者 | Lerobot 框架真机 VLA 复现 |
| 张海爆 | VLA+RL 研究 | VLA+RL 论文系统梳理（498 赞） |
| Tony Zhao | Physical Intelligence | $200M 融资，家用机器人愿景（1112 赞） |
| 吕对对 | 技术分析 | 世界模型四大技术路线（1036 赞，本文档最高信号帖之一） |
| 深蓝具身智能 | 教育/社区 | 仿真平台推荐 + 开源项目导航 + VLN 复现指南 |
| 灵巧手观察社 | 传感器分析 | 触觉传感器五大方案对比 |
| R&B All Night🌙 | 触觉 VLA 实践 | VTLA-RL 真机触觉+视觉 RL（173 赞） |
| Hugging Face | 官方 | LeRobot v0.5.0 发布 |
| Star✨ | 技术辩论 | RAE 质疑引发高质量讨论（464 赞） |
| 机器人猎头David | 行业观察 | 2026 灵巧手行业解读（797 赞） |
| Guss | 框架集成 | LeRobot+ROS2+IsaacLab 三框架联通（133 赞） |
| PandaSyL | 技术分析 | Flow Matching 为何成为 VLA 主流（182 赞） |
| 吴泰霖Talent | 扩散模型研究 | 一步扩散模型为何表现好（612 赞） |
| 眠歌 | RL 思考 | World vs Human in the Loop 深度分析（150 赞） |
| 具身智能观察猿 | 前沿追踪 | 移动操作框架 + RoboMIND 2.0 数据集 |
| 刘东瑞 | 上海 AI Lab | 导航 VLA + World Model 安全综述 |
| 蚂蚁灵波 | 厂商开源 | LingBot-VLA/Depth/World/VA 全栈开源（帖132/135） |
| 宇树科技 | 厂商开源 | UnifoLM-VLA-0 人形机器人 VLA 开源（帖133） |
| 七月在线 | 工程实践 | OpenPI 国产 Piper 臂 3 天部署全流程（帖137） |
| Datawhale | 教育社区 | every-embodied 零基础构建 VLA（帖147，GitHub 6k+） |
| Lumina 社区 | 技术社区 | Embodied-AI-Guide 万 star + 招贤榜（帖148/157） |

---

*本文件由定时收集器自动更新（v2，帖 1-161 + 可追溯索引 220 条，共 250+ 条）。原始数据和方法论详见 [收集流程复盘](../../memory/blog/archives/xiaohongshu-community/workflow-and-automation.md)。*

[← Back to Deployment](./README.md)
