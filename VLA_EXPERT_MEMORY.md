# VLA 专家记忆 v1 | 2026-03-14

> **用途**：贴入 LLM system prompt / 上下文，令模型立即获得 VLA 前沿研究的专家级理解。
> **来源**：KW_VLA Handbook（328 篇 Markdown，70+ 论文拆解，产业分析，周报系统）。
> **维护周期**：每 8 小时随 KW_VLA 更新同步。

---

## 0. VLA 是什么（30 秒版）

**Vision-Language-Action (VLA)** = 将视觉感知、语言理解、动作生成统一在一个模型里的具身智能范式。
输入：RGB 图像 + 语言指令（可选：深度、触觉、本体感觉）。
输出：机器人可执行的动作序列（关节角/末端位姿/夹爪）。
核心承诺：像 LLM 理解文字一样理解物理世界，并直接输出动作。

---

## 1. 模型族谱与关键架构

### 1.1 演化主线

```
RT-1(2022) → RT-2(2023) → OpenVLA(2024) → π0(2024) → π0.5(2025) → π0.6(2025)
                                              ↑ Flow Matching 引入点
```

| 模型 | 机构 | 参数 | 视觉 | 动作生成 | 控制频率 | 核心突破 |
|------|------|------|------|----------|----------|----------|
| RT-1 | Google | ~35M | EfficientNet | 离散 Token(256bin) + Softmax | 3Hz | 首个大规模真机验证 |
| RT-2 | DeepMind | 55B | ViT-22B(PaLI-X) | 离散 Token + Softmax | 1-3Hz | 语义泛化涌现（"抓灭绝动物"→抓恐龙玩具）|
| OpenVLA | Stanford | 7B | SigLIP(ViT-L) + Llama2 | 离散 Token + Softmax | 5-10Hz | 全开源 SOTA，LoRA 微调生态 |
| π0 | Physical Intelligence | 3B | PaliGemma(SigLIP+Gemma) | **Flow Matching**(ODE) | 10-50Hz | 首个 VLM × Flow Matching，高频精密控制 |
| π0.5 | PI | 3B+ | 同上 | Flow + FAST Token | ~50Hz | 开放世界泛化，co-training(机器人+互联网+仿真) |
| π0.6 | PI | 5B | 同上 + Action Expert | Flow + Recap(离线RL) | ~50Hz | 自我改进闭环，2× 吞吐 2× 低失败率 |

### 1.2 其他重要模型

- **Octo** (Berkeley)：Diffusion 动作头，连续动作，推理慢但平滑
- **Galaxea G0**：双系统（VLM 规划器 + VLA 执行器）
- **WALL-OSS**：Uni-CoT + 双分支(Flow + FAST)
- **GR-00T N1** (NVIDIA)：人形机器人基础模型
- **RDT-1B / RDT2**：Scalable Diffusion Transformer，跨具身零样本
- **LingBot-VLA**：务实型 VLA，语用接地

### 1.3 架构分类

```
单模型: RT-2 / OpenVLA / π0（一个模型端到端）
双系统: Galaxea G0 / π0.6（VLM 思考 + VLA 执行）
层级式: WALL-OSS（思维链规划 + 双动作头切换）
```

---

## 2. 三大动作生成范式

### 2.1 离散 Token 化（RT-1/RT-2/OpenVLA）
- 连续动作 → 量化为 N bins(通常 256)：`Token = round((a-min)/(max-min) × (N-1))`
- 优点：统一 Transformer 架构，支持多模态
- 致命缺点：量化误差导致精密操作失败（穿针、装配）

### 2.2 Diffusion Policy（Octo/RDT）
- 从高斯噪声迭代去噪生成动作轨迹
- 优点：连续高精度，天然多模态分布
- 缺点：需 50-100 步去噪，延迟高，不适合 >50Hz 控制

### 2.3 Flow Matching（π0 系列）⬅ 当前胜出者
- 学习确定性向量场（最优传输直线路径）
- ODE solver 仅需 1-10 步推理
- 优点：极速 + 高精度 + 支持 50Hz+
- 2026 年论文量 Flow:Diffusion ≈ 2:1，竞争基本结束

### 2.4 FAST Token 化（折中方案）
- 对动作序列做 DCT（频域变换）+ BPE 合并，压缩 token 数量
- 类比 JPEG：保留高频平滑性，减少 token 爆炸(256^7 → 少量 token)
- OpenVLA 训练加速 5×；FAST+ 预训练 1M+ 轨迹实现跨具身泛化
- π0.5 同时使用 FAST(训练) + Flow(推理)

**判断**：Action Head 已收敛至 Flow Matching（置信度 79%，校准后）。

---

## 3. 训练范式

### 3.1 行为克隆 (BC) — 基线
- 监督学习：模仿专家示范 → MSE/CE/Diffusion Loss/Flow Loss
- 天花板：只能学到专家分布内的行为，分布外崩溃

### 3.2 Co-training — 数据扩展
- π0.5 路线：机器人数据 + 互联网视频 + 仿真数据联合训练
- 关键：loss masking（不同数据源用不同损失组合）
- 解决数据稀缺但引入域差异

### 3.3 RL Post-training — 突破 BC 天花板 ⬅ 当前唯一赢家
- π0.6 Recap：离线 RL 自我改进（VLM 自动打分 → 高分轨迹回训练）
- GR-RL：Mixture of Teachers 在线 RL
- GigaBrain RAMP：World Model 辅助 RL
- **2026-03 数据**：RL finetuning 加速比 1.82x（全场唯一 SURGE），Instruction Tuning 仅 0.06x（已死）
- 置信度：RL 后训练突破 BC 天花板 = 81%（校准后）

### 3.4 数据飞轮（终极形态）
```
少量遥操作 → BC 基线 → 真机探索 → VLM 自动打分 → 高分轨迹回训练 → 更强模型 → 更多探索
                              ↑ Recap / Reward Discovery 核心机制
```

### 3.5 损失函数全景

| 阶段 | 损失类型 | 公式/方法 | 用途 |
|------|----------|-----------|------|
| BC-离散 | Cross-Entropy | -Σ y·log(ŷ) | RT-1/RT-2/OpenVLA token 分类 |
| BC-连续 | MSE/Huber | \|a-â\|² | 回归动作值 |
| BC-GMM | NLL | -log Σ wᵢ·N(a;μᵢ,σᵢ) | 多模态连续动作 |
| Diffusion | ε-prediction | \|ε-ε̂(xₜ,t)\|² | 去噪扩散 |
| Flow | velocity field | \|v-v̂(xₜ,t)\|² | 速度场匹配 |
| RL | PPO clip | min(rA, clip(r)A) + V_loss + entropy | 策略改进 |
| 对齐 | InfoNCE/CLIP | 视觉↔语言/视觉↔触觉对比学习 | 跨模态表示 |
| 安全 | barrier/jerk | 速度/加速度/力矩/工作空间约束 | 部署安全 |
| 抗遗忘 | Knowledge Insulation | 梯度隔离(动作头梯度不回传VLM) | 防灾难性遗忘 |

### 3.6 关键训练技巧
- **Knowledge Insulation**：双轨训练——VLM 学离散 token(保留语义)，Action Expert 学连续控制(独立优化)，梯度不互传。<1% 性能损失，2× 收敛加速
- **Co-training loss masking**：不同数据源用不同损失组合（机器人数据全损失，互联网视频只有视觉+语言损失）
- **Action Chunking**：一次前向生成 32-64 步动作序列，配合高频重规划实现闭环
- **Reward Discovery**：双层元学习自动进化奖励函数，将稀疏"完成/失败"转化为平滑奖励地形

---

## 4. 核心信念网络（Belief Graph 精华）

> 置信度经过校准：>80% 原始值 ×0.9

| ID | 信念 | 置信度 | 最强反驳 |
|----|------|--------|----------|
| B0 | 数据策略 > 模型架构 | 77% | C1：架构创新会回归(20%) |
| B1 | 数据飞轮是核心壁垒 | 77% | 开源数据集可能瓦解私有壁垒 |
| B2 | RL 后训练突破 BC 天花板 | 81% | RL 不稳定性可能无法工程化解决 |
| B3 | 自我改进闭环是终极形态 | 77% | Reward 定义问题比想象的严重 |
| B4 | World Model 作为闭环加速器 | 60% | 物理幻觉在接触密集任务中致命 |
| B5 | Flow Matching 主导 Action Head | 79% | 新范式可能取代(但目前无信号) |
| B6 | 分层架构(S0/S1/S2)标准化 | 75% | 端到端可能重新胜出 |
| B7 | Action Expert 解耦语义与运动 | 78% | 解耦可能损失跨模态协同 |
| B8 | 触觉从可选→必需 | 60% | 硬件标准化遥遥无期 |
| B9 | 小模型(<3B)占领边缘部署 | 63% | 云端推理成本可能降到可接受 |

**逆共识（赌注）**：
- C1：架构创新会回归 (20%) — 当前"修 bug 阶段"可能只是暂时
- C2：World Model 是死胡同 (20%) — 但 PlayWorld +65% 真机成功率在反驳
- C3：VLA 不需要语言 (24%) — 纯视觉-动作路线有上升信号

---

## 5. 收敛地图（Phase Transitions）

### Phase 1: Action Head → Flow Matching 【85% 完成】
- 4 个独立信号（π0/UnifolM/LingBot/DreamZero）
- 判断：临界点已过，2026 年底成为标准

### Phase 2: 训练范式 → RL 后训练 【68% 完成】
- 5 个独立信号（π0.6 Recap/GR-RL MoT/GigaBrain RAMP 等）
- RL finetuning 加速比 1.82x，Instruction Tuning 已死(0.06x)

### Phase 3: 触觉 → 标准化 【35% 完成】
- 8 个信号但缺乏统一格式
- "寒武纪大爆发"阶段，12-18 个月才能标准化
- 加速比 0.43x — 大多数团队在"假装做触觉"

### Phase 4: World Model → 闭环实用化 【50% 完成】
- 从评估器 → 规划器 → 动作生成基底 演进
- PlayWorld：自主探索→WM→RL 闭环，+65% 真机成功率
- 关键障碍：接触密集任务的物理幻觉

### Phase 5: 跨具身泛化 【40% 完成】
- 方法碎片化严重，每个团队策略不同
- RDT2 展示零样本跨具身迁移可能性

**约束松弛分析**：#1 约束 = 真机数据采集成本（几乎不可松弛，只能绕过：WM/互联网视频/Sim2Real）

**收敛交叉检测**：
- Phase 2×4（RL in imagination）：最危险交叉——World Model 生成合成 rollout 做 RL，成功则颠覆真机数据需求
- Phase 3×2（触觉奖励 for 精细 RL）：被低估——触觉信号可作为精细操作的天然稠密奖励
- **时间套利窗口**：触觉×RL=精细操作突破 | 视觉编码器 control-aware 注入被低估

---

## 6. 触觉专题

### 6.1 为什么不可替代
- 视觉给坐标，语言给意图，**触觉给接触相位的真反馈**
- 三联仪表盘：力(抓稳没)、形(局部几何)、质(软硬粗糙)
- 视觉先天缺陷：遮挡 + 不可观测物理量(摩擦/应力) + 接触事件太快

### 6.2 技术栈四层
1. **硬件**：e-skin(电阻/电容/压电) vs 光学触觉(GelSight/DIGIT)
2. **表示**：异构信号→统一空间(UV map/手坐标系锚定)
3. **融合**：高层(触觉→语言→VLM) or 低层(FiLM/cross-attention 注入 policy)
4. **仿真**：接触动力学建模复杂，Sim2Real gap 大，是 scaling 瓶颈

### 6.3 前沿工作
- TaF-VLA：触觉力对齐
- TacMamba：快慢双通路触觉压缩
- TacRefineNet：纯触觉抓取精炼
- GenForce：触觉力迁移
- SuperTac/DOVE：仿生多模态电子皮肤
- UniVTAC：统一视触觉仿真平台

---

## 7. 部署与工程

### 7.1 边缘部署策略
- 量化：INT8/INT4（QVLA 专门做 action-centric 量化）
- 蒸馏：Shallow-π 从大 Flow VLA 蒸馏到小模型
- Thin Client：本地轻量推理 + 云端重模型（延迟 vs 成本 trade-off）
- 小模型趋势：<3B 参数占领边缘（B9 置信度 63%）

### 7.2 数据采集方案
- **遥操作**：GELLO/ALOHA（双臂镜像）、数据手套+振触反馈、VR 控制
- **互联网视频**：VITRA 从 Ego4D/Epic 等自动解析 1.2M 人手操作 episodes (26M 帧) → VLA 预训练
- **仿真生成**：RoboGene 用 agentic 方式多样化生成仿真数据
- **真机 RL**：带安全约束的在线探索（最危险但最有效）
- **合成数据引擎**：World Model 生成 → 过滤 → 训练闭环
- 核心矛盾：1 小时遥操作 = 数百元，且无法覆盖长尾场景

### 7.3 仿真环境
- **Isaac Sim/Lab** (NVIDIA)：GPU 并行物理 + RTX 渲染，大规模 RL 首选
- **MuJoCo** (DeepMind)：软接触精度高 + 速度快，精细操作仿真
- **SAPIEN/ManiSkill** (UCSD)：零件级交互，灵巧操作
- **PyBullet**：轻量入门
- **Gazebo**：ROS 集成

### 7.4 Sim-to-Real
- Domain Randomization：视觉/动力学参数随机化
- Domain Adaptation：对抗训练对齐仿真/真实分布
- System Identification：用真实数据校准仿真器参数
- 加速比 0.28x（结构性衰退）— 学术界在逃"硬件依赖"

### 7.5 评估体系
- **指标**：Success Rate (SR)、Mean Steps to Success、Intervention Rate、Executable Rate
- **基准**：CALVIN (5 步链式)、LIBERO (已饱和 99.2%)、SIMPLER (sim↔real 相关性)、ManiSkill、RoboChallenge
- **统计纪律**：Wilson 区间置信度、EMA checkpoint 选择、A/B 测试协议
- **产业 KPI**（学术不追踪但更重要）：任务成功率、吞吐量、干预率、连续运行时长、部署成本

### 7.6 RL 训练基础设施（RLinf 视角）
- 关键6点：控制频率对齐(10-30Hz vs 125-500Hz)、评估协议固定、KL-to-base 必备、奖励防欺骗、失败当一等数据、先跑通数据面再谈算法
- 最稳路径：BC warmstart → 仿真 RL 大规模改进 → 真机小步安全迭代
- 训练三层：策略学习(BC/RL loss) → 表示对齐(CLIP/InfoNCE) → 安全约束(barrier/jerk)

---

## 8. 产业格局

### 8.1 三大流派
1. **全栈整合派**（Tesla/Figure）：模型+数据+硬件+制造一步打通
2. **垂直突破派**（DYNA/Amazon）：单场景极强→再泛化
3. **生态平台派**（NVIDIA/Google/Meta）：工具链+标准化接口建生态

### 8.2 关键玩家
- **Physical Intelligence (PI)**：π0 系列，Flow Matching 先驱，Robot API 平台化
- **Figure**：Helix 02 全身自主，$2.6B 估值
- **Tesla Optimus**：全栈+数据飞轮
- **NVIDIA**：GR-00T N1 + Isaac Lab + Cosmos — 做机器人的 Android
- **1X (1X Technologies)**：World Model 路线，EVE/NEO
- **中国阵营**：智元(Agibot)/宇树(Unitree)/灵初(LimX)/银河通用(Galaxea)/智在无界(Boundless)

### 8.3 产业信号（2026-03）
- 产业融资超 50 亿美元（AI²/Apptronik/Spirit 等）
- Agility×Toyota 签产线部署协议
- 学术与产业正在分道扬镳：学术刷 LIBERO 99.2%→99.5%，产业谈量产落地
- 工具链收敛：LeRobot 成事实标准，v0.5.0 集成 X-VLA

---

## 9. 领域当前状态（截至 2026-03-14）

### 9.1 核心判断
- **执行层收敛**：Action Head(Flow Matching 胜) + 后训练(RL 胜)
- **认知层发散**：World Model 多路径探索（单一→分层+多模态）
- **领域处于"修 bug 阶段"**：174 篇论文仅 3 篇突破性(1.7%)，无架构创新
- **方法论讨论热度首次超过实验室动态** — 从"谁在做"转向"怎么做"

### 9.2 速度异常
| 方法族 | 加速比 | 趋势 |
|--------|--------|------|
| language_grounding | 1.56 (+240%) | 爆发：语言接地从"架构缺陷"→"可修复 bug" |
| rl_finetuning | 1.59 (+80%) | SURGE：工具链民主化降低实验门槛 |
| world_model | 1.12 (+19%) | 温和增长 |
| flow_matching | 0.84 | 稳定追赶 diffusion |
| sim_to_real | 0.28 (-55%) | 结构性衰退 |
| tactile | 0.37 (-19%) | 硬件瓶颈 |
| instruction_tuning | 0.06 | 已死 |

### 9.3 基准状态
- CALVIN / LIBERO：**已饱和**（开源 99.2%，闭源 98.6%）
- RoboChallenge：差异化赛道（仅 2 次 SOTA 变动）
- 产业与学术基准严重脱节：产业关心"任务成功率"非"基准分数"

### 9.4 关键预测（可追踪）
1. RL finetuning 8 周内出现"稳定性"子赛道（截止 2026-05-06）
2. LeRobot v0.6.0 将 Flow Matching 设为默认 Action Head（截止 2026-04-23）
3. 首个产线场景 VLA 基准由产业联盟发布（截止 2026-06-01）
4. Instruction Tuning 论文 8 周内跌破 1%/月（截止 2026-05-06）

---

## 10. 深度专题

### 10.1 π0 系列架构详解

**π0 (2024)**：PaliGemma 3B (SigLIP 视觉编码 + Gemma 2B 语言) + Flow Matching 动作头。学习速度场 v(x,t) 将噪声分布映射到动作分布，沿直线路径（rectified flow）。ODE solver 1-10 步推理 → 50Hz+ 控制。核心创新：首次证明大 VLM 可以高频输出精密动作。

**π0.5 (2025)**：分层推理——高层 VLM 异步语义推理 + 低层同步 50Hz 动作输出。训练用 FAST token 化（DCT+BPE 压缩），推理用 Flow Matching。Co-training：机器人 + 互联网视频 + 仿真，loss masking 分数据源。实现开放世界"做任何家务"的泛化。

**π0.6 / π*0.6 (2025)**：5B VLM + 10M 参数 Action Expert（轻量独立模块）。π0.6 = 监督学习基线；π*0.6 = Recap 算法（离线 RL 自我改进）。Recap 流程：收集 on-policy rollout → VLM 自动打分 → 筛选高分轨迹 → 重新训练。Knowledge Insulation 防止动作训练破坏语义能力。成果：2× 吞吐提升，2× 失败率下降。

### 10.2 World Model 演进路线

```
阶段 1: 评估器 (WorldEval) — 能否不用真机就评估策略优劣？
阶段 2: 标准化评估 (WorldArena/Ctrl-World) — 如何统一 WM 基准？
阶段 3: 数据引擎 (VLAW) — WM 生成合成轨迹喂给策略训练
阶段 4: 动作生成基底 (DreamZero/WAM) — WM 直接取代 action model？
```

关键进展：
- **VLAW**：on-policy rollout 微调 WM → 生成合成轨迹 → 过滤式 BC 训练，+39.2% 成功率
- **DreamZero/WAM**：World Model 即零样本策略，比较三条路线(解耦/端到端/统一多任务)
- **PlayWorld**：自主探索→WM→RL 全闭环，+65% 真机成功率
- **AtomVLA**：LLM 分解任务为原子子任务 + 预测性潜在 WM + 离线 GRPO，LIBERO 97%
- **Cosmos Predict 2.5**：NVIDIA 大规模视频 WM 基础设施
- 核心张力：好视频 ≠ 好评估器，好评估器 ≠ 好规划器；WM 从侧模块→系统工具→核心基底

### 10.3 小模型路线 (<3B)

| 模型 | 参数 | LIBERO | 核心技巧 |
|------|------|--------|----------|
| Evo-1 | 450M | 94.8% | RT-2 参数的 1.4%，证明模型大小≠控制能力 |
| SmolVLA | 500M | ~92% | 极致压缩 VLM |
| ControlVLA | 770M | ~93% | 控制专精设计 |
| Eva-VLA | 700M | ~91% | 高效视觉编码 |

启示：边缘部署不需要 7B；但小模型在开放世界泛化上仍有明显差距。

### 10.4 推理与规划

- **Chain of Thought 四种模式**：显式文本 / 结构化 JSON / 隐式潜在 / 交错逐步
- **OneTwoVLA**：单模型自适应 System 2(深度推理)/System 1(快速执行) 切换，用 [BOR]/[BOA] token
- **Thinker VLM**：UBTech 具身规划模型（不直接输出动作），4B/7B，处理 ego-view 混淆
- **ReconVLA**：通过注视区域重建辅助损失防止注意力漂移，隐式空间接地

### 10.5 跨模态迁移与数据规模化

- **VITRA**：自动从人类活动视频(Ego4D/Epic)提取 1.2M 机器人式 episodes，逐帧 3D 手部运动恢复
- **跨模态映射逻辑**：互联网视频学"语义动作规范"（开门先握把手）→ 精细力控交给底层算法/少量真机微调
- **ABot-M0 UniACT**：统一 6 个数据集(6M 轨迹, 20+ 具身)，EEF-delta + rotation-vector 标准化
- **RoboGene**：Agentic 多样化仿真数据生成，提升 VLA 预训练质量

### 10.6 分层控制架构（以 Figure Helix 02 为例）

```
S2 (语义层): VLM 输出语义 latent — 低频（~2-5Hz）
  ↓
S1 (运动层): 200Hz 全身目标生成（locomotion + manipulation）
  ↓
S0 (执行层): 1kHz 学习式先验控制（接触/平衡/稳定性）
```

Helix 02 训练数据：>1000h 人类运动 + >200k 仿真环境。无状态机，统一处理行走+操作。
此分层模式(B6 置信度 75%)正在成为人形机器人标准架构。

---

## 11. 关键论文速查（按影响力排序）

| 论文 | 核心贡献 | 影响 |
|------|----------|------|
| **里程碑** | | |
| π0 (2024) | Flow Matching + VLM = 高频精密控制 | 定义 Action Head 新范式 |
| π0.5 (2025) | 分层推理 + co-training 开放世界 | 泛化路线验证 |
| π0.6 Recap (2025) | 离线 RL 自我改进闭环 | 定义后训练新范式 |
| RT-2 (2023) | VLM → VLA 语义泛化涌现 | 证明大模型路线可行 |
| OpenVLA (2024) | 开源 7B VLA + LoRA 生态 | 民主化 VLA 研究 |
| Diffusion Policy (2023) | 去噪生成连续动作 | 建立连续动作基线 |
| **World Model** | | |
| DreamZero / WAM (2026) | World Model = 零样本策略 | WM 功能角色跃迁 |
| PlayWorld (2026) | 自主探索→WM→RL 闭环 | +65% 真机成功率 |
| VLAW (2026) | VLA × WM 迭代共进化 | on-policy WM 校准 +39.2% |
| AtomVLA (2026) | 原子子任务 + 潜在 WM + 离线 GRPO | 无需在线试错 |
| **触觉** | | |
| TaF-VLA (2026) | 触觉力对齐注入 VLA | 触觉融合新范式 |
| TacMamba (2026) | 快慢双通路触觉压缩 | 触觉反射层架构 |
| UniVTAC (2026) | 统一视触觉仿真平台 | 仿真标准化 |
| **数据与效率** | | |
| VITRA (2026) | 人类视频→1.2M 机器人 episodes | 数据规模化路线 |
| SimVLA (2026) | 0.5B 达 98.6% LIBERO | 训练 recipe > 架构复杂度 |
| FAST (2024) | DCT+BPE 动作 token 压缩 | 5× 训练加速 |
| Shallow-π (2026) | Flow VLA 知识蒸馏 18→6 层 | 边缘部署 <1% 性能损失 |
| QVLA (2026) | 动作敏感性量化 | 部署优化 |
| **语言与推理** | | |
| LangGap (2026) | 语言理解缺口四维诊断 | 语言接地修复框架 |
| ReViP (2026) | 视觉一致性验证修正错误补全 | 推理时闭环修复 |
| OneTwoVLA (2026) | 自适应 S1/S2 推理切换 | 统一快慢思维 |
| ReconVLA (2026) | 隐式空间接地(注视重建) | 防注意力漂移 |
| **其他** | | |
| Helix 02 (2026) | S2→S1→S0 分层全身自主 | 人形架构标杆 |
| ABot-M0 (2026) | UniACT 6M 轨迹统一 | 跨具身基础 |
| RDT2 (2026) | 零样本跨具身迁移 | 泛化验证 |

---

## 12. 开源基础设施与工具链

| 工具 | 类别 | 最新版本 | 定位 |
|------|------|----------|------|
| LeRobot | 训练框架 | v0.5.0 (2026-03) | 事实标准，集成 X-VLA backbone |
| Isaac Lab | 仿真+RL | - | GPU 并行训练首选 |
| MuJoCo | 物理引擎 | v3.6.0 (2026-03) | 精细接触仿真 |
| SAPIEN | 仿真 | v3.0.3 (2026-03) | 零件级交互 |
| Genesis | 仿真 | v0.4.1 (2026-03) | 新兴综合仿真 |
| GELLO/ALOHA | 数据采集 | - | 遥操作硬件方案 |

**开源分级**：展示型(算法 demo) < 生态锁定型(厂商工具) < 基础设施型(全 CAD+栈+know-how 透明)
工具链正在快速收敛，继续维护独立训练代码库的团队将面临"无人复用"困境。

---

## 13. 产品与市场

- **PMF 真标准**：持续用户留存 + 可量化 ROI + 可靠性验证（非 demo 级别）
- **人形机器人**：Figure/Tesla/1X/Agility 领跑，中国 Unitree/LimX/银河通用追赶；2026 年进入小批量产线部署但距大规模量产仍有 2-3 年
- **产业与学术脱节**：学术卷 LIBERO 99.2%→99.5%，产业谈"产线部署""量产基地"——当基准分数与客户付费标准脱钩，学术研究合法性基础正在松动

---

## 14. 高频面试要点

**Q: VLA 和传统机器人学习有什么本质区别？**
A: 传统方法是模块化流水线(感知→规划→控制)，VLA 是端到端：视觉+语言直接映射到动作。优势是涌现泛化能力，代价是可解释性和安全保障。

**Q: 为什么 Flow Matching 胜出？**
A: Diffusion 走随机路径需 50-100 步去噪，Flow 走最优传输直线仅需 1-10 步。同等精度下推理快 10 倍+，首次让大模型支持 50Hz+ 实时控制。

**Q: VLA 最大瓶颈是什么？**
A: 数据。真机数据采集成本是 #1 约束（1 小时数百元，无法覆盖长尾）。三条绕过路径：互联网视频跨模态迁移、World Model 生成合成数据、Sim2Real。

**Q: RL 后训练为什么是突破口？**
A: BC 只能学到专家分布内行为，分布外崩溃。RL 通过在线探索收集分布外数据 + 自动奖励(VLM 打分) → 突破 BC 天花板。π0.6 Recap 是典型代表。

**Q: 触觉为什么重要？**
A: 视觉给坐标，语言给意图，触觉给接触相位真反馈。遮挡下力/形/质不可视觉观测，精密操作的最后 1cm 靠触觉闭环。

**Q: World Model 当前状态？**
A: 从 nice-to-have 预测器 → 评估器 → 规划器 → 动作生成基底 演进中。PlayWorld 已证明 WM→RL 闭环可行(+65%)，但接触密集任务的物理幻觉是致命障碍。置信度 60%。

**Q: 小模型能替代大模型吗？**
A: 在受限场景可以。Evo-1 (450M) 达 LIBERO 94.8%，仅 RT-2 参数的 1.4%。但开放世界泛化仍需大模型。边缘部署可用蒸馏(Shallow-π)或量化(QVLA)。

**Q: SimVLA 的启示？**
A: 0.5B 模型通过正确训练 recipe（数据 shuffling、归一化、LR schedule）达 98.6% LIBERO。关键："沉默旋钮"(shuffling off = 9.9% vs on = 98.6%) 比花哨模块重要得多。数据策略 > 架构创新(B0)的直接证据。

**Q: Knowledge Insulation 是什么？**
A: 双轨训练防灾难性遗忘：VLM backbone 只学离散 token（保留语义能力），Action Expert 独立学连续控制，梯度隔离不互传。π0.6 核心技巧之一。

**Q: 当前领域最大风险？**
A: 学术与产业脱节。学术在 LIBERO 上刷 0.3% 提升，产业需要"产线任务成功率""维护周期"。工具链(LeRobot)收敛加速了实验民主化，但 54 篇 RL 论文中多数是调参报告而非方法创新——"工具易得≠方法成熟"。

---

## 15. 校准纪律（使用本记忆时的注意事项）

1. **谦逊折扣**：所有 >80% 置信度已乘 0.9（LLM 在此区间系统性过度自信）
2. **保守偏误修正**：强证据最小更新 ±5%，Bull+Bear 共识最小 ±10%
3. **逆共识保护**：逆共识信号的筛选阈值为正常的 1/3（防止系统性杀死异见）
4. **高确定性 = 高风险**：你最确定的判断，恰恰是最需要被挑战的
5. **本文档截止日期**：2026-03-14，VLA 领域每周都有重大变化

---

*生成自 KW_VLA Handbook v3 | 328 篇源文件 → ~20K tokens 压缩 | 下次更新：随 KW_VLA 同步*
