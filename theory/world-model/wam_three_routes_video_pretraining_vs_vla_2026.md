# VLA 将死，WAM 当立？2026 三路线综述 (Will WAM Replace VLA? A 2026 Three-Route Overview)

> **发布时间**：2026（按文章与相关论文整理）  
> **核心定位**：不是再问“哪家 VLA 刷了多少分”，而是追问一个更底层的问题：**机器人应当优先继承 VLM 的语义先验，还是继承视频模型的物理动态先验？**  
> **一句话结论**：2026 年的 World Action Model（WAM）并没有“杀死” VLA，但它正在把具身基础模型的主叙事从“让 VLM 学会动手”改写成“让视频模型学会动手”。  
> **关键词**：WAM、video pretraining、world model、inverse dynamics、joint video-action generation、latent action、cross-embodiment  
> **主要来源**：UniPi [`arXiv:2302.00111`](https://arxiv.org/pdf/2302.00111)、VPP [`arXiv:2412.14803`](https://arxiv.org/pdf/2412.14803)、mimic-video [`arXiv:2512.15692`](https://arxiv.org/abs/2512.15692)、Vidar [`arXiv:2507.12898`](https://arxiv.org/abs/2507.12898)、UVA [`arXiv:2503.00200`](https://arxiv.org/html/2503.00200v3)、UWM [`arXiv:2504.02792`](https://arxiv.org/html/2504.02792v3)、Cosmos Policy [`arXiv:2601.16163`](https://arxiv.org/abs/2601.16163)、DreamZero [`arXiv:2602.15922`](https://arxiv.org/abs/2602.15922)、Motus [`arXiv:2512.13030`](https://arxiv.org/abs/2512.13030)、视频生成机器人综述 [`arXiv:2601.07823`](https://arxiv.org/abs/2601.07823)

如果把 VLA 看成“给定图像与语言，直接回归动作”的主路线，那么 WAM 的核心主张是：**动作不该只从语义里长出来，而应该被“未来世界如何演化”的视频动力学约束住**。  
这并不是一个小补丁，而是训练起点、可用数据、系统瓶颈、泛化机理全都随之变化的范式争论。

## X-Ray（非本领域也能复述）
- VLA 擅长“看懂是什么”，但在新工具、新接触、新材料形变上，常常缺乏“世界会怎么动”的物理直觉。  
- WAM 想把“预测未来视频”和“生成动作”绑在一起，让机器人先学世界动力学，再从这个未来里提取动作。  
- 2026 年最重要的变化不是某篇论文单点超分，而是三条 WAM 路线都在指向同一个结论：**视频预训练可能比 VLM 预训练更接近具身控制的主监督。**

## 📍 研究全景时间线
```text
RT-1 / RT-2 / pi0
  └─ VLA 主线：VLM 预训练 -> 机器人 post-training
     强项：语义泛化
     弱项：物理动态、接触因果、动作多样性

UniPi / VPP / mimic-video / Vidar
  └─ 路线一：视频基座 + IDM/动作头（两阶段解耦）

PAD / UVA / UWM / Cosmos Policy / DreamZero
  └─ 路线二/三：video-action 联合建模 / 统一多功能模型

2026 的关键争论
  ├─ 视频质量是否决定策略质量？
  ├─ 解耦训练还是端到端联合生成更优？
  ├─ WAM 的算力成本能否降到真实部署可接受？
  └─ VAE / latent bottleneck 会不会卡死高精度操作？
```

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 路线 | 代表工作 | 核心做法 | 长处 | 主要代价 |
|---|---|---|---|---|
| **VLA 主线** | RT-2 / pi0 / pi0.5 | `Pr(a | l, o)`：图像+语言直接到动作 | 语义泛化强、部署相对便宜 | 对物理因果与新动作模式理解弱 |
| **路线一：两阶段解耦派** | UniPi / VPP / mimic-video / Vidar | 先做视频预测，再用 IDM/动作头从未来视频反推动作 | 模块化强、可复用更强视频基座 | 中间表征有信息损失，推理常多一步 |
| **路线二：端到端联合生成派** | PAD / UVA / VideoVLA / WorldVLA / Cosmos Policy / DreamZero | 同一个模型联合生成未来视频与动作 | video-action 对齐更强，物理监督更直接 | 训练/推理更贵，系统优化复杂 |
| **路线三：统一多功能模型派** | UWM / LingBot-VA / Motus | 一个模型同时承担 policy / forward dynamics / inverse dynamics / video generation | 数据利用率高，可吃纯视频与机器人数据 | 目标多、优化难、模态干扰严重 |

### 1.2 ⚡ Eureka Moment（关键洞见）
**WAM 的真正创新不是“生成视频”，而是把“未来视频”从展示结果变成动作学习的硬约束。**

### 1.3 信息流/架构图 (Flow / Diagram)
```text
传统 VLA:
  image + language + proprio
      -> action model
      -> action

WAM:
  image/video history + language + proprio
      -> world/action model
      -> future video / latent future
      -> action inferred or co-generated from that future

本质差异:
  VLA 学的是 "现在看到什么 -> 我该怎么动"
  WAM 学的是 "世界接下来会怎么变 -> 为了得到那个未来我该怎么动"
```

### 1.4 核心矛盾：VLA 到底缺了什么？

把具身智能体压缩成两个条件分布就清楚了：

```text
Policy:
  Pr(a | l, o)

World model:
  Pr(o' | l, a, o)

WAM:
  同时建模世界演化与动作生成，
  让动作预测受未来世界约束
```

VLA 的问题并不是完全不会动，而是它的预训练主监督来自**静态图文**，更偏“语义分类/对齐”；而视频天然携带：
- 时间连续性
- 物体运动轨迹
- 接触与形变
- 因果顺序
- 成败差异的动态信号

## 2. 数学核心：WAM 到底比 VLA 多了什么？(Math Core)

> Napkin Formula：VLA 学 `Pr(a | l, o)`；WAM 则试图让动作服从 `Pr(a, o_future | l, o_now)` 或它的等价分解。

### 2.1 两阶段解耦派的最小形式
```text
Step 1: video_model predicts future visual states
  o_future ~ p_theta(o_future | o_now, l)

Step 2: inverse dynamics / action head infers action
  a ~ p_phi(a | o_now, o_future, q)
```

直觉：先“做梦”，再问“为了让这个梦发生，我该怎么动？”

### 2.2 联合生成派的最小形式
```text
p(o_future, a | context)
  = p(o_future | context) * p(a | o_future, context)
```

在实现上通常并不是显式拆成两个独立模型，而是用 joint denoising / joint latent prediction 一起学。  
DreamZero 就把这件事做成：**视频 latent + action chunk 联合 flow matching**，并在闭环中用真实观测回写，打断纯自回归漂移。

### 2.3 统一多功能模型派：扩散时间步就是任务开关

UWM 一类工作的优雅点在于：同一模型只通过视频/动作模态各自的噪声时间步设置，就能切出不同功能。

```text
tau_a random, tau_v = 0
  -> policy mode

tau_a = 0, tau_v random
  -> forward dynamics mode

tau_a = T, tau_v random
  -> pure video prediction mode

tau_a random, tau_v = T
  -> inverse dynamics mode
```

这意味着“策略、世界模型、逆动力学”不再是三个网络，而是同一个大模型在不同条件下的不同工作点。

## 3. 三条路线带数字走一遍 (Worked Example)

### 3.1 路线一：两阶段解耦派

这一路线最直观，也最早成熟。

代表脉络：
- **UniPi（2023）**：文本条件视频生成 + CNN IDM，首次把“视频作为通用接口”清晰地放进机器人策略学习里。  
- **VPP（2025）**：不再做完整视频去噪，而是直接拿视频扩散模型中间层的 predictive visual representation，只需一次前向。  
- **mimic-video（2025）**：提出 partial denoising，让模型走到中间噪声层就停；既保留动力学，又大幅降低生成成本。  
- **Vidar（2025）**：强调跨本体与极低适配数据，用 masked IDM 学动作相关区域。  

文中最值得记住的几个数字：
- `mimic-video` 在 SIMPLER-Bridge 上约 **56.3%** 平均成功率，对比 OpenVLA 的 **14.6%**。  
- 文中归纳其样本效率提升约 **10×**、收敛速度约 **2×**。  
- `Vidar` 用约 **20 分钟** 人类示范，在 RoboTwin 50 个任务上达到 **65.8%** 成功率。  

这一路线的最强结论是：

```text
Video quality strongly correlates with policy quality
```

也就是：如果视频 backbone 无法预测靠谱的未来，动作头几乎一定学不好。

### 3.2 路线二：端到端联合生成派

这条路线认为：既然视频与动作本质上属于同一个未来，不该先分开再拼回来。

代表脉络：
- **PAD（2024）**：提出图像预测与动作生成共享相同去噪动力学。  
- **VideoVLA（2025）**：把联合生成扩到更大的视频基座；文中给出从 CogVideoX-5B 初始化约 **80.4%**，而从头训练约 **12.6%**，说明视频预训练不是锦上添花。  
- **WorldVLA**：在统一模型内同时更新 action model 与 world model，并加 action attention masking 防误差积累。  
- **Cosmos Policy（2026）**：最“极简”的方案之一，把动作/本体感受/value 都伪装成 latent frames，直接塞回视频模型序列。  
- **DreamZero（2026）**：当前最完整的 WAM 工程答案之一。  

DreamZero 在这条路线上的象征意义最大：
- 零样本环境与任务泛化里，文中整理其平均任务进度约 **62.2% vs 27.4%**（相对最强预训练 VLA 超过 2×）  
- 只用 **30 分钟** play data 迁移到新机器人  
- 通过异步执行、CFG 并行、DiT cache、`torch.compile`、CUDA graphs、量化等整套优化，把 **14B** 视频扩散模型做到 **7Hz** 闭环控制  

### 3.3 路线三：统一多功能模型派

如果说路线二还主要在讨论“怎么把视频与动作一起学”，路线三更进一步：**一个模型干四件事**。

代表脉络：
- **UVA**：共享 latent backbone + 解耦轻量 diffusion heads；若推理只要动作，可以完全跳过视频头。  
- **UWM**：用独立的 `tau_a` / `tau_v` 让同一网络自然切出 policy / forward / inverse / video generation。  
- **LingBot-VA**：从 chunk-level 换到 token-level，更像“边想边做边纠错”；引入 Mixture-of-Transformers 避免视频 token 与动作 token 互相污染。  
- **Motus**：把 optical flow 当成跨本体通用运动表征，用三专家 MoT 融合语义理解、视频生成和动作。  

文中给出的最惊艳结论来自 `Motus`：
- 在 RoboTwin 2.0 的 50 任务训练里，随着任务数增加，`Motus` 平均成功率持续上升，而 `pi0.5` 持续下降  
- 文中整理最终约 **87.0%**，比 `pi0.5` 高约 **45 个百分点**  

路线三最值得关注的不是单个分数，而是它给出的 scaling 叙事：

```text
任务越多 -> 共享 world knowledge 越丰富 -> 更多任务一起受益
```

这和早期多任务 VLA 常见的“任务一多互相干扰”形成鲜明对比。

## 4. 工程视角：三条路线到底怎么取舍？(Engineering View)

### 4.1 解耦 vs 联合：更像“scale 问题”而不是哲学问题

| 问题 | 解耦派答案 | 联合派答案 | 更可能的工程结论 |
|---|---|---|---|
| 数据少时怎么训？ | 冻住视频 backbone，更稳 | 端到端容易过拟合/扰动视频先验 | 小数据阶段解耦更稳 |
| 数据多时怎么扩？ | 中间接口可能成瓶颈 | 联合建模上限更高 | 大模型大数据更适合联合 |
| 推理代价 | 常多一步视频/latent 处理 | 一体化但 backbone 更重 | 要靠 partial denoising / skip video head / one-step 去噪 折中 |

`mimic-video` 给出的信号是“解耦训练反而更好”；`DreamZero` 给出的信号是“端到端联合 + 大规模优化更强”。  
这两者并不真正冲突，更像是在不同 scale、不同 backbone、不同数据分布下分别成立。

### 4.2 推理速度是不是 WAM 的死穴？

目前还不能说彻底解决，但已经出现三种很重要的“中间道路”：
- **partial denoising**：不生成完整视频，只保留足够的动力学 latent  
- **skip video head**：像 UVA 那样训练时学视频，推理时若只要动作就跳过视频生成  
- **one-step / few-step denoising**：像 DreamZero-Flash 那样进一步压缩去噪步数  

换句话说，工程落地不一定要“完整生成高保真视频”，而是要**尽可能提取视频世界先验，同时不支付全部生成成本**。

### 4.3 VAE / latent bottleneck 会不会卡死高精度任务？

这篇长文提出了一个非常关键但目前尚未被系统回答的问题：  
如果控制性能确实受视频表征质量支配，那么 latent VAE 的下采样、模糊化、压缩误差，是否会成为插 USB、拧螺丝、精密装配的硬上限？

目前可以保守地说：
- 对中等精度 manipulation，视频 latent 已足够提供强监督  
- 对亚毫米级接触任务，这个瓶颈大概率迟早会显性化  

## 5. 数据与评测：WAM 为什么有不同的 scaling story？(Data & Eval)

### 5.1 VLA 与 WAM 的数据来源差异

| 维度 | VLA | WAM |
|---|---|---|
| 预训练主数据 | 图文对 / 静态图像 | 互联网视频 / 人类操作视频 / 机器人视频 |
| 对动作标注依赖 | 高 | 可低很多，部分路线只需少量动作对齐 |
| 学到的先验 | 语义、类别、语言对齐 | 时序、因果、运动、接触变化 |
| scaling 上限 | 受高质量机器人数据限制 | 可同时吃机器人数据和互联网视频 |

### 5.2 为什么这会改写 scaling 叙事？

```text
VLA scaling bottleneck:
  更多机器人轨迹 -> 更贵的人类/真机/仿真采集

WAM scaling bottleneck:
  更强的视频基座 + 更好的动作对齐方式 + 更便宜的推理
```

这意味着一旦视频模型基础设施继续改善，WAM 的天花板可能不再由“你采了多少条机器人示教”决定，而是由“你见过多少种物理现象”决定。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

### 6.1 这三条路线真正带来的能力
- **更强的新动作泛化**：尤其在训练集里不常见的行为模式上，不容易塌到 `pick-and-place` 模板。  
- **跨本体迁移更自然**：视频是共享接口，动作对齐可后置。  
- **能同时吸收 web 视频与机器人轨迹**：给出不同于 VLA 的规模化路径。  

### 6.2 失败模式与悬而未决的问题
- **视频质量 = 策略质量** 的边界在哪？这件事可能只在中等精度任务成立。  
- **推理频率不足**：7Hz 已经很强，但对高速接触任务仍可能不够。  
- **模态互相污染**：统一模型里视频 token、动作 token、语义 token 很容易争抢同一表示空间。  
- **算力门槛高**：视频模型微调与部署成本目前仍显著高于多数 VLA。  
- **评测协议不可比**：不同论文的机器人平台、任务集、控制频率、成功定义差异很大。  

### 6.3 Hidden Assumptions（隐含假设）
- **未来视频足以决定好动作**：也就是“能想清未来，就能倒推出动作”。  
- **视频表征确实包含控制所需的关键物理变量**：而不是只提供外观上的时序连贯。  
- **互联网视频中的物理规律能迁移到机器人本体执行**：这是 WAM 最大胆、也是最值钱的假设。  

## 7. 与相关工作对比：这场争论到底该怎么回答？(Comparison)

### 7.1 三条路线的回答方式

| 问题 | 路线一：解耦派 | 路线二：联合派 | 路线三：统一派 |
|---|---|---|---|
| “世界模型”扮演什么角色？ | 先想未来，再倒推动作 | 未来与动作联合生成 | 一个模型里自然涌现多种角色 |
| 最强卖点 | 训练稳、模块化、好替换 backbone | 对齐最强、物理监督直接 | 数据利用率最高、功能最全 |
| 最大风险 | 中间接口丢信息 | 训练/推理成本太高 | 目标过多、优化更难 |
| 适合谁 | 先把视频先验接进现有政策栈 | 追求上限的端到端团队 | 想做 foundation model 的大团队 |

### 7.2 面试 Tip

如果被问“VLA 会被 WAM 取代吗？”，一个稳妥回答是：

```text
短期不会，长期未必。

VLA 仍然是最成熟、最便宜、最容易部署的语义动作范式；
WAM 的真正价值在于把视频动力学变成控制主监督，
让具身基础模型第一次拥有了"从互联网视频规模化吸收物理先验"的路径。

更现实的判断不是“二选一”，而是：
未来几年 VLA 负责语义接口，WAM 负责物理生成，
两者很可能会融合成更强的统一具身模型。
```

---

## 相关笔记
- DreamZero 深拆：[`../dreamzero_world_action_models_zero_shot_policies_2026.md`](../dreamzero_world_action_models_zero_shot_policies_2026.md)
- 早期 WAM / DreamZero 旧稿：[`../world_action_models_are_zero_shot_policies_dissection.md`](../world_action_models_are_zero_shot_policies_dissection.md)
- 视频世界模型机器人综述：[`./video_generation_models_in_robotics_survey_2026.md`](./video_generation_models_in_robotics_survey_2026.md)

## 参考链接
- UniPi: [`Learning Universal Policies via Text-Guided Video Generation`](https://arxiv.org/pdf/2302.00111)
- VPP: [`Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations`](https://arxiv.org/pdf/2412.14803)
- mimic-video: [`mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs`](https://arxiv.org/abs/2512.15692)
- Vidar: [`Vidar: Embodied Video Diffusion Model for Generalist Manipulation`](https://arxiv.org/abs/2507.12898)
- UVA: [`Unified Video Action Model`](https://arxiv.org/html/2503.00200v3)
- UWM: [`Unified World Models: Coupling Video and Action Diffusion for Pretraining on Large Robotic Datasets`](https://arxiv.org/html/2504.02792v3)
- Cosmos Policy: [`Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning`](https://arxiv.org/abs/2601.16163)
- DreamZero: [`World Action Models are Zero-shot Policies`](https://arxiv.org/abs/2602.15922)
- Motus: [`Motus: A Unified Latent Action World Model`](https://arxiv.org/abs/2512.13030)
- Survey: [`Video Generation Models in Robotics - Applications, Research Challenges, Future Directions`](https://arxiv.org/abs/2601.07823)

---
[← Back to Theory](../README.md)

