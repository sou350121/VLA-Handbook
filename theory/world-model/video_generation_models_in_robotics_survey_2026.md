# 视频生成模型在机器人中的应用：近 300 篇体系化综述提炼（2026）(Video Generation Models in Robotics, 2026 Survey)

> **发布时间**：2026-01-12（arXiv v1）  
> **论文题目**：Video Generation Models in Robotics - Applications, Research Challenges, Future Directions  
> **作者/机构**：Princeton University / Temple University（Mei, Yin, Shorinwa 等）  
> **论文链接**：`https://arxiv.org/abs/2601.07823`（PDF：`https://arxiv.org/pdf/2601.07823`）  
> **DOI（arXiv）**：`https://doi.org/10.48550/arXiv.2601.07823`  
> **核心定位**：把**视频生成模型**作为机器人可学习的 **具身世界模型（embodied world model）**：用于 **(i) 低成本数据生成与动作预测（IL）**、**(ii) RL 的动力学/奖励建模**、**(iii) 可扩展策略评估**、**(iv) 视觉规划（visual planning）**。  
> **一句话 takeaway**：真正的转折点不是“画质”，而是把机器人决策从“直接出动作”推向“先想象再行动”；但要落地，必须把 **物理幻觉/指令偏差/不确定性/长时一致性/成本与安全**当成一等公民。

```text
                 Video Generation Models in Robotics (Survey, 2026)
                 ================================================

   Section 2: Background / Building Blocks
   --------------------------------------
     (A) Markovian state-based world models
         s_{t+1} ~ p(s_{t+1} | s_t, a_t)  + reward model

     (B) Diffusion / Flow-matching video models
         latent diffusion, U-Net/DiT, guidance, multi-modal conditioning

     (C) Video JEPA (V-JEPA)
         masked spatiotemporal latent prediction (understanding / prediction / planning)

                            |
                            v
   Section 3: Robotics Applications (4 big buckets)
   ------------------------------------------------
     (1) IL: data generation + action prediction
         - generate demos -> estimate actions (latent actions / IDM / modular tracking)
         - or use video models as policy backbones (video + action)

     (2) RL: dynamics + rewards modeling
         - world model as environment / reward proxy / dense rewards via VLMs

     (3) Policy evaluation (scalable, reproducible)
         - rollouts in video WM -> success rate estimates -> ranking (Pearson / MMRV)
         - knobs: multi-view + wrist, history conditioning, joint-state conditioning

     (4) Visual planning
         - action-guided: propose actions -> rollout -> score
         - action-free: generate video plan -> use frames as subgoals -> policy/IDM follow

                            |
                            v
   Section 4: Evaluation
   ---------------------
     - metrics: PSNR/SSIM/CLIP/FID/LPIPS + FVD/KVD/FVMD
     - benchmarks: WorldModelBench, VBench, EvalCrafter, Physics-IQ, ... + safety benches

                            |
                            v
   Section 5: Open challenges -> Future directions
   -----------------------------------------------
     - hallucinations & physics violations
     - uncertainty quantification (UQ)
     - instruction following & camera control
     - long-horizon video generation
     - data curation (success + failure) + training/inference cost + safety guardrails
```

---

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 作者给出的总地图：4 类机器人应用 + 3 类背景模块

论文把“视频模型进机器人”拆成两层（结构非常干净）：

- **背景（Section 2）**：
  - **Markovian state-based world models**（经典 latent dynamics：RSSM/Transformer/latent diffusion 等）
  - **Diffusion / Flow-matching video models**（U-Net/DiT + latent diffusion + guidance）
  - **Video JEPA models（V-JEPA）**：在 latent space 预测被 mask 的时空特征，用于理解/预测/规划

- **四大机器人应用（Section 3）**：
  - **(i) IL：robot data generation + action prediction**
  - **(ii) RL：dynamics + rewards modeling**
  - **(iii) policy evaluation**
  - **(iv) visual planning**

> 你现在看到的“太表浅”，核心原因就是没把这张作者提供的地图吃进去：只要按它写，天然会更“体系化”。

### 1.2 世界模型的两派：state-based vs video world model

论文在 Background 里明确区分：

- **Markovian state-based**：假设未来只依赖 \(s_t\) 与 \(a_t\)，多在 latent space 学动力学/奖励。
- **Video world model**：不显式维护马尔可夫状态，学习“时空映射”，直接建模世界随时间演化的视觉变化（更高保真、也更贵/更难控）。

### 1.3 视频世界模型的两种形态：implicit vs explicit（落地分水岭）

论文在应用章节把 video world model 再细分：

- **Implicit video world model**：3D 表征“只在模型里”，你只能通过生成视频来“看见世界”。
- **Explicit video world model**：生成可用于重建的中间产物（multi-view / depth / raymap），再显式构建 3D/4D 表征（point cloud / voxel map / 4D Gaussian Splatting）。

工程直觉：**explicit 更容易做一致性检查与 safety gating**，但对多视角数据/标定/后处理要求更高。

---

## 2. 数学核心：作者在论文里用到的“最小公式骨架” (Math Core)

### 2.1 Markovian state-based world model（论文给的标准形式）

作者给的形式（符号与论文一致）：

$$
s_{t+1}\sim p_{\eta}(s_{t+1}\mid s_t,a_t)
$$

并拆成 encoder / dynamics / rewards：

$$
\text{Encoder: } s_t\sim \mathcal{E}_{\gamma}(s_t\mid o_t),\quad
\text{Dynamics: } \hat{s}_{t+1}\sim p_{\eta}(\hat{s}_{t+1}\mid s_t,a_t),\quad
\text{Reward: } \hat{r}_{t+1}\sim p_{\zeta}(\hat{r}_{t+1}\mid \hat{s}_{t+1})
$$

### 2.2 Diffusion video model 的核心 loss（论文给了两种常用写法）

噪声回归：

$$
\mathcal{L}_{\epsilon}=\mathbb{E}_{x_0,t,\epsilon}\left[\left\|\epsilon-\epsilon_{\theta}(x_t,t)\right\|^2\right]
$$

速度参数化（更稳定）：

$$
\mathcal{L}_{v}=\mathbb{E}_{x_0,t,v_t}\left[\left\|v_t-v_{\theta}(x_t,t)\right\|^2\right]
$$

classifier-free guidance（控制条件强度）：

$$
\tilde{\epsilon}_{\theta}(x_t,t,y)=(1+w)\epsilon_{\theta}(x_t,t,y)-w\epsilon_{\theta}(x_t,t)
$$

工程含义：把 \(y\) 设计成 **action / robot state / trajectory / multi-view history**，就是“动作条件世界模型”的一条主流实现路径。

---

## 3. 四大应用：怎么用、怎么评估、代表性工作入口 (Applications)

> 下面按论文四类应用写“可检索索引”。我只列论文中反复出现、且能作为入口的代表性工作（不把 300 篇全抄一遍）。

### 3.1 IL：数据生成 & 动作预测（Cost-effective data generation + action prediction）

论文认为：用视频模型做 IL 有两条主线：

- **(A) Video model 作为 data generator**：生成“专家演示视频”，再把视频转成 action label。
- **(B) Video model 作为 policy backbone**：联合预测 future video + action，把 action 与“世界如何变”绑定。

#### A) 生成数据后，怎么把 video 变成 action？（Figure 5）

论文把 action estimation 分成两条路线：

- **端到端（End-to-end）**：
  - **Latent action models**：学习离散 latent action codebook（常需少量 action-labeled 数据对齐 latent↔robot action）
  - **Inverse dynamics models (IDM)**：监督 video→action（需要 action-labeled 数据，但可 zero-shot 部署，无需再对齐）

- **模块化（Modular）**：
  - pose tracking / depth / optical flow / CAD → retargeting
  - 优点：可解释、常可 zero-shot；代价：对“相机假设”（比如 static camera）极敏感

论文点名入口：

- DreamGen（IDM 路线）：`https://arxiv.org/abs/2505.12705`
- Video Prediction Policy (VPP)：`https://arxiv.org/abs/2412.14803`
- ARDuP（Active region mask conditioning）：IROS 2024（论文内引用）
- Vidar：`https://arxiv.org/abs/2507.12898`

#### B) 作为数据生成器：为什么只用 text/image 条件不够？

论文直说：text-conditioned / image-conditioned 的表达力往往不够“可控”，因此会依赖更细粒度控制（keypoints/trajectory/control）。它点名的“可大规模视频模型入口”包括：

- Cosmos（NVIDIA）：`https://arxiv.org/abs/2501.03575`
- Wan：`https://arxiv.org/abs/2503.20314`
- Human2Robot（paired human-robot videos）：`https://arxiv.org/abs/2502.16587`

#### C) 作为 policy backbone：unified video-action

论文点名入口：

- Unified Video Action Model（UVA）：`https://arxiv.org/abs/2503.00200`
- DreamVLA：`https://arxiv.org/abs/2507.04447`

### 3.2 RL：动力学与奖励（Dynamics + rewards modeling）

论文点名的 RL 路线非常清晰：

- Dreamer 4（在可扩展 world model 里训练 agent）：`https://arxiv.org/abs/2509.24527`
- World-Env（video world model + VLM 作为 reward reflector）：`https://arxiv.org/abs/2509.24948`
- VIPER（NeurIPS 2023）：视频预测似然作为 reward（论文内引用）
- Diffusion Reward（ECCV 2024）：条件熵作为 reward（论文内引用）

### 3.3 Policy evaluation：用视频 world model 做可扩展评估

论文给了几个非常“落地”的经验结论（这也是你要的“更多细节”）：

- **multi-view，尤其 wrist camera**：实证上可减少 hallucination（但对“违反物理规律”帮助有限）
- **history conditioning**：用过去窗口帧/稀疏帧缓解长 rollout 误差累积
- **加入 robot joint poses**：提升 frame-level action controllability

评估指标（论文明确写了）：

- **Pearson correlation**：预测 success rate 与真实 success rate 的相关
- **MMRV（Mean Maximum Rank Violation）**：衡量“排序错乱”的严重程度（对 policy ranking 很关键）

论文点名入口：

- 1X World Model（technical report）：论文内引用
- Veo world simulator 评估 Gemini robotics policies：`https://arxiv.org/abs/2512.10675`
- WorldModelBench（world model judge）：`https://arxiv.org/abs/2502.20694`

### 3.4 Visual planning：动作引导 vs 无动作

论文把视觉规划拆成两派：

- **Action-guided**：action proposals → video rollout → objective 打分选最优
- **Action-free**：直接生成 video plan，用中间帧当 image subgoals，再用 BC/IDM 去跟随

论文点名入口：

- Video Language Planning：`https://arxiv.org/abs/2310.10625`
- FLIP：`https://arxiv.org/abs/2412.08261`
- MindJourney：`https://arxiv.org/abs/2507.12508`
- NovaFlow：`https://arxiv.org/abs/2510.08568`

---

## 4. 评测：指标与基准（Metrics & Benchmarks）

### 4.1 Metrics（论文列举）

- frame-level：PSNR / SSIM / CLIP similarity / FID / LPIPS
- spatiotemporal：FVD / KVD / FVMD（motion feature）
- 工具型评估：optical flow 一致性；VLM judge 物理一致性（论文提到有工作用 VLM 评估 physics consistency）

### 4.2 Benchmarks（论文列举）

- **WorldModelBench**：`https://arxiv.org/abs/2502.20694`
- EvalCrafter（CVPR 2024）、VBench（CVPR 2024）、PAI-Bench、T2V-CompBench、WorldSimBench
- Physics-IQ、PhyGenBench、VideoPhy、VP\(^2\)（control-centric benchmark；论文强调“感知指标≠物理一致性”）
- Safety：SAFEWatch、T2VSafetyBench（论文列举）

---

## 5. 失败模式与未来方向（论文 Section 5 的“可直接工程化”部分）

### 5.1 Hallucinations & physics violations

论文指出：多视角输入（特别 wrist view）能减少 hallucination，但对“物理规律违背”帮助有限。并列出常见违背：

- Newton laws / energy & mass conservation / gravity effects
- 固-固接触：不理解材料属性、动量守恒、不可穿透
- 流体：倒入杯子但体积不变等（缺乏流体力学与质量守恒）

未来方向（论文列举）：

- 引入 physics priors / physics sim（Hamiltonian / Lagrangian priors；PhysGen / WonderPlay 等“sim 粗轨迹 + diffusion 修复”思路）
- affordance-based video understanding：预测交互热点（hotspots / affordance maps）作为 guidance

### 5.2 Uncertainty quantification（UQ）

论文点名：S-QUBED（task-level）与 C\(^3\)（dense subpatch uncertainty），并强调 video UQ 的难点来自时序相关性与生成开销。

### 5.3 Instruction following

论文指出：很多模型能生成 prompt 里的 agent，但对 action 跟不住；且 camera motion 难控会直接破坏机器人数据生成的假设（比如 static camera → 3D back-projection）。

### 5.4 Long video generation

论文给出“硬事实”：主流 SOTA 视频模型仍是秒级：

- Veo 3.1：8 秒
- Wan 2.5：10 秒

并列举了 MALT / FramePack / TTTVideo / LaCT / LCT / MoC / Diffusion Forcing / NUWA-XL 等长视频方向，但强调离“分钟级稳定”仍远。

### 5.5 Data curation costs（数据管线）

论文把数据整理管线拆成三段：video splitting → filtering → annotation（VLM captioning + 可能的人审）。并强调机器人 world model 需要 **成功 + 失败**数据，否则会出现 optimistic bias（例如“把物体 hallucinate 到更好抓的位置”）。

### 5.6 Training & inference costs

论文给出两个关键数字：

- Open-Sora 2.0 训练成本约 \$200k（开源 SOTA 之一）
- Veo 3 在 A100 上约 12 fps

并列举压缩、稀疏 temporal attention、cache、shortcut/consistency、量化、蒸馏等加速方向。

---

## 参考（论文内反复出现的入口）

- Survey：`https://arxiv.org/abs/2601.07823`
- DreamGen：`https://arxiv.org/abs/2505.12705`
- World-Env：`https://arxiv.org/abs/2509.24948`
- Dreamer 4：`https://arxiv.org/abs/2509.24527`
- WorldModelBench：`https://arxiv.org/abs/2502.20694`
- Veo world simulator（Gemini robotics eval）：`https://arxiv.org/abs/2512.10675`

---
[← Back to Theory](../README.md)
