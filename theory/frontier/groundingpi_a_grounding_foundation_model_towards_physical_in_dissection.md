# GroundingPI：以视觉基元构建面向物理智能的 grounding 基础模型 (GroundingPI: A Grounding Foundation Model towards Physical Intelligence with Visual Primitives)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-11-23
>
> **论文**: GroundingPI: A Grounding Foundation Model towards Physical Intelligence with Visual Primitives
> **链接**: https://arxiv.org/abs/2609.39601
> **核心定位**: 用 4B 自回归 grounding 模型把点/框表示为量化坐标 token，作为 VLA/WAM 的感知前端；在 34 个 grounding benchmark 上平均 73.68%，并迁移到 nuScenes 开环轨迹与 RoboTwin 2.0 / RoboCasa-GR1 仿真操作。

> **版本与摘要说明**: 本文只依据 arXiv:2609.39601v3 PDF 正文（HTML 抓取，页眉标注 `[cs.CV] 19 Nov 2026`）写作。候选元数据摘要到 “generates” 处截断，未给出完整实验主张，因此不用于补写或替代 PDF 正文。下文数字均取 PDF 正文；若候选摘要与正文有差异，以 PDF 正文为准。候选来源记录为 cs.LG，而 PDF 页眉为 cs.CV，本文以 PDF 正文为证据。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 把 grounding 做成语言模型原生输出协议，能同时提升密集/小目标感知与下游动作学习；平均 73.68%，高于 GPT-6 Astra 的 71.54% |
| 适合精读 | 做 VLA/WAM 感知前端、密集/小目标 grounding、仿真操作迁移与 action-data efficiency 的读者，重点看 §1.2、§2、§5、§7 |
| 可以跳过 | 只关心真实机器人闭环控制、实时延迟或完整双系统部署的读者；论文没有真实机器人实验 |
| 落地可行性 | 中：接口简单、输出 token 省、仿真迁移有证据；但闭环延迟、真实部署与 grounding→action 因果未验证 |
| 主要风险 | 预训练基座、规模与监督不同，架构归因不干净；OCR 催化与部分机制仍是作者假设 |

💡 **X-Ray 开场**  
这篇论文问的是：VLA/WAM 的感知前端是否应专门为“精确指代与定位”预训练，而不是直接拿通用 VLM 或视频生成骨干？作者给出 GroundingPI：一个 4B 自回归 grounding 模型，用共享词表输出框和点，再用受控 Action DiT 比较不同预训练骨干对动作学习的影响。对 VLA 研究者而言，它的价值不在“又一个 grounding 榜”，而在于把感知前端当作可替换模块，并给出仿真操作与开环驾驶的条件性迁移证据。

📍 **研究全景时间线**

```
2024  π0：VLA flow model（论文引用 Black et al., 2024）
2025  π0.5 / GR00T N1：感知监督与双系统参考（论文引用）
2026  GroundingPI v3：把 grounding 作为感知基础，而不是动作模型的辅助头（本文）
```

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 模块/设置 | GroundingPI | 关键数字/备注 |
|---|---|---|
| 视觉前端 | MoonViT-V2（Kimi K3） | 27 层、hidden 1024、patch 14；动态 patch 上限 4096，2×2 聚合后最多 1024 个视觉 token |
| 语言解码器 | Qwen3-4B | 36 层 full attention；hidden 2560；32 query heads / 8 KV heads；无 sliding window；RoPE base 5,000,000；位置容量 262,144（不是训练序列长度） |
| 输出协议 | 标签 + 协议标记 + 1,000 个量化坐标 token `<0>–<999>` | 框用 4 个连续坐标 token，点用 2 个；缺失目标输出 `None`；总词表 152,670 |
| 训练阶段 | 基础 VLM 预训练（Stage 1–3）→ 坐标对齐 SFT（Stage 4）→ GRPO RL | 总训练曝光 500.41B + 221.17B tokens；SFT 每 stage 配置 1 epoch |
| 下游动作接口 | π-style Action DiT，16 个原子 block（8 cross-attention + 8 self-attention） | 8 个 normalized-depth 特征，每层 64 tokens、width 1024；action horizon 16；flow matching；推理 4 Euler steps |
| 驾驶接口 | 当前前视图像 + 最多 7 条 ego-state 记录（0.5 s 间隔，含当前，最多 3 s 历史） | 输出 6 个未来 waypoint，0.5 s 间隔、3 s horizon；开环 L2 |
| 评测规模 | 44 个 baseline、34 个 grounding benchmark、11 类感知能力 | GroundingPI 为 10 runs（5 seeds × 2 runs/seed）；本地 baseline 为 3 runs（3 seeds × 1 run/seed） |

### 1.2 关键机制 (Key Mechanism)

- **感知先于动作**：作者主张先学“指向哪里/目标在哪里”，再让有限动作数据学“怎么动”。GroundingPI 不是把框预测塞进策略当辅助 loss，而是先训练一个可复用的 grounding 基础模型。
- **共享词表统一任务**：grounding、referring、pointing、OCR、GUI、layout、visual prompting 都被写成“标签 + 几何 payload”的结构化生成；框和点只差坐标数量。
- **数据引擎做字段级融合**：多教师标注按 semantic / geometric field 融合；字段经验证为 accepted / rejected / unresolved，只有所有必填字段 accepted 才接受监督。缺失目标也要验证“确实不存在”，空预测不能证明缺失。
- **SFT 学协议，GRPO 优化定位**：SFT 用 teacher forcing 学坐标与输出格式；RL 用 `G=8` 条响应/图像-查询对，grounding 奖励由 `R_set` 与 `R_strict` 组成，OCR 用 Hungarian 一对一匹配的复合奖励。
- **受控 backbone 比较**：下游固定 Action DiT、动作数据、优化预算、动作表示与推理流程，只替换预训练骨干；这是论文把“感知基础”与“动作学习”拆开的关键实验设计。

⚡ **Eureka Moment**：把 grounding 从任务头改造成语言模型的原生输出协议，再用受控 action head 验证它作为感知前端的迁移价值。

### 1.3 信息流/架构图 (Flow / Diagram)

```
Image I + Prompt P
  → MoonViT-V2（≤4096 patches；2×2 聚合）
  → projector（4096→4096→2560）
  → Qwen3-4B decoder（36 层 full attention）
  → 结构化响应：
       标签 + <0>–<999> 坐标
       ├─ grounding / pointing / OCR / GUI / layout
       └─ driving：6 个未来 waypoint（metric bins）
  → 下游：
       ├─ 仿真操作：Action DiT（16 blocks）→ action chunk 16
       └─ 驾驶：开环 L2
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**：

```
p_θ(Y|I,P)=∏_{i=1}^{|Y|} p_θ(y_i|V,P,y_<i)
```

目标：用自回归语言模型统一生成语义标签与量化几何；训练时只监督 assistant response 中的语义、坐标和协议 token。

```
L_SFT = -(1/|S|) Σ_{t∈S} log p_θ(y_t | I,P,y_<t)
```

坐标量化：

```
q(v)=floor((999v+500)/1000)，v∈[0,1000]
x_rec=(q/999)·图像维度
```

驾驶使用独立 metric bins：

```
q_z=round_even(999(z-a)/(b-a))，z∈[a,b]
ẑ=a+(b-a)q_z/999
L2@h=(2h)^{-1} Σ_{k=1}^{2h} d̄_k，h∈{1,2,3}
L2_Avg=(1/3) Σ_{h=1}^{3} L2@h
```

GRPO 优势与目标（简化写法）：

```
Ã_i=0.7 Z(R_set,i)+0.3 Z(R_strict,i)
A_i=(Ã_i-μ_Ã)/(s_Ã+δ)
J(θ)=E[1/N Σ_i 1/|Y_i| Σ_t {min(r_i,t A_i, clip(r_i,t,1-ε,1+ε)A_i)-β d_i,t}]
```

`R_set` 是 reference-wise max-IoU 匹配的 F1 风格覆盖率代理；`R_strict` 由多阈值 F1、定位、格式、计数、排序和重复/超大框惩罚组成。`d_i,t` 是采样 KL 正则项；参考策略冻结自 SFT checkpoint。

| 符号 | 含义 |
|---|---|
| `I,P` | 图像与语言查询 |
| `V` | projector 后的视觉 embedding |
| `S` | 被监督的 assistant response 位置 |
| `q(v)` | 图像坐标量化函数 |
| `d̄_k` | 第 k 个未来 step 的平均欧氏误差 |
| `r_i,t` | 新策略/旧策略的 token 概率比 |
| `A_i` | 响应级标准化优势 |
| `ε,β` | GRPO clip 与 KL 系数；论文取 0.2 与 0.02 |

直觉：GroundingPI 把“框/点”变成语言 token 预测问题，因此不需要额外的 IoU 回归头；坐标 token 与语义 token 共享同一个输出头。代价是坐标被离散化，严格边界精度受量化与自回归生成成本约束。

## 3. 带数字走一遍：玩具例子 (Worked Example)

> **假设例子，不是论文实验。** 只用于验算论文给出的 IoU 容忍公式与时间跨度公式。

**例 A：小目标对位移的敏感性。** 假设两个同高同宽框为 `w=20 px`、`h=20 px`，仅水平位移 `δ`。论文式 (3) 给出：

```
IoU(δ)=(w-|δ|)/(w+|δ|)
|δ|≤w(1-τ)/(1+τ)
```

验算：

| τ | 允许位移 `|δ|` | 验算 |
|---|---:|---|
| 0.50 | 20×0.5/1.5 = 6.667 px | (20-6.667)/(20+6.667)=13.333/26.667=0.500 |
| 0.75 | 20×0.25/1.75 = 2.857 px | (20-2.857)/(20+2.857)=17.143/22.857≈0.750 |
| 0.95 | 20×0.05/1.95 = 0.513 px | (20-0.513)/(20+0.513)=19.487/20.513≈0.950 |

这说明小目标的绝对定位容忍度随框宽线性缩小；密集场景中“分开相邻实例”与“小目标精确边界”是同一类空间判别问题。

**例 B：历史帧与未来 waypoint 的跨度。** 假设 `Δt=0.5 s`，历史记录 `K=7` 且包含当前帧，则首末跨度是：

```
(K-1)×Δt = (7-1)×0.5 = 3.0 s
```

未来 6 个 waypoint、间隔 0.5 s：首末预测点跨度是 `(6-1)×0.5=2.5 s`，但按论文“over a three-second horizon”，终点相对当前参考是 `6×0.5=3.0 s`。**不要把 2.5 s 的首末跨度写成 3 s 历史记忆窗口；也不要把 6 个未来 waypoint 当作已观测历史。**

## 4. 工程视角 (Engineering View)

| 工程点 | 论文证据 | 工程含义 |
|---|---|---|
| 输出 token 效率 | COCO/Dense200 上 7.6/5.1 tokens per box；SEED1.5-VL 为 148.8/74.5 | 论文计算约 19.6× / 14.6× fewer tokens；这是序列化效率，不是实测加速比 |
| 坐标表示 | 量化坐标 Avg 73.68；文本坐标 71.21；文本坐标相对速度 0.25× | 量化显著省 token，但严格边界仍受离散化影响 |
| 视觉 token 上限 | 最多 4096 patches → 最多 1024 projected visual tokens | 控制上下文与显存；高分辨率小目标仍可能受 patch 预算限制 |
| 动作推理 | 4 Euler steps；每次观测缓存一次 backbone forward | 动作端计算可控，但论文没有给出闭环控制频率或端到端延迟 |
| 上下文长度 | 训练 context 8192 / 32768；位置容量 262,144 | 位置容量不是训练序列长度；不能把它当作“记忆窗口” |
| 动作 chunk | action horizon 16，仅是该受控 Action DiT 设置 | 不能写成所有实验共有的记忆窗口或历史上下文 |

关键 trade-off：GroundingPI 用离散坐标换紧凑序列与统一接口，但高 IoU 边界、极小目标和密集输出仍受自回归生成本身限制。论文的 generation cost 随预测框数上升，但没有测量闭环动作延迟。

## 5. 数据与评测 (Data & Eval)

| 项目 | 论文设置 | 备注 |
|---|---|---|
| 预训练数据 | 公开数据集 + 专用 data engine 标注；多教师字段级融合与任务验证 | 接受标签训练统一 grounding expert，再迭代标注；不是单一数据集 |
| Grounding 评测 | 34 benchmarks、11 类能力、44 baselines | 指标含 F1mIoU、F1@Point、task-specific accuracy；不是所有 baseline 都支持所有接口 |
| 实验次数 | GroundingPI 10 runs（5 seeds × 2 runs/seed）；本地 baseline 3 runs（3 seeds × 1 run/seed） | 这是评测重复次数，不是训练 epoch 或时间步 |
| 驾驶 | nuScenes 离线开环；当前前视图像 + 最多 7 条 ego-state（0.5 s 间隔，含当前，最多 3 s 历史）；6 个未来 waypoint | 论文报告 L2 Avg 0.296 m；Qwen3-VL-4B 0.301 m，RynnBrain 0.308 m |
| 仿真操作 | RoboTwin 2.0：50 个桌面任务，Full 与 Clean2Random；RoboCasa-GR1：humanoid dexterous-hand，ID + 三个 OOD suite | 固定 Action DiT、动作数据、优化预算；40,000 updates；40 accelerators；global batch 1280；推理 4 Euler steps |
| 仿真结果 | RoboTwin Full 66.20%；RoboCasa-GR1 Full 37.75%，RynnBrain 39.00%；四个 OOD 设置均第一 | 50% 动作数据时 28.75% SR，高于 baseline 在 75% 数据下的最高 27.75% |
| 真实机器人 | 论文明确：没有部署真实车辆或机器人；驾驶离线、操作为仿真 | 不能外推为真实闭环能力 |
| 数据/代码开放 | Reproducibility Statement 称将公开 code、model weights、evaluation data | 这是作者声称的 release 计划；本文不列项目页或代码 URL，也不推断 baseline 的开放范围 |

需要特别区分：

- **已观测历史**：驾驶输入的历史 ego-state 最多 7 条，含当前，覆盖最多 3 s；这是条件输入。
- **未来预测**：6 个 waypoint 是监督目标，不是历史上下文，也不是记忆。
- **动作 chunk 16**：只属于该仿真操作的 Action DiT 设置，不是所有实验共有的时间窗口。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

| 能力 | 论文实测/主张 | 条件与失败模式 |
|---|---|---|
| 密集/小目标 | Dense200 74.53，Astra 65.04；VisDrone 40.44 | VisDrone 在 IoU 0.95 仅 3.31，极小目标严格边界仍是瓶颈 |
| Referring | HumanRef 88.56；RefCOCO family avg 83.92 | 论文也指出 DeepSeek-VL2-27B 在三个完整表聚合上领先，不能称全面统治 |
| 空间/GUI | RefSpatial avg 75.50；RoboSpatial Context 73.77；ScreenSpot-Pro 65.78；ScreenSpot-V2 96.15；OSWorld-G 74.82 | Astra 在 GUI 上仍显著更高（93.17 / 97.88 / 86.70）；专业界面仍有 frontier gap |
| OCR/layout | SROIE 72.47；M6Doc 74.82；DocLayNet 85.08 | TotalText 49.32，低于 Astra 53.55；OCR 催化作用是假设，未做因果隔离 |
| 操作迁移 | 六个操作评测中五个第一，四个 OOD 全第一；RoboTwin Full 66.20% | RoboCasa-GR1 Full 37.75% 低于 RynnBrain 39.00%；全量 ID 不是全面第一 |
| 数据效率 | RoboCasa-GR1 50% 数据 28.75%，高于 baseline 75% 数据最高 27.75% | 只测操作数据效率；未证明驾驶数据效率或更低预训练总成本 |
| 驾驶 | 开环 L2 0.296 m | 开环误差不等于闭环控制；论文未测闭环、未测真实车辆 |

### 6.1 隐含假设 (Hidden Assumptions)

- **grounding 提升会转化为动作提升**：论文用受控 Action DiT 降低混杂，但预训练骨干仍在规模、架构与监督上不同；不能把排名差全部归因于 grounding。
- **字段级融合标签足够可靠**：多教师一致性与任务验证提高质量，但论文承认多教师 agreement 不保证标注正确。
- **量化坐标足以支撑精确交互**：高 IoU 结果暴露边界精度限制；对极小目标与高精度插入任务，离散化误差可能放大。
- **OCR 是感知催化剂**：去除 OCR 组造成最大操作损失，但 OCR 单独迁移差；作者明确这是 hypothesis，未识别 OCR-only 因果效应。
- **Astra 比较可解释为感知前沿**：论文把 Astra 当前沿参考，而不是已验证的 VLA backbone；不能据此判断 Astra 直接适配动作会更差。
- **仿真迁移可外推**：没有真实机器人；真实接触、动力学与延迟未覆盖。

## 7. 与相关工作对比 (Comparison)

| 模型 | 角色/基础 | 架构与记忆属性（逐模型归属） | 论文中的证据 |
|---|---|---|---|
| GroundingPI | 4.844B grounding foundation | MoonViT-V2 + Qwen3-4B；36 层 full attention；单一视觉接口；无 sliding window；无 Gated DeltaNet | 34 benchmark Avg 73.68%；仿真操作 5/6 第一 |
| Qwen3-VL-4B | 通用 VLM baseline / 下游骨干 | 论文称 Qwen3-VL 使用 DeepStack，将 projected features 注入前三个语言层；4B | COCO 46.53；Dense200 14.02；驾驶 L2 0.301 m |
| Rex-Omni-3B | grounding model baseline | 基于 Qwen2.5-VL；强调 detection、referring、coordinate prediction；论文未给其记忆机制细节 | COCO 56.28；Dense200 53.29；RoboTwin OOD 第二 14.10% |
| LocateAnything-3B | grounding model baseline | 基于 Qwen2.5-VL；parallel box decoding（Hybrid/Slow 等变体） | RefCOCO Hybrid 81.37；Dense200 Hybrid 50.07 |
| RynnBrain-2B | embodied foundation baseline | Qwen3-VL + full-attention language layers + DeepStack | RoboCasa-GR1 Full 39.00%；OOD Type/Appearance/Container 第二 |
| RynnBrain1.1-2B | embodied foundation baseline | Qwen3.5：18 层 linear attention + 6 层 full attention + attention-output gating + DeepStack | 六个操作设置均落后 RynnBrain，但驾驶更好；论文只做条件性机制分析 |
| PaliGemma-3B | 通用 VLM baseline | π0 系列常用基座之一；论文未展开其记忆机制 | 作为操作 backbone 对照 |
| Wan2.2-TI2V-5B / Cosmos-Predict2.5-2B | 视频生成/WAM 类骨干 | 属于 WAM 组；固定特征提取状态后接入同一 Action DiT | 用于比较 VLA 与 WAM 类预训练 |

注意：RynnBrain 与 RynnBrain1.1 不是同一架构；GroundingPI 没有 Gated DeltaNet，也没有把 `262,144` 位置容量当作记忆窗口。论文对 RynnBrain1.1 的递归压缩与输出门控分析是**条件性解释**，不是已证明的架构因果。

**面试 Tip**：如果被问“GroundingPI 和 Rex-Omni/LocateAnything 差在哪”，答：它不只是刷 grounding 榜，而是把 grounding 模型当作可替换感知前端，在固定动作头与预算下比较多种 VLA/WAM 骨干；证据是仿真操作与开环驾驶，不是真实机器人闭环。

## 8. 精读建议 (Reading Guide)

- **值得精读原文的人**：做 VLA/WAM 感知前端的研究者；需要评估 grounding 预训练是否能提升仿真操作与 OOD 泛化的工程师；研究量化坐标接口与 token 效率的读者。
- **建议章节路径**：先读 §3.1–§3.3 与 §7.1–§7.6，理解接口、SFT 与 GRPO；再看 §5.2–§5.3 的下游与消融；最后读 §8.2 的受控 Action DiT 与 §10 的条件性机制分析。
- **可跳章节**：如果只关心方法，可略读 §12 的完整 benchmark 表；如果只关心落地，重点看 §5.2、§5.3.4 与 §11 Limitations。
- **不值得精读的理由**：如果你的目标是真实机器人闭环控制或低延迟部署，这篇没有真实机器人实验，也没有闭环延迟指标；读摘要与限制节即可。

---

[← Back to Theory](./README.md)

**关键引用**：论文链接 https://arxiv.org/abs/2609.39601（本文所有数字与主张均取该 PDF 正文；未使用候选截断摘要补写）。