# 让 LLM 直接改机器人策略权重：词法 grounding 的稀疏修复 (Can LLMs Debug Robots by Editing Their Weights?)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-11-11
>
> **论文**: Can LLMs Debug Robots by Editing Their Weights?
> **链接**: https://arxiv.org/abs/2611.09376
> **核心定位**: 把“权重空间编程”搬到机器人基础策略：LLM 只改 π0.5 的少量 token-embedding 行，修复指令改写下的词法 grounding，而不是学习新技能。
> **证据版本**: arXiv:2611.09376v1（2026-11-06）HTML 全文，对应 PDF 正文。来源记录中的 `candidate.abstract_snippet` 是截断片段，止于“48”，未用于补写；正文摘要在完整度和措辞上不同，见下。

**摘要差异与证据边界**

- 候选摘要片段只写到“48”，正文摘要完整给出：48 repairs、4 regressions、exact `p=1.31×10^{-10}`。
- 候选片段称“765 held-out paraphrases”，正文更精确地称“765 held-out Language Instructions variants”；这不是同义替换，涉及评测总体定义。
- 正文还包含 16 个编辑会话、12/13 拒绝、FastWAM 指令 metadata 敏感性等候选片段没有的内容。
- 来源记录的 `abstract_verification` 为 null；本文不把候选摘要当作独立证据，也不以它补写正文主张。以下数字均以 arXiv:2611.09376v1 正文的表格与附录为准。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 在 π0.5 上，LLM 编辑 65 行 token embedding，把 765 个 held-out 语言变体的成功率从 86.3% 提到 92.0%；修复 48 个、回归 4 个；含被编辑 token 的 335 个指令从 74.3% 提到 87.5%，其余 430 个结果不变。 |
| 适合精读 | 做 VLA 指令 grounding、权重空间编辑、机器人评测严谨性的人，重点看 §3.2–3.5、§4、§5。 |
| 可以跳过 | 只关心真实机器人部署、新技能学习、通用 VLA 架构的人，这篇距离中等偏大。 |
| 落地可行性 | 低到中：仿真内可复现，编辑只改权重值；但需要权重访问、simulator feedback，真实机器人未验证。 |
| 主要风险 | 只修词法映射，不补视觉/恢复/新任务；full replacement 控制已解释大部分增益；共享 held-out 与 seed 7，独立性只在“搜索”层面。 |

💡 **X-Ray 开场**

这篇论文问三个问题：机器人明明会转炉灶，为什么换个说法就抓盘子？LLM 能不能不训练、不新增示范，直接改策略权重来修这个错？这种修复能推广到别的失败吗？它的答案是：在 π0.5 上，失败更像是“词到任务表示的 lookup 错位”，不是动作能力缺失；改 65 行 embedding 就能把词接回已有技能。但视觉扰动、恢复、换位置/换任务并没有被可靠修好，12/13 个其他会话最终拒绝提交编辑。对 VLA 研究者的意义是：某些 instruction-following 失败可以在表示层做稀疏、可检查的修复；但证据只到仿真和两个 released checkpoints。

📍 **研究全景时间线**

```
2023 LIBERO 基准
  → 2025 π0.5 开放 VLA checkpoint
  → 2026 权重空间编程（Takehi et al.，LLM 直接读写权重）
  → 2026-11 本文：把权重空间编程用于机器人策略的词法 grounding 修复
     当前位置：强结果在 π0.5 仿真；局限：真实机器人、通用能力、独立环境覆盖未证
```

## 1. 核心架构/方法总览

### 1.1 系统对比概览

| 对象/角色 | 架构归属（逐模型） | 输入 | 输出/时序 | 训练/推理 | 数据开放范围 |
|---|---|---|---|---|---|
| π0.5（被编辑的 VLA） | SigLIP 视觉编码 + PaliGemma 风格 VLM prefix + 较小的 flow-matching action expert；导出 3.62B 参数 | 当前/已观测图像 `V(o)`、指令 token、固定 flow noise `z` | 未来 action chunk：horizon 10，denoising 10 步，执行 5 步后重规划 | 已预训练/微调；编辑不做梯度更新、不新增示范 | 使用 openpi 的 `pi05_libero` released checkpoint；编辑配方随 artifact 发布 |
| FastWAM（被调查的 WAM） | video expert 宽 3072、action expert 宽 1024；各 30 blocks、24 heads；冻结 umT5-XXL 文本编码；action-only inference | 指令、观测、proprioception | 动作；本文不把未来视频帧当作记忆输入 | 已训练 checkpoint；编辑改 text-conditioning MLP 与 Q/K | 论文称有归档候选与结果记录；未给出公开仓库链接 |
| Qwen3.5-0.8B p50c（语言模型复现） | 语言模型，不是机器人策略 | MATH-500 题目 | 文本答案 | 复现 Takehi et al. 的 released edit；非本文机器人编辑器新发现 | 论文只报告复现协议与截断率 |
| GPT-6 Astra（编辑器） | 通用 LLM 编辑器；读 tensor 名/形状、原始权重、tokenizer；持久 CPU Python workspace；独立 forward tool | 权重、练习 rollout、replay probe | 替换 tensor 值；最终 retain/revert | 禁止梯度更新、训练循环、loss fitting、在 kernel 内执行目标网络 | 16 个会话 transcript 与配对记录随 artifact 发布 |

> 架构归属必须分开：π0.5 是 VLM prefix + flow-matching action expert；FastWAM 是 video/action 双专家 WAM；Qwen 是语言模型。不能把 FastWAM 的双专家属性写给 π0.5，也不能把 π0.5 的 action horizon 写成 FastWAM 的默认设置。

### 1.2 关键机制

- **编辑接口**：Astra 可读 tensor 名、形状、原始权重和 tokenizer；可在 CPU Python kernel 中做静态线性代数并写 replacement tensors。协议禁止梯度更新、训练循环、迭代 loss fitting，也禁止在 kernel 内执行目标网络。目标网络以 `gradients disabled` 方式评测。
- **只改权重值**：tensor 名、形状、输入指令、action scaling、推理代码都固定。这意味着它不是插入新模块，而是在已有模块内替换数值。
- **独立 forward tool**：跑 simulator episode 或 replay 保存的模型输入，返回观测、动作、activation。所谓“无新 demonstrations”不等于“无观测/反馈”：策略本来就在示范上预训练/微调过，编辑器仍使用 simulator 反馈。
- **诊断路径**：π0.5 在 LIBERO-Goal 的改写指令上常选错任务/物体；三个会话都判断这是 content 问题，不是“完全不看语言”。动作专家可以 attend 到指令，却仍执行错任务表示。
- **编辑 recipe**：保护 canonical 指令中出现的 token ID，把选定的 source-token embedding 朝 task anchor（如 stove、bowl、wine）重映射。提交的语言编辑不改 vision、attention、action-expert tensor。
- **验证协议**：练习集与 held-out 按 suite/variant ID 的 SHA-256 parity 分开；候选在 paired held-out 评测前冻结；held-out 结果不回传给编辑器。统计用 paired `(g-l)/n`、normal interval 和 exact two-sided McNemar。

⚡ **Eureka Moment**：这不是“让机器人学会新技能”，而是“把已有技能接到正确的词上”；65 行 token embedding 是词法路由器，不是能力注入器。

### 1.3 信息流/架构图

```
instruction c
  → tokenizer → I(c)
  → embedding lookup E[I(c)]
  → VLM prefix → flow-matching action expert
  → future action chunk: horizon 10, denoise 10, execute 5

编辑只发生在 lookup 表：
E'[i] = (1-α_i)E[i] + α_i * centroid(T_i)   for i in S
若 I(c) ∩ S = ∅，则 E'[I(c)] = E[I(c)]

Editor loop:
read weights → propose rows → forward tool (practice rollouts / replays)
→ retain or revert → freeze candidate → paired held-out eval（结果不回传）
```

> 时序口径：`H=10` 是未来动作 chunk，不是历史帧，也不是记忆窗口；`execute 5` 是执行长度，不是上下文长度。论文没有描述跨 episode 的外部记忆模块。若记录 K 个均匀采样帧、间隔 Δt，首末跨度是 `(K-1)×Δt`；若从当前参考 t 预测 t+1 至 t+H，终点距当前参考是 `H×Δt`。连续未来 chunk 的预测区间不能直接当作历史上下文或记忆窗口。

## 2. 数学核心

📌 **Napkin Formula**：

```
E'[i] = (1-α_i) E[i] + α_i * mean(E[j] for j in T_i)
```

目标：在只改权重值、不改张量名/形状/指令/动作缩放/推理代码的约束下，提高 held-out 成功率，并让未触及指令保持不变。

**词法局部性**：

```
a = F_ϑ(V(o), E[I(c)], z)

若 I(c) ∩ S = ∅：
E'[I(c)] = E[I(c)]
⇒ 在确定性执行下，a' = a
```

**配对统计**：

```
Δ = (g - l) / n
g = 原失败变成功的数量
l = 原成功变失败的数量
n = 配对 episode 数

近似区间：
100 × [Δ ± 1.96 × sqrt(((g+l)/n - Δ²) / n)]
```

显著性用 exact two-sided McNemar test，条件在 `g+l` 个 discordance 上；normal interval 可能排除 0 而 exact test 不显著，正文以 exact p 为准。

**机制诊断 ratio**：

```
ρ = median_i ( ||ã_i - a_i^anchor||₂ / ||a_i - a_i^anchor||₂ )
```

| 符号 | 含义 |
|---|---|
| `E[i]` | 原 token embedding 行 |
| `T_i` | 目标 anchor token 集合；多目标时取 centroid |
| `α_i` | 插值强度，C 中范围 0.5–1.0 |
| `S` | 被编辑 row ID 集合 |
| `I(c)` | 指令 c 经固定 tokenizer 得到的 token ID |
| `g/l` | repairs/regressions |
| `ρ` | 编辑后动作 chunk 到 anchor 动作的距离，相对原动作到 anchor 的距离 |

直觉：embedding 行是 lookup 表的一行；只改被读到的行，未读到的行在固定输入下不会改变动作。`α` 控制“保留多少原词义、推向多少 anchor”；centroid 用于歧义词，例如 dairy 不被强行只推向 cheese。局部性是结构性的，不是“语义上无关”的假设。

## 3. 带数字走一遍：玩具例子

> 以下为**假设验算**，不是论文实测；用论文表 S8 的 cosine 数值做近似演示。

假设 source 与 destination 都是单位向量，`activate → turn` 的编辑前 cosine 为论文实测的 0.909，取 `α=0.75`：

```
E_s · E_t = 0.909
E' = 0.25 E_s + 0.75 E_t

|E'|² = 0.25² + 0.75² + 2×0.25×0.75×0.909
      = 0.0625 + 0.5625 + 0.340875
      = 0.965875
|E'| = 0.98279

cos(E', E_t) = (0.25×0.909 + 0.75) / 0.98279
             = 0.97725 / 0.98279
             ≈ 0.9944
```

论文实测 `activate → turn` 从 0.909 提到 0.992；上述假设算得 0.9944，差异来自单位向量假设与舍入。若 `α=1.0`，如 `cooker → stove`，则完全替换为目标行，cosine 变为 1.000；论文实测从 0.927 到 1.000。这个例子说明 `α` 不是“学习率”，而是“保留原词义的比例”。

再看局部性：C 的 430 个未含编辑 token 的 held-out 指令，论文实测 `g/l=0/0`，411 个成功保持不变。固定输入下这是结构保证；闭环中仍可能因渲染/模拟器差异出现个别 discordance，所以论文区分 replay invariance 与 rollout identity。

## 4. 工程视角

| 工程项 | 论文给出的约束/数字 | 含义 |
|---|---|---|
| 编辑规模 | C 改 65 行、132,768 个 scalar entries，占 3.62B 参数的 0.00367%；A/B 分别 36,764/40,822 entries | “65 行”不等于 65 个 scalar，也不等于 65 行 Python；行内可能因舍入保持相等 |
| 搜索预算 | 每个语言会话 50 个 forward requests；CPU 执行上限 900 秒、96 GB | request 是 batch，不是单个 NN forward，也不等于一个 episode |
| 部署边界 | 只改 weight values；tensor 名/形状、指令、action scaling、推理代码固定 | 适合做“权重补丁”，但需要权重访问与可回滚机制 |
| 动作时序 | π0.5：horizon 10、denoising 10、execute 5；论文未给控制频率 | 不能把 10 或 5 写成所有实验共有的记忆窗口；真实控制频率未知 |
| 失败处理 | 2 个 full-replacement、3 个 random-control runtime errors 在主张量中计失败；附录给 valid-pair 敏感性 | 错误处理约定不改变定性结论，但会改变小数点 |
| 闭环抖动 | 同一未触及 prompt 的 paired rollout 可出现 1–16 个 channel entry 的初始图像差异 | 固定输入动作相同，不等于闭环 episode 完全相同 |

**时序口径补充**：若以 Δt 均匀采样 K 帧，首末跨度是 `(K-1)×Δt`；若从当前参考 t 预测 t+1 至 t+H，终点距当前参考是 `H×Δt`。这两个式子只用于避免把“未来动作 chunk”误读成“历史上下文”。论文的 π0.5 输出是未来 action chunk；FastWAM 在本文是 action-only inference；Qwen 没有机器人帧。没有证据支持把某个单一设置的步长或 chunk 长度推广为所有实验共有的记忆窗口。

## 5. 数据与评测

### 5.1 预训练与 checkpoint

- π0.5 使用 openpi 的 `pi05_libero` released checkpoint；导出 3.62B 参数；已在 demonstrations 上预训练/微调。编辑器**不做**额外策略训练、不新增 demonstrations。
- FastWAM 是另一个 released checkpoint；本文调查其 suffix/instruction metadata 敏感性，并尝试编辑 text MLP 与 Q/K。没有保留一个显著 held-out 修复。
- Qwen3.5-0.8B p50c 是语言模型编辑复现，不是机器人实验；用于检查“LLM 读写权重”这一前置能力是否可复现。

### 5.2 仿真评测

- 基准：LIBERO、LIBERO-Plus、LIBERO-PRO、LIBERO-RECOVER；全部机器人实验在仿真中，使用两个 released checkpoints。
- 语言任务：4,986 个 practice variants；held-out 语言总体 765 个 Language Instructions variants；另有 360 个其他六类样本；更早还有 200 个 held-out 语言样本，未支持显著增益。
- 配对条件：同一 variant、seed 7、initial state 0；original 与 edited 权重在同一 episode 上比较；编辑后恢复权重并校验。
- 切分：practice/held-out 按 SHA-256 parity；held-out 列表不暴露给编辑器；候选在评测前冻结，结果不回传。
- 独立性：三个 π0.5 语言会话是**独立搜索**，共享同一 checkpoint、同一 held-out 总体、同一 seed；不是独立环境样本。FastWAM 的三个 follow-up 继承第一个会话的 notes 和候选，不是独立再发现。

**主结果（论文 Table 1）**

| Run | Rows | Original | Edited | g/l | Exact p |
|---|---:|---:|---:|---:|---:|
| A | 18 | 86.3 | 89.4 | 32/8 | 1.82×10^-4 |
| B | 20 | 86.3 | 88.0 | 18/5 | .0106 |
| C | 65 | 86.3 | 92.0 | 48/4 | 1.31×10^-10 |

**局部性（论文 Table 2）**

| Run | 含编辑 token 的 n | Orig→Edit | g/l | 未触及 n | g/l |
|---|---:|---:|---:|---:|---:|
| A | 216 | 67.6→78.7 | 32/8 | 549 | 0/0 |
| B | 125 | 52.8→63.2 | 18/5 | 640 | 0/0 |
| C | 335 | 74.3→87.5 | 48/4 | 430 | 0/0 |

C 的 335 个受影响指令中，48 个原失败被修复（55.8%），paired 95% interval `[9.16, 17.11] pp`；430 个未触及指令结果不变。C 的 44 个净修复中，level 5 贡献 32 个；level 5 从 104/174 到 136/174，即 59.8%→78.2%。这不是均匀提升，而是集中在“描述替代任务名”的困难改写上。

**控制（论文 Table 3）**

| 同 65 个 source rows | Success | g/l | Exact p |
|---|---:|---:|---:|
| No edit | 86.3 | – | – |
| Random direction, matched norm | 85.2 | 7/15 | .134 |
| Full destination replacement | 90.5 | 43/11 | 1.40×10^-5 |
| Editor C | 92.0 | 48/4 | 1.31×10^-10 |

Full replacement 使用 C 自己发现的 source–destination 对，已经解释了大部分增益；C 相对 full replacement 的 20 wins/9 losses，exact `p=.0614`，在 0.05 下不显著。论文明确说：主贡献是“从 rollout、probe、weights 中诊断并构造词典”，不是证明 partial/blended move 的额外收益。

**16 个会话的最终决定（论文 Table 6 摘要）**

| 设置 | 会话数 | 最终保留 | 关键 held-out 结果 |
|---|---:|---:|---|
| π0.5 paraphrases A/B/C | 3 | 3 | 全部显著；86.3→89.4/88.0/92.0 |
| FastWAM suffix 第一次 | 1 | 0 | 47.7→51.3，p=.00187，但被撤回/拒绝 |
| FastWAM suffix follow-up C | 1 | 1 | 48.0→48.7，p=.542，不显著 |
| FastWAM camera/background/language | 3 | 0 | p=.230/1.000/.791 |
| π0.5 failure recovery A/B/C | 3 | 0 | 无保留编辑 |
| LIBERO-PRO A/B/C | 3 | 0 | A 的 1,484 对 58.5→60.0，p=.0448，但被撤回；position 28.0→27.3 |
| 合计 | 16 | 4 | 12/13 非语言会话拒绝候选 |

### 5.3 真实机器人

论文没有真实机器人实验。所有机器人 rollout 都在仿真中；没有 sim-to-real、没有物理机器人部署、没有真实相机/延迟/执行器噪声证据。因此“可落地”只能理解为“仿真内可复现的权重编辑流程”，不是“已可在真实机器人上安全部署”。

### 5.4 数据开放范围

论文称 reproducibility package 包含：harness 与 domain code、16 个会话 transcript、逐 episode 配对结果（含 runtime errors）、sparse 65-row edit 与应用脚本、figure sources、provenance hashes。公开 code 与 weight-hosting 链接“将随 artifact release 提供”；论文正文没有给出仓库 URL。本文不杜撰代码仓库，也不把“将提供”写成“已公开”。

### 5.5 独立的指令 metadata 发现

在同一 210 个 matched LIBERO-Plus episodes 上：

| 模型 | task-only 指令 | 带 metadata 的 tagged 指令 | 配对 |
|---|---:|---:|---|
| FastWAM | 70.0% | 48.6% | 47 gains/2 losses，p=4.36×10^-12 |
| π0.5 | 88.6% | 88.6% | 9 gains/9 losses，p=1 |

这说明 benchmark 的 instruction-construction procedure 会强烈影响 FastWAM，但不改变 π0.5 的 aggregate rate；后者不是逐轨迹不变，仍有 9/9 的 paired 变化。论文强调：这是两个 checkpoint 的证据，不是“所有 WAM 对 all VLA”的普遍结论。

## 6. 能力与失败模式

| 能做 | 证据 | 不能做/未证 |
|---|---|---|
| 修复 π0.5 的词法 grounding | 3 个独立搜索均显著；C 65 行，86.3→92.0；受影响子集 74.3→87.5 | 不能证明通用语言泛化；共享 held-out 与 seed 7 |
| 保持未触及指令不变 | C 的 430 个未触及指令 0/0；固定输入局部性 | 闭环渲染差异可造成未触及 prompt 的个别 discordance |
| 用控制区分方向与编辑大小 | random-direction 不显著；full replacement 显著 | C 相对 full replacement 不显著，词典发现成本未被控制 |
| 拒绝缺乏练习支持的编辑 | 12/13 非语言会话拒绝；作者引用编辑器的原话 | 不是 calibrated failure classifier；拒绝率不是准确率 |
| 暴露 benchmark metadata 敏感性 | FastWAM 70.0→48.6；π0.5 88.6 不变 | 只测两个 checkpoint；不能推广到所有 WAM/VLA |

**失败模式与边界**

- **只修“词法可达性”**：视觉扰动、物体掉落、恢复状态、换位置/换任务没有保留的、独立确认的修复。PRO A 的 aggregate `p=.0448` 是名义正效应，但 position 子组从 28.0% 降到 27.3%，且编辑被撤回。
- **歧义描述仍会错**：C 的四个回归包括“圆形食物容器放到上层储物区”等；词法线索单独熟悉，不等于完整指令选中正确任务。
- **full replacement 已解释大部分增益**：C 的点估计更高、回归更少，但与 full replacement 的直接差异不显著；不能把“编辑器插值”当成已证明的独立贡献。
- **闭环 vs 固定输入**：同一 saved input 的 original/edited actions 相同；separately rendered paired episodes 仍可因初始图像 1–16 个 channel entry 差异而不同。不能把未触及 prompt 的单个 discordance 直接归因于权重编辑。
- **搜索独立性有限**：三个 π0.5 会话独立搜索，但共享 held-out population 和 seed；FastWAM follow-up 还继承 notes/candidate。独立的是“搜索轨迹”，不是“环境覆盖”。

### 6.1 隐含假设

1. **失败主要是词法 lookup 错位**：论文的强结果支持这一点，但不能反推所有 instruction-following 失败都如此。
2. **tokenizer 与 token ID 稳定**：局部性分区依赖固定 tokenization；换 tokenizer 或指令预处理，受影响集合会变。
3. **anchor 语义可定义**：stove、bowl、wine 等 anchor 是任务相关的；歧义词用 centroid，但 centroid 的语义合法性未被独立验证。
4. **simulator feedback 可代表目标分布**：练习集来自仿真；held-out 也不回传。真实世界的分布偏移未测。
5. **retain/revert 决定可信**：编辑器的最终决定与其 practice validation 一致，但论文没有把它校准成故障分类器。
6. **固定输入不变性可分离**：replay 能固定观测与 flow noise；rollout 仍含模拟器/渲染变化。论文承认这个区别，但闭环结果仍是主要行为证据。
7. **“65 行”不是普适 recipe**：A/B/C 分别是 18/20/65 行；C 的 row 数、α、destination 都是该次搜索的产物，不能写成所有实验共有的记忆窗口或通用编辑预算。

## 7. 与相关工作对比

| 方法/线索 | 对象 | 更新方式 | 数据/反馈 | 与本文关系 |
|---|---|---|---|---|
| 本文 | π0.5 / FastWAM 的权重 | 稀疏 token embedding 行；FastWAM 试改 text MLP/QK | simulator practice + replay；无梯度 | 机器人词法 grounding 的权重空间修复 |
| Takehi et al. 权重空间编程 | 游戏、语言模型 | LLM 直接读写权重 | 工具与实验日志 | 本文的直接灵感；附录复现 Qwen 编辑 |
| ROME / MEMIT / MEND | LLM 关联知识 | 参数更新/编辑 | 监督式编辑数据 | 相关但非机器人；本文未直接比较 |
| LoRA | 通用网络适配 | 训练低秩更新 | 梯度训练 | 论文明确说未证明优于 LoRA；不是已测基线 |
| Prompt rewriting | 推理时指令 | 改输入文本 | 无需权重 | 论文固定部署指令，未做对比；不能声称本文更优 |
| LEACE / ITI | 表示干预 | 线性擦除/推理时干预 | 表示统计 | 相关机制；非机器人词法修复 |
| LIBERO-Plus / PRO / RECOVER | 评测基准 | 不改模型 | 扰动/恢复/位置任务 | 本文用它们区分 nominal success、鲁棒性与恢复 |

**面试 Tip**：被问到时一句话答——“这篇不是让 LLM 学会机器人技能，而是用 65 行 token embedding 的稀疏权重编辑修 π0.5 的词法 grounding；强结果是仿真里 86.3→92.0，弱结果是 12/13 其他失败拒绝提交、FastWAM 不显著，所以它是‘精确词法修复’，不是‘通用机器人调试’。”

## 8. 精读建议

- **值得精读原文的人**：做 VLA 指令 grounding / 词法鲁棒性的研究者；想做权重空间编辑、模型修补、可解释调试的工程师；需要学习 paired evaluation、McNemar、replay vs rollout 区分的评测人员。
- **建议章节路径**：先读 §2 的编辑接口与统计协议 → §3.2–3.5 的主结果、局部性与控制 → §4 的 16 会话拒绝记录 → §5 的 metadata 发现。若已熟悉 LIBERO 与 VLA 架构，可跳过 §6 相关工作。
- **不值得精读的理由**：如果你只关心真实机器人部署、新技能学习、或通用 VLA 架构，这篇的证据范围不够；读摘要与 §6 失败模式即可。

---
[← Back to Theory](./README.md)

**关键引用**：论文链接 https://arxiv.org/abs/2611.09376（本文所有数字、表格与协议均出自该论文的 arXiv:2611.09376v1 正文；未使用候选摘要补写，未引用论文之外的 URL）。