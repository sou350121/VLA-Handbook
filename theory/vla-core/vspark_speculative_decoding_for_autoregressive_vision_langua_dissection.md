# VSpark：面向自回归视觉-语言-动作模型的投机解码 (VSpark: Speculative Decoding for Autoregressive Vision-Language-Action Models)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-11-19
>
> **论文**: VSpark: Speculative Decoding for Autoregressive Vision-Language-Action Models
> **链接**: https://arxiv.org/abs/2611.22666
> **核心定位**: 在不重训 target、不放宽验证的前提下，把自回归 VLA 的逐 token 动作生成加速 1.28×–2.10×。

> **来源与版本说明**：本文只依据 arXiv:2611.22666v1 的 HTML/PDF 正文（首页标注 [cs.CV] 17 Nov 2026）。候选元数据中的摘要为截断片段，且 source 标为 cs.RO；这些元数据未用于补写或替代正文主张。正文中的数字、表格编号均来自该版本 PDF/HTML。

## ⚡ 快速判断

| 维度 | 判断 |
|------|------|
| 核心结论 | 用 calibrated action noise rollouts、feature alignment、continuation reuse 三个设计，让 draft 更贴合 target，在四个 VLA 上实现 1.28×–2.10× 精确验证加速。 |
| 适合精读 | 如果你在做 VLA 推理延迟/控制频率优化，重点看 §3.2–3.4、§4.2 表 1–3、§4.3 表 4–5。 |
| 可以跳过 | 如果你只关心 VLA 表征学习或新任务泛化，这篇是推理加速插件，距离中等。 |
| 落地可行性 | 中高：draft 轻量、target 冻结，但需要白盒 hidden features 和树并行验证；真实机器人噪声 rollout 未验证。 |
| 主要风险 | G0.5 类 hybrid/linear attention 加速有限；continuation reuse 只在约一半拒绝中有效；BF16 下非 bitwise 一致。 |

💡 **X-Ray 开场**

自回归 VLA 每生成一个动作 token 就要跑一次大模型，控制频率被卡住。投机解码本可以“小模型先猜、大模型一次验”，但机器人动作里很多不同 token 对应几乎相同的运动，严格匹配会把“猜得差不多”的 token 全部拒掉。VSpark 的答案是：不放宽验证，而是让 draft 在连续特征空间更贴近 target，并把 target 在拒绝分支上已经算出的预测回收进下一轮候选树。

📍 **研究全景时间线**

```
2023  LLM 投机解码（Leviathan et al.; Chen et al.）
2024  EAGLE / EAGLE-2；OpenVLA 将动作逐维离散化
2025  EAGLE-3；FAST 用 DCT+BPE 压缩动作 chunk
2026  DSpark 并行 draft；VSpark（本文，arXiv:2611.22666v1，17 Nov 2026）
      └─ 当前位置：把投机解码从 LLM 迁到压缩 tokenizer 的自回归 VLA
      └─ 局限：真实机器人噪声 rollout、hybrid attention 并行验证仍未解
```

## 1. 核心架构/方法总览

### 1.1 系统对比概览

VSpark 不改变 target、action tokenizer 和验证规则，只改变 draft 的训练方式与每轮验证的候选构造。

| 组件 | 角色 | 输入 | 输出 | 训练/推理 | 时序/频率 |
|------|------|------|------|-----------|-----------|
| Target VLA | 被加速的自回归策略 | observation o、instruction ℓ、confirmed prefix | 动作 token 分布/预测 | 冻结，不重训 | 每个验证周期一次并行前向 |
| Draft | 轻量候选生成器 | 最后 confirmed token、mask embedding、target hidden features | 每个位置的 base logits + Markov 修正 | 训练 10 epochs；推理时每轮构建树 | 一层 draft，树 depth 7、60 节点 |
| Verifier | target 的并行树验证 | draft 树所有节点 | 接受路径 + correction token | 推理时执行 | 每轮验证一次 |
| Continuation reuse | 复用 rejected branch 上 target 已算出的预测 | 同一轮验证的 target 预测 | 下一轮 draft 树的候选 | 纯推理，无需训练 | 每轮验证后 |
| Action tokenizer | token 序列与连续动作互转 | 动作 token 序列 | 连续动作 chunk | 各 target 自带，冻结 | 解码完整序列后才得到动作 |

**逐模型归属**（不能把一组模型的不同属性合并给每个模型）：

| 模型 | 动作 tokenizer | target 规模 | VSpark draft 规模 | feature tap layers | 评测范围 |
|------|----------------|-------------|-------------------|--------------------|----------|
| π0-FAST | FAST（DCT+BPE 压缩 chunk） | 3B | 205M | 2, 5, 9, 13, 16 | LIBERO、LIBERO-Plus、LIBERO-Pro |
| StarVLA-FAST | FAST | 3B | 147M | 2, 10, 18, 26, 34 | LIBERO 系列用 LIBERO checkpoint；SimplerEnv 用 WidowX checkpoint |
| ActionCodec-AR | learned VQ + residual quantization | 2.2B | 119M | 2, 7, 12, 17, 22 | LIBERO、LIBERO-Plus、LIBERO-Pro |
| G0.5 | learned VQ + residual quantization；含 linear attention | 3B | 150M | 2, 7, 12, 17, 22 | LIBERO、LIBERO-Plus、LIBERO-Pro |

**数据开放范围**：论文表 A.3 列出了四个 target 的公开 checkpoint 标识符和五个 baseline 的仓库标识；正文未给出 VSpark 自身的代码仓库或权重链接，候选元数据的 repo_url 也为空。因此本文不把 VSpark 视为已确认开源。SimplerEnv 的 StarVLA-FAST 使用单独的 WidowX checkpoint，不能与 LIBERO checkpoint 混用。

**记忆机制**：VSpark 不引入跨 episode 或跨 chunk 的记忆模块。Continuation reuse 复用的是同一轮验证中 target 沿 rejected branch 已计算的预测，属于单次 decoding 内的临时候选缓存；它不改变 target 参数，也不把历史帧写入长期记忆。对每个 target，draft 只条件于当前 observation、instruction 和 confirmed prefix，不维护额外记忆。

### 1.2 关键机制

**为什么不能直接放宽验证。** 现有 VLA 投机解码方法多在 OpenVLA 的逐维 bin 上做“邻近即接受”。但 FAST、ActionCodec、G0.5 使用压缩 tokenizer：一个 token 携带多个维度/时间步的信息，必须解码完整 token 序列才能得到动作 chunk。逐 token 比较解码动作，等于要求 target 先生成剩余 token，投机收益被抵消。论文还给出两个反例：在 π0-FAST 和 ActionCodec 上，按 token embedding 距离排序与按解码动作距离排序分别有 30.62% 和 31.47% 的有效对不一致；π0-FAST 上 18.15% 的单 token 替换无法解码出有效系数序列（§4.3、Fig. 4）。在 π0-FAST 上用 embedding 距离放宽验证，速度从 1.60× 提到 1.90× 时，成功率从 91.7% 降到 71.4%（表 4）。

**三个设计选择。**

1. **Calibrated action noise rollouts**：DART 原本用“学习策略 vs 监督者”的差异校准噪声；VSpark 改用“draft vs target”的动作差异校准，在仿真中扰动 target 执行的动作，让 draft 见到更多偏离示范轨迹的状态。监督信号仍是 target 在扰动前产生的干净 token 和特征。
2. **Feature alignment**：draft 的输入拼接 target 多层 hidden states，并回归 target 输出头之前的特征。损失从 DSpark 的 TV 分布匹配改为 Smooth L1 特征回归，CE 项保留。这样“解码到几乎相同运动”的错误 token 不再与完全错误的 token 同等惩罚。
3. **Continuation reuse**：target 验证树时，即使某个候选被拒，它仍会沿该分支算出后续预测。VSpark 把这些预测提取为 continuation，在下一轮树中把对应 token 的分数提升到同父节点下最高 draft 分数，再交给 target 精确验证。该操作只改候选集合，不改接受规则。

⚡ **Eureka Moment**：机器人动作的“相似性”不应该在验证阶段用来放宽接受，而应该在 draft 阶段用来构造更可能被接受的候选；验证仍然 exact，lossless 的承诺不变。

**时序语义（避免把 token 数当成时间步）。** 论文的 observation o 是当前及可用的历史帧，action tokens 是未来动作 chunk。若以 Δt 均匀采样 K 个历史帧，首末跨度是 `(K-1)×Δt`；若从当前参考 t 预测 t+1 至 t+H，终点距当前参考是 `H×Δt`。VSpark 的 τ 是“每个验证周期提交的 token 数（含 correction 或 additional token）”，不是时间步数；动作 chunk 长度由各 tokenizer 的压缩率决定，不能把某一模型的 token 数或 chunk 长度写成所有实验共有的记忆窗口。连续未来 chunk 的预测区间也不能直接当作历史上下文。

### 1.3 信息流/架构图

```
训练阶段（target 冻结）
simulator states
   │
   ├─ target rollout，执行时注入 calibrated action noise
   │      └─ 每个 observation 存 target 的干净 token + hidden features
   │
   └─ draft 训练：CE(token) + SmoothL1(draft feature, target feature)
         输入 = 最后 confirmed token + mask embedding + target 多层特征

推理阶段
confirmed prefix
   │
   ├─ draft 一次并行出 base logits U_i
   │      └─ Markov head 用父 token 修正：softmax(U_i + W1[x_{i-1}]W2)
   │
   ├─ 构建 draft tree（depth 7，60 节点，每父 8 候选）
   │      └─ continuation reuse：把上一轮 rejected branch 的 target 预测提升为候选
   │
   ├─ target 一次并行验证整棵树（tree attention mask）
   │      └─ 只接受与 target 最高概率预测相同的 token
   │
   └─ 接受路径进入 confirmed prefix；correction token 补上
          └─ 被拒分支的 target 预测留给下一轮 continuation reuse
```

## 2. 数学核心

📌 **Napkin Formula**：

```
CE 保 token 精确，SmoothL1 把 draft 拉到 target 的连续特征空间；
τ 越大、每轮验证越便宜，speedup 越高——但 τ 是 token 数，不是时间步。
```

**目标**：给定 observation o 和 instruction ℓ，用更少的 target 前向次数生成与 target 贪心解码完全一致的动作 token 序列。

自回归分解：

```
p_Mt(x_{1:L} | o, ℓ) = ∏_{i=1}^{L} p_Mt(x_i | o, ℓ, x_{<i})
```

DSpark 的并行 draft 与 Markov 修正：

```
p_Md(· | o, ℓ, x_{<i}) = softmax(U_i + W1[x_{i-1}] W2)
```

其中 `U_i` 是一次双向并行前向得到的 base logits，`W1[x_{i-1}]` 嵌入前一个 token，`W2` 投影到词表 logits。这个修正让相邻 token 有依赖，但不需要重跑 backbone。

DSpark 的预测损失：

```
L_pred = Σ_i w_i ( λ·ℓ_i^CE + (1-λ)·ℓ_i^TV )
ℓ_i^CE = -log p_Md(x_i)
ℓ_i^TV = || p_Md(·) - p_Mt(·) ||_1
```

VSpark 把 TV 项替换为特征回归：

```
L = Σ_i w_i ( λ·ℓ_i^CE + (1-λ)·ℓ_i^feat )
ℓ_i^feat = SmoothL1(ĥ_i, h_i^t)
```

噪声协方差校准：

```
Σ = (1/N) Σ_{n=1}^{N} r_n r_n^T
```

`r_n` 是第 n 个有效样本上 draft 替换 target token 后解码动作与 target 动作的 6 维差异（平移+旋转）；`N` 是所有有效时间步残差数，不是输入样本数。噪声为零均值高斯，保留 6×6 全协方差，不减去残差均值。

Continuation 分数提升：

```
s̃_i(v) = max_u s_i(u),  若 v ∈ H_i
s̃_i(v) = s_i(v),        否则
```

`H_i` 是在该分支上由 continuation 建议的 token 集合。提升后，建议 token 与当前父节点下最高分 draft 候选并列；它仍要经过 target 精确验证。

指标：

```
τ = 每个验证周期提交的 token 数（含 correction 或 additional token）
speedup = T_AR / T_spec（同一设置下）
```

> 符号说明：`Mt` 为 target，`Md` 为 draft，`h_i^t` 为 target 预测 x_i 时的特征，`ĥ_i` 为 draft 对应特征。论文默认贪心解码与 exact verification；BF16 下的数值差异见 §6。

**直觉**：CE 项要求 token 完全一致，特征回归项则告诉 draft“即使 token 不同，也不要跑到另一个动作模式去”。Continuation reuse 利用的是动作序列的结构：被拒 token 与修正 token 可能编码相近运动，因此 target 在被拒 token 之后的预测，在修正 prefix 下仍有较大概率成立。

## 3. 带数字走一遍：玩具例子

> 以下均为**假设性玩具计算**，只用于验算公式和建立直觉，不是论文实验结论。

**玩具 1：噪声协方差。**

假设二维动作残差 `r1 = (1, 0)`，`r2 = (0, 2)`，N=2：

```
Σ = (1/2) · ( r1 r1^T + r2 r2^T )
  = (1/2) · ( [[1,0],[0,0]] + [[0,0],[0,4]] )
  = [[0.5, 0],
     [0,   2 ]]
```

于是平移方向标准差为 `sqrt(0.5) ≈ 0.707`，另一方向为 `sqrt(2) ≈ 1.414`。这演示了公式 4 保留“不同动作维度的误差结构”，而不是只用一个标量噪声。论文表 A.6 的实测尺度因 target 而异，例如 π0-FAST 的平移 σ 约为 3.88/7.10/3.08 mm，说明不能全模型共用一个噪声标量。

**玩具 2：continuation 分数提升。**

假设某父节点下 draft 分数为 `a: -0.2`、`b: -0.5`、`c: -1.0`，continuation 建议 token `d`：

```
max_u s_i(u) = -0.2
s̃(d) = -0.2   （与 a 并列）
s̃(a) = -0.2
s̃(b) = -0.5
s̃(c) = -1.0
```

验算：`d` 原本可能因分数低被剪掉，提升后进入候选集合；但它仍要在 target 验证时与 target 的最高概率预测逐 token 匹配，否则被拒。因此“提升”只影响候选构造，不影响 lossless 性质。

**玩具 3：τ 与 speedup 的算术关系（条件性分析）。**

假设 AR 每 token 耗时 `T_AR = 14.47 ms`，VSpark 每 token 耗时 `T_spec = 7.44 ms`（这两个数来自论文表 B.9 的 π0-FAST 实测），则：

```
speedup ≈ 14.47 / 7.44 ≈ 1.945
```

论文报告的平均 speedup 是 1.93×，差异来自表中数值的四舍五入和更精确的计时。这个例子说明：τ 提高会减少 target 前向次数，但每轮树验证、draft 构建、prefill 仍有固定开销；因此“τ 翻倍”不等于“速度翻倍”。论文的 G0.5 就是反例：τ 从 DSpark 的 2.33 提到 2.99，但 speedup 只从 1.05× 提到 1.28×，因为 linear attention 的 recurrent states 让并行树验证更贵。

## 4. 工程视角

**延迟分解（π0-FAST，LIBERO，表 B.9，单位 ms/output token）：**

| 方法 | Draft | Target | Prefill | Action | Other | Total |
|------|-------|--------|---------|--------|-------|-------|
| AR | — | 12.02 | 2.40 | 0.02 | 0.03 | 14.47 |
| DSpark | 1.58 | 4.23 | 2.41 | 0.02 | 0.70 | 8.93 |
| VSpark | 1.32 | 3.30 | 2.24 | 0.02 | 0.56 | 7.44 |

工程含义：

- 主要节省来自 target verification：12.02 → 4.23 → 3.30 ms。τ 提高后，每个输出 token 分摊到的 target 前向次数减少。
- draft 与树构建也从 1.58 → 1.32 ms，说明 continuation reuse 和 early stopping 没有把树构造拖慢；论文把 early stopping 的收益归因于减少树构造工作（表 B.2）。
- prefill 仍占 2.24 ms，是固定成本；在短 chunk、高控制频率场景下，prefill 占比会更高，加速比会被稀释。
- 不能直接用 7.44 ms/token 换算控制频率：一个动作 chunk 有多少 token 取决于 tokenizer。FAST 压缩整个 chunk，ActionCodec/G0.5 用学习 VQ，token 数与时间步不是一一对应。条件性估算应写成 `控制周期 ≈ token 数 × 每 token 时间 + action decode + 通信/执行开销`，其中 token 数逐模型取值。

**步数与并行度：**

- 树配置：最大深度 7、60 节点、每父 8 候选；draft 默认 1 层。表 6 显示 5 层 draft 通常提高 τ 但降低 speedup（π0-FAST 3.94→3.92，1.93×→1.45×），所以论文选一层。
- 树验证需要 target 支持带 tree attention mask 的并行前向。G0.5 的 linear attention 层有 recurrent states，并行树验证更复杂，这是其加速只有 1.28× 的主要原因之一（作者在结论中明确列为局限）。
- 内存：论文只给 draft 参数量 119M–205M 和 target 2.2B–3B，没有给显存占用；不能编造。树 60 节点是候选预算，不是 batch size。

**数值精度与“lossless”的边界：**

- 论文的 exact verification 指：在相同 target 计算下，只接受与 target 最高概率预测相同的 token。理论上贪心验证保持 target 的自回归输出。
- BF16 下，并行树验证与顺序 AR 的浮点运算顺序不同，leading logits 接近时可能选到不同 token。表 B.7：π0-FAST 的 AR/VSpark 平均 SR 为 92.71%/92.22%，episode outcome agreement 90.38%；ActionCodec 为 85.80%/86.11%，agreement 83.61%。作者把差异归因于有限精度，而不是放宽接受规则。
- FP32 下，seed 0 的 AR 与 VSpark 在每个 suite 的成功率相同，episode agreement 100%，生成的 action token 完全一致。这个结果只覆盖该 seed 和这两个 target，不能外推为所有硬件/所有模型都 bitwise 一致。
- 工程上应把“lossless”理解为“验证规则 exact”，而不是“任何精度下 bitwise 相同”。部署时要固定精度和 kernel，或做 FP32 对照。

**部署约束：**

- 需要白盒访问 target 的 hidden features 和 logits；黑盒 API 不可行。
- 需要能一次并行验证树；target 的 attention 实现要支持 tree mask。
- draft 是独立小模型，需要额外显存和一次前向；在 target 很大、draft 很小时收益更明显。
- 真实机器人上收集 calibrated noise rollouts 需要可重置环境或安全扰动策略；论文只在仿真中做噪声 rollout，真实机器人未验证。

## 5. 数据与评测

**预训练 / draft 训练（与评测分开）：**

| 项目 | 设置 |
|------|------|
| Target | 四个公开自回归 VLA，冻结；π0-FAST、StarVLA-FAST 用 FAST；ActionCodec-AR、G0.5 用 learned VQ |
| Draft 训练 | 每个 target 27,548 个 LIBERO rollout 样本；baseline 用无噪声 rollout，VSpark 用校准噪声 rollout；样本数和 optimizer updates 匹配 |
| SimplerEnv draft | StarVLA-FAST 在 26,413 个 BridgeData V2 rollout 上训练 |
| 优化 | 10 epochs，AdamW，batch size 16，peak lr 1e-4，β=(0.9,0.95)，无 weight decay，5% warmup + linear decay，grad clip 0.5，λ=0.1 |
| 校准 | 每个 target 64 个输入（每 suite 16 个）；每个序列最多 4 个均匀间隔起点，替换连续 7 个 token |
| 硬件 | 单 NVIDIA RTX 4090 |

**仿真评测（不是真实机器人）：**

| Benchmark | 设置 | Episode 数（每 target/method） |
|-----------|------|-------------------------------|
| LIBERO | 4 suites × 10 tasks × 50 initial states | 2,000 |
| LIBERO-Plus | 每个 variation case 跑一次；Goal/Spatial/Object/Long 分别 2,591/2,402/2,518/2,519 | 10,030 |
| LIBERO-Pro | 4 suites × 5 variants × 10 tasks × 50 initial states | 10,000 |
| SimplerEnv (WidowX) | 4 tasks × 24 episodes；StarVLA-FAST 用 WidowX checkpoint | 96 |

**真实机器人设置**：论文没有真实机器人部署实验。SimplerEnv 是仿真基准，虽然 StarVLA-FAST 在 BridgeData V2 真实数据上训练，但评测仍在仿真中。作者在结论中明确说：在没有模拟器的情况下在真实机器人上收集噪声 rollout 需要进一步研究。

**指标与统计口径：**

- τ：平均每个验证周期提交的 token 数，含 correction/additional token；不是时间步。
- speedup：同一设置下 target 自回归解码时间 / 投机解码时间。
- SR：任务成功率；生成错误 episode 在成功率中计为失败，但在效率指标中排除。
- 生成错误率（表 B.8）：VSpark 在 LIBERO/LIBERO-Plus/LIBERO-Pro 分别为 0.28%/1.62%/1.17%，AR 为 0.35%/1.65%/1.14%。
- BF16 用 5 个 seed（0–4）报告均值与 95% CI；其他 LIBERO 表用 simulator seed 7。
- 时间分解用 CUDA events，每第 10 个 action chunk 记录一次（§B.9）。

**主要结果（论文实测）：**

| 模型 | LIBERO 平均 τ / speedup | LIBERO-Plus/Pro 平均 τ / speedup | 备注 |
|------|--------------------------|-----------------------------------|------|
| π0-FAST | 3.94 / 1.93× | 3.56 / 1.96× | DSpark 为 3.16 / 1.60× |
| StarVLA-FAST | 2.72 / 1.61× | 2.43 / 1.60× | SimplerEnv 3.90 / 1.95× |
| ActionCodec-AR | 4.21 / 2.10× | 3.73 / 2.04× | DSpark 为 3.69 / 1.93× |
| G0.5 | 2.99 / 1.28× | 2.76 / 1.29× | 多个 baseline 慢于 AR |

## 6. 能力与失败模式

**能做什么：**

- 在四个不同 tokenizer 的自回归 VLA 上，用 exact verification 获得 1.28×–2.10× 加速；π0-FAST 和 ActionCodec 的增益最稳定。
- 不改 target、不改 tokenizer、不改接受规则；continuation reuse 是纯推理操作，不需要重训。
- 在 LIBERO-Plus/Pro 的分布偏移下保持相对 DSpark 的优势；τ 会下降，但排序不变。
- 在 SimplerEnv 的 WidowX 实施例上，StarVLA-FAST 仍有 1.95× 平均加速，说明不是单一 embodiment 的偶然。

**不能做什么 / 具体失败模式：**

- **G0.5 类 hybrid attention**：linear attention 的 recurrent states 让并行树验证变贵，draft 接受长度提高但 speedup 只有 1.28×；三个 baseline 甚至慢于 AR（表 1）。
- **Continuation reuse 依赖“拒绝后 continuation 不变”**：表 5 中 target 在被拒 token 后与修正后预测同一 next token 的比例为 45.38%（π0-FAST）和 54.88%（ActionCodec），约一半；换 tokenizer 会变。语言模型上该机制没有增益（表 B.11，Qwen3-4B/Gemma-4-12B-it 的 τ 变化 ≤0.043，speedup 不变或略低），作者推测是动作序列结构特殊。
- **真实机器人噪声 rollout 未验证**：论文的 calibrated noise 在仿真中注入；真实机器人没有模拟器重置和安全扰动，收集成本与风险未解。
- **BF16 下非 bitwise 一致**：并行树验证与顺序 AR 的浮点顺序不同，leading logits 接近时可能改变 token；FP32 下该 seed 一致，但不能保证所有硬件。
- **依赖白盒 target**：需要 hidden features、logits 和 tree-mask 注意力；黑盒部署不可行。
- **压缩 tokenizer 不能逐 token 放宽验证**：必须等完整序列解码动作；这是论文反对 relaxed acceptance 的核心理由，也是其方法与 OpenVLA 类方法的关键区别。

### 6.1 隐含假设

1. **Draft 能在 target 特征空间对齐**：feature regression 假设 target 的 hidden feature 是动作相似性的可靠连续表示；论文用 Smooth L1 回归，但没有证明所有层/所有 tokenizer 都同样有效。表 B.1 显示 regression 单独带来主要增益（3.23→3.67），conditioning 单独几乎不变（3.23→3.27），说明“给特征”不等于“会用特征”。
2. **仿真噪声能覆盖真实 draft 误差**：校准噪声来自 draft-target 动作差异，假设仿真中的扰动状态分布与部署时遇到的偏差足够接近；真实机器人上的分布偏移未验证。
3. **贪心解码**：论文默认 greedy decoding 和 exact verification；如果 target 用采样解码，验证规则和 lossless 论证都要改。
4. **树验证的并行效率**：假设 target 的并行树前向显著便宜于多次顺序前向；G0.5 说明这个假设对 hybrid attention 不成立。
5. **动作 token 的相似性结构**：continuation reuse 假设被拒 token 与修正 token 编码相近运动，因此后续预测仍有用；语言模型上该假设不成立，说明它是动作序列的特有结构，而不是通用投机解码技巧。

## 7. 与相关工作对比

| 方法 | draft 类型 | 验证规则 | 是否 lossless | 支持压缩 tokenizer | 关键差异 |
|------|-----------|----------|---------------|--------------------|----------|
| Spec-VLA | EAGLE-2 树 | 按动作 bin 距离放宽 | 否 | 主要面向 OpenVLA 逐维 bin | 需要 per-dimension 邻近性 |
| KERV | EAGLE-2 + Kalman | 放宽 + 运动学容差 | 否 | 主要面向逐维 bin | Kalman 预测剩余维度；论文表 B.6 显示其误差常大于 token 替换引起的动作差异 |
| HeiSD | 检索 draft + 模型 draft | 放宽/跳过验证 | 否 | 主要面向逐维 bin | 按近期运动切换 draft |
| EAGLE-3 | 自回归单层 draft | exact | 是 | 可适配，但需额外 draft pass 扩树 | 表 1 中 τ 不低但 speedup 受自回归 draft 限制 |
| P-EAGLE / DFlash / DFlash2 | 并行 draft | exact | 是 | 可适配 | 需额外 pass 扩树；VSpark 复用 base logits + Markov 修正 |
| DSpark | 并行 draft + Markov | exact | 是 | 是 | VSpark 的基座；VSpark 增加噪声 rollout、特征回归、continuation reuse |
| VSpark | DSpark + 特征对齐 + continuation reuse | exact | 是（规则 exact；BF16 有数值差异） | 是 | 不放宽验证，把动作相似性用于 draft 训练和候选构造 |

**面试 Tip**：被问到“VSpark 和 Spec-VLA 的根本区别”时，一句话答：“Spec-VLA 在验证阶段用动作 bin 距离放宽接受，VSpark 保持 exact verification，把动作相似性前移到 draft：用 target 特征回归训练 draft，并把被拒分支上 target 已算出的 continuation 回收进下一轮树。”

## 8. 精读建议

- **值得精读原文的人**：
  - 做 VLA 推理加速、控制频率优化、投机解码落地的研究者/工程师；
  - 需要在不重训 target 的前提下降低自回归动作生成延迟的团队；
  - 想理解“压缩 tokenizer 为什么不能逐 token 放宽验证”的读者。
- **建议章节路径**：先读 §3.2–3.4（三个设计的具体做法）→ 再看 §4.2 表 1–3（四模型/四基准的主结果）→ 然后 §4.3 表 4–5（放宽验证的反例、continuation 一致率）→ 最后 §B.7（BF16/FP32 精确性边界）。如果只关心工程收益，可跳 §2 文献综述和 §B.11 语言模型实验。
- **不值得精读的理由**：如果你不做机器人学习、不关心推理延迟，或已经熟悉 EAGLE/DSpark 系列，读摘要和表 1 即可；论文的机制增量主要在 draft 训练和候选复用，不在新的 VLA 架构。

---
[← Back to Theory](./README.md)

**关键引用**：VSpark 论文，arXiv:2611.22666v1，https://arxiv.org/html/2611.22666