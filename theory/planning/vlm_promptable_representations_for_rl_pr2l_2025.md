# VLM Promptable Representations：用“可提示表征”给 RL 注入常识 (PR2L, 2024/2025)

> **发布时间**：2024 arXiv（`arXiv:2402.02651`），2025-03 TMLR 录用发表  
> **论文题目**：Vision-Language Models Provide Promptable Representations for Reinforcement Learning  
> **核心定位**：不让 VLM 直接“出动作”（容易不接地气），而是让 VLM 通过 **task-relevant prompt** 把互联网常识与语义线索“抽取成状态表征”（promptable representations），再用 RL/BC 把这些表征落地为低层动作。核心结论：同一个 VLM 的纯图像 embedding 不如“prompt 后的 embedding”；在 Habitat ObjectNav 上加入 CoT prompt，成功率从 **27.8%→41.9%（1.5×）**。

这篇对 VLA 很关键：它把“VLM 的推理/常识”变成 **可插拔的 state representation**，非常适合解释为什么“上层语义（System2）”不必直接接管动作，而是可以通过表征层影响 System1/S0。

**核心来源**：
- arXiv：`https://arxiv.org/abs/2402.02651`
- TMLR（OpenReview）：`https://openreview.net/forum?id=vQDKYYuqWA`
- 项目页：`https://pr2l.github.io/`

---

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 方案 | VLM 角色 | 输入给 RL 的是什么 | 典型问题 | PR2L 的选择 |
|---|---|---|---|---|
| **让 VLM 直接出动作**（RT-2 style / 直接问“该做什么”） | planner / actor | “动作文本”或动作 token | 低层动作不接地气、容易幻觉、难稳定优化 | PR2L 认为这条路在低层控制上很脆 |
| **把 VLM 当 encoder**（不 prompt） | feature extractor | 任务无关的视觉 embedding | 用不上 VLM 的世界知识/常识推理 | PR2L 认为“浪费了 VLM” |
| **PR2L（本文）** | **promptable state representation** | 对“prompt + 图像 + 生成 token”的 **内部 embedding** | 需要设计 prompt，且推理慢 | 通过 prompt 把常识变成可学习状态；动作仍由 RL 学 |

### 1.2 关键机制 (Key Mechanism)

核心动作是：对每个 observation \(I_t\)，用 task-relevant prompt \(c\) 去问 VLM：

- 生成文本 \(x_{1:K}\)（可能不完全正确）
- **但我们丢掉文本，只取 embedding**（promptable representations）
- 把这些 embedding 当作 RL policy 的输入状态

### 1.3 信息流/架构图 (Flow / Diagram)

```text
env obs I_t
   │
   ├─ prompt c (task-relevant questions, optional CoT)
   │
   v
Generative VLM: sample x_1:K ~ p(x|I_t, c)
   │
   ├─ decoded text: discard
   └─ token embeddings (prompt + image + decoded tokens): keep
            │ (variable length)
            v
Policy-side Transformer (+ CLS token) -> summary vector
            v
RL policy π(a | summary, proprio, prev_action, ...)
```

---

## 2. 数学核心：PR2L 的“promptable representation”是什么？(Math Core)

### 2.1 VLM 的形式化（论文用来解释“我们取哪一层 embedding”）

生成式 VLM 定义：

```text
x_1:K ~ p(x_1:K | I, c)
```

Transformer VLM 在每个时间步会产生 token-wise 表征：

- \(\phi_t(I,c,x_{1:t-1})\)：表示“在生成第 t 个 token 前”的内部 embedding（多层 self-attn 输出）
- 传统做法只用最终输出文本或仅用 image encoder embedding
- PR2L 取 **后几层 token embeddings**，作为“可提示的语义状态”

### 2.2 policy 侧怎么把“变长 token”变成可用状态？

因为 token 数量不固定，PR2L 在 policy 里加了一个小 Transformer，并用 CLS summarization：

```text
summary = TransformerEncoderDecoder( tokens, CLS ).CLS_embedding
```

然后再接 policy head（PPO/BC/CQL 等）。

---

## 3. 带数字走一遍：prompt 真的改变了 representation 吗？(Worked Example)

### 3.1 Minecraft：同一个图像，prompt 让 embedding 出现“可分结构”

论文对 Minecraft 任务做 PCA 分析：PR2L 的表征往往呈 **双峰结构**，其中高价值 transition 聚到某一簇（例如 VLM 判定“目标实体在画面中”时）。直觉上，这等价于把稀疏奖励任务的 credit assignment 变得更简单。

### 3.2 Habitat：用 CoT prompt 把“房间类型”这类抽象变成状态

ObjectNav 的核心常识是“厕所通常在浴室、床通常在卧室、沙发通常在客厅”。prompt：

```text
Would a [target object] be found here? Why or why not?
```

后半句诱导 CoT，让 VLM 把“可泛化的语义特征”写入 embedding（即使目标物体并不在视野里）。

---

## 4. 工程视角：怎么把它落到 VLA 系统里 (Engineering View)

### 4.1 PR2L 对 VLA 的系统意义

你可以把 PR2L 看成一个“上层语义注入接口”：

- System2（慢）：用 prompt 问 VLM，让它把常识/语义推理“折叠进 embedding”
- System1（快）：RL policy 直接消费 embedding 做决策
- System0（更快）：执行与接触闭环仍然由控制器/反射层兜底

### 4.2 Prompt 设计不是“拍脑袋”：论文给了一个可复述的 prompt 评估法

Minecraft 的 prompt 选择用一个代理指标：
- 给少量标注帧，测 VLM 在 decoded text 上的 True Positive / True Negative（实体是否在画面）
- 或者对 embedding 做 probing（用小模型/线性分类器看看某语义特征是否可读出）

经验结论（很工程）：**给蜘蛛加辅助文本有用，但给牛/羊加“常识性废话”会导致模型几乎总回答“有”**，进而伤害下游 RL。

### 4.3 计算与延迟

论文指出 VLM 推理会让 policy 频率降到约 **3–5 Hz**（与一些大模型机器人策略同量级）。因此：
- 在线 RL 可能受限（慢、贵）
- 离线/BC 更友好（可并行预处理 embedding，训练可加速）

---

## 5. 数据与评测 (Data & Eval)

### 5.1 Minecraft（在线 RL）

- 环境：MineDojo 程序化任务（combat spider / milk cow / shear sheep / 多种 combat）
- RL：PPO
- 结论（表 2）：PR2L 在所有任务上都优于非 oracle baselines（VLM image encoder / RT-2-style / Dreamer / VC-1 / R3M）

（他们用的是 IQM successes + 标准误差；你面试不必背每个数字，但要能背“PR2L 普遍赢 baseline；oracle 模型在部分任务更强”。）

### 5.2 Habitat ObjectNav（离线 BC，泛化到未见场景）

核心数字（表 3，validation scenes）：

- PR2L **with CoT**：Average **41.9%**
- PR2L **without CoT**：Average **27.8%**
- 其它 baseline（VLM image encoder / VC-1 等）明显更低（例如 VLM image encoder 11.6%）

一句话：**CoT prompt 不直接出动作，但能显著提升状态表征的可泛化性。**

### 5.3 TMLR 版本信息

- OpenReview（TMLR）页面显示：Published 2025-03-31  
  `https://openreview.net/forum?id=vQDKYYuqWA`

---

## 6. 能力与失败模式 (Capabilities & Failure Modes)

### 6.1 能力
- **把 VLM 的常识/语义推理“变成状态”**，而不是把 VLM 当成 action generator
- **可控性强**：policy 仍由 RL 学，VLM 只提供特征（降低“语言幻觉直接驱动动作”的风险）
- **CoT 作为表征增强**：CoT 的价值不在文本本身，而在它迫使 embedding 编码更丰富的语义关系

### 6.2 失败模式（面试可用）
- **prompt 选错 → representation 变坏**（尤其“常识废话”会让 VLM 输出偏置）
- **VLM domain gap**：Minecraft 的 stylized entity（如 enderman/pigman）可能更难从自然图像预训练迁移
- **推理慢**：在线 RL/高频控制受限；需要系统级缓存/离线 embedding/异步流水线

---

## 7. 与相关工作对比 (Comparison)

| 方向 | 代表 | PR2L 的差异 |
|---|---|---|
| VLM 直接指导动作 | RT-2-style prompting、LM planners | PR2L 不让 VLM 出动作，而让它“出可学习状态” |
| 纯视觉表征预训练 | VC-1 / R3M | PR2L 认为“prompt”能把世界知识对齐到任务相关特征 |
| ECoT（具身思维链） | `embodied_chain_of_thought_robotic_control_2024.md` | PR2L 偏“状态表征层注入”，ECoT 偏“policy 内部显式推理链” |

**面试 Tip（一句话）**：被问“VLM 对 RL/VLA 的价值到底是什么？”——答：“PR2L 的观点是：VLM 最值钱的是可索引的世界知识和语义推理，但它不擅长直接出低层动作；因此用 task-relevant prompt 把知识折叠进 embedding，再用 RL 去做 grounding，会比‘直接问 VLM 该怎么动’更稳、更可控。”

---

[← Back to Theory](../README.md)

