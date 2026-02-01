# 众擎机器人：具身 VLA 算法面经（注意力/Transformer 手撕/BN vs LN/Norm 取舍）(2025)

> **来源**：用户粘贴面经原文  
> **日期**：2025-09-20  
> **轮次**：一面（技术面，投递后一周）→ 二面（主管面，2 天约面）  
> **特点**：问题围绕项目深挖，且对 Transformer/Norm 细节问得较深

---

## 0. 总览：这家在筛什么？

从题目结构看，他们重点筛：
- **你是否真的理解 Transformer**（attention 公式、手撕 forward）
- **你是否懂训练稳定性与归一化**（BN/LN、不同 norm 的优缺点与选择）
- **你能否把项目决策讲清楚**：你怎么做、为什么这么做、还有什么替代方案
- **技术路线判断**：端到端 VLA + RL 是否理解其难点与落地边界

---

## 1. 一面：技术面（项目八股，灵活且深入）

### 1.1 题目清单（原文还原）

- 注意力机制公式
- 手撕 Transformer：输入一个 embedding，怎么过一层 Transformer
- 不同归一化方式的优缺点，一般选择什么归一化方式，为什么
- 详细说说 BN 和 LN
- 反问技术路线：回答“端到端的 VLA + RL”

### 1.2 标准答法：Attention 公式（写对符号 + 讲清复杂度）

Scaled Dot-Product Attention：

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

其中（以单头为例）：
- $Q = XW_Q$, $K = XW_K$, $V = XW_V$
- $X\in\mathbb{R}^{T\times d}$，$d_k$ 是每头维度

多头注意力（MHA）：
- 每头独立算 attention，然后 concat 再乘 $W_O$

面试加分点：
- **为什么要除以 $\sqrt{d_k}$**：避免点积随维度增大导致 softmax 饱和、梯度变小
- **复杂度**：注意力矩阵是 $T\times T$，时间/显存 $O(T^2)$

### 1.3 手撕 Transformer（一层）：给出“最小可执行 forward”

面试里最稳的“手撕结构”是按模块写：

1) 线性投影：$Q,K,V$  
2) attention：softmax 权重 + 加权求和  
3) 残差 + Norm  
4) FFN（两层 MLP）  
5) 残差 + Norm

伪代码（单层，Pre-LN 版本更常见）：

```python
def transformer_block(x):
    # x: [T, d_model]
    h = layer_norm(x)
    q = h @ Wq
    k = h @ Wk
    v = h @ Wv
    att = softmax((q @ k.T) / sqrt(d_k)) @ v
    x = x + att @ Wo              # residual

    h = layer_norm(x)
    x = x + ffn(h)                # residual
    return x
```

补一句避免歧义：如果面试官真的限定“输入只有一个 embedding（单 token）”，那么 $T=1$ 时 $QK^\top$ 只有一个标量，softmax 权重恒为 1，self-attention 退化成对该 token 的线性变换（本质上输出就是 $V$ 经输出投影后再走残差/FFN）。

如果对方追问“Post-LN 呢？”：
- Post-LN：把 layer_norm 放在 residual 后面（Transformer 原论文）
- 现代大模型通常更偏 Pre-LN（更稳、深层更好训）

### 1.4 归一化方式怎么选？（先给结论，再给理由）

面试里最安全的结论：
- **大多数 Transformer / VLM / VLA**：默认选 **LayerNorm（LN）**（或 RMSNorm）
- **CNN / batch 足够大、分布稳定**：BN 更常见

原因要点：
- **BN** 依赖 batch 统计量（均值/方差），对 batch size、小 batch、多卡不同步、时序/自回归等场景更敏感
- **LN** 在样本内按 feature 维归一化，不依赖 batch 维统计量，更适配 Transformer（尤其是自回归/小 batch）

### 1.5 详细说 BN vs LN（把“统计维度”讲清楚）

**BatchNorm（BN）**
- 归一化维度：对每个通道，用 batch（以及空间维）统计量
- 优点：在 CNN 上经验极强、收敛快、一定正则效果
- 缺点：
  - 小 batch 时统计不稳定
  - train/infer 行为不一致（running mean/var）
  - 分布式训练要考虑 sync BN

**LayerNorm（LN）**
- 归一化维度：对每个 token/样本，在 feature 维上归一化
- 优点：与 batch size 无关，更适合 Transformer、序列模型
- 缺点：在某些 CNN 场景未必如 BN（但现代视觉 Transformer 也普遍用 LN/RMSNorm）

可顺带提一句 RMSNorm（加分但别展开过度）：
- 不减均值，只除 RMS，计算更省，很多 LLM 用它替代 LN

---

## 2. 二面：主管面（“你怎么做/为什么/还知道什么替代方案”）

### 2.1 题目风格（原文总结）

主管面常见三连问：
- 你是怎么做的？
- 为什么这么做？
- 实现了什么？
- 你还了解什么做法？

你可以提前准备每个项目的“三件套”：
- **决策点**：当时有哪些选项？为什么选这个？
- **指标**：成功率/吞吐/延迟/稳定性/泛化/OOD
- **替代方案**：2–3 个（并说明 trade-off）

### 2.2 反问（原文还原）

- 激励机制
- 工作时间
- demo 状况

---

## 3. “端到端 VLA + RL”路线：反问时如何答得更像懂系统

如果你说“端到端 VLA + RL”，主管往往会追问：
- **为什么需要 RL**：BC 不够在哪里？（误差累积、恢复策略、长时程、稀疏奖励）
- **怎么做安全与稳定**：离线 RL、限制动作、MPC/规则兜底、仿真到真机
- **数据闭环**：失败数据怎么收集、怎么标注、怎么回放与复训

一句话版本：
> “端到端并不排斥分层：语义/感知可端到端，控制侧仍需要实时约束与安全兜底；RL 主要用于补齐恢复与长时程，而不是把所有风险交给策略探索。”

---

[← Back to Question Bank](./README.md)

