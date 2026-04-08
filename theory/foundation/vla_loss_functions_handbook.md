# VLA Loss Functions Handbook（VLM-Robot Policy 训练目标实务手册）

> 目标：这是一份可以直接用于**设计 / 复盘 / debug VLA（Vision-Language-Action）与 VTLA（Vision-Tactile-Language-Action）训练目标**的工程手册。  
> 覆盖：BC、sequence/chunk、GMM、Tokenization、Diffusion/Flow、RL fine-tune、Perception/Grounding、Safety regularization、Representation alignment、Co-training loss masking。

> **数学前置知识**：在深入研究 Loss 之前，建议先阅读 [VLA 数学必备：从直觉到实作](./math_for_vla.md)。

---

## 0. 使用方式（如何快速查表）

### 0.1 你要解决的是什么问题？

- **动作学不会/学得慢**：先看 [1. BC/回归类](#1-行为克隆behavior-cloning-bc--回归类损失) 与 [2. 序列chunk](#2-序列与chunk动作损失sequencechunk)
- **动作发散/抖动/不平滑**：看 [8. 安全与可控性正则](#8-安全与可控性正则safety--controllability-regularization)
- **多解任务被“平均化”**：看 [4. GMM](#4-gmm-动作分布高斯混合的负对数似然) 与 [5. Diffusion](#5-diffusion-policy-条件去噪损失)
- **推理太慢（Hz 上不去）**：看 [6. Flow/Flow Matching](#6-flow-matchingflow-policy-向量场损失) 与 [2. chunk](#2-序列与chunk动作损失sequencechunk)
- **加入触觉/新数据后“忘记动作”**：看 [10. 蒸馏与遗忘抑制](#10-蒸馏与遗忘抑制distillation--anti-forgetting) 与 [9. 联合训练与loss-masking](#9-联合训练与loss-maskingco-training--loss-masking)
- **语言理解还行但执行不对（grounding 失败）**：看 [7. Perception/Grounding](#7-perceptiongrounding-感知与语义落地损失)
- **RL 微调不稳/奖励稀疏**：看 [11. RL Fine-tune](#11-rl-fine-tune-策略梯度与离线微调损失)

---

## 0.2 统一记号：输入/输出张量（最常用）

### 观测与条件（conditioning）

- **图像**：

  $$
  I \in \mathbb{R}^{B\times T_{\text{obs}}\times C\times H\times W}
  $$

- **语言 token**：

  $$
  x^{\text{text}} \in \mathbb{N}^{B\times L}
  $$

  （注意 mask：`m_text` 为 0/1）
- **触觉（图像型）**：

  $$
  I^{\text{tac}} \in \mathbb{R}^{B\times T_{\text{tac}}\times C_t\times H_t\times W_t}
  $$

- **本体/状态（proprio）**：

  $$
  s \in \mathbb{R}^{B\times T_{\text{obs}}\times D_s}
  $$


### 动作（action）

- **连续动作序列**：

  $$
  a \in \mathbb{R}^{B\times T_{\text{act}}\times D_a}
  $$

- **动作 chunk（一次预测一段）**：

  $$
  a_{t:t+K-1} \in \mathbb{R}^{B\times K\times D_a}
  $$

- **离散 token 动作**：

  $$
  y \in \mathbb{N}^{B\times T_{\text{act}}\times D_a}
  $$

  （每一维动作被离散化成 bins）

### 重要：训练/推理频率不匹配（teleop/VLA 常见）

- 输入观测频率可能是 10-30Hz（视觉/触觉）
- 控制环频率可能是 125-500Hz（RTDE/servoJ）
- 常见工程做法：模型输出低频 **chunk**，控制端做插值/滤波补成高频命令流

---

## 0.3 VLA Pipeline 挂载位置（你要写 loss 的“插槽”）

```text
Sensors -> Encoders -> Fusion -> Policy Head -> Action -> Robot
             |            |         |
             |            |         +-- action losses (BC/GMM/Diffusion/Flow/RL)
             |            +------------ alignment / grounding losses
             +------------------------- perception/self-supervised losses
```

更细的实践对齐：

- **Encoder 级**：InfoNCE/CLIP 对齐、深度/姿态监督、MAE/contrastive
- **Fusion 级**：跨模态对齐（vision↔tactile↔text）、adapter/LoRA 训练策略
- **Policy/Action Head 级**：BC/Token CE/GMM NLL/Diffusion eps loss/Flow matching loss/RL loss
- **输出后处理**：动作平滑正则、约束/安全 barrier、contact/torque 限制

---

## 1. 行为克隆（Behavior Cloning, BC）= 回归类损失

> 用途：最常见、最稳健的 VLA 基线。  
> 适用：有大量示教轨迹（teleop），目标是“像专家一样做”。

### 1.1 L2（MSE）回归：最小二乘

**定义**

$$
\mathcal{L}_{\text{MSE}}=\frac{1}{N}\sum_{i}\|a_i-\hat a_i\|_2^2
$$

**张量**

- `pred_action`: `[B, T, D_a]`
- `gt_action`: `[B, T, D_a]`

**超参**

- 动作维度加权：对位置/旋转/夹爪开合给不同权重（避免某一维支配梯度）

**优缺点**

- 优点：实现简单、收敛快、可作为所有复杂方法的 warm-start
- 缺点：多模态任务会“取平均”（绕障左右两解 → 撞中间）

**最小 PyTorch**

```python
import torch
import torch.nn.functional as F

loss = F.mse_loss(pred_action, gt_action)
```

**挂载位置**

- policy/action head 输出连续动作（Δpose / joint targets / velocities）

---

### 1.2 L1 / Huber：抗异常、抗偶发噪声

$$
\mathcal{L}_{\text{L1}}=\frac{1}{N}\sum_i |a_i-\hat a_i|
$$

Huber（$\delta$ 为阈值）：

$$
\mathcal{L}_{\text{Huber}}=
\begin{cases}
\frac{1}{2}e^2,& |e|\le \delta\\
\delta(|e|-\frac{1}{2}\delta),& \text{otherwise}
\end{cases}
$$

**什么时候用**

- teleop 数据偶尔跳点（丢帧/重投影）  
- 触觉信号偶发尖峰

**最小 PyTorch**

```python
loss_l1 = torch.abs(pred_action - gt_action).mean()
loss_huber = F.smooth_l1_loss(pred_action, gt_action, beta=1.0)
```

---

### 1.3 高斯 NLL（学不确定性）：heteroscedastic regression

如果模型输出均值 $\mu$ 与方差 $\sigma^2$（或 log-variance），则：

$$
\mathcal{L}_{\text{NLL}}=\frac{1}{2}\sum_i\left(\frac{(a_i-\mu_i)^2}{\sigma_i^2} + \log\sigma_i^2\right)
$$

**输入输出**

- `mu`: `[B, T, D_a]`
- `log_var`: `[B, T, D_a]`（数值更稳定）

**常见超参**

- clamp `log_var` 范围（防止发散）

**优缺点**

- 优点：能表达“这一步我不确定”，利于安全/异常检测
- 缺点：不处理多模态（仍可能平均），且方差容易被滥用（学成“全都不确定”）

**最小 PyTorch**

```python
import torch

var = torch.exp(log_var).clamp_min(1e-6)
loss = 0.5 * (((gt_action - mu) ** 2) / var + log_var).mean()
```

---

## 2. 序列与 chunk 动作损失（sequence/chunk）

> 目的：解决“模型输出频率低、控制频率高”的不匹配；并减少误差累积。

### 2.1 Chunking BC：一次预测 K 步

**定义**：对 chunk 维度做 BC：

$$
\mathcal{L}_{\text{chunk}}=\frac{1}{BKD_a}\sum\|a_{t:t+K-1}-\hat a_{t:t+K-1}\|
$$

**关键超参**

- `K`：chunk 长度（K 越大越能“看远”，但越难学、越怕分布漂移）
- `receding horizon`：推理只执行前 M 步（M < K），滚动重规划

**挂载位置**

- action head 输出 `[B, K, D_a]`

---

### 2.2 时间加权（越近越重要）

$$
\mathcal{L}=\sum_{k=0}^{K-1} w_k \cdot \ell(a_{t+k},\hat a_{t+k}),\quad w_0\ge w_1\ge\cdots
$$

**为什么**

- 远期标签更不可靠（示教噪声、动力学误差）
- 强化“短期可控性”

---

## 3. 离散动作（Tokenization）= Cross Entropy

> 代表：RT-1/RT-2 风格。把每个动作维度离散成 bins → 预测分类。

### 3.1 Cross Entropy（每维 bins 分类）

$$
\mathcal{L}_{\text{CE}} = -\sum \log p_\theta(y)
$$

**张量**

- `logits`: `[B, T, D_a, N_bins]`
- `target_bins`: `[B, T, D_a]`

**超参**

- `N_bins`：常见 256
- label smoothing：缓解过拟合

**优缺点**

- 优点：天然多模态（能在分布上表达多解）
- 缺点：bins 跳变导致抖动；精密任务受量化误差影响

**最小 PyTorch**

```python
import torch.nn.functional as F

loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),
    target_bins.view(-1),
)
```

---

## 4. GMM：动作分布（高斯混合）的负对数似然

> 用途：解决 BC “平均化”，但又不想上 diffusion。  
> 输出：混合权重 \(\pi_k\)、均值 \(\mu_k\)、方差 \(\Sigma_k\)。

### 4.1 GMM NLL

$$
p(a)=\sum_{k=1}^{K}\pi_k \mathcal{N}(a;\mu_k,\Sigma_k),\quad 
\mathcal{L}_{\text{GMM}}=-\log p(a)
$$

**张量（常见对角协方差）**

- `logits_pi`: `[B, T, K]`
- `mu`: `[B, T, K, D_a]`
- `log_std`: `[B, T, K, D_a]`

**超参**

- `K`：混合数（2~10 常见）
- std clamp：防止数值爆炸

**优缺点**

- 优点：能表达多解；推理快（采样/取最大分量）
- 缺点：高维动作下训练不稳；组件塌陷（只用一个高斯）

---

## 5. Diffusion Policy：条件去噪损失

> 核心：预测噪声 \(\epsilon\)（或直接预测 \(x_0\)/v-parameterization），训练用 MSE。  
> 适用：高精度、多模态轨迹；代价：推理步数多（可用 DDIM/少步采样缓解）。

### 5.1 经典 $\epsilon$-prediction MSE

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,\quad
\mathcal{L}_{\text{diff}}=\mathbb{E}\|\epsilon-\epsilon_\theta(x_t,t,c)\|^2
$$

**张量**

- `x0`（动作 chunk）：`[B, K, D_a]`
- `t`：`[B]`（随机时间步）
- `eps`：`[B, K, D_a]`
- `eps_pred`：`[B, K, D_a]`

**超参**

- 噪声 schedule（cosine/linear）
- 采样步数 T（训练）与推理步数 N（推理）

**挂载位置**

- action head 是去噪网络（U-Net/Transformer/DiT）

---

## 6. Flow Matching/Flow Policy：向量场损失

> 目标：用更少步数逼近 diffusion 的质量（利于 50Hz+）。  
> 核心：学习向量场 \(v_\theta(x,t,c)\)，用 MSE 拟合目标速度。

### 6.1 Flow Matching（基本形式）

$$
\mathcal{L}_{\text{flow}}=\mathbb{E}\|v_\theta(x_t,t,c)-v^\ast(x_t,t)\|^2
$$

> 注意：不同论文/实现的 \(v^\ast\) 定义不同（OT-CFM、rectified flow 等）。工程上把它当作“对 diffusion 的少步替代”即可。

---

## 7. Perception/Grounding：感知与语义落地损失

> 当你发现“语言理解 OK，但执行错对象/错位置”，通常是 grounding 的监督不够。

### 7.1 CLIP/InfoNCE 对齐（图文/触觉-视觉）

$$
\mathcal{L}_{\text{InfoNCE}}=
-\log \frac{\exp(\text{sim}(q,k^+)/\tau)}{\sum_j \exp(\text{sim}(q,k_j)/\tau)}
$$

**张量**

- `img_emb`: `[B, D]`
- `txt_emb`: `[B, D]`
- `tau`：温度

**常见用途**

- 图文对齐：提升指令理解与对象检索
- 视触觉对齐：让触觉 token 能“语义化”

---

## 8. 安全与可控性正则（Safety & Controllability Regularization）

> 目标：减少抖动、超力、碰撞，提升可部署性。

### 8.1 动作平滑（速度/加速度/jerk 正则）

$$
\mathcal{L}_{\text{smooth}}=
\lambda_v\sum_t\|a_t-a_{t-1}\|^2+
\lambda_a\sum_t\|(a_t-a_{t-1})-(a_{t-1}-a_{t-2})\|^2
$$

**适用**

- teleop / VTLA 接触阶段（抖动直接导致滑移/卡死）

---

### 8.2 约束/安全 barrier（软约束）

$$
\mathcal{L}_{\text{barrier}}=\sum_i \lambda_i \cdot \max(0, g_i(a,s))^2
$$

例：关节限位、速度限位、力矩限位、TCP workspace。

---

## 9. 联合训练与 Loss Masking（Co-training & Loss Masking）

> 核心：不同数据源标签不同（web data 没动作），要 mask 掉不该回传的梯度。  
> 参考：[`co_training.md`](./co_training.md)。

**最小原则**

- robot batch：算 action loss（+可选 text loss）
- web batch：只算 text loss，action head 梯度为 0

---

## 10. 蒸馏与遗忘抑制（Distillation & Anti-forgetting）

> 典型场景：加入 tactile、新任务或 RL fine-tune 后，“旧动作能力掉了”。

### 10.1 行为蒸馏（policy distillation / KL）

离散动作：

$$
\mathcal{L}_{\text{KL}}=\text{KL}(p_{\text{teacher}}\|p_{\text{student}})
$$

连续动作（回归蒸馏）：

$$
\mathcal{L}_{\text{distill}}=\|a_{\text{student}}-a_{\text{teacher}}\|
$$

**工程要点**

- 蒸馏 loss 通常在“旧数据回放 batch”上启用（replay）
- 配合 adapter/LoRA，减少对动作头的破坏

---

## 11. RL Fine-tune：策略梯度与离线微调损失

> 注意：VLA 里 RL 往往是“少量微调”，不是从零开始。

### 11.1 PPO（最常见）

$$
\mathcal{L}_{\text{PPO}}=
-\mathbb{E}\left[\min\left(r_t(\theta)\hat A_t,\ \text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t\right)\right]
$$

并加价值函数与熵正则：

$$
\mathcal{L}=\mathcal{L}_{\text{PPO}}+\lambda_v \mathcal{L}_{V}-\lambda_H H(\pi)
$$

**VLA 适配要点**

- RL 只动“小模块”（adapter/最后几层），保留 base policy 的动作分布
- 加 KL penalty（对齐旧策略），防止策略崩坏

---

## 12. 代码：最小可用 PyTorch snippets

配套文件：[`loss_functions_snippets.py`](./loss_functions_snippets.py)（可直接复制到工程中）。

---

## 13. 推荐的组合配方（工程上最常用的 6 套）

1. **BC 基线**：MSE + action smooth  
2. **多解任务**：GMM NLL + smooth  
3. **高精度轨迹**：Diffusion eps loss + chunk RHC  
4. **高频部署**：Flow matching + chunk + smooth  
5. **加入触觉不忘动作**：BC + replay + distill(KL/MSE) + gating  
6. **RL 微调**：BC warmstart + PPO + KL-to-base + safety barrier

---

[← Back to Theory](./README.md)


