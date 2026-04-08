# 联合训练 (Co-training)

> **定义**: 在训练 VLA 模型时，同时混合 **机器人动作数据 (Robot Action Data)** 和 **互联网视觉语言数据 (Internet Vision-Language Data)**。

```
┌─────────────────────────────────────────────────────────────────┐
│                   Co-training 数据混合策略                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐       ┌──────────────────┐               │
│  │   机器人数据      │       │   互联网数据      │               │
│  │   (Robot Data)   │       │   (Web Data)     │               │
│  │                  │       │                  │               │
│  │  📷 + 🎯 + 🦾    │       │  📷 + 📝         │               │
│  │  图像 指令 动作   │       │  图像 文本        │               │
│  └────────┬─────────┘       └────────┬─────────┘               │
│           │                          │                          │
│           │    混合比例 1:1          │                          │
│           └──────────┬───────────────┘                          │
│                      ▼                                          │
│           ┌──────────────────────┐                              │
│           │      VLA 模型        │                              │
│           │   ┌──────┬──────┐   │                              │
│           │   │Action│ Text │   │                              │
│           │   │ Head │ Head │   │                              │
│           │   └──┬───┴──┬───┘   │                              │
│           └──────┼──────┼───────┘                              │
│                  │      │                                       │
│           ┌──────┴──┐ ┌─┴──────┐                               │
│           │ Action  │ │  Text  │                               │
│           │  Loss   │ │  Loss  │                               │
│           │ (机器人)│ │ (互联网)│                               │
│           └─────────┘ └────────┘                               │
│                                                                 │
│   效果: 保持语义能力 ✓  学习动作控制 ✓  防止灾难性遗忘 ✓         │
└─────────────────────────────────────────────────────────────────┘
```

## 1. 为什么需要 Co-training?

### 1.1. 防止灾难性遗忘 (Catastrophic Forgetting)
VLA 模型通常基于预训练的 VLM (如 LLaVA, PaLI) 微调。如果只用机器人数据 (通常只有简单的指令如 "Pick apple") 训练，模型会迅速忘记通用的视觉语义知识。
- **现象**: 模型能抓苹果，但认不出"苹果"是"水果"，或者认不出未见过的物体。
- **后果**: 丧失了 VLM 最宝贵的通用常识 (Common Sense)。

### 1.2. 保持通用泛化能力 (Generalization)
互联网数据包含了丰富的物体、场景和概念，Co-training 能让机器人利用这些知识处理未见过的指令 (Zero-shot)。
- **举例**: 训练数据里只有 "Pick up the apple"，但用户指令是 "Pick up the red fruit"。如果模型保留了 VLM 的知识，它就能理解 "red fruit" 指的是 apple。

## 2. 实施策略 (Implementation)

### 2.1. 数据配比 (Mixing Ratio)
通常采用 **1:1** 或 **1:X** 的比例混合。
- **RT-2**: 机器人数据 : 互联网数据 = **1 : 1** (Batch 内部混合)。
- **OpenVLA**: 机器人数据 (Bridge/DROID) : LLaVA Instruct Data = **50% : 50%**。

### 2.2. Loss 计算 (Loss Masking)
由于两种数据的标签不同，计算 Loss 时需要进行 Masking：

| 数据类型 | 输入 (Input) | 输出 (Output) | Loss 计算 |
| :--- | :--- | :--- | :--- |
| **机器人数据** | Image + Instruction | Action + (Optional) Text | **Action Head Loss** (MSE/CE) + Text Loss |
| **互联网数据** | Image + Text | Text (Caption/VQA) | **Text Token Loss** (Next Token Prediction) |

> **注意**: 对于互联网数据，Action Head 的输出被 Mask 掉，不产生梯度，因为这些数据没有动作标签。

### 2.3. 代码逻辑 (Pseudo-code)

```python
# 在一个 Batch 中混合两种数据
batch_robot = get_robot_batch() # {image, text, action}
batch_web = get_web_batch()     # {image, text, action=None}

# 1. Forward Robot Data
out_robot = model(batch_robot.image, batch_robot.text)
# 计算动作损失 (e.g., MSE for diffusion, CE for tokenization)
loss_action = mse_loss(out_robot.pred_action, batch_robot.gt_action)

# 2. Forward Web Data
out_web = model(batch_web.image, batch_web.text)
# 计算文本损失 (Next Token Prediction)
loss_text = cross_entropy(out_web.logits, batch_web.gt_text)

# 3. Combined Loss
# lambda 通常为 1.0，也可以根据任务调整
total_loss = loss_action + lambda * loss_text

total_loss.backward()
```

## 3. 案例分析 (Case Studies)

### 3.1. RT-2 (Google DeepMind)
- **发现**: Co-training 对于保持模型的逻辑推理能力至关重要。
- **实验**: 如果不加 Web Data，模型在"将可乐罐放到泰勒斯威夫特照片上"这种需要语义理解的任务上成功率会暴跌。因为模型忘记了"泰勒斯威夫特"是谁。

### 3.2. OpenVLA (Stanford/Berkeley)
- **策略**: 使用 LLaVA 的微调数据 (COCO, GQA, ScienceQA) 进行 Co-training。
- **效果**: 确保了模型在微调动作控制的同时，依然是一个合格的 VLM (能聊天，能描述图像)。这使得 OpenVLA 既能控制机器人，也能当 VLM 用。

## 4. 面试高频问题

**Q: 为什么 Co-training 能提高 Zero-shot 能力？**
A: 机器人数据通常是 Narrow Domain 的（特定场景、特定物体）。通过混合 Web Data，模型保持了对 Wide Domain（通用物体、复杂语义）的理解。当遇到未见过的物体（如"恐龙玩具"）时，模型可以利用 VLM 的知识识别它，并结合学到的抓取动作进行操作。

**Q: 混合比例对性能有什么影响？**
A: 
- **Web Data 过少**: 灾难性遗忘，通用能力下降。
- **Web Data 过多**: 机器人动作学习变慢，因为 Action Loss 的权重被稀释。
- **经验值**: 1:1 是一个稳健的起点。
