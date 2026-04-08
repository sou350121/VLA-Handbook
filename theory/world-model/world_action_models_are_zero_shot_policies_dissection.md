---
auto_generated: true
generated_at: "2026-02-20T08:02:40Z"
source_paper: "https://arxiv.org/abs/2602.15922"
arxiv_id: "2602.15922"
---
# World Action Models are Zero-shot Policies (DreamZero)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-02-20
>
> **论文**: World Action Models are Zero-shot Policies
> **链接**: https://arxiv.org/abs/2602.15922
> **核心定位**: 用视频扩散模型做世界模型，联合预测视频帧和动作，实现 2 倍于 SOTA VLA 的零样本泛化能力 + 7Hz 实时控制

**一句话 takeaway**: 把 VLA 的 VLM 先验换成视频扩散先验，机器人策略从"语义理解"升级为"物理动力学理解"，无需重复演示即可从异构数据中学习新技能。

---

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 模块 | 输入 | 输出 | 频率/时序 | 训练/推理差异 |
|------|------|------|-----------|---------------|
| VAE 视觉编码器 | 当前帧 + 历史帧 (o_0:l) | 视觉 latent | 每帧 | 训练/推理一致 |
| 文本编码器 | 语言指令 (c) | 文本 embedding | 每任务 | 训练/推理一致 |
| 状态编码器 | 本体感知 (q_l) | 状态 embedding | 每帧 | 训练/推理一致 |
| DiT 主干 (14B) | 上述三者 + 噪声 latent | 去噪后的视频+动作 latent | 自回归 chunk 级 | 推理用 KV cache 加速 |
| 视频解码器 | 视频 latent | 未来 H 帧 (o_l:l+H) | 每 chunk | 训练用 teacher-forcing，推理自回归 |
| 动作解码器 | 动作 latent | 动作序列 (a_l:l+H) | 每 chunk | 与视频同步输出 |

### 1.2 关键机制 (Key Mechanism)

**为什么联合预测视频 + 动作？**

DreamZero 的核心洞察是：视频预测 = 隐式物理规划器。联合建模的分解形式为：

```
π_0(o_l:l+H, a_l:l+H | o_0:l, c, q_l) = π_0(o_l:l+H | o_0:l, c, q_l) × π_0(a_l:l+H | o_0:l+H, q_l)
         ↑ DreamZero                    ↑ 视频预测                      ↑ 逆动力学模型 (IDM)
```

- **视频预测头**继承 web-scale 视频扩散模型的时空先验（物理动力学、物体持久性、运动连续性）
- **动作预测头**学习从生成的视频未来中提取对应的电机命令（逆动力学）
- **端到端训练**确保视频和动作深度对齐，而非两个独立模型的松散耦合

**为什么用自回归架构而非双向？**

| 架构 | 优点 | 缺点 | DreamZero 选择 |
|------|------|------|----------------|
| 双向 (Bidirectional) | 全局上下文，生成质量高 | 无法用于闭环控制，推理慢 | ❌ |
| 自回归 (Autoregressive) | 支持 KV cache 加速，天然适合闭环 | 误差累积风险 | ✅ 用真实观测替换预测帧消除累积误差 |

推理时的关键技巧：每个 action chunk 执行后，将**真实观测**而非预测帧写入 KV cache，防止误差累积。

### 1.3 信息流/架构图 (Flow / Diagram)

```
┌─────────────────────────────────────────────────────────────────┐
│                        DreamZero Inference                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Language Instruction ──→ [Text Encoder] ──┐                     │
│                                            │                     │
│  Current Obs + History ──→ [VAE] ──────────┼─→ [DiT Backbone] ──┼──→ [Video Decoder] ──→ Future Frames
│                                            │    (14B, AR+FM)    │                        (o_l:l+H)
│  Proprioception ──────────→ [State Enc] ──┘                     │                     │
│                                                                  │                     │
│                                                                  └──→ [Action Decoder] ──→ Action Chunk
│                                                                                          (a_l:l+H)
│                                                                  │
│  Real-World Execution ←──────────────────────────────────────────┘
│       │
│       └──→ New Observation ──→ Update KV Cache (replace predicted frames)
│
└─────────────────────────────────────────────────────────────────┘
```

**训练流程** (chunk-wise teacher-forcing):
```
For each chunk in trajectory:
  1. Sample noise for video & action latents
  2. Condition on clean context (o_0:l, c, q_l)
  3. Denoise both modalities jointly with flow matching
  4. Compute loss on video + action prediction
```

---

## 2. 数学核心 (Math Core)

### 2.1 训练目标

DreamZero 使用 **flow matching** 作为扩散目标，联合优化视频和动作：

```
L = E_{t,ε_v,ε_a} [ ||v_θ(o_t, a_t, t, c, o_0:l, q_l) - (ε_v, ε_a)||² ]
```

其中：
- `o_t = t·o_clean + (1-t)·ε_v` 是视频在时间 t 的噪声插值
- `a_t = t·a_clean + (1-t)·ε_a` 是动作在时间 t 的噪声插值
- `v_θ` 是模型预测的速度场 (velocity field)
- `t ∈ [0,1]` 是扩散时间步

### 2.2 自回归分解

联合预测的链式法则分解：

```
p(o_l:l+H, a_l:l+H | context) = p(o_l:l+H | context) · p(a_l:l+H | o_l:l+H, context)
```

**直觉**：先生成"世界会如何演变"，再根据这个未来决定"我该做什么动作"。这比直接从当前状态映射到动作多了物理一致性约束。

### 2.3 推理加速公式

DreamZero-Flash 使用 **解耦的去噪调度**：

```
视频去噪步数：N_v (默认 16 步)
动作去噪步数：N_a (可降至 1-4 步)

总延迟 ≈ N_v · T_viT + N_a · T_action_head
```

通过减少 N_a 并缓存视频 latent，可实现 38 倍加速。

> 符号与本文/相关文档保持一致：o=视觉观测，a=动作，c=语言指令，q=本体感知状态，H=预测 horizon

---

## 3. 带数字走一遍：玩具例子 (Worked Example)

**场景**：机器人看到桌上有一个杯子，指令是"把杯子推到桌子边缘"。

**输入**：
- 视觉上下文 o_0:l: 3 帧 RGB 历史 (720p, VAE 压缩到 64×64 latent)
- 语言 c: "push the cup to the edge"
- 本体感知 q_l: 末端位置 [x=0.3, y=0.2, z=0.5], 关节角度 [θ1...θ7]

**推理过程** (单 chunk, H=8 帧，动作 chunk 长度=8)：

```
Step 1: 采样初始噪声
  o_noise ~ N(0, I), shape: (8, 64, 64, 4)  # 8 帧，4 通道 VAE latent
  a_noise ~ N(0, I), shape: (8, 7)          # 8 步，7 维动作 (6DoF + gripper)

Step 2: 迭代去噪 (16 步)
  For t = 16 → 1:
    v_pred = DiT(o_t, a_t, t, c, o_0:l, q_l)
    o_{t-1} = o_t - step_size · v_pred_video
    a_{t-1} = a_t - step_size · v_pred_action

Step 3: 解码输出
  视频：8 帧预测，显示杯子逐渐移动到桌边
  动作：8 步末端位移，Δx=+0.02m/步，Δy=0, Δz=0, gripper=open

Step 4: 执行 + 更新
  执行前 4 步动作 (500ms)
  用真实观测替换 KV cache 中的预测帧
  进入下一个 chunk
```

**关键数值**：
- 单次推理延迟：150ms (优化后)
- 控制频率：7Hz (1000ms / 150ms ≈ 6.7，chunk 重叠实现平滑)
- 视频 - 动作对齐误差：<2 像素 (通过联合训练保证)

---

## 4. 工程视角 (Engineering View)

### 4.1 延迟分解与优化

| 优化类别 | 技术 | 加速比 | 性能影响 |
|----------|------|--------|----------|
| 算法级 | DreamZero-Flash (解耦调度) | 4× | 无 |
| 系统级 | KV Cache + 异步推理 | 6× | 无 |
| 系统级 | 算子融合 + 并行 | 3× | 无 |
| 低级别 | INT8 量化 + CUDA kernel 调优 | 2.5× | <1% 精度损失 |
| **总计** | | **38×** | **可忽略** |

**原始延迟**: 5.7s/action chunk → **优化后**: 150ms/action chunk

### 4.2 部署约束

| 约束 | 要求 | DreamZero 方案 |
|------|------|----------------|
| 控制频率 | ≥5Hz 才能平滑操作 | 7Hz (达标) |
| 显存 | 14B 模型 + KV cache | 80GB A100/H100 (单卡) |
| 观测延迟 | 相机→推理→动作 <200ms | 150ms 推理 + 相机帧率 30Hz |
| 动作平滑 | 无抖动 | Chunk 重叠 + 指数移动平均 |

### 4.3 关键工程权衡

**视频分辨率 vs 延迟**：
- 高分辨率 (720p): 视频预测质量高，但 VAE encode/decode 延迟增加 40%
- 低分辨率 (256p): 延迟降低，但精细操作 (如插孔) 失败率上升 15%
- DreamZero 选择：640×480 VAE latent，平衡点

**Chunk 长度 vs 误差累积**：
- 长 chunk (H=16): 规划更长远，但模型预测误差累积
- 短 chunk (H=4): 更频繁纠偏，但动作可能不连贯
- DreamZero 选择：H=8，执行前 4 步后重新观测

---

## 5. 数据与评测 (Data & Eval)

### 5.1 训练数据组成

| 数据来源 | 时长 | 类型 | 特点 |
|----------|------|------|------|
| AgiBot G1 采集 | ~400 小时 | 真实机器人 | 异构轨迹 (非重复演示) |
| DROID 公开数据 | ~100 小时 | 真实机器人 (Franka) | 多任务多环境 |
| **总计** | **~500 小时** | | **无仿真数据** |

**关键洞察**：数据多样性 > 数据量。500 小时异构数据 > 2000 小时重复演示。

### 5.2 评测基准与结果

**AgiBot 零样本泛化** (4 台机器人，80 个 rollout):

| 任务类型 | DreamZero | 最佳 VLA 基线 | VLA (从头训练) |
|----------|-----------|---------------|----------------|
| Seen Tasks | 72.4% | 45.2% | 8.1% |
| Unseen Tasks | 39.5% | 12.3% | 2.4% |
| **平均** | **62.2%** | **27.4%** | **5.3%** |

**Unseen Tasks 示例**：解鞋带、握手、扇汉堡——这些动作从未出现在训练数据中。

**DROID 未见动词泛化**：

| 模型 | Seen Verbs | Unseen Verbs |
|------|------------|--------------|
| DreamZero | 61% | 49% |
| OpenVLA | 58% | 25% |
| RT-2-X | 63% | 32% |

**Post-Training 后泛化保持**：

在 3 个下游任务 (折衣服、装水果、收拾桌子) 微调后：
- DreamZero: 环境泛化能力保持 90%+
- VLA 基线：微调后泛化下降 40-60%

### 5.3 跨具身迁移

| 迁移类型 | 数据量 | 性能提升 |
|----------|--------|----------|
| 视频演示 (人/其他机器人) | 10-20 分钟 | +42% (未见任务) |
| Play Data (新机器人 YAM) | 30 分钟 (55 条轨迹) | 零样本泛化保持 |

---

## 6. 能力与失败模式 (Capabilities & Failure Modes)

### 6.1 能做什么 (Capabilities)

| 能力 | 场景示例 | 关键原因 |
|------|----------|----------|
| 零样本新技能 | 解鞋带、摇铃铛、弹木琴 | 视频先验编码了物理交互 |
| 跨环境泛化 | 新桌子、新光照、新背景 | 视频扩散的域不变性 |
| 跨具身学习 | 从 AgiBot G1 → YAM | 视频是具身无关的表示 |
| 长序列操作 | 多步任务 (打开→取出→放置) | 自回归视频预测提供长期规划 |
| 精细操作 | 插孔、倒水、翻煎饼 | 视频 - 动作紧密对齐 |

### 6.2 不能做什么 (Failure Modes)

| 失败模式 | 触发条件 | 根本原因 |
|----------|----------|----------|
| 高速动态任务 | 抛接、快速击打 | 视频预测 horizon 有限 (H=8 帧) |
| 触觉依赖任务 | 盲抓、力控装配 | 缺少触觉/力觉模态 |
| 长文本指令 | >50 词的复杂指令 | 文本编码器上下文窗口限制 |
| 极端域偏移 | 水下、太空、微观操作 | 视频先验不覆盖这些物理 |
| 多人交互 | 动态 human-robot 协作 | 社会推理能力有限 |

---

## 7. 与相关工作对比 (Comparison)

### 7.1 VLA vs WAM 架构对比

| 维度 | VLA (如 OpenVLA, RT-2) | WAM (DreamZero) |
|------|------------------------|-----------------|
| 基础先验 | VLM (静态图像 - 文本) | 视频扩散 (动态时空) |
| 训练目标 | 语言→动作 | 语言 + 观测→视频 + 动作 |
| 物理理解 | 隐式 (从动作数据学习) | 显式 (视频预测约束) |
| 数据效率 | 需要重复演示 | 异构轨迹即可 |
| 零样本技能 | 有限 (依赖 VLM 知识) | 强 (视频先验泛化) |
| 推理速度 | 快 (50-100ms) | 较慢 (150ms，优化后) |
| 显存需求 | 7-13B, 40-80GB | 14B, 80GB+ |

### 7.2 WAM 系列工作对比

| 模型 | 基础架构 | 视频 - 动作对齐 | 实时控制 | 跨具身 |
|------|----------|-----------------|----------|--------|
| DreamZero | Wan 视频扩散 (14B) | 联合端到端 | 7Hz | 30 分钟 play data |
| Genie 2 | 自回归 Transformer | 隐式 | 未报告 | 未报告 |
| WAFT | 视频扩散 + VLA | 两阶段 | 未报告 | 需要校准 |
| Video-Action Transformer | ViT + 动作头 | 联合 | <3Hz | 不支持 |

### 7.3 面试 Tip

**问**: "DreamZero 相比 VLA 的核心优势是什么？为什么视频预测能帮助动作学习？"

**答**: "VLA 从静态 VLM 继承语义先验，但缺少时空动力学理解。DreamZero 用视频扩散模型做世界模型，联合预测视频和动作——视频预测充当隐式物理规划器，约束动作必须符合物理规律。结果是：1) 从异构数据学习 (无需重复演示)，2) 零样本新技能泛化提升 2 倍，3) 跨具身迁移只需 30 分钟数据。"

---

## 关键引用

- **论文**: https://arxiv.org/abs/2602.15922
- **项目主页**: https://dreamzero0.github.io
- **代码**: https://github.com/dreamzero0/dreamzero
- **评测视频库**: https://dreamzero0.github.io/evals_gallery
- **YAM 机器人适配**: https://dreamzero0.github.io/yam_gallery

---

[← Back to Theory](./README.md)
