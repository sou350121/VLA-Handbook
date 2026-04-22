# VLA 模型全方位对比（16+ 模型）

> 2026-04-21 更新 · 覆盖 RT-1 / RT-2 / OpenVLA / Octo / π0 / π0.5 / π0.6 / π0.7 / GR00T-N1.7 / LingBot / SmolVLA / HAMLET / WVA / VGA / ACT / RDT

---

## 📊 第 1 层：核心架构对比

| 模型 | 机构 | 时间 | Vision | Lang/VLM | Action Head | 参数量 |
|------|------|:----:|--------|----------|-------------|:------:|
| **RT-1** | Google | 2022.12 | EfficientNet-B3 | Universal Sentence Encoder | Discrete Classification | 35M |
| **RT-2** | Google | 2023.07 | ViT-22B (PaLI-X) | PaLM-E / PaLI-X | Discrete Tokens | 55B+ |
| **ACT** | Stanford | 2023.03 | ResNet-18 | — (imitation) | Continuous (action chunking) | 80M |
| **Octo** | Berkeley | 2023.10 | ViT-Base | Transformer | **Diffusion** (DDIM) | 93M |
| **OpenVLA** | Stanford | 2024.06 | SigLIP (ViT-L) | Llama 2 7B | Discrete Tokens | 7B |
| **RDT** | 清华 | 2024.07 | ViT-L | — | Diffusion Transformer | 1B |
| **π0** | PI | 2024.10 | PaliGemma | Gemma 2B | **Flow Matching** | 3B |
| **π0-FAST** | PI | 2024.11 | PaliGemma | Gemma 2B | FAST tokens (DCT) | 3B |
| **π0.5** | PI | 2025.04 | PaliGemma | Gemma 2B | Flow Matching + 分层推理 | 3B |
| **SmolVLA** | HF | 2025.05 | SigLIP-L | SmolLM | Flow Matching | 450M |
| **π0.6** | PI | 2025.11 | PaliGemma 2 | 5B backbone | Flow Matching + Advantage Condition | 5B |
| **GR00T-N1.7** | NVIDIA | 2025.11 | 多形态 VLM | Eagle-1.5 | Diffusion (多形态) | 2B |
| **LingBot-VLA** | Robbyant | 2025.10 | SigLIP | Gemma | Flow Matching | — |
| **WVA** | — | 2025.12 | ViT | — | Value-implicit Planning + Flow | — |
| **VGA** | — | 2025.11 | 视频几何编码 | 无 / 轻 | Flow | — |
| **HAMLET** | CMU | 2026.04 | ViT | 基础 VLM | 历史感知 policy | — |
| **π0.7** | PI | 2026.04 | 3-model 组合（见下） | Gemma 3 | Flow Matching + RECAP 蒸馏 | 5B + 14B 旁路 |

> 💡 π0.7 的组合：4B Gemma 3 VLM + 400M MEM 视觉编码器 + 860M flow matching expert（= 5B 主体） + 14B BAGEL world model 旁路

---

## 📈 第 2 层：性能 × 真机表现

| 模型 | LIBERO | SimplerEnv | GM-100 真机 | 推理速度 |
|------|:------:|:-----------:|:-----------:|:--------:|
| **RT-1** | — | — | — | 3Hz |
| **RT-2** | — | — | — | 1-3Hz |
| **Octo** | 70%+ | 中 | — | 慢（Diffusion） |
| **OpenVLA** | 76%（vanilla）| 中等 | — | 5-10Hz（量化后） |
| **RDT** | 89% | — | — | 10-20Hz |
| **π0** | 93%+ | 强 | — | 10-50Hz（高频） |
| **π0.5** | **96.9%** | **强** | 52-77% | 10-50Hz |
| **π0.6** | — | 强 | — | 类似 π0.5 |
| **GR00T-N1.7** | — | — | — | — |
| **LingBot** | — | — | **18%（100 任务）** · +7.76% over π0.5 | — |
| **WVA** | 99.6% | — | **75.6%**（dual-arm Piper）| — |
| **VGA** | 98.1% | — | 58-75% | — |
| **FAVLA** | — | — | **80.8%**（接触操作 · 260 demos）| — |
| **π0.7** | — | — | 超 π0.6 | 1×H100 / 38ms · world model 4×H100 / 1.25s per subgoal |

⚠️ **LIBERO 数字危险**：参考 [benchmarks 警告 1-3](./benchmarks.md)——这些高分很可能是记忆化。扰动（换物体/位置/指令/环境）下跌到 0-30%。**面试中不要直接报数字，要补充鲁棒性 baseline**。

---

## 🔧 第 3 层：Action Generation 对比

| 类型 | 代表模型 | 特点 | Pros | Cons |
|------|---------|------|------|------|
| **Discrete (Token)** | RT-1 · RT-2 · OpenVLA | 1-256 bins 离散化 | 简单、可迁移 VLM 架构 | 🔴 动作抖动 · 精度天花板 |
| **Continuous MLP** | ACT | 直接回归 | 快 | 🔴 单峰分布，难多模态 |
| **Diffusion (DDIM)** | Octo · Diffusion Policy · RDT | 加噪-去噪 | 多峰分布、表达力强 | 🔴 推理慢 · RL 难做 |
| **Flow Matching (ODE)** | π 系列 · SmolVLA · LingBot | ODE 学向量场 | 🟢 推理快 · 动作分块 | 🟡 RL 微调仍难 |
| **FAST (DCT)** | π0-FAST · WALL-OSS | 频域离散 token | 🟢 压缩 + 预测快 | 🟡 解码需反向 DCT |
| **Value-conditioned** | WVA · π*0.6 | 价值函数引导 | 🟢 支持隐式规划 | 🟡 需要训 value head |

---

## 🎯 核心差异决策树

```
你要选哪个 VLA？
│
├─ 纯学术复现 + 想理解 VLA 范式
│   └─ RT-2（但闭源）/ OpenVLA（开源等价）
│
├─ 开源且想商用
│   ├─ 最轻量 → SmolVLA (450M)
│   ├─ 强鲁棒 → OpenVLA + LoRA 微调
│   └─ 双臂高频 → RDT / π0.5
│
├─ 预训好的基础模型 + 想微调
│   ├─ 多形态 → GR00T-N1.7
│   ├─ 闭源但效果好 → π 系列
│   └─ 全开源 → 看 [open-source-audit](./open-source-audit.md)
│
├─ 想做 RL 后训练
│   ├─ Flow Matching 架构 → π*0.6 (ACP) 或 RL Token
│   └─ Discrete token → OpenVLA + 标准 PPO
│
├─ 接触丰富操作
│   └─ FAVLA + TaF-VLA + 触觉数据
│
├─ 跨形态训练
│   └─ GR00T-N1.7 或 π 系列
│
└─ 研究历史感知 / 长程规划
    └─ HAMLET（唯一明确处理）
```

---

## 🏆 当前（2026-04）最强 baseline 的"官方推荐"

| 场景 | 推荐 | 理由 |
|------|------|------|
| **单任务 baseline** | **ACT** | 📎 50-100 demos 起，80%+ |
| **多任务泛化** | **OpenVLA** | 开源、社区成熟、LIBERO 76% |
| **真机高频控制** | **π0.5 / π0.6** | Flow Matching · 10-50Hz |
| **跨形态预训练** | **GR00T-N1.7** | 5 demos 起 · NVIDIA 开源 |
| **想做 RL 自我进化** | **π*0.6** | ACP + 迭代 · 真机验证 |
| **想做精细操作 RL** | **RL Token** | 冻结 VLA · 轻量微调 · 超人类速度 |
| **人类视频预训练** | **EgoScale 方法复现** | log-linear scaling 实证 |
| **接触操作** | **FAVLA** | 80.8% · 260 demos |

---

## ❗ 常见面试误区

### 误区 1：混用 Diffusion 和 Flow Matching 的术语

| 对比 | Diffusion | Flow Matching |
|------|-----------|--------------|
| **数学形式** | SDE（随机微分方程） | ODE（常微分方程） |
| **训练 loss** | 噪声预测 (MSE) | 向量场预测 (MSE on velocity) |
| **推理** | 多步随机采样 (DDPM) 或 ODE 求解 (DDIM) | **确定性** ODE 积分 (Euler) |
| **速度** | 慢（多步）→ 或 DDIM 加速 | **快** · 少步数（通常 4-10 步） |
| **RL 难度** | 每步降噪建模为 Gaussian policy · 可做 DPPO | 需要拆解 ODE 整体分布 · 更难 |

### 误区 2：只讲参数量不讲 action head

- 📎 π0.5 主体只有 3B 但超 OpenVLA 7B — 因为 **Flow Matching 头 + 分层推理**
- 📎 π0.7 的 5B 主体对比 π0.6 的 5B：差异在于 **RECAP strategy-metadata 蒸馏**

### 误区 3：忽略"机器人"侧的硬件-数据-模型耦合

VLA 比较不能只看算法，还要看：
- **数据量**（20K hrs 真实 vs 50 demos）
- **硬件**（单臂 vs 双臂 vs 人形 × 夹爪 vs 灵巧手）
- **评估环境**（LIBERO 仿真 vs GM-100 真机 vs BEHAVIOR-1K）
- **许可证**（CC BY 4.0 可商用 vs CC BY-NC 禁商用）

---

## 🚩 Red Flag：论文声称强但你要追问

| 论文说… | 你追问… |
|---------|--------|
| "我们 LIBERO 98%" | "LIBERO-PRO / Plus / X 呢？"（📎 可能 0.0%） |
| "我们做到 SOTA" | "哪个 benchmark？哪个 split？真机还是仿真？" |
| "预训练 XX 万小时" | "数据混合的 license 分布是？能商用吗？" |
| "语言条件控制强" | "做过语言消融吗？mask 语言后成功率多少？" |
| "扩散政策 95% 成功" | "失败情况下恢复能力？F6 动作偏斜测试？" |
| "跨形态泛化" | "embodiment gap 量化值？哪些形态？" |
| "完全开源" | "fully open / weight-only / inference-only？"（参考 [open-source-audit](./open-source-audit.md)） |

---

## 📚 详细深度解读（推荐路径）

| 模型 | 深度文章 |
|------|---------|
| π 系列整体 | [VLA 架构主线](../theory/vla-core/vla_arch.md) |
| π0.7 最新 | [π0.7 deep dive](../theory/vla-core/pi0_7_steerable_compositional_generalization_2026.md) |
| PI 愿景访谈 | [Sergey Levine 深度访谈](../theory/vla-core/physical_intelligence_sergey_levine_foundation_model_vision_2026.md) |
| GR00T | [GR00T-N1.7 deep dive](../theory/vla-core/groot_n1_7_nvidia_open_foundation_model_2026.md) |
| LingBot | [LingBot-VLA deep dive](../theory/vla-core/lingbot_vla_pragmatic_foundation_model_2026.md) |
| HAMLET | 见最新 VLA 周报 |

---

[← Back to Cheat Sheet](./README.md)
