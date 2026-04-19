# 完全开源 VLA 选型指南：谁是真开源，谁在"开源洗"

> **"开源"在 VLA 领域是一个被严重滥用的词。** 有的模型放了权重但没训练代码，有的放了代码但许可证不允许商用，有的连许可证都没声明。这篇指南逐个拆解，帮你在 10 分钟内选对 baseline。

<table><tr><td>

**更新**：2026-04-19 · Claude Opus 4.6 × [Pulsar 照见](https://github.com/sou350121/Pulsar-KenVersion)
**方法**：逐一检查 GitHub 仓库、HuggingFace 模型卡、论文附录、许可证文件。所有 stars 数据为实时查询。

</td></tr></table>

---

## 开源三级制

不是所有"开源"都一样。我们定义三个等级：

| 等级 | 要求 | 你能做什么 |
|:----:|------|-----------|
| 🟢 **完全开源** | 权重 ✅ + 完整训练代码 ✅ + 推理代码 ✅ + 训练数据可获取 ✅ + 宽松许可（MIT/Apache） | 从头训练 · 修改架构 · 复现论文结果 · 商用 |
| 🟡 **半开源** | 权重 ✅ + 推理/微调代码 ✅ + 预训练代码 ❌ 或 数据 ❌ 或 许可证不明 | 微调到你的任务 · 不能从头复现 · 商用风险 |
| 🔴 **闭源** | 只有论文/博客/demo 视频 | 只能读论文学思路 |

> 💡 **"开源洗"（Open-washing）的常见手法**：
> 1. 放权重但不放训练代码 → 你只能用，不能改
> 2. 放代码但数据不公开 → 你不能复现论文结果
> 3. 不声明许可证 → 默认是"保留所有权利"，商用有法律风险
> 4. 用非标准许可证（如"仅限研究"）→ 不是真正的开源

---

## 🟢 完全开源（10 个，按推荐度排序）

### 1. LeRobot / SmolVLA — 最大社区，入门首选

| 维度 | 详情 |
|------|------|
| **GitHub** | [huggingface/lerobot](https://github.com/huggingface/lerobot) · **23.3K** ⭐ |
| **参数** | SmolVLA 450M · 也支持 ACT/Diffusion Policy/TDMPC 等多种策略 |
| **许可证** | Apache-2.0 ✅ |
| **权重** | ✅ HuggingFace `lerobot/smolvla_base` |
| **训练代码** | ✅ 完整（预训练 + 微调 + 评估） |
| **训练数据** | ✅ LeRobot 社区数据集（开放贡献） |
| **LIBERO** | 82-90%（OpenVLA 的 95% 但参数只有 1/16） |
| **硬件需求** | 训练：1× RTX 3090 即可 · 推理：消费级 GPU |
| **维护** | 🟢 极活跃（2026-04-19 最后更新） |
| **安装** | `pip install lerobot` |

**为什么排第一**：社区最大（23K stars）、HuggingFace 官方维护、与 LeRobot 硬件生态绑定、支持多种策略不只是 SmolVLA、**最容易加新模态**（框架设计的目标之一）。

**适合**：初学者 · 消费级硬件 · 想加新传感器（触觉等）· 快速原型

---

### 2. ACT — 代码最干净，学术金标准

| 维度 | 详情 |
|------|------|
| **GitHub** | [tonyzhaozh/act](https://github.com/tonyzhaozh/act) · **1.9K** ⭐ |
| **参数** | ~80M（CVAE + Transformer） |
| **许可证** | MIT ✅ |
| **训练数据** | ✅ ALOHA 演示数据 |
| **硬件需求** | 训练：1× RTX 2080 即可 · $20K ALOHA 硬件 |
| **维护** | 🟡 稳定（核心代码成熟，不频繁更新） |

**为什么排第二**：代码不到 1K 行、所有 VLA 论文的默认 baseline、ALOHA 硬件生态最成熟。**如果你想改架构做实验，ACT 的代码是最好改的。**

**适合**：学术研究 · 改架构实验 · 双臂遥操作

---

### 3. RDT-1B — 首个 1B 扩散策略，积极维护

| 维度 | 详情 |
|------|------|
| **GitHub** | [thu-ml/RoboticsDiffusionTransformer](https://github.com/thu-ml/RoboticsDiffusionTransformer) · **1.7K** ⭐ |
| **参数** | RDT-170M / **RDT-1B** |
| **许可证** | MIT ✅ |
| **训练数据** | ✅ OXE（Open X-Embodiment） |
| **LIBERO** | 竞争力强 |
| **维护** | 🟢 活跃（清华 MARS Lab + 字节跳动） |

**为什么排第三**：1B 参数级别的开源扩散策略只此一家。在 Diffusion Policy 方向上是最强的开源选择。

**适合**：Diffusion Policy 方向研究 · 跨形态预训练 · 大规模训练

---

### 4. OpenVLA + OFT — 经典 baseline（注意停更）

| 维度 | 详情 |
|------|------|
| **GitHub** | [openvla/openvla](https://github.com/openvla/openvla) · **5.9K** ⭐ |
| **OFT 版** | [moojink/openvla-oft](https://github.com/moojink/openvla-oft) · **1.1K** ⭐ |
| **参数** | 7B（Llama 2 + SigLIP） |
| **许可证** | MIT ✅ |
| **训练数据** | ✅ OXE |
| **LIBERO** | 76.5%（vanilla）/ 97.1%（OFT） |
| **OFT 改进** | 推理 25-50x 加速 · 多图输入 · 双臂 |
| **维护** | 🟡 **停更**（OpenVLA 2025-03，OFT 2025-09） |

**⚠️ 注意**：虽然 stars 多但已停更。论文引用多但代码不再维护。**新项目建议用 LeRobot 或 RDT 替代。**

---

### 5. GR00T-N1.7 — NVIDIA 官方，工程部署首选

| 维度 | 详情 |
|------|------|
| **GitHub** | [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) · **6.7K** ⭐ |
| **参数** | 2.2B（Cosmos-Reason2-2B / Qwen3-VL + Diffusion Transformer） |
| **许可证** | Apache-2.0 ✅ |
| **权重** | ✅ HuggingFace |
| **训练代码** | ✅ 微调脚本（`launch_finetune.py`）· SONIC 训练 |
| **预训练代码** | ❌ 不公开 |
| **训练数据** | ⚠️ 部分（LeRobot 格式示例数据，预训练数据不公开） |
| **硬件** | 推理：RTX 4090 / Jetson · 微调：H100 推荐 |
| **维护** | 🟢 极活跃（2026-04-18） |

**严格说是 🟡 半开源**（预训练不可复现），但微调生态最完整。NVIDIA 生态 = Isaac Lab + Omniverse + GR00T。

**适合**：工程部署 · 人形机器人 · NVIDIA GPU 用户

---

### 6. LingBot-VLA — 最大真实数据预训练

| 维度 | 详情 |
|------|------|
| **GitHub** | [Robbyant/LingBot-VLA](https://github.com/Robbyant/LingBot-VLA) · **1.1K** ⭐ |
| **许可证** | Apache-2.0 ✅ |
| **训练数据** | ✅ **20,000 小时真实机器人数据** · 9 种双臂配置 |
| **工具链** | 数据处理 + 微调 + 自动评估（生产级） |
| **维护** | 🟢 活跃（千寻智能/Robbyant） |

**亮点**：目前公开的**最大真实机器人预训练数据规模**。如果你有双臂平台，这是最直接可用的。

---

### 7. CrossFormer — 30 种形态跨形态策略

| 维度 | 详情 |
|------|------|
| **GitHub** | [rail-berkeley/crossformer](https://github.com/rail-berkeley/crossformer) · **282** ⭐ |
| **许可证** | MIT ✅ |
| **训练数据** | ✅ 900K 轨迹（OXE，30 种形态） |

**亮点**：唯一一个在 30 种不同机器人形态上预训练的开源策略。跨形态迁移的最佳起点。

---

### 8. CogACT — Microsoft 认知+动作协同

| 维度 | 详情 |
|------|------|
| **GitHub** | [microsoft/CogACT](https://github.com/microsoft/CogACT) · **419** ⭐ |
| **参数** | Small / Base / **Large** 三个尺寸 |
| **许可证** | MIT ✅ |
| **权重** | ✅ HuggingFace（完整 checkpoint + config） |

**亮点**：模型先做认知推理再生成动作，架构天然适合"先理解再操作"。

---

### 9. HybridVLA — Diffusion + Autoregressive 混合

| 维度 | 详情 |
|------|------|
| **GitHub** | [PKU-HMI-Lab/Hybrid-VLA](https://github.com/PKU-HMI-Lab/Hybrid-VLA) · **346** ⭐ |
| **许可证** | MIT ✅ |

**亮点**：两种 Action Head 协同——Diffusion 做精细动作，Autoregressive 做粗粒度规划。

---

### 10. HPT — 异构预训练 Transformer

| 维度 | 详情 |
|------|------|
| **GitHub** | [liruiw/HPT](https://github.com/liruiw/HPT) · **534** ⭐ |
| **许可证** | MIT ✅ |
| **训练数据** | ✅ 多源异构数据 |

**亮点**：能从多种不同格式/来源的数据中统一预训练，不要求数据格式对齐。

---

## 🟡 半开源（能用但有坑）

| 模型 | Stars | 有什么 | 缺什么 | 许可证 | 风险 |
|------|------:|--------|--------|--------|------|
| **π0 (openpi)** | 11.4K | 权重(π0+π0.5) + 推理 + 微调 | 预训练代码 · 预训练数据 · π0-FAST PyTorch 不完整 | Apache-2.0 | 不能从头复现 |
| **StarVLA** | 1.9K | 权重 + 训练 + 推理 | — | **未声明** ⚠️ | 商用法律风险 |
| **WALL-OSS** | — | 权重(HF) + 推理 | 训练代码有限 | **未声明** ⚠️ | 商用法律风险 |
| **DexVLA** | 55 | 权重 + 代码 | — | **未声明** ⚠️ | 低活跃 + 法律风险 |

> ⚠️ **"未声明许可证"= 默认保留所有权利。** 没有许可证的代码，法律上你不能复制、修改或分发。用于学术研究通常没问题，但商用前**必须联系作者确认**。

---

## 🔴 闭源（只有论文）

| 模型 | 为什么关注 | 有什么 |
|------|-----------|--------|
| **π\*0.6** | Recap RL | 只有论文 |
| **π0.7** | 组合泛化 | 只有博客 + 媒体报道，**连论文都没有** |
| **Helix 02** | 全身 loco-manipulation | 只有 Figure AI 博客 |
| **VGA** | 3D backbone | 论文，但 VGGT backbone 本身开源（Meta） |
| **WVA** | 隐式规划 99.6% | 只有论文 |

---

## 按场景选型

### "我是学生，第一次做 VLA"
→ **LeRobot/SmolVLA**。450M 参数，消费级 GPU，社区最大，遇到问题有人回答。

### "我要做学术论文，需要 baseline 对比"
→ **ACT**（代码金标准）+ **RDT-1B**（Diffusion 最强开源）+ **OpenVLA-OFT**（VLM 方向最强开源）

### "我要加新传感器（触觉等）"
→ **LeRobot** 框架最容易扩展。或者 **ACT** 代码最好改。

### "我要做跨形态迁移"
→ **CrossFormer**（30 形态预训练）或 **HPT**（异构数据预训练）

### "我要工程部署到产品中"
→ **GR00T-N1.7**（NVIDIA 生态完整）或 **LingBot-VLA**（生产级工具链 + 20K 小时数据）

### "我只有一台 RTX 3090"
→ **SmolVLA**（450M，单卡可训）或 **ACT**（80M，最轻）

### "我有 8×A100，想追 SOTA"
→ **RDT-1B** 或 **LingBot-VLA** 做预训练起点，+ RL 后训练（参考 π\*0.6 Recap 论文）

---

## 快速开始

### SmolVLA（推荐入门）

```bash
pip install lerobot
python -m lerobot.scripts.train \
  --policy.type=smolvla \
  --env.type=libero \
  --env.task=libero_spatial
```

### ACT（最小代码）

```bash
git clone https://github.com/tonyzhaozh/act.git
cd act
pip install -r requirements.txt
python train.py --task_name sim_transfer_cube_scripted
```

### RDT-1B（Diffusion 方向）

```bash
git clone https://github.com/thu-ml/RoboticsDiffusionTransformer.git
cd RoboticsDiffusionTransformer
pip install -r requirements.txt
# 下载预训练权重
python scripts/download_pretrained.py
```

---

## 延伸阅读

| 方向 | 推荐 |
|------|------|
| VLA 架构全景 | [VLA 核心架构](vla_arch.md) |
| 研究主线 | [VLA 赌注清单](vla_research_mainline.md) |
| π0 系列 | [openpi 微调指南](pi0_code_analysis.md) |
| 3D 感知工具 | [点云与 SLAM 60+ 工具](../perception/pointcloud_slam.md) |
| 潜空间理论 | [潜空间综述](../foundation/latent_space_survey_foundation_evolution_mechanism_ability_2026.md) |
| 数学基础 | [VLA 数学必备](../foundation/math_for_vla.md) |

---

[← Back to Explorer's Map](../README.md)
