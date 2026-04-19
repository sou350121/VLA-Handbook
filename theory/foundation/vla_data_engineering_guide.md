# VLA 数据工程指南：从采集到训练的完整链路

> 模型可以换，架构可以改，但**数据错了就全错了**。
>
> 这篇是 VLA 数据工程的"交通枢纽"——覆盖采集、格式、质量、规模、模态的全链路，每个主题可以展开为独立的深度文章。

<table><tr><td>

**更新**：2026-04-19 · Claude Opus 4.6 × [Pulsar 照见](https://github.com/sou350121/Pulsar-KenVersion)

</td></tr></table>

---

## 0. VLA 数据的独特挑战

VLA 数据不是"图文对"也不是"视频标注"——它是**多模态时序信号的精密同步采集**：

```
时间轴 →
  RGB 相机 1:  ████████████████  30Hz
  RGB 相机 2:  ████████████████  30Hz
  深度相机:    ████████████████  30Hz
  关节角:     ████████████████████████████████  100Hz
  末端位姿:   ████████████████████████████████  100Hz
  夹爪状态:   ████████████████████████████████  100Hz
  语言指令:   █                               1x（任务开始时）

  → 所有信号必须在 <10ms 内对齐，否则动作标签对不上图像
```

**和 NLP/CV 数据的本质区别**：
- NLP：文本是离散的，没有时间同步问题
- CV：图像是静态的，没有动作标签
- VLA：**连续、多模态、时序对齐、物理耦合**——任何一个环节出错，整条轨迹都废了

---

## 1. 数据采集硬件

### 主流遥操作平台

| 平台 | 价格 | 形态 | 特点 | 开源 |
|------|:----:|------|------|:----:|
| **ALOHA** | ~$20K | 双臂 | ACT 的配套硬件，社区最大 | ✅ MIT |
| **ALOHA 2** | ~$25K | 双臂 | 升级版，Stanford 维护 | ✅ |
| **UMI** | ~$5K | 单臂 | 手持采集（不需要机器人做遥操主端）| ✅ |
| **Gello** | ~$3K | 单臂 | 低成本 3D 打印遥操主端 | ✅ |
| **DROID** | ~$30K | 单臂 | Toyota Research，多机构标准化 | ✅ |
| **SO-100** | ~$500 | 单臂 | LeRobot 入门级，最便宜 | ✅ |

### 传感器配置参考

| 传感器 | 型号参考 | 频率 | 用途 | 价格 |
|--------|---------|:----:|------|:----:|
| RGB 相机 | RealSense D435i | 30Hz | 主视觉 | ~$300 |
| 手腕相机 | RealSense D405 | 30Hz | 近距离操作 | ~$200 |
| 深度 | RealSense 内置 | 30Hz | 3D 感知 | 含在上面 |
| 关节编码器 | 机械臂自带 | 100-1000Hz | 本体感觉 | 含在臂中 |
| 触觉 | GelSight Mini | 30Hz | 接触感知 | ~$300 |

### 时间同步——最容易出错的环节

| 方式 | 精度 | 适用 | 复杂度 |
|------|:----:|------|:------:|
| **软件时戳** | ~10-50ms | 原型 | 低 |
| **ROS2 时间同步** | ~5-10ms | 学术 | 中 |
| **PTP (IEEE 1588)** | ~1ms | 工程 | 高 |
| **硬件触发** | <0.1ms | 高精度 | 高 |

> ⚠️ **常见致命错误**：USB 相机的软件时戳漂移可达 50ms。在 30Hz 视频中 50ms = 1.5 帧的错位。如果动作标签对着错误的帧学，模型学到的是错误的时序关系。

---

## 2. 数据格式标准

### 当前主流格式

| 格式 | 推动者 | 用谁 | 特点 |
|------|--------|------|------|
| **LeRobot v2** | HuggingFace | SmolVLA, GR00T-N1.7, LingBot | Parquet + MP4 · `modality.json` 定义键映射 · **当前推荐** |
| **OXE / RLDS** | Google | OpenVLA, Octo, RT-2 | TFRecord · Open X-Embodiment 标准 · 数据量最大 |
| **HDF5** | Stanford | ACT, ALOHA | 单文件 · 简单 · 不适合大规模 |
| **Zarr** | — | Diffusion Policy | 分块存储 · 随机访问快 |

### LeRobot v2 结构（推荐）

```
dataset/
├── meta/
│   ├── info.json          # 数据集元信息（fps, robot_type, shapes）
│   ├── episodes.jsonl     # episode 列表（start_idx, end_idx, task）
│   ├── tasks.jsonl        # 任务描述列表
│   └── modality.json      # state/action/video 键映射（GR00T 特有）
├── data/
│   └── chunk-000/
│       └── episode_000.parquet   # 数值数据（关节角、动作、时戳）
└── videos/
    └── chunk-000/
        └── episode_000.mp4       # 视频（压缩存储）
```

**为什么推荐 LeRobot v2**：
1. HuggingFace 维护 → 社区最大 → 兼容性最好
2. Parquet 列存储 → 读取特定列极快（只读动作不读图像）
3. MP4 视频 → 存储小（比原始帧小 10-50x）
4. `modality.json` → 一个文件描述所有模态映射，换机器人只需改这个文件

### 格式转换

```python
# OXE → LeRobot
from lerobot.common.datasets.push_dataset_to_hub import convert_oxe_to_lerobot

# HDF5 → LeRobot
from lerobot.common.datasets.push_dataset_to_hub import convert_hdf5_to_lerobot
```

---

## 3. 数据质量

### 什么是"好的 demo"

| 维度 | 好 | 差 |
|------|-----|-----|
| **速度** | 匀速、流畅 | 突然加速/停顿/犹豫 |
| **轨迹** | 最短路径或自然路径 | 绕路、来回纠正 |
| **抓取** | 一次到位 | 抓了放、放了抓 |
| **视角** | 目标始终可见 | 遮挡、出框 |
| **一致性** | 同一任务的不同 demo 策略相似 | 每次做法完全不同 |

### 常见采集错误

| 错误 | 后果 | 检测方法 |
|------|------|---------|
| 时间戳漂移 | 动作对错帧 | 对比 action 与 image 的时间差分布 |
| 遥操卡顿 | 轨迹中插入"冻结帧" | 检测连续相同的 action 值 |
| 相机曝光变化 | 训练数据分布不一致 | 检查图像亮度直方图 |
| 夹爪状态丢失 | 模型不知道什么时候该抓 | 检查 gripper 信号的变化频率 |
| 坐标系不统一 | 不同采集批次的动作不可比 | 标定验证 |

### 质量检查 Pipeline

```python
# 建议在采集后立即运行
def check_episode_quality(episode):
    checks = {
        'timestamp_gap': max(diff(timestamps)) < 50ms,
        'action_frozen': count(identical_consecutive_actions) < 5,
        'image_brightness': std(brightness) < threshold,
        'gripper_active': count(gripper_changes) > 0,
        'duration': 5s < episode_length < 120s,
    }
    return all(checks.values()), checks
```

---

## 4. 数据规模：需要多少数据

### 各模型的数据量 vs 性能

| 模型 | 预训练数据 | 微调数据 | 效果 |
|------|-----------|---------|------|
| ACT | — | **50-100 demos/task** | 桌面操作 80%+ |
| OpenVLA | OXE 970K episodes | 20-50 demos/task | LIBERO 76%（vanilla）|
| π₀ | 7 平台 + 互联网 | — | LIBERO 96.9% |
| LingBot | **20K hrs 真实** | 130 demos/task | GM-100 真机 18% |
| GR00T-N1.7 | 多形态 + 20K hrs 视频 | 5 demos 起步 | — |
| **FAVLA** | — | **260 轨迹** | 真机 80.8%（接触操作）|

> 💡 **经验法则**：
> - 单任务 baseline：**50 demos 起步**（ACT/DP 足够）
> - 多任务泛化：**500+ demos**（需要覆盖变异）
> - 跨形态预训练：**10K+ hours**（LingBot/π₀ 级别）
> - 接触操作：**200+ demos**（力信号稀疏，需要更多覆盖）

### 数据 Scaling Law（有定量证据）

> 参考：[Data Scaling Laws in Imitation Learning for Robotic Manipulation](https://data-scaling-laws.github.io/)（ICLR 2025 Oral）

**关键发现**：
- 泛化性能与环境/物体数量呈**幂律关系**
- **多样性比数量更重要**：10 个环境各 5 条 demo > 1 个环境 50 条 demo
- 4 个采集员 1 个下午的数据 → 2 个任务在新环境/新物体上 ~90% 成功率

| 数据量 | 典型效果 | 参考 |
|--------|---------|------|
| 50 demos/task | 单任务 80%+（ACT baseline） | [ACT 论文](../vla-core/act.md) |
| 260 demos（全任务） | 真机接触操作 80.8%（1× A100, 6hrs） | [FAVLA](../tactile/favla_a_force_adaptive_fast_slow_vla_model_for_contact_rich_dissection.md) |
| 100 demos + 触觉 | 等效 200 demos 纯视觉（2x 数据效率） | [TaF-VLA](../tactile/taf_vla_tactile_force_alignment_2026.md) |
| 1 hr 数据 | 99% 成功率（特定任务，GEN-1 级别） | [GEN-1 报告](https://generalistai.com/blog/apr-02-2026-GEN-1) |
| 20K hrs | 真机 100 任务 18%（GM-100，比 π₀.₅ +7.76%） | [LingBot-VLA](../vla-core/lingbot_vla_pragmatic_foundation_model_2026.md) |

### 数据增强：有效性定量证据

| 增强方法 | 效果（有数字） | 来源 |
|---------|-------------|------|
| **RoboEngine**（背景替换+语义分割） | **+210%** vs 无增强；+20% vs 最佳 baseline | [arXiv:2503.18738](https://arxiv.org/abs/2503.18738)（ICRA 2026） |
| **TraceVLA**（视觉轨迹提示） | 视角变化下 **+20%** | TraceVLA 2025 |
| **VLA²**（Agentic 框架增强） | 困难任务 **+44.2%** 成功率 | VLA² 2025 |
| **自我改进 VLA** | LIBERO 99% · SimplerEnv **+50%** | [Self-Improving VLA](https://wenlixiao.com/self-improve-VLA-PLD/) |
| **深度蒸馏** | RoboTwin +2%（86.5→88.6%）| [LingBot-VLA](../vla-core/lingbot_vla_pragmatic_foundation_model_2026.md) |
| **语言重写**（LLM 生成多种指令） | 语言泛化显著提升（定性） | 多篇 VLA 论文 |

> 💡 **最高 ROI 的增强**：RoboEngine 的背景替换。只需 1 个场景的 demo，就能泛化到 6 个新场景。成本几乎为零。

### 数据采集成本参考

| 方式 | 每条轨迹成本 | 每条时间 | 适用 |
|------|:----------:|:-------:|------|
| 传统遥操作 | $0.42-0.70 | ~50s | 精细操作 |
| **FastUMI Pro** | **<$0.08** | ~10s | 快速采集（5x 效率） |
| ALOHA 双臂 | ~$1 | ~60s | 双臂协同 |
| VR 遥操作 | ~$0.50 | ~45s | 空间操作 |
| 仿真自动生成 | **~$0** | <1s | 大规模补充 |

> DROID 数据集（76K episodes）= 13 个机构 × 18 个月的协调采集。这是"暴力采集"路线。
> 相比之下：4 个人 × 1 个下午 → 2 个任务 90%（Scaling Law 论文）。这是"聪明采集"路线。

---

## 5. 数据模态

### 核心模态

| 模态 | 维度 | 频率 | 必要性 | 说明 |
|------|------|:----:|:------:|------|
| **RGB** | H×W×3 | 30Hz | 必须 | 主视觉输入 |
| **关节角** | N_joints | 100Hz+ | 必须 | 本体感觉 |
| **末端位姿** | 6D (xyz+rpy) | 100Hz+ | 推荐 | FK 计算或直接读 |
| **夹爪状态** | 1 (开/合) | 100Hz+ | 必须 | 抓取动作的关键信号 |
| **深度** | H×W×1 | 30Hz | 推荐 | 3D 感知 · 可蒸馏 |
| **语言指令** | 文本 | 1x/task | 推荐 | 多任务需要 |
| **触觉** | 变化大 | 30-200Hz | 可选 | 接触操作 |

### 多模态同步对齐

```
采集时：
  所有传感器 → 统一时间戳（ROS2 /clock 或 PTP）

存储时：
  高频信号（关节 100Hz）和低频信号（图像 30Hz）分开存
  用时间戳做对齐索引

训练时：
  按图像帧率采样 → 高频信号插值到图像时间戳
  action chunk 从当前帧开始的未来 k 步
```

---

## 6. 开放数据集地图

### ⚠️ 许可证决定了你能做什么——不是所有"开源"数据都能商用

| 许可证 | 学术论文 | 商用 | 修改再分发 | 代表数据集 |
|--------|:-------:|:----:|:---------:|-----------|
| **CC BY 4.0** | ✅ | ✅ | ✅ 需署名 | OXE, DROID, Bridge v2 |
| **MIT** | ✅ | ✅ | ✅ | RoboSet, CALVIN, LIBERO |
| **Apache-2.0** | ✅ | ✅ | ✅ | GM-100, LingBot 数据 |
| **CC BY-NC-SA 4.0** | ✅ | ❌ **禁止商用** | ⚠️ 需相同许可 | **AgiBot World 2026** |
| **CC BY-NC 4.0** | ✅ | ❌ **禁止商用** | ✅ 需署名 | RH20T |
| **未声明** | ⚠️ 风险 | ❌ 默认不可 | ❌ | 部分 HF 上的数据 |

> 🔴 **CC BY-NC = 不能商用。** 在这个许可下训练的模型，法律上也不能商用。AgiBot World 2026 数据量大（1M+ 轨迹）但**禁止商用**——用之前看清许可证。

### 真实机器人操作数据集

| 数据集 | 规模 | 形态 | 模态 | 许可证 | 适用 |
|--------|------|------|------|--------|------|
| **[OXE](https://github.com/google-deepmind/open_x_embodiment)** | 1M+ episodes | 22 形态 | RGB + action | CC BY 4.0 ✅ | 跨形态预训练 · **商用安全** |
| **[OXE-AugE](https://arxiv.org/abs/2512.13100)** | 4.4M trajectories | 扩展 OXE 3x | RGB + action | CC BY 4.0 ✅ | OXE 的增强版 |
| **[DROID](https://droid-dataset.github.io/)** | 76K demos · 350hrs | Franka · 564 场景 | RGB + depth | CC BY 4.0 ✅ | 多环境泛化 · **商用安全** |
| **[Bridge v2](https://rail-berkeley.github.io/bridgedata/)** | 54K trajectories | WidowX · 24 环境 | RGB + action | CC BY 4.0 ✅ | 桌面操作 · **商用安全** |
| **[RoboSet](https://robopen.github.io/roboset/)** | 多任务 · 厨房 | 多种 | RGB × 4 视角 | MIT ✅ | 多任务家庭 · **商用安全** |
| **[RH20T](https://rh20t.github.io/)** | 大规模 · 多技能 | 多种 | RGB + depth + F/T | **CC BY-NC** ⚠️ | 学术研究 · **禁止商用** |
| **[AgiBot World 2026](https://github.com/OpenDriveLab/AgiBot-World)** | **1M+ trajectories** · 2976 hrs | AgiBot G2 | RGB-D + 触觉 + LiDAR + IMU | **CC BY-NC-SA** ⚠️ | 最大多模态 · **禁止商用** |
| **[GM-100](https://huggingface.co/datasets/robbyant/lingbot-GM-100)** | 100 任务 × 3 平台 | 双臂 | RGB + action | Apache-2.0 ✅ | 真机评测 · **商用安全** |
| **[TaF-Dataset](https://arxiv.org/abs/2601.20321)** | 10M 触觉-力配对 | 6 种触觉传感器 | 触觉 + 6 轴 F/T | 未明确 ⚠️ | 触觉预训练 |

### 2025-2026 新发布的大规模数据集

| 数据集 | 规模 | 形态 | 模态 | 许可证 | 亮点 |
|--------|------|------|------|--------|------|
| **[AgiBot World 2026](https://github.com/OpenDriveLab/AgiBot-World)** | **1M+ 轨迹** · 2976 hrs | AgiBot G2 | RGB-D + 触觉 + LiDAR + IMU | **CC BY-NC-SA** ⚠️ | 最大多模态 · 5 个研究方向 · IROS'25 Best Paper 提名 |
| **[RoboMIND 2.0](https://x-humanoid-robomind.github.io/)** | **310K 轨迹** | 6 种形态含人形 | RGB + 本体 + 触觉(12K) + 移动(20K) | 未明确 ⚠️ | 739 任务 · 含 5K 失败案例 · 双臂+灵巧手 · RSS'25 |
| **[OXE-AugE](https://arxiv.org/abs/2512.13100)** | **4.4M 轨迹** | OXE 3x 扩展 | RGB + action | CC BY 4.0 ✅ | OXE 增强版 · **商用安全** |
| **[OmniAction](https://huggingface.co/datasets/OpenMOSS-Team/OmniAction)** | 140K episodes | 多种 | RGB + 音频 + 语音 | 未明确 ⚠️ | 5096 种语音 · 2482 种环境音 · 多模态 |
| **[Humanoid Everyday](https://arxiv.org/abs/2510.08807)** | 10.3K 轨迹 · 3M+ 帧 | 人形 | RGB + depth + LiDAR + 触觉 | 未明确 ⚠️ | 260 任务 · 全传感器人形 |
| **[Hoi!](https://arxiv.org/abs/2512.04884)** | 3048 序列 | 4 种末端 | RGB + 力 + 触觉(Digit) + F/T | 未明确 ⚠️ | 力感知铰接操作 · 381 物体 |
| **[HRDexDB](https://arxiv.org/abs/2604.14944)** | 大规模 | 人手 + 多种机器人手 | 多模态 | 未明确 ⚠️ | 灵巧手抓取 · 多手型对比 |

> 💡 **2025-2026 趋势**：数据集从"RGB + 动作"向**多模态**演进（触觉 + 力 + LiDAR + 音频）。但新数据集的许可证普遍不明确——**用之前一定要确认**。
>
> **HuggingFace 机器人数据集从 2024 年的 1,145 个增长到 2025 年的 26,991 个**——增长 23 倍。

### Benchmark 全景地图（仿真 + 真机）

> ⚠️ **LIBERO-PRO 的警告**：现有模型在 LIBERO 上 90%+，但在轻微扰动（换物体/换初始位置/换指令措辞/换环境）下**跌到 0%**。LIBERO 的高分可能是"记忆"而不是"泛化"。
> → [LIBERO-PRO](https://arxiv.org/abs/2510.03827)

#### 仿真 Benchmark

| Benchmark | 任务数 | 机器人 | 测试什么 | 许可证 | 状态 |
|-----------|:-----:|--------|---------|--------|:----:|
| **[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)** | 4×10 | Franka | 知识迁移 · 终身学习 | MIT ✅ | ⚠️ **饱和** |
| **[LIBERO-PRO](https://arxiv.org/abs/2510.03827)** | LIBERO 扩展 | Franka | **鲁棒性**：扰动下泛化 | MIT ✅ | 2025 新 |
| **[CALVIN](https://github.com/mees/calvin)** | 34 | 桌面 | **长程语言条件** · 技能组合 | MIT ✅ | 活跃 |
| **[RLBench](https://github.com/stepjam/RLBench)** | **100** | Franka | 手工设计多样任务 | MIT ✅ | 经典 |
| **[SimplerEnv](https://simpler-env.github.io/)** | 25 | 多种 | **Sim2Real 可信度验证** | MIT ✅ | 活跃 |
| **[RoboTwin 2.0](https://robotwin-benchmark.github.io/)** | 多任务 | 双臂 | 双臂 · Sim2Real | MIT ✅ | 活跃 |
| **[ManiSkill v3](https://github.com/haosulab/ManiSkill)** | 多种 | 多种 | GPU 加速仿真 | Apache-2.0 ✅ | RSS'25 |
| **[RoboCasa](https://robocasa.ai/)** | 厨房 | 移动操作 | 家庭场景泛化 | MIT ✅ | 活跃 |
| **[BEHAVIOR-1K](https://behavior.stanford.edu/)** | **1000** | 多种 | 开放世界 · 最全面 | 未明确 ⚠️ | 活跃 |
| **[VLABench](https://arxiv.org/abs/2502.09587)** | 大规模 | 多种 | 长程语言条件操作 | 未明确 ⚠️ | ICCV'25 |

#### 真机 Benchmark

| Benchmark | 任务数 | 机器人 | 测试什么 | 许可证 | 为什么重要 |
|-----------|:-----:|--------|---------|--------|-----------|
| **[RoboChallenge Table30](https://robochallenge.ai/)** | **30** | 标准化平台 | 精确 3D 定位 · 遮挡 · 多视角 · 时序依赖 | 未明确 ⚠️ | **首个在线真机评测平台**，可远程提交模型 |
| **[GM-100](https://huggingface.co/datasets/robbyant/lingbot-GM-100)** | **100** | 3 平台（双臂） | 跨平台真机泛化 | Apache-2.0 ✅ | 最多任务的真机 benchmark |
| **[RoCo Challenge](https://arxiv.org/abs/2603.15469)** | 多任务 | 协作双臂 | **协作装配** · 工业场景 | 未明确 ⚠️ | AAAI 2026 竞赛 |

#### 🚨 Benchmark 可信度警告（2025-2026 多篇论文证据）

> **这不是一个警告——是五个。** 每一个都有论文支持。

**警告 1：LIBERO 高分 = 记忆，不是泛化**

> "尽管现有模型在标准 LIBERO 上 90%+，在泛化设定下**跌到 0.0%**。模型依赖的是对训练集动作序列和环境布局的死记硬背，而不是真正的任务理解。"
> — [LIBERO-PRO (arXiv:2510.03827)](https://arxiv.org/abs/2510.03827)

**警告 2：VLA 完全忽略语言指令**

> "模型对语言变化基本不敏感。进一步实验揭示模型倾向于**完全忽略语言指令**。只喂视觉观测（遮蔽语言）的 vision-only 模型达到 44.6%，接近语言条件 baseline 的 47.8%。"
> — [LIBERO-PRO](https://arxiv.org/abs/2510.03827) + [LIBERO-Para (arXiv:2603.28301)](https://arxiv.org/abs/2603.28301)

**含义**：VLA 的 "L"（Language）在很多 benchmark 上可能是摆设。模型靠视觉快捷方式做任务，不需要理解指令。

**警告 3：轻微扰动导致崩溃**

> "模型对摄像头视角和机器人初始状态的扰动极端敏感，成功率从 95% 跌到 **30% 以下**。"
> — [LIBERO-Plus (arXiv:2510.13626)](https://arxiv.org/abs/2510.13626) · [LIBERO-X (arXiv:2602.06556)](https://arxiv.org/abs/2602.06556)

4 种扰动维度全部导致崩溃：
| 扰动 | 标准 LIBERO | 扰动后 |
|------|:----------:|:------:|
| 换物体 | 90%+ | <30% |
| 换初始位置 | 90%+ | ~0% |
| 换指令措辞 | 90%+ | 不变（因为根本不读指令）|
| 换环境 | 90%+ | ~0% |

**警告 4：仿真排名 ≠ 真机排名**

| 模型 | LIBERO (仿真) | 真机 | 跌幅 |
|------|:---:|:---:|:---:|
| π₀.₅ | 96.9% | 52-77% | -20~45 |
| VGA | 98.1% | 58-75% | -23~40 |
| WVA | 99.6% | 75.6% | -24 |

仿真 99.6% 和 96.9% 的差距（2.7%）在真机上完全消失。

**警告 5：真机 benchmark 也有问题**

> RoboChallenge Table30 只有 30 个桌面任务。GM-100 只测了 3 个机器人平台。两者都不包含移动操作、柔软物体、动态环境。真机 benchmark 的覆盖面仍然远远不够。

### 对研究者的建议

1. **不要只报 LIBERO 标准分**——必须同时报 LIBERO-PRO/Plus/X 的扰动结果
2. **做语言消融**——遮蔽语言后如果成功率不变，说明你的模型没用到语言
3. **报真机数字**——仿真 95%+ 的论文，如果没有真机验证，价值存疑
4. **报失败案例**——RoboMIND 2.0 包含了 5K 失败案例数据，这比只报成功率更有价值

---

> 💡 **Benchmark 选型建议**：
> - 快速验证想法 → LIBERO（⚠️ 注意饱和 + LIBERO-PRO 验证鲁棒性）
> - 长程任务 → CALVIN
> - 全面评估 → RLBench（100 任务）或 BEHAVIOR-1K（1000 任务）
> - Sim2Real 可信度 → SimplerEnv
> - **真机排名** → RoboChallenge Table30（在线提交）或 GM-100（本地评测）
> - 双臂 → RoboTwin 2.0 + GM-100
> - 工业协作 → RoCo Challenge

### 数据集选型决策树

```
你要做什么？
│
├─ 学术论文 baseline
│   ├─ 仿真评测 → LIBERO（⚠️饱和）或 CALVIN（长程）
│   └─ 真机评测 → GM-100
│
├─ 跨形态预训练
│   ├─ 可商用 → OXE / OXE-AugE / Bridge v2
│   └─ 仅学术 → + AgiBot World 2026（最大多模态）
│
├─ 多环境泛化
│   └─ DROID（564 场景，CC BY 4.0）
│
├─ 双臂操作
│   └─ GM-100 + RoboTwin 2.0
│
├─ 触觉研究
│   └─ TaF-Dataset（许可证待确认）
│
├─ 多模态（RGB-D + 触觉 + LiDAR）
│   └─ AgiBot World 2026（⚠️ 禁止商用）
│
└─ 商用产品
    └─ **只能用 CC BY / MIT / Apache 的数据**
       OXE + DROID + Bridge + RoboSet + GM-100
```

> 💡 **关键提醒**：
> - **模型继承数据的许可证**。用 CC BY-NC 数据训练的模型，不能商用——即使模型代码是 MIT
> - **混合数据需看最严的**。如果训练数据中有一部分是 CC BY-NC，整个模型都受限
> - **仿真数据通常无限制**。LIBERO/CALVIN/ManiSkill 等仿真数据可自由使用

---

## 7. 仿真数据

### 仿真 vs 真实的权衡

| | 仿真 | 真实 |
|--|------|------|
| **成本** | 几乎为零 | $50-500/小时人工 |
| **规模** | 无限 | 受限于人力 |
| **物理精度** | 接触/摩擦/变形差 | 完美 |
| **视觉真实度** | Domain gap | 完美 |
| **安全** | 无风险 | 可能摔坏 |

### 主流仿真器

| 仿真器 | 维护者 | GPU 加速 | 适用 |
|--------|--------|:-------:|------|
| **Isaac Lab** | NVIDIA | ✅ | GR00T 训练标准 |
| **MuJoCo** | Google | ❌ | 物理精度最高 |
| **RoboCasa** | UT Austin | ✅ | 家庭场景 |
| **RoboGen** | — | ✅ | 自动生成任务 |

### Sim2Real 策略

1. **Domain Randomization**：随机化纹理/光照/物理参数 → 减少 domain gap
2. **仿真预训练 + 真实微调**：大部分数据仿真，少量真实做适配
3. **深度蒸馏**：在仿真中用完美深度训练，部署时用 Depth Anything 估计

→ 详见 [Isaac Lab](../deployment/isaac_lab.md)

---

## 8. 数据管道工程

### 典型流水线

```
采集 → 质量检查 → 格式转换 → 标注 → 增强 → 训练
  │        │          │         │       │
  │        │          │         │       └→ 图像增强 + 语言增强
  │        │          │         └→ 语言指令 + 子任务分解
  │        │          └→ HDF5/MP4 → LeRobot v2
  │        └→ 自动化检查脚本（时戳/亮度/冻结帧）
  └→ 遥操作 + 传感器同步
```

### 版本管理

```bash
# 推荐用 HuggingFace Hub 管理数据集版本
huggingface-cli upload your-org/your-dataset ./local_dataset
# 每次采集后 push 新版本，保留历史
```

---

## 未来扩展方向

> 以下主题将在后续文章中深入展开：

| 主题 | 说明 | 状态 |
|------|------|:----:|
| 多模态传感器融合采集 | 多种传感器的同步采集工程 | 📌 计划中 |
| 数据增强专题 | 图像/动作/语言增强的最佳实践 | 📌 计划中 |
| 数据标注工具链 | 语言标注/子任务分割/奖励标注 | 📌 计划中 |
| 特定形态采集指南 | 双臂/人形/移动操作各自的采集要点 | 📌 计划中 |
| 数据集构建实战 | 从零构建一个 1000-demo 数据集 | 📌 计划中 |

---

## 延伸阅读

| 方向 | 推荐 |
|------|------|
| 数据飞轮 | [数据飞轮与跨模态](data_flywheel_and_cross_modal.md) |
| 开源 VLA（含数据） | [完全开源 VLA 指南](../vla-core/open_source_vla_guide.md) |
| LingBot 20K 小时 | [LingBot-VLA](../vla-core/lingbot_vla_pragmatic_foundation_model_2026.md) |
| GR00T 数据格式 | [GR00T-N1.7](../vla-core/groot_n1_7_nvidia_open_foundation_model_2026.md) |
| 仿真 | [Isaac Lab](../deployment/isaac_lab.md) |
| 触觉数据 | [TaF-VLA](../tactile/taf_vla_tactile_force_alignment_2026.md) · [触觉主线](../tactile/tactile_mainline.md) |
| VLA 数学 | [VLA 数学必备](math_for_vla.md)（含动作表示的数学） |

---

[← Back to Explorer's Map](../README.md)
