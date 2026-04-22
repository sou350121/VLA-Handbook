# VLA 数据集速查（2023-2026 · 25+ 数据集）

> 2026-04-21 · 🚨 **数据集许可证会继承到模型**——用 CC BY-NC 数据训练的模型不能商用，**哪怕你的代码是 MIT**

---

## 🚨 首先读：许可证继承规则

| 许可证 | 学术论文 | 商用 | 修改再分发 | 关键含义 |
|--------|:------:|:----:|:---------:|---------|
| **CC BY 4.0** | ✅ | ✅ | ✅ 需署名 | **商用安全** |
| **MIT** | ✅ | ✅ | ✅ | **商用安全** |
| **Apache-2.0** | ✅ | ✅ | ✅ | **商用安全** |
| **CC BY-NC-SA 4.0** | ✅ | ❌ | ⚠️ 需同许可 | **禁商用** · 衍生品也必须 NC-SA |
| **CC BY-NC 4.0** | ✅ | ❌ | ✅ 需署名 | **禁商用** |
| **未声明** | ⚠️ | ❌ | ❌ | 默认 all rights reserved |

### 🔴 关键规则（很多团队会翻车）

1. **模型继承数据许可**：CC BY-NC 数据 → 模型不能商用
2. **混合时看最严的**：如果 1% 训练数据是 NC → 整个模型都 NC
3. **仿真数据通常无限制**：LIBERO / CALVIN / ManiSkill 等可自由使用
4. **AgiBot World 2026** 数据大到诱人（1M+ 轨迹）但 **CC BY-NC-SA** 禁商用
5. **未声明 ≠ 可以随便用**：默认 all rights reserved，需主动联系获取许可

---

## 📊 一、经典基础层数据集（2023-2024）

### 跨形态预训练数据

| 数据集 | 规模 | 形态 | 模态 | 许可 | 用途 |
|--------|------|------|------|:----:|------|
| **[OXE (Open X-Embodiment)](https://github.com/google-deepmind/open_x_embodiment)** | 1M+ episodes · 22 子集 | 22 种形态 | RGB + action | CC BY 4.0 ✅ | **跨形态预训练标配** · 商用安全 |
| **[OXE-AugE (arXiv:2512.13100)](https://arxiv.org/abs/2512.13100)** | **4.4M 轨迹** | OXE 3× 扩展 | RGB + action | CC BY 4.0 ✅ | OXE 增强版 · 2026 · 商用 |
| **[DROID](https://droid-dataset.github.io/)** | 76K demos · 350 hrs | Franka · **564 场景** | RGB + depth | CC BY 4.0 ✅ | **多环境泛化**（13 机构 × 18 个月）· 商用 |
| **[Bridge v2](https://rail-berkeley.github.io/bridgedata/)** | 54K trajectories | WidowX · 24 环境 | RGB + action | CC BY 4.0 ✅ | 桌面操作经典 · 商用 |
| **[RoboSet](https://robopen.github.io/roboset/)** | 多任务 · 厨房 | 多种 | RGB × 4 视角 | MIT ✅ | 多任务家庭 · 商用 |

### 学术专用（禁商用）

| 数据集 | 规模 | 形态 | 模态 | 许可 | 适用 |
|--------|------|------|------|:----:|------|
| **[RH20T](https://rh20t.github.io/)** | 大规模 · 多技能 | 多种 | RGB + depth + F/T | **CC BY-NC** ⚠️ | 纯学术研究 |

### 真机评测数据

| 数据集 | 规模 | 形态 | 模态 | 许可 | 用途 |
|--------|------|------|------|:----:|------|
| **[GM-100](https://huggingface.co/datasets/robbyant/lingbot-GM-100)** | 100 任务 × 3 平台 | 双臂 | RGB + action | Apache-2.0 ✅ | **真机评测** · 商用 |

### 触觉专用

| 数据集 | 规模 | 传感器 | 许可 | 用途 |
|--------|------|-------|:----:|------|
| **[TaF-Dataset (arXiv:2601.20321)](https://arxiv.org/abs/2601.20321)** | 10M 触觉-力配对 | 6 种触觉 | 未明确 ⚠️ | 触觉跨传感器预训练 |

---

## 🚀 二、2025-2026 新发布的大规模数据集

| 数据集 | 规模 | 形态 | 模态 | 许可 | 亮点 |
|--------|------|------|------|:----:|------|
| **[AgiBot World 2026](https://github.com/OpenDriveLab/AgiBot-World)** | **1M+ 轨迹 · 2976 hrs** | AgiBot G2 | RGB-D + 触觉 + LiDAR + IMU | **CC BY-NC-SA** ⚠️ | 最大多模态 · IROS'25 Best Paper 提名 · **禁商用** |
| **[RoboMIND 2.0](https://x-humanoid-robomind.github.io/)** | **310K 轨迹** | 6 形态含人形 | RGB + 本体 + 触觉(12K) + 移动(20K) | 未明确 ⚠️ | **739 任务** · 含 **5K 失败案例** · 双臂+灵巧手 · RSS'25 |
| **[OmniAction](https://huggingface.co/datasets/OpenMOSS-Team/OmniAction)** | 140K episodes | 多种 | RGB + 音频 + 语音 | 未明确 ⚠️ | 5096 种语音 · 2482 种环境音 · 多模态音频 |
| **[Humanoid Everyday](https://arxiv.org/abs/2510.08807)** | 10.3K 轨迹 · 3M+ 帧 | 人形 | RGB + depth + LiDAR + 触觉 | 未明确 ⚠️ | **260 任务** · 全传感器人形 |
| **[Hoi!](https://arxiv.org/abs/2512.04884)** | 3048 序列 | 4 种末端 | RGB + 力 + 触觉(Digit) + F/T | 未明确 ⚠️ | **力感知铰接操作** · 381 物体 |
| **[HRDexDB](https://arxiv.org/abs/2604.14944)** | 大规模 | 人手 + 多种机器人手 | 多模态 | 未明确 ⚠️ | 灵巧手抓取 · 多手型对比 |

---

## 🧠 三、人类视频 / Egocentric 数据集（智能方向核心）

### 📎 为什么重要
EgoScale 已用实证（R²=0.9983 log-linear scaling）证明：**人类视频是 VLA 预训练的有效监督信号**。详见 [Danfei Xu 访谈](../theory/foundation/human_data_sensorimotor_ghost_danfei_xu_interview_2026.md)。

| 数据集 | 规模 | 采集设备 | 模态 | 许可 | 亮点 |
|--------|------|---------|------|:----:|------|
| **[EgoScale](https://arxiv.org/abs/2602.16710)** ⭐ | **20,854 小时** | MANUS 手套 + 头戴相机 | 自中心 RGB + 手部动作 | 未明确 ⚠️ | 📎 **log-linear scaling (R²=0.9983)** · 真机 +54% · NVIDIA GEAR |
| **[Ego4D](https://ego4d-data.org/)** | 3,670 小时 · 9 国 | 多种头戴相机 | RGB + 音频 + 文本 | 自定义 ⚠️ | 最大人类活动数据 · Meta · 需协议 |
| **Project Aria Everyday Activities** | 140+ 小时（采样公开） | **Project Aria** 眼镜 | 多相机 + IMU + SLAM | Meta 许可 ⚠️ | Aria 技术栈支撑的数据 |
| **[EgoDex](https://arxiv.org/abs/2403.18906)** | 灵巧操作子集 | 头戴 + 手部追踪 | 第一视角 + 灵巧手动作 | 未明确 ⚠️ | 灵巧操作专用 |
| **Xperience-10M** ⭐ | **10M 样本** | 多样采集 | 第一视角视频 | 未明确 ⚠️ | 2025 · 规模最大人类视频集之一 |

---

## 🏛️ 四、Benchmark 配套数据集

### 用于 benchmark 训练的数据（不用自采）

| Benchmark | 自带数据 | 规模 | 用途 |
|-----------|---------|------|------|
| **LIBERO** | 130 任务 demos | ~1.3K/task | LIBERO 训练 |
| **CALVIN** | 24 小时 teleop | 34 任务 | CALVIN 训练 |
| **BEHAVIOR-1K** | **10,000 demos · 1,200+ 小时** JoyLo 遥操 | 50 Challenge 任务 | NeurIPS 2025 Challenge · Stanford 申请 |
| **RLBench** | 自动生成 demos | 100 任务 | RLBench 训练 |
| **VLABench** | 大规模 | 多任务 | VLABench 训练 |

---

## 🎯 数据集选型决策树

```
你要做什么？
│
├─ 跨形态基础模型预训练（你想做 π0 / GR00T 类）
│   ├─ 可商用 → OXE 或 OXE-AugE（4.4M 轨迹，CC BY 4.0）
│   ├─ 学术最大 → OXE + AgiBot World 2026（⚠️ NC-SA）
│   └─ 人类视频路线 → EgoScale 方法 + Ego4D
│
├─ 单一机器人微调（你有目标机器人）
│   ├─ 通用预训模型 + 50-500 demos 自采
│   └─ 找相近形态数据 → DROID（Franka）· Bridge（WidowX）
│
├─ 多环境泛化
│   └─ DROID（564 场景，CC BY 4.0）
│
├─ 双臂操作
│   └─ GM-100（100 任务 × 3 平台）· AgiBot World
│
├─ 人形 + 全身
│   ├─ 商业可行 → 等更开放的人形数据集
│   └─ 学术 → Humanoid Everyday · RoboMIND 2.0
│
├─ 触觉 / 力感知
│   ├─ 跨传感器 → TaF-Dataset
│   ├─ 力感知铰接 → Hoi!
│   └─ 灵巧手 → HRDexDB · DexYCB
│
├─ 长程家务 / 真实场景
│   ├─ 顶配 → BEHAVIOR-1K 10K 遥操数据（Stanford 申请）
│   └─ 仿真自采 → OmniGibson 直接 JoyLo
│
├─ 人类视频预训练
│   ├─ 最大规模 → EgoScale 路线（20K+ 小时）
│   ├─ 最成熟 → Ego4D（需协议）
│   └─ 高质量单位 → Project Aria 数据
│
├─ 多模态（含音频）
│   └─ OmniAction（RGB + 音频 + 语音）
│
├─ 失败案例 / F6 防御（见 [failure-modes](./failure-modes.md)）
│   └─ RoboMIND 2.0 含 **5K 失败轨迹**（📎 刻意收集）
│
└─ 纯商用产品（严格法律合规）
    └─ **只用** CC BY 4.0 / MIT / Apache：OXE + OXE-AugE + DROID + Bridge + RoboSet + GM-100
```

---

## 📋 许可证深度对照表

| 数据集 | 总轨迹/时长 | 许可证 | 可商用 | 可修改 |
|--------|------------|--------|:------:|:------:|
| OXE | 1M+ episodes | CC BY 4.0 | ✅ | ✅ |
| OXE-AugE | 4.4M | CC BY 4.0 | ✅ | ✅ |
| DROID | 76K / 350 hrs | CC BY 4.0 | ✅ | ✅ |
| Bridge v2 | 54K | CC BY 4.0 | ✅ | ✅ |
| RoboSet | 多任务 | MIT | ✅ | ✅ |
| GM-100 | 100 任务 | Apache-2.0 | ✅ | ✅ |
| RH20T | 大规模 | **CC BY-NC** | ❌ | ✅ |
| AgiBot World 2026 | **1M+ / 2976 hrs** | **CC BY-NC-SA** | ❌ | ⚠️ 需同许可 |
| RoboMIND 2.0 | 310K | 未明确 | ⚠️ | ⚠️ |
| TaF-Dataset | 10M 对 | 未明确 | ⚠️ | ⚠️ |
| Hoi! | 3048 序列 | 未明确 | ⚠️ | ⚠️ |
| HRDexDB | 大规模 | 未明确 | ⚠️ | ⚠️ |
| OmniAction | 140K | 未明确 | ⚠️ | ⚠️ |
| Humanoid Everyday | 10.3K | 未明确 | ⚠️ | ⚠️ |
| EgoScale | 20,854 hrs | 未明确 | ⚠️ | ⚠️ |
| Ego4D | 3,670 hrs | 自定义 | ⚠️ 需协议 | ⚠️ 需协议 |
| BEHAVIOR-1K demos | 10K / 1,200+ hrs | 未明确 | ⚠️ 需申请 | ⚠️ |

---

## 💡 数据集组合推荐（按场景）

### 研究型：最大化规模和多样性
```
OXE-AugE (4.4M) + AgiBot World 2026 (1M+) + RoboMIND 2.0 (310K) + EgoScale 方法
→ 跨形态 + 多模态 + 人类视频全覆盖
→ ⚠️ 最终模型 CC BY-NC-SA 禁商用
```

### 商用产品：严格合规
```
OXE + DROID + Bridge v2 + RoboSet + GM-100
→ 全 CC BY 4.0 / MIT / Apache
→ 模型可商用，只需署名
→ 约 1.5M+ 干净可商用轨迹
```

### 快速原型（少量精准）
```
ACT / Diffusion Policy 50-100 demos 自采 → 跑通
+ OpenVLA 预训权重 → fine-tune
```

### 接触操作研究
```
TaF-Dataset（预训）+ Hoi!（力铰接）+ 自采目标任务 demos
→ 📎 参考 FAVLA 路线：260 demos → 80.8% 接触操作成功率
```

---

## 🚩 数据集 Red Flag（论文声明存疑）

| 论文说… | 你追问… |
|---------|--------|
| "预训练 XX 万小时" | 📎 **数据混合 license 分布？哪些子集？能商用吗？** |
| "我们用 OXE" | "用了哪 22 个子集的子集？数据预处理细节？" |
| "模型开源 · MIT" | 📎 "训练数据的 license？CC BY-NC 混进来的话模型仍然受限" |
| "跨形态泛化" | "embodiment 的分布？哪些形态？实验 ablation？" |
| "人类视频预训练" | 📎 "用的是 Ego4D? EgoScale? 有 Meta 协议吗？" |
| "失败案例也训了" | 📎 "数量 · 采集方法 · 占总数据百分比？" |

---

## 📊 数据经济学对比

📎 参考 [Danfei Xu 访谈](../theory/foundation/human_data_sensorimotor_ghost_danfei_xu_interview_2026.md)：

| 数据来源 | 每小时成本 | 扩展上限 | 形态适配 | 失败案例 |
|---------|:---------:|:-------:|:-------:|:-------:|
| 遥操作（机器人）| $50-500 | 受机器人数限制 | ✅ 直接 | ✅ 可主动采 |
| 人类第一视角视频 | $5-50 | 理论上无限 | ❌ 需 retarget | ❌ 几乎没有 |
| 仿真自动生成 | ~$0 | 无限 | ⚠️ Sim2Real gap | ✅ 可控 |
| UMI / FastUMI | $0.08 (FastUMI Pro) | 中等 | 与机器人对齐 | ⚠️ |

---

## 📎 2025-2026 趋势观察

1. **多模态爆发**：从"RGB + 动作"向 RGB-D + 触觉 + LiDAR + 音频演进（AgiBot World, Humanoid Everyday, OmniAction 都是多模态）
2. **人类数据正规化**：EgoScale 之后，人类视频从"实验性"变成"一级公民"
3. **HuggingFace 机器人数据集**：📎 2024 年 1,145 个 → 2025 年 26,991 个 · **增长 23 倍**
4. **Egocentric data 公司涌现**：📎 Danfei 观察——50-60 家专门采集第一视角数据的公司
5. **许可证日益严格**：2025 后新数据集大量 NC-SA（AgiBot）或未明确，**商用路径变窄**

---

## 📚 延伸阅读

- [VLA 数据工程指南](../theory/foundation/vla_data_engineering_guide.md) · 完整 §6 开放数据集地图 + 3 层防御
- [Danfei Xu 访谈](../theory/foundation/human_data_sensorimotor_ghost_danfei_xu_interview_2026.md) · 人类数据的哲学视角
- [Benchmark 地图](./benchmarks.md) · 对应的评测基准
- [失效模式 F6](./failure-modes.md) · 为什么需要失败案例数据

---

[← Back to Cheat Sheet](./README.md)
