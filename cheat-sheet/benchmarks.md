# Benchmark 地图 + 可信度警告

> 2026-04-21 · 🚨 **面试时别只报单一 benchmark 分数**——所有 LIBERO 系列都有记忆化风险

---

## 🚨 首先读这里：6 条 benchmark 可信度警告

### ⚠️ 警告 1 · LIBERO 高分 = 记忆化（不是泛化）

📎 **[LIBERO-PRO (arXiv:2510.03827)](https://arxiv.org/abs/2510.03827)** 实证：
- 标准 LIBERO：**90%+**
- LIBERO-PRO 泛化设定：**0.0%**
- 模型死记硬背动作序列和环境布局，不是真正理解任务

### ⚠️ 警告 2 · VLA 完全忽略语言指令

📎 **[LIBERO-PRO](https://arxiv.org/abs/2510.03827) + [LIBERO-Para (arXiv:2603.28301)](https://arxiv.org/abs/2603.28301)**：
- Vision-only（mask 语言）：**44.6%**
- Language-conditioned：**47.8%**（≈ vision-only！）
- **含义**：VLA 的 "L" 在很多 benchmark 上是摆设

### ⚠️ 警告 3 · 轻微扰动导致崩溃

📎 **[LIBERO-Plus (arXiv:2510.13626)](https://arxiv.org/abs/2510.13626) · [LIBERO-X (arXiv:2602.06556)](https://arxiv.org/abs/2602.06556)**：

| 扰动 | 标准 LIBERO | 扰动后 |
|------|:----------:|:------:|
| 换物体 | 90%+ | <30% |
| 换初始位置 | 90%+ | ~0% |
| 换指令措辞 | 90%+ | 不变（因为不读指令） |
| 换环境 | 90%+ | ~0% |

### ⚠️ 警告 4 · 仿真排名 ≠ 真机排名

| 模型 | LIBERO 仿真 | 真机 | 跌幅 |
|------|:-----------:|:----:|:----:|
| π0.5 | 96.9% | 52-77% | -20~45 |
| VGA | 98.1% | 58-75% | -23~40 |
| WVA | 99.6% | 75.6% | -24 |

仿真 99.6% 和 96.9% 的差距（2.7%）在真机上**完全消失**。

### ⚠️ 警告 5 · 真机 benchmark 也有覆盖缺陷

- **RoboChallenge Table30**：只有 30 个桌面任务
- **GM-100**：只测了 3 个机器人平台
- 都不包含：移动操作 · 柔软物体 · 动态环境

### ⚠️ 警告 6 · 评测泄漏（🧠 作者提出，尚无 VLA 专论）

📎 类比 [How Contaminated Is Your Benchmark (arXiv:2502.00678)](https://arxiv.org/abs/2502.00678)（LLM 领域已成熟议题）：
- OXE 含 22 个子集，其中可能与 LIBERO/CALVIN 有视觉相似性
- HuggingFace "LIBERO demo" 数据集存在，预训练是否过滤取决于实现
- **但是 VLA 领域尚无 pHash / 嵌入空间重叠率的系统研究**

---

## 📋 仿真 Benchmark 总览

| Benchmark | 任务数 | 机器人 | 测什么 | 许可证 | 状态 |
|-----------|:-----:|-------|-------|:------:|:----:|
| **[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)** | 4×10 | Franka | 知识迁移、终身学习 | MIT ✅ | ⚠️ **饱和** |
| **[LIBERO-PRO](https://arxiv.org/abs/2510.03827)** | LIBERO+扰动 | Franka | **鲁棒性** | MIT ✅ | 2025 |
| **[LIBERO-Plus](https://arxiv.org/abs/2510.13626)** | 扰动扩展 | Franka | 更多扰动维度 | MIT ✅ | 2025 |
| **[LIBERO-X](https://arxiv.org/abs/2602.06556)** | 扰动 | Franka | 多维扰动 | MIT ✅ | 2025 |
| **[LIBERO-Para](https://arxiv.org/abs/2603.28301)** | 扰动 | Franka | 语言指令多变 | MIT ✅ | 2026 |
| **[CALVIN](https://github.com/mees/calvin)** | 34 | 桌面 | **长程语言条件** | MIT ✅ | 活跃 |
| **[RLBench](https://github.com/stepjam/RLBench)** | **100** | Franka | 手工多样任务 | MIT ✅ | 经典 |
| **[SimplerEnv](https://simpler-env.github.io/)** | 25 | 多种 | **Sim2Real 可信度验证** | MIT ✅ | 活跃 |
| **[RoboTwin 2.0](https://robotwin-benchmark.github.io/)** | 多任务 | 双臂 | 双臂 · Sim2Real | MIT ✅ | 活跃 |
| **[ManiSkill v3](https://github.com/haosulab/ManiSkill)** | 多种 | 多种 | GPU 加速仿真 | Apache-2.0 ✅ | RSS'25 |
| **[RoboCasa](https://robocasa.ai/)** | 厨房 | 移动操作 | 家庭场景泛化 | MIT ✅ | 活跃 |
| **[BEHAVIOR-1K](https://behavior.stanford.edu/)** | **1000** | 多种 | 开放世界最全面 | 未明确 ⚠️ | 活跃 |
| **[VLABench](https://arxiv.org/abs/2502.09587)** | 大规模 | 多种 | 长程语言条件 | 未明确 ⚠️ | ICCV'25 |

## 🤖 真机 Benchmark

| Benchmark | 任务数 | 机器人 | 许可证 | 为什么重要 |
|-----------|:-----:|-------|:------:|-----------|
| **[RoboChallenge Table30](https://robochallenge.ai/)** | **30** | 标准化平台 | 未明确 ⚠️ | **首个在线真机评测**，可远程提交 |
| **[GM-100](https://huggingface.co/datasets/robbyant/lingbot-GM-100)** | **100** | 3 平台 | Apache-2.0 ✅ | 最多任务的真机 benchmark |
| **[RoCo Challenge](https://arxiv.org/abs/2603.15469)** | 多任务 | 协作双臂 | 未明确 ⚠️ | **工业装配** · AAAI 2026 |

---

## 🎯 Benchmark 选型决策树

```
你要做什么？
│
├─ 快速验证想法（1 天内）
│   └─ LIBERO（⚠️ 必须报 PRO 对照）
│
├─ 长程语言任务
│   └─ CALVIN（34 任务）
│
├─ 全面评估能力
│   ├─ 100 任务 → RLBench
│   └─ 1000 任务 → BEHAVIOR-1K
│
├─ Sim2Real 可信度
│   └─ SimplerEnv
│
├─ 真机排名（可 reproducible）
│   ├─ 在线提交 → RoboChallenge Table30
│   └─ 本地测 → GM-100
│
├─ 双臂
│   └─ RoboTwin 2.0 + GM-100
│
├─ 工业装配
│   └─ RoCo Challenge
│
└─ 语义安全
    └─ HazardArena（📎 2026 新，填补安全评估缺口）
```

---

## 💡 Benchmark 报告的"黄金三件套"

被问成绩时，**必须同时给出**：

1. **主 benchmark 分数**（LIBERO / CALVIN / etc.）
2. **鲁棒性对照**（LIBERO-PRO / Plus / X）
3. **真机验证**（≥1 个真机 benchmark）

只给第 1 个 = 暴露不专业。

---

## 🏆 对研究者的 5 条硬建议

1. ❌ **不要只报 LIBERO 标准分**——必须同时报 LIBERO-PRO/Plus/X 扰动
2. ❌ **不要省略语言消融实验**——遮蔽语言后如果成功率不变，说明模型没用到语言
3. ✅ **必须报真机数字**——仿真 95%+ 的论文如果没有真机验证，价值存疑
4. ✅ **报失败案例**——RoboMIND 2.0 包含 5K 失败案例数据，这比只报成功率更有价值
5. ✅ **披露数据混合的来源**——预训练语料子集构成要可审计，否则无法判断是否泄漏

---

## ❗ 面试中的对话范例

**Bad**：
> "我们在 LIBERO 上做到 97.2%"

**Good**：
> "LIBERO 标准 97.2%，LIBERO-PRO 泛化设定 **32.5%**——这个差距告诉我们模型学到了一部分模式，但在新物体/新位置下仍然脆弱。我们的下一步是对 **vision-only baseline** 做消融，确保模型真的在用语言。"

---

## 📚 延伸阅读

- [VLA 数据工程指南](../theory/foundation/vla_data_engineering_guide.md) · 全景 + 6 条警告详解
- [失效模式 F1-F6](./failure-modes.md) · 真机出问题时怎么定位
- [开源审计 / license](./open-source-audit.md) · 数据继承风险

---

[← Back to Cheat Sheet](./README.md)
