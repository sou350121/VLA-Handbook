# 仿真基准与训练工具：LIBERO、RLinf、SimpleVLA-RL (Simulation Benchmarks & Tooling)

> **定位**：把“仿真环境/评测基准/训练基础设施”串成一条可落地的 VLA+RL 工具链。  
> **覆盖**：LIBERO（任务基准）、RLinf（训练 infra）、SimpleVLA-RL（RL 框架）。

这三者不是同一类东西：**LIBERO 是评测任务集**，**RLinf 是训练基础设施**，**SimpleVLA-RL 是面向 VLA 的 RL 训练框架**。理解它们的分工，才能搭出稳定、可复现的训练/评测流水线。

**一手来源**：
- LIBERO：GitHub `https://github.com/Lifelong-Robot-Learning/LIBERO` / 论文 `https://arxiv.org/abs/2306.03310` / 项目页 `https://libero-project.github.io/main.html`
- RLinf：GitHub `https://github.com/RLinf/RLinf` / 文档 `https://rlinf.readthedocs.io/en/latest/` / 论文 `https://arxiv.org/abs/2510.06710`
- SimpleVLA-RL：GitHub `https://github.com/PRIME-RL/SimpleVLA-RL` / 论文 `https://arxiv.org/abs/2509.09674` / OpenReview `https://openreview.net/forum?id=TQhSodCM4r`

---

## 0. 先把结论讲清楚（1 分钟版）
- **LIBERO**：终身学习/多任务泛化的标准操控基准，适合做 VLA 的“回归测试集”。  
- **RLinf**：把 rollout/训练/评估做成“生产线”，解决吞吐、可复现与系统级不稳定。  
- **SimpleVLA-RL**：面向 VLA 的 RL 训练框架，强调大规模并行与高效采样（论文口径）。  
- **组合使用**：LIBERO 选任务 + RLinf 搭基础设施 + SimpleVLA-RL 跑策略改进，是高效路径。

---

## 1. 快速对比：它们分别解决什么问题

| 维度 | LIBERO | RLinf | SimpleVLA-RL |
|---|---|---|---|
| 角色 | 评测基准 / 任务套件 | 训练基础设施 | RL 训练框架 |
| 目标 | 测试终身学习与泛化 | 稳定大规模训练 | 提升 VLA 策略表现 |
| 输入 | 任务定义 + 示范数据 | 环境/模型/调度 | VLA 策略 + 环境 |
| 输出 | 成功率/迁移能力 | 可复现训练流水线 | RL 微调后的策略 |
| 典型关注 | 套件划分/评测协议 | 吞吐/回归/稳定性 | 采样/损失/并行 |

---

## 2. LIBERO：终身学习与泛化评测基准

### 2.1 套件与任务结构
LIBERO 提供多套任务，覆盖空间变化、物体变化与目标变化，适合做分布偏移评测。  
经典套件包括 **Spatial/Object/Goal** 与 **LIBERO-100**（论文口径，[Paper](https://arxiv.org/abs/2306.03310)）。

| 套件 | 任务数 | 迁移挑战 | 说明 |
|---|---|---|---|
| LIBERO-Spatial | 30 | 空间关系 | 同物体，不同摆放 |
| LIBERO-Object | 30 | 物体变化 | 语义+抓取泛化 |
| LIBERO-Goal | 30 | 目标变化 | 同场景，多目标 |
| LIBERO-100 | 100 | 混合 | 终身学习难度更高 |

### 2.2 评测关注点
- **Forward transfer**：新任务学习速度  
- **Backward transfer**：旧任务遗忘程度  
- **Success rate**：每个任务的完成率  
这三者是判断“是否真的泛化”的核心指标。

### 2.3 与 VLA 的结合方式
1) 用 LIBERO-90 预训练或微调 Action Head  
2) 在 LIBERO-10 / Object / Goal 做回归测试  
3) 把“任务套件 + 评测协议”当成流水线的固定合同

**常见坑**：
- **no-ops/子版本不一致** → 指标不可比  
- **窗口长度/控制频率不对齐** → 成功率虚高或虚低  
- **任务顺序未固定** → 结果不可复现

---

## 3. RLinf：把 RL 训练做成可扩展生产线

### 3.1 核心价值
RLinf 不是新算法，而是**训练基础设施**：把 rollout、数据面、训练调度、评估回归做成可复用组件（见 [RLinf GitHub](https://github.com/RLinf/RLinf) 与 [Docs](https://rlinf.readthedocs.io/en/latest/)）。

### 3.2 典型管线组件

```
Env pool (sim) ──> Rollout workers ──> Buffer/Stats ──> Trainer ──> Eval
                      ▲                               │
                      └────────────── scheduler ──────┘
```

**工程含义**：
- **Rollout 吞吐优先**：并行环境 + 统一采样接口  
- **评测协议固化**：种子、初始分布、成功判定  
- **可复现**：同配置重复跑出可对比曲线

### 3.3 跟 VLA 的对齐方式
- **Observation contract**：RGB/深度/触觉/语言的时间窗  
- **Action contract**：输出频率 vs 控制频率（chunk + 插值/滤波）  
- **Reward/Safety contract**：仿真 reward 与真机 safety 分离

### 3.4 常见坑
- rollout 频率与控制频率不一致  
- 评测协议未固定导致结果漂移  
- 训练/评测在不同环境或不同 seed 上“比苹果和橘子”

---

## 4. SimpleVLA-RL：面向 VLA 的 RL 训练框架

### 4.1 定位与目标
SimpleVLA-RL 聚焦于**把 VLA 的 RL 训练规模化**，解决数据昂贵与泛化不足问题（论文口径，[arXiv](https://arxiv.org/abs/2509.09674)）。

### 4.2 技术亮点（论文口径）
- 基于现有 RL 训练 infra（论文提到 veRL 体系）  
- 多环境并行渲染与高效采样  
- 针对 VLA 的轨迹采样与损失计算优化  
- 采用 Group Relative Policy Optimization / 扩展 PPO clipping 等策略

### 4.3 训练流程（简化版）
1) **BC warmstart**：先用示范数据让策略“能动”  
2) **RL fine-tune**：在仿真中做大规模 rollout  
3) **评测回归**：在 LIBERO 或 RoboTwin 上跑固定协议（论文口径）

### 4.4 与 LIBERO / RLinf 的关系
- **LIBERO**：作为评测基准与回归测试集  
- **RLinf**：作为训练基础设施  
- **SimpleVLA-RL**：作为 RL 训练算法/框架

---

## 5. 组合使用：一条可落地的工具链

```
选任务基准 (LIBERO)
   └─ 固定评测协议与套件
搭训练 infra (RLinf)
   └─ rollout / buffer / eval 可复现
选 RL 框架 (SimpleVLA-RL)
   └─ BC warmstart + RL fine-tune
回到 LIBERO 做回归测试
   └─ 再挑小规模真机验证
```

---

## 6. 实战 checklist
- [ ] 明确任务套件与评测协议（LIBERO 版本/套件固定）  
- [ ] 明确观测/动作 contract（频率/窗口长度）  
- [ ] rollout 吞吐稳定（并行环境不掉帧）  
- [ ] 评测回归可重复（固定 seed + 初始分布）  
- [ ] 仿真 reward 与真机 safety 分离  
- [ ] 小规模真机验证用于校准模型偏差

---

## 参考链接
- LIBERO GitHub：`https://github.com/Lifelong-Robot-Learning/LIBERO`  
- LIBERO Paper：`https://arxiv.org/abs/2306.03310`  
- RLinf GitHub：`https://github.com/RLinf/RLinf`  
- RLinf Docs：`https://rlinf.readthedocs.io/en/latest/`  
- RLinf-VLA Paper：`https://arxiv.org/abs/2510.06710`  
- SimpleVLA-RL GitHub：`https://github.com/PRIME-RL/SimpleVLA-RL`  
- SimpleVLA-RL Paper：`https://arxiv.org/abs/2509.09674`  
- OpenReview：`https://openreview.net/forum?id=TQhSodCM4r`

---
[← Back to Deployment](./README.md)
