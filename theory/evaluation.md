# 评估体系详解 (Evaluation Protocols Deep Dive)

在 VLA 领域，"怎么算做好了"是一个极其复杂的问题。与 CV/NLP 不同，机器人没有静态的 Test Set，必须在动态环境中评估。本章深入探讨评估的数学定义、基准细节及实战协议。

## 1. 核心评估指标 (Core Metrics)

### 1.1. Success Rate (SR) - 成功率
最直观但也最粗糙的指标。

$$
SR = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(\text{task}_i \text{ completed})

$$
- **定义**: $N$ 次尝试中成功的次数。
- **置信区间 (Confidence Interval)**: 由于 $N$ 通常较小 (真机实验昂贵)，SR 的方差很大。建议使用 **Wald Interval** 或 **Wilson Score Interval** 报告误差范围。

$$
\hat{p} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{N}}

$$
(当 $N=20, \hat{p}=0.5$ 时，误差范围高达 $\pm 22\%$！所以真机评估至少要跑 50 次以上才具备统计意义。)

### 1.2. Mean Steps to Success (MSS) - 平均成功步数
衡量效率的指标。
- **定义**: 在成功的 Episodes 中，平均消耗的时间步数。
- **意义**: 两个模型 SR 都是 100%，但模型 A 用了 5 秒，模型 B 用了 10 秒，显然 A 更好 (更流畅，犹豫更少)。

### 1.3. Intervention Rate (IR) - 干预率
适用于长程任务 (Long-horizon) 或自动驾驶。

$$
IR = \frac{\text{Total Interventions}}{\text{Total Operation Time (hours)}} \quad \text{or} \quad \frac{\text{Interventions}}{\text{Total Steps}}

$$
- **定义**: 人类专家为了防止灾难性后果 (碰撞、掉落) 而接管机器人的频率。
- **MPI (Miles Per Intervention)**: 自动驾驶常用，VLA 中对应 **Steps Per Intervention (SPI)**。

### 1.4. Executable Rate (ER) - 可执行率
衡量模型的**安全性**和**运动学一致性**。
- **定义**: 模型生成的 Action 能够被底层控制器 (IK Solver / Impedance Controller) 执行的比例。
- **常见失败**:
    - **IK No Solution**: 目标点超出工作空间。
    - **Self-Collision**: 机械臂自碰撞。
    - **Singularity**: 奇异点。

---

## 2. 仿真基准深度解析 (Simulation Benchmarks)

### 2.1. CALVIN (Computer Vision and Language for Interaction)
> **核心能力**: **Long-horizon Language Following** (长序列语义跟随)。

- **任务**: 连续完成 5 个指令 (e.g., "Open drawer" -> "Pick up block" -> "Place in drawer" -> "Turn on light" -> "Close drawer")。
- **评估方式**: **Chain Success Rate**。
    - 1-step SR: 完成第 1 个任务的概率。
    - 5-step SR: 连续完成 5 个任务的概率 (极难，SOTA 通常 < 10%)。
- **意义**: 测试 VLA 模型的**状态保持能力** (Stateful) 和**上下文理解能力**。

### 2.2. SIMPLER (Sim-to-Real Evaluation)
> **核心能力**: **Sim-to-Real Correlation** (仿真与真机的相关性)。

- **痛点**: 以前我们在仿真里跑分高，真机一塌糊涂。
- **创新**: SIMPLER 使用真实视频作为纹理，并精细调节物理参数，使得仿真中的排名 (Rank) 与真机排名高度正相关 (Pearson Correlation > 0.8)。
- **用途**: 在上真机前，先用 SIMPLER 筛选 Checkpoint，节省昂贵的真机测试时间。

### 2.3. ManiSkill 2/3
> **核心能力**: **Generalizable Manipulation** (泛化抓取)。

- **引擎**: SAPIEN (PhysX)。
- **特点**: 包含 2000+ 种不同的物体 (PartNet Mobility)，每个物体都有不同的拓扑结构 (不同的门把手、不同的水龙头)。
- **评估**: 测试模型在 **Unseen Object Instances** 上的 Zero-shot 成功率。

---

## 3. 真机评估协议 (Real World Protocols)

真机评估是“玄学”的重灾区。为了保证公平，必须遵循严格的协议。

### 3.1. 变量控制 (Variable Control)
- **物体位置**:
    - **Fixed**: 每次都放在完全相同的位置 (测试记忆能力)。
    - **Randomized**: 在 $10cm \times 10cm$ 的区域内随机放置 (测试泛化能力)。
- **干扰项 (Distractors)**:
    - 场景中是否包含与任务无关的物体？(测试抗干扰能力)。
- **光照**:
    - 固定光源 vs 自然光变化。

### 3.2. A/B Testing
在真机上对比模型 A 和 B 时，必须交替进行 (Interleaved)，以消除环境随时间变化 (e.g., 电机发热、光照变化) 的影响。
- **错误做法**: 上午测模型 A (50次)，下午测模型 B (50次)。
- **正确做法**: A, B, A, B, ... 交替测试。

### 3.3. Reset-Free Evaluation
- **定义**: 机器人完成任务后，自动执行“复位”动作，或者下一个任务的初始状态就是上一个任务的结束状态。
- **意义**: 实现 24/7 无人值守的自动化评估 (Scale Up Evaluation)。

---

## 4. 模型选择策略 (Checkpoint Selection)

在 VLA 训练中，Loss 不代表一切。

### 4.1. 为什么 Loss 失效？
- **多模态分布**: 比如面对障碍物，向左走和向右走都是对的。MSE Loss 会让模型走中间 (撞墙)，Loss 可能不降反升，但策略其实变好了 (学会了多模态)。
- **过拟合**: 模型可能记住了训练集里的特定噪声。

### 4.2. EMA (Exponential Moving Average)
- **策略**: 维护一份模型权重的滑动平均版本。

$$
\theta_{EMA} = \alpha \theta_{EMA} + (1-\alpha) \theta_{current}

$$
- **作用**: 极大地稳定了评估时的表现，平滑了训练过程中的震荡。**所有 SOTA 模型 (RT-2, Octo, Pi0) 评估时用的都是 EMA 权重，而不是当前权重**。

### 4.3. 最佳实践
1. 每 5000 Steps 保存一个 Checkpoint。
2. 使用 SIMPLER 或 CALVIN 进行并行评估。
3. 选取仿真 SR 最高的 Top-3 Checkpoint。
4. 在真机上对这 Top-3 进行小样本 (N=10) 快速筛选。
5. 选定最佳模型进行大规模 (N=50) 测试。

---

## 5. Evaluation Pipeline 构建 (Building Evaluation Pipeline)

### 5.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                   Evaluation Pipeline 架构                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌────────┐│
│   │   Data    │───▶│   Model   │───▶│  Metrics  │───▶│ Report ││
│   │  Loader   │    │ Inference │    │ Compute   │    │ & Log  ││
│   └───────────┘    └───────────┘    └───────────┘    └────────┘│
│        │                │                │               │      │
│        ▼                ▼                ▼               ▼      │
│   RLDS/HDF5        Batch/Stream     SR/MSS/IR      W&B/TB      │
│   CALVIN/SIMPLER   Multi-ckpt       Confidence     Artifacts   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 关键组件

```python
class EvaluationPipeline:
    """VLA 评估流水线"""
    
    def __init__(self, config):
        # 1. 数据管理
        self.test_set = load_test_set(config.benchmark)  # CALVIN, SIMPLER, etc.
        
        # 2. 模型加载
        self.model = load_model(config.checkpoint)
        
        # 3. 指标计算器
        self.metrics = {
            'success_rate': SuccessRateMetric(),
            'mean_steps': MeanStepsMetric(),
            'intervention_rate': InterventionRateMetric(),
        }
        
        # 4. 日志记录
        self.logger = WandBLogger(project=config.project)
    
    def run_episode(self, task):
        """运行单个 Episode"""
        obs = self.env.reset(task)
        done = False
        steps = 0
        
        while not done and steps < self.max_steps:
            action = self.model.predict(obs)
            obs, reward, done, info = self.env.step(action)
            steps += 1
        
        return {
            'success': info['success'],
            'steps': steps,
            'trajectory': info['trajectory']
        }
    
    def evaluate(self, num_episodes=50):
        """完整评估流程"""
        results = []
        
        for i in range(num_episodes):
            task = self.test_set.sample()
            result = self.run_episode(task)
            results.append(result)
            
            # 实时日志
            self.logger.log_episode(i, result)
        
        # 计算指标
        metrics = self.compute_metrics(results)
        
        # 置信区间
        metrics['sr_ci'] = self.wilson_interval(
            metrics['success_rate'], num_episodes
        )
        
        # 保存结果
        self.logger.log_summary(metrics)
        self.save_artifacts(results)
        
        return metrics
    
    def wilson_interval(self, p, n, z=1.96):
        """Wilson Score Interval for SR"""
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2*n)) / denominator
        spread = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denominator
        return (center - spread, center + spread)
```

### 5.3 CI/CD 集成

```yaml
# .github/workflows/eval.yml
name: Model Evaluation

on:
  push:
    paths:
      - 'checkpoints/**'

jobs:
  evaluate:
    runs-on: self-hosted-gpu
    steps:
      - name: Run CALVIN Evaluation
        run: |
          python eval.py \
            --benchmark calvin \
            --checkpoint ${{ github.sha }} \
            --num_episodes 100
      
      - name: Upload Results
        uses: wandb/upload-artifact@v1
        with:
          name: eval-results
          path: results/
```

### 5.4 失败案例分析

```python
class FailureAnalyzer:
    """自动分析失败原因"""
    
    def analyze(self, failed_episodes):
        categories = {
            'perception': [],      # 视觉识别错误
            'planning': [],        # 规划路径错误
            'execution': [],       # 执行精度不足
            'ik_failure': [],      # 逆运动学无解
            'collision': [],       # 碰撞
        }
        
        for ep in failed_episodes:
            reason = self.classify_failure(ep)
            categories[reason].append(ep)
        
        # 可视化
        self.plot_failure_distribution(categories)
        
        # 生成报告
        return self.generate_report(categories)
```

---

## 6. 面试高频考点

**Q: 具体讲讲怎么构建 Evaluation Pipeline 的？**
A: 核心组件包括：
1. **数据管理**: 标准化测试集 (CALVIN/SIMPLER)，版本控制，确保可复现
2. **推理服务**: 支持多 Checkpoint 并行评估，Batch/Streaming 两种模式
3. **指标计算**: 自动化 SR/MSS/IR 计算，Wilson Interval 置信区间
4. **可视化**: 失败案例分析，Attention 可视化，轨迹回放
5. **CI/CD 集成**: 每次训练自动触发评估，结果上传 W&B

**Q: 什么是 "Cherry-picking"？如何避免？**
A: Cherry-picking 指只展示成功的视频片段。避免方法是报告严格定义的 Success Rate，并公开所有尝试的原始视频 (Uncut Videos) 或日志。

**Q: 为什么 Sim-to-Real 评估很难？**
A: 因为 Reality Gap。仿真里的成功不代表真机成功。SIMPLER 等工作试图缩小这个 Gap，但目前仍未完全解决。最可靠的评估依然是真机。

**Q: 如何评估模型的泛化性 (Generalization)？**
A: 定义三个级别的泛化：
L1: **Interpolation** (训练分布内，新位置/新角度)。
L2: **Visual Extrapolation** (未见过的物体颜色/纹理/背景)。
L3: **Semantic Extrapolation** (未见过的物体类别/新指令)。
面试时要明确指出模型达到了哪一级别的泛化。

---
[← Back to Theory](./README.md)
