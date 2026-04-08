# 1X World Model（1XWM）：把“视频世界模型”用在人形 NEO 上

> 一句话：和传统 VLA“看一帧→出动作”不同，1X 的路线是 **“看一帧 + 指令 → 先生成未来视频（想象成功）→ 再用逆动力学把视频变成动作”**，试图让机器人直接吃到互联网视频预训练的红利。  
> 核心参考：1X 官方博文（world-model-self-learning）与其对外介绍材料。  
> - 官方：[`1x.tech/discover/world-model-self-learning`](https://www.1x.tech/discover/world-model-self-learning)  
> - 媒体/通稿（中文概述）：[`globenewswire.com`（中文稿）](https://www.globenewswire.com/news-release/2026/01/13/3217488/0/zh-hans/1X-%E6%8F%AD%E7%A4%BA%E4%BA%BA%E5%BD%A2%E6%9C%BA%E5%99%A8%E4%BA%BA-AI-%E8%8C%83%E5%BC%8F%E8%BD%AC%E7%A7%BB-NEO-%E5%BC%80%E5%A7%8B%E8%87%AA%E4%B8%BB%E5%AD%A6%E4%B9%A0.html)

---

## 1) 为什么他们说“VLA 不够”

1X 的批评点很直白：很多 VLA 是在一个预训练 VLM 上接一个动作头（“VLM + Action Head”），擅长语义理解，但并不直接学习**物理动态**与**长时序因果**。因此：
- 简单任务也要花非常多昂贵 robot data 才能学会；
- 对交互物理的“预判”（比如接下来会发生什么）能力不足。  
参考：[`1x.tech/discover/world-model-self-learning`](https://www.1x.tech/discover/world-model-self-learning)。

---

## 2) 1XWM 的核心结构：世界模型 + 逆动力学（IDM）

他们把系统拆成两个关键模块（逻辑上接近 DreamGen / UniPi 这类“生成→控制”路线）：

### A) 世界模型主干（World Model Backbone）

- **形式**：文本条件（text-conditioned）的**视频生成扩散模型**  
- **目标**：在给定初始画面与指令时，生成“接下来场景如何演化”的未来视频序列（同时要满足视觉/空间/物理一致性）  
参考：[`1x.tech/discover/world-model-self-learning`](https://www.1x.tech/discover/world-model-self-learning)。

### B) 逆动力学模型（Inverse Dynamics Model, IDM）

- **目标**：把“相邻视频帧之间的状态变化”映射成“需要执行的动作序列”
- **额外作用**：用 IDM 的约束/打分做**拒绝采样**（reject / resample），把不符合机器人运动学/执行器能力的生成结果筛掉  
参考：[`1x.tech/discover/world-model-self-learning`](https://www.1x.tech/discover/world-model-self-learning)。

---

## 3) 训练数据与配比（为什么它能少用 robot data）

官方描述的一个关键点：**先用互联网视频学“世界怎么动”，再用少量机器人数据把“世界”对齐到 NEO 的具身**。

### 3.1 多阶段训练（概念流程）

- **互联网规模视频预训练**：学通用物理动态先验（掉落、推动、开门等）
- **人类第一视角（POV）中期训练**：把“操作行为模式”对齐到第一人称交互
- **NEO 传感器-运动日志微调**：对齐到机器人外观、相机视角、运动学与执行器约束  
参考：[`1x.tech/discover/world-model-self-learning`](https://www.1x.tech/discover/world-model-self-learning)。

### 3.2 文本对齐：字幕上采样（Caption Upsampling）

他们指出第一视角数据集往往只有简短任务描述，于是用 VLM 生成更细的字幕，提升提示词遵循能力。  
参考：[`1x.tech/discover/world-model-self-learning`](https://www.1x.tech/discover/world-model-self-learning)。

---

## 4) 推理时怎么用：并行“想象” + 选最可能成功的那个

推理阶段的典型流程是：

1. 输入：**一帧初始画面 + 文本指令**
2. 世界模型生成：未来视频序列（多采样）
3. IDM 提取：动作轨迹并下发执行
4. 为平滑：对 IDM 输出做时间平均（多噪声样本 + 滑动窗口）  
参考：[`1x.tech/discover/world-model-self-learning`](https://www.1x.tech/discover/world-model-self-learning)。

他们进一步提出一个很“工程”的假设：**生成视频质量和任务成功率相关**。因此可以：
- 并行生成多个候选视频
- 用人工或 **VLM 评估器**自动挑一个“看起来会成功”的再执行  
参考：[`1x.tech/discover/world-model-self-learning`](https://www.1x.tech/discover/world-model-self-learning)。

---

## 5) 这条路线的优点 / 风险（读 demo 要看什么）

### 优点（如果成立）
- **更强的 OOD 泛化**：从互联网视频学到的物理动态先验，可让策略更快覆盖新物体/新动作模式  
- **更少依赖遥操作示教**：至少在“会发生什么”的部分减少 robot data 压力  
参考：[`1x.tech/discover/world-model-self-learning`](https://www.1x.tech/discover/world-model-self-learning)。

### 风险（常见断点）
- **“脑子会了，手没会”**：想象视频很完美，但动作执行抓空/碰撞（world→action 的落差）
- **可控性与安全**：生成式世界模型如何提供可验证的安全边界（碰撞、力限、失败回退）
- **评估口径**：最关键不是“视频像不像”，而是**真实成功率、干预率、长时序稳定性**  
参考：[`globenewswire.com`（中文稿）](https://www.globenewswire.com/news-release/2026/01/13/3217488/0/zh-hans/1X-%E6%8F%AD%E7%A4%BA%E4%BA%BA%E5%BD%A2%E6%9C%BA%E5%99%A8%E4%BA%BA-AI-%E8%8C%83%E5%BC%8F%E8%BD%AC%E7%A7%BB-NEO-%E5%BC%80%E5%A7%8B%E8%87%AA%E4%B8%BB%E5%AD%A6%E4%B9%A0.html)。

---

## 6) 和 VLA Handbook 里的其它路线怎么对齐

- **和“视频世界模型”趋势一致**：Jim Fan 的行业复盘也提到“更合理的方向：视频世界模型”（见 [`jim_fan_2025_robotics_lessons.md`](./jim_fan_2025_robotics_lessons.md)）。
- **和“产业路线图”对齐**：这更接近“生态平台/标准化 + 预训练规模化”的路线（见 [`industry_paths_to_generalization.md`](./industry_paths_to_generalization.md)）。

