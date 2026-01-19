# Ken Goldberg：在 AI 时代重估传统机器人学 —— 数据质量、基础设施与“GOFE”回归

这篇笔记整理自一场以 **Ken Goldberg（UC Berkeley AUTOLab）** 为核心的对谈摘要（Waymo 视角切入），并结合他近年的公开演讲/文章，提炼出对 VLA/具身系统最可落地的工程结论：

- **不要迷信端到端**：经典机器人学（控制/估计/几何）仍是“让系统先跑起来”的最短路径。
- **数据的关键不在“大”，而在“关键时刻（bottleneck moments）”**：真正有信息密度的是接触、插装、滑移等微窗口。
- **基础设施是生产力**：数据的存储/索引/同步/回放/随机访问能力，决定了你能否持续迭代。
- **云-边-雾协同（Fog Robotics）**：机器人寿命长、算力迭代快，系统必须分层解耦。

---

## 1) GOFE：Good Old-Fashioned Engineering 的“引导程序”价值

Goldberg 的核心观点不是“反对大模型”，而是强调一种 **Bootstrap** 路线：

- **先用传统模块让系统可用**：PID / 运动学 / SLAM / Kalman filter / 视觉几何等，让系统在真实世界跑起来。
- **再用真实运行的数据喂给学习系统**：当数据与 failure modes 足够多时，再逐步用学习模型替换局部模块。

这套思路的工程收益是：

- **模块化/可测试/可替换**（debug 能力强）
- **更可解释的失效模式**（能定位到底是感知/规划/控制/时序同步出了问题）

---

## 2) 为什么 Sim-to-Real 在 Manipulation 上特别难

对谈里给了一个很“工程”的解释：

- **自由空间任务（避障/走路）**：对力的绝对精度不总是敏感，很多时候“方向/符号/接触稳定”更重要。
- **操纵任务（manipulation）**：高度依赖接触力学细节：摩擦、微形变、滑移、力矩、材料各向异性。

因此会出现一种常见陷阱：

> 图形学仿真“看起来很真”，但真实受力可能差两个数量级，策略迁移必然崩。

与我们在灵巧手/触觉讨论里反复遇到的结论一致：**触觉/接触状态是操纵的关键隐变量**。

---

## 3) “Render is the new Sim”：Real-to-Render-to-View 的思路

Goldberg 提出一个很实用的替代路径：与其强行把物理仿真做对，不如在一些任务上先把“视觉分布”做丰富：

- 从一次真实演示/真实场景重建出可渲染表示（例如 3D Gaussian Splatting）
- 对视角/光照/初始条件做大量扰动，生成大量视觉变化样本
- 用这些样本训练策略对“视觉变化”鲁棒

这更适用于：

- **准静态（quasi-static）**、对力精确性不极端敏感的任务
- “主要难点是视觉长尾而非接触力学”的场景

参考背景（方法学）：3D Gaussian Splatting 原始工作：`https://arxiv.org/abs/2308.04079`

---

## 4) 数据质量：别做“数据垃圾场”，要找 bottleneck moments

Goldberg 反对把机器人数据简单堆成“大锅”（典型讨论对象包括 OXE 类跨具身混合数据）：

- **可用数据可能只有 10%**：遮挡、误标、标定漂移、光照问题、时间同步问题都会让轨迹变成“废料”。
- 机器人 90% 时间在自由空间移动，这些样本对学习“接触技能”帮助有限。

他主张的策略可以总结为：

- **把数据采集/训练重点放在“瓶颈瞬间”**：
  - 插销入孔、对孔装配
  - 捡起布料边缘/翻折
  - 旋拧到位/是否拧紧
  - 预滑移→补偿的毫秒窗口

可以把它看成机器人版本的 **attention**：把算力与标注预算砸在“信息密度最高的时间片”上。

---

## 5) VLM 的新角色：数据管理员（data curator）

对谈里一个很实用的观点：VLM 不只用来做指令理解，也能当“数据筛选器”：

- 用自然语言查询：**“找出光照好、无遮挡、正在用刷子、视角清晰的片段”**
- 自动生成 VQA 来做数据 sanity check（利用已知元数据做答案对照）

这与我们在 handbook 里写的“数据闭环”方向是一致的：**用模型来治理数据，而不是只训练模型**。

---

## 6) 基础设施：随机访问 vs 时间序列（训练与回放的天然冲突）

对谈里点出了一个非常工程化、但极其关键的矛盾：

- **训练**需要随机访问（random access）来做采样、打乱、batch
- **部署/回放/诊断**需要时间序列（time series）来对齐传感器、重放 failure

Goldberg 的态度是：这些“很不性感”的系统工程，才是行业长期进步的地基。

可参考的系统方向（数据管理）：Robo-DM（机器人数据管理）`https://arxiv.org/abs/2505.15558`

与本 handbook 内部内容的映射：

- 多模态同步：[`deployment/multimodal_data_synchronization.md`](../deployment/multimodal_data_synchronization.md)
- 具身数据采集概览：[`deployment/embodied_data_collection_overview.md`](../deployment/embodied_data_collection_overview.md)

---

## 7) Fog Robotics：把算力从机器人身上“搬走”

Goldberg 对 Fog Robotics 的核心解释非常直接：

- 机器人机体可能服役 **10 年**
- 计算硬件可能 **2 年就过时**

因此应该把系统设计成：

- 机器人端：实时控制、最低闭环、关键安全兜底
- 雾端/边缘：重计算（VLM/VLA 推理、重建、索引、回放）、缓存、近场协同
- 云端：训练、评估、模型发布、长期存储

Fog/Cloud Robotics 的代表性工作之一：`https://arxiv.org/abs/2108.11355`

---

## 8) 对 VLA / 灵巧手 / 长期部署的“可执行清单”

- **先工程后端到端**：先把系统跑稳（控制/估计/同步/校准），再谈大模型替换。
- **把数据预算集中在 bottleneck moments**：采集/标注/评估都围绕接触窗口设计。
- **用 VLM 做数据治理**：自动筛掉遮挡/失焦/失败样本，或给数据打可检索标签。
- **把“可观测性”当作第一等公民**：遮挡严重时，用触觉/近接/力估计补足（参考 GR-Dexter 的指尖触觉，以及我们在软体机器人灵敏度椭球那篇里讨论的“不可观测性”）。
- **云-边-雾协同**：把重算力与快速迭代放到可升级的外部基础设施上。

---

## 参考（可点击）

- Berkeley News（Ken Goldberg，数据差距与人形机器人预期）：`https://news.berkeley.edu/2025/08/27/are-we-truly-on-the-verge-of-the-humanoid-robot-revolution/`
- GOFE 相关公开讲座页（示例）：`https://cse.umn.edu/mnri/events/data-all-you-need-large-robot-action-models-and-good-old-fashioned-engineering`
- Robo-DM（机器人数据管理）：`https://arxiv.org/abs/2505.15558`
- Fog/Cloud Robotics（Fog Robotics 代表工作之一）：`https://arxiv.org/abs/2108.11355`
- 3D Gaussian Splatting（渲染增广的技术基础）：`https://arxiv.org/abs/2308.04079`

