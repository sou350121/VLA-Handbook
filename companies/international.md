# 国际头部机器人公司

> 本页面整理了全球领先的机器人公司（不含中国、亚洲），包括融资信息和求职参考。

## 公司概览

| 公司 | 核心产品 | 领域 | 融资/规模 (Est.) | 地点 (HQ/Branches) | 备注 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tesla (US)** | Optimus (Gen 2) | 人形 | **上市巨头** | Palo Alto, CA / Austin, TX (HQ) | 拥有最强的量产制造能力和 FSD 数据闭环，行业风向标。 |
| **Figure AI (US)** | Figure 01/02 | 人形 | **B轮 ($675M)** | Sunnyvale, CA (HQ) | OpenAI/Microsoft/NVIDIA 投资，端到端模型能力强，落地 BMW 工厂。 |
| **Boston Dynamics (US)** | Atlas (Electric) | 人形, 四足 | **Hyundai 收购** | Waltham, MA (HQ) | 运动控制 (Control) 的天花板，液压转电驱后更适合商业化。 |
| **Agility Robotics (US)** | Digit | 人形 (双足) | **B轮+ ($150M+)** | Corvallis, OR (HQ) / Pittsburgh / Palo Alto | Amazon 投资，专注物流场景，Digit 已在亚马逊仓库试运行。 |
| **1X Technologies (Norway)** | Eve, Neo | 人形 (轮式/双足) | **B轮 ($100M)** | Moss, Norway (HQ) / Sunnyvale, CA | OpenAI 投资，Eve 是轮式人形，Neo 是双足。强调安全与家庭应用。 |
| **Sanctuary AI (Canada)** | Phoenix | 人形 | **B轮 ($140M+)** | Vancouver, Canada (HQ) | 强调通用智能 (General Purpose)，Phoenix 拥有极强的灵巧手操作能力。 |
| **Apptronik (US)** | Apollo | 人形 | **A轮 ($14.6M+)** | Austin, TX (HQ) | NASA 背景，Apollo 设计紧凑，与 Mercedes-Benz 合作。 |
| **DYNA Robotics (US)** | DYNA-1 / DYNA-1i（机器人基础模型 + 商用机器人系统） | 商用通用机器人（工厂/餐饮/洗衣等） | **A轮 ($120M)** | Redwood City, CA | 主打“部署驱动的持续学习”：在真实环境训练 VLA，并引入 Reward Model 支撑长时无干预运行与自我纠错。 |

## 求职建议

### 算法岗位热门方向
- **端到端 VLA**: Figure AI, Tesla
- **强化学习/运动控制**: Boston Dynamics, Agility Robotics
- **安全与通用智能**: 1X Technologies, Sanctuary AI
- **工业落地**: Apptronik

---

## DYNA Robotics：产品与技术路线分析

> 结论一句话：DYNA 不是“只做模型”，而是把 **VLA + Reward Model + 真实部署数据闭环** 做成一个商用系统，核心指标不是单次成功率，而是 **长时（小时级/24h）无干预 + 吞吐 + 质量**。

### 1) 他们在卖什么（产品形态）

- **商用机器人系统（DYNA-1 作为核心 AI）**：官网明确把 DYNA-1 描述为“commercial AI system”，面向工厂、餐饮、洗衣等真实行业场景（Factory / Restaurant / Laundry）。  
  参考：DYNA 官网首页的行业描述与 “DYNA-1 is powering real output today” 表述（见 [DYNA 官网](https://www.dyna.co/)）。
- **DYNA-1（Dynamism v1）**：强调“round-the-clock, high-throughput dexterous autonomy”，并给出一个非常工程化的展示任务：**24 小时连续折餐巾（850+），~60% 人类速度，99.4% 成功率，0 干预**。  
  参考：[DYNA-1 Research](https://www.dyna.co/dyna-1/research)。
- **DYNA-1i（DYNA-1 improved / Open-world generalization）**：用“tens of hours post-training data（全部在办公室采集）”来把能力扩展到 **完全未见环境**；并用“30 分钟连续 trial、统计 30 分钟内连续折叠数量”这种更贴近部署的评测方式呈现泛化。  
  参考：[Open-World Dexterity and Live Demos](https://www.dyna.co/dyna-2/research)。

### 2) 他们的关键技术抓手：Reward Model (RM) + 连续部署数据

DYNA 的公开叙述里，RM 是“生产级鲁棒性”的核心，因为它让系统能在没有明确 episode 边界的连续流数据里：
- **估计任务进度（progress estimation）**、提供细粒度反馈  
- **支持“Intentional Error Recovery”（有目的的错误恢复）**  
- **把部署数据变成高质量训练数据（自动分段、subtask 标注）**  

这类能力在“无重置、长时运行”的商用任务里很关键：不是做到一次成功，而是 **遇到极小概率 bad state 还能自救并继续跑**。  
参考：[DYNA-1 Research](https://www.dyna.co/dyna-1/research)。

### 3) 他们的“落地指标”选得很对（对 VLA/具身团队的启发）

DYNA 的公开指标设计非常“部署导向”，值得当作你评估任何 VLA 产品的 checklist：
- **长时无干预**：从“demo-run 30 分钟就漂移崩溃”这个行业通病出发，直接用 8h/24h 连续运行证明稳定性（见 DYNA-1 逐周改进叙述）。  
  参考：[DYNA-1 Research](https://www.dyna.co/dyna-1/research)。
- **吞吐（throughput）+ 质量（quality）**：不仅看成功率，还强调“生产级质量”差异可能只在初始折痕的 \(< 1/3\) inch 精度。  
  参考：[DYNA-1 Research](https://www.dyna.co/dyna-1/research)。
- **跨环境泛化（open-world）**：用 seen vs unseen 的 30 分钟连续表现对比，而不是只做离线 benchmark。  
  参考：[DYNA-1i / dyna-2](https://www.dyna.co/dyna-2/research)。

### 4) 商业化与融资信息（用来判断“是不是在真落地”）

- **A 轮 $120M（2025-09-15）**：PRNewswire 的新闻稿明确提到 DYNA-1、24 小时非停运行 99%+ 成功率、以及在酒店/餐厅/洗衣店/健身房等场景的部署叙述。  
  参考：[PRNewswire 120M Series A](https://www.prnewswire.com/news-releases/dyna-robotics-raises-120-million-to-advance-robotic-foundation-models-on-the-path-to-physical-artificial-general-intelligence-302556817.html)。
- **公司“第一性原理”文章**：CEO 文章把“Distribution is King / ROI is PMF / Iteration Speed”写得非常直白，并把 DYNA-1 的“60% human throughput at stringent quality bar”作为里程碑。  
  参考：[DYNA 120M Series A 博文](https://www.dyna.co/blog/dyna-robotics-closes-120m-series-a)。

### 5) 风险与疑点（面向面试/尽调的提问清单）

- **公开技术细节仍有限**：他们描述了 RM、自动分段/进度估计、持续部署，但对数据格式、模型结构、训练/推理延迟、硬件规格等披露不多（这很符合商用公司风格）。  
- **任务分布与泛化边界**：公开 demo 主要是折叠（餐巾/衣物）与杯子填充；这些任务很适合展示长时鲁棒性，但你在评估时仍应追问：新 SKU、新抓取物、不同光照/相机位姿、桌面变化等情况下的失败模式是什么、恢复策略是什么。  
- **“RM-in-the-loop”可能带来的工程成本**：RM 需要标注/监督信号或自监督 proxy，且要与部署数据流强耦合；你可以追问它在不同任务、不同 robot station 之间怎么迁移。

---
[← Back to Companies](./README.md)

