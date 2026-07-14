# 国际头部机器人公司

> 本页面整理了全球领先的机器人公司（不含中国、亚洲），包括融资信息和求职参考。

## 公司概览

| 公司 | 核心产品 | 领域 | 融资/规模 (Est.) | 地点 (HQ/Branches) | 备注 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tesla (US)** | Optimus (Gen 2) | 人形 | **上市巨头** | Palo Alto, CA / Austin, TX (HQ) | 拥有最强的量产制造能力和 FSD 数据闭环，行业风向标。 |
| **Figure AI (US)** | Figure 01/02 | 人形 | **B轮 ($675M)** | Sunnyvale, CA (HQ) | OpenAI/Microsoft/NVIDIA 投资，端到端模型能力强，落地 BMW 工厂。 |
| **Boston Dynamics (US)** | Atlas (Electric) | 人形, 四足 | **Hyundai 收购** | Waltham, MA (HQ) | 运动控制 (Control) 的天花板，液压转电驱后更适合商业化。 |
| **Agility Robotics (US)** | Digit | 人形 (双足) | **B轮+ ($150M+)** | Corvallis, OR (HQ) / Pittsburgh / Palo Alto | Amazon 投资，专注物流场景，Digit 已在亚马逊仓库试运行。 |
| **1X Technologies (Norway)** | Eve, Neo | 人形 (轮式/双足) | **B轮 ($100M)** | Moss, Norway (HQ) / Sunnyvale, CA | OpenAI 投资，Eve 是轮式人形，Neo 是双足。强调安全与家庭应用；近期提出 **1X World Model（视频世界模型+逆动力学 IDM）** 路线（见 [1X 官方](https://www.1x.tech/discover/world-model-self-learning)）。 |
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

---

## 🤖 Moltbot Updates

> ⚙️ 本节由 industry-radar 自动追加 | 人工内容止于上方分隔线 | 协议见 AGENTS.md

| 日期 | 标记 | 公司 | 事件 | 影响 | 来源 |
|---|---|---|---|---|---|
| 2026-06-10 | 🔧 | Boston Dynamics | 首席财务官Amanda McMaster临时接任Boston Dynamics首席执行官职务 | 保障量产交付与商业化节奏下的治理连续性，未出现管理层真空 | [来源](https://www.163.com/dy/article/KRJ82P8605568W0A.html) |
| 2026-06-10 | ⚡ | Boston Dynamics | 现代汽车宣布投资260亿美元在美国建设新工厂，目标年产3万台Boston Dynamics Atlas机器人 | 形成全球最大规模人形机器人制造基础设施，深度绑定Boston Dynamics产能与技术输出 | [来源](https://www.163.com/dy/article/KRJ82P8605568W0A.html) |
| 2026-06-10 | 🔧 | Boston Dynamics | Boston Dynamics Atlas机器人已集成物理AI能力，并通过Orbit软件实现自主导航与换电功能 | 提升工业场景连续作业能力，支撑现代汽车等客户产线级规模化部署需求 | [来源](https://www.163.com/dy/article/KRJ82P8605568W0A.html) |
| 2026-06-10 | 🔧 | Boston Dynamics | Boston Dynamics与谷歌DeepMind持续推进Orbit软件集成及物理AI联合研发 | 加速具身智能底层模型与硬件闭环迭代，强化Atlas在复杂任务泛化能力 | [来源](https://www.163.com/dy/article/KRJ82P8605568W0A.html) |
| 2026-04-22 | ⚡ | Boston Dynamics | 截至2026年4月22日，Boston Dynamics估值被重估至约210亿美元 | 反映其从研发阶段迈入商业化拐点，显著提升市场与产业界对其商业可行性的认可 | [来源](https://www.163.com/dy/article/KUVMD0PU05198NMR.html) |
| 2026-05-07 | ⚡ | Agility Robotics | CEO Peggy Johnson在2026年丰盛峰会炉边谈话中宣布公司当前估值约20亿美元，并计划于2026年晚些时候启动新一轮融资。 | 释放明确资本扩张信号，支撑Robofab工厂产能提升至年产10,000台，强化商业化落地节奏。 | [来源](https://www.cet.com.cn/xwsd/10392643.shtml) |
| 2026-05-07 | 🔧 | Agility Robotics | CEO Peggy Johnson在2026年丰盛峰会披露第五代双足机器人Digit载重能力达50磅，当前运营成本为10–25美元/小时，目标降至2–3美元/小时。 | 明确成本收敛路径，为物流仓储等B端场景规模化部署提供关键经济性依据。 | [来源](https://www.cet.com.cn/xwsd/10392643.shtml) |
| 2026-05-07 | ⚡ | Agility Robotics | CEO Peggy Johnson在2026年丰盛峰会宣布将扩充Robofab机器人工厂产能至年产10,000台。 | 标志从原型验证转向工业化量产阶段，是人形机器人硬件制造能力的关键里程碑。 | [来源](https://www.cet.com.cn/xwsd/10392643.shtml) |
| 2026-05-11 | ⚡ | Apptronik | 2026年5月11日，专注于机器人及物理人工智能的封闭式投资基金RoboStrategy Inc.（NASDAQ: BOT）在其上市当日将Apptronik列为投资组合中的核心高增长企业。 | 标志Apptronik获得专业机器人主题资本市场的正式认可，提升其在AI具身智能领域的机构能见度与背书层级。 | [来源](https://blog.csdn.net/techforward/article/details/161819392) |
| 2026-05-15 | ⚡ | Tesla Optimus | 特斯拉Optimus V3进入量产前夜，绿的谐波、拓普集团人形机器人相关订单显著放量，供应链已启动加班生产 | 标志Optimus从概念验证正式迈入工程化量产阶段，验证技术可行性与商业节奏 | [来源](https://www.163.com/dy/article/KUJDTUOM0534A4SC.html) |
| 2026-06-01 | 🔧 | Tesla Optimus | 特斯拉就Optimus核心技术机密遭非法获取一事，对一名前工程师提起法律诉讼 | 凸显项目保密等级升级及商业化临界点敏感性，预示知识产权保护机制全面启动 | [来源](https://guba.sina.com.cn/?s=thread&bid=21046&tid=197586) |
| 2026-06-02 | ⚡ | Tesla Optimus | 特斯拉在AWE 2026展会上公开展示Optimus Gen3，并宣布将于2026年夏季开始生产、2027年大规模量产并面向公众销售 | 首次明确Gen3量产时间表与商业化路径，确立行业量产基准线 | [来源](https://caifuhao.eastmoney.com/news/20250515142551617713680) |
| 2026-06-04 | ⚡ | Tesla Optimus | 特斯拉确认Optimus V3量产临近，得州超级工厂建设专属制造设施，弗里蒙特工厂试点产线已启动，样机已在自有工厂执行电池分类等任务 | 证实内部产线部署与真实场景落地能力，强化‘从实验室到车间’的可信度 | [来源](https://caifuhao.eastmoney.com/news/20250515142551617713680) |
| 2026-06-08 | ⚡ | Tesla Optimus | Tesla Optimus（T-Bot）于2026年6月8日前后确认进入量产启动（SOP）阶段，核心及新晋供应商均已收到Q2正式采购订单。 | 标志Optimus从工程验证正式迈入规模化交付阶段，为2026年7–8月弗里蒙特工厂V3量产奠定供应链基础。 | [来源](https://finance.eastmoney.com/a/202606083763085170.html) |
| 2026-06-08 | 🔧 | Tesla Optimus | 截至2026年6月8日，Tesla Optimus当前月度交付量级已超100台且呈逐月上升趋势。 | 反映产线爬坡与供应链协同初见成效，为年内百万台产能目标提供初步数据支撑。 | [来源](https://finance.eastmoney.com/a/202606083763085170.html) |
| 2026-06-08 | ⚡ | Tesla Optimus | 特斯拉确认Optimus V3将于2026年7–8月在弗里蒙特工厂正式启动量产，首年目标产能为100万台。 | 首次明确V3量产时间窗口与产能规划，显著增强市场对商业化节奏的确定性预期。 | [来源](https://finance.eastmoney.com/a/202606083763085170.html) |
| 2026-06-15 | ⚡ | Tesla Optimus | 2026年6月15日机构研报确认特斯拉第三代Optimus预计2026年年中亮相、7–8月启动正式投产，加州工厂规划年产能100万台，德州工厂筹备第二代产线。 | 强化市场对Optimus量产节奏的共识，驱动丝杆/减速器/电机等核心零部件供应链关注度提升。 | [来源](https://guba.sina.com.cn/?s=thread&bid=21046&tid=234640) |
| 2026-06-11 | ⚡ | 1X Technologies | 1X Technologies于2026年6月11日宣布其NEO家用人形机器人在美国启动首批面向消费者的交付，系全球首款以‘家用电器’身份进入家庭的人形机器人。 | 标志着人形机器人正式从实验室和工业场景迈入消费级市场，确立1X在商业化落地上的先发地位。 | [来源](https://cloud.tencent.com/developer/news/3433946) |
| 2026-06-13 | ⚡ | Tesla Optimus | 权威信源于2026年6月13日确认特斯拉Optimus Gen3（V3）将于2026年7–8月在弗里蒙特工厂启动量产，全身37关节、灵巧手22自由度，目标2027年底上市。 | 标志Optimus正式进入规模化生产准备阶段，为2027年商业化落地奠定关键制造基础。 | [来源](https://news.qq.com/rain/a/20260423A01NQ100?adChannelId=tech) |
| 2026-06-15 | 🔧 | Tesla Optimus | 国盛证券于2026年6月15日报告指出特斯拉第三代人形机器人预计年中亮相，7–8月启动正式投产，加州工厂规划年产能100万台。 | 强化市场对Optimus量产节奏的共识预期，支撑产业链配套与产能爬坡决策。 | [来源](http://finance.sina.com.cn/jjxw/2026-06-15/doc-inicnisv8287300.shtml) |
| 2026-06-17 | ⚡ | Sanctuary AI | Sanctuary AI在Tier 1汽车供应商现场完成物理AI（Physical AI）性能验证，强调其技术已具备生产就绪（production-ready）能力，而非依赖未来 humanoid 形态。 | 凸显‘物理AI先行’策略，加速AI与真实工业场景闭环验证，绕过通用人形机器人长周期研发瓶颈。 | [来源](https://www.therobotreport.com/sanctuary-ai-validates-physical-ai-performance-tier-1-automotive-supplier/) |
| 2026-06-17 | ⚡ | Tesla Optimus | 特斯拉已完成Optimus供应链核心协议签署，弗里蒙特工厂产线改造完毕，确认于2026年7月启动规模化量产，目标年产能达百万台。 | 验证量产节奏可信度，凸显供应链整合能力与产能雄心，加速人形机器人产业临界点到来。 | [来源](https://m.163.com/dy/article/KVJJBEPP055616YL.html) |
| 2026-06-24 | ⚡ | Agility Robotics | 2026年6月24日，网易新闻报道 Agility Robotics 为‘美国第一家上市的人形机器人公司’，估值近两百亿元人民币，标志其IPO进程取得实质性突破。 | 释放明确的资本市场认可信号，推动人形机器人行业进入规模化商业化与资本化新阶段。 | [来源](https://www.163.com/dy/article/L0725AC80556BPTV.html) |
| 2026-06-24 | ⚡ | Agility Robotics | Agility Robotics 将通过与 Churchill Capital Corp XI 的 SPAC 合并上市，融资 6.2 亿美元，用于推进 Digit v5 量产及交付客户订单。 | 标志具身智能硬件公司首次大规模公开融资，加速人形机器人商业化落地进程。 | [来源](https://www.therobotreport.com/humanoid-maker-agility-robotics-go-public-through-spac-merger/) |
| 2026-06-21 | ⚡ | Boston Dynamics | Boston Dynamics CEO Robert Playter于2026年6月21日向全体员工发布内部信，确认现代汽车将以5000亿韩元收购软银所持剩余9.65%股份，交易完成后Boston Dynamics将成为现代汽车集团全资子公司。 | 软银彻底退出持股，Boston Dynamics完成从软银到现代汽车的控制权转移，标志着其商业化路径正式纳入现代汽车全球机器人战略体系。 | [来源](https://news.mydrivers.com/tag/boshidundongli.htm) |
| 2026-06-24 | ⚡ | Tesla Optimus | 2026年6月24日媒体报道称特斯拉Optimus人形机器人已进入量产准备阶段，谐波减速器、关节模块等核心部件由上游供应商开始供货。 | 标志Optimus从原型验证正式转向规模化生产准备，供应链协同进入实质交付阶段。 | [来源](https://www.163.com/dy/article/L0661LF20519QIKK.html) |
| 2026-06-24 | ⚡ | Tesla Optimus | 2026年6月24日多家媒体援引供应链报告指出，特斯拉Optimus大规模量产将于2026年下半年启动，加州与德州工厂同步推进。 | 首次明确双基地量产节奏，强化商业化落地时间表可信度。 | [来源](https://www.163.com/dy/article/L0661LF20519QIKK.html) |
| 2026-06-24 | 🔧 | Tesla Optimus | 2026年6月24日报道称特斯拉正与美诺自动化、亚洲光电、盟立集团等供应商开展Optimus核心部件的深度合作。 | 揭示关键供应链布局进展，反映本土化与全球化并行的制造策略。 | [来源](https://www.163.com/dy/article/L0661LF20519QIKK.html) |
| 2026-06-22 | ⚡ | Tesla Optimus | 2026年6月22日，特斯拉与台湾盟立自动化（供应谐波减速器和关节模组）及亚洲光学（负责光学‘眼睛’组件）达成供货合作，标志Optimus 3量产进入实质性备货阶段。 | 核心执行部件供应链落地，为7月下旬至8月量产提供确定性保障。 | [来源](https://www.163.com/dy/article/L0661LF20519QIKK.html) |
| 2026-06-22 | ⚡ | Agility Robotics | 2026年6月22日，英伟达发布Halos for Robotics系统，Agility Robotics为其全球首家合作方。 | 接入英伟达端到端物理AI安全栈，提升机器人自主决策安全性与部署效率。 | [来源](https://blog.csdn.net/txg666/article/details/162332380) |
| 2026-06-23 | ⚡ | Tesla Optimus | 2026年6月23日，特斯拉关停Model S/X整车产线，全面改造弗里蒙特工厂为Optimus 3专用产线，设计年产能100万台；得州超级工厂专属Optimus工厂同步动工，规划年产能1000万台。 | 确立规模化制造基础设施，释放量产物理能力边界。 | [来源](https://www.163.com/dy/article/L0661LF20519QIKK.html) |
| 2026-06-24 | ⚡ | Tesla Optimus | 2026年6月24日，产业链厂商确认谐波减速器、关节模组等关键零部件已开始供货，多家机构援引该进展指出Optimus 3量产窗口仅余约1个月，明确指向2026年7月下旬至8月启动。 | 量产时间表从预期转向倒计时，触发二级市场与上游产能爬坡响应。 | [来源](https://www.163.com/dy/article/KV5TI46M0519QIKK.html) |
| 2026-06-24 | ⚡ | Agility Robotics | 公司确认本次交易共募集超6.2亿美元，其中4.2亿美元来自Churchill XI信托现金，超2亿美元通过PIPE方式由富士康领投。 | 显著增强量产与商业化落地资金保障能力，并强化与富士康在制造端的战略协同。 | [来源](https://m.ebrun.com/684584.html) |
| 2026-06-25 | ⚡ | Tesla Optimus | 2026年6月25日，美光科技在第三财季电话会议中确认，Optimus人形机器人将成为其内存芯片最大增量客户，采购需求预期超过特斯拉汽车业务。 | 验证Optimus硬件采购权重跃升至公司级战略优先级，重塑产业链估值逻辑。 | [来源](https://mbd.baidu.com/newspage/data/dtlandingsuper?nid=dt_4478792863668663492) |
| 2026-06-25 | ⚡ | Agility Robotics | Agility Robotics于2026年6月25日完成与Churchill Capital Corp. XI（CCXI）的SPAC合并，正式登陆公开市场。 | 确立其全球首家专注人形机器人研发与销售的上市企业地位，估值约25亿美元。 | [来源](https://m.163.com/dy/article/L06MNSVS0519QIKK.html) |
| 2026-06-25 | 🔧 | Agility Robotics | 截至2026年6月25日，Digit机器人已在丰田（含丰田加拿大制造公司）、舍弗勒、GXO、Mercado Libre及亚马逊等9个客户站点实现商用部署。 | 验证产品真实场景可靠性与工业适配性，支撑后续规模化复制。 | [来源](https://m.163.com/dy/article/L0A9AG6605118UGF.html) |
| 2026-06-25 | ⚡ | Agility Robotics | 新一代Digit V5机型已获得超3亿美元多年期订单，并有30余家潜在客户正在评估大规模部署方案。 | 标志产品迭代成功及市场需求加速释放，奠定2026年下半年收入增长基础。 | [来源](https://m.163.com/dy/article/L0A9AG6605118UGF.html) |
| 2026-06-26 | 🔧 | Tesla Optimus | 2026年6月26日，巨轮智能公告与特斯拉签订28–50亿元框架协议，为其Optimus提供RV减速器及膝踝关节减速模组，首批订单3500台，XT减速器已进入小批量供货阶段。 | 国产关键运动部件获特斯拉认证并启动交付，加速本土供应链替代进程。 | [来源](https://caifuhao.eastmoney.com/news/20260626181836920154950) |
| 2026-06-24 | ⚡ | Agility Robotics | Agility Robotics于2026年6月24日宣布将与Churchill Capital Corp XI（CCXI）合并上市，投后估值约25亿美元，预计募集超6亿美元资金。 | 标志着公司正式进入公开资本市场，为Robofab工厂扩产及Digit V5规模化部署提供关键资本支持。 | [来源](https://m.163.com/dy/article/L06MNSVS0519QIKK.html) |
| 2026-06-24 | 🔧 | Agility Robotics | Agility Robotics于2026年6月24日确认已就Digit第五代（V5）获得客户订单，新版具备更高灵巧性（可移动更小物体）和更高安全标准。 | 验证产品迭代路径获市场认可，加速从测试验证向商业化交付过渡。 | [来源](https://m.163.com/dy/article/L06MNSVS0519QIKK.html) |
| 2026-06-26 | ⚡ | Apptronik | Apptronik于2026年6月26日正式推出Apollo 2人形机器人，采用模块化架构、可更换移动底盘、自研高能效驱动模组（能效＞90%）、LED交互嘴、碰撞检测与换电设计，并同步发布Artemis智能系统与Fleet Connect集群管理系统。 | 标志着其从单机能力向规模化部署与系统级协同的重大演进，强化在工业与物流场景的商业化落地能力。 | [来源](https://tech.ifeng.com/c/8uHGwrDGDD7) |
| 2026-06-29 | ⚡ | Figure AI | BMW集团在南卡罗来纳州工厂正式部署Figure 03人形机器人，此前Figure 02已支持超3万辆X3生产。 | 全球头部车企首次规模化商用新一代人形机器人于汽车产线，标志工业级具身智能进入实际交付阶段。 | [来源](https://www.therobotreport.com/bmw-group-deploys-figure-03-humanoid-after-tests-previous-version/) |
| 2026-07-01 | ⚡ | Apptronik | Apptronik发布新一代通用 humanoid 平台 Apollo 2，并同步启用旗舰级数据采集与训练设施。 | Apollo 2 定位为持续学习平台，强化真实场景部署驱动的机器人模型迭代能力，标志其从原型走向量产级AI机器人基础设施建设。 | [来源](https://www.therobotreport.com/apptronik-unveils-apollo-2-flagship-data-collection-training-facility/) |
| 2026-07-02 | ⚡ | Tesla Optimus | 2026年7月2日，埃隆·马斯克在社交平台发布与近30位员工在弗里蒙特工厂Optimus生产线的合影，并配文‘在弗里蒙特工厂参观Optimus机器人生产线’，特斯拉同步确认Model S/X产线已完成改造，将用于Optimus V3量产。 | 标志Optimus正式进入量产准备阶段，产线落地时间点明确，验证V3硬件工程闭环完成。 | [来源](https://www.163.com/dy/article/L0R8OVDT051191D6.html) |
| 2026-07-01 | ⚡ | Tesla Optimus | 2026年7月1日，埃隆·马斯克在社交平台发布其本人在加州弗里蒙特工厂Optimus Gen3生产线上的合影，并称量产初期将极其缓慢。 | 该影像被市场视为Gen3产线完成FAT验收、具备启动条件的强信号，标志Optimus正式进入量产准备阶段。 | [来源](https://k.sina.cn/article_7857201856_1d45362c001907qgro.html) |
| 2026-07-02 | ⚡ | Tesla Optimus | 2026年7月2日，特斯拉Optimus Gen3专属产线（由原Model S/X产线改造）全员合照公开，工程高管确认将于7月底至8月启动小批量量产，初期周产100台。 | 首次明确Gen3量产启动时间窗口与初始产能节奏，为产业链提供可验证的交付节点预期。 | [来源](https://k.sina.cn/article_7857201856_1d45362c001907qgro.html) |
| 2026-07-02 | 🔧 | Tesla Optimus | 2026年7月2日，野村证券基于产线进展同步上调特斯拉弗里蒙特工厂Optimus年化产能预测至7万台。 | 反映机构对量产爬坡速度的乐观修正，强化供应链备货与资本开支决策依据。 | [来源](https://k.sina.cn/article_7857201856_1d45362c001907qgro.html) |
| 2026-07-01 | 🔧 | Tesla Optimus | 埃隆·马斯克在X平台发布与Optimus工厂团队合影，并明确表示Optimus生产初期将极其缓慢，因所有环节均为全新构建。 | 提示市场对量产爬坡节奏需保持理性预期，影响供应链交付节奏与投资者短期预期。 | [来源](http://k.sina.com.cn/article_7857201856_1d45362c001907kx2y.html) |
| 2026-07-03 | ⚡ | Tesla Optimus | 特斯拉副总裁陶琳在2026全球数字经济大会上正式宣布Optimus人形机器人将于2026年底启动规模化量产，弗里蒙特工厂已启动产线切换。 | 标志Optimus项目从研发验证阶段进入工业量产阶段，为后续商业化落地奠定基础。 | [来源](https://view.inews.qq.com/a/20260703A09AYD00) |
| 2026-07-01 | ⚡ | Tesla Optimus | 2026年7月1日，埃隆·马斯克在社交媒体发布于加州弗里蒙特工厂Optimus生产线的合影，并配文‘正在参观该产线’，标志Optimus V3正式进入量产准备阶段。 | 确认V3已脱离原型验证阶段，启动工业化落地进程，为后续小批量交付奠定基础。 | [来源](http://k.sina.com.cn/article_7857201856_1d45362c001907qgro.html) |
| 2026-07-03 | 🔧 | Tesla Optimus | 2026年7月3日，马斯克强调Optimus初期量产将‘极其缓慢’，因其属全栈自研、从零构建的新制造体系，与汽车生产存在本质差异。 | 向供应链与投资者传递现实预期，抑制对短期规模化交付的过度乐观。 | [来源](https://view.inews.qq.com/a/20260702A076G700) |
| 2026-07-06 | ⚡ | Tesla Optimus | 2026年7月6日，权威资料确认Optimus V3将于2026年7–8月在弗里蒙特工厂启动量产，初期规划年产能5万–10万台。 | 首次明确量产起始窗口与初始产能规模，成为产业链备货与资本投入的关键锚点。 | [来源](http://mbd.baidu.com/newspage/data/dtlandingsuper?nid=dt_4010323531850318124) |
| 2026-07-06 | ⚡ | Boston Dynamics | 波士顿动力联合现代汽车，将其腿式机器人（Atlas与Spot）部署至2026年FIFA世界杯相关场景。 | 标志性商业化落地事件，验证人形/四足机器人在高动态、高曝光公共环境中的可靠性与品牌协同价值。 | [来源](https://www.therobotreport.com/boston-dynamics-brings-its-legged-robots-to-the-fifa-world-cup/) |
| 2026-07-07 | 🔧 | Boston Dynamics | 2026年7月7日，Boston Dynamics在巴黎MACHINA Summit峰会上以‘新一代人形机器人’身份演示Atlas全电动平台在负重、跳跃及模拟工地作业等场景下的持续能力演进。 | 体现Atlas从运动性能向实用化任务能力延伸，支撑工业与基建领域商业化预期。 | [来源](https://hk.prnasia.com/story/540028-2.shtml) |
| 2026-07-08 | ⚡ | Boston Dynamics | 2026年7月8日，Boston Dynamics研发的量产版Atlas人形机器人在纽约/新泽西球场举行的FIFA World Cup 2026™ 16强赛中场休息期间完成全球首次真实高动态体育场景下的公开展示，执行球星庆祝动作并递球给裁判。 | 标志Atlas从实验室走向规模化商用场景的关键里程碑，验证其运动控制、环境适应与系统可靠性。 | [来源](https://www.prnasia.com/story/540027-1.shtml) |
| 2026-07-08 | ⚡ | Boston Dynamics | 2026年7月8日，作为现代汽车集团全资子公司，Boston Dynamics依托后者世界杯官方机器人合作伙伴身份，推动Atlas在FIFA World Cup 2026™中完成技术叙事与商业化联合展示。 | 强化Atlas品牌全球认知度，并确立以车企生态为支点的B2B2C商业化路径。 | [来源](https://www.donews.com/news/detail/8/6625256.html) |
| 2026-07-08 | ⚡ | Boston Dynamics | 2026年7月8日，Boston Dynamics量产版Atlas在FIFA World Cup 2026™现场完成首次高动态、高干扰体育环境下的稳定运行与交互任务。 | 为后续面向赛事运营、公共安全、大型活动服务等场景的商业部署提供关键实证。 | [来源](https://beareyes.com.cn/2/lib/202607/08/20260708706.htm) |
| 2026-07-08 | ⚡ | Boston Dynamics | 2026年7月8日，Boston Dynamics联合现代汽车，在FIFA World Cup 2026™中完成Atlas机器人与赛事流程的端到端整合，包括实时定位、动作同步与裁判交互。 | 验证跨企业技术协同与大型活动系统集成能力，奠定未来体育科技合作范式。 | [来源](https://www.21ic.com/a/1007579.html) |
| 2026-07-09 | ⚡ | Apptronik | Apptronik获得3.31亿美元战略投资，资金将用于加速其通用机器人平台Apollo的商业化落地。 | 显著增强其在工业场景（如奔驰工厂）及太空应用（源自NASA实验室背景）的规模化部署能力。 | [来源](https://zhuanlan.zhihu.com/p/1980283492366557830) |
| 2026-07-03 | ⚡ | Tesla Optimus | 特斯拉CEO马斯克于2026年7月3日发布弗里蒙特工厂Optimus生产线合影，并确认Gen-3产线改造基本完成，为7月底试产铺路。 | 标志Optimus正式从原型验证阶段转入工业级量产准备阶段，产线专属化改造完成。 | [来源](https://k.sina.cn/article_7857201856_1d45362c001907qgro.html) |
| 2026-07-03 | 🔧 | Tesla Optimus | 多家财经媒体交叉印证，Optimus Gen-3将于2026年7月底至8月启动小批量量产（试产爬坡），初期周产约100台，9月目标周产1000台。 | 明确试产节奏与爬坡路径，为后续产能释放提供可追踪节点。 | [来源](https://k.sina.com.cn/article_7857201856_1d45362c001907r4du.html) |
| 2026-07-06 | ⚡ | Tesla Optimus | 特斯拉副总裁陶琳在2026全球数字经济大会上宣布Optimus将于2026年Q4（10–12月）启动规模化量产，弗里蒙特工厂远期年产能目标为100万台。 | 首次由高管公开明确量产时间窗口与产能规划，确立商业化节奏里程碑。 | [来源](https://k.sina.cn/article_7857201856_1d45362c001907qgro.html) |
| 2026-07-06 | ⚡ | Agility Robotics | Agility Robotics宣布将通过与Churchill Capital Corp. XI合并实现美股上市，交易前估值25亿美元，预计募资超6.2亿美元。 | 标志着公司进入公开资本市场，为规模化量产和全球交付提供关键资金支持。 | [来源](https://cj.sina.com.cn/articles/view/6522004851/184bde57300101e3bu) |
| 2026-07-06 | ⚡ | Agility Robotics | Agility Robotics确认已签署含SLA和罚则的商业合同，锁定超3亿美元签约营收，对应约1000台第五代Digit机器人分批交付，客户包括GXO Logistics、亚马逊AWS Robotics生态伙伴、丰田汽车北美、舍弗勒集团及Mercado Libre。 | 验证商业化落地能力，确立其在仓储物流场景的首批规模化付费客户群。 | [来源](https://www.163.com/dy/article/L16AR6VJ05569K8R.html) |
| 2026-07-06 | 🔧 | Agility Robotics | CEO Peggy Johnson向TechCrunch明确Agility采用按月租赁的RaaS模式，所有POC客户均已进入实际部署阶段。 | 表明商业模式完成从验证到履约的跨越，支撑可持续经常性收入（ARR）增长。 | [来源](https://www.163.com/dy/article/L16AR6VJ05569K8R.html) |
| 2026-07-06 | 🔧 | Agility Robotics | Agility Robotics公布第五代Digit机器人载重50磅（≈22.7公斤）、续航约22小时、最大触及高度2.1米，并具备反向弯曲膝关节设计。 | 性能升级直接适配主流仓储货架作业需求，强化产品在目标场景的技术竞争力。 | [来源](https://caifuhao.eastmoney.com/news/20260703205341259716360) |
| 2026-07-06 | 🔧 | Agility Robotics | Agility Robotics表示所募资金将用于扩建俄勒冈州塞勒姆RoboFab工厂（现70,000平方英尺），以加速交付积压订单。 | 指向制造能力瓶颈突破，是支撑千台级交付承诺的必要基础设施投入。 | [来源](https://baike.baidu.com/item/Robo%20Fab/68047701) |
| 2026-07-05 | ⚡ | Boston Dynamics | Boston Dynamics于2026年7月5日通过达沃斯科技峰会成果披露，正式展示新一代Atlas人形机器人在负重、跳跃及真实工地场景下的实机作业能力。 | 标志Atlas从实验室演示迈向结构化工业场景落地的关键进展，强化其在具身智能硬件领域的技术领导地位。 | [来源](https://caifuhao.eastmoney.com/news/20260705070328317051380) |
| 2026-07-07 | 🔧 | Tesla Optimus | 2026年7月7日，Tesla Optimus Gen 3在法国巴黎Machina Summit峰会完成欧洲首秀，特斯拉官方参与并提供实机行走、抓取与交互演示视频。 | 实现关键海外市场技术亮相，验证Gen 3工程成熟度，支撑后续本地化合作与场景拓展预期。 | [来源](https://k.sina.com.cn/article_7857201856_1d45362c001907txai.html) |
| 2026-07-12 | ⚡ | Tesla Optimus | 特斯拉于2026年7月12日在弗里蒙特工厂完成Model S/X产线拆除，耗时46天，正式启用Optimus人形机器人专用产线。 | 标志Optimus量产物理基础设施就绪，进入规模化制造阶段。 | [来源](https://www.sina.cn/news/detail/5317291130749597.html) |
| 2026-07-12 | ⚡ | Tesla Optimus | IT之家报道特斯拉仅用46天完成弗里蒙特工厂Model S/X整车产线拆除，全面切换为Optimus第三代（Gen3）人形机器人量产产线。 | 标志特斯拉正式从电动车制造转向具身智能硬件规模化量产，产线资源完成战略性重构。 | [来源](https://view.inews.qq.com/a/20260706A0B1WZ00) |
| 2026-07-08 | ⚡ | Boston Dynamics | Boston Dynamics研发的Atlas机器人作为现代汽车官方合作伙伴，在FIFA世界杯2026十六强赛（纽约/新泽西球场）执行中场表演并交付比赛用球。 | 标志其人形机器人首次在顶级国际体育赛事中规模化、实时化落地应用，强化与现代汽车的战略协同及公众技术形象。 | [来源](https://www.prnasia.com/lightnews/lightnews-1-102-97467.shtml) |
| 2026-07-08 | 🔧 | Boston Dynamics | Atlas在2026 FIFA世界杯现场实时演绎哈里·凯恩、哈兰德、库尼亚、孙兴慜等球星的经典进球庆祝动作。 | 验证其高精度运动控制与具身智能在动态非结构化环境下的实时响应能力。 | [来源](https://www.prnasia.com/lightnews/lightnews-1-102-97467.shtml) |
| 2026-07-10 | 🔧 | 1X Technologies | 1X Technologies于2026年7月10日正式发布NEO人形机器人搭载的新一代灵巧手，强调22自由度与闭环控制能力。 | 提升末端操作精度与鲁棒性，支撑家庭场景中复杂物操作任务落地。 | [来源](https://baike.baidu.com/item/1X/65295746) |
| 2026-07-14 | 🔧 | Tesla Optimus | 特斯拉在46天内完成弗里蒙特工厂Model S/X产线拆除，明确将该产能空间转用于Optimus规模化生产。 | 实质性释放物理产能，支撑Optimus V3按期投产，体现资源倾斜力度。 | [来源](http://finance.sina.com.cn/stock/stockzmt/2026-07-07/doc-inifxpii1650902.shtml) |
