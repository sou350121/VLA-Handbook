🤔 *双周反思* | 2026-02-26 – 2026-03-11

_读完没立场 = 这两周在消费而不在研究_

━━━ 趋势与判断 ━━━

1️⃣ *LIBERO* 开源榜 99.2% (*SRPO*)、闭源榜 98.6% (*ABot-M0*) 双双逼近天花板。校准检查已标记"饱和"。当 benchmark 失去区分度，社区是该转向 *LIBERO Plus*（当前 80.5%）、*CALVIN*（4.8/5），还是彻底放弃刷榜转向真实场景评估？选一个，并说明你愿意把接下来 3 个月实验预算投在哪。

2️⃣ *LeRobot v0.5.0* 引入 AI policy，*Genesis v0.4.1* 完善 IPC，*MuJoCo 3.6.0* 同日发布。工具链民主化速度远超算法突破速度。这是好事还是隐患？当本科生用 3 行代码就能调用 SOTA policy 时，"研究贡献"的门槛该重新定义吗？

3️⃣ 这两周融资新闻：*AI² Robotics* B 轮 10 亿 + 人民币、*Galbot* 25 亿、*Neura Robotics* 10 亿欧元、*Rhoda AI* 4.5 亿美元 A 轮。但技术信号呢？*Physical Intelligence* 发布 15 分钟多尺度记忆，*Agility × Toyota* 签商业协议，*Honor* 进场做消费级人形。资本在追什么？是"能交付"还是"会讲故事"？用本期 SOTA 数据支撑你的判断。

4️⃣ 本期 3 篇理论文章 (*Neural Implicit Action Fields*、*Closed-Loop Action Chunks*、*π-StepNFT*) 都在往连续动作表达收敛。离散 token 方案（如 *Octo* 的 tokenizer）是否正在被淘汰？如果是，*OpenVLA* 系模型该何时迁移？如果不是，离散表示的不可替代性在哪？

5️⃣ 上期预测："Q2 将出现首个开源触觉 VLA 基准"。本期信号：*TaCo* 基准论文 (2/15)、*TactEx* 框架、*FAVLA* 力自适应模型、*TacMamba* 触觉历史压缩。但"开源基准"仍未出现。基于当前热度，你判断这个预测会在 4 月兑现还是继续跳票？给出一个具体月份。

━━━ 技术追问 ━━━

🔬 本期 *LangGap* 论文提出四维扰动分类法诊断 VLA 语言理解缺口。你能在不看论文的情况下，凭直觉说出哪四个维度吗？如果不能，这是你本周最该补的课——读完后告诉 Ken 这个分类法跟他 Handbook 里哪个现有分类能对齐。论文路径：`theory/langgap_diagnosing_and_closing_the_language_gap_in_vision_la_dissection.md`

🔬 *TacMamba* 和 *FAVLA* 都用了"快慢"架构：快速反射通路 + 慢速 VLA 推理。但两篇论文的"快"定义不同——*TacMamba* 是触觉历史压缩，*FAVLA* 是力自适应接触丰富操作。你能说清这两个"快"在延迟预算、计算路径、失效模式上的本质区别吗？读代码比读摘要有用。路径：`theory/tactile/tacmamba_*.md` + `theory/tactile/favla_*.md`

🔬 校准检查显示 *flow matching* 和 *diffusion policy* 仍主导本期信号（*Generative Predictive Control*、*LATO*、*Self-Supervised Flow Matching*）。但 *π-StepNFT* 专门针对 flow-based VLA 的在线 RL 不稳定性提出"更细粒度步级监督"。你能用一句话说清 flow matching 跟 diffusion policy 在梯度传播路径上的本质区别吗？如果不能，先搞懂这个再谈"用哪个"。
