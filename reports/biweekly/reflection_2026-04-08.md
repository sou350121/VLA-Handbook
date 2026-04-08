🤔 *双周反思* | 2026-03-26 – 2026-04-08

_读完没立场 = 这两周在消费而不在研究_

━━━ 趋势与判断 ━━━

1️⃣ *flow_matching* 是唯一加速的方法族（accel_7d=1.41），同期 *Genesis* 7 天内两次更新（v0.4.4→v0.4.5）、*LeRobot* v0.5.1 发布。这是「工程优先、论文滞后」的典型信号——你相信 flow_matching 会在 6 周内被至少 2 个顶级 VLA 团队正式采用吗？给出你的判断依据，不允许说「看情况」。

2️⃣ *CALVIN* 全系列已标记 saturated（ABC-D 4.8%、ABCD-D 4.8%、D-D 4.3%），但 41 次 SOTA 变动中仍占 15 次。如果你的团队现在还在 CALVIN/LIBERO 上刷分，你是在做研究还是在浪费计算资源？说出你的理由。

3️⃣ *tactile*（accel_7d=0.44）和 *dexterous_hand*（accel_7d=0.58）学术衰退，但磅策医疗 AI 手术机器人已完成近 1000 台手术、优必选营收 +53.3%。学术界追求通用基准 SOTA，产业界追求垂直场景 ROI——这两条路线会在 2027 年彻底分叉吗？你押注哪一边？

4️⃣ *diffusion_policy* 正在快速衰退（accel_7d=0.39），在 ACTION HEAD 竞争对中以 19 篇对 41 篇被 *flow_matching* 碾压。如果你的代码库还在用 diffusion_policy 做 action head，你计划什么时候迁移？给出具体时间节点。

5️⃣ 社交情报显示 3/29-4/5 期间 *Tesla Optimus Gen3* 量产演示、*乐聚万台产线*、*智元万台下线* 密集披露。这些产业信号与 flow_matching 学术加速的滞后共振约 2-3 周——你相信这个滞后关系是因果还是巧合？

━━━ 技术追问 ━━━

🔬 本期 3 篇突破论文（*OmniVTA*、*VP-VLA*、*LaMP*）都涉及多模态融合或表示学习。但你能说清 *flow matching* 跟 *diffusion policy* 在数学本质上的区别吗？如果不能，这是你这两周最该补的课——去读《Flow Matching for Generative Modeling》原始论文，然后回来回答：为什么 flow matching 的单步生成特性在实时控制场景中比多步扩散更高效？

🔬 *OmniVTA* 成功整合触觉与世界模型，解决接触丰富操作中的状态估计问题。在 tactile 方法族整体衰退的背景下，这篇论文的成功路径是「垂直场景 + 多模态融合 + 世界模型框架」。你认为这个路径可复制到其他垂直场景（如手术、工业装配）吗？如果可以，下一个值得投入的垂直场景是什么？

🔬 上期预测：「6 周内至少 2 个顶级 VLA 团队将宣布采用 flow_matching 替代 diffusion_policy」（时间窗口：2026-05-20 前）。基于本期 flow_matching accel_7d=1.41 且 diffusion_policy accel_7d=0.39 的数据，你的判断是？✅ 已验证 / ❌ 落空 / ⏳ 待观察——给出你的理由，不允许回答「两方面都有道理」。
