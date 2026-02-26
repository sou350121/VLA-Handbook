# 🤔 双周反思 | 2026-02-13 – 2026-02-26

> 基于本期双周报告生成。不需要回答，但如果你读完没有立场，说明这两周你在消费而不是研究。

## 趋势与判断

1. CALVIN 榜单在 14 天内被刷新两次（pi-RL 4.71 → Xiaomi-Robotics-0 4.75/4.80）。这是真进步还是刷榜？Xiaomi 的论文里有没有公开训练细节让你能复现？

2. 触觉 VLA 突然升温——TaCo 基准、TactEx 框架、力控夹爪论文集中出现。这是领域到了爆发前夜，还是单纯的数据采集硬件终于便宜到实验室能批量买了？

3. 世界模型 +VLA 方向两周冒出 MIND、Olaf-World、Agent World Model、VLA-JEPA 四个新工作。社区是在收敛到"想象 - 执行"范式，还是在各自造轮子？选一个你相信会活过 2026 年的架构。

4. LIBERO 榜单上 ACoT-VLA 和 ABot-M0 交替领先（98.6 vs 99.1），差距不到 1%。这种微幅刷新还有意义吗？还是说 LIBERO 已经饱和，需要新基准了？

5. Genesis 模拟器两周连发两版（0.3.14 → 0.4.0），直接迁移到 Quadrants 编译器。如果让你选 sim-to-real 的底层依赖，你押 Genesis 还是押 MuJoCo 3.5.0？为什么？

## 技术追问

6. TaCo 基准论文里对比了哪些触觉编解码方案？无损 vs 有损在 VLA 策略训练里的实际影响测过吗？没测过的话，这是不是你该补的实验？

7. 本期世界模型工作（MIND、Olaf-World）都提到了"latent action"。这个概念跟 Diffusion Policy 里的 action chunking 有什么区别？说不清楚的话，建议把两篇论文的 method 部分对照着读一遍。

8. Xiaomi-Robotics-0 刷新 CALVIN 记录时用了什么架构？OpenVLA 微调还是从头训练？如果它没公开代码，你能从论文图表里反推出关键设计选择吗？

9. CausalGDP 那篇把 causality 引入 diffusion policy。你知道它具体在 diffusion 的哪个环节注入因果约束吗？是去噪过程、条件输入、还是 reward shaping？

10. "Scaling Verification > Scaling Policy"这篇提出了测试时验证框架。如果让你把这个方法用到你的 VLA pipeline 里，你会在哪个环节加 verification？推理前、推理后、还是训练时？
