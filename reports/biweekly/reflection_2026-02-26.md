# 🤔 双周反思 | 2026-02-12 – 2026-02-25

> 基于本期双周报告生成。不需要回答，但如果你读完没有立场，说明这两周你在消费而不是研究。

## 趋势与判断

1. CALVIN benchmark 在两周内被刷新两次（pi-RL 4.71 → Xiaomi-Robotics-0 4.80），但 LIBERO 已逼近饱和（SimpleVLA-RL 99.1%）。你认为 CALVIN 的头部竞争还能持续多久，还是说社区应该转向更难的 RoboChallenge（当前仅 72.25）？

2. 本期 9 篇 theory 文章中，5 篇聚焦 world model（Agent World Model、MIND、Olaf-World、World Action Models、MVISTA-4D），2 篇聚焦触觉（TaCo、TactEx）。如果下季度你只能深入一个方向，选哪个？证据是什么？

3. Genesis 在 8 天内连发两个版本（v0.3.14 → v0.4.0），完成 Taichi 迁移。MuJoCo 3.5.0 同期发布但声量明显更低。这是模拟器赛道的收敛信号，还是开源社区对"易用性"的投票？

4. 触觉 VLA 升温是真实机会还是灌水赛道？TaCo 基准刚提出编解码标准，但本期 SOTA tracker 中没有任何触觉相关 benchmark 上榜。如果触觉真是精细操作的主流，为什么 leaderboard 还没体现？

5. "世界模型+VLA 在 sim-to-real 超越纯 VLA"是本期预测。但 World Action Models 论文声称 zero-shot policy，而 MIND 在 benchmark 记忆一致性。这两个方向在打架还是互补？你站哪边？

## 技术追问

6. TaCo 基准提出了异构触觉数据的无损/有损编解码评估。你能说清 GelSight、DIGIT、Tactile 3D 这三种传感器的原始数据格式差异吗？如果不能，建议从 TaCo 论文 Appendix 开始读。

7. Xiaomi-Robotics-0 用 4.80 avg_len 刷新 CALVIN ABC-D。你知道它跟 pi-RL（4.71）在架构上的核心差异吗？是数据规模、训练策略、还是推理时的 test-time compute？去读它的 method section。

8. CausalGDP 提出 causality-guided diffusion policy。你知道 standard diffusion policy 在因果推断上的盲点是什么吗？如果不知道，这是你这两周最该补的课——从 CausalGDP 的 Related Work 开始。

9. TwinVLA 用"孪生单臂"实现双臂操作。这种设计跟传统 bimanual VLA 的数据效率优势在哪里？是数据复用还是架构归纳偏置？读它的 ablation study。

10. RoboGene 用 agentic framework 生成真实世界任务来 boost VLA 预训练。你知道它跟传统 data augmentation 的本质区别吗？提示：看它的 task generation pipeline 是否引入环境交互。
