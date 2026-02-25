# 🤔 双周反思 | 2026-02-12 – 2026-02-25

> 基于本期双周报告生成。不需要回答，但如果你读完没有立场，说明这两周你在消费而不是研究。

## 趋势与判断

1. **触觉 VLA 是否在从"感知融合"转向"数据标准化"？** 本期 TaCo 基准明确提出无损/有损编解码问题，TactEx 和力控夹爪论文集中出现。这是赛道成熟的信号，还是新一轮灌水前的基建竞赛？

2. **世界模型 + VLA 的"想象 - 执行"范式（MVISTA-4D、VLA-JEPA、Olaf-World）是否在收敛？** 两周内 4 篇相关工作，但 MIND 基准专门测"记忆一致性"——这是否暗示当前世界模型在长视野规划上仍有根本缺陷？

3. **如果只能选一个 sim 平台投入（Genesis v0.4.0 vs MuJoCo 3.5.0 vs ManiSkill v3），你选哪个？** Genesis 完成 Quadrants 编译器迁移，MuJoCo 3.5.0 更新日志语焉不详。端侧部署和 sim-to-real 转移率，哪个优先级更高？

4. **CALVIN 榜单被 Xiaomi-Robotics-0 刷新到 4.75/4.8，但 ABC-D 和 ABCD-D 差距仅 0.05——这是否意味着泛化性瓶颈已现？** 当跨场景迁移提升不到 5%，是该继续堆数据还是换架构？

5. **RoboGene 用"多样性驱动的任务生成"做 VLA 预训练，World Action Models 声称 zero-shot 策略——这两条路线哪条更接近"通用"的本质？** 一个是数据端增强，一个是模型端泛化，资源有限时你押哪边？

## 技术追问

6. **本期 3 篇世界模型论文（MVISTA-4D、Olaf-World、Agent World Model）都提到"latent action"，但你能说清它跟传统 action token 的本质区别吗？** 如果不能，建议从 Olaf-World 的 method  section 开始读，搞清楚 latent space 里到底在建模什么。

7. **TaCo 基准测触觉编解码，但你知道主流触觉传感器（GelSight、DIGIT、Tactile-Transformer）的原始数据格式和带宽需求吗？** 如果不知道，先去读 TaCo 的 dataset  section——连数据长什么样都没概念，谈什么"无损压缩"？

8. **CausalGDP 把 causality 引入 diffusion policy，但你知道 diffusion policy 在 VLA 里的标准训练流程吗？** 建议先复现一个基础版（比如 Diffusion Policy 原论文），再回来想 causality 该加在 denoising 的哪一步。
