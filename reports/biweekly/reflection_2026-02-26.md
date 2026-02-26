# 🤔 双周反思 | 2026-02-12 – 2026-02-25

> 基于本期双周报告生成。不需要回答，但如果你读完没有立场，说明这两周你在消费而不是研究。

## 趋势与判断

1. 触觉 VLA 在 2026-02-12 至 2026-02-25 期间集中出现 TaCo 基准、TactEx 框架、力控夹爪论文——这是领域成熟信号还是论文灌水周期？用 TaCo 的异构触觉数据类型数量回答。

2. 世界模型相关理论文章占本期 9 篇中的 4 篇（MVISTA-4D、VLA-JEPA、MIND、Olaf-World）——社区是在收敛到统一范式，还是在发散探索不同路径？用"想象 - 执行"在 MVISTA-4D 和 VLA-JEPA 中的实现差异证明你的判断。

3. CALVIN 榜单上 Xiaomi-Robotics-0 以 4.75/4.80 刷新 SOTA，LIBERO 上 ACoT-VLA 与 ABot-M0 交替领先（差距<1%）——VLA benchmark 是在逼近性能天花板，还是已进入微幅刷榜阶段？用 CALVIN ABC-D 分裂上前 5 名模型的分数方差回答。

4. 如果未来 6 个月只能投入一个方向：触觉 VLA（TaCo/TactEx）、效率优化（Habilis-β端侧/VLA-Perf）、还是世界模型融合（MVISTA-4D/VLA-JEPA）？选一个，并用本期至少两个具体信号支撑你的选择。

5. Genesis v0.4.0 迁移到 Quadrants 编译器、MuJoCo 3.5.0 发布——底层工具链迭代对 VLA 研究的实际推动力有多大？对比 Genesis v0.3.14 到 v0.4.0 的 release note，列出直接影响 VLA 训练效率的变更。

## 技术追问

1. TaCo 基准提出触觉数据的无损与有损编解码方案——你能说清两种方案在带宽、延迟、信息保留上的 trade-off 吗？如果不能，从 TaCo 论文 Table 2 开始读，搞懂为什么 GelSight 和 DIGIT 需要不同的编码策略。

2. CausalGDP 将因果推理引入扩散策略——它跟传统 diffusion policy 在动作生成机制上有什么本质区别？不知道的话，对比 CausalGDP 和 diffusion policy 的 sampling 公式，找出因果图介入的位置。

3. MVISTA-4D 和 VLA-JEPA 都引入"想象 - 执行"范式——两者的"想象"模块在 latent space 构建方式上有什么不同？建议对照阅读两篇论文的 Figure 2，画出各自的 latent rollout 流程。

4. Scaling Verification 论文提出"扩展验证比扩展策略学习更有效"——test-time verification 在 VLA 中的具体实现机制是什么？从论文 Method 部分找出 verification 模块如何与 VLA 的 action head 交互。
