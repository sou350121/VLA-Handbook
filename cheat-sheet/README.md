# 速查表 (Cheat Sheet)

面试前的"急救包"，涵盖 VLA 领域核心概念、公式、模型对比、常见陷阱与红线。

> **更新**：2026-04-21 · 覆盖 2022 CLIP → 2026 Q2 π0.7 / EgoScale / GR00T-N1.7

---

## 📚 基础篇

1. **[关键论文时间线](./timeline.md)** — 从 CLIP 到 π0.7、EgoScale、GR00T-N1.7 · 34 篇关键工作
2. **[核心公式速查](./formulas.md)** — Attention · Diffusion · Flow Matching · LoRA · 6D 旋转 · RL 基础 · 评估指标
3. **[模型对比总表](./model_comparison.md)** — 16+ 主流 VLA 模型架构 × 能力 × 真机成绩

## 🔍 专项篇

4. **[Benchmark 地图 + 可信度警告](./benchmarks.md)** — 10+ 仿真 + 3+ 真机 benchmark · 6 条 LIBERO-PRO / LIBERO-Para 警告
5. **[数据失效模式 F1-F6](./failure-modes.md)** — 虚假相关 / OOD / 时序错位 / 坐标漂移 / 语言解耦 / 动作偏斜
6. **[RL 后训练三流派](./rl-post-training.md)** — 传统 RL / ACP / RL Token · World Model 辅助

## 使用建议

- **面试前 1 小时**：扫一遍 [timeline](./timeline.md)，记住每个 epoch 的代表模型和"唯一的一件事"
- **技术面准备**：复习 [formulas](./formulas.md)——能手写 Attention + 解释 LoRA + 区分 Diffusion vs Flow Matching
- **遇到"为什么选 X 模型"**：翻 [model_comparison](./model_comparison.md) 决策矩阵
- **被问"LIBERO 你们做到多少"**：先讲 [benchmarks 警告](./benchmarks.md)——别只报数字，要讲 PRO/Plus/X 扰动
- **被问"你 debug 真机出问题怎么找原因"**：用 [failure-modes F1-F6](./failure-modes.md) 分类
- **被问"为什么用/不用 RL"**：看 [rl-post-training](./rl-post-training.md)

## 🚨 面试红线（绝不要犯）

1. ❌ "OpenVLA 是 SOTA"——SOTA 早被 π0.5/π0.6/π0.7/GR00T-N1.7 超越，OpenVLA 是"第一代开源标杆"
2. ❌ "LIBERO 96% 就够"——只报单 benchmark 数字等于不懂记忆化风险（见警告 1-3）
3. ❌ "Flow Matching 和 Diffusion 一样"——两者 MDP 建模完全不同，RL 微调路线不同
4. ❌ "人类数据没用"——EgoScale log-linear scaling 已经实证（R²=0.9983）
5. ❌ "CC BY-NC 数据我们商用"——许可证继承，模型也不能商用

---

[← Back to VLA Handbook](../README.md)
