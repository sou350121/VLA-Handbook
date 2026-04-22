# 速查表 (Cheat Sheet)

涵盖 VLA 领域核心概念、公式、模型对比、常见陷阱与工程误区的快速参考。

> **更新**：2026-04-21 · 覆盖 2022 CLIP → 2026 Q2 π0.7 / EgoScale / GR00T-N1.7

---

## 📚 基础篇

1. **[关键论文时间线](./timeline.md)** — 从 CLIP 到 π0.7、EgoScale、GR00T-N1.7 · 34 篇关键工作
2. **[核心公式速查](./formulas.md)** — Attention · Diffusion · Flow Matching · LoRA · 6D 旋转 · RL 基础 · 评估指标
3. **[模型对比总表](./model_comparison.md)** — 16+ 主流 VLA 模型架构 × 能力 × 真机成绩

## 🔍 专项篇

4. **[Benchmark 地图 + 可信度警告](./benchmarks.md)** — **30+ benchmark · 9 大类 + 智能/推理专节** + BEHAVIOR-1K 深度拆解
5. **[数据集速查](./datasets.md)** — 25+ 数据集 · 许可证继承规则 · 商用合规组合推荐
6. **[数据失效模式 F1-F6](./failure-modes.md)** — 虚假相关 / OOD / 时序错位 / 坐标漂移 / 语言解耦 / 动作偏斜
7. **[RL 后训练三流派](./rl-post-training.md)** — 传统 RL / ACP / RL Token · World Model 辅助

## 使用建议

- **建立领域全景**：扫一遍 [timeline](./timeline.md)，按"epoch → 代表模型 → 唯一重要贡献"组织记忆
- **复习核心算法**：翻 [formulas](./formulas.md)——Attention / LoRA / Diffusion vs Flow Matching / 6D 旋转 / RL Advantage
- **需要比较多个模型**：看 [model_comparison](./model_comparison.md) 架构 × 能力 × 真机成绩矩阵
- **报 benchmark 分数前**：先过 [benchmarks 警告](./benchmarks.md)——PRO/Plus/X 扰动结果 + 真机验证**一起报**
- **选数据集**：看 [datasets](./datasets.md)——**先看许可证再看规模**，商用合规数据集组合
- **Debug 真机出问题**：用 [failure-modes F1-F6](./failure-modes.md) 分类排查
- **规划后训练路线**：看 [rl-post-training](./rl-post-training.md) 三流派选型

## 🚨 常见工程/学术误区

1. ❌ **把 OpenVLA 当 SOTA** —— OpenVLA 是"第一代开源标杆"，真机性能已被 π0.5/π0.6/π0.7/GR00T-N1.7 等超越
2. ❌ **只报单 benchmark 数字** —— LIBERO 90%+ 很可能是记忆化，需要 PRO/Plus/X 扰动对照（见警告 1-3）
3. ❌ **把 Flow Matching 和 Diffusion 当同义词** —— 两者 MDP 建模完全不同，RL 后训练路线也不同
4. ❌ **认为"人类数据没用"** —— EgoScale log-linear scaling 实证 (R²=0.9983)
5. ❌ **混用 CC BY-NC 数据做商用** —— 许可证继承到模型，即使代码 MIT 也不能商用

---

[← Back to VLA Handbook](../README.md)
