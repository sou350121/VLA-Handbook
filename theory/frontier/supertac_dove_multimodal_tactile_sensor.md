# SuperTac：多模态“电子皮肤” + 触觉语言模型 DOVE（Nature Sensors 2025）

> **定位**：一篇典型的“硬件×模型”协同：用更高信息密度的触觉模态（多光谱 + 摩擦电 + IMU + 温度/接近/振动）去支撑更强的触觉语义理解（Tactile Language Model）。  
> **主来源**：Nature Sensors 论文页：`https://www.nature.com/articles/s44460-025-00006-y`  
> **补充来源**（公开新闻稿，含摘要式指标）：清华 SIGS 新闻：`https://www.sigs.tsinghua.edu.cn/2026/0119/c7688a288292/page.htm`

---

## 已合并：用一篇“可落地”的版本做主文档

这篇笔记与 `deployment/perception/supertac_dove_biomimetic_multimodal_tactile_sensing.md` 的内容高度重叠（同一论文/同一系统），为避免两处各写一遍，本条目已**收敛为跳转页**：

- **主文档（推荐阅读）**：[`../../deployment/perception/supertac_dove_biomimetic_multimodal_tactile_sensing.md`](../../deployment/perception/supertac_dove_biomimetic_multimodal_tactile_sensing.md)  
  - 覆盖：硬件结构要点、关键 numbers、多模态同步/串扰/散热等工程坑、以及为什么它应当进入 perception 体系。

如果你只想记住一句话：**SuperTac 的价值不在“更细”，而在“把接触相位的多个隐变量（滑移/碰撞/材质/温度/接近）变成可观测证据，并让这些证据能被大模型组织成可推理的语义”。**

---

## 参考

- Nature Sensors 论文页：`https://www.nature.com/articles/s44460-025-00006-y`
- 清华 SIGS 新闻（摘要式指标）：`https://www.sigs.tsinghua.edu.cn/2026/0119/c7688a288292/page.htm`

