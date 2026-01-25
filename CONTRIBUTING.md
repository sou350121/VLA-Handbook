# 贡献指南 (Contributing Guide)

感谢你对 VLA Handbook 的关注！我们需要社区的力量来保持这份文档的实效性和深度。

## 先读这些（强烈建议）

- 项目主页与入口：[`README.md`](./README.md)
- AI/自动化维护规范与写作标准：[`AGENT.md`](./AGENT.md)

## 如何贡献 (How to Contribute)

我们非常欢迎以下类型的贡献：

1.  **补充最新的 VLA 论文解读**: 领域发展很快，如果你读到了新的好论文 (如 Pi, Wall-X 等)，欢迎提交 PR 补充到 `theory/` 目录下。
2.  **分享真机部署经验**: 理论与现实总有差距。如果你有在 Jetson 或其他边缘设备上的部署经验，请分享到 `deployment/`。
3.  **提供面试真题**: 如果你在面试中遇到了新的题目，欢迎补充到 `question-bank/`。
4.  **纠正错误**: 发现公式写错或链接失效？请直接提交 PR 修复。

## 文档类贡献的“可执行流程”（建议照做）

### 1) 新增或改名文档时（必须同步索引）

- **文件命名**：使用 `snake_case.md`，避免空格与大写。
- **引用来源**：关键结论附近给出来源链接（论文优先 arXiv/DOI/官网；代码/模型优先 GitHub/HuggingFace）。
- **同步索引**（至少做一项，通常都要做）：
  - `theory/`：更新 [`theory/README.md`](./theory/README.md)，必要时更新 `theory/paper_index.md` 与 `theory/literature_review.md`
  - `deployment/`：更新 [`deployment/README.md`](./deployment/README.md)
  - `question-bank/`：更新 [`question-bank/README.md`](./question-bank/README.md)
  - `cheat-sheet/`：更新 [`cheat-sheet/README.md`](./cheat-sheet/README.md)
  - 新增“核心入口”时：更新根 [`README.md`](./README.md)
- **自检**：
  - 内部链接可点击且指向存在的文件
  - 文内术语与缩写一致（首次出现给全称/英文）

### 2) theory/ 文档最小质量门槛（推荐对齐 `AGENT.md`）

为了保证可面试复述、可工程落地、可快速定位，建议至少包含：
- 开头元信息块：发布时间/版本、定位、一句话 takeaway
- 至少 1 个架构/信息流图（ASCII 也可以）
- 至少 1 张“组件/系统对比”表格
- 数学核心：目标 → 公式 → 变量解释 → 直觉
- 工程视角：延迟/步数/抖动/部署约束等 trade-off
- 文末返回索引链接（例如 `[← Back to Theory](./README.md)`）+ 关键引用

## 避免事项（请不要这样做）

- **不要编造论文、实验结果或指标**；不确定就用 `TODO/待证/待补 citation` 标注。
- **不要批量重写现有文档**：优先小范围、可 review 的增量更新。
- **不要改动生成内容**（如电子书输出目录），除非你明确知道影响范围。

## 提交 Pull Request 的流程

1.  Fork 本仓库。
2.  创建一个新的分支: `git checkout -b my-new-feature`。
3.  提交你的更改: `git commit -am 'Add some feature'`。
4.  推送到分支: `git push origin my-new-feature`。
5.  提交 Pull Request。

## 格式规范

- 请使用 Markdown 格式。
- 引用论文时请附上 ArXiv 链接。
- 尽量保持目录结构的整洁。

再次感谢你的贡献！
