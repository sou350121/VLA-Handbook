# AGENT.md

本文件面向在 VLA-Handbook 仓库中工作的自动化/AI agent，说明写作与维护规范。

## 目标
- 维护结构化、可检索的 VLA 知识库，避免信息碎片化。
- 保持目录索引与内容同步，方便读者快速定位。
- 坚持可验证、可追溯的技术信息。

## 仓库快速导航
- theory/: 理论与核心算法（主要入口）
- deployment/: 真机与部署
- question-bank/: 面试题库与实战
- cheat-sheet/: 速查表
- book/: 电子书（输出多为生成内容）
- product/: 机器人产品
- system-design/: 系统设计
- companies/: 公司与求职

## 内容与格式规范
- 语言：中文为主，保留英文术语与官方命名；模型/论文名保持原文拼写。
- 文件名：使用 `snake_case.md`，避免空格与大写。
- 结构建议（理论类）：概述 → Main Mathematical Idea → 方法/要点 → 优缺点 → 应用/工程 → 参考。
- 结构建议（部署类）：环境/硬件 → 步骤 → 配置/参数 → 常见坑 → 参考。
- 公式与代码：用 Markdown 代码块或 LaTeX；示例可运行且简洁。
- 引用：必须给出来源链接（论文优先 arXiv/DOI/官网；代码/模型优先 GitHub/HuggingFace）。

## 索引维护（必须同步）
- theory/ 新增或改名：更新 `theory/README.md`（必要时 `theory/README_FUN.md`）。
- 新论文/综述：更新 `theory/paper_index.md` 与 `theory/literature_review.md`。
- deployment/ 新增：更新 `deployment/README.md`。
- question-bank/ 新增：更新 `question-bank/README.md`。
- cheat-sheet/ 新增：更新 `cheat-sheet/README.md`。
- 需要新增入口时：更新根 `README.md`。

## 资产与图片
- 新增图片放 `assets/`，并使用相对路径引用。
- 避免提交大体积二进制文件，除非有明确需求。

## 避免事项
- 不要编造论文、实验结果或指标；不确定时标注 TODO/待证。
- 不要批量重写现有文档，优先小范围增量更新。
- 不要改动 `book/output/` 等生成内容，除非明确要求。

## 变更自检
- 索引可找到新增/修改文档
- 内部链接有效
- 术语与命名一致
- 有可靠来源链接