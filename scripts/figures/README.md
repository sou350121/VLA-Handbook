# 社区笔记配图生成脚本

`deployment/assets/fig*.svg` 由这些脚本生成。**不要手改 SVG**，改脚本后重新生成。

```bash
python3 gen1.py   # 图1 文档地图, 图2 故障诊断树
python3 gen2.py   # 图3 模型谱系, 图4 黑话全景, 图5 黑话解码器
python3 fix67.py  # 图6 心态曲线, 图7 黑话四象限
python3 gen4.py   # 图8 演化时间线, 图9 主题饼图
```

为什么用 SVG 而不是 Mermaid：GitHub 的 Mermaid 走客户端 viewscreen iframe 渲染，
在本文件上 9 张图全部显示 `Unable to render rich display`（连基础 flowchart 也失败）。
静态 SVG 不依赖 JS，必然渲染。
