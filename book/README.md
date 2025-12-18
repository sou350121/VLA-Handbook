# 📚 VLA Handbook - 电子书版本

将 `theory/` 目录下的所有 Markdown 文件合并成一本完整的电子书，支持 PDF 和 HTML 格式输出。

## 快速开始

### 1. 生成合并的 Markdown

```bash
cd book
python build_book.py
```

输出文件：`book/output/VLA_Handbook.md`

### 2. 生成 PDF（需要安装依赖）

```bash
# 安装依赖 (Ubuntu/Debian)
sudo apt update
sudo apt install -y pandoc texlive-xetex texlive-lang-chinese fonts-noto-cjk

# 生成 PDF
python build_book.py --pdf
```

输出文件：`book/output/VLA_Handbook.pdf`

### 3. 生成 HTML

```bash
python build_book.py --html
```

输出文件：`book/output/VLA_Handbook.html`

## 书籍结构

| 部分 | 章节 | 内容 |
| :--- | :--- | :--- |
| **第一部分** | 1-4 | 基础架构 (Transformer, Flash Attention, 多模态, VLA) |
| **第二部分** | 5-9 | 策略生成与动作表示 (Diffusion Policy, ACT, Flow Matching) |
| **第三部分** | 10-15 | 训练技术与优化 (LoRA, RLHF, 蒸馏, SSL, 量化) |
| **第四部分** | 16-20 | 感知与空间智能 (空间数学, 控制, 感知, SLAM) |
| **第五部分** | 21-23 | 抓取与运动规划 |
| **第六部分** | 24-28 | 前沿模型解析 (RDT, π0, Galaxea, WALL-OSS) |
| **第七部分** | 29-31 | 评估与推理 |
| **附录** | A-C | 数据格式, 文献综述, ASCII 速查 |

## 自定义

### 修改章节顺序

编辑 `build_book.py` 中的 `CHAPTERS` 列表：

```python
CHAPTERS = [
    ("第1章 Transformer vs CNN", "transformer_vs_cnn.md"),
    ("第2章 Flash Attention", "flash_attention.md"),
    # ... 添加或调整顺序
]
```

### 修改 PDF 样式

编辑 `BOOK_HEADER` 中的 YAML front matter：

```yaml
geometry: margin=2.5cm    # 页边距
fontsize: 11pt            # 字体大小
toc-depth: 3              # 目录深度
```

## 其他导出方式

### 使用 mdBook（推荐用于在线阅读）

```bash
# 安装 mdBook
cargo install mdbook

# 初始化并构建
mdbook init
mdbook build
```

### 使用 Typora 导出

1. 打开 `output/VLA_Handbook.md`
2. 文件 → 导出 → PDF

### 使用 VS Code 插件

1. 安装 "Markdown PDF" 插件
2. 右键 → Markdown PDF: Export (pdf)

## 常见问题

### PDF 中文显示问题

确保安装了中文字体：

```bash
# Ubuntu
sudo apt install fonts-noto-cjk

# macOS
brew install font-noto-sans-cjk
```

### pandoc 命令找不到

```bash
# Ubuntu
sudo apt install pandoc

# macOS
brew install pandoc

# Windows
choco install pandoc
```

### 代码块语法高亮

PDF 默认使用 `tango` 主题，可在 `build_book.py` 中修改：

```python
"--highlight-style=tango",  # 可选: pygments, kate, monochrome, etc.
```





