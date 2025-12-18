#!/usr/bin/env python3
"""
VLA Handbook - Book Builder
将 theory/ 目录下的 Markdown 文件合并成一本完整的电子书

使用方法:
    python build_book.py              # 生成合并的 Markdown
    python build_book.py --pdf        # 生成 PDF (需要 pandoc + latex)
    python build_book.py --html       # 生成 HTML
"""

import os
import re
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

# 章节顺序定义
CHAPTERS = [
    # 第一部分：基础架构
    ("第1章 Transformer vs CNN", "transformer_vs_cnn.md"),
    ("第2章 Flash Attention 与推理优化", "flash_attention.md"),
    ("第3章 多模态模型基础", "multimodal_models.md"),
    ("第4章 VLA 架构总览", "vla_arch.md"),
    
    # 第二部分：策略生成与动作表示
    ("第5章 动作表示方法", "action_representations.md"),
    ("第6章 Diffusion Policy", "diffusion_policy.md"),
    ("第7章 Action Chunking Transformer (ACT)", "act.md"),
    ("第8章 Flow Matching", "pi0_flow_matching.md"),
    ("第9章 FAST 动作序列编码", "fast.md"),
    
    # 第三部分：训练技术与优化
    ("第10章 参数高效微调 (PEFT/LoRA)", "peft_lora.md"),
    ("第11章 强化学习基础与 RLHF", "reinforcement_learning.md"),
    ("第12章 知识蒸馏", "knowledge_distillation.md"),
    ("第13章 自监督学习", "self_supervised_learning.md"),
    ("第14章 迁移学习与 Co-training", "transfer_learning.md"),
    ("第14章附 Co-training", "co_training.md"),
    ("第15章 量化技术", "quantization_theory.md"),
    
    # 第四部分：感知与空间智能
    ("第16章 空间数学基础", "spatial_math.md"),
    ("第17章 机器人控制方法", "robot_control.md"),
    ("第18章 感知技术", "perception_techniques.md"),
    ("第19章 点云与 SLAM", "pointcloud_slam.md"),
    ("第20章 状态估计", "state_estimation.md"),
    ("第21章 具身导航 (VLN) / DualVLN 快慢系统", "vln_dualvln.md"),
    
    # 第五部分：抓取与运动规划
    ("第22章 抓取算法", "grasp_algorithms.md"),
    ("第23章 运动规划", "motion_planning.md"),
    ("第24章 触觉 VLA", "tactile_vla.md"),
    
    # 第六部分：前沿模型解析
    ("第25章 RDT (Robotics Diffusion Transformer)", "rdt.md"),
    ("第26章 π0.5 解析", "pi0_5_dissection.md"),
    ("第27章 π0.6 解析", "pi0_6_dissection.md"),
    ("第28章 Galaxea G0", "galaxea_g0.md"),
    ("第29章 WALL-OSS", "wall_oss.md"),
    
    # 第七部分：评估与推理
    ("第30章 Chain-of-Thought 推理", "chain_of_thought.md"),
    ("第31章 评估方法论", "evaluation.md"),
    ("第32章 知识隔离", "knowledge_insulation.md"),
    
    # 附录
    ("附录A 数据格式与处理", "data.md"),
    ("附录B 文献综述", "literature_review.md"),
    ("附录C ASCII 图表速查", "ascii_cheatsheet.md"),
]

BOOK_HEADER = """---
title: "VLA Handbook：从理论到实践"
subtitle: "Vision-Language-Action 完全指南"
author: "VLA Handbook Contributors"
date: "{date}"
documentclass: report
geometry: margin=2.5cm
fontsize: 11pt
toc: true
toc-depth: 3
numbersections: true
colorlinks: true
linkcolor: blue
urlcolor: blue
header-includes:
  - \\usepackage{{ctex}}
  - \\usepackage{{fancyhdr}}
  - \\pagestyle{{fancy}}
  - \\fancyhead[L]{{VLA Handbook}}
  - \\fancyhead[R]{{\\thepage}}
  - \\fancyfoot[C]{{}}
---

\\newpage

# 前言

本书是 **VLA Handbook** 项目的完整理论部分，系统性地介绍了视觉-语言-动作 (Vision-Language-Action) 模型的核心概念、关键技术与工程实践。

**适用读者**：
- 准备机器人/具身智能方向面试的工程师
- 希望系统学习 VLA 技术栈的研究者
- 对多模态机器人感兴趣的学生

**如何使用本书**：
1. **系统学习**：按章节顺序阅读，建立完整知识体系
2. **面试准备**：重点关注每章末尾的 Q&A 部分
3. **查阅参考**：使用目录快速定位特定主题

**在线版本**：https://github.com/sou350121/VLA-Handbook

\\newpage

"""


def clean_markdown(content: str, chapter_title: str) -> str:
    """清理和调整 Markdown 内容"""
    lines = content.split('\n')
    cleaned_lines = []
    skip_header = True
    
    for line in lines:
        # 跳过原文件的第一个标题（会用章节标题替换）
        if skip_header and line.startswith('# '):
            skip_header = False
            continue
        
        # 移除返回链接
        if '[← Back to' in line or '[← 返回' in line:
            continue
            
        # 调整标题级别（## -> ###，### -> ####）
        if line.startswith('## '):
            line = '##' + line[2:]  # 保持 ## 不变，作为章节内的主要标题
        elif line.startswith('# '):
            line = '##' + line[1:]  # # 变成 ##
            
        cleaned_lines.append(line)
    
    # 添加章节标题
    result = f"\n\\newpage\n\n# {chapter_title}\n\n"
    result += '\n'.join(cleaned_lines)
    
    return result


def build_combined_markdown(theory_dir: Path, output_path: Path):
    """合并所有章节为单个 Markdown 文件"""
    
    content = BOOK_HEADER.format(date=datetime.now().strftime("%Y年%m月%d日"))
    
    current_part = ""
    part_titles = {
        "第1章": "# 第一部分：基础架构\n\n",
        "第5章": "\n\\newpage\n\n# 第二部分：策略生成与动作表示\n\n",
        "第10章": "\n\\newpage\n\n# 第三部分：训练技术与优化\n\n",
        "第16章": "\n\\newpage\n\n# 第四部分：感知与空间智能\n\n",
        "第21章": "\n\\newpage\n\n# 第五部分：抓取与运动规划\n\n",
        "第24章": "\n\\newpage\n\n# 第六部分：前沿模型解析\n\n",
        "第29章": "\n\\newpage\n\n# 第七部分：评估与推理\n\n",
        "附录A": "\n\\newpage\n\n# 附录\n\n",
    }
    
    for chapter_title, filename in CHAPTERS:
        # 添加部分标题
        for part_key, part_title in part_titles.items():
            if chapter_title.startswith(part_key) and current_part != part_key:
                content += part_title
                current_part = part_key
                break
        
        filepath = theory_dir / filename
        if not filepath.exists():
            print(f"[WARN] 跳过不存在的文件: {filename}")
            continue
            
        print(f"[CHAPTER] 处理: {chapter_title}")
        chapter_content = filepath.read_text(encoding='utf-8')
        content += clean_markdown(chapter_content, chapter_title)
    
    # 写入合并文件
    output_path.write_text(content, encoding='utf-8')
    print(f"\n[OK] 合并完成: {output_path}")
    print(f"   文件大小: {output_path.stat().st_size / 1024:.1f} KB")
    
    return output_path


def build_pdf(markdown_path: Path, output_path: Path):
    """使用 pandoc 生成 PDF"""
    print("\n[INFO] 生成 PDF...")
    
    cmd = [
        "pandoc",
        str(markdown_path),
        "-o", str(output_path),
        "--pdf-engine=xelatex",
        "-V", "mainfont=Noto Sans CJK SC",
        "-V", "monofont=Noto Sans Mono CJK SC",
        "--highlight-style=tango",
        "--toc",
        "--toc-depth=3",
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"[OK] PDF 生成完成: {output_path}")
        print(f"   文件大小: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    except subprocess.CalledProcessError as e:
        print(f"[ERR] PDF 生成失败: {e}")
        print("   请确保已安装 pandoc 和 texlive-xetex")
        print("   Ubuntu: sudo apt install pandoc texlive-xetex texlive-lang-chinese fonts-noto-cjk")
    except FileNotFoundError:
        print("[ERR] 未找到 pandoc，请先安装")
        print("   Ubuntu: sudo apt install pandoc")


def build_html(markdown_path: Path, output_path: Path):
    """使用 pandoc 生成 HTML"""
    print("\n🌐 生成 HTML...")
    
    cmd = [
        "pandoc",
        str(markdown_path),
        "-o", str(output_path),
        "--standalone",
        "--toc",
        "--toc-depth=3",
        "-c", "https://cdn.jsdelivr.net/npm/github-markdown-css/github-markdown.min.css",
        "--metadata", "title=VLA Handbook",
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"[OK] HTML 生成完成: {output_path}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[ERR] HTML 生成失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="Build VLA Handbook Book")
    parser.add_argument("--pdf", action="store_true", help="Generate PDF")
    parser.add_argument("--html", action="store_true", help="Generate HTML")
    args = parser.parse_args()
    
    # 路径设置
    script_dir = Path(__file__).resolve().parent
    theory_dir = script_dir.parent / "theory"
    output_dir = script_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    print(f"[INFO] Theory 目录: {theory_dir}")
    print(f"[INFO] 输出目录: {output_dir}")
    
    # 生成合并的 Markdown
    md_path = output_dir / "VLA_Handbook.md"
    build_combined_markdown(theory_dir, md_path)
    
    # 生成 PDF
    if args.pdf:
        pdf_path = output_dir / "VLA_Handbook.pdf"
        build_pdf(md_path, pdf_path)
    
    # 生成 HTML
    if args.html:
        html_path = output_dir / "VLA_Handbook.html"
        build_html(md_path, html_path)
    
    print("\n[OK] 构建完成!")
    print(f"   输出目录: {output_dir}")


if __name__ == "__main__":
    main()

