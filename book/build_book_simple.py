#!/usr/bin/env python3
"""
VLA Interview Handbook - Simple Book Builder
使用纯 Python 生成合并的 Markdown（无需 pandoc）

PDF 转换可以使用:
1. 在线工具: https://md2pdf.netlify.app/
2. VS Code 插件: Markdown PDF
3. Typora 导出
4. grip + 浏览器打印
"""

import os
from pathlib import Path
from datetime import datetime

# 章节顺序定义
CHAPTERS = [
    # 第一部分：基础架构
    ("part1", "# 第一部分：基础架构"),
    ("第1章 Transformer vs CNN", "transformer_vs_cnn.md"),
    ("第2章 Flash Attention 与推理优化", "flash_attention.md"),
    ("第3章 多模态模型基础", "multimodal_models.md"),
    ("第4章 VLA 架构总览", "vla_arch.md"),
    
    # 第二部分：策略生成与动作表示
    ("part2", "# 第二部分：策略生成与动作表示"),
    ("第5章 动作表示方法", "action_representations.md"),
    ("第6章 Diffusion Policy", "diffusion_policy.md"),
    ("第7章 Action Chunking Transformer (ACT)", "act.md"),
    ("第8章 Flow Matching", "pi0_flow_matching.md"),
    ("第9章 FAST 动作序列编码", "fast.md"),
    
    # 第三部分：训练技术与优化
    ("part3", "# 第三部分：训练技术与优化"),
    ("第10章 参数高效微调 (PEFT/LoRA)", "peft_lora.md"),
    ("第11章 强化学习基础与 RLHF", "reinforcement_learning.md"),
    ("第12章 知识蒸馏", "knowledge_distillation.md"),
    ("第13章 自监督学习", "self_supervised_learning.md"),
    ("第14章 迁移学习与 Co-training", "transfer_learning.md"),
    ("第14章附 Co-training 详解", "co_training.md"),
    ("第15章 量化技术", "quantization_theory.md"),
    
    # 第四部分：感知与空间智能
    ("part4", "# 第四部分：感知与空间智能"),
    ("第16章 空间数学基础", "spatial_math.md"),
    ("第17章 机器人控制方法", "robot_control.md"),
    ("第18章 感知技术", "perception_techniques.md"),
    ("第19章 点云与 SLAM", "pointcloud_slam.md"),
    ("第20章 状态估计", "state_estimation.md"),
    
    # 第五部分：抓取与运动规划
    ("part5", "# 第五部分：抓取与运动规划"),
    ("第21章 抓取算法", "grasp_algorithms.md"),
    ("第22章 运动规划", "motion_planning.md"),
    ("第23章 触觉 VLA", "tactile_vla.md"),
    
    # 第六部分：前沿模型解析
    ("part6", "# 第六部分：前沿模型解析"),
    ("第24章 RDT (Robotics Diffusion Transformer)", "rdt.md"),
    ("第25章 π0.5 解析", "pi0_5_dissection.md"),
    ("第26章 π0.6 解析", "pi0_6_dissection.md"),
    ("第27章 Galaxea G0", "galaxea_g0.md"),
    ("第28章 WALL-OSS", "wall_oss.md"),
    
    # 第七部分：评估与推理
    ("part7", "# 第七部分：评估与推理"),
    ("第29章 Chain-of-Thought 推理", "chain_of_thought.md"),
    ("第30章 评估方法论", "evaluation.md"),
    ("第31章 知识隔离", "knowledge_insulation.md"),
    
    # 附录
    ("appendix", "# 附录"),
    ("附录A 数据格式与处理", "data.md"),
    ("附录B 文献综述", "literature_review.md"),
    ("附录C ASCII 图表速查", "ascii_cheatsheet.md"),
]


def generate_toc():
    """生成目录"""
    toc = ["# 目录\n"]
    chapter_num = 0
    
    for title, filename in CHAPTERS:
        if title.startswith("part") or title.startswith("appendix"):
            toc.append(f"\n{filename}\n")
        elif title.startswith("附录"):
            toc.append(f"- {title}")
        else:
            chapter_num += 1
            toc.append(f"- {title}")
    
    return "\n".join(toc) + "\n"


def clean_content(content: str) -> str:
    """清理内容"""
    lines = []
    for line in content.split('\n'):
        # 移除返回链接
        if '[← Back to' in line or '[← 返回' in line:
            continue
        lines.append(line)
    return '\n'.join(lines)


def main():
    script_dir = Path(__file__).resolve().parent
    theory_dir = script_dir.parent / "theory"
    output_dir = script_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    # 书籍头部
    book = f"""# VLA 面试手册：从理论到实践

> **Vision-Language-Action 完全指南**
>
> 生成日期: {datetime.now().strftime("%Y年%m月%d日")}
>
> 在线版本: https://github.com/sou350121/vla-interview-handbook

---

{generate_toc()}

---

"""
    
    # 处理每个章节
    for title, filename in CHAPTERS:
        if title.startswith("part") or title.startswith("appendix"):
            book += f"\n---\n\n{filename}\n\n"
            continue
            
        filepath = theory_dir / filename
        if not filepath.exists():
            print(f"⚠️  跳过: {filename}")
            continue
        
        print(f"📖 {title}")
        content = filepath.read_text(encoding='utf-8')
        content = clean_content(content)
        
        # 添加章节分隔
        book += f"\n---\n\n## {title}\n\n"
        
        # 调整标题级别
        for line in content.split('\n'):
            if line.startswith('# '):
                # 跳过原标题，已经用章节标题替换
                continue
            elif line.startswith('## '):
                book += '###' + line[2:] + '\n'
            elif line.startswith('### '):
                book += '####' + line[3:] + '\n'
            else:
                book += line + '\n'
    
    # 写入文件
    output_path = output_dir / "VLA_Interview_Handbook_Full.md"
    output_path.write_text(book, encoding='utf-8')
    
    print(f"\n✅ 生成完成: {output_path}")
    print(f"   文件大小: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"   总字数约: {len(book)} 字符")
    print("\n📄 转换为 PDF 的方法:")
    print("   1. 在线: https://md2pdf.netlify.app/")
    print("   2. VS Code: 安装 'Markdown PDF' 插件，右键导出")
    print("   3. Typora: 文件 → 导出 → PDF")
    print("   4. grip: pip install grip && grip --export output.html")


if __name__ == "__main__":
    main()


