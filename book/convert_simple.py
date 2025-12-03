#!/usr/bin/env python3
"""
简单的 Markdown 到 HTML 转换器（无依赖）
然后可以用浏览器打印为 PDF
"""

import re
from pathlib import Path


def simple_md_to_html(md_content: str) -> str:
    """简单的 Markdown 转 HTML"""
    html = md_content
    
    # 代码块 (先处理，避免被其他规则影响)
    html = re.sub(
        r'```(\w*)\n(.*?)```',
        lambda m: f'<pre><code class="{m.group(1)}">{escape_html(m.group(2))}</code></pre>',
        html,
        flags=re.DOTALL
    )
    
    # 行内代码
    html = re.sub(r'`([^`]+)`', r'<code>\1</code>', html)
    
    # 标题
    html = re.sub(r'^###### (.+)$', r'<h6>\1</h6>', html, flags=re.MULTILINE)
    html = re.sub(r'^##### (.+)$', r'<h5>\1</h5>', html, flags=re.MULTILINE)
    html = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    
    # 粗体和斜体
    html = re.sub(r'\*\*\*(.+?)\*\*\*', r'<strong><em>\1</em></strong>', html)
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
    
    # 链接
    html = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', html)
    
    # 图片
    html = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', r'<img src="\2" alt="\1">', html)
    
    # 引用块
    html = re.sub(r'^> (.+)$', r'<blockquote>\1</blockquote>', html, flags=re.MULTILINE)
    
    # 水平线
    html = re.sub(r'^---+$', r'<hr>', html, flags=re.MULTILINE)
    
    # 表格 (简单处理)
    html = convert_tables(html)
    
    # 无序列表
    html = re.sub(r'^- (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    html = re.sub(r'(<li>.*</li>\n?)+', r'<ul>\g<0></ul>', html)
    
    # 有序列表
    html = re.sub(r'^\d+\. (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    
    # 段落
    html = re.sub(r'\n\n+', '</p>\n\n<p>', html)
    html = f'<p>{html}</p>'
    
    # 清理
    html = re.sub(r'<p>\s*</p>', '', html)
    html = re.sub(r'<p>\s*(<h[1-6]>)', r'\1', html)
    html = re.sub(r'(</h[1-6]>)\s*</p>', r'\1', html)
    html = re.sub(r'<p>\s*(<pre>)', r'\1', html)
    html = re.sub(r'(</pre>)\s*</p>', r'\1', html)
    html = re.sub(r'<p>\s*(<ul>)', r'\1', html)
    html = re.sub(r'(</ul>)\s*</p>', r'\1', html)
    html = re.sub(r'<p>\s*(<table>)', r'\1', html)
    html = re.sub(r'(</table>)\s*</p>', r'\1', html)
    html = re.sub(r'<p>\s*(<blockquote>)', r'\1', html)
    html = re.sub(r'(</blockquote>)\s*</p>', r'\1', html)
    html = re.sub(r'<p>\s*(<hr>)', r'\1', html)
    
    return html


def escape_html(text: str) -> str:
    """转义 HTML 特殊字符"""
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def convert_tables(html: str) -> str:
    """转换 Markdown 表格"""
    lines = html.split('\n')
    result = []
    in_table = False
    
    for i, line in enumerate(lines):
        if '|' in line and not line.strip().startswith('```'):
            cells = [c.strip() for c in line.split('|')[1:-1]]
            
            if not in_table:
                # 检查下一行是否是分隔符
                if i + 1 < len(lines) and re.match(r'^[\s|:-]+$', lines[i + 1]):
                    result.append('<table>')
                    result.append('<thead><tr>')
                    for cell in cells:
                        result.append(f'<th>{cell}</th>')
                    result.append('</tr></thead>')
                    result.append('<tbody>')
                    in_table = True
                    continue
            
            if in_table:
                if re.match(r'^[\s|:-]+$', line):
                    continue
                result.append('<tr>')
                for cell in cells:
                    result.append(f'<td>{cell}</td>')
                result.append('</tr>')
                continue
        else:
            if in_table:
                result.append('</tbody></table>')
                in_table = False
        
        result.append(line)
    
    if in_table:
        result.append('</tbody></table>')
    
    return '\n'.join(result)


def main():
    book_dir = Path(__file__).resolve().parent
    output_dir = book_dir / "output"
    
    md_path = output_dir / "VLA_Interview_Handbook_Full.md"
    html_path = output_dir / "VLA_Interview_Handbook.html"
    
    if not md_path.exists():
        print("❌ 请先运行 build_book_simple.py")
        return
    
    print("📖 读取 Markdown...")
    md_content = md_path.read_text(encoding='utf-8')
    
    print("🔄 转换为 HTML...")
    html_body = simple_md_to_html(md_content)
    
    # 完整 HTML
    full_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>VLA 面试手册：从理论到实践</title>
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", sans-serif;
            font-size: 14px;
            line-height: 1.7;
            color: #24292e;
            max-width: 900px;
            margin: 0 auto;
            padding: 40px 20px;
            background: #fff;
        }}
        h1 {{
            color: #1a5f7a;
            border-bottom: 3px solid #1a5f7a;
            padding-bottom: 12px;
            margin-top: 50px;
            font-size: 28px;
        }}
        h2 {{
            color: #2c3e50;
            border-bottom: 2px solid #eaecef;
            padding-bottom: 8px;
            margin-top: 40px;
            font-size: 22px;
        }}
        h3 {{
            color: #34495e;
            margin-top: 30px;
            font-size: 18px;
        }}
        h4 {{
            color: #444;
            margin-top: 25px;
            font-size: 16px;
        }}
        code {{
            background-color: rgba(27, 31, 35, 0.05);
            padding: 3px 6px;
            border-radius: 4px;
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
            font-size: 13px;
        }}
        pre {{
            background-color: #f6f8fa;
            border: 1px solid #e1e4e8;
            border-radius: 6px;
            padding: 16px;
            overflow-x: auto;
            font-size: 12px;
            line-height: 1.5;
        }}
        pre code {{
            background: none;
            padding: 0;
            font-size: 12px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            font-size: 13px;
        }}
        th, td {{
            border: 1px solid #dfe2e5;
            padding: 10px 14px;
            text-align: left;
        }}
        th {{
            background-color: #f6f8fa;
            font-weight: 600;
        }}
        tr:nth-child(even) {{
            background-color: #fafbfc;
        }}
        blockquote {{
            border-left: 4px solid #1a5f7a;
            margin: 20px 0;
            padding: 12px 20px;
            background-color: #f6f8fa;
            color: #555;
        }}
        a {{
            color: #0366d6;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        hr {{
            border: none;
            border-top: 2px solid #eaecef;
            margin: 40px 0;
        }}
        ul, ol {{
            padding-left: 25px;
        }}
        li {{
            margin: 6px 0;
        }}
        img {{
            max-width: 100%;
            height: auto;
        }}
        /* 打印样式 */
        @media print {{
            body {{
                font-size: 11pt;
                max-width: 100%;
                padding: 0;
            }}
            h1 {{
                page-break-before: always;
                font-size: 20pt;
            }}
            h1:first-of-type {{
                page-break-before: avoid;
            }}
            pre {{
                white-space: pre-wrap;
                word-wrap: break-word;
            }}
            a {{
                color: #000;
            }}
        }}
    </style>
</head>
<body>
{html_body}
<hr>
<p style="text-align: center; color: #666; font-size: 12px;">
    VLA 面试手册 | <a href="https://github.com/sou350121/vla-interview-handbook">GitHub</a>
</p>
</body>
</html>
"""
    
    html_path.write_text(full_html, encoding='utf-8')
    print(f"✅ HTML 生成完成: {html_path}")
    print(f"   文件大小: {html_path.stat().st_size / 1024:.1f} KB")
    print("\n📄 转换为 PDF:")
    print("   1. 用浏览器打开 HTML 文件")
    print("   2. Ctrl+P (或 Cmd+P) 打印")
    print("   3. 选择 '另存为 PDF'")
    print(f"\n   文件路径: file://{html_path}")


if __name__ == "__main__":
    main()

