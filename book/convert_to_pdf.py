#!/usr/bin/env python3
"""
将合并的 Markdown 转换为 PDF
使用 weasyprint 进行渲染
"""

import markdown
from pathlib import Path

def convert_md_to_html(md_path: Path, html_path: Path):
    """Markdown -> HTML"""
    md_content = md_path.read_text(encoding='utf-8')
    
    # 转换为 HTML
    html_content = markdown.markdown(
        md_content,
        extensions=['tables', 'fenced_code', 'codehilite', 'toc'],
        extension_configs={
            'codehilite': {'css_class': 'highlight'},
        }
    )
    
    # 完整 HTML 模板
    full_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>VLA 面试手册</title>
    <style>
        @page {{
            size: A4;
            margin: 2cm;
            @top-center {{
                content: "VLA 面试手册";
                font-size: 10pt;
                color: #666;
            }}
            @bottom-center {{
                content: counter(page);
                font-size: 10pt;
            }}
        }}
        body {{
            font-family: "Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", sans-serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #333;
            max-width: 100%;
        }}
        h1 {{
            color: #1a5f7a;
            border-bottom: 2px solid #1a5f7a;
            padding-bottom: 10px;
            page-break-before: always;
        }}
        h1:first-of-type {{
            page-break-before: avoid;
        }}
        h2 {{
            color: #2c3e50;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
        }}
        h3 {{
            color: #34495e;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: "Consolas", "Monaco", monospace;
            font-size: 10pt;
        }}
        pre {{
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            overflow-x: auto;
            font-size: 9pt;
            line-height: 1.4;
        }}
        pre code {{
            background: none;
            padding: 0;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
            font-size: 10pt;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px 12px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        blockquote {{
            border-left: 4px solid #1a5f7a;
            margin: 15px 0;
            padding: 10px 20px;
            background-color: #f9f9f9;
            color: #555;
        }}
        a {{
            color: #1a5f7a;
            text-decoration: none;
        }}
        hr {{
            border: none;
            border-top: 1px solid #ddd;
            margin: 30px 0;
        }}
        .highlight {{
            background-color: #f8f8f8;
        }}
        /* 目录样式 */
        .toc {{
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
        }}
        .toc ul {{
            list-style-type: none;
            padding-left: 20px;
        }}
        .toc a {{
            text-decoration: none;
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>
"""
    
    html_path.write_text(full_html, encoding='utf-8')
    print(f"✅ HTML 生成: {html_path}")
    return html_path


def convert_html_to_pdf(html_path: Path, pdf_path: Path):
    """HTML -> PDF using weasyprint"""
    try:
        from weasyprint import HTML
        HTML(filename=str(html_path)).write_pdf(str(pdf_path))
        print(f"✅ PDF 生成: {pdf_path}")
        print(f"   文件大小: {pdf_path.stat().st_size / 1024 / 1024:.2f} MB")
        return True
    except Exception as e:
        print(f"❌ PDF 生成失败: {e}")
        print("   请尝试其他方法转换 HTML 为 PDF")
        return False


def main():
    book_dir = Path(__file__).resolve().parent
    output_dir = book_dir / "output"
    
    md_path = output_dir / "VLA_Interview_Handbook_Full.md"
    html_path = output_dir / "VLA_Interview_Handbook.html"
    pdf_path = output_dir / "VLA_Interview_Handbook.pdf"
    
    if not md_path.exists():
        print("❌ 请先运行 build_book_simple.py 生成 Markdown")
        return
    
    print("📖 转换 Markdown → HTML...")
    convert_md_to_html(md_path, html_path)
    
    print("\n📄 转换 HTML → PDF...")
    convert_html_to_pdf(html_path, pdf_path)


if __name__ == "__main__":
    main()





