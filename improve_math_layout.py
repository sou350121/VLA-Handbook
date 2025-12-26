import os
import re

def improve_math(content):
    # 1. 规范化块公式 $$ ... $$
    # 匹配所有 $$ ... $$ 块，无论是在一行还是多行
    # 使用非贪婪匹配，确保处理多个公式
    
    # 步骤 A: 提取公式内容并规范化格式
    def normalize_block(match):
        inner = match.group(1).strip()
        # 移除内部多余的空行，GitHub 渲染器有时会因为内部空行而崩溃
        inner = re.sub(r'\n\s*\n', r'\n', inner)
        # 确保公式前后有空行，且 $$ 独占一行
        return f"\n\n$$\n{inner}\n$$\n\n"

    # 先处理可能紧凑在一起的公式
    content = re.sub(r'\$\$(.*?)\$\$', normalize_block, content, flags=re.DOTALL)
    
    # 2. 规范化行内公式 $...$
    # 确保中文和 $ 之间有空格
    content = re.sub(r'([\u4e00-\u9fa5])\$', r'\1 $', content)
    content = re.sub(r'\$([\u4e00-\u9fa5])', r'$ \1', content)
    
    # 3. 修复列表项中的公式缩进问题
    # 如果 $$ 在列表项后面，GitHub 需要它另起一行且上方有空行
    # (上面的 normalize_block 已经通过 \n\n 强制解决了大部分问题)
    
    # 4. 清理连续的空行，保持页面整洁
    content = re.sub(r'\n{3,}', r'\n\n', content)
    
    # 5. 修正特定符号的渲染问题
    # 有些 LaTeX 符号在 Markdown 环境中需要转义，但在 $$ 中通常不需要
    # 主要是确保没有奇怪的 Markdown 语法干扰公式内部
    
    return content

# 目标目录
base_dir = '.'
dirs_to_process = ['theory', 'deployment']

for d in dirs_to_process:
    target_dir = os.path.join(base_dir, d)
    if not os.path.exists(target_dir):
        continue
        
    print(f"Deep processing directory: {target_dir}")
    for root, dirs, files in os.walk(target_dir):
        for filename in files:
            if filename.endswith('.md'):
                filepath = os.path.join(root, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                new_content = improve_math(content)
                
                if new_content != content:
                    print(f"Applying deep math fix to {filepath}...")
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(new_content)

print("Deep optimization complete!")
