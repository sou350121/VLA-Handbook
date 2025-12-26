import os
import re

def improve_math(content):
    # 1. 处理块公式 $$ ... $$
    # 先把所有形式的 $$ 统一拉出来，前后各留一空行，$$ 符号各占一行
    # 匹配可能跨行的 $$ ... $$，允许中间有空行
    content = re.sub(r'\n*\s*\$\$(.*?)\$\$\s*\n*', r'\n\n$$\n\1\n$$\n\n', content, flags=re.DOTALL)
    
    # 清理公式内部多余的空行（如果有的话）
    def clean_block(match):
        inner = match.group(1).strip()
        # 移除内部连续的空行，防止渲染失败
        inner = re.sub(r'\n\s*\n', r'\n', inner)
        return f"\n\n$$\n{inner}\n$$\n\n"
    
    content = re.sub(r'\n\n\$\$\n(.*?)\n\$\$\n\n', clean_block, content, flags=re.DOTALL)

    # 2. 处理行内公式 $...$
    # 确保中文和 $ 之间有空格
    # 中文 + $ -> 中文 + 空格 + $
    content = re.sub(r'([\u4e00-\u9fa5])\$', r'\1 $', content)
    # $ + 中文 -> $ + 空格 + 中文
    content = re.sub(r'\$([\u4e00-\u9fa5])', r'$ \1', content)
    
    # 3. 连续空行压缩
    content = re.sub(r'\n{3,}', r'\n\n', content)
    
    return content

# 目标目录（在项目根目录下运行）
base_dir = '.'
dirs_to_process = ['theory', 'deployment']

for d in dirs_to_process:
    target_dir = os.path.join(base_dir, d)
    if not os.path.exists(target_dir):
        print(f"Directory {target_dir} not found, skipping...")
        continue
        
    print(f"Processing directory: {target_dir}")
    for root, dirs, files in os.walk(target_dir):
        for filename in files:
            if filename.endswith('.md'):
                filepath = os.path.join(root, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                new_content = improve_math(content)
                
                if new_content != content:
                    print(f"Improving math in {filepath}...")
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(new_content)

print("Optimization complete!")
