import os
import re

def improve_math(content):
    # 1. Ensure block math $$ is on its own line and has blank lines around it
    content = re.sub(r'\n?\s*\$\$\s*\n?(.*?)\n?\s*\$\$\s*\n?', r'\n\n$$\n\1\n$$\n\n', content, flags=re.DOTALL)
    
    # 2. Fix inline math - ensure spaces around $ if adjacent to CJK or other non-space chars
    content = re.sub(r'([\u4e00-\u9fa5])\$', r'\1 $', content)
    content = re.sub(r'\$([\u4e00-\u9fa5])', r'$ \1', content)
    
    # 3. Ensure blank lines around $$
    content = re.sub(r'\n\s*\$\$\s*\n', r'\n\n$$\n', content)
    
    # 4. Final cleanup of multiple blank lines
    content = re.sub(r'\n{3,}', r'\n\n', content)
    
    return content

# Determine directories to process
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
                else:
                    # print(f"No changes for {filepath}")
                    pass

