from pathlib import Path

def export_markdown(root_path=".", output_file="structure.md"):
    root = Path(root_path).resolve()
    
    def tree(path, level=0):
        try:
            contents = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        except PermissionError:
            return
        
        for item in contents:
            if item.name.startswith('.') or item.name in ['__pycache__', 'node_modules', '.git']:
                continue
            
            indent = "  " * level
            if item.is_dir():
                yield f"{indent}- 📁 **{item.name}/**"
                yield from tree(item, level + 1)
            else:
                yield f"{indent}- 📄 {item.name}"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# 项目结构\n\n")
        f.write(f"**根目录:** `{root}`\n\n")
        f.write("```\n")
        for line in tree(root):
            f.write(line + "\n")
        f.write("```\n")

export_markdown(".")