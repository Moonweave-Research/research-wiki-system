import os
import yaml
import re

def slugify(value: str) -> str:
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(r'[-\s]+', '_', value)

def generate_wiki_page(parsed_data: dict, base_dir: str = "wiki") -> str:
    domain = parsed_data.get("domain", "general")
    title = parsed_data.get("title", "Untitled")
    
    base_filename = slugify(title)
    if not base_filename:
        base_filename = "untitled"
        
    target_dir = os.path.join(base_dir, domain)
    os.makedirs(target_dir, exist_ok=True)
    
    filename = f"{base_filename}.md"
    filepath = os.path.join(target_dir, filename)
    counter = 1
    while os.path.exists(filepath):
        filename = f"{base_filename}_{counter}.md"
        filepath = os.path.join(target_dir, filename)
        counter += 1
    
    frontmatter = {
        "domain": domain,
        "category": parsed_data.get("category", "misc"),
        "tags": parsed_data.get("tags") or [],
        "sources": parsed_data.get("sources") or []
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("---\n")
        yaml.dump(frontmatter, f, default_flow_style=False, sort_keys=False)
        f.write("---\n\n")
        f.write(f"# {title}\n\n")
        f.write(parsed_data.get("content", ""))
        
    return filepath
