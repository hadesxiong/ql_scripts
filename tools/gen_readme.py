#!/usr/bin/env python3
import os
import re
import subprocess

scan_ext = (".py", ".sh", ".js", ".ts")
ignore_dirs = (".github", "tools", "components", "output", "docs")
lang_folder_map = {
    "py": "Python脚本",
    "shell": "Shell脚本",
    "js": "JS脚本",
    "ts": "TS脚本"
}

def extract_script_info(readme_path):
    """从单脚本README提取名称、简介"""
    try:
        with open(readme_path, "r", encoding="utf-8") as f:
            text = f.read()
        title_match = re.search(r"# (.+)", text)
        desc_match = re.search(r"## 功能简介\s*\n(.+)", text, re.MULTILINE)
        name = title_match.group(1).strip() if title_match else "未知脚本"
        desc = desc_match.group(1).strip() if desc_match else "无简介"
        return name, desc
    except:
        return "读取失败", ""

def collect_all_scripts():
    result = {}
    for lang_dir, lang_name in lang_folder_map.items():
        result[lang_name] = []
        if not os.path.exists(lang_dir):
            continue
        for folder in os.listdir(lang_dir):
            fd_path = os.path.join(lang_dir, folder)
            if not os.path.isdir(fd_path):
                continue
            readme_fp = os.path.join(fd_path, "README.md")
            rel_readme = os.path.relpath(readme_fp, ".")
            script_name, desc = extract_script_info(readme_fp)
            line = f"- [{script_name}](./{rel_readme})：{desc}"
            result[lang_name].append(line)
    return result

def replace_readme_block():
    start_tag = "<!-- AUTO_SCRIPT_LIST_START -->"
    end_tag = "<!-- AUTO_SCRIPT_LIST_END -->"
    with open("README.md", "r", encoding="utf-8") as f:
        full_text = f.read()
    before, rest = full_text.split(start_tag)
    _, after = rest.split(end_tag)

    script_groups = collect_all_scripts()
    block_lines = ["## 全部业务脚本清单（自动生成）\n"]
    for group_title, items in script_groups.items():
        if not items:
            continue
        block_lines.append(f"### {group_title}")
        block_lines.extend(items)
        block_lines.append("")
    new_block_content = "\n".join(block_lines)
    new_full = f"{before}{start_tag}\n{new_block_content}\n{end_tag}{after}"
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(new_full)

if __name__ == "__main__":
    replace_readme_block()
    subprocess.run(["git", "add", "README.md"])
    print("根README脚本清单自动刷新完成")