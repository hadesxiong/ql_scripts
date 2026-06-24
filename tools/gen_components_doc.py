#!/usr/bin/env python3
import os
import re
import subprocess

COMPONENT_ROOT = "./components"
OUTPUT_MD = "./docs/components_list.md"
SUFFIX = (".py", ".sh", ".js", ".ts")
desc_reg = re.compile(r"^# @desc\s+(.*)$|^// @desc\s+(.*)$")

def get_desc(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) < 2:
            return "缺少第二行@desc注释"
        line2 = lines[1].strip()
        match = desc_reg.match(line2)
        if match:
            return match.group(1) or match.group(2)
        return "未识别@desc说明"
    except Exception as err:
        return f"读取失败：{str(err)}"

def scan_all_components():
    group = {
        "Python组件": [],
        "Shell组件": [],
        "JS组件": [],
        "TS组件": []
    }
    for root, _, files in os.walk(COMPONENT_ROOT):
        for fname in files:
            if fname.endswith(SUFFIX):
                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, ".")
                desc_text = get_desc(full_path)
                line = f"- [{fname}](./{rel_path})：{desc_text}"
                if full_path.endswith(".py"):
                    group["Python组件"].append(line)
                elif full_path.endswith(".sh"):
                    group["Shell组件"].append(line)
                elif full_path.endswith(".js"):
                    group["JS组件"].append(line)
                elif full_path.endswith(".ts"):
                    group["TS组件"].append(line)
    return group

def build_md(group_data):
    md = [
        "# 公共组件清单（自动生成，禁止手动修改）",
        "> 本文件由 tools/gen_components_doc.py 生成，读取组件文件第二行 `@desc` 作为介绍\n"
    ]
    for title, items in group_data.items():
        if not items:
            continue
        md.append(f"## {title}")
        md.extend(items)
        md.append("")
    return "\n".join(md)

if __name__ == "__main__":
    data = scan_all_components()
    content = build_md(data)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(content)
    subprocess.run(["git", "add", OUTPUT_MD])
    print(f"{OUTPUT_MD} 组件文档刷新完成")