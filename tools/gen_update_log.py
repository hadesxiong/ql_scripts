#!/usr/bin/env python3
import subprocess
import re
from datetime import datetime

def run_cmd(cmd):
    return subprocess.check_output(cmd, shell=True, encoding="utf-8").strip()

# 获取本次暂存变更文件
changed_files = run_cmd("git diff --cached --name-only").splitlines()
script_ext = (".py", ".sh", ".js", ".ts")
# 过滤仅业务脚本，排除组件、工具、CI
target_files = []
for f in changed_files:
    if f.endswith(script_ext) and not f.startswith(("components/", "tools/", ".github/")):
        target_files.append(f)

if not target_files:
    print("无业务脚本变更，跳过更新日志生成")
    exit(0)

short_hash = run_cmd("git rev-parse --short HEAD")
today = datetime.now().strftime("%Y-%m-%d")
log_lines = [f"\n## {today} | Commit:{short_hash}"]

for path in target_files:
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        name = re.search(r"# @name\s+(.*)", content)
        desc = re.search(r"# @desc\s+(.*)", content)
        name = name.group(1) if name else "未命名脚本"
        desc = desc.group(1) if desc else "无简介"
        log_lines.append(f"- `{path}`：{name} | {desc}")
    except Exception as e:
        log_lines.append(f"- `{path}`：读取异常 {str(e)}")

# 追加写入更新日志
with open("UPDATE_LOG.md", "a", encoding="utf-8") as f:
    f.write("\n".join(log_lines))

subprocess.run(["git", "add", "UPDATE_LOG.md"])
print("UPDATE_LOG.md 自动更新完成，已加入暂存")