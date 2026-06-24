#!/usr/bin/env python3
"""运行所有文档生成工具"""
import subprocess, sys

tools = [
    "tools/gen_update_log.py",
    "tools/gen_components_doc.py",
    "tools/gen_readme.py",
]
codes = [subprocess.call(["python3", t]) for t in tools]
sys.exit(max(codes))  # 任一失败即返回非零