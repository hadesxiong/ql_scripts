#!/usr/bin/env python3
import os

def create_gitkeep(root_dir):
    for dirpath, _, _ in os.walk(root_dir):
        keep_file = os.path.join(dirpath, ".gitkeep")
        if not os.path.exists(keep_file):
            open(keep_file, "w", encoding="utf-8").close()

if __name__ == "__main__":
    create_gitkeep("output")
    print("output 目录全部生成 .gitkeep 占位文件")