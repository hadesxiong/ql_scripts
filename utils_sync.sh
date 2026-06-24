#!/bin/bash
# 青龙 /ql/scripts 一键同步私有仓库
cd /ql/scripts
git fetch origin
git reset --hard origin/main
echo "====================================="
echo "脚本同步完成，最新更新记录："
cat UPDATE_LOG.md
echo "====================================="