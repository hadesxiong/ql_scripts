# 每日工作日报 daily_report

## 基础信息
- 脚本入口：`py/daily_report/main.py`
- 开发语言：Python3
- 依赖公共组件：无
- 青龙推荐定时：0 9 * * *

## 功能简介
1. 查询 Plane 数据库中 7 日内待完成任务；
2. 逐条推送到 Hookshot 通知。

## 运行产出
无文件产出

## 环境变量配置
复制 config.tpl.env 内变量填入青龙面板环境变量；

## 本地调试运行
```bash
cd py/daily_report
python3 main.py
```

## 更新记录
2026-06-24：脚本初始化